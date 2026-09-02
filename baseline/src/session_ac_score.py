"""按 session 汇总声学似然：给「整份预测」打分，用于跨源选源的方向检验。

## 与已有打分脚本的区别

| 脚本 | 比什么 | 分割 |
|---|---|---|
| `ac_score.py` / `ac_test.py` | 同一段音频上的两条**文本** | **同分割** |
| `word_arb.py` | 同一段内的**词块**混合 | **同分割** |
| **本脚本** | 两个源的**整份预测** | **跨分割** |

前两者的公平性来自「同一段音频、两条候选文本、同一个打分器」。**本脚本要跨分割比较**，
公平性不再自明 —— 这正是要检验的东西，也是 2026-08-09 机会点调研里列出的冲突点 3。

## 打分

对该源的每一段：裁音频 → FireRedASR2-AED 编码器 → teacher forcing 算 `log P(text | clip)`，
**累加 token log-prob（不做段内平均）**，最后按 session 的总 token 数归一化：

    score = sum_segments sum_tokens logP / sum_segments n_tokens

按**总 token 归一化**而非段均值，是为了不让「切得碎的源」因段数多而占便宜或吃亏。

⚠️ 该分数**看不见说话人标签**（teacher forcing 只条件于音频与文本）。
它能反映「文本是否与音频相符」与「段边界切得对不对」，**不能直接反映归属对错**。
这是已知的信号盲区，判读结果时必须计入。

## 判据（方向门，不扫阈值）

按「哪个源的实际错误更少」分组，比较分差方向。方向对才继续；方向错或无区分 → 当场停。
2026-08-09 已有两个选源信号栽在此处（轮廓系数「方向对但决策废」、塌缩检测「方向直接反」），
故**除方向外必须同时报「按该信号决策后的错误数」**才算通过。

用法：
    cd baseline
    diarizen_venv/bin/python src/session_ac_score.py output/wa_m0_dev \\
        ../data/extracted/dev/dev/wav /tmp/acsel_moss.json

已算过的 session 会跳过（可断点续跑）。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile

import soundfile as sf  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# FireRedASR2 源码与权重不在本仓库（体积大 + 非本项目代码），路径可用环境变量覆盖。
# noqa S108：只读的模型目录，不是本进程创建的临时文件，无抢占风险。
FIRERED_SRC = os.environ.get("FIRERED_SRC", "/tmp/FireRedASR2S")  # noqa: S108
FIRERED_CKPT = os.environ.get("FIRERED_CKPT", "/tmp/FireRedASR2-AED")  # noqa: S108
MIN_SEG_SEC = 0.2   # 短于此的段不打分（声学证据不足）

sys.path.insert(0, FIRERED_SRC)
from fireredasr2s.fireredasr2.asr import (  # type: ignore[import-not-found]  # noqa: E402
    FireRedAsr2,
    FireRedAsr2Config,
)

_m = FireRedAsr2.from_pretrained(
    "aed", FIRERED_CKPT, FireRedAsr2Config(use_gpu=False, return_timestamp=False)
)
model, tokz, feat = _m.model, _m.tokenizer, _m.feat_extractor
dec = model.decoder


@torch.no_grad()
def seg_logprob(clip, sr: int, text: str) -> tuple[float, int]:
    """返回 (该段所有 token 的 log-prob 之和, token 数)。"""
    if not text:
        return 0.0, 0
    _, ids = tokz.tokenize(text)
    if not ids:
        return 0.0, 0
    with tempfile.NamedTemporaryFile(suffix=".wav") as fh:
        sf.write(fh.name, clip, sr)
        fe, ln, _, _, _ = feat([fh.name], ["c"])
    enc, _, mask = model.encoder(fe, ln)
    ys = torch.tensor([[dec.sos_id] + list(ids)]).long()
    tm = dec.ignored_target_position_is_0(ys, dec.pad_id)
    d = dec.dropout(dec.tgt_word_emb(ys) * dec.scale + dec.positional_encoding(ys))
    for layer in dec.layer_stack:
        d = layer.forward(d, enc, tm, mask, cache=None)
    d = dec.layer_norm_out(d)
    lp = torch.log_softmax(dec.tgt_word_prj(d)[:, :-1], -1)
    tok = lp.gather(-1, ys[:, 1:].unsqueeze(-1)).squeeze(-1)
    return float(tok.sum()), int(tok.numel())


def score_session(wav_path: str, recs: list[dict]) -> dict:
    x, sr = sf.read(wav_path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    total_lp = 0.0
    total_tok = 0
    used = 0
    for r in recs:
        if r["end_time"] - r["start_time"] < MIN_SEG_SEC:
            continue
        clip = x[max(0, int(r["start_time"] * sr)) : min(len(x), int(r["end_time"] * sr))]
        if len(clip) < int(MIN_SEG_SEC * sr):
            continue
        lp, n = seg_logprob(clip, sr, "".join(r["words"].split()))
        total_lp += lp
        total_tok += n
        used += 1
    return {
        "logprob_sum": total_lp,
        "n_tokens": total_tok,
        "n_segs": used,
        # 按总 token 归一化：切得碎的源不因段数多而占便宜
        "score": total_lp / total_tok if total_tok else -99.0,
    }


def main() -> int:
    if len(sys.argv) < 4:
        logger.error("用法见模块 docstring：%s", __doc__)
        return 2
    pred_dir, wav_dir, out_path = sys.argv[1:4]
    sids = sorted(
        f.split(".")[0]
        for f in os.listdir(pred_dir)
        if f.endswith(".seglst.json") and f[0].isdigit()
    )
    if not sids:
        logger.error("预测目录为空：%s", pred_dir)
        return 1

    res: dict[str, dict] = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            res = json.load(fh)
        logger.info("续跑：已有 %d session", len(res))

    for k, sid in enumerate(sids, 1):
        if sid in res:
            continue
        with open(f"{pred_dir}/{sid}.seglst.json", encoding="utf-8") as fh:
            recs = sorted(json.load(fh), key=lambda r: r["start_time"])
        res[sid] = score_session(f"{wav_dir}/{sid}.wav", recs)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False)
        logger.info("[%d/%d] %s: score %.4f（%d 段 / %d token）",
                    k, len(sids), sid, res[sid]["score"], res[sid]["n_segs"],
                    res[sid]["n_tokens"])
    logger.info("SESSION AC SCORE DONE %d session → %s", len(res), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
