"""语气词声学回填：逐段试插一个语气词，只在**音频支持**时才插。

## 为什么是这条轴

全 dev 实测（v028_dev vs ref），语气词是**分散型**缺口：

| 词 | ref | hyp | 缺口 | 值 |
|---|---|---|---|---|
| **嗯** | 288 | 172 | **+116** | **0.596 点** |
| 噢 | 53 | 33 | +20 | 0.103 点 |
| 唉 | 36 | 19 | +17 | 0.087 点 |
| 其余（吧/嘛/啊/呀/呃）| — | — | +34 | 0.176 点 |
| **合计** | 798 | 614 | **+187** | **0.962 点** |

`嗯` 一个词占缺口的 62%，故本脚本**只补 `嗯`**（单变量）。

## 与已封死的「段首语气词补全」的区别

2026-08-09 试过**盲插**（每段首直接补 `嗯`）：15.643 → 18.42（+2.78），死因是
「段首内容起首段与 ref 语气词对齐率仅 6-12%，补 `嗯` 纯噪声」—— **没有信号决定该不该补**。

本脚本的信号来自音频，遵循本项目已验证三次的原则
（见 wiki/insights/arbitration-signal-must-be-acoustic.md）：
用 FireRedASR2-AED 的 teacher forcing 似然 `P(text | audio)` 比较
`原文` / `嗯+原文` / `原文+嗯`，**只有加了 `嗯` 之后似然反而更高**才采纳。

直觉：音频里真有那声 `嗯` 时，把它写进文本会拿到很高的 token 似然，拉高长度归一化均值；
音频里没有时，硬插的 `嗯` 似然很低，把均值拉下去。**判别方向天然正确。**

## 形状为什么重要

历史上唯一干净过重尾关卡的是 v028（字符规范化）：**多段小改 + 反向损失小**。
语气词回填是同一形状（每段最多加 1 个词，触发分散在很多 session），
而归属轴那类「少数段大好」的机制已 12 个配置全数 FAIL。

用法：
    cd baseline   # 需 FireRed 权重，实际在 GPU 机器上跑
    MARGIN=0.0 python src/filler_backfill.py output/v026_pick <wav_dir> output/filler_m0

已存在的输出文件会被跳过（可断点续跑）。
"""
from __future__ import annotations

import glob
import json
import logging
import os
import sys
import tempfile
import time

import soundfile as sf  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 与 word_arb.py 同一份权重/源码（只读），路径可用环境变量覆盖。
# noqa S108：这是**只读**的模型源码/权重目录，不是本进程创建的临时文件。
FIRERED_SRC = os.environ.get("FIRERED_SRC", "/tmp/FireRedASR2S")  # noqa: S108
FIRERED_CKPT = os.environ.get("FIRERED_CKPT", "/tmp/FireRedASR2-AED")  # noqa: S108

sys.path.insert(0, FIRERED_SRC)
from fireredasr2s.fireredasr2.asr import (  # type: ignore[import-not-found]  # noqa: E402
    FireRedAsr2,
    FireRedAsr2Config,
)

FILLER = os.environ.get("FILLER", "嗯")
# 加了语气词后似然要高出多少才采纳。0.0 = 只要更优就补。
MARGIN = float(os.environ.get("MARGIN", "0.0"))
# 短于此长度的段不动（声学证据不足，且编码器对极短片段不稳）
MIN_SEG_SEC = 0.4
# 段内已有该语气词就不再补（避免叠加）
SKIP_IF_PRESENT = os.environ.get("SKIP_IF_PRESENT", "1") == "1"

_m = FireRedAsr2.from_pretrained(
    "aed", FIRERED_CKPT, FireRedAsr2Config(use_gpu=True, use_half=True, return_timestamp=False)
)
model, tokz, feat = _m.model, _m.tokenizer, _m.feat_extractor
dec = model.decoder


@torch.no_grad()
def make_enc(clip, sr):
    """把一段音频编码成 encoder 输出，供同段内多个候选文本共享。"""
    with tempfile.NamedTemporaryFile(suffix=".wav") as fh:
        sf.write(fh.name, clip, sr)
        fe, ln, _, _, _ = feat([fh.name], ["c"])
    enc, _, mask = model.encoder(fe, ln)
    return enc, mask


@torch.no_grad()
def score(enc, mask, text: str) -> float:
    """teacher forcing 声学似然 P(text | audio)，长度归一化。"""
    if not text:
        return -99.0
    _, ids = tokz.tokenize(text)
    if not ids:
        return -99.0
    ys = torch.tensor([[dec.sos_id] + list(ids)]).long()
    tm = dec.ignored_target_position_is_0(ys, dec.pad_id)
    d = dec.dropout(dec.tgt_word_emb(ys) * dec.scale + dec.positional_encoding(ys))
    for layer in dec.layer_stack:
        d = layer.forward(d, enc, tm, mask, cache=None)
    d = dec.layer_norm_out(d)
    lp = torch.log_softmax(dec.tgt_word_prj(d)[:, :-1], -1)
    return float(lp.gather(-1, ys[:, 1:].unsqueeze(-1)).squeeze(-1).mean())


def backfill_session(recs: list[dict], wav_path: str) -> tuple[list[dict], int, int]:
    """返回 (结果, 补入段数, 打分次数)。"""
    x, sr = sf.read(wav_path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    out: list[dict] = []
    n_add = ncalls = 0
    for r in recs:
        toks = r["words"].split()
        too_short = r["end_time"] - r["start_time"] < MIN_SEG_SEC
        if not toks or too_short or (SKIP_IF_PRESENT and FILLER in toks):
            out.append(dict(r))
            continue
        clip = x[int(r["start_time"] * sr) : int(r["end_time"] * sr)]
        if len(clip) < int(MIN_SEG_SEC * sr):
            out.append(dict(r))
            continue
        enc, mask = make_enc(clip, sr)
        base = score(enc, mask, "".join(toks))
        ncalls += 1
        best_val, best_toks = base + MARGIN, None
        for cand in ([FILLER, *toks], [*toks, FILLER]):
            v = score(enc, mask, "".join(cand))
            ncalls += 1
            if v > best_val:
                best_val, best_toks = v, cand
        if best_toks is None:
            out.append(dict(r))
        else:
            n_add += 1
            out.append({**r, "words": " ".join(best_toks)})
    return out, n_add, ncalls


def main() -> int:
    if len(sys.argv) < 4:
        logger.error("用法见模块 docstring：%s", __doc__)
        return 2
    pred_dir, wav_dir, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    sids = sorted(
        os.path.basename(p).split(".")[0]
        for p in glob.glob(f"{pred_dir}/[0-9]*.seglst.json")
    )
    if not sids:
        logger.error("预测目录为空：%s", pred_dir)
        return 1
    logger.info("配方 FILLER=%s MARGIN=%.3f SKIP_IF_PRESENT=%s", FILLER, MARGIN, SKIP_IF_PRESENT)

    t0 = time.time()
    total_add = total_calls = 0
    for k, sid in enumerate(sids, 1):
        out_path = f"{out_dir}/{sid}.seglst.json"
        if os.path.exists(out_path):
            continue
        with open(f"{pred_dir}/{sid}.seglst.json", encoding="utf-8") as fh:
            recs = sorted(json.load(fh), key=lambda r: r["start_time"])
        res, n_add, ncalls = backfill_session(recs, f"{wav_dir}/{sid}.wav")
        total_add += n_add
        total_calls += ncalls
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        logger.info("[%d/%d] %s: 累计补 %d 段，打分 %d 次 (%.0fs)",
                    k, len(sids), sid, total_add, total_calls, time.time() - t0)
    logger.info("FILLER BACKFILL DONE 补入 %d 个「%s」，打分 %d 次 → %s",
                total_add, FILLER, total_calls, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
