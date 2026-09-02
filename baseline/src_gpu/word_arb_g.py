"""词级仲裁：把 MOSS/FireRed 两个候选按差异块拆开，枚举组合后用声学裁判选最优。

## 与段级仲裁（ac_test.py）的差别

段级只有 2 个选择（整段用 MOSS 或整段用 FireRed）；词级先把一段对齐成
`eq` / `diff` 块，再对最长的 MAX_BLOCKS 个差异块枚举 2^k 种混合。

关键点：**判别负担没有放大**。n-best 是「换一整段候选」，候选越多越难选（2026-08-08
实测 oracle 扩 1.093 但捕获退 0.186，失败）；词级是「同一段内逐块二选一」，
每个决策都是局部的 —— 这解释了为何 n-best 失败而词级成功。

裁判沿用已验证的 FireRedASR2-AED teacher forcing 声学似然
（见 wiki/insights/arbitration-signal-must-be-acoustic.md）：仲裁信号必须来自音频，
文本派生信号全部无效。

## dev 验证结果（2026-08-08，logs/2026-08-08.md §七）

| MOSS 源 | tcpWER |
|---|---|
| 原始 MOSS | 18.3884% |
| 段级仲裁（v024 配方）| 17.6171% |
| **词级仲裁（本脚本）** | **17.4063%** |

集成后（v020 选择规则 + CAM）：全dev 15.920% / train82 16.175% / holdout24 15.047%，
**holdout 改善（−0.250）大于 train（−0.146）**，与过拟合形状相反。

配方 = `MAX_BLOCKS=4, MARGIN=0.0`。MARGIN=0.05 更保守（全dev 15.987%），略差。

用法：
    cd baseline
    # dev（复现 17.4063%）
    MARGIN=0.0 diarizen_venv/bin/python src/word_arb.py \\
        output/v007_moss output/fr_retext ../data/extracted/dev/dev/wav output/wa_m0
    # test
    MARGIN=0.0 diarizen_venv/bin/python src/word_arb.py \\
        output_test_moss output_test_fr3 ../data/extracted/test/test/wav output_test_wa0

已存在的输出文件会被跳过（可断点续跑）。
"""
from __future__ import annotations

import glob
import itertools
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

# FireRedASR2 的源码与权重不在本仓库（体积大 + 非本项目代码），路径可用环境变量覆盖。
# noqa S108：这两个是**只读**的模型源码/权重目录，不是本进程创建的临时文件，
# 不存在可预测临时文件名被抢占的风险。沿用 ac_test.py 的既有落盘位置。
FIRERED_SRC = os.environ.get("FIRERED_SRC", "/tmp/FireRedASR2S")  # noqa: S108
FIRERED_CKPT = os.environ.get("FIRERED_CKPT", "/tmp/FireRedASR2-AED")  # noqa: S108

sys.path.insert(0, FIRERED_SRC)
from fireredasr2s.fireredasr2.asr import (  # type: ignore[import-not-found]  # noqa: E402
    FireRedAsr2,
    FireRedAsr2Config,
)

# 差异块数上限：枚举量是 2^MAX_BLOCKS，4 → 16 种混合。dev 上 6 与 4 换段数相同
# （均 370 段）但打分次数多 16%，无收益，故取 4。
MAX_BLOCKS = int(os.environ.get("MAX_BLOCKS", "4"))
# 领先纯 MOSS 多少才换。0.0 = 只要更优就换（dev 最优）。改前请重跑双切分验证。
MARGIN = float(os.environ.get("MARGIN", "0.0"))
# 短于此长度的段不仲裁（声学证据不足，且编码器对极短片段不稳）
MIN_SEG_SEC = float(os.environ.get("MIN_SEG_SEC", "0.2"))

_m = FireRedAsr2.from_pretrained(
    "aed", FIRERED_CKPT, FireRedAsr2Config(use_gpu=os.environ.get("WA_GPU","0")=="1", return_timestamp=False)
)
model, tokz, feat = _m.model, _m.tokenizer, _m.feat_extractor
dec = model.decoder


def blocks(a: list[str], b: list[str]) -> list[list]:
    """对齐 a,b 后返回 [[kind, a_seg, b_seg], ...]，kind in {'eq','diff'}。"""
    n, mm = len(a), len(b)
    D = [[0] * (mm + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(mm + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, mm + 1):
            D[i][j] = min(
                D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1])
            )
    ops = []
    i, j = n, mm
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("eq" if a[i - 1] == b[j - 1] else "x", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("x", a[i - 1], None))
            i -= 1
        else:
            ops.append(("x", None, b[j - 1]))
            j -= 1
    ops = ops[::-1]
    out: list[list] = []
    cur: list | None = None
    for k, av, bv in ops:
        if k == "eq":
            if cur and cur[0] == "eq":
                cur[1].append(av)
                cur[2].append(av)
            else:
                if cur:
                    out.append(cur)
                cur = ["eq", [av], [av]]
        else:
            if cur and cur[0] == "diff":
                if av is not None:
                    cur[1].append(av)
                if bv is not None:
                    cur[2].append(bv)
            else:
                if cur:
                    out.append(cur)
                cur = ["diff", [av] if av else [], [bv] if bv else []]
    if cur:
        out.append(cur)
    return out


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


def arbitrate_session(
    moss_recs: list[dict], fr_recs: list[dict], wav_path: str
) -> tuple[list[dict], int, int]:
    """逐段词级仲裁，返回 (结果, 换段数, 打分次数)。"""
    x, sr = sf.read(wav_path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    res: list[dict] = []
    nsw = ncalls = 0
    for a, b in zip(moss_recs, fr_recs):
        ta, tb = a["words"].split(), b["words"].split()
        if ta == tb or a["end_time"] - a["start_time"] < MIN_SEG_SEC:
            res.append(dict(a))
            continue
        bl = blocks(ta, tb)
        di = [t for t, blk in enumerate(bl) if blk[0] == "diff"]
        if not di:
            res.append(dict(a))
            continue
        # 只枚举最长的几个差异块，其余留 MOSS —— 控制 2^k 爆炸
        di = sorted(di, key=lambda t: -max(len(bl[t][1]), len(bl[t][2])))[:MAX_BLOCKS]
        clip = x[int(a["start_time"] * sr) : int(a["end_time"] * sr)]
        enc, mask = make_enc(clip, sr)
        keep_moss = tuple([0] * len(di))  # 全取 MOSS 的那一种组合
        best: tuple[float, list[str], tuple[int, ...]] = (float("-inf"), ta, keep_moss)
        for combo in itertools.product([0, 1], repeat=len(di)):
            toks: list[str] = []
            for t, blk in enumerate(bl):
                if blk[0] == "eq":
                    toks += blk[1]
                elif t in di:
                    toks += blk[2] if combo[di.index(t)] else blk[1]
                else:
                    toks += blk[1]
            ncalls += 1
            v = score(enc, mask, "".join(toks))
            if best is None or v > best[0]:
                best = (v, toks, combo)
        base = score(enc, mask, "".join(ta))
        ncalls += 1
        if best[2] != tuple([0] * len(di)) and best[0] - base >= MARGIN:
            nsw += 1
            res.append({**a, "words": " ".join(best[1])})
        else:
            res.append(dict(a))
    return res, nsw, ncalls


def main() -> int:
    if len(sys.argv) < 5:
        logger.error("用法见模块 docstring：%s", __doc__)
        return 2
    moss_dir, fr_dir, wav_dir, out_dir = sys.argv[1:5]
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
    os.makedirs(out_dir, exist_ok=True)

    sids = sorted(
        os.path.basename(p).split(".")[0] for p in glob.glob(f"{moss_dir}/[0-9]*.seglst.json")
    )
    if limit:
        sids = sids[:limit]
    if not sids:
        logger.error("MOSS 预测目录为空：%s", moss_dir)
        return 1

    t0 = time.time()
    total_sw = total_calls = n_fallback = 0
    for k, sid in enumerate(sids, 1):
        out_path = f"{out_dir}/{sid}.seglst.json"
        if os.path.exists(out_path):
            continue
        with open(f"{moss_dir}/{sid}.seglst.json", encoding="utf-8") as fh:
            moss_recs = json.load(fh)
        fr_path = f"{fr_dir}/{sid}.seglst.json"
        fr_recs = None
        if os.path.exists(fr_path):
            with open(fr_path, encoding="utf-8") as fh:
                fr_recs = json.load(fh)
        # 段结构必须一致才能逐段配对；否则保守回退 MOSS，绝不丢 session
        if fr_recs is None or len(fr_recs) != len(moss_recs):
            n_fallback += 1
            res = [dict(r) for r in moss_recs]
            nsw = ncalls = 0
        else:
            res, nsw, ncalls = arbitrate_session(moss_recs, fr_recs, f"{wav_dir}/{sid}.wav")
        total_sw += nsw
        total_calls += ncalls
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        logger.info(
            "[%d/%d] %s: 累计换 %d 段，打分 %d 次 (%.0fs)",
            k, len(sids), sid, total_sw, total_calls, time.time() - t0,
        )
    logger.info(
        "WORD ARB DONE 换 %d 段，打分 %d 次，%d 段因源结构不一致回退 MOSS",
        total_sw, total_calls, n_fallback,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
