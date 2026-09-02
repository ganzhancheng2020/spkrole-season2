"""归属轴结合式方向检验：细粒度窗口 + 说话人档案 → 逐段「挑战标签」。

## 与已证伪做法的区别（三条约束，均由 2026-08-08 实测推出，见 logs/2026-08-08.md §十）

| 做法 | 结果 | 死因 |
|---|---|---|
| 替换（外部模型重做分割+标注）| 6 次全塌 | 必须端到端打赢 MOSS |
| 盲重聚类（扔掉 MOSS 标签重聚）| **29.804%**（vs v026 15.920%）| 26.2% 的段跨说话人 → 段级嵌入是混合体；且丢掉了 MOSS 标签里的信息 |
| **本方案** | 待验 | —— |

本方案的三条约束：

1. **不产出分割** —— 段边界与文本完全沿用现有预测，一个字不动
2. **不整体覆盖标签** —— MOSS 标签是默认答案，只在证据超过 margin 时才改（留待下一步）
3. **嵌入粒度细于段** —— 段内切 `WIN_SEC` 小窗逐个取嵌入，避开混合体问题

这与文本轴 v024/v026 连续两次兑现的形状同构：**保留在位者，挑战者赢过 margin 才换**。

## 信号构造

1. 每段内以 `WIN_SEC` / `WIN_SHIFT` 切窗，逐窗取嵌入（判官 = ERes2NetV2，与 CAM++ 独立）
2. 每个 hyp 说话人建档案 = 其名下所有窗嵌入的均值（**留一法**：算某段时排除该段自己的窗，
   否则在位者分数被自己抬高，比较不公平）
3. 逐段：`incumbent` = 该段各窗与在位档案的平均余弦相似度；
   `challenger` = 其它说话人档案里的最大值；`delta = challenger - incumbent`

## 判据（方向门，不扫阈值）

真值：该段是否**应当**改标签 —— 若该段的真实归属不等于「它当前 hyp 标签所对应的
多数真实归属」，则应改。

- **应改**的段，`delta` 应显著为正
- **不应改**的段，`delta` 应显著为负

方向对 → 才值得做真实改写器并扫 margin。方向错或无区分 → **当场停**。
⚠️ 2026-08-08 已有两个信号栽在「方向对但决策废」（轮廓系数）与「方向直接反」（塌缩检测），
故本探针除方向外**必须同时报「按该信号决策后的错误数」**才算通过。

用法：
    cd baseline
    diarizen_venv/bin/python src/spk_relabel_probe.py \\
        output/v026_pick output/oracle_spk_v026 \\
        ../data/extracted/dev/dev/wav /tmp/relabel_probe.json

已算过的 session 会跳过（可断点续跑）。
"""
from __future__ import annotations

import collections
import json
import logging
import os
import sys
import tempfile

import numpy as np  # type: ignore[import-not-found]
import soundfile as sf  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 判官嵌入模型。刻意不用 campplus —— CAM 源标签由它聚出，会循环论证。
JUDGE_EMB_MODEL = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
# 窗长/窗移。1.5s 是上游 CAM++ 管线默认子段长，且短于 dev 段长中位数 1.93s，
# 足以拆开「4 秒内两人交替」这类混合段（logs/2026-08-08.md §十 第 4 条的死因）。
WIN_SEC = 1.5
WIN_SHIFT = 0.75
# 短于此长度的窗不取嵌入（声纹对极短片段不稳）
MIN_WIN_SEC = 0.5


def build_embedder():
    from modelscope.pipelines import pipeline  # type: ignore[import-not-found]
    from modelscope.utils.constant import Tasks  # type: ignore[import-not-found]

    logger.info("加载判官嵌入模型: %s", JUDGE_EMB_MODEL)
    return pipeline(task=Tasks.speaker_verification, model=JUDGE_EMB_MODEL)


def windows(start: float, end: float) -> list[tuple[float, float]]:
    """段内切窗；段本身短于一个窗时整段作一个窗。"""
    if end - start <= WIN_SEC:
        return [(start, end)] if end - start >= MIN_WIN_SEC else []
    out, t = [], start
    while t < end:
        w1 = min(t + WIN_SEC, end)
        if w1 - t >= MIN_WIN_SEC:
            out.append((t, w1))
        t += WIN_SHIFT
    return out


def embed_windows(sv, wav_path: str, recs: list[dict]) -> list[np.ndarray]:
    """逐段返回其各窗的 L2 归一化嵌入矩阵（无有效窗则为空矩阵）。"""
    x, sr = sf.read(wav_path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    per_seg: list[np.ndarray] = []
    for r in recs:
        embs = []
        for w0, w1 in windows(r["start_time"], r["end_time"]):
            clip = x[max(0, int(w0 * sr)) : min(len(x), int(w1 * sr))]
            if len(clip) < int(MIN_WIN_SEC * sr):
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav") as fh:
                sf.write(fh.name, clip, sr)
                out = sv([fh.name], output_emb=True)
            e = np.asarray(out["embs"][0], dtype=np.float64)
            embs.append(e / (np.linalg.norm(e) + 1e-9))
        per_seg.append(np.vstack(embs) if embs else np.zeros((0, 1)))
    return per_seg


def session_signal(recs: list[dict], per_seg: list[np.ndarray]) -> list[dict]:
    """留一法算 incumbent / challenger 分数。"""
    spks = sorted({r["speaker"] for r in recs})
    dim = next((e.shape[1] for e in per_seg if len(e)), 0)
    if len(spks) < 2 or not dim:
        return []

    # 每个说话人的窗嵌入之和与计数，用于留一法快速扣除
    tot = {s: np.zeros(dim) for s in spks}
    cnt: collections.Counter = collections.Counter()
    for r, e in zip(recs, per_seg):
        if len(e):
            tot[r["speaker"]] += e.sum(axis=0)
            cnt[r["speaker"]] += len(e)

    rows = []
    for i, (r, e) in enumerate(zip(recs, per_seg)):
        if not len(e):
            continue
        inc = r["speaker"]
        scores = {}
        for s in spks:
            v, n = tot[s].copy(), cnt[s]
            if s == inc:                       # 留一：扣掉本段自己的窗
                v = v - e.sum(axis=0)
                n = n - len(e)
            if n <= 0:
                continue
            prof = v / n
            prof = prof / (np.linalg.norm(prof) + 1e-9)
            scores[s] = float((e @ prof).mean())
        if inc not in scores or len(scores) < 2:
            continue
        ch_val, ch_spk = max((v, s) for s, v in scores.items() if s != inc)
        rows.append({
            "seg": i,
            "incumbent": scores[inc],
            "challenger": ch_val,
            "challenger_spk": ch_spk,
            "delta": ch_val - scores[inc],
        })
    return rows


def truth_should_change(recs: list[dict], oracle: list[dict]) -> list[bool]:
    """真值：该段真实归属 != 其当前 hyp 标签所对应的多数真实归属 → 应改。"""
    major: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for h, o in zip(recs, oracle):
        major[h["speaker"]][o["speaker"]] += 1
    ref_of = {s: c.most_common(1)[0][0] for s, c in major.items()}
    return [o["speaker"] != ref_of[h["speaker"]] for h, o in zip(recs, oracle)]


def main() -> int:
    if len(sys.argv) < 5:
        logger.error("用法见模块 docstring：%s", __doc__)
        return 2
    pred_dir, oracle_dir, wav_dir, out_path = sys.argv[1:5]
    # 取嵌入是本探针的全部成本。缓存后「换打分方式」只需秒级重算 —— 2026-08-06 已因
    # 未落盘中间产物踩坑一次，不再重蹈。
    cache_dir = sys.argv[5] if len(sys.argv) > 5 else None
    sids = sorted(
        f.split(".")[0]
        for f in os.listdir(pred_dir)
        if f.endswith(".seglst.json") and f[0].isdigit()
    )
    if not sids:
        logger.error("预测目录为空：%s", pred_dir)
        return 1

    res: dict[str, list] = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            res = json.load(fh)
        logger.info("续跑：已有 %d session", len(res))

    sv = build_embedder()
    for k, sid in enumerate(sids, 1):
        if sid in res:
            continue
        with open(f"{pred_dir}/{sid}.seglst.json", encoding="utf-8") as fh:
            recs = sorted(json.load(fh), key=lambda r: r["start_time"])
        with open(f"{oracle_dir}/{sid}.seglst.json", encoding="utf-8") as fh:
            orc = sorted(json.load(fh), key=lambda r: r["start_time"])
        if len(orc) != len(recs):
            logger.warning("%s 段数不一致，跳过", sid)
            continue
        per_seg = embed_windows(sv, f"{wav_dir}/{sid}.wav", recs)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            # 变长的逐段窗嵌入拍平成一个矩阵 + 一个「每个窗属于第几段」的索引
            flat = [e for e in per_seg if len(e)]
            owner = [i for i, e in enumerate(per_seg) for _ in range(len(e))]
            np.savez(
                f"{cache_dir}/{sid}.npz",
                embs=np.vstack(flat) if flat else np.zeros((0, 1)),
                seg_of_win=np.array(owner, dtype=int),
                speakers=np.array([r["speaker"] for r in recs]),
                oracle=np.array([r["speaker"] for r in orc]),
            )
        rows = session_signal(recs, per_seg)
        flags = truth_should_change(recs, orc)
        for row in rows:
            row["should_change"] = flags[row["seg"]]
        res[sid] = rows
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False)
        logger.info("[%d/%d] %s: %d 段有信号", k, len(sids), sid, len(rows))
    logger.info("RELABEL PROBE DONE %d session → %s", len(res), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
