"""归属轴改写：细粒度窗口 + 子档案 + margin 门控，只改 speaker 标签。

生产版（不依赖 ref）。方向检验与真值分组见 `spk_relabel_probe.py`。

## 机制

段边界与文本**一字不动**，只重贴 speaker：

1. 每段内切 `WIN_SEC`/`WIN_SHIFT` 小窗，逐窗取嵌入（判官 ERes2NetV2，与 CAM++ 独立）
2. **每个 hyp 说话人内部试着一分为二** —— MOSS 把两人并成一个时，那个说话人的档案本身
   就是混合体；拆开后被并进去的人正好现形（这同时就是「开新人」分支）
3. 逐段算 `delta = challenger - incumbent`（留一法，避免在位者被自己的窗抬高）
4. **只有 delta >= MARGIN 才改**

三条约束的来由（logs/2026-08-08.md §十 实测）：**不产出分割**（替换已塌 6 次）、
**不整体覆盖标签**（盲重聚类 29.804%，X12 同死因）、**嵌入粒度细于段**
（26.2% 的段跨说话人，段级嵌入是混合体）。

与文本轴 v024/v026 同构：**保留在位者，挑战者赢过 margin 才换** —— 第三条轴上第三次成立。

## dev 验证（logs/2026-08-08.md §十一/§十二）

| | train82 | holdout24 | 全dev |
|---|---|---|---|
| v026 基准 | 16.175 | 15.047 | 15.920 |
| **本脚本** | **15.976 (-0.199)** | **14.387 (-0.660)** | **15.617 (-0.303)** |

**holdout 改善是 train 的 3.3 倍**；ins/del/sub 三项同降；`missed_speaker` 41->38，
`falarm_speaker` 保持 8（拆出新说话人没有造成误报恶化）。

参数由 dev 扫描确定：`SPLIT_THR=0.45`（0.40-0.45 同分，0.50 起退化）、
`MARGIN=0.15`（0.12-0.20 平滑无悬崖）、`MIN_SUB_WIN=2`。**迭代无收益**（第 2 轮即收敛且略差）。

用法：
    cd baseline
    # dev（复现 15.617%）
    diarizen_venv/bin/python src/spk_relabel.py output/v026_pick \\
        ../data/extracted/dev/dev/wav output/v027_dev
    # test
    diarizen_venv/bin/python src/spk_relabel.py output_test_v026 \\
        ../data/extracted/test/test/wav output_test_v027

已存在的输出文件会被跳过（可断点续跑）。
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
from sklearn.cluster import AgglomerativeClustering  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 判官嵌入模型。刻意不用 campplus —— CAM 源标签由它聚出，会循环论证。
JUDGE_EMB_MODEL = os.environ.get(
    "JUDGE_EMB", "iic/speech_eres2netv2_sv_zh-cn_16k-common"
)
WIN_SEC = 1.5           # 上游 CAM++ 默认子段长，且短于 dev 段长中位数 1.93s
WIN_SHIFT = 0.75
MIN_WIN_SEC = 0.5       # 短于此的窗不取嵌入（声纹对极短片段不稳）
# 说话人内部两个子簇的质心余弦距离超过此值，才认为「这个档案被污染了」，拆开
SPLIT_THR = float(os.environ.get("SPLIT_THR", "0.45"))
# 挑战者要赢在位者这么多才改标签
MARGIN = float(os.environ.get("MARGIN", "0.15"))
MIN_SUB_WIN = int(os.environ.get("MIN_SUB_WIN", "2"))   # 子簇最少窗数


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


def embed_windows(sv, wav_path: str, recs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """返回 (所有窗的 L2 归一化嵌入矩阵, 每个窗属于第几段)。"""
    x, sr = sf.read(wav_path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    embs: list[np.ndarray] = []
    owner: list[int] = []
    for i, r in enumerate(recs):
        for w0, w1 in windows(r["start_time"], r["end_time"]):
            clip = x[max(0, int(w0 * sr)) : min(len(x), int(w1 * sr))]
            if len(clip) < int(MIN_WIN_SEC * sr):
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav") as fh:
                sf.write(fh.name, clip, sr)
                out = sv([fh.name], output_emb=True)
            e = np.asarray(out["embs"][0], dtype=np.float64)
            embs.append(e / (np.linalg.norm(e) + 1e-9))
            owner.append(i)
    if not embs:
        return np.zeros((0, 1)), np.zeros(0, dtype=int)
    return np.vstack(embs), np.array(owner, dtype=int)


def subprofiles(embs: np.ndarray, win_spk: np.ndarray) -> np.ndarray:
    """每个 hyp 说话人内部试着一分为二，返回每个窗的子档案 id。

    MOSS 把两人并成一个时，那个说话人的窗嵌入呈双峰；拆开后被并进去的人现形。
    分不开（质心距离不够 / 一侧太小）就保持单档案。
    """
    sub = np.empty(len(embs), dtype=object)
    for s in sorted(set(win_spk.tolist())):
        idx = np.where(win_spk == s)[0]
        sub[idx] = f"{s}#0"
        if len(idx) < 2 * MIN_SUB_WIN:
            continue
        lab = AgglomerativeClustering(
            2, metric="cosine", linkage="average"
        ).fit_predict(embs[idx])
        a, b = idx[lab == 0], idx[lab == 1]
        if len(a) < MIN_SUB_WIN or len(b) < MIN_SUB_WIN:
            continue
        ca, cb = embs[a].mean(0), embs[b].mean(0)
        ca /= np.linalg.norm(ca) + 1e-9
        cb /= np.linalg.norm(cb) + 1e-9
        if 1.0 - float(ca @ cb) < SPLIT_THR:
            continue
        sub[a], sub[b] = f"{s}#0", f"{s}#1"
    return sub


def relabel(recs: list[dict], embs: np.ndarray, owner: np.ndarray) -> tuple[list[str], int]:
    """返回 (新标签, 改写段数)。"""
    cur = np.array([r["speaker"] for r in recs], dtype=object)
    if len(embs) == 0 or len(set(cur.tolist())) < 2:
        return [str(v) for v in cur], 0

    sub = subprofiles(embs, np.array([cur[i] for i in owner], dtype=object))
    tot: dict[str, np.ndarray] = {}
    cnt: collections.Counter = collections.Counter()
    for e, p in zip(embs, sub):
        tot[p] = tot.get(p, np.zeros(embs.shape[1])) + e
        cnt[p] += 1

    new = cur.copy()
    n_changed = 0
    for i in range(len(cur)):
        w = np.where(owner == i)[0]
        if len(w) == 0:
            continue
        scores = {}
        for p in tot:
            v, n = tot[p].copy(), cnt[p]
            own = w[sub[w] == p]                   # 留一：扣掉本段自己的窗
            if len(own):
                v = v - embs[own].sum(0)
                n -= len(own)
            if n <= 0:
                continue
            prof = v / n
            prof = prof / (np.linalg.norm(prof) + 1e-9)
            scores[p] = float((embs[w] @ prof).mean())
        mine = [p for p in scores if p.split("#")[0] == cur[i]]
        others = [(v, p) for p, v in scores.items() if p.split("#")[0] != cur[i]]
        if not mine or not others:
            continue
        ch_val, ch_p = max(others)
        if ch_val - max(scores[p] for p in mine) >= MARGIN:
            # `#1` 是从某个被污染档案里拆出来的人 -> 落成一个新说话人标签
            new[i] = ch_p.split("#")[0] + ("_s" if ch_p.endswith("#1") else "")
            n_changed += 1
    return [str(v) for v in new], n_changed


def main() -> int:
    if len(sys.argv) < 4:
        logger.error("用法见模块 docstring：%s", __doc__)
        return 2
    pred_dir, wav_dir, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    sids = sorted(
        f.split(".")[0]
        for f in os.listdir(pred_dir)
        if f.endswith(".seglst.json") and f[0].isdigit()
    )
    if not sids:
        logger.error("预测目录为空：%s", pred_dir)
        return 1

    sv = build_embedder()
    total = 0
    for k, sid in enumerate(sids, 1):
        out_path = f"{out_dir}/{sid}.seglst.json"
        if os.path.exists(out_path):
            continue
        with open(f"{pred_dir}/{sid}.seglst.json", encoding="utf-8") as fh:
            recs = sorted(json.load(fh), key=lambda r: r["start_time"])
        embs, owner = embed_windows(sv, f"{wav_dir}/{sid}.wav", recs)
        labels, n = relabel(recs, embs, owner)
        total += n
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump([{**h, "speaker": s} for h, s in zip(recs, labels)],
                      fh, ensure_ascii=False, indent=2)
        logger.info("[%d/%d] %s: 改写 %d 段（累计 %d）", k, len(sids), sid, n, total)
    logger.info("SPK RELABEL DONE 累计改写 %d 段 → %s", total, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
