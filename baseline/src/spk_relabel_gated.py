"""归属轴·跨系统异议门控：只在 CAM 与 MOSS 判断不一致的段上启用声学判官。

## 为什么（2026-08-17，对抗性审查的方向 1）

审查用**生效口径**（只统计 pick 会选中的 session）重算后发现：

| 分支 | 生效段 | 可捞纯段 | 判官已改 | 命中 | 召回 |
|---|---|---|---|---|---|
| CAM | 278 | 18 | 3 | 1 | 6% |
| **MOSS** | **949** | **46** | 4 | **0** | **0%** |

**空间在 MOSS（46 段），力气却花在 CAM。** 而 MOSS 分支上：

- 放宽 margin（0.05→0.01）只从 6 段到 8 段
- 放宽一致性门控（COH 0.55→0.30）反而更差（+12 vs −36）

→ **不是阈值问题**。推测根因是**档案污染**：MOSS 系统性把 B 的段贴给 A 时，
A 的档案里混进了 B，于是 B 的段反而匹配 A 的档案，判官的 delta 自然为负。

## 机制：换一个独立证据源来缩小搜索空间

CAM 分支对**同一段音频**有完全独立的说话人判断（声纹聚类，与 MOSS 无关）。
把 CAM 标签按时间重叠映射到 MOSS 标签空间后，**取两者不一致的段作为候选**。

实测（dev，MOSS 分支生效的 69 个 session）：

| | 值 |
|---|---|
| 可捞纯段 | 46 |
| CAM 提出异议 | 233 段 |
| **异议且改对** | **12 段** |
| 召回 / 精确 | **26.1% / 5.2%** |

**单用 CAM 不行**（精确 5.2%，162 次误报），但它把搜索空间从 949 段压到 233 段，
且这 233 段里含 12 个真目标。**在这个子集内放低 margin，让声学判官做最终裁决** ——
筛子提召回，判官保精度。

⚠️ 与已封死方向的区别：
- **X12**（整体用 CAM 标签覆盖，29.804% 惨败）：本脚本**不覆盖标签**，CAM 只决定「看哪些段」。
- **`diarize_dover.py`**（DOVER-Lap 合并多套 diarization 产出新标签）：本脚本**不投票、不产标签**。

用法：
    cd baseline
    MARGIN=0.03 diarizen_venv/bin/python src/spk_relabel_gated.py \\
        output/v026_pick output/hyp_sw_m3_7_0.70 ../data/extracted/dev/dev/wav output/moss_gated

    # 对照组：关掉门控退化为全量仲裁
    GATE_ON_DISAGREE=0 MARGIN=0.03 ... （其余同上）

已存在的输出文件会被跳过（可断点续跑）。
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np  # type: ignore[import-not-found]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spk_relabel import build_embedder, embed_windows  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 候选子集内的门槛。搜索空间已被 CAM 缩小，故可比全量扫描时更低。
MARGIN = float(os.environ.get("MARGIN", "0.03"))
MIN_PROFILE_WIN = int(os.environ.get("MIN_PROFILE_WIN", "2"))
# 只在 CAM 异议段上仲裁；设 "0" 退化为全量仲裁（对照组）
GATE_ON_DISAGREE = os.environ.get("GATE_ON_DISAGREE", "1") == "1"


def cam_opinion(recs: list[dict], cam: list[dict]) -> dict[int, str]:
    """把 CAM 标签映射到 MOSS 标签空间，返回 {段序号: CAM 认为该段属于哪个 MOSS 标签}。"""
    ov: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for c in cam:
        for m in recs:
            o = min(c["end_time"], m["end_time"]) - max(c["start_time"], m["start_time"])
            if o > 0:
                ov[c["speaker"]][m["speaker"]] += o
    cmap = {k: max(v.items(), key=lambda kv: kv[1])[0] for k, v in ov.items() if v}
    out: dict[int, str] = {}
    for i, s in enumerate(recs):
        best, bo = None, 0.0
        for c in cam:
            x = min(c["end_time"], s["end_time"]) - max(c["start_time"], s["start_time"])
            if x > bo:
                bo, best = x, c
        if best is not None and best["speaker"] in cmap:
            out[i] = cmap[best["speaker"]]
    return out


def relabel(
    recs: list[dict], embs: np.ndarray, owner: np.ndarray, cand: set[int]
) -> tuple[list[str], int]:
    """只在 cand 指定的段上仲裁；档案用全部段建，留一法扣掉本段自己的窗。"""
    cur = [r["speaker"] for r in recs]
    if len(embs) == 0 or len(set(cur)) < 2 or not cand:
        return list(cur), 0
    win_of = [np.where(owner == i)[0] for i in range(len(recs))]
    prof: dict[str, np.ndarray] = {}
    cnt: dict[str, int] = {}
    for i, w in enumerate(win_of):
        if len(w) == 0:
            continue
        s = cur[i]
        prof[s] = prof.get(s, np.zeros(embs.shape[1])) + embs[w].sum(0)
        cnt[s] = cnt.get(s, 0) + len(w)
    usable = {s for s in prof if cnt[s] >= MIN_PROFILE_WIN}
    if len(usable) < 2:
        return list(cur), 0
    new = list(cur)
    n = 0
    for i in sorted(cand):
        w = win_of[i] if i < len(win_of) else []
        if len(w) == 0 or cur[i] not in usable:
            continue
        sc: dict[str, float] = {}
        for s in usable:
            v, k = prof[s].copy(), cnt[s]
            if cur[i] == s:
                v, k = v - embs[w].sum(0), k - len(w)
            if k < MIN_PROFILE_WIN:
                continue
            p = v / k
            sc[s] = float((embs[w] @ (p / (np.linalg.norm(p) + 1e-9))).mean())
        if cur[i] not in sc or len(sc) < 2:
            continue
        ch_v, ch = max((v, s) for s, v in sc.items() if s != cur[i])
        if ch_v - sc[cur[i]] >= MARGIN:
            new[i] = ch
            n += 1
    return new, n


def main() -> int:
    if len(sys.argv) < 5:
        logger.error("用法见模块 docstring：%s", __doc__)
        return 2
    pred_dir, cam_dir, wav_dir, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    sids = sorted(
        f.split(".")[0]
        for f in os.listdir(pred_dir)
        if f.endswith(".seglst.json") and f[0].isdigit()
    )
    if not sids:
        logger.error("预测目录为空：%s", pred_dir)
        return 1
    logger.info("配方 MARGIN=%.3f GATE_ON_DISAGREE=%s", MARGIN, GATE_ON_DISAGREE)

    sv = build_embedder()
    total = total_cand = 0
    for k, sid in enumerate(sids, 1):
        out_path = f"{out_dir}/{sid}.seglst.json"
        if os.path.exists(out_path):
            continue
        with open(f"{pred_dir}/{sid}.seglst.json", encoding="utf-8") as fh:
            recs = sorted(json.load(fh), key=lambda r: r["start_time"])
        cam_path = f"{cam_dir}/{sid}.seglst.json"
        cand: set[int] = set(range(len(recs)))
        if GATE_ON_DISAGREE and os.path.exists(cam_path):
            with open(cam_path, encoding="utf-8") as fh:
                cam = sorted(json.load(fh), key=lambda r: r["start_time"])
            op = cam_opinion(recs, cam)
            cand = {i for i, v in op.items() if v != recs[i]["speaker"]}
        total_cand += len(cand)
        embs, owner = embed_windows(sv, f"{wav_dir}/{sid}.wav", recs)
        labels, n = relabel(recs, embs, owner, cand)
        total += n
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump([{**h, "speaker": s} for h, s in zip(recs, labels)],
                      fh, ensure_ascii=False, indent=2)
        logger.info("[%d/%d] %s: 候选 %d 段，改写 %d 段（累计改 %d / 候选 %d）",
                    k, len(sids), sid, len(cand), n, total, total_cand)
    logger.info("GATED RELABEL DONE 改写 %d 段（候选共 %d 段）→ %s",
                total, total_cand, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
