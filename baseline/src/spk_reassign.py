"""归属轴·纯重新分配：段边界与文本一字不动，只在**已有说话人之间**改标签。

## 与 v027 `spk_relabel.py` 的分工（不是替代，是补上它漏掉的那一半）

拆解 oracle 的改动（2026-08-13，v031 谱系 holdout-24）：

| 改动类型 | 占比 | 谁在打 |
|---|---|---|
| **纯重新分配**（目标说话人在该 session 的 hyp 里**已存在**）| **77.6%** | **本脚本** |
| 需要开新人（拆被合并的说话人）| 22.4% | v027（已封死：重尾关卡全 9 配置 FAIL）|

v027 把**每个**说话人都拆成两半当挑战者，副作用是在位者可以「取自己两个子档案里
最好的那个」—— 人为抬高在位者，并把结果推向 `_s` 新说话人。本脚本**不拆、不开新人、
说话人数守恒**。

## 为什么单靠「关掉拆分」还不够（2026-08-13 实测）

`SPLIT_THR=99` 关掉拆分后（等价于本脚本的朴素版），全 dev 净 −0.118 点，但按源拆开：

| 子集 | 净 | 改善 | 恶化 |
|---|---|---|---|
| CAM 源 37 段（段来自声纹聚类，**声学同质**）| −0.237 点 | 2 | **0** |
| MOSS 源 69 段（段由文本驱动，**26.2% 跨说话人**）| **+0.118 点** | 7 | **12** |

**判官在 MOSS 段上等于抛硬币。** 两个病根：

1. **被判的段本身是混合体** —— 一段里有 2 个说话人时，它的窗嵌入是混合，
   任何单一标签都只对一半，判官无从判起（`oracle_speaker.py` 实测 26.2% 的段不纯）。
2. **档案是用这些错标段建的** —— 判官在拿被污染的尺子量。

## 本脚本的两道机制（对应上面两个病根）

1. **段内一致性门控**：段内多窗两两余弦相似度的均值 `coh`。
   `coh < COH_MIN` = 段内声音不一致 = 大概率跨说话人 → **该段不参与仲裁**（保持原标签）。
   单窗段无从判断一致性，按 `SINGLE_WIN_TRUST` 决定是否信任。
2. **只用可信段建档案**：档案只累加 `coh >= COH_MIN` 的段的窗，
   切断「错标段污染档案」的回路。档案窗数不足的说话人不参与竞争。

打分沿用已验证三次的形态（v024 文本 / v026 词级 / v027 归属）：
**保留在位者，挑战者赢过 `MARGIN` 才换**，且留一法扣掉本段自己的窗。

用法：
    cd baseline
    diarizen_venv/bin/python src/spk_reassign.py output/v026_pick \\
        ../data/extracted/dev/dev/wav output/reassign_v2

已存在的输出文件会被跳过（可断点续跑）。
"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np  # type: ignore[import-not-found]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spk_relabel import build_embedder, embed_windows  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 段内一致性下限。低于此值认为段内混了不止一个人，不参与仲裁。
COH_MIN = float(os.environ.get("COH_MIN", "0.55"))
# 挑战者要赢在位者多少才换。沿用在干净全 dev 上扫出的峰值。
MARGIN = float(os.environ.get("MARGIN", "0.15"))
# 只有一个窗的段测不出一致性；"1" = 当作可信（参与仲裁与建档案）。
SINGLE_WIN_TRUST = os.environ.get("SINGLE_WIN_TRUST", "0") == "1"
# 一个说话人的档案至少要这么多窗才算数
MIN_PROFILE_WIN = int(os.environ.get("MIN_PROFILE_WIN", "3"))


def coherence(seg_embs: np.ndarray) -> float | None:
    """段内窗两两余弦相似度均值。窗数 <2 时返回 None（测不出）。"""
    if len(seg_embs) < 2:
        return None
    sims = seg_embs @ seg_embs.T
    n = len(seg_embs)
    return float((sims.sum() - np.trace(sims)) / (n * (n - 1)))


def reassign(recs: list[dict], embs: np.ndarray, owner: np.ndarray) -> tuple[list[str], int, int]:
    """返回 (新标签, 改写段数, 因段内不一致被跳过的段数)。"""
    cur = [r["speaker"] for r in recs]
    if len(embs) == 0 or len(set(cur)) < 2:
        return list(cur), 0, 0

    # ---- 1. 逐段一致性 ----
    win_of = [np.where(owner == i)[0] for i in range(len(recs))]
    trust: list[bool] = []
    n_skip = 0
    for w in win_of:
        if len(w) == 0:
            trust.append(False)
            continue
        c = coherence(embs[w])
        if c is None:
            trust.append(SINGLE_WIN_TRUST)
            continue
        ok = c >= COH_MIN
        trust.append(ok)
        if not ok:
            n_skip += 1

    # ---- 2. 只用可信段建档案 ----
    prof: dict[str, np.ndarray] = {}
    cnt: dict[str, int] = {}
    for i, w in enumerate(win_of):
        if not trust[i] or len(w) == 0:
            continue
        s = cur[i]
        prof[s] = prof.get(s, np.zeros(embs.shape[1])) + embs[w].sum(0)
        cnt[s] = cnt.get(s, 0) + len(w)
    usable = {s for s in prof if cnt[s] >= MIN_PROFILE_WIN}
    if len(usable) < 2:
        return list(cur), 0, n_skip

    # ---- 3. 逐段仲裁（留一法：扣掉本段自己的窗）----
    new = list(cur)
    n_changed = 0
    for i, w in enumerate(win_of):
        if not trust[i] or len(w) == 0 or cur[i] not in usable:
            continue
        scores: dict[str, float] = {}
        for s in usable:
            v, n = prof[s].copy(), cnt[s]
            if cur[i] == s:                       # 在位者要扣掉本段贡献
                v, n = v - embs[w].sum(0), n - len(w)
            if n < MIN_PROFILE_WIN:
                continue
            p = v / n
            scores[s] = float((embs[w] @ (p / (np.linalg.norm(p) + 1e-9))).mean())
        if cur[i] not in scores or len(scores) < 2:
            continue
        ch_val, ch = max((v, s) for s, v in scores.items() if s != cur[i])
        if ch_val - scores[cur[i]] >= MARGIN:
            new[i] = ch
            n_changed += 1
    return new, n_changed, n_skip


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
    logger.info("配方 COH_MIN=%.2f MARGIN=%.2f SINGLE_WIN_TRUST=%s",
                COH_MIN, MARGIN, SINGLE_WIN_TRUST)

    sv = build_embedder()
    total = total_skip = 0
    for k, sid in enumerate(sids, 1):
        out_path = f"{out_dir}/{sid}.seglst.json"
        if os.path.exists(out_path):
            continue
        with open(f"{pred_dir}/{sid}.seglst.json", encoding="utf-8") as fh:
            recs = sorted(json.load(fh), key=lambda r: r["start_time"])
        embs, owner = embed_windows(sv, f"{wav_dir}/{sid}.wav", recs)
        labels, n, n_skip = reassign(recs, embs, owner)
        total += n
        total_skip += n_skip
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump([{**h, "speaker": s} for h, s in zip(recs, labels)],
                      fh, ensure_ascii=False, indent=2)
        logger.info("[%d/%d] %s: 改写 %d 段，跳过不一致 %d 段（累计 %d / %d）",
                    k, len(sids), sid, n, n_skip, total, total_skip)
    logger.info("SPK REASSIGN DONE 改写 %d 段，因段内不一致跳过 %d 段 → %s",
                total, total_skip, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
