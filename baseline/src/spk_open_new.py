"""「开新说话人」：MOSS 认为某段属于它自己没输出的说话人时，为该段开一个新说话人。

## 为什么需要这个操作（2026-08-20）

`logs/2026-08-08.md` 在记录归属轴第一个奏效机制（`spk_relabel`，区分度 +0.3013，
仅捕获 relabel oracle 的 3.7%）时，明确列了三条**未试**的改进方向，第 2 条：

> **缺「开新人」分支**：ref 6 人而预测 4 人时，真正的说话人不在候选集里，
> 当前实现只能在已有标签间挑（session 071 的 spk5 就属此类）。

实测这块有多大：干净 holdout-24 上，归属轴 5.736 点空间里
**60%（3.46 点）集中在 6 个「oracle 需要用到 hyp 没有的说话人」的 session**。
任何只在既有标签间重贴的机制**结构上够不到它**。

## 信号

`spk_ac_probe.py` 记录的是数字 1–9 的**全部**对数概率，不只是会话里已有的那几个。
所以「MOSS 认为这里该是第 K+1 个人」这件事，数据里现成就有：

    gap = max(logp[K+1..9]) − logp[当前说话人]

| 分组 | gap 均值 |
|---|---|
| oracle 需要开新人的段（13）| **−6.909** |
| 其余段（295）| −16.702 |
| **区分度** | **+9.793** |

参照：本项目归属轴唯一奏效机制 +0.3013；同日测的按段改写信号 +3.220。
**这是本项目见过的最强归属信号。**

## 与「按段改标签」的结构性差别

[[segment-label-proxy-misaligned]] 证明：按段改标签的段级命中率**不转化为 tcpWER**，
因为 tcpWER 做 session 级全局最优映射，局部改对可能全局改错。

**开新人不是局部重贴**：它改变说话人**数量**。当 ref 有一个说话人在 hyp 里
完全没有对应物时，它的词在任何映射下都是硬损失 —— 补上这个簇是**结构性**修复。
⚠️ 但这是推理，不是证据。本模块一律以 **7 模型合并的兑现 tcpWER** 为准。

用法：
    python src/spk_open_new.py --pred-dir <dir> --scores <scores.json> --out <dir> [--tau -4]
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description="按 MOSS 越界说话人似然开新说话人")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=-4.0,
                    help="越界数字最高分 − 当前标签分 超过该值才开新人")
    ap.add_argument("--max-new", type=int, default=2, help="每个 session 最多开几个新说话人")
    ap.add_argument("--min-support", type=int, default=1,
                    help="至少几段指向同一个新说话人才承认它（真实漏判者应有多段）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sc = {o["session"]: {round(s["start_time"], 3): s for s in o["segs"]}
          for o in json.load(open(args.scores, encoding="utf-8"))}

    n_seg = n_new_seg = 0
    n_sess = 0
    for p in sorted(glob.glob(f"{args.pred_dir}/[0-9]*.seglst.json")):
        sid = os.path.basename(p).split(".")[0]
        recs = sorted(json.load(open(p, encoding="utf-8")), key=lambda r: r["start_time"])
        order: dict[str, int] = {}
        for r in recs:
            if r["speaker"] not in order:
                order[r["speaker"]] = len(order) + 1
        K = len(order)
        oob = [str(d) for d in range(K + 1, 10)]

        # 先收集候选，按 gap 排序，只保留最强的 max_new 个「新说话人身份」
        cands = []
        for i, r in enumerate(recs):
            s = sc.get(sid, {}).get(round(r["start_time"], 3))
            if s is None or not oob:
                continue
            lp = s["logp"]
            cur = str(order[r["speaker"]])
            best = max(oob, key=lambda d: lp[d])
            gap = lp[best] - lp[cur]
            if gap > args.tau:
                cands.append((gap, i, best))
        # 支持度门：真实的漏判说话人应有多段指向它，只出现一次的多半是噪声
        sup: dict[str, int] = {}
        for _, _, d in cands:
            sup[d] = sup.get(d, 0) + 1
        keep_ids = {d for d, c in sup.items() if c >= args.min_support}
        keep_ids = set(sorted(keep_ids, key=lambda d: -max(
            (g for g, _, dd in cands if dd == d), default=-1e9))[:args.max_new])

        out = []
        touched = False
        for i, r in enumerate(recs):
            r2 = dict(r)
            hit = [(g, d) for g, j, d in cands if j == i and d in keep_ids]
            if hit:
                r2["speaker"] = f"spknew{hit[0][1]}"
                n_new_seg += 1
                touched = True
            out.append(r2)
            n_seg += 1
        n_sess += touched
        with open(f"{args.out}/{sid}.seglst.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)

    logger.info("SPK_OPEN_NEW τ=%.1f: %d 段改判为新说话人，涉及 %d 个 session（共 %d 段）",
                args.tau, n_new_seg, n_sess, n_seg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
