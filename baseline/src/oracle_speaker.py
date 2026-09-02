"""归属轴可达性探针：锁住文本与时间轴，只把 speaker 打到最优。

与 `oracle_text.py` 互为镜像 —— 那个锁 speaker 换 words，这个锁 words 换 speaker。
两者夹出「文本轴空间」与「归属轴空间」，避免把两种成本混为一谈。

## 为什么需要它（CLAUDE.md「可达性探针门」）

归属轴上已有 6+ 个分离器全部失败（DiariZen / Sortformer / pyannote / cVBx /
ERes2NetV2 / SoulX-30B）。但那些全是**整体替换**：外部模型产出一整套分割+标注，
必须端到端打赢 MOSS/CAM 才算数。**替换失败不等于这条轴没空间，更不等于「结合」也没用。**

本探针把归属成本拆成两块，因为它们要靠**完全不同**的机制去修：

| 口径 | 构造 | 含义 |
|---|---|---|
| **relabel oracle** | 每段 speaker 换成「与该段时间重叠最多的 ref 说话人」 | 只重贴标签能修多少 |
| **段纯度** | 一段内若有 2+ 个 ref 说话人各占可观时长，则该段"不纯" | 重贴标签修不了、必须重切时间轴 |

**读法**：
- relabel oracle 拿走的那部分 → 可由「外部模型只做标签决策」的结合式机制够到
  （声纹重聚类 / 说话人确认 / TS-VAD 二次确认），**不需要它产出完整分割**
- 段纯度揭示的残差 → 必须动时间轴（切分 / 边界细化），是另一类机制

用法：
    cd baseline
    .venv/bin/python src/oracle_speaker.py --pred-dir output/v026_pick --out output/oracle_spk_v026
    PYTHONPATH=src .venv/bin/python src/batch_evaluate.py --dir output/oracle_spk_v026
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEV_REF = BASE_DIR.parent / "data/extracted/dev/dev/ref.seglst.json"

# 一段内某 ref 说话人占比低于此值就不算「实质参与」，避免边界抖动被误判为不纯
PURITY_MIN_SHARE = 0.15


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def best_speaker(hyp: dict, refs: list[dict]) -> tuple[str | None, dict[str, float]]:
    """返回 (重叠最多的 ref 说话人, 各 ref 说话人在该段内的重叠时长)。"""
    shares: dict[str, float] = defaultdict(float)
    for r in refs:
        ov = overlap(hyp["start_time"], hyp["end_time"], r["start_time"], r["end_time"])
        if ov > 0:
            shares[r["speaker"]] += ov
    if not shares:
        return None, {}
    return max(shares, key=lambda k: shares[k]), dict(shares)


def main() -> int:
    ap = argparse.ArgumentParser(description="归属轴 oracle 探针（只换 speaker）")
    ap.add_argument("--pred-dir", required=True, help="提供文本与时间轴的预测目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=str(DEV_REF))
    args = ap.parse_args()

    with open(args.ref, encoding="utf-8") as fh:
        ref_all = json.load(fh)
    by_session: dict[str, list[dict]] = defaultdict(list)
    for r in ref_all:
        by_session[r["session_id"]].append(r)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = sorted(Path(args.pred_dir).glob("[0-9]*.seglst.json"))
    if not preds:
        logger.error("预测目录为空：%s", args.pred_dir)
        return 1

    n_seg = n_impure = n_relabeled = n_nomatch = 0
    impure_words = total_words = 0
    for p in preds:
        sid = p.name.split(".")[0]
        with open(p, encoding="utf-8") as fh:
            recs = json.load(fh)
        refs = by_session.get(sid, [])
        out = []
        for h in recs:
            spk, shares = best_speaker(h, refs)
            n_seg += 1
            nw = len(h["words"].split())
            total_words += nw
            if spk is None:
                n_nomatch += 1          # 该段与任何 ref 语音都不重叠（纯插入）
                out.append(dict(h))
                continue
            tot = sum(shares.values())
            if sum(1 for v in shares.values() if v / tot >= PURITY_MIN_SHARE) > 1:
                n_impure += 1
                impure_words += nw
            if spk != h["speaker"]:
                n_relabeled += 1
            out.append({**h, "speaker": spk})
        with open(out_dir / f"{sid}.seglst.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)

    logger.info("共 %d 段，其中 %d 段(%.1f%%)标签被改写",
                n_seg, n_relabeled, n_relabeled / n_seg * 100)
    logger.info("与任何 ref 语音都不重叠：%d 段（纯插入，重贴标签救不了）", n_nomatch)
    logger.info("**不纯段**（含 2+ 个 ref 说话人各占 ≥%.0f%%）：%d 段(%.1f%%)，"
                "覆盖 %d 词(%.1f%%) —— 这部分必须重切时间轴",
                PURITY_MIN_SHARE * 100, n_impure, n_impure / n_seg * 100,
                impure_words, impure_words / total_words * 100)
    logger.info("→ %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
