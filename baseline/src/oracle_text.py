"""文本轴可达性探针：锁住说话人时间轴，把文本直接给到 oracle。

## 为什么需要它（CLAUDE.md「可达性探针门」）

X13 封死了「换更强 ASR」这条机制（Qwen3-ASR 1.7B 劣于 MOSS 0.9B 1.74 点，
详见 `logs/2026-08-05.md`）。但**死墙只证明某类机制够不到这条轴，不证明轴没空间**。
判「文本轴机制耗尽」之前必须先跑一道机制无关的探针：锁住其它指标不退，
直接把目标轴打到理论最优，看还剩多少空间。

这里的构造 = **说话人与时间戳全部沿用 hyp（不动一个字段），只把 words 换成 ref 原文**。
于是：
  - 文本轴误差归零（所有 ref 词都出现且仅出现一次，词数守恒）
  - 剩余误差 = 纯归属代价 + 超出 collar 的时间偏移

**读法**：`v011 全 dev 16.979% − 本探针结果 = 文本轴的可达上限`。
若该差值 < 2.08 点（破榜首所需），说明**即使文本完美也不够**，
文本轴该整条判死、力气挪回归属轴；若远大于 2.08，才值得为文本轴新起一个机制 family。

⚠️ 这不是「某个 ASR 能达到的分」，是**任何 ASR 都不可能超过的下界**。

## 构造细节

每条 ref 记录归给**时间重叠最多**的 hyp 段；完全无重叠时归给中心点最近的 hyp 段
（MOSS 时间轴没覆盖到的语音，其代价必须留在探针里，否则探针会偏乐观）。
同一 hyp 段收到多条 ref 时按时间顺序拼接。没收到任何 ref 词的 hyp 段被丢弃
（= oracle 认定它是多余段）。

⚠️ 归属用的是**时间重叠**，不是 ref 的 speaker 字段 —— 探针只喂文本，不泄露说话人答案。
若把 ref speaker 也用上，测的就是段级 oracle 重打标签（≈15.3%），是另一个量。

用法：
    cd baseline
    .venv/bin/python src/oracle_text.py --pred-dir output/v011_pick --out output/oracle_text_v011
    PYTHONPATH=src .venv/bin/python src/batch_evaluate.py --dir output/oracle_text_v011
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


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign(ref_recs: list[dict], hyp_recs: list[dict]) -> dict[int, list[dict]]:
    """每条 ref 归给重叠最多的 hyp 段；无重叠时归给中心最近的段。"""
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in ref_recs:
        r0, r1 = r["start_time"], r["end_time"]
        best_i, best_ov = -1, 0.0
        for i, h in enumerate(hyp_recs):
            ov = overlap(r0, r1, h["start_time"], h["end_time"])
            if ov > best_ov:
                best_i, best_ov = i, ov
        if best_i < 0:
            # 无重叠：MOSS 时间轴漏掉的语音。丢掉会让探针偏乐观，故落到最近段
            rc = (r0 + r1) / 2
            best_i = min(range(len(hyp_recs)),
                         key=lambda i: abs((hyp_recs[i]["start_time"]
                                            + hyp_recs[i]["end_time"]) / 2 - rc))
        groups[best_i].append(r)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser(description="文本轴 oracle 探针（只换 words）")
    ap.add_argument("--pred-dir", required=True, help="提供说话人与时间轴的预测目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=str(DEV_REF))
    args = ap.parse_args()

    pred_dir, out_dir = Path(args.pred_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_all = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    by_sess: dict[str, list[dict]] = defaultdict(list)
    for r in ref_all:
        by_sess[r["session_id"]].append(r)
    for v in by_sess.values():
        v.sort(key=lambda r: r["start_time"])

    n_sess = n_hyp = n_kept = n_word = 0
    for p in sorted(pred_dir.glob("*.seglst.json")):
        # 只收纯 session 命名的产物：目录里混有 all.hyp / *_tcpwer 等评测中间件
        sid = p.name.split(".")[0]
        if not sid.isdigit():
            continue
        hyp = json.loads(p.read_text(encoding="utf-8"))
        if not hyp:
            continue
        n_sess += 1
        n_hyp += len(hyp)

        groups = assign(by_sess[sid], hyp)
        out_recs = []
        for i, h in enumerate(hyp):
            words = " ".join(r["words"] for r in sorted(
                groups.get(i, []), key=lambda r: r["start_time"]) if r["words"])
            if not words:
                continue          # oracle 判定为多余段
            out_recs.append({**h, "words": words})
            n_word += len(words.split())
        n_kept += len(out_recs)
        (out_dir / p.name).write_text(
            json.dumps(out_recs, ensure_ascii=False, indent=2), encoding="utf-8")

    ref_words = sum(len(r["words"].split()) for r in ref_all)
    logger.info("%d session: hyp 段 %d → 保留 %d（丢弃多余段 %d）",
                n_sess, n_hyp, n_kept, n_hyp - n_kept)
    logger.info("词数 %d（ref %d）—— 应完全相等，不等说明构造漏词", n_word, ref_words)
    if n_word != ref_words:
        logger.error("词数不守恒，探针无效")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
