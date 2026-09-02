"""多系统词级仲裁的**上界**探针：逐段选文本最接近 ref 的那个源。

## 为什么需要它

X13 封死了「换一个更强的单 ASR」（Qwen3-ASR 1.7B 劣于 MOSS 0.9B 1.74 点）。
`oracle_text.py` 随后证明**文本轴本身还有 5.563 点空间**（oracle 11.4156% vs 实际 16.979%），
所以死的是机制不是轴，下一个候选 family 是「多系统各自在音频上出结果 + 词级仲裁」（ROVER 类）。

但仲裁类机制的收益有个硬上界：**两个系统必须错在不同地方**，仲裁才有得选。
如果 MOSS 和 Qwen3 在同样的词上一起错，再好的仲裁器也救不回来。

本探针给出那个上界：**假设有一个完美仲裁器**（每段都能选中更接近 ref 的源），
能到多少分。这是任何 ROVER / N-best 重打分 / 置信度融合都不可能超过的天花板。

**读法**：
- 上界 ≈ 单系统最好成绩 → 两系统错在同处，**仲裁 family 直接判死**，力气转归属轴。
- 上界显著更好 → 仲裁值得做，且该数就是它的收益天花板，可提前对账量级。

⚠️ 这是**上界不是预期值**。真实仲裁器没有 ref，只能靠置信度/投票逼近，
历史上典型只能吃到 oracle 的一部分（本项目唯一对照：择优轴捕获率 42%）。

## 构造

各源必须**段结构完全一致**（同 session 同段数）——`asr_retext.py` 克隆记录只改 words，
天然满足；脚本内置断言，不一致直接报错而不是静默错位。

ref 按时间重叠归段（同 `oracle_text.py`，**不使用 ref 的 speaker 字段**），
逐段算各源与 ref 文本的 token 编辑距离，取最小者。距离相同则取**靠前的源**
（故 `--sources` 第一个应放当前主线，避免上界被无意义的平局抬高）。

用法：
    cd baseline
    .venv/bin/python src/oracle_arbitrate.py \\
        --sources output/v011_pick,output/retext_qwen3_pad25 --out output/oracle_arb
    # 纯文本口径
    .venv/bin/python src/erase_speaker.py --dir output/oracle_arb --out /tmp/arb_flat
    PYTHONPATH=src .venv/bin/python src/batch_evaluate.py \\
        --ref /tmp/arb_flat/ref_flat.seglst.json --dir /tmp/arb_flat
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from oracle_text import DEV_REF, assign

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def edit_distance(a: list[str], b: list[str]) -> int:
    """token 级 Levenshtein。段平均约 16 token，全量 1234 段 × N 源，开销可忽略。"""
    if not a:
        return len(b)
    prev = list(range(len(a) + 1))
    for j, bj in enumerate(b, 1):
        cur = [j]
        for i, ai in enumerate(a, 1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (ai != bj)))
        prev = cur
    return prev[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description="多系统仲裁上界探针")
    ap.add_argument("--sources", required=True,
                    help="逗号分隔的 SegLST 目录；第一个作为段结构与平局优先源")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=str(DEV_REF))
    args = ap.parse_args()

    dirs = [Path(d.strip()) for d in args.sources.split(",") if d.strip()]
    if len(dirs) < 2:
        logger.error("至少要两个源才谈得上仲裁")
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_all = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    by_sess: dict[str, list[dict]] = defaultdict(list)
    for r in ref_all:
        by_sess[r["session_id"]].append(r)
    for v in by_sess.values():
        v.sort(key=lambda r: r["start_time"])

    wins = [0] * len(dirs)          # 严格胜出（距离唯一最小）
    ties = 0                        # 各源距离相同（含文本完全一致）
    n_seg = n_sess = 0
    base_err = [0] * len(dirs)      # 各源单独的总编辑距离
    orc_err = 0                     # 完美仲裁后的总编辑距离

    for p in sorted(dirs[0].glob("*.seglst.json")):
        sid = p.name.split(".")[0]
        if not sid.isdigit():
            continue
        srcs = []
        for d in dirs:
            f = d / p.name
            if not f.exists():
                logger.error("源缺文件：%s", f)
                return 1
            srcs.append(json.loads(f.read_text(encoding="utf-8")))
        if len({len(s) for s in srcs}) != 1:
            logger.error("%s 段数不一致：%s —— 各源段结构必须相同",
                         sid, [len(s) for s in srcs])
            return 1
        n_sess += 1

        base = srcs[0]
        groups = assign(by_sess[sid], base)
        out_recs = []
        for i, h in enumerate(base):
            n_seg += 1
            gold = " ".join(r["words"] for r in sorted(
                groups.get(i, []), key=lambda r: r["start_time"]) if r["words"]).split()
            dists = [edit_distance(s[i]["words"].split(), gold) for s in srcs]
            for k, d in enumerate(dists):
                base_err[k] += d
            best = min(range(len(dists)), key=lambda k: dists[k])
            orc_err += dists[best]
            if dists.count(dists[best]) > 1:
                ties += 1
            else:
                wins[best] += 1
            out_recs.append({**h, "words": srcs[best][i]["words"]})
        (out_dir / p.name).write_text(
            json.dumps(out_recs, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("%d session / %d 段 → %s", n_sess, n_seg, out_dir)
    for k, d in enumerate(dirs):
        logger.info("源%d %-32s 单独编辑距离 %6d | 严格胜出 %4d 段 (%.1f%%)",
                    k, d.name, base_err[k], wins[k], 100 * wins[k] / n_seg)
    logger.info("平局 %d 段 (%.1f%%)", ties, 100 * ties / n_seg)
    logger.info("完美仲裁后编辑距离 %d —— 相对最好单源降 %.1f%%",
                orc_err, 100 * (min(base_err) - orc_err) / min(base_err))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
