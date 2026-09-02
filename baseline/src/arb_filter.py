"""词级仲裁的**换段过滤器**：用独立自由解码源投票，撤掉判官那些有害的换段。

## 为什么需要它（2026-08-28 实测的动机）

`wiki/insights/word-arb-capture-headroom.md` 长期把剩余空间归给「判官太保守、漏了 296 块」，
但放松 MARGIN 反而更差，这个矛盾一直没解开。今天把 shortfall 按**误换 / 漏换**拆开量化：

| 口径 | 106 段干净 CV | 相对出货 |
|---|---|---|
| MOSS 源 `_cv` | 17.7405% | — |
| 出货仲裁 `MARGIN=0.05 MAX_BLOCKS=6`（v048 配方） | 16.8098% | 基线 |
| **oracle{MOSS, 出货仲裁} —— 只撤掉有害换段** | **16.2853%** | **−0.5245** |

即：判官换了 341 段，其中 **88 段严格有害**。把这 88 段撤掉值 **0.5245 本地点**，
是破榜首所需 0.247 点的 **2.1 倍**。所以主因不是漏换而是误换，**该做精度不是召回**。

MARGIN 扫描已证明「打分差值大小」不是可用的过滤信号（0.03→16.8201 / 0.05→16.8098 /
0.07→16.8098 / 0.10→16.8869，平的）。所以过滤信号必须来自判官之外。

## 机制

判官用的是**教师强制似然**（给定文本算概率）。本过滤器用**自由解码**结果当独立投票人：
同一段音频上，另一个系统自己解出来的文本，离「换后」近还是离「换前」近？

- 离换后更近 → 这次换有第二方支持，保留；
- 离换前更近 → 只有判官一家之言，撤回 MOSS。

自由解码与教师强制是同一模型也**不同口径**（一个要生成、一个只打分），
所以 `_cv_aed_pseudo` 这类源即使单独成绩很差（24.84%）也仍可能是有效投票人。

⚠️ 已知陷阱（2026-08-28 实测）：把弱源当**候选文本源**加进 oracle 仲裁，
编辑距离代理会变好而 tcpWER 变差（3 源 proxy 3960/16.5887%、4 源 proxy 3906/16.6092%，
对照 2 源 proxy 4018/16.3881%）。**所以弱源只准当投票人，永远不准贡献文本。**
本脚本只在 MOSS 文本与仲裁文本之间二选一，投票人的文本不会进入输出。

## 用法

    cd baseline
    .venv/bin/python src/arb_filter.py \\
        --moss output/_cv --arb /tmp/wa_ctx0 \\
        --voters output/_cv_aed_pseudo,output/_cv_llm_seg \\
        --agree 1 --out /tmp/arb_f1

`--agree N`：至少 N 个投票人支持才保留换段（N=len(voters) 即全体一致）。
`--tie-keeps`：投票人到两边距离相等时算支持（默认算不支持，即倾向撤回）。

**阈值/变体只能在内层折选**（见 CLAUDE.md 硬约束），本脚本只负责产物，不做选择。
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def edit_distance(a: list[str], b: list[str]) -> int:
    """token 级 Levenshtein（与 oracle_arbitrate.py 同款，段平均 ~16 token）。"""
    if not a:
        return len(b)
    prev = list(range(len(a) + 1))
    for j, bj in enumerate(b, 1):
        cur = [j]
        for i, ai in enumerate(a, 1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (ai != bj)))
        prev = cur
    return prev[-1]


def load(d: Path, name: str) -> list[dict]:
    return json.loads((d / name).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="仲裁换段的投票过滤器")
    ap.add_argument("--moss", required=True, help="换前源（仲裁的输入）")
    ap.add_argument("--arb", required=True, help="仲裁产物（换后）")
    ap.add_argument("--voters", required=True, help="逗号分隔的自由解码投票人目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--agree", type=int, default=1, help="保留换段所需的最少支持票")
    ap.add_argument("--tie-keeps", action="store_true",
                    help="投票人两边距离相等时算支持（默认算不支持）")
    args = ap.parse_args()

    moss_dir, arb_dir = Path(args.moss), Path(args.arb)
    voters = [Path(v.strip()) for v in args.voters.split(",") if v.strip()]
    if not voters:
        logger.error("至少要一个投票人")
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_switch = n_kept = n_revert = n_seg = 0
    for p in sorted(arb_dir.glob("*.seglst.json")):
        if not p.name.split(".")[0].isdigit():
            continue
        arb = load(arb_dir, p.name)
        moss = load(moss_dir, p.name)
        vs = [load(v, p.name) for v in voters]
        if len({len(arb), len(moss), *(len(v) for v in vs)}) != 1:
            logger.error("%s 段数不一致：%s —— 各源段结构必须相同", p.name,
                         [len(arb), len(moss), *(len(v) for v in vs)])
            return 1

        out_recs = []
        for i, h in enumerate(arb):
            n_seg += 1
            new_t, old_t = h["words"], moss[i]["words"]
            if new_t == old_t:
                out_recs.append(dict(h))
                continue
            n_switch += 1
            new_w, old_w = new_t.split(), old_t.split()
            support = 0
            for v in vs:
                ref_w = v[i]["words"].split()
                d_new = edit_distance(new_w, ref_w)
                d_old = edit_distance(old_w, ref_w)
                if d_new < d_old or (d_new == d_old and args.tie_keeps):
                    support += 1
            if support >= args.agree:
                n_kept += 1
                out_recs.append(dict(h))
            else:
                n_revert += 1
                out_recs.append({**h, "words": old_t})
        (out_dir / p.name).write_text(
            json.dumps(out_recs, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("%d 段：仲裁换了 %d 段 → 保留 %d、撤回 %d（agree>=%d, tie_keeps=%s）",
                n_seg, n_switch, n_kept, n_revert, args.agree, args.tie_keeps)
    logger.info("→ %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
