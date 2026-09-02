"""短段归属改写：MOSS 说话人似然 + 长度门，只在既有说话人间重贴标签。

## 靶子必须打对（本机制的关键前提）

早先用「**纯段**标错」当目标测得边际命中率仅 19%，据此判死。那是打错了靶：
- 纯段标错只有 45 段（基准率 3.4%），且只值约 225 词
- 真正值 **1.96 点**的 oracle 要改 **125 段**（含混合段，基准率 **9.3%**）

换成正确目标后，同一批打分的边际命中率 **19% → 72%**。
**类别失衡确实是死因，但失衡程度是被错误的靶子夸大的。**

## 信号

MOSS 对说话人标记 `[Sxx]` 的 teacher forcing 似然（`spk_ac_probe.py` 产出），
裁判用**基座**模型 —— 出货模型的仿真素材含全部 dev，不能当裁判。

同时满足 `arbitration-signal-must-be-acoustic` 的两个必要条件：
**音频条件** ✅ 且 **看得见说话人标签** ✅。

## 长度门的依据

标错的段**是短段**：时长 1.21s vs 2.93s、词数 5.0 vs 14.3（均约正常段的 1/3）。
根因是错误集中在说话人 embedding 最弱的尺度上，而 MOSS 似然靠**整场上下文**，
不受短段音频量限制 —— 正好补这个位。

实测（106 段干净 CV 集，目标 = 全部该改的 125 段）：

| 词数门 | τ | 改写 | 改对 | 改坏 | 净 | 命中率 |
|---|---|---|---|---|---|---|
| **≤5** | **1.5–2.0** | 19–20 | 13 | 5 | **+8** | **72%** |
| ≤5 | 1.0 | 22 | 13 | 7 | +6 | 65% |
| ≤3 | 2.0 | 11 | 6 | 4 | +2 | 60% |
| 无门 | 2.0 | 34 | 15 | 17 | −2 | 47% |

**τ∈[1.5, 2.0] 是平台（结果相同），词数门 5 是明确最优。**

用法：
    python src/spk_short_relabel.py --pred-dir <dir> --scores <scores.json> --out <dir>
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
    ap = argparse.ArgumentParser(description="短段归属改写（MOSS 似然 + 长度门）")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--scores", required=True, help="spk_ac_probe.py 输出")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=1.5, help="替代者需超出当前者该幅度（平台 1.5–2.0）")
    ap.add_argument("--max-words", type=int, default=5, help="只改词数 ≤ 此值的段")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sc = {o["session"]: {round(s["start_time"], 3): s for s in o["segs"]}
          for o in json.load(open(args.scores, encoding="utf-8"))}

    n_ch = n_tot = miss = 0
    sess = set()
    for p in sorted(glob.glob(f"{args.pred_dir}/[0-9]*.seglst.json")):
        sid = os.path.basename(p).split(".")[0]
        recs = sorted(json.load(open(p, encoding="utf-8")), key=lambda r: r["start_time"])
        # 说话人按首次出现顺序编号，与 build_spk_probe_jsonl.py 一致
        order: dict[str, str] = {}
        for r in recs:
            if r["speaker"] not in order:
                order[r["speaker"]] = str(len(order) + 1)
        inv = {v: k for k, v in order.items()}

        out = []
        for r in recs:
            n_tot += 1
            r2 = dict(r)
            s = sc.get(sid, {}).get(round(r["start_time"], 3))
            if s is None:
                miss += 1
            elif len(order) > 1 and len(r["words"].split()) <= args.max_words:
                cur = order[r["speaker"]]
                lp = s["logp"]
                alt = max((d for d in order.values() if d != cur), key=lambda d: lp[d])
                if lp[alt] - lp[cur] > args.tau:
                    r2["speaker"] = inv[alt]
                    n_ch += 1
                    sess.add(sid)
            out.append(r2)
        with open(f"{args.out}/{sid}.seglst.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)

    logger.info("SPK_SHORT_RELABEL τ=%.1f 词数≤%d：改写 %d / %d 段（%.2f%%），涉及 %d 个 session，无打分 %d",
                args.tau, args.max_words, n_ch, n_tot, 100 * n_ch / max(1, n_tot), len(sess), miss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
