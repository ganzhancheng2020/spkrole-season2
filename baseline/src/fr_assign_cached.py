"""把**已缓存的** FireRedASR2 整场词级时间戳分配到任意切分（纯 CPU，不跑 ASR）。

## 为什么可以复用缓存

`fr_retext.py` 的做法是「整场 42 秒转写一次 → 用模型自带的词级时间戳把词分配回段边界」。
转写本身**只依赖音频**，与我们的切分无关；只有第二步（分配）依赖切分。
所以 `output/fr_retext_dev_ts/*.ts.json` 里那 106 场的词级时间戳
**对任何新切分都成立**，不必重跑 FireRed。

2026-08-06 曾因没落盘 ts 而被迫整体重跑（logs §二十八），当时特意加了缓存 —— 这里正是它的用处。

分配规则与 `fr_retext.py` 逐字一致：词按中点归入包含它的段；无包含段则归给边界最近的段。
段为空时保留原文本（不制造空段）。

用法：
    python src/fr_assign_cached.py <pred_dir> <ts_dir> <out_dir>
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PUNCT = re.compile(r"""[，。！？、；：""''…,.!?;:"'()\[\]【】]""")
TOKEN = re.compile(r"[A-Za-z]+|[0-9]+|[一-鿿]")


def main() -> int:
    if len(sys.argv) < 4:
        logger.error("用法见模块 docstring：%s", __doc__)
        return 2
    pred_dir, ts_dir, out_dir = (Path(a) for a in sys.argv[1:4])
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = sorted(p for p in pred_dir.glob("*.seglst.json") if p.name.split(".")[0].isdigit())
    n_ok = n_miss = 0
    for p in preds:
        sid = p.name.split(".")[0]
        tsp = ts_dir / f"{sid}.ts.json"
        if not tsp.exists():
            n_miss += 1
            continue
        recs = json.loads(p.read_text(encoding="utf-8"))
        ts = json.loads(tsp.read_text(encoding="utf-8"))

        buckets: dict[int, list[str]] = {j: [] for j in range(len(recs))}
        for tok, st, en in ts:
            c = (st + en) / 2
            best, bd = None, 1e9
            for j, x in enumerate(recs):
                if x["start_time"] <= c <= x["end_time"]:
                    best, bd = j, 0
                    break
                d = min(abs(c - x["start_time"]), abs(c - x["end_time"]))
                if d < bd:
                    best, bd = j, d
            if best is not None:
                buckets[best].append(tok)

        out = []
        for j, x in enumerate(recs):
            toks = [t for w in buckets[j] for t in TOKEN.findall(PUNCT.sub("", w))]
            w = " ".join(t.lower() if t.isascii() else t for t in toks).strip()
            out.append({**x, "words": w if w else x["words"]})
        (out_dir / p.name).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
        # ts 一并复制，下游 word_arb 需要它
        (out_dir / f"{sid}.ts.json").write_text(tsp.read_text(encoding="utf-8"),
                                                encoding="utf-8")
        n_ok += 1

    logger.info("FR_ASSIGN_CACHED %d 段完成，缺 ts %d 段 -> %s", n_ok, n_miss, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
