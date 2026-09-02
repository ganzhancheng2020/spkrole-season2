"""分析 MOSS 输出 vs ref 的 substitution 错误，找稳定错字模式（规范化延伸）。

思路：v028 的字符规范化只抓了"ref=0 的写法"。但 sub 错误里还有一类"ref 有、MOSS 写错
成别的字"的高频同音/形近错——如果能用对齐找到**稳定**的 MOSS→ref 错字对，且这些对
在多个 session 复现，就有机会做安全的同音字映射（规范化延伸）。

⚠️ 安全门槛：只有「MOSS 常写错成 X、ref 稳定用 Y」且「X 在 ref 里出现次数远低于 Y」的对
才值得映射。若 X 在 ref 里也常用，则 X 是真词，不能映射（会弄坏对的）。

用法：
    cd baseline
    .venv/bin/python src/analyze_sub.py output/v028_dev data/extracted/dev/dev/ref.seglst.json
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def align_count(hyp_toks: list, ref_toks: list) -> Counter:
    """SequenceMatcher 逐 session 对齐，统计 MOSS→ref 的 substitution 错字对。"""
    m = SequenceMatcher(None, hyp_toks, ref_toks, autojunk=False)
    out = Counter()
    for tag, i1, i2, j1, j2 in m.get_opcodes():
        if tag in ("replace",):
            for i in range(i1, i2):
                for j in range(j1, j2):
                    h, r = hyp_toks[i], ref_toks[j]
                    if len(h) == 1 and len(r) == 1:
                        out[(h, r)] += 1
    return out


def main() -> int:
    if len(sys.argv) < 3:
        logger.error("用法: %s <hyp_dir> <ref_path>", __file__)
        return 2
    hyp_dir, ref_path = Path(sys.argv[1]), Path(sys.argv[2])
    ref_all = json.loads(ref_path.read_text(encoding="utf-8"))
    ref_by_sid: dict[str, list[str]] = {}
    for r in ref_all:
        ref_by_sid.setdefault(r["session_id"], []).append(r["words"])
    ref_char_count: Counter = Counter()
    for r in ref_all:
        for w in r["words"].split():
            ref_char_count.update(w)

    sub = Counter()
    for p in sorted(hyp_dir.glob("[0-9]*.seglst.json")):
        sid = p.name.split(".")[0]
        if sid not in ref_by_sid:
            continue
        hyp_recs = json.loads(p.read_text(encoding="utf-8"))
        hyp_all = " ".join(r["words"] for r in hyp_recs).split()
        ref_flat = " ".join(ref_by_sid[sid]).split()
        sub.update(align_count(hyp_all, ref_flat))

    logger.info("=== 稳定错字对（MOSS→ref），ref 里该字出现次数 < 20 才值得（排除真词）===")
    rows = [(h, r, n) for (h, r), n in sub.items()]
    rows.sort(key=lambda x: -x[2])
    for h, r, n in rows:
        if n >= 2 and ref_char_count.get(h, 0) < 20:
            note = "⚠️ ref也常用" if ref_char_count.get(h, 0) > 5 else ""
            print(f"  {h!r}→{r!r}: sub={n} (ref中{h}= {ref_char_count.get(h,0)}) {note}")
    logger.info("总 substitution 错字对: %d", sum(sub.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())