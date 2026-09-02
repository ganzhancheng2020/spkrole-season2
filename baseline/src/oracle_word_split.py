"""归属轴可达性探针：用**真实字级时间戳** + oracle 说话人做词级归属与切分。

## 它回答的问题

2026-08-26 Phase 0 把归属轴的瓶颈定位在「词级文本归属」，但当时的两个口径都不可达：

| 口径 | Δ | 备注 |
|---|---|---|
| oracle 切 + oracle 词级分配 | −3.373 | 分配靠 ref，不可达 |
| oracle 切 + **时间比例**分配 | +0.704 | 可达但假设「字在段内均匀分布」，错 |

真正该问的是：**有了真实字级时间戳（`word_times.py`，96.1% 锚点覆盖），
词级归属的天花板是多少？** 本探针锁住文本与字时间戳，只把说话人换成 oracle。

与 `oracle_speaker.py` 的区别：那个是**整段**换标签（段不切），
所以 348 个「不纯段」（含 2+ ref 说话人）无论贴谁都错一半；
本探针**允许在段内按字切开**，正是那 348 段需要的操作。

## 构造

1. `word_times.transfer()` 给每个字打时间戳
2. 每个字按中点落进哪个 ref 段，取那个 ref 说话人（无包含段则取最近的）
3. 同一记录内**连续同说话人的字合成一个子段**，起止取该串首字 t0 / 末字 t1
4. `--min-run N`：短于 N 个字的子段并回前一个子段（抑制过度切分）

⚠️ 这是**上界**，不是某个机制能拿到的分。步骤 2 用了 ref。
读法：上界 − 当前 = 「词级归属 + 切分」这条路的全部可夺回量。
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import word_times as wt

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ref_intervals(ref_recs: list[dict]) -> dict[str, list[tuple[float, float, str]]]:
    by: dict[str, list[tuple[float, float, str]]] = {}
    for r in ref_recs:
        by.setdefault(str(r["session_id"]), []).append(
            (float(r["start_time"]), float(r["end_time"]), str(r["speaker"]))
        )
    for sid in by:
        by[sid].sort()
    return by


def speaker_at(t: float, ivs: list[tuple[float, float, str]]) -> str:
    for s, e, spk in ivs:
        if s <= t <= e:
            return spk
    best, bd = "", float("inf")
    for s, e, spk in ivs:
        d = min(abs(s - t), abs(e - t))
        if d < bd:
            best, bd = spk, d
    return best


def split_record(rec: dict, times: list[tuple[float, float, bool]],
                 ivs: list[tuple[float, float, str]], min_run: int) -> list[dict]:
    toks = str(rec.get("words", "")).split()
    if not toks or len(times) != len(toks):
        return [dict(rec)]
    owners = [speaker_at((t0 + t1) / 2.0, ivs) for t0, t1, _a in times]

    runs: list[list[int]] = []
    for i, o in enumerate(owners):
        if runs and owners[runs[-1][0]] == o:
            runs[-1].append(i)
        else:
            runs.append([i])

    if min_run > 1:
        merged: list[list[int]] = []
        for run in runs:
            if merged and len(run) < min_run:
                merged[-1].extend(run)
            else:
                merged.append(run)
        runs = merged

    out: list[dict] = []
    for run in runs:
        cand = {owners[i] for i in run}
        spk = max(cand, key=lambda s: sum(1 for i in run if owners[i] == s))
        out.append({
            "session_id": rec["session_id"],
            "speaker": spk,
            "start_time": times[run[0]][0],
            "end_time": max(times[run[-1]][1], times[run[0]][0]),
            "words": " ".join(toks[i] for i in run),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="词级归属 + 切分的可达性上界探针")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--ts-dir", default="output/fr_retext_dev_ts")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-run", type=int, default=1,
                    help="短于该字数的子段并回前一段（抑制过度切分）")
    args = ap.parse_args()

    ann = wt.annotate_dir(Path(args.pred_dir), Path(args.ts_dir))
    ivs_by = ref_intervals(json.load(open(args.ref, encoding="utf-8")))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_in = n_out = n_split = 0
    for sid, (recs, per) in ann.items():
        ivs = ivs_by.get(sid, [])
        new: list[dict] = []
        for rec, times in zip(recs, per):
            pieces = split_record(rec, times, ivs, args.min_run) if ivs else [dict(rec)]
            n_in += 1
            n_out += len(pieces)
            n_split += int(len(pieces) > 1)
            new += pieces
        json.dump(new, open(out_dir / f"{sid}.seglst.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    anc, tot = wt.coverage(ann)
    logger.info("锚点覆盖 %d/%d (%.1f%%)", anc, tot, 100.0 * anc / max(tot, 1))
    logger.info("段 %d → %d（被切开 %d 段，%.1f%%）",
                n_in, n_out, n_split, 100.0 * n_split / max(n_in, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
