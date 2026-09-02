"""复用已有 FireRed 词级时间戳，重分配到新的 MOSS 段边界。

fr_retext.py 每次都要重跑 FireRed 转写（慢）；本脚本复用已有的 .ts.json，
只做时间戳→段的重分配（秒级），适用于换 MOSS 模型后的 fr_retext 重建。

用法：
    cd baseline
    .venv/bin/python src/redistribute_fr.py \
        --ts-dir output/fr_retext_dev_ts \
        --pred-dir output/dev_ck84_fp32 \
        --out output/fr_retext_ck84
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PUNCT = re.compile(r"""[，。！？、；：""''…,.!?;:"'()\[\]【】]""")
TOKEN = re.compile(r"[A-Za-z]+|[0-9]+|[一-鿿]")


def redistribute(
    ts_data: list, recs: list[dict]
) -> list[dict]:
    """把 FireRed 词级时间戳分配到 MOSS 段上，保持段边界/说话人不变，只换 words。"""
    buckets: dict[int, list[str]] = {j: [] for j in range(len(recs))}
    for tok, st, en in ts_data:
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
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts-dir", required=True, help="FireRed .ts.json 目录")
    ap.add_argument("--pred-dir", required=True, help="MOSS 预测目录（.seglst.json）")
    ap.add_argument("--out", required=True, help="输出目录")
    args = ap.parse_args()

    ts_dir = Path(args.ts_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = sorted(
        p for p in pred_dir.glob("*.seglst.json") if p.name.split(".")[0].isdigit()
    )
    print(f"处理 {len(preds)} 个 session", flush=True)

    for i, p in enumerate(preds, 1):
        sid = p.name.split(".")[0]
        ts_path = ts_dir / f"{sid}.ts.json"
        if not ts_path.exists():
            print(f"  [{i}/{len(preds)}] {sid}: 缺 .ts.json，跳过", flush=True)
            continue

        recs = json.loads(p.read_text(encoding="utf-8"))
        ts_data = json.loads(ts_path.read_text(encoding="utf-8"))
        out = redistribute(ts_data, recs)

        (out_dir / p.name).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # 同时复制 .ts.json（word_arb 可能需要）
        import shutil
        shutil.copy2(ts_path, out_dir / f"{sid}.ts.json")

        print(
            f"  [{i}/{len(preds)}] {sid}: {len(ts_data)} 词 -> {len(recs)} 段",
            flush=True,
        )

    print("REDISTRIBUTE DONE", flush=True)


if __name__ == "__main__":
    main()
