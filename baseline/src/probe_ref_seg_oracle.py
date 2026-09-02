"""P1 授权测量：ref 分割天花板 —— 把 MOSS 的词按 ref 的 (段,说话人) 结构重新装桶。

## 测什么

解码时干预（改分割+归属）的真实天花板。此前归属 oracle（attr_bucket 3.574 点）只在
**已有 hyp 段内**映射标签，把「说话人缺失、文本熔进别人段」的部分算漏了。
本探针把 hyp 的每个 token 按线性插值时间归入重叠最大的 ref 段：
文本保持 MOSS 原词（错词仍是错词），**分割与说话人全部换成 ref 真值**。
- hyp token 落在 ref 覆盖外 → 归入时间最近的 ref 段（insertion 保留，不作弊）。
- 空的 ref 段丢弃（等价于 scoring 时的 del）。

判读：oracle − base ≥ ~1.5 dev 点 ⇒ 解码时干预有真实量级，进入 P2；
≤ 0.5 点 ⇒ 整个「改分割/归属」族（含解码时）当场死，换路。

用法：
    python src/probe_ref_seg_oracle.py --hyp output/_cv --out /tmp/ora_refseg
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE_DIR)
REF_PATH = os.path.join(ROOT, "data/extracted/dev/dev/ref.seglst.json")
MEET = os.path.join(ROOT, ".venv/bin/meeteval-wer")


def evaluate(records: list[dict], tag: str) -> float:
    with tempfile.TemporaryDirectory() as td:
        h = os.path.join(td, f"h_{tag}.json")
        json.dump(records, open(h, "w", encoding="utf-8"), ensure_ascii=False)
        subprocess.run([MEET, "tcpwer", "-r", REF_PATH, "-h", h, "--collar", "5"],
                       capture_output=True, check=True)
        res = json.load(open(h.replace(".json", "_tcpwer.json"), encoding="utf-8"))
        return float(res["error_rate"]) * 100


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref_by: dict[str, list[dict]] = {}
    for r in json.load(open(REF_PATH, encoding="utf-8")):
        ref_by.setdefault(r["session_id"], []).append(r)

    os.makedirs(args.out, exist_ok=True)
    base_recs, ora_recs = [], []
    for p in sorted(glob.glob(os.path.join(args.hyp, "[0-9]*.seglst.json"))):
        sid = os.path.basename(p).split(".")[0]
        hyp = json.load(open(p, encoding="utf-8"))
        ref = ref_by.get(sid, [])
        base_recs.extend(hyp)
        if not ref:
            ora_recs.extend(hyp)
            continue
        # 展开 hyp token 并线性插值时间
        toks = []  # (time_mid, token)
        for hrec in hyp:
            words = hrec["words"].split()
            n = len(words)
            s, e = hrec["start_time"], hrec["end_time"]
            for i, w in enumerate(words):
                t = s + (i + 0.5) * (e - s) / max(n, 1)
                toks.append((t, w))
        # 每个 token 归入重叠/最近 ref 段
        buckets: dict[int, list[tuple[float, str]]] = {i: [] for i in range(len(ref))}
        for t, w in toks:
            best, bov = None, -1e18
            for i, r in enumerate(ref):
                # 与 ref 段的重叠：token 在段内 = 距离 0；否则取到段边界的负距离
                dist = max(r["start_time"] - t, t - r["end_time"], 0.0)
                ov = -dist
                if ov > bov:
                    best, bov = i, ov
            buckets[best].append((t, w))
        for i, r in enumerate(ref):
            ws = [w for _, w in sorted(buckets[i])]
            if not ws:
                continue
            ora_recs.append({"session_id": sid, "speaker": r["speaker"],
                             "start_time": r["start_time"], "end_time": r["end_time"],
                             "words": " ".join(ws)})

    er_base = evaluate(base_recs, "base")
    er_ora = evaluate(ora_recs, "ora")
    print(f"基线（{args.hyp} raw）: {er_base:.4f}%")
    print(f"ref 分割天花板     : {er_ora:.4f}%")
    print(f"天花板收益         : {er_base - er_ora:.4f} dev 点")
    print(f"判读：≥1.5 授权 P2；≤0.5 当场死")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
