"""把 FireRedASR2-LLM 的整场文本折成可评测的 SegLST，与 AED 同口径比纯文本。

## 口径与其已知偏向

LLM 分支只返回整段 text，**没有词级时间戳、没有说话人**。
为了能过 meeteval，本脚本把整场文本按 MOSS 的段边界**按字数比例**切成伪段。

⚠️ **这个口径对 LLM 略有不利**（时间戳是构造的，不是真的）。故判读规则先定死：
- LLM **仍明显更好** → 结论可信（偏向与结论反向）
- LLM 更差 → **不能直接判死**，需先区分「模型差」与「伪时间戳的锅」

评测一律用 `erase_speaker` 抹平说话人后的纯文本口径 —— 与
`logs/2026-08-05` 给 AED 测的 19.42% 同口径。
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
    ap = argparse.ArgumentParser(description="LLM 整场文本 -> SegLST（按 MOSS 段边界按字数分配）")
    ap.add_argument("--llm-dir", required=True, help="fr_llm_transcribe 的输出")
    ap.add_argument("--seg-dir", required=True, help="提供段边界的目录（MOSS 预测）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n_sess = n_seg = miss = 0
    for p in sorted(glob.glob(f"{args.seg_dir}/[0-9]*.seglst.json")):
        sid = os.path.basename(p).split(".")[0]
        lp = os.path.join(args.llm_dir, f"{sid}.json")
        if not os.path.exists(lp):
            miss += 1
            continue
        text = json.load(open(lp, encoding="utf-8"))["text"]
        chars = [c for c in text if not c.isspace()]
        recs = sorted(json.load(open(p, encoding="utf-8")), key=lambda r: r["start_time"])
        # 按各段原有字数的比例切分整场文本
        weights = [max(1, len(r["words"].replace(" ", ""))) for r in recs]
        tot_w = sum(weights)
        out, pos = [], 0
        for r, w in zip(recs, weights):
            take = round(len(chars) * w / tot_w)
            piece = "".join(chars[pos:pos + take])
            pos += take
            out.append({**r, "words": " ".join(piece)})
        if pos < len(chars) and out:  # 余数并入最后一段，保证不丢字
            out[-1]["words"] = out[-1]["words"] + " " + " ".join(chars[pos:])
        json.dump(out, open(f"{args.out}/{sid}.seglst.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        n_sess += 1
        n_seg += len(out)
    logger.info("LLM_TO_SEGLST %d session / %d 段（缺 %d）-> %s", n_sess, n_seg, miss, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
