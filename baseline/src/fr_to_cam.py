"""把 FireRed 词级时间戳重排到 CAM++ 段，产出「FireRed 文本 + CAM++ 说话人」产物。

验证假设：CAM++ 说话人分离（声纹聚类）在近场短时对话上是否优于 MOSS 的端到端说话人。
用户要求实测「FireRed 文本 + CAM++ 说话人」的 dev 全口径 tcpWER。

输入：
- FireRed 词级时间戳（fr_retext.py 落盘的 {sid}.ts.json，[[token, start, end], ...]）
- CAM++ 段结构（output/hyp_sw_m3_7_0.70/{sid}.seglst.json，说话人+时间戳）

输出：
- {sid}.seglst.json：段结构+说话人来自 CAM++，words 来自 FireRed（按时间戳分配）

用法：
    cd baseline
    .venv/bin/python src/fr_to_cam.py \
      --ts output/fr_retext_dev_ts \
      --cam output/hyp_sw_m3_7_0.70 \
      --out output/fr_cam_dev
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PUNCT = re.compile(r"""[，。！？、；：""''…,.!?;:"'()\[\]【】]""")
TOKEN = re.compile(r"[A-Za-z]+|[0-9]+|[一-鿿]")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(records, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def reassign(ts: list, cam_recs: list[dict]) -> list[dict]:
    """按词级时间戳把 FireRed 词分配到 CAM++ 段。"""
    buckets = {j: [] for j in range(len(cam_recs))}
    for tok, st, en in ts:
        c = (st + en) / 2
        best, bd = None, 1e9
        for j, x in enumerate(cam_recs):
            if x["start_time"] <= c <= x["end_time"]:
                best, bd = j, 0
                break
            d = min(abs(c - x["start_time"]), abs(c - x["end_time"]))
            if d < bd:
                best, bd = j, d
        if best is not None:
            buckets[best].append(tok)
    out = []
    for j, x in enumerate(cam_recs):
        toks = [t for w in buckets[j] for t in TOKEN.findall(PUNCT.sub("", w))]
        w = " ".join(t.lower() if t.isascii() else t for t in toks).strip()
        out.append({**x, "words": w if w else x["words"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="FireRed 词级时间戳重排到 CAM++ 段")
    ap.add_argument("--ts", required=True, help="FireRed 词级时间戳目录（{sid}.ts.json）")
    ap.add_argument("--cam", required=True, help="CAM++ 段结构目录（{sid}.seglst.json）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ts_dir, cam_dir, out_dir = Path(args.ts), Path(args.cam), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_paths = sorted(cam_dir.glob("[0-9]*.seglst.json"))
    if not cam_paths:
        logger.error("CAM++ 目录为空：%s", cam_dir)
        return 1

    done = missing_ts = 0
    for cam_path in cam_paths:
        sid = cam_path.name.split(".")[0]
        ts_path = ts_dir / f"{sid}.ts.json"
        if not ts_path.exists():
            missing_ts += 1
            continue
        ts = load(ts_path)
        cam_recs = load(cam_path)
        out = reassign(ts, cam_recs)
        save(out, out_dir / f"{sid}.seglst.json")
        done += 1

    logger.info("%d 段重排完成，%d 段缺 FireRed 时间戳", done, missing_ts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())