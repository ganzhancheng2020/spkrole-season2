"""择优信号实验：spk_diff 方向（CAM 人数 vs MOSS 人数）作为换段依据。

发现（2026-08-09）：oracle 分析显示，择优轴漏网段（该换 CAM 没换）的 spk_diff
多为 -1（CAM 比 MOSS 少 1 人），而误伤段多为 spk_diff >= 0。这与现有 v018 规则
「CAM 人数多 -> 用 CAM」方向相反。本脚本实测「spk_diff 方向」作为择优信号。

用法：
    cd baseline
    .venv/bin/python src/spk_diff_pick.py output/wa_m0_dev output/hyp_sw_m3_7_0.70 output/sdpick <mode>
    mode: lt  = spk_diff <= -1 时用 CAM（CAM 少判人）
          le  = spk_diff <= 0  时用 CAM
          ne  = spk_diff != 1  时用 CAM（排除 CAM 多判1人）
          nn  = spk_diff == -1 时用 CAM（严格 CAM 少判1人）
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def n_spk(recs):
    return len({r["speaker"] for r in recs})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("moss_dir")
    ap.add_argument("cam_dir")
    ap.add_argument("out_dir")
    ap.add_argument("mode", choices=["lt", "le", "ne", "nn"])
    args = ap.parse_args()

    moss_dir, cam_dir, out_dir = Path(args.moss_dir), Path(args.cam_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def use_cam(diff: int) -> bool:
        return {
            "lt": diff <= -1,
            "le": diff <= 0,
            "ne": diff != 1,
            "nn": diff == -1,
        }[args.mode]

    switched = []
    for p in sorted(moss_dir.glob("[0-9]*.seglst.json")):
        sid = p.name.split(".")[0]
        cam_p = cam_dir / f"{sid}.seglst.json"
        if not cam_p.exists():
            shutil.copy(p, out_dir / p.name)
            continue
        mr, cr = load(p), load(cam_p)
        diff = n_spk(cr) - n_spk(mr)
        if use_cam(diff):
            shutil.copy(cam_p, out_dir / p.name)
            switched.append(sid)
        else:
            shutil.copy(p, out_dir / p.name)

    logger.info("mode=%s 切换到 CAM++ %d 段: %s", args.mode, len(switched), switched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())