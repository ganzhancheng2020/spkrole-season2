"""路径 1 思路 1：fun-asr 作为第三源加到 v011/v018 选优规则。

设计目标：oracle 三源上界 14.27%（vs v011 二源 14.73%），fun-asr 单独 26.07% 单独赢 28 段
（约 26%）。如果规则能挑出 fun-asr 赢的段，dev 实际分数预期可降到 15.0–16.0% 区间。

## v011/v018 规则（继承自 pick_ensemble.py:103-172）

- v011_ok: (m_nspk <= 2 OR m_turn <= 0.10) AND c_nspk >= 3
- cam_over (D): c_nspk >= 6
- cam_eq5 (H): c_nspk == 5 AND m_nspk >= 4

## 新增 fun-asr 触发条件（路径 1 思路 1）

设计原则：
- fun-asr 单独 26.07% > MOSS 18%：fun-asr 整体弱，但**它在某些段赢**（28 段）
- fun-asr 赢段特征：MOSS 漏段、fun-asr 段更多、文本更完整
- 触发：fun-asr 说话人数 >= MOSS 说话人数 + 2（说明 MOSS 漏人时 fun-asr 没漏）
- AND fun-asr 至少 3 人（避免 2 人段误触发）
- 排序：fun-asr 触发优先级**最低**（v011/v018 规则未触发时才考虑）

## 风险

- v011 单条件 holdout 95% 区间 [−3.06, +1.34] 含 0，新增 fun-asr 条件可能引入未确证判据
- dev 验证（路径 A）：< 16.98% 才认为赚到

## 用法

    cd baseline
    .venv/bin/python src/path1_pick.py \
      --moss output/v007_moss \
      --cam output/hyp_sw_m3_7_0.70 \
      --funasr output_dev_funasr \
      --out output/path1_dev
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 阈值（继承 v011/v018，模块级默认）
MOSS_MAX_SPK = 2
MOSS_MAX_TURN_RATE = 0.10
CAM_MIN_SPK = 3
CAM_OVER_SPK = 6
CAM_EQ5_MOSS_MIN = 4

# 新增 fun-asr 触发阈值
FUNASR_NSPK_DELTA = 2     # fun-asr 比 MOSS 多 ≥ 2 人 → 怀疑 MOSS 漏人
FUNASR_MIN_SPK = 3        # fun-asr 至少 3 人（避免 2 人段误触发）


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def n_speakers(records: list[dict]) -> int:
    return len({r["speaker"] for r in records})


def turn_rate(records: list[dict]) -> float:
    """继承自 pick_ensemble.py:86-100。"""
    if not records:
        return 0.0
    span = max(r["end_time"] for r in records) - min(r["start_time"] for r in records)
    if span <= 0:
        return 0.0
    ordered = sorted(records, key=lambda r: r["start_time"])
    turns = sum(1 for a, b in zip(ordered, ordered[1:]) if a["speaker"] != b["speaker"])
    return turns / span


def main() -> int:
    ap = argparse.ArgumentParser(description="路径 1 思路 1: MOSS+CAM+fun-asr 三源按段选优")
    ap.add_argument("--moss", required=True, help="MOSS 预测目录")
    ap.add_argument("--cam", required=True, help="CAM++ 预测目录")
    ap.add_argument("--funasr", required=True, help="fun-asr 预测目录")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--moss-max-spk", type=int, default=MOSS_MAX_SPK)
    ap.add_argument("--moss-max-turn-rate", type=float, default=MOSS_MAX_TURN_RATE)
    ap.add_argument("--cam-min-spk", type=int, default=CAM_MIN_SPK)
    ap.add_argument("--cam-over-spk", type=int, default=CAM_OVER_SPK)
    ap.add_argument("--cam-eq5-moss-min", type=int, default=CAM_EQ5_MOSS_MIN)
    ap.add_argument("--funasr-nspk-delta", type=int, default=FUNASR_NSPK_DELTA)
    ap.add_argument("--funasr-min-spk", type=int, default=FUNASR_MIN_SPK)
    args = ap.parse_args()

    moss_dir = Path(args.moss)
    cam_dir = Path(args.cam)
    funasr_dir = Path(args.funasr)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = sorted(moss_dir.glob("[0-9]*.seglst.json"))
    if not preds:
        logger.error("MOSS 预测目录为空：%s", moss_dir)
        return 1

    switched: dict[str, list[str]] = {"v011": [], "D": [], "H": [], "funasr": [], "default": []}
    no_cam: list[str] = []
    no_funasr: list[str] = []
    for moss_path in preds:
        sid = moss_path.name.split(".")[0]
        cam_path = cam_dir / f"{sid}.seglst.json"
        funasr_path = funasr_dir / f"{sid}.seglst.json"

        moss_recs = load(moss_path)
        m_nspk = n_speakers(moss_recs)
        m_turn = turn_rate(moss_recs)

        cam_recs = load(cam_path) if cam_path.exists() else None
        if cam_recs is None:
            no_cam.append(sid)
            c_nspk = 0
        else:
            c_nspk = n_speakers(cam_recs)

        funasr_recs = load(funasr_path) if funasr_path.exists() else None
        if funasr_recs is None:
            no_funasr.append(sid)
            f_nspk = 0
        else:
            f_nspk = n_speakers(funasr_recs)

        # v011 触发条件（继承）
        few_spk = m_nspk <= args.moss_max_spk
        low_turn = m_turn <= args.moss_max_turn_rate
        v011_ok = (few_spk or low_turn) and c_nspk >= args.cam_min_spk

        # v018 D 规则
        cam_over = c_nspk >= args.cam_over_spk

        # v018 H 规则
        cam_eq5 = (c_nspk == 5 and m_nspk >= args.cam_eq5_moss_min)

        # 路径 1 新增：fun-asr 触发条件（优先级最低）
        funasr_trigger = (
            funasr_recs is not None
            and f_nspk >= args.funasr_min_spk
            and f_nspk - m_nspk >= args.funasr_nspk_delta
            and not v011_ok
            and not cam_over
            and not cam_eq5
        )

        # 优先级：v011 > D > H > funasr > default MOSS
        if v011_ok:
            src_path = cam_path
            tag = "v011"
        elif cam_over:
            src_path = moss_path
            tag = "D"
        elif cam_eq5:
            src_path = cam_path
            tag = "H"
        elif funasr_trigger:
            src_path = funasr_path
            tag = "funasr"
        else:
            src_path = moss_path
            tag = "default"

        shutil.copy(src_path, out_dir / f"{sid}.seglst.json")
        switched[tag].append(sid)

    if no_cam:
        logger.warning("%d 段无 CAM++ 预测: %s", len(no_cam), no_cam[:5])
    if no_funasr:
        logger.warning("%d 段无 fun-asr 预测: %s", len(no_funasr), no_funasr[:5])
    logger.info("共 %d 段：v011=%d, D=%d, H=%d, funasr=%d, default=%d",
                len(preds),
                len(switched["v011"]), len(switched["D"]),
                len(switched["H"]), len(switched["funasr"]),
                len(switched["default"]))
    logger.info("fun-asr 触发 session: %s", switched["funasr"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
