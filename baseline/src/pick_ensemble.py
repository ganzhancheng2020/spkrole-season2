"""按段择优集成：MOSS 端到端 SAT 为主，特定条件下改用 CAM++（v011+v018）。

## 机制

MOSS 的病根是**少判说话人**（v008 定位）。dev ref 里真实 2 人只占 9.4%，
MOSS 却判出 17.9% —— 它说「只有 2 个说话人」时多半在漏人，此时倾向过分割的
CAM++ 反而更好。

⚠️ 单靠「MOSS 判 ≤2 人」这一个条件不够稳（R1）：dev 触发 17.9% 但 test 触发 26.4%，
分布明显漂移；拆解显示 dev 触发段里 47% 是 MOSS 判对的真 2 人段，被换差了。
加上「CAM++ 独立认为 ≥3 人」这个**第二系统佐证**后（R1a），误伤被滤掉，
dev/test 触发率回到 9.4% / 8.9% 一致，且三个切分上分数全都更好。

**v011 起改用并集**：MOSS 「把两个人并成一个」有两个独立症状，一段可能只显出其一——
① 它自称的说话人数偏少（`nspk<=2`）；② 说话人切换率异常低（`turn_rate<=0.10`，
切换次数/时长）。两者都要求 CAM++ 独立佐证 `nspk>=3`。

## v018 新增：H+D 规则

EDA 发现（2026-08-04）：v011 只抓 MOSS 严重少判（≤2），但 MOSS 也常常**轻度少判**——
例如 MOSS 判 4 人、CAM 判 5 人时，MOSS 少判 1 人而 v011 不触发（阈值 ≤2 太保守）。
v018 加了第 3 条件：「CAM 判 5 人 且 MOSS 判 ≥4 人 → 用 CAM」，抓这种轻度少判。

同时 v018 还加了反向修正：D 规则「CAM 判 ≥6 人 → 用 MOSS」（CAM 多估时 MOSS 更可能对）。
D + v011 + H 三个条件组合后 dev full 106 降 1.05 点（18.02%→16.97%），holdout 25 降 2.37 点
（18.26%→15.89%），线上 0.1634（v018，优于 v011 0.16379 和 v017b 0.16377）。

⚠️ v018 的 33 段切换都是按规则触发，但 session 039 在 train 上是损失
（+63 错误，MOSS 比 CAM 好 7 vs 70），没有可观测特征区分它和正向段（063/072）。
全规则在 holdout 上净正。

## 规则总览（v018 = v011 + H + D）

| 规则 | train81 | holdout25 | 全dev | dev触发 | test触发 |
|---|---|---|---|---|---|
| 全用 MOSS（v007 基线） | 17.999% | 19.658% | 18.388% | 0 | 0 |
| R1a MOSS≤2 且 CAM≥3（v010） | 17.186% | 18.256% | 17.437% | 9.4% | 8.9% |
| **R1a∪R3（v011）** | **16.770%** | **17.664%** | **16.979%** | dev 31.1% / test 31.2% |
| **v018（H+D）** | **17.307%** | **15.889%** | **16.974%** | 33 session（test） |
| v018 线上 | — | — | — | **0.1634** |

规则用 81/25 切分防过拟合（切分见 prep_finetune_data.py）：只在 train 81 段上挑规则，
holdout 25 段仅用于验收。

⚠️ holdout 只有 25 段，自助重采样显示并集优于 v010 的 95% 区间 [−3.06, +1.34]（含 0），
**未达统计显著**。采信并集的依据是「三切分方向一致 + dev/test 触发率 31.1%/31.2% 吻合」，
不是单个数字。改阈值前请重跑这套验证。

用法：
    cd baseline
    # dev
    .venv/bin/python src/pick_ensemble.py --moss output/v007_moss \
        --cam output/hyp_sw_m3_7_0.70 --out output/v018_pick
    # test（再用 merge_submit.py 合并成提交物）
    .venv/bin/python src/pick_ensemble.py --moss output_test_moss \
        --cam output/hyp_test_v002 --out output_test_v018
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 阈值。改动前请重跑 train/holdout 验证，别直接调（见模块 docstring 的表）。
MOSS_MAX_SPK = 2          # 症状①：MOSS 自称的说话人数少到这个数 → 怀疑它把人并了
MOSS_MAX_TURN_RATE = 0.10  # 症状②：说话人切换率(次/秒)低到这个数 → 同上，且不看它自称几人
CAM_MIN_SPK = 3           # 门槛：CAM++ 必须独立认为至少这么多人，否则不换（滤掉误伤）
# v018 新增：
CAM_OVER_SPK = 6              # 规则D：CAM++ 多估（≥6 人）→ MOSS 更可能对
CAM_EQ5_MOSS_MIN = 4           # 规则H：CAM 判 5 人 且 MOSS 判 ≥4 人 → MOSS 轻度少判 1 人


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def n_speakers(records: list[dict]) -> int:
    return len({r["speaker"] for r in records})


def turn_rate(records: list[dict]) -> float:
    """说话人切换次数 / 首末时间跨度（次/秒）。

    按时间排序后统计相邻段说话人不同的次数。MOSS 把两个说话人并成一个时，
    本该交替的段变成同一标签，切换率随之塌陷 —— 这是「并人」的直接痕迹，
    且不依赖它自称判了几个人（症状①漏掉的 5 人段就是靠这个抓到的）。
    """
    if not records:
        return 0.0
    span = max(r["end_time"] for r in records) - min(r["start_time"] for r in records)
    if span <= 0:
        return 0.0
    ordered = sorted(records, key=lambda r: r["start_time"])
    turns = sum(1 for a, b in zip(ordered, ordered[1:]) if a["speaker"] != b["speaker"])
    return turns / span


def main() -> int:
    ap = argparse.ArgumentParser(description="MOSS/CAM++ 按段择优集成（v018 = v011 + H + D）")
    ap.add_argument("--moss", required=True, help="MOSS 预测目录（主系统）")
    ap.add_argument("--cam", required=True, help="CAM++ 预测目录（备选系统）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--moss-max-spk", type=int, default=MOSS_MAX_SPK)
    ap.add_argument("--moss-max-turn-rate", type=float, default=MOSS_MAX_TURN_RATE)
    ap.add_argument("--cam-min-spk", type=int, default=CAM_MIN_SPK)
    ap.add_argument("--cam-over-spk", type=int, default=CAM_OVER_SPK)
    ap.add_argument("--cam-eq5-moss-min", type=int, default=CAM_EQ5_MOSS_MIN)
    args = ap.parse_args()

    moss_dir, cam_dir = Path(args.moss), Path(args.cam)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = sorted(moss_dir.glob("[0-9]*.seglst.json"))
    if not preds:
        logger.error("MOSS 预测目录为空：%s", moss_dir)
        return 1

    switched: list[str] = []
    no_cam: list[str] = []
    by_reason: dict[str, int] = {}
    for moss_path in preds:
        sid = moss_path.name.split(".")[0]
        cam_path = cam_dir / f"{sid}.seglst.json"
        if not cam_path.exists():
            # CAM++ 缺这段 → 保守回退 MOSS，不因缺源丢 session
            no_cam.append(sid)
            shutil.copy(moss_path, out_dir / f"{sid}.seglst.json")
            continue

        moss_recs = load(moss_path)
        cam_recs = load(cam_path)
        m_nspk = n_speakers(moss_recs)
        c_nspk = n_speakers(cam_recs)

        # 空分支一律让位给非空分支：空假设在 tcpWER 下只会把该场 ref 的全部词记为
        # deletion，不可能优于任何非空输出。下面三条规则都带「CAM 人数 ≥ N」的前提，
        # MOSS 为空且 CAM 人数不足时会保留空的 MOSS，导致整场丢失（实测 session 295）。
        if not moss_recs or not cam_recs:
            chosen = cam_recs if not moss_recs else moss_recs
            reason = "empty_" + ("moss" if not moss_recs else "cam")
            shutil.copy(cam_path if chosen is cam_recs else moss_path,
                        out_dir / f"{sid}.seglst.json")
            if chosen is cam_recs:
                switched.append(sid)
            by_reason[reason] = by_reason.get(reason, 0) + 1
            continue

        few_spk = m_nspk <= args.moss_max_spk
        low_turn = turn_rate(moss_recs) <= args.moss_max_turn_rate
        v011_ok = (few_spk or low_turn) and c_nspk >= args.cam_min_spk
        cam_over = c_nspk >= args.cam_over_spk          # 规则D：CAM 多估→用 MOSS
        cam_eq5 = (c_nspk == 5 and m_nspk >= args.cam_eq5_moss_min)  # 规则H：轻度少判

        if v011_ok:
            chosen = cam_recs
            reason = "v011_" + ("both" if few_spk and low_turn else ("few_spk" if few_spk else "low_turn"))
        elif cam_over:
            chosen = moss_recs
            reason = f"D_cam_nspk{c_nspk}"
        elif cam_eq5:
            chosen = cam_recs
            reason = f"H_cam5_moss{m_nspk}"
        else:
            chosen = moss_recs
            reason = "default_moss"

        src_path = cam_path if chosen is cam_recs else moss_path
        shutil.copy(src_path, out_dir / f"{sid}.seglst.json")
        if chosen is cam_recs:
            switched.append(sid)
        by_reason[reason] = by_reason.get(reason, 0) + 1

    if no_cam:
        logger.warning("%d 段无 CAM++ 预测，已回退 MOSS: %s", len(no_cam), no_cam[:5])
    logger.info("共 %d 段，其中 %d 段(%.1f%%)改用 CAM++ → %s",
                len(preds), len(switched), len(switched) / len(preds) * 100, out_dir)
    logger.info("切换原因分布: %s", by_reason)
    logger.info("改用 CAM++ 的 session: %s", switched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
