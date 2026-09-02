"""按学习到的路由规则在 MOSS / CAM++ 两分支间择源（取代 v018 手调规则级联）。

## 为什么换掉手调规则（2026-08-19）

在**干净的 holdout-24**（模型素材仅 train-82，这 24 段从未进过训练）上做路由轴的
可达性探针，结果推翻了现行规则的价值：

| 口径 | tcpWER |
|---|---|
| 全用 MOSS | 16.435% |
| 全用 CAM | 14.865% |
| **v018 手调规则 pick** | **14.842%** |
| **oracle 路由**（每段取更优分支） | **12.474%** |

→ 路由轴有 **2.390 点**空间，而手调规则只捕获 **1.0%** —— 它基本什么也没做。
根因是阈值当年是在 holdout 上挑的（`logs/2026-08-10.md`：「阈值在 holdout-24 上调，
而它是唯一评测集 —— 循环」），本身就是泄漏拟合出来的。

## 学出来的规则

数据 = 14 个干净模型 × 24 段 = 336 样本（同段被 14 个模型各跑一遍）。
标签 = 哪个分支 tcpWER 更低；样本权重 = |err_moss − err_cam|（代价敏感 —— 目标是降
tcpWER，不是猜对次数）。评估一律用**兑现的 tcpWER**，不看 accuracy/AUC。

逐特征子集 × 学习器扫描后，**最简的单特征模型也是最好的**：

    用 CAM  ⟺  d_nspk = nspk(MOSS) − nspk(CAM) <= 0

逐取值的真实代价把机制摊开了，且**单调**：

| d_nspk | 样本 | MOSS 错 | CAM 错 | 该选 |
|---|---|---|---|---|
| −1 | 53 | 2508 | **960** | CAM（差 1548）|
| 0 | 219 | 5451 | **5130** | CAM（差 321）|
| +1 | 63 | **2289** | 3021 | MOSS（差 732）|
| +2 | 1 | **22** | 31 | MOSS |

**这正是 `pick_ensemble.py` 开篇写的病根**：「MOSS 的病根是少判说话人」。
手调规则也用 nspk，但用的是 `≤2 且 CAM≥3` 这类硬阈值；学出来的是**连续差值上的单一分界**。

## 验证强度（诚实版）

- **双重留出**（留出模型 + 留出 session，模拟「未见模型跑未见段」的真实部署）：
  捕获 **54.0%**，14/14 个模型全部改善。
  ⚠️ 用全部 28 个特征时**只有 −9.6%** —— 绝对量（m_nseg 等）随模型漂移、不迁移；
  d_nspk 是跨分支相对量，才活下来。这条差异是本规则只用单特征的**理由**。
- 逐模型：14/14 优于全 CAM，也 14/14 优于全 MOSS。
- 留一 session 刀切：**24/24 仍为收益**（最坏 −0.447 点）。
- 对 session 自助重采样 95% 区间 **[−3.394, +0.559]，含 0，未达统计显著**；
  P(更好) = 88.3%。⚠️ 采信理由与当年 v011 相同（那次区间 [−3.06, +1.34] 也含 0）：
  机制吻合已知病根 + 单调 + 分布一致，不是单个数字。
- 触发率漂移检查：holdout-24 **79.2%** vs test **77.2%**（v040 基座）—— 一致。

**口径缺口已实测为零**：训练数据取自 word-arb 阶段（`ho_*_wa`），部署在归属改写之后
（`*_moss_re`）。曾担心两阶段不可比，实测 test 394 段上**两阶段的路由决策完全一致
（0 段不同，均 304/394 → CAM）** —— 因为 `spk_reassign` 只在既有说话人之间重贴标签、
不增删说话人，**d_nspk 对该阶段不变**。故此缺口不存在。

用法：
    python src/pick_router.py --moss <moss_dir> --cam <cam_dir> --out <out_dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 学到的分界：MOSS 判的人数不多于 CAM 时，MOSS 在漏人 → 改用 CAM
D_NSPK_MAX_FOR_CAM = 0


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def n_speakers(records: list[dict]) -> int:
    return len({r["speaker"] for r in records})


def main() -> int:
    ap = argparse.ArgumentParser(description="学习路由：按 d_nspk 在 MOSS/CAM 间择源")
    ap.add_argument("--moss", required=True, help="MOSS 预测目录（主系统）")
    ap.add_argument("--cam", required=True, help="CAM++ 预测目录（备选系统）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--d-nspk-max", type=int, default=D_NSPK_MAX_FOR_CAM,
                    help="d_nspk <= 此值则用 CAM（默认 0，学习所得）")
    args = ap.parse_args()

    moss_dir, cam_dir, out_dir = Path(args.moss), Path(args.cam), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = sorted(moss_dir.glob("[0-9]*.seglst.json"))
    if not preds:
        logger.error("MOSS 预测目录为空：%s", moss_dir)
        return 1

    switched: list[str] = []
    no_cam: list[str] = []
    dist: dict[int, int] = {}
    for moss_path in preds:
        sid = moss_path.name.split(".")[0]
        cam_path = cam_dir / f"{sid}.seglst.json"
        if not cam_path.exists():
            # CAM++ 缺这段 → 保守回退 MOSS，不因缺源丢 session（与 pick_ensemble 一致）
            no_cam.append(sid)
            shutil.copy(moss_path, out_dir / f"{sid}.seglst.json")
            continue

        moss_recs, cam_recs = load(moss_path), load(cam_path)
        d_nspk = n_speakers(moss_recs) - n_speakers(cam_recs)
        dist[d_nspk] = dist.get(d_nspk, 0) + 1

        use_cam = d_nspk <= args.d_nspk_max
        if use_cam:
            switched.append(sid)
        with open(out_dir / f"{sid}.seglst.json", "w", encoding="utf-8") as fh:
            json.dump(cam_recs if use_cam else moss_recs, fh, ensure_ascii=False, indent=1)

    n = len(preds)
    logger.info("PICK_ROUTER 共 %d 段：改用 CAM %d 段（%.1f%%），保留 MOSS %d 段",
                n, len(switched), 100 * len(switched) / n, n - len(switched))
    logger.info("d_nspk 分布：%s", dict(sorted(dist.items())))
    if no_cam:
        logger.warning("CAM 缺失回退 MOSS 的 session：%s", no_cam[:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
