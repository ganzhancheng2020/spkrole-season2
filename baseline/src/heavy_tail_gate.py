"""提交前强制关卡：剔除 top-N 收益 session 后，dev 净收益还剩多少。

## 为什么存在（v027 用一次线上机会换来的）

v027 dev 降 0.303 点、train/holdout 双切分同向、holdout 改善是 train 的 3.3 倍 —— 全绿，
**线上仍倒退 +0.00367**（迁移率 −121%）。事后拆开：净省 59 个错误里**前 3 段占 130.5%**，
剔除后是 **−0.093 点**；holdout 的 −0.660 实为 **068 单段**贡献（其余 23 段合计净亏）。

**双切分验收与本关卡是两道正交的关卡：**

| 关卡 | 抓什么 | v027 |
|---|---|---|
| train/holdout 双切分 | **参数过拟合**（train 调参、holdout 失效）| ✅ 通过 |
| **本关卡（top-N 重尾折扣）** | **收益集中在少数幸运 session** | ❌ −0.093，本应拦下 |

dev 只有 106 段，test 有 394 段。**任何「靠几段大赢」的改动都不可外推。**

## 判据

**剔 top-3 后的净收益 <= 0 → 不提交。**

### 关卡自校验（四对已知线上结局，4/4 判对）

| 版本对 | 剔top3 | 关卡 | 实际 |
|---|---|---|---|
| v011→v020 | **-0.015** | FAIL | 线上 0.16352，**未打赢当时最佳 v018** ✅ |
| v020→v024 | **+0.350** | PASS | 线上 0.15903，新最佳 ✅ |
| v024→v026 | **+0.087** | PASS | 线上 0.15756，新最佳 ✅ |
| v026→v027 | **-0.093** | FAIL | 线上 0.16123，**倒退** ✅ |

对通过的两版，本关卡系统性**低估**线上收益（0.350→0.449、0.087→0.147，约 1.3-1.7 倍）
—— 是个保守的预测器，不会把好版本挡在门外。

## ⚠️ 必须用净口径，不能用毛口径

只统计「变好的 session」得到的是**毛收益**，会系统性偏高：
2026-08-09 首次审查时用毛口径把归属轴空间报成 2.808 点，**净口径实为 2.175 点，虚高 23%**。
本脚本同时输出毛收益、反向损失与净收益，并**以净收益为准**。

用法：
    cd baseline
    # 两个目录都需先跑过 batch_evaluate.py（才有 all.hyp.seglst_tcpwer_per_reco.json）
    .venv/bin/python src/heavy_tail_gate.py output/v026_pick output/v027_dev
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PER_RECO = "all.hyp.seglst_tcpwer_per_reco.json"
# 判据阈值：剔除这么多个收益最大的 session 后，净收益仍须 > 0
GATE_TOP_N = 3


def load(d: str) -> dict[str, dict]:
    p = Path(d) / PER_RECO
    if not p.exists():
        msg = f"缺少 {p} —— 先对该目录跑 batch_evaluate.py"
        raise FileNotFoundError(msg)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    if len(sys.argv) < 3:
        logger.error("用法见模块 docstring：%s", __doc__)
        return 2
    base_dir, new_dir = sys.argv[1:3]
    a, b = load(base_dir), load(new_dir)
    ids = sorted(set(a) & set(b))
    if not ids:
        logger.error("两个目录没有共同 session")
        return 1

    total_words = sum(a[i]["length"] for i in ids)
    net = sum(a[i]["errors"] - b[i]["errors"] for i in ids)
    wins = sorted(
        ((a[i]["errors"] - b[i]["errors"], i) for i in ids if a[i]["errors"] > b[i]["errors"]),
        reverse=True,
    )
    gross = sum(v for v, _ in wins)
    damage = sum(b[i]["errors"] - a[i]["errors"] for i in ids if b[i]["errors"] > a[i]["errors"])

    def pts(errors: int) -> float:
        return errors / total_words * 100

    logger.info("基准 %s → 新版 %s（%d 个共同 session）", base_dir, new_dir, len(ids))
    logger.info("毛收益 %d / 反向损失 %d / **净 %d 个错误 = %+.3f 点**",
                gross, damage, net, pts(net))
    logger.info("改善 %d 个 session，恶化 %d 个",
                len(wins), sum(1 for i in ids if b[i]["errors"] > a[i]["errors"]))

    for n in (1, 3, 5, 10):
        remain = net - sum(v for v, _ in wins[:n])
        share = sum(v for v, _ in wins[:n]) / net * 100 if net else 0.0
        logger.info("  剔 top%-2d：前 %d 段占净收益 %6.1f%% → 剩 %+.3f 点", n, n, share, pts(remain))

    if wins:
        logger.info("收益前 5 段：%s", [(i, f"-{v}") for v, i in wins[:5]])

    remain3 = pts(net - sum(v for v, _ in wins[:GATE_TOP_N]))
    if remain3 > 0:
        logger.info("✅ PASS：剔 top%d 后仍有 %+.3f 点，可提交", GATE_TOP_N, remain3)
        return 0
    logger.error("❌ FAIL：剔 top%d 后只剩 %+.3f 点 —— **不要提交**（v027 就是这么倒退的）",
                 GATE_TOP_N, remain3)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
