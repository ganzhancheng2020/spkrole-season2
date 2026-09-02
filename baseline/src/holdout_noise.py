"""量评测集的噪声下限：多小的差异算噪声？两版差异是否可分辨？

## 为什么需要它（v032 用一次线上额度换来的）

2026-08-12 发现一个被误判的前提。此前从两个数据点得出「holdout-24 无预测力」：

| 对比 | Δholdout-24 | Δ线上 | |
|---|---|---|---|
| v031 → v033（**只改基座**）| +0.137 | +0.654 | **方向一致** ✅ |
| v031 → v032（基座 + **阈值**）| −0.501 | +0.783 | 方向相反 ❌ |

区别只有一处：**v032 的阈值 `camMin4` 就是在 holdout-24 上扫出来的**。
→ 真正的诊断不是「24 段太小」，是**同时把 holdout-24 当调参集和验收集**（评测集污染）。

但仍有一个问题没答：**0.137 点这种量级，算信号还是噪声？** 本脚本回答它。

## 方法

**按 session 自助重采样**（bootstrap）。合法性依据：本项目已实证
**tcpWER 的逐 session 误差可加**（查表复现 3221 = 实测 3221），
故对 session 重采样再按 `sum(errors)/sum(length)` 聚合是有效的。

两个输出：

1. **边际噪声** —— 单版 tcpWER 估计量的标准差，即「这个评测集的分辨率」
2. **配对差异**（更有用）—— 同一批重采样上算 `Δ = A − B` 的分布与 95% 区间。
   **区间含 0 → 该差异不可分辨，不得作为提交依据。**

配对比边际敏感得多：两版在同一批 session 上比，共同的 session 难度被抵消。

用法：
    cd baseline
    .venv/bin/python src/holdout_noise.py output/simft_pick08_norm output/sim3k_final
    .venv/bin/python src/holdout_noise.py output/simft_pick08_norm      # 只看单版噪声
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PER_RECO = "all.hyp.seglst_tcpwer_per_reco.json"
N_BOOT = 10000
SEED = 0


def load(d: str) -> dict[str, dict]:
    p = Path(d) / PER_RECO
    if not p.exists():
        msg = f"缺少 {p} —— 先对该目录跑 batch_evaluate.py"
        raise FileNotFoundError(msg)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def boot_curve(src: dict[str, dict], ids: list[str], draws: np.ndarray) -> np.ndarray:
    """每次重采样的 tcpWER（百分比）。"""
    err = np.array([src[i]["errors"] for i in ids], dtype=float)
    length = np.array([src[i]["length"] for i in ids], dtype=float)
    return err[draws].sum(axis=1) / length[draws].sum(axis=1) * 100


def point(src: dict[str, dict], ids: list[str]) -> float:
    return sum(src[i]["errors"] for i in ids) / sum(src[i]["length"] for i in ids) * 100


def main() -> int:
    if len(sys.argv) < 2:
        logger.error("用法见模块 docstring：%s", __doc__)
        return 2
    dirs = sys.argv[1:3]
    srcs = [load(d) for d in dirs]
    ids = sorted(set.intersection(*[set(s) for s in srcs]))
    if not ids:
        logger.error("没有共同 session")
        return 1

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))

    logger.info("评测集：%d 个 session / %d 词，自助重采样 %d 次",
                len(ids), sum(srcs[0][i]["length"] for i in ids), N_BOOT)

    curves = []
    for d, src in zip(dirs, srcs):
        c = boot_curve(src, ids, draws)
        curves.append(c)
        logger.info("%-24s 点估计 %.3f%% | 自助标准差 **%.3f 点** | 95%% 区间 [%.3f, %.3f]",
                    Path(d).name, point(src, ids), c.std(ddof=1),
                    np.percentile(c, 2.5), np.percentile(c, 97.5))

    if len(curves) == 2:
        diff = curves[0] - curves[1]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        pd_ = point(srcs[0], ids) - point(srcs[1], ids)
        logger.info("=== 配对差异 A − B（同一批重采样，抵消共同难度）===")
        logger.info("  点估计 %+.3f 点 | 95%% 区间 [%+.3f, %+.3f] | 标准差 %.3f",
                    pd_, lo, hi, diff.std(ddof=1))
        logger.info("  P(A 优于 B) = %.1f%%", (diff < 0).mean() * 100)
        if lo <= 0 <= hi:
            logger.error("  ❌ **区间含 0 → 差异不可分辨，不得作为提交依据**")
        else:
            logger.info("  ✅ 区间不含 0 → 差异可分辨")
        logger.info("  **判据线**：本评测集上，小于约 %.2f 点的差异一律当噪声。",
                    1.96 * diff.std(ddof=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
