"""Model soup：把同一基座、同一配方微调出的多个 checkpoint 做权重平均。

与已封死的「输出层集成」是**不同机制** —— 后者被 text-segmentation-coupling 封死
（跨系统搬文本会撞分割边界），而权重平均不产生新的分割，只挪同一个模型的参数。

依据：成员都从同一 base 出发、同配方微调，权重仍在同一 loss 盆地，
均匀/加权平均在 OOD 上通常不劣于最优单模（Wortsman et al., model soups）。
本赛正是 OOD：训练在仿真、验收在真实。

用法： python soup.py <out_dir> <ckpt_a>[:w] <ckpt_b>[:w] [...]

权重可选，缺省等权。⚠️ **权重必须按先验定**（例如按各成员已知的线上分），
不许在 holdout 上扫 —— 那是 v032 翻车的同一个坑（评测集既当调参集又当验收集）。

本脚本只在 GPU 机器上跑（torch/safetensors 不在本仓库任一 venv 里）。
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import torch  # type: ignore[import-not-found]
from safetensors.torch import load_file, save_file  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    if len(sys.argv) < 3:
        logger.error("用法见模块 docstring：%s", __doc__)
        return 2

    out = Path(sys.argv[1])
    spec = [a.rsplit(":", 1) if ":" in a else [a, "1"] for a in sys.argv[2:]]
    srcs = [Path(a) for a, _ in spec]
    raw = [float(w) for _, w in spec]
    ws = [w / sum(raw) for w in raw]

    out.mkdir(parents=True, exist_ok=True)
    # 配置/分词器等非权重文件从第一个源原样拷贝
    for f in srcs[0].iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name != "optimizer.pt":
            shutil.copy(f, out / f.name)

    acc: dict[str, torch.Tensor] = {}
    for src, w in zip(srcs, ws):
        sd = load_file(src / "model.safetensors")
        if not acc:
            acc = {k: v.to(torch.float32) * w for k, v in sd.items()}
        else:
            if set(acc) != set(sd):
                msg = f"{src} 的参数名与首个 checkpoint 不一致，不能平均"
                raise ValueError(msg)
            for k in acc:
                acc[k] += sd[k].to(torch.float32) * w
        logger.info("已累加 %s 权重 %.3f", src.name, w)

    save_file({k: v.to(torch.bfloat16) for k, v in acc.items()}, out / "model.safetensors")
    logger.info("soup 完成：%d 个 checkpoint 加权平均 %s -> %s",
                len(srcs), [round(w, 3) for w in ws], out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
