"""LoRA 微调 MOSS-Transcribe-Diarize，作为全参微调的对照。

## 为什么要试（2026-08-18，用户提出）

**直觉是对的**：800 条仿真样本调 0.9B 全参数，参数量远超数据量，过参数化严重。
LoRA 把更新约束到低秩子空间，是天然的正则化。

**但项目里有两条实测证据，方向都不支持「过拟合是当前的绑定约束」**：

| 证据 | 数据 | 读法 |
|---|---|---|
| epoch 曲线呈 **U 形** | 1ep 14.9784% / **2ep 14.7280%** / 3ep 14.8190% | 若严重过拟合，1ep 应优于 2ep；实际 1ep 差 0.25 点 → **仍在欠拟合一侧** |
| 基座插值（0.75 微调 + 0.25 基座）**中性** | 单独 +0.00041，配归属改写 −0.00002 | 若灾难性遗忘严重，插值应有明显收益 |

→ LoRA 的两个主要卖点（抗过拟合、抗遗忘）**在这里都没打到痛点**。
但推断是间接的，且 LoRA 另有好处（优化更好条件化、优化器状态极小、训练更快），
**值得一次实验证伪或证实**。

## 与上游 `finetune.py` 的关系

不改动模型仓库自带脚本，只在其之上包一层：**直接复用它的 `ConversationDataset`
与 `DataCollator`**（含已核实正确的 label 掩码 —— `labels[:, :len(prompt_ids)] = -100`，
只在 assistant 段算 loss），把模型换成 `get_peft_model()` 包装的版本。
训练完 `merge_and_unload()` 合并回完整权重并保存，使下游 `moss_sat.py` **无需任何改动**。

## 超参选择与一个诚实的保留

`r=16 / alpha=32 / dropout=0.05`，target = 注意力四投影 + MLP 三投影
（只调注意力常常表达力不足，而本任务要改的是「分段与说话人判断」这类结构性行为）。

LoRA 的 LR 通常要比全参高 1~2 个数量级（更新被低秩约束），故默认 `1e-4`（全参用 1e-5）。
⚠️ **这意味着本实验不是严格单变量**（同时换了适配方式与 LR）——
若结果为负，需再跑一档 `LR=1e-5` 才能归因到底是 LoRA 本身还是 LR。

用法（GPU 上）：
    python finetune_lora.py --train_jsonl <jsonl> --model_name_or_path <snap> \\
        --output_dir <out> --num_train_epochs 2 --learning_rate 1e-4
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 模型仓库路径（含 finetune.py 与自定义 modeling），可用环境变量覆盖
MOSS_REPO = os.environ.get("MOSS_REPO", "/root/autodl-tmp/MOSS-Transcribe-Diarize")

LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
TARGET_MODULES = os.environ.get(
    "LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
).split(",")


def main() -> int:
    sys.path.insert(0, MOSS_REPO)
    ap = argparse.ArgumentParser(description="LoRA 微调 MOSS（全参微调的对照）")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_train_epochs", type=float, default=2.0)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--lr_scheduler_type", default="cosine")
    args = ap.parse_args()

    # 以下依赖只在 GPU 的 moss 环境中存在，本地无需安装（与仓内其它 GPU 脚本一致）
    import torch  # type: ignore[import-not-found]  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]  # noqa: PLC0415
    from transformers import (  # type: ignore[import-not-found]  # noqa: PLC0415
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
    )

    from finetune import ConversationDataset, DataCollator  # type: ignore  # noqa: PLC0415
    # 路径与上游 finetune.py 第 22 行一致（processor 在包内，非顶层模块）
    from moss_transcribe_diarize.processing_moss_transcribe_diarize import (  # type: ignore  # noqa: PLC0415
        MossTranscribeDiarizeProcessor,
    )

    processor = MossTranscribeDiarizeProcessor.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    dataset = ConversationDataset(args.train_jsonl)
    collator = DataCollator(processor, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, dtype=torch.bfloat16
    )

    cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("LoRA r=%d alpha=%d | 可训练 %.2fM / %.2fM = %.3f%% | LR=%.1e",
                LORA_R, LORA_ALPHA, trainable / 1e6, total / 1e6,
                100 * trainable / total, args.learning_rate)

    targs = TrainingArguments(
        output_dir=args.output_dir + "_ckpt",
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True, gradient_checkpointing=True,
        logging_steps=1, save_strategy="no", report_to=[],
        # 必须与上游 finetune.py 一致：dataset 产出的 audio/prompt 是自定义字段，
        # Trainer 默认会按模型 forward 签名过滤掉它们（peft 包装后签名变化，报 KeyError: 'audio'）
        remove_unused_columns=False,
        label_names=["labels"],
    )
    Trainer(model=model, args=targs, train_dataset=dataset,
            data_collator=collator).train()

    logger.info("合并 LoRA 权重回完整模型（下游 moss_sat.py 无需改动）")
    merged = model.merge_and_unload()
    merged.to(torch.bfloat16).save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info("LORA_FINETUNE_DONE -> %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
