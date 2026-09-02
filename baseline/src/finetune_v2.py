"""带验证集监控 + 早停 + 取 best 的微调脚本，并集成若干 SFT 提分 trick。

## 为什么重写（2026-08-19）

此前所有微调实验都是「固定轮数训完 → 跑下游 tcpWER 看结果」：
**每次评估要 10+ 分钟推理，信号又粗（只有一个终点数），看不到 loss 拐点在哪。**
项目甚至从未建过验证集（`sim_jsonl/holdout.jsonl` 是空文件）。

代价是实打实的 —— 靠这种粗糙方式得出的结论翻过车：

| 事件 | 教训 |
|---|---|
| epoch 扫描（1/2/3）花了 3 次完整训练 + 推理 | 一条 val loss 曲线一次就能看出拐点 |
| AdamW 在 train-82 上 −0.478、全 106 上 +0.00196 | 训练 loss 显示 AdamW **末段 0.2656 vs adafactor 0.1906** —— 是**欠拟合**，因为 `lr=1e-5` 本是给 adafactor 调的 |

## 本脚本提供什么

1. **验证集 loss 监控**：`--eval_jsonl` + `eval_strategy=steps`，曲线落盘 `loss_curve.json`
2. **早停 + 取 best**：`EarlyStoppingCallback(patience)` + `load_best_model_at_end`
   （按 `eval_loss` 选最优步，而非取最后一步）
3. **`save_only_model=True`** —— 只存权重不存优化器状态，checkpoint 从 ~5.5G 降到 ~1.8G
   （AdamW 此前正因 checkpoint 撑爆磁盘在 200/400 崩过一次）

## 集成的 trick 及其依据

| trick | 参数 | 为什么用 |
|---|---|---|
| **NEFTune** | `--neftune_alpha` | embedding 加均匀噪声，小数据 SFT 上被反复验证的正则化；HF 原生支持 |
| **label smoothing** | `--label_smoothing`（默认 0）| 缓解过度自信；⚠️ 本任务输出含时间戳数字，过度平滑可能伤精确匹配，故默认关 |
| **weight decay** | `--weight_decay`（默认 0.01）| 上游用 HF 默认 0.0，对小数据全参微调偏松 |
| **梯度裁剪** | `--max_grad_norm`（默认 1.0）| 有效 batch 仅 4，梯度噪声大 |

⚠️ **纪律：一次只开一个 trick**，否则又是「一次改多个变量、无法归因」
（v032 已犯过，多花一次提交额度才拆开）。默认值刻意贴近现有配方，便于单变量对比。

## 验证集怎么选（关键）

- 训练用 `--source-split train`（train-82）→ 验证集用 **holdout-24** 素材生成，**完全干净**
- 训练用 `--source-split all`（全 106）→ **没有干净验证集**，只能用「同分布、不同 seed 的未见 clip」：
  它能测**对具体样本的过拟合**，但测不出 dev↔test 的分布差异。
  ⚠️ 此时 val loss 只可用于找拐点，**不可用于跨配置比较**。

用法：
    python finetune_v2.py --train_jsonl <train> --eval_jsonl <val> \\
        --model_name_or_path <snap> --output_dir <out> \\
        --learning_rate 3e-5 --optim adamw_torch --patience 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MOSS_REPO = os.environ.get("MOSS_REPO", "/root/autodl-tmp/MOSS-Transcribe-Diarize")


def main() -> int:
    sys.path.insert(0, MOSS_REPO)
    ap = argparse.ArgumentParser(description="带验证集/早停/取 best 的微调")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--eval_jsonl", required=True, help="验证集 jsonl（选择原则见 docstring）")
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_train_epochs", type=float, default=6.0,
                    help="设大一点，让早停决定实际轮数")
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--optim", default="adamw_torch")
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--lr_scheduler_type", default="cosine")
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--patience", type=int, default=3, help="早停耐心值（以 eval_steps 为单位）")
    # ---- trick 开关（一次只开一个）----
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--neftune_alpha", type=float, default=None)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    args = ap.parse_args()

    # 以下依赖只在 GPU 的 moss 环境中存在
    import torch  # type: ignore[import-not-found]  # noqa: PLC0415
    from transformers import (  # type: ignore[import-not-found]  # noqa: PLC0415
        AutoModelForCausalLM,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    from finetune import ConversationDataset, DataCollator  # type: ignore  # noqa: PLC0415
    from moss_transcribe_diarize.processing_moss_transcribe_diarize import (  # type: ignore  # noqa: PLC0415
        MossTranscribeDiarizeProcessor,
    )

    processor = MossTranscribeDiarizeProcessor.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    train_ds = ConversationDataset(args.train_jsonl)
    eval_ds = ConversationDataset(args.eval_jsonl)
    collator = DataCollator(processor, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, dtype=torch.bfloat16
    )
    logger.info("训练 %d 条 / 验证 %d 条 | optim=%s lr=%.1e patience=%d",
                len(train_ds), len(eval_ds), args.optim, args.learning_rate, args.patience)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        optim=args.optim,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        label_smoothing_factor=args.label_smoothing,
        neftune_noise_alpha=args.neftune_alpha,
        bf16=True, gradient_checkpointing=True,
        # ---- 验证集监控 + 早停取 best ----
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=1,
        # 只存权重不存优化器状态：AdamW 的 checkpoint 曾因此撑爆磁盘
        save_only_model=True,
        logging_steps=10, report_to=[],
        # 与上游一致：dataset 产出 audio/prompt 自定义字段，否则会被按 forward 签名过滤掉
        remove_unused_columns=False, label_names=["labels"],
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()

    hist = trainer.state.log_history
    curve = [{"step": h.get("step"),
              "train_loss": h.get("loss"),
              "eval_loss": h.get("eval_loss")}
             for h in hist if "loss" in h or "eval_loss" in h]
    with open(os.path.join(args.output_dir, "loss_curve.json"), "w", encoding="utf-8") as fh:
        json.dump(curve, fh, ensure_ascii=False, indent=1)

    evals = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    if evals:
        best = min(evals, key=lambda x: x[1])
        logger.info("验证集 loss：%s", " ".join(f"{s}:{v:.4f}" for s, v in evals))
        logger.info("**最优点 step=%d eval_loss=%.4f**（已按此取 best 权重）", best[0], best[1])

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info("FT_V2_DONE -> %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
