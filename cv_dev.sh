#!/usr/bin/env bash
# dev K 折交叉验证：为「出货配置」造一个 106 段的干净评测集。
#
# 为什么必须做（2026-08-20，v047 线上 0.14995 之后）
#   出货配置（all-106 模型）没有任何干净评测集 —— 它的仿真素材含全部 106 段 dev。
#   此前所有验证都是拿 train-82 模型在 holdout-24 上做的，那是**另一个配置**。
#   累计三个机制家族三次方向相反（v031/032、v046、v047）。
#   每折排除本折 session 再训练 → 该折对该模型就是真·未见过。
#   拼起来 = 106 段干净预测，功效是 24 段的 4.4 倍，且测的是出货配置本身。
#
# 配方与 v040 完全一致：800 条 / 2 epoch / lr 1e-5 / cosine / warmup 0.1 / bf16 / adafactor / accum 4
# 每折跑完即删模型（12G 可用磁盘放不下 4 个 1.8G 模型 + 中间产物）。
# 断点续跑：已有 pred 的折直接跳过。
set -euo pipefail
cd /root/autodl-tmp/ft
PY=/root/autodl-tmp/envs/moss/bin/python
REPO=/root/autodl-tmp/MOSS-Transcribe-Diarize
BASE=/root/autodl-tmp/hf/hub/models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/e8681d68e7042738ffca8ac8212bc8fcb1131ab8
SIM=/root/autodl-tmp/cvroot/baseline/src/simulate_conv.py
PROTO=/root/autodl-tmp/ft/sim_jsonl/train.jsonl
source /root/autodl-tmp/ft/folds.env

for i in "$@"; do
  eval "DROP=\$FOLD$i"
  OUT=cv$i
  if [ -d "$OUT/pred" ] && [ "$(ls $OUT/pred/*.seglst.json 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "== fold$i 已完成（$(ls $OUT/pred/*.seglst.json|wc -l) 段），跳过 =="; continue
  fi
  mkdir -p $OUT
  echo "===== fold$i：排除 $(echo $DROP|tr ',' '\n'|wc -l) 段 ====="

  echo "-- 1/4 仿真素材（排除本折）--"
  $PY $SIM --n 800 --seed 0 --exclude-sessions "$DROP" --out $OUT/sim 2>&1 | tail -3

  echo "-- 2/4 造 jsonl --"
  $PY mk_jsonl.py $OUT/sim/ref.seglst.json "$PWD/$OUT/sim/wav" $OUT/train.jsonl $PROTO

  echo "-- 3/4 微调（v040 配方）--"
  $PY $REPO/finetune.py --train_jsonl $OUT/train.jsonl \
      --model_name_or_path $BASE --output_dir $OUT/model \
      --num_train_epochs 2 --learning_rate 1e-5 --optim adafactor \
      --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 True \
      --gradient_accumulation_steps 4 --per_device_train_batch_size 1 \
      --gradient_checkpointing True --save_strategy no --logging_steps 40 2>&1 | tail -4

  echo "-- 4/4 在本折上推理（模型从没见过这些段）--"
  $PY moss_sat.py --model $OUT/model --processor $BASE --wav-dir /root/autodl-tmp/dev_wav \
      --sessions "$DROP" --out $OUT/pred 2>&1 | tail -3

  echo "-- 删模型与仿真音频腾磁盘 --"
  rm -rf $OUT/model $OUT/sim
  df -h /root/autodl-tmp | tail -1
  echo "FOLD${i}_DONE $(ls $OUT/pred/*.seglst.json 2>/dev/null|wc -l) 段"
done
echo "CV_BATCH_DONE"
