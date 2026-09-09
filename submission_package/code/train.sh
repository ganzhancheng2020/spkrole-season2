#!/usr/bin/env bash
# 训练入口 —— 代码审核规范 §4b + §附注「train.sh 训练示例脚本，必选」。
#
# 训练的是链路里唯一一个我们自己训的模型：MOSS-Transcribe-Diarize 的说话人归属转写微调。
# 其余三个模型（CAM++ / FireRedASR2-AED / fun-asr）全部直接用开源权重，不训练。
#
#   bash code/train.sh
#
# 三步，全部超参已按规范固定在下方，不接受命令行覆盖：
#   1) 数据增广 —— 把官方 dev 的单说话人片段跨 session 重组成 800 条仿真多人对话
#                  （固定 seed=0，逐字节可复现；test 全程不参与，赛题 §103/§148）
#   2) 转 JSONL —— SegLST → MOSS 微调所需的 conversation 格式
#   3) 微调     —— HF Trainer，2 epoch / lr 1e-5 / adafactor / bf16
#
# 产物：user_data/model_data/moss_sim_all106/（即 code/test.sh --full 里 Step 2 用的模型）
# 单卡 RTX 3090 24 GB 约 6 小时。

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"     # 全程以 code/ 为工作目录，路径一律相对

XF=../xfdata
UD=../user_data
PY="${PYTHON:-python3}"

DEV_ROOT=$XF/dev/dev                              # 官方 dev：ref.seglst.json + wav/
SIM_DIR=$UD/tmp_data/sim_all106                   # 增广数据（标注结果 + 音频）
JSONL_DIR=$UD/tmp_data/train_jsonl
MODEL_OUT=$UD/model_data/moss_sim_all106          # 微调产物

# 基座：Apache-2.0，公开可下。未预置时从 HuggingFace 拉取。
BASE="${MOSS_BASE:-OpenMOSS-Team/MOSS-Transcribe-Diarize}"

# ---- 固定超参（规范 §4b「请固定训练时的超参」）--------------------------------
SIM_N=800                 # 仿真对话条数
SIM_SEED=0                # 随机种子 —— 固定它才能逐字节重生成同一批数据
SIM_SOURCE=all            # 素材来源：全部 106 场 dev
MAX_LENGTH=8192
EPOCHS=2
LR=1e-5
BATCH_SIZE=1              # per-device
GRAD_ACCUM=4              # 等效 batch = 4
LR_SCHEDULER=cosine
WARMUP_RATIO=0.1
OPTIM=adafactor
# ------------------------------------------------------------------------------

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

[ -f "$DEV_ROOT/ref.seglst.json" ] || {
    echo "✗ 找不到 $DEV_ROOT/ref.seglst.json"
    echo "  请先把官方 dev 数据放到 xfdata/dev/dev/（含 ref.seglst.json 与 wav/）"
    exit 1
}

say "1/3  数据增广：dev 重组 → $SIM_N 条仿真多人对话（seed=$SIM_SEED）"
mkdir -p "$(dirname "$SIM_DIR")"
PYTHONPATH=src "$PY" src/simulate_conv.py \
    --n "$SIM_N" --seed "$SIM_SEED" --source-split "$SIM_SOURCE" \
    --dev-root "$DEV_ROOT" --out "$SIM_DIR"
echo "    → $SIM_DIR（$(ls -1 "$SIM_DIR/wav"/*.wav 2>/dev/null | wc -l | tr -d ' ') 个 wav + ref.seglst.json）"

say "2/3  转 JSONL（SegLST → MOSS conversation 格式）"
mkdir -p "$JSONL_DIR"
"$PY" train/mk_jsonl.py \
    "$SIM_DIR/ref.seglst.json" "$(cd "$SIM_DIR" && pwd)/wav" \
    "$JSONL_DIR/train.jsonl"

say "3/3  微调 MOSS-Transcribe-Diarize（$EPOCHS epoch，lr=$LR，$OPTIM，bf16）"
mkdir -p "$MODEL_OUT"
"$PY" train/finetune.py \
    --train_jsonl "$JSONL_DIR/train.jsonl" \
    --model_name_or_path "$BASE" \
    --max_length "$MAX_LENGTH" \
    --output_dir "$MODEL_OUT" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --lr_scheduler_type "$LR_SCHEDULER" \
    --warmup_ratio "$WARMUP_RATIO" \
    --bf16 \
    --gradient_checkpointing \
    --optim "$OPTIM" \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --report_to none

say "训练完成 → $MODEL_OUT"
echo "接下来可用 bash code/test.sh --full 走完整推理链路。"
