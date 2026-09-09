#!/usr/bin/env bash
# Step 1-9：从原始音频重跑重阶段。由 code/test.sh --full 调用，不单独使用。
#
#   bash code/test/run_stage1_9.sh <test wav 目录>
#
# 这一段需要 GPU 与三份开源权重，整机约 5 小时（RTX 3090 24 GB）：
#   MOSS-Transcribe-Diarize  微调产物，见 code/train.sh；包内 user_data/model_data/ 已预置
#   FireRedASR2-AED          环境变量 FIRERED_SRC / FIRERED_CKPT 指向源码与权重目录
#   CAM++ / fun-asr          首次运行自动从 ModelScope 下载
#
# 默认模式（code/test.sh 不加 --full）直接复用 user_data/tmp_data/stage_outputs/ 里
# 已归档的同名产物，不跑这个脚本，也不需要 GPU。

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."     # → code/

WAV="${1:?用法：bash code/test/run_stage1_9.sh <test wav 目录>}"
UD=../user_data
STAGE=$UD/tmp_data/stage_outputs
PY="${PYTHON:-python3}"
MODEL="${MOSS_MODEL:-$UD/model_data/moss_sim_all106}"

mkdir -p "$STAGE"
say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "Step 1  CAM++ 谱聚类分离 + 段级声纹（约 7 min）"
PYTHONPATH=src "$PY" src/diarize.py --wav-dir "$WAV" \
    --out "$STAGE/diar_test_m3_7_0.70" \
    --min-spk 3 --max-spk 7 --merge-thr 0.70 \
    --emb-cache "$STAGE/emb_test"

say "Step 2  MOSS-SAT 解码（约 2 h，GPU）"
PYTHONPATH=src "$PY" src/moss_sat.py --model "$MODEL" \
    --wav-dir "$WAV" --out "$STAGE/test_raw" --max-new-tokens 2048

say "Step 3  FireRedASR2 按 MOSS 切分重转写（约 1 h，GPU）"
PYTHONPATH=src "$PY" src/fr_retext.py "$STAGE/test_raw" "$WAV" "$STAGE/output_test_fr"

say "Step 4  词级声学仲裁（MOSS 分支，约 1 h）"
MARGIN=0.05 MAX_BLOCKS=6 PYTHONPATH=src "$PY" src/word_arb.py \
    "$STAGE/test_raw" "$STAGE/output_test_fr" "$WAV" "$STAGE/test_wa_m05b6"

say "Step 5  归属改写：段内多窗声纹一致性（约 25 min）"
COH_MIN=0.55 MARGIN=0.05 PYTHONPATH=src "$PY" src/spk_reassign.py \
    "$STAGE/test_wa_m05b6" "$WAV" "$STAGE/test_v048_re"

say "Step 7  fun-asr 转写（开 diarization）"
# 两条等价路径，产物格式相同，取决于部署方式：
#   本地开源权重：src/funasr_local.py（默认，不依赖任何外部服务）
#   自有托管实例：src/run.py（异步接口，需 .env 里的凭据）
PYTHONPATH=src "$PY" src/funasr_local.py --wav-dir "$WAV" --out "$STAGE/test_cam_multi"

say "Step 8  用 CAM++ 声纹为 fun-asr 分段重打标签（约 20 min）"
PYTHONPATH=src "$PY" src/spk_relabel.py \
    "$STAGE/test_cam_multi" "$WAV" "$STAGE/test_cam_re"

say "Step 8.5  FireRed 按 CAM 分支切分重转写（Step 9 的输入）"
PYTHONPATH=src "$PY" src/fr_retext.py "$STAGE/test_cam_re" "$WAV" "$STAGE/_test_camre_fr"

say "Step 9  词级声学仲裁（CAM 分支，配方同 Step 4，约 40 min）"
MARGIN=0.05 MAX_BLOCKS=6 PYTHONPATH=src "$PY" src/word_arb.py \
    "$STAGE/test_cam_re" "$STAGE/_test_camre_fr" "$WAV" "$STAGE/test_cam_wa"

say "Step 1-9 完成 → $STAGE"
