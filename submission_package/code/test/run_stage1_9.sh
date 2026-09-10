#!/usr/bin/env bash
# Step 1-9：从原始音频重跑重阶段。由 code/test.sh --full 调用，不单独使用。
#
#   bash code/test/run_stage1_9.sh <test wav 目录> [输出目录]
#
# 输出目录默认 user_data/tmp_data/stage_outputs_rerun/，**不覆盖**随包归档的
# stage_outputs/ —— 这样重跑失败或中途中断都不会污染归档，两份还能直接对比。
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

WAV_IN="${1:?用法：bash code/test/run_stage1_9.sh <test wav 目录> [输出目录]}"
UD=../user_data
STAGE_IN="${2:-$UD/tmp_data/stage_outputs_rerun}"
PY="${PYTHON:-python3}"

# 入参先展开成绝对路径再往下传。原因：build_spk_probe_jsonl.py 会把 --wav-dir
# 原样拼进 jsonl（`f"{wav_dir}/{sid}.wav"`），下游 spk_ac_probe.py 再打开它；
# 传相对路径时两者的解析基准不一定相同，实测报过 LibsndfileError。
# 脚本本身不含任何硬编码绝对路径 —— 这里只是把相对入参在运行时展开。
[ -d "$WAV_IN" ] || { echo "✗ 找不到音频目录：$WAV_IN"; exit 1; }
WAV="$(cd "$WAV_IN" && pwd)"
mkdir -p "$STAGE_IN"
STAGE="$(cd "$STAGE_IN" && pwd)"
MODEL="${MOSS_MODEL:-$UD/model_data/moss_sim_all106}"
# HF Trainer 存的 checkpoint 只带 tokenizer，对它调 AutoProcessor 会静默退化成
# Qwen2Tokenizer（无 feature_extractor），解码直接产出 0 条。所以 processor 必须
# 单独指向基座——包内 moss_base_processor/ 就是不含权重的那一份。
PROCESSOR="${MOSS_PROCESSOR:-$UD/model_data/moss_base_processor}"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
say "输入 $WAV"
say "输出 $STAGE"

say "Step 1  CAM++ 谱聚类分离 + 段级声纹（约 7 min）"
PYTHONPATH=src "$PY" src/diarize.py --wav-dir "$WAV" \
    --out "$STAGE/diar_test_m3_7_0.70" \
    --min-spk 3 --max-spk 7 --merge-thr 0.70 \
    --emb-cache "$STAGE/emb_test"

say "Step 2  MOSS-SAT 解码（约 2 h，GPU）"
[ -f "$PROCESSOR/preprocessor_config.json" ] || {
    echo "✗ 找不到 processor：$PROCESSOR"
    echo "  它是不含权重的基座副本（约 16 MB），随包放在 user_data/model_data/moss_base_processor/。"
    echo "  也可用 MOSS_PROCESSOR=<基座快照目录> 指向别处。"
    exit 1
}
PYTHONPATH=src "$PY" src/moss_sat.py --model "$MODEL" --processor "$PROCESSOR" \
    --wav-dir "$WAV" --out "$STAGE/test_raw" --max-new-tokens 2048

say "Step 3  FireRedASR2 按 MOSS 切分重转写（约 1 h，GPU）"
PYTHONPATH=src "$PY" src/fr_retext.py "$STAGE/test_raw" "$WAV" "$STAGE/output_test_fr"

say "Step 4  词级声学仲裁（MOSS 分支，约 1 h）"
MARGIN=0.05 MAX_BLOCKS=6 PYTHONPATH=src "$PY" src/word_arb.py \
    "$STAGE/test_raw" "$STAGE/output_test_fr" "$WAV" "$STAGE/test_wa_m05b6"

say "Step 5  归属改写：段内多窗声纹一致性（约 25 min）"
COH_MIN=0.55 MARGIN=0.05 PYTHONPATH=src "$PY" src/spk_reassign.py \
    "$STAGE/test_wa_m05b6" "$WAV" "$STAGE/test_v048_re"

say "Step 5.5  MOSS 说话人似然打分（Step 6 的输入，约 30 min，GPU）"
# 归档包里带的是 artifacts/test_spk_scores2.json，它绑定在归档的 test_v048_re 上。
# 重跑后 test_v048_re 变了，这份打分必须跟着重算，否则 Step 6 用的是过期分数。
# 这一步只用到 tokenizer，指 processor 目录即可（不必加载 1.8 G 权重）
PYTHONPATH=src "$PY" src/build_spk_probe_jsonl.py \
    --pred-dir "$STAGE/test_v048_re" --wav-dir "$WAV" \
    --model "$PROCESSOR" --out "$STAGE/spk_probe.jsonl"
PYTHONPATH=src "$PY" src/spk_ac_probe.py \
    --jsonl "$STAGE/spk_probe.jsonl" --model "$MODEL" \
    --out "$STAGE/test_spk_scores2.json"

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
