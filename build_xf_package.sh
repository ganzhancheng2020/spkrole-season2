#!/usr/bin/env bash
# 按《科大讯飞-代码审核规范》组装提交包。构建器本身不进包。
#
#   bash build_xf_package.sh <输出目录> [--model <权重目录>] [--sim <仿真数据目录>]
#
# 例：
#   bash build_xf_package.sh /tmp/pkg                       # 只装代码+归档产物（本机）
#   bash build_xf_package.sh /root/pkg \                    # 连权重和 800 条仿真一起装（GPU）
#       --model /root/autodl-tmp/ft/sim_out_all106 \
#       --sim   /root/autodl-tmp/ft/sim_all106
#
# 产出的骨架：
#   README.md  requirements.txt  xfdata/  user_data/  prediction_result/  code/

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?用法：bash build_xf_package.sh <输出目录> [--model <目录>] [--sim <目录>]}"
shift

MODEL_SRC=""; SIM_SRC=""
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL_SRC="${2:?--model 需要一个目录}"; shift 2 ;;
        --sim)   SIM_SRC="${2:?--sim 需要一个目录}";     shift 2 ;;
        *) echo "未知参数：$1"; exit 2 ;;
    esac
done

P="$OUT/spkrole_season2"
say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "1/8  建目录骨架 → $P"
mkdir -p "$P"/{xfdata,prediction_result}
mkdir -p "$P"/user_data/{model_data,tmp_data}
mkdir -p "$P"/code/{src,test,train,docs}
mkdir -p "$P"/user_data/tmp_data/{stage_outputs,artifacts}

say "2/8  顶层文档与依赖清单"
cp "$SRC/submission_package/README.md"        "$P/README.md"
cp "$SRC/submission_package/requirements.txt" "$P/requirements.txt"

say "3/8  入口脚本"
cp "$SRC/submission_package/code/test.sh"                "$P/code/test.sh"
cp "$SRC/submission_package/code/train.sh"               "$P/code/train.sh"
cp "$SRC/submission_package/code/requirements_train.txt" "$P/code/requirements_train.txt"
cp "$SRC/submission_package/code/test/run_stage1_9.sh"   "$P/code/test/run_stage1_9.sh"
chmod +x "$P/code/test.sh" "$P/code/train.sh" "$P/code/test/run_stage1_9.sh"

say "4/8  Python 源码 → code/src/"
# 只搬 .py，不带 __pycache__ / .pyc
find "$SRC/baseline/src" -maxdepth 1 -name "*.py" -exec cp {} "$P/code/src/" \;
echo "    $(ls -1 "$P/code/src"/*.py | wc -l | tr -d ' ') 个脚本"

say "5/8  训练代码 → code/train/"
cp "$SRC/baseline/src_gpu/mk_jsonl.py" "$P/code/train/mk_jsonl.py"
cp "$SRC/baseline/MOSS-Transcribe-Diarize/finetune.py" "$P/code/train/finetune.py"
cp "$SRC/baseline/MOSS-Transcribe-Diarize/LICENSE"     "$P/code/train/LICENSE.MOSS"
cat > "$P/code/train/README.md" <<'EOF'
# 训练代码

| 文件 | 来源 | 许可 |
|---|---|---|
| `mk_jsonl.py` | 本队自研：SegLST → MOSS 微调 conversation JSONL | 同本包 |
| `finetune.py` | 取自 MOSS 官方仓库 `OpenMOSS-Team/MOSS-Transcribe-Diarize`，未作修改 | Apache-2.0，见 `LICENSE.MOSS` |

数据增广脚本 `simulate_conv.py` 在 `code/src/` 下（它同时被链路的其它部分引用）。
完整训练流程见包根 `README.md` §6，一键执行 `bash code/train.sh`。
EOF

say "6/8  方案文档 → code/docs/"
cp "$SRC/project.md" "$P/code/docs/project.md"

say "7/8  Step 1-9 归档产物 → user_data/tmp_data/stage_outputs/"
S="$P/user_data/tmp_data/stage_outputs"
for d in diar_test_m3_7_0.70 emb_test test_raw test_wa_m05b6 test_v048_re \
         test_v051_moss test_cam_multi test_cam_re test_cam_wa; do
    if [ -d "$SRC/baseline/output/$d" ]; then
        mkdir -p "$S/$d"
        find "$SRC/baseline/output/$d" -maxdepth 1 -type f \
             ! -name ".*" -exec cp {} "$S/$d/" \;
        printf '    %5s  %s\n' "$(ls -1 "$S/$d" | wc -l | tr -d ' ')" "$d"
    else
        echo "    ⚠ 缺 $d"
    fi
done
# Step 3 的产物在 baseline/ 顶层，不在 output/ 下
mkdir -p "$S/output_test_fr"
find "$SRC/baseline/output_test_fr" -maxdepth 1 -type f -name "*.json" \
     -exec cp {} "$S/output_test_fr/" \;
printf '    %5s  %s\n' "$(ls -1 "$S/output_test_fr" | wc -l | tr -d ' ')" "output_test_fr"
cp "$SRC/baseline/artifacts/test_spk_scores2.json" "$P/user_data/tmp_data/artifacts/"

say "8/8  占位说明 + 可选的权重与仿真数据"
cat > "$P/xfdata/README.txt" <<'EOF'
本目录留给官方原始数据（代码审核规范 §1：选手无需打包，审核方清空后放入）。

期望结构（保持官网的文件名与层级）：

    xfdata/
    ├── dev/dev/
    │   ├── ref.seglst.json     106 场参考标注（code/train.sh 用）
    │   └── wav/                106 个 wav
    └── test/test/
        └── wav/                394 个 wav（code/test.sh --full 用）

code/test.sh 会依次尝试 xfdata/test/test/wav、xfdata/test/wav、xfdata/wav，
命中任一即可，以兼容不同的解压层级。

注意：code/test.sh 的默认模式复用 user_data/tmp_data/stage_outputs/ 下已归档的
Step 1-9 产物，不需要本目录有任何文件。
EOF
cat > "$P/prediction_result/README.txt" <<'EOF'
本目录留给预测结果（代码审核规范 §3：初始时会被清空）。

执行 bash code/test.sh 后，此处会生成 result.json —— SegLST 格式、UTF-8，
与竞赛提交要求一致，即我们线上取得 tcpWER 0.14309 的那份文件
（5186 条，SHA256 47bfe5f98afbe1f0…）。
EOF

cat > "$P/user_data/model_data/README.md" <<'EOF'
# 训练好的模型

`moss_sim_all106/` —— MOSS-Transcribe-Diarize 0.9B 的微调权重，HuggingFace 目录格式
（`config.json` / `model.safetensors` / `tokenizer.json` / `chat_template.jinja` 等）。

- **基座**：`OpenMOSS-Team/MOSS-Transcribe-Diarize`（Apache-2.0，公开可下）
- **训练数据**：`../tmp_data/sim_all106/`（官方 dev 重组的 800 条仿真对话，seed=0）
- **训练配方与全部超参**：见包根 `README.md` §6，一键复现 `bash code/train.sh`
- **用在哪**：链路 Step 2（`code/src/moss_sat.py` 的 `--model`）

链路里的其余三个模型（CAM++ / FireRedASR2-AED / fun-asr）都直接使用开源权重，
未做任何训练，因此不在此目录中。
EOF

if [ -n "$MODEL_SRC" ]; then
    echo "    拷权重 $MODEL_SRC → user_data/model_data/moss_sim_all106/"
    mkdir -p "$P/user_data/model_data/moss_sim_all106"
    find "$MODEL_SRC" -maxdepth 1 -type f -exec cp {} "$P/user_data/model_data/moss_sim_all106/" \;
    echo "    $(du -sh "$P/user_data/model_data/moss_sim_all106" | cut -f1)"
else
    echo "    （未传 --model，跳过权重）"
fi

if [ -n "$SIM_SRC" ]; then
    echo "    拷仿真数据 $SIM_SRC → user_data/tmp_data/sim_all106/"
    mkdir -p "$P/user_data/tmp_data/sim_all106"
    cp -R "$SIM_SRC/." "$P/user_data/tmp_data/sim_all106/"
    echo "    $(du -sh "$P/user_data/tmp_data/sim_all106" | cut -f1)（$(ls -1 "$P/user_data/tmp_data/sim_all106/wav"/*.wav 2>/dev/null | wc -l | tr -d ' ') 个 wav）"
else
    echo "    （未传 --sim，跳过仿真数据）"
fi

say "清理临时文件"
find "$P" \( -name "__pycache__" -o -name "*.pyc" -o -name ".DS_Store" \
          -o -name "._*" -o -name ".ipynb_checkpoints" \) -print0 2>/dev/null \
    | xargs -0 rm -rf 2>/dev/null || true

say "完成"
echo "包根：$P"
echo "总大小：$(du -sh "$P" | cut -f1)，文件数：$(find "$P" -type f | wc -l | tr -d ' ')"
