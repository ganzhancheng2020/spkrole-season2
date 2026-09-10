#!/usr/bin/env bash
# 按《科大讯飞-代码审核规范》组装提交包。构建器本身不进包。
#
#   bash build_xf_package.sh <输出目录> [--model <权重目录>] [--base <基座快照>] [--sim <仿真数据>]
#
# 例：
#   bash build_xf_package.sh /tmp/pkg                       # 只装代码+归档产物（本机）
#   bash build_xf_package.sh /root/pkg \                    # 连权重和 800 条仿真一起装（GPU）
#       --model /root/autodl-tmp/ft/sim_out_all106 \
#       --base  /root/autodl-tmp/hf/hub/models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/<sha> \
#       --sim   /root/autodl-tmp/ft/sim_all106
#
# --base 用来把 processor 文件补进微调权重目录：HF Trainer 只存模型与 tokenizer，
# 少了 preprocessor_config.json 等，moss_sat.py 会报缺 feature_extractor 并产出 0 条。
#
# 产出的骨架：
#   README.md  requirements.txt  xfdata/  user_data/  prediction_result/  code/

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?用法：bash build_xf_package.sh <输出目录> [--model <目录>] [--sim <目录>]}"
shift

MODEL_SRC=""; SIM_SRC=""; BASE_SRC=""
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL_SRC="${2:?--model 需要一个目录}"; shift 2 ;;
        --sim)   SIM_SRC="${2:?--sim 需要一个目录}";     shift 2 ;;
        --base)  BASE_SRC="${2:?--base 需要一个目录}";   shift 2 ;;
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
# project.md 原本写给研究仓（位于仓库根），直接搬进 code/docs/ 会有两个问题：
#   ① 文中的 [xxx](README.md) 会解析成 code/docs/README.md —— 不存在
#   ② 文中提到的 run_full_pipeline.sh / online_ledger.md / build_v0*.sh 等
#      属研究仓的实验脚本，按规范不进提交包
# 故这里重写相对链接并前置一段来源说明，避免审核方读到断链。
{
    cat <<'HDR'
> **本文来源说明**：这份《方案归档与复盘》原文写给项目的研究仓库，收录进提交包
> 是作为算法方案与演进过程的完整说明（对应规范 §4「方案中的算法贡献」）。
>
> 阅读时请注意两点：
>
> 1. 文中形如 `run_full_pipeline.sh`、`online_ledger.md`、`build_v047.sh`、
>    `submission_v065.json` 的文件属于研究仓的实验脚本与流水账，**按规范不随包分发**；
>    提交包中与之对应的是 `code/test.sh`（一键复现全链路）与 `code/train.sh`（训练）。
> 2. 文中引用的 README 指**包根的 `README.md`**（本文位于 `code/docs/`）。

---

HDR
    sed 's|](README\.md|](../../README.md|g' "$SRC/project.md"
} > "$P/code/docs/project.md"
echo "    已重写相对链接并前置来源说明"

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

执行 bash code/test.sh 后，此处会生成：

    result        主产物 —— 本规范 §3 要求的文件名，SegLST 格式、UTF-8
    result.json   同内容副本，便于按赛题的 JSON 惯例直接查看与校验

两者内容逐字节相同，即我们线上取得 tcpWER 0.14309 的那份文件
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
    M="$P/user_data/model_data/moss_sim_all106"
    echo "    拷权重 $MODEL_SRC → user_data/model_data/moss_sim_all106/"
    mkdir -p "$M"
    find "$MODEL_SRC" -maxdepth 1 -type f -exec cp {} "$M/" \;
    echo "    $(du -sh "$M" | cut -f1)"

    # HF Trainer 存下来的 checkpoint 只带 tokenizer：对它调 AutoProcessor 会**静默退化**成
    # Qwen2Tokenizer（无 feature_extractor），moss_sat.py 因此产出 0 条。实测把基座的
    # preprocessor_config.json 等拷进 checkpoint 也修不好。moss_sat.py 的既定契约是
    # 用 --processor 指向基座快照，所以这里单独放一份**不含权重**的 processor 目录（约 16 MB）。
    if [ -n "$BASE_SRC" ]; then
        PR="$P/user_data/model_data/moss_base_processor"
        mkdir -p "$PR"
        for f in $(ls -1 "$BASE_SRC"); do
            case "$f" in *.safetensors|*.safetensors.index.json|*.png|*.bin) continue ;; esac
            cp -L "$BASE_SRC/$f" "$PR/" 2>/dev/null || true
        done
        echo "    processor（无权重）→ user_data/model_data/moss_base_processor/ $(du -sh "$PR" | cut -f1)"
        [ -f "$PR/preprocessor_config.json" ] && [ -f "$PR/processing_moss_transcribe_diarize.py" ] \
            && echo "    ✓ processor 文件齐备" \
            || echo "    ⚠ processor 目录不全，--full 的 Step 2 会失败"
    else
        echo "    ⚠ 未传 --base：包内没有 processor 目录，--full 的 Step 2 会失败"
    fi
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
