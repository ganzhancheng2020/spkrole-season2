#!/usr/bin/env bash
# 预测入口 —— 代码审核规范 §4a「请提供 test.sh 文件作为程序入口」。
#
#   bash code/test.sh          默认：复用 user_data/tmp_data/stage_outputs/ 下已归档的
#                              Step 1-9 产物，只跑 Step 10-13。纯 CPU，约 1 分钟。
#   bash code/test.sh --full   从 xfdata/ 的原始音频重跑 Step 1-9 再接 Step 10-13。
#                              需 GPU，约 5 小时（见 code/test/run_stage1_9.sh）。
#
# 结果写入 prediction_result/result.json（SegLST 格式，UTF-8），
# 并与我们线上提交文件的 SHA256 逐位比对。

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"     # 全程以 code/ 为工作目录，路径一律相对

XF=../xfdata
UD=../user_data
STAGE=$UD/tmp_data/stage_outputs
OUT=../prediction_result
PY="${PYTHON:-python3}"

FULL=0
for a in "$@"; do
    case "$a" in
        --full) FULL=1 ;;
        *) echo "未知参数：$a（可用：--full）"; exit 2 ;;
    esac
done

# 线上实测 0.14309 的最终提交文件；--full 重跑时 GPU 解码存在 bf16 非确定性，
# 哈希可能不一致，此时以「能产出结果」为准，不作硬失败。
EXPECT_FINAL="47bfe5f98afbe1f0"
EXPECT_V065="e78bba70c9d0c1f0"

ok=0; bad=0
say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
pass() { echo "  ✓ $*"; ok=$((ok+1)); }
fail() { echo "  ✗ $*"; bad=$((bad+1)); }
note() { echo "    $*"; }

say "环境检查"
command -v "$PY" >/dev/null || { fail "找不到 $PY（可用 PYTHON=... 指定）"; exit 1; }
pass "解释器 $("$PY" -V 2>&1)"
"$PY" -c "import numpy" 2>/dev/null \
    && pass "numpy（Step 13 需要）" \
    || { fail "缺 numpy，请先 pip install -r requirements.txt"; exit 1; }

say "输入数据"
WAV=""
for c in "$XF/test/test/wav" "$XF/test/wav" "$XF/wav"; do
    [ -d "$c" ] && { WAV="$c"; break; }
done
if [ -n "$WAV" ]; then
    pass "原始音频 $(ls -1 "$WAV"/*.wav 2>/dev/null | wc -l | tr -d ' ') 个 → $WAV"
else
    note "xfdata/ 下暂无 test 音频（默认模式不需要，Step 1-9 产物已归档）"
    [ "$FULL" = 1 ] && { fail "--full 模式必须有原始音频，请把官方数据放到 xfdata/"; exit 1; }
fi

if [ "$FULL" = 1 ]; then
    say "Step 1-9  从原始音频重跑（GPU）"
    bash test/run_stage1_9.sh "$WAV" || { fail "Step 1-9 失败"; exit 1; }
fi

say "Step 1-9  产物齐备性"
need() {  # need <目录> <期望条数> <阶段名>
    local n
    n=$(ls -1 "$STAGE/$1"/*.seglst.json 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" = "0" ] && n=$(ls -1 "$STAGE/$1"/* 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" = "$2" ]; then return 0; fi
    fail "$3 产物不全：$1 期望 $2 个，实得 $n"
    note "请用 bash code/test.sh --full 从原始音频重跑"
    exit 1
}
need diar_test_m3_7_0.70 394 "Step 1"
need emb_test            394 "Step 1"
need test_raw            394 "Step 2"
need test_wa_m05b6       394 "Step 4"
need test_v048_re        394 "Step 5"
need test_cam_multi      394 "Step 7"
need test_cam_re         394 "Step 8"
need test_cam_wa         394 "Step 9"
[ -f "$UD/tmp_data/artifacts/test_spk_scores2.json" ] \
    || { fail "缺 user_data/tmp_data/artifacts/test_spk_scores2.json（Step 6 输入）"; exit 1; }
pass "9 个阶段产物齐备"

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT

say "Step 6   短段归属改写 → MOSS 分支完成"
if PYTHONPATH=src "$PY" src/spk_short_relabel.py \
        --pred-dir "$STAGE/test_v048_re" \
        --scores "$UD/tmp_data/artifacts/test_spk_scores2.json" \
        --out "$W/test_v051_moss" --tau 1.5 --max-words 5 >"$W/s6.log" 2>&1; then
    grep -oE "改写 [0-9]+ / [0-9]+ 段[^，]*" "$W/s6.log" | head -1 | sed 's/^/    /'
    pass "test_v051_moss"
else
    fail "Step 6 失败：$(tail -1 "$W/s6.log")"; exit 1
fi

say "Step 10  逐场路由（MOSS 311 场 / CAM 83 场）"
PYTHONPATH=src "$PY" src/pick_ensemble.py \
    --moss "$W/test_v051_moss" --cam "$STAGE/test_cam_wa" \
    --out "$W/pick" --moss-max-turn-rate 0.05 2>&1 \
    | grep -E "共 [0-9]+ 段" | sed 's/^/    /'

say "Step 11  字符规范化"
PYTHONPATH=src "$PY" src/normalize_output.py "$W/pick" "$W/norm" 2>&1 \
    | grep -E "共 [0-9]+ 个" | sed 's/^/    /'

say "Step 12  合并"
PYTHONPATH=src "$PY" src/merge_submit.py --dir "$W/norm" \
    --out "$W/submission_v065.json" >/dev/null 2>&1

say "Step 13  CAM 引导 + 声纹校验的说话人拆分 → 最终结果"
mkdir -p "$OUT"
PYTHONPATH=src "$PY" src/cam_split_verify.py \
    --hyp "$W/submission_v065.json" \
    --diar "$STAGE/diar_test_m3_7_0.70" --emb "$STAGE/emb_test" \
    --out "$OUT/result.json" \
    --min-frac 0.05 --max-sim 0.50 --min-gap 1 --max-gap 1 2>&1 \
    | grep -E "拆分" | sed 's/^/    /'

say "结果校验"
[ -s "$OUT/result.json" ] || { fail "未产出 prediction_result/result.json"; exit 1; }
N=$("$PY" -c "import json,sys;print(len(json.load(open(sys.argv[1],encoding='utf-8'))))" "$OUT/result.json")
pass "prediction_result/result.json（$N 条 SegLST 记录）"

sha16() { if command -v shasum >/dev/null; then shasum -a 256 "$1" | cut -c1-16
          else sha256sum "$1" | cut -c1-16; fi; }
check() {  # check <文件> <期望前16位> <名称>
    local got; got=$(sha16 "$1")
    if [ "$got" = "$2" ]; then
        pass "$3"; note "SHA256 $got"
    elif [ "$FULL" = 1 ]; then
        pass "$3（--full 重跑；GPU bf16 解码非逐位确定，哈希允许不同）"
        note "期望 $2，实得 $got"
    else
        fail "$3 期望 $2，实得 $got"
    fi
}
check "$W/submission_v065.json" "$EXPECT_V065"  "中间件 submission_v065（线上 0.14354）"
check "$OUT/result.json"        "$EXPECT_FINAL" "最终结果 result.json（线上 0.14309）"

say "小结：通过 $ok 项，失败 $bad 项"
[ "$bad" = 0 ] || exit 1
echo "预测完成 → prediction_result/result.json"
