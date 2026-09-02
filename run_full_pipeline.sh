#!/usr/bin/env bash
# 一键跑通全链路：从原始音频到最终提交文件（各 Step 的说明见 README「Reproduce」）
#
#   Step 1     CAM++ 分离 + 段级声纹                   本地
#   Step 2-4   MOSS 解码 → FireRed 重转写 → 词级仲裁   GPU
#   Step 5-6   归属改写 → 短段归属改写                 本地   → MOSS 分支
#   Step 7-9   fun-asr → 重打标签 → 词级仲裁           云端/本地/GPU → CAM 分支
#   Step 10-13 路由 → 规范化 → 合并 → 声纹校验拆分     本地   → 最终文件
#
# 需 GPU 或云端服务的阶段（2/3/4/7/9），若产物已在仓库中则直接复用并标注 [复用]；
# 产物缺失时打印该阶段的完整命令并退出。
#
# 用法：
#   bash run_full_pipeline.sh              # 复用已归档产物，跑完可本地执行的阶段
#   bash run_full_pipeline.sh --rerun-slow # 额外重跑耗时阶段（Step 1 约 7 min、Step 5 约 25 min）
#   bash run_full_pipeline.sh --check      # 只做产物齐备性与谱系校验，不执行

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="$ROOT/baseline"
PY="$B/.venv/bin/python"             # 流水线脚本（无 numpy）
PY_NP="$ROOT/.venv/bin/python"       # 数值计算与评分（numpy + meeteval）
PY_DZ="$B/diarizen_venv/bin/python"  # CAM++ / 声纹 / 词级仲裁
WAV="$ROOT/data/extracted/test/test/wav"

RERUN_SLOW=0; CHECK_ONLY=0
for a in "$@"; do
    case "$a" in
        --rerun-slow) RERUN_SLOW=1 ;;
        --check)      CHECK_ONLY=1 ;;
        *) echo "未知参数：$a"; exit 2 ;;
    esac
done

EXPECT_V065="e78bba70c9d0c1f0"
EXPECT_FINAL="47bfe5f98afbe1f0"

ok=0; bad=0
say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
pass() { echo "  ✓ $*"; ok=$((ok+1)); }
fail() { echo "  ✗ $*"; bad=$((bad+1)); }
note() { echo "    $*"; }

need() {  # need <目录> <期望文件数> <阶段名> <缺失时应执行的命令>
    # 数该阶段真正的产物文件，忽略同目录下的日志等附带文件
    local n
    n=$(ls -1 "$B/$1"/*.seglst.json 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" = "0" ] && n=$(ls -1 "$B/$1" 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" = "$2" ] && return 0
    fail "$3 缺少产物 $1（期望 $2 个，实得 $n）"
    note "该阶段需 GPU 或云端服务，请先执行："
    note "$4"
    exit 1
}

say "环境检查"
for pair in "$PY:baseline/.venv" "$PY_NP:.venv" "$PY_DZ:baseline/diarizen_venv"; do
    [ -x "${pair%%:*}" ] && pass "解释器 ${pair##*:}" || { fail "找不到解释器 ${pair##*:}"; exit 1; }
done
"$PY_NP" -c "import numpy" 2>/dev/null && pass "numpy（cam_split_verify 需要）" \
    || { fail "$PY_NP 缺 numpy"; exit 1; }
[ -d "$WAV" ] && pass "原始音频 $(ls -1 "$WAV"/*.wav 2>/dev/null | wc -l | tr -d ' ') 个 wav" \
              || fail "找不到 $WAV"

say "Step 1  CAM++ 分离 + 段级声纹"
if [ "$RERUN_SLOW" = 1 ] && [ "$CHECK_ONLY" = 0 ]; then
    echo "  重跑中（约 7 分钟）…"
    PYTHONPATH="$B/src" "$PY_DZ" "$B/src/diarize.py" --wav-dir "$WAV" \
        --out "$B/output/diar_test_m3_7_0.70" --min-spk 3 --max-spk 7 \
        --merge-thr 0.70 --emb-cache "$B/output/emb_test" >/dev/null 2>&1 \
        && pass "已重跑" || fail "重跑失败"
else
    need output/diar_test_m3_7_0.70 394 "Step 1" \
"PYTHONPATH=src diarizen_venv/bin/python src/diarize.py --wav-dir \$WAV \\
        --out output/diar_test_m3_7_0.70 --min-spk 3 --max-spk 7 --merge-thr 0.70 --emb-cache output/emb_test"
    pass "[复用] diar_test_m3_7_0.70 + emb_test（分离结果为钉定基准，见 project.md 附录 C）"
fi

say "Step 2-4  MOSS 解码 → FireRed 重转写 → 词级仲裁（GPU）"
need output/test_raw      394 "Step 2" "python src/moss_sat.py --wav-dir \$WAV --out output/test_raw --model <微调模型>"
need output_test_fr       394 "Step 3" "python src/fr_retext.py output/test_raw \$WAV output_test_fr"
need output/test_wa_m05b6 394 "Step 4" "MARGIN=0.05 MAX_BLOCKS=6 diarizen_venv/bin/python src/word_arb.py output/test_raw output_test_fr \$WAV output/test_wa_m05b6"
pass "[复用] test_raw → output_test_fr → test_wa_m05b6"

say "Step 5  归属改写"
if [ "$RERUN_SLOW" = 1 ] && [ "$CHECK_ONLY" = 0 ]; then
    echo "  重跑中（约 25 分钟）…"
    W5=$(mktemp -d)
    COH_MIN=0.55 MARGIN=0.05 "$PY_DZ" "$B/src/spk_reassign.py" \
        "$B/output/test_wa_m05b6" "$WAV" "$W5/out" >/dev/null 2>&1 \
        && pass "已重跑 → $W5/out" || fail "重跑失败"
else
    need output/test_v048_re 394 "Step 5" \
"COH_MIN=0.55 MARGIN=0.05 diarizen_venv/bin/python src/spk_reassign.py output/test_wa_m05b6 \$WAV output/test_v048_re"
    pass "[复用] test_v048_re"
fi

W=$(mktemp -d)
MOSS_DIR="$B/output/test_v051_moss"

if [ "$CHECK_ONLY" = 0 ]; then
    say "Step 6  短段归属改写 → MOSS 分支完成"
    if PYTHONPATH="$B/src" "$PY_NP" "$B/src/spk_short_relabel.py" \
            --pred-dir "$B/output/test_v048_re" --scores "$B/artifacts/test_spk_scores2.json" \
            --out "$W/test_v051_moss" --tau 1.5 --max-words 5 >"$W/s6.log" 2>&1; then
        grep -oE "改写 [0-9]+ / [0-9]+ 段[^，]*" "$W/s6.log" | head -1 | sed 's/^/    /'
        MOSS_DIR="$W/test_v051_moss"
        pass "test_v051_moss"
    else
        fail "Step 6 失败：$(tail -1 "$W/s6.log")"; exit 1
    fi
fi

say "Step 7-9  fun-asr → 重打标签 → 词级仲裁 → CAM 分支完成"
need output/test_cam_multi 394 "Step 7" "PYTHONPATH=src .venv/bin/python src/run.py --out output/test_cam_multi"
need output/test_cam_re    394 "Step 8" "diarizen_venv/bin/python src/spk_relabel.py output/test_cam_multi \$WAV output/test_cam_re"
need output/test_cam_wa    394 "Step 9" "MARGIN=0.05 MAX_BLOCKS=6 diarizen_venv/bin/python src/word_arb.py output/test_cam_re output/_test_camre_fr \$WAV output/test_cam_wa"
pass "[复用] test_cam_multi → test_cam_re → test_cam_wa"

say "谱系校验（逐条内容比对）"
"$PY_NP" - "$B" <<'PY'
import json, glob, os, sys
B = sys.argv[1]
def load(d):
    r = {}
    for f in glob.glob(os.path.join(B, d, '*.seglst.json')):
        x = json.load(open(f, encoding='utf-8'))
        r[os.path.basename(f).split('.')[0]] = x if isinstance(x, list) else x['segments']
    return r
def flat(D, field=None):
    return [(s, round(float(r['start_time']), 3)) + ((r[field],) if field else ())
            for s in sorted(D) for r in D[s]]
EDGES = [
    ('output/test_raw',       'output/test_wa_m05b6', 1169,  0, 'Step 4 只改文本'),
    ('output/test_wa_m05b6',  'output/test_v048_re',     0, 26, 'Step 5 只改标签'),
    ('output/test_v048_re',   'output/test_v051_moss',   0, 47, 'Step 6 只改标签'),
    ('output/test_cam_multi', 'output/test_cam_re',      0, 87, 'Step 8 只改标签'),
    ('output/test_cam_re',    'output/test_cam_wa',    828,  0, 'Step 9 只改文本'),
]
bad = 0
for a, b, et, es, desc in EDGES:
    A, Bv = load(a), load(b)
    if flat(A) != flat(Bv):
        print(f'  ✗ {desc}: 切分不一致'); bad += 1; continue
    nt = sum(1 for x, y in zip(flat(A, 'words'),   flat(Bv, 'words'))   if x != y)
    ns = sum(1 for x, y in zip(flat(A, 'speaker'), flat(Bv, 'speaker')) if x != y)
    good = (nt == et and ns == es)
    print(f'  {"✓" if good else "✗"} {desc:16s} 文本差 {nt:5d}(期望 {et:4d})  说话人差 {ns:3d}(期望 {es:3d})')
    bad += 0 if good else 1
sys.exit(1 if bad else 0)
PY
[ $? = 0 ] && pass "5 条边全部一致" || fail "谱系比对存在不一致"

if [ "$CHECK_ONLY" = 1 ]; then
    say "小结（--check）：通过 $ok 项，失败 $bad 项"
    [ "$bad" = 0 ] || exit 1
    exit 0
fi

check() {  # check <文件> <期望 SHA256 前16位> <名称>
    local got; got=$(shasum -a 256 "$1" | cut -c1-16)
    if [ "$got" = "$2" ]; then pass "$3"; note "SHA256 $got"; else fail "$3 期望 $2，实得 $got"; fi
}

say "Step 10  逐场路由"
PYTHONPATH="$B/src" "$PY" "$B/src/pick_ensemble.py" --moss "$MOSS_DIR" \
    --cam "$B/output/test_cam_wa" --out "$W/pick" --moss-max-turn-rate 0.05 2>&1 \
    | grep -E "共 [0-9]+ 段" | sed 's/^/    /'

say "Step 11  字符规范化"
PYTHONPATH="$B/src" "$PY" "$B/src/normalize_output.py" "$W/pick" "$W/norm" 2>&1 \
    | grep -E "共 [0-9]+ 个" | sed 's/^/    /'

say "Step 12  合并 → submission_v065.json"
PYTHONPATH="$B/src" "$PY" "$B/src/merge_submit.py" --dir "$W/norm" \
    --out "$W/submission_v065.json" >/dev/null 2>&1
check "$W/submission_v065.json" "$EXPECT_V065" "submission_v065.json（线上 0.14354）"

say "Step 13  CAM 引导 + 声纹校验的说话人拆分 → 最终文件"
PYTHONPATH="$B/src" "$PY_NP" "$B/src/cam_split_verify.py" \
    --hyp "$W/submission_v065.json" --diar "$B/output/diar_test_m3_7_0.70" \
    --emb "$B/output/emb_test" --out "$W/submission_v094b.json" \
    --min-frac 0.05 --max-sim 0.50 --min-gap 1 --max-gap 1 2>&1 \
    | grep -E "拆分" | sed 's/^/    /'
check "$W/submission_v094b.json" "$EXPECT_FINAL" "submission_v094b.json（线上 0.14309，最终提交）"

say "小结：通过 $ok 项，失败 $bad 项"
[ "$bad" = 0 ] || exit 1
echo "全链路跑通。最终文件：$W/submission_v094b.json"
