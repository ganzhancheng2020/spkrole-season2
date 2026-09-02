#!/usr/bin/env bash
# v051 = v048 + CAM 分支声学仲裁 + 短段归属改写
#
# 两处改动相对 v048 都是**单变量可分离**的：
#   ① CAM 分支此前从未做过声学仲裁 —— spk_relabel 只改说话人标签、不动文本，
#      所以 pick 路由过去的 32.5% session 一直用未仲裁的 fun-asr 原始文本。
#      配方沿用 MOSS 分支已线上验证的 MARGIN=0.05 / MAX_BLOCKS=6。
#   ② 短段归属改写：MOSS 说话人似然 + 词数≤5 门 + τ=1.5。
#
# 106 段干净 CV 集实测（bench.py 四关全过）：
#   基线 15.751% → CAM 仲裁 15.596%(−0.154) → +短段改写 15.571%(−0.180)
#   触发 64 段 / 整段剔 top3 仍为收益 / 自助 P(更好)=93.3% / 改善 34 恶化 30
set -euo pipefail
cd "$(dirname "$0")/baseline"
V=.venv/bin/python

echo "== 前置检查 =="
for d in test_v048_re test_cam_wa; do
  n=$(ls output/$d/*.seglst.json 2>/dev/null | wc -l | tr -d ' ')
  echo "  $d: $n"
  [ "$n" = "394" ] || { echo "❌ $d 不是 394 段，中止"; exit 1; }
done
[ -f /tmp/test_spk_scores2.json ] || { echo "❌ 缺 test 侧说话人似然分数"; exit 1; }

echo "== 1/4 短段归属改写（作用于 MOSS 分支）=="
$V src/spk_short_relabel.py --pred-dir output/test_v048_re \
   --scores /tmp/test_spk_scores2.json --out output/test_v051_moss

echo "== 2/4 pick（MOSS=改写后, CAM=仲裁后）=="
$V src/pick_ensemble.py --moss output/test_v051_moss --cam output/test_cam_wa \
   --moss-max-turn-rate 0.08 --out output_test_v051 >/dev/null

echo "== 3/4 normalize =="
$V src/normalize_output.py output_test_v051 output_test_v051_norm | tail -1

echo "== 4/4 merge =="
$V src/merge_submit.py --dir output_test_v051_norm --out ../submit/submission_v051.json | tail -2
cp ../submit/submission_v051.json ../submit/archive/submission_v051_$(date +%Y%m%d).json

echo "== 与 v048 对比 =="
python3 - <<'PY'
import json, collections
a = json.load(open('../submit/archive/submission_v048_20260821.json', encoding='utf-8'))
b = json.load(open('../submit/submission_v051.json', encoding='utf-8'))
for t, r in [('v048', a), ('v051', b)]:
    bs = collections.defaultdict(set)
    for x in r:
        bs[x['session_id']].add(x['speaker'])
    print(f"{t}: {len(r)} 条 | {len(bs)} session | "
          f"词 {sum(len(x['words'].split()) for x in r)} | "
          f"人数分布 {dict(sorted(collections.Counter(len(v) for v in bs.values()).items()))}")
ka = collections.Counter((x['session_id'], round(x['start_time'], 3), x['speaker']) for x in a)
kb = collections.Counter((x['session_id'], round(x['start_time'], 3), x['speaker']) for x in b)
print(f"切分+说话人相同的记录: {sum((ka & kb).values())}")
PY
