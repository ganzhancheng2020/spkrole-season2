#!/usr/bin/env bash
# v053 = v051 链路，但 test 侧 CAM 分支改用与 dev 同型的 relabel(v026) 构造
#
# 依据（见 logs/2026-08-26.md + wiki/insights/dev-test-chain-mismatch.md）：
#   dev 的 CAM 分支一直是 relabel(v026_pick)，test 的却是 relabel(fun-asr)，
#   后者在 dev 106 段上实测差 5.122 点。即 v051 的 dev 验证（15.571%）在强 CAM 上做，
#   而实际提交用的是弱 CAM —— 一直在欠发挥。
#
# dev 证据（106 段）：
#   分支层 20.754% -> 15.632%，Δ-5.122，四关全过
#     （触发 68 / 改善 57 / 恶化 11 / 剔top3 -3.826 / P=100%）
#   链路层 16.229% -> 15.776%，Δ-0.453，挂第 2 关（+0.082）
#     —— 因 pick 把 2/3 路由给 MOSS、仅 12 段不同，是稀释不是脆弱
set -euo pipefail
cd "$(dirname "$0")/baseline"
V=.venv/bin/python

echo "== 前置检查 =="
for d in test_v051_moss tv027_wa; do
  n=$(ls output/$d/*.seglst.json 2>/dev/null | wc -l | tr -d ' ')
  echo "  $d: $n"
  [ "$n" = "394" ] || { echo "❌ $d 不是 394 段，中止"; exit 1; }
done

echo "== 1/3 pick（MOSS=v051 分支, CAM=v026 基仲裁后）=="
$V src/pick_ensemble.py --moss output/test_v051_moss --cam output/tv027_wa \
   --moss-max-turn-rate 0.08 --out output_test_v053 >/dev/null

echo "== 2/3 normalize =="
$V src/normalize_output.py output_test_v053 output_test_v053_norm | tail -1

echo "== 3/3 merge =="
$V src/merge_submit.py --dir output_test_v053_norm --out ../submit/submission_v053.json | tail -2
cp ../submit/submission_v053.json ../submit/archive/submission_v053_$(date +%Y%m%d).json

echo "== 与 v051 对比 =="
python3 - <<'PY'
import json, collections
a=json.load(open('../submit/submission_v051.json',encoding='utf-8'))
b=json.load(open('../submit/submission_v053.json',encoding='utf-8'))
for t,r in (('v051',a),('v053',b)):
    bs=collections.defaultdict(set)
    for x in r: bs[x['session_id']].add(x['speaker'])
    print(f"{t}: {len(r)} 条 | {len(bs)} session | 词 {sum(len(str(x['words']).split()) for x in r)} | "
          f"人数分布 {dict(sorted(collections.Counter(len(v) for v in bs.values()).items()))}")
st=lambda r:(r['session_id'],round(r['start_time'],3),round(r['end_time'],3))
ka={st(x) for x in a}; kb={st(x) for x in b}
print(f"切分相同的记录: {len(ka&kb)}；v053 独有 {len(kb-ka)}")
bad=[s for s,v in collections.Counter(x['session_id'] for x in b).items() if v<3]
print(f"退化 session(<3 条): {len(bad)}")
PY
