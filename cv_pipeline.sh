#!/usr/bin/env bash
# 把 106 段 CV 预测走完与 v040 完全一致的出货链路，得到可比的干净基线。
#
#   MOSS(CV) -> word_arb(GPU) -> spk_reassign(本地/embedding) -> pick -> normalize
#
# word_arb 必须在 GPU 上跑（FireRedASR2 声学似然）；其余三步本地即可。
# FireRed 的整场词级时间戳复用 output/fr_retext_dev_ts 的缓存，不重跑 ASR。
set -euo pipefail
cd "$(dirname "$0")/baseline"
V=.venv/bin/python

echo "== 1. 把缓存的 FireRed 文本分配到 CV 切分 =="
$V src/fr_assign_cached.py output/_cv output/fr_retext_dev_ts output/_cv_fr

echo "== 2. word_arb 需在 GPU 上跑，这里只打包上传 =="
tar czf /tmp/cv_for_wa.tgz output/_cv output/_cv_fr
echo "   -> /tmp/cv_for_wa.tgz（用 cv_wa_remote.sh 在 GPU 上跑）"

echo "== 3. 归属改写（本地，用 dev embedding 缓存）=="
if [ -d output/_cv_wa ]; then
  COH_MIN=0.55 MARGIN=0.05 SINGLE_WIN_TRUST=0 diarizen_venv/bin/python src/spk_reassign.py \
      output/_cv_wa ../data/extracted/dev/dev/wav output/_cv_re 2>&1 | tail -2
else
  echo "   跳过：还没有 output/_cv_wa（等 word_arb 回传）"
fi

echo "== 4. pick + normalize =="
SRC=$([ -d output/_cv_re ] && echo _cv_re || echo _cv)
$V src/pick_ensemble.py --moss output/$SRC --cam output/cam_re \
    --moss-max-turn-rate 0.08 --out output/_cv_pick >/dev/null
$V src/normalize_output.py output/_cv_pick output/_cv_final 2>&1 | tail -1

echo
echo "===== v040 等价链路在干净 CV 集上的分数（源=$SRC）====="
$V src/cv_subset_ref.py output/_cv_final /tmp/cv_ref.json
$V - <<'PY'
import json,glob
recs=[]
for p in sorted(glob.glob('output/_cv_final/[0-9]*.seglst.json')): recs+=json.load(open(p,encoding='utf-8'))
json.dump(recs,open('/tmp/h.json','w',encoding='utf-8'),ensure_ascii=False)
PY
../.venv/bin/meeteval-wer tcpwer -r /tmp/cv_ref.json -h /tmp/h.json --collar 5 2>&1 | grep -o '%tcpWER.*'
