#!/usr/bin/env bash
# CV 完成后的第一件事：在**出货配置**上重测轴分解。
# 此前的 14.73% / 10.54% / 10.15% 全是 train-82 模型在 24 段上的数。
set -euo pipefail
cd "$(dirname "$0")/baseline"
V=.venv/bin/python
M=../.venv/bin/meeteval-wer
REF=../data/extracted/dev/dev/ref.seglst.json

sc () {  # sc <dir> -> tcpWER 行
  $V - "$1" <<'PY'
import json,glob,sys
recs=[]
for p in sorted(glob.glob(f'output/{sys.argv[1]}/[0-9]*.seglst.json')): recs+=json.load(open(p,encoding='utf-8'))
json.dump(recs,open('/tmp/h.json','w',encoding='utf-8'),ensure_ascii=False)
sids={r['session_id'] for r in recs}
json.dump([r for r in json.load(open('../data/extracted/dev/dev/ref.seglst.json',encoding='utf-8')) if r['session_id'] in sids],open('/tmp/r.json','w',encoding='utf-8'),ensure_ascii=False)
PY
  $M tcpwer -r /tmp/r.json -h /tmp/h.json --collar 5 2>&1 | grep -o '%tcpWER.*'
}

echo "== 1. FireRed 文本（复用缓存 ts，纯 CPU）=="
$V src/fr_assign_cached.py output/_cv output/fr_retext_dev_ts output/_cv_fr

echo "== 2. 造子集 ref（oracle_text 有词数守恒断言，必须同口径）=="
$V src/cv_subset_ref.py output/_cv /tmp/cv_ref.json

echo "== 3. 归属 / 文本 oracle =="
$V src/oracle_speaker.py --pred-dir output/_cv --ref /tmp/cv_ref.json --out output/_cv_orc_spk 2>&1|tail -3
$V src/oracle_text.py    --pred-dir output/_cv --ref /tmp/cv_ref.json --out output/_cv_orc_txt 2>&1|tail -2

echo
echo "===== 出货配置在 106 段干净评测集上的轴分解 ====="
printf "  %-28s %s\n" "原始 MOSS（CV）"      "$(sc _cv)"
printf "  %-28s %s\n" "归属 oracle"          "$(sc _cv_orc_spk)"
printf "  %-28s %s\n" "文本 oracle"          "$(sc _cv_orc_txt)"
printf "  %-28s %s\n" "FireRed 换文本"       "$(sc _cv_fr)"
echo
echo "对照：train-82 模型在 24 段上的旧数 —— 全链路 14.73 / 归属 oracle 10.54 / 文本 oracle 10.15"
