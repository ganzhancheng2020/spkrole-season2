#!/usr/bin/env bash
# v047 = v040 + 开新说话人（spk_open_new，τ=-2.0）
# 前置：GPU 上跑完 test 侧探针，产出 test_spk_scores.json
set -euo pipefail
cd "$(dirname "$0")/baseline"
GPU='sshpass -p 1MKdLIjhr2uF ssh -p 43879 -o StrictHostKeyChecking=no root@connect.nmb2.seetacloud.com'
B=/root/autodl-tmp/hf/hub/models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/e8681d68e7042738ffca8ac8212bc8fcb1131ab8

if [ ! -f /tmp/test_spk_scores.json ]; then
  echo "== GPU: 跑 test 侧探针（394 session，约 25 分钟）=="
  $GPU "cd /root/autodl-tmp/ft && nohup /root/autodl-tmp/envs/moss/bin/python src/spk_ac_probe.py \
      --jsonl test_probe.jsonl --model $B --out test_spk_scores.json > testprobe.log 2>&1 &"
  until $GPU 'grep -q SPK_AC_PROBE_DONE\|DONE /root/autodl-tmp/ft/testprobe.log' 2>/dev/null; do sleep 60; done
  sshpass -p 1MKdLIjhr2uF scp -P 43879 -o StrictHostKeyChecking=no \
      root@connect.nmb2.seetacloud.com:/root/autodl-tmp/ft/test_spk_scores.json /tmp/
fi

echo "== 开新说话人 =="
.venv/bin/python src/spk_open_new.py --pred-dir output/test_all106_wa \
    --scores /tmp/test_spk_scores.json --out output/test_v047_open --tau -2.0

echo "== 其余环节与 v040 完全一致 =="
.venv/bin/python src/pick_ensemble.py --moss output/test_v047_open --cam output/test_cam_re \
    --moss-max-turn-rate 0.08 --out output_test_v047
.venv/bin/python src/normalize_output.py output_test_v047 output_test_v047_norm
.venv/bin/python src/merge_submit.py --dir output_test_v047_norm --out ../submit/submission_v047.json
cp ../submit/submission_v047.json ../submit/archive/submission_v047_$(date +%Y%m%d).json
echo "== 与 v040 对比 =="
python3 - <<'PY'
import json
for t,f in [('v040','../submit/archive/submission_v040_20260817.json'),
            ('v047','../submit/submission_v047.json')]:
    r=json.load(open(f,encoding='utf-8'))
    import collections
    bs=collections.defaultdict(set)
    for x in r: bs[x['session_id']].add(x['speaker'])
    print(f"{t}: {len(r)} 条 | 词 {sum(len(x['words'].split()) for x in r)} | "
          f"人数分布 {dict(sorted(collections.Counter(len(v) for v in bs.values()).items()))}")
PY
