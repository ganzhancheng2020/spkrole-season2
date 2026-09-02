#!/bin/bash
# 等 re_a 完成 → 拼装 re_b/re_c（相同 session 复用 re_a，差异用 diff 目录）→ 各跑 4-6 步
set -e
cd /root/autodl-tmp/ft
PY=/root/autodl-tmp/envs/moss/bin/python
while pgrep -f "spk_reassign.py wa_v091a" > /dev/null; do sleep 20; done
echo "re_a done: $(ls re_v091a | grep -cE \"^[0-9]\")"
$PY - <<PYE
import json, glob, os, shutil
def load(d):
    return {os.path.basename(p): json.load(open(p)) for p in glob.glob(d+"/[0-9]*.seglst.json")}
a = load("test_arms_jg/jg_0.1")
for tag, other in (("b","0.05"), ("c","0.0")):
    o = load(f"test_arms_jg/jg_{other}")
    os.makedirs(f"re_v091{tag}", exist_ok=True)
    n_copy = n_diff = 0
    for k, recs in a.items():
        dst = f"re_v091{tag}/{k}"
        same = json.dumps(recs, sort_keys=True) == json.dumps(o.get(k, []), sort_keys=True)
        if same:
            src = f"re_v091a/{k}"
            if os.path.exists(src):
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
                n_copy += 1
        else:
            src = f"re_v091{tag}_diff/{k}"
            if os.path.exists(src):
                shutil.copy(src, dst)
            n_diff += 1
    print(f"re_v091{tag}: copy {n_copy}, diff {n_diff}, have {len(glob.glob(f\"re_v091{tag}/[0-9]*.seglst.json\"))}")

for tag in b c; do
  echo "== v091$tag pick/norm/merge"
  $PY -u pick_ensemble.py --moss re_v091$tag --cam output/test_cam_wa --out pick_v091$tag --moss-max-turn-rate 0.05 > pick_v091$tag.log 2>&1
  $PY -u normalize_output.py pick_v091$tag norm_v091$tag > norm_v091$tag.log 2>&1
  $PY -u merge_submit.py --dir norm_v091$tag --out submission_v091$tag.json > merge_v091$tag.log 2>&1
  shasum -a 256 submission_v091$tag.json | cut -c1-16
done
echo ASSEMBLE_ALL_DONE
