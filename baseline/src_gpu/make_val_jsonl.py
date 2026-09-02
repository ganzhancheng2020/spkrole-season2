"""把仿真验证集（ref.seglst.json + wav/）打包成 MOSS 微调格式的 jsonl。

复用 prep_finetune_data.py 的 build_transcript / DEFAULT_PROMPT，保证目标串与训练端
逐字符一致（否则 loss 不可比）。

用法： python make_val_jsonl.py <sim_dir> <out.jsonl>
"""
import json, sys
from collections import defaultdict
sys.path.insert(0, "/root/autodl-tmp/simrepo/baseline/src")
from prep_finetune_data import DEFAULT_PROMPT, build_transcript

sim_dir, out_path = sys.argv[1], sys.argv[2]
recs = json.load(open(f"{sim_dir}/ref.seglst.json", encoding="utf-8"))
by_sid = defaultdict(list)
for r in recs:
    by_sid[r["session_id"]].append(r)
with open(out_path, "w", encoding="utf-8") as fh:
    for sid in sorted(by_sid):
        fh.write(json.dumps({"conversation": [
            {"role": "user", "message_type": "text", "content": DEFAULT_PROMPT},
            {"role": "user", "message_type": "audio", "content": f"{sim_dir}/wav/{sid}.wav"},
            {"role": "assistant", "message_type": "text", "content": build_transcript(by_sid[sid])},
        ]}, ensure_ascii=False) + "\n")
print(f"{len(by_sid)} 段 -> {out_path}")
