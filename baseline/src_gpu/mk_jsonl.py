"""SegLST -> MOSS 微调 JSONL。格式与既有 sim_jsonl/train.jsonl 逐字段一致。

用法：
    python mk_jsonl.py <ref.seglst.json> <wav 目录> <out.jsonl> [proto.jsonl]

第 4 个参数可省。省略时用下面的 DEFAULT_PROMPT —— 它就是出货模型
（user_data/model_data/moss_sim_all106）当初实际训练所用的提示词。
"""
import json, sys
from collections import defaultdict

DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，"
    "正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
)

src, wav_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
proto = sys.argv[4] if len(sys.argv) > 4 else None
PROMPT = (json.loads(open(proto, encoding="utf-8").readline())["conversation"][0]["content"]
          if proto else DEFAULT_PROMPT)

def join_tokens(words: str) -> str:
    t = words.split(); out_ = []
    for i, c in enumerate(t):
        if i and t[i-1].isascii() and c.isascii():
            out_.append(" ")
        out_.append(c)
    return "".join(out_)

def build(rs):
    m, parts = {}, []
    for r in sorted(rs, key=lambda r: r["start_time"]):
        tx = join_tokens(r["words"])
        if not tx:
            continue
        if r["speaker"] not in m:
            m[r["speaker"]] = "S%02d" % (len(m) + 1)
        parts.append("[%.2f][%s]%s[%.2f]" % (r["start_time"], m[r["speaker"]], tx, r["end_time"]))
    return "".join(parts)

S = defaultdict(list)
for x in json.load(open(src, encoding="utf-8")):
    S[x["session_id"]].append(x)

n = 0
with open(out, "w", encoding="utf-8") as fh:
    for sid in sorted(S):
        tgt = build(S[sid])
        if not tgt:
            continue
        rec = {"conversation": [
            {"role": "user", "message_type": "text", "content": PROMPT},
            {"role": "user", "message_type": "audio", "content": "%s/%s.wav" % (wav_dir, sid)},
            {"role": "assistant", "message_type": "text", "content": tgt},
        ]}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
print("wrote %d lines -> %s" % (n, out))
