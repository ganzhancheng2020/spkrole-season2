"""为 CV 预测目录造同口径的子集 ref。

`oracle_text.py` 内有词数守恒断言（hyp 词数必须等于 ref 词数），
拿全 dev ref（19447 词）比 CV 子集会直接判「探针无效」——那个断言是对的，
这里补上它要的同口径 ref。
"""
from __future__ import annotations

import glob
import json
import os
import sys

pred_dir, out_path = sys.argv[1], sys.argv[2]
ref_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data/extracted/dev/dev/ref.seglst.json")

sids = {os.path.basename(p).split(".")[0]
        for p in glob.glob(os.path.join(pred_dir, "[0-9]*.seglst.json"))}
ref = [r for r in json.load(open(ref_path, encoding="utf-8")) if r["session_id"] in sids]
json.dump(ref, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
print(f"子集 ref：{len(sids)} 段 / {sum(len(r['words'].split()) for r in ref)} 词 -> {out_path}")
