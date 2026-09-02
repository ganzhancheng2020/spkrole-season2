"""v027 限制版：只在 ec1.0 确认漏判段（MOSS 人数 < ec1.0 人数 且 ec1.0 <= 某阈值）跑 subprofile。
无 ref 代理：用 MOSS 人数 vs ec1.0 人数差识别漏判段。
其他段保留 v026 原样。
"""
import json, glob, os, sys
sys.path.insert(0, 'src')
from spk_relabel import build_embedder, embed_windows, relabel
import soundfile as sf, numpy as np

out_dir = 'output/v027_restricted'
os.makedirs(out_dir, exist_ok=True)
sv = build_embedder()

# 无 ref 代理：MOSS 人数 < ec1.0 人数 的段（可能漏判）
candidates = set()
for p in sorted(glob.glob('output/v007_moss/[0-9]*.seglst.json')):
    sid = p.split('/')[-1].replace('.seglst.json','')
    m = len({r['speaker'] for r in json.load(open(p))})
    e = json.load(open(f'output/diar_ec1.0/{sid}.diar.json'))['num_speakers']
    if e > m:  # ec1.0 认为更多说话人
        candidates.add(sid)
print(f'候选漏判段（ec1.0 多判人）: {len(candidates)} 段')

for p in sorted(glob.glob('output/v026_pick/[0-9]*.seglst.json')):
    sid = p.split('/')[-1].replace('.seglst.json','')
    recs = json.load(open(p))
    if sid in candidates:
        embs, owner = embed_windows(sv, f'../data/extracted/dev/dev/wav/{sid}.wav', recs)
        labels, n = relabel(recs, embs, owner)
        out = [{**r, 'speaker': s} for r, s in zip(recs, labels)]
    else:
        out = recs
    json.dump(out, open(f'{out_dir}/{sid}.seglst.json','w'), ensure_ascii=False)
print('done')
