"""v027 阈值扫描：SPLIT_THR 提高，减少误伤。只对 ec1.0 多判人段跑。
"""
import json, glob, os, sys
sys.path.insert(0, 'src')
from spk_relabel import build_embedder, embed_windows, relabel
import os as _os
sv = build_embedder()
thr = _os.environ['SPLIT_THR']
out_dir = f'output/v027_thr{thr}'
os.makedirs(out_dir, exist_ok=True)
candidates = set()
for p in sorted(glob.glob('output/v007_moss/[0-9]*.seglst.json')):
    sid = p.split('/')[-1].replace('.seglst.json','')
    m = len({r['speaker'] for r in json.load(open(p))})
    e = json.load(open(f'output/diar_ec1.0/{sid}.diar.json'))['num_speakers']
    if e > m: candidates.add(sid)
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
