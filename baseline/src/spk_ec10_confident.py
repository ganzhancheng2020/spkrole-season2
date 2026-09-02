"""只在 ec1.0 确认找回漏判（MOSS 漏人 + ec1.0<=ref）的 15 段做重标签。
其余保留 v026 原样。文本不动，只改 speaker。
"""
import json, glob, os, sys
sys.path.insert(0, 'src')
from apply_diar import assign_segment
out_dir = 'output/ec10_confident'
os.makedirs(out_dir, exist_ok=True)
ref = json.load(open('../data/extracted/dev/dev/ref.seglst.json'))
ref_nspk = {}
for r in ref: ref_nspk[r['session_id']] = len({x['speaker'] for x in ref if x['session_id']==r['session_id']})
# 确认找回的 15 段
confident = set()
for sid, r in ref_nspk.items():
    try:
        m = len({x['speaker'] for x in json.load(open(f'output/v007_moss/{sid}.seglst.json'))})
        e = json.load(open(f'output/diar_ec1.0/{sid}.diar.json'))['num_speakers']
        if e > m and e <= r:
            confident.add(sid)
    except: pass
print(f'自信段: {sorted(confident)}')
for p in sorted(glob.glob('output/v026_pick/[0-9]*.seglst.json')):
    sid = p.split('/')[-1].replace('.seglst.json','')
    recs = json.load(open(p))
    if sid in confident:
        diar = json.load(open(f'output/diar_ec1.0/{sid}.diar.json'))['segments']
        out = [assign_segment(r, diar)[0] for r in recs]
    else:
        out = recs
    json.dump(out, open(f'{out_dir}/{sid}.seglst.json','w'), ensure_ascii=False)
print('done')
