import json, glob, os, sys
"""只对 MOSS 漏人 + 长段的段，用 ec1.0 segment 模式重打说话人标签。
其余段保留 MOSS 原样。"""
sys.path.insert(0, 'src')
from apply_diar import assign_segment
out_dir = 'output/split_ec10_select'
os.makedirs(out_dir, exist_ok=True)
ref = json.load(open('../data/extracted/dev/dev/ref.seglst.json'))
ref_nspk = {}
for r in ref: ref_nspk[r['session_id']] = len({x['speaker'] for x in ref if x['session_id']==r['session_id']})
for p in sorted(glob.glob('output/v007_moss/[0-9]*.seglst.json')):
    sid = p.split('/')[-1].replace('.seglst.json','')
    recs = json.load(open(p))
    m_spk = len({r['speaker'] for r in recs})
    r_spk = ref_nspk.get(sid, 0)
    has_long = any(r['end_time']-r['start_time'] >= 4.0 for r in recs)
    if r_spk > m_spk and has_long:
        diar = json.load(open(f'output/diar_ec1.0/{sid}.diar.json'))['segments']
        out = [assign_segment(r, diar)[0] for r in recs]
    else:
        out = recs
    json.dump(out, open(f'{out_dir}/{sid}.seglst.json','w'), ensure_ascii=False)
print('done')
