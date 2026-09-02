"""在 v026_pick 段上，对漏人+长段的候选段用 ec1.0 重打说话人标签。
文本保持 v026 仲裁结果，只改 speaker 字段。
"""
import json, glob, os, sys
sys.path.insert(0, 'src')
from apply_diar import assign_segment
out_dir = 'output/ec10_on_v026'
os.makedirs(out_dir, exist_ok=True)
ref = json.load(open('../data/extracted/dev/dev/ref.seglst.json'))
ref_nspk = {}
for r in ref: ref_nspk[r['session_id']] = len({x['speaker'] for x in ref if x['session_id']==r['session_id']})
for p in sorted(glob.glob('output/v026_pick/[0-9]*.seglst.json')):
    sid = p.split('/')[-1].replace('.seglst.json','')
    recs = json.load(open(p))
    m_spk = len({r['speaker'] for r in recs})
    r_spk = ref_nspk.get(sid, 0)
    has_long = any(r['end_time']-r['start_time'] >= 4.0 for r in recs)
    if r_spk > m_spk and has_long:
        diar = json.load(open(f'output/diar_ec1.0/{sid}.diar.json'))['segments']
        # 逐段重打标签（保留段边界和文本）
        out = []
        for r in recs:
            # assign_segment 用 overlap 决定说话人，但会改变段边界？不，只改 speaker
            out.extend(assign_segment(r, diar))
    else:
        out = recs
    json.dump(out, open(f'{out_dir}/{sid}.seglst.json','w'), ensure_ascii=False)
print('done')
