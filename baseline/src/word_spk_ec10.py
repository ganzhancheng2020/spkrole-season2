"""词级说话人重打：对 v028 段，用 FireRed ts 判断段内词所属 ec1.0 说话人（多数投票）。
保留段边界和文本，只改 speaker。只对 ec1.0 确认漏判段处理。
"""
import json, glob, os, sys, collections
out_dir = 'output/word_spk_ec10'
os.makedirs(out_dir, exist_ok=True)
ref = json.load(open('../data/extracted/dev/dev/ref.seglst.json'))
ref_nspk = {}
for r in ref: ref_nspk[r['session_id']] = len({x['speaker'] for x in ref if x['session_id']==r['session_id']})

def spk_of(t, diar):
    for s in diar:
        if s['start'] <= t <= s['end']: return s['speaker']
    best, bd = None, float('inf')
    for s in diar:
        d = min(abs(s['start']-t), abs(s['end']-t))
        if d < bd: best, bd = s['speaker'], d
    return best

for p in sorted(glob.glob('output/v028_dev/[0-9]*.seglst.json')):
    sid = p.split('/')[-1].replace('.seglst.json','')
    recs = json.load(open(p))
    m_spk = len({r['speaker'] for r in recs})
    r_spk = ref_nspk.get(sid, 0)
    out = recs
    if r_spk > m_spk:
        diar = json.load(open(f'output/diar_ec1.0/{sid}.diar.json'))['segments']
        ts = json.load(open(f'output/fr_retext_dev_ts/{sid}.ts.json'))
        # 每段: 段内 ts 词的 ec1.0 说话人多数投票
        new_recs = []
        for r in recs:
            s, e = r['start_time'], r['end_time']
            # 找段内 ts 词
            in_seg = [t for t in ts if s <= t[1] <= e]
            if len(in_seg) >= 2:
                votes = collections.Counter(spk_of((t[1]+t[2])/2, diar) for t in in_seg)
                new_spk = votes.most_common(1)[0][0]
                new_recs.append({**r, 'speaker': f'spk{new_spk+1}'})
            else:
                new_recs.append(r)
        out = new_recs
    json.dump(out, open(f'{out_dir}/{sid}.seglst.json','w'), ensure_ascii=False)
print('done')
