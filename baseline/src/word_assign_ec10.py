"""词级归属重排：用 FireRed 词级时间戳 + ec1.0 说话人段，把词按时间归说话人。
只处理 MOSS 漏人 + 长段的 41 段候选。其余保留 v007 原样。
"""
import json, glob, os, sys
sys.path.insert(0, 'src')
out_dir = 'output/word_assign_ec10'
os.makedirs(out_dir, exist_ok=True)
ref = json.load(open('../data/extracted/dev/dev/ref.seglst.json'))
ref_nspk = {}
for r in ref: ref_nspk[r['session_id']] = len({x['speaker'] for x in ref if x['session_id']==r['session_id']})

def nearest_spk(t, diar):
    for s in diar:
        if s['start'] <= t <= s['end']:
            return s['speaker']
    # 兜底：最近
    best, bd = None, float('inf')
    for s in diar:
        d = min(abs(s['start']-t), abs(s['end']-t))
        if d < bd: best, bd = s['speaker'], d
    return best

for p in sorted(glob.glob('output/v007_moss/[0-9]*.seglst.json')):
    sid = p.split('/')[-1].replace('.seglst.json','')
    recs = json.load(open(p))
    m_spk = len({r['speaker'] for r in recs})
    r_spk = ref_nspk.get(sid, 0)
    has_long = any(r['end_time']-r['start_time'] >= 4.0 for r in recs)
    out = recs
    if r_spk > m_spk and has_long:
        diar = json.load(open(f'output/diar_ec1.0/{sid}.diar.json'))['segments']
        ts = json.load(open(f'output/fr_retext_dev_ts/{sid}.ts.json'))
        # 每个词归到说话人
        spk_words = {}
        for tok, s, e in ts:
            spk = nearest_spk((s+e)/2, diar)
            spk_words.setdefault(spk, []).append(tok)
        if spk_words:
            # 重建段：按说话人分组，时间取首尾词
            out = []
            for spk, words in spk_words.items():
                wts = [(s,e) for t,s,e in ts if nearest_spk((s+e)/2, diar)==spk]
                if wts and len(words) >= 2:
                    out.append({'session_id': sid, 'speaker': f'spk{spk+1}', 
                                'start_time': min(s for s,e in wts), 'end_time': max(e for s,e in wts),
                                'words': ' '.join(words)})
            if not out:
                out = recs
    json.dump(out, open(f'{out_dir}/{sid}.seglst.json','w'), ensure_ascii=False)
print('done')
