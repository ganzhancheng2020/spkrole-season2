"""长段归属可疑度：段嵌入 vs 各说话人档案（留一法）的余量。

靶子（2026-08-25 实测）：>=11 词的长段里只有 14 段标错，却值 -1.723 点，
且 oracle 过 bench 四关（剔 top3 后仍 -1.034）。
短段虽多但每段几乎不值钱（0-3 词桶 65 段只值 -0.021）。

为什么这次段级 SV 嵌入可能行：此前判死的 42% 命中率是**短段**拖的
（每 2.2s 换人、23% 段短于 1s，嵌入退化）。>=11 词 ≈ 4-8s，嵌入可靠。

档案用**留一法**：算某段时，其 assigned speaker 的档案排除该段自身，
否则档案被该段自己污染，永远判自己对。
"""
import json, glob, os, sys
import numpy as np, soundfile as sf
from collections import defaultdict

SR=16000
PRED, WAV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MINW=int(os.environ.get('MINW','11'))
BATCH=int(os.environ.get('BATCH','64'))

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
pl=pipeline(task=Tasks.speaker_verification, model='iic/speech_eres2netv2_sv_zh-cn_16k-common')

def embed(clips):
    out=[]
    for i in range(0,len(clips),BATCH):
        e=np.asarray(pl(clips[i:i+BATCH], output_emb=True)['embs'],dtype=np.float32)
        out.append(e)
    e=np.concatenate(out,axis=0)
    return e/np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-8)

res={}
for p in sorted(glob.glob(os.path.join(PRED,'[0-9]*.seglst.json'))):
    sid=os.path.basename(p).split('.')[0]
    recs=json.load(open(p,encoding='utf-8'))
    wp=os.path.join(WAV,f'{sid}.wav')
    if not os.path.exists(wp): continue
    wav,sr=sf.read(wp,dtype='float32')
    if wav.ndim>1: wav=wav.mean(axis=1)
    clips,keep=[],[]
    for i,r in enumerate(recs):
        a,b=int(r['start_time']*SR),int(r['end_time']*SR)
        a,b=max(0,a),min(len(wav),b)
        if b-a < int(0.35*SR): continue
        clips.append(wav[a:b]); keep.append(i)
    if not clips: continue
    E=embed(clips)
    idx={i:j for j,i in enumerate(keep)}
    byspk=defaultdict(list)
    for i in keep: byspk[recs[i]['speaker']].append(i)
    out=[]
    for i in keep:
        r=recs[i]; w=len(str(r.get('words','')).split())
        if w<MINW: continue
        s=r['speaker']
        own=[j for j in byspk[s] if j!=i]
        if not own: continue                      # 只有一段，无法留一
        prof={s: E[[idx[j] for j in own]].mean(axis=0)}
        for o,js in byspk.items():
            if o!=s: prof[o]=E[[idx[j] for j in js]].mean(axis=0)
        if len(prof)<2: continue
        v=E[idx[i]]
        sim={k: float(np.dot(v,u/max(np.linalg.norm(u),1e-8))) for k,u in prof.items()}
        alt=max((k for k in sim if k!=s), key=lambda k: sim[k])
        out.append({'i':i,'spk':s,'alt':alt,'nw':w,
                    'margin': sim[s]-sim[alt],
                    'sim_own': sim[s], 'sim_alt': sim[alt],
                    'dur': r['end_time']-r['start_time']})
    res[sid]=out
    json.dump(res,open(OUT,'w',encoding='utf-8'),ensure_ascii=False)
    print(f'{sid} 长段 {len(out)}',flush=True)
print('DONE',len(res))
