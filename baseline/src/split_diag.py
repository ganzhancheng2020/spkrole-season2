import json, glob, sys, tempfile
import numpy as np, soundfile as sf
sys.path.insert(0, 'src')
from spk_relabel import build_embedder, windows
sv = build_embedder()
all_sims = []
for p in sorted(glob.glob('output/v007_moss/[0-9]*.seglst.json')):
    sid = p.split('/')[-1].replace('.seglst.json','')
    recs = json.load(open(p))
    x, sr = sf.read(f'../data/extracted/dev/dev/wav/{sid}.wav', dtype='float32')
    if x.ndim > 1: x = x.mean(axis=1)
    for r in recs:
        if r['end_time']-r['start_time'] < 4.0: continue
        ws = windows(r['start_time'], r['end_time'])
        embs = []
        for w0, w1 in ws:
            clip = x[max(0,int(w0*sr)):min(len(x),int(w1*sr))]
            if len(clip) < int(0.5*sr): continue
            with tempfile.NamedTemporaryFile(suffix='.wav') as fh:
                sf.write(fh.name, clip, sr)
                o = sv([fh.name], output_emb=True)
            e = np.asarray(o['embs'][0], dtype=np.float64)
            embs.append(e/(np.linalg.norm(e)+1e-9))
        for i in range(1, len(embs)):
            all_sims.append(float(embs[i-1] @ embs[i]))
import statistics
print(f'相邻窗相似度: {len(all_sims)} 个')
print(f'min: {min(all_sims):.3f}, p10: {sorted(all_sims)[len(all_sims)//10]:.3f}, median: {statistics.median(all_sims):.3f}, p25: {sorted(all_sims)[len(all_sims)//4]:.3f}')
print(f'<0.5: {sum(1 for s in all_sims if s<0.5)} ({sum(1 for s in all_sims if s<0.5)/len(all_sims)*100:.1f}%)')
print(f'<0.6: {sum(1 for s in all_sims if s<0.6)} ({sum(1 for s in all_sims if s<0.6)/len(all_sims)*100:.1f}%)')
print(f'<0.7: {sum(1 for s in all_sims if s<0.7)} ({sum(1 for s in all_sims if s<0.7)/len(all_sims)*100:.1f}%)')
