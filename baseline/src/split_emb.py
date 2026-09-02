"""段拆分探针 v2：长段内用 ERes2NetV2 嵌入检测说话人切换点。
只对 MOSS 漏人 + 长段(>=4s) 的段处理。检测到明显切换点才拆分，保守。
"""
import json, glob, os, sys, tempfile
import numpy as np, soundfile as sf
def main():
    sys.path.insert(0, 'src')
    from spk_relabel import build_embedder, windows, embed_windows
    sv = build_embedder()
    out_dir = 'output/split_emb'
    os.makedirs(out_dir, exist_ok=True)
    ref = json.load(open('../data/extracted/dev/dev/ref.seglst.json'))
    ref_nspk = {}
    for r in ref: ref_nspk[r['session_id']] = len({x['speaker'] for x in ref if x['session_id']==r['session_id']})
    for p in sorted(glob.glob('output/v007_moss/[0-9]*.seglst.json')):
        sid = p.split('/')[-1].replace('.seglst.json','')
        recs = json.load(open(p))
        m_spk = len({r['speaker'] for r in recs})
        r_spk = ref_nspk.get(sid, 0)
        # 只处理漏人 + 长段
        if r_spk <= m_spk:
            json.dump(recs, open(f'{out_dir}/{sid}.seglst.json','w'), ensure_ascii=False)
            continue
        has_long = any(r['end_time']-r['start_time'] >= 4.0 for r in recs)
        if not has_long:
            json.dump(recs, open(f'{out_dir}/{sid}.seglst.json','w'), ensure_ascii=False)
            continue
        # 加载音频 + 逐段检测
        x, sr = sf.read(f'../data/extracted/dev/dev/wav/{sid}.wav', dtype='float32')
        if x.ndim > 1: x = x.mean(axis=1)
        out_recs = []
        for r in recs:
            dur = r['end_time'] - r['start_time']
            if dur < 4.0:
                out_recs.append(r); continue
            # 段内切 0.5s 窗取嵌入
            ws = windows(r['start_time'], r['end_time'])
            embs = []
            for w0, w1 in ws:
                clip = x[max(0,int(w0*sr)):min(len(x),int(w1*sr))]
                if len(clip) < int(0.5*sr): continue
                with tempfile.NamedTemporaryFile(suffix='.wav') as fh:
                    sf.write(fh.name, clip, sr)
                    o = sv([fh.name], output_emb=True)
                e = np.asarray(o['embs'][0], dtype=np.float64)
                embs.append((w0, w1, e/(np.linalg.norm(e)+1e-9)))
            if len(embs) < 2:
                out_recs.append(r); continue
            # 相邻窗相似度，找谷值（切换点）
            sims = []
            for i in range(1, len(embs)):
                sims.append(float(embs[i-1][2] @ embs[i][2]))
            min_sim = min(sims)
            if min_sim < 0.3:  # 高度不相似 = 切换点
                idx = sims.index(min_sim)
                split_t = (embs[idx][1] + embs[idx+1][0]) / 2
                # 拆成两段，第二段标新说话人
                w = r['words'].split()
                n = len(w)
                # 按时间比例切词
                t1 = split_t - r['start_time']
                t2 = r['end_time'] - split_t
                take = max(1, round(n * t1/(t1+t2)))
                take = min(take, n-1)
                r1 = {**r, 'end_time': split_t, 'words': ' '.join(w[:take])}
                r2 = {**r, 'start_time': split_t, 'words': ' '.join(w[take:]), 'speaker': r['speaker']+'_s'}
                out_recs.extend([r1, r2])
            else:
                out_recs.append(r)
        json.dump(out_recs, open(f'{out_dir}/{sid}.seglst.json','w'), ensure_ascii=False)
    print('done')
if __name__ == '__main__': main()
