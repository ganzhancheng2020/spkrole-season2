"""音频条件打分：在 FireRedASR2 的声学模型下计算 P(候选文本 | 音频段)。

这是「仲裁信号必须来自音频」结论的直接实现 —— 前 16 次失败的信号全是文本派生的，
看不见「文字是否与实际说出的话相符」；teacher forcing 的 log-prob 看得见。

⚠️ 已知偏置：FireRed 自己的假设由 beam search 选出，天然分高。
该偏置是系统性的，用 train 上校准的 margin 修正。
"""
import json, sys, torch, glob, os, time
import soundfile as sf
sys.path.insert(0, '/tmp/FireRedASR2S')
from fireredasr2s.fireredasr2.asr import FireRedAsr2, FireRedAsr2Config

m = FireRedAsr2.from_pretrained('aed', '/tmp/FireRedASR2-AED',
                                FireRedAsr2Config(use_gpu=False, return_timestamp=False))
model, tokz, feat = m.model, m.tokenizer, m.feat_extractor
dec = model.decoder

@torch.no_grad()
def score(wav_path, seg_list, cand_list):
    """seg_list=[(st,en)], cand_list=[[text_a, text_b]] -> [(score_a, score_b)]"""
    out = []
    x, sr = sf.read(wav_path, dtype='float32')
    for (st, en), cands in zip(seg_list, cand_list):
        a, b = int(st * sr), int(en * sr)
        clip = x[max(0, a):min(len(x), b)]
        if len(clip) < int(0.2 * sr): out.append((0.0, 0.0)); continue
        tmp = '/tmp/_clip.wav'; sf.write(tmp, clip, sr)
        feats, lens, _, _, _ = feat([tmp], ["c"])
        enc, enc_len, enc_mask = model.encoder(feats, lens)
        ss = []
        for t in cands:
            if not t: ss.append(-99.0); continue
            _, ids = tokz.tokenize(t)
            if not ids: ss.append(-99.0); continue
            ys = torch.tensor([[dec.sos_id] + list(ids)]).long()
            tgt_mask = dec.ignored_target_position_is_0(ys, dec.pad_id)
            d = dec.dropout(dec.tgt_word_emb(ys) * dec.scale + dec.positional_encoding(ys))
            for L in dec.layer_stack:
                d = L.forward(d, enc, tgt_mask, enc_mask, cache=None)
            d = dec.layer_norm_out(d)
            logit = dec.tgt_word_prj(d)
            lp = torch.log_softmax(logit[:, :-1], -1)
            tgt = ys[:, 1:]
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            ss.append(float(tok_lp.mean()))
        out.append(tuple(ss))
    return out


import json, glob, os, sys, time
sys.path.insert(0,'src')
from oracle_arbitrate import edit_distance
from oracle_text import assign
from collections import defaultdict
ref=json.load(open('../data/extracted/dev/dev/ref.seglst.json'))
by=defaultdict(list)
for r in ref: by[r['session_id']].append(r)
for v in by.values(): v.sort(key=lambda r:r['start_time'])
sids=sorted(os.path.basename(p).split('.')[0] for p in glob.glob('output/fr_retext/[0-9]*.seglst.json'))
rows=[]; t0=time.time()
for n,s in enumerate(sids,1):
    M=json.load(open(f'output/v007_moss/{s}.seglst.json')); F=json.load(open(f'output/fr_retext/{s}.seglst.json'))
    if len(M)!=len(F): continue
    g=assign(by[s],M)
    segs=[(x['start_time'],x['end_time']) for x in M]
    cands=[[''.join(a['words'].split()), ''.join(b['words'].split())] for a,b in zip(M,F)]
    sc=score(f'../data/extracted/dev/dev/wav/{s}.wav', segs, cands)
    for i,(m,(sa,sb)) in enumerate(zip(M,sc)):
        gold=' '.join(r['words'] for r in sorted(g.get(i,[]),key=lambda r:r['start_time']) if r['words']).split()
        rows.append({'sid':s,'em':edit_distance(m['words'].split(),gold),
                     'ef':edit_distance(F[i]['words'].split(),gold),'g':len(gold),
                     'am':sa,'af':sb,'same':cands[i][0]==cands[i][1]})
    if n%10==0: print(f'[{n}/{len(sids)}] {time.time()-t0:.0f}s', flush=True)
json.dump(rows, open('/tmp/ac_rows.json','w'))
print('AC SCORE DONE', len(rows))
