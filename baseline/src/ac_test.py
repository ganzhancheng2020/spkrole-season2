"""test 侧：对 MOSS/FireRed 两个候选做声学打分并按 margin 仲裁，产出最终 MOSS 源。

配方来自 dev 验证（logs/2026-08-06.md §二十七）：margin=0.05，
全部 8 个 margin 均通过 train82/holdout24 双切分验收。
"""
import json, sys, torch, glob, os, time
import soundfile as sf
sys.path.insert(0, '/tmp/FireRedASR2S')
from fireredasr2s.fireredasr2.asr import FireRedAsr2, FireRedAsr2Config
MARGIN = 0.05
m = FireRedAsr2.from_pretrained('aed', '/tmp/FireRedASR2-AED',
                                FireRedAsr2Config(use_gpu=False, return_timestamp=False))
model, tokz, feat = m.model, m.tokenizer, m.feat_extractor
dec = model.decoder

@torch.no_grad()
def score(wav, segs, cands):
    out = []
    x, sr = sf.read(wav, dtype='float32')
    if x.ndim > 1: x = x.mean(axis=1)
    for (st, en), cs in zip(segs, cands):
        a, b = int(st*sr), int(en*sr)
        clip = x[max(0,a):min(len(x),b)]
        if len(clip) < int(0.2*sr) or cs[0] == cs[1]: out.append((0.0, 0.0)); continue
        sf.write('/tmp/_tclip.wav', clip, sr)
        feats, lens, _, _, _ = feat(['/tmp/_tclip.wav'], ['c'])
        enc, _, enc_mask = model.encoder(feats, lens)
        ss = []
        for t in cs:
            _, ids = tokz.tokenize(t) if t else (None, [])
            if not ids: ss.append(-99.0); continue
            ys = torch.tensor([[dec.sos_id] + list(ids)]).long()
            tm = dec.ignored_target_position_is_0(ys, dec.pad_id)
            d = dec.dropout(dec.tgt_word_emb(ys)*dec.scale + dec.positional_encoding(ys))
            for L in dec.layer_stack: d = L.forward(d, enc, tm, enc_mask, cache=None)
            d = dec.layer_norm_out(d)
            lp = torch.log_softmax(dec.tgt_word_prj(d)[:, :-1], -1)
            ss.append(float(lp.gather(-1, ys[:,1:].unsqueeze(-1)).squeeze(-1).mean()))
        out.append(tuple(ss))
    return out

moss_dir, fr_dir, wav_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(out_dir, exist_ok=True)
preds = sorted(p for p in glob.glob(f'{moss_dir}/*.seglst.json') if os.path.basename(p).split('.')[0].isdigit())
t0 = time.time(); nsw = 0
for i, p in enumerate(preds, 1):
    s = os.path.basename(p).split('.')[0]; o = f'{out_dir}/{s}.seglst.json'
    if os.path.exists(o): continue
    M = json.load(open(p))
    fp = f'{fr_dir}/{s}.seglst.json'
    if not os.path.exists(fp) or len(json.load(open(fp))) != len(M):
        json.dump(M, open(o,'w'), ensure_ascii=False, indent=2); continue
    F = json.load(open(fp))
    segs = [(x['start_time'], x['end_time']) for x in M]
    cands = [[''.join(a['words'].split()), ''.join(b['words'].split())] for a,b in zip(M,F)]
    sc = score(f'{wav_dir}/{s}.wav', segs, cands)
    res = []
    for mm, ff, (sa, sb) in zip(M, F, sc):
        use = (sb - sa) >= MARGIN and sa != 0.0
        if use: nsw += 1
        res.append({**mm, 'words': ff['words'] if use else mm['words']})
    json.dump(res, open(o,'w'), ensure_ascii=False, indent=2)
    if i % 20 == 0: print(f'[{i}/{len(preds)}] 换{nsw}段 ({time.time()-t0:.0f}s)', flush=True)
print(f'AC TEST DONE 换 {nsw} 段')
