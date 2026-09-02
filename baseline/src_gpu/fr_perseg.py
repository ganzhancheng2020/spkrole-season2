"""第三文本候选：用同一个 FireRedASR2-AED **逐段单独**转写。

与 fr_retext.py 的单变量差别：
  fr_retext = 整场 42s 一次转写 + 按词级时间戳分桶回段（有上下文，但对齐靠时间戳）
  本脚本   = 每段单独喂模型（丢上下文，但**天然精确对齐**，不需要时间戳）

目的不是更强，是**错误模式不同**。word_arb 逐块选优，
一个整体更差但错法不同的源照样有贡献 —— FireRed 整体比 MOSS 差 1.68 点，
却帮 word_arb 拿了 0.931 点，就是证据。
"""
import json, sys, time, re, os
from pathlib import Path
import soundfile as sf
sys.path.insert(0, '/tmp/FireRedASR2S')
from fireredasr2s.fireredasr2.asr import FireRedAsr2, FireRedAsr2Config

PUNCT = re.compile(r"""[，。！？、；：""''…,.!?;:"'()\[\]【】]""")
TOKEN = re.compile(r"[A-Za-z]+|[0-9]+|[一-鿿]")
pred_dir, wav_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
TMP = Path('/tmp/_seg_clips'); TMP.mkdir(exist_ok=True)
cfg = FireRedAsr2Config(use_gpu=True, beam_size=3)
m = FireRedAsr2.from_pretrained('aed', '/tmp/FireRedASR2-AED', cfg)
preds = sorted(p for p in pred_dir.glob('*.seglst.json') if p.name.split('.')[0].isdigit())
SR = 16000
t0 = time.time()
for i, p in enumerate(preds, 1):
    sid = p.name.split('.')[0]
    o = out_dir / p.name
    if o.exists(): continue
    recs = json.loads(p.read_text(encoding='utf-8'))
    wav, sr = sf.read(str(wav_dir / f'{sid}.wav'), dtype='float32')
    if wav.ndim > 1: wav = wav.mean(axis=1)
    ids, paths, keep = [], [], []
    for j, x in enumerate(recs):
        a, b = max(0, int(x['start_time']*SR)), min(len(wav), int(x['end_time']*SR))
        if b - a < int(0.20*SR):        # 太短喂不了模型，保留原文本
            continue
        fp = TMP / f'{sid}_{j}.wav'
        sf.write(str(fp), wav[a:b], SR)
        ids.append(f'{sid}_{j}'); paths.append(str(fp)); keep.append(j)
    texts = {}
    if paths:
        for r in m.transcribe(ids, paths):
            texts[r['uttid']] = r.get('text', '')
    out = []
    for j, x in enumerate(recs):
        t = texts.get(f'{sid}_{j}', '')
        toks = [tk for w in t.split() for tk in TOKEN.findall(PUNCT.sub('', w))] if t else []
        if not toks: toks = TOKEN.findall(PUNCT.sub('', t)) if t else []
        w = ' '.join(tk.lower() if tk.isascii() else tk for tk in toks).strip()
        out.append({**x, 'words': w if w else x['words']})
    for fp in TMP.glob(f'{sid}_*.wav'): fp.unlink()
    o.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    el = time.time() - t0
    print(f'[{i}/{len(preds)}] {sid}: {len(keep)}/{len(recs)} 段 ({el:.0f}s, 均{el/i:.1f}s)', flush=True)
print('FR_PERSEG_DONE')
