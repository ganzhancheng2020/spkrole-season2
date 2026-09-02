"""FireRedASR2 整段转写 + 词级时间戳 + 词级置信度 → 按 MOSS 段聚合。

置信度来自**整段解码**过程（beam search 逐 token 的 exp(log_softmax)），
不是碎片模式，故能反映整段转写文本的真实质量。
每段保存原始置信度列表，便于后续试 mean/min/几何平均等聚合方式。
"""
import json, sys, time, re
from pathlib import Path
sys.path.insert(0, '/tmp/FireRedASR2S')
from fireredasr2s.fireredasr2.asr import FireRedAsr2, FireRedAsr2Config
PUNCT = re.compile(r"""[，。！？、；：""''…,.!?;:"'()\[\]【】]""")
TOKEN = re.compile(r"[A-Za-z]+|[0-9]+|[一-鿿]")
pred_dir, wav_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
m = FireRedAsr2.from_pretrained('aed', '/tmp/FireRedASR2-AED',
                                FireRedAsr2Config(use_gpu=False, return_timestamp=True, beam_size=3))
preds = sorted(p for p in pred_dir.glob('*.seglst.json') if p.name.split('.')[0].isdigit())
t0 = time.time()
for i, p in enumerate(preds, 1):
    sid = p.name.split('.')[0]; o = out_dir / p.name
    if o.exists(): continue
    recs = json.loads(p.read_text(encoding='utf-8'))
    r = m.transcribe([sid], [str(wav_dir / f'{sid}.wav')])[0]
    ts, cs = r.get('timestamp') or [], r.get('conf_seq') or []
    if len(cs) != len(ts): cs = [1.0] * len(ts)
    buckets = {j: ([], []) for j in range(len(recs))}
    for (tok, st, en), cf in zip(ts, cs):
        c = (st + en) / 2
        best, bd = 0, 1e9
        for j, x in enumerate(recs):
            if x['start_time'] <= c <= x['end_time']: best, bd = j, 0; break
            d = min(abs(c - x['start_time']), abs(c - x['end_time']))
            if d < bd: best, bd = j, d
        buckets[best][0].append(tok); buckets[best][1].append(cf)
    out = []
    for j, x in enumerate(recs):
        toks = [t for w in buckets[j][0] for t in TOKEN.findall(PUNCT.sub('', w))]
        w = ' '.join(t.lower() if t.isascii() else t for t in toks).strip()
        out.append({**x, 'words': w if w else x['words'],
                    'fr_conf': buckets[j][1], 'fr_empty': not w})
    o.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    el = time.time() - t0
    print(f'[{i}/{len(preds)}] {sid}: {len(ts)}词 ({el:.0f}s, 均{el/i:.1f}s)', flush=True)
print('FR CONF DONE')
