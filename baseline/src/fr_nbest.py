"""FireRedASR2 n-best 候选生成：每段产出 N 个候选文本（各自按词级时间戳重排到基准段）。

动机（logs/2026-08-08.md §三）：两侧仲裁做到 oracle 也只有 0.951 点，
需 98% 捕获率才够榜首。唯一能**扩大 oracle 本身**的是增加候选数。
冒烟验证：dev 001/069/094 三段 nbest=4 全部 4/4 互不相同，差异落在不确定位置
（如 094「第一晚」vs「低晚」）→ 是有针对性的多样性，不是噪声。
"""
import json, sys, time, re
from pathlib import Path
sys.path.insert(0, '/tmp/FireRedASR2S')
from fireredasr2s.fireredasr2.asr import FireRedAsr2, FireRedAsr2Config
PUNCT = re.compile(r"""[，。！？、；：""''…,.!?;:"'()\[\]【】]""")
TOKEN = re.compile(r"[A-Za-z]+|[0-9]+|[一-鿿]")
pred_dir, wav_dir, out_dir, N = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
out_dir.mkdir(parents=True, exist_ok=True)
m = FireRedAsr2.from_pretrained('aed', '/tmp/FireRedASR2-AED',
    FireRedAsr2Config(use_gpu=False, return_timestamp=True, beam_size=8, nbest=N))
preds = sorted(p for p in pred_dir.glob('*.seglst.json') if p.name.split('.')[0].isdigit())
t0 = time.time()
for i, p in enumerate(preds, 1):
    sid = p.name.split('.')[0]
    if (out_dir / f'{sid}.nbest.json').exists(): continue
    recs = json.loads(p.read_text(encoding='utf-8'))
    r = m.transcribe([sid], [str(wav_dir / f'{sid}.wav')])[0]
    cands = []
    for h in (r.get('nbest') or [{'timestamp': r.get('timestamp') or []}]):
        buckets = {j: [] for j in range(len(recs))}
        for tok, st, en in (h.get('timestamp') or []):
            c = (st + en) / 2
            best, bd = 0, 1e9
            for j, x in enumerate(recs):
                if x['start_time'] <= c <= x['end_time']: best, bd = j, 0; break
                d = min(abs(c - x['start_time']), abs(c - x['end_time']))
                if d < bd: best, bd = j, d
            buckets[best].append(tok)
        seg = []
        for j, x in enumerate(recs):
            toks = [t for w in buckets[j] for t in TOKEN.findall(PUNCT.sub('', w))]
            w = ' '.join(t.lower() if t.isascii() else t for t in toks).strip()
            seg.append(w if w else x['words'])
        cands.append(seg)
    (out_dir / f'{sid}.nbest.json').write_text(json.dumps(cands, ensure_ascii=False), encoding='utf-8')
    el = time.time() - t0
    print(f'[{i}/{len(preds)}] {sid}: {len(cands)} 候选 ({el:.0f}s, 均{el/i:.1f}s)', flush=True)
print('FR NBEST DONE')
