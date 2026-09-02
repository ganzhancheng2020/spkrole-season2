"""第三文本候选：Paraformer 整场转写 + 字级时间戳分桶回已有段（只换 words）。

与 fr_retext.py 同型（全上下文 + 真时间戳），但换一个**完全不同的模型族**
（Paraformer non-autoregressive vs FireRed AED），错误模式独立。

上一次第三源（FireRed 逐段）失败的教训：候选质量有下限。
逐段转写丢上下文 -> 文本差太多 -> 裁判 54 次只判对 4 次。
故第三源必须全上下文。
"""
import json, sys, re, time
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

PUNCT = re.compile(r"""[，。！？、；：""''…,.!?;:"'()\[\]【】\s]""")
TOKEN = re.compile(r"[A-Za-z]+|[0-9]+|[一-鿿]")
pred_dir, wav_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
p = pipeline(task=Tasks.auto_speech_recognition,
             model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
preds = sorted(x for x in pred_dir.glob('*.seglst.json') if x.name.split('.')[0].isdigit())
t0 = time.time()
for i, pp in enumerate(preds, 1):
    sid = pp.name.split('.')[0]
    o = out_dir / pp.name
    if o.exists(): continue
    recs = json.loads(pp.read_text(encoding='utf-8'))
    r = p(str(wav_dir / f'{sid}.wav'))
    d = r[0] if isinstance(r, list) else r
    text = str(d.get('text', '')); ts = d.get('timestamp') or []
    chars = [c for c in PUNCT.sub('', text)]
    n = min(len(chars), len(ts))
    buckets = {j: [] for j in range(len(recs))}
    for k in range(n):
        st, en = ts[k][0] / 1000.0, ts[k][1] / 1000.0
        c = (st + en) / 2
        best, bd = None, 1e9
        for j, x in enumerate(recs):
            if x['start_time'] <= c <= x['end_time']: best, bd = j, 0; break
            dd = min(abs(c - x['start_time']), abs(c - x['end_time']))
            if dd < bd: best, bd = j, dd
        if best is not None: buckets[best].append(chars[k])
    out = []
    for j, x in enumerate(recs):
        toks = [t for w in buckets[j] for t in TOKEN.findall(w)]
        w = ' '.join(t.lower() if t.isascii() else t for t in toks).strip()
        out.append({**x, 'words': w if w else x['words']})
    o.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[{i}/{len(preds)}] {sid}: {n} 字 -> {len(recs)} 段 ({time.time()-t0:.0f}s)', flush=True)
print('PARA_RETEXT_DONE')
