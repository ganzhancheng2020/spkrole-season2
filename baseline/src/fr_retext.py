"""FireRedASR2 整段转写 + 按词级时间戳分配到已有说话人段（单变量：只换 words）。

与 X13 的关键区别：X13 把音频切成 1.9 秒碎片喂 ASR（丢上下文）；
这里整段 42 秒一次转写，再用模型自带的词级时间戳把词分配回原有段边界。
speaker / start_time / end_time 一律不动。
"""
import json
import os
import re
import sys
import time
from pathlib import Path

# FireRedASR2 的源码目录与权重目录。默认沿用历史路径（不传环境变量时行为不变），
# 打包成 xfdata/ 布局时用环境变量指进包内 —— 变量名与 word_arb.py 保持同一套。
# noqa S108：这两个是**只读**的源码/权重目录，不是本进程创建的临时文件。
FIRERED_SRC = os.environ.get("FIRERED_SRC", "/tmp/FireRedASR2S")  # noqa: S108
FIRERED_CKPT = os.environ.get("FIRERED_CKPT", "/tmp/FireRedASR2-AED")  # noqa: S108

# FORCE_RERUN=1 时无视已有产物、逐 session 全部重算（完整复现用）。
# 默认 0 = 断点续跑，长任务中断后可接着跑。全链路脚本会显式置 1。
FORCE_RERUN = os.environ.get("FORCE_RERUN", "0") == "1"

# FIRERED_GPU=1 走 GPU（CPU 实测约 170 s/场，394 场要 18 h；GPU 快一个量级）。
# 默认 0 = CPU，与历史产物同口径。命名沿用 word_arb.py 的 WORDARB_GPU 约定。
USE_GPU = os.environ.get("FIRERED_GPU", "0") == "1"

sys.path.insert(0, FIRERED_SRC)
from fireredasr2s.fireredasr2.asr import (  # type: ignore[import-not-found]  # noqa: E402
    FireRedAsr2,
    FireRedAsr2Config,
)

PUNCT = re.compile(r"""[，。！？、；：""''…,.!?;:"'()\[\]【】]""")
TOKEN = re.compile(r"[A-Za-z]+|[0-9]+|[一-鿿]")
pred_dir, wav_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
cfg = FireRedAsr2Config(use_gpu=USE_GPU, return_timestamp=True, beam_size=3)
m = FireRedAsr2.from_pretrained('aed', FIRERED_CKPT, cfg)
preds = sorted(p for p in pred_dir.glob('*.seglst.json') if p.name.split('.')[0].isdigit())
t0 = time.time()
for i, p in enumerate(preds, 1):
    sid = p.name.split('.')[0]
    o = out_dir / p.name
    if o.exists() and not FORCE_RERUN:
        continue
    recs = json.loads(p.read_text(encoding='utf-8'))
    r = m.transcribe([sid], [str(wav_dir / f'{sid}.wav')])[0]
    ts = r.get('timestamp') or []
    buckets = {j: [] for j in range(len(recs))}
    for tok, st, en in ts:
        c = (st + en) / 2
        best, bd = None, 1e9
        for j, x in enumerate(recs):
            if x['start_time'] <= c <= x['end_time']:
                best, bd = j, 0
                break
            d = min(abs(c - x['start_time']), abs(c - x['end_time']))
            if d < bd:
                best, bd = j, d
        if best is not None:
            buckets[best].append(tok)
    out = []
    for j, x in enumerate(recs):
        toks = [t for w in buckets[j] for t in TOKEN.findall(PUNCT.sub('', w))]
        w = ' '.join(t.lower() if t.isascii() else t for t in toks).strip()
        out.append({**x, 'words': w if w else x['words']})
    # 同时落盘 FireRed 的原始词级时间戳：后续「词级仲裁 / 跨分段重排到 CAM 段」都要用它，
    # 缺了就得整体重跑（2026-08-06 已因此踩过一次坑，见 logs §二十八）
    (out_dir / f'{sid}.ts.json').write_text(
        json.dumps([[tok, round(st,3), round(en,3)] for tok, st, en in ts],
                   ensure_ascii=False), encoding='utf-8')
    o.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    el = time.time() - t0
    print(f'[{i}/{len(preds)}] {sid}: {len(ts)} 词 -> {len(recs)} 段 ({el:.0f}s, 均{el/i:.1f}s)', flush=True)
print('FR RETEXT DONE')
