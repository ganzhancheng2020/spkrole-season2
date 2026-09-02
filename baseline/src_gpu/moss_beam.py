"""MOSS 解码旋钮探针：贪心 -> beam / 采样。

MOSS 一直用 do_sample=False 贪心解码（moss_sat.py:201），从未动过。
它是链路里最强的组件，且这个旋钮影响所有 session（广覆盖）。
inference_utils 里 generation_config 深拷贝自 model.generation_config，
故直接在 model 上设 num_beams 即可注入。
"""
import sys, os, json, time
sys.argv = ['x']
NB = int(os.environ.get('NBEAMS', '4'))
SESS = os.environ.get('SESS', '')
OUT = os.environ.get('OUT', '_cv_beam4')
WAV = '/root/autodl-tmp/dev_wav'
sys.path.insert(0, '/root/autodl-tmp/MOSS-Transcribe-Diarize')
sys.path.insert(0, '/root/autodl-tmp/ft')
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages, generate_transcription, resolve_device)
import importlib.util
spec = importlib.util.spec_from_file_location('ms', '/root/autodl-tmp/ft/moss_sat.py')
ms = importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)
MODEL = os.environ.get('MOSS_MODEL', 'OpenMOSS-Team/MOSS-Transcribe-Diarize')
dev = resolve_device('auto') if hasattr(resolve_device, '__call__') else None
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                             torch_dtype=torch.bfloat16).eval().cuda()
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
if NB > 1:
    model.generation_config.num_beams = NB
    model.generation_config.early_stopping = True
print(f'num_beams={getattr(model.generation_config,"num_beams",1)}', flush=True)
os.makedirs(OUT, exist_ok=True)
sids = SESS.split(',') if SESS else [f'{i:03d}' for i in range(1, 107)]
t0 = time.time()
for i, sid in enumerate(sids, 1):
    o = os.path.join(OUT, f'{sid}.seglst.json')
    if os.path.exists(o): continue
    w = os.path.join(WAV, f'{sid}.wav')
    if not os.path.exists(w): continue
    msgs = build_transcription_messages(w)
    r = generate_transcription(model, proc, msgs, max_new_tokens=2048,
                               do_sample=False, device=next(model.parameters()).device,
                               dtype=torch.bfloat16)
    recs = ms.to_seglst(parse_transcript(r["text"]), sid)
    json.dump(recs, open(o, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'[{i}/{len(sids)}] {sid}: {len(recs)}段 ({time.time()-t0:.0f}s)', flush=True)
print('MOSS_BEAM_DONE')
