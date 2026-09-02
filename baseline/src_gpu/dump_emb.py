"""导出逐段 ERes2NetV2 嵌入，供机制#2(轮廓系数)/#3(塌缩检测) 本地复算。

判官用 ERes2NetV2 而非 CAM++：CAM 分支的标签来自 CAM++ 聚类，
拿 CAM++ 当裁判是循环论证。
"""
import json,glob,os,sys
import numpy as np, soundfile as sf
SR=16000; BATCH=64
pred_dir, wav_dir, out_npz = sys.argv[1], sys.argv[2], sys.argv[3]
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
pl=pipeline(task=Tasks.speaker_verification, model='iic/speech_eres2netv2_sv_zh-cn_16k-common')
def embed(clips):
    out=[]
    for i in range(0,len(clips),BATCH):
        e=np.asarray(pl(clips[i:i+BATCH], output_emb=True)['embs'],dtype=np.float32)
        out.append(e)
    e=np.concatenate(out,axis=0)
    return e/np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-8)
store={}
for p in sorted(glob.glob(os.path.join(pred_dir,'[0-9]*.seglst.json'))):
    sid=os.path.basename(p).split('.')[0]
    wp=os.path.join(wav_dir,f'{sid}.wav')
    if not os.path.exists(wp): continue
    wav,sr=sf.read(wp,dtype='float32')
    if wav.ndim>1: wav=wav.mean(axis=1)
    R=sorted(json.load(open(p,encoding='utf-8')),key=lambda r:r['start_time'])
    clips,labs=[],[]
    for x in R:
        a,b=max(0,int(x['start_time']*SR)),min(len(wav),int(x['end_time']*SR))
        if b-a<int(0.35*SR): continue
        clips.append(wav[a:b]); labs.append(x['speaker'])
    if len(clips)<2: continue
    store[sid+'_E']=embed(clips)
    store[sid+'_L']=np.array(labs,dtype=object)
    print(sid,len(clips),flush=True)
np.savez_compressed(out_npz, **store)
print('DONE',len(store)//2)
