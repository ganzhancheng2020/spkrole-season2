"""验证集 NLL —— D11 的核心缺件：finetune.py 原本没有任何 eval，无法判断过训练。

复用 finetune.py 的 ConversationDataset / DataCollator，保证与训练端**同一套**
分词、prompt 屏蔽（labels[:len(prompt)] = -100）与拼接逻辑，loss 才可比。

指标用 **token 加权 NLL**（总 loss / 总有效 token），不是 batch 均值 ——
段长差异大时 batch 均值会给短段过高权重。

用法： python eval_loss.py <val.jsonl> <ckpt_dir> [<ckpt_dir> ...]
"""
import sys, json, torch
sys.path.insert(0, "/root/autodl-tmp/MOSS-Transcribe-Diarize")
from finetune import ConversationDataset, DataCollator
from transformers import AutoModelForCausalLM
# 必须走 finetune.py 同一条 import 路径（安装包版，而非 HF 快照的 remote code）——
# 两者的 processor 接口不同（expand_audio_token vs _expand_audio_token），
# 走错会让 prompt 屏蔽长度不一致，loss 与训练端不可比。
from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor

SNAP = "/root/autodl-tmp/hf/hub/models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/e8681d68e7042738ffca8ac8212bc8fcb1131ab8/"
val_path, ckpts = sys.argv[1], sys.argv[2:]
proc = MossTranscribeDiarizeProcessor.from_pretrained(SNAP, trust_remote_code=True)
ds = ConversationDataset(val_path)
coll = DataCollator(proc, 8192)  # 与 run_sim_ft.sh 的 --max_length 一致
print(f"验证集 {len(ds)} 段", flush=True)

for ck in ckpts:
    model = AutoModelForCausalLM.from_pretrained(
        ck, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    tot_loss = tot_tok = 0.0
    with torch.no_grad():
        for i in range(len(ds)):
            batch = coll([ds[i]])
            batch = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in batch.items()}
            out = model(**batch)
            ntok = int((batch["labels"] != -100).sum())
            tot_loss += float(out.loss) * ntok
            tot_tok += ntok
    print(f"{ck.rstrip(chr(47)).split(chr(47))[-1]:20s} val NLL = {tot_loss/tot_tok:.5f}  ({int(tot_tok)} tokens)", flush=True)
    del model; torch.cuda.empty_cache()
