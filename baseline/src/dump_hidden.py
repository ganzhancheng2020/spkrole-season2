"""方向 3 探针：导出 MOSS 在说话人 token 位置的各层隐状态，作为段级「内部声纹」。

## 与 spk_ac_probe.py 的关系

同一 jsonl（build_spk_probe_jsonl.py 产物）、同一位置自检。区别：spk_ac_probe 读
logits（→ 标签概率，开新人家族已判死）；本脚本读 **hidden_states** ——
模型在「说出」说话人标签**之前**的内部表示。它是音频条件 + 整场上下文的
说话人表征，且与决策同空间（模型自己用它 argmax）。

## 用途（下游分析在本地）

session 内 K=说话人数 聚类 → 与 MOSS 自标签算一致性 → 分歧段是否更可能是
oracle 错标段（lift）。D10 遗留「质上不同、从未实测」的信号源
（MOSS 内部 speaker representation），区别于 ERes2NetV2 声纹（probe 10 判死）。

用法（GPU）：
    python dump_hidden.py --jsonl cv_merged_probe.jsonl --model sim_out_all106 --out spk_hidden.npz
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MOSS_REPO = os.environ.get("MOSS_REPO", "/root/autodl-tmp/MOSS-Transcribe-Diarize")


def main() -> int:
    sys.path.insert(0, MOSS_REPO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个 session（冒烟测试）")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    from finetune import ConversationDataset, DataCollator
    from moss_transcribe_diarize.processing_moss_transcribe_diarize import (
        MossTranscribeDiarizeProcessor,
    )

    if torch.cuda.is_available():
        dev, dtype = "cuda", torch.bfloat16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev, dtype = "mps", torch.float32
    else:
        dev, dtype = "cpu", torch.float32
    logger.info("device=%s dtype=%s", dev, dtype)

    processor = MossTranscribeDiarizeProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = processor.tokenizer
    digit_ids = {d: tok.encode(d, add_special_tokens=False)[0] for d in "0123456789"}

    ds = ConversationDataset(args.jsonl)
    meta = [json.loads(line)["meta"] for line in open(args.jsonl, encoding="utf-8") if line.strip()]
    collator = DataCollator(processor, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype).to(dev).eval()

    metas: list[dict] = []
    vecs: list[np.ndarray] = []
    import time
    n_total = len(ds) if not args.limit else min(args.limit, len(ds))
    for i in range(n_total):
        t00 = time.time()
        batch = collator([ds[i]])
        batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
        labels = batch["labels"][0]
        tgt_pos = (labels != -100).nonzero().flatten()
        if len(tgt_pos) == 0:
            logger.warning("%s: target 区为空，跳过", meta[i]["session"])
            continue
        t0 = int(tgt_pos[0])
        with torch.no_grad():
            out = model(**{k: v for k, v in batch.items() if k != "labels"},
                        output_hidden_states=True)
        hs = out.hidden_states  # tuple[L+1]，各 [1, seq, dim]
        if i == 0:
            logger.info("层数=%d 维度=%d", len(hs), hs[0].shape[-1])

        bad = 0
        for seg in meta[i]["segs"]:
            p = t0 + seg["tok_off"]
            if p <= 0 or p >= len(labels):
                bad += 1
                continue
            got = int(batch["input_ids"][0][p])
            want = digit_ids[seg["cur_digit"]]
            if got != want:
                bad += 1
                continue
            v = np.stack([h[0, p - 1, :].float().cpu().numpy() for h in hs]).astype(np.float16)
            vecs.append(v)
            metas.append({"session": meta[i]["session"], "start_time": seg["start_time"],
                          "cur": seg["cur_digit"]})
        logger.info("[%d/%d] %s: %d 段导出，自检失败 %d（%.1fs）",
                    i + 1, n_total, meta[i]["session"],
                    len([m for m in metas if m["session"] == meta[i]["session"]]),
                    bad, time.time() - t00)

    arr = np.stack(vecs)
    np.savez_compressed(args.out, vecs=arr,
                        meta=json.dumps(metas, ensure_ascii=False))
    tot_var = float(arr.astype(np.float32).std(axis=0).mean())
    logger.info("DUMP_HIDDEN DONE %d 段 × %d 层 × %d 维 -> %s（跨段 std 均值 %.4f，"
                "接近 0 = 常数向量警报）", arr.shape[0], arr.shape[1], arr.shape[2],
                args.out, tot_var)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
