"""归属轴声学探针：用 MOSS 自己给「说话人标记」打 teacher forcing 似然。

## 为什么是这个信号（2026-08-20 查重后的唯一缺口）

`arbitration-signal-must-be-acoustic` 的结论：**只有音频条件的信号有效**，
文本派生信号一律无效。而 `logs/2026-08-08.md` §选源轴给出的统一死因是：

> 能打分的信号要么**看不见说话人标签**（文本似然），要么被塌缩解欺骗。

MOSS 的说话人标记 `[Sxx]` 是它**自己输出序列里的 token**。对它做 teacher forcing
似然，恰好是「**音频条件 + 看得见说话人标签**」—— 正是那条死因指名的缺口。
段级说话人 embedding 已实测够不到（给完美档案 + 只用段内窗，边际命中率仍只有 42%，
低于 50% 拒绝线），因为本赛段太短（每 2.2 秒换人、23% 段短于 1 秒）。
MOSS 似然不受此限：它看的是**整段会话的上下文**，不是孤立的 2 秒音频。

## 先过量级门，再谈阈值

`logs/2026-08-08.md` 立的可复用判据：
文本轴奏效的裁判区分度 **+1.195**；选源轴三个失败信号是 +0.03~+0.08（小 15~40 倍）。
→ **方向对但区分度小一个数量级时，不必扫阈值，直接停。**

本脚本**只产出区分度所需的原始分**，不做任何决策、不扫任何阈值。

## 机制

目标序列格式 `[start][Sxx]text[end]`。数字在该 tokenizer 里**每个都是单 token**
（`processor._get_digit_token_ids` 内有断言），所以说话人 id 的末位数字占一个位置。
在该位置读 logits → 一次前向即得**全部候选说话人的精确对数概率**
（前缀完全相同，故各候选可直接比较，无需逐候选重跑）。

自检：断言该位置的 label token 确实等于当前说话人末位数字的 token id。对不上就报错退出，
不产出可疑数据（"位置找错了还照算" 是这类脚本最容易犯的静默错误）。

用法（GPU）：
    python spk_ac_probe.py --jsonl <probe.jsonl> --model <snap> --out <scores.json>
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MOSS_REPO = os.environ.get("MOSS_REPO", "/root/autodl-tmp/MOSS-Transcribe-Diarize")


def main() -> int:
    sys.path.insert(0, MOSS_REPO)
    ap = argparse.ArgumentParser(description="MOSS 说话人标记的 teacher forcing 似然")
    ap.add_argument("--jsonl", required=True, help="每行含 conversation + meta(段偏移)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_length", type=int, default=8192)
    args = ap.parse_args()

    import torch  # type: ignore[import-not-found]  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]  # noqa: PLC0415

    from finetune import ConversationDataset, DataCollator  # type: ignore  # noqa: PLC0415
    from moss_transcribe_diarize.processing_moss_transcribe_diarize import (  # type: ignore  # noqa: PLC0415
        MossTranscribeDiarizeProcessor,
    )

    processor = MossTranscribeDiarizeProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = processor.tokenizer
    digit_ids = {d: tok.encode(d, add_special_tokens=False)[0] for d in "0123456789"}

    ds = ConversationDataset(args.jsonl)
    meta = [json.loads(line)["meta"] for line in open(args.jsonl, encoding="utf-8") if line.strip()]
    collator = DataCollator(processor, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()

    out = []
    for i in range(len(ds)):
        batch = collator([ds[i]])
        batch = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in batch.items()}
        labels = batch["labels"][0]
        tgt_pos = (labels != -100).nonzero().flatten()
        if len(tgt_pos) == 0:
            logger.warning("%s: target 区为空，跳过", meta[i]["session"])
            continue
        t0 = int(tgt_pos[0])
        with torch.no_grad():
            logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits[0]
        logp = torch.log_softmax(logits.float(), dim=-1)

        recs = []
        bad = 0
        for seg in meta[i]["segs"]:
            p = t0 + seg["tok_off"]          # 该段说话人 id 末位数字在序列中的绝对位置
            if p <= 0 or p >= len(labels):
                bad += 1
                continue
            got = int(batch["input_ids"][0][p])
            want = digit_ids[seg["cur_digit"]]
            if got != want:                   # 自检：位置对不上就不产出该段
                bad += 1
                continue
            row = logp[p - 1]                 # logits[p-1] 预测位置 p 的 token
            recs.append({
                "start_time": seg["start_time"],
                "cur": seg["cur_digit"],
                "logp": {d: float(row[digit_ids[d]]) for d in "123456789"},
            })
        out.append({"session": meta[i]["session"], "segs": recs, "bad": bad})
        logger.info("[%d/%d] %s: %d 段打分，位置自检失败 %d",
                    i + 1, len(ds), meta[i]["session"], len(recs), bad)

    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    tot = sum(len(o["segs"]) for o in out)
    bad = sum(o["bad"] for o in out)
    logger.info("SPK_AC_PROBE DONE %d 段打分 / 自检失败 %d -> %s", tot, bad, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
