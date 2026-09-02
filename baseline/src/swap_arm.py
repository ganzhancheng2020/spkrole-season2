"""Script B：按 θ 档做外科手术换数字重解码，产出 arm 目录（SegLST）。

--thetas 支持多档一次跑完（每 session 的 audio features 只准备一次）。
另产 θ=none 的纯解码基线臂（gate 用）。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MOSS_REPO = os.environ.get("MOSS_REPO", "/root/autodl-tmp/MOSS-Transcribe-Diarize")
TS_RE = re.compile(r"\[\d+\.\d{2}\]")
SEG_RE = re.compile(r"\[(\d+\.\d{2})\]\[S(\d{2})\]")


def main() -> int:
    sys.path.insert(0, MOSS_REPO)
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--decode-dir", required=True, help="decode_gap.py 输出")
    ap.add_argument("--model", required=True)
    ap.add_argument("--processor", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--thetas", default="none,-2,-3,-4", help="逗号分隔；none=纯解码基线")
    ap.add_argument("--max-seg-tokens", type=int, default=200)
    args = ap.parse_args()

    import torch
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, prepare_inputs
    from moss_sat import load_model, to_seglst

    thetas = args.thetas.split(",")
    num_thetas = [t for t in thetas if t != "none"]
    model, processor, device, dtype = load_model(args.model, args.processor, "auto")
    tok = processor.tokenizer
    digit_ids = {d: tok.encode(d, add_special_tokens=False)[0] for d in "123456789"}
    outs = {t: Path(args.out_root) / f"arm_{t}" for t in thetas}
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=True)

    dec_files = sorted(Path(args.decode_dir).glob("*.json"))
    logger.info("共 %d 个解码文件，θ=%s", len(dec_files), thetas)
    for i, df in enumerate(dec_files, 1):
        sid = df.stem
        dec = json.load(open(df, encoding="utf-8"))
        gen_ids = dec["gen_ids"]
        text1 = dec["text"]
        wav = Path(args.wav_dir) / f"{sid}.wav"
        if not wav.exists():
            continue
        messages = build_transcription_messages(str(wav))
        inputs = prepare_inputs(processor, messages, max_length=131072, device=device).to(device)
        prompt_len = int(inputs["attention_mask"][0].sum().item())
        prompt_ids = inputs["input_ids"][0][:prompt_len].tolist()

        # 纯解码基线臂
        base_segs = to_seglst(parse_transcript(text1), sid)
        json.dump(base_segs, open(outs["none"] / f"{sid}.seglst.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

        toks_txt = None
        for theta in num_thetas:
            th = float(theta)
            gaps = [g for g in dec["gaps"] if g[3] > th]
            if not gaps:
                json.dump(base_segs, open(outs[theta] / f"{sid}.seglst.json", "w",
                                          encoding="utf-8"), ensure_ascii=False, indent=1)
                continue
            if toks_txt is None:
                toks_txt = [tok.decode([int(t)]) for t in gen_ids]
            spans = [(m.start(), m.end(), float(m.group(1))) for m in SEG_RE.finditer(text1)]
            repl: dict[int, str] = {}   # seg_i -> 新段全文
            for tok_idx, chosen, best, gap in gaps:
                char_pos = sum(len(t) for t in toks_txt[:tok_idx])
                seg_i = sum(1 for s in spans if s[0] < char_pos) - 1
                if seg_i < 0 or seg_i >= len(spans) or seg_i in repl:
                    continue
                s_start, s_end, ts = spans[seg_i]
                cap_ts = spans[seg_i + 1][2] if seg_i + 1 < len(spans) else None
                cont = model.generate(
                    input_ids=torch.tensor([prompt_ids + gen_ids[:tok_idx] + [digit_ids[best]]],
                                           device=device),
                    attention_mask=torch.ones(1, prompt_len + tok_idx + 1, device=device,
                                              dtype=torch.long),
                    input_features=inputs["input_features"],
                    audio_feature_lengths=inputs["audio_feature_lengths"],
                    audio_chunk_mapping=inputs["audio_chunk_mapping"],
                    max_new_tokens=args.max_seg_tokens, do_sample=False)
                cont_text = tok.decode(cont[0][prompt_len + tok_idx + 1:], skip_special_tokens=True)
                mstop = TS_RE.search(cont_text)
                if not mstop:
                    continue  # 模型没给出 [end]，无法定界，放弃
                new_body = cont_text[:mstop.start()]
                if not new_body.strip():
                    continue
                # 时间框：end 一律取原流下一个边界（有界伤害，不吞段、不倒流）
                end_ts = cap_ts if cap_ts is not None else float(mstop.group(0)[1:-1])
                repl[seg_i] = f"[{ts}][S{int(best):02d}]{new_body}[{end_ts:.2f}]"
            # 按 span 位置重建（原段全保留，只换被触发的段体）
            parts, last = [], 0
            for i, (s_start, s_end, ts) in enumerate(spans):
                nxt = spans[i + 1][0] if i + 1 < len(spans) else len(text1)
                parts.append(text1[last:s_start])
                parts.append(repl[i] if i in repl else text1[s_start:nxt])
                last = nxt
            parts.append(text1[last:])
            text2 = "".join(parts)
            segs2 = to_seglst(parse_transcript(text2), sid)
            json.dump(segs2, open(outs[theta] / f"{sid}.seglst.json", "w",
                                  encoding="utf-8"), ensure_ascii=False, indent=1)
            logger.info("[%d/%d] %s θ=%s: %d 段替换", i, len(dec_files), sid, theta, len(repl))
    logger.info("SWAP_ARMS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
