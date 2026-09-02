"""Script A：全量解码 + gap 捕获（每 session 一次贪心，scores 只在内存中用完即弃）。

输出 per session JSON：{gen_ids, text, gaps:[[tok_idx, chosen, best, gap], ...]}。
gaps 的语义与 v047/spk_ac_probe 相同：段边界 [S] 数字位上，
未用编号的最大 logp − 实际编号 logp（负值域，越接近 0 = 模型越想换人）。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MOSS_REPO = os.environ.get("MOSS_REPO", "/root/autodl-tmp/MOSS-Transcribe-Diarize")


def main() -> int:
    sys.path.insert(0, MOSS_REPO)
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--processor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--sessions", default="")
    args = ap.parse_args()

    import torch
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, prepare_inputs
    from moss_sat import load_model

    model, processor, device, dtype = load_model(args.model, args.processor, "auto")
    tok = processor.tokenizer
    digit_ids = {d: tok.encode(d, add_special_tokens=False)[0] for d in "123456789"}
    id2digit = {v: k for k, v in digit_ids.items()}
    os.makedirs(args.out, exist_ok=True)

    wavs = sorted(p for p in Path(args.wav_dir).glob("*.wav") if not p.name.startswith("._"))
    if args.sessions:
        want = set(args.sessions.split(","))
        wavs = [w for w in wavs if w.stem in want]
    logger.info("共 %d 个 wav", len(wavs))

    for i, wav in enumerate(wavs, 1):
        out_path = Path(args.out) / f"{wav.stem}.json"
        if out_path.exists():
            continue
        messages = build_transcription_messages(str(wav))
        inputs = prepare_inputs(processor, messages, max_length=131072, device=device).to(device)
        prompt_len = int(inputs["attention_mask"][0].sum().item())
        with torch.inference_mode():
            out1 = model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                input_features=inputs["input_features"],
                audio_feature_lengths=inputs["audio_feature_lengths"],
                audio_chunk_mapping=inputs["audio_chunk_mapping"],
                max_new_tokens=args.max_new_tokens, do_sample=False,
                return_dict_in_generate=True, output_scores=True)
        gen_ids = out1.sequences[0][prompt_len:]
        scores = out1.scores
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()

        gaps = []
        used: set[str] = set()
        ids_l = gen_ids.tolist()
        for j, tid in enumerate(ids_l):
            d = id2digit.get(int(tid))
            if d is None or j < 2:
                continue
            if not tok.decode([int(ids_l[j - 2])]).endswith("S"):
                continue
            unused = [dd for dd in "123456789" if dd not in used_so_far_safe(used) and dd != d]
            used.add(d)
            if not unused:
                continue
            lp = torch.log_softmax(scores[j][0].float(), dim=-1)
            best = max(unused, key=lambda dd: float(lp[digit_ids[dd]]))
            gap = float(lp[digit_ids[best]]) - float(lp[digit_ids[d]])
            gaps.append([j, d, best, round(gap, 3)])
        json.dump({"gen_ids": ids_l, "text": text, "gaps": gaps},
                  open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
        logger.info("[%d/%d] %s: %d 边界 top_gap=%s", i, len(wavs), wav.stem,
                    len(gaps), max((g[3] for g in gaps), default=None))
    logger.info("DECODE_GAP_DONE")
    return 0


def used_so_far_safe(used: set) -> set:
    return used


if __name__ == "__main__":
    raise SystemExit(main())
