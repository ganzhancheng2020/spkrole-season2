"""P2-v1 外科手术探针：现有段边界处换数字重解码（单段替换，不动下游）。

## 与 probe_forced_redecode.py（v0）的关系

v0 证明：强制新说话人假设下模型能原生转写出对的话（071 −40 错）；
但整段下游重解码高方差（+12~+37 的归属噪声）。
v1 把手术缩小到**单个段**：模型流里本来就有 [end][ts][Sxx] 边界（12 段/session，
多为同说话人内部切分）；在高 gap 边界处把数字 token 换成未用编号，只重解码该段，
前后原样保留。spk_open_new 的死因（新说话人拿到的是旧假设的词）由此根除。

## 触发器

gap = max(未用编号 logp) − 实际编号 logp（该段 [S] 数字位的 logits，
output_scores=True 一次贪心全部拿到）。θ 扫 {2, 4}。

用法（GPU）：
    python probe_seg_forced_digit.py --wav-dir /root/autodl-tmp/dev_wav \
        --ref <ref> --sessions 037,085,... --out /tmp/p2v1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MOSS_REPO = os.environ.get("MOSS_REPO", "/root/autodl-tmp/MOSS-Transcribe-Diarize")
BASE_SNAP = os.environ.get(
    "BASE_SNAP",
    "/root/autodl-tmp/hf/hub/models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/"
    "e8681d68e7042738ffca8ac8212bc8fcb1131ab8",
)
MEET = "/root/autodl-tmp/envs/moss/bin/meeteval-wer"
SEG_RE = re.compile(r"\[(\d+\.\d{2})\]\[S(\d{2})\]")
TS_RE = re.compile(r"\[\d+\.\d{2}\]")


def evaluate_session(sid, records, ref, td):
    h, r = f"{td}/h.json", f"{td}/r.json"
    json.dump(records, open(h, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(ref, open(r, "w", encoding="utf-8"), ensure_ascii=False)
    subprocess.run([MEET, "tcpwer", "-r", r, "-h", h, "--collar", "5"],
                   capture_output=True, check=True)
    return int(json.load(open(h.replace(".json", "_tcpwer.json"), encoding="utf-8"))["errors"])


def main() -> int:
    sys.path.insert(0, MOSS_REPO)
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thetas", default="2,4")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-seg-tokens", type=int, default=200)
    args = ap.parse_args()

    import torch
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages, prepare_inputs,
    )
    from moss_sat import load_model, to_seglst

    ref_by: dict[str, list[dict]] = {}
    for r in json.load(open(args.ref, encoding="utf-8")):
        ref_by.setdefault(r["session_id"], []).append(r)

    model, processor, device, dtype = load_model(BASE_SNAP, None, "auto")
    tok = processor.tokenizer
    digit_ids = {d: tok.encode(d, add_special_tokens=False)[0] for d in "123456789"}
    id2digit = {v: k for k, v in digit_ids.items()}
    os.makedirs(args.out, exist_ok=True)
    td = tempfile.mkdtemp()
    thetas = [float(x) for x in args.thetas.split(",")]

    for sid in args.sessions.split(","):
        wav = Path(args.wav_dir) / f"{sid}.wav"
        ref = ref_by.get(sid, [])
        if not wav.exists() or not ref:
            continue
        messages = build_transcription_messages(str(wav))
        inputs = prepare_inputs(processor, messages, max_length=131072, device=device).to(device)
        prompt_len = int(inputs["attention_mask"][0].sum().item())
        gen_kw = {
            "input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"],
            "input_features": inputs["input_features"],
            "audio_feature_lengths": inputs["audio_feature_lengths"],
            "audio_chunk_mapping": inputs["audio_chunk_mapping"],
            "max_new_tokens": args.max_new_tokens, "do_sample": False,
        }
        with torch.inference_mode():
            out1 = model.generate(return_dict_in_generate=True, output_scores=True, **gen_kw)
        gen_ids = out1.sequences[0][prompt_len:]
        scores = out1.scores  # tuple[len(gen_ids)] each (1, vocab)
        text1 = tok.decode(gen_ids, skip_special_tokens=True).strip()
        segs1 = to_seglst(parse_transcript(text1), sid)
        base_err = evaluate_session(sid, segs1, ref, td)

        # 找流中的 [S] 数字 token 位置（段边界），算 gap
        digit_pos = []  # (stream_token_idx, chosen_digit)
        for i, tid in enumerate(gen_ids.tolist()):
            d = id2digit.get(int(tid))
            if d is not None and i >= 2:
                prev2 = tok.decode([int(gen_ids[i - 2])])
                if prev2.endswith("S") or "[S" in prev2:
                    digit_pos.append((i, d))
        if not digit_pos:
            logger.info("%s: 无边界，跳过", sid)
            continue
        used_so_far: set[str] = set()
        cands = []  # (gap, tok_idx, chosen, best_unused)
        for idx, d in digit_pos:
            unused = [dd for dd in "123456789" if dd not in used_so_far and dd != d]
            used_so_far.add(d)
            if not unused:
                continue
            lp = torch.log_softmax(scores[idx][0].float(), dim=-1)
            chosen_lp = float(lp[digit_ids[d]])
            best = max(unused, key=lambda dd: float(lp[digit_ids[dd]]))
            gap = float(lp[digit_ids[best]]) - chosen_lp
            cands.append((gap, idx, d, best))
        # 段结构（文本级），供替换
        seg_spans = [(m.start(), m.end(), m.group(1), m.group(2)) for m in SEG_RE.finditer(text1)]

        results = {"base_err": base_err, "n_bounds": len(cands)}
        for theta in thetas:
            trig = [c for c in cands if c[0] > theta]
            if not trig:
                results[f"theta{theta}"] = {"err": base_err, "n": 0}
                continue
            text2 = text1
            notes = []
            for gap, tok_idx, chosen, best in trig:
                # 该边界在文本中的段（第几个 [Sxx]）
                order = [i for i, (gi, _) in enumerate(
                    [(i, id2digit.get(int(t))) for i, t in enumerate(gen_ids.tolist())
                     if id2digit.get(int(t)) and tok.decode([int(gen_ids[max(i - 2, 0)])]).endswith("S")]
                ) ]
                # 简化：按 token 位置找对应文本段序号
                char_pos = sum(len(tok.decode([int(t)])) for t in gen_ids[:tok_idx])
                seg_i = sum(1 for m in SEG_RE.finditer(text1) if m.start() < char_pos) - 1
                if seg_i < 0 or seg_i >= len(seg_spans):
                    continue
                s_start, s_end, ts, old_spk = seg_spans[seg_i]
                # 该段在文本中的完整范围：到下一个 [ts]（含其 [end]）
                nxt = TS_RE.search(text1, s_end)
                body_end = nxt.start() if nxt else len(text1)
                old_body = text1[s_start:body_end]
                # 重解码该段：前缀（含 [S，不含数字）→ 换数字 → 续写
                prefix_ids = gen_ids[:tok_idx].tolist()  # 到数字前
                cont = model.generate(
                    input_ids=torch.tensor(
                        [inputs["input_ids"][0][:prompt_len].tolist() + prefix_ids + [digit_ids[best]]],
                        device=gen_ids.device),
                    attention_mask=torch.ones(1, prompt_len + len(prefix_ids) + 1,
                                              device=gen_ids.device, dtype=torch.long),
                    input_features=inputs["input_features"],
                    audio_feature_lengths=inputs["audio_feature_lengths"],
                    audio_chunk_mapping=inputs["audio_chunk_mapping"],
                    max_new_tokens=args.max_seg_tokens, do_sample=False)
                cont_text = tok.decode(cont[0][prompt_len + len(prefix_ids) + 1:],
                                       skip_special_tokens=True)
                mstop = TS_RE.search(cont_text)
                new_body = cont_text[:mstop.start()] if mstop else cont_text
                if not new_body.strip():
                    continue
                new_seg = f"[{ts}][S{best}]{new_body}"
                text2 = text2.replace(old_body, new_seg, 1)
                notes.append((gap, old_spk, best, new_body[:60]))
            segs2 = to_seglst(parse_transcript(text2), sid)
            err2 = evaluate_session(sid, segs2, ref, td)
            results[f"theta{theta}"] = {"err": err2, "n": len(notes), "notes": notes}
            logger.info("%s θ=%g: %d 错（base %d）Δ%+d，触发 %d 段",
                        sid, theta, err2, base_err, err2 - base_err, len(notes))
            for g, o, b, nb in notes[:3]:
                logger.info("   gap=%.2f S%s→S%s 新段: %s", g, o, b, nb)
        json.dump(results, open(f"{args.out}/{sid}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    logger.info("P2V1_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
