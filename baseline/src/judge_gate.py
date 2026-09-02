"""裁判门换段：FireRed 独立裁判验收解码时换段。

链路：gap 触发（θ）→ 续写重解码 → FireRed 对同一音频 span 给旧/新文本各打
teacher-forcing 分 → 只接受新文本净赢 ≥ margin 的换段。
保存每次 swap 的双打分，margin 扫描离线可做（同一份数据重建多档臂）。

用法（GPU，moss env，cwd=/root/autodl-tmp/ft）：
    python judge_gate.py --wav-dir /root/autodl-tmp/dev_wav \
        --decode-dir dev_decode_gap --ref <ref> --out-root dev_arms_jg --theta -1.5
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

import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MOSS_REPO = os.environ.get("MOSS_REPO", "/root/autodl-tmp/MOSS-Transcribe-Diarize")
BASE_SNAP = os.environ.get(
    "BASE_SNAP",
    "/root/autodl-tmp/hf/hub/models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/"
    "e8681d68e7042738ffca8ac8212bc8fcb1131ab8")
MEET = "/root/autodl-tmp/envs/moss/bin/meeteval-wer"
TS_RE = re.compile(r"\[\d+\.\d{2}\]")
SEG_RE = re.compile(r"\[(\d+\.\d{2})\]\[S(\d{2})\]")
MARGINS = [0.0, 0.05, 0.1, 0.2]


def main() -> int:
    sys.path.insert(0, MOSS_REPO)
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--decode-dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--theta", type=float, default=-1.5)
    ap.add_argument("--max-seg-tokens", type=int, default=200)
    args = ap.parse_args()

    import torch
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, prepare_inputs
    from moss_sat import load_model, to_seglst
    import word_arb as wa  # import 即加载 FireRed（CPU 打分栈）

    ref_by: dict[str, list[dict]] = {}
    for r in json.load(open(args.ref, encoding="utf-8")):
        ref_by.setdefault(r["session_id"], []).append(r)
    td = tempfile.mkdtemp()

    def errs(sid, records):
        if sid not in ref_by:
            return -1
        h, r = f"{td}/h.json", f"{td}/r.json"
        json.dump(records, open(h, "w", encoding="utf-8"), ensure_ascii=False)
        json.dump(ref_by[sid], open(r, "w", encoding="utf-8"), ensure_ascii=False)
        subprocess.run([MEET, "tcpwer", "-r", r, "-h", h, "--collar", "5"],
                       capture_output=True, check=True)
        return int(json.load(open(h.replace(".json", "_tcpwer.json"), encoding="utf-8"))["errors"])

    model, processor, device, dtype = load_model(BASE_SNAP, None, "auto")
    tok = processor.tokenizer
    digit_ids = {d: tok.encode(d, add_special_tokens=False)[0] for d in "123456789"}

    outs = {m: Path(args.out_root) / f"jg_{m}" for m in MARGINS}
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=True)

    for df in sorted(Path(args.decode_dir).glob("*.json")):
        sid = df.stem
        dec = json.load(open(df, encoding="utf-8"))
        gen_ids = dec["gen_ids"]
        text1 = dec["text"]
        gaps = [g for g in dec["gaps"] if g[3] > args.theta]
        segs1 = to_seglst(parse_transcript(text1), sid)
        base_err = errs(sid, segs1)
        if not gaps:
            for m in MARGINS:
                json.dump(segs1, open(outs[m] / f"{sid}.seglst.json", "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
            continue
        wav = Path(args.wav_dir) / f"{sid}.wav"
        info = sf.info(str(wav))
        sr0 = info.samplerate
        inputs = prepare_inputs(processor, build_transcription_messages(str(wav)),
                                max_length=131072, device=device).to(device)
        prompt_len = int(inputs["attention_mask"][0].sum().item())
        prompt_ids = inputs["input_ids"][0][:prompt_len].tolist()
        toks_txt = [tok.decode([int(t)]) for t in gen_ids]
        spans = [(m.start(), m.end(), float(m.group(1))) for m in SEG_RE.finditer(text1)]
        swaps = []   # (seg_i, new_seg_text, s_old, s_new, gap, old_text, new_text)
        for tok_idx, chosen, best, gap in gaps:
            char_pos = sum(len(t) for t in toks_txt[:tok_idx])
            seg_i = sum(1 for s in spans if s[0] < char_pos) - 1
            if seg_i < 0 or seg_i >= len(spans) or any(s["seg_i"] == seg_i for s in swaps):
                continue
            s_start, s_end, ts = spans[seg_i]
            cap_ts = spans[seg_i + 1][2] if seg_i + 1 < len(spans) else None
            nxt = spans[seg_i + 1][0] if seg_i + 1 < len(spans) else len(text1)
            old_body = text1[s_start:nxt]
            old_segs = to_seglst(parse_transcript(old_body), sid)
            old_text = "".join(r["words"] for r in old_segs)
            o_e = cap_ts if cap_ts is not None else ts + 10.0
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
                continue
            new_body = cont_text[:mstop.start()]
            close_ts = cap_ts if cap_ts is not None else float(mstop.group(0)[1:-1])
            new_segs = to_seglst(parse_transcript(f"[{ts}][S{int(best):02d}]{new_body}[{close_ts:.2f}]"), sid)
            new_text = "".join(r["words"] for r in new_segs)
            if not new_text.strip() or not old_text.strip():
                continue
            clip, _ = sf.read(str(wav), start=int(ts * sr0), stop=int(o_e * sr0))
            if len(clip) < int(sr0 * 0.2):
                continue
            enc, mask = wa.make_enc(clip, sr0)
            s_old = wa.score(enc, mask, old_text)
            s_new = wa.score(enc, mask, new_text)
            swaps.append({"seg_i": seg_i, "ts": ts, "cap": o_e, "gap": gap,
                          "s_old": s_old, "s_new": s_new,
                          "new_seg": f"[{ts}][S{int(best):02d}]{new_body}[{close_ts:.2f}]",
                          "old_body": old_body})
        # margin 档建臂
        for m in MARGINS:
            acc = {s["seg_i"]: s for s in swaps if s["s_new"] - s["s_old"] >= m}
            parts, last = [], 0
            for i, (s_start, s_end, ts) in enumerate(spans):
                nxt = spans[i + 1][0] if i + 1 < len(spans) else len(text1)
                parts.append(text1[last:s_start])
                if i in acc:
                    parts.append(acc[i]["new_seg"])
                else:
                    parts.append(text1[s_start:nxt])
                last = nxt
            parts.append(text1[last:])
            text2 = "".join(parts)
            segs2 = to_seglst(parse_transcript(text2), sid)
            json.dump(segs2, open(outs[m] / f"{sid}.seglst.json", "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        json.dump({"base_err": base_err, "swaps": swaps},
                  open(f"{args.out_root}/{sid}.swaps.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        logger.info("%s: base %d错，尝试 %d swap（接受@0: %d）",
                    sid, base_err, len(swaps),
                    sum(1 for s in swaps if s["s_new"] - s["s_old"] >= 0))
    logger.info("JUDGE_GATE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
