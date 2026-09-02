"""MOSS teacher-forcing acoustic likelihood judge for word-arbitration candidates.

Second, independent acoustic judge (the existing one in word_arb.py is FireRedASR2-AED).
Scores log P(text | audio span) with the MOSS-Transcribe-Diarize base snapshot,
teacher forcing only (no free decoding, no sampling).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

SNAP = "/root/autodl-tmp/hf/hub/models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/e8681d68e7042738ffca8ac8212bc8fcb1131ab8/"
SR = 16000


def detok(words: str) -> str:
    """SegLST space-separated tokens -> natural text (space only between ASCII words)."""
    toks = str(words).split()
    out = []
    for i, t in enumerate(toks):
        if i and t.isascii() and toks[i - 1].isascii():
            out.append(" ")
        out.append(t)
    return "".join(out)


def load_dir(d: str) -> dict:
    recs = {}
    for f in glob.glob(os.path.join(d, "[0-9]*.seglst.json")):
        for s in json.load(open(f, encoding="utf-8")):
            k = (s["session_id"], round(float(s["start_time"]), 3), round(float(s["end_time"]), 3))
            recs[k] = s
    return recs


class Judge:
    def __init__(self, dtype_name: str, mode: str):
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]
        self.model = (
            AutoModelForCausalLM.from_pretrained(SNAP, trust_remote_code=True, dtype="auto", local_files_only=True)
            .to(dtype=self.dtype).to(self.device).eval()
        )
        self.proc = AutoProcessor.from_pretrained(SNAP, trust_remote_code=True, local_files_only=True)
        if not hasattr(self.proc, "feature_extractor"):
            raise SystemExit("processor degraded to tokenizer")
        assert int(self.proc.feature_extractor.sampling_rate) == SR
        from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
        msgs = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"},
                                             {"type": "text", "text": DEFAULT_PROMPT}]}]
        self.prompt_text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # mode fmt: an unscored in-distribution response prefix so the target text
        # starts where MOSS would actually emit it. Identical for both candidates.
        self.ctx_suffix = "[0.00][S01]" if mode == "fmt" else ""
        self.cache_key = None
        self.cache = None

    def prep_audio(self, clip: np.ndarray):
        """processor output for this clip (shared by both candidate texts)."""
        text = self.prompt_text + self.ctx_suffix
        inputs = self.proc(text=text, audio=[clip.astype(np.float32)], return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def score(self, inputs: dict, target: str):
        ids = self.proc.tokenizer.encode(target, add_special_tokens=False)
        if not ids:
            return None
        tgt = torch.tensor([ids], dtype=torch.long, device=self.device)
        full = torch.cat([inputs["input_ids"], tgt], dim=1)
        am = torch.cat([inputs["attention_mask"], torch.ones_like(tgt)], dim=1)
        plen = inputs["input_ids"].shape[1]
        out = self.model(
            input_ids=full, attention_mask=am,
            input_features=inputs["input_features"],
            audio_feature_lengths=inputs["audio_feature_lengths"],
            audio_chunk_mapping=inputs["audio_chunk_mapping"],
            use_cache=False, return_dict=True,
        )
        logits = out.logits[:, plen - 1 : full.shape[1] - 1, :].float()
        lp = torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
        s = float(lp.sum())
        return s, s / len(ids), len(ids)


def read_wav(wav_dir: str, sid: str, cache: dict):
    if sid not in cache:
        cache.clear()
        x, sr = sf.read(os.path.join(wav_dir, f"{sid}.wav"), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            raise SystemExit(f"{sid}: sr={sr} != {SR}")
        cache[sid] = x
    return cache[sid]


def cut(x: np.ndarray, a: float, b: float) -> np.ndarray:
    i, j = max(0, int(a * SR)), min(len(x), int(b * SR))
    if j <= i:
        j = min(len(x), i + int(0.1 * SR))
    return x[i:j]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--arb-dir", required=True)
    ap.add_argument("--wav-dir", default="/root/autodl-tmp/dev_wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--mode", default="plain", choices=["plain", "fmt"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sanity", action="store_true", help="3-seg smoke test + audio-dependence probe")
    a = ap.parse_args()

    A, B = load_dir(a.src_dir), load_dir(a.arb_dir)
    keys = sorted(k for k in A if k in B and A[k]["words"] != B[k]["words"])
    print(f"diff segments: {len(keys)}", flush=True)
    if a.limit:
        keys = keys[: a.limit]

    j = Judge(a.dtype, a.mode)
    wc: dict = {}
    t0 = time.time()

    if a.sanity:
        rows = []
        for k in keys[:3]:
            x = read_wav(a.wav_dir, k[0], wc)
            inp = j.prep_audio(cut(x, k[1], k[2]))
            rs = j.score(inp, detok(A[k]["words"]))
            ra = j.score(inp, detok(B[k]["words"]))
            rows.append((k, rs, ra))
            print(f"{k} src(mean={rs[1]:.4f} sum={rs[0]:.2f} n={rs[2]}) "
                  f"arb(mean={ra[1]:.4f} sum={ra[0]:.2f} n={ra[2]})", flush=True)
        # audio dependence: seg0 text vs seg0 audio, vs a DIFFERENT segment audio
        k0, k1 = keys[0], keys[1]
        x0 = read_wav(a.wav_dir, k0[0], wc)
        own = j.score(j.prep_audio(cut(x0, k0[1], k0[2])), detok(A[k0]["words"]))
        x1 = read_wav(a.wav_dir, k1[0], wc)
        other = j.score(j.prep_audio(cut(x1, k1[1], k1[2])), detok(A[k0]["words"]))
        print(f"AUDIO_DEP mode={a.mode} dtype={a.dtype} seg={k0} own_audio_mean={own[1]:.4f} "
              f"other_audio({k1})_mean={other[1]:.4f} drop={own[1]-other[1]:.4f}", flush=True)
        print(f"sanity done in {time.time()-t0:.0f}s", flush=True)
        return 0

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["session_id"], round(r["start_time"], 3), round(r["end_time"], 3)))
    fh = open(a.out, "a", encoding="utf-8")
    fails = []
    for n, k in enumerate(keys, 1):
        if k in done:
            continue
        try:
            x = read_wav(a.wav_dir, k[0], wc)
            inp = j.prep_audio(cut(x, k[1], k[2]))
            ws, wa = detok(A[k]["words"]), detok(B[k]["words"])
            rs, ra = j.score(inp, ws), j.score(inp, wa)
            if rs is None or ra is None:
                raise ValueError("empty tokenization")
            fh.write(json.dumps({
                "session_id": k[0], "start_time": A[k]["start_time"], "end_time": A[k]["end_time"],
                "words_src": A[k]["words"], "words_arb": B[k]["words"],
                "score_moss_src": rs[1], "score_moss_arb": ra[1],
                "sum_src": rs[0], "sum_arb": ra[0], "ntok_src": rs[2], "ntok_arb": ra[2],
            }, ensure_ascii=False) + "\n")
            fh.flush()
        except Exception as e:  # noqa: BLE001
            fails.append((k, repr(e)))
            print(f"FAIL {k}: {e!r}", flush=True)
        if n % 25 == 0:
            print(f"[{n}/{len(keys)}] {time.time()-t0:.0f}s", flush=True)
    fh.close()
    print(f"DONE {len(keys)} segs, {len(fails)} failures, {time.time()-t0:.0f}s", flush=True)
    for k, e in fails:
        print("FAILED", k, e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
