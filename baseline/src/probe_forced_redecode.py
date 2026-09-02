"""P2 授权测量：强制重解码 —— 在漏掉的说话人边界处切断贪心流，强制新说话人假设续写。

## 测什么（P1 之后的关键裁决）

P1 证毕：ref 分割天花板 = 0.555 dev 点（只重装桶、文本不变）。
解码时干预的**额外**奖金 = 在正确说话人假设下**重新转写**（找回熔段/漏词）。
本探针在 base 模型（dev 干净）上做单边界强制实验：

    greedy 流:  ...[ts][S02]text_AAA|text_BBB[end][ts'][S02]...
                             ^ tb 在这段内部（ref 说这里换人，模型没切）
    强制后:     ...[ts][S02]text_AAA[tb][tb][S0k]<继续生成...>

  [tb] 作前段 [end]，[tb][S0k] 开新段；其后全部音频由模型在 S0k 假设下原生续写。

## 判读

- 每会话 base vs forced 的 tcpWER（词错误数）+ 续写文本与 ref 对照（定性）。
- 续写转写出「另一个人」的词 ⇒ 机制成立，进入校准（触发器 = 边界处未用编号的
  概率质量，可从 output_scores 一次拿到）；续写乱码/复读 ⇒ 家族死。

用法（GPU，moss env）：
    python src/probe_forced_redecode.py --wav-dir /root/autodl-tmp/dev_wav \
        --sessions 009,047,... --ref <ref.seglst.json> --out /tmp/p2
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
MEET = "/root/autodl-tmp/cvroot/.venv/bin/meeteval-wer"
if not os.path.exists(MEET):
    MEET = "/root/miniconda3/bin/meeteval-wer"

PUNCT_RE = re.compile(r"""[，。！？、；：“”‘’…,.!?;:"'()\[\]【]】""")
TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[一-鿿]")


def to_words(text: str) -> str:
    toks = TOKEN_RE.findall(PUNCT_RE.sub("", text))
    return " ".join(t.lower() if t.isascii() else t for t in toks).strip()


def evaluate_session(sid: str, records: list[dict], ref: list[dict], td: str) -> int:
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
    ap.add_argument("--sessions", required=True, help="逗号分隔候选 session（按缺人最严重挑）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
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
    os.makedirs(args.out, exist_ok=True)
    td = tempfile.mkdtemp()

    for sid in args.sessions.split(","):
        wav = Path(args.wav_dir) / f"{sid}.wav"
        ref = ref_by.get(sid, [])
        if not wav.exists() or not ref:
            logger.warning("%s 缺 wav/ref，跳过", sid)
            continue
        messages = build_transcription_messages(str(wav))
        inputs = prepare_inputs(processor, messages, max_length=131072, device=device).to(device)
        prompt_len = int(inputs["attention_mask"][0].sum().item())
        gen_kw = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "input_features": inputs["input_features"],
            "audio_feature_lengths": inputs["audio_feature_lengths"],
            "audio_chunk_mapping": inputs["audio_chunk_mapping"],
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        }
        with torch.inference_mode():
            out1 = model.generate(return_dict_in_generate=True, **gen_kw)
        gen_ids = out1.sequences[0][prompt_len:]
        text1 = tok.decode(gen_ids, skip_special_tokens=True).strip()
        segs1 = to_seglst(parse_transcript(text1), sid)
        base_err = evaluate_session(sid, segs1, ref, td)
        base_nspk = len({r["speaker"] for r in segs1})
        ref_nspk = len({r["speaker"] for r in ref})

        # 逐 token 文本跨度（定位切割点用）
        toks = [tok.decode([int(i)]) for i in gen_ids]
        # 找漏掉的 ref 说话人切换边界
        ref_bounds = []  # (tb, ref_speaker)
        for a, b in zip(ref, ref[1:]):
            if b["speaker"] != a["speaker"]:
                ref_bounds.append((b["start_time"], b["speaker"]))
        hyp_bounds = [r["start_time"] for r in segs1] + [r["end_time"] for r in segs1]
        missed = [(tb, spk) for tb, spk in ref_bounds
                  if all(abs(tb - hb) > 1.0 for hb in hyp_bounds)]
        if not missed:
            logger.info("%s: base %d 错 %d/%d 人，无漏边界，跳过", sid, base_err, base_nspk, ref_nspk)
            continue
        tb, ref_spk = missed[0]  # 最早的漏边界
        # 定位 tb 在流中的字符位置：找包含 tb 的段，按段内线性比例切
        split_char = None
        for m in re.finditer(r"\[(\d+\.\d{2})\]\[S(\d+)\](.*?)(?=\[\d+\.\d{2}\]\[S\d+\]|$)", text1, re.S):
            ts, spklab, body = float(m.group(1)), m.group(2), m.group(3)
            # 段的 end = 下一个 ts 或粗略 ts+段长（无 end 捕获则用下一段起点）
            nxt = re.search(r"\[(\d+\.\d{2})\]\[S\d+\]", text1[m.end():])
            te = float(nxt.group(1)) if nxt else ts + 10.0
            if ts <= tb < te:
                frac = (tb - ts) / max(te - ts, 1e-6)
                split_char = m.start(3) + int(len(body) * frac)
                break
        if split_char is None:
            logger.info("%s: 未找到 tb=%.2f 所在段，跳过", sid, tb)
            continue
        # 字符位置 → token 位置
        cum = 0
        split_tok = None
        for ti, t in enumerate(toks):
            cum += len(t)
            if cum >= split_char:
                split_tok = ti
                break
        if split_tok is None:
            continue
        new_k = max((int(m.group(2)) for m in
                     re.finditer(r"\[S(\d+)\]", text1)), default=0) + 1
        forced_text = f"[{tb:.2f}][{tb:.2f}][S{new_k:02d}]"
        forced_ids = tok.encode(forced_text, add_special_tokens=False)
        prefix_ids = torch.cat([inputs["input_ids"][0][:prompt_len], gen_ids[:split_tok],
                                torch.tensor(forced_ids, device=gen_ids.device)])
        attn = torch.ones_like(prefix_ids)
        with torch.inference_mode():
            out2 = model.generate(
                input_ids=prefix_ids[None], attention_mask=attn[None],
                input_features=inputs["input_features"],
                audio_feature_lengths=inputs["audio_feature_lengths"],
                audio_chunk_mapping=inputs["audio_chunk_mapping"],
                max_new_tokens=args.max_new_tokens, do_sample=False)
        cont_ids = out2[0][prefix_ids.numel():]
        cont_text = tok.decode(cont_ids, skip_special_tokens=True).strip()
        full_text = text1[:split_char] + forced_text + " " + cont_text
        segs2 = to_seglst(parse_transcript(full_text), sid)
        forced_err = evaluate_session(sid, segs2, ref, td)

        # 定性：续写文本 vs ref 在 tb 之后的内容
        ref_after = " / ".join(r["words"] for r in ref if r["start_time"] >= tb - 0.5)[:200]
        logger.info(
            "\n===== %s =====\nbase: %d 错 %d人 | forced: %d 错 %d人 | Δ %+d\n"
            "tb=%.2f ref_speaker=%s 强制为 S%02d\n"
            "ref(t≥tb): %s\n续写: %s\n",
            sid, base_err, base_nspk, forced_err, len({r["speaker"] for r in segs2}),
            forced_err - base_err, tb, ref_spk, new_k, ref_after, cont_text[:250])
        json.dump({"sid": sid, "base_err": base_err, "forced_err": forced_err,
                   "cont": cont_text, "segs_forced": segs2},
                  open(f"{args.out}/{sid}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    logger.info("P2_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
