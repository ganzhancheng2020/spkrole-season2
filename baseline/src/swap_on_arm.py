"""v092：在生产臂（test_v051_moss，已含 word_arb+reassign）上做换段重解码 + FireRed 裁判。

## 与 v091 的差别（v091 回执的知识全部用上）

- v091 的基座惩罚 +0.0024 = 重解码漂移 +0.0013 + 无 reassign +0.0012。
  本脚本**不重解码整场**：臂的流直接由 test_v051_moss 记录渲染（零漂移），reassign 已在臂里。
- 触发：spk_ac_probe 的 teacher-forcing gap（臂自身边界，v047 信号）。
- 动作：浅 gap 段换未用编号 + 续写重解码该段（P2 证明能原生转写缺失说话人的词）。
- 验收：FireRed 对同 span 的新旧文本打分，新文本净赢 ≥ margin 才接受（judge_gate 证明能止损）。

用法（GPU，moss env，cwd=/root/autodl-tmp/ft）：
    python swap_on_arm.py --arm output/test_v051_moss --probe arm_probe.jsonl \
        --scores arm_scores.json --wav-dir /root/autodl-tmp/test_wav \
        --out-root v092_arms --theta -1.5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MOSS_REPO = os.environ.get("MOSS_REPO", "/root/autodl-tmp/MOSS-Transcribe-Diarize")
BASE_SNAP = os.environ.get(
    "BASE_SNAP",
    "/root/autodl-tmp/hf/hub/models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/"
    "e8681d68e7042738ffca8ac8212bc8fcb1131ab8")
FIRERED_SRC = os.environ.get("FIRERED_SRC", "/tmp/FireRedASR2S")
FIRERED_CKPT = os.environ.get("FIRERED_CKPT", "/tmp/FireRedASR2-AED")
TS_RE = re.compile(r"\[\d+\.\d{2}\]")
MARGINS = [0.0, 0.05, 0.1]
THETA_DEFAULT = -1.5


def render_stream(recs: list[dict]) -> tuple[str, list[int]]:
    """臂记录 → MOSS 流文本 + 每段首次出现的编号（与 build_spk_probe_jsonl 同构）。"""
    order: dict[str, int] = {}
    parts = []
    nums = []
    for r in sorted(recs, key=lambda x: x["start_time"]):
        if r["speaker"] not in order:
            order[r["speaker"]] = len(order) + 1
        k = order[r["speaker"]]
        nums.append(k)
        parts.append(f"[{r['start_time']:.2f}][S{k:02d}]{r['words']}[{r['end_time']:.2f}]")
    return "".join(parts), nums


def main() -> int:
    sys.path.insert(0, MOSS_REPO)
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--probe", required=True, help="build_spk_probe_jsonl 的 jsonl（含 tok_off）")
    ap.add_argument("--scores", required=True, help="spk_ac_probe 的 scores json")
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--theta", type=float, default=THETA_DEFAULT)
    ap.add_argument("--max-seg-tokens", type=int, default=200)
    args = ap.parse_args()

    import torch
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, prepare_inputs
    from moss_sat import load_model
    import word_arb as wa  # FireRed CPU 打分栈

    model, processor, device, dtype = load_model(BASE_SNAP, None, "auto")
    tok = processor.tokenizer
    digit_ids = {d: tok.encode(d, add_special_tokens=False)[0] for d in "123456789"}

    scores = {o["session"]: o for o in json.load(open(args.scores, encoding="utf-8"))}
    probe = {}
    for line in open(args.probe, encoding="utf-8"):
        d = json.loads(line)
        probe[d["meta"]["session"]] = d["meta"]

    outs = {m: Path(args.out_root) / f"v092_{m}" for m in MARGINS}
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=True)

    n_total_swap = 0
    for df in sorted(Path(args.arm).glob("[0-9]*.seglst.json")):
        sid = df.stem
        recs = json.load(open(df, encoding="utf-8"))
        if sid not in scores or sid not in probe:
            for m in MARGINS:
                json.dump(recs, open(outs[m] / f"{sid}.seglst.json", "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
            continue
        stream, nums = render_stream(recs)
        stream_ids = tok.encode(stream, add_special_tokens=False)
        # 每段的数字 token 位置：用 probe 的 tok_off（与 builder 同一口径）
        metas = {round(s["start_time"], 3): s for s in probe[sid]["segs"]}
        fired = []
        used: set[str] = set()
        for i, r in enumerate(sorted(recs, key=lambda x: x["start_time"])):
            k = nums[i]
            used_before = {str(x) for j, x in enumerate(nums) if j < i}
            sc = scores[sid].get("segs")
            seg_sc = {round(s["start_time"], 3): s for s in sc} if sc else {}
            s = seg_sc.get(round(r["start_time"], 3))
            pm = metas.get(round(r["start_time"], 3), {})
            if s is None or "tok_off" not in pm:
                continue
            unused = [d for d in "123456789" if d not in used_before and d != str(k)]
            if not unused:
                continue
            best = max(unused, key=lambda d: s["logp"][d])
            gap = s["logp"][best] - s["logp"][str(k)]
            if gap > args.theta:
                fired.append((i, pm["tok_off"], str(k), best, gap))
        if not fired:
            for m in MARGINS:
                json.dump(recs, open(outs[m] / f"{sid}.seglst.json", "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
            continue
        wav = Path(args.wav_dir) / f"{sid}.wav"
        info = sf.info(str(wav))
        sr0 = info.samplerate
        messages = build_transcription_messages(str(wav))
        inputs = prepare_inputs(processor, messages, max_length=131072, device=device).to(device)
        prompt_len = int(inputs["attention_mask"][0].sum().item())
        prompt_ids = inputs["input_ids"][0][:prompt_len].tolist()

        recs_sorted = sorted(recs, key=lambda x: x["start_time"])
        swaps = []
        for i, off, chosen, best, gap in fired:
            r = recs_sorted[i]
            cap = None
            if i + 1 < len(recs_sorted):
                cap = recs_sorted[i + 1]["start_time"] + 0.01
            span_end = cap if cap is not None else r["end_time"]
            old_text = r["words"]
            cont = model.generate(
                input_ids=torch.tensor([prompt_ids + stream_ids[:off] + [digit_ids[best]]],
                                       device=device),
                attention_mask=torch.ones(1, prompt_len + off + 1, device=device, dtype=torch.long),
                input_features=inputs["input_features"],
                audio_feature_lengths=inputs["audio_feature_lengths"],
                audio_chunk_mapping=inputs["audio_chunk_mapping"],
                max_new_tokens=args.max_seg_tokens, do_sample=False)
            cont_text = tok.decode(cont[0][prompt_len + off + 1:], skip_special_tokens=True)
            if cont_text.startswith("]"):
                cont_text = cont_text[1:]
            mstop = TS_RE.search(cont_text)
            if not mstop:
                continue
            new_text = cont_text[:mstop.start()]
            if not new_text.strip():
                continue
            clip, _ = sf.read(str(wav), start=int(r["start_time"] * sr0),
                              stop=int(min(span_end, r["start_time"] + 30) * sr0))
            if len(clip) < int(sr0 * 0.2):
                continue
            enc, mask = wa.make_enc(clip, sr0)
            s_old = wa.score(enc, mask, old_text)
            s_new = wa.score(enc, mask, new_text)
            swaps.append({"i": i, "gap": gap, "s_old": s_old, "s_new": s_new,
                          "new_text": new_text, "new_k": int(best), "chosen": chosen})
        n_total_swap += len(swaps)
        for m in MARGINS:
            acc = {s["i"]: s for s in swaps if s["s_new"] - s["s_old"] >= m}
            out = []
            for i, r in enumerate(recs):
                r2 = dict(r)
                if i in acc:
                    s = acc[i]
                    # 独立命名空间标签，避免与后续首次出现的 spkN 撞号
                    r2["speaker"] = f"spknew{i}x{s['new_k']}"
                    r2["words"] = s["new_text"]
                out.append(r2)
            json.dump(out, open(outs[m] / f"{sid}.seglst.json", "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        logger.info("%s: 触发 %d，FireRed 接受@0 %d（s_new 赢 %d）",
                    sid, len(fired), sum(1 for s in swaps if s["s_new"] >= s["s_old"]),
                    sum(1 for s in swaps if s["s_new"] > s["s_old"]))
        json.dump({"swaps": swaps}, open(f"{args.out_root}/{sid}.swaps.json", "w",
                                         encoding="utf-8"), ensure_ascii=False, indent=1)
    logger.info("SWAP_ON_ARM_DONE 总 swap %d", n_total_swap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
