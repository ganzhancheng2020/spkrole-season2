"""MagicData-RAMC -> MOSS 微调 JSONL（切 40 秒片段）。

MagicData-RAMC 对话平均 30 分钟，远超 MOSS max_length 8192（约 40 秒音频）。
切成 40 秒无重叠片段（类似本赛 41.7 秒），每片段切 wav + 转写 + 重编号。

txt 格式: [start,end]\\tperson_id\\tgender,lang\\ttext
"""
from __future__ import annotations

import argparse
import json
import logging
import re

import soundfile as sf  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
CLIP_SEC = 40.0
MIN_CLIP_SEC = 5.0
MIN_SEGS = 2


def parse_txt(txt_path):
    segs = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        f = line.split("\t")
        if len(f) < 4:
            continue
        m = re.match(r"\[(.+),(.+)\]", f[0])
        if not m:
            continue
        start, end = float(m.group(1)), float(m.group(2))
        text = f[3].strip()
        if text.startswith("[*]") or not text:
            continue
        text = re.sub(r"\[\+\]$", "", text).strip()
        if text and end - start > 0.05:
            segs.append((start, end, f[1], text))
    return segs


def clip_segments(segs, cs, ce):
    out = []
    for s, e, p, t in segs:
        if e <= cs or s >= ce:
            continue
        s2 = max(s, cs) - cs
        e2 = min(e, ce) - cs
        if e2 - s2 > 0.05:
            out.append((s2, e2, p, t))
    return out


def build_transcript(clipped):
    spk_map: dict[str, str] = {}
    parts = []
    for s, e, p, t in clipped:
        if p not in spk_map:
            spk_map[p] = f"S{len(spk_map) + 1:02d}"
        parts.append(f"[{s:.2f}][{spk_map[p]}]{t}[{e:.2f}]")
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="MDT2021S003 目录")
    ap.add_argument("--out-wav", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--clip-sec", type=float, default=CLIP_SEC)
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 个对话")
    args = ap.parse_args()

    from pathlib import Path

    txt_dir = Path(args.data_dir) / "TXT"
    wav_dir = Path(args.data_dir) / "WAV"
    out_wav = Path(args.out_wav)
    out_wav.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(txt_dir.glob("*.txt"))
    if args.limit:
        txt_files = txt_files[: args.limit]
    logger.info("处理 %d 个对话", len(txt_files))

    n_clips = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as fh:
        for ti, txt in enumerate(txt_files):
            name = txt.stem
            wav_path = wav_dir / f"{name}.wav"
            if not wav_path.exists():
                continue
            segs = parse_txt(txt)
            if not segs:
                continue
            x, sr = sf.read(str(wav_path), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            total = segs[-1][1]
            cs = 0.0
            while cs < total:
                ce = min(cs + args.clip_sec, total)
                if ce - cs < MIN_CLIP_SEC:
                    break
                clipped = clip_segments(segs, cs, ce)
                if len(clipped) < MIN_SEGS:
                    cs += args.clip_sec
                    continue
                transcript = build_transcript(clipped)
                if not transcript:
                    cs += args.clip_sec
                    continue
                cname = f"{name}_{int(cs)}_{int(ce)}"
                cwav = out_wav / f"{cname}.wav"
                s0, s1 = int(cs * sr), int(ce * sr)
                sf.write(str(cwav), x[s0:s1], sr)
                rec = {
                    "conversation": [
                        {"role": "user", "message_type": "text", "content": DEFAULT_PROMPT},
                        {"role": "user", "message_type": "audio", "content": str(cwav)},
                        {"role": "assistant", "message_type": "text", "content": transcript},
                    ]
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_clips += 1
                cs += args.clip_sec
            if (ti + 1) % 10 == 0:
                logger.info("[%d/%d] %d clips", ti + 1, len(txt_files), n_clips)
    logger.info("DONE %d clips -> %s", n_clips, args.out_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
