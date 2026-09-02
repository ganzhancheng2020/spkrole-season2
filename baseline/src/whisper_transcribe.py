"""Whisper 整段转写验证：测 whisper 在本赛 dev 上的纯文本质量（方案 A 门槛）。

强化文本基座的候选第三源。验证 whisper-medium 的中文纯文本 WER 是否值得作为
第三文本源（与 MOSS/FireRed 仲裁）。先跑少量段，若纯文本接近 FireRed 则值得。

用法：
    cd baseline
    diarizen_venv/bin/python src/whisper_transcribe.py \
        ../data/extracted/dev/dev/wav output/whisper_probe 5  # 5 段探针
"""
from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PUNCT = re.compile(r"""[，。！？、；：""''…,.!?;:"'()\[\]【】]""")
MODEL = "openai/whisper-medium"


def main() -> int:
    if len(sys.argv) < 3:
        logger.error("用法: %s <wav_dir> <out_dir> [limit]", __file__)
        return 2
    wav_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("加载 %s ...", MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL)
    processor = WhisperProcessor.from_pretrained(MODEL)

    wavs = sorted(wav_dir.glob("*.wav"))
    if limit:
        wavs = wavs[:limit]
    for k, wav in enumerate(wavs, 1):
        sid = wav.name.split(".")[0]
        out_p = out_dir / f"{sid}.seglst.json"
        if out_p.exists():
            continue
        x, sr = sf.read(wav, dtype="float32")
        feats = processor(x, sampling_rate=sr, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats, language="zh", task="transcribe")
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        toks = [t for w in text for t in re.findall(r"[A-Za-z]+|[0-9]+|[一-鿿]", w)]
        words = " ".join(t.lower() if t.isascii() else t for t in toks)
        out_p.write_text(
            f'[{{"session_id": "{sid}", "speaker": "spk1", '
            f'"start_time": 0.0, "end_time": {len(x)/sr:.2f}, "words": "{words}"}}]',
            encoding="utf-8",
        )
        logger.info("[%d/%d] %s: %d 词 (%.0fs)", k, len(wavs), sid, len(toks), time.time() - t0)
    logger.info("WHISPER DONE %d 段", len(wavs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())