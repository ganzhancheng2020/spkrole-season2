"""fun-asr 重排到 MOSS 段 + 落盘词级时间戳（fun-asr 第三源集成 Step 1）。

模仿 fr_retext.py 的结构：fun-asr 云端整段转写（含词级时间戳）→ 按 begin_time
把每个 word 分配到 MOSS 段 → 落盘 {sid}.ts.json（词级时间戳）和 {sid}.seglst.json
（段结构与 MOSS 一致）。

用法：
    cd baseline
    .venv/bin/python src/funasr_retext.py \
        ../data/extracted/dev/dev/wav \
        ../baseline/output/v007_moss \
        ../baseline/output/funasr_retext
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

# 复用现有 run.py 的 fun-asr 调用链
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import transcribe_audio  # type: ignore[import-not-found]  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PUNCT = re.compile(r"""[，。！？、；：""''…,.!?;:"'()\[\]【】]""")
TOKEN = re.compile(r"[A-Za-z]+|[0-9]+|[一-鿿]")


def tokenize_words(sent_words: list[dict]) -> list[tuple[str, float, float]]:
    """fun-asr 返回的 words[{begin_time,end_time,text,speaker_id}] → 字符级时间戳。"""
    out = []
    for word in sent_words:
        for char in TOKEN.findall(PUNCT.sub("", word["text"])):
            out.append(
                (
                    char,
                    word["begin_time"] / 1000.0,
                    word["end_time"] / 1000.0,
                )
            )
    return out


def reassign(
    moss_recs: list[dict], all_words: list[tuple[str, float, float]]
) -> list[dict]:
    """按 (st+en)/2 把每个 word 分配到最近的 MOSS 段。"""
    buckets: dict[int, list[str]] = {j: [] for j in range(len(moss_recs))}
    for token, start, end in all_words:
        center = (start + end) / 2
        best, best_distance = None, 1e9
        for index, record in enumerate(moss_recs):
            if record["start_time"] <= center <= record["end_time"]:
                best, best_distance = index, 0
                break
            distance = min(
                abs(center - record["start_time"]),
                abs(center - record["end_time"]),
            )
            if distance < best_distance:
                best, best_distance = index, distance
        if best is not None:
            buckets[best].append(token)
    output = []
    for index, record in enumerate(moss_recs):
        words = " ".join(buckets[index])
        output.append({**record, "words": words if words else record["words"]})
    return output


def process_one(sid: str, wav_dir: Path, moss_dir: Path, out_dir: Path) -> bool:
    """返回 True 表示产出新文件，False 表示跳过（已存在）。"""
    out_seglst = out_dir / f"{sid}.seglst.json"
    out_ts = out_dir / f"{sid}.ts.json"
    if out_seglst.exists() and out_ts.exists():
        return False
    wav = wav_dir / f"{sid}.wav"
    moss_path = moss_dir / f"{sid}.seglst.json"
    if not moss_path.exists():
        logger.warning("MOSS 段缺失: %s", sid)
        return False
    sentences = transcribe_audio(wav, speaker_count=None)
    all_words: list[tuple[str, float, float]] = []
    for sentence in sentences:
        all_words.extend(tokenize_words(sentence.get("words", [])))
    moss_recs = json.loads(moss_path.read_text(encoding="utf-8"))
    out_recs = reassign(moss_recs, all_words)
    out_seglst.write_text(
        json.dumps(out_recs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    out_ts.write_text(
        json.dumps(
            [[token, round(start, 3), round(end, 3)] for token, start, end in all_words],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return True


def main() -> int:
    if len(sys.argv) != 4:
        logger.error("用法: %s <wav_dir> <moss_dir> <out_dir>", __file__)
        return 2
    wav_dir, moss_dir, out_dir = (
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3]),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    session_ids = sorted(
        {path.name.removesuffix(".seglst.json") for path in moss_dir.glob("[0-9]*.seglst.json")}
    )
    for sid in session_ids:
        if process_one(sid, wav_dir, moss_dir, out_dir):
            done += 1
        else:
            skipped += 1
    logger.info("fun-asr retext 完成: 新增 %d, 跳过 %d", done, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
