"""端到端编排：遍历音频目录 → 云端转写 → SegLST → 写 output/。

用法：
    python run.py                  # 跑 audio/ 下所有 wav
    python run.py a.wav b.wav      # 跑指定文件
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import CFG
from inference import transcribe_audio
from seglst_converter import save_seglst, to_seglst

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def process_one(
    file_path: Path,
    speaker_count: int | None,
    out_dir: Path | None = None,
) -> Path:
    """处理单个音频，返回输出的 SegLST 路径。"""
    session_id = file_path.stem
    logger.info("=== 处理 %s ===", file_path.name)
    sentences = transcribe_audio(file_path, speaker_count=speaker_count)
    logger.info("转写得到 %d 句", len(sentences))

    records = to_seglst(sentences, session_id)
    out_path = (out_dir or CFG.output_dir) / f"{session_id}.seglst.json"
    save_seglst(records, out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="音频文件；省略则跑 audio/ 下所有 wav")
    ap.add_argument("--speakers", type=int, default=None, help="说话人数提示")
    ap.add_argument("--out", type=str, default=None, help="输出目录（默认 output/）")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else None

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = sorted(CFG.audio_dir.glob("*.wav"))
        if not files:
            logger.error("audio/ 下无 wav 文件，请放入音频或指定路径")
            sys.exit(1)

    for f in files:
        try:
            process_one(f, args.speakers, out_dir)
        except Exception as e:
            logger.error("处理 %s 失败: %s", f.name, e)


if __name__ == "__main__":
    main()
