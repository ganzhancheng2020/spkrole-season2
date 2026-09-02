"""SegLST 转换：fun-asr 句子 → 赛题 SegLST JSON 格式。

SegLST 字段（见 验证示例_meeteval/ref.seglst.json）：
    session_id: 音频名（去扩展名）
    speaker:    "spk{N}"，N 从 speaker_id 映射；None 归 0
    start_time: 秒，2 位小数
    end_time:   秒，2 位小数
    words:      字/英文词之间空格，英文小写
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from config import CFG

logger = logging.getLogger(__name__)


def _ms_to_sec(ms: int | float | None) -> float:
    """毫秒转秒，2 位小数。"""
    if ms is None:
        return 0.0
    return round(float(ms) / 1000.0, 2)


def _to_words(text: str) -> str:
    """中文逐字空格分割；英文单词小写保留。
    真实标注（ref.seglst.json）words 字段：
      - 中文字逐字空格（"我 现 在"）
      - 英文词整体小写
      - 不含标点（fun-asr 返回带标点，需剔除）
    """
    cleaned = re.sub(r"[，。！？、；：""''…,\.!\?;:\"'()\[\]【】]", "", text)
    tokens = re.findall(r"[A-Za-z]+|[0-9]+|[一-鿿]", cleaned)
    return " ".join(t.lower() if t.isascii() else t for t in tokens).strip()


def _speaker_label(speaker_id: int | None, mapping: dict) -> str:
    """speaker_id → spk{N}，从 spk1 起（对齐赛题标注 spk1/spk2/...）。"""
    key = speaker_id if speaker_id is not None else -1
    if key not in mapping:
        mapping[key] = f"spk{len(mapping) + 1}"
    return mapping[key]


def to_seglst(
    sentences: list[dict],
    session_id: str,
) -> list[dict]:
    """fun-asr 句子 → SegLST 记录列表。"""
    mapping: dict = {}
    out: list[dict] = []
    for s in sentences:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "session_id": session_id,
            "speaker": _speaker_label(s.get("speaker_id"), mapping),
            "start_time": _ms_to_sec(s.get("begin_time")),
            "end_time": _ms_to_sec(s.get("end_time")),
            "words": _to_words(text),
        })
    return out


def save_seglst(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("已写 SegLST %d 条 -> %s", len(records), out_path)


if __name__ == "__main__":
    demo = [
        {"speaker_id": 0, "begin_time": 0, "end_time": 1650, "text": "hi 今天天气咋样"},
        {"speaker_id": 1, "begin_time": 3200, "end_time": 4600, "text": "挺好的是晴天"},
    ]
    seg = to_seglst(demo, "session1")
    print(json.dumps(seg, ensure_ascii=False, indent=2))
