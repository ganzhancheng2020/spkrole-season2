"""路径 1 思路 2：MOSS 段漏字回填（fun-asr 词级补字）。

设计目标：fun-asr 单独 26.07% 整体弱，但**它漏字率更低**（fun-asr 是工业级 ASR）。
策略：保留 MOSS 的说话人边界 + 时间戳，用 fun-asr 段里 MOSS 漏的字补到 MOSS 段尾。

## 核心算法

对每个 session 的 MOSS 段：
1. 找 fun-asr 输出里时间窗 IoU > IoU_THRESH 的段
2. 合并词：MOSS 段 words + (fun-asr 词集合 - MOSS 词集合)
3. 写出 words（保持字符串格式）

## 风险

- 漏字可能在 MOSS 段尾被补到**错的说话人**（fun-asr 边界与 MOSS 边界错位）
- 中文字符级简单去重可能在重叠区形成"伪词"
- dev 验证：< 18.39% (MOSS 单独) 才认为不破坏

## 用法

    cd baseline
    .venv/bin/python src/path1_fuse.py \
      --moss output/v007_moss \
      --funasr output_dev_funasr \
      --out output/path1_fuse_dev
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IOU_THRESH = 0.3  # 时间窗 IoU 阈值


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def save(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def tokenize(words: str) -> list[str]:
    """空格分隔的字符串 → token list。"""
    return [t for t in words.split() if t]


def merge_words(moss_words: str, funasr_words: str) -> str:
    """fun-asr 补 MOSS 漏字。保留 MOSS 顺序，fun-asr 多的字追加到末尾。"""
    moss_tokens = tokenize(moss_words)
    funasr_tokens = tokenize(funasr_words)
    if not moss_tokens:
        return funasr_words
    if not funasr_tokens:
        return moss_words
    moss_set = set(moss_tokens)
    new_tokens = [t for t in funasr_tokens if t not in moss_set]
    if not new_tokens:
        return moss_words
    return moss_words + " " + " ".join(new_tokens)


def merge_session(moss_recs: list[dict], funasr_recs: list[dict]) -> list[dict]:
    """对每个 MOSS 段找 fun-asr 匹配的段，回填漏字。"""
    out = []
    for m_rec in moss_recs:
        m_start, m_end = m_rec["start_time"], m_rec["end_time"]
        best, best_iou = None, 0.0
        for f_rec in funasr_recs:
            i = iou(m_start, m_end, f_rec["start_time"], f_rec["end_time"])
            if i > best_iou:
                best_iou = i
                best = f_rec
        if best is not None and best_iou >= IOU_THRESH:
            new_words = merge_words(m_rec["words"], best["words"])
        else:
            new_words = m_rec["words"]
        out.append({
            "session_id": m_rec["session_id"],
            "speaker": m_rec["speaker"],
            "start_time": m_rec["start_time"],
            "end_time": m_rec["end_time"],
            "words": new_words,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="路径 1 思路 2: MOSS 段漏字回填（fun-asr 词级补字）")
    ap.add_argument("--moss", required=True, help="MOSS 预测目录")
    ap.add_argument("--funasr", required=True, help="fun-asr 预测目录")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--iou-thresh", type=float, default=IOU_THRESH)
    args = ap.parse_args()

    moss_dir, funasr_dir, out_dir = Path(args.moss), Path(args.funasr), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = sorted(moss_dir.glob("[0-9]*.seglst.json"))
    if not preds:
        logger.error("MOSS 预测目录为空：%s", moss_dir)
        return 1

    no_funasr: list[str] = []
    total_added = 0
    for moss_path in preds:
        sid = moss_path.name.split(".")[0]
        funasr_path = funasr_dir / f"{sid}.seglst.json"
        moss_recs = load(moss_path)
        if not funasr_path.exists():
            no_funasr.append(sid)
            save(moss_recs, out_dir / f"{sid}.seglst.json")
            continue
        funasr_recs = load(funasr_path)
        merged = merge_session(moss_recs, funasr_recs)
        for m, n in zip(moss_recs, merged):
            added = len(tokenize(n["words"])) - len(tokenize(m["words"]))
            total_added += added
        save(merged, out_dir / f"{sid}.seglst.json")

    if no_funasr:
        logger.warning("%d 段无 fun-asr 预测: %s", len(no_funasr), no_funasr[:5])
    logger.info("共 %d 段，总补字 %d 个", len(preds), total_added)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
