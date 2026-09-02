"""阶段2：用本地 diarization 的说话人标签，重写 fun-asr 转写的 speaker 字段。

文本与时间戳仍全部来自 fun-asr（阶段1不碰文本）；只把「谁在说」换成本地 CAM++ 的判断。
诊断依据：fun-asr 系统性少估说话人数，说话人对齐吃掉约 13 个点（见 SCORES.md v001 诊断）。

两种映射模式：
  segment  段级——整条 fun-asr 段归给时间重叠最多的说话人（v001 复现）。
  split    切分级——一条段跨多个说话人时，按时间比例把字分配到各说话人（v002 候选）。
           fun-asr 原始返回的词级时间戳未落盘且 transcription_url 已过期，
           故只能按「段内语速均匀」假设近似，不能用真实词时间。

用法：
    cd baseline
    PYTHONPATH=src .venv/bin/python src/apply_diar.py \
        --pred-dir output --diar-dir output/diar_dev --out output/v001_seg --mode segment
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_diar(path: Path) -> list[dict]:
    """读分离缓存，返回按时间排序的 [{start, end, speaker}]。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    segs = data.get("segments", [])
    return sorted(segs, key=lambda s: s["start"])


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """两区间的重叠时长（无重叠返回 0）。"""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _nearest_speaker(start: float, end: float, diar: list[dict]) -> int:
    """无重叠时的兜底：取时间上最近的分离段的说话人。"""
    mid = (start + end) / 2.0
    best, best_dist = 0, float("inf")
    for seg in diar:
        if seg["start"] <= mid <= seg["end"]:
            return seg["speaker"]
        dist = min(abs(seg["start"] - mid), abs(seg["end"] - mid))
        if dist < best_dist:
            best, best_dist = seg["speaker"], dist
    return best


def assign_segment(rec: dict, diar: list[dict]) -> list[dict]:
    """段级映射：整条记录归给重叠时长最大的说话人。"""
    start, end = rec["start_time"], rec["end_time"]
    totals: dict[int, float] = {}
    for seg in diar:
        ov = _overlap(start, end, seg["start"], seg["end"])
        if ov > 0:
            totals[seg["speaker"]] = totals.get(seg["speaker"], 0.0) + ov
    spk = max(totals, key=totals.get) if totals else _nearest_speaker(start, end, diar)
    return [{**rec, "speaker": spk}]


def assign_split(rec: dict, diar: list[dict]) -> list[dict]:
    """切分级映射：按与各分离段的时间重叠，把字按比例切给不同说话人。

    假设段内语速均匀（fun-asr 词级时间戳不可得）。切分后每个子段保留自己的时间区间。
    """
    start, end = rec["start_time"], rec["end_time"]
    tokens = rec["words"].split()
    if not tokens:
        return []

    # 收集与本段有重叠的分离片段，按时间排序
    pieces: list[tuple[float, float, int]] = []
    for seg in diar:
        ov_start, ov_end = max(start, seg["start"]), min(end, seg["end"])
        if ov_end > ov_start:
            pieces.append((ov_start, ov_end, seg["speaker"]))
    if not pieces:
        return [{**rec, "speaker": _nearest_speaker(start, end, diar)}]
    pieces.sort(key=lambda p: p[0])

    # 合并相邻同说话人片段，避免把一句话切成无意义的碎块
    merged: list[list] = []
    for ov_start, ov_end, spk in pieces:
        if merged and merged[-1][2] == spk:
            merged[-1][1] = max(merged[-1][1], ov_end)
        else:
            merged.append([ov_start, ov_end, spk])
    if len(merged) == 1:
        return [{**rec, "speaker": merged[0][2]}]

    # 按各片段时长占比分配 token 数（至少 1 个，保证不丢字）
    total = sum(p[1] - p[0] for p in merged)
    out: list[dict] = []
    cursor = 0
    for i, (ov_start, ov_end, spk) in enumerate(merged):
        if i == len(merged) - 1:
            take = len(tokens) - cursor
        else:
            share = (ov_end - ov_start) / total if total > 0 else 0.0
            take = max(1, round(share * len(tokens)))
            take = min(take, len(tokens) - cursor - (len(merged) - i - 1))
        if take <= 0:
            continue
        chunk = tokens[cursor : cursor + take]
        cursor += take
        out.append({
            "session_id": rec["session_id"],
            "speaker": spk,
            "start_time": round(ov_start, 2),
            "end_time": round(ov_end, 2),
            "words": " ".join(chunk),
        })
    return [r for r in out if r["words"]]


def relabel(records: list[dict]) -> list[dict]:
    """int 簇编号 → spk1/spk2/...，按首次出现顺序编号（对齐赛题标注习惯）。"""
    mapping: dict[int, str] = {}
    out: list[dict] = []
    for rec in sorted(records, key=lambda r: r["start_time"]):
        key = rec["speaker"]
        if key not in mapping:
            mapping[key] = f"spk{len(mapping) + 1}"
        out.append({**rec, "speaker": mapping[key]})
    return out


def process_session(pred_path: Path, diar_path: Path, mode: str) -> list[dict]:
    """对单个 session 做标签重写。"""
    records = json.loads(pred_path.read_text(encoding="utf-8"))
    diar = load_diar(diar_path)
    if not diar:
        logger.warning("%s 分离结果为空，原样保留", pred_path.stem)
        return records

    assign = assign_split if mode == "split" else assign_segment
    out: list[dict] = []
    for rec in records:
        out.extend(assign(rec, diar))
    return relabel(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="fun-asr 预测目录（提供文本）")
    ap.add_argument("--diar-dir", required=True, help="阶段1 分离缓存目录")
    ap.add_argument("--out", required=True, help="输出目录（勿指向 --pred-dir）")
    ap.add_argument("--mode", choices=["segment", "split"], default="segment")
    args = ap.parse_args()

    pred_dir, diar_dir, out_dir = (
        Path(p).resolve() for p in (args.pred_dir, args.diar_dir, args.out)
    )
    if out_dir == pred_dir:
        logger.error("--out 不能等于 --pred-dir，会覆盖 fun-asr 文本源")
        raise SystemExit(2)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 只取 {sid}.seglst.json，排除 all.*.seglst.json 等聚合产物
    preds = sorted(p for p in pred_dir.glob("*.seglst.json") if p.name[0].isdigit())
    logger.info("找到 %d 个预测文件，模式=%s", len(preds), args.mode)

    n_written, n_missing_diar = 0, 0
    for pred_path in preds:
        sid = pred_path.name.replace(".seglst.json", "")
        diar_path = diar_dir / f"{sid}.diar.json"
        if not diar_path.exists():
            n_missing_diar += 1
            continue
        records = process_session(pred_path, diar_path, args.mode)
        (out_dir / f"{sid}.seglst.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_written += 1

    logger.info("已写 %d 个 session -> %s", n_written, out_dir)
    if n_missing_diar:
        logger.warning("%d 个 session 缺分离缓存，已跳过", n_missing_diar)


if __name__ == "__main__":
    main()
