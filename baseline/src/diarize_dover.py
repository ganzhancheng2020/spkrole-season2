"""阶段1（DOVER-Lap 集成版）：合并多套 diarization 结果的说话人标签。

deep-research finding 4：DISPLACE 2024 / DIHARD III 冠军都是集成，不是单模型。
实测 per-reco 互补性强（Sortformer 赢 42 段 / CAM++ 赢 37 段 / 平 27），各有所长。
本脚本用 DOVER-Lap（label mapping + voting）合并 CAM++ + Sortformer 两套结果。

输入：多个 diar 目录（schema: {session_id, num_speakers, segments:[{start,end,speaker}]}）
输出：合并后的 diar 目录，schema 一致，apply_diar.py 零修改复用。

用法：
    cd baseline
    PYTHONPATH=src nemo_venv/bin/python src/diarize_dover.py \
        --diar-dirs output/diar_dev output/diar_nemo_dev --out output/diar_dover_dev
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def to_turns(diar_record: dict, file_id: str, system_id: str):
    """diar.json segments → DOVER-Lap Turn 列表。speaker 跨系统用 system_id 前缀区分。"""
    from dover_lap import Turn
    turns = []
    for seg in diar_record.get("segments", []):
        turns.append(Turn(
            onset=seg["start"],
            offset=seg["end"],
            speaker_id=f"{system_id}_{seg['speaker']}",
            file_id=file_id,
        ))
    return turns


def from_turns(turns, label_map: dict) -> list[dict]:
    """Turn 列表 → segments schema。label_map: 跨系统 speaker_id → 统一 int。"""
    out = []
    for t in turns:
        if t.offset <= t.onset:
            continue
        spk_str = t.speaker_id
        if spk_str not in label_map:
            label_map[spk_str] = len(label_map)
        out.append({
            "start": round(float(t.onset), 3),
            "end": round(float(t.offset), 3),
            "speaker": label_map[spk_str],
        })
    out.sort(key=lambda s: s["start"])
    return out


def merge_one(systems_data: list[dict], session_id: str, weights: list[float] | None = None):
    """对单个 session 合并多套 diar 结果。

    systems_data: 各系统的 {session_id, num_speakers, segments} 记录列表。
    weights: 各系统权重（None=用 rank weight 默认；自定义=等权或调参）。
    返回合并后的 {session_id, num_speakers, segments}。
    """
    from dover_lap import DOVERLap

    turns_list = []
    for i, rec in enumerate(systems_data):
        if rec is None:
            continue
        turns = to_turns(rec, session_id, system_id=f"sys{i}")
        turns_list.append(turns)

    if not turns_list:
        return {"session_id": session_id, "num_speakers": 0, "segments": []}
    if len(turns_list) == 1:
        label_map: dict = {}
        segs = from_turns(turns_list[0], label_map)
        return {"session_id": session_id, "num_speakers": len(label_map), "segments": segs}

    kwargs = dict(
        file_id=session_id,
        label_mapping="greedy",
        voting_method="average",
    )
    if weights is not None:
        kwargs["weight_type"] = "custom"
        kwargs["custom_weight"] = [str(w) for w in weights]
    else:
        kwargs["weight_type"] = "rank"
    combined = DOVERLap.combine_turns_list(turns_list, **kwargs)
    # combine_turns_list 返回 List[Turn]（合并后唯一序列）
    label_map = {}
    segs = from_turns(combined, label_map)
    return {"session_id": session_id, "num_speakers": len(label_map), "segments": segs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diar-dirs", nargs="+", required=True,
                    help="多个 diar 目录（各含 {NNN}.diar.json）")
    ap.add_argument("--out", required=True, help="合并后输出目录")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="各系统权重（与 --diar-dirs 同序），不传用 rank weight")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    diar_dirs = [Path(d).resolve() for d in args.diar_dirs]
    session_ids = set()
    for d in diar_dirs:
        for f in d.glob("*.diar.json"):
            # f.stem 给 "001.diar"，去掉 ".diar" 后缀得 "001"
            sid = f.stem
            if sid.endswith(".diar"):
                sid = sid[:-5]
            session_ids.add(sid)
    session_ids = sorted(session_ids)
    logger.info("共 %d 个 session，%d 套系统", len(session_ids), len(diar_dirs))

    n_done = 0
    for sid in session_ids:
        systems_data = []
        for d in diar_dirs:
            p = d / f"{sid}.diar.json"
            if p.exists():
                systems_data.append(json.loads(p.read_text(encoding="utf-8")))
            else:
                systems_data.append(None)
        if all(x is None for x in systems_data):
            continue
        merged = merge_one(systems_data, sid, args.weights)
        (out_dir / f"{sid}.diar.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_done += 1
        if n_done % 20 == 0:
            logger.info("已合并 %d/%d", n_done, len(session_ids))

    logger.info("完成: 合并 %d session -> %s", n_done, out_dir)


if __name__ == "__main__":
    main()
