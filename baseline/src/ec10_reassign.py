"""评估「用 CAM++ seg_dur=1.0 (ec1.0) 找回 MOSS 漏判说话人」在 v028 上的可行性。

策略：只动「ec1.0 确认 MOSS 漏人」的 session（ec1.0 说话人数 > MOSS(v028) 说话人数）。
对该 session 用 ec1.0 说话人段重分配 speaker（segment 段级 / split 切分级，复用 apply_diar）。
其余 session 原样保留 v028 文本与标签，完全不碰。

注意：FireRed ts 与 ec1.0 时钟不对齐（见 logs/2026-08-09），词级 ts 归属不可用；
split 模式按段内语速均匀近似切分。

用法：
    cd baseline
    PYTHONPATH=src .venv/bin/python src/ec10_reassign.py \
        --pred-dir output/v028_dev --diar-dir output/diar_ec1.0 \
        --out output/v029_ec10_seg --mode segment
        --out output/v029_ec10_split --mode split
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from apply_diar import assign_segment, assign_split, load_diar, relabel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def process_session(pred_path: Path, diar_path: Path, mode: str) -> tuple[list[dict], bool]:
    """返回 (records, was_targeted)。只在 ec1.0 确认更多说话人时才改标签。"""
    records = json.loads(pred_path.read_text(encoding="utf-8"))
    diar = load_diar(diar_path)
    if not diar:
        return records, False
    diar_count = len({s["speaker"] for s in diar})
    moss_count = len({r["speaker"] for r in records})
    if diar_count <= moss_count:
        return records, False  # 不确认漏人，原样保留

    assign = assign_split if mode == "split" else assign_segment
    out: list[dict] = []
    for rec in records:
        out.extend(assign(rec, diar))
    return relabel(out), True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--diar-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["segment", "split"], default="split")
    args = ap.parse_args()

    pred_dir, diar_dir, out_dir = (
        Path(p).resolve() for p in (args.pred_dir, args.diar_dir, args.out)
    )
    if out_dir == pred_dir:
        logger.error("--out 不能等于 --pred-dir")
        raise SystemExit(2)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = sorted(p for p in pred_dir.glob("*.seglst.json") if p.name[0].isdigit())
    n_written, n_targeted, n_no_diar = 0, 0, 0
    for pred_path in preds:
        sid = pred_path.name.replace(".seglst.json", "")
        diar_path = diar_dir / f"{sid}.diar.json"
        if not diar_path.exists():
            n_no_diar += 1
            continue
        records, targeted = process_session(pred_path, diar_path, args.mode)
        (out_dir / f"{sid}.seglst.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_written += 1
        n_targeted += int(targeted)

    logger.info("已写 %d session，目标(漏人) %d 个，缺 diar %d -> %s",
                n_written, n_targeted, n_no_diar, out_dir)


if __name__ == "__main__":
    main()