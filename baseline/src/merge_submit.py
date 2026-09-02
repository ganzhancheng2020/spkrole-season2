"""合并提交：把 output_test/*.seglst.json 合并成单个可交付 JSON。

用法：
    PYTHONPATH=src .venv/bin/python src/merge_submit.py --dir output_test --out submission.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def merge(dir_path: Path) -> list[dict]:
    """合并目录下所有 SegLST 文件，按 session_id 排序。"""
    files = sorted(dir_path.glob("*.seglst.json"))
    all_recs: list[dict] = []
    for f in files:
        recs = json.loads(f.read_text(encoding="utf-8"))
        all_recs.extend(recs)
    all_recs.sort(key=lambda r: (r["session_id"], r["start_time"]))
    return all_recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="output_test", help="预测 SegLST 目录")
    ap.add_argument("--out", type=str, default="submission.json", help="输出文件")
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.exists():
        logger.error("目录不存在: %s", d)
        return

    files = sorted(d.glob("*.seglst.json"))
    logger.info("找到 %d 个 SegLST 文件", len(files))

    merged = merge(d)
    sessions = sorted({r["session_id"] for r in merged})
    logger.info("合并 %d 条记录, 覆盖 %d 个 session", len(merged), len(sessions))

    out = Path(args.out)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已写 %s (%d 条)", out, len(merged))

    sample_fields = {"session_id", "speaker", "start_time", "end_time", "words"}
    for r in merged[:1]:
        assert set(r.keys()) == sample_fields, f"字段不符: {set(r.keys())}"
    logger.info("格式校验通过（字段: %s）", sorted(sample_fields))


if __name__ == "__main__":
    main()
