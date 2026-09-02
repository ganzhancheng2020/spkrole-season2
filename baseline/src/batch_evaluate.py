"""批量评测：合并 dev 全量预测与 ref，跑 meeteval tcpWER。

用法：
    PYTHONPATH=src .venv/bin/python src/batch_evaluate.py
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MEETEVAL_WER = ROOT_DIR / ".venv" / "bin" / "meeteval-wer"
DEV_REF = ROOT_DIR / "data" / "extracted" / "dev" / "dev" / "ref.seglst.json"
OUTPUT = ROOT_DIR / "baseline" / "output"


def build_hyp(ref_records: list[dict], pred_dir: Path = OUTPUT) -> list[dict]:
    """对 ref 里每个 session_id，读对应预测文件，拼成 hyp。"""
    session_ids = sorted({r["session_id"] for r in ref_records})
    hyp: list[dict] = []
    missing: list[str] = []
    for sid in session_ids:
        p = pred_dir / f"{sid}.seglst.json"
        if not p.exists():
            missing.append(sid)
            continue
        hyp.extend(json.loads(p.read_text(encoding="utf-8")))
    if missing:
        logger.warning("缺 %d 段预测: %s", len(missing), missing[:5])
    return hyp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(DEV_REF))
    ap.add_argument("--collar", type=int, default=5)
    ap.add_argument("--dir", default=str(OUTPUT),
                    help="预测目录（默认 baseline/output；评测变体时指向变体目录）")
    args = ap.parse_args()

    pred_dir = Path(args.dir).resolve()
    ref_records = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    session_ids = sorted({r["session_id"] for r in ref_records})
    logger.info("ref 含 %d 个 session, %d 条记录", len(session_ids), len(ref_records))
    logger.info("预测目录: %s", pred_dir)

    hyp = build_hyp(ref_records, pred_dir)
    logger.info("hyp 含 %d 条记录", len(hyp))

    ref_path = pred_dir / "all.ref.seglst.json"
    hyp_path = pred_dir / "all.hyp.seglst.json"
    ref_path.write_text(json.dumps(ref_records, ensure_ascii=False), encoding="utf-8")
    hyp_path.write_text(json.dumps(hyp, ensure_ascii=False), encoding="utf-8")

    if not MEETEVAL_WER.exists():
        logger.error("未找到 %s", MEETEVAL_WER)
        sys.exit(2)
    cmd = [
        str(MEETEVAL_WER), "tcpwer",
        "-r", str(ref_path), "-h", str(hyp_path),
        "--collar", str(args.collar),
    ]
    logger.info("运行: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error("meeteval 失败: %s", r.stderr)
        sys.exit(3)
    out_json = pred_dir / "all.hyp.seglst_tcpwer.json"
    data = json.loads(out_json.read_text(encoding="utf-8"))
    per_reco_path = pred_dir / "all.hyp.seglst_tcpwer_per_reco.json"
    per = json.loads(per_reco_path.read_text(encoding="utf-8")) if per_reco_path.exists() else {}

    print(f"\n=== 全量 tcpWER = {data['error_rate']:.4%} "
          f"(errors={data['errors']}, length={data['length']}) ===")
    print(f"insertions={data.get('insertions')}, "
          f"deletions={data.get('deletions')}, "
          f"substitutions={data.get('substitutions')}")
    print(f"missed_speaker={data.get('missed_speaker')}, "
          f"falarm_speaker={data.get('falarm_speaker')}, "
          f"scored_speaker={data.get('scored_speaker')}")

    if per:
        recos = per.get("per_reco", {})
        wers = [(sid, v.get("error_rate", 0), v.get("errors", 0), v.get("length", 0))
                for sid, v in recos.items()]
        wers.sort(key=lambda x: -x[1])
        print(f"\n=== 最差 5 段 ===")
        for sid, er, e, l in wers[:5]:
            print(f"  {sid}: {er:.2%} ({e}/{l})")
        print(f"=== 最好 5 段 ===")
        for sid, er, e, l in wers[-5:]:
            print(f"  {sid}: {er:.2%} ({e}/{l})")
        n = len(wers)
        if n:
            avg = sum(w[1] for w in wers) / n
            print(f"\n各段算术平均 tcpWER = {avg:.2%} (共 {n} 段)")


if __name__ == "__main__":
    main()
