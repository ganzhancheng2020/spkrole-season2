"""择优轴可分性诊断：CAM++ vs wa_m0（用整体跑 errors 口径，与 oracle_pick 一致）。

oracle_pick.py 用「整体跑一次 tcpwer → per_reco 的 errors」判定谁赢（35 段 CAM 赢）。
这里复用同一口径，分析「CAM 赢的段」vs「MOSS 赢的段」在特征空间是否可分，
找能无 ref 选源的新信号。

⚠️ 单段单独跑 tcpwer 的口径是错的（collar 对齐不同），勿用。

用法：
    cd baseline
    .venv/bin/python src/pick_gap_probe.py output/wa_m0_dev output/hyp_sw_m3_7_0.70
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT = BASE_DIR.parent
WER_BIN = ROOT / ".venv/bin/meeteval-wer"
DEV_REF = ROOT / "data/extracted/dev/dev/ref.seglst.json"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def n_spk(recs):
    return len({r["speaker"] for r in recs})


def n_seg(recs):
    return len(recs)


def turn_rate(recs):
    if not recs:
        return 0.0
    span = max(r["end_time"] for r in recs) - min(r["start_time"] for r in recs)
    if span <= 0:
        return 0.0
    ordered = sorted(recs, key=lambda r: r["start_time"])
    turns = sum(1 for a, b in zip(ordered, ordered[1:]) if a["speaker"] != b["speaker"])
    return turns / span


def per_reco_errors(hyp_recs: list, ref_recs: list, tmp: Path):
    """整体跑一次 tcpwer，返回 {sid: errors}。与 oracle_pick 同口径。"""
    rf, hf = tmp / "r.json", tmp / "h.json"
    rf.write_text(json.dumps(ref_recs, ensure_ascii=False), encoding="utf-8")
    hf.write_text(json.dumps(hyp_recs, ensure_ascii=False), encoding="utf-8")
    subprocess.run(
        [str(WER_BIN), "tcpwer", "-r", str(rf), "-h", str(hf), "--collar", "5"],
        capture_output=True, check=True, cwd=tmp,
    )
    per = json.loads((tmp / "h_tcpwer_per_reco.json").read_text(encoding="utf-8"))
    return {k: v["errors"] for k, v in per.items()}


def main() -> int:
    if len(sys.argv) < 3:
        logger.error("用法: %s <moss_dir> <cam_dir>", __file__)
        return 2
    moss_dir, cam_dir = Path(sys.argv[1]), Path(sys.argv[2])
    ref_all = load(DEV_REF)

    def load_dir(d: Path):
        recs = []
        for p in sorted(d.glob("*.seglst.json")):
            if p.name.split(".")[0].isdigit():
                recs += json.loads(p.read_text(encoding="utf-8"))
        return recs

    moss_all, cam_all = load_dir(moss_dir), load_dir(cam_dir)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        moss_err = per_reco_errors(moss_all, ref_all, tmp)
        cam_err = per_reco_errors(cam_all, ref_all, tmp)

    sids = set(moss_err) & set(cam_err)
    rows = []
    for sid in sids:
        mr = json.loads((moss_dir / f"{sid}.seglst.json").read_text(encoding="utf-8"))
        cr = json.loads((cam_dir / f"{sid}.seglst.json").read_text(encoding="utf-8"))
        rows.append({
            "sid": sid,
            "cam_wins": cam_err[sid] < moss_err[sid],
            "e_moss": moss_err[sid], "e_cam": cam_err[sid],
            "m_spk": n_spk(mr), "c_spk": n_spk(cr),
            "m_seg": n_seg(mr), "c_seg": n_seg(cr),
            "m_tr": turn_rate(mr), "c_tr": turn_rate(cr),
            "spk_diff": n_spk(cr) - n_spk(mr),
            "seg_diff": n_seg(cr) - n_seg(mr),
            "tr_diff": turn_rate(cr) - turn_rate(mr),
        })

    wins = [r for r in rows if r["cam_wins"]]
    losses = [r for r in rows if not r["cam_wins"]]
    logger.info("CAM++ 赢 %d 段 / MOSS 赢 %d 段（errors 口径）", len(wins), len(losses))

    def avg(rs, key):
        return sum(r[key] for r in rs) / len(rs) if rs else 0

    logger.info("=== 特征均值：CAM赢 vs MOSS赢 ===")
    for key in ["spk_diff", "seg_diff", "tr_diff", "m_spk", "c_spk", "m_seg", "c_seg", "m_tr", "c_tr"]:
        print(f"  {key:<10} CAM赢={avg(wins,key):+.3f}  MOSS赢={avg(losses,key):+.3f}")

    logger.info("=== CAM 赢的 session 列表 ===")
    print("  ", [r["sid"] for r in wins])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())