"""2D 扫参：seg_dur x merge_thr -> apply_diar -> pick_ensemble -> tcpWER

用法（系统 Python 即可，recluster 子进程用 diarizen_venv）:
    cd baseline
    PYTHONPATH=src /Users/jack/miniconda3/bin/python3 src/sweep_2d.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

# ── paths ──
BASELINE = Path(__file__).resolve().parent.parent
OUTPUT = BASELINE / "output"
DIARIZEN_PYTHON = str(BASELINE / "diarizen_venv/bin/python")
MEETEVAL_WER = str(BASELINE.parent / ".venv/bin/meeteval-wer")
DEV_REF = BASELINE.parent / "data/extracted/dev/dev/ref.seglst.json"
MOSS_DIR = OUTPUT / "v007_moss"
FUNASR_DIR = OUTPUT  # fun-asr text: {sid}.seglst.json

MIN_SPK = 3
MAX_SPK = 7

EMB_DIRS = {
    "sd10": (OUTPUT / "emb_ec1.0", 1.0),
    "sd15": (OUTPUT / "emb_dev", 1.5),
}
MERGE_THRS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


# ── apply_diar (inline, from apply_diar.py) ──
def _overlap(a_s, a_e, b_s, b_e):
    return max(0.0, min(a_e, b_e) - max(a_s, b_s))

def _nearest_speaker(start, end, diar):
    mid = (start + end) / 2.0
    best, best_dist = 0, float("inf")
    for seg in diar:
        if seg["start"] <= mid <= seg["end"]:
            return seg["speaker"]
        dist = min(abs(seg["start"] - mid), abs(seg["end"] - mid))
        if dist < best_dist:
            best, best_dist = seg["speaker"], dist
    return best

def assign_split(rec, diar):
    start, end = rec["start_time"], rec["end_time"]
    tokens = rec["words"].split()
    if not tokens:
        return []
    pieces = []
    for seg in diar:
        ov_s, ov_e = max(start, seg["start"]), min(end, seg["end"])
        if ov_e > ov_s:
            pieces.append((ov_s, ov_e, seg["speaker"]))
    if not pieces:
        return [{**rec, "speaker": _nearest_speaker(start, end, diar)}]
    pieces.sort(key=lambda p: p[0])
    merged = []
    for ov_s, ov_e, spk in pieces:
        if merged and merged[-1][2] == spk:
            merged[-1][1] = max(merged[-1][1], ov_e)
        else:
            merged.append([ov_s, ov_e, spk])
    if len(merged) == 1:
        return [{**rec, "speaker": merged[0][2]}]
    total = sum(p[1] - p[0] for p in merged)
    out = []
    cursor = 0
    for i, (ov_s, ov_e, spk) in enumerate(merged):
        if i == len(merged) - 1:
            take = len(tokens) - cursor
        else:
            share = (ov_e - ov_s) / total if total > 0 else 0.0
            take = max(1, round(share * len(tokens)))
            take = min(take, len(tokens) - cursor - (len(merged) - i - 1))
        if take <= 0:
            continue
        chunk = tokens[cursor: cursor + take]
        cursor += take
        out.append({
            "session_id": rec["session_id"],
            "speaker": spk,
            "start_time": round(ov_s, 2),
            "end_time": round(ov_e, 2),
            "words": " ".join(chunk),
        })
    return [r for r in out if r["words"]]

def relabel(records):
    mapping = {}
    out = []
    for rec in sorted(records, key=lambda r: r["start_time"]):
        key = rec["speaker"]
        if key not in mapping:
            mapping[key] = f"spk{len(mapping) + 1}"
        out.append({**rec, "speaker": mapping[key]})
    return out

def load_diar(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    segs = data.get("segments", [])
    return sorted(segs, key=lambda s: s["start"])

def apply_diar_to_dir(pred_dir, diar_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = sorted(p for p in pred_dir.glob("[0-9]*.seglst.json"))
    for pred_path in preds:
        sid = pred_path.name.replace(".seglst.json", "")
        diar_path = diar_dir / f"{sid}.diar.json"
        if not diar_path.exists():
            continue
        records = json.loads(pred_path.read_text(encoding="utf-8"))
        diar = load_diar(diar_path)
        if not diar:
            out_records = records
        else:
            out_records = []
            for rec in records:
                out_records.extend(assign_split(rec, diar))
            out_records = relabel(out_records)
        (out_dir / f"{sid}.seglst.json").write_text(
            json.dumps(out_records, ensure_ascii=False), encoding="utf-8"
        )


# ── pick_ensemble (inline, from pick_ensemble.py) ──
MOSS_MAX_SPK = 2
MOSS_MAX_TURN_RATE = 0.10
CAM_MIN_SPK = 3
CAM_OVER_SPK = 6
CAM_EQ5_MOSS_MIN = 4

def n_speakers(records):
    return len({r["speaker"] for r in records})

def turn_rate(records):
    if not records:
        return 0.0
    span = max(r["end_time"] for r in records) - min(r["start_time"] for r in records)
    if span <= 0:
        return 0.0
    ordered = sorted(records, key=lambda r: r["start_time"])
    turns = sum(1 for a, b in zip(ordered, ordered[1:]) if a["speaker"] != b["speaker"])
    return turns / span

def pick_ensemble_run(moss_dir, cam_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    preds = sorted(moss_dir.glob("[0-9]*.seglst.json"))
    switched = []
    for moss_path in preds:
        sid = moss_path.name.split(".")[0]
        cam_path = cam_dir / f"{sid}.seglst.json"
        if not cam_path.exists():
            shutil.copy(moss_path, out_dir / f"{sid}.seglst.json")
            continue
        moss_recs = json.loads(moss_path.read_text(encoding="utf-8"))
        cam_recs = json.loads(cam_path.read_text(encoding="utf-8"))
        m_nspk = n_speakers(moss_recs)
        c_nspk = n_speakers(cam_recs)
        few_spk = m_nspk <= MOSS_MAX_SPK
        low_turn = turn_rate(moss_recs) <= MOSS_MAX_TURN_RATE
        v011_ok = (few_spk or low_turn) and c_nspk >= CAM_MIN_SPK
        cam_over = c_nspk >= CAM_OVER_SPK
        cam_eq5 = (c_nspk == 5 and m_nspk >= CAM_EQ5_MOSS_MIN)
        if v011_ok:
            chosen, src = cam_recs, cam_path
            switched.append(sid)
        elif cam_over:
            chosen, src = moss_recs, moss_path
        elif cam_eq5:
            chosen, src = cam_recs, cam_path
            switched.append(sid)
        else:
            chosen, src = moss_recs, moss_path
        shutil.copy(src, out_dir / f"{sid}.seglst.json")
    return len(switched)


# ── evaluate (meeteval-wer) ──
def evaluate(pred_dir):
    ref_records = json.loads(DEV_REF.read_text(encoding="utf-8"))
    session_ids = sorted({r["session_id"] for r in ref_records})
    hyp = []
    for sid in session_ids:
        p = pred_dir / f"{sid}.seglst.json"
        if p.exists():
            hyp.extend(json.loads(p.read_text(encoding="utf-8")))
    ref_path = pred_dir / "all.ref.seglst.json"
    hyp_path = pred_dir / "all.hyp.seglst.json"
    ref_path.write_text(json.dumps(ref_records, ensure_ascii=False), encoding="utf-8")
    hyp_path.write_text(json.dumps(hyp, ensure_ascii=False), encoding="utf-8")
    cmd = [MEETEVAL_WER, "tcpwer", "-r", str(ref_path), "-h", str(hyp_path), "--collar", "5"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    out_json = pred_dir / "all.hyp.seglst_tcpwer.json"
    data = json.loads(out_json.read_text(encoding="utf-8"))
    return data


# ── recluster (subprocess) ──
def run_recluster(emb_dir, merge_thr, out_dir):
    cmd = [
        DIARIZEN_PYTHON, "src/recluster.py",
        "--emb-dir", str(emb_dir),
        "--out", str(out_dir),
        "--min-spk", str(MIN_SPK),
        "--max-spk", str(MAX_SPK),
        "--merge-thr", str(merge_thr),
        "--seed", "0",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASELINE))
    if r.returncode != 0:
        print(f"  RECLUSTER FAILED: {r.stderr[-500:]}")
        return False
    return True


# ── main ──
def main():
    results = []

    # baseline: existing CAM path
    print("=== Baseline: existing CAM (hyp_sw_m3_7_0.70) ===")
    baseline_cam = OUTPUT / "hyp_sw_m3_7_0.70"
    baseline_pick = OUTPUT / "sweep_baseline_pick"
    n_sw = pick_ensemble_run(MOSS_DIR, baseline_cam, baseline_pick)
    data = evaluate(baseline_pick)
    if data:
        er = data["error_rate"]
        print(f"  pick tcpWER={er:.4%} errors={data['errors']} miss={data.get('missed_speaker',0)} falm={data.get('falarm_speaker',0)} switched={n_sw}")
        results.append(("baseline_sd15_t0.70(existing)", er, data, n_sw))

    # sweep
    for emb_key, (emb_path, sd) in EMB_DIRS.items():
        for mt in MERGE_THRS:
            tag = f"{emb_key}_t{str(mt).replace('.','')}"
            diar_dir = OUTPUT / f"diar_{tag}"
            cam_dir = OUTPUT / f"cam_{tag}"
            pick_dir = OUTPUT / f"pick_{tag}"

            print(f"\n=== {tag} (seg_dur={sd}, merge_thr={mt}) ===")
            t0 = time.time()
            ok = run_recluster(emb_path, mt, diar_dir)
            if not ok:
                continue
            t1 = time.time()
            print(f"  recluster: {t1-t0:.1f}s")

            apply_diar_to_dir(FUNASR_DIR, diar_dir, cam_dir)
            n_sw = pick_ensemble_run(MOSS_DIR, cam_dir, pick_dir)
            data = evaluate(pick_dir)
            if data:
                er = data["error_rate"]
                print(f"  pick tcpWER={er:.4%} errors={data['errors']} miss={data.get('missed_speaker',0)} falm={data.get('falarm_speaker',0)} switched={n_sw}")
                results.append((tag, er, data, n_sw))

    # summary
    print("\n" + "=" * 90)
    print(f"{'combo':<35} {'tcpWER':>8} {'errors':>7} {'miss':>5} {'falm':>5} {'switch':>7}")
    print("-" * 90)
    results.sort(key=lambda x: x[1])
    for tag, er, data, n_sw in results:
        marker = " ***" if er < results[0][1] + 1e-9 else ""
        print(f"{tag:<35} {er:>7.4%} {data['errors']:>7} {data.get('missed_speaker',0):>5} {data.get('falarm_speaker',0):>5} {n_sw:>7}{marker}")
    print("=" * 90)
    print(f"v026_pick (current pick+arb):  15.9202%")
    print(f"v028_dev (current final):      15.6425%")


if __name__ == "__main__":
    main()
