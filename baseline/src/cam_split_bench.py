"""在 dev 上复核 cam_split_verify 的四道验收关卡（README §1.1 的表格由本脚本产出）。

与 `src/bench.py` 的区别：bench.py 依赖共享的 `bench_baselines.json`（当前锁在
`_cv_final` 口径上），跑它会改动仓库状态；本脚本自带基线、只写临时文件，
供审核者独立复核，不影响任何既有产物。

四道关卡（每道都由线上回执校准过，见 README §5）：
  1. 触发面 ≥ 8 场                 —— 少于 8 场时点估计不可用，无论正负
  2. 整段剔除最幸运 3 场后仍为收益 —— 否决权最高，防重尾外推
  3. 自助重采样 P(更好) ≥ 80%
  4. 改善场数 > 恶化场数

用法（根 .venv，需 meeteval + numpy）：
    cd baseline
    PYTHONPATH=src ../.venv/bin/python src/cam_split_bench.py \\
        --base output/_cv_mb6 \\
        --ref  ../data/extracted/dev/dev/ref.seglst.json \\
        --diar output/diar_sw_m3_7_0.70 --emb output/emb_dev [--sweep]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from cam_split_verify import load_records, run

MEETEVAL = os.environ.get("MEETEVAL", "../.venv/bin/meeteval-wer")


def score_per_session(records: list[dict], ref: str) -> dict[str, tuple[int, int]]:
    """逐 session 返回 (错误数, 参考词数)。tcpWER = Σ错误 / Σ词数，故可按场聚合。"""
    keep = {r["session_id"] for r in load_records(ref)}
    records = [r for r in records if r["session_id"] in keep]
    fd, path = tempfile.mkstemp(suffix=".seglst.json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    subprocess.run([MEETEVAL, "tcpwer", "-r", ref, "-h", path, "--collar", "5"],
                   capture_output=True, text=True, check=False)
    per_path = path[:-5] + "_tcpwer_per_reco.json"
    if not os.path.exists(per_path):
        raise RuntimeError(f"meeteval 未产出 {per_path}；检查 MEETEVAL={MEETEVAL} 是否可执行")
    with open(per_path, encoding="utf-8") as f:
        data = json.load(f)
    for leftover in glob.glob(path[:-5] + "*"):
        os.unlink(leftover)
    items = data.items() if isinstance(data, dict) else ((x["session_id"], x) for x in data)
    return {k: (v["insertions"] + v["deletions"] + v["substitutions"], v["length"])
            for k, v in items}


def gates(base: dict, cand: dict, seed: int = 0, boots: int = 2000) -> dict:
    keys = sorted(set(base) & set(cand))
    delta = {k: cand[k][0] - base[k][0] for k in keys}
    touched = [k for k in keys if delta[k]]
    total_words = sum(base[k][1] for k in keys)

    dropped = set(sorted(touched, key=lambda k: delta[k])[:3])       # 最幸运的 3 场
    rest = [k for k in keys if k not in dropped]

    # 对 session 做有放回重采样，统计「候选更好」的比例（与 bench.py 同口径）
    rng = np.random.default_rng(seed)
    d = np.array([delta[k] for k in keys], dtype=float)
    draws = rng.integers(0, len(keys), size=(boots, len(keys)))
    wins = int((d[draws].sum(axis=1) < 0).sum())

    return {
        "touched": len(touched),
        "delta": 100 * sum(delta.values()) / total_words,
        "drop3": 100 * sum(delta[k] for k in rest) / sum(base[k][1] for k in rest),
        "p": 100 * wins / boots,
        "better": sum(1 for k in touched if delta[k] < 0),
        "worse": sum(1 for k in touched if delta[k] > 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--base", required=True, help="dev 基线预测目录（例：output/_cv_mb6）")
    ap.add_argument("--ref", required=True, help="dev 参考标注 ref.seglst.json")
    ap.add_argument("--diar", required=True, help="dev CAM++ 分离目录")
    ap.add_argument("--emb", required=True, help="dev CAM++ 声纹目录")
    ap.add_argument("--min-frac", type=float, default=0.05)
    ap.add_argument("--max-sim", type=float, default=0.50)
    ap.add_argument("--min-gap", type=int, default=1)
    ap.add_argument("--max-gap", type=int, default=1)
    ap.add_argument("--sweep", action="store_true",
                    help="扫 max_sim，展示 0.60 这条分界线（README §1.1 的依据）")
    args = ap.parse_args()

    base_per = score_per_session(load_records(args.base), args.ref)
    base_rate = 100 * sum(v[0] for v in base_per.values()) / sum(v[1] for v in base_per.values())
    print(f"dev 基线：{args.base} = {base_rate:.4f}%  （{len(base_per)} 场）\n")

    sims = [0.75, 0.70, 0.65, 0.60, 0.55, 0.50] if args.sweep else [args.max_sim]
    print(f"{'max_sim':>8} {'触发':>5} {'Δ点':>9} {'剔top3':>9} {'P%':>7} {'好':>4} {'坏':>4}  四关")
    for sim in sims:
        cand, _, _ = run(args.base, args.diar, args.emb,
                         args.min_frac, sim, args.min_gap, args.max_gap)
        g = gates(base_per, score_per_session(cand, args.ref))
        ok = [g["touched"] >= 8, g["drop3"] < 0, g["p"] >= 80, g["better"] > g["worse"]]
        marks = "".join("✓" if x else "✗" for x in ok)
        flag = "  <<< 四关全过" if all(ok) else ""
        print(f"{sim:>8.2f} {g['touched']:>5d} {g['delta']:>+9.4f} {g['drop3']:>+9.4f} "
              f"{g['p']:>7.1f} {g['better']:>4d} {g['worse']:>4d}  {marks}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
