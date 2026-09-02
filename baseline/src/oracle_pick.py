"""按段择优上界诊断：给定多个 hyp 目录，每段选 tcpWER 最低的那个系统，算合并分数。

**这是诊断工具，不是提交路径。** 它用 dev ref 回头看每段谁最低（test 无 ref），
产出的是「若有完美选源器能到多少」的上界。真正无 ref 的生产规则在 `pick_ensemble.py`。

历史上界（见 wiki/insights/ensemble-oracle-bound.md）：
    MOSS + CAM++                      14.73%   ← 本脚本复算，与当年记录的 15.05% 吻合
    + ERes2NetV2 两档（D6.4）          14.31%   ← 只降 0.42 点 → 判该源不值得加

上界 ≠ 可达：v011 的启发式规则实际只兑现了约 15%。用上界筛掉不值得的源，
别拿它预期收益。

用法（评分二进制在根 venv，本脚本用 baseline venv 跑即可）：
    cd baseline
    .venv/bin/python src/oracle_pick.py output/v007_moss output/hyp_sw_m3_7_0.70
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT = BASE_DIR.parent
# 评分只走根 venv（meeteval 0.4.3），与转写 venv 严格分离，见 CLAUDE.md
WER_BIN = ROOT / ".venv/bin/meeteval-wer"
DEV_REF = ROOT / "data/extracted/dev/dev/ref.seglst.json"
COLLAR = 5

# 产物目录里混有评测中间件（all.hyp.seglst.json / p10_hyp.seglst.json /
# *_tcpwer.json 等）。全 glob 会把同一 session 重复计入 —— 曾让 MOSS 算出 127%
# 的假分数。只认纯 session 命名：001.seglst.json。
SESSION_NAME_RE = re.compile(r"\d+")


def load_dir(d: Path) -> list[dict]:
    """收一个 hyp 目录的 SegLST 记录（只认 NNN.seglst.json）。"""
    recs: list[dict] = []
    for p in sorted(d.glob("*.seglst.json")):
        if not SESSION_NAME_RE.fullmatch(p.name.split(".")[0]):
            continue
        recs += json.loads(p.read_text(encoding="utf-8"))
    return recs


def per_session_wer(ref_recs: list[dict], hyp_recs: list[dict],
                    tmp: Path) -> dict[str, tuple[int, int]]:
    """跑一次 tcpwer，返回 {session_id: (errors, length)}。"""
    ref_path, hyp_path = tmp / "r.json", tmp / "h.json"
    ref_path.write_text(json.dumps(ref_recs, ensure_ascii=False))
    hyp_path.write_text(json.dumps(hyp_recs, ensure_ascii=False))
    subprocess.run(
        [str(WER_BIN), "tcpwer", "-r", str(ref_path), "-h", str(hyp_path),
         "--collar", str(COLLAR)],
        capture_output=True, check=True, cwd=tmp)
    per = json.loads((tmp / "h_tcpwer_per_reco.json").read_text())
    return {k: (v["errors"], v["length"]) for k, v in per.items()}


def main() -> int:
    dirs = [Path(a) for a in sys.argv[1:]]
    if len(dirs) < 2:
        print(__doc__)
        return 1
    if missing := [str(d) for d in dirs if not d.is_dir()]:
        print(f"目录不存在：{missing}")
        return 1

    ref = json.loads(DEV_REF.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        stats = {d: per_session_wer(ref, load_dir(d), tmp) for d in dirs}

    # 只在所有系统都覆盖到的 session 上比较，否则缺段的系统会被不公平地算便宜
    sids = set.intersection(*[set(s) for s in stats.values()])
    tot_e = tot_l = 0
    wins: dict[Path, int] = defaultdict(int)
    for sid in sids:
        best = min(dirs, key=lambda d: stats[d][sid][0])
        e, ln = stats[best][sid]
        tot_e += e
        tot_l += ln
        wins[best] += 1

    for d in dirs:
        e = sum(stats[d][s][0] for s in sids)
        ln = sum(stats[d][s][1] for s in sids)
        print(f"{str(d):40s} 单独 {100 * e / ln:6.2f}%   赢 {wins[d]:3d} 段")
    print(f"{'按段择优上界':40s}      {100 * tot_e / tot_l:6.2f}%   ({len(sids)} 段)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
