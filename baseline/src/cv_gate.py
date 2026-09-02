"""在 dev 交叉验证集上评估一个机制，并跑全部关卡。

## 为什么需要它（2026-08-20，八连败之后）

此前的验证有两个致命缺陷，各害了一次线上：

1. **配置不对**：出货用 all-106 模型，而验证用 train-82 模型在 holdout-24 上做。
   那是**另一个配置**。三个机制家族（v031/032、v046、v047）三次方向相反。
2. **功效造假**：把「7 个模型 × 24 段」当成 168 个样本。它们**共享同一批 24 段** ——
   有效样本量是 **24**。v047 的 −0.091 点实际来自 **2 个 session**，
   而我的重尾关卡剔的是 (模型,段) 单元格、不是整段，所以那个 session 还留在另外 6 个模型里，
   关卡假通过 → 线上 +0.00312。

本模块用 `cv_dev.sh` 产出的 **106 段干净预测**（每段都由一个没见过它的模型产生，
配方与 v040 完全一致），并强制跑下面四道关卡。

## 四道关卡（任一不过就不该出货）

| 关卡 | 判据 | 来历 |
|---|---|---|
| **触发广度** | 触发 session ≥ 8 | v047 只触发 2 段，那个量级不可测 |
| **重尾（整段剔除）** | 剔掉贡献最大的 3 个 session 后仍为收益 | 拦下过 v027 / D9；v047 时被我用错 |
| **自助区间** | 对 **session** 重采样，P(更好) ≥ 80% | v011 采信时的同口径 |
| **符号一致** | 改善 session 数 > 恶化 session 数 | 防「一段吃掉全部收益」|

用法：
    python src/cv_gate.py --base <基线目录> --cand <候选目录> [--name 机制名]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE_DIR)
REF = os.path.join(ROOT, "data/extracted/dev/dev/ref.seglst.json")
MEET = os.path.join(ROOT, ".venv/bin/meeteval-wer")
WORK = "/tmp/cv_gate"

MIN_TRIGGER_SESSIONS = 8
HEAVY_TAIL_DROP = 3
BOOT_MIN_P = 80.0


def per_session(d: str, tag: str) -> dict[str, tuple[float, float]]:
    w = os.path.join(WORK, tag)
    os.makedirs(w, exist_ok=True)
    recs: list[dict] = []
    for p in sorted(glob.glob(os.path.join(d, "[0-9]*.seglst.json"))):
        recs += json.load(open(p, encoding="utf-8"))
    if not recs:
        sys.exit(f"目录为空：{d}")
    json.dump(recs, open(f"{w}/h.json", "w", encoding="utf-8"), ensure_ascii=False)
    sids = {r["session_id"] for r in recs}
    json.dump([r for r in json.load(open(REF, encoding="utf-8")) if r["session_id"] in sids],
              open(f"{w}/r.json", "w", encoding="utf-8"), ensure_ascii=False)
    subprocess.run([MEET, "tcpwer", "-r", f"{w}/r.json", "-h", f"{w}/h.json", "--collar", "5"],
                   capture_output=True, check=True)
    raw = json.load(open(f"{w}/h_tcpwer_per_reco.json", encoding="utf-8"))
    return {k: (float(v["errors"]), float(v["length"])) for k, v in raw.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="dev 交叉验证集上的机制关卡")
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--name", default="候选机制")
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()

    A = per_session(args.base, "b")
    B = per_session(args.cand, "c")
    common = sorted(set(A) & set(B))
    if not common:
        sys.exit("两个目录没有共同 session")
    L = sum(A[k][1] for k in common)
    delta = {k: B[k][0] - A[k][0] for k in common}          # 负 = 改善
    tot = sum(delta.values())

    ra = 100 * sum(A[k][0] for k in common) / L
    rb = 100 * sum(B[k][0] for k in common) / L
    trig = [k for k in common if delta[k] != 0]
    better = [k for k in common if delta[k] < 0]
    worse = [k for k in common if delta[k] > 0]

    print(f"=== {args.name} ===")
    print(f"  评测集 {len(common)} 个 session（dev 交叉验证，每段模型均未见过）")
    print(f"  基线 {ra:7.3f}%   候选 {rb:7.3f}%   Δ {rb-ra:+.3f} 点")
    print(f"  触发 {len(trig)} 段（改善 {len(better)} / 恶化 {len(worse)}）")

    # 关卡 1：触发广度
    g1 = len(trig) >= MIN_TRIGGER_SESSIONS
    print(f"\n  [1] 触发广度  {len(trig)} >= {MIN_TRIGGER_SESSIONS}?  {'✅' if g1 else '❌ 样本太窄，不可测'}")

    # 关卡 2：重尾（整段剔除）
    order = sorted(common, key=lambda k: delta[k])
    drop = set(order[:HEAVY_TAIL_DROP])
    rest = sum(delta[k] for k in common if k not in drop)
    Lr = sum(A[k][1] for k in common if k not in drop)
    g2 = rest < 0
    print(f"  [2] 重尾关卡  剔掉贡献最大的 {HEAVY_TAIL_DROP} 段后 {100*rest/Lr:+.3f} 点  "
          f"{'✅ 仍为收益' if g2 else '❌ 翻负'}")

    # 关卡 3：对 session 自助重采样
    rng = np.random.default_rng(0)
    arr_d = np.array([delta[k] for k in common])
    arr_l = np.array([A[k][1] for k in common])
    idx = rng.integers(0, len(common), size=(args.boot, len(common)))
    boots = arr_d[idx].sum(1) / arr_l[idx].sum(1) * 100
    p = 100 * float((boots < 0).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    g3 = p >= BOOT_MIN_P
    print(f"  [3] 自助区间  95% [{lo:+.3f}, {hi:+.3f}]  P(更好)={p:.1f}%  "
          f"{'✅' if g3 else '❌ 低于 ' + str(BOOT_MIN_P) + '%'}")

    # 关卡 4：符号一致
    g4 = len(better) > len(worse)
    print(f"  [4] 符号一致  改善 {len(better)} > 恶化 {len(worse)}?  {'✅' if g4 else '❌'}")

    ok = g1 and g2 and g3 and g4
    print(f"\n  >>> {'✅ 四关全过，可考虑出货' if ok else '❌ 未全过，不出货'}")
    if trig:
        print(f"\n  贡献最大的 5 段：")
        for k in order[:5]:
            print(f"    {k}: {delta[k]:+.0f} 词")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
