"""本项目的 benchmark：一条命令跑完链路 → tcpWER → 四道关卡 → 预测线上落点。

## 为什么需要它

这几天每验一个候选都要手工串：pick → normalize → meeteval → 逐 session 拆 →
四道关卡 → 换算线上。步骤多且易错（已因手工口径错过两次：
命名空间混用、重尾关卡剔错单位）。固化成一条命令，口径就不会再漂。

## 评测集

**106 段 dev 交叉验证集**：每折排除本折 session 再训练，只在该折推理，
配方与 v040 一致 → 每段都由**未见过它的模型**产生。
这是出货配置唯一干净的本地信号（见 wiki/insights/cv-gate-calibrated.md）。

⚠️ **基线口径（2026-08-26 修正）**：历史 `bench_baselines.json` 的基线
   `_cv_final` 用的 dev「CAM 分支」是 `relabel(v026_pick)` —— 一个**含 MOSS 内容的
   集成输出**，与 test 的 fun-asr 构造差 **6.021 点**，导致 dev CV 比出货链路
   **乐观约 0.5 点**，且**跨分支类结论（路由/选源/分支消融）全部失真**。
   忠实基线见 `bench_baselines_true.json`（16.229%，与 test 同构造）。
   见 wiki/insights/dev-test-chain-mismatch.md。
   **跨 dev/test 比较前必须用内容指纹认亲，目录名不是契约。**

⚠️ 四关有**否决权**：不接受「别的检验说可以」作为覆盖理由。
   新检验只能追加在四关之后（更严），不能替代其中任何一关。
   v052 就是越过第 2 关出货的：dev −0.509 点但剔 top3 后 +0.031，
   线上 0.14672（+0.00165 退步）。见 wiki/insights/gate2-not-overridable.md。

## 四道关卡（每道都由线上回执校准过）

1. **触发广度** ≥8 段 —— v047 只触发 2 段，线上 +0.00312
2. **重尾**：整段剔除 top3 后仍为收益 —— v047 时剔的是 (模型,段) 单元格，假通过
3. **自助区间**：对 session 重采样，P(更好) ≥80%
4. **符号一致**：改善 session 数 > 恶化数

正向校准：normalize_output 四关全过（31 改善 / 0 恶化）。
反向校准：spk_reassign 点估计 +0.062 但触发仅 3 段 → 关卡判「不可测」，
线上证明点估计符号是错的、关卡拦对了。

## 迁移率

v048 实测：本地 −0.077 点 → 线上 −0.00058，即 **0.0075 线上/本地点**
（与历史「架构级 65%」估计的 0.0065 吻合）。事前预测落点 0.1463、实测 0.14625。

用法：
    python src/bench.py --cand <成品目录> [--name 名字]      # 对比基线
    python src/bench.py --cand <目录> --set-baseline         # 设为新基线
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess as sp

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
MEET = os.path.join(ROOT, ".venv/bin/meeteval-wer")
REF = "/tmp/cv_ref.json"
STORE = os.path.join(ROOT, "bench_baselines.json")
TRANSFER = 0.00685         # 线上 / 本地点，两点标定：v048(-0.077→-0.00058)
                           # + v051(-0.180→-0.00118)。用 0.0075 时预测 v051
                           # 为 0.14490，实测 0.14507，偏乐观 0.00017
LEADER = 0.14185           # 榜首（2026-08-28 更新，原 0.142 / 0.14364）
BEST_ONLINE = 0.14354      # v065（2026-08-28 更新，原 v051=0.14507）
# 破榜首所需 = (BEST_ONLINE - LEADER)/TRANSFER = 0.247 本地点（截至 2026-08-28）。
# 这两个常数是「上界 < 所需」型判断的分母，过期就会把不够的方向判成够。
# 换榜/换最佳版本后必须同步改这里，并在 wiki 封条里写明当时的「所需」取值与日期。


def per_session(d: str) -> dict[str, tuple[float, float]]:
    recs = []
    for p in sorted(glob.glob(os.path.join(d, "[0-9]*.seglst.json"))):
        recs += json.load(open(p, encoding="utf-8"))
    if not recs:
        raise SystemExit(f"❌ 目录为空：{d}")
    json.dump(recs, open("/tmp/_b_h.json", "w", encoding="utf-8"), ensure_ascii=False)
    sids = {r["session_id"] for r in recs}
    ref = [r for r in json.load(open(REF, encoding="utf-8")) if r["session_id"] in sids]
    json.dump(ref, open("/tmp/_b_r.json", "w", encoding="utf-8"), ensure_ascii=False)
    sp.run([MEET, "tcpwer", "-r", "/tmp/_b_r.json", "-h", "/tmp/_b_h.json",
            "--collar", "5"], capture_output=True, check=True)
    raw = json.load(open("/tmp/_b_h_tcpwer_per_reco.json", encoding="utf-8"))
    return {k: (float(v["errors"]), float(v["length"])) for k, v in raw.items()}


def rate(d: dict) -> float:
    return 100.0 * sum(v[0] for v in d.values()) / sum(v[1] for v in d.values())


def gates(base: dict, cand: dict, name: str) -> bool:
    common = sorted(set(base) & set(cand))
    delta = {k: cand[k][0] - base[k][0] for k in common}
    L = sum(base[k][1] for k in common)
    tot = sum(delta.values())
    fired = [k for k in common if delta[k] != 0]
    better = sum(1 for k in fired if delta[k] < 0)
    worse = sum(1 for k in fired if delta[k] > 0)

    print(f"\n=== {name} ===")
    print(f"  评测集 {len(common)} 段（dev 交叉验证，每段模型均未见过）")
    print(f"  基线 {rate(base):7.3f}%   候选 {rate(cand):7.3f}%   "
          f"Δ {100*tot/L:+.3f} 点")
    print(f"  触发 {len(fired)} 段（改善 {better} / 恶化 {worse}）")

    g1 = len(fired) >= 8
    top3 = sorted(fired, key=lambda k: delta[k])[:3]
    rest = sum(delta[k] for k in common if k not in top3)
    g2 = rest < 0
    rng = np.random.default_rng(0)
    arr = np.array([delta[k] for k in common])
    boots = [arr[rng.integers(0, len(arr), len(arr))].sum() for _ in range(5000)]
    p_better = float(np.mean(np.array(boots) < 0))
    g3 = p_better >= 0.80
    g4 = better > worse

    print(f"\n  [1] 触发广度  {len(fired)} >= 8?  {'✅' if g1 else '❌ 样本太窄，不可测'}")
    print(f"  [2] 重尾关卡  整段剔 top3 后 {100*rest/L:+.3f} 点  "
          f"{'✅ 仍为收益' if g2 else '❌ 翻负'}")
    print(f"  [3] 自助区间  P(更好)={100*p_better:.1f}%  {'✅' if g3 else '❌ 低于 80%'}")
    print(f"  [4] 符号一致  改善 {better} > 恶化 {worse}?  {'✅' if g4 else '❌'}")

    ok = g1 and g2 and g3 and g4
    print(f"\n  >>> {'✅ 四关全过，可考虑出货' if ok else '❌ 未全过，不出货'}")

    pred = BEST_ONLINE + (100 * tot / L) * TRANSFER
    print(f"\n  预测线上：{BEST_ONLINE:.5f} → {pred:.5f}"
          f"（迁移率 {TRANSFER}/本地点）")
    if pred < LEADER:
        print(f"  ✅ 预测可破榜首 {LEADER}（余量 {LEADER-pred:+.5f}）")
    else:
        print(f"  ⚠️ 预测**不足以**破榜首 {LEADER}，仍差 {pred-LEADER:.5f}")

    print(f"\n  贡献最大的 5 段：{[(k, int(delta[k])) for k in top3[:5]]}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="本项目 benchmark：链路评测 + 四关 + 线上预测")
    ap.add_argument("--cand", required=True, help="候选成品目录（pick+normalize 之后）")
    ap.add_argument("--name", default="候选")
    ap.add_argument("--set-baseline", action="store_true", help="把候选设为新基线")
    args = ap.parse_args()

    cand = per_session(args.cand)
    store = json.load(open(STORE, encoding="utf-8")) if os.path.exists(STORE) else {}

    if args.set_baseline or "baseline" not in store:
        store["baseline"] = {"dir": args.cand, "name": args.name,
                             "tcpwer": rate(cand), "per": {k: list(v) for k, v in cand.items()}}
        json.dump(store, open(STORE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"✅ 已设为基线：{args.name} = {rate(cand):.3f}%  -> {STORE}")
        return 0

    b = store["baseline"]
    base = {k: tuple(v) for k, v in b["per"].items()}
    print(f"基线：{b['name']} = {b['tcpwer']:.3f}%")
    gates(base, cand, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
