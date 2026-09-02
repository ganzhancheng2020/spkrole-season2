"""方向 3 下游分析：MOSS 内部说话人表征的聚类一致性与 oracle-lift。

输入：spk_hidden.npz（dump_hidden.py 产物，1519 段 × 29 层 × 1024 维）
      + hyp 目录（段标签）+ ref（oracle 标签）。

## 判定逻辑

每 session、每层：
1. 向量 L2 归一化，K = hyp 说话人数，KMeans(seed=0) 聚类；
2. 匈牙利匹配聚类↔hyp 标签 → 一致率（≈100% = 表示只是 argmax 回声，死）；
3. 段级 oracle 对错：hyp 说话人按会话级总重叠 1:1 贪心映射到 ref 说话人，
   段的最大重叠 ref 说话人 = 映射对象 → 对；否则错（同 attr_bucket 的 hyp 命名空间口径）；
4. **lift = P(错 | 聚类分歧) / P(错 | 总体)** —— 分歧段若显著更常是错标段，信号才有价值。

会话级聚类质量另报轮廓系数。全部层跑完报最好的 3 层。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # baseline/
ROOT = os.path.dirname(BASE_DIR)                                          # 项目根
REF_PATH = os.path.join(ROOT, "data/extracted/dev/dev/ref.seglst.json")


def load_inputs(npz: str, hyp_dir: str):
    d = np.load(npz, allow_pickle=False)
    vecs = d["vecs"].astype(np.float32)          # [N, L, D]
    metas = json.loads(str(d["meta"]))
    hyp: dict[str, list[dict]] = {}
    for p in sorted(glob.glob(os.path.join(hyp_dir, "[0-9]*.seglst.json"))):
        sid = os.path.basename(p).split(".")[0]
        hyp[sid] = json.load(open(p, encoding="utf-8"))
    ref: dict[str, list[dict]] = {}
    for r in json.load(open(REF_PATH, encoding="utf-8")):
        ref.setdefault(r["session_id"], []).append(r)
    return vecs, metas, hyp, ref


def seg_oracle_label(hyp_recs: list[dict], ref_recs: list[dict]) -> dict[float, str]:
    """返回 {start_time: 该段最大重叠 ref 说话人}（oracle 视角）。"""
    out = {}
    for h in hyp_recs:
        best, bt = None, 0.0
        for r in ref_recs:
            ov = min(h["end_time"], r["end_time"]) - max(h["start_time"], r["start_time"])
            if ov > bt:
                best, bt = r["speaker"], ov
        out[round(h["start_time"], 3)] = best
    return out


def hyp_to_ref_mapping(hyp_recs: list[dict], ref_recs: list[dict]) -> dict[str, str]:
    """会话级 1:1 贪心：按总重叠降序把 hyp 说话人配给 ref 说话人。"""
    ov: dict[tuple[str, str], float] = {}
    for h in hyp_recs:
        for r in ref_recs:
            x = min(h["end_time"], r["end_time"]) - max(h["start_time"], r["start_time"])
            if x > 0:
                ov[(h["speaker"], r["speaker"])] = ov.get((h["speaker"], r["speaker"]), 0.0) + x
    pairs = sorted(ov.items(), key=lambda kv: -kv[1])
    mh, mr, m = set(), set(), {}
    for (hs, rs), _ in pairs:
        if hs not in mh and rs not in mr:
            mh.add(hs); mr.add(rs); m[hs] = rs
    return m


def main() -> int:
    npz, hyp_dir = sys.argv[1], sys.argv[2]
    vecs, metas, hyp, ref = load_inputs(npz, hyp_dir)

    by_sess: dict[str, list[int]] = {}
    for i, m in enumerate(metas):
        by_sess.setdefault(m["session"], []).append(i)

    L = vecs.shape[1]
    # 预计算每段 oracle 对错
    wrong: dict[str, set[float]] = {}
    n_wrong = n_all = 0
    for sid, idx in by_sess.items():
        hr, rr = hyp[sid], ref.get(sid, [])
        omap = seg_oracle_label(hr, rr)
        hmap = hyp_to_ref_mapping(hr, rr)
        w = set()
        for h in hr:
            t = round(h["start_time"], 3)
            oracle_spk = omap.get(t)
            ok = oracle_spk is not None and hmap.get(h["speaker"]) == oracle_spk
            if not ok:
                w.add(t)
        wrong[sid] = w
        n_wrong += len(w); n_all += len(hr)
    print(f"hyp 命名空间口径：{n_wrong}/{n_all} 段 oracle 错（{100*n_wrong/n_all:.1f}%）\n")

    rows = []
    for layer in range(L):
        agree = dis = 0
        dis_wrong = agree_wrong = 0
        sils = []
        for sid, idx in by_sess.items():
            hr = hyp[sid]
            X = vecs[idx, layer, :]
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            K = len({h["speaker"] for h in hr})
            if K < 2 or len(idx) < K:
                continue
            km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(Xn)
            labs = [h["speaker"] for h in hr]
            ul = sorted(set(labs))
            uc = sorted(set(int(x) for x in km.labels_))   # 实际簇 id（可能 < K）
            # 匈牙利：最大化 聚类↔标签 一致
            cost = np.zeros((len(uc), len(ul)))
            for ci, c in enumerate(uc):
                for ui, lab in enumerate(ul):
                    cost[ci, ui] = sum(1 for j, h in enumerate(hr)
                                       if km.labels_[j] == c and h["speaker"] == lab)
            ri, ci = linear_sum_assignment(-cost)
            cl2lab = {uc[int(a)]: ul[int(b)] for a, b in zip(ri, ci)}
            for j, h in enumerate(hr):
                t = round(h["start_time"], 3)
                pred = cl2lab.get(int(km.labels_[j]))   # 未匹配簇 → None → 分歧
                isw = t in wrong[sid]
                if pred == h["speaker"]:
                    agree += 1; agree_wrong += isw
                else:
                    dis += 1; dis_wrong += isw
        tot = agree + dis
        if tot == 0:
            continue
        lift = (dis_wrong / dis) / (n_wrong / n_all) if dis else float("nan")
        rows.append((layer, agree / tot, dis, dis_wrong / dis if dis else 0.0, lift))

    rows.sort(key=lambda r: -r[4])
    print(f"{'层':>3} {'一致率':>7} {'分歧段':>6} {'分歧段错率':>9} {'lift':>6}")
    for layer, agr, nd, dwr, lift in rows[:8]:
        print(f"{layer:>3} {agr*100:>6.1f}% {nd:>6} {dwr*100:>8.1f}% {lift:>6.2f}")
    best = rows[0]
    print(f"\n最优层 {best[0]}：一致率 {best[1]*100:.1f}%，分歧 {best[2]} 段，"
          f"分歧段错率 {best[3]*100:.1f}%（总体 {100*n_wrong/n_all:.1f}%），lift {best[4]:.2f}")
    print("判定：lift≥1.5 且分歧段≥20 才值得建机制；否则死。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
