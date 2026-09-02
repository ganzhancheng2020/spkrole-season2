"""方向 3 续：分歧段的翻标签置信度排序与 precision@k 曲线。

信号（analyze_hidden_spk.py 已证）：层 19 聚类分歧段的 oracle 错率 38.2%（lift 3.15）。
本脚本回答：**能不能只翻高置信子集，让边际精度 >50% 拒绝线**（v039 教训：边际命中 <50% 一律拒绝）。

边距定义（每分歧段）：margin = cos(v, 所属簇质心) − cos(v, 自己标签对应簇质心)。
margin 越大 = 内部几何越支持「这段属于另一个说话人」。

输出：按 margin 降序的累计精度/累计净词错误近似，以及推荐工作点。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE_DIR)
REF_PATH = os.path.join(ROOT, "data/extracted/dev/dev/ref.seglst.json")

LAYER = int(os.environ.get("LAYER", "19"))


def seg_oracle_label(hyp_recs, ref_recs):
    out = {}
    for h in hyp_recs:
        best, bt = None, 0.0
        for r in ref_recs:
            ov = min(h["end_time"], r["end_time"]) - max(h["start_time"], r["start_time"])
            if ov > bt:
                best, bt = r["speaker"], ov
        out[round(h["start_time"], 3)] = best
    return out


def hyp_to_ref_mapping(hyp_recs, ref_recs):
    ov = {}
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
    d = np.load(npz)
    vecs = d["vecs"].astype(np.float32)
    metas = json.loads(str(d["meta"]))
    hyp: dict[str, list[dict]] = {}
    for p in sorted(glob.glob(os.path.join(hyp_dir, "[0-9]*.seglst.json"))):
        sid = os.path.basename(p).split(".")[0]
        hyp[sid] = json.load(open(p, encoding="utf-8"))
    ref: dict[str, list[dict]] = {}
    for r in json.load(open(REF_PATH, encoding="utf-8")):
        ref.setdefault(r["session_id"], []).append(r)

    by_sess: dict[str, list[int]] = {}
    for i, m in enumerate(metas):
        by_sess.setdefault(m["session"], []).append(i)

    rows = []  # (margin, session, start_time, n_words, is_wrong, flip_target)
    for sid, idx in by_sess.items():
        hr = hyp[sid]
        K = len({h["speaker"] for h in hr})
        if K < 2 or len(idx) != len(hr):
            continue
        X = vecs[idx, LAYER, :]
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(Xn)
        ul = sorted({h["speaker"] for h in hr})
        uc = sorted({int(x) for x in km.labels_})
        cost = np.zeros((len(uc), len(ul)))
        for ci, c in enumerate(uc):
            for ui, lab in enumerate(ul):
                cost[ci, ui] = sum(1 for j, h in enumerate(hr)
                                   if km.labels_[j] == c and h["speaker"] == lab)
        ri, ci = linear_sum_assignment(-cost)
        cl2lab = {uc[int(a)]: ul[int(b)] for a, b in zip(ri, ci)}
        lab2cl = {v: k for k, v in cl2lab.items()}
        cents = {}
        for c in uc:
            m = Xn[km.labels_ == c]
            cents[c] = m.mean(axis=0)
            cents[c] /= (np.linalg.norm(cents[c]) + 1e-8)
        omap = seg_oracle_label(hr, ref.get(sid, []))
        hmap = hyp_to_ref_mapping(hr, ref.get(sid, []))
        for j, h in enumerate(hr):
            t = round(h["start_time"], 3)
            c_assigned = int(km.labels_[j])
            pred = cl2lab.get(c_assigned)
            if pred == h["speaker"] or pred is None or h["speaker"] not in lab2cl:
                continue
            c_own = lab2cl[h["speaker"]]
            margin = float(Xn[j] @ cents[c_assigned] - Xn[j] @ cents[c_own])
            oracle_spk = omap.get(t)
            is_wrong = not (oracle_spk is not None and hmap.get(h["speaker"]) == oracle_spk)
            n_words = len(h["words"])
            # 翻向是否正确：目标簇的标签映射到的 ref 说话人 == 段的 oracle 说话人
            flip_right = is_wrong and oracle_spk is not None and \
                hmap.get(pred) == oracle_spk
            rows.append((margin, sid, t, n_words, is_wrong, flip_right, pred))

    rows.sort(key=lambda r: -r[0])
    tot_wrong = sum(1 for r in rows if r[4])
    print(f"层 {LAYER}：分歧可翻段 {len(rows)}，其中 oracle 错 {tot_wrong}（{100*tot_wrong/len(rows):.1f}%）")
    print(f"{'k':>4} {'累计精度':>7} {'其中翻对':>7} {'净词错近似':>9}")
    cum_right = 0
    for k in (10, 20, 30, 40, 60, 80, len(rows)):
        k = min(k, len(rows))
        sub = rows[:k]
        acc = sum(1 for r in sub if r[4]) / k
        # 净词错近似：翻对段省其词数；翻错段（原对→错）赔其词数；oracle 错但翻向也错 ≈ 换个错法（计 0）
        net = sum(-r[3] if r[5] else (r[3] if (r[4] and not r[5]) else 0.0) for r in sub)
        cum_right = sum(1 for r in sub if r[5])
        print(f"{k:>4} {acc*100:>6.1f}% {cum_right:>7} {net:>+9.0f}")
    # 推荐工作点：最后一个 累计精度≥60% 的 k
    best_k, best_acc = 0, 1.0
    for k in range(1, len(rows) + 1):
        acc = sum(1 for r in rows[:k] if r[4]) / k
        if acc >= 0.6:
            best_k, best_acc = k, acc
        else:
            break
    print(f"\n精度≥60% 的最大工作点：k={best_k}（精度 {best_acc*100:.1f}%），"
          f"margin 门槛 = {rows[best_k-1][0]:.4f}" if best_k else "\n无任何 k 达到 60% 精度")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
