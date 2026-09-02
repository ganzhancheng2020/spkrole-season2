"""方向 3 机制化：按层 19 聚类分歧 + margin 门槛翻标签，产出候选目录。

机制：K=说话人数 KMeans（余弦归一，seed 0）→ 匈牙利匹配簇↔标签 →
分歧段计算 margin = cos(向量,所属簇质心) − cos(向量,自己标签对应簇质心)，
margin ≥ THRESH 才翻到目标标签。end-to-end（pick/normalize 之后的 tcpWER）才是裁决，
词级代理不算数。

用法：
    python src/spk_hidden_flip.py --npz /tmp/spk_hidden.npz --hyp output/_cv \
        --out output/_cv_flip --thresh 0.0673
    # thresh 来自 analyze_hidden_margin.py 的 60%-精度工作点（dev 上选定，诚实性缺陷已记录）
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LAYER = int(os.environ.get("LAYER", "19"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresh", type=float, required=True)
    args = ap.parse_args()

    d = np.load(args.npz)
    vecs = d["vecs"].astype(np.float32)
    metas = json.loads(str(d["meta"]))
    os.makedirs(args.out, exist_ok=True)

    by_sess: dict[str, list[int]] = {}
    for i, m in enumerate(metas):
        by_sess.setdefault(m["session"], []).append(i)

    n_flip = n_sess = 0
    for p in sorted(glob.glob(os.path.join(args.hyp, "[0-9]*.seglst.json"))):
        sid = os.path.basename(p).split(".")[0]
        recs = json.load(open(p, encoding="utf-8"))
        idx = by_sess.get(sid)
        flips: dict[int, str] = {}
        if idx is not None and len(idx) == len(recs):
            K = len({h["speaker"] for h in recs})
            if K >= 2:
                X = vecs[idx, LAYER, :]
                Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
                km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(Xn)
                ul = sorted({h["speaker"] for h in recs})
                uc = sorted({int(x) for x in km.labels_})
                cost = np.zeros((len(uc), len(ul)))
                for ci, c in enumerate(uc):
                    for ui, lab in enumerate(ul):
                        cost[ci, ui] = sum(1 for j, h in enumerate(recs)
                                           if km.labels_[j] == c and h["speaker"] == lab)
                ri, ci = linear_sum_assignment(-cost)
                cl2lab = {uc[int(a)]: ul[int(b)] for a, b in zip(ri, ci)}
                lab2cl = {v: k for k, v in cl2lab.items()}
                cents = {}
                for c in uc:
                    m = Xn[km.labels_ == c]
                    cents[c] = m.mean(axis=0)
                    cents[c] /= (np.linalg.norm(cents[c]) + 1e-8)
                for j, h in enumerate(recs):
                    c_assigned = int(km.labels_[j])
                    pred = cl2lab.get(c_assigned)
                    if pred == h["speaker"] or pred is None or h["speaker"] not in lab2cl:
                        continue
                    c_own = lab2cl[h["speaker"]]
                    margin = float(Xn[j] @ cents[c_assigned] - Xn[j] @ cents[c_own])
                    if margin >= args.thresh:
                        flips[j] = pred
        if flips:
            n_sess += 1
            n_flip += len(flips)
            for j, lab in flips.items():
                recs[j] = {**recs[j], "speaker": lab}
        with open(os.path.join(args.out, os.path.basename(p)), "w", encoding="utf-8") as fh:
            json.dump(recs, fh, ensure_ascii=False, indent=1)

    logger.info("SPK_HIDDEN_FLIP L%d thresh=%.4f: %d 段翻标签 / %d session", LAYER, args.thresh, n_flip, n_sess)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
