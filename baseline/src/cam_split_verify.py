"""CAM 引导 + 声纹校验的说话人拆分（v094b / 最终提交的最后一环）。

## 解决什么问题

MOSS-SAT 分支系统性**少估说话人数**：test 每场比 dev ref 分布少 0.43 人，
dev 链路少 0.18~0.33 人。少估的表现是「两个人被并成一个 speaker」，
在 tcpWER 下同时产生 insertion（并进来的词挂错人）和 deletion（真人的词没了），
所以修对一次是双份收益 —— 这也是本机制单场收益能到几十个错误的原因。

## 机制

对每个 session：
1. **触发**：CAM++ 聚类认为的人数 `camN` 比假设的人数 `hypN` 多，且 `camN-hypN == gap`。
2. **候选**：某个 hyp speaker 的分段在 CAM++ 时间轴上横跨 ≥2 个簇，
   且次多簇占该 speaker 总时长 ≥ `min_frac`。
3. **声纹校验（关键）**：把该 speaker 的分段按「是否属于次多簇」分成两半，
   各自取 CAM++ 段级声纹（192 维）的平均向量，算余弦相似度；
   **只有 `sim < max_sim` 才真的拆** —— 即先确认「这两半确实是两个人」。
4. 拆分只改 `speaker` 标签，**不改文本、不改时间戳**。

## 为什么第 3 步不可省

不带声纹校验时（仅 1+2），dev 上四关的第 2 关（整段剔除最幸运的 3 段后仍为收益）
**全档不过** —— 收益集中在少数几场，是典型不可外推的重尾。
加上声纹校验后 `max_sim ≤ 0.60` 的每一格都过第 2 关，`≥ 0.65` 的每一格都不过：
余弦 0.60 是「两个人」与「把一个人劈成两半」的分界。

## 参数来源（均在 dev 上用嵌套交叉验证选定，未在 test 上调过）

出货配置 `min_frac=0.05 / max_sim=0.50 / gap==1`：
dev 触发 9 场、Δ = −0.9822 点、整段剔 top3 后 −0.1320、自助 P=99.2%、改善 7 / 恶化 2（四关全过）。

用法：
    python src/cam_split_verify.py \
        --hyp submit/submission_v065.json \
        --diar output/diar_test_m3_7_0.70 \
        --emb  output/emb_test \
        --out  submit/submission_v094b.json \
        --min-frac 0.05 --max-sim 0.50 --min-gap 1 --max-gap 1
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_FRAC = 0.05      # 次多簇至少占该 speaker 总时长的比例
MAX_SIM = 0.50       # 两半平均声纹的余弦上限；超过则认为是同一个人，不拆
MIN_GAP = 1          # camN - hypN 的下界
MAX_GAP = 1          # camN - hypN 的上界（gap>=2 时 CAM 多半在多估，实测有害）


def load_records(path: str) -> list[dict]:
    """读入 SegLST。path 可以是单个合并文件，也可以是逐 session 的目录。"""
    def _one(p: str) -> list[dict]:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else d["segments"]

    if os.path.isdir(path):
        recs: list[dict] = []
        for name in sorted(os.listdir(path)):
            if name.endswith(".seglst.json"):
                recs += _one(os.path.join(path, name))
        return recs
    return _one(path)


def write_records(records: list[dict], out: str, per_session: bool) -> None:
    """per_session=True 时写成 <out>/<session>.seglst.json，供 bench.py 消费。"""
    if not per_session:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return
    os.makedirs(out, exist_ok=True)
    grouped: dict[str, list] = collections.defaultdict(list)
    for r in records:
        grouped[r["session_id"]].append(r)
    for sid, recs in grouped.items():
        with open(os.path.join(out, f"{sid}.seglst.json"), "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)


def load_diar(diar_dir: str) -> dict[str, dict]:
    out = {}
    for name in sorted(os.listdir(diar_dir)):
        if not name.endswith(".diar.json"):
            continue
        with open(os.path.join(diar_dir, name), encoding="utf-8") as f:
            d = json.load(f)
        out[d["session_id"]] = d
    return out


class EmbStore:
    """CAM++ 段级声纹（times: N×2 秒, embs: N×192）。"""

    def __init__(self, root: str) -> None:
        self.root = root
        self._cache: dict[str, Any] = {}

    def get(self, session_id: str) -> Any:
        if session_id not in self._cache:
            path = os.path.join(self.root, f"{session_id}.emb.npz")
            self._cache[session_id] = np.load(path, allow_pickle=True) if os.path.exists(path) else None
        return self._cache[session_id]

    def segment(self, session_id: str, start: float, end: float) -> Any:
        z = self.get(session_id)
        if z is None:
            return None
        times = z["times"]
        mask = (times[:, 1] > start) & (times[:, 0] < end)
        if not mask.any():
            return None
        v = z["embs"][mask].mean(0)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None


def ranked_by_mass(mass: dict[Any, float]) -> list[tuple[Any, float]]:
    """按时长降序排序；并列时保持插入顺序（与 Counter.most_common 行为一致）。"""
    return sorted(mass.items(), key=lambda kv: -kv[1])


def dominant_cluster(rec: dict, segments: list[dict]) -> tuple[Any, float]:
    """该分段在 CAM++ 时间轴上重叠最多的簇编号。"""
    s, e = float(rec["start_time"]), float(rec["end_time"])
    overlap: dict[Any, float] = collections.defaultdict(float)
    for g in segments:
        o = min(e, g["end"]) - max(s, g["start"])
        if o > 0:
            overlap[g["speaker"]] += o
    if not overlap:
        return None, e - s
    return ranked_by_mass(overlap)[0][0], e - s


def split_session(records, diar, embs, session_id, min_frac, max_sim):
    """返回 (新记录列表, 拆分次数)。只改 speaker 标签。"""
    cam_n = diar["num_speakers"]
    hyp_n = len({r["speaker"] for r in records})
    budget = cam_n - hyp_n

    # speaker -> {CAM++ 簇编号: 该 speaker 落在此簇的总时长（秒）}
    profile: dict[str, dict[Any, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float))
    dom: dict[int, object] = {}
    for i, r in enumerate(records):
        cluster, dur = dominant_cluster(r, diar["segments"])
        dom[i] = cluster
        if cluster is not None:
            profile[r["speaker"]][cluster] += dur

    used = {r["speaker"] for r in records}
    candidates = []
    for spk, counts in profile.items():
        total = sum(counts.values())
        if total <= 0 or len(counts) < 2:
            continue
        ranked = ranked_by_mass(counts)
        minority, minority_dur = ranked[1][0], ranked[1][1]
        frac = minority_dur / total
        if frac < min_frac:
            continue
        a_idx = [i for i, r in enumerate(records) if r["speaker"] == spk and dom[i] == minority]
        b_idx = [i for i, r in enumerate(records) if r["speaker"] == spk and dom[i] != minority]
        a_vecs = [v for v in (embs.segment(session_id, float(records[i]["start_time"]),
                                           float(records[i]["end_time"])) for i in a_idx) if v is not None]
        b_vecs = [v for v in (embs.segment(session_id, float(records[i]["start_time"]),
                                           float(records[i]["end_time"])) for i in b_idx) if v is not None]
        if not a_vecs or not b_vecs:
            continue
        va = np.mean(a_vecs, 0)
        va /= max(float(np.linalg.norm(va)), 1e-9)
        vb = np.mean(b_vecs, 0)
        vb /= max(float(np.linalg.norm(vb)), 1e-9)
        sim = float(va @ vb)
        if sim > max_sim:            # 声纹太像 -> 判为同一个人，不拆
            continue
        candidates.append((frac, spk, minority, sim))

    candidates.sort(reverse=True)
    remap: dict[tuple, str] = {}
    for frac, spk, minority, sim in candidates[: max(0, budget)]:
        n = 1
        while f"spk{n}" in used:
            n += 1
        used.add(f"spk{n}")
        remap[(spk, minority)] = f"spk{n}"
        logger.debug("%s: split %s (minority frac=%.2f, sim=%.3f) -> %s",
                     session_id, spk, frac, sim, f"spk{n}")

    out = []
    for i, r in enumerate(records):
        key = (r["speaker"], dom.get(i))
        if key in remap:
            r = dict(r)
            r["speaker"] = remap[key]
        out.append(r)
    return out, len(remap)


def run(hyp, diar_dir, emb_root, min_frac, max_sim, min_gap, max_gap):
    records = load_records(hyp)
    diars = load_diar(diar_dir)
    embs = EmbStore(emb_root)

    by_session: dict[str, list] = collections.defaultdict(list)
    for r in records:
        by_session[r["session_id"]].append(r)

    out, n_split, n_sess = [], 0, 0
    for sid, recs in by_session.items():
        d = diars.get(sid)
        if d is None:
            out += recs
            continue
        gap = d["num_speakers"] - len({r["speaker"] for r in recs})
        if not (min_gap <= gap <= max_gap):
            out += recs
            continue
        new, k = split_session(recs, d, embs, sid, min_frac, max_sim)
        out += new
        n_split += k
        n_sess += 1 if k else 0

    out.sort(key=lambda r: (r["session_id"], float(r["start_time"])))
    return out, n_split, n_sess


def check(original: list[dict], out: list[dict]) -> None:
    """出货前自检：只允许 speaker 变化，其余一律不变。

    用显式抛错而非 assert —— assert 在 `python -O` 下会被剥离，
    而这是提交前最后一道完整性闸门，不能允许被静默跳过。
    """
    def bad(msg: str) -> None:
        raise ValueError(f"完整性自检失败：{msg}")

    if len(out) != len(original):
        bad(f"记录条数变了：{len(original)} -> {len(out)}")
    a = sorted(original, key=lambda r: (r["session_id"], float(r["start_time"])))
    if [r["session_id"] for r in a] != [r["session_id"] for r in out]:
        bad("session 集合或顺序变了")
    if [r["words"] for r in a] != [r["words"] for r in out]:
        bad("文本被改动（本步骤只允许改 speaker）")
    if [r["start_time"] for r in a] != [r["start_time"] for r in out]:
        bad("时间戳被改动（本步骤只允许改 speaker）")

    by: dict[str, set] = collections.defaultdict(set)
    for r in out:
        by[r["session_id"]].add(r["speaker"])
    for sid, labels in by.items():
        if not all(x.startswith("spk") and x[3:].isdigit() for x in labels):
            bad(f"{sid}: 说话人标签不合规 {sorted(labels)}")
        nums = sorted(int(x[3:]) for x in labels)
        if nums != list(range(1, len(nums) + 1)):
            bad(f"{sid}: 说话人编号不连续 {nums}（应为 1..N）")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--hyp", required=True,
                    help="输入 SegLST：合并文件（如 submit/submission_v065.json）或逐 session 目录")
    ap.add_argument("--diar", required=True, help="CAM++ 分离结果目录（*.diar.json）")
    ap.add_argument("--emb", required=True, help="CAM++ 段级声纹目录（*.emb.npz）")
    ap.add_argument("--out", required=True, help="输出 SegLST（--per-session 时为目录）")
    ap.add_argument("--per-session", action="store_true",
                    help="按 session 分文件写出，供 src/bench.py --cand 消费（dev 验关用）")
    ap.add_argument("--min-frac", type=float, default=MIN_FRAC)
    ap.add_argument("--max-sim", type=float, default=MAX_SIM)
    ap.add_argument("--min-gap", type=int, default=MIN_GAP)
    ap.add_argument("--max-gap", type=int, default=MAX_GAP)
    args = ap.parse_args()

    original = load_records(args.hyp)
    out, n_split, n_sess = run(args.hyp, args.diar, args.emb,
                               args.min_frac, args.max_sim, args.min_gap, args.max_gap)
    check(original, out)
    write_records(out, args.out, args.per_session)
    logger.info("拆分 %d 次，覆盖 %d 个 session；写出 %d 条 -> %s", n_split, n_sess, len(out), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
