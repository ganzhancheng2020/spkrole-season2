"""给 hyp 的每个字打上**真实时间戳**——把 FireRed 缓存的字级时间戳对齐迁移过来。

## 为什么需要它

2026-08-26 Phase 0 定位到归属轴的瓶颈是**词级文本归属**：

| 口径 | Δ | 切开段数 |
|---|---|---|
| oracle 切 + **oracle 词级分配** | −3.373 | 35 |
| oracle 切 + **时间比例分配** | **+0.704** | 242 |

差 4.078 点。「时间比例分配」= 假设段内的字在时间上均匀分布，这个假设在
「一段里两个人各说一半」时错得最狠 —— 而那正好是唯一值得切的那类段。

**要切段就必须知道每个字什么时候被说出来。** 本模块提供这个。

## 为什么可以白拿

`output/fr_retext_dev_ts/*.ts.json` 里存着 FireRedASR2 对 106 场**整场**转写的
字级时间戳（`fr_assign_cached.py` 的文件头说明了缓存来由）。转写只依赖音频，
与我们的切分无关，**对任何新切分都成立，不必重跑 ASR（纯 CPU）**。

但 ts 里的字是 **FireRed 的**，我们要打时间戳的字是 **MOSS 的**，两者并不逐字相同。
好在 2026-08-28 实测两源**逐段平局率 93.9%** —— 绝大多数字是一样的，
所以字级对齐（`difflib.SequenceMatcher`）能把时间戳可靠地迁移过去。

⚠️ 与已有的 `word_spk_ec10.py` / `word_assign_ec10.py` 不同：那两个把 **FireRed 自己的词**
按时间戳贴到 v028 段上，且只处理 41 段挑出来的候选；本模块是把时间戳**迁移到另一套
（MOSS）字序列**上，对全量段成立。

## 构造

1. 按段序把整场 hyp 的字拼成一条序列（记住每个字属于哪条记录、第几位）
2. 与 FireRed 字序列做最长公共子序列对齐
3. **对齐上的字**直接取 FireRed 的 `(t0, t1)`
4. **没对齐上的字**在相邻两个锚点之间按位置线性插值；段首/段尾无锚点时退到该段边界
5. 全部时间戳**夹紧到所属记录的 [start_time, end_time]**，并强制单调不减

> ⚠️ 只有第 3 步是真实测量，第 4 步是插值。`coverage()` 给出真实锚点占比，
> 下游机制应当只在锚点覆盖高的段上动手。
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path


def load_ts(ts_path: Path) -> list[tuple[str, float, float]]:
    raw = json.load(open(ts_path, encoding="utf-8"))
    return [(str(w), float(a), float(b)) for w, a, b in raw]


def session_chars(recs: list[dict]) -> list[tuple[int, int, str]]:
    """→ [(rec_idx, char_idx_in_rec, char)]，按段序、段内序。"""
    out: list[tuple[int, int, str]] = []
    for i, r in enumerate(recs):
        toks = str(r.get("words", "")).split()
        for j, t in enumerate(toks):
            out.append((i, j, t))
    return out


def transfer(recs: list[dict], ts: list[tuple[str, float, float]]) -> list[list[tuple[float, float, bool]]]:
    """为每条记录返回 [(t0, t1, is_anchor), ...]，与该记录的字一一对应。"""
    hyp = session_chars(recs)
    a_seq = [c for _, _, c in hyp]
    b_seq = [w for w, _, _ in ts]

    times: list[tuple[float, float] | None] = [None] * len(hyp)
    anchor = [False] * len(hyp)
    if b_seq:
        sm = difflib.SequenceMatcher(a=a_seq, b=b_seq, autojunk=False)
        for i, j, n in sm.get_matching_blocks():
            for k in range(n):
                times[i + k] = (ts[j + k][1], ts[j + k][2])
                anchor[i + k] = True

    per_rec: list[list[tuple[float, float, bool]]] = [[] for _ in recs]
    idx_of: list[list[int]] = [[] for _ in recs]
    for gi, (ri, _, _) in enumerate(hyp):
        idx_of[ri].append(gi)

    for ri, rec in enumerate(recs):
        gis = idx_of[ri]
        if not gis:
            continue
        s, e = float(rec["start_time"]), float(rec["end_time"])
        n = len(gis)
        known: dict[int, tuple[float, float]] = {}
        for p, gi in enumerate(gis):
            tv = times[gi]
            if tv is not None:
                known[p] = tv
        for p, gi in enumerate(gis):
            t = times[gi]
            if t is None:
                # 用 -1 作哨兵而非 None：键 p 恒 >= 0，且避免 Optional 传播
                prev = max((k for k in known if k < p), default=-1)
                nxt = min((k for k in known if k > p), default=-1)
                if prev < 0 and nxt < 0:                  # 整段无锚点 → 均分
                    w = (e - s) / n
                    t = (s + p * w, s + (p + 1) * w)
                elif prev < 0:
                    t0 = known[nxt][0]
                    w = max((t0 - s) / max(nxt, 1), 1e-3)
                    t = (t0 - (nxt - p) * w, t0 - (nxt - p - 1) * w)
                elif nxt < 0:
                    t1 = known[prev][1]
                    w = max((e - t1) / max(n - prev - 1, 1), 1e-3)
                    t = (t1 + (p - prev - 1) * w, t1 + (p - prev) * w)
                else:
                    ta, tb = known[prev][1], known[nxt][0]
                    w = (tb - ta) / max(nxt - prev, 1)
                    t = (ta + (p - prev - 1) * w, ta + (p - prev) * w)
            t0 = min(max(float(t[0]), s), e)
            t1 = min(max(float(t[1]), t0), e)
            per_rec[ri].append((t0, t1, anchor[gi]))
        cur = s
        fixed = []
        for t0, t1, an in per_rec[ri]:
            t0 = max(t0, cur)
            t1 = max(t1, t0)
            fixed.append((t0, t1, an))
            cur = t0
        per_rec[ri] = fixed
    return per_rec


def annotate_dir(pred_dir: Path, ts_dir: Path) -> dict[str, tuple[list[dict], list[list[tuple[float, float, bool]]]]]:
    out: dict[str, tuple[list[dict], list[list[tuple[float, float, bool]]]]] = {}
    for p in sorted(pred_dir.glob("[0-9]*.seglst.json")):
        sid = p.name.split(".")[0]
        tsp = ts_dir / f"{sid}.ts.json"
        recs = json.load(open(p, encoding="utf-8"))
        ts = load_ts(tsp) if tsp.exists() else []
        out[sid] = (recs, transfer(recs, ts))
    return out


def coverage(ann: dict) -> tuple[int, int]:
    tot = anc = 0
    for _sid, (_recs, per) in ann.items():
        for lst in per:
            for _t0, _t1, a in lst:
                tot += 1
                anc += int(a)
    return anc, tot
