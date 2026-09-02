"""Oracle 探针：把 ref 的独立语气词段补建进 hyp，测「新增整段」轴的上界。

## 为什么有这条轴（2026-08-30，方向 1 的 GO/NO-GO）

三种既有封条都不覆盖「新增独立段」这个操作：
- 08-14 filler_backfill 封的是**往已有段内插词**（oracle +0.463 为负）；
- 08-29 格式轴封的是**改动已有词的时间/切分**（恰好 0）；
- 08-29 归属 oracle（attr_bucket）只映射**已存在的 1339 个段**。

量级线索：ref 的 嗯 有 101+ 个是独立成段应答（08-13）；v028 时代语气词缺口
嗯 +116 / 全部 +187 词（≈0.96 点）；当前链 hyp 仍比 ref 少 ~13.5% 语音时长。
ref 自重叠 = 0 ⇒ 语气词时间点上只有它在响 ⇒ 时间+说话人都对的补建段
不碰任何已有词的对齐，机械上零破坏面。

## 三个口径

| 口径 | 插入集合 | 含义 |
|---|---|---|
| all | 全部 132 个 ref 独立语气词段（每 ref 说话人一个新标签）| 粗上界（含重复伤害）|
| heur | 其中「无同语气词 hyp 段重叠 ≥50%」的 | 近似「只补缺失」|
| greedy | 逐段贪心：单段会话级实测，降错误才保留 | **真 oracle**（含位移互作）|

插入段 = ref 段原样（speaker 换成 `<ref_spk>#o` 避免与 hyp 标签碰撞；
words/时间逐字段照抄 ⇒ 对齐平凡精确）。meeteval 的说话人指派是最大化匹配，
新增 hyp 说话人在会话级只会改善或持平指派，不会伤害已有映射。

用法：
    python src/oracle_standalone_fillers.py --hyp /tmp/v065_norm --wav-root <unused> --out /tmp/ora
    # 依赖根 .venv 的 meeteval-wer（PATH 假定与 cv_gate.py 相同布局）
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

FILLERS = set("嗯噢唉啊呀呃吧嘛哦喔哎诶呢")
MAX_WORDS = 3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE_DIR)
REF = os.path.join(ROOT, "data/extracted/dev/dev/ref.seglst.json")
MEET = os.path.join(ROOT, ".venv/bin/meeteval-wer")


def is_filler_seg(rec: dict) -> bool:
    w = rec["words"]
    return 0 < len(w) <= MAX_WORDS and set(w) <= FILLERS


def evaluate(records: list[dict], tag: str) -> tuple[float, dict]:
    """整 dev 一次 tcpwer，返回 (error_rate, per_reco errors)。"""
    with tempfile.TemporaryDirectory() as td:
        h = os.path.join(td, f"h_{tag}.json")
        json.dump(records, open(h, "w", encoding="utf-8"), ensure_ascii=False)
        subprocess.run([MEET, "tcpwer", "-r", REF, "-h", h, "--collar", "5"],
                       capture_output=True, check=True)
        res = json.load(open(h.replace(".json", "_tcpwer.json"), encoding="utf-8"))
        per = json.load(open(h.replace(".json", "_tcpwer_per_reco.json"), encoding="utf-8"))
        return float(res["error_rate"]) * 100, {k: int(v["errors"]) for k, v in per.items()}


def eval_one_session(sid: str, recs: list[dict], ref_by_sid: dict) -> int:
    """单会话 tcpwer，返回 errors（贪心用）。"""
    with tempfile.TemporaryDirectory() as td:
        h = os.path.join(td, "h.json")
        r = os.path.join(td, "r.json")
        json.dump(recs, open(h, "w", encoding="utf-8"), ensure_ascii=False)
        json.dump(ref_by_sid[sid], open(r, "w", encoding="utf-8"), ensure_ascii=False)
        subprocess.run([MEET, "tcpwer", "-r", r, "-h", h, "--collar", "5"],
                       capture_output=True, check=True)
        res = json.load(open(h.replace(".json", "_tcpwer.json"), encoding="utf-8"))
        return int(res["errors"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True, help="基线 hyp 目录（如 /tmp/v065_norm）")
    args = ap.parse_args()

    ref = json.load(open(REF, encoding="utf-8"))
    ref_by_sid: dict[str, list[dict]] = {}
    for r in ref:
        ref_by_sid.setdefault(r["session_id"], []).append(r)

    hyp: dict[str, list[dict]] = {}
    for p in sorted(glob.glob(os.path.join(args.hyp, "[0-9]*.seglst.json"))):
        sid = os.path.basename(p).split(".")[0]
        hyp[sid] = json.load(open(p, encoding="utf-8"))

    # 候选 = ref 独立语气词段
    cands: dict[str, list[dict]] = {}
    for sid, recs in ref_by_sid.items():
        cs = [r for r in recs if is_filler_seg(r)]
        if cs:
            cands[sid] = cs
    n_cand = sum(len(v) for v in cands.values())
    print(f"候选：{n_cand} 个 ref 独立语气词段 / {len(cands)} session")

    def inserted(sid: str, keep: set[tuple[str, float, str]] | None = None) -> list[dict]:
        out = list(hyp[sid])
        for r in cands.get(sid, []):
            if keep is not None and (sid, r["start_time"], r["words"]) not in keep:
                continue
            out.append({**r, "speaker": f"{r['speaker']}#o"})
        return out

    # --- O-all ---
    all_recs = [r for sid in sorted(hyp) for r in inserted(sid)]
    er_all, _ = evaluate(all_recs, "all")

    # --- O-heur：跳过已被同语气词 hyp 段覆盖 ≥50% 的 ---
    heur_recs, n_heur = [], 0
    for sid in sorted(hyp):
        keep: list[dict] = []
        hs = hyp[sid]
        for r in cands.get(sid, []):
            mid = (r["start_time"] + r["end_time"]) / 2
            span = max(r["end_time"] - r["start_time"], 1e-6)
            covered = any(
                max(0.0, min(h["end_time"], r["end_time"]) - max(h["start_time"], r["start_time"])) / span >= 0.5
                and any(c in h["words"] for c in FILLERS)
                for h in hs
            )
            if not covered:
                keep.append(r)
        n_heur += len(keep)
        out = list(hyp[sid]) + [{**r, "speaker": f"{r['speaker']}#o"} for r in keep]
        heur_recs.extend(out)
    er_heur, _ = evaluate(heur_recs, "heur")

    # --- O-greedy：逐段贪心（真 oracle，含位移互作）---
    er_base, per_base = evaluate([r for sid in sorted(hyp) for r in hyp[sid]], "base")
    keep: set[tuple[str, float, str]] = set()
    n_keep = 0
    for sid in sorted(cands):
        cur = hyp[sid]
        cur_err = eval_one_session(sid, cur, ref_by_sid)
        for r in sorted(cands[sid], key=lambda x: -(x["end_time"] - x["start_time"])):
            trial = cur + [{**r, "speaker": f"{r['speaker']}#o"}]
            e = eval_one_session(sid, trial, ref_by_sid)
            if e < cur_err:
                cur, cur_err = trial, e
                keep.add((sid, r["start_time"], r["words"]))
                n_keep += 1
    greedy_recs = [r for sid in sorted(hyp) for r in inserted(sid, keep)]
    er_greedy, _ = evaluate(greedy_recs, "greedy")

    print(f"\n基线           : {er_base:.4f}%")
    print(f"O-all  ({n_cand:3d} 段): {er_all:.4f}%   Δ {er_all-er_base:+.4f}")
    print(f"O-heur ({n_heur:3d} 段): {er_heur:.4f}%   Δ {er_heur-er_base:+.4f}")
    print(f"O-greedy({n_keep:3d} 段): {er_greedy:.4f}%   Δ {er_greedy-er_base:+.4f}   ← 真 oracle")
    need = 0.247
    verdict = "GO" if -(er_greedy - er_base) >= 0.4 else ("WEAK" if -(er_greedy - er_base) >= 0.15 else "NO-GO")
    print(f"\n需求 0.247 dev 点；门槛 GO≥0.4 / WEAK≥0.15 → 判定: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
