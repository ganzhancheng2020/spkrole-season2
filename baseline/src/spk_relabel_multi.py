"""归属轴·双判官纯重分配：两个独立声纹模型各自打分后融合，再决定是否改标签。

## 为什么要第二个判官（2026-08-14）

单判官（ERes2NetV2）的重贴标签已线上验证有效（v037 −0.00280 / v038 −0.00064），
但**只吃掉重贴空间的 12%**：

| 层次 | 全 dev tcpWER | 空间 |
|---|---|---|
| 当前基准 | 15.6425% | — |
| **完美重贴标签（不拆段）** | **12.2487%** | **3.39 点** |
| 纯文本下界（归属全对） | 11.6522% | 拆段再赚仅 0.60 点 |

**88% 的重贴空间没拿到，且不需要拆段。** 而放宽 `MARGIN` 已证实无效 ——
边际段只值 1.25 错误/段（平均 6.5），**精度掉得比覆盖率涨得快**：

| MARGIN | 改动段数 | 每段省 | 恶化段 |
|---|---|---|---|
| 0.10 | 42 | −5.0 | 5 |
| 0.15 | 30 | **−6.5** | 2 |
| 0.25 | 19 | −8.2 | 1 |

→ **瓶颈是判官的判别力本身，不是阈值。** 加一个**独立**判官引入的是新信息，
而调阈值、改档案聚合方式都只是重新加工同一份嵌入。

## 两个判官的独立性

- `iic/speech_eres2netv2_sv_zh-cn_16k-common`（ERes2NetV2，v037/v038 在用）
- `iic/speech_campplus_sv_zh-cn_16k-common`（CAM++ SV，**不同架构、不同训练**）

⚠️ CAM++ **diarization** 模型是上游产出 CAM 分支标签的那个；这里用的是它的
**speaker-verification** 变体，只输出嵌入向量、不产出标签 —— 但仍与 CAM 分支同源，
故对 CAM 分支的独立性弱于对 MOSS 分支。**评测时按分支分开看。**

## 融合方式（`FUSE`）

每个判官对每段给出 `delta = 挑战者得分 − 在位者得分`（留一法，扣掉本段自己的窗）：

- `mean`（默认）：两判官 delta 的均值 ≥ MARGIN 才换 —— **提精度且不必然掉覆盖**
- `and`：取两者较小值 ≥ MARGIN —— 最保守，覆盖必降
- `or`：取两者较大值 ≥ MARGIN —— 覆盖最高，精度最低

⚠️ 两个模型的 delta 尺度不同，融合前按各判官**该 session 内 delta 的标准差**归一化，
否则尺度大的那个判官会主导结果。

**不拆分、不新增说话人**（说话人数守恒），与 `spk_relabel.py SPLIT_THR=99` 同语义。

用法：
    cd baseline
    FUSE=mean MARGIN=0.15 diarizen_venv/bin/python src/spk_relabel_multi.py \\
        output/hyp_sw_m3_7_0.70 ../data/extracted/dev/dev/wav output/camre_multi

已存在的输出文件会被跳过（可断点续跑）。
"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np  # type: ignore[import-not-found]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spk_relabel import embed_windows  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JUDGE_A = os.environ.get("JUDGE_A", "iic/speech_eres2netv2_sv_zh-cn_16k-common")
JUDGE_B = os.environ.get("JUDGE_B", "iic/speech_campplus_sv_zh-cn_16k-common")
MARGIN = float(os.environ.get("MARGIN", "0.15"))
FUSE = os.environ.get("FUSE", "mean")
MIN_PROFILE_WIN = int(os.environ.get("MIN_PROFILE_WIN", "2"))


def build(model_id: str):
    from modelscope.pipelines import pipeline  # type: ignore[import-not-found]
    from modelscope.utils.constant import Tasks  # type: ignore[import-not-found]

    logger.info("加载判官: %s", model_id)
    return pipeline(task=Tasks.speaker_verification, model=model_id)


def deltas(recs: list[dict], embs: np.ndarray, owner: np.ndarray) -> dict[int, dict[str, float]]:
    """单判官：返回 {段序号: {挑战者说话人: delta}}，delta = 挑战者 − 在位者。"""
    cur = [r["speaker"] for r in recs]
    out: dict[int, dict[str, float]] = {}
    if len(embs) == 0 or len(set(cur)) < 2:
        return out
    win_of = [np.where(owner == i)[0] for i in range(len(recs))]
    prof: dict[str, np.ndarray] = {}
    cnt: dict[str, int] = {}
    for i, w in enumerate(win_of):
        if len(w) == 0:
            continue
        s = cur[i]
        prof[s] = prof.get(s, np.zeros(embs.shape[1])) + embs[w].sum(0)
        cnt[s] = cnt.get(s, 0) + len(w)
    usable = {s for s in prof if cnt[s] >= MIN_PROFILE_WIN}
    if len(usable) < 2:
        return out
    for i, w in enumerate(win_of):
        if len(w) == 0 or cur[i] not in usable:
            continue
        sc: dict[str, float] = {}
        for s in usable:
            v, n = prof[s].copy(), cnt[s]
            if cur[i] == s:                       # 留一：扣掉本段自己的窗
                v, n = v - embs[w].sum(0), n - len(w)
            if n < MIN_PROFILE_WIN:
                continue
            p = v / n
            sc[s] = float((embs[w] @ (p / (np.linalg.norm(p) + 1e-9))).mean())
        if cur[i] not in sc or len(sc) < 2:
            continue
        inc = sc[cur[i]]
        out[i] = {s: v - inc for s, v in sc.items() if s != cur[i]}
    return out


def zscale(d: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
    """按该 session 内全部 delta 的标准差归一，消除两判官的尺度差异。"""
    vals = [v for m in d.values() for v in m.values()]
    if not vals:
        return d
    sd = float(np.std(vals)) or 1.0
    return {i: {s: v / sd for s, v in m.items()} for i, m in d.items()}


def fuse(da: dict, db: dict, cur: list[str]) -> list[str]:
    new = list(cur)
    for i in set(da) & set(db):
        cands = set(da[i]) & set(db[i])
        if not cands:
            continue
        if FUSE == "and":
            ok = {s: min(da[i][s], db[i][s]) for s in cands}
        elif FUSE == "or":
            ok = {s: max(da[i][s], db[i][s]) for s in cands}
        else:
            ok = {s: (da[i][s] + db[i][s]) / 2 for s in cands}
        best_v, best_s = max((v, s) for s, v in ok.items())
        if best_v >= MARGIN:
            new[i] = best_s
    return new


def main() -> int:
    if len(sys.argv) < 4:
        logger.error("用法见模块 docstring：%s", __doc__)
        return 2
    pred_dir, wav_dir, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    sids = sorted(
        f.split(".")[0]
        for f in os.listdir(pred_dir)
        if f.endswith(".seglst.json") and f[0].isdigit()
    )
    if not sids:
        logger.error("预测目录为空：%s", pred_dir)
        return 1
    logger.info("配方 FUSE=%s MARGIN=%.2f", FUSE, MARGIN)

    sv_a, sv_b = build(JUDGE_A), build(JUDGE_B)
    total = 0
    for k, sid in enumerate(sids, 1):
        out_path = f"{out_dir}/{sid}.seglst.json"
        if os.path.exists(out_path):
            continue
        with open(f"{pred_dir}/{sid}.seglst.json", encoding="utf-8") as fh:
            recs = sorted(json.load(fh), key=lambda r: r["start_time"])
        wav = f"{wav_dir}/{sid}.wav"
        ea, oa = embed_windows(sv_a, wav, recs)
        eb, ob = embed_windows(sv_b, wav, recs)
        cur = [r["speaker"] for r in recs]
        labels = fuse(zscale(deltas(recs, ea, oa)), zscale(deltas(recs, eb, ob)), cur)
        n = sum(1 for a, b in zip(cur, labels) if a != b)
        total += n
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump([{**h, "speaker": s} for h, s in zip(recs, labels)],
                      fh, ensure_ascii=False, indent=2)
        logger.info("[%d/%d] %s: 改写 %d 段（累计 %d）", k, len(sids), sid, n, total)
    logger.info("MULTI RELABEL DONE 改写 %d 段 → %s", total, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
