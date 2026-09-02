"""选源规则的**嵌套验证**搜索——在忠实（未污染）分支对上做。

## 为什么现在才能做

`dev-test-chain-mismatch` 记录：dev 的 CAM 分支 `cam_re = relabel(v026_pick)`
**含 MOSS 内容**，历史上 5 个路由机制全部在这个错误的分支对上评测，
**从未在 test 真正面对的问题上被测过**。

2026-08-28 用纯 CPU 重建了未污染的 CAM 分支并与档案 `truepick` 认亲成功
（16.2544% vs 16.229%），选源轴才第一次可以合法研究。

## 为什么可以用解析式评分

tcpWER = Σ错误 / Σ长度，且两支的 ref 长度相同。
所以「把某些 session 路由给 CAM」的分数可以由逐 session 错误数直接算出，
**不必重跑 meeteval**。已验证：解析法复算当前规则 = 15.9459%，与实测逐位相同。

## 纪律

- **外层折只用来评估，阈值只在内层折上选**（`gate2-not-overridable` /
  v052 / v053 / v036 三次的死因都是「在评测集上择优」）
- 只有 32 个正例（CAM 真赢的 session），过拟合风险极高 ——
  任何在全量上看起来的增益，必须先看嵌套后还剩多少
"""
from __future__ import annotations

import json
import logging
import os
import subprocess as sp
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT = BASE_DIR.parent
MEET = ROOT / ".venv/bin/meeteval-wer"          # 评分只走根 venv，见 CLAUDE.md
TMP = Path(tempfile.gettempdir())
# CV 参考集（bench.py 同款）。可用 CV_REF 覆盖。
REF = Path(os.environ.get("CV_REF", TMP / "cv_ref.json"))
MOSS_DIR = os.environ.get("MOSS_DIR", "output/_cv_re")
# 未污染的 CAM 分支（apply_diar 重建，见模块 docstring）；用 CAM_DIR 覆盖
CAM_DIR = os.environ.get("CAM_DIR", str(TMP / "cam_clean2n"))


def read_records(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_mapping(path: str | Path) -> dict[str, dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: list[dict]) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def per_session(d: str, ref: Path = REF, tag: str = "x") -> dict[str, tuple[float, float]]:
    """逐 session 的 (错误数, ref 长度)。单次 meeteval 调用，非逐段调用。"""
    recs: list[dict] = []
    for p in sorted(Path(d).glob("[0-9]*.seglst.json")):
        recs += read_records(p)
    if not recs:
        raise SystemExit(f"目录为空：{d}")
    hp, rp = TMP / f"_rn_{tag}_h.json", TMP / f"_rn_{tag}_r.json"
    write_json(hp, recs)
    sids = {r["session_id"] for r in recs}
    write_json(rp, [r for r in read_records(ref) if r["session_id"] in sids])
    cmd = [str(MEET), "tcpwer", "-r", str(rp), "-h", str(hp), "--collar", "5"]
    sp.run(cmd, capture_output=True, check=True, shell=False)  # noqa: S603
    raw = read_mapping(TMP / f"_rn_{tag}_h_tcpwer_per_reco.json")
    return {k: (float(v["errors"]), float(v["length"])) for k, v in raw.items()}


def n_speakers(records: list[dict]) -> int:
    return len({r["speaker"] for r in records})


def turn_rate(records: list[dict]) -> float:
    """与 `pick_ensemble.turn_rate` 逐字一致：切换次数 / 首末跨度（次/秒）。"""
    if not records:
        return 0.0
    span = max(r["end_time"] for r in records) - min(r["start_time"] for r in records)
    if span <= 0:
        return 0.0
    ordered = sorted(records, key=lambda r: r["start_time"])
    turns = sum(1 for a, b in zip(ordered, ordered[1:], strict=False) if a["speaker"] != b["speaker"])
    return turns / span


def turn_per_seg(records: list[dict]) -> float:
    """切换次数 / (段数 - 1) —— 与 `turn_rate` 同一痕迹的**尺度无关**归一化。

    `turn_rate` 用时长归一，把「交替频繁程度」和「分段密度」混在一起：
    一段被切得很碎的 session，即使说话人正常交替，turns/秒 也会偏高；
    反之段少而长的 session 即使交替正常也显得切换率低。
    按段数归一后得到「相邻段对里有多少比例换了人」，只反映交替本身。
    """
    if len(records) < 2:
        return 0.0
    ordered = sorted(records, key=lambda r: r["start_time"])
    turns = sum(1 for a, b in zip(ordered, ordered[1:], strict=False)
                if a["speaker"] != b["speaker"])
    return turns / (len(ordered) - 1)


def base_rule(f: dict, turn_thr: float, moss_max_spk: int = 2,
              cam_min_spk: int = 3, cam_over: int = 6, eq5_min: int = 4,
              feat_key: str = "m_turn") -> bool:
    """出货规则（v018 族）。返回 True = 路由给 CAM。

    `feat_key` 选用哪个切换率归一化做症状②，默认 `m_turn` = 出货口径逐字不变。
    """
    few = f["m_nspk"] <= moss_max_spk
    low = f[feat_key] <= turn_thr
    v011_ok = (few or low) and f["c_nspk"] >= cam_min_spk
    eq5 = f["c_nspk"] == 5 and f["m_nspk"] >= eq5_min
    over = f["c_nspk"] >= cam_over
    return bool((v011_ok or eq5) and not over)


def rate_of(sel: set[str], sids: list[str], moss: dict, cam: dict, length: float) -> float:
    return 100.0 * sum((cam[s][0] if s in sel else moss[s][0]) for s in sids) / length


def folds(sids: list[str], k: int) -> list[list[str]]:
    return [[s for i, s in enumerate(sids) if i % k == j] for j in range(k)]


def main() -> int:
    moss = per_session(MOSS_DIR, tag="m")
    cam = per_session(CAM_DIR, tag="c")
    sids = sorted(set(moss) & set(cam))
    length = sum(moss[s][1] for s in sids)

    feats: dict[str, dict] = {}
    for sid in sids:
        mr = read_records(f"{MOSS_DIR}/{sid}.seglst.json")
        cr = read_records(f"{CAM_DIR}/{sid}.seglst.json")
        feats[sid] = {"m_nspk": n_speakers(mr), "c_nspk": n_speakers(cr),
                      "m_turn": turn_rate(mr), "m_turn_seg": turn_per_seg(mr)}

    ship = {s for s in sids if base_rule(feats[s], 0.05)}
    logger.info("出货规则(turn=0.05) 路由 %d 段 → %.4f%%",
                len(ship), rate_of(ship, sids, moss, cam, length))
    logger.info("全 MOSS %.4f%% | oracle %.4f%%",
                rate_of(set(), sids, moss, cam, length),
                100.0 * sum(min(moss[s][0], cam[s][0]) for s in sids) / length)
    ship_err = sum((cam[s][0] if s in ship else moss[s][0]) for s in sids)

    k_folds = 5
    fs = folds(sids, k_folds)

    # 两种切换率归一化，**同一嵌套协议**下对照：
    #   m_turn     = 切换次数/秒（出货口径，把交替频率与分段密度混在一起）
    #   m_turn_seg = 切换次数/(段数-1)（尺度无关，只反映交替本身）
    variants = [("m_turn", [round(0.01 * i, 2) for i in range(41)]),
                ("m_turn_seg", [round(0.02 * i, 2) for i in range(51)])]
    for feat_key, grid in variants:
        picked: list[float] = []
        held_err = 0.0
        for j in range(k_folds):
            te = fs[j]
            tr = [s for i, f_ in enumerate(fs) if i != j for s in f_]
            tr_len = sum(moss[s][1] for s in tr)
            best_t, best_r = grid[0], float("inf")
            for t in grid:
                sel = {s for s in tr if base_rule(feats[s], t, feat_key=feat_key)}
                r = rate_of(sel, tr, moss, cam, tr_len)
                if r < best_r:
                    best_r, best_t = r, t
            picked.append(best_t)
            sel_te = {s for s in te if base_rule(feats[s], best_t, feat_key=feat_key)}
            held_err += sum((cam[s][0] if s in sel_te else moss[s][0]) for s in te)
        logger.info("[%s] 嵌套各折阈值 %s → 留出 %.4f%% | vs 出货 %+.4f 点",
                    feat_key, picked, 100.0 * held_err / length,
                    100.0 * (held_err - ship_err) / length)

    logger.info("出货规则同口径 %.4f%%", 100.0 * ship_err / length)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
