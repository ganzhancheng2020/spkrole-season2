"""用 MOSS 说话人标记的声学似然改写归属（`spk_ac_probe.py` 打分的消费端）。

## 定位

归属轴在干净 holdout-24 上有 **4.19 点** oracle 空间，其中 **1.91 点**只需
「把纯段的标签改对」、无需重新切分。段级说话人 embedding 够不到这块：
即便给**完美档案 + 只用段内窗**，边际命中率也只有 42%（低于 50% 拒绝线），
因为本赛每 2.2 秒换人、23% 的段短于 1 秒，SV embedding 在这个尺度上退化。

本模块换一类信号：**MOSS 自己对说话人标记 `[Sxx]` 的 teacher forcing 似然**。
它同时满足 `arbitration-signal-must-be-acoustic` 的两个必要条件 ——
**音频条件** ✅ 且 **看得见说话人标签** ✅（后者正是 `logs/2026-08-08.md`
给选源轴统一死因时指名的缺口：「能打分的信号看不见说话人标签」）。

## 证据（全部在干净 holdout-24 上，裁判用**基座**模型，从未见过 dev）

| 指标 | 值 | 参照 |
|---|---|---|
| 区分度（错组均值 − 对组均值）| **+3.220** | 文本轴奏效裁判 +1.195；选源轴失败信号 +0.03~+0.08 |
| 边际命中率 @ τ=0.5 | **73%**（27 对 / 10 错）| 判据线 50% |
| 决策点 | 2221（7 个干净模型 × 24 段）| — |

命中率在 **τ∈[0.25, 2.0] 是平台（67~77%）而非刀刃**，且 7 个模型逐个净正。
τ=0.5 取自**单模型 LOSO 的独立选择**（23/24 折），不是上表的 argmax ——
刻意避免在唯一评测集上择优（v046 正是栽在这一步）。

## ⚠️ 裁判必须用基座，不能用出货模型

出货模型（`sim_out_all106`）的仿真素材**包含 holdout-24**，它见过这些音频。
用它当裁判测出的区分度是 +2.116，用基座反而更高（+3.220）——
说明信号不是记忆产物，但**验证必须用基座**，否则数字不可信。

## 规模的诚实说明

τ=0.5 只改约 **1.3% 的段**（wa_ho_sim 上 4/308）。这是**高精度低召回**的取法：
1.91 点空间里只捕获约 5%。**预期收益小**，不要按 oracle 上界外推
（见 [[capture-rate-scales-with-headroom]]）。

用法：
    python src/spk_ac_relabel.py --pred-dir <dir> --scores <scores.json> --out <dir> [--tau 0.5]
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TAU = float(os.environ.get("AC_TAU", "0.5"))


def main() -> int:
    ap = argparse.ArgumentParser(description="按 MOSS 说话人标记声学似然改写归属")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--scores", required=True, help="spk_ac_probe.py 的输出")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=TAU,
                    help="替代者需超出当前者该幅度才改写（默认 0.5，来自 LOSO）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sc = {o["session"]: {round(s["start_time"], 3): s for s in o["segs"]}
          for o in json.load(open(args.scores, encoding="utf-8"))}

    n_ch = n_tot = 0
    miss = 0
    for p in sorted(glob.glob(f"{args.pred_dir}/[0-9]*.seglst.json")):
        sid = os.path.basename(p).split(".")[0]
        recs = sorted(json.load(open(p, encoding="utf-8")), key=lambda r: r["start_time"])
        # 说话人按首次出现顺序编号，与 build_spk_probe_jsonl.py 完全一致
        order: dict[str, str] = {}
        for r in recs:
            if r["speaker"] not in order:
                order[r["speaker"]] = str(len(order) + 1)
        inv = {v: k for k, v in order.items()}

        out = []
        for r in recs:
            n_tot += 1
            r2 = dict(r)
            s = sc.get(sid, {}).get(round(r["start_time"], 3))
            if s is None:
                miss += 1
            elif len(order) > 1:
                cur = order[r["speaker"]]
                lp = s["logp"]
                alt = max((d for d in order.values() if d != cur), key=lambda d: lp[d])
                if lp[alt] - lp[cur] > args.tau:
                    r2["speaker"] = inv[alt]
                    n_ch += 1
            out.append(r2)
        with open(f"{args.out}/{sid}.seglst.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)

    logger.info("SPK_AC_RELABEL τ=%.2f 改写 %d / %d 段（%.2f%%），无打分 %d 段 -> %s",
                args.tau, n_ch, n_tot, 100 * n_ch / max(1, n_tot), miss, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
