"""路由信号：说话人变化点一致性（判归属质量，偏置与轮廓系数相反）。

## 为什么要再造一个选源信号

选源轴上已死三个机制（2026-08-08，见 logs/2026-08-08.md §2-3）：

1. 480 条手工规则 —— 特征全是文本/结构派生，被
   `wiki/insights/arbitration-signal-must-be-acoustic.md` 判为盲信号；
2. **轮廓系数**（`acoustic_pick_probe.py`，ERes2NetV2）—— 方向门过了
   （区分度 +0.0825）但逐段命中仅 58.5%，按错误数全线告负。
   **机制性死因：轮廓系数衡量簇内干不干净，不衡量对不对。把两个人并成一个
   反而让它升高（少而紧的簇天然得分高）—— 而这正是 MOSS 的失败模式。**
3. 塌缩检测（speaker 名下嵌入双峰）—— 方向直接反，当场停。

另有 `session_ac_score.py`（teacher forcing 声学似然）**自带盲区**：
只条件于音频与文本，**看不见说话人标签**。

2026-08-25 实测（`oracle_speaker.py` 分解）：MOSS/CAM 的逐段错误差里
**归属占 66%、文本占 34%**；锁 oracle 归属后两支几乎打平
（MOSS 12.388% / CAM 12.562%，MOSS 文本反而略好）。
即 MOSS 落后的那 1 点**全部是归属成本**（4.422 vs 3.234）。
→ 选源信号必须判**归属**；文本类信号上限只有 34%。

## 本信号的构造与偏置

不看簇紧不紧，看**该源声称的说话人变化点在声学上是否真的存在**：

- `prec`：在该源每个切换时刻，取前后各 `WIN` 秒嵌入算余弦距离。距离大 = 切换是真的。
  **过分割**会 claim 大量假切换 → prec 低。
- `rec`：在候选时刻里取对比度 top-K（声学上最像换人的地方），
  看该源有没有在附近声称切换。**塌缩**会漏掉真切换 → rec 低。

**关键：两种失败模式被反向惩罚，塌缩由 rec 抓住 —— 偏置与轮廓系数相反。**

⚠️ 判官必须用 ERes2NetV2，**不能用 CAM++**：CAM 分支的分割本身就是 CAM++ 变化点的
产物，拿 CAM++ 当裁判必然判 CAM 全对（循环论证）。

## 两支同跑（性能）

候选时刻取**两支段边界 + 两支切换点的并集**，不用稠密网格：
真实换人只会发生在至少一支提出的边界上，稠密网格纯属浪费。
两支在同一次里算，嵌入**全部共享**；再按 `BATCH` 批量前向。
初版逐窗单条调用 ~1 s/窗、2.5 min/段（106 段要 4+ 小时），故必须批。
"""
from __future__ import annotations

import glob
import json
import logging
import os
import sys

import numpy as np  # type: ignore[import-not-found]
import soundfile as sf  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JUDGE = os.environ.get("JUDGE_EMB", "iic/speech_eres2netv2_sv_zh-cn_16k-common")
WIN = float(os.environ.get("CP_WIN", "1.0"))         # 对比窗长（秒）
TOL = float(os.environ.get("CP_TOL", "0.75"))        # 声称切换与声学峰的容差（秒）
TOPK_RATE = float(os.environ.get("CP_TOPK", "1.0"))  # top-K = 声称切换数 × 此系数
DEDUP = float(os.environ.get("CP_DEDUP", "0.25"))    # 候选去重间隔（秒）
BATCH = int(os.environ.get("CP_BATCH", "64"))
SR = 16000
MIN_LEN = int(0.35 * SR)


def _emb_fn():
    from modelscope.pipelines import pipeline  # type: ignore[import-not-found]
    from modelscope.utils.constant import Tasks  # type: ignore[import-not-found]
    pl = pipeline(task=Tasks.speaker_verification, model=JUDGE)

    def embed(clips: list[np.ndarray]) -> np.ndarray:
        """批量取嵌入并 L2 归一化。"""
        out = []
        for i in range(0, len(clips), BATCH):
            chunk = clips[i:i + BATCH]
            e = np.asarray(pl(chunk, output_emb=True)["embs"], dtype=np.float32)
            out.append(e)
        if not out:
            return np.zeros((0, 192), dtype=np.float32)
        e = np.concatenate(out, axis=0)
        n = np.linalg.norm(e, axis=1, keepdims=True)
        return e / np.maximum(n, 1e-8)
    return embed


def switches(recs: list[dict]) -> list[float]:
    """该源声称的说话人切换时刻（相邻段说话人不同 → 取中点）。"""
    rs = sorted(recs, key=lambda r: r["start_time"])
    return [0.5 * (rs[i]["end_time"] + rs[i + 1]["start_time"])
            for i in range(len(rs) - 1)
            if rs[i]["speaker"] != rs[i + 1]["speaker"]]


def boundaries(recs: list[dict]) -> list[float]:
    """该源的所有相邻段边界中点（不论是否换人）。"""
    rs = sorted(recs, key=lambda r: r["start_time"])
    return [0.5 * (rs[i]["end_time"] + rs[i + 1]["start_time"])
            for i in range(len(rs) - 1)]


def dedup(ts: list[float], gap: float) -> list[float]:
    out: list[float] = []
    for t in sorted(ts):
        if not out or t - out[-1] >= gap:
            out.append(t)
    return out


def main() -> None:
    moss_dir, cam_dir, wav_dir, out_p = sys.argv[1:5]
    res = json.load(open(out_p, encoding="utf-8")) if os.path.exists(out_p) else {}
    embed = _emb_fn()

    for p in sorted(glob.glob(os.path.join(moss_dir, "[0-9]*.seglst.json"))):
        sid = os.path.basename(p).split(".")[0]
        if sid in res:
            continue
        cam_p = os.path.join(cam_dir, f"{sid}.seglst.json")
        wp = os.path.join(wav_dir, f"{sid}.wav")
        if not os.path.exists(cam_p) or not os.path.exists(wp):
            logger.warning("缺 CAM 预测或音频，跳过 %s", sid)
            continue

        rm = json.load(open(p, encoding="utf-8"))
        rc = json.load(open(cam_p, encoding="utf-8"))
        wav, sr = sf.read(wp, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            raise SystemExit(f"❌ {sid} 采样率 {sr} != {SR}，嵌入判官口径不符")
        dur = len(wav) / SR

        sw_m, sw_c = switches(rm), switches(rc)
        # 候选 = 两支边界 + 两支切换点的并集（真换人只会落在某一支提出的边界上）
        cand = dedup([t for t in boundaries(rm) + boundaries(rc) + sw_m + sw_c
                      if WIN <= t <= dur - WIN], DEDUP)
        need = dedup(cand + [t for t in sw_m + sw_c if WIN <= t <= dur - WIN], 1e-6)
        if not need:
            res[sid] = {"moss": {"prec": 0.0, "rec": 0.0, "n_chg": len(sw_m)},
                        "cam": {"prec": 0.0, "rec": 0.0, "n_chg": len(sw_c)},
                        "dur": dur}
            continue

        clips, idx = [], {}
        for t in need:
            a0, b1 = int((t - WIN) * SR), int((t + WIN) * SR)
            mid = int(t * SR)
            if a0 < 0 or b1 > len(wav) or mid - a0 < MIN_LEN or b1 - mid < MIN_LEN:
                continue
            idx[t] = len(clips)
            clips += [wav[a0:mid], wav[mid:b1]]
        if not clips:
            continue
        E = embed(clips)
        con = {t: float(1.0 - float(np.dot(E[i], E[i + 1])))
               for t, i in idx.items()}

        peaks_all = sorted(((c, t) for t, c in con.items()
                            if any(abs(t - x) < 1e-6 for x in cand)), reverse=True)
        out = {}
        for tag, sw in (("moss", sw_m), ("cam", sw_c)):
            pv = [con[t] for t in sw if t in con]
            k = max(1, int(round(len(sw) * TOPK_RATE)))
            peaks = [t for _, t in peaks_all[:k]]
            hit = sum(1 for t in peaks if any(abs(t - s) <= TOL for s in sw))
            out[tag] = {"prec": float(np.mean(pv)) if pv else 0.0,
                        "rec": hit / len(peaks) if peaks else 0.0,
                        "n_chg": len(sw)}
        out["dur"] = dur
        out["n_cand"] = len(cand)
        res[sid] = out
        json.dump(res, open(out_p, "w", encoding="utf-8"), ensure_ascii=False)
        logger.info("%s clips=%d moss(p=%.3f r=%.2f c=%d) cam(p=%.3f r=%.2f c=%d)",
                    sid, len(clips), out["moss"]["prec"], out["moss"]["rec"],
                    out["moss"]["n_chg"], out["cam"]["prec"], out["cam"]["rec"],
                    out["cam"]["n_chg"])

    logger.info("完成 %d session -> %s", len(res), out_p)


if __name__ == "__main__":
    main()
