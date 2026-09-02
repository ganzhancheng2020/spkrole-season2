"""从 AISHELL-4 提取「单说话人片段」，作为仿真管线的音色素材。

## 为什么（2026-08-17，数据轴的第二步）

v040 线上 0.14683 证实了数据轴的关键判据：**说话人多样性 ≠ clip 数量**。

| 版本 | 素材源 | 说话人 | 条数 | 线上Δ |
|---|---|---|---|---|
| v031 | dev train-82 | 318 | 800 | −0.00477 |
| v032/v033 | **同一批 82 session** | **318** | **3000** | **+0.00654** 🔴 |
| **v040** | 全 106 session | **410** | 800 | **−0.00110** ✅ |

**加人有效、加 clip 有害。** 但 dev 的说话人已用尽（410 个），要继续加只能靠外部语料。

## 为什么是 AISHELL-4，以及为什么当素材而不是直接微调

外部语料此前被先验排除（「画像与本赛无一匹配」），**但那个理由只在「直接拿来微调」时成立** ——
RAMC 的死因正是模型学到了它「只有两个人」的先验（+2.41 点）。

**仿真管线的价值是画像由我们控制**：只借音色、不继承其轮次结构，即可绕开该死因。

选 AISHELL-4 的三条理由：

1. **CC BY-SA 4.0，允许衍生**（MagicData-RAMC 是 CC BY-**ND**，禁止衍生 —— 本赛有奖金，是实质合规风险）
2. **每场 6–7 人**，与本赛 3–6 人的上沿吻合（RAMC 只有 2 人）
3. 16kHz 采样率与本赛一致

⚠️ **已知风险**：AISHELL-4 是 **8 通道远场会议**录音，本赛是**近场口语对话**。
当素材用只借音色，影响应小于直接微调，但**必须实测** ——
所幸该实验**可本地验证**（用 dev train-82 + AISHELL-4 作素材，holdout-24 保持干净）。

## 本脚本做什么

对每个 session：8 通道取第 0 通道降为单声道 → 按 RTTM 的说话人时间段切片 →
从 TextGrid 的对应 IntervalTier 取该时间段的文本 → 落盘。

**只保留满足条件的片段**（与 `simulate_conv.py` 的素材要求一致）：
时长在 `[MIN_SEC, MAX_SEC]` 内、文本非空、且**不与其它说话人重叠** ——
会议重叠率高，重叠段会把两个人的声音混进同一个「单说话人」片段，污染音色。

用法（在 GPU 上跑，需 soundfile）：
    python extract_a4_segments.py /root/autodl-tmp/aishell4/train_L /root/autodl-tmp/ft/a4_material
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import defaultdict

import soundfile as sf  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 片段时长范围。本赛 dev 段中位约 2s；取 [0.5, 8] 覆盖短应答到长句。
MIN_SEC = float(os.environ.get("MIN_SEC", "0.5"))
MAX_SEC = float(os.environ.get("MAX_SEC", "8.0"))
# 与其它说话人重叠超过该秒数就丢弃
MAX_OVERLAP = float(os.environ.get("MAX_OVERLAP", "0.05"))
CHANNEL = int(os.environ.get("CHANNEL", "0"))


def parse_rttm(path: str) -> list[tuple[float, float, str]]:
    """RTTM -> [(start, end, speaker)]。"""
    out: list[tuple[float, float, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 8 or p[0] != "SPEAKER":
                continue
            st, dur = float(p[3]), float(p[4])
            out.append((st, st + dur, p[7]))
    return out


def parse_textgrid(path: str) -> dict[str, list[tuple[float, float, str]]]:
    """Praat TextGrid -> {speaker: [(xmin, xmax, text)]}，不依赖第三方库。"""
    with open(path, encoding="utf-8", errors="ignore") as fh:
        txt = fh.read()
    out: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    blocks = re.split(r"\n\s*item\s*\[\d+\]:", txt)[1:]
    for b in blocks:
        m = re.search(r'name\s*=\s*"([^"]*)"', b)
        if not m:
            continue
        spk = m.group(1)
        pat = r'xmin\s*=\s*([\d.]+)\s*\n\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*"([^"]*)"'
        for iv in re.finditer(pat, b):
            a, z, t = float(iv.group(1)), float(iv.group(2)), iv.group(3).strip()
            if t:
                out[spk].append((a, z, t))
    return out


def overlap_sec(seg: tuple[float, float, str], others: list[tuple[float, float, str]]) -> float:
    """该片段与**其它说话人**片段的总重叠时长。"""
    s, e, spk = seg
    tot = 0.0
    for a, z, o in others:
        if o == spk:
            continue
        ov = min(e, z) - max(s, a)
        if ov > 0:
            tot += ov
    return tot


def text_in(tiers: list[tuple[float, float, str]], s: float, e: float) -> str:
    """取落在 [s,e] 内的 TextGrid 文本（重叠过半才算），按时间拼接。"""
    parts = [t for a, z, t in sorted(tiers) if min(e, z) - max(s, a) > 0.5 * (z - a)]
    return "".join(parts)


def main() -> int:
    if len(sys.argv) < 3:
        logger.error("用法见模块 docstring：%s", __doc__)
        return 2
    root, out_dir = sys.argv[1], sys.argv[2]
    wav_out = os.path.join(out_dir, "wav")
    os.makedirs(wav_out, exist_ok=True)

    tg_dir = os.path.join(root, "TextGrid")
    sids = sorted(
        f[:-5] for f in os.listdir(tg_dir)
        if f.endswith(".rttm") and os.path.exists(os.path.join(root, "wav", f[:-5] + ".flac"))
    )
    if not sids:
        logger.error("没有配套的 wav+rttm：%s", root)
        return 1
    logger.info("配套 session %d 个 | MIN_SEC=%.1f MAX_SEC=%.1f MAX_OVERLAP=%.2f",
                len(sids), MIN_SEC, MAX_SEC, MAX_OVERLAP)

    meta: list[dict] = []
    seen: set[str] = set()
    for k, sid in enumerate(sids, 1):
        segs = parse_rttm(os.path.join(tg_dir, sid + ".rttm"))
        tiers = parse_textgrid(os.path.join(tg_dir, sid + ".TextGrid"))
        x, sr = sf.read(os.path.join(root, "wav", sid + ".flac"), dtype="float32")
        if x.ndim > 1:
            x = x[:, CHANNEL]
        kept = 0
        for i, (s, e, spk) in enumerate(segs):
            d = e - s
            if d < MIN_SEC or d > MAX_SEC:
                continue
            if overlap_sec((s, e, spk), segs) > MAX_OVERLAP:
                continue
            tx = text_in(tiers.get(spk, []), s, e)
            if not tx:
                continue
            gid = "%s_%s" % (sid, spk)          # 全局唯一说话人 id
            name = "%s_%05d.wav" % (gid, i)
            sf.write(os.path.join(wav_out, name), x[int(s * sr):int(e * sr)], sr)
            meta.append({"speaker": gid, "wav": name, "duration": round(d, 3), "text": tx})
            seen.add(gid)
            kept += 1
        logger.info("[%d/%d] %s: 保留 %d / %d 段（累计 %d 段 / %d 人）",
                    k, len(sids), sid, kept, len(segs), len(meta), len(seen))

    with open(os.path.join(out_dir, "segments.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    logger.info("A4 EXTRACT DONE %d 段 / %d 个说话人 → %s", len(meta), len(seen), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
