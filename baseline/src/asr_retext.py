"""按已有段边界重新识别文本：保留说话人与时间轴，只换 words（T1 主攻）。

## 为什么

2026-08-05 误差分解（v011 全 dev，抹掉说话人口径 12.39%）：

    识别错 sub      1122 词   5.77%   ← 最大块
    净漏词 del-ins   771 词   3.96%
    对齐错           258 对   1.33%

且**语音覆盖 98.6%**（v011 3779.8s vs ref 3832.0s）—— 音频基本全被处理了，
不是没听到，是听错了。所以主攻识别准确度，不是补覆盖。

破榜首需 dev 降 2.08 点（16.979% → ≤14.90%），即吃掉约 36% 的 sub。

## 机制：归属与文本可分离

MOSS 的价值在**说话人时间轴**（v007 靠它把线上从 0.19467 打到 0.16593），
文本只是搭售。这里保留前者、只替换后者 —— 按 v011 已有的段边界切音频，
每段单独送 ASR，**天然单变量**，也不需要 ASR 自己出时间戳。

这是 X12 的镜像：X12 试过「用 CAM++ 的说话人覆盖 MOSS 文本」，失败是因为
CAM++ 的说话人更差；反向操作（用更强 ASR 的文本覆盖 MOSS 文本）从没做过，
而被替换的这一方确实是弱项。

⚠️ D4（文本轴）2026-07-27 封死时针对的是 **fun-asr**，MOSS 当时还不存在。
基座换了，那道封条从未重验。

⚠️ X9 教训：paraformer-zh 的 benchmark 好看但实测文本差于 fun-asr。
**选型不看 benchmark，只认全 dev 实测。**

## 验收（只测纯文本，不碰说话人）

抹掉说话人后与当前 **12.39%** 比：≤10.31% 才够 2.08 点。

用法：
    cd baseline
    moss_venv/bin/python src/asr_retext.py --pred-dir output/v011_pick \\
        --wav-dir ../data/extracted/dev/dev/wav --out output/retext_qwen3 \\
        --model Qwen/Qwen3-ASR-1.7B
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 与 moss_sat.py 同一套口径，避免两套分词标准（标点在 tcpWER 不计分）
PUNCT_RE = re.compile(r"""[，。！？、；：""''…,.!?;:"'()\[\]【】]""")
TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[一-鿿]")

SR = 16000
# 段太短时 ASR 容易吐空或幻觉；短于此值保留原文本（dev ref 有 23% 的段 <1s）
MIN_SEG_SEC = 0.20


def to_words(text: str) -> str:
    """ASR 原始文本 → tcpWER 计分粒度的空格分隔 token 串（同 moss_sat.to_words）。"""
    tokens = TOKEN_RE.findall(PUNCT_RE.sub("", text))
    return " ".join(t.lower() if t.isascii() else t for t in tokens).strip()


def load_asr(model_id: str, device: str, batch_size: int):
    """加载 ASR，返回 transcribe_batch(list[np.ndarray], sr) -> list[str]。

    ⚠️ 必须用官方 `qwen_asr` 包，不能用裸 transformers 的 AutoModelForSpeechSeq2Seq：
    该模型的架构是 `Qwen3ASRForConditionalGeneration`（config.model_type=qwen3_asr），
    用错类会静默地「部分加载」—— 实测报 `MISSING: those params were newly initialized
    because missing from the checkpoint`，等于拿半个随机初始化的模型在跑。
    同 v012 的 AutoProcessor 静默降级，属同一类事故。

    ⚠️ 环境：必须用 `qwen_venv`。qwen-asr 依赖 transformers 4.57.6，装进 moss_venv
    会把它从 5.14.1 **降级**，毁掉 v007/MOSS 的复现环境（dry-run 已证实）。
    """
    import torch  # type: ignore[import-not-found]
    from qwen_asr import Qwen3ASRModel  # type: ignore[import-not-found]

    logger.info("加载 %s (device=%s, batch=%d) ...", model_id, device, batch_size)
    model = Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map=device,
        max_inference_batch_size=batch_size,
        max_new_tokens=256,
    )

    def transcribe_batch(clips: list, sr: int) -> list[str]:
        # language 固定中文：本赛是中文对话，省掉每段的语种检测，也避免误判成别的语言
        results = model.transcribe(audio=[(c, sr) for c in clips], language="Chinese")
        return [r.text for r in results]

    return transcribe_batch


def main() -> int:
    ap = argparse.ArgumentParser(description="按已有段边界重识别文本")
    ap.add_argument("--pred-dir", required=True, help="提供段边界与说话人的预测目录")
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--batch-size", type=int, default=8,
                    help="批量推理上限（段平均仅约 2 秒，批量比逐段快得多）")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="切片前后各多取的秒数，只用于识别、不改输出时间戳。"
                         "无 pad 时段边界会切在词中间（实测 ref『没有没有』被识成"
                         "『没没』）；但 pad 过大会引入邻段的词造成 insertion")
    ap.add_argument("--sessions", help="只跑指定 session，逗号分隔")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import numpy as np  # type: ignore[import-not-found]
    import soundfile as sf  # type: ignore[import-not-found]

    pred_dir, wav_dir, out_dir = Path(args.pred_dir), Path(args.wav_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = sorted(p for p in pred_dir.glob("*.seglst.json")
                   if p.name.split(".")[0].isdigit())
    if args.sessions:
        want = {s.strip() for s in args.sessions.split(",") if s.strip()}
        preds = [p for p in preds if p.name.split(".")[0] in want]
    if args.limit:
        preds = preds[: args.limit]
    if not preds:
        logger.error("没有匹配到预测文件：%s", pred_dir)
        return 1

    done = {p.name for p in out_dir.glob("*.seglst.json")}
    todo = [p for p in preds if p.name not in done]
    logger.info("总计 %d session，已完成 %d，待跑 %d", len(preds), len(done), len(todo))
    if not todo:
        return 0

    transcribe_batch = load_asr(args.model, args.device, args.batch_size)

    start = time.time()
    n_seg = n_kept = n_empty = 0
    for i, p in enumerate(todo, 1):
        sid = p.name.split(".")[0]
        recs = json.loads(p.read_text(encoding="utf-8"))
        data, sr = sf.read(str(wav_dir / f"{sid}.wav"), dtype="float32", always_2d=True)
        audio = data.mean(axis=1).astype(np.float32)

        # 输出先原样克隆：speaker/start_time/end_time 一律不动，只覆盖 words
        out_recs = [dict(r) for r in recs]
        clips, idxs = [], []
        pad = int(args.pad * sr)
        for j, r in enumerate(recs):
            n_seg += 1
            a, b = int(r["start_time"] * sr), int(r["end_time"] * sr)
            # pad 只影响送进 ASR 的音频，输出时间戳仍用原始 a/b（见 out_recs 克隆）
            clip = audio[max(0, a - pad):min(len(audio), b + pad)]
            if len(clip) < MIN_SEG_SEC * sr:
                n_kept += 1        # 太短，保留原文本
            else:
                clips.append(clip)
                idxs.append(j)

        for j, text in zip(idxs, transcribe_batch(clips, sr)) if clips else []:
            words = to_words(text)
            if words:
                out_recs[j]["words"] = words
            else:
                n_empty += 1       # ASR 吐空，保留原文本而不是丢词
                n_kept += 1

        (out_dir / f"{sid}.seglst.json").write_text(
            json.dumps(out_recs, ensure_ascii=False, indent=2), encoding="utf-8")
        el = time.time() - start
        logger.info("[%d/%d] %s: %d 段 (%.0fs, 均 %.1fs/session)",
                    i, len(todo), sid, len(recs), el, el / i)

    logger.info("完成 %d session / %d 段 → %s", len(todo), n_seg, out_dir)
    logger.info("保留原文本 %d 段（其中 ASR 吐空 %d）", n_kept, n_empty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
