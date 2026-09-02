"""用 FireRedASR2-**LLM**（8.3B）整场转写，与 AED（1.1B）同口径对比。

## 这一步在验什么

`word-arb-capture-headroom` 定位到：词级仲裁还剩 0.43 本地点，
而瓶颈是**裁判在 988 个可判块上准确率仅 64.6%**，且已证明
那是 FireRedASR2-AED 声学模型本身的性质（五种打分/阈值口径全部封死）。
所以只剩「换更强的声学模型」。

**但先验不乐观**：本赛外部模型 0/6 —— SoulX 30B 差 13 点、Qwen 三代全出局，
且 `ts-asr-model-selection` 的硬结论是「任何论文数字对本赛不具预测力」。
**8.3B 比 1.1B 大不构成任何证据。**

所以先做最便宜的一步：**整场转写 106 段，直接和 AED 的 19.42% 比**。
- ≥ 19.42% → 方向当场判死，不再投逐段转写
- 明显更低 → 才值得做逐段转写 + 仲裁

⚠️ LLM 版**不输出词级时间戳**（`asr.py` 的 llm 分支只返回 text/rtf），
所以它无法像 AED 那样按时间戳映射到 MOSS 切分 —— 本脚本只做整场转写用于**选型比较**，
真要用作候选源必须改走逐段转写。

用法（GPU）：
    python fr_llm_transcribe.py <wav_dir> <out_dir> [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIRERED_SRC = os.environ.get("FIRERED_SRC", "/tmp/FireRedASR2S")
LLM_CKPT = os.environ.get("LLM_CKPT", "")


def main() -> int:
    ap = argparse.ArgumentParser(description="FireRedASR2-LLM 整场转写（选型比较用）")
    ap.add_argument("wav_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not LLM_CKPT:
        logger.error("需要 LLM_CKPT 指向 FireRedASR2-LLM 目录")
        return 2
    sys.path.insert(0, FIRERED_SRC)
    from fireredasr2s.fireredasr2.asr import (  # type: ignore[import-not-found]  # noqa: PLC0415
        FireRedAsr2,
        FireRedAsr2Config,
    )

    # 解码参数取自仓库官方示例 examples_infer/asr/inference_asr_llm.sh：
    #   --beam_size 3 --repetition_penalty 3.0 --llm_length_penalty 1.0 --temperature 1.0
    # ⚠️ 默认 llm_length_penalty=0.0 会让 beam search 没有产出长序列的动力 →
    #    立刻选 EOS，每段只出 1 个字符（本脚本初版即因此产出 '%'，差点误判模型失效）
    # 精度：必须走 bf16，两个坑叠在一起
    #  ① use_half=True 时 asr.py 对整个模型调 .half()（fp16），覆盖掉 llm 加载器
    #     特意选的 bfloat16（其注释写明 "Training use torch.bfloat16"）→
    #     Qwen2 数值溢出，每段输出同一个 '%'
    #  ② use_half=False 时编码器留在 fp32(2.7G) + bf16 Qwen2(15.2G) → 24G 卡 OOM
    # 解法：环境变量 FR_BF16=1 让 asr.py 统一转 bfloat16（同 fp32 数值范围，
    # 且编码器减半到 1.4G，总占用约 18G）。该开关默认关闭，不影响 AED/word_arb。
    cfg = FireRedAsr2Config(use_gpu=True, use_half=True, beam_size=3,
                            repetition_penalty=3.0, llm_length_penalty=1.0,
                            temperature=1.0)
    m = FireRedAsr2.from_pretrained("llm", LLM_CKPT, cfg)
    os.makedirs(args.out_dir, exist_ok=True)

    wavs = sorted(f for f in os.listdir(args.wav_dir) if f.endswith(".wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    logger.info("待转写 %d 段 | ckpt=%s", len(wavs), LLM_CKPT)

    t0 = time.time()
    for i, w in enumerate(wavs, 1):
        sid = w[:-4]
        out_p = os.path.join(args.out_dir, f"{sid}.json")
        if os.path.exists(out_p):
            continue
        res = m.transcribe([sid], [os.path.join(args.wav_dir, w)])
        # LLM 分支只返回 text/rtf/wav，没有 timestamp/confidence
        txt = res[0].get("text", "") if res else ""
        with open(out_p, "w", encoding="utf-8") as fh:
            json.dump({"session_id": sid, "text": txt}, fh, ensure_ascii=False)
        if i % 10 == 0 or i == len(wavs):
            logger.info("[%d/%d] %s: %d 字 (%.0fs)", i, len(wavs), sid, len(txt), time.time() - t0)

    logger.info("FR_LLM_DONE %d 段 -> %s", len(wavs), args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
