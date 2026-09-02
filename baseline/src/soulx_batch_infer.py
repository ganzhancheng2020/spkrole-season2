"""SoulX-Transcriber 批量推理（在 GPU 机上跑，不在本机）。

为什么要这个脚本：官方 `inference/infer.py` 是**单文件**入口，每跑一个 wav 就
`init_model()` 一次 —— 30B MoE 加载一次约 1-2 分钟。106 段 dev 逐个调用，光模型
加载就要两小时 GPU 计费。这里把「加载一次 / 循环推理」拆开。

放在 SoulX-Transcriber/inference/ 下运行（需 import 同目录的 infer.py）：
    cd /root/workspace/SoulX-Transcriber
    /root/workspace/vllm_omni/bin/python inference/soulx_batch_infer.py \
        --model /root/workspace/soulx \
        --wav-dir /root/workspace/data/wav \
        --out /root/workspace/out_dev.jsonl \
        --stage-configs-path data/config/soulx_transcriber.yaml

输出 JSONL，字段沿用官方 infer.py 的 schema：
    {"index": "001", "hyp": "[00:00:01.234 --> 00:00:03.456] Speaker 1: 示例文本"}

原始文本**不在这里解析** —— 解析与 SegLST 转换留在本机，保证 GPU 机上只做
不可复算的推理（按小时计费），可复算的部分随时能在本地重跑。

断点续跑：已在 out 里出现的 index 会跳过（GPU 计费中断不能从头再来）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from argparse import Namespace
from pathlib import Path

# infer.py / utils.py 与本文件同目录（SoulX-Transcriber/inference/）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from infer import init_model, sdr_system  # noqa: E402
from utils import detect_and_fix_hallucination_repetition  # noqa: E402
from vllm.multimodal.media.audio import load_audio  # noqa: E402

# 官方给了两套 prompt：infer.py 用「有标点」，infer_with_retry.py 用「无标点」。
# 本赛 ref 是**无标点逐字**，且 tcpWER 不计标点 → nopunct 更贴合口径。
# 另注意 infer.py 的 punct 版末尾多一个字面 "<audio>"，而 prompt 模板里已有真正的
# <|audio_start|><|audio_pad|><|audio_end|> —— retry 版没有这个多余标签，此处照其原样。
_RULES = """
    Task: Speaker Diarization and ASR.
    Rules:
    1. Identify each speaker and their spoken content {punct_clause}.
    2. Format each turn as: [start_time --> end_time] Speaker X: text
    3. Timestamps should be precise to the millisecond (e.g., 00:00:01.234).
    4. {rule4}
    5. Handle overlapping speech by showing concurrent turns with their own time ranges.
    6. Output only the formatted results. No preamble, no explanation.{extra}
"""
# 官方原始的第 4 条（要求「不要切分、保持轮次完整」）—— 实测这正是段粒度只有
# ref 一半（1015 vs 1988）的直接原因：本赛是轮次密集的短对话，不是长会议。
_RULE4_KEEP = ("Do NOT split an utterance to avoid overlap — "
               "keep each speaker turn complete.")
_RULE4_SPLIT = ("Start a new turn at every speaker change. "
                "Keep turns short; never merge consecutive turns from different speakers.")
# 说话人数是**赛题写明的任务级先验**（2-6 人，赛题 §30/§41），不是 dev 拟合出来的
# 答案，可迁移 test。直接针对实测的塌缩（009: 5 人判成 2 人，全量 missed_spk=100）。
_EXTRA_NSPK = ("\n    7. The audio contains 2 to 6 distinct speakers. "
               "Identify every one of them; never merge two different speakers "
               "into the same Speaker label.")
# 强化版：nspk 实测有效（31.44→30.45，missed_spk 100→87）但幅度不够，且 6 人段仍为 0。
# 这里追加两条针对**具体失败机制**的提示：① 短插话（ref 里大量「嗯/对」一两字回应）
# 最容易被并进邻近说话人；② 明示人数偏多，抵消模型往 2-3 人塌缩的倾向。
_EXTRA_NSPK2 = ("\n    7. This is a multi-party conversation with 2 to 6 distinct "
                "speakers, most often 4 or more. Carefully distinguish every "
                "speaker, including those who speak only briefly (even a single "
                "short word such as a backchannel). Never merge two different "
                "speakers into the same Speaker label, and never omit a speaker "
                "just because their total speaking time is short.")


def _mk(punct: bool, split: bool = False, nspk: bool = False,
        nspk2: bool = False) -> str:
    """按三个自变量组装 prompt。规则 2（格式定义）任何变体都不动 —— X11 的教训是
    连格式指令一起改会让输出直接 parse 不出来。"""
    s = _RULES.format(
        punct_clause="with punctuation" if punct else "without punctuation",
        rule4=_RULE4_SPLIT if split else _RULE4_KEEP,
        extra=_EXTRA_NSPK2 if nspk2 else (_EXTRA_NSPK if nspk else ""),
    )
    return s + "<audio>\n" if punct else s


# 增量实验一律建在 **punct** 基线上：实测 punct 31.44% 优于 nopunct 35.30%，
# 在劣基线上做增量会把两个变量搅在一起。
PROMPT_VARIANTS = {
    "punct": _mk(punct=True),                              # infer.py 原样（v015，31.44%）
    "nopunct": _mk(punct=False),                           # 官方 retry 版（v015c，35.30%）
    "split": _mk(punct=True, split=True),                  # 只动第 4 条：鼓励细切
    "nspk": _mk(punct=True, nspk=True),                    # 只加第 7 条：提示 2-6 人
    "split_nspk": _mk(punct=True, split=True, nspk=True),  # 两者合并
    "nspk2": _mk(punct=True, nspk2=True),                  # 强化版说话人提示
}

# 官方 validate_format 的判据：合法行占比 <80% 即视为格式失败并重试
FORMAT_RE = re.compile(
    r"\[(\d{2}:\d{2}\.\d{2,3})\s*-->\s*(\d{2}:\d{2}\.\d{2,3})\]\s*(Speaker \d+):\s*(.*)")


def _parse_time(s: str) -> float:
    """MM:SS.ss → 秒（同官方 utils.parse_time）。"""
    try:
        m, sec = s.split(":")
        return float(m) * 60 + float(sec)
    except ValueError:
        return 0.0


def validate_format(hyp: str) -> bool:
    """官方同款格式校验：无匹配行 或 合法率 <80% 判失败。"""
    total = ok = 0
    for line in hyp.strip().split("\n"):
        m = FORMAT_RE.match(line.strip())
        if not m:
            continue
        total += 1
        if _parse_time(m.group(2)) > _parse_time(m.group(1)):
            ok += 1
    return total > 0 and ok / total >= 0.8


def build_query(audio_path: str, user_query: str, sampling_rate: int) -> dict:
    """同 infer.get_audio_query，但 USER_QUERY 可切换（这是本实验的自变量）。"""
    prompt = (
        f"<|im_start|>system\n{sdr_system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        f"{user_query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    signal, sr = load_audio(audio_path, sr=sampling_rate)
    return {"prompt": prompt,
            "multi_modal_data": {"audio": (signal.astype(np.float32), sr)}}


def load_done(out_path: Path) -> set[str]:
    """读已完成的 index，用于断点续跑。"""
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["index"])
        except (json.JSONDecodeError, KeyError):
            continue  # 半行/坏行忽略，重跑那一段即可
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description="SoulX-Transcriber 批量推理")
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out", required=True, help="输出 JSONL")
    ap.add_argument("--stage-configs-path", default=None)
    ap.add_argument("--sessions", help="只跑指定 session，逗号分隔")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个（调试）")
    ap.add_argument("--sampling-rate", type=int, default=16000)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--num-prompts", type=int, default=1)
    ap.add_argument("--py-generator", action="store_true", default=False)
    ap.add_argument("--log-stats", action="store_true", default=False)
    ap.add_argument("--stage-init-timeout", type=int, default=6000)
    ap.add_argument("--init-timeout", type=int, default=6000)
    ap.add_argument("--prompt-variant", choices=sorted(PROMPT_VARIANTS),
                    default="punct",
                    help="punct=infer.py 原样（v015 用的）；nopunct=官方 retry 版，"
                         "更贴合本赛无标点 ref")
    ap.add_argument("--max-retries", type=int, default=1,
                    help=">1 时启用官方式重试：格式校验失败则升温重试")
    ap.add_argument("--fix-hallucination", action="store_true",
                    help="启用官方重复幻觉修复（v015 实测检出 0/106，默认关）")
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir)
    # 同 moss_sat.py：跳过 macOS AppleDouble（`._xxx.wav`），否则解码崩溃
    wavs = sorted(p for p in wav_dir.glob("*.wav") if not p.name.startswith("._"))
    if args.sessions:
        want = {s.strip() for s in args.sessions.split(",") if s.strip()}
        wavs = [w for w in wavs if w.stem in want]
    if args.limit:
        wavs = wavs[: args.limit]
    if not wavs:
        print(f"没有匹配到 wav: {wav_dir}", flush=True)
        return 1

    out_path = Path(args.out)
    done = load_done(out_path)
    todo = [w for w in wavs if w.stem not in done]
    print(f"总计 {len(wavs)} 段，已完成 {len(done)}，待跑 {len(todo)}", flush=True)
    if not todo:
        return 0

    # ── 只加载一次模型（这就是本脚本存在的理由）──
    t0 = time.time()
    omni, sampling_params_list = init_model(Namespace(**vars(args)))
    print(f"模型加载完成 {time.time() - t0:.0f}s", flush=True)

    user_query = PROMPT_VARIANTS[args.prompt_variant]
    print(f"prompt-variant={args.prompt_variant} max-retries={args.max_retries}",
          flush=True)

    start = time.time()
    n_retry = n_bad = n_hal = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i, wav in enumerate(todo, 1):
            t = time.time()
            text, ok = "", False
            base_temp = args.temperature
            for attempt in range(args.max_retries):
                # 官方重试策略：失败则升温再试（温度写回 SamplingParams）
                if attempt:
                    sampling_params_list[0].temperature = min(
                        base_temp + 0.2 * attempt, 1.0)
                    n_retry += 1
                query = build_query(str(wav), user_query, args.sampling_rate)
                for stage_outputs in omni.generate([query], sampling_params_list):
                    if stage_outputs.final_output_type == "text":
                        text = stage_outputs.request_output.outputs[0].text
                if validate_format(text):
                    ok = True
                    break
            sampling_params_list[0].temperature = base_temp  # 复位，勿污染下一段
            if not ok:
                n_bad += 1
            if args.fix_hallucination and text:
                res = detect_and_fix_hallucination_repetition(text)
                if res["has_hallucination"]:
                    n_hal += 1
                    text = res["repaired_text"]
            # 逐条 flush：GPU 按小时计费，中断时已跑的段必须留下
            f.write(json.dumps({"index": wav.stem, "hyp": text},
                               ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - start
            print(f"[{i}/{len(todo)}] {wav.stem}: {len(text)} 字符"
                  f"{'' if ok else ' [格式未通过]'} "
                  f"({time.time() - t:.1f}s, 累计 {el:.0f}s, 均 {el / i:.1f}s/段)",
                  flush=True)
    print(f"格式失败 {n_bad} 段 / 重试 {n_retry} 次 / 幻觉修复 {n_hal} 段", flush=True)

    omni.close()
    print(f"完成 {len(todo)} 段 -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
