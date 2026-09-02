"""MOSS-Transcribe-Diarize 端到端 SAT 推理（v007 主线产物的唯一生成路径）。

端到端出「谁在何时说了什么」，不走 fun-asr + 本地 diar 两段式：模型直接输出
`[start][Sxx]text[end]`，parse_transcript 解析后转 SegLST。这是 v007 相对 v002
（CAM++ 重打标签）的架构切换点。

模型 `OpenMOSS-Team/MOSS-Transcribe-Diarize`（0.9B）Apache-2.0，合规。

⚠️ 默认 `--max-new-tokens 2048` 是 **v007 的原始配置**，改了就复现不出 dev 18.39%。
   GPU 探针（logs/2026-07-30_gpu.md）用的是 4096，属另一配置。

用法（注意用 moss_venv，不是 .venv）：
    cd baseline
    # dev 全量（复现 v007 dev=18.39%）
    moss_venv/bin/python src/moss_sat.py \
        --wav-dir ../data/extracted/dev/dev/wav --out output/v007_moss
    # test 全量（提交物来源，务必用独立 out 目录，别覆盖 dev）
    moss_venv/bin/python src/moss_sat.py \
        --wav-dir ../data/extracted/test/test/wav --out output_test_moss
    # v008 已证伪的 prompt 探针（复验用）
    moss_venv/bin/python src/moss_sat.py --prompt-preset fine \
        --wav-dir ../data/extracted/dev/dev/wav --out output/v008_moss_fine \
        --sessions 016,047,030,091,001 --max-new-tokens 4096
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SAT_MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

# 标点在 tcpWER 里不计分，转 SegLST 前一律剥掉（ref 也是无标点逐字）
PUNCT_RE = re.compile(r"""[，。！？、；：“”‘’…,.!?;:"'()\[\]【】]""")
# 逐字切分：英文/数字保留成词，中文按单字（与 ref 的分词粒度对齐）
TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[一-鿿]")

# v008 探针用的 prompt（均已证伪，见 logs/2026-07-29_v008.md，保留供复验）
PROMPT_PRESETS: dict[str, str] = {
    # 细切段：段数几乎不变，49.94% vs 49.81%，无效
    "fine": (
        "请将音频逐句转写为文本。每一句独立成段，每段以起始时间戳和说话人编号"
        "（[S01]、[S02]等）开头，正文为该句语音内容，段末标注结束时间戳。"
        "务必每句话单独一段，不要把多句合并成一段，保持段数尽可能多。"
    ),
    # 强调不漏判说话人：6 段里 5 段 0 输出（破坏 parse 格式），危险
    "speaker": (
        "请将音频转写为文本，每段以起始时间戳和说话人编号（[S01]、[S02]、[S03]等）开头，"
        "正文为语音内容，段末标注结束时间戳。注意：仔细区分每个说话人，音频中通常有"
        "2-6个说话人，务必识别所有说话人，不要遗漏任何说话人，即使说话时间很短也要单独标注。"
    ),
    # 热词提示：默认 prompt 追加热词，未见收益
    "hotword": (
        "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，"
        "正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
        "热词提示：嗯, 唉, 哟, 哦, 雾霾, 戴口罩, 黄磊"
    ),
}


def to_words(text: str) -> str:
    """MOSS 段文本 → tcpWER 计分粒度的空格分隔 token 串。"""
    tokens = TOKEN_RE.findall(PUNCT_RE.sub("", text))
    return " ".join(t.lower() if t.isascii() else t for t in tokens).strip()


def to_seglst(segs, session_id: str) -> list[dict]:
    """MOSS 段序列 → SegLST 记录。

    MOSS 的 speaker 标签（S01/S02…）按**首次出现顺序**重编号成 spk1/spk2…，
    tcpWER 做说话人最优匹配，编号本身无意义但需在 session 内自洽。
    """
    spk_map: dict[str, int] = {}
    records = []
    for seg in segs:
        if not seg.text.strip():
            continue
        if seg.speaker not in spk_map:
            spk_map[seg.speaker] = len(spk_map) + 1
        records.append(
            {
                "session_id": session_id,
                "speaker": f"spk{spk_map[seg.speaker]}",
                "start_time": round(float(seg.start), 2),
                "end_time": round(float(seg.end), 2),
                "words": to_words(seg.text),
            }
        )
    return records


def load_model(model_id: str, processor_id: str | None = None, dtype_name: str = "auto"):
    """加载 MOSS 模型与 processor。CUDA 走 bfloat16，CPU 走 float32。

    `processor_id` 用于微调 checkpoint：HF Trainer 存的 checkpoint 里 processor 相关文件
    不全，`AutoProcessor` 会**静默退化成 Qwen2Tokenizer**（不报错，直到推理中途取
    `processor.feature_extractor` 才炸）。此时把 processor 指向原始基座快照。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    from moss_transcribe_diarize.inference_utils import resolve_device

    device = resolve_device("auto")
    # ⚠️ 精度会显著影响分数：同模型同数据实测 bf16 比 fp32 差 0.877 点（holdout 25 段）。
    # 跨版本比分数必须统一精度，否则会把精度损失误算成模型效果（v012 就栽在这）。
    if dtype_name == "fp32":
        dtype = torch.float32
    elif dtype_name == "bf16":
        dtype = torch.bfloat16
    else:  # auto：CUDA 走 bf16（快），CPU 走 fp32
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    logger.info("加载模型 %s (device=%s dtype=%s) ...", model_id, device, dtype)
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype="auto", local_files_only=True)
        .to(dtype=dtype)
        .to(device)
        .eval()
    )
    src = processor_id or model_id
    logger.info("加载 processor %s ...", src)
    processor = AutoProcessor.from_pretrained(src, trust_remote_code=True, local_files_only=True)
    # 必须核实类型：退化成 tokenizer 时不抛异常，跑到推理中途才失败
    if not hasattr(processor, "feature_extractor"):
        logger.error("processor 加载错误（得到 %s，缺 feature_extractor）。"
                     "评测 checkpoint 请用 --processor 指向基座快照目录。",
                     type(processor).__name__)
        raise SystemExit(2)
    return model, processor, device, dtype


def select_wavs(args: argparse.Namespace) -> list[Path] | None:
    """按 --sessions / --limit 挑出要跑的 wav。出错返回 None。"""
    wav_dir = Path(args.wav_dir)
    if not wav_dir.is_dir():
        logger.error("wav 目录不存在：%s", wav_dir)
        return None

    # 跳过 macOS AppleDouble 元数据文件（`._xxx.wav`）——从 Mac 打 tar 传到 Linux 时会混进来，
    # 它不是音频，解码直接抛 NoBackendError 并中断整轮。用 --sessions 过滤时不会暴露，
    # 跑全集时才炸。
    wavs = sorted(p for p in wav_dir.glob("*.wav") if not p.name.startswith("._"))
    if args.sessions:
        wanted = {s.strip() for s in args.sessions.split(",") if s.strip()}
        wavs = [w for w in wavs if w.stem in wanted]
        missing = wanted - {w.stem for w in wavs}
        if missing:
            logger.error("以下 session 在 %s 中不存在：%s", wav_dir, sorted(missing))
            return None
    if args.limit:
        wavs = wavs[: args.limit]
    if not wavs:
        logger.error("没有匹配到任何 wav：%s", wav_dir)
        return None
    return wavs


def transcribe_dir(args: argparse.Namespace) -> int:
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        generate_transcription,
    )

    wavs = select_wavs(args)
    if wavs is None:
        return 1

    prompt = PROMPT_PRESETS[args.prompt_preset] if args.prompt_preset else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor, device, dtype = load_model(args.model, args.processor, args.dtype)

    start = time.time()
    for i, wav in enumerate(wavs, 1):
        session_id = wav.stem
        out_path = out_dir / f"{session_id}.seglst.json"
        if out_path.exists() and not args.overwrite:
            logger.info("[%d/%d] %s: 已存在，跳过", i, len(wavs), session_id)
            continue

        messages = (
            build_transcription_messages(str(wav), prompt=prompt)
            if prompt
            else build_transcription_messages(str(wav))
        )
        result = generate_transcription(
            model,
            processor,
            messages,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            device=device,
            dtype=dtype,
        )
        records = to_seglst(parse_transcript(result["text"]), session_id)
        if not records:
            # prompt 改坏时 MOSS 会输出 parse 不出来的格式（v008 speaker preset 踩过）
            logger.warning("[%d/%d] %s: 0 段，原始输出前 200 字符：%s",
                           i, len(wavs), session_id, result["text"][:200])
        # encoding 必须显式指定：GPU 服务器默认 ascii，会 UnicodeEncodeError 写出 0 字节文件
        out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

        elapsed = time.time() - start
        logger.info("[%d/%d] %s: %d段 %d人 (%.0fs, 均%.1fs)", i, len(wavs), session_id,
                    len(records), len({r["speaker"] for r in records}), elapsed, elapsed / i)

    logger.info("完成 %d 个 session → %s", len(wavs), out_dir)
    return 0


def main() -> int:
    # GPU 服务器默认 ascii 编码，不 reconfigure 则中文日志直接 UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="MOSS-Transcribe-Diarize 端到端 SAT 推理")
    parser.add_argument("--wav-dir", required=True, help="输入 wav 目录")
    parser.add_argument("--out", required=True, help="SegLST 输出目录（dev/test 务必分开）")
    parser.add_argument("--model", default=SAT_MODEL, help=f"模型 id（默认 {SAT_MODEL}）")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16"],
                        help="推理精度。auto=CUDA用bf16/CPU用fp32。⚠️ bf16 比 fp32 差约 0.88 点，"
                             "跨版本比分数时必须统一（v012 线上倒退的主因之一）")
    parser.add_argument("--processor", default=None,
                        help="processor 来源（默认同 --model）。评测微调 checkpoint 时"
                             "必须指向基座快照，否则 AutoProcessor 会静默退化成 tokenizer")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="默认 2048 = v007 原始配置，改了复现不出 dev 18.39%%")
    parser.add_argument("--prompt-preset", choices=sorted(PROMPT_PRESETS),
                        help="v008 已证伪的 prompt 探针，仅供复验；不传 = MOSS 默认 prompt")
    parser.add_argument("--sessions", help="只跑指定 session，逗号分隔（如 016,047,001）")
    parser.add_argument("--limit", type=int, help="只跑前 N 个 wav")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出（默认断点续跑）")
    return transcribe_dir(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
