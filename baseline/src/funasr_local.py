"""Step 7 的本地推理版：用开源 Fun-ASR 权重在本机 GPU 上出带说话人的转写。

与 `run.py` 的关系
------------------
`run.py` 走的是托管实例的异步接口（上传 → 提交任务 → 轮询 → 下载 JSON），
本脚本改为**把开源权重下到本机直接推理**，产物格式与 `run.py` 完全一致（SegLST）。
两者可互相替换，用哪个取决于算力与部署方式。

模型（均为 ModelScope 公开可下载）
----------------------------------
    ASR    默认 iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
    VAD    fsmn-vad
    PUNC   ct-punc（spk_model 依赖它切句，缺了每场直接产出 0 条）
    说话人  cam++（iic/speech_campplus_sv_zh-cn_16k-common）

⚠️ 两条已实测的结论（RTX 3090 / funasr 1.4.3）：

1. **产物不与托管实例逐字节相同。** 公开权重与托管实例上跑的并非同一份模型，
   输出必然有差异。本脚本证明的是「这条分支可以完全用开源权重在自有算力上跑通」，
   **不是「复现出货文件」**。
2. **`FunAudioLLM/Fun-ASR-Nano-2512` 与 funasr 1.4.3 的说话人分离后处理不兼容。**
   2.0 G 权重可正常下载，VAD / PUNC / cam++ 三个模型也都能加载，但推理时
   `auto_model.py` 拿不到 `raw_text`，报 `Missing punc_model, which is required by
   spk_model`，最终每场产出 0 条。换成 Paraformer 后同一套代码正常产出
   （实测 3 场：18 条 / 2 人、10 条 / 7 人、20 条 / 2 人）。
   故默认 ASR 取 Paraformer；要试 Fun-ASR-Nano 用 `--asr-model` 覆盖。

用法：
    MODELSCOPE_CACHE=/root/funasr_models \\
    python src/funasr_local.py --wav-dir <wav 目录> --out output/test_cam_local
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ASR_MODEL = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
ASR_MODEL_FUNASR_NANO = "FunAudioLLM/Fun-ASR-Nano-2512"   # 见文件头第 2 条：与 1.4.3 的分离后处理不兼容
VAD_MODEL = "fsmn-vad"
PUNC_MODEL = "ct-punc"   # spk_model 依赖它切句，缺了会直接产出 0 条
SPK_MODEL = "cam++"

# 与 seglst_converter._to_words 保持一致：去标点、中文逐字、英文小写。
# 三引号 raw：串里含 ASCII 双引号，单引号 raw 会被截断成两段（后半段非 raw，\. 是无效转义）。
_PUNCT = r"""[，。！？、；：""''…,.!?;:"'()\[\]【】]"""


def to_words(text: str) -> str:
    cleaned = re.sub(_PUNCT, "", text or "")
    tokens = re.findall(r"[A-Za-z]+|[0-9]+|[一-鿿]", cleaned)
    return " ".join(t.lower() if t.isascii() else t for t in tokens).strip()


def speaker_label(spk, mapping: dict) -> str:
    """说话人编号 → spk1/spk2/…（按首次出现顺序）。"""
    key = spk if spk is not None else -1
    if key not in mapping:
        mapping[key] = f"spk{len(mapping) + 1}"
    return mapping[key]


def to_seglst(segments: list[dict], session_id: str) -> list[dict]:
    """FunASR 的 sentence_info → SegLST 记录。"""
    mapping: dict = {}
    out: list[dict] = []
    for seg in segments:
        words = to_words(seg.get("text", ""))
        if not words:
            continue
        out.append({
            "session_id": session_id,
            "start_time": round(float(seg.get("start", 0)) / 1000.0, 2),
            "end_time": round(float(seg.get("end", 0)) / 1000.0, 2),
            "speaker": speaker_label(seg.get("spk"), mapping),
            "words": words,
        })
    out.sort(key=lambda r: r["start_time"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out", required=True, help="SegLST 输出目录（每场一个文件）")
    ap.add_argument("--asr-model", default=ASR_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size-s", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个（调试用）")
    args = ap.parse_args()

    from funasr import AutoModel  # 延迟导入：没装 funasr 时也能看 --help

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(Path(args.wav_dir).glob("*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    logger.info("待处理 %d 个 wav → %s", len(wavs), out_dir)

    model = AutoModel(
        model=args.asr_model,
        vad_model=VAD_MODEL,
        punc_model=PUNC_MODEL,
        spk_model=SPK_MODEL,
        device=args.device,
        disable_update=True,
    )

    done = skipped = 0
    for i, wav in enumerate(wavs, 1):
        dst = out_dir / f"{wav.stem}.seglst.json"
        if dst.exists():          # 断点续跑
            skipped += 1
            continue
        try:
            res = model.generate(input=str(wav), batch_size_s=args.batch_size_s)
        except Exception as exc:  # 单场失败不阻断整批
            logger.error("[%d/%d] %s 推理失败：%s", i, len(wavs), wav.stem, exc)
            continue
        segs = (res[0] or {}).get("sentence_info", []) if res else []
        records = to_seglst(segs, wav.stem)
        dst.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        done += 1
        if i % 20 == 0 or i == len(wavs):
            logger.info("[%d/%d] %s → %d 条", i, len(wavs), wav.stem, len(records))

    logger.info("FUNASR_LOCAL DONE 新产出 %d，跳过已存在 %d → %s", done, skipped, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
