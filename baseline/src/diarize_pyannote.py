"""pyannote 说话人分离，产出与 CAM++ 同格式的时间轴缓存（D6.2）。

动机：v011 已证明「MOSS 主 + 强 diar 备选」的按段择优在线上有效（+0.214 点）。
CAM++ 是当前备选源；pyannote community-1 是更强的同范式替换（DER 全面优于 3.1）。
另有一层机会：X12 当年结论是「MOSS 自带说话人优于 CAM++，覆盖反而差 9.4 点」，
换更强 diarizer 后该结论**可能反转**，值得复测。

⚠️ 域风险：本赛数据是「多人短时自由对话」的脱敏互联网数据（赛题 §30/§41），
**不是远场会议**。DiariZen(X4) / Sortformer(X5) / cVBx 都死在 domain transfer 上，
所以**先小样本探针再全量**，别直接铺开。

许可：community-1 = CC-BY-4.0，3.1 = MIT，均合规（赛题 §149 需说明来源）。
需 HF token（gated: auto，网页接受条款后自动批准），从 baseline/.env 读 HF_TOKEN。

输出格式与 `diarize.py`（CAM++）一致，供 `apply_diar.py` 直接消费：
    {"session_id": "001", "num_speakers": 2,
     "segments": [{"start": 0.0, "end": 7.02, "speaker": 0}, ...]}

用法：
    cd baseline
    # 探针（先跑 overlap 最密集的 5 段，和 X4/X5 同一批，可横向对比）
    pyannote_venv/bin/python src/diarize_pyannote.py \
        --wav-dir ../data/extracted/dev/dev/wav --out output/diar_pyann_probe \
        --sessions 009,011,024,069,098
    # 全量
    pyannote_venv/bin/python src/diarize_pyannote.py \
        --wav-dir ../data/extracted/dev/dev/wav --out output/diar_pyann_dev
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"

# 赛题真值只有 2-6 人（dev ref 实测 {2:10,3:29,4:38,5:23,6:6}）。
# CAM++ 吃过亏：上游默认 min=1/max=15 会判出十几个人，falarm 17→43。
# 这是任务级先验（不是逐 session 的答案），可迁移 test。
MIN_SPEAKERS = 2
MAX_SPEAKERS = 6


def load_token() -> str | None:
    """从环境或 baseline/.env 读 HF token（.env 已 gitignore，勿入库）。"""
    if tok := os.environ.get("HF_TOKEN"):
        return tok
    env = BASE_DIR / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def build_pipeline(model_id: str, token: str | None):
    import torch
    from pyannote.audio import Pipeline

    logger.info("加载 %s ...", model_id)
    pipeline = Pipeline.from_pretrained(model_id, token=token)
    if pipeline is None:
        # from_pretrained 在 gated 未授权时返回 None 而不抛异常，必须显式检查，
        # 否则会拖到调用时才炸（同 moss_sat.py 的 AutoProcessor 静默降级教训）
        logger.error("pipeline 加载失败：多半是 gated 未授权。"
                     "请在 HF 模型页点 Agree，并确认 token 有 canReadGatedRepos。")
        raise SystemExit(2)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        logger.info("使用 CUDA")
    return pipeline


def read_waveform(path: Path) -> dict:
    """读 wav 为 pyannote 接受的波形字典。

    pyannote.audio 4.x 默认要 `torchcodec` 才能读文件，而它对 torch 版本挑剔、
    macOS 支持不稳。直接传 {'waveform', 'sample_rate'} 可完全绕开该依赖。
    赛题音频统一 16k/16bit/单通道（赛题 §45），这里仍做单声道归并以防万一。
    """
    import numpy as np
    import soundfile as sf
    import torch

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    return {"waveform": torch.from_numpy(mono).unsqueeze(0), "sample_rate": sr}


def to_cache(annotation, session_id: str) -> dict:
    """pyannote Annotation → CAM++ 同构缓存。

    speaker 重编号为 int（按首次出现顺序），与 `diarize.py` 输出对齐；
    `apply_diar.py` 只读 segments 里的 start/end/speaker。
    """
    spk_map: dict[str, int] = {}
    segs = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        if speaker not in spk_map:
            spk_map[speaker] = len(spk_map)
        segs.append({
            "start": round(float(turn.start), 2),
            "end": round(float(turn.end), 2),
            "speaker": spk_map[speaker],
        })
    segs.sort(key=lambda s: s["start"])
    return {"session_id": session_id, "num_speakers": len(spk_map), "segments": segs}


def main() -> int:
    ap = argparse.ArgumentParser(description="pyannote 说话人分离（D6.2）")
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out", required=True, help="时间轴缓存输出目录")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sessions", help="只跑指定 session，逗号分隔")
    ap.add_argument("--min-speakers", type=int, default=MIN_SPEAKERS)
    ap.add_argument("--max-speakers", type=int, default=MAX_SPEAKERS)
    ap.add_argument("--overlapping", action="store_true",
                    help="用含重叠的 speaker_diarization（默认用 exclusive 版，"
                         "与 CAM++ 语义一致、apply_diar 按最大重叠映射不需要重叠段）")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir)
    if not wav_dir.is_dir():
        logger.error("wav 目录不存在：%s", wav_dir)
        return 1
    # 同 moss_sat.py：跳过 macOS AppleDouble 文件（`._xxx.wav`），否则解码崩溃
    wavs = sorted(p for p in wav_dir.glob("*.wav") if not p.name.startswith("._"))
    if args.sessions:
        wanted = {s.strip() for s in args.sessions.split(",") if s.strip()}
        wavs = [w for w in wavs if w.stem in wanted]
        if missing := wanted - {w.stem for w in wavs}:
            logger.error("以下 session 不存在：%s", sorted(missing))
            return 1
    if not wavs:
        logger.error("没有匹配到 wav")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline(args.model, load_token())

    start = time.time()
    for i, wav in enumerate(wavs, 1):
        # 命名必须是 {sid}.diar.json —— apply_diar.py 按这个模式找缓存（与 diarize.py 一致）
        out_path = out_dir / f"{wav.stem}.diar.json"
        if out_path.exists() and not args.overwrite:
            logger.info("[%d/%d] %s: 已存在，跳过", i, len(wavs), wav.stem)
            continue
        out = pipeline(read_waveform(wav),
                       min_speakers=args.min_speakers,
                       max_speakers=args.max_speakers)
        # pyannote 4.x 返回 DiarizeOutput（不是旧版的 Annotation）。
        # exclusive_speaker_diarization 不含重叠段，官方注释写明「adapted to
        # downstream transcription」—— 与 CAM++ 输出语义一致，也正是
        # apply_diar.py 按时间重叠映射标签所需。
        ann = out.speaker_diarization if args.overlapping else out.exclusive_speaker_diarization
        cache = to_cache(ann, wav.stem)
        out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        el = time.time() - start
        logger.info("[%d/%d] %s: %d段 %d人 (%.0fs, 均%.1fs)", i, len(wavs), wav.stem,
                    len(cache["segments"]), cache["num_speakers"], el, el / i)

    logger.info("完成 %d 个 session → %s", len(wavs), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
