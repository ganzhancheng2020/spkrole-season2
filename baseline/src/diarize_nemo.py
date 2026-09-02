"""阶段1（NeMo Sortformer v2 版）：4-spk EEND + Arrival-Order Speaker Cache 本地分离。

与 `diarize_dz.py`（DiariZen）/ `diarize.py`（CAM++）产出**完全相同的 schema**，
因此阶段2 `apply_diar.py` 可以原样复用，三条路线只换分离结果目录即可对比。

为什么值得试：CAM++ 是聚类范式，结构性处理不了 overlap（dev 83/106 段含 overlap）。
Sortformer v2 是 EEND powerset 范式，原生支持重叠说话；deep-research 调出
AliMeeting（中文会议 19% overlap）DER 7.0%，是中文 overlap 场景最强开源方案。
模型 CC BY 4.0 可商用，FSQ#7 合规。

⚠️ 必须用 nemo_venv（torch 2.13.0 + nemo-toolkit 2.7.3）。dz_venv 的 torch 2.2.2
装不了 nemo 2.7.3（要求 torch≥2.6.0）。nemo_venv 已 patch nv_one_logger/NeptuneLogger
两处版本不匹配，详见 wiki/insights/。

用法：
    cd baseline
    PYTHONPATH=src nemo_venv/bin/python src/diarize_nemo.py \
        --wav-dir ../data/extracted/dev/dev/wav --out output/diar_nemo_dev
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NEMO_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--diar_streaming_sortformer_4spk-v2/"
    "snapshots/6dbf0d69730bfee097056692b86525a0a23b32f9/"
    "diar_streaming_sortformer_4spk-v2.nemo"
)

_SEG_RE = re.compile(r"^([\d.]+)\s+([\d.]+)\s+speaker_(\d+)$")


def parse_segments(seg_strs: list[str]) -> list[dict]:
    """['0.800 6.880 speaker_0', ...] → [{start, end, speaker:int}]。"""
    out: list[dict] = []
    for s in seg_strs:
        m = _SEG_RE.match(s.strip())
        if not m:
            logger.warning("无法解析段: %r", s)
            continue
        start, end, spk = float(m.group(1)), float(m.group(2)), int(m.group(3))
        if end <= start:
            continue
        out.append({"start": round(start, 3), "end": round(end, 3), "speaker": spk})
    out.sort(key=lambda s: s["start"])
    return out


def diarize_one(model, wav_path: Path) -> dict:
    """对单个 wav 跑 Sortformer 分离。"""
    out = model.diarize(audio=str(wav_path), verbose=False)
    seg_strs = out[0] if out and out[0] else []
    segments = parse_segments(seg_strs)
    return {
        "session_id": wav_path.stem,
        "num_speakers": len({s["speaker"] for s in segments}),
        "segments": segments,
    }


def load_model(model_path: str):
    """加载 Sortformer v2 模型（CPU 推理）。"""
    import warnings
    warnings.filterwarnings("ignore")
    from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
    m = SortformerEncLabelModel.restore_from(model_path)
    m = m.to("cpu")
    m.eval()
    logger.info("Sortformer 加载完成: %s", model_path)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=NEMO_MODEL_PATH, help=".nemo 模型路径")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个（0=全部）")
    ap.add_argument("--stride", type=int, default=1,
                    help="每隔 N 个取一个，用于抽样对比（1=不抽样）")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    wav_dir, out_dir = Path(args.wav_dir).resolve(), Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(wav_dir.glob("*.wav"))[:: args.stride]
    if args.limit:
        wavs = wavs[: args.limit]
    if not wavs:
        logger.error("在 %s 下没找到 wav", wav_dir)
        raise SystemExit(2)

    todo = [w for w in wavs if args.overwrite or not (out_dir / f"{w.stem}.diar.json").exists()]
    logger.info("共 %d 个 wav，待跑 %d 个（已缓存 %d）", len(wavs), len(todo), len(wavs) - len(todo))
    if not todo:
        return

    model = load_model(args.model)

    t0 = time.time()
    failures: list[str] = []
    for i, wav in enumerate(todo, 1):
        try:
            record = diarize_one(model, wav)
        except Exception as exc:
            logger.error("[%d/%d] %s 失败: %s", i, len(todo), wav.stem, exc)
            failures.append(wav.stem)
            continue
        (out_dir / f"{wav.stem}.diar.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        elapsed = time.time() - t0
        logger.info(
            "[%d/%d] %s -> %d 段 / %d 人 (累计 %.0fs, 均 %.1fs/段)",
            i, len(todo), wav.stem, len(record["segments"]),
            record["num_speakers"], elapsed, elapsed / i,
        )

    if failures:
        logger.warning("失败 %d 段: %s", len(failures), failures)


if __name__ == "__main__":
    main()
