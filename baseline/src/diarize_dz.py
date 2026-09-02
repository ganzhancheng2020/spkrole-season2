"""阶段1（DiariZen 版）：WavLM + EEND 本地说话人分离，逐 session 落盘缓存。

与 `diarize.py`（FunASR CAM++ 谱聚类）产出**完全相同的 schema**，因此阶段2
`apply_diar.py` 可以原样复用，两条路线只换分离结果目录即可对比。

为什么值得换：CAM++ 是「切片 → 声纹 → 聚类」范式，每帧只能归一个说话人，短插话与
overlap 会被整段吞掉（实测 dev 001 的 13.88-14.34 短插话被完全漏掉）。
DiariZen 是 EEND powerset 范式，原生支持重叠说话；dev 有 83/106 段含跨说话人 overlap。

⚠️ 必须用 dz_venv（torch 2.2.2）——`diarizen_venv` 的 torch 2.13 与 vendored
pyannote-audio 不兼容（`torchaudio.AudioMetaData` 已被移除）。

用法：
    cd baseline
    PYTHONPATH=src dz_venv/bin/python src/diarize_dz.py \
        --wav-dir ../data/extracted/dev/dev/wav --out output/diar_dz_dev
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DZ_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md"


def build_pipeline(
    repo_id: str,
    min_spk: int | None,
    max_spk: int | None,
    ahc_threshold: float | None,
    segmentation_step: float | None,
    fa: float | None = None,
    fb: float | None = None,
):
    """加载 DiariZen pipeline，并按需覆盖聚类/推理参数。

    CAM++ 的经验：说话人数上下界与聚类阈值对 tcpWER 影响极大（上游默认 1-15 人
    在本赛 2-6 人音频上会判出十几个人）。DiariZen 默认同样是 min=1/max=20，故一并暴露。

    Fa/Fb 是 VBx 聚类核心参数（arxiv 2510.19572 的 cVBx 即调这两个）：
    Fa 影响充分统计量缩放，Fb 控制最终说话人数（越大说话人越少）。
    默认 Fa=0.07/Fb=0.8。cVBx 验证 = 扫不同 Fa/Fb 组合。
    """
    from diarizen.pipelines.inference import DiariZenPipeline

    sd_pipeline = DiariZenPipeline.from_pretrained(repo_id)

    if min_spk is not None:
        sd_pipeline.min_speakers = min_spk
    if max_spk is not None:
        sd_pipeline.max_speakers = max_spk
    clustering_overrides: dict = {}
    if ahc_threshold is not None:
        clustering_overrides["ahc_threshold"] = ahc_threshold
    if fa is not None:
        clustering_overrides["Fa"] = fa
    if fb is not None:
        clustering_overrides["Fb"] = fb
    if clustering_overrides:
        params = dict(sd_pipeline.PIPELINE_PARAMS)
        params["clustering"] = {**params["clustering"], **clustering_overrides}
        sd_pipeline.PIPELINE_PARAMS = params
        sd_pipeline.instantiate(params)
    if segmentation_step is not None:
        # 步长越大窗口越少、越快，但边界精度下降
        sd_pipeline.segmentation_step = segmentation_step

    logger.info(
        "DiariZen 就绪: %s  min_spk=%s max_spk=%s ahc_thr=%s Fa=%s Fb=%s seg_step=%s",
        repo_id, sd_pipeline.min_speakers, sd_pipeline.max_speakers,
        ahc_threshold, fa, fb, segmentation_step,
    )
    return sd_pipeline


def to_segments(annotation) -> list[dict]:
    """pyannote Annotation → [{start, end, speaker}]，speaker 归一化为 int。"""
    label_map: dict[str, int] = {}
    segments: list[dict] = []
    for turn, _, label in annotation.itertracks(yield_label=True):
        if turn.end <= turn.start:
            continue
        if label not in label_map:
            label_map[label] = len(label_map)
        segments.append({
            "start": round(float(turn.start), 3),
            "end": round(float(turn.end), 3),
            "speaker": label_map[label],
        })
    segments.sort(key=lambda s: s["start"])
    return segments


def diarize_one(sd_pipeline, wav_path: Path) -> dict:
    """对单个 wav 跑 DiariZen 分离。"""
    annotation = sd_pipeline(str(wav_path))
    segments = to_segments(annotation)
    return {
        "session_id": wav_path.stem,
        "num_speakers": len({s["speaker"] for s in segments}),
        "segments": segments,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DZ_MODEL, help="HF 仓库 id（md / md-v2）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个（0=全部）")
    ap.add_argument("--stride", type=int, default=1,
                    help="每隔 N 个取一个，用于抽样对比（1=不抽样）")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--min-spk", type=int, default=None, help="上游默认 1")
    ap.add_argument("--max-spk", type=int, default=None, help="上游默认 20")
    ap.add_argument("--ahc-threshold", type=float, default=None, help="上游默认 0.6")
    ap.add_argument("--fa", type=float, default=None, help="VBx Fa 参数，上游默认 0.07")
    ap.add_argument("--fb", type=float, default=None, help="VBx Fb 参数，上游默认 0.8（越大说话人越少）")
    ap.add_argument("--seg-step", type=float, default=None,
                    help="分割窗步长比例，上游默认 0.1；调大更快但边界更粗")
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

    sd_pipeline = build_pipeline(
        args.model, args.min_spk, args.max_spk, args.ahc_threshold,
        args.seg_step, args.fa, args.fb,
    )

    t0 = time.time()
    failures: list[str] = []
    for i, wav in enumerate(todo, 1):
        try:
            record = diarize_one(sd_pipeline, wav)
        except Exception as exc:  # 单段失败不拖垮整批
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
