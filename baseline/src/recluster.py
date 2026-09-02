"""从缓存的声纹向量重跑聚类——用于秒级扫参。

`diarize.py --emb-cache` 已把 VAD 切片时间 + CAM++ 声纹落盘。
本脚本只重做「聚类 + 后处理」，跳过最贵的 VAD 与声纹提取：
全量 dev 106 段从 ~8 分钟降到 ~10 秒，才谈得上扫 merge_thr / 说话人数上下界。

⚠️ 与 diarize.py 的唯一差异：后处理时**不启用 change_locator**（它需要原始音频来精修
说话人切换点，而缓存里没有音频）。因此扫参选出的参数应再用 diarize.py 全量跑一遍确认。

用法：
    cd baseline
    PYTHONPATH=src diarizen_venv/bin/python src/recluster.py \
        --emb-dir output/emb_dev --out output/diar_sweep_t85 --max-spk 6 --merge-thr 0.85
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIAR_MODEL = "iic/speech_campplus_speaker-diarization_common"
MODEL_REVISION = "master"


def build_pipeline(min_spk: int, max_spk: int, merge_thr: float | None):
    """构建 pipeline 并摘掉 change_locator（缓存里没有音频，用不了它）。"""
    from modelscope.models.audio.sv.cluster_backend import SpectralCluster
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    sd_pipeline = pipeline(
        task=Tasks.speaker_diarization,
        model=DIAR_MODEL,
        model_revision=MODEL_REVISION,
    )
    sd_pipeline.model.spectral_cluster = SpectralCluster(
        min_num_spks=min_spk, max_num_spks=max_spk
    )
    if merge_thr is not None:
        sd_pipeline.model.model_config["merge_thr"] = merge_thr
    sd_pipeline.config.pop("change_locator", None)
    return sd_pipeline


def recluster_one(sd_pipeline, npz_path: Path) -> dict:
    """读缓存 → 聚类 → 后处理 → diar 记录（schema 同 diarize.py）。"""
    from diarize import parse_segments

    data = np.load(npz_path)
    times, embs = data["times"], data["embs"]
    # postprocess 只读 segments 的前两列（起止秒），第三位音频占位即可
    segments = [[float(t[0]), float(t[1]), None] for t in times]

    labels = sd_pipeline.clustering(embs)
    raw = sd_pipeline.postprocess(segments, [], labels, embs)

    segs = parse_segments(raw)
    return {
        "session_id": npz_path.name.replace(".emb.npz", ""),
        "num_speakers": len({s["speaker"] for s in segs}),
        "segments": segs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True, help="diarize.py --emb-cache 产出的目录")
    ap.add_argument("--out", required=True, help="diar 输出目录")
    ap.add_argument("--min-spk", type=int, default=2)
    ap.add_argument("--max-spk", type=int, default=6)
    ap.add_argument("--merge-thr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0, help="k-means 随机种子（上游未固定）")
    args = ap.parse_args()

    from diarize import patch_kmeans_seed

    patch_kmeans_seed(args.seed)

    emb_dir, out_dir = Path(args.emb_dir).resolve(), Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    npzs = sorted(emb_dir.glob("*.emb.npz"))
    if not npzs:
        logger.error("在 %s 下没找到 *.emb.npz", emb_dir)
        raise SystemExit(2)

    sd_pipeline = build_pipeline(args.min_spk, args.max_spk, args.merge_thr)
    logger.info(
        "重聚类 %d 段: min_spk=%d max_spk=%d merge_thr=%s",
        len(npzs), args.min_spk, args.max_spk, args.merge_thr,
    )

    counts: dict[int, int] = {}
    for npz in npzs:
        record = recluster_one(sd_pipeline, npz)
        (out_dir / f"{record['session_id']}.diar.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
        counts[record["num_speakers"]] = counts.get(record["num_speakers"], 0) + 1

    logger.info("完成 -> %s", out_dir)
    logger.info("说话人数分布: %s", sorted(counts.items()))


if __name__ == "__main__":
    main()
