"""阶段1：本地 CAM++ 说话人分离，逐 session 落盘缓存（可断点续跑）。

只产出「时间轴上谁在说」，不碰文本。文本仍来自 fun-asr（见 output/*.seglst.json），
两者在阶段2（apply_diar.py）按时间重叠合并。

模型 `iic/speech_campplus_speaker-diarization_common` 内部 = FSMN-VAD + CAM++ 声纹 + 谱聚类，
modelscope 托管、开源，与云端 fun-asr 同源，合规。

用法（注意用 diarizen_venv，不是 .venv）：
    cd baseline
    PYTHONPATH=src diarizen_venv/bin/python src/diarize.py \
        --wav-dir ../data/extracted/dev/dev/wav --out output/diar_dev
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DIAR_MODEL = "iic/speech_campplus_speaker-diarization_common"
# 可热替换的声纹嵌入模型（三者接口一致，均输出 192 维）。
# 域墙结论（wiki/insights/neural-diarizer-domain-wall.md）：本赛数据上只有
# 「嵌入+聚类」范式活着，EEND 系全灭 —— 所以升级要在同范式内换更强的嵌入模型。
# 中文 CNCeleb EER：ERes2NetV2(17.8M) 6.14% < ERes2Net-large 6.17% < CAM++(7.2M) 6.78%
EMB_MODELS = {
    "campplus": "iic/speech_campplus_sv_zh-cn_16k-common",      # 管线自带，v002 用的就是它
    "eres2netv2": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
    "eres2net": "iic/speech_eres2net_sv_zh-cn_16k-common",
}
MODEL_REVISION = "master"  # 该模型仓库只有 master 分支，无版本 tag（已查 /revisions 接口）


def patch_kmeans_seed(seed: int) -> None:
    """给谱聚类里的 k-means 固定随机种子。

    上游 `SpectralCluster.cluster_embs` 调的是 `k_means(emb, k)`，**没传 random_state**
    （见 modelscope/models/audio/sv/cluster_backend.py:91），同配置重跑 dev 会漂 ±0.25 点，
    扫参排序因此不可信、SCORES.md 也无法记可复现的数字。这里补上种子。
    """
    from modelscope.models.audio.sv.cluster_backend import SpectralCluster  # type: ignore[import-not-found]
    from sklearn.cluster._kmeans import k_means  # type: ignore[import-not-found]

    def cluster_embs(self, emb, k):  # noqa: ANN001 - 猴补需匹配上游签名
        _, labels, _ = k_means(emb, k, random_state=seed, n_init=10)
        return labels

    SpectralCluster.cluster_embs = cluster_embs
    logger.info("已固定 k-means 随机种子 = %d", seed)


def build_pipeline(min_spk: int, max_spk: int, merge_thr: float | None,
                   emb_model: str = "campplus",
                   seg_dur: float | None = None, seg_shift: float | None = None):
    """构建 modelscope 说话人分离 pipeline（首次调用会下载模型）。

    上游默认 `SpectralCluster(min_num_spks=1, max_num_spks=15)` 是写死的，
    会让谱聚类在本赛 2-6 人的音频上判出十几个说话人。这里按赛题先验收紧上下界。
    注意：这是**任务级先验**（dev 真值分布 2-6 人），不是逐 session 的答案，可迁移 test。

    `seg_dur`/`seg_shift` = **声纹子段的窗长与步长**（上游硬编码 1.5s / 0.75s，
    见 segmentation_clustering_pipeline.py:52-54），2026-08-06 之前从未调过。
    依据 CSSD 挑战赛（ISCSLP 2022，中文短语对话，形态最接近本赛的公开赛事）的核心发现：
    **子段越长 CDER 越低** —— 一句话被切成多个短子段后会被聚成 2 个以上说话人
    （arXiv:2210.14653 表 3 与图 2）。本赛 tcpWER 按词计权、与 CDER 同族
    （短句长句权重相等），故该结论适用；而 dev 段长中位数仅 1.93s，
    正落在「一句被切成两段」的最坏区间。
    """
    from modelscope.models.audio.sv.cluster_backend import SpectralCluster  # type: ignore[import-not-found]
    from modelscope.pipelines import pipeline  # type: ignore[import-not-found]
    from modelscope.utils.constant import Tasks  # type: ignore[import-not-found]

    logger.info("加载 %s (%s) ...", DIAR_MODEL, MODEL_REVISION)
    sd_pipeline = pipeline(
        task=Tasks.speaker_diarization,
        model=DIAR_MODEL,
        model_revision=MODEL_REVISION,
    )

    sd_pipeline.model.spectral_cluster = SpectralCluster(
        min_num_spks=min_spk, max_num_spks=max_spk
    )
    logger.info("聚类说话人数约束: [%d, %d]", min_spk, max_spk)
    if merge_thr is not None:
        sd_pipeline.model.model_config["merge_thr"] = merge_thr
        logger.info("merge_thr 覆盖为 %.3f", merge_thr)

    # sd_pipeline.config 是普通 dict（__init__ 里 self.config.update({...})），可直接改。
    # 必须在 diarize_one 调 sd_pipeline.chunk() 之前设好，否则不生效。
    if seg_dur is not None:
        sd_pipeline.config["seg_dur"] = seg_dur
    if seg_shift is not None:
        sd_pipeline.config["seg_shift"] = seg_shift
    logger.info("声纹子段: seg_dur=%.2fs seg_shift=%.2fs",
                sd_pipeline.config["seg_dur"], sd_pipeline.config["seg_shift"])

    # 热替换声纹嵌入模型。管线内部只以 `sv_pipeline([audio], output_emb=True)` 取嵌入
    # （见 modelscope segmentation_clustering_pipeline.py:95），三个候选签名一致、
    # 均输出 192 维，故可直接替换而不必改管线逻辑。
    if emb_model != "campplus":
        model_id = EMB_MODELS[emb_model]
        logger.info("替换嵌入模型: %s -> %s", EMB_MODELS["campplus"], model_id)
        sd_pipeline.sv_pipeline = pipeline(
            task=Tasks.speaker_verification, model=model_id
        )
    return sd_pipeline


def parse_segments(raw: object) -> list[dict]:
    """pipeline 输出 → [{start, end, speaker}]。

    modelscope 返回 {'text': [[start, end, spk_id], ...]}，spk_id 为 int。
    """
    items = raw.get("text", []) if isinstance(raw, dict) else raw
    if not isinstance(items, list):   # 上游偶有非 list 返回，静默迭代会炸在半路
        items = []
    segments: list[dict] = []
    for item in items:
        if len(item) < 3:
            continue
        start, end, spk = float(item[0]), float(item[1]), int(item[2])
        if end <= start:
            continue
        segments.append({"start": round(start, 3), "end": round(end, 3), "speaker": spk})
    segments.sort(key=lambda s: s["start"])
    return segments


def diarize_one(sd_pipeline, wav_path: Path, emb_cache_dir: Path | None = None) -> dict:
    """对单个 wav 跑分离，返回缓存记录。

    这里把官方 `pipeline.__call__` 逐步展开（VAD → 切片 → 声纹 → 聚类 → 后处理），
    步骤与顺序完全一致，目的只是能把「声纹向量」截留下来：
    重跑聚类只要毫秒，而 VAD + 声纹提取要数秒/段。缓存后即可用 recluster.py 秒级扫参。
    """
    vad_segments = sd_pipeline.preprocess(str(wav_path))
    sd_pipeline.check_audio_list(vad_segments)
    segments = sd_pipeline.chunk(vad_segments)
    embeddings = sd_pipeline.forward(segments)

    if emb_cache_dir is not None:
        import numpy as np  # type: ignore[import-not-found]

        times = np.asarray([[s[0], s[1]] for s in segments], dtype="float32")
        np.savez_compressed(
            emb_cache_dir / f"{wav_path.stem}.emb.npz",
            times=times,
            embs=embeddings.astype("float32"),
        )

    labels = sd_pipeline.clustering(embeddings)
    raw = sd_pipeline.postprocess(segments, vad_segments, labels, embeddings)

    segs = parse_segments(raw)
    speakers = sorted({s["speaker"] for s in segs})
    return {
        "session_id": wav_path.stem,
        "num_speakers": len(speakers),
        "segments": segs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True, help="wav 目录")
    ap.add_argument("--out", required=True, help="缓存输出目录")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个（0=全部，调试用）")
    ap.add_argument("--overwrite", action="store_true", help="忽略已有缓存重跑")
    ap.add_argument("--min-spk", type=int, default=2, help="最少说话人数（上游默认 1）")
    ap.add_argument("--max-spk", type=int, default=6, help="最多说话人数（上游默认 15）")
    ap.add_argument("--merge-thr", type=float, default=None,
                    help="同说话人合并的余弦阈值，越低合并越激进（模型默认 0.78）")
    ap.add_argument("--emb-cache", default=None,
                    help="声纹向量缓存目录，供 recluster.py 秒级重聚类扫参")
    ap.add_argument("--emb-model", default="campplus", choices=sorted(EMB_MODELS),
                    help="声纹嵌入模型。默认 campplus = v002 配置；"
                         "eres2netv2 在中文 CNCeleb 上 EER 更低（6.14% vs 6.78%）")
    ap.add_argument("--seed", type=int, default=0, help="k-means 随机种子（上游未固定）")
    ap.add_argument("--seg-dur", type=float, default=None,
                    help="声纹子段窗长秒（上游 1.5）。CSSD 结论：子段越长 CDER 越低，"
                         "因为一句话被切碎后会被聚成多个说话人。dev 段长中位数仅 1.93s")
    ap.add_argument("--seg-shift", type=float, default=None,
                    help="声纹子段步长秒（上游 0.75，惯例取 seg-dur 的一半）")
    args = ap.parse_args()

    patch_kmeans_seed(args.seed)

    wav_dir = Path(args.wav_dir).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(args.emb_cache).resolve() if args.emb_cache else None
    if emb_dir is not None:
        emb_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(wav_dir.glob("*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    if not wavs:
        logger.error("在 %s 下没找到 wav", wav_dir)
        raise SystemExit(2)

    todo = [w for w in wavs if args.overwrite or not (out_dir / f"{w.stem}.diar.json").exists()]
    logger.info("共 %d 个 wav，待跑 %d 个（已缓存 %d）", len(wavs), len(todo), len(wavs) - len(todo))
    if not todo:
        return

    sd_pipeline = build_pipeline(args.min_spk, args.max_spk, args.merge_thr, args.emb_model,
                                 args.seg_dur, args.seg_shift)

    t0 = time.time()
    failures: list[str] = []
    for i, wav in enumerate(todo, 1):
        cache_path = out_dir / f"{wav.stem}.diar.json"
        try:
            record = diarize_one(sd_pipeline, wav, emb_dir)
        except Exception as exc:  # 单段失败不拖垮整批，记录后继续
            logger.error("[%d/%d] %s 失败: %s", i, len(todo), wav.stem, exc)
            failures.append(wav.stem)
            continue
        cache_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        elapsed = time.time() - t0
        logger.info(
            "[%d/%d] %s -> %d 段 / %d 人 (累计 %.1fs, 均 %.1fs/段)",
            i, len(todo), wav.stem, len(record["segments"]),
            record["num_speakers"], elapsed, elapsed / i,
        )

    if failures:
        logger.warning("失败 %d 段: %s", len(failures), failures)


if __name__ == "__main__":
    main()
