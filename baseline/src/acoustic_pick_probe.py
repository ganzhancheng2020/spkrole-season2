"""择优轴方向检验：用独立声纹嵌入判「哪一套说话人标注更可信」。

## 为什么做这个

择优轴（每个 session 选 MOSS 源还是 CAM 源）2026-08-06 被 480 配置穷举判为耗尽。
但那 480 个配置的特征全是 `moss_nspk` / `turn_rate` / `cam_nspk` —— **全部文本与结构派生**。
而 `wiki/insights/arbitration-signal-must-be-acoustic.md` 用 16 次失败 + 1 次成功证明：
判「哪个源更优」时，**看不见音频的信号一律无效**。

所以那次穷举证明的是「瞎信号族内无更优点」，**不是这条轴够不到**。
可达性探针说轴上还有 1.831 点（oracle 14.090% vs v026 的 15.920%），是需求 0.81 点的 2.3 倍。

## 信号构造

给一套预测的**说话人标注**打分：切出每段音频取嵌入，看同一 speaker 的段是否抱团、
不同 speaker 的段是否分开（轮廓系数 silhouette，余弦距离）。

- MOSS 把两个人并成一个 → 该 speaker 名下的嵌入被拉散 → 轮廓系数低
- 标注正确 → 同标签紧、异标签远 → 轮廓系数高

⚠️ **判官必须用 ERes2NetV2，不能用 CAM++**：CAM 源的标签本身就是 CAM++ 嵌入聚类的产物，
拿 CAM++ 当裁判是循环论证（必然判自己对）。二者独立性由 D6.4 佐证（同一管线换嵌入，
本赛表现差 4.52 点，不是同一个东西）。

## 判据（先验方向，不扫阈值）

按上述 insight 的操作规则 2：**方向检验优先于阈值扫描**。
把 session 按「哪个源实际错误更少」分两组，比较信号均值：

- CAM 实际更优的组，`sil_cam - sil_moss` 应**显著为正**
- MOSS 实际更优的组，应**显著为负**

方向对 → 才值得做真实选择器并扫 margin。方向错或无区分 → **当场停，别扫阈值**
（LM 打分那次若先做这一步，可省 22 组阈值的试错）。

⚠️ 已知系统性偏置：两套源的分割粒度与说话人数不同，轮廓系数本身对 k 敏感。
该偏置是**系统性**的，不影响分组均值差的方向判读，但会影响绝对阈值。

用法：
    cd baseline
    diarizen_venv/bin/python src/acoustic_pick_probe.py \\
        output/wa_m0_dev output/hyp_sw_m3_7_0.70 \\
        ../data/extracted/dev/dev/wav /tmp/pick_probe.json

已算过的 session 会跳过（可断点续跑）。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile

import numpy as np  # type: ignore[import-not-found]
import soundfile as sf  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 判官嵌入模型。**刻意不用 campplus** —— CAM 源的标签就是它聚出来的，会循环论证。
JUDGE_EMB_MODEL = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
# 短于此长度的段不取嵌入（声纹对极短片段不稳；本赛 23% 的段短于 1 秒是已知难点）
MIN_CLIP_SEC = 0.4


def build_embedder():
    from modelscope.pipelines import pipeline  # type: ignore[import-not-found]
    from modelscope.utils.constant import Tasks  # type: ignore[import-not-found]

    logger.info("加载判官嵌入模型: %s", JUDGE_EMB_MODEL)
    return pipeline(task=Tasks.speaker_verification, model=JUDGE_EMB_MODEL)


def embed_segments(sv, wav_path: str, recs: list[dict]) -> tuple[np.ndarray, list[int]]:
    """返回 (L2 归一化的嵌入矩阵, 对应的记录下标)。太短的段被跳过。"""
    x, sr = sf.read(wav_path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    embs, keep = [], []
    for i, r in enumerate(recs):
        a, b = int(r["start_time"] * sr), int(r["end_time"] * sr)
        clip = x[max(0, a) : min(len(x), b)]
        if len(clip) < int(MIN_CLIP_SEC * sr):
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav") as fh:
            sf.write(fh.name, clip, sr)
            out = sv([fh.name], output_emb=True)
        emb = np.asarray(out["embs"][0], dtype=np.float64)
        embs.append(emb / (np.linalg.norm(emb) + 1e-9))
        keep.append(i)
    return (np.vstack(embs) if embs else np.zeros((0, 1))), keep


def silhouette(embs: np.ndarray, labels: list[str]) -> float:
    """按标签算轮廓系数均值（余弦距离）。标签少于 2 类时无定义，返回 0。"""
    uniq = sorted(set(labels))
    if len(uniq) < 2 or len(embs) < 3:
        return 0.0
    dist = 1.0 - embs @ embs.T
    lab = np.array(labels)
    scores = []
    for i in range(len(embs)):
        same = (lab == lab[i]) & (np.arange(len(embs)) != i)
        if not same.any():
            continue  # 该标签只有这一段，轮廓无定义
        a = dist[i][same].mean()
        b = min(dist[i][lab == u].mean() for u in uniq if u != lab[i])
        denom = max(a, b)
        if denom > 0:
            scores.append((b - a) / denom)
    return float(np.mean(scores)) if scores else 0.0


def score_dir(sv, pred_dir: str, wav_dir: str, sid: str,
              cache_dir: str | None = None, tag: str = "") -> tuple[float, int]:
    with open(f"{pred_dir}/{sid}.seglst.json", encoding="utf-8") as fh:
        recs = json.load(fh)
    recs = sorted(recs, key=lambda r: r["start_time"])
    embs, keep = embed_segments(sv, f"{wav_dir}/{sid}.wav", recs)
    n_spk = len({r["speaker"] for r in recs})
    if len(keep) == 0:
        return 0.0, n_spk
    labels = [recs[i]["speaker"] for i in keep]
    if cache_dir:
        # 取嵌入是本探针的全部成本；缓存后换信号只需秒级重算，别再为每个想法重跑一遍
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(
            f"{cache_dir}/{sid}.{tag}.npz",
            embs=embs,
            labels=np.array(labels),
            starts=np.array([recs[i]["start_time"] for i in keep]),
            ends=np.array([recs[i]["end_time"] for i in keep]),
        )
    return silhouette(embs, labels), n_spk


def main() -> int:
    if len(sys.argv) < 5:
        logger.error("用法见模块 docstring：%s", __doc__)
        return 2
    moss_dir, cam_dir, wav_dir, out_path = sys.argv[1:5]
    cache_dir = sys.argv[5] if len(sys.argv) > 5 else None
    sids = sorted(
        f.split(".")[0]
        for f in os.listdir(moss_dir)
        if f.endswith(".seglst.json") and f[0].isdigit()
    )
    if not sids:
        logger.error("MOSS 预测目录为空：%s", moss_dir)
        return 1

    res: dict[str, dict] = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            res = json.load(fh)
        logger.info("续跑：已有 %d 段", len(res))

    sv = build_embedder()
    for k, sid in enumerate(sids, 1):
        if sid in res or not os.path.exists(f"{cam_dir}/{sid}.seglst.json"):
            continue
        sm, nm = score_dir(sv, moss_dir, wav_dir, sid, cache_dir, "moss")
        sc, nc = score_dir(sv, cam_dir, wav_dir, sid, cache_dir, "cam")
        res[sid] = {"sil_moss": sm, "sil_cam": sc, "n_moss": nm, "n_cam": nc}
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        logger.info(
            "[%d/%d] %s  MOSS %.4f (%d人) / CAM %.4f (%d人)  Δ=%+.4f",
            k, len(sids), sid, sm, nm, sc, nc, sc - sm,
        )
    logger.info("PICK PROBE DONE %d 段 → %s", len(res), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
