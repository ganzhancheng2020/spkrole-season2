"""仿真对话生成：把 dev 拆成单说话人片段，跨 session 重组成新的多人对话。

## 为什么做这个

微调 MOSS 的病根是「少判说话人」，而 dev 只有 106 段、其中 6 人段仅 6 个 —— 数据量
和难段密度都不够。外部语料全都撞域墙（D6.1/6.2/6.4/6.5 连续六次），或**人数不对**
（RAMC/CSSD/SeniorTalk 只有 2 人），或**许可存疑**（RAMC 是 CC BY-NC-ND）。

dev 自重组同时占四个便宜：声学域完全一致、零合规风险（赛题提供的数据）、零下载成本、
轮次结构可直接照抄真实分布。

## 方法：Simulated Conversations 而非 Simulated Mixtures

arXiv 2204.00890 的结论：早期 EEND 用的「随机拼单人片段」(SM) **不像真实对话，
尤其超过 2 人时**；改成按真实轮次结构生成的 SC 才把 EEND 推到 SOTA。

所以这里不随机拼，而是照抄 dev 实测的结构参数（`--stats` 可重新打印核对）：

    说话人数分布   {2:10, 3:29, 4:38, 5:23, 6:6}
    单条发言时长   中位 1.94s，23% 短于 1s（短插话）
    相邻发言重叠   22.1%，重叠时长中位 0.58s
    静音间隔       中位 0.35s，30% 紧接（<0.1s）
    语音占空比     0.88
    session 时长   中位 41.9s
    说话人切换率   0.235 次/秒

## 关键设计：跨 session 混合说话人

同 session 内重排只是换顺序，说话人组合没变，增益有限。跨 session 抽取说话人才能
造出 dev 里不存在的新组合（410 个说话人实例 → 组合空间远大于原始 106 段）。

代价：不同 session 录音条件不同，拼接处有声学跳变。这会让「区分说话人」在训练时
**比真实情况更容易**，是 SM 的已知缺陷。用 `--same-session-ratio` 混入一部分同 session
重组来缓解（那部分声学连续但组合新）。

⚠️ 合规：**只读 dev**，test 集不得以任何形式参与训练（赛题 §103/§148）。路径硬编码。

用法：
    cd baseline
    moss_venv/bin/python src/simulate_conv.py --stats            # 只打印真实分布
    moss_venv/bin/python src/simulate_conv.py --n 1000 --out output/sim_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
import soundfile as sf  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT = BASE_DIR.parent
# 硬编码 dev：test 不得参与训练（赛题 §103/§148），不开放为参数
DEV_REF = ROOT / "data/extracted/dev/dev/ref.seglst.json"
DEV_WAV = ROOT / "data/extracted/dev/dev/wav"
SR = 16000
# 外部素材片段的 session_id 前缀。见 load_pool(extra_material=...) 与 render()：
# 带该前缀的片段从素材目录取音频，而非从 dev wav 取。
EXT_PREFIX = "EXT::"

TARGET_DUR = 42.0               # 目标时长，dev 中位 41.9s
MIN_DUR, MAX_DUR = 28.0, 45.0   # dev 实测范围 28.3-44.9s
OVERLAP_RATE = 0.221            # 相邻发言对重叠比例（dev 实测）
# 同一人最多连续几条。**这原本是 3，是换人率偏高的主因**：dev 实测最长连说 15 条、
# 均值 1.76，硬截断在 3 会强行制造换人。加入实测续说概率 p 后，连说长度自然服从几何分布
# （均值 1/(1-p)，4 人段 p=0.439 → 1.78，与实测 1.76 吻合），故此处只留一个防跑飞的安全阀。
MAX_SAME_SPK_RUN = 20
MIN_RECORDS = 4                 # 少于这么多条的丢弃（不成对话）
# 剩余时长放不下时，最多再翻几条片段找能放下的；设 1 等价于「放不下就收尾」。
# 实测（train-82 素材，n=150，目标段长中位 1.91s / 换人率 0.232）：
#   FIT=1 → 段长 1.57s、换人率 0.254、<1s 占 32%
#   FIT=8 → 段长 1.63s、换人率 0.232、<1s 占 27%   ← 三项都更优，故取 8
MAX_FIT_TRIES = int(os.environ.get("MAX_FIT_TRIES", "8"))
# 重叠片段的增益区间。实测：按 dev 的 22.1% 重叠率做**等响度波形相加**时，MOSS 的
# del 率从 8.3% 飙到 17.4%（tcpWER 35.52%）；完全去掉重叠则 del 回到 8.9%
# （28.04%）。原因是真实对话的重叠是「插话者离麦更远、音量更低」，不是两路等响
# 度精确叠加 —— 后者直接把 ASR 打懵。这里给重叠插入的片段做衰减。
OVERLAP_GAIN = (0.25, 0.55)
# 真实 dev **全 106 段**的峰值区间（实测 0.725-0.908，中位 0.821）。
# 渲染后缩放到此区间，以免仿真数据留下可识别的响度标记。
DEV_PEAK_RANGE = (0.725, 0.908)


def source_sessions(split: str) -> set[str] | None:
    """按切分挑素材来源 session。`all` 返回 None（不过滤）。

    ⚠️ **这是本脚本最要紧的一个开关。** 用全部 dev 合成、再拿 dev 验证，
    等于把 dev 洗一道再喂给模型 —— 与 v012/v013/v030 及 RAMC 污染版是**同一个泄露机制**。
    2026-08-10 实测：仅 81 段 dev 进训练集，就把 dev 分数伪造出 **7.0 个点**。
    做微调实验时必须用 `train`，把 holdout 留成真·未见过的验证集。
    """
    if split == "all":
        return None
    from prep_finetune_data import (  # type: ignore[import-not-found]  # noqa: PLC0415
        stratified_split,
    )

    ref = json.loads(DEV_REF.read_text(encoding="utf-8"))
    by_sess: dict[str, list[dict]] = defaultdict(list)
    for x in ref:
        by_sess[x["session_id"]].append(x)
    train, holdout = stratified_split(by_sess, 25, 0)
    return set(train) if split == "train" else set(holdout)


def _all_session_ids() -> set[str]:
    """dev ref 里出现的全部 session id。"""
    ref = json.loads(DEV_REF.read_text(encoding="utf-8"))
    return {x["session_id"] for x in ref}


def match_duration_dist(meta: list[dict], ref: list[dict],
                        rng: random.Random, bins: int = 12) -> list[dict]:
    """按 dev 的片段时长分布，对外部素材做拒绝采样。

    ## 为什么必须做（2026-08-17，A 版实测换来的）

    首版只做「并入音色」，没管素材自身的时长分布，结果**画像从另一个口子漏了进来**：

    | | 中位 | P25 | P75 | <1s 占比 |
    |---|---|---|---|---|
    | AISHELL-4 素材 | **3.42s** | 1.92 | 5.17 | **9%** |
    | dev 素材 | **1.94s** | 1.07 | 2.69 | **23%** |

    素材长 76%、短插话只有一半 → 模型跟着学长了段：
    A 版输出段长中位 2.24s（v031 是 2.08s，真值 1.99s），段数 11.0（v031 11.5，真值 19.0）。
    **说话人数判断确实变准了**（3.62→3.79，真值 3.83，正是外部音色该有的收益），
    **但段长/段数反而更差**，两相抵消后 holdout 净退 0.228 点。

    → **「只借音色」要借得干净：时长分布也必须对齐，否则等于把会议的节奏教给模型。**

    做法：按 dev 时长直方图算每个 bin 的目标占比，对外部素材在各 bin 内**无放回抽样**，
    使并入池的外部片段与 dev 同分布。返回筛选后的子集（总量会缩小，这是预期代价）。
    """
    dev_d = [x["end_time"] - x["start_time"] for x in ref]
    lo, hi = min(dev_d), max(dev_d)
    width = (hi - lo) / bins

    def bin_of(d: float) -> int:
        return min(bins - 1, max(0, int((d - lo) / width))) if width > 0 else 0

    dev_hist: Counter = Counter(bin_of(d) for d in dev_d)
    by_bin: dict[int, list[dict]] = defaultdict(list)
    for m in meta:
        d = float(m["duration"])
        if lo <= d <= hi:                       # dev 里不存在的时长直接丢弃
            by_bin[bin_of(d)].append(m)

    # 以「最紧的 bin」定总量：保证每个 bin 都能按 dev 比例取满
    total_dev = sum(dev_hist.values())
    caps = [len(by_bin[b]) / (dev_hist[b] / total_dev)
            for b in dev_hist if dev_hist[b] > 0 and by_bin[b]]
    budget = int(min(caps)) if caps else 0

    out: list[dict] = []
    for b, cnt in dev_hist.items():
        want = int(round(budget * cnt / total_dev))
        avail = by_bin.get(b, [])
        out.extend(rng.sample(avail, min(want, len(avail))))
    logger.info("外部素材按 dev 时长分布重采样：%d → %d 段", len(meta), len(out))
    return out


def load_pool(sessions: set[str] | None = None,
              extra_material: str | None = None
              ) -> dict[tuple[str, str], list[dict]]:
    """dev ref → 按「(session, speaker)」分组的单说话人片段池。

    `sessions` 为 None 时用全部 dev；给定集合时只取这些 session 的说话人作素材。

    `extra_material` 指向 `extract_a4_segments.py` 的产物目录（含 `segments.json` 与 `wav/`），
    把外部语料的单说话人片段**并入音色池**。两个要点：

    - **只并音色，不并画像** —— `real_stats()` 仍然只从 dev 统计
      （静音间隔 / 重叠 / 说话人数分布 / 同人续说概率），
      故外部语料的轮次结构（如会议的慢换人节奏）**不会**被带进训练数据。
      这正是绕开 RAMC 死因（模型学到「只有两个人」的先验，+2.41 点）的关键。
    - 每个外部片段自成一个「session」，`session_id` 形如 `EXT::<wav 文件名>`、`start_time=0`；
      `render()` 见到 `EXT::` 前缀时改从素材目录读音频。
    """
    ref = json.loads(DEV_REF.read_text(encoding="utf-8"))
    pool: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for x in ref:
        if sessions is not None and x["session_id"] not in sessions:
            continue
        pool[(x["session_id"], x["speaker"])].append(x)

    if extra_material:
        meta = json.loads((Path(extra_material) / "segments.json").read_text(encoding="utf-8"))
        meta = match_duration_dist(meta, ref, random.Random(0))  # noqa: S311
        for m in meta:
            sid = f"{EXT_PREFIX}{m['wav']}"
            pool[(sid, m["speaker"])].append({
                "session_id": sid,
                "speaker": m["speaker"],
                "start_time": 0.0,
                "end_time": float(m["duration"]),
                "words": m["text"],
            })
        logger.info("并入外部素材 %s：%d 段 / %d 人",
                    extra_material, len(meta), len({m["speaker"] for m in meta}))

    for v in pool.values():
        v.sort(key=lambda r: r["start_time"])
    return dict(pool)


def real_stats(
    sessions: set[str] | None = None,
) -> tuple[list[float], list[float], dict[int, int], dict[int, float]]:
    """从 dev 提取生成器需要的四组分布：静音间隔、重叠时长、说话人数分布、同人续说概率。

    分布同样只从 `sessions` 统计 —— 否则 holdout 的结构信息会经由分布参数漏进训练数据。
    """
    ref = json.loads(DEV_REF.read_text(encoding="utf-8"))
    if sessions is not None:
        ref = [x for x in ref if x["session_id"] in sessions]
    by = defaultdict(list)
    for x in ref:
        by[x["session_id"]].append(x)
    gaps: list[float] = []
    ovls: list[float] = []
    # 逐「说话人数」统计相邻段同一人续说的概率。真实对话的说话人是「黏」的：
    # 4 人段实测 44.5% 续说，而在 n 人里均匀挑只有 25% —— 均匀挑会让换人率快 25%，
    # 等于教模型「换人比实际更频繁」。RAMC 那轮的致命伤正是说话人先验被教错（方向相反）。
    cont = defaultdict(lambda: [0, 0])   # nspk -> [同人续说数, 相邻对总数]
    for r in by.values():
        r.sort(key=lambda x: x["start_time"])
        n = len({x["speaker"] for x in r})
        for a, b in zip(r, r[1:]):
            d = b["start_time"] - a["end_time"]
            (ovls if d < 0 else gaps).append(abs(d))
            cont[n][1] += 1
            if a["speaker"] == b["speaker"]:
                cont[n][0] += 1
    spk = defaultdict(set)
    for x in ref:
        spk[x["session_id"]].add(x["speaker"])
    same_prob = {n: s / t for n, (s, t) in cont.items() if t > 0}
    return gaps, ovls, dict(Counter(len(v) for v in spk.values())), same_prob


def print_stats() -> None:
    """打印 dev 真实结构参数 —— 生成器的依据，也便于核对本文件注释是否过期。"""
    ref = json.loads(DEV_REF.read_text(encoding="utf-8"))
    gaps, ovls, nspk, same_prob = real_stats()
    durs = sorted(x["end_time"] - x["start_time"] for x in ref)
    logger.info("记录 %d / 说话人实例 %d", len(ref),
                len({(x["session_id"], x["speaker"]) for x in ref}))
    logger.info("发言时长 中位 %.2fs, <1s 占 %.0f%%", st.median(durs),
                100 * sum(1 for d in durs if d < 1) / len(durs))
    logger.info("重叠 %.1f%% (中位 %.2fs) / 静音间隔 中位 %.2fs",
                100 * len(ovls) / (len(ovls) + len(gaps)), st.median(ovls),
                st.median(gaps))
    logger.info("说话人数分布 %s", dict(sorted(nspk.items())))


def pick_speakers(rng: random.Random, pool: dict, n: int,
                  same_session: bool) -> list | None:
    """选 n 个说话人实例。same_session=True 时全部来自同一 session（声学连续）。"""
    by_sess = defaultdict(list)
    for k in pool:
        by_sess[k[0]].append(k)
    if same_session:
        cands = [s for s, v in by_sess.items() if len(v) >= n]
        if not cands:
            return None
        return rng.sample(by_sess[rng.choice(sorted(cands))], n)
    if len(by_sess) < n:
        return None
    # 跨 session：每人来自不同 session，制造 dev 里不存在的新组合
    return [rng.choice(by_sess[s]) for s in rng.sample(sorted(by_sess), n)]


def build_one(rng: random.Random, pool: dict, gaps: list[float],
              ovls: list[float], nspk_dist: dict[int, int],
              same_session: bool,
              same_prob: dict[int, float] | None = None,
              overlap_rate: float = OVERLAP_RATE
              ) -> tuple[list[dict], list[tuple]] | None:
    """生成一段仿真对话的时间轴。

    返回 (records, plan)：records 为 SegLST（`_spk` 为内部键，调用方重编号），
    plan 为 [(源session, 源start, 源end, 落位start)] 供音频渲染。
    """
    ks, ws = zip(*sorted(nspk_dist.items()))
    n = rng.choices(ks, weights=ws, k=1)[0]
    chosen = pick_speakers(rng, pool, n, same_session)
    if chosen is None:
        return None

    queues = {k: rng.sample(pool[k], len(pool[k])) for k in chosen}
    cursors = dict.fromkeys(chosen, 0)

    records: list[dict] = []
    plan: list[tuple] = []
    t = round(rng.uniform(0.0, 0.6), 2)   # 起始留一点静音
    last, run = None, 0
    next_is_overlap = False
    # 按实测的「同人续说概率」挑下一个说话人，而不是在 n 人里均匀挑。
    # dev 实测 4 人段续说 44.5%，均匀挑只有 25% —— 均匀挑会让换人率快约 25%，
    # 等于把「换人比实际更频繁」教给模型。用 None 时退回旧的均匀行为（向后兼容）。
    p_cont = (same_prob or {}).get(n)
    while t < TARGET_DUR:
        if last is not None and p_cont is not None and run < MAX_SAME_SPK_RUN \
                and rng.random() < p_cont:
            spk = last                       # 同一人继续说
        else:
            others = [k for k in chosen if k != last]
            spk = rng.choice(others or chosen)
        run = run + 1 if spk == last else 1
        last = spk

        # 取一条放得下的片段。**原实现是「放不下就 break」，这会系统性丢弃长片段**：
        # 越接近时长上限，长段越容易被砍，采样偏向短段 —— 实测段长中位被压到 1.75s
        # （素材 train-82 是 1.91s），并连带推高 <1s 占比与换人率。
        # 改为跳过放不下的片段继续找，有限次数内找不到才收尾。
        seg = None
        rejected: list[dict] = []
        for _ in range(MAX_FIT_TRIES):
            if cursors[spk] >= len(queues[spk]):   # 片段用尽 → 重新洗牌复用
                queues[spk] = rng.sample(pool[spk], len(pool[spk]))
                cursors[spk] = 0
            cand = queues[spk][cursors[spk]]
            cursors[spk] += 1
            if t + (cand["end_time"] - cand["start_time"]) <= MAX_DUR:
                seg = cand
                break
            rejected.append(cand)
        # 放不下的候选**不算「用过」**，塞回队列尾部延后再试。
        # 原实现让它们被永久消耗：一个段最多烧掉 MAX_FIT_TRIES=8 条素材，
        # 而 dev 每个 (session, speaker) 只有约 5 条 —— 池子瞬间耗尽就触发上面的
        # 「重新洗牌复用」，同一句话在一段里反复出现。
        # 实测该 bug 让**段内重复句占比 34.3%，而真实 dev 只有 0.3%**（单句最高重复 19 次）。
        queues[spk].extend(rejected)
        if seg is None:
            break

        dur = seg["end_time"] - seg["start_time"]
        records.append({"_spk": spk, "start_time": round(t, 2),
                        "end_time": round(t + dur, 2), "words": seg["words"]})
        # 上一条决定了本条是否为「重叠插入」→ 决定增益
        plan.append((seg["session_id"], seg["start_time"], seg["end_time"], t,
                     rng.uniform(*OVERLAP_GAIN) if next_is_overlap else 1.0))

        # 下一条落位：按真实分布决定重叠还是静音
        next_is_overlap = rng.random() < overlap_rate and bool(ovls)
        if next_is_overlap:
            t += dur - min(rng.choice(ovls), dur * 0.8)   # 重叠不吞掉整条
        else:
            t += dur + rng.choice(gaps)

    if len(records) < MIN_RECORDS or records[-1]["end_time"] < MIN_DUR:
        return None
    if len({r["_spk"] for r in records}) < n:   # 有人一句没说 → 丢弃
        return None
    return records, plan


def render(plan: list[tuple], total: float, cache: dict,
           rng: random.Random, extra_material: str | None = None) -> np.ndarray:
    """按 plan 把源片段叠加成一条新波形（重叠处直接相加）。

    `sess` 带 `EXT_PREFIX` 前缀时从 `extra_material/wav/` 取音频（外部语料素材），
    否则从 dev wav 取。
    """
    buf = np.zeros(int((total + 0.5) * SR), dtype=np.float32)
    for sess, s0, e0, at, gain in plan:
        if sess not in cache:
            if sess.startswith(EXT_PREFIX):
                if not extra_material:
                    raise ValueError(f"plan 含外部素材 {sess} 但未提供 --extra-material")
                path = Path(extra_material) / "wav" / sess[len(EXT_PREFIX):]
            else:
                path = DEV_WAV / f"{sess}.wav"
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            if sr != SR:
                raise ValueError(f"{path} 采样率 {sr} != {SR}")
            cache[sess] = data.mean(axis=1).astype(np.float32)
        clip = cache[sess][int(s0 * SR):int(e0 * SR)]
        p = int(at * SR)
        buf[p:p + len(clip)] += gain * clip[:max(0, len(buf) - p)]
    # 峰值必须落回真实 dev 的区间。首版只在 peak>1 时缩放到 1.0，结果 12/20 段峰值
    # 恰好 =1.000，而真实 dev 是 0.767-0.877 —— 等于给仿真数据打了个可识别标记，
    # 模型可能学成「峰值 1.0 = 仿真」这种无关特征。这里一律缩放到实测区间内。
    peak = float(np.abs(buf).max())
    if peak > 0:
        buf *= rng.uniform(*DEV_PEAK_RANGE) / peak
    return buf


def main() -> int:
    ap = argparse.ArgumentParser(description="仿真多人对话生成（只用 dev）")
    ap.add_argument("--n", type=int, default=500, help="生成多少段")
    ap.add_argument("--out", help="输出目录（含 wav/ 与 ref.seglst.json）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--same-session-ratio", type=float, default=0.3,
                    help="同 session 重组的比例（声学连续，缓解跨段拼接痕迹）")
    ap.add_argument("--overlap-rate", type=float, default=OVERLAP_RATE,
                    help="相邻发言重叠概率（默认 0.221 = dev 实测；设 0 可诊断"
                         "「波形相加式重叠是否为文本崩坏主因」）")
    ap.add_argument("--stats", action="store_true", help="只打印 dev 真实分布后退出")
    ap.add_argument("--hard-boost", type=float, default=1.0,
                    help="难例过采样倍率：5 人段权重 ×boost、6 人段 ×boost²。"
                         "默认 1.0 = 照抄 dev 真实分布（行为不变）。")
    ap.add_argument("--extra-material",
                    help="外部语料素材目录（extract_a4_segments.py 的产物，含 segments.json 与 wav/）。"
                         "**只并音色，画像仍全部来自 dev** —— real_stats() 不受影响。")
    ap.add_argument("--exclude-sessions", default=None,
                    help="逗号分隔的 session id，从素材里排除。用于 dev 交叉验证："
                         "每折排除该折的 session，训出的模型对该折就是干净的。"
                         "与 --source-split 互斥（本参数优先）。")
    ap.add_argument("--source-split", choices=["train", "holdout", "all"], default="all",
                    help="素材来源切分。**做微调实验必须用 train**，否则 holdout 也进了"
                         "训练数据，验证集不再干净（实测 81 段泄露可伪造 7.0 点）")
    args = ap.parse_args()

    if args.stats:
        print_stats()
        return 0
    if not args.out:
        logger.error("需要 --out")
        return 1

    if args.exclude_sessions:
        # 交叉验证模式：素材 = 全部 106 段减去本折
        drop = {x.strip() for x in args.exclude_sessions.split(",") if x.strip()}
        allsess = set(_all_session_ids())
        missing = drop - allsess
        if missing:
            logger.error("要排除的 session 不存在：%s", sorted(missing))
            return 1
        src = allsess - drop
        logger.info("交叉验证模式：排除 %d 段，素材 %d 段", len(drop), len(src))
    else:
        src = source_sessions(args.source_split)
        logger.info("素材来源切分 = %s（%s 个 session）", args.source_split,
                    len(src) if src is not None else "全部 106")
    gaps, ovls, nspk_dist, same_prob = real_stats(src)
    if args.hard_boost != 1.0:
        # 难例过采样：MOSS 的病根是**少判说话人**，而 5-6 人段最难
        # （dev 里 6 人段仅 6 个，难例密度天然不足）。按人数给权重乘以 boost^(n-4)，
        # 即 5 人段 ×boost、6 人段 ×boost²，4 人及以下不变。
        # ⚠️ 这会让训练分布**偏离** dev 真实分布 —— 这是刻意的：
        # 训练分布该对齐「哪里最容易错」，而不是「真实世界长什么样」。
        nspk_dist = {k: max(1, int(round(v * (args.hard_boost ** max(0, k - 4)))))
                     for k, v in nspk_dist.items()}
        logger.info("难例过采样 boost=%.1f → 说话人数分布 %s",
                    args.hard_boost, dict(sorted(nspk_dist.items())))
    logger.info("同人续说概率（实测，按说话人数）: %s",
                {k: round(v, 3) for k, v in sorted(same_prob.items())})
    pool = load_pool(src, args.extra_material)
    # 仿真采样，非安全用途；固定 --seed 是为了可复现
    rng = random.Random(args.seed)  # noqa: S311

    out_dir = Path(args.out)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[str, np.ndarray] = {}
    all_records: list[dict] = []
    made = tries = 0
    while made < args.n and tries < args.n * 20:
        tries += 1
        same = rng.random() < args.same_session_ratio
        got = build_one(rng, pool, gaps, ovls, nspk_dist, same,
                        same_prob=same_prob, overlap_rate=args.overlap_rate)
        if got is None:
            continue
        records, plan = got
        sid = f"sim_{made + 1:05d}"
        total = records[-1]["end_time"]
        sf.write(str(wav_dir / f"{sid}.wav"), render(plan, total, cache, rng, args.extra_material),
                 SR, subtype="PCM_16")
        # speaker 按首次出现顺序重编号 spk1/spk2…（与 ref 口径一致）
        spk_map: dict = {}
        for r in records:
            if r["_spk"] not in spk_map:
                spk_map[r["_spk"]] = len(spk_map) + 1
            all_records.append({
                "session_id": sid,
                "speaker": f"spk{spk_map[r['_spk']]}",
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "words": r["words"],
            })
        made += 1
        if made % 100 == 0:
            logger.info("已生成 %d/%d", made, args.n)

    (out_dir / "ref.seglst.json").write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")

    # 自检：生成分布是否贴合 dev（不贴合就说明参数抄错了）
    gen_spk = defaultdict(set)
    gen_dur: dict[str, float] = defaultdict(float)
    for r in all_records:
        gen_spk[r["session_id"]].add(r["speaker"])
        gen_dur[r["session_id"]] = max(gen_dur[r["session_id"]], r["end_time"])
    logger.info("生成 %d 段 / %d 条记录 -> %s", made, len(all_records), out_dir)
    logger.info("说话人数分布 %s  (dev %s)",
                dict(sorted(Counter(len(v) for v in gen_spk.values()).items())),
                dict(sorted(nspk_dist.items())))
    logger.info("时长 中位 %.1fs  (dev 41.9s)", st.median(list(gen_dur.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
