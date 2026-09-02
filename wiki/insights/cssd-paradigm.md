---
slug: cssd-paradigm
type: bound
status: 主线依据
induced_by: [deep-research-2026-08-06b]
related: [[neural-diarizer-domain-wall]] [[ts-asr-model-selection]] [[dev-online-transfer]]
---

# 本赛任务 = CSSD；八次「域墙」中的五次是范式错配（2026-08-06）

> ⚠️ 标题原为「八次…全是范式错配」，当日经自查收窄为「五次」，理由见 §五之二。

> **本页推翻 [neural-diarizer-domain-wall](neural-diarizer-domain-wall.md) 的归因。**
> 那页把 6–8 次神经分离器失败归为「本赛数据有神秘域特性」。本页给出已发表的、
> 可预测的机制解释：**我们的指标属 CDER 族，而帧级范式在该族上有系统性劣势。**
> ⚠️ 该解释覆盖 EEND/TS-VAD 族的失败（4–5 次），**不覆盖自回归语音 LLM 族**（SoulX/Qwen-Omni）。

## 一、本赛数据画像（2026-08-06 实测，非记忆）

| 维度 | dev 实测（106 段） |
|---|---|
| 音频时长 | 均 **41.9s**，中位 42.8s，范围 28.6–45.0s |
| 说话人数 | 2:10 / 3:29 / **4:38** / 5:23 / 6:6（众数 4 人，2 人仅 9.4%）|
| 轮次密度 | **18.8 轮 / 42 秒 ≈ 每 2.2 秒换人** |
| 段长 | 中位 **1.93s**，**23% 短于 1 秒** |
| 重叠 | **仅 5.6%**（低！）|
| 信道 | 16kHz **单声道近场** |

**结论：本赛不是会议转写任务。** 是「40 秒 / 多人 / 极快轮换 / 大量 1 秒内短插话 / 低重叠」的口语对话。
难点**不是重叠分离**（那是 NOTSOFAR/AliMeeting 的难点），**是短插话导致的说话人漏判**。

⚠️ 此前所有选型调研都拿会议 benchmark 当依据（AliMeeting/AISHELL-4 = 8 麦阵列 40 分钟远场、
AMI/NOTSOFAR = 英文远场、MLC-SLM = 固定 2 人且无中文），**没有一个匹配上表**。

## 二、这个任务在文献里有专名：CSSD

**Conversational Short-phrase Speaker Diarization**，ISCSLP 2022 专门办过挑战赛：

- 数据 = **MagicData-RAMC**：中文、手机录制、**16kHz 单声道**、短语、频繁轮换 —— 形态最接近本赛的公开赛事
- 专门造了新指标 **CDER**，理由：「DER 按**时长**计权，**对短语给的权重不够**」
- CDER 在**句子级**计分，长短句权重相等

**本赛 tcpWER 按词计权 → 与 CDER 同族，不是 DER 族。** 一个 0.5 秒的「嗯」和一个 5 秒长句，词权重平等。

## 三、决定性证据（TSUP，CSSD 第三名，arXiv:2210.14653 表 3）

同一份中文短语对话数据上三种范式正面对比：

| 系统 | dev DER | **dev CDER** | test DER | **test CDER** | **盲测 CDER** |
|---|---|---|---|---|---|
| VBx 基线 | **5.57** | 26.90 | **7.96** | 28.20 | — |
| **SC（谱聚类）** | 14.33 | **12.00** | 18.19 | **9.50** | **9.10** ✅ |
| TS-VAD | 12.79 | 9.70 | 17.52 | 10.50 | **16.40** ⚠️ |
| RX-EEND | 12.05 | 16.30 | 12.01 | 19.20 | — |
| SC-EEND | 11.29 | 17.80 | 13.36 | 24.20 | — |

### 三条硬结论

**① DER 与 CDER 反相关。** VBx 的 DER 最好（5.57）而 CDER 最差（26.90）；
SC 的 DER 最差（14.33）而 CDER 最好（12.00）。
→ **按 DER 选模型，在本赛指标上是反向优化。** 我们读过的所有报 DER 的分离 SOTA 论文，优化的都不是我们要的东西。

**② EEND 族在 CDER 上系统性劣于谱聚类**，差近一倍。论文给出机制：
「TS-VAD 和 EEND 的**帧级预测**本质会在短段上产生更多错误，这对 DER 影响很小，
但对平等对待长短句的 CDER 影响巨大」。

**③ TS-VAD 在 dev/test 上最好（9.70），盲测崩到 16.40。** 与本赛域墙的失败形态同型。

## 四、对照本赛八次失败

| 试过 | 范式 | 结果 |
|---|---|---|
| **CAM++**（FSMN-VAD + 声纹 + **谱聚类**）| **SC** | **活，22.79%，一直是备选源** |
| DiariZen（WavLM + EEND powerset）| EEND | 塌 |
| Sortformer | EEND | 塌 |
| pyannote community-1 | EEND+聚类混合 | 塌（46.25%）|
| cVBx | — | 塌 |
| ERes2NetV2 | SC 内换嵌入 | 27.27%，输 CAM++ 4.52 点 |
| SoulX-Transcriber 30B / Qwen-Omni | 帧级 LLM | 塌（30.45%）|

**帧级族（DiariZen/Sortformer/pyannote/cVBx）全在 CSSD 预言会输的那一族，这 4 次失败可预测、不是运气。**
⚠️ 但 SoulX/Qwen-Omni 属自回归语音 LLM，CSSD 未测该类，**它们的失败仍无机制解释**（见 §五之二）。

## 五、行动含义

1. **EEND / TS-VAD 帧级族先验为负**，不再作为主攻。
   ⚠️ **原写「端到端族先验为负」是错的**（见 §五之二）：MOSS 自己就是自回归语音 LLM 且是最强主线，
   Speaker-Reasoner 与它同族，先验应为**中性**而非负。DiCoW 属 TS-ASR/帧级条件，受本条覆盖。
2. **改进应在谱聚类族内做。** CSSD 第二条核心发现给了具体旋钮：
   > **「CDER 随子段长度增大而降低」** —— 同一个人的一句话被切成多个短子段后会被聚成 2+ 说话人。
3. **`diarize.py` 用的是上游默认 `seg_dur=1.5 / seg_shift=0.75`，此前从未调过**
   （只调过 min_spk / max_spk / merge_thr）。而 dev 段长中位数 1.93s，
   正落在「一句被切成两段」的最坏区间。
   → 2026-08-06 已加 `--seg-dur / --seg-shift` 参数并开扫（单变量：其余锁 min=3/max=7/thr=0.70/seed=0）。

## 五之二、⚠️ 自我收窄：CSSD 只解释八次里的五次（2026-08-06 当日补正）

初版本页宣称「八次失败全被 CSSD 解释」，**这是过度外推**。CSSD 表 3 比的是
**SC vs TS-VAD vs EEND** 三类，而我们的失败分两类：

| 失败案例 | 范式 | CSSD 是否覆盖 |
|---|---|---|
| DiariZen / Sortformer / pyannote / cVBx | EEND / TS-VAD 帧级 | ✅ **覆盖**，机制吻合 |
| ERes2NetV2 | SC 内换嵌入 | ⚠️ 部分（同族内失败，另有原因）|
| SoulX-30B / Qwen-Omni | **自回归语音 LLM** | ❌ **未覆盖**，CSSD 没测过这一类 |

**关键推论：MOSS 本身也是自回归语音 LLM**（输出 `[start][Sxx]text[end]` 是段级生成，
不是帧级分类）—— 它不在 CSSD 判定劣势的那一族里，反而是我们的主线且最强。
所以**不能用 CSSD 去否定 Speaker-Reasoner 这类自回归模型**，那是 MOSS 的同族。
本页 §5.1「端到端/帧级族先验为负」应收窄为：**EEND/TS-VAD 帧级族先验为负；
自回归语音 LLM 族未被 CSSD 覆盖，先验中性。**

## 五之三、seg_dur 扫参：假设被证伪，上游默认已是最优

CSSD「子段越长 CDER 越低」在本赛**两个方向都不成立**（单变量，其余锁 min3/max7/thr0.70/seed0）：

| seg_dur | dev tcpWER | missed_spk | falarm_spk |
|---|---|---|---|
| 1.00 | 23.433% | **49** ✅ | 18 ⚠️ |
| 1.25 | 23.546% | 56 | 16 |
| **1.50（上游默认）** | **22.754%** ⭐ | 61 | 11 |
| 2.00 | 24.461% | 55 | 16 |
| 2.50 | 25.197% | 65 | 11 |
| 3.00 | 28.657% | 60 | 14 |

**U 形，最优就在上游默认 1.5** —— 这个旋钮本来就调好了，无收益。

**但留下一个真信号**：短窗把 `missed_spk` 从 61 压到 **49**（说话人漏判正是本赛核心病根），
代价是 `falarm_spk` 从 11 涨到 18。**两者可能通过「短窗 + 更严聚类约束」的二维组合同时拿到**，
该组合从未试过 → 用 `recluster.py`（从缓存嵌入重聚类，秒级）扫。

## 六、诚实的边界

- MagicData-RAMC 是**两人**对话，本赛 3–6 人（众数 4）。**该结论不是完美迁移**，
  只是比会议 benchmark 近得多，不可当铁证。
- CSSD 结论基于 CDER，本赛是 tcpWER。同族但不同量，**方向可信、幅度不可外推**。
- TSUP 是第三名不是第一名；第一名方案未公开。

## 来源

1. [CSSD 任务论文 arXiv:2208.08042](https://doi.org/10.48550/arxiv.2208.08042) — 任务定义 / CDER 指标 / 基线
2. [TSUP CSSD 系统 arXiv:2210.14653](https://arxiv.org/pdf/2210.14653v1) — **表 3 三范式对比 + 子段长度趋势（图 2）**
3. [CSSD 挑战赛官方页](https://www.colips.org/conferences/iscslp2022/web/grand-challenge/)
4. [MagicData-RAMC 数据与基线](https://github.com/MagicHub-io/MagicData-RAMC) — VBx 基线 DER/CDER 对照
5. [CDER 指标实现](https://github.com/SpeechClub/CDER_Metric)
6. [短语自适应切分 + 时长鲁棒声纹](https://sah.borca.ai/papers/272380903) — CSSD 专项改进方法（后续候选）
7. [NVIDIA NeMo issue #7558](https://github.com/NVIDIA/NeMo/issues/7558) — 官方确认「1–2 分钟音频内数准 6–8 人极难，
   系统难以累积说话人画像」，与本赛 42 秒 / 最多 6 人的困难同源
