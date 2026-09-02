---
slug: ts-asr-model-selection
type: bound
status: 结论已出，待验证
induced_by: [deep-research-2026-08-06]
related: [[neural-diarizer-domain-wall]] [[ensemble-oracle-bound]] [[dev-online-transfer]] [[diarization-sota-2026]]
---

# TS-ASR 路线调研与开源模型选型（deep-research 2026-08-06，20 源）

> **与 [diarization-sota-2026](diarization-sota-2026.md) 的分工**：那一页从「**替换 diarization**」角度评估，
> 故把 SE-DiCoW 判为 ✗（"不替换 diar"）。本页从「**文本轴 / TS-ASR 范式**」角度重估同一批模型，
> 并补上那页缺的：MLC-SLM 全表、DiCoW 的中文缺口、X13 的外部验证、FireRedASR2、Speaker-Reasoner。

**约束**：仅开源权重，闭源/API 一律排除 ｜ **决策口径**：线上需降 1.317 点 = dev **2.03 点**（架构级 65% 迁移率）

> 🔴 **2026-08-06 当日稍晚：本页第 5 条选型结论已作废。**
> 本页整套候选是按「会议转写 benchmark 排名」筛的，而本赛实测是
> **41.9 秒 / 3–6 人 / 每 2.2 秒换人 / 23% 段短于 1 秒 / 重叠仅 5.6% 的近场口语对话**，
> 与 AliMeeting / AISHELL-4 / AMI / MLC-SLM **无一匹配**。
> 按任务形态重搜后，首选 Speaker-Reasoner、次选 DiCoW 的排序**反了**——
> 二者均属被 CSSD 实测预言劣势的帧级族。见 [cssd-paradigm](cssd-paradigm.md)。
> 本页 §3.1（域墙定量）、§3.2（X13 外部验证）仍然有效且重要。

## 执行摘要

1. **我们的主线 MOSS-Transcribe-Diarize 是最接近的公开赛事的卫冕冠军**（MLC-SLM 2026 Task 1 第一名，
   tcpMER 14.93）。换掉它 = 打赢一个卫冕冠军，门槛比此前认知高得多。
2. **X13 得到外部独立验证**：MLC-SLM 2026 第 6 名（17.97）正是「CAM++ + Qwen3-ASR-1.7B」，
   且做满**全量 SFT + 合成数据 LoRA + GRPO 强化学习**，仍输 MOSS 3.04 点。
3. **域墙被量化**：SoulX 官方报 AliMeeting DER **5.39**，本赛 dev 实测 **30.45%** 且塌说话人。
   **任何论文数字对本赛不具预测力**——选型只能按机制筛，不能按 benchmark 排序。
4. **DiCoW 需下调评级**（推翻本人当日早些时候的初判）：MLC-SLM 训练语种**无中文**；
   且在 GT 切分下未微调的 DiCoW（22.0）**输给普通 Whisper-large-v3（16.8）**。
5. 便宜档唯一机制不同的候选 = **FireRedASR2**；贵档唯一机制对路的候选 = **Speaker-Reasoner**。

## 1. 范式分类

| 范式 | 机制 | 代表 | 本赛经历 |
|---|---|---|---|
| **级联 Cascade** | ASR 与 diar 各自独立，事后按时间对齐 | 3D-Speaker + SLLM | v000/v002，22.79–26.01% |
| **端到端 SATS/SDR** | 单模型直出「谁在何时说了什么」 | MOSS、SoulX、SpeakerLM、TagSpeech、Speaker-Reasoner | v007 MOSS 18.388%（主线）|
| **TS-ASR** | 先 diar 出时间轴，**再把时间轴作为逐帧条件喂给 ASR**，逐人转写 | DiCoW（FDDT）、SE-DiCoW | **未试** |

⚠️ **TS-ASR ≠ 级联**。级联是两模块事后拼接；TS-ASR 把说话人活动概率注入 ASR 每一帧
（FDDT：静音/目标/非目标/重叠 四态各走一套可训练仿射变换）。

### 各赛事实际获胜范式

| 赛事 | 冠军 | 范式 | 分数 |
|---|---|---|---|
| CHiME-8 NOTSOFAR-1 SC | USTC-NERCSLIP | **Dia-Sep-ASR** | tcpWER 22.2% |
| CHiME-8 NOTSOFAR-1 MC | USTC-NERCSLIP | 同上 | tcpWER 10.8% |
| MLC-SLM 2025 Task 2 | MegaAIS | **级联**（FSMN + CAM++ + SLLM）| tcpMER 16.53 |
| MLC-SLM 2025 Task 2 亚军 | BUT | **TS-ASR**（DiariZen + DiCoW）| tcpMER 16.75 |
| **MLC-SLM 2026 Task 1** | **MOSS-Transcribe-Diarize** | **端到端** | **tcpMER 14.93** |

**趋势**：2025 年端到端垫底（唯一一支 27.25，第 8 名），级联夺冠；**2026 年端到端反超夺冠**。
范式优劣在一年内翻转，而我们正站在赢的那侧。

## 2. 候选清单（权重可用性全部核过）

| 模型 | 参数 | 权重 | 许可 | 中文 | 关键数字 | 判定 |
|---|---|---|---|---|---|---|
| **MOSS-Transcribe-Diarize 0.9B** | 0.9B | ✅ HF | Apache-2.0 | ✅ | MLC-SLM 2026 **冠军**；AliMeeting cpCER 22.17 | **现主线** |
| MOSS-Transcribe-Diarize **Pro** | ? | ❌ **仅 API** | 闭源 | ✅ | AliMeeting cpCER **13.94**（强于 0.9B 37%）| **排除**（闭源）|
| **Speaker-Reasoner / -4194h** | 32B (MoE 3B 激活) | ✅ HF | Apache-2.0 | ✅ 原生 | AISHELL4 cpWER 8.14 / AliMeeting-Far cpWER 19.92 | **首选贵档** |
| **FireRedASR2-AED / -LLM** | 1.1B / 8.3B | ✅ HF+MS | 开源 | ✅ SOTA | Mandarin-4 **CER 2.89%**，明确优于 Qwen3-ASR-1.7B / Doubao / Fun-ASR | **首选便宜档** |
| DiCoW_v3_MLC / SE-DiCoW | 1.0B | ✅ HF | CC BY 4.0 / Apache | ⚠️ 训练无中文 | GT 切分下 OOD **22.0 输给 Whisper 16.8** | **降级** |
| SpeakerLM (AAAI-26) | ? | ❌ 未放 | — | ✅ | AliMeeting cpCER 16.05 (7638h) | 阻塞 |
| TagSpeech-Alimeeting | 7B | ✅ HF | — | ✅ | 第三方复测 AliMeeting cpCER **68.74** | 排除 |
| SoulX-Transcriber | 30B | ✅ | — | ✅ | 官方 DER 5.39 / **本赛实测 30.45%** | 已封死（D6.5）|
| Qwen2.5-Omni-7B | 7B | ✅ | Apache | ✅ | AISHELL4 **DER 85.68** | 排除（印证 X10）|

## 3. 三条硬结论

### 3.1 域墙的定量形态（[[neural-diarizer-domain-wall]] 第八次印证）

SoulX-Transcriber 官方：AliMeeting DER **5.39** / cpWER 13.61，AISHELL-4 DER **2.89** —— 全面 SOTA。
本赛 dev 实测 **30.45%**，009 段 ref 5 人判成 2 人，全量 6 人段为 0。

**含义**：本赛数据与 AliMeeting/AISHELL-4 的分布差异，大到能让 DER 5.39 的模型塌成灾难。
**任何论文/榜单数字对本赛不构成证据。**

### 3.2 X13 的外部独立验证

MLC-SLM 2026 第 6 名（SQZ，tcpMER 17.97，arXiv 2607.08208）系统构成：
FSMN VAD → 子段切分 → CAMPPlus 声纹 → 谱聚类 → RTTM 切音频 → **Qwen3-ASR-1.7B**，
ASR 侧做满**全量 SFT + TTS 合成数据 LoRA + GRPO 强化学习**三段适配。结果仍差 MOSS **3.04 点**。

**含义**：我们 X13 用零适配 Qwen3-ASR 输 MOSS 1.74 点；人家做满三段适配仍输 3 点。
**「Qwen3-ASR 在这类任务上不如 MOSS」是可复现的普遍事实，不是我们实现问题。**
**推论：X13 封条应扩展覆盖「微调版 Qwen3-ASR」**——加微调救不回来。

### 3.3 DiCoW 评级下调（推翻本人早前初判）

早前判断「DiCoW 是唯一没被死墙覆盖的候选」，理由是不切片 + 消费我们已有时间轴。**该判断需下调**：

| 证据 | 内容 |
|---|---|
| 训练语种 | MLC-SLM 15 语种：英/法/德/意/日/韩/葡/俄/西/泰/越/菲…… **无中文** |
| GT 切分下表现 | Baseline Whisper-large-v3 **16.8** vs DiCoW OOD **22.0** vs DiCoW 微调后 12.9 |

**第二条致命**：DiCoW 的收益主要来自**兜住烂 diarization**（real-diar 口径 76.1→28.4，巨大），
而**切分已经不差**时它反不如普通 Whisper。本赛正属后者（归属轴 oracle 仅差 3.784 点）。
要翻盘必须微调，而微调在本项目已两次线上倒退（v012 0.17023 / v013 0.16872）。

## 4. 附带发现：切分时长是被忽略的廉价杠杆

MLC-SLM 2025 第 4 名 Seewo 消融：

| 声纹模型 | 最大切分时长 | DER | tcpWER |
|---|---|---|---|
| CAM++ | 60s | 16.71% | **24.64%** |
| CAM++ | 8s | 17.28% | **19.32%** |
| ERes2Net | 8s | 17.22% | 19.27% |
| ResNet-101 (Wespeaker) | 8s | **16.78%** | **18.03%** |

**仅把切分上限 60s→8s 就降 5.32 点，而 DER 反而略升。** 原因：60s 超 Whisper 编码器最大长度。

⚠️ 对本赛适用性存疑：我们音频本身仅 30–45s，MOSS 又是 128k 长上下文模型，未必有同款瓶颈。
但通则成立：**diar 质量（DER）与下游 ASR 质量不单调**——与 D6.4「人数判对 ≠ 时间轴切对」同源。

另：**ResNet-101（Wespeaker，训练于 CnCeleb1/2 + VoxBlink1/2）在该消融中优于 CAM++ 与 ERes2Net**。
本赛只试过 CAM++（活）与 ERes2NetV2（死，D6.4），**ResNet-101 未试**。

## 5. 选型结论

### 决策约束

- 需 dev **2.03 点**（架构级 65% 迁移，见 [[dev-online-transfer]]）
- 文本轴空间 5.563 / 归属轴 3.784 / 择优轴 2.252（结构性不足）
- 现有 3 源仲裁上界 **1.732 < 2.03**（见 [[ensemble-oracle-bound]]）
- CLAUDE.md 方向选择原则 4：**有便宜活路时别先碰贵的**

### 推荐次序

**第一档（便宜，先做）：FireRedASR2 作为新文本源**

- **机制差异**：中文专精，且**明确在 Mandarin 基准上打败 Qwen3-ASR-1.7B**，
  与 X13 用的模型是不同数据点，不被 X13 封条覆盖。
- **成本极低**：`asr_retext.py` 已能按任意段边界跑任意 ASR，换模型≈换一行。
- **决定性判据**：用 `oracle_arbitrate.py` 重算「MOSS + FireRedASR2」仲裁上界。
  - 上界 > 2.03 → 文本轴多源重开，值得做真实仲裁器
  - 上界仍 ≈1.7 → **文本轴多源彻底判死**，只剩换主线一条路
- 一次全 dev 推理同时回答「换单源」与「多源仲裁」两个问题。

**第二档（贵，仅当第一档判死后）：Speaker-Reasoner 替换主线**

- **唯一机制真正不同的端到端候选**：agentic 多轮时间推理（全局说话人摘要 → 边界预测 → 分片解码），
  非单遍解码。**前八次域墙全是单遍模型。**
- **训练与评测按 40–50s 分段**，与本赛 30–45s 音频高度吻合。
- ⚠️ **最大风险**：backbone = Qwen3-Omni-30B-A3B，**与已封死的 SoulX 同一底座**。
  若失败方式又是塌说话人 → 可判定「Qwen3-Omni 系底座对本赛不可用」，封死整个家族（信息价值高）。
- 成本：32B MoE，4090 24GB 装不下 bf16，需 A100 80GB 级或量化。

**不推荐**：DiCoW（§3.3）、TagSpeech（第三方复测极差）、SpeakerLM（无权重）、
MOSS Pro（闭源，违反约束）、任何 Qwen3-ASR 变体（X13 + 外部验证双重封死）。

### 对「换掉 MOSS」的正确期望

MOSS 是 MLC-SLM 2026 冠军。**替换它 = 打赢卫冕冠军**，不是「换个更大模型碰运气」。
公开数据上唯一明确强于 MOSS 0.9B 的是 **MOSS Pro（闭源，排除）**。
Speaker-Reasoner 与 MOSS **从未在同一 benchmark 同口径对比过，孰强孰弱无公开证据**。

## 来源

1. [DiCoW 期刊论文 (Computer Speech & Language 2026)](https://www.sciencedirect.com/science/article/pii/S088523082500066X)
2. [Target Speaker ASR with Whisper (ICASSP 2025)](https://arxiv.org/html/2409.09543v2)
3. [BUT-FIT/DiCoW_v3_MLC 模型卡](https://huggingface.co/BUT-FIT/DiCoW_v3_MLC) — 权重/许可/MLC-SLM dev 全表（无中文）
4. [BUTSpeechFIT/DiCoW GitHub](https://github.com/BUTSpeechFIT/DiCoW) — 组件许可拆分
5. [CHiME-8 NOTSOFAR-1 官方结果](https://www.chimechallenge.org/challenges/chime8/task2/results)
6. [NOTSOFAR-1 挑战赛总结 arXiv:2501.17304](https://arxiv.org/html/2501.17304v2)
7. [USTC-NERCSLIP CHiME-8 系统报告](https://www.isca-archive.org/chime_2024/niu24_chime.pdf)
8. [BUT/JHU CHiME-8 系统报告](https://www.isca-archive.org/chime_2024/polok24_chime.pdf)
9. [MLC-SLM 官方赛事页](https://www.nexdata.ai/competition/mlc-slm) — 2026 Task 1 排行榜
10. [MLC-SLM 挑战赛总结 arXiv:2509.13785](https://arxiv.org/html/2509.13785v1)
11. [Seewo MLC-SLM 提交 arXiv:2506.13300](https://arxiv.org/html/2506.13300v3) — 切分时长/声纹消融
12. [Diarization-Guided Qwen-ASR Adaptation arXiv:2607.08208](https://arxiv.org/abs/2607.08208v2) — X13 外部验证
13. [MOSS Transcribe Diarize 技术报告 arXiv:2601.01554](https://doi.org/10.48550/arxiv.2601.01554)
14. [OpenMOSS/MOSS-Transcribe-Diarize GitHub](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) — 夺冠公告 / Pro 版 API-only
15. [Speaker-Reasoner arXiv:2604.03074](https://arxiv.org/abs/2604.03074v1)
16. [Speaker-Reasoner GitHub](https://github.com/ASLP-lab/Speaker-Reasoner) ｜ [权重](https://huggingface.co/ASLP-lab/Speaker-Reasoner-4194h)
17. [SoulX-Transcriber 官方 Demo](https://soul-ailab.github.io/soulx-transcriber/) — 官方 DER 5.39 对照本赛 30.45%
18. [FireRedASR2S GitHub](https://github.com/FireRedTeam/FireRedASR2S) ｜ [技术报告 arXiv:2603.10420](https://arxiv.org/html/2603.10420v1)
19. [AudenAI/TagSpeech-Alimeeting 模型卡](https://huggingface.co/AudenAI/TagSpeech-Alimeeting)
20. [SpeakerLM arXiv:2508.06372 (AAAI-26)](https://arxiv.org/abs/2508.06372) — 权重未释出
21. [多说话人 E2E ASR 综述 arXiv:2505.10975](https://ar5iv.labs.arxiv.org/html/2505.10975)

## 方法

子问题 5 个：① TS-ASR 机制全景 ② 三大赛事获胜方案实际构成 ③ 中文实测数字
④ 权重可用性与许可证 ⑤ 输入接口与落地约束。
6 轮检索 + 6 个关键源全文精读（DiCoW 模型卡/仓库、MOSS 仓库、Speaker-Reasoner 仓库、
TagSpeech 模型卡、arXiv 2607.08208）。共 21 个来源。
