---
slug: diarization-sota-2026
type: bound
induced_by: [deep-research-2026-07-28]
---

# 2026 说话人分离 SOTA 地图（deep-research 2026-07-28，102 agents / 20 源 / 92 claims / 21 confirmed）

## 结论

**2026 年没有单一模型满足本赛全部约束**（2026 新发布 + Mac M4 本地 + 中文 overlap + 开源合规 + 优于 CAM++ 22.79% tcpWER）。但有两个**未在本赛试过的 2026 端到端 SAT 模型**最值得探针：

1. **MOSS-Transcribe-Diarize 0.9B**（OpenMOSS，2026-07-09 发布）— 端到端 ASR+diarization，中文 cpCER 15.83(AISHELL-4)/22.17(Alimeeting)，MLC-SLM 2026 挑战赛冠军。Apache-2.0 repo。**阻塞：M4/CUDA 兼容性未验证**（vLLM/SGLang 是 CUDA 专用，MPS 路径未文档化）。
2. **TagSpeech**（ACL 2026 Main Oral）— 端到端 LLM 联合 ASR+diar，在 AliMeeting(中文) 比 LLM 基线降 36% DER。HF 权重存在（AudenAI/TagSpeech-Alimeeting）。**阻塞：权重许可证未确认 + M4 兼容性未验证**。

其余 2026 模型都不适用：
- **JEDIS-LLM**（ICASSP 2026，Microsoft）：架构理想但**无开源权重/许可证**（Microsoft 专有，Phi-4 底座非 OSI 合规），不可用。
- **DM-ASR**（2026-04，武大/腾讯）：论文自述"小模型规模下 LLM 预测说话人标签不优于强 diar 前端"——**反而强化当前 CAM++ 架构**，不是替代品。
- **Ultra-Sortformer**（2026-04，Apache-2.0）：扩展已证伪的 Sortformer 家族（X5），且训练数据是韩语合成语音，中文 zero-shot，out-of-domain。
- **SE-DiCoW**（ICASSP 2026，BUTSpeechFIT）：target-speaker ASR，用 diar 输出做 conditioning，**不替换 diar**（默认用已证伪的 DiariZen）。
  > ⚠️ **2026-08-06 复评并降级**：本页当时只从「替换 diar」角度看。从**文本轴**角度重估后仍不推荐——
  > DiCoW 训练语种**无中文**，且在 GT 切分下未微调版（22.0）**输给普通 Whisper-large-v3（16.8）**，
  > 其收益主要来自兜住烂 diarization，而本赛切分已不差。详见 [ts-asr-model-selection](ts-asr-model-selection.md)。

## 证据（deep-research 21 confirmed / 4 refuted）

### 2026 候选矩阵

| 模型 | 年份 | 中文? | 指标(非tcpWER) | 许可证 | M4? | 端到端? | 本赛试过? |
|---|---|---|---|---|---|---|---|
| **MOSS-Transcribe-Diarize 0.9B** | 2026-07 | ✓(cpCER 22.17 Alimeeting) | cpCER | Apache-2.0 repo | ❓未验证 | ✓ | 未试 |
| **TagSpeech** | 2026(ACL) | ✓(AliMeeting DER 22.13) | DER | ❓权重许可未确认 | ❓未验证 | ✓ | 未试 |
| JEDIS-LLM | 2026(ICASSP) | — | DER | ✗无开源 | ✗CUDA only | ✓ | 不可用 |
| DM-ASR | 2026-04 | ✓ | tcpCER | 未确认 | ❓ | ✓(需外diar) | 不适用(强化当前架构) |
| Ultra-Sortformer | 2026-04 | ✗韩语训练 | DER 16.96(4spk) | Apache-2.0 | ❓H100 | ✗(同X5家族) | X5已证伪 |
| SE-DiCoW | 2026(ICASSP) | — | tcpWER(EMMA) | 未确认 | ❓ | ✗(conditioning) | 不替换diar |
| pyannote community-1 | 2025-09 | ✓(AliMeeting DER 20.3) | DER | CC-BY-4.0 | ✓纯PyTorch | ✗(聚类范式) | 未试(gated token阻塞) |
| EEND-TA | 2025(Interspeech) | ✓(AliMeeting-far DER 11.41) | DER | 未确认 | ❓13.3M轻量 | ✓EEND | 未试(权重未发布) |
| DiariZen | 2025 | ✓ | DER | CC BY-NC ✗ | 已跑 | ✓EEND | X4证伪 |

### 关键 confirmed findings

1. **MOSS-Transcribe-Diarize**（3-0 confirmed）：OpenMOSS 官方 GitHub（1143 stars，Apache-2.0，2026-05-13 创建），HF 卡 OpenMOSS-Team/MOSS-Transcribe-Diarize，arXiv 2601.01554。中文 cpCER 数字逐字核对（0.9B: AISHELL-4 15.83 / Alimeeting 22.17；Pro: 14.02/13.94）。MLC-SLM 2026 挑战赛冠军被 refuted（0-3，证据不足）但模型本身 confirmed。
2. **TagSpeech**（3-0 confirmed）：arxiv 2601.06896 真实（ACL 2026 Main Oral）。AliMeeting-Far DER 22.13 vs Qwen3.0-Omni-30B 34.60（36% 相对改善）。HF 权重 AudenAI/TagSpeech-AMI + AudenAI/TagSpeech-Alimeeting 存在。**caveat**：AMI 上 cascade(Pyannote3.1+Whisper-large-v3, DER 23.05) 略胜 TagSpeech(24.84) → E2E 不总优。
3. **JEDIS-LLM**（3-0 confirmed 不可用）：arXiv 2511.16046，ICASSP 2026。全文无 weights/GitHub/license。底座 Phi-4-Multimodal 非 OSI 合规。训练要 16×A100 80GB。
4. **DM-ASR**（3-0 confirmed 强化当前架构）：arXiv 2604.22467。论文自述"fully LLM-predicted speaker labels and timestamps still do not consistently outperform strong diarization front-ends"——**这是 2026 论文亲口说 LLM 端到端不如强 diar 前端**，支持本项目保留 CAM++ 前端 + 改进文本端的思路。
5. **Ultra-Sortformer**（3-0 confirmed 不适用）：repo benchmark 只有 4spk 基座（DER 16.96）+ pyannote(19.00)，**无 N=5/N=8 DER**。训练数据韩语合成（~10k 小时，3400+ 韩语说话人），中文仅 zero-shot 测试。X5 家族，已证伪。

### 关键 refuted claims

- ❌ "SE-DiCoW 在 EMMA 上 tcpWER 降 52.4%"（1-2 refuted）：该 tcpWER 数字主源验证不足。
- ❌ "MOSS 拿了 MLC-SLM 2026 冠军"（0-3 refuted）：证据不足（但模型本身真实）。
- ❌ "MOSS 必须 CUDA，M4 不行"（1-2 refuted）：M4 兼容性**不确定**（不是确认不行）——是 open question 非 hard block。

## 含义（对方向选择）

### 两个未试的 2026 端到端 SAT 候选
- **MOSS-Transcribe-Diarize**：最优先。Apache-2.0 合规，中文 cpCER 数字最硬（22.17 Alimeeting），2026-07 刚发布。**真正阻塞是 M4 能不能跑**（vLLM/SGLang 是 CUDA 专用，但 0.9B 小模型可能 CPU/MPS 能跑——要烟测）。
- **TagSpeech**：备选。HF 权重在，但权重许可证未确认（可能不合规）+ M4 兼容性未验证。

### DM-ASR 的反向信号
2026 论文亲口说"LLM 端到端不如强 diar 前端"——**支持当前架构**（保留 CAM++ 前端）。这意味着即便 MOSS/TagSpeech 的 E2E 路线在 benchmark 上好看，在本赛可能不如当前 cascade 架构。但仍值得探针验证（benchmark ≠ 本赛数据，X4/X5 都是 domain transfer 失败的实例）。

### 指标不可比（critical caveat）
- 所有 2026 模型报的是 DER（0.25s collar，no overlap skip）或 cpCER，**没一个报 tcpWER**。
- DER/cpCER 排名不预测 tcpWER 排名（本赛 v003/v004 都是 benchmark 看好但本赛实测输的实例）。
- **任何候选必须在本赛 dev 上实测**，不能凭 benchmark 下判断。

## 下一步（落 wiki/log.md select）

1. **MOSS-Transcribe-Diarize M4 烟测**（最优先）：
   - clone OpenMOSS/MOSS-Transcribe-Diarize，下 0.9B 权重
   - 在 M4 上跑 1 段 dev（如 094），看能不能跑通 + 速度
   - 若能跑 → 全量探针 + 评测
2. **TagSpeech 权重许可证确认**：查 HF 卡 AudenAI/TagSpeech-Alimeeting 的 license 字段。若 OSI 合规 → M4 烟测。
3. 主线维持 v002=22.79%（线上 19.47%）。

## 引用源（primary）
- arxiv 2601.06896（TagSpeech, ACL 2026）
- github.com/OpenMOSS/MOSS-Transcribe-Diarize（Apache-2.0, 2026-07）
- arxiv 2601.01554（MOSS tech report）
- arxiv 2511.16046（JEDIS-LLM, ICASSP 2026，无开源）
- arxiv 2604.22467（DM-ASR, 2026-04，强化当前架构）
- github.com/mago-research/Ultra-Sortformer（X5 家族，韩语训练）
- huggingface.co/pyannote/speaker-diarization-community-1（2025, CC-BY-4.0）
- arxiv 2509.14737（EEND-TA, Interspeech 2025）
