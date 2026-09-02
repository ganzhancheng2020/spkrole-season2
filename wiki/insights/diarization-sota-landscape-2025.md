---
slug: diarization-sota-landscape-2025
type: bound
induced_by: [deep-research-2026-07-26]
---

# 说话人分离 SOTA 地图（2024-2025，deep-research 2026-07-26）

## 结论

**CAM++ 谱聚类不是 SOTA**——但本赛约束（FSQ#7 开源合规 + M4 本地 + 无 HF token 偏好 + tcpWER 非 DER）让"换不换、换成什么"不是一句话能答。三条硬约束框死了候选空间：

1. **CAM++ 是聚类范式，结构性处理不了 overlap**（arxiv 2303.05397：聚类假设每段单说话人，overlap 被整段吞，是范式限制非可调参数）。本赛 dev 83/106 段含 overlap → 这正是 v002 距 oracle 上界（15.32%）还差 7.5 点的根因之一。
2. **X4（D3-DiariZen 封死）有漏洞**：v003 探针用的是 DiariZen **默认 AHC 聚类**，**不是 SOTA 的 cVBx 配置**（cVBx 比 AHC 降 ~1.5 点 DER）。那 17.5 点反差可能是聚类配置 artifact，不是模型本身失败——**是未验证假设，不是已确认证伪**。
3. **DiariZen 权重 CC BY-NC 4.0**（非商用）→ 即便 cVBx 能赢回分数，FSQ#7 合规存疑，路线被许可证堵死。

## 证据（deep-research 21 源 / 102 claims / 19 confirmed / 6 refuted）

### 候选方案矩阵

| 方案 | DER（公开 benchmark） | overlap | 许可证 | M4 可行性 | 本赛实测 |
|---|---|---|---|---|---|
| **CAM++（当前）** | 聚类范式，无原生 overlap | 弱 | modelscope 开源 ✓ | 已跑通 | dev 22.79% / 线上 19.47% |
| **DiariZen-Large + cVBx** | 13.0% avg（8 benchmark），DIHARD3 14.5% | 原生（EEND powerset） | **CC BY-NC 4.0** ✗ | dz_venv 已装，~33s/段 | v003 用 AHC 探针：overlap 段 43.14% vs CAM++ 25.61%（差 17.5 点）|
| **DiariZen-Large + 默认 AHC** | 非 SOTA（cVBx 才是） | 原生 | CC BY-NC ✗ | 同上 | 就是 v003 测的那个，次优配置 |
| **Sortformer v2-streaming（NVIDIA）** | AliMeeting（中文会议，19% overlap）**DER 7.0%**，最佳 | 原生（4-spk EEND + Arrival-Order Cache） | **CC BY 4.0**（可商用）✓ | 未知（无 ARM 测试）| 未测 |
| **EEND-TA（Interspeech 2025）** | DIHARD III **14.49%**（least-forgiving，含 overlap，无 collar）| 原生（8 speaker attractor）| **未知**（权重是否发布未确认）| **最可行**：13.3M 参数，CPU 460x realtime，VRAM 1.6 GiB | 未测 |
| **pyannote.audio community-1** | DIHARD3 20.2 vs 3.1 的 21.4；AMI/CALLHOME 全胜 | 弱-中 | MIT ✓ | 纯 PyTorch wrapper，MPS/CPU 可跑 | 未测 |
| **3D-Speaker（modelscope）** | 同聚类范式（=CAM++ 家族） | 弱 | 开源 ✓ | 已可装 | 不是范式升级 |
| **DISPLACE 2024 冠军** | DER 27.1%（far-field 多语 3-5 人 30-60min）| — | — | — | 集成（pyannote powerset + PixIT），非单模型 |

### 关键 confirmed findings

1. **DiariZen-Large 是平均最强开源**（13.3% DER，仅次于商用 pyannoteAI 11.2%），但 SOTA 要求 cVBx 替换默认 AHC（arxiv 2510.19572，ICASSP 2026，DiariZen 维护者论文）。
2. **Sortformer v2 在中文 overlap 场景最强**：AliMeeting DER 7.0%，beat DiariZen 10.8% / pyannote 24.4%（但 caveat：NVIDIA 自己 HF 卡在另一个 split 报 19.6-22.1%，7.0% 可能不复现）。
3. **EEND-TA 是最新 + 最轻**（Interspeech 2025）：13.3M 参数 vs pyannote 8.1M，CPU 460x realtime vs pyannote 3x，DIHARD III 14.49% 无 forgiveness。**M4 最可行候选**（但权重发布状态 + 许可证未确认，是 open question）。
4. **DISPLACE 2024 + DIHARD III 冠军都不是单模型 EEND**：DISPLACE 冠军是 pyannote powerset + PixIT 集成；DIHARD III 冠军是 VB-HMM + TS-VAD + DOVER-Lap 系统组合。→ **overlap-hard 场景单模型 EEND 不是必胜模式，混合集成仍竞争力强**。
5. **MISP 2025 Track 1/3 一等奖**：hybrid WavLM-Large E2E + VBx（overlap-adaptive，按 overlap 程度切换 EEND/聚类）→ **2025 overlap 会议音频 SOTA 是混合而非纯 EEND**。

### 关键 refuted claims（对抗验证否决的）

- ❌ "pyannote 3.1 在中文特别弱"（0-3 否决）：聚合中文 SF v2=9.2 / DiariZen=10.1 / pyannote=19.8 的排序被否，benchmark 不含 CAM++，不能据此否定当前方向。
- ❌ "混合架构是获胜模式，纯 EEND 在 overlap-count-ambiguous 段会输"（0-3 否决）：断言过度，DISPLACE/DIHARD 冠军用集成不代表纯 EEND 必输。
- ❌ "DiariZen 默认 AHC 就是 SOTA"（1-2 否决）：明确 SOTA 要求 cVBx。

## 含义（对方向选择）

### X4 封死的修正
- **不能继续当 D3-DiariZen 已确认死**。v003 用了次优 AHC，cVBx 没测。但 **CC BY-NC 许可证堵死了 DiariZen 路线**（即便 cVBx 能赢回分数，FSQ#7 合规存疑）。→ **X4 维持封死，但死因从"模型无优势"修正为"许可证不合规"**，留备注"cVBx 配置未测，若许可证问题解决可重测"。

### cVBx 扫参验证（2026-07-26，见 logs/2026-07-26_cvbx.md）

deep-research caveat 6 的"配置 artifact"假设**部分成立但未反转结论**：
- 修改 diarize_dz.py 加 `--fa`/`--fb` 暴露 VBx 超参，在 5 段 overlap 密集段扫 4 组 Fa/Fb：
  - 0.07/0.8（默认）→ 43.14%（v003 原数据）
  - 0.3/1.0 → 32.18%（-11 点）
  - 0.5/1.2 → **30.22%**（-2 点，渐近）
  - 0.7/1.5 → 30.22%（饱和）
- **cVBx 调参把 17.5 点反差缩到 4.6 点**，但仍输 CAM++ 默认 25.61%（在这最难的 5 段）。
- **关键事实修正**：DiariZen 默认聚类已是 VBxClustering（非 deep-research 说的"默认 AHC"），caveat 6 判断有误读，但 Fa/Fb 调参确实有效。
- **X4 死因再修正**：从"许可证不合规"再修正为"配置调优后仍输 CAM++ 默认 + 许可证堵死"。
- **未决**：全量 106 段（含简单段）cVBx 是否追平 CAM++ 未测（30-60min + 许可证堵死，不优先）。

### 新候选（按 cost × 可行性排序）
1. **EEND-TA**（最优先）：M4 最可行（13.3M 参数，CPU 460x RT），Interspeech 2025 SOTA。**阻塞**：权重是否开源发布 + 许可证未确认 → 先查 repo/权重，确认后再投入实测。
2. **Sortformer v2-streaming**：中文 overlap 最强（AliMeeting 7.0%），CC BY 4.0 可商用。**阻塞**：NVIDIA 卡报另一个 split 19.6-22.1%，7.0% 可能不复现；M4 ARM 可行性未知 → 需先小样本探针验证 domain transfer。
3. **pyannote community-1**：MIT 合规，纯 PyTorch 可 M4 跑，DER 全面优于 3.1。**阻塞**：HF gated（需一次性 token 接受），与"无 HF token 偏好"冲突但可接受（一次性）。是 CAM++ 的同范式升级，不是范式跳变。
4. **MISP 2025 hybrid（WavLM E2E + VBx，overlap-adaptive 切换）**：2025 overlap SOTA，但实现复杂度高（要拼 EEND + 聚类 + overlap 检测 + 切换逻辑），cost 档最贵，仅当便宜档都试穿再考虑。

### 不可越的硬约束（对所有候选）
- **DER ≠ tcpWER**：所有 benchmark 用 0.25s collar + skip_overlap=False，本赛 tcpWER 是 word-level + collar=5。DER 排名不预测 tcpWER 排名 → **任何候选都必须在本赛 dev 上实测，不能凭 benchmark DER 排序下判断**。
- **无 ARM/M4 实测**：所有 DER 数字来自 NVIDIA A6000。M4 可行性只能从参数量+RTF 推断，必须本地烟测确认。
- **domain transfer 风险**：会议语音是 DIHARD III 最难 domain（median DER 35-45%）。本赛中文短对话 overlap 分布相关但不等于 AliMeeting。**v003 的 17.5 点反差就是 domain transfer 失败的实例**，general-benchmark SOTA 排名不能 refute 它。

## 下一步（落 wiki/log.md select）

1. 先查 EEND-TA 权重发布状态 + 许可证（open question #1）——决定它能不能进候选池。
2. EEND-TA 若开源 → 小样本探针（同 v003 方法：overlap 密集段 + 简单段各 3-5 段，和 CAM++ 同段对照）。
3. Sortformer v2 若 M4 能跑 → 同样探针。
4. 都不行 → 回 D3.1 调 CAM++ 聚类阈值（recluster.py 扫参，主线维持 v002=22.79%）。

## 引用源（primary）
- arxiv 2509.26177（Lanzendoerfer 2025，benchmark 4 模型 × 4 数据集 × 5 语言）
- arxiv 2510.19572（Palka ICASSP 2026，DiariZen+cVBx SOTA）
- isca-archive interspeech_2025/broughton25（EEND-TA，14.49% DIHARD III）
- github.com/BUTSpeechFIT/DiariZen（benchmark 表 + CC BY-NC 许可）
- github.com/pyannote/pyannote-audio（community-1 MIT + benchmark）
- arxiv 2407.12743（DISPLACE 2024 冠军集成）
- isca-archive interspeech_2021/ryant21（DIHARD III domain 难度分析）
- arxiv 2303.05397（聚类范式 overlap 结构限制）
