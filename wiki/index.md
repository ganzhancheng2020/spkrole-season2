# 思路目录

> 按状态/family 分组的目录视图。ingest 时同步此文件，别让它和实际页脱节。
> 现存方向沿用 `memory/directions.md` 的 D/X 编号体系；方向数 <10，暂用线性管理，ingest 时在此同步分组。

## 主线（当前最佳方向）

- **D5.3 微调 MOSS 0.9B ＋ D5.1 按段择优** — **主线（v012，待提交）**。holdout 25 段 **16.196%**，同口径 v011 为 18.562% → 降 **2.37 点**。微调单独 −2.43、集成单独 −1.97，**近乎完美可加**。4 轮最优（12 轮过训），训练仅 176 秒。见 `logs/2026-08-01_finetune.md`。
- **D5.1 按段择优（双症状并集）** — **已收官，不再主攻**。v011 = dev 16.98% / **线上 0.16379（当前最佳）**。规则：`CAM++≥3人` 且（`MOSS≤2人` 或 `MOSS切换率≤0.10`）。脚本 `baseline/src/pick_ensemble.py`。⚠️ ~~**该轴结构性耗尽**~~ **← 2026-08-25 已推翻，见 [routing-gap-is-attribution](insights/routing-gap-is-attribution.md)：两支都换过后 oracle 13.750%、剔 top3 仍剩 1.471 点、收益分布在 50/106 段不再集中，而需求已从 0.81 降到 0.21 点。原判**：oracle 上界 14.73% 收益高度集中（前 10 段占 77.8%），剩余 2.25 点按 15% 迁移率仅值线上 0.34 点，**完美 oracle 也差榜首 1.02 点**。见 [dev-online-transfer](insights/dev-online-transfer.md)。
- **D5 端到端 SAT — MOSS 0.9B** — v010 的主系统底座。**v007 = dev 18.39% / 线上 0.16593**（Apache-2.0）。脚本 `baseline/src/moss_sat.py`（默认 max_new_tokens=2048 = v007 原始配置，改了复现不出）。

## 待试（dormant，仍开阔）

- **D5.1 续挖（换粒度）** — 残余 2.25 点。**session 级聚合特征已试穿**（nspk/nseg/seg_max/seg_mean/top_share/turn_rate/dur比/nseg比，325 条候选，嵌套 CV 仅 −0.38 点）。未试：两系统**时间轴级**分歧度（当前特征全是 session 级标量，看不见局部冲突）、或引入第三个源抬高 oracle 上界。
- **D5.3 续挖：扩训练数据** — dev 仅 56 分钟音频。赛题 §149 允许外部开源数据（需说明来源/规模/处理方式），中文多说话人候选：AliMeeting / AISHELL-4 / MagicData-RAMC。训练成本极低（176 秒/次），瓶颈在数据不在算力。
- **pyannote community-1** — MIT 合规，同范式升级（DER 全面优于 3.1）。阻塞：HF gated token。

## 部分有效 / 已降级

- **D3** — 本地 diarization 重打说话人标签（文本来自 fun-asr）。曾为主线，v002=22.79%（min=3/max=7/merge_thr=0.70，seed=0 可复现）。**被 D5 取代**（端到端 SAT 破其 4.40 点）。CAM++ 仍作为 D5.1 的第二源存活。
- **D3.1** — 逼近 oracle 上界（15.32%）：merge_thr 扫参到头（0.68/0.70 最优 22.82/22.79%，精细扫无更优），CAM++ 范式结构性限制，已挖到头。
- **D3.2 DOVER-Lap 集成** — 已封死（见 X6），但其「按段择优有真收益」的发现被 D5.1 继承，上界从 21.37% 放大到 15.05%。

## 封死（别再撞）

- **v004 Sortformer v2 单独** — 全量 dev 29.70% 输 v002 6.9 点。EEND powerset 多人段少判说话人（069:6人判3人），和 DiariZen 同病。falarm=2 极低（保守少判）。但 42 段赢 CAM++ → 进 D3.2 集成。详见 `logs/2026-07-27.md`。
- **D3-DiariZen** — EEND powerset 原生 overlap，但实测无优势。简单段 dz=15.94% 与 v000 同分；overlap 密集段 dz 默认 43.14% vs CAM++ 25.61%，差 17.5 点。**cVBx 扫参**（Fa0.5/Fb1.2）缩差到 4.6 点（30.22%）但仍输。DiariZen 默认已是 VBx 非 AHC。死因 = 配置调优后仍输 CAM++ + CC BY-NC 许可证。详见 `logs/2026-07-26.md` + `logs/2026-07-26_cvbx.md`。
- **X1** fun-asr-mtl 多模态端点 — 无提升（024/045/069 三段对比无改善）
- **X2** TTS 合成音频验证说话人分离 — 假阴性陷阱（声纹太近，4 句全标 spk0）
- **X3** LLM 云端 API 后处理 — 合规存疑（FAQ#7），暂锁直到组委会书面确认
- **D3.1① 按字数比例切分段（split 模式）** — oracle 下仍比 segment 差 2.8 点（sub 1097→1618），方向作废
- **X10** v006 qwen-omni-turbo 端到端 SAT — 通用多模态 LLM **猜时间戳**（001 音频 27.8s 只给到 14.56s），单段 113.99%。副产物有价值：专属实例 omni 可调 + `X-DashScope-OssResourceResolve` 机制可复用
- **X11** MOSS prompt 调参 — 段粒度/说话人数由权重决定不受 prompt 控制，但**输出格式对 prompt 极敏感**（改 prompt 后 6 段中 5 段 0 输出）→ 纯下行风险
- **X12** MOSS 文本 + CAM++ 说话人覆盖 — 27.79%，比 MOSS 单独差 9.4 点。**MOSS 自带说话人已优于 CAM++**（missed 52 vs 63），推翻「本地 diar 必优于模型自带 diar」的旧前提（那只对 fun-asr 成立）
- **X13 D5.2 MOSS 2B/Pro** — `MOSS-Transcribe-preview-2B` 是**纯英文 ASR 无 diarization**（language: en / pipeline_tag: ASR / 训练集全英文 / 架构不同），中文对话+说话人标签两条都不满足；v009 的「下载卡死」是伪问题（实测 20 分钟可下完，但下了也没用）。MOSS Pro 闭源 API 违 FAQ#7。**教训：换模型前先核 model card，别只看参数量**
- **R2 MOSS 段数≤5 → 用 CAM++** — train 上 −0.44 点、holdout 上 **+0.07 点**，仅 4 个训练段触发，判定过拟合丢弃。**留作「防过拟合切分为何必要」的实证**
- **R3 单用 turn_rate≤0.10** — train 比 v010 好 0.71 点、holdout 差 0.81 点。**前置检查全过仍不泛化**（嵌套 CV 4/5 折选中它、阈值扫描平滑最优、逐折最坏值优于 v010、dev/test 触发率吻合）。逐段诊断发现它漏掉 091（省 64 错误）→ 与 R1a **互补非替代**，改取并集（v011）。**留作「聚合指标看不见互补性，必须下到 session 级诊断」的实证**

## 按 family 分组

| family | 当前状态 | 代表版本 | 活跃 idea |
|--------|---------|---------|----------|
| D1 降 deletion/时长缺口 | 降级 | v000 诊断 | del=2473 是最大块，但纯文本 WER=12.89% 已含此漏字 → 次要因素 |
| D2 给 fun-asr 传正确说话人数 | 诊断工具 | — | dev oracle 验证用，不可迁移（test 无 ref） |
| D3 本地 diarization 重打标签 | 降级（被 D5 取代） | v002 | CAM++ 仍作 D5.1 第二源；DiariZen/Sortformer 均封死 |
| D3.1 逼近 oracle 上界 | 部分有效，已挖到头 | v002 | 词级映射证伪；merge_thr 扫参饱和 |
| D4 文本端改进 | 全证伪 | — | LLM/换 ASR/规则三路皆死，fun-asr 是文本天花板 |
| **D5 端到端 SAT** | **主线** | **v007** | MOSS 0.9B；prompt 调参与 CAM++ 覆盖已证伪（X11/X12） |
| D5.1 MOSS+CAM++ 按段择优 | 已收官（轴耗尽） | v011 | 线上 0.16379；完美 oracle 也差榜首 1.02 点 |
| D5.2 MOSS 2B/Pro | **封死**（X13） | — | 2B 是纯英文 ASR 无 diarization |
| **D5.3 微调 MOSS 0.9B** | **主线** | **v012** | holdout 16.196%（+集成）；与 D5.1 可加；下一步扩数据 |

> ⚠️ **惰性升级已触发**：方向数达 21（D 系 9 + X 系 12），超 CLAUDE.md 阈值（≥10 拆 `wiki/ideas/` 单页）。方向页正文目前仍全部存在 `memory/directions.md`，**拆页待人工确认**（2026-07-31 补录时未做，属结构性重构）。

## insights（跨思路硬结论）

- **[ts-asr-model-selection](insights/ts-asr-model-selection.md)** — TS-ASR 路线调研与开源模型选型（2026-08-06，21 源）。三条硬结论：① MOSS 是 MLC-SLM 2026 卫冕冠军，换掉它=打赢冠军；② X13 得外部独立验证（做满 SFT+LoRA+GRPO 的 Qwen3-ASR 仍输 MOSS 3.04 点）；③ 域墙定量化（SoulX 官方 DER 5.39 → 本赛 30.45%），**论文数字对本赛无预测力**。选型：便宜档 FireRedASR2、贵档 Speaker-Reasoner，DiCoW 降级。

- **[cssd-paradigm](insights/cssd-paradigm.md)** — ⭐ **本赛任务 = CSSD（对话短语说话人分离），八次「域墙」实为范式错配**。tcpWER 属 CDER 族而非 DER 族；CSSD 实测 SC 谱聚类 CDER 12.00 完胜 EEND 16.30–17.80，且 DER 与 CDER **反相关**。活下来的 CAM++ 正是谱聚类族。行动：改进留在谱聚类族内，首个旋钮 = 从未调过的 `seg_dur/seg_shift`。
- **[arbitration-signal-must-be-acoustic](insights/arbitration-signal-must-be-acoustic.md)** — ⭐⭐ **仲裁信号必须来自音频**。2026-08-06 十七次实验的分界线：文本派生信号（长度/自一致性/模型置信度/LM 流畅度）全部失败 16 次，因为看不见「文字是否与说出的话相符」；teacher forcing 声学似然成功（区分度 +1.195 方向正确），产出 v024（dev 16.090%，双切分同向 −0.47）。含 4 条可复用操作规则。⚠️ 其中「捕获率 43.7%」一句已被 [capture-rate-scales-with-headroom](insights/capture-rate-scales-with-headroom.md) 修正——那是 MOSS 源口径，不可外推。
- **[capture-rate-scales-with-headroom](insights/capture-rate-scales-with-headroom.md)** — ⭐⭐ **捕获率是源 headroom 的函数，不是系统常数**。D14 实测：同一裁判同一次运行，源在 27.8% 时捕获 **77.3%**、源在 12.9% 时捕获 **2.1%**（oracle 两边几乎相同 −1.505/−1.464）——**空间均匀，捕获不均匀**。机制：源已很好时两候选声学近乎等价，裁判退化为随机（15 胜/18 负/4 平）。教训：「oracle × 历史捕获率」外推到**被质量筛过的子集**会放大 20 倍，D14 就是这么被误判为「值得跑」的。副产物：仲裁把择优轴 oracle 从 0.730 扩到 1.054 点，但闸门仍是选源器。
- **[downstream-compensation-absorbs-gains](insights/downstream-compensation-absorbs-gains.md)** — ⭐⭐⭐ **下游补偿吃掉上游增益**。`pick_ensemble` 是强均衡器（切走的段 CAM 侧 8.81%，保留的段 18.91%），**任何让 MOSS 在「它本来就弱的段」上变好的改进端到端都归零**。2026-08-12 三次同形：D14 词级仲裁移到 CAM 段（77.3% 收益落在 pick 不选的 69 段，端到端 +0.057）、D13 model soup（raw −1.411 点，**−60 错误全在 pick 切走的 10 段**，保留的 14 段反而 +4，端到端 +0.068）、D12 保真度（切换率钉死，够不着瓶颈）。与 [capture-rate-scales-with-headroom](insights/capture-rate-scales-with-headroom.md) 是硬币两面：源好抓不到、源差已被补，**中间才是真空间**。**下一轮靶子**：pick 保留 MOSS 的 14 段 = 18.91%，ins128/del222/sub137、missed_spk 仅 6/54 → 说话人数找对了，错在边界与归属。
- **[cam-count-axis-has-optimum](insights/cam-count-axis-has-optimum.md)** — ⭐⭐⭐ **「少用 CAM」不是单调轴，它有极小值，v065 已经站在上面**。三点线上实测 (132段 0.14507 / 83段 0.14354 / 7段 0.14952) 二次拟合顶点 **x=89.7 / y=0.14350**，v065 距顶点仅 0.00004 → **该轴挖尽**；连带判死 v066(50段→0.14489) 与 v068(0段→0.15058)，两者都在顶点的下坡侧。死因：**两点定直线残差恒为 0，我把「拟合完美」当成「模型正确」**，而 dev 早就给出了转弯证据（纯 MOSS 16.810% vs 路由 16.229%，差 0.581 点，折算后正中线上实测）。预报误差 **+0.00835，本项目最大**。与 [gate2-not-overridable](insights/gate2-not-overridable.md) 同型的第三次复发：**为越过一个正确的反对意见而编出例外理由，三次误差 +0.0038/+0.0038/+0.0084 —— 越精心辩护错得越狠**。
- **[requirement-relative-verdicts-expire](insights/requirement-relative-verdicts-expire.md)** — ⭐⭐⭐ **「上界 < 所需」型封条会过期**。需求从 2026-08-05 的 **2.03 dev 点** 缩到今天的 **0.247 点（8.2×）**，但封条上的比较对象没重算 → `SCORES.md:326`「仲裁 family 上界 1.732 < 所需 2.03」**在算术上已失效**（1.732 是需求的 7.0 倍）。同型失效还有 `logs/2026-08-08.md:22`。**D5.1 那次（08-25）已经是同一种失效，我没把它一般化成规则，于是文本轴又躺了三周。**新测当前链路空间：**文本 4.757 点 / 归属 3.327 点**（需求 = 文本空间的 5.2%）。⚠️ 但 [capture-rate-scales-with-headroom](insights/capture-rate-scales-with-headroom.md) 是反证——源变好后捕获率会掉，悲观 2.1% 折算仅 0.036 点仍不够。**结论只到「重新可试」，不到「预期能赢」。**纪律：封条必须写明「所需」的取值与日期；死因要区分*机制死*（永久）与*需求死*（会复活）。🔴 **2026-08-28 晚追加**：同日那道「重新封死它」的探针**用错了源**（`_cv_wa`/`_cv_re` 都是 word_arb 输出，真正的输入是 `_cv`），正确上界 **0.5245 点 = 所需 0.247 的 2.1 倍** → 仲裁轴**仍开着**，但四个文本一致性过滤机制精度只有 32~39%（盈亏平衡 ~50%）⇒ **轴开着、机制枯**。第三条纪律：探针在盖封条前必须先证「输入 ≠ 输出」。
