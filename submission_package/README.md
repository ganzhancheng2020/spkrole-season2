# 说话人角色分离挑战赛 赛季2 — 参赛方案

多人对话音频 → 带说话人归属的转写（SegLST）。指标 **tcpWER**（meeteval 0.4.3，`--collar 5`，越低越好）。

| | |
|---|---|
| **最终成绩** | **tcpWER = 0.14309** |
| 官方基线 | fun-asr 直出 dev 26.01% |
| 线上提交 | 46 次，17 次刷新纪录（0.19467 → 0.14309，相对提升 26.5%）|
| 结果文件 | `prediction_result/result.json`（5186 条，SHA256 `47bfe5f98afbe1f0…`）|

**一句话方法**：双分支 + 规则路由 + 声纹校验的说话人拆分。

---

## 1. 目录结构

```
.
├── README.md                      本文件
├── requirements.txt               预测环境依赖
├── xfdata/                        官方原始数据（审核方放入，我方不打包）
├── user_data/
│   ├── model_data/
│   │   └── moss_sim_all106/       训练好的模型：MOSS-SAT 微调权重
│   └── tmp_data/
│       ├── stage_outputs/         Step 1-9 的归档产物（默认模式直接复用）
│       ├── sim_all106/            数据增广结果：800 条仿真对话（wav + 标注）
│       ├── train_jsonl/           微调用的 JSONL 训练清单
│       └── artifacts/             Step 6 所需的说话人似然打分
├── prediction_result/
│   └── result.json                test.sh 的产出
└── code/
    ├── test.sh                    预测入口（必选）
    ├── train.sh                   训练入口（必选）
    ├── requirements_train.txt     GPU 训练/重跑环境依赖
    ├── src/                       全部 Python 源码
    ├── test/run_stage1_9.sh       从原始音频重跑重阶段
    ├── train/                     微调脚本（finetune.py 来自 MOSS 官方仓库，Apache-2.0）
    └── docs/project.md            方案演进与复盘全文
```

**所有路径均为相对路径**：`code/` 下的脚本一律以 `code/` 为工作目录，用 `../xfdata/`、
`../user_data/`、`../prediction_result/` 访问包内其它目录，不含任何绝对路径。

---

## 2. 快速开始

### 2.1 预测

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
bash code/test.sh
```

约 1 分钟跑完，纯 CPU。预期输出结尾：

```
Step 13  CAM 引导 + 声纹校验的说话人拆分 → 最终结果
    INFO 拆分 14 次，覆盖 14 个 session；写出 5186 条

结果校验
  ✓ prediction_result/result.json（5186 条 SegLST 记录）
  ✓ 中间件 submission_v065（线上 0.14354）
    SHA256 e78bba70c9d0c1f0
  ✓ 最终结果 result.json（线上 0.14309）
    SHA256 47bfe5f98afbe1f0
```

默认模式复用 `user_data/tmp_data/stage_outputs/` 下已归档的 Step 1-9 产物。
要从原始音频完整重跑，把官方数据放进 `xfdata/` 后：

```bash
pip install -r code/requirements_train.txt      # 需 GPU
bash code/test.sh --full                        # 约 5 小时
```

### 2.2 训练

```bash
pip install -r code/requirements_train.txt
bash code/train.sh                              # 单卡 RTX 3090 约 6 小时
```

---

## 3. 系统依赖

### 3.1 硬件

| 用途 | 配置 |
|---|---|
| 预测（默认模式：Step 6 + Step 10-13） | 任意 x86/ARM，纯 CPU，内存 4 GB |
| 训练与全量重跑（Step 1-9） | NVIDIA RTX 3090 24 GB |

### 3.2 软件

| 项 | 版本 |
|---|---|
| 操作系统（GPU 侧，出货所用） | Ubuntu 22.04.5 LTS |
| 操作系统（预测侧，已验证） | macOS 26.2 / Ubuntu 22.04 均可 |
| Python | 3.11 ~ 3.13（预测）；3.12.13（训练，出货所用）|
| NVIDIA 驱动 | 580.105.08 |
| CUDA | 12.8（torch 轮子 cu128）；驱动报告 13.0 |
| cuDNN | 9.10.2 |
| PyTorch | 2.9.1 + torchaudio 2.9.1 |
| transformers | 5.14.1 |
| funasr / modelscope | 1.4.3 / 1.39.1 |

完整依赖见 `requirements.txt`（预测）与 `code/requirements_train.txt`（训练/重跑）。

**无需编译。** 全部为 Python 源码 + pip 依赖，不含 C/C++ 扩展，构建过程不需要 GCC/nvcc 参与。

---

## 4. 数据

仅使用官方数据，**未引入任何外部数据集**。

### 4.1 `xfdata/` 期望结构

审核方把官方原始文件放到 `xfdata/` 下，保持官网的文件名与层级：

```
xfdata/
├── dev/dev/
│   ├── ref.seglst.json     106 场参考标注（train.sh 用）
│   └── wav/                106 个 wav
└── test/test/
    └── wav/                394 个 wav（test.sh --full 用）
```

`code/test.sh` 会依次尝试 `xfdata/test/test/wav`、`xfdata/test/wav`、`xfdata/wav`，
命中任一即可，以兼容不同的解压层级。

SegLST 记录格式：

```json
{"session_id": "001", "start_time": 0.0, "end_time": 7.88, "speaker": "spk1", "words": "你 好 请 问"}
```

### 4.2 数据增广（规范 §4b）

我们对训练集做了一次数据增广，**增广代码与标注结果都在包内**：

| 项 | 位置 |
|---|---|
| 增广代码 | `code/src/simulate_conv.py` |
| 标注结果 | `user_data/tmp_data/sim_all106/ref.seglst.json` |
| 增广音频 | `user_data/tmp_data/sim_all106/wav/`（800 条）|
| 训练清单 | `user_data/tmp_data/train_jsonl/train.jsonl` |

**做法**：把官方 dev 的 106 场拆成单说话人片段，**跨 session** 重组成 800 条新的多人对话，
轮次结构（说话人数分布、发言时长、重叠率、静音间隔、切换率）照抄 dev 实测分布
（依据 arXiv 2204.00890：Simulated Conversations 优于随机拼接的 Simulated Mixtures）。

**两条纪律**：

1. **素材只来自 dev。** test 集全程不参与训练（赛题 §103/§148）。`simulate_conv.py` 把
   「素材只能来自 dev」写死在代码里，`--dev-root` 只能改 dev 目录的**位置**，不能换成 test。
2. **固定随机种子 `--seed 0`**，因此这 800 条可以逐字节重新生成 —— 我们实测过，
   重跑产出的 `ref.seglst.json` 与包内归档的 SHA256 完全一致。

---

## 5. 解决方案与算法

端到端 SAT 模型（MOSS）文本强但**系统性少估说话人数**；「ASR + 聚类」（fun-asr + CAM++）
说话人边界更贴合但文本弱。逐场路由取长补短，最后用独立声纹校验把被并成一个人的说话人拆开。

```
test wav ─┬─▶ MOSS-SAT ─▶ FireRed 重转写 ─▶ 词级仲裁 ─▶ 归属改写 ──┐ 311/394 场
          │                                                       ├─▶ pick_ensemble
          ├─▶ fun-asr ─▶ CAM++ 重打标签 ─▶ 词级仲裁 ──────────────┘  83/394 场
          │                                                            │
          │                                                     normalize + merge
          │                                                            │
          └─▶ CAM++ 分离 + 段级声纹 ─────────────▶ cam_split_verify ◀──┘
                                                          │
                                                   result.json
```

### 5.1 逐步说明

| Step | 做什么 | 脚本 | 耗时 | 输出 |
|---|---|---|---|---|
| 1 | CAM++ 谱聚类分离，同时缓存段级声纹（192 维）| `diarize.py` | ~7 min | `diar_test_m3_7_0.70` + `emb_test` |
| 2 | MOSS-SAT 解码：一次前向给出文本、时间戳与初始说话人 | `moss_sat.py` | ~2 h | `test_raw` |
| 3 | FireRedASR2 按 MOSS 的切分重新转写同一段音频 | `fr_retext.py` | ~1 h | `output_test_fr` |
| 4 | 词级声学仲裁：按差异块枚举 MOSS/FireRed 混合，声学似然选最优 | `word_arb.py` | ~1 h | `test_wa_m05b6` |
| 5 | 归属改写：段内多窗声纹一致性，留一法扣掉本段自己的窗 | `spk_reassign.py` | ~25 min | `test_v048_re` |
| 6 | 短段归属改写：用 MOSS 说话人似然，只改词数 ≤5 的段 → **MOSS 分支完成** | `spk_short_relabel.py` | ~10 s | `test_v051_moss` |
| 7 | fun-asr 转写（开 diarization，原生给出说话人）| `funasr_local.py` | ~1 h | `test_cam_multi` |
| 8 | 用 CAM++ 声纹为 fun-asr 的分段重打说话人标签 | `spk_relabel.py` | ~20 min | `test_cam_re` |
| 9 | CAM 分支的词级声学仲裁（配方同 Step 4）→ **CAM 分支完成** | `word_arb.py` | ~40 min | `test_cam_wa` |
| 10 | 逐场路由：按说话人数与轮换率在两分支间选一个（83/394 场走 CAM）| `pick_ensemble.py` | ~5 s | 路由后的 SegLST |
| 11 | 字符规范化：把 ref 词表中出现 0 次的字符改写成 ref 的写法 | `normalize_output.py` | ~3 s | 规范化后的 SegLST |
| 12 | 合并为单个 SegLST（此时线上 0.14354）| `merge_submit.py` | ~2 s | `submission_v065.json` |
| 13 | CAM 引导 + 声纹校验的说话人拆分 → **最终结果**（线上 0.14309）| `cam_split_verify.py` | ~15 s | `result.json` |

### 5.2 三个关键机制

**① 逐场路由（Step 10）。** MOSS 在部分场次会把多人塌缩成一两个人，而 CAM++ 在这些场更准。
判据是两个症状取并集：

```
few_spk  = MOSS 人数 ≤ 2
low_turn = MOSS 轮换率 ≤ 0.05
若 (few_spk 或 low_turn) 且 CAM 人数 ≥ 3  →  该场改用 CAM 分支
```

轮换率 = 该场中「与上一段说话人不同」的段数 ÷ 总段数。它低说明模型把长段连成一片、
很可能漏掉了说话人交替。

**② 词级声学仲裁（Step 4/9）。** 同一段音频，MOSS 与 FireRedASR2 各给一个转写；按差异块
对齐拆开，对最长的若干块枚举 2^k 种混合，用 FireRed 的 teacher-forcing 声学似然当裁判选最优。
关键是**裁判信号必须来自音频** —— 候选本身就是各模型的 argmax，任何文本层面的判据都是模型
已经用过的信息，只有回到波形才有增量。

**③ 声纹校验的说话人拆分（Step 13，本方案的关键增量）。** 我们的输出每场平均比参考少 0.426
个说话人。而 tcpWER 里「归错人」要赔两次（真说话人那边算 deletion、错的人那边算 insertion），
所以修对一次是双份收益。做法：某个 MOSS 说话人若在 CAM++ 时间轴上横跨 ≥2 个簇，就按簇拆开，
**但只有两半分段的 CAM++ 段级声纹（192 维）平均向量余弦 `< 0.50` 才真的拆**。

加这道校验前后的对比（dev 4 折交叉验证）：

| 配置 | 触发场数 | dev Δ | 剔除最幸运 3 场后 | P(更好) | 改善/恶化 |
|---|---|---|---|---|---|
| 无声纹校验 | 12 | −0.648 | **+0.301 ❌** | 83.0% | 7/5 |
| **加声纹校验** | **9** | **−0.982** | **−0.132 ✅** | **98.5%** | **7/2** |

余弦 0.60 是「两个人」与「把一个人劈成两半」的物理分界：`max_sim ≤ 0.60` 的每一格都通过
重尾检验，`≥ 0.65` 的每一格都不通过。嵌套验证（阈值只在内层折里选）给出诚实估计 **−0.7405 dev 点**。

更完整的演进过程、失败方向与量化归因见 `code/docs/project.md`。

---

## 6. 训练复现流程

链路里**只有一个模型是我们自己训的**：MOSS-Transcribe-Diarize 的说话人归属转写微调。
CAM++ / FireRedASR2-AED / fun-asr 三者全部直接使用开源权重，不做任何训练。

```bash
bash code/train.sh
```

三步（全部写在 `code/train.sh` 里，超参已固定，不接受命令行覆盖）：

| 步骤 | 做什么 | 脚本 |
|---|---|---|
| 1 | 数据增广：dev 跨 session 重组 800 条仿真对话 | `code/src/simulate_conv.py` |
| 2 | SegLST → MOSS 微调所需的 conversation JSONL | `code/train/mk_jsonl.py` |
| 3 | HF Trainer 微调 | `code/train/finetune.py`（MOSS 官方，Apache-2.0）|

**固定超参**（规范 §4b）：

| 超参 | 值 |
|---|---|
| 基座 | `OpenMOSS-Team/MOSS-Transcribe-Diarize`（0.9B，Apache-2.0）|
| 仿真数据条数 / 随机种子 | 800 / `seed=0` |
| max_length | 8192 |
| num_train_epochs | 2 |
| learning_rate | 1e-5 |
| lr_scheduler_type | cosine |
| warmup_ratio | 0.1 |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 4（等效 batch = 4）|
| optim | adafactor |
| 精度 | bf16 |
| gradient_checkpointing | 开 |

产物写到 `user_data/model_data/moss_sim_all106/`，即包内已预置的那份权重。

---

## 7. 结果

线上 46 次提交，**17 次刷新纪录**；累计 0.19467 → 0.14309（**−0.05158，相对提升 26.5%**）。

| # | 日期 | 线上 | Δ | 关键改动 |
|---|---|---|---|---|
| 1 | 07-26 | 0.19467 | — | 本地 CAM++ 重做说话人分离（min=3 / max=7 / merge_thr=0.70）|
| 2 | 07-29 | **0.16593** | **−0.02874** | **换范式：MOSS 端到端 SAT 0.9B（全程最大单次跃迁）** |
| 3 | 08-01 | 0.16379 | −0.00214 | 双分支按场路由：MOSS 人数≤2 或 轮换率≤0.10，且 CAM 人数≥3 → 改用 CAM |
| 4 | 08-03 | 0.16377 | −0.00002 | 路由加规则 D：CAM 人数≥6（多估）→ 换回 MOSS |
| 5 | 08-04 | 0.1634 | −0.00037 | 路由加规则 H：CAM==5 且 MOSS≥4（轻度少判）→ 换 CAM |
| 6 | 08-08 | 0.15903 | −0.00437 | 引入 FireRed 声学仲裁：**段级**二选一，teacher-forcing 似然打分 |
| 7 | 08-08 | 0.15756 | −0.00147 | 仲裁改**词级**粒度：段内按差异块切开，枚举 2⁴ 种混合 |
| 8 | 08-09 | 0.15573 | −0.00183 | 输出层规范化：把 ref 词表中出现 0 次的字符改写成 ref 的写法 |
| 9 | 08-11 | 0.15096 | −0.00477 | 仿真拼接微调 MOSS（dev 重组 800 条，2 epoch，lr=1e-5）|
| 10 | 08-13 | **0.14857** | **−0.00239** | **基座插值 soup + 归属轴 CAM 源重分配（首次破榜首）** |
| 11 | 08-14 | 0.14793 | −0.00064 | 归属轴双分支改写：MOSS 分支加一致性门控（0.55 / 0.05）|
| 12 | 08-17 | 0.14683 | −0.00110 | 全 106 场作仿真素材重训（源说话人 318 → 410，+29%）|
| 13 | 08-21 | 0.14625 | −0.00058 | word_arb 参数：`MARGIN` 0 → 0.05、`MAX_BLOCKS` 4 → 6 |
| 14 | 08-25 | 0.14507 | −0.00118 | CAM 分支**首次**做声学仲裁 + 短段归属改写 |
| 15 | 08-27 | 0.14443 | −0.00064 | 短段改写 1.5/3 + 路由 0.05 + 定向拆分（三层叠加）|
| 16 | 08-28 | **0.14354** | **−0.00089** | **路由 `turn_rate` 0.08 → 0.05（单变量）** |
| 17 | 08-31 | **0.14309** | **−0.00045** | **CAM 引导 + 声纹校验的说话人拆分（`sim<0.50`）** |

---

## 8. 模型与许可

| 模型 | 来源 | 许可 | 用途 | 是否训练 |
|---|---|---|---|---|
| MOSS-Transcribe-Diarize 0.9B | `OpenMOSS-Team/MOSS-Transcribe-Diarize`（HuggingFace）| Apache-2.0 | 端到端说话人归属转写 | **是**（见 §6）|
| CAM++ 说话人分离 | `iic/speech_campplus_speaker-diarization_common`（ModelScope）| 开源 | 分离 + 段级声纹 | 否 |
| FireRedASR2-AED | FireRedTeam | 开源 | 词级仲裁的声学裁判 | 否 |
| fun-asr | 阿里 FunASR（ModelScope）| 开源 | CAM 分支的转写文本 | 否 |

四个模型全部为可公开获取的开源模型，且全部运行在本队自有算力上，
**未通过 API 调用任何闭源大模型或方案**。

---

## 9. 运行时注意事项

1. **默认模式不需要 GPU，也不需要 `xfdata/` 里的音频。** Step 1-9 的产物已归档在
   `user_data/tmp_data/stage_outputs/`，`bash code/test.sh` 约 1 分钟即可产出结果并校验 SHA256。
2. **`--full` 重跑的哈希可能与归档值不同。** MOSS 在 bf16 下的贪心解码在 GPU 上不是逐位确定的，
   重解码与原始 `test_raw` 的差异实测约 +0.0013 tcpWER。`code/test.sh --full` 因此不把哈希
   不一致当作失败，只在默认模式下硬校验。
3. **Step 7 有两条等价路径。** `code/src/funasr_local.py` 用本地开源权重推理（默认，不依赖任何
   外部服务）；`code/src/run.py` 走我们自有托管实例的异步接口。二者产物格式相同，
   包内归档产物由后者生成。
4. **FireRedASR2-AED 需要手工装。** 它不在 PyPI 上，装好后用 `FIRERED_SRC` / `FIRERED_CKPT`
   两个环境变量指过去（`code/src/fr_retext.py`、`code/src/word_arb.py` 都读这两个变量）。
5. **`code/src/` 里除出货脚本外还有约 90 个探针脚本**，是探索过程中已否定方向的留存，
   不参与最终链路。出货链路用到的 12 个是：`diarize.py`、`moss_sat.py`、`fr_retext.py`、
   `word_arb.py`、`spk_reassign.py`、`spk_short_relabel.py`、`funasr_local.py`（或 `run.py`）、
   `spk_relabel.py`、`pick_ensemble.py`、`normalize_output.py`、`merge_submit.py`、
   `cam_split_verify.py`。
