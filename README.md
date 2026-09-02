# 说话人角色分离挑战赛 赛季2 — 参赛方案

多人对话音频 → 带说话人归属的转写（SegLST）。指标 **tcpWER**（meeteval 0.4.3，`--collar 5`，越低越好）。

| | |
|---|---|
| **最终成绩** | **tcpWER = 0.14309** |
| 提交文件 | `submit/archive/submission_v094b_20260831.json`（SHA256 `47bfe5f98afbe1f0…`）|
| 官方基线 | fun-asr 直出 dev 26.01% |

- **方案怎么长出来的、每一步具体怎么做的** → [project.md](project.md)（方案归档与复盘）

> ### ⚠️ 关于本公开仓库
>
> 赛题规定「大赛提供的全部数据、信息视为保密信息，未经允许不可以任何形式传播、披露」，
> 因此**本公开仓库不包含任何赛事数据及其衍生产物** —— 音频、参考标注、各阶段的转写产物
> （`baseline/output/`）、声纹缓存与提交文件（`submit/`）均未上传。
>
> 这意味着：**直接 clone 本仓库无法执行下面的一键复现**（缺中间产物）。
> 仓库提供的是**完整源码、方法文档与全过程记录**；
> 完整可复现包（含全部中间产物）随作品单独提交给主办方。

---

## Quick Start

**一条命令跑通全链路**（复用已归档的 GPU/云端阶段产物，本地约 1 分钟，最后校验 SHA256）：

```bash
bash run_full_pipeline.sh
```

```
Step 12  合并 → submission_v065.json
  ✓ submission_v065.json（线上 0.14354）
    SHA256 e78bba70c9d0c1f0
Step 13  CAM 引导 + 声纹校验的说话人拆分 → 最终文件
    INFO 拆分 14 次，覆盖 14 个 session；写出 5186 条
  ✓ submission_v094b.json（线上 0.14309，最终提交）
    SHA256 47bfe5f98afbe1f0
```

---

## Contents

[Hardware](#hardware) · [Software](#software) · [Data](#data) · [Solution Overview](#solution-overview)
· [Reproduce](#reproduce) · [Results](#results) · [Repository Structure](#repository-structure)
· [Models & Licenses](#models--licenses)

---

## Hardware

| 用途 | 配置 |
|---|---|
| 本地（合流、分离、评测） | Apple M4 / 16 GB / macOS 26.2 |
| 云端（MOSS 解码、FireRed 仲裁、微调） | NVIDIA RTX 3090 24 GB |

Quick Start 只需本地机器；从原始音频完整重跑需 GPU，见 [Reproduce](#reproduce)。

## Software

按用途分三个虚拟环境，**不要混用**：

| venv | Python | 关键依赖 | 用途 |
|---|---|---|---|
| `.venv` | 3.13.12 | `meeteval 0.4.3`、`numpy 2.5.1` | 评测与数值计算（`cam_split_verify` 需要）|
| `baseline/.venv` | 3.13.12 | `requests 2.34.2`、`openai 2.47.0` | 流水线脚本（**不含 numpy**）|
| `baseline/diarizen_venv` | 3.11.15 | `torch 2.13.0`、`modelscope 1.38.1`、`funasr 1.3.29`、`transformers 5.14.1`、`scikit-learn 1.9.0` | CAM++ 分离、声纹、词级仲裁 |

`run_full_pipeline.sh` 会自动分派解释器，并在启动前检查可用性。

## Data

仅使用官方数据，**未引入任何外部数据集**。

| 数据 | 规模 | 用途 |
|---|---|---|
| `data/extracted/dev/dev/` | 106 场 wav + `ref.seglst.json` | 参数选择、4 折交叉验证 |
| `data/extracted/test/test/wav/` | 394 场 wav | **仅推理**，从未参与训练或调参 |
| 仿真对话（自建） | 800 条 | 微调 MOSS；由官方 **dev 切片重组**而成 |

SegLST 记录格式：

```json
{"session_id": "001", "start_time": 0.0, "end_time": 7.88, "speaker": "spk1", "words": "你 好 请 问"}
```

---

## Solution Overview

**双分支 + 规则路由 + 声纹校验的说话人拆分。**

端到端 SAT 模型（MOSS）文本强但系统性少估说话人数；「ASR + 聚类」（fun-asr + CAM++）
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
                                             submission_v094b.json
```

最后一步是本方案的关键增量：CAM++ 认为人数更多时，把跨簇的 MOSS 说话人按簇拆开，
**但只有两半的段级声纹余弦 < 0.50 才真的拆**。加上这道校验后 dev 收益由 −0.648 提升到 −0.982，
且通过全部四道验收关卡（推导见 [project.md](project.md) 2.6）。

---

## Reproduce

```bash
bash run_full_pipeline.sh                 # 复用已归档产物，跑完可本地执行的阶段
bash run_full_pipeline.sh --rerun-slow    # 额外重跑耗时阶段（Step 1、Step 5）
bash run_full_pipeline.sh --check         # 只做产物齐备性与谱系校验，不执行
```

脚本按下表顺序执行；需 GPU 或云端服务的阶段若产物已在仓库中则直接复用并标注 `[复用]`，
缺失时打印该阶段的完整命令并退出。各阶段做什么、各环节的线上实测增益、以及谱系证据见 [project.md](project.md) 3.2~3.4。

| Step | 做什么 | 脚本 | 耗时 | 输出 |
|---|---|---|---|---|
| 1 | CAM++ 谱聚类分离，同时缓存段级声纹（192 维）| `diarize.py` | ~7 min | `diar_test_m3_7_0.70` + `emb_test` |
| 2 | MOSS-SAT 解码：一次前向给出文本、时间戳与初始说话人 | `moss_sat.py` | ~2 h | `test_raw` |
| 3 | FireRedASR2 按 MOSS 的切分重新转写同一段音频 | `fr_retext.py` | ~1 h | `output_test_fr` |
| 4 | 词级声学仲裁：按差异块枚举 MOSS/FireRed 混合，声学似然选最优 | `word_arb.py` | ~1 h | `test_wa_m05b6` |
| 5 | 归属改写：段内多窗声纹一致性，留一法扣掉本段自己的窗 | `spk_reassign.py` | ~25 min | `test_v048_re` |
| 6 | 短段归属改写：用 MOSS 说话人似然，只改词数 ≤5 的段 → **MOSS 分支完成** | `spk_short_relabel.py` | ~10 s | `test_v051_moss` |
| 7 | fun-asr 转写（开 diarization，原生给出说话人）| `run.py` | — | `test_cam_multi` |
| 8 | 用 CAM++ 声纹为 fun-asr 的分段重打说话人标签 | `spk_relabel.py` | ~20 min | `test_cam_re` |
| 9 | CAM 分支的词级声学仲裁（配方同 Step 4）→ **CAM 分支完成** | `word_arb.py` | ~40 min | `test_cam_wa` |
| 10 | 逐场路由：按说话人数与轮换率在两分支间选一个（83/394 场走 CAM）| `pick_ensemble.py` | ~5 s | 路由后的 SegLST |
| 11 | 字符规范化：把 ref 词表中出现 0 次的字符改写成 ref 的写法 | `normalize_output.py` | ~3 s | 规范化后的 SegLST |
| 12 | 合并为单个 SegLST → **submission_v065.json**（线上 0.14354）| `merge_submit.py` | ~2 s | `submission_v065.json` |
| 13 | CAM 引导 + 声纹校验的说话人拆分 → **最终文件**（线上 0.14309）| `cam_split_verify.py` | ~15 s | `submission_v094b.json` |

Step 1~9 的产物均已在仓库中归档，因此默认从 Step 10 起真正执行；
`--rerun-slow` 会额外重跑 Step 1 与 Step 5。

---

## Results

线上 46 次提交，**17 次刷新纪录**；累计 0.19467 → 0.14309（**−0.05158，相对提升 26.5%**）。

| # | 日期 | 版本 | 线上 | Δ | 关键改动 |
|---|---|---|---|---|---|
| 1 | 07-26 | v002 | 0.19467 | — | 本地 CAM++ 重做说话人分离（min=3 / max=7 / merge_thr=0.70）|
| 2 | 07-29 | **v007** | **0.16593** | **−0.02874** | **换范式：MOSS 端到端 SAT 0.9B（全程最大单次跃迁）** |
| 3 | 08-01 | v011 | 0.16379 | −0.00214 | 双分支按场路由：MOSS 人数≤2 或 轮换率≤0.10，且 CAM 人数≥3 → 改用 CAM |
| 4 | 08-03 | v017b | 0.16377 | −0.00002 | 路由加规则 D：CAM 人数≥6（多估）→ 换回 MOSS |
| 5 | 08-04 | v018 | 0.1634 | −0.00037 | 路由加规则 H：CAM==5 且 MOSS≥4（轻度少判）→ 换 CAM |
| 6 | 08-08 | v024 | 0.15903 | −0.00437 | 引入 FireRed 声学仲裁：**段级**二选一，teacher-forcing 似然打分 |
| 7 | 08-08 | v026 | 0.15756 | −0.00147 | 仲裁改**词级**粒度：段内按差异块切开，枚举 2⁴ 种混合 |
| 8 | 08-09 | v028 | 0.15573 | −0.00183 | 输出层规范化：把 ref 词表中出现 0 次的字符改写成 ref 的写法 |
| 9 | 08-11 | v031 | 0.15096 | −0.00477 | 仿真拼接微调 MOSS（dev 重组 800 条，2 epoch，lr=1e-5）|
| 10 | 08-13 | **v037** | **0.14857** | **−0.00239** | **基座插值 soup + 归属轴 CAM 源重分配（首次破榜首）** |
| 11 | 08-14 | v038 | 0.14793 | −0.00064 | 归属轴双分支改写：MOSS 分支加一致性门控 `spk_reassign`（0.55 / 0.05）|
| 12 | 08-17 | v040 | 0.14683 | −0.00110 | 全 106 场作仿真素材重训（源说话人 318 → 410，+29%）|
| 13 | 08-21 | v048 | 0.14625 | −0.00058 | word_arb 参数：`MARGIN` 0 → 0.05、`MAX_BLOCKS` 4 → 6 |
| 14 | 08-25 | v051 | 0.14507 | −0.00118 | CAM 分支**首次**做声学仲裁 + 短段归属改写（两机制可加）|
| 15 | 08-27 | v063 | 0.14443 | −0.00064 | 短段改写 1.5/3 + 路由 0.05 + 定向拆分（三层叠加）|
| 16 | 08-28 | **v065** | **0.14354** | **−0.00089** | **路由 `turn_rate` 0.08 → 0.05（单变量）** |
| 17 | 08-31 | **v094b** | **0.14309** | **−0.00045** | **CAM 引导 + 声纹校验的说话人拆分（`sim<0.50`）** |

完整台账见 [`online_ledger.md`](online_ledger.md)，分数事实表见 [`SCORES.md`](SCORES.md)。

---

## Repository Structure

```
.
├── README.md                本文件（部署与复现）
├── project.md               方案归档与复盘（演进过程 + 逐阶段明细 + 谱系证据）
├── run_full_pipeline.sh     一键跑通全链路（含谱系校验与 SHA256 校验）
├── baseline/
│   ├── src/                       源码（106 个文件，含探针与已否定方向）
│   └── src_gpu/                   仅存于 GPU 机的脚本（打包时取回）
├── logs/YYYY-MM-DD.md             逐日实验记录
├── wiki/insights/                 31 条跨实验硬结论
└── online_ledger.md               线上提交台账
```

最终链路只用到 12 个脚本：`moss_sat.py` · `fr_retext.py` · `word_arb.py` · `spk_reassign.py` ·
`spk_short_relabel.py` · `run.py` · `spk_relabel.py` · `diarize.py` · `pick_ensemble.py` ·
`normalize_output.py` · `merge_submit.py` · `cam_split_verify.py`。
`src/` 下其余文件是探索过程中的探针与已否定方向。

---

## Models & Licenses

| 模型 | 来源 | 许可 | 用途 |
|---|---|---|---|
| MOSS-Transcribe-Diarize 0.9B | `OpenMOSS-Team/MOSS-Transcribe-Diarize`（HuggingFace）| Apache-2.0 | 端到端说话人归属转写 |
| CAM++ 说话人分离 | `iic/speech_campplus_speaker-diarization_common`（ModelScope）| 开源 | 分离 + 段级声纹 |
| FireRedASR2-AED | FireRedTeam | 开源 | 词级仲裁的声学裁判 |
| fun-asr | 阿里 FunASR（ModelScope）| 开源 | CAM 分支的转写文本 |

四个模型全部为可公开获取的开源模型，且全部运行在本队自有算力上。
