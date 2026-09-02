# fun-asr 第三源词级仲裁集成设计（v029 方向）

> 日期：2026-08-09
> 项目根：`/Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2`

## Context

### 项目当前状态（按磁盘核对，2026-08-09）

- **当前最优** v026 线上 **0.15756**，距榜首 0.15023 差 **0.00733 点**
- 当前链路：MOSS 0.9B（说话人+文本） + FireRedASR2-AED（文本，词级时间戳）→ word_arb 词级仲裁 → pick_ensemble H/D → normalize_output
- dev 纯文本 tcpWER（抹说话人）= **11.64%**（v026 词级仲裁后）
- 文本轴·词级剩余剔top3 空间 **0.602 点**，已捕获 **46.7%**

### 项目方法论（CLAUDE.md + HANDOFF §5）

- 提交前两道关卡：双切分（train82/holdout24）+ 重尾（剔 top-3 后净收益）
- 一律按 v0NN 命名 + dev/test 双产物
- 校准规则（v008 后）：线上收益 ≈ 剔 top-3 后 dev 净收益 × 1.0～1.7
- 提交必留人工，提交前 `shasum -a 256` 比对上一版

### 限制与上下文

- 每日最多 **3 次**线上提交，作品截止 **2026-09-01 17:00**
- 已封死方向：换说话人源（pyannote/DiariZen/Sortformer/SoulX/CAM++ 全塌）；MOSS 参数调优（5 条假设全否）；裁判偏置修（FireRed 自己打 FireRed 文本）；n-best 扩候选（段级失败）；MOSS 微调（v012 倒退）
- 当前三条轴状态：
  - 文本轴·词级（2 候选内）：**0.602 空间**，活
  - 择优轴：**1.846 空间**，五类信号全否
  - 归属轴：**2.175 空间**，结构性重尾封死

### 之前 brainstorming 的探索结论（2026-08-09）

- 方案 A（Whisper-medium 第三源）：门槛失败，中文 CER 9.86%，远不如 FireRed，3 段探针证伪
- ASR 榜单调研：FireRedASR2-LLM（8.3B）是开源中文 SOTA（CER 2.89%）但参数大、本地推理慢
- 用户选 fun-asr 第三源而非 FireRed2-LLM（优先确定性而非新收益）

---

## 设计目标

**把 fun-asr 作为第三文本源加入 word_arb 做三候选词级仲裁**。目标：捕获剩余文本轴·词级空间 0.602 中的至少 30%，让 v026 线上 0.15756 进一步下降。

## 架构

```
output/v007_moss/(106) ────────┐
                               ├─ src/word_arb.py (扩展 3 候选) ─→ output/wa_funasr_dev/(106)
output/fr_retext/(106) ────────┤   MAX_BLOCKS=4 MARGIN=0.0
output/funasr_retext/(106) ─────┘   依赖 /tmp/FireRedASR2S + /tmp/FireRedASR2-AED
                                   + fun-asr 词级 ts.json
        ↓
src/pick_ensemble.py --moss output/wa_funasr_dev --cam output/hyp_sw_m3_7_0.70
        ↓ output/v029_pick/(106)
src/merge_submit.py → submission_v029.json
```

test 侧对应链：`output_test_moss/(394)` + `output_test_fr3/(394)` + `output_test/funasr_retext/(394)` → `output_test/wa_funasr/(394)` → `output_test_v029/(394)` → `submission_v029.json`

## 组件

### 1. fun-asr 重跑 dev/test 推理 + 词级时间戳落盘

**新增文件**：`baseline/src/funasr_retext.py`

参考 `baseline/src/fr_retext.py` 的实现（FireRed 词级时间戳重排到 MOSS 段）：

```python
# 核心逻辑
# 1. 调用 fun-asr 云端（run.py:103-118 transcribe_audio）
# 2. _extract_sentences 解析（已支持 words[begin_time,end_time,text,speaker_id]）
# 3. 按 begin_time 把每个 word 分配到 MOSS 段
# 4. 落盘 {sid}.ts.json（词级时间戳）和 {sid}.seglst.json（段结构）
```

**关键参数**：
- 沿用 `run.py:103` 的 fun-asr 调用，`speaker_count=None`（自动判）
- `diarization_enabled=True`（与 v000 一致）
- 输出格式与 FireRed 词级 ts.json 对齐：`[[token, start_sec, end_sec], ...]`

**成本**：dev 106 段 × ~30s/段 ≈ 1 小时云端；test 394 段 × ~30s/段 ≈ 3-4 小时

### 2. word_arb 扩展为 3 候选

**修改文件**：`baseline/src/word_arb.py`

- 增加 `--third_dir` 参数（fun-asr 重排产物）
- 对每个差异块，枚举 **3^k 种混合**（MOSS/FireRed/fun-asr 三选一）
- 关键约束：MAX_BLOCKS=4 → 3^4=81 种混合（比 2^4=16 复杂度 5x）

**枚举策略选择**：方案 C（两轮仲裁，避免组合爆炸）

```python
# 方案 C 实现（替代直接 3^k 枚举）：
# 第 1 轮：MOSS vs FireRed，word_arb 现有逻辑选 best-2
# 第 2 轮：best-2 vs fun-asr，仅在 fun-asr 更好的块上替换
# 这样总打分次数 = 16 + 4 = 20 次（vs 直接 81 次），复杂度可控
```

**判据**：每段仍用 FireRedASR2-AED teacher forcing 声学似然 `P(文本|音频)`，最高分胜出，MARGIN=0.0

### 3. pick_ensemble 沿用 v018 H/D

新 wa_funasr_dev 已包含 fun-asr 仲裁文本 → 现有 `pick_ensemble.py:103-164` 的 H+D 规则继续用，**无需改动**。

### 4. normalize_output 沿用 v028

`baseline/src/normalize_output.py:50-59` 的 CHAR_MAP 继续用，**无需改动**。

### 5. 提交前两道关卡

- **双切分**：`prep_finetune_data.py` 的 stratified_split（seed=0, 25 sessions）切 train82/holdout24，两边都改善才 PASS
- **重尾**：`heavy_tail_gate.py` 剔 top-3 后净收益 > 0 才 PASS

## 关键文件

| 路径 | 用途 | 改动 |
|---|---|---|
| `baseline/src/run.py:103` `transcribe_audio` | fun-asr 云端调用 | 不改（复用） |
| `baseline/src/inference.py:73` `_extract_sentences` | 解析 fun-asr 返回 | 不改（已支持 words[]） |
| **新建** `baseline/src/funasr_retext.py` | fun-asr 词级时间戳重排到 MOSS 段 | 新写 |
| `baseline/src/word_arb.py:71-73` | 当前 MAX_BLOCKS=4 / MARGIN=0.0 | 改：加 `--third_dir` + 两轮仲裁 |
| `baseline/src/pick_ensemble.py:103-164` | MOSS/CAM H+D 规则 | 不改 |
| `baseline/src/normalize_output.py:50-59` | 输出层规范化 | 不改 |
| `baseline/src/heavy_tail_gate.py` | 提交前重尾关卡 | 不改（直接调用） |
| `baseline/src/batch_evaluate.py` | dev 全量评测 | 不改 |
| `baseline/src/merge_submit.py:17-25` | test 侧合并提交物 | 不改 |
| `baseline/output/v007_moss/(106)` | MOSS dev 预测 | 不改（已有） |
| `baseline/output/fr_retext/(106)` | FireRed dev 重排 | 不改（已有） |
| `baseline/output/fr_retext_dev_ts/(106)` | FireRed dev 词级时间戳 | 不改（已有） |
| `baseline/output_dev_funasr/(106)` | fun-asr dev 旧产物（无 ts.json） | 不改 |
| **新建** `baseline/output/funasr_retext/(106)` | fun-asr dev 重排+词级 ts.json | 新生成 |
| **新建** `baseline/output/wa_funasr_dev/(106)` | 三候选仲裁后 dev 产物 | 新生成 |
| **新建** `baseline/output/v029_pick/(106)` | 三路择优后 dev 产物 | 新生成 |
| **新建** `baseline/output_test/funasr_retext/(394)` | fun-asr test 重排+词级 ts.json | 新生成 |
| **新建** `baseline/output_test/wa_funasr/(394)` | 三候选仲裁后 test 产物 | 新生成 |
| **新建** `baseline/output_test_v029/(394)` | 三路择优后 test 产物 | 新生成 |
| **新建** `baseline/submission_v029.json` | v029 提交物 | 新生成 |

可复用函数（不要重写）：

- `baseline/src/run.py:103` `transcribe_audio`：fun-asr 云端调用
- `baseline/src/inference.py:73` `_extract_sentences`：解析（已含 words[begin_time,end_time]）
- `baseline/src/fr_retext.py:20-49`：FireRed 词级时间戳重排逻辑，fun-asr_retext 模仿
- `baseline/src/word_arb.py:84-134` `blocks()`：差异块切分
- `baseline/src/word_arb.py:148-162` `score()`：FireRed teacher forcing 声学似然
- `baseline/src/pick_ensemble.py:103-164`：MOSS/CAM H+D 规则
- `baseline/src/normalize_output.py:50-59`：CHAR_MAP
- `baseline/src/heavy_tail_gate.py`：重尾关卡
- `baseline/src/merge_submit.py:17-25`：合并

## 实现步骤

1. **新建 `funasr_retext.py`**：模仿 `fr_retext.py`，调用 `run.transcribe_audio`，按词级 begin_time 重排到 MOSS 段，落盘 ts.json + seglst.json
2. **跑 dev**：funasr_retext.py --wav dev → output/funasr_retext/（~1 小时）
3. **跑 test**：funasr_retext.py --wav test → output_test/funasr_retext/（~3-4 小时）
4. **改 `word_arb.py`**：加 `--third_dir` 参数，实现两轮仲裁（先 MOSS vs FireRed 选 best-2，再 best-2 vs fun-asr）
5. **跑 dev word_arb**：word_arb.py --moss wa_m0 --fr fr_retext --third funasr_retext --wav dev → output/wa_funasr_dev/
6. **dev 评测**：batch_evaluate.py --dir output/wa_funasr_dev → 看相对 v026 16.090% 改善多少
7. **若 dev 改善**：跑 pick_ensemble.py --moss wa_funasr_dev --cam hyp_sw_m3_7_0.70 --out v029_pick
8. **normalize_output**：normalize_output.py v029_pick → v029_norm
9. **重尾关卡**：heavy_tail_gate.py v026_pick v029_norm → 期望 PASS（剔 top-3 后净收益 > 0）
10. **若 PASS**：跑 test word_arb + pick_ensemble + normalize + merge → submission_v029.json
11. **提交前硬校验**：`shasum -a 256 submission_v029.json submit/archive/submission_v028_20260809.json` 必异
12. **人工提交**（agent 不擅自提交）：dev 改善 + 双切分 + 重尾 PASS + shasum 异 → 通知人工提交

## 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| fun-asr 纯文本略差（12.92% vs FireRed 12.71%），三候选仲裁可能没改善 | 中 | oracle 三源上限 13.67%（vs 二源 14.01%）仍下降，说明有 0.34 点可捕获 |
| 3 候选枚举复杂度暴涨 | 中 | 方案 C（两轮仲裁）保证总打分次数 = 16+4=20（与 2^4=16 同量级） |
| fun-asr 词级时间戳 API 不稳定 | 低 | inference.py:73 已支持 words[] 结构，fun-asr 官方文档确认支持 |
| 重排到 MOSS 段边界不对齐 | 低 | fr_retext.py 已验证此机制可用，fun-asr_retext 用相同逻辑 |
| 提交后线上仍倒退（与 v027 类似）| 中 | 两道关卡（双切分 + 重尾）必须 PASS 才提交，PASS 后仍有 ~13% 倒退概率（HANDOFF 校准） |

## 验证流程

### dev 端验证

```bash
cd baseline

# 步骤 1：fun-asr 重跑 dev 推理 + 词级 ts
PYTHONPATH=src .venv/bin/python src/funasr_retext.py \
  ../data/extracted/dev/dev/wav ../baseline/output/v007_moss \
  ../baseline/output/funasr_retext

# 步骤 2：扩展 word_arb 跑 dev 三候选仲裁
PYTHONPATH=src diarizen_venv/bin/python src/word_arb.py \
  ../baseline/output/v007_moss ../baseline/output/fr_retext \
  ../baseline/output/funasr_retext ../data/extracted/dev/dev/wav \
  ../baseline/output/wa_funasr_dev

# 步骤 3：dev 全量评测
PYTHONPATH=src .venv/bin/python src/batch_evaluate.py --dir ../baseline/output/wa_funasr_dev

# 预期：dev tcpWER 较 v026 16.090% 改善 ≥ 0.3 点（即 ≤ 15.79%）
```

### 双切分 + 重尾关卡

```bash
# 双切分（沿用 prep_finetune_data.py 的 stratified_split）
PYTHONPATH=src .venv/bin/python src/batch_evaluate.py --dir ../baseline/output/wa_funasr_dev/train82
PYTHONPATH=src .venv/bin/python src/batch_evaluate.py --dir ../baseline/output/wa_funasr_dev/holdout24

# 重尾关卡
PYTHONPATH=src .venv/bin/python src/heavy_tail_gate.py ../baseline/output/v026_pick ../baseline/output/wa_funasr_dev

# 两道 PASS 才进 test
```

### test 端提交物生成

```bash
# 步骤 4：fun-asr 重跑 test 推理 + 词级 ts
PYTHONPATH=src .venv/bin/python src/funasr_retext.py \
  ../data/extracted/test/test/wav ../baseline/output_test_moss \
  ../baseline/output_test/funasr_retext

# 步骤 5：扩展 word_arb 跑 test 三候选仲裁
PYTHONPATH=src diarizen_venv/bin/python src/word_arb.py \
  ../baseline/output_test_moss ../baseline/output_test_fr3 \
  ../baseline/output_test/funasr_retext ../data/extracted/test/test/wav \
  ../baseline/output_test/wa_funasr

# 步骤 6：择优 + 规范化 + 合并
PYTHONPATH=src .venv/bin/python src/pick_ensemble.py \
  --moss ../baseline/output_test/wa_funasr --cam ../baseline/output/hyp_test_v002 \
  --out ../baseline/output_test_v029
PYTHONPATH=src .venv/bin/python src/normalize_output.py \
  ../baseline/output_test_v029 ../baseline/output_test_v029_norm
PYTHONPATH=src .venv/bin/python src/merge_submit.py \
  --dir ../baseline/output_test_v029_norm --out ../baseline/submission_v029.json

# 步骤 7：提交前硬校验
shasum -a 256 ../baseline/submission_v029.json ../submit/archive/submission_v028_20260809.json
# 必异（不同 sha256）
```

### 验收标准

| 指标 | 必须 |
|---|---|
| dev tcpWER（wa_funasr_dev）| ≤ v026 16.090% − 0.3 点 |
| train82 vs holdout24 同向改善 | 是（双切分 PASS） |
| 剔 top-3 后净收益 | > 0（重尾 PASS） |
| submission_v029.json shasum | 与 v028 不同 |
| dev 反向损失 | 无任何 session 变差（v028 经验） |

### 反向损失检测

```bash
# v028 经验：反向损失 = 0 是上线 PASS 的关键（v027 失败原因）
PYTHONPATH=src .venv/bin/python -c "
import json, glob
v026 = {r['session_id']: r for f in glob.glob('../baseline/output/v026_pick/*.seglst.json') for r in json.load(open(f))}
v029 = {r['session_id']: r for f in glob.glob('../baseline/output/wa_funasr_dev/*.seglst.json') for r in json.load(open(f))}
# 逐 session 对比：v029 任何 session 比 v026 差 = 反向损失
# 若反向损失 > 5 个 session，FAIL
"
```

## 提交决策

按 CLAUDE.md §停止边界：dev 显著优于 v026 + 双切分 PASS + 重尾 PASS + shasum 异 → **通知人工提交**（agent 不擅自提交线上）。