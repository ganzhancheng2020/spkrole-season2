---
slug: firered-llm-setup-traps
type: 工程记录
status: 已跑通
family: 文本轴 / 外部模型
date: 2026-08-22
related: [[word-arb-capture-headroom]] [[ts-asr-model-selection]]
---

# FireRedASR2-LLM 8.3B 接入的三个坑

跑通前连撞三个，且**每个的表征都不像真正的病因**，记下来免得重踩。

## ① `llm_length_penalty=0.0`（默认值）

官方示例 `examples_infer/asr/inference_asr_llm.sh` 用的是：

```
--beam_size 3 --repetition_penalty 3.0 --llm_length_penalty 1.0 --temperature 1.0
```

而 `FireRedAsr2Config` 的默认值是 `repetition_penalty=1.0, llm_length_penalty=0.0`。
长度惩罚为 0 时 beam search 没有产出长序列的动力 → 倾向立刻选 EOS。

⚠️ **但这不是本次「1 个字」的主因** —— 改对之后仍然只出 1 个字。
（差点据此误判，教训见 ②）

## ② 精度：`use_half` 是给 AED 设计的，与 LLM 加载器冲突

`fireredasr_llm.py` 的加载器注释写明 **"Training use torch.bfloat16"**，用 bf16 载入 Qwen2。
但 `asr.py` 在 `use_half=True` 时对**整个模型**调 `.half()`（fp16），**覆盖掉 bf16**
→ Qwen2 数值溢出 → **每段输出同一个字符 `'%'`**。

**表征极具误导性**：输出恒定且极短，看起来像「模型坏了 / 权重没加载 / 生成立刻停止」。
我据此先后怀疑过长度惩罚、LFS 指针文件、`load_state_dict(strict=False)` 静默丢权重 ——
**全都不是**。

排除权重问题的决定性证据：`model.pth.tar` 3.63 GB ≈ 894M 参数 × 4 字节，
与日志 `#param of FireRedAsrLlm is 894051936` **精确吻合**；
且逐文件比对源端大小，本地全部一致（`asr_encoder.pth.tar` 本来就只有 1472 字节）。

## ③ `use_half=False` 会 OOM

关掉 half 后编码器留在 fp32（2710 MB）+ bf16 Qwen2（15.2 G）+ 激活 → 24G 卡 **OOM**。

## 正解：统一 bf16

给 `asr.py` 加 `FR_BF16` 环境变量开关（**默认关闭，AED / word_arb 行为不变**），
置位时模型与特征统一转 `bfloat16` —— 与 fp32 同数值范围（不溢出），
编码器减半到 1.4 G，总占用约 18 G，24 G 可容。

跑通后：12 秒/段，输出正常中文转写（143–193 字/段）。
补丁前已 `cp asr.py asr.py.orig` 备份。

## 对预期的下调（重要）

仓库 README 自报 Mandarin-4 平均 CER：**LLM 2.89% vs AED 3.05%**，**仅差 0.16 点**。
所以「8.3B 比 1.1B 大 7.5 倍 ⇒ 明显更强」在**通用榜单上就不成立**，
何况本赛还有域墙（[[ts-asr-model-selection]]：任何论文数字对本赛不具预测力）。

**验收判据（事前定死）**：整场转写与 AED 的 **19.42%** 同口径比，
≥ 19.42% 即判死，不再投逐段转写。

## 🔴 结果：判死（106 段干净集）

| 口径 | tcpWER |
|---|---|
| MOSS 原始 | 17.74% |
| AED **真**时间戳 | **19.42%** |
| AED **伪**时间戳 | 24.84% |
| **LLM 8.3B 伪时间戳** | **25.85%** |

### 必须做的公平对比

LLM 分支不出词级时间戳，只能把整场文本按字数比例切到 MOSS 段边界（伪对齐）。
脚本 docstring 里事前写死了规则：**「LLM 更差 → 不能直接判死，需先区分
模型差与伪时间戳的锅」**。故把 AED 也做同样的伪对齐再比。

**伪对齐本身要付 5.42 点**（19.42 → 24.84）。所以初看的 6.43 点差距里，
**只有 1.01 点是模型的真实差距**。

**但方向不变**：同条件下 LLM 8.3B 仍比 AED 1.1B **差 1.01 点**，
且文本更少（18396 vs 18651 字）、del 更高（2253 vs 2075）。

### 结论

**FireRedASR2-LLM 8.3B 在本赛音频上劣于 AED 1.1B，方向封死。**
两种用法（当候选源 / 当裁判）**一并死** —— 整场转写是二者共同的前置门：
声学建模在本赛音频上就不占优，换到哪个位置都不会更好。

**外部模型在本赛战绩：0/7**（Qwen3-ASR、Qwen2.5-Omni、Qwen3-Omni、SoulX 30B、
DiCoW、TagSpeech 族、FireRedASR2-LLM）。
再次印证 [[ts-asr-model-selection]]：**任何论文数字对本赛不具预测力**
（README 自报 LLM/AED 仅差 0.16 点，实测反向差 1.01 点）。

### 可复用的量

**伪对齐（丢弃真时间戳、按字数比例切段）的代价 = 5.42 点。**
今后评估任何不出时间戳的 ASR 时，先扣掉这个数再比，别误判。
