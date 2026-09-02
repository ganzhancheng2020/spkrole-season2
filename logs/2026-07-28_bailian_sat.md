# 2026-07-28 百烈 SAT 模型调研

## 调研结果

用户"开干，这个模型你先看看百炼有没有"。查百烈 MOSS-Transcribe-Diarize + 类似端到端 SAT 模型。

### 百烈可用模型（已确认 code 空或仅配额限制）
| 模型 | 端点 | 音频格式 | 状态 |
|---|---|---|---|
| `qwen-audio-turbo` | multimodal-generation | `content=[{'audio':oss_url}]` | ✓ 格式正确，仅配额超限 |
| `qwen2-audio-instruct` | multimodal-generation | 同上 | ✓ 格式正确，仅配额超限 |
| `qwen-omni-turbo` | multimodal-generation | 同上 | code 空（被接受）|
| `qwen-omni-turbo-latest` | multimodal-generation | — | code 空 |
| `qwen2.5-omni-7b` | multimodal-generation | — | code 空 |

### 不在百烈
- `MOSS-Transcribe-Diarize` / `moss-transcribe-diarize` / `OpenMOSS/MOSS-Transcribe-Diarize` → Model not exist（公共和专属都无）
- `TagSpeech` → 无
- `qwen-audio` / `qwen2-audio` → model_not_found
- `qwen3-omni-30b` / `qwen3-omni` → Model not exist（TagSpeech 论文的基线之一，百烈没有）

### OpenAI 兼容接口限制
- `qwen-audio-turbo` / `qwen2-audio-instruct` 在 OpenAI 兼容接口下报"Unsupported model for OpenAI compatibility"——必须走百烈原生 multimodal-generation 端点。
- OpenAI 兼容接口 content 只支持 `text`/`image_url`/`video_url`，**不支持音频**。
- 百烈原生端点 content 支持 `{'audio': oss_url}` 格式（已验证）。

## 含义

**百烈有音频 LLM 但没有专门的端到端 SAT 模型：**
- `qwen-audio-turbo` 是音频理解 LLM，能听音频回答问题，但**不专门训练说话人分离**。
- `qwen-omni-turbo` 是多模态 LLM（文本/图/视频/音频），同样不专门做 SAT。
- MOSS-Transcribe-Diarize（专门训练端到端 ASR+diar）**不在百烈**，要本地跑。

### 两条路
1. **百烈调 qwen-audio-turbo**（云端，不用本地跑）：
   - 配额要付费（免费额度已超）。
   - 用 prompt 指示它"输出 [说话人] 文本 逐句"，但它不是专门训练的，可能效果差（和之前 LLM 后处理证伪一样，LLM 没有专门训练说话人分离能力）。
   - 但值得一试——直接喂音频比喂 ASR 文本让它纠错强（它能听到原始音频）。
2. **本地跑 MOSS-Transcribe-Diarize 0.9B**（Apache-2.0，deep-research 首选）：
   - clone OpenMOSS/MOSS-Transcribe-Diarize + 下 0.9B 权重。
   - M4 兼容性未验证（vLLM 是 CUDA 专用，但 0.9B 小可能 CPU/MPS 能跑）。
   - 专门训练端到端 SAT，理论效果比 qwen-audio-turbo prompt 好。

## 下一步（落 wiki/log.md select）

1. 先试百烈 qwen-audio-turbo（需用户确认付费配额）——快，云端，不用本地装。
2. 若不行 → 本地 MOSS clone + M4 烟测。
3. 主线维持 v002=22.79%（线上 19.47%）。
