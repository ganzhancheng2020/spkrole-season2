# 2026-07-28 云端 ASR 调研：公共百烈有 paraformer-v2/sensevoice 但当前用不了

## 调研结果

用户问"线上除了 fun-asr 还有没有其他 api 可用"。调研发现：

### 公共百烈（dashscope.aliyuncs.com）有这些 ASR 模型
| 模型 | task 接受 | 实际能跑 | 备注 |
|---|---|---|---|
| `paraformer-v2` | ✓ PENDING | ✗ FAILED（SERVER_ERROR / DownloadFailed）| 2024 新版 paraformer |
| `paraformer-v1` | ✓ PENDING | ✗ FAILED | 旧版 |
| `sensevoice-v1` | ✓ PENDING | ✗ DownloadFailed | 多语种 ASR |
| `paraformer-8k-v2` | ✓ PENDING | 未测（8k 不适合本赛 16k）| — |
| `paraformer-16k-v2` | — | ✗ Model not exist | — |

### 专属实例 vs 公共百烈不通
- 当前 `.env` 用 MaaS 专属实例（MAAS_HOST），只部署了 `fun-asr`。
- 专属实例跑 paraformer-v2/sensevoice-v1/paraformer-v1 全 FAILED（实例没部署这些模型）。
- 专属实例的 oss url（`oss://dashscope-instant/...`）公共百烈 ASR 读不了（`InvalidFile.DownloadFailed`）——同 bucket 名但不同账号/权限。
- 公共百烈 upload policy 上传成功（HTTP 200），但 ASR 仍 DownloadFailed（可能 bucket ACL 或路径权限问题）。
- 公共百烈 ASR 要**公网 https 可访问 url**，不收 oss url 也不收 base64 audio（同步端点报 `url error`）。

### fun-asr-flash
- `fun-asr-flash-2026-06-15`（config.py 已配但没用过）专属实例也报 `url error`——它在公共百烈，不在专属实例。

## 要试 paraformer-v2/sensevoice-v1 需要什么

公共百烈 ASR 要公网 https url。两条路：
1. **配阿里云 OSS 公网可读桶**：把 wav 上传成 https url 喂公共百烈 ASR。
2. **开通百烈数据集托管 API**：不同于 oss upload，专门给 ASR 喂数据。

这是 D4 文本端唯一可能有突破的路：
- `paraformer-v2` 是 2024 新模型，可能比 fun-asr 文本质量好（之前本地 paraformer-zh 证伪是因为那是旧版 seaco_paraformer，云端 paraformer-v2 是更新版不同）。
- `sensevoice-v1` 是多语种 ASR，FunASR 团队出品，可能中文质量好。
- 如果云端 paraformer-v2/sensevoice 文本错误比 fun-asr 少，能降 del（2473 是最大块），可破文本天花板。

## 状态

- 需用户配公网 OSS 桶或开通百烈数据集托管。
- 主线维持 v002=22.79%（线上 19.47%），fun-asr 是当前能用的唯一 ASR。
