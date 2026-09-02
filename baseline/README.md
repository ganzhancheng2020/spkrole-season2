# SpkRole-Cup Baseline v0（云端 fun-asr API）

针对讯飞"说话人角色分离挑战赛 赛季2"，用阿里云百炼托管的 **fun-asr**（开源模型，已合规确认）做端到端转写，输出 SegLST JSON。

## 架构

```
本地 wav → 上传凭证接口拿 oss:// URL
        → fun-asr 异步转写（diarization_enabled=true）
        → 轮询 task 状态
        → 下载 transcription_url（含 sentence/speaker_id/begin_time/end_time/words）
        → 映射成 SegLST JSON
        → meeteval tcpwer --collar 5 评分（用根目录 .venv）
```

## 文件结构

```
baseline/
├── .env                 # API key/base_url（已 gitignore）
├── .venv/               # Python 环境（requests/openai/python-dotenv）
├── requirements.txt
├── audio/               # 放入待转写 wav（16k/16bit/mono）
├── output/              # 生成的 SegLST JSON
└── src/
    ├── config.py         # 配置 + 端点常量
    ├── upload.py         # 本地音频 → oss:// 临时 URL
    ├── inference.py      # 提交转写 + 轮询 + 下载
    ├── seglst_converter.py  # fun-asr 句子 → SegLST
    ├── evaluate.py      # 调根目录 meeteval 跑 tcpWER
    └── run.py            # 端到端编排
```

## 安装

```bash
cd baseline
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 运行

```bash
# 1. 放入音频到 audio/
# 2. 跑转写
PYTHONPATH=src .venv/bin/python src/run.py audio/your.wav
#    批量：PYTHONPATH=src .venv/bin/python src/run.py   # 跑 audio/ 下所有 wav
#    指定说话人数：加 --speakers 2

# 3. 评分（需根目录 .venv 已装 meeteval）
PYTHONPATH=src .venv/bin/python src/evaluate.py \
    --ref 路径/ref.seglst.json \
    --hyp output/your.seglst.json
```

## 已验证（2026-07-23 烟测）

- ✅ 上传凭证接口可达，本地音频可上传拿 `oss://` URL
- ✅ fun-asr 异步转写全链路跑通（提交→轮询→下载）
- ✅ SegLST 输出格式对齐官方 `验证示例_meeteval/ref.seglst.json`
- ✅ ASR 文本识别正确（中文逐字空格、英文小写、时间戳秒级 2 位小数）
- ⚠️ **说话人分离在 TTS 合成音频上失败**（4 句全标 spk0）：macOS `say` 的 Tingting/Liangliang 声纹可能太接近，或 fun-asr diarization 对干净短音频区分不出说话人。**需用赛题真实音频复测**才能判断说话人分离质量。

## 已知限制

1. **overlap 处理弱**：fun-asr 的 diarization 是传统聚类范式，非 EEND-VC 原生处理 overlap。赛题含 overlap 音频，这是结构性弱点。
2. **上传接口限流 100 QPS**（阿里云主账号+模型维度），生产环境建议自有 OSS。
3. **transcription_url 24h 有效**，需及时落盘。

## 下一步

- 拿到赛题真实音频后复测说话人分离，确认 spk0/spk1 能否正确分出
- 若 fun-asr diarization 质量不足，考虑混搭：fun-asr 出文本 + DiariZen 出说话人标签
- 字级时间戳对齐（SegLST 只需句级，但 tcpWER 对边界敏感——见 references 论文）
