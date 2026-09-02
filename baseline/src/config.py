"""配置：从 .env 读取凭据与端点。云端 fun-asr API 路径（已合规确认）。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# 阿里云百炼 MaaS（专属实例，已确认 fun-asr 开源模型合规）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://llm-bg1a3oej9mdod2s0.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)
# 专属实例的 API 根 host（去掉 /compatible-mode/v1 后缀）
MAAS_HOST = DASHSCOPE_BASE_URL.replace("/compatible-mode/v1", "")

# 云端 ASR 模型与端点（已实测可达）
ASR_MODEL = "fun-asr"                      # 异步，支持 diarization，≤12h 音频
ASR_FLASH_MODEL = "fun-asr-flash-2026-06-15"  # 同步，≤5min 音频
TRANSCRIPTION_ENDPOINT = f"{MAAS_HOST}/api/v1/services/audio/asr/transcription"
MULTIMODAL_ENDPOINT = f"{MAAS_HOST}/api/v1/services/aigc/multimodal-generation/generation"
UPLOAD_POLICY_ENDPOINT = f"{MAAS_HOST}/api/v1/uploads"
TASK_QUERY_ENDPOINT = f"{MAAS_HOST}/api/v1/tasks"

# 说话人分离参数
DIARIZATION_ENABLED = True
DEFAULT_SPEAKER_COUNT = 2     # 赛题多为 2-3 人对话；未知时传 None 自动判
LANGUAGE_HINTS = ["zh", "en"]


@dataclass(frozen=True)
class Config:
    api_key: str = DASHSCOPE_API_KEY
    asr_model: str = ASR_MODEL
    asr_flash_model: str = ASR_FLASH_MODEL
    transcription_endpoint: str = TRANSCRIPTION_ENDPOINT
    multimodal_endpoint: str = MULTIMODAL_ENDPOINT
    upload_endpoint: str = UPLOAD_POLICY_ENDPOINT
    task_query_endpoint: str = TASK_QUERY_ENDPOINT
    diarization: bool = DIARIZATION_ENABLED
    language_hints: tuple[str, ...] = tuple(LANGUAGE_HINTS)
    audio_dir: Path = BASE_DIR / "audio"
    output_dir: Path = BASE_DIR / "output"


CFG = Config()
