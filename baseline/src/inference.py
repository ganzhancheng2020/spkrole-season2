"""云端 fun-asr 推理：上传 → 异步转写（带说话人分离）→ 轮询 → 解析。

返回结构（每句）：
    {"speaker_id": int|None, "begin_time": int(ms), "end_time": int(ms),
     "text": str, "words": [{"begin_time","end_time","text","speaker_id"}]}
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from config import CFG
from upload import upload_audio

logger = logging.getLogger(__name__)


def _submit_transcription(oss_url: str, speaker_count: int | None) -> str:
    """提交异步转写任务，返回 task_id。"""
    params: dict = {
        "channel_id": [0],
        "language_hints": list(CFG.language_hints),
        "diarization_enabled": CFG.diarization,
    }
    if speaker_count is not None:
        params["speaker_count"] = speaker_count
    payload = {
        "model": CFG.asr_model,
        "input": {"file_urls": [oss_url]},
        "parameters": params,
    }
    resp = requests.post(
        CFG.transcription_endpoint,
        json=payload,
        headers={
            "Authorization": f"Bearer {CFG.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
            "X-DashScope-OssResourceResolve": "enable",
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    task_id = data.get("output", {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"提交转写任务失败: {data}")
    logger.info("已提交转写任务 task_id=%s", task_id)
    return task_id


def _query_task(task_id: str) -> dict:
    """查询任务状态。"""
    resp = requests.get(
        f"{CFG.task_query_endpoint}/{task_id}",
        headers={"Authorization": f"Bearer {CFG.api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _download_transcription(transcription_url: str) -> dict:
    """下载结果 JSON。transcription_url 24h 有效。"""
    resp = requests.get(transcription_url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _extract_sentences(result_json: dict) -> list[dict]:
    """从结果 JSON 提取句子列表，兼容多种返回结构。"""
    if "transcripts" in result_json:
        for t in result_json["transcripts"]:
            if "sentences" in t:
                return t["sentences"]
        return result_json["transcripts"]
    if "sentences" in result_json:
        return result_json["sentences"]
    if "sentence" in result_json:
        s = result_json["sentence"]
        return s if isinstance(s, list) else [s]
    logger.warning("未知结果结构，keys=%s", list(result_json.keys()))
    return []


def transcribe_audio(
    file_path: Path,
    speaker_count: int | None = None,
    poll_interval: float = 3.0,
    timeout: float = 600.0,
) -> list[dict]:
    """端到端：上传 → 转写 → 轮询 → 返回句子列表。

    Args:
        file_path: 本地音频路径
        speaker_count: 说话人数提示；None 自动判
        poll_interval: 轮询间隔秒
        timeout: 最长等待秒
    """
    oss_url = upload_audio(file_path, model=CFG.asr_model)
    task_id = _submit_transcription(oss_url, speaker_count)

    deadline = time.time() + timeout
    while time.time() < deadline:
        info = _query_task(task_id)
        status = info.get("output", {}).get("task_status", "")
        if status == "SUCCEEDED":
            results = info["output"].get("results", [])
            if not results:
                return []
            t_url = results[0].get("transcription_url")
            if not t_url:
                return []
            rj = _download_transcription(t_url)
            return _extract_sentences(rj)
        if status == "FAILED":
            raise RuntimeError(f"转写失败: {info}")
        logger.info("任务 %s 状态=%s，等待...", task_id, status)
        time.sleep(poll_interval)
    raise TimeoutError(f"任务 {task_id} 超时")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = Path(sys.argv[1])
    sents = transcribe_audio(p)
    for s in sents:
        print(s.get("speaker_id"), s.get("begin_time"), s.get("end_time"), s.get("text"))
