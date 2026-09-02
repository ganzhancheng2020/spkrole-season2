"""音频上传：把本地音频推到百炼临时存储，拿 oss:// URL 喂给 fun-asr。

三步流程（官方文档 https://help.aliyun.com/zh/model-studio/get-temporary-file-url）：
1. GET /api/v1/uploads?action=getPolicy&model=... 拿上传凭证
2. POST 文件到 upload_host（multipart/form-data）
3. 拼接 oss:// + key 作为音频 URL
"""
from __future__ import annotations

import logging
from pathlib import Path

import requests

from config import CFG

logger = logging.getLogger(__name__)


def _get_upload_policy(model: str) -> dict:
    """步骤1：拿上传凭证。"""
    resp = requests.get(
        CFG.upload_endpoint,
        params={"action": "getPolicy", "model": model},
        headers={
            "Authorization": f"Bearer {CFG.api_key}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "data" not in data:
        raise RuntimeError(f"获取上传凭证失败: {data}")
    logger.info("已获取上传凭证，有效期 %s 秒", data["data"].get("expire_in_seconds"))
    return data["data"]


def _post_file_to_oss(policy: dict, file_path: Path, model: str) -> str:
    """步骤2：上传文件到 OSS，返回 oss:// key。"""
    key = f"{policy['upload_dir']}/{file_path.name}"
    form_fields = {
        "OSSAccessKeyId": policy["oss_access_key_id"],
        "policy": policy["policy"],
        "Signature": policy["signature"],
        "key": key,
        "x-oss-object-acl": policy["x_oss_object_acl"],
        "x-oss-forbid-overwrite": policy["x_oss_forbid_overwrite"],
        "success_action_status": "200",
    }
    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f, "application/octet-stream")}
        resp = requests.post(
            policy["upload_host"],
            data=form_fields,
            files=files,
            timeout=300,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"上传失败 HTTP {resp.status_code}: {resp.text[:300]}")
    oss_url = f"oss://{key}"
    logger.info("已上传 %s -> %s", file_path.name, oss_url)
    return oss_url


def upload_audio(file_path: Path, model: str = CFG.asr_model) -> str:
    """上传本地音频，返回 oss:// 临时 URL（48 小时有效）。"""
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    policy = _get_upload_policy(model)
    return _post_file_to_oss(policy, file_path, model)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = Path(sys.argv[1])
    print("oss_url:", upload_audio(p))
