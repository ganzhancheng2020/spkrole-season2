"""SoulX-Transcriber 输出 JSONL → SegLST（本机跑，可随时重算）。

分工：GPU 机只做不可复算的推理（`soulx_batch_infer.py`，按小时计费），
解析与格式转换留在本机 —— 解析规则改了不必重跑 GPU。

输入：每行 {"index": "001", "hyp": "<SoulX 原始文本>"}
SoulX 的 prompt 规定输出格式为（见官方 inference/infer.py 的 USER_QUERY）：
    [00:00:01.234 --> 00:00:03.456] Speaker 1: 文本内容
输出：<out>/001.seglst.json，与 ref.seglst.json 同构，可直接喂 batch_evaluate.py。

文本规范化直接复用 moss_sat.py 的 to_words（标点在 tcpWER 不计分，
ref 是无标点逐字），避免两套口径。
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from moss_sat import to_words

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# [00:00:01.234 --> 00:00:03.456] Speaker 1: 文本
# 说话人写法宽松匹配（Speaker 1 / Speaker1 / SPEAKER_1），模型偶尔飘格式
LINE_RE = re.compile(
    r"\[\s*(?P<start>[\d:.]+)\s*-+>\s*(?P<end>[\d:.]+)\s*\]\s*"
    r"(?P<spk>[Ss][Pp][Ee][Aa][Kk][Ee][Rr][\s_]*\d+)\s*[:：]\s*(?P<text>.*)"
)


def parse_ts(ts: str) -> float | None:
    """00:00:01.234 / 00:01.234 / 1.234 → 秒。格式飘了返回 None，丢弃该段。"""
    parts = ts.strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, s = nums
    elif len(nums) == 2:
        h, m, s = 0.0, nums[0], nums[1]
    elif len(nums) == 1:
        h, m, s = 0.0, 0.0, nums[0]
    else:
        return None
    return h * 3600 + m * 60 + s


def parse_hyp(hyp: str, session_id: str) -> tuple[list[dict], int]:
    """SoulX 原始文本 → (SegLST 记录列表, 解析失败行数)。

    speaker 按**首次出现顺序**重编号成 spk1/spk2…（同 moss_sat.to_seglst）：
    tcpWER 对说话人做最优匹配，编号本身无意义但须在 session 内自洽。
    """
    spk_map: dict[str, int] = {}
    records: list[dict] = []
    bad = 0
    for raw in hyp.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            bad += 1
            continue
        start, end = parse_ts(m.group("start")), parse_ts(m.group("end"))
        if start is None or end is None or end < start:
            bad += 1
            continue
        words = to_words(m.group("text"))
        if not words:
            continue  # 空段不计入，但不算解析失败
        spk = re.sub(r"[\s_]+", "", m.group("spk")).lower()
        if spk not in spk_map:
            spk_map[spk] = len(spk_map) + 1
        records.append({
            "session_id": session_id,
            "speaker": f"spk{spk_map[spk]}",
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "words": words,
        })
    records.sort(key=lambda r: r["start_time"])
    return records, bad


def main() -> int:
    ap = argparse.ArgumentParser(description="SoulX JSONL → SegLST")
    ap.add_argument("--jsonl", required=True, help="soulx_batch_infer.py 的输出")
    ap.add_argument("--out", required=True, help="SegLST 输出目录")
    args = ap.parse_args()

    src = Path(args.jsonl)
    if not src.exists():
        logger.error("输入不存在：%s", src)
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_sess = n_rec = n_bad = n_empty = 0
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        sid, hyp = obj["index"], obj.get("hyp", "")
        records, bad = parse_hyp(hyp, sid)
        n_sess += 1
        n_rec += len(records)
        n_bad += bad
        if not records:
            n_empty += 1
            logger.warning("%s: 0 条记录（原始 %d 字符）", sid, len(hyp))
        (out_dir / f"{sid}.seglst.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("%d session / %d 条记录 -> %s", n_sess, n_rec, out_dir)
    if n_bad:
        logger.warning("解析失败 %d 行（格式不匹配）", n_bad)
    if n_empty:
        logger.warning("空输出 %d 个 session", n_empty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
