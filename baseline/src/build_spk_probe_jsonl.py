"""把我们的预测渲染成 MOSS 目标格式，并标出每段说话人 id 末位数字的 token 下标。

配套 `spk_ac_probe.py`。渲染格式取自真实训练样本：`[start][Sxx]text[end]`，
时间戳两位小数。说话人按**首次出现顺序**编号（与 `moss_sat.to_seglst` 的反向映射一致）。

tok_off 的算法：目标串里定位该段 `[Sxx]` 末位数字的字符偏移，
对 `target[:偏移]` 单独 tokenize，长度即该数字在完整目标里的 token 下标。
数字在本 tokenizer 中恒为单 token，边界干净，故前缀切分是稳的；
`spk_ac_probe.py` 里另有一道 token id 断言兜底。
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MOSS_REPO = os.environ.get("MOSS_REPO", "/root/autodl-tmp/MOSS-Transcribe-Diarize")
PROMPT = ("请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，"
          "正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。")


def main() -> int:
    sys.path.insert(0, MOSS_REPO)
    ap = argparse.ArgumentParser(description="构造说话人似然探针的 jsonl")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from moss_transcribe_diarize.processing_moss_transcribe_diarize import (  # type: ignore  # noqa: PLC0415
        MossTranscribeDiarizeProcessor,
    )
    tok = MossTranscribeDiarizeProcessor.from_pretrained(
        args.model, trust_remote_code=True).tokenizer

    n_seg = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for p in sorted(glob.glob(f"{args.pred_dir}/[0-9]*.seglst.json")):
            sid = os.path.basename(p).split(".")[0]
            wav = f"{args.wav_dir}/{sid}.wav"
            if not os.path.exists(wav):
                logger.warning("缺音频 %s，跳过", wav)
                continue
            recs = sorted(json.load(open(p, encoding="utf-8")), key=lambda r: r["start_time"])
            order: dict[str, int] = {}
            for r in recs:
                if r["speaker"] not in order:
                    order[r["speaker"]] = len(order) + 1
            if len(order) > 9:
                logger.warning("%s 说话人 >9，跳过", sid)
                continue

            target = ""
            segs = []
            for r in recs:
                k = order[r["speaker"]]
                tag = f"[S{k:02d}]"
                head = f"[{r['start_time']:.2f}]"
                # 末位数字的字符偏移 = 本段起点 + head + "[S" + 第一位数字
                digit_char = len(target) + len(head) + 3
                target += f"{head}{tag}{r['words']}[{r['end_time']:.2f}]"
                segs.append({
                    "start_time": r["start_time"],
                    "cur_digit": str(k),
                    "tok_off": len(tok.encode(target[:digit_char], add_special_tokens=False)),
                })
            n_seg += len(segs)
            fh.write(json.dumps({
                "conversation": [
                    {"role": "user", "message_type": "text", "content": PROMPT},
                    {"role": "user", "message_type": "audio", "content": wav},
                    {"role": "assistant", "message_type": "text", "content": target},
                ],
                "meta": {"session": sid, "nspk": len(order), "segs": segs},
            }, ensure_ascii=False) + "\n")
    logger.info("BUILD DONE %d 段 -> %s", n_seg, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
