"""Step 7 合规自证：把托管实例 fun-asr 的归档转写与**本地开源 Fun-ASR** 逐场比对。

## 为什么有这个脚本

赛题 FAQ #7：「可以使用开源大模型，但不可使用 API 调用方式使用闭源大模型或方案」。
我们 Step 7 的归档产物由托管实例的 `fun-asr` 产出。调用方看不到后端权重，
因此**无法仅凭 API 自证后端是开源模型**。

本脚本给出可复现的旁证：用 ModelScope 上公开可下的 `FunAudioLLM/Fun-ASR-Nano-2512`
在本机重新转写同一批音频，与归档产物逐场算字错率（CER）。
CER 足够低 ⇒ 两者是同一模型族的输出，托管实例并未替换成闭源方案。

## 为什么关掉说话人分离

funasr 1.4.3 的 `spk_model` 后处理与 Fun-ASR-Nano 不兼容（`auto_model.py` 取不到
`raw_text`，报 `Missing punc_model` 并产出 0 条）。但本脚本只比对**文本**，
分离与否不影响结论，故只装 VAD + PUNC。

## 已知的系统性差异（不代表模型不同）

托管实例输出阿拉伯数字（`2012年`、`10年前`），本地开源权重输出中文数字
（`二零一二年`、`十年前`）。这是**后处理的数字规整策略**差异，不是声学模型差异。
故报告同时给出两个口径：
    cer          —— 原样比对
    cer_zh_digit —— 把两边的阿拉伯数字统一转成中文数字后再比对

用法：
    MODELSCOPE_CACHE=/root/.cache/modelscope \\
    python src/funasr_api_vs_local.py \\
        --wav-dir <test wav 目录> \\
        --api-dir <归档的 test_cam_multi 目录> \\
        --out artifacts/funasr_api_vs_local.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ModelScope 上公开可下载的开源 Fun-ASR
LOCAL_ASR_MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"
VAD_MODEL = "fsmn-vad"
PUNC_MODEL = "ct-punc"

# 与 seglst_converter._to_words 同一套：去标点、中文逐字、英文小写
_PUNCT = r"""[，。！？、；：""''…,.!?;:"'()\[\]【】]"""
_TOKEN = r"[A-Za-z]+|[0-9]+|[一-鿿]"

_DIGIT_ZH = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
             "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"}


def to_chars(text: str) -> str:
    """转写文本 → 可比对的字符串（去标点与空白，中文逐字、英文成词）。"""
    cleaned = re.sub(_PUNCT, "", text or "")
    return "".join(t.lower() if t.isascii() else t for t in re.findall(_TOKEN, cleaned))


def zh_digits(s: str) -> str:
    """阿拉伯数字逐位转中文，消除两侧数字规整策略的差异。"""
    return "".join(_DIGIT_ZH.get(c, c) for c in s)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein 距离，滚动数组实现。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """字错率 = 编辑距离 / 参考长度。ref 为空时：hyp 也空记 0，否则记 1。"""
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def api_text(path: Path) -> str:
    """归档 SegLST → 按时间拼接的纯文本。"""
    recs = json.loads(path.read_text(encoding="utf-8"))
    recs = recs if isinstance(recs, list) else recs["segments"]
    return "".join(r["words"].replace(" ", "")
                   for r in sorted(recs, key=lambda r: float(r["start_time"])))


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--wav-dir", required=True, help="test 音频目录")
    ap.add_argument("--api-dir", required=True, help="归档的托管实例产物（test_cam_multi）")
    ap.add_argument("--out", required=True, help="对比报告 JSON")
    ap.add_argument("--model", default=LOCAL_ASR_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size-s", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="只比对前 N 场（0 = 全部）")
    args = ap.parse_args()

    from funasr import AutoModel  # 延迟导入：没装 funasr 时也能看 --help

    api_dir, wav_dir = Path(args.api_dir), Path(args.wav_dir)
    sids = sorted(p.name.split(".")[0] for p in api_dir.glob("*.seglst.json"))
    if args.limit:
        sids = sids[: args.limit]
    logger.info("待比对 %d 场；本地模型 %s", len(sids), args.model)

    model = AutoModel(model=args.model, vad_model=VAD_MODEL, punc_model=PUNC_MODEL,
                      device=args.device, disable_update=True)

    rows: list[dict] = []
    for i, sid in enumerate(sids, 1):
        wav = wav_dir / f"{sid}.wav"
        if not wav.exists():
            logger.warning("缺音频 %s，跳过", wav)
            continue
        try:
            res = model.generate(input=str(wav), batch_size_s=args.batch_size_s)
        except Exception as exc:                       # 单场失败不阻断整批
            logger.error("[%d/%d] %s 推理失败：%s", i, len(sids), sid, exc)
            continue
        a = to_chars(api_text(api_dir / f"{sid}.seglst.json"))
        b = to_chars((res[0] or {}).get("text", "") if res else "")
        rows.append({
            "session_id": sid,
            "n_api": len(a),
            "n_local": len(b),
            "cer": round(cer(a, b), 4),
            "cer_zh_digit": round(cer(zh_digits(a), zh_digits(b)), 4),
            "api_text": a,
            "local_text": b,
        })
        if i % 20 == 0 or i == len(sids):
            done = [r["cer_zh_digit"] for r in rows]
            logger.info("[%d/%d] 已比对 %d 场，当前均值 CER(数字归一) %.4f",
                        i, len(sids), len(done), sum(done) / len(done))

    if not rows:
        logger.error("没有可比对的场次")
        return 1

    # 汇总：按总字数加权，避免短场次被等权放大
    tot_ref = sum(r["n_api"] for r in rows)
    w_cer = sum(r["cer"] * r["n_api"] for r in rows) / tot_ref
    w_cer_d = sum(r["cer_zh_digit"] * r["n_api"] for r in rows) / tot_ref
    summary = {
        "n_sessions": len(rows),
        "local_model": args.model,
        "total_ref_chars": tot_ref,
        "weighted_cer": round(w_cer, 4),
        "weighted_cer_zh_digit": round(w_cer_d, 4),
        "mean_cer_zh_digit": round(sum(r["cer_zh_digit"] for r in rows) / len(rows), 4),
        "identical_sessions": sum(1 for r in rows if r["cer_zh_digit"] == 0.0),
        "sessions_under_0.10": sum(1 for r in rows if r["cer_zh_digit"] < 0.10),
        "sessions_under_0.20": sum(1 for r in rows if r["cer_zh_digit"] < 0.20),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "sessions": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("=== 汇总 ===")
    for k, v in summary.items():
        logger.info("  %-24s %s", k, v)
    logger.info("报告 → %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
