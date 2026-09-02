"""把 dev 的 ref.seglst.json 转成 MOSS 微调用的 JSONL，并切 train/holdout。

微调动机：v007 的 del=1620 是最大误差块，根因是 MOSS **少判 1-2 个说话人**
（016 判 2 人 vs ref 3 人；069 判 5 vs 6）——通用语料训出来的模型没见过本赛的
中文多人短对话。prompt 调不动（X11 已证伪），只能改模型本身。

⚠️ 合规：只用 dev（官方标注数据）。**test 集不可以任何形式参与训练**
（赛题说明.md §103/§148）。本脚本硬编码只读 dev 目录。

⚠️ dev 是我们唯一的验证集。全拿去训练就没法测了，所以切 holdout 并**按说话人数
分层**——6 人段全 dev 只有 6 个，随机切容易全漏进训练集，而少判说话人正是要修的病，
holdout 必须保住难段。

标签格式必须和推理时逐字符对齐（`[start][Sxx]text[end]`，见 transcript_parser），
否则微调等于教模型说另一种方言。

用法：
    cd baseline
    .venv/bin/python src/prep_finetune_data.py --out output/finetune
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEV_DIR = BASE_DIR.parent / "data/extracted/dev/dev"

# 必须与 moss_transcribe_diarize.inference_utils.DEFAULT_PROMPT 完全一致
DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


def join_tokens(words: str) -> str:
    """ref 的逐字空格串 → 自然文本。

    只在两个 ASCII token 之间保留空格（英文单词要分开，中文字不要）。
    训练标签贴近模型原生输出分布；评分前 `to_words()` 会重新切字，不影响 tcpWER。
    """
    tokens = words.split()
    if not tokens:
        return ""
    out = [tokens[0]]
    for prev, cur in zip(tokens, tokens[1:]):
        if prev.isascii() and cur.isascii():
            out.append(" ")
        out.append(cur)
    return "".join(out)


def build_transcript(records: list[dict]) -> str:
    """session 的 ref 记录 → MOSS 目标格式 `[start][Sxx]text[end]` 拼接串。

    speaker 按**首次出现顺序**重编号为 S01/S02…，与推理端 `to_seglst` 的重编号
    规则对称（tcpWER 做说话人最优匹配，编号本身无意义但需自洽）。
    """
    spk_map: dict[str, str] = {}
    parts = []
    for rec in sorted(records, key=lambda r: r["start_time"]):
        text = join_tokens(rec["words"])
        if not text:
            continue
        spk = rec["speaker"]
        if spk not in spk_map:
            spk_map[spk] = f"S{len(spk_map) + 1:02d}"
        parts.append(f"[{rec['start_time']:.2f}][{spk_map[spk]}]{text}[{rec['end_time']:.2f}]")
    return "".join(parts)


def stratified_split(
    sessions: dict[str, list[dict]], n_holdout: int, seed: int
) -> tuple[list[str], list[str]]:
    """按说话人数分层切分，保证 holdout 覆盖各难度档（尤其稀有的 6 人段）。"""
    by_nspk: dict[int, list[str]] = defaultdict(list)
    for sid, recs in sessions.items():
        by_nspk[len({r["speaker"] for r in recs})].append(sid)

    rng = random.Random(seed)
    holdout: list[str] = []
    total = len(sessions)
    for nspk in sorted(by_nspk):
        group = sorted(by_nspk[nspk])
        rng.shuffle(group)
        # 按该档占比分配名额，至少留 1 个（稀有档也要进 holdout）
        quota = max(1, round(n_holdout * len(group) / total))
        holdout.extend(group[:quota])

    holdout_set = set(holdout[:n_holdout])
    train = sorted(set(sessions) - holdout_set)
    return train, sorted(holdout_set)


def write_jsonl(path: Path, sids: list[str], sessions: dict[str, list[dict]], wav_dir: str) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for sid in sids:
            record = {
                "conversation": [
                    {"role": "user", "message_type": "text", "content": DEFAULT_PROMPT},
                    {"role": "user", "message_type": "audio", "content": f"{wav_dir}/{sid}.wav"},
                    {
                        "role": "assistant",
                        "message_type": "text",
                        "content": build_transcript(sessions[sid]),
                    },
                ]
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 MOSS 微调 JSONL（仅用 dev）")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--holdout", type=int, default=26, help="holdout session 数（默认 26）")
    parser.add_argument("--seed", type=int, default=0, help="切分随机种子（默认 0，可复现）")
    parser.add_argument("--wav-dir", default="/root/autodl-tmp/dev_wav",
                        help="JSONL 里写入的音频目录（GPU 机上的路径）")
    parser.add_argument("--ref", default=None,
                        help="SegLST 来源（默认 dev ref）。用于喂仿真数据，"
                             "如 output/sim_t82/ref.seglst.json")
    args = parser.parse_args()

    ref_path = Path(args.ref) if args.ref else DEV_DIR / "ref.seglst.json"
    # 合规守卫：test 集不得以任何形式参与训练（赛题说明.md §103/§148）
    if "test" in str(ref_path).lower():
        logger.error("拒绝：--ref 含 test，test 集不得参与训练（赛题 §103/§148）：%s", ref_path)
        return 2
    if not ref_path.exists():
        logger.error("找不到 ref：%s", ref_path)
        return 1

    sessions: dict[str, list[dict]] = defaultdict(list)
    for rec in json.loads(ref_path.read_text(encoding="utf-8")):
        sessions[rec["session_id"]].append(rec)
    logger.info("读入 %d 个 session / %d 条记录", len(sessions), sum(map(len, sessions.values())))

    train, holdout = stratified_split(sessions, args.holdout, args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train, sessions, args.wav_dir.rstrip("/"))
    write_jsonl(out_dir / "holdout.jsonl", holdout, sessions, args.wav_dir.rstrip("/"))
    (out_dir / "holdout_sessions.json").write_text(
        json.dumps(holdout, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def dist(sids: list[str]) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for sid in sids:
            counts[len({r["speaker"] for r in sessions[sid]})] += 1
        return dict(sorted(counts.items()))

    logger.info("train   %3d 段  说话人数分布 %s", len(train), dist(train))
    logger.info("holdout %3d 段  说话人数分布 %s", len(holdout), dist(holdout))
    logger.info("产物 → %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
