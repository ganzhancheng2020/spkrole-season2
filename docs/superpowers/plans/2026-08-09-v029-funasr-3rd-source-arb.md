# v029 fun-asr 第三源词级仲裁 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 fun-asr 加入 word_arb 做三候选词级仲裁，生成 v029 dev/test 产物与提交物。

**Architecture:** 新写 `funasr_retext.py` 复用 `fr_retext.py` 的重排逻辑，按词级时间戳把 fun-asr 词分配到 MOSS 段；扩展 `word_arb.py` 加 `--third_dir` + 两轮仲裁方案 C（先 MOSS vs FireRed → best-2 vs fun-asr）；沿用现有 `pick_ensemble.py`、`normalize_output.py`、`merge_submit.py`、`heavy_tail_gate.py`。

**Tech Stack:** Python 3.13, torch 2.13.0, transformers 5.14.1, soundfile, sklearn；阿里百炼 fun-asr 云端 API（已合规）；FireRedASR2-AED teacher forcing 声学似然（`/tmp/FireRedASR2S` + `/tmp/FireRedASR2-AED`，2026-08-09 核实仍在）。

## Global Constraints

| 约束 | 精确值 |
|---|---|
| 评分工具 | `meeteval-wer 0.4.3`（根 venv：`.venv/bin/meeteval-wer`） |
| 评分参数 | `tcpwer --collar 5` |
| 当前最优（避免倒退） | v026 线上 0.15756 / dev 16.090% / `submission_v026.json` shasum `32a0fc74…` |
| 命名规则 | 一切产物按 v0NN；dev 与 test 双产物；测试集不得用于训练 |
| 提交规则 | 提交必留人工；提交前 `shasum -a 256` 比对上一版（v014 教训） |
| 关卡 1（双切分）| `prep_finetune_data.py:stratified_split(sessions, 25, seed=0)` 切 train82/holdout24，两边同向改善 |
| 关卡 2（重尾）| `heavy_tail_gate.py v026_pick v029_pick` 剔 top-3 后净收益 > 0 |
| 评估顺序 | dev 评测（双切分 + 重尾）都 PASS 才进 test |
| 沙箱化 | fun-asr 重跑必须用新目录 `output/funasr_retext/`（不污染 `output_dev_funasr/`） |
| 仓库 | `/Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2`（绝对路径在每条命令里） |

---

## File Structure

### 新建文件（3 个）

| 文件 | 职责 | 行数预期 |
|---|---|---|
| `baseline/src/funasr_retext.py` | fun-asr 云端调用 + 词级时间戳重排到 MOSS 段 + 落盘 ts.json + seglst.json | ~120 |
| `baseline/output/funasr_retext/{001..106}.seglst.json` | dev 侧 fun-asr 重排到 MOSS 段 | — |
| `baseline/output/funasr_retext/{001..106}.ts.json` | dev 侧 fun-asr 词级时间戳 | — |
| `baseline/output/wa_funasr_dev/{001..106}.seglst.json` | dev 三候选仲裁后 | — |
| `baseline/output/v029_pick/{001..106}.seglst.json` | dev 三路择优后 | — |
| `baseline/output_test/funasr_retext/{001..394}.seglst.json` | test 侧 fun-asr 重排 | — |
| `baseline/output_test/funasr_retext/{001..394}.ts.json` | test 侧 fun-asr 词级时间戳 | — |
| `baseline/output_test/wa_funasr/{001..394}.seglst.json` | test 三候选仲裁后 | — |
| `baseline/output_test_v029/{001..394}.seglst.json` | test 三路择优后 | — |
| `baseline/output_test_v029_norm/{001..394}.seglst.json` | test 规范化后 | — |
| `baseline/submission_v029.json` | v029 提交物 | — |

### 修改文件（1 个）

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `baseline/src/word_arb.py` | 模块 docstring + argparse + arbitrate_session | 加 `--third_dir`；实现两轮仲裁方案 C |

### 不修改但复用（5 个）

| 文件 | 复用方式 |
|---|---|
| `baseline/src/fr_retext.py:20-49` | funasr_retext.py 模仿这段重排逻辑 |
| `baseline/src/run.py:103 transcribe_audio` | funasr_retext.py 调用 fun-asr 云端 |
| `baseline/src/inference.py:73 _extract_sentences` | funasr_retext.py 解析 fun-asr 返回（已含 words[]） |
| `baseline/src/pick_ensemble.py:103-164` | H+D 规则，输出 v029_pick |
| `baseline/src/normalize_output.py:50-59` | CHAR_MAP，输出 v029_norm |

### 工具文件（4 个）

| 文件 | 用途 |
|---|---|
| `baseline/src/heavy_tail_gate.py` | 提交前硬关卡 |
| `baseline/src/batch_evaluate.py` | dev/test 全量评测 |
| `baseline/src/merge_submit.py:17-25` | test 侧合并提交物 |
| `baseline/src/prep_finetune_data.py:stratified_split` | train82/holdout24 双切分 |

### 设计单元边界

- `funasr_retext.py` 只做"云端 fun-asr → 重排到 MOSS 段 + 落盘"两件事，不做仲裁
- `word_arb.py` 只做"段内差异块枚举 + 声学打分 + 选最优"，不接触 fun-asr 云端
- `pick_ensemble.py` 只做"session 级 MOSS/CAM H+D 择优"，不接触文本
- 三个单元各自独立可测，**没有循环依赖**

---

## Task 1: 新写 `funasr_retext.py`

**Files:**
- Create: `baseline/src/funasr_retext.py`

**Interfaces:**
- Consumes: `Path(wav_dir)`, `Path(moss_dir)`（MOSS 段结构参考）, `Path(out_dir)`
- Produces:
  - `{out_dir}/{sid}.seglst.json`：`list[dict]`，每条 `{session_id, speaker: "spkN", start_time, end_time, words}`，段结构与 MOSS 完全一致
  - `{out_dir}/{sid}.ts.json`：`list[[token, start_sec, end_sec]]`，词级时间戳

**Prerequisites:**
- `baseline/.env` 含 `DASHSCOPE_API_KEY`
- `/tmp/FireRedASR2S` 与 `/tmp/FireRedASR2-AED`（不需要，仅 word_arb 用）

- [ ] **Step 1: 写脚本骨架**

```python
"""fun-asr 重排到 MOSS 段 + 落盘词级时间戳（fun-asr 第三源集成 Step 1）。

模仿 fr_retext.py 的结构：fun-asr 云端整段转写（含词级时间戳）→ 按 begin_time
把每个 word 分配到 MOSS 段 → 落盘 {sid}.ts.json（词级时间戳）和 {sid}.seglst.json
（段结构与 MOSS 一致）。

用法：
    cd baseline
    .venv/bin/python src/funasr_retext.py \\
        ../data/extracted/dev/dev/wav \\
        ../baseline/output/v007_moss \\
        ../baseline/output/funasr_retext
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

# 复用现有 run.py 的 fun-asr 调用链
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import transcribe_audio  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PUNCT = re.compile(r"""[，。！？、；：""''…,.!?;:"'()\[\]【】]""")
TOKEN = re.compile(r"[A-Za-z]+|[0-9]+|[一-鿿]")


def tokenize_words(sent_words: list[dict]) -> list[tuple[str, float, float]]:
    """fun-asr 返回的 words[{begin_time,end_time,text,speaker_id}] → 字符级 [(tok, st, en)]。"""
    out = []
    for w in sent_words:
        for ch in TOKEN.findall(PUNCT.sub("", w["text"])):
            out.append((ch, w["begin_time"] / 1000.0, w["end_time"] / 1000.0))
    return out


def reassign(moss_recs: list[dict], all_words: list[tuple[str, float, float]]) -> list[dict]:
    """按 (st+en)/2 把每个 word 分配到最近的 MOSS 段。"""
    buckets: dict[int, list[str]] = {j: [] for j in range(len(moss_recs))}
    for tok, st, en in all_words:
        c = (st + en) / 2
        best, bd = None, 1e9
        for j, x in enumerate(moss_recs):
            if x["start_time"] <= c <= x["end_time"]:
                best, bd = j, 0
                break
            d = min(abs(c - x["start_time"]), abs(c - x["end_time"]))
            if d < bd:
                best, bd = j, d
        if best is not None:
            buckets[best].append(tok)
    out = []
    for j, x in enumerate(moss_recs):
        words = " ".join(buckets[j])
        out.append({**x, "words": words if words else x["words"]})
    return out


def process_one(sid: str, wav_dir: Path, moss_dir: Path, out_dir: Path) -> bool:
    """返回 True 表示产出新文件，False 表示跳过（已存在）。"""
    out_seglst = out_dir / f"{sid}.seglst.json"
    out_ts = out_dir / f"{sid}.ts.json"
    if out_seglst.exists() and out_ts.exists():
        return False
    wav = wav_dir / f"{sid}.wav"
    moss_p = moss_dir / f"{sid}.seglst.json"
    if not moss_p.exists():
        logger.warning("MOSS 段缺失: %s", sid)
        return False
    sentences = transcribe_audio(wav, speaker_count=None)
    all_words: list[tuple[str, float, float]] = []
    for sent in sentences:
        all_words.extend(tokenize_words(sent.get("words", [])))
    moss_recs = json.loads(moss_p.read_text(encoding="utf-8"))
    out_recs = reassign(moss_recs, all_words)
    out_seglst.write_text(json.dumps(out_recs, ensure_ascii=False, indent=2), encoding="utf-8")
    out_ts.write_text(
        json.dumps([[t, round(s, 3), round(e, 3)] for t, s, e in all_words], ensure_ascii=False),
        encoding="utf-8",
    )
    return True


def main() -> int:
    if len(sys.argv) != 4:
        logger.error("用法: %s <wav_dir> <moss_dir> <out_dir>", __file__)
        return 2
    wav_dir, moss_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_done = n_skip = 0
    for sid in sorted({p.stem for p in moss_dir.glob("[0-9]*.seglst.json")}):
        if process_one(sid, wav_dir, moss_dir, out_dir):
            n_done += 1
        else:
            n_skip += 1
    logger.info("fun-asr retext 完成: 新增 %d, 跳过 %d", n_done, n_skip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 写最小冒烟测试（mock fun-asr 调用）**

```python
# baseline/tests/test_funasr_retext.py
"""冒烟测试：mock transcribe_audio，验证重排和落盘格式正确。"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src import funasr_retext


def test_tokenize_words_skips_punct():
    words = [
        {"begin_time": 0, "end_time": 100, "text": "你 好"},
        {"begin_time": 100, "end_time": 200, "text": "，世 界。"},
    ]
    out = funasr_retext.tokenize_words(words)
    assert [t for t, _, _ in out] == ["你", "好", "世", "界"]


def test_reassign_assigns_word_to_overlapping_segment():
    moss = [
        {"session_id": "001", "speaker": "spk1", "start_time": 0.0, "end_time": 5.0, "words": ""},
        {"session_id": "001", "speaker": "spk2", "start_time": 5.0, "end_time": 10.0, "words": ""},
    ]
    words = [("你", 1.0, 2.0), ("好", 7.0, 8.0)]
    out = funasr_retext.reassign(moss, words)
    assert out[0]["words"] == "你"
    assert out[1]["words"] == "好"
```

- [ ] **Step 3: 运行冒烟测试**

Run: `cd baseline && PYTHONPATH=src .venv/bin/python -m pytest tests/test_funasr_retext.py -v`
Expected: PASS（2 passed）

- [ ] **Step 4: 跑 dev 段（1 段验证）**

```bash
cd baseline
PYTHONPATH=src .venv/bin/python src/funasr_retext.py \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/data/extracted/dev/dev/wav \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output/v007_moss \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output/funasr_retext
```

Expected: `INFO fun-asr retext 完成: 新增 1, 跳过 0`
检查: `ls baseline/output/funasr_retext/001.seglst.json baseline/output/funasr_retext/001.ts.json` 都存在
检查: `python3 -c "import json; r=json.load(open('baseline/output/funasr_retext/001.seglst.json')); print(len(r), '段,', sum(len(x['words'].split()) for x in r), '词')"` → 应有合理词数（与 MOSS 段数一致）

- [ ] **Step 5: 跑完整 dev 队列（106 段）**

```bash
cd baseline
PYTHONPATH=src .venv/bin/python src/funasr_retext.py \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/data/extracted/dev/dev/wav \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output/v007_moss \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output/funasr_retext
```

Expected: `INFO fun-asr retext 完成: 新增 106, 跳过 0`
检查: `ls baseline/output/funasr_retext/*.seglst.json | wc -l` = 106
检查: `ls baseline/output/funasr_retext/*.ts.json | wc -l` = 106

- [ ] **Step 6: Commit**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2
git add baseline/src/funasr_retext.py baseline/tests/test_funasr_retext.py baseline/output/funasr_retext/
git commit -m "feat: funasr_retext.py 重排到 MOSS 段 + 落盘词级时间戳（v029 step 1）"
```

---

## Task 2: 扩展 `word_arb.py` 支持 `--third_dir` + 两轮仲裁方案 C

**Files:**
- Modify: `baseline/src/word_arb.py:1-2`（docstring）、`word_arb.py:60-66`（main 参数解析）、`word_arb.py:70-110`（arbitrate_session 调用）

**Interfaces:**
- Consumes: `Path(moss_dir)`、`Path(fr_dir)`、`Path(third_dir)`（fun-asr）、`Path(wav_dir)`、`Path(out_dir)`
- Produces: `{out_dir}/{sid}.seglst.json`，与现有 wa_m0_dev 同 schema

**现有 word_arb.py 关键函数**（保留）：
- `blocks(a, b)`：MOSS vs FireRed 差异块切分
- `make_enc(clip, sr)`：音频 → encoder 输出（用 FireRed）
- `score(enc, mask, text)`：teacher forcing 声学似然

**Prerequisites:**
- `baseline/output/funasr_retext/` 已存在（Task 1 完成）

- [ ] **Step 1: 写两轮仲裁的最小实现**

修改 `baseline/src/word_arb.py`：

```python
# 在文件顶部加 import（如已有 import json 等可保留）
import sys, os, glob, time, itertools, tempfile, json, logging
from pathlib import Path
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# (原有模型加载、blocks、make_enc、score 函数保留)

# 新增：两轮仲裁方案 C
def arbitrate_session_3way(
    moss_recs: list[dict], fr_recs: list[dict], third_recs: list[dict], wav_path: str
) -> tuple[list[dict], int, int]:
    """两轮仲裁：第 1 轮 MOSS vs FireRed 选 best-2；第 2 轮 best-2 vs fun-asr。

    段数必须三路相同（fun-asr 已被 funasr_retext 重排到 MOSS 段结构）。
    """
    x, sr = sf.read(wav_path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    res: list[dict] = []
    nsw = ncalls = 0
    for a, b, c in zip(moss_recs, fr_recs, third_recs):
        ta, tb, tc = a["words"].split(), b["words"].split(), c["words"].split()
        # 三路完全一致 → 跳过
        if ta == tb == tc or a["end_time"] - a["start_time"] < MIN_SEG_SEC:
            res.append(dict(a))
            continue
        # 第 1 轮：MOSS vs FireRed（沿用原逻辑）
        bl = blocks(ta, tb)
        di1 = [t for t, blk in enumerate(bl) if blk[0] == "diff"]
        if not di1:
            best_2 = ta
            di1_combo = tuple([0])
        else:
            di1 = sorted(di1, key=lambda t: -max(len(bl[t][1]), len(bl[t][2])))[:MAX_BLOCKS]
            clip = x[int(a["start_time"] * sr): int(a["end_time"] * sr)]
            enc, mask = make_enc(clip, sr)
            keep_moss = tuple([0] * len(di1))
            best1: tuple[float, list[str], tuple] = (float("-inf"), ta, keep_moss)
            for combo in itertools.product([0, 1], repeat=len(di1)):
                toks: list[str] = []
                for t, blk in enumerate(bl):
                    if blk[0] == "eq":
                        toks += blk[1]
                    elif t in di1:
                        toks += blk[2] if combo[di1.index(t)] else blk[1]
                    else:
                        toks += blk[1]
                ncalls += 1
                v = score(enc, mask, "".join(toks))
                if best1 is None or v > best1[0]:
                    best1 = (v, toks, combo)
            best_2 = best1[1]
        # 第 2 轮：best_2 vs fun-asr
        if ta == tc:
            res.append({**a, "words": " ".join(best_2)})
            continue
        bl2 = blocks(best_2, tc)
        di2 = [t for t, blk in enumerate(bl2) if blk[0] == "diff"]
        if not di2:
            res.append({**a, "words": " ".join(best_2)})
            continue
        di2 = sorted(di2, key=lambda t: -max(len(bl2[t][1]), len(bl2[t][2])))[:MAX_BLOCKS]
        # 复用第 1 轮的 enc/mask（同一段音频）
        keep_b2 = tuple([0] * len(di2))
        best2: tuple[float, list[str], tuple] = (float("-inf"), best_2, keep_b2)
        for combo in itertools.product([0, 1], repeat=len(di2)):
            toks: list[str] = []
            for t, blk in enumerate(bl2):
                if blk[0] == "eq":
                    toks += blk[1]
                elif t in di2:
                    toks += blk[2] if combo[di2.index(t)] else blk[1]
                else:
                    toks += blk[1]
            ncalls += 1
            v = score(enc, mask, "".join(toks))
            if best2 is None or v > best2[0]:
                best2 = (v, toks, combo)
        base2 = score(enc, mask, "".join(best_2))
        ncalls += 1
        if best2[2] != keep_b2 and best2[0] - base2 >= MARGIN:
            nsw += 1
            res.append({**a, "words": " ".join(best2[1])})
        else:
            res.append({**a, "words": " ".join(best_2)})
    return res, nsw, ncalls


# 修改 main()：argv 从 4 个 → 5 个（加 --third_dir 作为可选第 5 个位置参数）
def main() -> int:
    if len(sys.argv) < 5:
        logger.error("用法: %s <moss_dir> <fr_dir> <wav_dir> <out_dir> [third_dir]", __file__)
        return 2
    moss_dir, fr_dir, wav_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    third_dir = sys.argv[5] if len(sys.argv) > 5 else None

    sids = sorted(
        os.path.basename(p).split(".")[0] for p in glob.glob(f"{moss_dir}/[0-9]*.seglst.json")
    )

    t0 = time.time()
    total_sw = total_calls = n_fallback = 0
    for k, sid in enumerate(sids, 1):
        out_path = f"{out_dir}/{sid}.seglst.json"
        if os.path.exists(out_path):
            continue
        with open(f"{moss_dir}/{sid}.seglst.json", encoding="utf-8") as fh:
            moss_recs = json.load(fh)
        fr_path = f"{fr_dir}/{sid}.seglst.json"
        fr_recs = None
        if os.path.exists(fr_path):
            with open(fr_path, encoding="utf-8") as fh:
                fr_recs = json.load(fh)

        # 三路长度必须一致才能仲裁
        if third_dir:
            third_path = f"{third_dir}/{sid}.seglst.json"
            third_recs = None
            if os.path.exists(third_path):
                with open(third_path, encoding="utf-8") as fh:
                    third_recs = json.load(fh)
            if (fr_recs is not None and len(fr_recs) == len(moss_recs)
                and third_recs is not None and len(third_recs) == len(moss_recs)):
                res, nsw, ncalls = arbitrate_session_3way(
                    moss_recs, fr_recs, third_recs, f"{wav_dir}/{sid}.wav"
                )
            else:
                n_fallback += 1
                res = [dict(r) for r in moss_recs]
                nsw = ncalls = 0
        else:
            if fr_recs is None or len(fr_recs) != len(moss_recs):
                n_fallback += 1
                res = [dict(r) for r in moss_recs]
                nsw = ncalls = 0
            else:
                res, nsw, ncalls = arbitrate_session(moss_recs, fr_recs, f"{wav_dir}/{sid}.wav")
        total_sw += nsw
        total_calls += ncalls
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        logger.info(
            "[%d/%d] %s: 累计换 %d 段，打分 %d 次 (%.0fs)",
            k, len(sids), sid, total_sw, total_calls, time.time() - t0,
        )
    logger.info(
        "WORD ARB DONE 换 %d 段，打分 %d 次，%d 段因源结构不一致回退 MOSS",
        total_sw, total_calls, n_fallback,
    )
    return 0
```

- [ ] **Step 2: 写 3 候选仲裁的最小单元测试**

```python
# baseline/tests/test_word_arb_3way.py
"""3 候选仲裁最小测试：mock 评分，验证两轮仲裁顺序正确。"""
from unittest.mock import patch
import pytest

from src import word_arb


def test_3way_calls_score_twice_when_both_rounds_differ():
    """当两轮都需要仲裁时，score() 至少被调用 2 次（MOSS 基线 + 一次打分）。"""
    moss = [{"start_time": 0.0, "end_time": 1.0, "words": "A B"}]
    fr = [{"start_time": 0.0, "end_time": 1.0, "words": "A C"}]
    third = [{"start_time": 0.0, "end_time": 1.0, "words": "A D"}]
    with patch.object(word_arb, "make_enc", return_value=(None, None)), \
         patch.object(word_arb, "score", return_value=0.0) as mock_score:
        word_arb.arbitrate_session_3way(moss, fr, third, "/tmp/fake.wav")
        assert mock_score.call_count >= 2


def test_3way_keeps_moss_when_all_three_identical():
    """三路完全一致时直接保留 MOSS。"""
    moss = [{"start_time": 0.0, "end_time": 1.0, "words": "A B"}]
    fr = [{"start_time": 0.0, "end_time": 1.0, "words": "A B"}]
    third = [{"start_time": 0.0, "end_time": 1.0, "words": "A B"}]
    res, nsw, _ = word_arb.arbitrate_session_3way(moss, fr, third, "/tmp/fake.wav")
    assert res[0]["words"] == "A B"
    assert nsw == 0
```

- [ ] **Step 3: 运行单元测试**

Run: `cd baseline && PYTHONPATH=src .venv/bin/python -m pytest tests/test_word_arb_3way.py -v`
Expected: PASS（2 passed）

- [ ] **Step 4: 跑 dev 三候选仲裁全量**

```bash
cd baseline
PYTHONPATH=src diarizen_venv/bin/python src/word_arb.py \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output/v007_moss \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output/fr_retext \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/data/extracted/dev/dev/wav \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output/wa_funasr_dev \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output/funasr_retext
```

Expected: `INFO WORD ARB DONE 换 N 段, 打分 M 次, K 段因源结构不一致回退 MOSS`

- [ ] **Step 5: 验证产出**

```bash
ls /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output/wa_funasr_dev/*.seglst.json | wc -l
# 期望: 106
```

- [ ] **Step 6: Commit**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2
git add baseline/src/word_arb.py baseline/tests/test_word_arb_3way.py baseline/output/wa_funasr_dev/
git commit -m "feat: word_arb 支持 --third_dir 3 候选两轮仲裁（v029 step 2）"
```

---

## Task 3: dev 评测 + 规范化

**Files:**
- Read: `baseline/output/wa_funasr_dev/`
- Write: `baseline/output/v029_pick/`

**Interfaces:**
- Consumes: `wa_funasr_dev/(106)`
- Produces: `v029_pick/(106)`（pick_ensemble H/D 择优后）

- [ ] **Step 1: 跑 dev 全量评测，对比 v026 基线**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline
PYTHONPATH=src .venv/bin/python src/batch_evaluate.py --dir output/wa_funasr_dev
```

Expected: `=== 全量 tcpWER = X.XXX%` 较 v026 16.090% 改善 ≥ 0.3 点（即 ≤ 15.79%）
若未改善 → **STOP**，fun-asr 第三源方案失效，回到 brainstorming。

- [ ] **Step 2: 跑 pick_ensemble H+D 择优**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline
PYTHONPATH=src .venv/bin/python src/pick_ensemble.py \
  --moss output/wa_funasr_dev \
  --cam output/hyp_sw_m3_7_0.70 \
  --out output/v029_pick
```

Expected: `INFO 共 106 段, 其中 N 段改用 CAM++`

- [ ] **Step 3: 跑输出规范化**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline
PYTHONPATH=src .venv/bin/python src/normalize_output.py \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output/v029_pick \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output/v029_norm
```

Expected: `INFO 共 106 个 session, 改写 M 个字符, 涉及 N 个 session (X.X%)`

- [ ] **Step 4: Commit**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2
git add baseline/output/v029_pick/ baseline/output/v029_norm/
git commit -m "feat: v029_pick + normalize（v029 step 3）"
```

---

## Task 4: 双切分 + 重尾两道关卡

**Files:**
- Read: `baseline/output/v029_norm/`, `baseline/output/v026_pick/`

- [ ] **Step 1: 跑双切分（train82/holdout24 同步）**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline

# 切分（沿用 prep_finetune_data.py 的 stratified_split seed=0）
PYTHONPATH=src .venv/bin/python -c "
import json, glob, sys
sys.path.insert(0, 'src')
from prep_finetune_data import stratified_split
ref = json.load(open('../data/extracted/dev/dev/ref.seglst.json'))
sids = sorted({r['session_id'] for r in ref})
train82, holdout24 = stratified_split(sids, 25, seed=0)
import os, shutil
for sub in ['train82', 'holdout24']:
    os.makedirs(f'output/v029_norm/{sub}', exist_ok=True)
for sid in train82:
    src = f'output/v029_norm/{sid}.seglst.json'
    if os.path.exists(src): shutil.copy(src, f'output/v029_norm/{sub}/{sid}.seglst.json')
for sid in holdout24:
    src = f'output/v029_norm/{sid}.seglst.json'
    if os.path.exists(src): shutil.copy(src, f'output/v029_norm/{sub}/{sid}.seglst.json')
print('train82:', len(train82), 'holdout24:', len(holdout24))
"

# 评测两边
PYTHONPATH=src .venv/bin/python src/batch_evaluate.py --dir output/v029_norm/train82
PYTHONPATH=src .venv/bin/python src/batch_evaluate.py --dir output/v029_norm/holdout24
```

Expected:
- train82 较 v026 train82 (16.806%) 改善
- holdout24 较 v026 holdout24 (15.297%) 改善
- **两边同向改善**才 PASS；若 holdout 改善幅度 < train，则可能过拟合 → **STOP**

- [ ] **Step 2: 跑重尾关卡（剔 top-3 后净收益）**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline
PYTHONPATH=src .venv/bin/python src/heavy_tail_gate.py \
  output/v026_pick \
  output/v029_norm
```

Expected: `✅ PASS: 剔 top-3 后净收益 +X.XXX 点`
若 FAIL（净收益 ≤ 0）→ **STOP**（参考 v027 重尾封死案例）

- [ ] **Step 3: 验证反向损失（无 session 变差）**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline
PYTHONPATH=src .venv/bin/python -c "
import json, glob
from collections import defaultdict
def load_dir(d):
    by_sid = defaultdict(list)
    for f in glob.glob(f'{d}/*.seglst.json'):
        sid = f.split('/')[-1].split('.')[0]
        if sid.isdigit():
            by_sid[sid] = json.load(open(f))
    return by_sid
v026 = load_dir('output/v026_pick')
v029 = load_dir('output/v029_norm')
import subprocess, tempfile
from pathlib import Path
WER = '/Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/.venv/bin/meeteval-wer'
ref = json.load(open('../data/extracted/dev/dev/ref.seglst.json'))
ref_by = {}
for r in ref: ref_by.setdefault(r['session_id'], []).append(r)
worse = []
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    rf = td / 'r.json'; rf.write_text(json.dumps(ref, ensure_ascii=False))
    for sid in v029:
        hf = td / 'h.json'; hf.write_text(json.dumps(v029[sid], ensure_ascii=False))
        subprocess.run([WER, 'tcpwer', '-r', str(rf), '-h', str(hf), '--collar', '5'], capture_output=True, check=True, cwd=td)
        e29 = int(json.load(open(td / 'h_tcpwer_per_reco.json'))[sid]['errors'])
        hf.write_text(json.dumps(v026[sid], ensure_ascii=False))
        subprocess.run([WER, 'tcpwer', '-r', str(rf), '-h', str(hf), '--collar', '5'], capture_output=True, check=True, cwd=td)
        e26 = int(json.load(open(td / 'h_tcpwer_per_reco.json'))[sid]['errors'])
        if e29 > e26: worse.append(sid)
print('反向损失 session 数:', len(worse))
print(worse[:10])
"
```

Expected: `反向损失 session 数: 0`（v028 经验：反向损失 = 0 是上线 PASS 关键）
若 > 5 → **STOP**（参考 v027 失败）

- [ ] **Step 4: Commit（仅 dev 产物）**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2
git add baseline/output/v029_norm/
git commit -m "feat: v029 dev 通过双切分 + 重尾关卡（v029 step 4）"
```

---

## Task 5: 跑 test 端 fun-asr 重排

**Files:**
- Write: `baseline/output_test/funasr_retext/{001..394}.seglst.json`
- Write: `baseline/output_test/funasr_retext/{001..394}.ts.json`

**Prerequisites:**
- Task 1/2/3/4 全部 PASS（dev 端验证通过）

- [ ] **Step 1: 跑 test 队列（394 段，约 3-4 小时）**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline
PYTHONPATH=src .venv/bin/python src/funasr_retext.py \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/data/extracted/test/test/wav \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output_test_moss \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output_test/funasr_retext
```

Expected: `INFO fun-asr retext 完成: 新增 394, 跳过 0`
预计耗时: 3-4 小时（394 段 × ~30s/段），后台运行

- [ ] **Step 2: 验证 test 产出**

```bash
ls /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output_test/funasr_retext/*.seglst.json | wc -l
ls /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output_test/funasr_retext/*.ts.json | wc -l
```

Expected: 394 / 394

- [ ] **Step 3: Commit**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2
git add baseline/output_test/funasr_retext/
git commit -m "feat: fun-asr test 重排 394 段（v029 step 5）"
```

---

## Task 6: 跑 test 三候选仲裁 + 择优 + 规范化 + 合并

**Files:**
- Write: `baseline/output_test/wa_funasr/(394)`
- Write: `baseline/output_test_v029/(394)`
- Write: `baseline/output_test_v029_norm/(394)`
- Write: `baseline/submission_v029.json`

- [ ] **Step 1: 跑 test 三候选仲裁**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline
PYTHONPATH=src diarizen_venv/bin/python src/word_arb.py \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output_test_moss \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output_test_fr3 \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/data/extracted/test/test/wav \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output_test/wa_funasr \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output_test/funasr_retext
```

Expected: `INFO WORD ARB DONE 换 N 段, 打分 M 次`

- [ ] **Step 2: 跑 pick_ensemble H+D 择优**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline
PYTHONPATH=src .venv/bin/python src/pick_ensemble.py \
  --moss output_test/wa_funasr \
  --cam output/hyp_test_v002 \
  --out output_test_v029
```

- [ ] **Step 3: 跑输出规范化**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline
PYTHONPATH=src .venv/bin/python src/normalize_output.py \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output_test_v029 \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/output_test_v029_norm
```

- [ ] **Step 4: 跑 merge_submit 合并**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline
PYTHONPATH=src .venv/bin/python src/merge_submit.py \
  --dir output_test_v029_norm \
  --out /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/submission_v029.json
```

Expected: `INFO 合并 4429 条记录, 覆盖 394 个 session`

- [ ] **Step 5: 提交前硬校验（shasum 必须异）**

```bash
shasum -a 256 \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/submission_v029.json \
  /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/submit/archive/submission_v028_20260809.json
```

Expected: 两个 sha256 **必须不同**（v014 教训：重复提交 = 白耗一次机会）
若相同 → **STOP**，检查为什么产物与 v028 相同

- [ ] **Step 6: Commit test 端全部产物**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2
git add baseline/output_test/wa_funasr/ baseline/output_test_v029/ baseline/output_test_v029_norm/ baseline/submission_v029.json
git commit -m "feat: v029 test 提交物生成（dev 16.090%→15.XXX%，双切分+重尾 PASS）"
```

---

## Task 7: 更新 SCORES + ledger + 通知人工提交

**Files:**
- Modify: `SCORES.md` 第 19 行后插入 v029 行
- Modify: `online_ledger.md` 在 v028 行前插入 v029 行
- Create: `submit/archive/submission_v029_20260809.json`

- [ ] **Step 1: 归档 v029 提交物**

```bash
cp /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/baseline/submission_v029.json \
   /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/submit/archive/submission_v029_20260809.json
```

- [ ] **Step 2: 在 SCORES.md 第 19 行（v028 行）后插入 v029 行**

读 `SCORES.md` 第 19 行上下文，用 Edit 工具插入新行。v029 行格式与 v028 类似：

```markdown
| **v029** | **v026 + fun-asr 第三源词级仲裁**（MOSS+FireRed+fun-asr 三候选，word_arb 两轮仲裁方案 C：MOSS vs FireRed → best-2 vs fun-asr，MAX_BLOCKS=4 MARGIN=0.0）| **X.XXX%** | 待提交 | — | 2026-08-09 | train82 X.XXX% / holdout24 X.XXX% / 全dev X.XXX%（≤ v026 16.090% − 0.3 点）。两道正交关卡首次同时通过：① 双切分同向改善；② 重尾关卡剔 top-3 后 +X.XXX PASS。**反向损失 = 0**（无任何 session 变差）。fun-asr 是阿里云百炼 MaaS 合规开源实例，已跑过 dev 106 段，重跑启用词级时间戳。textual axis oracle 三源上界 13.67% 已捕获 X.XXX 点。test 提交物 4429 条 / shasum <v029-sha>，与 v028 <v028-sha> 不同。脚本 `src/funasr_retext.py`（新）+ `src/word_arb.py`（扩 --third_dir） |
```

其中 `X.XXX%` 和 `shasum` 用实际跑出的数字（从 Task 6 的产出读出）。

- [ ] **Step 3: 在 online_ledger.md 的 v028 行前插入 v029 行**

读 `online_ledger.md` 第 47 行上下文，用 Edit 工具插入：

```markdown
| **v029** | 2026-08-09 | **待返回** | dev 较 v026 16.090% 改善 X.XXX 点 | v026 + **fun-asr 第三源词级仲裁**：word_arb 加 --third_dir，两轮仲裁方案 C（MOSS vs FireRed → best-2 vs fun-asr），MAX_BLOCKS=4 MARGIN=0.0。**dev 双切分 PASS + 重尾 PASS**。fun-asr 是阿里云百炼 MaaS 合规开源实例（已确认 Apache-2.0），重跑 dev 启用词级时间戳。textual axis oracle 三源上界 13.67% 捕获。test 提交物 4429 条 / shasum <v029-sha>，与 v028 不同 | `submit/archive/submission_v029_20260809.json` |
```

- [ ] **Step 4: Commit 文档**

```bash
cd /Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2
git add SCORES.md online_ledger.md submit/archive/submission_v029_20260809.json
git commit -m "ledger+scores: v029 待提交（dev 16.090%→15.XXX%，双切分+重尾 PASS）"
```

- [ ] **Step 5: 通知人工提交（agent 不擅自提交线上）**

向用户报告：
- v029 dev 改善多少（相对 v026 16.090%）
- 双切分（train82/holdout24）结果
- 重尾关卡（剔 top-3 后净收益）
- 反向损失（应为 0）
- test 提交物 shasum
- **等待人工决策：是否提交线上**

---

## Self-Review

### Spec 覆盖

| Spec 章节 | 任务 |
|---|---|
| Context（项目当前状态）| 在 Global Constraints 中体现为 `v026 16.090% / shasum 32a0fc74…` |
| 架构（v007_moss + fr_retext + funasr_retext → word_arb → pick_ensemble → normalize_output → merge_submit）| Task 1（funasr_retext）+ Task 2（word_arb 扩展）+ Task 3（pick_ensemble + normalize）+ Task 6（merge_submit） |
| 组件 1（fun-asr 词级时间戳）| Task 1 |
| 组件 2（word_arb 3 候选两轮仲裁方案 C）| Task 2 |
| 组件 3（pick_ensemble 沿用）| Task 3 Step 2 |
| 组件 4（normalize_output 沿用）| Task 3 Step 3 |
| 组件 5（提交前两道关卡）| Task 4 |
| 实现步骤（1-12）| Task 1-7 |
| 风险与缓解（5 项）| Task 3（dev 改善 STOP）、Task 4（双切分+重尾 PASS 才进 test）、Task 4 Step 3（反向损失检测）|
| 验收标准 | Task 4 全项（dev 改善 + 双切分 + 重尾 PASS + 反向损失=0）+ Task 6 Step 5（shasum 必异）|
| 提交决策（人工提交）| Task 7 Step 5 |

无遗漏。

### 占位扫描

无 "TBD" / "TODO" / "implement later" / "similar to Task N" 等占位词。所有代码块都是完整的。

### 类型一致性

| 跨任务引用 | 类型一致 |
|---|---|
| `transcribe_audio(wav, speaker_count=None)` | Task 1 用 → Task 1 用，Task 5/6 不直接调（仅重排后的产物） |
| `arbitrate_session_3way(moss, fr, third, wav)` | Task 2 定义 → Task 2 调用 |
| `out_dir/{sid}.seglst.json` | Task 1/2/3/4/5/6 全部使用 |
| `out_dir/{sid}.ts.json` | Task 1/5 使用 |
| `submission_v029.json` | Task 6 产出 → Task 7 归档 |

无类型冲突。

### 关键检查点

| 检查点 | 在哪 |
|---|---|
| dev 全量评测必须改善 ≥ 0.3 点 | Task 3 Step 1 |
| 反向损失 = 0 | Task 4 Step 3 |
| shasum 必须与 v028 不同 | Task 6 Step 5 |
| 重尾关卡 PASS | Task 4 Step 2 |
| 双切分 PASS | Task 4 Step 1 |

每个 STOP 条件都有明确判定标准，符合 CLAUDE.md "每版必填 STOP 条件"。

### 风险 STOP 位置

| 风险 | STOP 在哪 |
|---|---|
| fun-asr 词级时间戳 API 不稳定 | Task 1 Step 5（跑全 dev 106 段失败 → STOP） |
| fun-asr 文本偏差大 | Task 3 Step 1（dev 未改善 → STOP，回到 brainstorming）|
| 双切分过拟合 | Task 4 Step 1（holdout 改善 < train → STOP）|
| 重尾收益失败 | Task 4 Step 2（FAIL → STOP）|
| 反向损失大 | Task 4 Step 3（> 5 session → STOP）|
| 提交物重复 v028 | Task 6 Step 5（shasum 同 → STOP）|

风险覆盖完整。