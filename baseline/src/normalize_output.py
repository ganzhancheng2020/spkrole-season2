"""输出层规范化：把 ref 标注从不使用的字符改写成它使用的等价写法。

## 为什么这是「必赚」而不是「调参」

2026-08-09 第一性原理复盘发现：v026 的 11.652% 纯文本误差里有一块**根本不是识别失败**，
而是**书写约定不匹配**。dev ref 实测：

| 我们输出 | ref 中出现次数 |
|---|---|
| 哦 ×33 / 喔 | **0 / 0**（ref 一律写「噢」53 次）|
| 哎 ×13 / 诶 ×5 | **0 / 0**（ref 一律写「唉」36 次）|
| 阿拉伯数字 ×5 | **0**（ref 全用中文数字）|

**ref 里出现 0 次的字符，我们每输出一次就必然错一次。** 换成 ref 的写法后：
- 最坏情况：还是错（本来也错），**不可能弄坏本来对的**
- 最好情况：直接变对

对照组证明这是约定而非巧合：`嗯/呃` ref 都用（288/48）、`吗/嘛/吧` 都用 —— 那些是**真区分，不能动**。

## ⚠️ 唯一的主观判断：只映射「语气词 / 数字 / 异体字」，内容词一律不碰

dev 里「ref 出现 0 次」的字符共 39 种 103 次，但其中 `嘟/归/哄/软/谷/稻/孟/军/采/艳` 等
是 **dev 特有人名地名的识别错误**。按 dev 对齐结果映射它们 = 纯拟合 dev 内容，**不会迁移到 test**。
故本表**只收语气词、数字、异体/敬语**三类，且每一项都已核实 `ref_count == 0`。

## 为什么它天然通过重尾关卡

「噢」分布在几十个 session、数字散落各处 —— 收益**由构造就是分散的**，不依赖少数幸运段。
这正是 v011（迁移率 15%）与 v027（线上倒退）栽掉的那个形状的反面。
见 `wiki/insights/dev-online-transfer.md`。

用法：
    cd baseline
    .venv/bin/python src/normalize_output.py output/v026_pick output/v028_dev
    PYTHONPATH=src .venv/bin/python src/batch_evaluate.py --dir output/v028_dev
    .venv/bin/python src/heavy_tail_gate.py output/v026_pick output/v028_dev
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 每一项都已核实：源字符在 dev ref 中出现 **0 次**，目标字符在 ref 中大量存在。
# 改动前请重跑核对（见模块 docstring 的表）。**不要往这里加内容词。**
CHAR_MAP: dict[str, str] = {
    # 语气词：ref 只写「噢」「唉」
    "哦": "噢", "喔": "噢",
    "哎": "唉", "诶": "唉",
    # 敬语：ref 只写「你」（261 次），从不写「您」
    "您": "你",
    # 数字：ref 全用中文数字，无一个阿拉伯数字
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
}


def normalize(words: str) -> tuple[str, int]:
    """按字符改写，返回 (新文本, 改动字符数)。

    ⚠️ **纯数字 token 必须拆成多个单字 token，英文 token 必须保持整体。**
    dev ref 实测：多字 token 只有 6 种，**全是英文**（`ip` `ok` `oppo` `ccpu` `pro` `gdp`），
    **一个多位数都没有**；数字一律写成分开的单字（`五 十` / `一 百` / `二 零 二 二`）。

    故 `"20"` 拆成 `"二 零"` 是**严格占优**的：
    - 对 ref 的 `二 十`：拆前 1 替换 + 1 漏 = 2 错；拆后「二」命中 = 1 错
    - 对 ref 的 `二 零`（年份/编号读法）：拆后全中 = 0 错

    该分支 dev 上未被充分覆盖（dev 仅 14 个多字 token），是靠 ref 词表结构推出来的。
    """
    out = []
    n = 0
    for tok in words.split():
        if tok.isdigit():
            # 纯数字：逐位转中文并**拆成独立 token**
            chars = [CHAR_MAP.get(ch, ch) for ch in tok]
            n += len(tok)
            out.extend(chars)
            continue
        if tok.isascii() and tok.isalpha():
            out.append(tok)      # 英文整词保持不动（ref 也是整词）
            continue
        new = "".join(CHAR_MAP.get(ch, ch) for ch in tok)
        n += sum(1 for a, b in zip(tok, new) if a != b)
        out.append(new)
    return " ".join(out), n


def renumber_speakers(recs: list[dict]) -> tuple[list[dict], bool]:
    """把一场内的说话人重编号为连续的 spk1..spkN（按首次出现顺序）。

    上游的归属改写（spk_relabel / spk_reassign / spk_short_relabel）会把段在说话人
    之间搬动，某个说话人被搬空时就留下编号空洞（实测出现过 [1,2,3,4,6]）。
    cam_split_verify 的完整性自检要求连续编号，遇到空洞即中止。

    重编号是安全的：tcpWER 做说话人全局最优置换，同一场内一致改名不影响分数。
    对编号本已连续的输入是空操作（原样返回，changed=False）。
    """
    labels = {r["speaker"] for r in recs}
    nums = sorted(int(x[3:]) for x in labels
                  if x.startswith("spk") and x[3:].isdigit())
    # 已经是连续的 spk1..spkN 就一个字都不动 —— 自检只要求编号连续，
    # 不要求「首个出现的说话人必须是 spk1」，过度改名会平白改变产物哈希。
    if len(nums) == len(labels) and nums == list(range(1, len(nums) + 1)):
        return recs, False
    order: dict[str, str] = {}
    for r in sorted(recs, key=lambda x: float(x["start_time"])):
        if r["speaker"] not in order:
            order[r["speaker"]] = f"spk{len(order) + 1}"
    return [{**r, "speaker": order[r["speaker"]]} for r in recs], True


def main() -> int:
    if len(sys.argv) < 3:
        logger.error("用法见模块 docstring：%s", __doc__)
        return 2
    in_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = sorted(in_dir.glob("[0-9]*.seglst.json"))
    if not preds:
        logger.error("输入目录为空：%s", in_dir)
        return 1

    total = 0
    touched = 0
    renumbered = 0
    for p in preds:
        with open(p, encoding="utf-8") as fh:
            recs = json.load(fh)
        out = []
        n_sess = 0
        for r in recs:
            new_words, n = normalize(r["words"])
            n_sess += n
            out.append({**r, "words": new_words})
        total += n_sess
        if n_sess:
            touched += 1
        out, changed = renumber_speakers(out)
        renumbered += changed
        with open(out_dir / p.name, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)

    logger.info("共 %d 个 session，改写 %d 个字符，涉及 %d 个 session（%.1f%%）→ %s",
                len(preds), total, touched, touched / len(preds) * 100, out_dir)
    if renumbered:
        logger.info("说话人编号规范化：%d 个 session 重编号为连续 spk1..spkN", renumbered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
