"""抹掉说话人标签，只测纯文本质量（T1 主攻的验收口径）。

## 为什么需要它

tcpWER 同时惩罚「认错字」和「归错人」。要单独看文本有多好，就得把说话人这一维
消掉 —— 把所有记录归到同一个 speaker，tcpWER 退化成纯词级错误率。

2026-08-05 用这个口径测出的分解（v011 全 dev）：

    v011 dev                16.979%
    v011 dev（抹说话人）     12.390%   ← 文本成本
    ------------------------------------
    归属成本                 4.589 点

进一步拆 12.390%：sub 1122（认错，5.77%）/ 净漏词 771（3.96%）/ 对齐错 258（1.33%）。
**语音覆盖 98.6%** —— 音频基本都处理了，是听错不是没听到，所以主攻识别准确度。

⚠️ 别把这个数当成「完美归属下的 tcpWER」。那是另一个量（段级 oracle 重打标签 ≈15.3%）：
抹标签会连「说话人切分错误」的代价一起抹掉，比完美归属更宽松，两者差约 2.5 点。
这个混淆 2026-08-05 犯过一次（把 12.89% 说成「完美归属」），记在此处防止再犯。

## ⚠️ ref 必须一起抹平

只抹 hyp 会得到废数。tcpWER 对说话人做最优匹配：hyp 只剩 1 个说话人、ref 仍有 2-6 个时，
**只有一个 ref 说话人能配上，其余说话人的词全算 deletion，而 hyp 里那些词全算 insertion**。
实测这样跑出 **96.81%**（ins 8700 / del 9471）—— 纯属口径错误，不是文本差。
（2026-08-05 首版就这么错过一次，故本脚本默认连 ref 一起抹。）

## 用法

    cd baseline
    .venv/bin/python src/erase_speaker.py --dir output/v011_pick --out /tmp/v011_flat
    PYTHONPATH=src .venv/bin/python src/batch_evaluate.py \\
        --ref /tmp/v011_flat/ref_flat.seglst.json --dir /tmp/v011_flat
    # 应复现 12.39%；对不上说明口径漂了，别拿去做对比

基线（回归对照）：`output/v011_pick` → **12.39%**（ins 258 / del 1029 / sub 1122）。
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FLAT_SPEAKER = "spk1"
BASE_DIR = Path(__file__).resolve().parent.parent
DEV_REF = BASE_DIR.parent / "data/extracted/dev/dev/ref.seglst.json"
REF_FLAT_NAME = "ref_flat.seglst.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="抹掉说话人标签（纯文本口径）")
    ap.add_argument("--dir", required=True, help="输入 SegLST 目录")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--ref", default=str(DEV_REF),
                    help="要一并抹平的 ref（默认 dev ref）")
    ap.add_argument("--no-ref", action="store_true",
                    help="不生成抹平 ref —— 只在你已有同口径 ref 时才用")
    args = ap.parse_args()

    src, out_dir = Path(args.dir), Path(args.out)
    if not src.is_dir():
        logger.error("目录不存在：%s", src)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    # 只收纯 session 命名的产物：output 子目录里混有 all.hyp / *_tcpwer 等评测中间件，
    # 全 glob 会把同一 session 重复计入（oracle_pick.py 曾因此算出 127% 的假分数）
    n_sess = n_rec = 0
    for p in sorted(src.glob("*.seglst.json")):
        if not p.name.split(".")[0].isdigit():
            continue
        recs = json.loads(p.read_text(encoding="utf-8"))
        flat = [{**r, "speaker": FLAT_SPEAKER} for r in recs]
        (out_dir / p.name).write_text(
            json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8")
        n_sess += 1
        n_rec += len(flat)

    if not n_sess:
        logger.error("没收到任何 NNN.seglst.json：%s", src)
        return 1
    logger.info("hyp: %d session / %d 条 → %s（speaker 全部置为 %s）",
                n_sess, n_rec, out_dir, FLAT_SPEAKER)

    # ref 必须同口径抹平，否则说话人匹配把跨说话人的词全判成 ins+del（见文件头）
    if not args.no_ref:
        ref_path = Path(args.ref)
        if not ref_path.exists():
            logger.error("ref 不存在：%s（如已有同口径 ref 请加 --no-ref）", ref_path)
            return 1
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        flat_ref = [{**r, "speaker": FLAT_SPEAKER} for r in ref]
        dst = out_dir / REF_FLAT_NAME
        dst.write_text(json.dumps(flat_ref, ensure_ascii=False), encoding="utf-8")
        logger.info("ref: %d 条 → %s", len(flat_ref), dst)
        logger.info("评测：PYTHONPATH=src .venv/bin/python src/batch_evaluate.py "
                    "--ref %s --dir %s", dst, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
