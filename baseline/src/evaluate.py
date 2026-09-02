"""本地评测：调根目录 meeteval 环境跑 tcpWER --collar 5。

baseline venv 不含 meeteval；用项目根 .venv（Python 3.13）的 meeteval-wer。
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MEETEVAL_WER = ROOT_DIR / ".venv" / "bin" / "meeteval-wer"
COLLAR = 5


def run_tcpwer(ref: Path, hyp: Path, collar: int = COLLAR) -> dict:
    """跑 meeteval tcpwer，返回结果 dict。"""
    if not MEETEVAL_WER.exists():
        logger.error("未找到 %s，请在项目根执行: pip install -r requirements.txt", MEETEVAL_WER)
        sys.exit(2)
    cmd = [
        str(MEETEVAL_WER), "tcpwer",
        "-r", str(ref),
        "-h", str(hyp),
        "--collar", str(collar),
    ]
    logger.info("运行: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("meeteval 失败: %s", result.stderr)
        sys.exit(3)
    out_json = hyp.parent / f"{hyp.stem}_tcpwer.json"
    data = json.loads(out_json.read_text(encoding="utf-8"))
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="参考 SegLST JSON")
    ap.add_argument("--hyp", required=True, help="预测 SegLST JSON")
    ap.add_argument("--collar", type=int, default=COLLAR)
    args = ap.parse_args()
    r = run_tcpwer(Path(args.ref), Path(args.hyp), args.collar)
    print(f"tcpWER = {r['error_rate']:.4%}  (errors={r['errors']}, length={r['length']})")
    print(f"  missed_speaker={r.get('missed_speaker')}, "
          f"falarm_speaker={r.get('falarm_speaker')}, "
          f"scored_speaker={r.get('scored_speaker')}")


if __name__ == "__main__":
    main()
