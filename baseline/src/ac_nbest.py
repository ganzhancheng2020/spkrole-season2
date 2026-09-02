"""N 候选声学排序仲裁：在 FireRed 声学模型下给「基准 + n-best」全部候选打分，取最优。

## 为什么需要它

`ac_test.py` 只做 2 候选（MOSS vs FireRed 1-best）仲裁，产出 v024（线上 0.15903）。
但 2026-08-08 的上界算术显示：**两侧仲裁全做到 oracle 也只有 0.951 点空间**，
而够榜首需 0.93 点 → 要 98% 捕获率，实测仅 46%。
**唯一能扩大 oracle 本身的手段是增加候选数**（见 `logs/2026-08-08.md` §三、§四）。

10 段子集实测（编辑距离口径）：oracle 从 `MOSS+1best` 的 25.355%
扩到 `MOSS+4best` 的 **24.262%（+1.093 点）**，与整个剩余空间同量级。

## 打分口径（与 ac_test.py 完全一致，保证可比）

teacher forcing：裁段音频 → Conformer 编码 → 强制解码候选 token → 每步 log-prob **取均值**
（长度归一化）。所有候选在**同一声学模型、同一段音频**下打分。

⚠️ **已知偏置**：FireRed 的候选由 beam search 最大化该分数，天然分高；
基准（MOSS）文本是「外人」。故**不是取全局 argmax，而是要求领先基准 ≥ margin 才换**，
margin 在 train 上校准。这与 ac_test.py 一致，v024 已验证 8 个 margin 全过双切分。

用法：
    cd baseline
    diarizen_venv/bin/python src/ac_nbest.py <base_dir> <nbest_dir> <wav_dir> <out_dir> [margin]

`<nbest_dir>` 内为 `{sid}.nbest.json` = `[["段0文本","段1文本",…], …]`
（外层候选、内层段），由 `fr_nbest.py` 生成。
"""
import glob
import json
import logging
import os
import sys
import tempfile

import soundfile as sf  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]

# FireRed 代码与权重路径可用环境变量覆盖。
# ⚠️ 默认值在 /tmp，**重启会丢**（HANDOFF「踩过的坑」已列）——长期使用请设这两个变量
#    指向项目内目录，例如 export FIRERED_HOME=.../baseline/third_party/FireRedASR2S
FIRERED_HOME = os.environ.get("FIRERED_HOME", "/tmp/FireRedASR2S")  # noqa: S108 — 默认值仅为便利，风险已在上方注明
FIRERED_MODEL = os.environ.get("FIRERED_MODEL", "/tmp/FireRedASR2-AED")  # noqa: S108
sys.path.insert(0, FIRERED_HOME)
from fireredasr2s.fireredasr2.asr import (  # type: ignore[import-not-found]  # noqa: E402
    FireRedAsr2,
    FireRedAsr2Config,
)

MIN_SEG_SEC = 0.2
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_m = FireRedAsr2.from_pretrained(
    'aed', FIRERED_MODEL,
    FireRedAsr2Config(use_gpu=False, return_timestamp=False))
_model, _tokz, _feat = _m.model, _m.tokenizer, _m.feat_extractor
_dec = _model.decoder


@torch.no_grad()
def score_segment(enc, enc_mask, text: str) -> float:
    """teacher forcing 下 P(text|该段音频) 的 per-token 平均 log-prob。"""
    if not text:
        return -99.0
    _, ids = _tokz.tokenize(text)
    if not ids:
        return -99.0
    ys = torch.tensor([[_dec.sos_id] + list(ids)]).long()
    tm = _dec.ignored_target_position_is_0(ys, _dec.pad_id)
    d = _dec.dropout(_dec.tgt_word_emb(ys) * _dec.scale + _dec.positional_encoding(ys))
    for layer in _dec.layer_stack:
        d = layer.forward(d, enc, tm, enc_mask, cache=None)
    d = _dec.layer_norm_out(d)
    lp = torch.log_softmax(_dec.tgt_word_prj(d)[:, :-1], -1)
    return float(lp.gather(-1, ys[:, 1:].unsqueeze(-1)).squeeze(-1).mean())


@torch.no_grad()
def score_all(wav: str, segs: list, cand_sets: list) -> list:
    """逐段给全部候选打分。cand_sets[i] = 该段候选文本列表，[0] 恒为基准。"""
    out = []
    x, sr = sf.read(wav, dtype='float32')
    if x.ndim > 1:
        x = x.mean(axis=1)
    for (st, en), cands in zip(segs, cand_sets):
        # 候选全相同 → 无需打分（也避免把偏置引入无意义的比较）
        if len(set(cands)) <= 1:
            out.append([0.0] * len(cands))
            continue
        a, b = int(st * sr), int(en * sr)
        clip = x[max(0, a):min(len(x), b)]
        if len(clip) < int(MIN_SEG_SEC * sr):
            out.append([0.0] * len(cands))
            continue
        with tempfile.NamedTemporaryFile(suffix='.wav') as fh:
            sf.write(fh.name, clip, sr)
            feats, lens, _, _, _ = _feat([fh.name], ['c'])
        enc, _, enc_mask = _model.encoder(feats, lens)
        out.append([score_segment(enc, enc_mask, t) for t in cands])
    return out


def main() -> int:
    if len(sys.argv) < 5:
        logger.error("参数不足。用法见文件头 docstring。")
        return 2
    base_dir, nb_dir, wav_dir, out_dir = sys.argv[1:5]
    margin = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
    os.makedirs(out_dir, exist_ok=True)
    preds = sorted(p for p in glob.glob(f'{base_dir}/*.seglst.json')
                   if os.path.basename(p).split('.')[0].isdigit())
    nsw = 0
    for i, p in enumerate(preds, 1):
        sid = os.path.basename(p).split('.')[0]
        o = f'{out_dir}/{sid}.seglst.json'
        if os.path.exists(o):
            continue
        base = json.load(open(p))
        nbf = f'{nb_dir}/{sid}.nbest.json'
        # 缺候选或段数不一致 → 原样保留基准，绝不静默错位
        nb = []
        if os.path.exists(nbf):
            nb = [c for c in json.load(open(nbf)) if len(c) == len(base)]
        if not nb:
            json.dump(base, open(o, 'w'), ensure_ascii=False, indent=2)
            continue
        segs = [(x['start_time'], x['end_time']) for x in base]
        cand_sets = [[''.join(base[j]['words'].split())]
                     + [''.join(c[j].split()) for c in nb]
                     for j in range(len(base))]
        scores = score_all(f'{wav_dir}/{sid}.wav', segs, cand_sets)
        res = []
        for j, x in enumerate(base):
            ss, cands = scores[j], cand_sets[j]
            k = max(range(len(ss)), key=lambda t: ss[t])
            # 只有领先基准 ≥ margin 才换（抵消 FireRed 候选的自评偏置）
            if k != 0 and ss[k] - ss[0] >= margin and ss[0] != 0.0:
                nsw += 1
                res.append({**x, 'words': ' '.join(cands[k])})
            else:
                res.append(dict(x))
        json.dump(res, open(o, 'w'), ensure_ascii=False, indent=2)
        if i % 20 == 0:
            logger.info('[%d/%d] 换 %d 段', i, len(preds), nsw)
    logger.info('AC NBEST DONE 换 %d 段 (margin=%.3f)', nsw, margin)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
