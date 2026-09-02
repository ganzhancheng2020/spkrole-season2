"""给 word_arb.py 加可切换的打分口径（mean / sum）。

## 为什么

原实现 `score()` 返回**整段 token 的平均**对数概率，决策用 `best_mean - base_mean >= MARGIN`。
被换掉的只是几个词块，所以一个 20 token 的段里换 2 个 token，均值最多变动 2/20 ——
**有效阈值随段长反比缩放**，对长段过严、对短段过松。

理论上，比较「同一段音频的两个转写谁更可能」应当用**总对数概率**（似然比），
而非平均；平均是通用 ASR 重打分里为比较不同长度候选引入的启发式。

⚠️ 但纯总和有反向偏置：token 越少总分越高 → 偏好更短的文本（可能推高 del）。
**两种口径各有病，必须在干净 CV 集上实测，不能先验断言。**

SCORE_NORM=mean（默认，保持现行行为）| sum
"""
import io
import re
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

s = s.replace(
    'MARGIN = float(os.environ.get("MARGIN", "0.0"))',
    'MARGIN = float(os.environ.get("MARGIN", "0.0"))\n'
    '# 打分口径：mean = 整段 token 平均（原行为）；sum = 总对数概率（似然比，长度中性）\n'
    'SCORE_NORM = os.environ.get("SCORE_NORM", "mean")'
)

s = s.replace(
    '    lp = torch.log_softmax(dec.tgt_word_prj(d)[:, :-1], -1)\n'
    '    return float(lp.gather(-1, ys[:, 1:].unsqueeze(-1)).squeeze(-1).mean())',
    '    lp = torch.log_softmax(dec.tgt_word_prj(d)[:, :-1], -1)\n'
    '    tok_lp = lp.gather(-1, ys[:, 1:].unsqueeze(-1)).squeeze(-1)\n'
    '    return float(tok_lp.sum() if SCORE_NORM == "sum" else tok_lp.mean())'
)

s = s.replace(
    '"""teacher forcing 声学似然 P(text | audio)，长度归一化。"""',
    '"""teacher forcing 声学似然 P(text | audio)。\n\n'
    '    SCORE_NORM=mean 时返回 token 平均（原行为，长度归一化）；\n'
    '    =sum 时返回总对数概率（似然比口径，不被段长稀释）。\n'
    '    """'
)

io.open(p, "w", encoding="utf-8").write(s)
print("patched:", p)
