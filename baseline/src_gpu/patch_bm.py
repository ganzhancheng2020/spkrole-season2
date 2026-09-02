"""加 SCORE_NORM=blockmean —— 只在差异区上比较**每 token 平均** log-prob。

三种口径的病：
- `mean`（整段平均）：局部改动被整段长度稀释
- `sum` / `block`（求和）：token 越少总分越高 → 偏好更短候选，推高 del
  （block m=0 实测 del 1435→1738，全链路 +0.848，7 改善/44 恶化）
- **`blockmean`**：差异区内取平均 → **两个病同时消掉**
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

s = s.replace(
    '    if SCORE_NORM == "block":\n        return list(ids), tok_lp',
    '    if SCORE_NORM in ("block", "blockmean"):\n        return list(ids), tok_lp'
)
s = s.replace(
    '    b = float(base_lp[p:len(base_ids) - q].sum()) if len(base_ids) - q > p else 0.0\n'
    '    c = float(cand_lp[p:len(cand_ids) - q].sum()) if len(cand_ids) - q > p else 0.0\n'
    '    return c - b',
    '    nb, nc = len(base_ids) - q - p, len(cand_ids) - q - p\n'
    '    b = float(base_lp[p:len(base_ids) - q].sum()) if nb > 0 else 0.0\n'
    '    c = float(cand_lp[p:len(cand_ids) - q].sum()) if nc > 0 else 0.0\n'
    '    if SCORE_NORM == "blockmean":\n'
    '        # 差异区内取每 token 平均：去稀释（不看整段长度）且去长度偏置（不看差异区长度）\n'
    '        return (c / nc if nc > 0 else 0.0) - (b / nb if nb > 0 else 0.0)\n'
    '    return c - b'
)
s = s.replace('        if SCORE_NORM == "block":\n            b_ids, b_lp',
              '        if SCORE_NORM in ("block", "blockmean"):\n            b_ids, b_lp')
io.open(p, "w", encoding="utf-8").write(s)
print("patched blockmean")
