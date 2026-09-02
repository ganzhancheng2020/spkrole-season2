"""加 SCORE_NORM=znorm —— 保留整段打分，但阈值按块长的噪声尺度校准。

## 依据

已实测排除的三种口径：
- `mean` 整段平均（现行，−0.077）
- `sum` 整段求和 → 长度偏置，最优 −0.031
- `block` / `blockmean` 差异区打分 → **+0.848 / +1.049**，
  因为**整段上下文是信号**，砍掉它反而更差

所以打分必须留在整段上。问题出在**阈值**：
每 token 的 log-prob 差有方差 σ²，在 k 个变动 token 上求和后噪声 ∝ √k。
固定 MARGIN 对所有 (块长, 段长) 一视同仁 → **短块被过度轻信、长块被过度苛求**。

判据改为信噪比：**z = Δ整段总和 / √k ≥ MARGIN**（k = 实际变动的 token 数）。

这同时是对一处未解矛盾的检验：「漏 296 词 > 误 154 词」暗示裁判太保守，
但放松全局 margin 反而更差 —— 若噪声尺度确实随块长变化，
**两个方向各有得失正是全局标量的必然结果**，长度自适应应能同时降低漏与误。
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

s = s.replace(
    '    tok_lp = lp.gather(-1, ys[:, 1:].unsqueeze(-1)).squeeze(-1)\n'
    '    return float(tok_lp.sum() if SCORE_NORM == "sum" else tok_lp.mean())',
    '    tok_lp = lp.gather(-1, ys[:, 1:].unsqueeze(-1)).squeeze(-1)[0]\n'
    '    if SCORE_NORM == "znorm":\n'
    '        return list(ids), tok_lp\n'
    '    return float(tok_lp.sum() if SCORE_NORM == "sum" else tok_lp.mean())'
)

s = s.replace(
    'def arbitrate_session(',
    'def z_delta(base_ids, base_lp, cand_ids, cand_lp) -> float:\n'
    '    """整段总和之差 / √k。k = 变动 token 数（去掉公共前后缀后的长度）。\n\n'
    '    分子留在整段上（保住上下文信号），分母按噪声尺度归一（消掉块长依赖）。\n'
    '    """\n'
    '    import math\n'
    '    n = min(len(base_ids), len(cand_ids))\n'
    '    p = 0\n'
    '    while p < n and base_ids[p] == cand_ids[p]:\n'
    '        p += 1\n'
    '    q = 0\n'
    '    while q < n - p and base_ids[len(base_ids) - 1 - q] == cand_ids[len(cand_ids) - 1 - q]:\n'
    '        q += 1\n'
    '    k = max(len(base_ids) - q - p, len(cand_ids) - q - p, 1)\n'
    '    return float(cand_lp.sum() - base_lp.sum()) / math.sqrt(k)\n'
    '\n'
    '\n'
    'def arbitrate_session('
)

s = s.replace(
    '''        keep_moss = tuple([0] * len(di))  # 全取 MOSS 的那一种组合
        best: tuple[float, list[str], tuple[int, ...]] = (float("-inf"), ta, keep_moss)
        for combo in itertools.product([0, 1], repeat=len(di)):''',
    '''        keep_moss = tuple([0] * len(di))  # 全取 MOSS 的那一种组合
        best: tuple[float, list[str], tuple[int, ...]] = (float("-inf"), ta, keep_moss)
        if SCORE_NORM == "znorm":
            b_ids, b_lp = score(enc, mask, "".join(ta))
            ncalls += 1
            best = (0.0, ta, keep_moss)
            for combo in itertools.product([0, 1], repeat=len(di)):
                toks = []
                for t, blk in enumerate(bl):
                    if blk[0] == "eq":
                        toks += blk[1]
                    elif t in di:
                        toks += blk[2] if combo[di.index(t)] else blk[1]
                    else:
                        toks += blk[1]
                ncalls += 1
                c_ids, c_lp = score(enc, mask, "".join(toks))
                zv = z_delta(b_ids, b_lp, c_ids, c_lp)
                if zv > best[0]:
                    best = (zv, toks, combo)
            if best[2] != keep_moss and best[0] >= MARGIN:
                nsw += 1
                res.append({**a, "words": " ".join(best[1])})
            else:
                res.append(dict(a))
            continue
        for combo in itertools.product([0, 1], repeat=len(di)):'''
)
io.open(p, "w", encoding="utf-8").write(s)
print("patched znorm")
