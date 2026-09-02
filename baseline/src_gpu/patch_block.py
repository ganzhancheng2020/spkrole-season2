"""给 word_arb.py 加 SCORE_NORM=block —— 只在**被换块**的 token 位置上比较。

## 诊断依据

裁判在 988 个可判差异块上只有 **64.6%** 与 oracle 一致。
而现有两种口径都在**整段**上算分：
- `mean`：整段 token 平均 → 20 token 的段里换 2 个，均值最多动 2/20，被稀释
- `sum`：整段总和 → 引入长度偏置（偏好更短文本），实测更差 0.046 点

两者都把**没变的那些 token** 算进了比较。它们在两个候选里几乎同分，是纯噪声。

## 本口径

对 base（全 MOSS）与候选各自取 **token 级 log-prob 向量**，
按 id 序列求**公共前缀 / 公共后缀**，只在中间这段（= 实际差异区）求和作差。
不需要 tokenizer 偏移量计算，因为前后缀是逐 id 比出来的。

SCORE_NORM=mean（默认，现行）| sum | block
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

# 1) score 返回 (ids, 逐 token log-prob)，供三种口径共用
s = s.replace(
    '    lp = torch.log_softmax(dec.tgt_word_prj(d)[:, :-1], -1)\n'
    '    tok_lp = lp.gather(-1, ys[:, 1:].unsqueeze(-1)).squeeze(-1)\n'
    '    return float(tok_lp.sum() if SCORE_NORM == "sum" else tok_lp.mean())',
    '    lp = torch.log_softmax(dec.tgt_word_prj(d)[:, :-1], -1)\n'
    '    tok_lp = lp.gather(-1, ys[:, 1:].unsqueeze(-1)).squeeze(-1)[0]\n'
    '    if SCORE_NORM == "block":\n'
    '        return list(ids), tok_lp\n'
    '    return float(tok_lp.sum() if SCORE_NORM == "sum" else tok_lp.mean())'
)

# 2) 块级差分：公共前后缀之外求和作差
s = s.replace(
    'def arbitrate_session(',
    'def block_delta(base_ids, base_lp, cand_ids, cand_lp) -> float:\n'
    '    """只在差异区（去掉公共前缀/后缀）上比较总 log-prob。"""\n'
    '    n = min(len(base_ids), len(cand_ids))\n'
    '    p = 0\n'
    '    while p < n and base_ids[p] == cand_ids[p]:\n'
    '        p += 1\n'
    '    q = 0\n'
    '    while q < n - p and base_ids[len(base_ids) - 1 - q] == cand_ids[len(cand_ids) - 1 - q]:\n'
    '        q += 1\n'
    '    b = float(base_lp[p:len(base_ids) - q].sum()) if len(base_ids) - q > p else 0.0\n'
    '    c = float(cand_lp[p:len(cand_ids) - q].sum()) if len(cand_ids) - q > p else 0.0\n'
    '    return c - b\n'
    '\n'
    '\n'
    'def arbitrate_session('
)

# 3) 决策处按口径分流
s = s.replace(
    '''        keep_moss = tuple([0] * len(di))  # 全取 MOSS 的那一种组合
        best: tuple[float, list[str], tuple[int, ...]] = (float("-inf"), ta, keep_moss)
        for combo in itertools.product([0, 1], repeat=len(di)):''',
    '''        keep_moss = tuple([0] * len(di))  # 全取 MOSS 的那一种组合
        best: tuple[float, list[str], tuple[int, ...]] = (float("-inf"), ta, keep_moss)
        if SCORE_NORM == "block":
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
                dv = block_delta(b_ids, b_lp, c_ids, c_lp)
                if dv > best[0]:
                    best = (dv, toks, combo)
            if best[2] != keep_moss and best[0] >= MARGIN:
                nsw += 1
                res.append({**a, "words": " ".join(best[1])})
            else:
                res.append(dict(a))
            continue
        for combo in itertools.product([0, 1], repeat=len(di)):'''
)
io.open(p, "w", encoding="utf-8").write(s)
print("patched")
