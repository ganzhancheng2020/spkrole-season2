"""在**原版** word_arb.py 上加 SCORE_NORM=znorm（阈值按块长噪声尺度校准）。

每处替换都断言命中，替换不到直接报错退出 —— 上一版补丁无条件打印 "patched"
导致连续两次假成功，此处修掉。
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()


def sub(old, new, tag):
    global s
    if old not in s:
        raise SystemExit(f"❌ 替换点未命中：{tag}")
    s = s.replace(old, new, 1)
    print(f"✅ {tag}")


sub('MARGIN = float(os.environ.get("MARGIN", "0.0"))',
    'MARGIN = float(os.environ.get("MARGIN", "0.0"))\n'
    '# mean（默认，现行出货）| znorm：分子仍是整段总和之差（保住上下文信号），\n'
    '# 分母 √k 按变动 token 数归一，消掉「短块被轻信、长块被苛求」的偏差\n'
    'SCORE_NORM = os.environ.get("SCORE_NORM", "mean")',
    "加 SCORE_NORM 开关")

sub('    return float(lp.gather(-1, ys[:, 1:].unsqueeze(-1)).squeeze(-1).mean())',
    '    tok_lp = lp.gather(-1, ys[:, 1:].unsqueeze(-1)).squeeze(-1)[0]\n'
    '    if SCORE_NORM == "znorm":\n'
    '        return list(ids), tok_lp\n'
    '    return float(tok_lp.mean())',
    "score 支持返回 token 级向量")

sub('def arbitrate_session(',
    'def z_delta(base_ids, base_lp, cand_ids, cand_lp) -> float:\n'
    '    """整段总和之差 / √k，k = 变动 token 数（去掉公共前后缀）。"""\n'
    '    import math\n'
    '    n = min(len(base_ids), len(cand_ids))\n'
    '    a = 0\n'
    '    while a < n and base_ids[a] == cand_ids[a]:\n'
    '        a += 1\n'
    '    b = 0\n'
    '    while b < n - a and base_ids[len(base_ids)-1-b] == cand_ids[len(cand_ids)-1-b]:\n'
    '        b += 1\n'
    '    k = max(len(base_ids) - b - a, len(cand_ids) - b - a, 1)\n'
    '    return float(cand_lp.sum() - base_lp.sum()) / math.sqrt(k)\n'
    '\n'
    '\n'
    'def arbitrate_session(',
    "加 z_delta")

sub('''        keep_moss = tuple([0] * len(di))  # 全取 MOSS 的那一种组合
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
        for combo in itertools.product([0, 1], repeat=len(di)):''',
    "决策处分流")

io.open(p, "w", encoding="utf-8").write(s)
print("ALL_PATCHED")
