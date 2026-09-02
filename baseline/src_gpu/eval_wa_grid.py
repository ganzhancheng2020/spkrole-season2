"""评测 wa_cv* 网格：合并 4 折的 wa 输出，对 dev 完整评测，输出网格 tcpWER。
"""
import json, glob, subprocess, os
import sys
ref = json.load(open('/Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/data/extracted/dev/dev/ref.seglst.json'))
ref_sids = sorted({r['session_id'] for r in ref})
sids_by_fold = {
    0: '001,005,044,017,024,034,047,076,087,091,011,018,027,036,041,058,067,081,095,106,037,061,068,080,097,042,074'.split(','),
    1: '002,006,054,019,028,043,049,077,088,094,012,021,029,038,052,059,073,083,100,009,048,063,070,085,098,053,101'.split(','),
    2: '003,007,014,020,030,045,051,082,089,105,013,022,031,039,055,060,075,092,103,026,050,065,072,086,099,069'.split(','),
    3: '004,008,016,023,033,046,064,084,090,010,015,025,032,040,057,062,079,093,104,035,056,066,078,096,102,071'.split(','),
}
def collect(cfg):
    out = []
    for f, sids in sids_by_fold.items():
        d = f'/root/autodl-tmp/ft/wa_cv{f}_{cfg}'
        for sid in sids:
            p = f'{d}/{sid}.seglst.json'
            if not os.path.exists(p): print(f'MISSING: {p}'); continue
            out.extend(json.load(open(p)))
    return out
def evaluate(name, hyps):
    tmp = f'/tmp/wa_eval_{name}'
    os.makedirs(tmp, exist_ok=True)
    for sid in ref_sids:
        records = [r for r in hyps if r['session_id']==sid]
        if records: json.dump(records, open(f'{tmp}/{sid}.seglst.json','w'), ensure_ascii=False)
    result = subprocess.run(['/Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/.venv/bin/meeteval-wer', 'tcpwer', '-r', f'{tmp}.ref.seglst.json'.replace('.seglst.json',''), '-h', f'{tmp}.hyp.seglst.json'.replace('.seglst.json',''), '--collar', '5'], capture_output=True, text=True)
    print(f'{name}: {result.stdout.strip().split(chr(10))[-1] if result.stdout else result.stderr}')
    ref_path = '/Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/data/extracted/dev/dev/ref.seglst.json'
    ref_data = json.load(open(ref_path))
    json.dump(ref_data, open(f'{tmp}.ref.seglst.json','w'), ensure_ascii=False)
    json.dump([r for r in hyps], open(f'{tmp}.hyp.seglst.json','w'), ensure_ascii=False)
    result = subprocess.run(['/Users/jack/Desktop/竞赛/26年7月讯飞星火AI开发者大赛/说话人角色分离挑战赛_赛季2/.venv/bin/meeteval-wer', 'tcpwer', '-r', f'{tmp}.ref.seglst.json', '-h', f'{tmp}.hyp.seglst.json', '--collar', '5'], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if '%tcpWER' in line: return line.strip()
    return result.stdout.strip()
configs = [(m, b) for m in ['0.03','0.04','0.05','0.06','0.07'] for b in ['4','6','8']]
results = []
for m, b in configs:
    cfg = f'm{m}_b{b}'
    hyps = collect(cfg)
    if not hyps: continue
    r = evaluate(cfg, hyps)
    results.append((cfg, r))
    print(f'm={m} b={b}: {r}')
print('\n=== 网格汇总 ===')
for cfg, r in results: print(f'  {cfg}: {r}')
