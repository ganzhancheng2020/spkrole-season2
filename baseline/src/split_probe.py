import json, glob, argparse, os
"""段拆分探针：只对 MOSS 漏人 + 长段的 41 段做拆分。
用 ec1.0 diar 边界，把 MOSS 长段按说话人重叠拆分，文本按比例切分。
"""
def main():
    import sys
    sys.path.insert(0, 'src')
    from apply_diar import assign_split
    out_dir = 'output/split_probe'
    os.makedirs(out_dir, exist_ok=True)
    # 41 段候选 + 长段阈值
    ref = json.load(open('../data/extracted/dev/dev/ref.seglst.json'))
    ref_nspk = {}
    for r in ref: ref_nspk[r['session_id']] = ref_nspk.get(r['session_id'], 0) or len({x['speaker'] for x in ref if x['session_id']==r['session_id']})
    # 简化：所有段都过，但只在长段上切
    for p in sorted(glob.glob('output/v007_moss/[0-9]*.seglst.json')):
        sid = p.split('/')[-1].replace('.seglst.json','')
        recs = json.load(open(p))
        diar = json.load(open(f'output/diar_ec1.0/{sid}.diar.json'))['segments']
        out_recs = []
        for r in recs:
            dur = r['end_time'] - r['start_time']
            if dur >= 4.0:
                out_recs.extend(assign_split(r, diar))
            else:
                out_recs.append(r)
        json.dump(out_recs, open(f'{out_dir}/{sid}.seglst.json','w'), ensure_ascii=False)
    print('done')
if __name__ == '__main__': main()
