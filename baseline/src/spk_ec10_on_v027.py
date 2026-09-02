"""v027 结果 + 15 自信段 ec1.0 重标签。文本保持 v027，只改 speaker。
"""
import json, glob, os, sys
sys.path.insert(0, 'src')
from apply_diar import assign_segment
out_dir = 'output/ec10_on_v027'
os.makedirs(out_dir, exist_ok=True)
confident = {'009','012','013','016','020','029','037','040','066','071','074','085','098','099','102'}
for p in sorted(glob.glob('output/v027_dev/[0-9]*.seglst.json')):
    sid = p.split('/')[-1].replace('.seglst.json','')
    recs = json.load(open(p))
    if sid in confident:
        diar = json.load(open(f'output/diar_ec1.0/{sid}.diar.json'))['segments']
        out = [assign_segment(r, diar)[0] for r in recs]
    else:
        out = recs
    json.dump(out, open(f'{out_dir}/{sid}.seglst.json','w'), ensure_ascii=False)
print('done')
