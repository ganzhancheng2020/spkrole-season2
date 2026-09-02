import json, glob, os
def sig(p):
    R = sorted(json.load(open(p, encoding='utf-8')), key=lambda r: r['start_time'])
    return tuple((round(x['start_time'], 2), round(x['end_time'], 2)) for x in R)
n_same = n = 0
for p in sorted(glob.glob('/root/autodl-tmp/ft/hyp_sw_m3_7_0.70/[0-9]*.seglst.json')):
    sid = os.path.basename(p).split('.')[0]
    q = f'/root/autodl-tmp/ft/truecam_re/{sid}.seglst.json'
    if not os.path.exists(q): continue
    n += 1
    if sig(p) == sig(q): n_same += 1
print(f'hyp_sw 与 truecam_re 切分一致: {n_same}/{n}  (一致 => spk_relabel 只改标签)')
a = sorted(json.load(open('/root/autodl-tmp/ft/hyp_sw_m3_7_0.70/001.seglst.json', encoding='utf-8')), key=lambda r: r['start_time'])
print(f'hyp_sw 001: {len(a)} 段, 前3段 {[(round(x["start_time"],1), round(x["end_time"],1)) for x in a[:3]]}')
print(f'  说话人序列 {[x["speaker"] for x in a]}')
