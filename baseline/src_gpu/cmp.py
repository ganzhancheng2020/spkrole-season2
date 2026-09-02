import json
moss = json.load(open("/root/autodl-tmp/ft/cv0/pred/001.seglst.json"))
fr = json.load(open("/root/autodl-tmp/ft/fr_retext/001.seglst.json"))
print(f"cv0 MOSS 001 段数: {len(moss)}")
print(f"fr_retext 001 段数: {len(fr)}")
print(f"match: {len(moss)==len(fr)}")
print(f"MOSS speakers: {sorted(set(r['speaker'] for r in moss))}")
print(f"FR   speakers: {sorted(set(r['speaker'] for r in fr))}")
# 时间重叠检查
moss_dur = moss[-1]['end_time'] - moss[0]['start_time']
fr_dur = fr[-1]['end_time'] - fr[0]['start_time']
print(f"MOSS dur: {moss_dur:.1f}s, FR dur: {fr_dur:.1f}s")
