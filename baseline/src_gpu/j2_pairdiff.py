import json, os, sys, itertools, glob
base="/root/autodl-tmp/ft"
dirs=[d for d in sys.argv[1:]]
def load(d):
    recs={}
    for f in glob.glob(os.path.join(base,d,"*.seglst.json")):
        try:
            obj=json.load(open(f))
        except Exception as e:
            continue
        segs=obj["segments"] if isinstance(obj,dict) and "segments" in obj else obj
        for s in segs:
            k=(s.get("session_id"), round(float(s.get("start_time")),3), round(float(s.get("end_time")),3))
            recs[k]=s.get("words","")
    return recs
L={d:load(d) for d in dirs}
for d in dirs:
    print(d, "nsegs=", len(L[d]))
print("--- pairwise (common keys, words differ) ---")
for a,b in itertools.combinations(dirs,2):
    A,B=L[a],L[b]
    common=set(A)&set(B)
    diff=sum(1 for k in common if A[k]!=B[k])
    onlyA=len(set(A)-set(B)); onlyB=len(set(B)-set(A))
    print(f"{a} vs {b}: common={len(common)} worddiff={diff} onlyA={onlyA} onlyB={onlyB}")
