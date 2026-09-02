import io
p="/root/autodl-tmp/ft/spk_relabel.py"
s=io.open(p,encoding="utf-8").read()
a=s.replace("WIN_SEC = 1.5", 'WIN_SEC = float(os.environ.get("WIN_SEC", "1.5"))',1)
if a==s: raise SystemExit("WIN_SEC 替换未命中")
b=a.replace("WIN_SHIFT = 0.75", 'WIN_SHIFT = float(os.environ.get("WIN_SHIFT", "0.75"))',1)
if b==a: raise SystemExit("WIN_SHIFT 替换未命中")
io.open(p,"w",encoding="utf-8").write(b)
print("patched OK")
