"""给 FireRedASR2S 的 asr.py 加 FR_BF16 开关（默认关，不影响 AED/word_arb）。

## 为什么

`fireredasr_llm.py` 的加载器注释写明 "Training use torch.bfloat16"，用 bf16 载入 Qwen2。
但 `asr.py` 在 use_half=True 时对**整个模型**调 `.half()`（fp16），覆盖掉 bf16 →
Qwen2 数值溢出，每段输出同一个 '%'。

而 use_half=False 又会让编码器留在 fp32（2.7G），加上 bf16 的 Qwen2（15.2G）
在 24G 卡上 OOM。

解法：让转换精度可选 bf16 —— bf16 与 fp32 同数值范围（不溢出），
且编码器减半到 1.4G，总占用约 18G，可容。

⚠️ 默认 FR_BF16 未设置时行为与原版完全一致（fp16），
AED 侧（word_arb 依赖）不受影响。
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()


def sub(old, new, tag):
    global s
    if old not in s:
        raise SystemExit(f"❌ 未命中：{tag}")
    s = s.replace(old, new, 1)
    print(f"✅ {tag}")


if "FR_BF16" in s:
    print("已打过补丁，跳过")
    raise SystemExit(0)

sub("import time", "import os as _os\nimport time\n\n"
    "# 非空时模型与特征统一转 bfloat16（LLM 变体需要；默认空 = 原版 fp16 行为）\n"
    "_FR_BF16 = bool(_os.environ.get('FR_BF16', ''))", "加 FR_BF16 开关")

sub("                    self.model.half()",
    "                    self.model.bfloat16() if _FR_BF16 else self.model.half()",
    "模型转换精度")

s2 = s.replace("                    feats = feats.half()",
               "                    feats = feats.bfloat16() if _FR_BF16 else feats.half()")
n = s.count("                    feats = feats.half()")
if n == 0:
    raise SystemExit("❌ 未命中：特征转换精度")
s = s2
print(f"✅ 特征转换精度（{n} 处）")

io.open(p, "w", encoding="utf-8").write(s)
print("ALL_PATCHED")
