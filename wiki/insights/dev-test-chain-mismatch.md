---
slug: dev-test-chain-mismatch
type: 硬结论
status: 🔴 已证实（2026-08-25，文本指纹 + 端到端实测）
family: 测量有效性 / 基础设施
versions: 影响 dev CV 全部跨分支测量
date: 2026-08-25
related: [[cv-gate-calibrated]] [[moss-branch-earns-nothing]] [[routing-gap-is-attribution]] [[cam-branch-never-arbitrated]] [[dev-online-transfer]]
---

# dev CV 的「CAM 分支」和 test 的不是同一个东西（差 6.021 点）

## 证据链

**血缘鉴定**（逐段文本 + 切分指纹，非文档推断）：

| | dev | test |
|---|---|---|
| `cam_re` 的上游 | **`v026_pick`（集成输出）**，文本 106/106 一致 | **`hyp_test_v002` / `output_test`（fun-asr）**，文本 394/394 一致 |
| session 001 文本 | 「没**有**没**有**」= v026_pick | 「未删**减**版」= fun-asr |
| 对照 | fun-asr 是「没没」 | MOSS 是「未删**节**版」 |

即：**dev 的「CAM 分支」是 pick 集成的产物（里面含 MOSS 内容），
test 的「CAM 分支」是纯 fun-asr 文本。**

## 量化损害

在 dev 上按 test 的构造重建真 CAM 分支（`spk_relabel(hyp_sw_m3_7_0.70)`）：

| dev 上的「CAM 分支」 | tcpWER |
|---|---|
| `cam_re`（= relabel(v026_pick)）| **15.797%** ← 历史全部分析用的 |
| `truecam_re`（= relabel(funasr+CAM)，与 test 同构造）| **21.818%** |
| **差距** | **6.021 点** |

## 被推翻的结论

### 1. 「MOSS 分支不值钱」—— 完全反了

| | 用假 CAM | 用真 CAM |
|---|---|---|
| MOSS 16.810% vs CAM | CAM 好 1.013 点 → 判「MOSS 可删」 | **MOSS 好 5.008 点** |

[[moss-branch-earns-nothing]] 的核心数字作废。**MOSS 是强分支，CAM 是弱分支。**
之所以看反，是因为 dev 的「CAM」里**偷偷含着 MOSS 内容** ——
拿一个含 MOSS 的东西去证明「MOSS 可以删」，是循环论证。

### 2. 路由轴的全部测量口径不对

| | 假 CAM | 真 CAM |
|---|---|---|
| oracle 路由 | 13.750% | 14.228% |
| 相对最优单支的空间 | 2.047 点 | **2.581 点** |
| 剔 top3 后 | 1.383 点 | **1.568 点** |
| CAM 赢 / MOSS 赢 | 44 / 55（42%）| **29 / 73（28%）** |

空间仍在（甚至更大），但**类别分布明显不同**。
→ **5 个路由机制（手工规则 / 轮廓系数 / 塌缩检测 / 变化点 / 段余量）
全部是在错误的分支对上评测的**，从未在 test 真正面对的问题上被测过。
**这极可能就是路由轴 dev 结论从不转移的原因。**

### 3. `bench.py` 的基线本身失真

`_cv_final = pick(_cv_wa, cam_re)` —— CAM 侧是假的。
故四关是在一条**与出货链路不同的链路**上做验收。

⚠️ **但不是全部作废**：v048/v051 是**分支内**改进
（word_arb 参数、CAM 侧补仲裁、短段改写），这类改动在两种构造下方向一致，
所以 v051 四关全过 + 线上 −0.00118 + 预报误差 0.00017 依然成立。
**失真主要打击「跨分支」类结论（路由、选源、分支消融）。**

## 为什么一直没发现

从未做过**血缘鉴定** —— 只读脚本与文档（`build_v051.sh`、insight 页），
而文档里「CAM 分支文本是 fun-asr」这句**对 test 成立、对 dev 不成立**。
**同一个名字 `cam_re` 在两侧指向不同构造的对象。**

> **教训：目录名不是契约。跨 dev/test 比较前必须用内容指纹确认两侧同构造，
> 不能靠名字或文档。** 廉价做法：取每段文本拼成指纹，与各候选上游逐段比对，
> 命中 106/106 才算认亲。

## 修复方向

重建**忠实 dev CV**：`truecam_re → FireRed 重转写 → word_arb(0.05/6)`
得到与 `test_cam_wa` 同构造的 CAM 分支，再与 MOSS 分支 pick，
用它替换 `bench.py` 的基线。修好之后，路由轴的 5 个机制
**值得在正确口径下复测** —— 它们的「死」目前是无效判决。
