# 各数据层角色定义

> 本项目数据层少（dev + test 两层），不像通用 solver 优化题有分层 test case。角色定义仍集中记一处，取舍判断时读它。

## 层角色表

| 层 | 角色 | 怎么用 |
|----|------|--------|
| dev (106 session) | 本地可评测层 | 决定一版值不值得提交——dev tcpWER 真降且无回归才是绿灯。有 ref，可拆 holdout 子集防过拟合（AGENTS.md 惰性升级条款） |
| test (394 session) | 线上真值层 | 无 ref，只能提交线上评测机看分。线上分唯一不可复算，记 `online_ledger.md` |
| oracle (dev+ref 说话人时间轴) | 上界诊断层 | 用 ref 说话人段给 fun-asr 词重打标签，算"分离修好后文本能到多少"。当前 15.32%，证明瓶颈全在分离质量，不在文本 |

## 关键设计

- **dev ↔ test 分裂风险**：dev 有 ref 可数说话人数做 oracle 提示（D2），test 无 ref 不能用。任何依赖 ref 信息的优化（D2/oracle）只能验证不能直接迁移提交。
- **本地↔线上分裂程度**：尚未提交过线上，无数据。首次提交后，若 dev 涨但线上不涨 → 从 dev 切 holdout 子集防过拟合（AGENTS.md 惰性升级第7章）。
- **overlap 分布**：dev 83/106 session 含跨说话人 overlap（464 对）；说话人数分布 2人×10/3人×29/4人×38/5人×23/6人×6。`reference_self_overlap=0` 是被 collar=5 折叠的假象。

## 校准触发

`online_ledger.md` 自上次校准后累积满 3 次新线上提交时，触发重分层：重跑 `submit/archive/` 归档提交物，看 dev 涨分是否真实迁移到线上。本项目当前 0 次提交，未触发。
