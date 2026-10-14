# Stage6 Macro长期收益闭环验证与信用分配定位

## 结论摘要

- `CLOSED_LOOP_POSITIVE_UPDATE_COUNT`: 1/3
- `BEST_CLOSED_LOOP_DELTA_G`: +0.057007
- `MEAN_CLOSED_LOOP_DELTA_G`: -0.102905
- `CACHE_UPDATE_PAYBACK_SLOT` (首次累计 ΔG>0)：{"20101": null, "20201": 167, "20300": 153}; 末槽仍为正：{"20101": false, "20201": true, "20300": false}
- `CACHE_REVERTED_BY_PPO`: {"20101": {"final_cache_equals_A": true, "reached_original_cache_by_slot_169": true, "reverted": true, "slot": 155}, "20201": {"final_cache_equals_A": false, "reached_original_cache_by_slot_169": false, "reverted": false, "slot": null}, "20300": {"final_cache_equals_A": true, "reached_original_cache_by_slot_169": true, "reverted": true, "slot": 155}}
- `HOLDING_CAUSED_VALUE_LOSS_SUPPORTED`: NOT_SUPPORTED_BY_OBSERVED_BRANCHES; A/B holding choices identical and holding was not independently intervened
- `PPO_MISSED_BENEFICIAL_UPDATE_SUPPORTED`: True
- `MACRO_CREDIT_ASSIGNMENT_PROBLEM_SUPPORTED`: UNKNOWN
- `MACRO_POLICY_MODIFICATION_RECOMMENDED`: True；范围：Follow-up Macro Cache gate/long-horizon credit-assignment work is justified by one missed positive closed-loop result; this three-state diagnostic alone does not justify changing the formal algorithm.

**回答：** 在真实确定性PPO闭环中，三个固定 UPDATE 有 1 个在20槽末仍优于原策略；seed 20201 保留正收益，seed 20101 与 20300 在 PPO 于 Slot 155 重新决策并恢复原 Cache 后转为负值。

本结论只适用于三个指定 Training Episode 的 Slot 150–169。C 来自既有固定 Cache、UAV STAY 报告；C 与 A/B 的后续 UAV 策略不同，不能将 C−A 作为单独 Cache 的闭环因果效应。A/B 才是配对的真实 PPO 确定性闭环。

## 三个状态的 A/B 闭环结果

| Seed / UPDATE | ΔQ | ΔCache Cost | ΔMovement Cost | ΔG (B−A) | 首次累计 ΔG>0 / 末槽仍正 | 最终 Cache 相同 | PPO 退回原 Cache | 执行路径变化任务数 |
|---|---:|---:|---:|---:|---:|---|---|---:|
| 20101 / #118 | +0.011257 | +0.301957 | +0.000000 | -0.290700 | 未越过零 / False | True | True at 155 | 3 |
| 20201 / #167 | +0.167403 | +0.110395 | +0.000000 | +0.057007 | 167 / True | False | False  | 39 |
| 20300 / #71 | +0.162982 | +0.238004 | +0.000000 | -0.075022 | 153 / False | True | True at 155 | 8 |

### 分状态收益形成与消失位置

#### Seed 20101，候选 #118

- 固定 Cache 参考 C：ΔQ=+0.178053，ΔCacheCost=+0.123518，ΔG=+0.054535。
- PPO 闭环 B−A：ΔQ=+0.011257，ΔCacheCost=+0.301957，ΔMovementCost=+0.000000，ΔG=-0.290700。
- 执行分配总数 A→B：{'CLOUD': 227, 'LOCAL': 0, 'RSU': 13, 'SV': 0, 'UAV': 32} → {'CLOUD': 227, 'LOCAL': 0, 'RSU': 13, 'SV': 0, 'UAV': 32}；总时延差 +0.161318 s，车辆能耗差 -0.110397 J。
- 新Macro frame的Cache候选：A 150→#0, 155→#0, 160→#0, 165→#0；B 150→#118, 155→#118, 160→#0, 165→#0。A在这些决策点选择KEEP；B的首槽候选是强制覆盖值。
- B 首次 PPO 重新选择 Cache 的 Slot：155；B 与原始 Cache 配置首次重新相同：155；A/B 实际 Cache 首次相同：155。
- PPO Holding：A/B 在 0 个 Slot 的后槽 Holding 决策不同；首次强制 UPDATE 所在 Macro frame 持续到 Slot 159。两分支首次Macro frame均到最大5槽后在Slot 155重新决策；观察到的损失发生于Slot 155的Cache重配置成本，Holding没有分支差异。没有独立Holding干预，不能估计其单独因果效应。
- 累计 ΔG 检查点（按 B−A）：150:-0.0848, 154:-0.1123, 159:-0.2907, 164:-0.2907, 169:-0.2907。完整曲线在 JSON。

| Slot | ΔQ | ΔC | ΔM | ΔG | 累计 ΔG | B cache action | B Holding | B UAV移动距离 (m) | B−A 路径改变任务数 |
|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 150 | +0.0387 | +0.1235 | +0.0000 | -0.0848 | -0.0848 | #118 (forced_UPDATE_override) | CONTINUE | 7.500 | 1 |
| 151 | -0.0274 | +0.0000 | +0.0000 | -0.0274 | -0.1123 | #118 (held_Macro_frame) | CONTINUE | 7.500 | 2 |
| 152 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.1123 | #118 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 153 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.1123 | #118 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 154 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.1123 | #118 (held_Macro_frame) | FORCED_TERMINATE_MAX | 7.500 | 0 |
| 155 | +0.0000 | +0.1784 | +0.0000 | -0.1784 | -0.2907 | #118 (PPO_deterministic) | CONTINUE | 7.500 | 0 |
| 156 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.2907 | #118 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 157 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.2907 | #118 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 158 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.2907 | #118 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 159 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.2907 | #118 (held_Macro_frame) | FORCED_TERMINATE_MAX | 7.500 | 0 |
| 160 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.2907 | #0 (PPO_deterministic) | CONTINUE | 3.322 | 0 |
| 161 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.2907 | #0 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 162 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.2907 | #0 (held_Macro_frame) | CONTINUE | 4.554 | 0 |
| 163 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.2907 | #0 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 164 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.2907 | #0 (held_Macro_frame) | FORCED_TERMINATE_MAX | 7.500 | 0 |
| 165 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.2907 | #0 (PPO_deterministic) | CONTINUE | 7.500 | 0 |
| 166 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.2907 | #0 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 167 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.2907 | #0 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 168 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.2907 | #0 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 169 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.2907 | #0 (held_Macro_frame) | FORCED_TERMINATE_MAX | 7.500 | 0 |

#### Seed 20201，候选 #167

- 固定 Cache 参考 C：ΔQ=+0.168895，ΔCacheCost=+0.110395，ΔG=+0.058500。
- PPO 闭环 B−A：ΔQ=+0.167403，ΔCacheCost=+0.110395，ΔMovementCost=+0.000000，ΔG=+0.057007。
- 执行分配总数 A→B：{'CLOUD': 210, 'LOCAL': 0, 'RSU': 37, 'SV': 0, 'UAV': 34} → {'CLOUD': 207, 'LOCAL': 0, 'RSU': 37, 'SV': 0, 'UAV': 37}；总时延差 -7.972463 s，车辆能耗差 -0.085975 J。
- 新Macro frame的Cache候选：A 150→#0, 155→#0, 160→#0, 165→#0；B 150→#167, 155→#0, 160→#0, 165→#0。A在这些决策点选择KEEP；B的首槽候选是强制覆盖值。
- B 首次 PPO 重新选择 Cache 的 Slot：155；B 与原始 Cache 配置首次重新相同：Slot 169 前未恢复；A/B 实际 Cache 首次相同：Slot 169 前未相同。
- PPO Holding：A/B 在 0 个 Slot 的后槽 Holding 决策不同；首次强制 UPDATE 所在 Macro frame 持续到 Slot 154。两分支首次Macro frame均到最大5槽后在Slot 155重新决策；观察到的损失发生于Slot 155的Cache重配置成本，Holding没有分支差异。没有独立Holding干预，不能估计其单独因果效应。
- 累计 ΔG 检查点（按 B−A）：150:-0.0781, 154:-0.0695, 159:-0.0446, 164:-0.1184, 169:+0.0570。完整曲线在 JSON。

| Slot | ΔQ | ΔC | ΔM | ΔG | 累计 ΔG | B cache action | B Holding | B UAV移动距离 (m) | B−A 路径改变任务数 |
|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 150 | +0.0323 | +0.1104 | +0.0000 | -0.0781 | -0.0781 | #167 (forced_UPDATE_override) | CONTINUE | 7.500 | 1 |
| 151 | +0.0248 | +0.0000 | +0.0000 | +0.0248 | -0.0534 | #167 (held_Macro_frame) | CONTINUE | 7.500 | 3 |
| 152 | +0.0023 | +0.0000 | +0.0000 | +0.0023 | -0.0511 | #167 (held_Macro_frame) | CONTINUE | 7.500 | 2 |
| 153 | -0.0087 | +0.0000 | +0.0000 | -0.0087 | -0.0598 | #167 (held_Macro_frame) | CONTINUE | 7.500 | 4 |
| 154 | -0.0098 | +0.0000 | +0.0000 | -0.0098 | -0.0695 | #167 (held_Macro_frame) | FORCED_TERMINATE_MAX | 7.500 | 2 |
| 155 | +0.0082 | +0.0000 | +0.0000 | +0.0082 | -0.0613 | #0 (PPO_deterministic) | CONTINUE | 7.500 | 3 |
| 156 | +0.0163 | +0.0000 | +0.0000 | +0.0163 | -0.0450 | #0 (held_Macro_frame) | CONTINUE | 7.500 | 2 |
| 157 | -0.0097 | +0.0000 | +0.0000 | -0.0097 | -0.0547 | #0 (held_Macro_frame) | CONTINUE | 7.500 | 2 |
| 158 | +0.0016 | +0.0000 | +0.0000 | +0.0016 | -0.0531 | #0 (held_Macro_frame) | CONTINUE | 7.500 | 2 |
| 159 | +0.0085 | +0.0000 | +0.0000 | +0.0085 | -0.0446 | #0 (held_Macro_frame) | FORCED_TERMINATE_MAX | 7.500 | 3 |
| 160 | -0.0726 | +0.0000 | +0.0000 | -0.0726 | -0.1173 | #0 (PPO_deterministic) | CONTINUE | 7.500 | 1 |
| 161 | +0.0036 | +0.0000 | +0.0000 | +0.0036 | -0.1137 | #0 (held_Macro_frame) | CONTINUE | 15.000 | 2 |
| 162 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.1137 | #0 (held_Macro_frame) | CONTINUE | 15.000 | 0 |
| 163 | +0.0003 | +0.0000 | +0.0010 | -0.0007 | -0.1143 | #0 (held_Macro_frame) | CONTINUE | 15.000 | 2 |
| 164 | -0.0051 | +0.0000 | -0.0010 | -0.0041 | -0.1184 | #0 (held_Macro_frame) | FORCED_TERMINATE_MAX | 7.500 | 3 |
| 165 | +0.0549 | +0.0000 | +0.0000 | +0.0549 | -0.0635 | #0 (PPO_deterministic) | CONTINUE | 15.000 | 1 |
| 166 | +0.0475 | +0.0000 | +0.0010 | +0.0465 | -0.0170 | #0 (held_Macro_frame) | CONTINUE | 15.000 | 1 |
| 167 | +0.0413 | +0.0000 | -0.0010 | +0.0423 | +0.0253 | #0 (held_Macro_frame) | CONTINUE | 7.500 | 1 |
| 168 | +0.0162 | +0.0000 | +0.0010 | +0.0152 | +0.0405 | #0 (held_Macro_frame) | CONTINUE | 15.000 | 2 |
| 169 | +0.0156 | +0.0000 | -0.0010 | +0.0165 | +0.0570 | #0 (held_Macro_frame) | FORCED_TERMINATE_MAX | 7.500 | 2 |

#### Seed 20300，候选 #71

- 固定 Cache 参考 C：ΔQ=+0.355084，ΔCacheCost=+0.123518，ΔG=+0.231566。
- PPO 闭环 B−A：ΔQ=+0.162982，ΔCacheCost=+0.238004，ΔMovementCost=+0.000000，ΔG=-0.075022。
- 执行分配总数 A→B：{'CLOUD': 206, 'LOCAL': 0, 'RSU': 26, 'SV': 0, 'UAV': 36} → {'CLOUD': 202, 'LOCAL': 0, 'RSU': 31, 'SV': 0, 'UAV': 35}；总时延差 -10.022474 s，车辆能耗差 +0.255374 J。
- 新Macro frame的Cache候选：A 150→#0, 155→#0, 160→#0, 165→#0；B 150→#71, 155→#101, 160→#0, 165→#0。A在这些决策点选择KEEP；B的首槽候选是强制覆盖值。
- B 首次 PPO 重新选择 Cache 的 Slot：155；B 与原始 Cache 配置首次重新相同：155；A/B 实际 Cache 首次相同：155。
- PPO Holding：A/B 在 0 个 Slot 的后槽 Holding 决策不同；首次强制 UPDATE 所在 Macro frame 持续到 Slot 154。两分支首次Macro frame均到最大5槽后在Slot 155重新决策；观察到的损失发生于Slot 155的Cache重配置成本，Holding没有分支差异。没有独立Holding干预，不能估计其单独因果效应。
- 累计 ΔG 检查点（按 B−A）：150:-0.1026, 154:+0.0395, 159:-0.0750, 164:-0.0750, 169:-0.0750。完整曲线在 JSON。

| Slot | ΔQ | ΔC | ΔM | ΔG | 累计 ΔG | B cache action | B Holding | B UAV移动距离 (m) | B−A 路径改变任务数 |
|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 150 | +0.0209 | +0.1235 | +0.0000 | -0.1026 | -0.1026 | #71 (forced_UPDATE_override) | CONTINUE | 7.500 | 1 |
| 151 | +0.0264 | +0.0000 | +0.0000 | +0.0264 | -0.0763 | #71 (held_Macro_frame) | CONTINUE | 7.500 | 1 |
| 152 | +0.0082 | +0.0000 | +0.0000 | +0.0082 | -0.0680 | #71 (held_Macro_frame) | CONTINUE | 7.500 | 3 |
| 153 | +0.1075 | +0.0000 | +0.0000 | +0.1075 | +0.0395 | #71 (held_Macro_frame) | CONTINUE | 7.500 | 3 |
| 154 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0395 | #71 (held_Macro_frame) | FORCED_TERMINATE_MAX | 6.676 | 0 |
| 155 | +0.0000 | +0.1145 | +0.0000 | -0.1145 | -0.0750 | #101 (PPO_deterministic) | CONTINUE | 7.500 | 0 |
| 156 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0750 | #101 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 157 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0750 | #101 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 158 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0750 | #101 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 159 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0750 | #101 (held_Macro_frame) | FORCED_TERMINATE_MAX | 7.500 | 0 |
| 160 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0750 | #0 (PPO_deterministic) | CONTINUE | 7.500 | 0 |
| 161 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0750 | #0 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 162 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0750 | #0 (held_Macro_frame) | CONTINUE | 5.812 | 0 |
| 163 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0750 | #0 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 164 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0750 | #0 (held_Macro_frame) | FORCED_TERMINATE_MAX | 7.500 | 0 |
| 165 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0750 | #0 (PPO_deterministic) | CONTINUE | 6.585 | 0 |
| 166 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0750 | #0 (held_Macro_frame) | CONTINUE | 3.285 | 0 |
| 167 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0750 | #0 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 168 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0750 | #0 (held_Macro_frame) | CONTINUE | 7.500 | 0 |
| 169 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0750 | #0 (held_Macro_frame) | FORCED_TERMINATE_MAX | 7.500 | 0 |

## Macro PPO信用分配与候选选择机制

- Macro Actor 的优势来自正式 SMDP GAE：δ_k = R_k + γ^d V(s_{k+1})·(1−done) − V(s_k); A_k = δ_k + (γλ)^d·(1−done)·A_{k+1}, with actual Macro duration d.。Actor 的Cache动作与该Macro frame中的Termination序列用同一联合PPO优势；Critic使用N2两步目标：for nonterminal k: R_k + γ^{d_k}R_{k+1} + γ^{d_k+d_{k+1}}V(s_{k+2}); terminal tails omit unavailable bootstrap terms.。N2用于Macro Critic训练，不是单独Cache动作的Q(s,a)。
- Long-Value Critic以episode realized return-to-go为目标独立训练，拥有独立参数和optimizer；Macro Actor优势没有调用它：NO: Long-Value Critic is fit separately to realized return-to-go with a disjoint optimizer; _update_macro Actor advantage is the Macro SMDP GAE.。因此它不为候选UPDATE提供直接动作价值排序。
- Cache确定性策略先按校正后的概率质量选择KEEP或UPDATE组（平局KEEP），只有选到UPDATE组后才按UPDATE组内raw logit最大者取候选。更新组每个logit减去合法UPDATE数的log(cardinality)，用于候选组基数校正：corrected logits subtract log(legal UPDATE count) from legal UPDATE logits; deterministic action compares summed corrected probability mass for KEEP vs UPDATE (tie -> KEEP), then uses within-UPDATE raw-logit argmax.。三状态确定性选择均为KEEP；各指定候选的组内排名/概率见下表。
- Holding达到最小长度后按Termination Actor决定继续或终止，达到最大长度时强制结束；持续期间Cache frame动作仍生效。A/B差异会经Holding与后续状态传播，但本实验没有单独干预Holding，因此其独立因果贡献标记UNKNOWN。
- 此次源码检查能解释机制，但Checkpoint和既有报告没有这三个状态训练时的实际Macro advantage符号/梯度，不能反推出训练时为什么对具体候选形成KEEP偏好；该项为UNKNOWN，不把 V(s) 当作动作Q值。

| Seed | PPO首槽确定性动作 | P(KEEP) | P(UPDATE组) | 指定候选组内rank | 指定候选校正概率 |
|---:|---|---:|---:|---:|---:|
| 20101 | KEEP #0 | 0.911895 | 0.088105 | 212 | 4.4452272e-06 |
| 20201 | KEEP #0 | 0.912031 | 0.087969 | 1 | 0.011579446 |
| 20300 | KEEP #0 | 0.911154 | 0.088846 | 175 | 1.5728694e-05 |

## 执行边界与复现信息

- Python执行环境：`C:\software\anaconda3\python.exe` (3.12.7)；Conda环境：`base`。`adaptive_cache_mec` Conda环境本机不可用；使用项目现有Python，禁用不兼容/不可用的Numba导入并采用正式纯Python KKT fallback：project-supported pure-Python KKT fallback (Numba import blocked to select fallback)。没有安装/升级依赖。
- 配置SHA-256：`684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77`；Checkpoint SHA-256：`21b9d97c65ce8c3f14406472743be92225341e168c8ba6bc04fa9de0d76ee8f6`，完成Episode：300。Training Episode manifest与Slot 150任务、车辆位置、预测输入及保存Observation均通过与既有JSON的一致性核对。
- 执行正式 `execute_configured_slot` 120 次，每次重跑Matching/KKT；A/B每个分支20槽。策略参数指纹执行前后相同：True。没有训练、更新Checkpoint、读取Validation/Test或修改正式源码/模型/Reward/物理参数。
- C是前一报告中首槽更新、其后Cache固定且UAV STAY的20槽反事实。A/B为真实确定性PPO闭环；C用于解释固定动作收益与闭环结果的差别，不视作与A/B相同策略条件下的Cache处理效应。

## 范围限制

这是三个预先指定Training状态、20槽窗口上的离线确定性反事实。Slot 169仍与初始Cache不同的分支只报告到20槽末；不外推到完整Episode或其他Training状态。闭环效应是强制一次Cache动作后整套PPO控制产生的总效应，包含后续UAV/Holding/Cache响应。
