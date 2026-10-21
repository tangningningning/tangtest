# Stage4 Cache–UAV 多时间尺度协调机制设计

日期：2026-10-18  
范围：只读梳理与研究方案设计；本报告不实现 Stage4，不修改系统、模型、实验产物或 Stage3 诊断报告。

## 结论摘要

**优先推荐：先做配对因子反事实诊断；仅在确认存在可重复的正向 Cache–UAV 交互后，再实现“Stage3 Cache 候选价值 + Cache 条件化 UAV 动作残差”的有限候选协调器。** 该方案保留 Stage3 对合法 Cache UPDATE、触发及内容价值的职责，不再训练第二套 Cache Candidate 价值头；Stage4 只估计当前 UAV 动作相对 Stage3 分支内既有 Micro 策略的增量，并测量该增量是否随 Cache 分支变化。

当前可确认的源码已具备部分条件：Cache/Micro 策略共享全局 Observation；Micro 决策在当前 Cache 目标确定后生成；Stage3 的 Cache 反事实回放会在每个 Cache 分支内重新计算 UAV Micro 动作，并用正式 Slot 执行器得到含成本的折扣回报。因此“只把 Cache 和 UAV 特征拼到一个新网络”不是有根据的 Stage4 贡献。另一方面，Stage3 Validation 的增量价值排序仍是 `UNRESOLVED`，且三个已评价场景中各策略均未实际移动 UAV；目前尚无证据证明动态 UAV 动作带来净收益或互补收益。

## 1. 已核实的系统合同

### 1.1 Cache 与 UAV 的时间尺度

Stage3 实验合同采用 400 维全局 Observation、9 个 Macro Cache 节点与固定 2-Slot Macro（模型合同见 [Stage3 主实验计划](../mappo_stage3_main_experiment_plan.md#L20)）。当前 Stage3 主线收集器在 `src/ppo_simple/mappo_baseline.py:65, 877-902` 每两个 Slot 开始一次 Cache 决策，选中的 `active_cache` 在 Macro 帧内保持；`src/ppo_simple/mappo_baseline.py:963-1024` 在每个 Slot 用该 Cache 和当前 UAV 位置构造 Observation、生成 UAV 候选并采样 Micro 动作。Episode 最后不足两 Slot 时会形成实际长度为一的终端 Macro transition（`src/ppo_simple/mappo_baseline.py:1196-1238`）。

另有通用 `src/ppo_simple/rollout.py:789-819, 1554-1600` 支持可选 1–5 Slot 自适应持有期。它不是 Stage3 主实验计划所记录的固定 Macro 2 合同。Stage4 应沿用主实验采用的 `mappo_baseline` 合同；若未来改为自适应持有，必须按真实持续时间重新定义反事实回报和终止 bootstrap，不能混用两个入口的结果。

Cache 候选从 KEEP 和合法 UPDATE 目标构成，目标容量会被校验（`src/ppo_simple/mappo_baseline.py:453-589, 900-902`; `src/decision/cache_configuration.py:915-1012, 1244-1280`）。UAV 候选包含 STAY/MOVE 两类；MOVE 目标经枚举、应用和整组转移约束校验（`src/decision/uav_configuration.py:325-441, 453-576`; Stage3 候选采样与执行核对见 `src/ppo_simple/mappo_baseline.py:993-1146`）。

### 1.2 Observation 与策略条件关系

正式 Observation 的切片包含当前 Cache、当前 UAV 位置、预测车辆位置、预测服务流行度、Horizon Mask 和剩余时间（`src/ppo_simple/interface.py:499-518, 1041-1138`）。按 Stage3 主实验规格，其维度为 400（`reports/mappo_stage3_main_experiment_plan.md:20`）。Cache Actor 在 Macro 决策时看到决策前 Cache/UAV 状态；Macro Cache 确定后，Micro Actor 收到以 `active_cache` 重建的 Observation（`src/ppo_simple/mappo_baseline.py:877-902, 963-993`），所以 UAV 策略已经能条件化于实际提交的 Cache 状态。

Policy 定义了独立的 Macro Actor/Critic、Micro Actor/Critic 和 Long-Value 网络（`src/ppo_simple/policy.py:328-359`）。Macro/Micro Critic 均以全局 Observation 为输入；Micro Critic 是状态价值接口 `V_micro(o)`，并不直接返回某个 UAV 候选的动作价值（`src/ppo_simple/policy.py:403-458`）。Micro Actor 将全局状态编码与 UAV 候选特征、UAV_0 的顺序条件拼接后评分（`src/ppo_simple/policy.py:1663-1694`）。因此现状不是两个完全独立、互不知情的 Actor；主要缺口是没有经验证的、对 Cache 分支和 UAV 候选成对的动作价值残差。

### 1.3 Stage3 反事实价值已经覆盖的部分

Stage3 的 `src/ppo_simple/cache_counterfactual_value.py:811-834` 先将某个 Cache 节点候选应用到已执行的共同配置；其他节点行在 UPDATE/KEEP 两分支中保持相同。`src/ppo_simple/cache_counterfactual_value.py:837-984` 对每个 Cache 分支从同一 `pre_cache/pre_uav` 初始化；每个 Slot 在该分支的 Cache 下重建 UAV Observation、候选和 Micro 分布，使用配对 Micro 随机数，再调用正式 `execute_configured_slot()`；随后传播 UAV 位置，并按实际 Macro 时长计算折扣回报及非终端 bootstrap。`evaluate_counterfactual_trace()` 在合法 Cache 候选间生成 `candidate_delta_q = q_candidate - q_keep`（`src/ppo_simple/cache_counterfactual_value.py:1000-1097`）。

这里的 Stage3 标签是“Cache 改变、当前冻结 Micro 策略在新 Cache 状态下重新响应”的策略条件化增量；它不是保持 UAV 轨迹完全不变的纯 Cache 直接效应，也不是 Cache×任意 UAV 动作的完整联合 Q 表。Stage4 若再训练一个相同定义的 Cache `ΔQ`/Candidate 头，会重复 Stage3；若直接将另一个完整 joint-Q 与 Stage3 `ΔQ` 相加，也会发生价值重复计入。

### 1.4 Matching/KKT、回报和成本

每个 Slot 的正式执行调用 `task_allocation.optimize_task_matching()`，获得当前 Cache/UAV/任务位置下的卸载和资源响应；随后系统效用按 `G(t)=Q(t)-C_ca(t)-C_q(t)` 组合并核对恒等式（`src/decision/slot_environment.py:1176, 1256-1280`）。配置注释明确：Q 含任务时延与车辆上传能耗效用，Cache/UAV 配置成本单列且各扣一次（`configs/proposed/system_config.toml:385-401`）。Cache 成本在实际 Cache 改变时收取；UAV 固定成本及位移相关成本在实际 MOVE 时核算。Stage4 的分支价值必须复用这些正式接口，不应在辅助价值中再扣一次成本。

正式配置把配置成本写为：

```text
C_ca(t) = cache_cost_weight × Σ_m [η_ca,m × z_ca_actual,m(t)
                                  + λ_cache × ξ_ca,m × loaded_mb_m(t)]
C_q(t)  = uav_movement_cost_weight × Σ_u [movement_fixed_cost_u × z_q_actual,u(t)
                                          + λ_movement × movement_energy_coefficient × distance_u(t)]
```

`z_ca_actual` 只在 Cache 实际改变时为 1，`loaded_mb` 只计新增装载；`z_q_actual` 只在 UAV 实际移动时为 1。当前配置将 `lambda_cache=0.1`、`lambda_movement=1.76e-5`、`cache_cost_weight=0.1`、`uav_movement_cost_weight=0.50` 标为固定的实验目标权衡尺度，而非可直接解释为焦耳的物理成本（`configs/proposed/system_config.toml:378-411`）。Stage4 不调整这些数值；若测得的互补只存在于 `Q` 而未反映到 `G`，应如实报告为被既定成本抵消。

Stage3 Macro 帧回报为帧内逐 Slot `G(t)` 的折扣和，并保存实际持续时间；Macro SMDP GAE 按 `γ^duration` 和对应 bootstrap 计算（`src/ppo_simple/mappo_baseline.py:1178-1238, 1783-1806`; `src/ppo_simple/rollout.py:2242-2284`）。Stage4 必须沿用这一终端与截断合同。对于长度为 2 的 Macro 帧，Cache 成本只在实际发生配置变化的首 Slot 扣一次；UAV 成本随每次实际位置变化在对应 Micro Slot 计入一次。

## 2. Stage4 要验证的假设

Cache 与 UAV 的互补不能从“两个动作同时发生”或收益变化本身推出。对同一 Macro 起点，`C0` 与 `C1` 是只在一个目标 Cache 节点的 KEEP/UPDATE 上不同、其他 Cache 行保持完全相同的完整配置；若复用 Stage3 每节点标签，其他 Cache 行固定为该 Macro 样本相同的分支配置。`U0` 是两架 UAV 本 Slot 的 STAY/STAY 联合动作，`U1` 是一个两架 UAV 动作元组，其中一架执行共同且在两 Cache 分支都合法的 MOVE，另一架 STAY。候选按目标位置和完整动作语义匹配，不按候选索引匹配。由此定义四条配对路径：

- `V00 = V(KEEP, STAY)`；
- `V10 = V(UPDATE c, STAY)`；
- `V01 = V(KEEP, MOVE u)`；
- `V11 = V(UPDATE c, MOVE u)`。

所有分支从同一决策前 Cache/UAV 状态开始，在相同 Macro 时域、外生任务/车辆输入、预测信息、下游 Micro 策略和 Critic 快照下评价。`U1` 必须在 `C0/C1` 下都合法并语义匹配；不能仅按候选索引匹配。如果没有共同合法目标，则该组不具备严格的双因素对照，应另行标记，不能把候选支持集变化伪装成交互。另将全局所有 Cache 节点均 KEEP 且 UAV 均 STAY 的策略完整回报单独报告为系统基线；若它与此处的 `C0/U0` 一致才可共用 `V00`，否则不能把它混入局部 `I_G`。

计算：

```text
ΔC|STAY = V10 - V00
ΔC|MOVE = V11 - V01
ΔU|KEEP = V01 - V00
ΔU|UPDATE = V11 - V10
I_G       = V11 - V10 - V01 + V00
```

对任务效用 Q 同样计算 `I_Q`，并单独分解 Cache 成本、UAV 成本与净效用 `G`。`I_Q > 0` 表明该 Cache/MOVE 配对的任务效用互补；只有 `I_G > 0` 且联合净效用高于 KEEP+STAY 和最佳单动作，才支持当前成本合同下有可利用的净收益互补。若 `I_Q` 接近零而成本上升，或任务效用增量被 `C_ca/C_q` 抵消，则它不是值得用新模型追逐的净收益机会。若 `I_G≈0`，动作价值近似可加，维持现有分层策略即可；负 `I_G` 表示当前配对呈替代关系，而非互补。

离线真实任务和位置只可用于事后分支评价，不得用于线上候选生成或动作选择。线上阶段仅用当前可见预测来生成/评分候选；分支真实回报只用于训练标签与冻结评价。不同场景、预测窗口、宏时域或下游控制器下的值不能做上述差分。

## 3. 优先推荐：Stage3 候选价值加 Cache 条件化 UAV 残差

### 3.1 定义

令 `π_U` 为固定 Stage3 行为策略下、每个分支依照其当前 Cache/UAV 状态重新采样的 Micro 控制器。对所研究目标 Cache 节点，令 `c0` 为其 KEEP 行，`c` 为 Stage3 已生成且合法的 UPDATE 行；其余 Cache 行在两个分支保持同一 Macro 配置。这与 Stage3 每节点反事实标签的比较条件相符。Stage3 的现有值近似：

```text
ΔV3(s,c) = V(s, do(c), π_U) - V(s, do(c0), π_U)
```

它保留原有候选集合、Trigger 与内容价值职责。对 Macro 开始时的一个合法 UAV Micro 候选 `u`，只新增其相对该 Cache 分支当前 Micro 策略的有限干预残差：

```text
A_U^res(s,c,u) = V(s, do(c), do(u at first Micro decision), π_U thereafter)
                 - V(s, do(c), π_U from first Micro decision)
```

两个值使用相同的 Cache 分支、同一首状态、同一任务/车辆序列、同一未来 Micro policy snapshot、同一共同随机数和同一 bootstrap critic。只在首个 Micro 决策强制 `u`；当前 Macro 帧剩余 Slot 使用分支状态下的同一冻结 Micro 策略，帧末再以该分支状态调用同版本 Macro Critic bootstrap。这样残差是相对现有 Stage3 基线的增量，不重算 Cache 价值，也不把整个 UAV 路径价值叠加到已经包含 UAV 响应的 Stage3 标签上。

有界联合候选得分可写为：

```text
J4(s,c,u) - V(s, do(c0), π_U) = ΔV3(s,c) + A_U^res(s,c,u)
```

`ΔV3(s,c0)=0`，因此该联合分数相对的是目标节点 KEEP、其他 Cache 行固定时的当前 Micro 策略分支回报。匹配的局部 KEEP+STAY 分数为 `Score_C0_STAY = A_U^res(s,c0,STAY)`；若它高于其他合法配对分数，则保持该局部动作。全局 KEEP+STAY 仍作为单独的系统策略基线，不将全局基线和局部候选混为同一个 `ΔV3` 标签。沿用 Stage3 已生成的合法 Cache 候选，只保留少量候选；每个候选只评估 STAY 与最多一至两个预测驱动、且在配对 Cache 状态都合法的 MOVE。选出的 Cache 只在 Macro 边界提交并锁定两 Slot；首个 UAV 动作在该 Micro Slot 执行，后续 Micro Slot 由同一 Cache 条件化策略重新决策。候选支持、动作概率和 PPO 旧/新 log-prob 合同必须在实际实现中保持一致；本报告不假设可以绕过现有 PPO 概率计算直接改用一个不参与 log-prob 的后处理动作。

在配对因子诊断中，交互项 `I_G` 是主要机制指标；残差头只学习 `A_U^res`，不再学习 `ΔV3`。同一候选在不同 Cache 分支下的 `A_U^res` 差异可反映 Cache 是否改变了 MOVE 的边际价值。Stage3 Cache 候选值仍从冻结策略分支总回报产生；Micro 残差更新后，需要用新冻结策略重新产生后续 Stage3 标签，避免把旧策略值与新 Micro 策略混用。

### 对五个核心问题的直接回答

- **互补收益是否存在：** 现有日志只能说明被评价策略没有实际 MOVE；先用 `I_Q/I_G` 的配对反事实确认合法 MOVE 是否改变任务效用，以及该增量能否覆盖成本。
- **如何区分动作贡献：** 固定四条路径的决策前状态、任务/轨迹与续策，分别记录 `Q`、`C_ca`、`C_q`，再用 `ΔC|STAY`、`ΔC|MOVE`、`ΔU|KEEP`、`ΔU|UPDATE` 和差中之差 `I_G` 分开主效应与交互。
- **如何避免局部价值误导：** 不把 Cache 单独最优与 UAV 单独最优拼接；在既有 Cache finalist 与少量共同合法 UAV finalist 上比较配对值，联合打分只在同一回报合同下使用。
- **如何保持合同一致：** Cache 仍按 Macro SMDP 持有并只收取实际配置成本，UAV 按 Micro Slot 执行并只收取实际移动成本；每分支重算 Matching+KKT，帧内折扣与终端/截断 bootstrap 不变。
- **若当前 UAV 常 STAY：** 若配对 MOVE 在相关 Cache 分支下也没有稳定正的净增益，Stage4 没有值得增加的学习模块；只有 MOVE 的条件增益反复出现且当前策略漏选时，才有研究价值。

### 3.2 成立前提与理论边界

1. **Stage3 标签合同先可验证。** 当前实验报告中 Validation 的 ΔQ 诊断为 `UNRESOLVED`：Stage1C Frozen evaluator 因 `_sample_micro_head()` 不接受 `random_uniforms` 而拒绝所有分支标签（`reports/mappo_stage3_main_experiment_results.md:73-79`）。因此先要在后续独立工作中最小修复该回放接口，并确认同一冻结策略、分支隔离、共同外生输入和终端 bootstrap；未通过前不训练 Stage4，也不把当前 Validation 结果当成有效 Cache 价值校准。
2. 反事实标签只代表冻结策略及当前有限候选集的价值，不是全局最优、最优控制或无限期价值。只强制一个首 Micro 动作，后续按当前策略滚动，所以 `A_U^res` 不是 UAV 全路径最优收益。
3. 随机策略下，同一个 common uniform 会让不同分支产生不同动作，因为观察与分布改变；它降低配对方差，但不等于动作相同。若要估计期望值，应对预先固定的一小组 common-random seeds 取均值，不能挑选最有利的随机数。
4. 成本已在分支回报里，不得在 `J4` 再扣成本；若只报任务效用交互 `I_Q`，再单独报告成本差与 `I_G`。
5. 当前 Micro Actor 已将 Cache 状态作为 Observation 输入。新机制不是首次让 UAV “看到” Cache，而是给该条件依赖提供成对动作价值/监督，并将其限制在已有候选集合里。
6. 400 维全局状态的 Stage3 Critic 与有限分支样本不能保证对未见状态或所有 Cache/UAV 配对校准；更新会使 Macro/Micro PPO 策略分布漂移，标签必须按行为 Checkpoint 版本标记与刷新。

## 4. 机制对比

| 机制 | 目标与依赖关系 | 计算代价 | 主要风险与适用条件 |
|---|---|---:|---|
| 现有分层 MAPPO | `π_C(c|s)` 先选 Macro Cache；`π_U(u|s,c)` 在每 Slot 条件选 UAV。Stage3 `ΔV3` 用分支 Micro 响应评价 Cache。 | 最低；现有主线。 | 已有全局状态条件，但没有显式 Cache×UAV 动作残差。适合作为基线；若 paired `I_G` 近零，不需 Stage4 新模块。 |
| 全联合反事实 `V(s,c,u)` / Pair-Q | 对 Cache 候选与 Micro 候选对直接估计完整联合价值。 | 高，候选对数量乘以 Macro Slot 数和场景数；状态/候选支持还会变。 | 容易重复 Stage3 Cache `ΔV3`、出现 Max 乐观偏差、训练样本稀疏、PPO log-prob 接口改变大。仅适合作为很小范围离线对照，不建议作为首个实现。 |
| **条件价值分解：`ΔV3(c) + A_U^res(c,u)`（推荐）** | Stage3 保留 Cache 触发/内容增量值；Stage4 估计在同一 Cache 分支下，首个 UAV 动作相对现有 Micro 策略的净增量。用共同分支和配对残差显示动作互补。 | 有界：复用 Stage3 的 KEEP/Cache 基线分支，每个 Cache finalist 只增加少数首 Micro MOVE 分支。粗略成本为 `O(K_C K_U d)` 次 Slot 资源评价，`K_C/K_U` 固定小上限，`d≤2`。 | 需要冻结策略重放、配对公共输入和低噪声 residual label；Max over 候选会偏乐观，应控制 finalists 并在独立配对样本验证。最符合“不重复 Stage3 Cache 候选学习”。 |
| 受约束交替决策 | 先固定当前 UAV 策略优化 Cache，再固定 Cache 优化 UAV；可做一次有界 best response。 | 中低。 | 顺序依赖、局部最优、两个阶段的目标基线可能漂移；Cache-first/Micro-first 会得不同答案。只有当因子价值可稳定估计、且交互项小或有单调改进证据时才作为简化对照，不作为首推。 |

## 5. 最小验证实验

### 5.1 前置冻结

- 不访问 Test。冻结当前 Stage3 Checkpoint、正式配置与算法代码哈希，以及任务/轨迹/预测文件哈希；基础 Stage3 和 Stage4 分支必须使用相同文件、动作候选、物理合同、Oracle/KKT 和 `γ`。
- 先修复并单独核对 Stage3 Validation counterfactual adapter 的 `random_uniforms` 接口问题；本任务不修改该报告或源文件。所有 Cache/UAV 分支共用同一冻结 Actor/Critic；任务、车辆轨迹、预测源、状态和合法动作规则固定。
- 预先按决策时可见的预测 Cache 缺口/需求、TV/SV几何和合法 MOVE 覆盖差选择少量 Macro 起点；不得按实际收益选 Slot。可用 3 个已有 Training/Validation Episode、每个 2 个有完整两 Slot 后续窗口的 Macro 起点（约 6 个配对状态）作为机制试验；记录 Validation seed 的历史使用并仅作探索性筛选，Training 结果不当作独立泛化证据。不得用 Test 补位或重造场景。
- 每个状态只保留 KEEP 与最多一个已有 Stage3 合法 UPDATE finalist；UAV 只保留 STAY 和一个语义一致、在两个 Cache 分支都合法的预测 MOVE。四因素单元格最多 4 条受控路径；Stage3 已算出的 `V(s,do(c),π_U)` 基线分支可复用以形成 `A_U^res`。若缺少可复用基线，必须为每种 Cache 状态另外保留 Micro 策略分支，不能把 STAY 伪装成策略基线；此时记录实际总分支数。

### 5.2 四分支执行与记录

1. 复制决策前 Cache、UAV 位置、仿真/策略随机状态；四分支从该快照独立恢复。Cache 两 Slot 锁定；UAV 仅在首个 Micro Slot 强制 `STAY/MOVE`，后续按同一冻结 Micro Actor 重新决策。
2. 为离线真实反事实评价，四分支使用同一现存 Slot 任务、车辆位置和信道输入；每 Slot 都运行正式 Matching+KKT。若做线上候选评分，只可使用该 Origin 当时存在的预测场景，不得读未来实际任务或车辆位置。
3. 同时输出 `Q`、`C_ca`、`C_q`、`G`、更新/移动是否实际发生、卸载目标变化、CPU/带宽/Cloud资源可行性、决策所用场景及完整路径。Bootstrap 全部分支使用同一被冻结版本的 Macro Critic；真实 Episode 终止则全分支 bootstrap 为 0，截断则用各自合法后状态的同一 Critic。
4. 报告 `ΔC|STAY`、`ΔC|MOVE`、`ΔU|KEEP`、`ΔU|UPDATE`、`I_Q` 与 `I_G`，以及联合候选相对 KEEP+STAY、最佳单动作的差。使用以 Episode 为簇的配对区间/逐 Episode 符号，避免把同一 Episode 的相邻 Slot 当独立样本。
5. 为估计行为策略期望，使用一组运行前固定的 common-random seeds；所有分支一一配对。不得在运行后挑选 seeds。至少报告单次分支成本与多随机数重复平均的差异。

### 5.3 解释准则

- **有互补且可利用：** `I_Q` 在预先抽取的多个 Episode 同向为正；`I_G` 也稳定为正；UPDATE+MOVE 的净值高于 KEEP+STAY 和同状态下最佳单动作；没有不同预测/评价窗口差异，且约束全部满足。小样本只支持进入下一轮 Stage4 原型，不支持长期/统计泛化结论。
- **近似可加：** 只有预先依据数值误差与配对 Monte Carlo 噪声设定的等价带足够窄，且 `I_G` 区间完全落在该等价带内、单动作增量可在同一误差范围内解释联合值，才判为近似可加。区间仅仅跨过 0 但仍覆盖较大的正负交互时，结论是 `UNRESOLVED`，不能据此说没有互补；保留现有分层 MAPPO，不增加交互网络。
- **主要增加成本：** `ΔQ` 不增或不足以覆盖成本，`ΔG` 为负；保留 Cache/UAV 成本合同，停止该候选/当前条件下 Stage4 扩展。
- **场景/评价混杂：** 配对外生输入、决策信息截止、Macro 时域、下游 Micro 策略或 bootstrap 不一致；该批交互标签无效，修复评价设计后才重做最小试验。

## 6. 实施顺序与停止条件

1. **Gate 0：价值标签可审计。** 先使现有 Stage3 counterfactual Validation diagnostic 能完成，并证明分支状态隔离、同一个冻结行为策略、公共输入、合法候选和终端处理正确。当前报告指出该 Gate 仍失败，故现在不能直接开始 Stage4 训练。
2. **Gate 1：四分支识别。** 固定少量状态与候选，先离线估计 `I_Q/I_G`。若正向交互不跨 Episode 重现或净联合不优于最佳单动作，则停止，不建 Stage4 模型。若区间过宽而无法区分零交互与有意义交互，则标记 `UNRESOLVED`，也不启动 Stage4 学习；只有预先冻结的配对精度足以区分这些情况后，才进入下一 Gate。
3. **Gate 2：最小 residual 原型。** Gate 1 通过后，只加一个 Cache 条件化 UAV residual value/advantage 组件；复用 Stage3 的 Cache candidates、ΔV3 supervision、现有 Macro/Micro PPO、环境奖励与执行器。不得新增 Cache Candidate 价值头或改变成本权重。训练端候选与 PPO sampling / old-new log-prob 必须同一支持；如实现不了概率一致性，则先只做离线 finalist reranking，不宣称是有效 PPO 策略。
4. **Gate 3：冻结配对筛选。** 在未用于 Stage4 标签拟合的固定 Validation Episode 上，将 Stage3 与 Stage3+Stage4 配对比较；Test 保留到 Stage4 代码、损失与评价合同冻结后。若效用改进仅靠成本变化、只出现在一个场景、标签排序失效、发生约束/数值违规或计算代价超预算，则停止扩展。

## 7. 当前证据与判断

Stage3 主实验报告显示，在 Validation seeds 30001–30003 上，Update 13、Stage3 Update 16 与 KEEP+STAY 三种策略均为 0 次实际 UAV 位移，UAV 成本均为 0；Stage3 与 Update 13 的净效用差在三个场景均为负（报告表格及结论见 `reports/mappo_stage3_main_experiment_results.md:34-58`）。这证明该小批策略运行没有用到 UAV 移动，**不证明**任何合法 MOVE 的反事实价值为负，也不证明缓存与 UAV 没有互补。

同一报告将 Stage3 Validation ΔQ 预测/排序指标标为 `UNRESOLVED`，原因是评价器拒绝了反事实分支（`reports/mappo_stage3_main_experiment_results.md:73-79`）。所以目前不能据此断言现有策略正确学会了 Cache，也不能把 Stage4 的价值头训练建立在未经核验的 Validation 价值排序上。

**阶段结论：Stage4 有一个明确可检验的问题，但现在没有足够证据支持直接增加学习模块。先做与 Stage3 分支语义一致的四分支 Cache×UAV 配对反事实诊断；若它显示跨场景、成本后仍为正的交互，再实现条件化 UAV 动作残差。若 UAV MOVE 的任务效用增量不能覆盖其成本，或交互项近似零，Stage4 应停止为学习模块，只将 UAV 留作现有分层策略中的可选控制。**

## 8. 研究边界

本方案不主张全局最优、长期最优或对所有候选的完整 MPC。条件 residual 仅覆盖给定 Cache finalist、给定 UAV Micro 候选、一个首 Micro 干预和有限 Macro SMDP 时域；其后的状态按冻结的当前策略响应。小样本配对只用于判断机制是否值得继续，不是显著性/泛化证明。没有新数据生成，没有读取 Test，没有执行训练或 Episode，也没有修改源代码、配置、Checkpoint、Manifest、正式数据或 Stage3 报告。
