# Stage6 UAV 合法候选与连续移动价值诊断

## 结论摘要

- 四个指定 middle 状态均通过配置、Checkpoint 和状态 JSON 哈希核对；Cache 固定为决策前配置。单Slot正式候选中每架 UAV 含 Stay 有 19 个候选，每架有 18 个 MOVE；最大位移为 15.0 m。
- PPO 确定性动作在四个状态均为 UAV0 移动 7.50 m、UAV1 STAY；正式候选生成器没有把15m上限误限成7.5m。
- 单UAV正式MOVE候选共 144 个，正ΔG候选 0 个；最佳单槽ΔG为 -0.001589。最优已测候选相对PPO动作的平均净收益差（regret）为 0.000233。
- 单槽最大原始Task Utility增量ΔQ=0.000560，来自 UAV_1 1（demand_centroid，15.00m）；其移动成本增量=0.002860、净ΔG=-0.002300，15m动作仍未覆盖成本。
- 最多9组Top-3×Top-3联合动作均经过完整位置约束检查并重新求解；正联合交互 0 个。5槽/20槽正累计ΔG轨迹数分别为 0 / 1。
- 20槽正净收益轨迹明细：20101_middle_slot_150 B2_BEST_SINGLE_FIRST_THEN_FIXED_PREDICTED_WAYPOINT: ΔQ=0.067034, ΔMovementCost=0.056210, ΔG=0.010824; net UAV task-assignment change=0; changes: Slot 158 task 12: UAV_1→CLOUD, Slot 158 task 26: CLOUD→UAV_1。
- 连续轨迹使用Slot 150预测生成固定路点或固定初始方向，只读取后续实际Training任务/车辆位置计算效用；Cache固定，逐槽调用正式Matching/KKT。没有把最终位置效用当作净收益，累计ΔG按每槽ΔQ减去真实移动成本计算。

## 执行依据与合法候选核对

- 配置 SHA-256 `684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77` 与指定哈希一致；Checkpoint SHA-256 `21b9d97c65ce8c3f14406472743be92225341e168c8ba6bc04fa9de0d76ee8f6`、300 Episode / 75 updates 与指定最终Checkpoint一致；源状态文件 `C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\stage6_cache_uav_joint_marginal_effect.json` SHA-256 `aab3cad12e46e2e7798f04df01a419dcd51801f187e7b16a2bf4d7ccee2bc99d`，只读四个 `20001_middle_slot_150, 20101_middle_slot_150, 20201_middle_slot_150, 20300_middle_slot_150`。
- `conda activate adaptive_cache_mec` / `conda run -n adaptive_cache_mec` 在本机无法执行该环境；使用现有 Python 3.12.7（NumPy 2.2.6，Torch 2.13.0+cpu)。Numba 与本机 NumPy 不兼容，按项目可选依赖缺失路径运行纯Python KKT fallback；未安装或升级依赖。
- 正式条件来自 `C:\Users\tangning\Desktop\code\adaptive_cache_mec\src\decision\uav_configuration.py`、`C:\Users\tangning\Desktop\code\adaptive_cache_mec\src\ppo_simple\rollout.py` 和 `C:\Users\tangning\Desktop\code\adaptive_cache_mec\src\ppo_simple\policy.py`。`max_move_distance = dt × speed = 1.0 × 15.0 = 15.0 m`；最低安全间距 10.0 m，飞行边界及高度保持配置值。
- 当前实现包含需求质心、预测需求投影、几何探索候选。每状态候选组合 `361` 个；以正式 `validate_uav_configuration_transition` 复核后，361–361 组联合合法。保存的逐头PPO概率与条件联合概率之和为1，联合概率等于 `P(UAV0) × P(UAV1 | UAV0)`。
- 选择的PPO动作是组平衡概率下的确定性联合动作。每个单独MOVE候选的概率是包含另一架STAY的联合动作概率；UAV1候选按UAV0动作条件计算。候选与概率直接来自Checkpoint哈希匹配的既有状态JSON，并逐坐标复核当前候选生成器。

## 单槽所有单UAV MOVE候选

G00是正式双UAV STAY；每个候选分支均从同一Slot 150决策前状态复制Cache/UAV/任务/车辆输入，再独立执行 `execute_configured_slot` 与完整Matching/KKT。ΔG=ΔQ−ΔMovementCost，并与正式Return差逐槽核对。下表展示每状态的最佳单UAV候选和原PPO确定性动作；逐候选PPO概率、来源、坐标、无线速率、时延、能耗、执行节点变化见JSON。

| 状态 | G00 Task Utility | 单UAVMOVE候选数 | 最佳单UAV候选 | 距离(m) | PPO联合概率 | ΔQ | Δ移动成本 | ΔG | 改变执行节点数 | PPO动作ΔG | regret |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20001_middle_slot_150 | 13.470192 | 36 | UAV_1 2 (projected_demand) | 7.50 | 0.013530 | 0.000248 | 0.001870 | -0.001622 | 0 | -0.001945 | 0.000323 |
| 20101_middle_slot_150 | 14.383667 | 36 | UAV_1 8 (geometry_exploration) | 7.50 | 0.013047 | 0.000199 | 0.001870 | -0.001671 | 0 | -0.001908 | 0.000236 |
| 20201_middle_slot_150 | 10.570229 | 36 | UAV_1 2 (projected_demand) | 7.50 | 0.013570 | 0.000281 | 0.001870 | -0.001589 | 0 | -0.001870 | 0.000281 |
| 20300_middle_slot_150 | 12.370462 | 36 | UAV_1 2 (projected_demand) | 7.50 | 0.013666 | 0.000172 | 0.001870 | -0.001698 | 0 | -0.001791 | 0.000093 |

所有状态合计评估 **144** 个单UAV MOVE；正ΔG为 **0**。有至少一个任务执行节点改变的候选 **0** 个，实际全slot matching 不变的候选中仍可能因UAV链路速率改变而改变已分配UAV任务的效用。

### Top-3 × Top-3联合移动

对每架UAV按单独MOVE的ΔG选Top 3，再以原动作坐标到逐UAV0条件候选列表匹配；只有被正式联合候选记录和全配置验证接受的组合才执行，最多9组/状态。交互增益定义为 `ΔG_joint - ΔG_UAV0 - ΔG_UAV1`，三个分支均以同一个双STAY为基准。

| 状态 | 合法联合组合数 | 最佳联合动作 | 联合ΔG | 联合交互增益 | 改变任务执行节点数 |
|---|---:|---|---:|---:|---:|
| 20001_middle_slot_150 | 9 | 4+2 | -0.003416 | 0.000000 | 0 |
| 20101_middle_slot_150 | 9 | 6+8 | -0.003494 | 0.000000 | 0 |
| 20201_middle_slot_150 | 9 | 4+2 | -0.003459 | 0.000000 | 0 |
| 20300_middle_slot_150 | 9 | 5+2 | -0.003429 | 0.000000 | 0 |

## 5槽 / 20槽连续移动

每条轨迹从同一个Slot 150 Cache/UAV状态独立开始，按Seed读取Slot 150–169真实Training任务序列，每槽重新执行完整Matching/KKT。B0双STAY为配对基准；B1复现PPO首槽动作并在后续保持位置。另有最多两条候选轨迹，因而每状态非STAY轨迹不超过3条：B2首槽采用最佳单UAV净收益候选，之后向Slot 150固定预测需求质心合法移动；B3首槽采用Top-3组合中ΔG最佳的联合候选，之后沿其初始航向逐槽移动。所有路点/航向只由Slot 150预测与正式候选产生，不使用未来真实任务决策移动方向。若后续联合转移非法，正式约束校验后该槽两机STAY并记录原因。

| 状态 | 轨迹 | H | 累计ΔQ | 累计移动成本 | 累计ΔG | UAV任务分配累计变化 | 执行节点变化槽数 | UAV累计移动距离(m) | 路径回退次数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20001_middle_slot_150 | B1_PPO_FIRST_MOVE_THEN_STAY | 5 | 0.000355 | 0.001870 | -0.001515 | 0 | 0 | 7.50 | 0 |
| 20001_middle_slot_150 | B1_PPO_FIRST_MOVE_THEN_STAY | 20 | 0.001334 | 0.001870 | -0.000536 | 0 | 0 | 7.50 | 0 |
| 20001_middle_slot_150 | B2_BEST_SINGLE_FIRST_THEN_FIXED_PREDICTED_WAYPOINT | 5 | 0.002660 | 0.013310 | -0.010650 | 0 | 0 | 67.50 | 0 |
| 20001_middle_slot_150 | B2_BEST_SINGLE_FIRST_THEN_FIXED_PREDICTED_WAYPOINT | 20 | 0.055160 | 0.056210 | -0.001050 | 0 | 0 | 292.50 | 0 |
| 20001_middle_slot_150 | B3_BEST_TOP3X3_JOINT_HEADING | 5 | -0.000483 | 0.026620 | -0.027103 | 0 | 0 | 135.00 | 0 |
| 20001_middle_slot_150 | B3_BEST_TOP3X3_JOINT_HEADING | 20 | 0.027300 | 0.112420 | -0.085120 | 0 | 1 | 585.00 | 0 |
| 20101_middle_slot_150 | B1_PPO_FIRST_MOVE_THEN_STAY | 5 | 0.000065 | 0.001870 | -0.001805 | 0 | 0 | 7.50 | 0 |
| 20101_middle_slot_150 | B1_PPO_FIRST_MOVE_THEN_STAY | 20 | -0.000464 | 0.001870 | -0.002334 | 0 | 0 | 7.50 | 0 |
| 20101_middle_slot_150 | B2_BEST_SINGLE_FIRST_THEN_FIXED_PREDICTED_WAYPOINT | 5 | 0.004825 | 0.013310 | -0.008485 | 0 | 0 | 67.50 | 0 |
| 20101_middle_slot_150 | B2_BEST_SINGLE_FIRST_THEN_FIXED_PREDICTED_WAYPOINT | 20 | 0.067034 | 0.056210 | 0.010824 | 0 | 1 | 292.50 | 0 |
| 20101_middle_slot_150 | B3_BEST_TOP3X3_JOINT_HEADING | 5 | 0.005056 | 0.026620 | -0.021564 | 0 | 0 | 135.00 | 0 |
| 20101_middle_slot_150 | B3_BEST_TOP3X3_JOINT_HEADING | 20 | 0.080280 | 0.112420 | -0.032140 | 0 | 2 | 585.00 | 0 |
| 20201_middle_slot_150 | B1_PPO_FIRST_MOVE_THEN_STAY | 5 | 0.000149 | 0.001870 | -0.001721 | 0 | 0 | 7.50 | 0 |
| 20201_middle_slot_150 | B1_PPO_FIRST_MOVE_THEN_STAY | 20 | 0.000912 | 0.001870 | -0.000958 | 0 | 0 | 7.50 | 0 |
| 20201_middle_slot_150 | B2_BEST_SINGLE_FIRST_THEN_FIXED_PREDICTED_WAYPOINT | 5 | 0.004598 | 0.013310 | -0.008712 | 0 | 1 | 67.50 | 0 |
| 20201_middle_slot_150 | B2_BEST_SINGLE_FIRST_THEN_FIXED_PREDICTED_WAYPOINT | 20 | 0.055602 | 0.056210 | -0.000608 | 1 | 2 | 292.50 | 0 |
| 20201_middle_slot_150 | B3_BEST_TOP3X3_JOINT_HEADING | 5 | 0.005593 | 0.026620 | -0.021027 | 0 | 1 | 135.00 | 0 |
| 20201_middle_slot_150 | B3_BEST_TOP3X3_JOINT_HEADING | 20 | 0.068268 | 0.112420 | -0.044152 | 1 | 2 | 585.00 | 0 |
| 20300_middle_slot_150 | B1_PPO_FIRST_MOVE_THEN_STAY | 5 | 0.000282 | 0.001870 | -0.001588 | 0 | 0 | 7.50 | 0 |
| 20300_middle_slot_150 | B1_PPO_FIRST_MOVE_THEN_STAY | 20 | 0.000812 | 0.001870 | -0.001058 | 0 | 0 | 7.50 | 0 |
| 20300_middle_slot_150 | B2_BEST_SINGLE_FIRST_THEN_FIXED_PREDICTED_WAYPOINT | 5 | 0.002958 | 0.013310 | -0.010352 | 0 | 0 | 67.50 | 0 |
| 20300_middle_slot_150 | B2_BEST_SINGLE_FIRST_THEN_FIXED_PREDICTED_WAYPOINT | 20 | 0.046776 | 0.056210 | -0.009434 | 1 | 1 | 292.50 | 0 |
| 20300_middle_slot_150 | B3_BEST_TOP3X3_JOINT_HEADING | 5 | 0.003028 | 0.026620 | -0.023592 | 0 | 0 | 135.00 | 0 |
| 20300_middle_slot_150 | B3_BEST_TOP3X3_JOINT_HEADING | 20 | 0.050725 | 0.112420 | -0.061695 | 1 | 1 | 585.00 | 0 |

每槽坐标、距离、累计成本、Task Utility、Return、Cloud/RSU/UAV分配、UAV有效上行速率及任务时延都保存在JSON `trajectory_records`。H=5和H=20均为完整逐槽差值求和；B0移动成本为零。

## 根因判别与边界

- **候选范围：** 所有四个状态中正式候选均达到15m单槽上限，且包含多个需求/几何方向；7.5m是PPO所选 projected-demand 候选坐标，不是候选生成器的上限。
- **几何/执行路径：** 单槽候选导致Matching执行节点变化 0 次；连续轨迹出现 10 个移动-基准配对槽有至少一个任务切换执行节点。逐候选/逐槽改变任务ID均有记录。
- **净效用：** 单槽、5槽和20槽净收益统计见上表；移动成本始终按正式系统逐槽计费。若任务路径不变，Q变化来自同一UAV任务的信道/带宽分配；若路径改变，JSON记录Cloud/UAV任务数变化与完整Utility差。
- **缓存可用性：** Cache始终固定；每状态每个任务的服务缓存命中及正式可执行目标在JSON中保留。只在候选坐标变化后正式Matching选中UAV时，才认为出现可观测卸载路径变化。
- **范围限制：** 连续实验只覆盖按规定筛选的单槽最优、固定预测路点、Top-3联合航向及PPO原动作；并非全局最优路径搜索，也没有每槽重新预测的策略分支。超出这些方向的移动价值为UNKNOWN。

## 最终字段

- `UAV_POSITIVE_SINGLE_SLOT_MOVE_COUNT`: 0（四状态全部合法单UAV MOVE候选）。
- `UAV_BEST_SINGLE_SLOT_DELTA_G`: -0.001589。
- `UAV_POSITIVE_5_SLOT_TRAJECTORY_COUNT`: 0。
- `UAV_POSITIVE_20_SLOT_TRAJECTORY_COUNT`: 1。
- `UAV_BEST_20_SLOT_DELTA_G`: 0.010824。
- `UAV_PPO_SELECTION_REGRET`: 平均 0.000233，最大 0.000323；逐状态值见JSON。
- `UAV_OFFLOADING_PATH_CHANGED_COUNT`: 0 个单槽候选、10 个连续轨迹槽改变了至少一个执行节点。
- `UAV_JOINT_MOVE_SYNERGY_SUPPORTED`: NO（正交互 0 / 36 组）。
- `UAV_EFFECTIVE_MOVEMENT_SPACE_SUPPORTED`: YES_WITHIN_TESTED_SET。

## 执行边界

- 正式 `execute_configured_slot` / Matching / KKT调用 508 次；最小一致性检查的双STAY复现全部通过。
- 只读取四个指定Training Episode、正式配置、CheckPoint与用户指定状态JSON；未读取Validation/Test，未训练或更新PPO。正式源码/配置/数据/Checkpoint未修改；未commit/push。临时Runner在完成后删除。
