# 第二阶段道路约束车辆预测记录

## 本轮范围

本轮只实现车辆轨迹预测辅助模块：SUMO lane 内有界运动、普通道路一阶转移、单路径输出、驻留时间下界和 0/1 覆盖接口。预测结果没有接入流行度、缓存、UAV、Matching、KKT、reward 或 PPO 决策。

## 成熟实现调查与复用

- 本机已安装并实际检查官方 SUMO Python 接口：`traci`、`sumolib`，sumolib 版本为 1.27.1。
- 复用 `sumolib.net.readNet(..., withInternal=True)`、`Net.getEdge/getLane`、`Edge.getFunction/getOutgoing`、`Lane.getLength/getSpeed/getShape` 和 `sumolib.geomhelper.positionAtShapeOffset`。
- 复用 TraCI 当前状态接口：`vehicle.getRoadID`、`getLaneID`、`getLanePosition`、`getSpeed`、`getAcceleration`、`getAllowedSpeed`、`getPosition`。预测器没有调用 `vehicle.getRoute(...)`，也没有把 route 文件或未来轨迹作为输入。
- 没有引入外部 PPM、Transformer、GNN、LSTM/GRU、复杂概率树或新道路几何计算器；当前模型用已观测普通道路转移计数完成一阶概率和合法出口回退。

## 实现与接口

- `data.py` 新增 `RoadTransitionModel`：用历史观测统计 `P(e_j|e_i)`，递归跳过 SUMO internal edge，将 `A -> :junction -> B` 折叠为 `A -> B`；历史不足时仅在合法出口上均匀回退。
- `data.py` 新增 `RoadConstrainedVehiclePredictor`：使用当前位置、lane 纵向位置、速度和可选加速度沿 lane shape 投影；到 lane 末端时只选择最高概率合法出口，第一版保留一个概率为 1.0 的分支。
- 每个预测包含 `vehicle_id`、`prediction_origin_slot`、`future_slots`、`trajectory_branches`、未来位置、未来 edge/lane 和转移概率；`predict_residence_time` 在窗口末仍未离开时返回 `censored=True` 与 `H*dt` 下界；`predict_coverage` 返回节点×未来时隙的覆盖指标。
- `get_vehicle_states` 增加当前 lane position、lane index、加速度和允许速度字段；`load_sumo_scene` 让预测器可读取 internal edge，但 SUMO 运行仍由官方 TraCI 控制。
- `environment.py` 的实时 SUMO 分支改用道路预测器；缺少 edge/lane 元数据的离线 NPZ 仍保留 Kalman 兼容回退。该回退不参与正式实时 SUMO 路径。
- `config.yaml` 增加 `trajectory_prediction.horizon_slots=3`、`max_path_branches=1` 和有限历史长度配置。

## 两项短测试

1. **真实 SUMO 基础短跑**：固定项目配置，推进 100 s 到 160 s，检查 281 个预测、843 个未来 lane 点；观测到 45 次普通道路转移、35 个转移统计行。所有 lane 归属、lane 纵向范围、单分支概率和坐标有限性检查通过。
2. **固定种子因果回放**：seed=20260811，在每个未来窗口生成预测后才推进 SUMO；319 条完整三步样本可评估，ADE=51.46 m，FDE=93.05 m，下一道路 Top-1=82.25%，293 个下一道路样本；预测器源码未出现未来 `getRoute(...)` 调用。

另做了一步环境冒烟验证：实时模式为 `road_constrained`，预测矩阵 `(12,2)`、reward 和预测位置均有限，原有 solver 可继续执行。由于本机没有 PyTorch，测试显式关闭了已有 GRU checkpoint，使用项目原有观测流行度 smoke 回退；没有训练或修改 GRU/PPO。

## 删除内容与剩余问题

- 本轮没有删除旧 Kalman 类；它只作为缺少道路元数据的离线兼容接口保留。实时 SUMO 的正式位置预测已不再走 Kalman。
- 当前只输出单条最高概率路线，覆盖接口虽支持按分支概率加权，但尚未启用多分支；内部 edge 的几何段在跨路口时按合法普通出口衔接，仍是第一版近似。
- ADE/FDE 仅为短时 sanity check，不构成算法对比或训练结论；后续只有在简单预测明显影响缓存性能时，才考虑概率覆盖或更复杂模型。

# 第三阶段移动感知服务需求/流行度预测记录

## 本轮范围

本轮只实现“车辆历史服务偏好 + 车辆未来覆盖 -> 节点未来服务需求/流行度”。
没有修改自适应缓存更新判断、UAV位置优化、Matching、KKT、reward、PPO或第二阶段
道路约束车辆预测主体。

## 成熟实现复用

- 复用 `data.py` 已有的 `_service_popularity_from_requests` 请求统计、服务 ID
  对齐、有限数组校验和 `RoadConstrainedVehiclePredictor` 输出的
  `predicted_positions_m`；没有另建一套服务编号或轨迹模型。
- 历史统计采用成熟的 categorical frequency + EWMA 思路，并用一个全局历史先验
  做简单 Dirichlet/Laplace 风格平滑：车辆历史较少时自动更多依赖全局先验。
- 没有增加第三方依赖，也没有采用 GRU/LSTM/Transformer/GNN/AutoEncoder/FL；当前
  自适应缓存核心只需要可解释、严格因果的节点需求排序。

## 修改文件与正式接口

- `data.py` 新增 `MobilityAwarePopularityPredictor`。它维护全局和逐车 EWMA，按
  `p_vf=(H_vf+alpha*p_global_f)/(sum_f H_vf+alpha)` 平滑车辆偏好；预测窗口使用
  等权 `omega_h=1/H`，将 Stage 2 的 0/1 未来覆盖映射到 SV、RSU、UAV 节点。
- `data.py` 新增 `build_causal_popularity_sequence`，仅供离线短测/兼容接口生成 F 维
  因果序列；正式 data source 不预扫描全量请求，在线环境逐 slot 维护同一预测器。
  节点级输出至少包含
  `node_id`、`service_id`、`predicted_demand`、`predicted_popularity`、
  `prediction_origin_slot`，并保证每个节点的服务流行度和为 1。
- `environment.py` 正式 SUMO 闭环使用道路预测器提供 SV/任务车辆未来位置，RSU
  固定，UAV 保持当前已知位置；最近一次完整 M×F 节点结果可由
  `last_node_prediction` 或 step info 中的 `predicted_node_*` 读取。旧 observation
  的 F 维接口使用节点结果的简单均值，未改变缓存/策略维度。
- `train.py` 正式 data source 改为 `mobility_aware_popularity`，不再加载或调用 GRU。
  `config.yaml` 新增唯一的 `popularity_prediction.history_decay` 和
  `preference_smoothing`，GRU checkpoint 置为 `null`。

## 最终因果流程与公式

```text
slot t 先读取 τ<t 的请求历史
    -> 车辆服务偏好 p_v(t)
    -> Stage 2 道路约束未来位置/覆盖 I_v,m(t+h)
    -> D_hat_m,f(t)=sum_h (1/H) sum_v I_v,m(t+h) p_v,f(t)
    -> rho_hat_m,f(t)=(D_hat_m,f+eps)/sum_j(D_hat_m,j+eps)
```

没有预测覆盖车辆的节点直接回退到全局历史先验，不生成随机值；UAV/RSU也不引入
额外的区域历史权重或 mobility/personal 多重混合参数。

## 两项必要测试

1. 因果/数值测试：PASS。固定 `N_V=10`、`N_S=2`、服务数量 `F=20`、`H=3`，
   节点×服务需求维度为 `(2,20)`，输出记录 40 条，每个节点流行度和为 1，无
   NaN/Inf；当前请求用于当前预测 0 次，未来请求读取 0 次，并验证同一 slot
   请求登记后不能再次预测该 slot。
2. 短因果 SUMO 回放：PASS。固定 seed=20260811，回放 20 个 slot、评估 20 个
   三步未来窗口；移动感知预测 Top-3 Recall=0.4500、Top-5 Recall=0.6500，
   简单全局历史基线 Top-3 Recall=0.5167、Top-5 Recall=0.6500。该结果只用于
   sanity check；短窗口和随机请求下全局基线的 Top-3 略高，不构成缓存性能结论，
   因此没有据此增加复杂模型。另完成一次正式环境 reset/one-step smoke，节点级
   输出为 `(6,10)` 且 reward、流行度和请求历史状态均有限。

## 旧 GRU 状态

旧 `build_gru_model`、`predict_service_popularity`、`load_gru_checkpoint` 仅保留为
隔离的离线 baseline/兼容代码；正式 `load_data`、`MECEnvironment` 和 `train.py`
均无 GRU 依赖、无 GRU/EWMA/mobility blending，也不会读取 `gru_f10.pt`。

## 清理与停止点

- 删除正式环境、训练数据源中的冻结 GRU 调用和 observed-popularity smoke 回退逻辑；
  GRU baseline 函数未参与正式路径。
- 本轮未生成 CSV、临时分析脚本或 debug 输出；测试结束后清理本轮产生的
  `__pycache__`。
- 本轮停止在未来节点服务需求/流行度预测，不进入自适应缓存更新决策。

# 第四阶段自适应缓存更新判断记录

## 本轮范围

本轮只实现“候选缓存是否值得执行”的事件触发判断，不重写候选缓存生成、UAV
位置、轨迹/流行度预测、Matching、KKT、reward 主体或 PPO。

## 成熟方案与复用

- 复用 `solver.py` 现有 `solve_cache` 的逐节点 0-1 knapsack、容量可行化和
  `final_cache` 生成逻辑；它现在明确作为 `C_cand` 输出，候选生成规则未改变。
- 复用 `environment.py` 的 `cache_update_cost`、已有服务大小单位换算和
  `system_utility`。没有复制缓存成本、背包或系统效用计算器。
- 触发思想采用成熟的 event-triggered/reconfiguration-gain 形式：只有候选相对
  当前缓存真正变化且扣除切换成本后净收益为正，才执行候选；没有增加阈值或学习模型。

## 修改与正式流程

- `solver.py` 新增 `decide_adaptive_cache_update`，统一区分：
  `C_old` 当前真实缓存、`C_cand` 现有候选、`C_exec` 最终执行缓存。
- `environment.py` 把 Stage 3 节点级预测传入触发判断；后续 feasible nodes、
  matching、KKT 和 reward 只读取 `C_exec`。step info 增加
  `cache_candidate_changed`、`cache_trigger`、`cache_changed`、`cache_gain`、
  `cache_switch_cost`、`keep_predicted_utility`、`candidate_predicted_utility`、
  `services_changed` 和 `cache_check`。
- `train.py` 将上述判断聚合到正式 episode metrics，增加 check、candidate change、
  trigger、actual change、平均 gain、平均切换成本和 SV replacement 统计。
- `SolverResult.final_cache` 现在表示 `C_exec`，并额外保留 `candidate_cache`；实际
  缓存成本始终按 `C_old -> C_exec` 计算。

## 最终触发公式与量纲

本轮采用方案 A。预测命中效用不含缓存切换成本，使用 Stage 3 节点流行度计算：

```text
U_keep  = sum_f rho_hat[m,f] * C_old[m,f]
U_cand  = sum_f rho_hat[m,f] * C_cand[m,f]
G_cache = U_cand - U_keep - C_switch(C_old, C_cand)
```

当 `C_cand == C_old` 时直接输出 `cache_trigger=0`、`cache_changed=0`、
`cache_switch_cost=0`；否则仅当 `G_cache > 0` 执行 `C_exec=C_cand`，其余情况
保持 `C_exec=C_old`。预测效用不含成本，真实 reward 仍只对 `C_old -> C_exec`
调用一次已有缓存成本函数，因此没有重复计费。

SV 替换仍是独立物理事件：新服务车辆缓存行先置空并记录
`service_vehicle_replacement_rows/count`，不会仅因为身份替换自动产生 adaptive
cache trigger。

## 两项必要测试

1. 静态候选测试：PASS。`C_cand=C_old` 时得到
   `cache_trigger=[0,0]`、`cache_changed=[0,0]`、
   `cache_switch_cost=[0.0,0.0]`。
2. 短 SUMO 因果运行：PASS。固定 seed=20260811、`N_V=10`、`N_S=2`，运行 20 个
   slot：

```text
check = 20
candidate changed = 51（节点事件）
trigger = 42
actual changed = 42
平均 cache_gain = 0.0187773924
平均 cache_switch_cost = 0.0070279333
容量违反 = 0
未来/当前真实请求用于缓存判断 = 0
NaN/Inf = 0
```

## 代表方向事件

```text
slot  node  U_keep     U_candidate  switch_cost  gain         decision
0     0     0.100000   0.100000      0.014952     -0.014952    keep
1     0     0.080000   0.146667      0.014584      0.052083    update
2     0     0.117333   0.190667      0.014952      0.058381    update
```

方向与 `G_cache > 0` 规则一致。

## 清理与停止点

- 移除正式路径中“候选变化即无条件执行”的旧逻辑；保留现有候选生成，不保留多套
  `old_trigger/new_trigger/legacy_trigger/gain_trigger` 正式实现。
- 本轮未生成 CSV、临时脚本或 debug 输出；测试结束后清理 `__pycache__`。
- 本轮停止在自适应缓存更新判断，不进入协同缓存配置算法重构。

## 第五阶段：协同服务缓存配置（2026-08-11）

### 成熟代码调查与复用

- 检索了成熟的 submodular/marginal-gain greedy 与 cooperative caching 方法，参考
  [Constrained Submodular Maximization via Greedy Local Search](https://arxiv.org/abs/1705.06319)
  和
  [Cooperative Edge Caching in User-Centric Clustered Mobile Networks](https://arxiv.org/abs/1710.08582)。
- 没有发现可以直接适配当前“SV+RSU+UAV、异构容量、SUMO覆盖、现有reward”的现成
  代码；因此没有引入外部依赖，而是在项目已有代码上做最小适配。
- 直接复用了 `solve_cache` 的严格输入校验、`_integerize_cache_weights` 的容量单位
  处理、`environment.task_utility`/`execute_task` 的时延能耗效用、现有节点覆盖预测、
  `cache_update_cost` 和第四阶段 C_old/C_cand/C_exec 流程。
- `solve_cache` 保留为明确的 `legacy baseline` 供 Old vs New 对照，正式 `solve_slot`
  不再调用它作为候选生成器。

### 修改与协同算法

- `data.py` 在同一份严格因果 Stage 3 预测中增加
  `predicted_coverage(M,V,H)`、车辆偏好、车辆未来位置、节点未来位置和时域权重；
  没有修改轨迹预测或流行度预测模型。
- `solver.py` 新增 `predicted_cache_utility` 和 `solve_cooperative_cache`，SV、RSU、UAV
  统一使用同一套候选算法。policy 的 cache trigger 只作为一次全局候选检查开关，
  不再让每个节点独立选择热门服务。
- 预测效用对每个 `(v,f,h)` 只取覆盖该车辆的已缓存节点中的最佳边缘执行效用；重叠
  节点的同一服务不会机械重复计收益，空间分离节点仍可同时缓存同一服务。
- 候选从当前缓存出发，按 `边际收益/服务大小` 贪心加入服务；贪心结束后最多执行
  一次 1-swap，允许删除低价值服务并加入更高价值服务。容量继续使用真实服务大小，
  没有改成固定 K 个服务。
- 边际收益只使用配置 workload 中点构造轻量预测执行成本，复用已有 delay/energy/
  utility 公式，不运行 Matching/KKT，不优化 UAV 位置。

### 第四阶段 utility 同步

候选生成和触发判断共用同一个 `predicted_cache_utility`：

```text
U_keep = predicted_cache_utility(C_old)
U_cand = predicted_cache_utility(C_cand)
G_cache = U_cand - U_keep - C_switch(C_old, C_cand)
```

候选阶段不重复扣完整切换成本；实际 reward 仍只对 `C_old -> C_exec` 调用一次已有
成本函数。新增日志：
`cooperative_candidate_utility`、`cache_duplicate_count`、
`cache_unique_services`、`cache_local_swap_count`。

### 三项测试

1. 容量与可行性：PASS。人工矩阵、真实服务大小和异构容量下容量违反为 `0`，候选
   始终为合法 0/1 服务 ID，效用和诊断无 NaN/Inf。
2. 协同性：PASS。两个高度重叠节点对同一热点服务只保留一份；两个空间分离节点
   对同一服务允许两份。1-swap 测试成功替换低价值服务，`cache_local_swap_count=1`。
   Old vs New 对照中：

```text
Old predicted utility = 1.0
New predicted utility = 2.0
Old duplicate count   = 2
New duplicate count   = 0
```

3. 真实 SUMO 短运行：PASS。固定 `seed=20260811`、`N_V=10`、`N_S=2`，运行 20 个
   slot：

```text
cache_check = 20
candidate_changed = 27
trigger = 1
actual_changed = 1
mean_keep_predicted_utility = 0.0140126741
mean_candidate_predicted_utility = 0.0223926519
mean_cache_gain = -0.0031082893
mean_switch_cost = 0.0032233333
capacity_violations = 0
NaN/Inf = 0
failed = 0
Local = 0 (0.0000)
SV = 3 (0.0283)
RSU = 36 (0.3396)
UAV = 24 (0.2264)
Cloud = 43 (0.4057)
```

候选被拒绝的 slot 保持 `C_exec=C_old`，与第四阶段正净收益触发规则一致；本测试
只作闭环 sanity check，不据此讨论最终性能优劣。

### 清理与停止点

- 正式路径删除了逐节点 `solve_cache` 候选调用；旧 `solve_cache` 只作为显式 baseline
  保留，没有与新算法混用。
- 没有新增 PPO/MAPPO、神经网络、整数规划、2-swap/3-swap、遗传算法或 UAV 联合优化。
- 本轮未修改 Matching、KKT、reward 主体和预测模型；未生成临时脚本、CSV 或 debug 输出。
- 测试结束后清理 `__pycache__`。
- 本轮停止在“协同服务缓存配置”，不进入 UAV 位置优化和 PPO 训练。

# 第六阶段：需求感知 UAV 位置自适应调整（2026-08-11）

## 本轮范围

本轮只处理“UAV 应该移动到哪里、是否值得移动，以及移动后是否同步更新协同缓存”。
不修改道路约束轨迹预测、移动感知流行度预测、Matching、KKT、reward 主体和缓存/预测
参数，也不进行 PPO/MAPPO 训练。

## 成熟实现复用与删除内容

- 复用了 `solver.py` 已有的 `_region_values`、`_uav_original_constraints`、
  `environment.uav_movement_cost`、`predicted_cache_utility` 和 Stage5 的
  `solve_cooperative_cache`/`decide_adaptive_cache_update`。
- 检查项目依赖和源码后没有发现可直接复用的 KMeans、sklearn 或 SciPy 位置搜索实现；
  因此采用当前工程最小适配的有限候选比较，不新增第三方依赖。
- 正式路径删除了旧的 raw displacement + CVXPY/CLARABEL SCA UAV 位置求解器；
  `uav_triggers` 和 raw displacement 仅保留为策略接口兼容输入，不再决定正式 UAV 位置。
- 没有新增连续轨迹优化、遗传算法、整数规划、Matching/KKT 联合求解或神经网络。

## 地面服务缺口与候选位置

对每个预测车辆和未来时隙，先计算 SV/RSU 当前缓存能否提供其历史偏好服务：

```text
gap[v,h] = sum_f p[v,f] * 1{未来(v,h)未被已缓存 SV/RSU 覆盖}
```

将 `gap[v,h]` 作为车辆未来位置的权重，生成一个加权质心和少量最高权重位置，
每个 UAV 的候选集合固定包含当前位置，并加入投影到活动区域及最大位移圆盘内的热点：

```text
Q_u^cand = {Q_u^old} ∪ {project(hotspot, region, vmax * Δt)}
```

候选点逐个检查活动区域、最大速度/位移和 UAV 最小安全距离。多 UAV 采用逐个贪心选择，
每次把新位置与已经选定的其他 UAV 一起检查，避免所有 UAV 聚集到同一点；`N_U=0/1/>1`
均可运行。

## A/B/C/D 一次联合判断

每个候选位置只在临时预测数组上重算 UAV 覆盖，不写入环境，不执行 Matching/KKT。使用
同一个 `predicted_cache_utility` 比较：

```text
J_A = U(C_old, Q_old)
J_B = U(C_cache, Q_old) - C_cache(C_old, C_cache)
J_C = U(C_old, Q_candidate) - C_uav(Q_old, Q_candidate)
J_D = U(C_joint, Q_candidate)
     - C_cache(C_old, C_joint) - C_uav(Q_old, Q_candidate)
```

其中 `C_cache` 和 `C_joint` 来自 Stage5 协同贪心缓存配置；最终选择四者最大值，标签记录为
`A_keep`、`B_cache_only`、`C_uav_only` 或 `D_joint`。选中 A/B 时 UAV 保持原位，选中 A/C
时缓存保持原矩阵。候选评价只扣一次缓存切换成本和一次 UAV 位置成本，真实 reward 仍只对
最终 `C_old -> C_exec` 和最终 `Q_old -> Q_exec` 调用一次现有成本函数，避免重复计费。

## 修改文件与诊断字段

- `solver.py`：新增 `select_demand_aware_uav_cache_state` 及需求缺口热点、候选投影、
  A/B/C/D 评价；`SolverResult` 增加候选移动、动作标签、四个 `J_*`、候选数、移动成本、
  地面缺口等字段。
- `environment.py`：把上述联合结果正式写回缓存/UAV状态；step info 增加
  `uav_check_slots`、`uav_candidate_move_slots`、`uav_moved_slots`、`uav_nodes_moved`、
  `uav_move_distance`、`J_keep`、`J_cache_only`、`J_uav_only`、`J_joint`、
  `cache_check_slots`、`cache_candidate_changed_slots`、`cache_changed_slots`、
  `cache_changed_nodes`、`cache_changed_services`、`cache_cost`、`uav_move_cost` 等字段。
- `train.py`：聚合上述 slot 级计数、距离、成本和四个联合效用均值，并移除对固定 UAV 数量
  的诊断索引假设。

## 三项必要测试

1. 无移动：PASS。构造未来缺口但将 UAV 固定更新成本设为明显高于位置收益，最终为
   `A_keep`，实际移动距离和 UAV 成本均为 `0`；另验证无预测元数据时候选集合仍只含当前位置。
2. 地面服务缺口：PASS。区域 A 已由 SV 服务，区域 B 无地面缓存服务；预测车辆偏好使 B
   的 `ground_service_gap=1.0`，UAV 从 `[0,0]` 移到单时隙可达的 `[125,0]`，选择 `D_joint`：

```text
J_keep       = 0.0000000000
J_cache_only = 0.0250758937
J_uav_only   = -0.0002430000
J_joint      = 0.0526975928
```

最终 UAV 为 `[125,0]`，并在 UAV 缓存中放入缺口服务；未修改真实环境状态后再执行最终一步。
3. 真实 SUMO 短运行：PASS。固定 `seed=20260811`、`N_V=10`、`N_S=2`、`N_U=1`，运行
   20 个 slot，策略接口输入为有限的全节点缓存检查和零 raw displacement：

```text
UAV: check_slots=20, candidate_move_slots=0, moved_slots=0,
     nodes_moved=0, move_distance=0, move_cost=0,
     candidate_count=56, ground_service_gap_sum=576.2750946746
Cache: check_slots=20, candidate_changed_slots=18, changed_slots=1,
       changed_nodes=1, changed_services=2
J_mean: keep=0.0114839943, cache_only=0.0114882764,
        uav_only=0.0114839943, joint=0.0114882764
Execution: Local=0 (0.0000), SV=3 (0.0283), RSU=39 (0.3679),
           UAV=11 (0.1038), Cloud=53 (0.5000)
failed=0, capacity_violations=0, NaN/Inf=0
```

短 SUMO 结果只作闭环 sanity check；UAV 没有移动是因为当前配置下所有移动候选的净收益未
超过保持原位，不据此调低移动成本或修改预测参数。

## 清理与停止点

- 未生成新的 CSV、临时脚本、debug 文件或训练 checkpoint。
- 测试产生的 `__pycache__` 在交付前清理。
- 本轮完成“需求预测 → 地面缺口 → 少量候选 → A/B/C/D → 最终缓存/UAV 状态”，停止在
  需求感知 UAV 位置自适应调整，不进入多 UAV 复杂优化或 DRL 升级。

## 第七阶段：短时隙任务卸载与资源分配验证及必要修正（2026-08-11）

### 本轮范围与成熟实现复用

本轮只验证 `C_exec/Q_exec → 可行执行节点 → Matching → KKT 资源分配 → 实际任务执行`，不修改缓存、UAV、轨迹、流行度预测、PPO/MAPPO 或实验参数。复用既有的任务候选节点构造、迁移/交换 Matching、计算与带宽 KKT、时延/能耗/任务效用、任务执行和 Cloud fallback。

### 最小修改

- `solver.py`：在进入 Matching 前，保留 Local/Cloud 规则，并过滤当前通信速率不为正的 SV/RSU/UAV 候选；Matching 候选评估继续调用实际 KKT 和实际任务效用。增加迁移、交换、KKT 失败和失败原因计数。
- `environment.py`：将上述 Matching/KKT/失败诊断计数写入 `step info`。
- `train.py`：校验、汇总并记录新增诊断字段，不改变训练算法。

该通信过滤是本轮唯一的执行链路修正：缓存命中只表示 `C_exec` 可用，不能替代当前时隙的物理通信可达条件。

### Local=0 根因（A/B/C/D）

结论为 **D：Local 已进入候选，但 Matching 的实际效用比较没有选择 Local**。诊断短跑中共有 5 个 Local 可行任务，但最终 Local=0；不是 A（没有 Local 候选）、不是 B（Local 被错误过滤）、也不是 C（Local 执行分支缺失）。Matching 从 Cloud 初始化后，对候选迁移重新计算 KKT、时延、能耗和 `task_utility`，只有真实效用严格提升才接受，因此 Local 未被强制优先。

Cloud 的 53 个任务中，32 个没有任何边缘缓存候选，属于正常 Cloud fallback；其余 21 个虽有可达边缘缓存，但在统一真实效用评估下 Cloud 未被边缘候选严格超过，不存在“固定 Cloud 50%”或人为 Cloud 惩罚。

### KKT 验证

对固定分配的 3 任务、1 个边缘节点分别验证计算资源和带宽资源 KKT。CVXPY 未安装，因此仅用 SciPy SLSQP 做一次性 reference 对照，未进入正式代码路径。

```text
计算资源：KKT objective = 0.699411254999994
          reference objective = 0.6994112549695455
          gap = 3.04e-11
带宽资源：KKT objective = 0.5860555501823033
          reference objective = 0.5860555501952452
          gap = 1.29e-11
资源和：compute = 0.9999999999564617
        bandwidth = 1.0000000000220672
```

两类资源均非负，资源和满足不超过 1 的约束；短跑中 KKT failure count=0。

### Matching 反馈验证

既有 Matching 闭环已满足要求：从 Cloud 初始化，候选迁移/交换均重新执行资源分配和实际任务效用评估，只接受 `U_after > U_before + tolerance` 的操作；接受后从新匹配重新评估后续候选。本轮仅增加计数，没有重写算法。

代表性测试结果：真实效用由 `U_before=0.4995` 提升到 `U_after=0.899` 时接受 1 次迁移；效用相同的测试保持原匹配，`migration=0, swap=0`。

### SUMO 短跑验证

固定 `seed=20260811`、`N_V=10`、`N_S=2`、`N_U=1`，运行 20 个 slot；不训练、不扩大实验规模。

```text
任务总数 = 106
Local = 0，SV = 3，RSU = 39，UAV = 11，Cloud = 53，Failed = 0
任务守恒：0 + 3 + 39 + 11 + 53 + 0 = 106，PASS
执行比例：Local 0.0000，SV 0.0283，RSU 0.3679，UAV 0.1038，Cloud 0.5000
平均时延 = 0.09230971368575472
平均能耗 = 0.0027570317823782556
平均任务效用 = 0.9700299921226158
Matching migration = 53，swap = 4
KKT failure = 0
failure reasons = no_feasible_node:0,
                  resource_infeasible:0,
                  latency_constraint:0,
                  communication_unavailable:0,
                  numerical_failure:0
capacity violations = 0，NaN/Inf = 0
```

### 清理与停止点

- 未修改缓存、UAV、轨迹、流行度预测、PPO/MAPPO 和参数配置。
- 未新增训练 checkpoint、CSV、临时脚本或 debug 输出。
- 已清理本轮测试生成的 `__pycache__`。
- 本轮停止在短时隙任务卸载与资源分配验证；不继续进行算法升级或大规模实验。

## 第八阶段：无强化学习 Reference Solver 整体闭环验收（2026-08-11）

### 测试配置与结论

本轮只运行无 PPO 更新的 Reference Solver 正式闭环。测试使用项目真实 SUMO 链，固定 `seed=20260811`，`N_V=10`、`N_S=2`、默认 `N_R=2`、`N_U=1`、`F=20`，主验收 100 个 slot。`F=20` 和 `N_U=1` 仅在内存测试配置中设置，未修改 `config.yaml`。

闭环顺序通过：

```text
SUMO → 车辆角色 → 道路约束预测 → 服务需求预测
→ 协同缓存候选 → 自适应缓存 → UAV候选判断
→ C_exec/Q_exec → 可行节点 → Matching → KKT
→ execute_task → reward → 历史更新
```

Reference Solver 结论：**PASS**。

### 因果、状态提交与目标尺度

- 当前 slot 请求在预测、缓存/UAV决策和任务执行完成后才登记；动态预测审计得到 `future request leakage=0`。
- 主路径无离线未来轨迹输入，RoadConstrainedVehiclePredictor 只接收当前 SUMO 状态；`future trajectory leakage=0`。
- 候选评价只使用临时缓存/位置副本；每个 slot 最终只提交一次 `C_exec` 和一次 `Q_exec`。缓存和 UAV 的最终状态对账通过。
- 服务车辆替换先清空新 SV 缓存行，再进入本 slot 的 `C_old`；新 SV 不继承旧 SV cache，替换不额外增加 adaptive cache trigger。
- `predicted_cache_utility` 使用相对于 Cloud 的正边缘服务收益，实际任务效用对更低时延/能耗给出更高值，最终 reward 使用 `Q-C_cache-C_uav`；方向一致，未增加调节系数。

### 100-slot 结果

```text
Tasks = 506
Local = 0       (0.000000)
SV = 0          (0.000000)
RSU = 113       (0.223320)
UAV = 34        (0.067194)
Cloud = 359     (0.709486)
Failed = 0      (0.000000)

mean_delay = 0.11957618333729875
mean_energy = 0.00334316421687935
mean_task_utility = 0.9617480436234771
system_reward = 486.6445100734794
```

任务守恒：`0 + 0 + 113 + 34 + 359 + 0 = 506`，通过。

### Cache

```text
cache_check_slots = 100
cache_candidate_changed_slots = 93
cache_changed_slots = 0
cache_changed_nodes = 0
cache_changed_services = 0
cache_switch_cost = 0.0
```

候选缓存有变化但最终 `C_exec` 在 100 个 slot 内均未超过更新成本阈值，因此没有实际提交；实际 cache state 不变且物理缓存成本为 0。

### UAV

```text
uav_check_slots = 100
uav_candidate_move_slots = 0
uav_moved_slots = 0
uav_move_distance = 0.0
uav_move_cost = 0.0
```

UAV 候选检查正常执行，当前位置始终作为合法候选，净收益比较没有选择移动；不将 UAV 不移动判定为错误。

### 动态车辆与 Solver

```text
service_vehicle_replacement_count = 3
matching_migration = 147
matching_swap = 0
kkt_failure = 0
```

替换事件的缓存行初始化审计通过；没有发现旧 SV cache 被新 SV 继承的情况。

### 约束与数值

```text
cache capacity violations = 0
bandwidth violations = 0
compute violations = 0
UAV region violations = 0
UAV movement violations = 0
UAV safety-distance violations = 0
invalid service IDs = 0
invalid node IDs = 0
NaN/Inf = 0

future request leakage = 0
future trajectory leakage = 0
task conservation = PASS
```

缓存成本、UAV成本、最终 reward 与实际提交状态重新对账均通过；角色刷新后的 `C_old` 是本 slot 正式成本基准。

### 固定 seed 复现

未再运行第二次 100-slot 正式 episode；使用同一 seed 的 20-slot 前缀复现，任务车辆 ID、服务车辆 ID、缓存决策、UAV决策和逐 slot Local/SV/RSU/UAV/Cloud 计数完全一致：

```text
20-slot fixed-seed replay = PASS
20-slot tasks = 106
20-slot counts = Local 0, SV 0, RSU 15, UAV 7, Cloud 84, Failed 0
```

### 修改与清理

- 未修改核心算法；未修改 Matching、KKT、缓存、UAV、轨迹、流行度预测、reward 或 PPO。
- 本轮没有产生业务代码、baseline、临时 trace、CSV、helper 或 checkpoint。
- 已清理测试生成的 `__pycache__`。
- 本轮完成 Reference Solver 整体闭环验收，具备进入 PPO 高层缓存控制器接入阶段的条件；不在本轮实现 PPO 或开展正式长训练。

## 第九阶段：PPO 高层自适应缓存控制器接口（2026-08-11）

### 成熟实现与模式

本轮先检查运行环境：

```text
stable_baselines3 = 未安装
gymnasium = 未安装
torch = 未安装
```

因此没有从零实现 PPO clip、GAE、actor、critic、buffer 或 optimizer，也没有调用 `learn`/`optimizer.step`。新增 `ppo_env.py` 只提供后续接入官方 SB3 PPO 所需的 Gymnasium 形状契约；依赖缺失时使用不冒充 Gymnasium 的本地 fallback space 做动作契约测试。

项目原有 `policy.py`、`ppo.py`、`train.py` 仍相互引用，属于既有 Factorized/HAPPO 历史训练实现。本轮未贸然删除以免破坏现有入口，也未让它参与新 PPO wrapper 或任何测试；正式 SB3 依赖安装后再单独迁移/隔离旧训练入口。

配置新增：

```yaml
cache_controller:
  mode: reference   # reference 或 ppo

ppo:
  enabled: false
  seed: 20260811
```

`reference` 模式保留原 `G_cache > 0` 自适应触发；`ppo` 模式只替换高层缓存触发决定，下面的协同缓存、UAV、Matching、KKT、任务执行和 reward 均复用原实现。

### PPO 状态

新增 `AdaptiveCachePPOEnv`，状态固定为 8 个 `float32` 特征，不输入完整缓存矩阵、Matching 矩阵、KKT 变量或车辆逐个状态：

```text
s_t = [
  J_keep,
  J_update,
  delta_predicted_utility,
  candidate_cache_cost,
  changed_service_ratio,
  demand_shift,
  cache_age,
  mobility_index
]
```

状态 preview 只使用截至 `t-1` 的历史预测、当前因果车辆预测、缓存候选和 UAV 候选评价；不读取当前 `request(t)`、实际任务效用或当前执行结果。`J_*` 只用于 state/diagnostic，不作为 reward。

### PPO 动作与 A/B/C/D 分组

```text
action_space = Discrete(2)
0 = KEEP_CACHE
1 = UPDATE_CACHE
```

```text
action 0 → max(A, C)：C_exec = C_old
action 1 → max(B, D)：C_exec = C_candidate
```

PPO action=1 时不再由 `G_cache > 0` 否决候选；候选相同时允许实际 no-op，且 cache cost=0。UAV 仍由确定性候选比较在组内选择，PPO 不输出缓存矩阵或连续 UAV 位移。

### Reward

直接复用原有真实 `system reward = Q - C_cache - C_uav`，来自实际 `C_exec/Q_exec → Matching → KKT → execute_task`。没有 PPO update bonus、cache hit bonus、Cloud penalty 或其他 reward shaping。

### 测试结果

官方 SB3 `check_env`：由于 `gymnasium` 未安装，结果为 `unavailable:gymnasium_missing`；本地接口契约检查通过：

```text
action_space = fallback Discrete(2)
observation shape = (8,)
observation dtype = float32
observation finite = true
local contract = PASS
```

在同一 seed 的确定性重放状态（slot=1）上比较两个动作：

```text
tested decision states = 1
action 0 cache changes = 0
action 1 cache changes = 1
action 1 candidate changed = 1
ppo_action_mismatch = 0
capacity violations = 0
```

代表状态中，action 0 与 action 1 的 reward 分别为 `5.7684281264618775` 和 `5.731312205376889`；action 1 当前 reward 较低是允许的，关键是缓存控制动作确实改变了 `C_exec`。

KEEP scripted smoke（20 slot）：

```text
KEEP run = PASS
Tasks = 106
Local = 0, SV = 0, RSU = 15, UAV = 7, Cloud = 84, Failed = 0
ppo_keep_count = 20
ppo_effective_update_count = 0
ppo_action_mismatch_count = 0
task conservation = PASS
NaN/Inf = 0
KKT failure = 0
```

UPDATE scripted smoke（20 slot）：

```text
UPDATE run = PASS
Tasks = 106
Local = 0, SV = 1, RSU = 26, UAV = 6, Cloud = 73, Failed = 0
ppo_update_count = 20
ppo_effective_update_count = 19
ppo_update_noop_count = 1
ppo_action_mismatch_count = 0
task conservation = PASS
NaN/Inf = 0
KKT failure = 0
```

Reference 模式回归：

```text
reference mode replay = PASS
20-slot counts = Local 0, SV 0, RSU 15, UAV 7, Cloud 84, Failed 0
```

### 修改与清理

- 新增 `ppo_env.py`，修改 `environment.py`、`solver.py` 和 `config.yaml`；核心底层求解器逻辑保持复用，仅增加 PPO 动作分组和强制 UPDATE 分支。
- 未实现或调用 PPO 训练；未删除 Reference Solver。
- 未删除旧 Factorized/HAPPO 文件，已明确不与新 wrapper 混用，待 SB3 依赖和正式迁移方案确定后再隔离。
- 未生成 checkpoint、CSV、snapshot 或临时 helper；测试后的 `__pycache__` 已清理。
- 本轮停止在 PPO 环境、状态、动作、reward 和动作有效性接口，不进入正式 PPO 训练。
## 第十阶段：PPO 训练环境准备与正式入口整理（2026-08-12）

### Python 环境

本项目固定使用 Conda 环境：

```text
Environment = adaptive_cache_mec
Python = 3.11.15
Executable = C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe
```

本轮没有向 base 环境安装依赖。目标环境已有 `torch 2.11.0+cu130`，并新增：

```text
gymnasium = 1.3.0
stable_baselines3 = 2.9.0
CUDA available = True
GPU = NVIDIA GeForce RTX 5060 Laptop GPU
```

### SB3 接口与入口

- `AdaptiveCachePPOEnv` 保留 8 维 `float32` 状态和 `Discrete(2)` 动作空间。
- 新增唯一正式入口 `train_ppo.py`，当前只负责依赖检查、环境检查、SB3 PPO 构造和一次 `predict -> env.step` smoke。
- Reference 模式仍由 `config.yaml` 的 `cache_controller.mode=reference` 独立控制；PPO 入口只修改内存配置副本为 `ppo`。
- 本轮未调用 `model.learn`，也未调用 `optimizer.step`；没有生成新的 checkpoint、训练 CSV 或训练结果。

### 旧 PPO 代码

审计确认 `policy.py -> ppo.py -> train.py` 只属于旧 Factorized/HAPPO 链路，项目其余正式模块没有引用。本轮删除这三份旧入口/实现，保留 `ppo_env.py` 作为当前 SB3 环境适配层，避免新旧 PPO 混用。

### 验收结果

```text
official check_env = PASS
PPO construction = PASS
predict -> env.step = PASS
observation shape = (8,)
observation dtype = float32
reward = finite
model.learn = 0
optimizer.step = 0
20-slot reference replay = PASS
tasks = 106
Local = 0, SV = 0, RSU = 15, UAV = 7, Cloud = 84, Failed = 0
```

本阶段到此停止，不进入正式 PPO 训练和超参数调优。
## 第十一阶段：PPO 第一次受控训练（2026-08-12）

### 训练配置

本阶段只执行一次短技术验收，不代表 PPO 已经收敛，也不进入长训练：

```text
total_timesteps = 512
PPO updates = 2
episodes_started = 6
episodes_completed = 5
train_seed_base = 20260811
train_episode_seeds = 20260811 ... 20260816
eval_seed = 20261811
device = cpu
policy = MlpPolicy
action_space = Discrete(2)
observation shape = (8,)
```

使用 SB3 `VecNormalize`，只规范化 observation，不规范化真实 reward：

```text
norm_obs = true
norm_reward = false
clip_obs = 10.0
```

PPO 参数为 `learning_rate=3e-4`、`n_steps=256`、`batch_size=64`、
`n_epochs=10`、`gamma=0.99`、`gae_lambda=0.95`、`clip_range=0.2`、
`ent_coef=0.0`、`vf_coef=0.5`、`max_grad_norm=0.5`、`target_kl=None`。

### 训练动作与 PPO 指标

```text
KEEP = 246
UPDATE = 266
effective UPDATE = 171
UPDATE noop = 95
action mismatch = 0
rollout/ep_rew_mean = 490.93418992757796
rollout/ep_len_mean = 100.0
```

最后一次 SB3 更新指标全部为有限值：

```text
approx_kl = 0.0005146348848938942
clip_fraction = 0.0
entropy_loss = -0.6895104080438614
explained_variance = 0.0011938810348510742
policy_gradient_loss = -0.0015582389198243618
value_loss = 4595.000927734375
```

代表状态的策略概率在训练前后确实发生了变化，但没有出现立即塌缩为单一动作；
例如 `slot_1_delta<0` 的 `P(KEEP)` 从 `0.5009301` 变为 `0.5102737`，
`slot_0_delta≈0` 的 `P(UPDATE)` 从 `0.5021384` 变为 `0.5656320`。

### 正确性检查

```text
official check_env = PASS
NaN/Inf = 0
future request leakage = 0
future trajectory leakage = 0
capacity violations = 0
UAV violations = 0
KKT failure = 0
action mismatch = 0
```

KEEP/UPDATE 均通过真实环境执行；UPDATE 与候选缓存相同的 95 个时隙按约定记为
合法 no-op，未使用 action masking，也没有修改 reward。

### Held-out 技术评估

在独立 `eval_seed=20261811` 的 100-slot episode 上，使用保存的 checkpoint 和
冻结的 `vecnormalize.pkl` 做 deterministic `model.predict`：

```text
Reference reward = 490.2126099880571
PPO reward = 490.9276896715164
```

```text
Reference: Local 0, SV 9, RSU 164, UAV 103, Cloud 230, Failed 0
PPO:       Local 0, SV 15, RSU 168, UAV 124, Cloud 199, Failed 0
PPO KEEP = 63, UPDATE = 37
PPO effective UPDATE = 17
PPO UPDATE noop = 20
PPO cache_changed_slots = 17
PPO uav_moved_slots = 7
```

该结果只有一个 held-out seed，只能说明本次技术链路的方向性结果，不能得出 PPO
优于或劣于 Reference 的正式实验结论。

### Reference 回归

训练完成后重新执行既有固定 20-slot Reference replay：

```text
20-slot reference replay = PASS
Tasks = 106
Local = 0, SV = 0, RSU = 15, UAV = 7, Cloud = 84, Failed = 0
NaN/Inf = 0
KKT failure = 0
```

### 正式产物与停点

本次 run 目录为：

```text
outputs/ppo_training/20260812_104115/
```

已保存：`model.zip`、`vecnormalize.pkl`、`config_snapshot.json`、
`training_summary.json`。未生成中间 checkpoint、临时 debug 日志或训练图，
本阶段在 512 timesteps 后停止，未继续执行 5k/10k/20k 长训练。
## 第十二阶段：PPO 中等规模学习有效性验证（2026-08-12）

### 本阶段边界

本阶段使用已经通过第十一阶段验证的 PPO 实现，未修改 8 维状态、KEEP/UPDATE 动作、真实 system reward、底层缓存/UAV/Matching/KKT/预测模块和物理参数。训练从全新模型开始，未继续第十一阶段的 512-step checkpoint；总步数固定为 5120，不进行 10k 以上训练、超参数搜索或统计显著性检验。

### 训练配置与记录

```text
total_timesteps = 5120
PPO updates = 20
n_steps = 256
episodes_started = 52
episodes_completed = 51
train_episode_seeds = 20260811 ... 20260862
device = cpu
policy = MlpPolicy
action_space = Discrete(2)
observation_shape = (8,)
VecNormalize: norm_obs=true, norm_reward=false, clip_obs=10.0
```

沿用第十一阶段参数：`learning_rate=3e-4`、`batch_size=64`、`n_epochs=10`、`gamma=0.99`、`gae_lambda=0.95`、`clip_range=0.2`、`ent_coef=0`、`vf_coef=0.5`、`max_grad_norm=0.5`、`target_kl=None`。`update_metrics.json/csv` 均核对为 20 行，timesteps 从 256 递增到 5120。

训练累计动作及真实环境结果如下：

```text
KEEP = 2489
UPDATE = 2631
effective UPDATE = 1755
UPDATE noop = 876
action mismatch = 0
cache_changed_slots = 1755
cache_changed_services = 3639
uav_moved_slots = 388
task_total = 25912
system_reward = 25133.484758853912
```

最后一个 update 的 SB3 指标均为有限值：

```text
approx_kl = 0.0001518030185252428
clip_fraction = 0.0
entropy_loss = -0.6898181900382042
explained_variance = -0.003600478172302246
policy_gradient_loss = -0.0004997689975425601
value_loss = 3790.3753051757812
```

Critic 观察点为：`explained_variance` start/mid/end = `0.006603 / -0.018557 / -0.003600`，`value_loss` start/mid/end = `4821.8585 / 4536.1377 / 3790.3753`。因此 value loss 有下降，explained variance 仍接近 0 且未形成稳定正趋势；本阶段不设置固定 EV 阈值，也不把该结果表述为 PPO 已收敛。

### 固定真实 probe 状态

probe 状态来自训练 seed 的自然环境轨迹，包含 `delta<0`、`delta≈0`、`delta>0` 及不同 slot/缓存年龄/需求状态。记录点为 update 0、5、10、15、20，均保存 `P(KEEP)`、`P(UPDATE)` 和 `V(s)`；策略未塌缩为单一动作，且概率随状态与训练阶段发生变化。完整数值在 `training_summary.json` 的 `probe_states` 字段中。

### 正确性验收

```text
official check_env = PASS
action mismatch = 0
future request leakage = 0
future trajectory leakage = 0
cache capacity violations = 0
compute violations = 0
bandwidth violations = 0
UAV violations = 0
NaN/Inf = 0
KKT failure = 0
```

训练后 20-slot Reference 回归仍通过：`Tasks=106`，`Local=0, SV=0, RSU=15, UAV=7, Cloud=84, Failed=0`，NaN/Inf 和 KKT failure 均为 0。

### 五个未见 seed 的 held-out 对照

评估 seed 为 `20261811 ... 20261815`，每个 seed 运行 100 slots；PPO 使用保存后冻结的 `vecnormalize.pkl`，并在 `PPO.load` 后显式重新设置 held-out seed，避免模型自身训练 seed 覆盖评估 seed。

```text
seed       Reference reward     PPO reward        PPO-Reference
20261811   490.212610           490.521164        +0.308554
20261812   491.626590           491.556998        -0.069592
20261813   491.001626           491.565144        +0.563518
20261814   491.525094           491.525098        +0.000004
20261815   491.184541           490.132881        -1.051661
```

```text
mean Reference reward = 491.110092
mean PPO reward = 491.060257
mean delta = -0.049835
std delta = 0.549759
PPO wins = 3 / 5
```

该结果只能说明当前 PPO checkpoint 完成了多 seed 技术链路验证，不能宣称 PPO 优于 Reference。

### 产物与停点

本阶段正式 run 目录为：

```text
outputs/ppo_training/20260812_105817/
```

已保存 `model.zip`、`vecnormalize.pkl`、`config_snapshot.json`、`training_summary.json`、`update_metrics.json`、`update_metrics.csv` 和 `evaluation_summary.json`。第十一阶段目录 `outputs/ppo_training/20260812_104115/` 保持不变。

综合判断：本阶段学习链路技术验收为 `PASS`，但 critic 尚未显示稳定学习趋势，五个 held-out seed 的平均 reward 也未超过 Reference。因此当前结论是：可以进入结果分析和后续受控实验，不可以据此直接启动更长 PPO 训练并宣称性能优势；若要升级到更长训练，应先由后续实验明确提出需求。
## 第十三阶段：PPO 有限时域状态修正与 Critic 受控复验（2026-08-12）

### Episode 结束语义

论文和项目指标把训练/测试定义为固定的有限时隙窗口 `T_e/T_te`；`simulation.slot_count=100` 不是无限系统的任意截断。现有环境实测最后一个动作（slot 99）返回：

```text
100-slot horizon = finite optimization episode
terminated = True
truncated = False
```

本阶段没有修改该结束语义；`ppo_env.py` 仍把底层 `terminated` 原样返回，并固定 `truncated=False`。

### 理论修正与新状态

第十二阶段旧状态和第十三阶段新状态都保持 8 维：

```text
old = [J_keep, J_update, delta_J, candidate_cache_cost,
       changed_service_ratio, demand_shift, cache_age, mobility_index]

new = [J_keep, J_update, episode_progress, candidate_cache_cost,
       changed_service_ratio, demand_shift, cache_age, mobility_index]
```

`delta_J` 可以由 `J_update - J_keep` 精确恢复，和前两个输入存在冗余；有限时域 Critic 还需要知道当前处于回合哪一段。因此第三维改为：

```text
episode_progress = min(slot_index, T-1) / (T-1)
```

其中 `T` 读取当前配置的 `simulation.slot_count`，没有写死 100；本配置下覆盖 `[0,1]`。没有增加第 9 维，也没有把完整缓存矩阵、车辆坐标、Matching 或 UAV 动作放入 PPO。`norm_obs=True`、`norm_reward=False` 保持不变，未做 reward normalization。

短验证结果：官方 `check_env=PASS`；自然 episode 的 probe progress 为 `0.0、0.242424、0.494949、0.747475、1.0`。

### 训练配置

本阶段从头创建 PPO 模型，未加载第十一/十二阶段模型继续训练。训练 seed 序列、5 个 held-out seed、网络和 PPO 参数与第十二阶段完全一致：

```text
timesteps = 5120
updates = 20
training seeds = 20260811 ... 20260862
held-out seeds = 20261811 ... 20261815
device = cpu
MlpPolicy + Discrete(2)
learning_rate = 0.0003
n_steps = 256
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.0
vf_coef = 0.5
max_grad_norm = 0.5
norm_obs = true
norm_reward = false
```

### Critic 最小诊断

每个 PPO update 只保存一行 reward/return/value 的均值、标准差、RMSE、Pearson、EV 和 value loss；没有保存完整 transition。每个 rollout 只保存 8 维原始状态的 mean/std/min/max。Pearson 在零方差时记录 `None + status`，不产生 NaN 传播。

旧版第十二阶段没有保存 rollout buffer 的 return/value 统计，因此旧 Pearson、return std、value std 明确为 unavailable；旧 EV/value loss 来自第十二阶段结果文件。

```text
                         Old Stage12          New Stage13
EV start                 0.006603             0.010272
EV mid                   -0.018557            0.112092
EV end                   -0.003600            0.032946
Pearson start            unavailable          0.285242
Pearson mid              unavailable          0.490056
Pearson end              unavailable          0.670176
value loss start         4821.8585            4810.0475
value loss mid           4536.1377            4471.2457
value loss end           3790.3753            3788.3912
```

新版本 Pearson 的 20-update 序列均为有限值，整体从 `0.285242` 上升至 `0.670176`，update 15 达到 `0.719223`；value-return RMSE 从 `69.6101` 降至 `62.0960`。新版本 `return_std` start/mid/end 为 `20.2185 / 20.9931 / 24.6848`，`value_std` start/mid/end 为 `0.3763 / 2.7752 / 0.6183`。value std 与 return std 的跨-update Pearson 为 `-0.1474`，因此“value std 跟随 return std”不是本次改善依据；主要改善证据是 Pearson 正趋势和 EV 相比旧版的改善。

结论：在本次受控规模下，有限时域信息缺失是旧 Critic 难以学习的重要原因之一，Critic 判定为 `明显改善`。这不是 PPO 已经优于 Reference 的结论。
### Actor 与正确性

```text
KEEP = 2477
UPDATE = 2643
effective UPDATE = 1807
UPDATE noop = 836
action mismatch = 0
```

```text
future request leakage = 0
future trajectory leakage = 0
cache capacity violations = 0
compute violations = 0
bandwidth violations = 0
UAV violations = 0
NaN/Inf = 0
KKT failure = 0
```

### 五个 held-out seed：Reference / Old PPO / Time-aware PPO

Reference 与旧 PPO 直接复用了第十二阶段完整结果，本阶段只新运行 Time-aware PPO；每个 PPO 评估均使用冻结的 VecNormalize 统计，并显式设置对应 held-out seed。

```text
seed       Reference       Old PPO        Time-aware PPO   Old Δ        New Δ
20261811   490.212610      490.521164     490.721709       +0.308554    +0.509099
20261812   491.626590      491.556998     491.556998       -0.069592    -0.069592
20261813   491.001626      491.565144     491.532448       +0.563518    +0.530822
20261814   491.525094      491.525098     491.242080       +0.000004    -0.283014
20261815   491.184541      490.132881     490.726810       -1.051661    -0.457731
```

```text
Old mean Δ = -0.049835
New mean Δ = +0.045917
Old std Δ = 0.549759
New std Δ = 0.406170
Old wins = 3 / 5
New wins = 2 / 5
```

Time-aware PPO 的平均 reward 略高于 Reference，但只有 2/5 seed 获胜；该表用于状态修正和 Critic 诊断，不宣称 PPO 优于 Reference。

### 产物、清理与停点

本阶段 run：

```text
outputs/ppo_training/20260812_113235/
```

已保存 `model.zip`、`vecnormalize.pkl`、`config_snapshot.json`、`training_summary.json`、`update_metrics.json`、`update_metrics.csv`、`evaluation_summary.json`。已保留第十一、十二阶段目录，未覆盖历史 checkpoint；已清理临时诊断脚本、失败空 run、debug/log 文件和 `__pycache__`。

本阶段在 5120 steps 后停止，不继续 10k/20k/50k 训练。结论为：`Critic明显改善`；下一阶段可以进入更长训练，但本轮不自行启动。

## 第十四阶段：Reward Normalization 单变量受控验证（2026-08-12）

### 单变量边界与正式配置

本阶段只改变一个变量：`VecNormalize(norm_obs=True, norm_reward=False)` 改为 `VecNormalize(norm_obs=True, norm_reward=True)`；`clip_obs=10.0`、Stage13 的 8 维状态、Reference/底层环境、网络、全部 PPO 超参数、训练 seed 和 held-out seed 均未改变。PPO 从头创建，没有加载 Stage13 模型续训。

```text
state = [J_keep, J_update, episode_progress, candidate_cache_cost,
         changed_service_ratio, demand_shift, cache_age, mobility_index]
timesteps = 5120, updates = 20, device = cpu
training seeds = 20260811 ... 20260862
held-out seeds = 20261811 ... 20261815
MlpPolicy + Discrete(2)
learning_rate=0.0003, n_steps=256, batch_size=64, n_epochs=10
gamma=0.99, gae_lambda=0.95, clip_range=0.2
ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, target_kl=None
```

训练期通过 `info["ppo_raw_reward"]` 单独保存未归一化的真实 system reward；SB3 rollout、return 和 value 使用官方 `VecNormalize` 的 normalized reward。held-out 评估加载本阶段 `vecnormalize.pkl` 后设置 `training=False` 且 `norm_reward=False`，五 seed 对照全部使用 raw system reward。

### 训练结果与 Critic 诊断

正式 run 目录为 `outputs/ppo_training/20260812_115856/`。训练完整完成：`5120 timesteps / 20 updates / 52 episodes started / 51 episodes completed / CPU`。每个 update 的 raw/normalized reward mean/std/min/max、return/value、RMSE、Pearson、EV、value loss、`value_return_scale_ratio`、SB3 actor 指标和 8 维状态范围均写入 `update_metrics.json/csv`；没有保存 transition trace。

`value_return_scale_ratio = value_std / (return_std + 1e-12)`。Stage13 与 Stage14 start/mid/end 对比如下：

```text
metric                   Stage13 start / mid / end          Stage14 start / mid / end
explained_variance       0.010272 / 0.112092 / 0.032946     -0.036475 / 0.964265 / 0.987650
value_return_pearson     0.285242 / 0.490056 / 0.670176      0.076679 / 0.982112 / 0.993836
return_std               20.218499 / 20.993141 / 24.684840   1.332239 / 0.886431 / 1.041949
value_std                0.376332 / 2.775242 / 0.618306      0.376332 / 0.855764 / 1.043637
scale_ratio              0.018613 / 0.132198 / 0.025048      0.282481 / 0.965404 / 1.001619
```

Stage14 raw reward mean/std 为 `4.899208/1.406885 -> 4.884546/1.402639 -> 4.940373/1.360726`；normalized reward mean/std 为 `0.137694/0.640488 -> 0.056611/0.016234 -> 0.057338/0.015784`。EV、Pearson 和 scale ratio 明显改善，scale ratio 到 1 的平均对数距离由 Stage13 的 `2.710741` 降为 `0.188047`；但 Pearson 全序列标准差由 `0.278983` 增至 `0.391263`，稳定性判据未通过。

### Actor 与固定 probe

```text
KEEP=2475, UPDATE=2645, effective UPDATE=1662, UPDATE noop=983
action mismatch=0, last approx_kl=0.0031916
last clip_fraction=0.0082031, last entropy_loss=-0.5949353
```

KEEP/UPDATE 均有大量样本，未发生 actor 单一动作塌缩。Stage13 的五个固定真实 probe 未重新选取；历史 summary 未持久化 raw observation，因此本阶段用相同训练 seed、相同 slot 抽样和相同环境确定性重构，并核对 slot/progress/delta 元数据后复用。Stage14 保存了 update `0/5/10/15/20` 的 `P(KEEP)`、`P(UPDATE)` 和 `V(s)`。固定五点的 `P(UPDATE)` 从 update 0 的约 `[.500,.501,.501,.502,.502]` 变为 update 20 的 `[.809,.104,.124,.127,.202]`，`V(s)` 从 `[0.000,-0.652,-0.895,-0.799,-1.102]` 变为 `[3.511,2.707,2.105,1.332,0.650]`，说明策略和 value 对状态位置发生了变化；绝对 V 数值不跨 reward scale 直接比较。

### 硬正确性验收

```text
official check_env=PASS
future request leakage=0, future trajectory leakage=0
cache capacity=0, compute=0, bandwidth=0, UAV=0
invalid service IDs=0, invalid node IDs=0
NaN/Inf=0, KKT failure=0, action mismatch=0
```

训练后 20-slot Reference 回归复用既有验收结果，状态为 `PASS`，`Tasks=106`，`Local=0, SV=0, RSU=15, UAV=7, Cloud=84, Failed=0`。

### 五个 held-out seed：Reference / Stage13 / Stage14

```text
seed       Reference       Stage13 PPO     Stage14 PPO     Stage13 Δ     Stage14 Δ
20261811   490.212610      490.721709      490.321428      +0.509099     +0.108818
20261812   491.626590      491.556998      491.699674      -0.069592     +0.073084
20261813   491.001626      491.532448      490.781584      +0.530822     -0.220042
20261814   491.525094      491.242080      491.518247      -0.283014     -0.006847
20261815   491.184541      490.726810      491.180257      -0.457731     -0.004284
```

```text
Stage13 mean PPO reward=491.156009, Stage14 mean PPO reward=491.100238
Stage13 mean Δ=+0.045917, Stage14 mean Δ=-0.009854
Stage13 std Δ=0.406170, Stage14 std Δ=0.114176
Stage13 wins=2/5, Stage14 wins=2/5
```

Stage14 raw held-out mean reward 比 Stage13 低 `0.055771`，mean reward delta 由正变负；raw system reward 无退化判据未通过。没有进行 t-test、Wilcoxon 或 p-value 推断。

### 最终决策、产物与停点

本阶段受控实验技术完整性为 `Stage14 PASS`，最终推荐为：

```text
norm_reward=false
recommendation=restore_norm_reward_false
```

Critic EV/Pearson/scale ratio 和 actor 诊断改善，但 Pearson 稳定性、raw held-out system reward 无退化未同时满足；因此 reward scale 不是当前 system reward 性能的主要问题，项目默认配置已恢复 `norm_reward=false`。正式 Stage14 的 `config_snapshot.json` 仍记录 `norm_reward=true`，与当前默认配置保持可区分。

本阶段正式保留 `model.zip`、`vecnormalize.pkl`、`config_snapshot.json`、`training_summary.json`、`update_metrics.json`、`update_metrics.csv` 和 `evaluation_summary.json`。已清理项目内 `__pycache__`、临时脚本/trace/重复 CSV、失败空目录和 debug/log 文件；Stage11、12、13、14 正式 run 均保留。本阶段没有启动长训练，也没有自动延长 5120 steps；受控验证已完成，项目具备进入长训练参数准备阶段条件，长训练需后续单独授权和启动。

## Stage15：PPO 中等长度正式训练参数准备（2026-08-12）

### 最终固定算法与状态

正式方案冻结为：PPO 只决定 `KEEP_CACHE/UPDATE_CACHE`，协同边际收益算法决定缓存服务内容；UAV 位置继续由确定性算法求解，Matching/KKT 继续负责短时隙任务关联与资源分配。PPO 动作空间仍为 `Discrete(2)`，不加入完整 cache matrix、UAV 位置、任务卸载节点、带宽或 CPU 动作。

正式 observation 固定为 8 维 episode-progress 版本：

```text
[J_keep, J_update, episode_progress, candidate_cache_cost,
 changed_service_ratio, demand_shift, cache_age, mobility_index]
```

不恢复 `delta_J`，不增加第 9 维。`episode_progress` 的有限时域语义保持 Stage13 定义。

### 正式 PPO 参数

```text
state_dim = 8
norm_obs = true
norm_reward = true
clip_obs = 10.0
learning_rate = 0.0003
n_steps = 256
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.0
vf_coef = 0.5
max_grad_norm = 0.5
device = cpu
policy = MlpPolicy
action_space = Discrete(2)
total_timesteps = 25600
PPO updates = 100
```

`norm_reward=true` 只用于 SB3 训练层的数值稳定，继续直接使用官方 `VecNormalize`，不加入自定义 Critic target normalization、return normalization、manual reward scaling、value clipping 或 PPO/Critic 补丁。环境原始 reward 公式保持不变，最终评价只使用 original system reward。

Stage14 的 EV/Pearson/scale ratio 显著改善；5-seed raw reward 尚不足以证明 normalization 导致系统性退化。因此正式训练优先采用 `norm_reward=true`，但 validation、best checkpoint 和 test 全部严格按 raw system reward 判断。

### Training / Validation / Test 隔离

```text
training seed rule = train_seed_base + episode_index
train_seed_base = 20260811
training seeds = 20260811 ... 20261066
validation seeds = 20271811 ... 20271815
test seeds = 20281811 ... 20281820
```

训练共 256 个 100-slot episode，训练 seed 与 validation/test 不重叠。Validation 每 `5120` timesteps（约 20 updates）执行一次，共 5 个 checkpoint validation；每个 validation seed 同时运行相同 SUMO/task/system 配置下的 Reference 和当前 PPO，唯一差别为 cache controller。Reference 可在完全相同配置缓存后复用。

Test seed 在训练和 best checkpoint 选择期间禁止运行；只有 100 updates 完成并确定 best 后才运行一次。Test 不反向修改 Stage15 参数。

### Best 与 checkpoint 规则

Best model 唯一选择标准为：

```text
criterion = mean original validation system reward
best = argmax over validation checkpoints of mean_i(R_i_original)
```

禁止根据 normalized reward、training episode reward、EV、value loss、Pearson、Cloud rate 或单个 seed 选择 best。

只保存 update `20/40/60/80/100` 五个 checkpoint。每个 checkpoint 必须成对保存：

```text
checkpoint_<update>/model.zip
checkpoint_<update>/vecnormalize.pkl
best/model.zip + best/vecnormalize.pkl
final/model.zip + final/vecnormalize.pkl
```

禁止混合旧 model 与新 VecNormalize。每个 update 只记录必要标量诊断：`approx_kl`、`clip_fraction`、`entropy_loss`、`explained_variance`、`value_return_pearson`、`value_return_scale_ratio`、`value_loss`、KEEP/UPDATE/effective UPDATE/UPDATE noop/action mismatch，以及系统 reward、时延、能耗、Local/SV/RSU/UAV/Cloud/Failed、缓存变化和 UAV 移动指标。

### Hard-stop 与停止边界

训练累计检查 future request/trajectory leakage、cache capacity、compute、bandwidth、UAV、invalid service/node IDs、NaN/Inf、KKT failure 和 action mismatch。上述错误任一大于 0，或出现 SUMO/environment fatal failure，允许立即停止；EV 短时下降、Pearson 波动、单次 validation reward 下降、UPDATE 数量变化或 Cloud rate 变化均不得触发自动调参/停止。25600 steps 完成后必须停止，不自动 resume 50k/100k。

本阶段仅完成参数、入口、validation、checkpoint、best、test 隔离和非训练检查；正式 `model.learn(total_timesteps=25600)` 尚未启动，必须由用户本人执行唯一命令。
## Stage15A：离线 SUMO 轨迹与因果 Trace Replay（2026-08-12）

### 本阶段原因

Stage15 首次正式训练尚未启动。本阶段不调用 `model.learn`，只解决训练前的移动数据源问题：验证过程中不能重复启动同名 TraCI 连接；车辆移动又不受 PPO、缓存、UAV、Matching 或 KKT 动作影响，因此将 SUMO 车辆移动真值离线生成并在训练/验证/测试中因果回放。

### 统一接口与数据边界

新增 `mobility.py`，环境只依赖一个当前 slot 接口 `MobilityProvider`：

```text
MECEnvironment
    └── MobilityProvider
        ├── SumoMobilityProvider   # 轨迹生成和小规模 SUMO 对照
        └── TraceMobilityProvider  # Stage15 正式训练/验证/测试
```

`seed_<seed>.npz` 每个 seed 一个文件，schema version=1。文件只包含 SUMO 地面真值：slot/time、车辆 ID、active、x/y、speed、速度分量、angle、acceleration、allowed speed、edge、lane、lane position 和 lane index；不包含请求、未来 popularity、预测轨迹、车辆角色、缓存、UAV、Matching、KKT、资源分配或 reward。

现有 `VehicleRoleManager` 仍在 replay 运行时使用当前 active 状态逐 slot 选择任务车辆、服务车辆并执行离开补位；缓存和服务车辆替换逻辑没有冻结。请求仍由既有因果 workload 生成器按原 seed 序列提供。

### 因果性与 slot 语义

`TraceMobilityProvider` 只允许按 `slot=0,1,...` 顺序读取当前行，预测器继续只从已经观察到的历史和当前状态更新；未来 trace 行不会通过环境接口暴露。Trace 文件另保存 reset 时的 `t=95s` 初始观测，`slot 0..99` 保存环境第一次 `step` 推进到 `100s..595s` 后的 SUMO 真值，保持原环境的 reset/step 顺序和终止语义。

`mobility.source=trace` 缺失文件时直接报错，绝不回退启动 SUMO。正式 Stage15 配置已固定：`source=trace`、`trace_root=data/mobility_traces/default`。离线生成入口 `generate_mobility_traces.py` 只负责 SUMO 启动、状态记录和 NPZ 保存，不复制任何 solver 或 PPO 逻辑。

### 小规模验收

已生成并验证 `20260811`、`20260812` 两个 trace：

```text
SUMO/Trace 100-slot consistency: PASS for both seeds
active ID / x-y / speed / edge / lane: exact/discrete PASS
roles / requests / predictions / cache / UAV: PASS
Matching / KKT / execution distribution / reward: PASS
Trace-side SUMO starts: 0
```

20-slot Trace Reference regression（seed `20260811`）保持：`Tasks=106, Local=0, SV=0, RSU=15, UAV=7, Cloud=84, Failed=0, future request leakage=0, future trajectory leakage=0, KKT failure=0, NaN/Inf=0`。

本阶段状态：`Stage15A PASS`；正式 25600-step PPO 训练仍未启动，`model.learn calls=0`。下一步只生成 Stage15 全部 train/validation/test trace，不启动训练：

```powershell
& 'C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe' generate_mobility_traces.py --stage15
```

## Stage15B：正式 PPO 训练前 Trace 数据集与启动链路最终验收（2026-08-12）

本阶段只做验收，不调用 `model.learn`，不修改 PPO、缓存、UAV、Matching、KKT、请求、预测、车辆角色、物理参数或 Trace schema。

### Trace 数据集

```text
expected = 271
present = 271
missing = 0
corrupted = 0
schema mismatch = 0
seed mismatch = 0
slot mismatch = 0
NaN/Inf = 0
training = 256 (20260811 ... 20261066)
validation = 5 (20271811 ... 20271815)
test = 10 (20281811 ... 20281820)
```

271 个正式 NPZ 均通过轻量打开、字段集合、schema version、filename/internal seed、`slot_count=100`、`slots=0..99`、时间序列、shape 以及 active mask 有效数值检查。非 active 车辆的 padding 值不作为 NaN/Inf 缺陷。

### Seed 隔离与 Trace 容量

```text
train ∩ validation = 0
train ∩ test = 0
validation ∩ test = 0
training seed rule = 20260811 + episode_index, episode_index=0..255
256 training traces cover 25600 timesteps = YES
```

代码依据为 `validate_stage15_configuration` 的 256 个训练 seed、100-slot episode 和 `total_timesteps=25600` 映射；Stage15 不会把 validation/test seed 纳入训练生成器。

### Mobility 与启动链路

```text
mobility.source = trace
trace_root = data/mobility_traces/default
missing-trace fallback to SUMO = NO
missing trace behavior = explicit error
Stage15 training path starts = 0
validation Reference starts = 0
validation PPO starts = 0
test starts = 0
```

训练、Reference validation、PPO validation 和 final test 均沿用 Trace provider；本阶段只执行代表性 seed 的单时隙链路检查，没有运行完整 episode 或正式训练。缺失 Trace 的负向检查确认不会 fallback 到 SUMO。

### Reference/PPO 公平性与 PPO 冻结参数

同 seed 的 mobility、requests、vehicle roles 和 physical parameters 保持一致，唯一控制器差异为 cache controller；训练、validation 和 test 代表 seed 的 request/workload 与 roles 检查均为 SAME。

```text
state_dim = 8
state = [J_keep, J_update, episode_progress, candidate_cache_cost,
         changed_service_ratio, demand_shift, cache_age, mobility_index]
delta_J = absent
norm_obs = true
norm_reward = true
clip_obs = 10.0
learning_rate = 0.0003
n_steps = 256
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.0
vf_coef = 0.5
max_grad_norm = 0.5
device = cpu
total_timesteps = 25600
policy = MlpPolicy
action_space = Discrete(2)
parameter drift = 0
```

`episode_progress` 代表性链路检查值为 `0.0`，并满足 `[0,1]`。Validation 使用 `20271811..20271815`、`eval_freq=5120`，在 `n_steps=256` 下对应 20 updates；保留 update `20/40/60/80/100`，best criterion 为 `mean original validation system reward`。Test 使用 `20281811..20281820`，训练/选 best 期间不访问 test。

### Checkpoint、Hard-stop 与清理

正式训练输出尚未创建；不伪造 checkpoint。正式运行时只保存 `checkpoint_20/40/60/80/100`、`best`、`final`，每个目录均须成对包含 `model.zip` 和 `vecnormalize.pkl`。Evaluation 逻辑固定 `training=False`、`norm_reward=False` 并使用 original system reward。

Hard-stop 保留 future request/trajectory leakage、cache capacity、compute、bandwidth、UAV、invalid service/node ID、NaN/Inf、KKT failure 和 action mismatch；任一非零立即停止，不因 EV、Pearson、单次 validation reward 或 UPDATE 比例波动而自动调参。

```text
py_compile = PASS
config parse = PASS
trace dataset verification = PASS
automatic hyperparameter adjustment = NONE
model.learn calls = 0
Stage15 formal training started = NO
```

本阶段新增 Trace 文件级验收入口 `generate_mobility_traces.py --verify-stage15`；清理了本轮 `__pycache__` 和临时检查产物，保留 271 个正式 mobility traces 以及既有 Stage11–14 结果目录。Stage15B 状态：`PASS`。

## Stage15C：SB3 check_env 与有限 Trace seed 冲突修复（2026-08-12）

### 根因

Stage15 正式入口之前调用通用 `sb3_check_env()`；SB3 官方 checker 内部固定执行 `reset(seed=0)`，而正式离线 Trace 集合只包含训练、validation、test 的预生成 seed，因此错误访问了不存在的 `seed_0.npz`。这不是 Trace 数据集缺失问题。

### 修复

`ppo_env.py` 中通用 `sb3_check_env()` 保留不变，继续供开发接口检查使用。Stage15 `run_stage15_training()` 不再调用该 checker，改为使用 `settings["train_seeds"][0]` 创建独立临时 PPO environment，执行合法 seed 的一次 reset 和一次 KEEP step；该环境在正式 VecNormalize/训练环境创建前关闭。

```text
official check_env called reset(seed=0) = YES (原错误路径)
Stage15 official check_env skipped = YES
generic sb3_check_env preserved = YES
valid-seed preflight added = YES
preflight seed = 20260811
formal training first seed = 20260811
reset seeds observed = [20260811]
seed_0.npz generated = NO
seed_0 accesses = 0
seed remapping = NONE
missing trace fallback = NO
```

### Preflight 结果

```text
observation shape = (8,)
observation dtype = float32
observation finite = YES
observation in observation_space = YES
action space = Discrete(2)
step API = obs,reward,terminated,truncated,info
steps = 1
preflight environment independent = YES
VecNormalize polluted by preflight = NO
preflight TraCI starts = 0
formal trace training path TraCI starts = 0
```

Stage15 函数 AST 检查确认不再包含 `sb3_check_env()` 调用；PPO 状态、动作、超参数、validation/test、checkpoint 和 hard-stop 规则均未修改。已删除本次失败产生的 `outputs/ppo_training/20260812_135203` 目录、本轮 `__pycache__` 及临时检查产物；271 个正式 Trace 和既有 Stage11–14 结果保留。

```text
py_compile = PASS
config parse = PASS
Stage15 preflight = PASS
model.learn calls = 0
Stage15 formal training started = NO
Stage15C = PASS
```

## Stage15D：Validation/Test 固定 seed 与 VecEnv terminal 自动 reset 越界修复（2026-08-12）

### 根因

`FAILED_STAGE15_VALIDATION_SEED_RESET`：Stage15 checkpoint validation 使用 `validation seed = 20271815` 完成 100 slots 后，SB3 `DummyVecEnv` 在 terminal 后自动调用 wrapper 的 `reset(seed=None)`。旧 wrapper 将其按 `episode_seed_base + episode_index` 解释为 `20271816`，从而访问不存在的 `seed_20271816.npz`。问题属于 evaluation environment 生命周期，不属于 271 个 Trace 数据集缺失。

### 最小修复

`ppo_env.py` 增加 `seed_mode = sequential/fixed` 与 `fixed_seed`：

```text
Training = sequential seed mode
Validation = fixed seed mode
Test = fixed seed mode
```

训练 builder 继续使用 `episode_seed_base + episode_index`；Validation/Test builder 将当前评估 seed 绑定为 `fixed_seed`。因此 terminal 后的 VecEnv 自动 reset 仍使用当前 `S`，不会推进到 `S+1`。不修改 TraceMobilityProvider、不生成额外 Trace、不 fallback SUMO。

### 100-slot terminal 回归

```text
validation seed = 20271815
slots = 100
completed = YES
post-terminal reset seed = 20271815
20271816 access count = 0
reward finite = YES
TraCI starts = 0
```

回归路径为 `DummyVecEnv -> VecMonitor -> VecNormalize -> _run_ppo_episode`，使用独立临时 evaluation wrapper；终止后的 wrapper diagnostics 仍为 `trace_seed=20271815`。

### Training seed 回归与冻结配置

```text
training episode 0 = 20260811
training episode 1 = 20260812
sequential training seed = PASS
parameter drift = 0
validation seeds unchanged = YES
test seeds unchanged = YES
best checkpoint rule unchanged = YES
```

本轮未调用 `model.learn`，未重新启动正式 Stage15。已删除刚才失败的 `outputs/ppo_training/20260812_135935`（该目录包含 update 20 的不可复用 checkpoint，不作为正式结果，也不从中续训）；失败原因已记录为 `FAILED_STAGE15_VALIDATION_SEED_RESET`。Stage15D 状态：`PASS`。

```text
py_compile = PASS
config parse = PASS
terminal regression = PASS
model.learn calls = 0
Stage15 formal training restarted = NO
```

## Stage15E：Training Trace 容量与最后一个 terminal reset reserve（2026-08-12）

### 根因

`FAILED_STAGE15_TRAIN_RESET_RESERVE_MISSING`：原 Stage15B 仅按 `25600 / 100 = 256` 计算 training Trace 数量，遗漏了单环境 `DummyVecEnv` 在第 256 个正式 episode terminal 后自动请求的下一次 reset observation。该 reset 不是第 257 个 completed training episode，但需要额外的 training-side reserve Trace。

### 容量修正

```text
total_timesteps = 25600
n_steps = 256
n_envs = 1
effective env steps = 25600
episode horizon = 100
capacity formula = floor(effective_env_steps / episode_horizon) + 1
formal training traces = 256
training reset reserve = 1
reserve seed = 20261067
validation traces = 5
test traces = 10
total required = 272
```

正式 training seed 仍为 `20260811 ... 20261066`；`20261067` 只用于最后一个 terminal 后的自动 reset，不计入 completed training episodes，不复用 `20261066`，不 wrap around，也不扩展 validation/test seed 集合。

### 生成与验证

`generate_mobility_traces.py --stage15` 已将 reserve 纳入目标。已有 271 个 Trace 全部跳过，仅生成缺失的 `seed_20261067.npz`：

```text
generated = 1
skipped = 271
required = 272
```

`--verify-stage15` 已明确区分 formal training、training reset reserve、validation 和 test，并通过：

```text
expected = 272
present = 272
missing = 0
corrupted = 0
schema mismatch = 0
seed mismatch = 0
slot mismatch = 0
NaN/Inf = 0
formal training count = 256
training reset reserve count = 1
reserve seed = 20261067
validation count = 5
test count = 10
```

精确边界回归确认：

```text
formal episode0 = 20260811
formal episode255 = 20261066
post-terminal reset seed = 20261067
formal completed episodes = 256
reserve used as completed episode = NO
TraCI starts = 0
```

本轮未修改 PPO、state、action、reward、normalization、seed_mode、Environment、TraceMobilityProvider、VehicleRoleManager、缓存、UAV、Matching、KKT 或 validation/test 规则；参数检查为 `parameter drift = 0`。本次失败目录 `outputs/ppo_training/20260812_141626` 已删除，不从其部分结果续训；Stage15 修复后必须从零开始。

```text
py_compile = PASS
config parse = PASS
capacity boundary regression = PASS
trace dataset verification = PASS
model.learn calls = 0
formal Stage15 restarted = NO
Stage15E = PASS
```

## Stage15 正式训练前控制台输出与自动绘图验收（2026-08-12）

本轮只改进训练过程可观测性和 Stage15 成功后的 summary 后处理绘图，不修改 PPO、state、action、raw reward 公式、Trace、缓存、UAV、Matching、KKT、seed 规则或正式训练参数。

控制台输出已覆盖：

```text
startup summary = PASS
per-update progress = PASS
validation summary = PASS
hard-stop message = PASS
final/test summary = PASS
print flush = YES
```

启动摘要明确显示 `mobility source=trace`、formal training traces=256、training reset reserve=1、validation=5、test=10、state_dim=8、`norm_obs=true`、`norm_reward=true`、`total_timesteps=25600`、`n_steps=256`、expected PPO updates=100、episode horizon=100、device=cpu 和 validation interval=5120。每个 PPO update 只输出一行，原始系统指标明确标记为 `raw_reward=`；不可用的 Pearson 等诊断显示为 `NA`，不伪造数值。Validation 在 update `20/40/60/80/100` 显示 Reference/PPO 原始 reward、Delta、当前 best 和 `NEW BEST CHECKPOINT`。

Stage15 成功后自动从已保存的 `training_summary.json`/`test_summary.json` 后处理绘图，不重新运行 environment、PPO、Reference 或 SUMO；绘图异常只标记 `training=PASS, plotting=FAIL`，不覆盖训练产物。核心图使用原始 system reward，执行率分母严格为最终 test 的 total tasks，并同时比较 Reference/PPO，包含 Failed 类别且各组和为约 100%。

```text
reward plot source = raw/original system reward
normalized reward used as paper metric = NO
execution-rate denominator = total tasks
Reference/PPO execution comparison = enabled
plot smoke PNG = PASS
plot smoke PDF = PASS
figure output = stage15_reward_convergence.{png,pdf}; stage15_validation_reward.{png,pdf}; stage15_execution_rate.{png,pdf}
formatter smoke = PASS
headless matplotlib = PASS
temporary smoke artifacts deleted = YES
```

本轮未启动正式 Stage15，未调用 `model.learn`，参数漂移为 0；Stage15E 的 272 个 Trace 容量和 `20261067` reset reserve 保持不变。

```text
formal Stage15 training NOT STARTED
model.learn calls = 0
parameter drift = 0
Stage15E = PASS
```

## Stage16A：初始缓存与 Reference/PPO 公平性审计（2026-08-12）

本阶段只审计 reset 初始缓存的真实生成链、公平性和 Reference 的缓存更新触发行为；不修改 PPO、reward、缓存/UAV/Matching/KKT、请求/轨迹预测或物理参数，不重新训练。

### 初始缓存真实调用链

```text
Stage15 evaluation builder
-> train_ppo._make_evaluation_vector_environment
-> ppo_env.AdaptiveCachePPOEnv.reset
-> environment.MECEnvironment.reset
-> environment.MECEnvironment._random_feasible_cache
```

Reference 路径从 `train_ppo._run_reference_episode` 进入同一个 `MECEnvironment.reset`。实际 edge 初始缓存由 `environment.py:746` 的 `_random_feasible_cache` 生成，并在 `environment.py:1819` 调用；task vehicle local cache 在 `environment.py:1834` 调用同一函数。输入只有服务大小、各节点容量和 `np.random.default_rng(reset_seed + 1)` 的服务排列；`seed` 只控制随机可行初始化，不代表按该 seed 的未来请求或轨迹进行预适配。

该生成发生在 `TraceMobilityProvider` 创建和 `reset` 之前，未读取当前请求、历史请求、episode-wide popularity、预测流行度或未来 mobility。Trace provider 虽然在 reset 时把 NPZ 载入私有 archive，但初始缓存生成器没有消费未来行；`states_at` 只按请求的当前 slot 返回状态，`future_access_count=0`。reset 后用于 PPO observation 的当前状态道路预测/空历史流行度先验不会改写实际初始缓存。

### Validation 公平性审计

使用 Stage15 best checkpoint `outputs/ppo_training/20260812_154717/best/model.zip` 及对应 `best/vecnormalize.pkl`，validation 仅使用 `20271811 ... 20271815`。Reference 每个 seed 运行 100 slots；PPO 只 reset 并短 replay 5 slots；本阶段 `model.learn calls = 0`。

```text
initial cache identical = YES
initial UAV position/height identical = YES
task vehicles identical = YES
service vehicles identical = YES
vehicle positions identical = YES
request sequence identical = YES
role assignment identical = YES
physical config identical = YES
future request leakage = NO
future mobility leakage = NO
trigger anomalies = NONE
```

### Reference 缓存更新行为

```text
candidate_changed_slots by seed = 89, 99, 99, 87, 88
candidate_changed_slots mean = 92.4
positive_gain_slots by seed = 1, 1, 0, 1, 1
positive_gain_slots mean = 0.8
actual_cache_changes by seed = 1, 1, 0, 1, 1
actual_cache_changes mean = 0.8
reason class = B
```

结论为 B：候选缓存经常变化，但 `G = U_candidate - U_keep - cache_switch_cost` 在绝大多数时隙不为正，因此现有 Reference trigger 按设计拒绝切换；未发现 trigger 与实际缓存变化不一致。该结果不是 A（候选几乎不变化），也没有证据支持把它解释成 D 异常；初始缓存质量只作为 episode 结束后的离线诊断，未反向输入 reset。

正式审计文件：

```text
outputs/stage16a_initial_cache_audit/initial_cache_audit.json
outputs/stage16a_initial_cache_audit/initial_cache_by_seed.csv
```

```text
Stage16A = PASS
initial cache causal = PASS
initial cache/fairness explains weak PPO result = NO
next step = reward-component audit
model.learn calls = 0
formal Stage15 restarted = NO
```

## 统一训练结果主图风格（2026-08-12）

只修改 `plot_results.py` 和 `train_ppo.py` 的绘图/结果汇总输出，不涉及算法、
PPO、reward、cache、UAV、Matching、KKT、预测器或训练参数；本轮未启动训练。

新增/统一正式主图：

1. `reward_convergence.png/.pdf`：`Average Reward per Slot`，显示
   `Raw Average Reward` 与 `MA10`；累计 reward 按 episode steps 转换，不使用
   update mean。未完成的 terminal 后 partial episode 不作为 convergence 点。
2. `final_execution_rate.png/.pdf`：`Final Execution Rate`，固定五类
   `Local / Service Vehicle / RSU / UAV / Cloud`，分母为五类 completed task
   总数，柱顶标注百分比，右上角显示 `Completed` 和 `Failed`。completed=0 时
   五类比例均安全显示为 0.0%。

learning-check 结束时自动优先生成上述两张主图，原有 `execution_rates.png`、
`trigger_actual_rates.png` 等诊断图保留；Stage15/legacy 已保存结果可通过
`plot_results.py --result-dir <result_dir>` 后处理。控制台首先打印两张主图的
PNG/PDF路径。本轮未启动训练。

## 2026-08-12 第四章 P0 求解主链重构

任务：将第四章正式工程求解器从旧全局 `Discrete(2)` KEEP/UPDATE 重构为
`MultiBinary adaptive trigger + deterministic configuration optimization`。

修改原因：旧实现用固定 8 维 `J_keep/J_update` 状态和全局二元动作控制 A/B/C/D
候选，并在正式缓存路径使用 addition + 最多一次 1-swap；同时 `solve_slot()`
丢弃 UAV trigger。该结构没有直接表达“逐缓存节点/逐 UAV 是否允许配置”的 P0
在线近似决策。

```text
第三章 P0 changed = NO
C1-C16 changed = NO
Matching changed = NO
KKT changed = NO
reward changed = NO
```

### 原求解器与当前求解器

`paper_spec.md` 已把服务评分 `y_{m,f}`、连续位移 `e_u`、条件因子化混合动作
明确标记为 **11.pdf original solver design**；当前正式工程方案标记为
**current implementation solver**：

```text
causal prediction
-> MultiBinary(M+U) trigger
-> finite-candidate UAV positioning
-> one-round sequential cooperative full 0-1 knapsack
-> transfer-exchange Matching
-> KKT
-> G(t)
-> standard SB3 PPO
```

这是第四章分层近似在线求解器替换，不声称获得 P0 全局最优。

### 成熟实现复用

本机 conda 环境确认：`stable-baselines3==2.9.0`、`gymnasium==1.3.0`。
SB3 `PPO.__init__` 官方实现原生声明支持 `spaces.MultiBinary`，实际策略分布为
`BernoulliDistribution/Bernoulli`。项目直接复用：

```text
stable_baselines3.PPO + MlpPolicy
gymnasium.spaces.MultiBinary
gymnasium.spaces.Box
stable_baselines3 VecNormalize / DummyVecEnv / VecMonitor
```

项目 adapter 仅构造因果状态、拆分 trigger、调用确定性配置器并返回原始系统
reward；未自定义 PPO 分布、loss、GAE、clip 或 entropy。

### 新 observation/action

```text
observation groups = C(t-1), q_U(t-1), qhat_V(t), rhohat(t), t/T
observation dimension (current config) = 99
observation dtype = float32
observation range before VecNormalize = [0,1]
J_keep/J_update in Proposed observation = NO

action space = MultiBinary(8)
cache trigger bits = 6
UAV position trigger bits = 2
fixed order = SV cache, RSU cache, UAV cache, UAV position
```

`action_layout` 在 reset/info 中显式输出 `cache_node_ids`、`cache_node_types`、
`uav_ids` 和各 bit 索引。服务车辆替换继续按 vehicle id 重映射；新 SV 行置零，
不继承旧 SV cache，动作/缓存行/节点类型仍一一对应。

### Cache/UAV 配置

被触发 Cache 节点不再执行 addition/1-swap；先以其它节点当前配置计算逐服务
边际预测价值，再复用 `_integerize_cache_weights` 与 `_solve_knapsack_dp` 求一次
完整严格 0-1 knapsack。节点按 SV→RSU→UAV、同类型稳定行号只协同一轮；未
触发行逐元素保持。PPO trigger 与 actual update 分离，成本只按 `C_old->C_final`。

UAV 对未触发节点严格保持；触发节点只比较当前位置、预测需求热点和地面服务
缺口热点，复用 C6-C8 速度/区域/安全约束，并按预测服务效用减原 UAV movement
cost 选择。每个 slot 固定 UAV 先于 Cache，不执行多轮 alternating。

### 修改与删除

修改：`paper_spec.md`、`config.yaml`、`environment.py`、`solver.py`、`ppo_env.py`、
`train_ppo.py`、`generate_mobility_traces.py`、`jilv.md`。

删除正式生产逻辑：固定 8 维状态、全局 `Discrete(2)` KEEP/UPDATE、A/B/C/D、
forced UPDATE、`ppo_J_keep/ppo_J_update` preview、PPO service scores、PPO continuous
displacement、addition + `max_local_swaps=1`、Stage15 固定状态/动作训练诊断主链。
Stage15 历史结果目录和 272 条 mobility trace 均未删除或重新生成。

### 轻量验收

```text
py_compile = PASS
config parse = PASS
SB3 check_env = PASS
SB3 MultiBinary predict = PASS
VecNormalize interface = PASS
10-slot deterministic smoke = PASS

all-zero cache exact = PASS
all-zero UAV exact = PASS
all-zero cache/UAV cost = 0 / 0
single cache trigger other rows exact = PASS
single UAV trigger cache exact = PASS
single UAV trigger other UAVs exact = PASS
cache trigger no-op actual update/cost = 0 / 0
UAV trigger no-op actual update/cost = 0 / 0

10-slot Q = 51.40921997459373
10-slot cache cost = 0.004
10-slot UAV cost = 0.0
10-slot reward = 51.405219974593734
reward formula residual max = 0.0

future request leakage = 0
future mobility leakage = 0
capacity violations = 0
UAV speed/region/safety violations = 0/0/0
compute violations = 0
bandwidth violations = 0
KKT failures = 0
NaN/Inf = 0

model.learn calls = 0
formal training restarted = NO
current formal training allowed = NO (本轮只完成重构与轻量验收)
```

## 2026-08-12 Stage16临时审计代码清理

Stage16A/B 仅用于问题定位；本次审计临时代码、审计输出和运行时缓存已清理，未继续扩展 Stage16 审计框架。

```text
deleted Stage16A outputs = outputs/stage16a_initial_cache_audit/
deleted Stage16B outputs = outputs/stage16b_reward_audit/
deleted audit-only scripts = stage16a_initial_cache_audit.py, stage16b_reward_audit.py
removed audit-only production code = NONE FOUND
removed unused/debug files = project __pycache__/ and *.pyc generated by validation

Stage15 formal run preserved = YES
Stage15 best checkpoint preserved = YES
Stage15 figures preserved = YES
272 mobility traces preserved = YES
seed_20261067 reserve preserved = YES

MaskablePPO present in source = NO
G-added state present in source = NO
counterfactual framework present in source = NO
parameter drift = 0
model.learn calls = 0
formal training restarted = NO

py_compile = PASS
config parse = PASS
Stage15 module import = PASS
current PPO config equals Stage15 snapshot = YES
```

Stage15 正式系统、训练结果、绘图、Offline Trace、车辆角色、轨迹预测、需求预测、协同缓存、Reference、UAV、Matching、KKT、Environment、reward 和 PPO 均保留。`jilv.md` 中的 Stage16A/B 审计结论只作为历史记录保留，不代表审计输出文件仍存在。

## Stage16B：Reward组成与缓存收益/切换成本尺度审计（2026-08-12）

本阶段只读取 Stage15 best checkpoint，在 5 个固定 validation seed 上执行 Reference/PPO 各 100 个 slot 的逐 slot 对账；不修改 PPO、state、action、reward、normalization、缓存、UAV、Matching、KKT 或任何训练参数，不调用 `model.learn`，不重启 Stage15 正式训练。

### 公式与账目

```text
raw reward = Q - C_cache_actual - C_uav_actual
G = U_candidate - U_keep - C_switch(C_old, C_candidate)
J_keep = max(keep predicted utility, UAV-only predicted utility - UAV move cost)
J_update = max(candidate utility - candidate cache cost,
               joint candidate utility - candidate cache cost - UAV move cost)
```

`Q` 是匹配、KKT 和任务执行后的实际归一化时延/能耗任务效用，不含缓存更新成本；执行 reward 只从 `SolverResult.final_cache` 与最终 UAV 位置取得实际成本并各扣除一次。`G` 是 Reference 的独立缓存触发判据，`J_update` 是 PPO preview 分支的候选价值，二者不是同一个表达式，因此 `G` 中出现切换成本不构成 raw reward 的重复扣除。

```text
cache cost in J_update = YES
cache cost subtracted again in G = YES (independent Reference trigger formula)
cache cost already inside Q = NO
cache cost subtracted again in reward = NO
double counting = NO
```

逐 slot 机器核对结果：

```text
raw reward formula residual max = 0.0
G formula residual max = 0.0
candidate cost recompute residual max = 0.0
same raw reward implementation = YES
```

### 收益、成本与尺度

```text
candidate_changed mean = 92.4
positive_gain mean = 0.8
candidate rows = 462
utility_gain mean/median = 0.009196377591590397 / 0.008689902690481532
cache_cost mean/median = 0.02827075324675325 / 0.027424
G mean/median = -0.019074375655162852 / -0.018021619198189225
fraction gain > cost = 0.008658008658008658
median cost/gain ratio = 3.2191318596031016
```

预测缓存效用是使用与 `task_utility` 相同归一化体系得到的无量纲、相对云端的边缘服务经济效用；缓存成本是配置中的经济 reward 成本（固定更新成本加按 bit 加载能耗的加权成本），不是原始 Joule 或原始任务数。Reference 全部 500 slot 的 raw `Q` 均值/中位数为 `4.907190819866335 / 4.868006241743148`，raw reward 均值/中位数为 `4.907058917866334 / 4.867521190762177`；因此当前单位与尺度一致，`unit/scale consistency = PASS`。

### Reference/PPO 分解（每个 100-slot episode 的 5 个 seed 均值）

```text
Reference:
Q = 490.7190819866335
cache cost = 0.011003200000000001
UAV cost = 0.0021870000000000006
raw reward = 490.7058917866335

PPO:
Q = 491.222705482079
cache cost = 0.5859504
UAV cost = 0.0019926
raw reward = 490.634762482079
```

Stage16A 的 `92.4` 个候选变化与 `0.8` 个接受变化的根因确认是经济权衡：候选生成器经常产生变化，但只有 `U_candidate - U_keep - C_switch > 0` 才提交，绝大多数候选收益不足以覆盖当前固定加加载成本；未发现缓存成本重复计入或实现错误。

正式审计文件：

```text
outputs/stage16b_reward_audit/reward_component_audit.json
outputs/stage16b_reward_audit/gain_cost_distribution.csv
```

```text
Stage16B = PASS
scale/accounting conclusion = A correct accounting and scale
model.learn calls = 0
formal Stage15 restarted = NO
```

## MultiBinary PPO 5120-step learning-effectiveness check preparation（2026-08-12）

本轮只为当前第四章 MultiBinary 自适应 Cache/UAV 框架增加短训练有效性验证入口，
不改动第三章 P0、C1-C16、reward、cache solver、UAV solver、Matching、KKT、
预测算法、PPO action/state 或现有 PPO 超参数；Codex 本轮不调用 `model.learn`，
也不启动 Stage15 正式训练。

### 代码修改

`train_ppo.py` 新增 `--learning-check` 独立入口，固定 `total_timesteps=5120`、
`n_steps=256`，因此预期 20 个 PPO updates。其余 PPO 参数继续从 `config.yaml`
读取，使用现有 SB3 PPO、VecNormalize 和单环境链路。训练输出写入独立的
`outputs/ppo_learning_check/<run_id>/`，不会覆盖 Stage15 输出。

入口记录：

```text
config_snapshot.json
training_updates.csv
training_episodes.csv
action_trigger_rates.csv
validation_results.csv
learning_summary.json
model.zip
vecnormalize.pkl
```

训练 episode 和每个 PPO update 同时记录原始 `G=Q-C_cache-C_uav`、`Q`、缓存/UAV
成本、时延、车辆能耗、Local/SV/RSU/UAV/Cloud/Failed 计数与比率；动作统计从
`environment.action_layout` 读取，包含每个 MultiBinary bit 的 requested trigger
count/rate、每个 cache node 的 effective update/no-op/services changed，以及每个
UAV 的 requested trigger、actual movement/no-op。这里的 `cache_service_availability`
定义为当前到达请求中由 SV/RSU/UAV 边缘缓存执行的比例，不加入 reward。

每个 update 还记录 `approx_kl`、`clip_fraction`、`entropy_loss`、
`policy_gradient_loss`、`value_loss`、`explained_variance`，并输出 Early(1--10)、
Late(11--20) 与 `Late-Early` 的 `G/Q/cache cost/UAV cost/delay/energy`、执行率和
动作率。验证阶段使用当前模型的 deterministic policy，只使用
`validation_seeds=[20271811,...,20271815]`，每个 seed 输出逐 seed 指标并给出
mean/std；正式 test seeds 不进入验证路径。

训练和验证累计检查 future request/trajectory/mobility leakage、cache capacity、
compute、bandwidth、UAV speed/region/safety、invalid node/service、NaN/Inf 与 KKT
failure；任一非零时 `learning_check_status=FAIL`，不会由脚本自动修改算法或成本。

```text
training seed mode = sequential
validation seed mode = fixed
test seeds accessed = NO
model.learn calls in preparation = 0
formal Stage15 restarted = NO
```

## 正式冻结论文基础场景为 8 任务车辆 + 2 服务车辆 + 10 类服务（2026-08-12）

本轮只完成基础参数定稿和最小一致性检查，没有修改 P0、C1-C16、PPO
state/action、reward、缓存、UAV、Matching、KKT、预测器或服务车辆替换逻辑，
也没有调用 `model.learn`。`config.yaml` 的 `vehicle_roles.num_task_vehicles`
由 10 改为 8，`num_service_vehicles=2`、`services.count=10` 及其余冻结参数保持不变。

### 冻结场景与动态维度

```text
research vehicles = 10
task vehicles = 8
service vehicles = 2
RSUs = 2
UAVs = 2
services = 10
cache nodes = 6
action = MultiBinary(8)       # 6 cache trigger bits + 2 UAV position trigger bits
observation = 95              # 60 + 4 + 20 + 10 + 1，代码动态计算
edge cache shape = (6, 10)
local task cache shape = (8, 10)
predicted vehicle position shape = (10, 2)
predicted popularity shape = (10,)
```

### 角色、Trace 与 workload 兼容性

正式 SUMO route 集合共包含 70 个车辆 ID；Trace 文件保存完整车辆真值，
不是旧配置下固定的 `10 task + 2 service` 角色子集。当前 `VehicleRoleManager`
仍按当前时隙状态动态形成 Task/SV/Background 三个互斥集合，全部 272 个 Trace
均可提供至少 10 辆有效车辆，逐 slot 角色数量检查通过：

```text
Task/SV overlap = 0
Task/Background overlap = 0
SV/Background overlap = 0
role count = 8 task + 2 service = PASS
existing trace compatible = YES
trace regeneration required = NO
trace files checked = 272
trace schema/shape errors = 0
```

工作负载由当前配置动态生成，四个任务数组均为 `(100, 8)`，到达任务服务 ID
范围为 `[0, 9]`；本次 100-slot 轻量生成得到 405 个请求，符合
`8 × 100 × 0.5` 的统计期望，不固定请求总数。单 slot deterministic smoke
通过：`future request leakage=0`、`future trajectory leakage=0`、缓存容量违规、
UAV违规和 NaN/Inf 均为 0。未重新生成任何 mobility trace。

```text
Base configuration freeze = PASS
frozen parameter check = PASS
algorithm changed = NO
reward changed = NO
PPO parameters changed = NO
model.learn calls = 0
training restarted = NO
```

## Proposed 分层联合求解正式实现（2026-08-12）

本轮正式确认并补齐“预测辅助的自适应 Cache/UAV 配置触发 + 确定性配置优化
+ Matching/KKT”主链。第三章 P0、`G(t)`、C1-C16、通信/计算模型、缓存成本、
UAV 移动成本、reward 和 PPO 超参数均未修改；本轮没有调用 `model.learn`。

### 理论与主链

`paper_spec.md` 新增第四章 Proposed 的 A-H 结构：问题分解、因果移动/需求预测、
PPO 自适应 trigger、需求感知 UAV 候选、顺序协同缓存重配置、transfer-exchange
Matching、KKT 资源分配和复杂度。明确当前方法是 state-driven adaptive
configuration，不使用参考论文的 fixed-period cache/UAV 更新，也不声称 P0 全局最优。

正式每 slot 顺序保持：

```text
history/prediction -> PPO MultiBinary trigger -> UAV candidates
-> SV/RSU/UAV sequential 0-1 knapsack -> C(t), q_U(t)
-> current task -> feasible nodes -> Matching -> KKT
-> execution -> Q(t) -> G(t)=Q(t)-C_cache-C_UAV
```

现有 `solver.py` 的确定性实现继续复用：UAV 候选只包含当前位置、预测需求热点
和地面覆盖缺口热点；缓存使用严格 0-1 knapsack DP，按 `SV -> RSU -> UAV` 只做
一轮协同；Matching、KKT 和 Cloud fallback 未替换。

### 输出契约补齐

- `environment.step()` 显式保存 `requested_cache_triggers`、`actual_cache_update`、
  `cache_noop_trigger`，以及对应的 UAV requested/actual/noop 向量。
- episode/update 汇总增加 `non_cloud_rate`；原始 `G`、`Q`、实际缓存成本、实际
  UAV 成本、执行计数、节点级 requested/actual/noop/services changed 和硬约束诊断
  继续由 `train_ppo.py` 记录。
- Proposed 结果图进入独立目录：`figures/` 保存两张主图，
  `figures/diagnostics/` 保存附加诊断图；结果同时写入兼容的
  `learning_summary.json` 和 `final_summary.json`。

### 基础场景与验证

```text
Task = 8
Service Vehicle = 2
Research vehicles = 10
RSU = 2
UAV = 2
F = 10
cache nodes = 6
action = MultiBinary(8)
observation = 95
```

```text
SB3 check_env = PASS
model construction/predict = PASS
all-zero exact cache/UAV keep = PASS
all-zero cache cost = 0
all-zero UAV cost = 0
one-cache-trigger isolation = PASS
one-UAV-trigger isolation = PASS
actual update/movement separated from requested trigger = PASS
10-slot formula residual max = 0.0
future request leakage = 0
future trajectory leakage = 0
capacity violations = 0
UAV speed/region/safety violations = 0
compute violations = 0
bandwidth violations = 0
KKT failures = 0
NaN/Inf = 0
```

人工合法 episode 绘图 smoke 已生成并检查：

```text
reward_convergence.png/.pdf = PASS
final_execution_rate.png/.pdf = PASS
MA10 with fewer than 10 episodes = PASS
fixed five-class percentage denominator = completed tasks
temporary test outputs removed = YES
model.learn calls = 0
formal training restarted = NO
```

## 2026-08-12：正式训练前实现一致性修复

本轮只修复 `adaptive_cache_mec` 正式训练前的实现一致性问题，未启动正式或长步数 PPO，
`model.learn` 调用次数为 0。第三章 P0、C1-C16、reward、Matching、KKT、0-1 knapsack
缓存求解结构、有限候选 UAV 求解结构及物理场景参数均保持不变，parameter drift = 0。

### 基础场景与角色

```text
Task = 8
SV = 2
RSU = 2
UAV = 2
F = 10
cache nodes = 6
action space = MultiBinary(8)
observation dimension = 95
role mode = fixed_episode
task IDs = tv_000 ... tv_007
SV IDs = sv_000, sv_001
silent replacement/remap = NO
```

固定角色 trace 预检通过：每个 episode 内研究任务车和服务车 ID、角色及 action layout
不漂移；背景车辆不进入 workload。服务目录固定为 10 类，episode seed 只改变 workload
随机流，不改变服务目录和角色映射。

### Seed、trace 与因果性

```text
formal training episodes = 256
training reset reserve = 1
training traces = 20260811 ... 20261067
validation traces = 20271811 ... 20271815
test traces = 20281811 ... 20281820
total required traces = 272
missing seeds = []
```

25600 environment steps、100-slot episode 的容量按
`floor(effective_env_steps / horizon) + 1` 计算，最后 1 条仅用于 DummyVecEnv
terminal 后的 automatic reset，不计入正式完成 episode。训练、验证、测试 seed 语义分别为
sequential、fixed、fixed；workload 的 resolved seeds 已写入诊断。

请求、轨迹和 mobility 的未来信息诊断均来自实际运行时 origin/observation slot，
不是硬编码。正常 smoke 结果为：

```text
future_request_leakage = 0
future_trajectory_leakage = 0
future_mobility_leakage = 0
```

### 通信、指标与正式入口

SV/RSU/UAV 正式价值统一调用现有物理通信速率、任务执行和任务效用模型；300 m 半径
仅保留为覆盖展示/辅助字段，不替代正式物理价值。实际执行仍使用当前时隙真实位置、
真实请求、Matching 和 C13-C16 KKT 资源分配。

指标已分离为：edge cache availability（到达任务可由边缘缓存物理服务的比例）、
edge execution rate（已完成任务中的 SV/RSU/UAV 执行比例）和 non-cloud rate
（已完成任务中的非 Cloud 执行比例），不再互相替代。

正式入口只在用户显式执行 `--train` 时调用一次 `model.learn`。callback 仅按
`eval_freq = 5120` 做 validation，按 validation 平均 raw `G` 选择 best checkpoint，
并同时保存匹配的 PPO 与 VecNormalize；test 只在 best 选择之后运行。Figure 1 使用
training raw `G` per slot，Figure 2 使用 best-model final test，四种 PNG/PDF 绘图
smoke 均通过。

### 本轮最小验收

```text
py_compile = PASS
seed reproducibility/difference = PASS
fixed-role and action-layout test = PASS
trace preflight = PASS
SB3 check_env/startup smoke = PASS
causality diagnostics smoke = PASS
communication physical-value consistency = PASS
metric definition test = PASS
formal callback best-selection mock = PASS
plot smoke = PASS
model.learn calls = 0
formal training started = NO
base HEAD = 3dcf672
```

commit SHA and `origin/main` push result are recorded in the final Git handoff after the
single consistency-fix commit is created.
