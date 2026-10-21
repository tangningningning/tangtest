# 双预测时空状态特征设计

日期：2026-10-18  
范围：只读核对当前预测、任务、节点与正式配置合同；本文是 MAPPO 对接设计，不是 Actor/Critic/Trainer/Runner 实现，也不改预测器、Zipf 或任务数据。

## 结论摘要

- 当前正式预测参考 `H=5`，每个 origin 的预测时域是 `h=0..4`，目标 slot 为 `origin_slot+h`。两类预测都以 `origin_slot-1` 为观测截止点。
- 车辆预测提供道路约束的名义坐标与由独立 calibration residual bank 采样的时空位置场景。在线车辆预测表的真实坐标和逐行误差字段为空；没有可直接输入策略的在线位置概率。
- 流行度 GRU 输出全局 `rho_hat[origin,horizon,service]`，没有 region 轴。真实任务生成使用同一条动态 Zipf 分布，未按地理位置改变服务概率。因此当前不需要区域流行度概率，也不能从现有合同推断它。
- 适合的中间表示是可变节点集合上的节点—服务—时域需求/工作量张量。它是预测需求代理；实际净收益仍由真实任务经 Matching/KKT 执行后计算。
- 以两处 RSU 的最近站点划分作静态诊断，两区有足够的请求总量；SUMO 子图不是两个连通区域，而且两区请求数不均衡。可用于分析分箱，不能据此引入区域偏好的数据生成假设。

## 1. 当前预测与任务合同

### 1.1 车辆轨迹预测

`src/scenario/vehicle_prediction.py` 的 CSV 行键为 `(origin_slot, vehicle_id, horizon)`，并写出 `target_slot/time`、`s_pred`、`x_pred/y_pred`。当前 `H_p=5`、`dt=1s`；`target_slot=origin_slot+horizon`，`horizon` 从 0 到 4。预测由截止 `origin_slot-1` 的 Kalman posterior 向前推进，名义点投影到车辆 planned route 的几何上。

`vehicle_predictions.csv` 的字段还包括 `s_true/x_true/y_true`、`longitudinal_error`、`lateral_error`、`e_parallel`、`e_perp`。当前 formal predictor 把这些 truth/误差列留空，避免把未来真值放进在线预测 artifact；因此该文件没有逐行的真实预测误差数值，也没有在线误差区间。Calibration residual blocks 的两个分量按切向/法向投影，单位为米；场景生成代码按 `p_xy = p_nominal + r_parallel*tangent + r_perp*normal` 应用 residual。CSV 中 `longitudinal_error/lateral_error` 与 `e_parallel/e_perp` 的逐行 signed-error 计算式当前没有在 formal 输出中实现/保存。Calibration bank 是独立轨迹产生的 `[origin,H,2]` residual blocks；其 metadata 报告 calibration position RMSE 随 horizon 从 9.16m（h=0）增至 21.69m（h=4）。这属于独立校准诊断，不是每个 formal forecast 的已知误差或置信区间。

`prediction_scenarios.npz` 当前的相关字段为：

- `positions_xy[origin,S,H,V,2]`；正式 `S=20`，`H=5`，车辆维包含 research vehicles。
- `vehicle_ids`、`vehicle_roles`、`task_vehicle_ids`、`service_vehicle_ids`，用于把任务车、服务车及节点身份对齐。
- `origin_slots/times`、`scenario_ids`、`horizons` 和 metadata。

位置场景来自 calibration residual blocks 的整段抽样并经道路合法性检查。场景编号是有限样本；合同未输出经校准的 `P(vehicle in region)` 或显式场景权重。

### 1.2 服务流行度预测

`src/scenario/popularity_prediction.py` 的 GRU 输入是过去 `L_rho=20` 个 slot 的全局服务请求份额观测（无请求时 carry-forward），单层 GRU 一次输出 5 个 horizon 的服务分布。当前输出 `rho_hat[origin,horizon,service]`，服务维 `F=10`，每个 horizon 沿服务维归一化。

`popularity_predictions.csv` 有 `rho_pred`、`rho_star`、`abs_error` 等字段。`rho_star` 和 `abs_error` 是离线真值标签/诊断字段，策略只能读 `rho_pred`（或相应 NPZ 的 `rho_hat`），不能读未来标签。

预测和真实过程都是全局服务分布，没有 node、坐标或 region 维。真实服务 rank 按全局动态 Zipf 过程演化（正式 `zeta=0.8`，每 20 slots 局部调整一次）；现有 task generator 不给不同区域配置不同 Zipf 参数。有限样本中按地点统计的服务占比波动不等于存在真实区域差异。

### 1.3 预测任务到达量

正式任务池每 slot 对每个 TV 独立抽样 Bernoulli 到达，`lambda=0.7`。有任务时，真实 `tasks.csv` 的服务 ID 按该 slot 的全局真实 `rho_star` 抽样；输入量和计算 cycles 带有合同内的均值为 1 的乘性扰动（输入 ±20%，负载 ±10%）。正式服务目录有 10 个服务。

`prediction_scenarios.npz` 中 `arrival/service_id/input_size/cpu_cycles[origin,S,H,T]` 是预测场景：到达仍按 `lambda` 抽样，服务按对应 horizon 的 `rho_hat` 抽样，工作量从服务目录均值和扰动生成。它们不是 `tasks.csv` 里的实际未来任务。策略可使用从这些预测场景聚合出的统计量；不得把实际任务行或实际未来任务的服务/工作量喂给 Actor。

### 1.4 节点位置、覆盖和 Cache 状态

正式主线是 5 SV、2 RSU、2 UAV，Cache 节点集合共 9 个；服务车辆 ID 与 `vehicle_ids` 对齐。SV 的位置随对应车辆预测移动；RSU 坐标由 `data/generated/node_positions.json` 提供；UAV 有当前策略位置，之后的位置由 Micro 动作改变。每个节点的 Cache 状态为服务维二进制向量，节点 Cache 容量按角色配置；服务大小来自 service catalog。

现有 Cache proposal 特征使用按角色的距离尺度（SV 100m、RSU 250m、UAV 300m）生成线性 taper 权重 `max(0,1-d/r)`。代码注释明确它只是 proposal/serviceability 信号，不是实际 Matching/KKT 的硬覆盖规则。实际候选可行性和效用仍由通信链路、Cache 命中、Transfer–Exchange Matching 与资源分配流程确定。现有 `node_specific_predicted_demand` 已把预测服务需求按车辆—节点距离加权，`node_local_predicted_service_workload` 可汇总预测请求、bit 和 cycles；这些现有量可以作为设计依据，但它们不是实际回报。

### 1.5 Cloud 与 Edge 资源竞争

当前每个 slot 独立执行任务，不跨 slot 保存任务队列。Edge 节点的 CPU 和带宽池由当 slot 的任务竞争，资源份额由 Matching/KKT 决定。正式能力参考为 SV 1GHz、RSU 5GHz、UAV 5GHz；Edge 系统带宽 10MHz。Cloud 使用总量 40GHz 的有限共享 CPU；上行是每任务等效 20Mbps，当前不按任务数共享带宽，Cloud 上传时延系数为 13。

因此，单纯的“周围请求多”不能等同“节点服务价值高”。需求特征应带上候选缓存可服务的服务集合、上传/计算工作量相对资源能力，以及潜在替代 Edge/Cloud 选项。建议用同一通信模型在预测距离下算出的 full-bandwidth link rate 作参照，构造 `edge_upload_airtime_ratio[m,h] = sum_{i,f}(predicted_bits[i,f,h] * serviceable_cache_weight / (R_full[m,i,h]*dt))`；再构造 `edge_cpu_ratio[m,h] = candidate_cached_cycles[m,h]/(cpu_m*dt)`，并提供可服务 Edge 候选数/替代节点数。Cloud 侧单独提供预测 fallback CPU cycles 相对 40GHz 池的负载代理、fallback 请求质量，以及每任务的 20Mbps/13 倍上传时延参考；Cloud 上传带宽当前不按用户共享。Cloud 的 CPU 竞争和 Edge 的 per-node CPU/带宽竞争会改变后续服务时延与任务效用；这些 offered-load 比值不等于实际 KKT 资源分配，也不等于跨 slot backlog。

## 2. 节点—服务—时空需求特征接口

以下是与网络结构无关的输入/中间表示建议，不规定 MAPPO 的 tensor 名称、Actor 头、Critic 输入或动作编码。

### 2.1 因果边界和允许字段

对当前决策 origin `t`，只读 `observation_cutoff_slot=t-1` 及其之前的观测、由这些观测生成的 `x_pred/y_pred` 和 `rho_hat`、当前 Cache/UAV 状态、静态 service catalog、节点资源和合法动作 mask。`t=0` 使用 `formal_slot=-L_rho..-1` 的 prehistory 和 warm-up 车辆状态。

必须排除 Actor 输入：`tasks.csv` 当前/未来实际行；`popularity_predictions.csv.rho_star/abs_error`；`vehicle_predictions.csv` 的 truth/误差列；future matching/KKT 输出；由真实未来任务算出的节点收益。Episode 末尾设置 horizon mask：只有 `t+h < 300` 的 forecast 可用；不把越界预测当成 episode 内需求。

### 2.2 核心张量

按动态集合表达，而非将车辆/节点 ID 写死到固定位置：

- 节点描述 `node[m]`：`node_id`、`role`、当前位置/高度、Cache 容量、CPU、带宽、角色距离尺度、当前缓存占用。
- 服务描述 `service[f]`：`service_id`、缓存大小 MB、平均输入 bits、cycles/bit。
- 预测输入：`position[i,h]` 或当前有限场景 `position[s,h,i]`、`rho_hat[h,f]`、`lambda`、TV/SV 角色索引、`horizon_mask[h]`。
- 当前决策状态：`cache[m,f]`、UAV 当前坐标和候选坐标、候选合法 mask；候选 cache state 与现有状态的增删服务差分。
- 主要节点服务预测输出：`demand[m,f,h]`、`input_bits[m,f,h]`、`cpu_cycles[m,f,h]`，及聚合的 `discounted_demand[m,f]`。

### 2.3 预测需求计算

令 `i` 遍历 task vehicles，`f` 遍历服务，`h=0..4`。核心确定性版本可使用因果名义位置；如果采用 `positions_xy` 的 20 条残差场景，则仅取有限场景的算术样本均值，不把它命名为位置概率：

```text
w[m,i,h] = role_distance_taper(distance(node_position[m,h], vehicle_position[i,h]))
demand[m,f,h] = lambda * rho_hat[h,f] * sum_i w[m,i,h]
input_bits[m,f,h] = demand[m,f,h] * mean_input_bits[f]
cpu_cycles[m,f,h] = input_bits[m,f,h] * cycles_per_bit[f]
discounted_demand[m,f] = sum_h beta^h * horizon_mask[h] * demand[m,f,h]
```

若位置场景被选为位置输入，可额外保留 `mean_s(w)` 和 `std_s(w)` 作为有限场景 summary；不创建 `P(region)`、不拟合新位置分布，也不把 `S=20` 的样本方差称作 calibration confidence。服务流行度始终用全局 `rho_hat[h,f]`。每项工作量使用固定目录均值；如果之后使用预测场景内的工作量样本均值，应只聚合预测 scenario 的样本，不读实际任务表。

`w` 是软 proposal 权重；真实服务可行性应另外使用环境既有的物理/Cache mask。最终使用时按固定尺度规范化，并同时提供 TV 数量/节点资源容量，避免车数增大后把“总请求更多”和“单位车需求更高”混为一项。尺度只由训练合同、service catalog 和系统能力确定，不从 Validation/Test 估计。

### 2.4 Macro Cache 与 Micro UAV 用途

**Macro Cache（节点×服务）**：对每个 Cache 节点的候选配置，输入当前/候选 `cache[m,f]`、`demand[m,f,0:5]`、预测输入 bits/cycles、服务大小、节点容量余量和资源能力。显式保留 horizon 维，让策略能区分近端需求与第 2–5 slot 的预测变化；可附加 `beta=0.95` 的折扣聚合量。对于 SV，使用该服务车自身的预测位置；RSU 用固定坐标。Macro 决策看不到未来 UAV 路径，UAV 的 cache serviceability 只能按当前 UAV 位置计算，或把“未知未来 UAV 位置”作为 mask/标记；不可隐含假定其移动轨迹。

**Micro UAV（UAV×候选位置）**：对每个 STAY/MOVE 候选位置 `p_u^cand`，在同一 `rho_hat` 和车辆位置预测下重算该候选位置对应的节点—服务覆盖需求。未来 5 slots 以候选位置固定作 proxy，并明确这是候选评分的 persistence 假设，不是 UAV 运动预测。输入当前/候选位置、到候选位移、合法性 mask、Cache 服务集合、节点服务需求/工作量、与其他 UAV 的间距，以及单独的移动成本 proxy。对于第二架及后续 UAV，依照实际动作顺序用已选/条件候选位置更新其他 UAV 坐标。

### 2.5 需求代理与真实净收益分离

`demand/input_bits/cpu_cycles` 只表示“预测位置和全局预测服务分布下，节点可能遇到的加权需求”。这些值没有执行 Cache/匹配/资源分配，不是任务效用，也不扣除配置成本。

候选级预估净收益如未来确需使用，应单独命名为 `predicted_candidate_surrogate`，并说明其需求场景与 Matching/KKT 近似。真实 slot 收益依现有合同在实际任务上计算：

```text
net_utility[t] = actual_task_utility[t] - actual_cache_cost[t] - actual_uav_movement_cost[t]
```

真实 Episode return 是 300 个 slot 的上述值之和。只有该执行结果才是正式净效用；预测 demand 不应冒充它。

### 2.6 不同节点/车辆数量的兼容

数据接口使用变长 `N_node`、`N_TV`、`N_SV`、`N_RSU`、`N_UAV` 和 `F` 维集合，保留 `H=5` 时间轴。对 batch 填充使用显式 mask；对节点/车辆重排保持 permutation-equivariant 的共享特征编码和 set/pooling 聚合。外部 node ID 仅用于回写动作和指标，不把随机 SUMO vehicle ID 当作连续语义特征。输出可按节点逐项列出，再由策略消费；不要求固定 9 个 Cache 节点或 2 架 UAV。

当前 legacy PPO 接口自身对 `MULTI_NODE_FACTORIZED` 有固定 5-SV/2-RSU/2-UAV 顺序检查，当前 checkpoint 的 tensor shape 也与固定 schema 绑定。因此“特征合同可变长”是 MAPPO 对接要求，不代表当前 PPO checkpoint 或网络已具备变长节点能力。

Stage1A 的实际 MAPPO 输入、动作和值函数接口现已能从只读参考 worktree 确认；见第 5 节。当前实现仍使用既有固定维度 observation 与候选特征，不使用本报告第 2.3–2.6 节提出的新增时空需求张量。下面的设计仍是后续可选接口方案，不代表 Stage1A 已实现。

## 3. 两区域拓扑与请求样本静态诊断

诊断只读 `osm.net.xml.gz`、现有 RSU 坐标和 11 个分散 Training seeds（20001、20030、20060、20090、20120、20150、20180、20210、20240、20270、20300）。按请求发生时 TV 的轨迹坐标，将每辆车分到最近 RSU 的 Voronoi 区；这是样本量检查，不是新的正式区域定义，也没有读 Validation/Test episode。

- SUMO 文件解析到 175 个带坐标 junction 和 127 条普通有向 edge。最近 RSU 分区的跨区 edge 为 10 条（7.9%）；RSU 坐标位于拓扑空间的两端附近，UAV 初始点也分别邻近两处 RSU，地理上可以定义两个分析 catchment。
- 该切分后的子图并非两块连通路网：RSU_0 区有 82 个 junction、49 个弱连通分量（最大分量 34）；RSU_1 区有 93 个 junction、61 个弱连通分量（最大分量 33）。因此宜作为连续地理环境上的两个统计分箱，不宜宣称是两个完全隔离的道路区域。
- 11 个 Training Episode 有 66,000 个 TV-slot 请求机会，观测到 46,123 个到达（69.9%，与 `lambda=0.7` 相符），轨迹位置均可匹配。RSU_0 最近区有 29,747 个到达（64.5%），RSU_1 最近区有 16,376 个（35.5%）；每个 Episode 的两区分别约 2,475–3,060 和 1,147–1,751 个请求，且两区在几乎全部 Episode slots 都有请求。11 个样本中每个服务在每区至少有 970 次请求。

该规模足以对两个地理分箱的汇总需求做描述性统计；较小区仍有每个 Episode 千级请求。它不支持“真实区域流行度不同”的结论：任务生成明确按全局同一 `rho_star` 为所有 TV 抽服务，当前观察到的局部差异只能先视为有限样本、车辆分布或边界分箱效应。保持本轮 Zipf 和真实任务生成不变；若将来需要真实的区域流行度，应另立数据合同、生成机制和训练/验证证据。

## 4. 代码与数据证据

- 预测输出/因果：`src/scenario/vehicle_prediction.py`、`src/scenario/scenario_generator.py`、`src/scenario/popularity_prediction.py`。
- 真实任务与 Zipf：`src/scenario/task_scenario.py`、`configs/proposed/system_config.toml`。
- Cache 节点及预测需求：`src/decision/cache_configuration.py`；正式状态与预测输入组装：`src/ppo_simple/interface.py`、`src/ppo_simple/rollout.py`。
- 节点位置：`data/generated/node_positions.json`；SUMO 图：`data/sumo/map/osm.net.xml.gz`。
- 只写本报告；未修改 Actor/Critic/Trainer/Runner、正式配置、Zipf、任务池或 MAPPO 工作区；未训练、回放完整 Episode 或搜索参数。

## 5. Stage1A 实际 MAPPO 接口与 Slot 顺序

本节依据只读参考 worktree `C:\Users\tangning\Desktop\code\adaptive_cache_mec-mappo-baseline-stage1` 的 `src/ppo_simple/mappo_baseline.py`、`policy.py`、`interface.py`、`rollout.py`、`primitive_control.py`、`scripts/training/run_mappo_baseline_stage1b.py` 与正式配置。Stage1A 固定要求 `MULTI_NODE_FACTORIZED`、9 个 Cache 节点（5 SV、2 RSU、2 UAV）和 2 个 UAV head；这不是变长节点实现。

### 5.1 Macro/Micro Actor 当前输入

Macro 与 Micro 共享 `build_step_observation` 构造的全局 observation。当前正式维数为 400：

| 顺序 | 字段 | 当前形状 | 编码 |
|---|---|---:|---|
| 1 | 决策前 Cache | 9×10 = 90 | 按固定 node/service 顺序展平的 0/1 状态 |
| 2 | 当前 UAV 位置 | 2×2 = 4 | 按飞行边界归一化的 x/y |
| 3 | 车辆预测位置 | 5×25×2 = 250 | 按 horizon、固定 vehicle 顺序排列的归一化 x/y |
| 4 | 全局服务流行度预测 | 5×10 = 50 | `rho_hat[h,f]` |
| 5 | 预测时域有效掩码 | 5 | 目标 slot 不越过 Episode 末端 |
| 6 | `time_to_go` | 1 | 归一化剩余训练时域 |

两 Actor 的状态向量均为上述当前状态；Critic 不接收候选特征。状态构造没有启用 Ledger workload 特征，也没有区域流行度概率或本报告建议的新 `demand[m,f,h]` 张量。

Actor 还接收按动作候选生成的特征和合法 mask：

- **Macro**：`candidate_features[9,K,200]` 和 `candidate_masks[9,K]`。每个节点共享 actor scorer；候选特征由 181 维 Cache 基础描述（目标 90 位、相对当前变化 90 位、该节点候选是否为 UPDATE 的 `1/9` 标记）和 19 维经济/需求特征组成。后 19 维为 `is_keep`、节点索引/角色、缓存成本/容量变化、复制多样性变化，以及添加/移除服务在 `h=0..4` 的需求摘要。每个节点的 `K` 可变，上限为 KEEP 加 64 个配置候选。
- **Micro**：`candidate_features[2,C,41]` 和合法 mask。41 维由现有 UAV 候选描述（`17 + 2 个 UAV + 2×10 个服务`，39 维）及 STAY/MOVE 两个组标记组成；`C` 是补齐后的候选轴，当前配置最多为 25（24 个 MOVE 候选容量加 STAY）。候选 0 是 STAY，其余是当前 UAV 的合法 MOVE 目标。UAV1 的 Actor 额外接收 UAV0 已选目标位置的归一化二维条件；UAV0 的条件为零。

因此 Macro 的单次动作输出是 9 个节点各自的候选索引 `[9]`：0 表示 KEEP，正索引表示该节点 UPDATE 到对应合法候选。每个节点先选 KEEP/UPDATE 组，再在 UPDATE 时选具体内容。Micro 每 Slot 顺序输出 UAV0、UAV1 两个候选索引 `[2]`；0 表示 STAY，其他索引表示对应 MOVE 候选。各 head 的输出由保存的候选 mask 限定。

### 5.2 Critic 输入与时间尺度

Stage1A 有分开的 Macro、Micro centralized/global state value critic：`macro_value(obs)` 在 Macro 决策帧输出一个标量；`micro_value(obs)` 在每个 Slot 输出一个标量。两者都只接收同一 400 维全局 observation，并在进入 Critic 前使用 `2*obs-1` 变换，不接收 Actor 的候选特征、动作 mask 或本 Slot 真实任务。Macro 转移固定持续 2 Slot（Episode 最后可为 1 Slot）；Micro 转移持续 1 Slot。复用网络内的 Macro termination scorer 和 Long-Value critic 在此 MAPPO trainer 中冻结，不承担当前 value 输入接口。

### 5.3 每 Slot 实际顺序与 Actor 截止时刻

预测 artifact 在 Episode 执行前离线生成；MAPPO collector 不在每 Slot 重跑预测器。对决策 origin `t`，车辆/流行度预测按文件对齐为目标 `t..t+4`，输入截止 `t-1`；Episode 尾部由 horizon mask 关闭超界步。Collector 的 `AlignedRolloutStep` 同时装载了预测场景、实际 `tasks.csv` 行和执行所需的真实轨迹坐标，但实际值用于后续环境执行。

每个 Slot 的实际调用顺序为：

1. 取 origin `t` 对应的 frozen prediction/scenario slice，并使用进入该 Slot 的 Cache、UAV 状态。
2. 若 Slot 索引为偶数，在 Cache 动作前构造 Macro observation；根据预测场景生成 9 组候选和 mask；Macro Actor 输出并应用目标 Cache。其目标配置保持到下一个两 Slot Macro 边界。
3. 用已应用的目标 Cache 和当前 UAV 状态构造 Micro observation。生成 UAV0 候选并选动作，再将 UAV0 选中位置作为条件生成/评分 UAV1 候选并选动作。
4. 调用正式 `execute_configured_slot`，此时才把 `step.actual_tasks` 与 `step.actual_vehicle_positions` 传入真实任务执行、Matching/KKT 和资源分配。
5. 用实际结果形成 `system_utility = actual_task_utility - cache_cost - movement_cost`，验证动作/奖励合同并更新 Cache/UAV 状态供下一 Slot 使用。

故 Actor 的有效信息截止点是：预测输入只来自截至 `t-1` 的信息；Cache/UAV 状态是当前决策前环境状态（Micro 使用本 Slot Macro 选出的 Cache）；系统静态配置和当前候选/mask 可见。Actor 不读取 `tasks.csv` 当前/未来行、当前真实车辆坐标、预测标签/误差或未来 Matching/KKT 结果。实现边界的细节是 `AlignedRolloutStep` 本身包含实际任务和真实坐标，虽然 `_sample_macro_action`/`_sample_micro_head` 只收到 observation、candidate features、mask 和 UAV0 条件，collector 随后才将实际字段传给环境；这是当前调用数据流约束，不是由独立类型或进程隔离强制的安全边界。

### 5.4 当前接口边界

当前接口已明确为固定 9 节点、2 UAV、H=5、400 维全局 observation、200 维 Macro 候选、41 维 Micro 候选、Macro 每 2 Slot、Micro 每 Slot。它与此前尚未确定的字段相比已不再是接口未知；但它没有实现变长车辆/Cache/UAV 集合，也没有使用本报告提出的新时空需求张量。后续网络对接应遵守 Stage1A 的实际合同，除非另行版本化并实现新的 observation/action schema。
