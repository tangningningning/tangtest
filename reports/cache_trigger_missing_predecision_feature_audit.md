# Cache Trigger 缺失 pre-decision 特征静态审计

## 范围

只回答：Cache Trigger 在 KEEP/UPDATE 前已合法可用、会影响 UPDATE 相对 KEEP 价值、但当前没有进入 Trigger observation 的信息是什么。

已关闭的既有诊断结论按用户要求直接复用；没有重跑 raw Ridge/MLP、symmetric-Q、task marginalization、saturation、temporal、encoder 或 gradient 诊断。没有读取 checkpoint、执行 Q 分支或修改源码/配置。

## 正式 proposed 输入路径

当前配置选择 adaptive joint Cache/UAV reconfiguration 与 adaptive joint Cache/UAV policy（system_config.toml:543-547）。trainer 从 scenario/cache nodes、service catalog 与 action spec 建 observation spec（trainer.py:3572-3624）；rollout 先调用 build_step_observation，再进入 joint primitive（rollout.py:230-274）。

配置规模为 5 SV、20 TV、10 services、2 RSU、2 UAV、prediction horizon 5（system_config.toml:82-87,104-107,185,212,430-432）。Cache node 的固定顺序是 SV、RSU、UAV（cache_configuration.py:356-400）。因此 M=9、F=10、U=2、V=25、H=5。

每个 Cache head 收到一个 400-D 全局 conditional context：
M*F + 2*U + 2*V*H + F*H + H + 1
= 90 + 4 + 250 + 50 + 5 + 1 = 400。
Cache contexts 的张量形状为 9×400。Cache heads 顺序决策；较早 head 已选的更新会反映在后续 head 的 working-cache 字段。UAV heads 在 Cache heads 之后执行，因此 Cache Trigger 看到的是本 slot 当前 UAV 位置。

policy 将 observation 输入 cache_encoder，再由 cache_trigger_head 产生 [batch, 9 heads, 2 logits]；node_index 选择一个输出行（policy.py:1741-1805,1891-1934）。head-specific 输出权重不等于 head-specific observation。

### 当前 Trigger observation/cache_context

| FEATURE_NAME | SOURCE_VARIABLE | DIMENSION | NORMALIZATION | TIME_INDEX | PREDICTED_OR_CURRENT | GLOBAL_OR_NODE_LOCAL | SOURCE |
|---|---|---:|---|---|---|---|---|
| working_cache_state | primitive 的 cache，初值 previous_cache_state | 9×10=90 | 原始 0/1 | 当前 slot；带同 slot 较早 Cache heads 的顺序决策前缀 | 当前/顺序条件状态 | 全局 node×service 矩阵，固定 ID 顺序 | primitive_control.py:1218-1228,1262-1268；interface.py:391-410,433-435,478-489 |
| uav_positions | primitive 的 current_uav，初值 previous_uav_positions | 2×2=4 | 按 flight boundary 对 x/y min-max 到 [0,1] | 当前 slot，Cache 决策时尚未执行本 slot 的 UAV heads | 当前 | 两架 UAV 的全局坐标 | primitive_control.py:1220-1228,1445-1458；interface.py:206-228,412-439 |
| predicted_vehicle_positions | step.predicted_vehicle_positions_horizon | 25×5×2=250 | 每个 x/y 按 flight boundary min-max 到 [0,1]；无效 horizon 置零 | h=0..4，对应 slot t..t+4 | 预测 | 全车 global tensor，无按 cache node 聚合 | interface.py:519-536,290-318,441-459,884-925 |
| predicted_service_popularity | step.predicted_service_popularity_horizon / rho_pred | 10×5=50 | 每 horizon 是总和为 1 的 service 概率；不再缩放 | h=0..4，对应 slot t..t+4 | 预测 | 全局 service distribution，无 node/vehicle 条件 | interface.py:703-737,927-958,321-348,427-432,460 |
| prediction_horizon_mask | step.prediction_horizon_mask | 5 | bool/0-1 | h=0..4 forecast validity | 预测有效性元数据 | 全局 horizon mask | interface.py:371-388,417-425,461-465；trainer.py:3610-3624 |
| time_to_go | normalized_time_to_go(origin_index, train_start, train_end) | 1 | (end-origin)/(end-start)，范围 [0,1] | 当前 origin 到 training terminal | 当前时序元数据 | 全局 | interface.py:351-368,508-536,466-476 |

没有独立的 node ID/role、RSU 坐标、cache capacity、service MB size、request count、input bits 或 CPU cycles observation 坐标。Service/node 身份通过固定坐标顺序及外部 head index 对齐。

### 不在 encoder vector 里的 Trigger 控制量

- head_index=node_index 是选择一个固定 node/head 输出行的标量元数据，不是 observation 坐标（primitive_control.py:1262-1287；policy.py:1891-1907）。
- operation_mask=[KEEP=True, UPDATE=bool(update_candidates)] 是两个 action-validity 位。它只告知至少存在一个 UPDATE candidate；不传候选数量或质量（primitive_control.py:1269-1287）。

## KEEP/UPDATE value-dependency chain

| 环节 | 源码事实 | 对 UPDATE−KEEP 的影响 |
|---|---|---|
| Trigger/candidate mask | 每 head 在 Trigger 前以 current cache、catalog、proposal nodes、prediction scenario、TV local cache、cache config 枚举候选；Trigger 只收到候选非空 mask（primitive_control.py:1262-1287）。 | 相同的 UPDATE 概率可能对应不同候选集和条件结果。 |
| conditional UPDATE target | enumerator 使用节点 cache、service sizes、capacity、node-specific predicted demand、local-cache usage，生成 low-reconfiguration/prediction/replica-aware/intermediate targets，再去重并做 capacity filter（cache_configuration.py:1612-1694）。 | target 的服务增删、可行性、需求收益及副本覆盖改变。 |
| candidate scorer | 只有选中 UPDATE 后才生成 candidate features 并调用 candidate scorer（primitive_control.py:1324-1367；policy.py:1936-1968）。特征含 before/target/delta、预测需求增益、local usage 增益、capacity、loaded/released MB、cache cost/energy、replica summary（primitive_control.py:357-462）。 | 候选 quality 会改变 UPDATE 条件分布，但没有回传给 Trigger。 |
| target configuration/cache cost | build_policy_target_configuration 验证 trigger 与 target，并计算 cache cost（slot_environment.py:843-917）。system_model 的 cost 由实际变更的 eta_ca 和新增服务 loaded MB×xi_ca×lambda_cache，再乘 cache_cost_weight（system_model.py:224-304）。 | 是否更新、目标服务与 footprint 直接改变即时成本。 |
| offloading feasibility/Matching | execute_slot 由 final cache 和节点位置构造 service positions，再用 actual_tasks/positions 与 local cache 调 optimize_task_matching（slot_environment.py:425-460）。feasible targets 由 task.service_id 的 local/edge cache 命中与边缘物理路径决定（task_allocation.py:227-294）；transfer/exchange 优化 task_utility_sum（task_allocation.py:1010-1023,1085-1135）。 | Cache target 可能增删某任务的 edge target，改变 matching 和节点任务集。 |
| KKT/delay/energy/utility | edge node 的 task set 进入 KKT；edge_execution 使用 input bits、cpu cycles、bandwidth、tx power、channel gain、CPU 和资源份额；task_utility 组合 delay/energy（task_allocation.py:2741-2822,1950-1967；system_model.py:90-128,368-394）。 | 预测 workload/位置与节点资源影响分支 utility。 |
| reward/next state | reward=actual_task_utility-cache_cost-movement_cost，next state 返回 final cache/UAV state（slot_environment.py:523-597；rollout.py:584-650）。 | 即时效用、cache cost 和后继状态均可改变分支结果。 |
| critic bootstrap / Q | 现有 helper 静态定义 q=reward+gamma*critic_value(next_observation)（cache_credit.py:488-505）；其 branch helper 通过正式 execute_slot（cache_credit.py:331-448）。本任务未执行 helper 或任何 branch。 | Q_UPDATE−Q_KEEP 同时依赖即时 reward 差与候选造成的 next-state/value 差。 |

## Value-relevant variable checklist

当前 observation 指进入 Cache encoder 的 400-D conditional_context；合法性 mask 另列。预测 scenario/catalog/config 虽在决策前可得，但不代表已进 Trigger observation。

| VALUE_RELEVANT_VARIABLE | WHERE_USED / HOW IT CHANGES ΔQ | AVAILABLE_BEFORE | IN_TRIGGER_OBSERVATION | DERIVABLE_FROM_INPUT | 分类 |
|---|---|---|---|---|---|
| 当前各 Cache node/service bits | candidate delta 与 edge-feasible service；决定当前和目标内容 | YES | YES，完整 9×10 | YES | ALREADY_PRESENT |
| 其他 edge nodes 的 cache replicas | replica-aware targets/替代服务覆盖 | YES | YES，其他 node 行均存在 | YES | ALREADY_PRESENT |
| node-local forecast workload by service | forecast arrival/service/position；node-specific demand；workload 会进入 Matching/KKT | YES，prediction scenario summary | NO | NO；全局概率/位置无 request mass 和 node aggregation | DIRECTLY_MISSING |
| predicted vehicle-to-node coverage/association | 改变任务可能被哪个 cache node 服务 | YES，预测坐标与固定几何 | PARTIAL；坐标全局，无 node-relative score | PARTIAL | PARTIALLY_REPRESENTED |
| TV local-cache membership | feasible_targets 发现 task.service_id 已本地缓存时加入 local target，减少新增 edge opportunity | YES，固定 local-cache artifact | NO | NO | DIRECTLY_MISSING |
| service MB size/current used/residual cache capacity | candidate feasibility、loaded MB、cache cost | YES，catalog/config/current cache | PARTIAL；service/cache bits 有，MB 值无；mask 仅表示非空 | NO，无外部尺寸/容量时不可由 bitset 恢复 | PARTIALLY_REPRESENTED |
| UPDATE candidate count/target/demand/replica/cost/capacity summary | 决定 UPDATE 条件结果；Trigger 当前仅获知是否非空 | YES，候选先于 Trigger 枚举 | NO | NO；boolean mask 丢失数量/内容 | DIRECTLY_MISSING |
| global popularity | per-service future request probability | YES | YES | YES | ALREADY_PRESENT |
| expected request count/workload intensity | 决定服务访问质量；概率分布不包含数量 | YES，forecast scenario | PARTIAL；无计数/负载 | NO | DIRECTLY_MISSING（node-local forecast feature） |
| predicted input bits/CPU cycles | 改变 KKT、delay、energy、utility | YES，forecast scenario draws；非 actual_tasks | NO | NO | DIRECTLY_MISSING（forecast workload channel） |
| service size/cycles-per-bit catalog | size 进容量/cache cost，cycles-per-bit 影响 workload | YES，静态 catalog | NO raw numeric coords；service ID 坐标存在 | PARTIAL，固定 catalog 可外部查，不由 observation 数值恢复 | PARTIALLY_REPRESENTED |
| fixed node CPU/bandwidth/tx power/role | 通过 KKT、channel/edge execution 影响 utility | YES，静态角色配置 | NO raw coords | YES，经固定 head/role/config 映射；不是逐 slot state | REDUNDANT_DERIVABLE |
| actual task IDs/services/input/cycles、realized Matching/KKT/delay/energy/reward | execute_slot 输出与分支结果 | NO，决策后 realization/result | NO | NO | ILLEGAL_FUTURE_INFORMATION |
| next-state critic value / exact candidate Q | branch bootstrap value | NO，执行/评价后才有 | NO | NO | ILLEGAL_FUTURE_INFORMATION / REJECTED |

预测场景从 prediction_scenarios artifact 加载（cache_configuration.py:318-345）；generator 基于 forecast rho_hat 采样 arrival/service/workload scenario（scenario_generator.py:603-669）。这类值只允许作为预测统计量，不是实际 task stream。正式接口分别装载 prediction scenario 与 actual tasks（interface.py:808-835,1006-1026）。

## PRESENT_GLOBALLY_BUT_NOT_HEAD_CONDITIONED

1. vehicle positions 与 service popularity 都以全车/全服务、全 horizon 全局张量进入每个 Cache context；没有 node×vehicle×service 的请求、workload 或 serviceability 汇总。候选枚举另算 node_specific_predicted_demand，却没有把它写入 Trigger context（cache_configuration.py:1429-1552）。
2. 全部 cache bits 已在 observation，未被删除；但没有当前 head 的 MB 占用/剩余容量坐标。
3. UAV 坐标和 SV 的预测坐标在全局字段；没有 node-relative distance/coverage。RSU 固定坐标留在 node config。
4. role/head identity 由固定输出 row/config 关联，不是 encoder observation feature；encoder 仍收到全局向量。

## Candidate information 是否反馈给 Trigger

TRIGGER_CANDIDATE_VALUE_CONTEXT_MISSING = YES。

候选在 Trigger 前枚举，但 mask 只有 bool(update_candidates)。Trigger 没收到 candidate count、candidate_features、candidate policy probabilities 或 quality summary。选中 UPDATE 后才构造特征并调用 candidate scorer。候选特征已有 candidate target/delta、predicted demand gain、observed local-cache usage gain、capacity utilization、loaded/released MB、cache cost/loading energy 和 replica summary（primitive_control.py:1269-1287,1324-1358,357-462）。这是结构性信息缺失，不代表把 exact-Q 或 reward 放进 observation。

## 最小修复候选（只定位，不改代码）

### TIER_A_DIRECT_VALUE_DEPENDENCY

| FEATURE | SOURCE_VARIABLE | PRE_DECISION_LEGAL | CURRENTLY_ABSENT | DIRECT_VALUE_PATH | SOURCE_CODE_EVIDENCE |
|---|---|---|---|---|---|
| node_local_predicted_service_workload[m,f,:]：按 service 聚合 forecast request mass、input-bit workload、CPU-cycle workload，并条件化 node serviceability/coverage；可标记 local-cache hit | step.scenario_slice[arrival, service_id, input_size, cpu_cycles, positions_xy] + context.local_cache + cache_nodes | YES；只用预测 scenario/local-cache/config，不用 actual_tasks | YES | 预测服务机会决定新增 edge target 的价值；input/cycles 进入 Matching/KKT、delay/energy/utility；local-cache hit 直接改变 feasible targets | cache_configuration.py:1429-1552；scenario_generator.py:603-669；task_allocation.py:227-294,2741-2822 |
| update_candidate_policy_summary[m]：legal candidate count；按 pi_old(c|UPDATE) 汇总 per-service target/demand/local-usage delta，以及 loaded MB/cache cost/capacity/replica effects | update_candidates + candidate_features + candidate_probabilities + current cache/catalog/config | YES；合法 targets 与预测/catalog/state summary，无 execute_slot/branch Q | YES；Trigger 当前只见候选非空 bit | target 控制更新后服务集与 next state；新增 footprint/eta/xi 直接进入 cache cost；候选分布决定 UPDATE 的条件结果 | primitive_control.py:1269-1287,1324-1367,357-462；cache_configuration.py:1612-1694；system_model.py:224-304；slot_environment.py:843-917,523-597 |

这些候选仅描述 prediction、合法候选及已知 cache state/cost proxy；不是实际 task、reward、Matching output、critic Q 或 exact-Q branch。

### TIER_B_PLAUSIBLE_BUT_INDIRECT

- 单独输入所有 service catalog 常量或固定 per-head CPU/bandwidth/tx-power：这些值参加容量/费用/KKT，但固定 service/head/role/config 已知；candidate/workload summary 能带入当前状态相关影响。
- 更精细 predicted link-rate/edge-feasibility margin：global coordinates 与 fixed geometry 可导出，但不是比 node-local workload/candidate summary 更小且更直接的特征集合。
- 无源码价值定义的手工 service marginal utility 指标：可能有用，当前静态证据不足以放进 Tier A。

### REJECTED_NOT_JUSTIFIED

- actual_tasks、真实 task arrivals/services/input/cpu cycles、依赖真实 task IDs 的 actual vehicle subset、realized Matching/KKT、delay/energy/reward、branch Q：执行后信息，禁止作为 Trigger input。
- raw future task draw：只能用 forecast expectations/summary，不能暴露实际任务。
- 重复新增 global cache bits、predicted coordinates/popularity 或固定 head/role identity：已有或由固定配置/head 映射确定。

## 最终判定

CURRENT_TRIGGER_OBSERVATION_COMPLETE_FOR_CACHE_VALUE = NO
NODE_LOCAL_VALUE_INFORMATION_MISSING = YES
CANDIDATE_VALUE_CONTEXT_MISSING = YES
RESOURCE_STATE_INFORMATION_MISSING = YES
CACHE_COST_RELEVANT_INFORMATION_MISSING = YES
DEMAND_LOCALIZATION_INFORMATION_MISSING = YES
MINIMAL_JUSTIFIED_FEATURE_SET = [
  "node_local_predicted_service_workload[m,f,:]",
  "update_candidate_policy_summary[m]"
]

ACTUAL_TASKS_REQUIRED = NO
FUTURE_INFORMATION_REQUIRED = NO
TRIGGER_CANDIDATE_VALUE_CONTEXT_MISSING = YES

RESOURCE_STATE_INFORMATION_MISSING 指 forecast node/service workload 和 candidate storage/capacity/load summary；SV/RSU/UAV 的固定 CPU、bandwidth、tx-power 是 head/role/config 常量，不把它们误报成动态逐 slot state。

## 执行与交付

- Source/config files modified: NO
- Training/backward/optimizer.step/validation/test: NO
- New exact-Q branches executed: 0
- Reports: reports/cache_trigger_missing_predecision_feature_audit.md 与 reports/cache_trigger_missing_predecision_feature_audit.json
