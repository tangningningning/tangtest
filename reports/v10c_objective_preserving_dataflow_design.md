# V10-C Objective-Preserving Dataflow Design

STAGE = V10-C_OBJECTIVE_PRESERVING_DATAFLOW_DESIGN

本文件是基于当前训练代码、系统模型实现和已存在诊断报告的静态审计。此次只新增本报告；没有修改 trainer、policy、配置或系统模型，也没有运行训练、评估、resume、episode generation、pytest 或 final seeds。工作区中已有的其他用户修改均保持原样。

## 1. 原始系统目标：由 active code path 核验

对一次可行的 slot transition，`src/decision/slot_environment.py::execute_slot` 使用实际 matching/KKT 结果，而不是预测配置 surrogate 的 utility：

\[
u_k(t)=\omega_T\left(1-\frac{D_k(t)}{T_{ref}}\right)
       +\omega_E\left(1-\frac{E_k(t)}{E_{ref}}\right),
\qquad \omega_T+\omega_E=1,
\]

\[
Q(t)=\sum_{k\in\mathcal T_t}u_k(t),
\qquad G(t)=Q(t)-C_{ca}(t)-C_q(t).
\]

`D_k` 和 `E_k` 是所选 local/cloud/edge 执行路径及实际资源分配得到的任务 delay、vehicle energy。Delay 与 vehicle energy 是 `Q` 内的效用项，并非另外叠加的 reward/cost。`src/common/system_model.py::task_utility` 直接实现上式。

实际 Cache 成本由旧、新 Cache 状态计算。对 Cache node `j`，`z_j^ca=1` 当且仅当该行实际发生变化；仅新增服务产生加载量：

\[
L_j(t)=\sum_f [c^{new}_{jf}-c^{old}_{jf}]_+ S_f,
\quad E^{ca}_j(t)=\xi^{ca}_j L_j(t),
\quad C_{ca}(t)=w_{ca}\sum_j\left(z_j^{ca}\eta^{ca}_j+\lambda_{ca}E^{ca}_j(t)\right).
\]

这是 `system_model.cache_update_cost` 的实现：删除不产生 loading energy，但只要 Cache 实际改变，就计 fixed trigger cost。

对 UAV `i`，实际距离 `d_i=||q_i^{new}-q_i^{old}||`，代码按位置容差判定实际移动 `z_i^q`，有效距离为 `z_i^q d_i`：

\[
E^q_i(t)=\xi^q_i z_i^q d_i,
\quad C_q(t)=w_q\sum_i\left(z_i^q\eta^q_i+\lambda_q E^q_i(t)\right).
\]

`system_model.uav_movement_cost` 还验证最大移动距离约束。空地/地面信道、rate、CPU/BW allocation 会影响任务执行的 delay/energy，进而影响 `Q`；它们没有另一个独立 reward 项。active reward path 没有 MOVE bonus、STAY penalty、目标移动率或 Cache 改变率奖励。

\[
\texttt{GLOBAL\_OBJECTIVE\_FORMULA}
=\max_\pi\;\mathbb E_\pi\left[\sum_{t=0}^{T-1}\gamma^tG(t)\right]
\quad (\text{formal V10 config: }\gamma=1.0),
\]

等价的最小化写法是 `E[Σ(-Q + C_ca + C_q)]`。`GLOBAL_REWARD_FORMULA = reward_t = system_utility_t = actual_task_utility_t - cache_cost_t - movement_cost_t`。在当前 formal V10 配置中，其他 PPO 参数（如 `gae_lambda`、clip、learning rates）是优化参数，不是系统目标项。

## 2. 从 observation 到 reward、next state、PPO 的实际路径

```text
slot t 的 previous Cache / previous UAV positions
      + aligned step 的 vehicle/popularity predictions、horizon mask、time-to-go
                         |
                         v
        interface.build_step_observation -> o_t (400-d)
             |                         |
             |                         +--> separate critic encoder -> V(o_t)
             v
        共享 actor encoder（当前 V10）
             |
             +--> Cache heads，按 cache_node_ids 顺序逐个采样
             |       -> 每个选中的候选立即应用到本地 Cache copy
             |       -> C_t^+（整轮 Cache 决策结束状态）
             |
             +--> UAV head 0：基于 C_t^+、当前 UAV positions、step predictions
             |       -> STAY/MOVE mask -> 如 MOVE 再选候选 -> 更新 UAV_0
             +--> UAV head 1：重建 context 和可行候选（包含 UAV_0 的已执行结果）
                     -> STAY/MOVE mask -> 如 MOVE 再选候选 -> 更新 UAV_1
                         |
                         v
       build_policy_primitive_configuration -> configuration_override
                         |
                         v
       execute_slot -> final Cache/UAV state -> actual task matching
                    -> KKT CPU/BW allocation -> actual delay/energy -> Q-C_ca-C_q
                         |
              reward_t, next Cache, next UAV positions
                         |
                         v
       rollout stores (o_t, reward_t, V(o_t), action contexts/log-probs, next state)
                         |
                         v
       global GAE -> one A_t and return target per slot -> V10 branchwise PPO update
```

`collect_rollout` 在策略采样完成后才把 `step.actual_tasks`、`step.actual_vehicle_positions`、scenario slice 交给 `execute_slot`。配置 override 走 `build_policy_primitive_configuration`；随后 `execute_slot` 对 override 做验证，基于最终 Cache/UAV 位置构造 service positions，并调用 `task_allocation.optimize_task_matching`。匹配候选的真实效用经 `evaluate_matching_with_kkt` / `allocate_resources_kkt` 计算。Rollout 的下一状态直接取 `result.next_cache_state` 和 `result.next_uav_positions`。

预测配置 surrogate utility 不替代 PPO 环境 reward：直接 policy primitive override 标记 prediction solve skipped；当前 reward 仍由实际 matching/KKT 的 `task_utility_sum` 产生。

## 3. 400-d observation：字段、维度与使用方

V10 branchwise schema 属于 `HORIZON_POLICY_SCHEMAS` 和 `MODERN_POLICY_SCHEMAS`。active builder 按 Cache state、UAV position、vehicle prediction、popularity prediction、horizon mask、time-to-go 的顺序拼接。formal V10 配置给出 25 辆车辆（5 SV + 20 TV）、10 项服务、2 架 UAV、预测 horizon `H=5`；Cache node 数为 5 SV + 2 RSU + 2 UAV = 9。因而 `90+4+250+50+5+1=400`：

| 字段 / slice | Source 与 dimension | Builder 的归一化 | 决策时可见；Cache / UAV / Critic 使用 |
|---|---|---|---|
| previous Cache `[0:90]` | slot 开始时的 `previous_cache_state`，9×10 binary | 原样 0/1，row-major flatten | 是；是 / 是（UAV context 会替换成 Cache 决策后的状态）/ 是（critic 输入原始 pre-action observation） |
| previous UAV positions `[90:94]` | 2 架 UAV × (x,y) | 按 `flight_boundary` 分别线性映射到 [0,1] | 是；是 / 是（每个 head 重建）/ 是（critic 输入原始 pre-action observation） |
| predicted vehicle positions `[94:344]` | `step.predicted_vehicle_positions_horizon`，5×25×2=250 | 每个有效 horizon 的 x/y 按 flight boundary 映射到 [0,1]；无效 horizon 清零 | predictions 在决策前由 aligned step 提供；是 / 是 / 是 |
| predicted service popularity `[344:394]` | `step.predicted_service_popularity_horizon`，5×10=50 | 输入必须已经是每个 horizon 总和为 1 的概率分布；builder 校验，不再缩放；无效 horizon 清零 | 是；是 / 是 / 是 |
| prediction horizon mask `[394:399]` | horizon 长度 5 的 valid bits | bool 转 0/1 | 是；是 / 是 / 是 |
| normalized time-to-go `[399:400]` | train origin 与边界 | `(train_origin_end-origin)/(train_origin_end-train_origin_start)`，范围 [0,1] | 是；是 / 是 / 是 |

注意：此 400-d 向量**不含**当前 actual task/request list、queue/workload、actual vehicle positions、channel gains/rates、CPU/BW allocations 或其它资源状态；这些不是隐藏的 observation 字段。实际 task 与实际位置在策略采样后由环境执行路径使用。`predicted_service_popularity` 是预测需求统计，不等于实际 request/task state。

所有已列观测字段都进入 `critic_value(observation)`；critic 使用独立 `critic_encoder`，且取的是 slot 开始时的原始 `o_t`，不是 Cache/UAV action 后的条件 context。Cache、UAV actor 共享 actor encoder，但各 head 的 observation context 会按已发生的前序 Cache/UAV 决策重建。

## 4. Cache decision：context、mask、log-prob 与应用时机

- `CACHE_CONTEXT`：对 node `j` 调用 `_intermediate_observation(current_cache, previous_uav_positions, step, context)`。第一个 Cache head 看到 slot 开始 Cache；后续 head 看到前面 Cache node 已应用后的本地 Cache copy。Cache 轮次中 UAV positions 仍是 slot 开始位置。
- `CACHE_MASK`：`enumerate_cache_primitive_actions` 根据当下 Cache、node 和容量生成实际可行 primitive 候选；operation mask 始终允许 `NO_CHANGE`，其它 `{ADD, DELETE, SWAP}` 仅在存在相应候选时允许。选定 operation 后，candidate mask 只开放该 operation group 的有效候选（padding 不开放）。V8/V9 使用 replica-aware Cache candidate features。
- `CACHE_ACTION`：每个固定 `action_spec.cache_node_ids` head 先选 operation；非 `NO_CHANGE` 时再用共享 `cache_candidate_scorer` 从同 operation 候选中选择一个具体 primitive。节点集合和遍历顺序固定；变化的是基于逐步更新 Cache 状态生成的 feasible candidate/mask。
- `CACHE_LOG_PROB`：

  `L_cache = Σ_j [log π_op(a_j | c_j, m_j) + 1[a_j != NO_CHANGE] log π_cand(k_j | c_j, F_j, m^cand_j)]`。

  operation old log-probs 与 candidate old log-probs 分别按 head 保存；NO_CHANGE 不走 candidate categorical。
- `CACHE_ACTION_APPLIED_WHEN`：选出具体 candidate 后立即由 `apply_cache_candidate` 更新本地 `cache` copy。因此它会影响后续 Cache head 的 state、feasible candidates 和 candidate context。Cache head 集合本身不改变。全部 Cache head 完成后，`cache_result.final_cache_state` 被传给每个 UAV 的 `_intermediate_observation`，所以 UAV context 能看到最终 Cache action。随后 `build_policy_primitive_configuration` 将 final Cache state 与 trigger 一起形成 override；`execute_slot` 验证并将其用于 actual matching 和 next state。

这里的“立即应用”指策略采样流程中的本地状态更新；不是另一个独立 environment reward，也不在 Cache 与 UAV 之间执行一次环境 transition。

## 5. UAV_0 / UAV_1 的实际顺序与 log-prob

`sample_and_apply_v9_joint_cache_uav_operation_configuration` 按 `action_spec.uav_ids` 顺序执行；本配置两架 UAV，即 index/head 0 再 index/head 1，**不是并行采样**。

- `UAV_DECISION_ORDER = action_spec.uav_ids[0] -> action_spec.uav_ids[1]`。
- `UAV0_CONTEXT`：Cache 全部完成后的 `C_t^+`；两架 UAV 均为 slot 开始位置；本步的 prediction/horizon/time-to-go 字段。UAV_0 的 feasible MOVE candidates 由当前/旧 UAV positions、trigger map、flight boundary 和 mobility config 枚举。
- `UAV1_CONTEXT`：仍使用 `C_t^+`，但位置来自 UAV_0 决策已应用后的 `current_uav`；UAV_1 尚未移动。其候选枚举也收到保留前序选择的 `uav_triggers`。因此 MOVE 时 UAV_1 明确看到 UAV_0 更新后的位置；STAY 不改变位置，且 STAY 不作为额外 one-hot 字段拼进 observation，但前序 trigger map/实际位置进入后续候选可行性计算。
- `Cache action visible to UAV = YES`：final Cache state 进入 UAV context。UAV 的几何 feasible MOVE candidate/mask 本身由位置、trigger map、flight boundary 与 mobility config 生成，不因 Cache 状态重算。
- **operation head**：V9/V10 是一个 `uav_operation_head` Linear 模块，输出 reshape 为 `[batch, num_uav_heads, 2]`；head index 取对应 UAV 的两项 logits，顺序固定为 `STAY=0, MOVE=1`。所以 operation module 被两个 UAV 共用，但每个 UAV 使用各自的输出切片，不是两个独立网络。
- **mask**：`[STAY=True, MOVE=(MOVE candidate count > 0)]`。无可行 MOVE 时只允许 STAY。若选 MOVE，`uav_candidate_scorer` 对当时 enumerated 的 MOVE candidates 评分，candidate mask 开放真实候选、屏蔽 padding。候选特征含 UAV/head identity、operation、归一化 before/after position 与 delta、归一化距离、movement energy/cost 的 `log1p`、当前预测车辆位置统计量；这组 scorer 对两个 UAV 共用。
- 每架 UAV 的 sampled old log-prob 在 rollout 中按 head 分开保存 operation 部分与 candidate 部分。定义 `l_i^old = l_{i,op}^old + 1[MOVE_i] l_{i,cand}^old`，STAY 的 candidate 项为 0；新策略同理：

  `JOINT_UAV_LOG_PROB = l_0 + l_1 = Σ_i (l_{i,op} + 1[MOVE_i] l_{i,cand})`。

  `evaluate_joint_cache_uav_operation` 对 operation categorical 的 log-prob 沿 UAV 维度求和；candidate log-prob 只对 MOVE head 求和，再与 Cache log-prob 相加。故代码满足 `joint_uav_log_prob = uav0_log_prob + uav1_log_prob`。
- PPO update 虽收到 per-head old operation/candidate arrays，但当前将两架 UAV 的 old log-probs 求和，并用 evaluator 返回的 joint UAV branch new log-prob 得到**一个** `r_U = exp((l_0^new+l_1^new)-(l_0^old+l_1^old))`，随后作为 joint UAV branch ratio/clipping。因而 opposing per-head log-ratio 可以在 joint ratio 中抵消。当前 V10-C transition instrumentation 明确把该量标成 `joint_uav_branch_per_transition`；per-UAV MOVE/STAY 聚合中的 joint branch ratio 不是独立 per-head ratios。

## 6. Environment、reward 与 state transition 不改

`execute_slot` 使用策略 override 的 final Cache/UAV 状态构造 service positions，再以实际任务、actual vehicle positions、final Cache、局部缓存状态执行 matching。`optimize_task_matching` 评估候选 matching；`evaluate_matching_with_kkt` 对 edge tasks 调用 CPU/BW KKT fraction allocation，再计算 local/cloud/edge 的实际 delay、vehicle energy 和 per-task utility。然后 `actual_q = matching.task_utility_sum`，按 `actual_q-cache_cost-movement_cost` 形成 system utility/reward。成功 transition 的 next state 就是 final Cache 和 final UAV positions。

因此以下行为在 V10-C 中保持原样：task generation、prediction、Cache feasibility、UAV mobility、channel/rate model、matching/offloading、KKT/resource allocation、delay、energy、Cache cost、movement cost、global reward、state transition。当前研究问题不要求修改 system/environment path：

`SYSTEM_MODEL_CHANGE_REQUIRED = NO`。

## 7. Critic、GAE、当前 PPO 耦合点及证据等级

### Critic / GAE

每 slot 在任何 action 之前，以原始 `o_t` 计算一个 global state value `V(o_t)`。V10 从 V6 PopArt separate-critic 网络继承独立 critic encoder/head；它没有 `Q(s, a_cache, a_uav)` 或 per-UAV value head。Rollout 用原始环境 reward 和 value 调用标准 GAE：

\[
\delta_t=r_t+\gamma(1-d_t)V_{t+1}-V_t,
\quad A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1},
\quad R_t=A_t+V_t.
\]

V10 配置采用 centered/std 全 rollout advantage normalization。这个单一 normalized `A_t` 同时进入 Cache branch loss 和 UAV branch loss；它不是 Cache/UAV 或 UAV_0/UAV_1 独立 credit。

### 当前 V10 branchwise actor 更新

V10 将 Cache 和 UAV **两个大 branch**分开计算 PPO ratio 与 clipped loss：

`r_C = exp(L_C^new - L_C^old)`，`r_U = exp(L_U^new - L_U^old)`，

`L_actor_policy = 0.5 * (L_PPO(r_C, A_t) + L_PPO(r_U, A_t))`。

但 `L_U` 已经是两个 UAV 的 joint log-prob。Cache、UAV policy losses 都经过同一个 actor `encoder`；Cache operation/candidate heads 只属于 Cache branch，UAV operation/candidate scorers 只属于 UAV branch。Critic encoder/head 走独立 critic optimizer/loss。

Target-KL early-stop 不是 branch-specific：`approximate_kl = mean(old_joint_log_prob - new_joint_log_prob)` 使用完整 Cache+UAV joint policy log-prob。所有 minibatch 完成后若该 epoch 均值超过 `target_kl`，同一个 gate break 后续 epochs，从而停止剩余 actor 和 critic update epochs。

### 5e-4 / 6e-4 证据解释

既有 `v10c_uav_polarization_root_cause.md` 记录了 5e-4 的 UAV move collapse（MA20 <10% 于 epoch 215）及 6e-4 的高 move（MA20 >90% 于 epoch 212），表现与 actor LR 不是单调对应。最终 20 epochs：

| run | 实际 UAVMove | joint KL | Cache KL | UAV KL | KL early-stop |
|---|---:|---:|---:|---:|---:|
| 5e-4 | 3.00% | 0.0409 | 0.0306 | 0.0045 | 0% |
| 6e-4 | 95.12% | 0.0424 | 0.0415 | 0.0064 | 0% |

历史 Cache-only head gradient norm 显著高于 UAV-only head norm（最后窗口比值 5e-4=12.0651、6e-4=4.5876），但记录的 shared encoder norm 是联合 actor backward 后的单一范数，没有 `grad_shared_from_cache`、`grad_shared_from_uav` 或 cosine。故这证明不同 head 的 update pressure 不同，不证明 Cache 对 shared representation 的梯度主导或冲突导致极化。历史报告也明确将 shared-gradient dominance 标为 `PARTIAL`。

历史 formal 5e-4/6e-4 的 MOVE/STAY advantage 汇总不是每个 transition、每个 UAV 的独立 action-conditioned advantage/log-prob 配对；正负比例和因果效应不可由均值/中位数恢复。已提交的 V10-C instrumentation 报告描述的是一个 development runtime gate，不是 5e-4/6e-4 新 formal training；其 action-condition statistics 同样是 observational，不是 causal identification。

| hypothesis | code-mechanism status | existing 5e-4/6e-4 是否证明它导致极化 |
|---|---|---|
| A. Cache-UAV shared representation | **SUPPORTED**：两 branch 使用同一 actor encoder。 | **PARTIAL**：head gradient 不平衡可见，shared 梯度来源/cosine 不可见；不能归因。 |
| B. Cache-UAV branch credit | **SUPPORTED**：两个 PPO branch 使用同一 global normalized `A_t`。 | **PARTIAL**：机制确定，历史数据没有独立 branch return/因果 credit。 |
| C. UAV_0-UAV_1 joint ratio/clipping | **SUPPORTED**：两个 head 的 log-prob 求和后形成一个 ratio 和 clip。 | **NOT ESTABLISHED**：历史 per-head new ratios 不可用，不能声称实际发生抵消。 |
| D. joint KL early-stop | **SUPPORTED**：完整 joint KL 通过一个 gate 控制全部后续 actor/critic epochs。 | **NOT ESTABLISHED**：5e-4/6e-4 最后 20 epochs gate 均为 0%；不能据此归因为终段极化。 |
| E. global advantage 的单 UAV 高方差 / 状态条件混合 | `PARTIAL`：同一 global GAE signal 被复用于多个 conditional action；“高方差”本身未被独立测量。 | **PARTIAL**：历史 action-conditioned sign/ratio 样本不足，不能判断其因果作用。 |

```text
SUPPORTED = A, B, C, D（仅指代码机制确实存在，不等于已证实是根因）
PARTIAL = E（global-signal 结构存在；高方差/credit 诊断不足）
NOT_SUPPORTED = 无机制被代码否定；A/C/D 的“造成极化”因果结论均未获现有日志支持
```

## 8. Objective-preserving V10-C proposal：先增加可辨识性，不先改环境/奖励

本阶段建议保留同一个 system action、同一个环境执行路径和同一个 `G(t)`。V10-C diagnostics 可以把当前 rollout 已存的 per-head old operation/candidate log-probs 保持为 `[batch, UAV]`，并在同一 evaluator forward 中返回每个 UAV 的 `new_log_prob_i`、`r_i` 与 per-head clip flag；不要再只导出 joint UAV ratio 并重复到 per-UAV 行。该变化是 deterministic log-prob evaluation/diagnostic，不增加 action sampling，也不改变 RNG、mask、action 或 reward。

后续可以单独评估：

1. `r_i = exp(l_i^new-l_i^old)` 与独立 per-UAV clipped surrogate。它会改变 PPO clipping surrogate（不是纯日志变更），所以在没有 per-head ratios 证明 joint cancellation 影响 update 前，不建议与 encoder/baseline/KL 改动捆绑。
2. branch-specific KL gate。虽然 gate coupling 已由代码确认，但当前窗口没有显示 KL gate 驱动 5e-4/6e-4 极化。若将来实现，需要明确 branch actor 梯度 gating；只把阈值分成两个数并不能自动隔离共享 encoder 更新。
3. Cache/UAV encoder separation。若将来记录 per-loss shared gradients/cosine 并看到稳定冲突，再做单变量 architecture ablation；分开 encoder 不改变 action space/reward，但会改变 policy representation 与训练参数。
4. UAV contextual baseline `b_i(c_{t,i})`。若后续证据支持 variance reduction，可形成 `A_{t,i}=A_t-stopgrad(b_i(c_{t,i}))`；baseline 只能用 action 采样前 context、以同一个 global return/GAE target 训练，且不可作为新的 reward。`c_{t,i}` 对 autoregressive UAV_i 必须是该 head 决策发生前的 prefix state。

仅作候选设计、未被本阶段选中时的 per-head PPO 写法为：

`L_UAV = Σ_i L_PPO(r_i, A_t - stopgrad(b_i(c_{t,i})))`。

若不使用 baseline，则 `b_i=0`；若不使用 per-UAV clipping，则继续采用当前 `r_U=r_0r_1`。无论选择哪项，environment action 仍进入原 `execute_slot`，且最终优化对象仍是同一 `G(t)`，不能新增 branch reward。

### Baseline context 边界

有效 `c_{t,i}` 字段必须来自采样 UAV_i 当前 action 之前已构造的信息：

- `C_t^+`（全部 Cache decisions 完成后的 Cache state）；
- 所有 UAV 的当前 pre-action positions、previous positions；对 UAV_1 可包含 UAV_0 已完成的前序决策/位置；
- 当前 `step` 提供给 policy 的 predicted vehicle positions、predicted popularity、horizon mask、time-to-go；
- UAV_i 的 head identity，以及该时刻已生成的 feasible operation mask、MOVE candidates/candidate features；
- 已处理 heads 的 prefix trigger/position state（仅按当前决策执行顺序使用）。

严格排除：当前 UAV_i 已抽取的 STAY/MOVE、当前 candidate index、当前动作后的 UAV_i position/actual-moved flag、`reward_t`、`q_total_t`、matching/KKT outputs、delay/energy/cost、next state，以及未实现的 future realizations。Actual tasks、actual vehicle positions 虽由环境在同一 rollout step 中持有，但不在本 policy observation/conditional context 中；此设计不把它们偷偷喂入 baseline。

## 9. Objective invariants 与可能更新的参数

| invariant | 设计约束 |
|---|---|
| `GLOBAL_REWARD_CHANGED` | NO |
| `SYSTEM_COST_CHANGED` | NO |
| `CACHE_ACTION_SPACE_CHANGED` | NO；保留既有 operation、feasible candidates 与逐 node masks |
| `UAV_ACTION_SPACE_CHANGED` | NO；保留 STAY/MOVE 与现有 feasible movement candidates/masks |
| `CACHE_TEMPORAL_ADAPTATION_PRESERVED` | YES；仍按当前 Cache state 每 slot 顺序决策 |
| `UAV_TEMPORAL_ADAPTATION_PRESERVED` | YES；仍按当前 positions、prefix action 和 predictions 每 slot 顺序决策 |
| `MATCHING_OFFLOADING_CHANGED` | NO |
| `KKT_CHANGED` | NO |

禁止 MOVE bonus、STAY penalty、目标移动/Cache 更新率、人工 action balancing reward 和新 branch-specific reward。

| module / tensor | 当前可接收的梯度 | 未来方案的边界 |
|---|---|---|
| shared actor `encoder` 参数 | 当前 Cache 与 joint-UAV policy losses 的梯度都回传到此共享表示 | 若 encoder split，经各自 actor loss 更新；未决定实施 |
| Cache operation head、Cache candidate scorer 参数 | Cache branch PPO loss | 不接收 UAV branch/critic loss 梯度 |
| UAV operation head、UAV candidate scorer 参数 | joint UAV branch PPO loss；该 loss 汇合 UAV_0/UAV_1 | 若 per-UAV loss，仅按对应 head 的 `l_i` 项回传，shared encoder 仍会汇总各 head 梯度 |
| separate critic encoder/head、PopArt statistics | global return/value critic loss 与 PopArt 统计更新；独立 critic optimizer | 若增加 baseline head，以 detached global return/GAE target 做 baseline regression；baseline 输出 stop-gradient 到 actor loss |
| observations、masks、candidate features、actions、old log-probs、GAE targets | PPO update 中作为输入/target，不作为可学习参数 | per-head new log-probs 可在同一 forward 中计算；不增加采样/RNG |

## 10. 本阶段各项是否立刻修改

| decision | status | basis |
|---|---|---|
| `DECOUPLE_CACHE_UAV_ENCODER` | `INSUFFICIENT_EVIDENCE` | shared encoder 确认存在，但历史 shared gradient 没有 branch decomposition/cosine。 |
| `PER_UAV_RATIO_CLIPPING` | `INSUFFICIENT_EVIDENCE` | joint-ratio coupling 确认存在；历史 per-head new ratios/cancellation 不可用，独立 clipping 会改变 PPO surrogate。先保留/导出逐 UAV log-probs。 |
| `BRANCH_SPECIFIC_KL_GATE` | `INSUFFICIENT_EVIDENCE` | 单 joint gate 确认存在；现有极化终段 gate=0%，尚不足以将其列为原因。 |
| `UAV_CONTEXTUAL_BASELINE` | `INSUFFICIENT_EVIDENCE` | global advantage 的观察性分组不能证明 action-conditioned variance；先收集 per-UAV pre-action diagnostics。 |
| `PER_UAV_CONTEXTUAL_BASELINE` | `INSUFFICIENT_EVIDENCE` | 当前没有证据决定一个共享 UAV baseline 或两个独立 baseline；如采用，必须满足上述 action-independent prefix context 条件。 |

## 11. Source evidence index

- Observation and normalization: `src/ppo_simple/interface.py::ConfigurationObservationSpec.create`, `_normalize_xy`, `_ordered_popularity_array`, `normalized_time_to_go`, `build_configuration_observation`, `build_step_observation`。
- V10 observation schema/context: `src/ppo_simple/trainer.py::load_environment_context`（`HORIZON_POLICY_SCHEMAS` / `MODERN_POLICY_SCHEMAS` membership 与 observation spec）。
- Sequential Cache/UAV sampling and local state updates: `src/ppo_simple/primitive_control.py::_intermediate_observation`, `_cache_candidate_feature`, `_uav_candidate_feature`, `sample_and_apply_v7_cache_operation_configuration`, `sample_and_apply_v8_cache_operation_configuration`, `sample_and_apply_v9_joint_cache_uav_operation_configuration`。
- Actor/critic architecture and joint log-probs: `src/ppo_simple/policy.py::AdaptivePrimitiveHorizonGroupedSeparateCriticV4ActorCritic`, V6 PopArt critic, `AdaptiveJointCacheReplicaAwareUAVOperationV9PopArtActorCritic`, V10 branchwise subclass, `evaluate_joint_cache_uav_operation`。
- Actual environment, next state and reward: `src/decision/slot_environment.py::build_policy_primitive_configuration`, `execute_slot`; `src/decision/task_allocation.py::optimize_task_matching`, `allocate_resources_kkt`, `evaluate_matching_with_kkt`; `src/common/system_model.py::task_utility`, `cache_update_cost`, `uav_movement_cost`。
- Rollout and GAE: `src/ppo_simple/rollout.py::collect_rollout`, `_check_reward`, `compute_gae`; PPO branch losses/gate: `src/ppo_simple/trainer.py::PPOV2Trainer.update`。
- Historical limits: `reports/v10c_uav_polarization_root_cause.md`; current instrumentation semantics/runtime-gate scope: `reports/v10c_action_conditional_instrumentation.md`; V10 horizon/count/config values: `configs/lr_sweep/v10_branchwise_actor_lr_6e4_kl.toml`.

## Final status

GLOBAL_OBJECTIVE_VERIFIED = YES
GLOBAL_REWARD_VERIFIED = YES
OBSERVATION_PATH_VERIFIED = YES
CACHE_PATH_VERIFIED = YES
UAV0_PATH_VERIFIED = YES
UAV1_PATH_VERIFIED = YES
REWARD_PATH_VERIFIED = YES
GAE_PATH_VERIFIED = YES
SYSTEM_MODEL_CHANGE_REQUIRED = NO
CACHE_UAV_COUPLING_FOUND = YES
UAV0_UAV1_COUPLING_FOUND = YES
KL_GATE_COUPLING_FOUND = YES
CREDIT_ASSIGNMENT_ISSUE_SUPPORTED = PARTIAL
DECOUPLE_CACHE_UAV_ENCODER = INSUFFICIENT_EVIDENCE
PER_UAV_RATIO_CLIPPING = INSUFFICIENT_EVIDENCE
BRANCH_SPECIFIC_KL_GATE = INSUFFICIENT_EVIDENCE
UAV_CONTEXTUAL_BASELINE = INSUFFICIENT_EVIDENCE
PER_UAV_CONTEXTUAL_BASELINE = INSUFFICIENT_EVIDENCE
OBJECTIVE_PRESERVED = YES
TEMPORAL_ADAPTATION_PRESERVED = YES
TRAINING_EXECUTED = NO
EVALUATION_EXECUTED = NO
RESUME_EXECUTED = NO
EPISODE_GENERATION_EXECUTED = NO
PYTEST_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
GIT_DIFF_CHECK = PASS
TEMP_FILES_REMOVED = YES（本阶段未生成临时文件或测试脚本）
PUSHED = NO
