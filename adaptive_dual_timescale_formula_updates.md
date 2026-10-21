# Adaptive 双时间尺度 Cache/UAV PPO 数学合同

本文件是正式算法和实现的长期合同。公式描述策略决策、rollout 与 PPO target；Cache loading、通信、delay、energy 和 utility 的物理计算仍由现有模块负责。本版不迁移训练框架，也没有直接复制第三方代码。

## 1. PROPOSED 正式控制定义

唯一正式控制器使用 Macro Cache SMDP 和每 slot Micro UAV PPO。Macro frame 起点选择 Unified Cache candidate；每个 slot 执行后，termination policy 根据最新状态决定是否结束该 frame。Cache 在同一 frame 内固定。Micro 对 UAV0、UAV1 使用顺序条件式 masked categorical。完整 episode 收集结束后分别更新两套 PPO。

- Cache candidate 0 表示 KEEP，其余合法候选表示 UPDATE；合法 UPDATE logits 按候选数量做 cardinality correction。
- Holding bounds 固定为 `adaptive_holding_min_slots = 1`、`adaptive_holding_max_slots = 5`，并要求最大值不超过 prediction horizon。
- Cache policy 只决定当前 frame 使用哪个 Cache configuration。每个已执行 slot 后，若 episode 尚未结束且 elapsed holding 小于 5，则 termination categorical 采样 `0=CONTINUE` 或 `1=TERMINATE`。
- Termination input 是实际 post-slot/next-slot observation 与 `elapsed_holding_fraction = elapsed_holding_slots / adaptive_holding_max_slots`。正式 observation 已包含 active Cache 时不重复拼接 Cache tensor。
- duration 不是预定义 categorical。若 `z_{k,e}` 是 elapsed slot `e` 时真正采样的 decision，则 `D_k = min({e ∈ {1,...,D_max-1}: z_{k,e}=TERMINATE} ∪ {D_max})`；episode terminal 优先结束 frame，且不产生额外 decision。
- 一个 frame 的 joint log probability 为 `log π_macro,k = log π_cache(C_k | S_k) + Σ_j log π_term(z_{k,j} | X_{k,j})`，只对真实采样的 termination decision 求和。
- Micro 将 STAY 和 MOVE 放在同一 masked candidate head 中；先选择 UAV0，再依据其最终位置重建 UAV1 的安全候选集。
- Macro/Micro rollout、Critic、optimizer 和 advantage normalization 分离。
- Macro 使用 SMDP GAE 与 two-step TD Critic target；训练回报由现有 Matching、KKT 和系统 utility 产生。强制 max-holding termination 与 episode terminal 不采样 policy action，也不进入 log probability。

## 2. Macro 决策边界和 Cache persistence

令 episode 的 slot 数为 `T`，Macro frame 起点为 `τ_k`，实际 holding duration 为 `D_k`。初值与边界递推为：

```text
τ_0 = 0
τ_{k+1} = τ_k + D_k
D_k = min({e ∈ {1,...,D_max-1}: z_{k,e}=TERMINATE} ∪ {D_max})
```

因此 episode 尾帧会在真实 terminal 处结束，即使尚未到 `D_max`；terminal slot 后没有 termination action。其它 frame 在第一个 sampled TERMINATE 处结束，若所有 eligible decision 都采样 CONTINUE，则在 `D_max` 强制结束。Macro SMDP return、discount 和 GAE trace 全部使用实际 `D_k`。

Cache 在同一 frame 内保持不变：

```text
c_{m,f}(t) = c_{m,f}^{(k)},  τ_k <= t < τ_{k+1}
```

Macro action 是联合动作：

```text
C_k ~ π_cache(. | S_k)
X_{k,j} = [S_{k,j}^{post-slot}, elapsed_holding_slots / D_max]
z_{k,j} ~ π_term(. | X_{k,j}),  0=CONTINUE, 1=TERMINATE
log π_macro,k = log π_cache(C_k | S_k) + Σ_j log π_term(z_{k,j} | X_{k,j})
r_k = exp(log π_macro,k,new - log π_macro,k,old)
```

Cache candidate 0 是当前实际配置，代表 KEEP。实际 Cache 更新按节点定义：

```text
z_m^ca(k) = I[存在 f: c_{m,f}^{(k)} != c_{m,f}^{(k-1)}]
```

重新进行 Macro 决策不等于发生 Cache 更新。若 `C_k = C_{k-1}`，则 `z_m^ca(k)=0`，loaded amount 为 0，且不产生 Cache reconfiguration cost；若配置不同，才在 boundary slot 执行实际更新并调用现有 Cache loading/cost helper。更新 cost 只在新配置生效的 boundary slot 计算一次。

实际 Cache 下载仍按新增服务计算，服务大小是十进制 MB、回传速率是 bit/s：

```text
T_ca,m(t) = sum_f [(8e6 * s_f_MB / R_back,m_bps) * max(c_new,mf(t)-c_old,mf(t), 0)]
T_ca,m_ref = 800 s  (fixed configured per-node design scale)
C_ca(t) = sum_m [T_ca,m(t) / T_ca,m_ref]
```

当前正式 `MULTI_NODE_FACTORIZED` 策略对 5 个 SV、2 个 RSU、2 个 UAV 缓存节点提供在线配置动作，故全部 9 行缓存状态按节点分别归一化；不变节点的实际下载成本仍为零。运行初始化对每个节点固定 `T_ca,m_ref=800 s`，这是设计归一化尺度，不是物理上界。按当前容量及回传速率，各 SV/RSU/UAV 的物理下载时延上界为 `480/1000/800 s`，全系统合计 `6000 s`；所以归一化 Cache 成本的容量约束上界为 `7.5`，RSU 单节点成本可能超过 `1`。Cache 成本函数返回未加权的 `C_ca=sum_m(T_ca,m/T_ca,m_ref)`。`beta_ca=1.0` 在唯一系统效用聚合函数中乘一次。配置中的 `lambda_cache` 仅作为旧合同的分析/报告字段保留，不进入 Cache 成本或当前奖励。参考尺度必须为有限正数；零容量或非法回传速率在初始化时明确拒绝。

UAV 能耗仍使用实际前后水平二维位置差：

```text
E_mov,u(t) = xi_mov,u * ||q_u(t)-q_u(t-1)||_2
E_mov,u_ref = xi_mov,u * vmax,u * dt
C_q(t) = sum_u [E_mov,u(t) / E_mov,u_ref]
```

每架 UAV 的速度上限给出保守能耗上界；位置边界与联合安全距离只会进一步限制实际能耗。UAV 成本函数返回未加权的 `C_q=sum_u(E_mov,u/E_mov,u_ref)`；显式外部权重 `beta_q=0.00225` 在唯一系统效用聚合处乘一次。它保持原有效系数 `(5e-6 J^-1)*(sum_u E_mov,u_ref=450 J)`，而 `lambda_movement` 仅作为旧合同的历史/分析字段，不再用于当前奖励。`slot_environment` 调用 `system_model.aggregate_system_utility` 计算 `G(t)=Q(t)-beta_ca*C_ca(t)-beta_q*C_q(t)`；Cache/UAV 成本字段都不含外部权重。

当前每个 Cache 节点固定使用 `800 s` 参考尺度，9 个节点的参考值总和为 `7200 s`；按各节点容量与速率计算的物理下载上界合计 `6000 s`，因此 `C_ca<=7.5`，而非每节点均严格不超过 `1`。每架 UAV 的移动参考值仍为 `225 J`，两架总和 `450 J`；未加权 UAV 归一化成本上界仍为 `2`，`beta_q=0.00225`，加权扣费上界仍为 `0.0045`。正式配置没有 `cache_cost_weight=0.1` 和 `uav_movement_cost_weight=0.50` 这两个参数，配置校验拒绝它们。当前 `lambda_cache=2e-5`、`lambda_movement=5e-6` 仅保留为旧合同/分析字段，不再计算当前系统效用。Cache 分母由节点容量上界改为固定设计尺度，改变了缓存成本的相对贡献及优化目标。成本合同版本为 `node_level_cost_fixed_reference_explicit_weights_v2`；严格恢复门禁逻辑未修改，配置哈希及成本合同可区分旧 Checkpoint。

## 3. Macro frame reward 与 SMDP target

Macro frame reward 保留 slot reward 的折扣和，不做按 duration 平均：

```text
R_k^C = Σ_{j=0}^{D_k-1} γ^j G(τ_k + j)
```

SMDP TD error 使用经过的实际 slot 数作为指数：

```text
δ_k^C = R_k^C + γ^(D_k) V_C(S_{k+1}^C) - V_C(S_k^C)
```

duration-aware GAE 为：

```text
A_k^C = δ_k^C + (γ λ)^(D_k) A_{k+1}^C
```

terminal transition 的 `next_value` 和后续 advantage 均由 `done` 截断。禁止将 Macro TD 写成 `R + γ λ V_next - V`，也禁止用 requested duration 替代 actual duration。

实现路径：`rollout.collect_dual_timescale_episode` 累积 `frame_reward`；`rollout.compute_macro_smdp_gae` 使用每条 transition 的 `actual_duration`；`trainer.PPOTrainer._update_macro` 构造 Macro PPO target。

## 4. Micro UAV 条件式联合动作

每个 UAV 使用统一候选集合：

```text
A_t^u = {STAY, P_1, ..., P_K}
candidate 0 = 当前 UAV position = STAY
candidate 1..K = 当前正式生成器给出的可行 MOVE positions
```

两 UAV 的联合策略按固定 ID 顺序分解：

```text
π(a_0, a_1 | S_t) = π(a_0 | S_t) · π(a_1 | S_t, a_0)
joint_uav_log_prob = log_prob_uav0 + log_prob_uav1
```

先从 UAV0 当前候选和合法 mask 中采样；将 UAV0 最终位置应用到临时配置后，再调用现有 UAV candidate generator 为 UAV1 重建安全约束、候选和 mask。UAV1 的条件描述符包含已选择 UAV0 的位置。sampling 与 update evaluate 使用相同候选描述符、当时保存的 masks、固定 UAV 顺序和 group-balance 变换。

mask 合同为：mask 在构造 categorical distribution 前作用于 logits；每个 head 至少有一个合法动作；被 mask 的动作概率为 0；rollout 保存实际动作时的 candidate descriptors 与 masks；update 只重用这些历史数据，不按当前环境重建旧动作支持。

### STAY/MOVE 组熵底限

对每个 UAV，令 `p_stay` 为当前 masked categorical 中 candidate 0 的概率，`p_move = Σ_{i>0} π_i` 为全部合法 MOVE candidates 的概率总和。STAY/MOVE 二元组熵为：

```text
H_group = -p_stay log(p_stay) - p_move log(p_move)
```

概率软上限 `p_cap = 0.90` 只用于推导熵底限，不会直接裁剪或限制 categorical 概率：

```text
H_floor = -p_cap log(p_cap) - (1 - p_cap) log(1 - p_cap)
        = -0.9 log(0.9) - 0.1 log(0.1)
        ≈ 0.325083
```

Micro Actor 的优化 loss 增加仅在低于底限时生效的软正则项：

```text
L_floor = c_floor · mean_{UAV0,UAV1,batch}(ReLU(H_floor - H_group))
L_micro_actor = L_joint_PPO + L_floor
c_floor = 0.05
```

当两个 UAV 的组熵都不低于底限时，`L_floor` 严格为 0。该项只对 Micro Actor 参数产生梯度，不向 Micro Critic 传播；它不修改实际 action distribution、训练中的随机采样或 deterministic group-aware evaluation，也不加入普通的完整 categorical entropy bonus。现有 `micro_candidate_entropy`/`micro_entropy` 仍是整个候选 categorical 的熵诊断；`H_group` 只衡量 STAY 与 MOVE 两组之间的概率分配，不可互换。

实现路径：`policy._masked_categorical`、`policy._group_balance_candidate_logits`、`policy.DualTimescaleActorCritic.sample_micro_head`、`policy.evaluate_micro_joint`、`rollout._uav_candidates` 和 `rollout.collect_dual_timescale_episode`。

## 5. 独立 rollout 与 GAE

每 slot 一条 Micro transition，保存状态、两 UAV action index、joint old log probability、Micro value、真实 `G(t)`、next state、done、UAV0 mask、基于 UAV0 已选位置条件化的 UAV1 mask、候选描述符及 active Cache configuration identifier。

每个完整 Cache frame 一条 Macro transition，保存 frame-start observation、Cache candidate features/mask/action、old Cache log probability、实际 duration、Macro old joint log probability/value、frame reward、next state 和 done。可变长 termination trace 保存 `termination_observations`、`termination_elapsed_slots`、`termination_actions`、`old_termination_log_probs`、`old_termination_entropies`；只包含真正采样的决策，不包含 padding、min-holding 强制 CONTINUE、max-holding 强制 TERMINATE 或 episode-terminal decision。Macro batch 将这些 trace pad 到 `[B,T_max,dim]` 并用 mask 排除 padding。两个 buffer 永不混为同一 transition list。

Micro 使用 ordinary GAE：

```text
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
A_t = δ_t + γ λ A_{t+1}
```

训练以四个独立 realization 组成一个 policy update group：

```text
B_k = {τ_k^1, τ_k^2, τ_k^3, τ_k^4}
τ_k^i ~ π_{θ_old}
```

组内四条轨迹都使用相同且冻结的 `π_{θ_old}`。对每条 realization 独立计算 Micro GAE、Macro SMDP GAE、Macro N2 target 和 realized return-to-go；任何递归目标都不跨 realization。随后拼接四条轨迹的 raw advantages 与 targets，只在整个 update group 上分别标准化 Micro 和 Macro advantage：

```text
A_norm = (A - mean(A)) / (std(A) + ε)
```

Macro frame reward 不除以 `D_k`；两尺度不共享 reward/advantage/value normalization state。本版没有启用 PopArt。每个 update group 共用一次标准化，不能先对单条 realization 分别标准化。

实现路径：`rollout.MicroRolloutBuffer`、`rollout.MacroRolloutBuffer`、`rollout.compute_gae`、`rollout.compute_macro_smdp_gae` 和 `trainer.PPOTrainer.update_from_episodes`。

## 6. PPO updates、优化器与 checkpoint

四条 realization 的 collection 完成前，Macro/Micro Actor、Macro/Micro Critic、Long-Value 与 optimizer 都不更新。组内完成后先执行一次 Micro PPO phase，再执行一次 Macro PPO phase。Micro 使用合并后的 slot transitions 与配置 minibatches，UAV0/UAV1 仍通过一个 joint log probability、一个 joint ratio 和一个 clipped surrogate 执行 PPO；STAY/MOVE 组熵底限仅作为 Micro Actor loss 的附加软正则项，不拆分 joint PPO。Macro 对合并后的全部 Macro transitions 做 full-batch PPO。令 `log π_macro = log π_cache + Σ_j log π_term,j`，则 `r_macro = exp(log π_macro,new - log π_macro,old)`，并使用唯一目标 `L_macro = -mean(min(r_macro A, clip(r_macro, 1-ε, 1+ε) A))`。Cache candidate 与 sampled termination sequence 共用一个 Macro joint log probability、一个 Macro ratio、一个 clipped surrogate 和一个 Macro optimizer。Cache ratio 与逐 decision termination ratio 只作为诊断，不构成独立 PPO loss，也不设 Trigger policy。

Macro 与 Micro 使用互相独立的 target KL：`macro_target_kl = 0.10`、`micro_target_kl = 0.02`。Micro update 当 joint approximate KL 超过 `1.5 × micro_target_kl = 0.03` 时 early-stop；不会回滚此前已经完成的 optimizer steps。

正式训练包含 300 个 realization，因此共有 75 次 policy update。Macro Actor 使用按 policy update 计数的 cosine schedule；第一个 update 使用 5e-4，最后一个 update 使用 1e-4。resume 根据 checkpoint 的 `policy_update_count` 继续该 schedule。

Macro actor/critic 与 Micro actor/critic 分离，不共享 Critic、optimizer 或 advantage statistics。checkpoint 分开保存 Macro/Micro policy state 与 optimizer state，并保存 checkpoint/schema/architecture 版本、候选/观测合同、training counters 和 normalization state（当前为两个 `None`）。当前 architecture version 为 27、checkpoint version 为 53；v50/v51/v52 不做 migration，均拒绝加载。

episode metrics 每个 realization 一条，update metrics 每个四-realization group 一条。checkpoint 只在 `completed_episodes` 为 4 的倍数时提交；Macro Critic 的 Last1 retention 保留上一已提交 group 的最后一个 realization。旧 checkpoint 不做 migration。

Adaptive holding diagnostics 报告 duration 1–5 的分布、termination 决策数量/概率/熵、forced max-holding 和 terminal frame 数，并按 elapsed 1–4 汇总 `P(TERMINATE)` 与实际选择率。termination 状态及 sampled action 在 rollout 时冻结，PPO update 不从 environment 重建。

Macro update 另记录 `macro_sampled_termination_decisions_per_frame_mean`、`macro_sampled_termination_decisions_per_frame_max`，以及按实际 duration `D=1…5` 分组的 `macro_joint_log_ratio_abs_mean_by_duration`。它们仅用于诊断，不参与 PPO loss，也不改变 ratio 或 advantage。

实现路径：`trainer.PPOTrainer._update_micro`、`trainer.PPOTrainer._update_macro`、`trainer.PPOTrainer.update_from_episodes`、`trainer.PPOTrainer.checkpoint_payload`、`trainer.PPOTrainer.save_checkpoint` 和 `trainer.PPOTrainer.load_checkpoint`。

## 7. 保持不变的系统模型

以下机制标记为 **UNCHANGED**：

- communication rate
- transmission delay
- execution delay
- vehicle energy
- Cache capacity
- UAV speed
- UAV boundary
- UAV safety distance
- Transfer-Exchange Matching
- KKT resource allocation
- Matching/KKT 任务效用 `Q(t)` 及任务效用内部公式
- 现有车辆/popularity prediction 模块与 prediction artifacts

策略 observation 和候选特征复用当前 prediction artifact 和 observation builder；prediction 网络不在 PPO rollout 中重新训练。正式每 slot 只执行采样到的真实配置，并调用当前 slot environment 中 Matching/KKT 路径。

## 8. 代码映射与状态

### Cache candidate cardinality correction

Cache training samples one candidate from the cardinality-corrected categorical. Candidate index `0` is KEEP and indices `> 0` are UPDATE. Deterministic evaluation first compares KEEP and UPDATE group probability mass, choosing KEEP on a tie; when UPDATE wins, it selects the legal UPDATE candidate with the highest raw score. Both paths use the same corrected categorical probabilities, and training stores the log probability of the sampled candidate.

For the current mask, let `C_valid` be the set of valid candidates, `g(i)` the operation group of candidate `i`, and `N_g = |{j in C_valid : g(j) = g}|`. The logits used by the categorical are:

```text
z_tilde_i = z_i - log(N_g(i)),  i in C_valid
z_tilde_i = -infinity,          i not in C_valid

π_C(i | S) = exp(z_tilde_i) / Σ_{j in C_valid} exp(z_tilde_j)
```

With one KEEP candidate and `K_s` legal UPDATE candidates, subtracting `log(K_s)` from each legal UPDATE logit prevents candidate count alone from inflating UPDATE group mass. This does not change candidate feasibility, the physical system model, or reward.

| 数学/实现合同 | 正式代码位置 | 状态 |
|---|---|---|
| Adaptive holding bounds 与预测 horizon 约束 | `src/common/project_config.py::validate_for_formal_experiment`; `src/ppo_simple/interface.py::load_environment_context` | IMPLEMENTED：正式 bounds 固定为 1–5，并约束 max 不超过 prediction horizon |
| Cache candidate 0=KEEP 与 proposal 复用 | `src/ppo_simple/rollout.py::_cache_candidates` | IMPLEMENTED / VERIFIED：候选 0 运行时断言及源码检查 |
| Cache KEEP/UPDATE candidate cardinality correction 与确定性组选择 | `src/ppo_simple/policy.py::_cardinality_corrected_cache_logits`; `_group_aware_deterministic_candidate` | IMPLEMENTED：训练采样与评价选择共用当前 Cache categorical 概率 |
| 状态相关 termination、Cache persistence、Macro reward | `src/ppo_simple/rollout.py::collect_dual_timescale_episode`; `src/ppo_simple/policy.py::sample_termination` | IMPLEMENTED：逐 slot sampled CONTINUE/TERMINATE、terminal/max 强制边界不计 logprob |
| Cache update mask 与 boundary cost | `src/ppo_simple/rollout.py::cache_update_mask`; `src/decision/slot_environment.py::execute_configured_slot` | IMPLEMENTED / VERIFIED：相同/变化配置的成本合成检查 |
| Cache 下载时延节点级上界归一化 | `src/decision/cache_configuration.py::initialize_cache_cost_normalization`; `src/common/system_model.py::cache_update_cost` | IMPLEMENTED：各在线节点按容量和有效回传速率固定独立参考值；实际新增下载时延逐节点归一化并受逐节点上界检查 |
| UAV 移动能耗节点级上界归一化 | `src/decision/uav_configuration.py::movement_energy_upper_bounds_by_uav_j`; `src/common/system_model.py::uav_movement_cost` | IMPLEMENTED：每架 UAV 由速度、Slot 时长和单位距离能耗系数计算独立上界；只按实际水平位移计费 |
| 归一化成本与 Checkpoint 合同 | `src/ppo_simple/interface.py::load_environment_context`; `src/ppo_simple/trainer.py::PPOTrainer.load_checkpoint` | IMPLEMENTED：运行上下文固定成本参考量；训练恢复仍按配置 raw/semantic hash 严格校验。评价专用的旧 Checkpoint 成本合同兼容门禁在 `evaluator.py` 中未实现 |
| actual-duration SMDP GAE | `src/ppo_simple/rollout.py::compute_macro_smdp_gae` | IMPLEMENTED：SMDP GAE 与 N2 使用真实 adaptive duration |
| Micro ordinary GAE | `src/ppo_simple/rollout.py::compute_gae` | IMPLEMENTED / VERIFIED：代码路径与 terminal 截断检查 |
| group-balanced masked conditional UAV policy | `src/ppo_simple/policy.py`; `src/ppo_simple/rollout.py::_uav_candidates` | IMPLEMENTED / VERIFIED：synthetic sample/evaluate、mask support 与 log probability 检查 |
| STAY/MOVE 组熵软底限与 actor-only 梯度 | `src/ppo_simple/policy.py::evaluate_micro_joint`; `src/ppo_simple/trainer.py::_update_micro` | IMPLEMENTED：从 0.90 概率软上限推导 `H_floor`，低熵时正则，保持 joint PPO |
| Micro/Macro 独立 target KL | `src/ppo_simple/trainer.py::_update_micro`; `src/ppo_simple/trainer.py::_update_macro` | IMPLEMENTED：Micro 0.02、Macro 0.10，Micro joint KL 超过 0.03 时提前停止 |
| 四条 episode-local rollout 合并为一次 PPO update | `src/ppo_simple/trainer.py::PPOTrainer.update_from_episodes` | IMPLEMENTED：源码实现；本轮 synthetic batching 检查 |
| 任务分配、物理成本原始量与 slot reward 聚合边界 | `src/decision/task_allocation.py`; `src/decision/slot_environment.py` | Matching/KKT 与实际时延/能耗公式保持；成本归一化和系统效用合同已更新 |

本次没有从 GitHub 复制或引入代码，也没有增加 dependency。Masked PPO 遵循任务说明给出的 MaskablePPO 设计语义：先 mask 再构造 distribution、rollout 保存 masks、evaluate 复用历史 action support。
