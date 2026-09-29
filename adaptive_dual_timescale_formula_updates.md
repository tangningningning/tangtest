# Adaptive 双时间尺度 Cache/UAV PPO 数学合同

本文件是正式算法和实现的长期合同。公式描述策略决策、rollout 与 PPO target；Cache loading、通信、delay、energy 和 utility 的物理计算仍由现有模块负责。本版不迁移训练框架，也没有直接复制第三方代码。

## 1. 原定义与修改原因

### 原正式控制定义

旧主线在每个 slot 分别采样 Cache Trigger 与 Cache Candidate、UAV Operation 与 UAV Candidate，再将触发和候选组合为 slot 配置。策略对各控制 head 分别记录概率/credit；候选评估链还可接入 Exact-Q 反事实 teacher 与 critic MC warmup。

### 新正式定义

新的控制器使用 Macro Cache SMDP 和每 slot Micro UAV PPO。Macro 一次选择 Cache 配置与持续时长；frame 内 Cache 固定。Micro 对 UAV0、UAV1 使用顺序条件式 masked categorical。完整 episode 收集结束后分别更新两套 PPO。

### 修改原因

- Cache 配置是跨 slot 持续的控制量，显式 duration 使决策间隔和折扣与真实经过的 slot 数一致。
- 当前配置纳入 Macro candidate 集合后，KEEP 与 UPDATE 是一个 Cache 配置动作空间，不再需要 Trigger/Candidate 双层策略。
- UAV 位置候选把 STAY 和 MOVE 放在同一 masked candidate head 中；先选择 UAV0，再依据其最终位置重建 UAV1 的安全候选集。
- 分离 Macro/Micro buffer、Critic、optimizer 和 advantage normalization，避免不同决策时间尺度共用 return 统计。
- 正式训练不依赖 Exact-Q teacher 或强制 MC warmup；Matching、KKT 与真实系统 utility 继续提供训练回报。

## 2. Macro 决策边界和 Cache persistence

令 episode 的 slot 数为 `T`，Macro frame 起点为 `τ_k`。初值与边界递推为：

```text
τ_0 = 0
τ_{k+1} = τ_k + L_k^actual
L_k^actual = min(L_k^requested, T - τ_k)
```

因此尾帧使用剩余 slot 数。例如 `T=300, τ_k=290, L_requested=20` 时，`L_actual=10`。Macro SMDP return、discount 和 GAE trace 全部使用 10。

Cache 在同一 frame 内保持不变：

```text
c_{m,f}(t) = c_{m,f}^{(k)},  τ_k <= t < τ_{k+1}
```

Macro action 是联合动作：

```text
A_k^C = (C_k, L_k)
π(A_k^C | S_k^C) = π_C(C_k | S_k^C) · π_L(L_k | S_k^C, C_k)
log π(A_k^C | S_k^C) = log π_C + log π_L
ratio_k = exp(new_macro_log_prob - old_macro_log_prob)
```

Cache candidate 0 是当前实际配置，代表 KEEP。实际 Cache 更新按节点定义：

```text
z_m^ca(k) = I[存在 f: c_{m,f}^{(k)} != c_{m,f}^{(k-1)}]
```

重新进行 Macro 决策不等于发生 Cache 更新。若 `C_k = C_{k-1}`，则 `z_m^ca(k)=0`，loaded amount 为 0，且不产生 Cache reconfiguration cost；若配置不同，才在 boundary slot 执行实际更新并调用现有 Cache loading/cost helper。更新 cost 只在新配置生效的 boundary slot 计算一次。

现有 Cache loading / cost 数学公式保持原样；本版只改变其调用时机。实现路径：`rollout._cache_candidates`、`rollout.collect_dual_timescale_episode`、`rollout.cache_update_mask`、`slot_environment.execute_configured_slot` 和 `slot_environment._cache_cost`。

## 3. Macro frame reward 与 SMDP target

Macro frame reward 保留 slot reward 的折扣和，不做按 duration 平均：

```text
R_k^C = Σ_{j=0}^{L_k^actual-1} γ^j G(τ_k + j)
```

SMDP TD error 使用经过的实际 slot 数作为指数：

```text
δ_k^C = R_k^C + γ^(L_k^actual) V_C(S_{k+1}^C) - V_C(S_k^C)
```

duration-aware GAE 为：

```text
A_k^C = δ_k^C + (γ λ)^(L_k^actual) A_{k+1}^C
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

实现路径：`policy._masked_categorical`、`policy._group_balance_candidate_logits`、`policy.DualTimescaleActorCritic.sample_micro_head`、`policy.evaluate_micro_joint`、`rollout._uav_candidates` 和 `rollout.collect_dual_timescale_episode`。

## 5. 独立 rollout 与 GAE

每 slot 一条 Micro transition，保存状态、两 UAV action index、joint old log probability、Micro value、真实 `G(t)`、next state、done、UAV0 mask、基于 UAV0 已选位置条件化的 UAV1 mask、候选描述符及 active Cache configuration identifier。

每个完整 Cache frame 一条 Macro transition，保存 Macro state、Cache/duration action index、requested/actual duration、Macro old joint log probability/value、frame reward、next state、done、Cache/duration masks 和候选描述符。两个 buffer 永不混为同一 transition list。

Micro 使用 ordinary GAE：

```text
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
A_t = δ_t + γ λ A_{t+1}
```

terminal 的 bootstrap 与 trace 由 `done` 截断。Micro 和 Macro advantages 分别标准化：

```text
A_norm = (A - mean(A)) / (std(A) + ε)
```

Macro frame reward 不除以 `L_k`；两尺度不共享 reward/advantage/value normalization state。本版没有启用 PopArt。

实现路径：`rollout.MicroRolloutBuffer`、`rollout.MacroRolloutBuffer`、`rollout.compute_gae`、`rollout.compute_macro_smdp_gae` 和 `trainer.PPOTrainer.update_from_episode`。

## 6. PPO updates、优化器与 checkpoint

整个 episode rollout 收集期间 Macro 和 Micro policy 都不更新。episode 结束后先执行 Micro PPO phase，再执行 Macro PPO phase。Micro 使用配置 minibatches；Macro 对本 episode 全部 Macro transitions 做 full-batch PPO。Cache candidate 与 duration 使用一个 Macro joint log probability、一个 Macro ratio 和一个 Macro optimizer。

Macro actor/critic 与 Micro actor/critic 分离，不共享 Critic、optimizer 或 advantage statistics。checkpoint 分开保存 Macro/Micro policy state 与 optimizer state，并保存 checkpoint/schema/architecture 版本、候选/观测合同、training counters 和 normalization state（当前为两个 `None`）。旧结构不做 migration。

实现路径：`trainer.PPOTrainer._update_micro`、`trainer.PPOTrainer._update_macro`、`trainer.PPOTrainer.checkpoint_payload`、`trainer.PPOTrainer.save_checkpoint` 和 `trainer.PPOTrainer.load_checkpoint`。

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
- slot utility `G(t)`
- 现有车辆/popularity prediction 模块与 prediction artifacts

策略 observation 和候选特征复用当前 prediction artifact 和 observation builder；prediction 网络不在 PPO rollout 中重新训练。正式每 slot 只执行采样到的真实配置，并调用当前 slot environment 中 Matching/KKT 路径。不调用 Full-information Exact-Q teacher；Exact-Q 诊断模块仍保留为独立未来能力。

## 8. 代码映射与状态

| 数学/实现合同 | 正式代码位置 | 状态 |
|---|---|---|
| Macro duration candidates 校验与预测 horizon 约束 | `src/common/project_config.py::validate_for_formal_experiment`; `src/ppo_simple/interface.py::macro_duration_candidates_from_project` | IMPLEMENTED / VERIFIED：正式 TOML parse 与校验通过 |
| Cache candidate 0=KEEP 与 proposal 复用 | `src/ppo_simple/rollout.py::_cache_candidates` | IMPLEMENTED / VERIFIED：候选 0 运行时断言及源码检查 |
| duration tail clip、Cache persistence、Macro reward | `src/ppo_simple/rollout.py::actual_duration_for_frame`; `collect_dual_timescale_episode` | IMPLEMENTED / VERIFIED：尾帧合成检查及源码检查 |
| Cache update mask 与 boundary cost | `src/ppo_simple/rollout.py::cache_update_mask`; `src/decision/slot_environment.py::execute_configured_slot` | IMPLEMENTED / VERIFIED：相同/变化配置的成本合成检查 |
| duration-aware SMDP GAE | `src/ppo_simple/rollout.py::compute_macro_smdp_gae` | IMPLEMENTED / VERIFIED：手算数组与函数结果一致 |
| Micro ordinary GAE | `src/ppo_simple/rollout.py::compute_gae` | IMPLEMENTED / VERIFIED：代码路径与 terminal 截断检查 |
| group-balanced masked conditional UAV policy | `src/ppo_simple/policy.py`; `src/ppo_simple/rollout.py::_uav_candidates` | IMPLEMENTED / VERIFIED：synthetic sample/evaluate、mask support 与 log probability 检查 |
| 完整 episode 结束后分离 PPO updates | `src/ppo_simple/trainer.py::PPOTrainer.update_from_episode` | IMPLEMENTED / VERIFIED：compile/import 与调用路径检查；未运行训练 |
| 物理系统与 allocation 数学不变 | `src/decision/task_allocation.py`; `src/decision/slot_environment.py` | UNCHANGED / VERIFIED：task allocation 无 diff，slot environment 仅新增最终配置入口 |

本次没有从 GitHub 复制或引入代码，也没有增加 dependency。Masked PPO 遵循任务说明给出的 MaskablePPO 设计语义：先 mask 再构造 distribution、rollout 保存 masks、evaluate 复用历史 action support。
