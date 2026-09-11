# PPO / Direct-CF final training-protocol check

审计基线：`62f515edc12860d0e49ec77a19c418576ce376b8`
审计对象：当前 `main` 工作区
范围：只检查 PPO rollout、observation/action、actual reward、GAE/return、PPO/Direct-CF update 链路。没有重新审计系统模型、Cloud、Matching/KKT、Cache/UAV optimizer 或参数结论。

## 1. 当前正式训练入口

| 项目 | 当前值 |
|---|---|
| method | `proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift` |
| credit assignment | `hierarchical_gate_direct_cf` |
| actor architecture | `v3_full_shared_actor_frozen_policy_batch_v1` |
| `cf_samples_per_episode` | `8` |
| `gate_cf_samples_per_episode` | `8`（代码固定） |
| `cf_batch_episodes` | `4` |
| `cf_parallel_workers` | `8` |
| `reference_probability_drift_cap` | `0.005` |
| `gamma` / `gae_lambda` | `1.0` / `0.95` |
| `clip_epsilon` | `0.2` |
| `learning_rate` | `3e-4` |
| `update_epochs` / `num_minibatches` | `1` / `4`（legacy scalar PPO；当前 Gate/WHAT update 各为一轮） |
| `entropy_coefficient` | `0.0` |
| train origins | `0..239` |
| held-out test origins | `240..299` |

当前正式 Gate/WHAT update 不使用 learned Q critic；Gate 和 WHAT 使用不重叠的独立 Adam optimizer。

## 2. Observation contract

代码入口是 `ConfigurationObservationSpec` 和 `build_configuration_observation()`。当前真实维度为 `154`：

| slice | 内容 | 维度 | 来源 |
|---|---|---:|---|
| `0:90` | previous cache state `C(t-1)` | `9×10` | 上一 slot 的 dynamic cache state |
| `90:94` | previous UAV positions `Q^U(t-1)` | `2×2` | 上一 slot state，按 flight boundary 归一化 |
| `94:144` | predicted vehicle positions `q_hat^V(t)` | `25×2` | 当前 origin 对齐的预测 artifact |
| `144:154` | predicted service popularity `rho_hat(t)` | `10` | 当前 origin 对齐的预测 artifact |

确认项：

- observation 不包含 slot `t` 的 actual tasks、actual vehicle positions、actual Matching/KKT、`Q` 或 `G`。
- prediction loader 要求 prediction 的 target slot/time 与当前 origin 一致；normal rollout 的当前 observation 只从 train episode 的 aligned steps 构造。
- formal collector 的 `_formal_aligned_steps()` 只接受 protocol 定义的 `0..239`，并检查不得越过 train/test 边界；held-out test 另走 `240..299` 的只读路径。
- rollout 和 inference 使用同一个 observation builder；没有发现旧的 `164/184` 维路径进入当前 active collector。

判定：**PASS**。

## 3. Policy request 与 environment actual action

当前 PPO request 是长度 `11` 的 binary trigger vector：

```text
9 cache requests + 2 UAV movement requests
```

前 9 位对应 dynamic SV/RSU/UAV cache nodes，后 2 位对应 UAV movement trigger。它不直接选择 Local、Edge 或 Cloud target。

`collect_on_policy_rollout()` 将 sampled policy request 保存为 `RolloutTransition.action_vector`，并把对应 cache/UAV trigger 传给 `execute_slot()`。actual action 不被替换写回 rollout action；actual cache/UAV change 由 `SlotExecutionResult` 的最终 state 差异、`configuration_summary` 和 actual cost 体现。

合同检查包括：

- request `0` 的 cache row/UAV position 必须不变；
- request `1` 只允许进入 configuration search，不保证一定产生实际变化；
- Gate `g=0` 映射为严格全零的 11-D environment action；Gate `g=1` 才暴露 WHAT request；
- rollout 保存的 old joint log-prob 等于 Gate old log-prob 加 WHAT chosen log-prob 之和。

判定：**PASS**。

## 4. Actual reward boundary

rollout 每个可行 slot 的 reward 直接取：

```text
G(t) = result.system_utility
     = Q_actual(t) - C_ca_actual(t) - C_q_actual(t)
```

`_check_reward_invariant()` 在 transition 写入前检查该等式。预测 configuration utility / `J` 没有被重新加入 actual reward，policy request 本身也没有单独收费；cost 只来自实际 cache state change 和实际 UAV movement。

没有发现 reward sign flip、预测 utility 与 actual utility 混加或 cache/movement cost double-count。当前代码没有在 active hierarchical Direct-CF actor update 前做 reward normalization；legacy generic scalar PPO 路径同样没有隐式 reward normalization。

判定：**PASS**。

## 5. Rollout buffer、state propagation 与 boundary

每个 transition 保存 observation、sampled request、old joint/per-action log-prob、old value、actual `G`、actual `Q`、cache cost、movement cost、next cache state、next UAV positions，以及 hierarchical Gate 字段。

下一 slot 的 observation 使用本 transition 的 next cache/UAV state。collector 在相邻 transition 间做 exact state propagation 检查，并在 action `0` 时检查相应 state 不变。

formal episode 的最后 train origin `239` 是 true terminal：

- formal collector 拒绝 truncated batch；
- true terminal 时 `bootstrap_value=0`；
- test origins 不进入 formal optimizer update；
- test evaluation 使用独立的 `240..299` range 和只读执行路径。

判定：**PASS**。

## 6. GAE / return mapping

代码实现的是标准 reverse GAE：

```text
delta_t = r_t + gamma * (1-done_t) * V_{t+1} - V_t
A_t     = delta_t + gamma * gae_lambda * (1-done_t) * A_{t+1}
return_t = A_t + V_t
```

当前配置为 `gamma=1.0`、`gae_lambda=0.95`。`done` 会切断 bootstrap 和后续 advantage；true terminal 的 bootstrap 为零，截断 rollout 才使用额外 next observation 的 value。Monte-Carlo return 同样在 done 处重置 suffix。

需要区分两条语义：

- generic/legacy scalar PPO 的 `compute_ppo_loss()` 使用 rollout `advantages` 和 `returns`；
- 当前 active hierarchical Gate/WHAT Direct-CF actor 不把 scalar GAE/value loss 当作 actor credit，而使用 frozen same-state branch return difference。`gamma/gae_lambda` 仍保留给 rollout return/reporting 和兼容路径。

没有修改 `gamma` 或 GAE 公式。

判定：**PASS**。

## 7. PPO loss 与 optimizer update

### Generic scalar PPO compatibility path

`compute_ppo_loss()` 使用：

```text
ratio = exp(new_joint_log_prob - old_joint_log_prob)
surrogate = min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)
policy_loss = -mean(surrogate)
value_loss = 0.5 * mean((V - return)^2)
total = policy_loss + cV*value_loss - cH*entropy
```

formal legacy executor 明确冻结 old log-prob、old value、advantage、return 和 action；每个 epoch 无放回 shuffle，执行 `zero_grad -> loss -> backward -> finite check -> clip_grad_norm -> optimizer.step`。当前配置为 `update_epochs=1`、`num_minibatches=4`，没有额外的 advantage normalization、value clipping 或 KL early stop。

### Current hierarchical Gate/WHAT path

- Gate loss 使用 `A_gate=(g-p_old_gate)*Delta_gate`，只再乘 likelihood ratio，不重复乘 `(g-p)`；不加入 value/entropy loss。
- WHAT loss 使用 target-only `A_CF=(a-p_old)*Delta_G`，只评价被选 trigger 的 new/old probability ratio；不加入 scalar value loss 或 entropy loss。
- Gate optimizer 先执行一次，WHAT/ref-drift optimizer 后执行一次；两个参数集合不重叠。
- WHAT 的 value head 和规定的非 actor block 在 active sparse Direct-CF update 中保持冻结。
- Adam 后只对一次已完成的参数位移做 reference drift backtracking，不重新推进 Adam state；reference cap 为 `0.005`。

判定：**PASS（静态合同）**。

## 8. Direct-CF reference mechanism

### Reference source and timing

Reference records 来自 normal rollout 期间 sampled Gate=`CONFIG` 的 free WHAT decisions，记录 observation、action prefix/vector、mask、`p_old`、collection policy version 和 baseline RNG snapshot。它们不是从 actual reward 反推的 action，也不是 test data。

### Branch semantics

- WHAT branch 从 selected decision 的 pre-action state 重放到 true terminal，target trigger 翻转，suffix 用 frozen old policy。
- Gate branch 使用同一 pre-action state 的 paired `CONFIG` / `NULL` Joint-Null branch；NULL action 严格全零。
- branch 前后恢复 baseline RNG，CONFIG/NULL 共用对齐的 policy/environment continuation randomness。
- branch return 使用实际执行得到的 `G=Q-cache-UAV`，并检查：

```text
Delta_G = Delta_Q - Delta_Cache - Delta_UAV
A_CF    = (a - p_old) * Delta_G
Delta_gate = Return_CONFIG - Return_NULL
A_gate    = (g - p_old_gate) * Delta_gate
```

### Sampling and update

formal config 保持 `cf_samples_per_episode=8`、`gate_cf_samples_per_episode=8`、`cf_batch_episodes=4`、`trigger_stratified` 和 `cf_parallel_workers=8`。CF worker 只跨独立 sample 并行；每个 CF suffix 内 slot 仍串行。若 worker 条件不满足，代码回退 `mode=serial, workers=1`。

### Reference drift

当前 ref-drift update：

1. 在 collection policy 上检查未选 reference 的 stored `p_old` 重构误差；
2. 执行一次 WHAT Adam candidate step；
3. 在同一 old/raw 参数位移上做 backtracking/bisection；
4. 约束未选 reference 的 masked conditional probability drift `<=0.005 + dtype tolerance`；
5. 保留一次 Adam state 更新，不重复调用 optimizer step。

没有发现预测 `J`、预测 configuration request 或 test reward 混入 Direct-CF `Delta_G`。

判定：**PASS（静态合同）**。

## 9. Minimal dynamic verification

没有运行完整 240-slot rollout、4-episode training、Batch4、test evaluation 或 parallel CF training。

### 9.1 Stable real train-origin rollout

使用现有真实 origin `0`，只收集 1 个 normal train slot，Gate 强制 NULL 以保持 bounded no-op fixture；没有启动 CF worker、GPU 或正式训练循环。

结果：

| 项目 | 结果 |
|---|---:|
| rollout steps | `1` |
| observation shape | `(1, 154)` |
| request shape | `(11,)` |
| request | `[0,0,0,0,0,0,0,0,0,0,0]` |
| actual state transition | no-op；cache/UAV 状态保持 |
| actual reward `G` | `13.4968865947` |
| old joint log-prob | `-8.40492093563` |
| old value | `-0.00749147310853` |
| advantage | `13.4969520569` |
| return | `13.4894609451` |
| native crash | 未发生 |

该 bounded collector 因只包含 origin `0` 而不是正式 terminal，返回 `truncated=True`；这只用于短 fixture，formal collector 仍要求最后 train origin `239` true terminal。

### 9.2 One in-memory generic PPO update smoke check

在同一个真实 train-origin normal rollout 上执行一次内存中的 generic hierarchical PPO loss/update，用于确认 observation/action/reward/buffer/gradient/optimizer 链路；没有写 checkpoint/history，也没有改变工作区算法。

| 项目 | 结果 |
|---|---:|
| optimizer steps | `1` |
| pre-update policy loss | `-13.4969520569` |
| pre-update value loss | `91.0838546753` |
| entropy | `7.61711025238` |
| ratio min/mean/max | `1 / 1 / 1` |
| clip fraction | `0` |
| gradient finite | `True` |
| pre-clipping gradient norm | `182.765396118` |
| parameter delta norm | `0.0647043173` |
| post-update ratio mean | `1.31503939629` |

这不是把 generic scalar loss 当作当前 formal hierarchical Direct-CF credit；当前正式训练仍按第 7、8 节的 Gate/WHAT Direct-CF dispatcher 执行。

### 9.3 Direct-CF bounded attempt

为避免再次触发 nested multiprocessing，临时 fixture 将内存中的 prediction-batch worker 和 CF worker 都设为 `1`，不改 `system_config.toml`。使用 1 个 train origin、1 个 in-memory WHAT reservoir capacity 和 1 个 Gate reservoir capacity，尝试进入真实 serial WHAT/Gate branch。

normal baseline rollout 已完成；进入 counterfactual branch 后约 90 秒没有返回。该过程被停止，没有 native crash，也没有重复运行。由于 branch 未返回：

- 没有声称 Direct-CF sample 已完成；
- 没有声称 `reference probability before/after` 或 `max drift` 数值；
- 没有声称 `0.005` cap 的动态满足结果；
- 没有把这次 runtime overrun 当成算法语义失败。

这说明当前 bounded real Direct-CF branch 仍具有明显运行时间风险，但它不是 observation、reward、buffer、GAE、loss 公式或 train/test isolation 的 formal contract blocker。本轮按要求不做优化、不扩大运行窗口。

## 10. Blocker classification

| 类别 | 结果 |
|---|---|
| CRITICAL formal training blocker | 无 |
| NON_BLOCKING verification/runtime observation | 1-slot serial Direct-CF branch 在约 90 秒内未返回；未形成可报告的 dynamic CF drift 数值 |
| 已修改算法/参数 | 无 |
| test data 进入 train optimizer | 未发现 |
| 12.pdf | 未搜索、未创建、未修改 |

## 11. Final verdict

基于当前代码合同、formal train/test boundary、真实 train-origin normal rollout、一次最小内存 PPO update 以及已有 Direct-CF/RefDrift 实现检查：

**PPO_PROTOCOL_READY = YES**

该结论表示训练协议的 state/action/reward/return/loss/update/data-boundary 合同已闭合，不表示本轮已经完成正式 240-slot 或 4-episode runtime，也不表示 bounded Direct-CF branch 的性能问题已经解决。Direct-CF branch 的运行时间风险保留为后续单独性能任务。

## 12. 本轮文件与清理

- 新增正式报告：`reports/ppo_direct_cf_final_protocol_check.md`
- 未修改 `src/`、`system_config.toml`、checkpoint、history schema 或已有算法实现。
- 临时 fixture 脚本已删除；没有留下 CSV、JSON、benchmark 或 profiling 输出文件。
- 用户文件 `打包并上传.bat` 未修改、未删除、未移动、未加入 commit。
