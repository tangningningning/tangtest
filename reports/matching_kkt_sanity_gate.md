# Matching / KKT representative sanity gate

TRAINING_EXECUTED = NO

本审计只读取现有 train-only episode pool
`environment_data/proposed_joint_training`，使用正式 1e-4 配置的只读副本
`runs/proposed_actor1e4_uavfix_300_epoch/run_config.toml`。使用的 episode seeds
为 20001–20005；没有使用 dev/final/test seed，没有调用 PPO trainer，没有创建
training run，也没有读取或写入 checkpoint、metrics 或正式 run 文件。

## Representative sample

每个 episode 选择低任务数、中位任务数、高任务数和 slot 150，共 20 个
代表性 state。括号内为 `slot / arrived task count`：

| train episode | low | middle | high | extra |
|---:|---:|---:|---:|---:|
| 20001 | 169 / 7 | 1 / 14 | 72 / 19 | 150 / 14 |
| 20002 | 217 / 8 | 10 / 14 | 118 / 19 | 150 / 12 |
| 20003 | 23 / 7 | 2 / 14 | 119 / 20 | 150 / 15 |
| 20004 | 76 / 7 | 19 / 14 | 8 / 19 | 150 / 12 |
| 20005 | 21 / 7 | 17 / 14 | 296 / 18 | 150 / 16 |

总计 24 个 Matching calls：

- 6 个 actual configuration evaluation，使用现有 `execute_slot` 路径，覆盖
  cache、UAV、cache+UAV trigger；
- 14 个 direct Matching state calls，复用上述 actual configuration 产生的合法
  cache state / UAV position；
- 4 个 direct policy-target counterfactual configuration calls，使用已有合法
  UAV candidate 生成 policy-target override。

因此样本同时覆盖低/中/高任务数、不同 cache state、不同 UAV position、actual
configuration evaluation 和少量 counterfactual configuration；没有进行全量
episode 或 Exact-Q sweep。

## Matching termination results

每个 call 都记录了 `stop_reason`、`updates`、`transfer_exchange_stable`、
`MAX_UPDATES_REACHED`、final feasibility 和 `stable_check_passed`。当前配置的
`matching.max_updates` 保持为 100，Matching 决策逻辑未修改。

| metric | result |
|---|---:|
| total Matching calls | 24 |
| final feasible calls | 24 / 24 |
| `converged_by_no_improvement` | 24 / 24 (100%) |
| `stopped_by_max_updates` | 0 / 24 (0%) |
| `transfer_exchange_stable = True` | 24 / 24 |
| `stable_check_passed = True` | 24 / 24 |
| `MAX_UPDATES_REACHED = True` | 0 / 24 |
| update count mean | 7.5 |
| update count P95 | 12 |
| update count max | 14 |

所有 call 的 stop reason 都是 `converged_by_no_improvement`；没有把
`NO_IMPROVING_TRANSFER_EXCHANGE` 与 `MAX_UPDATES_REACHED` 混为同一 reason。
该结果关闭的是本代表性样本 gate，不是对所有未来可能状态的数学穷尽证明。

MATCHING_MAX_UPDATE_HIT_PROBLEM = NO

MATCHING_STABILITY_GATE = PASS

## KKT results

诊断复用现有 `solve_kkt_fraction_allocation` 及其现有内部 multiplier；没有
重写或并行实现第二套 solver。总计观察到 3654 次 CPU/BW fraction solve。

| metric | result |
|---|---:|
| solve count | 3654 |
| converged by tolerance | 3654 / 3654 (100%) |
| max bisection iteration hit | 0 / 3654 (0%) |
| finite status | 3654 / 3654 (100%) |
| bisection iteration mean / P95 / max | 29.7915 / 40 / 44 |
| resource residual mean / max | `3.4634e-13` / `9.9942e-13` |
| CPU fraction sum mean / P95 / max | `1.000000000000032` / `1.000000000000671` / `1.000000000000953` |
| bandwidth fraction sum mean / P95 / max | `0.999999999999950` / `1.000000000000530` / `1.000000000000948` |
| allocation non-negativity | PASS |
| allocation finite status | PASS |
| stationarity residual mean / max | `9.7263e-16` / `5.6843e-14` |
| complementary-slackness residual mean / max | `2.1628e-12` / `1.6289e-10` |

CPU and bandwidth fraction sums were checked from the returned per-node
allocations. Stationarity and complementary-slackness residuals were computed
directly from the returned allocation and multiplier; no optimizer rewrite was
needed. The active KKT resource residual stayed below the configured
`1e-12` bisection tolerance up to floating-point summation error.

KKT_MAX_ITERATION_PROBLEM = NO

KKT_FEASIBILITY_GATE = PASS

KKT_CONVERGENCE_GATE = PASS

KKT_CORRECTNESS_GATE = PASS

该 PASS 仅表示 representative gate passed，不是对所有可能状态的数学证明。

## Scope and limitations

The PASS results establish a representative train-only sanity gate across the
selected task-count, cache-state, UAV-position, actual-configuration and
counterfactual cases. They do not prove Matching stability or KKT correctness
for every possible state, task set, floating-point input, or future run.

No PPO, LR, reward, candidate policy, Matching algorithm, KKT solver logic, or
formal hyperparameter was changed. The temporary audit script and its compiled
cache were deleted after the sweep.
