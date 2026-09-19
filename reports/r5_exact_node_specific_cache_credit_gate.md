# R5 exact node-specific Cache credit gate

日期：2026-09-19  
仓库：`C:\Users\tangning\Desktop\code\adaptive_cache_mec`  
checkpoint：`runs/ppo_recovery_r4_wmove_0p50_exactcandidate_dev3_retry_20260919/latest_checkpoint.pt`  
冻结 realization：`runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20001`  
correctness seed：`4401`

## 结论

R5 correctness gate 和 bounded runtime gate 均通过。R5 使用现有 global critic 的 exact environment branch bootstrap，不新增 learned `Q_cache`，也没有执行 PPO 或 optimizer update。

静态审计发现当前生产 Cache action 不是一个二路独立 policy，而是 9 个按 production node mapping 排列的四路 operation head：`NO_CHANGE/ADD/DELETE/SWAP`，其中每个非 `NO_CHANGE` operation 再使用该 operation group 内的 conditional candidate。R5 将这个真实结构严格聚合为：

```text
KEEP   = NO_CHANGE
UPDATE = {ADD, DELETE, SWAP}
```

因此没有假设不存在的 per-node policy。`p(UPDATE)` 是三个合法更新 operation 的总概率；`pi(c | UPDATE)` 是 operation-group 概率与 group 内 candidate 概率的精确乘积后再归一化。

## 生产 node mapping

| head index | node | role |
|---:|---|---|
| 0 | 46 | SV |
| 1 | 10 | SV |
| 2 | 0 | SV |
| 3 | 23 | SV |
| 4 | 59 | SV |
| 5 | RSU_0 | RSU |
| 6 | RSU_1 | RSU |
| 7 | UAV_0 | UAV-cache |
| 8 | UAV_1 | UAV-cache |

当前 `cache_operation_head` 是一个 reshape 后的 `[9, 4]` node-indexed output head；candidate scorer 为 Cache branch 共享模块，Cache encoder 与 UAV encoder 保持 R4 的 decoupled layout。R5 没有引入独立的 learned Cache value network。

## R5 credit definition

对每个 transition 和 target node `m`，其他 8 个 Cache operation、两个 UAV operation、task/vehicle/prediction inputs 固定，仅替换 target node：

```text
Q_keep,m = r_keep,m + gamma * V(next_keep,m)
Q_m,c    = r_m,c    + gamma * V(next_m,c)
Q_update,m = sum_c pi_old(c | UPDATE,s,m) * Q_m,c
B_m = p_old(KEEP) * Q_keep,m + p_old(UPDATE) * Q_update,m
```

Trigger advantage 是 sampled `KEEP/UPDATE` branch 相对 `B_m` 的差异；candidate advantage 是 `Q_m,c - Q_update,m`。所有 branch 均通过正式 `execute_slot`、Matching/KKT、正式 reward 和正式 next observation；branch next-state critic evaluation 使用 batch forward。

R5 loss 的 ratio 范围为 target node 自己的 trigger probability，candidate ratio 范围为 target node 的 `pi(c | UPDATE)`。Cache surrogate 在 valid node-time entry 上求均值，不使用 whole-cache joint ratio，也不使用 Cache trigger/candidate normalization。

## Correctness gate

gate 使用 slots `20, 80, 140, 200` × `RSU_0/RSU_1`，共 8 个 state；未调用 `optimizer.step()`，virtual step 后恢复全部参数。

```text
WITNESS_STATE_COUNT = 8
CORRECTNESS_GATE = PASS

MAX_TRIGGER_POLICY_WEIGHTED_ADV_RESIDUAL = 2.4980018054066022e-14
MAX_CANDIDATE_POLICY_WEIGHTED_ADV_RESIDUAL = 1.3684886557285836e-13

TRIGGER_VIRTUAL_DIRECTION_PASS = 8/8
TRIGGER_VIRTUAL_DIRECTION_TIES = 0/8
CANDIDATE_VIRTUAL_EXPECTED_Q_PASS = 8/8
CANDIDATE_VIRTUAL_EXPECTED_Q_TIES = 0/8

NODE_SPECIFIC_HEAD_GRADIENT_ISOLATION = PASS
NODE_RATIO_ISOLATION = PASS
BASELINE_BRANCH_EQUIVALENCE = PASS
```

Direct operation-head gradient isolation had target-row gradient on every witness and zero non-target operation-head row gradient. Artificially changing a non-target node's new log-probability changed the synthetic whole-cache ratio but left the target-node ratio unchanged (`max absolute target-ratio change = 0`).

For the 8 witness states, all 8 policy-conditional `Q_update` values were below `Q_keep`; the sampled operation was UPDATE in 7 states and KEEP in 1 state. The 8-state exact branch set contained 212 legal UPDATE candidates: 14 beneficial, 198 harmful, and 111 with a nonzero task/offloading change relative to that node's KEEP branch. This is a bounded mechanism check, not a training result.

| slot | target | sampled op index | `p(KEEP)` | `p(UPDATE)` | legal UPDATE candidates | `Q_keep` | `Q_update` | sampled trigger A |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 20 | RSU_0 | 3 | 0.253305 | 0.746695 | 29 | -286.920 | -290.980 | -1.030 |
| 20 | RSU_1 | 1 | 0.222907 | 0.777093 | 27 | -286.660 | -295.070 | -1.880 |
| 80 | RSU_0 | 1 | 0.210176 | 0.789824 | 20 | -285.300 | -286.470 | -0.250 |
| 80 | RSU_1 | 2 | 0.256860 | 0.743140 | 19 | -285.390 | -287.400 | -0.510 |
| 140 | RSU_0 | 1 | 0.222966 | 0.777034 | 26 | -298.010 | -299.520 | -0.340 |
| 140 | RSU_1 | 3 | 0.248332 | 0.751668 | 31 | -279.610 | -286.090 | -1.610 |
| 200 | RSU_0 | 2 | 0.210386 | 0.789614 | 26 | -275.930 | -277.500 | -0.330 |
| 200 | RSU_1 | 0 | 0.256694 | 0.743306 | 34 | -275.820 | -277.340 | +1.130 |

`Q` values include the checkpoint critic next-state bootstrap. The branch comparison showed that RSU Cache changes can alter formal execution-node matching, affected tasks, task utility, cache cost, and system utility; therefore the structural path to offloading is valid:

```text
RSU_STRUCTURAL_PATH_VALID = YES
RSU_CACHE_AFFECTS_OFFLOADING = YES
BENEFICIAL_COUNTERFACTUAL_COUNT = 14
HARMFUL_COUNTERFACTUAL_COUNT = 198
```

## Bounded runtime gate

The full production R5 helper was run on one 300-slot `episode_20001` rollout. It enumerated every legal candidate for all 9 Cache nodes and executed one KEEP branch plus every legal UPDATE candidate. No pruning, top-k, surrogate environment, or scalar-per-candidate critic evaluation was used.

```text
TOTAL_CACHE_CF_ENV_EXECUTIONS = 38912
MEAN_CACHE_CF_EXECUTIONS_PER_TRANSITION = 129.70666666666668
CACHE_CF_COLLECTION_RUNTIME_SECONDS = 591.0465970999794

UAV_CF_ENV_EXECUTIONS = 48600
TOTAL_R5_CF_EXECUTIONS_PER_ROLLOUT = 87512
```

The full-run legal UPDATE candidate statistics were:

| node | mean | min | max | total |
|---|---:|---:|---:|---:|
| 46 | 9.1267 | 3 | 13 | 2738 |
| 10 | 9.1367 | 3 | 13 | 2741 |
| 0 | 9.2433 | 3 | 13 | 2773 |
| 23 | 8.3200 | 3 | 13 | 2496 |
| 59 | 8.7367 | 3 | 13 | 2621 |
| RSU_0 | 21.1700 | 6 | 34 | 6351 |
| RSU_1 | 20.7367 | 6 | 34 | 6221 |
| UAV_0 | 16.8900 | 5 | 30 | 5067 |
| UAV_1 | 17.3467 | 5 | 30 | 5204 |

The production helper's full-run candidate weighted residual was `1.9984014443252818e-13`.

## R5 schema and source changes

```text
R5_SCHEMA = adaptive_joint_cache_replica_aware_uav_operation_recovery_r5

CACHE_TRIGGER_ADV_SOURCE = EXACT_NODE_COUNTERFACTUAL_Q
CACHE_CANDIDATE_ADV_SOURCE = EXACT_NODE_CONDITIONAL_COUNTERFACTUAL_Q
CACHE_TRIGGER_RATIO_GRANULARITY = NODE_ONLY
CACHE_CANDIDATE_RATIO_GRANULARITY = NODE_CONDITIONAL
CACHE_TRIGGER_ADV_NORMALIZATION = NONE
CACHE_CANDIDATE_ADV_NORMALIZATION = NONE
CACHE_SURROGATE_REDUCTION = MEAN_OVER_VALID_NODE_TIME

UAV_OPERATION_ADV_SOURCE = COUNTERFACTUAL_ONE_STEP_TD
UAV_CANDIDATE_ADV_SOURCE = EXACT_CONDITIONAL_COUNTERFACTUAL_Q
OP_ADV_NORMALIZATION = FIXED_REFERENCE_SCALE
OP_ADV_REFERENCE_SCALE = 0.02
```

R5-only source changes add the schema class, config validation/manifest declarations, the exact node-specific Cache replay helper, and the trainer-side node-only trigger plus node-conditional candidate surrogate. R4's network layout, UAV credit path, reward, movement cost, Matching/KKT, and system model remain unchanged. The R4 checkpoint was loaded successfully for the gate; no R5 training checkpoint was created.

## Final audit fields

```text
TRAINING_EXECUTED = NO
OPTIMIZER_STEP_EXECUTED = NO

LEARNED_Q_CACHE_ADDED = NO
REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
MATCHING_KKT_CHANGED = NO
UAV_ALGORITHM_CHANGED = NO

EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO

SOURCE_MODIFIED = YES   # R5 credit/schema only; no system/reward/UAV change
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
```
