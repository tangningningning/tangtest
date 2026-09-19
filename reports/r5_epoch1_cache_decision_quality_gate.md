# R5 epoch1 Cache decision-quality gate

日期：2026-09-19  
仓库：`C:\Users\tangning\Desktop\code\adaptive_cache_mec`  
任务：`R5_EPOCH1_CACHE_DECISION_QUALITY_GATE`

## 结论

R5 epoch1 checkpoint 有效，且 node-specific Cache credit 在这组固定 witness 上表现出**朝 exact counterfactual preference 移动的方向，但尚未完全对齐**：

- `p(KEEP)` 均值由 fresh/pretraining gate 的 `0.23520325` 上升到 `0.31285300`；
- `p(UPDATE)` 均值由 `0.76479675` 降到 `0.68714700`；
- policy trigger correctness 由 pretraining 的 `0/8` 增加到 `2/8`；
- 当前 exact preference 为 `KEEP 6/8`、`UPDATE 2/8`，所以仍有 6 个 KEEP-preferred witness 被较高的 UPDATE 概率覆盖。

因此本 gate 的结论是：

```text
R5_EPOCH1_LEARNING_DIRECTION = SUPPORTED
PRIMARY_DIAGNOSIS = R5_NODE_SPECIFIC_CACHE_CREDIT_DIRECTION_SUPPORTED_BUT_PARTIAL
```

这里的 `SUPPORTED` 只表示第一 epoch 的概率移动方向和 correctness count 改善得到支持，不表示 policy 已经收敛，也不授权继续 epoch2/epoch3。

## Checkpoint 与 frozen data

使用的 checkpoint：

```text
runs/ppo_recovery_r5_wmove_0p50_nodecache_dev3_20260919/latest_checkpoint.pt
```

metadata 核验：

```text
completed_episodes = 1
completed_training_episodes = 1
next_pool_index = 1
current_train_origin = 300
run_summary.completed_epochs = 1
DEV3_STATUS = INTERRUPTED_FOR_RUNTIME_OPTIMIZATION
partial_episode_excluded_from_formal_results = true
```

因此使用的是完整 epoch1 checkpoint；没有使用 partial Episode 2 内存状态，也没有运行 Episode 2/3。

frozen realization：

```text
runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20001
```

报告 witness 严格为 `20, 80, 140, 200 × RSU_0, RSU_1`，共 8 个 state。为确保 slot 200 也使用正式 next-state critic bootstrap，origin 201 仅作为隐藏 bootstrap-support transition，未计入 witness 统计。

## Exact definition

沿用当前 optimized exact-safe helper。每个 branch 都经过正式 `execute_slot`、Matching/KKT、正式 reward、正式 next state，并使用当前 epoch1 critic 对 next state 做 bootstrap：

```text
Q_keep = r_keep + gamma * V(next_keep)
Q_c    = r_c    + gamma * V(next_c)
Q_update = sum_c pi(c | UPDATE) * Q_c
DeltaQ = Q_update - Q_keep
```

trigger correctness 定义为：

```text
DeltaQ < 0  -> preferred KEEP  -> p_keep > p_update
DeltaQ > 0  -> preferred UPDATE -> p_update > p_keep
```

本次 exact helper 统计为：

```text
logical_branch_count = 683
counterfactual_environment_executions = 643
dedup_reused_branch_count = 40
```

branch reuse 只复用完整 physical post-configuration state 相同的 branch；没有使用 surrogate environment 或 candidate pruning。

## Eight-state result

| slot | target | p_keep | p_update | Q_keep | Q_update | DeltaQ | exact preferred | trigger correct | legal UPDATE candidates | good candidate mass |
|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| 20 | RSU_0 | 0.279457 | 0.720543 | -63.831894 | -63.706188 | +0.125706 | UPDATE | YES | 26 | 0.436112 |
| 20 | RSU_1 | 0.350682 | 0.649318 | -62.237225 | -62.087097 | +0.150127 | UPDATE | YES | 25 | 0.566771 |
| 80 | RSU_0 | 0.276858 | 0.723142 | -59.529781 | -60.315006 | -0.785225 | KEEP | NO | 18 | 0.468157 |
| 80 | RSU_1 | 0.354118 | 0.645882 | -59.529781 | -59.746906 | -0.217125 | KEEP | NO | 32 | 0.526607 |
| 140 | RSU_0 | 0.274302 | 0.725698 | -64.199020 | -66.135292 | -1.936272 | KEEP | NO | 31 | 0.530532 |
| 140 | RSU_1 | 0.346955 | 0.653045 | -67.071640 | -67.551964 | -0.480324 | KEEP | NO | 25 | 0.652974 |
| 200 | RSU_0 | 0.272866 | 0.727134 | -52.239716 | -53.302761 | -1.063046 | KEEP | NO | 19 | 0.489613 |
| 200 | RSU_1 | 0.347586 | 0.652414 | -52.351231 | -55.859905 | -3.508674 | KEEP | NO | 19 | 0.386916 |

## Required aggregates

```text
EPOCH1_CHECKPOINT_VALID = YES
STATE_COUNT = 8

KEEP_BETTER_STATE_COUNT = 6
UPDATE_BETTER_STATE_COUNT = 2
POLICY_TRIGGER_CORRECT_COUNT = 2/8

MEAN_P_KEEP = 0.3128530048
MEAN_P_UPDATE = 0.6871469989
MEAN_Q_UPDATE_MINUS_Q_KEEP = -0.9643540382

MEAN_P_KEEP_CHANGE = +0.0776497548
MEAN_P_UPDATE_CHANGE = -0.0776497511

MEAN_GOOD_UPDATE_CANDIDATE_MASS = 0.5072101871
```

`good candidate` 定义为 `Q_c > Q_keep`。8 个 witness 的合法 UPDATE candidates 中，good candidates 共 76 个，harmful candidates 共 119 个；这里的 candidate mass 是每个 RSU witness 内 good candidate 的 `pi(c | UPDATE)` 质量，再对 8 个 witness 求均值。

## 与 fresh/pretraining gate 的比较

R5 correctness gate 的同组 8 states 在 training 前为 `Q_update < Q_keep` 的 `8/8`，且 `p_update` 约为 `0.743140~0.789824`；对应的 rounded gate values 均满足 `p_update > p_keep`，所以 trigger correctness 为 `0/8`。本报告不比较当时 sampled action，只比较 policy probabilities 和 exact preference。

epoch1 后，当前 critic/环境 exact replay 给出 2 个 UPDATE-preferred state 和 6 个 KEEP-preferred state。与此同时 `p_keep` 均值提高约 `0.07765`、`p_update` 均值降低约 `0.07765`，correctness 增加 `2/8`。这支持 node-specific Cache credit 的第一轮学习方向，但 6/8 KEEP-preferred state 仍然是 `p_update > p_keep`，所以改善是部分的。

## Execution and change-control audit

```text
TRAINING_EXECUTED = NO
OPTIMIZER_STEP_EXECUTED = NO

REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
CACHE_CREDIT_CHANGED = NO
UAV_CREDIT_CHANGED = NO
KL_CHANGED = NO
LR_CHANGED = NO
MATCHING_KKT_CHANGED = NO

EPOCH2_EXECUTED = NO
EPOCH3_EXECUTED = NO
H10_H30_EXECUTED = NO
FORMAL_EVALUATION_EXECUTED = NO

EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO

OPTIMIZER_STEP_COUNTERS_BEFORE = (10, 40, 10)
OPTIMIZER_STEP_COUNTERS_AFTER  = (10, 40, 10)

SOURCE_MODIFIED = NO
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
```
