# R4 operation temporal-horizon gate

日期：2026-09-19  
仓库：`C:\Users\tangning\Desktop\code\adaptive_cache_mec`  
checkpoint：`runs/ppo_recovery_r4_wmove_0p50_exactcandidate_dev3_retry_20260919/latest_checkpoint.pt`  
冻结 realization：`runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20001`

## Decision

本轮没有训练、优化器更新、checkpoint 写入或 production source 修改。使用 R4 epoch3 checkpoint 对同一组 20 个 `slot × UAV` 状态完成 H1 重确认和 bounded H3 continuation diagnostic。H3 显示 best MOVE 可行状态增加，但 policy-expected MOVE 仍全部低于 STAY，且 expected delta 没有向 0 改善。

```text
CHECKPOINT_EPOCH = 3
STATE_COUNT = 20
H1_EXPECTED_MOVE_BEATS_STAY_COUNT = 0/20
H1_BEST_MOVE_BEATS_STAY_COUNT = 8/20

H3_EXPECTED_MOVE_BEATS_STAY_COUNT = 0/20
H3_BEST_MOVE_BEATS_STAY_COUNT = 12/20

EXPECTED_NEG_TO_POS_REVERSALS_H1_TO_H3 = 0
BEST_NEG_TO_POS_REVERSALS_H1_TO_H3 = 4

PRIMARY_DIAGNOSIS = TEMPORAL_VALUE_EXISTS_BUT_POLICY_EXPECTATION_REMAINS_NEGATIVE
H5_EXECUTED = NO
```

这属于 Case C：延长 horizon 暴露出更多 individually good MOVE candidate，但冻结 candidate policy 的 conditional expectation 仍然为负。因此不能把当前 operation collapse 单独归因于 one-step temporal myopia；时间尺度和 candidate concentration 仍共同存在。

## Protocol

目标状态严格为 slots `20,50,80,110,140,170,200,230,260,290` × `UAV_0/UAV_1`，没有扩样本。H1 使用现有 R4 exact counterfactual helper，保留 STAY 与全部 legal MOVE candidate。H3 对每个第一步 branch 从 `t+1` 到 `t+2` 继续执行 production action，并计算：

```text
Q_H3 = r_t + gamma*r_t+1 + gamma^2*r_t+2 + gamma^3*V(s_t+3)
Q_move_expected_H3 = sum_c pi_old(c | MOVE, s_t) * Q_c_H3
```

`gamma=1.0`。未来 action 使用 epoch3 R4 checkpoint 的确定性 production selection：

```text
FUTURE_POLICY_MODE = FROZEN_DETERMINISTIC
```

没有 candidate pruning、top-k、surrogate reward、future action-id forcing、fixed UAV position，也没有跳过正式 `execute_slot`、Matching 或 KKT。H3 共执行 1,620 个第一步 exact branches 和 3,240 个两-slot deterministic continuation branches，共 4,860 次环境执行。H1 与 H3 的 policy expectation 都使用第一步的 `pi_old(c | MOVE,s_t)`。

## H1 re-confirmation

H1 与此前 R4 epoch3 probe 一致：

```text
H1_EXPECTED_DELTA_MEAN = -0.0018906710111
H1_EXPECTED_DELTA_MEDIAN = -0.0018870957971
H1_BEST_DELTA_MEAN = +0.0010642493500
H1_BEST_DELTA_MEDIAN = -0.0008717313101
MEAN_H1_GOOD_POLICY_MASS = 0.0749460169
H1_RUNTIME_SECONDS = 47.6724
```

## H3 result

```text
H3_EXPECTED_DELTA_MEAN = -0.0019102630106
H3_EXPECTED_DELTA_MEDIAN = -0.0019035071198
H3_BEST_DELTA_MEAN = +0.0099185930627
H3_BEST_DELTA_MEDIAN = +0.0014240422596

MEAN_EXPECTED_TEMPORAL_GAIN = -0.0000195919996
MEDIAN_EXPECTED_TEMPORAL_GAIN = -0.0000208692501
MEAN_BEST_TEMPORAL_GAIN = +0.0088543437127
MEDIAN_BEST_TEMPORAL_GAIN = +0.0003918299156

GOOD_H3_CANDIDATE_COUNT = 317
MEAN_H3_GOOD_POLICY_MASS = 0.1922513171
H3_RUNTIME_SECONDS = 277.7688
```

The expected-MOVE mean changed from `-0.0018906710` at H1 to `-0.0019102630` at H3, a small movement away from zero rather than recovery. The best-MOVE count rose from `8/20` to `12/20`, with four H1-negative states becoming H3-positive for their individually best candidate. None of the policy-expected states recovered sign. The H3 good-candidate policy mass also increased from `0.0749460` to `0.1922513`, but the remaining probability on other candidates kept every expected MOVE below STAY.

Because none of the H5 gate conditions held—no expected sign recovery, no expected H1-to-H3 negative-to-positive reversal, and no expected-delta improvement toward zero—H5 was not executed. H10 and H30 were not executed.

## Required final keys

```text
CHECKPOINT_EPOCH = 3
STATE_COUNT = 20

H1_EXPECTED_MOVE_BEATS_STAY_COUNT = 0/20
H1_BEST_MOVE_BEATS_STAY_COUNT = 8/20
H3_EXPECTED_MOVE_BEATS_STAY_COUNT = 0/20
H3_BEST_MOVE_BEATS_STAY_COUNT = 12/20

EXPECTED_NEG_TO_POS_REVERSALS_H1_TO_H3 = 0
BEST_NEG_TO_POS_REVERSALS_H1_TO_H3 = 4
H1_EXPECTED_DELTA_MEAN = -0.0018906710111
H3_EXPECTED_DELTA_MEAN = -0.0019102630106
MEAN_EXPECTED_TEMPORAL_GAIN = -0.0000195919996
MEDIAN_EXPECTED_TEMPORAL_GAIN = -0.0000208692501
MEAN_H1_GOOD_POLICY_MASS = 0.0749460169
MEAN_H3_GOOD_POLICY_MASS = 0.1922513171

H5_EXECUTED = NO
PRIMARY_DIAGNOSIS = TEMPORAL_VALUE_EXISTS_BUT_POLICY_EXPECTATION_REMAINS_NEGATIVE

TRAINING_EXECUTED = NO
OPTIMIZER_STEP_EXECUTED = NO
SOURCE_MODIFIED = NO
REWARD_CHANGED = NO
W_MOVE_CHANGED = NO
CANDIDATE_CREDIT_CHANGED = NO
OPERATION_CREDIT_CHANGED = NO
CACHE_CREDIT_CHANGED = NO
EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
```
