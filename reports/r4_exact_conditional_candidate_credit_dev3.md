# R4 exact conditional candidate credit — Dev3

日期：2026-09-19  
仓库：`C:\Users\tangning\Desktop\code\adaptive_cache_mec`  
最终训练目录：`runs/ppo_recovery_r4_wmove_0p50_exactcandidate_dev3_retry_20260919`  
最终 checkpoint：`runs/ppo_recovery_r4_wmove_0p50_exactcandidate_dev3_retry_20260919/latest_checkpoint.pt`

## Outcome

R4 的 exact conditional candidate-credit correctness gate 通过，随后使用 fresh actor、critic 和 optimizers 完成了 3 个 300-slot epoch。candidate ranking 在 bounded H1 probe 中明显改善，但 policy-conditional expected MOVE 仍未在任何目标状态超过 STAY，且训练中的 operation policy 继续快速收缩到 STAY。

```text
R4_SCHEMA = adaptive_joint_cache_replica_aware_uav_operation_recovery_r4
R4_CONFIG = configs/proposed/system_config_ppo_recovery_r4_wmove_0p50_exactcandidate_dev3.toml

UAV_OPERATION_ADV_SOURCE = COUNTERFACTUAL_ONE_STEP_TD
UAV_CANDIDATE_ADV_SOURCE = EXACT_CONDITIONAL_COUNTERFACTUAL_Q
CANDIDATE_ADV_NORMALIZATION = NONE
OP_ADV_NORMALIZATION = FIXED_REFERENCE_SCALE
OP_ADV_REFERENCE_SCALE = 0.02

CORRECTNESS_GATE = PASS
DEV3_TRAINING_EXECUTED = YES
OPERATION_POLICY_RESULT = STAY_COLLAPSE_PERSISTS
CANDIDATE_CREDIT_FIX_EFFECT = NOT_SUPPORTED
```

`CANDIDATE_CREDIT_FIX_EFFECT = NOT_SUPPORTED` 的含义是：候选排序指标改善，但 operation STAY collapse 没有减缓或解除；不能仅因 candidate ranking 改善就把本轮修复判定为整体成功。

## R4 semantics and implementation

R4 复用了已有 exact operation counterfactual replay 中的 STAY 和全部 legal MOVE branch，不增加 candidate loss 专用的 `execute_slot`、Matching 或 KKT 枚举。对每个 legal MOVE candidate 使用：

```text
Q_c = r_c + gamma * V(next_c)
Q_move = sum_c pi_old(c | MOVE, s) * Q_c
A_candidate,c = Q_c - Q_move
```

candidate PPO loss 是全部 legal candidate 的 exact expectation：

```text
-sum_c pi_old(c) * min(ratio_c * A_c,
                       clip(ratio_c, 1-eps, 1+eps) * A_c)
```

其中 `eps=0.2`，没有 batch mean-centering、STD normalization、best-candidate target 或 MOVE bonus。loss weighting 保持 `0.5 * (operation_loss + candidate_loss)`。candidate loss 对 sampled operation 为 STAY 的 rollout state 也计算，因此本轮 candidate 更新是 `COUNTERFACTUAL_CONDITIONAL_POLICY_UPDATE`，不是 observed-action-only PPO sample。

operation credit、operation fixed reference scale、Cache global GAE、UAV encoder、operation head、candidate scorer、reward、movement cost、KL、learning rate 和 Cache/RSU 路径均保持原 R3 语义。

## Correctness gate

Gate 使用 seed `4401`、冻结 realization `episode_20001`，目标为 slots `20, 80, 140, 200` × `UAV_0/UAV_1`，共 8 个状态。gate 没有调用 `optimizer.step()`，也没有写回模型参数。

```text
MAX_WEIGHTED_CANDIDATE_ADV_MEAN_RESIDUAL = 1.0587335858274621e-15
CANDIDATE_LOSS_IDENTITY_MAX_ABS_DIFF = 2.6363656285808987e-12
GRADIENT_DIRECTION_POSITIVE_PASS = YES
GRADIENT_DIRECTION_NEGATIVE_PASS = YES
GRADIENT_DOT_WITH_ADVANTAGE_MIN = -7.259197445819154e-05
GRADIENT_DOT_WITH_ADVANTAGE_MAX = -3.023577619387652e-07

VIRTUAL_STEP_EXPECTED_Q_IMPROVEMENT = 0/8 strict
VIRTUAL_STEP_EXPECTED_Q_TIE = 8/8 explicit floating-point ties
VIRTUAL_STEP_GATE = PASS

OPTIMIZER_STEPS = 0
ACTOR_OPTIMIZER_STEPS = 0
CRITIC_OPTIMIZER_STEPS = 0
MAX_PARAMETER_ABS_DELTA_GATE = 0.0

COUNTERFACTUAL_ENV_EXECUTIONS_REUSED = 635
EXTRA_ENV_EXECUTIONS_DUE_TO_R4_CANDIDATE_CREDIT = 0
```

The virtual-step ties use canonical `raw_logits=log(pi_old)` and `epsilon=1e-8`; the explicit tie classification is why the gate passes despite zero strict positive changes.

## Fresh Dev3 training

Training used seed `4401`, the frozen pool `runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911`, and `episode_20001/20002/20003`. No checkpoint was loaded, no resume or warm start was used, and all PPO configuration values were inherited from R3. The first pre-fix launch stopped before an update because of a replay-terminal guard regression; it was left untouched. The final retry run below was fresh and completed normally with exit code 0.

`P(MOVE)` is the mean policy probability; `UAVMove` is the observed actual movement rate. Candidate residual is the logged training-batch diagnostic and is separate from the double-precision correctness-gate residual above.

| epoch | return | P(MOVE) | UAVMove | candidate loss | mean |A_candidate| | weighted residual | operation entropy | UAV KL | Cache KL | critic EV | actor steps | critic steps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -975.8790221 | 0.5194006145 | 0.5750000000 | -1.1266799625e-05 | 0.0016319269 | 2.8108128838e-07 | 0.6901012530 | 0.6351928244 | 0.0771209175 | -0.0000355070 | 20 | 40 |
| 2 | -2186.0100973 | 0.1917130939 | 0.1733333333 | -2.8789078416e-05 | 0.0022118685 | 1.4112167236e-06 | 0.4846533080 | 0.3772935497 | 0.0372983119 | 0.0066679184 | 60 | 80 |
| 3 | -1602.4670233 | 0.0171950374 | 0.0166666667 | -3.3459372243e-05 | 0.0021032131 | 3.9697028740e-06 | 0.0865123739 | 0.0148849119 | 0.0265258188 | 0.0125382585 | 100 | 120 |

The branch-specific PPO state paths were:

```text
EPOCH1_PPO_STATE_PATH = TT -> TF -> FF -> FF
EPOCH2_PPO_STATE_PATH = TT -> TT -> TF -> TF
EPOCH3_PPO_STATE_PATH = TT -> TT -> TT -> TT
```

The training diagnostics did not add candidate entropy or candidate top-1 fields; the required candidate loss, mean absolute candidate advantage and weighted residual were recorded for every epoch.

## Bounded H1 candidate-ranking probe

The probe used the epoch3 checkpoint and `episode_20001` at slots `20,50,80,110,140,170,200,230,260,290` × two UAV heads. It used the existing exact `Q_c` helper and enumerated 1,600 legal MOVE candidates plus 20 STAY branches, for 1,620 exact environment branches in the probe itself.

```text
EPOCH3_POLICY_TOP1_IS_BEST_COUNT = 0/20
EPOCH3_MEAN_SPEARMAN_POLICY_Q = 0.4847382557
EPOCH3_MEAN_GOOD_POLICY_MASS = 0.0749460169
EPOCH3_EXPECTED_MOVE_BEATS_STAY_COUNT = 0/20
EPOCH3_BEST_MOVE_BEATS_STAY_COUNT = 8/20
```

Here `good policy mass` is the candidate-policy probability mass on candidates with `Q_c > Q_stay`. For descriptive comparison only, the old R2 epoch7 probe was `top1 is best = 0/20`, mean Spearman approximately `-0.4707`, and expected MOVE beats STAY `0/20`. The checkpoints differ, so this is a mechanism comparison and not a statistical performance claim. The positive R4 Spearman and larger good-candidate mass indicate better candidate credit alignment, but the zero expected-MOVE count shows that the operation-level STAY preference remains unresolved.

## Required final keys

```text
R4_SCHEMA = adaptive_joint_cache_replica_aware_uav_operation_recovery_r4
R4_CONFIG = configs/proposed/system_config_ppo_recovery_r4_wmove_0p50_exactcandidate_dev3.toml

UAV_OPERATION_ADV_SOURCE = COUNTERFACTUAL_ONE_STEP_TD
UAV_CANDIDATE_ADV_SOURCE = EXACT_CONDITIONAL_COUNTERFACTUAL_Q
CANDIDATE_ADV_NORMALIZATION = NONE
OP_ADV_NORMALIZATION = FIXED_REFERENCE_SCALE
OP_ADV_REFERENCE_SCALE = 0.02

CORRECTNESS_GATE = PASS
MAX_WEIGHTED_CANDIDATE_ADV_MEAN_RESIDUAL = 1.0587335858274621e-15
CANDIDATE_LOSS_IDENTITY_MAX_ABS_DIFF = 2.6363656285808987e-12
VIRTUAL_STEP_EXPECTED_Q_IMPROVEMENT = 0/8 strict; 8/8 explicit ties
MAX_PARAMETER_ABS_DELTA_GATE = 0
EXTRA_ENV_EXECUTIONS_DUE_TO_R4_CANDIDATE_CREDIT = 0

DEV3_TRAINING_EXECUTED = YES
EPOCH1_P_MOVE = 0.5194006145
EPOCH2_P_MOVE = 0.1917130939
EPOCH3_P_MOVE = 0.0171950374
EPOCH1_UAV_MOVE = 0.5750000000
EPOCH2_UAV_MOVE = 0.1733333333
EPOCH3_UAV_MOVE = 0.0166666667
EPOCH1_CANDIDATE_LOSS = -1.1266799625e-05
EPOCH2_CANDIDATE_LOSS = -2.8789078416e-05
EPOCH3_CANDIDATE_LOSS = -3.3459372243e-05
EPOCH1_MEAN_ABS_CANDIDATE_ADV = 0.0016319269
EPOCH2_MEAN_ABS_CANDIDATE_ADV = 0.0022118685
EPOCH3_MEAN_ABS_CANDIDATE_ADV = 0.0021032131

EPOCH3_POLICY_TOP1_IS_BEST_COUNT = 0/20
EPOCH3_MEAN_SPEARMAN_POLICY_Q = 0.4847382557
EPOCH3_MEAN_GOOD_POLICY_MASS = 0.0749460169
EPOCH3_EXPECTED_MOVE_BEATS_STAY_COUNT = 0/20
EPOCH3_BEST_MOVE_BEATS_STAY_COUNT = 8/20

OPERATION_POLICY_RESULT = STAY_COLLAPSE_PERSISTS
CANDIDATE_CREDIT_FIX_EFFECT = NOT_SUPPORTED

REWARD_CHANGED = NO
MOVEMENT_COST_CHANGED = NO
W_MOVE_CHANGED = NO
OPERATION_CREDIT_CHANGED = NO
CANDIDATE_CREDIT_CHANGED = YES
CACHE_CREDIT_CHANGED = NO
TARGET_KL_CHANGED = NO
ACTOR_LR_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
MATCHING_KKT_CHANGED = NO
EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
```
