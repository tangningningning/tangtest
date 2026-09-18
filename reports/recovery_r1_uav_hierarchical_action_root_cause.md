# RECOVERY_R1_UAV_HIERARCHICAL_ACTION_ROOT_CAUSE

Bounded frozen V10-D diagnostic. No training, algorithm patch, reward change, or system-model change was performed.

- Checkpoint: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v10d_decoupled_encoder_branch_kl_6e4_formal\latest_checkpoint.pt`
- Frozen realizations: `[20001, 20100, 20200, 20300]`; origins `[80, 200]`; head-states `16`
- H1 evaluation: real `execute_slot`/Matching/KKT path; Cache and other UAV action fixed to production.

## Required fields

- `MEAN_TOP1_TO_UNIFORM_RATIO = 1.000540329`
- `MEAN_NORMALIZED_CANDIDATE_ENTROPY = 0.999999948`
- `MEAN_CANDIDATE_LOGIT_STD = 0.000522070`
- `UAV_CANDIDATE_PARAMS_IN_OPTIMIZER = YES`
- `UAV_CANDIDATE_LOGPROB_INCLUDED = YES`
- `UAV_CANDIDATE_GRAD_PRESENT_ON_MOVE = YES`
- `BEST_MOVE_BEATS_STAY_COUNT = 0`
- `PRODUCTION_MOVE_BEATS_STAY_COUNT = 0`
- `MEAN_PRODUCTION_MOVE_ENV_RANK = 25.125000000`
- `MEAN_PRODUCTION_MOVE_REGRET = 0.011043078`
- `MEAN_POLICY_ENV_SPEARMAN = 0.34001064435807216`
- `UAV_OPERATION_SELECTION_ERROR = SUPPORTED`
- `UAV_CANDIDATE_SELECTION_ERROR = NOT_SUPPORTED`
- `CANDIDATE_CLIP_STARVATION_POSSIBLE = YES`
- `CONFIRMED_UAV_CANDIDATE_IMPLEMENTATION_BUG = NO`

## Hierarchical probability and optimizer path

- Operation log-prob included: `YES`
- MOVE candidate log-prob included: `YES`
- Branch log-prob is sum of UAV heads: `YES`
- STAY has no candidate log-prob: `YES`
- Old/new same-formula max error: `1.907348633e-06`
- Candidate features: `NO` identical; variation `YES`.
- Constant feature dimensions across every state: `['uav_index', 'uav_identity_0', 'uav_identity_1', 'operation_STAY', 'operation_MOVE', 'before_x', 'before_y', 'vehicle_mean_x', 'vehicle_mean_y', 'vehicle_std_x', 'vehicle_std_y']`

## Real H1 candidate enumeration

`PRODUCTION_MOVE_REGRET = G_BEST_MOVE - G_PRODUCTION_MOVE`; positive values mean the policy-selected MOVE candidate lost to another legal MOVE candidate.

| Episode | Origin | UAV | N | G_STAY | G_PRODUCTION_MOVE | G_BEST_MOVE | Best idx | Production rank | Regret | Best−Stay | Policy/E env |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20001 | 80 | UAV_0 | 80 | 9.692343 | 9.663163 | 9.666686 | 0 | 22 | 0.003523 | -0.025657 | 0.862698 |
| 20001 | 80 | UAV_1 | 44 | 9.710145 | 9.663163 | 9.684643 | 5 | 34 | 0.021480 | -0.025502 | -0.500564 |
| 20001 | 200 | UAV_0 | 44 | 10.009621 | 9.981300 | 9.983636 | 43 | 13 | 0.002336 | -0.025985 | 0.922383 |
| 20001 | 200 | UAV_1 | 44 | 10.009693 | 9.981300 | 9.983222 | 43 | 13 | 0.001922 | -0.026471 | 0.864463 |
| 20100 | 80 | UAV_0 | 80 | 9.581068 | 9.552556 | 9.554576 | 0 | 24 | 0.002020 | -0.026491 | 0.867789 |
| 20100 | 80 | UAV_1 | 44 | 9.590798 | 9.552556 | 9.564293 | 6 | 26 | 0.011737 | -0.026505 | 0.573957 |
| 20100 | 200 | UAV_0 | 44 | 8.054856 | 8.026290 | 8.032126 | 1 | 16 | 0.005836 | -0.022730 | 0.798054 |
| 20100 | 200 | UAV_1 | 44 | 8.079312 | 8.026290 | 8.054247 | 7 | 40 | 0.027957 | -0.025065 | -0.549108 |
| 20200 | 80 | UAV_0 | 80 | 11.852175 | 11.820245 | 11.829090 | 15 | 32 | 0.008844 | -0.023085 | 0.734296 |
| 20200 | 80 | UAV_1 | 56 | 11.877142 | 11.820245 | 11.849969 | 10 | 54 | 0.029724 | -0.027174 | -0.924742 |
| 20200 | 200 | UAV_0 | 44 | 9.859098 | 9.830865 | 9.834346 | 1 | 13 | 0.003481 | -0.024752 | 0.872571 |
| 20200 | 200 | UAV_1 | 42 | 9.879676 | 9.830865 | 9.854311 | 8 | 37 | 0.023446 | -0.025365 | -0.908560 |
| 20300 | 80 | UAV_0 | 80 | 10.708789 | 10.680322 | 10.682254 | 15 | 24 | 0.001932 | -0.026536 | 0.877479 |
| 20300 | 80 | UAV_1 | 44 | 10.707570 | 10.680322 | 10.680407 | 3 | 6 | 0.000085 | -0.027163 | 0.990659 |
| 20300 | 200 | UAV_0 | 44 | 9.088376 | 9.063404 | 9.066465 | 43 | 9 | 0.003061 | -0.021910 | 0.812073 |
| 20300 | 200 | UAV_1 | 44 | 9.120006 | 9.063404 | 9.092709 | 3 | 39 | 0.029305 | -0.027296 | -0.853276 |

## Candidate ranking interpretation

Policy/environment Spearman status: `NEAR_UNIFORM`. Near-uniform candidate probabilities are not treated as meaningful ranking evidence.
Feature variation dimensions: `['after_x', 'after_y', 'delta_x', 'delta_y', 'distance_normalized', 'log_movement_energy', 'log_movement_cost']`.

## Gradient and clipping probe

- Probe: `episode_20001_300_slots_fixed_stochastic_seed_40101`; optimizer step: `NO`; parameters unchanged: `YES`.
- UAV encoder grad norm: `7.285622573e-03`; operation head: `6.840989835e-02`; candidate scorer: `1.682255102e-03`.
- Candidate grad on MOVE: `YES`; on STAY: `NO`.
- PPO ratio granularity: `JOINT_UAV_BRANCH`; operation/candidate share ratio: `YES`; starvation possible: `YES`.

This is a bounded witness, not a significance or generalization claim. It does not redesign or assess Cache policy quality.

## Status

`STAGE = RECOVERY_R1_UAV_HIERARCHICAL_ACTION_ROOT_CAUSE`; `TRAINING_EXECUTED = NO`; `FINAL_TEST_SEEDS_USED = NO`; `EPISODE_REGENERATION = NO`.
`SOURCE_MODIFIED = NO`; `SYSTEM_MODEL_CHANGED = NO`; `REWARD_CHANGED = NO`; `PUSHED = NO`.
