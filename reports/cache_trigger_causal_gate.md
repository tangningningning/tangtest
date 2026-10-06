# Cache Trigger Causal Gate

## Code path and diagnostic scope

唯一代码基线为当前 `main` 工作区。生产 Trigger 使用 `ONE_STEP_TD_Q_DIFFERENCE`；实际 UPDATE 的生产信用是 sampled UPDATE 对 KEEP，实际 KEEP 的生产信用是 KEEP 对 candidate-policy argmax UPDATE。此生产训练目标在本任务中未改。Candidate 使用 `GLOBAL_GAE`。生产 one-head branch 替换目标 Cache head，其余 Cache heads 与 UAV 复用实际 rollout 的结果，故 downstream mode 为 fixed-downstream production-rollout reuse。

`CURRENT_CACHE_TRIGGER_CREDIT = ONE_STEP_TD_Q_DIFFERENCE`
`CURRENT_CACHE_CANDIDATE_CREDIT = GLOBAL_GAE`
`CURRENT_COUNTERFACTUAL_DOWNSTREAM_MODE = FIXED_DOWNSTREAM_PRODUCTION_ROLLOUT_REUSE`
`CURRENT_CACHE_TRIGGER_ENTROPY_COEFFICIENT = 0.0` (shared PPO entropy setting; no Cache-specific override is configured)

本诊断统一 `delta = Q_UPDATE - Q_KEEP`，且 branch value 按现有实现取 `reward + gamma * frozen_critic(next_observation)`。`BEST_EXACT_UPDATE` 对每个合法完整 target cache configuration 真实调用现有 exact Cache branch；相同物理结果只复用相同 branch。policy-consistent replay 固定 checkpoint、同一 rollout transition 与 exogenous step，保留目标 head 前的生产 Cache prefix，强制目标 KEEP 或 BEST_EXACT_UPDATE，并通过生产 deterministic primitive selector 重新决策后续 Cache heads 和 UAV。两种 replay 都复用正式 environment → Matching → KKT → reward 路径。

`POLICY_CONSISTENT_MODE = DETERMINISTIC_FROZEN_POLICY_REPLAY`。该量不是 stochastic policy expectation，不报告为 exact policy Q。下一状态 bootstrap 使用当前冻结 global critic，因此它也不是整段未来 trajectory 的精确 rollout value。

固定 realization：`environment_data/proposed_joint_training/episode_20001`；只对每个 checkpoint 选出的 8 个 transition 做 full exact，9 个 Cache heads 均保留。状态选择确定性地覆盖 slot、CacheChange pattern、KEEP/UPDATE 样本、legal UPDATE 和 Trigger 概率区间。采样 rollout 固定 seed `73019`。

## Checkpoints

| Checkpoint | Preferred epoch | Analyzed file | Epoch | SHA256 | Selection status |
|---|---:|---|---:|---|---|
| LR=5e-4 | epoch_205_checkpoint.pt | `runs/stage2d_v3_fresh300_actor_lr_5e_minus_4/checkpoints/latest_checkpoint.pt` | 233 | `2ff73b63e327f229…` | LATEST_AVAILABLE_FALLBACK |
| LR=5e-3 | epoch_059_checkpoint.pt | `runs/stage2d_v3_fresh300_actor_lr_5e_minus_3/checkpoints/latest_checkpoint.pt` | 252 | `7a8ea452582afc01…` | LATEST_AVAILABLE_FALLBACK |

## Core statistics

`epoch_205_checkpoint.pt` and `epoch_059_checkpoint.pt` = CHECKPOINT_NOT_AVAILABLE; latest local checkpoints were analyzed instead. No training or model download was attempted.

### Overall

| Metric | Value |
|---|---:|
| POLICY_ARGMAX_VS_BEST_SIGN_FLIP_RATE | 0.465 (67/144) |
| SAMPLED_VS_BEST_SIGN_FLIP_RATE | 0.592 (42/71) |
| FIXED_VS_POLICY_CONSISTENT_SIGN_FLIP_RATE | 0.076 (11/144) |
| SELECTED_CANDIDATE_TOP1_EXACT_RATE | 0.042 |
| POLICY_ARGMAX_CANDIDATE_TOP1_EXACT_RATE | 0.049 |
| SELECTED_CANDIDATE_MEAN_EXACT_RANK | 13.127 |
| POLICY_ARGMAX_CANDIDATE_MEAN_EXACT_RANK | 9.083 |
| SELECTED_CANDIDATE_MEAN_REGRET | 22.344 |
| POLICY_ARGMAX_CANDIDATE_MEAN_REGRET | 60.882 |
| BEST_EXACT_CANDIDATE_MEAN_POLICY_PROB | 0.070 |
| MEAN_ABS_DELTA_DIFFERENCE | 6.929 |
| MAX_ABS_DELTA_DIFFERENCE | 124.243 |

### By checkpoint

| Checkpoint | Transitions | Head rows | UPDATE / KEEP samples | Legal rows | Policy argmax→best flip rate | Sampled→best flip rate | Selected top1 / mean rank / mean regret | Policy argmax top1 / mean rank / mean regret | Best exact candidate mean policy prob. | Fixed→consistent flip rate | Mean / max abs Δ difference |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| LR=5e-4 | 8 | 72 | 15 / 57 | 72 | 0.542 (39/72) | 0.667 (10/15) | 0.000 / 18.800 / 18.847 | 0.097 / 10.208 / 105.431 | 0.067 | 0.000 (0/72) | 1.448 / 20.281 |
| LR=5e-3 | 8 | 72 | 56 / 16 | 72 | 0.389 (28/72) | 0.571 (32/56) | 0.054 / 11.607 / 23.281 | 0.000 / 7.958 / 16.334 | 0.072 | 0.153 (11/72) | 12.411 / 124.243 |

### By checkpoint and Cache head

| Checkpoint | Head | UPDATE | KEEP | Mean legal candidates | Policy→best flip | Sampled→best flip | Selected top1 / rank / regret | Policy argmax top1 / rank / regret | Best exact policy prob. | Fixed→consistent flip | Mean / max abs Δ difference | Δ sign flips across states (fixed / consistent) | p_update stays extreme while delta flips |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|
| LR=5e-4 | 0 | 0 | 8 | 8.125 | 0.000 | NA | NA / NA / NA | 0.000 / 8.125 / 78.533 | 0.026 | 0.000 | 1.112 / 2.963 | False / False | False |
| LR=5e-4 | 1 | 0 | 8 | 9.000 | 0.250 | NA | NA / NA / NA | 0.000 / 5.000 / 112.497 | 0.023 | 0.000 | 0.741 / 1.924 | True / True | True |
| LR=5e-4 | 2 | 0 | 8 | 8.000 | 0.000 | NA | NA / NA / NA | 0.875 / 1.125 / 2.245 | 0.457 | 0.000 | 4.086 / 20.281 | False / False | False |
| LR=5e-4 | 3 | 0 | 8 | 9.000 | 1.000 | NA | NA / NA / NA | 0.000 / 7.875 / 86.644 | 0.057 | 0.000 | 1.381 / 4.141 | False / False | False |
| LR=5e-4 | 4 | 0 | 8 | 7.000 | 1.000 | NA | NA / NA / NA | 0.000 / 6.875 / 100.281 | 0.009 | 0.000 | 5.108 / 14.273 | False / False | False |
| LR=5e-4 | 5 | 7 | 1 | 31.000 | 0.625 | 0.571 | 0.000 / 19.143 / 19.376 | 0.000 / 17.750 / 18.262 | 0.002 | 0.000 | 0.567 / 1.319 | False / False | False |
| LR=5e-4 | 6 | 8 | 0 | 30.750 | 0.750 | 0.750 | 0.000 / 18.500 / 18.384 | 0.000 / 21.000 / 18.269 | 0.003 | 0.000 | 0.011 / 0.034 | False / False | False |
| LR=5e-4 | 7 | 0 | 8 | 16.375 | 0.250 | NA | NA / NA / NA | 0.000 / 12.125 / 126.098 | 0.016 | 0.000 | 0.003 / 0.009 | True / True | True |
| LR=5e-4 | 8 | 0 | 8 | 12.000 | 1.000 | NA | NA / NA / NA | 0.000 / 12.000 / 406.046 | 0.011 | 0.000 | 0.022 / 0.058 | False / False | False |
| LR=5e-3 | 0 | 8 | 0 | 11.625 | 0.250 | 0.250 | 0.250 / 5.000 / 13.809 | 0.000 / 5.000 / 16.474 | 0.089 | 0.375 | 10.591 / 62.923 | True / True | True |
| LR=5e-3 | 1 | 0 | 8 | 9.125 | 0.500 | NA | NA / NA / NA | 0.000 / 7.750 / 16.892 | 0.110 | 0.500 | 32.066 / 124.243 | False / True | True |
| LR=5e-3 | 2 | 8 | 0 | 12.625 | 0.375 | 0.375 | 0.000 / 7.500 / 20.754 | 0.000 / 6.125 / 14.698 | 0.081 | 0.250 | 15.558 / 115.719 | False / True | True |
| LR=5e-3 | 3 | 8 | 0 | 9.750 | 0.750 | 0.625 | 0.000 / 6.250 / 18.947 | 0.000 / 6.125 / 13.172 | 0.116 | 0.125 | 8.317 / 59.160 | False / True | True |
| LR=5e-3 | 4 | 8 | 0 | 12.000 | 0.625 | 0.500 | 0.125 / 5.875 / 19.342 | 0.000 / 6.750 / 14.335 | 0.099 | 0.125 | 17.526 / 115.892 | False / True | True |
| LR=5e-3 | 5 | 0 | 8 | 20.875 | 0.125 | NA | NA / NA / NA | 0.000 / 6.625 / 24.792 | 0.048 | 0.000 | 19.134 / 123.650 | False / False | False |
| LR=5e-3 | 6 | 8 | 0 | 30.875 | 0.250 | 0.750 | 0.000 / 19.500 / 32.250 | 0.000 / 11.750 / 13.427 | 0.033 | 0.000 | 1.577 / 4.324 | False / False | False |
| LR=5e-3 | 7 | 8 | 0 | 25.625 | 0.125 | 0.875 | 0.000 / 17.375 / 30.902 | 0.000 / 7.000 / 15.199 | 0.040 | 0.000 | 6.923 / 50.854 | False / False | False |
| LR=5e-3 | 8 | 8 | 0 | 27.375 | 0.500 | 0.625 | 0.000 / 19.750 / 26.962 | 0.000 / 14.500 / 18.016 | 0.037 | 0.000 | 0.005 / 0.019 | False / False | False |

### Trigger sample responsiveness by checkpoint and Cache head

| Checkpoint | Head | States | p_update min–max | Fixed exact signs across states | Policy-consistent signs across states | p_update stays on same extreme side while either delta changes sign |
|---|---:|---:|---|---|---|---|
| LR=5e-4 | 0 | 8 | 0.000–0.004 | KEEP_BETTER | KEEP_BETTER | False |
| LR=5e-4 | 1 | 8 | 0.000–0.002 | KEEP_BETTER, UPDATE_BETTER | KEEP_BETTER, UPDATE_BETTER | True |
| LR=5e-4 | 2 | 8 | 0.000–0.001 | KEEP_BETTER | KEEP_BETTER | False |
| LR=5e-4 | 3 | 8 | 0.000–0.000 | UPDATE_BETTER | UPDATE_BETTER | False |
| LR=5e-4 | 4 | 8 | 0.000–0.007 | UPDATE_BETTER | UPDATE_BETTER | False |
| LR=5e-4 | 5 | 8 | 0.973–0.998 | UPDATE_BETTER | UPDATE_BETTER | False |
| LR=5e-4 | 6 | 8 | 0.941–0.999 | UPDATE_BETTER | UPDATE_BETTER | False |
| LR=5e-4 | 7 | 8 | 0.000–0.008 | KEEP_BETTER, UPDATE_BETTER | KEEP_BETTER, UPDATE_BETTER | True |
| LR=5e-4 | 8 | 8 | 0.000–0.001 | UPDATE_BETTER | UPDATE_BETTER | False |
| LR=5e-3 | 0 | 8 | 1.000–1.000 | KEEP_BETTER, UPDATE_BETTER | KEEP_BETTER, UPDATE_BETTER | True |
| LR=5e-3 | 1 | 8 | 0.000–0.000 | UPDATE_BETTER | KEEP_BETTER, UPDATE_BETTER | True |
| LR=5e-3 | 2 | 8 | 1.000–1.000 | UPDATE_BETTER | KEEP_BETTER, UPDATE_BETTER | True |
| LR=5e-3 | 3 | 8 | 1.000–1.000 | UPDATE_BETTER | KEEP_BETTER, UPDATE_BETTER | True |
| LR=5e-3 | 4 | 8 | 1.000–1.000 | UPDATE_BETTER | KEEP_BETTER, UPDATE_BETTER | True |
| LR=5e-3 | 5 | 8 | 0.000–0.000 | UPDATE_BETTER | UPDATE_BETTER | False |
| LR=5e-3 | 6 | 8 | 1.000–1.000 | UPDATE_BETTER | UPDATE_BETTER | False |
| LR=5e-3 | 7 | 8 | 1.000–1.000 | UPDATE_BETTER | UPDATE_BETTER | False |
| LR=5e-3 | 8 | 8 | 1.000–1.000 | UPDATE_BETTER | UPDATE_BETTER | False |

### Transition × Cache head details

| Checkpoint | Transition | Slot | Head index / id | Action | p_update | Legal | Δ sampled fixed | Δ policy argmax fixed | Δ best exact fixed | Δ best policy-consistent | Selected rank / regret | Policy argmax rank / regret | Best candidate prob. | Fixed / consistent sign | Flip | Cache changed |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|---|
| LR=5e-4 | 298 | 298 | 0 / 46 | KEEP | 0.004 | 8 | NA | -65.618 | -45.286 | -44.365 | NA / NA | 8 / 20.332 | 0.003 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 298 | 298 | 1 / 10 | KEEP | 0.002 | 9 | NA | -31.128 | 5.247 | 5.452 | NA / NA | 4 / 36.375 | 0.002 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 298 | 298 | 2 / 0 | KEEP | 0.001 | 8 | NA | -36.086 | -18.124 | -18.031 | NA / NA | 2 / 17.962 | 0.337 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 298 | 298 | 3 / 23 | KEEP | 0.000 | 9 | NA | -16.082 | 9.033 | 9.541 | NA / NA | 6 / 25.116 | 0.004 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 298 | 298 | 4 / 59 | KEEP | 0.007 | 7 | NA | -15.304 | 7.555 | 6.567 | NA / NA | 7 / 22.859 | 0.005 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 298 | 298 | 5 / RSU_0 | UPDATE | 0.973 | 21 | 0.693 | -6.004 | 14.568 | 14.568 | 10 / 13.876 | 16 / 20.573 | 0.000 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-4 | 298 | 298 | 6 / RSU_1 | UPDATE | 0.941 | 32 | -4.215 | -4.215 | 9.824 | 9.824 | 18 / 14.039 | 18 / 14.039 | 0.000 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-4 | 298 | 298 | 7 / UAV_0 | KEEP | 0.008 | 16 | NA | -40.781 | 1.910 | 1.910 | NA / NA | 10 / 42.690 | 0.004 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 298 | 298 | 8 / UAV_1 | KEEP | 0.001 | 12 | NA | -74.295 | 83.227 | 83.227 | NA / NA | 12 / 157.522 | 0.001 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 183 | 183 | 0 / 46 | KEEP | 0.000 | 8 | NA | -208.374 | -111.008 | -110.101 | NA / NA | 8 / 97.367 | 0.061 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 183 | 183 | 1 / 10 | KEEP | 0.000 | 9 | NA | -128.013 | 7.844 | 6.084 | NA / NA | 6 / 135.857 | 0.039 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 183 | 183 | 2 / 0 | KEEP | 0.000 | 8 | NA | -25.621 | -25.621 | -25.140 | NA / NA | 1 / 0.000 | 0.479 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 183 | 183 | 3 / 23 | KEEP | 0.000 | 9 | NA | -95.210 | 35.003 | 34.541 | NA / NA | 8 / 130.212 | 0.037 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 183 | 183 | 4 / 59 | KEEP | 0.000 | 7 | NA | -42.250 | 81.184 | 71.827 | NA / NA | 7 / 123.434 | 0.016 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 183 | 183 | 5 / RSU_0 | KEEP | 0.998 | 30 | NA | -1.640 | 21.747 | 23.066 | NA / NA | 18 / 23.388 | 0.002 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 183 | 183 | 6 / RSU_1 | UPDATE | 0.999 | 21 | -2.702 | 2.498 | 27.108 | 27.083 | 20 / 29.810 | 16 / 24.610 | 0.000 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-4 | 183 | 183 | 7 / UAV_0 | KEEP | 0.000 | 17 | NA | -169.305 | -28.589 | -28.580 | NA / NA | 11 / 140.716 | 0.022 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 183 | 183 | 8 / UAV_1 | KEEP | 0.000 | 12 | NA | -343.227 | 161.397 | 161.343 | NA / NA | 12 / 504.624 | 0.017 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 84 | 84 | 0 / 46 | KEEP | 0.000 | 9 | NA | -171.167 | -88.747 | -91.710 | NA / NA | 9 / 82.419 | 0.003 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 84 | 84 | 1 / 10 | KEEP | 0.000 | 9 | NA | -123.504 | -2.933 | -3.296 | NA / NA | 4 / 120.570 | 0.003 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 84 | 84 | 2 / 0 | KEEP | 0.000 | 8 | NA | -19.442 | -19.442 | -15.975 | NA / NA | 1 / 0.000 | 0.501 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 84 | 84 | 3 / 23 | KEEP | 0.000 | 9 | NA | -73.061 | 12.265 | 11.241 | NA / NA | 8 / 85.326 | 0.002 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 84 | 84 | 4 / 59 | KEEP | 0.000 | 6 | NA | -40.980 | 66.137 | 66.994 | NA / NA | 6 / 107.117 | 0.006 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 84 | 84 | 5 / RSU_0 | UPDATE | 0.996 | 33 | 3.513 | 0.748 | 11.319 | 12.488 | 12 / 7.806 | 17 / 10.571 | 0.000 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-4 | 84 | 84 | 6 / RSU_1 | UPDATE | 0.999 | 33 | -4.510 | -11.186 | 8.130 | 8.096 | 15 / 12.640 | 25 / 19.316 | 0.001 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-4 | 84 | 84 | 7 / UAV_0 | KEEP | 0.000 | 16 | NA | -157.330 | -8.573 | -8.574 | NA / NA | 13 / 148.758 | 0.024 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 84 | 84 | 8 / UAV_1 | KEEP | 0.000 | 12 | NA | -328.908 | 149.685 | 149.672 | NA / NA | 12 / 478.592 | 0.014 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 93 | 93 | 0 / 46 | KEEP | 0.000 | 8 | NA | -173.883 | -93.398 | -92.285 | NA / NA | 8 / 80.485 | 0.014 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 93 | 93 | 1 / 10 | KEEP | 0.000 | 9 | NA | -122.482 | -0.711 | -0.573 | NA / NA | 4 / 121.771 | 0.021 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 93 | 93 | 2 / 0 | KEEP | 0.000 | 8 | NA | -20.674 | -20.674 | -27.543 | NA / NA | 1 / 0.000 | 0.500 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 93 | 93 | 3 / 23 | KEEP | 0.000 | 9 | NA | -80.165 | 13.880 | 9.740 | NA / NA | 8 / 94.046 | 0.040 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 93 | 93 | 4 / 59 | KEEP | 0.000 | 7 | NA | -42.123 | 76.738 | 77.992 | NA / NA | 7 / 118.861 | 0.009 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 93 | 93 | 5 / RSU_0 | UPDATE | 0.996 | 33 | -9.321 | -2.469 | 11.878 | 11.863 | 27 / 21.199 | 22 / 14.347 | 0.003 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-4 | 93 | 93 | 6 / RSU_1 | UPDATE | 0.998 | 33 | -6.894 | -6.894 | 8.988 | 8.987 | 27 / 15.882 | 27 / 15.882 | 0.002 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-4 | 93 | 93 | 7 / UAV_0 | KEEP | 0.000 | 16 | NA | -154.021 | -9.522 | -9.522 | NA / NA | 13 / 144.499 | 0.019 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 93 | 93 | 8 / UAV_1 | KEEP | 0.000 | 12 | NA | -326.520 | 168.443 | 168.443 | NA / NA | 12 / 494.964 | 0.010 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 263 | 263 | 0 / 46 | KEEP | 0.000 | 8 | NA | -189.119 | -102.759 | -102.762 | NA / NA | 8 / 86.360 | 0.051 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 263 | 263 | 1 / 10 | KEEP | 0.000 | 9 | NA | -112.995 | -5.650 | -5.643 | NA / NA | 6 / 107.345 | 0.040 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 263 | 263 | 2 / 0 | KEEP | 0.000 | 8 | NA | -15.450 | -15.450 | -15.686 | NA / NA | 1 / 0.000 | 0.443 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 263 | 263 | 3 / 23 | KEEP | 0.000 | 9 | NA | -89.329 | 29.470 | 29.489 | NA / NA | 9 / 118.799 | 0.038 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 263 | 263 | 4 / 59 | KEEP | 0.000 | 7 | NA | -34.780 | 76.514 | 62.241 | NA / NA | 7 / 111.293 | 0.008 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 263 | 263 | 5 / RSU_0 | UPDATE | 0.998 | 33 | -2.841 | -2.841 | 20.334 | 19.449 | 25 / 23.175 | 25 / 23.175 | 0.001 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-4 | 263 | 263 | 6 / RSU_1 | UPDATE | 0.999 | 32 | 1.382 | 1.382 | 10.861 | 10.854 | 9 / 9.479 | 9 / 9.479 | 0.002 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-4 | 263 | 263 | 7 / UAV_0 | KEEP | 0.000 | 17 | NA | -181.511 | -32.443 | -32.441 | NA / NA | 14 / 149.068 | 0.015 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 263 | 263 | 8 / UAV_1 | KEEP | 0.000 | 12 | NA | -238.507 | 162.024 | 162.048 | NA / NA | 12 / 400.532 | 0.009 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 6 | 6 | 0 / 46 | KEEP | 0.000 | 8 | NA | -92.746 | -7.464 | -6.515 | NA / NA | 8 / 85.281 | 0.005 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 6 | 6 | 1 / 10 | KEEP | 0.000 | 9 | NA | -51.631 | -15.334 | -16.753 | NA / NA | 2 / 36.297 | 0.015 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 6 | 6 | 2 / 0 | KEEP | 0.000 | 8 | NA | -24.169 | -24.169 | -24.015 | NA / NA | 1 / 0.000 | 0.488 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 6 | 6 | 3 / 23 | KEEP | 0.000 | 9 | NA | -23.304 | 6.528 | 7.423 | NA / NA | 8 / 29.832 | 0.284 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 6 | 6 | 4 / 59 | KEEP | 0.000 | 8 | NA | -1.270 | 61.579 | 63.826 | NA / NA | 7 / 62.848 | 0.009 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 6 | 6 | 5 / RSU_0 | UPDATE | 0.997 | 32 | 5.442 | 5.931 | 13.893 | 14.357 | 12 / 8.451 | 10 / 7.963 | 0.002 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-4 | 6 | 6 | 6 / RSU_1 | UPDATE | 0.998 | 32 | 2.583 | -3.264 | 4.913 | 4.922 | 3 / 2.330 | 19 / 8.177 | 0.007 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-4 | 6 | 6 | 7 / UAV_0 | KEEP | 0.000 | 16 | NA | -81.945 | 3.033 | 3.031 | NA / NA | 11 / 84.978 | 0.009 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 6 | 6 | 8 / UAV_1 | KEEP | 0.000 | 12 | NA | -195.680 | 66.492 | 66.477 | NA / NA | 12 / 262.171 | 0.010 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 134 | 134 | 0 / 46 | KEEP | 0.000 | 8 | NA | -187.753 | -95.553 | -95.323 | NA / NA | 8 / 92.200 | 0.017 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 134 | 134 | 1 / 10 | KEEP | 0.000 | 9 | NA | -120.879 | -5.559 | -7.483 | NA / NA | 5 / 115.320 | 0.031 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 134 | 134 | 2 / 0 | KEEP | 0.000 | 8 | NA | -47.092 | -47.092 | -26.811 | NA / NA | 1 / 0.000 | 0.494 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 134 | 134 | 3 / 23 | KEEP | 0.000 | 9 | NA | -85.026 | 14.447 | 17.019 | NA / NA | 8 / 99.472 | 0.017 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 134 | 134 | 4 / 59 | KEEP | 0.000 | 7 | NA | -30.624 | 91.217 | 102.243 | NA / NA | 7 / 121.841 | 0.011 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 134 | 134 | 5 / RSU_0 | UPDATE | 0.996 | 34 | -8.564 | -1.782 | 23.223 | 23.216 | 23 / 31.787 | 18 / 25.004 | 0.003 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-4 | 134 | 134 | 6 / RSU_1 | UPDATE | 0.999 | 30 | -3.596 | -3.596 | 20.835 | 20.830 | 25 / 24.431 | 25 / 24.431 | 0.005 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-4 | 134 | 134 | 7 / UAV_0 | KEEP | 0.000 | 17 | NA | -148.056 | -8.402 | -8.393 | NA / NA | 13 / 139.654 | 0.024 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 134 | 134 | 8 / UAV_1 | KEEP | 0.000 | 12 | NA | -341.963 | 165.591 | 165.533 | NA / NA | 12 / 507.555 | 0.016 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 216 | 216 | 0 / 46 | KEEP | 0.000 | 8 | NA | -194.863 | -111.048 | -109.233 | NA / NA | 8 / 83.815 | 0.051 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 216 | 216 | 1 / 10 | KEEP | 0.000 | 9 | NA | -236.647 | -10.210 | -10.095 | NA / NA | 9 / 226.438 | 0.035 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 216 | 216 | 2 / 0 | KEEP | 0.000 | 8 | NA | -12.670 | -12.670 | -11.565 | NA / NA | 1 / 0.000 | 0.413 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 216 | 216 | 3 / 23 | KEEP | 0.000 | 9 | NA | -91.376 | 18.970 | 17.539 | NA / NA | 8 / 110.346 | 0.035 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 216 | 216 | 4 / 59 | KEEP | 0.000 | 7 | NA | -44.724 | 89.272 | 90.133 | NA / NA | 7 / 133.996 | 0.009 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-4 | 216 | 216 | 5 / RSU_0 | UPDATE | 0.997 | 32 | -8.044 | 0.220 | 21.297 | 21.975 | 25 / 29.341 | 16 / 21.077 | 0.002 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-4 | 216 | 216 | 6 / RSU_1 | UPDATE | 0.998 | 33 | -14.668 | -6.427 | 23.796 | 23.801 | 31 / 38.463 | 29 / 30.223 | 0.006 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-4 | 216 | 216 | 7 / UAV_0 | KEEP | 0.000 | 16 | NA | -195.182 | -36.759 | -36.760 | NA / NA | 12 / 158.422 | 0.014 | KEEP_BETTER / KEEP_BETTER | False | False |
| LR=5e-4 | 216 | 216 | 8 / UAV_1 | KEEP | 0.000 | 12 | NA | -283.080 | 159.329 | 159.342 | NA / NA | 12 / 442.410 | 0.012 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-3 | 61 | 61 | 0 / 46 | UPDATE | 1.000 | 13 | 23.232 | 3.257 | 23.232 | 23.078 | 1 / 0.000 | 5 / 19.975 | 0.077 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 61 | 61 | 1 / 10 | KEEP | 0.000 | 9 | NA | -0.449 | 14.725 | 13.215 | NA / NA | 9 / 15.175 | 0.111 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-3 | 61 | 61 | 2 / 0 | UPDATE | 1.000 | 10 | 0.645 | 1.726 | 8.666 | 8.776 | 6 / 8.020 | 4 / 6.939 | 0.100 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 61 | 61 | 3 / 23 | UPDATE | 1.000 | 9 | 2.900 | -0.575 | 4.009 | 3.724 | 2 / 1.109 | 6 / 4.584 | 0.111 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 61 | 61 | 4 / 59 | UPDATE | 1.000 | 14 | -1.677 | -1.677 | 9.341 | 10.004 | 6 / 11.018 | 6 / 11.018 | 0.071 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 61 | 61 | 5 / RSU_0 | KEEP | 0.000 | 21 | NA | 1.799 | 46.667 | 45.504 | NA / NA | 12 / 44.868 | 0.048 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-3 | 61 | 61 | 6 / RSU_1 | UPDATE | 1.000 | 34 | 1.750 | 1.378 | 4.724 | 4.634 | 10 / 2.975 | 11 / 3.346 | 0.029 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 61 | 61 | 7 / UAV_0 | UPDATE | 1.000 | 28 | -1.917 | 0.494 | 19.580 | 18.990 | 14 / 21.498 | 11 / 19.087 | 0.036 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 61 | 61 | 8 / UAV_1 | UPDATE | 1.000 | 31 | -2.054 | -3.539 | 10.293 | 10.296 | 27 / 12.347 | 31 / 13.832 | 0.032 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 11 | 11 | 0 / 46 | UPDATE | 1.000 | 13 | 62.805 | 4.826 | 62.805 | -0.118 | 1 / 0.000 | 6 / 57.979 | 0.077 | UPDATE_BETTER / KEEP_BETTER | True | True |
| LR=5e-3 | 11 | 11 | 1 / 10 | KEEP | 0.000 | 10 | NA | 3.887 | 22.867 | -10.962 | NA / NA | 10 / 18.980 | 0.100 | UPDATE_BETTER / KEEP_BETTER | True | False |
| LR=5e-3 | 11 | 11 | 2 / 0 | UPDATE | 1.000 | 14 | -3.573 | -1.934 | 2.414 | -0.327 | 12 / 5.987 | 9 / 4.348 | 0.071 | UPDATE_BETTER / KEEP_BETTER | True | True |
| LR=5e-3 | 11 | 11 | 3 / 23 | UPDATE | 1.000 | 12 | -2.078 | 0.301 | 2.814 | 2.847 | 12 / 4.892 | 8 / 2.513 | 0.083 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 11 | 11 | 4 / 59 | UPDATE | 1.000 | 14 | 2.644 | 0.881 | 7.533 | 3.255 | 7 / 4.889 | 11 / 6.652 | 0.071 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 11 | 11 | 5 / RSU_0 | KEEP | 0.000 | 21 | NA | 39.704 | 41.121 | 44.541 | NA / NA | 2 / 1.417 | 0.048 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-3 | 11 | 11 | 6 / RSU_1 | UPDATE | 1.000 | 31 | 4.750 | 1.913 | 5.742 | 6.302 | 2 / 0.992 | 7 / 3.829 | 0.032 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 11 | 11 | 7 / UAV_0 | UPDATE | 1.000 | 19 | -13.487 | -1.526 | 4.692 | 4.711 | 15 / 18.179 | 11 / 6.218 | 0.053 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 11 | 11 | 8 / UAV_1 | UPDATE | 1.000 | 31 | -2.419 | 1.058 | 10.218 | 10.219 | 27 / 12.637 | 18 / 9.160 | 0.032 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 248 | 248 | 0 / 46 | UPDATE | 1.000 | 12 | 4.670 | 6.851 | 26.123 | 24.836 | 5 / 21.453 | 2 / 19.272 | 0.083 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 248 | 248 | 1 / 10 | KEEP | 0.000 | 9 | NA | 27.458 | 32.374 | 30.713 | NA / NA | 7 / 4.915 | 0.111 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-3 | 248 | 248 | 2 / 0 | UPDATE | 1.000 | 13 | 1.669 | 2.467 | 26.959 | 25.702 | 8 / 25.291 | 5 / 24.492 | 0.077 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 248 | 248 | 3 / 23 | UPDATE | 1.000 | 13 | 2.206 | -3.843 | 19.899 | 21.071 | 2 / 17.694 | 9 / 23.743 | 0.077 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 248 | 248 | 4 / 59 | UPDATE | 1.000 | 11 | 5.572 | -1.593 | 15.984 | 15.833 | 2 / 10.412 | 9 / 17.577 | 0.091 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 248 | 248 | 5 / RSU_0 | KEEP | 0.000 | 20 | NA | -2.148 | 24.108 | 16.206 | NA / NA | 14 / 26.256 | 0.050 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-3 | 248 | 248 | 6 / RSU_1 | UPDATE | 1.000 | 32 | -11.934 | 0.804 | 31.359 | 27.035 | 28 / 43.293 | 14 / 30.555 | 0.031 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 248 | 248 | 7 / UAV_0 | UPDATE | 1.000 | 30 | -1.174 | 22.844 | 28.576 | 28.643 | 21 / 29.751 | 3 / 5.732 | 0.033 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 248 | 248 | 8 / UAV_1 | UPDATE | 1.000 | 21 | -6.373 | -0.009 | 29.729 | 29.724 | 20 / 36.102 | 11 / 29.738 | 0.048 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 113 | 113 | 0 / 46 | UPDATE | 1.000 | 10 | -0.596 | 1.711 | 2.411 | 0.247 | 5 / 3.007 | 2 / 0.700 | 0.100 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 113 | 113 | 1 / 10 | KEEP | 0.000 | 9 | NA | 4.871 | 10.406 | 18.299 | NA / NA | 9 / 5.535 | 0.111 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-3 | 113 | 113 | 2 / 0 | UPDATE | 1.000 | 13 | -1.257 | 0.868 | 3.707 | 1.862 | 11 / 4.964 | 5 / 2.839 | 0.077 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 113 | 113 | 3 / 23 | UPDATE | 1.000 | 10 | -1.475 | -1.923 | 2.980 | 2.432 | 6 / 4.456 | 8 / 4.903 | 0.100 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 113 | 113 | 4 / 59 | UPDATE | 1.000 | 14 | -1.500 | -0.591 | 0.916 | -15.166 | 5 / 2.416 | 3 / 1.506 | 0.071 | UPDATE_BETTER / KEEP_BETTER | True | True |
| LR=5e-3 | 113 | 113 | 5 / RSU_0 | KEEP | 0.000 | 21 | NA | 6.629 | 48.139 | 50.339 | NA / NA | 12 / 41.510 | 0.048 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-3 | 113 | 113 | 6 / RSU_1 | UPDATE | 1.000 | 31 | -2.480 | -4.137 | 6.716 | 6.169 | 16 / 9.196 | 22 / 10.853 | 0.032 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 113 | 113 | 7 / UAV_0 | UPDATE | 1.000 | 26 | 2.412 | 2.130 | 6.803 | 8.247 | 6 / 4.391 | 7 / 4.673 | 0.038 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 113 | 113 | 8 / UAV_1 | UPDATE | 1.000 | 25 | -1.259 | 13.704 | 14.010 | 14.010 | 21 / 15.269 | 2 / 0.305 | 0.040 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 135 | 135 | 0 / 46 | UPDATE | 1.000 | 13 | 8.162 | -0.706 | 19.452 | 11.191 | 2 / 11.289 | 8 / 20.158 | 0.077 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 135 | 135 | 1 / 10 | KEEP | 0.000 | 9 | NA | -10.327 | 9.232 | -53.784 | NA / NA | 9 / 19.559 | 0.111 | UPDATE_BETTER / KEEP_BETTER | True | False |
| LR=5e-3 | 135 | 135 | 2 / 0 | UPDATE | 1.000 | 15 | 12.601 | -1.950 | 18.440 | 21.003 | 3 / 5.838 | 12 / 20.389 | 0.067 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 135 | 135 | 3 / 23 | UPDATE | 1.000 | 9 | -5.103 | -4.296 | 13.756 | 15.879 | 9 / 18.859 | 7 / 18.051 | 0.111 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 135 | 135 | 4 / 59 | UPDATE | 1.000 | 4 | 3.498 | -0.918 | 3.498 | 4.364 | 1 / 0.000 | 3 / 4.415 | 0.250 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 135 | 135 | 5 / RSU_0 | KEEP | 0.000 | 22 | NA | 27.099 | 45.348 | 53.536 | NA / NA | 4 / 18.248 | 0.045 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-3 | 135 | 135 | 6 / RSU_1 | UPDATE | 1.000 | 29 | -8.471 | 2.443 | 26.277 | 25.923 | 27 / 34.748 | 11 / 23.835 | 0.034 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 135 | 135 | 7 / UAV_0 | UPDATE | 1.000 | 28 | -3.329 | 3.313 | 24.415 | 24.339 | 27 / 27.744 | 10 / 21.103 | 0.036 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 135 | 135 | 8 / UAV_1 | UPDATE | 1.000 | 29 | -2.735 | 0.108 | 16.612 | 16.593 | 25 / 19.347 | 11 / 16.504 | 0.034 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 161 | 161 | 0 / 46 | UPDATE | 1.000 | 14 | -1.237 | -1.171 | 1.897 | 0.723 | 10 / 3.135 | 9 / 3.069 | 0.071 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 161 | 161 | 1 / 10 | KEEP | 0.000 | 9 | NA | -0.000 | 3.393 | 15.626 | NA / NA | 8 / 3.393 | 0.111 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-3 | 161 | 161 | 2 / 0 | UPDATE | 1.000 | 10 | 0.555 | -3.217 | 2.435 | 2.294 | 5 / 1.880 | 9 / 5.653 | 0.100 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 161 | 161 | 3 / 23 | UPDATE | 1.000 | 9 | -1.005 | -0.299 | 1.805 | 4.575 | 7 / 2.810 | 5 / 2.104 | 0.111 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 161 | 161 | 4 / 59 | UPDATE | 1.000 | 11 | -1.122 | 1.320 | 4.768 | 7.013 | 10 / 5.890 | 7 / 3.448 | 0.091 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 161 | 161 | 5 / RSU_0 | KEEP | 0.000 | 20 | NA | 16.020 | 28.927 | 34.516 | NA / NA | 2 / 12.907 | 0.050 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-3 | 161 | 161 | 6 / RSU_1 | UPDATE | 1.000 | 27 | -2.187 | 12.454 | 14.184 | 18.252 | 23 / 16.371 | 2 / 1.729 | 0.037 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 161 | 161 | 7 / UAV_0 | UPDATE | 1.000 | 29 | -2.529 | 2.995 | 10.129 | 10.472 | 28 / 12.659 | 7 / 7.134 | 0.034 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 161 | 161 | 8 / UAV_1 | UPDATE | 1.000 | 28 | 0.212 | -0.181 | 7.765 | 7.765 | 19 / 7.553 | 21 / 7.946 | 0.036 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 232 | 232 | 0 / 46 | UPDATE | 1.000 | 8 | -63.107 | -5.488 | -0.196 | 5.475 | 6 / 62.911 | 3 / 5.292 | 0.125 | KEEP_BETTER / UPDATE_BETTER | True | True |
| LR=5e-3 | 232 | 232 | 1 / 10 | KEEP | 0.000 | 9 | NA | 53.111 | 113.643 | -10.600 | NA / NA | 2 / 60.533 | 0.111 | UPDATE_BETTER / KEEP_BETTER | True | False |
| LR=5e-3 | 232 | 232 | 2 / 0 | UPDATE | 1.000 | 13 | 1.958 | 61.428 | 113.959 | -1.760 | 10 / 112.001 | 2 / 52.531 | 0.077 | UPDATE_BETTER / KEEP_BETTER | True | True |
| LR=5e-3 | 232 | 232 | 3 / 23 | UPDATE | 1.000 | 12 | -53.912 | -1.636 | 44.682 | -14.477 | 8 / 98.595 | 2 / 46.319 | 0.083 | UPDATE_BETTER / KEEP_BETTER | True | True |
| LR=5e-3 | 232 | 232 | 4 / 59 | UPDATE | 1.000 | 14 | 10.035 | 60.686 | 122.807 | 6.915 | 5 / 112.771 | 3 / 62.121 | 0.071 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 232 | 232 | 5 / RSU_0 | KEEP | 0.000 | 21 | NA | 75.730 | 125.905 | 2.255 | NA / NA | 5 / 50.176 | 0.048 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-3 | 232 | 232 | 6 / RSU_1 | UPDATE | 1.000 | 29 | -127.363 | -19.405 | 12.247 | 13.270 | 29 / 139.610 | 24 / 31.652 | 0.034 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 232 | 232 | 7 / UAV_0 | UPDATE | 1.000 | 20 | -1.493 | 64.215 | 120.588 | 69.734 | 13 / 122.081 | 4 / 56.373 | 0.050 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 232 | 232 | 8 / UAV_1 | UPDATE | 1.000 | 30 | 9.470 | 62.024 | 116.323 | 116.330 | 12 / 106.853 | 6 / 54.299 | 0.033 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 198 | 198 | 0 / 46 | UPDATE | 1.000 | 10 | -9.073 | -5.745 | -0.394 | 2.703 | 10 / 8.679 | 5 / 5.351 | 0.100 | KEEP_BETTER / UPDATE_BETTER | True | True |
| LR=5e-3 | 198 | 198 | 1 / 10 | KEEP | 0.000 | 9 | NA | -3.987 | 3.060 | -9.081 | NA / NA | 8 / 7.046 | 0.111 | UPDATE_BETTER / KEEP_BETTER | True | False |
| LR=5e-3 | 198 | 198 | 2 / 0 | UPDATE | 1.000 | 13 | -0.615 | 1.042 | 1.433 | 1.523 | 5 / 2.048 | 3 / 0.391 | 0.077 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 198 | 198 | 3 / 23 | UPDATE | 1.000 | 4 | 0.882 | 0.882 | 4.041 | 4.490 | 4 / 3.160 | 4 / 3.160 | 0.250 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 198 | 198 | 4 / 59 | UPDATE | 1.000 | 14 | -2.116 | -2.720 | 5.219 | 5.190 | 11 / 7.334 | 12 / 7.939 | 0.071 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 198 | 198 | 5 / RSU_0 | KEEP | 0.000 | 21 | NA | 47.212 | 50.163 | 49.206 | NA / NA | 2 / 2.951 | 0.048 | UPDATE_BETTER / UPDATE_BETTER | False | False |
| LR=5e-3 | 198 | 198 | 6 / RSU_1 | UPDATE | 1.000 | 34 | -2.935 | 6.259 | 7.880 | 9.527 | 21 / 10.815 | 3 / 1.621 | 0.029 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 198 | 198 | 7 / UAV_0 | UPDATE | 1.000 | 25 | -3.778 | 5.859 | 7.135 | 5.145 | 15 / 10.913 | 3 / 1.276 | 0.040 | UPDATE_BETTER / UPDATE_BETTER | False | True |
| LR=5e-3 | 198 | 198 | 8 / UAV_1 | UPDATE | 1.000 | 24 | 5.833 | -0.926 | 11.418 | 11.415 | 7 / 5.586 | 16 / 12.344 | 0.042 | UPDATE_BETTER / UPDATE_BETTER | False | True |
## Interpretation

- `CACHE_TRIGGER_REFERENCE_COUPLING_PROBLEM = SUPPORTED`. Policy argmax→KEEP while best-exact→UPDATE occurred 67 times; total policy/best sign flips were 67/144. This is evidence about reference coupling on this small batch, not a population causal estimate.
- `FIXED_DOWNSTREAM_COUNTERFACTUAL_BIAS = SUPPORTED`. Fixed and deterministic policy-consistent signs differed 11/144 times. This indicates whether the one-head ceteris-paribus ranking was sensitive to downstream replay on these transitions.
- `CACHE_TRIGGER_STATE_RESPONSE_LOCKIN = SUPPORTED`. A head qualifies only with at least three analyzed states, p_update on the same extreme side (≤0.05 or ≥0.95) in every state, and both KEEP_BETTER and UPDATE_BETTER among fixed or policy-consistent deltas.
- `CACHE_CANDIDATE_STARVATION = SUPPORTED`. A head qualifies only when all analyzed legal rows have p_update≤0.05, no sampled UPDATE occurs, and the candidate-policy argmax has exact rank >1 or positive exact regret. Qualifying heads: [('LR=5e-4', 0), ('LR=5e-4', 1), ('LR=5e-4', 2), ('LR=5e-4', 3), ('LR=5e-4', 4), ('LR=5e-4', 7), ('LR=5e-4', 8), ('LR=5e-3', 1), ('LR=5e-3', 5)].

`GLOBAL_CRITIC_HEALTH_INTERPRETATION = NOT_ASSESSED_SEPARATELY`; all Q comparisons use the same frozen global critic bootstrap. This batch does not establish critic calibration or overall critic health.
`COUNTERFACTUAL_VALUE_DIFFERENCE_ACCURACY = sign agreement 0.924 on 144 legal rows; mean/max |delta_fixed - delta_policy_consistent| = 6.929/124.243. These measure replay disagreement, not accuracy against stochastic Q^π.
`NEXT_REQUIRED_ALGORITHM_CHANGE = 改为 action-independent 的 `Q_keep vs best-update-value` Trigger credit。 正式 credit 使用 policy-consistent downstream evaluator 或经验证的近似。 把 Cache Candidate 从 global GAE 改为 candidate-specific value/ranking credit。

## Required output summary

```text
TRAINING_EXECUTED = NO
CHECKPOINTS_ANALYZED = LR=5e-4:233, LR=5e-3:252
TRANSITIONS_ANALYZED = 16
CACHE_HEAD_ROWS_ANALYZED = 144
FULL_EXACT_UPDATE_BRANCHES = 2329 executed / 2329 legal complete UPDATE configurations evaluated
POLICY_ARGMAX_VS_BEST_SIGN_FLIP_RATE = 0.465 (67/144)
SAMPLED_VS_BEST_SIGN_FLIP_RATE = 0.592 (42/71)
FIXED_VS_POLICY_CONSISTENT_SIGN_FLIP_RATE = 0.076 (11/144)
CACHE_TRIGGER_REFERENCE_COUPLING_PROBLEM = SUPPORTED
FIXED_DOWNSTREAM_COUNTERFACTUAL_BIAS = SUPPORTED
CACHE_TRIGGER_STATE_RESPONSE_LOCKIN = SUPPORTED
CACHE_CANDIDATE_STARVATION = SUPPORTED
GLOBAL_CRITIC_HEALTH_INTERPRETATION = NOT_ASSESSED_SEPARATELY
COUNTERFACTUAL_VALUE_DIFFERENCE_ACCURACY = SIGN_AGREEMENT_0.924; MEAN_ABS_DELTA_DIFFERENCE_6.929; MAX_ABS_DELTA_DIFFERENCE_124.243
NEXT_REQUIRED_ALGORITHM_CHANGE = 改为 action-independent 的 `Q_keep vs best-update-value` Trigger credit。 正式 credit 使用 policy-consistent downstream evaluator 或经验证的近似。 把 Cache Candidate 从 global GAE 改为 candidate-specific value/ranking credit。
```

## Reproducibility and cleanup

Selected rollout slots by checkpoint: [('LR=5e-4', [298, 183, 84, 93, 263, 6, 134, 216]), ('LR=5e-3', [61, 11, 248, 113, 135, 161, 232, 198])]
Diagnostic runtime by checkpoint: [('LR=5e-4', 44.1), ('LR=5e-3', 38.1)] seconds for the full-exact batch.
The report and the single maintained runner are the only new persistent task outputs. No checkpoint, formal run artifact, JSONL, or training output was modified. No temporary script or synthetic file was retained.

CHANGED_FILES = src/ppo_simple/cache_credit.py; scripts/diagnostics/run_cache_trigger_causal_gate.py; reports/cache_trigger_causal_gate.md
DELETED_OBSOLETE_SOURCE_FILES = NONE
VALIDATIONS = Python compile (both touched Python files) PASS; synthetic Q_update−Q_keep sign check PASS; real checkpoint 1 transition × 1 head full-exact smoke PASS; causal gate 2 checkpoints × 8 transitions × 9 heads PASS; git diff --check PASS
TEMP_FILES_REMOVED = no temporary scripts or synthetic files were created; six __pycache__ directories removed after validation
GIT_DIFF_CHECK = PASS
COMMIT = NO
PUSH = NO
