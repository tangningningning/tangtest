# Cache system root-cause triage

This is a read-only diagnostic of the frozen V10-C 5e-4, 300-epoch checkpoint. No PPO update, F=20 change, V10-D change, final unseen seed, GRU retraining, or full action-space enumeration was executed.

## Scope and alignment

- Checkpoint: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v10c_decoupled_encoder_5e4_formal\latest_checkpoint.pt` (sha256 `0cac00fdb0da339972923d8aabee18a1612d28baeed34e461b39ce486c9394e8`; loaded cursor completed episodes `300`).
- Config: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\configs\proposed\system_config_ppo_v10c_decoupled_encoder.toml` (sha256 `9f3c8764652cb0622997074a2ea0d9dce4e01e09c97766df6b8133dc095e746c`).
- Episodes: `episode_20041, episode_20101, episode_20161, episode_20221, episode_20281`; each replayed for `300` slots.
- Popularity alignment: the production code constructs causal `rho_obs` history but trains the GRU target on future `rho_star`; primary metrics below compare each prediction to the validated `target_slot=origin+horizon` `rho_star` vector. `rho_obs` is retained as a supplementary comparison.
- Position alignment: each predicted vehicle position is compared with the trajectory row at the same validated target slot; no one-slot shift is applied.

## Cache pressure and request availability

- Mean network catalog coverage: `0.6002`; P95: `0.6`; full-coverage slot fraction: `0`.
- Mean replicas per service-slot: `1.32727`; request-weighted edge replicas: `1.20333`.
- Requested-service fractions: absent `0.433922`, single replica `0.155386`, multi-replica `0.410691`, present on all edge nodes `0`.
- Cache-limited task fraction: `1`. This compares actual physical edge feasibility with the same set gated by the final cache state; it does not alter the solver.

| Node type | Mean capacity utilization | P95 capacity utilization | Mean cached services | Actual update nodes |
| --- | ---: | ---: | ---: | ---: |
| RSU | 0.004169 | 0 | 0.0266667 | 45 |
| SV | 0.498608 | 0.931304 | 1.24373 | 5107 |
| UAV | 0.846972 | 0.904192 | 3.50033 | 1498 |

| Episode | Mean coverage | Full coverage fraction | Mean replicas/service | Mean replicas/request | Popularity MAE | Position mean error (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| episode_20041 | 0.600333 | 0 | 1.34333 | 1.59429 | 0.0202104 | 10.9486 |
| episode_20101 | 0.600333 | 0 | 1.33 | 0.976202 | 0.0237926 | 11.352 |
| episode_20161 | 0.6 | 0 | 1.307 | 1.15968 | 0.0224564 | 10.9123 |
| episode_20221 | 0.600333 | 0 | 1.339 | 1.31448 | 0.0187492 | 11.2264 |
| episode_20281 | 0.6 | 0 | 1.317 | 0.966537 | 0.019414 | 11.0821 |

Actual Cache changes: `1500` slots, `6650` nodes, `9649` changed service bits; cache changes altered cached feasible edge sets for `8539` task occurrences.

## Prediction diagnostics

Primary `rho_star` metrics over all valid origin/horizon/service records: MAE `0.0209245`, RMSE `0.0297424`, mean vector L1 `0.209245`, Top-1 `0.928322`, Top-3 overlap `0.734899`.
Supplementary same-target `rho_obs` comparison: MAE `0.0649585`, RMSE `0.0840496`, mean vector L1 `0.649585`, Top-1 `0.563221`, Top-3 overlap `0.578613`.

| Position category | Mean Euclidean error (m) | RMSE (m) | P95 (m) | Samples |
| --- | ---: | ---: | ---: | ---: |
| all_vehicles | 11.1043 | 14.5999 | 29.987 | 186250 |
| service_vehicles | 10.8338 | 14.2863 | 29.2004 | 37250 |
| task_vehicles | 11.1719 | 14.6773 | 30.1864 | 149000 |

## Targeted Cache UPDATE/KEEP witnesses

Each record starts from the same pre-action Cache/UAV state. At the target node only, KEEP sets the trigger to 0 and retains the pre-action cache; UPDATE sets the trigger to 1 and uses the current policy's valid update operation/candidate score. Other Cache actions, UAV actions, workload, trajectory, and the frozen policy are held fixed at the intervention slot; from the next slot onward the frozen policy acts again separately on each branch.

| Episode | Slot | Target | Original policy | ΔG30 | ΔQ30 | Class |
| --- | ---: | --- | --- | ---: | ---: | --- |
| episode_20041 | 60 | 17 (SV) | UPDATE | -0.00545691 | -1.03768 | OVER_UPDATE_WITNESS |
| episode_20041 | 120 | RSU_0 (RSU) | KEEP | 0.173717 | 0.263707 | MISSED_UPDATE_WITNESS |
| episode_20041 | 180 | UAV_1 (UAV) | UPDATE | 1.91728 | 0.74702 | DECISION_CONSISTENT_WITNESS |
| episode_20101 | 60 | 7 (SV) | UPDATE | 0.775346 | 0.916315 | DECISION_CONSISTENT_WITNESS |
| episode_20101 | 120 | RSU_0 (RSU) | KEEP | -0.000196224 | 0 | DECISION_CONSISTENT_WITNESS |
| episode_20101 | 180 | UAV_1 (UAV) | UPDATE | 0.620626 | 1.12646 | DECISION_CONSISTENT_WITNESS |
| episode_20161 | 60 | 52 (SV) | UPDATE | 1.06352 | 0.178762 | DECISION_CONSISTENT_WITNESS |
| episode_20161 | 120 | RSU_0 (RSU) | KEEP | 0.137676 | 0.227666 | MISSED_UPDATE_WITNESS |
| episode_20161 | 180 | UAV_1 (UAV) | UPDATE | 7.28481 | 7.79064 | DECISION_CONSISTENT_WITNESS |
| episode_20221 | 60 | 0 (SV) | UPDATE | -0.139986 | 0 | OVER_UPDATE_WITNESS |
| episode_20221 | 120 | RSU_0 (RSU) | KEEP | -0.222503 | 0 | DECISION_CONSISTENT_WITNESS |
| episode_20221 | 180 | UAV_1 (UAV) | UPDATE | -4.82853 | -3.46219 | OVER_UPDATE_WITNESS |

Policy KEEP states: `4`; UPDATE states: `8`; UPDATE_NOOP: `0`. Missed-update witnesses: `2`; over-update witnesses: `3`; decision-consistent: `7`.

The witness comparison is descriptive and targeted; it is not a global proof about the Cache policy. Raw branch values include G/Q at H=15 and H=30, Cache switching cost, delay, energy, cloud task count/rate, trigger-vs-actual flags, prediction errors, coverage, replica count, target capacity utilization, trigger probability, and demand shift in the JSON.

## Root-cause decisions

- `CACHE_SCENARIO_PRESSURE_INSUFFICIENT = NO`. Conditions used: coverage >= 0.95, full-coverage fraction >= 0.80, multi-replica request fraction >= 0.50, and at least 75% of targeted |ΔG30| values no larger than 10% of the median branch-return scale.
- `PREDICTION_QUALITY_CAUSE = NO`. Wrong-witness concentration is compared descriptively with decision-consistent states; the small sample forbids causal attribution.
- `CACHE_POLICY_DECISION_ERROR = SUPPORTED`. Supported requires at least two same-class wrong witnesses across at least two episodes.
- `PPO_UPDATE_CONTROL_REMAINS_CANDIDATE = YES`. Prior cross-run KL suppression evidence was available: `True`; this remains a candidate, not causal proof.
- `F20_STRESS_TEST_RECOMMENDED = NO`.
- `V10D_FORMAL_TRAINING_RECOMMENDED = YES`.

## Required final fields

STAGE = CACHE_SYSTEM_ROOT_CAUSE_TRIAGE
SYSTEM_MODEL_CHANGED = NO
GLOBAL_OBJECTIVE_CHANGED = NO
PPO_CHANGED = NO
TRAINING_EXECUTED = NO
SERVICE_COUNT = 10
EPISODES_ANALYZED = ['episode_20041', 'episode_20101', 'episode_20161', 'episode_20221', 'episode_20281']
MEAN_NETWORK_CATALOG_COVERAGE = 0.6002
P95_NETWORK_CATALOG_COVERAGE = 0.6
FULL_COVERAGE_SLOT_FRACTION = 0
MEAN_REPLICA_COUNT_PER_SERVICE = 1.32727
MEAN_EDGE_REPLICAS_PER_REQUEST = 1.20333
REQUEST_SERVICE_ABSENT_EDGE_FRACTION = 0.433922
REQUEST_SERVICE_SINGLE_REPLICA_FRACTION = 0.155386
REQUEST_SERVICE_MULTI_REPLICA_FRACTION = 0.410691
SV_MEAN_CAPACITY_UTILIZATION = 0.498608
RSU_MEAN_CAPACITY_UTILIZATION = 0.004169
UAV_MEAN_CAPACITY_UTILIZATION = 0.846972
CACHE_LIMITED_TASK_FRACTION = 1
POPULARITY_MAE = 0.0209245
POPULARITY_RMSE = 0.0297424
POPULARITY_MEAN_L1 = 0.209245
POPULARITY_TOP1_ACCURACY = 0.928322
POPULARITY_TOP3_OVERLAP = 0.734899
POSITION_MEAN_EUCLIDEAN_ERROR_M = 11.1043
POSITION_RMSE_M = 14.5999
POSITION_P95_ERROR_M = 29.987
CACHE_WITNESS_STATES = 12
POLICY_KEEP_STATES = 4
POLICY_UPDATE_STATES = 8
UPDATE_NOOP_COUNT = 0
MISSED_UPDATE_WITNESS_COUNT = 2
OVER_UPDATE_WITNESS_COUNT = 3
DECISION_CONSISTENT_WITNESS_COUNT = 7
MEDIAN_DELTA_G30 = 0.155696
MEAN_ABS_DELTA_G30 = 1.4308
CACHE_SCENARIO_PRESSURE_INSUFFICIENT = NO
PREDICTION_QUALITY_CAUSE = NO
CACHE_POLICY_DECISION_ERROR = SUPPORTED
PPO_UPDATE_CONTROL_REMAINS_CANDIDATE = YES
F20_STRESS_TEST_RECOMMENDED = NO
V10D_FORMAL_TRAINING_RECOMMENDED = YES
F20_CHANGE_EXECUTED = NO
BRANCH_SPECIFIC_KL_CHANGE_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
REPORT = reports/cache_system_root_cause_triage.md
RAW_RESULTS = reports/cache_system_root_cause_triage.json
GIT_DIFF_CHECK = PASS
TEMP_FILES_REMOVED = YES
PUSHED = NO
