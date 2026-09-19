# R6 output-first full root-cause decomposition

本报告只分析现有 R6 epoch-1 checkpoint 的一次固定 300-slot production rollout。没有训练、optimizer.step、算法修改、epoch2、formal evaluation 或 final seeds。所有 exact counterfactual 均经过正式 execute_slot、Matching/KKT、reward 与 next-state 路径；报告只保存 state-level CSV，不保存逐 branch dump。

## Input and execution gate

R6_RUN_DIR = `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_recovery_r6_true_binary_cache_trigger_epoch1_20260919`
CHECKPOINT = `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_recovery_r6_true_binary_cache_trigger_epoch1_20260919\latest_checkpoint.pt`
FROZEN_REALIZATION = `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v6_popart_actor1e4_lazy_episode_pool_20260911\episode_20001`
POLICY_SCHEMA = adaptive_joint_cache_replica_aware_uav_operation_recovery_r6
COMPLETED_EPISODES = 1
COMPLETED_TRAINING_EPISODES = 1
NEXT_POOL_INDEX = 1
R5_R4_CHECKPOINT_LOADED = NO
PRODUCTION_ROLLOUT_COUNT = 1
PRODUCTION_ROLLOUT_SLOTS = 300
PRODUCTION_MATCHING_KKT_SUMMARIES_RETAINED = YES

## Cache full-population result

Classification tolerance = 1e-8. Candidate bottleneck fraction is reported primarily over states with a beneficial UPDATE; the all-state fraction is also shown.
CACHE_TOTAL_NODE_STATES = 2700
CACHE_STATES_WITH_BENEFICIAL_UPDATE = 2493
CACHE_NO_BENEFICIAL_UPDATE_COUNT = 207
CACHE_EXPECTED_KEEP_STATE_COUNT = 1557
CACHE_EXPECTED_UPDATE_STATE_COUNT = 1143
CACHE_CANDIDATE_BOTTLENECK_COUNT = 1350
CACHE_CANDIDATE_BOTTLENECK_FRACTION = 0.54151625
CACHE_CANDIDATE_BOTTLENECK_ALL_STATE_FRACTION = 0.50000000
CACHE_TRIGGER_CORRECT_COUNT = 1265
CACHE_TRIGGER_CORRECT_FRACTION = 0.46851852
CACHE_TRIGGER_OVER_UPDATE_COUNT = 875
CACHE_TRIGGER_UNDER_UPDATE_COUNT = 560
CACHE_MEAN_GOOD_CANDIDATE_MASS = 0.33593453
CACHE_MEDIAN_GOOD_CANDIDATE_MASS = 0.28528720
CACHE_MEAN_BEST_CANDIDATE_PROBABILITY = 0.08204868
CACHE_MEAN_Q_BEST_MINUS_Q_UPDATE = 8.51233859
MEAN_P_UPDATE_WHEN_UPDATE_BETTER = 0.49501089
MEAN_P_KEEP_WHEN_KEEP_BETTER = 0.49858447
CACHE_R_Q_SIGN_MISMATCH_COUNT = 29
CACHE_MEAN_ABS_DELTA_R = 2.73272932
CACHE_MEAN_ABS_GAMMA_DELTA_V = 0.01460733
CACHE_MEAN_ABS_DELTA_Q = 2.73261630
CACHE_ACTUAL_KEEP_COUNT = 1409
CACHE_ACTUAL_UPDATE_COUNT = 1291
CACHE_ACTUAL_UPDATE_RATE = 0.47814815
ACTUAL_CACHE_TRIGGER_CORRECT_COUNT = 1308

### Cache value/trigger classification counts

```json
{
  "trigger_quality": {
    "CORRECT_KEEP": 682,
    "CORRECT_UPDATE": 583,
    "TRIGGER_OVER_UPDATE": 875,
    "TRIGGER_UNDER_UPDATE": 560
  },
  "value_structure": {
    "BENEFICIAL_UPDATE": 1143,
    "CANDIDATE_BOTTLENECK": 1350,
    "NO_BENEFICIAL_UPDATE": 207
  }
}
```

### Cache by node role

```json
{
  "RSU": {
    "actual_keep": 280,
    "actual_trigger_correct": 286,
    "actual_update": 320,
    "actual_update_rate": 0.5333333333333333,
    "beneficial": 594,
    "candidate_bottleneck": 388,
    "candidate_bottleneck_all_fraction": 0.6466666666666666,
    "candidate_bottleneck_fraction": 0.6531986531986532,
    "keep_better": 394,
    "mean_abs_delta_q": 4.24548617512999,
    "mean_abs_delta_r": 4.244105998678078,
    "mean_abs_gamma_delta_v": 0.012727779833344688,
    "mean_best_candidate_probability": 0.0445145862302445,
    "mean_good_candidate_mass": 0.321550699537275,
    "mean_p_keep_when_keep_better": 0.45061404844225966,
    "mean_p_update_when_update_better": 0.5579227981636825,
    "mean_q_best_minus_q_update": 10.721784604010677,
    "median_good_candidate_mass": 0.3028790419921279,
    "no_beneficial": 6,
    "operation_class_counts": {
      "CORRECT_KEEP": 16,
      "CORRECT_UPDATE": 201,
      "TRIGGER_OVER_UPDATE": 378,
      "TRIGGER_UNDER_UPDATE": 5
    },
    "rq_sign_mismatch": 2,
    "total": 600,
    "trigger_correct": 217,
    "trigger_correct_fraction": 0.3616666666666667,
    "trigger_over": 378,
    "trigger_under": 5,
    "update_better": 206,
    "value_class_counts": {
      "BENEFICIAL_UPDATE": 206,
      "CANDIDATE_BOTTLENECK": 388,
      "NO_BENEFICIAL_UPDATE": 6
    }
  },
  "SV": {
    "actual_keep": 824,
    "actual_trigger_correct": 729,
    "actual_update": 676,
    "actual_update_rate": 0.45066666666666666,
    "beneficial": 1300,
    "candidate_bottleneck": 695,
    "candidate_bottleneck_all_fraction": 0.4633333333333333,
    "candidate_bottleneck_fraction": 0.5346153846153846,
    "keep_better": 895,
    "mean_abs_delta_q": 2.243189755290291,
    "mean_abs_delta_r": 2.2435895537954176,
    "mean_abs_gamma_delta_v": 0.015211657510197929,
    "mean_best_candidate_probability": 0.11082271812626948,
    "mean_good_candidate_mass": 0.30821724752394053,
    "mean_p_keep_when_keep_better": 0.5127711948402767,
    "mean_p_update_when_update_better": 0.4801686807604861,
    "mean_q_best_minus_q_update": 6.33436599482764,
    "median_good_candidate_mass": 0.25373732671141624,
    "no_beneficial": 200,
    "operation_class_counts": {
      "CORRECT_KEEP": 488,
      "CORRECT_UPDATE": 250,
      "TRIGGER_OVER_UPDATE": 407,
      "TRIGGER_UNDER_UPDATE": 355
    },
    "rq_sign_mismatch": 23,
    "total": 1500,
    "trigger_correct": 738,
    "trigger_correct_fraction": 0.492,
    "trigger_over": 407,
    "trigger_under": 355,
    "update_better": 605,
    "value_class_counts": {
      "BENEFICIAL_UPDATE": 605,
      "CANDIDATE_BOTTLENECK": 695,
      "NO_BENEFICIAL_UPDATE": 200
    }
  },
  "UAV": {
    "actual_keep": 305,
    "actual_trigger_correct": 293,
    "actual_update": 295,
    "actual_update_rate": 0.49166666666666664,
    "beneficial": 599,
    "candidate_bottleneck": 267,
    "candidate_bottleneck_all_fraction": 0.445,
    "candidate_bottleneck_fraction": 0.44574290484140233,
    "keep_better": 268,
    "mean_abs_delta_q": 2.4433127717892695,
    "mean_abs_delta_r": 2.444202049273372,
    "mean_abs_gamma_delta_v": 0.014976047522568882,
    "mean_best_candidate_probability": 0.05682163722564362,
    "mean_good_candidate_mass": 0.4103526802009006,
    "mean_p_keep_when_keep_better": 0.5217308348684169,
    "mean_p_update_when_update_better": 0.4830219798597945,
    "mean_q_best_minus_q_update": 11.048154008095649,
    "median_good_candidate_mass": 0.4033457413315773,
    "no_beneficial": 1,
    "operation_class_counts": {
      "CORRECT_KEEP": 178,
      "CORRECT_UPDATE": 132,
      "TRIGGER_OVER_UPDATE": 90,
      "TRIGGER_UNDER_UPDATE": 200
    },
    "rq_sign_mismatch": 4,
    "total": 600,
    "trigger_correct": 310,
    "trigger_correct_fraction": 0.5166666666666667,
    "trigger_over": 90,
    "trigger_under": 200,
    "update_better": 332,
    "value_class_counts": {
      "BENEFICIAL_UPDATE": 332,
      "CANDIDATE_BOTTLENECK": 267,
      "NO_BENEFICIAL_UPDATE": 1
    }
  }
}
```

### Cache critic interpretation

CACHE_BOOTSTRAP_ABS_RATIO = 0.00534533
The R/V/Q decomposition is descriptive only; it does not by itself label the critic as incorrect.

## UAV full-population result

UAV_TOTAL_OPERATION_STATES = 600
UAV_STATES_WITH_BENEFICIAL_MOVE = 321
UAV_NO_BENEFICIAL_MOVE_COUNT = 279
UAV_EXPECTED_MOVE_BETTER_COUNT = 0
UAV_EXPECTED_STAY_BETTER_COUNT = 600
UAV_CANDIDATE_BOTTLENECK_COUNT = 321
UAV_CANDIDATE_BOTTLENECK_FRACTION = 1.00000000
UAV_CANDIDATE_BOTTLENECK_ALL_STATE_FRACTION = 0.53500000
UAV_OPERATION_CORRECT_COUNT = 600
UAV_OPERATION_CORRECT_FRACTION = 1.00000000
UAV_OPERATION_UNDER_MOVE_COUNT = 0
UAV_OPERATION_OVER_MOVE_COUNT = 0
UAV_MEAN_GOOD_MOVE_CANDIDATE_MASS = 0.24368405
UAV_MEDIAN_GOOD_MOVE_CANDIDATE_MASS = 0.23337016
UAV_MEAN_BEST_MOVE_PROBABILITY = 0.01181098
UAV_MEAN_Q_BEST_MOVE_MINUS_Q_MOVE = 0.00836445
UAV_R_Q_SIGN_MISMATCH_COUNT = 0
UAV_MEAN_ABS_DELTA_R_MOVE = 0.00201260
UAV_MEAN_ABS_GAMMA_DELTA_V_MOVE = 0.00000810
UAV_MEAN_ABS_DELTA_Q_MOVE = 0.00201153
UAV_ACTUAL_STAY_COUNT = 476
UAV_ACTUAL_MOVE_COUNT = 124
UAV_ACTUAL_MOVE_RATE = 0.20666667
ACTUAL_UAV_OPERATION_CORRECT_COUNT = 476

### UAV by head

```json
{
  "UAV_0": {
    "actual_move": 73,
    "actual_move_rate": 0.24333333333333335,
    "actual_operation_correct": 227,
    "actual_stay": 227,
    "beneficial": 142,
    "candidate_bottleneck": 142,
    "candidate_bottleneck_all_fraction": 0.47333333333333333,
    "candidate_bottleneck_fraction": 1.0,
    "mean_abs_delta_q_move": 0.002020177642504374,
    "mean_abs_delta_r_move": 0.0020207762706015596,
    "mean_abs_gamma_delta_v_move": 7.521316724452739e-06,
    "mean_best_move_probability": 0.011763469361206076,
    "mean_good_move_candidate_mass": 0.21016210710056094,
    "mean_q_best_move_minus_q_move": 0.006536970728650755,
    "median_good_move_candidate_mass": 0.18184087323812953,
    "move_better": 0,
    "no_beneficial": 158,
    "operation_class_counts": {
      "EXPECTED_STAY": 158,
      "UAV_CANDIDATE_BOTTLENECK": 142
    },
    "operation_correct": 300,
    "operation_correct_fraction": 1.0,
    "operation_over": 0,
    "operation_under": 0,
    "rq_sign_mismatch": 0,
    "stay_better": 300,
    "total": 300
  },
  "UAV_1": {
    "actual_move": 51,
    "actual_move_rate": 0.17,
    "actual_operation_correct": 249,
    "actual_stay": 249,
    "beneficial": 179,
    "candidate_bottleneck": 179,
    "candidate_bottleneck_all_fraction": 0.5966666666666667,
    "candidate_bottleneck_fraction": 1.0,
    "mean_abs_delta_q_move": 0.0020028908054033913,
    "mean_abs_delta_r_move": 0.0020044298153633045,
    "mean_abs_gamma_delta_v_move": 8.6746118875769e-06,
    "mean_best_move_probability": 0.01184867870338607,
    "mean_good_move_candidate_mass": 0.2702768675750521,
    "mean_q_best_move_minus_q_move": 0.009814177770029828,
    "median_good_move_candidate_mass": 0.2726112495763751,
    "move_better": 0,
    "no_beneficial": 121,
    "operation_class_counts": {
      "EXPECTED_STAY": 121,
      "UAV_CANDIDATE_BOTTLENECK": 179
    },
    "operation_correct": 300,
    "operation_correct_fraction": 1.0,
    "operation_over": 0,
    "operation_under": 0,
    "rq_sign_mismatch": 0,
    "stay_better": 300,
    "total": 300
  }
}
```

### UAV critic interpretation

UAV_BOOTSTRAP_ABS_RATIO = 0.00402363
The R/V/Q decomposition is descriptive only; it does not by itself label the critic as incorrect.

## Actual production actions

Actual actions are read from the retained production conditional-action records, not inferred from probabilities.
ACTUAL_KEEP_COUNT = 1409
ACTUAL_UPDATE_COUNT = 1291
ACTUAL_UPDATE_RATE = 0.47814815
ACTUAL_CACHE_TRIGGER_CORRECT_COUNT = 1308
ACTUAL_STAY_COUNT = 476
ACTUAL_MOVE_COUNT = 124
ACTUAL_MOVE_RATE = 0.20666667
ACTUAL_UAV_OPERATION_CORRECT_COUNT = 476

## Time windows

### Cache

| window | candidate bottleneck fraction (beneficial denominator) | trigger correct fraction | over-update fraction | under-update fraction |
|---|---:|---:|---:|---:|
| 0-59 | 0.38844622 | 0.50185185 | 0.25185185 | 0.24629630 |
| 60-119 | 0.57684631 | 0.43703704 | 0.36851852 | 0.19444444 |
| 120-179 | 0.59443340 | 0.45000000 | 0.35925926 | 0.19074074 |
| 180-239 | 0.51911469 | 0.48518519 | 0.30925926 | 0.20555556 |
| 240-299 | 0.63061224 | 0.46851852 | 0.33148148 | 0.20000000 |

### UAV

| window | candidate bottleneck fraction (beneficial denominator) | operation correct fraction | under-move fraction | over-move fraction |
|---|---:|---:|---:|---:|
| 0-59 | 1.00000000 | 1.00000000 | 0.00000000 | 0.00000000 |
| 60-119 | 1.00000000 | 1.00000000 | 0.00000000 | 0.00000000 |
| 120-179 | 1.00000000 | 1.00000000 | 0.00000000 | 0.00000000 |
| 180-239 | 1.00000000 | 1.00000000 | 0.00000000 | 0.00000000 |
| 240-299 | 1.00000000 | 1.00000000 | 0.00000000 | 0.00000000 |

## Representative failure states (maximum numerical severity, at most five each)

### Cache candidate bottleneck

| slot | node | role | p_update | Q_keep | Q_update | Q_best | candidate bottleneck gap | actual | |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 265 | UAV_1 | UAV | 0.48716733 | -111.19184576 | -208.32017517 | -111.12140417 | 97.19877100 | UPDATE |
| 266 | RSU_0 | RSU | 0.53220445 | -169.33274342 | -174.50202942 | -108.29089323 | 66.21113618 | UPDATE |
| 163 | RSU_0 | RSU | 0.58131599 | -110.05944078 | -172.77665710 | -109.99438280 | 62.78227431 | KEEP |
| 293 | RSU_0 | RSU | 0.51819116 | -166.58884246 | -171.27032471 | -110.85966117 | 60.41066354 | UPDATE |
| 29 | UAV_0 | UAV | 0.41325077 | -172.47178111 | -178.32101440 | -122.36581080 | 55.95520361 | UPDATE |

### Cache trigger over-update

| slot | node | role | p_update | Q_keep | Q_update | Q_best | candidate bottleneck gap | actual | |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 74 | UAV_1 | UAV | 0.53744709 | -109.11428029 | -109.11633301 | -107.09159351 | 2.02473950 | KEEP |
| 109 | RSU_0 | RSU | 0.56976163 | -123.91868294 | -123.92090607 | -119.45495507 | 4.46595100 | KEEP |
| 27 | 59 | SV | 0.54731214 | -260.06995909 | -260.07351685 | -260.06671232 | 0.00680452 | UPDATE |
| 26 | 59 | SV | 0.55116779 | -208.00785550 | -208.01210022 | -208.00369321 | 0.00840701 | KEEP |
| 299 | 59 | SV | 0.51240224 | -0.35591693 | -0.36021143 | 0.31649450 | 0.67670593 | UPDATE |

### Cache trigger under-update

| slot | node | role | p_update | Q_keep | Q_update | Q_best | candidate bottleneck gap | actual | |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 39 | 23 | SV | 0.38701099 | -235.18205480 | -120.55037689 | -106.16595921 | 14.38441768 | KEEP |
| 25 | 23 | SV | 0.39153290 | -159.98156223 | -113.65542603 | -107.42782087 | 6.22760516 | KEEP |
| 75 | 23 | SV | 0.39368892 | -139.21788938 | -115.84615326 | -108.85991934 | 6.98623392 | KEEP |
| 289 | 0 | SV | 0.49114659 | -136.31483862 | -113.33351135 | -108.60972773 | 4.72378362 | KEEP |
| 227 | 23 | SV | 0.37885180 | -128.29756857 | -106.53092957 | -106.05222049 | 0.47870908 | UPDATE |

### UAV candidate bottleneck

| slot | node | p_move | Q_stay | Q_move | Q_best_move | gap | actual | |
|---:|---|---:|---:|---:|---:|---:|---|
| 30 | UAV_1 | 0.15162448 | -128.46987915 | -128.47207642 | -128.39045084 | 0.08162557 | STAY |
| 263 | UAV_1 | 0.17431631 | -111.74518585 | -111.74728394 | -111.68812824 | 0.05915569 | STAY |
| 142 | UAV_1 | 0.16594108 | -111.59453583 | -111.59665680 | -111.53905629 | 0.05760050 | STAY |
| 265 | UAV_1 | 0.17102181 | -111.30187225 | -111.30394745 | -111.25244840 | 0.05149905 | STAY |
| 9 | UAV_0 | 0.23302364 | -119.39445496 | -119.39655304 | -119.34917417 | 0.04737887 | STAY |

### UAV operation wrong-side

| slot | node | p_move | Q_stay | Q_move | Q_best_move | gap | actual | |
|---:|---|---:|---:|---:|---:|---:|---|

## Root-cause classification

The ranking below is computed from the full-population fractions and the bootstrap dominance check; it is not inherited from earlier gates.
CACHE_ROOT_CAUSE_SCORE = {"CACHE_CANDIDATE_POLICY": 0.5415162454873647, "CACHE_CRITIC_BOOTSTRAP": 0.0, "CACHE_ENVIRONMENT_LANDSCAPE": 0.07666666666666666, "CACHE_TRIGGER_POLICY": 0.5314814814814814}
UAV_ROOT_CAUSE_SCORE = {"UAV_CANDIDATE_POLICY": 1.0, "UAV_CRITIC_BOOTSTRAP": 0.0, "UAV_ENVIRONMENT_LANDSCAPE": 0.465, "UAV_OPERATION_POLICY": 0.0}
CACHE_PRIMARY_ROOT_CAUSE = CACHE_CANDIDATE_POLICY
CACHE_SECONDARY_ROOT_CAUSE = CACHE_TRIGGER_POLICY
UAV_PRIMARY_ROOT_CAUSE = UAV_CANDIDATE_POLICY
UAV_SECONDARY_ROOT_CAUSE = UAV_ENVIRONMENT_LANDSCAPE

Interpretation rule: candidate-policy scores use bottleneck fraction among beneficial states; environment scores use no-beneficial fraction; trigger/operation scores use wrong-side fraction; critic-bootstrap scores are enabled only when sign mismatch and bootstrap magnitude are both material.

## Runtime and artifacts

WALL_CLOCK_RUNTIME_SECONDS = 1831.639
CACHE_CF_RUNTIME_SECONDS = 815.700
CACHE_CF_ENV_EXECUTIONS = 40553
CACHE_CF_LOGICAL_BRANCHES = 42953
CACHE_CF_DEDUP_REUSED_BRANCHES = 2400
UAV_CF_RUNTIME_SECONDS = 897.322
UAV_CF_LOGICAL_BRANCHES = 48538
UAV_CF_ENV_EXECUTIONS = 48238
UAV_CF_DEDUP_REUSED_BRANCHES = 300
CACHE_ANALYSIS_WALL_SECONDS = 847.203
UAV_ANALYSIS_WALL_SECONDS = 928.642
CSV_SUMMARY = `C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\r6_output_first_full_root_cause_summary.csv`

## Final control fields

TRAINING_EXECUTED = NO
OPTIMIZER_STEP_EXECUTED = NO
SOURCE_ALGORITHM_CHANGED = NO
REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
MATCHING_KKT_CHANGED = NO
CACHE_POLICY_CHANGED = NO
UAV_POLICY_CHANGED = NO
CRITIC_CHANGED = NO
EPOCH2_EXECUTED = NO
FORMAL_EVALUATION_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
