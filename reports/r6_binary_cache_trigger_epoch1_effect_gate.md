# R6 binary Cache trigger epoch-1 effect gate

本报告只覆盖 fresh R6 初始化后的一个完整 epoch 与 bounded frozen witness probe；没有 epoch2/epoch3、formal evaluation 或 final-seed evaluation。exact probe 的每个 branch 均经过正式 execute_slot、Matching/KKT、reward 与 next-state 路径。

## 结论字段

R6_EPOCH1_COMPLETED = YES
EPOCH2_EXECUTED = NO
FORMAL_EVALUATION_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
FRESH_INITIALIZATION = YES
R5_CHECKPOINT_LOADED = NO
R6_EPOCH1_CHECKPOINT_LOADED_FOR_PROBE = YES
CHECKPOINT_COMPLETED_EPISODES = 1
CHECKPOINT_COMPLETED_TRAINING_EPISODES = 1
CHECKPOINT_NEXT_POOL_INDEX = 1
CHECKPOINT_CURRENT_TRAIN_ORIGIN = 300
CACHE_TRIGGER_POLICY = TRUE_BINARY_KEEP_UPDATE
CACHE_UPDATE_CANDIDATE_POLICY = UNIFIED_CONDITIONAL_UPDATE_CANDIDATES
CACHE_TRIGGER_ADV_SOURCE = EXACT_NODE_COUNTERFACTUAL_Q
CACHE_CANDIDATE_ADV_SOURCE = EXACT_NODE_CONDITIONAL_COUNTERFACTUAL_Q
EXACT_SEMANTICS_PRESERVED = YES
CANDIDATE_PRUNING = NO
TOP_K_OR_APPROXIMATION = NO

## Frozen witness

FROZEN_EPISODE = episode_20001
TARGET_SLOTS = (20, 80, 140, 200)
TARGET_NODES = ['RSU_0', 'RSU_1']
WITNESS_STATE_COUNT = 8
R6_SCHEMA = adaptive_joint_cache_replica_aware_uav_operation_recovery_r6

## Exact Cache decision quality

KEEP_BETTER_STATE_COUNT = 4
UPDATE_BETTER_STATE_COUNT = 4
KEEP_PREFERRED_AND_POLICY_KEEP_COUNT = 1
UPDATE_PREFERRED_AND_POLICY_UPDATE_COUNT = 4
POLICY_TRIGGER_CORRECT_COUNT = 5/8
MEAN_P_KEEP = 0.4422833808
MEAN_P_UPDATE = 0.5577166192
MEAN_Q_UPDATE_MINUS_Q_KEEP = -0.8572187424
MEAN_CORRECT_TRIGGER_MARGIN = 0.0385968611
POSITIVE_MARGIN_STATE_COUNT = 5/8
MEAN_P_KEEP_ON_KEEP_PREFERRED = 0.4615818039
MEAN_P_UPDATE_ON_UPDATE_PREFERRED = 0.5770150572
MEAN_P_KEEP_ON_KEEP_PREFERRED_STATES = 0.4615818039
MEAN_P_UPDATE_ON_UPDATE_PREFERRED_STATES = 0.5770150572
MEAN_GOOD_UPDATE_CANDIDATE_MASS = 0.3856817116
R6_EPOCH1_EFFECT = NOT_SUPPORTED
R6_EPOCH1_LEARNING_DIRECTION = NOT_SUPPORTED

## Fresh R6 reference

Fresh reference values come from the preceding R6 true-binary correctness gate on the same frozen episode and witness definition.
INITIAL_MEAN_P_KEEP = 0.4776745713
INITIAL_MEAN_P_UPDATE = 0.5223254287
FRESH_INITIAL_MEAN_P_KEEP = 0.4776745713
FRESH_INITIAL_MEAN_P_UPDATE = 0.5223254287
FRESH_WITNESS_MEAN_P_KEEP = 0.4851456918
FRESH_WITNESS_MEAN_P_UPDATE = 0.5148543268
FRESH_KEEP_PREFERRED_STATE_COUNT = 4
FRESH_UPDATE_PREFERRED_STATE_COUNT = 4
FRESH_MEAN_CORRECT_TRIGGER_MARGIN = 0.0140974261
FRESH_POSITIVE_MARGIN_STATE_COUNT = 5/8
FRESH_MEAN_P_KEEP_ON_KEEP_PREFERRED = 0.4921944067
FRESH_MEAN_P_UPDATE_ON_UPDATE_PREFERRED = 0.5219030380
MEAN_P_KEEP_CHANGE_FROM_FRESH = -0.0428623110
MEAN_P_UPDATE_CHANGE_FROM_FRESH = 0.0428622924
MEAN_P_KEEP_CHANGE_FROM_INITIAL_GLOBAL = -0.0353911905
MEAN_P_UPDATE_CHANGE_FROM_INITIAL_GLOBAL = 0.0353911905
MEAN_CORRECT_TRIGGER_MARGIN_CHANGE_FROM_FRESH = 0.0244994350
POSITIVE_MARGIN_STATE_COUNT_CHANGE_FROM_FRESH = 0

## Per-state result

| slot | node | p_keep | p_update | Q_keep | Q_update | DeltaQ | preferred | correct | correct margin | legal UPDATE candidates | good UPDATE mass |
|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| 20 | RSU_0 | 0.4176064730 | 0.5823935270 | -116.0872039795 | -115.6569137573 | 0.4302902222 | UPDATE | YES | 0.1647870541 | 31 | 0.3616928700 |
| 20 | RSU_1 | 0.4253387451 | 0.5746612549 | -116.2866821289 | -113.1075363159 | 3.1791458130 | UPDATE | YES | 0.1493225098 | 14 | 0.6564376652 |
| 80 | RSU_0 | 0.4060811102 | 0.5939189196 | -124.5324401855 | -124.4462509155 | 0.0861892700 | UPDATE | YES | 0.1878378093 | 16 | 0.3794754073 |
| 80 | RSU_1 | 0.4282316267 | 0.5717683434 | -110.0656509399 | -112.5176239014 | -2.4519729614 | KEEP | NO | -0.1435367167 | 33 | 0.0875996817 |
| 140 | RSU_0 | 0.4179856479 | 0.5820143819 | -107.5720291138 | -109.2138900757 | -1.6418609619 | KEEP | NO | -0.1640287340 | 27 | 0.2659671828 |
| 140 | RSU_1 | 0.4974617064 | 0.5025382638 | -110.9717941284 | -117.1213455200 | -6.1495513916 | KEEP | NO | -0.0050765574 | 24 | 0.1704495288 |
| 200 | RSU_0 | 0.4429135025 | 0.5570865273 | -126.5180892944 | -121.2977523804 | 5.2203369141 | UPDATE | YES | 0.1141730249 | 25 | 0.6442990340 |
| 200 | RSU_1 | 0.5026482344 | 0.4973517358 | -126.5483016968 | -132.0786285400 | -5.5303268433 | KEEP | YES | 0.0052964985 | 11 | 0.5195323229 |

## Runtime accounting

WALL_CLOCK_RUNTIME_SECONDS = 2004.1
TRAINING_ELAPSED_SECONDS = 2004.1
EXACT_CACHE_CF_RUNTIME_SECONDS = 3.3343377000
PROBE_WALL_CLOCK_SECONDS = 23.3632503000
CACHE_CF_ENV_EXECUTIONS = 185
CACHE_CF_LOGICAL_BRANCHES = 189
CACHE_CF_DEDUP_REUSED_BRANCHES = 4

## Execution-control audit

TRAINING_EXECUTED = YES (exactly one requested epoch)
OPTIMIZER_STEP_EXECUTED = YES (epoch-1 PPO update only)
REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
MATCHING_KKT_CHANGED = NO
CACHE_CREDIT_CHANGED_DURING_TASK = NO
UAV_ALGORITHM_CHANGED = NO
CAPACITY_CHANGED = NO
MOVEMENT_COST_CHANGED = NO
TASK_OR_PREDICTION_INPUTS_CHANGED = NO
HOLDOUT_EPISODE_20002_PROBE = N/A (omitted; bounded requested probe completed)
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO

## Probe metadata (machine-readable)

```json
{
  "checkpoint_completed_episodes": 1,
  "checkpoint_completed_training_episodes": 1,
  "checkpoint_config_sha256": "c57835933528d9360ba6a5550564b0e904f86c226de0ae7d0c41360520a694ce",
  "checkpoint_current_train_origin": 300,
  "checkpoint_next_pool_index": 1,
  "checkpoint_policy_schema": "adaptive_joint_cache_replica_aware_uav_operation_recovery_r6",
  "counterfactual_environment_executions": 185,
  "dedup_reused_branch_count": 4,
  "fresh_initial_mean_p_keep": 0.4776745713,
  "fresh_initial_mean_p_update": 0.5223254287,
  "fresh_policy_rows": {
    "140/RSU_0": {
      "p_keep": 0.4969503579706167,
      "p_update": 0.5030496420293833
    },
    "140/RSU_1": {
      "p_keep": 0.5063005685806274,
      "p_update": 0.49369943141937256
    },
    "20/RSU_0": {
      "p_keep": 0.48058684228101844,
      "p_update": 0.5194131577189816
    },
    "20/RSU_1": {
      "p_keep": 0.48124149329909643,
      "p_update": 0.5187585067009036
    },
    "200/RSU_0": {
      "p_keep": 0.4808660304288259,
      "p_update": 0.5191339695711741
    },
    "200/RSU_1": {
      "p_keep": 0.4976225645058583,
      "p_update": 0.5023774354941417
    },
    "80/RSU_0": {
      "p_keep": 0.46969352572242995,
      "p_update": 0.53030647427757
    },
    "80/RSU_1": {
      "p_keep": 0.46790411877060645,
      "p_update": 0.5320958812293936
    }
  },
  "fresh_reference": {
    "correct_margin_mean": 0.014097426099999977,
    "keep_preferred_count": 4,
    "keep_preferred_mean_p_keep": 0.4921944067,
    "keep_preferred_policy_keep_count": 1,
    "mean_p_keep": 0.4851456917875,
    "mean_p_update": 0.5148543268375,
    "positive_margin_count": 5,
    "update_preferred_count": 4,
    "update_preferred_mean_p_update": 0.521903038025,
    "update_preferred_policy_update_count": 4
  },
  "logical_branch_count": 189,
  "probe_wall_seconds": 23.363250299997162,
  "rows": [
    {
      "candidate_count": 31,
      "correct_trigger_margin": 0.16478705406188965,
      "delta_q": 0.43029022216796875,
      "good_update_candidate_mass": 0.36169287003576756,
      "node": "RSU_0",
      "origin_index": 20,
      "p_keep": 0.4176064729690552,
      "p_update": 0.5823935270309448,
      "policy_trigger_correct": true,
      "preferred": "UPDATE",
      "q_keep": -116.08720397949219,
      "q_update": -115.65691375732422,
      "slot": 20
    },
    {
      "candidate_count": 14,
      "correct_trigger_margin": 0.149322509765625,
      "delta_q": 3.1791458129882812,
      "good_update_candidate_mass": 0.6564376652240753,
      "node": "RSU_1",
      "origin_index": 20,
      "p_keep": 0.4253387451171875,
      "p_update": 0.5746612548828125,
      "policy_trigger_correct": true,
      "preferred": "UPDATE",
      "q_keep": -116.28668212890625,
      "q_update": -113.10753631591797,
      "slot": 20
    },
    {
      "candidate_count": 16,
      "correct_trigger_margin": 0.18783780932426453,
      "delta_q": 0.08618927001953125,
      "good_update_candidate_mass": 0.3794754073023796,
      "node": "RSU_0",
      "origin_index": 80,
      "p_keep": 0.40608111023902893,
      "p_update": 0.5939189195632935,
      "policy_trigger_correct": true,
      "preferred": "UPDATE",
      "q_keep": -124.53244018554688,
      "q_update": -124.44625091552734,
      "slot": 80
    },
    {
      "candidate_count": 33,
      "correct_trigger_margin": -0.14353671669960022,
      "delta_q": -2.4519729614257812,
      "good_update_candidate_mass": 0.08759968169033527,
      "node": "RSU_1",
      "origin_index": 80,
      "p_keep": 0.4282316267490387,
      "p_update": 0.5717683434486389,
      "policy_trigger_correct": false,
      "preferred": "KEEP",
      "q_keep": -110.0656509399414,
      "q_update": -112.51762390136719,
      "slot": 80
    },
    {
      "candidate_count": 27,
      "correct_trigger_margin": -0.16402873396873474,
      "delta_q": -1.6418609619140625,
      "good_update_candidate_mass": 0.2659671828150749,
      "node": "RSU_0",
      "origin_index": 140,
      "p_keep": 0.4179856479167938,
      "p_update": 0.5820143818855286,
      "policy_trigger_correct": false,
      "preferred": "KEEP",
      "q_keep": -107.57202911376953,
      "q_update": -109.2138900756836,
      "slot": 140
    },
    {
      "candidate_count": 24,
      "correct_trigger_margin": -0.005076557397842407,
      "delta_q": -6.1495513916015625,
      "good_update_candidate_mass": 0.17044952884316444,
      "node": "RSU_1",
      "origin_index": 140,
      "p_keep": 0.4974617063999176,
      "p_update": 0.50253826379776,
      "policy_trigger_correct": false,
      "preferred": "KEEP",
      "q_keep": -110.97179412841797,
      "q_update": -117.12134552001953,
      "slot": 140
    },
    {
      "candidate_count": 25,
      "correct_trigger_margin": 0.114173024892807,
      "delta_q": 5.2203369140625,
      "good_update_candidate_mass": 0.6442990340292454,
      "node": "RSU_0",
      "origin_index": 200,
      "p_keep": 0.4429135024547577,
      "p_update": 0.5570865273475647,
      "policy_trigger_correct": true,
      "preferred": "UPDATE",
      "q_keep": -126.5180892944336,
      "q_update": -121.2977523803711,
      "slot": 200
    },
    {
      "candidate_count": 11,
      "correct_trigger_margin": 0.005296498537063599,
      "delta_q": -5.530326843261719,
      "good_update_candidate_mass": 0.519532322883606,
      "node": "RSU_1",
      "origin_index": 200,
      "p_keep": 0.5026482343673706,
      "p_update": 0.497351735830307,
      "policy_trigger_correct": true,
      "preferred": "KEEP",
      "q_keep": -126.54830169677734,
      "q_update": -132.07862854003906,
      "slot": 200
    }
  ]
}
```
