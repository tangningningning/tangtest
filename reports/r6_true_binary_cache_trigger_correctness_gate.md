# R6 true binary Cache trigger correctness gate

本 gate 只在 frozen `episode_20001` 上运行 bounded witness；没有训练、optimizer.step、epoch、formal evaluation 或 300-slot full runtime collection。

## 结论字段

R6_SCHEMA = adaptive_joint_cache_replica_aware_uav_operation_recovery_r6
CACHE_TRIGGER_HEAD_SHAPE = [9, 2]
TRUE_BINARY_TRIGGER = YES
FOUR_WAY_MULTIPLICITY_PRIOR_REMOVED = YES
R6_ACTION_SPACE_EQUIVALENT_TO_R5 = YES
MAX_TRIGGER_POLICY_WEIGHTED_ADV_RESIDUAL = 0.0000003401
MAX_CANDIDATE_POLICY_WEIGHTED_ADV_RESIDUAL = 0.0000001206
TRIGGER_VIRTUAL_DIRECTION_PASS = 8/8
CANDIDATE_VIRTUAL_EXPECTED_Q_PASS = 8/8
ACTION_SPACE_EQUIVALENCE_PASS = YES
BASELINE_BRANCH_EQUIVALENCE = PASS
INITIAL_MEAN_P_KEEP = 0.4776745713
INITIAL_MEAN_P_UPDATE = 0.5223254287
REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
MATCHING_KKT_CHANGED = NO
CACHE_COST_CHANGED = NO
CAPACITY_CHANGED = NO
UAV_ALGORITHM_CHANGED = NO
TRAINING_EXECUTED = NO
OPTIMIZER_STEP_EXECUTED = NO
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO

## Bounded witness and exact credit

WITNESS_STATE_COUNT = 8
TARGET_SLOTS = (20, 80, 140, 200)
TARGET_NODES = ['RSU_0', 'RSU_1']
BENEFICIAL_COUNTERFACTUAL_COUNT = 4
HARMFUL_COUNTERFACTUAL_COUNT = 4
LEGAL_UPDATE_CANDIDATE_COUNT = 181
EXACT_LOGICAL_BRANCH_COUNT = 189
EXACT_UNIQUE_ENVIRONMENT_EXECUTIONS = 185
EXACT_DEDUP_REUSED_BRANCH_COUNT = 4

Q values use the formal branch `execute_slot` path and the current frozen critic bootstrap; candidate policy probability is the R6 unified conditional distribution.

| slot | target | p_keep | p_update | Q_keep | Q_update | DeltaQ | preferred | legal UPDATE candidates | good candidate mass |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 20 | RSU_0 | 0.4805868566 | 0.5194131732 | 2.5686719418 | 2.9138052464 | 0.3451333046 | UPDATE | 31 | 0.2510249308 |
| 20 | RSU_1 | 0.4812414944 | 0.5187585354 | 2.4330399036 | 5.5242457390 | 3.0912058353 | UPDATE | 14 | 0.4419244204 |
| 80 | RSU_0 | 0.4696935117 | 0.5303064585 | -6.3027734756 | -6.2162809372 | 0.0864925385 | UPDATE | 16 | 0.2592358050 |
| 80 | RSU_1 | 0.4679041207 | 0.5320959091 | 8.1992053986 | 5.5109376907 | -2.6882677078 | KEEP | 33 | 0.0000000000 |
| 140 | RSU_0 | 0.4969503582 | 0.5030496716 | 10.1372051239 | 8.3127279282 | -1.8244771957 | KEEP | 27 | 0.1478621962 |
| 140 | RSU_1 | 0.5063005686 | 0.4936994314 | 6.7981972694 | 0.0355654545 | -6.7626318149 | KEEP | 24 | 0.0817634118 |
| 200 | RSU_0 | 0.4808660448 | 0.5191339850 | -8.9531421661 | -4.2648515701 | 4.6882905960 | UPDATE | 25 | 0.6383469342 |
| 200 | RSU_1 | 0.4976225793 | 0.5023774505 | -8.9309215546 | -13.8720178604 | -4.9410963058 | KEEP | 11 | 0.2666186959 |

## Structural checks

- R5 retains its four-way production operation generator and exact grouped credit path.
- R6 uses the same `enumerate_cache_primitive_actions` output; only the trigger factorization changes.
- R6 candidate feature operation one-hot remains `NO_CHANGE/ADD/DELETE/SWAP`; the unified UPDATE set contains every legal `ADD`, `DELETE`, and `SWAP` candidate.
- Action-space set comparisons passed: 8/8.
- Baseline branch comparisons: 189 physical candidates; max reward/system diff=0.000e+00, max next-observation diff=0.000e+00, max Matching diff=0.000e+00, max next-cache diff=0.000e+00.

## Policy identities

Trigger identity: `p_keep*A_keep + p_update*A_update = 0` up to the reported residual.
Candidate identity: `sum_c pi(c|UPDATE)*A_c = 0` up to the reported residual.
The virtual gradient test uses detached Q values and temporary logits/scores only; no model parameter or optimizer state is modified.

## Execution-control audit

- TRAINING_EXECUTED = NO
- OPTIMIZER_STEP_EXECUTED = NO
- R5 checkpoint/run directories were not loaded or modified.
- No reward, system model, Matching/KKT, cache cost, capacity, UAV algorithm, or physical cache transition semantics were changed.
