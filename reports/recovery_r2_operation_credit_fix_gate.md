# RECOVERY_R2_OPERATION_CREDIT_FIX bounded gate

- STAGE = RECOVERY_R2_OPERATION_CREDIT_FIX
- SOURCE_MODIFIED = YES
- Schema: adaptive_joint_cache_replica_aware_uav_operation_recovery_r2
- Config: [system_config_ppo_recovery_r2_6e4.toml](../configs/proposed/system_config_ppo_recovery_r2_6e4.toml)
- Base: Recovery R1/V10-D policy graph; the V10-D, Recovery R1, and V11-A schemas were not overwritten.

## Frozen paths

CACHE_CHANGED = NO, CANDIDATE_CREDIT_CHANGED = NO, SYSTEM_MODEL_CHANGED = NO, REWARD_CHANGED = NO, and MOVEMENT_COST_CHANGED = NO.

Cache keeps GLOBAL_GAE. The candidate head keeps the Recovery R1 production PPO credit path. Observation/action dimensions, global critic, PopArt, GAE, matching/KKT, cache cost, and movement semantics remain unchanged.

## Operation credit change

For each rollout transition and UAV head, R2 executes the production Cache and non-target UAV actions while evaluating STAY and every legal MOVE candidate through the real execute_slot path. MOVE value is the detached current candidate-policy expectation of the one-step TD values.

- UAV_OPERATION_ADV_SOURCE = COUNTERFACTUAL_ONE_STEP_TD
- UAV_OPERATION_RATIO_GRANULARITY = OPERATION_ONLY
- OP_ADV_NORMALIZATION = STD_ONLY
- OP_ADV_CENTERING = NO
- OP_ADV_SIGN_FLIP_COUNT = 0
- L_uav = 0.5 * (L_operation + L_candidate)
- GLOBAL_CRITIC_STEPS = 40

## Bounded gate

Frozen episode_20001, seed 4401, 300 slots, and one PPO update were run. No formal training, final-seed evaluation, or episode regeneration was performed.

- OBS_DIM = 400
- ACTION_DIM = 11
- Q_CACHE_USED = NO
- CACHE_ADV_SOURCE = GLOBAL_GAE
- UAV_CANDIDATE_ADV_SOURCE = RECOVERY_R1_ORIGINAL
- COUNTERFACTUAL_ENV_EXECUTIONS = 47328
- MEAN_MOVE_CANDIDATE_COUNT = 77.88
- GATE_RUNTIME_SECONDS = 1303.2466
- NAN_INF_DETECTED = NO

The full gate used the exact all-legal-candidate implementation; no candidate approximation was introduced.

## Fixed witness regression

The four required witnesses all matched the one-step TD preference (STAY) and the R2 operation loss gradient lowers the MOVE probability:

- OPERATION_WITNESS_ALIGNMENT = 4/4
- OPERATION_AUTOGRAD_DIRECTION_CORRECT = 4/4

See the JSON report for the per-witness Q values, sampled operation advantages, and dLoss/dMargin.

## Cleanup/status

- FORMAL_TRAINING_EXECUTED = NO
- FINAL_TEST_SEEDS_USED = NO
- EPISODE_REGENERATION = NO
- TEMP_FILES_REMOVED = YES
- GIT_DIFF_CHECK = PASS
- PUSHED = NO

The detailed machine-readable record is [recovery_r2_operation_credit_fix_gate.json](recovery_r2_operation_credit_fix_gate.json).

