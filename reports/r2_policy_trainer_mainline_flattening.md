# R2 policy/trainer mainline flattening

This pass makes Recovery R2 the only current learned Joint PPO schema at
source level.  It does not train, collect a rollout, regenerate episodes,
run SUMO, or perform formal evaluation.

## Mainline and dependency graph

```text
ppo-train / ppo-eval
        -> PPOV2Trainer shared policy/checkpoint factory
        -> AdaptiveJointCacheReplicaAwareUAVOperationRecoveryR2ActorCritic
        -> AdaptiveJointCacheReplicaAwareUAVOperationDecoupledEncoderActorCritic
        -> AdaptiveJointCacheReplicaAwareUAVOperationJointActorCritic
        -> V8 replica-aware baseline backbone -> V7 -> primitive/V6 backbone
```

```text
CURRENT_HEAD = 525404ce0d595a56e1940cb0135b779d8141cfd4
MAINLINE_SCHEMA = adaptive_joint_cache_replica_aware_uav_operation_recovery_r2
MAINLINE_CONFIG = configs/proposed/system_config_ppo_recovery_r2_6e4.toml
R2_POLICY_STANDALONE = YES
R2_DEPENDS_ON_R1_CLASS = NO
R2_DEPENDS_ON_V10D_CLASS = NO
R2_DEPENDS_ON_V10C_CLASS = NO
```

The neutral base classes preserve the former R2 module construction order and
module/state-dict layout.  R2 remains directly responsible for the current
schema identity; V7/V8 and primitive policies remain explicit non-mainline
baseline/ablation paths.

## Removed current-source history

```text
HISTORICAL_POLICY_CLASSES_REMOVED =
  V9, V10-branch-observable, V10-branchwise, V10-C, V10-D, Recovery-R1,
  V11-A policy classes
HISTORICAL_TRAINER_SCHEMA_BRANCHES_REMOVED =
  V9, V10-branch-observable, V10-branchwise, V10-C, V10-D, Recovery-R1,
  V11-A schema sets/dispatch and V11-A Q/readiness path
SUPPORTED_MAINLINE_PPO_SCHEMAS =
  adaptive_joint_cache_replica_aware_uav_operation_recovery_r2
SUPPORTED_BASELINE_SCHEMAS = primitive V1..V6, V7, V8, V8-no-prediction
```

The current R2 control mode is named
`policy_primitive_r2_joint_cache_uav_operation`; its sampling and environment
application semantics are unchanged.

## State-dict and numerical compatibility

The comparison was run against the policy source at `CURRENT_HEAD`, with
`torch.manual_seed(4401)`, observation dimension 400, action dimension 11,
Cache feature dimension 8, UAV feature dimension 20, seven Cache heads, and
hidden sizes `(128, 128)`.  A legal synthetic joint-action tensor was used;
no environment transition was executed.

```text
STATE_DICT_PARAMETER_TENSOR_COUNT_BEFORE = 28
STATE_DICT_PARAMETER_TENSOR_COUNT_AFTER = 28
STATE_DICT_PARAMETER_COUNT_BEFORE = 228146
STATE_DICT_PARAMETER_COUNT_AFTER = 228146
STATE_DICT_STATE_TENSOR_COUNT_BEFORE = 30
STATE_DICT_STATE_TENSOR_COUNT_AFTER = 30
STATE_DICT_HASH_BEFORE = ef27b39eb6675e1067fa132dec0d6d1e992f31e69fe4784ad39348736a6ff367
STATE_DICT_HASH_AFTER  = ef27b39eb6675e1067fa132dec0d6d1e992f31e69fe4784ad39348736a6ff367
STATE_DICT_KEY_SET_IDENTICAL = YES
STATE_DICT_SHAPES_IDENTICAL = YES
FRESH_INIT_PARAMETER_MAX_ABS_DIFF = 0.0
POLICY_FORWARD_MAX_ABS_DIFF = 0.0
POLICY_LOGPROB_MAX_ABS_DIFF = 0.0
POLICY_VALUE_MAX_ABS_DIFF = 0.0
OPERATION_LOGITS_MAX_ABS_DIFF = 0.0
CANDIDATE_SCORES_MAX_ABS_DIFF = 0.0
```

The hash is the deterministic ordered state-dict hash over key, shape, dtype,
and raw tensor bytes.

## R2 behavior contract

```text
CACHE_ADV_SOURCE = GLOBAL_GAE
UAV_OPERATION_ADV_SOURCE = COUNTERFACTUAL_ONE_STEP_TD
UAV_OPERATION_RATIO_GRANULARITY = OPERATION_ONLY
UAV_CANDIDATE_ADV_SOURCE = RECOVERY_R1_ORIGINAL
OP_ADV_NORMALIZATION = STD_ONLY
OP_ADV_CENTERING = NO
OBSERVATION_DIM = 400
ACTION_DIM = 11
TENSOR_LAYOUT_CHANGED = NO
INIT_ORDER_CHANGED = NO
OPTIMIZER_GROUPS_CHANGED = NO
PPO_OBJECTIVE_CHANGED = NO
CREDIT_FORMULA_CHANGED = NO
REWARD_CHANGED = NO
MOVEMENT_SEMANTICS_CHANGED = NO
MATCHING_KKT_CHANGED = NO
POPART_CHANGED = NO
GAE_CHANGED = NO
```

## Checks

```text
CONFIG_LOAD_AND_FORMAL_VALIDATION = PASS
R2_TRAINER_CONSTRUCTION = PASS
R2_TRAINER_MRO = R2 -> neutral decoupled encoder -> neutral joint base -> V8 -> V7
R2_CHECKPOINT_VERSION = 13
ACTOR_OPTIMIZER_GROUPS = 1
CRITIC_OPTIMIZER_GROUPS = 1
POLICY_CONSTRUCTION_OBS400_ACTION11 = PASS
POLICY_FORWARD_FINITE = PASS
PPO_TRAIN_HELP = PASS
PPO_EVAL_HELP = PASS
PYTHON_COMPILEALL_SRC_SCRIPTS = PASS
GIT_DIFF_CHECK = PASS (after temporary bytecode cleanup)
```

## Files and execution boundary

```text
FILES_MODIFIED =
  src/ppo_simple/policy.py
  src/ppo_simple/trainer.py
  src/ppo_simple/rollout.py
  src/ppo_simple/primitive_control.py
  src/ppo_simple/__init__.py
  src/ppo_simple/evaluator.py
  src/common/project_config.py
  configs/proposed/system_config_ppo_recovery_r2_6e4.toml
  configs/README.md
  configs/experiment_registry.toml
  README.md
  reports/r2_policy_trainer_mainline_flattening.md
FILES_DELETED_THIS_STAGE = none
REPORTS_DELETED = 0
PENDING_CLEANUP = none in the current R2 source path; archival historical
  reports/run artifacts are intentionally retained
TEMP_FILES_REMOVED = YES
FORMAL_TRAINING_EXECUTED = NO
ROLLOUT_EXECUTED = NO
EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
FORMAL_EVALUATION_EXECUTED = NO
COMMITTED = NO
PUSHED = NO
```
