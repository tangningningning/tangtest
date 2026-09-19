# Exact-Q listwise candidate policy implementation gate

本 gate 只验证 candidate policy implementation，不运行训练、environment
counterfactual、SUMO 或 optimizer.step。历史 R6 schema 保留原有 candidate PPO
语义；新的显式 schema 才启用 EXACT_Q_LISTWISE。

## Implementation

- New schema: `adaptive_joint_cache_replica_aware_exactq_listwise_candidate`
- New mode: `algorithm.ppo.candidate_policy_improvement = "EXACT_Q_LISTWISE"`
- Temperature: `1.0`
- Cache trigger remains binary KEEP/UPDATE and its PPO surrogate is unchanged.
- UAV operation remains STAY/MOVE and its PPO surrogate is unchanged.
- Candidate Q tensors are the existing exact conditional counterfactual Q tensors.
- All legal candidates remain in the conditional softmax; no pruning, top-k, or
  argmax target was added.
- The legacy candidate PPO path remains available for historical R4/R5/R6
  schemas.
- Update metrics are emitted from existing exact-Q tensors only, including
  aggregate and SV/RSU/UAV Cache role-group Cache metrics and UAV-head metrics.

## Static and pure-tensor checks

`py_compile` passed for the modified policy, trainer, project-config, and
reporting modules. Both the unchanged R6 configuration and the new exact-Q
configuration passed project-config validation with artifact checks disabled.

The bounded pure-tensor check passed:

- exact-Q teacher ordering: higher Q received strictly higher teacher mass for
  both Cache and UAV;
- equal-Q teacher: uniform for both Cache and UAV;
- one small virtual gradient step: expected Q did not decrease for both Cache
  and UAV;
- candidate loss gradient reached `cache_candidate_scorer` and its current
  Cache encoder, and `uav_candidate_scorer` and its current UAV encoder;
- no optimizer was constructed or stepped by the check.

## Final gate fields

NEW_SCHEMA = adaptive_joint_cache_replica_aware_exactq_listwise_candidate
CANDIDATE_POLICY_IMPROVEMENT = EXACT_Q_LISTWISE

CACHE_TEACHER_ORDERING_PASS = YES
UAV_TEACHER_ORDERING_PASS = YES
CACHE_EQUAL_Q_UNIFORM_PASS = YES
UAV_EQUAL_Q_UNIFORM_PASS = YES
CACHE_VIRTUAL_EXPECTED_Q_NONDECREASE = YES
UAV_VIRTUAL_EXPECTED_Q_NONDECREASE = YES

EXTRA_COUNTERFACTUAL_ENV_EXECUTIONS = 0

CACHE_TRIGGER_CHANGED = NO
UAV_OPERATION_CHANGED = NO
CRITIC_CHANGED = NO
REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
MATCHING_KKT_CHANGED = NO
ACTION_SPACE_CHANGED = NO

TRAINING_EXECUTED = NO
OPTIMIZER_STEP_EXECUTED = NO

TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
