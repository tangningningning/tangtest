# One-Epoch Training Gate

Date: 2026-09-28

## Decision

The one-epoch run was not executed. The proposed-joint candidate critic
cold-start repair is implemented. The authorized compensatory traceable
Candidate-Q diagnostic completed its TRAIN-only warmup and exact evaluation,
and its complete numeric evidence is persisted in the accompanying [result
JSON](traceable_candidate_q_integrity_result.json). No formal PPO epoch
was run.

```text
ONE_EPOCH_TRAINING_EXECUTED = NO
ONE_EPOCH_ACTOR_LR = 5e-4
ONE_EPOCH_TARGET_EPOCHS = 1
ONE_EPOCH_COMPLETED_EPOCHS = 0
LAST_COMPLETED_EPOCH = NONE
EPOCH_2_STARTED = NO
ONE_EPOCH_TRAINING_STATUS = NOT_EXECUTED
ONE_EPOCH_NOT_EXECUTED_REASON = formal PPO epoch remains deferred to CRITIC_DATA_COVERAGE_MINIMAL_EXPERIMENT after the authorized Critic target-alignment follow-up

ONE_EPOCH_RETURN = NOT RUN
ONE_EPOCH_CRITIC_EV = NOT RUN
ONE_EPOCH_CACHE_CHANGE_RATE = NOT RUN
ONE_EPOCH_UAV_MOVE_RATE = NOT RUN
ONE_EPOCH_CACHE_KL_MEAN = NOT RUN
ONE_EPOCH_UAV_OPERATION_KL_MEAN = NOT RUN
ONE_EPOCH_UAV_CANDIDATE_KL_MEAN = NOT RUN
ONE_EPOCH_NO_NAN_INF = NOT RUN
HYBRID_FALLBACK_DURING_ONE_EPOCH = NOT RUN

LATEST_CHECKPOINT = NOT CREATED
SLOT_METRICS_JSONL = NOT CREATED
UPDATE_METRICS_JSONL = NOT CREATED
EPISODE_METRICS_JSONL = NOT CREATED
CRITIC_DIAGNOSTICS_JSONL = NOT CREATED

READY_FOR_FORMAL_TRAINING_STAGE = NO
FORMAL_300_EPOCH_TRAINING_EXECUTED = NO
GIT_DIFF_CHECK = PASS
```

## Deferred work

The historical K=16 result remains 6/9 recall and normalized regret P95
0.0611901, with RSU_1, UAV_0, and UAV_1 failing. RSU_1's historical failure is
now identified as the Hybrid surrogate-only-winner semantic bug and is repaired
by final exact-confirmation (checkpoint version 24 → 25). UAV_0/UAV_1 remain a
screening/ranking problem whose root cause is INCONCLUSIVE on current evidence.
Candidate teacher state/head normalization is resolved, but Hybrid Candidate
quality remains deferred. The service-level autoregressive prototype fails the
Candidate quality gate (recommendation KEEP_CURRENT_PRODUCTION_PENDING_ANALYSIS)
and the production Candidate architecture is unchanged.

## Latest TRAIN-only decision

Transition-149 / UAV_0 reward root cause: REWARD_OUTLIER_VALID_TASK_EXECUTION. Evidence: [uav0_transition149_reward_root_cause.json](uav0_transition149_reward_root_cause.json). CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION is now CANDIDATE_Q_INTEGRITY_CONCERN; current next stage: COUNTERFACTUAL_CRITIC_REPAIR_DESIGN. Formal PPO update and one-epoch training remain unexecuted; readiness remains NO.

## Latest Counterfactual Critic Reliability Decision

Counterfactual Critic Reliability Gate: `CONCERN`. Candidate-Q classification: `CANDIDATE_Q_INTEGRITY_CONCERN`; next stage: `COUNTERFACTUAL_CRITIC_REPAIR_DESIGN`. No PPO update or Epoch 1 was run; formal-training readiness remains NO.
The prior report serialization failure is retained in history. The authorized persistence-recovery run stored 12/12 raw continuation records in JSONL.

## Latest Critic Target Alignment Decision

Authorized compensatory Critic Target Alignment follow-up: `PARTIAL`. Current blocker / next stage: `CRITIC_DATA_COVERAGE_MINIMAL_EXPERIMENT`. Formal PPO update and Epoch 1 remain NO; readiness remains NO. Counterfactual Critic Reliability remains CONCERN, Candidate-Q remains CANDIDATE_Q_INTEGRITY_CONCERN, and surrogate repair remains blocked.

## Latest Critic Data Coverage Decision

Four-episode unique-coverage Critic experiment: `INCONCLUSIVE` because the fixed diagnostic rollout SHA did not reproduce. Coverage training completed and its diagnostic checkpoint reload passed; metric and UAV_1 branch evaluation stopped at the SHA gate. Next stage: `CRITIC_DATA_COVERAGE_EXPERIMENT_FOLLOWUP`. Formal PPO update and Epoch 1 remain NO; READY_FOR_FORMAL_TRAINING_STAGE remains NO. Counterfactual Critic Reliability remains CONCERN; Candidate-Q remains CANDIDATE_Q_INTEGRITY_CONCERN; surrogate repair remains blocked.
