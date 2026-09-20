# Adaptive joint Cache/UAV action reconfiguration

## 1. Baseline and scope

- Branch before modification: `main`
- HEAD before modification: `0296816`
- Current HEAD remains `0296816`; no commit or push was performed.
- The only formal experiment remains `proposed_joint`.
- The only proposed configuration remains `configs/proposed/system_config.toml`.
- Policy schema remains `adaptive_joint_cache_uav_ppo`.
- Candidate objective remains `EXACT_Q_LISTWISE`.
- The proposed control-mode label is now `adaptive_joint_cache_uav_reconfiguration`; this is a semantic control-mode rename, not a new experiment or version chain.

The previously running process was PID `41772`. It was found and a stop request was issued when this task started. The process was no longer present afterward; the preserved run log shows that its requested one-epoch run had already completed before the interrupt took effect. Its checkpoint and the earlier validation report were preserved. No training was started after this action-schema modification.

## 2. Before/after action semantics

### Cache

Before, the proposed conditional UPDATE policy consumed primitive candidate types derived from `ADD`, `DELETE`, and `SWAP`, with a binary trigger layered over those primitives.

After, the active proposed path is:

```text
Cache high-level trigger: KEEP / UPDATE
UPDATE conditional candidate: one complete target binary cache configuration
```

The target proposal enumerator is deterministic and stably deduplicated. It covers:

1. low-reconfiguration targets from the existing neighborhood constructor;
2. a node-specific predicted-demand target;
3. a replica-aware target using current replica counts;
4. configurable intermediate demand/current targets.

The proposal cap is configurable as `algorithm.cache.target_candidate_capacity = 64`. Every target is binary, shape-aligned, capacity-feasible, and different from the current node configuration. `apply_cache_target_configuration` rejects an UPDATE target equal to the current state, so KEEP and UPDATE cannot silently collapse to the same action.

The target feature retains current/target/delta cache vectors, node role, predicted demand before/after/gain, replica-count/diversity features, observed local-replica usage proxy, loading/cost signals, and deterministic proposal provenance. The observed usage proxy is explicitly diagnostic/feature information only; it is not an exact counterfactual gain and is not used by reward or Exact-Q.

The node-specific demand signal uses prediction-scenario `arrival` and `service_id`, prediction horizon discounting, predicted positions, and serviceability. The scenario artifact has task-vehicle labels for `arrival` but positions for all research vehicles; the implementation maps task IDs into the full vehicle-ID index before constructing per-vehicle demand. No realized future task stream is read for this proposal.

The physical cache transition and cost remain the existing `cache_update_cost`/system-model path. No cache-cost formula was changed.

### UAV

Before, the proposed conditional MOVE policy consumed the generic primitive position set, which included a STAY primitive in the candidate representation.

After, the active proposed path is:

```text
UAV high-level trigger: STAY / MOVE
MOVE conditional candidate: one target position; STAY is not a candidate
```

The target proposal uses current UAV cache contents, predicted vehicle/service demand, and predicted positions to compute service-aware vehicle weights and a weighted demand centroid. It adds projected demand anchors and a small deterministic polar geometry fallback. Zero-weight cases fall back to finite geometric proposals/current-position feature values; no NaN/Inf fallback is used.

The configurable UAV target cap is `algorithm.uav.target_candidate_capacity = 24`. Every target is filtered through the existing boundary, one-slot maximum displacement, and sequential multi-UAV safety checks. Applying a target delegates to the existing physical movement validation/cost path. Movement cost and movement constraints are unchanged.

## 3. Adaptive timescale and frozen components

- KEEP/UPDATE is still selected per Cache node at every slot.
- STAY/MOVE is still selected per UAV at every slot.
- No fixed Cache period was added.
- No fixed UAV period was added.
- The execution sequence remains prediction/observation -> Cache target reconfiguration -> UAV target position using the updated Cache -> Matching -> KKT -> utility/cost/reward.
- Reward, system model/objective, Matching, KKT, action-vector dimensionality, vehicle/task generation, popularity prediction, and PPO update design were not changed.
- Exact-Q counterfactual replay now evaluates complete target configurations/target positions, while the Exact-Q listwise objective remains enabled.

The old primitive enumerators and generic primitive sampler remain only for preserved legacy/baseline/offline paths. The active proposed control mode no longer calls those primitive candidate paths, and proposed action metadata no longer emits `operation_type`, `added_service_ids`, or `removed_service_ids`.

## 4. Minimal checks

All checks used the fixed `adaptive_cache_mec` Python environment.

### Parsed active configuration

```text
EXPERIMENT_ID = proposed_joint
POLICY_SCHEMA = adaptive_joint_cache_uav_ppo
CONFIG_PATH = configs/proposed/system_config.toml
CANDIDATE_OBJECTIVE = EXACT_Q_LISTWISE
CACHE_HIGH_LEVEL_ACTIONS = KEEP / UPDATE
UAV_HIGH_LEVEL_ACTIONS = STAY / MOVE
CONTROL_MODE = adaptive_joint_cache_uav_reconfiguration
UAV_METHOD = demand_aware_target_position
```

### Proposal and tensor sanity

- Cache target counts for the first aligned training state: `[7, 9, 8, 9, 7, 21, 15, 15, 11]`; all were within the configured cap and validated for binary/capacity feasibility.
- UAV target counts with both UAV triggers enabled: `[18, 18]`; all were within the configured cap and passed the existing movement/safety feasibility checks.
- Cache candidate tensor shape: `(9, 64, 85)`.
- UAV candidate tensor shape: `(2, 24, 39)`.
- The policy was constructed with 2 UAV heads explicitly; policy evaluation returned a finite log probability.
- Exact Cache replay state construction returned 21 `CacheTargetCandidate` entries and finite candidate probabilities summing to one.
- One deterministic real one-slot rollout reached the existing environment execution path and was feasible. Its Cache metadata contained only `KEEP`/`TARGET_CONFIGURATION`; its UAV metadata contained only `STAY`/`TARGET_POSITION`.
- No `ADD`/`DELETE`/`SWAP` metadata was emitted by that active proposed rollout, and STAY was not in the UAV candidate set.

## 5. Design answers

```text
CACHE_WHEN = Per-slot learned KEEP/UPDATE trigger; no fixed period.
CACHE_WHAT = Complete binary target cache configuration, proposed from demand/replica-aware/intermediate/low-reconfiguration targets.
UAV_WHEN = Per-slot learned STAY/MOVE trigger; no fixed period.
UAV_WHERE = Demand-aware weighted centroid/projected anchors plus small deterministic geometry fallback, filtered by existing hard constraints.
OFFLOADING_WHERE = Unchanged existing Matching/KKT execution after the updated Cache and UAV configuration.
RESOURCE_HOW_MUCH = Unchanged existing system model, capacity checks, Matching, KKT, and reward/cost semantics.
```

## 6. Checkpoint compatibility

The earlier checkpoint at `runs/exact_q_listwise_one_epoch_validation_20260919/latest_checkpoint.pt` was produced under the old proposed candidate tensor/action semantics. It is intentionally preserved for historical comparison, but it is not compatible with the new target-configuration/target-position action schema: feature widths, candidate capacities, control-mode semantics, and the explicit UAV-head construction differ. A future run must use fresh initialization; no checkpoint migration was added.

## 7. Final status

```text
TRAINING_PROCESS_FOUND = YES (PID 41772)
TRAINING_PROCESS_STOPPED = YES / no process remains; the one-epoch run had naturally completed before the stop request took effect
TRAINING_EXECUTED_AFTER_MODIFICATION = NO
FINAL_TEST_SEEDS_USED = NO
CONFIG_NEW_VERSION_CREATED = NO
CODE_VERSION_CHAIN_CREATED = NO
COMMITTED = NO
PUSHED = NO
```

## 8. Machine-readable final fields

```text
BRANCH = main
HEAD_BEFORE_MODIFICATION = 0296816

TRAINING_PROCESS_FOUND = YES
TRAINING_PROCESS_STOPPED = YES (PID 41772 no longer running; natural completion preceded the stop request)
STOPPED_PID = 41772

EXPERIMENT_ID = proposed_joint
CONFIG_PATH = configs/proposed/system_config.toml
POLICY_SCHEMA = adaptive_joint_cache_uav_ppo

ADAPTIVE_TIMESCALE_PRESERVED = YES
FIXED_CACHE_PERIOD_ADDED = NO
FIXED_UAV_PERIOD_ADDED = NO

CACHE_TRIGGER = KEEP / UPDATE
CACHE_CONDITIONAL_ACTION_BEFORE = ADD / DELETE / SWAP primitive candidates after a binary trigger
CACHE_CONDITIONAL_ACTION_AFTER = complete binary target cache configuration after UPDATE

UAV_TRIGGER = STAY / MOVE
UAV_CANDIDATE_PROPOSAL_BEFORE = generic primitive coordinate candidates including STAY representation
UAV_CANDIDATE_PROPOSAL_AFTER = demand-aware target positions plus bounded geometry fallback; STAY is high-level only

ADD_DELETE_SWAP_REMOVED_FROM_PRODUCTION_POLICY_ACTIONS = YES

CACHE_TARGET_CONFIGURATION_CAPACITY_FEASIBLE = YES
CACHE_UPDATE_TARGET_DIFFERS_FROM_KEEP = YES

UAV_DEMAND_AWARE_PROPOSAL_IMPLEMENTED = YES
UAV_HARD_MOVEMENT_CONSTRAINTS_PRESERVED = YES

EXACT_Q_LISTWISE_PRESERVED = YES
MATCHING_CHANGED = NO
KKT_CHANGED = NO
SYSTEM_OBJECTIVE_CHANGED = NO
CACHE_COST_CHANGED = NO
UAV_MOVEMENT_COST_CHANGED = NO
CLOUD_FALLBACK_CHANGED = NO

CONFIG_NEW_VERSION_CREATED = NO
CODE_VERSION_CHAIN_CREATED = NO

OLD_CHECKPOINT_COMPATIBLE_WITH_NEW_ACTION_SCHEMA = NO
FRESH_INITIALIZATION_REQUIRED = YES

TRAINING_EXECUTED_AFTER_MODIFICATION = NO
FINAL_TEST_SEEDS_USED = NO

SYNTAX_IMPORT_CHECK = PASS
CONFIG_LOAD_CHECK = PASS
CACHE_PROPOSAL_SANITY_CHECK = PASS
UAV_PROPOSAL_SANITY_CHECK = PASS
POLICY_TENSOR_SHAPE_CHECK = PASS

REPORT_PATH = reports/adaptive_joint_cache_uav_action_reconfiguration.md
GIT_DIFF_CHECK = PASS
TEMP_FILES_CLEANED = YES
COMMITTED = NO
PUSHED = NO
```
