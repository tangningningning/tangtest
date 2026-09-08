# Adaptive Cache/UAV direct policy implementation audit

## Outcome

The requested `policy_direct / adaptive_cache_uav_v1` production path was **not implemented** because the source-level HARD GATE in section 7 is triggered for both Cache and UAV control. A single legacy trigger does not select one concrete action. It activates an iterative optimizer whose accepted operation sequence and later candidate sets depend on K=100 objective results and on intermediate state mutations. The complete trigger-level macro-action therefore cannot be enumerated cheaply without running the scoring mechanism that `policy_direct` is intended to bypass.

No Matching, KKT, delay, energy, Q, G, reward, legacy policy, or legacy exact-configuration logic was changed by this task. The interrupted multi-fidelity experiment from the preceding task was precisely removed before this audit; existing production changes that predated it were retained.

```ini
BLOCKED = YES
DIRECT_CACHE_ACTION_EXTRACTION_BLOCKED = YES
DIRECT_UAV_ACTION_EXTRACTION_BLOCKED = YES
WORKTREE_INTERRUPTED_STATE_REVIEWED = YES
```

## Interrupted working-tree review

- Branch: `main`
- HEAD: `b7e90d4015bbeac682ce7b55a4a29d71a7e174d7`
- Existing modified production files retained: `src/decision/cache_configuration.py`, `src/decision/slot_environment.py`, `src/decision/task_allocation.py`, and `src/ppo_simple/trainer.py`.
- The incomplete multi-fidelity changes in `src/common/project_config.py`, `src/decision/cache_configuration.py`, `src/decision/uav_configuration.py`, and `system_config.toml` were removed hunk-by-hunk. Content hashes for `project_config.py`, `uav_configuration.py`, and `system_config.toml` again match HEAD; the remaining content diff is limited to the pre-existing production files above.
- Temporary multi-fidelity and best-UB runners/data were removed. `打包并上传.bat` and all existing reports were preserved.

## Static facts

### PPO trigger mapping

`TriggerActionSpec` constructs Cache heads from `cache_nodes["node_ids"]`, followed by UAV movement heads from configured UAV IDs. With the current five SVs, two RSUs, and two UAV cache nodes, the 11 bits are:

| Bit | Trigger |
|---:|---|
| 0 | `z_23^ca` |
| 1 | `z_87^ca` |
| 2 | `z_43^ca` |
| 3 | `z_75^ca` |
| 4 | `z_58^ca` |
| 5 | `z_RSU_0^ca` |
| 6 | `z_RSU_1^ca` |
| 7 | `z_UAV_0^ca` |
| 8 | `z_UAV_1^ca` |
| 9 | `z_UAV_0^q` |
| 10 | `z_UAV_1^q` |

Evidence: `src/ppo_simple/interface.py:41-75`, `src/decision/cache_configuration.py:337-379`, `system_config.toml:86,184,211`, and `data/generated/vehicle_roles.json`.

### Cache trigger cardinality

For each active Cache node, `initialize_cache_for_triggers()` may first accept a knapsack-generated whole-node cache vector. `optimize_cache()` then runs up to 20 coordinate rounds. In each round, each triggered node enumerates Add/Delete/Swap neighbors, evaluates every candidate through the prediction objective, accepts one best strict improvement, immediately mutates `current_cache[node_index]`, and uses that intermediate state for subsequent nodes/rounds. The outer joint optimizer may invoke `optimize_cache()` again during up to 10 Cache/UAV refinement rounds.

Consequently, one Cache trigger can yield no state change, one whole-vector initialization change, or multiple sequential Add/Delete/Swap state changes. The final actual trigger is computed from final versus previous cache state, not copied from the request.

Evidence: `src/decision/cache_configuration.py:1514-1611`, `src/decision/cache_configuration.py:1847-1897`, and `src/decision/slot_environment.py:978-1119`.

### UAV trigger cardinality

For each active UAV, `optimize_uav_configuration()` may accept one coarse initialization position and then one best position in each of up to 10 coordinate rounds. Every accepted position is immediately written to `final_positions`; later candidate generation uses that intermediate position. The outer joint optimizer may invoke the UAV optimizer again after Cache refinement.

Consequently, one UAV trigger can yield STAY or multiple sequential accepted positions, not one categorical movement candidate.

Evidence: `src/decision/uav_configuration.py:331-357`, `src/decision/uav_configuration.py:788-876`, and `src/decision/slot_environment.py:1003-1037,1121-1143`.

### Candidate generation and application

Primitive neighborhood generation is score-free:

- Cache Add/Delete/Swap vectors are produced by `generate_node_neighbors()` using only current node cache, service sizes, and capacity.
- UAV polar candidates are produced by `initialization_candidates()` / `coordinate_candidates()` using positions and configured geometry; `_candidate_is_feasible()` checks boundary, displacement, and UAV separation.

However, the required trigger-level candidate is a complete variable-length macro-action. Its next primitive candidate set depends on which previous candidate won the K=100 objective comparison. Enumerating all possible full Cache or UAV operation sequences would be combinatorial and would duplicate optimizer search. Therefore trigger-level candidate generation is not score-free.

The formal application operation is also currently inline (`current_cache[node_index] = node_candidate` and `final_positions[uav_id] = best_position`), rather than exposed as a reusable candidate-application API. Implementing the requested direct path safely would first require extracting these mutations together with their validation and rollback contracts.

### Legacy order and dependencies

The legacy joint order is:

1. Cache initialization at previous UAV positions.
2. UAV initialization using the resulting Cache state.
3. Joint objective evaluation.
4. For each outer round: Cache refinement at current UAV positions, immediate acceptance if improved, then UAV refinement using current Cache, immediate acceptance if improved.
5. Inside Cache blocks: triggered cache nodes in cache-node order, round by round.
6. Inside UAV blocks: triggered UAVs in configured UAV-ID order, round by round.

Candidate legality and generation therefore depend on prior conditional actions and must be sequential/autoregressive if a direct policy is introduced later.

## Cost and environment semantics

`system_model.cache_update_cost()` derives `actual_trigger` from `previous_cache != new_cache`. Fixed Cache cost and loading cost are both zero when `z_ca=1` but the cache is unchanged. `system_model.uav_movement_cost()` derives `actual_z` from displacement above tolerance; fixed movement cost and movement energy are zero when `z_q=1` but the UAV stays.

The unchanged formal environment applies final Cache/UAV configuration before actual feasible-target construction, Matching, KKT allocation, delay/energy calculation, Q, G, and reward. Thus Cache-to-offloading and UAV-to-channel coupling remain intact.

## Required minimal refactor before direct policy is safe

The smallest viable next design step is not a single categorical head. It must first define a formal variable-length macro-action protocol with an explicit STOP/NO_CHANGE action for Cache and STAY/STOP for UAV, while preserving the legacy block/node order. That refactor must:

1. Extract score-free primitive enumeration, feasibility checks, and candidate application into shared production APIs.
2. Define bounded sequential termination independently of K=100 scoring without silently changing the system problem.
3. Record every active head's intermediate state context, candidate features/mask, selected index, and old log-probability in rollout storage.
4. Recompute PPO probabilities solely from stored tensors, never by re-enumerating candidates from a future environment state.
5. Add full Cache/UAV operation histories (currently only aggregate counts/final states are returned) so a legacy exact decision can be mapped into the same macro-action schema for teacher extraction.

Until those semantics are explicitly approved, implementing a one-candidate-per-trigger policy would violate the task's cardinality and application-order requirements.

## Required status fields

```ini
LEGACY_POLICY_PRESERVED = YES
LEGACY_EXACT_SOLVER_PRESERVED = YES

TRIGGER_SEMANTICS_PRESERVED = YES

CACHE_TRIGGER_TO_ACTION_CARDINALITY = ZERO_OR_MULTIPLE_SEQUENTIAL_OPERATIONS
UAV_TRIGGER_TO_ACTION_CARDINALITY = ZERO_OR_MULTIPLE_SEQUENTIAL_MOVEMENTS

CACHE_CONDITIONAL_NO_CHANGE_SUPPORTED = YES
UAV_CONDITIONAL_STAY_SUPPORTED = YES

CANDIDATE_GENERATION_SCORE_FREE = NO

CANDIDATE_SET_DEPENDS_ON_PRIOR_ACTIONS = YES
ACTION_APPLICATION_MODE = SEQUENTIAL
LEGACY_CONFIGURATION_UPDATE_ORDER = CACHE_INITIALIZATION -> UAV_INITIALIZATION -> REPEATED(CACHE_REFINEMENT -> UAV_REFINEMENT); NODE/UAV ORDER WITHIN EACH BLOCK

VARIABLE_CANDIDATE_POLICY_IMPLEMENTED = NO
PPO_UPDATE_REQUIRES_ENV_REENUMERATION = NO

PREDICTION_POLICY_FEATURES = [CURRENT_PPO_OBSERVATION: predicted_vehicle_positions_h0, predicted_service_popularity_h0/rho_hat; NO_NEW_DIRECT_POLICY_FEATURES_IMPLEMENTED]

CACHE_COST_DEPENDS_ON = ACTUAL_CHANGE_AND_0_TO_1_LOADED_MB
UAV_COST_DEPENDS_ON = ACTUAL_MOVEMENT_AND_DISTANCE

POLICY_DIRECT_ONLINE_EXACT_CONFIGURATION_CALLS = NOT_APPLICABLE

MATCHING_CHANGED = NO
KKT_CHANGED = NO
DELAY_FORMULA_CHANGED = NO
ENERGY_FORMULA_CHANGED = NO
Q_FORMULA_CHANGED = NO
G_FORMULA_CHANGED = NO
REWARD_FORMULA_CHANGED = NO

CACHE_OFFLOADING_COUPLING_PRESERVED = YES
UAV_CHANNEL_COUPLING_PRESERVED = YES

SMOKE_ORIGINS_COMPLETED = 0
SMOKE_MEAN_RUNTIME = NOT_RUN_HARD_GATE_BLOCKED

REWARD_FINITE = NOT_VERIFIED
G_FINITE = NOT_VERIFIED
REWARD_G_CONSISTENT = NOT_VERIFIED

PPO_UPDATE_SMOKE = NOT_RUN_DUE_TO_REQUIRED_ROLLOUT_LENGTH

READY_FOR_TEACHER_WARMSTART = NO
```

No convergence or long-training claim is made.

## Final verification

```ini
TARGETED_IMPORT_CHECK = PASS
PYTHON_CACHE_DIRS_REMAINING = 0
PYC_FILES_REMAINING = 0
TEMPORARY_RUNNER_FILES_REMAINING = 0
GIT_DIFF_CHECK = PASS
```

`git status --short` still shows the pre-existing modified production files and historical untracked reports, plus this report. It also reports stat-only modifications for `src/common/project_config.py`, `src/decision/uav_configuration.py`, and `system_config.toml` after the interrupted edit/revert cycle; their Git content hashes equal HEAD and `git diff` contains no content changes for those paths. No commit, push, merge, or rebase was performed.
