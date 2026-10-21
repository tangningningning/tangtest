# Cloud Shared Resource Allocation Review

## Scope and active configuration

The current formal runner is `scripts/training/mainline_runner.py`; the Trainer's default formal configuration is `configs/proposed/alternating_actor_updates_v1.toml`. Its Cloud settings are `resource_mode = "finite_pool"`, `total_bandwidth_bps = 20.0e6`, `uplink_rate_bps = 20.0e6`, and `cpu_capacity_cycles_per_s = 20.0e9`. The tracked `system_config.toml` has the same 20 Mbps and 20 GHz values. These remain simulation baseline assumptions, not measured network or server capabilities.

Historical baseline before the shared-Cloud-resource update: the exact Matching evaluator passed the full `cloud_total_bandwidth_bps` value to every Cloud task. The equal `nodes.cloud.uplink_rate_bps` and `communication.cloud.uplink_rate_bps` fields were configuration aliases/checks; exact Matching read `nodes.cloud.total_bandwidth_bps` through `cloud_total_bandwidth_bps`. Therefore, two simultaneous Cloud uploads could each use 20 Mbps. At that historical point, the finite-pool CPU KKT shared `cpu_capacity_cycles_per_s` for tasks in `NEW` and `COMPUTING`; `UPLOADING` and `NETWORK_DELAY` tasks fell back to the full Cloud CPU capacity individually even though their compute delay was included. The current allocation semantics are specified below. `Task.transmit_time_share` defaults to 1.0 for tasks loaded by the formal task loader. In the Cloud delay estimate it scales wall-clock upload time as `L/(R_i q_i)`; it does not constrain total Cloud rate. CPU allocations are service rates in cycles/s, not task-cycle counts or time fractions.

Configuration validation now treats `total_bandwidth_bps` as the canonical shared rate cap and enforces equality with both rate aliases. In `finite_pool` mode, it also enforces that the active CPU capacity equals the retained `total_cpu_cycles_per_s` compatibility value; the two legacy CPU alias fields remain required to agree with each other.

## Implemented resource allocation

For Cloud-assigned task `i`, let `x_i = R_i / R_total` and `y_i = f_i / F_total`. The formal path now uses:

```text
sum_i x_i <= 1,       epsilon_bw <= x_i <= 1   (tasks with modeled upload)
sum_i y_i <= 1,       epsilon_cpu <= y_i <= 1  (all tasks in the Cloud CPU plan)
R_total = 20e6 bit/s
F_total = 20e9 cycles/s
```

The task completion delay and vehicle radio energy remain:

```text
T_i = psi_c * L_i / (R_i * q_i) + C_i / f_i + remaining_network_delay_i
E_i = P_tx * L_i / R_i
```

`psi_c` remains the existing communication-delay multiplier and does not enter vehicle transmit energy. The Cloud path remains an independent cellular link; this change adds no RSU relay, RSU–Cloud backhaul, bandwidth, channel-gain, or SINR model.

The existing normalized task-utility objective is retained. Ignoring terms constant with respect to resource allocation, maximizing summed utility is equivalent to minimizing the two separable pool objectives:

```text
sum_upload a_i / x_i + sum_cloud b_i / y_i
a_i = (L_i / R_total) * (omega_T * psi_c / (T_ref * q_i) + omega_E * P_tx / E_ref)
b_i = omega_T * C_i / (T_ref * F_total)
```

For positive coefficients, each reciprocal term is convex (`d²(a/x)/dx² = 2a/x³ > 0`), and the capacity/simplex and lower-bound constraints are linear. The rate and CPU pools have separate constraints and an additive objective, so the existing bounded-simplex KKT solver can solve each pool independently without assuming a non-convex extension is convex. Its existing stationarity form is `x_i = clip(sqrt(a_i/lambda), epsilon_bw, 1)` and `y_i = clip(sqrt(b_i/mu), epsilon_cpu, 1)`, with deterministic bisection for the multipliers. This is not equal-share allocation unless the coefficients imply it.

The rate KKT covers tasks in `NEW` or `UPLOADING` with positive remaining input. The finite CPU KKT allocates service rate over the complete Cloud task set whose computation delay is included, including upload/network-delay stages. A Cloud assignment-set change causes the incremental evaluator to recompute the full old/new Cloud pool; unrelated Edge task evaluations remain incremental. The relaxed per-task screening bound still gives one task the full configured pools, and exact candidate evaluations use the shared KKT result.

## Verification

Using the current formal configuration and a short synthetic check (no Episode or training):

- A single Cloud task received 20 Mbps and 20 GHz.
- Two simultaneous Cloud tasks each received positive allocations; total rate was 20,000,000 bit/s and total CPU was 19,999,999,999.995331 cycles/s, within the 20e9 capacity (the small difference is solver floating-point tolerance).
- The exact delay and energy matched `psi_c * L/(R_i*q_i) + C_i/f_i` and `P_tx * L/R_i`; the check also matched `system_model.cloud_execution()` for the formal `q_i = 1` case.
- Mixed `UPLOADING`/`NETWORK_DELAY` tasks both received positive CPU rates whose sum stayed within 20 GHz; only the uploading task consumed uplink capacity. A `q_i = 0.5` check confirmed `q_i` lengthens wall-clock upload delay while radio energy remains based on `L_i/R_i`.
- The two KKT solves were finite and converged; stationarity and complementary-slackness residuals were each below `1e-7`, and interior allocations satisfied the shared-multiplier stationarity checks.
- A synthetic incremental move from Cloud to a valid SV recomputed the remaining Cloud task's rate/CPU allocation against the full pools and passed the resource constraint check.
- The formal config loaded successfully. Parsed values in the tracked `system_config.toml` and `sparse_coordinated_cache_v1.toml` match their `HEAD` values; current alternating config values also parse as 20 Mbps and 20 GHz. Only Cloud-semantics comments changed in TOML files.
- Python AST parsing and `git diff --check` passed.

Slot reward still takes `task_utility_sum` once, then subtracts `cache_cost` and `movement_cost` once: `G = Q - C_cache - C_movement`. Cloud vehicle energy is already included once in each task's utility and is not separately subtracted as another cost. No reward weights, Matching transfer/exchange rules, MAPPO behavior, or data were changed.

## Modified files

- `src/decision/task_allocation.py`: shared Cloud rate KKT, complete finite-pool CPU allocation, task-level rate use in delay/energy, resource diagnostics, and incremental pool invalidation/propagation.
- `src/common/project_config.py`: Cloud rate alias validation and compatibility-field semantics.
- `src/decision/cache_configuration.py`: Cloud resource-sharing flag is retained as a compatibility field; shared rate is enforced regardless of its value.
- `src/common/system_model.py`: documentation clarifies that the caller supplies per-task allocated rates; formulas did not change.
- `src/decision/slot_environment.py`: matching summary exposes per-task Cloud rate allocation and pool usage/capacity.
- `configs/proposed/system_config.toml`, `configs/proposed/alternating_actor_updates_v1.toml`, and `configs/proposed/sparse_coordinated_cache_v1.toml`: comments only; no TOML values changed.

The `legacy_unlimited` CPU compatibility mode remains available to historical configurations. It does not provide a Cloud CPU pool constraint; the formal Alternating MAPPO configuration uses `finite_pool` and was the path verified here. No Training Pool, Run, Checkpoint, Validation/Test data, or formal configuration value was modified. No training was run.

## Cloud Slot-level static resource-planning semantics (2026-10-20)

### Resource pools and allocation sets

The active `configs/proposed/alternating_actor_updates_v1.toml` loads Cloud `resource_mode = "finite_pool"`, `total_bandwidth_bps = 20.0e6 bit/s`, and `cpu_capacity_cycles_per_s = 20.0e9 cycles/s`. The matching Cloud rate aliases in the active configuration agree with the canonical total rate. These are per-Slot equivalent simulation capacities, not measured network or server guarantees.

For a proposed Matching, let `C_t` be the tasks assigned to Cloud. `_allocate_cloud_resources_kkt()` forms the upload set from tasks in `NEW` or `UPLOADING` with positive input bits. It forms the finite-pool CPU plan from the entire Cloud-assigned task list, regardless of whether a task is labeled `NEW`, `UPLOADING`, `NETWORK_DELAY`, or `COMPUTING`. With `x_i = R_i/R_total` and `y_i = f_i/F_total`, the current exact evaluator enforces:

```text
sum(i in C_t^up) R_i(t) <= R_total = 20e6 bit/s
sum(i in C_t^CPU) f_i(t) <= F_total = 20e9 cycles/s
```

The KKT CPU service-rate allocations and upload-rate allocations are returned by the same Cloud allocator call. `evaluate_matching_with_kkt()` uses those returned per-task values to compute each Cloud task's delay, vehicle radio energy, and utility. Matching candidate evaluations use this exact evaluation for the current and candidate assignments; the separate single-task/full-pool value is an optimistic screening bound, not the final candidate utility. Thus the completion-delay estimate is internally consistent with the static allocation returned for that candidate.

### Stage-aware completion-time estimate and units

`Task.input_bits` and `Task.cpu_cycles` are respectively bits and cycles. Task generation configures the input range in KB and stores task/catalog input sizes in bits (`INPUT_SIZE_BITS_PER_KB = 8000`); it computes `cpu_cycles = cycles_per_bit * input_bits * load_factor`. `R_i` is bit/s, `f_i` is cycles/s, `q_i = transmit_time_share` is dimensionless (default 1.0 for tasks loaded by the formal path), `psi_c = cloud_delay_factor` is dimensionless, and `remaining_network_delay_s` is seconds.

For the current evaluator, define the modeled upload amount as `L_i^rem = input_bits` for `NEW`/`UPLOADING`, and zero for `NETWORK_DELAY`/`COMPUTING`. Then:

```text
T_i^radio = L_i^rem / R_i
T_i^up,eff = psi_c * L_i^rem / (R_i * q_i)
T_i^network = remaining_network_delay_s, if stage == NETWORK_DELAY; otherwise 0
T_i^comp = cpu_cycles / f_i
T_i^complete,estimate = T_i^up,eff + T_i^network + T_i^comp
E_i^up = P_i^tx * T_i^radio
```

The formal loader creates each arrived task as `NEW`, with `q_i = 1.0`; in that path, the estimate is `psi_c * L_i/R_i + C_i/f_i`. `cloud_delay_factor` multiplies the upload communication-delay term; it is not an extra additive delay and is not applied again. Vehicle radio energy uses the unscaled base airtime and does not include the factor or network-delay term. For `UPLOADING`, the current task model uses the full `input_bits` because it has no remaining-upload field; for every supported stage it uses the full `cpu_cycles` because it has no remaining-compute field.

### Static planning boundary

The current model is per-Slot static joint resource planning and task completion-time estimation. `f_i(t)` is the equivalent CPU service rate selected by that Slot's KKT optimization for the candidate; it is not a record of actual CPU occupancy during upload, nor does it prove that Cloud physically reserves that share before computation begins. Although `Task` accepts the stage labels above, the formal path loads arrivals independently per Slot as `NEW`; a source search found no cross-Slot Cloud task queue, stage-transition update, remaining-upload tracking, remaining-cycle tracking, or actual CPU-release timestamp.

Consequently, the model enforces the CPU capacity constraint in its per-Slot optimization but does not produce a real cross-Slot Cloud schedule, CPU utilization trace, queueing delay, or stage-entry/completion-triggered reallocation. Static resource planning and actual execution scheduling are distinct modeling layers. The current study may use the former as an approximation, provided its paper system model, algorithm description, and experimental conclusions state this assumption and do not claim dynamic CPU scheduling or measured Cloud occupancy.

### Targeted verification and limits

The formal Alternating configuration was loaded and `evaluate_matching_with_kkt()` was called directly with four synthetic tasks labeled `NEW`, `UPLOADING`, `NETWORK_DELAY`, and `COMPUTING` (no Episode or training). The returned allocation used 20e6 bit/s in total and 20e9 cycles/s in total. `NEW`/`UPLOADING` tasks alone received rate allocations; all four stages received positive CPU plan allocations. For each task, the exact returned delay matched `psi_c * L_i^rem/(R_i*q_i) + T_i^network + C_i/f_i`, and returned radio energy matched `P_tx * L_i^rem/R_i`. This verifies the mapping between KKT variables and the exact candidate estimate for this synthetic call, not a physical Cloud schedule or a full-system rollout.

For that exact path, the completion-time estimate includes the modeled upload term, any applicable `NETWORK_DELAY` remainder, and the compute term once each. The task utility is computed once from that delay and radio energy; Slot reward uses the resulting task-utility sum and does not separately deduct Cloud radio energy. No duplicate delay term or Cloud energy deduction was found in the inspected final evaluation path.

No dedicated existing Cloud system-model chapter was present under `docs/`; `docs/mappo_complete_solver.md` documents the MAPPO/Checkpoint contract but has no Cloud resource-model section to amend. This report remains the existing Cloud resource review; no duplicate system-model document was added. Source, formal configuration values, Training Pool, Checkpoints, and algorithm were not changed for this clarification.

```ini
RESOURCE_ALLOCATION_SEMANTICS = SLOT_LEVEL_STATIC_PLANNING
CROSS_SLOT_CLOUD_TASK_STATE = NOT_IMPLEMENTED
DYNAMIC_CPU_SCHEDULING = NOT_IMPLEMENTED
CPU_ALLOCATION_DELAY_CONSISTENCY = SYNTHETIC_EXACT_EVALUATOR_CHECK_PASS
CLOUD_DELAY_COMPOSITION = SOURCE_AND_SYNTHETIC_CHECK_PASS
UPLINK_ENERGY_CONSISTENCY = SOURCE_AND_SYNTHETIC_CHECK_PASS
DOUBLE_COUNTING = NO_IN_INSPECTED_FINAL_EVALUATION_PATH
```

## Controlled Cloud shared-resource Run audit (2026-10-20)

Run inspected: `runs/cloud_shared_resources_controlled_updates2_20261020`. This section uses only that Run's configuration snapshot, Manifest, Summary, JSONL metrics, and Checkpoints, plus a specifically identified earlier controlled Run for identity comparability. No source, configuration, Run, Pool, or Checkpoint file was changed.

### Run completion and artifact consistency

- The Manifest and Summary both record target 300 Episodes, controlled stop after 2 Policy Updates, `CONTROLLED_STOP_AFTER_UPDATES`, 8 completed Episodes, and 2 completed Updates. The controlled-stop limit and completed-update count both equal 2; this is not recorded as completion of the 300-Episode target.
- There are 8 Episode rows, 2 Update rows, 2 Critic diagnostic rows, and 2,400 Slot rows. Each Episode has exactly 300 unique Slot indices `0..299`, 300 environment executions, and `episode_terminal_observed = true`. Thus all 8 completed Episodes have full Slot logs; the remaining target Episodes were not run because the controlled stop fired.
- Each Episode's sum of Slot `G` matches its `episode_return` within floating-point tolerance. The latest Checkpoint was safely read in weights-only mode: its `training_counters` exactly match both Manifest and Summary, its schedule state matches, and its role is `RESUMABLE_MAINLINE_CHECKPOINT`. It records 8 Episodes / 2 Updates, 2 Macro / Micro / Long-Value Critic update groups, and `next_active_actor = MICRO`. The initial Checkpoint has zero completed Episodes / Updates; the best-return diagnostic Checkpoint also records the final 8 / 2 counters.
- The latest Checkpoint's seed, raw/semantic config hashes, config path values, Pool path, data protocol, Episode length, actor-update mode, and scheduler state agree with the Run identity. The current formal TOML and saved `run_config.toml` have identical parsed values and the saved raw config hash matches the Manifest. Checkpoint counters are internally consistent; the Checkpoint payload records the Pool path but not the Pool Manifest digest, so the Pool digest itself is evidenced by the Run Manifest rather than by a Checkpoint field.

### Actual-Run Cloud allocation evidence

The saved Run configuration confirms `resource_mode = finite_pool`, Cloud total uplink capacity `20e6 bit/s` (and its equal compatibility alias), and Cloud CPU capacity `20e9 cycles/s`. However, the recursively inspected Episode, Update, Critic, and Slot log fields contain no per-task Cloud rate allocation, Cloud CPU allocation, or Cloud pool-used/capacity fields. `slot_metrics.jsonl` contains Cloud task counts under `node_task_count` and KKT call counts, but these do not expose the selected Matching's actual resource allocations. Consequently:

- Actual-Run Cloud shared uplink constraint: **UNRESOLVED from logs**; no per-task rates or total used-rate values are recorded.
- Actual-Run Cloud shared CPU constraint: **UNRESOLVED from logs**; no per-task CPU rates or total used-CPU values are recorded.
- Actual-Run allocation-to-delay consistency: **UNRESOLVED from logs**; task-level rates, CPU shares, delays, and utilities are not jointly recorded.

These gaps cannot be filled by the earlier synthetic evaluator check or by the static KKT source. The Run logs alone therefore do not demonstrate that every executed Matching stayed within both capacities.

### Alternating Actor updates and numerical checks

Update row 1 (`policy_update_index = 0`, Episodes 1–4) enables only Micro Actor: Micro Actor took 40 optimizer steps; Macro Actor took 0. Update row 2 (`policy_update_index = 1`, Episodes 5–8) enables only Macro Actor: Macro Actor took 4 optimizer steps; Micro Actor took 0. The scheduler and latest Checkpoint both select Micro next. Per-update Critic work is consistent: Macro Critic 50 steps, Micro Critic 40 steps, and Long-Value Critic 50 steps in each group; cumulative counters are 100, 80, and 100 respectively. The logs evidence actor enable flags and optimizer steps; they do not contain an intermediate model/Adam-state snapshot after Update 1, so per-group parameter/moment equality for the inactive actor is not independently measured here.

Both actor updates have finite reported KL/clip metrics and no KL early-stop flag: Update 1 Micro approximate KL `0.00384195`, clip fraction `0.0164583`; Update 2 Macro post-step joint KL `0.00697215` (cache `0.00287592`, termination `0.00241918`), clip fraction `0.00527108`. All numeric values in the four inspected JSONL files are finite. A safe weights-only read of the latest Checkpoint found 166 tensors and no non-finite floating-point tensor values. This is numerical-integrity evidence for these artifacts, not a convergence assessment.

### Reward and action behavior

For all 2,400 Slot rows, `G = task_utility - cache_cost - movement_cost` has zero identity violations. The Run totals are task utility `25,273.726789`, Cache cost `30.826391`, movement cost `1.900687`, and `G = 25,240.999710`; the per-Episode `G` sums match Episode returns.

`CacheChange` in Episode logging maps to `actual_cache_update_rate`: the fraction of Macro Cache decisions whose selected configuration changed at least one node. Across the Run, 1,296 of 1,297 Macro decisions changed one or more node configurations (99.923%). This is not the fraction of all nodes, nor an UPDATE-action count per Slot. The actual changed-node distribution per Macro decision was: 0 nodes: 1; 1: 31; 2: 102; 3: 210; 4: 309; 5: 310; 6: 222; 7: 90; 8: 20; 9: 2. The nine per-node change counts ranged from 617 to 659 across the 1,297 Macro decisions; the Slot metric's changed-node counts sum to 5,791 and match the Episode-level node counts.

`UAVMove` here is an action-level non-STAY rate, not distance moved: UAV_0 selected MOVE in 1,120/2,400 decisions (46.667%); UAV_1 in 1,137/2,400 (47.375%); combined 2,257/4,800 (47.021%). Movement distance is not included in this action-rate denominator.

Mean Episode return was `3011.883105` for Episodes 1–4 and `3298.366823` for Episodes 5–8, a descriptive increase of `286.483718`. With only two Policy Updates and four Episodes per group, this does not establish policy improvement or convergence.

### Earlier controlled Run comparison and readiness

The earlier `runs/alternating_idmap_controlled_2updates_20261020` artifacts still exist. Its parsed `run_config.toml`, semantic config hash, Training Pool Manifest digest, seed and Episode seed range match this Run; its initial model and optimizer state dictionaries also match exactly. The raw config-file hashes differ, and neither Run Manifest records a source revision or source-code fingerprint. Therefore code-level experiment identity cannot be confirmed, so this report makes no Return comparison or causal attribution to the earlier Run.

```ini
CONTROLLED_RUN_INTEGRITY = PASS_FOR_8_COMPLETE_EPISODES_AND_2_UPDATES
CLOUD_UPLINK_SHARED_CONSTRAINT_IN_ACTUAL_RUN = UNRESOLVED_NO_PER_TASK_RATE_LOGS
CLOUD_CPU_SHARED_CONSTRAINT_IN_ACTUAL_RUN = UNRESOLVED_NO_PER_TASK_CPU_LOGS
ALTERNATING_ACTOR_SCHEDULE = PASS_FROM_UPDATE_METRICS_AND_CHECKPOINT
NUMERICAL_STABILITY = PASS_FOR_JSONL_NUMBERS_AND_LATEST_CHECKPOINT_TENSORS
REWARD_CONSISTENCY = PASS_FOR_SLOT_IDENTITIES_AND_EPISODE_RETURN_SUMS
POLICY_IMPROVEMENT_EVIDENCE = DESCRIPTIVE_RETURN_INCREASE_ONLY_NOT_CAUSAL
READY_FOR_300EP_TRAINING = NO_CLOUD_RUNTIME_ALLOCATION_EVIDENCE_MISSING
```

## Cloud shared-resource runtime evidence fields (2026-10-20)

The reporting path now writes compact Cloud allocation evidence to each new `slot_metrics.jsonl` row. In `task_allocation.py`, the final `TransferExchangeMatchingResult` is the selected Matching allocation source; its per-task Cloud rate and CPU maps are summed directly for the selected totals. Capacity comes from the result's configured resource contract (with the formal finite-pool capacity used when an empty selected Cloud set has no allocation result). Violation is computed as `max(0, allocated_total - capacity)`. The selected row also records whether both allocation maps contain only finite values.

For within-Slot candidate solves, `optimize_task_matching()` snapshots Cloud keys in its per-Matching KKT resource cache, then aggregates only newly computed Cloud cache entries. Each entry is the actual Cloud allocation result for one unique candidate Cloud task set; cache hits are not counted as new solves. `cloud_kkt_solve_count` counts the scalar Cloud CPU/uplink KKT solver results stored on those actual allocation results (so a finite-pool plan can contribute one CPU solve and, when uploading tasks exist, one uplink solve). The max allocation and violation fields are computed from those result objects, not by rerunning KKT. `cloud_kkt_nonfinite_result_count` counts a computed Cloud allocation result if a solver diagnostic or its allocation values/capacity are non-finite. With no Cloud KKT result in a Slot, the solve count is zero and the aggregate maxima are `null`, rather than presenting an unexecuted solve as a passing zero allocation.

New flat Slot fields:

```text
cloud_task_count
cloud_uplink_allocated_bps
cloud_uplink_capacity_bps
cloud_uplink_capacity_violation_bps
cloud_cpu_allocated_cycles_per_s
cloud_cpu_capacity_cycles_per_s
cloud_cpu_capacity_violation_cycles_per_s
cloud_selected_resource_values_finite
cloud_kkt_solve_count
cloud_kkt_max_uplink_allocated_bps
cloud_kkt_max_cpu_allocated_cycles_per_s
cloud_kkt_max_uplink_capacity_violation_bps
cloud_kkt_max_cpu_capacity_violation_cycles_per_s
cloud_kkt_nonfinite_result_count
```

Existing Slot fields, including `node_task_count`, `kkt_calls`, Reward, task metrics, and movement/cache metrics, retain their names and meanings. The new fields are observational only: they are not read by Matching, Reward, or policy code. No per-task resource arrays are added to the Slot log.

Targeted verification used the project interpreter and in-memory synthetic Cloud tasks (one task and two concurrent tasks), without executing an Episode or training. For each case, selected summary totals matched `math.fsum()` of the actual allocation maps returned by the Cloud allocator; CPU and uplink limits were not exceeded, violations were zero, and resource values were finite. A second candidate-allocation set was aggregated separately from the selected result; the solve count included the scalar CPU and uplink solver diagnostics from both newly computed candidate sets and excluded cache hits. The Slot JSONL append function serialized all new fields with existing fields intact, using strict JSON (`allow_nan=False`). This verifies the data flow and serialization on synthetic calls only; it does not constitute evidence about a later actual training Run.

This instrumentation applies to Runs produced after the code change. Existing controlled Run `cloud_shared_resources_controlled_updates2_20261020` is unchanged and lacks these fields, so its actual Cloud capacity compliance remains unresolved until a new Run records and is checked against them.

```ini
SELECTED_MATCHING_RESOURCE_LOGGING = PASS_SYNTHETIC_OBJECT_AND_SERIALIZATION_CHECK
ALL_KKT_CALLS_COVERED_BY_SLOT_AGGREGATES = PASS_NEW_CLOUD_CACHE_MISSES_AND_SCALAR_DIAGNOSTICS
UPLINK_CAPACITY_VIOLATION_MEASUREMENT = PASS_SYNTHETIC_ALLOCATION_OBJECT_CHECK
CPU_CAPACITY_VIOLATION_MEASUREMENT = PASS_SYNTHETIC_ALLOCATION_OBJECT_CHECK
NONFINITE_RESOURCE_DETECTION = PASS_SYNTHETIC_FINITE_AND_NONFINITE_INPUT_CHECK
RESOURCE_ALLOCATION_ALGORITHM_CHANGED = NO
FORMAL_CONFIG_CHANGED = NO
TRAINING_POOL_MODIFIED = NO
MAPPO_TRAINING_EXECUTED = NO
VALIDATION_TEST_READ = NO
```

## Logged Cloud shared-resource controlled Run audit (2026-10-20)

Run inspected: `runs/cloud_shared_resources_logged_controlled_updates2_20261020`. This audit read only this Run's Manifest, Summary, Episode, Update, Critic diagnostic, Slot, and Checkpoint artifacts, plus Episode metrics from the previous controlled Run `runs/cloud_shared_resources_controlled_updates2_20261020`. The Checkpoint was loaded with PyTorch's weights-only loader and a restricted NumPy dtype allowlist. No training or data generation was run.

### Run integrity and checkpoint

- Manifest and Summary agree on target 300 Episodes, stop limit 2 Policy Updates, 8 completed Episodes, 2 completed Policy Updates, and `CONTROLLED_STOP_AFTER_UPDATES`. The 300-Episode run is therefore controlled-stopped, not marked complete.
- There are 8 parseable Episode rows, 2 Update rows, 2 `critic_diagnostics.jsonl` rows, and 2,400 parseable Slot rows. Every Episode has exactly 300 unique slots covering `0..299`; none are missing or duplicated.
- For every Episode, its `episode_return` equals the sum of Slot `G` values (zero mismatches). Across 2,400 Slots, `G = task_utility - cache_cost - movement_cost` also has zero mismatches. Totals are task utility `25,273.726789`, cache cost `30.826391`, movement cost `1.900688`, and `G = 25,240.999710`.
- `latest_checkpoint.pt` has role `RESUMABLE_MAINLINE_CHECKPOINT`; its training counters exactly equal Manifest and Summary counters: 8 Episodes, 2 Policy Updates, 2 Macro/Micro update groups, 100 Macro Critic steps, 80 Micro Critic steps, and 100 Long-Value Critic steps. Its schedule state says `next_policy_update_index = 2` and `next_active_actor = MICRO`. Config hashes, seed, Pool path, episode length, and alternating schedule match the Run identity. The stop marker is in Manifest and Summary; the Checkpoint records matching counters and scheduler state but has no separate stop-status field. The Pool digest is recorded in the Manifest, not in the Checkpoint payload.

### Cloud resource logs from the actual Run

All 14 required Cloud fields are present and parseable in all 2,400 Slot records; there are no missing, invalid-type, or null selected-allocation fields. Both capacity fields are consistently `20,000,000 bit/s` and `20,000,000,000 cycles/s`.

- Final selected Matching totals never exceed capacity. Maximum observed selected totals are `20,000,000 bit/s` and `20,000,000,000 cycles/s`; selected violation fields are zero for every Slot, and recomputing `max(0, allocated - capacity)` matches every recorded violation. `cloud_selected_resource_values_finite` is true in all 2,400 rows. `cloud_task_count` matches `node_task_count.Cloud` in every row and ranges from 1 to 14.
- Every Slot has at least one Cloud KKT scalar result: 2,400 Slots with KKT and selected Cloud tasks, zero Slots with no Cloud KKT result, and zero Slots with KKT results but no selected Cloud tasks. `cloud_kkt_solve_count` sums to 71,848 scalar CPU/uplink solver results (2-240 per Slot); it is not the number of Matching calls. Slot `kkt_calls` sum to 293,042, while `matching_calls` sum to 2,400.
- Since each Slot has KKT results, all four KKT aggregate maxima are populated. Their maximum allocation values are at the respective capacities; both aggregate violation fields are exactly zero in every Slot. `cloud_kkt_nonfinite_result_count` sums to zero. These are the logged aggregates of newly computed Cloud KKT allocation objects under the definitions above; per-task resource arrays are not stored, so this audit does not independently reconstruct each candidate's allocation from task-level values.

### Alternating updates and numerical stability

Update 1 (Episodes 1-4) enables Micro only: 40 Micro Actor steps and 0 Macro Actor steps. Update 2 (Episodes 5-8) enables Macro only: 4 Macro Actor steps and 0 Micro Actor steps. Both groups perform 50 Macro Critic steps, 40 Micro Critic steps, and 50 Long-Value Critic steps. Update metrics and cumulative Manifest/Summary/Checkpoint counters agree. Inactive Actor step counts are zero; these artifacts do not include a per-group parameter/Adam-state snapshot for direct tensor-by-tensor comparison.

The active Micro update reports approximate KL `0.003842`, clip fraction `0.016458`, with no early stop. The active Macro update reports post-step joint KL `0.006972`, joint clip fraction `0.005271`, with its four logged actor steps accepted and no early stop. All numeric values in Episode, Update, Critic diagnostic, and Slot JSONL records are finite. The latest Checkpoint's 1,403,661 tensor elements were scanned; no non-finite tensor or floating-point values were found. This establishes artifact-level numerical integrity, not convergence.

### Previous controlled Run comparison

The previous `runs/cloud_shared_resources_controlled_updates2_20261020` has Episode rows for the same 8 Episodes. Per-Episode returns are exactly equal between Runs, and all non-runtime Episode metrics match within `1e-9`; recorded runtime fields differ. This is a consistency comparison only and is not evidence of a policy effect. Two Policy Updates remain insufficient to assess convergence.

```ini
CONTROLLED_RUN_INTEGRITY = PASS_8_OF_300_EPISODES_2_UPDATES_2400_OF_2400_SLOTS
SELECTED_CLOUD_UPLINK_CAPACITY = PASS
SELECTED_CLOUD_CPU_CAPACITY = PASS
ALL_RECORDED_KKT_RESULTS_WITHIN_CAPACITY = PASS
NONFINITE_RESOURCE_RESULTS = PASS
ALTERNATING_ACTOR_SCHEDULE = PASS
REWARD_CONSISTENCY = PASS
PREVIOUS_RUN_METRICS_COMPARISON = CONSISTENT_8_OF_8_RETURNS_EXACT_OTHER_NON_RUNTIME_METRICS_WITHIN_1E-9
READY_FOR_300EP_TRAINING = YES
```
