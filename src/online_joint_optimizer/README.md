# Online Cache–UAV Joint Optimizer

This package contains a bounded, prediction-driven optimizer for cache and UAV
targets. Its objective is the project's system utility. It is intended to
replace the action-selection step in a future algorithm, but it is not connected
to the current training or evaluation entry points.

## Design

- Cache updates and UAV scheduling are compared using the same system-utility
  result from the formal Slot executor.
- Repeated service copies across RSU, SV, and UAV caches are allowed; the search
  does not impose cache complementarity.
- Cache candidates retain KEEP and use the existing legal REPLACE generator.
  A changed node cannot be a pure add, pure delete, or unchanged UPDATE.
- Cache and UAV decisions are not declared mutually exclusive. Formal
  execution and existing resource constraints determine feasibility and cost.
- Cache candidates are scored over the current Slot plus explicitly supplied
  future predicted Slots, with the cache target held fixed and UAV targets
  re-optimized for each Slot. UAV target selection uses current-Slot system
  utility and the existing legal target generator.
- The formal `execute_configured_slot` path is reused, including its matching,
  KKT/resource allocation, and system cost calculations. Each candidate gets
  copied inputs, fresh KKT/node-evaluation caches, and fresh timing/diagnostic
  dictionaries.
- Search is finite and budgeted. KEEP wins utility ties within the configured
  tolerance, and only strict utility improvements replace a current UAV target.
  Candidate utility failures and non-finite results remain explicit failures.

Finite candidate generation and coordinate improvement do not guarantee a
global optimum. A one-Slot system-utility improvement also does not establish
that a cache update maximizes long-term value.

## Modules and public API

- `candidate_evaluator.py` defines `SystemModelContext`, `SlotInput`,
  `PredictedSlotInput`, `CandidateEvaluation`, `SearchBudget`, and
  `CandidateEvaluator`. The evaluator accepts a cache target and UAV positions,
  then returns the formal Slot result and diagnostics from isolated state.
- `cache_optimizer.py` generates KEEP and bounded legal REPLACE targets,
  permits changes at multiple cache nodes, and scores candidates over explicit
  predicted Slot inputs while rescheduling UAVs at each Slot.
- `uav_scheduler.py` enumerates existing legal STAY/MOVE targets and performs
  bounded per-UAV coordinate improvement using full system utility. Existing
  map, travel-distance, and UAV separation checks are applied by the formal
  candidate generator and action executor.
- `joint_optimizer.py` coordinates cache scoring, UAV improvement, and joint
  re-evaluation under one shared evaluation ledger. It returns target states,
  utility changes, prediction provenance, and search diagnostics.
- `prediction_adapter.py` converts the current origin's saved prediction sample
  into horizon-aligned `PredictedSlotInput` values accepted by both optimizers.

The package exports these data structures and optimizers from
`online_joint_optimizer`.

## Prediction contract

The formal prediction pipeline saves complete future scenario paths in
`prediction_scenarios.npz`. `scenario_generator.py` combines GRU `rho_hat`,
vehicle prediction scenarios, Bernoulli arrivals, categorical service draws,
and service-catalog workload samples. Its metadata records that vehicle and
popularity predictions use information through the origin Slot minus one,
tasks are sampled from `rho_hat` rather than future `rho_star`, and future
ground truth is not saved.

Call `adapt_aligned_rollout_step(step, model, scenario_index=...)` with the
current `AlignedRolloutStep` and an explicit sample index. Horizon zero is the
current observed Slot and is excluded; horizons 1 onward become predicted Slot
inputs. Each future input contains `Task` objects in bits/cycles, all predicted
vehicle positions in meters, the matching recentered scenario tail, a valid
horizon mask, Slot/time indices, and provenance metadata. The adapter validates
the artifact's information boundary, current Slot alignment, formal task
probability and jitter settings, service-catalog workload units, and prediction
mask. It does not read tasks.csv, trajectory.csv, or future realized values and
does not resample tasks.

The selected sample index is fixed across the full path and reused by every
candidate in one optimizer call, so candidate comparisons share common random
numbers. This is one predicted sample path, not an expectation over all saved
samples; callers can repeat bounded calls for other explicit sample indices if
they need a scenario ensemble. Without a valid future horizon the adapter
returns `is_complete=False`; cache candidates remain unscored and KEEP is
retained.

The handoff is explicit:

```python
adapted = adapt_aligned_rollout_step(step, model, scenario_index=sample_index)
if adapted.is_complete:
    result = joint_optimizer.optimize(
        current_cache_state=cache_state,
        current_uav_positions=uav_positions,
        slot_input=current_slot_input,
        forecast_slots=adapted.forecast_slots,
    )
```

## Default evaluation budget

`SearchBudget` defaults to at most 256 formal Slot evaluations in total: up to
128 cache-stage, 64 UAV-stage, and 64 joint-stage evaluations. It generates at
most 8 cache targets including KEEP, uses at most 3 supplied future Slots, caps
the UAV candidate list at 4 entries per UAV (including STAY), and bounds UAV
coordinate rounds and joint cache/UAV comparisons. A shared ledger enforces
both per-stage and total limits; callers can lower the limits for tighter
latency budgets.

No training, actor update, or formal-path integration is performed by this
package.
