# Prediction-Driven Global Cache Value Solver

## Status

This is the single new non-RL Cache solver path. The first bounded Training
label set and value checkpoint have been created, and one independent 10-Slot
Training-window comparison has run. The solver accepted no Cache updates in
that window and matched KEEP+STAY utility; the model is not yet performance
validated. Stage3 MAPPO source and artifacts remain historical comparison
material and are not modified by this solver.

## Objective and supervised target

For a decision state `s_t`, legal single-node Cache target `c`, and KEEP target
`c_0`, the model learns the paired finite-horizon net-utility difference

```text
Delta_U_H(s_t, c) = sum_{k=0}^{H-1} U_system(c, t+k)
                    - sum_{k=0}^{H-1} U_system(c_0, t+k),  H <= 5
```

`U_system` comes from `slot_environment.execute_configured_slot`, so each branch
uses the formal task utility, Matching+KKT, and configuration costs. Each Cache
branch is held fixed for the label horizon; UAVs stay at their decision-time
positions; the cache transition cost is charged on the first branch slot only;
there is no terminal bootstrap. Labels use only formal Training tasks and are
never features. The target is a conditional fixed-configuration finite-horizon
estimate, not a Bellman target or long-term optimal value.

The training profile fixes the label budget at eight Training Episodes, six
non-overlapping decision states per Episode, at most 256 paired candidate
labels total, at most two proposals per node/state, and five actual Training
slots per branch. KEEP is evaluated once per state and shared by its paired
candidate labels. The state windows are selected in fixed chronological order
from the Training Pool. Entire Episodes are separated into fit, early-stop,
and calibration partitions; the independent calibration partition must have
at least two Episodes and 32 candidate pairs. The current implementation
starts from each Episode's formal initial Cache, so the initial supervised
distribution does not yet cover all post-update joint configurations.

## Predictor-only feature contract

`build_global_candidate_features` accepts prediction scenario arrays, the
prediction horizon mask, current UAV positions, Cache/config descriptors, and
the current/candidate Cache matrices. It does not accept rollout steps, real
task rows, future rewards, or Matching/KKT results.

The shared-width feature is independent of Cache-node count and contains:

- current and candidate Cache row, service additions/removals, global replica
  counts, role-wise Cache occupancy, and the formal transition cost;
- per-service predicted request count, input bits, CPU cycles, and `rho_hat` for
  each valid forecast horizon;
- finite prediction-sample vehicle position summaries and current UAV position
  centroid/spread;
- candidate node role, capacity, CPU, bandwidth, coverage radius, position and
  locally serviceable forecast workload;
- role-aggregated coverage, normalized offered CPU cycles/input workload, and
  aggregate CPU/bandwidth capacities, plus Cloud resource and forecast-load
  descriptors.

Coverage and offered-load features are predictors, not hard serviceability
constraints or resource allocations. The formal executor remains the source of
actual legality, Matching+KKT, reward, and cost. The feature schema supports
different Cache-node counts through role aggregations; model weights remain
bound to a service-count/horizon feature width in Checkpoint metadata.

## Bounded joint Cache search

At each Cache decision, `search_cache_configuration` proposes legal targets
using the existing Cache proposal semantics: prediction/coverage/intermediate
anchors and feasible add/delete/swap neighbors. It evaluates candidates with a
shared global value model, accepts the best target only if its predicted
candidate-minus-current-configuration gain is strictly above the empirical
q90 absolute residual from a grouped Training-only holdout, then rebuilds the
candidate set and re-estimates from the updated joint Cache configuration.
Previously visited joint configurations are skipped. KEEP is always the
fallback.

The current research profile caps one decision at 36 value evaluations, four
accepted single-node updates, four proposals per node per round, and a one
second best-effort search deadline. Forecast-feature preparation is bounded by
the fixed forecast/node/task arrays but cannot be interrupted mid-call; if the
deadline has passed, no candidate update is accepted and the timing overrun is
reported. It never enumerates joint Cache combinations.
After each accepted update, proposals are regenerated against the new joint
configuration for every node, including nodes already changed in this Macro;
previously visited joint configurations are skipped to prevent cycles.
The residual margin is a conservative heuristic only; it is not a confidence
bound and does not guarantee an actual utility increase. When a time or
candidate budget is exhausted, search returns the current feasible Cache.
Formal Matching+KKT is executed once for the selected Cache target by the
environment runner; no online counterfactual Oracle is used.

## Code paths and commands

- Value features, paired Training-label executor, shared MLP, bounded proposal
  search, and strict model Checkpoint loader:
  `src/decision/predictive_global_cache.py`
- Provenance-validated Training Pool window loader:
  `scripts/training/predictive_global_cache_data.py`
- Bounded Training label preparation and supervised fitting entry:
  `scripts/training/train_predictive_global_cache_value.py`
- Training-window solver/evaluator entry using the formal executor:
  `scripts/evaluation/run_predictive_global_cache_solver.py`
- Fixed budgets and model settings:
  `configs/research/predictive_global_cache_value_v1.json`

The preparation command uses only Training and writes a new `.npz` dataset plus
its metadata JSON. The fitting command uses Episode-disjoint Training
partitions: fit episodes determine feature/target normalization and gradients,
early-stop episodes select the epoch, and calibration episodes alone determine
the q90 residual gate. Calibration metrics compare MAE with a zero-ΔQ predictor
and report positive/negative sign discrimination. A failed basic
discrimination gate blocks solver evaluation. Fitting saves the model,
optimizer, best-epoch progress, partition IDs, and random states without
overwriting existing files. The solver entry validates the formal config,
pool manifest, service count, profile hash, calibration gate, and model schema
before execution; it rejects Training Episodes used for fitting or
calibration.

Training labels, model parameters, calibration metrics, and the paired
Training-window result are recorded in
`reports/predictive_global_cache_first_training.md`. Validation and Test remain
unread.
