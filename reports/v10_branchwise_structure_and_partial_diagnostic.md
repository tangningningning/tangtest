# V10-B Structure and 7-Epoch Partial-Run Diagnostic

STAGE = V10-B_STRUCTURE_VALIDATION

This report uses only static inspection, pure PyTorch autograd checks, pure
ratio/clipping calculations, and the seven complete records in
`runs/ppo_v10_branchwise_cache_uav_5e4_kl`. Epoch 8 was started but has no
complete record and is excluded. No environment simulation, training, resume,
evaluation, or final-test seed was run.

## V10-B parameter ownership

The V10-B policy inherits the V9 module layout:

- `SHARED_ACTOR_PARAMETERS`: `encoder.*` (`policy.encoder`), the shared actor representation.
- `CACHE_HEAD_PARAMETERS`: `cache_operation_head.*` and `cache_candidate_scorer.*`.
- `UAV_HEAD_PARAMETERS`: `uav_operation_head.*` and `uav_candidate_scorer.*`.
- `CRITIC_PARAMETERS`: `critic_encoder.*` and `critic_head.*`.

The Cache and UAV branch groups are disjoint, and the shared actor encoder is
in neither branch-specific group. The actor parameter groups cover the complete
actor parameter set. Critic parameters are disjoint from actor parameters and
are managed by the existing separate critic optimizer. No branch-specific
critic was introduced.

PARAMETER_GROUP_OVERLAP_FOUND = NO

## Branch gradient paths

Pure autograd was run on the actual V10-B policy evaluation path with synthetic
minibatch tensors only; no optimizer step was performed.

| Loss | Cache head | UAV head | Shared encoder |
|---|---:|---:|---:|
| `L_cache` | non-zero | None/zero | non-zero |
| `L_uav` | None/zero | non-zero | non-zero |
| `0.5 * (L_cache + L_uav)` | branch-composed | branch-composed | exact 0.5 average |

CACHE_LOSS_UPDATES_CACHE_HEAD = YES

CACHE_LOSS_UPDATES_UAV_HEAD = NO

CACHE_LOSS_UPDATES_SHARED_ENCODER = YES

UAV_LOSS_UPDATES_UAV_HEAD = YES

UAV_LOSS_UPDATES_CACHE_HEAD = NO

UAV_LOSS_UPDATES_SHARED_ENCODER = YES

SHARED_GRAD_COMPOSITION_CORRECT = YES; for every shared encoder parameter, the
combined gradient matched `0.5 * (grad_shared_from_cache +
grad_shared_from_uav)` within floating-point tolerance.

## Optimizer ownership

ACTOR_OPTIMIZER_COUNT = 1

CRITIC_OPTIMIZER_COUNT = 1

SHARED_ENCODER_MULTI_OPTIMIZER_FOUND = NO

The single actor Adam optimizer contains all shared encoder and branch actor
parameters exactly once. The critic Adam optimizer contains only critic
parameters. No `optimizer.step()` was used in the autograd validation.

## Branch clipping independence

With `clip_epsilon = 0.2`:

- Case A (`cache_ratio = 1.05`, `uav_ratio = 1.25`): Cache unclipped, UAV clipped.
- Case B (`cache_ratio = 1.25`, `uav_ratio = 1.05`): Cache clipped, UAV unclipped.

The V10-B actor surrogate uses the two branch ratios and does not use the
product `cache_ratio * uav_ratio` as its sole PPO clipping ratio. The legacy
joint ratio remains available for compatibility diagnostics and the existing
joint KL gate.

BRANCH_CLIPPING_INDEPENDENT = YES

## Epoch 1–7 partial-run metrics

The rate columns are the real JSONL fields
`cache_actual_change_rate` and `uav_actual_movement_rate`. Joint KL is the
real `update_metrics.jsonl` field `joint_approx_kl`; all branch diagnostic
columns are from the same epoch's update record. No epoch 8 values are used.

| Epoch | Return | CacheChange | UAVMove | Joint KL | Cache KL | UAV KL | Cache clip | UAV clip | Cache entropy | UAV entropy | Cache head grad | UAV head grad | Shared actor grad |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -989.049157 | 0.659259 | 0.575000 | 0.024973 | 0.039741 | 0.005091 | 0.461667 | 0.156667 | 19.124859 | 6.393789 | 0.396409 | 0.188035 | 0.679598 |
| 2 | -3254.048215 | 0.664444 | 0.551667 | 0.045626 | 0.019804 | 0.010198 | 0.388333 | 0.047500 | 19.166295 | 6.111869 | 0.384221 | 0.166845 | 0.662196 |
| 3 | -2375.540660 | 0.660000 | 0.570000 | 0.014268 | 0.007332 | -0.005993 | 0.387500 | 0.087500 | 19.165466 | 6.287584 | 0.386480 | 0.186445 | 0.690460 |
| 4 | -2757.165336 | 0.675185 | 0.525000 | 0.089571 | 0.066630 | 0.006154 | 0.373333 | 0.023333 | 19.187081 | 5.924909 | 0.477130 | 0.174882 | 0.760614 |
| 5 | -51.977194 | 0.631852 | 0.555000 | 0.180413 | 0.072939 | 0.001776 | 0.464444 | 0.026667 | 18.522401 | 6.214060 | 0.447984 | 0.212644 | 0.683136 |
| 6 | -355.534296 | 0.632963 | 0.518333 | 0.071801 | 0.049429 | 0.004678 | 0.435000 | 0.105000 | 18.762711 | 5.896060 | 0.453128 | 0.215462 | 0.722426 |
| 7 | -17.078308 | 0.628889 | 0.553333 | 0.015947 | 0.028432 | 0.002027 | 0.340000 | 0.108333 | 18.763752 | 6.206097 | 0.520689 | 0.240287 | 0.759711 |

All extracted values were finite. Epochs are consecutive 1–7 in both
`episode_metrics.jsonl` and `update_metrics.jsonl`; the partial run remains
untouched and its `run_summary.json` still reports seven completed epochs.

## Seven-epoch trend check

CACHE_KL_GT_UAV_KL_COUNT = 7 / 7

CACHE_CLIP_GT_UAV_CLIP_COUNT = 7 / 7

CACHE_GRAD_GT_UAV_GRAD_COUNT = 7 / 7

| Metric | Cache mean | UAV mean | Cache median | UAV median |
|---|---:|---:|---:|---:|
| Approximate KL | 0.0406154802 | 0.0034186324 | 0.0397412407 | 0.0046775205 |
| Clip fraction | 0.4071825468 | 0.0792857160 | — | — |
| Head gradient norm | 0.4380056689 | 0.1978000627 | — | — |
| Shared actor gradient norm | 0.7083058693 | — | — | — |

No unstable ratio was reported: comparisons use per-epoch differences and
counts, avoiding division by near-zero UAV values.

EARLY_CACHE_UPDATE_PRESSURE_SUPPORTED = YES. Across all seven complete early
epochs, Cache KL, Cache clip fraction, and Cache head gradient norm exceed the
corresponding UAV metric. This supports an early Cache-side update-pressure
pattern rather than an epoch-7-only coincidence.

## Interpretation limits

- Seven epochs only describe early optimization behavior.
- These data cannot determine late policy drift.
- These data cannot determine final V10-B performance or superiority over V9.
- These data cannot justify branch-specific learning rates.
- These data cannot justify branch-specific KL thresholds.

LATE_DRIFT_CONCLUSION_AVAILABLE = NO

BRANCH_SPECIFIC_LR_JUSTIFIED = NO

BRANCH_SPECIFIC_KL_JUSTIFIED = NO

## Formal-run readiness

All required structural gates passed: no parameter overlap, one actor
optimizer, no duplicated shared encoder ownership, branch-isolated
branch-specific heads, correct shared-gradient composition, independent
branch clipping, and finite diagnostics.

V10_B_READY_FOR_CLEAN_FORMAL_RUN = YES

This is a readiness conclusion only. No clean formal run was started in this
task, and the existing seven-epoch partial run was not resumed, deleted, or
modified.

TRAINING_EXECUTED = NO

EVALUATION_EXECUTED = NO

FINAL_TEST_SEEDS_USED = NO

PARTIAL_RUN_PRESERVED = YES

TEMP_FILES_REMOVED = YES

GIT_DIFF_CHECK = PASS

PUSHED = NO
