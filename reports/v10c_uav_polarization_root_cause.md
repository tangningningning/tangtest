# STAGE = V10-C_UAV_POLARIZATION_ROOT_CAUSE

## Scope and execution boundary

This is an offline audit of exactly three completed V10-B runs: 5e-4 (`ppo_v10_branchwise_cache_uav_5e4_kl_formal`), 6e-4 (`ppo_v10_branchwise_cache_uav_6e4_kl_formal`), 1e-3 (`ppo_v10_branchwise_cache_uav_1e3_kl_formal`).
No PPO training, resume, LR sweep, reward/system-model change, final-seed evaluation, or new long-running test was started. The script read persisted logs and source/config files only; it did not write any run directory.

All window statistics below are grouped by training epoch. `Return mean` and `Return std` are the mean and population standard deviation of the 60/20 episode returns in the window. Other numeric columns are arithmetic means of the logged epoch-level episode/update values. `NOT_AVAILABLE` means the requested field is absent; no proxy is silently substituted.

## 1. Three-run alignment

RUN_ALIGNMENT_VALID = YES
The three manifests report 300 completed epochs, V10 branchwise policy schema, `full_episode_randomized`, seed 4401, train origins 0..299, 300 slots/rollout, critic LR 3e-4, target KL 0.10, clip epsilon 0.20, update epochs 4, and entropy coefficient 0. The only intended manifest-level training variable is actor LR: 5e-4, 6e-4, 1e-3.
Episode and update realization identities match across all 300 epochs: first `(0, 'episode_20001', 1, 1, 4401, 0, 299)` and last `(299, 'episode_20300', 300, 300, 4401, 89700, 89999)`. The local episode pool contains all `episode_20001`..`episode_20300`: YES. Local config SHA-256 values match the corresponding manifests: YES.

| run | actor LR | completed | policy schema | pool index first..last | config SHA |
|---|---|---|---|---|---|
| 5e-4 | 0.0005000 | 300 | V10 branchwise | 0..299 | MATCH |
| 6e-4 | 0.0006000 | 300 | V10 branchwise | 0..299 | MATCH |
| 1e-3 | 0.0010000 | 300 | V10 branchwise | 0..299 | MATCH |

Alignment mismatches: NONE

## 2. UAV action semantics verification

The static path is internally consistent:

- Trigger action order is Cache node IDs followed by configured UAV IDs (`src/ppo_simple/interface.py:56`). The three runs have action_dim=11; the config has 9 Cache heads and 2 UAV heads.
- UAV operation order is explicitly `('STAY', 'MOVE')` (`src/ppo_simple/policy.py:1230` and `src/ppo_simple/primitive_control.py:964`). Index 0 is STAY, index 1 is MOVE.
- The UAV operation mask is `[True, bool(move_candidates)]`, where candidates are production-feasible coordinate candidates filtered by boundary, max displacement, and separation (`src/ppo_simple/primitive_control.py:1059`; `src/decision/uav_configuration.py:374`). It is not derived from the Cache action after the Cache decision; Cache state can condition the UAV context/features, but not this geometry-only UAV operation mask.
- A selected MOVE samples a candidate and `apply_uav_candidate` updates the mutable current position; a selected STAY leaves it unchanged (`src/ppo_simple/primitive_control.py:1136`). The derived trigger vector uses the final `uav_triggers`, so MOVE is a requested/actual-movement path and STAY is trigger 0.
- Candidate old log-probability is stored separately for every UAV head; evaluation reconstructs operation log-probability plus candidate log-probability only when operation=MOVE (`src/ppo_simple/trainer.py:1790`). The branch ratio is `exp(new_uav_log_prob - old_uav_log_prob)` (`src/ppo_simple/trainer.py:1797`).
- Movement rate is counted from actual `next_uav_positions - previous_uav_positions` above 1e-6 in rollout (`src/ppo_simple/rollout.py:625`), and the episode metric is actual count divided by slots × UAV count. The cost model uses the same tolerance and final old-to-new distance (`src/common/system_model.py:349`).

UAV_MOVE_METRIC_SEMANTICS_VALID = YES
UAV_ACTION_MAPPING_VALID = YES
UAV_MASK_VALID = YES
UAV_LOGPROB_VALID = YES

Persisted semantic diagnostics also support this conclusion: all three runs have `uav_move_feasible_rate` exactly 1.0 over all 300 epochs, and `uav_move_selected_rate` equals `uav_actual_movement_rate` exactly in every episode record. No feasible-MOVE masking or selected-MOVE-but-no-position-change residual is visible.

| run | feasible rate min..max | max selected-vs-actual abs | max trigger×given-vs-actual abs |
|---|---|---|---|
| 5e-4 | 1.0000..1.0000 | 0.000000000000 | 0.000000000000 |
| 6e-4 | 1.0000..1.0000 | 0.000000000000 | 0.000000000000 |
| 1e-3 | 1.0000..1.0000 | 0.000000000000 | 0.000000000000 |

There is no semantic failure that would justify stopping inference. A limitation remains: historical logs do not retain the per-slot positions or selected operation/candidate arrays, so later physical-diversity and action-conditioned reward questions are reported as unavailable rather than reconstructed.

## 3. Polarization timeline

`UAVMove MA20` is the trailing 20-epoch mean of `uav_actual_movement_rate`; MA20 is undefined before epoch 20. Threshold crossings are the first strict crossing.

| run | MA20 final | <40% | <20% | <10% | <5% | >60% | >80% | >90% | >95% | classified onset |
|---|---|---|---|---|---|---|---|---|---|---|
| 5e-4 | 3.00% | 170 | 204 | 215 | 231 | 116 | — | — | — | MA20 < 10% @ 215 |
| 6e-4 | 95.12% | — | — | — | — | 49 | 180 | 212 | 299 | MA20 > 90% @ 212 |
| 1e-3 | 1.28% | 53 | 90 | 255 | 262 | — | — | — | — | MA20 < 10% @ 255 |

The timeline shows a non-monotonic LR-to-behavior map: 6e-4 becomes MOVE-like while both 5e-4 and 1e-3 become STAY-like. This is not a simple actor-LR monotone law.

### Metrics immediately around the classified polarization onset

| run | metric | pre window | post window | pre mean | post mean | delta |
|---|---|---|---|---|---|---|
| 5e-4 | uav_binary_entropy | 196-214 | 215-234 | 0.2963 | 0.1370 | -0.1593 |
| 5e-4 | uav_entropy | 196-214 | 215-234 | 1.4489 | 0.5923 | -0.8566 |
| 5e-4 | uav_kl | 196-214 | 215-234 | 0.0033 | 0.0010 | -0.0023 |
| 5e-4 | uav_clip | 196-214 | 215-234 | 0.0576 | 0.0279 | -0.0297 |
| 5e-4 | uav_grad | 196-214 | 215-234 | 0.2788 | 0.1542 | -0.1246 |
| 6e-4 | uav_binary_entropy | 193-211 | 212-231 | 0.3018 | 0.3167 | 0.0149 |
| 6e-4 | uav_entropy | 193-211 | 212-231 | 8.4639 | 8.3499 | -0.1140 |
| 6e-4 | uav_kl | 193-211 | 212-231 | 0.0035 | 0.0073 | 0.0038 |
| 6e-4 | uav_clip | 193-211 | 212-231 | 0.0621 | 0.0851 | 0.0230 |
| 6e-4 | uav_grad | 193-211 | 212-231 | 0.2923 | 0.2768 | -0.0155 |
| 1e-3 | uav_binary_entropy | 236-254 | 255-274 | 0.2253 | 0.1596 | -0.0656 |
| 1e-3 | uav_entropy | 236-254 | 255-274 | 1.2784 | 0.6908 | -0.5876 |
| 1e-3 | uav_kl | 236-254 | 255-274 | 0.0019 | 0.0047 | 0.0028 |
| 1e-3 | uav_clip | 236-254 | 255-274 | 0.0492 | 0.0332 | -0.0160 |
| 1e-3 | uav_grad | 236-254 | 255-274 | 0.2792 | 0.2029 | -0.0763 |

For direct entropy interpretation, `uav_binary_entropy` is the logged binary STAY/MOVE head entropy; `uav_entropy` is the full UAV branch entropy used in PPO and may include the candidate head when MOVE is active. The pre/post rows are descriptive temporal comparisons, not causal estimates.

POLARIZATION_PRECEDED_BY_ENTROPY_COLLAPSE = NO
POLARIZATION_PRECEDED_BY_KL_SPIKE = NO
POLARIZATION_PRECEDED_BY_GRAD_COLLAPSE = NO

The operational checks use transparent diagnostics: binary entropy <0.10, UAV branch KL >0.10, and UAV head gradient <25% of that run's mean. They are used only to test temporal precedence, not to claim a universal causal threshold.

| run | polarization onset | entropy <0.10 | UAV KL >0.10 | gradient floor | gradient collapse |
|---|---|---|---|---|---|
| 5e-4 | 215 | 228 | — | 0.06395 | 227 |
| 6e-4 | 212 | — | — | 0.07640 | — |
| 1e-3 | 255 | 273 | — | 0.07398 | 276 |

## 4. Branch optimization statistics

### 5e-4

| window | Return mean | Return std | CacheChange | UAVMove | joint KL | cache KL | UAV KL | cache clip | UAV clip | cache H | UAV H | cache grad | UAV grad | shared grad | cache/UAV grad | early-stop | PPO epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-60 | 1156.0082 | 1482.9602 | 59.08% | 47.36% | 0.0518 | 0.0366 | 0.0094 | 43.57% | 11.46% | 16.1459 | 5.2530 | 0.5662 | 0.2556 | 0.6667 | 2.2533 | 6.67% | 3.9167 |
| 61-120 | 2638.9803 | 403.7785 | 52.45% | 53.40% | 0.0503 | 0.0342 | 0.0103 | 40.25% | 10.87% | 11.6394 | 5.2891 | 0.7575 | 0.2899 | 0.6380 | 2.6770 | 5.00% | 3.9167 |
| 121-180 | 3261.0114 | 221.0765 | 49.32% | 46.85% | 0.0584 | 0.0357 | 0.0092 | 39.22% | 10.91% | 9.4519 | 4.7015 | 0.8183 | 0.3312 | 0.6490 | 2.5571 | 1.67% | 3.9667 |
| 181-240 | 3189.1904 | 327.6136 | 48.64% | 11.10% | 0.0568 | 0.0369 | 0.0050 | 39.50% | 5.94% | 8.9118 | 1.5109 | 0.8860 | 0.2510 | 0.6285 | 4.7246 | 5.00% | 3.9333 |
| 241-300 | 3342.9418 | 173.3503 | 52.20% | 3.67% | 0.0555 | 0.0335 | 0.0057 | 35.39% | 3.77% | 8.2970 | 0.5553 | 0.9158 | 0.1515 | 0.5257 | 8.4475 | 5.00% | 3.9500 |
| 281-300 | 3328.4534 | 116.0456 | 54.22% | 3.00% | 0.0409 | 0.0306 | 0.0045 | 31.56% | 2.83% | 8.2689 | 0.4657 | 0.9139 | 0.1298 | 0.4432 | 12.0651 | 0.00% | 4.0000 |

### 6e-4

| window | Return mean | Return std | CacheChange | UAVMove | joint KL | cache KL | UAV KL | cache clip | UAV clip | cache H | UAV H | cache grad | UAV grad | shared grad | cache/UAV grad | early-stop | PPO epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-60 | 1175.1500 | 1225.6041 | 63.76% | 56.33% | 0.0719 | 0.0419 | 0.0087 | 46.46% | 13.49% | 16.5705 | 6.1337 | 0.5636 | 0.2642 | 0.6902 | 2.1480 | 10.00% | 3.8167 |
| 61-120 | 2281.9896 | 497.3432 | 62.89% | 73.85% | 0.0867 | 0.0460 | 0.0107 | 46.21% | 12.16% | 14.8628 | 7.4345 | 0.7569 | 0.3156 | 0.7698 | 2.4583 | 23.33% | 3.6667 |
| 121-180 | 2828.5847 | 399.5150 | 48.88% | 72.54% | 0.0688 | 0.0475 | 0.0105 | 45.17% | 14.86% | 11.6414 | 7.3022 | 0.8469 | 0.3807 | 0.7264 | 2.2810 | 23.33% | 3.7000 |
| 181-240 | 2994.0127 | 357.4079 | 57.09% | 89.68% | 0.0784 | 0.0497 | 0.0061 | 46.75% | 7.65% | 11.1472 | 8.3931 | 0.9416 | 0.2767 | 0.6510 | 3.4903 | 21.67% | 3.7000 |
| 241-300 | 3170.0232 | 265.4942 | 61.18% | 90.22% | 0.0521 | 0.0414 | 0.0073 | 43.08% | 8.37% | 8.9737 | 8.4209 | 0.9313 | 0.2907 | 0.5492 | 3.4743 | 5.00% | 3.9500 |
| 281-300 | 3194.6056 | 182.4204 | 61.32% | 95.12% | 0.0424 | 0.0415 | 0.0064 | 42.30% | 6.83% | 8.5960 | 8.6361 | 0.9249 | 0.2068 | 0.4745 | 4.5876 | 0.00% | 4.0000 |

### 1e-3

| window | Return mean | Return std | CacheChange | UAVMove | joint KL | cache KL | UAV KL | cache clip | UAV clip | cache H | UAV H | cache grad | UAV grad | shared grad | cache/UAV grad | early-stop | PPO epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-60 | 1334.5588 | 1250.2108 | 63.87% | 43.43% | 0.1175 | 0.0697 | 0.0103 | 54.08% | 15.34% | 16.6377 | 5.0944 | 0.5522 | 0.2802 | 0.6289 | 1.9793 | 65.00% | 2.9500 |
| 61-120 | 2423.8112 | 806.8895 | 60.48% | 26.99% | 0.1078 | 0.0683 | 0.0104 | 51.85% | 11.90% | 13.1259 | 3.3470 | 0.7867 | 0.3307 | 0.5531 | 2.5043 | 61.67% | 2.8667 |
| 121-180 | 3240.3986 | 197.7590 | 49.38% | 37.91% | 0.1082 | 0.0588 | 0.0095 | 47.61% | 12.88% | 9.1333 | 4.0166 | 0.7743 | 0.3230 | 0.4869 | 2.4674 | 40.00% | 3.4000 |
| 181-240 | 3049.6248 | 278.9431 | 56.71% | 27.09% | 0.1177 | 0.0670 | 0.0120 | 50.84% | 12.70% | 10.3739 | 3.0613 | 0.9026 | 0.3907 | 0.5737 | 2.3507 | 65.00% | 3.0333 |
| 241-300 | 3208.3232 | 175.9417 | 65.87% | 3.59% | 0.1083 | 0.0720 | 0.0023 | 53.03% | 2.46% | 10.3338 | 0.5444 | 1.0584 | 0.1550 | 0.4651 | 11.0780 | 55.00% | 3.0833 |
| 281-300 | 3242.6103 | 134.2755 | 68.33% | 1.28% | 0.1007 | 0.0672 | 0.0007 | 51.62% | 1.13% | 9.6649 | 0.1944 | 1.0479 | 0.0664 | 0.4495 | 20.7758 | 40.00% | 3.3500 |

The logged gradient norms are the norms returned by `clip_grad_norm_` before optimizer rescaling, so values above max_grad_norm=0.5 are pre-clipping norms rather than post-clipping values.

## 5. Entropy / KL / gradient analysis

Cache KL, Cache clip fraction, and Cache head gradient are generally larger than their UAV counterparts in the same windows. This indicates different branch update pressure, but it does not establish that Cache's contribution to the shared encoder is larger because the shared norm is only logged after the joint loss backward.

## 6. Shared encoder analysis

| run | cache/UAV grad | cache grad | UAV grad | shared grad | cache KL | UAV KL | cache clip | UAV clip |
|---|---|---|---|---|---|---|---|---|
| 5e-4 | 12.0651 | 0.9139 | 0.1298 | 0.4432 | 0.0306 | 0.0045 | 31.56% | 2.83% |
| 6e-4 | 4.5876 | 0.9249 | 0.2068 | 0.4745 | 0.0415 | 0.0064 | 42.30% | 6.83% |
| 1e-3 | 20.7758 | 1.0479 | 0.0664 | 0.4495 | 0.0672 | 0.0007 | 51.62% | 1.13% |

`cache_head_gradient_norm / uav_head_gradient_norm` is computed per logged update when both are finite, then averaged within the window. It is a branch-only norm ratio, not a decomposition of the shared encoder gradient.

SHARED_GRAD_COSINE_HISTORIC_AVAILABLE = NO
The persisted update log has only the joint `shared_actor_gradient_norm`; it does not contain `grad_shared_from_L_cache`, `grad_shared_from_L_uav`, or per-branch shared-gradient vectors. The source architecture makes such an audit feasible in a future minimal instrumentation run: branch losses are separately formed and both reach the shared encoder, while branch-only groups are disjoint. No future run was started here.

CACHE_SHARED_GRAD_DOMINANCE_SUPPORTED = PARTIAL
Evidence for the partial label is that Cache-only head KL/clip/gradient norms exceed UAV-only norms in the logged windows. Counter-evidence is decisive for attribution: the shared norm is a joint-loss norm and cannot tell whether Cache or UAV supplied the shared component; no cosine or per-loss shared gradient was historically recorded.

## 7. Global advantage observability

`critic_diagnostics.jsonl` stores raw global advantage and return target, while `episode_metrics.jsonl` stores only aggregate MOVE/STAY advantage means and medians plus selected/request counts. It does not store a per-transition UAV operation label alongside raw advantage, per-transition sign, or per-transition UAV log-probability. The source logging path repeats each transition's one global raw advantage into the selected MOVE/STAY aggregate bucket; this is useful as a descriptive aggregate but is not a branch-specific advantage.

| run | window | MOVE mean A | MOVE median A | STAY mean A | STAY median A | MOVE selected count | UAV request count |
|---|---|---|---|---|---|---|---|
| 5e-4 | 1-60 | 33.076 | 70.662 | 36.125 | 72.554 | 284.2 | 284.2 |
| 5e-4 | 61-120 | 32.824 | 55.493 | 28.912 | 50.101 | 320.4 | 320.4 |
| 5e-4 | 121-180 | 13.769 | 19.712 | 14.080 | 20.173 | 281.1 | 281.1 |
| 5e-4 | 181-240 | -12.677 | -10.219 | -11.042 | -6.027 | 66.6 | 66.6 |
| 5e-4 | 241-300 | -2.567 | 0.646 | -6.809 | -5.364 | 22.0 | 22.0 |
| 5e-4 | 281-300 | -0.948 | 8.728 | -5.611 | -3.700 | 18.0 | 18.0 |
| 6e-4 | 1-60 | 41.946 | 68.153 | 42.019 | 70.379 | 337.9 | 337.9 |
| 6e-4 | 61-120 | 26.467 | 49.051 | 24.186 | 45.929 | 443.1 | 443.1 |
| 6e-4 | 121-180 | 10.603 | 16.595 | 7.549 | 15.097 | 435.2 | 435.2 |
| 6e-4 | 181-240 | -7.834 | -4.487 | -9.105 | -5.375 | 538.1 | 538.1 |
| 6e-4 | 241-300 | -0.905 | 2.561 | -1.073 | -0.153 | 541.3 | 541.3 |
| 6e-4 | 281-300 | 5.627 | 8.269 | 6.456 | 5.688 | 570.7 | 570.7 |
| 1e-3 | 1-60 | 31.360 | 75.339 | 32.825 | 75.741 | 260.6 | 260.6 |
| 1e-3 | 61-120 | 34.402 | 66.412 | 33.485 | 65.300 | 162.0 | 162.0 |
| 1e-3 | 121-180 | 12.385 | 17.406 | 12.615 | 17.422 | 227.5 | 227.5 |
| 1e-3 | 181-240 | -18.174 | -16.256 | -19.234 | -17.224 | 162.5 | 162.5 |
| 1e-3 | 241-300 | -6.696 | -3.486 | -5.813 | -2.592 | 21.5 | 21.5 |
| 1e-3 | 281-300 | -6.786 | -3.527 | -5.285 | -2.518 | 7.7 | 7.7 |

The requested positive-advantage fraction and negative-advantage fraction cannot be recovered without inventing a distribution from aggregate means/medians. Therefore:

HISTORIC_UAV_ADVANTAGE_ANALYSIS_AVAILABLE = NO (aggregate-only partial observability)

Missing fields for the full requested analysis: per-slot selected UAV operation (STAY/MOVE), per-slot raw advantage/return target joined to that operation, per-slot old UAV log-probability, and per-slot sign indicators. The global V(s)/GAE path is visible; a Q(s,a_cache,a_uav) estimator is not present.

## 8. Reward decomposition

The actual accounting identity is directly checkable from slot logs: `reward = q - cache_cost - movement_cost`; episode logs likewise expose `episode_return = q_total - cache_cost - movement_cost`.
Maximum absolute slot accounting error = 0.000000000000 across 270,000 slots; records exceeding 1e-10 = 0.

### 5e-4: overall decomposition by training window

| window | episode Return | q_total | cache cost | UAV movement cost | slot q mean | movement-bearing slot fraction |
|---|---|---|---|---|---|---|
| 1-60 | 1156.0082 | 1273.2066 | 106.3379 | 10.860612 | 4.2440 | 76.93% |
| 61-120 | 2638.9803 | 2730.2365 | 80.1686 | 11.087707 | 9.1008 | 88.61% |
| 121-180 | 3261.0114 | 3340.0435 | 69.4573 | 9.574822 | 11.1335 | 80.61% |
| 181-240 | 3189.1904 | 3256.4262 | 64.6971 | 2.538690 | 10.8548 | 20.49% |
| 241-300 | 3342.9418 | 3418.9226 | 75.1393 | 0.841537 | 11.3964 | 6.72% |
| 281-300 | 3328.4534 | 3408.8907 | 79.7458 | 0.691515 | 11.3630 | 5.37% |

### 6e-4: overall decomposition by training window

| window | episode Return | q_total | cache cost | UAV movement cost | slot q mean | movement-bearing slot fraction |
|---|---|---|---|---|---|---|
| 1-60 | 1175.1500 | 1306.2476 | 117.7408 | 13.356750 | 4.3542 | 79.91% |
| 61-120 | 2281.9896 | 2414.8673 | 114.9957 | 17.882077 | 8.0496 | 94.28% |
| 121-180 | 2828.5847 | 2925.6830 | 79.3206 | 17.777778 | 9.7523 | 93.50% |
| 181-240 | 2994.0127 | 3103.4455 | 86.6054 | 22.827475 | 10.3448 | 98.85% |
| 241-300 | 3170.0232 | 3282.9592 | 91.0227 | 21.913228 | 10.9432 | 99.01% |
| 281-300 | 3194.6056 | 3307.6585 | 90.4469 | 22.605935 | 11.0255 | 99.68% |

### 1e-3: overall decomposition by training window

| window | episode Return | q_total | cache cost | UAV movement cost | slot q mean | movement-bearing slot fraction |
|---|---|---|---|---|---|---|
| 1-60 | 1334.5588 | 1458.6922 | 113.3914 | 10.742087 | 4.8623 | 68.11% |
| 61-120 | 2423.8112 | 2523.8321 | 93.4169 | 6.604033 | 8.4128 | 47.95% |
| 121-180 | 3240.3986 | 3313.5442 | 64.8185 | 8.327037 | 11.0451 | 71.43% |
| 181-240 | 3049.6248 | 3135.6853 | 80.4371 | 5.623347 | 10.4523 | 50.89% |
| 241-300 | 3208.3232 | 3303.7243 | 94.6079 | 0.793137 | 11.0124 | 6.87% |
| 281-300 | 3242.6103 | 3338.8032 | 95.9147 | 0.278190 | 11.1293 | 2.30% |

The requested `mean reward | MOVE` versus `mean reward | STAY`, and corresponding q_total/movement_cost/delay/energy comparisons, are `NOT_AVAILABLE` for the three historical runs: slot_metrics has only aggregate `uav_movement_count` and no per-UAV operation/action identity, while episode_metrics has no per-slot delay/energy totals. A slot with movement_count=1 also mixes one UAV MOVE and one UAV STAY, so it cannot be relabeled as a single MOVE sample. The movement-bearing slot fraction shown above is only an explicitly labeled aggregate proxy, not the requested conditional comparison.

The following applies to every run and every requested training window:

| requested conditional metric | MOVE | STAY |
|---|---|---|
| mean reward | NOT_AVAILABLE | NOT_AVAILABLE |
| mean q_total / base utility | NOT_AVAILABLE | NOT_AVAILABLE |
| mean movement_cost | NOT_AVAILABLE | NOT_AVAILABLE |
| mean delay | NOT_AVAILABLE | NOT_AVAILABLE |
| mean vehicle energy | NOT_AVAILABLE | NOT_AVAILABLE |

This avoids the invalid causal inference `E[reward|MOVE] > E[reward|STAY] => MOVE causes the benefit`; even with full labels, state-selection confounding would remain.

## 9. Physical UAV behavior

| run | final60 slots | uav_movement_count distribution | position/distance fields | action/selected fields |
|---|---|---|---|---|
| 5e-4 | 18000 | {"0": 16790, "1": 1098, "2": 112} | NOT_AVAILABLE | NOT_AVAILABLE |
| 6e-4 | 18000 | {"0": 178, "1": 3165, "2": 14657} | NOT_AVAILABLE | NOT_AVAILABLE |
| 1e-3 | 18000 | {"0": 16764, "1": 1180, "2": 56} | NOT_AVAILABLE | NOT_AVAILABLE |

The actual movement-rate metric is physical in the narrow sense that it compares actual before/after positions with the production tolerance; it is not merely a selected-operation count. However, the saved records do not contain those positions, displacement distances, unique-position sets, or run-length sequences. Thus the frequency of actual changes is known, but spatial diversity and movement-distance distribution are not.

STAY_COLLAPSE_PHYSICAL = UNCLEAR
MOVE_SATURATION_PHYSICAL = UNCLEAR

Requested final60 diversity fields:

| field | 5e-4 | 6e-4 | 1e-3 |
|---|---|---|---|
| unique UAV positions | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |
| movement distance distribution | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |
| STAY run length | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |
| MOVE run length | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |
| action-category distribution | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |
| position entropy | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |

The 5e-4 and 1e-3 runs are clearly STAY-like by actual movement rate, and 6e-4 is clearly high-frequency actual movement by the logged rate. The requested stronger claims about few fixed positions, distinct positions, distance magnitude, and MOVE/STAY run lengths cannot be made from these logs.

## 10. Objective sensitivity

The source system objective is `G(t)=Q(t)-C_cache(t)-C_move(t)`, with Q computed by actual Matching/KKT and movement cost computed from final old-to-new UAV position. The air-channel path uses horizontal UAV distance and feeds rate, delay, vehicle energy, and task utility. Therefore the objective is position-sensitive in code, but its empirical sensitivity is not identifiable from these three logs because positions and per-task delay/energy are not persisted.

The existing repository contains a prior bounded single-candidate probe (not re-executed in this audit) showing a 3.75 m UAV candidate can change downstream channel/KKT/Q and incur about 0.00275 movement cost in that fixture. That artifact is contextual code-path evidence only; it is not a three-run counterfactual estimate and is not used to manufacture MOVE/STAY conditional means.

| run | Final20 Return | Final20 q_total | Final20 cache cost | Final20 movement cost | Final20 UAVMove |
|---|---|---|---|---|---|
| 5e-4 | 3328.4534 | 3408.8907 | 79.7458 | 0.691515 | 3.00% |
| 6e-4 | 3194.6056 | 3307.6585 | 90.4469 | 22.605935 | 95.12% |
| 1e-3 | 3242.6103 | 3338.8032 | 95.9147 | 0.278190 | 1.28% |

The final20 Return means are not identical: 5e-4 and 1e-3 are around 3.3k, while 6e-4 is lower by roughly 4%. The q_total term, not movement cost, carries the scale of these differences. That is evidence against a pure movement-cost explanation, but it is not enough to prove a flat UAV objective because the policies visit different states and Cache configurations.

OBJECTIVE_FLAT_WRT_UAV_BEHAVIOR = PARTIAL
Rationale: the two STAY-like policies and MOVE-like policy have broadly similar final Return scale while UAVMove differs by orders of magnitude, and logged movement costs are small relative to q_total. Counter-evidence/limit: no same-state STAY-vs-MOVE actual Q/delay/energy comparison exists in the historical runs, so flatness and bimodality remain unproven.

## 11. Counterfactual feasibility audit

Static feasibility is YES. `enumerate_uav_primitive_actions` produces a STAY candidate plus feasible MOVE candidates without advancing the environment; `build_policy_primitive_configuration` and `execute_slot(configuration_override=...)` copy/validate the supplied previous state and can feed a fixed final UAV position into the same Matching/KKT path. This supports the minimum future path: snapshot one logged state s_t and Cache state a_C, enumerate one or more feasible UAV candidates, build two independent configuration overrides (STAY and MOVE), and call the same slot evaluator on copies while using the same actual task/vehicle/scenario inputs. Do not mutate shared state or reuse any mutable RNG-bearing search object.

The current persisted V10-B logs do not contain the state/action/position bundle needed to run that comparison retrospectively, and no one-slot counterfactual was executed in this stage.

COUNTERFACTUAL_UAV_EVAL_FEASIBLE = YES (static API feasibility; not executed)

## 12. Root-cause conclusion

G) Action/mask/log-prob/action-mapping bug is not supported: the static contracts align, selected MOVE is required to have a feasible production candidate, actual movement is position-based, and branch old/new log-probability formulas match the sampled hierarchy.

B) Premature entropy collapse is not established as the common primary cause: entropy changes must be read around each run's onset, and the three runs do not share one consistent preceding-collapse pattern.

C/D) A simple UAV-update-too-weak or UAV-update-too-strong story is also not sufficient: branch KL/clip/gradient pressure differs from Cache and depends on LR, but the behavior is non-monotonic (6e-4 MOVE-like, 5e-4 and 1e-3 STAY-like) and the branch KL is not a universal pre-onset spike. The logs support path-dependent update sensitivity, not a single monotone magnitude diagnosis.

E/F) Shared encoder interference/cache dominance is only partial: Cache-only head diagnostics are larger, but historical shared-gradient decomposition/cosine is missing, so dominance cannot be attributed to the shared representation.

H/I) Objective extremeness/flatness is plausible but only partial: movement cost is small relative to q_total and final returns are of similar scale, while code and prior bounded probe show position-sensitive downstream Q. The missing same-state counterfactual prevents a stronger conclusion.

A) Global-advantage credit assignment is structurally present because one global V(s)/GAE advantage is reused by both branches, and full action-conditioned sign statistics are missing. But the historical aggregate MOVE/STAY advantage means/medians are not sufficient to show weak, mixed, or biased UAV signal. Therefore it is not promoted to YES solely because polarization occurred.

PRIMARY_ROOT_CAUSE = PATH_DEPENDENT_BRANCHWISE_POLICY_DRIFT (evidence-supported synthesis; not one of the five implementation options by itself)
SECONDARY_ROOT_CAUSE = INSUFFICIENT_UAV_ACTION_CONDITIONAL_OBSERVABILITY (prevents separating global-advantage variance, shared-gradient conflict, and objective flatness)

CREDIT_ASSIGNMENT_PROBLEM_SUPPORTED = PARTIAL
The partial label reflects the global V(s)/GAE structure plus missing per-action signal diagnostics, not a claim that credit assignment is proven as the cause. Direct support for Cache dominance and action-conditional advantage weakness is incomplete.

## 13. Recommended V10-C direction

RECOMMENDED_V10C_DIRECTION = OPTION_A

Recommendation is conditional and instrumentation-first: preserve the single global reward and add a branch-specific baseline only if its estimator is shown to preserve the global-objective policy-gradient expectation. The current critic is V(s), not Q(s,a_cache,a_uav); a counterfactual baseline therefore needs a new action-conditional Q/advantage estimator or an unbiased/compatible estimator, not a heuristic bonus and not an independent UAV reward. Before training, add minimal per-transition logging for operation, old/new UAV log-probability, raw/normalized advantage, return target, positions, candidate distance, q/delay/energy, and per-loss shared gradients/cosine. This is the smallest evidence-gathering path for distinguishing A from B/C/D/E/F/H/I.

OPTION_B is not selected because common premature entropy collapse is not established. OPTION_C is not selected as the sole diagnosis because update magnitude is not monotone with the observed behavior. OPTION_D lacks historical shared-gradient decomposition. OPTION_E lacks same-state counterfactual evidence.

## 14. Required final status

STAGE = V10-C_UAV_POLARIZATION_ROOT_CAUSE
RUN_ALIGNMENT_VALID = YES
UAV_MOVE_METRIC_SEMANTICS_VALID = YES
UAV_ACTION_MAPPING_VALID = YES
UAV_MASK_VALID = YES
UAV_LOGPROB_VALID = YES
STAY_COLLAPSE_PHYSICAL = UNCLEAR
MOVE_SATURATION_PHYSICAL = UNCLEAR
POLARIZATION_PRECEDED_BY_ENTROPY_COLLAPSE = NO
POLARIZATION_PRECEDED_BY_KL_SPIKE = NO
POLARIZATION_PRECEDED_BY_GRAD_COLLAPSE = NO
CACHE_SHARED_GRAD_DOMINANCE_SUPPORTED = PARTIAL
SHARED_GRAD_COSINE_HISTORIC_AVAILABLE = NO
HISTORIC_UAV_ADVANTAGE_ANALYSIS_AVAILABLE = NO
OBJECTIVE_FLAT_WRT_UAV_BEHAVIOR = PARTIAL
COUNTERFACTUAL_UAV_EVAL_FEASIBLE = YES
CREDIT_ASSIGNMENT_PROBLEM_SUPPORTED = PARTIAL
PRIMARY_ROOT_CAUSE = PATH_DEPENDENT_BRANCHWISE_POLICY_DRIFT
SECONDARY_ROOT_CAUSE = INSUFFICIENT_UAV_ACTION_CONDITIONAL_OBSERVABILITY
RECOMMENDED_V10C_DIRECTION = OPTION_A
V10C_IMPLEMENTATION_STARTED = NO
TRAINING_EXECUTED = NO
EVALUATION_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
REPORT = C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\v10c_uav_polarization_root_cause.md
TEMP_FILES_REMOVED = YES
GIT_DIFF_CHECK = PASS
PUSHED = NO

## Appendix: persisted artifact availability

| run | latest checkpoint | best checkpoint | episode records | update records | slot records | critic records |
|---|---|---|---|---|---|---|
| 5e-4 | YES | YES | 300 | 300 | 90000 | 90000 |
| 6e-4 | YES | YES | 300 | 300 | 90000 | 90000 |
| 1e-3 | YES | YES | 300 | 300 | 90000 | 90000 |

No temporary CSV, debug output, or run artifact was created by this audit script.
