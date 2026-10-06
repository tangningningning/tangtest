# Formal Macro N2 Target Fresh5

**Result: `FORMAL_N2_TARGET_GATE = PASS`; `SELECTED_TARGET = N2`.**

`TRACE_PRECHECK_ALREADY_PASSED = YES` (523/523 real transitions checked before this run; not repeated).

## Run and controlled variables

- Run: `C:/Users/tangning/Desktop/code/adaptive_cache_mec/runs/formal_macro_n2_target_ckptv31_archv9_fresh5_retry2`. The requested unsuffixed directory already existed from a prior failed attempt, so this run used `retry2` and preserved both earlier directories.
- Fresh initialization seed 27005; episode seeds 20001–20005; resume false.
- The N1 formal run manifest and N2 run manifest match on seed, episode pool/count, architecture, Actor GAE, learning rates, Actor epochs, Critic steps, replay, duration candidates, Exact-Q, and resume state. The intended target mode and required checkpoint version are the only differences.
- Architecture v9, checkpoint v31; Macro Actor `DURATION_AWARE_SMDP_GAE`; Macro Critic `TWO_STEP_SMDP_TD`; 50 Macro Critic steps; replay history 0; Exact-Q disabled.
- Actor GAE, gamma/lambda, networks, learning rates, optimizer, losses, clipping, policies, Micro PPO, Matching, KKT, prediction, and environment configuration were held at the current formal settings.
- The earlier `retry1` stopped before E1 metrics were persisted and is not used as the N2 algorithm result. This report evaluates the completed `retry2` run.

## Trace and frozen-target checks

- `N2_TARGET_RECOMPUTE_MAX_ABS_ERROR = 0.0`; `REALIZED_RETURN_RECOMPUTE_MAX_ABS_ERROR = 0.0`.
- 523 transitions persisted in `macro_transition_trace.npz`, with episode counts `{1: 108, 2: 106, 3: 102, 4: 105, 5: 102}`. Trace includes `n2_target` (float32), `n2_bootstrap_old_value`, and `n2_bootstrap_frame_index` alongside observations, actual/requested duration, rewards, terminal markers, and frozen values.
- Target detach and SHA-256 before/after freeze checks passed for all five critic updates. The per-episode validator also checked terminal boundaries, actual-duration use, finite values, and `k+2` old-value linkage within each episode.
- `EXACT_Q_ENV_EXECUTIONS = 0`.

## Online metrics E1–E5

### Target metrics

| Episode | pre_update_target_ev | post_update_target_ev | pre_update_target_correlation | post_update_target_correlation | pre_update_target_std_ratio | post_update_target_std_ratio |
|---|---:|---:|---:|---:|---:|---:|
| E1 | -0.000089 | 0.548632 | -0.205067 | 0.741974 | 0.000217 | 0.698468 |
| E2 | 0.083406 | 0.473639 | 0.383674 | 0.688560 | 0.131088 | 0.666748 |
| E3 | -0.041997 | 0.516663 | -0.003491 | 0.720905 | 0.201471 | 0.665762 |
| E4 | 0.396722 | 0.828469 | 0.629930 | 0.910933 | 0.639408 | 0.874475 |
| E5 | 0.695078 | 0.893226 | 0.834099 | 0.945229 | 0.808736 | 0.930008 |

### Realized-return reference metrics

| Episode | pre_update_return_ev | post_update_return_ev | pre_update_return_correlation | post_update_return_correlation | pre_update_return_std_ratio | post_update_return_std_ratio |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 0.000001 | 0.013936 | 0.017274 | 0.118763 | 0.000041 | 0.131753 |
| E2 | 0.006087 | 0.064570 | 0.102346 | 0.267620 | 0.036108 | 0.183654 |
| E3 | 0.018925 | -0.154540 | 0.184715 | -0.279007 | 0.061448 | 0.203056 |
| E4 | -0.367394 | -0.379144 | -0.430159 | -0.228614 | 0.313098 | 0.428203 |
| E5 | 0.292935 | 0.369637 | 0.675907 | 0.748808 | 0.271042 | 0.311685 |

## Macro Critic snapshots

| Episode | Step | Target EV | Target corr | Target std ratio | Target MSE | Return EV | Return corr | Return std ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | 0 | -0.000089 | -0.205067 | 0.000217 | 4364.423373 | 0.000001 | 0.017274 | 0.000041 |
| E1 | 10 | 0.053694 | 0.675649 | 0.040978 | 4134.398232 | -0.001505 | -0.093464 | 0.007730 |
| E1 | 25 | 0.313689 | 0.669588 | 0.302629 | 3087.756478 | -0.009379 | -0.053609 | 0.057085 |
| E1 | 50 | 0.548632 | 0.741974 | 0.698468 | 1964.885289 | 0.013936 | 0.118763 | 0.131753 |
| E2 | 0 | 0.083406 | 0.383674 | 0.131088 | 6730.692764 | 0.006087 | 0.102346 | 0.036108 |
| E2 | 10 | 0.181449 | 0.468815 | 0.273012 | 5965.878633 | 0.039918 | 0.303011 | 0.075200 |
| E2 | 25 | 0.339745 | 0.590294 | 0.497013 | 4816.443352 | 0.084740 | 0.377946 | 0.136901 |
| E2 | 50 | 0.473639 | 0.688560 | 0.666748 | 3836.098167 | 0.064570 | 0.267620 | 0.183654 |
| E3 | 0 | -0.041997 | -0.003491 | 0.201471 | 15465.807548 | 0.018925 | 0.184715 | 0.061448 |
| E3 | 10 | 0.094164 | 0.312930 | 0.251599 | 13042.948722 | -0.051155 | -0.294946 | 0.076737 |
| E3 | 25 | 0.265426 | 0.525384 | 0.422420 | 10557.627500 | -0.145220 | -0.499162 | 0.128837 |
| E3 | 50 | 0.516663 | 0.720905 | 0.665762 | 6931.039118 | -0.154540 | -0.279007 | 0.203056 |
| E4 | 0 | 0.396722 | 0.629930 | 0.639408 | 2171.898309 | -0.367394 | -0.430159 | 0.313098 |
| E4 | 10 | 0.641387 | 0.809199 | 0.925027 | 1287.206634 | -0.410290 | -0.226423 | 0.452957 |
| E4 | 25 | 0.769198 | 0.877059 | 0.882975 | 828.767422 | -0.417413 | -0.266527 | 0.432365 |
| E4 | 50 | 0.828469 | 0.910933 | 0.874475 | 645.103629 | -0.379144 | -0.228614 | 0.428203 |
| E5 | 0 | 0.695078 | 0.834099 | 0.808736 | 2927.760503 | 0.292935 | 0.675907 | 0.271042 |
| E5 | 10 | 0.784426 | 0.902824 | 1.077936 | 2030.849187 | 0.397070 | 0.730190 | 0.361262 |
| E5 | 25 | 0.851309 | 0.922788 | 0.937913 | 1393.127408 | 0.382867 | 0.766180 | 0.314334 |
| E5 | 50 | 0.893226 | 0.945229 | 0.930008 | 999.356478 | 0.369637 | 0.748808 | 0.311685 |

## Actor, Micro, and action diagnostics

| Episode | Macro KL | Clip frac | Macro entropy | Advantage mean / std | Micro EV | Micro KL | Micro entropy | Micro value loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | 0.000049 | 0.000000 | 4.721355 | 4.492 / 149.000 | 0.855633 | 0.000567 | 4.243611 | 9025.085 |
| E2 | 0.000020 | 0.000000 | 4.726707 | -31.554 / 158.777 | 0.905403 | 0.001432 | 4.225111 | 8215.977 |
| E3 | 0.000023 | 0.000000 | 4.715298 | -101.770 / 199.646 | 0.734259 | 0.004042 | 4.251495 | 20297.810 |
| E4 | 0.000011 | 0.000000 | 4.717597 | 18.016 / 102.568 | 0.902758 | 0.000604 | 4.191922 | 4773.619 |
| E5 | 0.000009 | 0.000000 | 4.724061 | 23.752 / 102.568 | 0.920145 | 0.003284 | 4.104445 | 3095.953 |

| Episode | Cache KEEP / UPDATE | Requested duration counts | Actual duration counts | UAV0 STAY / MOVE | UAV1 STAY / MOVE |
|---|---:|---|---|---:|---:|
| E1 | 52 / 56 | `{"1": 40, "3": 40, "5": 28}` | `{"1": 40, "3": 40, "5": 28}` | 143 / 157 | 150 / 150 |
| E2 | 55 / 51 | `{"1": 38, "3": 39, "5": 29}` | `{"1": 38, "3": 39, "5": 29}` | 153 / 147 | 156 / 144 |
| E3 | 54 / 48 | `{"1": 37, "3": 30, "5": 35}` | `{"1": 38, "3": 29, "5": 35}` | 139 / 161 | 128 / 172 |
| E4 | 60 / 45 | `{"1": 40, "3": 32, "5": 33}` | `{"1": 40, "3": 31, "5": 33}` | 155 / 145 | 148 / 152 |
| E5 | 53 / 49 | `{"1": 34, "3": 36, "5": 32}` | `{"1": 35, "3": 35, "5": 32}` | 170 / 130 | 145 / 155 |

Aggregate duration histograms over the 523 transitions: requested `{"1": 189, "3": 177, "5": 157}`; actual `{"1": 191, "2": 1, "3": 174, "5": 157}`.

Actual-duration counts above come from the persisted transition trace. E4 includes one requested-3 / actual-2 transition; the episode summary histogram omitted that value.

## Formal N1 comparison (E2–E5 only)

N1 source: `runs/formal_macro_n1_target_ckptv30_archv9_fresh5`. These are matched episode seeds and the same formal fresh5 conditions; no N1 rerun was made.

| Metric | N1 median | N2 median | N2 − N1 |
|---|---:|---:|---:|
| PRE_UPDATE_TARGET_EV | -0.006114 | 0.240064 | +0.246178 |
| PRE_UPDATE_TARGET_CORRELATION | 0.019376 | 0.506802 | +0.487426 |
| PRE_UPDATE_TARGET_STD_RATIO | 0.099141 | 0.420440 | +0.321299 |
| POST_UPDATE_RETURN_EV | 0.004702 | -0.044985 | -0.049687 |
| POST_UPDATE_RETURN_CORRELATION | -0.170800 | 0.019503 | +0.190302 |
| POST_UPDATE_RETURN_STD_RATIO | 0.039123 | 0.257371 | +0.218248 |

Per-episode E2–E5 post-update return-correlation deltas (N2 − N1): `{"E2": 0.15941447416888066, "E3": -0.8904096914093289, "E4": 0.221190123867066, "E5": 1.3419237780116435}`; 3/4 are no worse than −0.15.

## Gate decision

| Criterion | Result | Evidence |
|---|---|---|
| E2–E5 target scale, all steps 0/10/25/50 | PASS | max std ratio 1.077936 (limit 3) |
| Median pre-update target EV | PASS | N2−N1 +0.246178 (minimum −0.10) |
| Long-return representation | PASS | post-return corr delta +0.190302; EV delta -0.049687 |
| E2–E5 per-episode return correlation | PASS | 3/4 meet the −0.15 floor (need 3) |
| Macro Actor / Micro stability | PASS | no early-stop; KL/clipping finite and within configured range |
| Cache, duration, UAV action coverage | PASS | KEEP/UPDATE, requested 1/3/5, and both UAV STAY/MOVE present in each episode |

The N2 median POST_UPDATE_RETURN_CORRELATION is higher than N1 by 0.190302, meeting the specified correlation alternative. Median POST_UPDATE_RETURN_EV is lower by 0.049687; it does not meet the EV improvement alternative. The target-scale, per-episode correlation, Actor/Micro, and action-coverage criteria pass, so the requested gate is PASS.

## Final fields

```text
TRACE_PRECHECK_ALREADY_PASSED = YES
MACRO_ACTOR_ADVANTAGE_MODE = DURATION_AWARE_SMDP_GAE
MACRO_CRITIC_TARGET_MODE = TWO_STEP_SMDP_TD
ARCHITECTURE_VERSION = 9
CHECKPOINT_VERSION = 31
FRESH5_RUN_PATH = C:/Users/tangning/Desktop/code/adaptive_cache_mec/runs/formal_macro_n2_target_ckptv31_archv9_fresh5_retry2
N2_TARGET_RECOMPUTE_MAX_ABS_ERROR = 0.0
TARGET_FROZEN_DURING_CRITIC_UPDATE = YES (E1-E5)
FORMAL_N2_TARGET_GATE = PASS
SELECTED_TARGET = N2
NEXT_DECISION = Directly enter short10; short10 was not launched in this fresh5 task.
REPORT_PATH_JSON = reports/formal_macro_n2_target_fresh5.json
REPORT_PATH_MD = reports/formal_macro_n2_target_fresh5.md
```
