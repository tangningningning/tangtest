# Formal Macro SMDP Lambda-Return Fresh5

## Result

FORMAL_LAMBDA_RETURN_GATE = PARTIAL
SELECTED_FORMAL_TARGET = N2
FORMAL_CONFIG_FINAL_TARGET_MODE = two_step_smdp_td

The 1,017-transition offline safety gate passed. Fresh5 completed with the critic-specific duration-aware lambda-return. Local training-target fit improved, but the long-horizon alignment thresholds were not met. The result is PARTIAL, so N2 remains the formal target.

## Algorithm and execution

- Actor advantage remains duration_aware_smdp_gae; Actor GAE lambda remains 0.95.
- Critic target tested: smdp_lambda_return; critic lambda 0.7706141475, independent of Actor GAE.
- Derivation: (lambda + lambda^3 + lambda^5) / 3 = 0.5; solution 0.7706141475 (residual 6.87959e-11).
- The recurrence uses gamma from config and actual_duration, computes internally in float64, then casts the complete target to float32.
- Architecture version 9; checkpoint version 32; fresh initialization seed 27005; resume false; critic update steps 50; replay history 0.
- Fresh5 run: runs/formal_macro_smdp_lambda_target_ckptv32_archv9_fresh5
- Production trace: C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\formal_macro_smdp_lambda_target_ckptv32_archv9_fresh5\macro_transition_trace.npz; 523 transitions across five episodes.
- Recompute errors: critic target 0.0; realized return 0.0.
- Target detached and frozen through critic updates; before/after target hashes match for every episode.

## Static trace check

Source: runs/formal_macro_n2_target_ckptv31_archv9_short10/macro_transition_trace.npz.

- 1,017 real transitions; episode isolation, contiguous frame indices, terminal boundary, finite values, and actual-duration use passed.
- Requested duration counts: 1:354, 3:329, 5:334. Actual duration counts: 1:357, 2:3, 3:324, 5:333.
- Float32 repeat reconstruction max absolute error: 0.
- Maximum per-episode lambda-target/N2-target standard-deviation ratio: 0.974222, below the 3x safety limit.
- Global lambda target: mean -52.5902, std 141.4727, min -869.4514, max 197.4401; Pearson/Spearman with realized return: 0.3129 / 0.2169.

## Local training-target quality

Values compare critic predictions at snapshots 0 and 50 against the lambda training target. Snapshot 50 target explained variance was positive in every episode. The E2-E5 median rose from 0.2210 before updates to 0.7363 after updates. This measures fit to the training target, not long-horizon value alignment.

| Episode | Pre target EV | Post target EV | Pre target corr. | Post target corr. | Pre target std ratio | Post target std ratio |
|---|---:|---:|---:|---:|---:|---:|
| E1 | -0.000113 | 0.652114 | -0.228411 | 0.808989 | 0.000248 | 0.760517 |
| E2 | 0.088099 | 0.587884 | 0.412573 | 0.766941 | 0.126012 | 0.749197 |
| E3 | -0.057119 | 0.530863 | -0.038147 | 0.731233 | 0.203874 | 0.669274 |
| E4 | 0.353981 | 0.884702 | 0.596960 | 0.940891 | 0.645757 | 0.964828 |
| E5 | 0.654651 | 0.905234 | 0.809676 | 0.951503 | 0.840065 | 0.962603 |

## Long-horizon realized-return alignment

Values compare critic predictions with realized return-to-go. The decision gate requires either a median E2-E5 return-correlation gain of at least 0.10 or a median return-EV gain of at least 0.05 versus same-seed N2.

| Episode | Post return EV | Post return corr. | Post return std ratio | N2 post return EV | N2 post return corr. | N2 post return std ratio | Corr. delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | 0.030429 | 0.183819 | 0.125849 | 0.013936 | 0.118763 | 0.131753 | +0.065057 |
| E2 | 0.079718 | 0.301539 | 0.195672 | 0.064570 | 0.267620 | 0.183654 | +0.033919 |
| E3 | -0.147769 | -0.302373 | 0.186706 | -0.154540 | -0.279007 | 0.203056 | -0.023366 |
| E4 | -0.330512 | -0.162964 | 0.434589 | -0.379144 | -0.228614 | 0.428203 | +0.065650 |
| E5 | 0.324512 | 0.787698 | 0.243679 | 0.369637 | 0.748808 | 0.311685 | +0.038890 |

E2-E5 medians:
- Post target EV: lambda 0.736293, N2 0.672566, delta +0.063727.
- Post return EV: lambda -0.034026, N2 -0.044985, delta +0.010959.
- Post return correlation: lambda 0.069288, N2 0.019503, delta +0.049785.
- Post return std ratio: lambda 0.219676, N2 0.257371, delta -0.037695.

The correlation gain (+0.0498) is below +0.10 and the return-EV gain (+0.0110) is below +0.05. Gate D passes (4/4 E2-E5 episodes are no more than 0.15 worse), but gate C fails both alternatives.

## Stability and decision

- Macro Actor stable: True; per-episode KL 4.89677e-05, 2.01001e-05, 2.26254e-05, 1.08784e-05, 8.69015e-06, clip fraction 0 in every episode, no early stops.
- Micro stable: True; per-episode EV 0.856, 0.905, 0.734, 0.903, 0.920; no early stops.
- No Cache/Duration/UAV action collapse: True; patterns were NORMAL in all five episodes. Cache Keep 274, Update 249; UAV0 stay/move 760/740; UAV1 stay/move 727/773.
- Exact-Q environment executions: 0.
- No N3/N4/N8 search or lambda sweep was performed. This critic-specific duration-aware compound-return test was intended to retain the N2-scale target while adding a decaying long tail; Actor GAE remained unchanged.
- Decision: keep N2 as the formal target. Consider an independent long-horizon auxiliary value head as the next separate design question.

## Files

- JSON: C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\formal_macro_smdp_lambda_target_fresh5.json
- Markdown: C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\formal_macro_smdp_lambda_target_fresh5.md
