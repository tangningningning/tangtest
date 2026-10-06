# Macro State Target Predictability Diagnostic

## Decision

- `MACRO_STATE_INFORMATION_INSUFFICIENT = YES`
- `STALE_POLICY_TARGET_NONSTATIONARITY = POSSIBLE` as a secondary contributor.
- Root cause: cross-realization transfer remains poor under a same-policy, same-action-duration control, so stale policy targets alone do not explain the failure.
- Next: design a minimal context augmentation. This diagnostic did not change Macro or Micro observations or architecture.

## Cross-realization generalization

Historical targets used one common fresh critic initialization (seed 27005), verified exactly against the original run's episode 20001 step-0 metrics. Each pooled arm ran 200 full-batch critic-only Adam steps with LR 1e-3 and gradient clipping at 0.5.

| Train episodes | Test | Train EV | Test EV | Train corr | Test corr | Test RMSE | Test std ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| episode_20001 | episode_20002 | 0.990 | 0.016 | 0.995 | 0.214 | 174.83 | 0.387 |
| episode_20001, episode_20002 | episode_20003 | 0.964 | -0.562 | 0.982 | -0.542 | 266.13 | 0.383 |
| episode_20001, episode_20002, episode_20003 | episode_20004 | 0.959 | -2.745 | 0.979 | -0.049 | 214.84 | 1.608 |
| episode_20001, episode_20002, episode_20003, episode_20004 | episode_20005 | 0.949 | -3.990 | 0.974 | -0.362 | 144.36 | 1.668 |

The historical pairwise transfer EVs were 0.016, -0.083, -4.418, -1.038.

## State-target aliasing

For historical batches, median nearest-state distance was 0.587; normalized target-gap median/p90 was 1.021/3.349; Spearman rho was 0.053 (n=415).

The historical replay schema did not retain per-transition action group or duration, so action-conditioned matching of those targets is unavailable. In the same-policy control, action-conditioned nearest-neighbor target-gap median/p90 was 2.183/6.682 with 100% coverage. The deterministic policy selected KEEP 298 of 300 frames and requested duration 5 for every frame, so this conditional result primarily reflects KEEP/duration-5 coverage.

## Same-policy control

Checkpoint 29 (architecture 9) parameters were frozen across all five existing realizations. The state hash did not change; no optimizer updates or checkpoint writes occurred. Matching/KKT was run only for this authorized confirmation rollout.

Same-policy unseen TEST_EV: -0.374, 0.480, -22.157, -7.146.

## Targets, KL, and observation coverage

Historical target means: episode_20001=4.49, episode_20002=-41.04, episode_20003=-105.65, episode_20004=-86.53, episode_20005=-93.00.

Historical target stds: episode_20001=149.00, episode_20002=169.63, episode_20003=194.83, episode_20004=105.03, episode_20005=64.34.

Macro KL by episode: episode_20001=4.90e-05, episode_20002=2.04e-05, episode_20003=2.33e-05, episode_20004=1.22e-05, episode_20005=8.41e-06.

Micro KL by episode: episode_20001=5.67e-04, episode_20002=1.43e-03, episode_20003=4.04e-03, episode_20004=6.04e-04, episode_20005=3.28e-03.

The 400-dimensional Macro observation already contains the cache configuration, UAV positions, five-horizon vehicle positions, five-horizon predicted service popularity, horizon-validity mask, and time-to-go. It has no explicit recent request-rate, workload, or delay/energy/system-cost summaries, and no aggregate predicted demand scalar.

Suggested first feature: an aggregate predicted request count over the existing horizon, derived from fields already used by scenario/candidate construction; if supported by those same arrays, add aggregate predicted cycles/input workload. Do not duplicate the existing vehicle-position or popularity features.

## Formal configuration and files

`macro_critic_replay_history_episodes=0`; steps=50, LR=1e-3, max_grad_norm=0.5, critic input `2*x-1`, SiLU→SiLU, architecture 9, checkpoint 29. Micro and Actor settings remain as before.

JSON: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\macro_state_target_predictability_diagnostic.json`  
Markdown: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\macro_state_target_predictability_diagnostic.md`
