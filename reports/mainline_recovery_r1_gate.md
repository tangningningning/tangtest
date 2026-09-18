# MAINLINE_RECOVERY_R1 gate

Status: PASS.

The recovery path is a separately named V10-D-based schema using the verified actor/critic semantics. The only formal gate used one frozen `episode_20001` realization with 300 slots, seed 4401, and one normal PPO update. No missing episode was generated and no final test seeds were used.

| Check | Result |
|---|---|
| Observation/action dimensions | 400 / 11 |
| Actor/critic learning rates | 0.0006 / 0.0003 |
| Global GAE / PopArt critic | PASS |
| Branch-specific KL / target KL | `branch_specific` / 0.1 |
| Global critic schedule | 40 steps = 4 epochs × 10 minibatches |
| Q_cache steps/state | 0 / absent |
| Cache/UAV inactive gradients | `None` / `None` |
| NaN/Inf checks | none detected |
| V11-A CF/node-PPO/minibatch-KL fields | absent from gate logs |

The no-environment control probe constructed both actor branches as inactive. Under the old outer-break behavior it would execute 10 critic steps; after the fix it executes the full 40-step schedule. The actor loss remains graph-connected but inactive branch gradients are cleared to `None`, preserving stale-Adam protection.

The recovery gate did not assess performance, convergence, improvement, or superiority.

Formal artifacts:

- [mainline_recovery_r1_gate.json](C:/Users/tangning/Desktop/code/adaptive_cache_mec/reports/mainline_recovery_r1_gate.json)
- [system_config_ppo_recovery_r1_6e4.toml](C:/Users/tangning/Desktop/code/adaptive_cache_mec/configs/proposed/system_config_ppo_recovery_r1_6e4.toml)
