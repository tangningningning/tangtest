# Joint V9 epoch-222 UAV deterministic zero-move diagnostic

## Fixed inputs

- Checkpoint: `runs/ppo_v9_joint_cache_uav_stage1_ep50_20001_20050/checkpoint_ep222_frozen.pt`
- Development pool: `runs/ppo_v6_dev_eval_pool_30001_30020`
- Seeds: `30001..30020`
- Policy mode: deterministic
- Policy schema: `adaptive_joint_cache_replica_aware_uav_operation_v9_popart`
- Configuration control mode: `policy_primitive_v9_joint_cache_uav_operation`
- Replay consistency: exact existing deterministic returns and selected-action SHA-256 matched for all 20 episodes.

## UAV operation-head evidence

The fixed pool contains 300 slots × 2 UAV heads × 20 seeds = **12,000** decisions.

| quantity | value |
| --- | ---: |
| total UAV decisions | 12000 |
| decisions with a feasible MOVE candidate | 12000 |
| MOVE-feasible rate | 1.0 |
| MOVE masked by feasibility rate | 0.0 |
| mean P(MOVE) | 0.3459138174528877 |
| median P(MOVE) | 0.3604547381401062 |
| P25 P(MOVE) | 0.29008931666612625 |
| P75 P(MOVE) | 0.40082447975873947 |
| max P(MOVE) | 0.4349684417247772 |
| fraction P(MOVE) > 0.5 | 0.0 |
| fraction P(MOVE) > 0.4 | 0.27125 |
| fraction P(MOVE) > 0.3 | 0.52 |
| UAV_0 mean P(MOVE) | 0.2903103098471960 |
| UAV_1 mean P(MOVE) | 0.4015173250585795 |

The formal deterministic decoder matched `argmax([P(STAY), P(MOVE)])` on all 12,000 decisions and mismatched on none. It selected MOVE zero times, and actual UAV movement was zero times.

As the permitted minimal stochastic check, one fixed replay of seed `30001` with the same checkpoint produced an actual UAV movement rate of `0.335`.

## Diagnosis

This is CASE_A: MOVE candidates were present, the feasibility mask did not suppress MOVE, and the deterministic decoder was correct, but P(MOVE) never exceeded 0.5. The evidence identifies deterministic STAY-mode policy collapse/selection as the cause of the zero-move evaluation. It does not indicate a decoder, candidate-generation, boundary, or execution-side failure.

No policy, PPO, reward, PopArt, Matching, KKT, or system-parameter files were changed. No new realization was generated.
