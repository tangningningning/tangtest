# Macro State Context Augmentation Gate

**Gate: FAIL.** B and C did not improve cross-realization value prediction or action-conditioned target aliasing. The formal Macro observation remains 400D (v9/ckpt29); no production network change and no fresh5 run were made.

## Same-policy trace

Captured 300 Macro frames across episode_20001, episode_20002, episode_20003, episode_20004, episode_20005, one deterministic frozen-policy rollout per realization. Matching calls: 1,500; KKT calls: 225,612; Exact-Q executions: 0. Both the formal SMDP GAE target and reward-only SMDP return-to-go are stored.

Trace: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\macro_same_policy_context_trace.npz` (348,302 bytes; SHA-256 `1d573b2686317640aee045399e373c269646c0ffe853a775412691f4d0e4c58d`). No model weights are included.

## Step 50 unseen TEST_EV

Lists are in test episode order 20002, 20003, 20004, 20005.

| Target | A (400D) | B (+REQ, 403D) | C (+REQ+WORK, 406D) |
|---|---|---|---|
| Formal GAE | -0.0924, 0.0868, -0.0213, -0.4690 | -0.0934, 0.0868, -0.0249, -0.4750 | -0.0940, 0.0866, -0.0290, -0.4779 |
| Realized return-to-go | 0.0292, -0.0147, 0.1129, 0.0264 | 0.0292, -0.0149, 0.1129, 0.0270 | 0.0289, -0.0145, 0.1097, 0.0264 |

## Action-conditioned normalized target gaps

Each cell is median / P90, using the same KEEP/UPDATE group and requested duration, nearest prior state, and per-test-episode target standard deviation.

| Target | A | B | C |
|---|---:|---:|---:|
| Formal GAE | 2.1833 / 6.6817 | 2.1833 / 6.6817 | 2.1833 / 6.6817 |
| Realized return-to-go | 1.9653 / 3.1452 | 1.9653 / 3.1452 | 1.9653 / 3.1452 |

Across step50/100/200, the appended contexts changed test metrics only marginally. A/B/C selected the same nearest action-conditioned states, so both target-gap distributions are identical across candidates. REQ-only and REQ+WORK gates both fail.

## Decision

AGGREGATE_DEMAND_CONTEXT_INSUFFICIENT; aggregate request/workload context did not improve cross-realization prediction or action-conditioned aliasing. Keep formal Macro state unchanged; next diagnostic may evaluate recent causal context.

No source or config changes were made in this task. Temporary scripts, per-episode sidecars, Python caches, and `.pyc` files were removed.
