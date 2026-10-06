# Formal Dual-Value Macro Critic Fresh5

## Result

FRESH5_GATE = PARTIAL_LONG_VALUE_AUXILIARY_WEAK
LONG_VALUE_AUXILIARY_EFFECT = WEAK
FORMAL_MAIN_CRITIC = TWO_STEP_SMDP_TD
FORMAL_TRAINING_PATH = USABLE
SHORT10_EXECUTED = YES

Fresh5 completed with architecture v10/checkpoint v33. N2, Macro Actor, Micro, action, and Exact-Q checks passed. The independent Long Critic missed the fresh5 auxiliary median thresholds. It remains diagnostic and does not replace or block N2; short10 was continued under the weak-auxiliary rule.

## Architecture and training

- Macro Actor remains duration-aware SMDP GAE with lambda 0.95; the formal main Critic remains N2.
- Long Critic is an independent raw-observation 400→128→128→1 SiLU network with no shared encoder or parameters.
- Separate Adam: lr 0.001, eps 1e-5, max gradient norm 0.5; 50 updates after each episode.
- Loss: MSE against detached, same-episode realized_return_to_go computed with configured gamma and actual_duration; terminal bootstrap is zero.
- No target normalization, PopArt, EMA, Huber, replay, or multi-episode mixing.
- Static validation passed: import/compile, 400D finite scalar output, target detach, optimizer separation, checkpoint metadata, and synthetic batch RMSE decreased from 1.42323730 to 1.00675433.
- Checkpoint v33 contains N2 state, Long state, Long optimizer state, long_value_enabled=true, target=realized_return_to_go.
- Fresh initialization seed 27005; seeds 20001–20005; resume=false; run: runs/formal_macro_dual_value_archv10_ckptv33_fresh5.
- Trace: C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\formal_macro_dual_value_archv10_ckptv33_fresh5\macro_transition_trace.npz; 512 transitions; N2 target and realized-return recomputation errors are zero.

## Episode metrics

| Episode | N2 pre EV | N2 post EV | N2 post corr. | N2 post std ratio | Long pre EV | Long post EV | Long pre corr. | Long post corr. | Long post std ratio | Long RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | 0.0001 | 0.5141 | 0.7176 | 0.6891 | -0.0000 | 0.1456 | -0.3688 | 0.8299 | 0.0929 | 193.975 |
| E2 | -0.0219 | 0.1625 | 0.5224 | 0.1902 | -0.0067 | 0.0002 | -0.6780 | 0.0126 | 0.0105 | 3010.313 |
| E3 | 0.1778 | 0.6206 | 0.7888 | 0.7486 | -0.1279 | 0.0268 | -0.3123 | 0.3677 | 0.0384 | 301.695 |
| E4 | -0.1844 | 0.4746 | 0.6939 | 0.6108 | 0.0010 | 0.0376 | 0.0607 | 0.4917 | 0.0399 | 1085.467 |
| E5 | 0.0944 | 0.5903 | 0.7705 | 0.7118 | -0.0059 | -0.0008 | 0.0010 | -0.0130 | 0.0185 | 1105.253 |

N2 E2–E5 median post target EV = 0.532439, above 0.3.
Long E2–E5 median post EV = 0.013452 and median post correlation = 0.190109; thresholds are EV >0.15 or correlation >0.30.
- Macro Actor stable: True; Micro stable: True; no Cache/Duration/UAV action collapse: True; Exact-Q executions: 0.
- Decision: Long auxiliary effect is weak on fresh5, but formal training path is usable with N2. The requested short10 continuation completed.

JSON contains critic snapshots 0/10/25/50 and per-episode Actor, Micro, return, action, and checkpoint diagnostics.

## Report files
- JSON: C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\formal_macro_dual_value_fresh5.json
- Markdown: C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\formal_macro_dual_value_fresh5.md
