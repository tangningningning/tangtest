# Formal Dual-Value Macro Critic Short10

## Result

DUAL_VALUE_SHORT10_GATE = PASS
FORMAL_TRAINING_PATH = USABLE
READY_FOR_FORMAL_LONG_TRAINING = YES

Short10 completed from fresh initialization with seed 27005 and episode seeds 20001–20010. N2, Macro Actor, Micro, actions, Exact-Q, and the Long Critic median correlation checks passed.

## Configuration

- Formal main Macro Critic: TWO_STEP_SMDP_TD (N2); Macro Actor advantage remains DURATION_AWARE_SMDP_GAE with lambda 0.95.
- Architecture v10/checkpoint v33. Long Critic remains independent and is trained on detached same-episode realized returns using actual_duration.
- Fresh initialization, resume=false; run: runs/formal_macro_dual_value_archv10_ckptv33_short10.
- Production trace transitions: 1024; N2 target reconstruction error 0.0; realized-return reconstruction error 0.0.
- Latest checkpoint contains explicit N2 Critic, Long Critic, and Long optimizer states; Long Critic enabled with realized_return_to_go target.

## Episode metrics

| Episode | N2 pre EV | N2 post EV | N2 post corr. | N2 post std ratio | Long pre EV | Long post EV | Long pre corr. | Long post corr. | Long post std ratio | Long RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | 0.0001 | 0.5141 | 0.7176 | 0.6891 | -0.0000 | 0.1456 | -0.3688 | 0.8299 | 0.0929 | 193.975 |
| E2 | -0.0219 | 0.1625 | 0.5224 | 0.1902 | -0.0067 | 0.0002 | -0.6780 | 0.0126 | 0.0105 | 3010.313 |
| E3 | 0.1778 | 0.6206 | 0.7888 | 0.7486 | -0.1279 | 0.0268 | -0.3123 | 0.3677 | 0.0384 | 301.695 |
| E4 | -0.1844 | 0.4746 | 0.6939 | 0.6108 | 0.0010 | 0.0376 | 0.0607 | 0.4917 | 0.0399 | 1085.467 |
| E5 | 0.0944 | 0.5903 | 0.7705 | 0.7118 | -0.0059 | -0.0008 | 0.0010 | -0.0130 | 0.0185 | 1105.253 |
| E6 | 0.8305 | 0.9511 | 0.9753 | 0.9741 | 0.0026 | 0.3458 | 0.0562 | 0.8850 | 0.2236 | 277.477 |
| E7 | 0.0866 | 0.4972 | 0.7064 | 0.6636 | 0.0092 | 0.0288 | 0.3701 | 0.2682 | 0.0606 | 1495.305 |
| E8 | 0.2472 | 0.7165 | 0.8466 | 0.8285 | -0.2094 | -0.0681 | -0.4276 | -0.2163 | 0.1226 | 951.264 |
| E9 | -0.0267 | 0.4554 | 0.6748 | 0.6679 | -0.0703 | 0.4667 | -0.6247 | 0.7567 | 0.4313 | 231.482 |
| E10 | 0.4422 | 0.7703 | 0.8777 | 0.8722 | 0.0056 | 0.7693 | 0.0749 | 0.9145 | 0.6557 | 169.287 |

N2 E2–E10 median post target EV = 0.590257 (threshold >0.3).
Long E1–E10 median post EV = 0.033218; median post correlation = 0.429676 (threshold >0.2).
- Macro Actor stable: True; Micro stable: True; no action collapse: True; Exact-Q executions: 0.
- Macro KL max 4.09792e-05, clip fraction max 0; Micro KL max 0.00747769, clip fraction max 0.0841667; no early stops.
- All duration and action patterns remained NORMAL; realized returns and traces were finite.
- Decision: N2 remains the formal main Critic and the run is READY_FOR_FORMAL_LONG_TRAINING=YES. The Long Critic stays an isolated auxiliary.

JSON contains full per-episode N2/Long fit snapshots, target freeze checks, Actor/Micro metrics, and action counts.

## Report files
- JSON: C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\formal_macro_dual_value_short10.json
- Markdown: C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\formal_macro_dual_value_short10.md
