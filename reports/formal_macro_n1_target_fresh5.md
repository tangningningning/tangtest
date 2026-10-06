# Formal Macro N1 Target Fresh5

## Decision

- **FORMAL_N1_TARGET_CAUSAL_GATE:** PARTIAL
- **MACRO_CRITIC_TARGET_ROOT_CAUSE:** PARTIAL
- **Next:** Do not start short10 yet; analyze whether N1 is only easier to fit or improves long-term value representation using the realized-return reference.

The N1 Critic learns its selected one-step target during the 50-step fit, and E2–E5 pre-update target EV is no longer severely negative. The critic-independent realized-return reference does not improve consistently, so the evidence supports only a partial target-estimator explanation.

## Modes and formal run

- Macro Actor advantage: `DURATION_AWARE_SMDP_GAE` (unchanged).
- Macro Critic target: `ONE_STEP_SMDP_TD`.
- Observation: 400D, unchanged; architecture v9; checkpoint v30.
- Fresh init seed: 27005; episode seeds: 20001–20005; resume: `false`.
- Run: `C:/Users/tangning/Desktop/code/adaptive_cache_mec/runs/formal_macro_n1_target_ckptv30_archv9_fresh5`.
- Replay history in current config: 0 episodes; Critic update: 50 steps per episode.

## Training target metrics

| Episode | Pre-update EV | Post-update EV | Pre corr. | Pre std ratio | Step-50 EV | Macro KL |
|---|---:|---:|---:|---:|---:|---:|
| E1 | -0.0006 | 0.3808 | -0.3096 | 0.0010 | 0.3808 | 4.45423e-05 |
| E2 | -0.0081 | 0.3495 | 0.0157 | 0.1073 | 0.3495 | 2.25531e-05 |
| E3 | -0.0586 | 0.2194 | -0.0310 | 0.2131 | 0.2194 | 1.2474e-05 |
| E4 | 0.0539 | 0.0289 | 0.4895 | 0.0585 | 0.0289 | 2.03633e-05 |
| E5 | -0.0041 | 0.3224 | 0.0231 | 0.0910 | 0.3224 | 1.31381e-05 |

Critic target summaries (`mean / std / min / max`):
- E1: 13.381 / 30.532 / -184.166 / 54.484; old value μ/σ 0.001/0.030; old next value μ/σ 0.001/0.030; TD error μ/σ 13.380/30.541.
- E2: -0.099 / 74.724 / -587.994 / 56.678; old value μ/σ 7.331/8.019; old next value μ/σ 7.512/7.647; TD error μ/σ -7.430/75.028.
- E3: 3.169 / 42.643 / -208.053 / 61.052; old value μ/σ 11.794/9.085; old next value μ/σ 12.170/7.699; TD error μ/σ -8.625/43.874.
- E4: -60.654 / 124.358 / -510.791 / 64.842; old value μ/σ 12.560/7.280; old next value μ/σ 12.345/7.374; TD error μ/σ -73.214/120.962.
- E5: -17.092 / 25.592 / -116.821 / 32.636; old value μ/σ -23.512/2.328; old next value μ/σ -23.287/3.222; TD error μ/σ 6.420/25.644.

Per-episode Critic snapshots (EV / correlation / prediction-to-target std ratio / MSE):
- E1: 0: -0.001/-0.310/0.001/1111.8; 10: 0.050/0.537/0.049/976.7; 25: 0.265/0.544/0.369/728.5; 50: 0.381/0.617/0.607/577.2
- E2: 0: -0.008/0.016/0.107/5684.3; 10: 0.049/0.223/0.194/5335.0; 25: 0.159/0.448/0.244/4696.3; 50: 0.350/0.601/0.493/3641.8
- E3: 0: -0.059/-0.031/0.213/1999.4; 10: 0.060/0.386/0.088/1709.6; 25: 0.189/0.446/0.344/1551.8; 50: 0.219/0.509/0.310/1448.4
- E4: 0: 0.054/0.489/0.059/19992.0; 10: 0.029/0.375/0.041/18622.4; 25: -0.006/-0.223/0.013/17981.7; 50: 0.029/0.373/0.041/16355.3
- E5: 0: -0.004/0.023/0.091/698.8; 10: 0.052/0.268/0.126/621.5; 25: 0.232/0.483/0.520/503.3; 50: 0.322/0.568/0.560/443.9

## Critic-independent realized-return reference

| Episode | Pre EV | Post EV | Pre corr. | Post corr. | Pre std ratio | Post std ratio |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 0.0001 | -0.0141 | 0.7090 | -0.2146 | 0.00005 | 0.03069 |
| E2 | 0.0266 | 0.0117 | 0.5992 | 0.1082 | 0.02263 | 0.10385 |
| E3 | 0.0168 | 0.0288 | 0.5164 | 0.6114 | 0.01655 | 0.02405 |
| E4 | -0.0055 | -0.0023 | -0.7525 | -0.4498 | 0.00363 | 0.00253 |
| E5 | -0.0045 | -0.0672 | -0.2485 | -0.5931 | 0.00881 | 0.05420 |

This reference is computed from the realized Macro reward sequence using the formal gamma and actual durations, with terminal bootstrap zero. It is diagnostic-only and is never used as a loss target. The historical GAE fresh5 did not save this reference metric, so direct GAE-vs-N1 reference comparison is `NOT_AVAILABLE`.

## Historical GAE comparison and gate

| Episode | N1 pre target EV | Historical GAE pre target EV | Δ EV |
|---|---:|---:|---:|
| E2 | -0.0081 | 0.1299 | -0.1381 |
| E3 | -0.0586 | -0.0852 | +0.0266 |
| E4 | 0.0539 | -0.1217 | +0.1755 |
| E5 | -0.0041 | -2.7384 | +2.7343 |

- N1 E2–E5 median pre-update target EV: `-0.0061`; historical GAE median: `-0.1034`.
- N1 E2–E5 maximum pre-update prediction/target std ratio: `0.2131`; no 3×/5× prediction scale explosion occurred.
- Return-correlation drops greater than 0.10 from pre-update to step 50: `2/4`; directions are mixed.
- The old fixed-return reference is unavailable. The historical run also used replay history 4, while the current proposed config had replay history 0 before this task and that state was preserved. This weakens strict historical causal attribution.

## Actor, Micro, actions, and environment counters

- Macro Actor stable: **YES**; all KL, entropy, and advantage values are finite.
- Micro stable: **YES**; all checked Micro metrics are finite.
- Cache counts: KEEP `255`, UPDATE `241`.
- Requested duration counts: `{"1": 166, "3": 153, "5": 177}`.
- UAV action counts: `{"UAV0": {"MOVE": 849, "STAY": 651}, "UAV1": {"MOVE": 785, "STAY": 715}}`.
- Environment executions `1500`, Matching calls `1500`, KKT calls `157534`, Exact-Q environment executions `0`.

## Static checks

- A–B: synthetic 3-transition N1 formula and actual duration pass; target values `[7.0, 6.75, 5.0]`.
- C–D: target detached; frozen Critic target tensor and SHA-256 equal before/after 50 steps.
- E: original duration-aware SMDP GAE source unchanged; synthetic actor advantages match exactly after the original float32 output cast (max error 0).
- F: `_update_micro` source is byte-identical to HEAD.
- Checkpoint: synthetic loader call rejects a v29 payload missing `macro_critic_target_mode` as legacy semantics.
- Config import/validation, 400D shape, finite synthetic outputs, and snapshot steps 0/10/25/50 pass.

## Checkpoint and artifact paths

- Latest checkpoint metadata: version `30`, architecture `9`, mode `one_step_smdp_td`; fresh initialization verified.
- Runtime used `C:\software\Anaconda_envs\envs\adaptive\python.exe` with PyTorch `2.0.0+cpu`; the requested `adaptive_cache_mec` interpreter had previously failed to import PyTorch with WinError 1455. `MKL_THREADING_LAYER=SEQUENTIAL` prevented the duplicate Intel OpenMP runtime error without changing training configuration.
- JSON report: `C:/Users/tangning/Desktop/code/adaptive_cache_mec/reports/formal_macro_n1_target_fresh5.json`.
- Markdown report: `C:/Users/tangning/Desktop/code/adaptive_cache_mec/reports/formal_macro_n1_target_fresh5.md`.
- Source files: `src/ppo_simple/trainer.py`, `src/ppo_simple/rollout.py`, `src/ppo_simple/reporting.py`.
- Config: `configs/proposed/system_config.toml`.


## Cleanup

- No temporary scripts, synthetic artifacts, or debug checkpoints were created.
- No `__pycache__` directories or `*.pyc` files were present under `src/`, `scripts/`, or `reports/`.
- The fresh5 checkpoints, metrics, and final reports are retained.
