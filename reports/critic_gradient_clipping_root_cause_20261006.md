# 5B Critic Gradient Clipping Root-Cause Check — Fixed-Seed Recheck (2026-10-06)

上一轮完整固定 episode rollout 已返回，但临时分析器错误访问 EpisodeCollection.macro_buffer 导致运行时梯度分解未完成；本报告为修复为 collection.macro/collection.micro 后的固定 seed 复验。

## Execution and integrity

- Formal training/update: no. Formal optimizer steps: 0. One frozen `episode_20001` rollout completed using the existing 15-episode latest checkpoint, Unified Cache + Cardinality Correction, episode seed 20001, action sampling seed 27005.
- Correct collection API was checked before rollout and used after return: `collection.micro` (300 transitions), `collection.macro` (93 frames), `collection.summary`.
- Checkpoint counters: 15 completed episodes before and after diagnostic. Formal policy, Macro Critic Adam, Micro Adam, counters, and saved last1 buffer hashes all unchanged: **True**.
- Shadow Adam step calls: one clipped and one unclipped step per Critic, each on a distinct deep-copied policy and optimizer state. No formal optimizer was stepped.

## Inherited 15-episode evidence

Macro preclip norm mean 234.8754, clip scale mean 0.00231526, EV mean 0.9897894, historical final/current norm ratio 1, gradient/current raw-MSE correlation 0.6201117.
Micro preclip norm mean 473.3942, clip scale mean 0.001415507, EV mean 0.9856647, gradient/value-MSE correlation 0.8639414.
Historical fit first5→last5: Macro RMSE 59.09047→58.80835, value loss 3643.24→3559.956; Micro RMSE 66.73435→36.49898, value loss 19795.09→7589.626.

## Micro Critic

Full 300-slot GAE batch: target mean/std 229.9626/581.9333; prediction mean/std 495.8291/427.2468; residual mean/std/RMS 265.8665/312.0414/409.945; raw MSE 168054.9; EV 0.7124733.
For `L=0.01*mean(residual²)`: output-gradient L2/RMS/max-abs 0.4733637/0.02732967/0.07341926; parameter-gradient global norm 2109.472; Jacobian amplification `||∇θL||/||dL/dV||`=4456.345 (per-sqrt-sample=257.2872).
Per-layer gradient norms: `{"first_linear": 1057.5904371582376, "hidden_linear_1": 738.1184415001609, "final_value_head": 1669.298003459978}`.
Using `scale=target_std=581.9333`, diagnostic normalized gradient norm 0.006229129, reduction factor 338646.4. This does not change the formal loss.
For the Shadow Adam comparison, selected median-norm real official-size partition 2/10; its gradient norm was 1996.173. Partition gradient norms: `[472.338752, 661.8854874, 1996.1727792, 2168.8698686, 4360.4287835, 6813.2186995, 5912.5421985, 1385.0776366, 369.2599443, 215.5631532]`.

## Macro Critic current / last1 / PCGrad

Current N2 target: target mean/std 682.9823/889.6033; prediction mean/std 789.4213/853.8783; residual mean/std/RMS 106.439/181.7505/210.6241; raw MSE 44362.49; EV 0.9582594.
Current `dL/dV` L2/RMS/max-abs 0.4368139/0.0452955/0.1381197; parameter norm 2265.254; Jacobian amplification 5185.857 (per-sqrt-sample=537.7483). Per-layer norms: `{"first_linear": 1994.4729102068432, "hidden_linear_1": 686.3468118630984, "final_value_head": 826.0650690486966}`.
Last1 checkpoint episode 15: target std 418.0184, residual RMS 147.6385, raw MSE 21797.12, gradient norm 1636.743.
PCGrad chain: current norm 2265.254; last1 raw norm 1636.743; norm-matched last1 norm 2265.254; mixed norm 4043.475; final norm 2265.255; final/current ratio 1; projection active=False; raw cosine 0.5931096; post-projection cosine 0.5931095.
Using current target std 889.6033, scaled diagnostic gradient norm 0.00286236; reduction factor 791394.

## Clip direction and Shadow Adam

Tiny negative `1-cosine` values caused by floating-point rounding are shown as 0 in this table; JSON preserves raw cosine values.

| Gradient witness | Preclip norm | Postclip norm | Cosine | 1-cosine |
|---|---:|---:|---:|---:|
| Micro full rollout batch | 2109.472 | 0.5 | 1 | 0 |
| Micro shadow real minibatch | 1996.173 | 0.5 | 1 | 0 |
| Macro current | 2265.254 | 0.5 | 1 | 3.996803e-15 |
| Macro final combined | 2265.255 | 0.5 | 1 | 5.995204e-15 |

| Shadow Adam | Clipped delta L2 | Unclipped delta L2 | Relative deltas (clip / no clip) | Delta ratio clip/no-clip | Update cosine | State copied exactly |
|---|---:|---:|---:|---:|---:|---|
| Micro | 0.2996405 | 1.060013 | 0.003575968 / 0.0126504 | 0.2826763 | 0.221843 | step 600–600; moments match |
| Macro | 0.05720296 | 0.5782038 | 0.0009582307 / 0.009685734 | 0.09893218 | 0.2836957 | step 750–750; moments match |

Shadow optimizers loaded deep-copied formal `state_dict()` values. For every Critic parameter, `step`, `exp_avg`, and `exp_avg_sq` matched the formal optimizer before the shadow step.

## Decision

- Root-cause class: **MIXED_OR_INSUFFICIENT**.
- Macro/Micro fit health from current frozen batch plus inherited 15-episode trajectories: True/True.
- Root-cause rationale: target-scale normalization reduces the measured gradient norm by 338646x (Micro) and 791394x (Macro), while the prescribed parameter/output-gradient quotients are 4456 and 5186 (per-sqrt-sample 257 and 538). The quotients are large, but this check has no calibrated healthy reference range; the measurements do not establish Jacobian amplification as dominant over the target-scale effect. PCGrad final/current remains 1.0, so the evidence does not isolate one dominant cause.
- Maximum runtime clipping `1-cosine`: 5.995204e-15. Norm clipping preserved the measured gradient directions.
- Shadow delta ratios Macro/Micro: 0.09893218/0.2826763; update cosines Macro/Micro: 0.2836957/0.221843.
- Clipping materially changes the single-step Shadow Adam updates, but inherited 15-episode EV/RMSE remains healthy; the current evidence does not establish Critic fit failure or justify an immediate change.
- Explicit clipping bottleneck: False. Immediate Critic changes: no.
- First real Critic issue: 持续 global-norm clip saturation：历史 Macro mean=234.875, mean clip scale=0.002315; Micro mean=473.394, mean clip scale=0.001416. 本次没有发现非有限值/PCGrad final norm amplification/明确 fit 故障。
- 5B status: **通过，暂不修改 Critic**.
- Next step: 停止，等待确定性 Cache evaluation protocol 检查.

## Limit

One fixed realization and one stochastic action trajectory provide a diagnostic witness, not a cross-realization estimate. The target-standardized loss is a read-only scale counterfactual, not a proposed training change.
