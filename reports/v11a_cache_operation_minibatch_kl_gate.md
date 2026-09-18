# V11A Cache Operation Minibatch KL Gate

本 gate 只定位 Cache PPO epoch 1 内的 KL 形成过程，不执行正式训练。使用固定 `episode_20001`、300 slots、seed `4401`、V11-A actor learning rate `6e-4` 和 centralized `Q_cache`。由于上一阶段没有持久化 centralized-Q 权重，本次在同一 realization/seed 上重构 bounded Q readiness；完成 `4 × 10 = 40` 个 Q steps 后冻结 Q，再执行唯一的 Cache PPO epoch 1（10 个 minibatch）。没有第二个独立 rollout、epoch 2–4、正式 300 epoch 或 checkpoint 写入。

诊断通过临时进程级 hook 在生产 Cache actor optimizer step 后重新评价整个 frozen rollout。V11-A 使用的 Cache branch estimator 保持为：

```text
CACHE_BRANCH_APPROX_KL_k
  = mean_t sum_m(old_selected_logprob_m - new_selected_logprob_m)
```

Operation 和 candidate 分量使用相同的 sampled-action estimator；每个 `NODE_APPROX_KL_m,k` 是对应节点在 300 个 rollout transition 上的均值。Actor step 后紧接的 disjoint global-critic step 不改变 Cache actor 参数。

## Final fields

```text
STAGE = V11A_CACHE_OPERATION_MINIBATCH_KL_GATE
TRAINING_EXECUTED = NO
FORMAL_TRAINING_EXECUTED = NO

CACHE_KL_AFTER_MB = [
  0.015018596313893795, 0.067522332072258, 0.14567503333091736,
  0.23758448660373688, 0.32918715476989746, 0.41956183314323425,
  0.5049706101417542, 0.5890456438064575, 0.6677715182304382,
  0.740821123123169
]
FIRST_MB_KL_GT_0_1 = 3

KL_BEFORE_CROSSING = 0.067522332072258
KL_AFTER_CROSSING = 0.14567503333091736

OPERATION_KL_AFTER_MB = [
  0.0163434948772192, 0.06918787956237793, 0.14727477729320526,
  0.23860697448253632, 0.3296760618686676, 0.4196077287197113,
  0.5048577785491943, 0.5888111591339111, 0.6674952507019043,
  0.7404242157936096
]
CANDIDATE_KL_AFTER_MB = [
  -0.0013249035691842437, -0.0016655405052006245,
  -0.0015997248701751232, -0.0010225033620372415,
  -0.000488844292704016, -0.00004587431612890214,
  0.00011279046884737909, 0.00023444474209100008,
  0.0002761445357464254, 0.000396988180000335
]

OPERATION_DOMINANCE_REPRODUCED = YES

CACHE_GRAD_NORM_AFTER_MB = [
  {MB=1, before=0.14615647679931118, after=0.13241146665134282},
  {MB=2, before=0.1443558184980823, after=0.10416538237522295},
  {MB=3, before=0.12452184425557271, after=0.08568720034426253},
  {MB=4, before=0.1182213988543153, after=0.1182213988543153},
  {MB=5, before=0.06768627492832476, after=0.06768627492832476},
  {MB=6, before=0.04941634071744147, after=0.04941634071744147},
  {MB=7, before=0.045898042011917134, after=0.045898042011917134},
  {MB=8, before=0.033656673692995465, after=0.033656673692995465},
  {MB=9, before=0.04059756371270462, after=0.04059756371270462},
  {MB=10, before=0.033047471121412896, after=0.033047471121412896}
]
VALID_NODE_TIME_AFTER_MB = [270, 270, 270, 270, 270, 270, 270, 270, 270, 270]

CACHE_UPDATE_EXCESS_MODE = WITHIN_EPOCH_ACCUMULATION
SPIKE_MB = NONE
SPIKE_DELTA_KL = NONE
NEXT_CONTROL_DIRECTION = MINIBATCH_LEVEL_KL_EARLY_STOP

FORMAL_300_EPOCH_RECOMMENDED = WAIT

SYSTEM_MODEL_CHANGED = NO
GLOBAL_REWARD_CHANGED = NO
Q_CACHE_CHANGED = NO
UAV_CHANGED = NO

GIT_DIFF_CHECK = PASS: diagnostic-only instance hooks; production algorithm/config unchanged
TEMP_FILES_REMOVED = YES
PUSHED = NO
```

## KL trajectory

| minibatch | Cache KL | Operation KL | Candidate KL | Operation / total |
|---:|---:|---:|---:|---:|
| 1 | 0.015019 | 0.016343 | -0.001325 | 1.088217 |
| 2 | 0.067522 | 0.069188 | -0.001666 | 1.024667 |
| 3 | 0.145675 | 0.147275 | -0.001600 | 1.010982 |
| 4 | 0.237584 | 0.238607 | -0.001023 | 1.004304 |
| 5 | 0.329187 | 0.329676 | -0.000489 | 1.001485 |
| 6 | 0.419562 | 0.419608 | -0.000046 | 1.000109 |
| 7 | 0.504971 | 0.504858 | 0.000113 | 0.999777 |
| 8 | 0.589046 | 0.588811 | 0.000234 | 0.999602 |
| 9 | 0.667772 | 0.667495 | 0.000276 | 0.999586 |
| 10 | 0.740821 | 0.740424 | 0.000397 | 0.999464 |

The first crossing is at MB3: KL rises from `0.0675223` after MB2 to `0.1456750` after MB3. All ten increments are positive. The largest increment is into MB4, `0.09190945`; the median absolute value of the other increments is `0.07872587`, a ratio of only `1.1675`, so the trajectory is not a minibatch spike.

The operation component remains dominant from the first step. Its absolute contribution is at least 99.9% once the total KL is materially nonzero; the final split is operation `0.74042422` versus candidate `0.00039699`. The first-step ratio exceeds one because the candidate component is slightly negative, not because candidate drift dominates.

## Node KL by minibatch

The node order is `SV_0..SV_4`, `RSU_0`, `RSU_1`, `UAV_CACHE_0`, `UAV_CACHE_1`; the corresponding runtime IDs are `46`, `10`, `0`, `23`, `59`, `RSU_0`, `RSU_1`, `UAV_0`, `UAV_1`.

| node | MB1 | MB2 | MB3 | MB4 | MB5 | MB6 | MB7 | MB8 | MB9 | MB10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SV_0 | 0.002784 | 0.016863 | 0.033417 | 0.049105 | 0.064810 | 0.080021 | 0.094729 | 0.108147 | 0.120016 | 0.130559 |
| SV_1 | -0.001131 | 0.004196 | 0.012875 | 0.021384 | 0.028690 | 0.036850 | 0.046400 | 0.055413 | 0.065475 | 0.074699 |
| SV_2 | 0.002802 | 0.009655 | 0.020141 | 0.034052 | 0.050402 | 0.066511 | 0.081727 | 0.095875 | 0.108853 | 0.120351 |
| SV_3 | -0.001880 | -0.001713 | 0.001493 | 0.008592 | 0.018491 | 0.029800 | 0.040414 | 0.051139 | 0.061002 | 0.069720 |
| SV_4 | 0.002238 | 0.005011 | 0.012915 | 0.026855 | 0.040593 | 0.053840 | 0.062947 | 0.073094 | 0.082162 | 0.091521 |
| RSU_0 | 0.002244 | 0.006121 | 0.012461 | 0.019845 | 0.026639 | 0.033187 | 0.038262 | 0.042643 | 0.045202 | 0.046700 |
| RSU_1 | 0.003488 | 0.009089 | 0.015372 | 0.023959 | 0.030056 | 0.034217 | 0.039391 | 0.045143 | 0.050952 | 0.057530 |
| UAV_CACHE_0 | -0.002482 | -0.001980 | 0.000155 | 0.004460 | 0.009741 | 0.015761 | 0.021529 | 0.029215 | 0.037443 | 0.045889 |
| UAV_CACHE_1 | 0.006956 | 0.020280 | 0.036844 | 0.049331 | 0.059765 | 0.069374 | 0.079572 | 0.088377 | 0.096666 | 0.103852 |

## Step diagnostics

Cache actor loss is the production node-specific Cache surrogate loss; entropy coefficient is zero in this configuration. Cache gradient norms are the `cache` actor parameter group (operation head plus candidate scorer) measured before and after the production global actor clipping call. `mean_abs_normalized_A_cf` and valid counts are computed on the current 30-transition minibatch; full-rollout valid count is 2700.

| MB | Cache actor loss | grad before | grad after | valid node-time | mean abs norm. A_cf | op-head cumulative Δ | op-head step Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.006235 | 0.146156 | 0.132411 | 270 | 0.829430 | 0.040884 | 0.040884 |
| 2 | -0.093669 | 0.144356 | 0.104165 | 270 | 0.828681 | 0.072066 | 0.034525 |
| 3 | -0.005847 | 0.124522 | 0.085687 | 270 | 0.788347 | 0.099519 | 0.030577 |
| 4 | -0.125510 | 0.118221 | 0.118221 | 270 | 0.825520 | 0.124713 | 0.028793 |
| 5 | -0.140500 | 0.067686 | 0.067686 | 270 | 0.781047 | 0.147681 | 0.026398 |
| 6 | -0.134603 | 0.049416 | 0.049416 | 270 | 0.845816 | 0.168400 | 0.023699 |
| 7 | -0.171119 | 0.045898 | 0.045898 | 270 | 0.789318 | 0.187170 | 0.021800 |
| 8 | -0.192220 | 0.033657 | 0.033657 | 270 | 0.829690 | 0.204317 | 0.019663 |
| 9 | -0.126808 | 0.040598 | 0.040598 | 270 | 0.805146 | 0.219784 | 0.018321 |
| 10 | -0.095260 | 0.033047 | 0.033047 | 270 | 0.821821 | 0.234155 | 0.017762 |

## Gate decision

This is `WITHIN_EPOCH_ACCUMULATION`: the threshold is crossed at minibatch 3 and the Cache KL then grows across the remainder of epoch 1 without a distinct spike. The next control direction is `MINIBATCH_LEVEL_KL_EARLY_STOP`; it is only a recommendation for the next stage and is not implemented here. No learning rate, target KL, clipping, reward, Q, UAV, node-specific KL, bonus, or entropy setting was changed.

The machine-readable per-minibatch node/component/step diagnostics are in `reports/v11a_cache_operation_minibatch_kl_gate.json`.
