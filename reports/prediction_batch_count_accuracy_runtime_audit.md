# Prediction batch count accuracy/runtime audit

## Executive conclusion

本轮是 prediction/configuration surrogate 的精度—速度审计，不是 PPO 训练。K=100 保持为唯一 exact/reference 路径；没有修改正式配置、PPO、reward、Matching/KKT、Cache/UAV candidate/search 逻辑或 worker 数。

origin 0 上，K=75/50/25 都显著减少了 surrogate wall time，并且在同一组 K=100 实际访问过的 candidate states 上保持了 Cache/UAV top-1、top-3、top-5 排名一致。但 reduced-K 的 surrogate objective 数值误差并不稳定；在跨 origin 的 actual exact environment 验证中，K=50 和 K=25 在 late train origin 239 改变了最终 Cache state，并造成明显的实际 `G(t)` 差异。

因此本轮不把任何 reduced-K 写入正式默认配置，也不建议现在实现 accelerated mode。正式 PPO episode 仍未 ready。

## Git and scope

- branch: `main`
- HEAD: `b7e90d4015bbeac682ce7b55a4a29d71a7e174d7`
- working tree 在本轮开始时已有用户/历史任务修改：`src/decision/task_allocation.py`、`src/ppo_simple/trainer.py` 以及既有 reports 和 `打包并上传.bat`；本轮没有修改或删除它们。
- 当前正式 prediction worker 数保持 requested/effective `8/8`。
- 没有 PPO update、完整 episode、多 episode 或 convergence test。

## Reference and subset protocol

当前正式 prediction context 是 `20 scenarios × 5 horizons = 100` 个 batch。原始顺序是 scenario 外层、horizon 内层；K=100 使用完整 `[0..99]`，没有覆盖或替换。

K<100 使用固定、按 horizon 分层的 deterministic subset：在正式 scenario 顺序中等距选 scenario，并对每个选中的 scenario 保留全部 5 个 horizon。这样避免把 flattened index 每隔 4 个取一次而意外只保留同一个 horizon。具体选择为：

| K | selected scenario indices | flattened batch indices |
|---:|---|---|
| 100 | `0..19` | `0..99` |
| 75 | `0,1,2,4,5,6,8,9,10,11,13,14,15,17,19` | `0..14, 20..34, 40..54, 60..74, 80..89, 95..99` |
| 50 | `0,2,4,6,8,10,12,14,16,19` | `0..4, 10..14, 20..24, 30..34, 40..44, 50..54, 60..64, 70..74, 80..84, 95..99` |
| 25 | `0,4,9,14,19` | `0..4, 20..24, 45..49, 70..74, 95..99` |

这些 index 由固定 index rule 生成，不使用结果、未来 actual reward 或随机重排。为保持 K=100 的现有 aggregation scale，K<100 的选中 batch 加权折扣和临时乘以 `100/K`，再沿用当前 scenario normalization；否则只减少 batch 数会把目标的量纲人为缩小，而不是测试 prediction sample 数量的影响。该参数化只存在于本轮 harness，已经恢复；正式 K=100 代码没有变化。

## Tested states and actions

使用同一个未更新的 PPO policy、`seed=4401` 的正常 Bernoulli sampling，为每个 origin 采样一次 action，并在该 origin 的 K=100/75/50/25 比较中复用完全相同 action、initial Cache/UAV state、actual tasks、actual vehicle positions 和 scenario slice。

- early origin `0`: action `[0,0,1,0,0,0,0,0,0,0,1]`，Cache node `43` 与 `UAV_1` request-on；四个 K 全部运行。
- middle origin `120`: action `[0,1,1,1,1,0,1,1,0,0,0]`，多个 Cache request-on、UAV request-off；K=100、50、25 运行。
- late origin `239`: action `[0,1,1,0,1,0,0,1,0,0,0]`，多个 Cache request-on、UAV request-off；K=100、50、25 运行。

这是 independent-origin fixed-state comparison：每个 selected origin 的 K runs 使用相同 initial state；没有把某个 reduced-K 的 state transition 带入另一个 K。K=100 在三个 origin 都作为 exact actual reference。

## Candidate-level surrogate accuracy — origin 0

K=100 optimizer 实际访问的 candidate/evaluator state 集合被完整去重后重放：17 个 Cache states、77 个 UAV states。K=75/50/25 对完全相同的 state 集合重新计算 `J_K = predicted_service_utility_K - configuration_cost`；没有用各自 reduced-K 搜索路径替代 reference candidate set。

Reference candidate set 中，Cache 有 1 个 positive-improvement candidate、15 个 reference-relative rejected/non-improving states；UAV 有 38 个 positive-improvement 和 38 个 rejected/non-improving states。另记录了 reference best 与 second-best 的 close competing pair。

### Cache candidates

| K | mean abs error | median abs error | max abs error | mean relative error | max relative error | top-1 | top-3 | top-5 | Spearman |
|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| 100 | 0 | 0 | 0 | 0 | 0 | YES | 1.000 | 1.000 | 1.000000 |
| 75 | 1.092736 | 1.549021 | 1.604125 | 0.070832 | 0.117183 | YES | 1.000 | 1.000 | 0.995098 |
| 50 | 2.284411 | 3.216225 | 3.363967 | 0.148177 | 0.245973 | YES | 1.000 | 1.000 | 0.987745 |
| 25 | 0.452849 | 0.606246 | 0.847592 | 0.030383 | 0.061341 | YES | 1.000 | 1.000 | 0.987745 |

### UAV candidates

| K | mean abs error | median abs error | max abs error | mean relative error | max relative error | top-1 | top-3 | top-5 | Spearman |
|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| 100 | 0 | 0 | 0 | 0 | 0 | YES | 1.000 | 1.000 | 1.000000 |
| 75 | 0.713029 | 0.713041 | 0.713595 | 0.028280 | 0.028311 | YES | 1.000 | 1.000 | 0.999737 |
| 50 | 1.444095 | 1.444084 | 1.446042 | 0.057276 | 0.057396 | YES | 1.000 | 1.000 | 0.998317 |
| 25 | 0.515910 | 0.515824 | 0.517203 | 0.020462 | 0.020514 | YES | 1.000 | 1.000 | 0.999185 |

The non-monotonic error is a property of the fixed subset composition, not evidence that fewer batches are always more accurate. Ranking agreement alone is insufficient because the objective magnitude shifts by up to 24.6% for the Cache candidate set at K=50.

## Optimizer decision agreement and actual exact environment

All actual-slot Matching/KKT computations below used the unchanged exact path. `Q` is actual task utility, `G`/reward is actual system utility, and cache/movement costs are the exact configuration costs.

| origin | K | cache final/decision agreement vs K=100 | UAV final/decision agreement | overall config | Q | G/reward | relative G error |
|---:|---:|---|---|---|---:|---:|---:|
| 0 | 75 | YES | YES | YES | 10.0216672861 | 10.0159472861 | 0 |
| 0 | 50 | YES | YES | YES | 10.0216672861 | 10.0159472861 | 0 |
| 0 | 25 | YES | YES | YES | 10.0216672861 | 10.0159472861 | 0 |
| 120 | 50 | YES | YES | YES | 10.9409110083 | 7.2940154532 | 0 |
| 120 | 25 | YES | YES | YES | 10.9409110083 | 7.2940154532 | 0 |
| 239 | 50 | **NO** | YES | **NO** | 5.8240828485 | 3.8027206871 | 0.137528 |
| 239 | 25 | **NO** | YES | **NO** | 5.1475983555 | 3.9477356924 | 0.180907 |

K=100 actual references were:

| origin | Q100 | G100/reward100 | Cache cost | movement cost | actual Cache update |
|---:|---:|---:|---:|---:|---|
| 0 | 10.0216672861 | 10.0159472861 | 0 | 0.005720 | all zero; UAV_1 moved |
| 120 | 10.9409110083 | 7.2940154532 | 3.646896 | 0 | nodes 43/75/87/58/RSU_1/UAV_0 state changed as requested/feasible |
| 239 | 5.7426931590 | 3.3429678328 | 2.399725 | 0 | nodes 43/58/87/UAV_0 candidate diagnostics; final update on 43 and 87 |

At origin 239, K=50 changed the final Cache update pattern and produced `delta_Q=+0.081390`, `delta_G=+0.459753`, `delta_reward=+0.459753`, `delta_delay_sum=+8.296749 s`, `delta_vehicle_energy_sum=-1.867733 J`, and `delta_cache_cost=-0.378363`. K=25 produced `delta_Q=-0.595095`, `delta_G=+0.604768`, `delta_reward=+0.604768`, `delta_delay_sum=+26.776253 s`, `delta_vehicle_energy_sum=+0.540350 J`, and `delta_cache_cost=-1.199863`.

The origin-239 result is the key failure for a formal default: a lower surrogate cost did not imply a better or equivalent actual decision; the exact actual environment changed the Cache state, Q, delay, energy, and reward.

K=75 was fully evaluated at origin 0, where its final decision and actual G matched K=100, but it was not retained for the middle/late second-stage runs. Its middle/late actual G and decision agreement are therefore **NOT VERIFIED**, not assumed from the origin-0 result.

## Runtime

All runs used the same 8-worker parallel prediction runtime. Matching search and KKT were not approximated. Runtime differences come from the number of selected prediction batches; no candidate generator or worker count was reduced.

### Origin 0

| K | prediction Matching calls | candidate evaluations | surrogate seconds | configuration seconds | worker wait seconds | execute_slot seconds | speedup vs K=100 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 9,200 | 3,727,699 | 176.249 | 176.358 | 173.604 | 176.438 | 1.000x |
| 75 | 6,900 | 2,720,379 | 128.636 | 128.720 | 126.277 | 128.797 | 1.370x |
| 50 | 4,600 | 1,665,710 | 72.128 | 72.208 | 69.893 | 72.289 | 2.441x |
| 25 | 2,300 | 881,069 | 37.557 | 37.630 | 35.438 | 37.707 | 4.679x |

### Middle/late second-stage runs

| origin | K | prediction Matching calls | candidate evaluations | execute_slot seconds | speedup vs same-origin K=100 |
|---:|---:|---:|---:|---:|---:|
| 120 | 100 | 17,700 | 6,511,907 | 235.021 | 1.000x |
| 120 | 50 | 8,850 | 3,368,683 | 139.440 | 1.685x |
| 120 | 25 | 4,500 | 1,410,440 | 64.598 | 3.638x |
| 239 | 100 | 6,200 | 2,310,377 | 85.194 | 1.000x |
| 239 | 50 | 4,150 | 1,565,074 | 61.238 | 1.391x |
| 239 | 25 | 2,075 | 887,483 | 39.181 | 2.174x |

The speedup is substantial but not linear: configuration path length, candidate states, process-pool startup, worker wait, and Matching workload vary by origin and by reduced-K optimizer path.

## Status assessment

| Item | Status | Finding |
|---|---|---|
| K=100 exact/reference preserved | **CONFIRMED** | Formal source/default remains K=100; all comparisons used it as the baseline. |
| deterministic subset selection | **CONFIRMED** | Formal ordering, fixed indices, no result-based selection, no future reward selection. |
| K<100 candidate ranking on common K=100 set | **CONFIRMED** | Origin 0 fixed set: top-1/top-3/top-5 agreement for all reduced K. |
| K<100 objective numerical fidelity | **FAILED** | Errors are nonzero and can be material; Cache max relative error reached 24.6% at K=50. |
| K50/K25 decision equivalence across selected origins | **FAILED** | Both changed the late-origin Cache decision and actual G. |
| K75 decision equivalence across selected origins | **NOT VERIFIED** | Only origin 0 was run for K75 in the staged protocol. |
| actual exact Q/G/reward path | **CONFIRMED** | K<100 only changes configuration surrogate evaluation; actual Matching/KKT/reward remained exact. |

## Final values

```ini
K100_REFERENCE_PRESERVED = YES
K75_EVALUATED = YES
K50_EVALUATED = YES
K25_EVALUATED = YES

K75_TOP1_AGREEMENT = YES (Cache/UAV common K=100 candidate set, origin 0)
K50_TOP1_AGREEMENT = YES (Cache/UAV common K=100 candidate set, origin 0)
K25_TOP1_AGREEMENT = YES (Cache/UAV common K=100 candidate set, origin 0)

K75_G_ERROR = 0.0 (origin 0 only; middle/late NOT VERIFIED)
K50_G_ERROR = 0.137528 relative max over tested origins
K25_G_ERROR = 0.180907 relative max over tested origins

K75_SPEEDUP = 1.369890x (origin 0)
K50_SPEEDUP = 2.440751x (origin 0; 1.685467x origin 120; 1.391185x origin 239)
K25_SPEEDUP = 4.679241x (origin 0; 3.638196x origin 120; 2.174354x origin 239)

RECOMMENDED_K = 100
ALGORITHMIC_ACCELERATION_WORTHWHILE = NO
READY_TO_IMPLEMENT_ACCELERATED_MODE = NO
READY_FOR_FORMAL_PPO_EPISODE = NO
```

`ALGORITHMIC_ACCELERATION_WORTHWHILE = NO` means no reduced-K mode has yet demonstrated sufficiently stable actual decisions for the formal default, despite clear runtime savings. A future accelerated mode would need an explicit train-only selection/validation protocol and must retain an independently selectable exact K=100 reference mode.

临时 batch parameterization、capture/ranking harness、subset JSON 和 Python cache 已删除；没有 commit、push、merge 或 rebase。
