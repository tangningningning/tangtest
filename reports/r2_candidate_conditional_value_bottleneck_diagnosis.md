# R2 candidate-conditioned value bottleneck diagnosis

日期：2026-09-18  
仓库：`C:\Users\tangning\Desktop\code\adaptive_cache_mec`  
checkpoint：`runs/ppo_recovery_r2_dev20_20260918/latest_checkpoint.pt`，`CHECKPOINT_EPOCH = 7`  
冻结 realization：`runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20001`  
案例：slot `20,50,80,110,140,170,200,230,260,290` × UAV `0,1`，`STATE_COUNT = 20`。

## 结论

`PRIMARY_DIAGNOSIS = MIXED`。

在 20 个 target-UAV state、共 1600 个合法 MOVE candidate 中：

- 19/20 state 的全部合法 MOVE candidate 都有 `DeltaQ_c <= 0`，即这些 state 的最优合法 MOVE 也不能击败 STAY。
- 仅 1/20 state（slot 200, UAV_0）存在 good candidate，且只有 2/80 个 candidate good；其 `BEST_DELTA_Q=+0.00296045`，但 good candidate 的 policy probability mass 只有 `0.02760945`，policy-expected `DeltaQ` 仍为 `-0.04114249`。
- 20/20 state 的 policy-expected MOVE 都不如 STAY；20/20 state 的 policy top-1 candidate 也不如 STAY，且 20/20 state 的 policy top-1 都不是真实 best candidate。

因此，`SYSTEM_OBJECTIVE_PREFERS_STAY` 是主导解释（19/20 state）；`CANDIDATE_POLICY_BOTTLENECK` 只得到局部支持（slot 200/UAV_0），不足以解释整个 20-state 的 STAY collapse。不能把本轮结果归类为“所有 MOVE 都只是 candidate policy 选差了”。

这仍然确认 `OPERATION_TARGET_BIAS = YES`，但本任务不评估 movement-cost implementation 是否有 bug。

## 方法与一致性

本轮只调用现有 R2 `_compute_r2_operation_credit`，在临时脚本中捕获该 helper 实际执行的 candidate policy logits、合法 candidate 枚举、`execute_slot` 结果以及 helper 当次 critic 输出；没有修改 trainer 或 production logging。

对每个 target-UAV state 保持 production Cache action、non-target UAV action、正式 `execute_slot`、Matching/KKT、正式 reward 和 movement cost，只执行 STAY 与全部合法 MOVE candidate。没有 candidate sampling、pruning、top-k shortcut、surrogate Q、action-id forcing 或固定位置替换。

对每个 MOVE candidate 使用：

```text
Q_c       = r_c + gamma * V(next_c)
Q_stay    = r_stay + gamma * V(next_stay)
DeltaQ_c  = Q_c - Q_stay
Q_move    = sum_c pi_c * Q_c
DeltaQexp = sum_c pi_c * DeltaQ_c
```

candidate policy 概率仅在从 torch float32 转为 float64 后做归一化，以消除约 `1e-7` 的概率质量 roundoff；这是数值修正，不是 policy 或参数修改。使用同一 double-precision candidate Q 检验 `sum(pi*Q_c)` 与 `Q_stay+sum(pi*DeltaQ_c)` 的 identity：

```text
EXPECTED_Q_IDENTITY_MAX_ABS_DIFF = 5.6843e-14
```

生产 helper 返回的 float32 `q_move-q_stay` 与 double-precision 重组值的最大 roundoff 为 `7.8497e-05`；这只记录为存储精度差异，不混入 identity 判定。

## Per-state candidate statistics

`GOOD_CANDIDATE_COUNT = count(DeltaQ_c > 0)`；`GOOD_POLICY_MASS` 为所有 good candidate 的当前 candidate policy 概率之和。Spearman 是 candidate probability 与 `DeltaQ_c` 的描述性 rank correlation。

| slot | UAV | legal | good | good frac | best DeltaQ | worst DeltaQ | median DeltaQ | policy expected DeltaQ | policy top1 DeltaQ | good policy mass | top1 is best | best-candidate probability | Spearman |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 20 | UAV_0 | 80 | 0 | 0.000 | -0.016399 | -0.098094 | -0.034747 | -0.041173 | -0.024572 | 0.000000 | NO | 0.013804 | -0.2031 |
| 20 | UAV_1 | 80 | 0 | 0.000 | -0.024241 | -0.070349 | -0.038974 | -0.041222 | -0.054371 | 0.000000 | NO | 0.011354 | -0.8684 |
| 50 | UAV_0 | 80 | 0 | 0.000 | -0.018292 | -0.094047 | -0.034386 | -0.041142 | -0.027399 | 0.000000 | NO | 0.011355 | -0.2822 |
| 50 | UAV_1 | 80 | 0 | 0.000 | -0.025365 | -0.065822 | -0.039322 | -0.041130 | -0.052815 | 0.000000 | NO | 0.011355 | -0.8666 |
| 80 | UAV_0 | 80 | 0 | 0.000 | -0.018530 | -0.093463 | -0.034384 | -0.041185 | -0.027683 | 0.000000 | NO | 0.011355 | -0.2890 |
| 80 | UAV_1 | 80 | 0 | 0.000 | -0.026028 | -0.063217 | -0.039003 | -0.041198 | -0.052903 | 0.000000 | NO | 0.011355 | -0.8741 |
| 110 | UAV_0 | 80 | 0 | 0.000 | -0.016391 | -0.098113 | -0.034697 | -0.041184 | -0.024390 | 0.000000 | NO | 0.013804 | -0.2004 |
| 110 | UAV_1 | 80 | 0 | 0.000 | -0.025272 | -0.066243 | -0.039889 | -0.041171 | -0.050357 | 0.000000 | NO | 0.011355 | -0.8442 |
| 140 | UAV_0 | 80 | 0 | 0.000 | -0.007705 | -0.106773 | -0.034462 | -0.041151 | -0.014695 | 0.000000 | NO | 0.013806 | -0.1066 |
| 140 | UAV_1 | 80 | 0 | 0.000 | -0.025211 | -0.066583 | -0.039951 | -0.041168 | -0.049970 | 0.000000 | NO | 0.011355 | -0.8365 |
| 170 | UAV_0 | 80 | 0 | 0.000 | -0.003424 | -0.110987 | -0.034960 | -0.041140 | -0.010194 | 0.000000 | NO | 0.013806 | -0.0801 |
| 170 | UAV_1 | 80 | 0 | 0.000 | -0.024109 | -0.070702 | -0.038438 | -0.041153 | -0.050301 | 0.000000 | NO | 0.011355 | -0.8138 |
| 200 | UAV_0 | 80 | 2 | 0.025 | +0.002960 | -0.117417 | -0.035329 | -0.041142 | -0.004691 | 0.027609 | NO | 0.013806 | -0.0493 |
| 200 | UAV_1 | 80 | 0 | 0.000 | -0.021937 | -0.079462 | -0.035487 | -0.041144 | -0.048418 | 0.000000 | NO | 0.011354 | -0.6629 |
| 230 | UAV_0 | 80 | 0 | 0.000 | -0.000075 | -0.114332 | -0.034632 | -0.041121 | -0.005926 | 0.000000 | NO | 0.013806 | -0.0488 |
| 230 | UAV_1 | 80 | 0 | 0.000 | -0.022291 | -0.077778 | -0.036435 | -0.041129 | -0.052363 | 0.000000 | NO | 0.011354 | -0.7074 |
| 260 | UAV_0 | 80 | 0 | 0.000 | -0.000690 | -0.113774 | -0.034797 | -0.041129 | -0.007301 | 0.000000 | NO | 0.013806 | -0.0609 |
| 260 | UAV_1 | 80 | 0 | 0.000 | -0.021812 | -0.079333 | -0.035145 | -0.041108 | -0.052631 | 0.000000 | NO | 0.011354 | -0.6866 |
| 290 | UAV_0 | 80 | 0 | 0.000 | -0.000918 | -0.113668 | -0.035482 | -0.041196 | -0.009078 | 0.000000 | NO | 0.013806 | -0.0701 |
| 290 | UAV_1 | 80 | 0 | 0.000 | -0.025577 | -0.065140 | -0.039495 | -0.041210 | -0.051613 | 0.000000 | NO | 0.011355 | -0.8628 |

## Aggregate result

```text
CHECKPOINT_EPOCH = 7
STATE_COUNT = 20
TOTAL_LEGAL_MOVE_CANDIDATES = 1600

STATES_WITH_ANY_GOOD_CANDIDATE = 1
STATES_WITH_NO_GOOD_CANDIDATE = 19
BEST_MOVE_BEATS_STAY_COUNT = 1
POLICY_TOP1_BEATS_STAY_COUNT = 0
EXPECTED_MOVE_BEATS_STAY_COUNT = 0

MEAN_BEST_DELTA_Q = -0.0160653287
MEAN_POLICY_TOP1_DELTA_Q = -0.0335834916
MEAN_POLICY_EXPECTED_DELTA_Q = -0.0411598216

MEAN_GOOD_CANDIDATE_FRACTION = 0.0012500000
MEDIAN_GOOD_CANDIDATE_FRACTION = 0.0000000000

MEAN_GOOD_POLICY_MASS = 0.0013804725
MEDIAN_GOOD_POLICY_MASS = 0.0000000000
MIN_GOOD_POLICY_MASS = 0.0000000000
MAX_GOOD_POLICY_MASS = 0.0276094495

NEAR_UNIFORM_STATE_COUNT = 0
MEAN_SPEARMAN_VALID = -0.4706809658
VALID_SPEARMAN_STATE_COUNT = 20

MEAN_BEST_CANDIDATE_MOVEMENT_COST = 0.0393800000
MEAN_POLICY_TOP1_MOVEMENT_COST = 0.0572000000
MEAN_POLICY_EXPECTED_MOVEMENT_COST = 0.0411518436

EXPECTED_Q_IDENTITY_MAX_ABS_DIFF = 5.6843e-14
CORR_DISTANCE_DELTA_Q = -0.5483480
CORR_MOVEMENT_COST_DELTA_Q = -0.5483480
```

Policy top-1 的平均 probability 为 `0.0138068`，best candidate 的平均 probability 为 `0.0123350`；top-1 在 20/20 state 都不是 best。候选概率并非 near-uniform（near-uniform state count 为 0），因此 Spearman 不应被简单解释为概率完全相同；其总体负值说明 candidate policy ranking 与 candidate value ranking 反向/未对齐，尤其 UAV_1 多数 state 的 Spearman 约为 `-0.66` 至 `-0.87`。

movement distance 与 `DeltaQ_c` 的相关系数为 `-0.5483480`，movement cost 与 `DeltaQ_c` 也为 `-0.5483480`。这是描述性关系，不是 movement-cost implementation bug 的证明。best candidate 的平均 movement cost 为 `0.03938`，policy top-1 为 `0.05720`，policy-expected MOVE 为 `0.04115184`，说明在唯一存在 good candidate 的局部 state，candidate policy 没有把概率集中到最优 candidate；但在另外 19 个 state，问题在 candidate policy 之前已经发生，因为不存在能击败 STAY 的合法 candidate。

## 根因分类

```text
PRIMARY_DIAGNOSIS = MIXED
OPERATION_TARGET_BIAS = YES
CANDIDATE_POLICY_BOTTLENECK = LOCALIZED_SUPPORT_ONLY (1/20 states)
SYSTEM_OBJECTIVE_PREFERS_STAY = DOMINANT (19/20 states)
MOVEMENT_COST_IMPLEMENTATION_BUG = NOT_ASSESSED_IN_THIS_TASK
FRESH_CHECKPOINT_COMPARISON = NOT_NEEDED (epoch7 result is classifiable)
H3/H5/H10/H30 = NOT_RUN_IN_THIS_TASK
```

推荐后续方向是先按 state 分类处理：19/20 state 应进入系统目标、movement-cost scale/geometry 与 STAY 对比的一致性核对；slot 200/UAV_0 这类局部 state 才适合继续做 candidate policy ranking/credit alignment 诊断。当前不应把所有 STAY collapse 归因于 candidate policy，也不应据此直接修改 movement cost 或训练参数。

## No-change flags

```text
SOURCE_MODIFIED = NO
TRAINING_EXECUTED = NO
OPTIMIZER_STEP_EXECUTED = NO
REWARD_CHANGED = NO
MOVEMENT_COST_CHANGED = NO
CACHE_CREDIT_CHANGED = NO
CANDIDATE_CREDIT_CHANGED = NO
OPERATION_CREDIT_CHANGED = NO
EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
FORMAL_EVALUATION_EXECUTED = NO
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
```
