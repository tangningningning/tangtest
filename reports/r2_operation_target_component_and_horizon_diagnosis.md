# R2 operation target component and horizon diagnosis

日期：2026-09-18  
仓库：`C:\Users\tangning\Desktop\code\adaptive_cache_mec`  
冻结 realization：`runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20001`  
配置：`configs/proposed/system_config_ppo_recovery_r2_6e4.toml`  
案例：slot `20,50,80,110,140,170,200,230,260,290` × UAV `0,1`，共 20 条记录/版本。

## 结论

`PRIMARY_DIAGNOSIS = TARGET_BIAS` 仍然成立，而且本轮把来源进一步定位到即时 reward target：MOVE 的期望正式 `movement_cost` 增量约为 `+0.0411518`，使 `DeltaR` 约为 `-0.0411682`；critic continuation 项 `DeltaV` 的最新 epoch7 均值只有 `+0.0000037`。因此当前 operation target 的负 MOVE 偏置主要不是 horizon/critic 项造成的，而是即时 reward 中的移动成本项造成的。

最新 epoch7 的 H=3 bounded gate 仍为 20/20 负值，均值从 H=1 的 `-0.0411645` 变为 `-0.0412183`，没有负转正。没有证据支持“拉长 target horizon 会自行恢复 MOVE”。

建议方向：`RECOMMEND_TARGET_COMPONENT_ALIGNMENT_AUDIT_BEFORE_ANY_FURTHER_TRAINING`。下一轮应优先审计 operation-side target 与正式 movement-cost 量纲/权重、MOVE 候选语义及 credit alignment；不建议仅通过继续训练或单纯增加 horizon 处理当前现象。本报告不修改这些语义。

## 诊断契约与执行范围

本轮调用现有 `_compute_r2_operation_credit`，只在临时诊断脚本中拦截其已经调用的 `execute_slot` 返回值，以保留 exact branch result。每个目标 transition 都保留 STAY 和全部合法 MOVE 候选，并使用现有 policy candidate distribution 做 MOVE 期望；没有候选抽样、没有 action-id forcing、没有固定位置替换。

对每个 UAV 验证：

```text
Q_stay = r_stay + gamma * V_stay_next
Q_move = sum_c pi_old(c|MOVE,s) * (r_move,c + gamma * V_move,c,next)
DeltaR = R_move_expected - r_stay
DeltaV = gamma * (V_move_expected - V_stay_next)
DeltaQ = DeltaR + DeltaV
```

其中 reward 使用正式 `execute_slot().system_utility`，并由现有 `_check_reward` 验证为：

```text
system_utility = actual_task_utility - cache_cost - movement_cost
```

`cache_cost` 与 `movement_cost` 是结果对象独立返回的正式 cost。结果对象没有独立的 `delay_cost` 或 `energy_cost` 字段/系数，因此没有反推这两项；报告中的 `delay_total_s`、`energy_total_j` 若需要仅可作为原始物理量，不能称为 reward cost。

## One-step component decomposition

负/正比例均按 20 条 `DeltaQ`/`DeltaR`/`DeltaV` 记录统计；`negative:positive magnitude ratio` 在没有正样本时记为 N/A。

| checkpoint | DeltaQ mean / median | DeltaR mean / median | DeltaV mean / median | movement-cost delta mean | cache-cost delta mean | DeltaQ neg / pos | residual max |
|---|---:|---:|---:|---:|---:|---:|---:|
| fresh | -0.039604712 / -0.039614033 | -0.039608931 / -0.039657135 | +0.000004219 / +0.000021281 | +0.040251759 | -0.0000000056 | 1.00 / 0.00 | 3.01e-15 |
| best epoch5 | -0.041143808 / -0.041142114 | -0.041162152 / -0.041153069 | +0.000018344 / +0.000017329 | +0.041151337 | -0.0000000033 | 1.00 / 0.00 | 8.70e-14 |
| latest epoch7 | -0.041164464 / -0.041147635 | -0.041168164 / -0.041166014 | +0.000003700 / +0.000010061 | +0.041151844 | +0.0000000040 | 1.00 / 0.00 | 7.64e-14 |

按版本的完整统计：

```text
DELTA_Q_COMPONENT_DECOMPOSITION_AVAILABLE = true
FRESH_DELTA_Q_MEAN = -0.0396047118
FRESH_DELTA_R_MEAN = -0.0396089308
FRESH_DELTA_V_MEAN = +0.0000042190
FRESH_DELTA_Q_NEGATIVE_FRACTION = 1.00
FRESH_DELTA_Q_POSITIVE_FRACTION = 0.00
FRESH_DELTA_Q_NEGATIVE_TO_POSITIVE_RATIO = N/A

BEST_EPOCH5_DELTA_Q_MEAN = -0.0411438075
BEST_EPOCH5_DELTA_R_MEAN = -0.0411621515
BEST_EPOCH5_DELTA_V_MEAN = +0.0000183440
BEST_EPOCH5_DELTA_Q_NEGATIVE_FRACTION = 1.00
BEST_EPOCH5_DELTA_Q_POSITIVE_FRACTION = 0.00
BEST_EPOCH5_DELTA_Q_NEGATIVE_TO_POSITIVE_RATIO = N/A

LATEST_EPOCH7_DELTA_Q_MEAN = -0.0411644643
LATEST_EPOCH7_DELTA_R_MEAN = -0.0411681642
LATEST_EPOCH7_DELTA_V_MEAN = +0.0000036999
LATEST_EPOCH7_DELTA_Q_MEDIAN = -0.0411476347
LATEST_EPOCH7_DELTA_R_MEDIAN = -0.0411660137
LATEST_EPOCH7_DELTA_V_MEDIAN = +0.0000100609
LATEST_EPOCH7_DELTA_Q_NEGATIVE_FRACTION = 1.00
LATEST_EPOCH7_DELTA_Q_POSITIVE_FRACTION = 0.00
LATEST_EPOCH7_DELTA_Q_NEGATIVE_TO_POSITIVE_RATIO = N/A

FRESH_DELTA_MOVEMENT_COST_MEAN = +0.0402517589
BEST_EPOCH5_DELTA_MOVEMENT_COST_MEAN = +0.0411513373
LATEST_EPOCH7_DELTA_MOVEMENT_COST_MEAN = +0.0411518441
FRESH_DELTA_CACHE_COST_MEAN = -5.5585e-09
BEST_EPOCH5_DELTA_CACHE_COST_MEAN = -3.3350e-09
LATEST_EPOCH7_DELTA_CACHE_COST_MEAN = +3.9957e-09
DELTA_DELAY_COST_MEAN = N/A (no independent formal delay-cost component)
DELTA_ENERGY_COST_MEAN = N/A (no independent formal energy-cost component)
ONE_STEP_TARGET_MECHANISM = IMMEDIATE_REWARD_MOVEMENT_COST_DOMINATES
```

各 UAV 的最新 epoch7 均值也一致：UAV0 `DeltaQ=-0.0411580`、`DeltaR=-0.0411681`、`DeltaV=+0.0000101`；UAV1 `DeltaQ=-0.0411709`、`DeltaR=-0.0411683`、`DeltaV=-0.0000027`。这不是单一 UAV 的局部异常。

identity residual 的最大绝对值为 `8.70e-14`（三版本合并最大值），远低于要求的 `1e-6`。

## Horizon gate on latest epoch7

H=1 与 H=3 使用同一冻结 realization、同一 20 条案例和同一 latest epoch7 policy。当前 slot 的 MOVE 候选全部枚举；后续 slot 使用 production rollout 的确定性 frozen-policy action，不固定位置、不强制 action id。由于后续路径采用确定性策略，common-random-number 对本次 continuation 不适用。

| horizon | executed | count | negative / positive fraction | mean | median | sign reversal from H1 |
|---|---|---:|---:|---:|---:|---:|
| H=1 | yes | 20 | 1.00 / 0.00 | -0.0411644643 | -0.0411476347 | — |
| H=3 | yes | 20 | 1.00 / 0.00 | -0.0412183128 | -0.0412062636 | 0 negative→positive; 0 positive→negative |
| H=5 | no | — | N/A | N/A | N/A | N/A |
| H=10 | no | — | N/A | N/A | N/A | N/A |

H=3 exact continuation耗时约 `1469.2 s`（约 24.5 分钟），因此按 bounded cost gate 停止，没有把 H=5/H=10 的结果伪装成已测量结果，也没有从 H=3 外推它们。

```text
H1_NEGATIVE_FRACTION = 1.00
H1_POSITIVE_FRACTION = 0.00
H3_NEGATIVE_FRACTION = 1.00
H3_POSITIVE_FRACTION = 0.00
H3_NEGATIVE_TO_POSITIVE_REVERSALS = 0
MULTISTEP_SIGN_RECOVERY = false (H=1 -> H=3)
H5_STATS = NOT_RUN_TOO_EXPENSIVE
H10_STATS = NOT_RUN_TOO_EXPENSIVE
```

## Required no-change flags

```text
NO_TRAINING = true
NO_OPTIMIZER_STEP = true
NO_TRAINING_RESUME = true
CHECKPOINTS_READ_ONLY = true
NO_PRODUCTION_SOURCE_MODIFICATION = true
NO_CONFIG_MODIFICATION = true
NO_CHECKPOINT_MODIFICATION = true
NO_FORMAL_EVALUATION = true
NO_FINAL_SEED_EVALUATION = true
NO_SUMO_OR_REALIZATION_REGENERATION = true
NO_SAMPLE_EXPANSION = true
NO_ACTION_ID_FORCING = true
NO_FIXED_POSITION_OVERRIDE = true
NO_TRAINING_TERMINATION_OR_RESTART = true
CURRENT_FORMAL_RUN_PRESERVED = true
CURRENT_FORMAL_RUN_COMPLETED_EPOCHS = 7
```

临时诊断脚本和中间 JSON 在报告确认后删除；正式 run、checkpoint、既有报告和其他用户修改保持不变。
