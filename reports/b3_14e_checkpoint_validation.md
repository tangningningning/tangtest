# B3-14E：四组PPO现有Checkpoint策略有效性验证

## 范围与合同

- 使用本地 `adaptive_cache_mec` 环境、正式配置及Validation池，仅做冻结、确定性评价；每个候选Checkpoint使用Episode 30001–30005（每个300 Slot），没有PPO更新。
- 正式配置哈希与Checkpoint一致：raw `c56e70cde2a489bd4fa3b56cd26b75adc633d3eb48e40402eaa44a9275b892ff`；semantic `e4c5868737fde54b4e7eb474a3622242eb2dd177215a3f2a8f36dda872314b13`。架构33、Checkpoint 60、策略Schema `adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v4_multi_node_cache_factorized_v1`。
- 四个Run的Training Pool provenance一致；初始化策略网络权重逐张量完全相同。四组实际Macro Actor学习率由Run Manifest、Summary及Checkpoint核实，分别为1e-4、5e-4、1e-3、5e-3；其余记录的Actor/Critic学习率一致。
- 1e-3 Latest只到完整Update 5（20 Episode）；未加载或评价中断的Update 6状态。Validation由现有Runner/Evaluator完成，并通过其冻结参数与零优化器步检查。
- Return定义按既有指标核对：Task Utility − Cache Cost − UAV Movement Cost；所有行均平衡至1e-6内。Return标准差为5个Validation Episode的样本标准差。

## 1. Checkpoint清单

| Macro LR | Run | 有效Macro LR | Latest进度 | 实际存在的Checkpoint及进度 |
|---:|---|---:|---:|---|
| 1e-4 | `stageB3_14a_macro_actor_lr_1e_minus_4` | 0.0001 | E52/U13 | best_train_return_checkpoint.pt: U9/E36<br>initial_checkpoint.pt: U0/E0<br>latest_checkpoint.pt: U13/E52<br>update_010_checkpoint.pt: U10/E40 |
| 5e-4 | `stageB3_14a_macro_actor_lr_5e_minus_4` | 0.0005 | E4/U1 | best_train_return_checkpoint.pt: U1/E4<br>initial_checkpoint.pt: U0/E0<br>latest_checkpoint.pt: U1/E4 |
| 1e-3 | `stageB3_14a_macro_actor_lr_1e_minus_3` | 0.001 | E20/U5 | best_train_return_checkpoint.pt: U2/E8<br>initial_checkpoint.pt: U0/E0<br>latest_checkpoint.pt: U5/E20 |
| 5e-3 | `stageB3_14a_macro_actor_lr_5e_minus_3` | 0.005 | E88/U22 | best_train_return_checkpoint.pt: U9/E36<br>initial_checkpoint.pt: U0/E0<br>latest_checkpoint.pt: U22/E88<br>update_010_checkpoint.pt: U10/E40<br>update_020_checkpoint.pt: U20/E80 |

每组还实际存在的`best_train_return_checkpoint.pt`按Checkpoint计数列于上表；本轮未评价该训练诊断Checkpoint，也未用它替代固定Validation候选。仅评价Init、可用的Update 10/20与Latest：共11个Checkpoint、每个5个Episode。

## 2. Validation比较

| LR | Update | Checkpoint | Mean Return ± SD | Δ vs Init | Return改善数 | Mean Task Utility | Mean Cache Cost | Mean UAV Cost | Mean Holding |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 0 | Init | 2993.93 ± 195.77 | +0.00 | 0/5 | 2994.56 | 0.623 | 0.000 | 1.327 |
| 1e-4 | 10 | Update 10 | 2998.69 ± 192.28 | +4.76 | 4/5 | 3000.68 | 0.269 | 1.716 | 5.000 |
| 1e-4 | 13 | Latest | 2992.81 ± 217.34 | -1.12 | 3/5 | 2994.60 | 0.069 | 1.716 | 5.000 |
| 5e-4 | 0 | Init | 2993.93 ± 195.77 | +0.00 | 0/5 | 2994.56 | 0.623 | 0.000 | 1.327 |
| 5e-4 | 1 | Latest | 2901.16 ± 227.15 | -92.78 | 0/5 | 2902.95 | 0.168 | 1.627 | 1.018 |
| 1e-3 | 0 | Init | 2993.93 ± 195.77 | +0.00 | 0/5 | 2994.56 | 0.623 | 0.000 | 1.327 |
| 1e-3 | 5 | Latest | 2944.49 ± 213.77 | -49.45 | 1/5 | 2946.49 | 0.371 | 1.634 | 4.762 |
| 5e-3 | 0 | Init | 2993.93 ± 195.77 | +0.00 | 0/5 | 2994.56 | 0.623 | 0.000 | 1.327 |
| 5e-3 | 10 | Update 10 | 3070.24 ± 202.59 | +76.30 | 5/5 | 3072.43 | 0.477 | 1.716 | 5.000 |
| 5e-3 | 20 | Update 20 | 3051.43 ± 224.51 | +57.49 | 3/5 | 3053.72 | 0.575 | 1.716 | 4.632 |
| 5e-3 | 22 | Latest | 3024.18 ± 219.29 | +30.24 | 4/5 | 3026.23 | 0.332 | 1.716 | 4.762 |

配对`ΔReturn = Return_checkpoint − Return_init`，Seed顺序均为30001、30002、30003、30004、30005：

| LR / Checkpoint | 逐Seed ΔReturn | Mean Δ | 改善Episode |
|---|---|---:|---:|
| 1e-4 U10 | -13.92, +9.88, +9.91, +5.42, +12.50 | +4.76 | 4/5 |
| 1e-4 U13 | -21.34, -39.95, +22.88, +26.60, +6.21 | -1.12 | 3/5 |
| 5e-4 U1 | -105.89, -147.80, -89.44, -52.28, -68.46 | -92.78 | 0/5 |
| 1e-3 U5 | -82.01, -45.18, -77.56, +8.74, -51.23 | -49.45 | 1/5 |
| 5e-3 U10 | +108.44, +34.24, +72.04, +40.56, +126.24 | +76.30 | 5/5 |
| 5e-3 U20 | +145.74, -30.01, +28.18, -4.21, +147.76 | +57.49 | 3/5 |
| 5e-3 U22 | +61.85, -39.51, +33.06, +15.40, +80.42 | +30.24 | 4/5 |

Update 10同预算比较：5e-3相对1e-4的逐Seed Return差为`+122.37, +24.36, +62.13, +35.14, +113.74`，均值`+71.55`（5/5为正）。这是5个Episode上的初步配对证据，不是最终学习率排名。

## 3. Cache、Holding、UAV及系统指标

| LR / Update | ΔUtility | ΔCache Cost | ΔUAV Cost | Mean Holding | KEEP动作率 / 实际Cache更新率 | UAV0/1非STAY动作率 | UAV0/1 MOVE概率质量 | Cloud/Edge卸载率 | Mean Delay (s) | Mean Energy (J) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 U10 | +6.12 | -0.355 | +1.716 | 5.000 | 0.9833 / 0.0167 | 1.000 / 1.000 | 0.543 / 0.518 | 0.594 / 0.406 | 13.223 | 0.2586 |
| 1e-4 U13 | +0.04 | -0.554 | +1.716 | 5.000 | 0.9833 / 0.0167 | 1.000 / 1.000 | 0.644 / 0.572 | 0.599 / 0.401 | 13.386 | 0.2454 |
| 5e-4 U1 | -91.60 | -0.456 | +1.627 | 1.018 | 0.9966 / 0.0034 | 0.968 / 1.000 | 0.513 / 0.516 | 0.641 / 0.359 | 14.485 | 0.2460 |
| 1e-3 U5 | -48.07 | -0.253 | +1.634 | 4.762 | 0.9841 / 0.0159 | 0.905 / 1.000 | 0.522 / 0.530 | 0.600 / 0.400 | 13.892 | 0.2559 |
| 5e-3 U10 | +77.87 | -0.147 | +1.716 | 5.000 | 0.9833 / 0.0167 | 1.000 / 1.000 | 0.572 / 0.514 | 0.590 / 0.410 | 12.229 | 0.2787 |
| 5e-3 U20 | +59.16 | -0.048 | +1.716 | 4.632 | 0.9846 / 0.0154 | 1.000 / 1.000 | 0.670 / 0.527 | 0.592 / 0.408 | 12.519 | 0.2694 |
| 5e-3 U22 | +31.67 | -0.291 | +1.716 | 4.762 | 0.9841 / 0.0159 | 1.000 / 1.000 | 0.612 / 0.515 | 0.583 / 0.417 | 12.870 | 0.2660 |

- Evaluator保存的是KEEP动作率，不是Cache KEEP概率；后者`NOT_MEASURED`。实际Cache更新率已记录。所有更新策略的KEEP动作率约0.983–0.997；实际更新率约0.003–0.017。不能把KEEP动作率解释成策略概率。
- UAV MOVE率按非零（非STAY）动作索引计；Evaluator没有单独记录实际坐标位移率，因此物理位移率`NOT_MEASURED`。Evaluator只保存Cloud/Edge总卸载率，没有SV、RSU、UAV分型任务数，分型卸载率`NOT_MEASURED`。
- 5e-3 U10在这组配对场景中Return最高（均值较Init +76.30、5/5改善），Task Utility +77.87，Cache Cost −0.147，UAV Cost +1.716，平均任务时延从Init的13.338降至12.229 s；车辆能耗升至0.2787 J（Init 0.2526 J）。

## 4. 判断与后续

- `CACHE_POLICY_IMPROVEMENT`: **INSUFFICIENT_EVIDENCE（净收益层面）**。多个Checkpoint的Cache成本下降，但1e-4 U10/Latest的实际更新率反而高于Init，5e-4与1e-3的Task Utility及Return下降。低成本本身不能证明Cache策略整体改善；与UAV及其他Macro动作共同变化，无法作单机制因果归因。
- `HOLDING_POLICY_IMPROVEMENT`: **观察到Holding延长，效益未证实**。1e-4、1e-3、5e-3的候选平均Holding约4.63–5 Slot；5e-4 Update 1约1.02 Slot。Return表现并不随Holding单调改善。
- `UAV_POLICY_IMPROVEMENT`: **INSUFFICIENT_EVIDENCE（通用结论）**。更新策略的非STAY动作率约0.90–1.00，MOVE概率质量约0.51–0.67，且UAV成本增加；只有5e-3 U10同时呈现更高Task Utility与更低时延，其他候选并未呈现一致收益。实体坐标位移率及分节点卸载率未测量。
- `VALIDATION_RETURN_IMPROVEMENT`: **有候选出现配对改善，但不普遍**。5e-3 U10为5/5；1e-4 U10为4/5但均值仅+4.76，1e-4 U13为3/5且均值−1.12；5e-4 U1为0/5且均值−92.78；1e-3 U5为1/5且均值−49.45；5e-3 U20为3/5、U22为4/5且均值仍高于Init。
- `LEARNING_RATE_COMPARISON_VALID`: **YES，仅Update 10的1e-4与5e-3可公平横向比较**。其余学习率没有Update 10 Checkpoint；不同更新组的Latest仅作各自纵向结果展示，不能作同预算排名。
- **是否支持修改Macro/Micro PPO：否。** 现有结果不足以定位算法缺陷或区分Macro与Micro贡献；没有修改依据。
- **是否继续现有正式训练：建议完成各LR的原定相同训练预算后再做最终选择。** 当前组进度不等（1、5、13、22更新），5个Episode只是初筛；本轮没有恢复或启动训练。5e-3 U10是当前较有希望的Checkpoint，但不能称为最优学习率。

## 交付状态

`TRAINING_STARTED = NO`  
`TRAINING_RESUMED = NO`  
`PPO_PARAMETERS_CHANGED = NO`  
`CHECKPOINTS_MODIFIED = NO`  
`VALIDATION_EXECUTED = YES (11 Checkpoints × 5 Episodes)`  
`READY_FOR_NEXT_DECISION = YES`
