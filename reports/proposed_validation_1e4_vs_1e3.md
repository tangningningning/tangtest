# 第一阶段正式 Validation：Macro Actor LR 1e-4 vs 1e-3

日期：2026-10-09

## Checkpoint 与兼容性

两组均实际使用正式 run 下的 latest_checkpoint.pt（未使用 best_train_return_checkpoint.pt）。run_manifest、run_summary 与训练日志计数均为 300 Episodes / 75 Policy Updates；Evaluator 元数据检查与正式 PPOTrainer.load_checkpoint() 均通过。Checkpoint 在评价前后 SHA 一致。

| Macro Actor LR | Episodes / Updates | Checkpoint / Architecture | Manifest LR | Config raw SHA-256 | Config semantic SHA-256 | latest Checkpoint SHA-256 |
|---|---:|---:|---:|---|---|---|
| 1e-4 | 300 / 75 | 58 / 31 | 1e-04 | fe5d4f7bc57a424911166bcfe921c2ec5a7627b0a6aa759c4993f7f488a3775c | 64e367ee5c67cbcc86b7741ef0e97fe0de1fecd6b3c07039f77a29c1b17ae45e | 34e0377c521e03a944ccfc59817a31bca24cee0c5731555aeea10a8fd6e0e9d1 |
| 1e-3 | 300 / 75 | 58 / 31 | 1e-03 | fe5d4f7bc57a424911166bcfe921c2ec5a7627b0a6aa759c4993f7f488a3775c | 64e367ee5c67cbcc86b7741ef0e97fe0de1fecd6b3c07039f77a29c1b17ae45e | e6ed46f593f724ff08c89106376ed33aba5f476e6d79e24e5a2613323d379b2e |

正式 config 使用项目内 configs/proposed/system_config.toml；其 raw SHA-256 为 `fe5d4f7bc57a424911166bcfe921c2ec5a7627b0a6aa759c4993f7f488a3775c`，与两个 manifest 记录一致，semantic hash 也一致。两份 latest Checkpoint 均由正式 evaluator 的元数据校验，并在正式 Trainer load 中以 raw/semantic config hash 完整加载。

## Evaluation 协议与原始数据

- 环境池：environment_data/proposed_joint_validation；同一批 episode_30001–episode_30030。
- 每个 realization 300 Slot；每个策略 repeats=1，policy-action-mode=deterministic，CPU。
- 使用正式 python -m ppo_simple.evaluator。两组各执行30个独立 Validation realization，不访问 Test seeds 40001–40100。
- 结果分别保留在：
  - runs/proposed_validation_lr_1e_minus_4_seed30001_30030_deterministic/
  - runs/proposed_validation_lr_1e_minus_3_seed30001_30030_deterministic/
- 各 seed 的原始结果位于 episode_<seed>/proposed_repeat_summaries.json，根级 proposed_episode_pool_summary.json 保存正式 aggregate。
- 核验两份 aggregate 均为30条唯一 seed 30001–30030、seed_start=30001、episodes=30、repeats=1、deterministic、statistical_unit=environment_realization；每个 episode 的联合动作计数合计300 Slot。

实际调用参数：

```text
python -m ppo_simple.evaluator --config configs/proposed/system_config.toml --checkpoint <latest_checkpoint.pt> --run-dir <fresh Validation directory> --episode-pool environment_data/proposed_joint_validation --seed-start 30001 --episodes 30 --repeats 1 --policy-action-mode deterministic --device cpu
```

## 汇总指标

数值先在每个 environment realization 内计算，再对30个 realization 等权汇总。配对差值均为 1e-3 − 1e-4；比例的差值用百分点（pp）表示。

| 指标 | LR 1e-4 均值 | LR 1e-3 均值 | 配对均值差 (1e-3−1e-4) | 单位 / 口径 |
|---|---|---|---|---|
| Return | 2210.221 | 3061.470 | 851.249 | return units |
| Task Utility (total_q) | 2212.190 | 3063.141 | 850.952 | utility units |
| Mean task delay | 16.4475 | 10.8121 | -5.6354 | s/task |
| Mean vehicle energy | 1.1639 | 0.4605 | -0.7034 | J/task |
| Cache cost | 0.2530 | 0.8172 | 0.5642 | cost units |
| UAV movement cost | 1.7160 | 0.8544 | -0.8616 | cost units |
| Actual Cache update rate | 0.333% | 6.667% | 6.333% pp | % macro opportunities |
| Cache change rate | 0.00333 | 0.01333 | 0.01000 | changed nodes/slot |
| UAV0 MOVE rate | 100.00% | 99.96% | -0.04% pp | % slots |
| UAV1 MOVE rate | 100.00% | 0.00% | -100.00% pp | % slots |
| STAY/STAY (U0/U1) | 0.00% | 0.04% | 0.04% pp | % slots |
| MOVE/STAY (U0/U1) | 0.00% | 99.96% | 99.96% pp | % slots |
| STAY/MOVE (U0/U1) | 0.00% | 0.00% | 0.00% pp | % slots |
| MOVE/MOVE (U0/U1) | 100.00% | 0.00% | -100.00% pp | % slots |
| Mean Holding duration | 1.000 | 5.000 | 4.000 | slots/frame |

Cache 指标口径按正式代码字段区分：actual_cache_update_rate 是 Macro transition 中发生实际缓存更新的比例，分母为宏决策机会数；cache_change_rate 是所有 Micro Slot 的 actual_cache_update_count 除以 Slot 数，表示每 Slot 平均实际变更的缓存节点数。二者分别报告，不能互换。

Holding 分布按每个 Episode 内各实际 Holding duration 的次数归一化后，在30个 realization 间等权求均值；同时列出每 Episode 对应 duration 的平均 frame 数。

| 实际 Holding duration | LR 1e-4 Episode平均占比 | LR 1e-3 Episode平均占比 | 配对差 | LR 1e-4 平均frame数 | LR 1e-3 平均frame数 |
|---|---|---|---|---|---|
| 1 | 100.00% | 0.00% | -100.00 pp | 300.00 | 0.00 |
| 2 | 0.00% | 0.00% | +0.00 pp | 0.00 | 0.00 |
| 3 | 0.00% | 0.00% | +0.00 pp | 0.00 | 0.00 |
| 4 | 0.00% | 0.00% | +0.00 pp | 0.00 | 0.00 |
| 5 | 0.00% | 100.00% | +100.00 pp | 0.00 | 60.00 |

## Return 配对差值

配对 Δ Return = Return(1e-3) − Return(1e-4)，n=30 个独立 Validation realization。均值 +851.249，中位数 +632.151，样本标准差 523.903，95%配对 t 区间 [+655.620, +1046.878]。1e-3 胜出 30 个 seed，1e-4 胜出 0 个，平局 0 个。

| episode_seed | Return 1e-4 | Return 1e-3 | Δ Return (1e-3−1e-4) | Seed胜出方 |
|---|---|---|---|---|
| 30001 | 2692.828 | 3378.646 | +685.818 | 1e-3 |
| 30002 | 2143.969 | 2713.039 | +569.071 | 1e-3 |
| 30003 | 2647.569 | 3283.124 | +635.555 | 1e-3 |
| 30004 | 3154.041 | 3408.911 | +254.870 | 1e-3 |
| 30005 | 2675.523 | 3306.261 | +630.738 | 1e-3 |
| 30006 | 1766.253 | 2717.925 | +951.672 | 1e-3 |
| 30007 | 2685.495 | 3055.650 | +370.156 | 1e-3 |
| 30008 | 2505.799 | 3100.424 | +594.625 | 1e-3 |
| 30009 | 2717.698 | 3131.775 | +414.076 | 1e-3 |
| 30010 | 774.409 | 2754.619 | +1980.210 | 1e-3 |
| 30011 | 3105.092 | 3465.062 | +359.970 | 1e-3 |
| 30012 | 2519.431 | 3149.386 | +629.955 | 1e-3 |
| 30013 | 2898.335 | 3285.234 | +386.899 | 1e-3 |
| 30014 | 2394.502 | 2904.199 | +509.696 | 1e-3 |
| 30015 | 2009.334 | 3053.120 | +1043.786 | 1e-3 |
| 30016 | 2476.967 | 3075.872 | +598.906 | 1e-3 |
| 30017 | 2878.055 | 3086.105 | +208.051 | 1e-3 |
| 30018 | 1194.706 | 2929.253 | +1734.547 | 1e-3 |
| 30019 | 2128.421 | 2926.855 | +798.434 | 1e-3 |
| 30020 | 2158.025 | 2791.589 | +633.563 | 1e-3 |
| 30021 | 1207.565 | 2708.875 | +1501.310 | 1e-3 |
| 30022 | 1445.763 | 2957.698 | +1511.936 | 1e-3 |
| 30023 | 2648.486 | 3172.427 | +523.942 | 1e-3 |
| 30024 | 1382.337 | 2881.264 | +1498.927 | 1e-3 |
| 30025 | 2407.559 | 3271.297 | +863.738 | 1e-3 |
| 30026 | 2038.035 | 3290.014 | +1251.980 | 1e-3 |
| 30027 | 2769.397 | 3232.502 | +463.106 | 1e-3 |
| 30028 | 1331.428 | 2884.992 | +1553.564 | 1e-3 |
| 30029 | 2841.107 | 3242.037 | +400.930 | 1e-3 |
| 30030 | 708.490 | 2685.931 | +1977.441 | 1e-3 |

## Return 差异来源

以 evaluator 的累计字段作观测分解，不重算系统 Reward。每 seed 满足 Return 差值约等于 Task Utility 差值 − Cache cost 差值 − UAV movement cost 差值；最大绝对残差 7.049e-12。

| 组成项 | LR 1e-3 − LR 1e-4 | 说明 |
|---|---|---|
| Task Utility (Δ total_q) | +850.9516 | 对 Return 的正向贡献 |
| Cache cost (Δ total_cache_cost) | +0.5642 | 对 Return 贡献 -0.5642（取负） |
| UAV movement cost (Δ total_movement_cost) | -0.8616 | 对 Return 贡献 +0.8616（取负） |
| 重构 Δ Return | +851.2490 | 与实测配对均值差对照 |
| 实测 Δ Return | +851.2490 | Return(1e-3) − Return(1e-4) |

实测差异主要来自 Task Utility：total_q 增加 850.952。LR 1e-3 的 Cache cost 高 0.564，抵消 Return；UAV movement cost 低 0.862，增加 Return；两类成本合计仅贡献约 +0.297。因此这次 Return 提升不是由更低的 Cache 重配置成本解释。

## UAV 联合 MOVE 与系统效用的观测关联

以下 Pearson r 是30个不同环境 realization 上的描述性相关，不是动作因果效应；环境难度、任务分布等可同时影响动作比例与效用。any-UAV-MOVE = 1 − STAY/STAY。

| LR | r(any-MOVE, Return) | r(any-MOVE, Task Utility) | r(MOVE/MOVE, Return) | r(MOVE/MOVE, Task Utility) |
|---|---|---|---|---|
| 1e-4 | 不可计算（比例无变化） | 不可计算（比例无变化） | 不可计算（比例无变化） | 不可计算（比例无变化） |
| 1e-3 | +0.199 | +0.199 | 不可计算（比例无变化） | 不可计算（比例无变化） |

| 配对变化 | Pearson r 与 Δ Return |
|---|---|
| Δ any-UAV-MOVE rate vs Δ Return | +0.029 |
| Δ MOVE/MOVE rate vs Δ Return | 不可计算（比例无变化） |

联合比例本身见汇总指标表（STAY/STAY、MOVE/STAY、STAY/MOVE、MOVE/MOVE，动作顺序均为 UAV0/UAV1）。相关仅描述当前策略跨环境变化，不能证明增加 MOVE 导致效用升降。

## 结论

1. **1e-3 在本次独立 Validation 池中的 Return 表现：** 在这30个配对Validation realization中，1e-3的平均Return较高，且配对95% t区间整体高于0；这支持1e-3在本配对池内有更高的平均Return，不能外推为所有环境或四学习率的全局结论。
2. **Task Utility 与成本来源：** Return 提升主要由 total_q 增加 850.952 解释；Cache cost 上升抵消 0.564，UAV movement cost 下降贡献 0.862，成本净贡献约 +0.297。与此同时，平均任务时延降低 5.6354 s/task，车辆平均能耗降低 0.7034 J/task。
3. **Cache 与 Holding：** LR 1e-3 的 actual_cache_update_rate 为 6.667%，高于 1e-4 的 0.333%；cache_change_rate 为 0.01333 对 0.00333 changed nodes/slot。Holding 也从 1-slot 占比 100% 变为 5-slot 占比 100%。这些行为变化与更高 Task Utility 同时出现，但不能据此断言因果。
4. **UAV 联合动作：** UAV0 在两组几乎总 MOVE；主要差异是 UAV1：1e-4 为 100% MOVE/MOVE，1e-3 为 99.96% MOVE/STAY。1e-3 的 any-UAV-MOVE 跨环境比例与 Return/Task Utility 的相关为 +0.199，而配对 Δ any-MOVE 与 Δ Return 的相关为 +0.029；比例变异有限且环境难度等混杂，不能把观察到的行为差异解释为 Return 提升的原因。
5. **是否足以正式选择全局学习率：** 本结果只比较1e-4和1e-3，不覆盖仍在训练的5e-4、5e-3，也没有使用保留的 Test seeds。即便本配对池支持其中一方，仍只能形成这两个已完成 Checkpoint 的 pairwise Validation 证据，不能替代四学习率统一 Validation 和最终 Test。

## 限制与完整性

- 30个 seed 是本报告唯一统计单位；未把Slot或repeat当成独立样本。deterministic策略每环境只评价一次。
- 结果针对固定的 Validation realization 池，适用于配对比较；不代表所有未来环境分布。
- 没有训练、optimizer 更新、PPO/Reward/系统模型修改或 Test 数据读取。
- 两个 evaluator 原始结果目录均保留，供后续四学习率统一比较。
