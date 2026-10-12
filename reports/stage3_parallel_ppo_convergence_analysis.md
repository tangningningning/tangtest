# 阶段3并行任务B：PPO 收敛瓶颈分析

## 范围与数据

只读分析当前 src/ppo_simple/trainer.py、Stage3B PPO 配置和已完成的四组旧环境正式训练日志。每组读取对应的 episode_metrics.jsonl、update_metrics.jsonl、run_manifest.json 和 run_summary.json（300 Episodes、75 Policy Updates）；回报分段由原始 Episode 行计算，优化诊断与耗时由 75 行 Update 记录汇总。另引用已存在的旧环境确定性 Validation 汇总，不在本任务重新运行 Validation。

四个训练目录是旧 Cloud 合同下的结果，不能用来判定 Stage3B 有限 Cloud 池的最优学习率。四个 run_config.toml 保留 5e-4 的基准值；变体 Macro Actor LR 由训练脚本的运行时覆盖给出，并由 Manifest 与 macro_actor_learning_rate_applied 日志交叉确认。

## 四个 Macro Actor 学习率的学习轨迹

下表 Episode Return 为各连续 100 Episode 区间均值；每区间对应 25 次 Policy Update。末列为旧环境中已有的 30-seed 确定性 Validation 均值±SD，只作历史结果，不代表 Stage3B 性能。

| Macro Actor LR | Ep 1–100 | Ep 101–200 | Ep 201–300 | 最后20 Episode | 旧环境已有 Validation Return 均值±SD |
|---:|---:|---:|---:|---:|---:|
| 1e-4 | -608.8 | -182.1 | 590.5 | 824.0 | 2210.2 ± 685.9 |
| 5e-4 | 321.6 | 2553.9 | 3085.6 | 3127.5 | **3175.5 ± 183.9** |
| 1e-3 | 1066.1 | 2670.9 | 2927.3 | 2935.6 | 3061.5 ± 230.1 |
| 5e-3 | **1457.5** | 2677.6 | 2791.2 | 2663.1 | 2577.9 ± 672.6 |

解读：1e-4 的前 200 Episode 进展最慢；其 Macro KL 很小，策略更新幅度也小。1e-3、5e-3 前100 Episode 均值更高，5e-3 最快，但后段均值回落且有较大 KL/Clip 峰值。5e-4 起步较慢于 1e-3/5e-3，随后提升到最高后期训练回报；旧环境既有 Validation 也以 5e-4 最高。由此能确认 LR 会影响旧环境的样本效率和策略变化幅度；不能推出 5e-4 是 Stage3B 的最优值。

Episode Return 包含不同轨迹状态下的整回合收益；它不是固定状态的因果学习曲线。Validation 已在本次之前完成，本报告没有读取最终 Test seeds。

## 每4个 Episode 的更新机制与步数

PPOTrainer 强制 realizations_per_policy_update=4。run_training() 在这四个完整 Episode 的采样期间冻结策略，并核对 Optimizer 计数未改变；四条轨迹全部收集后调用一次 update_from_episodes()。因此 300 个 Episode 对应 75 次更新，而不是每个 Episode 更新一次。

| 优化器 | 每次 Policy Update 的实际步数 | 300 Episodes 的日志计数 |
|---|---:|---:|
| Micro Actor | 4 epochs × 10 minibatches = 40 | 3000 |
| Micro Critic | 同 Micro minibatch 循环，40 | 3000 |
| Macro Actor | 4 epochs × 1 full-batch step = 4 | 300（5e-3 因两次 KL early stop 为 298） |
| Macro Critic | 固定 50 步 | 3750 |
| Long-Value Critic | 固定 50 步 | 3750 |

所以 Macro 的 optimizer_steps=4050 是 Macro Actor 300 + Macro Critic 3750；5e-3 的 Macro 总数为 4048。Micro 的 Actor 和 Critic 各 3000 步。所有 LR 变体每四个 Episode 更新一次、总计75次；只有 5e-3 的 Macro Actor 有两次更新提前结束。日志没有对照每 Episode 更新或其他组大小的训练，因此无法用这些结果判断更新频率是否过低。

## KL、Clip、Entropy 与梯度裁剪

配置 clip_epsilon=0.2、Macro/Micro target KL 分别为 0.10/0.02；代码在本批次 KL 大于 target 的 1.5 倍时提前结束相应 PPO epoch。全组平均诊断如下：

| Macro LR | Macro KL | Macro Clip Fraction | Macro Entropy | Micro KL | Micro Clip Fraction | Micro Entropy | Macro early stop |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 0.00035 | 0.21% | 4.474 | 0.00240 | 1.79% | 4.652 | 0/75 |
| 5e-4 | 0.00219 | 2.19% | 2.292 | 0.00312 | 2.62% | 2.012 | 0/75 |
| 1e-3 | 0.00608 | 7.09% | 2.787 | 0.00378 | 2.97% | 3.905 | 0/75 |
| 5e-3 | 0.01676 | 9.53% | 2.505 | 0.00258 | 2.21% | 4.990 | 2/75 |

5e-3 的 Macro KL 最大值为 0.435、Clip Fraction 最大值为 47.48%，出现两次 early stop；但不是持续 early stop。其余组没有 Macro early stop，所有组 Micro 均未 early stop。平均 KL/Clip/Entropy 是聚合诊断，不能单独当作收敛证明；Macro Entropy 还受 Cache 候选数和 Holding 决策分布影响。配置熵系数为0，Micro 另有 STAY/MOVE 组熵下限项。

梯度范数裁剪上限为0.5。5e-4 下 Macro Actor 裁剪前范数均值/P95/最大值为0.132/0.212/0.265，Micro Actor 为0.174/0.217/0.231；这两类 Actor 的典型范数低于上限。Critic 的梯度则经常触发裁剪：Macro Critic 的裁剪事件比例均值为99.92%，裁剪前范数均值30.46，裁剪后约0.5；Micro Critic 裁剪前范数均值338.29，裁剪后约0.5。Long-Value Critic 代码同样使用0.5范数上限，但日志未记录其逐步裁剪比例。高 Critic 梯度裁剪说明梯度规模受控，不单独证明 Critic 是性能瓶颈。

## Critic Explained Variance 与训练目标误差

下列为75个 Policy Update 的均值。RMSE 以相应 value target 的原始 Return/Reward 单位计；Macro、Micro、Long-Value 的目标和尺度不同，RMSE 不能跨 head 直接比较。

| Macro LR | Macro post-EV / RMSE | Micro EV / RMSE | Long-Value post-EV / RMSE |
|---:|---:|---:|---:|
| 1e-4 | 0.861 / 39.14 | 0.861 / 73.05 | 0.498 / 536.89 |
| 5e-4 | 0.911 / 28.86 | 0.941 / 59.33 | 0.729 / 333.25 |
| 1e-3 | 0.869 / 41.07 | 0.941 / 60.55 | 0.844 / 226.44 |
| 5e-3 | 0.908 / 36.51 | 0.935 / 61.43 | 0.835 / 234.18 |

5e-4 Macro Critic 平均 explained variance 从更新前0.844升至更新后0.911；Long-Value Critic 从更新前0.415升至更新后0.729。后期更新的 Macro/Micro EV 通常更高，但 Long-Value 目标误差仍明显，且跨更新波动较大。Macro Critic 的两个 SMDP step target、Micro episode return target、Long-Value realized return-to-go 是不同监督目标。现有日志显示 Critic 能拟合相当一部分目标方差，但尚未证明价值误差没有限制 Actor，也没有显示普遍数值发散。

## 采样耗时与参数更新耗时

计时按当前源码定义：单 Episode collection timer 从加载该 realization 的 environment context 前开始，到完整 rollout 收集后结束，包含环境执行、策略推理、Matching/KKT 及该段的数据加载开销；policy_update_runtime_seconds 单独包住四条轨迹后的 PPO 更新。因此 collection 是“采样/执行路径”时间，不是纯物理仿真时间。

| Macro LR | 平均每个4-Episode组 collection 时间 | 平均 PPO update 时间 | Update P95 | Group collection 占记录总训练时间 |
|---:|---:|---:|---:|---:|
| 1e-4 | 1351.8 s | 8.80 s | 12.69 s | 99.35% |
| 5e-4 | 1575.5 s | 9.60 s | 11.78 s | 99.39% |
| 1e-3 | 1042.3 s | 6.54 s | 7.87 s | 99.37% |
| 5e-3 | 1430.8 s | 70.71 s | 162.32 s | 95.29% |

四组旧环境 run 中，采样/执行路径的累计时间均远高于参数更新时间，因此按这些记录的壁钟时间，环境交互/rollout collection 是主要耗时段。5e-3 的参数更新明显更慢且长尾更大，但其总 collection 仍占记录训练时间约95%。不同训练进程可能有机器负载、候选/KKT数量和日志调度差异；本数据不能将不同 LR 的耗时差因果归于 LR，也不能把 collection 全部归因于 Matching/KKT。

## 三种“学习速度”及瓶颈判断

- Episode 样本效率：5e-3 和1e-3早期 Episode Return 上升更快；1e-4明显落后；5e-4在中后期和旧环境 Validation 更好。该排序受策略轨迹和环境 realization 影响。
- Policy Update 学习速度：四组都是每4 Episode一次、75次更新，Micro/Macro步数相同（只有5e-3两次 Macro KL stop）。LR改变了每次策略更新幅度；没有按每 Episode 更新、每2/8 Episode更新或不同 minibatch预算的对照，不能判定现有更新频率限制学习。
- 实际运行时间效率：旧 run 的 rollout collection 占记录总时间95.3–99.4%。这支持“旧环境训练壁钟时间主要耗在采样/执行路径”，但不是 Stage3B 的运行效率结论，也不能区分数据读取、策略前向与 Matching/KKT 各自的耗时。

综合判断：有证据说明学习率会改变旧环境早期回报与 KL/Clip 更新幅度，但没有证据证明学习率是主要瓶颈。旧环境 5e-4 的最终验证结果不能直接作为 Stage3B 新环境最优学习率。更新频率问题没有因果对照。rollout collection 的壁钟占比则在旧训练日志中明确占主导。

~~~ini
PPO_CONVERGENCE_ANALYSIS_COMPLETE = YES
LEARNING_RATE_MAIN_BOTTLENECK_SUPPORTED = UNDETERMINED
POLICY_UPDATE_FREQUENCY_ISSUE_SUPPORTED = UNDETERMINED
ENVIRONMENT_RUNTIME_BOTTLENECK_SUPPORTED = YES
CODE_CHANGED = NO
TRAINING_EXECUTED = NO
~~~

## 主要数据与代码依据

- 训练结果：runs/formal_cache_economic_v2_lr_{1e_minus_4,5e_minus_4,1e_minus_3,5e_minus_3}/episode_metrics.jsonl、update_metrics.jsonl、run_manifest.json、run_summary.json。
- 已有旧环境 Validation 汇总：reports/four_macro_lr_validation_comparison.md；训练回报汇总：reports/four_lr_training_return_comparison.md。
- 更新组大小和 frozen collection：src/ppo_simple/trainer.py::PPOTrainer.__init__()、update_from_episodes()、run_training()。
- Micro PPO 更新：trainer.py::_update_micro()；Macro PPO 更新：_update_macro()；Macro/Long-Value Critic 更新：_update_macro_critic_only()、_update_long_value_critic()。
- collection 与 PPO 更新独立计时：trainer.py::run_training() 中 collection_started_at、update_started_at、group_collection_runtime_seconds、policy_update_runtime_seconds。

