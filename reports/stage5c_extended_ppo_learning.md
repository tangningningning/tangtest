# Stage5C：72 Episode最终Validation及学习效果分析

日期：2026-10-12（Asia/Shanghai）

## 1. 训练完成性与环境

Stage5C的`run_manifest.json`和`run_summary.json`一致记录：72/72个完整Episode、18/18次PPO更新、21,600次Slot环境执行；Episode seed为20001–20072，策略初始化seed为0，`resume=false`，`early_stop_reason=null`。最终Checkpoint存在，SHA-256为`15a0f92ca8a5e1a8ae62825af0eda2b7dfb3ec5ce960d8a59064bc2754990800`。训练日志最后记录`Epoch 72/72`，Trainer退出码为0。

训练使用冻结Stage3B配置：`configs/proposed/stage3_cloud_finite_60ghz.toml`，原始SHA-256=`dd3f0ae104109f264758ce47a2f478d5df13655638497fe576c1874424f02ba0`，语义SHA-256=`d69c6416adb3d1356ea0f233d963ba4f25d66c2eb0fd733798bd1684ef327d75`。错误日志只有绘图阶段“没有带标签的图例项”警告，没有训练异常或早停。

本阶段Evaluator加载的是Stage5A已修复的合同读取版本（`src/ppo_simple/evaluator.py` SHA-256=`60ce71d14a12275b9c4b535cb4115242e636689a38cab7dd5e77fb904dc79d10`）。本阶段没有修改配置、PPO、Reward或系统模型。

## 2. 最终策略Validation与配对口径

只对Stage5C最终Checkpoint运行了4个Validation Episode；没有重跑Stage5A的8个Episode。命令参数如下：

```powershell
conda activate adaptive_cache_mec
$env:PYTHONPATH = "src"
python -m ppo_simple.evaluator `
  --config configs/proposed/stage3_cloud_finite_60ghz.toml `
  --checkpoint runs/stage5c_extended_pilot_stage3b_seed0_72ep/trainer/checkpoints/latest_checkpoint.pt `
  --run-dir runs/stage5c_extended_pilot_stage3b_seed0_72ep/validation_final `
  --episode-pool environment_data/proposed_joint_validation `
  --seed-start 30001 --episodes 4 --repeats 1 `
  --device cpu --policy-action-mode deterministic --policy-sampling-seed 0
```

Evaluator退出码为0。Stage5C和Stage5A最终策略结果的配置原始/语义SHA、Validation数据池、seed（30001–30004）、重复次数、确定性动作模式及采样seed均一致；每个seed两边任务数完全相同。设备通过本次命令明确指定为CPU。Stage5A已保存的最终策略结果直接复用。这里只比较24 Episode最终策略与72 Episode最终策略，没有把Stage5A初始Checkpoint当作Stage5C的初始策略，也没有假设两个Run的初始化权重相同。

结果位于`runs/stage5c_extended_pilot_stage3b_seed0_72ep/validation_final/`。Return恒等式为`Return = Task Utility − Cache Cost − UAV Movement Cost`；Stage5C四条结果的最大残差为`2.3e-12`。

## 3. 同seed净效用比较

金额/效用以模型原始单位记录；`Δ`均为72 Episode结果减24 Episode结果。

| Seed | Return 24ep | Return 72ep | Δ Return | Δ Task Utility | Cache Cost 24→72 | UAV Cost 24→72 | 任务数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30001 | 3982.591733 | 3987.444290 | +4.852558 | +5.104238 | 0→0 | 0→0.251680 | 4302/4302 |
| 30002 | 3782.982705 | 3789.588362 | +6.605657 | +6.877357 | 0→0 | 0→0.271700 | 4196/4196 |
| 30003 | 3893.167095 | 3900.140113 | +6.973018 | +7.241616 | 0→0 | 0→0.268598 | 4255/4255 |
| 30004 | 3963.190108 | 3966.561214 | +3.371107 | +3.642756 | 0→0 | 0→0.271650 | 4225/4225 |
| 平均 | 3905.482910 | 3910.933495 | **+5.450585** | **+5.716492** | 0→0 | 0→0.265907 | — |

四个seed的ΔReturn均为正；平均增幅约为24 Episode平均Return的0.140%。提升来自Task Utility平均增加5.716492，扣除新增UAV移动成本0.265907后得到Return平均增加5.450585。Cache成本没有变化，不能把净收益改善归因于Cache成本下降。

## 4. Cache、Candidate、Holding与UAV策略

| Seed | Cache KEEP/UPDATE（24→72） | UAV0 MOVE（24→72） | UAV1 MOVE（24→72） | Holding均值（24→72） |
|---|---:|---:|---:|---:|
| 30001 | 100%/0% → 100%/0% | 0% → 1.000% | 0% → 28.333% | 4.918 → 5.000 Slot |
| 30002 | 100%/0% → 100%/0% | 0% → 8.000% | 0% → 23.667% | 4.918 → 5.000 Slot |
| 30003 | 100%/0% → 100%/0% | 0% → 0.333% | 0% → 31.000% | 5.000 → 5.000 Slot |
| 30004 | 100%/0% → 100%/0% | 0% → 2.000% | 0% → 29.667% | 4.918 → 5.000 Slot |

72 Episode确定性Validation仍全部选择Cache KEEP，没有实际UPDATE，Cache成本为0；因此没有观察到Cache UPDATE长期经济价值被最终确定性策略采用。训练期间随机采样的Cache UPDATE比例并非为0，但从前4个Episode均值46.81%降至最后4个Episode的32.85%。UPDATE条件下候选选择熵从4.263降至3.811，top-1比例从2.04%升至2.75%，每Episode平均不同候选数从71.5降至45.5（每个状态最多351个合法UPDATE候选）。这显示采样分布有所收窄，但仍分散；确定性验证没有执行UPDATE，所以不足以证明候选按经济价值正确排序。

UAV策略出现可观察变化：Stage5A最终策略四个seed均STAY；72 Episode策略的UAV0 MOVE率为0.33%–8.00%，UAV1为23.67%–31.00%。在Cache动作保持不变的配对结果中，四个seed的Task Utility均增加，同时每Episode增加约0.252–0.272的移动成本，净Return仍全部增加。这构成初步的UAV移动净收益信号；由于没有另做冻结UAV动作的因果消融，不能把全部Utility增量确认为移动动作的单独因果效果。

训练采样中的平均Holding时长从前4个Episode的1.951 Slot变为最后4个的2.160 Slot。最终确定性Validation中四个Episode都选择继续Holding至5 Slot上限，Macro frame数均为60。Stage5A最终策略也已接近5 Slot，故本次没有看到Holding时长的实质性改变。

## 5. 训练曲线与PPO数值诊断

下表每行是连续4个训练Episode的均值；这些是训练采样结果，不是独立Validation结果。Return有波动，不呈单调上升。

| Episode | Return | Task Utility | Cache Cost | UAV Cost | Cache UPDATE采样率 | UAV0/UAV1 MOVE采样率 |
|---|---:|---:|---:|---:|---:|---:|
| 1–4 | 3800.78 | 3813.80 | 12.37 | 0.650 | 46.81% | 44.42% / 47.08% |
| 5–8 | 3858.83 | 3872.63 | 13.11 | 0.695 | 50.00% | 47.17% / 50.67% |
| 9–12 | 3861.46 | 3875.96 | 13.78 | 0.715 | 49.39% | 49.00% / 50.42% |
| 13–16 | 3834.78 | 3847.27 | 11.84 | 0.648 | 45.52% | 45.00% / 46.75% |
| 17–20 | 3849.05 | 3861.43 | 11.75 | 0.629 | 45.61% | 41.83% / 48.17% |
| 21–24 | 3869.98 | 3882.01 | 11.45 | 0.591 | 48.80% | 39.67% / 44.25% |
| 25–28 | 3818.01 | 3827.93 | 9.27 | 0.647 | 42.11% | 43.50% / 47.67% |
| 29–32 | 3827.24 | 3837.50 | 9.56 | 0.695 | 39.11% | 45.25% / 51.83% |
| 33–36 | 3864.13 | 3872.51 | 7.72 | 0.660 | 36.70% | 43.92% / 49.92% |
| 37–40 | 3828.80 | 3839.88 | 10.45 | 0.630 | 43.71% | 41.58% / 47.50% |
| 41–44 | 3830.34 | 3840.64 | 9.62 | 0.681 | 43.82% | 46.50% / 49.92% |
| 45–48 | 3826.31 | 3835.55 | 8.53 | 0.712 | 42.08% | 48.17% / 52.00% |
| 49–52 | 3861.67 | 3869.94 | 7.50 | 0.768 | 39.15% | 52.92% / 56.00% |
| 53–56 | 3811.85 | 3821.36 | 8.77 | 0.743 | 41.63% | 50.00% / 53.67% |
| 57–60 | 3808.04 | 3815.64 | 6.85 | 0.758 | 35.24% | 54.58% / 51.33% |
| 61–64 | 3840.60 | 3849.77 | 8.43 | 0.735 | 35.71% | 50.17% / 53.00% |
| 65–68 | 3781.32 | 3790.29 | 8.26 | 0.711 | 38.51% | 49.50% / 50.50% |
| 69–72 | 3848.68 | 3856.75 | 7.39 | 0.689 | 32.85% | 44.58% / 51.25% |

第1–4 Episode与第69–72 Episode的训练均值Return相差+47.89，但期间明显波动，不能据此认定收敛。18次更新的学习指标汇总如下：

| 指标 | Macro | Micro | Long-Value / Critic |
|---|---:|---:|---:|
| Actor KL均值（范围） | 0.00382（0.00025–0.01431） | 0.00290（0.00021–0.00667） | — |
| Clip Fraction均值（范围） | 0.0402（0–0.1826） | 0.0243（0–0.1029） | — |
| Entropy均值（范围） | 4.5206（3.8452–4.9145） | 4.1956（3.7814–4.4669） | — |
| Critic EV均值；最后一次 | 0.6797；0.9441 | 0.8985；0.9632 | 0.7475；0.8736 |

18条更新记录中的上述关键数值均为有限值；Macro/Micro没有KL触发的提前停止。Critic EV整体在训练过程中提升。日志未显示数值故障或明显PPO更新失败；这不代表策略已收敛。

## 6. 结论及证据边界

1. **24→72 Episode最终策略的同seed Validation Return改善。** 四个seed均为正差值，平均+5.450585，约+0.140%。样本只有4个seed且来自同一训练初始化，证据支持“这组配对状态有小幅改善”，不代表广泛统计结论。
2. **提升来自Task Utility增加，不是Cache成本下降。** 平均Task Utility增加5.716492，平均UAV移动成本增加0.265907，Cache成本保持0。
3. **Cache长期经济学习尚无确定性采用证据。** 训练随机采样仍有UPDATE，但72 Episode最终策略在这4个确定性Episode中仍全部KEEP；候选分布稍收窄不足以证明其学会经济价值排序。
4. **UAV有初步正向经济信号。** 最终策略增加了UAV移动，四个seed的Utility提升均超过对应移动成本；有限配对样本和缺少动作消融限制因果解释。
5. **没有发现要求修改PPO学习机制的证据。** 训练与Checkpoint完整，关键更新指标数值有效，Validation运行成功。当前结果不支持为提高UPDATE/MOVE频率而修改算法。
6. **可进入正式300 Episode确认性训练，但不应视为已证明收敛。** 正式训练仍须保持Stage3B冻结环境并使用独立Validation；本报告没有启动该训练。

```makefile
STAGE5C_TRAINING_COMPLETE = YES
EPISODES_COMPLETED = 72
PPO_UPDATES_COMPLETED = 18
VALIDATION_COMPLETED = YES; CPU_DETERMINISTIC; SEEDS_30001_30004
VALIDATION_RETURN_72EP = 3910.933495
VALIDATION_DELTA_VS_24EP = +5.450585; POSITIVE_SEEDS=4_OF_4
TASK_UTILITY_DELTA = +5.716492
CACHE_COST_DELTA = 0
UAV_COST_DELTA = +0.265907
CACHE_ECONOMIC_LEARNING_EVIDENCE = NO_DETERMINISTIC_UPDATE_EVIDENCE
UAV_ECONOMIC_LEARNING_EVIDENCE = PRELIMINARY_POSITIVE; FOUR_PAIRED_SEEDS
PPO_IMPLEMENTATION_ISSUE = NO_CLEAR_ISSUE_FOUND
READY_FOR_FORMAL_300EP_TRAINING = YES; CONFIRMATORY_ONLY
NEXT_ACTION_RECOMMENDATION = FORMAL_300EP_CONFIRMATORY_RUN; NO_AUTOMATIC_LAUNCH
```
