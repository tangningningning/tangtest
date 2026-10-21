# 第三阶段：预测模型与决策输入检查

## 范围与方法

只读取Training Episode `20001`、`20150`、`20300` 的既有正式预测/任务/轨迹文件和当前正式代码。位置预测的真值列在 `vehicle_predictions.csv` 中为空，因此用 `(target_time_s, vehicle_id)` 与同Episode轨迹作精确连接；只统计 `target_slot < 300` 的正式Episode目标。三个Episode在每个有效目标上均有完整的25车位置匹配。没有读取Validation/Test，没有生成数据、训练模型、运行PPO或Matching/KKT。运行环境为 `C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`。

流行度基线采用同一信息截止时刻的上一时刻观测流行度：origin `t` 使用 `rho_obs` at `t-1`，origin 0 使用 `popularity_history.csv` 的formal slot -1；同一向量用于该Origin的全部Horizon。误差和Top-3仅在正式目标slot 0–299上计算。

## A. 车辆位置预测

| Horizon | 有效车位置样本 | 误差P50 (m) | P90 (m) | P95 (m) | 最近SV判断不同 |
|---:|---:|---:|---:|---:|---:|
| 0 | 22,500 | 6.25 | 14.43 | 17.25 | 3.63% |
| 1 | 22,425 | 6.91 | 16.51 | 19.90 | 4.12% |
| 2 | 22,350 | 8.13 | 20.04 | 24.81 | 5.11% |
| 3 | 22,275 | 10.32 | 26.22 | 32.05 | 6.65% |
| 4 | 22,200 | 13.43 | 35.87 | 43.98 | 8.82% |

三个Episode的形状一致。Horizon 0 / 4 的P95分别为：Episode 20001：16.90 / 44.26 m；20150：17.18 / 44.08 m；20300：17.80 / 43.59 m。按origin的0–99、100–199、200–299三段汇总全部有效Horizon，位置误差P95为30.75、29.95、29.45 m；这个混合Horizon分段统计受尾部有效Horizon较少影响。固定Horizon查看各Episode/时间段，Horizon 0的P95范围为16.17–18.62 m，Horizon 4为42.17–47.02 m，没有某一时间段持续偏高。

最近SV判断按TV的五个SV欧氏距离最小值比较预测点和真实轨迹。Horizon 4仍有91.18%一致；真实最近与第二近SV距离差的中位数约119 m。误差随Horizon增长，确会改变少数TV的最近SV判断，但在本样本中没有普遍翻转几何归属。预测适合作为短期候选/状态输入，不等价于精确链路位置。

三个Episode均有25个预测车辆ID，ID集合与场景NPZ一致；车辆预测行无重复，`target_slot=origin_slot+horizon`、`observation_cutoff_slot=origin_slot-1`全部成立。有效目标位置与轨迹匹配率为100%。位置场景20样本的坐标均值与CSV点预测的差异中位数约2.1 m、P95约5.8 m。

## B. 流行度预测

| Horizon | GRU MAE | 上一时刻保持 MAE | GRU Top-3重合数/3 | 保持基线Top-3重合数/3 |
|---:|---:|---:|---:|---:|
| 0 | 0.02044 | 0.06097 | 2.397 | 1.864 |
| 1 | 0.02117 | 0.06095 | 2.366 | 1.863 |
| 2 | 0.02125 | 0.06098 | 2.385 | 1.864 |
| 3 | 0.02167 | 0.06111 | 2.351 | 1.863 |
| 4 | 0.02189 | 0.06123 | 2.338 | 1.860 |

GRU绝对误差P50/P90/P95随Horizon由 `[0.0146, 0.0428, 0.0541]` 增至 `[0.0154, 0.0466, 0.0585]`。在真实概率变化边界前、边界当Slot及边界后各一Slot的6,300条服务概率记录中，GRU MAE为0.02181，保持基线为0.06172；Top-3平均重合分别为2.354/3和1.806/3。GRU在五个Horizon的MAE与Top-3重合都优于该基线。

每个Episode的14次20-Slot概率分布边界均有 `rho_star` 变化；相邻Epoch平均分布的Total Variation中位数在三个Episode分别为0.0195、0.0128、0.0078。Top-3集合只在Episode 20001的3个、20150的1个、20300的1个边界发生变化。因而需求概率持续变化，但热门服务身份大多稳定；不能据此断言缓存UPDATE会有正净收益。

## C. 预测场景与采样合同

每个NPZ包含300个Origin、20个场景、5个Horizon、25车位置、20个TV任务槽和10维 `rho_hat`。三个Episode的 `rho_hat` 与 `popularity_predictions.csv` 的 `rho_pred` 最大差小于 `1e-16`；服务ID为0–9。数据元信息写明任务从 `rho_hat` 抽样、未保存未来真值；车辆残差块来自独立Calibration Artifact。正式车辆预测文件中的 `x_true/y_true/s_true` 全为空，流行度预测读入器只取 `rho_pred`，不读取同表的诊断标签 `rho_star`。

每Episode共有600,000个预测TV-Slot样本。观测到达率为0.699370、0.700165、0.700205，与合同概率0.7一致；`arrival=0` 时服务ID为-1且输入/周期为0。有效任务的输入扰动因子实测范围为0.8000–1.2000，计算负载因子为0.9000–1.1000。按预测 `rho_hat` 计算的服务样本份额与观测份额最大绝对差分别为0.00066、0.00085、0.00069。没有发现预测任务采样偏离当前到达/服务/工作量合同或使用未来真实任务的迹象。

场景车辆位置由因果点预测加独立残差块抽样形成；预测任务按 `lambda=0.7`、`rho_hat` 和输入/计算扰动抽样。位置真实轨迹用于离线几何/相对距离诊断；决策读入位置预测文件和场景NPZ中的预测样本，而非未来真值。

## D. 正式决策输入

当前`ConfigurationObservationSpec`为400维，实际加载的切片为：

- Cache状态：0–89（9节点×10服务）；
- UAV位置：90–93（2×2）；
- 25车×5 Horizon位置：94–343（250）；
- 5×10流行度概率：344–393（50）；
- Horizon Mask：394–398（5）；
- 剩余时间：399（1）。

抽查Origin 0、150、296、299，观察向量均为400维且有限；Cache状态、ID顺序和流行度向量对齐。尾部Mask与终点299一致：Origin 296为`11110`、297为`11100`、298为`11000`、299为`10000`。非法Horizon的Observation位置和流行度字段置零。Rollout对预测车辆/服务ID、target slot/time进行严格对齐检查。

Cache候选路径从同一`step.scenario_slice`取预测任务，Cache候选生成与经济特征均传入Horizon Mask；节点需求服务性权重使用同一Origin各Horizon的预测位置样本均值。抽查Episode 20001的Origin 150和299，经济特征有限：加载/释放容量比例约0–1，`log_cache_cost`最大0.225；按每TV归一化的新增需求特征最大约0.04，没有看到明显数值爆炸或NaN/Inf。抽查时各节点保留10–56个UPDATE（上限64）。这只是初始Cache的少量决策点尺度检查，不是所有状态的分布审计。

**确认到一项正式输入不一致：** `_uav_candidates` 调用 `predicted_vehicle_service_demand` 时没有传入`step.prediction_horizon_mask`。相比Cache候选，UAV候选因此把Episode终点以后的预测任务也纳入需求权重。Origin 296、297、298、299中，无Mask需求总量分别有17.96%、36.17%、57.69%、77.44%来自无效Horizon。数据仍是因果预测抽样，不是真实未来任务，但这些场景不属于Episode内可执行时域。另一个表示差异是UAV需求按多个Horizon汇总，而需求质心位置使用单一Horizon-0点预测。

此外，预测场景有`input_size`和`cpu_cycles`，但当前Rollout的Cache经济候选特征与UAV需求特征仅消费到达、服务ID、位置/流行度；`node_local_predicted_service_workload`目前无正式调用。因此工作量字段生成正确，却没有进入当前Cache/UAV候选输入。它可能使不同输入大小/计算负载的服务在候选需求代理中仅按请求数计权；是否应改为工作量感知属于后续算法设计判断，不是生成错误。

## 代码路径

- 位置因果cutoff与空真值列：`src/scenario/scenario_generator.py:280-346`；独立残差库校验与场景采样：`src/scenario/scenario_generator.py:109-153, 355-465`。
- 任务生成：`src/scenario/scenario_generator.py:593-659`；在线边界元数据：`src/scenario/scenario_generator.py:1139-1174`。
- 流行度因果生成/合同校验：`src/scenario/popularity_prediction.py:370-440, 522-526`；正式读取只取`rho_pred`：`src/ppo_simple/interface.py:1301-1345`。
- Observation结构与Mask：`src/ppo_simple/interface.py:420-542, 654-682, 1012-1192`；预测ID、target slot/time校验：`src/ppo_simple/interface.py:1358-1665`。
- Cache预测需求/服务性：`src/decision/cache_configuration.py:587-750`；候选特征：`src/ppo_simple/primitive_control.py:191-315`；Rollout Cache候选传入Mask：`src/ppo_simple/rollout.py:560-681`。
- UAV候选与需求聚合：`src/ppo_simple/rollout.py:684-727`、`src/decision/uav_configuration.py:298-380`。这里未传Horizon Mask。

## 结论

```ini
VEHICLE_PREDICTION_ALIGNMENT = YES
VEHICLE_PREDICTION_QUALITY = ACCEPTABLE

POPULARITY_PREDICTION_ALIGNMENT = YES
GRU_BETTER_THAN_PERSISTENCE = YES

PREDICTION_SCENARIO_CONSISTENT = YES
DECISION_INPUT_CONSISTENT = NO
FUTURE_INFORMATION_LEAKAGE = NO

PREDICTION_REPAIR_REQUIRED = YES

SOURCE_CHANGED = NO
CONFIG_CHANGED = NO
FORMAL_DATA_CHANGED = NO
PPO_TRAINING_EXECUTED = NO
```

最小后续修正方向是将正式Horizon Mask贯穿UAV需求汇总/候选生成，使Episode尾部只使用仍在正式决策时域内的预测。工作量预测是否加入Cache/UAV候选特征需单独确定其建模语义和归一化，不应与本次Mask问题混为一项已证实实现错误。
