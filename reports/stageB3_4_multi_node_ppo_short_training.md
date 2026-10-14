# Stage B3-4：9节点联合Cache PPO短程训练验证

## 结果状态

本次完成了4个Training Episode的真实Rollout和1个正式PPO策略更新组，随后Runner在更新后Checkpoint写入前因本机绘图依赖错误退出。因此确认了联合策略可以执行Rollout和一次有限优化，但没有保存训练后策略，也无法完成训练后固定候选探针；本报告状态为 **PARTIAL**。

## 运行配置与训练器对应关系

- Conda环境激活失败，使用已有 `C:\software\anaconda3\python.exe`（Python 3.12.7）。NumPy为2.2.6，Numba导入仅在本次进程内屏蔽，使用项目纯Python KKT fallback；没有安装或升级依赖。
- 使用独立配置 `configs/proposed/system_config_multi_node_factorized.toml`，SHA-256为 `a86d87ca5d586e89e367aee393b7a1cd9b27813f41ac3c5a19005bfa40a9ea6c`。配置快照保存在本次隔离Run目录。
- Runner的 `--epochs` 参数实际是完整Episode数量。配置规定每4个同一冻结策略收集的Episode执行一次PPO策略更新；Macro Actor内部每次更新含4个配置更新Epoch。本次使用Episode 20010–20013，各300个Slot，收集4集后完成1次策略更新组，Macro Actor的4个内部Epoch均执行。
- 预算最多6集，但剩余2集不足以组成下一次正式更新组。没有缩小每组样本数或改变任何PPO参数，也没有继续收集无更新的Episode。

## Rollout与系统表现

共执行1,200个Slot、1,200次正式Matching、132,035次KKT评价。Rollout组耗时321.90秒，PPO更新耗时9.92秒，总计331.83秒。各Episode使用不同Training任务，所以Return之间的差异不能视为策略逐集改善；策略是在四集采样全部结束后才更新。

| Training Episode | 任务数 | Return | Task Utility | Cache Cost | UAV移动成本 | 加权平均时延(s) | 车辆能耗(J) | Edge任务数 | Cloud任务数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20010 | 4,187 | 3,886.840 | 3,977.373 | 89.839 | 0.694 | 2.4054 | 94.436 | 846 | 3,341 |
| 20011 | 4,251 | 3,948.546 | 4,032.397 | 83.151 | 0.699 | 2.4660 | 101.469 | 784 | 3,467 |
| 20012 | 4,216 | 3,950.980 | 4,031.533 | 79.856 | 0.696 | 2.0946 | 87.877 | 764 | 3,452 |
| 20013 | 4,143 | 3,826.135 | 3,915.944 | 89.058 | 0.751 | 2.6296 | 104.449 | 780 | 3,363 |

四集合计Task Utility为15,957.246、Cache Cost为341.904、UAV移动成本为2.841。Cache成本没有主导总效用数值，但随机初始化策略的缓存变化频率很高：711个Macro决策帧全部至少改变一个节点，平均每帧改变4.571个节点。Episode级Cache成本为79.856–89.839；四集间没有单调变化，且任务难度不同，不能把这些成本或Return差异解释为学习趋势。

所有动作均通过正式合法候选、Cache矩阵应用、Matching/KKT和Reward路径执行；没有出现Runner记录的NaN/Inf、非法Cache或资源求解中止。详细逐Slot结果由正式 `slot_metrics.jsonl` 保留。

### 多节点动作与成本

| 每个Macro实际更新节点数 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 711帧计数 | 0 | 16 | 51 | 104 | 167 | 181 | 119 | 53 | 18 | 2 |

各节点的KEEP概率约为0.495–0.503，实际UPDATE频率约为0.435–0.563。由于是随机初始化策略并使用逐节点分布，平均更新节点数较高；这说明短程初始行为偏更新频繁，不能解释为策略已经学会偏好UPDATE。四集按节点角色汇总Cache成本为SV 156.225、RSU 72.658、UAV 113.021。每个Episode的动态SV ID、节点索引、KEEP/UPDATE概率、实际更新频率、候选熵和节点成本都保存在诊断JSON及 `episode_metrics.jsonl` 中。

逐Macro的实际候选索引没有被Runner默认日志记录：本次未启用Macro transition trace，逐Slot日志仅包含更新节点数、成本和任务指标。因此候选索引及对应缓存行变化不能从已保存文件精确恢复；没有为补齐该序列重新执行Rollout。

## PPO更新与数值诊断

- 完成1次联合策略更新。Macro Actor完成4个被KL保护接受的Optimizer Step；Macro Critic 50步、Long-Value Critic 50步、Micro Actor和Critic各40步。Macro Actor更新前梯度范数为0.1236，Micro Actor为0.1833，均为有限且非零。Cache Actor与共享Observation Encoder均属于Macro Actor优化器参数；Cache KL更新后非零（0.006114），表明缓存分布有变化。由于训练后模型状态没有保存，无法直接比较这两个模块的权重张量。
- Macro Advantage均值251.696、标准差44.983，Critic目标有限。N2解释方差从更新前约0.000025到更新后0.0332；单次更新不足以判断Critic质量是否稳定。
- 行为策略Log-Probability重算最大误差为 `3.81e-6`，联合概率分解最大误差为 `1.67e-6`。联合KL有限：更新前0.005734、更新后0.007388；联合Clip Fraction为4.18%，Cache Ratio Clip Fraction为0.39%。四个Actor Step均按原学习率5e-4接受，没有候选被拒绝，也没有达到配置的Joint KL停止阈值0.15。没有观察到联合Ratio溢出或异常裁剪。
- Micro PPO更新40步，KL为0.005261、Clip Fraction为4.83%，未触发提前停止。
- Trainer保存各节点熵，但当前实现没有输出逐节点KL；报告使用联合/Cache汇总KL，不将节点Log-Probability差值冒充KL。节点熵按固定动作顺序（5个SV、2个RSU、2个UAV）计算，分别约为 `[1.821, 1.818, 1.816, 1.815, 1.813, 2.643, 2.639, 2.403, 2.399]`，这是9个节点的熵向量，不能直接与单节点熵比较。

## 固定B2-8候选探针

训练前，使用随机初始化的同一策略权重、Episode 20007–20009 Slot 0和正式候选接口完成探针；候选按节点ID与完整目标缓存行匹配。各集全部9节点KEEP联合概率约为0.00185。目标A/B在指定节点“候选概率条件于UPDATE组”及其UPDATE组内排名如下：

| Episode | 候选 | 节点 | UPDATE条件概率 | UPDATE组排名 | 节点KEEP概率 | 该节点选候选、其余8节点KEEP的联合概率 | 确定性节点动作是否命中 |
|---:|---|---|---:|---:|---:|---:|---|
| 20007 | A | UAV_0 | 0.02937 | 19/34 | 0.49787 | 0.00005487 | 否 |
| 20007 | B | RSU_1 | 0.01779 | 33/56 | 0.49996 | 0.00003297 | 否 |
| 20008 | A | SV ID 23 | 0.10074 | 2/10 | 0.48895 | 0.00019492 | 否 |
| 20008 | B | RSU_1 | 0.01779 | 33/56 | 0.49995 | 0.00003295 | 否 |
| 20009 | A | SV ID 17 | 0.10013 | 5/10 | 0.50168 | 0.00018419 | 否 |
| 20009 | B | RSU_1 | 0.01779 | 33/56 | 0.49994 | 0.00003296 | 否 |

单节点条件候选概率不能与9节点完整联合动作概率混用。训练前目标行A/B都不是对应节点的确定性动作。由于该探针只测随机初始化策略，且训练后权重未保存，不能据此回答学习后是否能识别B2-8的收益排序；该问题为 **UNDETERMINED**。B2-8的100槽固定Cache收益仅作诊断参照，本次没有将其用作训练标签。

## Checkpoint阻断与后续条件

正式Runner已将1次策略更新指标、Episode摘要和训练计数写入隔离目录；随后在 `update_training_epoch_return_plot()` 导入Matplotlib时失败。错误是本机Matplotlib扩展针对NumPy 1.x编译，而当前Python使用NumPy 2.2.6。异常发生在更新后Checkpoint保存之前，因此进程内训练后的模型状态丢失，训练后探针无法运行。该问题不是Matching/KKT或PPO数值错误；没有安装或升级依赖。

新架构33、Checkpoint版本60的训练前随机初始化Checkpoint保存并通过正式Loader重载，Schema检查通过，但其中训练计数仍为0。为避免把它误认为训练后模型，文件保留为 `initial_random_checkpoint_before_training.pt`。训练后Checkpoint保存状态为FAIL，不能将该初始化文件作为本轮最终策略。

没有继续执行Epoch：当前Runner要求每4个完整Training Episode更新一次，最多6集只允许一个完整更新组；本次门控组在PPO更新后通过数值和Optimizer Step检查，但Checkpoint验收失败。因此剩余Episode 20014、20015不足以构成下一组，也没有继续长训。下一步需先在不改变PPO算法和依赖的情况下，让隔离诊断Runner在更新后Checkpoint保存不受绘图导入阻断，再重新执行受限验证；当前不建议进入更长训练。

## 最终字段

```ini
STAGE_B3_4 = PARTIAL
ARCHITECTURE_VERSION = 33
CHECKPOINT_VERSION = 60
TRAINING_EPISODES_USED = 20010, 20011, 20012, 20013
PPO_EPOCHS_COMPLETED = 1 policy-update group; Macro internal update_epochs=4/4
MACRO_OPTIMIZER_STEPS = Actor 4 accepted; Critic 50; Long-Value 50; Micro Actor/Critic 40 each
CACHE_SCORER_UPDATED = YES (Cache distribution KL changed; direct tensor diff unavailable)
ENCODER_UPDATED = YES (included in accepted Macro Actor steps; direct tensor diff unavailable)
JOINT_RATIO_FINITE = YES
JOINT_KL_STATUS = FINITE; pre=0.005734; post=0.007388; four steps accepted
JOINT_CLIP_STATUS = OK; joint clip fraction=0.04184; Cache ratio clip fraction=0.00387
MEAN_CACHE_UPDATES_PER_MACRO = 4.571
CACHE_COST_TREND = 79.856–89.839 per Episode; no monotonic trend
RETURN_TREND = 3826.135–3950.980 across different Training Episodes; not a learning trend
CHECKPOINT_RELOAD = FAIL for trained checkpoint; initial random checkpoint reload PASS (counters=0)
CANDIDATE_VALUE_RECOGNITION = UNDETERMINED
READY_FOR_LONGER_TRAINING = NO
FORMAL_TRAINING_MODIFIED = NO
SYSTEM_MODEL_CHANGED = NO
```
