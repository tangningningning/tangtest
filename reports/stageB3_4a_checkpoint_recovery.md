# Stage B3-4A：Checkpoint保存恢复与验证

## 结果

本轮完成一次真实的多节点PPO更新组。Matplotlib/NumPy导入异常只跳过可选绘图，更新后的正式Checkpoint在绘图前保存；随后使用正式Loader成功重载，并验证训练计数、优化器状态、模型参数和固定候选探针。

## 根因与修复

上次阻断发生在 `src/ppo_simple/trainer.py`：训练更新、Episode与更新指标、训练计数、Manifest及摘要已写入后，Runner先调用 `update_training_epoch_return_plot()`，之后才保存最终Checkpoint。本机Matplotlib的二进制扩展按NumPy 1.x编译，而现有Python使用NumPy 2.2.6；导入 `matplotlib._path` 时触发 `ImportError: initialization failed`。

本次将训练后保存顺序调整为：正式PPO更新 → 训练计数及指标落盘 → 保存主Checkpoint → 可选最佳回报Checkpoint → 可选绘图及Transition Trace。两个可选绘图入口（续训摘要图和更新后训练图）均使用局部绘图包装器；只在绘图调用本身失败时输出明确警告并跳过。Checkpoint保存调用位于包装器之外，任何Checkpoint保存错误仍会直接报错。Matplotlib可用时原绘图函数保持不变。

本轮实际触发了NumPy/Matplotlib导入异常，终端记录了 `WARNING: optional training-return plot failed and was skipped (ImportError: initialization failed)`。绘图文件未生成；Checkpoint、数值指标和后续探针均完成。

## 受限训练结果

使用随机初始化的Architecture 33策略、宏Actor学习率 `5e-4`、新配置SHA-256 `a86d87ca5d586e89e367aee393b7a1cd9b27813f41ac3c5a19005bfa40a9ea6c`。原Conda环境无法在本机激活，实际执行使用已有 `C:\software\anaconda3\python.exe`（Python 3.12.7）；Numba仅在当前进程中禁用，正式纯Python KKT回退生效。未改动依赖。

只完成Episode 20010–20013，共1,200个Slot、1,200次Matching和132,035次KKT评价，Rollout约321.55秒，PPO更新约10.33秒。4集Return来自不同Training任务，不能当作策略逐集变好或变差的证据。

| Episode | 任务数 | Return | Task Utility | Cache Cost | UAV成本 | 平均任务时延(s) | 车辆能耗(J) | Edge任务 | Cloud任务 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20010 | 4,187 | 3,886.840 | 3,977.373 | 89.839 | 0.694 | 2.4054 | 94.436 | 846 | 3,341 |
| 20011 | 4,251 | 3,948.546 | 4,032.397 | 83.151 | 0.699 | 2.4660 | 101.469 | 784 | 3,467 |
| 20012 | 4,216 | 3,950.980 | 4,031.533 | 79.856 | 0.696 | 2.0946 | 87.877 | 764 | 3,452 |
| 20013 | 4,143 | 3,826.135 | 3,915.944 | 89.058 | 0.751 | 2.6296 | 104.449 | 780 | 3,363 |

四集共711个Macro决策帧，平均每帧实际更新4.571个节点；更新节点数分布为0–9个节点时分别有 `0, 16, 51, 104, 167, 181, 119, 53, 18, 2` 帧。按动态节点ID记录的逐Episode更新频率保存在诊断JSON中。按节点类型汇总，SV、RSU、UAV节点更新频率的范围分别为43.5%–56.3%、44.6%–55.4%、46.9%–55.6%。累计Cache成本为341.904（SV 156.225、RSU 72.658、UAV 113.021），UAV移动成本合计2.841。平均Holding时长约1.67–1.70 Slot，范围仍为配置的1–5 Slot。

### PPO数值与实际参数更新

- Macro Actor 4/4个Optimizer Step通过KL保护并被接受，拒绝0步；Macro Critic 50步、Long-Value Critic 50步，Micro Actor/Critic各40步。
- 联合KL更新前/后为0.005734/0.007388，联合Clip Fraction为4.18%，Cache Ratio Clip Fraction为0.39%；Micro KL为0.005261，Micro Clip Fraction为4.83%。Ratio、KL、Advantage、Critic目标及Loss均为有限值。
- Macro Actor Loss为−0.007704，Macro Critic Value Loss为386.038；Macro Advantage均值/标准差为251.696/44.983。N2解释方差由0.000025升至0.033234。单个更新组不足以判断收敛或Critic已稳定。
- 训练前后直接比较参数张量：Macro Actor 14/14个张量改变；Cache Candidate Scorer 4/4个张量改变，最大绝对变化0.002001；候选编码器2/2个张量改变；共享Macro Observation Encoder 4/4个张量改变，最大绝对变化0.001965。

## 训练后Checkpoint验收

正式Checkpoint位于 [`latest_checkpoint.pt`](../runs/stageB3_4A_multi_node_factorized_short_20261013/checkpoints/latest_checkpoint.pt)，大小约6.40 MB。其合同为Architecture 33、Checkpoint 60及多节点v4 Schema。保存进度为4个已完成Episode、1个策略更新组；计数器记录Macro Actor 4步、Macro Critic 50步、Long-Value 50步、Micro Actor/Critic各40步，以及1,200次环境执行、1,200次Matching和132,035次KKT。正式Loader恢复了计数器及含真实Adam Step的优化器状态；主Checkpoint中的模型参数SHA-256与保存前内存策略一致。

第一次诊断Loader尝试使用Episode 20010的上下文，但正式Runner会逐集切换上下文，保存时合同中的动态SV节点ID来自最后一集Episode 20013。改用与Checkpoint合同一致的Episode 20013上下文后，正式Loader完整重载通过；没有修改或绕过合同检查。对20007–20009的相同Observation重新生成候选后，两次正式加载所得完整节点概率、评分输入、合法Mask和动作均逐项完全一致；保存前捕获的A/B候选输出也与重载结果一致。

## 固定候选探针

候选按节点ID和完整目标缓存行在当前正式候选集合中重新匹配，没有沿用旧候选索引。表中概率为指定节点的候选概率条件于该节点UPDATE组；“联合概率”是该节点选择候选、其余8个节点KEEP的Cache联合动作概率。分数为Cache Scorer基数校正前logit。B2-8净收益仅作固定动作离线参照，不作为训练标签。

| Episode/候选 | UPDATE内概率（前→后） | UPDATE内排名（前→后） | 原始分数（前→后） | 候选+其他8节点KEEP联合概率（前→后） | 确定性策略命中 |
|---|---:|---:|---:|---:|---|
| 20007 A / UAV_0 | 0.029370→0.029312 | 19→19 | −0.095442→−0.132499 | 5.487e−5→6.580e−5 | 否 |
| 20007 B / RSU_1 | 0.017794→0.018086 | 33→25 | −0.105938→−0.073318 | 3.297e−5→4.238e−5 | 否 |
| 20008 A / SV 23 | 0.100743→0.098782 | 2→8 | −0.023058→−0.057906 | 1.949e−4→2.339e−4 | 否 |
| 20008 B / RSU_1 | 0.017795→0.018087 | 33→25 | −0.077976→−0.043889 | 3.295e−5→4.235e−5 | 否 |
| 20009 A / SV 17 | 0.100130→0.098655 | 5→8 | −0.075333→−0.139213 | 1.842e−4→2.150e−4 | 否 |
| 20009 B / RSU_1 | 0.017794→0.018086 | 33→25 | −0.073223→−0.040977 | 3.296e−5→4.236e−5 | 否 |

三个探针状态的9节点全KEEP联合概率均由约0.00185升至约0.00238。按节点角色汇总的平均KEEP概率：SV为0.4951→0.5106，RSU为0.5008→0.5090，UAV为0.4982→0.5148；每个动态节点的具体值保存在诊断JSON。一次更新没有稳定提高20007正收益A的概率或排名；20008/20009 A排名下降，B排名从33升到25但仍未成为确定性动作。因此本轮只证明策略完成了有效参数更新，不能称为已经学会B2-8的长期候选价值排序或已经收敛。

## 产物与结论

新隔离Run目录 [`stageB3_4A_multi_node_factorized_short_20261013`](../runs/stageB3_4A_multi_node_factorized_short_20261013) 保留训练后Checkpoint、原配置快照、Manifest、Episode/Update/Critic/Slot数值指标和候选诊断JSON。旧B3-4目录及其初始化Checkpoint、历史诊断数据均未覆盖。

该实现已具备进入下一受控训练阶段的运行条件；仍应按阶段门控检查更新指标和Checkpoint。单次更新只是一项运行及参数更新验证，不构成收敛结论。

```ini
STAGE_B3_4A = COMPLETE
PLOTTING_FAILURE_ISOLATED = YES
CHECKPOINT_SAVED_BEFORE_OPTIONAL_PLOTTING = YES
TRAINING_EPISODES_COMPLETED = 20010, 20011, 20012, 20013
PPO_UPDATE_GROUPS_COMPLETED = 1
MACRO_ACTOR_STEPS = 4 accepted / 4 attempted
CACHE_SCORER_PARAMETER_CHANGED = YES
ENCODER_PARAMETER_CHANGED = YES
TRAINED_CHECKPOINT_SAVED = YES
TRAINED_CHECKPOINT_RELOAD = PASS
TRAINING_COUNTERS_RESTORED = YES
BEFORE_AFTER_CANDIDATE_PROBE = COMPLETE
JOINT_PPO_NUMERIC_STATUS = FINITE; joint KL post=0.007388; joint clip fraction=0.041842
READY_FOR_NEXT_TRAINING_STAGE = YES
SYSTEM_MODEL_CHANGED = NO
PPO_ALGORITHM_CHANGED = NO
```
