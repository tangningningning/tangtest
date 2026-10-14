# Stage6 Cache 更新反事实收益诊断

## 结论摘要

- **CONFIRMED（样本内）**：12 个决策状态中，对 KEEP 与 42 个去重后的 UPDATE 分支分别用正式 Slot 接口和 Matching/KKT 重算；42 个 UPDATE 的单槽净收益 `ΔG` 全部小于 0。11 个分支提高了 `TaskUtility`，但每一个的额外 Cache 成本都大于任务收益；14 个分支降低 `TaskUtility`，17 个分支没有改变它。
- **SUPPORTED**：UPDATE 候选之间有价值差异，但在本次每状态只抽查 3–4 个 UPDATE 的条件下，测试候选的 `ΔG` 跨度为 0.050–0.242（中位数 0.116），全部仍低于 KEEP 的 0。PPO 最高评分候选只在 4/12 个状态中是被测 UPDATE 里净收益最高者。不能据此排出全部 230 个候选。
- **SUPPORTED**：已保存训练日志中的 UPDATE 组概率约为 0.50–0.53，但组内分布仍很分散。按条件熵换算，四组平均有效候选数约为 195–230/230；这支持“当前组内排序信号较弱”，但不证明 230 候选完全没有可学习信号。
- **CONFIRMED（样本内）**：四个五槽离线对照中，评分最高的合法 UPDATE 有 1 个取得正净收益（+0.0185），另外 3 个为负（−0.120 至 −0.140）。因此存在持续收益的例子，但不足以说明其经常发生。
- **NOT_SUPPORTED**：没有发现 Cloud 优先回退或 Matching/KKT 分支不一致。54 次单槽 Matching 均通过 transfer-exchange 稳定性复检；7,652 次 KKT 求解全部收敛，未触及最大迭代数。
- **UNKNOWN**：全体 230 个 UPDATE 的完整价值排序、300 Episode 完整训练后的 PPO 对齐情况，以及超过 5 Slot 或包含 PPO 后续重新决策的累计收益，本次均未测量。

## 数据、Checkpoint 与反事实方法

使用正式 Training 池 `environment_data/proposed_joint_training/`，选择了四个不同 seed：20001、20101、20201、20300；每个 seed 对应一个现存学习率 Checkpoint。所选 Episode manifest 的 raw config hash 均为 `684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77`，与当前配置及训练池清单一致；semantic hash 为 `10f893fc549c4016bedb1a5389ad996f9a3ff646aa012596a7230c284b13f542`。训练池清单记录 seeds 20001–20300，共 300 个 Episode。

每个 Checkpoint 回放前五个 Macro 帧，再从中选择三个状态，合计 12 个状态。选择时尽量纳入 KEEP 与 UPDATE、不同 Holding 长度和不同实际 Cloud/Edge 分布。每个状态记录当前 Cache、UAV 起点和 PPO Micro 动作后的 UAV 位置，详见下一节。

候选分支使用同一状态下的 KEEP、Checkpoint 策略实际抽样动作、UPDATE 组最高评分、中央评分和最低评分候选；重复候选只执行一次。每个 UPDATE 都来自项目现有候选生成器，没有额外构造候选。每个分支固定同一组实际任务、车辆位置、scenario 和该状态 PPO Micro 动作选出的 UAV 目标位置；分支之间不共享 Matching/KKT 缓存，分别执行正式的 `execute_configured_slot`。12 个 PPO 抽样动作的单槽结果均与原回放结果精确匹配。

历史日志没有提供足以重建这些状态下完整候选张量、任务匹配和效用的逐步轨迹。因此报告中的“PPO 实际选择”指用对应 Checkpoint 按正式随机 `sample_macro` 接口进行的可复现诊断抽样，诊断随机种子固定为 Episode seed；它不是对过去训练过程中某个动作的伪称。这里没有调用训练更新。

## 正式代码路径与模型规则

1. `src/ppo_simple/interface.py:117-151,236-308` 根据明确传入的 Episode 目录加载 service catalog、prediction scenario、TV 本地缓存，并将 Episode 的任务、轨迹和预测对齐为 Slot。`src/ppo_simple/rollout.py:389-475` 以当前 Cache 为候选 0（KEEP），调用每个 Cache 节点的正式 target proposal 生成器；随后 `rollout.py:645-660` 将完整候选和 observation 交给 PPO Macro policy。每个 Slot 中，`rollout.py:916-933` 将选中的 Cache、PPO Micro UAV 目标、`step.actual_tasks`、当前车辆位置和 scenario 传给 `slot_environment.execute_configured_slot`。
2. `_cache_candidates` 通过 `primitive_control._target_cache_proposals_with_features` 和 `cache_configuration.enumerate_all_cache_target_configurations` / `select_cache_target_configurations` 生成候选（`src/ppo_simple/primitive_control.py:326-425`、`src/decision/cache_configuration.py:939-1156`）。候选只改变一个 Cache 节点；节点容量、二值状态和形状由 `apply_cache_target_configuration` / `validate_cache_shape_and_capacity` 检查（`cache_configuration.py:1161-1220`）。这四组训练日志每个 Macro 状态均记录 230 个合法 UPDATE 候选；本次 12 个回放状态也各有 230 个。
3. Local 只有在该任务所属车辆的 TV 本地缓存中包含所需 service 时才成为候选，且 Local target 绑定该 task 自己（`src/decision/task_allocation.py:277-280,325-335`）。SV/RSU/UAV 必须缓存任务 service；有完整物理上下文时还需车辆和节点坐标有效，并且节点 CPU、带宽、信道增益和 uplink rate 均为正且有限（`task_allocation.py:235-322,1809-1854`）。这些 Edge 检查没有额外的最大时延门槛；时延和能耗进入 Task Utility。KKT 的 CPU 与带宽份额、资源总和约束在 `task_allocation.py:2341-2534` 校验。
4. 当前配置采用 `RESOURCE_AWARE_FALLBACK`。该模式保留 Cloud 和所有合法 Edge 候选；Edge/Cloud 任务分配由 Matching 继续比较，而不是无条件回退到 Cloud（`task_allocation.py:302-308`）。`optimize_task_matching` 每轮评估 Transfer 与 Exchange 操作的实际任务效用；只有超过容差的改进会被接受，资源感知模式下效用平局偏向 Edge（`task_allocation.py:966-1300`）。这是一种 transfer-exchange 稳定性，不是对所有整体 matching 的穷举全局最优证明。
5. 对每个 Matching，`evaluate_matching_with_kkt` 重新计算该 Matching 的 Edge 节点资源分配与 Cloud 任务资源分配（`task_allocation.py:2782-2990`）。Edge 按 SV/RSU/UAV 节点分别共享 CPU 和带宽（`allocate_resources_kkt`, `task_allocation.py:2341-2534`）。Cloud 在 `finite_pool` 模式下以全体 Cloud 任务共享一个 40 GHz CPU 池，由同一 KKT 分配器分配（`_allocate_cloud_cpu_kkt`, `task_allocation.py:2537-2679`）；Cloud 不进入 Edge 带宽份额分配图（`task_allocation.py:2521-2522`）。
6. 当前配置实际解析值为 `cloud_resource_mode=finite_pool`、`cloud_cpu_capacity=40e9 cycles/s`、每 Cloud 任务 `200e6 bps`、`cloud_delay_factor=15`、`cloud_fallback_mode=RESOURCE_AWARE_FALLBACK`。Cloud 公式为 `15 × input_bits / 200 Mbps + cpu_cycles / allocated_cloud_cpu`；`cloud_execution` 直接使用调用者传入的 rate，并未按 Cloud 任务数分带宽（`src/common/system_model.py:131-169`，调用位置 `task_allocation.py:2946-2954`）。配置中的 `resource_sharing=false` 与此行为一致。`system_config.toml:306` 的注释文字仍写“固定值为 11.0”，但生效配置键 `system_config.toml:309` 和运行时解析值是 15.0；本报告按有效值 15.0 计算。
7. Matching 输出的 Task Utility 总和为 `Q`；Slot 的系统效用为 `G = Q − CacheCost − MovementCost`（`src/decision/slot_environment.py:510-523,590-614`）。CacheCost 由项目现有函数计算：变化节点有触发成本，新增 service 按 service size 计算加载成本，删除不抵消新增加载量（`src/common/system_model.py:218-300`）。PPO 的 Macro reward 因而可同时包含当前更新成本和后续 Holding Slot 收益；本次 `ΔG` 按实际 Slot Utility 逐分支计算。
8. PPO 用 cardinality correction 将同一合法 UPDATE 组内的每个 logit 减去 `log(230)`，保留组内排序并调整 KEEP/UPDATE 组概率；随后正式接口按校正分布抽样，另存原始分布概率（`src/ppo_simple/policy.py:368-390,475-593`）。所以“PPO 评分最高”按原始 UPDATE 概率最大取值，等价于组内校正概率最大。

## 12 个回放状态

`Cache` 列记录决策前各节点缓存的 service ID；`UAV` 列记录决策前坐标和同一个固定 Micro 动作的目标坐标（米）。任务分布格式为 Cloud/Edge/Local。PPO 抽样概率为对应候选动作的校正概率；UPDATE rank 按全体 230 个合法 UPDATE 从高到低排序。

| Seed@Slot | LR | PPO 动作 | Holding | Tasks (Cloud/Edge/Local) | 决策前 Cache：node[service IDs] | UAV 决策前 -> Micro 目标 (m) |
|---|---:|---|---:|---:|---|---|
| 20001@0 | 1e-4 | UPDATE #222 (p=.00190, rank 155/230) | 5 | 12 (10/2/0) | 46[4,7];10[8];0[9];23[4,5];59[8];RSU_0[3,5];RSU_1[5,7,9];UAV_0[5,8];UAV_1[1,5,7] | UAV_0(3364.0,1119.9)->(3364.0,1119.9); UAV_1(5179.6,1710.4)->(5194.6,1710.4) |
| 20001@10 | 1e-4 | UPDATE #182 (p=.00192, rank 149/230) | 4 | 15 (11/4/0) | 46[4,7];10[8];0[9];23[4,5];59[8];RSU_0[3,5];RSU_1[1,2,4];UAV_0[5,8];UAV_1[4,6] | UAV_0(3353.4,1122.4)->(3353.4,1122.4); UAV_1(5167.7,1701.3)->(5167.7,1701.3) |
| 20001@19 | 1e-4 | UPDATE #201 (p=.00210, rank 64/230) | 5 | 18 (16/2/0) | 46[4,7];10[8];0[9];23[4,5];59[8];RSU_0[];RSU_1[1,2,4];UAV_0[4,9];UAV_1[4,6] | UAV_0(3353.4,1129.9)->(3348.1,1135.2); UAV_1(5135.9,1733.1)->(5135.9,1733.1) |
| 20101@0 | 5e-4 | UPDATE #153 (p=.00243, rank 93/230) | 1 | 14 (12/2/0) | 14[4,7];54[8];7[9];51[4,5];3[8];RSU_0[3,5];RSU_1[5,7,9];UAV_0[5,8];UAV_1[1,5,7] | UAV_0(3364.0,1119.9)->(3358.7,1114.6); UAV_1(5179.6,1710.4)->(5179.6,1710.4) |
| 20101@1 | 5e-4 | KEEP (p=.45060) | 5 | 14 (10/4/0) | 14[4,7];54[8];7[9];51[4,5];3[8];RSU_0[3,5];RSU_1[1,5];UAV_0[5,8];UAV_1[1,5,7] | UAV_0(3358.7,1114.6)->(3358.7,1114.6); UAV_1(5179.6,1710.4)->(5179.6,1710.4) |
| 20101@16 | 5e-4 | KEEP (p=.45066) | 1 | 14 (12/2/0) | 14[4,7];54[8];7[9];51[4,5];3[8];RSU_0[3,5];RSU_1[1,5];UAV_0[5,8];UAV_1[1,5,7] | UAV_0(3355.6,1104.0)->(3355.6,1119.0); UAV_1(5155.2,1733.3)->(5149.9,1728.0) |
| 20201@0 | 1e-3 | UPDATE #153 (p=.00029, rank 187/230) | 2 | 15 (11/4/0) | 6[4,7];16[8];23[9];8[4,5];43[8];RSU_0[3,5];RSU_1[5,7,9];UAV_0[5,8];UAV_1[1,5,7] | UAV_0(3364.0,1119.9)->(3364.0,1119.9); UAV_1(5179.6,1710.4)->(5173.4,1706.3) |
| 20201@3 | 1e-3 | KEEP (p=.51875) | 2 | 12 (11/1/0) | 6[4,7];16[8];23[9];8[4,5];43[8];RSU_0[5,9];RSU_1[1,5];UAV_0[5,8];UAV_1[1,5,7] | UAV_0(3364.0,1119.9)->(3358.7,1114.6); UAV_1(5165.9,1706.3)->(5165.9,1706.3) |
| 20201@5 | 1e-3 | KEEP (p=.51839) | 5 | 11 (9/2/0) | 6[4,7];16[8];23[9];8[4,5];43[8];RSU_0[5,9];RSU_1[1,5];UAV_0[5,8];UAV_1[1,5,7] | UAV_0(3366.2,1114.6)->(3366.2,1114.6); UAV_1(5165.9,1706.3)->(5165.9,1706.3) |
| 20300@0 | 5e-3 | UPDATE #227 (p=.00915, rank 18/230) | 1 | 15 (12/3/0) | 8[4,7];13[8];34[9];24[4,5];18[8];RSU_0[3,5];RSU_1[5,7,9];UAV_0[5,8];UAV_1[1,5,7] | UAV_0(3364.0,1119.9)->(3364.0,1119.9); UAV_1(5179.6,1710.4)->(5179.6,1725.4) |
| 20300@2 | 5e-3 | KEEP (p=.47529) | 1 | 16 (12/4/0) | 8[4,7];13[8];34[9];24[4,5];18[4,5];RSU_0[3,5];RSU_1[5,7,9];UAV_0[5,8];UAV_1[1,8] | UAV_0(3358.7,1129.6)->(3348.1,1119.0); UAV_1(5179.6,1702.9)->(5179.6,1702.9) |
| 20300@4 | 5e-3 | KEEP (p=.35977) | 1 | 14 (10/4/0) | 8[4,7];13[8];34[9];24[4,5];18[4,5];RSU_0[3,5];RSU_1[2,9];UAV_0[5,8];UAV_1[1,8] | UAV_0(3348.1,1119.0)->(3348.1,1119.0); UAV_1(5172.1,1702.9)->(5172.1,1702.9) |

## 单槽收益与候选排序

下表对每个状态列出 PPO 抽样动作、UPDATE 组最高评分、中央评分和最低评分候选的变化。`ΔG=ΔQ−ΔC−ΔM`；本次同一状态的固定 Micro 动作使所有配对的 `ΔM=0`。正的 `ΔQ` 是任务效用提高，不代表净收益为正。

| Seed@Slot | PPO 抽样动作 | 最高评分 UPDATE（rank 1） | 中央 UPDATE（rank 115） | 最低评分 UPDATE（rank 230） | 被测 UPDATE 的 ΔG 跨度 |
|---|---|---|---|---|---:|
| 20001@0 | #222: −.025/.250/−.275 | #141: +.000/.034/−.034 | #99: −.043/.073/−.116 | #39: +.000/.099/−.099 | .242 |
| 20001@10 | #182: +.000/.175/−.175 | #228: +.007/.202/−.195 | #70: −.071/.105/−.176 | #39: +.000/.099/−.099 | .096 |
| 20001@19 | #201: −.064/.015/−.079 | #228: −.001/.202/−.204 | #120: −.057/.173/−.230 | #39: +.000/.099/−.099 | .151 |
| 20101@0 | #153: +.000/.049/−.049 | #170: +.000/.110/−.110 | #26: +.000/.081/−.081 | #176: +.020/.222/−.202 | .152 |
| 20101@1 | KEEP: .000/.000/.000 | #126: +.006/.177/−.171 | #85: −.096/.159/−.256 | #163: −.006/.222/−.228 | .085 |
| 20101@16 | KEEP: .000/.000/.000 | #126: +.008/.177/−.169 | #85: +.067/.159/−.093 | #164: +.006/.222/−.216 | .124 |
| 20201@0 | #153: +.000/.049/−.049 | #168: −.055/.135/−.190 | #101: −.002/.154/−.156 | #159: +.000/.073/−.073 | .141 |
| 20201@3 | KEEP: .000/.000/.000 | #133: +.000/.161/−.161 | #57: +.000/.049/−.049 | #66: +.068/.154/−.086 | .112 |
| 20201@5 | KEEP: .000/.000/.000 | #133: +.037/.161/−.124 | #195: −.009/.218/−.227 | #65: +.047/.154/−.107 | .120 |
| 20300@0 | #227: +.044/.154/−.109 | #6: +.000/.120/−.120 | #50: +.000/.137/−.137 | #160: +.000/.170/−.170 | .060 |
| 20300@2 | KEEP: .000/.000/.000 | #6: +.000/.120/−.120 | #226: −.040/.088/−.129 | #160: +.000/.170/−.170 | .050 |
| 20300@4 | KEEP: .000/.000/.000 | #6: +.000/.120/−.120 | #192: −.010/.178/−.187 | #67: −.023/.177/−.200 | .080 |

上表每个三元数依次为 `ΔQ/ΔC/ΔG`，单位为项目效用。42 个 UPDATE 对照的汇总：

- `ΔQ` 范围 −0.0964 至 +0.0681；11/42 为正、14/42 为负、17/42 约为 0。
- `ΔC` 范围 +0.015 至 +0.2501；42/42 满足 `ΔC > ΔQ`，所以单槽 `ΔG` 为 −0.2752 至 −0.0336，正净收益 0/42。
- 41/42 的 UPDATE 有新增 service 加载；加载量中位数 13,545 MB，最大 23,855 MB。剩下一个分支没有新增加载，但仍有 +0.015 的节点更新触发成本。
- 在六个 PPO 抽样 UPDATE 中，只有两个提升单槽 `Q`，六个 `G` 都低于各自 KEEP。六个抽样 UPDATE 在 230 个动作中的模型评分名次为 155、149、64、93、187、18。

## 对卸载分布的影响

12 个 PPO 抽样动作共涉及 170 个任务：136 Cloud（80%）、12 RSU、22 UAV、0 SV、0 Local。KEEP 对照为 134 Cloud、12 RSU、24 UAV；最高评分 UPDATE 为 134 Cloud、15 RSU、21 UAV；中央 UPDATE 为 139 Cloud、8 RSU、23 UAV；最低评分 UPDATE 为 131 Cloud、14 RSU、25 UAV。被测更新对照的任务匹配变化累计为 18 次 Cloud→Edge、22 次 Edge→Cloud 和 6 次 Edge 内节点切换；这是跨 42 个重复候选比较的事件数，不是 42 个独立 Episode。

因此 Cache UPDATE 确实改变了一些任务的可用 Edge 服务和资源分配，但本样本没有显示它能稳定地把 Cloud 任务转到 Edge。PPO 实际动作和 KEEP 的 Cloud 占比接近，UPDATE 方向的任务切换也略偏向 Edge→Cloud。这里没有统计未入选候选的可行任务集合，因此不能把样本中的 0 个 SV/Local 执行归因到某一条单独缓存约束。

## 五槽持续收益

对四个 seed 的 slot 0，选择 PPO 评分最高的合法 UPDATE，与 KEEP 对照相同的未来五个 Training task slots。更新后的 Cache 在这五槽中固定不变；两分支使用原 Checkpoint rollout 记录的 Micro UAV 位置轨迹。未来真实任务只用于离线评价，不参与 Macro/Micro 决策。

| Seed | Update | ΔQ（5 Slot） | ΔC（一次更新） | ΔM | ΔG（5 Slot） | 正净收益 |
|---:|---:|---:|---:|---:|---:|---|
| 20001 | #141 | +0.052137 | +0.033611 | 0 | +0.018526 | 是 |
| 20101 | #170 | −0.028212 | +0.110395 | 0 | −0.138607 | 否 |
| 20201 | #168 | −0.004722 | +0.134984 | 0 | −0.139706 | 否 |
| 20300 | #6 | +0.000000 | +0.119986 | 0 | −0.119986 | 否 |

这是“缓存维持五槽、Micro 轨迹固定”的离线情景，不包含 PPO 在中间 Slot 重新选 Cache 或 UAV 后的响应。它证明了某些 UPDATE 可能靠持续收益摊平加载成本，但样本中 3/4 的最高评分候选五槽净收益仍为负。

## PPO 排序信号与已训练策略

| Macro Actor LR | Checkpoint 完成 Episode / 300 | 已完成 PPO 更新 / 75 | 训练日志 UPDATE 率 | 平均 Holding Slot | 每状态 UPDATE 候选数 | UPDATE 组内平均有效候选数 |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 136 | 34 | 50.36% | 3.744 | 230 | 229.8 |
| 5e-4 | 64 | 16 | 53.47% | 2.447 | 230 | 227.7 |
| 1e-3 | 120 | 30 | 50.31% | 2.273 | 230 | 203.7 |
| 5e-3 | 88 | 22 | 51.32% | 1.114 | 230 | 195.3 |

“有效候选数”按训练日志的校正概率计算 `exp(H(UPDATE候选 | UPDATE组))`，上限为 `exp(log(230))=230`。四组各自的组内条件熵均靠近上限。再看本次有限候选重算，最高模型评分候选只在 4/12 个状态里取得被测候选最高 `ΔG`；每状态只有 3–4 个被测 UPDATE 的排名相关性会大幅变化，不能用于全候选统计。

判断：**SUPPORTED** 当前策略把 KEEP/UPDATE 组概率维持在约一半，并对若干候选给出不同评分；**NOT_SUPPORTED** 现有检查支持“PPO 已可靠学到 230 个 UPDATE 的净收益排序”；**UNKNOWN** 完整 230 排序是否存在更好的价值信号。本次高评分 UPDATE 有 1/4 在五槽内取得净收益，但四个 Checkpoint 只完成了 64–136 个 Episode、16–34 次 PPO 更新，尚未到设定的 300 Episode/75 次更新，不能代表完整训练后的最终策略。

## Matching/KKT 与 Cloud 审查结果

在 54 个单槽分支（12 KEEP + 42 UPDATE）中：

- 54/54 的 Transfer-Exchange Matching 均达到 no-improvement 停止条件并通过稳定性复检；未观察到非改进操作被接受。
- 7,652/7,652 次 KKT solve 均收敛，最大迭代次数触发为 0；最大资源残差 `9.995e-13`，最大 stationarity residual `1.78e-15`，最大 complementary slackness residual `5.77e-12`。
- 全部分支均报告 `cloud_resource_mode=finite_pool`、容量 40 GHz。Cloud CPU 实际用量为约 39,999,999,999.968–40,000,000,000 cycles/s，证明共享池确实参与并在这些状态中接近饱和。
- 本次实际抽样匹配 Cloud 占 80%，同时任务效用由同一 `system_model.task_utility` 公式评价；代码路径没有按 Cloud/Edge 分设奖励。Edge 的路径和缓存候选由正式 feasibility 检查构造，KKT 分配则按完整节点任务组重算。

结论为 **NOT_SUPPORTED（当前证据未发现实现错误）**。Cloud 有配置层面的带宽优势：每 Cloud task 固定等效 200 Mbps，Cloud 任务间不分摊带宽；但 Cloud CPU 池有限、按任务共同分配并在样本中饱和，且 delay factor=15 已在上传时延项生效。这些结果与当前模型下 Cloud 仍经常取得更高总任务效用相容；它不能证明该物理假设适用于真实部署。

## 证据边界与运行限制

- 本次执行了四个Training Episode的短策略回放、12 个状态的 KEEP/UPDATE 对照和四个最多五槽的离线对照；没有启动 PPO 训练，没有使用 Validation/Test，没有遍历全部 230 个候选，没有重新生成数据。
- 所选 Checkpoint 只完成 64–136/300 Episode、16–34/75 PPO 更新。结论是有限样本的诊断，不是完整训练或全池估计。
- 五槽结果采用固定 Cache 和实际 PPO Micro UAV 路径；不包含 Cache 更新后 PPO 对未来观测作出的新 Macro/Micro 决策。
- 当前 Windows 主机无法激活 `conda activate adaptive_cache_mec`：未找到该环境目录，Conda 启动还报 Windows 系统 DLL relocation 错误。为完成回放，使用 `C:\software\anaconda3\python.exe` 3.12.7，并让项目可选 Numba 求解器走其官方纯 Python fallback。正式项目源码、配置和算法均未修改。该路径与 Numba 分支使用同一 KKT 目标和约束，但未在用户指定 Conda 环境中重跑。

## 状态汇总

| 命题 | 状态 | 证据范围 |
|---|---|---|
| 单槽 UPDATE 能改善任务效用 Q | **SUPPORTED** | 11/42 UPDATE 的 ΔQ 为正；14/42 为负，17/42 约为零 |
| 单槽 UPDATE 能取得正净收益 G | **NOT_SUPPORTED（本样本）** | 42/42 的 ΔC 大于 ΔQ，正 ΔG 为 0/42 |
| UPDATE 有持续收益潜力 | **SUPPORTED（小样本）** | 固定缓存五槽对照为 1/4 正收益 |
| 测试候选之间有价值分离 | **SUPPORTED** | 每状态有限候选 ΔG 跨度中位数 0.116；所有被测 UPDATE 仍低于 KEEP |
| 当前 PPO 已学会完整 230 候选收益排序 | **NOT_SUPPORTED** | 最高策略评分候选仅 4/12 是被测 UPDATE 中净收益最高；有效组内支持数仍为 195–230/230 |
| Cloud CPU finite_pool 确实生效 | **CONFIRMED** | 54 个分支均为 finite_pool，40 GHz 池在样本中饱和 |
| Cloud 有无条件 Matching 优先 | **NOT_SUPPORTED** | Cloud 与所有合法 Edge 均进入资源感知候选，实际 matching 由 Task Utility 改善驱动，平局偏向 Edge |
| Matching/KKT 有实现错误 | **NOT_SUPPORTED（检查样本内）** | Matching 全部稳定，KKT 全部收敛且残差小；不能据此证明任意输入均无错误 |
