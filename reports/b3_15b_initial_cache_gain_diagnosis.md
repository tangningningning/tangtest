# B3-15B：初始Cache配置及真实更新收益诊断

## 范围与执行合同

- 使用当前正式配置和Training Seeds 20001、20101、20201。每个Episode只用Slot 0预测生成候选；Slot 0基线及每个唯一更新分支执行同一实际任务、车辆输入的未来5个Slot。没有完整Episode、PPO训练、Validation或Test。
- Conda激活入口在终端不可执行；使用备用解释器，sys.prefix确认是adaptive_cache_mec环境。
- 策略概率使用runs/stageB3_14h_fresh_macro_actor_lr_1e_minus_3/checkpoints/latest_checkpoint.pt。严格Trainer只读加载通过：Architecture 33、Checkpoint 60、完成Episode 300、完整Update 75、Macro Actor LR 0.001。当前配置两类哈希、Training Pool Manifest和integrity baseline均与Run provenance一致。未恢复RNG、未调用优化器更新。
- 9个初始Cache节点按角色稳定顺序取每个Episode第一个SV、RSU、UAV作代表节点。各分支从相同C(0)起步，只替换一节点；UAV在5个Slot固定停留初始位置。Matching/KKT每个Slot使用新缓存，不在分支间共享求解器状态。任务/车辆/预测输入在分支间使用同一对齐数据并逐Slot校验指纹；每次Matching前恢复同一Python/NumPy/Torch随机状态。
- Slot 0预测推荐只用正式预测需求函数与正式贪心Cache目标构造；未来真实任务只作离线后果评价，不参与候选生成。容量、Matching/KKT、Cache成本、Reward均通过现有正式实现。

## 1. 初始Cache C(0)

初始化种子为4401。当前源码用固定种子对Service Catalog均匀随机排序，再逐节点按容量装入；它不读取预测或真实需求，因此C(0)是确定、容量可行的初始化，不是需求最优化配置。三个Episode的Cache矩阵相同，SHA256=2ec5e77e758ff7634c59c71b79de45495787165091a02746f0ffc58430a1094c；SV车辆实例ID随Episode变化，缓存行按角色稳定顺序对应。

| 行 | 角色 | Seed 20001节点ID | C(0)服务ID | 已用/容量 MB | 剩余 MB | 容量利用率 |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 0 | SV | 46 | 4, 7 | 11322.61 / 12000.00 | 677.39 | 94.36% |
| 1 | SV | 10 | 8 | 11553.56 / 12000.00 | 446.44 | 96.28% |
| 2 | SV | 0 | 9 | 9998.63 / 12000.00 | 2001.37 | 83.32% |
| 3 | SV | 23 | 4, 5 | 9487.42 / 12000.00 | 2512.58 | 79.06% |
| 4 | SV | 59 | 8 | 11553.56 / 12000.00 | 446.44 | 96.28% |
| 5 | RSU | RSU_0 | 3, 5 | 23333.74 / 25000.00 | 1666.26 | 93.33% |
| 6 | RSU | RSU_1 | 5, 7, 9 | 24062.67 / 25000.00 | 937.33 | 96.25% |
| 7 | UAV | UAV_0 | 5, 8 | 17667.98 / 20000.00 | 2332.02 | 88.34% |
| 8 | UAV | UAV_1 | 1, 5, 7 | 19659.87 / 20000.00 | 340.13 | 98.30% |

Service副本分布：

| Service ID | 大小 MB | 副本数 |
| --- | ---: | ---: |
| 0 | 19753.56 | 0 |
| 1 | 5595.83 | 1 |
| 2 | 13856.85 | 0 |
| 3 | 17219.32 | 1 |
| 4 | 3373.00 | 2 |
| 5 | 6114.43 | 5 |
| 6 | 16216.88 | 0 |
| 7 | 7949.62 | 3 |
| 8 | 11553.56 | 3 |
| 9 | 9998.63 | 2 |

C(0)跨Episode核对：

| Seed | 节点ID顺序（5 SV、2 RSU、2 UAV） | C(0) SHA256 |
| ---: | --- | --- |
| 20001 | 46, 10, 0, 23, 59, RSU_0, RSU_1, UAV_0, UAV_1 | 2ec5e77e758ff7634c59c71b79de45495787165091a02746f0ffc58430a1094c |
| 20101 | 14, 54, 7, 51, 3, RSU_0, RSU_1, UAV_0, UAV_1 | 2ec5e77e758ff7634c59c71b79de45495787165091a02746f0ffc58430a1094c |
| 20201 | 6, 16, 23, 8, 43, RSU_0, RSU_1, UAV_0, UAV_1 | 2ec5e77e758ff7634c59c71b79de45495787165091a02746f0ffc58430a1094c |

## 2. Slot 0缓存命中、物理可执行与实际卸载

缓存命中=至少一个Edge Cache有该服务；可执行Edge=正式build_feasible_targets结合当前实际车辆/节点位置、无线与资源基础条件后至少含一个Edge节点；实际Edge卸载=Matching最终选Edge。Cloud有Edge候选但最终选Cloud仅代表完整Matching/共享资源条件下的选择，不证明Cloud单任务效用必然更优。

| Seed | 任务总数 | 缓存命中 | 有可执行Edge | 实际Edge分配 | Cloud无Edge | Cloud有Edge候选 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20001 | 12 | 91.67% (11/12) | 91.67% (11/12) | 75.00% (9/12) | 1 | 2 |
| 20101 | 14 | 57.14% (8/14) | 57.14% (8/14) | 28.57% (4/14) | 6 | 4 |
| 20201 | 15 | 93.33% (14/15) | 93.33% (14/15) | 53.33% (8/15) | 1 | 6 |

合计41个任务：缓存命中33（80.49% (33/41)）；有可执行Edge候选33（80.49% (33/41)）；实际Edge分配21（51.22% (21/41)）；Cloud选择20，其中无Edge候选8、有Edge候选12。合法候选角色覆盖（可重叠）：SV 22（53.66% (22/41)）、RSU 17（41.46% (17/41)）、UAV 20（48.78% (20/41)）。

## 3. Slot 0状态下未来5个Slot的真实反事实

同一节点的候选配置按内容去重；若预测推荐、最少删除、策略最高概率动作指向相同Cache，只执行一次Matching/KKT并合并标签。最少删除从正式合法动作中按删除服务数、变更位数、候选序号依次排序。预测推荐采用正式node-specific predicted demand及贪心容量装入逻辑，并核对该Target是否在正式策略动作集合中。策略动作从Update75 Checkpoint输出的基数校正分布取UPDATE组内最高单候选概率。

| Seed | 节点 | 更新类别 | 删除 → 加入Service | P(Update组) | 候选概率 / Update组内概率 | ΔTask Utility | ΔCache Cost | ΔMovement Cost | ΔReturn | 分配改变数 | 同节点CPU/BW/效用改变数 |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20001 | SV 46 | MINIMUM_DELETION_LEGAL_UPDATE | 4 → ∅ | 0.466778 | 0.0280806 / 0.0601583 | 0.000000 | 0.020000 | 0.000000 | **-0.020000** | 0 | CPU 0 / BW 0 / utility 0 |
| 20001 | SV 46 | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 4,7 → 1,5 | 0.466778 | 0.0782514 / 0.167642 | -0.026866 | 0.137103 | 0.000000 | **-0.163969** | 3 | CPU 3 / BW 3 / utility 8 |
| 20001 | SV 46 | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 7 → 1 | 0.466778 | 0.0608413 / 0.130343 | -0.026866 | 0.075958 | 0.000000 | **-0.102824** | 3 | CPU 3 / BW 3 / utility 8 |
| 20001 | RSU RSU_0 | MINIMUM_DELETION_LEGAL_UPDATE | 3 → ∅ | 0.316547 | 0.00375053 / 0.0118483 | -0.306284 | 0.010000 | 0.000000 | **-0.316284** | 2 | CPU 1 / BW 1 / utility 10 |
| 20001 | RSU RSU_0 | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 3 → 1,4 | 0.316547 | 0.0170642 / 0.0539075 | -0.306284 | 0.072782 | 0.000000 | **-0.379066** | 2 | CPU 1 / BW 1 / utility 10 |
| 20001 | RSU RSU_0 | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 3,5 → 0,4 | 0.316547 | 0.0130367 / 0.0411842 | -0.336571 | 0.171886 | 0.000000 | **-0.508457** | 7 | CPU 3 / BW 3 / utility 10 |
| 20001 | UAV UAV_0 | MINIMUM_DELETION_LEGAL_UPDATE | 5 → ∅ | 0.387662 | 0.0132105 / 0.0340772 | 0.000000 | 0.015000 | 0.000000 | **-0.015000** | 0 | CPU 0 / BW 0 / utility 0 |
| 20001 | UAV UAV_0 | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 5,8 → 4,6 | 0.387662 | 0.0298309 / 0.0769508 | -0.609864 | 0.250079 | 0.000000 | **-0.859943** | 22 | CPU 4 / BW 4 / utility 31 |
| 20001 | UAV UAV_0 | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 5 → 4 | 0.387662 | 0.0223848 / 0.057743 | 0.000000 | 0.055476 | 0.000000 | **-0.055476** | 0 | CPU 0 / BW 0 / utility 0 |
| 20101 | SV 14 | MINIMUM_DELETION_LEGAL_UPDATE | 4 → ∅ | 0.467183 | 0.0293767 / 0.0628805 | 0.000000 | 0.020000 | 0.000000 | **-0.020000** | 0 | CPU 0 / BW 0 / utility 0 |
| 20101 | SV 14 | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 4,7 → 1,5 | 0.467183 | 0.075376 / 0.161341 | 0.066101 | 0.137103 | 0.000000 | **-0.071001** | 4 | CPU 3 / BW 3 / utility 13 |
| 20101 | SV 14 | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 7 → 5 | 0.467183 | 0.0643041 / 0.137642 | -0.034022 | 0.081144 | 0.000000 | **-0.115166** | 2 | CPU 2 / BW 2 / utility 12 |
| 20101 | RSU RSU_0 | MINIMUM_DELETION_LEGAL_UPDATE | 3 → ∅ | 0.322833 | 0.00402937 / 0.0124813 | -1.441298 | 0.010000 | 0.000000 | **-1.451298** | 10 | CPU 1 / BW 1 / utility 49 |
| 20101 | RSU RSU_0 | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 3 → 1,4 | 0.322833 | 0.0162997 / 0.0504895 | -1.400148 | 0.072782 | 0.000000 | **-1.472929** | 12 | CPU 0 / BW 0 / utility 48 |
| 20101 | RSU RSU_0 | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 3 → 4,8 | 0.322833 | 0.0044035 / 0.0136402 | -1.441298 | 0.114486 | 0.000000 | **-1.555784** | 10 | CPU 1 / BW 1 / utility 49 |
| 20101 | UAV UAV_0 | MINIMUM_DELETION_LEGAL_UPDATE | 5 → ∅ | 0.390471 | 0.013437 / 0.0344122 | 0.021513 | 0.015000 | 0.000000 | **0.006513** | 4 | CPU 0 / BW 0 / utility 0 |
| 20101 | UAV UAV_0 | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 5,8 → 4,6 | 0.390471 | 0.0282259 / 0.0722867 | 0.812744 | 0.250079 | 0.000000 | **0.562665** | 13 | CPU 0 / BW 0 / utility 39 |
| 20101 | UAV UAV_0 | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 8 → 4,9 | 0.390471 | 0.00997095 / 0.0255357 | 0.500309 | 0.175459 | 0.000000 | **0.324850** | 11 | CPU 0 / BW 0 / utility 32 |
| 20201 | SV 6 | MINIMUM_DELETION_LEGAL_UPDATE | 4 → ∅ | 0.468196 | 0.0295796 / 0.0631777 | -0.030957 | 0.020000 | 0.000000 | **-0.050957** | 1 | CPU 0 / BW 0 / utility 10 |
| 20201 | SV 6 | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 4,7 → 1,5 | 0.468196 | 0.0752784 / 0.160784 | -0.106757 | 0.137103 | 0.000000 | **-0.243859** | 2 | CPU 2 / BW 2 / utility 12 |
| 20201 | SV 6 | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 7 → 1 | 0.468196 | 0.0601852 / 0.128547 | -0.075800 | 0.075958 | 0.000000 | **-0.151758** | 1 | CPU 2 / BW 2 / utility 2 |
| 20201 | RSU RSU_0 | MINIMUM_DELETION_LEGAL_UPDATE | 3 → ∅ | 0.321505 | 0.00407589 / 0.0126775 | -0.655390 | 0.010000 | 0.000000 | **-0.665390** | 3 | CPU 0 / BW 0 / utility 24 |
| 20201 | RSU RSU_0 | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 3 → 1,4 | 0.321505 | 0.0160722 / 0.0499905 | -0.112832 | 0.072782 | 0.000000 | **-0.185614** | 9 | CPU 0 / BW 0 / utility 28 |
| 20201 | RSU RSU_0 | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 3 → 4,9 | 0.321505 | 0.00585879 / 0.018223 | 0.238177 | 0.103601 | 0.000000 | **0.134576** | 14 | CPU 2 / BW 2 / utility 33 |
| 20201 | UAV UAV_0 | MINIMUM_DELETION_LEGAL_UPDATE | 5 → ∅ | 0.38766 | 0.0135018 / 0.034829 | -0.070859 | 0.015000 | 0.000000 | **-0.085859** | 6 | CPU 2 / BW 2 / utility 12 |
| 20201 | UAV UAV_0 | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 5,8 → 4,6 | 0.38766 | 0.0276525 / 0.0713319 | 0.181977 | 0.250079 | 0.000000 | **-0.068101** | 19 | CPU 1 / BW 1 / utility 32 |
| 20201 | UAV UAV_0 | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 8 → 1,4 | 0.38766 | 0.0102513 / 0.026444 | 0.271413 | 0.122626 | 0.000000 | **0.148787** | 16 | CPU 5 / BW 5 / utility 30 |

同一Slot中任务仍在相同执行节点、但CPU/BW分配改变的计数可显示共享资源重新分配伴随任务效用改变；该统计来自完整Matching/KKT反事实，不将不同任务Utility差当作单任务Cloud/Edge优劣。

资源与输入核对：所有执行的UPDATE均只改变所选目标Cache节点；分支容量校验、Matching/KKT可行及Reward恒等式检查通过。两架UAV固定停留，故Movement Cost的配对差应为0。数据输入在所有分支逐Slot复用同一指纹。

任务分配变化实例（最多24条）：

| Seed | 角色 | 更新类别 | Slot | Task | KEEP分配 | UPDATE分配 |
| ---: | --- | --- | ---: | --- | --- | --- |
| 20001 | SV | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 1 | 15 | 46 | UAV_1 |
| 20001 | SV | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 1 | 25 | CLOUD | 46 |
| 20001 | SV | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 3 | 15 | UAV_1 | 46 |
| 20001 | SV | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 1 | 15 | 46 | UAV_1 |
| 20001 | SV | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 1 | 25 | CLOUD | 46 |
| 20001 | SV | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 3 | 15 | UAV_1 | 46 |
| 20001 | RSU | MINIMUM_DELETION_LEGAL_UPDATE | 0 | 45 | RSU_0 | CLOUD |
| 20001 | RSU | MINIMUM_DELETION_LEGAL_UPDATE | 1 | 45 | RSU_0 | CLOUD |
| 20001 | RSU | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 0 | 45 | RSU_0 | CLOUD |
| 20001 | RSU | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 1 | 45 | RSU_0 | CLOUD |
| 20001 | RSU | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 0 | 15 | UAV_0 | CLOUD |
| 20001 | RSU | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 0 | 31 | UAV_0 | CLOUD |
| 20001 | RSU | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 0 | 44 | CLOUD | UAV_1 |
| 20001 | RSU | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 0 | 45 | RSU_0 | CLOUD |
| 20001 | RSU | SLOT0_PREDICTION_RECOMMENDED_UPDATE | 0 | 7 | RSU_0 | UAV_0 |
| 20001 | UAV | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 0 | 15 | UAV_0 | CLOUD |
| 20001 | UAV | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 0 | 31 | UAV_0 | CLOUD |
| 20001 | UAV | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 0 | 44 | CLOUD | UAV_1 |
| 20001 | UAV | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 0 | 50 | UAV_0 | CLOUD |
| 20001 | UAV | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 1 | 25 | CLOUD | UAV_1 |
| 20101 | SV | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 2 | 15 | 14 | UAV_1 |
| 20101 | SV | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 2 | 26 | UAV_1 | CLOUD |
| 20101 | SV | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 2 | 38 | RSU_0 | UAV_0 |
| 20101 | SV | POLICY_MAX_PROBABILITY_LEGAL_UPDATE | 2 | 58 | UAV_0 | RSU_1 |

## 4. 结论及边界

- INITIAL_CACHE_IS_DEMAND_OPTIMAL = NO（按构造逻辑，非需求最优设计）；这不证明任一Episode的C(0)比所有替代配置都差。
- INITIAL_CACHE_COVERAGE = Slot 0三Episode: 缓存命中33/41（80.49% (33/41)）；物理Edge可执行33/41（80.49% (33/41)）；实际Edge卸载21/41（51.22% (21/41)）。样本范围仅三个Slot 0。
- UPDATE_POSITIVE_NET_GAIN_COUNT = 5 / 27个去重的有效UPDATE分支，分母为实际执行且容量/Matching可行的候选；正向占比=18.52% (5/27)，定义为未来5个Slot累计ΔReturn > 0。
- MAX_POSITIVE_DELTA_RETURN = 0.562665。
- CLOUD_DOMINANCE_CAUSE = 在三个Slot 0中Cloud被选择20次：无物理Edge候选8次、有Edge候选仍由整体Matching选Cloud12次。前一类与缓存覆盖/物理可行条件有关；后一类不能单独归因为Cloud优于Edge。
- INITIAL_CONFIGURATION_LIMITS_GAIN = YES（仅在观测到正收益的对应状态）。存在至少一个正式合法替换在所测5-Slot窗口内抵消更新成本后净收益为正，支持该状态C(0)局部限制收益；不外推全局。
- CANDIDATE_GENERATION_SUSPECTED = INSUFFICIENT_EVIDENCE：只测量每个状态每个角色最多三个正式候选，没有完整动作空间Exact-Q。未从抽取结果证明候选生成错误或完备。
- NEXT_STAGE = 停止扩样。本轮已经有局部正收益证据；如需继续，仅复核正收益样例及其资源变化。

未确认的问题：5-Slot固定UAV停留不评价Cache更新与UAV移动的交互；三个Slot 0不能代表Episode整体；预测推荐的预测分数不是实际收益；即使有Edge候选，也须完整Matching/KKT后才能观察共享资源竞争结果。本轮没有PPO更新、Checkpoint写入、Validation或Test。
