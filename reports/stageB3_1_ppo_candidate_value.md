# Stage B3-1：PPO Cache Candidate Value Recognition

## 结论摘要

本次完成Episode 20007、Slot 0服务2预测新增需求特征的溯源，并固定复用B2-8的三个Training Episode候选长期净收益标签。现有PPO策略检查被Checkpoint合同不匹配阻断：当前可确认的Stage6权重属于旧的无迁移兼容版本，不能用于当前源码的策略推理或继续训练。因此没有策略概率、候选排序或短程学习结果；不加载旧权重、不部分加载，也未启动训练。

**服务2预测需求归零不是当前缓存状态屏蔽未缓存服务的结果。** GRU预测及五步任务需求均为正；归零来自Cache经济特征中的节点预测可服务性软权重：对RSU_1而言，所有五步的预测车辆平均位置均在特征半径250米之外，故权重全部为零。该半径是候选评分代理，不是Matching的硬通信覆盖条件。

## 1. 执行环境与服务2需求特征溯源

使用本机现有Python 3.12.7及PyTorch 2.13.0+cpu读取预测输入、策略/Checkpoint元数据并核对当前源码；没有安装或升级依赖。Conda环境在本机此前激活失败（系统错误9020），本次未改动环境。

固定候选为Episode 20007、Slot 0的B2-8规则B候选：RSU_1 `{5,7,9} → {2}`。它在正式候选集合中合法，完整目标配置与节点身份匹配为候选#121。Slot 0的五步预测有效掩码均为True。

| 预测步（从Slot 0起） | 服务2原始GRU流行度 `rho_pred` | 场景平均服务2需求代理 | RSU_1最近预测车辆距离（均值坐标） | 250米内正可服务权重数 / 权重和 | 最终新增需求特征 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.082648 | 1.60 | 666.697 m | 0 / 0 | 0 |
| 2 | 0.081090 | 1.10 | 683.116 m | 0 / 0 | 0 |
| 3 | 0.079810 | 1.05 | 695.548 m | 0 / 0 | 0 |
| 4 | 0.081714 | 1.25 | 667.486 m | 0 / 0 | 0 |
| 5 | 0.082478 | 1.15 | 639.378 m | 0 / 0 | 0 |

表中场景平均需求为现有预测场景对20辆任务车辆及场景样本的平均聚合需求代理；不是未来真实任务数。RSU_1的服务性权重由预测车辆平均位置到节点的距离按250米半径线性衰减计算。五步最近距离仍至少约639米，因此每辆车的该项权重均为零，服务2的节点加权需求归零。

实现链路为：`predicted_vehicle_service_demand_by_horizon` 生成各服务、各预测步的车辆需求；`_node_predicted_serviceability_weights` 依据预测位置和节点评分半径产生节点权重；`cache_economic_candidate_features` 用新增服务集合聚合节点加权需求，形成200维候选输入中的经济特征。生成需求和节点权重的函数不以当前Cache命中作为条件；候选B的新增服务集合确实包含服务2。因此没有证据表明代码错误地用“当前未缓存”屏蔽了假设更新后的服务需求。

**解释边界：** 这是候选经济特征中的几何可服务性代理。该250米半径不等于实际Matching链路的硬覆盖约束，也不能据此断言物理层无法通信。零值表示该候选特征在这项代理下没有给RSU_1计入预测需求，不表示GRU预测需求为零或任务不可执行。代码位置：`src/decision/cache_configuration.py` 中 `_serviceability_weight`、`predicted_vehicle_service_demand_by_horizon`、`_node_predicted_serviceability_weights`、`node_specific_predicted_demand`；`src/ppo_simple/primitive_control.py` 中 `cache_economic_candidate_features`。

## 2. 固定B2-8反事实标签

以下结果只代表Slot 0执行一次UPDATE、之后100槽固定Cache且UAV STAY的Training反事实，不是PPO闭环动作Q值。候选按节点身份和完整目标配置识别，不把历史候选序号当作跨Episode稳定身份。

| Episode | A候选（节点与目标变化） | A的100槽ΔG | KEEP | B候选（节点与目标变化） | B的100槽ΔG | 标签排序 |
|---:|---|---:|---:|---|---:|---|
| 20007 | UAV_0 `{5,8}→{1,4,9}`（历史#194） | +0.537131 | 0 | RSU_1 `{5,7,9}→{2}`（历史#121） | −2.028440 | A > KEEP > B |
| 20008 | SV_23 `{8}→{9}`（历史#11） | −0.062523 | 0 | RSU_1 `{5,7,9}→{2}`（历史#121） | −1.902746 | KEEP > A > B |
| 20009 | SV_17 `{8}→{4,5}`（历史#41） | −0.114874 | 0 | RSU_1 `{5,7,9}→{2}`（历史#121） | −0.448822 | KEEP > A > B |

这些标签证明已测候选存在不同的事后长期净收益，但不能作为动作Q值或训练监督标签，也没有覆盖全部合法UPDATE候选。

## 3. 当前PPO与Checkpoint兼容性

| 项目 | 当前本地正式源码/配置 | 可确认的Stage6 Checkpoint与运行清单 | 兼容性 |
|---|---|---|---|
| Policy schema / Architecture ID | `adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v3_no_tv_local` | `adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v2` | 不兼容 |
| Architecture version | 32 | 31 | 不兼容 |
| Checkpoint version | 59 | 58 | 不兼容 |
| Observation维度 | 400 | 400 | 相同，但不足以证明兼容 |
| Macro Cache候选特征维度 | 200（181基础+19经济特征） | 200 | 相同，但特征语义/策略版本不同 |
| 配置原始SHA-256 | `f3131efb7832efa925c181d9abd07bec5f2bfe3e7507eca2cc2515b77067699b` | `684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77` | 不匹配 |
| Episode Pool记录 | 当前本地Training池 | `/home/9900299/tn/1/environment_data/proposed_joint_training` | 路径/来源合同不匹配 |

当前源码在 `src/ppo_simple/policy.py` 声明Architecture version 32及v3 no-TV-Local schema，Macro候选Actor将Observation与每个200维候选分别编码后评分并进行正式基数校正；训练器要求Checkpoint version 59。现有Stage6清单指向version 58、Architecture version 31及旧schema。已发现的16份run manifest均记录旧版schema/version；没有找到当前v3/version 32的兼容Checkpoint。虽Observation和候选维度碰巧相同，版本、配置哈希、数据池合同和特征语义仍不匹配，不能据形状相同加载权重。

策略实际概率检查未执行：没有从兼容Checkpoint重建正式策略，也没有将旧权重用于当前策略。因此下列结果均为**未确定**：KEEP/UPDATE概率、候选A/B条件概率及排序、A/B完整动作概率、全体UPDATE排名、确定性策略动作，以及Scorer是否能区分A与B。静态网络结构只能说明存在候选评分器，不能说明其已学会区分候选价值。

## 4. 短程PPO学习验证

未执行Episode 20010–20015上的续训（0 Epoch）。首要前提“存在与当前正式源码兼容的Checkpoint”不成立，且旧Checkpoint的版本、配置哈希和Episode Pool合同均不匹配。为避免强行加载旧权重、部分加载或覆盖正式实验，没有创建临时训练输出，也未读取B2-8标签作为训练监督。

故没有短程策略变化、Macro Actor Loss、KL、Entropy或Critic指标可比较；不能判断6个Epoch是否会改善候选识别，也不将静态反事实标签与策略偏好不一致解释为PPO信用分配Bug。

## 5. 判断与下一步

- 服务2需求代理为零的原因已定位为RSU_1预测几何权重全零，不是当前Cache屏蔽新增服务需求的错误。
- B2-8在三个Episode中确认了候选价值差异，但当前策略输出不可测；PPO是否识别正收益A、是否在20008/20009倾向KEEP、是否区分A/B均未确定。
- 现有证据不支持直接修改PPO信用分配，也不足以证明候选Scorer存在缺陷；更不能据此判断9节点独立UPDATE的学习效果。
- 建议先为当前v3 no-TV-Local、Architecture version 32及Checkpoint version 59建立经过正式配置/数据合同验证的兼容策略Checkpoint，再对20007–20009做确定性策略探针。完成策略输出探针后，若差异确实存在，再决定是否进行限定Episode的短程继续学习。服务性评分半径是否需调整属于单独的特征/模型选择问题，本次未改动。

## 状态

```ini
STAGE_B3_1 = PARTIAL
SERVICE2_ZERO_DEMAND_CAUSE = RSU_1预测车辆均在250m特征服务性半径之外，节点加权新增需求为0；非当前未缓存掩码
FEATURE_IMPLEMENTATION_BUG = NO
COMPATIBLE_CHECKPOINT_FOUND = NO
PPO_POLICY_PROBE_COMPLETED = NO
PPO_CANDIDATE_RANKINGS = UNDETERMINED（无兼容策略权重）
PPO_KEEP_UPDATE_PREFERENCES = UNDETERMINED（无兼容策略权重）
PPO_SHORT_TRAINING_EXECUTED = NO
PPO_SHORT_TRAINING_EPOCHS = 0
CANDIDATE_VALUE_LEARNING_SUPPORTED = UNDETERMINED
NEXT_RECOMMENDED_STEP = 建立当前v3/版本32兼容Checkpoint后再做策略探针；随后再决定短程续训或9节点动作改造
SYSTEM_MODEL_CHANGED = NO
FORMAL_TRAINING_CHANGED = NO
```
