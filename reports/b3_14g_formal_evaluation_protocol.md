# B3-14G：正式 PPO 评价与 Checkpoint 选择协议

## 状态与适用范围

本协议适用于四组 Macro Actor 学习率（`1e-4`、`5e-4`、`1e-3`、`5e-3`）的后续正式训练结果、Checkpoint 选择及最终独立 Test。协议不改变 PPO、Reward、系统模型或现有评价入口。

本次正式研究设计已确定：在线部署和 Primary 评价均采用 deterministic，Secondary 评价采用 stochastic，最终 Test 采用 deterministic。该决定是本项目的正式实验设计选择。它与现有 Evaluator 及主线 Runner 的确定性 Validation 路径一致。

## 正式执行模式

| 用途 | 模式 | 规则 |
| --- | --- | --- |
| 在线部署与 Primary：Checkpoint 选择 | `deterministic` | 对合法动作按当前策略的确定性选择规则执行；保留候选 Mask、Holding 边界和 UAV 条件决策顺序。 |
| Secondary：策略分布与随机行为补充分析 | `stochastic` | 按正式策略分布采样，使用相同合法动作 Mask、Cache 基数校正、Holding 边界及 UAV0→UAV1 条件采样。结果单独报告，不与 deterministic 结果合并，也不参与主要 Checkpoint 选择。 |
| 最终 Test | `deterministic` | 使用已选定的学习率和 Checkpoint，在100个独立 Test Episode上执行一次最终评价。 |

正式设计决定：`ONLINE_DEPLOYMENT_MODE = deterministic`。代码事实：正式 Evaluator 的默认模式为 deterministic，主线 Runner 的 Checkpoint Validation 也显式传入 deterministic；本次将相同模式确定为在线部署及最终 Test 合同。

核对依据：`src/ppo_simple/evaluator.py` 的函数参数与 CLI 默认值为 deterministic；`scripts/training/mainline_runner.py` 的正式 Checkpoint Validation 显式传入 `policy_action_mode="deterministic"`。没有发现单独规定线上动作选择模式的正式部署入口。策略库函数的默认采样参数属于调用接口默认值，不能据此确定部署合同。

确定性动作细则沿用当前策略实现：Cache 按经过基数校正的 KEEP/UPDATE 组概率选择组；选择 UPDATE 时，再在合法候选中按当前策略分数选择候选。Holding 按合法 CONTINUE/TERMINATE 组概率选择，并遵守 1–5 Slot 限制。UAV 按合法 STAY/MOVE 组概率选择，再按条件候选规则选择动作；UAV1 的候选条件于 UAV0 已选位置。Stochastic 评价按对应正式策略分布采样。两种评价模式均使用现有 Mask、合法候选和决策顺序。

**训练采样保持随机。** PPO Training 继续使用现有随机策略采样及训练机制；deterministic 仅用于在线部署、Primary Validation 和最终 Test，不将训练动作选择改为 argmax。

确定性与随机模式均须处于推理状态，不执行梯度或优化器更新。Stochastic 补充评价预先固定每个环境的重复次数和策略 RNG 映射；环境 Episode Seed 与策略采样 Seed 分开管理，不同 Checkpoint 复用同一映射。每个 Validation Episode 固定 **5 次策略采样**：Episode 在数据池中的序号为 `i=0..29`，重复序号为 `r=0..4`，策略采样 Seed 使用 `141401 + 30*r + i`。这与 B3-14F 已用过的五个 Episode 在首轮重复上的 Seed 映射一致。环境 Seed 为 `30001+i`。Stochastic 结果仅用于补充分析，并只在该模式内部汇总，不参与主要 Checkpoint 选择。

## 目标和指标

系统目标为：

```text
Episode Return = Episode Task Utility − Episode Cache Cost − Episode UAV Movement Cost
```

主要 Checkpoint 选择指标为 **30 个固定 Validation Episode 的 Mean Episode Return**。同时报告 Episode Return 的样本标准差（`ddof=1`）以及按相同环境 Seed 计算的逐 Episode 配对差值。不得使用单个 Episode 的最高 Return 选模，也不得用 Training Return、Critic EV、Entropy 或 KL 替代实际系统表现。

每个策略至少同时报告：

- Mean Task Utility、Mean Cache Cost、Mean UAV Movement Cost。
- 平均任务时延、平均车辆能耗。
- Cache 实际配置更新频率及 Holding 时长。
- UAV 实际移动率、移动距离及移动成本。
- Cloud、SV、RSU、UAV 的任务卸载比例。
- Mean Episode Return、Return 标准差和逐 Seed 配对差值。

Return 先按 Episode 汇总，再对 30 个 Episode 等权求均值。辅助指标也先按 Episode 使用统一分母计算，再对 Episode 等权求均值；任务时延和车辆能耗按该 Episode 的任务数求平均。Holding 时长按完整 Holding 区间统计；Cache 更新率按节点级决策和 Macro 帧分别统计；卸载比例按任务数统计；UAV 距离同时报告每个 UAV 决策的平均距离和每 Episode 总距离。所有策略沿用相同字段定义与分母，并在结果中注明统计量。任何硬资源或物理可行性检查失败、或出现 NaN/Inf 的评价结果都必须显式标记，不能作为合格 Checkpoint 静默参与选择。

## 数据划分与隔离

| 用途 | 数据池 | Episode Seed | 规则 |
| --- | --- | --- | --- |
| PPO Training | `environment_data/proposed_joint_training/` | `20001–20300`，300 Episode | 仅用于训练与 PPO 参数更新。每 4 Episode 更新一次，共 75 组更新。 |
| Checkpoint Selection | `environment_data/proposed_joint_validation/` | `30001–30030`，30 Episode | 使用 deterministic Mean Episode Return 选择学习率和 Checkpoint；所有候选使用相同场景及协议。 |
| 最终独立评价 | `environment_data/proposed_joint_test/` | `40001–40100`，100 Episode | 学习率及 Checkpoint 完全确定后，使用 deterministic 执行最终独立评价；不得用于调参或 Checkpoint 选择。 |

不得重新生成或混用数据池。所有算法及学习率使用相同场景、Episode 次序和评价口径。此前 B3-14E 与 B3-14F 使用的 `30001–30005` 是 Validation 开发子集，属于已查看样本；这些结果保留为探索性证据，不称为未使用的独立验证结果。

## Checkpoint 保存与选择

四组正式比较固定使用相同架构（Architecture 33、Checkpoint 60、策略Schema `adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v4_multi_node_cache_factorized_v1`）、系统模型、Reward、Training Pool、PPO 初始化 Seed `27005` 及 300 Episode 预算；每 4 Episode 完成一组更新，`update_epochs=4`，目标为完整提交 75 组更新。每组应保留初始化 Checkpoint、每 10 组完整更新的 Checkpoint（Update 10、20、30、40、50、60、70）及最终完整 Update 75 Checkpoint。中断或未完整提交的更新状态不视为候选。只评价真实存在且合同有效的 Checkpoint，保留全部 Checkpoint、Manifest 与原始结果，不覆盖历史文件。

选择步骤：

1. 每个学习率组内，对初始化、已完成的周期 Checkpoint 和最终 Checkpoint 使用相同 30 个 Validation Episode，以 deterministic 模式评价。
2. 每组按 30 Episode 的 Deterministic Mean Episode Return 选择最高的有效 Checkpoint。并列或均值接近时，报告逐 Seed 配对差值、样本波动及成本指标；不设无依据的最小差异阈值，也不把微小均值差表述为显著优势。
3. 四组分别完成组内选择后，再比较四组入选模型；并结合相同更新预算的对照（尤其是同为 Update 10 的学习率对比）报告结果。若某组尚未完成 300 Episode / Update 75，只能报告阶段性观察，不能宣布最终最佳学习率。
4. 学习率和 Checkpoint 完全确定后，使用 deterministic 模式在独立 Test 池进行最终评价。Test 结果不得反向用于选择模型、调整参数或更改协议。

Stochastic 是补充行为分析，不替代 Primary Checkpoint 选择指标。它只对四组经 Primary 选出的 Checkpoint 和共享初始化策略执行，不对每个周期 Checkpoint 全量重复评价。Stochastic 与 deterministic 分表、分模式统计，不计算跨模式混合平均值。

## Cache 与 UAV 指标口径

报告不得把字段名 `keep_rate` 直接解释成逐节点 KEEP 动作率。分别统计以下量，并注明分子、分母与层级：

1. **Cache KEEP 组概率**：Macro 决策中策略分配给 KEEP 组的概率质量。
2. **Cache 逐节点 KEEP 动作率**：节点级 Cache 决策中采到/选到 KEEP 的次数除以节点级决策总数。
3. **Cache 实际配置变化率**：发生实际缓存内容变化的节点决策数除以节点级决策总数。
4. **无变化 Macro 帧比例**：所有节点均未发生缓存内容变化的 Macro 帧数除以 Macro 帧总数。
5. **UAV MOVE 组概率**：策略分配给 MOVE 组的概率质量。
6. **UAV 非 STAY 动作率**：采到/选到非 STAY 候选的次数除以 UAV 决策总数。
7. **UAV 实际非零位移率**：坐标实际发生非零变化的 UAV 决策数除以 UAV 决策总数。
8. **UAV 距离与成本**：分别报告实际移动距离和进入 Return 的 UAV Movement Cost。

动作概率、动作选择和物理执行结果是不同统计量。尤其不能用 MOVE 候选率代替实际位移率，也不能用 KEEP 组概率代替节点级 KEEP 动作率。

## B3-14E/F 结果的使用

B3-14E 的 deterministic 结果和 B3-14F 的 stochastic 结果作为已完成的探索性结果保留，可用于解释确定性选择与随机采样下的行为差异。它们覆盖的五个 Validation Episode 已被查看，样本量也不足以证明长期收敛、泛化优势或最终最优学习率。后续正式比较按本协议使用完整 30 Episode Validation；不得将 B3-14E 与 B3-14F 的不同模式结果混为同一统计总体。

## 冻结状态

在线部署、Primary Validation、Secondary Stochastic 补充评价和最终 Test 模式均已按本次研究设计确定。数据划分、指标口径、Checkpoint 选择步骤及 Stochastic Seed 映射一并冻结。B3-14E/F 已有结果继续保留为探索性证据，不重新运行，也不作为独立未使用验证样本。
