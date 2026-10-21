# 节点级归一化后的 Cache 与 UAV 成本—收益分析

核对日期：2026-10-21。项目：`C:/Users/tangning/Desktop/code/adaptive_cache_mec`。本报告只读取已有代码、配置、训练日志和报告；未启动训练或评价，未读取或修改 Checkpoint。

## 当前公式与日志口径

当前合同为 `node_level_cost_upper_bounds_v1`：

\[
C^{ca}(t)=\sum_{m\in\mathcal M}\frac{T_m^{ca}(t)}{T_{m,ref}^{ca}},\quad
C^q(t)=\sum_{u\in\mathcal U}\frac{E_u^{mov}(t)}{E_{u,ref}^{mov}},\quad
G(t)=Q(t)-0.12C^{ca}(t)-0.00225C^q(t).
\]

配置初始化固定逐节点参考量，当前系统的容量和速率给出：

| 缓存节点 | 数量 | 容量 | 回传速率 | 每节点缓存参考时延 |
|---|---:|---:|---:|---:|
| SV | 5 | 12000 MB | 200,000,000 bit/s | 480 s |
| RSU | 2 | 25000 MB | 200,000,000 bit/s | 1000 s |
| UAV 缓存 | 2 | 20000 MB | 200,000,000 bit/s | 800 s |

缓存参考值按 `8e6 × 容量_MB / 回传速率_bps` 计算，是容量约束下的逐节点保守上界；参考值总和 6000 s 只用于保留权重来源，不是归一化分母。两架 UAV 各自的参考能耗为 `15 J/m × 15 m/s × 1 s = 225 J`，是速度约束下的保守上界；位置边界与安全距离可能使实际可达值更低。参考值在环境初始化时固定。

正式权重数值仍为 `beta_ca=0.12`、`beta_q=0.00225`，分别由配置中的 `lambda_cache=2e-5 s^-1` 与 `lambda_movement=5e-6 J^-1` 乘以各自参考值总和得到。逐节点分母改变了节点间及动作成本的有效影响，不能视为旧目标的数值等价改写。归一化成本的保守上界为 `Cca≤9`、`Cq≤2`，对应最大罚项上界分别为 1.08、0.0045；这些联合上界不保证可同时达到。

日志字段含义：`task_utility` 是 `Q`，`cache_cost` 已是 `0.12×Cca`，`movement_cost` 已是 `0.00225×Cq`，`G` 是系统效用；写日志时会校验 `G=Q-cache_cost-movement_cost`。`cache_download_delay_s` 是实际新增服务产生的物理时延总和。当前 Slot 日志没有逐节点的 `Tca_m`、参考值、`Cca` 分项，也没有 UAV 前后位置、逐机实际距离/能耗或 `Cq` 分项；`uav0_move/uav1_move` 是动作标签，不能替代实际位置变化。`actual_cache_update_count` 是实际发生缓存状态变化的节点数，不是服务数。日志虽记录 Cloud 与边缘任务计数，但不记录逐任务缓存命中状态。

## 可用数据及合同筛选

当前正式配置身份为 raw SHA-256 `a2286e45c7be8bb724ce91aeaebe48e0ec5188e0a43610ae264b98e7859936bfa`、semantic SHA-256 `bd16ba188992a215982cd101b0630aa3fe67ddf0802c65f3b6d7de5a3bfe362c`。现有正式训练 Run 的配置身份均不匹配：四个 LR Run 使用 raw `297822402a783b4bbbb28b1be2e61aefb1fa2cef5e110af279e814f58e4b670f`、semantic `2a4f74c8e55d6f1329c790d5265988c4e1bbf5600b1a069570b93cd3a63c01f5`；另一个 8 Episode 受控 Run 使用 raw `2798296ed073e020ab4249c68539e017fc48c01d89c07752973f7f2a2a34e003`、同一旧 semantic hash。旧 `run_config.toml` 还记载任务效用参考值 `T_ref=30.4737 s`、`E_ref=5.15746 J`，与当前正式配置不同。它们的 Slot 奖励不能用于当前合同的成本占比或效用分布。

| 已有训练日志 | 完成状态 | Slot 数 | 当前合同可用性 |
|---|---:|---:|---|
| `alternating_actor_updates_v1_macro_lr_1e-4` | 300/300 Episode，75 次更新 | 90000 | 不可用：旧配置身份 |
| `alternating_actor_updates_v1_macro_lr_5e-4` | 276/300，69 次更新 | 82800 | 不可用：旧配置身份 |
| `alternating_actor_updates_v1_macro_lr_1e-3` | 268/300，67 次更新 | 80400 | 不可用：旧配置身份 |
| `alternating_actor_updates_v1_macro_lr_5e-3` | 280/300，70 次更新 | 84000 | 不可用：旧配置身份 |
| `alternating_idmap_controlled_2updates_20261020` | 8/300，2 次更新后受控停止 | 2400 | 不可用：旧配置身份且未完成 |

四个 LR Run 的训练 seed 为 4401，Manifest 声明的 Episode seed 范围为 20001–20300；受控 Run 的训练 seed 为 27005。它们是训练日志而非当前合同下的评价数据。在已检查的正式主线 Run 和相关对照报告中，没有发现匹配当前节点级合同身份的运行结果，因此当前合同的均值、分位数、总量以及 Cache 更新率、UAV 实际移动率均为**不可计算（有效样本数为 0）**，不能把缺数据写成零。

已有配对报告 `reports/cache_uav_frozen_cooperation_evaluation.md` 对旧配置的 Checkpoint 在 4 个 Validation seed 上做了 KEEP/STAY 干预。报告记载四组动作均未造成缓存变化或 UAV 位移，成本为零，A/B/C/D 的 `Q` 与 `G` 完全相同；它没有激活任一动作，且配置 raw/semantic hash 与当前合同不符，故不能判断当前成本是否过高。

较早的 `reports/uav_stay_move_counterfactual.md` 报告了另一 Run、Checkpoint v58 上的 4 个单次 MOVE 后缀对照：3 个净 Return 增加、1 个减少，报告中的移动罚项为每次 0.001870。该 Run/Checkpoint 与当前正式主线不同；配置哈希只记在所引用的结果 JSON 中，而该文件当前不可读，因此不能核实其与当前合同的身份关系，也不能用来标定当前 `beta_q`。没有找到当前合同下完整 Matching/KKT 的 Cache REPLACE 对 KEEP 配对结果；`sv_cache_proposal_effectiveness.md` 是候选需求代理分析，并未测量全系统任务效用及缓存成本的配对变化。

## 成本—收益判断

当前合同下没有可用 Slot 或评价样本，故以下项目均无法给出有依据的均值、分位数或总量：Cache 更新 Slot 比例；SV/RSU/UAV 缓存各自的实际下载时延、归一化贡献和扣费；Cache 扣费相对 `Q` 的比例；逐机实际位移、能耗、归一化成本和扣费；动作前后的 `Q`、时延、命中率、Cloud 比例与 `G`；以及任务效用、两项罚项、系统效用的分布统计。当前资料不足以判断任一成本是否在多数情况下压过有效动作收益，也不足以判断 Cache Actor 或 UAV Actor 是否被成本压制；现有证据不支持调整权重。

参考尺度可用于解释单次成本，但不是收益证据：一个节点若达到自身缓存参考时延，其缓存罚项为 0.12；一个 UAV 若达到 225 J 参考能耗，其罚项为 0.00225。按当前参数，实际二维移动 7.5 m 对应 112.5 J、`Cq=0.5`、移动罚项 0.001125；这只是公式计算示例，不是当前合同评价观测。

## 最小补充证据

在不改权重的前提下，先需要一份配置 raw/semantic hash 与当前合同一致的冻结策略及其验证身份；随后在同一环境 seed、任务输入和初始状态下配对比较 KEEP/合法 REPLACE、STAY/实际合法 MOVE，并记录实际缓存前后状态、逐节点下载时延/参考值/扣费、逐机前后二维位置/距离/能耗/参考值/扣费，以及每 Slot 的 `Q`、两项罚项、`G`、任务时延、缓存命中与 Cloud 任务数。报告应列出每个 seed 的配对差和动作是否真实发生。没有这种当前合同数据前，保持权重不变；本报告不建议重新训练，也不据旧结果作因果推断。

源码依据：`src/common/system_model.py`、`src/decision/cache_configuration.py`、`src/decision/uav_configuration.py`、`src/decision/slot_environment.py`、`src/ppo_simple/interface.py`、`src/ppo_simple/reporting.py`；配置：`configs/proposed/alternating_actor_updates_v1.toml`。
