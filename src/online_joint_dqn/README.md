# 在线缓存—UAV 两阶段 Dueling Double DQN

独立算法，复用正式配置决策—Matching—KKT—实际系统效用执行链路。
每个 Slot 仅执行所选合法候选一次；候选生成、网络评分和 Replay 重建均不调用 Matching/KKT。

### Matching、资源与终止目标

只对新到达任务搜索节点；活动任务的执行节点固定到任务终止。Transfer-Exchange 以当前时隙由该候选和资源分配产生的终止任务效用作候选目标，并记录局部稳定状态；它不估计未终止任务的未来价值，也不宣称全局最优。Edge KKT 仍求解每节点、按任务阶段筛选的倒数时延/能耗代理问题；有限时域终止奖励含有截止期事件与跨时隙剩余量，因此该 KKT 解不是论文 P0 的最优解。Cloud 采用每个任务独立的固定 `R_c/F_c`，不建立共享 Cloud 资源池，也不运行 Cloud KKT。

## 与系统模型的对应关系

依据项目最新 `13.pdf`：配置决策只读取当前可用预测和已知状态；Matching 使用当前时隙真实车辆位置与已到达任务，不能读取未来真实轨迹或请求。
系统每个扩展时隙的奖励为 `G(t)=Q(t)-C_ca(t)-C_q(t)`。`Q(t)` 只累计在该时隙末成功或超时的任务终止效用，每个任务只结算一次；缓存和 UAV 成本系数按论文目标均为 1，不使用旧配置中的 `beta_ca/beta_q`。
缓存与 UAV 的实际配置跨 Slot 保留；当前环境每个动作推进一个 Slot，终止时不 bootstrap，截断窗口保留下一状态。

P0 优化固定长度 Episode 的平均期望系统效用。固定长度下，最大化不折扣的累计效用与该目标等价，所以独立 DQN 默认及共享 Runner 显式使用 `gamma=1.0`；保留 `gamma ** duration_slots` 接口。没有修改正式奖励、成本权重或系统配置。

本项目现行的 `KEEP_STAY` / `CACHE_UPDATE` / `UAV_MOVE` 互斥规则、缓存必须同时移除并新增服务的 REPLACE 规则，是比文档 P0 更严格的动作限制。DQN 候选完整保留正式规则枚举出的全部合法单节点替换。另将有界 `CacheOptimizer` 提议加入候选，其中至多包含一个“各活跃节点首个提议”的合并配置，其余单节点提议按 Slot 轮转的节点顺序和提议秩加入；这类提议的生成上限也单独记录。双节点提议先为每个节点按预测/覆盖提议来源优先级选一个代表；同优先级候选以来源状态指纹绑定的 SHA-256 排序打破平局。再对全部代表节点对按状态指纹与规范化节点/服务配置生成稳定优先级，之后才执行最多扫描 32 对、最多保留 8 个新配置的限制。诊断分开记录代表节点对总数、已排名数、实际扫描数、合法通过数、拒绝数、重复数、保留数、保留上限遗漏数及未扫描数；扫描预算与保留预算造成的截断会标记为不完整。`cache_cross_node_candidates_retained_count` 统计所有发生多节点变化的缓存目标，可能包括上述一个合并配置；`cache_cross_node_pair_candidates_retained_count` 只统计双节点提议保留数。排序不依赖节点/服务数组遍历顺序，但每节点仅一个双节点代表，因此该提议空间仍是有界启发式集合。较高阶跨节点组合和连续 UAV 位置仍未穷举。本算法因此优化受限动作空间，不能宣称求解完整动作空间或全局最优解。

## 信息与候选合同

状态复用正式 `build_step_observation()`：当前缓存、UAV 位置、预测时域的车辆位置及服务流行度、有效 mask、正式时间特征，另加候选需求摘要和活动任务状态。活动任务按规范 TV 顺序、到达时隙和任务 ID 排列，并按 `K_max` 固定填充；每项包含阶段、原始单位的剩余上传 bit、剩余计算 cycle、剩余期限 slot、相对剩余比例、服务 one-hot 和执行节点 one-hot。当前时隙新到达请求在决策前以“未分配节点”状态进入观测；已活动任务保留其固定节点。候选生成仍不接收任务对象或当前真实车辆位置，活动任务状态只进入 DQN 状态特征。
候选特征保存目标缓存、节点变化、服务增删、正式下载成本、UAV 前后位置、位移成本及预测距离。新增的每服务互补性向量依次编码 UAV 局部预测需求、该需求中固定节点的预测加权可覆盖量、固定节点预测覆盖缺口、删除服务对应的 UAV 局部需求、候选的全网预测流行度净变化、变更服务已有副本比例及副本数变化；末尾记录预测缓存覆盖率的可用标记、更新前后值和差值。服务大小、下载量和成本继续复用原候选字段，不重复造特征。

空间需求和固定节点覆盖复用正式 `cache_configuration` 的预测需求与节点服务性权重，使用有效预测时域及预测车辆位置；UAV 位置使用当前决策可见位置，在未来预测时域不假设 UAV 已移动。该权重是正式候选生成/预测覆盖中的软几何信号，不等同于真实 Matching 的可达结果或任务命中率。输入缺失会报错，预测覆盖无有效需求时由显式可用标记区分，不伪造零覆盖。候选集合仍由既有合法 REPLACE 生成器产生，没有额外枚举动作。

## 两阶段价值与 Double DQN

类别头输出 `V_c(s)`；候选头输出原始评分 `A(s,a)`。对当前合法类别内集合 `L_c(s)`，统一定义：

```
Q(s,a) = V_c(s) + A(s,a) - mean_{b in L_c(s)} A(s,b)
```

因此 `V_c(s)` 等于该类别合法候选的平均 Q 值，而不是类别内最佳候选 Q。贪心时在 `KEEP_STAY`、`CACHE_UPDATE`、`UAV_MOVE` 的全部合法候选间比较完整 `Q(s,a)`；不能只比较类别头。完全相等的 Q 值使用候选 ID 稳定打破并列，避免候选遍历顺序决定动作。探索仍先均匀选择当前可用类别，再在该类别内均匀选择候选，KEEP 始终可选但不被强制执行。

下一状态在线网络在全部合法候选间选择最大完整 Q；目标网络只评估同一个候选，并使用该候选类别的合法集合均值归一化：

```
a* = argmax_{a in L(s_next)} Q_online(s_next,a)
y = reward + gamma ** duration_slots * Q_target(s_next,a*)
```

终止目标为实际 reward。两个在线头通过 `Huber(Q_online(s,a), y)` 的同一 TD 误差更新。均值中心化下，所选候选即使是类别内最高分，也具有 `1 - 1/|L_c|` 的直接优势梯度系数；单候选类别的中心化优势恒为零，类别价值头仍学习该动作价值。目标同步、Adam、学习率、epsilon 调度和梯度裁剪沿用共享入口。非有限 loss/梯度在优化器更新前报错。

按类别合法集合做均值中心化是为动态候选集所作的 Dueling 结构适配；这不代表相关论文已验证缓存—UAV 任务。候选越多，存在价值估计噪声极值的理论机会通常越大；类别均值中心化不会校正最大值噪声，Double DQN 主要分离 TD bootstrap 的选择与估值，也不消除行为策略贪心选择中的全部偏差。当前没有独立训练/评价数据足以估计各类别的误差分布，因此未添加候选数惩罚或人为 Q 校正。逐 Slot 日志记录各类候选数、完整 Q 与中心化优势的均值/标准差/分位数/极值、类别均值价值、最大 Q 对应候选、探索/贪心来源、实际奖励及实际配置变化；Episode 汇总记录候选总体矩及逐状态最大值/分位数的分布。所有决策排序、TD target、统计和实际贪心选择均采用同一完整 Q 定义；是否提升策略仍需真实训练和独立评价确认。

## 跨 Episode 车辆槽位与 Replay 重建

正式 ObservationSpec 将车辆轴按 SV、TV 分组，但组内保留各 Episode 预测 NPZ 的车辆顺序；Training Pool 不保证不同 Episode 的车辆身份或数组序号相同。DQN 在每个 Episode 第一个有效决策预测上分别对 SV、TV 建立稳定空间槽位：先按首个有效预测坐标 `(x, y)`，再按该车辆其余有效预测坐标及预测服务需求向量排序；对 SV 来说，如预测签名也相同，则继续按当前缓存行区分；全部决策可见特征仍相同时，才以该 Episode 的正式组内 NPZ 顺序打破平局。该顺序在整个 Episode 固定，不把不同 Episode 的物理车辆视为同一身份。

状态中的预测车辆轴、缓存节点行、预测候选需求及互补性特征均使用 Episode 规范槽位。缓存动作仍保存并执行正式环境中的真实节点 ID 和源矩阵行顺序；重建前以 Episode 上下文还原对应编码器，动作不通过跨 Episode 车辆名或数组位置重新解释。状态/候选合同和网络架构合同均已更新；旧网络 Checkpoint 因输入语义合同不同而拒绝恢复。

Replay 每条上下文现在包括 Episode 身份、规格摘要、源车辆/任务车辆/缓存节点 ID 顺序及规范槽位映射。Replay Buffer 注册按 Episode 身份隔离的轻量 `EpisodeCandidateRebuildAdapter`，只保存该 Episode 的系统模型、ObservationSpec、编码器、搜索预算及映射，不保存完整环境、任务序列或车辆轨迹；经验引用其上下文时保留对应重建器，环形覆盖后释放未再使用的 Episode 重建器。候选重建按上下文身份派发，并核验状态指纹、状态特征、候选数量和有序 ID、动作真实节点映射、特征维度与数值。Checkpoint 恢复先在暂存对象中校验 Replay schema、每个 Episode 重建器摘要和全部上下文，再一次性替换状态；旧 Replay schema 不兼容时明确拒绝。

Replay 保存语义动作、所选动作特征、实际奖励、时长，以及当前/下一状态的压缩预测重建上下文和候选数量/ID 摘要，不保存完整候选特征矩阵。相邻转移重复的状态上下文通过引用共享，环形淘汰时释放引用；终止转移仍不 bootstrap。

状态合同 `formal_observation_spatial_role_slots_v4`、候选特征合同 `prediction_only_cache_spatial_role_slots_v4`、候选生成合同 `state_ranked_bounded_pair_cache_replace_v2`、网络架构版本 `legal_candidate_mean_centered_dueling_q_v5`、Agent schema 5、Replay schema 7、Manifest v7、Checkpoint schema 8。候选重建上下文采用 v7，Episode 重建规格采用 `dqn_episode_candidate_rebuild_spec_v1`；旧状态特征、候选特征、网络架构、候选上下文或 Replay 合同不会静默迁移。训练身份继续包含特征/架构版本、源码与数据身份。

Replay 保存语义动作、所选动作特征、实际奖励、时长，以及当前/下一状态的压缩预测重建上下文和候选数量/ID 摘要，不保存完整候选特征矩阵。重建当前类别候选用于优势归一化，重建下一候选用于 Double DQN。相邻转移重复的状态上下文通过引用共享，环形淘汰时释放引用，Checkpoint 恢复时重新共享。
所有候选重建均校验状态指纹、候选 ID、所选目标和特征；上下文只包含决策可见的状态与预测。

旧候选上下文 v6、Replay schema 6 及更早合同在恢复阶段明确拒绝，不会修改已有智能体、Replay 或运行目录。学习率、折扣、完整源码身份与数据身份继续参与恢复校验。

`selected_category_q` 是类别均值 Q（类别头输出）；`selected_candidate_q` 与 `selected_q_value` 都是所选动作完整 Q；`selected_candidate_advantage` 是类别均值中心化优势，可正可负且类内均值为零。逐 Slot `candidate_q_statistics_by_operation` 统计完整 Q，而 `candidate_advantage_statistics_by_operation` 单独统计中心化优势；类别候选数量和价值也分开记录。`td_loss` 是统一 TD loss。旧的 `category_td_loss` / `candidate_advantage_loss` 字段保留为不可用值 `null`，不伪造两份独立 loss。
缓存变化、服务增删和 UAV 移动取实际前后状态差；成本和任务执行节点来自正式结果。预测缓存覆盖率独立标记，不能称为实际命中率或据其宣称缓存更新的因果收益。

## 来源

- CleanRL: https://github.com/vwxyzjn/cleanrl ，参考 `cleanrl/dqn.py`，commit `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`。
- 复用与动作空间无关的在线/目标网络初始化、回放随机采样、epsilon 调度、优化与同步骨架；未复制完整仓库或引入 Gymnasium/Tyro/TensorBoard。MIT 声明保留在 `CLEANRL_LICENSE.txt`，Copyright (c) 2019 CleanRL developers。
- Dueling: https://proceedings.mlr.press/v48/wangf16.pdf ，公式 (8) 的最大优势归一化。
- Double DQN: https://arxiv.org/abs/1509.06461 ，在线选择、目标网络估值。

## 入口与使用

共享入口 `python -m online_joint_dqn.run_training`。四个 `scripts/training/run_online_joint_dqn_lr_*.py` 入口仍使用同一训练循环并只覆盖各自学习率。
输入、公式和回放定向检查不代表训练收敛或收益改善；真实 Slot 与训练仅在明确授权后运行。
