# 在线缓存—UAV 两阶段 Dueling Double DQN

独立算法，复用正式配置决策—Matching—KKT—实际系统效用执行链路。
每个 Slot 仅执行所选合法候选一次；候选生成、网络评分和 Replay 重建均不调用 Matching/KKT。

## 与系统模型的对应关系

依据用户提供的 12.pdf（A—F 部分，第 1—11 页）：配置决策只能读取截至上一 Slot 的历史与当前可用预测；实际任务及实际车辆位置在配置确定后进入任务调度。
奖励直接取正式执行器的单 Slot `system_utility = actual_task_utility - beta_ca * cache_cost - beta_q * movement_cost`，不重复加入预测时域效用。
缓存与 UAV 的实际配置跨 Slot 保留；当前环境每个动作推进一个 Slot，终止时不 bootstrap，截断窗口保留下一状态。

P0 优化固定长度 Episode 的平均期望系统效用。固定长度下，最大化不折扣的累计效用与该目标等价，所以独立 DQN 默认及共享 Runner 显式使用 `gamma=1.0`；保留 `gamma ** duration_slots` 接口。没有修改正式奖励、成本权重或系统配置。

本项目现行的 `KEEP_STAY` / `CACHE_UPDATE` / `UAV_MOVE` 互斥规则、缓存必须同时移除并新增服务的 REPLACE 规则，是比文档 P0 更严格的动作限制。缓存只保证全部合法单节点替换完整保留，跨节点组合和连续 UAV 位置仍为有限候选。本算法因此优化受限动作空间，不能宣称求解了 P0 的完整动作空间或其全局最优解。

## 信息与候选合同

状态复用正式 `build_step_observation()`：当前缓存、UAV 位置、5 个预测时域的车辆位置及服务流行度、有效 mask、正式时间特征，另加由服务目录计算的预测工作量摘要。真实当前任务及未来真实轨迹不进入网络或候选生成。
候选特征保存目标缓存、节点变化、服务增删、正式下载成本、UAV 前后位置、位移成本及预测距离。缓存与 UAV 变化在候选构造、适配器和执行结果三处校验互斥。

## 两阶段价值与 Double DQN

类别头输出 `V_c(s)`；候选头输出原始优势 `A(s,a)`。对当前合法类别内集合 `L_c(s)`，统一定义：

```
Q(s,a) = V_c(s) + A(s,a) - max_{b in L_c(s)} A(s,b)
```

因此 `max_{a in L_c(s)} Q(s,a) = V_c(s)`。类别头具有类别最佳候选价值的明确含义，不再把随机探索的低价值目标直接当成整个类别的监督目标。
贪心时先按 `V_c` 在可用类别中选择，再只在选定类别内按优势选择目标。探索仍先均匀选择可用类别，再在该类别内均匀选择候选，KEEP 始终可选。

下一状态在线类别头选择类别，在线候选头选择该类别中的目标；目标双头按相同聚合公式估计这个在线动作的 Q。目标网络的类别内最大优势只用于归一化，不替代在线动作选择：

```
c* = argmax_c V_online,c(s_next)
a* = argmax_{a in L_c*(s_next)} A_online(s_next,a)
y = reward + gamma ** duration_slots * Q_target(s_next,a*)
```

终止目标为实际 reward。两个在线头通过 `Huber(Q_online(s,a), y)` 的同一 TD 误差更新；不再分别将类别值与候选残差回归两个互相移动的目标。目标同步、Adam、学习率、epsilon 调度和梯度裁剪沿用共享入口。非有限 loss/梯度在优化器更新前报错。

最大值归一化借鉴 Dueling 论文公式 (8)，并按操作类别扩展；这是本项目的结构适配，不代表该论文验证过缓存—UAV 任务。最大值在排序切换处不光滑，选中类别内最高优势时该动作的优势梯度抵消，其他候选的相对价值主要通过探索样本学习；需要后续短程训练与独立评价检验实际稳定性。

## Replay、Checkpoint 与指标

Replay 保存语义动作、所选动作特征、实际奖励、时长，以及当前/下一状态的压缩预测重建上下文和候选数量/ID 摘要，不保存完整候选特征矩阵。重建当前类别候选用于优势归一化，重建下一候选用于 Double DQN。相邻转移重复的状态上下文通过引用共享，环形淘汰时释放引用，Checkpoint 恢复时重新共享。
所有候选重建均校验状态指纹、候选 ID、所选目标和特征；上下文只包含决策可见的状态与预测。

架构版本为 `category_max_centered_dueling_q_v2`；Agent schema 3、Replay schema 4、Manifest v6、Checkpoint schema 7。旧架构/旧回放明确拒绝恢复；历史文件保留。学习率、折扣、源码身份与数据身份继续参与恢复校验。

`selected_category_q` 是类别最佳 Q；`selected_candidate_q` 与 `selected_q_value` 都是所选动作完整 Q；`selected_candidate_advantage` 是归一化优势（非正）。类别候选 Q 统计采用完整 Q，`td_loss` 是统一 TD loss。旧的 `category_td_loss` / `candidate_advantage_loss` 字段保留为不可用值 `null`，不伪造两份独立 loss。
缓存变化、服务增删和 UAV 移动取实际前后状态差；成本和任务执行节点来自正式结果。预测缓存覆盖率独立标记，不能称为实际命中率或据其宣称缓存更新的因果收益。

## 来源

- CleanRL: https://github.com/vwxyzjn/cleanrl ，参考 `cleanrl/dqn.py`，commit `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`。
- 复用与动作空间无关的在线/目标网络初始化、回放随机采样、epsilon 调度、优化与同步骨架；未复制完整仓库或引入 Gymnasium/Tyro/TensorBoard。MIT 声明保留在 `CLEANRL_LICENSE.txt`，Copyright (c) 2019 CleanRL developers。
- Dueling: https://proceedings.mlr.press/v48/wangf16.pdf ，公式 (8) 的最大优势归一化。
- Double DQN: https://arxiv.org/abs/1509.06461 ，在线选择、目标网络估值。

## 入口与使用

共享入口 `python -m online_joint_dqn.run_training`。四个 `scripts/training/run_online_joint_dqn_lr_*.py` 入口仍使用同一训练循环并只覆盖各自学习率。
输入、公式和回放定向检查不代表训练收敛或收益改善；真实 Slot 与训练仅在明确授权后运行。
