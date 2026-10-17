# 联合优化研究实现（独立分支 v1）

## 固定物理合同

- TV 仅产生和上传任务，不本地执行；任务执行节点为 SV / RSU / UAV / Cloud。
- Edge 执行必须满足服务缓存与当前无线链路可行性；Cloud 在 RESOURCE_AWARE_FALLBACK 下始终为合法候选。
- Edge 共享 CPU 和通信带宽；通信速率遵从正式 BANDWIDTH_DEPENDENT_NOISE_V1。
- Cloud 每任务独立 20 Mbps 上行，上传时延因子 13；所有 Cloud 任务共享总计 40 GHz 的有限 CPU 池。
- 每个 slot 的任务独立完成解析评价，不引入跨 slot 排队、任务残留或新的 deadline。
- 真正的净效用恒为 G=Q-C_cache-C_UAV；Cache 和 UAV 的实际配置状态可跨 slot 延续。

## 数学拆分与正确性边界

设 V 为当前到达任务集合，x_{vm} 为任务执行节点指示，w_{vm}、b_{vm} 为相应 CPU/带宽份额。
固定缓存 c、UAV 水平位置 q、任务—节点选择 x 时：

- Edge CPU 成本为 sum_v A_{vm}/w_{vm}，其中 A_{vm}=omega_T L_v/(T_ref F_m)，约束 sum_v w_{vm}<=1、w_{vm}>=epsilon_w。该子问题凸。下界不活跃时，可用 w* = sqrt(A)/sum sqrt(A) 和 F_CPU* = (sum sqrt(A))^2 精确消元。
- 带宽相关噪声的速率为 R(b)=bB log2(1+p g/(PSD_eff bB))，其 R'>0、R''<0，因此 sum_v H_v/R(b_{vm}) 在 b>0 上为凸目标，约束 sum b<=1、b>=epsilon_b。对应单调 KKT 求根器已在 task_allocation 中实现。
- 有限共享 Cloud CPU 同样具有凸的 CPU 分配结构，上传保持每任务固定等效速率。
- 任务节点选择 x 仍为离散、非可加分组成本优化，不能因连续子问题凸而声称整个问题凸或全局易解。

## 新增文件与边界

1. `src/decision/assignment_optimizer.py`
   - 完全独立于 Transfer-Exchange Matching 的执行节点搜索；合法目标和实际物理效用沿用正式评估器。
   - 小任务集合：在预算内做完整分支限界搜索；只有完整遍历并未被预算打断才报告容差意义上的条件全局最优。
   - 大任务集合：对高资源竞争任务作三任务联合重分配；此模式 **不宣称全局最优**。
   - 上界：每任务“单独占用全部合法节点资源”的最佳效用之和；在固定通信增益、无任务间干扰和独立时隙条件下是合法上界，通常偏松。
   - 返回 best feasible assignment、资源分配、效用、搜索预算消耗、上界及残余保守 gap。
   - 目前评价器仍调用现有 CPU/带宽 KKT：替换的是任务选择机制，**不是假装完全移除了 KKT**。

2. `src/decision/predictive_assignment_value.py`
   - 从预测场景文件已有 [S,H,V] arrival/service_id/input_size/cpu_cycles 和 [S,H,vehicles,2] positions_xy 构造 **预测任务**。
   - 对同一 Cache/UAV 状态逐场景调用上述内层，输出折扣后的预测任务效用；绝不读取未来真实任务。
   - 不在此层扣配置成本，避免重复扣费。

3. `src/decision/joint_held_optimizer.py`
   - 接收外部生成的 Cache 状态候选集和 UAV 完整位置候选集，枚举物理可行联合配置（包括 KEEP+STAY）。
   - 利用生产约束函数核对 Cache 容量、UAV 区域/单时隙速度/安全距离，并只计一次当前 Cache/UAV 配置成本。
   - 所有候选共享相同的预测场景子集；第一阶段低预算筛选，第二阶段对入围候选使用同一较大预测/求解预算。
   - KEEP+STAY 一定进入完整评价，未超过其预测净效用阈值则不执行更新。
   - **当前是“未来 H 时隙保持本次配置”的滚动时域首动作原型，尚不是多阶段带未来追索决策的完整随机 MPC。** 不宣称真实轨迹上的单调提升。

## 尚需实施的两项集成工作

- **受控评价入口。** 在独立 research runner 中读取指定 realization 和固定 slot，使用完全相同的初始缓存、UAV 位置、当前任务、Cloud 合同，比较旧 Matching 和新任务分配器。明确统计每时隙净效用、资源调用次数、运行时间、可行性及预估上下界，不修改正式 PPO runner。
- **在线滚动控制与动态规模。** 基于已有 CacheTargetCandidate / UavTargetCandidate 生产候选生成器，把合法候选交给联合搜索器；每 slot 只执行当前动作，随后依次用实际到达任务重新求解任务分配及资源；下一 slot 更新预测再规划。扩展大网络时，首先限制候选、场景和内层调用预算。完整多阶段 MPC 非预知约束与未来状态转移需另行实现并验证。

## 公平实验协议

对照至少含 KEEP+STAY、当前 Transfer-Exchange Matching + KKT、独立任务分配搜索、单变量缓存/UAV 变化、双变量联合 H=1、联合 H=5。比较算法时保持同一任务/轨迹/预测种子与系统配置；分别报告 CPU/带宽优化器底层调用数量（不仅是外层 oracle 调用次数）及墙钟耗时。不要把“优化预算 24 次 oracle”误写为“仅 24 次 KKT”。

研究分支上的这些模块 **不接入** 目前正式训练、PPO、Checkpoint 和评价器。现阶段未运行训练、未声称收敛或性能提高。后续 Codex 环境应使用 `conda activate adaptive_cache_mec`，仅进行必要的导入/单场景验证，完成后删除临时文件与测试脚本；不得触碰既有正式实验 artifacts。
