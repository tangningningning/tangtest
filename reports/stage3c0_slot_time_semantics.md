# 阶段3C-0：Slot 时间语义与跨 Slot 任务执行核对

核对范围为当前本地 Stage3B 工作树、Stage3B 配置及冻结旧 `run_config.toml`。只读检查，没有运行训练、仿真、Validation 或测试，也未改动源码和 TOML。项目目录中找到 `11.pdf`，未找到 `12.pdf`；因此涉及“最新系统模型”的完整文献确认保留为未确定。

## 1. 文献时间模型

以下页码均为 `11.pdf` 的 PDF 页码。

- **第1页**：时间被划分为等长基础时隙，集合为 `T={1,…,T}`；`Δt` 是相邻时隙决策时刻的间隔。车辆位置、通信状态和任务请求随时隙变化。该页没有给出 `Δt` 的秒数。
- **第2页**：每个时隙有配置决策和任务调度两个逻辑阶段，并明确这只是逻辑顺序，不将 `Δt` 物理切分。调度在当前实际位置和请求下决定执行节点、计算资源及通信带宽。
- **第3页**：任务车辆在每个时隙至多产生一个不可分割任务 `J_v(t)={f_v(t),D_v(t),L_v(t)}`，由到达变量 `a_v(t)` 表示；未出现任务队列或剩余工作量变量。
- **第4页**：本地计算时延 `T^cmp_{v,v}(t)=L_v(t)/F_v`；边缘计算时延 `T^cmp_{v,m}(t)=L_v(t)/(w_{v,m}(t)F_m)`；Cloud 计算时延 `T^cmp_{v,c}(t)=L_v(t)/F_c`。边缘 CPU 约束按时隙写为 `Σ_v w_{v,m}(t)≤1`。
- **第7–8页**：卸载任务完成时延由上传和计算时延相加；Cloud 为 `T_{v,c}(t)=ψ_cD_v(t)/R_c+L_v(t)/F_c`。该时延进入时隙任务效用 `Q(t)`，允许超过归一化参考值并产生负效用。未发现 `T_v(t)≤Δt` 的截止期限约束。
- **第10–11页**：优化目标是跨时隙效用平均值 `max (1/T)Σ_t G(t)`。约束 C4、C6 明确了 Cache 与 UAV 位置的 `t−1 → t` 状态联系；计算与带宽容量约束 C14、C16 则分别按当前时隙 `t` 表达。文中指出跨时隙依赖的是 Cache 和 UAV 位置。相关页未定义任务剩余周期、排队状态、跨时隙 CPU/带宽占用或完成事件。

**文献边界：** `11.pdf` 的数学表达支持“每时隙任务与资源分配快照”的解释，但没有明确规定若任务延迟超过 `Δt`，物理执行是否持续并占用后续时隙资源。缺少 `12.pdf`，故不把未写出的队列/执行语义当作文献明示结论。

## 2. 当前源码时间模型

- **1 秒时隙及配置约定：** Stage3B 配置 `simulation.dt_s=1.0`、`execution.cross_slot_task_model="independent_per_slot"`；旧冻结 `runs/formal_cache_economic_v2_lr_5e_minus_4/run_config.toml` 采用相同设置。`project_config.py:974-986,979-982` 校验正时长以及只接受该任务模型。
- **每槽任务到达：** `scenario/task_scenario.py:588-648` 对每个 slot、每辆任务车独立采样 Bernoulli 到达并写一行任务状态；`scenario/scenario_generator.py:593-659` 对预测 origin、horizon、scenario 和车辆也逐槽采样。PPO 输入构造在 `ppo_simple/interface.py:1186-1205` 按当前 slot 调 `load_actual_tasks`。
- **任务对象及实际读取：** `decision/task_allocation.py:1491-1516` 的 `Task` 只含任务 ID、车辆、服务、输入量和总 CPU cycles；读取函数只取 `row.slot == slot` 且 `arrival == 1` 的任务。没有 arrival time 以外的进度、剩余 cycles、完成标记或跨槽队列。
- **延迟是评价值：** `common/system_model.py:32-50,90-128,131-170` 直接由工作量和所分 CPU/速率计算延迟；`task_allocation.py:2782-2793,2919-2984` 对本次 matching 得到逐任务时延和效用。未见将 `delay_s` 与 `dt_s` 比较或把时延推进为未来状态的代码。
- **Slot 状态传递：** `decision/slot_environment.py:331-357,620-666` 每次执行接收本槽任务，返回逐任务延迟/效用，并把 `next_cache_state`、`next_uav_positions` 作为下一槽状态。`ppo_simple/rollout.py:601-602,872-911,900-955` 每个 step 执行一次；下一步显式继承 Cache 和 UAV 位置并读取 `aligned_steps[slot_index+1]`，未继承任务或资源占用。
- **CPU/带宽与有限 Cloud 池：** Edge KKT 在本次 matching 的任务组内分配 CPU/带宽。Stage3B 配置 `nodes.cloud.resource_mode="finite_pool"`、`cpu_capacity_cycles_per_s=60e9`（配置第284–296行）；`task_allocation.py:2538-2589,2623-2645,2782-2911` 对本次 Cloud 任务集合求比例并约束总分配不超过该池。下一槽由新任务集合重新计算，因此 Cloud 池只约束同一槽的 matching；旧冻结配置则是 5 GHz/任务、无有限共享池。
- **KKT 缓存：** `optimize_task_matching` 在每次 matching 内创建缓存（`task_allocation.py:966-1007`）；其作用是复用候选资源分配计算。正式 PPO rollout 的 `execute_configured_slot` 调用未传入跨步 KKT 缓存（`rollout.py:872-889`）。即使诊断路径显式复用 solver cache，它也只缓存数值结果，不记录正在运行任务或资源占用。

## 3. 5 秒任务示例

若 `t=100s` 到达的任务按现有公式算得延迟 `5s`，该值会用于 `t=100s` 的任务效用；代码没有要求它在 1 秒内完成，也没有把延迟截断到 1 秒。

在 `t=101s`，代码不会查询该任务是否仍未完成，因为 Task 中没有剩余工作状态，下一步只装载 slot 101 的到达任务。它也不会继续占用原节点的 CPU 或带宽。`t=101s` 新任务与 slot 101 同批任务按该槽的 Edge KKT/Cloud 池约束分配资源，不与 slot 100 的任务联合竞争。

因此，`5s` 是**时延性能评价**，不是执行仿真中的 5 秒进度或资源租约。代码既不表示该任务于 `t=101s` 完成，也不表示它在 `t=101s` 仍运行；跨槽执行在该抽象中不存在。

## 4. 两种解释与自洽性

**A. 独立 Slot 任务性能快照模型**

当前代码属于此模型。任务集合按槽提供，matching/KKT 对该槽单独给出资源分配和完成时延，任务效用进入当前槽 reward；Cache 和 UAV 位置继续影响未来槽。这个结构与 `11.pdf` 的每槽决策变量、按槽资源约束及跨槽 Cache/UAV 状态方程相符。超过 1 秒的时延可作为时延指标，不自动构成代数矛盾；代码未宣称执行在槽末完成。**在此解释下模型自洽，暂不需要改代码。**完整文献一致性仍受 `12.pdf` 缺失限制。

**B. 连续时间任务执行与资源调度模型**

当前实现不符合此解释。一个运行 5 秒的任务理应在后续时段保留剩余工作并参与资源竞争；当前的跨槽状态只有 Cache、UAV 位置，Edge 资源约束和 60 GHz Cloud 池都只对本槽 matching 的任务集合生效。缺口发生在 `load_actual_tasks` 每槽重建任务集以及 rollout 到下一槽时未传递队列/进度的状态边界。

若研究目标要求 B，最小模型范围至少包括：任务剩余上传量/计算量及到达、完成、排队规则；每槽对存量与新任务共同分配 CPU/带宽；Cloud 容量对所有仍在执行的 Cloud 任务生效；以及将任务进度/队列加入环境状态和 PPO observation/reward/episode 边界。它会影响 `task_scenario.py`、`Task`/`task_allocation.py`、`slot_environment.py`、`interface.py` 与 `rollout.py`，不是仅改 `dt_s` 或一个时延公式。

## 5. 对长期收益与训练目标的影响

- **Cache：** 长期收益来自后续槽新到任务的服务命中和当前槽任务效用；没有正在执行任务对后续 Cache 变更的反馈，也没有队列消退收益。
- **UAV：** UAV 位置跨槽继承，移动可改变后续槽任务的覆盖、链路和效用；一个跨多秒执行任务的持续链路/移动期间状态不在模型内。
- **PPO：** Micro reward 是当前槽快照效用，Macro 按 holding 期间累计这些槽 reward。训练目标因而是多槽 Cache/UAV 策略下的快照效用和，不是含 backlog、任务完成时间线或持续资源竞争的连续执行目标。若实际系统意图为 B，忽略存量任务可能使资源竞争与长期效用评价偏离。

## 6. 结论与 Stage3C-1

在论文第11页可见的槽级优化抽象下，当前代码没有发现内部时间状态矛盾；发现的是其**不能表达连续时间下的跨槽任务和资源占用**。多秒时延本身不等于必须在当前槽完成。建议 Stage3C-1 先固定采用 A 还是要求 B；若采用 A，可继续，不需本阶段改代码；若要求 B，再按上述跨模块范围设计变更。`12.pdf` 补齐后应复核完整文献语义。

```makefile
TIME_MODEL_TYPE = INDEPENDENT_SLOT_PERFORMANCE_SNAPSHOT
TASKS_CARRY_ACROSS_SLOTS = NO
UNFINISHED_WORKLOAD_STATE_EXISTS = NO
RESOURCE_OCCUPANCY_CARRIES_ACROSS_SLOTS = NO
TASK_DELAY_CAN_EXCEED_SLOT = YES
PAPER_TIME_MODEL_MATCH = UNDETERMINED
TIME_MODEL_CONSISTENT = YES_FOR_A; NO_FOR_B
MODEL_CHANGE_REQUIRED = NO_FOR_A; YES_FOR_B
READY_FOR_STAGE3C1 = YES
```

`PAPER_TIME_MODEL_MATCH` 标为 `UNDETERMINED` 是因为项目内缺少 `12.pdf`，且 `11.pdf` 未明确规定超槽时延对应的真实执行生命周期；可确认的是其槽级公式和当前快照实现吻合。
