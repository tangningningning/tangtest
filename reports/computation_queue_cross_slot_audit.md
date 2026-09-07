# 计算资源、队列与跨时隙执行审计

**审计依据：`C:\Users\tangning\Desktop\12.pdf`（唯一系统模型来源）**  
**审计对象：** `adaptive_cache_mec` 当前工作区代码、`system_config.toml`、当前生成数据及既有审计报告  
**审计性质：** 只读诊断；本次没有修改模型、配置、参数、生成数据或既有报告，也没有运行 PPO 训练、GRU 训练或长时仿真。

## 结论先行

当前计算公式、边缘 KKT 资源分配、批次级跨时隙剩余工作量、实际/预测队列传递和 PPO 队列观测之间已经形成一条数值上自洽的链路。固定种子验证没有发现会立即使当前实际执行结果失真的严重计算或队列错误。

审计发现以下一项严重问题和四项中等/提示问题：

1. 当前 Cloud 已取消同一 slot 内按任务数均分资源，但仍被统一 `CLOUD` 队列纳入跨 slot 等待；这会让后一任务等待无关的前一 Cloud 任务，违反已经确认的 Cloud 固定每任务资源、无任务间资源竞争合同。
2. Local 队列键的通用函数参数叫 `task_id`，而模块注释和 `12.pdf` 的任务车辆语义更适合使用 `vehicle_id`。当前正式 loader 恰好令两者相等，所以 artifact 路径没有失败；如果外部任务 ID 按 slot 变化，同一车辆的 Local backlog 会断开。
3. `advance_queue_workload` 接收的是已经包含上传、计算和旧队列等待的完整端到端时延；这不会重复加 wait，但会把通信时间纳入节点剩余工作量。它是否合理取决于 workload 被定义为“端到端批次占用”还是“CPU 剩余工作”，`12.pdf` 没有给出该队列扩展定义。
4. 外部 `Task` 列表允许重复 `task_id`，随后构造的按任务字典会覆盖前一条任务；当前正式 `tasks.csv` 的 TV×slot 生成方式没有触发它，但 API 缺少显式防护。
5. `execute_slot` 只检查输入队列的数值合法性，不检查节点 ID；PPO 观测构造会拒绝未知队列节点，两个入口的校验边界不一致。

因此：当前 Cloud 统一队列是严重问题，直接阻塞 PPO 训练。必须使 Cloud `queue_wait=0`，并且 `advance_queue_workload` 不生成、不传播 `CLOUD` backlog；实际路径、surrogate 路径和 PPO 状态必须同步。Cloud queue feature 可以暂时保留固定维度，但训练前必须恒为 0，或统一从 observation 中移除。Cloud 修正并完成回归验证后，才适合进入目标函数/PPO 状态审计；其他 Local/workload/输入校验问题不要求推倒当前 KKT 主链路。

## 1. 模型依据与审计边界

本次以 `12.pdf` 为系统模型，不以仓库中的旧报告反推模型。`12.pdf` 的相关定义如下：

- 第 2 页规定每个基础时隙包含配置决策和任务调度两个逻辑阶段，但这不是对 `Δt` 的物理切分。
- 第 3-4 页规定每个任务车辆每个 slot 至多产生一个不可分割任务；任务有输入数据量 `D` 和 CPU 周期数 `L`，已到达任务只能选择一个执行节点。
- 第 4 页规定本地执行使用任务车辆完整计算能力；服务车辆、RSU、UAV 才参与计算资源比例分配。
- 第 4-5 页给出本地和边缘计算时延；第 7-8 页给出边缘/Cloud 传输、完成时延和车辆能耗；Cloud 使用独立蜂窝链路，不参与服务节点共享带宽。
- 第 9 页定义 `G(t)=Q(t)-C_ca(t)-C_q(t)`，其中任务服务效用由完成时延和车辆能耗决定。
- 第 11 页的 C13-C16 只对服务车辆、RSU、UAV 的计算比例和上传带宽比例施加共享资源约束。

`12.pdf` 没有定义队列长度、等待时延、跨 slot 未完成任务、任务级 FIFO、Cloud 处理队列或队列观测变量。因此，当前 `deterministic_batch_fifo` 是在 `12.pdf` 计算/时隙模型之上的工程扩展，下面会把“与模型直接一致”和“扩展假设”分开说明。

上一轮已经确认的 UAV 固定高度、Cloud 固定每任务资源、`psi_c=11`、实际/预测 Cloud gate 等合同本次视为输入，不重新打开，也没有回退这些改动。

## 2. 计算公式、单位与资源数值

### 2.1 当前代码公式

对应实现位于 `src/common/system_model.py` 的 `local_execution`、`edge_execution`、`cloud_execution` 和 `task_utility`，并由 `src/decision/task_allocation.py` 的 Matching/KKT 评价调用。

| 执行位置 | 当前实现的服务时延 | 车辆能耗 | 资源含义 |
|---|---|---|---|
| Local | `L / F_v` | `κ L F_v^2` | 任务车辆使用完整本地 CPU，不进公共 KKT |
| SV/RSU/UAV | `D/R_vm + L/(w_vm F_m)` | `p_v D/R_vm` | `w_vm`、`b_vm` 进入各自节点的 KKT 共享 |
| Cloud | `psi_c D/R_c + L/F_c` | `p_v D/R_c` | 固定每任务 `F_c`、`R_c`；不进边缘 KKT，`psi_c` 不进能耗 |

单位链闭合如下：`L` 为 cycle，`F` 为 cycle/s，因此计算项为 s；`D` 为 bit，`R` 为 bit/s，因此传输项为 s；`p` 为 W，乘以传输秒数得到 J。队列等待量也是 s，只加到完成时延，不改变车辆发射能耗。

### 2.2 当前冻结数值

| 资源/参数 | 当前值 | 代码/配置位置 | 审计判断 |
|---|---:|---|---|
| TV Local CPU | `1.0e9 cycle/s` | `system_config.toml` `[nodes.vehicle]` | 与 `12.pdf` 的完整本地 CPU语义一致 |
| SV CPU | `1.0e9 cycle/s` | `[nodes.sv]` | 进入 edge KKT |
| RSU CPU | `5.0e9 cycle/s` | `[nodes.rsu]` | 进入 edge KKT |
| UAV CPU | `5.0e9 cycle/s` | `[nodes.uav]` | 进入 edge KKT；高度为 100 m |
| SV/RSU/UAV 带宽 | `10.0e6 Hz` | 各节点 bandwidth 配置 | 用 `b_vm` 分配，速率由正式 Shannon 链路计算 |
| Cloud CPU | `5.0e9 cycle/s` per task | `[nodes.cloud]` | 不按同槽 Cloud 任务数均分 |
| Cloud 等效上传速率 | `200.0e6 bit/s` per task | `[nodes.cloud]` | 独立蜂窝链路，不进边缘带宽 KKT |
| Cloud 扩展系数 | `psi_c=11.0` | `[nodes.cloud]` | 只乘 Cloud 传输时延 |
| slot 推进量 | `dt_s=1.0 s` | `[simulation]` | 队列唯一推进时间尺度 |
| 时延参考值 | `T_ref=30.79009347 s` | `[objective]` | 任务效用和 PPO 队列特征归一化共同使用 |

### 2.3 KKT 资源分配

`src/decision/task_allocation.py:1191` 的 `solve_kkt_fraction_allocation` 对正系数使用

`r_i = clip(sqrt(c_i / lambda), epsilon, 1)`

并通过确定性二分使比例和接近 1。`allocate_resources_kkt` 只将 SV、RSU、UAV 任务分组，计算系数为：

`c_i^cpu = omega_t * L_i / (T_ref * F_m)`

`c_i^bw = D_i / Rbar_i * (omega_t / T_ref + omega_e * p_i / E_ref)`

因此 CPU 比例和带宽比例分别受 `sum_i w_i <= 1`、`sum_i b_i <= 1` 约束；单任务节点返回比例 1，多任务节点按 KKT 分配。Cloud 明确不出现在 `cpu_allocations`、`bandwidth_allocations` 和边缘资源和中，这与 `12.pdf` C13-C16 的适用范围一致。

当前代码没有建模服务节点侧 CPU 消耗，只建模任务车辆本地 CPU 能耗或车辆上传能耗。这与 `12.pdf` 的 `E_v^veh` 定义一致，不应把它误判为漏算 Cloud/RSU/UAV 服务器电费；服务器侧能耗若要加入，是另一个目标函数扩展。

## 3. 队列与跨时隙执行语义

### 3.1 实际数据结构

`src/decision/task_allocation.py:66` 的 `advance_queue_workload` 保存的是

`execution_node_id -> remaining_workload_s`

而不是任务对象列表。当前 ID 约定为：

- Local：`LOCAL:<task_id>`；正式任务加载时 `task_id=vehicle_id`，所以正式 Local 队列按任务车辆区分。
- SV/RSU/UAV：直接使用稳定的 cache/execution node ID，例如 `43`、`RSU_0`、`UAV_0`。
- Cloud：使用 `CLOUD`。

这套映射避免把所有 Local 任务错误地合成一个公共节点，也使服务车辆和 UAV 移动时仍能按节点身份保留上一 slot 的剩余量。

需要特别区分“当前正式路径”和“通用接口合同”：`load_actual_tasks` 与预测 surrogate 都把 `task_id` 设置为 TV ID，因此当前 artifact 中 `LOCAL:<task_id>` 等于 `LOCAL:<vehicle_id>`。但是 `local_target` 的入参本身是 `task_id`，`build_feasible_targets`、`evaluate_matching_with_kkt` 和 `slot_environment` 的 wait lookup 也沿用该键；它们没有强制 `task_id==vehicle_id`。如果调用者在相邻 slot 使用 `veh-7@t0`、`veh-7@t1` 这类任务 ID，Local backlog 不会自动按车辆继承。

### 3.2 当前推进公式

令节点 `n` 的上一步剩余工作量为 `W_n`，本 slot 分配到该节点的任务在不计旧队列等待时的完成时延为 `S_i`，则代码先把当前任务等待量加入结果：

`T_i = W_n + S_i`

再按 `dt_s` 推进：

`W_n(next) = max(0, W_n - dt_s, max_i(T_i - dt_s))`

没有当前任务的节点只做 `max(0, W_n-dt_s)`。同一 slot 同一节点的任务按 KKT 比例并发执行，最慢任务形成下一 slot 的节点剩余工作量；当前批次结束后节点才释放。因此 `deterministic_batch_fifo` 的准确含义是“旧批次先于新批次，批次内部并发”，不是任务级逐个串行 FIFO。

这与当前代码的实际调用一致：

1. `execute_slot` 接收 slot 开始时的 `previous_queue_workload_s`。
2. 实际 Matching/KKT 先计算当前任务的物理服务时延。
3. `evaluate_matching_with_kkt` 对每个任务只加一次 `queue_wait_s`。
4. `advance_queue_workload` 用已含等待的 `per_task_delay_s` 产生下一状态。
5. 当前任务效用 `Q(t)` 使用含等待的完成时延，系统效用仍为 `Q(t)-C_ca(t)-C_q(t)`。

队列等待不在 `execute_slot` 中再次扣除，也没有被单独作为第二个成本重复扣除。旧任务不会在下一 slot 重新出现在 `actual_tasks` 中，下一 slot 只携带标量剩余工作量，所以没有发现旧任务被重复奖励或重复计算的路径。

### 3.3 与 1 秒 slot 的关系

配置的 `simulation.dt_s=1.0`，队列推进直接读取该值，没有第二套队列时间尺度。一个服务时长为 0.4 s 的任务在无旧负载时下一队列为 0；2.5 s 的任务从 slot 0 开始时，连续推进后的残量为：

| 推进后时刻 | 剩余工作量 |
|---|---:|
| slot 0 结束 | 1.5 s |
| slot 1 结束 | 0.5 s |
| slot 2 结束 | 0.0 s |

所以“任务时延超过 slot”目前不会被截断为一个 slot，也不会丢失跨 slot 残量。

### 3.4 现有 backlog 与新任务

对 Local 节点设置旧工作量 `W=2.0 s`，加入服务时长 `S=0.5 s` 的新任务，代码得到：

- 新任务等待：`2.0 s`；
- 完成时延：`2.5 s`；
- 下一 slot 剩余：`1.5 s`；
- 车辆能耗与无旧队列时相同。

这正是当前“队列只影响等待时延，不改变已选通信/计算资源和车辆能耗”的实现语义。

### 3.5 三个重点疑点复核

**疑点 1：Local 队列是否错误地按跨 slot 变化的 task ID 分队列？**  
在当前正式数据中没有实际触发，因为 `src/decision/task_allocation.py:1820` 的 `load_actual_tasks` 使用 `task_id=vehicle_id`，`cache_configuration.py` 的预测事件也使用 `tv_id`。但最小通用案例确实复现了潜在问题：slot 0 任务 `t0`、车辆 `veh-7`、服务时长 2.5 s 产生 `{'LOCAL:t0': 1.5}`；slot 1 同一车辆使用新任务 ID `t1` 时，当前键查找为 0 s，而按稳定车辆键 `LOCAL:veh-7` 才能继承 1.5 s。`slot_environment.py:293` 的 `per_task_queue_wait_s` 直接按 `matching.matching[task.task_id]` 查上一队列，因此也会沿用这一行为。

结论：当前正式 artifact 路径通过“任务 ID 等于车辆 ID”规避了问题，但通用接口与模块顶部“`LOCAL:<vehicle_id>`”的注释不一致。严重度为**中等的接口健壮性风险**，不是当前固定数据中的已发生严重故障。修复方向是在 Local queue identity 处显式使用 `vehicle_id`，同时保留 `task_id` 作为任务字典主键；或者在正式入口明确强制并校验 `task_id==vehicle_id`，不能依赖调用者默契。

**疑点 2：Cloud 固定每任务资源是否仍被错误地串成共享 FIFO？**  
同槽资源层面没有发现错误：`cloud_execution` 对 `cloud_task_count=1` 和更大的兼容参数返回相同的固定 `F_c/R_c`，Cloud 不出现在 edge KKT map。跨 slot 层面则确实存在一条公共 `CLOUD` queue：`evaluate_matching_with_kkt` 用 `queue_target=CLOUD` 读取同一个等待量，`advance_queue_workload` 对所有当前 Cloud 任务取最大完成时延并写回 `next_queue['CLOUD']`。因此上一 slot 的 Cloud backlog 会给下一 slot 的每一个新 Cloud 任务加同一个等待量。

固定 seed 的 Cloud 微型用例把上一队列增加 3.0 s，观察到每个新 Cloud 任务的 wait 增加恰好 3.0 s、车辆能耗不变、仍生成 next workload；`seed=4401, slot=20` 的实际用例也观察到 Cloud 3 个任务均增加 0.25 s wait，Cloud next workload 从 2.7346424619 s 变为 2.9846424619 s。

结论：这不是“同一 slot 按任务数均分资源”，而是“不同 slot 的 Cloud 批次共享一个排队门”。虽然 `12.pdf` 没有扩展定义 queue，但项目已经确认的正式 Cloud 合同要求每个任务独立获得固定 `F_c/R_c`，不存在任务间资源竞争；因此统一 `CLOUD` backlog 会引入不应存在的跨任务等待。这是**严重问题并阻塞 PPO**。修复目标是 Cloud `queue_wait=0`，`advance_queue_workload` 不生成/传播 `CLOUD` workload，surrogate 与实际路径保持一致；PPO 可暂时保留 Cloud queue feature 以维持维度，但其值必须恒为 0，或训练前统一移除该 feature。

**疑点 3：把完整端到端 delay 送进 workload 是否重复等待，或错误占用 CPU？**  
没有发现旧 wait 被算两次。当前任务的 `per_task_delay_s` 已是 `queue_wait + physical_service_delay`，`advance_queue_workload` 不再把 wait 直接加一遍，而是用 `max(previous-dt, per_task_delay-dt)` 保留较大的剩余量；因此 `W=2.0 s`、服务时长 0.5 s 时结果是 next 1.5 s，而不是重复叠加后的更大值。

但通信时间确实被带入了 node workload。最小 edge 例中，上传 1.0 s、计算 0.5 s、完整时延 1.5 s；传入 `advance_queue_workload` 后，`dt=1.0 s` 的 next workload 为 0.5 s，而如果 workload 仅表示 CPU 剩余工作，CPU-only 残量应为 0 s。也就是说，代码当前建模的是“从当前批次开始到任务端到端完成，节点批次仍占用/阻塞”的时间，而不是“只有 CPU 计算阶段占用节点”的 workload。

结论：这是**中等的模型语义风险**，是否为 bug 取决于项目要采用哪种队列定义。若队列表示节点对完整任务批次的端到端 reservation，当前行为可接受；若队列表示 CPU backlog，则应把 `compute_delay` 与 `upload_delay` 分开推进，通信还应由带宽资源的时间语义单独处理。`12.pdf` 没有 queue/workload 定义，不能把当前选择冒充为 PDF 原式。

### 3.6 UAV 移动与 backlog

UAV backlog 按 `UAV_0`/`UAV_1` 身份保留，而不是按旧水平坐标复制。新任务在当前最终 UAV 位置上重新计算距离、信道和服务时延；已经进入上一 slot 的剩余量只作为秒数推进，不按新位置重新计算旧任务的信道。这在“剩余工作量代表已进入节点的计算/批次服务工作”这一抽象下是自洽的，也没有发现旧任务因 UAV 移动而被重新分配或重复计费。

但若后续希望把剩余量解释为“仍在空中传输中的任务”，当前标量队列就不够表达传输链路、任务身份、迁移或取消；这属于模型粒度提示，不是当前代码的数值错误。

## 4. 实际执行、预测评价与 PPO 状态的时序

### 4.1 实际 slot 时序

当前 `src/decision/slot_environment.py:107` 的主要时序为：

`previous (C, UAV, queue)`
`-> observation/action trigger`
`-> 可选预测配置优化`
`-> 当前实际车辆位置和最终 UAV 位置`
`-> 当前实际 arrival 任务 Matching/KKT`
`-> queue wait + 当前 Q`
`-> queue advance`
`-> G(t)`
`-> next (C, UAV, queue)`

配置阶段和任务调度阶段仍是逻辑顺序，未把 slot 人为拆成两段物理时间，符合 `12.pdf` 第 2 页。配置预测阶段使用预测位置/预测任务；真正 reward 使用实际位置/实际到达任务；两者都从同一个 slot 开始队列副本出发。

但在已确认 Cloud 合同下，“同一个队列副本”不能包含可传播的 `CLOUD` backlog；当前实现的同步只说明实际与预测会一致地复现该错误，不说明其语义正确。S1 修正后，实际和预测都应从同一非 Cloud queue 状态出发。

### 4.2 预测队列

`src/decision/cache_configuration.py:479` 的 `build_surrogate_evaluation_context` 为每个 scenario 保留 `scenario_queues[scenario]`，初值是当前 slot 开始的队列。`evaluate_surrogate_context` 按 scenario × horizon 形成完整批次，空预测批次也保留并调用 `advance_queue_workload`，因此预测窗口中没有任务的 slot 会正确排空旧队列。

没有发现预测评价读取未来实际 `next_queue` 的路径：预测只使用当前传入的初始队列和预测 scenario 的任务/位置/流行度；实际下一队列要等当前实际 Matching 完成后才生成。场景队列彼此独立，未出现一个 scenario 的 backlog 污染另一个 scenario 的情况。但当前 surrogate 会照样读取/传播 `CLOUD`，所以在 S1 修正前，实际与预测虽然同源，都会保留错误的 Cloud 跨 slot 等待。

### 4.3 Full 与 incremental

`evaluate_matching_candidate_incremental` 对受影响的 edge 节点重算 KKT；Cloud 因为固定每任务资源不参与同槽 KKT 耦合，但当前统一 `CLOUD` queue 仍制造跨 slot 耦合。非 Cloud 队列等待是同一个节点级输入，既不会让不受影响的任务被错误重算，也不会让受影响任务沿用旧的 edge 资源；Cloud 则必须在 S1 后从 queue wait 和 queue advance 中排除。

固定测试中对带队列的 full/incremental 候选评价得到：

| 比较项 | 最大差异 |
|---|---:|
| `J` | 0 |
| CPU 分配 | 0 |
| 带宽分配 | 0 |
| 每任务时延 | 0 |
| 每任务车辆能耗 | 0 |
| 每任务效用 | 0 |
| 节点 CPU 和 | 0 |
| 节点带宽和 | 0 |
| next queue | 完全相等 |

### 4.4 PPO 观测与动作状态传递

`src/ppo/ppo_trigger_policy.py:112` 的 `ConfigurationObservationSpec` 固定执行节点顺序为：

`LOCAL:<task vehicle ids> + cache node ids + CLOUD`

当前正式规模为 20 个 Local、9 个服务节点、1 个 Cloud，共 30 个队列特征；观测总维度为 184。代码当前会把 Cloud 作为一个 queue feature，但在 S1 修正后该 feature 必须恒为 0，或与 observation spec 一起移除。其他队列特征为单调有界的

`q_feature = q_s / (q_s + T_ref)`

不是硬截断，且 0、长队列、任意非负 finite 秒数都能进入同一固定维度。

`src/ppo/ppo_rollout.py:972` 的 on-policy rollout 在动作前用上一步的 `previous_queue` 构造观测，把同一队列传入 `execute_slot`，再把 `result.next_queue_workload_s` 原样作为下一步状态。已有 state propagation 检查也比较了 next queue 与下一步 previous queue。因此在当前“节点标量批次工作量”抽象下，队列已进入 PPO 状态，动作前后没有明显的 Markov 状态丢失。

## 5. 固定种子只读数值验证

验证使用当前配置、当前生成数据和固定 seed `4401`；没有改写任何文件。纯函数/短路径结果如下。

### 5.1 资源和队列微型用例

| 用例 | 结果 | 判断 |
|---|---|---|
| 单节点单任务，服务时长 0.4 s | next workload `0.0 s` | 小于 slot 的任务不残留 |
| 单节点单任务，服务时长 2.5 s | `1.5 -> 0.5 -> 0.0 s` | 跨 3 个 slot 正确推进 |
| 同节点两任务 | CPU fractions `0.5857864376, 0.4142135624`；带宽 fractions `0.3882964182, 0.6117035818` | CPU 和 `0.9999999999999`，带宽和 `1.0000000000004`；KKT 满足约束 |
| 同节点两任务的最慢服务时延 | `0.3596027524 s` | 小于 slot，next workload `0.0 s` |
| 旧 Local backlog 2.0 s + 新任务服务 0.5 s | wait `2.0 s`，total `2.5 s`，next `1.5 s` | 等待只加一次，能耗不变 |
| 同一车辆、相邻 slot 使用 `t0`/`t1` 两个 Local task ID | slot 0 queue `LOCAL:t0=1.5 s`；slot 1 按当前 task ID 查到 `0 s` | 证明通用接口不能把变化的 task ID 当稳定 Local 节点键 |
| Edge 上传/计算分解 | upload `1.0 s`，compute `0.5 s`，total `1.5 s`，next `0.5 s` | 当前 workload 包含通信；CPU-only 对照残量为 `0 s` |
| 带队列 full/incremental | 所有列出的差异为 0，next queue 相等 | 两条评价路径一致；Cloud 不在 KKT map |
| NaN/Inf/负 workload | 三者均被拒绝 | 数值输入防护有效 |

### 5.2 旧 backlog + 新任务连续三个 slot 对照

为把“旧 backlog + 新任务”与代码推进公式逐 slot 对照，统一设置 `W_0=2.0 s`、`dt=1.0 s`，每个 slot 都加入一个新的任务批次，并将当前任务的完整时延按 `T_i=W_i+S` 传入 `advance_queue_workload`。Local 使用稳定节点键 `LOCAL:veh-7`；edge 使用 `RSU_0`；Cloud 使用 `CLOUD`。

| 节点 | 当前任务服务时延 `S` | slot 0：wait / total / next | slot 1：wait / total / next | slot 2：wait / total / next |
|---|---:|---:|---:|---:|
| Local | `0.50 s` | `2.00 / 2.50 / 1.50 s` | `1.50 / 2.00 / 1.00 s` | `1.00 / 1.50 / 0.50 s` |
| Edge `RSU_0` | upload `1.00` + compute `0.50` = `1.50 s` | `2.00 / 3.50 / 2.50 s` | `2.50 / 4.00 / 3.00 s` | `3.00 / 4.50 / 3.50 s` |
| Cloud | communication `0.55` + compute `1.00` = `1.55 s` | `2.00 / 3.55 / 2.55 s` | `2.55 / 4.10 / 3.10 s` | `3.10 / 4.65 / 3.65 s` |

三行均与纯函数实际输出一致。Local 行使用稳定 `vehicle_id` 键，是对当前正式模型意图的对照；若相邻 slot 改用 `LOCAL:t0`、`LOCAL:t1`，则前一段的 1.5 s 不会被下一任务读取，这正是 Local 键风险而不是推进公式本身的差异。Edge 和 Cloud 的结果还显示：只要采用当前定义的完整端到端 `S`，通信时间就会持续影响后续 workload。

### 5.3 Cloud 跨 slot 观察

通用队列微型用例给 Cloud 设置旧 workload 增量 3.0 s：

- Cloud 新任务等待量增加恰好 `3.0 s`；
- 车辆上传能耗不变；
- next workload 仍生成，本例为 `2.155 s`；
- Cloud 没有进入同槽 CPU/带宽 KKT map。

这证明当前实现虽然满足“Cloud 同槽固定每任务资源”，却仍通过统一 `CLOUD` 键给无关的后续 Cloud 任务制造跨 slot 等待。按照已经确认的 Cloud 合同，该行为不应存在；Cloud wait 必须为 0，Cloud 不应形成或传播节点 backlog。

另一个纯函数对照在同一 slot 放入两个 Cloud 任务，令各自固定资源下的物理服务时延为 5 s 和 7 s。当前 `advance_queue_workload({}, matching, {task_a: 5, task_b: 7}, 1)` 返回 `CLOUD=6 s`；在正式 Cloud 合同下两个任务互不竞争，正确的 Cloud queue wait 和 next Cloud backlog 都应为 0。这是“统一 `CLOUD` FIFO 会产生 6 s 虚假残量”的直接证据。

### 5.4 固定 seed 实际 slot

`seed=4401, slot=20`，实际到达任务数为 13；空队列和仅加入 `CLOUD:0.25 s` 初始 backlog 的结果如下：

| 项目 | 空队列 | Cloud 初始 backlog 0.25 s |
|---|---:|---:|
| Cloud 任务数 | 3 | 3 |
| Cloud 任务等待 | 0 | 每个 Cloud 任务 `0.25 s` |
| `UAV_0` next workload | 11.3630293293 | 11.3630293293 |
| SV `43` next workload | 21.6084417542 | 21.6084417542 |
| `RSU_0` next workload | 18.3946231276 | 18.3946231276 |
| `CLOUD` next workload | 2.7346424619 | 2.9846424619 |
| `RSU_1` next workload | 12.9805787398 | 12.9805787398 |
| `UAV_1` next workload | 11.4236016888 | 11.4236016888 |

两次匹配的 Cloud 数均为 3；只有 Cloud 等待和 Cloud next workload 增加 0.25 s，说明旧 Cloud backlog 没有意外改变边缘 KKT 资源或其他节点队列。

同一固定 slot 的正式 arrival 任务 ID 没有重复，`duplicate_arrival_id_slots={}`。但对直接传入的重复 `Task` 列表测试，当前结果为：

- `duplicate_task_id_rejected=false`；
- 结果字典 key 数为 1。

这不是当前正式 artifact 的已发生错误，而是应在 API 边界修复的健壮性问题。

### 5.5 PPO 队列观测

构造当前正式规模的配置观测，输入 Local、RSU 和 Cloud 的非零队列后得到：

- `observation_dim=184`；
- `queue_node_count=30`；
- queue slice 长度为 30；
- queue slice 与 `q/(q+T_ref)` 逐项一致；
- 全部观测值 finite；
- 未知节点 `UNKNOWN` 被拒绝。

## 6. 既有报告的数值时效性

既有报告没有被修改，但其中部分 Cloud 数值属于上一套 Cloud 合同，不能直接作为当前基线数值引用：

- `reports/computation_load_analysis.md` 使用过 `20e9 cycle/s` 和同槽按任务数共享的 Cloud 解释；当前为 `5e9 cycle/s` per task 且不做同槽 `N` 均分，因此其中 Cloud 延迟/资源表已过时。
- `reports/cloud_delay_analysis.md` 中的旧 Cloud 延迟数值也不能与当前 `psi_c=11`、固定每任务资源直接混用。
- `reports/baseline_v2_rollout_validation.md` 的旧 slot 任务数、Cloud 数量和路径统计是在旧 Cloud 合同下形成，不应与本次 fixed-seed 结果横向比较。
- `reports/communication_contract_repair_validation.md` 是上一轮合同修复验证，当前只作为已确认前置条件参考；本审计没有回退或重新解释其中的 Cloud/UAV 合同。

## 7. 问题分级

### 严重问题：1 项，阻塞 PPO

**S1 - 统一 `CLOUD` backlog 违反已确认的 Cloud 固定每任务资源合同。**  
正式合同已经固定：每个 Cloud 任务独立获得 `F_c=5 GHz`、`R_c=200 Mbps`，不按同槽任务数共享，没有 Cloud resource allocation variable，也不存在 Cloud 任务之间的资源竞争。当前代码却用一个全局 `CLOUD` 队列把上一批次的 max completion 传给下一批次，因而给无关 Cloud 任务增加 wait。`0.25 s` 的旧 Cloud backlog 会等量抬高下一 slot 每个 Cloud 任务的完成时延；同槽物理时延为 5 s 和 7 s 的两个 Cloud 任务还会生成 `CLOUD=6 s` 的虚假残量。该行为会改变实际 reward、surrogate objective 和 PPO observation，不能进入正式 PPO 训练。

最小修复目标（本次只记录、不执行）是：实际 Matching 的 Cloud `queue_wait` 恒为 0，`advance_queue_workload` 不生成或传播 `CLOUD` backlog，surrogate 复用同一规则。PPO 可以暂时保留固定维度的 `CLOUD` queue feature，但必须恒为 0；或者在训练前统一移除该 feature 并同步调整 observation spec。

除 S1 外，没有发现以下严重故障：

- CPU/带宽 KKT 比例和系统性超过 1；
- 长任务被 slot 截断且残量丢失；
- queue wait 被重复加入；
- 旧任务在后续 slot 被重复奖励；
- full/incremental 在有队列时产生数值分叉；
- 预测读取未来实际 queue；
- PPO 下一步没有继承 next queue；
- NaN、Inf 或负队列静默进入计算。

### 中等问题：PPO 前应定案

**M1 - Local 队列键对通用 task ID 不稳定。**  
正式 loader 当前令 `task_id=vehicle_id`，所以现有 artifact 没有复现；但通用 `Task` 允许同一 `vehicle_id` 跨 slot 使用不同 `task_id`，`local_target(task.task_id)` 和 `per_task_queue_wait_s` 会让 backlog 断开。修复方向是 Local queue identity 使用 `vehicle_id`，或在 formal API 明确强制二者相等。

**M2 - workload 是否应包含通信时间尚未定案。**  
当前 `per_task_delay_s` 是完整端到端完成时延，队列因此会携带 upload/Cloud communication 时间。若 workload 定义为完整批次 reservation，这是可解释的；若定义为 CPU backlog，则会把不占 CPU 的通信阶段计入节点工作量。需要先写清队列物理含义，再决定是否拆分 compute/communication workload。

**M3 - 重复 task ID 会造成字典覆盖。**  
 `load_actual_tasks` 对正式 CSV 是 TV×slot 唯一 ID，但通用 Matching/evaluator 接口没有在构造 `matching`、`per_task_delay_s` 等字典前拒绝重复 ID。直接 API 调用可以“可行地”返回，但丢失同 ID 的前一条任务，属于静默数据损坏风险。

**M4 - 队列节点 ID 校验不一致。**  
 `execute_slot` 对 `previous_queue_workload_s` 检查了 finite/non-negative，但未知节点键会被保留并可能继续传播；`build_configuration_observation` 则明确拒绝未知节点。内部正式生产路径使用稳定 ID 时没有复现，但外部传入 malformed queue 可能在环境执行成功后于下一步 PPO 观测失败。

### 提示项：当前扩展需写进模型说明

**P1 - `deterministic_batch_fifo` 不是任务级 FIFO。**  
它是“旧批次先行、同批次 KKT 并发、最慢任务形成残量”的标量模型。状态中没有 active task ID、剩余 CPU cycle、剩余 bit、到达时间或取消/迁移状态；如果后续要研究严格任务级队列，需要扩大状态和执行器定义。

**P2 - UAV 移动后的 backlog 采用节点身份粘连。**  
这对已进入节点的计算工作量合理；对仍在传输中的任务则不够精细。当前没有观察到重复执行或错误重算，但需要在模型扩展文档中说明 backlog 的物理含义。

**P3 - 队列本身是 `12.pdf` 之外的扩展。**  
`12.pdf` 定义的是 slot 内任务完成时延，没有等待项。当前代码可以作为离散时隙的批次排队扩展，但不能把它描述成 `12.pdf` 已经明确定义的 FIFO 队列模型。

## 8. PPO 前建议的下一步

建议顺序如下：

1. 先修正 S1：Cloud `queue_wait` 恒为 0，`advance_queue_workload` 不生成/传播 `CLOUD` backlog，实际和 surrogate 同步；PPO queue feature 暂保留但恒为 0，或训练前统一移除。
2. 再确认 M1/M2：Local queue 是否按 `vehicle_id` 稳定，非 Cloud workload 是完整端到端 reservation 还是 CPU backlog。Local 当前正式 artifact 路径可用，workload 则应作为明确的模型扩展记录。
3. 在 Matching/evaluator 输入边界增加重复 `task_id` 拒绝，并让 `execute_slot` 与 PPO observation 使用同一套 execution-node ID 校验。两项都属于小范围健壮性修复。
4. S1 修正并完成回归验证后，才进入目标函数/PPO 状态审计：重点核对队列等待对 `Q(t)`、`G(t)`、预测缓存选择、优势估计和 observation normalization 的一致性。当前结果不支持直接声称“PPO 已适配”或“训练已验证”；本次只证明非 Cloud 队列短路径数值和状态传递基本闭合。

## 9. 变更与验证记录

- 本次唯一新增文件为本报告。
- 未修改 `system_config.toml`、`12.pdf`、任何源码、生成 artifact 或既有报告。
- 未运行正式 PPO/GRU 训练、长 SUMO、train/validation/test split 或正式 artifact regeneration。
- 已清理本次 Python 只读校验产生的 `__pycache__` 临时目录。
- 本报告中的数值均来自固定 seed、短路径或纯函数检查，不代表 PPO 训练收益，也不替代后续目标函数审计。
