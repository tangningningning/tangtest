# Stage6 执行节点选择根因分析

日期：2026-10-12  
范围：只读源码分析 + Training 池内四个完整 Episode 的冻结策略回放。未修改正式源码、配置或算法。

## 结论摘要

- **NODE_SELECTION_ROOT_CAUSE = SUPPORTED**：当前 Matching 没有 Cloud 优先分支。它给每个任务建立缓存/链路合法的 Local、SV、RSU、UAV 候选，并始终把 Cloud 放进资源感知候选集；Transfer-Exchange 以 KKT 后的全 slot `task_utility_sum` 选择目标。四个样本 Episode 中 Cloud 占 80.17%，与给出的现象相符。
- **CLOUD_DOMINANCE_CAUSE = SUPPORTED**：Cloud 使用固定的 200 Mbps/任务上行等效速率（不随 Cloud 任务数分摊），同时使用有限的 40 GHz 共享 CPU 池。池确实被 KKT 分配并在样本的每个 slot 用满。Edge 的 CPU 和带宽则按节点组内任务分配；在资源重分配后的精确 Cloud→Edge transfer 候选中，记录到的总任务效用差均为负。证据支持“现有模型下 Cloud 常胜出”，不支持“Cloud 无限 CPU”或“Matching 错误地优先回退 Cloud”。
- **LOCAL_ZERO_CAUSE = SUPPORTED**：Training 样本中 93.50% 的任务到达实例没有 TV 本地服务副本；另有 1,087 个实例有本地缓存、进入候选集，但最终没有一个选为 Local。对最终稳定检查中记录到的 1,000 个 Cloud→Local 精确重分配候选，效用差全部为负。
- **SV_LOW_EXECUTION_CAUSE = SUPPORTED**：样本中有 10,849 个 SV 服务缓存命中的“任务—SV 节点”候选对，路径/正资源检查没有排除任何缓存命中对；最终仅 8 个任务分配给 SV。观察到的精确 Cloud→SV 重分配候选均未提高全 slot 效用。SV 的低份额主要与 KKT 后效用排序一致；仅凭这些耦合结果，不能把差异进一步归因到某个单一物理量。
- **MATCHING_KKT_IMPLEMENTATION_ERROR = NO**：本次未发现实际实现错误。四个 Episode 的 1,200/1,200 个 slot 均通过 Transfer-Exchange 稳定性检查；12 个 Cloud→SV/RSU/UAV 候选的 incremental KKT 差值与独立 full KKT 重算完全相同。此结论限定于本次样本和代码路径，不是对所有输入/所有可能匹配的形式化证明。

## 诊断方法与证据边界

使用当前正式 Training 池 `environment_data/proposed_joint_training/` 的四个完整 300-slot Episode，每个学习率取对应目录现存的 `latest_checkpoint.pt`，以 seed 20001–20004 各回放一个 Episode。回放的宏策略检查点分别已完成 136、64、120、88 个训练 Episode，**尚未达到 300 Episode**；因此这是当前可用检查点的行为诊断，不代表最终 300-Episode 检查点的最终比例。四个不同策略对应不同 seed，结果不是配对策略比较。

通过当前配置哈希校验加载检查点，并使用正式 evaluator 相同的底层调用：`aligned_train_steps`、`collect_dual_timescale_episode(deterministic=True)`、`execute_configured_slot`、`execute_slot` 和实际 `optimize_task_matching`。分别确认策略参数和 PPO optimizer step 计数在回放前后不变。单 Episode 的 `evaluate_pool` 会在完成回放后因其汇总器要求至少两个 realization 而拒绝聚合；所以按正式 evaluator 的冻结策略内层路径直接处理单 Episode，没有生成 evaluator 输出文件。只读了 Training 数据，未使用 Validation/Test，也未执行 PPO 更新。

当前 PowerShell 下 `conda activate adaptive_cache_mec` 失败；诊断使用该环境实际解释器 `C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`，已成功加载项目依赖。报告哈希：raw config SHA-256 `684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77`；semantic config SHA-256 `10f893fc549c4016bedb1a5389ad996f9a3ff646aa012596a7230c284b13f542`。

## 真实执行代码路径

1. `src/ppo_simple/interface.py` 的 `load_environment_context` 载入并验证正式配置、预测场景、Service Catalog、缓存节点和 TV local cache（117–150 行）。`aligned_train_steps` 从 Episode realization 对齐预测与真实轨迹；每个 slot 调用 `load_actual_tasks(tasks.csv, slot)` 并取实际 TV/SV 坐标（1186–1205 行）。`src/decision/task_allocation.py` 的 `load_actual_tasks` 只接受该 slot 中 `arrival == 1` 的行，并将 CSV 的 `service_id`、`input_size`、`cpu_cycles` 构成实际 Task（1499–1516 行）。
2. `src/ppo_simple/rollout.py` 的 `collect_dual_timescale_episode` 在每个 slot 生成冻结策略的 Cache/UAV 配置，并调用 `execute_configured_slot`（561 行起，916–933 行）。后者构建正式配置成本并转入 `src/decision/slot_environment.py:execute_slot`（825–905 行）。
3. `execute_slot` 根据最终 Cache 和 UAV 位置构建 Service Node 坐标，然后调用 `task_allocation.optimize_task_matching`（510–523 行）。其输入是实际任务、TV/SV 位置、当前 local cache、最终 edge cache 和运行时 cache config。
4. Matching 返回每任务最终 delay、vehicle energy、utility 及 CPU/BW allocation。slot 任务效用为 `Q = task_utility_sum`；环境 system utility 另减 Cache 与 UAV movement cost，即 `G = Q - C_ca - C_q`（`slot_environment.py:590–614`）。目标节点的单次任务选择比较使用 Q；同一候选匹配比较中 Cache/movement 配置保持不变。

## 合法节点与选择规则

| 目标 | 进入候选集的正式规则 | 采样中的观测 |
|---|---|---|
| Local | `service_id` 必须存在于该 TV 的 `local_cache_state`；无无线链路。 | 16,728 个任务到达实例中 1,087 个有本地副本，其余 15,641 个没有。合法 Local 候选 1,087 个，最终选择 0 个。 |
| SV / RSU / UAV | 当前最终 Cache 必须在该节点缓存任务服务；随后要求当前 TV/node 坐标存在且是有限二维向量、节点角色有效、CPU 和带宽为正、信道增益和 Shannon 速率为正。 | SV/RSU/UAV 的缓存命中后物理路径检查拒绝数均为 0。 |
| Cloud | Stage6 使用 `RESOURCE_AWARE_FALLBACK`，Cloud 始终保留为候选，包括存在 Edge 候选时。 | 每个任务均有 Cloud 候选。 |

规则对应 `build_feasible_targets`（`src/decision/task_allocation.py:235–322`）和 `_edge_path_is_feasible`（1809–1854）。Matching 只把缓存服务匹配作为 Edge 的服务可用条件；路径 helper 不做 `range_m` 距离门限、任务截止期或最大 delay 检查。Range 参数不能从配置注释直接当成 Matching 硬约束。当前样本无路径/正资源 rejection。

`optimize_task_matching` 先取 `feasible_targets[task][0]` 作初始匹配（1110–1145 行）。候选构造把 Local、所有合法 Edge 与 Cloud 放入同一任务目标列表；Transfer 和可行的两任务 Exchange 都计算候选（`enumerate_best_operation`，565–609 行）。候选通过 KKT 后，以 `candidate.task_utility_sum - current.task_utility_sum` 计算真实差值（690–809 行）。普通更新接受正效用改进；`RESOURCE_AWARE_FALLBACK` 下只有数值平局且 Cloud 数减少时才允许 Edge tie-break（`_better_candidate` 387–413 行；主循环 1193–1312 行）。每个收敛 slot 还执行 terminal `evaluate_only` 稳定性检查（1263–1289 行）。因此没有 Cloud 优先回退逻辑；反而是平局时 Edge 优先。

## Cloud、Edge 与 KKT 的实际实现

从 `system_config.toml` 解析到运行时 `CacheOptimizationConfig` 的实值为：

| 参数 | 运行时值 | 代码行为 |
|---|---:|---|
| Cloud CPU | `finite_pool`, 40,000,000,000 cycles/s | 对完整 Cloud 任务集合运行 CPU KKT 分配；capacity 为全池上限。 |
| Cloud uplink | 200,000,000 bit/s | Cloud `cloud_execution` 将该速率直接作为每任务速率；不产生 Cloud bandwidth allocation。 |
| Cloud bandwidth sharing | `resource_sharing = false` | 该字段控制带宽；并不关闭有限 Cloud CPU 池。Cloud 不进入 SV/RSU/UAV BW fraction allocation。 |
| Cloud delay factor | 15.0 | Cloud communication delay 为 `15 * input_bits / 200 Mbps`；vehicle transmit energy 为 `power * input_bits / 200 Mbps`，delay factor 不乘入能耗。 |
| Edge CPU/BW | SV 1 GHz / 20 MHz；RSU 10 GHz / 20 MHz；UAV 5 GHz / 10 MHz | 各 Edge 节点分别对分配到该节点的完整任务组运行 CPU 和 BW KKT fractions。 |

运行值来自 `cache_config_from_project`（`src/decision/cache_configuration.py:192–255`），且与配置 [system_config.toml](../configs/proposed/system_config.toml#L286) 的字段一致（286–309、358–360、520–530 行）。Cloud 计算与 Edge 的实际公式分别见 [system_model.py](../src/common/system_model.py#L90) 的 edge execution 和 131–170 行的 cloud execution；Cloud 固定 task uplink 与有限 pool CPU 是两种不同资源合同。

`allocate_resources_kkt` 将 Edge 任务按节点分组，对每个节点分别解 CPU fraction 和带宽 fraction KKT，并受节点总份额约束（`task_allocation.py:2341–2534`）。Cloud 不在 Edge 带宽 map 内（2521–2523 行）；`_allocate_cloud_cpu_kkt` 在 finite_pool 下对完整 Cloud task set 解共享 CPU pool KKT，再把份额乘以 40 GHz capacity（2537–2679 行）。Matching 改变 Cloud 任务集合时，incremental evaluator 清空旧 Cloud 分配并重算候选的完整 Cloud 池（3055–3110、3250–3305 行），没有沿用旧 Cloud CPU 份额。

所有目标统一使用 `task_utility(delay, vehicle_energy)`（`system_model.py:362–388`）：`omega_T * (1 - delay/T_ref) + omega_E * (1 - energy/E_ref)`。Local、Edge、Cloud 共享同一效用函数。Edge delay/energy 受共享 BW fraction、CPU fraction、当前信道影响；Cloud delay 用 15 倍通信项且 CPU 由共享池分配。Cloud 没有无限 CPU，但每任务 200 Mbps 等效上行不按 Cloud 任务数共享。这个非对称带宽假设是 Cloud 有竞争力的模型条件；本次未修改或建议调节物理参数。

## 四个完整 Episode 的实际结果

任务数均为该 Training realization 的 300 slots 中 `arrival == 1` 的实例数。表内角色计数来自每个 slot 最终 Matching。

| Macro Actor LR | seed | checkpoint 已训练 Episode | 到达任务 | Local | SV | RSU | UAV | Cloud | 稳定 slot |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 20001 | 136 | 4,190 | 0 | 0 | 170 | 548 | 3,472 | 300/300 |
| 5e-4 | 20002 | 64 | 4,201 | 0 | 1 | 223 | 533 | 3,444 | 300/300 |
| 1e-3 | 20003 | 120 | 4,191 | 0 | 7 | 418 | 585 | 3,181 | 300/300 |
| 5e-3 | 20004 | 88 | 4,146 | 0 | 0 | 299 | 534 | 3,313 | 300/300 |
| **合计** | — | — | **16,728** | **0** | **8** | **1,110** | **2,200** | **13,410** | **1,200/1,200** |

合计占比：Cloud **80.17%**、UAV **13.15%**、RSU **6.64%**、SV **0.048%**、Local **0%**。与题述的约 80/12/7/0/0 相符。

| 角色 | 缓存命中候选“任务—节点”对 | 缓存不匹配“任务—节点”对 | 因路径/正资源检查排除 | 最终任务分配 |
|---|---:|---:|---:|---:|
| Local（TV 本地服务副本） | 1,087 个任务实例 | 15,641 个任务实例 | 不适用 | 0 |
| SV | 10,849 | 72,791 | 0 | 8 |
| RSU | 8,513 | 24,943 | 0 | 1,110 |
| UAV | 8,346 | 25,110 | 0 | 2,200 |

Edge 表中的“命中/不匹配”按每个任务与每个同角色 Cache 节点分别计数，一个任务可能有多个合法节点候选；因此候选对数不能当作不同任务数。Local 服务副本缺失比例为 15,641/16,728 = **93.50%**。SV 有合法缓存命中，但候选最终效用很少胜出；并非通信/计算路径硬约束将 SV 排除。

### 资源与 paired utility 证据

四个 Episode 共 1,200 个 slot；每个 slot 的 Cloud CPU 使用量均约为 40 GHz capacity 的 100%，Cloud bandwidth allocation map 每个 slot 均为空。正式 Matching 记录的 185,012 次 KKT solve 全部收敛、无 max-iteration hit；最大 resource residual 约 `1.0e-12`，stationarity residual 最大 `1.42e-14`，complementary slackness residual 最大 `4.98e-11`。

以下 `ΔJ = J(单个 Cloud 任务转至目标节点、按候选状态重新分配共享资源) - J(当前完整 slot matching)`。负值表示 Cloud 当前方案的全 slot 总任务效用更高。每格给出终态稳定检查实际精确评估的候选数、均值及最小至最大；表内全部差值均为负数。

| LR | 目标 | 精确 paired transfer 数 | 平均 ΔJ | ΔJ 范围 |
|---:|---|---:|---:|---:|
| 1e-4 | Local | 405 | -0.09537 | [-0.14669, -0.04728] |
| 1e-4 | SV | 1,514 | -0.10101 | [-0.50925, -0.00453] |
| 1e-4 | RSU | 163 | -0.17855 | [-0.67542, -0.00530] |
| 1e-4 | UAV | 832 | -0.06918 | [-0.23624, -0.00044] |
| 5e-4 | Local | 183 | -0.09005 | [-0.13537, -0.05306] |
| 5e-4 | SV | 1,122 | -0.11543 | [-0.49605, -0.00075] |
| 5e-4 | RSU | 555 | -0.21589 | [-0.94495, -0.00039] |
| 5e-4 | UAV | 954 | -0.06911 | [-0.20844, -0.00010] |
| 1e-3 | Local | 215 | -0.08865 | [-0.13767, -0.04514] |
| 1e-3 | SV | 1,249 | -0.14139 | [-0.66124, -0.00281] |
| 1e-3 | RSU | 1,011 | -0.26239 | [-1.03928, -0.00061] |
| 1e-3 | UAV | 1,680 | -0.09484 | [-0.26256, -0.00079] |
| 5e-3 | Local | 197 | -0.09029 | [-0.13808, -0.05078] |
| 5e-3 | SV | 1,754 | -0.15790 | [-0.61859, -0.00086] |
| 5e-3 | RSU | 614 | -0.26610 | [-0.98861, -0.00055] |
| 5e-3 | UAV | 804 | -0.07751 | [-0.20790, -0.00041] |

这个 paired 比较不是把彼此不同任务、不同 KKT 上下文的单任务均值直接相减：每个候选都改变实际 Cloud task set，并重分配完整 Cloud pool 以及受影响 Edge 节点的任务组。`evaluate_matching_candidate_incremental` 的影响范围代码明确把有限 Cloud 池的旧/新任务集合全部纳入（`task_allocation.py:416–472、3055–3110`）。12 个 incremental/full 候选对照（四个 LR × SV/RSU/UAV）差值及 base utility 均完全一致，绝对误差均为 0。

## Local、SV 与 A–E 任务分类

**Local**：93.50% 的 task-slot arrival 根本没有本地服务副本。剩余 1,087 个实例都实际进入 Local 候选，但没有被选择；该样本中终态稳定检查有 1,000 个 Cloud→Local 精确 paired transfer 记录，全部 `ΔJ < 0`。这支持两层原因：服务缓存覆盖不足是主要候选约束，合法 Local 下 Cloud KKT 后总效用仍更高。Local CPU 为 1 GHz、Local execution 使用本地计算时延/能耗公式；现有证据不能把后一差异单独归因于 CPU 或能耗其中一项。

**SV**：16,728 个任务实例中，跨 SV 节点共有 10,849 个缓存命中“任务—节点”候选对；路径 helper 无拒绝。最终只选 8 个 SV。精确 paired Cloud→SV 候选全部 `ΔJ < 0`。SV 运行时 CPU 为 1 GHz，带宽 20 MHz，且在各 SV 节点内与该节点任务共享；缓存命中并不保证 KKT 后收益超过 Cloud。该证据支持“收益排序”解释，但不足以隔离 CPU、带宽、信道和任务输入组成各自的因果权重。

对 seed 20001（LR 1e-4）的 3,472 个 Cloud 最终任务，依据该 slot 的实际合法目标和 terminal stability check 精确 transfer 记录分类。`C` 的“接近”阈值在本报告定义为 `|ΔJ| ≤ 1e-3`；这是诊断分箱阈值，正式 Matching tolerance 为 `1e-9`。

| 类别 | 数量 | 判定 |
|---|---:|---|
| **A** | 1,943 | 没有任何合法 Edge 候选；复核也没有 Local 候选，因此 Cloud 是唯一候选。 |
| **B** | 874 | 有合法 Edge 候选，所有已完整精确评估的 Cloud→Edge 方案总效用都比 Cloud 低至少 `1e-3`。 |
| **C** | 2 | 最佳 Edge 差值在 `[-1e-3, +1e-3]` 内；观察值仍为负，且不构成正式 `1e-9` 平局。 |
| **D** | 0 | 没有发现 `ΔJ > 1e-9` 的 Cloud→Edge 精确 transfer 仍被 Cloud 保留。 |
| **E** | 653 | 存在 Edge 候选，但至少一个目标在 upper-bound screening 后没有逐项精确重算；因此无法给该任务列出完整的 paired 差值集合。所在 slot 的 Matching 稳定检查仍通过。 |

该分类仅对 seed 20001 的一个完整 Episode，其他三个 Episode 的 paired delta 是汇总级采样，不生成同等完整的 A–E 逐任务表。稳定性证明也仅覆盖实现定义的单任务 transfer 与两任务 exchange 邻域；不证明任意多任务联合改派的全局最优。

## Matching/KKT 实现问题判断

- **Cloud CPU 没有被当作无限资源：** 本次 1,200 个最终 slot 都使用完整 40 GHz finite pool。Cloud CPU KKT 对候选 Cloud 任务集合重分配，任务移动会触发全池重算。
- **Cloud bandwidth 没有按任务数错误分摊：** `resource_sharing=false` 与当前系统假设一致；Cloud rate 是每任务 200 Mbps 等效值。实现没有 Cloud BW fraction map。该点是当前明确的建模合同，而不是本次诊断发现的代码错误。
- **不存在 Cloud 优先回退：** `RESOURCE_AWARE_FALLBACK` 保留 Cloud 和所有合法 Edge 候选；初始目标序列也先列 Local/Edge，Cloud 放在后面。最终 Cloud 选择来自全 slot 效用比较。
- **效用口径一致：** Edge 和 Cloud 最终都通过同一 `task_utility` 计算；Matching 优化该效用总和。Cloud 的 psi=15 延迟惩罚有实际参与计算，未进入 vehicle energy 项，这与源码公式一致。
- **增量 KKT 暂未发现偏差：** 12 组抽样候选的 incremental 与 full 重算 `ΔJ` 一致；全样本 1,200 个 slot 稳定检查通过，KKT 诊断没有未收敛项。

结论标记：**CONFIRMED**：上述代码调用链、候选规则、运行参数映射、样本节点计数及 KKT pool 用量；**SUPPORTED**：Cloud 竞争力、Local 缓存覆盖不足、SV 效用较低是 observed distribution 的主要原因；**NOT_SUPPORTED**：Cloud 优先代码、Cloud CPU 无限/未共享、incremental/full KKT 在抽样中的实现不一致；**UNKNOWN**：本次未精确逐项记录的 653 个合法 Edge 候选、任意三任务以上联合交换的全局最优性、以及训练完成至 300 Episode 后各策略的最终节点比例。

## 本次操作边界

只生成本报告。未修改正式代码、系统配置、Matching/KKT/PPO、Cache 或 Reward；未启动正式训练、未运行 Validation/Test、未创建/覆盖 Checkpoint；无 commit/push。临时回放目录和脚本均已清理。
