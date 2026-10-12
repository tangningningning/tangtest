# 阶段4C-0：Cloud任务占比过高的根因定位

## 1. 范围与数据来源

- 只读核对最新Stage3B配置 `configs/proposed/stage3_cloud_finite_60ghz.toml`。其SHA-256为 `1b14a49f430aca3d4c795e576010d9aa0c5e71405701440b3aebaec348578527`，与3C-2轨迹元数据一致。
- 全Episode统计只读取 `reports/stage3c2_cache_uav_long_horizon_counterfactual_slots.jsonl` 中seed 30020、30023的`BASELINE`分支，每seed 300 Slot；没有重新推理或推进Episode。逐任务输入取对应已有 `environment_data/proposed_joint_validation/episode_{seed}/tasks.csv`。
- 四个固定状态的逐任务候选和执行信息复用 `runs/diagnostics/cloud_capacity_sensitivity/task_results.csv` 的`KEEP_STAY / F_C_TOTAL_60_GHZ`记录。其66个任务的执行节点与3C-2基准轨迹逐任务完全一致。
- 整Slot受约束迁移复用 `reports/stage3c4_cache_offloading_value_gap.md` 中五个RSU1案例。4A参数敏感性及3阶段参数文献审阅直接复用既有报告，不重复运行。
- 本地发现 `11.pdf`，未发现`12.pdf`。未写入4B数据、代码或配置；没有运行PPO、Validation、Test或完整仿真。

## 2. Cloud任务分类统计

按轨迹保存的`matching`、`cache_available_hit_task_ids`和当前Slot缓存位图分类。所有8,394个任务ID都能与同Slot到达任务输入匹配；逐任务比较发现，服务在全网缓存中的状态与轨迹保存的Cache可用候选ID **8,394/8,394一致**。因此A类可直接归因于当Slot服务未出现在任何SV、RSU或UAV缓存中；没有观察到“服务已缓存、但当前正式路径检查拒绝该Edge节点”的任务。

| Seed | 到达任务 | Cloud执行 | Cloud占比 | A：无合法缓存Edge候选 | A占Cloud | B：有Edge候选仍选Cloud | B占Cloud | 最终Edge执行 | Local |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30020 | 4,194 | 3,430 | 81.78% | 1,619 | 47.20% | 1,811 | 52.80% | 764 | 0 |
| 30023 | 4,200 | 3,497 | 83.26% | 1,967 | 56.25% | 1,530 | 43.75% | 703 | 0 |
| 合计 | 8,394 | 6,927 | 82.52% | 3,586 | 51.77% | 3,341 | 48.23% | 1,467 | 0 |

最终节点任务数合计：Cloud 6,927、UAV0 594、UAV1 533、RSU 326、SV 14、Local 0。A类3,586个请求中，服务0有882个、服务2有1,829个、服务3有875个；这三种服务在对应时刻全网Edge缓存中均不存在。其原因是缓存内容未提供该服务，而不是节点容量或无线链路把已有副本拒绝。3C-3已确认这些缓存节点存在合法替换空间，但可行替换空间并不意味着当前缓存覆盖了这些请求。

## 3. 当前代码中的候选与Matching语义

- `src/decision/task_allocation.py::build_feasible_targets()`（约235–322行）在`RESOURCE_AWARE_FALLBACK`下始终保留Cloud，并把所有合法缓存Edge节点加入候选；Cloud不是仅在Edge不可行时才出现。当前配置采用该模式（TOML约527行）。
- `_edge_path_is_feasible()`（约1809–1854行）要求节点角色、车辆/节点坐标、CPU、带宽、信道增益和Shannon速率为有限正值；没有最远距离、最低速率、SINR或链路中断门槛。配置中的SV/RSU/UAV `range_m`均注明只是服务评分距离尺度，不是Matching硬覆盖截断。
- 因此，“合法Edge候选”是当前公式和正值检查下存在可计算的缓存路径；它不保证链路速率足以竞争Cloud。距离很远的低速链路仍可能是合法候选，速度和资源份额影响后续效用。
- 两seed的600个基准Slot均记录`matching_stable=true`、`converged_by_no_improvement`和`resource_feasible=true`。这说明当前Transfer-Exchange Matching在其候选和局部操作范围内停止于无改善状态，不等价于全局混合整数最优证明。

## 4. 时延、能耗和候选选择的代表证据

### 4.1 四个固定状态的已执行任务

按C1四状态的Stage3B `F_C_TOTAL_60_GHZ`逐任务结果分组。组内任务本身由Matching选择，样本不是同一任务的配对迁移，因此下表只描述已执行子集，不把均值差解释为因果效应。

| 最终执行节点 | 任务数 | 平均总时延(s) | 通信(s) | 计算(s) | 平均车辆能耗(J/任务) |
|---|---:|---:|---:|---:|---:|
| Cloud | 53 | 4.7171 | 1.2924 | 3.4247 | 0.02344 |
| RSU | 5 | 3.7792 | 1.0631 | 2.7161 | 0.21211 |
| UAV | 8 | 4.3206 | 0.4856 | 3.8351 | 0.09688 |

Cloud已执行子集中有32个任务没有Edge候选；另外21个任务存在Edge候选仍选Cloud。已执行Edge子集平均时延较短但能耗较高，是选择后的样本差异；不能据此否定合法Edge候选在特定任务上的效用比较。

### 4.2 同Slot整组KKT受约束迁移

以下五项来自3C-4：强制一个任务转至RSU1后，重新为受影响Edge/Cloud任务组分配KKT资源，并比较整个Slot任务效用和。前四项使用候选#160缓存配置后才出现的RSU1路径，不代表自然基准Cache状态；最后一项还包含RSU1上的现有任务。车辆能耗按原公式由保存的通信时延和`p=0.1995262315 W`还原。

| Seed/Slot/Task/服务 | 当前Cloud通信+计算(s) | Cloud能耗(J) | 强制RSU通信+计算(s) | RSU能耗(J) | 整Slot效用变化ΔQ |
|---|---:|---:|---:|---:|---:|
| 30020/107/37/0 | 0.916 + 2.699 | 0.0166 | 7.795 + 2.606 | 1.5553 | -0.285262 |
| 30020/116/41/2 | 2.065 + 5.231 | 0.0375 | 3.449 + 6.113 | 0.6882 | -0.036864 |
| 30023/108/52/0 | 0.993 + 2.163 | 0.0180 | 1.714 + 2.584 | 0.3420 | -0.026541 |
| 30023/114/52/2 | 2.005 + 3.671 | 0.0364 | 3.462 + 5.470 | 0.6908 | -0.086263 |
| 30020/113/46/2，RSU1已有task51 | 2.180 + 5.707 | 0.0395 | 14.355 + 9.581 | 2.8642 | -0.856578 |

五项的整Slot效用变化均为负。四项RSU1独占时，路径速率为2.137–10.887 Mbps；已占用RSU1的案例中，迁入任务只分到0.5995 CPU比例、0.6940带宽比例，速率2.761 Mbps，既有任务也因共享而变慢。这里通信时延、车辆上行能耗和部分案例的计算时延共同造成差异；不能只用“缓存命中”推断应执行Edge。

### 4.3 UAV候选与实际UAV执行

- 30023/Slot100/task56/service6在基准状态有`UAV_0`和Cloud两个合法候选，最终选择Cloud。实际Cloud分配CPU为4.1765 GHz，通信1.4008 s、计算3.1858 s、总时延4.5866 s、车辆能耗0.02541 J。
- 使用同一Slot车辆坐标、UAV0坐标和当前空地Shannon公式，UAV水平距离790.78 m、斜距797.07 m、仰角7.207°、LoS概率0.06616。若单任务独占10 MHz和5 GHz，算得速率32.8475 Mbps、通信0.7754 s、计算2.6611 s、能耗0.15470 J、总时延3.4365 s。这个满资源点是链路/计算上界式对照，不是任务迁入后的真实KKT分配；同槽UAV0已有task9，迁入会发生资源竞争。本次没有保存该任务的整Slot强制UAV重分配结果，故不据此断言Cloud效用更高。
- 对照：30020/Slot100/task6/service6实际选择UAV0，时延2.8281 s（通信0.4423、计算2.3858），能耗0.08825 J，使用5 GHz CPU和10 MHz带宽。UAV能承担任务，且UAV链路不必然慢于Cloud；不同任务的距离、能耗、CPU竞争和带宽份额会改变联合效用。

## 5. Edge链路预算、Shannon公式与单位

源码位置：`src/common/system_model.py::ground_channel_gain()`与`uplink_rate()`（约53–87行）、`edge_execution()`（90–128行）、`cloud_execution()`（131–170行）；调用与候选检查在`src/decision/task_allocation.py`约1761–1919行。

- 地面增益为`g=β_g d^{-α_g}`；速率为`b B log2(1+P g/N)`。代码使用Hz、W、无量纲功率增益、bit/s，和`11.pdf`第5–6页的地面路径损耗与第6页Shannon公式一致。
- Stage3B值：`P=0.1995262315 W`；`β_g=1.6368e-5`（1m参考增益，约-47.86 dB）；`α_g=2.75`；节点带宽10 MHz；`N=3.16227766e-13 W`，即-95 dBm的**总等效噪声功率**，不是以W/Hz表示的噪声谱密度。
- 将这些配置代入同一全带宽公式：距离1,045.6 m得0.723 Mbps；1,920.3 m得0.139 Mbps；352.8 m得10.138 Mbps。与3C-4保存值0.723、0.139、10.139 Mbps相符，未发现单位换算或Shannon实现错误。RSU1样本低速率可由地面路径损耗和距离解释，不是单凭速率低就能判为Bug。
- 当前速率实现将`N`固定在SNR项中，再按带宽比例线性缩放吞吐；带宽比例变化时不将噪声功率改为`N0·bB`。这与`11.pdf`第6页所写的公式一致，但配置/论文没有提供噪声谱密度或接收机噪声系数来标定窄带份额下的物理噪声口径。它是待明确的模型假设，不是相对该公式的代码错误。
- 当前RSU/UAV合法性没有硬覆盖半径。Stage3B的`range_m=250/300 m`注释明确说不是Matching硬截断，所以352.8 m、1–2 km的路径仍可能进入候选池。它们是否属于目标部署的可服务范围，没有目标网络覆盖参数支持；不能从现有文献或仿真值断言为非法，也不应把它们当作有保证的物理链路。

## 6. Cloud上行合同与有限CPU池

### 上行

- Stage3B配置为`nodes.cloud.uplink_rate_bps=200e6`、`total_bandwidth_bps=200e6`、`resource_sharing=false`、`psi_c=11`（TOML约274–307、349–358行）。`task_allocation._allocate_cloud_cpu_kkt()`只对有限Cloud CPU池做KKT；`system_model.cloud_execution()`每次调用均把给定Cloud速率作为该任务完整速率，不按Cloud任务数拆分，也没有Cloud上行带宽KKT。
- 结果是：每个Cloud任务公式上各自得到200 Mbps；通信时延为`11D/200 Mbps`，等效为18.18 Mbps的`D/R`时延速率；车辆上行能耗则为`P D/200 Mbps`，`psi_c`不乘进能耗。这与`11.pdf`第2页的“蜂窝接入后经回传/核心网至远程Cloud”、第7–8页的`T_c=psi_c D/R_c+L/f_c`及`E_c=P D/R_c`公式结构一致。
- 11.pdf第7页将Cloud置于独立蜂窝链路，并说明它不加入SV/RSU/UAV的共享带宽资源池；没有给出本项目的`200 Mbps`数值。配置注释引用的Al-Shareeda等论文Table IV确有`T^up_iC=200 Mbps`的平均Cloud上行数值先例，但该文模拟设置为20 GHz Cloud总CPU、每车U[2,4] GHz，并非本项目60 GHz池，也不能证明目标网络在并发时向每个任务独立保证200 Mbps。参考：[作者上传的接受稿及Table IV](https://www.researchgate.net/publication/396572081_Novel_DT-assisted_Vehicular_Task_Offloading_for_Cloud_Edge_and_Hybrid_Deployments)，[DOI 10.1109/TVT.2025.3620810](https://doi.org/10.1109/TVT.2025.3620810)。
- 配置注释存在合同语义冲突：一处称“每任务固定等效速率”，另一处又称“200 Mbps作为单Cloud节点总uplink”；可执行代码实现的是前者，不是共享总池。文献支持独立蜂窝链路和200 Mbps数值先例，但没有证据说明目标系统是每任务独立保障还是一个共享200 Mbps总池。因此尚不能判定物理合同是否与部署不符。

### CPU池

- Cloud模式是`finite_pool`、总CPU 60 GHz；`_allocate_cloud_cpu_kkt()`对同Slot Cloud任务集合分配每任务CPU并限制总和不超过池容量。它不是5 GHz/任务，也不是60 GHz/任务。Edge的SV、RSU、UAV CPU和带宽分别在各节点内按KKT共享。
- C2两条完整基准轨迹的600/600个Slot均资源可行、Matching稳定、Cloud CPU池满额使用；池残差最大绝对值分别为0.060和0.060 cycles/s，超容量超过`1e-6 cycles/s`的Slot为0。C1四个固定状态也报告60 GHz池绑定且满足数值容差。因此CPU总池约束确实生效，没有发现实现超配。
- 60 GHz是实验锚点，非目标Cloud硬件标定；4A已做过20/40/60/80 GHz等参数敏感性，结果不能替代目标Cloud配额和并发资料。本任务未重复该分析。

## 7. 根因分级与一致性判断

| 判断项 | 结论 | 证据等级/边界 |
|---|---|---|
| A：缺少合法缓存Edge路径 | 解释了Cloud任务的51.77%；涉及服务0/2/3当Slot全网未缓存。 | 两seed完整300 Slot轨迹的逐任务计数；0个“服务全网已缓存但保存候选不可用”的任务。 |
| 无线链路不可达 | 不是上述A类的已观测根因。低速率Edge路径仍被当作合法候选。 | 当前资格代码仅查正速率，无距离/SINR门槛；具体无线覆盖是否符合部署假设未标定。 |
| B：有Edge候选仍选择Cloud | 占Cloud任务48.23%。五个RSU1强制迁移的整Slot效用均下降；样本支持Cloud在这些状态的联合效用较高。 | 五个受约束案例是局部证据，不证明所有Cloud任务或多任务联合重派都更优。UAV Cloud候选有一例未做整Slot迁移。 |
| C：物理参数/通信合同 | Shannon和时延/能耗公式与11.pdf一致；没有发现公式或单位实现错误。 | 200Mbps per-task/总上行语义冲突、`psi_c=11`数值、60GHz池和噪声口径均缺目标物理标定。 |
| Cloud CPU池约束 | 正常生效并在基准轨迹每Slot饱和。 | 600/600 Slot可行且没有超容量；这不证明60GHz对应实际云实例。 |

整体上Cloud高占比有两部分来源：约一半Cloud任务的请求服务当Slot根本不在Edge缓存；其余约一半虽有候选，但当前Matching/KKT在样本中因Cloud链路/能耗合同、CPU分配和Edge资源竞争而选择Cloud。RSU的低速率由距离衰减公式复现；Cloud的每任务独立上行及`psi_c`扩展时延又给予它固定而不随并发分摊的接入合同。它们是共同作用的模型假设，不是一个已确认的单一代码Bug。

## 8. 是否需要模型修改及最小后续范围

- **当前不建议修公式或调Cloud/Edge参数来压低Cloud占比。** Shannon公式、Cloud时延/能耗分解和60 GHz CPU池实现与当前系统公式一致；没有发现应立即修复的计算错误。
- **首先需要澄清Cloud 200 Mbps合同。** 若目标含义为“每辆车/每任务独立固定等效蜂窝速率”，当前实现可保留，最小改动是将配置字段/注释明确为per-task或per-vehicle，避免`total_bandwidth_bps`及“单Cloud总uplink”被误读。若目标含义为“整个Cloud接入总计200 Mbps”，则需在task allocation中新增共享Cloud上行带宽分配/KKT，并同步改通信时延、上传能耗、Matching评价和资源可行性统计；不能只改注释或`cloud_execution()`的一个常数。
- `psi_c=11`、60 GHz和噪声口径需要目标链路/实例证据后再标定。若需要有限无线覆盖或最低服务质量，也需先有目标部署的覆盖/中断/最低SINR资料，再把一致阈值纳入候选合法性；本次不臆造距离阈值。
- 因Cloud上行的per-task与总池含义未定，当前无法判断是否需要改系统资源约束；Stage4C-1若用于模型修正，应先确定该合同及目标部署证据。未经此确认，模型修改必要性和进入修正阶段的准备状态均为`UNDETERMINED`。

```makefile
STAGE4C0_DONE = YES
CLOUD_OFFLOADING_CLASSIFIED = YES (6927/8394; 82.52%)
NO_EDGE_CANDIDATE_FRACTION = 51.77%_OF_CLOUD (all due to service absent in Edge caches)
EDGE_AVAILABLE_BUT_CLOUD_SELECTED_FRACTION = 48.23%_OF_CLOUD
EDGE_LINK_BUDGET_CHECKED = YES
EDGE_RATE_IMPL_ERROR_FOUND = NO
CLOUD_ACCESS_RESOURCE_CONTRACT_CHECKED = YES
CLOUD_COMMUNICATION_MODEL_MISMATCH = UNDETERMINED (per-task code vs total-uplink config comment)
CPU_RESOURCE_CONSTRAINT_PASS = YES (600/600 slots feasible; no over-capacity)
DOMINANT_CAUSES = UNCACHED_SERVICE + CLOUD_UTILITY_ADVANTAGE_IN_TESTED_STATES + UNCALIBRATED_ACCESS_CONTRACT
MODEL_MODIFICATION_REQUIRED = UNDETERMINED
READY_FOR_STAGE4C1 = UNDETERMINED (clarify intended Cloud uplink contract first)
```
