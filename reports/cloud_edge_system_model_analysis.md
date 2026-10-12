# Cloud 与 Edge 系统模型合理性分析（阶段1，只读）

审查日期：2026-10-10（Asia/Shanghai）  
项目：adaptive_cache_mec

## 1. 范围与证据口径

本报告只分析当前静态源码、当前默认 TOML、正式 5e-4 运行的冻结配置、项目系统模型 PDF 与已完成的 Cache 反事实报告。没有读取 runs/diagnostics 下正在写入的 UAV 结果文件；没有运行项目 Python 模块、Matching/KKT、Validation、PPO 或其他实验；没有改动源码、TOML、环境数据或已有结果。唯一新增文件为本报告。

正式运行配置取 runs/formal_cache_economic_v2_lr_5e_minus_4/run_config.toml：配置名为 PROPOSED_CACHE_ECONOMIC_FEATURES_FIXED_LR_HORIZON_MASK_V2，Macro Actor LR 为 5e-4。该冻结文件中 Cloud 为 5 GHz、200 Mbps、resource_sharing=false、cloud_delay_factor=11；SV 为 1 GHz、RSU/UAV 各 5 GHz，三类 Edge 带宽均为 10 MHz；cross_slot_task_model=independent_per_slot。以上均直接核对冻结文件，不从默认配置推断正式运行参数。

证据优先级：当前实际源码与上述冻结配置用于判定实现和运行值；项目自己的 11.pdf（第 4–8 页）用于判定本项目公式合同；外部论文用于核对配置声明的文献参数。configs/proposed/system_config.toml 将多项 Cloud、SV、RSU、通信参数标成 CONFIRMED_DATA，但现有证据显示其中不少是模型设定或文献基线，不是项目实测数据，报告中不把状态标签当作文献证明。

另外，reports/communication_mathematical_model_audit.md 中 Cloud 总资源按 N_cloud 均分、reports/cache_offloading_coupling_audit.md 中 Cloud 始终在候选集的描述，与当前源码和冻结配置相反；这些材料对应早期实现，不能作为本阶段的当前实现依据。

### 源码定位

- configs/proposed/system_config.toml：SV/RSU/UAV 数值位于 158–242 行；Cloud 资源与 psi_c 位于 274–303 行；参数来源标签位于 593–684 行。冻结 5e-4 运行的同一组资源值位于 runs/formal_cache_economic_v2_lr_5e_minus_4/run_config.toml 的 274–303 行，macro_actor_learning_rate 在 556 行，cross_slot_task_model 在 101 行。
- src/common/system_model.py：Edge 公式 90–128 行；Cloud 公式 131–165 行；Air LoS/NLoS 模型 169–206 行；任务效用 358–384 行。
- src/decision/task_allocation.py：候选构造 224–291 行；Transfer-Exchange 枚举 425 行起；资源可行性检查 842–863 行；节点 CPU/带宽与物理路径检查 1587–1700 行；理想资源上界剪枝 1703–1830 行；KKT 分配 2182–2366 行；实际 KKT matching 评价 2478–2622 行；增量 matching 评价 2635 行起。
- src/decision/cache_configuration.py：禁止 Cloud sharing=true 的合同检查在 142–148 行，读取显式 resource_sharing 配置在 173–223 行。
- 项目 11.pdf：任务分配、边缘/Cloud 计算在第 4–5 页；通信与两类时延/能耗在第 6–8 页。已完成 Cache 反事实的汇总为 reports/cache_keep_update_counterfactual.md。
## 2. 核心结论

1. 项目自己的 11.pdf 定义了 Cloud 每任务等效计算频率与独立蜂窝链路，并未给 Cloud 增加边缘节点的共享资源约束。因此当前 Cloud 固定每任务资源在本项目公式层面是一个可识别的建模假设。但 5 GHz/任务、无限并发且不共享 CPU 的具体合同没有对应实测依据；配置所引的 2026 年论文并不支持“5 GHz/任务且无总量上限”。故 Cloud 资源是否适合目标实际系统尚不能确认。
2. Cloud 时延公式和车辆能耗公式与项目 11.pdf 一致：psi_c 只放大 D/R_c 的时延，不进入 pD/R_c 能耗。psi_c=11 的数值没有在所查主模型或引用论文中得到推导，且代码没有显式回程路由、传播 RTT、排队或结果下载时延；因此公式结构可保留，psi_c 的物理解释与数值仍未确认。
3. UAV 5 GHz 总 CPU、10 MHz 总带宽以及共享分配有直接文献结构与数值先例；当前代码也通过 KKT 在节点内共享 CPU、带宽。SV 和 RSU 的具体 CPU/带宽数值缺少同等级的直接来源，不能仅凭 TOML 注释称作已验证。
4. 比较 5 GHz 共享 UAV 与 5 GHz 每任务独享 Cloud，不是同一容量合同下的对照。若 Cloud 被刻意抽象为有充足后端资源的远端服务，这一不对称可以成立；若两者都代表有限节点，则缺少公平的总资源约束。现有资料不足以判断项目要采用哪种物理解释。
5. 当前 Matching 的 Cloud 兜底规则确实排除可行 Edge 有缓存的任务的 Cloud 选项，且“Edge 可行”并不检查实际 KKT 时延或共享负载。此规则不是真正的资源感知后备机制，应在进入阶段2前调整候选集合。

## 3. Cloud 资源与参数依据

| 项目 | 源码实际实现 | 冻结配置数值 | 文献依据与本项目假设 | 判断 |
|---|---|---|---|---|
| Cloud CPU | system_model.cloud_execution() 每次对单个任务直接使用传入 F_c；task_allocation.py 不将 Cloud 任务加入 Edge KKT 组，也不按 Cloud 任务数除 CPU。cache_configuration.py 明确要求 resource_sharing=false。 | cpu_cycles_per_s=5e9、total_cpu_cycles_per_s=5e9；两字段值相同，代码中 total 字段仍表示每任务资源。 | 项目 11.pdf（第 4–5、7–8 页）以 L/F_c 计算 Cloud 任务，并将 F_c 解释为 Cloud 为任务提供的等效计算频率，结构上允许固定 per-task 假设；但没有给 5 GHz 数值或无限并发容量。配置引用的 Al-Shareeda et al. 2026 论文模型则为单个 Cloud 总容量 F_C，满足所有车辆分配之和不超过 F_C；其仿真参数为 F_C=20 GHz，单车 F_C^V_i 从 U[2,4] GHz 取值。该来源不支持当前 5 GHz/任务且不设总量上限的数值合同。[论文 DOI](https://doi.org/10.1109/TVT.2025.3620810) | 项目公式内部一致；引用与数值的对应关系错误已确认。固定每任务模型是否符合目标部署仍属 UNCONFIRMED。 |
| Cloud 上行 | cloud_execution() 使用固定 R_c；不参与 Edge 的带宽分配或 KKT。 | uplink_rate_bps=200e6、total_bandwidth_bps=200e6；resource_sharing=false。两个字段在当前模型是每任务等效值，不是多个 Cloud 任务共用再均分的 200 Mbps 总带宽。 | Al-Shareeda et al. Table IV 给出 T_up_iC=200 Mbps，支持其场景中的每车 Cloud 上行速率数值；该论文还分别列出 400 Mbps 下行速率。项目 11.pdf 明确 Cloud 走独立蜂窝链路，不进入 SV/RSU/UAV 的共享带宽约束。论文数值是仿真基线而非本项目蜂窝实测；论文也不能证明本项目环境下绝无 Cloud 接入竞争。 | 200 Mbps 可保留为有文献先例的每任务等效速率；不要把 total_* 字段名误读为共享总量。总链路是否应随并发共享仍属建模假设。 |
| Cloud 时延系数 | 代码计算 T_c=psi_c D/R_c + L/F_c；车辆能耗为 pD/R_c，不乘 psi_c。Cloud 传输时延是 input upload 的倍数，不是单独的常数 RTT 项。 | cloud_delay_factor=11。 | 项目 11.pdf（第 7–8 页）给出相同结构并规定 psi_c>=1，但没有给 11 的数值。所引 2026 论文的 Cloud 延迟另有上传、计算、结果下载与 2D_C/s 传播项；其模型没有将这些项推成 psi_c=11。项目配置注释说 psi_c 吸收回传/核心网附加时延，这是项目假设，不能作为文献验证。 | 公式形式和能耗拆分与项目主模型一致；psi_c=11 及“代表哪些网络延迟”的映射 UNCONFIRMED。不要为让 Edge 更快而随意调大或调小。 |
| Cloud 来源状态标签 | 当前 source/config 明确采用固定 Cloud per-task 模型。 | system_config.toml 把 Cloud CPU、200 Mbps、resource_sharing=false、psi_c 等字段标成 CONFIRMED_DATA。 | 所查来源只有 200 Mbps 上行有直接相同数值先例；5 GHz/任务、独享总量语义与 psi_c=11 均未被所引论文确认。 | 参数来源分类过强：应改成逐参数注明 LITERATURE_DERIVED（仅实际映射的参数）或 EXPERIMENT_DESIGN/MODEL_ASSUMPTION，而非统一 CONFIRMED_DATA。标签修正本身不改变物理环境。 |

Al-Shareeda et al. 的模型同时说明 Cloud 可以有远端链路与有限总计算容量；其参数不是本项目的直接标定。核对的是该论文作者接受稿及其 Table IV，未逐项比较最终 IEEE 排版版本。该论文的仿真结果甚至在其自身设定下出现 CT 平均时延低于 ET；这只能说明 Cloud 在某些设定下可以更快，不能移植成当前系统的结果或保证 Edge 总体更差/更好。项目不应通过人为惩罚 Cloud 来强迫 Edge 获胜。

## 4. Edge 资源与竞争

当前统一计算链路如下（F 为 cycle/s，B 为 Hz，R 为 bit/s）：

- SV：1 GHz CPU、10 MHz 带宽；服务必须在该 SV 缓存。每个 SV 的所有卸载任务共享 CPU 与带宽比例。
- RSU：5 GHz CPU、10 MHz 带宽；服务必须在 RSU 缓存。每个 RSU 的所有卸载任务共享 CPU 与带宽比例。
- UAV：5 GHz CPU、10 MHz 带宽；服务必须在 UAV 缓存。每架 UAV 分别形成一组 KKT CPU/带宽竞争。

KKT 对每个 Edge 节点分别约束 sum_v w_vm<=1、sum_v b_vm<=1，单任务资源份额严格为正并有 1e-6 的数值下限。CPU 和带宽分别求分配；CPU 时延为 L/(wF)，上传时延为 D/[bB log2(1+P h/sigma^2)]。它们不是“每个任务都能拿满 5 GHz 和 10 MHz”的独享模型。共享负载增加时，任务的 w、b 通常下降，计算和传输时延因倒数项放大。若 n 个相同任务在同一节点按 w=b=1/n 分配，当前固定噪声模型下两项时延均约放大 n 倍；这是解释资源竞争的等份示例，不是实际 Matching 结果，KKT 可以按任务系数不均匀分配。

- SV/RSU 地面信道为 h=beta_g d^(-alpha_g)，上传用 Shannon 速率；当前位置进入 d。当前 Matching 不把 range_m 当硬覆盖距离，因此有限距离上的正但很小速率仍可能通过路径检查。
- UAV 信道根据水平距离与固定高度形成三维距离和仰角，以 S 形 LoS 概率混合 LoS/NLoS 路径增益；位置变化因此改变 h、Shannon 速率、上传时延与车辆上传能耗。UAV 高度在当前默认与冻结配置均为 100 m。
- 项目源码的 Air channel 公式是当前配置，不等同于 Li et al. 2021 的自由空间 LoS 公式。Li et al. 的 Table 1 确实给出 UAV 总 CPU 5 GHz、总带宽 10 MHz、高度 100 m；其模型把 CPU 与带宽资源在接入设备间共享，支持这些 UAV 数值/资源结构作为一个文献基线。该论文采用不同任务大小、算力密度、发射功率和信道环境，不能据此宣称本项目所有 Air 参数已校准。[Springer 原文及 Table 1](https://doi.org/10.1186/s13638-020-01861-8)
Li et al. 的速率公式把接收噪声写成带宽相关项 B b N_0；本项目 11.pdf 与 system_model.py 则把 sigma^2 当作固定总噪声功率，SNR 不随带宽份额 b 改变，速率按 bB 线性缩放。这是两种不同噪声口径。当前实现与本项目 11.pdf 一致，但仅借用 Li 的 5 GHz/10 MHz 数值，不等于逐项复现其无线速率模型。
- RSU 的 5 GHz、SV 的 1 GHz、地面 10 MHz 与当前任务工作负载没有在本阶段找到可逐项对照的直接硬件/文献标定；它们应被称为实验基线或模型假设。不能把 UAV Table 1 的 5 GHz 自动当作 RSU 数值依据。

因此，“5 GHz UAV 总量共享”有直接 UAV-MEC 参数先例；“5 GHz Cloud 每任务独享”则是另一个更宽松的资源合同。相同标称 5 GHz 不等于同等资源：一个节点的所有 UAV 任务分这 5 GHz，Cloud 上每个并发任务都各自使用完整 5 GHz。若 Cloud 表示超大规模云平台，该抽象可以是有意的；若描述同一有限单节点的真实比较，则当前模型给 Cloud 额外并发 CPU 而未给 Edge。

## 5. Cloud 与 Edge 实际时延公式

### 5.1 代码/模型一致性与单位

项目模型 PDF 的 Edge 公式为 T_vm=D_v/R_vm+L_v/(w_vm F_m)，Cloud 公式为 T_vc=psi_c D_v/R_c+L_v/F_c。当前 system_model.py 实现一致：

| 项 | 当前 Edge | 当前 Cloud | 单位 |
|---|---|---|---|
| 通信 | D/R，其中 R=bB log2(1+P h/sigma^2) | psi_c D/R_c | bit ÷ (bit/s) = s |
| 计算 | L/(wF_m) | L/F_c | cycle ÷ (cycle/s) = s |
| 回传/额外网络 | 没有单独的回传、排队或结果下载项 | 由 psi_c 倍乘输入上传时延作为合并代理 | s；不是显式每跳 RTT |
| 车辆上传能耗 | P D/R | P D/R_c，不含 psi_c | W·s = J |
| 共享资源 | w、b 由同节点全部 Edge 任务的 KKT 分配 | 每个 Cloud 任务固定获得 R_c、F_c，不共享、不进 Edge KKT | — |

项目 11.pdf 也说明结果数据相对小而忽略返回下行时延和车辆能耗。因此当前纯执行公式和项目主模型吻合。冻结配置使用 independent_per_slot，不跨 slot 保留队列状态；这不是带等待时间的排队系统。不能把 psi_c 误写成车辆 radio energy 的倍数。

### 5.2 Cloud 可能更快的条件

令 R_e 为某 Edge 任务经过实际 KKT 带宽份额后的速率，w 为其实际 CPU 份额，则纯时延差为：

    T_edge - T_cloud = D(1/R_e - psi_c/R_c) + L(1/(w F_e) - 1/F_c)

Edge 低时延当且仅当右侧小于 0。对于当前 5 GHz UAV/RSU 与 5 GHz 每任务 Cloud，psi_c=11、R_c=200 Mbps：

    R_c / psi_c = 18.18 Mbps（Cloud 通信时间的等效未放大速率）
    T_edge - T_cloud = D(1/R_e - 1/18.18 Mbps) + L/(5 GHz)(1/w - 1)

因此在纯时延比较中：若 Edge 实际速率不超过 18.18 Mbps，且 w<=1，则 Cloud 通信不慢于 Edge，Cloud 计算也不慢于 Edge；Edge 不会因此自动获得低时延优势。若 Edge 实际速率高于 18.18 Mbps，Edge 的上行节省仍须大于共享 CPU 造成的 L/(5 GHz)(1/w-1) 计算时延差，Edge 才更快。SV CPU 只有 1 GHz，计算项更可能由 Cloud 占优。该条件不包含能耗效用，因此不能直接替代 Matching 的综合 utility 比较。

静态量级示例：按配置基础输入范围 500–5000 KB、1 KB=8000 bit、计算密度 400–1000 cycle/bit，不计任务扰动、共享、队列或实际信道，Cloud 的 psi_c 上传项为约 0.22–2.20 s，5 GHz 计算项约 0.32–8.00 s，总计约 0.54–10.20 s。该区间只是代入配置基础边界的公式示例，不是任何仿真测量或典型平均值。

## 6. Matching 候选与 Cloud 兜底

当前 src/decision/task_allocation.py 的 build_feasible_targets() 实际规则是：保留本地缓存目标；收集缓存了该服务且通过当前链路检查的 SV/RSU/UAV；若 edge_targets 非空，只返回这些 Edge 目标；仅当 edge_targets 为空时才追加 CLOUD。因此：

1. 是，当前规则严格要求只要有至少一个可用缓存 Edge，就从该任务候选中排除 Cloud。Cloud 只在无可用缓存 Edge 时兜底。
2. “可用”主要由服务缓存、当前坐标/角色有效、CPU>0、带宽>0、信道增益>0、Shannon 速率>0 判定。没有速率阈值、硬距离/覆盖限制、剩余资源门槛、实际任务时延条件；range_m 不是 Matching 硬截断。
3. build_feasible_targets() 不求实际 KKT，也不比较 Edge/Cloud 时延。其后的 Transfer-Exchange 搜索用单任务 transfer 或双任务 exchange 生成匹配。候选会先做不共享资源理想效用上界筛选；未被安全剪枝的候选再由 exact evaluator 对受影响的共享 Edge 节点重算 KKT 和任务效用。因此实际候选评分考虑 Edge 的 KKT 资源耦合，但 Cloud 被候选门控排除时根本没有机会与 Edge 比较。这个搜索是有限邻域的局部改进，并非对所有任务联合分配的全局最优证明。
4. 因而，新增缓存命中可把一个原先可走 Cloud 的服务变成仅能走缓存 Edge；即使后续 KKT 给该 Edge 很小的 w/b、时延高于 Cloud，Matching 也没有 Cloud 作为替代项。增加 Cache hit 反而增加时延在模型上完全可能。

资源感知的 Edge 优先设计建议是：把所有有服务的 Cloud 始终保留为候选，并继续保留全部缓存 Edge；用当前整组任务的实际 KKT evaluator 比较 Transfer/Exchange 后的系统 utility。若要“Edge 优先”，可以只在 utility 相同或容差内作为稳定 tie-break，不能在缓存命中后先验删掉 Cloud，也不能用独占 CPU/带宽的 Edge 理想时延替代实际 KKT。当前 exact KKT 候选评价机制可以保留。

## 7. 与已有 Cache 反事实结论的对应

仅引用 reports/cache_keep_update_counterfactual.md，不重跑实验，也不读取其报告所列出的正在运行 UAV 诊断文件。该报告中的 4 个状态里，强制 UPDATE 均提高缓存命中、减少 Cloud 分配，却降低 Return 并提高平均时延和车辆能耗：

| Seed / Slot | Cache hit KEEP → UPDATE | Cloud tasks KEEP → UPDATE | 平均时延 KEEP → UPDATE | 车辆能耗 KEEP → UPDATE | Return 差 |
|---|---:|---:|---:|---:|---:|
| 30020 / 100 | 1769 → 2057 | 1030 → 742 | 10.418 → 15.460 s | 1134.596 → 1814.987 J | -365.611 |
| 30020 / 200 | 923 → 1031 | 463 → 355 | 10.994 → 14.888 s | 573.891 → 839.953 J | -140.801 |
| 30023 / 100 | 1411 → 1491 | 1390 → 1310 | 7.521 → 10.123 s | 677.310 → 948.203 J | -178.467 |
| 30023 / 200 | 637 → 663 | 741 → 715 | 6.655 → 7.877 s | 303.291 → 372.321 J | -42.324 |

这些观测与当前候选门控相容：Cloud 分配下降、Cache hit 增加可由 UPDATE 后 Edge 候选扩大且 Cloud 被排除解释一部分。转到 UAV/RSU 后，任务竞争共享 CPU 和带宽，KKT 分配可使时延与能耗上升。Cloud 的 per-task 固定资源也可能使其在高 Edge 负载下相对有利；但 psi_c=11 又增加 Cloud 输入通信时延。现有报告没有把每任务效用差拆解成通信、计算、能耗、w/b 份额，也没有比较 351 个 UPDATE 候选；它覆盖两个 seed、两个时点、每点一个最高评分 UPDATE。因此不能由这些聚合结果确认根因。

证据分层：

- **已确认机制**：Cloud 兜底门控会排除存在缓存/正链路的 Edge 任务之 Cloud 候选；UAV/RSU CPU 和带宽是共享的；Cloud 当前每任务固定资源。
- **有机制支持但未证实为这四个差值根因**：UPDATE 造成更多任务进入 UAV/RSU 后，CPU 或带宽竞争导致时延/能耗上升；Cloud 每任务独享资源有竞争优势；Cache gate 造成被迫离开 Cloud。
- **仍待验证**：哪个因素对四个 Return 差值贡献最大；RSU/UAV 各组的 KKT 份额变化；Transfer-Exchange 局部搜索是否漏掉更优组合；Cloud 每任务资源的数值是否现实。

## 8. 模型改进建议与阶段2边界

| 项目 | 当前模型 | 发现的问题 | 依据 | 建议 |
|---|---|---|---|---|
| Cloud 计算 | 每任务固定 5 GHz、完全不共享 | 项目 11.pdf 支持 per-task 公式形式，但没有数值；所引 Cloud 论文为 20 GHz 总量、2–4 GHz/车且有总量约束，当前文献标注不准确；与共享 UAV 5 GHz 总量不对称 | system_model.py / cache_configuration.py / 冻结 run_config；11.pdf pp.4–8；Al-Shareeda et al. 2026 Table IV、P1 | **阶段1不改数值**。先决定研究对象是“资源充足 Cloud per-task”还是“有限总容量 Cloud”。修正文献来源/CONFIRMED_DATA 标签；若选有限 Cloud，必须在阶段2显式建模共享，不能仅把 5 改小 |
| Cloud 通信 | 每任务 200 Mbps，psi_c=11 只乘通信时延 | 200 Mbps 有表格先例；psi_c 的值和物理映射未证实。配置中的 total_bandwidth 名称容易被误解为总带宽，但当前 contract 是 per-task | system_model.py；冻结 run_config；11.pdf pp.7–8；Al-Shareeda et al. Table IV | 保留 200 Mbps 与当前公式作为冻结假设；把 psi_c=11 重新标为待论证的模型假设，后续做有依据的敏感性核验。不要用增大 psi 保证 Edge 更快 |
| SV 资源 | 1 GHz / 10 MHz；共享 KKT | 共享结构与本项目模型一致；数值未找到直接标定来源 | system_config.toml；11.pdf pp.4–7 | 物理值暂保留；补充准确来源或明确为实验设计参数 |
| RSU 资源 | 5 GHz / 10 MHz；缓存约束、共享 KKT、地面 Shannon 链路 | 共享结构合理；5 GHz 数值没有在所查资料中得到直接 RSU 支持 | system_config.toml；11.pdf pp.4–7；当前 KKT 代码 | 数值暂保留，不套用 UAV 文献或不同部署论文的 RSU 参数；补足来源/系统规格 |
| UAV 资源 | 每架 5 GHz / 10 MHz；KKT 共享；100 m；位置进入空地链路 | 5 GHz、10 MHz、100 m 有直接 UAV 文献先例；本项目 urban LoS/NLoS channel 与任务/流量规模不同，不能称作完整复现 | system_config.toml；system_model.py；Li et al. 2021 Table 1 | 保留当前值和 KKT 共享结构；确认 channel 参数对应的城市环境来源即可，不为偏向 Cloud/Edge 改值 |
| Matching 候选 | 有任一缓存 Edge 正链路即排除 Cloud | 确定造成可用 Edge 时不能回退 Cloud；没有 KKT 资源/时延门槛 | task_allocation.py::build_feasible_targets() | **需要改**：每任务始终保留 Cloud 和全部可行 Edge；保留 exact KKT 整组候选评价。Edge 优先只作同效用 tie-break（如确有需要） |
| Cloud 兜底 | 只有没有缓存/正链路 Edge 时存在 | 对“边缘优先、Cloud 后备”而言不满足资源感知后备原则；任意正速率 Edge 都会阻止 Cloud | task_allocation.py::_edge_path_is_feasible() / build_feasible_targets() | **需要改**：Cloud 作为常驻备选，让 Matching 在当前共享资源状态下比较；不加新 deadline 或惩罚项 |

### 必须改、暂时保留、待实验确认

- **阶段2前必须改的算法合同**：将 Cloud 加入所有任务的候选集合，避免由缓存命中先验锁死 Edge。若仅保留当前 fixed-per-task Cloud 模型，最小必要函数为 task_allocation.build_feasible_targets()；当前 exact KKT evaluator 已足以给候选集合内的实际 KKT 任务 utility 评分。
- **必须改的文献/状态说明**：不能继续把 Cloud 5 GHz/任务和 psi_c=11 标为已由所引论文确认。应明确区别代码实现、配置值、文献先例、实验设计参数和待确认假设。修改状态注释不会改变物理任务执行公式，但会改变冻结配置文本/哈希与来源记录。
- **暂时保留的数值**：Cloud 200 Mbps、psi_c D/R_c 公式形式、UAV 5 GHz/10 MHz/100 m、RSU/SV CPU 与带宽现值。此处“保留”表示本阶段不按主观判断改数，不表示全部数值已验证。Cloud 5 GHz/任务和 resource_sharing=false 仅在明确承认“Cloud 资源充足、每任务固定”合同后才能作为正式比较前提。
- **需要后续实验确认**：Cloud 候选开放后 Cache UPDATE 的任务路由、KKT 份额、延迟/能耗/Return 是否改善；如果要校准 Cloud 资源模型，还需用固定任务和相同状态比较 per-task 与有限总量合同、通信系数的有据范围。阶段1没有运行这些实验。
- **会改变正式训练环境的调整**：改变 build_feasible_targets() 候选规则；改变 Cloud 5 GHz/CPU-sharing 合同；改变 psi_c、200 Mbps、任一 Edge CPU/带宽/信道参数；这些会改变 Matching、时延/能耗或效用，正式训练/Validation 结果不再可直接沿用。仅修正文献来源标签不改变数值动力学，但会改变 run_config 文本与溯源信息。
- 不建议也不依据现有证据新增截止时间、惩罚项或降低 Cloud 性能以确保 Edge 获胜。

### 阶段2最小代码范围

1. 若 Cloud 继续采用 fixed-per-task 合同：仅需改 task_allocation.build_feasible_targets()，使每个任务均有 CLOUD 候选；保留 task_allocation.enumerate_best_operation() 与 evaluate_matching_candidate_incremental() 的实际 KKT 评价路径。若希望 Edge 在数值容差内优先，只调整稳定 tie-break，不引入额外 reward。
2. 若 Cloud 改成有限共享资源：除候选门控外，至少需要改 cache_configuration.CacheOptimizationConfig.validate()/from_project_config()、task_allocation.allocate_resources_kkt()、KKT 云任务分组与 evaluate_matching_with_kkt()，并让增量评估的 impact scope 覆盖所有共享 Cloud 任务；system_model.cloud_execution() 需接收实际分配到的 Cloud CPU/带宽。当前代码主动禁止 cloud_resource_sharing=true，因此不能仅改 TOML 开关。
3. 若把 psi_c 拆成明确回传/核心网时延，需更新 system_model.cloud_execution() 及其配置输入，并在完整任务 utility 路径一致使用；不能把该改动混同为当前 psi_c 数值的验证。

## 9. 阶段1最终状态

~~~ini
CLOUD_RESOURCE_MODEL_REASONABLE = UNCONFIRMED  # 项目主公式支持固定每任务假设；5 GHz/任务和无限并发未获目标部署/所引 Cloud 论文支持。
CLOUD_DELAY_MODEL_REASONABLE = UNCONFIRMED  # 公式匹配 11.pdf；psi_c=11 的物理映射未验证，且没有显式网络 RTT/回传路径。
EDGE_RESOURCE_MODEL_REASONABLE = UNCONFIRMED  # UAV 数值及共享结构有文献支持；SV/RSU 数值及本项目负载标定不足。
EDGE_CLOUD_COMPARISON_FAIR = UNCONFIRMED  # Edge 总量共享而 Cloud 每任务独享；意图可能是资源充足 Cloud，但真实合同未确认。
CLOUD_FALLBACK_RULE_NEEDS_CHANGE = YES  # build_feasible_targets() 在存在任一正链路缓存 Edge 时删除 Cloud。

CONFIRMED_MODEL_ISSUES = [Cloud/Edge候选门控阻止资源感知Cloud回退；所引Cloud论文不支持5GHz每任务无总上限；多个模型假设被标为CONFIRMED_DATA]
UNVERIFIED_MODEL_ASSUMPTIONS = [Cloud每任务无限资源是否代表目标部署；psi_c=11的物理映射；SV/RSU数值的直接校准；Cloud链路是否随并发共享]

RECOMMENDED_SYSTEM_MODEL_CHANGES = [阶段2先确认Cloud资源合同并修正来源分类；保留本项目时延/能耗公式形式和现有物理参数，除非获得直接证据]
RECOMMENDED_MATCHING_CHANGES = [每个任务始终保留Cloud及所有缓存Edge候选；在真实共享KKT utility上比较；Edge偏好仅作同效用tie-break]

READY_FOR_STAGE_2 = NO  # 需先确认Cloud是资源充足的per-task服务还是有限总容量节点；阶段1不擅自选定。
~~~


