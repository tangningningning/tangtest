# 阶段3并行任务A：系统参数文献依据

## 范围与结论

只读核对了 Stage3B 配置、旧默认/冻结配置、当前公式与资源分配代码、项目 11.pdf 和现有文献分析报告。没有运行仿真或生成新任务。

11.pdf 第1–8页给出系统时隙、任务、计算资源、带宽和 Cloud/Edge 公式；第24–26页描述仿真场景，但本文件没有可核对的资源数值表（第26页还写有“参数应在后续参数表中统一列出”）。因此，11.pdf 能支持公式和资源约束结构，不能直接确认本项目的 60 GHz、200 Mbps、psi_c=11 或 SV/RSU/UAV 数值。

| 参数 | Stage3B 当前值及实际合同 | 公式/项目模型依据 | 文献数值依据与场景差异 | 参数状态与后续敏感性 |
|---|---|---|---|---|
| Cloud CPU 总量 | finite_pool，60e9 cycles/s。task_allocation._allocate_cloud_cpu_kkt() 对当前 Cloud 任务组分配 f_i 并约束 sum(f_i) <= 60e9；5e9 字段仅为 legacy 兼容合同。 | 11.pdf 第4–5页只定义 Cloud 每任务等效计算频率 F_c 及 L_v/F_c，未定义 Cloud 总池约束。Stage3B 的总池是对资源合同的扩展。 | 项目已有 Cloud 文献审阅记录：Al-Shareeda 等的接受稿仿真设置采用 20 GHz 总 Cloud CPU、每车约 U[2,4] GHz，并非 60 GHz；其场景与本项目负载、节点与时延项不同。没有目标 Cloud 实例规格或并发压测作为 60 GHz 的物理标定。 | 实验设计锚点，未标定。需要敏感性分析。已有 reports/cloud_capacity_sensitivity.md 比较 20/40/60/80 GHz，但只有四个固定状态，不能视作物理定标或覆盖不同负载。 |
| Cloud 上行速率 | 200 Mbps，实际每个 Cloud 任务都将相同 R_c 传入 cloud_execution()；不按 Cloud 任务数分摊带宽。resource_sharing=false。配置中的 total_bandwidth_bps 字段名及注释容易让人理解成总池，但当前路径没有共享 Cloud 带宽池。 | 11.pdf 第7页称 Cloud 使用独立蜂窝链路，不参加 SV/RSU/UAV 共享带宽分配；以 R_c 计算 Cloud 上行。公式为 T_tr,c = psi_c D_v/R_c，车辆上行能耗 E_tr,c = p_v D_v/R_c。 | 项目已有来源核对报告记录，Al-Shareeda 等 Table IV 有 200 Mbps 的 Cloud 平均上行速率数值先例；这可支持“有相同仿真数值先例”，但不证明目标网络在并发时为每个任务保证 200 Mbps，也不证明聚合 Cloud 接入链路无竞争。该论文的任务、车辆密度和 Cloud 传播/结果下载模型与本项目不同。项目报告所核对的是作者接受稿，未逐项对照最终 IEEE 排版版。 | 有文献数值先例；本项目独立 per-task 合同仍是建模假设，未做目标网络标定。需要核对并发语义与敏感性分析。 |
| Cloud 时延系数 | psi_c=11，仅用于 Cloud 通信时延，不进入车辆上行能耗。 | 11.pdf 第7–8页给出 T_c = psi_c D_v/R_c + L_v/F_c，并要求 psi_c >= 1；忽略结果下行时延和能耗。当前 system_model.cloud_execution() 保持该公式。 | 11.pdf 没有给出 psi_c=11。Al-Shareeda 等的模型另列上传、执行、结果下行及传播项，不能从中推出本项目的 11。把回传/核心网延迟并入 psi_c D/R_c 是本项目假设；它是随输入量线性缩放的扩展时延，不是已测固定 RTT。 | 公式结构受支持，数值与物理映射未标定。需要以目标 Cloud 路径数据验证，并做敏感性分析。不建议为改变 Edge/Cache/UAV 收益而人为调整。 |
| SV CPU / 带宽 | 每个 SV 1 GHz / 10 MHz；节点内由 KKT 在任务间共享。 | 11.pdf 第4页有服务节点 CPU 比例总和 sum_v w_v,m <= 1；第6页有带宽比例总和 sum_v b_v,m <= 1 及 Shannon 速率公式。没有 SV 专属数值。 | system_config.toml 将该值称为文献校准基线，但现有项目资料没有找到能直接对应本项目 SV 的 CPU/带宽原始表格或硬件测量。 | 共享结构有论文依据；数值属于项目基线/实验设计，未直接标定。需要来源核验或敏感性分析。 |
| RSU CPU / 带宽 | 每个 RSU 5 GHz / 10 MHz；每个 RSU 单独共享资源池。 | 同上，11.pdf 第4–7页统一覆盖服务节点资源和上行模型，没有 RSU 专属数值。 | 项目现有材料未找到直接支持本项目 RSU 5 GHz / 10 MHz 的原始参数表或目标 RSU 规格。不能把 UAV 参数先例移作 RSU 证据。 | 结构有依据，数值为实验设计基线，未直接标定。需要来源核验或敏感性分析。 |
| UAV CPU / 带宽 | 每架 UAV 5 GHz / 10 MHz；每架分别在节点内共享。 | 11.pdf 第4–7页支持 UAV 纳入服务节点 CPU/带宽共享约束及空地速率模型，不给数值。 | Li 等的 UAV-MEC 论文 Table 1 有 UAV 最大 CPU 5 GHz、总带宽 10 MHz 的数值先例；原场景是 10 个地面节点、每槽任务 100–500 kbit、900–1000 cycles/bit、0.5 s 时隙，目标和本项目不同。来源：[Li et al., 2021](https://link.springer.com/article/10.1186/s13638-020-01861-8)。因此是 UAV 参数先例，不是本项目完整场景或硬件定标。 | 有直接数值先例但场景不匹配。需要随本项目工作负载做敏感性分析；不应仅凭该论文声称已定标。 |

## Cloud 与 Edge 通信/计算时延的差异

Edge 上行速率按节点带宽份额和当前信道计算：

~~~text
R_v,m = b_v,m B_m log2(1 + p_v h_v,m / sigma^2)
T_v,m = D_v / R_v,m + L_v / (w_v,m F_m)
~~~

Cloud 则使用每任务固定等效蜂窝速率和扩展系数：

~~~text
T_v,c = psi_c D_v / R_c + L_v / f_v,c
E_v,c = p_v D_v / R_c
~~~

11.pdf 第5–7页说明 Edge 速率取决于地面/空地信道和共享带宽，Edge CPU 与带宽在各服务节点分别共享；第7页将 Cloud 放在独立蜂窝链路上，不占这些 Edge 带宽。第8页说明 Cloud 结果下行被忽略。Stage3B 只对 Cloud CPU 总池做共享分配；上行 200 Mbps 仍是每个 Cloud 任务各自输入到公式的速率，且 psi_c 仅扩大 Cloud 通信时延。

## 当前任务负载与资源量级

配置给出服务输入基准范围 500–5000 KB、输入扰动 ±20%、计算密度 400–1000 cycles/bit、负载扰动 ±10%。scenario_generator.sample_tasks() 使用 D = mean_input_size * input_factor、L = cycles_per_bit * D * load_factor。由参数包络推导（十进制 KB，1 KB = 8000 bit）：

| 量 | 配置/推导包络 |
|---|---:|
| 单任务输入 D | 400–6000 KB，即 3.2–48 Mbit |
| 计算密度 | 360–1100 cycles/bit |
| 单任务计算量 L | 1.152–52.8 Gcycles |
| 同一时隙任务到达 | 20 个任务车辆，Bernoulli 概率 0.7；期望 14 个，不等于每槽恒定 14 个 |

量级对照只作公式推算：若一个任务独占全部 60 GHz Cloud 池，其计算时间约为 0.019–0.88 s；实际池内有其他任务时每任务获得的 CPU 更少。旧合同 5 GHz/task 对应约 0.23–10.56 s。以 200 Mbps 和 psi_c=11 计算，Cloud 通信项范围约 0.176–2.64 s。Edge 的 5 GHz 节点独占计算时间约 0.23–10.56 s，SV 1 GHz 约 1.15–52.8 s；真实任务还会按 KKT 共享 CPU/带宽，Edge 通信时间另外依赖距离、信道和分配份额。

这些结果说明项目负载足以让所选 CPU 池和 Edge 节点发生竞争，单位换算也与代码的 bit/cycle/s 公式一致；它们不证明资源值与目标硬件匹配。与 Li 等的 100–500 kbit UAV 任务相比，本项目 3.2–48 Mbit 的配置包络大得多；同一 CPU GHz 或带宽 MHz 的数值不能脱离工作负载直接移植。项目没有用任务截止时间强制 T <= 1s，时延是效用评价量，因此出现多秒时延本身不构成资源合同或公式矛盾。

## 后续取证建议

- 60 GHz：已有四状态容量敏感性结果可作为局部证据；若要声称物理匹配，应取得 Cloud 规格/配额和并发服务数据，并在目标任务到达与负载范围下评估更广状态。
- 200 Mbps：核实目标蜂窝/Cloud 的 per-user 吞吐、并发时共享或拥塞语义；文献平均速率不能替代本项目网络测量。
- psi_c=11：用端到端网络时延证据确定是否可以用线性 D/R 扩展项表达；再做敏感性分析。当前没有数字依据。
- SV/RSU/UAV：优先补齐目标节点规格；在规格缺失时通过敏感性分析呈现模型依赖。UAV 的文献数值只作场景有限的基线参照。

本任务没有发现需要立即改动物理参数的证据。参数未标定意味着后续需补来源、场景敏感性或目标规格，不意味着为了让 Cache/UAV 收益变大而改数。

~~~ini
PARAMETER_EVIDENCE_COMPLETE = YES
CLOUD_CAPACITY_PHYSICALLY_CALIBRATED = NO
CLOUD_DELAY_FACTOR_SUPPORTED = NO
~~~

CLOUD_DELAY_FACTOR_SUPPORTED = NO 指 psi_c=11 这个数值没有直接依据；11.pdf 支持的是公式形式及 psi_c >= 1。

## 主要本地依据

- 11.pdf：第1–3页时隙和任务定义；第4–8页 CPU、带宽、Cloud/Edge 时延与能耗公式；第24–26页仿真场景说明。
- configs/proposed/stage3_cloud_finite_60ghz.toml：任务范围及 Stage3B Cloud/Edge 数值。
- src/common/system_model.py::edge_execution()、cloud_execution()；src/decision/task_allocation.py::_allocate_cloud_cpu_kkt()；src/scenario/scenario_generator.py::sample_tasks()。
- reports/cloud_edge_system_model_analysis.md、reports/cloud_capacity_sensitivity.md、reports/cloud_edge_model_stage2b_design.md：项目已完成的 Cloud 数值来源审阅和有限池固定状态敏感性结果。前一份报告特别记录：所引用 Cloud 文献的 CPU 合同不是本项目 legacy 5 GHz/task。

