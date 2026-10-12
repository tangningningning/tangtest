# Cloud 与 Edge 资源模型设计建议（阶段 2B）

日期：2026-10-10  
项目：adaptive_cache_mec

## 范围与结论摘要

本报告仅读取项目源码、正式 5e-4 运行冻结配置、项目 11.pdf、阶段 1 / 阶段 2A 已有报告，以及公开原始论文资料；未运行仿真、Matching/KKT、Validation 或训练，未读取测试集合或正在写入的诊断结果，也未修改源码和 TOML。阶段 2A 报告是本阶段使用的对照结果来源。

**唯一推荐：Cloud 采用每 Slot 有限共享 CPU 总容量，Cloud 上行继续使用独立的车辆到 Cloud 等效链路；Matching 保留 Cloud 与全部缓存且链路可行的 Edge 候选，按完整共享资源分配后的 Task Utility 选优，仅在效用数值容差内偏好 Edge。** 该方案保留项目 11.pdf 的时延/能耗形式和独立蜂窝链路假设，只把 Cloud 的 CPU 资源合同改成显式有限池；不把外部论文的 20 GHz、2–4 GHz/task 当作本项目取值，也不人为提高 Cloud 时延或修改 Reward。

容量合同的形态可以确定，但现有资料没有给出本项目可用 Cloud 的总 CPU 份额、宿主规格或并发压测结果。因此本报告不指定 Cloud 总 CPU 的数值；在取得部署规格或完成有明确 workload 的标定前，尚不能把它作为正式训练模型的已标定参数。

## 1. 源码实际实现与两类 Cloud 合同

### 1.1 当前固定每任务合同（方案 A）

在正式冻结文件 **runs/formal_cache_economic_v2_lr_5e_minus_4/run_config.toml** 中，Cloud 的 **cpu_cycles_per_s = 5.0e9**、**uplink_rate_bps = 200.0e6**、**resource_sharing = false**、**cloud_delay_factor = 11.0**。默认 **configs/proposed/system_config.toml** 具有相同取值；本报告以冻结运行配置为准，而非仅从默认配置推断。

源码合同比字段名更重要：

- **src/common/system_model.py::cloud_execution()** 将传入的 CPU 与速率原样作为该任务分配的 CPU 和速率；计算为 \(L_v/F_c\)，通信时延为 \(\psi_c D_v/R_c\)，车辆传输能耗为 \(p_vD_v/R_c\)。
- **src/decision/task_allocation.py::allocate_resources_kkt()** 将 Cloud 排除在 Edge KKT 资源分配之外；**evaluate_matching_with_kkt()** 对每个 Cloud 任务分别传入同一组完整 Cloud 速率与 CPU 值。
- **src/decision/cache_configuration.py::CacheOptimizationConfig.validate()** 和 **src/common/project_config.py** 明确拒绝 Cloud sharing=true。因此当前行为不是“同一 Cloud 池内均分 5 GHz / 200 Mbps”。
- TOML 中名为 **total_cpu_cycles_per_s**、**total_bandwidth_bps** 的字段目前只是与每任务字段相等的兼容值；它们没有作为同槽所有 Cloud 任务的总容量约束。Cloud 任务数增加时，源码不会相应降低任一 Cloud 任务的 CPU 或上行速率。

因此，方案 A 的 5 GHz 是每个被选中 Cloud 任务的等效服务能力，不是 Cloud 单实例总能力。若同槽有 \(n_c\) 个 Cloud 任务，合同隐含提供 \(5n_c\) GHz 的并发算力；以 20 个任务全部选 Cloud 为例，隐含并发服务量为 100 GHz。200 Mbps 同样按任务独立使用，而非总链路限额。该设定可作为“Cloud 资源充足、并发任务各有固定服务份额”的抽象，但目前没有本项目的部署证据证明 Cloud 会为每个车辆任务保证该独立份额。

### 1.2 有限总容量合同（方案 B）

定义一个逻辑 Cloud CPU 池 \(F_C^{\mathrm{tot}}\)，单位 cycle/s。对该 Slot 被分配至 Cloud 的任务给出 \(f_{v,C}>0\)，并满足

\[
\sum_{v\in\mathcal V_C(t)} f_{v,C}(t)\le F_C^{\mathrm{tot}}.
\]

任务计算时延改为 \(L_v/f_{v,C}\)。因此任务计算性能随当前 Cloud 任务集合和资源优化结果变化，5 GHz 不再被解释为每个任务可同时独占的保证值。Cloud 资源使用共享池并不等于必须增加队列或截止时间：在项目现有的独立 Slot 任务模型中，第一步可以只建模同 Slot 的 CPU 分配；跨 Slot 请求排队属于另一个未建模的服务队列问题。

通信建议先沿用独立的车辆至 Cloud 等效链路 \(R_{v,C}\)，而不擅自添加一个 Cloud 总带宽池。项目 11.pdf 第 7 页明确将 Cloud 视为独立蜂窝链路，不放入服务节点共享带宽分配；外部车联网论文也以每车到 Cloud 的上/下行速率表示链路，同时另设 Cloud CPU 总容量。只有目标部署证实蜂窝小区、Cloud 接入网或回传确有共享瓶颈时，才应增加单独的总链路容量、按用户分配率及相应耦合。当前资料没有给出该瓶颈的容量数值。

## 2. 工作负载、并发与资源容量语义

正式冻结配置及其任务生成器给出的合同如下。任务输入和负载扰动由 **src/scenario/scenario_generator.py::sample_tasks()** 按服务基准值采样；并发数是静态设计范围，不是本阶段新跑出的仿真统计。

| 项目 | 冻结设置 / 推导 | 含义与边界 |
|---|---|---|
| 任务车辆与到达 | 20 个 TV；每个 TV 每 Slot 按 Bernoulli(\(0.7\)) 独立产生任务 | Slot 到达数支持集 0–20，期望 \(20\times0.7=14\)，标准差约 2.05。任务跨 Slot 独立，不保留任务队列。 |
| 输入数据 | 服务基准 500–5000 KB，输入扰动 ±20% | 生成器的名义 envelope 为 400–6000 KB；精确 bit 数受 KB 转换约定影响，本报告不将其伪装为实测流量。 |
| 计算密度 | 400–1000 cycles/bit，负载扰动 ±10% | 生成关系是 \(L=D\times\text{cycles/bit}\times\text{load factor}\)，负载因子 0.9–1.1；有效 envelope 为 360–1100 cycles/bit。按十进制 KB 仅作量级换算，单任务约 1.15–52.8 Gcycles。 |
| 5e-4 冻结资源 | SV：每节点 1 GHz / 10 MHz，共 5 个；RSU：每节点 5 GHz / 10 MHz，共 2 个；UAV：每架 5 GHz / 10 MHz，共 2 个 | 全部 Edge 名义 CPU 总和为 25 GHz，带宽总和为 90 MHz，但它们分散于 9 个节点，不能跨节点互换；服务缓存、车辆位置、链路以及各节点 KKT 份额决定单任务可访问的实际资源。 |
| 当前 Cloud 语义 | 每个 Cloud 任务分别获得完整的 5 GHz 与 200 Mbps；不按 Cloud 任务数共享 | 期望约 14 个到达任务意味着方案 A 的“全任务可同时获得”语义约为 70 GHz；上限 20 个到达时为 100 GHz。这里只是配置合同隐含的并发能力，不是已知 Cloud 硬件实测。 |

阶段 2A 的四个配对状态共有 66 个任务，Resource-aware 模式选择了 58 个 Cloud 任务（任务数加权比例 87.88%），即这四个 Slot 平均 14.5 个 Cloud 任务。按当前固定合同，这相当于平均每 Slot 隐含 72.5 GHz 的 Cloud 任务并发计算服务量。该小样本显示固定每任务合同具有很强替代性，但不能推导长期策略下的 Cloud 比例或有限总量容量。

容量语义决定了比较是否公平：当前 Edge 资源是“每节点有限总量、节点内共享”，而 Cloud CPU 是“每任务固定、对并发任务数无上限”。如果研究目标是对比任意充分供给的远端 Cloud，这可以是明确的系统假设；若研究目标是优化 Cache 与 UAV 以疏导有限边缘资源，则 Cloud 与 Edge 的资源合同不对称，而且 Cloud 算力无负载竞争，会改变 Cache/UAV 的边际作用。有限 Cloud CPU 总池能明确表示该竞争，同时不强行保证 Edge 永远胜过 Cloud。

## 3. Edge 资源与时延竞争

| 节点 | 5e-4 冻结资源 | 资源约束 / 时延来源 | 对 Cloud 比较的含义 |
|---|---|---|---|
| SV | 每节点 1 GHz、10 MHz | 任务命中该 SV 缓存且路径可用后进入节点级共享 CPU、带宽 KKT；使用地面距离损耗与 Shannon 上行率。 | CPU 低于 Cloud 当前 5 GHz/task，服务可用不代表计算性能更好。 |
| RSU | 每节点 5 GHz、10 MHz | 每个 RSU 独立的共享 CPU、带宽池；链路按地面信道和位置计算，传输时延为 \(D/R\)，没有单独回传项。 | 单任务分得的 CPU、带宽通常小于节点总量；多任务负载造成时延上升。 |
| UAV | 每架 5 GHz、10 MHz | 每架 UAV 各自运行共享 CPU 与带宽 KKT；空地平均 LoS/NLoS 增益进入 Shannon 速率。 | 水平位置改变距离、仰角和信道增益，进而改变可获得的通信时延；不改变 5 GHz CPU 总量。 |

对于每个 Edge 节点 \(m\)，CPU 分配比例 \(w_{v,m}\) 与带宽比例 \(b_{v,m}\) 分别受 \(\sum_v w_{v,m}\le1\) 和 \(\sum_v b_{v,m}\le1\) 约束。源码以当前分到该节点的完整任务集合分别求 CPU 与带宽 KKT；通信速率为 \(b_{v,m} B_m \log_2(1+p_vh_{v,m}/\sigma^2)\)。任务时延为

\[
T_{v,m}^{\mathrm{Edge}}
=\underbrace{\frac{D_v}{b_{v,m}B_m\log_2(1+p_vh_{v,m}/\sigma^2)}}_{\text{无线上传}}
+\underbrace{\frac{L_v}{w_{v,m}F_m}}_{\text{执行}}.
\]

每节点的 CPU 和带宽总量有限；任务增加时，KKT 总和约束迫使份额在更多任务间分配，典型情况下单任务可用份额下降，通信速率和 CPU 服务速率都可能下降，两个时延分项都可能上升。任务负载、链路信道及优化权重不同，资源分配不一定均分，但不能把每任务按独占 5 GHz / 10 MHz 算出的理想时延当成共享资源后的实际时延。UAV MOVE 只能通过位置/信道改变通信项（并伴随现有移动成本），不能扩张该 UAV 的 CPU 总量。

UAV 的 5 GHz 是每架 UAV 的共享总 CPU，而 Cloud 当前 5 GHz 是每个 Cloud 任务都可同时取得的等效 CPU。对于 UAV，\(w_{v,u}\le1\)，故 \(w_{v,u}F_u\le5\) GHz；与 Cloud 的 5 GHz/task 相比，Edge 执行项 \(L/(wF)\) 不可能因 CPU 更大而更快，多个任务共用 UAV 时通常严格更慢。Edge 仍可能靠更好的上传信道、不同缓存/路径和实际份额获得较低总时延；这取决于任务数据量与计算量及 KKT 份额，不构成“Edge 对每个任务必胜”的依据。SV/RSU/UAV 数值属于本项目冻结的实验设计；除了项目此前整理的 UAV 5 GHz / 10 MHz 文献基线，不能把所有节点数值一概称为实测或直接校准。

## 4. Cloud / Edge 时延公式与 Cloud 胜出的条件

项目 **11.pdf** 第 4–8 页规定：Edge 任务时延是 \(D/R + L/(wF)\)；Cloud 任务时延是 \(\psi_cD/R_c + L/F_c\)。云结果返回时延忽略，依据是输出相对于输入较小；Cloud 车辆能耗只计算蜂窝无线上传能耗，不乘 \(\psi_c\)。**src/common/system_model.py::edge_execution() / cloud_execution()** 与这些公式一致，速率单位 bit/s、CPU 单位 cycle/s、输入 bit、计算量 cycle、时延秒、车辆能耗焦耳。两者的通信/计算分项在代码输出结构中分开。

在当前固定合同下：

\[
T_v^{Cloud}=\underbrace{\psi_c\frac{D_v}{R_c}}_{\text{Cloud 上传的等效扩展时延}}
+\underbrace{\frac{L_v}{F_c}}_{\text{Cloud 执行时延}},
\qquad
E_v^{Cloud}=p_v\frac{D_v}{R_c}.
\]

\(\psi_c=11, R_c=200\) Mbps 意味着 Cloud 时延中的数据量系数为 \(\psi_c/R_c=55\) ns/bit，等效于“仅对该时延项”采用 18.18 Mbps；不能据此声称物理链路真实速率就是 18.18 Mbps。代码没有独立增加固定 RTT、单列回传/核心网 hop 或 Cloud 结果下载项；这些是 11.pdf 的建模抽象和“小结果”假设。该式没有明显的代码/论文公式不一致，但 \(\psi_c=11\) 的物理映射和目标部署依据未获确认。由于当前 \(\psi_cD/R_c\) 随 \(D\) 线性缩放，也不能将它解释为已经测得的固定传播 RTT。

推荐合同将计算项替换为实际分配 CPU：

\[
T_v^{Cloud}=\underbrace{\psi_c\frac{D_v}{R_{v,C}}}_{\text{独立等效蜂窝上传及远程扩展项}}
+\underbrace{\frac{L_v}{f_{v,C}}}_{\text{有限共享 Cloud CPU 执行}},
\quad
\sum_{v\in\mathcal V_C} f_{v,C}\le F_C^{\mathrm{tot}}.
\]

Cloud 比 Edge 快的精确条件为

\[
D_v\left(\frac{\psi_c}{R_{v,C}}-\frac{1}{R_{v,m}}\right)
+L_v\left(\frac{1}{f_{v,C}}-\frac{1}{w_{v,m}F_m}\right)<0,
\]

其中 \(R_{v,m}\) 是当前实际 Edge KKT 带宽份额与该任务当前位置/信道算出的速率。Edge 较快则该式大于 0；相等时才属于效用平局（在能耗相同或其差异也已经纳入 Task Utility 的前提下）。对应的 Edge 时延优势不能由“有缓存”直接推出。

以当前 \(\psi_c/R_c\) 为基线，Edge 通信项只有在实际 \(R_{v,m}>R_c/\psi_c\approx18.18\) Mbps 时才比 Cloud 通信项短。即使达到这个通信条件，Edge 计算项仍要看其实际 \(wF\) 与 Cloud 实际分配 \(f_{v,C}\)；增大 Cloud 并发后 \(f_{v,C}\) 下降，Cloud 计算优势会缩小，甚至反转。反之，低 Edge 份额、差 UAV 链路、较重计算任务或较少 Cloud 并发都可能使 Cloud 胜出。该不等式与速率阈值是从模型参数导出的静态判别，不是仿真结果。

## 5. 文献证据与适用边界

| 来源 | 可核实的依据 | 对本项目的用途与限制 |
|---|---|---|
| 项目 11.pdf，第 4–8 页 | Cloud \(T=\psi_cD/R_c+L/F_c\)、能耗 \(pD/R_c\)；Cloud 使用独立蜂窝链路，不参与服务节点共享带宽分配；Cloud 存全部服务；SV/RSU/UAV CPU 与带宽按节点共享。 | 是项目公式和候选系统语义的直接依据；没有为 Cloud 指定总 CPU 容量或本项目的 5 GHz/task、200 Mbps、\(\psi_c=11\) 数值。 |
| Al-Shareeda et al., “Novel DT-assisted Vehicular Task Offloading for Cloud, Edge, and Hybrid Deployments,” IEEE TVT 75(4), 2026, DOI 10.1109/TVT.2025.3620810，Table IV / 仿真设置 | 可访问的 IEEE 接受稿按车辆分别给 Cloud 上/下行速率；Cloud CPU 是总容量 \(F_C\)，每任务得到 \(F^C_{V_i}\)，并约束 \(\sum_i X_iF^C_{V_i}\le F_C\)。其仿真参数包含 20 GHz 总 Cloud CPU、每车 \(U[2,4]\) GHz、Cloud 上传 200 Mbps / 下载 400 Mbps。 | 支持“总 CPU 池 + 按任务分配”和“独立用户链路速率”的合同结构。论文是 10–100 车辆、31 KB 基准任务、5 Kcycle/byte 的另一部署和目标；20 GHz 与 2–4 GHz/task 只是该文仿真设置，不能直接作为本项目的最终值。该文 Cloud 时延还显式包括下载/传播，和 11.pdf 的简化式并不相同。链接：[IEEE DOI](https://doi.org/10.1109/TVT.2025.3620810)，[作者接受稿所在机构仓储](https://napier-repository.worktribe.com/output/4744571)。接受稿提示可能与最终排版版本有差别，引用参数时应以正式出版版复核。 |
| Li et al., “Collaborative offloading for UAV-enabled time-sensitive MEC networks,” EURASIP J Wireless Commun Netw, 2021, DOI 10.1186/s13638-020-01861-8 | 系统模型显式约束共享 UAV 计算总量与共享带宽；Table 1 的 5 GHz 计算能力、10 MHz 带宽是该文特定 UAV 仿真参数。 | 支持有限 UAV 资源竞争是常见建模选择；不能据此推断 SV、RSU 或 Cloud 的目标硬件值。链接：[Springer 原文](https://link.springer.com/article/10.1186/s13638-020-01861-8)。 |

当前 Cloud 5 GHz/task、200 Mbps/task 与 \(\psi_c=11\) 的代码/配置可直接确认；它们不是上述文献对本项目目标 Cloud 部署的直接测量。Stage 1 已明确说明数值依据不足，本报告不因外部表格存在相似数字就改写其证据状态。

## 6. Edge 优先、资源感知 Cloud 后备的 Matching 规则

建议将“Edge 优先”定义为**效用优先、Edge 平局优先**，而不是“只要 Edge 缓存可用就强制 Edge”：

1. **缓存可用**：任务所需服务在本地或 Edge 节点缓存中，才建立相应服务候选；Cloud 保存所有服务，不受此约束。Cache hit 说明 Edge 服务可用，不代表任务最终必然在 Edge 执行。
2. **物理链路可用**：Edge 候选还要通过当前车辆/节点位置、正 CPU/带宽、正信道速率的路径校验。源码 **build_feasible_targets()** 做候选筛选；此处没有先运行完整 KKT，也没有按理想独占时延决定 Edge 胜负。
3. **实际 KKT**：在候选 Matching 上按节点完整任务集合求 CPU、带宽 KKT，并重算该集合所有任务的分配、传输率、分项时延、能耗与 Task Utility。资源可行性和效用判断必须基于这个结果。
4. **方案比较**：在现有 Transfer-Exchange 搜索框架中，比较可行操作造成的完整 Task Utility 总变化；不增加 Edge reward、Cloud penalty、deadline 或惩罚项。Cloud 一直是合法备选，Edge 效用严格较高时自然选 Edge，Cloud 效用严格较高时可回退 Cloud。
5. **同效用平局**：只在既有 objective tolerance 内优先 Cloud 数较少的方案，即 Edge tie-break。它不能覆盖超过容差的 Cloud 效用增益。严格 Edge 优先则会在 Edge 已缓存但 KKT 份额极小、延迟很差时仍强制执行，和“性能明显不利时 Cloud 后备”冲突。

阶段 2A 的**RESOURCE_AWARE_FALLBACK** 已实现候选层面的 Cloud 常驻加合法 Edge 候选，当前正式流程默认仍为 **LEGACY_EDGE_GATE**。阶段 2A 的四状态受控配对中开放 Cloud 候选后 Task Utility 总和增加 6.27452182、66 个任务加权平均时延下降 3.33044291 秒；Cloud 任务增加 26，Edge 任务减少 26。该结果证实在**固定 5 GHz/task Cloud 合同**下，旧候选门控会排除有利的 Cloud 方案；它不证明有限 Cloud CPU 合同下仍有相同效应，也不证明该模式已适合作为正式默认值。

## 7. 对 Cache UPDATE、UAV MOVE 与已有反事实结果的解释

阶段 1 引用的 **reports/cache_keep_update_counterfactual.md** 显示，在所测四状态中强制 UPDATE 提高缓存命中，却降低 Return、增加平均时延及车辆能耗。可以支持的机制解释与待验证假设如下：

- Cache hit 增多扩大了 Edge 候选，但旧门控会在这些 Edge 存在时删除 Cloud 候选；因此任务可能从原本较快的 Cloud 被推到共享 Edge。阶段 2A 说明在相同固定 Cloud 合同下，打开 Cloud 候选确实能使一部分已命中服务的任务仍在 Cloud 执行；这能解释门控可能参与 UPDATE 后延迟恶化，但不能反推 UPDATE 结果的主因。
- 现有 Edge CPU 与带宽确实共享，更多任务落到同一 UAV/RSU/SV 时会改变 KKT 份额，使计算或通信时延上升。尚无证据量化各节点竞争对 UPDATE Return 差值的贡献。
- 当前每任务独享 Cloud CPU 可能令 Cloud 在高并发下依然保持计算优势；这是有源码支持的资源合同不对称，但不是对真实 Cloud 物理容量的测量。有限总 Cloud CPU 下，任务转出 Cloud 可能释放池内算力给其他 Cloud 任务，因此 UPDATE 的效用变化可能变大也可能变小，依然取决于 Edge KKT 和任务组成。
- UAV MOVE 通过信道/速率影响真正分给 UAV 的任务通信时延，但不增加 CPU；当任务可更优地留在 Cloud 时，MOVE 的通信收益可能变少。当有限 Cloud CPU 紧张、任务实际转到 UAV 时，改善 UAV 链路的边际价值可能增加。两种方向都需要同状态、不同合同的配对验证，不能保证 MOVE/UPDATE 比例或 Return 一定上升。
- Matching 使用 Transfer-Exchange 局部搜索；它是否遗漏有限共享 Cloud CPU 下的组合改进，不能仅由阶段 2A 少量增量/full 检查排除。

因此，能确认的有“旧门控排除了 Cloud”及“Edge 资源共享”；尚未确认四个 UPDATE 反事实的主要根因是 Cloud 无限资源语义、UAV CPU 竞争、带宽竞争、候选门控或局部搜索中的哪一个。

## 8. 唯一推荐方案及后续实现范围

### 推荐的资源合同与卸载规则

1. **Cloud CPU：有限总池。** 显式配置 \(F_C^{\mathrm{tot}}\) cycle/s，并按同 Slot Cloud 任务集合分配 \(f_{v,C}\)，\(\sum f_{v,C}\le F_C^{\mathrm{tot}}\)。不得把 20 GHz 或 2–4 GHz/task 直接作为最终值。
2. **Cloud 通信：独立等效用户链路。** 继续为每个任务提供 \(R_{v,C}\) 的等效车辆至 Cloud uplink，不进入 Edge 带宽 KKT；如以后有共同小区/回传资源容量证据，再单独建共享链路池。配置中的 200 Mbps 可作为复现实验的冻结基线，但只能标为外部场景参考/暂定值，不能写成目标部署已标定值。
3. **Cloud 时延：保持项目公式形式。** 使用 \(T^{Cloud}=\psi_cD/R_{v,C}+L/f_{v,C}\)，车辆上传能耗仍为 \(pD/R_{v,C}\)。先保留 11.pdf 的传输扩展项，给 \(\psi_c\) 明确标为未标定假设；用目标蜂窝/回传路径的测量决定是否保留该线性折算，或在未来有测量证据时拆成访问链路、固定 RTT/传播、回传速率和结果下载。现在没有依据给这些项填入新数字。
4. **Matching：资源感知候选 + 实际效用优先。** 所有任务保留 Cloud 和全部缓存/链路可行 Edge 候选；候选的执行决定采用资源共享后的实际 KKT Task Utility。效用在既有容差内相同时偏好 Edge；禁止严格 Edge 门控和人为 Cloud/Edge reward 偏置。

“典型负载下 Edge 总体较低时延”应作为校准后配对实验的检验目标，而不是候选门控或 Reward 中写入的保证。若使用有依据的 \(F_C^{\mathrm{tot}}\)、独立链路速率及 \(\psi_c\) 后仍观察不到 Edge 优势，应回看硬件/路径和 workload 假设，不应为满足预期而随意压低 Cloud 参数。

### 参数证据与标定方法

| 参数 | 当前依据状态 | 建议的取值/标定方法 |
|---|---|---|
| \(F_C^{\mathrm{tot}}\) | **没有本项目直接数值证据。** 外部论文仅证明有限池合同有先例。 | 以目标 Cloud 实例/容器的 CPU 配额和可用线程为起点；用项目定义的代表性 \(L\)、明确并发集合及同一运行窗口测量有效 cycle/s 或任务服务率，区分单租户配额与宿主总资源。由测量给出基准容量，再做标明上下界来源的敏感性，不把其他论文容量移植过来。 |
| \(R_{v,C}\)（当前 200 Mbps） | 项目 11.pdf 定义独立固定等效速率；外部 Table IV 有每车 Cloud uplink 200 Mbps 的数值先例，但不是本项目目标路径测量。 | 在目标蜂窝接入条件下采集按车辆/负载的有效上行速率或有来源的网络追踪；确认它是每用户有效速率而非小区总容量。 |
| \(\psi_c\)（当前 11） | 11.pdf 支持该扩展因子的公式作用；没有找到本项目 \(\psi_c=11\) 的实测 RTT/回传标定。 | 用端到端 Cloud 完成时间减去本地测得的上传、执行部分拟合剩余网络时延。检查剩余项对 \(D\) 是近似线性还是主要为固定传播/RTT；若不符合线性，再提出有测量依据的分解，不提前猜值。 |
| Cloud 共享链路容量 | 未知；参考模型使用独立蜂窝等效速率。 | 先不设共享容量。若目标蜂窝小区或回传 NIC 的吞吐随并发明显饱和，测出聚合可用容量，再将其独立建模，避免与每用户接入速率、\(\psi_c\) 重复计算。 |
| Edge 节点容量 | 冻结配置有明确实验基线；不是本项目部署的硬件测量。UAV 共享建模和 5 GHz / 10 MHz 有 UAV 文献先例。 | 本阶段保留现值；若正式研究声明真实硬件复现，应另行补足 SV/RSU/UAV 的来源或实测规格。 |

### 最小源码修改边界

**系统模型 / 配置：**

- **src/common/system_model.py::cloud_execution()**：让计算时延接收实际分配的 \(f_{v,C}\)，保留当前通信和能耗公式。
- **src/decision/cache_configuration.py::CacheOptimizationConfig**、**validate()**、**cache_config_from_project()**，以及 **src/common/project_config.py** 的键注册和验证：定义明确的有限 Cloud CPU 总量字段/合同版本；不要简单把原 **resource_sharing** 翻转为 true，因为现有校验禁止该语义，且 **total_cpu/total_bandwidth** 重复字段当前混用每任务与总量含义。

**Matching / KKT：**

- **src/decision/task_allocation.py::allocate_resources_kkt()**：为 Cloud 任务池加入 CPU KKT 分配。第一阶段不加 Cloud 带宽 KKT，因为推荐通信合同是每任务独立等效链路。
- **evaluate_matching_with_kkt()** 与必要的 Cloud 节点分组/缓存：每个 Cloud 任务使用真实分配 \(f_{v,C}\) 重算执行时延和 Task Utility。
- **_candidate_impact_scope()** 与 **evaluate_matching_candidate_incremental()**：当前作用域把 Cloud 视为“每任务固定资源”、跳过 Cloud 关联任务。共享 Cloud CPU 后，任何 Cloud 进出会改变同池其他任务分配，故受影响范围必须包括当前与候选 matching 中的**全部 Cloud 任务**，清除并重算该池的资源分配、时延和效用。若只重算被移动任务，增量结果将不再与完整 KKT 一致。
- **enumerate_best_operation()** 的增量筛选及 **NodeEvaluationCache / KKTResourceAllocationCache**：保持安全的候选上界；实际 Cloud 节点评价缓存键纳入 Cloud 池完整任务集合、任务工作量、总 CPU 容量和效用权重。当前 **_ideal_task_utility() / _strict_ideal_resource_upper_delta()** 的 Cloud 分支按固定 per-task 合同算“精确”单任务值；有限池下应改成仅用于剪枝的可采纳上界（可把全部 \(F_C^{\mathrm{tot}}\) 放宽给每个候选任务单独使用，因此上界可能偏松），并在候选精确评分时始终使用实际整池 KKT。若剪枝值没有保持上界性质，增量搜索可能错误丢掉改善方案。
- **build_feasible_targets()**：保留阶段 2A 的 Resource-aware Cloud 候选模式和正式旧模式默认状态，等确认后才决定正式默认。候选仍由缓存与物理链路决定，资源性能由 KKT/Utility 决定。

该范围改变 Cloud 总 CPU 资源语义、候选匹配评估和训练环境下的 Task Utility 分布。即使保留同一 Reward 公式和 Edge/Cloud 通信值，新的有限 Cloud pool 也会改变 Matching 结果，并进而改变正式策略训练数据；旧 checkpoint、旧正式训练和 Validation 不能视为同一环境结果。方案暂不建议改变 Cloud uplink、\(\psi_c\)、Edge 物理参数、Reward 或截止时间。

### 进入正式实现前仍需验证

- 目标部署或有据的实验设计能给出 \(F_C^{\mathrm{tot}}\)，以及 200 Mbps / \(\psi_c=11\) 是否作为暂定基线保留。
- 对同一固定 Cache/UAV/任务状态，比较方案 A 与方案 B 在多种 Cloud 并发下的完整 matching 和 KKT；确认增量结果与完整重算在 Cloud CPU 池发生变化时一致。
- 分别记录 Cloud pool CPU、Edge 各节点 CPU/带宽份额、通信/计算时延、车辆能耗及实际执行位置；区分 Cache hit 与执行节点。
- 在资源合同冻结后，再以配对实验检查 UPDATE/MOVE 的任务效用、配置成本、Return 和执行位置变化。阶段 2A 结果不能代替这一步，也不应在本报告阶段重跑。

## 9. 阶段 2B 建议状态

    RECOMMENDED_CLOUD_RESOURCE_CONTRACT = 有限的同 Slot Cloud 总 CPU 池；按任务分配并满足 sum(f_v,C) <= F_C_total；Cloud uplink 维持独立等效用户链路。
    RECOMMENDED_CLOUD_DELAY_MODEL = psi_c * D_v / R_v,C + L_v / f_v,C；车辆上传能耗 p_v * D_v / R_v,C；先保留 11.pdf 公式形态，psi_c 数值仍待标定。
    RECOMMENDED_EDGE_PRIORITY_RULE = 保留所有缓存及链路可行 Edge 候选；按完整共享资源 KKT 后的 Task Utility 选优，只在既有容差内平局优先 Edge。
    RECOMMENDED_CLOUD_FALLBACK_RULE = Cloud 始终在候选中；没有可用 Edge，Edge KKT 不可行，或 Cloud 的整池 KKT 总效用更高时由 Matching 选择 Cloud；不加 Cloud 惩罚或人为 Edge 奖励。

    PARAMETERS_WITH_DIRECT_EVIDENCE = [11.pdf 的 Cloud/Edge 公式与单位；冻结配置的任务/Edge 设计值；文献中有限 Cloud CPU 池、每任务分配、每用户 uplink 速率的模型结构；UAV 节点共享资源的结构]
    PARAMETERS_REQUIRING_CALIBRATION = [F_C_total；目标部署的 R_v,C；psi_c=11 的物理映射；是否存在共享小区/Cloud 回传容量；未有目标硬件依据的 SV/RSU/UAV 数值]

    MATCHING_KKT_CHANGE_SCOPE = [增加 Cloud CPU pool KKT；完整 matching 评价使用实际 Cloud CPU 分配；Cloud 任务进出时增量作用域覆盖旧/新 Cloud 池全部任务；安全上界与缓存键包含 Cloud 池状态]
    SYSTEM_MODEL_CHANGE_SCOPE = [cloud_execution 接收 f_v,C；显式版本化 Cloud CPU 总量配置和校验；保留当前 Cloud 独立 uplink 与 11.pdf delay/energy 形式]

    READY_FOR_IMPLEMENTATION = NO_FOR_FORMAL_MODEL_UNTIL_F_C_TOTAL_IS_CALIBRATED  # 合同形态和最小代码范围已确定；没有本项目 Cloud 总 CPU 数值证据，不能无依据完成正式资源模型。
