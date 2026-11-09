# L110 — Self-Adjusting Network Slicing for Dynamic Heterogeneous Task Offloading in UAV-Enabled Mobile Edge Computing

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2026
- 刊物/会议：IEEE Transactions on Cognitive Communications and Networking
- DOI：10.1109/TCCN.2025.3565942
- PDF页数：15
- 原文：`C:\Users\tangning\Desktop\文献\03_多业务非平稳环境下的服务部署与传输资源在线协同\Self-Adjusting Network Slicing for Dynamic Heterogeneous Task Offloading in UAV-Enabled Mobile Edge Computing.pdf`
- 领域分类：**D**；依据：摘要/模型段 MEC=17, vehicle=3, UAV=1
- 主研究方向：P3；次方向：无；依据：scores={'1': 23, '2': 2, '3': 53}; title=Self-Adjusting Network Slicing for Dynamic Heterogeneous Task Offloading in UAV-

## 2. 研究场景与问题

### 摘要窗口

未定位到章节；需回原文核查。

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

未定位到章节；需回原文核查。

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p4起）

PROBLEM FORMULATION
This section details the exemplary scenario, the system
model, and the problem formulation.
A. Exemplary Scenario
In this paper, we focus on the problem of heterogeneous
dynamic computational task offloading for UA V-enabled edge
computing based on network slicing techniques. Specifically,
as shown in Fig. 1, we use a smart campus scenario for
demonstration. We consider a square campus area with side
length L, where there exists a UA V with an edge server
to provide computation offloading services for users (e.g.,
students and faculty members) carrying smart terminals on
the ground. Let m ∈M = {1,..., M} represents the set of
users.
In the smart campus scenario, there are a large number
of heterogeneous smart devices (e.g., mobile terminals of
students and employees), and these smart terminals regu-
larly generate heterogeneous computing tasks, which have
different data sizes, task complexities, and QoS require-
ments. In order to provide differentiated services to these
heterogeneous computing tasks, we utilize network slicing
techniques to slice the UA V-enabled physical network into
multiple virtual, independent logical network slices. For exam-
ple, latency-sensitive tasks with strict latency requirements
(e.g., real-time interactive games among students) can be
assigned to low-latency network slices; data-intensive tasks
with high bandwidth requirements (e.g., transmission of
experimental data sets) can be assigned to high-bandwidth
slices; and compute-intensive tasks with high computational
requirements (e.g., inference for artificial intelligence large
models as personal intelligent learning assistants) can be
assigned to high-computational resource slices, which helps
to realize efficient collaborative work among various intel-
ligent devices

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p2起）

proposed scheme in dealing with
the problem of heterogeneous dynamic task offloading
is demonstrated. Specifically, compared to the current
state-of-the-art baseline schemes, i.e., the two-timescale
reinforcement learning scheme with a fixed network
slicing time window, the proposed scheme demonstrates
significant advantages with the lowest system cost. Also,
the robustness of the proposed scheme to the effects of
various parameters.
The rest of the paper is organized as follows. Section II
reviews the related works. Section III describes the system
model and formulates the optimization problem. The proposed
SaTPPO-based two-timescale optimization scheme is detailed
in Section IV. Simulation experiments and analysis are carried
out in Section V, and finally conclusions are drawn in
Section VI.
II. R ELATED WORKS
This section reviews the related works in terms of network
slicing for task offloading and two-timescale optimization for
wireless communication networks.
A. Network Slicing for Task Offloading
Network slicing technology provides customized services
for heterogeneous computing tasks by constructing multiple
logically independent network slices on the same set of
shared physical infrastructures, which effectively meets diverse
QoS requirements. Therefore, the combination of network
slicing and mobile edge computing techniques has become a
promising research hot topic.
Ne

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p2起）

simulations in a smart campus scenario, the
effectiveness of the proposed scheme in dealing with
the problem of heterogeneous dynamic task offloading
is demonstrated. Specifically, compared to the current
state-of-the-art baseline schemes, i.e., the two-timescale
reinforcement learning scheme with a fixed network
slicing time window, the proposed scheme demonstrates
significant advantages with the lowest system cost. Also,
the robustness of the proposed scheme to the effects of
various parameters.
The rest of the paper is organized as follows. Section II
reviews the related works. Section III describes the system
model and formulates the optimization problem. The proposed
SaTPPO-based two-timescale optimization scheme is detailed
in Section IV. Simulation experiments and analysis are carried
out in Section V, and finally conclusions are drawn in
Section VI.
II. R ELATED WORKS
This section reviews the related works in terms of network
slicing for task offloading and two-timescale optimization for
wireless communication networks.
A. Network Slicing for Task Offloading
Network slicing technology provides customized services
for heterogeneous computing tasks by constructing multiple
logically independent network slices on the same set of
shared physical infrastructures, which effectively meets diverse
QoS requirements. Therefore, the combination of network
slicing and mobile edge c

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 是 |
| 服务更新周期 | 是 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 部分 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 部分 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=部分；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=部分 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：None（PDF页）
- 问题窗口：4（PDF页）
- 方法窗口：2（PDF页）
- 仿真窗口：2（PDF页）
- 结论窗口：2（PDF页）

### 关键词命中片段

- p14: “Two-timescale resource management for ultrareliable and low-latency vehicular communi- cations,” IEEE Trans. Commun. , vol. 70, no. 5, pp. 3282–3294, May 2022. [28] Y . Shi, C. Yi, R. Wang, Q. Wu, B. Chen, and J. Cai, “Service migration or task rerouting: A two-timescale online resource optimization for MEC,” IEEE Trans. Wireless Commun. , vol. 23

## 12. 阅读闭合状态

- 全文文本：已提取（15页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
