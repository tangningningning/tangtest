# L025 — Collaborative Service Caching, Task Offloading, and Resource Allocation in Caching-Assisted Mobile Edge Computing

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2025
- 刊物/会议：IEEE Transactions on Services Computing
- DOI：10.1109/TSC.2025.3586093
- PDF页数：16
- 原文：`C:\Users\tangning\Desktop\文献\.analysis_work\proposal_revision_sources_tmp\Collaborative Service Caching, Task Offloading, and Resource Allocation in Caching-Assisted Mobile Edge Computing.pdf`
- 领域分类：**C**；依据：摘要/模型段 MEC=23, vehicle=2, UAV=0
- 主研究方向：P1；次方向：无；依据：scores={'1': 73, '2': 2, '3': 15}; title=Collaborative Service Caching, Task Offloading, and Resource Allocation in Cachi

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model, while SectionIV defines
the problem formulation. Section V introduces a multi-resource
allocation algorithm based on Lagrangian dual theory. Sec-
tion VI presents and demonstrates the A2C-based caching and
task offloading model. Section VII provides the performance
evaluation results of the proposed methods, and Section VIII
summarizes the main conclusions and discusses future research
directions.
II. R ELA TEDWORKS
Compared to conventional MEC, cache-assisted task offload-
ing offers significant potential for optimizing energy consump-
tion, response time, and other performance metrics, making
it a key research area in mobile edge computing [24], [25].
Additionally, user fairness, as a key performance metric for
assessing the satisfaction level of users, should not be neglected
during task offloading in MEC systems. In this section, we
review the most relevant studies, ana

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model, while SectionIV defines
the problem formulation. Section V introduces a multi-resource
allocation algorithm based on Lagrangian dual theory. Sec-
tion VI presents and demonstrates the A2C-based caching and
task offloading model. Section VII provides the performance
evaluation results of the proposed methods, and Section VIII
summarizes the main conclusions and discusses future research
directions.
II. R ELA TEDWORKS
Compared to conventional MEC, cache-assisted task offload-
ing offers significant potential for optimizing energy consump-
tion, response time, and other performance metrics, making
it a key research area in mobile edge computing [24], [25].
Additionally, user fairness, as a key performance metric for
assessing the satisfaction level of users, should not be neglected
during task offloading in MEC systems. In this section, we
review the most relevant studies, analyze their contributions,
and compare them with our work presented in this paper.
A. Fairness-Aware Optimization for MEC
Considering the complex computing demands of power In-
ternet of Things devices, Li et al. [5] proposed a two-layer
edge computing framework consisting of an edge server layer
and an access point layer. Specifically, a fairness mechanism
based on the Theil index was introduced to achieve balanced
allocation within the system. The Industrial Internet of Things
(IIoT) devices rely on computing resources to facilitate efficient
data exchange and task offloading. In [10], the authors proposed
a fairness-aware task offloading approach by formulating a
multi-constraint optimization problem and solving it using a
weighted max-min fairness algorithm.
Scheduling fairness can significantly impact the average QoS
of vehicles in the vehicular edge computing paradigm. To ad-
dress

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

problem formulation. Section V introduces a multi-resource
allocation algorithm based on Lagrangian dual theory. Sec-
tion VI presents and demonstrates the A2C-based caching and
task offloading model. Section VII provides the performance
evaluation results of the proposed methods, and Section VIII
summarizes the main conclusions and discusses future research
directions.
II. R ELA TEDWORKS
Compared to conventional MEC, cache-assisted task offload-
ing offers significant potential for optimizing energy consump-
tion, response time, and other performance metrics, making
it a key research area in mobile edge computing [24], [25].
Additionally, user fairness, as a key performance metric for
assessing the satisfaction level of users, should not be neglected
during task offloading in MEC systems. In this section, we
review the most relevant studies, analyze their contributions,
and compare them with our work presented in this paper.
A. Fairness-Aware Optimization for MEC
Considering the complex computing demands of power In-
ternet of Things devices, Li et al. [5] proposed a two-layer
edge computing framework consisting of an edge server layer
and an access point layer. Specifically, a fairness mechanism
based on the Theil index was introduced to achieve balanced
allocation within the system. The Industrial Internet of Things
(IIoT) devices rely on computing resources to facilitate efficient
data exchange and task offloading. In [10], the authors proposed
a fairness-aware task offloading approach by formulating a
multi-constraint optimization problem and solving it using a
weighted max-min fairness algorithm.
Scheduling fairness can significantly impact the average QoS
of vehicles in the vehicular edge computing paradigm. To ad-
dress this, a dependent task offloading scheduli

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p9起）

proposed algorithm ACSCTO.
Fig. 3. Matrix representation of caching model inputs.
Fig. 4. Critic network architecture.
a softmax transformation, ensuring that their summation equals
1. This result, referred to as Proca, denotes the probability
distribution of each service to be cached in the ES. To determine
the caching decision acat , we conduct non-repetitive sampling
based on Procat until the cumulative storage of the selected
services exceeds the maximum caching capacity of the ES.
Regarding the critic network, we have adopted a more efficient
approach to augment the learning capacity of the critic, distin-
guishing it from conventional critic networks. As illustrated in
Fig. 4, instead of exclusively generating a single state value,
the critic network produces the values for each service in the
current time slot. The output can be represented as a vector of
sizeK and the element of the vector is expressed as V (scat ,ak).
It shall be noted that, the notation ak in this context should be
clarified as representing servicek, rather than denoting an action.
With the assistance of the task offloading model, we also map
rewards to each service, referred as Rcat , with details provided
in Section VI-B2.
3) Model Learning: We employ backpropagation algorithm
and rewards Rcat derived from the task offloading model to
optimize the weights and biases of both the actor and critic
netw

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

simulation results
demonstrate the superior performance of the proposed solution
in comparison to several baseline methods.
Index Terms—Service caching, task scheduling, multi-resource
allocation, advantage actor-critic, edge computing.
I. I NTRODUCTION
M
OBILE Cloud Computing (MCC), with its abundant,
configurable, and shareable resources, was once per-
ceived as an efficient solution to the challenges posed by the
computation-intensive applications from mobile user devices
(UDs). UDs can offload the computation tasks to the cloud
center, enabling efficient data processing in MCC. However, the
primary limitation of MCC arises from the protracted latency in
Received 29 August 2024; revised 12 May 2025; accepted 12 June 2025.
Date of publication 4 July 2025; date of current version 8 August 2025.
(Corresponding author: Shuo Xiao.)
Chaogang Tang, Y ao Ding, and Shuo Xiao are with the School of Com-
puter Science and Technology, China University of Mining and Technology,
Xuzhou 221116, China (e-mail: cgtang@cumt.edu.cn; yding32@cumt.edu.cn;
sxiao@cumt.edu.cn).
Zhenzhen Huang is with the School of Computer Science and Technology,
China University of Mining and Technology, Xuzhou 221116, China, and also
with the Library, China University of Mining and Technology, Xuzhou 221116,
China (e-mail: huangzhenzhen@cumt.edu.cn).
Huaming Wu is with the Center for Applied Mathematics, Tianjin

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 是 |
| 服务更新周期 | 是 |
| 决策时间尺度自适应 | 部分 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 部分 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=是；时间尺度自适应=部分 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=部分 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：2（PDF页）
- 方法窗口：9（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：2（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（16页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
