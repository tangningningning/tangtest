# L078 — Joint Optimization of Communication and Storage Latencies for Vehicular Edge Computing

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2024
- 刊物/会议：IEEE Transactions on Intelligent Transportation Systems
- DOI：10.1109/TITS.2023.3336704
- PDF页数：15
- 原文：`C:\Users\tangning\Desktop\文献\18篇指定论文合集\Joint Optimization of Communication and Storage Latencies for Vehicular Edge Computing.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=41, UAV=0
- 主研究方向：P2；次方向：无；依据：scores={'1': 2, '2': 4, '3': 1}; title=Joint Optimization of Communication and Storage Latencies for Vehicular Edge Com

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model. In Section III, the problem addressed
in this paper is formulated. Section IV elaborates the pro-
posed solution. Then, in Section V, a performance evaluation
setup is outlined and simulation results are discussed. Finally,
Section VI concludes the paper.
II. S YSTEM MODEL
In this section, we first present a general system overview.
Afterwards, we elaborate models of the communication, stor-
age, and overall latency in respective subsections.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 30,2026 at 13:50:34 UTC from IEEE Xplore. Restrictions apply. 
KISHANI et al.: JOINT OPTIMIZATION OF COMMUNICATION AND STORAGE LATENCIES FOR VEC 5437
Fig. 1. Overview of the system model with unikernel i of edge computing
server k, Uv,ki associated to vehicle v (Vv).
A. General System Overview
We assume system consisting of V vehicles co

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model. In Section III, the problem addressed
in this paper is formulated. Section IV elaborates the pro-
posed solution. Then, in Section V, a performance evaluation
setup is outlined and simulation results are discussed. Finally,
Section VI concludes the paper.
II. S YSTEM MODEL
In this section, we first present a general system overview.
Afterwards, we elaborate models of the communication, stor-
age, and overall latency in respective subsections.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 30,2026 at 13:50:34 UTC from IEEE Xplore. Restrictions apply. 
KISHANI et al.: JOINT OPTIMIZATION OF COMMUNICATION AND STORAGE LATENCIES FOR VEC 5437
Fig. 1. Overview of the system model with unikernel i of edge computing
server k, Uv,ki associated to vehicle v (Vv).
A. General System Overview
We assume system consisting of V vehicles connected to
K base stations. Each base station hosts one edge computing
server serving the vehicle(s), as shown in Fig. 1. Therefore,
there are K edge computing servers. Each vehicular service
offloaded into the edge computing server appears in a form
of unikernels [32]. The unikernel is a lightweight, standalone
machine image, used in the cloud computing environment
due to its ability to improve performance, enhance security,
simplify deployment, or large scalability. We consider a single
unikernel for each service of the vehicle. For different services,
such as for collision avoidance or map/software updates, each
vehicle can be allocated with multiple unikernels, each han-
dling a specific service. Considering multiple unikernels, each
handling a single vehicle service has many benefits including
higher security, more efficient resource allocation regarding the
vehicle demands and prio

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

optimization problem is Nondeterministic Polynomial-
time (NP)-completed under realistic assumptions. Therefore,
finding solution for such problem is complex if not impossi-
ble [11]. Note that the storage I/O caching is different from
the content caching [12], [13] and optimization of the storage
I/O cache reduces the data access latency in a VEC regardless
of the data content [4].
The main focus of works addressing the latency in VEC
platforms is minimization of communication, computation,
and network caching latencies. However, a considerable source
of latency, the storage latency, is not addressed in these works.
A group of previous works is concentrated on computation
offloading [14], [15], [16], [17], [18], [19], [20], [21], [22],
[23], but only considering the computation processing using
metrics such as Central Processing Unit (CPU) cycles, while
storage as a significant source of latency is ignored.
Some other previous works concentrate on network caching
and storage allocation in vehicular edge computing [12], [13],
[24]. However, these works do not explore the storage access
latency and do not investigate a potential reduction in the
storage access latency via I/O caching optimization.
Data storage systems mainly rely on Hard Disk Drives
(HDDs) and low-end Solid-State Drives (SSDs) as the main
storage media to manage the storage services at a reasonable
cost. However, these storage media have a relatively high
latency. To provide the desired performance with a low latency,
an efficient and widely used approach is the I/O caching.
In the data storage system with I/O cache, the average storage
latency of each application is directly affected by the size of
I/O cache allocated to that application [25], [26], [27], [28],
[29]. Hence, allocating the I/O cache reso

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p2起）

proposed algorithm significantly reduces the overall
latency of vehicular services stored in edge computing
servers with respect to state-of-the-art-works.
The rest of paper is organized as follows. Section II presents
the system model. In Section III, the problem addressed
in this paper is formulated. Section IV elaborates the pro-
posed solution. Then, in Section V, a performance evaluation
setup is outlined and simulation results are discussed. Finally,
Section VI concludes the paper.
II. S YSTEM MODEL
In this section, we first present a general system overview.
Afterwards, we elaborate models of the communication, stor-
age, and overall latency in respective subsections.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 30,2026 at 13:50:34 UTC from IEEE Xplore. Restrictions apply. 
KISHANI et al.: JOINT OPTIMIZATION OF COMMUNICATION AND STORAGE LATENCIES FOR VEC 5437
Fig. 1. Overview of the system model with unikernel i of edge computing
server k, Uv,ki associated to vehicle v (Vv).
A. General System Overview
We assume system consisting of V vehicles connected to
K base stations. Each base station hosts one edge computing
server serving the vehicle(s), as shown in Fig. 1. Therefore,
there are K edge computing servers. Each vehicular service
offloaded into the edge computing server appears in a form
of unikernels [32]. The un

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p2起）

simulations with realistic models, we demonstrate
the proposed algorithm significantly reduces the overall
latency of vehicular services stored in edge computing
servers with respect to state-of-the-art-works.
The rest of paper is organized as follows. Section II presents
the system model. In Section III, the problem addressed
in this paper is formulated. Section IV elaborates the pro-
posed solution. Then, in Section V, a performance evaluation
setup is outlined and simulation results are discussed. Finally,
Section VI concludes the paper.
II. S YSTEM MODEL
In this section, we first present a general system overview.
Afterwards, we elaborate models of the communication, stor-
age, and overall latency in respective subsections.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 30,2026 at 13:50:34 UTC from IEEE Xplore. Restrictions apply. 
KISHANI et al.: JOINT OPTIMIZATION OF COMMUNICATION AND STORAGE LATENCIES FOR VEC 5437
Fig. 1. Overview of the system model with unikernel i of edge computing
server k, Uv,ki associated to vehicle v (Vv).
A. General System Overview
We assume system consisting of V vehicles connected to
K base stations. Each base station hosts one edge computing
server serving the vehicle(s), as shown in Fig. 1. Therefore,
there are K edge computing servers. Each vehicular service
offloaded into the edge computi

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 未提及 |
| 服务更新周期 | 部分 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 未提及 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=未提及；服务更新周期=部分；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：2（PDF页）
- 方法窗口：2（PDF页）
- 仿真窗口：2（PDF页）
- 结论窗口：2（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（15页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
