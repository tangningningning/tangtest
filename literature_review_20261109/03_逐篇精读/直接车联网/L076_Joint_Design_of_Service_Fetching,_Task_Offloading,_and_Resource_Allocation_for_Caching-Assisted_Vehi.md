# L076 — Joint Design of Service Fetching, Task Offloading, and Resource Allocation for Caching-Assisted Vehicular Edge Computing Networks

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2026
- 刊物/会议：IEEE Transactions on Intelligent Transportation Systems
- DOI：10.1109/TITS.2025.3639690
- PDF页数：14
- 原文：`C:\Users\tangning\Desktop\文献\车联网频繁切换-负载均衡-端到端时延研究\四篇重点论文\Joint_Design_of_Service_Fetching_Task_Offloading_and_Resource_Allocation_for_Caching-Assisted_Vehicular_Edge_Computing_Networks.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=65, UAV=0
- 主研究方向：P1；次方向：2；依据：scores={'1': 14, '2': 12, '3': 1}; title=Joint Design of Service Fetching, Task Offloading, and Resource Allocation for C

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model and formulates
the joint optimization of vehicular service fetching, task
oﬄoading, and resource allocation. Section III presents the
proposed solution in detail. The simulations and evaluations
are described in Section IV to demonstrate the superiority of
the proposed scheme. Finally, Section V concludes the paper.
II. S YSTEM MODEL AND PROBLEM FORMULATION
In this section, we first introduce the social-mobility-aware
network topology in caching-assisted VEC networks. Then,
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 30,2026 at 06:58:48 UTC from IEEE Xplore. Restrictions apply. 
TANG et al.: JOINT DESIGN OF SERVICE FETCHING, TASK OFFLOADING, AND RESOURCE ALLOCATION 3297
TABLE I
DIFFERENCE BETWEEN OUR SCHEME AND OTHER MAIN RELATED WORKS
TABLE II
PARAMETER NOTATIONS
we model the task o ﬄoading and vehicular distributed se

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model and formulates
the joint optimization of vehicular service fetching, task
oﬄoading, and resource allocation. Section III presents the
proposed solution in detail. The simulations and evaluations
are described in Section IV to demonstrate the superiority of
the proposed scheme. Finally, Section V concludes the paper.
II. S YSTEM MODEL AND PROBLEM FORMULATION
In this section, we first introduce the social-mobility-aware
network topology in caching-assisted VEC networks. Then,
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 30,2026 at 06:58:48 UTC from IEEE Xplore. Restrictions apply. 
TANG et al.: JOINT DESIGN OF SERVICE FETCHING, TASK OFFLOADING, AND RESOURCE ALLOCATION 3297
TABLE I
DIFFERENCE BETWEEN OUR SCHEME AND OTHER MAIN RELATED WORKS
TABLE II
PARAMETER NOTATIONS
we model the task o ﬄoading and vehicular distributed service
caching process, including task communication, task comput-
ing, and vehicular service fetching. Finally, an optimization
problem is formulated to minimize the overall task completion
delay. For better readability, the primary notations utilized are
summarized in Table II.
A. Network Topology
A typical two-tier VEC network, as depicted in Fig. 1,
where there are N vehicles and an RSU located alongside the
urban bidirectional road is considered in this work. The RSU
is equipped with an edge server that caches all service models
Fig. 1. Network topology illustration.
and provides various computational services to vehicles via
V2I communications. In contrast, due to the limited storage
capacity, each vehicle caches only a subset of models for local
computation or processing tasks o ﬄoaded from V2V links.
If vehicles are willing to process the o ﬄoaded tasks but do
not have the corresp

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

PROBLEM FORMULATION
In this section, we first introduce the social-mobility-aware
network topology in caching-assisted VEC networks. Then,
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 30,2026 at 06:58:48 UTC from IEEE Xplore. Restrictions apply. 
TANG et al.: JOINT DESIGN OF SERVICE FETCHING, TASK OFFLOADING, AND RESOURCE ALLOCATION 3297
TABLE I
DIFFERENCE BETWEEN OUR SCHEME AND OTHER MAIN RELATED WORKS
TABLE II
PARAMETER NOTATIONS
we model the task o ﬄoading and vehicular distributed service
caching process, including task communication, task comput-
ing, and vehicular service fetching. Finally, an optimization
problem is formulated to minimize the overall task completion
delay. For better readability, the primary notations utilized are
summarized in Table II.
A. Network Topology
A typical two-tier VEC network, as depicted in Fig. 1,
where there are N vehicles and an RSU located alongside the
urban bidirectional road is considered in this work. The RSU
is equipped with an edge server that caches all service models
Fig. 1. Network topology illustration.
and provides various computational services to vehicles via
V2I communications. In contrast, due to the limited storage
capacity, each vehicle caches only a subset of models for local
computation or processing tasks o ﬄoaded from V2V links.
If vehicles are willing to process the o ﬄoaded tasks but do
not have the corresponding service cached, they can actively
fetch the service model from the RSU or other vehicles. We
represent the set of vehicles asN ={1, 2,..., N}, and the RSU
is identified as 0.
In reality, not all vehicles are willing to assist surrounding
vehicles in performing tasks. Motivated by [16], we categorize
vehicles into three types based on whether th

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

proposed algorithm
achieves fast convergence and reduces the task completion delay
by 9.27%-59.77% compared to five representative baselines.
Index Terms —Vehicular edge computing, service caching,
vehicular service fetching, task o ﬄoading, resource allocation.
I. I NTRODUCTION
V
EHICULAR edge computing (VEC) has emerged as a
promising computing service paradigm in the Internet
of Vehicles, aiming to provide low-latency services to nearby
Received 15 October 2024; revised 7 June 2025, 8 September 2025,
13 October 2025, and 9 November 2025; accepted 28 November 2025.
Date of publication 15 December 2025; date of current version 10 March
2026. This work was supported in part by the Joint Funds of the NSFC
under Grant U22A2003 and Grant U23A20277; and in part by the Key
Research and Development Program of Shandong Province, China, under
Grant 2022CXGC020107. The Associate Editor for this article was M.
Barreau. (Corresponding author: Haixia Zhang.)
Haochen Tang is with the School of Control Science and Engineering,
Shandong University, Jinan 250100, China, and also with Shandong Key
Laboratory of Intelligent Communication and Sensing-Computing Integration,
Jinan 250100, China (e-mail: haochen.tang@mail.sdu.edu.cn).
Haixia Zhang and Minggao Zhang are with the Institute of Intelli-
gent Communication Technologies, Shandong University, Jinan 250100,
China, and also with Shandong Key

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

simulation results demonstrate that our proposed algorithm
achieves fast convergence and reduces the task completion delay
by 9.27%-59.77% compared to five representative baselines.
Index Terms —Vehicular edge computing, service caching,
vehicular service fetching, task o ﬄoading, resource allocation.
I. I NTRODUCTION
V
EHICULAR edge computing (VEC) has emerged as a
promising computing service paradigm in the Internet
of Vehicles, aiming to provide low-latency services to nearby
Received 15 October 2024; revised 7 June 2025, 8 September 2025,
13 October 2025, and 9 November 2025; accepted 28 November 2025.
Date of publication 15 December 2025; date of current version 10 March
2026. This work was supported in part by the Joint Funds of the NSFC
under Grant U22A2003 and Grant U23A20277; and in part by the Key
Research and Development Program of Shandong Province, China, under
Grant 2022CXGC020107. The Associate Editor for this article was M.
Barreau. (Corresponding author: Haixia Zhang.)
Haochen Tang is with the School of Control Science and Engineering,
Shandong University, Jinan 250100, China, and also with Shandong Key
Laboratory of Intelligent Communication and Sensing-Computing Integration,
Jinan 250100, China (e-mail: haochen.tang@mail.sdu.edu.cn).
Haixia Zhang and Minggao Zhang are with the Institute of Intelli-
gent Communication Technologies, Shandong University, Jinan 2

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 部分 |
| 服务更新周期 | 未提及 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 是 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 部分 |
| 多业务QoS | 部分 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=部分；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=是；活动任务连续=未提及；迁移传输=部分 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=部分；业务流量非平稳=未提及；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：2（PDF页）
- 方法窗口：1（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：None（PDF页）

### 关键词命中片段

- p2: nt design for communication, sensing, and cache-assisted computing in VEC networks, where vehicles can o ﬄoad tasks and service models to RSUs via V2I communications. Moreover, dynamic service replacement [28], [29] and service migration [30], [31], [32] in VEC networks have also been considered. However, the above studies only focus on service cac
- p13: ient caching in vehicular edge computing based on edge-cloud collaboration,” IEEE Trans. Veh. Technol., vol. 72, no. 2, pp. 2468–2481, Feb. 2023. [31] Q.-X. Fan, L. Chen, C. You, Y . Chen, and H. Yin, “Dependency- aware service migration for backhaul-free vehicular edge computing networks,” IEEE Trans. Veh. Technol. , vol. 73, no. 1, pp. 1337–1352,

## 12. 阅读闭合状态

- 全文文本：已提取（14页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
