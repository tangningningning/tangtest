# L113 — Service-Aware Computation Offloading for Parallel Tasks in VEC Networks

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2025
- 刊物/会议：IEEE Internet of Things Journal
- DOI：10.1109/JIOT.2024.3476476
- PDF页数：15
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\Service-Aware Computation Offloading for Parallel Tasks in VEC Networks.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=77, UAV=0
- 主研究方向：P1；次方向：无；依据：scores={'1': 70, '2': 10, '3': 6}; title=Service-Aware Computation Offloading for Parallel Tasks in VEC Networks

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system models and problem formu-
lation. Section IV presents the edge caching decision-making
approach in vehicular task offloading, followed by the
learning-based vehicular task offloading in Section V.I n
Section VI, we present the evaluations, followed by the
conclusion in Section VII.
II. R ELATED WORK
Computation offloading has been studied in many existing
works, where computing-intensive tasks can be offloaded
from the end devices to the edge nodes. In this section,
we review and summarize the related work parts from three
aspects: 1) vehicular computation offloading;2 ) service-based
computation offloading ; and 3) learning-based computation
offloading to explain the innovation and necessity of our
proposed approach.
A. Vehicular Computation Offloading
With the deepening exploration of the Internet of Vehicles
(IoV), the field of vehicle computation and offloading has
reached a r

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system models and problem formu-
lation. Section IV presents the edge caching decision-making
approach in vehicular task offloading, followed by the
learning-based vehicular task offloading in Section V.I n
Section VI, we present the evaluations, followed by the
conclusion in Section VII.
II. R ELATED WORK
Computation offloading has been studied in many existing
works, where computing-intensive tasks can be offloaded
from the end devices to the edge nodes. In this section,
we review and summarize the related work parts from three
aspects: 1) vehicular computation offloading;2 ) service-based
computation offloading ; and 3) learning-based computation
offloading to explain the innovation and necessity of our
proposed approach.
A. Vehicular Computation Offloading
With the deepening exploration of the Internet of Vehicles
(IoV), the field of vehicle computation and offloading has
reached a relatively mature stage. We conducted an exten-
sive literature review focusing on computational offloading
in the IoV environment. Targeting the time-varying char-
acteristics of vehicle motion trajectories in urban scenes,
Zhao et al. [13] investigated a computational offload model
based on VEC and proposed a mobile awareness offload
scheme based on deep learning, which effectively provides
real-time computing services and reduces energy consumption.
Fu et al. [14] proposed a framework assisted by digital twin
for resource allocation to address the problem of vehicle-
to-vehicle communication resource allocation using MADRL
and TacNet. Similarly, Zhao et al. [15] proposed a distributed
task unloading architecture based on digital twin-assisted
VEC, integrating it with the PSO, GA, and SA algorithms to
effectively reduce computation delay and energy consumption
in vehicular networking sy

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p3起）

PROBLEM FORMULATION
In this section, we first elaborate on the system models,
including the system overview, the parallel task model, the ser-
vice caching model, and the offloading model. After that, we
formulate the service-aware parallel vehicular task offloading
problem.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 26,2026 at 11:55:10 UTC from IEEE Xplore. Restrictions apply. 
2982 IEEE INTERNET OF THINGS JOURNAL, VOL. 12, NO. 3, 1 FEBRUARY 2025
Fig. 1. Illustration of parallel task offloading in VEC networks.
TABLE I
MAIN ACRONYMS
A. System Overview
We consider a VEC system that consists of a set of vehicles
V and RSUs M, as shown in Fig. 1. The timeline in our system
is discretized into multiple time slots T ={ 1,..., t,..., T}.
Let ltv denote the location (in longitude and latitude) of vehicle
v at time slot t. The vehicular task arrivals and vehicular rate
are assumed to be fixed within a single time slot, but vary
across time slots. Thus, we update the edge offloading for
each time slot. Several time slots constitute a time period.
During each period, RSUs update edge caching decisions for
the offloaded vehicular tasks. Furthermore, considering the
dense deployment of RSUs, vehicle v may be within the
communication coverage of several RSUs. The vehicle can
offload its computing-intensive tasks to the candidate RSUs.
The candidate RSUs for each vehicle, due to vehicular mobil-
ity and limited V2I communication coverage, may change
across epochs Bb = [tb, tb′ ], where b = 1,..., B, t1 = 1,
and tB′ = T. The time length of an epoch is longer than that
of a time period. Additionally, the candidate RSUs need to
adjust the edge caching decisions based on the prediction of
service requests at the following time slo

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

proposed algorithm. The results
demonstrate that the proposed algorithm reduces task delay by
45%, 58%, and 55% compared to the algorithms without service-
aware computation offloading under various CPU cycles, task
numbers, and time slots.
Index Terms—Computation offloading, online learning, service
caching, service prediction, vehicular edge computing (VEC).
I. I NTRODUCTION
V
EHICULAR edge computing (VEC), known as dis-
tributed cloud computing, has emerged as a promising
Received 28 June 2024; revised 1 September 2024; accepted 4 October
2024. Date of publication 9 October 2024; date of current version
24 January 2025. This work was supported in part by the National Key
Research and Development Program of China under Grant 2023YFB3001805;
in part by the Key Program of the National Natural Science Foundation of
China under Grant 62232007 and Grant 62272152; in part by the Special
Key Projects for Technological Innovation and Application Development
of Chongqing under Grant CSTB2022TIAD-DEX0016; in part by the
Key Research and Development Program of Hunan Province under Grant
2024AQ2032; in part by the Cultivation Project of Yuelu Mountain
Industrial Innovation Center under Grant 2023YCII0123; and in part by the
Natural Science Foundation Project of Chongqing under Grant cstc2021jcyj-
msxm4115 and Grant CSTB2024NSCQ-MSX0920. (Corresponding authors:
Kehua Yang; Zhu Xiao; Hongb

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

simulations to validate the proposed algorithm. The results
demonstrate that the proposed algorithm reduces task delay by
45%, 58%, and 55% compared to the algorithms without service-
aware computation offloading under various CPU cycles, task
numbers, and time slots.
Index Terms—Computation offloading, online learning, service
caching, service prediction, vehicular edge computing (VEC).
I. I NTRODUCTION
V
EHICULAR edge computing (VEC), known as dis-
tributed cloud computing, has emerged as a promising
Received 28 June 2024; revised 1 September 2024; accepted 4 October
2024. Date of publication 9 October 2024; date of current version
24 January 2025. This work was supported in part by the National Key
Research and Development Program of China under Grant 2023YFB3001805;
in part by the Key Program of the National Natural Science Foundation of
China under Grant 62232007 and Grant 62272152; in part by the Special
Key Projects for Technological Innovation and Application Development
of Chongqing under Grant CSTB2022TIAD-DEX0016; in part by the
Key Research and Development Program of Hunan Province under Grant
2024AQ2032; in part by the Cultivation Project of Yuelu Mountain
Industrial Innovation Center under Grant 2023YCII0123; and in part by the
Natural Science Foundation Project of Chongqing under Grant cstc2021jcyj-
msxm4115 and Grant CSTB2024NSCQ-MSX0920. (Corresponding authors:

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 部分 |
| 服务更新周期 | 是 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 是 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 未提及 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=部分；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=是；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：3（PDF页）
- 方法窗口：1（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：2（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（15页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
