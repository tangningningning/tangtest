# L131 — Two-Timescale Resource Management for Ultrareliable and Low-Latency Vehicular Communications

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2022
- 刊物/会议：IEEE Transactions on Communications
- DOI：10.1109/TCOMM.2022.3162366
- PDF页数：13
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\Two-Timescale Resource Management for Ultrareliable and Low-Latency Vehicular Communications.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=28, UAV=0
- 主研究方向：P3；次方向：1；依据：scores={'1': 10, '2': 2, '3': 17}; title=Two-Timescale Resource Management for Ultrareliable and Low-Latency Vehicular Co

## 2. 研究场景与问题

### 摘要窗口

（PDF p3起）

system model and proposes the novel two
time-scale resource allocation framework. Section III formu-
lates the large-scale resource optimization problem and derives
the optimal algorithm. In Section IV , the small-scale resource
optimization problem is formulated and the corresponding
optimal solution is devised. In Section V , simulation results
are provided to demonstrate the effectiveness of the proposed
algorithm. Finally, the article is concluded in Section VI.
II. S YSTEM MODEL AND RESOURCE
ALLOCATION FRAMEWORK
In this section, we first present the system scenario, trans-
mission model, and queuing model of the V2V network. After
that, a resource allocation framework is proposed to guarantee
the latency and reliability requirements of V2V links.
A. System Scenario
As shown in Fig. 1a, we consider a V2V network in the
D2D-enabled cellular system, where K pairs of proximate
vehicles,

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p3起）

system model and proposes the novel two
time-scale resource allocation framework. Section III formu-
lates the large-scale resource optimization problem and derives
the optimal algorithm. In Section IV , the small-scale resource
optimization problem is formulated and the corresponding
optimal solution is devised. In Section V , simulation results
are provided to demonstrate the effectiveness of the proposed
algorithm. Finally, the article is concluded in Section VI.
II. S YSTEM MODEL AND RESOURCE
ALLOCATION FRAMEWORK
In this section, we first present the system scenario, trans-
mission model, and queuing model of the V2V network. After
that, a resource allocation framework is proposed to guarantee
the latency and reliability requirements of V2V links.
A. System Scenario
As shown in Fig. 1a, we consider a V2V network in the
D2D-enabled cellular system, where K pairs of proximate
vehicles, denoted as vehicle u sers (VUEs), are performing
D2D communications with the assist of a single roadside unit
(RSU). The slotted transmission model is considered where
the slots are indexed by n, each with a duration of T .T h e
total system bandwidth is Bt and is divided into M orthogonal
RBs with bandwidth B = Bt/M. The sets of VUEs and RBs
are denoted by K = {1, 2,...,K } and M = {1, 2,...,M },
respectively. Assume that each VUE is equipped with a single
antenna and an infinite-size first-in first-out queue which
stores the data to be transmitted. In this work, we assume
that the transmissions of the V2V pairs have already been
synchronized. In addition, the time synchronization accuracy
is accurate enough so that the transmissions are not impacted
by the synchronization bias. 1
1The accuracy achieved by GNSS-based time synchronization is better than
40 ns for more than 95% of time [

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p1起）

optimization problem to
shorten the queue length of V2V pairs. In [13], the reliability
and latency metrics of autonomous vehicular networks have
been formulated and jointly optimized by utilizing the Euclid-
ean norm theory. In [14], an integrated communication and
control framework has been proposed to analyze the latency
and reliability performance and select the control parameters
of wireless connected vehicular platoons. Considering the
mmWave signal propagation characteristics and the road topol-
ogy, a performance analytical framework has been developed
and a relay selection scheme has been proposed to improve
the latency and reliability performance of mmWave multi-hop
V2V communications [15]. In [16], the distribution of queue
length is modeled by extreme value theory (EVT), which is
a powerful tool to characterize the occurrence probability of
extreme events [17]. Based on this, the sum power of V2V
links is minimized while guaranteeing the queue stability.
0090-6778 © 2022 I EEE. Personal u se is perm itted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Shaanxi Normal University. Downloaded on June 02,2026 at 10:11:34 UTC from IEEE Xplore. Restrictions apply. 
DING et al. : TWO-TIMESCALE RESOURCE MANAGEMENT FOR ULTRARELIABLE AND LOW-LATENCY VEHICULAR COMMUNICATIONS 3283
In order to reduce communication overhead, the work in [18]
has proposed a Lyapunov-based distributed resource allocation
algorithm to reduce the queuing latency by taking into account
both EVT and federated learning. In [19], the authors have pro-
posed a non-cooperative radio resource management algorithm
for V2V networks based on the game theory, where V2V pairs
comp

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

proposed algorithm can effectively improve the URLLC
performance compared against the benchmark algorithm.
Index Terms — 5G, ultra-reliable low-latency communications
(URLLC), vehicle-to-vehicle (V2V) communications, resource
allocation.
I. I NTRODUCTION
V
EHICLE-TO-VEHICLE (V2V ) communication, as an
important application scenario in the 5th-generation (5G)
network, is regarded as one of the most promising enablers
for autonomous driving and intelligent transportation systems
[1]–[3]. The communication link between vehicles supports
various autonomous applications such as forward collision
warning, lane change warning, and real-time navigation, which
Manuscript received August 24, 2021; revised December 17, 2021 and
February 8, 2022; accepted March 22, 2022. Date of publication March 25,
2022; date of current version May 18, 2022. This work was supported in
part by the National Key Research and Development Program of China
(No. 2018YFB1802300) and in part by GDNRC[2021]32. The associate editor
coordinating the review of this artic le and approving it for publication was
C. Li. (Corresponding author: Guanding Yu.)
Guangyao Ding and Guanding Yu are with the College of Information
Science and Electronic Engineering, Zhejiang University, Hangzhou 310027,
China, and also with the Zhejiang Provincial Key Laboratory of Information
Processing, Communication and Ne tworking (IPCAN), Ha

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

Simulation results demonstrate that
the proposed algorithm can effectively improve the URLLC
performance compared against the benchmark algorithm.
Index Terms — 5G, ultra-reliable low-latency communications
(URLLC), vehicle-to-vehicle (V2V) communications, resource
allocation.
I. I NTRODUCTION
V
EHICLE-TO-VEHICLE (V2V ) communication, as an
important application scenario in the 5th-generation (5G)
network, is regarded as one of the most promising enablers
for autonomous driving and intelligent transportation systems
[1]–[3]. The communication link between vehicles supports
various autonomous applications such as forward collision
warning, lane change warning, and real-time navigation, which
Manuscript received August 24, 2021; revised December 17, 2021 and
February 8, 2022; accepted March 22, 2022. Date of publication March 25,
2022; date of current version May 18, 2022. This work was supported in
part by the National Key Research and Development Program of China
(No. 2018YFB1802300) and in part by GDNRC[2021]32. The associate editor
coordinating the review of this artic le and approving it for publication was
C. Li. (Corresponding author: Guanding Yu.)
Guangyao Ding and Guanding Yu are with the College of Information
Science and Electronic Engineering, Zhejiang University, Hangzhou 310027,
China, and also with the Zhejiang Provincial Key Laboratory of Information
Processing, C

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
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 是 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=是；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：3（PDF页）
- 问题窗口：1（PDF页）
- 方法窗口：1（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：6（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（13页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
