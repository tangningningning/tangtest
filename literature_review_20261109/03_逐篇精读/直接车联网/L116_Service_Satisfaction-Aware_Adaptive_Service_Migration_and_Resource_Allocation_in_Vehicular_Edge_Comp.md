# L116 — Service Satisfaction-Aware Adaptive Service Migration and Resource Allocation in Vehicular Edge Computing

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2026
- 刊物/会议：IEEE Transactions on Mobile Computing
- DOI：10.1109/TMC.2025.3596342
- PDF页数：17
- 原文：`C:\Users\tangning\Desktop\文献\18篇指定论文合集\Service Satisfaction-Aware Adaptive Service Migration and Resource Allocation in Vehicular Edge Computing.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=19, UAV=0
- 主研究方向：P2；次方向：无；依据：scores={'1': 4, '2': 89, '3': 23}; title=Service Satisfaction-Aware Adaptive Service Migration and Resource Allocation in

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model is introduced in
Section III. Section IV presents the proposed SAMAO method.
The numerical results based on the Shanghai Telecom dataset are
presented in SectionV, followed by the conclusion in SectionVI.
II. R ELA TEDWORK
When vehicles leave the coverage area of an edge server,
service migration becomes a critical method to maintain ser-
vice continuity. The primary idea is to migrate services to the
nearest available edge server, seamlessly following the user’s
movement, to minimize service interruptions or degradation
for users. To address the impairment of QoS and latency in-
creasing posed by user mobility, service migration adaptively
balances energy consumption and latency by migrating services
to nearby servers. In this scenario, well-designed migration
methods can mitigate energy consumption and prevent QoS
degradation [20], [21]. Numerous studies have been conducte

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model is introduced in
Section III. Section IV presents the proposed SAMAO method.
The numerical results based on the Shanghai Telecom dataset are
presented in SectionV, followed by the conclusion in SectionVI.
II. R ELA TEDWORK
When vehicles leave the coverage area of an edge server,
service migration becomes a critical method to maintain ser-
vice continuity. The primary idea is to migrate services to the
nearest available edge server, seamlessly following the user’s
movement, to minimize service interruptions or degradation
for users. To address the impairment of QoS and latency in-
creasing posed by user mobility, service migration adaptively
balances energy consumption and latency by migrating services
to nearby servers. In this scenario, well-designed migration
methods can mitigate energy consumption and prevent QoS
degradation [20], [21]. Numerous studies have been conducted
on service migration in edge computing. These studies have
focused on optimizing the total migration time and the overall
cost of migration.
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 30,2026 at 13:51:35 UTC from IEEE Xplore. Restrictions apply. 
708 IEEE TRANSACTIONS ON MOBILE COMPUTING, VOL. 25, NO. 1, JANUARY 2026
A. Migration Time Optimization
Migration time is a crucial metric in service migration. Wang
et al. [22] utilize an entropy-based location privacy metric
to accurately assess the risk of location privacy leakage for
users. The service migration process is then formulated as a
joint optimization problem, and a Multi-Agent Soft Actor-Critic
(MASAC) algorithm is developed to minimize both service
response latency and the risk of location privacy breaches. Ning
et al. [3] establish a collaborative service migration frame

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

optimization problem, which is NP-hard.
Through a thorough analysis of its hierarchical structure,
we decompose the problem into a two-level optimization
problem. Then, we propose SAMAO to solve the two-level
optimization problem, which can minimize energy con-
sumption while ensuring service satisfaction.
r We propose a service satisfaction based adaptive compu-
tation resource allocation algorithm, which introduces a
weight factor for edge servers to balance service satisfac-
tion and computation cost. This algorithm can perceive the
number of vehicles deployed on edge servers and adjust the
weight factor to enable personalized decisions for servers
with different load status to maximize the average service
satisfaction.
r A utility-maximizing service migration decision-making
algorithm is proposed to make the optimal service migra-
tion decision with the goal of minimizing computation cost
and ensuring service satisfaction of vehicles.
r The simulation results on the Shanghai Telecom real-world
dataset demonstrate that SAMAO can efficiently utilize
computation resources and outperform existing strategies
in terms of average service satisfaction and computation
cost.
The rest of this paper is organized as follows. Section II
introduces the related work. The system model is introduced in
Section III. Section IV presents the proposed SAMAO method.
The numerical results based on the Shanghai Telecom dataset are
presented in SectionV, followed by the conclusion in SectionVI.
II. R ELA TEDWORK
When vehicles leave the coverage area of an edge server,
service migration becomes a critical method to maintain ser-
vice continuity. The primary idea is to migrate services to the
nearest available edge server, seamlessly following the user’s
movement, to minimize service interrupt

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

solution, providing low
latency and high computing capabilities for various vehicular
services.
By deploying services at the edge of VEC networks, such as
road side units (RSUs) and base stations (BSs), VEC can provide
vehicles with low-latency computing capability. However, due
to the limited coverage of VEC servers, vehicles traveling at
high speeds may frequently drive out of the coverage range
of the VEC servers. The increase in distance between VEC
servers and vehicles will significantly degrade the quality of
service (QoS), further leading to a decrease in the perceived
service satisfaction of vehicles. Service migration helps services
follow moving vehicles, thereby optimizing request routing and
ensuring low-latency, high-availability computational support
for the vehicles. In future smart driving scenarios, key appli-
cations such as driving decision-making, collision detection,
and high-precision map navigation have high requirements on
communication latency and reliability. By migrating services
to servers closer to the vehicle, faster server access speeds and
higher reliability of information transmission can be achieved.
Therefore, designing an efficient service migration strategy is
crucial.
In the past few years, service migration has been studied
in mobile networks [9], [10], [11], [12], [13], [14]. Existing
migration methods mainly focus on optimizing QoS metri

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

simulations based on Shanghai Telecom
real-world dataset show that SAMAO can achieve significant ad-
vantages in terms of average service satisfaction and computation
cost.
Index Terms—Service migration, resource allocation, vehicular
edge computing, service satisfaction.
Received 9 January 2025; revised 27 May 2025; accepted 28 July 2025.
Date of publication 6 August 2025; date of current version 3 December 2025.
This work was supported in part by the National Natural Science Foundation
of China under Grant 62471121, in part by the Fundamental Research Funds
for the Central Universities of China under Grant N2424010-18, in part by
Shenyang Science and Technology Plan Fund Project under Grant 23-503-6-17,
in part by Liaoning Provincial Science and Technology Plan Project under Grant
2023JH2/101700370, and in part by the Fund of National Key Laboratory of
Metallurgical Intelligent Manufacturing System. Recommended for acceptance
by L. Bai. (Corresponding author: Yuanguo Bi.)
Y ufei Liu, Y uanguo Bi, Y uheng Liu, and Kaiqi Y ang are with the School
of Computer Science and Engineering, Northeastern University, Shenyang
110169, China (e-mail: liuyufei1119@163.com; biyuanguo@mail.neu.edu.cn;
liuyuheng_jan@163.com; kaiqiyangkq@163.com).
Dusit Niyato is with the School of Computer Science and Engineer-
ing, Nanyang Technological University, Singapore 639798 (e-mail: dniyato
@ntu.edu.s

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
| 服务迁移 | 是 |
| 活动任务连续执行 | 部分 |
| 迁移传输 | 是 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 部分 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=是；活动任务连续=部分；迁移传输=是 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=部分 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：2（PDF页）
- 方法窗口：1（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：2（PDF页）

### 关键词命中片段

- p1: o Bi , Member , IEEE, Y uheng Liu , Dusit Niyato , Fellow, IEEE, Kaiqi Y ang , Liang Zhao , Member , IEEE, and Ammar Hawbani , Member , IEEE Abstract—With the rapid development of vehicle-to-everything (V2X) technology, service migration has become an important approach to provide low-latency computing services and ensure service continuity for hig
- p1:  and ensure service continuity for high-speed moving vehicles in vehicular edge computing (VEC), which enables VEC to efﬁciently support ad- vanced transportation services. However, optimizing service satis- faction for service migration in multi-vehicle heterogeneous VEC networks is challenging, since the complex, multifactorial, and nonlinear dep

## 12. 阅读闭合状态

- 全文文本：已提取（17页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
