# L102 — Optimizing Task Migration for Public and Private Services in Vehicular Edge Networks: A Dual- Layer Graph Neural Network Approach

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2025
- 刊物/会议：IEEE Transactions on Mobile Computing
- DOI：10.1109/TMC.2025.3589245
- PDF页数：18
- 原文：`C:\Users\tangning\Desktop\文献\02_移动性感知的服务迁移与传输资源协同\Optimizing_Task_Migration_for_Public_and_Private_Services_in_Vehicular_Edge_Networks_A_Dual-_Layer_Graph_Neural_Network_Approach.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=17, UAV=0
- 主研究方向：P2；次方向：3；依据：scores={'1': 4, '2': 28, '3': 14}; title=Optimizing Task Migration for Public and Private Services in Vehicular Edge Netw

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model
in Section IV. Next, an enhanced T-MFA algorithm is proposed
in Section V, with the corresponding simulation experiments
described and discussed in SectionVI. Finally, future researches
of related issues follows in Section VII, with conclusions in
Section VIII.
II. R ELA TEDWORKS
A. Resource Allocation in PMN
PMNs are tasked with sensing the real-time state of vehicles,
their driving trajectories, and the current status of network
resources, thereby dynamically allocating computation, storage,
and communication resources. This process necessitates the
prediction of vehicle trajectories and the real-time monitoring
of network resource utilization, posing significant challenges
in the development of resource allocation strategies capable of
processing such information instantaneously. Furthermore, the
Authorized licensed use limited to: Indian Institute of Petroleum and Energy

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model
in Section IV. Next, an enhanced T-MFA algorithm is proposed
in Section V, with the corresponding simulation experiments
described and discussed in SectionVI. Finally, future researches
of related issues follows in Section VII, with conclusions in
Section VIII.
II. R ELA TEDWORKS
A. Resource Allocation in PMN
PMNs are tasked with sensing the real-time state of vehicles,
their driving trajectories, and the current status of network
resources, thereby dynamically allocating computation, storage,
and communication resources. This process necessitates the
prediction of vehicle trajectories and the real-time monitoring
of network resource utilization, posing significant challenges
in the development of resource allocation strategies capable of
processing such information instantaneously. Furthermore, the
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on November 03,2026 at 13:03:01 UTC from IEEE Xplore. Restrictions apply. 
HUANG et al.: OPTIMIZING TASK MIGRA TION FOR PUBLIC AND PRIV A TE SERVICES IN VEN: A DUAL-LA YER GNN APPROACH 13193
accurate collection of information is impeded by the interference
between sensed signals and communications and the impact of
external environmental factors on sensing devices. To address
these issues, researchers are investigating a variety of technical
solutions, including the optimization of bandwidth allocation
strategies and the creation of algorithms for communication and
sensing that are resistant to interference [3], [5], [6], [7].
Xu et al. proposed a non-convex optimization approach for ad-
dressing the resource allocation problem, taking into account the
imperfections of channel state information. Initially, the bound-
aries of the small-scale fading uncertainty region for pot

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

optimization problem based on the system model
in Section IV. Next, an enhanced T-MFA algorithm is proposed
in Section V, with the corresponding simulation experiments
described and discussed in SectionVI. Finally, future researches
of related issues follows in Section VII, with conclusions in
Section VIII.
II. R ELA TEDWORKS
A. Resource Allocation in PMN
PMNs are tasked with sensing the real-time state of vehicles,
their driving trajectories, and the current status of network
resources, thereby dynamically allocating computation, storage,
and communication resources. This process necessitates the
prediction of vehicle trajectories and the real-time monitoring
of network resource utilization, posing significant challenges
in the development of resource allocation strategies capable of
processing such information instantaneously. Furthermore, the
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on November 03,2026 at 13:03:01 UTC from IEEE Xplore. Restrictions apply. 
HUANG et al.: OPTIMIZING TASK MIGRA TION FOR PUBLIC AND PRIV A TE SERVICES IN VEN: A DUAL-LA YER GNN APPROACH 13193
accurate collection of information is impeded by the interference
between sensed signals and communications and the impact of
external environmental factors on sensing devices. To address
these issues, researchers are investigating a variety of technical
solutions, including the optimization of bandwidth allocation
strategies and the creation of algorithms for communication and
sensing that are resistant to interference [3], [5], [6], [7].
Xu et al. proposed a non-convex optimization approach for ad-
dressing the resource allocation problem, taking into account the
imperfections of channel state information. Initially, the bound-
aries of the small-scale

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

proposed algorithm, demonstrating its capability
to achieve optimal task migration by accommodating differences
in tasks, resources, and capacities.
Index Terms —V ehicle edge network (VEN), task migration,
graph neural network, maximum flow algorithm.
Received 28 June 2024; revised 23 May 2025; accepted 4 July 2025. Date
of publication 15 July 2025; date of current version 5 November 2025. This
work was supported in part by the National Natural Science Foundation of
China under Grant 62072096, in part by the Program for Professor of Special
Appointment (Eastern Scholar) at Shanghai Institutions of Higher Learning,
in part by the Science and Technology Commission of Shanghai Municipality
under Grant 23XD1420100, and in part by the Fundamental Research Funds
for the Central Universities and Graduate Student Innovation Fund of Donghua
University. Recommended for acceptance by M. Segata.(Corresponding author:
Guanglin Zhang.)
Xiaowen Huang, Shuguang Zhao, and Guanglin Zhang are with the Col-
lege of Information Science and Technology, Donghua University, Shanghai
201620, China (e-mail: huangxiaowen@mail.dhu.edu.cn; sgzhao@dhu.edu.cn;
glzhang@dhu.edu.cn).
Tao Huang is with the College of Science and Engineering, James Cook
University, Smithfield, QLD 4878, Australia (e-mail: tao.huang1@jcu.edu.au).
Peng Cheng is with the Department of Computer Science and Information
Technology, La

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

Simulation experiments validate
the efficacy of the proposed algorithm, demonstrating its capability
to achieve optimal task migration by accommodating differences
in tasks, resources, and capacities.
Index Terms —V ehicle edge network (VEN), task migration,
graph neural network, maximum flow algorithm.
Received 28 June 2024; revised 23 May 2025; accepted 4 July 2025. Date
of publication 15 July 2025; date of current version 5 November 2025. This
work was supported in part by the National Natural Science Foundation of
China under Grant 62072096, in part by the Program for Professor of Special
Appointment (Eastern Scholar) at Shanghai Institutions of Higher Learning,
in part by the Science and Technology Commission of Shanghai Municipality
under Grant 23XD1420100, and in part by the Fundamental Research Funds
for the Central Universities and Graduate Student Innovation Fund of Donghua
University. Recommended for acceptance by M. Segata.(Corresponding author:
Guanglin Zhang.)
Xiaowen Huang, Shuguang Zhao, and Guanglin Zhang are with the Col-
lege of Information Science and Technology, Donghua University, Shanghai
201620, China (e-mail: huangxiaowen@mail.dhu.edu.cn; sgzhao@dhu.edu.cn;
glzhang@dhu.edu.cn).
Tao Huang is with the College of Science and Engineering, James Cook
University, Smithfield, QLD 4878, Australia (e-mail: tao.huang1@jcu.edu.au).
Peng Cheng is with the Departmen

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 未提及 |
| 服务更新周期 | 未提及 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 是 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 是 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=未提及；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=是；活动任务连续=未提及；迁移传输=是 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

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

- p3:  this challenge [16], [17], [18], [19], [20], [21]. To tackle the dynamic, unpredictable, and evolving nature of vehicle networks, numerous scholars have dedicated signiﬁcant efforts. For instance, Liu et al. considered service migration and vehicle resource availability comprehensively and proposed strategies for task and resource scheduling to ad
- p3: urce scheduling to address the joint optimization problem of distributed task ofﬂoading and multi- resource management [16]. Fan et al. introduced a non-backhaul migration scheme that leverages dependency perception for service migration [17], utilizing a time-varying Markov decision process to develop a joint ofﬂoading and migration optimiza- tion

## 12. 阅读闭合状态

- 全文文本：已提取（18页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
