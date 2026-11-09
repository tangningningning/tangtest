# L090 — Mobility-Aware Computation Offloading With Load Balancing in Smart City Networks Using MEC Federation

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2024
- 刊物/会议：IEEE Transactions on Mobile Computing
- DOI：10.1109/TMC.2024.3376377
- PDF页数：18
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\Mobility-Aware Computation Offloading With Load Balancing in Smart City Networks Using MEC Federation.pdf`
- 领域分类：**C**；依据：摘要/模型段 MEC=19, vehicle=1, UAV=0
- 主研究方向：P1；次方向：无；依据：scores={'1': 53, '2': 13, '3': 22}; title=Mobility-Aware Computation Offloading With Load Balancing in Smart City Networks

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model and problem for-
mulation are elaborated in SectionIII. Section IV then illustrates
the JOA approach of MDCM and DDPG-based decentralized
solution. In Section V, the performance evaluation is analyzed.
Finally, we conclude this paper in Section VI.
II. R ELA TEDWORK
Computation Offloading in MEC: Extensive studies have
recently been made to solve the challenges of computation
offloading and resource allocation in MEC. Most of these stud-
ies are based on optimization or AI-driven techniques. As for
optimization techniques, in [17], Wang et al. proposed a novel
edge computing platform to minimize the overall system cost
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 29,2026 at 12:26:38 UTC from IEEE Xplore. Restrictions apply. 
HUANG et al.: MOBILITY -AW ARE COMPUTA TION OFFLOADING WITH LOAD BALANCING 10413
during the offlo

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model and problem for-
mulation are elaborated in SectionIII. Section IV then illustrates
the JOA approach of MDCM and DDPG-based decentralized
solution. In Section V, the performance evaluation is analyzed.
Finally, we conclude this paper in Section VI.
II. R ELA TEDWORK
Computation Offloading in MEC: Extensive studies have
recently been made to solve the challenges of computation
offloading and resource allocation in MEC. Most of these stud-
ies are based on optimization or AI-driven techniques. As for
optimization techniques, in [17], Wang et al. proposed a novel
edge computing platform to minimize the overall system cost
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 29,2026 at 12:26:38 UTC from IEEE Xplore. Restrictions apply. 
HUANG et al.: MOBILITY -AW ARE COMPUTA TION OFFLOADING WITH LOAD BALANCING 10413
during the offloading process in a complicated dynamic campus
environment. Then they utilized the alternating direction method
of multipliers (ADMM) to address the multi-objective latency
optimization problem. Furthermore, Pan et al. [18] formulated a
multi-workflow offloading problem with the sub-deadline con-
straint. They proposed a multi-objective clustering evolutionary
algorithm to optimize the system performance regarding latency
and energy consumption. As for AI-driven techniques, Zhang
et al. [19] considered the dynamic decision-making problem
of computation offloading for multiple users under a dynamic
environment. The problem was formulated as a partially ob-
servable Markov decision process, and a decentralized compu-
tation offloading D-DRL framework was devised to obtain the
optimum offloading strategy. A joint optimization problem of
quality of experience and energy consumption as a const

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p1起）

optimization problem of computation
offloading, task migration, and resource allocation in the MEC
federation. The objective is to minimize the weighted sum of latency
and energy consumption while maintaining load balancing under
the constraint of the long-term migration cost budget of EIPs.
To address the problem, we decompose it into two sub-problems:
1) the MDs clustering sub-problem and 2) the sub-problem of joint
computation offloading, task migration, and resource allocation.
First, an MDs clustering matching (MDCM) algorithm is pro-
posed to cluster the MDs in edge servers (ESs) according to the
differences in channel gains. Afterward, the second sub-problem
is simplified by the Lyapunov optimization technique, and then
we propose a Transformer-based mobility prediction model and
a decentralized deep deterministic policy gradient (DDPG)-based
framework to solve it. Extensive simulation results demonstrate the
cost-efficiency of the proposed algorithm.
Index Terms —Computation offloading, deep reinforcement
learning, mobile edge computing, resource allocation, task
migration.
I. I NTRODUCTION
A
S A new model of urban development, the smart city
integrates the concept of efficiency and convenience and
the technical means of perception, interconnection, and intel-
ligence. This integration provides a promising development
opportunity for promoting industrial development and economic
prosperity [1]. Innovative Internet-of-Things (IoT) applications
Manuscript received 23 June 2023; revised 18 January 2024; accepted 5 March
2024. Date of publication 18 March 2024; date of current version 3 October 2024.
Recommended for acceptance by W. Gao. (Corresponding author: Wenhan
Zhan.)
Hualong Huang and Wenhan Zhan are with the University of Electronic
Science and Technology of

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

proposed algorithm.
Index Terms —Computation offloading, deep reinforcement
learning, mobile edge computing, resource allocation, task
migration.
I. I NTRODUCTION
A
S A new model of urban development, the smart city
integrates the concept of efficiency and convenience and
the technical means of perception, interconnection, and intel-
ligence. This integration provides a promising development
opportunity for promoting industrial development and economic
prosperity [1]. Innovative Internet-of-Things (IoT) applications
Manuscript received 23 June 2023; revised 18 January 2024; accepted 5 March
2024. Date of publication 18 March 2024; date of current version 3 October 2024.
Recommended for acceptance by W. Gao. (Corresponding author: Wenhan
Zhan.)
Hualong Huang and Wenhan Zhan are with the University of Electronic
Science and Technology of China, Chengdu 611731, China (e-mail: zhanwen-
han@uestc.edu.cn).
Geyong Min is with the College of Engineering, Mathematics and Physical
Sciences, University of Exeter, EX4 4QF Exeter, U.K.
Zhekai Duan is with the University of Edinburgh, EH8 9JU Edinburgh, U.K.
Kai Peng is with Huaqiao University, Quanzhou, Fujian 362000, China.
Digital Object Identifier 10.1109/TMC.2024.3376377
have brought significant breakthroughs to urban development
concepts and management models. In the past few years, the
number of mobile devices (MDs) has proliferated w

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

simulation results demonstrate the
cost-efficiency of the proposed algorithm.
Index Terms —Computation offloading, deep reinforcement
learning, mobile edge computing, resource allocation, task
migration.
I. I NTRODUCTION
A
S A new model of urban development, the smart city
integrates the concept of efficiency and convenience and
the technical means of perception, interconnection, and intel-
ligence. This integration provides a promising development
opportunity for promoting industrial development and economic
prosperity [1]. Innovative Internet-of-Things (IoT) applications
Manuscript received 23 June 2023; revised 18 January 2024; accepted 5 March
2024. Date of publication 18 March 2024; date of current version 3 October 2024.
Recommended for acceptance by W. Gao. (Corresponding author: Wenhan
Zhan.)
Hualong Huang and Wenhan Zhan are with the University of Electronic
Science and Technology of China, Chengdu 611731, China (e-mail: zhanwen-
han@uestc.edu.cn).
Geyong Min is with the College of Engineering, Mathematics and Physical
Sciences, University of Exeter, EX4 4QF Exeter, U.K.
Zhekai Duan is with the University of Edinburgh, EH8 9JU Edinburgh, U.K.
Kai Peng is with Huaqiao University, Quanzhou, Fujian 362000, China.
Digital Object Identifier 10.1109/TMC.2024.3376377
have brought significant breakthroughs to urban development
concepts and management models. In the past few ye

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 是 |
| 服务更新周期 | 未提及 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 是 |
| 活动任务连续执行 | 部分 |
| 迁移传输 | 部分 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=是；活动任务连续=部分；迁移传输=部分 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：1（PDF页）
- 方法窗口：1（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：None（PDF页）

### 关键词命中片段

- p3: . This algorithm addresses the challenges associated with mobility awareness and dynamic migration of MEC ser- vices, focusing on ensuring service continuity for vehicles. Li et al. [22] ﬁrst investigated the problem of service migration in fog computing-enabled cellular networks to maintain service continuity by considering the limited computing a
- p17: o, and A. Xiong, “Blockchain-based VEC network trust management: A DRL algorithm for vehicular service ofﬂoading and migration,” IEEE Trans. V eh. Technol. , vol. 70, no. 8, pp. 8148–8160, Aug. 2021. [22] J. Li et al., “Service migration in fog computing enabled cellular networks to support real-time vehicular communications,” IEEE Access ,v o l .7

## 12. 阅读闭合状态

- 全文文本：已提取（18页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
