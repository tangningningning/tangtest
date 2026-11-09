# L122 — TJCCT: A Two-Timescale Approach for UAV-Assisted Mobile Edge Computing

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2025
- 刊物/会议：IEEE Transactions on Mobile Computing
- DOI：10.1109/TMC.2024.3505155
- PDF页数：18
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\TJCCT- A Two-Timescale Approach for UAV-Assisted Mobile Edge Computing.pdf`
- 领域分类：**D**；依据：摘要/模型段 vehicle=6, UAV=1
- 主研究方向：P1；次方向：无；依据：scores={'1': 58, '2': 22, '3': 9}; title=TJCCT: A Two-Timescale Approach for UAV-Assisted Mobile Edge Computing

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model
and problem formulation. Section IV elaborates on the proposed
TJCCT. The theoretical analysis is given in SectionV. SectionVI
shows the simulation results and discussions. Finally, this work
is concluded in Section VIII.
II. R ELA TEDWORK
In this section, we review the related work on the edge com-
puting architecture, joint computation offloading, computing
resource allocation, and the optimization approach.
A. MEC Architecture
The MEC has been extensively studied to extend the com-
puting capabilities of MDs through computation offloading.
Numerous research efforts have focused on leveraging terrestrial
MEC to offer low-latency offloading services in different sce-
narios. For example, Wang et al.[2] presented a non-cooperative
computation offloading approach in a single-BS vehicular MEC
network. Furthermore, Tao et al.[3] proposed a dynamic pricing-
based computation off

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model
and problem formulation. Section IV elaborates on the proposed
TJCCT. The theoretical analysis is given in SectionV. SectionVI
shows the simulation results and discussions. Finally, this work
is concluded in Section VIII.
II. R ELA TEDWORK
In this section, we review the related work on the edge com-
puting architecture, joint computation offloading, computing
resource allocation, and the optimization approach.
A. MEC Architecture
The MEC has been extensively studied to extend the com-
puting capabilities of MDs through computation offloading.
Numerous research efforts have focused on leveraging terrestrial
MEC to offer low-latency offloading services in different sce-
narios. For example, Wang et al.[2] presented a non-cooperative
computation offloading approach in a single-BS vehicular MEC
network. Furthermore, Tao et al.[3] proposed a dynamic pricing-
based computation offloading scheme for a single-cell and mult-
user MEC system. However, these studies mainly rely on the
terrestrial MEC servers. In the dense area, the link between
users and MEC servers experience severe blockage and poor
signal strength.
To extend the capability of the traditional MEC system, some
studies focused on cooperative computing schemes to fully
exploit the computing resources of users. For example, Y ou
et al. [4] explored a co-computing system where a request user
offloads computation to a helper user with idle resources. Xie
et al. [5] focused on the cooperative computing for mobile
crowdsensing, where the sensor data of source users can be
offloaded to nearby mobile devices with unused computational
resources. Moreover, Wei et al. [6] proposed a novel multi-tier
task offloading approach in the vehicular fog computing, where
vehicles with idle resources cooperatively provide

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p2起）

problem formulation. Section IV elaborates on the proposed
TJCCT. The theoretical analysis is given in SectionV. SectionVI
shows the simulation results and discussions. Finally, this work
is concluded in Section VIII.
II. R ELA TEDWORK
In this section, we review the related work on the edge com-
puting architecture, joint computation offloading, computing
resource allocation, and the optimization approach.
A. MEC Architecture
The MEC has been extensively studied to extend the com-
puting capabilities of MDs through computation offloading.
Numerous research efforts have focused on leveraging terrestrial
MEC to offer low-latency offloading services in different sce-
narios. For example, Wang et al.[2] presented a non-cooperative
computation offloading approach in a single-BS vehicular MEC
network. Furthermore, Tao et al.[3] proposed a dynamic pricing-
based computation offloading scheme for a single-cell and mult-
user MEC system. However, these studies mainly rely on the
terrestrial MEC servers. In the dense area, the link between
users and MEC servers experience severe blockage and poor
signal strength.
To extend the capability of the traditional MEC system, some
studies focused on cooperative computing schemes to fully
exploit the computing resources of users. For example, Y ou
et al. [4] explored a co-computing system where a request user
offloads computation to a helper user with idle resources. Xie
et al. [5] focused on the cooperative computing for mobile
crowdsensing, where the sensor data of source users can be
offloaded to nearby mobile devices with unused computational
resources. Moreover, Wei et al. [6] proposed a novel multi-tier
task offloading approach in the vehicular fog computing, where
vehicles with idle resources cooperatively provide edge comput-
ing

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p13起）

proposed algorithm has a polynomial
worst-case complexity in each time slot, i.e., O(ιmax(|U|+
1)(2|Ktreq| +m i n{|U|+1,|Ktreq|})+( |I||U|)3.5 log2(1/ϵ)),
where |Ktreq| and |U| are the numbers of undecided tasks and
MEC servers in time slot t, respectively.
Proof: The detailed proof is given in Appendix L of the
supplemental material, available online. ■
VI. S IMULA TIONRESULTS AND ANAL YSIS
In this section, simulation results are presented to validate the
effectiveness of the proposed approach.
A. Simulation Setup
1) Scenarios: We consider a UA V -assisted MEC system
where one MBS and four UA Vs are deployed to jointly provide
offloading service for 30 MDs in a 1000×1000 m2 rectangular
area. The time horizon is set as 60s, and it is divided intoT = 600
time slots with equal length of δ = 100ms, and 10 time slots
are grouped into a time epoch, i.e., Δ=1 s.
2) Parameters: For the MBS, the location and height are set
as [500, 500] and 10 m [41], respectively. For UA Vs, the fixed
altitude is set asH= 100 m, and the initial positions and destina-
tions are set as (qI1, qF1 )=( [ 5 0,900],[500,500]), (qI2, qF2 )=
([900,900],[500,500]), (qI3, qF3 ) = ([100,100],[500,500]),
(qI4, qF4 ) = ([800,1000],[500,500]) , respectively. The default
values of the other parameters are listed in Table II.
3) Benchmarks: This work evaluates the proposed TJCCT in
comparison with the following scheme

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

simulation results demonstrate that the pro-
posed TJCCT is able to achieve superior performances in terms
of the system utility, average processing rate, average completion
delay, average completion ratio, and average cost, while meeting
the energy constraints despite the trade-off of the increased energy
consumption.
Index Terms —Computation offloading, computing resource
allocation, trajectory control, UA V-assisted MEC network.
I. I NTRODUCTION
T
HE development of wireless communications and the pro-
liferation of mobile devices (MDs) triers various emerging
applications, such as autonomous driving, online gaming, and
augmented reality. These applications often require extensive
computing resources and low latency to satisfy the quality of ex-
perience (QoE). However, fulfilling the computation-intensive
and delay-sensitive computation tasks of these applications
poses a great challenge to MDs with insufficient computational
capability and finite energy capacity. To tackle this challenge,
mobile edge computing (MEC) has been identified as a promis-
ing technology to meet the stringent requirements of these
applications [1]. By offloading the computation-intensive tasks
to proximate MEC servers, the QoE of MDs can be significantly
enhanced in a cost-effective and energy-efficient way. However,
due to the dependence on terrestrial infrastructures and envi-
ronment, convention

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
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 部分 |
| 多业务QoS | 未提及 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=是；活动任务连续=未提及；迁移传输=部分 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=未提及；业务流量非平稳=未提及；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：2（PDF页）
- 方法窗口：13（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：15（PDF页）

### 关键词命中片段

- p3: dies designed a multi-timescale approach to decouple the problem into multi- timescale subproblems. For example, Shi et al. [13], [14] pro- posed a novel two-timescale resource optimization method for MEC to balance the service migration and task rerouting of MDs whenever handovers occur. Y ang et al. [15] studied the two-timescale online optimizat
- p17: rks: An optimal transport theory perspective,” IEEE Wireless Commun. Lett. , vol. 11, no. 1, pp. 150–154, Jan. 2022. [13] Y . Shi, C. Yi, B. Chen, C. Y ang, and J. Cai, “A two-timescale online optimization for balancing service migration and task rerouting in MEC,” in Proc. Proc. IEEE Glob. Commun. Conf. , 2023, pp. 5500–5505. [14] Y . Shi, C. Yi, 

## 12. 阅读闭合状态

- 全文文本：已提取（18页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
