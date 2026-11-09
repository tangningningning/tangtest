# L016 — A joint optimization of resource allocation management and multi-task offloading in high-mobility vehicular multi-access edge computing networks

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：Hong Min
- 年份：2025
- 刊物/会议：Ad Hoc Networks
- DOI：10.1016/j.adhoc.2024.103656
- PDF页数：14
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\A joint optimization of resource allocation management and multi-task offloading in high-mobility vehicular multi-access edge computing networks.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=19, UAV=3
- 主研究方向：P1；次方向：2；依据：scores={'1': 36, '2': 21, '3': 14}; title=A joint optimization of resource allocation management and multi-task offloading

## 2. 研究场景与问题

### 摘要窗口

未定位到章节；需回原文核查。

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

未定位到章节；需回原文核查。

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p3起）

optimization problem considers UAV flight time, 3D
trajectory, TD binary offloading decisions, and time allocation to TDs.
Combining deep reinforcement learning and linear programming, the
proposed method optimizes offloading decisions and UAV trajectory,
outperformingexistingcounterparts.UnlikeNaLinetal.’swork,which
concentratesonUAVenergyefficiencyandtrajectoryoptimization,our
study addresses the challenges posed by vehicular networks’ dynamic
andmobilenature,focusingonminimizingenergy,computational,and
communication.
In another study, Khattak et al. [38] presented a framework that
utilizesVFCforhealthcareIoTdevices,emphasizingtimesensitivityand
resource efficiency. The framework incorporates reputation-based pri-
oritization and resource allocation methods. By integrating subjective
logic and perceived reputation-based trust, their approach reduces bia-
ses in trust and defends against attacks. The researchers identified and
eliminated malicious nodes, aiming to maximize utility through
individual, real-time, and incremental perceptions. Pham et al. [39]
addressed a combined optimization challenge involving partial off-
loadingandresourceallocationwithinaPVMECframeworktoimprove
task offloading for mobile devices. They defined this problem as a
mixed-integernonlinearprogrammingchallengetomaximizeoffloading
benefits while minimizing latency and resource expenses. Hasan et al.
[40] proposed a system to address two main issues in the current off-
loading system: consistent cloud computing access and security and
privacyconcerns.Thestudyhighlightsthechallengesofaccessingcloud
services across different operating systems and hardware and the
importance of maintaining security and privacy during data offloading
and resource management.
Liao et al. in [41] introduced a TSOF b

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p6起）

proposed algorithm extends the DDQN by integrating several
featuresthataddressvehicularnetworks’dynamicandcomplexnature.
3.5.3. Staterepresentation
The system’s state is represented by a vector that encapsulates in-
formation about the current tasks, resource availability, network con-
ditions,andcontextualfactorsatthetime.Thestatevector(st)includes
task characteristics Sij, Cij, Tdeadlineij , Pij, resource availability Ravailj ,
network conditions Li,j, B, Di,j, contextual information vvelocityi , ρtraffic,
upreferencesi , and edge server loadLj.
The state vector expressed as Eq. (23):
3.5.4. Actionspace
TheactionspaceAconsistsoftheproposedmodel’spossibleactions
tooptimizethesystem’sperformance.Theseactionsincludeselectingan
edge server ej for offloading task τij and adjusting resource allocation
dynamically based on current network conditions and task demands.
The actions (at) are represented as Eq. (24):
at =
[
πij,ACPUij ,AMemij
]
(24)
Where πij indicatestheoffloadingdecision,ACPUij andAMemij representthe
st =
[{
Sij,Cij,Tdeadlineij ,Pij
}
,
{
Ravailj
}
,
{
Li,j,B,Di,j
}
,
{
vvelocityi , ρtraffic,upreferencesi
}
,
{
Lj
}]
(23)
H.Minetal. Ad Hoc Networks 166 (2025) 103656 
6 
allocated CPU and memory resources, respectively. These allocations
are given as absolute values, specifying the exact amount of CPU and
memory resources assigned to each task. The action space is varia

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

simulation demonstrate optimized performance,
enhancing efficiency in vehicular MEC networks compared to baseline and the other well-known algorithms.
1. Introduction
TheInternetofThings(IoT)ischanginghowvehiclescommunicate,
making transportation systems more innovative and efficient [1-3].
IoT-enabled vehicular communication allows cars to share data with
infrastructure like traffic lights and road sensors [4,5]. This helps
improvetrafficflow,enhancesafety,andprovidereal-timeinformation
to drivers [6,7]. For example, vehicles can send alerts about road con-
ditions,accidents,ortrafficjamstonearbycars.Thesecapabilitiesmake
driving safer and reduce travel time and fuel consumption [8,9]. How-
ever, as more vehicles connect, communication management becomes
morecriticaltoensureasmoothandreliablesystem[10,11].Toaddress
thisproblem,vehicularMobileEdgecomputation(MEC)isemployed,as
depicted in Fig. 1. This approach brings computation and data storage
near cars, reducing latency and enhancing real-time application per-
formance[12].Nevertheless,MECalsohasseveralchallenges.Onemain
issue is the dynamic and mobile nature of vehicular networks, making
maintaining consistent connections complex because of frequent hand-
overs and changing signal strengths [13,14]. Another challenge is effi-
ciently allocating computational resources among many vehicles with
different demands since vehicles

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
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 部分 |
| 信道动态变化 | 未提及 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=部分；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：None（PDF页）
- 问题窗口：3（PDF页）
- 方法窗口：6（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：12（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（14页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
