# L059 — Event-Triggered Reinforcement Learning Based Joint Resource Allocation for Ultra-Reliable Low-Latency V2X Communications

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2024
- 刊物/会议：IEEE Transactions on Vehicular Technology
- DOI：10.1109/TVT.2024.3424398
- PDF页数：16
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\Event-Triggered Reinforcement Learning Based Joint Resource Allocation for Ultra-Reliable Low-Latency V2X Communications.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=27, UAV=0
- 主研究方向：P3；次方向：无；依据：scores={'1': 0, '2': 0, '3': 14}; title=Event-Triggered Reinforcement Learning Based Joint Resource Allocation for Ultra

## 2. 研究场景与问题

### 摘要窗口

（PDF p3起）

system model and assumptions
Fig. 1. System model for the V2X communication network.
used in the paper. SectionIII presents the mathematical formula-
tion of the worst-case decoding-error probability minimization
problem for the vehicular URLLC system. Section IV describes
the joint-convexity analysis and the optimization theory based
solution methodology for the transmit power and block length
optimization problem. Section V provides a brief background
on the design of the DRL algorithms and describes the proposed
event-triggered DRL-based algorithm for the transmit power and
block length allocation problem. Section VI evaluates the per-
formance of the proposed solution strategy. Finally, conclusions
and future research directions are presented in Section VII.
II. S YSTEM MODEL AND ASSUMPTIONS
We consider a single-cell V2I URLLC system, where vehicles
move on a multi-lane highway with

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p3起）

system model and assumptions
Fig. 1. System model for the V2X communication network.
used in the paper. SectionIII presents the mathematical formula-
tion of the worst-case decoding-error probability minimization
problem for the vehicular URLLC system. Section IV describes
the joint-convexity analysis and the optimization theory based
solution methodology for the transmit power and block length
optimization problem. Section V provides a brief background
on the design of the DRL algorithms and describes the proposed
event-triggered DRL-based algorithm for the transmit power and
block length allocation problem. Section VI evaluates the per-
formance of the proposed solution strategy. Finally, conclusions
and future research directions are presented in Section VII.
II. S YSTEM MODEL AND ASSUMPTIONS
We consider a single-cell V2I URLLC system, where vehicles
move on a multi-lane highway with the RSU at the center and a
fixed distance from the highway, as shown in Fig.1. The roadside
unit (RSU) plays the role of the 5G base station (gNB) for
centralized radio resource management [24] and sends unicast
safety-critical information to the vehicles for coordination and
safety alerts using short packets. We consider a practical scenario
where this RSU is deployed in a rural area and deprived of
a permanent grid source. We assume the RSU is equipped
with large batteries that are periodically recharged using energy
harvesting techniques (wind/solar) [29]. We assume that the
RSU and the vehicle user equipment (VUEs) are equipped
with a single transmit-receive antenna. The set of VUEs under
the coverage of the RSU is denoted by K = {1,...,K }.T h i s
system model represents different URLLC V2X scenarios, such
as time-sensitive High-Definition (HD) map data dissemination
for vehicle co

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p1起）

optimization problem. Incorporating event-triggered learning into
the DRL framework enables assessing whether to initiate the DRL
process, thereby reducing the number of DRL process executions
while maintaining reasonable reliability performance. The DRL
framework consists of a two-layered structure. In the first layer,
multiple deep Q-networks (DQNs) are established at the central
trainer for block length optimization. The second layer involves
an actor-critic network and utilizes the deep deterministic policy-
gradient (DDPG)-based algorithm to optimize the power allocation.
Simulation results demonstrate that the proposed event-triggered
DRL scheme can achieve 95 % of the performance of the joint
optimization scheme while reducing the DRL executions by up to
24% for different network settings.
Index Terms —Deep reinforcement learning (DRL), event-
triggered learning, finite block length transmission, 6G networks,
ultra-reliable and low-latency communications (URLLC),
V ehicular networks, vehicle-to-everything (V2X) communication.
Manuscript received 24 October 2023; revised 22 April 2024 and 5 June
2024; accepted 2 July 2024. Date of publication 11 July 2024; date of current
version 7 November 2024. This work was supported in part by Scientific and
Technological Research Council of Turkey under Grant 119C058 and in part
by Ford Otosan. The review of this article was coordinated by Prof. Li Wang.
(Corresponding author: Sinem Coleri.)
The authors are with the Department of Electrical and Electronics Engi-
neering, Koc University, 34450 Istanbul, Türkiye (e-mail: nkhan20@ku.edu.tr;
scoleri@ku.edu.tr).
Digital Object Identifier 10.1109/TVT.2024.3424398
I. I NTRODUCTION
V
EHICLE-TO-EVERYTHING (V2X) networks have re-
cently attracted a lot of attention as a crucial enabler

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p11起）

proposed algorithm
against several schemes by simulating a vehicular network with
multiple VUEs. In particular, we follow the simulation setup for
a freeway model described in 3GPP TR 36.885, which describes
TABLE I
SIMULA TIONPARAMETERS
TABLE II
HYPERPARAMETERS FOR ACTOR NETWORK
TABLE III
HYPERPARAMETERS FOR CRITIC NETWORK AND DEEP-Q NETWORK
in detail the vehicular channel models, vehicle mobility, and
vehicular data traffic [43]. In particular, N V2I links are initi-
ated by N vehicles and the K V2V links are formed between
each vehicle with its surrounding neighbors. The vehicles are
distributed according to Poisson distribution on a multi-lane
highway with the BS at the center and at a fixed distance from
the highway. The considered highway is a 3× 2 lane with a fixed
lane width. The vehicle density on the highway is determined
by the vehicle speedvs, and the average inter-vehicular distance
is 2.5× vs, which is considered fixed in our simulation results.
The large-scale fading is modeled using the WINNER path-loss
modelPL(d)= 128.1+ 37.6log10(d), where d is the distance
in [km] of the VUE from the RSU [44]. Fast fading has been
modeled using Rayleigh fading with the time correlation be-
tween successive time frames, as represented by Jake’s model.
The important simulation parameters are provided in Table I.
The hyperparameter values for the actor DNN, the critic DNN,
and t

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p1起）

Simulation results demonstrate that the proposed event-triggered
DRL scheme can achieve 95 % of the performance of the joint
optimization scheme while reducing the DRL executions by up to
24% for different network settings.
Index Terms —Deep reinforcement learning (DRL), event-
triggered learning, finite block length transmission, 6G networks,
ultra-reliable and low-latency communications (URLLC),
V ehicular networks, vehicle-to-everything (V2X) communication.
Manuscript received 24 October 2023; revised 22 April 2024 and 5 June
2024; accepted 2 July 2024. Date of publication 11 July 2024; date of current
version 7 November 2024. This work was supported in part by Scientific and
Technological Research Council of Turkey under Grant 119C058 and in part
by Ford Otosan. The review of this article was coordinated by Prof. Li Wang.
(Corresponding author: Sinem Coleri.)
The authors are with the Department of Electrical and Electronics Engi-
neering, Koc University, 34450 Istanbul, Türkiye (e-mail: nkhan20@ku.edu.tr;
scoleri@ku.edu.tr).
Digital Object Identifier 10.1109/TVT.2024.3424398
I. I NTRODUCTION
V
EHICLE-TO-EVERYTHING (V2X) networks have re-
cently attracted a lot of attention as a crucial enabler for
intelligent transportation systems (ITS) by enhancing traffic
safety, improving vehicle coordination, and enabling novel
transportation features such as platooning, advanced real-

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 是 |
| 服务更新周期 | 是 |
| 决策时间尺度自适应 | 是 |
| 车辆移动性 | 是 |
| 位置预测误差 | 未提及 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=是；时间尺度自适应=是 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：3（PDF页）
- 问题窗口：1（PDF页）
- 方法窗口：11（PDF页）
- 仿真窗口：1（PDF页）
- 结论窗口：3（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（16页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
