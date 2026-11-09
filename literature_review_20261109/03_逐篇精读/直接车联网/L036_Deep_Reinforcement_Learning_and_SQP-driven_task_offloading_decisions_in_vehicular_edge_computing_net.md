# L036 — Deep Reinforcement Learning and SQP-driven task offloading decisions in vehicular edge computing networks

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：Ehzaz Mustafa
- 年份：2025
- 刊物/会议：Computer Networks
- DOI：10.1016/j.comnet.2025.111180
- PDF页数：13
- 原文：`C:\Users\tangning\Desktop\文献\02_移动性感知的服务迁移与传输资源协同\Deep Reinforcement Learning and SQP-driven task offloading decisions in vehicular edge computing networks.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=34, UAV=0
- 主研究方向：P1；次方向：无；依据：scores={'1': 32, '2': 2, '3': 2}; title=Deep Reinforcement Learning and SQP-driven task offloading decisions in vehicula

## 2. 研究场景与问题

### 摘要窗口

（PDF p5起）

system model considers the processing cost of a vehicle’s task 
as a weighted sum of the energy consumption and task execution delay. 
Let 𝑤𝑖,𝑗 be the weight factor for vehicle 𝑖. In our proposed work, there 
are two cases of task processing. First, the task is processed locally, and 
second, the task is offloaded to the corresponding RSU. In case of local 
execution, we denote 𝛼𝑖,𝑗 = 0, and 𝑡𝑣𝑖,𝑗 is the task processing latency for 
vehicle 𝑖, and 𝐸𝑣𝑖,𝑗 is the energy consumption. Hence, the total cost can 
be expressed as follows: 
𝑢(1)𝑖,𝑗 = 𝛼𝑖,𝑗
(
𝑡𝑣𝑖,𝑗 + 𝑤𝑖,𝑗 𝐸𝑣𝑖,𝑗
)
. (17)
In case of remote execution 𝛽𝑖,𝑗,𝑗 = 0 , the task processing delay for 
vehicle 𝑖 encompasses the vehicle to RSU task communication delay 
from vehicle 𝑖 to RSU 𝑗, denoted as 𝑡𝑣𝑟𝑖,𝑗 , the execution delay at the 
RSU 𝑗, denoted as 𝑡𝑟𝑖,𝑗,𝑗 , and the task result migration delay from RSU 
𝑗 to vehicle 𝑖, denoted as 𝑡𝑟𝑣𝑖

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p5起）

system model considers the processing cost of a vehicle’s task 
as a weighted sum of the energy consumption and task execution delay. 
Let 𝑤𝑖,𝑗 be the weight factor for vehicle 𝑖. In our proposed work, there 
are two cases of task processing. First, the task is processed locally, and 
second, the task is offloaded to the corresponding RSU. In case of local 
execution, we denote 𝛼𝑖,𝑗 = 0, and 𝑡𝑣𝑖,𝑗 is the task processing latency for 
vehicle 𝑖, and 𝐸𝑣𝑖,𝑗 is the energy consumption. Hence, the total cost can 
be expressed as follows: 
𝑢(1)𝑖,𝑗 = 𝛼𝑖,𝑗
(
𝑡𝑣𝑖,𝑗 + 𝑤𝑖,𝑗 𝐸𝑣𝑖,𝑗
)
. (17)
In case of remote execution 𝛽𝑖,𝑗,𝑗 = 0 , the task processing delay for 
vehicle 𝑖 encompasses the vehicle to RSU task communication delay 
from vehicle 𝑖 to RSU 𝑗, denoted as 𝑡𝑣𝑟𝑖,𝑗 , the execution delay at the 
RSU 𝑗, denoted as 𝑡𝑟𝑖,𝑗,𝑗 , and the task result migration delay from RSU 
𝑗 to vehicle 𝑖, denoted as 𝑡𝑟𝑣𝑖,𝑗 ,𝑗 .
The energy consumption, in this case, is represented by the energy 
consumed during the vehicle to RSU transmission process, 𝐸𝑣𝑟𝑖,𝑗 . Fur-
thermore, vehicle 𝑖 must be within the coverage area of RSU 𝑗 to receive 
the task result transmitted from RSU. Mathematically: 
𝛽𝑖,𝑗,𝑗
(
𝑡𝑣𝑟𝑖,𝑗 + 𝑡𝑟𝑖,𝑗,𝑗 + 𝑡𝑟𝑣𝑖,𝑗 ,𝑗
)
≤ 𝜒𝑖,𝑗 , 𝑖 ∈ 𝑁𝑗 , 𝑗 ∈ 𝑀. (18)
Hence, we can express the total task processing mathematically as: 
𝑢(2)𝑖,𝑗 = 𝛽𝑖,𝑗,𝑗
(
𝑡𝑣𝑟𝑖,𝑗 + 𝑡𝑟𝑖,𝑗,𝑗 + 𝑡𝑟𝑣𝑖,𝑗 ,𝑗 + 𝑤𝑖,𝑗 𝐸𝑣𝑟𝑖,𝑗
)
. (19)
3.3. Vehicular mobility model
We assume that vehicles may enter or leave the road and zones at 
any time except during the time interval 𝑡. In this time frame, the set 
of vehicles within the coverage area of RSU 𝑟𝑖 is denoted by 𝑣𝑖(𝑡). We 
consider that the duration of a time slot 𝑡 is short, so the number of 
vehicles within the coverage area of RSU 𝑟𝑖 during a one-time slot can 
be considered constant and is

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p6起）

Problem formulation
We aim to optimize the offloading decisions and resource allocation 
to minimize the total cost associated with the energy, latency, and task 
loss rate. Simply, the objective function is to minimize the average total 
cost over time 𝑇 , which includes energy consumption, latency, and task 
loss rate:
(𝑃 1) min
𝐗,𝐀
1
𝑇
𝑇 −1∑
𝑡=0
(𝛼𝐸(𝑡) + 𝛽𝐿(𝑡) + 𝛾𝑅(𝑡)) (21)
where 𝐗 = { 𝑥𝑣,𝑖(𝑡)} and 𝐀 = { 𝑎𝑣,𝑗 (𝑡)} represent the complete set 
of decision variables. 𝐸(𝑡) denotes the total energy consumption at 
time 𝑡. 𝐿(𝑡) denotes the total latency at time 𝑡. 𝑅(𝑡) denotes the total 
task loss rate at time 𝑡. 𝛼, 𝛽, 𝛾 are the weights associated with energy 
consumption, latency, and task loss rate, respectively. We use empirical 
evaluation to choose the importance and weights of these factors based 
on the system objectives. As we consider a general vehicular network 
model where minimizing energy consumption and latency is critical, 
we assign more weights to prioritize these effects and task loss rate is 
considered as a secondary factor. Our empirical weight configuration 
directly influences the optimization of offloading decisions resource 
allocation.
The optimization problem is subject to the following constraints: 
𝐶1 ∶ 𝑥𝑣,𝑖(𝑡) ∈ {0 , 1} ∀ 𝑣 ∈ 𝑉 , ∀𝑖 ∈ 𝐼, ∀𝑡 ∈ 𝑇 (22)
𝐶2 ∶ 0 ≤ 𝑎𝑣,𝑗 (𝑡) ≤ 1 ∀ 𝑣 ∈ 𝑉 , ∀𝑗 ∈ 𝐽 ,∀𝑡 ∈ 𝑇 (23)
𝐶3 ∶
∑
𝑗∈𝐽
𝑎𝑣,𝑗 (𝑡) = 1 ∀ 𝑣 ∈ 𝑉 , ∀𝑡 ∈ 𝑇 (24)
𝐶4 ∶ 𝐿𝑣,𝑖(𝑡) ≤ 𝐿max ∀𝑣 ∈ 𝑉 , ∀𝑖 ∈ 𝐼, ∀𝑡 ∈ 𝑇 (25)
𝐶5 ∶ 𝐸𝑣(𝑡) ≤ 𝐸max ∀𝑣 ∈ 𝑉 , ∀𝑡 ∈ 𝑇 (26)
𝐶 ∶ 6𝑅𝑣(𝑡) ≤ 𝑅max ∀𝑣 ∈ 𝑉 , ∀𝑡 ∈ 𝑇 (27)
𝐶7 ∶ 𝛽𝑖,𝑗,𝑗
(
𝑡𝑣𝑟𝑖,𝑗 + 𝑡𝑟𝑖,𝑗,𝑗 + 𝑡𝑟𝑣𝑖,𝑗 ,𝑗
)
≤ 𝜒𝑖,𝑗 , 𝑖 ∈ 𝑁𝑗 , 𝑗 ∈ 𝑀. (28)
where C1 ensures that the offloading decision 𝑥𝑣,𝑖(𝑡) must be binary. 
C2 ensures that the resource allocation decision 𝑎𝑣,𝑗 (𝑡) must be normal-
ized between 0 and 1. C3 ensures that the total resource a

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

proposed algorithm in terms of average 
delay, energy consumption, and task loss rate. For example. the proposed algorithm reduces the system cost 
by 25.1% compared to DQN and 16.67% compared to both DDQN and DDPG. Similarly. our method reduces 
the task loss rate by 37.06% compared to DQN, 34.78% compared to DDPG and 10.2% compared to DDQN.
1. Introduction
Mobile Edge Computing (MEC) and the Internet of Vehicles (IoV) 
are two rapidly evolving technologies that hold great potential for en-
hancing the performance and capabilities of autonomous vehicles [1,2]. 
MEC brings computational resources and storage closer to the network 
edge, enabling faster processing and reduced latency for various ap-
plications. On the other hand, IoV facilitates communication between 
vehicles and roadside infrastructure, enabling intelligent transportation 
systems and advanced vehicular services [3]. Integrating MEC with 
IoV forms Vehicular Edge Computing (VEC), a distributed computing 
paradigm designed to position computing resources and data storage 
∗ Corresponding authors.
E-mail addresses: aizazmustafa@cuiatd.edu.pk (E. Mustafa), Kashif.bilal@aasu.edu.kw (K. Bilal).
near vehicles, to minimize latency and improve response times [4,5]. 
This setup allows vehicles to communicate with each other and with 
roadside infrastructure, enabling various applications, including au-
tonomous vehicle

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p2起）

simulation results show that the proposed approach 
works in a distributed manner and performs similarly to the central-
ized one. In paper [24], a blockchain-empowered VEC framework is 
proposed to boost the computing power of vehicles by using V2V task 
offloading. The challenges of vehicles with high mobility and allocating 
resources among unfamiliar vehicles have been addressed. A DRL-
based computation offloading scheme combined with a smart contract 
running on the blockchain is proposed to enable secure and efficient 
task offloading. The practical Byzantine fault tolerance mechanism and 
a consensus algorithm for node selection are used to facilitate effi-
cient consensus-making. This encourages the base stations to enhance 
reliability in the execution of task allocation.
In [25], the authors present a new MEC platform utilizing multiple 
UAVs and DRL to enhance service quality and path planning in complex 
environments. The paper addresses environments with obstacles, the 
cooperation among several UAVs to maintain connectivity, and the 
challenges encountered by mobile servers. The main contributions of 
this research are threefold: first, it formulates a combined problem of 
quality of service optimization and path planning within a joint DRL 
framework; second, it uses sigmoid-like functions to predict user de-
mands to improve the quality of service; and third, i

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 部分 |
| 服务更新周期 | 部分 |
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
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=部分；服务更新周期=部分；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=是；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：5（PDF页）
- 问题窗口：6（PDF页）
- 方法窗口：1（PDF页）
- 仿真窗口：2（PDF页）
- 结论窗口：12（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（13页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
