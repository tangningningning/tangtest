# L106 — Resilient Edge Service Placement Under Demand and Node Failure Uncertainties

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2024
- 刊物/会议：IEEE Transactions on Network and Service Management
- DOI：10.1109/TNSM.2023.3290137
- PDF页数：16
- 原文：`C:\Users\tangning\Desktop\文献\Resilient_Edge_Service_Placement_Under_Demand_and_Node_Failure_Uncertainties.pdf`
- 领域分类：**C**；依据：摘要/模型段 vehicle=0, UAV=0, MEC=2
- 主研究方向：P1；次方向：无；依据：scores={'1': 21, '2': 0, '3': 0}; title=Resilient Edge Service Placement Under Demand and Node Failure Uncertainties

## 2. 研究场景与问题

### 摘要窗口

（PDF p3起）

system model and problem formulation. The
solution approaches are introduced in Section III, followed
by the numerical results in Section IV. Section V discusses
related work. Finally, the conclusions are given in Section VI.
II. S YSTEM MODEL AND PROBLEM FORMULATION
A. System Model
In this paper, we investigate the edge resource procurement
and management problem for an SP offering a delay-sensitive
Fig. 2. System model.
service, such as AR/VR and cloud gaming. The SP does not
own any ENs but instead procures computing resources from
an EC market comprising numerous geographically distributed
heterogeneous ENs. The SP has subscribers located in differ-
ent areas, where user requests in each area are aggregated at
an access point (AP). To reduce network delay and enhance
user experience, the SP can place its services directly on the
ENs. The major concerns for the SP are TO determine whe

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p3起）

system model and problem formulation. The
solution approaches are introduced in Section III, followed
by the numerical results in Section IV. Section V discusses
related work. Finally, the conclusions are given in Section VI.
II. S YSTEM MODEL AND PROBLEM FORMULATION
A. System Model
In this paper, we investigate the edge resource procurement
and management problem for an SP offering a delay-sensitive
Fig. 2. System model.
service, such as AR/VR and cloud gaming. The SP does not
own any ENs but instead procures computing resources from
an EC market comprising numerous geographically distributed
heterogeneous ENs. The SP has subscribers located in differ-
ent areas, where user requests in each area are aggregated at
an access point (AP). To reduce network delay and enhance
user experience, the SP can place its services directly on the
ENs. The major concerns for the SP are TO determine where
to install its service and how much computing resources to
procure from each EN. Once the edge resources are procured,
the SP must optimally allocate the workload from different
A P st ot h eE N s .T h eS Pi sa s s u m e dt ob eap r i c e - t a k e rw i t h
a budget B for edge resource procurement.
The system model is shown in Fig. 2, which depicts five
areas and four ENs. The service is installed on EN1, EN2, and
EN4. In the absence of failures, user requests in areas 4 and 5
can be serviced by EN4, which is the closest EN. However, if
EN4 fails, the workload from these areas can be reassigned to
another EN, such as EN2, which also has the service installed.
Let i be the area index and I be the set of areas, each of
which is represented by an AP. The resource demand in area
i is λi . Define j and J as the EN index and the set of ENs,
respectively. The resource capacity of EN j is Cj

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p3起）

problem formulation. The
solution approaches are introduced in Section III, followed
by the numerical results in Section IV. Section V discusses
related work. Finally, the conclusions are given in Section VI.
II. S YSTEM MODEL AND PROBLEM FORMULATION
A. System Model
In this paper, we investigate the edge resource procurement
and management problem for an SP offering a delay-sensitive
Fig. 2. System model.
service, such as AR/VR and cloud gaming. The SP does not
own any ENs but instead procures computing resources from
an EC market comprising numerous geographically distributed
heterogeneous ENs. The SP has subscribers located in differ-
ent areas, where user requests in each area are aggregated at
an access point (AP). To reduce network delay and enhance
user experience, the SP can place its services directly on the
ENs. The major concerns for the SP are TO determine where
to install its service and how much computing resources to
procure from each EN. Once the edge resources are procured,
the SP must optimally allocate the workload from different
A P st ot h eE N s .T h eS Pi sa s s u m e dt ob eap r i c e - t a k e rw i t h
a budget B for edge resource procurement.
The system model is shown in Fig. 2, which depicts five
areas and four ENs. The service is installed on EN1, EN2, and
EN4. In the absence of failures, user requests in areas 4 and 5
can be serviced by EN4, which is the closest EN. However, if
EN4 fails, the workload from these areas can be reassigned to
another EN, such as EN2, which also has the service installed.
Let i be the area index and I be the set of areas, each of
which is represented by an AP. The resource demand in area
i is λi . Define j and J as the EN index and the set of ENs,
respectively. The resource capacity of EN j is Cj .A l s o ,pj is
t

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p1起）

solution within a finite number of
iterations. However, the running time of this iterative algorithm
heavily depends on the uncertainty set. Therefore, we further
present an affine decisions rule approximation approach, which
exhibits greater insensitivity to the uncertainty set, to tackle the
underlying problem. Extensive numerical results demonstrate the
advantages of the proposed model and approaches, which can
help the SP make proactive decisions to mitigate the impacts of
the uncertainties.
Index Terms—Resilient edge computing, robustness, resiliency,
adaptive robust optimization, node failures, service placement.
I. I NTRODUCTION
E
DGE computing (EC) has emerged as a crucial com-
puting paradigm that complements the cloud to meet
the stringent requirements of modern applications such as
augmented/virtual reality (AR/VR), autonomous driving, and
manufacturing automation. By locating cloud resources closer
to users, devices, and sensors, EC significantly reduces
network traffic, improves user experience, and enables the
deployment of various low-latency and high-reliability applica-
tions. The new network architecture has an EC layer positioned
between the cloud and the end devices, as shown in Fig. 1.
Manuscript received 6 December 2022; revised 15 April 2023 and 18 June
2023; accepted 19 June 2023. Date of publication 28 June 2023; date of
current version 7 February 2024.

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p3起）

Simulation: Extensive numerical results demonstrate the
superior out-of-sample performance of the proposed ARO
model compared to several benchmarks, including the
heuristic, deterministic, and stochastic models. We also
illustrate the benefits of considering both the demand
and failure uncertainties. Furthermore, our experiments
show that ADR can achieve comparable performance
with respect to the exact CCG-based methods while
requiring significantly less computational time. Finally,
sensitivity analyses are conducted to examine the impacts
of important system parameters on the optimal Solution.
The rest of the paper is organized as follows. In Section II,
we present the system model and problem formulation. The
solution approaches are introduced in Section III, followed
by the numerical results in Section IV. Section V discusses
related work. Finally, the conclusions are given in Section VI.
II. S YSTEM MODEL AND PROBLEM FORMULATION
A. System Model
In this paper, we investigate the edge resource procurement
and management problem for an SP offering a delay-sensitive
Fig. 2. System model.
service, such as AR/VR and cloud gaming. The SP does not
own any ENs but instead procures computing resources from
an EC market comprising numerous geographically distributed
heterogeneous ENs. The SP has subscribers located in differ-
ent areas, where user requests in each area are aggregated

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 部分 |
| 服务更新周期 | 部分 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 部分 |
| 服务迁移 | 未提及 |
| 活动任务连续执行 | 未提及 |
| 迁移传输 | 未提及 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 未提及 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=部分；服务更新周期=部分；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=部分；服务迁移=未提及；活动任务连续=未提及；迁移传输=未提及 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=未提及 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：3（PDF页）
- 问题窗口：3（PDF页）
- 方法窗口：1（PDF页）
- 仿真窗口：3（PDF页）
- 结论窗口：3（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（16页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
