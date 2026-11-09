# L095 — Mobility-aware service migration and two-timescale task offloading for distributed edge computing in open RAN

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2026
- 刊物/会议：待核
- DOI：10.1038/s41598-026-73893-x
- PDF页数：28
- 原文：`C:\Users\tangning\Desktop\文献\02_移动性感知的服务迁移与传输资源协同\Mobility-aware service migration and two-timescale task offloading for distributed edge computing in open RAN.pdf`
- 领域分类：**C**；依据：摘要/模型段 MEC=8, vehicle=1, UAV=0
- 主研究方向：P2；次方向：无；依据：scores={'1': 47, '2': 125, '3': 18}; title=Mobility-aware service migration and two-timescale task offloading for distribut

## 2. 研究场景与问题

### 摘要窗口

（PDF p3起）

system model and stochastic opti -
mization formulation are established to jointly represent user association, handover events, execution-site 
selection, service migration, radio allocation, computing allocation, and queue evolution. The end-to-end 
service model explicitly accounts for migration and handover overhead in addition to communication and 
computation delays, allowing service-continuity requirements to directly influence orchestration decisions.
• Mobility-aware graph-contextual orchestration : A Twin Delayed Deep Deterministic Policy Gradient 
(TD3)-based mechanism is introduced to capture both resource coupling and mobility transitions among 
neighboring edge regions. Regional representations incorporate congestion, available radio and computing 
capacity, service performance, and mobility-flow information. These representations are combined with local 
observations and lo

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p3起）

system model and stochastic opti -
mization formulation are established to jointly represent user association, handover events, execution-site 
selection, service migration, radio allocation, computing allocation, and queue evolution. The end-to-end 
service model explicitly accounts for migration and handover overhead in addition to communication and 
computation delays, allowing service-continuity requirements to directly influence orchestration decisions.
• Mobility-aware graph-contextual orchestration : A Twin Delayed Deep Deterministic Policy Gradient 
(TD3)-based mechanism is introduced to capture both resource coupling and mobility transitions among 
neighboring edge regions. Regional representations incorporate congestion, available radio and computing 
capacity, service performance, and mobility-flow information. These representations are combined with local 
observations and long-timescale guidance to support decentralized handover, migration, and offloading de-
cisions through a hybrid graph-enhanced policy.
• Mobility- and service-continuity-oriented evaluation : The proposed method is evaluated under dynamic 
multi-region mobility and heterogeneous service demand using performance indicators that characterize not 
only delay, resource efficiency, and queue behavior, but also service continuity, migration overhead, and hand-
over-induced service degradation. The evaluation is designed to isolate the contribution of mobility aware -
ness, service migration, and the two-timescale control structure through comparative and ablation-based 
benchmarks.
Paper organization
The remainder of this paper is organized as follows. Section II reviews related studies on edge task offloading, 
mobility management, service migration, and Open-RAN-enabled orchestration and ide

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p5起）

problem formulation
This section develops the system model for mobility-aware service migration and task offloading in distributed 
edge-enabled Open RAN. The formulation distinguishes two control horizons: longer-horizon mobility and 
resource planning at the Non-Real-Time RAN Intelligent Controller (Non-RT RIC), and short-timescale 
execution at Near-Real-Time RIC xApps. In addition to radio conditions, computation load, and queue dynamics, 
the model explicitly describes user movement, radio handovers, the location of active service state, and the cost of 
migrating that state between edge regions. These elements are integrated into the communication–computation 
latency model and the resulting stochastic orchestration problem.
Scientific Reports | (2026) 16:33688 6| https://doi.org/10.1038/s41598-026-73893-x
www.nature.com/scientificreports
O-RAN two-timescale architecture and mobility
Consider a distributed Open RAN edge network consisting of a set of service regions, where each region 
represents a localized radio–compute domain that jointly supports wireless access, edge processing, and 
mobility-aware service orchestration.
 ℳ= {1, … , 𝑀𝑀 },
where each region 𝑚𝑚 ∈ ℳ contains an O-RAN radio-access domain, represented by its O-DU/O-CU functions, 
together with an edge server of maximum radio bandwidth 𝐵𝐵max𝑚𝑚 and processing capacity 𝐹𝐹max𝑚𝑚 . The Non-
RT RIC coordinates the regions over relatively long planning windows, while the Near-RT RIC hosts an xApp 
responsible for short-timescale mobility and edge-resource decisions.
Time is divided into slots of duration Δ . A long planning window 𝑛𝑛 contains 𝐾𝐾 consecutive short slots, 
denoted by 𝒯𝒯𝑛𝑛. The Non-RT RIC updates network-wide planning information once per window, whereas the 
xApps execute runtime control at

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p23起）

proposed scheme preserves continuity 
by about 5.5, 9.3, 12.4, and 19.6% points over DRAC-SM, MT-RO, MA-EECOC, and GatMARL, respectively. 
This widening gap is significant because it shows that the benefit of the proposed mechanism is not limited to 
ordinary low-mobility operation; rather, its advantage becomes stronger precisely when handover and migration 
coordination becomes most critical.
Fig. 8. Average mobility-induced service interruption versus average user speed.
 
Scientific Reports | (2026) 16:33688 24| https://doi.org/10.1038/s41598-026-73893-x
www.nature.com/scientificreports
The superior continuity of MA-TT-O-RAN follows directly from how mobility-induced interruption is 
represented in both the system model and the learning objective. In the proposed formulation, continuity is 
not treated indirectly through end-to-end delay alone. Instead, the interruption term 𝐷𝐷Int𝑖𝑖 =𝐷𝐷 HO𝑖𝑖 + 𝜅𝜅 blk𝑖𝑖 𝐷𝐷Mig𝑖𝑖 
explicitly combines radio-handover interruption with the blocking portion of service migration, and a service 
is regarded as continuous only when this quantity remains within the class-specific interruption tolerance. This 
condition is then reinforced by the continuity indicator 𝑐𝑐cont𝑖𝑖 (𝑡𝑡) and the corresponding positive term in the 
reward function. Consequently, the actor learns to avoid actions that provide a short-term radio or compute 
advantage but cause ex

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p3起）

simulation methodology and evaluates the framework under different traffic and mobility 
conditions. Finally, Section VI concludes the paper and outlines relevant directions for further investigation.
Related works
Task offloading and edge orchestration
Computation offloading has become a fundamental mechanism for supporting resource-intensive and latency-
sensitive applications in mobile edge computing systems. The central objective is to determine an appropriate 
execution location for each task while allocating the communication and computation resources required to 
satisfy application-level performance requirements. Earlier approaches commonly formulated this problem 
using mathematical optimization, decomposition, matching, or Lyapunov-based methods, whereas more recent 
studies increasingly employ deep reinforcement learning to adapt decisions to stochastic traffic, wireless-channel 
variations, and time-varying resource availability10–17,28–33. Joint treatment of offloading and resource allocation 
is particularly important because transmission and processing decisions are inherently coupled: assigning a 
task to a remote server may reduce local computation pressure while simultaneously increasing wireless and 
transport latency.
Learning-based methods have further extended this formulation to distributed and multi-edge environments. 
For example, distributed deep reinf

**待核：** 场景、节点/车辆数、任务/通信/计算/部署参数、基线、指标、主要数值结果。当前不摘录具体数值，避免把提取文本中的图表碎片误当成作者结论。

## 7. 作者明确指出的局限

未逐段核对。需在全文精读时检索 `future work / limitation / future research / conclusion` 并区分作者原话与阅读者判断。当前不作转述。

## 8. 阅读者发现的潜在局限（筛查级，待全文确认）

根据关键词命中，仅提示需要核查的方向，不构成结论：

| 任务截止期限 | 是 |
| 服务更新周期 | 是 |
| 决策时间尺度自适应 | 未提及 |
| 车辆移动性 | 是 |
| 位置预测误差 | 部分 |
| 服务迁移 | 是 |
| 活动任务连续执行 | 是 |
| 迁移传输 | 是 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 是 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=部分；服务迁移=是；活动任务连续=是；迁移传输=是 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=是；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：3（PDF页）
- 问题窗口：5（PDF页）
- 方法窗口：23（PDF页）
- 仿真窗口：3（PDF页）
- 结论窗口：25（PDF页）

### 关键词命中片段

- p1: lectrical, Computer, and Biomedical Engineering, Toronto Metropolitan University, Toronto, ON M5B 2K3, Canada. 5Karunya Institute of Technology and Sciences, Coimbatore, India. email: a.mirzaei@iau.ac.ir Mobility-aware service migration and two-timescale task offloading for distributed edge computing in open RAN Amin Mohajer1, Abbas Mirzaei2 , Baba
- p1: addam2, Ali Ghaffari3 & Xavier Fernando4,5 This paper proposes a mobility-aware two-timescale orchestration framework for distributed edge computing in Open Radio Access Networks, where user mobility, handover events, service migration, task offloading, and radio–compute allocation are jointly considered. The framework exploits the functional separ

## 12. 阅读闭合状态

- 全文文本：已提取（28页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
