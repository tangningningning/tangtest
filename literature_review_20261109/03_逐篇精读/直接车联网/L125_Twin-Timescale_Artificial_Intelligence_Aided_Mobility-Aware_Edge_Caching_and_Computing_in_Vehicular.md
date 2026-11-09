# L125 — Twin-Timescale Artificial Intelligence Aided Mobility-Aware Edge Caching and Computing in Vehicular Networks

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：待首页署名核验
- 年份：2019
- 刊物/会议：IEEE Transactions on Vehicular Technology
- DOI：10.1109/TVT.2019.2893898
- PDF页数：14
- 原文：`C:\Users\tangning\Desktop\文献\01_时延约束下服务部署与任务卸载的自适应多时间尺度协同\Twin-Timescale Artificial Intelligence Aided Mobility-Aware Edge Caching and Computing in Vehicular Networks.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=6, UAV=0
- 主研究方向：P1；次方向：2；依据：scores={'1': 18, '2': 12, '3': 3}; title=Twin-Timescale Artificial Intelligence Aided Mobility-Aware Edge Caching and Com

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model. Section IV briefly
presents the proposed framework of artificial intelligent-based
mobility-aware edge caching and computing for the resource
allocation. Then we describe the large timescale model relying
on PSO-based reward maximization in Section V and the small
timescale deep Q-learning in Section VI. Section VII presents
our performance results followed by our concluding remarks in
Section VIII.
II. R ELA TEDWORKS,B ACKGROUND, AND APPLICA TIONS
A. Related Works
V arious research problems and solutions have been consid-
ered in the edge caching and computing literature. The strategy
of caching popular contents at various local devices aims for
placing contents closer to mobile users with the assistance of
device-to-device (D2D) communications in HetNets, as studied
in [12], [13]. In [12], relay nodes with caching capability are
introduced to deliver the stored messages c

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model. Section IV briefly
presents the proposed framework of artificial intelligent-based
mobility-aware edge caching and computing for the resource
allocation. Then we describe the large timescale model relying
on PSO-based reward maximization in Section V and the small
timescale deep Q-learning in Section VI. Section VII presents
our performance results followed by our concluding remarks in
Section VIII.
II. R ELA TEDWORKS,B ACKGROUND, AND APPLICA TIONS
A. Related Works
V arious research problems and solutions have been consid-
ered in the edge caching and computing literature. The strategy
of caching popular contents at various local devices aims for
placing contents closer to mobile users with the assistance of
device-to-device (D2D) communications in HetNets, as studied
in [12], [13]. In [12], relay nodes with caching capability are
introduced to deliver the stored messages cooperatively with
the BS, yielding a low delay. In [13], Ji et al. considered the
combined effect of using both coding in the delivery phase,
achieving “coded multicast gain” and spatial reuse as a ben-
efit of local short-range D2D communication. Moreover, the
wireless cooperative caching policy constitutes one of the most
attractive caching network paradigms, where both the local user
terminals and the relay nodes can collaboratively store the mul-
timedia contents [12]–[16].
Small-cell caching, referred to as “Femtocaching” in [14],
[15], utilized SBSs in HetNets as distributed caching devices.
In [16], Li et al. analyzed a deterministic content placement,
where the caching placement strategy was optimized by ex-
ploiting both the knowledge of the node locations and of the
instantaneous wireless channels. Moreover, Gregori et al. [17]
investigated the case that both the relays and the

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

（PDF p4起）

optimization problem by employing traditional methods. We in-
voke both deep reinforcement learning [11] and the classic PSO
[10], which can efficiently determine a near-optimal action, de-
spite searching through only a small fraction of the potentially
excessive total search space at a much reduced complexity.
In this paper, a twin-timescale framework is developed based
on [9]. To improve the performance, we consider not only the
resource requirements of communications, storage and com-
puting but also the hard deadline for every requested content.
So we integrate the resource consumption and the hard dead-
line into a twin-timescale optimization framework. The large
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 26,2026 at 12:05:26 UTC from IEEE Xplore. Restrictions apply. 
3090 IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 68, NO. 4, APRIL 2019
Fig. 2. Graphical illustration of multi-timescale model. The large timescale is
denoted as tl , while the small timescale is denoted as ts .
timescale model is developed at epoch level (corresponding to
Td time slots), while the small timescale model is at the time
slot level. Similar to [9], the large timescale model receives the
requested content and computes the corresponding tasks within
each epoch, while the small timescale model performs actions
based on the raw states. In particular, the large timescale model
provides resources to guarantee that the vehicle receives a suf-
ficiently high number of Fountain-coded segments and offloads
all the tasks within Td . To do so, the system selects the ap-
propriate sets of potential RSUs eligible for caching as well as
the sets of MEC servers suitable for computing. However, the
algorithms proposed in [9] do not scale well,

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p2起）

proposed algorithms by using the optimal
parameter configuration found for caching, for comput-
ing and for vehicular mobility. The impact of the user-
mobility, data size, RSUs’ caching storage, backhaul ca-
pacities and cloud resources on the system performance
is also studied.
The outline of this paper is as follows. In Section II-A, we
discuss the related literature, background and potential applica-
tions. Section III describes our system model. Section IV briefly
presents the proposed framework of artificial intelligent-based
mobility-aware edge caching and computing for the resource
allocation. Then we describe the large timescale model relying
on PSO-based reward maximization in Section V and the small
timescale deep Q-learning in Section VI. Section VII presents
our performance results followed by our concluding remarks in
Section VIII.
II. R ELA TEDWORKS,B ACKGROUND, AND APPLICA TIONS
A. Related Works
V arious research problems and solutions have been consid-
ered in the edge caching and computing literature. The strategy
of caching popular contents at various local devices aims for
placing contents closer to mobile users with the assistance of
device-to-device (D2D) communications in HetNets, as studied
in [12], [13]. In [12], relay nodes with caching capability are
introduced to deliver the stored messages cooperatively with
the BS, yielding a low delay. In [13], Ji

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p10起）

simulation results validate
the theoretical findings. More detailed results have to be omitted
here owing to the space limit, but motivated readers can find the
details in [41].
Fig. 3 shows the success probability versus the coded
packet size lc for λTd = 0.1. Here, “ success” means that the
tagged vehicle i completes downloading the requested content
and offloads its corresponding tasks for computation within
the hard deadline Td without assistance from the BS. We
compare our proposed scheme against two benchmark cases:
Authorized licensed use limited to: Indian Institute of Petroleum and Energy. Downloaded on October 26,2026 at 12:05:26 UTC from IEEE Xplore. Restrictions apply. 
3096 IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, VOL. 68, NO. 4, APRIL 2019
Fig. 3. Probability of success vs coded packet size lc for λTd = 0.1.
1) random resource allocation; and 2) equal resource alloca-
tion. Similar to our proposed scheme, both the random resource
allocation and the equal resource allocation select the sets of
eligible RSUs at every epoch in order to reduce the number of
reserved RSUs for reducing the cost of their caching storage.
However, they pre-select these sets in a random manner. In ev-
ery time slot, the random resource allocation allocates caching
contents and computational resources, while the equal resource
allocation performs caching and computing allocation equally.

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
| 迁移传输 | 是 |
| 多业务QoS | 是 |
| 业务流量非平稳 | 未提及 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=是；服务更新周期=是；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=未提及；活动任务连续=未提及；迁移传输=是 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=是；业务流量非平稳=未提及；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：4（PDF页）
- 方法窗口：2（PDF页）
- 仿真窗口：10（PDF页）
- 结论窗口：None（PDF页）

### 关键词命中片段

- 未定位到明确片段

## 12. 阅读闭合状态

- 全文文本：已提取（14页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
