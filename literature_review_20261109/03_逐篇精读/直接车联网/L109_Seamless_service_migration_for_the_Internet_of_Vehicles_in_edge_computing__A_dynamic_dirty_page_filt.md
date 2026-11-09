# L109 — Seamless service migration for the Internet of Vehicles in edge computing: A dynamic dirty page filtering and two-stages compression technique

> **阅读状态：筛查稿，不是全文精读完成。** 本文件基于本地PDF全文文本提取，完成章节定位、关键词命中、分类和证据片段摘录；尚未逐页逐式核对全部公式、图表与附录。凡标“未核实”的内容不得在论文中当作已证实结论。

## 1. 准确文献信息

- 作者：Kaifeng Hua
- 年份：2026
- 刊物/会议：Journal of Network and Computer Applications
- DOI：10.1016/j.jnca.2025.104412
- PDF页数：15
- 原文：`C:\Users\tangning\Desktop\文献\02_移动性感知的服务迁移与传输资源协同\Seamless service migration for the Internet of Vehicles in edge computing - A dynamic dirty page filtering and two-stages compression technique.pdf`
- 领域分类：**A**；依据：摘要/模型段 vehicle=5, UAV=0, MEC=2
- 主研究方向：P2；次方向：无；依据：scores={'1': 0, '2': 52, '3': 16}; title=Seamless service migration for the Internet of Vehicles in edge computing: A dyn

## 2. 研究场景与问题

### 摘要窗口

（PDF p2起）

system model of the service migration technique 
in detail, and the proposed algorithm, IACMT, is elaborated in Section 4. 
Experimental results and performance evaluation are shown in Section 
5. Finally, conclusions and future work are presented in Section 6.
2. Related work
In recent years, research on container service migration techniques 
has focused on the balance between migration efficiency and resource 
utilization. Migration efficiency is usually based on service interruption 
time (i.e., downtime) and data transfer latency as key indicators, 
including the container image, the transfer length of in-memory data, 
and the synchronization latency between the source node and the target 
node. Resource utilization, on the other hand, is reflected in the con-
sumption of compute resources, storage space, and network bandwidth. 
Existing studies have explored the optimization scheme

### 阅读判断

本文件只确认该文属于上述领域的候选文献，并记录可检索到的模型/问题窗口。作者声称的“现有不足”与实际模型是否一致，须在全文精读时逐段核对；当前不作创新性判断。

## 3. 系统模型（筛查）

（PDF p2起）

system model of the service migration technique 
in detail, and the proposed algorithm, IACMT, is elaborated in Section 4. 
Experimental results and performance evaluation are shown in Section 
5. Finally, conclusions and future work are presented in Section 6.
2. Related work
In recent years, research on container service migration techniques 
has focused on the balance between migration efficiency and resource 
utilization. Migration efficiency is usually based on service interruption 
time (i.e., downtime) and data transfer latency as key indicators, 
including the container image, the transfer length of in-memory data, 
and the synchronization latency between the source node and the target 
node. Resource utilization, on the other hand, is reflected in the con-
sumption of compute resources, storage space, and network bandwidth. 
Existing studies have explored the optimization schemes of service 
migration techniques from different perspectives, highlighting their 
respective advantages and limitations.
In the literature (Devi et al., 2023), the authors proposed a service 
migration method called pre-copy, which gradually reduces the down-
time during migration by iteratively transferring dirty page data several 
times during the migration process; however, this method will lead to 
longer migration time due to frequent updates of dirty page data when 
dealing with write-intensive applications. Literature (Im et al., 2022), on 
the other hand, proposes a migration technique based on post-copy, 
which reduces downtime by first migrating the base state of the ser-
vice and then gradually extracting the remaining data during the oper-
ation of the target node; however, this method is prone to performance 
degradation of the target server when the network bandwidth is l

**对象边界待核：** 通信节点、计算节点、服务存储节点是否区分；服务对象是内容、程序/镜像、服务实例还是任务；是否允许迁移/状态同步/回传。上述均需回原文逐节确认，不能由摘要补全。

## 4. 优化目标、决策变量与约束（筛查）

未定位到章节；需回原文核查。

**当前只能确认：** 上文窗口出现了问题建模/优化问题章节。具体目标函数、离散/连续变量、时间尺度、逐项约束、任务时延定义、迁移开销及通信—计算耦合，尚未逐式摘录；标记为**未核实**。

## 5. 求解方法（筛查）

（PDF p2起）

proposed algorithm, IACMT, is elaborated in Section 4. 
Experimental results and performance evaluation are shown in Section 
5. Finally, conclusions and future work are presented in Section 6.
2. Related work
In recent years, research on container service migration techniques 
has focused on the balance between migration efficiency and resource 
utilization. Migration efficiency is usually based on service interruption 
time (i.e., downtime) and data transfer latency as key indicators, 
including the container image, the transfer length of in-memory data, 
and the synchronization latency between the source node and the target 
node. Resource utilization, on the other hand, is reflected in the con-
sumption of compute resources, storage space, and network bandwidth. 
Existing studies have explored the optimization schemes of service 
migration techniques from different perspectives, highlighting their 
respective advantages and limitations.
In the literature (Devi et al., 2023), the authors proposed a service 
migration method called pre-copy, which gradually reduces the down-
time during migration by iteratively transferring dirty page data several 
times during the migration process; however, this method will lead to 
longer migration time due to frequent updates of dirty page data when 
dealing with write-intensive applications. Literature (Im et al., 2022), on 
the other ha

**待核：** 数学建模方法与实际求解方法是否一致；是否使用精确MIP求解器、凸优化、Lyapunov、DRL、P-DQN、启发式或在线算法；离散/连续决策如何处理；是否有收敛性/复杂度/近似保证。

## 6. 仿真实验（筛查）

（PDF p10起）

Simulation parameters setting.
Notation Description Value
W Wireless channel bandwidth [10, 50] MHz
Dc Average container runtime size 150 MB
F Edge server provided CPU clock frequency 3.5 GHz
ε Computing performance degradation parameter 0.1
D(pk) The number of bytes per page 4096 Byte
ζ Data encapsulation overhead 0.15
φ The time constant for generating checkpoints 10 ms
ρ The time constant for restoring checkpoints 30 ms
σ Page update rate threshold 0.7
ϖ Throughput variation range 0–4000 Ops/s
Table 3 
In-vehicle service load information.
Service Load Name Description Image Size 
(MB)
Tensorflow/object_detection: 
latest
TensorFlow object detection 
model
578.94
Openvino/sign- 
recognition:2.0
Traffic sign recognition model 852.37
NVIDIA/lanenet:v1.2 LaneNet lane detection task 423.59
Detectron2/tracking:latest Detectron2 object tracking 
service
953.72
Yolov12/pedestrian:v1.0 YOLOv12 pedestrian detection 
model
780.47
Pytorch/collision-detect:2.1 Collision detection task 1203.91
Opencv/depth-est:4.5 Depth estimation algorithm 648.33
Fig. 5. Cpu usage rate comparison.
Fig. 6. Compression performance comparison of several migration 
technologies.
K. Hua et al. Journal of Network and Computer Applications 247 (2026) 104412 
10 
This phenomenon suggests that the use of a single compression algo-
rithm has limited effectiveness and is poorly adapted to the data char-
acteristics

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
| 活动任务连续执行 | 是 |
| 迁移传输 | 是 |
| 多业务QoS | 未提及 |
| 业务流量非平稳 | 部分 |
| 信道动态变化 | 是 |

上述“是/部分/未提及”是全文文本关键词命中结果，可能包含引言/相关工作对他人工作的描述，**不能直接当作本文模型已考虑该要素**。必须回到系统模型、问题式和实验节确认。

## 9. 与三个研究点的关系（筛查级）

| 研究点 | 当前证据状态 | 待核问题 |
|---|---|---|
| P1 服务配置/任务执行多时间尺度 | 关键词：任务截止期限=未提及；服务更新周期=未提及；时间尺度自适应=未提及 | 是否联合服务部署与卸载；慢时标是否固定；是否按剩余时限调整保持时间；快层结果是否反馈慢层 |
| P2 不完美预测迁移/传输资源 | 关键词：车辆移动性=是；位置预测误差=未提及；服务迁移=是；活动任务连续=是；迁移传输=是 | 是否建立迁移窗口；是否联合时机/目标/带宽；迁移与正常任务是否共享资源；是否有源端持续服务/回退 |
| P3 多业务非平稳在线协同 | 关键词：多业务QoS=未提及；业务流量非平稳=部分；信道动态=是 | 是否区分业务构成/流量/信道；是否联合部署/带宽/CPU；是否有部署调整成本/队列/长期约束 |

## 10. 可复用证据与不能说的话

- 可复用：该文题名、作者、年份、DOI、领域归属（在完成身份核验后）；上述模型/问题窗口可作为回查入口。
- 不能据此说：该文“未考虑”某要素；“该领域没有”某机制；该文提出了某项创新。筛查稿没有完成逐式/逐图核对。

## 11. 关键原文证据与页码

- 模型窗口：2（PDF页）
- 问题窗口：None（PDF页）
- 方法窗口：2（PDF页）
- 仿真窗口：10（PDF页）
- 结论窗口：2（PDF页）

### 关键词命中片段

- p1: Seamless service migration for the Internet of Vehicles in edge computing: A dynamic dirty page filtering and two-stages compression technique Kaifeng Hua a, Shengchao Su a,b,* , Nannan Zhang a a School of Electronic and Electrical Engineering,
- p1: ty of Engineering Science, Shanghai, China b Collaborative Innovation Center of Smart Energy Equipment and Power Conversion, Suzhou Vocational University, Suzhou, China ARTICLE INFO Keywords: Internet of Vehicles Docker Service migration Container virtualization Data compression Mobile edge computing ABSTRACT With the growing demand for dynamic res

## 12. 阅读闭合状态

- 全文文本：已提取（15页）。
- 逐页阅读：未完成。
- 公式/图/表核对：未完成。
- 附录：未核实。
- 结论：**筛查级，不得计入全文精读完成数。**
