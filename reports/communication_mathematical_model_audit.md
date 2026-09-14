# 通信数学模型完整审计

审计日期：2026-09-06
审计对象：C:\Users\tangning\Desktop\code\adaptive_cache_mec
审计性质：只读审计；未修改算法、配置、reward、PPO、SUMO 或正式 artifact。

本报告严格以
C:\Users\tangning\AppData\Local\Temp\codex-file-preview-HWc0Ve\12.pdf
作为本项目的主系统模型。第一份 IEEE 论文
Joint_Service_Caching_Communication_and_Computing_Resource_Allocation_in_Collaborative_MEC_Systems_A_DRL-Based_Two-Timescale_Approach.pdf
只作为独立外部参照，不替代 12.pdf 的系统定义。

## 1. 审计范围与结论

本次检查了：

- 12.pdf 第 1、2、3、5、6、7、8 页中的位置、地面/空地信道、带宽、速率、传输时延、计算时延、能耗和总时延公式；
- system_config.toml 的通信、CPU、Cloud、UAV 和时间参数；
- src/common/system_model.py 的纯公式实现；
- src/decision/task_allocation.py 的实际位置、Matching、KKT、边缘/Cloud 执行路径；
- src/decision/cache_configuration.py 的预测位置、预测任务和 surrogate 复用路径；
- src/decision/slot_environment.py 的配置阶段到实际调度阶段的位置耦合；
- 已有 cloud_delay_analysis.md、computation_load_analysis.md、cache_offloading_coupling_audit.md 和 task_mathematical_model_audit.md；
- 固定 seed=4401、slot=20、实际 trajectory/tasks artifact 的一条真实任务链路。

总体结论如下：

| 检查项 | 结论 |
|---|---|
| 输入量、距离、带宽、速率、功率、能耗单位 | 通过；代码内部单位链一致 |
| 地面链路 | 通过；与 12.pdf 的 d、beta_g d^(-alpha_g) 和 Shannon 速率一致 |
| UAV 链路 | 公式通过；位置水平耦合通过，但 UAV 通信高度存在 50 m / 100 m 双字段问题 |
| 边缘 CPU / 带宽 KKT | 通过；闭式比例、资源和约束均通过固定 seed 数值复算 |
| Cloud 传输、计算和能耗 | 公式内部一致，psi_c 只乘通信时延且不乘能耗，符合 12.pdf |
| Cloud 同槽资源均分 | 代码内部一致，但这是 12.pdf 未明确写出的附加建模假设，且未显式配置化 |
| 实际位置到通信链路的耦合 | 通过；实际调度使用当前真实车辆位置和最终 UAV 位置 |
| 是否可直接开始正式 PPO | 不建议直接开始；先确认 UAV 高度与 Cloud 共享语义 |

因此，本次没有发现地面/空地 Shannon 公式、边缘 KKT 约束或 Cloud psi_c 能耗处理的实现断裂；但在训练前仍有两个模型合同需要由系统模型最终确认：

1. 12.pdf 中的固定 UAV 高度 H 是否应为当前通信配置的 100 m，还是 artifact 中部署位置的 50 m；
2. Cloud 的 R_c、F_c 是每任务等效能力，还是当前代码采用的单 Cloud 总资源按同槽 Cloud 任务数均分。

## 2. 12.pdf 主模型与代码路径

### 2.1 12.pdf 中的主公式

12.pdf 给出的通信/执行口径是：

1. 地面距离：

   d_v,m(t) = ||q_v(t) - q_m(t)||_2

2. 地面平均信道功率增益：

   h_v,m(t) = beta_g [d_v,m(t)]^(-alpha_g)

3. UAV 水平距离、三维距离和仰角：

   d_h,v,u(t) = ||q_v(t) - q_u(t)||_2

   d_v,u(t) = sqrt(d_h,v,u(t)^2 + H^2)

   theta_v,u(t) = arcsin(H / d_v,u(t))

4. UAV LoS/NLoS 平均增益：

   P_L = 1 / (1 + eta1 exp(-eta2(theta - eta1)))

   h = P_L beta_L d^(-alpha_L) + (1-P_L) beta_N d^(-alpha_N)

5. 服务节点上传速率：

   R_v,m(t) = b_v,m(t) B_m log2(1 + p_v h_v,m(t) / sigma^2)

6. 边缘上传和执行：

   T_tr = D_v / R_v,m

   T_cmp = L_v / (w_v,m F_m)

   T = T_tr + T_cmp

   E = p_v T_tr

7. Cloud：

   T_v,c = psi_c D_v / R_c + L_v / F_c

   E_v,c = p_v D_v / R_c

   其中 Cloud 车辆发射能耗不乘 psi_c，结果返回时延和能耗忽略。

8. 本地执行：

   T_local = L_v / F_v

   E_local = kappa_v L_v F_v^2

   本地不上传，不产生无线传输时延和传输能耗。

以上公式来自 12.pdf 第 5--8 页；代表性页面已做文字提取和渲染复核。

### 2.2 代码的实际调用链

纯公式层：

- src/common/system_model.py:32-50：本地计算；
- src/common/system_model.py:53-64：地面信道增益；
- src/common/system_model.py:66-87：Shannon 上行速率；
- src/common/system_model.py:90-128：边缘上传、计算和车辆传输能耗；
- src/common/system_model.py:131-178：Cloud 共享资源、psi_c、Cloud 计算和车辆传输能耗；
- src/common/system_model.py:181-225：UAV 三维距离、仰角、LoS 概率和平均增益。

实际状态层：

- src/decision/task_allocation.py:1016-1043：按 SV/RSU/UAV 角色选择地面或空地信道；
- src/decision/task_allocation.py:1046-1061：用完整带宽计算节点基准速率；
- src/decision/task_allocation.py:1153-1255：边缘 KKT 资源分配和 Cloud 同槽资源均分；
- src/decision/task_allocation.py:1258-1385：将实际 matching、资源比例、通信、计算和排队等待合成为任务结果；
- src/decision/task_allocation.py:1844-1875：从 trajectory.csv 读取车辆真实位置，并构造 SV/RSU/UAV 服务节点位置。

预测配置层：

- src/decision/cache_configuration.py:468-559：从预测场景读取预测车辆位置和预测任务；
- src/decision/cache_configuration.py:562-645：预测配置评价复用 Matching/KKT 口径；
- src/decision/slot_environment.py:125-130：明确配置阶段使用预测切片、实际 reward 使用真实任务和实际 Matching；
- src/decision/slot_environment.py:241-312：配置完成后用最终 UAV 位置执行实际调度。

## 3. 单位和参数映射

### 3.1 代码单位合同

src/common/system_model.py:7-15 明确约定：

| 量 | 代码字段 | 单位 |
|---|---|---|
| 输入数据 | input_bits、D | bit |
| 计算量 | cpu_cycles、L | cycle |
| CPU 频率 | *_cpu_cycles_per_s、F | cycle/s |
| 系统带宽 | bandwidth_hz、B | Hz |
| 传输速率 | rate_bps、R | bit/s |
| 发射功率 | transmit_power_w、p | W |
| 噪声总功率 | noise_power_w、sigma^2 | W |
| 距离、高度 | trajectory/node position/UAV altitude | m |
| 时延 | *_delay_s | s |
| 车辆能耗 | vehicle_energy_j | J |

当前任务 artifact 的 tasks.csv.input_size 已经是 bit，不能再次乘 8000。这个结论与 task_mathematical_model_audit.md 一致。

### 3.2 当前配置

| 参数 | 当前值 | 代码入口 |
|---|---:|---|
| dt_s | 1.0 s | simulation.dt_s |
| SV / RSU / UAV 带宽 | 10.0 MHz | nodes.sv/rsu/uav.bandwidth_hz |
| TV 发射功率 | 0.1995262315 W | nodes.vehicle.transmit_power_w |
| 地面参考增益 | 1.6368e-5 | communication.ground.channel_gain_at_1m |
| 地面路径损耗指数 | 2.75 | communication.ground.pathloss_exponent |
| 噪声总功率 | 3.16227766e-13 W | communication.noise.noise_power_w |
| UAV eta1、eta2 | 9.61、0.16 | communication.air |
| UAV beta_L、beta_N | 1.1302166e-4、1.4228584e-6 | communication.air |
| UAV alpha_L、alpha_N | 2.0、2.0 | communication.air |
| Cloud 总上行 | 200.0 Mbps | nodes.cloud.total_bandwidth_bps |
| Cloud 总 CPU | 20.0 GHz | nodes.cloud.total_cpu_cycles_per_s |
| psi_c | 11.0 | nodes.cloud.cloud_delay_factor |

固定 seed 样本中的输入数据和计算量直接进入上述 bit/cycle 公式，没有发现重复单位换算或 bit/byte 混用。

## 4. 位置、距离和 UAV 耦合

### 4.1 实际位置耦合通过

12.pdf 第 1 页区分了配置阶段的预测位置和任务调度阶段的实际位置；当前代码保持了这个边界：

- SV 位置由当前真实 trajectory.csv 读取；
- RSU 位置由 node_positions.json 固定；
- UAV 位置由当前配置结果的最终水平位置提供；
- 实际 slot_environment.execute_slot() 在 src/decision/slot_environment.py:241-258 重新构造服务节点位置后，才调用实际 Matching/KKT；
- UAV 移动结果因此会进入后续 UAV 水平距离、空地信道增益、速率、传输时延、能耗和 utility。

预测配置阶段则用 prediction_scenarios.npz 中的预测车辆位置，且 cache_configuration.py 的 surrogate 评价复用同一套边缘信道和 KKT 公式。未发现预测位置泄漏到实际 reward 的问题。

### 4.2 无硬距离 cutoff 与主模型一致

build_feasible_targets()（src/decision/task_allocation.py:125-149）只根据缓存可用性构造边缘候选，Cloud 始终可用；range_m 字段在当前正式 Matching/KKT 中没有作为硬覆盖截断。

这与 12.pdf 的通信定义一致：12.pdf 给出了距离对信道增益和速率的影响，但没有在执行可行集合中额外定义 SV/RSU/UAV 的硬覆盖半径。当前 range 字段的配置注释也将其标为 legacy compatibility parameter。

因此，长距离链路会自然变慢，不会因为距离超过 legacy range 直接从候选集合删除。这个行为应继续作为模型约定保留，不能把 range 字段误读成已生效的覆盖约束。

### 4.3 UAV 高度存在模型合同风险

当前项目同时存在两个高度：

- system_config.toml:215：nodes.uav.deployment_altitude_m = 50.0；
- data/generated/node_positions.json：UAV position 的 z 坐标为 50.0；
- system_config.toml:227：nodes.uav.altitude_m = 100.0；
- src/decision/cache_configuration.py:193 将后者读入 uav_altitude_m；
- src/decision/task_allocation.py:1025-1034 的空地信道始终使用 cache_config.uav_altitude_m，即 100.0 m；
- 实际服务节点的通信位置只保留二维 x/y，因此 node_positions.json 的 z 不会进入 air_channel_gain()。

配置注释将 50 m 解释为部署高度、100 m 解释为动态飞行模型高度，这可能是有意的字段分工；但是 12.pdf 的公式只有一个固定高度 H，没有定义“部署 z”和“通信 H”可以不同。因此，从物理位置模型看，这是训练前必须确认的接口。

固定 seed、slot=20、time=120 s、真实任务 TV 18 到 UAV_0 的敏感性复算如下：

| 采用的 H | 三维距离 (m) | LoS 概率 | 速率 (Mbps) | 上传时延 (s) | 总边缘时延 (s) |
|---:|---:|---:|---:|---:|---:|
| 50 m | 1608.396531 | 0.028878 | 10.929291 | 2.249951 | 6.673955 |
| 100 m（当前代码） | 1610.726358 | 0.038019 | 12.496860 | 1.967724 | 6.391728 |

同一任务下仅高度解释不同，速率相对变化约 14.34%，不是可以忽略的数值差异。报告不在本次审计中修改高度或重新生成 artifact；需要先确认 12.pdf 的 H 在当前系统合同中究竟对应哪一个字段。

## 5. 信道、速率和通信能耗

### 5.1 地面链路

system_model.ground_channel_gain() 实现：

h = beta_g * distance_m^(-alpha_g)

task_allocation._channel_gain() 用 TV 与 SV/RSU 的二维欧氏距离调用该函数。距离为 0 时仅使用 max(horizontal, 1e-9) 作数值保护；这避免了公式奇异，不改变正常样本的结果。

### 5.2 UAV 链路

system_model.air_channel_gain() 依次计算：

1. sqrt(horizontal^2 + H^2)；
2. asin(H/d) 并转换为 degree；
3. 12.pdf 同形的 LoS 概率；
4. LoS/NLoS 增益按概率加权。

没有加入额外干扰项或小尺度随机衰落。12.pdf 使用平均信道功率增益并假设正交频谱、忽略同频干扰，因此当前实现与主模型一致。第一份 IEEE 论文使用了带 Rayleigh 小尺度项的外部模型；那是参照差异，不应混入当前 12.pdf 主模型。

### 5.3 Shannon 速率

system_model.uplink_rate() 实现：

R = fraction * B * log2(1 + p * h / noise_power)

bandwidth_fraction 只乘在可用带宽外侧，信噪比不随带宽比例改变；这与 12.pdf 的 b_v,m B_m 形式一致。不同任务在同一服务节点使用正交带宽比例，因此代码没有额外加入同频干扰项。

### 5.4 边缘传输能耗

edge_execution() 实现：

T_upload = D / R

E_vehicle = p * T_upload

边缘节点 CPU 只影响 L/(wF)，不影响车辆上传能耗；这与 12.pdf 第 7--8 页一致。

### 5.5 Cloud 传输能耗

cloud_execution() 实现：

T_cloud_upload = psi_c * D / R_i

T_cloud_compute = L / F_i

E_cloud_vehicle = p * D / R_i

psi_c 没有乘到能耗项，符合 12.pdf 对“车辆仅在无线接入阶段消耗发射能量”的定义。现有 cloud_delay_analysis.md 的 psi_c=11.0 分析与此一致。

## 6. 带宽/CPU 共享、KKT 和资源约束

### 6.1 边缘 KKT

当前边缘节点按 node_id 分组任务：

- CPU 目标系数：

  c_cpu,v = omega_t * L_v / (T_ref * F_m)

- 带宽目标系数：

  c_bw,v = D_v / Rbar_v * (omega_t/T_ref + omega_e*p_v/E_ref)

其中 Rbar_v 是带宽比例为 1 时的完整带宽基准速率。

代码在 src/decision/task_allocation.py:1064-1118 中对 sum c_i/r_i 使用

r_i = clip(sqrt(c_i/lambda), epsilon, 1)

并通过二分搜索使资源总和尽量等于 1。资源约束和实现检查如下：

- 每个边缘节点 sum w_v,m <= 1；
- 每个边缘节点 sum b_v,m <= 1；
- epsilon * N <= 1，否则返回不可行；
- 单任务时 CPU 和带宽比例均为 1；
- 本地任务不参加边缘 CPU/带宽分配；
- Cloud 不参加 SV/RSU/UAV 共享带宽分配。

这与 12.pdf 第 4、6 页的 sum w <= 1 和 sum b <= 1 约束一致。KKT 目标中的能耗项对 CPU 比例没有依赖，因此 CPU KKT 只包含计算时延，带宽 KKT 同时包含时延和车辆传输能耗，这是当前公式的正确分解。

### 6.2 Cloud 同槽资源均分

当前正式代码在 cache_configuration.CacheOptimizationConfig 中以 cloud_resource_sharing=True 为默认值：

- src/decision/task_allocation.py:1231-1244：同一 matching 中的 Cloud 任务数为 N_cloud，每任务分配 1/N_cloud；
- src/decision/task_allocation.py:1303-1337：Cloud 执行时将 cloud_task_count=N_cloud 传入纯公式；
- 因此 R_i = R_total/N_cloud、F_i = F_total/N_cloud。

这个实现内部是闭合的，也与当前 cloud_delay_analysis.md 和 computation_load_analysis.md 的说明一致；但是要区分“代码内部一致”和“主 PDF 已明确规定”：

- 12.pdf 第 7 页把 R_c 定义为任务车辆到蜂窝接入网络的固定等效上传速率，并把 F_c 定义为 Cloud 为任务提供的等效计算频率；
- 12.pdf 明确 Cloud 不参加 SV/RSU/UAV 的共享带宽分配，但没有明确写出 R_c、F_c 还要被同槽 Cloud 任务数均分；
- 当前 system_config.toml 有 Cloud 总带宽和总 CPU 的注释，但 cloud_resource_sharing 本身不是一个显式 TOML 配置键，而是 Python dataclass 默认值。

因此该项不是公式代码的数值 bug，而是一个应在训练前写入正式模型合同的附加假设。如果主模型希望 Cloud 是每任务固定等效能力，应采用 R_c、F_c 的 per-task 语义；如果主模型希望 Cloud 是单个有限共享节点，则当前均分逻辑可以保留，但应把该语义明确写入 12.pdf 对应的系统说明和配置合同。

## 7. 固定 seed 实际任务数值复算

### 7.1 任务和路径设置

固定设置：

- seed=4401；
- slot=20，time_s=100+20*1=120 s；
- 真实任务：TV 18，service 5；
- D=24,590,367.807820376 bit；
- L=22,120,021,568.55435 cycle；
- 本次数值诊断使用 deterministic_initial_cache；该 service 对 SV、RSU、UAV 和 Cloud 均属于可行目标；
- 在这组真实任务/真实位置加 deterministic_initial_cache 的离线 Matching 中选择 RSU_1；
- 同一 slot 的实际 matching 中 Cloud 任务数为 7。

为检查通信公式，SV、RSU、UAV 三条边缘路径均使用同一真实任务、同一真实 TV 位置和当前节点位置，带宽/CPU 比例取单任务完整资源 1；Cloud 同时列出单任务 N=1 和当前 slot 的 N=7。除 Cloud 并发比较外，不混入其他任务的资源竞争。

### 7.2 TV 18 到各节点

| 路径 | 节点 | 距离 | 信道增益 | 速率 | 上传时延 | 计算时延 | 总时延 | 车辆能耗 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TV→SV | SV 23 | 687.699789 m | 2.577203e-13 | 2.173679 Mbps | 11.312788 s | 22.120022 s | 33.432810 s | 2.257198 J |
| TV→RSU | RSU_1 | 346.039042 m | 1.703733e-12 | 10.530995 Mbps | 2.335047 s | 4.424004 s | 6.759051 s | 0.465903 J |
| TV→UAV | UAV_0 | 水平 1607.619171 m；3D 1610.726358 m | 2.183819e-12 | 12.496860 Mbps | 1.967724 s | 4.424004 s | 6.391728 s | 0.392612 J |
| TV→Cloud，N=1 | Cloud | 独立等效链路 | — | 200.000000 Mbps | 1.352470 s | 1.106001 s | 2.458471 s | 0.024532 J |
| TV→Cloud，N=7 | Cloud | 独立等效链路 | — | 28.571429 Mbps | 9.467292 s | 7.742008 s | 17.209299 s | 0.171725 J |

RSU 路径的数值与实际 Matching 返回的 TV 18 结果完全一致：delay=6.75905125979695 s、energy=0.46590311752813895 J。SV/UAV 行是同一真实任务在可行缓存目标上的公式对照，Cloud 行是同一 slot 并发口径的对照。

### 7.3 手算与代码对照

独立按公式计算 h、R、D/R、L/(wF)、pD/R 后，与 system_model.py 返回值的最大绝对误差为：

| 检查 | 最大绝对误差 |
|---|---:|
| 地面 SV/RSU 的 gain、rate、delay、energy | 0 |
| UAV 的 3D distance、仰角、LoS、gain、rate、delay、energy | 4.04e-28 |
| Cloud，N=1 | 0 |
| Cloud，N=7 | 0 |

这证明当前实际 bit/cycle 数据、位置、信道、速率、通信时延、计算时延和车辆能耗在公式层是闭合的。

### 7.4 实际 slot 的 KKT 资源和约束

slot=20 的实际 matching 任务计数为：

- UAV_0：2 个任务；
- UAV_1：2 个任务；
- RSU_0：1 个任务；
- RSU_1：1 个任务；
- Cloud：7 个任务。

对 UAV_0 的两个实际任务独立使用 sqrt(c_i)/sum(sqrt(c_i)) 复算：

| 资源 | 闭式比例 | 代码比例 | 最大误差 |
|---|---|---|---:|
| CPU | 0.5901371215、0.4098628785 | 0.5901371215、0.4098628785 | 2.01e-13 |
| 带宽 | 0.6768643203、0.3231356797 | 0.6768643203、0.3231356797 | 5.46e-13 |

实际 resource sum 如下；超过 1 的部分仅为二分和浮点误差：

| 节点 | CPU sum | bandwidth sum |
|---|---:|---:|
| RSU_0 | 1.000000000000 | 1.000000000000 |
| RSU_1 | 1.000000000000 | 1.000000000000 |
| UAV_0 | 1.000000000000341 | 1.000000000000806 |
| UAV_1 | 0.999999999999722 | 1.000000000000014 |
| Cloud | 1.000000000000 | 1.000000000000 |

固定 seed 的数值诊断没有发现 KKT 比例、节点资源和公式结果不一致。

## 8. 排队等待与总时延口径

12.pdf 第 8 页的任务完成时延是通信与计算项的和，没有显式排队项。当前工程在实际执行层追加了跨 slot 工作量：

1. evaluate_matching_with_kkt() 先计算 local/edge/Cloud 的纯执行时延；
2. src/decision/task_allocation.py:1359-1362 将相应执行节点的 queue_wait_s 加到任务 delay；
3. slot_environment.py:293-307 按 dt_s 推进下一 slot queue workload。

因此代码的实际总时延是：

T_code = T_pdf_execution + T_queue_wait

这不是通信公式本身的错误，而是工程在 1 s 基础时隙下加入的跨时隙服务扩展。当前 KKT 资源分配对一个节点内的任务使用本 slot 资源竞争；已有 queue workload 对同节点任务是外加等待量，不改变本 slot 的 D/R 和 L/(wF) 资源系数。

这一点与 computation_load_analysis.md 的结论一致：当前任务计算量相对 1 s slot 较重，排队可能是实际总 delay 的主要扩展项。建议下一步计算模型审计专门检查：

- queue workload 的单位是否始终为秒；
- queue 更新是按完整服务时延还是按 slot 内可完成工作量；
- 同一节点多个任务的服务顺序与 KKT 并发执行是否符合最终计算模型；
- delay normalization 的 T_ref 是否与可能出现的排队时延量级匹配。

## 9. 与第一份 IEEE 论文的独立对照

第一份 IEEE 论文仅作为外部参照，不能替代 12.pdf。其第 4--5 页和第 10 页附近给出的主要口径如下：

| 对照项 | 第一份 IEEE 论文 | 当前项目/12.pdf | 判断 |
|---|---|---|---|
| 输入与计算量 | 输入 bit、计算 cycle | 输入 bit、计算 cycle | 一致 |
| 上行速率 | b W log2(1+p h/sigma^2) | b B log2(1+p h/sigma^2) | 核心形式一致 |
| 地面信道 | Rayleigh 小尺度项乘距离衰落 | 12.pdf/当前代码使用平均增益 | 外部信道更细，不应混入主模型 |
| 边缘传输时延 | input/R | D/R | 一致 |
| 边缘能耗 | p*input/R | p*D/R | 一致 |
| 边缘计算 | cycles/(fraction*F) | L/(wF) | 一致 |
| Cloud/协作 | 有 associated ES、cooperative ES 和固定 R_co/R_back 的中继路径 | 12.pdf 使用直接等效 R_c 和 psi_c D/R_c | 路径定义不同 |
| 任务产生 | 每个短时隙随机产生任务 | 12.pdf/当前代码为 Bernoulli 到达、每 slot 至多一个 | 当前以 12.pdf 为准 |
| 缓存和带宽 | ES 级资源约束 | SV/RSU/UAV KKT 资源约束 | 结构可参照，参数语义不直接替换 |

因此不能从第一份论文的 R_co、R_back、Rayleigh fading 或 cooperative relay 路径反推当前系统必须增加对应项。当前审计只使用其支持核心 D/R、pD/R、L/(wF) 和带宽比例约束的部分。

## 10. 与已有报告的一致性

- reports/task_mathematical_model_audit.md：确认 tasks.csv 的 input_size 已是 bit、cpu_cycles 已是 cycle；本报告直接复用该单位合同。
- reports/computation_load_analysis.md：确认 Cloud 总 CPU 在当前模型中可按同槽任务数共享；本报告进一步验证了 Cloud 通信/计算/能耗的数值链。
- reports/cloud_delay_analysis.md：确认 psi_c=11 只放大 Cloud 通信时延，不改变 Cloud 计算时延和车辆上传能耗；本报告公式复算通过。
- reports/cache_offloading_coupling_audit.md：确认 cache state 先决定 feasible target，再由 Matching/KKT 决定实际路径；本报告选取的 TV 18 任务确实同时具备 SV/RSU/UAV/Cloud 候选，实际最终选择由 delay/energy/resource 共同决定。

现有报告之间没有发现通信单位、Cloud psi_c 或 KKT resource sum 的互相矛盾。当前新增的两个模型风险是 UAV 高度字段语义和 Cloud 总资源均分是否属于正式主模型。

## 11. 问题清单与训练前建议

| 编号 | 问题 | 当前判断 | 训练前要求 |
|---|---|---|---|
| C-1 | UAV 部署/位置 z=50 m，但通信 H=100 m | 公式实现内部一致，物理位置合同不唯一；对样本速率有 14.34% 影响 | 明确 12.pdf 的 H 对应 50 m 还是 100 m；确认后再决定是否同步位置 artifact 或只保留单一通信高度 |
| C-2 | Cloud 同槽 R_total/N_cloud、F_total/N_cloud | 代码和已有分析内部一致，但 12.pdf 未明确该均分，且开关是 Python 默认值 | 明确 Cloud 是 per-task 等效资源还是单节点总资源；把最终语义写入正式配置/模型合同 |
| C-3 | 代码总 delay 额外包含 queue wait | 工程扩展，不是通信公式断裂 | 在下一步计算/排队模型审计中确认 queue 更新、服务顺序和 dt_s 口径 |
| C-4 | 12.pdf 任务输入单位只写 KB，未声明十进制/二进制 | 代码和 artifact 内部一致，属于外部规格歧义 | 继续保留 1 KB=1000 byte=8000 bit 的明确实验口径 |
| C-5 | SV/RSU/UAV range_m 未作为 hard cutoff | 与 12.pdf 距离衰落主模型一致 | 不要在训练外部脚本中擅自把 legacy range 当作通信可行性约束 |

## 12. 最终结论

通信公式主链路已经通过固定 seed 的实际任务复算：

真实任务 -> 真实 TV/SV/RSU/UAV 位置 -> 信道增益 -> Shannon 速率 -> 上传时延/能耗 -> KKT CPU/带宽 -> 计算时延 -> Cloud/edge 总时延

在当前代码和 artifact 下：

- 地面、空地、边缘、Cloud 的纯公式结果可独立手算复现；
- 实际 SV/RSU/UAV 位置与最终 UAV 配置结果确实进入通信链路；
- KKT 闭式比例和资源和约束通过；
- Cloud psi_c 的时延/能耗分离通过；
- 没有发现重复 bit 转换、重复带宽比例、能耗乘错 psi_c 或把 Cloud 混入边缘共享带宽的实现错误。

但这还不足以给出“无需确认即可正式 PPO 训练”的结论。建议下一步进入计算/排队模型审计，同时先由模型定义者确认 C-1 和 C-2；确认后再做短时、非训练的回归验证，最后再决定是否进入正式 PPO。

本次只应新增本报告；没有修改配置、源代码、算法、PPO、reward 或 data/generated/ artifact。
