# PPO 训练前通信 / Cloud 模型合同修复验证

本报告只针对已确认的正式系统模型执行修复验证。主模型来源是
C:\Users\tangning\AppData\Local\Temp\codex-file-preview-HWc0Ve\12.pdf；第一篇 IEEE 论文只用于 Cloud 参数和语义校准，不替代 12.pdf 的系统定义。

本次没有启动 PPO、没有训练预测器、没有做 train/test split、没有启动 SUMO，也没有修改已有历史审计报告。

## 1. 最终合同

| 项目 | 正式合同 |
|---|---|
| UAV 高度 | deployment altitude、channel altitude、node artifact 的 z 坐标统一为 100 m；UAV 只做二维水平移动 |
| Cloud CPU | 每任务固定 F_c = 5e9 cycle/s |
| Cloud uplink | 每任务固定 R_c = 200e6 bit/s |
| Cloud 扩展系数 | psi_c = 11，未修改 |
| Cloud 资源开关 | nodes.cloud.resource_sharing = false，缺失或 true 均 fail-closed |
| Cloud 时延 | T_c = psi_c × D / R_c + L / F_c |
| Cloud 车辆能耗 | E_c = p × D / R_c；psi_c 不进入能耗 |
| Cloud 资源共享 | 不按同槽 Cloud 任务数 N_cloud 做除法；Cloud 不进入 edge KKT CPU/BW 资源和 |
| Cloud 候选 | 若可执行缓存 SV/RSU/UAV 的 M_edge 非空，只加入所有可用 edge 节点，不加入 Cloud；M_edge 为空才加入 Cloud |
| Local | Local 候选独立判断，可与 edge 共存，不改变 Cloud 门控 |
| 单位 | 1 KB = 1000 byte = 8000 bit；service ID 为 0-based |

## 2. 实现修复

- system_config.toml：统一 UAV 高度为 100；Cloud 两组兼容字段均为 5e9 / 200e6；新增显式 resource_sharing=false；保留 psi_c=11。
- src/common/project_config.py：正式校验显式读取 Cloud 资源开关，拒绝静默回退；校验 deployment_altitude_m 与 altitude_m 一致，并把新字段纳入正式参数集合。
- src/common/system_model.py：cloud_execution 使用固定每任务资源。保留 cloud_task_count 仅作为兼容性、正整数校验参数，其值不再改变任何返回量。
- src/decision/task_allocation.py：build_feasible_targets 在正式实际和预测调用中都接收当前车辆位置、服务节点位置和 cache_config；节点缺失、坐标非法、角色无效、CPU/BW/信道/速率不可用时，缓存 edge 节点被排除并触发 Cloud 后备。Cloud 从 KKT 资源分配图和资源约束中移除。
- src/decision/cache_configuration.py：预测 surrogate 复用实际 Matching/KKT 的同一候选门控，并显式读取 Cloud 资源开关。
- data/generated/node_positions.json：通过 scripts/regenerate_artifacts.py 正式入口重生成，UAV_0/UAV_1 的 z 均为 100。
- PARAMETER_CALIBRATION.md：同步最终 Cloud per-task 语义和 UAV 高度合同。

## 3. 验证结果

### 3.1 配置和 artifact

通过 system_config.toml 解析及正式实验配置校验。正式 artifact 入口的 check-only 结果为全通过，且 writes_performed=0；生成入口只更新了受高度配置影响的节点位置、generation metadata 和 artifact consistency 状态，没有重生成任务、预测或 GRU。

实际检查结果：

- nodes.uav.deployment_altitude_m = nodes.uav.altitude_m = 100.0；
- node_positions.json 中 UAV_0、UAV_1 的 z 均为 100.0；
- Cloud cpu_cycles_per_s = total_cpu_cycles_per_s = 5.0e9；
- Cloud uplink_rate_bps = total_bandwidth_bps = 200.0e6；
- nodes.cloud.resource_sharing = false；
- psi_c = 11.0。

### 3.2 Cloud 纯公式

用 D=8.0e6 bit、L=3.2e9 cycle 和配置发射功率复算：

- cloud_task_count=1 与 cloud_task_count=7 的完整返回值逐项相同，证明没有同槽 N 共享；
- communication delay = 11 × D / 200e6；
- compute delay = L / 5e9；
- vehicle energy = p × D / 200e6；
- 将 psi 从 11 改为 10 的对照只改变通信时延，计算时延和能耗保持不变。

### 3.3 候选门控

固定 seed=4401、slot=20 的实际任务和节点位置上，使用 task 64、service 7 做最小门控测试：

| 状态 | 候选结果 |
|---|---|
| 没有 edge 缓存 | [CLOUD] |
| 缓存于一个当前可执行 edge 节点 | [edge]，不含 Cloud |
| Local 与 edge 同时命中 | [LOCAL:64, edge]，不含 Cloud |
| 名义上缓存但缺少节点位置 | [CLOUD] |
| 名义上缓存但节点坐标为 NaN | [CLOUD] |

同一 task/service 使用 actual 车辆位置和 prediction 场景的 horizon-0 位置分别门控：有可执行 RSU_0 时两者均为 [RSU_0]，无 edge 缓存时两者均为 [CLOUD]。因此实际卸载、预测 surrogate、cache candidate utility 和 UAV joint evaluation 共享同一 optimize_task_matching -> build_feasible_targets 门控链；源码中的正式调用点均传入物理上下文。

operation_generation_sanity 现已包含并通过三项互斥语义断言：无 edge 缓存必有 Cloud；有 edge 缓存必无 Cloud；Local 可与 edge 共存但不改变 Cloud 门控。

### 3.4 实际 Matching/KKT 短回归

slot=20、seed=4401 的 13 个到达任务通过实际位置、缓存状态、Matching、KKT、队列推进和 utility 计算：

| 指标 | 结果 |
|---|---:|
| full / incremental matching | 相同 |
| J、delay、energy、Q、CPU allocation、BW allocation 最大差异 | 0.0 |
| full / incremental 更新和稳定性 | 全部相同，重复运行确定 |
| 选中 Cloud 任务 | 3 |
| 选中 edge 任务 | 10 |
| 选中 Local 任务 | 0 |
| Cloud 出现在 KKT CPU/BW 映射 | 否 |
| candidate 集合中同时出现 edge 和 Cloud | 0 例 |

纯 KKT 最小验证的闭式 allocation 误差为 0.0，Cloud case resource_feasible=true，所有 delay/energy/utility 均为 finite。

零触发单 slot 执行也通过：cache trigger=0、UAV trigger=0、accepted cache updates=0、accepted UAV updates=0、cache changed entries=0；同一输入重复执行的 selected path changes=0，system utility 和所有逐任务量均 finite。该回归用于保持短时、非训练边界；没有把非零触发的完整配置搜索冒充为短回归结果。

## 4. 与修复前历史审计的对照

已有历史报告中的 slot 0 service 6 反事实案例为：

- 修复前：before=[CLOUD]，after=[RSU_0, CLOUD]；
- 修复后同一门控规则：若 RSU_0 是唯一可执行 edge，则 before=[CLOUD]，after=[RSU_0]。

这消除了“edge 缓存存在但 Cloud 仍然作为同一任务候选”的旧候选集合歧义。历史报告保持原样，没有被改写；该对照只在本报告中记录。

## 5. 结论和后续边界

通信、UAV 高度、Cloud per-task 资源和 Cloud 候选门控合同已经在训练前落到配置、纯公式、实际 Matching/KKT、预测复用链和正式 artifact。当前没有通信合同阻塞项，可以进入下一阶段。

建议后续单独做 queue / cross-slot execution 审计，重点检查已有历史风险中的 queue wait、deterministic_batch_fifo、slot duration 和跨节点 workload 更新；这属于下一阶段排队模型问题，不应回改本次已确认的 12.pdf 通信合同。
