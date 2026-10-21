# 自适应 Cache–UAV 联合 MAPPO 合同

## 正式实现与入口

正式算法为 **Alternating Actor Optimization V1（MAPPO）**（算法标识 `ADAPTIVE_CACHE_UAV_JOINT_MAPPO`），Checkpoint 方法为 `ADAPTIVE_CACHE_UAV_MAPPO`。唯一训练入口为 `scripts/training/mainline_runner.py`，它调用 `src/ppo_simple/trainer.py:PPOTrainer`；Trainer 模块不再作为独立命令行入口。冻结评价入口为 `python -m ppo_simple.evaluator` / `ppo-eval`。这里的 Joint 表示联合 Cache/UAV 策略；Actor 更新采用 Macro/Micro 交替调度。正式训练、评价及共享物理系统和数据生成参数统一使用 `configs/proposed/alternating_actor_updates_v1.toml`。

执行仍由正式环境完成：Cache 配置进入 `slot_environment.execute_configured_slot`，任务匹配与资源分配继续使用原 Matching/KKT，奖励来自正式 `SlotExecutionResult.system_utility`。

## Slot 系统效用与成本归一化

每个任务的效用为 `q_v(t)=omega_delay*(1-T_v(t)/T_v,max)+omega_energy*(1-E_v,veh(t)/E_v,max)`。`T_v(t)` 用秒计量，`E_v,veh(t)` 用焦耳计量；两者分别除以固定的系统级时延与车辆能耗上界。正式配置没有任务截止时间或任务级能耗预算，因此所有任务共用这两个系统级参考值。当前代码仍通过兼容配置键 `objective.T_ref`、`objective.E_ref` 读取它们。`Q(t)=sum_v q_v(t)` 是 Slot 内实际任务效用之和，因而为无量纲值。

本次正式配置的任务模型是每 Slot 独立完成，`T_v` 包含上传与计算时延；Cloud 上传项乘 `psi_c=17`，边缘节点上传项不乘该系数。最大任务规模由 `D_max=1000 KB*8000 bit/KB*(1+0.20)=9.6e6 bit`、`L_max=1000 cycle/bit*D_max*(1+0.10)=10.56e9 cycle` 给出。KKT 资源份额下限为 `1e-6`。按 `node_positions.json` 中 SUMO 区域边界对角线 `2664.329 m` 计算，最小地面链路速率下界为 `119.381 bit/s`，最小空地链路速率下界为 `169.464 bit/s`；Cloud 最小共享速率为 `20 Mbps*1e-6=20 bit/s`。因此对 Cloud、SV、RSU、UAV 分别计算 `T_r^upper=psi_r*D_max/R_r,min+L_max/F_r,min` 后取最大，得到 `T_v,max=10,640,415 s`（向上取整）。SV 的 1 GHz CPU 与最小 CPU 份额决定该上界；Cloud 路径上界为 `8,688,000 s`。这是覆盖当前合法路径的保守上界，不是精确可达最大值。

车辆能耗口径只包含上传无线发射能耗，不含计算能耗；Cloud 的 `psi_c` 不进入能耗。以 `P_tx=0.1995262315 W` 和各路径最小速率计算 `E_r^upper=P_tx*D_max/R_r,min` 后取最大，得到 `E_v,max=95,772.59112 J`，由 Cloud 的 `20 bit/s` 最小共享速率决定。该值同样是保守上界，不是训练集统计分位数。

缓存下载时延只按实际执行前后的缓存配置计算：

```text
T_ca,m(t) = sum_f [(8e6 * s_f_MB / R_back,m_bps) * max(c_new,mf(t)-c_old,mf(t), 0)]
T_ca,m_ref = 8e6 * cache_capacity_m_MB / R_back,m_bps
C_ca(t) = sum_m [T_ca,m(t) / T_ca,m_ref]
```

服务大小使用十进制 MB，回传速率使用 bit/s，因此 `8e6` 的单位换算正确。当前 `MULTI_NODE_FACTORIZED` 策略允许 5 个 SV、2 个 RSU 和 2 个 UAV 缓存节点在线重配置，故这些节点各自有固定参考值；未发生变化的节点成本为零。容量参考值是保守上界：新增服务大小不超过更新后的缓存容量，离散目录和合法替换规则可能使上界不可达到。缓存删除不计下载开销，UPDATE 意图不单独计费。

UAV 移动能耗仍使用实际水平二维位移：

```text
E_mov,u(t) = xi_mov,u * ||q_u(t)-q_u(t-1)||_2
E_mov,u_ref = xi_mov,u * vmax,u * dt
C_q(t) = sum_u [E_mov,u(t) / E_mov,u_ref]
```

每架 UAV 的参考值是速度约束给出的保守上界；地图边界和多 UAV 安全距离可能降低可达最大值。位置不变时能耗为零，策略输出 MOVE 标签本身不计费。

正式系统目标为 `G(t)=Q(t)-beta_ca*C_ca(t)-beta_q*C_q(t)`。配置保留 `lambda_cache=2e-5 s^-1` 和 `lambda_movement=5e-6 J^-1`，运行时仍按参考值总和派生无量纲权重：`beta_ca=lambda_cache*sum_m(T_ca,m_ref)`、`beta_q=lambda_movement*sum_u(E_mov,u_ref)`。当前每个节点的 Cache 参考值为 SV `480 s`（5 个）、RSU `1000 s`（2 个）、UAV `800 s`（2 个），合计 `6000 s`，故 `beta_ca=0.12`。每架 UAV 的移动参考值为 `225 J`（2 架合计 `450 J`），故 `beta_q=0.00225`。各节点成本分别除以自己的参考值；归一化成本上界为 `C_ca<=9`、`C_q<=2`，对应成本上界分别为 `1.08` 和 `0.0045`。相较旧系统级分母，缓存成本的实际时延系数按节点变为 `2.5e-4 s^-1`（SV）、`1.2e-4 s^-1`（RSU）、`1.5e-4 s^-1`（UAV）；移动成本系数变为每 UAV `1e-5 J^-1`。这次节点级归一化改变了系统优化目标，不与旧公式数值等价。两类成本在 Slot 系统效用中仍各扣除一次。

参考值在环境初始化时由固定容量、回传速率、移动速度和能耗系数生成，并写入运行时上下文及初始化日志。成本合同版本已改变，严格训练恢复仍由现有配置哈希门禁控制；旧 Checkpoint 不因结构可加载就与节点级奖励目标兼容。本次未修改恢复门禁或 Checkpoint 格式。

## 决策信息与双时间尺度

### Alternating Actor Optimization V1 更新合同

每组使用同一冻结旧策略采集 4 个完整 Episode 后执行一次 Policy Update。零起始 Update index 为偶数时只允许 Micro Actor 更新，奇数时只允许 Macro Actor 更新；首组因此更新 Micro Actor。每组仍更新 Macro Critic、Micro Critic 和 Long-Value Critic。冻结的 Actor 不产生 Actor 梯度，且其 Adam 状态不推进。正式 Runner 与 Training Pool 合同仅接受 `ALTERNATING_ACTOR_UPDATES_V1` 配置，不提供切换回 Joint Actor 更新的模式。

受控停止只在完整的 4-Episode 更新组保存日志、Manifest、Summary 和 Checkpoint 后生效；它不会被记录为完成目标 Episode 数的正式训练。

每次决策状态包含当前 Cache 矩阵、当前 UAV 坐标、决策时可用的车辆轨迹预测、服务流行度预测、预测 horizon mask 及 time-to-go。Macro 候选描述和 Micro UAV 候选描述使用同一时点的预测场景与合法性合同。真实任务和当 Slot 的实际车辆位置只在动作采样之后交给环境执行，不进入 Actor 状态或在线候选价值输入。

### 车辆路线先验与研究车辆筛选边界

车辆预测器在每个预测起点 `origin_slot=t` 只使用截至 `t-1` 的轨迹观测，并沿 Episode 的 `vehicle_roles.json` 中为该研究车辆记录的 `planned_routes.route_edges` 投影名义位置。当前系统模型将该 SUMO 已分配路线视为决策时可用的路线先验；预测器不读取起点之后的真实轨迹来构造 Actor 输入。若目标部署无法获得已提交路线，这一先验就不成立，必须先修订信息可用性合同，再讨论替换预测方法。

正式数据生成只把在整个 Formal 区间每个 Slot 恰有一条有效轨迹记录、且在 Formal 起点前至少有一条有效 warm-up 记录的车辆纳入完整候选；随后按固定 seed 从中抽取研究车辆，再划分 SV 与任务车辆。因此当前车辆集合是整段 Episode 固定的完整轨迹 cohort，不覆盖 Formal 区间内新进入或提前离开的车辆，也不把背景车辆作为 Actor 的研究实体。该筛选适用于固定 cohort 的离线场景；动态车流成员变化需要额外的实体注册、存活 mask 和部分轨迹处理合同，当前实现不提供这类支持。

- Macro Cache Actor 在 Holding 边界一次性选择 Cache 配置。多节点因子化模式对每个 Cache 节点使用共享参数策略，以 KEEP 为候选 0，对合法 UPDATE 候选使用 mask；各节点在同一决策前状态下采样，联合 log-prob 是各节点 log-prob 之和。稀疏协同动作模式使用“全 KEEP 或一个节点 UPDATE”的组分布，再按所选节点采样合法候选。
- Micro UAV Actor 每 Slot 依稳定 UAV ID 顺序逐头采样。第一个头不依赖先前选择；后续头接收此前已选 UAV 坐标作为自回归条件。各头 log-prob 求和形成该 Slot 联合 log-prob。每个 UAV 都包含显式合法 STAY 候选。
- 自适应 Holding 由 Macro 策略中的 termination head 决定。配置合同为 1–5 Slot；在达到最短间隔后，策略读取当前 Slot 执行后的观测决定继续或结束，达到 5 Slot 时强制结束。`fixed2` 通过同一 Trainer 的 `adaptive_holding_enabled=false, min=max=2` 固定为两个 Slot，不维护第二个 Trainer。
- Episode 的真实终止 transition 不 bootstrap。完整训练与评价输入必须覆盖各自合同内的 Episode；不把数据不足或文件截断当作终止，也不从分区终点外读取 bootstrap Slot。

## Transition、回报与 GAE

`MicroTransition` 每 Slot 保存观测、各 UAV 的候选特征/合法 mask、动作、条件、冻结行为 log-prob、奖励、done 和执行指标。`HoldingTransition` 保存每次实际可采样的 Holding 终止决策、决策后观测、已持续 Slot 数、动作及行为概率。`MacroTransition` 保存 Cache 决策、真实持续时间 `d`、整段折扣奖励、终止观测、done 和对应的 Holding transitions。Rollout buffer 检查动作在保存 mask 下合法、Holding 索引递增、Macro 联合 log-prob 分解一致，以及终止与截断语义。

对于 Macro frame 起点 `t`、实际持续时间 `d`，奖励为

`R_t = Σ[k=0..d-1] γ^k r_(t+k)`。

SMDP TD bootstrap 使用 `γ^d V(s_(t+d))`；duration-aware GAE 的 frame trace 使用 `(γλ)^d`。Micro GAE 按普通逐 Slot `γ`、`λ` 递推。因而 Cache 更新与后续 Holding 长度会按真实持续时间计入目标，而不是假设固定两 Slot。训练时每个策略更新组的数据先完整采样，组内行为策略参数哈希保持冻结，再进行 PPO 更新。

## 合法候选与规模接口

正式多节点 MAPPO 的候选动作张量由运行时 Cache 节点数 `N`、服务数 `F`、节点候选数 `K` 及 UAV 数 `U` 构造；Actor 不要求固定九节点或两个 UAV。角色按运行时节点顺序编码，UAV 条件宽度为 `2 × max(1, U−1)`。观测和候选特征从正式 `ConfigurationObservationSpec`、`PrimitiveTensorSpec` 和当前服务目录派生。训练过程中或 Checkpoint 恢复时，若维度、角色顺序、静态 RSU/UAV 身份或其他模型合同不一致即拒绝加载。

正式多节点路径使用 `generate_bounded_cache_target_configurations`，按预测需求锚点、覆盖锚点、有限单项变化和有界交换生成候选，数量不超过配置的每节点候选预算 `K`；候选准备复杂度为 `O(F log F + KF)`，不会枚举全部 `2^F` 缓存子集。KEEP 永远保留为合法候选。

这些接口支持不同车辆、服务、Cache 节点和 UAV 数量形成新的合法状态/动作张量，但每种维度都会改变网络和 Checkpoint 合同；代码支持不等于已经训练或证明跨规模泛化。

## Checkpoint 与历史兼容

当前联合算法标识为 `ADAPTIVE_CACHE_UAV_JOINT_MAPPO`，架构按 Cache 动作模式使用独立 schema：

- 统一单节点动作：`adaptive_joint_cache_uav_mappo_v1_unified`，格式版本 64。
- 共享参数多节点因子化动作：`adaptive_joint_cache_uav_mappo_v1_factorized`，格式版本 65。
- 稀疏协同多节点动作：`adaptive_joint_cache_uav_mappo_v1_sparse`，格式版本 66。

Checkpoint 合同记录观测、候选特征、节点角色/数量、服务数、UAV ID/数量/条件宽度、Holding、奖励模型、动作概率、优化器、学习率、训练进度、Pool/config 标识与 Python/NumPy/PyTorch RNG 状态。严格续训必须逐项匹配。旧策略或不同维度 Checkpoint 不会静默映射为当前联合算法；只有显式权重初始化流程证明张量及语义兼容时才可迁移，且不得将重置过的优化器、训练进度或 RNG 称为严格续训。

## 检查边界

当前合同通过的必要检查包括：源码语法/导入、动态 UAV 联合 log-prob 的采样与 Replay 形状、合法 Cache mask、SMDP duration 索引、正式 Reward 逐 Slot 一致性和当前 schema 的 Checkpoint round-trip。单次短训练只验证执行链路，不作为策略学习效果或泛化结论。完整 Validation/Test 评价和跨规模训练属于后续实验，不在代码结构交付中自动运行。

正式冻结评价入口 `ppo-eval` 只接受 `environment_data/proposed_joint_validation/`。运行前会核对 Training 与 Validation Manifest 的 Episode seed/ID、源 Episode Manifest 哈希及共享 split Manifest，并把两份 Pool Manifest 哈希和实际选择的 Validation seed 写入结果；Training 或临时 Pool 会在读取 Episode 内容和创建输出目录前被拒绝。该入口不读取 Test Pool。
