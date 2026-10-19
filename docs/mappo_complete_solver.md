# 自适应 Cache–UAV 联合 MAPPO 合同

## 正式实现与入口

当前正式学习器为 `src/ppo_simple/trainer.py:PPOTrainer`，训练入口为 `python -m ppo_simple.trainer` / `ppo-train`，冻结评价入口为 `python -m ppo_simple.evaluator` / `ppo-eval`。`scripts/training/mainline_runner.py` 与 `src/ppo_simple/stage3_cli.py` 只转发到正式 Trainer。旧 Stage3 MAPPO 与反事实监督实现保留作历史实验读取和复现，不是当前正式 Trainer 的第二条更新路径。

执行仍由正式环境完成：Cache 配置进入 `slot_environment.execute_configured_slot`，任务匹配与资源分配继续使用原 Matching/KKT，奖励继续来自正式 `SlotExecutionResult.system_utility`。本改动没有改变任务效用、Cache 配置成本、UAV 移动成本或奖励公式。

## 决策信息与双时间尺度

每次决策状态包含当前 Cache 矩阵、当前 UAV 坐标、决策时可用的车辆轨迹预测、服务流行度预测、预测 horizon mask 及 time-to-go。Macro 候选描述和 Micro UAV 候选描述使用同一时点的预测场景与合法性合同。真实任务和当 Slot 的实际车辆位置只在动作采样之后交给环境执行，不进入 Actor 状态或在线候选价值输入。

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

正式多节点路径使用 `generate_bounded_cache_target_configurations`，按预测需求锚点、覆盖锚点、有限单项变化和有界交换生成候选，数量不超过配置的每节点候选预算 `K`；候选准备复杂度为 `O(F log F + KF)`，不会枚举全部 `2^F` 缓存子集。KEEP 永远保留为合法候选。历史 Stage3 需要的旧候选生成路径仍与这一正式有界生成器分离。

这些接口支持不同车辆、服务、Cache 节点和 UAV 数量形成新的合法状态/动作张量，但每种维度都会改变网络和 Checkpoint 合同；代码支持不等于已经训练或证明跨规模泛化。

## Checkpoint 与历史兼容

当前联合算法标识为 `ADAPTIVE_CACHE_UAV_JOINT_MAPPO`，架构按 Cache 动作模式使用独立 schema：

- 统一单节点动作：`adaptive_joint_cache_uav_mappo_v1_unified`，格式版本 64。
- 共享参数多节点因子化动作：`adaptive_joint_cache_uav_mappo_v1_factorized`，格式版本 65。
- 稀疏协同多节点动作：`adaptive_joint_cache_uav_mappo_v1_sparse`，格式版本 66。

Checkpoint 合同记录观测、候选特征、节点角色/数量、服务数、UAV ID/数量/条件宽度、Holding、奖励模型、动作概率、优化器、学习率、训练进度、Pool/config 标识与 Python/NumPy/PyTorch RNG 状态。严格续训必须逐项匹配。历史 Stage3、旧 PROPOSED 或不同维度 Checkpoint 不会静默映射为当前联合算法；只有显式权重初始化流程证明张量及语义兼容时才可迁移，且不得将重置过的优化器、训练进度或 RNG 称为严格续训。

Stage3 的反事实辅助监督仍作为历史研究机制保留，当前 MAPPO PPO 更新不依赖该监督，不包含独立监督价值网络或新增 Candidate Loss。

## 检查边界

当前合同通过的必要检查包括：源码语法/导入、动态 UAV 联合 log-prob 的采样与 Replay 形状、合法 Cache mask、SMDP duration 索引、正式 Reward 逐 Slot 一致性和当前 schema 的 Checkpoint round-trip。单次短训练只验证执行链路，不作为策略学习效果或泛化结论。完整 Validation/Test 评价和跨规模训练属于后续实验，不在代码结构交付中自动运行。
