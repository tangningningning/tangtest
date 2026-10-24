# 《13.pdf》跨时隙执行、Matching 与 DQN 对齐报告

**结论：实现已对齐到代码层；正式执行仍因缺少必要参数而阻塞。** 本报告不表示整个求解系统已通过论文模型验证。

## 论文合同

- 《13.pdf》第 6–8 页规定 Cloud 使用独立链路和每任务固定 `R_c`，计算节点提供每任务固定 `F_c`；不按共享池分配。Cloud 上传速率按 `R_c` 直接限制每时隙工作量。
- 第 7–9 页规定任务先上传、后计算；每时隙更新剩余数据、剩余计算量和剩余期限。成功或超时后任务退出活动集合，并结算实际执行时长和生命周期累计车辆上传能耗。
- 成功终止效用为 `ω_T(1-T_act/T_ref)+ω_E(1-E_life/E_ref)`；超时效用为 `-Φ_fail-ω_E E_life/E_ref`。`Q(t)` 只统计时隙末终止的任务。
- 系统时隙奖励为 `G(t)=Q(t)-C_ca(t)-C_q(t)`，固定扩展时域内求和。DQN Runner 已使用 `gamma=1.0`。

## 实现调整

- Matching 只允许新到达任务改变节点；活动任务节点固定到成功或超时。Transfer-Exchange 仍保留，生命周期模式下逐一评价候选的当前时隙终止任务效用。摘要记录匹配结果、局部稳定状态和合法性，并显式标记未声称全局最优。
- Edge 每时隙只给上传阶段任务分带宽，只给计算阶段任务分 CPU；上传完成后从下一时隙起计算。实际物理工作量按时隙长度和剩余工作量截断，状态账本逐时隙保存剩余上传量、剩余计算量、期限、固定节点和累计上传能耗。
- Edge KKT 仍用于固定节点、当前信道和阶段任务集合上的每节点倒数时延/能耗资源份额子问题。其凸性与 KKT 条件只适用于该连续代理子问题；它没有覆盖跨时隙剩余状态、期限触发的终止效用和完整离散 Matching，因此不能据此声称求得论文有限时域 P0 的最优解。Cloud 不运行 KKT。
- Cloud 候选评价和实际执行都使用每任务固定 `R_c/F_c`；不计共享 Cloud 容量。旧配置字段 `cloud_delay_factor` 不属于《13.pdf》这套 Cloud 速率模型，不再进入 `cloud_execution()`。
- DQN 每个 TV 固定保留 `K_max` 个活动任务槽，按规范 TV、到达时隙、任务 ID 排序；特征包括任务阶段、原始单位剩余 bit/cycle/期限、对应比例、服务 one-hot、执行节点 one-hot。决策前可见的本时隙新请求以 `UNASSIGNED` 节点表示，已活动任务保留固定节点。特征为 `float32` 且编码后校验有限值。
- 系统奖励只在任务终止时结算其效用，并通过任务 ID 集合拒绝重复结算；旧 Matching 即时效用只保留为诊断，不再作为终止任务奖励。系统成本按论文系数 1 聚合。
- DQN 状态、网络、Agent、Replay、候选重建、Runner manifest 和 checkpoint 合同版本已同步升级。旧 checkpoint 不会静默丢弃活动任务特征或沿用旧奖励口径，将被拒绝加载。

## 定向验证

在 Conda 环境 `adaptive_cache_mec` 中执行了修改模块的 `py_compile` 和一次内联合成状态验证，没有运行训练、正式评价或完整测试套件。验证通过的项目：

1. 成功与超时效用公式、系统奖励聚合；
2. 先上传后计算、剩余工作/期限更新、成功和超时各产生单次终止事件；
3. 活动任务固定 Cloud 节点、仅新任务可变；Transfer-Exchange 稳定性按局部范围报告；
4. Edge 上传/计算资源集合分离，带宽与 CPU 分配分别满足每节点份额和约束；
5. Cloud 候选资源和实际工作量使用相同的每任务 `R_c/F_c`；
6. 当前请求的 `UNASSIGNED` 特征、固定长度、`float32`、有限数值，以及网络输入维度；
7. 旧 Agent checkpoint schema 被拒绝。

合成用例为检验公式和接口使用了测试内存对象，不写入或替代正式系统参数，不构成正式数据或算法性能验证。

## 正式执行阻塞

- 正式配置 `configs/dqn/online_joint_dqn.toml` 仍为 `execution.cross_slot_task_model="independent_per_slot"`，并缺少 `execution.k_max_slots` 和 `objective.failure_penalty`。`13.pdf` 定义了 `K_max` 与 `Φ_fail` 的符号和公式，但当前正式配置未给出可用数值；实现不会自行补值。
- 正式 Episode 当前为 `T=300` 个 1 秒时隙，已有轨迹没有 `T` 之后的尾段。确定 `K_max` 后，需要按 `T+K_max-1` 覆盖任务执行时域；本次没有生成尾段或修改数据。
- 车辆在任务活动期间离开研究区域时的固定车辆集合语义仍需系统模型定义；现有数据能否覆盖未来尾时域尚不能核验。

因此正式 DQN 环境构造、跨时隙完整 Episode 和论文奖励目标均无法用当前正式配置验证。后续需权威确定 `K_max`、`Φ_fail`、将运行配置切换到跨时隙模型的安排，并提供符合扩展时域的车辆轨迹；若车辆离场可能发生，还需定义对应语义。

## 文件变更

修改了 `src/common/system_model.py`、`src/common/project_config.py`、`src/decision/cache_configuration.py`、`src/decision/task_allocation.py`、`src/decision/slot_environment.py`、`src/online_joint_dqn/candidate_features.py`、`src/online_joint_dqn/environment_adapter.py`、`src/online_joint_dqn/agent.py`、`src/online_joint_dqn/replay_buffer.py`、`src/online_joint_dqn/run_training.py`、`src/online_joint_dqn/trainer.py`、`src/online_joint_dqn/plot_training.py` 和 `src/online_joint_dqn/README.md`；新增 `src/decision/task_lifecycle.py` 与本报告。

未修改正式配置、路网、Train/Validation/Test 数据、历史实验结果或检查点；未提交或推送 Git 变更。
