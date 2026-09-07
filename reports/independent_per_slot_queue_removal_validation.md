# Independent-per-slot queue removal validation

日期：2026-09-06  
系统模型：`C:\Users\tangning\Desktop\12.pdf`（本项目唯一的系统模型来源）

## 结论

已将正式执行口径收敛为 `execution.cross_slot_task_model = "independent_per_slot"`：

- slot `t` 的到达任务在当前 slot 内完成卸载、资源分配、完整时延和车辆能耗评价；
- 同一 SV/RSU/UAV 上的任务仍通过现有 KKT 并行分配 CPU 与带宽，分别满足 `sum(w) <= 1`、`sum(b) <= 1`；
- 完成时延可以大于 `simulation.dt_s`，但不会形成下一个 slot 的任务状态、等待时间或资源占用；
- Local、SV、RSU、UAV、Cloud 均不保存 backlog、FIFO、queue wait 或剩余 workload；
- 下一 slot 只接收配置状态 `C(t)` / `Q^U(t)`、下一 slot 的任务和预测输入，不接收任务队列状态；
- 既有 `reports/computation_queue_cross_slot_audit.md` 等旧报告保留为历史审计记录，没有将其中的 FIFO 扩展继续作为正式实现。

## 代码与配置收敛

已移除任务队列参数、推进函数和跨 slot 传播：

- `src/decision/task_allocation.py`：删除 `advance_queue_workload`，Matching/KKT/full 与 incremental evaluator 不再接收 queue 参数；
- `src/decision/slot_environment.py`：`execute_slot` 和 `SlotExecutionResult` 不再有 previous/next queue 或 per-task queue wait；
- `src/decision/cache_configuration.py`：代理场景按 `scenario × horizon` 形成独立批次，不构造或传播 queue；
- `src/ppo/ppo_trigger_policy.py`：删除 queue observation slice 与 execution-node queue features；
- `src/ppo/ppo_rollout.py`、`src/ppo/ppo_training.py`、`src/ppo/ppo_update.py`：删除 rollout、snapshot、sanity 和 serialization 接口中的 queue 状态；
- `system_config.toml`：只保留 `independent_per_slot`，删除 `execution.slot_duration_source`；
- `src/common/project_config.py`：正式配置强制使用 `independent_per_slot`，不允许隐藏的 FIFO/default queue 语义。

源代码审计命令对 `src`、配置和校准文档中的
`queue_workload_s`、`queue_wait`、`next_queue`、`previous_queue`、
`deterministic_batch_fifo`、`slot_duration_source`、`backlog`、`FIFO` 和
`remaining workload` 均无任务队列语义残留；场景生成中的 `workload_variation`
仍只是任务输入/计算量扰动，不是队列。

## 正式公式与时序

当前 slot 的任务评价为：

- Local：`T_v = L_v / F_v`，`E_v^veh = kappa_v L_v F_v^2`；
- SV/RSU/UAV：`T_v = D_v / R_v + L_v / (w_v F_m)`，上传能耗为 `p_v D_v / R_v`；
- Cloud：`T_vc = psi_c D_v / R_c + L_v / F_c`，车辆能耗为 `p_v D_v / R_c`；
- 系统效用：`G(t) = Q(t) - C_ca(t) - C_q(t)`。

其中时延与能耗评价没有额外的 queue-wait 项。配置阶段仍使用预测输入，实际
Matching/KKT 使用当前 slot 的实际任务和实际车辆位置；这只是配置状态与任务执行
数据的边界，不是任务队列传播。

## 数值验证

### KKT 同节点并发

在 slot 20 取两个同时执行在 `RSU_0` 的任务 `64`、`19`，独立运行 KKT：

| 项目 | task 64 | task 19 | 节点和 |
|---|---:|---:|---:|
| CPU fraction `w` | 0.5248820480 | 0.4751179520 | 0.999999999999 |
| bandwidth fraction `b` | 0.5066984452 | 0.4933015548 | 0.999999999999 |
| delay (s) | 8.0647538712 | 7.7695434992 | — |
| vehicle energy (J) | 1.3698818960 | 1.3336628040 | — |

资源可行、时延和能耗均为有限值；没有 FIFO 串行等待。

### slot 20 full/incremental 与重复性

`task_allocation.run_slot20_validation("system_config.toml")` 处理 13 个到达任务，
结果如下：

- full 与 incremental matching 相同，`J_diff = 0`；
- delay、energy、`Q`、CPU allocation、bandwidth allocation 的最大差异均为 `0`；
- candidate 计数和 resource-infeasible 计数相同；
- 多任务边缘节点场景 `full_J = incremental_J = -59.41070448593542`，最大分配差异为 `0`；
- 重复运行确定性为 `True`，所有任务结果有限。

### 长时延与连续三个 slot

使用一条构造的长任务验证 Cloud 路径，得到三个连续 slot 的结果：

- target 均为 `CLOUD`；
- 每个 slot 的完整时延均为 `56000.0 s`，而 `dt = 1.0 s`；
- 三个 slot 的结果均有限；
- `SlotExecutionResult` 字段中不存在任何 queue/backlog 字段，下一 slot 没有残留任务 workload。

这验证了“完成时延可以超过 dt，但不占用下一 slot 资源”的最终口径。

### 实际、代理与联合配置接口

代理上下文构造结果为 20 个 scenario、100 个独立 `(scenario, horizon)` 批次、1413
个到达事件；每个批次使用自己的事件和位置快照，没有跨 horizon 的 queue 状态。
实际执行、代理评价、联合配置、rollout 和 formal episode 的接口签名均没有 queue、
backlog 或 remaining-work 参数。配置上下文与 cache config dataclass 也没有 queue 字段。

## Cloud 与 UAV 合同

配置与运行时强制：

- Cloud 每任务 `F_c = 5e9 cycle/s`、`R_c = 200e6 bit/s`；
- `nodes.cloud.resource_sharing = false`；Cloud 不参加 SV/RSU/UAV 的 edge KKT 资源和；
- Cloud 只在当前任务没有可执行的缓存 SV/RSU/UAV 节点时作为后备候选；
- `psi_c = 11` 只乘 Cloud 通信时延，不乘计算时延或车辆能耗；
- `cloud_execution(..., cloud_task_count=1)` 与 `cloud_task_count=7` 的所有返回值完全相同；
- 同一输入下 `psi=11` 与 `psi=1` 的计算时延和能耗相同，只改变通信时延；
- UAV 部署高度与空地信道高度统一为 `100.0 m`，配置校验强制两者一致。

## PPO 接口与依赖边界

实际 artifact 构造得到 9 个 cache/edge node、10 个 service、2 个 UAV、25 个 research
vehicle，因此：

`observation_dim = 9*10 + 2*2 + 25*2 + 10 = 154`，`action_dim = 9 + 2 = 11`。

纯 observation/action 接口验证通过：observation shape 为 `(154,)`、全为有限值，
action shape/dimension 为 `11`，ObservationSpec 没有 queue 字段。未运行 PPO 训练、GRU
训练或长 SUMO。`run_minimal_sanity` 的网络前向尝试因当前可用 Python 环境均未安装
PyTorch 而未执行；这属于环境依赖缺失，不是 observation/action 接口错误。

## Artifact 与代码检查

- `python -m compileall -q src`：通过；
- 正式 config validation：通过；
- `scripts/regenerate_artifacts.py --config system_config.toml --check-only`：通过，
  `writes_performed = 0`，全部 artifact、metadata 签名和 SHA256 检查通过；
- 为使配置依赖检查恢复一致，只更新了 `data/generated/generation_metadata.json` 中
  的 `config_sha256`，没有重生成任务、轨迹、预测或节点数据；
- `git diff --check`：通过；
- 未修改既有历史审计报告，也未清理工作区中原有的用户改动或 untracked 报告。

