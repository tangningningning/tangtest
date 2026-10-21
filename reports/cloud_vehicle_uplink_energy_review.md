# Cloud 车辆上行能耗公式核查

## 结论

当前代码按配置声明的抽象通信合同计算 Cloud 任务：车辆的基础上行使用固定的每任务等效速率 `R_c`；Cloud 通信时延再乘 `psi_c`，用于表示蜂窝基础上传之外的回传网络/核心网附加时延。配置明确说明该速率不代表实测无线链路，也没有指定实际接入点。因此，代码没有建模“车辆直接无线连接 Cloud”的物理链路，也没有建模车辆先到某个 RSU/UAV 再经独立 Cloud 回传链路的拓扑；实际物理接入路径仍无法从本项目配置确认。

在这个抽象合同下，现有能耗公式无需修改。车辆仅为无线传输的 radio airtime 付发射能耗；`psi_c` 代表的附加网络时延不应按车载无线发射功率计费。当前正式任务构造的 `transmit_time_share` 默认为 1，Cloud 上传时延和能耗使用同一个固定等效上行速率。

## 实际参数与调用链

- `configs/proposed/system_config.toml:139`：`nodes.vehicle.transmit_power_w = 0.1995262315 W`，单位为瓦。
- `D_bits`：实际任务从 `tasks.csv` 的 `input_size` 读入 `Task.input_bits`，不在 Matching 中再次转换；任务生成合同将 `input_size_unit` 定义为 `bit`（`src/scenario/task_scenario.py:82`）。
- `configs/proposed/system_config.toml:257-268`：`nodes.cloud.uplink_rate_bps = 20.0e6 bit/s`，`nodes.cloud.total_bandwidth_bps = 20.0e6 bit/s`，`cloud_delay_factor = 17.0`。配置注释说明 `total_bandwidth_bps` 是与上行速率相等的兼容字段，不代表共享 Cloud 带宽池。
- `configs/proposed/system_config.toml:330-332`：`communication.cloud.uplink_rate_bps = 20.0e6 bit/s`，必须与 `nodes.cloud.uplink_rate_bps` 一致。
- `src/common/project_config.py:1325-1338`：加载校验强制 `communication.cloud.uplink_rate_bps`、`nodes.cloud.uplink_rate_bps` 与兼容字段 `nodes.cloud.total_bandwidth_bps` 相等。
- `src/decision/cache_configuration.py:298-300`：将 `nodes.cloud.uplink_rate_bps` 和 `nodes.cloud.total_bandwidth_bps` 分别载入 `cloud_uplink_rate_bps`、`cloud_total_bandwidth_bps`。
- `src/decision/task_allocation.py:3730-3738`：实际 Matching/KKT 结果计算用的是 `cache_config.cloud_total_bandwidth_bps`。`communication.cloud.uplink_rate_bps` 不被实际上传公式直接读取，只用于配置一致性校验；`cloud_uplink_rate_bps` 当前还用于 Cache 策略的配置特征（`predictive_global_cache.py:360`）。由于配置校验要求三者相等，实际速率均为 20 Mbps。
- `src/common/system_model.py:221-259`：统一 Cloud 评价 helper 使用调用方传入的速率；时延为 `psi_c * D / R_c`，车辆能耗为 `P_v * D / R_c`。`task_allocation.py:1935-1941` 将 Cloud 固定等效速率、时延系数和车辆发射功率传入该 helper。

Cloud 上传速率是每任务固定等效速率，不按 Cloud 任务数重新分摊；当前配置禁止 Cloud uplink sharing。Cloud CPU 使用有限共享池的 KKT 分配，但 Cloud uplink 不进入 Edge 的无线带宽 KKT 分配。Edge 则通过 `src/decision/task_allocation.py:3487-3508` 使用各 SV/RSU/UAV 的位置相关信道增益和 Shannon uplink 速率，并由 `edge_execution()` 以 `P_v * upload_delay` 计车辆发射能耗。因此两条执行路径使用不同的链路速率模型，但都只按车辆 radio airtime 计能耗。

## 公式与量纲

当前 Cloud 公式（修改前后相同）：

```text
radio_airtime_s       = D_bits / R_c
upload_wall_s         = D_bits / (R_c * q)
cloud_communication_s = psi_c * upload_wall_s
vehicle_energy_j      = P_v * D_bits / R_c
```

`q` 是任务发送时间份额，`Task` 默认 `q=1`（`task_allocation.py:1524-1531`，实际 CSV loader 不覆盖该默认值）。代码中 Cloud 延迟按 `1/q` 放大，而能耗不按 `1/q` 放大；这对应时间份额为 `q` 时的 radio airtime `q * upload_wall_s = D/R_c`。项目对 Edge KKT 的注释也说明时间份额同时缩放 wall-clock 速率与平均发射功率（`task_allocation.py:2909`）。当前 Cloud 的实际 `q=1`，无需依赖非默认份额行为来得出本次结论。

单位成立：`bit / (bit/s) = s`；`W * s = J`。`psi_c` 只放大模型中的 Cloud 通信时延，不进入车辆无线发射能耗。若错误地用 `P_v * psi_c * D/R_c`，会把模型中的回传/核心网附加延迟也计作车辆无线发射时间。

按正式配置，取十进制 `1 MB = 8.0e6 bit`、`q=1`：基础 radio airtime 为 `0.4 s`；模型 Cloud 通信时延为 `17 * 0.4 = 6.8 s`；车辆上传能耗为 `0.1995262315 * 0.4 = 0.0798104926 J`。能耗等于 `P_v * radio_airtime_s`，不是 `P_v * cloud_communication_s`。

## Matching 与 Reward 是否重复计能耗

Matching 的 Cloud 和 Edge 评价均将每个任务的 `vehicle_energy_j` 传入 `system_model.task_utility()`，作为任务效用中的能耗项。完整匹配评价汇总一次 `task_utility_sum`；`src/decision/slot_environment.py:616-630` 将其作为 `actual_q`，再计算 `system_utility = actual_q - cache_cost - movement_cost`。Reward 没有再单独扣一次车辆能耗，因此同一段车辆上传能耗没有在 Reward 中重复扣除。Cloud 回传/核心网也没有独立的时延、无线能耗或能耗参数；其附加时延只通过 `cloud_delay_factor` 表示。

## 报告字段

```ini
CLOUD_COMMUNICATION_PATH = Cloud 是逻辑执行目标；车辆上行使用固定等效速率，psi_c 表示蜂窝基础上传之外的回传/核心网附加时延；未建模具体 RSU/UAV 接入路径，真实物理路径 UNRESOLVED
CLOUD_UPLINK_RATE_SEMANTICS = nodes.cloud.uplink_rate_bps 和 communication.cloud.uplink_rate_bps 均为相同的每任务固定等效上行速率；实际 Matching 用 nodes.cloud.total_bandwidth_bps 兼容字段，加载校验保证三者相等
VEHICLE_UPLINK_ENERGY_FORMULA_BEFORE = P_v * D_bits / R_c；q=1 时为 P_v * radio_airtime_s；psi_c 不进入能耗
VEHICLE_UPLINK_ENERGY_FORMULA_AFTER = P_v * D_bits / R_c；公式正确，无源码修改
FORMULA_PHYSICALLY_CONSISTENT = YES_WITH_DECLARED_ABSTRACT_MODEL; 实际无线接入点和实测 Cloud 链路未由配置确认
BACKHAUL_DELAY_MODELED_SEPARATELY = NO；回传/核心网附加时延由 psi_c 乘数表示，没有独立链路速率、传播时延或能耗
DOUBLE_COUNTING_FOUND = NO；车辆能耗只进入任务效用一次，Slot Reward 不再次扣除
SOURCE_OF_EACH_PARAMETER = D_bits: tasks.csv 的 input_size，由 load_actual_tasks 读入，任务合同单位 bit；P_v: nodes.vehicle.transmit_power_w；R_c: nodes.cloud.uplink_rate_bps（实际 Matching 使用相等的 nodes.cloud.total_bandwidth_bps）；communication.cloud.uplink_rate_bps: 相等性校验字段；psi_c: nodes.cloud.cloud_delay_factor；q: Task.transmit_time_share，正式 loader 默认 1
CODE_MODIFIED = NO
FORMAL_CONFIG_VALUES_CHANGED = YES
MAPPO_TRAINING_EXECUTED = NO
VALIDATION_TEST_READ = NO
COMMIT_PUSH = NO
```

## 验证

- 使用项目解释器从当前 `system_config.toml` 读取配置并核算上述 1 MB 单位示例：三个速率字段均为 `20,000,000 bit/s`，能耗为 `0.0798104926 J`，与 `P_v * radio_airtime_s` 相同。
- 对本次关联的 Python 源文件执行 AST 语法解析：通过；未执行 Episode、Matching/KKT 求解或训练。
- `git diff --check`：通过。
- 未修改源码或正式配置。新增本报告用于记录核查结果。
