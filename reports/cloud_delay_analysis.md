# Cloud Offloading Delay 参数分析

## 1. 变更范围

本次只调整 Cloud 传输扩展系数 `psi_c`，不修改 Cloud CPU、Cloud 带宽、其他通信参数、reward 公式或任务 artifact。项目已有正式配置键 `nodes.cloud.cloud_delay_factor`，其语义对应公式中的 `psi_c`；为避免引入第二套配置，继续使用该统一入口。

| 参数 | 修改前 | 修改后 | 说明 |
|---|---:|---:|---|
| `nodes.cloud.cloud_delay_factor`（`psi_c`） | 10.0 | **11.0** | 仅影响 Cloud 上传/通信时延项 |
| Cloud total CPU | 20.0e9 cycle/s | 20.0e9 cycle/s | 保持不变 |
| Cloud total bandwidth | 200.0e6 bit/s | 200.0e6 bit/s | 保持不变 |
| `objective.omega_delay`（alpha(delay)） | 0.5 | **0.6** | 按追加 baseline 要求调整目标权重 |
| `objective.omega_energy`（beta(energy)） | 0.5 | **0.4** | 按追加 baseline 要求调整目标权重 |

目标权重只影响目标函数中的时延/能耗组合，不改变 Cloud latency 的物理计算公式。

## 2. Cloud offloading delay 公式

当前实现使用：

```text
T_cloud = psi_c * D / R_cloud_i + L / F_cloud_i
```

其中：

- `D`：任务输入大小，单位 bit；
- `L`：任务计算量，单位 cycle；
- `R_cloud_i = 200.0e6 / N_cloud` bit/s；
- `F_cloud_i = 20.0e9 / N_cloud` cycle/s；
- `N_cloud`：同一 slot 共享 Cloud 资源的任务数。

`psi_c` 只乘在通信/上传时延项上，因此：

```text
T_comm(11) = 1.1 * T_comm(10)
T_compute(11) = T_compute(10)
```

公式已由 `src/common/system_model.py::cloud_execution()` 实现，并由 Cloud 任务分配、缓存配置和 Cloud 资源审计路径传入当前配置值。

## 3. 分析数据与口径

本次采用当前 `data/generated/tasks.csv` 中的 4210 个 arrival 任务，覆盖 300 个 slot。为隔离 `psi_c` 对 latency 的影响，按“每个 slot 的 arrival 任务全部共享 Cloud”进行公式级计算：

- 最大同 slot Cloud 任务数：19；
- 平均同 slot 任务数：14.0333；
- 未运行任务分配、PPO、SUMO 或完整仿真；
- 修改前一列使用 `psi_c=10.0`，修改后一列使用当前配置 `psi_c=11.0`；
- Cloud CPU 和带宽分摊规则两列完全相同。

## 4. 修改前后 Cloud latency

### 4.1 通信与计算分项

| 指标 | `psi_c=10.0` | `psi_c=11.0` | 变化 |
|---|---:|---:|---:|
| 平均 communication delay (s) | 13.073721 | 14.381093 | +1.307372 (+10.000%) |
| communication median (s) | 10.626633 | 11.689297 | +1.062663 |
| communication P95 (s) | 27.891536 | 30.680689 | +2.789154 |
| communication max (s) | 42.016883 | 46.218571 | +4.201688 |
| 平均 compute delay (s) | 9.928811 | 9.928811 | 0 |
| compute median (s) | 9.021361 | 9.021361 | 0 |
| compute P95 (s) | 22.352222 | 22.352222 | 0 |
| compute max (s) | 31.929897 | 31.929897 | 0 |

### 4.2 总 Cloud offloading delay

| 指标 | `psi_c=10.0` | `psi_c=11.0` | 变化 |
|---|---:|---:|---:|
| 平均 total latency (s) | 23.002532 | 24.309904 | +1.307372 (+5.684%) |
| median total latency (s) | 19.962656 | 21.015522 | +1.052866 |
| P95 total latency (s) | 49.298578 | 52.046219 | +2.747641 (+5.573%) |
| max total latency (s) | 72.307310 | 76.484411 | +4.177101 (+5.777%) |

## 5. 结论

1. `psi_c` 已从 10.0 调整为 11.0，并通过现有 `cloud_execution()` 接入 `psi_c * D / R_cloud_i`；未新增第二套 Cloud 参数。
2. 通信时延按理论预期增加 10%；计算时延保持完全不变。
3. 在当前任务和 Cloud 共享资源口径下，总平均 Cloud latency 增加约 1.307372 s，即 5.684%。
4. Cloud CPU、Cloud 带宽、其他通信参数和 reward 实现未修改。

## 6. 修改文件与证据

- `system_config.toml`：更新 Cloud `psi_c` 的统一配置入口，并将目标权重设为 `omega_delay=0.6`、`omega_energy=0.4`。
- `PARAMETER_CALIBRATION.md`：同步 `psi_c=11.0`。
- `reports/cloud_delay_analysis.md`：本报告。
- `src/common/system_model.py`：未修改；现有公式已接收并使用该参数。
- `src/decision/task_allocation.py`、`src/decision/cache_configuration.py`：未修改；现有 Cloud 调用链已传递该参数。

