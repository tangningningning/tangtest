# Baseline v2.0 计算负载分析

## 1. 分析范围

本报告只读取当前 `system_config.toml`、`data/generated/service_catalog.json` 和
`data/generated/tasks.csv`，不启动 PPO、SUMO 或完整仿真。

计算时间使用既有执行模型的基本关系：

```text
T_compute = cpu_cycles / cpu_cycles_per_s
```

表中任务统计只包含 `arrival=1` 的有效任务；未到达行的 `input_size` 和
`cpu_cycles` 为零，不纳入平均值。

## 2. 任务负载统计

当前任务 artifact 共 6000 行，其中有效到达任务 4210 个。输入数据在任务行中以
bit 保存，报告换算为十进制 KB（`1 KB = 8000 bit`）。

| 指标 | 最小值 | 平均值 | 最大值 | 单位 |
|---|---:|---:|---:|---|
| `input_size` | 673.640852 | 2283.808027 | 5878.270324 | KB |
| `cycles_per_bit`（任务有效值） | 362.535299 | 724.742977 | 1088.993883 | cycle/bit |
| `cpu_cycles` | 2.331817e9 | 1.387014e10 | 3.938091e10 | cycle |

配置中的基准范围仍为 `input_size = 500--5000 KB`、
`cycles_per_bit = 400--1000 cycle/bit`。任务生成器保留了既有的
`input_relative_jitter = 0.20` 和 `load_relative_jitter = 0.10`，所以任务行的
实际 envelope 分别约为 `400--6000 KB` 和 `360--1100 cycle/bit`；本次观测值
落在该 envelope 内。这是扰动结果，不是配置单位错配。

## 3. 各类 CPU 执行时间

下表按同一有效任务集分别代入各类节点的 CPU 参数，表示单任务、未叠加通信和排队
时间的计算时间。

| 节点 | CPU 能力 (cycle/s) | 平均计算时间 (s) | 最大计算时间 (s) | 平均计算时间 (ms) | 最大计算时间 (ms) |
|---|---:|---:|---:|---:|---:|
| TV | 1.0e9 | 13.870136 | 39.380915 | 13870.136 | 39380.915 |
| SV | 1.0e9 | 13.870136 | 39.380915 | 13870.136 | 39380.915 |
| RSU | 5.0e9 | 2.774027 | 7.876183 | 2774.027 | 7876.183 |
| UAV | 5.0e9 | 2.774027 | 7.876183 | 2774.027 | 7876.183 |
| Cloud | 20.0e9 | 0.693507 | 1.969046 | 693.507 | 1969.046 |

Cloud 的 `20e9 cycle/s` 是单个共享 Cloud 节点总能力；当同一 slot 有多个 Cloud
任务时，既有模型按任务数均分 CPU，实际单任务计算时间还会随并发数增加。上表的
Cloud 数值是 `cloud_task_count=1` 的等效计算时间。

## 4. 负载判断

1. 单位和参数读取关系一致：service catalog 的输入与 cycles 参数进入 task
   artifact，`cache_configuration.py` 从配置读取 TV/SV/RSU/UAV/Cloud CPU 参数，
   执行公式保持 `cpu_cycles / cpu_rate`。
2. 从 `dt_s = 1 s` 的时间尺度看，TV/SV 的平均单任务计算时间约为 13.87 s，
   RSU/UAV 约为 2.77 s；这意味着车辆和边缘节点侧属于高计算压力配置，若任务
   连续落在同一节点，可能出现排队或 slot 内不可完成的情况。
3. Cloud 单任务等效计算时间低于 2 s，但 Cloud 共享能力会受到同 slot 并发任务
   数影响，不能把单任务结果直接当作高并发时延。

结论：当前 baseline v2.0 在单位、CPU 字段和执行公式上没有发现参数接口错误，
但任务负载相对于 1 s 基础 slot 偏重，适合先进行短时、小规模可行性验证并观察
排队/资源可行性；本报告不提出也不实施参数修改。

## 5. 代码证据

- `src/common/system_model.py`：本地、边缘和 Cloud 执行时间计算接口。
- `src/decision/cache_configuration.py`：从统一配置读取各类 CPU 参数。
- `src/decision/task_allocation.py`：Cloud 总 CPU 按同 slot 任务数共享。
- `system_config.toml`：当前 TV/SV=1e9、RSU/UAV=5e9、Cloud=20e9 cycle/s。
