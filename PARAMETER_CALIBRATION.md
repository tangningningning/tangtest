# 正式 baseline 系统规模与参数状态

本报告对应当前 `system_config.toml` 和 `data/generated/` 正式 artifact。
`system_config.toml` 是唯一正式参数入口。当前正式规模为 5 个 SV、20 个 TV、
2 个 RSU、2 个 UAV、1 个 Cloud 和 10 个 service；artifact 一致性状态见第 5 节。

## 1. 系统规模

| 项目 | 正式值 |
|---|---:|
| research vehicles | 25 |
| SV / TV | 5 / 20 |
| RSU | 2 (`RSU_0` ... `RSU_1`) |
| UAV | 2 (`UAV_0` ... `UAV_1`) |
| Cloud | 1 个共享节点 |
| service | 10 |

车辆角色来自已有真实 SUMO `trajectory.csv` 的完整轨迹筛选；本次未启动
SUMO、未生成合成车辆。角色划分仍使用固定 seed 选择，研究车辆均覆盖正式
300-slot episode 和 warm-up 历史。

## 2. 正式资源和成本参数

| 参数 | 值 |
|---|---:|
| TV local CPU | `1.0e9` cycle/s |
| SV CPU | `1.0e9` cycle/s |
| RSU CPU | `5.0e9` cycle/s |
| UAV CPU | `5.0e9` cycle/s |
| Cloud total CPU | `20.0e9` cycle/s |
| Cloud total bandwidth | `200.0e6` bit/s |
| TV / SV / RSU / UAV cache | `5581.583442 / 16744.750325 / 44652.667533 / 33489.500649` MB |
| `eta_ca` (SV / RSU / UAV) | `0.20 / 0.10 / 0.15` |
| `xi_ca` (SV / RSU / UAV) | `0.0010 / 0.0007 / 0.0012` |
| `objective.lambda_cache` | `0.1` |
| `psi_c` | `10.0` |

Cloud 仍是单节点有限共享模型：同一 slot 的 `N_cloud` 个任务分别获得
`20e9 / N_cloud` CPU 和 `200e6 / N_cloud` uplink；兼容字段
`nodes.cloud.cpu_cycles_per_s` 也统一为 `20e9`，不作为每任务独占资源。

reward、utility、`omega_T`、`omega_E`、`T_ref`、`E_ref`、movement model、
prediction model 和 cache cost 公式均未改动。

## 3. Service catalog 与 cache 比例

当前 `data/generated/service_catalog.json` 为 F=10，服务大小来自
2--20 GB 分层生成范围；artifact 同时保存 `service_size_gb`（GB）和运行时
兼容字段 `program_size`（MB）：

任务生成参数为：`arrival_probability = 0.7`，输入规模基准范围为
`500--5000 KB`，`cycles_per_bit = 400--1000`，
`input_relative_jitter = 0.20`，`load_relative_jitter = 0.10`；Zipf 和
popularity 演化过程不变。

| statistic | value (MB) |
|---|---:|
| min | 3372.996237 |
| median | 10776.092214 |
| mean | 11163.166883 |
| max | 19753.560543 |
| total | 111631.668831 |

| node role | capacity (MB) | capacity / catalog total |
|---|---:|---:|
| TV local | 5581.583442 | 5.000% |
| SV | 16744.750325 | 15.000% |
| RSU | 44652.667533 | 40.000% |
| UAV | 33489.500649 | 30.000% |

容量由 `cache.capacity_policy` 中的 5%/15%/40%/30% 自动乘以实际 catalog
总大小得到，严格满足 `RSU > UAV > SV > TV`，且任一节点都不能容纳全部 10 个
service。当前 TV 容量可完整容纳 1 个 service，因此 `tv_local_cache.json`
从可放入容量的服务集合中为每个 TV 固定放置 1 个 service，不读取未来 request；
运行时 cache optimizer 的容量约束和 placement 算法未修改。

catalog 中的 mean input size 基准实际为 839.586341--4902.258898 KB。
保留现有 ±20% input jitter 后，当前生成的有效任务观测范围为
673.640852--5878.270324 KB；理论有效 envelope 为 400--6000 KB。

## 4. Dynamic action / observation

cache node 列表按角色文件和配置实例动态构造：5 SV + 2 RSU + 2 UAV = 9
个 cache node。UAV movement action 由 2 个 UAV 实例动态构造。

正式 action 顺序为：

`[z_<all cache node>^ca, z_UAV_0^q, z_UAV_1^q]`

因此当前 `action_dim = 9 + 2 = 11`，不存在固定的旧 action schema。
以 9 cache node、10 service、2 UAV、25 research vehicle 构造 observation
时，`observation_dim = 9*10 + 2*2 + 25*2 + 10 = 164`。
observation 只包含 previous configuration、causal prediction state 和
固定 schema，不读取同 slot actual task、Matching/KKT、Q 或 G。

## 5. 生成与验证边界

本次已按依赖链重新生成 `vehicle_roles.json`、`service_catalog.json`、
`popularity.csv`、`popularity_history.csv`、`tasks.csv`、`tv_local_cache.json`、
`popularity_gru.pt`、`popularity_predictions.csv`、`vehicle_predictions.csv` 和
`prediction_scenarios.npz`；`node_positions.json` 保持现有 2 RSU + 2 UAV。
统一入口同时生成 `generation_metadata.json` 和
`reports/artifact_consistency_report.md`，一致性检查全部通过。
角色筛选复用了已有轨迹，预测场景只执行既有 GRU/场景生成流程；未运行 PPO、
完整 SUMO 仿真、pytest、benchmark 或额外训练实验。

本报告不表示 PPO 已接受；PPO 训练需在本 baseline 的配置、artifact、dynamic
action/observation 和 invariant validation 全部通过后另行进行。

本阶段不 commit、不 push。
