# 正式 baseline 系统规模与参数状态

本报告对应当前 `system_config.toml` 和 `data/generated/` 正式 artifact。
当前唯一正式规模为 5 个 SV、25 个 TV、3 个 RSU、3 个 UAV、1 个 Cloud
和 20 个 service。旧的 3 SV / 9 TV / 1 RSU / 2 UAV / 10 service
不再作为正式 baseline。

## 1. 系统规模

| 项目 | 正式值 |
|---|---:|
| research vehicles | 30 |
| SV / TV | 5 / 25 |
| RSU | 3 (`RSU_0` ... `RSU_2`) |
| UAV | 3 (`UAV_0` ... `UAV_2`) |
| Cloud | 1 个共享节点 |
| service | 20 |

车辆角色来自已有真实 SUMO `trajectory.csv` 的完整轨迹筛选；本次未启动
SUMO、未生成合成车辆。角色划分仍使用固定 seed 选择，研究车辆均覆盖正式
300-slot episode 和 warm-up 历史。

## 2. 正式资源和成本参数

| 参数 | 值 |
|---|---:|
| SV CPU | `1.0e9` cycle/s |
| RSU CPU | `8.0e9` cycle/s |
| UAV CPU | `5.0e9` cycle/s |
| Cloud total CPU | `20.0e9` cycle/s |
| Cloud total bandwidth | `200.0e6` bit/s |
| SV / RSU / UAV cache | `6000 / 15000 / 10000` MB |
| `eta_ca` (SV / RSU / UAV) | `0.20 / 0.10 / 0.15` |
| `xi_ca` (SV / RSU / UAV) | `0.0010 / 0.0007 / 0.0012` |
| `objective.lambda_cache` | `0.1` |
| `psi_c` | `2.30` |

Cloud 仍是单节点有限共享模型：同一 slot 的 `N_cloud` 个任务分别获得
`20e9 / N_cloud` CPU 和 `200e6 / N_cloud` uplink。旧 Cloud 标量配置仅为
兼容字段，不作为每任务独占资源。

reward、utility、`omega_T`、`omega_E`、`T_ref`、`E_ref`、movement model、
prediction model 和 cache cost 公式均未改动。

## 3. Service catalog 与 cache 比例

当前 `data/generated/service_catalog.json` 为 F=20，服务大小来自原有
1,000--2,000 MB 分层生成范围：

| statistic | value (MB) |
|---|---:|
| min | 1038.138784 |
| median | 1515.443138 |
| mean | 1500.296463 |
| max | 1989.139878 |
| total | 30005.929269 |

| node role | capacity (MB) | capacity / catalog total |
|---|---:|---:|
| SV | 6000 | 19.996% |
| RSU | 15000 | 49.990% |
| UAV | 10000 | 33.327% |

按实际 service size 的 exact subset sanity，单个 SV、RSU、UAV 最多可分别
完整容纳 5、11、8 个 service；任一节点都不能容纳全部 20 个 service。
TV 本地 cache 仍为 episode 前生成并固定，每个 TV 放置一个完整 service，
容量为 2000 MB，不读取未来 request。

## 4. Dynamic action / observation

cache node 列表按角色文件和配置实例动态构造：5 SV + 3 RSU + 3 UAV = 11
个 cache node。UAV movement action 由 3 个 UAV 实例动态构造。

正式 action 顺序为：

`[z_<all cache node>^ca, z_UAV_0^q, z_UAV_1^q, z_UAV_2^q]`

因此当前 `action_dim = 11 + 3 = 14`，不存在旧的固定 8-action schema。
以 11 cache node、20 service、3 UAV、30 research vehicle 构造 observation
时，`observation_dim = 11*20 + 3*2 + 30*2 + 20 = 306`。
observation 只包含 previous configuration、causal prediction state 和
固定 schema，不读取同 slot actual task、Matching/KKT、Q 或 G。

## 5. 生成与验证边界

本次按新规模重新生成了 service catalog、popularity、tasks、TV local cache、
F=20 GRU checkpoint 和 formal prediction scenarios；复用了已有独立
calibration residual bank。未运行 PPO、SUMO、pytest、benchmark 或 300-slot
trajectory scan。

本报告不表示 PPO 已接受；PPO 训练需在本 baseline 的配置、artifact、dynamic
action/observation 和 invariant validation 全部通过后另行进行。

本阶段不 commit、不 push。
