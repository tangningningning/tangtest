# 系统参数校准状态

本报告记录当前系统参数层的冻结状态、配置成本尺度口径和 formal validation 剩余阻塞项。本阶段没有修改任何 `data/generated/` 正式数据。

## 1. Q_slot_ref 统一口径

`Q_slot_ref` 只用于配置成本尺度校准，不写入 `system_config.toml`。

统一计算口径：

- 任务集合：`data/generated/tasks.csv` 中 `arrival=1` 的真实任务。
- slot 集合：按真实 `slot` 聚合，不包含 `arrival=0` 任务。
- execution reference：使用当前 Cloud reference mode 逐任务计算 `Q_cloud`。
- 当前 Cloud 参数：`F_c=3.0e9 cycle/s`、`R_c=200.0e6 bit/s`、`psi_c=2.30`。
- 统计方式：每个 slot 内真实 arrival 任务的 `Q_cloud` 求和，再取 slot median。

当前统一结果：

`Q_slot_ref = 3.544207`

历史差异原因：

| 口径 | Cloud 参数 | slot median utility |
|---|---|---:|
| 旧 Cloud 口径 | `F_c=20GHz`, `R_c=200Mbps`, `psi_c=0.05` | 3.916001 |
| 更早旧口径 | `F_c=20GHz`, `R_c=20Mbps`, `psi_c=0.05` | 3.738972 |
| 当前统一口径 | `F_c=3GHz`, `R_c=200Mbps`, `psi_c=2.30` | 3.544207 |

因此，`3.916 -> 3.544207` 的差异来自 Cloud 参数语义修正，不是 arrival 集合、mean/median 混用、arrival=0 任务或 best-mode 选择导致。

## 2. 当前系统参数冻结状态

### 已冻结系统参数

| parameter | value | unit | basis |
|---|---:|---|---|
| `objective.T_ref` | 0.65 | s | generated data |
| `objective.E_ref` | 0.065 | J | generated data |
| `objective.omega_delay` | 0.5 | - | system-scale calibrated baseline |
| `objective.omega_energy` | 0.5 | - | system-scale calibrated baseline |
| `nodes.vehicle.local_cpu_cycles_per_s` | 1.0e9 | cycle/s | direct literature baseline |
| `nodes.vehicle.kappa` | 1.0e-28 | - | direct literature baseline |
| `nodes.sv.cpu_cycles_per_s` | 3.0e9 | cycle/s | direct literature baseline |
| `nodes.rsu.cpu_cycles_per_s` | 8.0e9 | cycle/s | direct literature baseline |
| `nodes.uav.cpu_cycles_per_s` | 5.0e9 | cycle/s | direct literature baseline |
| `nodes.uav.bandwidth_hz` | 10.0e6 | Hz | direct literature baseline |
| `nodes.uav.altitude_m` | 100.0 | m | direct literature simulation baseline |
| `nodes.uav.max_horizontal_speed_mps` | 15.0 | m/s | direct literature simulation baseline |
| `nodes.uav.movement.movement_energy_coefficient_j_per_m` | 15.0 | J/m | literature rotor-model equivalent mapping |
| `nodes.cloud.cpu_cycles_per_s` | 3.0e9 | cycle/s | literature-derived mapping |
| `nodes.cloud.uplink_rate_bps` | 200.0e6 | bit/s | direct literature baseline |
| `nodes.cloud.cloud_delay_factor` | 2.30 | - | literature-derived mapping |
| `communication.air.eta1` | 9.61 | - | published-literature Urban baseline |
| `communication.air.eta2` | 0.16 | - | published-literature Urban baseline |
| `communication.air.carrier_frequency_hz` | 2.0e9 | Hz | direct literature simulation baseline |
| `communication.air.beta_los` | 1.1302166e-4 | power gain | literature-derived mapping |
| `communication.air.beta_nlos` | 1.4228584e-6 | power gain | literature-derived mapping |
| `communication.air.alpha_los` | 2.0 | - | literature-derived model mapping |
| `communication.air.alpha_nlos` | 2.0 | - | literature-derived model mapping |
| `communication.ground.bandwidth_hz` | 10.0e6 | Hz | direct literature baseline |
| `nodes.vehicle.transmit_power_w` | 0.1995262315 | W | literature-derived mapping |
| `communication.noise.noise_power_w` | 3.16227766e-13 | W | literature-derived mapping |
| `communication.ground.channel_gain_at_1m` | 1.6368e-5 | power gain | literature-derived mapping |
| `communication.ground.pathloss_exponent` | 2.75 | - | direct literature baseline |
| `objective.lambda_cache` | 0.1 | - | system-scale calibrated baseline |
| `objective.lambda_movement` | 0.002 | - | system-scale calibrated baseline |
| `nodes.uav.movement.movement_fixed_cost` | 0.2 | cost | system-scale calibrated baseline |

### Cache capacity 与 cache cost baseline

当前 10 个 service 总大小：

`total_service_size = 15851.373 MB`

| node | capacity MB | total ratio | max actual services | smallest-services count | largest-services count | status |
|---|---:|---:|---:|---:|---:|---|
| SV | 4500.0 | 28.39% | 3 | 3 | 2 | CONFIRMED_DATA |
| UAV | 6000.0 | 37.85% | 4 | 4 | 3 | CONFIRMED_DATA |
| RSU | 8000.0 | 50.47% | 5 | 5 | 4 | CONFIRMED_DATA |

三类节点均无法缓存全部 10 个 service，且保持 `K_SV < K_UAV < K_RSU < sum_f s_f`。这些容量是 system-scale experimental baseline，不是某篇论文直接给出的硬件容量。

| node | eta_ca | xi_ca | fixed / Q_slot_ref | median add / Q_slot_ref | max add / Q_slot_ref | status |
|---|---:|---:|---:|---:|---:|---|
| SV | 0.20 | 0.0010 | 5.64% | 10.08% | 10.98% | CONFIRMED_DATA |
| RSU | 0.10 | 0.0007 | 2.82% | 5.93% | 6.56% | CONFIRMED_DATA |
| UAV | 0.30 | 0.0012 | 8.46% | 13.79% | 14.87% | CONFIRMED_DATA |

`xi_ca` 在当前项目中表示单位 service 加载量对应的等效 cache loading coefficient，单位为 cost/MB；不解释为严格物理 Joule/MB。

### TV固定本地缓存初始状态

最新 `11.pdf` 规定：任务车辆本地缓存 `c_vf(0)` 在优化时域开始前生成，生成后整个优化时域固定，不属于在线优化变量；Local 执行仅在 `c_{v,f_v(t)}(0)=1` 时可行，且 TV local cache 不产生在线 cache update cost。

当前正式 `data/generated/service_catalog.json` 中：

- `F = 10`
- `min_service_size = 1244.2733515029472 MB`，service `1`
- `second_min_service_size = 1322.8042118988528 MB`，service `8`
- `min_two_service_sum = 2567.0775634018 MB`
- `max_single_service = 1890.4713524956487 MB`
- `total_service_size = 15851.373045559802 MB`

因此采用：

`nodes.vehicle.local_cache_capacity_mb = 2000.0`

该容量满足：

`max_single_service < 2000MB < min_two_service_sum`

所以当前 10 个 service 中任意单个完整 service 都能放入 TV local cache，而任意两个完整 service 都不能同时放入。当前设计等价为每辆 TV 恰好缓存 1 个 service。

参数性质：`2000MB` 是 system-scale experimental baseline，由当前正式 service catalog 推导得到；不是文献直接硬件参数，也不是 11.pdf 直接给定值。

生成规则：

- artifact：`data/generated/tv_local_cache.json`
- TV ID 来源：`data/generated/vehicle_roles.json`
- service ID 来源：`data/generated/service_catalog.json`
- RNG seed：`experiment.seed = 4401`
- 规则：对每个 TV 按稳定 TV ID 顺序，从全部 service IDs 中 uniform random 选择恰好 1 个 service
- 不使用 `tasks.csv` 全 episode 真实请求频率，不使用 future ground truth，不使用 popularity/GRU 预测
- 相同 seed、TV 集合、service catalog 下完全复现

当前生成结果：

| TV | cached service | used capacity MB |
|---|---:|---:|
| 12 | 3 | 1890.471352 |
| 30 | 3 | 1890.471352 |
| 40 | 8 | 1322.804212 |
| 42 | 4 | 1469.153117 |
| 59 | 4 | 1469.153117 |
| 66 | 9 | 1618.283315 |
| 70 | 0 | 1810.220550 |
| 78 | 2 | 1528.427194 |
| 96 | 3 | 1890.471352 |

只读 sanity：当前 `tasks.csv` 中 arrival=1 的 1344 个真实任务里，TV local-cache availability 为 `120/1344 = 8.928571%`。该统计只用于检查 artifact 结构，不反向影响 local cache 生成。

### UAV movement cost baseline

当前：

- `xi_mov = 15 J/m`
- `lambda_movement = 0.002`
- `eta_q = 0.2`

单 UAV：

| distance | cost | cost / Q_slot_ref |
|---:|---:|---:|
| 1m | 0.230 | 6.49% |
| 5m | 0.350 | 9.88% |
| 10m | 0.500 | 14.11% |
| 15m | 0.650 | 18.34% |

双 UAV：

| scenario | total cost | cost / Q_slot_ref |
|---|---:|---:|
| two UAV each 5m | 0.700 | 19.75% |
| two UAV each 10m | 1.000 | 28.22% |
| two UAV each 15m | 1.300 | 36.68% |

`movement_energy_coefficient_j_per_m=15.0` 是线性等效参数，不是文献直接给出的固定物理常数。Li et al., *Collaborative offloading for UAV-enabled time-sensitive MEC networks*, DOI `10.1186/s13638-020-01861-8`，Table 1 给出了旋翼 UAV 功率模型所需参数：

- `P0 = 158.76 W`
- `Pi = 88.63 W`
- `U_tip = 120 m/s`
- `rho = 1.225 kg/m^3`
- `A = 1 m^2`
- `v0 = 4.03 m/s`
- `d0 = 0.3`
- `s = 0.05`

旋翼功率模型：

`P(V)=P0(1+3V^2/U_tip^2)+Pi(sqrt(1+V^4/(4v0^4))-V^2/(2v0^2))^(1/2)+0.5*d0*rho*s*A*V^3`

在 `V_ref=15m/s` 处：

- `P(15) = 220.960140 W`
- `P(15)/15 = 14.730676 J/m`

当前项目采用 `11.pdf` 的线性移动能耗形式 `E_mov = xi_mov * distance`，因此将 `P(15)/15` 取整为 `xi_mov = 15 J/m`。该参数状态升级为 `CONFIRMED_DATA`，证据类型为“文献旋翼功率模型到当前线性移动能耗模型的等效映射”。

### Joint configuration sanity

| scenario | C_ca | C_q | C_total | C_total / Q_slot_ref |
|---|---:|---:|---:|---:|
| SV median cache add | 0.357336 | 0.000000 | 0.357336 | 10.08% |
| UAV median cache add | 0.488803 | 0.000000 | 0.488803 | 13.79% |
| UAV move 10m | 0.000000 | 0.500000 | 0.500000 | 14.11% |
| UAV median cache add + move 10m | 0.488803 | 0.500000 | 0.988803 | 27.90% |
| two UAV each move 10m | 0.000000 | 1.000000 | 1.000000 | 28.22% |
| SV median add + RSU median add + UAV move 10m | 0.567470 | 0.500000 | 1.067470 | 30.12% |

没有发现单节点普通配置动作超过 50% 的明显尺度失衡。

## 3. UAV air-channel 与 movement 证据状态

最新 `11.pdf` 确认了 UAV 固定高度、水平位置优化、UAV 最大速度约束、V2U LoS/NLoS 概率模型、平均空地信道增益公式，以及系统成本中的 `E_mov = xi_mov * distance` 形式。本阶段补齐以下参数证据链，并将其升级为 `CONFIRMED_DATA`。

| parameter | value | unit | evidence type | calibration note |
|---|---:|---|---|---|
| `nodes.uav.altitude_m` | 100.0 | m | direct literature simulation baseline | Li et al. Table 1: UAV altitude baseline |
| `nodes.uav.max_horizontal_speed_mps` | 15.0 | m/s | direct literature simulation baseline | Li et al. Table 1: maximum UAV speed baseline |
| `communication.air.eta1` | 9.61 | - | published-literature Urban baseline | LoS S-curve model follows Al-Hourani et al.; `9.61/0.16` uses later Urban simulation baseline, not a direct Al-Hourani table claim |
| `communication.air.eta2` | 0.16 | - | published-literature Urban baseline | same as above |
| `communication.air.carrier_frequency_hz` | 2.0e9 | Hz | direct literature simulation baseline | common air-to-ground simulation carrier; also matches Al-Hourani 2000MHz pathloss example |
| `communication.air.alpha_los` | 2.0 | - | literature-derived model mapping | FSPL distance power term gives `d^-2` |
| `communication.air.alpha_nlos` | 2.0 | - | literature-derived model mapping | NLoS keeps same free-space distance exponent, with additional excess loss in beta |
| `communication.air.beta_los` | 1.1302166e-4 | power gain | literature-derived mapping | `beta0=(c/(4πf_c))^2`, then multiply by `10^(-1/10)` |
| `communication.air.beta_nlos` | 1.4228584e-6 | power gain | literature-derived mapping | `beta0=(c/(4πf_c))^2`, then multiply by `10^(-20/10)` |

Al-Hourani et al., *Optimal LAP Altitude for Maximum Coverage*, DOI `10.1109/LWC.2014.2342736`，给出了空地链路平均 pathloss 结构：

- `PL_LoS = FSPL + eta_LoS`
- `PL_NLoS = FSPL + eta_NLoS`
- `P_LoS(theta)=1/(1+a exp(-b(theta-a)))`
- `P_NLoS = 1 - P_LoS`

因此，本项目将 2GHz 处的 1m 自由空间参考功率增益写成：

- `beta0 = (c/(4*pi*f_c))^2 = 1.422858414e-4`
- Urban LoS excess loss 1dB：`beta_los = beta0 * 10^(-1/10) = 1.130216612e-4`
- Urban NLoS excess loss 20dB：`beta_nlos = beta0 * 10^(-20/10) = 1.422858414e-6`

当前配置中的 `beta_los = 1.1302166e-4`、`beta_nlos = 1.4228584e-6` 与上述推导只存在四舍五入差异。

空地信道 sanity check：

| horizontal distance m | 3D distance m | elevation deg | P_LoS | gain |
|---:|---:|---:|---:|---:|
| 0 | 100.000 | 90.000 | 0.999975 | 1.130189e-8 |
| 50 | 111.803 | 63.435 | 0.998255 | 9.026153e-9 |
| 100 | 141.421 | 45.000 | 0.967692 | 5.470806e-9 |
| 300 | 316.228 | 18.435 | 0.299262 | 3.482019e-10 |

检查结果：`P_LoS` 位于 `[0,1]`，平均信道增益为正且 finite。

## 4. Communication range 分类

`nodes.sv.range_m`、`nodes.rsu.range_m`、`nodes.uav.range_m` 当前仅在 `src/cache_configuration.py` 中作为配置阶段轻量 surrogate 的候选节点筛选阈值：

`if distance > range: skip candidate`

最新系统模型没有把 `distance <= range` 写成最终 Matching/KKT 的物理硬覆盖约束；物理信道质量由 ground pathloss 和 UAV air LoS/NLoS channel model 计算。因此 range 是 surrogate / algorithm parameter，不是需要文献物理校准的系统参数。

## 5. 算法超参数

以下参数不要求作为物理文献参数校准，后续通过算法敏感性、收敛性和运行时间实验检查：

| parameter | value | purpose |
|---|---:|---|
| `cache.discount_beta` | 0.95 | cache surrogate horizon discount |
| `cache.local_search_tolerance` | 1.0e-9 | strict improvement tolerance |
| `cache.max_coordinate_rounds` | 20 | coordinate search stop limit |
| `cache.knapsack_initialization` | enumeration | deterministic initialization method |
| `nodes.sv.range_m` | 100.0 | surrogate candidate cutoff |
| `nodes.rsu.range_m` | 250.0 | surrogate candidate cutoff |
| `nodes.uav.range_m` | 300.0 | surrogate candidate cutoff |

## 6. Formal validation 语义

`src/project_config.py` 现在使用最小白名单 `FORMAL_SYSTEM_PARAMETER_KEYS` 区分：

- 系统物理/目标函数参数：若仍为 `TODO_LITERATURE_CALIBRATION` 或未冻结的 `PROVISIONAL_IMPLEMENTATION`，formal experiment 阻塞。
- 算法超参数和 surrogate cutoff：只由 `validate_config()` 检查存在性、类型和范围，不要求文献物理来源。

当前 formal validation 已不再因系统物理/目标函数参数阻塞。仍保持 `PROVISIONAL_IMPLEMENTATION` 的条目属于算法超参数或 surrogate cutoff，不作为系统参数层 formal experiment blocker。

## 7. Dynamic cache nodes

从 `vehicle_roles.json` 与 `system_config.toml` 动态读取：

- SV：`46`, `29`, `80`
- RSU：`RSU_0`
- UAV：`UAV_0`, `UAV_1`

合计 dynamic cache nodes = 6。`nodes.uav.count >= 2` 校验继续保留。

## 8. 最小验证结论

- TOML 解析正常。
- `Q_slot_ref` 差异原因已定位为 Cloud 参数口径变化。
- 10 个 service 仍无法被任一节点全部缓存。
- Cache capacity、eta/xi/lambda cache 尺度 finite。
- UAV movement cost 尺度 finite；`xi_mov` 已由旋翼功率模型等效映射为 `15 J/m`。
- Air 参数可读取；LoS 概率、LoS/NLoS beta 与 alpha 的证据链已补齐。
- Dynamic cache nodes = 6。
- `cache_configuration.py` 最小加载正常。
- `validate_for_formal_experiment()` 通过。
- 本阶段没有修改任何 `data/generated/*`。
