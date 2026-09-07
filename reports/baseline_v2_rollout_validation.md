# Baseline v2.0 单 slot Rollout 验证与 Reward / Delay 审计

## 1. 验证范围

本次使用当前 `data/generated/` artifact 进行一个固定 seed、单 episode、单执行 slot 的最小 rollout。执行链路为：任务读取 → service request → cache lookup → offloading matching → KKT resource allocation → delay / energy → system reward。未运行 PPO 训练、长时间 SUMO 仿真，也未修改算法、reward、cache 机制或任何 artifact。

## 2. 当前生效配置

| 类别 | 参数 | 当前值 |
|---|---|---:|
| 车辆 | research / SV / TV | 25 / 5 / 20 |
| 节点 | RSU / UAV | 2 / 2 |
| 服务 | `service.count` | 10 |
| 任务 | input size | 500–5000 KB（配置范围） |
| 任务 | cycles per bit | 400–1000（配置范围） |
| CPU | TV / SV / RSU / UAV / Cloud | 1e9 / 1e9 / 5e9 / 5e9 / 20e9 cycle/s |
| Cloud | `psi_c`（`nodes.cloud.cloud_delay_factor`） | 11 |
| 目标权重 | alpha(delay) / beta(energy) | 0.6 / 0.4 |
| 实验 | fixed seed | 4401 |

运行时实际读取结果：`psi_c=11.0`、`omega_delay=0.6`、`omega_energy=0.4`，权重和为 1.0；Cloud CPU 和总带宽分别为 `20e9 cycle/s`、`200e6 bit/s`。

## 3. 单 slot rollout 结果

| 项目 | 结果 |
|---|---:|
| episode / origin index / slot | 0 / 0 / 0 |
| time | 100.0 s |
| 对齐 steps | 2（其中 1 个用于 bootstrap） |
| observation shape | `[1, 154]` |
| action shape | `[1, 11]` |
| action vector | `[1,0,0,0,0,1,0,1,0,1,1]` |
| step feasible | `true` |
| reward | `-108.1685658993` |
| reward finite | `true` |
| PPO float32 reward | `-108.1685638428` |

固定 seed 的最小 rollout 复跑结果一致；old log-prob/value 重算误差为 0，state propagation 和 reward invariant 均通过。

## 4. Delay 分解

本 slot 共处理 15 个 arrival task。Cloud 任务数为 7，同一 slot 按当前单 Cloud 共享模型分配 CPU 和带宽。表中边缘节点时延为上传通信时延与节点计算时延之和；Cloud 显式使用 `psi_c * D / R_cloud_i + L / F_cloud_i`。

| 执行路径 | 任务数 | 平均 communication delay (s) | 平均 computation delay (s) | 平均 total delay (s) | 平均 vehicle energy (J) |
|---|---:|---:|---:|---:|---:|
| TV local | 0 | 0 | 0 | 未在本 slot 选中 | 0 |
| RSU edge | 2 | 2.017003 | 4.357907 | 6.374910 | 0.402445 |
| UAV edge | 6 | 1.128292 | 3.410798 | 4.539091 | 0.225124 |
| Cloud | 7 | 6.783362 | 3.693125 | 10.476487 | 0.123042 |

TV local 公式路径已保留，但当前固定 seed 与 slot 的实际 matching 没有选择 local；这表示该路径在本次 workload 中没有样本，不代表 local 公式报错。

## 5. Energy 结果

| 项目 | 总值 |
|---|---:|
| local energy | 0.000000 J |
| edge energy（RSU + UAV） | 2.155634 J |
| Cloud energy | 0.861292 J |
| total vehicle energy | 3.016926 J |
| UAV movement energy | 0.000000 J |

所有能耗结果均为 finite，Cloud 能耗仍按实际上传数据量与 Cloud 分配带宽计算；`psi_c` 只影响 Cloud latency，不改变能耗公式。

## 6. Reward 组成审计

当前 reward 使用 `alpha(delay)=0.6` 和 `beta(energy)=0.4` 对 15 个任务的 utility 求和：

```text
delay contribution  = -95.6028695966
energy contribution = -12.5656963026
actual task utility = -108.1685658993
cache cost          = 0.0000000000
movement cost       = 0.0000000000
system reward G     = -108.1685658993
```

因此：

```text
G = actual_task_utility - cache_cost - movement_cost
```

审计确认没有使用旧的 0.5/0.5 权重；reward 公式和实现本身未修改。

## 7. Cache 基础检查

- service 维度：10，当前 service ID 为 `0..9`；
- cache state：`[9, 10]`，即 5 SV + 2 RSU + 2 UAV 的动态缓存节点 × 10 个 service；
- 当前 slot cache hit：8 / 15 = 53.3333%；
- 当前 slot cache miss：7 / 15 = 46.6667%；
- 所有节点缓存容量约束通过。

| 节点类型 | 使用容量 (MB) | 容量 (MB) | 余量 (MB) | 约束 |
|---|---:|---:|---:|---|
| SV（每个） | 15083.251431 | 16744.750325 | 1661.498894 | 通过 |
| RSU（每个） | 41292.518338 | 44652.667533 | 3360.149195 | 通过 |
| UAV（每个） | 31293.891707 | 33489.500649 | 2195.608943 | 通过 |

当前运行未发现旧的 20-service cache state。

## 8. Cache dynamic validation

### 8.1 Cache trigger 统计

本次 PPO trigger action 的 cache 部分为：`[1,0,0,0,0,1,0,1,0]`。TV 使用固定 local cache，不存在动态 `z_m^ca`；UAV movement trigger `[1,1]` 与 UAV cache trigger 分开统计。

| 节点类型 | 可触发 cache 节点数 | trigger=1 数量 | trigger 比例 | 具体触发 |
|---|---:|---:|---:|---|
| TV | 0 | 0 | 不适用 | TV local cache 固定，无 `z_m^ca` |
| SV | 5 | 1 | 20.0% | node `23` |
| RSU | 2 | 1 | 50.0% | `RSU_0` |
| UAV | 2 | 1 | 50.0% | `UAV_0` |

### 8.2 Cache state 前后变化

| 指标 | 结果 |
|---|---:|
| input cache trigger 数量 | 3 |
| `cache_change_count`（实际更新节点数） | 0 |
| 改变的 service 数量 | 0 |
| 替换前后 service ID | 无变化 |
| accepted cache updates | 0 |
| trigger feasible | `true` |
| optimizer stop reason | `converged_by_no_improvement` |

触发节点的 service 集合如下：

| 节点 | 更新前 service ID | 更新后 service ID |
|---|---|---|
| SV `23` | `[1,4,5]` | `[1,4,5]` |
| RSU `RSU_0` | `[2,4,5,7,9]` | `[2,4,5,7,9]` |
| UAV `UAV_0` | `[2,4,5,7]` | `[2,4,5,7]` |

### 8.3 Hit 变化与更新成本

| 指标 | 更新前 | 更新后 | 变化 |
|---|---:|---:|---:|
| cache availability hit | 8 / 15 = 53.3333% | 8 / 15 = 53.3333% | `delta_hit=0` |
| 实际 matching hit（非 Cloud） | 8 / 15 | 8 / 15 | 0 |
| 实际 matching miss（Cloud） | 7 / 15 | 7 / 15 | 0 |

当前配置中的更新成本参数为：

- SV：`xi_ca=0.0010`、`eta_ca=0.20`；
- RSU：`xi_ca=0.0007`、`eta_ca=0.10`；
- UAV：`xi_ca=0.0012`、`eta_ca=0.15`；
- `lambda_cache=0.1`。

实际结果为：

```text
loaded_mb = 0
loading energy (xi_ca) = 0
actual control trigger = 0
control cost (eta_ca) = 0
cache update cost = 0
cache-related utility delta = 0
```

因此本 slot 属于 **B 类：存在 trigger，但 cache 没有实际变化**。trigger 只是允许对应节点进入配置优化；只有候选缓存状态相对参考状态产生有效目标改进时，才会形成 `C_m(t+1) != C_m(t)`，并按 `xi_ca`、`eta_ca` 计费。本次优化以 `converged_by_no_improvement` 停止，说明当前 slot 的候选更新在成本/收益判断后没有被接受，不是缓存约束失败或 NaN/Inf。

## 9. 是否进入 PPO 训练

单 slot 运行链路、维度、offloading、资源分配、delay、energy 和 reward 均可执行，说明系统具备进入短时接口验证的条件。

但本次缓存动态结果为 B 类，尚未在该 slot 观察到实际 cache replacement。因此建议：

- 可以进入短时、非长期的 PPO 接口/rollout 验证；
- 不建议仅凭这一个 slot 直接开始正式或长期 PPO 训练；
- 进入训练前应再做少量多 slot 的 trigger 覆盖统计，确认至少有场景出现有效 `cache_change_count > 0`，同时保持当前算法和参数不变。

