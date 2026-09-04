# Baseline v2.0 缓存容量压力分析与更新机制审计

> 分析日期：2026-09-04。本报告只读分析当前 baseline v2.0，未修改配置、算法或 artifact。

## 1. 分析输入与当前参数

- 配置：`system_config.toml`（唯一配置入口）。
- 服务目录：`data/generated/service_catalog.json`；配置 service 数量为 `10`，artifact service 数量为 `10`。
- 请求分布：`data/generated/popularity.csv`，使用 `popularity.csv 的 rho_star`。
- popularity epoch：15 个；配置 epoch 长度：20 slot；每 epoch 变化次数：1；Zipf 指数：0.800。
- arrival probability：0.700。该参数不改变容量比例，但用于记录当前请求模型。

| 节点类型 | 缓存容量 (MB) | 配置读取路径 |
|---|---:|---|
| TV | 5581.583442 | `nodes.vehicle.local_cache_capacity_mb` |
| SV | 16744.750325 | `nodes.sv.cache_capacity_mb` |
| RSU | 44652.667533 | `nodes.rsu.cache_capacity_mb` |
| UAV | 33489.500649 | `nodes.uav.cache_capacity_mb` |

当前 `objective.lambda_cache = 0.100000`；动态 cache 更新成本参数如下。TV local cache 是静态 artifact，不参与下表的动态 cache reconfiguration。

| 动态节点 | eta_ca（控制成本） | xi_ca（加载成本/MB） |
|---|---:|---:|
| SV | 0.200000 | 0.001000 |
| RSU | 0.100000 | 0.000700 |
| UAV | 0.150000 | 0.001200 |

## 2. Service catalog 总大小

`S_total = sum(service_size) = 111631.668831 MB`。

| 统计量 | 数值 (MB) |
|---|---:|
| service 数量 | 10 |
| 最小 service size | 3372.996237 |
| 最大 service size | 19753.560543 |
| 平均 service size（仅作描述） | 11163.166883 |

容量比例按 `cache_capacity / S_total` 计算；可缓存数量和热门覆盖率均使用 catalog 中每个 service 的实际大小。

## 3. 容量压力与热门缓存覆盖率

`最大可缓存服务数` 按实际大小从小到大累加，因而是同一容量下的最大数量；没有使用平均大小估算。
`top-K 排名覆盖率` 是按热门排名前 K 个服务的概率和，用于展示热度上限；由于不同 service 大小不同，它不保证前 K 个能同时装入容量。
`理想热门命中率` 则在真实大小约束下求解容量内概率最大的 0-1 服务集合，更接近可实现的理想上界。

| 节点 | 容量 (MB) | 容量 / catalog | 最大可缓存服务数 | top-K 排名覆盖率（epoch均值） | 容量约束理想命中率（均值） | 理想命中率范围 |
|---|---:|---:|---:|---:|---:|---:|
| TV | 5581.583442 | 5.000% | 1 | 28.050% | 6.132% | 5.314%--6.690% |
| SV | 16744.750325 | 15.000% | 3 | 55.807% | 38.378% | 35.790%--39.697% |
| RSU | 44652.667533 | 40.000% | 6 | 79.490% | 69.850% | 63.547%--72.648% |
| UAV | 33489.500649 | 30.000% | 5 | 72.800% | 64.486% | 57.588%--67.333% |

按当前真实大小，四类节点的最大服务数量分别由最小服务组合决定：TV=1个、SV=3个、RSU=6个、UAV=5个。具体的热门命中率已在上表中按每个 epoch 重新计算。

## 4. Zipf 请求压力

当前任务模型为 Zipf `zeta=0.800`，请求分布按 `popularity.csv` 的 `rho_star` 读取。`rho_star` 的单个概率值范围为 0.044456--0.280496。
排序后的 Zipf 概率集合跨 epoch 保持不变；热门服务排序共有 11 种，连续 epoch 间发生 14 次排名变化。
因此，分布的 Zipf 形状相对稳定，但服务 ID 的热门顺序并非全程固定；容量内的理想命中集合存在随 epoch 调整的收益空间。

## 5. 缓存更新机制审计

以下为源代码静态审计结果，没有调用训练、SUMO 或完整 slot 仿真。

| 审计项 | 结果 | 结论 | 代码证据 |
|---|---|---|---|
| z_m^ca 等价触发变量 | 发现 | 存在 z_m^ca 的等价实现：动作标签和优化器输入均使用 cache trigger。 | src/ppo/ppo_trigger_policy.py:73 `labels = tuple([f"z_{node_id}^ca" for node_id in cache_node_ids] + [f"z_{uav_id}^q" for uav_id in uav_ids])`；src/decision/cache_configuration.py:855 `cache_trigger: np.ndarray,` |
| 实际更新判定 | 发现 | 实际更新由 new cache 与 old cache 的逐节点差异决定，而不是只看动作意图。 | src/common/system_model.py:287 `actual_trigger = np.any(old != new, axis=1).astype(np.int8)`；src/decision/cache_configuration.py:1374 `actual_cache_trigger = np.any(` |
| new cache == old cache 的无效更新处理 | 发现 | new cache 与 old cache 相同时 actual trigger 为 0，验证路径要求更新成本和能耗为 0。 | src/decision/cache_configuration.py:1521 `if not np.any(expected):`；src/common/system_model.py:287 `actual_trigger = np.any(old != new, axis=1).astype(np.int8)` |
| 加载成本 | 发现 | 0->1 的服务按真实大小计算 loaded MB，并通过 xi_ca 计算加载能耗/成本项。 | src/common/system_model.py:288 `additions = np.maximum(new - old, 0)`；src/common/system_model.py:289 `loaded_mb = additions @ sizes`；src/common/system_model.py:290 `loading_energy = xi * loaded_mb` |
| 控制成本 | 发现 | 实际发生更新的节点承担 eta_ca 固定控制成本，加载项再乘 lambda_ca。 | src/common/system_model.py:291 `cost_by_node = actual_trigger.astype(float) * eta + lam * loading_energy`；src/common/system_model.py:291 `cost_by_node = actual_trigger.astype(float) * eta + lam * loading_energy` |
| 替换/邻域动作 | 发现 | 缓存优化器提供 add/delete/swap 邻域，并只接受目标函数严格改善的替换。 | src/decision/cache_configuration.py:1012 `def generate_node_neighbors(`；src/decision/cache_configuration.py:1317 `best_move: tuple[str, np.ndarray, float] \| None = None` |
| 触发门控 | 发现 | 没有 cache trigger 时存在不搜索的快路径；有 trigger 时只优化被触发节点。 | src/decision/slot_environment.py:700 `if int(cache_trigger.sum()) == 0 and not any(int(value) == 1 for value in uav_triggers.values()):`；src/decision/cache_configuration.py:1312 `triggered_indices = [index for index, value in enumerate(cache_trigger.tolist()) if value == 1]` |
| 成本进入目标函数 | 发现 | 更新成本进入 cache objective，并由 slot environment 统一计入系统效用。 | src/decision/slot_environment.py:487 `return cache_configuration.cache_update_cost(`；src/decision/slot_environment.py:590 `objective = float(predictive_utility - cache_cost - movement_cost)`；src/decision/cache_configuration.py:1378 `final_objective = float(predicted_utility - update_cost)` |

审计结论：

1. 代码中存在与论文 `z_m^ca(t)` 等价的 cache trigger；动作触发只是更新意图，最终是否真实更新由新旧 cache 状态差异确定。
2. 当 `new_cache == old_cache` 时，不会被计为 actual update；相应加载能耗和 cache update cost 应为 0。
3. 代码同时计算 0->1 服务加载量、`xi_ca` 加载项和 `eta_ca` 固定控制项，并将更新成本扣除到 configuration objective。
4. 更新不是每 slot 强制发生：没有 trigger 时走不搜索快路径；有 trigger 但候选目标不严格改善时也会保留原 cache。因此“动作触发”与“有效 cache 更新”是两个不同事件。

## 6. 是否存在缓存长期不更新风险

当前最大容量比例为 40.000%，RSU 也只覆盖 catalog 的 40.0%；当前容量摘要为 TV=5581.583442MB、SV=16744.750325MB、RSU=44652.667533MB、UAV=33489.500649MB，并非接近完整 catalog 的容量。仅从容量看，没有“整个 catalog 都能长期驻留”导致更新必然消失的条件。
同时，`rho_star` 的概率集合 稳定，但热门排序在 epoch 间发生变化；而更新控制成本、加载成本和严格改善门槛均存在。由此判断：当前配置不存在由容量直接造成的必然长期不更新风险，但存在条件性的稀疏更新风险——若外层策略长期输出 0 trigger，或每次热度变化带来的收益不足以抵消更新成本，cache 可以保持不变。

## 7. 容量建议

本次不建议仅凭该压力分析修改缓存容量。当前容量提供了从 TV 的单服务缓存到 RSU/UAV 的多服务缓存层次，且热门排序会变化；建议保留当前由 catalog 比例推导的 TV=5581.583442MB、SV=16744.750325MB、RSU=44652.667533MB、UAV=33489.500649MB baseline。若后续实验观测到长期无更新，应优先统计外层 trigger 发出率、actual_cache_trigger 发出率和“候选收益 - 更新成本”，再决定是否做容量或成本敏感性实验。

## 8. 修改边界

本次分析只读取 `system_config.toml`、service catalog、popularity 和缓存相关源码；没有修改配置、缓存算法、reward、PPO、task generation 或任何 artifact。
