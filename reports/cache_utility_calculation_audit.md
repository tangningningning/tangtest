# Cache predicted utility gain 计算链审计

## 审计结论

当前链路不属于“candidate cache 没有传入 utility 计算”的 A 类。`predicted_service_utility` 明确读取 candidate cache，并对已缓存 service 计算 SV/RSU/UAV edge utility。

实际原因最接近 **B 类**，但需要更精确地表述为：

1. candidate cache 的状态确实进入了 utility evaluator；
2. 一些候选提高了动态 cache 覆盖率，但没有改变每个任务的 winning execution path；当前 slot 的 1,413 个预测事件前后都选择 Cloud；
3. 因此 `utility_after - utility_before = 0`；
4. 随后 cache update cost 为正，使 net gain 全部为负。

这不是一个正 utility 被成本完全抵消的单一现象，而是“有效路径 utility 没有改善”之后又扣除了控制/加载成本。也不属于 C 类：没有发现读取错误状态，candidate cache、utility、成本和 net gain 的代数关系均闭合。

本次只做静态检查和内存中的 candidate replay，没有修改代码、参数、reward、cache algorithm、PPO 或 artifact，没有运行 PPO 训练。

## 1. 审计范围与数据

全窗口采用固定 `seed=4401`、50 个 execution slot 的既有 rollout 口径：

| 指标 | 结果 |
|---|---:|
| cache trigger | 209 |
| candidate 总数 | 2,864 |
| candidate predicted utility gain 为正 | 0 |
| `delta_hit > 0` 的候选 | 362 |
| candidate net gain 最大值 | -0.100000 |
| candidate net gain 平均值 | -0.503634 |
| 实际 cache update | 0 |

为展开完整计算链，本报告另外对 slot 0 使用同一初始 cache 和已记录 action 进行 52 个候选的逐候选 replay：

```text
cache_trigger = [1,0,0,0,0,1,0,1,0]
uav_trigger   = {UAV_0: 1, UAV_1: 1}
```

## 2. Candidate cache 前后状态

slot 0 的 52 个候选均不是 reference cache 的同一状态：

| 检查项 | 结果 |
|---|---:|
| candidate 数量 | 52 |
| 改变 cache cell 数量 | 1–2，平均 1.538462 |
| 改变节点数量 | 全部为 1 个节点 |
| candidate 与 reference 完全相同 | 0 个 |

代表性正命中候选为一个 RSU swap（服务 ID 为 0-based）：

```text
changed cells = [[5,2], [5,6]]
含义：RSU_0 删除 service 2，加入 service 6
```

所以 candidate 状态变化是实际存在的，并非因为 candidate 与 old cache 相同而造成 gain 为零。

## 3. Hit ratio 是否传递到 utility

代表性候选的动态 cache 覆盖率变化如下：

| 指标 | candidate 前 | candidate 后 | 变化 |
|---|---:|---:|---:|
| 预测动态 cache hit coverage | 0.731145 | 0.786811 | +0.055667 |
| predicted service utility | -41.312266 | -41.312266 | 0 |
| 选中路径数量 | Cloud 1,413 | Cloud 1,413 | 不变 |

slot 0 的 52 个候选整体结果为：

- 8/52 个候选的 `delta_hit > 0`；
- 所有 52 个候选的 `utility_before == utility_after`（容差 `1e-12`）；
- 所有候选的路径选择仍为 Cloud，共 1,413 个预测事件；
- 因而 hit coverage 的变化进入了 cache gating，但没有转化为最终 winning path 的 utility 变化。

这里的 `delta_hit` 是“至少一个动态 cache 节点包含所请求 service”的覆盖率指标；它不等价于 `predicted_service_utility`。后者对每个事件取 local、Cloud 和可用 edge path 的最大任务 utility。如果新增 edge path 没有严格超过 Cloud，cache 覆盖率可以增加而 winning path 不变。

## 4. Utility、delay、energy 和 cache cost

### 4.1 Utility 与路径 delay/energy

slot 0 的 52 个候选中，candidate 前后选中路径的变化为：

| 指标 | 前后最大绝对变化 |
|---|---:|
| predicted service utility | 0 |
| 折扣事件加权 delay | 0 s |
| 折扣事件加权 vehicle energy | 0 J |
| winning path label | 0 个事件变化 |

代表性候选的折扣事件加权路径量为：

```text
delay_before = 1.662175 s
delay_after  = 1.662175 s
energy_before = 0.018214 J
energy_after  = 0.018214 J
```

上述 delay/energy 是审计用的预测事件加权汇总；系统 utility 本身仍按实现中的逐事件 `task_utility` 加权累计，不是使用该汇总值代替原公式。

### 4.2 Cache cost

slot 0 的候选 cache cost 为：

| 指标 | candidate 前 | candidate 后 |
|---|---:|---:|
| cache update cost 最小值 | 0 | 0.100000 |
| cache update cost 最大值 | 0 | 1.536427 |
| cache update cost 平均值 | 0 | 0.535070 |
| service loading cost 平均值 | 0 | 0.404301 |
| control cost 平均值 | 0 | 0.130769 |

代表性 RSU swap：

```text
utility gain       = 0
loading cost       = 1.135182
control cost       = 0.100000
cache update cost  = 1.235182
net gain           = -1.235182
```

candidate 前使用 `C_prev -> C_prev`，所以 cache cost 为 0；candidate 后使用 `C_prev -> C_candidate`，因此只要 cache state 发生变化就会产生相应成本。

## 5. Gain 公式实际实现

### 5.1 Utility evaluator

实现先为每个预测事件计算不依赖 candidate cache 的 base utility：

```text
base_utility = max(local_utility_if_available, cloud_utility)
```

然后遍历 candidate cache 中包含该 service 的节点：

```text
if cache[node_index, event.service_id] == 0:
    continue
edge_utility = edge_task_utility(...)
if edge_utility > best:
    best = edge_utility
```

最终按 horizon discount 累计：

```text
predicted_service_utility(C)
    = sum(discount * best_event_utility) / num_scenarios
```

因此 cache 状态进入了计算，但只有当 cache 提供的 edge path 严格优于当前 best path 时，才会改变 utility。slot 0 replay 中 winning path 全部是 Cloud，说明当前 edge cache path 没有超过 Cloud base utility。

### 5.2 Objective 与 net gain

实际 objective 为：

```text
objective(C_candidate)
    = predicted_service_utility(C_candidate)
      - cache_update_cost(C_prev, C_candidate)
```

所以：

```text
net_gain
    = utility_after - utility_before
      - (loading_cost + control_cost + other_penalty)
```

本次 52 个详细候选的公式残差为 0；全窗口候选的 `other_penalty` 也仅在约 `3.22e-15` 的浮点舍入范围内。

### 5.3 Cache update cost

成本实现为：

```text
actual_trigger = any(C_candidate != C_prev, axis=1)
loaded_mb      = additions @ service_sizes_mb
loading_energy = xi_ca * loaded_mb
cache_cost     = eta_ca * actual_trigger
                 + lambda_cache * loading_energy
```

对应实现位置：

- `src/decision/cache_configuration.py:633`：candidate cache gating 与 predicted utility；
- `src/decision/cache_configuration.py:815`：`predicted_service_utility` 入口；
- `src/decision/cache_configuration.py:872`：objective；
- `src/decision/cache_configuration.py:1336`：严格改进接受条件；
- `src/common/system_model.py:226`：cache update cost 分解。

## 6. A/B/C 分类

| 分类 | 是否成立 | 证据 |
|---|---|---|
| A：cache 变化没有进入 utility | 否 | evaluator 明确读取 `cache[node, service_id]` 并计算 edge utility |
| B：变化进入，但最终没有形成收益并被成本扣除 | 最接近，且是当前现象 | 8/52 个候选 `delta_hit>0`，但 52/52 个 utility gain=0，之后成本为正 |
| C：utility 使用错误状态 | 否 | candidate 前后状态、utility、路径和成本公式可复核且代数闭合 |

最终判断：当前系统属于 **B 的精确变体：cache 命中覆盖变化进入了 evaluator，但被“逐任务最大路径选择”压成 utility-neutral；随后 cache update cost 使 net gain 为负**。因此不需要修改代码来“修复 gain=0”；若研究目标要求 cache 更新产生正收益，需要另行评估 edge/cloud 相对时延、覆盖范围或 utility 标定，但本次不做任何调参或修改。

## 7. 修改范围与检查

本次新增文件：

- `reports/cache_utility_calculation_audit.md`

未修改源代码、配置、reward、PPO、cache algorithm、service/task artifact 或其他数据文件。
