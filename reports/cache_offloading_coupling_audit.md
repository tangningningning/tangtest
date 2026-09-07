# Cache state 对任务执行路径选择的耦合审计

## 审计结论

当前 cache state 确实影响 offloading 的可行目标集合，但 cache hit 并不会强制任务改走 edge。实现流程是：

```text
edge_cache_state
    -> feasible_targets
    -> matching candidate
    -> KKT resource evaluation
    -> selected target / delay / energy
```

真实 baseline rollout 的 50 个 slot 中 `cache_change=0`，因此真实观测到的“由 cache 变化造成的路径变化”为 0，这是因为没有实际 cache state 变化，不是因为 coupling 断开。

为审计 cache change 后的实际耦合，在内存中对 50 个 slot 各构造一个满足容量约束的 counterfactual cache candidate，并保持任务、车辆位置、UAV/RSU 位置和其他参数不变。结果为：

- 50/50 个 slot 发生了反事实 cache state 变化；
- 4/50 个 slot 出现 selected target 改变；
- 共 10 个任务实例改变执行节点；
- 另有 17 个任务保持同一执行节点，但因共享资源分配变化而出现 delay/energy 变化；
- 50 个 slot 共 682 个任务实例，selected target 改变比例为 1.47%。

因此，cache-offloading coupling 存在，但表现为“cache 改变可行集合，matching 再根据 task utility 选择路径”，不是“cache hit 直接奖励 edge 路径”。

## 1. 服务可执行节点集合

slot 0 的初始动态 cache 为：

```text
SV:  [1, 4, 5]
RSU: [2, 4, 5, 7, 9]
UAV: [2, 4, 5, 7]
```

因此 service 6 在所有动态 edge 节点中均未缓存。slot 0 的一个反事实 RSU_0 swap 将 service 6 加入 RSU_0；对实际任务 55，其 local cache 也不包含 service 6，得到：

| 状态 | service 6 的可执行目标 |
|---|---|
| before | `[CLOUD]` |
| after | `[RSU_0, CLOUD]` |

这说明 cache state 已经改变了 service availability 和 feasible target set。该任务在 slot 0 的 matching 仍选择 Cloud，说明“可执行”不等于“必然选择”。

## 2. Cache state 到 matching 的实现链

### 2.1 可行目标集合

`src/decision/task_allocation.py:82` 的 `build_feasible_targets()` 对每个任务执行：

```text
如果 task service 在该车 local cache：加入 LOCAL:<task_id>
如果 edge_cache_state[node_index, task.service_id] == 1：加入 node_id
始终加入 CLOUD
```

所以 Cloud 是 cache-independent fallback，edge 节点只有在 cache state 对应位置为 1 时才会进入可行集合。

### 2.2 Matching 初始状态与路径选择

`src/decision/task_allocation.py:349` 的 `optimize_task_matching()`：

1. 先通过 `build_feasible_targets()` 读取 cache availability；
2. 将所有任务初始化为 Cloud；
3. 只枚举 feasible target 中与当前 target 不同的 transfer/exchange；
4. 通过 KKT evaluator 计算 candidate 的 task utility、delay 和 vehicle energy；
5. 只有 `delta_J > tolerance` 的 matching 才会被接受。

因此 cache 变化的作用是开放或关闭候选 edge target，最终是否 offload 到该节点由实际 delay/energy 和资源竞争决定。

## 3. 50-slot counterfactual 结果

反事实设置仅用于审计：每个 slot 从 reference cache 的可行邻居中选择一个新增当前任务所请求 service 的 candidate，随后分别运行现有 matching。没有写回 cache artifact，也没有替换正式 rollout 状态。

| 指标 | 真实 baseline rollout | counterfactual coupling sweep |
|---|---:|---:|
| slot 数量 | 50 | 50 |
| cache change slot 数量 | 0 | 50 |
| selected target 改变 slot 数量 | 0 | 4 |
| selected target 改变任务数 | 0 | 10 |
| target 不变但 delay/energy 改变任务数 | 0 | 17 |
| 总任务实例 | 682 | 682 |

发生 selected target 改变的 slot 为 18、26、44、47。具体路径变化如下；任务 ID 按 slot 区分：

| slot | task | service | before | after | delay before → after (s) | energy before → after (J) |
|---:|---:|---:|---|---|---:|---:|
| 18 | 95 | 5 | RSU_0 | UAV_0 | 9.567790 → 13.644547 | 0.837293 → 0.244277 |
| 18 | 83 | 8 | CLOUD | RSU_0 | 17.805666 → 10.911149 | 0.216353 → 1.395185 |
| 18 | 9 | 7 | UAV_0 | UAV_1 | 5.003858 → 4.143347 | 0.293931 → 0.283428 |
| 18 | 55 | 7 | UAV_0 | UAV_1 | 4.766863 → 3.790456 | 0.289052 → 0.245721 |
| 18 | 98 | 2 | UAV_1 | CLOUD | 12.470951 → 18.618846 | 0.551228 → 0.197270 |
| 26 | 64 | 9 | RSU_0 | CLOUD | 10.016066 → 8.025447 | 1.414151 → 0.079172 |
| 26 | 95 | 6 | CLOUD | RSU_0 | 11.501294 → 9.252723 | 0.146095 → 1.295945 |
| 44 | 83 | 8 | CLOUD | RSU_0 | 27.117759 → 7.567327 | 0.311954 → 0.630233 |
| 47 | 8 | 2 | RSU_0 | UAV_0 | 10.500405 → 13.331257 | 0.854600 → 0.330715 |
| 47 | 19 | 4 | UAV_0 | CLOUD | 10.341214 → 15.015354 | 0.329732 → 0.158017 |

### 同一 selected node 但 delay/energy 改变

反事实 cache change 还使 17 个任务的 selected node 保持不变、但 KKT 资源分配结果改变。例如 slot 18 的 task 64 仍选择 UAV_0，但：

```text
delay:  4.462700 s -> 4.818464 s
energy: 0.221828 J -> 0.173493 J
```

这是共享 edge/Cloud 资源下其他任务 target 改变后的间接影响，不是 cache state 直接修改了同一 execution formula。

## 4. 三类原因判断

### A：cache 没有进入 offloading constraint

**不成立。** `build_feasible_targets()` 直接读取：

```text
edge_cache_state[node_index, task.service_id]
```

并决定 RSU/UAV 是否加入该任务的 feasible target。service 6 的 `[CLOUD] -> [RSU_0, CLOUD]` 实例直接证明 cache state 改变了可行集合。

### B：Cloud 始终优于 edge

**不完全成立，但解释了多数路径不变情况。** 大多数反事实任务虽然新增了 RSU/UAV 可行目标，最终仍保持 Cloud 或原 edge target；这是因为 matching 使用实际路径的 delay/energy 和资源分配计算，Cloud 或已有节点的 task utility 可能更高。

但 Cloud 并非始终优于 edge：slot 18、26、44 的任务出现 `CLOUD -> RSU_0`，说明 cache change 能够在 edge path 更优时改变 selected node。

### C：cache hit 收益模型不足

**部分成立，属于当前模型接口的结构性特征。** cache hit 在 matching 层主要表示“该 edge target 可用”，没有单独的 cache-hit bonus，也不会绕过 delay/energy/KKT utility 比较。因此：

- cache hit 增加只会扩展候选集合；
- 如果 edge 路径没有带来更高 `delta_J`，matching 不会切换；
- 反事实中仍有 46/50 个 slot 没有 selected target 改变。

这不是 cache availability 没有接入，而是 availability 与 execution benefit 被分成两个阶段处理。

## 5. 实际路径与延迟/能耗计算

selected target 确定后，`evaluate_matching_with_kkt()` 根据目标类型计算：

- local：本地 CPU delay 和 vehicle energy；
- Cloud：通信 delay、Cloud computation delay 和 vehicle energy；
- RSU/UAV：上行传输 delay、edge computation delay 和 vehicle energy。

最终 matching 结果直接返回：

```text
per_task_delay_s
per_task_vehicle_energy_j
per_task_utility
```

本次对照未发现 cache state 绕过这些执行计算，也未发现 selected target 改变但 delay/energy 没有重新评估的情况。

## 6. 最终判断

当前系统的 cache-to-offloading coupling 是有效的，但强度有限：

```text
cache state 改变
    -> feasible edge targets 改变
    -> matching 候选改变
    -> 只有 utility 改善时才切换路径
    -> delay/energy 重新计算
```

真实 baseline 的 0 次路径变化由真实 cache_change=0 直接决定。反事实审计显示，cache change 可以改变执行路径，但在当前参数和任务分布下只影响少量任务。没有发现需要修改实现的耦合断点。

## 7. 修改范围与检查

本次新增文件：

- `reports/cache_offloading_coupling_audit.md`

未修改 cache algorithm、reward、PPO、参数、SUMO 逻辑、任务逻辑或 artifact。
