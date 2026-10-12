# 阶段3B：有限 Cloud CPU 池与资源感知卸载实现

## 结论

阶段3B实现已完成。旧配置缺少新字段时仍采用 `legacy_unlimited` 和 `LEGACY_EDGE_GATE`；新模型通过独立配置启用 60 GHz 有限 Cloud CPU 池与 `RESOURCE_AWARE_FALLBACK`。Matching 候选评价继续使用完整任务组资源分配后的实际 Task Utility，Cloud 任务集合变化时重算整个 Cloud CPU 池。

本次只完成 seed 30020 的 Slot 100、200 状态重放，没有运行 PPO、正式 Validation 或训练。Cloud 60 GHz 是实验锚点，尚未完成物理标定。

## 修改文件

- `src/common/project_config.py`：识别和校验 Cloud 资源模式、有限 CPU 容量及候选门控模式；不带新字段的旧配置采用兼容默认值。
- `src/decision/cache_configuration.py`：把 Cloud CPU 合同和候选门控模式带入决策配置，并保留旧配置字段兼容。
- `src/decision/task_allocation.py`：加入完整 Cloud 任务组 CPU KKT 分配、有限池资源可行性检查、Cloud 集合变化时的增量影响范围更新，以及效用容差内 Edge 平局规则。
- `src/decision/slot_environment.py`：接入配置的候选模式，并将执行节点、Cloud/Edge CPU 使用和性能统计写入 slot Matching summary。
- `src/common/system_model.py`：仅说明 `cloud_execution` 接收调用方分配的 CPU 速率；执行公式未改变。
- `configs/proposed/stage3_cloud_finite_60ghz.toml`：新增独立 Stage3B 配置。原有正式配置和冻结 run 配置未修改。

## 配置与资源分配流程

新配置设置：

| 项目 | Stage3B 配置 | 说明 |
|---|---:|---|
| `nodes.cloud.resource_mode` | `finite_pool` | 共享 Cloud CPU 池 |
| `nodes.cloud.cpu_capacity_cycles_per_s` | `60.0e9` | 60 GHz 实验锚点 |
| `nodes.cloud.uplink_rate_bps` | `200.0e6` | 保持每任务独立等效上行链路 |
| `nodes.cloud.resource_sharing` | `false` | 不引入 Cloud 共享带宽池 |
| `nodes.cloud.cloud_delay_factor` | `11.0` | 沿用原通信时延系数 |
| `algorithm.matching.cloud_fallback_mode` | `RESOURCE_AWARE_FALLBACK` | 同时保留合法 Edge 与 Cloud 候选 |

有限池下，对当前全部 Cloud 任务联合分配 CPU 份额：

\[
f_i = x_i F_C^{total},\qquad \sum_i x_i\leq 1,
\]

因此 \(\sum_i f_i\leq60\times10^9\) cycles/s，计算时延继续使用原执行式 \(L_i/f_i\)。分配器复用项目已有的 KKT fraction solver，按当前 Cloud 任务组一次性求解；Cloud 上行仍按每任务独立速率计算，未加入共享通信资源或改变 Cloud 时延公式。旧模式仍逐任务使用旧配置的 `cloud_total_cpu_cycles_per_s`，本项目冻结配置对应 5 GHz/task。

Matching 每次候选比较均以候选匹配为输入重算其资源分配和任务效用。有限池中只要 Cloud 任务集合变化，就把当前与候选 Cloud 任务集合并入影响范围，并整体重新计算 Cloud CPU 分配；不复用只针对单个 Cloud 任务的旧分配。受候选变化影响的 Edge 资源组仍由既有 Edge CPU/带宽 KKT 逻辑处理，不改变 Edge 资源参数。最终接受依据是重新分配后计算出的 Task Utility。

## Matching 与平局规则

- 旧 `LEGACY_EDGE_GATE` 保留原逻辑，作为默认配置行为。
- 新 `RESOURCE_AWARE_FALLBACK` 将所有合法缓存 Edge 目标与 Cloud 一并纳入候选；缓存可用不再自动排除 Cloud。
- Edge 不获得额外效用，Cloud 也没有新增惩罚。只有候选效用差异在现有数值容差内，且结果减少 Cloud 任务数时，才接受 Edge 平局优先转移，并累计 `tie_break_count`。
- 对非平局操作，仍要求 Task Utility 增量超过原容差。Matching 仍是 Transfer-Exchange 局部搜索，不保证全局最优。

## Slot summary 字段

`matching_summary` 增加或汇总：

- `node_task_count`：Local、SV总数和逐SV数、RSU总数和逐RSU数、UAV0、UAV1、逐UAV数、Cloud。计数来自 Matching 的最终执行节点，不是 Cache 命中数。
- `resource`：Cloud CPU 总使用量、容量、剩余量和逐任务分配；Edge CPU 总使用量及逐节点使用量。
- `performance`：平均任务时延、平均车辆能耗、Task Utility 总和与逐任务平均值。
- `tie_break_count`、资源模式和候选门控模式。

Reward 公式、Cache cost、Movement cost、PPO 网络、Observation 和 Action space 均未改。重放中检查了 `system_utility = task_utility - cache_cost - movement_cost` 仍成立。

## 最小配对重放

输入使用 seed 30020 的既有 Validation episode 和 Cache KEEP 反事实保存的 `per_slot_A` 状态分支，固定 Slot 100、200 的任务、Cache 和 UAV 位置，只执行 Matching/KKT。结果对应指定的 `formal_cache_economic_v2_lr_5e_minus_4` 实验状态来源；本次没有重新执行 PPO policy 推理。对照同时切换了旧/新候选门控及 Cloud CPU 合同，因此这是旧模型与 Stage3B 新模型的端到端配对对照，不能把差异单独归因于 Cloud CPU 容量。

### 任务执行结果

| Slot | 模式 | 任务数 | Local | SV | RSU | UAV0 | UAV1 | Cloud | Task Utility 总和 | 平均时延 (s) | 平均车辆能耗 (J) | tie break |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | Legacy | 16 | 0 | 3 | 2 | 2 | 2 | 7 | 12.4934542617 | 9.2030706493 | 0.2976946745 | 0 |
| 100 | Stage3B | 16 | 0 | 0 | 2 | 1 | 1 | 12 | 14.2057007448 | 5.2066638184 | 0.0777123846 | 0 |
| 200 | Legacy | 15 | 0 | 3 | 2 | 3 | 2 | 5 | 11.4493450509 | 9.5703719146 | 0.3769935089 | 0 |
| 200 | Stage3B | 15 | 0 | 0 | 1 | 1 | 1 | 12 | 13.5254178141 | 4.7056099954 | 0.0469077657 | 0 |

两槽重放中，新模式的 Task Utility 均高于对应旧模式，平均任务时延和车辆能耗均较低；Cloud 执行数增加。这个小样本结果只描述该固定状态下 Matching 的选择，不证明其他 seed、slot 或训练策略下也会如此。Stage3B 两槽 Cloud CPU 用量分别为约 `59,999,999,999.9526` 与 `60,000,000,000` cycles/s，均满足 \(\leq60\) GHz。槽 100 的微小余量来自浮点计算。

### 可行性与兼容性检查

- 两种模式下，最终执行节点数之和分别等于 16、15 个任务；逐任务效用条目数与任务数相等。
- Legacy 最终匹配与已完成 Cloud 门控对照结果的逐任务执行节点一致；Task Utility 和平均时延在设定数值容差内一致。
- 每个重放结果均通过 Matching 稳定检查。将最终匹配送入完整资源评价器后，资源可行且 Task Utility 与增量 Matching 结果一致。
- 增量结果与完整评价器的 Edge CPU 和带宽分配逐节点、逐任务一致；Stage3B Cloud CPU 分配与完整评价器逐任务一致，Cloud 总分配不超过容量。
- `tie_break_count=0`，因此这两个状态没有实际触发平局规则。
- 项目正式代码模块通过静态语法解析，旧冻结配置与新增配置均可解析；`git diff --check` 通过。

## 风险与边界

1. 60 GHz 仅是实验锚点，不是经硬件、运营商合同或实测负载标定的物理结论。
2. Cloud CPU 已共享，但 200 Mbps uplink 仍为每任务独立等效链路；高并发下 Cloud 网络侧没有共享瓶颈，可能继续使 Cloud 具有较强吸引力。
3. 当 Cloud 任务成员变化时需重算整个 Cloud CPU KKT，可能增加 Matching 计算量。当前两槽小规模检查通过，但未做高并发 Runtime 压测。
4. Cloud 最小 CPU 份额约束可能使大任务组不可行；配置和 KKT 会拒绝不可行结果，需在后续负载规模下评估。
5. 平局规则只影响容差内且能减少 Cloud 数量的候选，两个验证槽均未触发；其跨场景影响仍需后续验证。
6. 本次仅一个 seed、两个 slot，且联合切换门控与 CPU 合同，不能据此断言容量池单独带来的因果效果，也不能推断 Cache UPDATE 或 UAV MOVE 的长期价值。
7. 工作树可见其他非 Stage3B 变更；本阶段未触碰 `src/ppo_simple/evaluator.py`、既有运行目录或其他诊断结果。

## 最终状态

```ini
STAGE3B_IMPLEMENTATION_DONE = YES
LEGACY_COMPATIBILITY = PASS
CLOUD_CPU_CONSTRAINT = PASS
MATCHING_KKT_INTEGRATION = PASS
READY_FOR_STAGE3C = YES
```
