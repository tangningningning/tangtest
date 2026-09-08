# `ppo_simple` 到真实环境 non-neutral action 完整链审计

审计日期：2026-09-08  
审计范围：当前工作区最新代码、一个真实 fixture 的同一初始 state/slot。  
本审计不修改算法、系统模型、reward、Matching/KKT、Cache/UAV optimizer 或 PPO 参数；未进行正式 PPO 训练。

## 1. Git 基线

| 项目 | 结果 |
|---|---|
| branch | `main`，tracking `origin/main` |
| HEAD | `b7e90d4 Fix default formal training entrypoint` |
| 审计开始时 tracked diff | 无 |
| 审计开始时 `git diff --check` | 通过，无输出 |
| 原有未跟踪文件 | `打包并上传.bat`，未修改、未删除 |
| commit/push/merge/rebase | 均未执行 |

审计过程中只临时创建了一个 replay/capture 脚本；报告写入后已删除。源码没有修改。

## 2. 验证方法与固定 fixture

使用：

- `trainer.load_environment_context("system_config.toml")`；
- `trainer.aligned_train_steps(origin_start=0, max_steps=1, include_bootstrap_step=False)`；
- `origin_index=0`、`slot=0`、`time_s=100.0`；
- 真实 fixture 的 15 个 actual tasks；
- 初始 Cache shape 为 `(9, 10)`，9 个 Cache/service node 的初始 Cache item 数为 `[2,1,2,1,2,4,5,3,4]`；
- 初始 UAV：`UAV_0=[3364.03,1119.88]`、`UAV_1=[5179.65,1710.41]`。

四个 case 都从完全相同的初始 Cache/UAV state 开始：

| case | raw PPO action（按 action spec 顺序） | non-zero label |
|---|---|---|
| neutral | `[0,0,0,0,0,0,0,0,0,0,0]` | 无 |
| cache-only | `[1,0,0,0,0,0,0,0,0,0,0]` | `z_23^ca=1` |
| UAV-only | `[0,0,0,0,0,0,0,0,0,1,0]` | `z_UAV_0^q=1` |
| cache+UAV | `[1,0,0,0,0,0,0,0,0,1,0]` | `z_23^ca=1`, `z_UAV_0^q=1` |

动作由真实 `BernoulliActorCritic.sample()` 生成；仅在内存中将 actor logits 设为确定性 probe，以便在不训练 PPO 的情况下精确选择上述 binary action。没有绕过 `policy.sample()`、action mask、interface 或 `execute_slot()`。replay wrapper 只捕获函数输入/输出，执行后立即恢复。

## 3. 实际调用链

1. `src/ppo_simple/trainer.py:179` 准备真实 aligned step；`PPOV2Trainer.collect_chunk()` 调用 `rollout.collect_rollout()`。
2. `src/ppo_simple/rollout.py:134-152` 用 previous Cache/UAV state 构造 observation 和 action-mask provider，再调用 `policy.sample()`。
3. `src/ppo_simple/policy.py:132-172` 逐 bit 采样。每个 bit 的 mask 先变成 binary categorical distribution 的 `-inf` 屏蔽项；选中的 bit 原样写入 `PolicySample.action`。
4. `src/ppo_simple/interface.py:67-75` 将 action vector 的前 9 位直接 split 为 `cache_triggers`，后 2 位直接 split 为 UAV triggers。
5. `src/ppo_simple/rollout.py:156-172` 将这两个 trigger 向量和真实 actual tasks/vehicle positions 传给 `decision.slot_environment.execute_slot()`。
6. `src/decision/slot_environment.py:224-280`：全零 action 走 neutral fast path；任意 non-neutral action 调用 `optimize_joint_configuration()`。
7. `src/decision/slot_environment.py:969-1028`：joint optimizer 先用 `previous_cache` 做 Cache block，再用 `q_prev` 做 UAV block，产生 `final_cache_state` 和 `final_uav_positions`。
8. `src/decision/slot_environment.py:342-358`：同一个 slot 立即用 configuration 的 final Cache/UAV state 构造 service positions，并把 `final_cache` 传给 `task_allocation.optimize_task_matching()`。
9. `src/decision/task_allocation.py:394-500` 生成 feasible/candidate targets 并运行 Transfer-Exchange Matching；`evaluate_matching_with_kkt()`（`1375` 起）计算 CPU/BW KKT allocation、delay、vehicle energy、per-task utility 和 `Q(t)`。
10. `src/decision/slot_environment.py:407-417` 计算 `G(t)=Q(t)-cache_cost-movement_cost`；`src/ppo_simple/rollout.py:69-74` 再次核对该恒等式并将 `system_utility` 作为 reward。
11. `src/decision/slot_environment.py:465-466` 把本 slot 的 final state 作为 next state；`rollout.py:218-219` 将它传给下一 transition。因此 configuration state 在当前 slot 的 Matching 生效，也作为下一 slot 的 reference state。

## 4. Mask 与 action 是否被变成 no-op

**CONFIRMED：本 fixture 的 11 个 action index 全部返回 `[True, True]`。**

- Cache index `0..8` 在 `interface.py:332-333` 固定允许 0/1。
- UAV index 在 `interface.py:335-349` 通过 `has_feasible_trigger_assignment()` 检查；本 fixture 中 `z_UAV_0^q=1` 和其余 UAV request 均可行，因此 mask 也为 `[True, True]`。
- 四个 case 的每个 selected bit 都满足 `action_masks[index, selected_bit] == True`。
- `action_vector_to_triggers()` 没有把 1 重置为 0；capture 到的 configuration input 与 raw action 一致。

所以本次 cache no-op 不是 mask/filter/reset 造成的；UAV 1 也确实穿过了 mask、interface 和 environment。

## 5. 四个 action 的实际结果

### 5.1 结果总表

| case | optimizer input | Cache after vs before | UAV displacement | delay sum (s) | vehicle energy sum (J) | `Q(t)` | cache cost | movement cost | `G(t)` = reward |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| neutral | 无 configuration call；全零 gate | 0 个 node 改变 | `UAV_0=0`, `UAV_1=0` m | 163.487371 | 13.633249 | 10.016961 | 0 | 0 | 10.016961 |
| cache-only | Cache node 23 request=1；UAV 0/0 | 0 个 node 改变 | 0, 0 m | 163.487371 | 13.633249 | 10.016961 | 0 | 0 | 10.016961 |
| UAV-only | Cache 全 0；`UAV_0=1`, `UAV_1=0` | 0 个 node 改变 | `15.0`, `0` m | 163.408497 | 13.617512 | 10.020562 | 0 | 0.005720 | 10.014842 |
| cache+UAV | Cache node 23=1；`UAV_0=1` | 0 个 node 改变 | `15.0`, `0` m | 163.408497 | 13.617512 | 10.020562 | 0 | 0.005720 | 10.014842 |

所有 case `step_feasible=True`，`reward_identity_error=0`。相对 neutral：

- cache-only：`delta_G=0`、`delta_reward=0`；
- UAV-only/cache+UAV：`delta_Q=+0.003600316708333`，但 movement cost 扣除后 `delta_G=delta_reward=-0.002119683291668`。

### 5.2 Cache-only：request 到达，但实际配置没有变化

**FAILED：在该真实 state/slot，`z_23^ca=1` 没有产生 Cache state change。**

捕获到的路径如下：

- effective Cache request：`[1,0,0,0,0,0,0,0,0]`；effective UAV request：`{"UAV_0":0,"UAV_1":0}`；
- configuration optimizer input 的 `previous_cache_state`、`previous_uav_positions` 是初始 reference state，request 没有被清零；
- configuration output 的 final Cache 与 previous Cache 完全相同；`actual_cache_update=[0,0,0,0,0,0,0,0,0]`；
- `accepted_cache_updates=0`、`stop_reason=converged_by_no_improvement`；
- node 23 diagnostic：`requested_trigger=1`、`actual_update=0`、`best_delta_objective=0`、`candidate_evaluations_recorded=8`；
- Matching input 的 `edge_cache_state` 与 neutral 完全相同；service node positions 也完全相同；
- Matching/KKT、delay、energy、Q、G、reward 与 neutral 完全相同。

第一个出现“预期 Cache 应变化、实际没有变化”的具体位置是 `src/decision/cache_configuration.py`：

- 初始化候选在 `1524` 只有 `candidate_objective > current_objective + config.tolerance` 才接受；
- coordinate candidate 在 `1807-1810` 只有严格 objective improvement 才写回 `current_cache[node_index]`；本次 node 23 的最佳 delta 为 0，因此两处均不接受；
- 随后 `1846-1849` 通过 `np.any(current_cache != previous_cache, axis=1)` 生成 `actual_cache_trigger=0`。

因此这是 optimizer 的 objective acceptance 结果，不是 PPO action mask 或 interface 丢失 action。

### 5.3 UAV-only：request 到达并在当前 slot 生效

**CONFIRMED：`z_UAV_0^q=1` 到达 UAV optimizer，并产生实际 UAV state change。**

- effective request：`{"UAV_0":1,"UAV_1":0}`；mask 没有屏蔽 selected 1；
- UAV optimizer input 的 `previous_uav_positions` 为 `[3364.03,1119.88]` / `[5179.65,1710.41]`；
- final positions 为 `UAV_0=[3379.03,1119.88]`、`UAV_1=[5179.65,1710.41]`，即 UAV_0 移动 15 m；
- `actual_uav_movement={"UAV_0":1,"UAV_1":0}`，movement energy 为 `225.0 J`，movement cost 为 `0.00572`；
- UAV diagnostic 的 `best_delta_objective=0.02029310254262029`、candidate evaluations 为 76；
- Matching input 的 UAV_0 service position 已经是 `[3379.03,1119.88]`，不是旧位置；因此变化在当前 slot 的 Matching/KKT 中生效，而非延迟到下一 slot。

### 5.4 cache+UAV

**CONFIRMED：组合 action 的 UAV 部分生效；Cache 部分仍被 optimizer 保持不变。**

该 case 的 final configuration 与 UAV-only 相同：Cache node 23 没有变化，UAV_0 移动 15 m。因而 Matching、KKT allocations、delay、energy、Q、G、reward 与 UAV-only 相同。它同时证明：Cache request 和 UAV request 可以在同一个 action 中到达 joint optimizer，但 Cache request 不会强制产生 Cache mutation。

## 6. Matching、candidate nodes 与资源分配

四个 replay 的 candidate target 集相同；UAV_0 移动 15 m 没有改变本 slot 的可执行候选集合。完整 feasible target 摘要为：

```text
8:CLOUD
16:RSU_0,RSU_1,UAV_0,UAV_1
18:58
19:RSU_0
31:RSU_0,RSU_1,UAV_1
35:23,RSU_1,UAV_0,UAV_1
47:23,RSU_1,UAV_0,UAV_1
48:43,RSU_1
50:87,75,UAV_0
55:87,75,UAV_0
64:23,RSU_1,UAV_0,UAV_1
73:23,RSU_1,UAV_0,UAV_1
83:23,RSU_1,UAV_0,UAV_1
88:RSU_0
98:CLOUD
```

四个 case 的最终 Matching map 相同：

```text
8:CLOUD, 16:UAV_0, 18:58, 19:RSU_0, 31:UAV_1,
35:UAV_1, 47:UAV_0, 48:RSU_1, 50:UAV_0, 55:87,
64:UAV_1, 73:23, 83:UAV_0, 88:RSU_0, 98:CLOUD
```

Matching 均为 `updates=9`、`transfer_updates=9`、`exchange_updates=0`、`candidate_evaluations_total=348`、`transfer_exchange_stable=True`。但 UAV movement 改变了 UAV_0 的无线条件，KKT bandwidth allocation 和 per-task delay/energy 随之改变。

CPU allocation（四个 case 观察到相同）为：

```text
58:{18:1.000000}
87:{55:1.000000}
RSU_0:{19:0.492935890, 88:0.507064110}
RSU_1:{48:1.000000}
23:{73:1.000000}
UAV_0:{16:0.443758617, 47:0.127907374, 50:0.268998380, 83:0.159335630}
UAV_1:{31:0.626634487, 35:0.190446772, 64:0.182918742}
```

Bandwidth allocation 的非 UAV_0 部分四个 case 相同：

```text
58:{18:1.000000}
87:{55:1.000000}
RSU_0:{19:0.149485545, 88:0.850514455}
RSU_1:{48:1.000000}
23:{73:1.000000}
UAV_1:{31:0.505474376, 35:0.254270604, 64:0.240255021}
```

UAV_0 的 bandwidth allocation 从 neutral/cache-only 的

```text
{16:0.273167202, 47:0.193148772, 50:0.333409213, 83:0.200274813}
```

变为 UAV-only/cache+UAV 的

```text
{16:0.274458423, 47:0.192735632, 50:0.332507638, 83:0.200298308}
```

这与 observed delay、vehicle energy 和 `Q(t)` 变化一致。

## 7. 状态、时间对齐与 reward 检查结论

| 检查项 | 状态 | 证据 |
|---|---|---|
| binary `1` 是否被 mask/filter/reset | **CONFIRMED：没有** | 四个 case selected bit 均被允许；effective request 与 raw action 一致 |
| configuration state 在当前还是下一 slot 生效 | **CONFIRMED：当前 slot** | `execute_slot()` 先取 `configuration.final_*`，再构造 service positions 并调用 Matching；之后才写 `next_*` |
| Matching/KKT 用旧 state 还是新 state | **CONFIRMED：用新 state** | `final_cache` 和 `final_uav` 分别进入 `optimize_task_matching()` 的 edge cache/service positions |
| reward 使用的 G(t) 是否时间对齐 | **CONFIRMED：对齐** | reward 直接等于 actual Matching/KKT 的 `Q(t)` 减 actual configuration costs；四个 case 恒等式误差均为 0 |
| 是否出现 non-neutral action 但 Cache change=0 | **CONFIRMED：出现** | cache-only、cache+UAV 的 Cache change 均为 0 |
| 是否出现 non-neutral action 但 UAV movement=0 | **FAILED：本次 UAV request 未出现** | UAV-only、cache+UAV 的 UAV_0 均移动 15 m；不代表其他 state/slot 都不会出现 |
| 是否出现 non-neutral action 但 delta_G=0 | **CONFIRMED：Cache-only 出现** | cache-only 相对 neutral 的 `delta_G=0`；UAV request 的 delta_G 非 0 |
| 其他 state/slot、其他 Cache node 的泛化行为 | **NOT VERIFIED** | 本任务只要求并执行同一真实 state/slot 的最小 replay |
| 正式 PPO 训练后的策略分布/性能 | **NOT VERIFIED** | 本任务明确未进行正式训练 |

## 8. 最终结论（针对本次真实 fixture 的 state/slot）

```text
PPO_ACTION_REACHES_ENV = YES
CACHE_ACTION_CHANGES_STATE = NO
UAV_ACTION_CHANGES_STATE = YES
ACTION_CHANGES_G = YES
ACTION_CHANGES_REWARD = YES
```

解释：`PPO_ACTION_REACHES_ENV=YES` 表示 binary action 已完整到达真实 `execute_slot()`、configuration optimizer 和 actual Matching/KKT。`CACHE_ACTION_CHANGES_STATE=NO` 是对本次 `z_23^ca=1` replay 的实测结论：request 到达，但 optimizer 没有找到严格更优的 Cache configuration，所以实际 Cache state 和 G/reward 都不变；这不是 mask 丢 action。UAV request 则实际改变 UAV state，并通过同 slot 的无线/KKT 计算改变 `Q(t)`，最终在扣除 movement cost 后改变 G/reward。

## 9. 清理与最终检查

- 已删除本任务临时 replay/capture 脚本；未保留临时数据或测试文件。
- 未修改源代码、算法参数或配置。
- 最终执行 `git diff --check`：通过，无输出。
- 未 commit、未 push、未 merge、未 rebase。
