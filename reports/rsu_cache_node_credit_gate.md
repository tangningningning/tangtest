# RSU_CACHE_NODE_CREDIT_GATE

## 结论

当前 `CACHE_ADV_SOURCE=GLOBAL_GAE` 不能稳定、正确地反映 RSU cache action 对后续 offloading / task utility / system utility 的真实贡献。

在 12 个来自 frozen `episode_20001` 的实际、非终止 RSU cache witness 上，正式 actor-consumed `GLOBAL_GAE` 符号与目标 RSU 的 exact counterfactual local-Q 符号只有 6/12 对齐，6/12 不对齐。其中 2 个是 GLOBAL_GAE 漏记有益更新，4 个是 GLOBAL_GAE 支持了局部 Q 为负的过度更新。因此本轮 gate 的主要诊断是 **CACHE_GLOBAL_GAE_CREDIT_MISALIGNMENT**。

本轮只做诊断。没有训练、没有 optimizer step、没有修改 reward/cache policy/UAV/RSU 参数，也没有修改生产源码或 checkpoint。

## 最终 gate 字段

```text
RSU_STRUCTURAL_PATH_VALID = YES
RSU_CACHE_AFFECTS_OFFLOADING = YES

WITNESS_STATE_COUNT = 12
BENEFICIAL_COUNTERFACTUAL_COUNT = 87
HARMFUL_COUNTERFACTUAL_COUNT = 163

GLOBAL_GAE_SIGN_ALIGNMENT_COUNT = 6
GLOBAL_GAE_SIGN_MISMATCH_COUNT = 6

MISSED_BENEFICIAL_UPDATE_COUNT = 2
OVER_UPDATE_COUNT = 4
CONSISTENT_COUNT = 6

PRIMARY_DIAGNOSIS = CACHE_GLOBAL_GAE_CREDIT_MISALIGNMENT

TRAINING_EXECUTED = NO
OPTIMIZER_STEP_EXECUTED = NO

SOURCE_MODIFIED = NO
REWARD_CHANGED = NO
UAV_CHANGED = NO

EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO

TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
```

## 1. 当前 Cache action 的真实粒度

静态代码路径和 checkpoint rollout 均确认：当前 action 不是一个不存在的“per-node cache policy head”，而是：

1. 一个 joint binary cache-trigger vector；
2. 对每个被触发 cache node，再使用该 node 对应的 conditional primitive candidate。

当前配置为 `policy_primitive_r2_joint_cache_uav_operation`，action dimension 为 11：9 个 cache trigger 加 2 个 UAV movement action。Cache head tensor 为 `[9, 36]`，9 个 cache head 的实际 node 顺序为：

| head_index | node | role | action label |
|---:|---|---|---|
| 0 | 46 | SV | `z_46^ca` |
| 1 | 10 | SV | `z_10^ca` |
| 2 | 0 | SV | `z_0^ca` |
| 3 | 23 | SV | `z_23^ca` |
| 4 | 59 | SV | `z_59^ca` |
| 5 | RSU_0 | RSU | `z_RSU_0^ca` |
| 6 | RSU_1 | RSU | `z_RSU_1^ca` |
| 7 | UAV_0 | UAV | `z_UAV_0^ca` |
| 8 | UAV_1 | UAV | `z_UAV_1^ca` |

因此 RSU_0 / RSU_1 存在真实的 `head_index=5/6`，且 conditional action record 中实际带有 `head_id`、`head_index`、`operation_type`、added/removed service IDs 和 `actual_changed`。RSU decision 不是通过假设出来的独立 operation head 表达的。

## 2. Exact counterfactual 方法

- checkpoint：`runs/ppo_recovery_r4_wmove_0p50_exactcandidate_dev3_retry_20260919/latest_checkpoint.pt`，checkpoint epoch = 3。
- episode：frozen `episode_20001`，不重新生成 episode。
- witness：从当前实际 rollout 的 actual-changing RSU action 中选择 12 个 bounded witness，RSU_0 与 RSU_1 各 6 个；排除 terminal transition。
- 每个 branch 只替换目标 RSU 的合法 cache candidate；UAV operation、非目标 cache decisions、vehicle/task/prediction 和其余 transition inputs 保持不变。
- 每个 branch 均经过正式 `execute_slot`、Matching/KKT、正式 reward 和正式 next state；没有使用简化 offloading 或手工 reward。
- `Δtask utility`、`Δcache cost`、`Δsystem utility` 均相对于该 witness 的 target-RSU `NO_CHANGE` branch。
- node-local exact counterfactual Q 使用正式 branch system utility 加上 checkpoint critic 对正式 next observation 的 bootstrap（`gamma=1.0`）。
- `GLOBAL_GAE_SIGN` 使用当前配置实际送入 actor 的 centered-std normalized GAE 符号；raw GAE 同时保留作诊断，不用 raw 符号替代正式 actor sign。

`BENEFICIAL_COUNTERFACTUAL_COUNT=87` 和 `HARMFUL_COUNTERFACTUAL_COUNT=163` 统计的是 12 个 witness 中所有非 `NO_CHANGE` 合法 candidate 相对于 target-RSU `NO_CHANGE` 的 local-Q 差分，不是只统计 production 已选 branch。

## 3. Witness 结果

表中 `local ΔQ` 是实际 production 选中 candidate 相对 `NO_CHANGE` 的 exact counterfactual Q 差；`exec Δ` 是 execution-node count 的变化；`RSU svc Δ` 是目标 RSU cached-service 数量的变化；`B/H` 是该 witness 的所有合法非 `NO_CHANGE` candidate 中 local-Q 有益/有害数量。

| slot | RSU | actual op | legal | formal GAE | local ΔQ | Δtask | Δcache | Δsystem | affected tasks | exec Δ | RSU svc Δ | B/H |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | RSU_0 | DELETE | 20 | - | +0.042 | 0.000 | +0.010 | -0.010 | 0 | `{}` | -1 | 11/8 |
| 1 | RSU_1 | DELETE | 14 | - | +0.003 | 0.000 | +0.010 | -0.010 | 0 | `{}` | -1 | 11/2 |
| 56 | RSU_0 | DELETE | 24 | + | +15.624 | +15.638 | +0.010 | +15.628 | 5 | `CLOUD+4, RSU-4` | -1 | 10/13 |
| 63 | RSU_1 | ADD | 20 | + | -0.131 | 0.000 | +0.066 | -0.066 | 0 | `{}` | +1 | 4/15 |
| 117 | RSU_1 | DELETE | 19 | + | +0.229 | 0.000 | +0.010 | -0.010 | 0 | `{}` | -1 | 10/8 |
| 118 | RSU_0 | ADD | 26 | + | -0.122 | 0.000 | +0.034 | -0.034 | 0 | `{}` | +1 | 9/16 |
| 175 | RSU_0 | DELETE | 27 | + | +0.038 | +0.004 | +0.010 | -0.006 | 1 | `CLOUD+1, RSU-1` | -1 | 7/19 |
| 181 | RSU_1 | SWAP | 20 | + | +0.354 | +0.679 | +0.131 | +0.548 | 2 | `RSU+1, SV-1` | 0 | 6/13 |
| 229 | RSU_0 | ADD | 22 | - | -0.525 | 0.000 | +0.049 | -0.049 | 0 | `{}` | +1 | 5/16 |
| 237 | RSU_1 | ADD | 20 | - | -90.414 | -89.898 | +0.124 | -90.021 | 5 | `CLOUD-5, RSU+5` | +1 | 2/17 |
| 297 | RSU_0 | ADD | 30 | + | -2.267 | 0.000 | +0.034 | -0.034 | 0 | `{}` | +1 | 9/20 |
| 298 | RSU_1 | ADD | 20 | + | -7.801 | 0.000 | +0.080 | -0.080 | 0 | `{}` | +1 | 3/16 |

### Sign comparison

- Alignment：slot 56 RSU_0、117 RSU_1、175 RSU_0、181 RSU_1、229 RSU_0、237 RSU_1，共 6。
- Missed beneficial update：slot 1 RSU_0、slot 1 RSU_1。GLOBAL_GAE 为负，但 actual RSU candidate 的 local ΔQ 为正，共 2。
- Over-update：slot 63 RSU_1、118 RSU_0、297 RSU_0、298 RSU_1。GLOBAL_GAE 为正，但 actual RSU candidate 的 local ΔQ 为负，共 4。

## 4. 是否真实影响 offloading

`RSU_CACHE_AFFECTS_OFFLOADING = YES`。

原因不是仅观察 cache service 数量变化，而是 exact branch 的正式执行结果发生了 downstream 变化：12/12 个 witness 都至少存在一个合法 RSU candidate branch 会改变 task utility；各 witness 的 branch execution-node change 数分别为 7、8、12、12、6、10、17、6、8、8、18、7；对应 affected-task branch 也均非空。典型直接证据包括：

- slot 56 / RSU_0：`Δtask=+15.638`，`CLOUD+4, RSU-4`，affected tasks = 5；
- slot 181 / RSU_1：`Δtask=+0.679`，`RSU+1, SV-1`，affected tasks = 2；
- slot 237 / RSU_1：`Δtask=-89.898`，`CLOUD-5, RSU+5`，affected tasks = 5。

这证明 RSU cache branch 能够改变 executable node set、offloading result 和 task/system utility。但也有部分真实 cache change 只改变 cache cost/service count 而没有立即改变当前 slot 的 offloading，例如 slot 1、63、117、118、229、297、298；这不否定结构路径有效，只说明 action effect 具有 state-dependent、时序性的稀疏性。

## 5. 对问题 A/B/C 的回答

### A. RSU cache action 是否真实改变 executable node set、offloading result、task/system utility？

是。正式 branch 中存在 execution-node set、affected task、task utility 和 system utility 同时变化的 counterfactual；因此 RSU cache action 不是只改变 cache bookkeeping。

### B. GLOBAL_GAE 是否把收益正确给到产生动作的 RSU cache decision？

不完全正确。本 gate 的 actual-branch sign test 为 6/12 alignment、6/12 mismatch。GLOBAL_GAE 同时出现漏记有益 RSU update 和支持局部 Q 为负的 over-update，说明 global trajectory-level credit 没有稳定隔离目标 RSU cache decision 对后续结果的贡献。

### C. 下一版本是否应设计 node-specific cache credit？

本轮只完成 gate，没有修改 credit。由于 mismatch 已达到 50%，且误差方向同时包含 missed beneficial 与 over-update，下一版本可以在保持当前 policy/action structure 的前提下设计 node-specific cache credit，并用同样的 exact counterfactual gate 验证；本报告不实施该改动。

## 6. 运行与变更审计

```text
TRAINING_EXECUTED = NO
OPTIMIZER_STEP_EXECUTED = NO
SOURCE_MODIFIED = NO
REWARD_CHANGED = NO
UAV_CHANGED = NO
EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
```

最终诊断脚本只在临时路径运行并已删除；checkpoint 未写回，生产源码未修改。branch diagnostic 的计时约为 3.83 秒（不含 Python import、checkpoint load 和 rollout 初始化）。
