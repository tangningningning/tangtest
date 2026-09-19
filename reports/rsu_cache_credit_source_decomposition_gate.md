# RSU_CACHE_CREDIT_SOURCE_DECOMPOSITION_GATE

## 结论

本轮 direct-return 分解支持：

```text
PRIMARY_DIAGNOSIS = CACHE_GLOBAL_TRAJECTORY_CREDIT_ISOLATION_PROBLEM
```

当前问题更符合 global joint-action credit 对 node-specific RSU cache learning 的隔离不足/高方差，而不是 GAE 数学错误、centered normalization sign distortion 或 critic bootstrap 已完全失效。

证据：

- RAW_GLOBAL_GAE 与 critic-independent H10 direct return：7/12 对齐；
- ACTOR_NORMALIZED_GLOBAL_GAE 与 H10：7/12 对齐；
- RAW→NORMALIZED sign flip：0/12；
- one-step `LOCAL_Q_H1 = r_t + V(next)` 与 H10 direct：9/12 对齐。

因此 normalized GAE 没有在本组 witness 中改变符号，critic one-step Q 也比 raw global GAE 更接近 H10 direct truth。剩余不一致主要保留在 global trajectory-level credit isolation。

## 最终字段

```text
WITNESS_COUNT = 12

SYSTEM_MODEL_ACTION_LANDSCAPE_VALID = YES

H10_BENEFICIAL_SELECTED_COUNT = 3
H10_HARMFUL_SELECTED_COUNT = 9

RAW_TO_NORMALIZED_SIGN_FLIP_COUNT = 0

RAW_GAE_H3_SIGN_ALIGNMENT = 6
RAW_GAE_H3_SIGN_MISMATCH = 6

RAW_GAE_H10_SIGN_ALIGNMENT = 7
RAW_GAE_H10_SIGN_MISMATCH = 5

NORMALIZED_GAE_H10_SIGN_ALIGNMENT = 7
NORMALIZED_GAE_H10_SIGN_MISMATCH = 5

LOCAL_Q_H3_SIGN_ALIGNMENT = 8
LOCAL_Q_H3_SIGN_MISMATCH = 4

LOCAL_Q_H10_SIGN_ALIGNMENT = 9
LOCAL_Q_H10_SIGN_MISMATCH = 3

DIRECT_RETURN_USES_CRITIC = NO

PRIMARY_DIAGNOSIS = CACHE_GLOBAL_TRAJECTORY_CREDIT_ISOLATION_PROBLEM

SYSTEM_MODEL_CHANGED = NO
ALGORITHM_CHANGED = NO
TRAINING_EXECUTED = NO
OPTIMIZER_STEP_EXECUTED = NO

EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO

TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
```

## 1. 输入与 witness

- checkpoint：`runs/ppo_recovery_r4_wmove_0p50_exactcandidate_dev3_retry_20260919/latest_checkpoint.pt`；已验证 `completed_episodes=3`。
- frozen realization：`runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20001`。
- production rollout：当前 R4 epoch3 policy 的 stochastic production action，sampling seed = 4401。
- witness selection：沿用上一轮 RSU gate 的固定 anchors `[0, 60, 120, 180, 240, 299]`，每个 anchor 选择对应 RSU 最近的 actual-changing state；先限制为非终止 `slot<=289`，因此末端原 witness 由更早 eligible state 替代，不根据 branch 结果筛选。

最终 witness：

```text
RSU_0 = [1, 62, 121, 180, 240, 288]
RSU_1 = [1, 59, 120, 181, 240, 289]
```

## 2. Direct counterfactual 定义

每个 witness 只比较两个首步 branch：

- A：production selected RSU primitive；
- B：同一 joint trigger、同一 UAV action、同一非目标 cache decisions 下，将目标 RSU conditional primitive 替换为 `NO_CHANGE`。

两条首步 branch 都经过正式 `execute_slot`、Matching/KKT、正式 reward 和正式 next state。首步之后，A/B 从各自 next state 分开使用 frozen R4 epoch3 policy 的 deterministic production action，继续 9 个 slot。

```text
H1_DIRECT  = r_t
H3_DIRECT  = r_t + r_t+1 + r_t+2
H10_DIRECT = sum(r_t ... r_t+9)
```

这些 direct return 只使用正式 environment reward，明确不加 `V(next)` 或任何 critic bootstrap。`LOCAL_Q_H1` 仅作为额外 diagnostic：

```text
LOCAL_Q_H1 = r_t + V(next)
```

`RAW_GLOBAL_GAE` 和 actor 使用的 `centered_std GLOBAL_GAE` 均来自 300-slot production rollout；未执行 actor/critic update。

## 3. Witness 结果

表中 `ΔHk` 是 A 相对 B 的 direct return 差；`ΔQ_H1` 是 one-step local-Q 差；`T/C/S` 分别是首步 task utility、cache cost、system utility 差；`matchΔ` 是首步 Matching 中被重新分配的 task 数。

| RSU | slot | op | RAW sign | NORM sign | ΔH1 | ΔH3 | ΔH10 | ΔQ_H1 | T/C/S at H1 | matchΔ | RAW-H10 | NORM-H10 | Q-H10 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|
| RSU_0 | 1 | DELETE | - | - | -0.010 | +4.010 | -18.462 | +0.042 | 0.000/+0.010/-0.010 | 0 | ✓ | ✓ | × |
| RSU_0 | 62 | ADD | + | + | -0.131 | -0.131 | -0.131 | -0.151 | 0.000/+0.131/-0.131 | 0 | × | × | ✓ |
| RSU_0 | 121 | DELETE | + | + | -0.010 | -1.047 | -1.263 | -0.005 | 0.000/+0.010/-0.010 | 0 | × | × | ✓ |
| RSU_0 | 180 | SWAP | + | + | +0.382 | -0.849 | +3.888 | +0.281 | +0.434/+0.053/+0.382 | 1 | ✓ | ✓ | ✓ |
| RSU_0 | 240 | SWAP | - | - | -9.986 | -9.986 | -9.986 | -9.769 | -9.838/+0.148/-9.986 | 6 | ✓ | ✓ | ✓ |
| RSU_0 | 288 | ADD | + | + | -0.049 | +0.369 | -0.832 | -0.352 | 0.000/+0.049/-0.049 | 0 | × | × | ✓ |
| RSU_1 | 1 | DELETE | - | - | -0.010 | -0.034 | -0.034 | +0.003 | 0.000/+0.010/-0.010 | 0 | ✓ | ✓ | × |
| RSU_1 | 59 | DELETE | + | + | +10.619 | +10.607 | +10.710 | +10.697 | +10.629/+0.010/+10.619 | 2 | ✓ | ✓ | ✓ |
| RSU_1 | 120 | ADD | + | + | -0.091 | -0.091 | -18.840 | -0.162 | 0.000/+0.091/-0.091 | 0 | × | × | ✓ |
| RSU_1 | 181 | SWAP | + | + | +0.548 | +2.734 | +1.885 | +0.354 | +0.679/+0.131/+0.548 | 2 | ✓ | ✓ | ✓ |
| RSU_1 | 240 | DELETE | - | - | -0.010 | -0.056 | -23.603 | +0.560 | 0.000/+0.010/-0.010 | 0 | ✓ | ✓ | × |
| RSU_1 | 289 | DELETE | + | + | -0.010 | -6.515 | -12.058 | -0.083 | 0.000/+0.010/-0.010 | 0 | × | × | ✓ |

`H10_BENEFICIAL_SELECTED_COUNT=3` 的 positive branches 为 RSU_0 slot 180、RSU_1 slot 59、RSU_1 slot 181；其余 9 个 selected branch 的 H10 direct delta 为负，因此 system objective 对 RSU action 存在 state-dependent discrimination。

首步正式 offloading 变化的代表性 branch 为：

- RSU_0 slot 180：execution counts 从 `CLOUD=3, LOCAL=1, RSU=0, SV=3, UAV=6` 变为 `CLOUD=3, LOCAL=1, RSU=1, SV=2, UAV=6`，task utility `+0.434`，system utility `+0.382`；
- RSU_0 slot 240：从 `CLOUD=4, RSU=2, SV=4, UAV=6` 变为 `CLOUD=0, RSU=5, SV=5, UAV=6`，task utility `-9.838`，system utility `-9.986`；
- RSU_1 slot 59：从 `CLOUD=9, RSU=2, SV=1, UAV=2` 变为 `CLOUD=11, RSU=0, SV=1, UAV=2`，task utility `+10.629`，system utility `+10.619`；
- RSU_1 slot 181：从 `CLOUD=3, RSU=1, SV=1, UAV=9` 变为 `CLOUD=3, RSU=2, SV=0, UAV=9`，task utility `+0.679`，system utility `+0.548`。

因此 `SYSTEM_MODEL_ACTION_LANDSCAPE_VALID=YES`：RSU cache action 的正式 branch 确实可以改变 offloading、task utility 和 system utility，同时也存在只体现为 cache cost 的 state。

## 4. 三类原因分解

### A. SYSTEM_MODEL

不支持 system model 是主因。正式 branch 中既有正的 H10 direct utility，也有负的 H10 direct utility；offloading 改变与 utility 改变在正式 Matching/KKT 结果中可复现。

### B. CACHE CREDIT ALGORITHM

这是当前主要问题。RAW GAE 对 H10 direct 只有 7/12 对齐，仍有 5 个 state 的 global trajectory sign 与 critic-independent direct truth 不同；但 local-Q H10 达到 9/12。应将现象描述为 global joint-action credit 对 node-specific cache contribution 的隔离不足/高方差，不应描述为 GLOBAL_GAE 数学错误。

### C. CRITIC / ADVANTAGE NORMALIZATION

- centered normalization 在本组 12 个 witness 上没有造成任何 raw-to-normalized sign flip；因此 `CACHE_GAE_NORMALIZATION_SIGN_DISTORTION` 不成立。
- one-step local-Q 与 H10 为 9/12 对齐，不能支持 `CRITIC_BOOTSTRAP_UNRELIABLE` 作为主诊断。H3 为 8/12，说明短 horizon 与 H10 仍有少量 temporal difference，但不是主要失败源。

## 5. 变更审计

本轮没有修改 reward、system model、cache cost、RSU capacity、Cache policy、GAE、normalization、critic、PPO 或 UAV。没有生成 episode，没有执行 SUMO，没有训练，没有调用 optimizer step；临时诊断脚本和 Python 缓存已在结束后删除。
