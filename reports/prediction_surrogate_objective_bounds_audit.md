# Prediction surrogate objective strict-bound / early-stop audit

## 结论摘要

本轮只审计当前 K=100 prediction surrogate 的数学边界和候选级提前终止条件。没有修改正式代码、K=100 batch 数、PPO、reward、Matching/KKT、Cache/UAV objective、candidate 搜索空间或 worker 数；实际 reference evaluation 仍完整执行 100 个 prediction batches。

结论是：当前正式 PPO 路径可以推导严格的 deterministic remaining-batch bound。每个可行 task utility 的上界是 `1`，因此每个剩余 batch 有一个由已知 arrival/task count、discount 和 scenario normalization 给出的 objective upper bound。Cache/UAV configuration cost 与 prediction batch 无关，可以在 bound 中精确扣除。

这个 bound 可以安全地提前拒绝“即使剩余 batches 全部达到理论最大 utility，也不可能超过 current-best”的候选；它不是 confidence interval，也不使用概率近似。它不能让一个需要成为 current-best 的候选在当前坐标搜索语义下省略最终 exact objective，因为后续 `current_objective`、严格改善判断和下一轮搜索都依赖该 exact value。

在固定真实状态、固定 action、K=100 reference 的离线回放中：

- origin 0：所有 Cache candidates 都可以在第 77–97 个 batch 间安全排除，Cache 最终 no-update decision 可被认证；UAV request-on 的 218 个 candidates 全部仍需 100 batches。
- origin 120：269 个 Cache candidate evaluations 中 210 个可提前排除，但 59 个需要完整 100 batches；因此整个 Cache decision 不能早于完整 reference。
- origin 239：119 个 Cache candidate evaluations 中 101 个可提前排除，但 18 个需要完整 100 batches；整个 Cache decision 不能早于完整 reference。

因此存在严格的 candidate-level exact early stop，但不存在对所有真实 origin/configuration 都有效的“提前结束整个 Cache/UAV decision”规则。

## Scope and protocol

- branch: `main`
- HEAD: `b7e90d4015bbeac682ce7b55a4a29d71a7e174d7`
- K=100、20 scenarios、5 horizons、100 flattened batches；原始 scenario-major/horizon-minor 顺序不变。
- 使用与前一轮 K 审计一致的固定真实状态和 seed=4401 的 normal Bernoulli action：每个 origin 只采样/固定一次 action，并在该 origin 的 K=100 reference path 中复用。
- origin 0 action：Cache node `43` request-on、`UAV_1` request-on。
- origin 120 action：Cache requests on，两个 UAV requests 均为 0。
- origin 239 action：Cache requests on，两个 UAV requests 均为 0。
- 每个 origin 的 K=100 full configuration search 均完整执行；本轮没有实际截断 future 或改变 candidate result。

K=100 observation-only reference counters：

| origin | unique configuration states | prediction Matching calls | total task events in 100 batches | raw per-batch upper total |
|---:|---:|---:|---:|---:|
| 0 | 92 | 9,200 | 1,413 | 63.9114334375 |
| 120 | 177 | 17,700 | 1,358 | 61.5694596875 |
| 239 | 62 | 6,200 | 1,424 | 64.4150212500 |

`raw per-batch upper total` 是 `sum_b beta^h_b * N_b / S`，即把每个 task utility 都替换成上界 `1`；它不是 observed utility，也不是新的 objective 参数。

## 1. Per-batch utility dependency and bound

### 1.1 Actual per-task utility

`src/common/system_model.py:355-381` 定义：

```text
u_v = omega_T * (1 - delay_v / T_ref)
    + omega_E * (1 - energy_v / E_ref)
```

同一函数验证：

- `delay_v >= 0`
- `energy_v >= 0`
- `omega_T, omega_E in [0,1]`
- `omega_T + omega_E = 1`

所以对每个 resource-feasible task：

```text
u_v <= omega_T + omega_E = 1
```

代码没有对 utility 做 clipping；utility 可以为负，不能把它错误地当成 `[0,1]` 概率或 reward。上界 `1` 是严格数学上界，不依赖 candidate、Matching 顺序或历史观测。

### 1.2 Raw batch contribution

`SurrogateBatch` 在 `src/decision/cache_configuration.py:508-513` 保存固定的 `discount`、batch events 和位置。`_evaluate_prediction_batch()` 在 `:650-693` 对一个 batch 执行正式 Matching/KKT，并返回 `task_utility_sum`。

令：

- `N_b = len(batch.events)`；
- `R_b(C,Q) = sum_v u_{b,v}(C,Q)`；
- `h_b = batch.horizon`；
- `S = context.num_scenarios`；
- `beta = config.beta_discount`。

当前 `evaluate_surrogate_context()` 在 `src/decision/cache_configuration.py:997-1016` 按原始 batch 顺序做：

```text
Q_hat(C,Q) = (1/S) * sum_b beta^h_b * R_b(C,Q)
```

因此：

```text
R_b(C,Q) <= N_b
U_b = beta^h_b * N_b / S
```

每个 batch 的 weighted contribution 有严格 upper bound `U_b`。

### 1.3 Formal-path finite lower bound

utility 没有由 `task_utility()` 单独提供一个对任意 API 输入都成立的有限下界：如果允许任意远的 finite position 或任意大的 delay/energy，utility 会继续线性下降。因此不能用“observed minimum”冒充未计算 batch 的证明。

但当前正式 PPO path 的 candidate domain 是有界的，故可为本任务范围推导有限 lower bound：

1. prediction scenario 的 `input_bits`、`cpu_cycles`、vehicle positions 是 finite artifact 数组；
2. SV 位置来自 finite prediction batch，RSU 位置固定；
3. UAV candidate 受 SUMO `flight_boundary` 和 single-slot max movement 约束，当前生成器的 candidate 集合也是有限的；
4. edge KKT fractions 至少为 `epsilon_w`/`epsilon_b`；Cloud 使用固定正资源，Local 使用固定正 CPU；
5. 对 formal candidate domain 取有限最大几何距离 `D_max`。ground/air channel 的正参数和正 altitude 给出 `g_min > 0`，从而 `R_min > 0`；
6. 对任一 task/node 可使用保守上界：

```text
T_edge <= D / (epsilon_b * R_min) + L / (epsilon_w * F_node)
E_edge <= P * D / (epsilon_b * R_min)
```

Local/Cloud 直接使用当前 `system_model` 的 finite execution formula。对每个 task 取所有可能 execution target 的最小 utility lower bound，再求 batch 内和，即得到 `L_b`。

这个 lower bound 是严格的但非常松；当前 `epsilon=1e-6` 会让 edge worst-case lower bound 极端负。它可用于数学完整性或 winner certification，不适合作为性能 heuristic。正式代码当前没有把 `L_b/U_b` 作为新的 algorithm state，也没有在本轮添加它们。

## 2. Partial-sum objective bounds

对某个 candidate，已完成前 `k` 个 batch 后，令：

```text
P_k = (1/S) * sum_{b < k} beta^h_b * R_b
C(C,Q) = exact configuration cost
```

Cache objective 在 `src/decision/cache_configuration.py:1122-1151`：

```text
J_cache = Q_hat - C_cache
```

其中 `C_cache` 由 `cache_update_cost()` 根据 previous/new cache、actual update、loading energy 一次性计算，与剩余 prediction batches 无关。

UAV objective 在 `src/decision/uav_configuration.py:393-431`：

```text
J_uav = Q_hat - C_movement - C_cache_constant
```

movement cost 同样只由 old/final position、trigger 和 system-model movement cost 决定，不依赖未计算 batch。

所以严格 bounds 为：

```text
LB_k = P_k + (1/S) * sum_{b >= k} beta^h_b * L_b - C
UB_k = P_k + (1/S) * sum_{b >= k} beta^h_b * N_b - C
```

即使只使用容易验证的上界，也可以安全地做 candidate rejection：

```text
if UB_k <= current_best + objective_tolerance:
    candidate cannot satisfy the existing strict-improvement rule
```

Cache local search 在 `src/decision/cache_configuration.py:1524` 和 `:1807` 只接受 `value > current_objective + tolerance`；UAV initialization/coordinate search 在 `src/decision/uav_configuration.py:821` 和 `:869` 使用同样的 strict improvement semantics。`UB_k <= current_best + tolerance` 不会改变 acceptance、tie 或 candidate order 的结果。

## 3. Why this is not a free implementation change

当前 `evaluate_surrogate_context()` 的 parallel branch (`src/decision/cache_configuration.py:918-995`) 会一次提交 100 个 futures，等待它们完成后再按原顺序 reduction。它没有 batch-level incumbent callback，也没有在 future group 中途取消剩余工作。

要实际使用上界而保持 exact semantics，至少需要把 prediction evaluation 改成 candidate-aware 的 ordered/chunked execution，并把当前 Cache/UAV `current-best` threshold 传入 evaluator。还必须保证：

- 已接受 candidate 仍取得 exact final objective；
- 不因 asynchronous completion 改变原始 floating-point reduction order；
- worker 取消/未提交的 batch 不改变 profile、cache lifetime 或 resource-feasibility error semantics；
- Cache/UAV outer coordinate search 的下一轮仍使用 exact `current_objective`。

本轮没有实施这些改变。

## 4. Real K=100 bound replay

下面的结果来自完整 K=100 reference 运行之后的 observation-only replay：使用真实 candidate evaluation 顺序、真实 partial batch utility、exact candidate cost 和当时的 Cache/UAV current-best threshold，逐 candidate 计算第一次满足 `UB_k <= current-best + tolerance` 的 `k`。这不是把结果 hindsight 当作运行时输入，也没有真正跳过 batch。

### Cache candidates

| origin | candidate evaluations | safely prunable before 100 | need full 100 | earliest stop | latest safe stop among prunable | whole Cache decision before 100 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 15 | 15 | 0 | 77 | 97 | **YES** |
| 120 | 269 | 210 | 59 | 33 | 99 | **NO** |
| 239 | 119 | 101 | 18 | 56 | 99 | **NO** |

“whole Cache decision” 的 `YES` 只表示本次 origin 0 的所有 candidate 都被 current-best upper bound 排除，因而 reference 的 no-update Cache decision 可被认证；它不是对任意 future state 的保证。origin 120/239 存在 strict-improvement candidate，必须完整得到 candidate objective 才能更新 current state/objective 并继续原有坐标搜索。

详细 safe-stop histogram（单位：每个 candidate 的 batch 数）：

| origin | histogram for prunable candidates | full-100 candidates |
|---:|---|---:|
| 0 | `77:1, 78:6, 79:2, 96:4, 97:2` | 0 |
| 120 | `33:3, 44:3, 48:2, 50:1, 55:2, 56:1, 57:3, 58:3, 60:3, 61:5, 62:2, 63:4, 64:2, 65:5, 66:5, 67:2, 68:2, 69:1, 72:1, 75:9, 78:3, 79:3, 83:3, 86:3, 88:6, 90:2, 91:13, 92:5, 93:10, 94:9, 95:3, 96:51, 97:27, 98:8, 99:5` | 59 |
| 239 | `56:4, 58:3, 59:3, 60:9, 61:3, 78:2, 85:2, 88:3, 89:13, 91:3, 92:3, 93:6, 94:2, 95:14, 97:5, 98:11, 99:15` | 18 |

这些是“每个 candidate evaluation 内的 batch 数”，不是把所有 candidates 加在一起后的 total batch count。

### UAV candidates

origin 0 的 `UAV_1` request-on path 产生 218 个 UAV candidate evaluations；严格 upper bound replay 中：

| origin | UAV candidate evaluations | safely prunable before 100 | need full 100 | decision certification |
|---:|---:|---:|---:|---|
| 0 | 218 | 0 | 218 | **NO before 100** |
| 120 | 0 | 0 | 0 | no UAV candidate search (`request=0`) |
| 239 | 0 | 0 | 0 | no UAV candidate search (`request=0`) |

origin 120/239 的 UAV trigger 为 0，因此它们没有可用于验证 request-on early certification 的 UAV candidate set；这不应被误报成 UAV early-stop success。

## 5. Status assessment

| item | status | evidence |
|---|---|---|
| Per-task utility has strict upper bound 1 | **CONFIRMED** | `system_model.task_utility()` validates nonnegative delay/energy and normalized weights. |
| Formal-path finite lower bound is derivable | **CONFIRMED** | finite scenario data, bounded UAV domain, positive channel/resource floors; bound is conservative and not pre-existing code. |
| Remaining objective upper bound is strict | **CONFIRMED** | positive discount/normalization, known event counts, exact batch-independent cost. |
| Candidate rejection can be exact | **CONFIRMED** | `UB_k <= current-best + tolerance` implies the existing strict-improvement test cannot pass. |
| All candidates can be early-stopped | **FAILED** | origin 120/239 Cache and origin 0 UAV have full-100 candidates. |
| Whole Cache decision can always be early-certified | **FAILED** | only origin 0 passed; origin 120/239 require full candidates. |
| Whole UAV decision can early-certify request-on path | **FAILED** | all 218 origin-0 UAV candidates require full 100 batches. |
| Production early-stop implementation | **NOT IMPLEMENTED** | no formal source changes were made. |

## Final values

```ini
PER_BATCH_UTILITY_BOUNDED = YES
STRICT_REMAINING_BOUND_DERIVABLE = YES
EXACT_EARLY_STOP_POSSIBLE = YES
CACHE_DECISION_CAN_BE_CERTIFIED_EARLY = YES
UAV_DECISION_CAN_BE_CERTIFIED_EARLY = NO
K100_REFERENCE_UNCHANGED = YES
PRODUCTION_EARLY_STOP_IMPLEMENTED = NO
```

`CACHE_DECISION_CAN_BE_CERTIFIED_EARLY = YES` 的精确含义是：存在真实 origin/state（本轮 origin 0）可以在不改变 K=100 decision 的前提下提前排除全部 Cache candidates；它不表示 origin 120/239 或任意未来 state 都能在 batch 100 之前结束。若把该 flag 解释为“所有 formal origins 都能提前结束”，则答案是 `NO`。

本轮未 commit、push、merge 或 rebase；临时 bound harness、逐 batch JSON 和 Python cache 在任务结束时删除。
