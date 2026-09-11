# PPO primitive Cache/UAV direct control — Scheme B

## 结论

Scheme B 已实现并通过允许范围内的两个真实 origin smoke。正式 PPO 默认模式现为
`policy_primitive`，policy schema 为 `adaptive_primitive_v1`。11-bit Bernoulli
trigger 保留；每个 active Cache/UAV trigger 只采样并立即应用一个 conditional
primitive（包括 `NO_CHANGE` / `STAY`）。随后仍走原有 actual feasible-target、
Transfer-Exchange Matching、KKT、delay、energy、Q、G 与 reward 路径。

两个 smoke origin 的 `online_k100_configuration_calls` 都为 0；reward 与 actual
`G(t)=Q(t)-C_ca(t)-C_q(t)` 逐值完全相等。没有运行完整 episode、长 rollout、teacher、
BC 或 PPO convergence test。

## Git 基线与范围

- Branch：`main`
- HEAD：`b7e90d4015bbeac682ce7b55a4a29d71a7e174d7`
- 开始时 working tree 已包含用户此前的 Matching/KKT/runtime 优化、正式 metrics 和多份报告；均保留。
- 本任务未修改 `src/decision/task_allocation.py`，未修改 Matching/KKT、delay、energy、Q、G、reward 公式。
- 未 commit、push、merge 或 rebase。

## 两套配置路径

### `legacy_exact`

原路径完整保留：11-bit trigger → K=100 iterative Cache/UAV configuration optimizer →
actual Matching/KKT → G。`initialize_cache_for_triggers()`、`optimize_cache()`、
`optimize_uav_configuration()`、prediction surrogate 与 joint outer refinement 均仍存在。
本任务只让 legacy 接受动作时复用抽取出的 application helper，没有改变其候选生成、
评分、顺序、接受条件或迭代次数。

### `policy_primitive`

路径为：11-bit trigger → 顺序 conditional primitive sampling/application → actual
Matching/KKT → G。该路径不建立 prediction surrogate context，不创建 prediction worker
pool，不调用 K=100 configuration objective。

当前正式配置：

```toml
[algorithm.ppo]
method = "ppo_simple_v2"
configuration_control_mode = "policy_primitive"
```

## Primitive 空间与 application API

Cache 使用 production `generate_node_neighbors()`；新接口
`enumerate_cache_primitive_actions()` 在其结果前加入 `NO_CHANGE`，并由 before/after
binary vector 严格派生/核对 `ADD`、`DELETE`、`SWAP`。枚举只做 binary、service-size、
capacity 检查。`apply_cache_candidate()` 是 direct 与 legacy 共用的正式状态写入接口。

UAV 使用 production `coordinate_candidates(current,current,search_config)` 产生当前点的
一步候选，去重后使用原 `_candidate_is_feasible()` 过滤 boundary、single-slot maximum
displacement 与 UAV separation；首项为 `STAY`。`apply_uav_candidate()` 复用同一正式约束
检查并写入状态，完整 transition 另由 `validate_uav_configuration_transition()` 核验。

多个 Cache node 按 `cache_nodes["node_ids"]` 顺序处理；多个 UAV 按 configured UAV ID
顺序处理。每次枚举读取前一 head 已应用后的 intermediate Cache/UAV state。

## Policy、features 与 PPO probability

`AdaptivePrimitiveActorCritic` 使用一个共享两层 MLP state encoder，并连接：

- 原 11 个 Bernoulli trigger logits；
- 小型 Cache candidate MLP scorer；
- 小型 UAV candidate MLP scorer；
- value head。

Prediction features 继续通过 state observation 使用：

- `predicted_vehicle_positions_h0`（全车辆归一化位置）；
- `predicted_service_popularity_h0 / rho_hat`。

Cache candidate 的廉价字段包括 node index/type、operation type、before/after/delta cache
vector、before/after occupancy、remaining capacity、added/removed service size、added/removed
rho、现有 cache update cost/loading energy。UAV candidate 字段包括 UAV identity、STAY/MOVE、
current/candidate position、dx/dy、distance、现有 movement energy/cost，以及预测车辆位置的
mean/std 摘要。candidate feature 构造不调用 Matching、KKT 或 future exact objective。

Rollout 使用固定上界 padding + boolean mask 保存：

- 分开的 Cache/UAV head 类型及固定 ordered head identity；
- 每个 active head 的 intermediate observation/context；
- candidate features 与 mask；
- selected index；
- per-head old log-prob；
- transition total old log-prob。

PPO update 直接用这些数值 tensor 重算
`trigger log-prob + active Cache categorical log-prob + active UAV categorical log-prob`；
trigger=0 的 head 乘 active mask 后不贡献 conditional log-prob/entropy。内存级重算检查：

```ini
JOINT_LOGPROB_MAX_ABSOLUTE_DIFF = 0.0
CONDITIONAL_BACKWARD_GRADIENT_FINITE = YES
PPO_UPDATE_ENV_REENUMERATION_REQUIRED = NO
```

Checkpoint 行为：legacy schema/version 2 保留；`adaptive_primitive_v1` 使用 version 3，并
同时校验 `policy_schema` 与 `configuration_control_mode`。旧 11-bit-only checkpoint 不会
静默加载到 conditional heads。

## 真实 smoke 结果

环境：Conda `adaptive_cache_mec`，seed `4401`，正常 Bernoulli trigger sampling，正常
conditional Categorical sampling。origin 1 使用 origin 0 实际产生的 Cache/UAV state。

| 指标 | origin 0 | origin 1 |
|---|---:|---:|
| raw trigger bits | `01000110011` | `01000110011` |
| Cache triggers | 3 | 3 |
| UAV triggers | 2 | 2 |
| Cache conditional candidate counts | 9, 20, 14 | 13, 23, 26 |
| UAV conditional candidate counts | 81, 81 | 81, 81 |
| Cache NO_CHANGE selected | 0 | 0 |
| actual Cache changes | 3 | 3 |
| UAV STAY selected | 0 | 0 |
| actual UAV movements | 2 | 2 |
| policy selection seconds | 0.093264 | 0.040279 |
| direct configuration seconds | 0.000249 | 0.000346 |
| actual Matching/KKT seconds | 0.080782 | 0.116908 |
| total origin seconds | 0.175543 | 0.158417 |
| K=100 configuration calls | 0 | 0 |
| reward / G | 8.000806766882944 | 6.839646269250879 |
| reward − G | 0.0 | 0.0 |

Aggregate smoke：

```ini
SMOKE_ORIGINS_COMPLETED = 2
MEAN_SMOKE_ORIGIN_RUNTIME = 0.166980 s
CACHE_TRIGGER_COUNT = 6
UAV_TRIGGER_COUNT = 4
CACHE_NO_CHANGE_COUNT = 0
CACHE_ACTUAL_CHANGE_COUNT = 6
UAV_STAY_COUNT = 0
UAV_ACTUAL_MOVE_COUNT = 4
MEAN_ACTIVE_CACHE_CANDIDATE_COUNT = 17.5
MAX_CACHE_CANDIDATE_COUNT = 26
MEAN_ACTIVE_UAV_CANDIDATE_COUNT = 81.0
MAX_UAV_CANDIDATE_COUNT = 81
POLICY_PRIMITIVE_ONLINE_K100_CALLS = 0
```

两步 actual Matching 均以 `converged_by_no_improvement` 正常结束。origin 0/1 分别执行
194/313 次 full candidate evaluator、375/607 次 KKT request，说明 direct control 后并未
绕过 actual Matching/KKT。

## 有限性与未运行项

- reward、G、next Cache state、next UAV positions：finite。
- reward 与 G：两步完全相等。
- state propagation：origin 0 的三个 Cache primitive 与两个 UAV movement 进入 origin 1
  初始状态，origin 1 再次完成合法 conditional selection 与真实执行。
- PPO update smoke：两个 transitions 少于正式 `num_minibatches=4` 的 rollout 要求；按任务
  限制不额外运行 origin，因此未执行 update。联合 log-prob/gradient 路径已做无环境检查。

## 最终状态

```ini
SCHEME_B_IMPLEMENTED = YES

LEGACY_EXACT_PRESERVED = YES
LEGACY_11BIT_TRIGGER_PRESERVED = YES

NEW_POLICY_SCHEMA = adaptive_primitive_v1

CACHE_POLICY_SEMANTICS = TRIGGER -> ZERO_OR_ONE_PRIMITIVE_OPERATION
UAV_POLICY_SEMANTICS = TRIGGER -> ZERO_OR_ONE_PRIMITIVE_MOVEMENT

CACHE_NO_CHANGE_SUPPORTED = YES
UAV_STAY_SUPPORTED = YES

CACHE_PRIMITIVE_ENUMERATION_SCORE_FREE = YES
UAV_PRIMITIVE_ENUMERATION_SCORE_FREE = YES

CACHE_APPLICATION_API_EXTRACTED = YES
UAV_APPLICATION_API_EXTRACTED = YES

PREDICTION_FEATURES_USED = [predicted_vehicle_positions_h0, predicted_service_popularity_h0/rho_hat]

PPO_CONDITIONAL_LOGPROB_IMPLEMENTED = YES
PPO_UPDATE_ENV_REENUMERATION_REQUIRED = NO

POLICY_PRIMITIVE_ONLINE_K100_CALLS = 0

MATCHING_CHANGED = NO
KKT_CHANGED = NO
DELAY_CHANGED = NO
ENERGY_CHANGED = NO
Q_CHANGED = NO
G_CHANGED = NO
REWARD_CHANGED = NO

CACHE_OFFLOADING_COUPLING_PRESERVED = YES
UAV_CHANNEL_COUPLING_PRESERVED = YES

SMOKE_ORIGINS_COMPLETED = 2
MEAN_SMOKE_ORIGIN_RUNTIME = 0.166980 s

REWARD_FINITE = YES
G_FINITE = YES
REWARD_EQUALS_G = YES

PPO_UPDATE_SMOKE = NOT_RUN_DUE_TO_ROLLOUT_REQUIREMENT

READY_FOR_TEACHER_WARMSTART = YES
```

这里的 `MATCHING/KKT/..._CHANGED = NO` 表示正式实现与公式未修改；Scheme B 的
Cache/UAV 决策本来就允许与 `legacy_exact` 不同，本报告不声称二者 action/output 等价，
也不评价 PPO 收敛。
