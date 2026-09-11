# Scheme B PPO first real update and first-episode gate

## 最终结论

当前正式 `policy_primitive / adaptive_primitive_v1` PPO 训练链通过首个 16-origin
rollout 与原生 PPO update gate。通过后，按要求继续并完成唯一一个完整 240-origin
training episode。全程使用 seed 4401、正常 Bernoulli trigger sampling、正常 conditional
Categorical sampling；没有进入 legacy exact/K=100 configuration optimizer。

本报告只验证训练链完整性与数值有效性，不评价 PPO 是否收敛。

## 执行配置

```ini
branch = main
HEAD = b7e90d4015bbeac682ce7b55a4a29d71a7e174d7
method = ppo_simple_v2
configuration_control_mode = policy_primitive
policy_schema = adaptive_primitive_v1
seed = 4401
rollout_steps = 16
update_epochs = 1
num_minibatches = 4
```

未修改 Scheme B 算法、action space、reward、Matching、KKT、配置控制结构或 PPO
hyperparameters；未生成 teacher data，未运行 BC、legacy comparison、SCA、GA、ranker、
reduced-K、多 seed 或第二个 episode。

## 16-origin rollout/update gate

连续收集 train origins 0–15，共 16 transitions。所有 transition 均通过 reward/G、
Cache/UAV state、capacity/boundary/separation、Matching/KKT 和 K=100 guard 检查。

### Probability 与 update 数值

| 指标 | 结果 |
|---|---:|
| old total log-prob range | [-33.2811546326, -9.8112926483] |
| pre-update new total log-prob range | [-33.2811508179, -9.8112926483] |
| pre-update PPO ratio range | [0.9999980927, 1.0000038147] |
| post-update new total log-prob range | [-33.3708343506, -9.8198757172] |
| post-update PPO ratio range | [0.8937650760, 1.0966795937] |
| policy loss | -0.3574009538 |
| value loss | 191.1937713623 |
| entropy | 17.6293449402 |
| approximate KL | 0.0014767647 |
| clip fraction | 0.0 |
| gradient norm before clipping | 27.6151371002 |
| parameter delta norm | 0.2157701012 |
| optimizer steps completed | 4 |
| gate wall-clock | 4.646080 s |

`trainer.update()` 是一次原生 PPO update；根据正式 `num_minibatches=4`，内部完成 4
个 optimizer steps。return、advantage、loss、gradient 与 update 后全部 parameter tensors
均为 finite。

Gate 行为统计：65 个 active Cache conditional heads，其中 6 次选择 `NO_CHANGE`；15 个
active UAV conditional heads，未选择 `STAY`。`ONLINE_K100_CONFIGURATION_CALLS=0`。

## 首个完整 240-origin episode

Gate 通过后保留其 policy/optimizer 状态，从正式初始 Cache/UAV environment state 开始
train origins 0–239。episode 使用 15 个连续 16-origin rollout chunks；每个 chunk 后执行
当前 trainer 原生 update，共 15 次 PPO updates、60 个 optimizer steps。

### Episode 核心结果

```ini
NUM_TRANSITIONS = 240
PPO_UPDATES = 15
OPTIMIZER_STEPS_IN_EPISODE = 60

EPISODE_RETURN = -454.81933102272524
MEAN_REWARD = -1.8950805459280218
MEAN_G = -1.8950805459280218

TOTAL_EPISODE_RUNTIME = 68.77320210001199 s
MEAN_ORIGIN_RUNTIME = 0.09152583791631817 s
MEDIAN_ORIGIN_RUNTIME = 0.07570440002018586 s
MAX_ORIGIN_RUNTIME = 0.5101473000249825 s
```

`TOTAL_EPISODE_RUNTIME` 包含 240-origin collection、15 次 PPO update、本任务要求的逐步
验证与 log-prob/ratio 检查。origin runtime 是每个 transition 内的 policy selection、direct
configuration 与 actual Matching/KKT 执行 wall-clock。

### Cache/UAV 行为

Rate 定义：trigger rate 的分母分别是 `240×9` 与 `240×2`；NO_CHANGE/STAY 和 actual
change/move rate 的分母是对应 active trigger 数。

| 指标 | count | rate |
|---|---:|---:|
| Cache triggers | 1096 | 0.5074074074 |
| Cache NO_CHANGE | 80 | 0.0729927007 |
| Cache actual changes | 1016 | 0.9270072993 |
| UAV triggers | 222 | 0.4625000000 |
| UAV STAY | 5 | 0.0225225225 |
| UAV actual movements | 217 | 0.9774774775 |

Candidate counts 只统计 active conditional heads：

```ini
MEAN_CACHE_CANDIDATE_COUNT = 15.440693430656934
MAX_CACHE_CANDIDATE_COUNT = 35
MEAN_UAV_CANDIDATE_COUNT = 80.84234234234235
MAX_UAV_CANDIDATE_COUNT = 81
```

### PPO diagnostics

以下单值为 episode 最后一次 update；range 覆盖本 episode 的全部 update/minibatch 诊断。

| 指标 | 最后一次 | episode range |
|---|---:|---:|
| policy loss | -0.1768463403 | [-1.2230464220, 0.7928319573] |
| value loss | 191.5685882568 | [33.5097885132, 5742.1601562500] |
| entropy | 24.0406684875 | [18.9473075867, 28.5931034088] |
| approximate KL | -0.0006146431 | [-0.0163950920, 0.0315985680] |
| clip fraction | 0.0 | [0.0, 0.0] |
| gradient norm before clipping | 27.5800838470 | [4.4425168037, 198.1054077148] |

这些数值只用于 finite/update-chain gate；本任务未据此调参或判断收敛。

## 环境与系统模型核对

- 240/240 transitions：`reward == actual Q - cache cost - movement cost == actual G`，逐值相等。
- 所有 rewards、states、log-probs、ratios、returns、advantages、losses、gradients、parameters finite。
- 每步 final Cache 均通过正式 binary shape 与 node capacity 校验。
- 每步 final UAV transition 均通过正式 boundary、maximum displacement、trigger=0 stay 与
  final separation 校验。
- 240 步 Matching 均正常返回 stop reason；KKT 共记录 52,661 requests。
- Gate 与完整 episode 的 `online_k100_configuration_calls` 均为 0。
- actual feasible-target → Matching → KKT → delay/energy/Q/G/reward 路径保持运行。

## 正式 artifact

- Checkpoint：`runs/ppo_primitive_first_episode_20260908/latest_checkpoint.pt`
- Gate metrics：`runs/ppo_primitive_first_episode_20260908/gate_metrics.json`
- Episode metrics：`runs/ppo_primitive_first_episode_20260908/episode_metrics.jsonl`

Checkpoint 标记 `completed_episodes=1`，使用 `adaptive_primitive_v1` checkpoint schema。

## Required status

```ini
ROLL_OUT_16_COMPLETED = YES
PPO_UPDATE_COMPLETED = YES
PPO_NUMERICS_FINITE = YES

ONLINE_K100_CONFIGURATION_CALLS = 0

FIRST_240_ORIGIN_EPISODE_COMPLETED = YES

EPISODE_RETURN = -454.81933102272524
TOTAL_EPISODE_RUNTIME = 68.77320210001199 s
MEAN_ORIGIN_RUNTIME = 0.09152583791631817 s

CACHE_TRIGGER_RATE = 0.5074074074
CACHE_NO_CHANGE_RATE = 0.0729927007
CACHE_ACTUAL_CHANGE_RATE = 0.9270072993

UAV_TRIGGER_RATE = 0.4625000000
UAV_STAY_RATE = 0.0225225225
UAV_ACTUAL_MOVE_RATE = 0.9774774775

REWARD_EQUALS_G_ALL_TRANSITIONS = YES
CACHE_FEASIBILITY_PASS = YES
UAV_FEASIBILITY_PASS = YES

READY_FOR_SHORT_PPO_LEARNING_RUN = YES
```

`READY_FOR_SHORT_PPO_LEARNING_RUN=YES` 仅表示单 episode 训练链、数值和环境 gate 已通过；
一个 episode 不能证明 PPO 收敛。
