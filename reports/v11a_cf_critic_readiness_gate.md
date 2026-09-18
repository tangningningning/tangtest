# V11A_COUNTERFACTUAL_CRITIC_READINESS_GATE

本 gate 使用冻结 pool realization `episode_20001` 的一个完整 300-slot rollout；没有执行正式 300 epoch 训练，也没有 final test。

## 静态 conditioning 检查

上一版 implementation gate 的 Q 只接收 `state/cache_context + target node + selected primitive feature`，所以原检查为不匹配：

```text
Q_CACHE_CONDITIONS_ON_FULL_CACHE_ACTION = NO
Q_CACHE_CONDITIONS_ON_OTHER_CACHE_ACTIONS = NO
Q_CACHE_CONDITIONS_ON_UAV_ACTIONS = NO
```

本次已改为 centralized `Q_cache(s, a_cache_all, a_uav_all)`。输入为 raw observation、target Cache node one-hot，以及所有 Cache 节点和所有 UAV 的生产动作编码；每个编码包含 operation one-hot、candidate one-hot 和实际 primitive feature。Counterfactual 只替换目标节点 slice，其他 Cache/UAV 动作保持不变。修改后的静态检查为 YES。

## Gate A：冻结策略上的 Q readiness

Q 使用当前 global PopArt-normalized GAE return target，学习率保持 `3e-4`，`max_grad_norm=0.5`，每个 rollout 严格执行 `4 epochs × 10 minibatches = 40` 个 Q optimizer steps。Actor/global critic/UAV 在此 Gate A 中冻结。

| 指标 | before | epoch 1 | epoch 2 | epoch 3 | epoch 4 / after |
|---|---:|---:|---:|---:|---:|
| Q loss | 0.51421249 | 0.46993470 | 0.40287837 | 0.33039084 | 0.26426974 |
| Q EV | -0.02752185 | 0.09093660 | 0.22296196 | 0.37524724 | 0.47201169 |

每个 Q epoch 的 loss、EV、raw/clipped grad norm 均 finite；Gate A 通过。CF signal 也 finite、非零并同时包含正负值。Gate A 有效 Cache node-time 数为 `2700 = 300 × 9`。

## Cache PPO one-update gate

随后在同一个 rollout 上执行一次普通 PPO update。该 update 的 Q 阶段仍先于 actor，并执行 40 个 Q steps；随后 Q 被固定并计算所有 detached `A_cf`，再进入 Cache/UAV PPO。Actor/global critic 各执行 40 steps。

Cache surrogate 明确采用 `mean_over_valid_cache_node_time`，而不是先对 9 个节点求和再对 transition 求均值。相对于 V10-D transition-mean 的归一化分母，记录的相对 scale 为 `300 / 2700 = 0.11111111`，因此不存在约 9 倍放大。

首个 Cache epoch 的 branch KL 为 `0.25595220`，仍高于固定 `target_kl=0.1`；branch-specific gate 因此将 Cache 后续 epoch 停用：`[true, false, false, false]`。未修改 target KL 或 actor learning rate，Cache PPO readiness 判定为 FAIL，按要求停止。

## Final fields

```text
STAGE = V11A_COUNTERFACTUAL_CRITIC_READINESS_GATE
TRAINING_EXECUTED = NO
FORMAL_TRAINING_EXECUTED = NO

Q_CACHE_CONDITIONS_ON_FULL_CACHE_ACTION = YES
Q_CACHE_CONDITIONS_ON_OTHER_CACHE_ACTIONS = YES
Q_CACHE_CONDITIONS_ON_UAV_ACTIONS = YES
COUNTERFACTUAL_CONDITIONING_MATCHES_DIAGNOSTIC = YES

Q_CACHE_OPTIMIZER_STEPS_PER_ROLLOUT = 40

Q_EV_BEFORE = -0.027521848678588867
Q_EV_AFTER = 0.4720116853713989
Q_LOSS_BEFORE = 0.5142124891281128
Q_LOSS_AFTER = 0.26426973938941956

Q_CACHE_READINESS = PASS

CF_ADV_POSITIVE_FRACTION = 0.4596296296296296
CF_ADV_NEGATIVE_FRACTION = 0.5403703703703704

CACHE_SURROGATE_REDUCTION = mean_over_valid_cache_node_time
CACHE_LOSS_SCALE_RELATIVE_TO_V10D = 0.1111111111111111

CACHE_KL_AFTER_UPDATE_EPOCH_1 = 0.25595219898968935
CACHE_ACTIVE_UPDATE_EPOCHS = [true, false, false, false]

CACHE_UPDATE_READINESS = FAIL

SYSTEM_MODEL_CHANGED = NO
GLOBAL_REWARD_CHANGED = NO
GLOBAL_GAE_FORMULA_CHANGED = NO
GLOBAL_CRITIC_CHANGED = NO
UAV_CODE_PATH_CHANGED = NO

FORMAL_300_EPOCH_RECOMMENDED = WAIT
REPORT = reports/v11a_cf_critic_readiness_gate.md and reports/v11a_cf_critic_readiness_gate.json
RAW_RESULTS = frozen pool episode_20001; detailed machine-readable results are in reports/v11a_cf_critic_readiness_gate.json
GIT_DIFF_CHECK = PASS: V11-A centralized Q/readiness diagnostics are present; unrelated pre-existing worktree changes were preserved
TEMP_FILES_REMOVED = YES
PUSHED = NO
```

