# Stage1B cross-branch stability fix

本任务只修改诊断/稳定性控制代码，没有启动 episode、训练、续训或任何
diagnostic training run；没有读写已有正式 checkpoint、metrics 或 run artifact。
Stage2A 的 `PER_CACHE_HEAD` Cache ratio 保留。没有修改 reward、teacher、Q
target、loss composition、optimizer 类型、LR、Matching、KKT 或正式超参数。

## Required status

```text
CANDIDATE_KL_GATE_OLD = MEAN_ONLY
CANDIDATE_KL_GATE_NEW = MEAN_OR_P90
CANDIDATE_TARGET_KL = 0.10

ACTOR_OPTIMIZER_SPLIT = NO
ACTOR_LR_SPLIT = NO
GRADIENT_CLIPPING_SPLIT = YES

CACHE_UAV_PARAMETER_OVERLAP = NO

SYNTHETIC_CASE_1 = PASS
SYNTHETIC_CASE_2 = PASS
SYNTHETIC_CASE_3 = PASS
SYNTHETIC_CASE_4 = PASS
SYNTHETIC_CASE_5 = PASS
SYNTHETIC_CASE_6 = PASS

CACHE_RATIO_CHANGED = NO
CACHE_CREDIT_CHANGED = NO
UAV_TEACHER_CHANGED = NO
CONFIDENCE_SCALE_CHANGED = NO
LR_CHANGED = NO
OBJECTIVE_CHANGED = NO

TRAINING_EXECUTED = NO
COMMIT = NO
PUSH = NO
STAGE1B_READY_FOR_SHORT_TRAIN = YES
```

`STAGE1B_READY_FOR_SHORT_TRAIN = YES` 只表示源码静态检查、参数覆盖检查和
内存 synthetic cases 已通过；它不是训练结果，也不是对长程性能的保证。

## Candidate KL gate

`src/ppo_simple/trainer.py` 的 candidate gate transition 现在接受同一
update epoch 的 `kl_mean` 和 `kl_p90`，并在 epoch 边界执行：

```text
stop = (KL_MEAN >= target_kl) OR (KL_P90 >= target_kl)
target_kl = 0.10
```

`KL_MAX` 没有进入 hard gate。`candidate_kl_gate_trigger_source` 根据实际
越过阈值的统计量记录为 `MEAN`、`P90`、`BOTH` 或 `NONE`；每个 epoch 的
`kl_epoch_diagnostics` 也保留 statistic/source。这个 gate 仍然只冻结后续
candidate actor epochs，不增加 candidate KL penalty、PPO clip、rollback 或
minibatch early stop。

Candidate policy 的 Exact-Q teacher、合法 candidate mask、temperature、loss
和 confidence weighting 均未改动。

## Cache/UAV clipping split

`AdaptiveJointCacheUAVPPOActorCritic.actor_parameters()` 仍按以下六个模块返回
一个扁平 actor 参数 tuple：

1. `cache_encoder`
2. `uav_encoder`
3. `cache_trigger_head`
4. `cache_candidate_scorer`
5. `uav_operation_head`
6. `uav_candidate_scorer`

`PPOV2Trainer` 仍执行一次：

```python
torch.optim.Adam(actor_parameters, lr=self.effective_actor_learning_rate)
```

源码没有创建 per-module Adam parameter groups，因此 optimizer 仍是单一
parameter group，actor LR 仍由同一个 `effective_actor_learning_rate` 应用于
六个模块。遗留的 `actor_head` 虽由基类保留，但不在 proposed policy 的
`actor_parameters()` 中，因而不进入当前正式 actor optimizer。

本 Stage1B 只在这一次 Adam `step()` 前分开处理梯度：

```text
Cache group = cache_encoder + cache_trigger_head + cache_candidate_scorer
UAV group   = uav_encoder + uav_operation_head + uav_candidate_scorer
max_grad_norm = existing hyper.max_grad_norm (= 0.5 in the proposed config)
```

两个 group 使用互斥 parameter IDs；任何 overlap 或 actor 参数遗漏都会抛出
错误。每个 update epoch 的 `update_metrics.jsonl` 记录执行 minibatch 的平均值：

```text
CACHE_ACTOR_GRAD_NORM_PRECLIP
CACHE_ACTOR_GRAD_CLIP_COEFFICIENT
UAV_ACTOR_GRAD_NORM_PRECLIP
UAV_ACTOR_GRAD_CLIP_COEFFICIENT
```

如果某分支 inactive，其 group 的梯度会先被清为 `None`，该分支的 norm 为
零、coefficient 为 1；另一分支的 clipping coefficient 只由另一分支自己的
梯度决定。未改变 optimizer state、LR、loss reduction 或 actor step 次数。

## Synthetic validation

验证使用内存中的 transition、`torch.nn.Parameter` 和最小 proposed policy，
没有 rollout、环境、Matching、KKT 或 optimizer training step。

| case | 检查 | 结果 |
|---:|---|---|
| 1 | `mean=0.05`, `P90=0.12`，candidate stop 必须触发且 source=`P90` | PASS |
| 2 | `mean=0.11`, `P90=0.08`，candidate stop 必须触发且 source=`MEAN` | PASS |
| 3 | `mean=0.05`, `P90=0.08`，candidate stop 不触发且 source=`NONE` | PASS |
| 4 | Cache norm 大、UAV norm 小；Cache 被缩放，UAV gradient 保持不变 | PASS |
| 5 | UAV norm 大、Cache norm 小；UAV 被缩放，Cache gradient 保持不变 | PASS |
| 6 | 六个 proposed actor 模块的 clipping groups 无 overlap 且完整覆盖 actor | PASS |

最小 policy identity check 还得到：

```text
actor_parameter_count = 16  (hidden_sizes=(8,) synthetic policy)
legacy_actor_head_in_actor_parameters = False
cache_clipping_group_count = 8
uav_clipping_group_count = 8
```

这些数字仅用于内存结构检查，不代表正式模型的参数数量。

## Scope preservation

工作区当前 proposed 配置仍声明：

```text
policy_schema = adaptive_joint_cache_uav_ppo
cache_trigger_advantage_source = GLOBAL_GAE
cache_candidate_advantage_source = GLOBAL_GAE
cache_trigger_ratio_granularity = PER_CACHE_HEAD
cache_candidate_ratio_granularity = PER_CACHE_HEAD
uav_candidate_advantage_source = EXACT_Q_LISTWISE
uav_candidate_q_reference_scale = 0.02
max_grad_norm = 0.5
target_kl = 0.10
update_epochs = 4
num_minibatches = 10
```

本任务没有改动这些配置，也没有改变 Stage1 confidence scale `0.02`、UAV
Exact-Q teacher 或 Stage2A ratio。已有工作树中其他任务的修改保持原样，未被
本任务回滚或覆盖。

## Verification

已用 `adaptive_cache_mec` 环境的 Python 对以下源码做 `py_compile`：

```text
src/ppo_simple/trainer.py
src/ppo_simple/policy.py
```

随后清理了本任务产生的 `__pycache__` / `.pyc`，没有生成临时 test script，
也没有创建新的 run 目录。最终还执行 `git diff --check`；退出码为 0。Git
仅打印了现有工作树的 LF/CRLF 转换 warning，不是 diff-check 错误。
