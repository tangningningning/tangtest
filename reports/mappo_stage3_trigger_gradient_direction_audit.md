# Stage3 Macro Trigger 梯度方向核查

日期：2026-10-18

## 结论

固定 Training 标签上的前向与反向检查确认：正 best-ΔQ 标签使 UPDATE 相对 KEEP 的 log-odds 上升，负标签使其下降，平局标签不产生 Trigger 梯度。辅助梯度到达旧 Trigger 打分路径中的共享编码器、候选编码器和候选评分器；这些参数均在 Macro Actor Adam 参数组中。未发现 detach、mask、清零或参数组实现错误。

无法从现有固定标签恢复同一状态上的真实 PPO 梯度：标签文件不含行为动作、行为时 log-prob 或归一化 Macro advantage。因此，真实 PPO 与辅助梯度是否冲突标为 `UNRESOLVED`。这不是已发现的代码错误。

```ini
TRIGGER_TARGET_DIRECTION_VALID = YES
AUX_GRADIENT_REACHES_TRIGGER = YES
PPO_AUX_GRADIENT_CONFLICT = UNRESOLVED
IMPLEMENTATION_BUG_FOUND = NO
READY_FOR_MINIMAL_FIX = NO
```

## 范围与复现输入

- 使用 Training 固定标签 `runs/research_mappo_stage3_candidate_supervised_learning/candidate_supervised_20261018T115258Z_f4fd13/training_labels.npz`，SHA-256：`0dcf47b2c44fee99b33394dfb1aeeb802bc9f9731b7e18c6f2a8605001900e49`。它包含 16 条固定状态，来自 seed 20031–20038；每条是一个节点的 400 维 observation、候选特征与掩码、候选 ΔQ 及 best-ΔQ 标签。
- 使用标签来源 Update 16 Checkpoint `runs/research_mappo_stage3_main_experiment/training/stage3_20261018T095817Z_1b3d4652/checkpoints/checkpoint_update_0016.pt`，SHA-256：`ca819d60320b3c65a46c2eb8217ca6d9e18f5a4bda2799f1778988f3976e611f`。先严格恢复原策略，再按 Stage3 的兼容路径安装独立 Candidate 头；Trigger 权重沿用 Update 16。
- 使用冻结 Profile `configs/research/mappo_stage3_loss_coefficients_v1.json`，SHA-256：`c91bccc6bf91f76c407a18c9953b4fa2c2d346fcfef3337ad63ee49b23d8bdb4`。Trigger 系数为 0.01，平局容差为 0.01。
- 只运行模型前向和 `torch.autograd.grad` 梯度读取；没有调用 `Trainer.update()`、没有调用优化器 `step()`，也没有运行 Episode、Matching+KKT 或 Test。

## 标签目标与 log-odds

`src/ppo_simple/mappo_baseline.py:1442-1674` 从每个有效反事实标签验证候选集合、有限值及 `best_delta_q == max(legal ΔQ)`，并将有合法 UPDATE 的有效标签标为 Trigger 样本。

`src/ppo_simple/mappo_baseline.py:2013-2027` 对合法 UPDATE 的 Trigger logits 做 log-sum-exp，并减去 KEEP logit：

`z = logsumexp(logit_UPDATE_legal) - logit_KEEP`。

这是合法 UPDATE 操作组相对于 KEEP 的 log-odds。`src/ppo_simple/policy.py:525-627` 生成 Trigger logits 与独立条件 Candidate logits；`policy.py:634-750` 的采样分布也将合法 UPDATE logits 合并为 UPDATE 操作组。因此监督所用的比较对象与策略的操作组一致。

`src/ppo_simple/cache_counterfactual_value.py:704-725` 的损失为：

`L = mean(softplus(-sign(best_ΔQ) × z))`

只纳入有限且 `abs(best_ΔQ) > tie_tolerance` 的标签。于是梯度下降方向为：

| best-ΔQ | 监督目标 | `dL/dz` 符号 | 梯度下降对 `z` 的作用 |
|---|---|---:|---|
| `> +0.01` | UPDATE 优于 KEEP | 负 | 增大 UPDATE log-odds |
| `< -0.01` | 最佳合法 UPDATE 仍劣于 KEEP | 正 | 减小 UPDATE log-odds |
| `[-0.01, +0.01]` | 视为平局 | 不参与损失 | 不改变 Trigger |

这确认了方向，但该二元排序目标不保留 ΔQ 幅度：刚超过容差的小正值和较大的正值方向相同。它提供的是操作组方向监督，不是概率校准或增益幅度回归。

## 固定状态梯度结果

16 条标签中有 11 个正值、1 个负值、4 个平局；共有 12 条进入 Trigger 损失。复算的未加权损失为 `0.6740600`，乘 Profile 系数后的辅助项为 `0.00674060`。

逐条 `dL/dz` 核验结果：11/11 正标签均为负梯度，1/1 负标签为正梯度，4/4 平局梯度严格为零。梯度下降的作用与上表一致，掩码中每条均有合法 KEEP 和至少一个合法 UPDATE。

辅助损失对参数的未加权 L2 梯度范数：

| Trigger 路径参数组 | 梯度范数 |
|---|---:|
| `macro_actor_encoder` | `0.00589985` |
| `macro_cache_candidate_encoder` | `0.16941553` |
| `macro_cache_scorer` | `0.21390218` |
| 三组联合 | `0.27292961` |
| 乘 0.01 系数后联合 | `0.00272930` |

共 10 个 Trigger 参数张量取得非零梯度。独立 `macro_conditional_candidate_encoder` / `macro_conditional_cache_scorer` 没有 Trigger-loss 梯度，这是预期隔离；`macro_termination_scorer`、Critic 也没有梯度。`detach_candidate_context=True` 只在 `policy.py:611-627` 对独立条件 Candidate 分支的隐藏上下文执行 detach；Trigger logits 在其前由未 detach 的旧路径生成。

## 优化器及 PPO 梯度关系

`src/ppo_simple/mappo_baseline.py:223-265` 将 `macro_actor_encoder`、旧 `macro_cache_candidate_encoder`、旧 `macro_cache_scorer` 和 Stage3 条件 Candidate 参数放入 Macro Actor 参数组。Stage3 安装头并迁移状态的路径在 `mappo_baseline.py:291-338`。本次按同一命名合同核对：全部 10 个获得 Trigger 梯度的参数均属于 Macro Actor Adam 参数组；没有梯度从该组外传入，也没有清零后遗漏。`zero_grad(set_to_none=True)` 在同轮 PPO 与 Trigger 联合反传之前执行（`mappo_baseline.py:1970`），`macro_actor_total_loss = macro_actor_loss + 0.01 × trigger_loss` 后统一 backward、裁剪并执行 Macro Actor step（`2030-2045`）。后续 Candidate-only pass 会再次清零，并因分离头和 detach 隔离 Trigger 参数（`2167-2223`）。

真实 PPO 与辅助梯度的逐状态冲突需要同一条 rollout 的动作、旧 log-prob 和归一化 advantage。固定标签 NPZ 只含 observation、node/candidate 特征与掩码、ΔQ 和 best-ΔQ；没有这些 PPO transition 字段，不能从中恢复真实 PPO 梯度。

在 `src/ppo_simple/ppo_math.py:20-25` 的未裁剪 PPO 区域、`ratio=1` 时，令 `A` 为该节点的 Macro advantage、`p` 为 UPDATE 概率：

- 采到 UPDATE：`dL_PPO/dz = -A(1-p)`。
- 采到 KEEP：`dL_PPO/dz = A p`。

因此 PPO 与 Trigger 辅助项可同向也可冲突，取决于实际动作及其 advantage 与 best-ΔQ 标签的关系；只凭标签无法判定冲突。

已有 formal smoke 的汇总记录可作有限旁证：其单个 Macro transition 的归一化 advantage 标准差为 0，`macro_actor_loss=0`，而 `stage3_macro_actor_total_loss=0.00652832`（`runs/research_mappo_stage3_trigger_candidate_decoupling/formal_smoke/stage3_20261018T122316Z_5ea49321/training_metrics.jsonl`）。该次 PPO surrogate 没有提供非零策略梯度，不能代替固定标签状态上的 PPO-vs-aux 对照，也不能说明一般训练中不存在冲突。

## 判断与后续

实现链条的方向、梯度可达性和优化器归属均通过；本轮没有发现应当修复的实现错误。若后续需要判定 PPO 与 Trigger 辅助目标的实际冲突，最小补充是保存同一 Macro rollout transition 的行为动作、old log-prob、归一化 advantage，并在该固定 batch 上分别计算两组梯度及余弦相似度，不调用 optimizer step。现阶段不建议改动监督映射或模型代码。

```ini
SOURCE_CODE_CHANGED = NO
CONFIG_CHANGED = NO
DATA_OR_CHECKPOINT_CHANGED = NO
TRAINER_UPDATE_CALLED = NO
OPTIMIZER_STEP_CALLED = NO
TEST_ACCESSED = NO
```
