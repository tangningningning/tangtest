# PPO train/test 隔离实现与验证报告

## 1. 依据与范围

本次实现以用户指定的系统模型文件 `C:\Users\tangning\Desktop\12.pdf` 为唯一系统模型依据。没有用 `11.pdf` 替代，也没有修改系统模型、任务模型、Cloud 合约、缓存/UAV 优化、奖励定义、154 维状态或 11 维动作。

以下约束保持不变：

- `cross_slot_task_model = independent_per_slot`，不引入跨 slot queue。
- `psi_c = 11`、Cloud `F_c = 5e9`、`R_c = 200e6`、`resource_sharing = false`。
- Cloud 只在没有可执行的 edge cached service 时作为 fallback。
- prediction horizon `H = 5` 保持不变，观测 cutoff 为 `origin - 1`。
- PPO 只使用 train/test 两个 origin 区间，不增加 PPO validation 区间。
- popularity GRU 的 `24/6/6` 协议没有修改。

实现使用项目指定环境：`C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`，PyTorch `2.11.0+cu130`。

## 2. 数据协议

新增 `system_config.toml` 的 `[algorithm.ppo.data_protocol]`，边界只从配置读取：

```text
train_origin_start = 0
train_origin_end   = 239
test_origin_start  = 240
test_origin_end    = 299
```

协议公式为：

```text
Train = { o | train_origin_start <= o <= train_origin_end }
Test  = { o | test_origin_start  <= o <= test_origin_end  }
```

配置校验和正式 prediction scenario 校验共同保证：

1. train 从正式 origin 0 开始；
2. train/test 连续、无重叠、无空洞；
3. test 终点覆盖当前正式 scenario 的最后一个 origin；
4. train 样本数可被当前 `num_minibatches = 4` 整除。

当前结果是 train 240 个 origin、test 60 个 origin，train minibatch size 为 `240 / 4 = 60`。代码中没有复制 `239/240/299` 作为边界常量；这些边界由 TOML 和 scenario 长度产生。

## 3. 训练路径

修改了 `src/ppo/ppo_training.py` 的 `FormalTrainingProtocol`、正式训练计划和 aligned rollout 构造：

- 每个正式 PPO episode 只构造 train origin `0..239`。
- 最后一个训练 origin 是 `239`，其 `done=True`，不会把 test origin 当作训练 bootstrap。
- `run_formal_ppo_training(..., execute=False)` 的 dry-run 计划已经报告 train/test 边界；`--protocol-check` 不执行环境、反向传播或 optimizer。
- 正式 train/test aligned steps 都关闭区间外的额外 bootstrap step，因此不会读取或执行边界外的 actual task/reward；现有 artifact 和 prediction boundary slot 没有被删除。
- `H=5` 的因果关系保持为 `observation_cutoff = origin - 1`，test 的 actual task/reward 不会进入 train rollout。

## 4. 独立 held-out test

旧的 in-trace deterministic evaluation 语义已重命名为 held-out test，并保留兼容别名以避免历史调用直接断裂。新入口为：

```text
python -m ppo.ppo_training --held-out-test --checkpoint PATH
python -m ppo.ppo_training --held-out-test --checkpoint PATH --test-report REPORT.json
```

held-out test 的行为：

- 只执行 test origin `240..299`；
- test 开始时将 cache `C` 和 UAV 位置 `Q` reset 到与正式训练相同的共同初始状态，然后只在 test 区间内顺序传播；
- 使用 `torch.no_grad()`；不创建 optimizer、不调用 optimizer、不训练、不修改网络；
- 输出 system utility、actual Q、cache cost、movement cost、cache trigger、UAV trigger、各实际执行位置比例、Cloud 比例、reward identity error 和可行性。

评估入口在执行前检查 checkpoint compatibility，包含 train/test origin range、terminal 信息和现有 action/observation/architecture 元数据。旧的 8-action checkpoint 被拒绝；将当前 compatibility metadata 的 `test_origin_end` 改为 298 也会被拒绝。

## 5. 变更文件

- `system_config.toml`：新增最小 PPO train/test data protocol。
- `src/common/project_config.py`：严格检查协议字段、连续性、覆盖范围和 minibatch 可整除性。
- `src/ppo/ppo_rollout.py`：增加可选的 bootstrap-step 开关，保留旧调用默认语义；正式 train/test 使用无额外 bootstrap 的区间闭合输入。
- `src/ppo/ppo_training.py`：实现协议对象、正式训练边界、held-out test 执行/报告、checkpoint compatibility、CLI protocol check 和 test-only 入口。
- `data/generated/generation_metadata.json`：仅同步当前 TOML 的 config hash 和既有容量浮点表示，使 artifact check-only 与当前配置一致；没有重新生成或覆盖 artifact 数据。
- `reports/ppo_train_test_split_validation.md`：本报告。

用户已有的历史 reports、runs 和 checkpoints 未清理、未覆盖。

## 6. 验证结果

已通过：

- `compileall -q src`；
- `git diff --check`；
- `python -m ppo.ppo_training --protocol-check`：`formal_validation=true`，environment/backward/optimizer executed 均为 0；
- formal plan dry-run：`train=0..239`、`test=240..299`、240 steps/episode、minibatch 60；
- boundary/alignment check：train aligned `240, 0..239`，test aligned `60, 240..299`，terminal 分别为 239 和 299；
- held-out test no-op policy check：60 个 test steps，location ratios 总和为 1，reward identity error 为 0；
- held-out test 的实际 policy wrapper 检查：网络 state_dict 前后完全不变，执行区间只有 test；
- 最小 hierarchical PPO loss/gradient check：loss 和 gradient finite，单次 Adam update 的参数变化为正；
- old checkpoint compatibility rejection 和 range mismatch rejection；
- `scripts/regenerate_artifacts.py --check-only`：11 项检查全部通过，`writes_performed=0`；
- bounded sanity 的首次运行和单次运行均在有限时间窗口内未返回，已停止挂起进程，没有把该结果冒充为通过；最小可复现梯度更新和独立 held-out test 已分别验证。

### 6.1 dry-run 展示语义修复

收尾复核发现旧版 `_dry_run_summary` 会因 `cf_batch_episodes=4` 误报 sparse frozen-policy batch，即使实际配置是 hierarchical Gate。已只修复展示层：

- `method_id` 现在报告 `proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift`；
- `architecture_id` 区分顶层 `hierarchical_reconfiguration_gate_v1` 与 WHAT 子策略的 `v3_full_shared_actor_frozen_policy_batch_v1`；
- `actor_optimizer_mode` 报告独立 Gate/WHAT optimizer，`credit_phase` 报告 `hierarchical_gate_direct_cf`；
- 明确报告 `A_cf=(a-p_old)*Delta_G`、`A_gate=(g-p_old_gate)*Delta_gate`、无 learned critic/value loss，以及 Gate-first-then-WHAT 更新顺序；
- 增加 Gate optimizer 实例数、WHAT/Gate counterfactual branch steps 和总 optimizer/backward 计划，避免把 Actor-side count 当成完整总数。

训练分派、loss、optimizer step 顺序和系统模型均未改变。最小 reporting branch check 已覆盖 hierarchical、sparse frozen、sparse RefDrift 和 legacy 四种分支。

本次没有执行正式长训练，没有生成或覆盖正式 checkpoint、training history 或 long-term result。

## 7. 限制

当前环境下完整 bounded rollout 的计算时间超过本次短时 sanity 窗口，因此尚未声称完整 PPO episode/update 的端到端耗时测试通过。最小梯度测试验证的是网络、loss、反向传播和参数更新链路，不替代正式 Direct-CF 训练。正式训练仍需由用户显式提供 episode 数和 run directory 后执行。
