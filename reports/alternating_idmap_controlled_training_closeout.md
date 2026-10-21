# Alternating Actor V1 受控训练 Closeout

核查日期：2026-10-20。只读检查了 `runs/alternating_idmap_controlled_2updates_20261020/`，没有运行训练或 Episode，也没有读取 Validation/Test。

## 结论

```ini
CONTROLLED_RUN_IDENTIFIED = runs/alternating_idmap_controlled_2updates_20261020
COMPLETED_EPISODES = 8 / 300
COMPLETED_POLICY_UPDATES = 2 / stop limit 2

MICRO_ACTOR_UPDATE_VERIFIED = YES (Update 1: enabled, 40 optimizer steps; initial-to-final actor parameters changed; KL=0.003210)
MACRO_ACTOR_UPDATE_VERIFIED = YES (Update 2: enabled, 4 accepted optimizer steps; initial-to-final actor parameters changed; joint KL=0.010530)
INACTIVE_ACTOR_FREEZE_VERIFIED = YES (Update 1 Macro=0 Actor steps; Update 2 Micro=0 Actor steps; current Trainer branches preserve inactive Actor parameters and Adam state)

MACRO_CRITIC_UPDATE_VERIFIED = YES (50 steps per group; 100 total)
MICRO_CRITIC_UPDATE_VERIFIED = YES (40 steps per group; 80 total)
LONG_VALUE_CRITIC_UPDATE_VERIFIED = YES (50 steps per group; 100 total)

NONFINITE_VALUES_FOUND = NO (all 8/2/2 JSONL records and latest Checkpoint floating tensors checked)
MANIFEST_SUMMARY_CHECKPOINT_CONSISTENT = YES (8 Episodes, 2 Updates, counters, schedule, config identity, algorithm and pool path agree)
CONTROLLED_STOP_STATUS_CORRECT = YES (Manifest/Summary: CONTROLLED_STOP_AFTER_UPDATES, limit=2, completed=2; Checkpoint counters end at 8/2)

N2EV_CHANGE = 0.041834 -> 0.161836 post-update (+0.120002 between groups)
LONGCORR_CHANGE = 0.216558 -> 0.102574 (-0.113983 between groups)
RETURN_CHANGE_INTERPRETATION = Group means 3849.79 -> 3890.25 (+40.46); two groups do not establish policy improvement or convergence

READY_FOR_300EP_TRAINING = YES
BLOCKING_ISSUES = NONE for starting a fresh Run; two updates are insufficient to assess convergence

TRAINING_EXECUTED = NO
FORMAL_CONFIG_MODIFIED = NO
CHECKPOINT_MODIFIED = NO
VALIDATION_TEST_READ = NO
COMMIT_PUSH = NO
```

## Actor 与 Critic 证据

`update_metrics.jsonl` 中 Update index 0 覆盖 Episode 1–4，`active_actor=MICRO`；Micro Actor 为 40 步，Macro Actor 为 0 步。Micro KL 为 0.003210、Clip fraction 为 0、Early Stop 为 false。Update index 1 覆盖 Episode 5–8，`active_actor=MACRO`；Macro Actor 有 4 个优化步，4 个候选步均被接受、无拒绝，joint KL 为 0.010530、Clip fraction 为 0.077516、Early Stop 为 false。Micro Actor 为 0 步。

初始与最终 Checkpoint 对比显示 Micro Actor 有 70,912 个参数标量变化（最大绝对差 0.007883），Macro Actor 有 135,811 个参数标量变化（最大绝对差 0.002000）。两组之间没有保存独立的 Actor 参数快照；冻结判断由每组 Actor 优化步数、Update enabled 标记及 Trainer 冻结分支共同支持。Micro 冻结时其共享 Adam 优化器仅对 Critic 梯度执行 step；Actor 梯度为 `None`，Actor 的 Adam 状态不会递增。Macro 冻结时 Actor PPO 循环为零步。

每组 Critic 步数均为 Macro 50、Micro 40、Long-Value 50。下表为各组更新记录的 EV、RMSE 与目标相关系数；Long-Value 的 EV 列为 post-update EV，相关系数列即日志 `LongCorr`。

| Critic | Update 1 EV / RMSE / Corr | Update 2 EV / RMSE / Corr |
|---|---:|---:|
| Macro | 0.041834 / 20.7946 / 0.207502 | 0.161836 / 22.1043 / 0.402993 |
| Micro | 0.503930 / 42.4871 / 0.712125 | 0.677318 / 46.5642 / 0.830437 |
| Long-Value | 0.003674 / 2129.4056 / 0.216558 | 0.007456 / 1713.8222 / 0.102574 |

这些指标来自两个不同的 4-Episode 更新组，Long-Value 的 LongCorr 在第二组下降；不据此判定任何 Critic 已收敛或未收敛。N2EV 的组内 pre/post 分别是 `-0.000355 -> 0.041834` 和 `0.081351 -> 0.161836`。

## Episode 指标口径

- Episode 1–4 平均 Return 为 **3849.7926**；Episode 5–8 为 **3890.2489**。Return 上升本身不能证明策略变好，两组数据不足以判断 MAPPO 收敛。
- `CacheChange` 来自 `actual_cache_update_rate`：在每个 Episode 的 Macro 决策帧上，统计是否至少一个缓存节点的最终缓存状态实际发生变化，再除以该 Episode 的 Macro 决策帧数。第一组为 **632/633=99.842%**，第二组 **615/616=99.838%**。这不是每个节点的 UPDATE 率，也不是每个时隙所有节点都更新。9 个节点累计实际变化分别为 **2811** 和 **2796** 次（共 **5607**）；节点级分母是每组 Macro 帧数乘 9。
- `UAVMove` 是每个 UAV 在 Micro 决策（每 Episode 300 个 Slot）中选择 MOVE 动作的比例；训练日志格式化时对两架 UAV 的比例取平均。第一组 MOVE 动作为 **49.50%**，第二组 **44.21%**。它统计 MOVE 动作标签，不等于实际位移距离或移动能耗。
- Episode 指标中的 Critic/N2EV/LongCorr 字段由所属 Policy Update 组指标填入；解释 Critic 变化应使用两个 Update 记录，而不是把同组 4 行当作 4 次独立 Critic 测量。

## Run、Pool 与 Checkpoint

`run_manifest.json` 与 `run_summary.json` 均记录目标 300 Episode、实际 8 Episode、2 次 Policy Update、受控停止上限 2，且没有标记 300 Episode 正式训练完成。`episode_metrics.jsonl` 有 8 条，`update_metrics.jsonl` 和 `critic_diagnostics.jsonl` 各有 2 条。Manifest、Summary、Update 记录及 `latest_checkpoint.pt` 中的更新计数和下一活跃 Actor（MICRO）一致。Checkpoint 中未单独保存受控停止文字标记；停止状态由 Manifest/Summary 记录，Checkpoint 的 8/2 计数与调度状态达到停止上限。

当前配置 raw SHA256 与 Run Manifest、Checkpoint 一致：`2798296ed073e020ab4249c68539e017fc48c01d89c07752973f7f2a2a34e003`；semantic SHA256 为 `2a4f74c8e55d6f1329c790d5265988c4e1bbf5600b1a069570b93cd3a63c01f5`。当前 Training Pool Manifest SHA256 与 Run Manifest 一致：`b6cae236b029f4d24c7488a91f0825247a20f475ec76b2d80737de44ae491bd3`。Checkpoint 记录相同 Pool 路径、seed 27005、算法 `ADAPTIVE_CACHE_UAV_JOINT_MAPPO`、架构版本 2、`ALTERNATING_ACTOR_UPDATES_V1` 与相同配置身份。Run Manifest 记录 Pool 为 300 个 Training Episode（seed 20001–20300）、`CURRENT_SNAPSHOT_VERIFIED`，并记录 `LEGACY_FORMAL_POOL_V1_VERSION_COMPATIBLE` 和按显式 ID 映射加载的兼容策略。Run 同时保留历史生成来源证明不可用的 provenance 状态，没有伪装为已获历史证明。当前 Runner 入口仍调用 Training Pool 合同门禁。

## 后续正式训练命令

新 Run 目录 `runs/alternating_idmap_formal_300ep_20261020` 在本次核查时不存在。Conda 激活在此机器上仍报 DLL/启动错误，使用已验证的环境解释器。下面的 PowerShell 命令仅供后续人工启动；不带受控停止上限、不带 `--resume`，从新初始化开始，不加载本次或其他 Run 的 Checkpoint：

```powershell
& 'C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe' scripts\training\mainline_runner.py `
  --config configs/proposed/alternating_actor_updates_v1.toml `
  --epochs 300 `
  --episode-pool environment_data/proposed_joint_training `
  --episode-seed-start 20001 `
  --seed 27005 `
  --run-dir runs/alternating_idmap_formal_300ep_20261020
```

该命令目标为完整 300 Episode；它不会使用受控停止参数，也不会 Resume Checkpoint。Pool 合同校验与旧排序/显式 ID 映射兼容门禁保持启用。
