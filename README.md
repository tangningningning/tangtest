# adaptive_cache_mec

## 正式训练算法

当前唯一正式 PPO 训练算法是 Stage3 MAPPO，实现在 `src/ppo_simple/mappo_baseline.py` 和 `src/ppo_simple/cache_counterfactual_value.py`，正式 Runner 为 `scripts/training/run_mappo_stage3_cache_value.py`。`ppo-train` 是同一 Runner 的打包命令入口；`scripts/training/mainline_runner.py` 仅保留为旧命令兼容转发器，不是另一套算法。

Stage3 使用 `configs/proposed/system_config.toml` 和已审查的 `environment_data/proposed_joint_training/` Training Pool。Runner 按 Manifest 校验来源，每次更新从一个 Training Episode 读取连续 Slot 窗口。默认执行 3 次更新、每次 40 Slot；默认输出到新的 `runs/research_mappo_stage3_cache_value/` 子目录，不覆盖已有运行。

```powershell
conda activate adaptive_cache_mec
python scripts/training/run_mappo_stage3_cache_value.py --help
```

选择且只选择一种初始化方式：

```powershell
# 从随机参数开始
python scripts/training/run_mappo_stage3_cache_value.py --initialize-random

# 从 MAPPO v2 初始化迁移，或严格恢复目标合同相同的 Stage3 v6 Checkpoint
python scripts/training/run_mappo_stage3_cache_value.py --source-checkpoint <checkpoint.pt>

# 从旧 Stage3 v4/v5 仅迁移模型权重；优化器、进度及随机状态会重新初始化
python scripts/training/run_mappo_stage3_cache_value.py --initialize-from-stage3-checkpoint <checkpoint.pt>
```

`--source-checkpoint` 遇到 MAPPO v2 时执行显式 v2→Stage3 初始化迁移：恢复兼容策略权重和可迁移的基础优化器状态，Stage3 ΔQ 头与 Stage3 进度从新初始化开始；这不等于完整恢复。当前 Stage3 v6 Checkpoint 则必须通过严格模型、目标、优化器、进度和随机状态校验后才能恢复。Stage3 v4/v5 没有可验证的 Trigger 目标版本，不能严格续训；上述专用参数只恢复形状兼容的模型权重，清空优化器状态、进度计数并保留新 Trainer 的 RNG。`--resume` 不是当前 Runner 支持的参数。严格恢复会在新的唯一运行目录继续 Checkpoint 状态，不会写回旧运行目录。

Stage3 损失系数由 `configs/research/mappo_stage3_loss_coefficients_v1.json` 提供并按规范化 LF 内容校验 SHA-256 `c91bccc6bf91f76c407a18c9953b4fa2c2d346fcfef3337ad63ee49b23d8bdb4`。固定 Trigger/Candidate 排序系数为 `0.01/0.01`，ΔQ Huber 系数为 `1.0`，平局容差为 `0.01/0.01`，pairwise temperature 与 Huber δ 为 `1.0`，ΔQ 头学习率为 `0.001`。

## Trigger、Candidate 与评价合同

训练使用随机策略采样以执行 MAPPO PPO 更新，Macro joint log-prob 和原有 PPO ratio 合同保持不变。Stage3 Candidate 排序损失比较全部合法 UPDATE 候选的真实反事实 ΔQ。

当前主要冻结策略评价采用确定性“先选操作组、再选候选”解码：按 Trigger 组概率选择 KEEP/UPDATE（相等时选 KEEP）；若选 UPDATE，再在合法 UPDATE 集合中选择条件概率最大的 Candidate。Stage3 的 Trigger 标签合同 `deterministic_update_candidate_delta_q_v1` 与该解码一致：取行为策略条件 Candidate 分布的确定性 argmax 候选，并以该候选的反事实 ΔQ 监督 UPDATE-vs-KEEP。候选 argmax 使用稳定的首个最大值规则。该 ΔQ 是有限期 Macro SMDP 回报加冻结 Critic bootstrap 的条件估计，不是长期最优价值；推理时不读取真实未来任务，也不运行反事实 Oracle。

Stage3 v6 Checkpoint 同时记录并校验 `trigger_target_schema`。不能将无目标标识的 v4/v5 Checkpoint 当作当前目标的严格恢复输入。评价脚本会分别报告 Actor 的确定性 Trigger 决策准确率与 ΔQ value head 的价值估计指标，避免把价值头的正负判别误称为 Actor 动作准确率。

Stage3 冻结 Validation 适配器为 `scripts/evaluation/run_mappo_stage3_main_experiment.py`，可先检查参数：

```powershell
python scripts/evaluation/run_mappo_stage3_main_experiment.py --help
```

该适配器使用 Validation seeds `30001–30003`、相同外生场景和上述确定性解码。`ppo-eval` 指向旧 PROPOSED Checkpoint 评价器，不接受 Stage3 MAPPO Checkpoint；Stage3 Checkpoint 应使用 Stage3 冻结 Validation 适配器。

## 数据边界

Training、Validation 与 Test 使用各自独立的数据划分。Stage3 训练 Runner 只接受 Training Pool；主冻结 Validation 适配器只读取指定 Validation seeds。除非明确执行数据生成任务，否则不要重生成或改写现有 Pool、Manifest 或 `environment_data/`。
