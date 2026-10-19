# adaptive_cache_mec

## 当前正式算法：自适应 Cache–UAV 联合 MAPPO

当前正式训练器是 `src/ppo_simple/trainer.py` 中的 `PPOTrainer`，训练入口为 `ppo-train` / `python -m ppo_simple.trainer`，冻结评价入口为 `ppo-eval`。该 MAPPO 使用每节点共享参数的合法 Cache 动作分布、逐 Slot 序贯 UAV 动作、1–5 Slot 自适应 Holding、Macro SMDP GAE 与每 Slot Micro GAE，并通过原正式环境执行 Matching/KKT 和系统奖励。`fixed2` 是相同 Trainer 的显式兼容模式。

```powershell
conda activate adaptive_cache_mec
python -m ppo_simple.trainer --help
python -m ppo_simple.evaluator --help
```

训练需要现有 Training Pool，`--epochs` 必须满足配置中的每次更新 realization 数量。冻结评价需要相应分区的 Episode Pool 与当前架构 Checkpoint。改变车辆、Cache 节点、UAV 或服务数量会改变观察、候选或模型合同；训练和 Checkpoint 校验会拒绝不兼容张量。代码可为不同规模建立相应接口，不代表策略已经跨规模训练或泛化。

完整时间尺度、概率、Transition、Checkpoint 与兼容合同见[正式算法设计](docs/mappo_complete_solver.md)。Stage3 反事实标签代码保留为历史实验资料；它不是当前 PPO 更新的必要条件，当前正式 Trainer 不启用其辅助损失。

## 研究基线：预测驱动的全局 Cache 求解器

此非强化学习基线以成对 Training 标签拟合共享全局边际净效用网络，再在有限预算内逐次搜索合法的单节点 Cache 更新；被选中的配置由正式 Matching+KKT 环境执行，UAV 暂时保持 STAY。它不是当前正式 Cache–UAV 联合控制器。

其实现入口与边界见[基线合同](docs/predictive_global_cache_solver.md)：

- `src/decision/predictive_global_cache.py`：预测特征、Training 配对标签、价值网络与有界联合搜索。
- `scripts/training/train_predictive_global_cache_value.py`：Training-only 标签生成与监督拟合。
- `scripts/evaluation/run_predictive_global_cache_solver.py`：Training 窗口求解器入口。
- `configs/research/predictive_global_cache_value_v1.json`：固定标签与搜索预算。

历史结果仅用于研究对照，见 `reports/predictive_global_cache_first_training.md`。模型预测有限期条件化净效用差，q90 残差接受门槛只是经验启发式，不保证真实收益增加。

## 历史 Stage3 MAPPO v7 实验路径

Stage3 v7 保留为历史强化学习对照，实现在 `src/ppo_simple/mappo_baseline.py` 和 `src/ppo_simple/cache_counterfactual_value.py`，历史 Runner 为 `scripts/training/run_mappo_stage3_cache_value.py`。当前 `ppo-train` 和 `scripts/training/mainline_runner.py` 使用新的自适应 Cache–UAV 联合 MAPPO Trainer；要复现实验历史，只显式调用 Stage3 Runner。

本节以下 Stage3 命令、Checkpoint 和评价合同只用于复现历史对照，不属于当前正式联合 MAPPO 路径。

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

该适配器使用 Validation seeds `30001–30003`、相同外生场景和上述确定性解码。`ppo-eval` 用于当前自适应联合 MAPPO Checkpoint；Stage3 Checkpoint 必须使用 Stage3 冻结 Validation 适配器。

## 数据边界

Training、Validation 与 Test 使用各自独立的数据划分。Stage3 训练 Runner 只接受 Training Pool；主冻结 Validation 适配器只读取指定 Validation seeds。除非明确执行数据生成任务，否则不要重生成或改写现有 Pool、Manifest 或 `environment_data/`。
