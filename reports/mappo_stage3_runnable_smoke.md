# Stage3 主项目整合与有界运行验证

日期：2026-10-18  
范围：Stage3 作为唯一正式 PPO 训练路径的主项目整合；Training seed `20007` 两 Slot 反事实与一次更新验证。

## 整合结果

- `ppo-train` 现在解析到 `ppo_simple.stage3_cli:main`；`scripts/training/mainline_runner.py` 也直接转发到 Stage3 Runner，不再提供旧/新算法选择开关。
- Stage3 Runner 使用主项目内的训练池来源合同、冻结系数 Profile、反事实标签生成和 MAPPO 更新实现。
- 旧 `PPOTrainer.update_from_episodes()`、`run_training()` 和旧命令入口已显式退役并报错。`ppo-eval`、历史 Checkpoint 和诊断读取器保留；它们不提供旧 PPO 训练路径。
- 删除了四个仅调用旧训练路径的 Macro Actor 学习率扫描包装脚本。未清理其他旧 PPO 诊断/评价代码，因为它们仍服务于历史结果读取。
- 损失 Profile 位于 `configs/research/mappo_stage3_loss_coefficients_v1.json`，SHA-256 为 `c91bccc6bf91f76c407a18c9953b4fa2c2d346fcfef3337ad63ee49b23d8bdb4`。固定值：Trigger/Candidate 排序系数 `0.01/0.01`、ΔQ Huber 系数 `1.0`、平局容差 `0.01/0.01`、pairwise temperature `1.0`、Huber δ `1.0`、价值头学习率 `0.001`。Runner 校验该 Profile 哈希并把配置写入运行元数据及 Checkpoint。

## 修改文件

新增：

- `src/ppo_simple/cache_counterfactual_value.py`：反事实快照结构化序列化及价值标签实现。
- `src/ppo_simple/mappo_baseline.py`：Stage3 MAPPO Trainer、价值头及更新接线。
- `src/ppo_simple/ppo_math.py`：共享 PPO 数学函数。
- `src/ppo_simple/stage3_cli.py`：项目内 Stage3 CLI。
- `scripts/training/mappo_stage3_pool_contract.py`：Training Pool 来源与 episode 输入合同校验。
- `scripts/training/run_mappo_stage3_cache_value.py`：Stage3 唯一正式训练 Runner。
- `configs/research/mappo_stage3_loss_coefficients_v1.json`：冻结 Profile。

修改：

- `pyproject.toml`：`ppo-train` 指向 Stage3 CLI。
- `scripts/training/mainline_runner.py`：Stage3-only 转发入口。
- `src/ppo_simple/trainer.py`：显式退役旧 PPO 更新与训练入口，保留历史评价/诊断读取功能。

删除：

- `scripts/training/run_proposed_macro_actor_lr_1e_minus_3.py`
- `scripts/training/run_proposed_macro_actor_lr_1e_minus_4.py`
- `scripts/training/run_proposed_macro_actor_lr_5e_minus_3.py`
- `scripts/training/run_proposed_macro_actor_lr_5e_minus_4.py`

## 反事实序列化修复与训练验证

Stage3 反事实 trace 原先在 pickle `AlignedRolloutStep` 时因重复模块/类身份而失败。修复后快照哈希基于结构化字段值编码；用于 Replay 的原始对齐步骤仍保留。对非有限 ΔQ 标签严格拒绝，不以默认零标签替代。

使用 Training Pool 中 seed `20007` 的连续 Slot `0–1`，从可兼容的 Stage1D Update 13 MAPPO v2 Checkpoint 初始化，并执行 **1 次 Stage3 更新**。本次不是完整 Episode；没有访问 Validation 或 Test。来源池 Manifest 未改动，运行记录同时保存了来源池哈希和当前配置哈希。经批准的来源兼容项仅为已审查的 Runtime/Metadata 字段；生成数据和外生预测来源保持原合同。

实测结果：

- 反事实 trace：`1`；合法 UPDATE 候选：`20`；有效有限标签：`2`；无效 trace/label 拒绝：`0/0`。没有标签被零值替代。ΔQ Huber loss `0.004551`、Trigger 排序 loss `0.687688`、Candidate 排序 loss `0.693429`，均有限。
- 两分支从同一决策前快照和外生输入开始；Micro 使用共同随机数 uniform，再分别根据分支自己的观测、合法候选集合和策略分布重采样。成功验证路径同时检查环境/策略指纹与 RNG 状态未被分支执行污染。共同随机数用于动作抽样配对，不强迫两个分支采取相同动作。Matching+KKT 按现有执行路径运行。
- ΔQ 价值头参数发生有限更新（参数差 L2 `0.256412`）；Macro/Micro Actor 与 Critic 均发生更新，所有更新参数有限。行为联合 log-prob 最大误差为 Macro `0`、Micro `0`；数值异常数 `0`，动作合法标志均通过。
- 两 Slot 实际净效用总计 `22.463655`；任务效用总计 `23.243564`；Cache 更新成本 `0.776169`；UAV 移动成本 `0.003740`。Cache UPDATE `7/9`，UAV MOVE `2/4`。奖励合同最大绝对误差 `0`。
- Matching+KKT 计时合计约 `4.2176 s`，429 次 solver 调用；记录的 Slot 端到端时间约为 `4.3872 s` 和 `0.0823 s`。KKT 均有限且收敛，最大资源残差约 `1.0e-12`、互补残差约 `1.36e-12`，未触及最大迭代次数。

## Checkpoint 往返与环境保护

运行成功完成 `1/1` 更新，生成 Update 14 Checkpoint。随后用 Stage3 Trainer 严格恢复该 Checkpoint：恢复的 update 计数为 `14`、Stage3 更新计数为 `1`、反事实 rollout 计数为 `1`；价值头及其优化器状态、随机状态和冻结 Profile 元数据均可读取。

`environment_data/` 完整保留，未删除、移动、重生成或写入任何文件。本轮没有修改正式配置、奖励、预测器、数据池或原始 Manifest；没有训练长任务、运行完整 Episode、Validation/Test、参数搜索，也没有 commit/push。

## 交付状态

```text
STAGE3_IS_SOLE_MAIN_ALGORITHM = YES  # 唯一正式 PPO 训练路径
OLD_PPO_EXECUTION_PATH_REMOVED = YES  # 旧训练/更新入口已退役；旧评价和诊断读取仍保留
COUNTERFACTUAL_LABEL_GENERATION_VALID = YES
STAGE3_TRAINING_UPDATE_VALID = YES
CHECKPOINT_ROUNDTRIP_VALID = YES
READY_FOR_MAIN_EXPERIMENT = YES  # 有界烟测通过；本任务未启动主实验
```

运行产物：`runs/research_mappo_stage3_cache_value/stage3_20261018T093959Z_6c5d4fa0/`，包括 `run_metadata.json`、`training_metrics.jsonl`、`run_completion.json` 和 `checkpoints/checkpoint_update_0014.pt`。
