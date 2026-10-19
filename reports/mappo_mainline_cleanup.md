# MAPPO 主线清理记录

## 正式主线

- 算法：**Adaptive Dual-Timescale Joint MAPPO**（自适应双时间尺度联合多智能体近端策略优化算法）。算法标识 `ADAPTIVE_CACHE_UAV_JOINT_MAPPO`；Checkpoint 方法 `ADAPTIVE_CACHE_UAV_MAPPO`。
- 唯一正式训练入口：`scripts/training/mainline_runner.py`，转发到 `src/ppo_simple/trainer.py:PPOTrainer`。
- 唯一正式冻结评价实现：`python -m ppo_simple.evaluator`（安装入口 `ppo-eval`）。`ppo-train` 与 `python -m ppo_simple.trainer` 是同一 Trainer 的兼容调用，不是独立实现。
- 实现采用共享参数、全局观测、因子化联合动作和集中式价值估计；不是完全分布式执行的标准 CTDE-MAPPO。
- 正式合同保持：9 个 Cache 节点的因子化动作，自适应 Cache Holding 1–5 Slot，UAV 每 Slot 决策，Matching+KKT，4 个完整 Episode 后更新一次策略，每 Episode 300 Slot；PPO Clip、SMDP-GAE、Micro-GAE 与现有 Critic 保留。

## 保留模块

正式训练和评价保留 `trainer.py`、`policy.py`、`rollout.py`、`interface.py`、`primitive_control.py`、`ppo_math.py`、`macro_targets.py`、`reporting.py`、`evaluator.py`，以及环境生成、预测、Cache/UAV、Matching/KKT 相关依赖。正式训练入口和评价入口只依赖当前 MAPPO Checkpoint 合同。

## 清理结果

以下 Stage3 独立强化学习训练链路已删除，因为静态导入检查确认其不属于正式 Trainer/Evaluator 依赖：

- `src/ppo_simple/mappo_baseline.py`
- `src/ppo_simple/cache_counterfactual_value.py`
- `src/ppo_simple/stage3_cli.py`
- `scripts/training/run_mappo_stage3_cache_value.py`
- `scripts/evaluation/run_mappo_stage3_main_experiment.py`
- `configs/research/mappo_stage3_loss_coefficients_v1.json`

原 `scripts/training/mappo_stage3_pool_contract.py` 含有被非强化学习预测 Cache 基线及既有测试复用的 Pool 合同工具，故将通用部分移至 `scripts/training/training_pool_contract.py`，并更新必要导入；此共用依赖保留。`run_prediction_greedy_cache_baseline.py` 不再导入 Stage3 Trainer 常量，保留其非强化学习对照语义。预测驱动全局 Cache 监督模型和求解器也作为已有非强化学习研究对照保留，不属于 MAPPO 训练主线。

`policy.py` 中仅服务 Stage3 的独立 Trigger/Candidate 分支已移除。正式分支的模型状态字典键、形状、初始化值及采样/确定性动作、概率、log-prob、熵和值输出经同随机种子对照保持一致；正式 Actor/Critic 参数形状与名称未变。Stage3 专属损失与检查点不会由正式入口恢复。

历史 Stage3 报告、已有 Checkpoint、`runs/`、正式数据池及其他既有研究文件均保留；`environment_data/` 未触碰。未发现另一个独立可运行的 MAPPO/PPO Trainer 实现；上述兼容命令均进入同一 `PPOTrainer`。

## 版本与工作区

- 清理基于 `main` 起始版本 `22826c4a0a939a5d72c083f26547e48bbd0e527c`。正式配置未改动，见上方哈希记录。
- `configs/proposed/system_config.toml` 未修改；当前 SHA-256：`ED4596E8DF7F0E7B15144C845BBC2C86F77E0C300C60183AD1DFD215777FC2A5`。配置读取、严格正式配置校验及从 Training Episode 20001 初始化 Trainer 均通过；检查到 Cloud 有限 CPU 池为 20 GHz、上行 20 Mbps、时延系数 13。
- 修改范围为本报告所列的 README、算法说明、正式入口文案、Stage3 文件删除、共享 Pool 工具通用化、非强化学习基线导入和对应测试引用。没有触碰用户已有的未跟踪数据、运行目录或历史报告。

## 最小检查

- 正式 `mainline_runner.py` 和冻结 Evaluator 导入及 `--help`：PASS。
- 正式配置解析和 `validate_for_formal_experiment`：PASS。
- Actor/Critic 按 Episode 20001 正式环境上下文初始化：PASS；未调用 `Trainer.update()`。
- 正式与保留的非强化学习模块静态导入、修改 Python 文件 AST 解析：PASS。
- Stage3 专属模块/入口的活动源码引用检查：无残留引用；共享 Pool 合同已迁移至通用模块。
- `git diff --check`：PASS（Git 仅提示现有文件换行符将来可能转换，不是空白错误）。
- 未运行 PPO 训练或 Episode，未改正式配置、Reward、正式 Actor/Critic 算法行为或 Training 数据。历史运行结果、Checkpoint、既有未跟踪报告和数据均保留。
