# adaptive_cache_mec

## 唯一正式强化学习算法

正式算法为 **Adaptive Dual-Timescale Joint MAPPO**（自适应双时间尺度联合多智能体近端策略优化算法），算法标识为 `ADAPTIVE_CACHE_UAV_JOINT_MAPPO`，Checkpoint 方法为 `ADAPTIVE_CACHE_UAV_MAPPO`。

该实现使用共享参数、全局观测、因子化联合动作和集中式价值估计。它不是完全分布式执行的标准 CTDE-MAPPO。Cache Holding 自适应选择 1–5 Slot；两架 UAV 每 Slot 决策；任务卸载和资源分配由正式 Matching+KKT 完成。训练使用 PPO Clip、Macro SMDP-GAE、Micro-GAE 和现有 Critic。

唯一正式训练入口：

```powershell
conda activate adaptive_cache_mec
python scripts/training/mainline_runner.py --help
```

冻结评价入口为 `python -m ppo_simple.evaluator`（已安装环境也可用 `ppo-eval`）。`ppo-train` 与 `python -m ppo_simple.trainer` 是同一正式 Trainer 的兼容命令，不是独立算法实现。

正式合同、时间尺度、动作概率、Transition 与 Checkpoint 兼容规则见[正式算法设计](docs/mappo_complete_solver.md)。正式配置唯一来源为 `configs/proposed/system_config.toml`。训练和评价按对应 Pool Manifest 与 Checkpoint 合同校验；不兼容配置或张量不会被静默加载。

## 非强化学习研究对照

预测驱动的全局 Cache 求解器是独立的非强化学习对照，不属于正式 MAPPO 主线。它使用 Training 配对标签拟合边际净效用模型，在有界预算内搜索单节点 Cache 更新；基线评价通过正式 Matching+KKT 执行，UAV 保持 STAY。其合同与结果见[预测缓存求解器说明](docs/predictive_global_cache_solver.md)和 `reports/predictive_global_cache_first_training.md`。

历史 Stage3 报告和 Checkpoint 保留为研究记录；对应的独立训练器、运行入口、Stage3 专属策略分支和损失配置已从可运行代码中清理。历史报告不代表当前正式算法或 Checkpoint 合同。

## 数据边界

Training、Validation 与 Test 使用各自的数据划分。正式训练仅使用 Training Pool，正式冻结评价仅使用当前 MAPPO Checkpoint 合同对应的评价 Pool。除非明确执行数据生成任务，不重生成或改写现有 Pool、Manifest 或 `environment_data/`。
