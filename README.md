# adaptive_cache_mec

## 唯一正式强化学习算法

正式算法为 **Alternating Actor Optimization V1（MAPPO）**，算法标识为 `ADAPTIVE_CACHE_UAV_JOINT_MAPPO`，Checkpoint 方法为 `ADAPTIVE_CACHE_UAV_MAPPO`。这里的 Joint 表示联合 Cache/UAV 策略；Macro 与 Micro Actor 按正式调度交替更新。

每组采集 4 个完整 Episode 后更新一次；第一组更新 Micro Actor，第二组更新 Macro Actor，随后交替。每组都更新 Macro、Micro 和 Long-Value Critic，非活跃 Actor 及其 Adam 状态保持不变。实现使用共享参数、全局观测、因子化联合动作和集中式价值估计；Cache Holding 自适应选择 1–5 Slot，两架 UAV 每 Slot 决策，任务卸载和资源分配由正式 Matching+KKT 完成。

唯一正式训练入口：

```powershell
conda activate adaptive_cache_mec
python scripts/training/mainline_runner.py --help
```

冻结评价入口为 `python -m ppo_simple.evaluator`（已安装环境也可用 `ppo-eval`）。`src/ppo_simple/trainer.py` 是由正式 Runner 调用的实现模块，不提供第二个命令行训练入口。

正式合同、时间尺度、动作概率、Transition 与 Checkpoint 兼容规则见[正式算法设计](docs/mappo_complete_solver.md)。MAPPO 训练唯一配置为 `configs/proposed/alternating_actor_updates_v1.toml`；`configs/proposed/system_config.toml` 继续提供系统物理参数和数据生成合同。训练和评价按对应 Pool Manifest 与 Checkpoint 合同校验；不兼容配置或张量不会被静默加载。

## 非强化学习研究对照

预测驱动的全局 Cache 求解器是独立的非强化学习对照，不属于正式 MAPPO 主线。它使用 Training 配对标签拟合边际净效用模型，在有界预算内搜索单节点 Cache 更新；基线评价通过正式 Matching+KKT 执行，UAV 保持 STAY。其合同与结果见[预测缓存求解器说明](docs/predictive_global_cache_solver.md)和 `reports/predictive_global_cache_first_training.md`。

历史报告和 Checkpoint 保留为研究记录。正式 Runner 只接受 Alternating Actor Optimization V1 配置；为历史 Checkpoint 评价和专项合同测试保留的其他 Cache 动作 Schema，不会通过正式 Training Pool 门禁。历史报告不代表当前正式算法或 Checkpoint 合同。

## 数据边界

Training、Validation 与 Test 使用各自的数据划分。正式训练仅使用 Training Pool，正式冻结评价仅使用当前 MAPPO Checkpoint 合同对应的评价 Pool。除非明确执行数据生成任务，不重生成或改写现有 Pool、Manifest 或 `environment_data/`。
