# adaptive_cache_mec

本仓库保留候选条件化 Double DQN，学习每个 Slot 的 Cache/UAV 互斥配置动作。DQN 训练实现位于 `src/online_joint_dqn/`，系统物理模型、任务执行、Matching 和 KKT 作为共享环境组件保留。

## 训练

```powershell
conda activate adaptive_cache_mec
python -m online_joint_dqn.run_training --help
```

共享物理与数据参数位于 `configs/dqn/online_joint_dqn.toml`。四个 `scripts/training/run_online_joint_dqn_lr_*.py` 入口使用相同训练循环，只覆盖学习率。运行身份和恢复检查绑定 DQN 源码、配置及冻结 Training Pool；DQN 不依赖其他策略的 Checkpoint。

DQN 的状态、候选生成、Double DQN 目标、Replay 重建和训练合同见 [`src/online_joint_dqn/README.md`](src/online_joint_dqn/README.md)。当前训练入口不会自动执行 Test 评价。

## 数据

Training、Validation 和 Test Pool 使用独立 Episode。训练只读冻结 Training Pool；没有明确要求时，不重生成或改写现有 Pool、Manifest、Run、Checkpoint 或 `data/`。
