# 在线缓存—UAV 候选条件化 Double DQN

本目录实现独立的集中式 Double DQN，不接入 MAPPO、训练主线或现有 checkpoint。
每个当前 Slot 的合法候选集合包含 `KEEP_STAY`，并可包含合法 `CACHE_UPDATE`
及 `UAV_MOVE`。缓存更新和 UAV 移动在动作构造、环境适配和正式执行结果三处检查互斥。

## CleanRL 上游来源

- 仓库：<https://github.com/vwxyzjn/cleanrl>
- 参考文件：[`cleanrl/dqn.py`](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/dqn.py)
- 检查的上游 `master` commit：`fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`
- 许可证：MIT，版权声明为 `Copyright (c) 2019 CleanRL developers`；`CLEANRL_LICENSE.txt` 保留 `dqn.py` 适用的上游 MIT 声明及许可条款。
- 借鉴部分：在线 Q 网络与目标网络初始化、经验回放和随机采样、epsilon 线性调度、优化更新周期、目标网络同步及训练指标记录。
- 本项目修改：不复制 CleanRL 固定离散动作输出层，也不照搬其普通 DQN 的 `max Q_target` TD 目标。网络改为 `Q(state, candidate_features) -> scalar`；Double DQN 先由在线网络在下一状态合法候选中选动作，再由目标网络估值。状态推进、动作特征、经验序列化和合法动作集合均按当前系统接口重写。

上游 `dqn.py` 使用固定动作维度并以目标网络最大值构造普通 DQN 目标；本项目的动作数量和动作语义会随状态改变，因此只移植与动作空间无关的训练骨架。没有复制 CleanRL 仓库，也没有增加 CleanRL 的 Gymnasium、Tyro 或 TensorBoard 依赖。

## 文件

- `candidate_features.py`：稳定动作表示、三类互斥检查，以及状态和候选特征编码。缓存候选包括目标缓存、节点变化、删除/新增服务、正式缓存下载成本与新增数据量；UAV 候选包括完整位置配置、位移、正式移动成本和当前任务距离特征。
- `agent.py`：候选条件化 Q 网络、epsilon-greedy 选择、Double DQN TD 目标、Huber 更新、梯度裁剪和目标网络同步。
- `replay_buffer.py`：固定容量回放。保存语义动作、选中动作特征、下一状态及该下一状态生成的全部合法候选；下一状态指纹必须与每个下一候选的来源指纹一致，不依赖候选序号。
- `environment_adapter.py`：从当前正式 Training Pool 读取并校验短 Episode 窗口；合法候选复用 `online_joint_optimizer` 的候选生成器；只把选中的动作交给 `slot_environment.execute_configured_slot`。
- `trainer.py`：独立训练循环，使用正式执行器返回的单 Slot 系统效用作为奖励。

## 动作与特征

动作以完整缓存矩阵和有序 UAV 目标位置序列序列化，候选 ID 同时绑定来源状态和目标配置。缓存 REPLACE 通过正式 `cache_configuration.is_legal_cache_replacement` 验证；UAV 目标通过正式边界、最大位移和安全距离验证。缓存候选的 UAV 配置保持不变，UAV 候选的缓存矩阵保持不变。

状态特征包括当前缓存、当前 UAV 位置、当前任务数量及服务/负载摘要、决策时可用的 GRU 服务流行度预测、当前预测位置和有效预测掩码。未来任务与未来真实车辆位置不会用于动作特征或奖励。

候选特征包括操作类型、完整目标缓存、发生变化的缓存节点、逐节点删除/新增服务、正式缓存下载成本与延迟、UAV 前后位置、移动距离/成本，以及当前任务车辆到 UAV 的距离摘要。候选生成不运行 Matching/KKT；每一步只正式执行已选动作一次。

## 学习语义

适配器当前每个动作恰好执行一个正式 Slot，转移时长为一个 Slot，因此 TD 折扣为 `gamma ** 1`。奖励是执行器对当前真实任务计算的一次 `system_utility`；预测时域累计分数不进入 replay 奖励。缓存配置和 UAV 位置由正式结果推进并跨 Slot 保留。若将来动作宏持续多 Slot，replay 中的 `duration_slots` 会用于 `gamma ** duration_slots`。

推理默认 `epsilon=0`。训练循环使用 epsilon-greedy；终止转移的 TD 目标只取即时奖励。非终止转移必须包含与下一状态相符的合法候选集合。
训练窗口在真实 Episode 结束前截断时，适配器会额外读取一个下一 Slot 作为 bootstrap 观测；该 Slot 不再执行动作，最后一个转移仍保留合法的下一状态候选，避免把截断误标成环境终止。

## 最小使用示例

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "src"))

from online_joint_dqn import (
    DoubleDQNAgent,
    ExclusiveDQNEnvironment,
    ReplayBuffer,
    TrainerConfig,
    train_episode,
)

env = ExclusiveDQNEnvironment.from_formal_training_episode(
    Path.cwd(), episode_seed=20001, origin_start=0, max_steps=300
)
agent = DoubleDQNAgent(env.encoder.state_dim, env.encoder.candidate_dim, seed=7)
replay = ReplayBuffer(10000, seed=7)
metrics = train_episode(
    env, agent, replay,
    config=TrainerConfig(learning_starts=1000),
)
```

上例只是独立入口示意。短窗口不足以建立稳定训练结论；本次实现验证不代表算法已收敛。正式配置、奖励和 MAPPO checkpoint 均不由该目录修改。
