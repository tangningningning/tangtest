# adaptive_cache_mec

## 正式主线

当前唯一正式算法为 `PROPOSED`，使用 Architecture 33、Checkpoint 60 和策略 Schema `adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v4_multi_node_cache_factorized_v1`。正式配置是 `configs/proposed/system_config.toml`：`MULTI_NODE_FACTORIZED` Cache 控制 9 个节点（5 个 SV、2 个 RSU、2 架 UAV），其中两架 UAV 也由 Micro 策略控制位置。

Macro Cache SMDP PPO 在每个 frame 起点为每个 Cache 节点各采样一个 masked categorical 候选动作。每个节点的候选 0 是 KEEP，其余合法候选是 UPDATE；UPDATE logits 按该节点合法 UPDATE 候选数进行 cardinality correction。给定 observation 和候选特征后，九个节点的 Cache 动作按分解分布采样，联合 Cache log-prob 是九个节点 log-prob 之和。Macro 联合策略还包含 frame 内实际采样的 holding termination 决策：Cache 联合 log-prob 与 termination log-prob 之和构成单一 PPO ratio；强制最大持有时间和 Episode 终点不计入 termination likelihood。Holding 范围为 1–5 slots。

Micro UAV PPO 每个 slot 按固定顺序先采样 UAV0 位置，再基于 UAV0 的位置条件采样 UAV1 位置；两项条件 log-prob 相加形成 UAV 联合 log-prob，并使用一个 PPO ratio。UAV 候选 0 表示 STAY，其余候选表示 MOVE。训练保留当前 Micro Actor 的 STAY/MOVE 组熵软正则及现有 PPO 更新机制。

## 通信合同

- SV、RSU、UAV 的链路系统带宽各为 10 MHz。
- Edge 使用 `BANDWIDTH_DEPENDENT_NOISE_V1`；热噪声按 KKT 分配给任务的带宽计算。
- Cloud 使用每任务固定等效速率 1633066.7164444982 bit/s、时延系数 1.0681932194051897，以及总量 40 GHz 的有限共享 CPU 池。
- Cloud 速率和时延系数是基于 Training 数据条件拟合的抽象等效参数，不表示真实无线覆盖、吞吐保证或物理可靠性验证。

## 正式训练

正式 Training Pool 为 `environment_data/proposed_joint_training/`，包含 300 个 Training Episode（seed 20001–20300），每个 Episode 有 300 个 slot。正式 Runner 固定 PPO 初始化 seed 27005；每 4 个 Episode 执行一次 Policy Update，共 75 组更新。Macro Actor 学习率为 5e-4，所有优化器采用 constant 学习率合同。

在 `adaptive_cache_mec` Conda 环境中，从头训练和严格恢复训练均使用同一个正式 Runner、配置、Training Pool 和输出目录：

```powershell
conda activate adaptive_cache_mec
python scripts/training/mainline_runner.py
python scripts/training/mainline_runner.py --resume
```

Run 输出固定为 `runs/stageB3_10j_multi_node_v4_mainline/`。默认命令从头初始化策略和优化器；只要该 Run 目录已存在就会拒绝启动。`--resume` 要求该目录中已有 `run_manifest.json` 和 `checkpoints/latest_checkpoint.pt`，随后由 Trainer 执行既有严格恢复校验，恢复模型、优化器、计数器和随机数状态，并从完整 PPO 更新组边界继续。配置哈希、策略/架构、Checkpoint 版本、Training Pool provenance、Episode 数量、更新组边界和 Run 日志不匹配时会停止。新的通信配置不能严格恢复旧通信合同的 Checkpoint；此时应使用从头训练命令。

## Checkpoint

- `runs/stageB3_10j_multi_node_v4_mainline/checkpoints/latest_checkpoint.pt`：正式恢复训练和后续评价所用的 Checkpoint。
- `runs/stageB3_10j_multi_node_v4_mainline/checkpoints/best_train_return_checkpoint.pt`：按单个 Training Episode return 保存的训练诊断产物，不用于最终模型选择。

## 数据与评价入口

Episode 数据生成入口：

- `scripts/generation/generate_episode_realization.py`：生成单个 Episode realization。
- `scripts/generation/generate_episode_pool.py`：按 seed 范围生成 Episode Pool。

```powershell
python scripts/generation/generate_episode_realization.py --help
python scripts/generation/generate_episode_pool.py --help
ppo-eval --help
```

评价入口 `ppo-eval` 使用冻结策略并校验 Checkpoint 合同；正式评价和模型选择使用 `latest_checkpoint.pt`。Training、Validation 与 Test 使用各自对应的数据划分。

## 主流程

```text
Episode observation
→ 九节点因子化 Cache 动作与两 UAV 自回归位置动作
→ Cache/UAV 执行
→ Transfer–Exchange Matching → KKT → slot reward
→ 每四个独立 Episode realization 合并并进行一次 Macro/Micro PPO 更新
→ 在完整更新组边界保存 latest checkpoint
```
