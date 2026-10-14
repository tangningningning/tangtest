# Stage B3-10K：新通信环境下的多节点 PPO 单 Slot 端到端验证

## 结论

**STAGE_B3_10K = COMPLETE**；Slot 0 完整通过。使用 Training Episode 20001、Slot 0（origin 0，绝对时间 100 s），实际到达任务 12 个。验证 seed 为 **4401**；未运行完整 Episode、PPO 训练或梯度更新。

## 配置与策略

- Conda 环境解释器：`C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`，`sys.prefix` 指向 `adaptive_cache_mec`。
- 配置：`configs/proposed/system_config_multi_node_comm_v2_stageB3_10j_equal_share.toml`（SHA-256 `e3b2f9e4768f9d699ead717a34b684f8316feee1909e5ce356a54f8fa1b8c6ec`）。Episode manifest 标记为 Training split，episode seed 20001。
- 新初始化 CPU Actor/Critic 和 4 个正式优化器；Schema `v4_multi_node_cache_factorized_v1`，Architecture 33，Checkpoint 合同 60。未加载 Checkpoint；优化器 state 初始化/结束均为空，optimizer steps = 0。
- Cache 次序：SV `46, 10, 0, 23, 59`；RSU `RSU_0, RSU_1`；UAV Cache 节点 `UAV_0, UAV_1`。UAV 控制次序 `UAV_0, UAV_1`。Observation 为 400 维且有限。

## 通信与动作

- SV/RSU/UAV 带宽均 10 MHz；噪声模型 `BANDWIDTH_DEPENDENT_NOISE_V1`，PSD −174 dBm/Hz，NF 9 dB。Ground 5.9 GHz；空地信道使用配置中的 2 GHz 参数。
- Cloud 等效上行速率 `1633066.7164444982 bit/s`，时延系数 `1.0681932194051897`，有限共享 CPU 池 40 GHz；等效速率不按任务数分摊。运行时 `resource_sharing=false`。
- Cache factorized 候选特征 `[9,57,200]`，Legal Mask `[9,57]`；UAV 候选张量 `[2,25,41]`。概率、log-prob 和 value 均有限；Cache 概率行和最大误差 `1.12e-7`，非法动作概率质量为 0，抽样动作全部合法。Cache 联合 log-prob `−28.26556253` 等于 9 个节点 log-prob 之和（误差 0）；UAV 条件联合 log-prob 为 `−7.18695211`。

| Cache 节点 | Actor 请求 | 实际变化的 node-service 项 |
|---|---:|---:|
| SV 46 | UPDATE | 2 |
| SV 10 | UPDATE | 2 |
| SV 0 | KEEP | 0 |
| SV 23 | KEEP | 0 |
| SV 59 | UPDATE | 2 |
| RSU 0、RSU 1 | 各 UPDATE | 各 2 |
| UAV Cache 0、UAV Cache 1 | 各 UPDATE | 各 4 |

Cache 请求为 KEEP 2、UPDATE 7；实际更新节点 7 个、变化的 node-service 项 18 个；未生效请求 0。策略目标与环境最终 Cache 一致，容量及服务可用性校验通过。

UAV 控制请求 MOVE 2、STAY 0；实际移动 2 架，各移动 7.5 m（最大速度 15 m/s）。目标位置与环境最终位置一致；位置/安全距离转移校验通过，无动作修正。

## Matching、KKT 与 Reward

- Matching 正常停止：`converged_by_no_improvement`，4 次更新；稳定性与 resource-feasible 检查均通过。
- KKT：22/22 求解收敛且有限，最大迭代命中 0。资源残差最大 `9.33e-13`，stationarity 残差最大 `1.53e-16`，互补松弛残差最大 `2.22e-13`。边缘 CPU/带宽分配满足容量（含浮点容差）；Cloud CPU 使用 40/40 GHz。任务分配、缓存可用性、UAV 约束均通过；节点任务数之和 12，与到达任务一致：SV 1、RSU 2、UAV 3、Cloud 6。
- 时延：总和 64.2648 task-s，均值 5.3554 s，最大 13.4182 s。车辆能耗总和 8.00644 J，均值 0.66720 J。该配置采用 `independent_per_slot` 效用合同，时延超过 1 s 不作为任务失败。
- Task Utility `Q = 9.6954958636`；Cache 成本 `C_cache = 0.7714714467`；UAV 成本 `C_UAV = 0.0037400000`（移动能耗 225 J）；Reward `G = 8.9202844170`。`G = Q − C_cache − C_UAV`，分解误差 0。NaN/Inf 数量为 0。

## 下一步

**INDEPENDENT_PPO_TRAINING_READY = YES**：本次指定通信配置下的真实单 Slot 执行链路通过；后续可从新初始化策略启动独立 PPO Run。训练本身及跨 Slot 学习稳定性未在本阶段验证。

本轮未修改源码、配置或 Runner。`configs/proposed/system_config.toml` 与 `scripts/training/mainline_runner.py` 在本轮开始前已显示为 modified，现状保留未动。未创建永久脚本、配置、Run 或大型诊断数据；临时 Numba 缓存未生成/残留。未 commit、未 push。
