# Stage B3-10J：多节点PPO通信配置接入

## 结果

**STAGE_B3_10J = COMPLETE。** 本轮按补充指令，以B3-6独立多节点v4实验配置为基线，创建了独立通信配置。该配置不替换或提升单节点v3正式主线。

## 策略与数据基线

新配置从 configs/proposed/system_config_multi_node_factorized.toml 派生，保留：

- Schema：adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v4_multi_node_cache_factorized_v1
- Architecture 33、Checkpoint合同60。
- MULTI_NODE_FACTORIZED及ABSTRACT_MACRO_BOUNDARY_RECONFIGURATION。
- 9个Cache节点（5 SV、2 RSU、2 UAV）及2个UAV控制节点。
- Actor/Critic、PPO更新机制、所有学习参数、Reward和惩罚权重。
- Training Pool路径及300 Episode、每Episode 300 Slot的full_episode_randomized协议。

正式主线 system_config.toml 仍为v3单节点Schema，未修改。正式 mainline_runner.py 仍固定选择该v3配置。新配置用于独立v4实验。

## 通信合同

Edge配置采用B3-10C字段：SV、RSU、UAV各10 MHz；Ground载频5.9 GHz；BANDWIDTH_DEPENDENT_NOISE_V1；噪声PSD −174 dBm/Hz、NF 9 dB；正交独立频谱池且无显式干扰。既有Ground和空地信道参数保留B3-10C合同，未改Matching/KKT源码。

Cloud配置采用B3-10H参数：固定等效速率1633066.7164444982 bit/s，时延系数1.0681932194051897，40 GHz有限共享CPU池。三个重复速率字段保持一致，resource_sharing=false，Cloud等效速率不按并发任务再次分摊；Cloud继续与合法Edge自由竞争。Cloud速率和时延系数来源均标为TRAIN_ONLY_CALIBRATION。原有Cloud时延、车辆能耗公式合同保持不变。

## 差异核对

对新配置和B3-6基线进行完整解析值比较，23项差异全部属于获准范围：13项通信配置值和10项对应来源状态。除通信参数及来源标记外，没有其他语义配置差异。

objective、Cache/UAV成本参数、PPO配置与数据协议均与B3-6基线一致；Reward仍为G = Q - C_cache - C_UAV。parameter_status中的Cloud速率和时延系数标记为训练集校准值，新增Edge噪声与资源模型来源标记沿用B3-10C。

## 最小兼容性检查

使用 C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe，确认解释器前缀属于adaptive_cache_mec。

| 检查 | 结果 |
|---|---|
| TOML解析、项目配置加载及正式参数校验 | PASS |
| Edge带宽、噪声、地面/空地合同字段 | PASS |
| Cloud重复速率、时延系数、有限CPU池及来源标记 | PASS |
| 基线语义差异范围 | PASS：无未授权的非通信差异 |
| Training数据协议 | PASS：使用Training episode_20001构造环境，300 Slot协议有效；Training Pool有300个Episode目录 |
| 多节点Actor/Critic构造 | PASS：Architecture 33、Checkpoint 60、v4 Schema；9 Cache节点按5/2/2角色排列，UAV为UAV_0、UAV_1 |
| 动作张量合同 | PASS：Cache动作[B,N]，候选特征[B,N,K,200]，联合log-prob为节点log-prob之和 |
| 无Checkpoint初始化 | PASS：新策略及优化器从空训练状态构造，计数为0；未加载B3-6旧Checkpoint |

没有执行Matching/KKT或PPO更新，符合本阶段边界。B3-6旧Checkpoint属于旧通信配置；虽有相同的v4网络架构，也不作为新通信实验的严格Resume起点。新实验应从新初始化开始并使用独立Run目录和Checkpoint谱系。

## 启动接口与隔离

底层Trainer支持 --config、--episode-pool 和 --run-dir 参数，可在后续独立实验中指定新配置和独立输出目录。正式Stage6主线启动器仍固定使用v3配置，本轮未修改。新配置继承B3-6的experiment.output_tag；后续运行必须通过独立--run-dir隔离产物。

## 最终状态

- STAGE_B3_10J = COMPLETE
- BASELINE = B3-6_INDEPENDENT_MULTI_NODE_V4
- FORMAL_MAINLINE = V3_UNCHANGED
- MULTI_NODE_SCHEMA_PRESERVED = YES
- NEW_CONFIG_CREATED = YES
- CONFIG_LOAD_CHECK = PASS
- POLICY_CONSTRUCTION_CHECK = PASS
- EDGE_COMMUNICATION_MODE = BANDWIDTH_DEPENDENT_NOISE_V1
- CLOUD_RATE_BPS = 1633066.7164444982
- CLOUD_DELAY_FACTOR = 1.0681932194051897
- REWARD_UNCHANGED = YES
- PENALTY_WEIGHTS_UNCHANGED = YES
- TRAINING_POOL_COMPATIBLE = YES
- FRESH_INITIALIZATION_REQUIRED = YES
- INDEPENDENT_PPO_READY = YES（配置加载与策略构造已验证；未验证训练/KKT运行）
- FORMAL_CONFIG_CHANGED = NO
- FORMAL_RUNNER_CHANGED = NO
- PPO_TRAINING_EXECUTED = NO
- MATCHING_KKT_EXECUTED = NO
