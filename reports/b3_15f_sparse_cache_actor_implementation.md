# B3-15F Sparse Coordinated Cache Actor

## 结果

已实现独立的稀疏协调Cache Actor与实验配置。正式9节点因子化策略、正式配置、已有Run和Checkpoint均保留；未迁移旧权重或优化器状态。

- 策略Schema：`adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v5_sparse_coordinated_cache_v1`
- Architecture：34
- Checkpoint版本：61
- 配置：[sparse_coordinated_cache_v1.toml](../configs/proposed/sparse_coordinated_cache_v1.toml)

## 动作与概率合同

每帧先在ALL_KEEP和具有合法UPDATE候选的Cache节点之间采样组动作。选择ALL_KEEP时九个节点均不变；选择节点`i`时，仅从该节点的合法目标配置中采样一个候选。因此动作构造本身保证每个Macro帧最多更新一个节点，不会先采样九个动作再强制覆写。

Cache联合对数概率为：

`log P(group | s) + log P(candidate | group, s)`

其中ALL_KEEP没有条件候选项。该Cache联合概率再与原Holding Termination对数概率组成Macro联合对数概率。组熵采用`H(group) + Σ P(group=i) H(candidate | group=i)`；PPO ratio及现有基于联合log-ratio的近似KL都使用同一联合分布。合法节点组与条件候选分别应用mask。动作特征复用现有候选编码器及正式成本/预测特征，不输入未来真实任务效用。

## 修改与隔离

- 新增独立配置；其系统、Reward、Training Pool和PPO超参数基于正式配置，新增实验名称、稀疏Schema/动作模式和单节点更新上限。
- 扩展配置校验、策略、Rollout、Trainer及Evaluator识别新Schema。新检查点使用Architecture 34 / Checkpoint 61；旧Schema的Checkpoint不会被当作新策略恢复。
- Matching/KKT、Reward、系统模型、Cloud、UAV策略及正式`system_config.toml`未修改。旧MULTI_NODE_FACTORIZED分支保持原概率与日志合同。

## 已执行的最小检查

- 新配置载入，Trainer构造为SparseCoordinatedCacheActorCritic；正式配置仍载入为MULTI_NODE_FACTORIZED。
- 从Training Episode 20001首个状态生成候选并检查候选配置容量合法；未执行Slot、Matching/KKT或Reward。
- 合成Macro Batch覆盖Holding时长1–5：ALL_KEEP不改缓存，UPDATE只改一个节点；采样与重新评价的联合Log-Prob满足原5e-6断言；错误概率被拒绝。
- 分层熵、同策略KL/Clip诊断及新增Actor头的有限非零梯度检查通过；新策略状态字典严格往返检查通过。
- 六个修改Python文件语法检查通过，`git diff --check`通过。正式配置、Reward及系统模型差异检查为空。

未运行PPO训练、更新组、完整Slot、Validation或Test；未创建Checkpoint。以上检查验证动作/概率合同，不证明稀疏策略的收益或训练效果。

## 建议的短程对照协议

如获准启动，先分别以正式因子化配置和本配置做独立Fresh Run：各4个Training Episode、1个PPO更新；初始化Seed 27005、Episode Seeds 20001–20004，更新组大小4、其余参数及数据相同，输出目录互不重叠且均须不存在。使用当前Trainer的`--config`入口直接指定对应配置，不走固定正式配置的`mainline_runner.py`。这仅是执行合同与早期信号检查，不能用于宣布策略优劣；如需判断收益，再预先约定更充分的训练与Validation方案。
