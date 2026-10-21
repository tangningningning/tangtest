# Stage3 MAPPO Trigger–Candidate 解耦修复与条件整合

日期：2026-10-18

## 结论

固定标签专项门槛全部通过。Stage3 已在现有 MAPPO Trainer 中接入独立的 UPDATE 条件 Candidate 排序更新，并完成一次 Training 两 Slot 有界真实更新。默认未启用 Stage3 时，基础 v2 Actor 仍保留原参数结构；Stage3 显式启用时才安装独立 Candidate 路径。

本次固定标签结果说明：Candidate 可以在不改变 Trigger 操作概率的情况下学习冻结有限期 ΔQ 排序。它不证明完整 Episode 净效用提升，也不代表长期最优价值或泛化结论。

## 输入与复用的数据

- 起点 Checkpoint：Update 16，`runs/research_mappo_stage3_main_experiment/training/stage3_20261018T095817Z_1b3d4652/checkpoints/checkpoint_update_0016.pt`，SHA-256 `ca819d60320b3c65a46c2eb8217ca6d9e18f5a4bda2799f1778988f3976e611f`。
- Loss Profile：`configs/research/mappo_stage3_loss_coefficients_v1.json`，SHA-256 `c91bccc6bf91f76c407a18c9953b4fa2c2d346fcfef3337ad63ee49b23d8bdb4`。系数、平局容差、温度、Huber δ 和价值头学习率均未改变。
- 复用原固定标签：Training 16 个节点状态，`training_labels.npz` SHA-256 `0dcf47b2c44fee99b33394dfb1aeeb802bc9f9731b7e18c6f2a8605001900e49`；独立诊断 8 个节点状态，`diagnostic_labels.npz` SHA-256 `cf7a0289931ec86d9a4221732f725abd45e2a668bfa06d36aba1628308144a6e`。Training seeds 为 20031–20038，诊断 seeds 为 20039–20042。
- Training Pool Manifest SHA-256 为 `b6cae236b029f4d24c7488a91f0825247a20f475ec76b2d80737de44ae491bd3`。本轮专项优化没有重放环境、Matching+KKT 或反事实分支；诊断标签仅用于指标，不进入梯度。
- ΔQ 标签是基于 Update 16 冻结策略/Critic 的有限期条件价值估计，不是长期最优价值。

## 实现内容

修改模块：

- `src/ppo_simple/policy.py`
- `src/ppo_simple/mappo_baseline.py`

`policy.py` 保留现有 Candidate encoder/scorer 作为 Trigger 路径；Stage3 显式启用时，创建一套独立的 UPDATE 条件 Candidate encoder/scorer，并逐张量复制现有权重。Trigger 继续依据合法候选上的 UPDATE log-sum-exp 与 KEEP 分数组成操作组概率；条件候选路径只在合法 UPDATE 候选上做 softmax。联合动作仍按 `P(UPDATE,c|s)=P_trigger(UPDATE|s)×P_candidate(c|s,UPDATE)` 计算，KEEP/UPDATE mask 与原动作索引不变。

`mappo_baseline.py` 将 Candidate 参数显式分组。Stage3 Candidate ranking loss 使用分离后的条件 logits、冻结共享观测编码上下文，只对独立 Candidate 参数产生梯度；Trigger ranking loss 仍作用于原 Trigger 路径。正式更新顺序为联合 PPO Actor/Critic 更新、Candidate 专项排序更新、再由现有入口保存 Checkpoint。PPO 联合策略仍按完整分解后的 Macro log-prob 更新。

Candidate 专项步骤复用 Macro Actor Adam，但反向传播只为独立 Candidate 参数产生梯度，Trigger 和共享参数的梯度为 `None`，因此不会更新其 Adam 状态。Checkpoint 合同记录 `legacy_trigger_group_plus_independent_update_candidate_v1` 和 `macro_actor_adam_candidate_only_gradient_isolated_v1`。未增加新价值网络、奖励项或损失系数。

为保持基础 MAPPO 默认行为，分离头默认为关闭；Stage3 开关显式启用后才安装新头并扩展 Macro Actor 优化器。迁移新头时复制旧 Candidate 权重，并将相应 Macro Adam 状态映射到复制参数。

## 初始化等价与专项学习结果

对 16 个已有 Training 节点标签，使用 Update 16 状态恢复并比较旧分布与分离后的概率：

| 检查项 | 最大绝对误差 |
|---|---:|
| 完整 KEEP/UPDATE + Candidate 联合动作概率 | `2.3842e-7` |
| UPDATE 条件 Candidate 概率 | `2.2352e-8` |
| 所选联合动作 log-prob | `4.7684e-7` |
| 非法候选概率 | `0` |

单个保存标签行仅包含一个节点状态，因此该数组不能重建真实同一时刻的九节点联合状态。已有代数分解检查将逐节点 log-prob 求和；其九节点重复行检查最大误差为 `3.8147e-6`。此外，本轮真实 rollout 的行为到 replay Macro log-prob 最大误差为 `0`。

使用同一 Candidate pairwise ranking loss、固定标签、Macro Adam 学习率 `0.0005` 和既有梯度裁剪执行恰好 100 步。下表的诊断指标均在同一 8 个冻结诊断状态、同一冻结标签上计算；期望 ΔQ 和 regret 是标签加权指标。

| Step | Training pairwise 准确率 | 诊断 pairwise 准确率 | 诊断 `P(best)` | 诊断期望 ΔQ | 诊断 regret | 平均 `P(UPDATE)`：Train / 诊断 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.46361 | 0.52364 | 0.04140 | -0.20609 | 0.27075 | 0.51288 / 0.51716 |
| 20 | 0.85829 | 0.81457 | 0.06133 | -0.18133 | 0.03431 | 0.51288 / 0.51716 |
| 50 | 0.92749 | 0.86006 | 0.25568 | -0.04750 | 0.03431 | 0.51288 / 0.51716 |
| 100 | 0.95916 | 0.86969 | 0.34698 | 0.00780 | 0.02130 | 0.51288 / 0.51716 |

Step 0 到 100：Training 排序准确率提高 `0.49555`；诊断准确率提高 `0.34605`；诊断期望 ΔQ 提高 `0.21389`，平均 regret 降低 `0.24946`；8/8 个诊断状态的期望 ΔQ 和 regret 均改善。整个专项优化期间，训练和诊断状态的 Trigger `P(UPDATE)` 最大漂移均为 `0`，确定性 KEEP/UPDATE 边界翻转均为 `0`。

方向检查覆盖 8,432 个非平局候选对：16/16 个训练状态中，最高 ΔQ 候选的排序梯度方向均使其相对低价值候选上升；独立 Candidate 参数取得有限非零梯度，Trigger/共享参数没有 Candidate loss 梯度。以上仅是冻结标签上的专项拟合与小样本诊断。

**预设专项门槛：全部通过。** Training 准确率超过 70% 且提升至少 0.15；诊断准确率提升至少 0.05；诊断期望 ΔQ 上升、regret 下降；至少 5/8 节点改善；无非有限值、非法候选或 Trigger 漂移。

## 正式 Trainer 有界更新

在固定标签门槛通过后，使用 Stage3 正式入口从 Update 16 初始化，Training seed `20043`，执行一次更新、两个连续 Slot、一个 Macro 状态、两个被选反事实目标节点；没有运行完整 Episode、Validation 或 Test。run 目录为 `runs/research_mappo_stage3_trigger_candidate_decoupling/formal_smoke/stage3_20261018T122316Z_5ea49321/`。

- Update 17 Checkpoint：`checkpoints/checkpoint_update_0017.pt`，SHA-256 `55bfc5349b59376649285dc2604163c169bd236dd9c1df961a0129d4a5a8ee21`。Update 16 v4 来源哈希与 Profile 哈希写入 run 元数据。
- 生成并接受 2 个有限真实反事实标签；无效标签拒绝数为 0。记录 1 条 trace、92 个分支，反事实 replay 约 `16.600` 秒，Candidate 有效比较对 2,011。
- Candidate ranking loss `0.695204`，Candidate 梯度范数 `0.001017`，Candidate 参数变化 L2 `0.025487`；Candidate 专项更新计数增加 1。Macro/Micro Actor 与两个 Critic 均发生有限参数更新。
- Macro/Micro 行为到 replay log-prob 误差均为 `0`；Macro/Micro KL 分别为 `1.19e-7` 和 `6.44e-6`，clip fraction 均为 `0`；无 NaN/Inf 或数值错误。
- 两 Slot 累计净效用 `23.64486`，Cache 成本 `0.30215`，UAV 移动成本 `0.00473`。这些是短 Training 窗口的运行诊断，不是完整 Episode 性能结果。
- Cache/UAV 动作合法、移动可行，奖励合同最大误差 `0`。Matching+KKT 总耗时 `4.2054` 秒、287 次 solver call，最大资源残差约 `9.99e-13`。Slot 端到端耗时为 `4.3964` 秒与 `0.0455` 秒；PPO 优化计时 `16.6732` 秒。
- 保存的 Update 17 为 Stage3 Checkpoint v5。恢复后策略张量、5 个优化器状态、Python/NumPy/Torch CPU/Trainer generator/Micro 共同随机流均与 Checkpoint 一致；进度恢复为 Update 17、Stage3 更新计数 4、Candidate 专项更新计数 1。

真实两 Slot 更新发生在本轮末尾的兼容性加固之前；随后加固只将分离头从构造时默认安装改为 Stage3 显式启用时安装，并保留相同模块初始化、权重复制、优化器状态迁移和更新合同。没有为了复核再执行第二次训练。本轮在最终代码上完成了默认/v2、Stage3 v4、Stage3 v5 的恢复与概率路径静态核验。

## Checkpoint 与默认兼容性

- 基础 MAPPO 默认构造时不包含新 Candidate 参数，Macro Actor 仍为 10 个参数张量；Update 13 v2 Checkpoint 在默认模式严格加载成功。
- 显式启用 Stage3 后，U13 v2 的 Stage3 初始化迁移通过：conditional Candidate 权重精确复制，Macro Adam 状态迁移；Update 16 v4 旧 Stage3 Checkpoint 也迁移至分离结构且 Candidate 权重逐张量一致。
- 当前 v5 Update 17 Checkpoint 可严格恢复策略、优化器、RNG 和进度状态。没有忽略未兼容参数或静默丢弃优化器状态。

## 限制与结论

Candidate 监督信号在固定样本上方向正确，且 Trigger 漂移已由参数隔离消除；Stage3 的 Candidate 专项更新已接入正式 MAPPO Trainer 并通过一次真实两 Slot 更新。**这不证明新 Candidate 已提高完整 Episode 净效用。** 后续受控训练可在已接入的 Stage3 路径上进行，但应在之后单独决定是否开展冻结 Validation；本报告不启动下一轮实验。

本轮只新增本报告并维护上述两个算法模块；未改变正式配置、奖励、数据或 Profile。未访问 Test，未修改、移动或删除 `environment_data/`，未创建 Worktree，未 commit/push。Python 检查均使用 `-B`，没有生成临时脚本。

```makefile
TRIGGER_CANDIDATE_DECOUPLED = YES
INITIAL_POLICY_EQUIVALENT = YES
TRIGGER_DRIFT_ELIMINATED = YES
CANDIDATE_SUPERVISED_LEARNING_VALID = YES
FORMAL_MAPPO_INTEGRATION = YES
CHECKPOINT_ROUNDTRIP_VALID = YES
READY_FOR_CONTROLLED_TRAINING = YES
```
