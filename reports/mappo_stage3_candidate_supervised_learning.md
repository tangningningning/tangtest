# Stage3 MAPPO Candidate 专项监督学习与条件整合

## 结论

固定 Update 16 反事实标签上的 Candidate 学习信号很强，且在独立诊断节点上也有一致改善；但 Candidate 编码器/打分器与 KEEP/UPDATE 操作组概率共用打分路径，导致 `P(UPDATE)` 大幅漂移。最终 Step 100 有 3/8 个诊断节点跨过 0.5 确定性操作边界，Step 50 则 8/8 个诊断节点都从 UPDATE 翻到 KEEP。预设条件 A 的 Trigger 漂移门槛未通过，因此本轮不整合正式 MAPPO，也不进行后续 MAPPO 验证。

本结果只表示在有限期、Update 16 Critic bootstrap 标签上的排序拟合和小样本独立节点改善；不能解释为长期最优价值、完整 Episode 净效用改善或统计显著的泛化结论。

## 固定样本与标签合同

- 起始策略、Critic 和 Stage3 价值头来自正式 Update 16：`runs/research_mappo_stage3_main_experiment/training/stage3_20261018T095817Z_1b3d4652/checkpoints/checkpoint_update_0016.pt`，SHA-256 `ca819d60320b3c65a46c2eb8217ca6d9e18f5a4bda2799f1778988f3976e611f`。每个 Episode 均严格加载此 Checkpoint；未使用 Update 17–24。
- 冻结损失 Profile `configs/research/mappo_stage3_loss_coefficients_v1.json` 的 SHA-256 为 `c91bccc6bf91f76c407a18c9953b4fa2c2d346fcfef3337ad63ee49b23d8bdb4`。
- Training Pool Manifest SHA-256：`b6cae236b029f4d24c7488a91f0825247a20f475ec76b2d80737de44ae491bd3`；生成 fingerprint：`4083b917948cd7203972c0edd5b40d2efefd8984dde27089a77da43d460b32f3`。
- 按既有 Manifest seed 顺序选择 8 个 Training Episode：20031–20038；另取互不重叠的 4 个独立诊断 Episode：20039–20042。每个 Episode 使用 Slot 0 起始的一个完整两 Slot Macro 状态，每状态选两个目标节点。Episode Manifest 哈希、节点 ID 与分层信息保存在 `fixed_labels_metadata.json`。
- 实际样本为 16 个训练节点行和 8 个诊断节点行。训练角色覆盖 SV/RSU/UAV 为 8/4/4；诊断为 2/4/2。不同 Episode 的 SV ID 会变化，节点角色与索引顺序保持 SV×5、RSU×2、UAV×2。
- 每个节点标签覆盖其完整合法 UPDATE 候选集合；训练集共有 440 个合法候选，诊断集 312 个。12 个 trace 均有效，合计 24 个有效节点标签，trace/label 拒绝均为 0。候选 mask 的 KEEP 索引 0 保持合法，但被排除在 UPDATE 监督索引之外。
- 每个状态的分支从同一决策前 trace 和外生输入重放，并复用该 trace 的 Micro 共同随机数。标签为“有限期 Macro SMDP 回报 + 冻结 Update 16 Critic bootstrap”。它们是相对于冻结后续策略/Critic 的条件估计，不是长期最优价值。
- 成功生成固定标签批次耗时 139.69 秒，其中分支重放计时合计 123.35 秒。训练与诊断标签分别保存在 `training_labels.npz`（SHA-256 `0dcf47b2c44fee99b33394dfb1aeeb802bc9f9731b7e18c6f2a8605001900e49`）和 `diagnostic_labels.npz`（SHA-256 `cf7a0289931ec86d9a4221732f725abd45e2a668bfa06d36aba1628308144a6e`）。训练特征形状为 observation `[16,400]`、node feature `[16,7]`、candidate feature `[16,57,200]`；诊断数组保留其自身候选 padding 宽度。
- 诊断标签只用于 Step 0/20/50/100 的指标计算，不进入损失、反向传播、早停或超参数选择。没有访问 Validation/Test，没有生成数据，也没有写入 `environment_data/`。

## 专项优化范围

现有 Cache Candidate 网络没有独立于操作组决策的打分头。实际参与条件候选概率的同一组模块为：

- 更新：`macro_cache_candidate_encoder.0.{weight,bias}`、`macro_cache_scorer.{0.weight,0.bias,2.weight,2.bias}`。
- 冻结：`macro_actor_encoder`（共享全局观测编码器）、Macro/Micro Critic、Micro Actor、`macro_termination_scorer`、Stage3 ΔQ 价值头及其他策略参数。
- Candidate loss 是唯一训练信号；没有 PPO loss、Trigger loss、ΔQ Huber loss、环境 rollout、Matching+KKT 或奖励计算。训练完成后 Trainer `update_count` 仍为 16。
- 优化器为新建的 Candidate-only Adam，学习率沿用 Update 16 Macro Actor optimizer 的 `0.0005`，betas `(0.9,0.999)`、epsilon `1e-8`、weight decay `0`、AMSGrad 关闭，梯度裁剪阈值沿用 `0.5`。重置 Adam moments，避免混入既有 PPO 梯度状态。Pairwise 系数 `0.01`、温度 `1.0`、平局容差 `0.01` 均来自冻结 Profile，未修改。
- 固定训练集完整 batch 执行恰好 100 个优化 step。优化与 Step 指标/快照写入合计计时 `0.737` 秒；不含反事实标签生成。
- 梯度方向检查中，16/16 个训练节点的最佳合法候选 score 梯度为负，梯度下降会提高其 score。Candidate encoder 的 2/2 个参数和 scorer 的 3/4 个参数收到有限、非零梯度；对应组梯度 L2 分别为 `4.44e-4` 和 `5.58e-4`。

Candidate scorer 同时为 KEEP 与 UPDATE 候选打分，而操作组概率由 `logsumexp(UPDATE logits)` 与 KEEP logit 共同计算。因此，即使只用 Candidate ranking loss，仍会改变 `P(UPDATE)`。本实验保留并量化了这一共享路径影响；没有额外操作组损失去抵消它。

## Step 0/20/50/100 总体指标

Pairwise accuracy 是先逐节点计算、再对节点取平均；候选对共享状态和标签，不能把候选对数量视为独立样本量。`P(best)` 是最佳合法 UPDATE 候选在 UPDATE 条件分布中的概率；`E[ΔQ]` 是该条件分布加权的冻结标签值。Top quartile mass 是按真实 ΔQ 取前四分之一候选后得到的条件概率质量。

### Training 样本（16 个节点状态）

| Step | Candidate loss | Pairwise accuracy | P(best) | 最佳候选平均排名 | 贪心候选 ΔQ | Regret | E[ΔQ] | Top quartile mass | 平均 P(UPDATE) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.69536 | 0.46361 | 0.06177 | 15.5625 | -0.22462 | 0.28375 | -0.18269 | 0.27748 | 0.51288 |
| 20 | 0.58738 | 0.86157 | 0.08649 | 1.6875 | 0.03848 | 0.02065 | -0.15926 | 0.35023 | 0.41318 |
| 50 | 0.27509 | 0.92753 | 0.33937 | 1.6250 | 0.03848 | 0.02065 | -0.03407 | 0.79179 | 0.17335 |
| 100 | 0.13051 | 0.95903 | 0.66404 | 1.1250 | 0.05524 | 0.00389 | 0.03198 | 0.97919 | 0.65494 |

Training 非平局候选比较数为 8,432。Step 100 相对 Step 0，16/16 个节点的期望 ΔQ 上升且 regret 下降。

### 独立诊断样本（8 个节点状态）

| Step | Candidate loss | Pairwise accuracy | P(best) | 最佳候选平均排名 | 贪心候选 ΔQ | Regret | E[ΔQ] | Top quartile mass | 平均 P(UPDATE) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.69397 | 0.52364 | 0.04140 | 18.5000 | -0.20851 | 0.27075 | -0.20609 | 0.26615 | 0.51716 |
| 20 | 0.59561 | 0.81492 | 0.06115 | 3.8750 | 0.02793 | 0.03431 | -0.18151 | 0.33819 | 0.40851 |
| 50 | 0.36375 | 0.86031 | 0.25459 | 3.3750 | 0.02793 | 0.03431 | -0.04788 | 0.77109 | 0.15306 |
| 100 | 0.34024 | 0.86977 | 0.34747 | 3.1250 | 0.04094 | 0.02130 | 0.00778 | 0.95432 | 0.55357 |

诊断非平局候选比较数为 7,079。Step 100 相对 Step 0，8/8 个独立节点的 Pairwise accuracy、最佳候选概率和期望 ΔQ 上升，最佳候选排名与 regret 下降。诊断集平均 E[ΔQ] 提高 `0.21388`，平均 regret 降低 `0.24946`。这只是 4 个 Episode、8 个节点状态上的固定标签诊断。

### 各节点 Step 0 → Step 100

下表保留每个训练及诊断节点的主要成对结果。Step 20/50 的完整逐节点指标（含高 ΔQ 概率质量、logit 变化与每节点 Trigger 概率）保存在 `supervised_learning_results.json`。

| 数据 | Seed / 节点 | Pairwise accuracy | P(best) | 最佳排名 | 贪心 ΔQ | Regret | E[ΔQ] | P(UPDATE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 20031 / SV 58 | 0.714→1.000 | 0.100→0.954 | 4→1 | -0.165→0.010 | 0.175→0.000 | -0.194→0.005 | 0.507→0.718 |
| Train | 20031 / RSU 0 | 0.266→0.878 | 0.017→0.312 | 48→1 | -0.598→-0.075 | 0.523→0.000 | -0.354→-0.134 | 0.521→0.055 |
| Train | 20032 / RSU 1 | 0.537→0.946 | 0.018→0.302 | 14→1 | -0.152→0.100 | 0.252→0.000 | -0.163→0.045 | 0.515→0.773 |
| Train | 20032 / UAV 0 | 0.395→0.893 | 0.029→0.812 | 28→1 | -0.487→0.008 | 0.496→0.000 | -0.309→-0.023 | 0.527→0.319 |
| Train | 20033 / UAV 1 | 0.517→0.946 | 0.029→0.520 | 24→1 | -0.012→0.077 | 0.089→0.000 | -0.155→0.050 | 0.527→0.845 |
| Train | 20033 / SV 28 | 0.455→1.000 | 0.101→0.697 | 5→1 | -0.085→0.094 | 0.179→0.000 | -0.074→0.075 | 0.507→0.897 |
| Train | 20034 / RSU 0 | 0.330→0.868 | 0.018→0.310 | 41→2 | -0.766→-0.037 | 0.768→0.039 | -0.315→-0.059 | 0.521→0.054 |
| Train | 20034 / RSU 1 | 0.601→0.953 | 0.018→0.173 | 4→2 | -0.056→0.099 | 0.178→0.023 | -0.119→0.077 | 0.515→0.775 |
| Train | 20035 / UAV 0 | 0.368→0.908 | 0.029→0.819 | 28→1 | -0.646→0.030 | 0.676→0.000 | -0.366→-0.004 | 0.527→0.324 |
| Train | 20035 / UAV 1 | 0.492→0.952 | 0.029→0.516 | 24→1 | -0.040→0.076 | 0.116→0.000 | -0.172→0.044 | 0.527→0.842 |
| Train | 20036 / SV 33 | 0.455→1.000 | 0.101→0.687 | 5→1 | -0.067→0.109 | 0.176→0.000 | -0.060→0.087 | 0.507→0.889 |
| Train | 20036 / SV 18 | 0.488→1.000 | 0.101→0.935 | 4→1 | -0.186→0.007 | 0.194→0.000 | -0.197→-0.000 | 0.500→0.731 |
| Train | 20037 / SV 4 | 0.372→1.000 | 0.099→0.969 | 5→1 | -0.043→0.135 | 0.178→0.000 | -0.093→0.130 | 0.498→0.801 |
| Train | 20037 / SV 34 | 0.545→1.000 | 0.100→0.724 | 6→1 | -0.065→0.107 | 0.172→0.000 | -0.072→0.087 | 0.511→0.961 |
| Train | 20038 / SV 8 | 0.512→1.000 | 0.101→0.932 | 4→1 | -0.185→0.007 | 0.192→0.000 | -0.188→0.000 | 0.500→0.725 |
| Train | 20038 / SV 45 | 0.372→1.000 | 0.099→0.963 | 5→1 | -0.041→0.136 | 0.177→0.000 | -0.091→0.130 | 0.498→0.771 |
| Diagnostic | 20039 / SV 34 | 0.714→1.000 | 0.100→0.954 | 4→1 | -0.136→0.037 | 0.173→0.000 | -0.170→0.032 | 0.507→0.731 |
| Diagnostic | 20039 / RSU 0 | 0.447→0.750 | 0.018→0.025 | 36→8 | -0.343→-0.005 | 0.418→0.081 | -0.157→-0.005 | 0.521→0.057 |
| Diagnostic | 20040 / RSU 1 | 0.588→0.894 | 0.018→0.171 | 4→2 | -0.048→0.050 | 0.122→0.025 | -0.145→0.029 | 0.515→0.765 |
| Diagnostic | 20040 / UAV 0 | 0.431→0.922 | 0.029→0.806 | 28→1 | -0.507→0.032 | 0.539→0.000 | -0.274→0.007 | 0.527→0.312 |
| Diagnostic | 20041 / UAV 1 | 0.426→0.839 | 0.029→0.513 | 24→1 | -0.202→0.073 | 0.275→0.000 | -0.370→-0.001 | 0.527→0.838 |
| Diagnostic | 20041 / SV 47 | 0.524→0.952 | 0.102→0.112 | 3→3 | -0.152→0.017 | 0.183→0.015 | -0.141→0.006 | 0.507→0.891 |
| Diagnostic | 20042 / RSU 0 | 0.424→0.707 | 0.017→0.026 | 45→7 | -0.292→0.026 | 0.343→0.024 | -0.209→-0.049 | 0.520→0.058 |
| Diagnostic | 20042 / RSU 1 | 0.635→0.894 | 0.018→0.173 | 4→2 | 0.011→0.098 | 0.113→0.026 | -0.182→0.044 | 0.515→0.777 |

## Trigger 漂移与预设门槛

`P(UPDATE)` 使用现有 MAPPO 操作组合同一计算：对合法 UPDATE logits 做 `logsumexp`，再与 KEEP logit 形成二元组分布。确定性操作选择按 `P(UPDATE) >= 0.5` 记为 UPDATE。

| 数据 | Step 0 平均 P(UPDATE) | Step 20 | Step 50 | Step 100 | 相对 Step 0 的操作边界翻转数（0/20/50/100） |
|---|---:|---:|---:|---:|---:|
| Training | 0.5129 | 0.4132 | 0.1733 | 0.6549 | 0/13/13/6（16 个节点） |
| 独立诊断 | 0.5172 | 0.4085 | 0.1531 | 0.5536 | 0/8/8/3（8 个节点） |

Step 100 的绝对概率变化平均值为 Training `0.3099`、诊断 `0.3217`；诊断单节点最大变化 `0.4637`，P(UPDATE) 范围为 `0.0567–0.8913`。Step 50 的全部 8 个诊断节点都从 Step 0 的 UPDATE 侧跨到 KEEP 侧；Step 100 仍有 3/8 节点保持翻转。诊断集 8/8 个 `best-ΔQ` 均大于平局容差 `0.01`，其中若干节点在 Step 100 的操作组概率仍低于 0.5。因而候选排名改善伴随了明显、未受 Trigger 目标约束的操作概率漂移。该漂移源于共用 scorer 的可观察结果；不把它断言为正式系统长期行为。

## 预设条件判定

| 条件 A 项目 | 结果 |
|---|---|
| Training 排序准确率 ≥70%，且比 Step 0 至少提高 0.15 | 通过：0.9590，提升 0.4954 |
| 独立诊断排序准确率比 Step 0 至少提高 0.05 | 通过：0.5236→0.8698，提升 0.3461 |
| 诊断平均条件期望 ΔQ 提高 | 通过：-0.2061→0.0078 |
| 诊断平均 regret 下降 | 通过：0.2708→0.0213 |
| 至少 5/8 个诊断状态在期望 ΔQ 或 regret 上改善 | 通过：8/8 两项均改善 |
| 无数值错误或动作非法 | 通过：100 step 全部有限；固定合法 mask 未改变，贪心候选始终从合法 UPDATE 集中选取；错误数为 0 |
| 无明显的非预期 Trigger 漂移 | **未通过**：最终 3/8 诊断、6/16 训练节点跨过 0.5 操作边界，且中间 Step 50 有 8/8 诊断节点翻转 |

因此不满足完整条件 A。也不属于 B（训练拟合并非仅限训练集），不属于 C（训练集可拟合）；按预设归为 **MIXED**：Candidate 条件价值学习与独立节点指标改善，但 Trigger 操作组行为被共享打分路径显著扰动。没有把候选对数量当作独立状态样本，也没有据此主张统计显著性。

## 快照与文件

- Step 0/20/50/100 Actor 快照保存在 `runs/research_mappo_stage3_candidate_supervised_learning/candidate_supervised_20261018T115258Z_f4fd13/candidate_supervised_checkpoints/`。Update 16 原 Checkpoint 未覆盖。Step 100 策略 state dict 以严格参数名/形状恢复成功，Candidate-only Adam 状态也成功载入（6 个参数状态条目）。
- 全部逐节点 Step 0/20/50/100 指标、候选概率与排序、Trigger 概率变化和参数记录：`runs/research_mappo_stage3_candidate_supervised_learning/candidate_supervised_20261018T115258Z_f4fd13/supervised_learning_results.json`，SHA-256 `e7d8e698c28b8b25ef3f3f6c1a7d085105ea1e520cb45c49b95edcc67849a3fa`。
- 每一步的训练 Candidate loss、有效 pair 数、梯度范数和耗时：同目录 `supervised_step_metrics.jsonl`。
- 固定样本来源及逐 Episode Manifest 哈希：同目录 `fixed_labels_metadata.json`，SHA-256 `65888218e90739c55c62fe01a5412052a6884c8255bfe5a29d23f37400621243`。
- Step 0/20/50/100 单独指标文件分别为 `metrics_step_000.json`、`metrics_step_020.json`、`metrics_step_050.json`、`metrics_step_100.json`；Step 0 的参数变化按基准定义为 0。快照 SHA-256 已记录在 `supervised_learning_results.json`。
- 本轮没有修改算法源码、正式配置、Profile、奖励或数据；没有调用 `Trainer.update()`，没有运行 MAPPO 训练、Validation 或完整 Episode。`environment_data/` 未写入、移动或删除；没有 commit/push。

## 后续唯一优先项

先解决 **Candidate 条件分布与 KEEP/UPDATE Trigger 组概率的参数耦合**：使 Candidate 专项排序更新能够改变合法 UPDATE 候选之间的相对概率，同时保持操作组概率不发生无监督漂移；再用同一批冻结标签复核候选排序和 Trigger 概率。该项仅为后续建议，本轮未改结构、未改正式 Trainer，也未整合专项更新。

```makefile
CANDIDATE_TRAIN_FIT = YES
CANDIDATE_HELDOUT_GENERALIZATION = YES
CANDIDATE_EXPECTED_DELTA_Q_IMPROVED = YES
CANDIDATE_REGRET_REDUCED = YES
SUPERVISED_LEARNING_RESULT = MIXED
INTEGRATED_INTO_FORMAL_MAPPO = NO
READY_FOR_CONTROLLED_MAPPO_VALIDATION = NO
```
