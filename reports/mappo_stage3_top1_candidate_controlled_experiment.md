# Stage3 Top-1 Candidate 受控对比实验

日期：2026-10-19  
结论范围：固定的 8 次更新（320 Training Slot）及三个冻结的 300-Slot Validation Episode；仅作机制筛选。

## 结果摘要

- v7 Best-vs-Rest Top-1 训练完成 8/8 次更新，共 320 Slot。每次更新得到 8 个有效反事实标签；总计 64 个，拒绝标签和 trace 均为 0。Candidate Actor 在 8 次更新中均有有限梯度和参数变化。
- 在 seed 30001、30002、30003 上，Top-1 的任务效用、Cache/UAV 成本、净效用及 Cache/UAV 行为指标均与已完成的 v6 All-Pairs 评价逐场景完全相同。逐 Slot 任务/净效用、Cache 成本与变更数、UAV 操作与成本、任务量、时延和能耗也逐项相同；运行时间字段因计时差异不同。
- Top-1 相对 All-Pairs 的三个净效用差均为 `0`，因此本实验未观察到 Top-1 带来的系统性能提升。
- Top-1 仍在三个场景全部低于 KEEP+STAY，平均净效用差为 `−54.1855`。本结果不支持宣称 Top-1 优于 All-Pairs 或 KEEP+STAY。

## 固定输入与可比性

| 项目 | 实际使用值 |
|---|---|
| 代码版本 | `2b23fb208b63f8117a6036516af0faef54f7dd16`；v6 对照运行记录版本为 `246e2a019a2ecf8bd4d1e317f378b26710a64382` |
| 初始化 Checkpoint | `runs/research_mappo_stage3_decoupled_controlled_evaluation/training/stage3_20261018T124359Z_ff9ab9d0/checkpoints/checkpoint_update_0024.pt` |
| 初始化 SHA-256 | `41e979bf193e13cd4f20f272dae5dbb77b86936ce0e61db56345ccc82fa87650` |
| 初始化方式 | 使用 `--initialize-from-stage3-checkpoint` 权重初始化；模型权重载入。优化器、训练进度与 RNG 均重置；不是严格续训。Trainer seed 为 `71301`。 |
| 训练合同 | 8 次更新，每次 40 Slot；Training seeds `20031–20038`；PPO epochs/update `1`；设备 `cpu`；每 rollout 最多 4 个 Macro 状态、每状态最多 2 个目标节点。 |
| Candidate 损失 | v6 `all_pairs_pairwise_logistic_v1` → v7 `best_vs_rest_top1_pairwise_logistic_v1`。除候选损失外，代码差异是对应的损失 Schema、Checkpoint 版本和权重初始化/元数据合同。冻结 Profile 未变。 |
| Loss Profile | `configs/research/mappo_stage3_loss_coefficients_v1.json`，规范化 SHA-256 `c91bccc6bf91f76c407a18c9953b4fa2c2d346fcfef3337ad63ee49b23d8bdb4` |
| 运行配置 SHA-256 | `03328b88658a6f15c0518eee9f4903d5647f12a800432e58c74b42fdef51d80c` |
| Training Pool Manifest SHA-256 | `b6cae236b029f4d24c7488a91f0825247a20f475ec76b2d80737de44ae491bd3` |
| Validation Pool Manifest SHA-256 | `a75f4935f598a2945f66547ab48ce5a24a86624bb142a9349787797c402da10a` |
| Validation Episode Manifest SHA-256 | 30001: `c41529ed61face7249694894195f5ed5b1c85d49a099e2536b3f78f45002468b`; 30002: `283ce0a9d91dfda4e086427e8df3cabbf92122853030354f3924c8c6708cd2e1`; 30003: `2b355260804eddeeb7bb5b9780b7ec669062eb2ddc6fed164706ca6b84834894` |

旧 All-Pairs 与 KEEP+STAY 结果来自 `runs/research_mappo_stage3_greedy_candidate_trigger_20261018/validation_paired_current_adapter/validation_results.json`（SHA-256 `1af70d49c5a26ccae65e0bf5267b17be8e6de30ef61f671afceeca318c949028`）。复用前确认了当前评价适配器 SHA 与旧结果相同（`97fc1cf1f86d7386256d37a22cb9be54daedfb5f6960981c6acd2191ce8ba101`）、Stage1C evaluator SHA 相同（`e0ce198871544723a0627069771878a121c3bda7ace29e228016c0b2383d5dd2`），且运行配置、Profile、Validation Manifest、种子、Episode 长度及确定性解码规则一致。本轮只评价新 Top-1 Checkpoint，没有重跑旧策略。

## 训练结果

- 输出目录：`runs/research_mappo_stage3_top1_candidate_controlled_experiment/training/stage3_20261018T162754Z_968b49aa/`
- Update 8 Checkpoint：`checkpoints/checkpoint_update_0008.pt`，SHA-256 `15e677c9846e027a29048a5c84e84481f2349448edf57537d625883636d190b6`
- 训练日志记录 8/8 更新完成；有效标签 `64`、无效标签/trace 拒绝 `0`、数值错误 `0`；所有更新参数有限，Cache/UAV 动作合法。
- Candidate 排序损失范围为 `0.6922–0.6993`（均值 `0.6956`）；Candidate Actor 梯度范数范围为 `0.001484–0.001731`，8 次更新均报告参数变化。
- 现有训练 `label_diagnostics` 给出 64 个训练状态标签的有限期 `best_delta_q` 与行为策略目标候选 `trigger_target_delta_q`。由此计算的 Top-1 regret 均值为 `0.5964`，64/64 超过 `0.01` 平局容差；42/64 个状态中，所选候选 ΔQ 为负而最佳合法候选 ΔQ 为正。该统计只覆盖训练 rollout 中被抽样并标注的节点，不是 Validation 全决策准确率，也不是完整系统因果收益。
- 训练采样耗时合计约 `87.020 s`，优化耗时合计约 `251.516 s`，其中反事实重放约 `245.042 s`；元数据创建至完成记录的墙钟时间约 `6 分 25 秒`。

## 冻结 Validation 结果

所有策略均采用确定性“先选合法操作组，再选组内合法候选”；KEEP+STAY 强制 KEEP/STAY。金额及效用单位沿用正式环境定义。

| Seed | Top-1 任务效用 | Cache 成本 | UAV 成本 | Top-1 净效用 | Cache 实际变更率 | UAV 实际移动率 | Δ净效用 vs All-Pairs | Δ净效用 vs KEEP+STAY |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30001 | 3951.5666 | 30.2699 | 0 | 3921.2967 | 22.44% | 0% | 0.0000 | −54.8417 |
| 30002 | 3866.0277 | 30.2699 | 0 | 3835.7578 | 22.44% | 0% | 0.0000 | −53.4525 |
| 30003 | 3918.8993 | 30.2699 | 0 | 3888.6294 | 22.44% | 0% | 0.0000 | −54.2624 |
| **均值** | **3912.1645** | **30.2699** | **0** | **3881.8946** | **22.44%** | **0%** | **0.0000** | **−54.1855** |

Top-1 相对 KEEP+STAY 的平均任务效用差为 `−23.9156`，平均 Cache 成本增加 `30.2699`，UAV 成本差为 `0`。净效用差的样本标准差为 `0.6978`；3/3 场景均低于 KEEP+STAY。三个场景只是小样本筛选，不支持显著性或泛化结论。

本轮未从验证结果生成新的 Candidate regret/选择准确率：现有 Top-1 验证指标不包含该统计，未为补齐它而重放或枚举 Validation 反事实分支。上文仅报告已有训练标签可直接计算的有限期 regret。

## 决定与产物

**是否继续采用 Top-1 作为已证实的性能改进：否。** 三个冻结场景中它与 All-Pairs 完全持平，且仍低于 KEEP+STAY；本轮没有继续训练或第二轮实验。结果不证明 Top-1 损失本身无效，但没有提供选择它优于 All-Pairs 的系统性能证据。本轮没有回退或修改实现。

正式产物：

- Training 元数据、指标和 8 个更新 Checkpoint：`runs/research_mappo_stage3_top1_candidate_controlled_experiment/training/stage3_20261018T162754Z_968b49aa/`
- Top-1 三 Episode 评价结果：`runs/research_mappo_stage3_top1_candidate_controlled_experiment/validation_top1_policy_only/validation_results.json`
- 配对差异：`runs/research_mappo_stage3_top1_candidate_controlled_experiment/validation_top1_policy_only/paired_validation_comparison.json`
- 评价合同元数据：`runs/research_mappo_stage3_top1_candidate_controlled_experiment/validation_top1_policy_only/evaluation_metadata.json`

未访问 Test；未修改源码、正式配置、Loss Profile、数据或 Manifest；未 commit/push。
