# Greedy-Candidate Trigger 受控实验

日期：2026-10-19（Asia/Shanghai）  
代码版本：`origin/main` / `246e2a019a2ecf8bd4d1e317f378b26710a64382`  
结论范围：8 次更新、320 个 Training Slot，以及 3 个冻结 Validation Episode。Validation 结果是小样本筛选，不代表统计显著或泛化结论。

## 结果摘要

- Greedy-Candidate Trigger 从 Update 24 权重初始化后完成 8/8 次更新；每次 40 Slot，共 320 Slot。8 次更新共产生 64 个有效反事实标签，没有无效标签、非法 Trace、数值异常或 Cache/UAV 合法性问题。
- 在 30001–30003 三个完整 300-Slot Episode 上，新策略均优于当前执行器下重跑的 Update 24：平均配对净效用增加 **44.1723**，3/3 场景为正。
- 新策略仍在 3/3 场景低于 KEEP+STAY，平均差 **−54.1855**。因此目前是相对 Update 24 的初步正向证据，尚无超过固定基线的系统级收益证据。
- 建议保留为研究候选，不将其认定为最终有效策略；本轮结束后不追加训练。

## 输入与复现合同

### 代码、Checkpoint 与初始化方式

- 训练入口：`scripts/training/run_mappo_stage3_cache_value.py`。
- 初始化 Checkpoint：`runs/research_mappo_stage3_decoupled_controlled_evaluation/training/stage3_20261018T124359Z_ff9ab9d0/checkpoints/checkpoint_update_0024.pt`。
- Checkpoint SHA-256：`41e979bf193e13cd4f20f272dae5dbb77b86936ce0e61db56345ccc82fa87650`；Schema v5；源 Trigger target schema 未记录。
- 通过现有 `--initialize-from-stage3-checkpoint` 合同进行**权重初始化**，不是严格续训。当前载入实现严格载入模型 `policy_state_dict`（含 Actor/Critic 参数）及 ΔQ value head；候选策略参数无需迁移。优化器状态、更新/Stage3 进度和 RNG 状态均未恢复；以 Trainer seed `71301` 建立新随机状态。目标 Trigger schema 为 `deterministic_update_candidate_delta_q_v1`。
- 本轮 Run：`runs/research_mappo_stage3_greedy_candidate_trigger_20261018/training/stage3_20261018T153035Z_5a99539b/`。
- 最终 Checkpoint：`checkpoints/checkpoint_update_0008.pt`，SHA-256 `ca503f1769f88aa792d4904169c37f03d3ed5f29445dbf121f27e2c36fa03f83`。这是“从 Update 24 权重初始化后的第 8 次更新”，不称为严格续训的 Update 32。
- 训练指标和完成状态：`training_metrics.jsonl`、`run_completion.json`；启动命令输出：`runs/research_mappo_stage3_greedy_candidate_trigger_20261018/training_console.log`。

### Profile、正式配置与数据

- 冻结 Profile：`configs/research/mappo_stage3_loss_coefficients_v1.json`。规范化换行后的 SHA-256 为 `c91bccc6bf91f76c407a18c9953b4fa2c2d346fcfef3337ad63ee49b23d8bdb4`；原始 CRLF 文件 SHA-256 为 `422f67bc8e27a4958450f682b434bf6d00a55709171d5715ec0b345b33b62c8b`。系数及容差未修改。
- 运行配置 SHA-256：`03328b88658a6f15c0518eee9f4903d5647f12a800432e58c74b42fdef51d80c`；数据生成 fingerprint：`4083b917948cd7203972c0edd5b40d2efefd8984dde27089a77da43d460b32f3`。
- Training Pool Manifest：`environment_data/proposed_joint_training/formal_pool_manifest.json`，SHA-256 `b6cae236b029f4d24c7488a91f0825247a20f475ec76b2d80737de44ae491bd3`。沿用此前期望 ΔQ Trigger 实验的固定顺序和场景，seed `20031–20038`。下表哈希为各已转换 Episode `episode_manifest.json` 的 SHA-256：

| Training seed | Episode Manifest SHA-256 |
|---:|---|
| 20031 | `3bf5b701b7c79f8839aefe72553754ee614ec1f51f2e6f470ab1c924d02fd62b` |
| 20032 | `e4fbf5453c5d5496ef56441cd000ac35396d1d62f0e0b7a6f41c74bdecf7f82d` |
| 20033 | `a8219b6762ccc280a93c5e9096433866783a15c9ed341ae655a87a9e7aebbff8` |
| 20034 | `df5124bec996b624fd381a2c822ee8a6f31728cf9a7c7208ab29f86013f6df93` |
| 20035 | `a47c8d1337badff313afccb6a86c60004a7c8755bdb4a2e8153b4dcdfce2a982` |
| 20036 | `746e89233deace2654ae64ed984ffe5f5fb4b7806a886c2cc27bdecb753bf331` |
| 20037 | `0a52e51f2cde1d29ae040872c9b6081cf8b102f38e87abe72b4b48c4ced136c0` |
| 20038 | `4005cf5d64056f4feff84c0ee1cce6117878d86e245d47952f2d9ff613c49aed` |

- Validation Pool Manifest：`environment_data/proposed_joint_validation/formal_pool_manifest.json`，SHA-256 `a75f4935f598a2945f66547ab48ce5a24a86624bb142a9349787797c402da10a`。按冻结 seed 顺序评价 `30001–30003`，每个 300 Slot：

| Validation seed | Episode Manifest SHA-256 |
|---:|---|
| 30001 | `c41529ed61face7249694894195f5ed5b1c85d49a099e2536b3f78f45002468b` |
| 30002 | `283ce0a9d91dfda4e086427e8df3cabbf92122853030354f3924c8c6708cd2e1` |
| 30003 | `2b355260804eddeeb7bb5b9780b7ec669062eb2ddc6fed164706ca6b84834894` |

## Training 结果

每行是 40-Slot 训练窗口累计实际净效用，不是完整 Episode 评价。Counterfactual 标签数为有效标签数；每次 Trigger 辅助监督行数也为 8，拒绝标签/Trace 均为 0。

| 更新 | Training seed | 净效用 | 有效标签 / Trigger 行 | Cache 变更率 | UAV 移动率 | 采样+优化耗时（秒） |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20031 | 533.0319 | 8 / 8 | 49.44% | 32.50% | 33.645 |
| 2 | 20032 | 529.5282 | 8 / 8 | 50.56% | 47.50% | 43.508 |
| 3 | 20033 | 508.7994 | 8 / 8 | 42.78% | 41.25% | 63.161 |
| 4 | 20034 | 516.4742 | 8 / 8 | 53.33% | 56.25% | 73.719 |
| 5 | 20035 | 507.2647 | 8 / 8 | 46.11% | 47.50% | 23.631 |
| 6 | 20036 | 491.6376 | 8 / 8 | 51.67% | 51.25% | 24.315 |
| 7 | 20037 | 504.0598 | 8 / 8 | 52.22% | 43.75% | 65.772 |
| 8 | 20038 | 479.4447 | 8 / 8 | 52.22% | 42.50% | 48.499 |

采样阶段合计 `78.758 s`，优化阶段合计 `297.492 s`，合计 `376.250 s`；日志记录反事实重放耗时合计 `290.361 s`。8 次更新均报告参数有限、Cache/UAV 动作合法、数值错误为 0。

## 冻结 Validation 配对结果

策略均通过当前 `origin/main` 的 Stage1C 执行器，在同一 seed 的同一上下文中顺序评价。确定性学习策略使用“先选合法操作组、再选组内合法候选”的条件贪心；KEEP+STAY 强制 Cache action 0 和 UAV action 0。净效用恒等式为任务效用减 Cache 成本及 UAV 成本。

历史 Update 24 结果的适配器 SHA 与当前适配器 SHA 不同，因此本轮没有直接拼接历史指标；Update 24 与 KEEP+STAY 均在当前执行器下重新评价。Update 24 通过 v5 权重初始化合同加载（策略/价值权重载入，优化器、进度和 RNG 重置）。

KEEP+STAY 旧强制动作适配器缺少当前 Collector 必需的 deterministic_update_candidate_indices Trace 元数据。本轮仅在临时评价适配器中用当前正式确定性合法候选解码器计算并补入该索引，验证其属于合法 UPDATE 候选；实际动作仍强制 KEEP/STAY，动作概率日志仍为 0，未启用 Stage3 反事实采集或训练。该兼容层未写入项目源码，临时脚本已删除。

| Episode | 策略 | 任务效用 | Cache 成本 | UAV 成本 | 净效用 | 实际 Cache 变更 | 实际 UAV 移动 | 运行秒数 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 30001 | Update 24 | 3940.5389 | 65.4308 | 0 | 3875.1081 | 696/1350 (51.56%) | 0/600 (0%) | 46.300 |
| 30001 | Greedy-Candidate（8次更新） | 3951.5666 | 30.2699 | 0 | 3921.2967 | 303/1350 (22.44%) | 0/600 (0%) | 53.306 |
| 30001 | KEEP+STAY | 3976.1384 | 0 | 0 | 3976.1384 | 0/1350 (0%) | 0/600 (0%) | 48.977 |
| 30002 | Update 24 | 3864.3449 | 68.7593 | 0 | 3795.5856 | 722/1350 (53.48%) | 0/600 (0%) | 55.301 |
| 30002 | Greedy-Candidate（8次更新） | 3866.0277 | 30.2699 | 0 | 3835.7578 | 303/1350 (22.44%) | 0/600 (0%) | 62.714 |
| 30002 | KEEP+STAY | 3889.2103 | 0 | 0 | 3889.2103 | 0/1350 (0%) | 0/600 (0%) | 62.278 |
| 30003 | Update 24 | 3911.4468 | 68.9733 | 0 | 3842.4735 | 722/1350 (53.48%) | 0/600 (0%) | 45.219 |
| 30003 | Greedy-Candidate（8次更新） | 3918.8993 | 30.2699 | 0 | 3888.6294 | 303/1350 (22.44%) | 0/600 (0%) | 55.337 |
| 30003 | KEEP+STAY | 3942.8919 | 0 | 0 | 3942.8919 | 0/1350 (0%) | 0/600 (0%) | 57.667 |

| Episode | Greedy-Candidate − Update 24 | Greedy-Candidate − KEEP+STAY |
|---:|---:|---:|
| 30001 | +46.1886 | −54.8417 |
| 30002 | +40.1722 | −53.4525 |
| 30003 | +46.1560 | −54.2624 |
| **均值** | **+44.1723** | **−54.1855** |
| **样本标准差** | **3.4642** | **0.6978** |
| **正向场景数** | **3/3** | **0/3** |

相对 Update 24，Greedy-Candidate 平均任务效用增加 `6.7210`，平均 Cache 成本减少 `37.4512`，UAV 成本不变（均为 0），因此净效用平均增加 `44.1723`。实际 Cache 变更率由 Update 24 的平均 `52.84%` 降至 `22.44%`。这三个场景中净效用改善主要伴随 Cache 成本下降，而不是任务效用大幅增加。

相对 KEEP+STAY，Greedy-Candidate 平均任务效用少 `23.9156`，并产生平均 `30.2699` Cache 成本，净效用平均低 `54.1855`。因此不能宣称已经胜过固定策略。

完整结果、逐 Slot 指标、配对差、哈希与执行元数据保存在：

- `runs/research_mappo_stage3_greedy_candidate_trigger_20261018/validation_paired_current_adapter/validation_results.json`
- `runs/research_mappo_stage3_greedy_candidate_trigger_20261018/validation_paired_current_adapter/paired_validation_comparison.json`
- `runs/research_mappo_stage3_greedy_candidate_trigger_20261018/validation_paired_current_adapter/evaluation_metadata.json`
- `runs/research_mappo_stage3_greedy_candidate_trigger_20261018/validation_console.log`

所有 9 个 Episode 运行均为完整 300 Slot；数值异常、约束违规和动作非法均为 0，奖励分解误差检查通过。Test 未访问。

## 决策

`READY_FOR_NEXT_EXPERIMENT = NO`（本轮按要求停止，不自动开启后续实验）。建议保留 Greedy-Candidate Trigger 作为研究候选：相对 Update 24 在这 3 个场景均有净效用改善，但仍全部低于 KEEP+STAY。该结果属于探索性筛选，不能据此声称普遍优越或统计显著。

本轮未修改正式代码、配置、数据、Manifest 或 Loss Profile；未运行 Test、未 commit/push。临时评价脚本已清理。
