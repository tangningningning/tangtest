# Stage3 主线筛选实验结果

日期：2026-10-18（Asia/Shanghai）  
判定范围：3 次 Stage3 更新（120 Training Slot）及三个预先冻结 Validation Episode。场景曾参与历史 PPO Validation，故结果是探索性筛选，不是独立确认性实验。

## 冻结输入与运行产物

- 初始策略：Stage1D Update 13 MAPPO v2，主项目副本 `runs/research_mappo_stage3_main_experiment/source/checkpoint_update_0013.pt`。来源与副本 SHA-256 均为 `aff1ed777c1100c9da69b86b079c0bcaa83e2bcdff1885769c9761e4858cd044`。Stage3 Runner 实际严格加载；元数据记载 schema `mappo_shared_actor_global_critic_fixed_macro_2_v2_group_balanced_ops`、Update 13、`v2_initialization_migrated=true`，观测/动作候选及概率合同匹配。没有从随机初始化开始。
- 损失系数 Profile SHA-256：`c91bccc6bf91f76c407a18c9953b4fa2c2d346fcfef3337ad63ee49b23d8bdb4`。本轮未调整系数、网络、奖励、预测器或概率合同。
- 正式运行配置 SHA-256：`03328b88658a6f15c0518eee9f4903d5647f12a800432e58c74b42fdef51d80c`。Training Pool Manifest SHA-256：`b6cae236b029f4d24c7488a91f0825247a20f475ec76b2d80737de44ae491bd3`。Validation Pool Manifest SHA-256：`a75f4935f598a2945f66547ab48ce5a24a86624bb142a9349787797c402da10a`。Episode SHA、文件集合 SHA、核心文件 SHA 及 300 Slot/Integrity Baseline 核验见 [实验计划与输入冻结报告](mappo_stage3_main_experiment_plan.md)。
- Training 使用 Stage2 A/B 同一顺序 `20017–20019`，每 Episode 采样 Slot `0–39` 与 bootstrap，trainer seed `71301`，每次一个 PPO epoch。Training 与 Validation seeds 不重叠；Test 只通过 Manifest 元数据核对 seed 范围 `40001–40100`，未读取 Test Episode 文件。
- Stage3 Run：`runs/research_mappo_stage3_main_experiment/training/stage3_20261018T095817Z_1b3d4652/`。Update 14、15、16 Checkpoint 均保留；最终 Update 16 SHA-256：`ca819d60320b3c65a46c2eb8217ca6d9e18f5a4bda2799f1778988f3976e611f`。既有 Stage3 Update 14 冒烟 Checkpoint 在其原目录保留，未覆盖。
- 最终配对评价结果：`runs/research_mappo_stage3_main_experiment/validation_3seeds_20261018_retry1/validation_results.json`。评价元数据记录 `validation_accessed=true`、`test_accessed=false`、`trainer_update_called=false`。
- 本轮主项目冻结评价适配器 SHA-256：`8c596f481f6f8cc76e782b7af2df506c476f37cc012686373e451e93fcd85d94`；最终结果 JSON 记录了同一适配器哈希。

## Training 结果

| Checkpoint | seed | Slot | 净采样回报 | Cache 成本 | UAV 移动成本 | Cache UPDATE | UAV MOVE | 有效/拒绝标签 | ΔQ RMSE（训练标签） | ΔQ Huber / Trigger 排序 / Candidate 排序 | Macro KL / clip | Micro KL / clip | Matching+KKT | 收集 / 更新耗时 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|
| 14 | 20017 | 40 | 518.938 | 10.0252 | 0.08250 | 92/180 (51.11%) | 33/80 (41.25%) | 8/0 | 0.5170 | 0.009884 / 0.686730 / 0.696496 | `2.862e-5` / 0 | `1.927e-5` / 0 | 5.637 s | 10.228 / 10.638 s |
| 15 | 20018 | 40 | 526.048 | 8.9198 | 0.06600 | 86/180 (47.78%) | 30/80 (37.50%) | 8/0 | 0.2958 | 0.109120 / 0.680846 / 0.694908 | `6.699e-5` / 0 | `7.823e-7` / 0 | 2.716 s | 7.286 / 40.179 s |
| 16 | 20019 | 40 | 520.106 | 9.0056 | 0.08316 | 88/180 (48.89%) | 36/80 (45.00%) | 8/0 | 0.2094 | 0.112263 / 0.680440 / 0.693178 | `2.197e-5` / 0 | `1.390e-6` / 0 | 1.212 s | 5.721 / 35.108 s |

每次更新的 4 个反事实 traces 均产生 8 个有限合法标签；三次共 24 个标签，拒绝为 0。Actor、Critic 与 ΔQ 价值头每次均实际更新，所有更新参数有限，数值异常为 0。PPO ratio 均值接近 1；Macro 观测范围约 `0.9773–1.0216`，Micro 约 `0.9914–1.0089`；三次 clip fraction 均为 0。训练 Candidate 排序损失仍约 `0.693–0.696`；训练 ΔQ RMSE 从 `0.5170` 降至 `0.2094`，只代表所采 Training 标签拟合，不代表 Validation 泛化。每次只有 8 个 Trigger 比较；没有可用的 Validation 排名指标。

实际 Training 120 Slot 的收集时间合计 `23.235 s`、更新时间合计 `85.924 s`；Runner 墙钟约 `123 s`。更新耗时主要包含准确反事实分支重放（各次 replay `9.993 / 39.599 / 34.529 s`）。没有改变更新预算或追加训练。

## 三策略完整 Validation Episode 结果

确定性策略按现有合同解码：先在合法操作组中贪心选组，再在该组内贪心选合法条件候选；KEEP+STAY 强制 Cache action 0 与 UAV action 0。每个场景三个策略均运行同一份正式 Validation realization、同一预测/实际任务输入、同一正式环境和奖励合同，各 300 Slot。

### 净效用与配对差

| Validation seed | Update 13 | Stage3 Update 16 | KEEP+STAY | Stage3 − Update 13 | Stage3 − KEEP+STAY |
|---:|---:|---:|---:|---:|---:|
| 30001 | 3931.2945 | 3905.2163 | 3976.1384 | -26.0781 | -70.9220 |
| 30002 | 3852.5230 | 3827.3487 | 3889.2103 | -25.1743 | -61.8616 |
| 30003 | 3907.8003 | 3881.4224 | 3942.8919 | -26.3780 | -61.4695 |
| **平均** | **3897.2059** | **3871.3291** | **3936.0802** | **-25.8768** | **-64.7510** |

Stage3 相对 Update 13 的正向场景为 `0/3`，未达到“平均差为正且至少两个场景为正”的门槛；相对 KEEP+STAY 也在 `0/3` 场景为正。

### 收益、成本与行为

| seed | 策略 | 任务效用 | Cache 更新成本 | UAV 移动成本 | Cache 实际变更 | UAV 实际位移 |
|---:|---|---:|---:|---:|---:|---:|
| 30001 | Update 13 | 3948.2404 | 16.9460 | 0 | 156/1350 (11.56%) | 0/600 |
| 30001 | Stage3 | 3941.4372 | 36.2209 | 0 | 314/1350 (23.26%) | 0/600 |
| 30001 | KEEP+STAY | 3976.1384 | 0 | 0 | 0/1350 | 0/600 |
| 30002 | Update 13 | 3869.4689 | 16.9460 | 0 | 156/1350 (11.56%) | 0/600 |
| 30002 | Stage3 | 3863.7862 | 36.4375 | 0 | 316/1350 (23.41%) | 0/600 |
| 30002 | KEEP+STAY | 3889.2103 | 0 | 0 | 0/1350 | 0/600 |
| 30003 | Update 13 | 3924.7463 | 16.9460 | 0 | 156/1350 (11.56%) | 0/600 |
| 30003 | Stage3 | 3919.5933 | 38.1710 | 0 | 332/1350 (24.59%) | 0/600 |
| 30003 | KEEP+STAY | 3942.8919 | 0 | 0 | 0/1350 | 0/600 |

三个场景中，Stage3 平均任务效用比 Update 13 低 `5.880`，平均 Cache 成本高 `19.997`；Stage3 共发生 `962` 次实际 Cache 变更，Update 13 为 `468` 次。两策略的 UAV 都没有移动。因此本轮性能下降同时包含略低的任务效用和明显更高的 Cache 重配置成本；不能据此把退化单因归咎于 Critic。

全部九个最终 Episode 均为 300 Slot，Cache/UAV 动作合法，实际 Cache 变更数与 UPDATE 操作数一致，实际 UAV 位移数与 MOVE 操作数一致；奖励合同最大误差为 0，数值异常为 0，约束违规为 0。

### 评价耗时

| 策略 | 三 Episode rollout 秒数 | 平均每 Episode | 900 Slot 端到端单 Slot均值 / P95 |
|---|---:|---:|---:|
| Update 13 | 152.830 s | 50.943 s | 0.1698 / 0.3409 s |
| Stage3 | 138.371 s | 46.124 s | 0.1537 / 0.3009 s |
| KEEP+STAY | 131.474 s | 43.825 s | 0.1461 / 0.2958 s |

九个完整 Episode 的逐策略总执行墙钟约 `8 min 40 s`（包括准备与 Checkpoint 加载）；表内 rollout 秒数为逐策略计时器结果。运行开销已记录，继续门槛则因性能差异失败。

## ΔQ Validation 诊断及评价器问题

训练阶段 24 个真实反事实标签有效且参与了 ΔQ Huber/排序损失。但三份 Stage3 Validation rollout 的 ΔQ 诊断均为 **UNRESOLVED**：每 Episode 的 4 条 trace、8 个目标标签都因

```text
TypeError: FrozenEvaluationTrainer._sample_micro_head() got an unexpected keyword argument 'random_uniforms'
```

而被 Stage3 严格拒绝；因此 Validation `branch_execution_count=0`，ΔQ RMSE、符号准确率和候选排序准确率没有计算。该问题是继承的 Stage1C evaluator override 未跟上 Stage3 Micro common-uniform 回放接口，不是有限性检查通过的 ΔQ 标签，也不是“无合法候选”的证据。评价器将其保留为拒绝原因，没有使用零标签。为避免在已确定性能门槛失败后继续重复完整 Episode，本轮不再重跑价值诊断。后续若要单独研究验证集价值泛化，应先最小修复该 override 并单独冻结诊断方案。

## 首次失败尝试、修改与项目副本清理

- 首次调用评估适配器时，旧 `run_mappo_stage3_cache_value.py` 的 `_load_stage1b_helpers` 已从 Stage3-only Runner 移除；适配器在读取 Episode 前报 `AttributeError`。修复为直接从主项目加载 `mappo_stage3_pool_contract.py`。
- 随后的首次 seed 30001 评价已跑过 Update 13 与 Stage3 两个策略，但在 Stage3 价值诊断错误时中止，未落盘策略结果。适配器随后增加逐策略落盘，并将无效 Validation 标签诊断如实标记 unresolved。上表与判定只使用最终完整 `retry1/validation_results.json`，没有混合前次结果。
- 本轮主项目新增/修改的评价代码仅为 `scripts/evaluation/run_mappo_stage3_main_experiment.py`（从 Stage3 参考目录导入后按主项目 helper 接线、逐策略保存结果并记录诊断拒绝）；基础 Update 13 Checkpoint 为 SHA 校验后复制。Stage3 Actor/Critic/Trainer、正式配置、奖励、预测器、Pool 与原始 Manifest 均未修改。
- 检查到的 Stage1D、Stage2-spatiotemporal、Stage2-parallel、Stage3 参考目录都是注册 Git Worktree，且都存在跟踪修改或未跟踪报告/实现；Stage3 目录还含 `environment_data/`。这些包含唯一研究成果，故本轮未删除、移动或清理任何参考 Worktree。`environment_data/` 未修改。
- 没有训练超出 3×40 Slot、没有 Test 访问、没有参数搜索、没有 commit/push。Python 使用 `-B`；未生成临时脚本或新的 Python 缓存。

## 预设继续门槛与结论

```text
STAGE3_TRAINING_COMPLETED = YES
VALIDATION_EPISODES_FROZEN = YES
PAIRED_EVALUATION_COMPLETED = YES
STAGE3_BEATS_FROZEN_UPDATE13 = NO
STAGE3_BEATS_KEEP_STAY = NO
READY_FOR_NEXT_STAGE = NO
```

建议停止扩大 Stage3 训练。此筛选中 Stage3 的 Cache 更新率约为 Update 13 的两倍，净效用在三个场景都更低；Stage3 训练更新与标签生成本身有效，但 Validation ΔQ 排序无法核验。当前没有进入下一阶段的性能证据。若后续继续研究，优先单独修复并验证 Stage1C/Stage3 反事实 Micro uniform 接口，同时把 Cache 更新行为与额外成本作为首要诊断项；不据此直接改系数或算法。
