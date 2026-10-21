# Stage3 主线筛选实验计划与输入冻结

日期：2026-10-18（Asia/Shanghai）  
状态：输入已冻结；3 次 Stage3 更新及 3 场景配对 Validation 已完成。实际结果见 [`mappo_stage3_main_experiment_results.md`](mappo_stage3_main_experiment_results.md)。

## 实验预算与判定规则

- 从 Stage1D Update 13 MAPPO v2 初始化；Stage3 运行时只载入一次源 Checkpoint，随后严格恢复 Stage3 Checkpoint。
- Stage3 共 3 次更新，每次 40 个连续 Training Slot，总计 120 Slot；每次 1 PPO epoch，Training episode seed 顺序为 `20017, 20018, 20019`，trainer seed 为 `71301`。Training 只用前 40 个 Slot 和一个 bootstrap 输入，不构成完整 Episode。
- 仅当每次更新有有效反事实标签、更新参数有限且动作/奖励合同合法时继续到下一次更新。若任一更新失败，停止，不自动重试或扩大预算。
- 训练完成后，在冻结的 Validation seeds `30001, 30002, 30003` 上各评价一次三种策略：Update 13、最终 Stage3 Checkpoint、KEEP+STAY。主评价使用现有 Stage1C 确定性动作合同（先对合法操作组取最大概率组，再在组内对合法候选取条件最大概率），KEEP+STAY 强制保留 Cache 与 UAV 位置。场景/种子固定，不按中间回报替换。
- 继续条件预先固定为：Stage3 相对 Update 13 的三场景平均净效用差为正、至少两个场景配对差为正、无数值异常与约束违规、ΔQ 标签/排序诊断可解释且有效、端到端运行成本可接受。另单独检查 Stage3 是否高于 KEEP+STAY；若没有，不报告系统级优势。结果是探索性筛选，不是独立确认性证据。

## 模型、Profile 与运行配置

| 项目 | 冻结输入 |
|---|---|
| Stage1D 初始 Checkpoint | `runs/research_mappo_stage3_main_experiment/source/checkpoint_update_0013.pt` |
| 源 Checkpoint SHA-256 | `aff1ed777c1100c9da69b86b079c0bcaa83e2bcdff1885769c9761e4858cd044` |
| 模型合同 | `mappo_shared_actor_global_critic_fixed_macro_2_v2_group_balanced_ops`；400 维观测、9 个 Macro 节点、200 维 Macro 候选、2 个 41 维 Micro head；Stage3 Runner 使用显式 v2→Stage3 初始化迁移，并验证名称/shape/合同 |
| Stage3 Profile | `configs/research/mappo_stage3_loss_coefficients_v1.json`；SHA-256 `c91bccc6bf91f76c407a18c9953b4fa2c2d346fcfef3337ad63ee49b23d8bdb4`；Trigger/Candidate 系数 `0.01/0.01`、ΔQ Huber `1.0`、平局容差 `0.01/0.01`、pairwise temperature `1.0`、Huber δ `1.0`、价值头学习率 `0.001` |
| 正式运行配置 SHA-256 | `03328b88658a6f15c0518eee9f4903d5647f12a800432e58c74b42fdef51d80c` |
| Training Pool Manifest SHA-256 | `environment_data/proposed_joint_training/formal_pool_manifest.json`：`b6cae236b029f4d24c7488a91f0825247a20f475ec76b2d80737de44ae491bd3` |
| Validation Pool Manifest SHA-256 | `environment_data/proposed_joint_validation/formal_pool_manifest.json`：`a75f4935f598a2945f66547ab48ce5a24a86624bb142a9349787797c402da10a` |
| Test 隔离核对 | 只读 Test Pool Manifest 元数据：split `test`，seed 范围 `40001–40100`，100 个 300-Slot Episode；不读 Test Episode 文件 |

Update 13 文件此前可读且 SHA 已核验；本项目内副本与来源文件逐字节一致。Stage3 Run Metadata 将再次记录实际加载结果与来源哈希；若严格加载失败，不会从头训练或忽略不匹配张量。Stage3 Update 14 冒烟 Checkpoint 作为历史验证产物保留，不覆盖。

## Training Episode 冻结

seed 选择复用 Stage2 A/B 实际使用的 `20017–20019`，其来源是 Stage2 A/B 报告 `mappo_stage2_spatiotemporal_ab.md`。当前 Training Pool Manifest 及每个 Episode 的 Integrity Baseline 均与这三个实际文件匹配；episode_manifest 表示每个 realization 为完整 300 Slot。训练顺序与 Stage2 报告相同，但本轮只采样每个 Episode 的 Slot `0–39` 和 Slot 40 bootstrap。

episode 数据集合 SHA-256 定义为：按 Integrity Baseline 中 `relative_path` 升序，对每个文件拼接 `relative_path + TAB + sha256 + LF`，再对 UTF-8 字节串求 SHA-256。Pool Manifest SHA 与逐文件 Integrity Baseline 不会被本实验改写。

| seed | episode_manifest SHA-256 | 完整 Episode 文件集合 SHA-256 | 实际校验 |
|---:|---|---|---|
| 20017 | `b054a93cd5484920a88f8aff9c360b4144acd38f1a44351c099be34ade77c545` | `529ad9d7f64ec595990e8888bdb9a90db8ec214159d883797e04762ad9bc714f` | 10/10 文件哈希/大小通过；300 Slot |
| 20018 | `c0793f0e71be947e2f9e9e9c95aae32ac5a8fcd75c4862ad91e8750c869e933a` | `75c4cba5117b727e7daa27af18df1bf31397954ef3257d57c68d52e6f5a938f7` | 10/10 文件哈希/大小通过；300 Slot |
| 20019 | `05a2d6c0befb78d8fe5759a2471ded97ce41515fb9559f973f890936eae2652b` | `b6077aa257205431e9c8975403812241153a7626647eb6e355b44f0e45d5aba0` | 10/10 文件哈希/大小通过；300 Slot |

## Validation Episode 冻结

按固定 seed 升序，从正式 Validation 范围 `30001–30030` 选前 3 个，不参考策略回报。三个 Episode 的 seed 与 Training `20017–20019`、Test `40001–40100` 范围互斥；Test 隔离依据只来自 Test Manifest 的 seed/元数据。实际文件与 Validation Integrity Baseline 的路径、SHA-256、文件大小均匹配；每个 Episode 300 Slot，episode_manifest SHA 与 Pool Manifest 一致。

三个 seed 已出现在既有确定性 PPO Validation 范围 `30001–30030` 中，且 seed `30001–30004` 曾用于早期配对检查；这属于允许的历史场景复用，故本轮只能称为探索性筛选，不能称为独立确认性实验。Stage2 A/B 的 MAPPO 对照本身使用 seed `30015`，不与本轮重叠。

每个 Episode 文件集合 SHA 的计算定义与 Training 表相同。核心行为输入文件 SHA-256 来自逐文件 Integrity Baseline：

| seed | episode_manifest SHA-256 | 文件集合 SHA-256 | tasks.csv | trajectory.csv | prediction_scenarios.npz | vehicle_predictions.csv | popularity_predictions.csv |
|---:|---|---|---|---|---|---|---|
| 30001 | `c41529ed61face7249694894195f5ed5b1c85d49a099e2536b3f78f45002468b` | `081bef0a4442b2282467a51a061bf8e4ca5b2e921e0741b4e4298203f82a7414` | `7431e74b39c7a3b6716526f9dad956428e9e0afeb8c540272c30aa6793563ea9` | `281146b836931dfbe96849b698d372919c74c3d072c931118e6eb80307e29fa4` | `e77ba34df4b0eed35521659606af5b5d444afebaaff2f6efec9f9692dd1fdf79` | `54cf2ad001edb0bd0eb58e741c8249b80664c82541b806d3b58b4e7b40b93111` | `6c9b18e8ac1f325ca403d0de129ba5da3ab59f323079c335babc6fa111fa0882` |
| 30002 | `283ce0a9d91dfda4e086427e8df3cabbf92122853030354f3924c8c6708cd2e1` | `2b416a25f262d79dc0ddb6d210aab9900f8071bf1404ae7ca724a2ebd6f9b199` | `74b5ef364f6951cbf6c5c52f99e2e0929bc1be32846f725ba89282f69906fe8b` | `82567df18e6e92f66d5235e22fbaad1ba79bf4f904bd0ba92f690442febc7c13` | `bf57d7ee2079f05d6877f84bb650404a4192f4c9f6231a6168003ccb5d4b2f19` | `3bfae1a8c5b5e36476780cc8c7971c9a7314c0ed1a202ed584f8de6fc1c563eb` | `21591db06a1e94d2ae98967bf1855a45df9bbd7ad163a5d0d16d3e7e6aa96b28` |
| 30003 | `2b355260804eddeeb7bb5b9780b7ec669062eb2ddc6fed164706ca6b84834894` | `ef1aa4ff764b6ab720eeee2ffa83b741ee7cd5e8a06a80d386a32f42f23fd0a0` | `4a4e8f98419ee065b10b352ddb6109741f5185b10f6f6a3c8aa835690a57b2f7` | `8fd99c39af8ec58e166ae5725164e301bfed6a62be1f82c5d9e89da3db6ad686` | `e08de1db548da2509734819db2c1982292f72bc51c9b81134d57b4f0b321d25a` | `c1910e622d5060050ba120a74bc11cde8d74bb7cbd36d2cd0bc73f82e267c6d9` | `500ba2328d4a318ec69e5e2f151192b7567cc0e8e4e7cb768873ba9c1d2f5bac` |

## Runbook and retained artifacts

- Training command contract: `run_mappo_stage3_cache_value.py --source-checkpoint <Update13-copy> --updates 3 --slots-per-update 40 --training-seed-start 20017 --trainer-seed 71301 --epochs 1 --device cpu --coefficient-profile <frozen-profile>`.
- The Stage3 frozen-evaluation adapter and Stage1C `FrozenEvaluationTrainer` use the existing action/metric contract. Full 300-Slot evaluation is sequential and deterministic per selected episode. Update calls are not made during evaluation.
- Training/evaluation outputs remain below `runs/research_mappo_stage3_main_experiment/`; initial Checkpoint, Update 14–16 Checkpoints, logs, validation result JSON, hashes, and reports are retained. Training run `training/stage3_20261018T095817Z_1b3d4652`; final paired results `validation_3seeds_20261018_retry1/validation_results.json`.
- No source/data generation, no Manifest writes, no Test Episode reads, no parameter search, no commit/push. `environment_data/` is excluded from cleanup and remains byte-for-byte untouched.
- Reference project cleanup is conditional on checking each Git/worktree state, running jobs, unique artifacts, and `environment_data/`; no directory is removed unless every condition is safe.

## Execution status

Completed. The frozen comparison shows a negative Stage3 net-utility difference on all three Validation episodes. Validation ΔQ prediction/ranking metrics remain unresolved because the inherited Stage1C Micro override rejects Stage3's `random_uniforms` replay argument; the failure is recorded without manufacturing labels or repeating the full comparison. The pre-registered continuation gate therefore fails. Reference worktrees were inspected and retained because they contain unique dirty/untracked research outputs; the Stage3 reference also contains `environment_data/`, which was not touched.
