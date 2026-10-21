# Training Slot Matching + KKT 求解效率诊断

生成时间：2026-10-19T13:39:48.124057+00:00

## 运行与数据来源

- 仓库：`main`，HEAD `faa73c932bcaa17f6f29d1f19481380b7f0cc274`。
- 工作区基线状态：本任务开始时已有大量未跟踪数据与报告；诊断只新增本脚本及本报告的 Markdown/JSON，不改写已有文件。
- Python：`C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`；`sys.prefix`=`C:\software\Anaconda_envs\envs\adaptive_cache_mec`。Conda 环境核验：`adaptive_cache_mec`。
- 硬件：`Windows-10-10.0.26300-SP0`；处理器 `AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD`；逻辑 CPU `32`；内存 `NOT_AVAILABLE GiB`。
- 配置：`configs/proposed/system_config.toml`，SHA-256 `0927080d4d33c640a7cb0005662e686c75ff41805e2439cff5d2046c2b4e4795`。Training Pool Manifest SHA-256 `b6cae236b029f4d24c7488a91f0825247a20f475ec76b2d80737de44ae491bd3`。
- Pool 配置哈希与当前配置哈希相同：`False`；任务生成合同逐字段一致且 Service Catalog 哈希匹配：`True`。
- Pool 的完整配置哈希与当前配置哈希不一致；任务生成合同字段和服务目录哈希通过核对，因此报告只确认任务数据合同一致，不声称 Pool 完整参数快照与当前配置完全相同。执行测量使用当前本地配置。
- 当前工作区还存在 `configs/proposed/system_config.toml` 的未提交成本系数差异（`lambda_cache=0.005`、`lambda_movement=5.0e-6`）；本诊断未编辑该文件并保留该差异。冻结 Cache/UAV 不触发配置成本，因此不改变本次 Matching/KKT 的任务评价。
- 仅扫描 `environment_data\proposed_joint_training`。未读取 Validation/Test。Pool 声明 `300` 个 Episode、每 Episode `300` Slot；其中经 Pool Manifest 与 Episode Manifest 绑定且任务表覆盖完整的 Episode `300` 个。
- Slot 任务数定义为 `tasks.csv` 对该 Slot 的实际行中 `arrival=1` 的数量；每个样本还与正式 `aligned_train_steps()` 的 `len(actual_tasks)` 核对。任务输入字段单位按正式 `Task.input_bits` 处理，汇总以十进制 MB 表示；计算需求以 Gcycle 表示。
- 选样规则：对每个目标到达任务数，按 (episode_seed, slot) 升序排列；每个 Episode 仅取最早匹配 Slot，再取最小的三个不同 Episode seed。选样不使用计时、效用或 Matching 结果。
- Cache/UAV 状态：`FROZEN_INITIAL_CACHE_UAV_REFERENCE`。每次测量使用该 Episode 由正式环境入口加载的初始 Cache/UAV，Slot 内冻结；不是 PPO 在所选 Slot 的策略动作或状态快照。
- 预测文件按正式接口加载并做 Slot 对齐，但因为 Cache/UAV triggers 均为零，执行器跳过配置预测求值；任务匹配只使用该 Slot 实际任务与实际车辆位置。

## 求解合同与计时口径

- Matching：`transfer_exchange`，`max_updates=100`，容差 `1e-09`，Cloud fallback `RESOURCE_AWARE_FALLBACK`；评价模式为源码默认 `incremental`。正式 Runner 未传持久 `NodeEvaluationCache`，诊断同样不传；每次独立调用使用全新的 `KKTResourceAllocationCache`。
- KKT：min CPU fraction `1e-06`，min bandwidth fraction `1e-06`，bisection tolerance `1e-12`，max iterations `100`。噪声模型 `BANDWIDTH_DEPENDENT_NOISE_V1`。
- 效用权重：`omega_delay=0.5`，`omega_energy=0.5`；T_ref `30.473728355325758` s，E_ref `5.1574642506299275` J。Edge CPU：SV `1`、RSU `10`、UAV `5` GHz；Cloud `40` GHz，模式 `finite_pool`。Cloud 等效上传 `20` Mbps，时延系数 `13.0`。
- 先用所选样本中任务最多的真实 Slot 执行一次不计时预热，排除 Numba 首次编译；预热运行不纳入统计。计时最多重复 3 次；若某 Slot 首次计时的 Matching 完整调用超过 2 s，则该 Slot 不再重复。各重复调用从同一初始状态开始，KKT cache 每次新建。
- `matching_total_runtime` 是源码在 `optimize_task_matching()` 调用边界内测得的完整 Matching 时间，作为主指标。`matching_runtime` 是 Matching 总时间扣除节点缓存查询、节点评价 miss 和 `kkt_runtime` 后的剩余部分。`matching_kkt_seconds` 是 Slot 包装层从构造 Service Node 位置开始到 Matching 返回的时间，包含位置构造，因此不与 `matching_total_runtime` 相加。外部 Slot wall 还包含入口校验和 Reward 组合。
- `kkt_runtime` 是 KKT 资源分配上下文总时间；`kkt_solver_core_runtime` 是其子项；CPU、带宽和 Cloud solver core 也是子项，不能彼此相加后再加到总时间。候选子计时（生成、上界筛选、精确评价 orchestration）嵌套于 Matching 的非 KKT 路径，仅用于定位，不与 `matching_runtime` 重复求和。
- 事后源码核对确认正式 `NEW` 任务路径未单独记录 Edge CPU 与带宽 solver core 时间；报告将这两个字段改列为 `NOT_AVAILABLE`，未重跑 Matching，既有完整调用/KKT 总耗时和计数保持原测量值。
- 候选剪枝比例使用 `upper_bound_pruned_candidates / upper_bound_screen_calls`；这是被上界检查的候选中实际剪枝比例，不等同于运行时间节省比例。候选定义数、候选评价数和接受更新数分别报告。

## 样本汇总

| 指标 | 10 任务 | 14 任务 | 20 任务 |
|---|---:|---:|---:|
| 可用真实 Slot | 3 | 3 | 3 |
| Matching 完整调用中位数 (ms) | 10.322 | 47.688 | 179.523 |
| KKT 总耗时中位数 (ms) | 3.863 | 23.906 | 86.688 |
| KKT / Matching 中位数 (%) | 43.997 | 48.607 | 43.198 |
| CPU KKT 核心中位数 (ms) | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |
| 带宽 KKT 核心中位数 (ms) | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |
| KKT Solver core 总中位数 (ms) | 2.264 | 18.184 | 69.384 |
| Cloud KKT 核心中位数 (ms) | 0.363 | 1.332 | 3.035 |
| 候选定义中位数 | 301 | 836 | 2429 |
| 候选实际评价中位数 | 30 | 114 | 338 |
| 上界剪枝中位数 | 59 | 157 | 379 |
| 上界剪枝率中位数 (%) | 60.204 | 57.412 | 52.859 |
| Transfer / Exchange 候选评价中位数 | 28 / 1 | 78 / 36 | 252 / 86 |
| Full evaluator 调用中位数 | 30 | 114 | 338 |
| KKT Solver 调用中位数 | 38 | 114 | 319 |
| KKT cache 命中率中位数 (%) | 46.939 | 61.951 | 67.203 |
| 每 KKT Solver core 中位数 (ms/call) | 0.060 | 0.160 | 0.218 |
| Matching 更新中位数 | 4 | 7 | 13 |
| 任务效用总和中位数 | 9.400892 | 13.081079 | 18.608967 |

每个任务规模的原始 Slot 与重复计时见下表。候选/KKT 计数应在同一 Slot 的重复间一致；若不一致，JSON 中保留每次结果。

| 任务数 | Episode Seed | Slot | 输入总量 (MB) | CPU 总量 (Gcycle) | 可行 Edge 任务-节点边 | 节点任务分布 (SV/RSU/UAV/Cloud) | Matching ms (中位数) | KKT ms (中位数) | 候选定义/评价/剪枝 | KKT 调用/缓存命中 | 稳定停止 |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 20001 | 21 | 7.086 | 40.687 | 8 | 0/0/1/9 | 5.481 | 2.401 | 171/22/29 | 29/20 | converged_by_no_improvement / stable=True |
| 10 | 20002 | 22 | 7.969 | 48.713 | 18 | 0/1/3/6 | 13.425 | 6.433 | 301/30/94 | 43/23 | converged_by_no_improvement / stable=True |
| 10 | 20003 | 21 | 6.805 | 39.808 | 14 | 0/2/2/6 | 10.322 | 3.863 | 330/39/59 | 38/44 | converged_by_no_improvement / stable=True |
| 14 | 20001 | 1 | 10.485 | 48.351 | 28 | 2/1/5/6 | 83.189 | 38.038 | 1140/181/244 | 184/227 | converged_by_no_improvement / stable=True |
| 14 | 20002 | 10 | 10.328 | 57.883 | 18 | 1/0/4/9 | 47.688 | 23.906 | 639/114/68 | 114/127 | converged_by_no_improvement / stable=True |
| 14 | 20003 | 2 | 10.011 | 58.511 | 22 | 1/2/3/8 | 28.294 | 13.733 | 836/80/157 | 90/84 | converged_by_no_improvement / stable=True |
| 20 | 20003 | 119 | 17.079 | 98.147 | 51 | 2/4/5/9 | 373.340 | 158.925 | 4131/855/776 | 698/1132 | converged_by_no_improvement / stable=True |
| 20 | 20011 | 48 | 16.275 | 81.831 | 34 | 2/4/4/10 | 89.777 | 38.776 | 2429/212/378 | 179/270 | converged_by_no_improvement / stable=True |
| 20 | 20012 | 127 | 13.940 | 72.949 | 33 | 1/2/4/13 | 179.523 | 86.688 | 2407/338/379 | 319/418 | converged_by_no_improvement / stable=True |

## 诊断结论

- 本次实际样本为 `9` 个 Slot，任务规模组分别有 `3`、`3`、`3` 个。样本先按真实到达任务数筛选，规模组中位数只描述这些冻结初始配置下的样本，不足以证明复杂度阶数。
- 主时间分解的跨样本中位数：Matching 剩余非 KKT 7.421 ms，节点效用评价（不含 KKT）16.356 ms，KKT `23.906` ms，其中 solver core `18.184` ms。主导项判定：**KKT 总耗时为主要项；正式 Slot 路径未分别记录 Edge CPU/带宽核心时间，子瓶颈未分辨**。
- Matching 内部候选生成中位数 `0.367` ms、上界筛选 `0.702` ms、完整评价编排 `2.867` ms；这些时间是非 KKT 路径的细分，不与该路径总时间相加。
- 本轮计时覆盖 `27` 次 Matching 调用，KKT Solver 调用合计 `5082` 次；KKT cache 请求/命中/未命中为 `10476` / `7035` / `3441`，总体命中率 `67.153%`，每次 Matching 的 KKT Solver 调用中位数 `114`。NodeEvaluationCache 按正式 Rollout 默认不启用，因此本轮请求/命中/未命中均为 0。
- KKT 核心细分完整性：CPU `NOT_AVAILABLE` ms、带宽 `NOT_AVAILABLE` ms、Cloud CPU `1.332` ms。正式任务均为 `NEW`；`src/decision/task_allocation.py:2988-3004` 将其导入阶段资源路径，该路径在 2886/2949 只累计总 solver core。CPU/带宽分项字段在 3063/3177 的另一分配路径中记录，因此当前路径的 CPU/带宽报告为 `NOT_AVAILABLE`，不解读为零。
- 按任务规模中位数估算的每次 KKT Solver core 时间：10/14/20 任务分别为 `0.060` / `0.160` / `0.218` ms。随着任务数增加，候选评价数与每次求解涉及的资源维度都改变；本小样本不能把时间增量单独归因于候选数。
- 初步判断：KKT 总时间不小于剩余 Matching 时间；当前正式 NEW-task 路径未分别记录 Edge CPU/带宽核心时间，先补齐分项计时再定位子瓶颈。
- 是否引入 GNN：运行时间数据只提供效率瓶颈线索，不证明 Matching 解质量不足，也不证明 GNN 可以加速或提升效用。基于本轮瓶颈，建议先 `暂不引入 GNN；若继续优化，先补齐正式解析式路径的 Edge CPU/带宽 KKT 分项计时，再按实测主要子项做局部实现优化。`。

## 口径与限制

- 表中任务效用仅解释样本，不是 Matching 最优性差距或跨规模因果结论。不同 Slot 的信道、计算需求、任务服务与可行节点边均不同。
- 配置为 `FROZEN_INITIAL_CACHE_UAV_REFERENCE`；本结果不表示 PPO 策略的真实 Cache/UAV 动作。任务模型为每 Slot 独立解析式评价，无跨 Slot 执行队列。
- KKT 与其 CPU/带宽/Cloud 核心细项是嵌套计时，不作重复求和。Matching 缓存与结果拼装不归类为 KKT solver core。
- 本轮未训练 PPO、未读取 Validation/Test、未修改正式算法/配置/数据，未运行完整 Episode。

## 最终字段

```text
TRAINING_SLOTS_AVAILABLE_10 = 3
TRAINING_SLOTS_AVAILABLE_14 = 3
TRAINING_SLOTS_AVAILABLE_20 = 3
MATCHING_TIME_10_MEDIAN_MS = 10.322
MATCHING_TIME_14_MEDIAN_MS = 47.688
MATCHING_TIME_20_MEDIAN_MS = 179.523
KKT_TIME_RATIO_10 = 43.997%
KKT_TIME_RATIO_14 = 48.607%
KKT_TIME_RATIO_20 = 43.198%
DOMINANT_COST_COMPONENT = KKT 总耗时为主要项；正式 Slot 路径未分别记录 Edge CPU/带宽核心时间，子瓶颈未分辨
MATCHING_SCALING_EVIDENCE = 样本组中位 Matching ms=10.322,47.688,179.523，候选评价中位数=30,114,338；仅是固定小样本的描述性变化，不能据此拟合或断言复杂度阶数。
GNN_SPEEDUP_OPPORTUNITY = 9 个抽样 Slot 的 Matching 调用均低于 1 s，最大约 375.612 ms；本次没有足够效率证据支持增加 GNN，不代表更大规模或更严实时限下也满足要求。
RECOMMENDED_NEXT_STEP = 暂不引入 GNN；若继续优化，先补齐正式解析式路径的 Edge CPU/带宽 KKT 分项计时，再按实测主要子项做局部实现优化。
PPO_TRAINING_EXECUTED = NO
FORMAL_ALGORITHM_MODIFIED = NO
FORMAL_CONFIG_MODIFIED = NO
```
