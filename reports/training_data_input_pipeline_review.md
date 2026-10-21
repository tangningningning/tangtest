# Formal Training 数据输入链路复核与修复记录

更新时间：2026-10-20。依据当前本地源码（基线 HEAD `9269e5cf459949021b011cf35ff5648e4e8868ea` 加本轮工作区修改）、Training Pool 抽样 Episode `20001`–`20003` 和现有生成/加载合同。只读了 Training 数据；没有读取 Validation/Test，没有生成或覆盖 Pool、Run、Checkpoint。

```ini
LATEST_LOCAL_CODE_REVIEWED = 9269e5cf459949021b011cf35ff5648e4e8868ea + current local changes + explicit-ID-order compatibility update
TRAINING_EPISODES_CHECKED = 20001, 20002, 20003

VEHICLE_ORDER_FIX = NATURAL SORT NOT REQUIRED; NPZ SOURCE ORDER IS PRESERVED AND OTHER AXES JOIN BY EXPLICIT ID
OBSERVATION_DIMENSION_PRESERVED = YES (same SV/TV counts and feature formula; sampled spec remains 400)
VEHICLE_ID_MAPPING_CONSISTENT = YES_FOR_20001_20003; NPZ ROLE AXES, PREDICTION CSV, TASK CSV AND ROLE JSON JOIN BY ID
CROSS_EPISODE_FEATURE_SEMANTICS_CONSISTENT = EACH_EPISODE_PRESERVES_ITS_STORED_ORDER; PHYSICAL IDENTITY IS NOT ASSERTED
VEHICLE_PREDICTION_CAUSAL = YES_FOR_CUTOFF_AND_TARGET; PLANNED_ROUTE_AVAILABILITY_IS_AN_ASSUMPTION
POPULARITY_PREDICTION_CAUSAL = YES; rho_star IS DIAGNOSTIC ONLY
TASK_SLOT_ALIGNMENT_CORRECT = YES_FOR_THE_3_CHECKED_OLD_EPISODES
PREDICTED_TASKS_USED_AS_ACTUAL_TASKS = NO
TASK_UNIT_CONVERSION_CORRECT = YES_DECIMAL_KB_TO_BIT (8000 bit/KB); GENERATION_JITTER_REMAINS
COMPLETE_VEHICLE_SELECTION_BIAS = POTENTIAL_COMPLETE_CASE_SELECTION; MAGNITUDE_UNRESOLVED
MAP_MATCHING_INFORMATION_ASSUMPTION = FULL_PLANNED_ROUTE_KNOWN_BEFORE_CUTOFF; DEPLOYMENT_AVAILABILITY_UNRESOLVED

RESIDUAL_ID_MATCHING_REMOVED = YES
RESIDUAL_SAMPLING_REPRODUCIBLE = YES (same seed, same indices/shape/output in focused check)
RESIDUAL_CONDITIONING_LIMITATIONS = GLOBAL POOLED DISTRIBUTION; MAY CHANGE CONDITIONAL DISTRIBUTION; NOT CLAIMED OPTIMAL

EXISTING_POOL_COMPATIBLE_WITH_NEW_CODE = YES; 300-EPISODE RUNNER GATE PASSES WITHOUT ORDER-VERSION REQUIREMENT
POOL_REGENERATION_REQUIRED = NO_FOR_LOADING; NEW GLOBAL-RESIDUAL DATA WOULD BE A SEPARATE DATASET
POOL_REGENERATION_EXECUTED = NO
DATA_INPUT_PIPELINE_ASSESSMENT = OLD POOL STRUCTURE IS LOADABLE AFTER PER-EPISODE ID VALIDATION AND SOURCE-ORDER ARRAY MAPPING
BLOCKING_ISSUES = HISTORICAL GENERATION ATTESTATION IS UNAVAILABLE; OLD CONTROLLED CHECKPOINT DOES NOT RECORD THE VEHICLE FEATURE ORDER AND IS NOT CLEARED FOR SAME-RUN RESUME
RECOMMENDED_MINIMAL_FIX = USE THE COMPATIBLE 300-EPISODE POOL WITH A NEW RUN ID; KEEP THE OLD CHECKPOINT AS A SEPARATE EXPERIMENT
FORMAL_DATA_MODIFIED = NO
TRAINING_EXECUTED = NO
VALIDATION_TEST_READ = NO
```

## 车辆角色顺序与映射

旧 Pool 的三个抽样 Episode 都有 5 个 SV、20 个 TV，但角色内部顺序随 Episode 变化。抽样文件中的 ID 集合可一一对应；`vehicle_predictions.csv` 的行顺序并不等于 `vehicle_roles.json` 的顺序，因此加载器必须按显式 ID 建表，不能把 CSV/NPZ 的相同行号当作同一车辆。不同 Episode 的车辆 ID 集合可以不同，本合同不会要求它们是同一批物理车辆。

当前实现已移除 `ROLE_NATURAL_ID_ORDER_V1` 的强制排序要求。每个 Episode 的 `prediction_scenarios.npz:vehicle_ids` 是位置轴的顺序来源；加载时按其 ID 与角色标签构造 SV、TV 两段，保留各角色在 NPZ 主轴中的相对顺序。任务数组依据 `task_vehicle_ids` 显式映射到 TV 主序列。角色 JSON 的车辆列表只用于角色成员校验，不被当作跨文件数组位置的依据。

正式加载端现在逐 Episode 核验：

- `vehicle_roles.json` 中 SV/TV/research ID 的唯一性、数量、分组和互斥关系；
- NPZ 中车辆 ID、角色带及 SV/TV 轴的集合和角色映射；
- 所有车辆/任务数组的 shape、Slot/horizon 轴和有限值；
- NPZ 位置轴按 ID 映射到保留的 SV/TV 主序列；任务数组按 ID 映射到 TV 子序列；
- CSV 预测与 `tasks.csv` 的车辆引用通过 ID 校验和查找，任务所属车辆、Slot 与时间不因对齐而改写。

加载器不要求历史排序标记或自然排序。缺 ID、重复 ID、角色冲突、数组维度不符及无法建立唯一映射仍会拒绝。`_assert_realization_context_compatible()` 校验观测维度、特征切片和角色数量，允许各 Episode 使用不同 ID 和角色内顺序。观测仍由 `ObservationSpec.vehicle_ids` 逐车展平，维度保持 400；特征排列来源改为该 Episode 保存的 NPZ 主轴顺序。

## 残差匹配

校准残差库是独立校准轨迹数据，原来有 3,552 个 `(5,2)` 残差块、12 个校准 ID，每个 ID 296 块。旧逻辑按 ID 字符串先匹配；Training 三集共 75 个车辆 ID 出现中，名称重合为 15 次（20%），覆盖 9/47 个不同 Training ID。校准与训练中的同名 ID 不是同一物理车辆；例如 ID `42` 的路线和位置不同。因此旧的 `same_vehicle_first_else_pooled` 不具备物理身份依据。

现在每辆车、每个场景都从全局残差块索引池采样，不再根据车辆 ID 选子池。保留整块抽样、独立抽样流、随机种子、残差维度/单位、道路合法性重采样和失败时回退 nominal 轨迹；增加了索引形状及输入有限值检查。输出标记为 `GLOBAL_POOLED_WHOLE_BLOCK_V1`。

校准库没有可靠的路线类别、速度分组等可用字段，因此本轮不虚构条件标签。移除 ID 条件后，残差的条件分布可能变化；全局池只是当前字段能支持的可复现方案，不能据此声称最优。

## Training Pool 兼容性与迁移

当前 Pool Manifest 和完整性基线仍原样标识旧数据，Pool 目标指纹为 `4083b917948cd7203972c0edd5b40d2efefd8984dde27089a77da43d460b32f3`。当前生成器运行指纹为 `bae8ecafedc9977c666a35d986d683bc6c9734487a4399b2b95cadce2be24cb9`；门禁仅对已知 Pool 指纹按旧 schema 与历史残差版本元数据重建精确匹配，其他配置字段、任务合同、服务目录、Pool Manifest 与完整性基线仍严格检查。车辆排序版本不再参与拒绝条件，未知 Pool 指纹仍不通过。

旧 NPZ 的预测场景直接读取，不重新生成。20001–20003 的文件明确记录 `vehicle_residual_bank_preference=same_vehicle_first_else_pooled`；加载后仅增加内存级 `loader_compatibility` 说明，原始 NPZ metadata 不改写。新生成场景仍使用当前 `GLOBAL_POOLED_WHOLE_BLOCK_V1`。旧 Pool Manifest 同时声明 `HISTORICAL_GENERATION_ATTESTATION_UNAVAILABLE_SOURCE_SNAPSHOT_HASHED`，因此不能把其历史生成器来源提升为已独立证明；兼容性只表示已保存文件的结构和语义能由当前加载器解释。

只读正式 Runner 门禁报告 300 Episode、300 Slot，`_resolve_training_episodes()` 找到原有 `episode_20001`–`episode_20300` 共 300 个目录；对 20001–20003 的 Episode 转换 Manifest 和 artifact 哈希逐文件验证通过。没有扫描全池文件哈希，也没有改动或重建任何 Pool 文件。未来若要研究新的全局残差生成样本，仍应使用独立目录和新的真实生成指纹；这不是当前加载旧 Pool 的前置要求。

正式训练运行时，`Run Manifest` 会记录池的历史指纹、当前加载器指纹、首个 Episode 的 NPZ 顺序来源与原始版本标记、残差来源标记和按 ID 映射策略；不会改写 Pool Manifest 或伪造旧场景的生成版本。

**受控 Checkpoint 单独核对：**Run Manifest、Run Summary 和 `latest_checkpoint.pt` 的受控停止状态、Episode/Update 计数及配置哈希一致（8 Episode、2 Update；配置哈希匹配当前 alternating 配置；Checkpoint 声明 `RESUMABLE_MAINLINE_CHECKPOINT`）。但 Checkpoint `config_contract` 没有车辆特征顺序版本，保存的动态 SV ID 集合为 `4,23,32,36,52`，与当前 Episode 20001 的集合 `0,10,23,46,59` 不同。当前 Checkpoint 检查允许动态 SV 身份变化，因此可能机械加载，却不能证明策略权重对应当前 Pool 数据。未修改 Checkpoint 格式或身份校验；该旧 Checkpoint 不应作为相同语义实验直接 Resume，应以新 Run 身份开始。

## 其他已核查输入合同

先前对三个 Training Episode 的只读核查显示：车辆预测 cutoff 为 `origin_slot - 1`，目标为 `origin_slot + horizon`；流行度模型输入截止上一 Slot，预测 `rho_star` 仅作为 CSV 诊断列而未进入 Observation；尾部无效 horizon 由 mask 屏蔽。实际任务按 `tasks.csv` arrival 进入 Matching/Reward，NPZ 预测任务未用作实际任务。Slot 与时间戳对齐。

Service Catalog 输入合同为十进制 KB 转 bit（`1 KB = 8000 bit`）；`cpu_cycles = cycles_per_bit × input_bits × load_factor`。旧样本有输入 ±20% 和工作量 ±10% 抖动。完整车辆筛选会固定 Episode 内车辆群体并排除不完整轨迹，存在 complete-case 选择偏差；其总体影响未量化。预测使用 cutoff 时已知的完整计划路线，这项真实部署信息是否可用仍未验证。

## 针对性验证

- 对 20001–20003 读取 Training 数据的既有映射、任务行和 1,500 组车辆预测；确认 CSV 行序可不同但 ID 集合匹配。
- 执行只读正式 Pool 合同门禁与 Runner Episode 路径解析，300 个 Episode 均存在；抽样 3 Episode 的存储清单和文件哈希验证通过。
- 加载抽样 3 Episode，并逐元素核对位置数组沿用 NPZ 主序列、`arrival/service_id/input_size/cpu_cycles` 按 TV ID 对齐；预测和任务对齐各读取 1 个 Slot，ObservationSpec 均为 400 维。另用内存打乱的车辆轴和任务轴验证按 ID 重排。
- 跨 Episode 兼容检查接受不同车辆 ID 与顺序；内存中制造重复 ID、角色冲突和未知任务 ID 后加载器均拒绝。300 Episode 旧 Pool 门禁通过，未知 Pool 指纹分支仍保留严格拒绝。
- 旧 `same_vehicle_first_else_pooled` metadata 在内存加载后保持原值；没有重新生成预测场景。
- 使用真实校准库验证每个车辆索引都指向完整残差池；用固定 seed 两次生成的采样形状、有限值和结果一致。
- 只读受控 Run/当前配置核对 Cache 成本结果，详情见 [Cache 成本公式一致性复核](cache_cost_formula_consistency_review.md)。
- 修改文件的 Python 源码语法检查通过；`git diff --check` 无空白错误（Git 对此前修改的三个文件提示 LF/CRLF 转换警告）。没有运行 Episode、PPO/MAPPO 或 Pool 生成。

PowerShell 中 `conda activate adaptive_cache_mec` 未能执行；按项目规则改用 `C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`，并验证 `sys.prefix` 指向 `adaptive_cache_mec`。本轮针对性检查使用该环境，没有调用 Matching/KKT 或训练入口。

