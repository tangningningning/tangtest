# Stage B1：移除 TV 本地执行机制

## 结果摘要

TV 现在只产生任务并上传；任务仅可分配到满足服务缓存与物理链路可行性的 SV、RSU、UAV，或 Cloud。正式配置的 RESOURCE_AWARE_FALLBACK 继续让 Cloud 与可行 Edge 同时参与 Matching。Transfer-Exchange 搜索、Edge 缓存约束、通信与计算模型、KKT 求解及 Reward 公式均未重写。

## 修改内容

- src/decision/task_allocation.py 删除 TV Local target、可行性、执行效用及 KKT evaluator 分支。无 Edge 缓存命中时任务只能走 Cloud；有合法 Edge 时，Edge 候选仍须命中服务缓存并通过通信链路检查。
- src/decision/slot_environment.py、src/ppo_simple/rollout.py 及其正式调用方移除本地缓存参数和 Local 分配统计。任务分配统计现在只列 SV、RSU、UAV、Cloud。
- src/common/system_model.py 删除只供 TV Local 使用的本地执行公式。车辆发射功率及卸载通信能耗仍保留。
- src/scenario/task_scenario.py 删除 TV 缓存容量检查、生成、验证和读取逻辑；scripts/generation/generate_episode_realization.py 不再生成 tv_local_cache.json，seed stream 中也不再包含 tv_cache_seed。
- src/scenario/realization.py 与 src/ppo_simple/interface.py 移除该 artifact 字段和加载依赖；scripts/training/mainline_runner.py 不再把它列为Episode必需文件。Episode完整性检查仍严格检查其它正式文件和manifest。
- scripts/analysis/analyze_training_environment_diversity.py 不再要求、散列或读取该文件；既有历史分析报告未改动。
- src/decision/cache_configuration.py、src/ppo_simple/primitive_control.py 删除按 TV 本地副本计算的 observed_local_service_usage 候选特征。内部候选特征宽度由 21 + 6S 改为 18 + 6S；Cache成本、装入/移除量、多样性和需求经济特征索引已同步调整。最终策略输入仍为 181 维基础特征加19维经济特征，共200维。
- configs/proposed/system_config.toml 原地移除 paths.tv_local_cache、TV本地CPU/缓存容量及仅用于本地计算的 kappa；nodes.vehicle.transmit_power_w、缓存/计算资源和系统Reward参数保留。T_ref、E_ref 数值未变，配置注释说明它们仍是整改前冻结的Training校准常数。
- 策略合同更新为 ...v3_no_tv_local，架构版本从31升至32，Checkpoint格式版本从58升至59。PPO网络输入宽度仍为200，优化器、GAE、Actor/Critic算法未改；旧Stage6 Checkpoint因策略/架构/Checkpoint合同不符，不可直接加载或续训。

## 数据与旧实验兼容性

旧Episode目录中的 tv_local_cache.json 被视为多余文件：新加载合同既不要求也不暴露它，额外存在时会被忽略；新生成的Episode不会产生它。历史300个Training Episode、Episode及Pool manifest、Checkpoint、runs、报告，以及 data/generated/generation_metadata.json 均未改动、删除或复制。本次没有保存旧配置副本。旧生成元数据仍记录原有缓存artifact来源，但源码不再读取该记录。

现有Training Pool不能作为新配置下的正式Stage6训练池直接复用。旧Pool记录的配置SHA-256为 684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77，当前原地修改后为 f3131efb7832efa925c181d9abd07bec5f2bfe3e7507eca2cc2515b77067699b；Pool的语义fingerprint也不匹配。mainline_runner 的配置与manifest一致性检查保持严格，没有绕过。旧任务/轨迹文件未证明必须重新计算，但在不改manifest、不放宽校验的约束下，后续正式训练必须先准备与新系统配置一致的新Training Pool/manifest，并训练新Checkpoint。

## 最小验证

- Python AST解析：17个修改过的Python文件通过；正式TOML解析通过；git diff --check 通过。
- 使用项目正式配置加载及 CacheOptimizationConfig.validate()：通过。
- 确定性可行节点检查：Edge未缓存服务时结果为 CLOUD；Edge缓存命中时为 SV_0 + CLOUD；不存在Local目标。
- 临时最小artifact目录检查：缺少 tv_local_cache.json 可通过；额外放入该旧文件仍可通过，且返回的artifact合同不含该字段。临时目录已自动清理。
- 未运行Training Episode、SUMO、完整Matching/KKT或PPO。

Windows上 conda activate adaptive_cache_mec 未能启用；验证使用 C:\software\anaconda3\python.exe（Python 3.12.7，NumPy 2.2.6）。此环境已安装的Numba与NumPy版本不兼容（Numba要求NumPy不高于2.0）；候选检查通过项目纯Python可行节点路径完成，未安装或升级依赖，也未运行KKT。

## 执行边界状态

TV_LOCAL_EXECUTION = DISABLED
TV_LOCAL_CACHE_REQUIRED = NO
EDGE_SERVICE_CACHE_REQUIRED = YES
CLOUD_FALLBACK = ENABLED
MATCHING_KKT_CORE_CHANGED = NO
REWARD_FORMULA_CHANGED = NO
PPO_LEARNING_ALGORITHM_CHANGED = NO
OLD_STAGE6_ARTIFACTS_PRESERVED = YES
OLD_STAGE6_TRAINING_POOL_REUSABLE_AS_FORMAL = NO
OLD_STAGE6_CHECKPOINT_REUSABLE = NO


## Stage B1 收尾：Training Pool 复用与配置身份核验

本节修正前文“Pool 因当前配置哈希变化而不能复用”的判断：**配置身份变化本身不再是拒绝原因**。Runner 现在把Episode生成合同与PPO运行合同分开核验；实际复用仍被旧Episode Manifest缺少完整来源与逐文件哈希阻断。

### 本次修改

- `scripts/training/mainline_runner.py` 不再要求Pool的历史 `config_sha256`、`semantic_config_sha256` 等于当前运行配置。它单独校验历史配置身份格式及其内部组合关系，要求Pool和Episode的生成指纹与当前生成合同相符，并保留Training split、300个种子、Pool登记的Episode Manifest哈希、服务目录及数据来源检查。对历史workload overlay，缺失源文件或输出文件哈希时会明确拒绝，不会用配置哈希变化作为替代错误，也不会跳过完整性核验。
- `scripts/generation/generate_episode_realization.py` 的历史来源校验不再要求旧 `source_git_sha`/dirty状态等于当前HEAD；改为核验旧SHA格式、dirty类型、生成参数字段、当前生成合同、源文件身份、派生种子及全部Episode文件哈希。
- `src/scenario/task_scenario.py` 恢复通用JSON对象读取函数。Stage B1清理TV缓存时一并删除了仍被GRU预测模块导入的通用接口，导致正式生成模块无法导入；该修复与TV本地缓存无关。
- `src/ppo_simple/reporting.py`、`src/ppo_simple/trainer.py` 让新run manifest分别记录当前运行配置身份和历史Pool来源身份，并在恢复训练时核对Pool Manifest哈希。未改变PPO算法或Checkpoint结构。
- 未修改正式配置、Training Episode、Episode/Pool Manifest、Checkpoint、runs或已有Stage6报告。

### 配置与数据身份结果

- 当前运行配置：原始SHA-256 `f3131efb7832efa925c181d9abd07bec5f2bfe3e7507eca2cc2515b77067699b`；语义SHA-256 `113dd940b745874c7f9690dc903128d5b052bc2e3b7bc102134f3704661b41ae`。
- 历史Pool仍记录旧原始SHA-256 `684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77` 和旧语义SHA-256 `10f893fc549c4016bedb1a5389ad996f9a3ff646aa012596a7230c284b13f542`；新Runner分别保留，不把它们与当前运行配置混为一谈。
- 当前正式生成指纹为 `cfb3ee9fae004108af2d6bcb641609142c305d3ad6cc3ca74be13faaa6ff9865`，与Pool记录一致。Training split、当前及历史Stage6服务目录哈希、300个Episode Manifest SHA-256映射均通过静态核对；300个Manifest哈希全部匹配Pool登记值，必需运行数据文件均存在。
- 但300份现有Episode Manifest全是 `stage6_workload_overlay_v1`，没有 `generation_config_fields`、原始Episode文件哈希、变换后文件哈希、`source_git_sha` 或 dirty状态。它们只记录了overlay描述和Pool登记的Manifest哈希；Manifest哈希不能证明`tasks.csv`、轨迹、预测场景等文件内容未变。前51个Episode的独立备份虽保留原生成器哈希，但不足以覆盖300个最终overlay Episode。
- 建立快照基线前，严格Runner因缺少历史来源与逐文件哈希而拒绝该Pool。下文记录的受限快照校验现已允许它在 `CURRENT_SNAPSHOT_VERIFIED` 策略下复用；这只解决从基线时刻起的文件完整性验证，不补足历史Git来源或最初生成时的文件证明。
- 旧目录里额外保留的 `tv_local_cache.json` 仍不进入新系统的环境加载、Matching或PPO特征；新模型不要求该文件。历史原始Manifest若登记了它，严格文件哈希校验会照常包含它。
- 旧Stage6 Checkpoint兼容性保持原状：当前策略合同为 `v3_no_tv_local`、架构版本32、Checkpoint版本59；其旧run manifest记录架构版本31、Checkpoint版本58、策略合同 `...horizon_mask_v2`，现有 `load_checkpoint()` 的版本与策略合同检查会拒绝它。本次仅读取run manifest核对身份，没有加载、修改或迁移Checkpoint。

### 最小验证

- 5个修改后的Python文件通过AST解析；正式TOML解析及项目配置加载通过。
- `_load_stage6_contract()` 成功；当前与历史Pool生成指纹相同。严格Pool校验在真实Training数据上执行负例，因历史Episode来源及逐文件哈希字段缺失而拒绝；没有跳过此检查。
- 正式加载模块导入通过；新执行路径中未发现读取TV缓存的引用。未运行训练、SUMO、完整Episode回放或Matching/KKT，也未打开 `proposed_joint_validation` / `proposed_joint_test` 下的Episode文件。范围说明：核验来源服务目录哈希时，读取了Training池外 `environment_data/stage6_candidate_feasibility_25_20_12_reduced/overlay_manifest.json` 元数据；该文件说明其来源池为Validation并记录两个Episode的文件哈希与窗口任务数。未进一步读取其来源Episode；这属于Training池外的派生元数据读取。
- `conda activate adaptive_cache_mec` 在本机失败（Windows返回9020）。使用已有 `C:\software\anaconda3\python.exe`（Python 3.12.7、NumPy 2.2.6），未安装或更改依赖。

### 建立快照基线之前的状态

```text
STAGE_B1_DATA_REUSE_FIX = FAIL
OLD_TRAINING_EPISODES_REGENERATED = NO
OLD_TRAINING_EPISODES_MODIFIED = NO
OLD_EPISODE_MANIFESTS_MODIFIED = NO
OLD_POOL_MANIFEST_MODIFIED = NO
TRAINING_POOL_REUSABLE = NO
GENERATION_PROVENANCE_VALIDATION = FAIL
RUNTIME_CONFIG_IDENTITY_SEPARATED = YES
TV_LOCAL_EXECUTION = DISABLED
EDGE_CACHE_CONSTRAINT = PRESERVED
OLD_CHECKPOINT_COMPATIBILITY = REJECTED
TRAINING_EXECUTED = NO
```

## Stage B1：Training Pool Current-Snapshot Integrity Baseline

### 基线建立与校验

- 在现有 `environment_data/proposed_joint_training` 内仅新增 `stage6_training_pool_integrity_baseline.json`。它是建立时刻的文件快照索引，不是Episode副本，也不补写历史Manifest。
- 基线建立时间：`2026-10-13T07:37:12Z`；建立时记录Git HEAD `1fe65bf36e0d5b4d34dd29a35a8038336c9c604f`、工作区dirty=`true`。这些Git信息只标识基线建立时的源码状态。
- Baseline SHA-256：`f25594d7da6fac36af2f8e7ce00b3f124e16e0fcc5d711d691ac99e5d0337b94`。索引覆盖300个Training Episode、3002个文件，共8,338,304,239字节；逐文件保存相对路径、大小与流式SHA-256。全部Episode包含历史 `tv_local_cache.json`；Episode 20001保留的两个SUMO需求文件也在索引中。
- 建立前核验通过：Pool Manifest SHA-256为 `c9c401dfe6c36b11faecfa080b86cb878d325c2799ee7e54435ad85365992d93`，schema/split/seed区间/300数量正确，根目录无多余项；300份Episode Manifest均匹配Pool登记哈希；Training生成指纹 `cfb3ee9fae004108af2d6bcb641609142c305d3ad6cc3ca74be13faaa6ff9865` 与当前生成合同相同；正式服务目录哈希与Pool登记一致，必需训练文件齐全。当前配置通过Stage B1系统模型和策略合同检查。
- 基线创建后，Runner的 `_validate_training_pool()` 对真实Training Pool重新校验并通过；会检查基线身份、300个Episode的确切文件集合及每个文件的大小和哈希。内存负例中篡改文件大小记录和Pool身份记录均被拒绝。训练入口只读取基线，基线不存在或不匹配时失败关闭，不自动生成或修复。

### 历史来源与运行身份

- 300份旧 `stage6_workload_overlay_v1` Manifest都缺少 `source_git_sha`、`source_git_dirty`、`generation_config_fields`、源/生成文件哈希字段。因此 `historical_generation_attestation=UNAVAILABLE`、来源状态为 `HISTORICAL_PROVENANCE_INCOMPLETE`。当前快照只能证明从上述基线时刻起文件未改变，不能追溯证明文件最初由哪一版源码生成。
- Pool仍记录历史运行配置SHA-256 `684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77` 和语义哈希 `10f893fc549c4016bedb1a5389ad996f9a3ff646aa012596a7230c284b13f542`。当前运行配置身份单独记录为原始SHA-256 `f3131efb7832efa925c181d9abd07bec5f2bfe3e7507eca2cc2515b77067699b`、语义SHA-256 `113dd940b745874c7f9690dc903128d5b052bc2e3b7bc102134f3704661b41ae`。历史配置身份不再与运行配置哈希混用；生成指纹仍须匹配。
- 新Run Manifest会记录当前运行配置、策略合同/架构/Checkpoint版本、Pool Manifest SHA、完整性基线SHA、生成指纹、`CURRENT_SNAPSHOT_VERIFIED` 和 `HISTORICAL_PROVENANCE_INCOMPLETE`。续训时既有运行身份比较会包含基线SHA及数据来源身份；Stage B1策略合同检查继续拒绝旧Stage6 Checkpoint。本次未生成Run Manifest，也未加载Checkpoint。

### 执行范围与状态

- 未修改或重新生成任何Episode数据；旧Episode Manifest和Pool Manifest保持原样。未读取Validation/Test，未启动训练。除唯一基线文件外未新增训练池持久文件，临时文件已清理。
- `conda` 环境列表工具因Windows系统DLL重定位错误无法启动，常见Conda环境目录下没有 `adaptive_cache_mec`。检查使用现有 `C:\software\anaconda3\python.exe`（Python 3.12.7）；未安装或升级依赖。
- 修改范围限于 `scripts/training/mainline_runner.py` 的显式一次性基线建立/只读验证、`src/ppo_simple/reporting.py` 与 `src/ppo_simple/trainer.py` 的Run Manifest来源记录。没有改系统模型、Reward、Matching/KKT或PPO学习算法。AST解析、TOML解析和 `git diff --check` 通过。

```text
STAGE_B1_CODE_MODIFICATION = COMPLETE
TRAINING_POOL_SNAPSHOT_BASELINE = PASS
BASELINE_EPISODE_COUNT = 300
BASELINE_FILE_COUNT = 3002
POOL_MANIFEST_VERIFICATION = PASS
EPISODE_MANIFEST_VERIFICATION = PASS
CURRENT_FILE_HASH_VERIFICATION = PASS
GENERATION_FINGERPRINT_MATCH = YES
HISTORICAL_GENERATION_ATTESTATION = UNAVAILABLE
RUNTIME_CONFIG_IDENTITY_SEPARATED = YES
TRAINING_POOL_REUSABLE_UNDER_SNAPSHOT_POLICY = YES
OLD_EPISODES_MODIFIED = NO
OLD_MANIFESTS_MODIFIED = NO
EPISODES_REGENERATED = NO
TV_LOCAL_EXECUTION = DISABLED
OLD_CHECKPOINT_COMPATIBILITY = REJECTED
MATCHING_KKT_CHANGED = NO
REWARD_CHANGED = NO
PPO_TRAINING_EXECUTED = NO
```


## Stage B1 最小运行验证（Training Episode 20001）

### 实际环境与数据接口

- 本机 `adaptive_cache_mec` Conda环境仍无法启动（Windows DLL重定位错误，返回码9020）；本机已有解释器为 `C:\software\anaconda3\python.exe`、Python 3.12.7、NumPy 2.2.6。已安装的Numba因NumPy版本不兼容而无法导入（要求NumPy不高于2.0）。本次没有改动或安装依赖；在单次临时进程中选择项目已有的纯Python KKT回退实现。
- 使用当前正式配置加载 `environment_data/proposed_joint_training/episode_20001`，环境上下文和训练协议校验通过。策略Schema为 `adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v3_no_tv_local`；Training origin范围为0–299。
- 正式初始化缓存为9×10，包含5个SV、2个RSU、2个UAV；没有TV缓存节点。读取真实数据并构造Slot 0、1、2：任务数分别为12、14、16，所需实际车辆位置数分别为17、19、21。
- 正式PPO Observation构造接口对这三个Training Slot均返回400维有限值。未加载Checkpoint；由于Observation与环境Schema接口已通过，没有另行实例化随机策略或做Actor前向。

### 候选生成与正式执行结果

| Training Slot | 任务数 | 正式可行目标对数 | Edge目标对数 | 每个任务均含Cloud候选 | Local/TV候选 |
| --- | ---: | ---: | ---: | --- | --- |
| 0 | 12 | 41 | 29 | 是 | 无 |
| 1 | 14 | 42 | 28 | 是 | 无 |
| 2 | 16 | 39 | 23 | 是 | 无 |

候选通过正式 `build_feasible_targets()` 按服务缓存命中与实际通信路径约束生成；逐个Edge候选再次检查了服务缓存位和链路可行性。配置的Cloud模式为 `RESOURCE_AWARE_FALLBACK`，三个Slot中每个任务的候选集合都包含Cloud。

随后用初始合法Cache、双UAV STAY及Slot 0真实任务调用正式 `execute_configured_slot()`。执行在Transfer-Exchange Matching的增量候选枚举中异常中止：

```text
NameError: name '_to_kkt_matching' is not defined
```

`src/decision/task_allocation.py` 中增量路径仍在第500和609行调用 `_to_kkt_matching`，但Stage B1移除Local时删除了该函数。异常发生在初始Matching/KKT评价之后、Transfer-Exchange候选处理阶段。由于Matching没有返回，无法验证最终任务分配、最终CPU/带宽分配、实际Cloud回退选择、时延、能耗或Slot Reward；也不能据候选集合把这些项目报告成运行通过。该阻断来自当前工作区源码中的遗留调用，不是本次Python/Numba依赖阻断。

本次只追加了本节，没有修改源码、配置、训练数据、Checkpoint或其他报告；未运行SUMO、训练或Validation/Test。使用内联命令运行，没有留下临时Runner或测试脚本。完整的2–3槽Slot执行验证需先修复上述Matching源码错误后再运行。

```text
STAGE_B1_ENVIRONMENT_LOAD = PASS
TRAINING_SLOT_ALIGNMENT_0_TO_2 = PASS
PPO_OBSERVATION_INTERFACE = PASS (400 finite features)
LEGAL_EDGE_CANDIDATES_AND_CLOUD_CANDIDATES = PASS
LOCAL_OR_TV_IN_CANDIDATES = NONE
OFFICIAL_SLOT_EXECUTION = BLOCKED
TRANSFER_EXCHANGE_MATCHING = FAIL (undefined _to_kkt_matching)
FINAL_ASSIGNMENT_AND_RESOURCE_CHECKS = NOT_COMPLETED
SLOT_DELAY_ENERGY_REWARD_FINITE_CHECK = NOT_COMPLETED
TRAINING_EXECUTED = NO
VALIDATION_TEST_READ = NO
SOURCE_OR_DATA_MODIFIED = NO
```


## Stage B1 Matching残留调用修复与最小运行复验

### 修复

- `src/decision/task_allocation.py` 的增量路径中，原 `_to_kkt_matching()` 只把旧版 `LOCAL:<task_id>` sentinel 转为KKT内部Local标记。B1模型已移除Local目标后，合法Matching与KKT目标共享同一任务ID→Edge/Cloud节点ID映射，故不再需要此转换。
- 将所有4处残留调用改为 `dict(matching)` 映射快照：当前匹配、候选匹配的预筛选，以及增量精算的当前/候选匹配。键、节点ID和值均保持原样；副本行为保留，未改变候选顺序、增量影响范围、KKT调用或缓存键语义。没有恢复Local常量或分支，既有转换计数仍记录映射准备次数。
- 针对B1移除项检查：`task_allocation.py` 中 `_to_kkt_matching`、`local_target`、`LOCAL_PREFIX`、`LOCAL:` 均无残留引用；文件AST解析和 `git diff --check -- src/decision/task_allocation.py` 通过。

### Training Episode 20001 Slot 0–2

使用正式 `load_environment_context()`、`aligned_train_steps()` 和 `execute_configured_slot()`，初始Cache/UAV状态来自当前环境接口，三个Slot均采用Cache KEEP、双UAV STAY。当前Conda环境仍不可用；使用现有Python 3.12.7 / NumPy 2.2.6，并在本次临时进程选择项目内置纯Python KKT回退，不改变依赖。未加载Checkpoint、未训练，也未读取Validation/Test。

| Slot | 任务数 | SV / RSU / UAV / Cloud分配 | Transfer / Exchange更新 | KKT求解（有限/收敛） | 最大资源残差 | Task Utility | Cache成本 | UAV成本 | Return | 延迟总和（秒） | 车辆能耗总和（焦耳） |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 12 | 0 / 1 / 2 / 9 | 9 / 2 | 205 / 205 / 205 | 9.90e-13 | 11.537621 | 0 | 0 | 11.537621 | 21.425883 | 0.326666 |
| 1 | 14 | 0 / 1 / 2 / 11 | 10 / 0 | 228 / 228 / 228 | 9.91e-13 | 13.431414 | 0 | 0 | 13.431414 | 27.052776 | 0.295889 |
| 2 | 16 | 0 / 0 / 1 / 15 | 8 / 0 | 229 / 229 / 229 | 9.97e-13 | 15.292653 | 0 | 0 | 15.292653 | 35.108181 | 0.150105 |

三个Slot均以 `converged_by_no_improvement` 结束，Transfer-Exchange稳定性复核通过。全部任务恰好分配一次，目标均为合法Edge或Cloud；无TV Local分配（总数0）。每个Edge分配均通过正式候选集合检查，且对应服务已缓存、实际链路可行。CPU和带宽分配均满足每节点份额和不超过1及最小份额约束；所有KKT求解有限、收敛，资源和残差低于本次检查使用的既有求解容差。每任务时延、车辆能耗、效用、资源分配及Slot Return均为有限值，且三槽均满足 `Return = Task Utility - Cache Cost - UAV Movement Cost`。

```text
MATCHING_NAME_ERROR_FIXED = YES
TRANSFER_EXCHANGE_COMPLETED = YES (3/3 slots stable)
KKT_RESOURCE_FEASIBLE = YES
TV_LOCAL_ASSIGNMENT_COUNT = 0
SLOT_REWARD_FINITE = YES (3/3 slots)
TRAINING_EXECUTED = NO
```
