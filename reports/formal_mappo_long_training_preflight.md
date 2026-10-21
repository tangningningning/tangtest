# Formal MAPPO 长训练预检

日期：2026-10-20
项目：adaptive_cache_mec
结论：生成数据兼容预检通过，可准备启动 300-Episode 长训练；本轮不启动训练。

## 1. 冻结的归一化参数

正式配置保留：

- T_ref = 30.473728355325758 s
- E_ref = 5.1574642506299275 J
- omega_delay = 0.5
- omega_energy = 0.5

本轮没有改写正式配置。归一化对照记录使用 Training Episode 20001–20012 的 120 个真实 Slot、1642 个任务。按 Task ID 升序重放可复现上述参考值；按 CSV 行序重放不能复现，因此历史样本原始遍历顺序仍未证实。既有标定依据来自历史 TV Local CPU 模型，且当时 Cloud CPU 为 40 GHz。当前 Cloud CPU 为 20 GHz；新配置下的复算不是原参数的历史来源，也不构成重新标定。为保持实验合同，本轮冻结原值并保留这一来源限制。

## 2. 配置与运行合同

正式配置文件：configs/proposed/system_config.toml

- Cloud：finite_pool，共享 CPU 20 GHz；每任务等效上行 20 Mbps；Cloud 时延系数 13。
- 任务效用权重：时延 0.5、车辆能耗 0.5。
- Cache / UAV 成本系数：Cache 0.005，UAV 5e-6。
- Cache 策略：因子化节点动作，自适应持有 1–5 Slot。
- 每次策略更新收集 4 个完整 Episode；每个 Episode 300 Slot。
- 正式训练入口：scripts/training/mainline_runner.py。
- 训练数据池：environment_data/proposed_joint_training。

配置哈希：

- Raw SHA-256：ed4596e8df7f0e7b15144c845bbc2c86f77e0c300c60183ad1dfd215777fc2a5
- Semantic SHA-256：bc9e1d2dff4cf01471a193bce6e76929baf06180bdac2d5dc86c85b3bb8f1e66

正式环境的 Python 解释器已确认是 C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe。Conda 启动包装命令 conda run -n adaptive_cache_mec 在本机报“系统无法执行指定的程序”；直接使用已确认的环境解释器，NumPy、Numba、正式 CLI 均可运行。

拟议的全新运行命令如下。本轮只核实参数，不执行该命令：

~~~text
C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe scripts/training/mainline_runner.py --config configs/proposed/system_config.toml --epochs 300 --run-dir runs/formal_mappo_20ghz_frozen_p95_300ep_v1 --episode-pool environment_data/proposed_joint_training --episode-seed-start 20001 --seed 0
~~~

此合同表示 300 个 Episode（seed 20001–20300），每 4 个 Episode 更新一次，共 75 次计划更新。新输出目录在检查时不存在；命令不使用 resume 或自动补生成数据的选项。

## 3. Training Pool 完整性与生成兼容合同

本轮复用此前已通过的 Training Pool 完整性核验，不重新计算 3002 个 Episode 文件的 SHA-256。既有核验结果为缺失、多余、哈希、文件大小、Slot 数及 schema 错误均为 0；本轮只读取冻结 Pool Manifest、完整性基线和服务目录元数据。

Training Pool 固定为 seed 20001–20300、300 Episode、每个 300 Slot。当前正式配置的生成器指纹为 `4083b917948cd7203972c0edd5b40d2efefd8984dde27089a77da43d460b32f3`，与 Pool Manifest 完全一致。任务数据合同 SHA-256 为 `dc7e83d99c69f1c3ea603c18fe9a2606f0bffe477556e75924e1d8dfd24387c3`，服务目录 `environment_data/proposed_joint_training/service_catalog_500_1000.json` 的 SHA-256 为 `6e8c497ed06d76e1251720ebc926e975a6a5c079fd224acab95ea5fb4799cf13`，均与 Manifest 和完整性基线一致。

生成器指纹覆盖车辆/研究车辆规模、轨迹生成参数、服务与任务生成范围、任务到达和流行度随机合同、预测模型生成参数、预测场景数与采样种子。任务数据合同另外逐字段匹配服务数、输入范围、程序大小范围、计算密度、扰动、到达概率、Zipf 参数、Slot 时长和服务目录身份。Episode 协议由 Pool 的 split/seed/count/slot 字段及正式 `full_episode_randomized` 配置共同约束。

兼容判定不再要求当前完整配置 raw/semantic 哈希等于 Pool 生成时的哈希。Cloud CPU、Reward 归一化、Cache/UAV 成本和 MAPPO 学习率不属于 Episode 生成指纹；这些值仍会进入新 Run 的配置身份，并继续受 Checkpoint Resume 的严格配置哈希校验。历史配置哈希只保留为 Pool 来源记录，不再作为新训练读取既有数据的阻断条件。

## 4. 正式流程核对与短时执行检查

静态检查确认：正式入口委托 src/ppo_simple/trainer.py；决策使用当前状态及已有预测输入，动作确定后才向 execute_configured_slot 提供当前真实任务和车辆位置。Cache 在宏决策边界选择动作并按自适应持有期锁定，UAV 每 Slot 决策；执行使用真实任务 Matching+KKT。Cloud 有限共享 CPU、上行速率和 Cloud 时延因子均由正式配置进入执行合同。任务效用以 T/E 和当前权重计算，系统奖励校验 Q - Cache 成本 - UAV 移动成本，配置成本不重复扣除。每 4 个冻结策略 Episode 收集后进行 PPO 更新；Checkpoint 保存策略、优化器进度、随机状态、schema 与配置合同信息。新运行应使用全新输出目录，不恢复旧 Checkpoint。

使用 Training Episode 20001 做了 3 Slot 的确定性 rollout 接口 smoke check，没有调用 Trainer.update() 或写入 Checkpoint：

- Observation 维度：400；Slot 到达任务数：12、14、16。
- 三个 Slot 的净奖励均为有限值：10.82581031、12.82720194、14.51333682。
- Cache 成本为有限非负值；UAV 移动成本为 0。
- 资源可行性均通过，任务目标节点计数与到达任务数一致。
- 本次新初始化的策略在三个 Slot 都发生 Cache 更新；这只验证执行链路，不代表策略表现。
- 未触发 UAV MOVE，因而 UAV 移动分支仅完成静态路径核对。
- 执行耗时约 4.51 秒，未进行训练更新。

## 5. Runner 门禁与长训练准备

`src/ppo_simple/trainer.py::run_training` 在解析 Episode、创建 Run 目录、写入 Run Manifest 或初始化 Trainer 之前调用 `scripts/training/training_pool_contract.py::validate_formal_training_pool_for_run`。该轻量门禁核对当前正式配置的生成器指纹、任务数据合同、服务目录哈希、固定 Training Manifest 与完整性基线身份、seed 20001–20300、300 Episode/300 Slot 协议；不遍历或重算 3002 个 Episode 文件哈希。缺少 Episode 自动生成选项、非正式 Pool、seed 或 Episode 数不匹配会在 Run 创建前被拒绝。

实际门禁检查：当前正式配置与原 Pool 通过；仅在内存中将任务到达概率从 0.7 改为 0.71 后，生成器指纹发生变化并被明确拒绝。没有写临时配置、生成 Episode 或启动 Rollout。既有 MAPPO 3-Slot Smoke Check 结果予以复用，本轮不重跑。

Resume 路径仍逐项要求 Run Manifest 与当前 raw/semantic 配置哈希、策略 Schema、Checkpoint 版本、数据 Pool 身份和训练进度匹配；本次没有放宽或跳过旧 Checkpoint 兼容校验。冻结的 T_ref/E_ref 来源顺序限制保留为标定记录，不作为本次 Training Pool 生成兼容阻断。

建议的新 Run 命令（仅准备，不执行）：

~~~text
C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe scripts/training/mainline_runner.py --config configs/proposed/system_config.toml --epochs 300 --run-dir runs/formal_mappo_20ghz_frozen_p95_300ep_v1 --episode-pool environment_data/proposed_joint_training --episode-seed-start 20001 --seed 0
~~~

该命令对应 300 Episode、每 Episode 300 Slot、每 4 Episode 更新一次，共 75 次 Policy Update。Run 目录在本轮检查时尚不存在；使用新 Run 从头训练，不 Resume。

## 6. 本轮变更

- 修改 `scripts/training/training_pool_contract.py`：移除完整配置哈希白名单/历史 Git 配置审计门槛，改为精确生成指纹、任务合同、服务目录、固定 Manifest/完整性基线及 Episode 协议检查。
- 修改 `src/ppo_simple/trainer.py`：正式 Runner 在解析 Episode、创建 Run 或初始化训练前执行一次上述门禁，并把兼容合同身份写入 Run 来源记录。
- 更新与旧哈希白名单耦合的两个单元测试断言，改为验证生成合同语义。
- 更新本预检报告；未修改正式配置、Checkpoint 或 Training 数据。
- 复用此前完整性核验与 3-Slot Smoke Check；本轮只执行当前合法配置通过和内存中生成参数变更拒绝检查。
- 未运行 PPO 训练、Validation 或 Test；未提交或推送 Git。
- 未创建临时脚本、临时数据或 Python 缓存。

~~~ini
NORMALIZATION_FROZEN = YES
T_REF = 30.473728355325758 s
E_REF = 5.1574642506299275 J
NORMALIZATION_SOURCE_ORDER_VERIFIED = PARTIAL
TRAINING_POOL_GENERATION_COMPATIBLE = YES
TRAINING_POOL_TASK_CONTRACT_MATCH = YES
SERVICE_CATALOG_MATCH = YES
CONFIG_DIFFERENCES_REVIEWED = YES (generation-affecting contract only)
UNAPPROVED_GENERATION_DIFFERENCES = NONE
RUNNER_ENFORCES_POOL_COMPATIBILITY = YES
VALID_CONFIG_GATE = PASS
INVALID_CONFIG_REJECTED = YES
CURRENT_TRAINING_POOL_FILES_VALID = YES (previous full integrity check reused; no 3002-file rehash)
CURRENT_CONFIG_POOL_COMPATIBILITY = PASS (generation contract)
RUNNER_FORMAL_PRECHECK = PASS
FORMAL_FLOW_SMOKE = PASS (3 Training Slots; no update)
READY_FOR_LONG_TRAINING = YES
BLOCKING_ISSUES = NONE
LONG_TRAINING_STARTED = NO
VALIDATION_OR_TEST_ACCESSED = NO
FORMAL_MODEL_CHANGED = NO
MAPPO_ALGORITHM_CHANGED = NO
TRAINING_DATA_CHANGED = NO
SOURCE_CHANGED = YES
FORMAL_CONFIG_CHANGED = NO
FORMAL_DATA_CHANGED = NO
GITHUB_MODIFIED = NO
~~~
