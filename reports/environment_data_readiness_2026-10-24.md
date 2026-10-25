# 车辆环境与任务数据核查报告（2026-10-24；2026-10-25 补充）

**结论：`ENVIRONMENT_BLOCKED_MISSING_PARAMETERS`**

现有正式数据可被当前 300-slot、独立时隙环境加载；但系统资料没有给出最大执行期限 `K_max`，不能据此验收或生成 `T + K_max - 1` 个正式时隙的车辆尾部轨迹。未生成新交通或任务数据。

## 核查清单

| 状态 | 结果 |
|---|---|
| 已满足 | 沿用 `data/sumo/map/osm.net.xml.gz`，SHA-256=`30379989e0f94505e30ed5ec27449367615847ab4baf1bfe2340d8c405388c7d`；`data_split_manifest.json` 与现行路网哈希一致。未改路网或 RSU。RSU 坐标仍为 RSU_0=(3364.03, 901.06, 10m)、RSU_1=(5179.65, 1491.59, 10m)。 |
| 已满足 | 正式池保留 300 个 Train、30 个 Validation、100 个 Test Episode；种子范围分别为 20001–20300、30001–30030、40001–40100。逐文件核验了 430 个 Episode 的角色、任务和正式期研究车辆轨迹。每 Episode 均为 5 SV + 20 TV，角色不重叠；每辆研究车在 1 秒网格上实际覆盖正式时刻 `[100,400)` 的 300 个状态点，轨迹位置均处于 `node_positions.json` 的路网边界内，且位置随时间变化。 |
| 已满足 | 430 条实际轨迹文件的 SHA-256 均与各自 Episode manifest 一致，彼此唯一，Train/Validation/Test 间无轨迹哈希重复。任务文件哈希也均与 Episode manifest 一致。任务行只含 slot 0–299；TV、到达时隙和时间戳逐项匹配。各 split 到达率为 Train 0.700088、Validation 0.701261、Test 0.699208；任务服务计数与逐 slot `popularity.csv` 计数完全一致；输入扰动为配置范围 ±20%，计算负载扰动为 ±10%。 |
| 已满足 | 正式池清单与现行 DQN 配置兼容性检查通过；抽取每个 split 的 Episode 核验了全部 artifact 哈希及来源元数据。Conda 解释器为 `C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`。现有 loader 成功读取默认 `data/generated` 及 Train/Validation/Test 各一个 Episode，并对齐首个 slot 的 25 个车辆位置与任务请求。 |
| 已满足（原模型时域） | 正式配置 `dt_s=1`、预热 100 秒、任务到达时域 `T=300` slot；现有正式 Episode 的交通仿真为 400 秒，正式决策轨迹为 300 秒。任务生成配置为 Bernoulli 到达概率 0.7，并将请求关联到 TV、slot 和服务类型。 |
| 可通过索引或配置修复 | 当前正式池路径和 split 清单完整；没有发现需要通过索引修复的问题。 |
| 需要补充数据（范围待定） | 当前正式池只包含 300 个正式轨迹 slot，没有 `T` 之后的执行尾段。待 `K_max` 有权威定义后，才能确定需要为哪些场景追加多少轨迹；保留已有合格数据，不重生成整个池。 |
| 缺少必要参数 | 项目配置及系统模型源码没有 `K_max` 或任务硬期限；本地工作区/用户目录未找到 `13.pdf`。不能把独立时隙模型推断成 `K_max=1`，也不能猜测尾段长度。 |

## 数据来源与兼容性

- 当前正式加载使用 `environment_data/proposed_joint_training/`、`proposed_joint_validation/`、`proposed_joint_test/` 下的 CSV、NPZ、角色文件和 Episode manifest。各池现有 episode 数为 300/30/100，保留 430 个，新增 0 个。
- 当前正式配置 `configs/dqn/online_joint_dqn.toml` SHA-256=`3800487e13a5052d218ffb597158c5fdc6ceef3661560014d8a8f179fc2f8833`；`environment_data/data_split_manifest.json` SHA-256=`70e6aea73efb23c929cecd2372d9c155b4aebea25c3bcfb59750c742f686f26d`。Episode manifest 记录各场景 ID、种子、配置指纹及文件哈希。
- `environment_data/continuous_v2/` 的 manifest 状态为 complete，共 131 个原始 SUMO 场景：Train 为一条 90,400 秒连续交通流，Validation/Test 各 30/100 条、每条 700 秒。它们保存 FCD XML gzip、车辆活动和 SUMO demand，不含当前 loader 要求的固定角色、任务请求及预测场景文件，因此未替代正式 realization 池。
- 正式池历史 manifest 指向当前仓库中不存在的 `configs/proposed/system_config.toml`。现有 `training_pool_contract` 对池与当前正式配置的遗留兼容校验通过；Train 的历史生成 attestation 标为不可用。因此当前文件及种子可追溯、可校验，但不声称可从缺失的历史配置完整重建原始交通生成过程。
- Validation 清单用途为 checkpoint/算法超参数选择，Test 为最终未见评估；场景 ID 与轨迹哈希互不重叠。本次没有训练或评价运行。

## 未解决的执行接口问题

`configs/dqn/online_joint_dqn.toml` 将 `execution.cross_slot_task_model` 设为 `independent_per_slot`。当前任务对象由每 slot 的 `tasks.csv` 行重新创建；环境没有跨 slot 的活动任务账本，也没有持续保存剩余上传量、剩余计算工作量及剩余期限的状态。因此，现有执行接口不支持题设要求的跨时隙任务继续执行。车辆离开研究区域时，活动任务的继续、迁移、结算或失败规则也未定义。本任务没有修改执行器或算法。

## 文件变更与后续事项

- 新增本报告：`reports/environment_data_readiness_2026-10-24.md`。
- 未改配置、路网、车辆/任务数据、历史报告、实验结果或检查点；未运行训练、正式评价或全量测试套件。
- 后续需提供《13.pdf》中的权威 `K_max` 数值及单位（秒或 slot）。确认前无法验收扩展执行时域或决定最小轨迹补充量。若之后要求端到端跨时隙执行，还需在其他任务中确定活动任务及车辆离场语义。

本结论仅描述环境资产对当前配置的可读性和已核验范围，不代表整个求解系统已满足论文模型。

## 2026-10-25：新环境生成状态与真实 Episode 验证

本节更新新环境重建后的当前状态；上文描述的是 2026-10-24 的旧环境，不代表当前数据池状态。

### 生成状态和身份

- 全量生成仍在 staging 中运行，未发布为正式训练池。核查时 Train 有 **4/300** 个完整 Episode，Validation **0/30**，Test **0/100**；生成进度为 `generating`，当前正在处理 Train seed `20005`。进度文件计数与各 Episode 目录中 `episode_manifest.json` 的实际数量一致。
- staging split manifest 为 `formal_vehicle_environment_split_manifest_v2`，状态仍是 `generating`；正式目录 `environment_data/proposed_joint_training/` 尚未发布。该清单记录的正式配置 SHA-256 与当前 `configs/dqn/online_joint_dqn.toml` 一致。
- seed `20001` 的 Episode manifest 列出 10 个生成 artifact；逐项 SHA-256 均匹配。其 314-origin `prediction_scenarios.npz`、车辆/流行度预测、轨迹、任务和角色文件均通过哈希核验。任务输入只覆盖到达 origins 0–299；预测场景 origins 为 0–313。
- 服务目录当前只在 staging 的 Train 目录中。其 SHA-256=`ae2b4c2cb7122c3ac6b2bc64d789554495e6158a3467aca3f9019a4cee1162d4`，与 split manifest 和 seed `20001` Episode manifest 一致。正式配置指向的 `environment_data/proposed_joint_training/service_catalog_500_1000.json` 尚未出现；单 Episode 加载使用 staging 的 `generation_config.toml`，其中服务目录路径指向同一哈希文件。
- 使用项目 `load_environment_context` 和 `aligned_execution_steps` 加载 seed `20001`，得到 25 辆研究车辆、20 辆任务车辆、10 种服务和 314 个执行 origins；到达 origins 为 0–299。全池 `training_pool_contract` 尚不能验收，因为完整 300/30/100 数据集和正式路径还未发布。

### Seed 20001 真实执行

- 首次真实执行在 slot 0 报 `NameError: cache_services_added is not defined`。`src/online_joint_dqn/environment_adapter.py` 的缓存服务需求诊断路径引用了未定义的前后服务集合。最小修正为从当前 slot 的执行前后缓存矩阵计算新增/移除服务 ID；该修正只影响诊断信息，不改变 Matching、资源分配或奖励。修正后从 slot 0 重跑。
- 使用正式对齐输入和 `ExclusiveDQNEnvironment`，每个时隙选择现有合法 `KEEP+STAY`。执行完成 314 个连续时隙，最后 slot 为 313。slots 0–299 到达 4,190 个任务，slots 300–313 到达数为 0。
- **任务结算：**成功 4,160，超时 30，末尾活动任务 0；唯一终止任务 ID 共 4,190 个，未发现重复结算。跨时隙执行节点对所有 4,190 个任务保持固定。
- **奖励：**Episode Return=`3940.6937459379133`；逐时隙奖励直接求和=`3940.693745937913`，差约 `3e-13`。314 个时隙均核对 `G(t)=Q(t)-C_ca(t)-C_q(t)`；候选预测评分没有进入实际奖励。KEEP+STAY 下缓存成本和 UAV 移动成本合计均为 0。
- **资源：**314 次 Matching 调用、19,580 次 KKT 求解调用。301 个有任务的时隙通过资源可行性检查。最大 Edge CPU 利用率为 `1.0000000000009714`，最大 RSU 带宽份额和为 `1.000000000000338`，均只超出 1 约 `1e-12`，在本次 `1e-8` 浮点容差内；未发现非有限或负工作量/资源。Cloud 资源使用为 0，因此本次 KEEP+STAY Episode 没有覆盖 Cloud 实际执行负载。
- **因果与隔离：**314 个预测 origins 由已生成预测输入加载；逐 slot 诊断均表明状态特征未使用未来真实数据，候选生成未接收当前真实任务或真实车辆位置。执行过程只读 seed `20001`，没有写入其正式 artifacts。场景生成进程仍在运行；验证期间 staging 完整 Episode 数从 2 增至 4，当前处理 seed `20005`。

### 结论

seed `20001` 的单 Episode 314-slot 端到端执行通过。**当前尚不具备启动 Online Joint DQN 正式训练的条件**：Train/Validation/Test 仅完成 4/300、0/30、0/100，split manifest 仍为 `generating`，正式池及正式服务目录路径未发布。Cloud 实际执行负载也未在本次固定 KEEP+STAY rollout 中覆盖。上文关于旧环境 430 个 Episode 的描述是历史记录，不能作为新环境生成完成的证据。

## 2026-10-25：Checkpoint、Validation 选模与 Test 入口

本节只记录运行时代码合同；上文的数据生成计数是当时的状态快照，本次没有重写或重新验收数据池。

- Checkpoint 合同版本提升为 14。每个完整 Episode 原子更新 `latest_checkpoint.pt`；Episode 50、100、150、200、250、300 生成身份校验后的不可变里程碑；`best_checkpoint.pt` 只由固定 Validation 池的平均 Episode 累计系统效用严格改善时更新，并列保留原 best。恢复训练限定使用同一 Run 的 `latest_checkpoint.pt`，校验源码/配置/训练池/Validation 池身份、Episode 与 Slot 进度、探索率进度、Replay 写入进度及优化器/目标网络/RNG 状态。
- 每完成 50 个 Episode，通过正式 Validation 池的 30 个固定场景运行冻结贪心策略。验证不接收 Replay，不执行优化器更新，不改变探索率或 Agent 随机状态；场景失败时整体中止，不跳过样本。每次结果记录 Episode 边界、策略权重摘要、Validation 场景身份、平均系统效用及成功/超时、生命周期时延、上传能耗和系统成本指标。
- 新增 `scripts/evaluation/evaluate_online_joint_dqn.py`，读取训练完成 Run 中的 best Checkpoint 和正式 Test Manifest，在全部 100 个 Test 场景上以同一环境适配器、执行器和合法动作集合运行 DQN 贪心、KEEP+STAY 与均匀随机合法动作。入口记录 Checkpoint、配置和场景身份，写出逐场景汇总与逐时隙指标；Test 不参与选模。
- 定向检查通过：四个 Python 文件 AST 解析及模块导入；内存 Checkpoint 保存/恢复；不可变里程碑重复保存只复用相同快照；Validation 主指标只在严格改善时更新；内存冻结执行核对未改变梯度步、探索 RNG 或 Replay 写入计数；`git diff --check` 通过。未运行正式 Validation/Test、训练或场景生成。
- 本次没有改写环境生成器及其依赖文件，也没有修改正式配置、数据、Manifest、生成日志或现有 Checkpoint 文件；生成期间的运行进程未停止或重启。由于完整 Train/Validation/Test 池仍需按正式合同验收，且 Cloud `finite_pool` 配置对齐仍是既有待办事项，**本节不宣称正式训练或 Test 评价现在已具备启动条件**。
## 2026-10-25：正式主实验脚本静态审计

### 范围、工作区和进程保护

- 审计开始时分支为 main，HEAD 为 7e766eb1d7a357c1d4655b9f072e6a9d5be81a4a，工作区干净。审计期间工作区出现 src/online_joint_dqn/environment_adapter.py 的未提交修改；只读差异显示该修改在 EpisodeCandidateRebuildAdapter 中新增一个返回空任务快照的静态方法。它不是本次审计产生的；未恢复、覆盖或改写，按要求保留。训练 Run 身份会摘要 DQN 源文件，正式启动前应确认这项改动属于预期源版本。
- 生成进程在审计开始和结束时均在运行：PowerShell → cmd → conda → Python；Python PID 48052 的命令为 scripts/generation/generate_formal_vehicle_environment.py --resume。没有停止、重启或向其发送信号。启动时另观察到一个独立 Python smoke 进程；本次没有对它采取操作。
- 清点到 scripts/ 下 18 个 Python 脚本、1 个 PowerShell 脚本；另有 27 个 src/ Python 模块和 3 个已跟踪的 Stable-Baselines3 示例脚本。仓库中没有其他已跟踪的 .bat、.cmd 或 .sh 启动脚本，也没有已跟踪的 tests/ Python 脚本。
- 本节是静态源码、调用关系、配置及现有报告审计。没有运行 Python 项目入口、导入项目模块、读取或遍历生成中的数据池、运行 SUMO、训练、正式评价或测试套件。下文“代码确认”指静态代码路径成立，不代表已运行验证。

### P0 / P1：进入训练门禁前必须解决的实验合同偏差

| 级别 | 代码确认的问题与证据 | 影响与最小处置 |
|---|---|---|
| P0 | **正式活动生成链采用 1 秒发车间隔，与本任务指定的 6 秒不一致。** configs/dqn/online_joint_dqn.toml:92-97 将 randomTrips 窗口设为 [0,60)，period=1.0；scripts/generation/generate_formal_vehicle_environment.py:214-260 又把 period=1.0 写成硬校验值；scripts/generation/generate_episode_realization.py:511-540 将配置传给 ScenarioConfig，src/scenario/sumo_scenario.py:344-400 最终把该值传给 randomTrips 的 -p 参数。正式 Manifest 也会记录该值（generate_formal_vehicle_environment.py:360-390；generate_episode_pool.py:103-129）。另一方面，src/scenario/sumo_scenario.py:67 的默认值和 scripts/generation/generate_continuous_sumo_dataset.py:55、107-127 的正式连续交通合同均为 6 秒；现有 reports/continuous_v2_demand_6s_validation.md 也建议 continuous_v2 用 6 秒/辆。 | 这是交通输入分布的 6 倍名义发车率差异，会改变研究车辆候选集合、轨迹完整性筛选和训练环境；当前代码不会自行发现“用户要求 6 秒、正式配置却是 1 秒”的差别，因为两个生成合同各自内部都通过。不要把当前生成结果标记成符合 6 秒合同。先确定 6 秒适用于连续流还是独立 Episode 的哪段时间窗，再冻结一致合同并在训练门禁校验。直接把当前 60 秒候选窗改成 6 秒会从 nominal_departure_count_for_interval 得到 10 个候选，而 src/scenario/sumo_scenario.py:636-734 的完整车辆筛选/角色分配要求至少 25 个候选；因此不能只改一个 period 参数。当前生成进程没有被干预。 |
| P1 | **当前正式入口没有采用任务清单要求的连续 Train 窗口切分。** generate_formal_vehicle_environment.py:978-1298 保留了连续流窗口生成实现，但 main 在 1988-1990 对 train、validation、test 三个 split 都调用 _generate_independent_split；_generate_train_windows 当前只有定义，没有从活动 main 调用。该入口声明 POOL_SCHEMA=formal_episode_pool_v3（:57-59），并在最终 Manifest 中写 independent_episode_sumo（:1653-1687）。training_pool_contract.py:760-828 对 v3 明确接受每个 split 的 independent_episode_sumo_v1 来源，因此训练池门禁也不会拒绝这一分布。 | 按本任务列出的“连续 Train 窗口、独立 Validation/Test”合同，这会改变 Train 的交通场景采样单位和时序依赖，属于训练数据设计偏差。若用户已批准三个 split 全部独立生成，这是设计选择而非代码错误；否则主入口应使用连续 Train 流的固定、不重叠窗口，Validation/Test 继续独立生成，并在 Manifest/schema 和门禁中区分。当前代码不能证明这次设计转换已获批准。生成完成后应先由启动门禁拒绝/要求确认，不应直接进入正式训练。 |

### P2：代码明确显示的恢复、身份或诊断风险

| 级别 | 证据与结论 | 影响与建议 |
|---|---|---|
| P2 | **生成计划的源码指纹没有覆盖全部实际影响输出的依赖。** generate_formal_vehicle_environment.py:397-407 只摘要了正式生成器、pool/realization 脚本和 task_scenario、popularity_prediction、scenario_generator、sumo_scenario。活动路径还调用 common/project_config.py（协议和配置解析）、scenario/realization.py（seed 派生）、scenario/vehicle_order.py（残差轴合同）及 scenario/vehicle_prediction.py（Kalman replay 和车辆预测；scenario_generator.py:28-30、240-368）。Episode Manifest 中记录的 Git SHA/dirty 状态不会替代阶段计划的逐文件差异校验；续跑核验使用计划中的 generation_source_sha256（generate_formal_vehicle_environment.py:631-653、756-804）。 | 在部分 Episode 已生成后，这些未纳入摘要的 helper 若发生修改，计划指纹可能不变，续跑就可能接受由不同源码生成的 Episode，造成单个池混合算法版本。将所有输出影响依赖加入计划和 Episode 身份摘要；修复前避免跨源码版本续跑。 |
| P2 | **失败和未完成 Episode 没有通用的自动恢复路径。** generate_formal_vehicle_environment.py:453-480 对已标记 failed 的同一计划明确拒绝 --resume；:1410-1438 对已有非空 Episode 目录但缺少 episode_manifest.json 也拒绝覆盖；generate_episode_pool.py:162-253 逐个调用 realization，失败时不把半成品变成完整 manifest。当前 _get_or_create_stage 中的自动重试只针对一个被识别的旧 pilot 错误（:483-505、604-655）。 | 失败会安全地停住且不覆盖数据，但 caught failure 或硬中断留下的半成品可能需要人工识别和受控修复，不能把 --resume 理解为通用断点恢复。可增加逐 Episode 暂存目录/完成标记和明确的失败恢复协议；当前勿手工清理正在生成的目录。 |
| P2 | **最终发布可回滚普通异常，但跨目录发布不具备进程/断电原子性。** generate_formal_vehicle_environment.py:1724-1779 先逐个 os.replace Train、Validation、Test 目录，再移动中心 Manifest；只有 Python 捕获到异常时才反向移动。若进程在移动中被强制终止，可能留下部分正式目录、缺少中心 Manifest；下次 _publish 会因目标目录已存在而拒绝覆盖。 | 这是由多次独立文件系统移动造成的恢复风险，静态代码证明存在暴露窗口，但本次没有运行或注入故障。用发布事务日志/可恢复阶段标记，或以单个版本目录加最后原子切换指针来完成整体提交。 |
| P2 | **Cloud 任务集合变化时，增量 Matching 的 Cloud 资源统计可能只包含受影响任务。** task_allocation.py:4276 识别 cloud_task_set_changed；:4475-4480 只把 impacted_task_ids 中的新 Cloud 任务放入 non_edge_tasks；:4515-4531 随后清空/替换 Cloud uplink/CPU map，并以这批任务重建 used 统计，未补回不受影响但仍留在 Cloud 的任务。 | 影响增量路径的 Cloud 资源明细和用量统计，可能令诊断/资源报告低计数；不影响当前跨时隙主实验：optimize_task_matching 在 task_allocation.py:1070-1072 发现 lifecycle metadata 后强制 evaluation_mode=full。应在未来继续使用非 lifecycle incremental 路径前修正，或让增量结果合并未受影响 Cloud 任务的固定 Rc/Fc。 |

### 已按主调用链核查的合同结论

- 划分及种子：正式生成器在 generate_formal_vehicle_environment.py:65-66、154-173 规划 Train 300 个、Validation 30 个、Test 100 个不同 seed；training_pool_contract.py:349-370 检查 split seed 无交集及数量。scenario.realization.derive_episode_seeds 为 Episode 派生 namespaced 随机流。静态未发现 seed 重叠或流名重复缺陷。
- 时间轴：配置明确 T=300、K_max=15、执行时域 314、末执行 slot=313（configs/dqn/online_joint_dqn.toml:511-523）；project_config.py:109-220 校验 execution=T+K_max-1、rollout 完整覆盖和末 slot。正式生成器同样校验 300/15/314（:214-228）。新到达只覆盖 300 个 slot，后续 14 个 slot 用于执行尾段。静态合同一致。
- 研究车辆筛选：sumo_scenario.py:636-734 只保留完整正式轨迹且至少有一条预热记录的车辆，再由固定 seed 选择 25 辆、其中 5 SV + 20 TV；正式生成器 :262-268、:676-719 校验该数量和任务行数。路径内完整车辆不足时会失败，不会静默降低数量。候选供给矛盾已在 P0 项说明。
- 因果预测：scenario_generator.py:308-317 对 origin 使用 warmup_step+origin-1 的状态 cutoff；vehicle_prediction.py:721-829 先逐步 Kalman replay，再只取 cutoff 之前最近 posterior，并纯用状态转移预测未来，不接受未来真值参数。popularity_prediction.py:231-270 用不含 origin 当期数据的历史窗口。scenario_generator.py:741-799 读取 truth_index 仅计算预测与真实距离的 sanity 汇总；该真值没有传入决策观测或候选生成。tail origins 由场景生成器延长到 execution horizon 并按 mask 描述有效预测。未发现已证实未来信息泄漏。
- Manifest、数据哈希和写入：正式计划、split/pool/integrity Manifest、Episode Manifest 和产物摘要均存在；Episode 产物集和 SHA 会在 resume/训练门禁逐项核验，配置/输入/seed/split 身份有显式记录；已有正式目标路径存在时 _publish 拒绝覆盖。JSON/CSV/Checkpoint 多处使用临时文件加 replace。已发现的源码摘要缺口及整体发布/失败恢复风险见 P2 表；没有读取生成中的数据或验证当前落盘 Manifest。
- DQN 动作和目标：agent.py:328-426 在探索、贪心选择中只从传入的合法候选集合选动作；ReplayBuffer.add 校验当前动作与合法候选一致、下一候选状态指纹一致（replay_buffer.py:370-501）；agent.py:449-507 在线网络对重建的合法下一动作选 argmax，目标网络评估所选候选，终止转移只用即时奖励，非终止 bootstrap 用 gamma^duration。正式 gamma=1、每个 transition duration 为 slot 计数，有限 314-slot episode。没发现 bootstrap 或 action mask 的代码错误；探索按可用 action 类别均衡抽取是显式策略设计，不等同全候选均匀随机。
- DQN 参数与非有限保护：正式入口在 run_training.py:1344-1402 将 epsilon 从 1.0 到 0.05 线性覆盖 300×314 步，gamma=1；TrainerConfig 使用 batch=64、learning_starts=1000、每个训练 slot 一次更新；replay 默认容量 10000；target 网络每 500 个梯度更新硬同步；agent 对 loss/梯度/TD target/Q 值执行 finite 检查并裁剪梯度（agent.py:449-623，trainer.py:878-908）。参数比例从静态代码看一致。没有实际训练，不能据此保证收敛或数值稳定。
- 状态与互斥动作：candidate_features.py:491-552 编码当前正规观测、有效预测、预测负载、场景需求和生命周期任务状态；adapter.py:990-1109 用当前真实任务构造状态，但给候选生成器传入 tasks=()、vehicle_positions={}，候选特征使用决策预测。缓存与 UAV 操作类别由 ActionCandidate 和 adapter 校验为互斥；候选只使用当前配置、预测和受限动作集合。没有静态证据表明状态包含未来真实任务/车辆位置。adapter.py:2063-2081 的 cache_services_added 当前从实际 cache diff 计算，没有发现旧报告提及的重复/错误统计。
- Matching、KKT 与任务生命周期：formal adapter 用 lifecycle_mode=True；task_allocation.py:1070-1072 对带生命周期任务强制完整 Matching。Edge 的 _allocate_phase_resources_kkt（:2981-3153）仅为 UPLOADING/NEW 任务解带宽，为 COMPUTING/NEW 任务解 CPU；Cloud 的 _allocate_cloud_fixed_resources（:3428-3480）使用逐任务固定 Rc/Fc，不做共享池 KKT。task_lifecycle.py:127-232 将新任务初始化为上传阶段、给任务绑定首次执行节点；:242-363 每 slot 扣减剩余工作/期限，只允许上传完成后计算，并只产生一次 SUCCESS 或 TIMEOUT。slot_environment.py:107-189 按任务阶段结算实际带宽/CPU 工作量。没有发现真实完成效用重复支付；candidate_matching 的预测 terminal utility 被标注为非实际 reward（slot_environment.py:307-329），adapter 只把生命周期终止事件的 utility 计入实际奖励（environment_adapter.py:1977-2000）。
- Validation/Checkpoint/Test：run_training.py:61-69 将验证边界及里程碑固定为 50、100、150、200、250、300；:652-723 保存 agent、optimizer、target 网络、Replay、Episode/slot 进度和 RNG 状态并原子落盘 latest；恢复时校验身份与计数（:273-375、1480-1609）。Validation 使用 pool_identity 中同序固定 30 seed，不更新 Replay/梯度/探索状态；best 只按 mean_episode_system_utility 严格改善更新（:856-1105）。Test 入口只加载完成训练 Run 的 best Checkpoint，校验 SHA、身份及 100 个 Test seed，不参与选模，输出目录必须新建（scripts/evaluation/evaluate_online_joint_dqn.py:79-210、213-328）。evaluation.py:156-190、193-337 提供同一环境、DQN greedy、唯一 KEEP+STAY 和对全部合法候选均匀随机三种策略；:386-481 要求每个固定场景完整成功，不静默跳过。静态未发现 Test 选模泄漏或策略执行环境不一致。

### 当前生成任务会读取/依赖的源码边界

活动命令 generate_formal_vehicle_environment.py --resume → _generate_independent_split → generate_episode_pool.generate_pool → generate_episode_realization.run_episode → sumo_scenario、vehicle_prediction/scenario_generator、popularity_prediction、task_scenario、project_config/realization/vehicle_order。它写入 staging、每个 Episode 的产物和最终 split/pool/integrity Manifest。生成任务不调用 DQN 训练、Test 评价或分析脚本。此依赖集包含 P2 所列的未纳入生成源码摘要 helper；审计没有修改其中任何文件或数据。

Online Joint DQN 正式调用链为 pyproject.toml 中 online-joint-dqn → src/online_joint_dqn/run_training.py → scripts/training/training_pool_contract.py、common/environment_interface/project_config/system_model、decision/cache_configuration/slot_environment/task_allocation/task_lifecycle/uav_configuration 及 online_joint_dqn 下的 agent、adapter、candidate、replay、trainer、evaluation 模块。四个 scripts/training/run_online_joint_dqn_lr_*.py 是同一共享训练入口的学习率包装器。scripts/evaluation/evaluate_online_joint_dqn.py 是分离的 Test 入口。

### 全脚本清单与逐文件结论

下表逐项列出全部 scripts/ 脚本。正式主线指当前活动生成命令、训练入口或预先定义的 Test 入口；其他脚本列为离线诊断或一次性工具。结论是静态检查结果，未运行。

#### scripts/generation 与项目启动辅助文件

| 文件 | 用途、调用与输出 | 结论 |
|---|---|---|
| scripts/generation/activate_cross_slot_replacements.py | 一次性替换/补充 formal pool 的工具；读现有 pool 和 reports/cross_slot_load_feasibility_2026-10-24.json，提供 --apply 后写审计快照、替换场景及索引。 | 非当前主线；正式生成器不调用。代码先保存旧场景摘要并核验补充场景，但会改正式数据索引；审计未运行。不要与正在生成的 staging 同时使用。 |
| scripts/generation/extend_formal_episode_trajectory.py | 显式指定 Episode/seed/split 的一次性轨迹延长工具，复用 SUMO、Episode realization 和 seed 派生，改写轨迹及相应 Manifest。 | 非当前主线；main 不调用。不是通用训练入口；未发现与当前生成器的自动冲突防护可以替代正式门禁，未运行且不建议对活动 staging 使用。 |
| scripts/generation/fast_validate_sumo_fcd.ps1 | 流式扫描 FCD 时间步、车辆观测及时间范围，写临时 timeline/observed 清单。 | 仅由 run_continuous_sumo_with_teleport_aware_validation.py 调用；不属于当前独立 Episode 主生成链。 |
| scripts/generation/generate_continuous_sumo_dataset.py | 按连续路线生成/恢复 SUMO 数据集、校验轨迹和 Manifest；自身正式 period 常量为 6 秒。 | 非当前正式 main 的数据生产者；由 run_continuous_sumo_with_teleport_aware_validation.py 直接使用，generate_formal_vehicle_environment.py 仅调用 nominal_departure_count_for_interval 计算清单数量。其 6 秒合同与当前独立 Episode generator 的 1 秒偏差见 P0。 |
| scripts/generation/generate_episode_pool.py | 根据 seed 批量调用 generate_episode_realization.run_episode，校验已完成 episode 及 independent traffic provenance。 | 当前正式生成主线直接依赖；生成器显式传 origin_count=314 和 split。未发现会静默覆盖已完成 Episode 的路径；部分失败恢复限制见 P2。 |
| scripts/generation/generate_episode_realization.py | 单 Episode 生成：SUMO、研究车辆/角色、任务、流行度/车辆预测场景及 Episode Manifest。 | 当前正式生成主线直接依赖；调用 scenario、SUMO 和配置模块并写 Episode 产物。因果 cutoff 与产物身份静态检查未发现 P0/P1；生成源码摘要覆盖不全见 P2。 |
| scripts/generation/generate_formal_vehicle_environment.py | 规划 seed、stage、验证 episode/pool 并发布三个 split；当前运行命令是 --resume。 | 当前唯一正式环境生成入口。明确要求 1 秒及独立 episode，并让所有 split 走独立生成；P0/P1 合同偏差及 P2 恢复/发布风险见前表。 |
| scripts/generation/run_continuous_sumo_with_teleport_aware_validation.py | 外层连续 SUMO 恢复及 teleport-aware FCD/summary 校验；调用 continuous generator 和 PowerShell scanner。 | 非当前正式环境入口；没有从正式 generator main 调用。输出和独立调用链不混入 DQN Train 池。 |

#### scripts/training、scripts/evaluation

| 文件 | 用途、调用与输出 | 结论 |
|---|---|---|
| scripts/training/run_online_joint_dqn_lr_1e_minus_3.py | 学习率 1e-3，run prefix online_joint_dqn_lr_1e-3；委托共享 run_training.main。 | 正式训练包装器；没有独立算法实现，不能视作与共享训练循环不同的模型。 |
| scripts/training/run_online_joint_dqn_lr_1e_minus_4.py | 学习率 1e-4，run prefix online_joint_dqn_lr_1e-4；委托共享 run_training.main。 | 正式训练包装器；输出由共享入口创建唯一 Run 目录。 |
| scripts/training/run_online_joint_dqn_lr_2_5e_minus_4.py | 学习率 2.5e-4，run prefix online_joint_dqn_lr_2.5e-4；委托共享 run_training.main。 | README 推荐的共享正式训练入口变体；无代码发现。 |
| scripts/training/run_online_joint_dqn_lr_5e_minus_4.py | 学习率 5e-4，run prefix online_joint_dqn_lr_5e-4；委托共享 run_training.main。 | 正式训练包装器；无代码发现。 |
| scripts/training/training_pool_contract.py | 训练/验证/测试 Manifest、split seed、场景/文件 SHA、配置和 Episode provenance 校验；由 run_training 及 evaluation 导入。 | 正式主线关键门禁。兼容 v3 independent Episode 数据，因此不会拦截 P1 的 Train 采样设计偏差；除此以外未发现静态漏检的数据哈希或 split 交叉问题。 |
| scripts/evaluation/evaluate_online_joint_dqn.py | 加载完成训练 Run 的 best Checkpoint，对 Test 100 场景执行 DQN/KEEP+STAY/均匀随机合法动作，原子写 evaluation Manifest/逐集逐 slot 指标。 | 正式 Test 入口，与训练和 Validation 分离；只读 best、不选模、输出目录拒绝覆盖。未运行。 |

#### scripts/analysis

| 文件 | 用途、调用与输出 | 结论 |
|---|---|---|
| scripts/analysis/analyze_training_environment_diversity.py | 读取 Train pool、Manifest、配置和 catalog，流式核对轨迹/任务/角色/流行度多样性，输出固定 Markdown/CSV 报告。 | 离线诊断，不被训练或生成入口调用；扫描完整 Train pool，报告路径已有时拒绝覆盖。未运行，避免遍历生成中数据。 |
| scripts/analysis/matching_cloud_optimality.py | 从 Train pool 取固定场景/slot，比较 Cloud/Edge、固定 assignment 和精确子集对照，调用 task_allocation.evaluate_matching_with_kkt，写报告。 | 离线算法诊断，不参与策略决策或训练；计算量可能大，未运行。增量 Cloud 摘要风险只在前述非 lifecycle incremental 路径。 |
| scripts/analysis/matching_kkt_training_efficiency.py | 从正式 Train pool 抽取匹配负载，重复测量 Matching/KKT 运行时间与求解量，写 JSON/Markdown。 | 性能诊断，不被正式入口调用；脚本会读取场景并执行 solver，未运行。 |
| scripts/analysis/online_joint_dqn_counterfactual.py | 根据指定 step record 和 environment-provider 恢复前状态，评估冻结 slot 的替代候选，输出 JSON 或 stdout。 | 离线单 slot 反事实工具，不是训练/正式评价入口；要求调用者显式提供环境工厂，文件输出用独占创建。未运行。 |
| scripts/analysis/online_joint_dqn_decision_audit.py | 只读 DQN step_metrics，汇总 Q、操作、奖励及资源指标，可输出 Markdown/JSON 到 stdout。 | 离线日志分析，不创建正式 Checkpoint/环境，也不被训练调用。未发现会重计正式 episode 的路径；未运行。 |

#### 额外已跟踪的 SB3 示例文件

| 文件 | 用途、调用与输出 | 结论 |
|---|---|---|
| .agents/skills/stable-baselines3/scripts/custom_env_template.py | Gymnasium 网格环境模板。 | vendor 示例，不是 adaptive_cache_mec 正式入口；未被 scripts/src 调用。 |
| .agents/skills/stable-baselines3/scripts/evaluate_agent.py | Stable-Baselines3 通用 PPO checkpoint 示例评价器。 | vendor 示例，不读取本项目 Online Joint DQN Run，也不参与本项目 Test。 |
| .agents/skills/stable-baselines3/scripts/train_rl_agent.py | Stable-Baselines3 通用 PPO 示例训练器。 | vendor 示例，不是本项目训练入口，不与 Online Joint DQN 共用数据/Checkpoint。 |

### src/共享模块逐文件结论

这些共享模块按上述生成、训练和评价调用关系核查一次。除前述明确风险外，没有静态确认会使正式主实验结果无效的额外缺陷。

#### common 与 decision

| 文件 | 用途/主要调用关系 | 结论 |
|---|---|---|
| src/common/environment_interface.py | 加载 realization、按时间轴对齐数据，构造正式观测/slot 输入；由 environment_adapter 和训练池工具调用。 | 正式观测源包含当前时隙及预测 mask；与 314-slot 执行器对齐。未发现输入顺序或尾段数缺陷。 |
| src/common/project_config.py | 加载配置、get_episode_data_protocol、cross-slot 与评价参数校验；生成器、训练池合同、环境及决策代码共享。 | 300/314/K_max、终止 slot 和配置边界检查明确；其源码未进入生成计划 hash 属 P2。 |
| src/common/system_model.py | 通信、Edge/Cloud、cache/UAV 成本、任务效用和 lifecycle reward；decision 与 adapter 调用。 | Cloud 单任务执行参数、terminal task utility 和系统成本分开；adapter 将实际终止 utility 一次计入奖励。未发现重复结算代码。 |
| src/decision/cache_configuration.py | cache 节点/容量、合法缓存配置、预测覆盖/成本；环境和候选编码调用。 | 候选由合法容量/替换合同检查，活动任务服务受保护；未发现当前路径绕过合法性。 |
| src/decision/slot_environment.py | execute_configured_slot/execute_slot，连接 cache/UAV、Matching/KKT、实际 slot 工作量和结果摘要。 | 上传使用带宽份额、计算使用 CPU 份额；统计 requested/actual 变更分开，Matching 分数明确非 actual reward。未发现已确认的 slot 奖励缺陷。 |
| src/decision/task_allocation.py | Transfer–Exchange Matching、完整/增量候选评价、Edge KKT 和 Cloud 固定资源分配。 | 主 lifecycle 路径强制完整评价；Edge 阶段资源和 Cloud 固定 Rc/Fc 符合当前实现合同。存在增量 Cloud map 诊断统计风险（P2）。 |
| src/decision/task_lifecycle.py | prepare_slot/advance_slot 管理任务到达、固定执行节点、剩余上传/计算/期限和终止事件。 | 每任务最多一次 SUCCESS/TIMEOUT；不允许未传完先计算、迁移或到达尾段新增任务。未发现重复终止/资源工作量错位。 |
| src/decision/uav_configuration.py | UAV 合法候选、飞行边界、安全距离、移动范围和实际移动摘要。 | adapter 在执行前重新检查边界、安全距离和过渡合法性；无实际代码缺陷证据。 |

#### online_joint_dqn

| 文件 | 用途/主要调用关系 | 结论 |
|---|---|---|
| src/online_joint_dqn/__init__.py | 包标识/导出。 | 无独立实验执行逻辑。 |
| src/online_joint_dqn/agent.py | 类别与候选 Q 网络、legal action 选择、Double DQN 目标、优化及 Agent state 保存。 | 合法 mask、terminal bootstrap、有限值检查、梯度裁剪和 target 同步路径静态一致；无已确认 P1。 |
| src/online_joint_dqn/cache_optimizer.py | 基于预测需求生成缓存目标候选并执行预算控制。 | 由 environment_adapter 调用；动作后续经过合法候选和互斥校验。未发现主实验绕过验证的路径。 |
| src/online_joint_dqn/candidate_evaluator.py | SlotInput、预测输入和候选 utility/缓存的评估上下文。 | 候选评价使用预测与冻结系统模型，实际状态由 slot executor 单独推进；无已证实未来真值输入。 |
| src/online_joint_dqn/candidate_features.py | ActionCandidate 合同、state/candidate 特征编码、候选 ID 和状态 fingerprint。 | 状态涵盖当前 cache/UAV、预测、场景负载和活动任务快照；候选特征使用预测时域，不含 future truth。 |
| src/online_joint_dqn/environment_adapter.py | 正式 Episode adapter、候选隔离/恢复、step reward 和 Observation。 | 核心路径静态符合 task lifecycle。当前工作区有未提交变化（2405 行附近新增 rebuild adapter 的空快照方法），已原样保留；运行身份将捕获文件身份。 |
| src/online_joint_dqn/evaluation.py | 固定 split/policy seed，冻结 episode 评价和汇总。 | 仅 greedy agent、不改变梯度/Replay；KEEP+STAY 与 uniform_random_legal 复用同一场景和 executor，场景失败整体报错、不跳过。 |
| src/online_joint_dqn/plot_training.py | 从训练产物生成曲线及诊断图。 | 被 run_training 异步绘图调用，不更新 agent/checkpoint；无策略反馈进入训练的路径。 |
| src/online_joint_dqn/replay_buffer.py | 环形 Replay、转移和压缩候选重建上下文持久化。 | add 校验 state/action/next state 与合法候选及 fingerprint 对齐；终止转移不允许带 bootstrap 状态；存取恢复有 schema 约束。 |
| src/online_joint_dqn/run_training.py | pyproject 官方 entrypoint；身份、池门禁、初始化、resume、validation、checkpoint、episode 主循环。 | 固定 run 身份、唯一 Run 目录、原子 latest、六个里程碑、Validation best 规则均可从代码追踪。池门禁未约束用户指定的连续 Train 来源，见 P1。 |
| src/online_joint_dqn/trainer.py | 训练 Episode 的 action→executor→Replay→gradient update、episode 汇总。 | current/next transition 由同一执行 slot 对齐；1-step duration 和 epsilon global step/target update 计数显式记录。无已确认 transition 错位。 |
| src/online_joint_dqn/uav_scheduler.py | UAV 合法移动目标候选与预测需求打分。 | 只产出候选，执行前由正式 UAV validator 再检；与 cache 动作互斥由 adapter 控制。 |

#### scenario

| 文件 | 用途/主要调用关系 | 结论 |
|---|---|---|
| src/scenario/popularity_prediction.py | 流行度 GRU 冻结推理、过去 history 输入和 formal prediction 输出。 | build_causal_formal_inputs 排除 origin 当期/未来历史；配置和 checkpoint shape 校验明确。 |
| src/scenario/realization.py | Episode artifact 文件集合和 seed stream 派生。 | Train/Validation/Test 使用不重叠的 base seed 范围，再派生不同 named RNG stream；该模块未进入 generator source hash。 |
| src/scenario/scenario_generator.py | 调用 vehicle_prediction、流行度预测、任务采样及 residual block，生成 314-slot prediction scenarios。 | vehicle cutoff 为 origin-1；预测以状态模型/residual bank 生成。真实轨迹只用于 sanity 比较，不写成正式未来输入。 |
| src/scenario/sumo_scenario.py | SUMO demand、仿真、轨迹完整性筛选和 seeded role assignment。 | route/FCD 校验和拒绝不足候选逻辑明确；默认 6 秒与正式 DQN 配置 1 秒的主线冲突见 P0。 |
| src/scenario/task_scenario.py | 服务目录、任务到达/服务/输入量/周期量生成和序列验证。 | 正式生成器核验 task ranges、服务数和 20 TV×300 到达表；seed 独立。未发现静默缺 task 路径。 |
| src/scenario/vehicle_order.py | 车辆/残差轴稳定排序和残差采样合同。 | 调用方校验 vehicle mapping 与轴顺序；未发现当前排序错位证据。其源码不在 generator source hash 中（P2）。 |
| src/scenario/vehicle_prediction.py | SUMO route geometry、真值/噪声观测、Kalman replay、cutoff posterior 和 nominal prediction。 | state_at_cutoff 只取不晚于 cutoff 的 posterior；predict_nominal 不接收 future truth。其源码不在 generator source hash 中（P2）。 |

### 修复优先级、当前不确定项及验证边界

1. **P0：先冻结交通合同。** 在训练门禁前裁定 6 秒的时间范围/场景采样单位，协调 60 秒独立 Episode 候选窗、25 辆完整车辆筛选与 6 秒流率；随后只按获批合同更新配置、生成器硬校验、pool provenance 和训练门禁。审计本轮没有改配置，也没有干预生成。
2. **P1：裁定 Train 场景采样设计。** 确认 Train 是否必须是连续 SUMO 流的固定窗口；如是，将活动 main 与 pool contract 改为连续 Train + 独立 Validation/Test。如果独立 Train 已批准，应在实验设计与报告中明确记录，避免误称连续窗口协议。
3. **P2：完善生成身份和恢复提交。** 将所有生成输出依赖纳入计划哈希；为单 Episode 失败、硬中断和三 split 最终发布提供显式可恢复协议，再进行小范围恢复验证。
4. **P2：修正增量 Cloud 结果合并。** 在任何非 lifecycle incremental Matching 诊断被用作正式资源统计前，保留未受影响 Cloud 固定资源并复核 full 与 incremental map/utility 一致性。

**设计选择而非已确认缺陷：** epsilon 探索时按可用操作类别先均衡抽取、Matching 使用 transfer-exchange 局部稳定而非全局最优、Cloud 使用逐任务固定 Rc/Fc、Validation 每 50 集运行。这些在当前代码/元数据中有明确标注；静态证据不足以判定为错误。

**无法静态确认：** 当前生成任务实际写到哪一个 Episode、现有半成品是否可续跑、Manifest 与 SHA 是否和产物一致、完整池是否通过门禁、KKT 在目标硬件和真实任务载荷下的数值收敛性、模型是否最终收敛、六小时之外交通负载是否稳定。它们需要运行态或真实数据验证；本轮按要求未读取生成数据、未运行 Python/SUMO/训练/评价/测试。

**审计产物：** 本节是本轮唯一报告更新；未创建每脚本独立报告或临时脚本，未修改算法、参数、配置、Manifest、数据、日志、Checkpoint 或生成代码。静态审计不等同端到端验证通过。