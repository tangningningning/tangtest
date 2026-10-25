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
