# 车辆环境与任务数据核查报告（2026-10-24）

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
