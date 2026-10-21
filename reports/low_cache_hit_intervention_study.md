# 低缓存命中率下 Cache 更新与 UAV 移动的配对反事实实验

## 数据与 Run 门禁

Run：`C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\cloud_shared_resources_logged_controlled_updates2_20261020`；Training Pool：`C:\Users\tangning\Desktop\code\adaptive_cache_mec\environment_data\proposed_joint_training`；冻结 Checkpoint SHA-256：`996677691734b22146b3547a585c16ec411e55eb25f29b04a81f63e2ee398a57`。Checkpoint 处于 8 Episode / 2 次 Update。策略以确定性 `torch.inference_mode()` 执行，不做梯度或优化器更新，也不保存 Checkpoint。

Run 状态为 `CONTROLLED_STOP_AFTER_UPDATES`，目标/完成 Episode=300/8，停止上限/完成 Policy Update=2/2。当前正式配置原始哈希、Run Manifest 哈希、语义哈希、Training Pool Manifest 和严格 Checkpoint 身份均匹配。

2400 条 Slot 日志完整覆盖 Episodes 1–8 的 Slot 0–299；各 Episode Return 与 Slot `G` 汇总一致，最大绝对差 0。Cloud 最终分配及 KKT 汇总日志全量解析：选中上行/CPU违反分别 0/0，KKT 上行/CPU违反分别 0/0，非有限结果 0。Cloud 能力为 20 Mbps、20 GHz；日志内 KKT 标量求解数合计 71848，最大上行分配 2e+07 bit/s，最大 CPU 分配 2e+10 cycles/s。

两个 Update 日志记录的活跃 Actor 顺序为 Micro、Macro；非活跃 Actor 步数均为 0，三个 Critic 每组均有正更新步数。

## 命中率、样本和决策信息

冻结策略在 8 个 Run Training Episode（Seed 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008）上重放 514 个 Macro Cache 决策状态，其中 514 个状态有任务。低命中阈值为有效状态缓存命中率的经验 P25=0.666667；全体有效状态的 min/median/max=0.294118/0.785714/1。阈值只用于本次 Training 样本。

最终抽取低命中 20 个状态，匹配较高命中对照 20 个。匹配先按同 Episode、同 30-Slot 时间段排序，再按任务数量、请求服务直方图、车辆位置中心和 UAV 位置中心接近程度匹配；该匹配在查看任何干预收益前完成。具体每对的匹配距离见 JSON。

主命中率按干预前 Cache 状态和该 Slot 实际任务计算：请求服务存在任意 SV/RSU/UAV Cache 的任务数除以任务数。另行记录 Matching 选中的 Edge 节点是否缓存请求服务、是否存在可行 Edge 路径、实际 Edge/Cloud 分配比例和 Cloud 选择原因。它们不等同于缓存目录覆盖率、Edge 卸载率或相互替代的命中指标。

低命中组基线任务原因统计：{"edge_service_hit": 128, "service_cached_but_feasible_edge_chose_cloud": 13, "service_not_cached_at_any_edge": 137}。分类区分服务未缓存、已缓存但无可行 Edge 路径、存在可行 Edge 但实际选择 Cloud，以及 Edge 命中。

当前策略观察使用保存的预测和配置状态，不含 `actual_tasks`。基线使用冻结策略实际确定性动作；Cache-only 固定其 UAV 位置并枚举合法单节点 UPDATE，UAV-only 固定其 Cache 状态并枚举合法单 UAV 动作。Oracle 在候选求解之后使用当前实际任务最大化 `G`，因此只是候选子集内的离线 Oracle 上界。冻结策略实际 Cache/Micro 动作也单独与 KEEP/STAY 参考比较。

## 即时配对收益

### 低命中组

| 方案 | ΔG 正/零/负 | 正收益比例 | 平均 ΔG | 中位 ΔG | 平均值 95% CI | Δ任务效用 | ΔCache 成本 | Δ移动成本 | Δ时延 s/任务 | Δ上传能耗 J | ΔCloud 率 | Δ任意缓存命中率 | Δ选中 Edge 命中率 | ΔEdge 可行率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cache-only Oracle | 18/2/0 | 0.9 | 2.69714 | 2.1827 | [1.83517, 3.62247] | 2.6655 | -0.0316407 | 0 | -9.90891 | -2.94602 | -0.273056 | 0.269143 | 0.273056 | 0.269143 |
| UAV-only Oracle | 0/20/0 | 0 | 0 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### 匹配高命中对照组

| 方案 | ΔG 正/零/负 | 正收益比例 | 平均 ΔG | 中位 ΔG | 平均值 95% CI | Δ任务效用 | ΔCache 成本 | Δ移动成本 | Δ时延 s/任务 | Δ上传能耗 J | ΔCloud 率 | Δ任意缓存命中率 | Δ选中 Edge 命中率 | ΔEdge 可行率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cache-only Oracle | 20/0/0 | 1 | 1.71055 | 1.19947 | [1.14034, 2.3257] | 1.67099 | -0.0395534 | 0 | -6.17898 | -2.19744 | -0.176329 | 0.151657 | 0.176329 | 0.151657 |
| UAV-only Oracle | 1/19/0 | 0.05 | 4.76042e-06 | 0 | [0, 1.42813e-05] | 6.10104e-05 | 0 | 5.625e-05 | -0.000106659 | -0.000340499 | 0 | 0 | 0 | 0 |

低命中组每状态 Oracle 最优 Cache 分支的平均新增加载量/下载时延/Cache 成本=20687.8 MB / 827.514 s / 0.0165503；UAV 分支平均移动距离/移动能耗/移动成本=0 m / 0 J / 0。
匹配高命中组每状态 Oracle 最优 Cache 分支的平均新增加载量/下载时延/Cache 成本=14122.6 MB / 564.904 s / 0.0112981；UAV 分支平均移动距离/移动能耗/移动成本=0.75 m / 11.25 J / 5.625e-05。

低命中减匹配对照的 Cache-only Oracle 即时收益差：n=20，均值/中位数=0.986596/0.505073，95% 配对 bootstrap CI=[0.0960446, 1.97113]，正/零/负配对=11/0/9。
低命中减匹配对照的 UAV-only Oracle 即时收益差：n=20，均值/中位数=-4.76042e-06/0，95% 配对 bootstrap CI=[-1.42813e-05, 0]，正/零/负配对=0/19/1。

低命中 Cache-only 候选数=1693，候选收益正/零/负=636/0/1057；UAV-only 候选数=720，正/零/负=0/0/720。低命中 Oracle 最佳动作标签 Cache/UAV={"SINGLE_NODE_UPDATE": 18, "POLICY_BASELINE": 2}/{"POLICY_BASELINE": 20}；高命中对照={"SINGLE_NODE_UPDATE": 20}/{"POLICY_BASELINE": 19, "SINGLE_UAV_ACTION": 1}。对照组完整候选统计及每个候选实际 Cloud 上行/CPU 分配、容量和 KKT 聚合值均保存在 JSON。

正式策略动作的可部署参照（相同策略 UAV 下 Cache 与 KEEP 比较；相同策略 Cache 下 UAV 与 ALL_STAY 比较）：
- 低命中组 Macro Cache 动作 ΔG 均值/95% CI=0.616453/[-0.18695, 1.33878]；Micro UAV 动作=0/[0, 0]。
- 匹配对照组 Macro Cache 动作 ΔG 均值/95% CI=-0.716934/[-1.2531, -0.20599]；Micro UAV 动作=0/[0, 0]。

## 10 Slot 配对回放

对 2 个代表性匹配状态对，Baseline、Cache-only、UAV-only 均从相同干预前 Cache/UAV 快照开始，并使用相同的连续 10 Slot 外生任务和车辆轨迹。首 Slot 后冻结策略恢复决策，Cache/UAV 状态和 Adaptive Holding 按正式状态转移演化。
- `LOW_HIT` cache_only：首 Slot 平均 ΔG=6.0358；10 Slot 累计 Reward 差均值/中位数=9.32235/9.32235，95% CI=[7.50009, 11.1446]（n=2）。
- `LOW_HIT` uav_only：首 Slot 平均 ΔG=0；10 Slot 累计 Reward 差均值/中位数=0/0，95% CI=[0, 0]（n=2）。
- `MATCHED_CONTROL` cache_only：首 Slot 平均 ΔG=3.96166；10 Slot 累计 Reward 差均值/中位数=7.69658/7.69658，95% CI=[0.684046, 14.7091]（n=2）。
- `MATCHED_CONTROL` uav_only：首 Slot 平均 ΔG=0；10 Slot 累计 Reward 差均值/中位数=0/0，95% CI=[0, 0]（n=2）。
- 10 Slot `cache_only` 低命中减匹配对照累计 ΔG：均值/中位数=1.62577/1.62577，95% paired bootstrap CI=[-7.20902, 10.4606]（n=2 对）。
- 10 Slot `uav_only` 低命中减匹配对照累计 ΔG：均值/中位数=0/0，95% paired bootstrap CI=[0, 0]（n=2 对）。
- `LOW_HIT` cache_only 175 起：首 Slot ΔG=5.5604，累计 ΔG=7.50009，Post-action 命中率变化=0.0955882，累计 Cache/movement 成本变化=-0.02473/0，平均时延变化=-3.04585 s，上传能耗变化=-7.0013 J，Cloud 比例变化=-0.0514706。
- `LOW_HIT` uav_only 175 起：首 Slot ΔG=0，累计 ΔG=0，Post-action 命中率变化=0，累计 Cache/movement 成本变化=0/0，平均时延变化=0 s，上传能耗变化=0 J，Cloud 比例变化=0。
- `MATCHED_CONTROL` cache_only 170 起：首 Slot ΔG=4.50476，累计 ΔG=14.7091，Post-action 命中率变化=0.205674，累计 Cache/movement 成本变化=-0.0346648/0，平均时延变化=-5.47916 s，上传能耗变化=-20.6153 J，Cloud 比例变化=-0.177305。
- `MATCHED_CONTROL` uav_only 170 起：首 Slot ΔG=0，累计 ΔG=0，Post-action 命中率变化=0，累计 Cache/movement 成本变化=0/0，平均时延变化=0 s，上传能耗变化=0 J，Cloud 比例变化=0。
- `LOW_HIT` cache_only 65 起：首 Slot ΔG=6.5112，累计 ΔG=11.1446，Post-action 命中率变化=0.191083，累计 Cache/movement 成本变化=-0.0335608/0，平均时延变化=-4.60179 s，上传能耗变化=7.66514 J，Cloud 比例变化=-0.140127。
- `LOW_HIT` uav_only 65 起：首 Slot ΔG=0，累计 ΔG=0，Post-action 命中率变化=0，累计 Cache/movement 成本变化=0/0，平均时延变化=0 s，上传能耗变化=0 J，Cloud 比例变化=0。
- `MATCHED_CONTROL` cache_only 185 起：首 Slot ΔG=3.41856，累计 ΔG=0.684046，Post-action 命中率变化=0.0070922，累计 Cache/movement 成本变化=-0.0471467/0，平均时延变化=-0.314324 s，上传能耗变化=0.931223 J，Cloud 比例变化=-0.0141844。
- `MATCHED_CONTROL` uav_only 185 起：首 Slot ΔG=0，累计 ΔG=0，Post-action 命中率变化=0，累计 Cache/movement 成本变化=0/0，平均时延变化=0 s，上传能耗变化=0 J，Cloud 比例变化=0。
这些 10 Slot 结果是少量代表性状态的连续配对轨迹，不外推为总体长期收益；每一 Slot 的 Reward、Matching/KKT 和 Cloud 资源证据仍来自正式执行器。

## 解释与结论边界

低命中 Cache 触发判定：`INCONCLUSIVE`；低命中 UAV 触发判定：`NOT_SUPPORTED`。判定基于低命中组自身 Oracle 收益和相对匹配高命中组的收益差及其 95% bootstrap 区间，定义见 JSON。
当前代码没有依照低命中率触发 Cache 或 UAV 干预的规则，故 `DEPLOYABLE_PREDICTION_DRIVEN_TRIGGER=NOT_IMPLEMENTED`。现有冻结 Actor 的预测驱动动作已另行报告，但不能把 Oracle 候选收益当成 Actor 已学会的行为。

所有效用都由正式路径计算并校验 `G = task_utility - cache_cost - movement_cost`。Cache-only 与 UAV-only 成本由正式 Slot 结果产生；每个候选和后续重放 Slot 都重新运行完整 Matching+KKT，资源分配违反/非有限值会使本分析失败。

40 个即时配对状态中，Cache-only 与 UAV-only 的最佳枚举动作都没有超过冻结策略基线的状态数为 2；低命中原因与不同主因下最佳单项干预见 JSON：{"edge_service_hit": {"CACHE_ONLY": 6, "NEITHER": 1}, "service_not_cached_at_any_edge": {"CACHE_ONLY": 12, "NEITHER": 1}}。

证据限制：Checkpoint 仅经历 8 Episode / 2 Update，状态来自同一 Training Pool 的训练内回放，候选只覆盖单节点 Cache 更新和单 UAV 动作；无联合动作搜索。Cache 更新立即生效，未模拟下载队列。Oracle 结果只代表枚举范围内的上界，不能证明 MAPPO 收敛或触发策略可部署。

```ini
CURRENT_RUN_RESOURCE_LOG_AUDIT = PASS; 2400 records; selected/KKT violations=0; nonfinite=0; KKT scalar solves=71848
LOW_HIT_THRESHOLD = 0.6666666666666666
LOW_HIT_SAMPLES = 20
MATCHED_CONTROL_SAMPLES = 20
CACHE_ONLY_IMMEDIATE_EFFECT = mean=2.69714; median=2.1827; 95%CI=[1.83517, 3.62247]
UAV_ONLY_IMMEDIATE_EFFECT = mean=0; median=0; 95%CI=[0, 0]
CACHE_ONLY_10_SLOT_EFFECT = 9.322353373339226
UAV_ONLY_10_SLOT_EFFECT = 0.0
LOW_HIT_AS_CACHE_TRIGGER = INCONCLUSIVE
LOW_HIT_AS_UAV_TRIGGER = NOT_SUPPORTED
DEPLOYABLE_PREDICTION_DRIVEN_TRIGGER = NOT_IMPLEMENTED; frozen policy action reference measured separately
MAPPO_TRAINING_EXECUTED = NO
FORMAL_ALGORITHM_MODIFIED = NO
VALIDATION_TEST_READ = NO
```

完整逐状态候选、成本、任务指标、资源分配和 10 Slot 分支见 `reports/low_cache_hit_intervention_study.json`。

## 全部 Training Slot 的任务级缓存与 Cloud 归因

Run 原始 Slot 日志没有保存逐任务服务 ID、选中执行节点或候选得分；它只保存按节点聚合的任务数。以下分类因此来自当前严格身份匹配的 latest Checkpoint `996677691734b22146b3547a585c16ec411e55eb25f29b04a81f63e2ee398a57` 对相同 8 个 Training Episode 的确定性冻结重放，不是原训练时逐任务动作的重建。重放覆盖 33654 个任务 / 2,400 个 Slot；每个 Slot 的 Pool 实际任务数与 Run `node_task_count` 总数一致（2400 / 2,400）。原 Run 实际节点聚合计数另见 JSON `run_actual_execution_node_counts_from_slot_logs`。

定义：A=执行时请求服务不在任一边缘缓存；B=服务已缓存但正式 feasible-target 构造器没有返回 Edge 路径；C=服务已缓存、有正式 Edge 候选但完整 Matching+KKT 选择 Cloud；D=服务命中所选 Edge 节点并由 Edge 执行。feasible Edge 候选按当前缓存、车辆/节点位置、角色、无线链路和正资源条件构造；共同资源分配再由正式 Matching+KKT 评价。

| 类别 | 任务数 | 占全部重放任务 |
|---|---:|---:|
| A：未缓存 | 8338 | 0.247757 |
| B：缓存但 Edge 不可行 | 0 | 0 |
| C：缓存且 Edge 可行，选择 Cloud | 2898 | 0.0861116 |
| D：缓存命中并由 Edge 执行 | 22418 | 0.666132 |

重放 Cloud 任务数=11236。Cloud 任务原因占比按 Cloud 任务为分母：A=0.742079，B=0，C=0.257921。原 Run 日志实际节点聚合为 SV=5360、RSU=6085、UAV=8761、Cloud=13448；冻结 Checkpoint 重放的执行类别则为 Edge=22418、Cloud=11236。两组是不同策略状态的证据，不应混作原 Run 实际逐任务分类。完整逐任务记录和 Episode/Slot/Service/执行节点交叉表均写入 JSON `task_level_attribution`。

#### 按 Episode 分类（冻结 Checkpoint 重放）

| Episode Seed | Tasks | A 未缓存 | B 缓存但 Edge 不可行 | C 可行 Edge 但选 Cloud | D Edge 执行 |
|---:|---:|---:|---:|---:|---:|
| 20001 | 4190 | 1550 | 0 | 194 | 2446 |
| 20002 | 4201 | 689 | 0 | 556 | 2956 |
| 20003 | 4191 | 940 | 0 | 298 | 2953 |
| 20004 | 4146 | 733 | 0 | 561 | 2852 |
| 20005 | 4211 | 1571 | 0 | 203 | 2437 |
| 20006 | 4218 | 789 | 0 | 477 | 2952 |
| 20007 | 4278 | 905 | 0 | 373 | 3000 |
| 20008 | 4219 | 1161 | 0 | 236 | 2822 |

#### 按请求 Service 分类（冻结 Checkpoint 重放）

| Service ID | Tasks | A | B | C | D |
|---:|---:|---:|---:|---:|---:|
| 0 | 3507 | 3127 | 0 | 0 | 380 |
| 1 | 4269 | 0 | 0 | 700 | 3569 |
| 2 | 3906 | 0 | 0 | 448 | 3458 |
| 3 | 2403 | 1238 | 0 | 293 | 872 |
| 4 | 2305 | 0 | 0 | 14 | 2291 |
| 5 | 4692 | 0 | 0 | 25 | 4667 |
| 6 | 2744 | 741 | 0 | 97 | 1906 |
| 7 | 3006 | 203 | 0 | 397 | 2406 |
| 8 | 2619 | 2302 | 0 | 48 | 269 |
| 9 | 4203 | 727 | 0 | 876 | 2600 |

### C 类任务：全 Slot 配对的强制 Edge 代表样本

C 类总量=2898；逐 Episode 最多取一个确定性首个 C 任务（按 Episode、Slot、task ID），并强制到正式节点顺序中的第一个可行 Edge 节点。每次使用同一 Slot 全部任务、车辆位置、活动 Cache/UAV 状态，调用生产 `optimize_task_matching`；其余任务继续参与完整重匹配与 KKT。代表样本 8 个，完整评价成功 8 个，未解决 0 个。

| Episode | Slot | Task | 强制 Edge | ΔG（强制 Edge−Cloud 基线） | Δ全 Slot 时延 (s) | Δ上传能耗 (J) | Cloud 率基线→反事实 | 改变分配的任务数 |
|---:|---:|---|---|---:|---:|---:|---|---:|
| 1 | 3 | 27 | RSU_0 | -0.167972 | -16.6896 | 4.55722 | 0.428571 → 0.357143 | 1 |
| 2 | 0 | 10 | RSU_1 | -0.274728 | 7.87163 | 1.50158 | 0.266667 → 0.266667 | 2 |
| 3 | 1 | 56 | RSU_1 | -0.257342 | -22.0556 | 6.38723 | 0.444444 → 0.388889 | 1 |
| 4 | 0 | 34 | 40 | -0.00552271 | 2.10535 | -0.299349 | 0.333333 → 0.25 | 1 |
| 5 | 0 | 0 | RSU_1 | -0.417438 | 2.82583 | 3.82759 | 0.1875 → 0.125 | 1 |
| 6 | 0 | 4 | 21 | -0.0213968 | 0.988997 | 0.0533255 | 0.333333 → 0.333333 | 2 |
| 7 | 0 | 51 | 45 | -0.0404868 | -6.56874 | 1.52933 | 0.384615 → 0.307692 | 1 |
| 8 | 2 | 22 | RSU_0 | -0.196394 | -12.1486 | 4.08186 | 0.416667 → 0.333333 | 1 |

可评价样本 ΔG 均值/中位数=-0.17266/-0.182183；正/零/负=0/0/8。这些是少量 C 类代表任务的强制目标反事实，不是全局最优或策略可部署性的证明。原 Run 未记录每次候选评分；重放保存的是最终任务效用、时延、能耗、资源分配及该任务参与的已接受 Matching 操作，未记录被拒绝候选的逐任务分数。

### 预测需求与 Trigger / Candidate 信用口径

8 个 Episode 共检查 2,400 个预测 Slot；保存场景包括 `arrival`、`service_id`、`rho_hat`、`horizons`、`positions_xy`、`task_vehicle_ids` 和 horizon mask。有效预测 horizon Slot=2400，存在正预测服务事件的 Slot=2400；首 Slot 张量形状={"arrival": [20, 5, 20], "service_id": [20, 5, 20], "rho_hat": [5, 10], "prediction_horizon_mask": [5]}。因此可构造未来窗口的逐服务需求估计。
正式候选经济特征宽度=19，需求预测窗口=5 个 horizon；候选输入包含预测新增/移除服务需求，以及由 Cache 更新成本、加载量和释放量计算的特征。因此预测需求和缓存下载代价/移除服务需求代理可用；但候选特征不是经 Matching+KKT 校准的未来净系统效用，当前日志也没有每个 KEEP/UPDATE 目标的独立预测净值。净边际效用仍需在一致的预测窗口上，对 KEEP 和候选分别运行正式预测任务 Matching+KKT 并扣除下载成本，不能直接由命中率推出。
当前 Run Macro advantage mode=["formal Macro SMDP GAE; joint Cache and sampled termination PPO"]。每节点同一 categorical head 以 candidate 0 表示 KEEP、candidate >0 表示合法 UPDATE 目标；Trigger 和 Candidate 共用当前 Macro SMDP GAE / PPO 回报信用，不是两个独立评分器。故 `CURRENT_TRIGGER_CREDIT_REASSESSMENT=SHARED_RETURN_CREDIT`，但不等于已有显式 KEEP 对每个 UPDATE 候选的预测净效用比较。历史 `cache_trigger_causal_gate.md` 的 stage2d LR Checkpoint 不是本次交替训练 Checkpoint，不能作为当前策略归因证据。

```ini
TASK_CLASSIFICATION_COVERAGE = 33654 / 33654 tasks; all 2400 slots; frozen latest-checkpoint replay, not original per-task Run assignments
SERVICE_NOT_CACHED_CAUSE = 8338
CACHED_BUT_EDGE_INFEASIBLE_CAUSE = 0
CACHED_AND_EDGE_FEASIBLE_BUT_CLOUD_SELECTED = 2898
EDGE_VS_CLOUD_COUNTERFACTUAL_EFFECT = 8 paired full-slot cases; mean_delta_G=-0.17266; unresolved=0
PREDICTED_DEMAND_DATA_AVAILABLE = YES; valid_forecast_slots=2400/2400
PREDICTED_CACHE_MARGINAL_VALUE_ESTIMABLE = PROXY_ESTIMABLE; calibrated future net system utility is not directly provided by the actor candidate score and requires same-window forecast Matching evaluations.
CURRENT_TRIGGER_CREDIT_REASSESSMENT = SHARED_RETURN_CREDIT; no separately logged predicted net-value comparator for KEEP versus each UPDATE target.
NEXT_REQUIRED_STEP = For any Trigger change, compare KEEP and legal UPDATE candidates over the same valid forecast window with production Matching+KKT, download cost, and evicted-service opportunity cost; validate on later/currently held-out Training realizations before deployment. Do not treat this 8-Episode replay as policy convergence evidence.
```
