# Stage B3-7：Cache候选价值代理与自适应配置持续性诊断

## 执行范围与版本合同

使用本地多节点配置 `configs/proposed/system_config_multi_node_factorized.toml` 与Training Episode 20007、20008、20009。Python 3.12.7、NumPy 2.2.6、PyTorch 2.13.0+cpu；本进程仅关闭可选Numba导入并使用项目纯Python KKT fallback。Architecture 33、Checkpoint 60、策略Schema `adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v4_multi_node_cache_factorized_v1`，由正式Loader加载。没有训练、SUMO、Validation/Test读取或正式模型/配置/Checkpoint修改。

配置SHA-256 `a86d87ca5d586e89e367aee393b7a1cd9b27813f41ac3c5a19005bfa40a9ea6c`，Checkpoint SHA-256 `2f79020abdbeacba8d527ab3a15b1022e7d591c7fba63e8169bbf8ccdef63710`。Git HEAD `1fe65bf36e0d5b4d34dd29a35a8038336c9c604f`。正式预测软服务性半径：SV 100m、RSU 250m、UAV 300m。配置中的γ=1.0，因此30槽累计差值为逐槽求和。

B3-7A按候选生成前固定的预测规则，筛选7个UPDATE（4个几何零代理、3个正代理对照），共14次相互独立的正式Slot 0执行。B3-7B每状态比较KEEP、普通UPDATE、锁定3/5/10槽UPDATE，共15条30槽分支、450次正式Slot执行。每条分支重新从相同初始状态开始；相同外生Training任务/轨迹，重用B3-6R随机种子。

## B3-7A：预测服务性代理与实际Matching/KKT价值

候选由当前正式`_cache_node_candidates`产生。正式200维特征前181维为缓存状态基项，后19维为经济项；`demand_added_h0..h4`实际位于全特征索引190–194。本次确认它与正式实现的逐步几何加权需求完全一致。raw需求从当前可见的五步预测场景`arrival/service_id`累计，不读未来真实任务；零代理定义为新增服务raw需求>0而五步加权需求和≤1e-10。每个角色在零代理和正代理组各取预测量最大候选，按 episode、节点ID、服务ID和候选序号固定打破并列，并纳入指定的Episode 20007 RSU_1服务2候选。规则冻结后才运行Matching/KKT。

正式`_serviceability_weight`在角色半径外归零（`src/decision/cache_configuration.py:502`附近），这是候选软代理。正式`_edge_path_is_feasible`使用缓存、有效位置、正CPU/带宽、正信道增益及正速率，不采用这些预测软半径作为硬覆盖门槛（`src/decision/task_allocation.py:1784`附近）。链路存在仍不代表Edge在共享资源与Cloud竞争后具有更高系统效用。

### 预测依据与缓存替换上下文

| # | Episode / 节点角色 | 当前→目标 | 新增 / 移除 | 新服务在其他Edge副本 | 被移除服务的其他副本 | 五步raw需求 / 候选代理 | 有预测请求车辆的距离min/加权均值 / 半径(m) | 选择依据 |
|---:|---|---|---|---|---|---:|---|---|
| 1 | 20007 / RSU_1 RSU | {5,7,9}→{2} | +2 / −5,7,9 | 2:none | 5:23,RSU_0,UAV_0,UAV_1; 7:46,UAV_1; 9:30 | 6.150 / 0.00000 | 639.4/1141.7 / 250 | predeclared Episode 20007 RSU_1 service-2 anchor requested by B3-7A |
| 2 | 20007 / 45 SV | {8}→{1,4} | +1,4 / −8 | 1:UAV_1; 4:46,23 | 8:5,UAV_0 | 25.050 / 0.00000 | 356.5/776.4 / 100 | SV: highest raw added-demand geometry-zero candidate |
| 3 | 20008 / RSU_0 RSU | {3,5}→{1,7,9} | +1,7,9 / −3,5 | 1:UAV_1; 7:4,RSU_1,UAV_1; 9:32,RSU_1 | 3:none; 5:36,RSU_1,UAV_0,UAV_1 | 33.300 / 0.00000 | 395.9/1147.8 / 250 | RSU: highest raw added-demand geometry-zero candidate |
| 4 | 20009 / UAV_0 UAV | {5,8}→{2,4} | +2,4 / −5,8 | 2:none; 4:10,59 | 5:59,RSU_0,RSU_1,UAV_1; 8:12,17 | 26.250 / 0.00000 | 304.5/872.6 / 300 | UAV: highest raw added-demand geometry-zero candidate |
| 5 | 20007 / 23 SV | {4,5}→{1} | +1 / −4,5 | 1:UAV_1 | 4:46; 5:RSU_0,RSU_1,UAV_0,UAV_1 | 19.850 / 0.13320 | 16.4/479.3 / 100 | SV: highest positive weighted-demand control |
| 6 | 20007 / RSU_0 RSU | {3,5}→{1,7,9} | +1,7,9 / −3,5 | 1:UAV_1; 7:46,RSU_1,UAV_1; 9:30,RSU_1 | 3:none; 5:23,RSU_1,UAV_0,UAV_1 | 35.100 / 0.05841 | 87.1/891.2 / 250 | RSU: highest positive weighted-demand control |
| 7 | 20007 / UAV_0 UAV | {5,8}→{1,4,9} | +1,4,9 / −5,8 | 1:UAV_1; 4:46,23; 9:30,RSU_1 | 5:23,RSU_0,RSU_1,UAV_1; 8:5,45 | 36.550 / 0.18739 | 6.1/870.5 / 300 | UAV: highest positive weighted-demand control |

### 成对正式Slot结果

KEEP与UPDATE分别从完整相同初始Cache/UAV状态执行`execute_configured_slot`，双UAV均STAY，除目标Cache节点外其余8行不动。两分支分别重新运行Transfer-Exchange Matching/KKT，无跨分支可变求解缓存。Cache Cost由正式Reward扣一次；`ΔG=ΔQ−ΔCache−ΔUAV`。

| # | Episode / 节点 | Slot0新增服务请求数 | 新服务目标Edge可行数 | UPDATE目标Edge实际执行数 | ΔQ | ΔCache | ΔG | ΔEdge / ΔCloud | 任务路径变化数 | 删除服务路径损失 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20007 / RSU_1 | 1 | 1 | 0 | +0.000000 | +0.106998 | -0.106998 | +0 / +0 | 0 | 0/2 |
| 2 | 20007 / 45 | 8 | 8 | 0 | +0.000000 | +0.109688 | -0.109688 | +0 / +0 | 0 | 0/0 |
| 3 | 20008 / RSU_0 | 7 | 7 | 0 | +0.000000 | +0.174808 | -0.174808 | +0 / +0 | 0 | 0/2 |
| 4 | 20009 / UAV_0 | 9 | 9 | 2 | +0.075091 | +0.221758 | -0.146667 | +2 / -2 | 2 | 0/0 |
| 5 | 20007 / 23 | 6 | 6 | 0 | +0.000000 | +0.075958 | -0.075958 | +0 / +0 | 0 | 0/2 |
| 6 | 20007 / RSU_0 | 8 | 8 | 2 | +0.082315 | +0.174808 | -0.092493 | +2 / -2 | 2 | 0/0 |
| 7 | 20007 / UAV_0 | 10 | 10 | 1 | +0.058871 | +0.242609 | -0.183739 | +1 / -1 | 1 | 0/0 |

### 新增服务任务：链路、实际执行节点与Task Utility

`Full-rate`列是在目标Edge链路可行时按该节点全带宽计算的链路参考速率/上传序列化时间，不代表任务独占带宽；只有正式Matching实际把任务分给目标Edge时才有KKT实际份额及实际上传时间。未分给目标Edge的行不人为构造其Edge效用。KEEP/UPDATE效用均来自实际完整Slot Matching/KKT。

| # | Task / 服务 | 目标Edge路径可行 | KEEP实际节点/效用 | UPDATE实际节点/效用 | 目标Edge全带宽速率/上传时延 | 实际KKT速率/上传时延 |
|---:|---|---|---|---|---|---|
| 1 | 38 / 2 | 是 | CLOUD / 0.96103 | CLOUD / 0.96103 | 3.822e+06 bit/s / 1.4834s | —（Matching未分给目标Edge） |
| 2 | 0 / 4 | 是 | CLOUD / 0.95859 | CLOUD / 0.95859 | 3.189e+06 bit/s / 1.8561s | —（Matching未分给目标Edge） |
| 2 | 34 / 1 | 是 | CLOUD / 0.94758 | CLOUD / 0.94758 | 5.990e+06 bit/s / 1.6702s | —（Matching未分给目标Edge） |
| 2 | 51 / 1 | 是 | CLOUD / 0.95001 | CLOUD / 0.95001 | 1.090e+06 bit/s / 8.3214s | —（Matching未分给目标Edge） |
| 2 | 55 / 1 | 是 | CLOUD / 0.94373 | CLOUD / 0.94373 | 1.011e+07 bit/s / 1.1125s | —（Matching未分给目标Edge） |
| 2 | 53 / 1 | 是 | CLOUD / 0.93726 | CLOUD / 0.93726 | 3.206e+06 bit/s / 3.8915s | —（Matching未分给目标Edge） |
| 2 | 11 / 1 | 是 | UAV_1 / 0.95399 | UAV_1 / 0.95399 | 3.257e+06 bit/s / 3.1384s | —（Matching未分给目标Edge） |
| 2 | 13 / 4 | 是 | CLOUD / 0.96317 | CLOUD / 0.96317 | 3.274e+06 bit/s / 1.3923s | —（Matching未分给目标Edge） |
| 2 | 24 / 1 | 是 | CLOUD / 0.94740 | CLOUD / 0.94740 | 1.059e+06 bit/s / 9.1020s | —（Matching未分给目标Edge） |
| 3 | 21 / 7 | 是 | CLOUD / 0.95508 | CLOUD / 0.95508 | 4.376e+05 bit/s / 27.3062s | —（Matching未分给目标Edge） |
| 3 | 58 / 7 | 是 | RSU_1 / 0.96286 | RSU_1 / 0.96286 | 4.736e+05 bit/s / 23.7817s | —（Matching未分给目标Edge） |
| 3 | 50 / 9 | 是 | CLOUD / 0.94389 | CLOUD / 0.94389 | 1.031e+06 bit/s / 10.5788s | —（Matching未分给目标Edge） |
| 3 | 19 / 9 | 是 | CLOUD / 0.93972 | CLOUD / 0.93972 | 1.244e+06 bit/s / 9.4708s | —（Matching未分给目标Edge） |
| 3 | 15 / 9 | 是 | CLOUD / 0.94979 | CLOUD / 0.94979 | 5.011e+06 bit/s / 1.8881s | —（Matching未分给目标Edge） |
| 3 | 45 / 7 | 是 | UAV_1 / 0.95637 | UAV_1 / 0.95637 | 1.086e+06 bit/s / 13.0744s | —（Matching未分给目标Edge） |
| 3 | 1 / 7 | 是 | CLOUD / 0.94980 | CLOUD / 0.94980 | 1.631e+06 bit/s / 8.6537s | —（Matching未分给目标Edge） |
| 4 | 42 / 4 | 是 | CLOUD / 0.95136 | CLOUD / 0.95585 | 1.721e+07 bit/s / 0.3192s | —（Matching未分给目标Edge） |
| 4 | 25 / 4 | 是 | CLOUD / 0.95683 | CLOUD / 0.96084 | 3.827e+07 bit/s / 0.1249s | —（Matching未分给目标Edge） |
| 4 | 21 / 2 | 是 | CLOUD / 0.94444 | CLOUD / 0.94940 | 1.882e+07 bit/s / 0.3848s | —（Matching未分给目标Edge） |
| 4 | 28 / 4 | 是 | CLOUD / 0.95467 | UAV_0 / 0.95889 | 7.400e+07 bit/s / 0.0668s | 3.065e+07 bit/s / 0.1613s |
| 4 | 27 / 2 | 是 | CLOUD / 0.95340 | CLOUD / 0.95765 | 3.397e+07 bit/s / 0.1630s | —（Matching未分给目标Edge） |
| 4 | 58 / 2 | 是 | CLOUD / 0.95225 | CLOUD / 0.95664 | 2.299e+07 bit/s / 0.2392s | —（Matching未分给目标Edge） |
| 4 | 14 / 4 | 是 | CLOUD / 0.95134 | UAV_0 / 0.95392 | 4.262e+07 bit/s / 0.1337s | 2.497e+07 bit/s / 0.2281s |
| 4 | 7 / 2 | 是 | CLOUD / 0.94758 | CLOUD / 0.95230 | 1.923e+07 bit/s / 0.3428s | —（Matching未分给目标Edge） |
| 4 | 15 / 2 | 是 | CLOUD / 0.94546 | CLOUD / 0.95025 | 3.533e+07 bit/s / 0.2131s | —（Matching未分给目标Edge） |
| 5 | 34 / 1 | 是 | CLOUD / 0.94758 | CLOUD / 0.94758 | 6.490e+06 bit/s / 1.5417s | —（Matching未分给目标Edge） |
| 5 | 51 / 1 | 是 | CLOUD / 0.95001 | CLOUD / 0.95001 | 1.265e+06 bit/s / 7.1705s | —（Matching未分给目标Edge） |
| 5 | 55 / 1 | 是 | CLOUD / 0.94373 | CLOUD / 0.94373 | 6.812e+06 bit/s / 1.6515s | —（Matching未分给目标Edge） |
| 5 | 53 / 1 | 是 | CLOUD / 0.93726 | CLOUD / 0.93726 | 2.875e+06 bit/s / 4.3398s | —（Matching未分给目标Edge） |
| 5 | 11 / 1 | 是 | UAV_1 / 0.95399 | UAV_1 / 0.95399 | 1.077e+08 bit/s / 0.0949s | —（Matching未分给目标Edge） |
| 5 | 24 / 1 | 是 | CLOUD / 0.94740 | CLOUD / 0.94740 | 1.271e+06 bit/s / 7.5896s | —（Matching未分给目标Edge） |
| 6 | 34 / 1 | 是 | CLOUD / 0.94758 | CLOUD / 0.95361 | 6.441e+06 bit/s / 1.5533s | —（Matching未分给目标Edge） |
| 6 | 9 / 9 | 是 | CLOUD / 0.93198 | CLOUD / 0.94020 | 1.634e+06 bit/s / 7.0159s | —（Matching未分给目标Edge） |
| 6 | 51 / 1 | 是 | CLOUD / 0.95001 | RSU_0 / 0.96091 | 5.586e+07 bit/s / 0.1623s | 2.655e+07 bit/s / 0.3415s |
| 6 | 28 / 9 | 是 | CLOUD / 0.93328 | CLOUD / 0.94135 | 3.923e+06 bit/s / 2.8652s | —（Matching未分给目标Edge） |
| 6 | 55 / 1 | 是 | CLOUD / 0.94373 | CLOUD / 0.95007 | 6.895e+05 bit/s / 16.3146s | —（Matching未分给目标Edge） |
| 6 | 53 / 1 | 是 | CLOUD / 0.93726 | CLOUD / 0.94434 | 2.118e+07 bit/s / 0.5891s | —（Matching未分给目标Edge） |
| 6 | 11 / 1 | 是 | UAV_1 / 0.95399 | UAV_1 / 0.95399 | 1.454e+06 bit/s / 7.0276s | —（Matching未分给目标Edge） |
| 6 | 24 / 1 | 是 | CLOUD / 0.94740 | RSU_0 / 0.95818 | 4.877e+07 bit/s / 0.1977s | 2.559e+07 bit/s / 0.3769s |
| 7 | 0 / 4 | 是 | CLOUD / 0.95859 | CLOUD / 0.96170 | 2.484e+07 bit/s / 0.2382s | —（Matching未分给目标Edge） |
| 7 | 34 / 1 | 是 | CLOUD / 0.94758 | CLOUD / 0.95113 | 4.246e+07 bit/s / 0.2356s | —（Matching未分给目标Edge） |
| 7 | 9 / 9 | 是 | CLOUD / 0.93198 | CLOUD / 0.93682 | 2.633e+07 bit/s / 0.4355s | —（Matching未分给目标Edge） |
| 7 | 51 / 1 | 是 | CLOUD / 0.95001 | CLOUD / 0.95347 | 1.265e+08 bit/s / 0.0717s | —（Matching未分给目标Edge） |
| 7 | 28 / 9 | 是 | CLOUD / 0.93328 | CLOUD / 0.93803 | 4.013e+07 bit/s / 0.2801s | —（Matching未分给目标Edge） |
| 7 | 55 / 1 | 是 | CLOUD / 0.94373 | CLOUD / 0.94746 | 1.518e+07 bit/s / 0.7410s | —（Matching未分给目标Edge） |
| 7 | 53 / 1 | 是 | CLOUD / 0.93726 | UAV_0 / 0.95752 | 6.246e+07 bit/s / 0.1998s | 6.246e+07 bit/s / 0.1998s |
| 7 | 11 / 1 | 是 | UAV_1 / 0.95399 | UAV_1 / 0.95399 | 2.480e+07 bit/s / 0.4121s | —（Matching未分给目标Edge） |
| 7 | 13 / 4 | 是 | CLOUD / 0.96317 | CLOUD / 0.96605 | 2.674e+07 bit/s / 0.1705s | —（Matching未分给目标Edge） |
| 7 | 24 / 1 | 是 | CLOUD / 0.94740 | CLOUD / 0.95103 | 1.278e+08 bit/s / 0.0755s | —（Matching未分给目标Edge） |

### B3-7A判断

7个UPDATE在该单槽窗口均为负ΔG（0个正净收益）。4个几何零代理候选的新增服务任务均存在至少一个正式目标Edge链路可行样本；其中3个配对没有改变卸载路径或Q。另有Episode 20009 UAV_0 `(5, 8)→(2, 4)`，五步raw预测新增需求26.25、候选几何代理为0，请求车辆预测最近距离304.5m（略超300m半径）；实际9个新增服务任务链路均可行，两个Cloud任务改由UAV_0执行，其Task Utility分别由0.95467升至0.95889、0.95134升至0.95392。该完整Matching/KKT比较的ΔQ为+0.075091，但一次Cache成本+0.221758使ΔG为−0.146667。此为本次唯一零代理正ΔQ候选，证明软代理可能漏掉Task Utility机会，但没有证明它漏掉了正净收益。

正代理对照也不等同于价值估计：SV节点23的正代理候选没有改变路径、ΔQ=0、ΔG=−0.075958；RSU_0正代理候选使两任务从Cloud转RSU、ΔQ=+0.082315但Cache成本+0.174808；UAV_0正代理候选使一任务转UAV、ΔQ=+0.058871但成本+0.242609。对移除服务的Slot0实际请求，7对分支均未观察到已在Edge执行的任务因目标行删除转为Cloud；这仅是该Slot样本，不否定长窗口可能出现替换机会损失。

结论：候选软代理和正式链路/Task Utility并不完全一致（`MISMATCH_SUPPORTED`），主要正值遗漏证据来自一个UAV近边界样本。所有本次零代理RSU/SV候选均未带来正ΔQ，但数量太少，不能作角色级普遍结论。所有正式路径均可行仍不保证Matching会选择目标Edge；RSU服务2锚点的任务38在KEEP/UPDATE都由Cloud执行，Utility保持0.96103。

## B3-7B：自然回退与目标Cache行保持干预

五个分支在Macro决策边界实施：Slot0强制目标节点KEEP或指定UPDATE；锁定分支在已有Macro边界且起始Slot<N时只令该节点选当前行KEEP，其他8个Cache节点、Termination/Holding及UAV仍由原Checkpoint采样。首个Macro边界起点≥N时恢复目标节点PPO动作；没有插入新边界。每个状态所有分支Slot0其余8个Cache动作向量相同。分支从此后的自身实际Cache/UAV状态继续。

结果是随机冻结PPO的受控Training干预，不是自然PPO策略整体表现。各分支的30槽逐槽ΔQ、ΔCache、ΔUAV、ΔG、累计曲线、实际Cache行、Holding长度、Edge/Cloud任务数、时延/能耗及Matching/KKT摘要保存在配套JSON。

| Episode / 目标节点 | 分支 | ΔQ (30槽) | ΔCache | ΔUAV | ΔG vs KEEP | 首次累计回本Slot | 目标行首次变化 / 回到初始行 | ΔEdge / ΔCloud | Δ总时延(s) | Δ能耗(J) |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 20007 / UAV_0 | KEEP_slot0_then_PPO | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0 | 3 / 无 | +0 / +0 | +0.0000 | +0.000000 |
| 20007 / UAV_0 | UPDATE_once_then_PPO | +0.297417 | +0.172344 | -0.005610 | +0.130683 | 5 | 3 / 无 | +2 / -2 | -19.3395 | +0.623528 |
| 20007 / UAV_0 | UPDATE_lock_3_slots | +0.297417 | +0.172344 | -0.005610 | +0.130683 | 5 | 3 / 无 | +2 / -2 | -19.3395 | +0.623528 |
| 20007 / UAV_0 | UPDATE_lock_5_slots | +0.474577 | +0.167078 | -0.003740 | +0.311239 | 3 | 8 / 无 | +4 / -4 | -29.6628 | +0.815458 |
| 20007 / UAV_0 | UPDATE_lock_10_slots | +0.177492 | -0.171580 | -0.001870 | +0.350942 | 3 | 16 / 无 | -1 / +1 | -9.2726 | +0.031786 |
| 20008 / 23 | KEEP_slot0_then_PPO | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0 | 6 / 8 | +0 / +0 | +0.0000 | +0.000000 |
| 20008 / 23 | UPDATE_once_then_PPO | +0.000000 | +0.119986 | +0.000000 | -0.119986 | 未回本 | 6 / 8 | +0 / +0 | +0.0000 | +0.000000 |
| 20008 / 23 | UPDATE_lock_3_slots | +0.000000 | +0.119986 | +0.000000 | -0.119986 | 未回本 | 6 / 8 | +0 / +0 | +0.0000 | +0.000000 |
| 20008 / 23 | UPDATE_lock_5_slots | +0.000000 | +0.119986 | +0.000000 | -0.119986 | 未回本 | 6 / 8 | +0 / +0 | +0.0000 | +0.000000 |
| 20008 / 23 | UPDATE_lock_10_slots | +0.018707 | -0.303998 | +0.000000 | +0.322705 | 6 | 13 / 无 | +0 / +0 | +0.3349 | -0.193487 |
| 20009 / 17 | KEEP_slot0_then_PPO | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0 | 3 / 无 | +0 / +0 | +0.0000 | +0.000000 |
| 20009 / 17 | UPDATE_once_then_PPO | +0.062159 | -0.049370 | +0.000000 | +0.111529 | 22 | 3 / 4 | +1 / -1 | -3.3571 | +0.027601 |
| 20009 / 17 | UPDATE_lock_3_slots | +0.062159 | -0.049370 | +0.000000 | +0.111529 | 22 | 3 / 4 | +1 / -1 | -3.3571 | +0.027601 |
| 20009 / 17 | UPDATE_lock_5_slots | +0.067696 | -0.241133 | +0.000000 | +0.308829 | 4 | 16 / 无 | +0 / +0 | -0.6080 | -0.427163 |
| 20009 / 17 | UPDATE_lock_10_slots | +0.067696 | -0.241133 | +0.000000 | +0.308829 | 4 | 16 / 无 | +0 / +0 | -0.6080 | -0.427163 |

### 最低保持时长与边界效应

| 最低保持槽数 | ΔG vs KEEP（20007 / 20008 / 20009） | 正ΔG状态 | 相对普通UPDATE的净收益提高 | 普通UPDATE在要求时长前已改目标行 | 首次改行Slot（各Episode） |
|---:|---|---:|---:|---:|---|
| 3 | +0.130683 / -0.119986 / +0.111529 | 2/3 | 0/3 | 0/3 | 3 / 6 / 3 |
| 5 | +0.311239 / -0.119986 / +0.308829 | 2/3 | 2/3 | 2/3 | 8 / 6 / 16 |
| 10 | +0.350942 / +0.322705 / +0.308829 | 3/3 | 3/3 | 3/3 | 16 / 13 / 16 |

### B3-7B判断

保持3槽没有改变结果：既有自然PPO在三状态的目标行首次变化分别为Slot3、6、3，锁定3槽恰在相同或更早的现有Macro边界释放。保持5槽在20007和20009将目标行首次变化从Slot3推迟到Slot8、Slot16，并使30槽ΔG分别比普通UPDATE多+0.180556、+0.197300；20008普通/5槽均在Slot6变化，均为−0.119986。保持10槽把三状态首次变化推迟至Slot16、Slot13、Slot16，三状态30槽ΔG都高于普通UPDATE；相对KEEP的ΔG为+0.350942、+0.322705、+0.308829。

自然UPDATE分支在20007和20009的30槽前缀分别+0.130683和+0.111529，首次回本Slot5、Slot22；20008为−0.119986、未回本。保持分支显示某些固定动作在更长留存窗口有较好30槽结果，但每个Episode只有一个预先固定随机种子。20008的10槽锁定分支总ΔCache为−0.303998，虽然Slot0更新仍支付+0.119986；负的30槽相对ΔCache来自后续整条策略轨迹的分支差异，不能归因于初始UPDATE免费。所有非目标节点、UAV、Matching和任务效用均可能在分支状态分化后改变。

只运行0–29槽；已有B3-6R自然UPDATE的300槽末结果分别为−5.417710、−0.119986、−0.645349，但这些不是本次锁定干预的长窗口标签。因此本次证明的是三个Training状态的前30槽持久性干预效应，不证明5/10槽锁定在100/300槽仍有正收益。现有30槽数据足以回答是否有早期回退效应，本次未再扩展到100槽。

## 综合结论与建议

- **候选价值代理：** 4个零代理样本里1个有正ΔQ、但没有正ΔG；3个没有Q改善。3个正代理对照中1个没有Q改善，2个改善Q但仍被Cache成本翻为负净收益。此有限证据支持进一步对齐候选输入与正式链路可行性/共享资源效用，不支持现在直接放大SV/RSU/UAV的软半径。
- **持续性控制：** Slot3锁定没有额外作用；5槽在2/3状态、10槽在3/3状态高于普通UPDATE的前30槽ΔG，支持进一步验证节点级Cache KEEP与全局Macro Holding的协调。它不支持统一规定所有Cache持有10槽。
- **架构与信用分配：** 本次没有训练或优势/梯度诊断，不能据此修改PPO联合损失、Critic或增加信用分配模块。Cache成本在所有7个单槽候选中均超过当槽ΔQ，不支持改惩罚权重。服务2的RSU零代理链路可行但最终仍选Cloud且Q不变；UAV近半径外的样本才出现一次正ΔQ，说明需求、几何、Cloud竞争和KKT资源必须共同解释。
- **建议：** 后续先以更多预先固定Training候选状态分别验证一个链路/资源一致的候选价值代理；再针对已发生早期改行的节点做小规模现有Macro边界保持对照。两项均仅是研究建议，当前没有修改源码、配置或策略。

完整逐候选、逐任务与450槽逐槽结果：[stageB3_7_cache_value_proxy_and_retention.json](stageB3_7_cache_value_proxy_and_retention.json)。

```ini
STAGE_B3_7 = COMPLETE
PROXY_FORMAL_LINK_CONSISTENCY = MISMATCH_SUPPORTED
ZERO_PROXY_POSITIVE_UTILITY_CASES = 1 (Task Utility; zero positive net Reward)
RSU_PROXY_ISSUE = UNDETERMINED (0/2 zero-proxy candidates had positive ΔQ)
UAV_PROXY_ISSUE = SUPPORTED (1/1 selected zero-proxy case had positive ΔQ, net negative after cost)
SV_PROXY_ISSUE = UNDETERMINED (0/1 zero-proxy candidate had positive ΔQ)
RETENTION_3_SLOT_RESULT = no difference from ordinary UPDATE in 3/3 states
RETENTION_5_SLOT_RESULT = higher than ordinary UPDATE in 2/3; positive vs KEEP in 2/3
RETENTION_10_SLOT_RESULT = higher than ordinary UPDATE in 3/3; positive vs KEEP in 3/3
EARLY_RECONFIGURATION_HARMS_VALUE = SUPPORTED (within these 3 Training states / 30 slots)
CANDIDATE_VALUE_FEATURE_CHANGE_RECOMMENDED = YES (further proxy validation; no immediate source change)
HOLDING_CACHE_COORDINATION_CHANGE_RECOMMENDED = YES (targeted follow-up only; no global holding change)
CACHE_PENALTY_CHANGE_RECOMMENDED = NO
RECOMMENDED_NEXT_STEP = validate a link/resource-aligned candidate proxy and a bounded node-retention comparison on predeclared Training states
PPO_TRAINING_EXECUTED = NO
SYSTEM_MODEL_CHANGED = NO
PPO_ALGORITHM_CHANGED = NO
FORMAL_CONFIG_CHANGED = NO
```
