# Stage6 Cache候选真实价值与PPO排序诊断

## 执行边界与复现输入

- 配置 SHA-256：`684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77`；Checkpoint SHA-256：`21b9d97c65ce8c3f14406472743be92225341e168c8ba6bc04fa9de0d76ee8f6`，均与任务给定值一致。
- 执行环境：`C:\software\anaconda3\python.exe`，Python 3.12.7，NumPy 2.2.6，PyTorch 2.13.0+cpu；Conda 环境 `adaptive_cache_mec` 不可用，使用项目支持的纯 Python KKT fallback（Numba disabled）。
- 数据仅来自 `environment_data/proposed_joint_training/episode_{20001,20101,20201,20300}`。每个状态的原始 Cache、UAV、任务、车辆坐标、预测输入、Observation、候选配置和候选特征都保存在 JSON `states[].replay_inputs` 与 `candidate_definitions` 中。
- 四个状态的候选配置、特征、Checkpoint 概率、候选排序和确定性动作均与既有回放逐项核验。所有单槽候选均调用 `execute_configured_slot`，分支重新执行正式 Transfer-Exchange Matching/KKT；本实验没有训练或修改正式源码/参数。
- PPO 确定性动作按正式分组语义：比较基数校正后的 `p(KEEP)` 与 `p(UPDATE group)`；选择 UPDATE 组后，在 UPDATE 内取策略评分最高候选。随机采样没有被当作确定性最优选择。
- 五槽/二十槽使用首槽筛选出的固定 Cache，UAV 固定在初始位置，后续每槽重新 Matching/KKT，首槽更新成本只支付一次。候选代表项仅按首槽单槽 ΔQ/ΔG 或首状态预测需求选取，不用未来实际任务挑动作。该实验是固定动作反事实，不是在线 Oracle 或完整 PPO 闭环。
- `edge_cache_hit_count` 定义为实际被 RSU/SV/UAV 执行且目标节点当槽 Cache 含该任务服务的任务数；Local 单独统计，不计为边缘 Cache 命中。

## 单槽结果

每个状态共评估 230 个合法 UPDATE 加 KEEP。`ΔG = ΔTask Utility − ΔCache Cost`；双 UAV STAY，因此移动成本为零。全部候选的逐项结果、策略概率和路径变化见 JSON。

| 状态 | PPO确定性动作 | PPO最高分UPDATE | max ΔQ UPDATE | max ΔG UPDATE | 含KEEP最佳动作 | 单槽遗憾 | ΔG>0 UPDATE | 路径改变候选 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20001_middle_slot_150 | KEEP #0 | #167 (p=0.0109) | #63 (+0.066750) | #52 (-0.010000) | KEEP #0 (+0.000000) | 0.000000 | 0/230 | 119/230 |
| 20101_middle_slot_150 | KEEP #0 | #168 (p=0.01127) | #118 (+0.038693) | #54 (-0.010000) | KEEP #0 (+0.000000) | 0.000000 | 0/230 | 105/230 |
| 20201_middle_slot_150 | KEEP #0 | #167 (p=0.01158) | #167 (+0.032203) | #53 (-0.010000) | KEEP #0 (+0.000000) | 0.000000 | 0/230 | 131/230 |
| 20300_middle_slot_150 | KEEP #0 | #167 (p=0.01097) | #107 (+0.034778) | #53 (-0.010000) | KEEP #0 (+0.000000) | 0.000000 | 0/230 | 90/230 |

### 四状态统计

| 指标（按状态的候选极值/遗憾） | 均值 | 中位数 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| 每状态 max UPDATE ΔTask Utility | 0.043106 | 0.036736 | 0.032203 | 0.066750 |
| 每状态 max UPDATE ΔG | -0.010000 | -0.010000 | -0.010000 | -0.010000 |
| PPO 单槽选择遗憾 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 每状态实际路径改变 UPDATE 数 | 111.250000 | 112.000000 | 90.000000 | 131.000000 |

全样本候选状态对中，ΔTask Utility>0 有 154 个，ΔG>0 有 0 个；有路径改变 445 个。最高单槽 UPDATE ΔTask Utility 为 +0.066750（20001_middle_slot_150 候选 #63），最高单槽 UPDATE ΔG 为 -0.010000（20001_middle_slot_150 候选 #52）。

### PPO 排序与价值差

| 状态 | p(KEEP) | p(UPDATE组) | PPO选中候选概率/名次 | 单槽最佳净收益候选概率/UPDATE名次 |
|---|---:|---:|---:|---:|
| 20001_middle_slot_150 | 0.9109 | 0.0891 | #0 p=0.9109, rank=None | #52 p=0.0008888, rank=24 |
| 20101_middle_slot_150 | 0.9119 | 0.0881 | #0 p=0.9119, rank=None | #54 p=0.001096, rank=15 |
| 20201_middle_slot_150 | 0.9120 | 0.0880 | #0 p=0.912, rank=None | #53 p=0.0008492, rank=24 |
| 20300_middle_slot_150 | 0.9112 | 0.0888 | #0 p=0.9112, rank=None | #53 p=0.001118, rank=15 |

确定性 PPO 选择由组概率与组内排序直接决定。若单槽最佳净收益候选未选中，表中的组概率和候选概率名次给出本次实际排序差距；本反事实只能识别策略评分与已实现回报的错位，不能仅凭四个状态断言训练中的具体因果来源。

## 固定动作五槽与二十槽累计结果

候选包含 PPO 确定性动作、单槽最佳 ΔG UPDATE、单槽最佳 ΔQ UPDATE，以及最多两个按当前预测需求质量选出的不同服务配置；同一候选去重。每个候选均从原状态独立开始，Cache 只首槽更新，UAV 全程 STAY。

| 状态 | 候选 | PPO校正概率/UPDATE名次 | 选择理由 | 5槽 ΔQ | ΔC | ΔM | ΔG | 20槽 ΔQ | ΔC | ΔM | ΔG | 20槽边缘Cache命中变化 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20001_middle_slot_150 | #0 | KEEP p=0.9109 | deterministic_PPO_action | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0 |
| 20001_middle_slot_150 | #51 | p=2.87e-05, rank=155 | predicted_demand_representative_1 | +0.026023 | +0.148275 | +0.000000 | -0.122252 | +0.121102 | +0.148275 | +0.000000 | -0.027173 | +13 |
| 20001_middle_slot_150 | #52 | p=0.000889, rank=24 | highest_single_slot_delta_g_nonkeep | -0.052698 | +0.010000 | +0.000000 | -0.062698 | -0.160321 | +0.010000 | +0.000000 | -0.170321 | -5 |
| 20001_middle_slot_150 | #63 | p=6.89e-06, rank=199 | highest_single_slot_delta_q_nonkeep | -0.038114 | +0.176988 | +0.000000 | -0.215103 | -0.485956 | +0.176988 | +0.000000 | -0.662944 | -15 |
| 20001_middle_slot_150 | #104 | p=0.000217, rank=76 | predicted_demand_representative_2 | +0.033533 | +0.108448 | +0.000000 | -0.074915 | +0.059245 | +0.108448 | +0.000000 | -0.049203 | +0 |
| 20101_middle_slot_150 | #0 | KEEP p=0.9119 | deterministic_PPO_action | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0 |
| 20101_middle_slot_150 | #54 | p=0.0011, rank=15 | highest_single_slot_delta_g_nonkeep | -0.062676 | +0.010000 | +0.000000 | -0.072676 | -0.128005 | +0.010000 | +0.000000 | -0.138005 | -4 |
| 20101_middle_slot_150 | #118 | p=4.45e-06, rank=212 | highest_single_slot_delta_q_nonkeep | +0.011257 | +0.123518 | +0.000000 | -0.112261 | +0.178053 | +0.123518 | +0.000000 | +0.054535 | +4 |
| 20101_middle_slot_150 | #163 | p=0.000693, rank=32 | predicted_demand_representative_1 | +0.016432 | +0.221758 | +0.000000 | -0.205326 | +0.200257 | +0.221758 | +0.000000 | -0.021501 | +9 |
| 20101_middle_slot_150 | #178 | p=0.000423, rank=54 | predicted_demand_representative_2 | +0.022456 | +0.248432 | +0.000000 | -0.225976 | +0.216869 | +0.248432 | +0.000000 | -0.031563 | +10 |
| 20201_middle_slot_150 | #0 | KEEP p=0.9120 | deterministic_PPO_action | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0 |
| 20201_middle_slot_150 | #11 | p=1.91e-05, rank=168 | predicted_demand_representative_1 | +0.000000 | +0.133226 | +0.000000 | -0.133226 | +0.000000 | +0.133226 | +0.000000 | -0.133226 | +0 |
| 20201_middle_slot_150 | #19 | p=5.09e-06, rank=209 | predicted_demand_representative_2 | +0.000000 | +0.114874 | +0.000000 | -0.114874 | +0.034431 | +0.114874 | +0.000000 | -0.080443 | +1 |
| 20201_middle_slot_150 | #53 | p=0.000849, rank=24 | highest_single_slot_delta_g_nonkeep | +0.000000 | +0.010000 | +0.000000 | -0.010000 | -0.083375 | +0.010000 | +0.000000 | -0.093375 | -1 |
| 20201_middle_slot_150 | #167 | p=0.0116, rank=1 | highest_single_slot_delta_q_nonkeep | +0.039588 | +0.110395 | +0.000000 | -0.070808 | +0.168895 | +0.110395 | +0.000000 | +0.058500 | +3 |
| 20300_middle_slot_150 | #0 | KEEP p=0.9112 | deterministic_PPO_action | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0 |
| 20300_middle_slot_150 | #53 | p=0.00112, rank=15 | highest_single_slot_delta_g_nonkeep | +0.000000 | +0.010000 | +0.000000 | -0.010000 | -0.070406 | +0.010000 | +0.000000 | -0.080406 | -2 |
| 20300_middle_slot_150 | #71 | p=1.57e-05, rank=175 | predicted_demand_representative_1 | +0.162851 | +0.123518 | +0.000000 | +0.039333 | +0.355084 | +0.123518 | +0.000000 | +0.231566 | +11 |
| 20300_middle_slot_150 | #89 | p=1.35e-05, rank=182 | predicted_demand_representative_2 | +0.030299 | +0.123518 | +0.000000 | -0.093219 | -0.115610 | +0.123518 | +0.000000 | -0.239128 | +3 |
| 20300_middle_slot_150 | #107 | p=8.21e-06, rank=195 | highest_single_slot_delta_q_nonkeep | +0.095558 | +0.171886 | +0.000000 | -0.076328 | +0.077505 | +0.171886 | +0.000000 | -0.094381 | +5 |

短名单中 ΔG>0 的候选状态对：5 槽 1，20 槽 3；按状态统计的 20 槽最佳 ΔG：均值 0.079357、中位数 0.056517、最小 -0.027173、最大 0.231566。所有短名单候选中的最大 20 槽 ΔG 为 +0.231566（20300_middle_slot_150 候选 #71）。

## 候选覆盖

以下是纯枚举与容量合法性检查，没有对遗漏配置运行 Matching/KKT，因此遗漏项没有被标记为低价值。保留比例以合法非 KEEP 单节点配置为分母；PPO 全局候选 231 项含一个 KEEP。

| 状态 | 节点 | 角色 | 可行非KEEP总数 | 进入PPO | 保留比例 | 遗漏合法配置 | 预测锚点保留 |
|---|---|---|---:|---:|---:|---:|---|
| 20001_middle_slot_150 | 46 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20001_middle_slot_150 | 10 | SV | 10 | 10 | 100.0% | 0 | prediction=是, coverage_anchor=是 |
| 20001_middle_slot_150 | 0 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20001_middle_slot_150 | 23 | SV | 10 | 10 | 100.0% | 0 | coverage_anchor=是 |
| 20001_middle_slot_150 | 59 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20001_middle_slot_150 | RSU_0 | RSU | 56 | 56 | 100.0% | 0 | prediction=是 |
| 20001_middle_slot_150 | RSU_1 | RSU | 56 | 56 | 100.0% | 0 | prediction=是, coverage_anchor=是 |
| 20001_middle_slot_150 | UAV_0 | UAV | 34 | 34 | 100.0% | 0 | prediction=是 |
| 20001_middle_slot_150 | UAV_1 | UAV | 34 | 34 | 100.0% | 0 | prediction=是, coverage_anchor=是 |
| 20101_middle_slot_150 | 14 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20101_middle_slot_150 | 54 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20101_middle_slot_150 | 7 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20101_middle_slot_150 | 51 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20101_middle_slot_150 | 3 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20101_middle_slot_150 | RSU_0 | RSU | 56 | 56 | 100.0% | 0 | prediction=是, intermediate=是 |
| 20101_middle_slot_150 | RSU_1 | RSU | 56 | 56 | 100.0% | 0 | prediction=是, coverage_anchor=是 |
| 20101_middle_slot_150 | UAV_0 | UAV | 34 | 34 | 100.0% | 0 | prediction=是, intermediate=是 |
| 20101_middle_slot_150 | UAV_1 | UAV | 34 | 34 | 100.0% | 0 | prediction=是, coverage_anchor=是 |
| 20201_middle_slot_150 | 6 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20201_middle_slot_150 | 16 | SV | 10 | 10 | 100.0% | 0 | prediction=是, coverage_anchor=是 |
| 20201_middle_slot_150 | 23 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20201_middle_slot_150 | 8 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20201_middle_slot_150 | 43 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20201_middle_slot_150 | RSU_0 | RSU | 56 | 56 | 100.0% | 0 | prediction=是, coverage_anchor=是 |
| 20201_middle_slot_150 | RSU_1 | RSU | 56 | 56 | 100.0% | 0 | prediction=是, coverage_anchor=是 |
| 20201_middle_slot_150 | UAV_0 | UAV | 34 | 34 | 100.0% | 0 | prediction=是 |
| 20201_middle_slot_150 | UAV_1 | UAV | 34 | 34 | 100.0% | 0 | prediction=是, coverage_anchor=是 |
| 20300_middle_slot_150 | 8 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20300_middle_slot_150 | 13 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20300_middle_slot_150 | 34 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20300_middle_slot_150 | 24 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20300_middle_slot_150 | 18 | SV | 10 | 10 | 100.0% | 0 | prediction=是 |
| 20300_middle_slot_150 | RSU_0 | RSU | 56 | 56 | 100.0% | 0 | prediction=是 |
| 20300_middle_slot_150 | RSU_1 | RSU | 56 | 56 | 100.0% | 0 | prediction=是, coverage_anchor=是 |
| 20300_middle_slot_150 | UAV_0 | UAV | 34 | 34 | 100.0% | 0 | prediction=是 |
| 20300_middle_slot_150 | UAV_1 | UAV | 34 | 34 | 100.0% | 0 | prediction=是, coverage_anchor=是 |

## 结论

- `CACHE_POSITIVE_SINGLE_SLOT_CANDIDATES`: ΔG>0 的 UPDATE 状态对 0 / 920；最大 UPDATE 单槽 ΔG -0.010000。
- `CACHE_BEST_SINGLE_SLOT_DELTA_G`: 含 KEEP 的最佳动作是 KEEP，ΔG +0.000000；UPDATE 最优值单独见上。
- `CACHE_POSITIVE_5_SLOT_CANDIDATES`: 1 个短名单候选状态对 ΔG>0。
- `CACHE_POSITIVE_20_SLOT_CANDIDATES`: 3 个短名单候选状态对 ΔG>0。
- `CACHE_BEST_20_SLOT_DELTA_G`: +0.231566，位于 20300_middle_slot_150 候选 #71。
- `PPO_CACHE_SELECTION_REGRET`: 20001_middle_slot_150=0.000000, 20101_middle_slot_150=0.000000, 20201_middle_slot_150=0.000000, 20300_middle_slot_150=0.000000；均值 0.000000。遗憾按本状态所有已评价候选（含 KEEP）的最高单槽 ΔG 减去 PPO 确定性动作 ΔG 计算。
- `CACHE_CANDIDATE_COVERAGE_LIMITATION`: 0 个已枚举合法单节点目标未进入正式候选集合；只评价了 PPO 可见集合，遗漏配置的物理价值未知。
- `CACHE_EFFECTIVE_UPDATE_SPACE_SUPPORTED`: SUPPORTED within the evaluated formal PPO candidate shortlist over 20 fixed-action Training slots on 3/4 states; no single-slot positive net UPDATE was found.

### 对根因的判断

920 个单槽 UPDATE 中有 154 个提高 Task Utility（最高 +0.066750），但没有一个抵过 Cache 更新成本；最佳 UPDATE ΔG 仍为 -0.010000，所以这四态单槽最优是 KEEP，PPO 单槽遗憾为零。固定 20 槽后，短名单有 3/16 个候选转为正净收益，体现更新成本摊销与后续需求收益。 PPO 四态确定性动作都选 KEEP：校正后 p(KEEP)=0.911–0.912，p(UPDATE组)=0.088–0.089。正 20 槽候选排序：20101_middle_slot_150 #118: ΔG20=+0.054535, PPO UPDATE 名次 212/230, p=4.45e-06；20201_middle_slot_150 #167: ΔG20=+0.058500, PPO UPDATE 名次 1/230, p=0.0116；20300_middle_slot_150 #71: ΔG20=+0.231566, PPO UPDATE 名次 175/230, p=1.57e-05。其中 20201 的候选 #167 是 UPDATE 组第 1 名，但组概率仍选择 KEEP；20300 的正收益候选 #71 排名较低。证据说明策略在这四态的组选择/候选打分没有优先未来净收益，但不能据此断言 PPO 训练的具体因果。单槽共有 445/920 个 UPDATE 改变执行节点，475 个完全未改变路径；路径改变但 |ΔQ|≤0.001 的有 16 个。

本结论覆盖指定的四个 Training 状态、正式 PPO 候选集合，以及固定 5/20 槽动作窗口。它不把结果外推为全部合法 Cache 动作空间的全局最优性判断。

## 原始结果

完整逐候选/逐槽数据、全部策略概率与排序、候选特征、复现状态输入和覆盖枚举结果见 [stage6_cache_candidate_value_diagnosis.json](stage6_cache_candidate_value_diagnosis.json)。
