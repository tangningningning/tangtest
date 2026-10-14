# Stage6 Cloud–Edge 卸载比例与边际效用诊断

## 结论摘要

- Training 样本共有 **176** 个实际任务，其中 Cloud **141 / 176 (80.11%)**。该比例只描述 12 个预先选定的 Macro 决策槽，不外推为全 300 Episode 的精确总体比例。
- Cloud 任务中，无任何 Edge 服务缓存且无物理可行 Edge 路径的比例为 **56.74%**（80/141）。其中 Cache 未命中 80；有缓存但通信/物理路径不可行 0。
- 存在物理可行 Edge 路径但仍选择 Cloud 的比例为 **43.26%**（61/141）。其中 61 个任务至少有一个 KKT 可行的一步 Edge 候选。
- 对 186 个合法单任务 Cloud→Edge + 全部受影响 KKT 重算候选，找到 **0** 个提升全 Slot Task Utility 的见证；186 个候选的 ΔQ 全为负，最好的观测值为 **-0.006407**（因此没有正 ΔQ 可报告）。固定任务并重优化其余 Matching 后，正收益代表见证 0/12。
- 当前 Cache/UAV 状态下的 Cloud–Edge Matching 诊断不能单独证明缓存或移动动作的因果收益；本实验固定了基线 PPO 选中的 Cache/UAV 目标。对 Cache/UAV 联合动作空间的结论与并行实验分开报告。

## 数据、状态选择与正式系统

- 配置：`C:\Users\tangning\Desktop\code\adaptive_cache_mec\configs\proposed\system_config.toml`（SHA-256 `684bdfa4dc200db6227b2b00a7718797a98bf658e54d2d4dbef6f7d1f5babf77`）；数据仅来自正式 Training Episode `20001, 20101, 20201, 20300`。没有读取 Validation/Test。
- Checkpoint：`C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\stage6_klguard_lr_5e_minus_4\checkpoints\latest_checkpoint.pt`，完成 300 Episode / 75 次 PPO 更新，SHA-256 `21b9d97c65ce8c3f14406472743be92225341e168c8ba6bc04fa9de0d76ee8f6`；执行前后哈希相同。没有训练或更新网络。
- 检查时不存在另一任务可复用的状态清单。本报告按固定规则选取每个 Episode 中严格晚于 Slot 20、150、280 的第一个合法 Macro 决策边界；共 12 个槽，选择不看任何分支收益。决策前 Cache、UAV、当前任务/车辆位置、完整 prediction step、候选可执行目标和 Checkpoint 信息保存在配套 JSON。
- Cloud 合同读到：CPU 模式 `finite_pool`，共享池 40.000 GHz；每任务固定等效上行 200.000 Mbps，时延系数 15.000；Cloud 上行带宽不按任务数拆分。Fallback `RESOURCE_AWARE_FALLBACK`。Matching 使用当前 `RESOURCE_AWARE_FALLBACK`、Transfer–Exchange 和 KKT。
- 运行环境：3.12.7 / NumPy 2.2.6 / Torch 2.13.0+cpu；本机未安装 `adaptive_cache_mec` Conda env，已安装 Numba 与 NumPy 版本不兼容，按项目可选 Numba 缺失时的纯 Python KKT fallback 执行。未改源码、配置、数据、Checkpoint 或正式 Runner。

## 正式策略的任务分配与实际效用

| 执行类别 | 任务数 | 占全部任务 | 平均总时延 (s) | 平均通信时延 (s) | 平均计算时延 (s) | 平均有效上行速率 (Mbps) | 平均车辆能耗 (J) | 平均 Task Utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cloud | 141 | 80.11% | 2.317299 | 0.652107 | 1.665192 | 200.000000 | 0.008674 | 0.953473 |
| RSU | 15 | 8.52% | 1.213851 | 0.319043 | 0.894808 | 51.352632 | 0.063657 | 0.967908 |
| UAV0 | 11 | 6.25% | 2.401709 | 0.180508 | 2.221201 | 78.417035 | 0.036016 | 0.948249 |
| UAV1 | 9 | 5.11% | 2.048106 | 0.611781 | 1.436325 | 23.681065 | 0.122066 | 0.943938 |
| SV | 0 | 0.00% | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| Local | 0 | 0.00% | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |

绝对任务数和比例按 12 个决策槽的实际 Matching 结果统计；节点级分配、可执行候选数量和逐任务记录见 JSON。下表的“合法 Edge 候选数”按正式 Cache 服务命中、当前位置和真实链路可行性构造；不会把候选可行误当成 Matching 选择。

| 状态 | 槽任务数 | Cloud | RSU | UAV0 | UAV1 | SV | Local | Cloud任务：无Edge缓存 | 缓存但无物理Edge路径 | 有Edge路径仍Cloud | 至少一个KKT可行Edge候选 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20001_after20_slot_25 | 17 | 13 | 2 | 1 | 1 | 0 | 0 | 5 | 0 | 8 | 8 |
| 20001_after150_slot_155 | 13 | 10 | 2 | 0 | 1 | 0 | 0 | 8 | 0 | 2 | 2 |
| 20001_after280_slot_285 | 13 | 10 | 1 | 1 | 1 | 0 | 0 | 5 | 0 | 5 | 5 |
| 20101_after20_slot_25 | 14 | 11 | 1 | 1 | 1 | 0 | 0 | 7 | 0 | 4 | 4 |
| 20101_after150_slot_155 | 16 | 15 | 0 | 1 | 0 | 0 | 0 | 14 | 0 | 1 | 1 |
| 20101_after280_slot_285 | 17 | 15 | 0 | 1 | 1 | 0 | 0 | 11 | 0 | 4 | 4 |
| 20201_after20_slot_25 | 11 | 9 | 1 | 1 | 0 | 0 | 0 | 3 | 0 | 6 | 6 |
| 20201_after150_slot_155 | 15 | 10 | 3 | 1 | 1 | 0 | 0 | 6 | 0 | 4 | 4 |
| 20201_after280_slot_285 | 18 | 15 | 1 | 1 | 1 | 0 | 0 | 6 | 0 | 9 | 9 |
| 20300_after20_slot_25 | 11 | 9 | 1 | 1 | 0 | 0 | 0 | 3 | 0 | 6 | 6 |
| 20300_after150_slot_155 | 13 | 10 | 1 | 1 | 1 | 0 | 0 | 6 | 0 | 4 | 4 |
| 20300_after280_slot_285 | 18 | 14 | 2 | 1 | 1 | 0 | 0 | 6 | 0 | 8 | 8 |

## Cache 命中、通信时延与计算资源竞争

Cloud 任务的互斥主分类：无任何 Edge 节点缓存任务服务 **80**；服务有缓存但所有被缓存节点缺少有效物理/通信路径 **0**；存在物理 Edge 候选但最终选择 Cloud **61**。第三类再分为 KKT 候选资源不可行、KKT 可行但任务/系统效用不胜 Cloud、以及正收益候选；分类计数 `{"edge_advantage_erased_under_kkt_resource_sharing": 28, "edge_no_sharing_upper_bound_not_better_than_cloud": 33, "no_edge_service_cached": 80}`。

当前 Edge 可行性检查只排除缺坐标、非正 CPU/带宽、非正信道增益/速率等路径；它没有用时延阈值淘汰慢链路。因此，若链路虽然慢但仍产生正速率，它会进入合法候选，通信劣势通过实际 Utility 与重新分配后的 KKT 候选体现，而不是记成“不可行”。

Cloud 实际固定上行速率为 200.000 Mbps，通信时延使用 `15 × input_bits / 200 Mbps`；样本 Cloud 任务平均通信时延 0.652107 s。基线已分配 Edge 任务的平均有效上行速率按执行类别列在上表。186 个合法单任务转移候选经 KKT 分配后的 Edge 上行速率均值/中位数/范围为 21.421 / 3.567 / 0.285–306.110 Mbps；其中 181 个速率低于 Cloud 的 200 Mbps。速率本身不等于时延优劣：Cloud 上传时延系数是 15，Edge 使用真实信道上传公式；在本样本中 Cloud 平均上传时延为 0.652107 s，RSU/UAV0/UAV1 分别为 0.319043 / 0.180508 / 0.611781 s。逐任务 Cloud/Edge 上传时延、计算时延、有效速率、CPU/BW 份额保存在 JSON `task_profiles` 与 transfer rows。

| Edge节点 | 角色 | 平均实际分配任务数 | 平均CPU份额使用率 | 平均带宽份额使用率 | 最大并发任务数 |
|---|---|---:|---:|---:|---:|
| 0 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 10 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 13 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 14 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 16 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 18 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 23 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 24 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 3 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 34 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 43 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 46 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 51 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 54 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 59 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 6 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 7 | SV | 0.00 | 0.00% | 0.00% | 0 |
| 8 | SV | 0.00 | 0.00% | 0.00% | 0 |
| RSU_0 | RSU | 0.92 | 66.67% | 66.67% | 2 |
| RSU_1 | RSU | 0.33 | 33.33% | 33.33% | 1 |
| UAV_0 | UAV | 0.92 | 91.67% | 91.67% | 1 |
| UAV_1 | UAV | 0.75 | 75.00% | 75.00% | 1 |

Cloud 40 GHz 池在 12/12 个状态都用满；平均使用 40.000 GHz，平均/最小/最大使用率 100.00% / 100.00% / 100.00%。逐状态 Cloud 每任务 CPU 分配、池余量以及各 Edge 节点 CPU/BW fraction 见 JSON `resource_usage`。Cloud 池饱和说明该资源确实被用满，但并未令 Cloud→Edge 反事实变成正收益。

## Cloud 与合法 Edge 的效用对照

分成三个口径，避免把局部估值误称为系统净收益：

1. **基线实际分配**：Cloud 侧是 Cloud KKT CPU 分配下的真实任务时延/能耗/效用；Edge 侧汇报当前真正分配到各节点任务的实际指标和资源使用。Cloud 任务没有被分配 Edge CPU/BW，因此不存在它在基线下的“实际 Edge utility”。
2. **无共享 Edge 上界**：对每个 Cloud 任务及合法 Edge，调用项目正式 Matching pruning 使用的 `_ideal_task_utility`，令该 Edge 任务独占完整 CPU/BW。这只是 Edge 侧上界，不是当前负载下效用，也不是系统反事实。若该上界仍不高于 Cloud 真实效用，Edge 资源竞争无法解释该任务的 Cloud 胜出；若上界更高，则只说明存在潜在空间。
3. **单任务转移 + KKT**：对每个 Cloud 任务逐个尝试其合法 Edge，固定其他任务当前执行节点，但重新对整组 Cloud 和所有 Edge 节点执行 KKT；记录该任务和所有受影响任务的时延、能耗、效用变化，以及全 Slot ΔQ。这是精确的一步 transfer 候选评价，不是重新运行后的局部效用猜测。

| 状态 | Cloud任务 | 合法Edge候选 | Cloud实际U / delay(s) / energy(J) | 无共享Edge最佳U | KKT一步最佳Edge U / delay(s) / energy(J) | Δ单任务U | Δ全Slot Q | 分类 |
|---|---|---|---|---:|---|---:|---:|---|
| 20001_after20_slot_25 | 1 | 10, 59, RSU_0, UAV_0 | U=0.9516 / 2.4095 / 0.0089 | 0.9706 | UAV_0: U=0.9287 / 3.1228 / 0.0778 | -0.022971 | -0.033488 | edge_advantage_erased_under_kkt_resource_sharing |
| 20001_after20_slot_25 | 12 | 0, RSU_1 | U=0.9352 / 3.2367 / 0.0106 | 0.6872 | 0: U=0.6872 / 12.9434 / 0.4534 | -0.247979 | -0.205441 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20001_after20_slot_25 | 15 | 无 | U=0.9585 / 2.0747 / 0.0063 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after20_slot_25 | 16 | 无 | U=0.9654 / 1.7371 / 0.0047 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after20_slot_25 | 22 | 46, RSU_1, UAV_1 | U=0.9440 / 2.7681 / 0.0134 | 0.9490 | UAV_1: U=0.9037 / 3.5159 / 0.2104 | -0.040376 | -0.053990 | edge_advantage_erased_under_kkt_resource_sharing |
| 20001_after20_slot_25 | 27 | 无 | U=0.9669 / 1.6579 / 0.0045 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after20_slot_25 | 31 | 0, RSU_1 | U=0.9384 / 3.0763 / 0.0100 | 0.7325 | 0: U=0.7325 / 11.3848 / 0.3408 | -0.205950 | -0.165204 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20001_after20_slot_25 | 34 | 10, 59, RSU_0, UAV_0 | U=0.9595 / 2.0223 / 0.0071 | 0.9631 | UAV_0: U=0.9198 / 3.0171 / 0.1617 | -0.039680 | -0.055780 | edge_advantage_erased_under_kkt_resource_sharing |
| 20001_after20_slot_25 | 36 | 无 | U=0.9563 / 2.1838 / 0.0068 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after20_slot_25 | 42 | RSU_0, UAV_1 | U=0.9477 / 2.6075 / 0.0091 | 0.9499 | UAV_1: U=0.9039 / 3.6999 / 0.1813 | -0.043876 | -0.055579 | edge_advantage_erased_under_kkt_resource_sharing |
| 20001_after20_slot_25 | 44 | 无 | U=0.9672 / 1.6465 / 0.0043 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after20_slot_25 | 48 | 46, RSU_1, UAV_1 | U=0.9444 / 2.7483 / 0.0132 | 0.9504 | UAV_1: U=0.9055 / 3.4651 / 0.2038 | -0.038928 | -0.052184 | edge_advantage_erased_under_kkt_resource_sharing |
| 20001_after20_slot_25 | 51 | 10, 59, RSU_0, UAV_0 | U=0.9570 / 2.1432 / 0.0079 | 0.9585 | UAV_0: U=0.9129 / 3.2395 / 0.1812 | -0.044117 | -0.061490 | edge_advantage_erased_under_kkt_resource_sharing |
| 20001_after150_slot_155 | 12 | 无 | U=0.9606 / 1.9418 / 0.0102 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after150_slot_155 | 16 | 46, 23, RSU_0 | U=0.9711 / 1.4429 / 0.0046 | 0.9083 | 23: U=0.8932 / 4.7890 / 0.0998 | -0.077972 | -0.058632 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20001_after150_slot_155 | 27 | 无 | U=0.9623 / 1.8603 / 0.0096 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after150_slot_155 | 31 | 无 | U=0.9597 / 1.9898 / 0.0100 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after150_slot_155 | 37 | 无 | U=0.9781 / 1.0963 / 0.0036 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after150_slot_155 | 42 | 无 | U=0.9781 / 1.0970 / 0.0034 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after150_slot_155 | 45 | 无 | U=0.9621 / 1.8840 / 0.0078 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after150_slot_155 | 50 | 46, 23, RSU_0 | U=0.9637 / 1.8074 / 0.0067 | 0.8801 | 23: U=0.8801 / 6.0898 / 0.0044 | -0.083584 | -0.061053 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20001_after150_slot_155 | 51 | 无 | U=0.9587 / 2.0390 / 0.0105 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after150_slot_155 | 7 | 无 | U=0.9724 / 1.3770 / 0.0049 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after280_slot_285 | 12 | 46, 23, RSU_0 | U=0.9673 / 1.6276 / 0.0060 | 0.9257 | 23: U=0.9027 / 4.9167 / 0.0073 | -0.064609 | -0.043930 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20001_after280_slot_285 | 15 | 10, 59, RSU_0, UAV_0 | U=0.9661 / 1.6829 / 0.0072 | 0.9731 | UAV_0: U=0.9365 / 2.6992 / 0.0813 | -0.029581 | -0.046116 | edge_advantage_erased_under_kkt_resource_sharing |
| 20001_after280_slot_285 | 22 | 46, 23, RSU_0 | U=0.9701 / 1.4904 / 0.0050 | 0.9280 | 23: U=0.9141 / 4.3622 / 0.0033 | -0.056031 | -0.036389 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20001_after280_slot_285 | 27 | 无 | U=0.9652 / 1.7202 / 0.0081 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after280_slot_285 | 30 | 10, 59, RSU_0, UAV_0 | U=0.9654 / 1.7138 / 0.0079 | 0.9786 | UAV_0: U=0.9456 / 2.4742 / 0.0458 | -0.019843 | -0.033079 | edge_advantage_erased_under_kkt_resource_sharing |
| 20001_after280_slot_285 | 33 | 10, 59, RSU_0, UAV_0 | U=0.9617 / 1.8988 / 0.0088 | 0.9725 | UAV_0: U=0.9350 / 2.8979 / 0.0630 | -0.026650 | -0.042478 | edge_advantage_erased_under_kkt_resource_sharing |
| 20001_after280_slot_285 | 36 | 无 | U=0.9695 / 1.5217 / 0.0055 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after280_slot_285 | 37 | 无 | U=0.9666 / 1.6617 / 0.0067 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after280_slot_285 | 44 | 无 | U=0.9656 / 1.7040 / 0.0080 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20001_after280_slot_285 | 50 | 无 | U=0.9696 / 1.5135 / 0.0055 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after20_slot_25 | 15 | 无 | U=0.9421 / 2.8656 / 0.0135 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after20_slot_25 | 22 | 无 | U=0.9402 / 2.9534 / 0.0145 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after20_slot_25 | 25 | 14, 51, RSU_0 | U=0.9666 / 1.6719 / 0.0047 | 0.9290 | RSU_0: U=0.9290 / 1.7780 / 0.2768 | -0.037611 | -0.013667 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20101_after20_slot_25 | 26 | 无 | U=0.9355 / 3.1810 / 0.0165 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after20_slot_25 | 30 | 无 | U=0.9534 / 2.3071 / 0.0105 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after20_slot_25 | 34 | 54, 3, RSU_0, UAV_0 | U=0.9564 / 2.1683 / 0.0088 | 0.9733 | UAV_0: U=0.9325 / 3.0492 / 0.0596 | -0.023883 | -0.037757 | edge_advantage_erased_under_kkt_resource_sharing |
| 20101_after20_slot_25 | 36 | 无 | U=0.9385 / 3.0389 / 0.0152 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after20_slot_25 | 43 | 14, 51, RSU_0 | U=0.9618 / 1.9117 / 0.0055 | 0.8915 | 14: U=0.8915 / 5.2441 / 0.0444 | -0.070360 | -0.043509 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20101_after20_slot_25 | 5 | 无 | U=0.9603 / 1.9847 / 0.0063 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after20_slot_25 | 56 | 无 | U=0.9579 / 2.0847 / 0.0093 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after20_slot_25 | 9 | 14, 51, RSU_0 | U=0.9593 / 2.0339 / 0.0064 | 0.8543 | 14: U=0.8543 / 6.2792 / 0.1735 | -0.104992 | -0.077221 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20101_after150_slot_155 | 12 | 无 | U=0.9545 / 2.2785 / 0.0066 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after150_slot_155 | 15 | 无 | U=0.9464 / 2.6640 / 0.0110 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after150_slot_155 | 17 | 无 | U=0.9477 / 2.5997 / 0.0106 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after150_slot_155 | 18 | 无 | U=0.9557 / 2.2234 / 0.0060 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after150_slot_155 | 26 | 无 | U=0.9560 / 2.2033 / 0.0062 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after150_slot_155 | 27 | 无 | U=0.9507 / 2.4590 / 0.0090 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after150_slot_155 | 28 | 无 | U=0.9436 / 2.8038 / 0.0113 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after150_slot_155 | 30 | 无 | U=0.9556 / 2.2252 / 0.0060 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after150_slot_155 | 34 | 54, 3, RSU_0, UAV_0 | U=0.9587 / 2.0643 / 0.0064 | 0.9748 | UAV_0: U=0.9392 / 2.5510 / 0.0832 | -0.019567 | -0.026146 | edge_advantage_erased_under_kkt_resource_sharing |
| 20101_after150_slot_155 | 38 | 无 | U=0.9678 / 1.6177 / 0.0038 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after150_slot_155 | 43 | 无 | U=0.9480 / 2.5857 / 0.0104 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after150_slot_155 | 5 | 无 | U=0.9521 / 2.4015 / 0.0067 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after150_slot_155 | 57 | 无 | U=0.9613 / 1.9445 / 0.0050 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after150_slot_155 | 58 | 无 | U=0.9463 / 2.6694 / 0.0106 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after150_slot_155 | 9 | 无 | U=0.9599 / 2.0127 / 0.0052 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after280_slot_285 | 12 | 无 | U=0.9537 / 2.3250 / 0.0058 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after280_slot_285 | 15 | 7, RSU_1 | U=0.9252 / 3.7401 / 0.0114 | 0.8327 | RSU_1: U=0.8327 / 4.2525 / 0.6432 | -0.092548 | -0.041172 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20101_after280_slot_285 | 17 | 无 | U=0.9524 / 2.3775 / 0.0082 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after280_slot_285 | 22 | 14, RSU_1, UAV_1 | U=0.9478 / 2.5930 / 0.0108 | 0.9325 | UAV_1: U=0.8711 / 3.7924 / 0.4183 | -0.076718 | -0.105153 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20101_after280_slot_285 | 25 | 7, RSU_1 | U=0.9332 / 3.3501 / 0.0087 | 0.7512 | 7: U=0.7512 / 10.5831 / 0.3174 | -0.181999 | -0.133656 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20101_after280_slot_285 | 26 | 无 | U=0.9444 / 2.7701 / 0.0103 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after280_slot_285 | 27 | 无 | U=0.9515 / 2.4188 / 0.0084 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after280_slot_285 | 34 | 无 | U=0.9423 / 2.8747 / 0.0109 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after280_slot_285 | 36 | 无 | U=0.9532 / 2.3463 / 0.0062 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after280_slot_285 | 43 | 无 | U=0.9476 / 2.6102 / 0.0095 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after280_slot_285 | 5 | 无 | U=0.9403 / 2.9752 / 0.0112 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after280_slot_285 | 53 | 无 | U=0.9253 / 3.7125 / 0.0154 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after280_slot_285 | 56 | 无 | U=0.9563 / 2.1975 / 0.0052 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20101_after280_slot_285 | 57 | 7, RSU_1 | U=0.9196 / 4.0227 / 0.0121 | 0.6732 | 7: U=0.6732 / 14.0161 / 0.4004 | -0.246466 | -0.191391 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20101_after280_slot_285 | 9 | 无 | U=0.9499 / 2.5124 / 0.0065 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20201_after20_slot_25 | 19 | 6, 8, RSU_0 | U=0.9692 / 1.5361 / 0.0052 | 0.9048 | 8: U=0.9048 / 4.6753 / 0.0279 | -0.064460 | -0.044295 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20201_after20_slot_25 | 37 | 6, 8, RSU_0 | U=0.9633 / 1.8270 / 0.0066 | 0.9184 | RSU_0: U=0.8882 / 3.0816 / 0.3935 | -0.075101 | -0.082183 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20201_after20_slot_25 | 45 | 无 | U=0.9679 / 1.6012 / 0.0059 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20201_after20_slot_25 | 5 | 无 | U=0.9747 / 1.2635 / 0.0042 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20201_after20_slot_25 | 50 | 无 | U=0.9464 / 2.6349 / 0.0148 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20201_after20_slot_25 | 53 | 23, RSU_1 | U=0.9560 / 2.1887 / 0.0086 | 0.8746 | RSU_1: U=0.8746 / 3.2196 / 0.4776 | -0.081444 | -0.055278 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20201_after20_slot_25 | 54 | 23, RSU_1 | U=0.9549 / 2.2435 / 0.0087 | 0.7865 | 23: U=0.7865 / 9.6908 / 0.1812 | -0.168413 | -0.141561 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20201_after20_slot_25 | 57 | 16, 43, RSU_0, UAV_0 | U=0.9664 / 1.6657 / 0.0075 | 0.9766 | UAV_0: U=0.9385 / 2.7084 / 0.0650 | -0.027941 | -0.046552 | edge_advantage_erased_under_kkt_resource_sharing |
| 20201_after20_slot_25 | 9 | 16, 43, RSU_0, UAV_0 | U=0.9643 / 1.7706 / 0.0078 | 0.9540 | UAV_0: U=0.9058 / 3.5659 / 0.1868 | -0.058543 | -0.085963 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20201_after150_slot_155 | 11 | 8, RSU_1, UAV_0, UAV_1 | U=0.9372 / 3.1115 / 0.0138 | 0.9394 | UAV_0: U=0.8850 / 5.3341 / 0.0804 | -0.052229 | -0.072399 | edge_advantage_erased_under_kkt_resource_sharing |
| 20201_after150_slot_155 | 12 | 无 | U=0.9725 / 1.3734 / 0.0042 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20201_after150_slot_155 | 19 | 23, RSU_1 | U=0.9458 / 2.6943 / 0.0107 | 0.5099 | 23: U=0.3954 / 19.4093 / 1.7182 | -0.550423 | -0.518670 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20201_after150_slot_155 | 30 | 无 | U=0.9711 / 1.4437 / 0.0045 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20201_after150_slot_155 | 37 | 6, 8, RSU_0 | U=0.9689 / 1.5549 / 0.0051 | 0.8314 | 6: U=0.8314 / 6.0524 / 0.3833 | -0.137497 | -0.116610 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20201_after150_slot_155 | 42 | 无 | U=0.9711 / 1.4429 / 0.0048 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20201_after150_slot_155 | 45 | 无 | U=0.9714 / 1.4309 / 0.0046 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20201_after150_slot_155 | 5 | 无 | U=0.9603 / 1.9741 / 0.0076 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20201_after150_slot_155 | 53 | 6, RSU_1, UAV_1 | U=0.9542 / 2.2510 / 0.0127 | 0.9385 | UAV_1: U=0.8800 / 3.8537 / 0.3407 | -0.074228 | -0.109824 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20201_after150_slot_155 | 55 | 无 | U=0.9716 / 1.4184 / 0.0044 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20201_after280_slot_285 | 10 | 无 | U=0.9236 / 3.7936 / 0.0157 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20201_after280_slot_285 | 11 | 6, RSU_1, UAV_1 | U=0.9320 / 3.3687 / 0.0150 | 0.8976 | UAV_1: U=0.8393 / 4.8837 / 0.4980 | -0.092751 | -0.110310 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20201_after280_slot_285 | 13 | 无 | U=0.9427 / 2.8556 / 0.0106 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20201_after280_slot_285 | 19 | 无 | U=0.9575 / 2.1342 / 0.0050 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20201_after280_slot_285 | 29 | 6, RSU_1, UAV_1 | U=0.9493 / 2.5194 / 0.0101 | 0.9692 | UAV_1: U=0.9356 / 2.3600 / 0.1391 | -0.013720 | -0.014701 | edge_advantage_erased_under_kkt_resource_sharing |
| 20201_after280_slot_285 | 33 | 无 | U=0.9520 / 2.4081 / 0.0061 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20201_after280_slot_285 | 35 | 16, 43, RSU_0, UAV_0 | U=0.9530 / 2.3544 / 0.0067 | 0.9744 | UAV_0: U=0.9361 / 2.7861 / 0.0717 | -0.016948 | -0.021193 | edge_advantage_erased_under_kkt_resource_sharing |
| 20201_after280_slot_285 | 37 | 23, RSU_1 | U=0.9339 / 3.3187 / 0.0085 | 0.8090 | 23: U=0.8090 / 9.3456 / 0.0606 | -0.124866 | -0.076825 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20201_after280_slot_285 | 45 | 6, RSU_1, UAV_1 | U=0.9407 / 2.9455 / 0.0121 | 0.9530 | UAV_1: U=0.9118 / 3.1108 / 0.2087 | -0.028916 | -0.032827 | edge_advantage_erased_under_kkt_resource_sharing |
| 20201_after280_slot_285 | 50 | 无 | U=0.9453 / 2.7282 / 0.0099 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20201_after280_slot_285 | 53 | 6, RSU_1, UAV_1 | U=0.9396 / 2.9964 / 0.0131 | 0.9435 | UAV_1: U=0.8989 / 3.3705 / 0.2688 | -0.040711 | -0.048521 | edge_advantage_erased_under_kkt_resource_sharing |
| 20201_after280_slot_285 | 54 | 6, 8, RSU_0 | U=0.9522 / 2.3998 / 0.0061 | 0.9354 | RSU_0: U=0.9113 / 2.5094 / 0.3027 | -0.040887 | -0.029302 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20201_after280_slot_285 | 55 | 16, 43, RSU_0, UAV_0 | U=0.9485 / 2.5772 / 0.0079 | 0.9696 | UAV_0: U=0.9280 / 3.0967 / 0.0865 | -0.020439 | -0.025595 | edge_advantage_erased_under_kkt_resource_sharing |
| 20201_after280_slot_285 | 57 | 8, RSU_1, UAV_0, UAV_1 | U=0.9184 / 4.0745 / 0.0133 | 0.9463 | UAV_0: U=0.8884 / 5.2876 / 0.0615 | -0.030056 | -0.033585 | edge_advantage_erased_under_kkt_resource_sharing |
| 20201_after280_slot_285 | 9 | 无 | U=0.9554 / 2.2402 / 0.0058 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20300_after20_slot_25 | 1 | 34, RSU_1 | U=0.9462 / 2.6681 / 0.0119 | 0.6832 | 34: U=0.6832 / 13.1010 / 0.4606 | -0.262962 | -0.233565 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20300_after20_slot_25 | 15 | 13, 18, RSU_0, UAV_0 | U=0.9672 / 1.6237 / 0.0072 | 0.9787 | UAV_0: U=0.9559 / 1.9204 / 0.0493 | -0.011307 | -0.014846 | edge_advantage_erased_under_kkt_resource_sharing |
| 20300_after20_slot_25 | 16 | 8, 24, RSU_0 | U=0.9638 / 1.8053 / 0.0068 | 0.9197 | RSU_0: U=0.8970 / 2.9139 / 0.3517 | -0.066756 | -0.066932 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20300_after20_slot_25 | 2 | 8, 24, RSU_0 | U=0.9692 / 1.5371 / 0.0053 | 0.9179 | RSU_0: U=0.8962 / 2.8168 / 0.3725 | -0.073020 | -0.074630 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20300_after20_slot_25 | 28 | 无 | U=0.9595 / 2.0017 / 0.0101 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20300_after20_slot_25 | 4 | 8, RSU_1, UAV_1 | U=0.9563 / 2.1520 / 0.0121 | 0.9282 | UAV_1: U=0.9282 / 2.1777 / 0.2234 | -0.028097 | -0.006407 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20300_after20_slot_25 | 45 | 13, 18, RSU_0, UAV_0 | U=0.9628 / 1.8416 / 0.0088 | 0.9599 | UAV_0: U=0.9301 / 2.6231 / 0.1414 | -0.032631 | -0.041571 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20300_after20_slot_25 | 46 | 无 | U=0.9476 / 2.5782 / 0.0149 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20300_after20_slot_25 | 6 | 无 | U=0.9607 / 1.9396 / 0.0098 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20300_after150_slot_155 | 30 | 无 | U=0.9572 / 2.1174 / 0.0101 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20300_after150_slot_155 | 35 | 无 | U=0.9708 / 1.4586 / 0.0049 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20300_after150_slot_155 | 40 | 无 | U=0.9534 / 2.3032 / 0.0112 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20300_after150_slot_155 | 44 | 无 | U=0.9561 / 2.1743 / 0.0103 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20300_after150_slot_155 | 45 | 24, RSU_1, UAV_0, UAV_1 | U=0.9286 / 3.5290 / 0.0171 | 0.9208 | UAV_0: U=0.8494 / 6.4539 / 0.1855 | -0.079280 | -0.113799 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20300_after150_slot_155 | 46 | 8, 24, RSU_0 | U=0.9629 / 1.8534 / 0.0064 | 0.8271 | 8: U=0.8271 / 6.8185 / 0.3009 | -0.135720 | -0.111510 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20300_after150_slot_155 | 47 | 无 | U=0.9679 / 1.6028 / 0.0053 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20300_after150_slot_155 | 51 | 13, 18, RSU_0, UAV_0 | U=0.9577 / 2.0993 / 0.0089 | 0.9664 | UAV_0: U=0.9199 / 3.3964 / 0.1039 | -0.037816 | -0.059056 | edge_advantage_erased_under_kkt_resource_sharing |
| 20300_after150_slot_155 | 58 | 无 | U=0.9588 / 2.0391 / 0.0094 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20300_after150_slot_155 | 7 | 13, 18, RSU_0, UAV_0 | U=0.9650 / 1.7407 / 0.0068 | 0.9790 | UAV_0: U=0.9423 / 2.5905 / 0.0536 | -0.022748 | -0.037581 | edge_advantage_erased_under_kkt_resource_sharing |
| 20300_after280_slot_285 | 1 | RSU_0, UAV_1 | U=0.9275 / 3.6131 / 0.0134 | 0.9417 | UAV_1: U=0.8896 / 4.3698 / 0.1902 | -0.037918 | -0.043134 | edge_advantage_erased_under_kkt_resource_sharing |
| 20300_after280_slot_285 | 16 | 34, RSU_1 | U=0.9268 / 3.6676 / 0.0102 | 0.7477 | 34: U=0.7477 / 11.6042 / 0.1910 | -0.179076 | -0.127536 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20300_after280_slot_285 | 2 | 8, RSU_1, UAV_1 | U=0.9362 / 3.1611 / 0.0140 | 0.9511 | UAV_1: U=0.9045 / 3.5151 / 0.2040 | -0.031752 | -0.039831 | edge_advantage_erased_under_kkt_resource_sharing |
| 20300_after280_slot_285 | 25 | 无 | U=0.9420 / 2.8889 / 0.0105 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20300_after280_slot_285 | 28 | 8, RSU_1, UAV_1 | U=0.9476 / 2.6057 / 0.0106 | 0.9673 | UAV_1: U=0.9287 / 2.7054 / 0.1399 | -0.018836 | -0.023930 | edge_advantage_erased_under_kkt_resource_sharing |
| 20300_after280_slot_285 | 35 | 无 | U=0.9546 / 2.2796 / 0.0054 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20300_after280_slot_285 | 40 | 34, RSU_1 | U=0.9208 / 3.9673 / 0.0115 | 0.8163 | RSU_1: U=0.8163 / 4.7193 / 0.6984 | -0.104433 | -0.049674 | edge_no_sharing_upper_bound_not_better_than_cloud |
| 20300_after280_slot_285 | 44 | 无 | U=0.9344 / 3.2584 / 0.0135 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20300_after280_slot_285 | 47 | 无 | U=0.9348 / 3.2473 / 0.0122 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20300_after280_slot_285 | 50 | RSU_0, UAV_1 | U=0.9359 / 3.1996 / 0.0105 | 0.9433 | UAV_1: U=0.8925 / 4.0844 / 0.2108 | -0.043484 | -0.050754 | edge_advantage_erased_under_kkt_resource_sharing |
| 20300_after280_slot_285 | 51 | RSU_0, UAV_1 | U=0.9255 / 3.7159 / 0.0134 | 0.9301 | UAV_1: U=0.8735 / 4.8319 / 0.2439 | -0.051955 | -0.060056 | edge_advantage_erased_under_kkt_resource_sharing |
| 20300_after280_slot_285 | 56 | 无 | U=0.9507 / 2.4610 / 0.0085 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20300_after280_slot_285 | 58 | 无 | U=0.9650 / 1.7606 / 0.0038 | UNKNOWN | KKT不可行/无合法Edge | UNKNOWN | UNKNOWN | no_edge_service_cached |
| 20300_after280_slot_285 | 6 | 8, 24, RSU_0 | U=0.9493 / 2.5454 / 0.0066 | 0.9602 | RSU_0: U=0.8991 / 3.1586 / 0.2991 | -0.050214 | -0.073789 | edge_advantage_erased_under_kkt_resource_sharing |

## 受控 Cloud→Edge 重新卸载

G0 是 `execute_configured_slot` 的正式自然 Matching/KKT 结果。现有 `execute_configured_slot` / `optimize_task_matching` 不暴露固定单任务目标参数。G1 因此复用正式 `build_feasible_targets`，只把预先选定代表任务的目标列表限制为它真实合法的第一个 Edge 候选；从 G0 matching 出发，调用同一 `enumerate_best_operation` Transfer–Exchange 枚举、相同目标容差/Edge 平局规则、同一 KKT evaluator 重优化其余任务。每个候选分支使用新的 KKT/Node evaluation cache。这里没有改动任何 Matching/KKT 源码，也没有把局部边缘估值当 G1。

代表任务规则固定为每个状态中 task_id 字典序第一的 Cloud 任务；目标固定为正式 feasible target 列表中的第一个 Edge。选择不查看分支收益。若无合法 Edge，状态记 unavailable。Cache、UAV、车辆位置、任务与预测输入在 G0/G1 相同。
G0/G1 固定相同 Cache/UAV 动作，因此 Cache 与移动成本相消，受控反事实的 ΔSystem Return 等于 ΔTask Utility。

| 状态 | 固定任务 | 固定Edge | G0 Cloud/Edge数 | G1 Cloud/Edge数 | Δ全Slot Task Utility | 其他任务重新分配数 | 其他任务Utility变化数 | 稳定复检 |
|---|---|---|---:|---:|---:|---:|---:|---|
| 20001_after20_slot_25 | 1 | 10 | 13/4 | 12/5 | -0.028031 | 0 | 12 | converged_by_no_improvement / True |
| 20001_after150_slot_155 | 16 | 46 | 10/3 | 10/3 | -0.104998 | 1 | 10 | converged_by_no_improvement / True |
| 20001_after280_slot_285 | 12 | 46 | 10/3 | 9/4 | -0.059051 | 2 | 10 | converged_by_no_improvement / True |
| 20101_after20_slot_25 | 25 | 14 | 11/3 | 10/4 | -0.020975 | 0 | 10 | converged_by_no_improvement / True |
| 20101_after150_slot_155 | 34 | 54 | 15/1 | 14/2 | -0.245943 | 0 | 14 | converged_by_no_improvement / True |
| 20101_after280_slot_285 | 15 | 7 | 15/2 | 14/3 | -1.272191 | 0 | 14 | converged_by_no_improvement / True |
| 20201_after20_slot_25 | 19 | 6 | 9/2 | 8/3 | -0.164927 | 0 | 8 | converged_by_no_improvement / True |
| 20201_after150_slot_155 | 11 | 8 | 10/5 | 9/6 | -0.167999 | 0 | 9 | converged_by_no_improvement / True |
| 20201_after280_slot_285 | 11 | 6 | 15/3 | 14/4 | -1.541810 | 0 | 14 | converged_by_no_improvement / True |
| 20300_after20_slot_25 | 1 | 34 | 9/2 | 8/3 | -0.233565 | 0 | 8 | converged_by_no_improvement / True |
| 20300_after150_slot_155 | 45 | 24 | 10/3 | 9/4 | -1.599815 | 2 | 10 | converged_by_no_improvement / True |
| 20300_after280_slot_285 | 1 | RSU_0 | 14/4 | 15/3 | -0.368835 | 2 | 15 | converged_by_no_improvement / True |

单任务 transfer 见证固定其他匹配但完整重算 KKT；G1 表则继续让其余任务在固定目标约束下进行 Transfer–Exchange。若正收益单 transfer 与正式 G0 的稳定复检同时出现，应进一步定位候选评价差异；若仅约束重优化终值变好，这可能说明局部 Transfer–Exchange 的路径/局部最优差异，不能单凭此称实现错误。

## Cloud 占比根因与可识别边界

- **Cache 可用性：** 无 Edge 服务缓存 80/141 个 Cloud 任务；这是 Cache 缺失的直接计数。其他缓存命中节点和候选的完整清单逐任务保存。
- **通信：** 缓存但没有物理可行路径 0 个 Cloud 任务；其中被识别为坐标/信道增益/速率问题 0 个。路径可行的慢链路通过实际 uplink rate、上传时延和 KKT 分配报告。
- **资源竞争：** 33 个 Cloud 任务的所有 Edge 独占资源上界都不高于实际 Cloud 效用；28 个任务的独占 Edge 上界高于 Cloud，但 KKT 共享重分配后最佳 Edge 单任务效用不再高于 Cloud。这 28 个候选在最佳 KKT Edge 节点上的平均 CPU/BW 份额分别为 42.70% / 51.91%，支持共享资源竞争会抹去理想 Edge 优势；不单独拆分 CPU 与带宽的因果贡献，也不是 Cloud 参数错误的证明。
- **Matching：** 共找到 0 个超过零容差的合法单 transfer 全 Slot 正收益候选；自然 Matching 的稳定检查逐状态见 JSON。如果计数为 0，则样本不支持“Matching 忽略了直接更好的单步 Edge transfer”。全约束 G1 结果用于区分单步局部最优与更大组合替代。
- **参数判断：** 当前数据只能验证此模型合同下的结果，不能验证 200 Mbps、15 倍上传时延或 40 GHz 是否代表真实 Cloud 服务。没有现实网络测量/外部校准数据，因此不建议从本实验单独推出具体 Cloud 参数改值。
- **Cache/UAV 联合空间：** 本诊断有意固定 Cache/UAV，衡量的是卸载 Matching 的即时空间；它不能因果证明改变缓存/UAV 的收益。是否足够影响联合优化，需与同状态 Cache–UAV 分支输出配对后判断；当前字段保留为 UNKNOWN，避免混淆两个实验。

## 结果文件与执行边界

- 完整状态、逐任务、候选、KKT结果和原始高精度输入：[`stage6_cloud_edge_offloading_diagnosis_results.json`](stage6_cloud_edge_offloading_diagnosis_results.json)。
- 基线实际 `execute_configured_slot` 调用：12 个（12 个选定槽各独立重放）；单任务 KKT 候选：186 个；固定任务受控重优化：12 个。
- Training 数据、配置、Checkpoint 只读；Validation/Test 未读取；没有启动 PPO 训练、没有改正式系统参数/Reward/Matching/KKT/Runner；不 commit/push。临时 Runner 与临时输出在完成后清理。
