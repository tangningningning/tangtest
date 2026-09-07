# Cache 收益敏感性分析

日期：2026-09-05。项目：C:/Users/tangning/Desktop/code/adaptive_cache_mec。

## 1. 实验边界与控制设计

本报告分析现有缓存优化器对系统参数的响应，不是 PPO 学习效果实验。所有参数覆盖只存在于各实验进程的内存副本；不写回 TOML，不修改算法、reward、service catalog 或 artifact 生成逻辑。

- 每个独立设置连续执行 slot 0–49（时间 100–149 s），单个短 episode；9 个正式设置共 450 个有效 slot，另做 50-slot 旧审计初始化复现。三项实验共享同一个正式 baseline，未把重复 baseline 当作独立样本。
- 固定实验/任务 seed=4401；保留任务生成器已有的独立随机流、预测场景 seed=20260820 和车辆预测输入。
- 各正式设置从相同的 `uniform_feasible` 缓存与配置中的 UAV 初始位置开始。每个设置内部传递 C(t+1)、Q(t+1)，不逐 slot 重置。
- 重放同一参考触发序列：9 个缓存动作与 2 个 UAV 移动动作。参考序列来自 seed=4401 的未训练 `ConditionalTriggerActorCritic` 前向采样，使用已有预测观测；没有反向传播、optimizer step 或 checkpoint 更新。变体不重新采样策略。
- 固定的是触发输入和初始状态，不是优化器输出；缓存优化、UAV 位置优化、实际 Matching/KKT 仍照常执行。因此 CPU 变化可以间接改变后续节点状态。
- 每组有 209 次缓存 trigger：SV 112、RSU 42、UAV 55；TV 是固定 local cache，没有动态 trigger。合计 trigger 比例为 209/(9×50)=46.4444%。各组 trigger 相同是控制设计，不能解释为这些参数对学习后的触发概率没有影响。
- 每组均有 682 个实际到达任务；Cloud/费用组完整复用磁盘任务与预测场景。Popularity 组保留每 slot 到达车辆集合、到达随机数与扰动随机数，但请求服务改变会改变服务对应的 D/L，故不同 Popularity 组的 reward 变化不能全部归因于缓存。

参考动作序列 SHA-256（对 Python 默认 `json.dumps(action_trace)` 的 UTF-8 字节计算）：

`4dfedc501704a4f631f393f366c37cfacc4f98be060f43121524826db8f36382`

### 固定 baseline

| 项目 | 所有实验共同设置（被研究参数除外） |
|---|---|
| 规模 | 25 vehicles = 5 SV + 20 TV；2 RSU；2 UAV；10 services |
| CPU | TV/SV 1 GHz；RSU/UAV 5 GHz；Cloud baseline 20 GHz |
| Cloud | psi_c=11；总 uplink=200 Mbit/s |
| 目标 | delay=0.6、energy=0.4；T_ref=0.65 s；E_ref=0.065 J |
| 缓存容量，MB | TV 5581.583442；SV 16744.750325；RSU 44652.667533；UAV 33489.500649 |
| Popularity baseline | Zipf ζ=0.8；epoch=20 slot；每 epoch 1 次相邻交换/局部移动 |
| 请求生成 | arrival=0.7；既有 service 输入基准及 ±20% input、±10% load jitter |
| 预测 | 冻结既有 GRU；H=5、S=20；不重新训练预测器 |
| 缓存目标 | discount=0.95；lambda_cache=0.1；严格改善容差 1e-9 |

### 初始缓存的口径修正

[正式训练入口](C:/Users/tangning/Desktop/code/adaptive_cache_mec/src/ppo/ppo_training.py:4108) 使用 [uniform feasible 初始化](C:/Users/tangning/Desktop/code/adaptive_cache_mec/src/decision/cache_configuration.py:1431)。此前最小审计入口 [run_minimal_sanity](C:/Users/tangning/Desktop/code/adaptive_cache_mec/src/ppo/ppo_rollout.py:2051) 使用 `deterministic_initial_cache`，两者不同。本报告不修改任何入口，只明确采用前者作为正式参数比较的初始状态。

| 节点 | 正式比较的初始 service ID |
|---|---|
| SV 23 | 1, 7 |
| SV 87 | 6 |
| SV 43 | 1, 9 |
| SV 75 | 6 |
| SV 58 | 4, 8 |
| RSU_0 | 2, 3, 4, 5 |
| RSU_1 | 2, 4, 5, 7, 9 |
| UAV_0 | 5, 6, 7 |
| UAV_1 | 2, 4, 5, 7 |

| 相同物理参数下的初始化对照 | trigger | change | 邻居评估次数 | 命中率 | mean slot reward |
|---|---:|---:|---:|---:|---:|
| 旧最小审计的确定性初始化（本次复现） | 209 | 0 | 2864 | 72.5806% | -113.795165 |
| 正式 uniform feasible 初始化 | 209 | 0 | 4234 | 89.8827% | -110.623638 |

后续所有敏感性对比均使用第二行作为 baseline；初始化对照不混入参数效应。

## 2. 统计定义

- `cache_change`：最终 C(t+1) 与 C(t) 不同的“节点-slot”数量。一个节点同一 slot 内做两次搜索操作仍只算一次实际重配置；另计改变的 0/1 项数。
- `hit ratio`：对实际到达任务，当前 TV local 或至少一个动态节点缓存其服务的比例，即存在非 Cloud 可执行目标。它不是最终选择 edge 的比例，也不包含 Cloud fallback。before/after 使用同一 slot 的同一批任务；窗口命中率按任务数量加权。
- `delta_hit`：同 slot after−before，不是与另一实验的命中率之差。
- Cloud/RSU/UAV 选择比例：实际 Matching 最终选择该类型的任务数/682。SV、TV local 单独统计，不并入 RSU/UAV。
- `reward`：实际 `system_utility` 的 50-slot 算术平均，不是 GAE、训练 loss 或预测 surrogate。
- `candidate`：现有优化器实际评估的邻居次数，包含外层重复评估，不是去重后的状态数。Knapsack 候选单列。
- “原始 replacement net gain”与“已接受收益”分开。后者为优化器返回的 `final_objective-initial_objective`；没有接受操作时为 0，不代表所有原始候选也为 0。数值绝对值不超过 1e-9 按零解释。

既有实现 [cache objective](C:/Users/tangning/Desktop/code/adaptive_cache_mec/src/decision/cache_configuration.py:872) 的实际口径为：

```text
J(C) = predicted_service_utility(C) - Cost(C_old, C)
candidate net gain = J(C_candidate) - J(C_current)
predicted utility gain = net gain + Cost(candidate) - Cost(current)

Cost = Σ_m [eta_m × 1(C_m != C_old,m)
            + lambda_cache × xi_m × Σ_f size_MB,f × max(C_mf-C_old,mf, 0)]

G(t) = Σ_actual tasks [0.6×(1-delay/0.65) + 0.4×(1-energy/0.065)]
       - actual cache cost - actual movement cost
```

费用按实际重配置而非单纯 trigger 扣除；只有新增服务产生加载费用。[费用实现](C:/Users/tangning/Desktop/code/adaptive_cache_mec/src/common/system_model.py:226)、[实际 reward](C:/Users/tangning/Desktop/code/adaptive_cache_mec/src/decision/slot_environment.py:222) 均未修改。记录器仅旁路读取原函数的输入/返回值并原样返回，不影响候选排序与接受条件。

## 3. 实验一：Cloud 竞争力

只在内存中同步覆盖 `nodes.cloud.cpu_cycles_per_s` 与 `total_cpu_cycles_per_s` 为 20/10/5 GHz；其余物理参数和所有任务输入保持不变。

| Cloud CPU | Cloud 选择 | RSU 选择 | UAV 选择 | RSU+UAV | trigger | cache change | mean reward | reward−baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 GHz | 289/682 (42.3754%) | 78/682 (11.4370%) | 234/682 (34.3109%) | 45.7478% | 209 | 0 | -110.623638 | 0.000000 |
| 10 GHz | 267/682 (39.1496%) | 81/682 (11.8768%) | 243/682 (35.6305%) | 47.5073% | 209 | 0 | -127.948120 | -17.324482 |
| 5 GHz | 221/682 (32.4047%) | 86/682 (12.6100%) | 266/682 (39.0029%) | 51.6129% | 209 | 1 | -148.239254 | -37.615616 |

其余执行路径：20/10/5 GHz 的 SV 任务数为 81/91/107，TV local 为 0/0/2。比例按最终路径统计，不把多候选可执行集合重复计数。

| Cloud CPU | 邻居评估次数 | 正 utility gain 候选次数 | 最大 utility gain | 正 net gain 候选次数 | 最大 net gain | 优化器最大已接受收益 |
|---|---:|---:|---:|---:|---:|---:|
| 20 GHz | 4234 | 0 | 0.000000 | 0 | -0.100000 | 0.000000 |
| 10 GHz | 4234 | 0 | 0.000000 | 0 | -0.100000 | 0.000000 |
| 5 GHz | 4328 | 140 | 1.864515 | 2 | 1.169642 | 1.221335 |

### 5 GHz 组的实际更新

- slot=38，只有 UAV_0 发生重配置：`[5,6,7] → [2,4,5,7]`。
- 删除 service 6，新增 service 2、4；改变 3 个缓存位。内部接受 2 次局部搜索操作，但最终只计 1 个节点-slot 更新。
- 更新后使用容量 31293.891707 MB，不超过 UAV 容量 33489.500649 MB。
- 加载成本为 2.067582075，控制成本为 0.15，实际 cache cost=2.217582075；其他 slot cache cost 为 0。
- 窗口 hit before/after 均为 89.8827%，更新当 slot 的 delta_hit=0。服务 2/4 本已由其他节点覆盖，变化的是可选择的执行节点；因此覆盖率不变与预测执行效用正收益并不矛盾。
- 5 GHz 组 140 次邻居评估有正的费用前 utility gain，只有 2 次具有正净收益。另有 28 次 knapsack 评估具有正 utility gain，但其净收益均未超过 0。
- Cloud 变慢后总体 reward 更低，不能因为缓存更新增多就认定 5 GHz 是更好的 baseline。该组还会改变卸载/资源分配及 UAV 优化结果，不把组间 reward 差值解释为单次缓存更新的因果收益。

## 4. 实验二：Cache cost

同时把 SV/RSU/UAV 的 `xi_ca` 与 `eta_ca` 乘以 0.5/1/2，保持 `lambda_cache=0.1`。这样加载与控制两部分费用都按同一倍数变化；不是只缩放其中一项。

| 类型 | baseline xi_ca /MB | baseline eta_ca |
|---|---:|---:|
| SV | 0.0010 | 0.20 |
| RSU | 0.0007 | 0.10 |
| UAV | 0.0012 | 0.15 |

| 倍率 | 邻居次数 | min net gain | mean net gain | max net gain | max utility gain | change | hit before/after | 最大 delta_hit | mean reward |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0.5× | 4234 | -1.108159 | -0.363289 | -0.050000 | 0.000000 | 0 | 89.8827% / 89.8827% | 0.000000 | -110.623638 |
| 1× | 4234 | -2.216318 | -0.726578 | -0.100000 | 0.000000 | 0 | 89.8827% / 89.8827% | 0.000000 | -110.623638 |
| 2× | 4234 | -4.432637 | -1.453157 | -0.200000 | 0.000000 | 0 | 89.8827% / 89.8827% | 0.000000 | -110.623638 |

三个费用组均有 209 次 trigger，全部没有正净收益候选、已接受收益和实际缓存费用均为 0。每组另外评估 209 次 knapsack 候选，也均无正 utility gain / net gain。

判断：在 20 GHz baseline 的本次窗口内，降低费用没有创造缓存执行收益；费用不是“把大量正收益全部抵消”的唯一根因。费用增大也不会额外降低本次 reward，因为没有实际缓存变化。这个结论不能外推成所有 CPU/请求设置对费用都不敏感；本次没有做 CPU×cost 交互实验。

## 5. 实验三：Popularity 敏感性

固定 Cloud=20 GHz、cache cost=1×、其余参数不变。按已确认设置分别测试 Zipf 指数 ζ=0.4/0.8/1.2，以及 popularity epoch 周期 5/20/50 slot。每组在内存中重新生成请求概率、请求服务与 workload 字段，并用同一个冻结 GRU checkpoint 进行因果预测；不写回 `popularity.csv`、`tasks.csv` 或 `prediction_scenarios.npz`。车辆集合、车辆位置和到达车辆集合保持一致。

| 设置 | 真实 top1变化次数 | 预测 top1变化次数 | 真实 ranking变化次数 | top3集合变化次数 | top-1概率(slot 0) | cache change | hit before/after | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ζ=0.4, epoch=20 | 0 | 10 | 2 | 0 | 0.175499 | 0 | 88.7097% / 88.7097% | -133.597855 |\n| ζ=0.8, epoch=20 | 0 | 0 | 2 | 0 | 0.280496 | 0 | 89.8827% / 89.8827% | -110.623638 |\n| ζ=1.2, epoch=20 | 0 | 0 | 2 | 0 | 0.405233 | 0 | 92.0821% / 92.0821% | -90.338898 |\n| ζ=0.8, epoch=5 | 3 | 7 | 9 | 2 | 0.280496 | 0 | 87.2434% / 87.2434% | -126.772317 |\n| ζ=0.8, epoch=50 | 0 | 0 | 0 | 0 | 0.280496 | 0 | 89.2962% / 89.2962% | -112.248048 |\n
| 设置 | 实际 Cloud | 实际 RSU | 实际 UAV | 实际 SV | 邻居次数 | max net gain | max utility gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| ζ=0.4, epoch=20 | 300 | 80 | 203 | 99 | 4234 | -0.100000 | 0.000000 |\n| ζ=0.8, epoch=20 | 289 | 78 | 234 | 81 | 4234 | -0.100000 | 0.000000 |\n| ζ=1.2, epoch=20 | 267 | 71 | 271 | 73 | 4234 | -0.100000 | 0.000000 |\n| ζ=0.8, epoch=5 | 307 | 77 | 210 | 88 | 4234 | -0.100000 | 0.000000 |\n| ζ=0.8, epoch=50 | 283 | 79 | 239 | 81 | 4234 | -0.100000 | 0.000000 |\n
观察：

- ζ 从 0.4 增至 1.2 时，窗口 hit ratio 从 88.7097% 提高到 92.0821%，Cloud 选择数从 300 降到 267，但两组都没有 cache change；max net gain 均为 -0.1。
- epoch=5 产生了 9 次真实 ranking 变化、7 次预测 top-1 变化，但仍无正 net gain；epoch=50 在窗口内无 ranking 变化，也无 cache change。
- 因此请求分布确实改变了服务请求、hit ratio、执行路径和 reward，但在当前 20 GHz Cloud / 1× cost 组合下，变化不足以跨过缓存更新接受阈值。
- Popularity 组的 reward 同时受服务请求对应的 input/cycle workload 与路径选择影响，不能单独当作 cache 更新收益。

## 6. 综合判断

**结论：B——当前 baseline 参数使缓存学习/更新信号不足；不支持 C。**

依据：

1. 固定 seed、固定触发序列和相同正式初始缓存下，20 GHz baseline 的 4234 个邻居候选没有正 utility gain，所有 net gain 均为负，最大为 -0.1，50 个 slot 实际更新为 0。
2. 仅把 Cloud 降到 5 GHz，就出现 140 个正的费用前 utility gain、2 个正 net gain，并产生 1 个真实 cache change。这表明 cache optimizer 的候选状态、utility 计算、容量约束、状态传递和实际重配置链路都可以工作；不是“实现永远无法更新”。
3. 0.5×/1×/2× cost 在 20 GHz 下均无更新，说明 baseline 的主要瓶颈不是单纯 cost 缩放。当前候选的费用前 utility gain在数值容差内为零，只有 Cloud 变慢时才明显出现正 utility improvement。
4. ζ 和 epoch 改变了请求分布与执行路径，但本次窗口中仍没有正 net gain；因此 popularity 非平稳性本身并不足以克服当前 baseline 的 Cloud 竞争力和缓存成本门槛。
5. 5 GHz 更新的 `delta_hit=0` 并不表示更新无效：新增 service 2/4 已由其他节点覆盖，变化的是可选执行节点，预测 utility gain 来自路径/执行效用而非“覆盖任务数量”增加。

这不是 A“稀疏更新完全合理”的充分证据：如果稀疏只是合理自然现象，参数从 20 GHz 改为 5 GHz 不应在同一触发序列中从 0 变为 1。也不是 C 的证据：当前静态链路和低 Cloud CPU 对照均显示更新机制能被激活。

### 解释范围与限制

- 该实验是 50-slot、未训练策略的 replay/短时敏感性分析，没有 PPO training，因此不能据此判断训练后触发策略的收敛频率。
- 触发序列被固定用于单因素隔离；它不随 CPU、cost 或 popularity 重新学习。因此结果回答的是“给定同一触发机会时系统参数能否把更新变成正收益”，不是参数对 trigger policy 发生率的完整因果结论。
- 当前 cache surrogate 对候选的 Cloud 资源默认采用单任务分析，而实际 Matching 对同 slot Cloud 任务进行共享；这属于现有模型口径差异，未在本任务中修改。相关报告只能作为参数压力证据，不能替代正式训练/完整实验。
- 不建议仅依据本报告直接修改 baseline 或 reward。若要继续定位，应在保持算法不变的前提下做 Cloud CPU × cache cost 的二维实验，并延长固定 seed 的短 rollout，再进入 PPO 前进行正式设计。

## 7. 完整性检查

- 未修改 `system_config.toml`、cache algorithm、reward、PPO 或 artifact。
- 各实验结束时对 36 个受保护源码/config/artifact 输入做 SHA-256 对照，所有实验均报告 `changed=[]`。
- 未运行 PPO training、长时间 SUMO 仿真或完整实验。
- 本报告是本任务唯一新增文件；没有生成临时测试脚本、日志、JSON/CSV 或 `__pycache__` 文件。
