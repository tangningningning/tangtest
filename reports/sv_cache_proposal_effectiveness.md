# SV缓存候选需求代理有效性检查

## 范围与对照设置

只检查本地正式候选生成/特征代码及Training Episode 20001、20150、20300的Slot 50、150、250，共9个预测起点。Validation/Test、PPO、完整Matching/KKT均未运行。Python解释器为`C:/software/Anaconda_envs/envs/adaptive_cache_mec/python.exe`，`sys.prefix`指向`adaptive_cache_mec`。

正式配置读得`T_ref=30.473728355325758 s`、`E_ref=5.1574642506299275 J`，与本任务指定值一致。每个episode使用同一份正式seed生成的初始Cache、同一初始UAV坐标；每组原方案/参考方案使用完全相同的预测场景、车辆状态、Cache初值和UAV位置。当前预测场景为20个预测样本、5个horizon；只读`prediction_scenarios.npz`中的`rho_hat`派生请求、预测输入/CPU工作量和预测位置，没有读取未来真实任务。RSU/UAV候选需求评分未替换。

## 代码链路

- `src/decision/cache_configuration.py:546-565, 689-750`：SV proposal权重为`max(0, 1-distance/100m)`；距离达到100m时权重归零。此权重只用于候选proposal，不是实际链路硬距离门限。
- `src/decision/cache_configuration.py:753-901`：同一权重进入节点请求数、预测输入量和CPU工作量特征。
- `src/decision/cache_configuration.py:994-1099, 1102-1232`与`src/ppo_simple/primitive_control.py:318-426`：先枚举容量合法的节点目标，再由需求锚点、低重配置目标和多样性补齐规则选择不超过64个UPDATE。正式Rollout的`src/ppo_simple/rollout.py:560-681`逐节点生成候选并建立动作Mask；经济特征另含新增/移除服务需求量（`src/ppo_simple/primitive_control.py:191-315`）。当前生成器没有把所有目标简单按单一需求分数排序；需求影响锚点、候选特征和保留顺序，最终策略仍根据候选特征给动作打分。
- 实际Edge速率、上传能耗、完整时延和任务效用由`src/decision/task_allocation.py:1807-1904`及`src/common/system_model.py:173-260, 453-479`计算。SV path没有100m截断；任务效用使用当前正式`T_ref/E_ref`和`omega_T/omega_E`。

## 轻量物理参考评分

参考评分是SV相对Cloud的局部服务收益，非正式Matching/KKT或全局系统目标：对每个预测scenario/horizon，Cloud基线将有限40GHz CPU按该样本预测到达任务数等分；每个Cloud任务使用正式20Mbps等效上行，13倍系数只作用于传输时延。单独评估某SV缓存某一服务时，该服务的预测任务在该SV内等分CPU和10MHz带宽，其余任务留在Cloud；对候选Cache行则将该行涉及的服务任务合并后等分该SV CPU/带宽。资源份额总和不超过节点能力；输入、计算量、位置来自决策时可见的预测scenario。每任务以正式Reward效用比较Edge与Cloud，按预测样本均值和配置的horizon折扣聚合。没有把正速率直接当作收益。

该参考评分刻意固定为“单SV服务组等分资源、其他任务走Cloud”，不求解任务分配，也不计Cache更新成本；因此它只能衡量相对Cloud的局部服务价值，不能代替完整系统净收益。单服务排序只包含在SV 12,000 MB容量内可存放的服务`{1,4,5,7,8,9}`；物理参考目标仅取正的服务收益/存储量，并检查目标容量。其余节点的配置固定，但局部SV-vs-Cloud对照不把其他Edge节点重新匹配给任务。

## 结果

### 特征衰减与排序

在9个起点、45个SV-起点组合中，当前100m权重后的预测请求质量总和，仅为未加该权重预测请求质量的：

| Training Episode | SV可缓存服务加权/未加权需求比例均值 | 6种可缓存服务的排名Spearman中位数 | 原需求Top-3与物理收益Top-3平均交集数 |
|---|---:|---:|---:|
| 20001 | 7.43% | -0.683 | 0.80 / 3 |
| 20150 | 5.94% | -0.714 | 0.73 / 3 |
| 20300 | 5.56% | -0.657 | 0.67 / 3 |

上表的需求比例只计算容量上可存放的六种服务。物理收益排名是“较不负”的相对顺序；在270个SV-可缓存服务-起点组合中，单服务相对Cloud收益均不为正。负相关和较低Top-3交集说明100m需求代理的服务顺序与本固定参考收益顺序不一致，但不代表出现了被遗漏的正收益热点。服务0、2、3、6单独就超过SV缓存容量，因此不纳入可执行单服务排名。

例：Slot 150，20001的SV 46上，100m需求锚点为服务`{1,4}`，局部参考目标为空；在单SV-vs-Cloud参考中，锚点配置相对“这些任务都由Cloud处理”的效用变化为`-6.223`，空配置为`0`。20150/SV 21为`{1,5}`与空配置（`-7.865` vs `0`）；20300/SV 8为`{1,4}`与空配置（`-7.325` vs `0`）。这些仅是固定参考分配的相对Cloud任务效用，不是相对实际当前多节点Cache状态，也不含Cache成本或Matching/KKT实测值。

### 候选集合覆盖

所有45个SV-起点组合中，合法单节点UPDATE均为10个；正式Rollout生成器也保留10个，配置上限为64个UPDATE（另有KEEP）。原方案与同预算物理参考方案的单节点合法候选集合重合率为100%。物理参考锚点（包括空Cache）均存在于原方案保留集合中。换言之，本次初始Cache样本中，需求特征/锚点会变化，但64上限没有截掉任何合法SV目标；参考方案没有发现被原方案漏掉的高价值单节点配置。

由于合法候选少于上限，候选保留的reference-score regret为0；**正式Matching/KKT下的系统目标损失没有测量**。且局部参考对单个SV的正收益为零，所以没有触发进一步Matching/KKT对照的依据。本结论不外推到Cache状态变化后合法目标超过64、服务目录改变或不同的预测/资源负载。

## 判断

归类为 **A：100m代理显著压低SV需求特征，代理服务排序也与固定物理相对收益顺序不一致，但在所抽Training场景的初始Cache状态下没有影响保留候选集合**。这一参考策略下未出现相对Cloud为正的单SV服务收益；而且该评分不含Cache成本、不考虑RSU/UAV任务匹配。因此数据不支持C（已证实实际目标损失），不建议据此立即修复算法。若后续正式状态下合法UPDATE候选数超过64，或需要判断全系统实际收益，应再用少量被截断候选做正式Matching/KKT评价。

```ini
SV_100M_FEATURE_MISMATCH = YES
CANDIDATE_SET_AFFECTED = NO
HIGH_VALUE_CANDIDATE_MISSED = NO
MEASURED_OBJECTIVE_LOSS = UNRESOLVED

ALGORITHM_REPAIR_REQUIRED = NO
RECOMMENDED_REPAIR = NONE

SOURCE_CHANGED = NO
FORMAL_CONFIG_CHANGED = NO
FORMAL_DATA_CHANGED = NO
PPO_TRAINING_EXECUTED = NO
```
