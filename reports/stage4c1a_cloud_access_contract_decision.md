# 阶段4C-1A：Cloud上行资源合同定型

## 结论摘要

按论文模型的抽象语义，Cloud上行应解释为**每个任务车辆各自使用固定等效速率 `R_c`**，不属于Cloud全局共享的200 Mbps吞吐池。Stage3B代码已经按该语义实现。这个结论只确定仿真合同，不证明目标蜂窝网络在并发负载下能为每辆车物理保障200 Mbps。

当前通信公式可以在明确写成“固定等效速率、未建模接入竞争”的条件下冻结。Stage3B TOML把同一个200 Mbps同时描述成“每任务固定速率”和“单Cloud节点总uplink”，两者冲突；应在后续把字段说明统一为代码实际采用的语义。现有材料没有给出蜂窝小区/接入网拓扑，因而不能据此改成全局共享池或按接入网络分组的共享池。

## 证据与当前实现

| 来源 | 可确认的语义 | 不能由此推出的结论 |
|---|---|---|
| `11.pdf` PDF第2页 | 车辆先经蜂窝接入链路上传，之后通过回传网络和核心网到远程Cloud；Cloud传输不占用SV、RSU、UAV的共享带宽资源。 | 没有给出Cloud接入的聚合容量，也没有定义不同蜂窝小区/接入网络之间的资源池边界。 |
| `11.pdf` PDF第7页 | Cloud使用独立蜂窝链路；`R_c`是任务车辆到蜂窝接入网络的固定等效上行速率；Cloud链路不参加服务节点共享带宽分配。 | “独立”指与文中Edge服务节点带宽分配分开；文献没有描述蜂窝MAC调度、并发用户竞争或独占物理信道保证。 |
| `11.pdf` PDF第7–8页公式 | `T_tr(v,c)=ψ_c D_v/R_c`；车辆上传能耗为`E_tr(v,c)=p_v D_v/R_c`；完成时延为上传时延加Cloud计算时延。`ψ_c`表示回传网络和核心网附加传输时延。 | 文献支持公式结构和`ψ_c≥1`，不支持本项目固定的`ψ_c=11`数值；`ψ_c`不是带宽分配或并发竞争机制。 |
| Stage3B配置及参数依据报告 | 配置取`R_c=200 Mbps`；参数来源报告指出被引用论文Table IV有“平均Cloud上行速率200 Mbps”的数值先例。 | 平均速率数值先例不等于每车的并发保障、不等于200 Mbps聚合容量，也不能标定本项目目标网络。该来源论文的场景不同。[DOI 10.1109/TVT.2025.3620810](https://doi.org/10.1109/TVT.2025.3620810) |

本结论同时核对了`reports/stage4c0_cloud_dominance_root_cause.md`和`reports/stage3_parallel_parameter_evidence.md`。两份既有报告均指出：论文抽象是固定等效Cloud速率，数值200 Mbps有文献先例，但目标部署中的并发语义未标定。

### 源码实际合同

- `src/common/system_model.py::cloud_execution()`（约131–170行）在**每次任务调用**中令`allocated_rate = rate`，计算`communication_delay = ψ_c D/R_c`、`compute_delay = L/F_c`、`vehicle_energy = pD/R_c`。速率不按同槽Cloud任务数量拆分，`ψ_c`不乘入上传能耗。
- `src/decision/task_allocation.py`（约1888–1893、2947–2952行）把`cache_config.cloud_total_bandwidth_bps`作为`cloud_execution()`的任务速率参数传入。Cloud的`_allocate_cloud_cpu_kkt()`（约2537–2549行）只对有限CPU池分配CPU，并明确保持Cloud上行速率独立、按任务给定。
- `src/decision/cache_configuration.py`（约76–79、245–248行）同时保存`cloud_uplink_rate_bps`和`cloud_total_bandwidth_bps`；当前调用路径使用后者。`src/common/project_config.py`（约1158–1169行）要求通信速率、节点速率和`total_bandwidth_bps`相等。这个相等校验只是重复字段一致性校验，**没有**形成`Σ R_i ≤ 200 Mbps`的资源约束。
- `src/common/project_config.py`（约1093–1102行）和`cache_configuration.py`（约145–150、192–196行）拒绝`resource_sharing=true`。因此，仅把配置开关翻成`true`不能得到共享Cloud上行模型；当前没有Cloud上行资源分配器、接入域分组或相应可行性检查。

## 配置注释与实现冲突

`configs/proposed/stage3_cloud_finite_60ghz.toml`的`[nodes.cloud]`约290–296行先称`uplink_rate_bps`是“每任务固定等效上行速率”，并称上行按每任务独立链路建模；约299–301行又把来源描述为“平均Cloud上行速率”，并写成“200 Mbps作为单Cloud节点总uplink”。此外，`total_bandwidth_bps`字段名以及`project_config.py`约1169行的“唯一总带宽容量”错误信息容易让读者把它理解成共享池。

可执行语义没有这种歧义：代码将同一速率完整传给每个Cloud任务，且无总速率约束；字段`total_bandwidth_bps`在此处只是与`uplink_rate_bps`相等的兼容字段，不代表运行时总容量。配置注释需修正，但本次没有改配置或源码。

## 两种合同的适用前提

| 合同 | 适用前提 | 当前资料支持情况 |
|---|---|---|
| 每任务/任务车辆固定等效速率 | 将接入网络吞吐视为外生常量；竞争、调度和拥塞未建模，或已折算进固定的有效速率。每个Cloud任务用`R_c`计算时延和上传能耗。 | **论文公式层面支持。** `11.pdf`对任务车辆定义固定等效`R_c`且不对Cloud上行建立共享约束。代码与此抽象一致。200 Mbps是数值先例，不是目标部署保证。 |
| 共享接入资源池 | 有明确接入拓扑、容量和共享范围，并定义每个活动车辆/任务如何分到速率或无线带宽。 | **当前资料不足。** 不能把表中的平均200 Mbps直接改解释为总池容量，也不能从“单个逻辑Cloud节点”推出所有车辆共享同一接入池。 |

如果采用共享池，应该全Cloud统一共享还是按车辆所属蜂窝接入网络分组，现有`11.pdf`、配置和参数报告都无法判定。论文没有蜂窝小区ID、车辆到接入点映射、各接入域容量、调度规则或回传拓扑。全局池和分组池都需要额外架构假设，不能在本阶段代选。

## `ψ_c`与竞争的边界

`ψ_c=11`在当前合同中仅把`D/R_c`扩展为Cloud传输时延，用于等效回传/核心网附加延迟；它不表示接入容量、带宽份额、排队等待或多车辆竞争。当前车辆上传能耗仍只按无线上传项`pD/R_c`计算。若未来明确共享无线接入池，应由每任务实际分配速率进入时延与能耗计算，不能用调大或调小`ψ_c`代替资源约束。若要显式表示排队或回传竞争，还需另行定义这些资源及其时延/能耗口径；现有资料没有足够依据添加。

## 冻结与后续最小范围

**可以冻结当前论文层面的固定等效速率公式，但需在合同文字中说明：每个活动Cloud任务均使用固定`R_c`，不建模Cloud接入间的竞争；200 Mbps是仿真等效值而非物理保障。** 不能把它冻结为目标部署的已验证无线合同。`ψ_c=11`和200 Mbps的物理标定仍属未确认假设。

如果后续采用当前固定速率合同，最小变更仅是统一TOML注释、字段名称/别名说明及配置校验报错文案；不需要改Matching/KKT、`cloud_execution()`公式、车辆能耗或CPU可行性约束。

如果研究者将来决定建模共享池，至少需要先确认：

1. 共享的对象及范围：一个全局接入域，还是多个蜂窝/接入网络分组；若分组，提供车辆到接入域的映射。
2. 每个资源池的容量和单位。200 Mbps是平均单用户速率，不能未经新证据直接当作池容量；若建模无线带宽，还需频谱带宽、信道/噪声和调度语义。
3. 并发任务间的分配规则，以及同槽Matching候选评价是否联合重算受影响任务的链路分配。

此后最小代码范围会包括配置和`CacheOptimizationConfig`字段、`task_allocation.py`中的Cloud速率分配及候选/Matching评价、`system_model.py`的按分配速率计算，并补齐资源和可行性记录。对每个任务必须用同一个已分配`R_i`计算`ψ_cD_i/R_i`与`p_iD_i/R_i`；资源约束应按已确认的池范围检查（例如总速率池为`Σ_i R_i≤C`，或无线频谱池为带宽份额和约束），并与Cloud CPU池分配共同保证Matching效用和可行性一致。这里不预设池容量、作用域或分配算法。

本次只读分析：已在`adaptive_cache_mec` Conda环境中激活；没有运行仿真、训练、Validation或Test，没有修改源码、TOML或既有文件。本报告为本次唯一新增文件。

```makefile
CLOUD_UPLINK_CURRENT_SEMANTICS = PER_TASK_FIXED_200_MBPS_NO_CROSS_TASK_POOL
CLOUD_UPLINK_INTENDED_SEMANTICS = FIXED_EQUIVALENT_R_C_PER_TASK_VEHICLE_AT_MODEL_LEVEL; PHYSICAL_GUARANTEE_UNDETERMINED
LITERATURE_SUPPORT = FIXED_R_C_ABSTRACTION_AND_CLOUD_LINK_OUTSIDE_EDGE_POOL; 200_MBPS_IS_NUMERIC_AVERAGE_PRECEDENT_ONLY
CONFIG_CODE_CONFLICT = YES (TOML_SAYS_PER_TASK_AND_SINGLE_CLOUD_TOTAL; CODE_APPLIES_FULL_RATE_PER_TASK)
MODEL_MODIFICATION_NEEDED = NO_FOR_FORMULA_IF_ABSTRACT_CONTRACT_IS_ACCEPTED; CONFIG_DOCUMENTATION_CLARIFICATION_REQUIRED
MINIMUM_MODIFICATION_SCOPE = FIXED_RATE: TOML_COMMENTS/FIELD_ALIAS_AND_VALIDATION_TEXT; SHARED_POOL: CONFIG+TASK_ALLOCATION/KKT+SYSTEM_MODEL+RESOURCE_FEASIBILITY/METRICS
READY_FOR_STAGE4C1B = YES_FOR_FIXED_RATE_CONTRACT_FREEZE_AND_DOC_CLARIFICATION; SHARED_POOL_BRANCH_REQUIRES_NEW_ARCHITECTURE_EVIDENCE
```
