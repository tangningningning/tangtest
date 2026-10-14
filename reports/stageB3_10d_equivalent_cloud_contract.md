# Stage B3-10D：等效Cloud通信合同核对

## 冻结结论

本阶段采用抽象等效Cloud链路。Cloud与满足缓存、通信和资源条件的Edge候选共同进入`RESOURCE_AWARE_FALLBACK`下的效用竞争；不设Edge优先，也不把Cloud限制成仅在Edge缺失时使用的后备。车辆只计自身等效上传发送能耗。系统不计RSU接收、回传设备或Cloud计算能耗。

本轮没有保留显式`TV→RSU→Cloud`路径代码，也没有新增独立Cloud配置。先前显式RSU接入方案的临时实现及其配置已撤回；该方案下曾启动的Slot运行结果不适用于当前合同，已排除在本报告结论之外。没有针对当前等效Cloud合同重新运行Matching/KKT。

## Cloud公式和资源合同

对任务$v$，配置值为`R_cloud=200 Mbps`、`psi_cloud=15`、共享Cloud CPU总池`F_cloud_total=40 GHz`：

```text
T_cloud(v) = psi_cloud * D_bits(v) / R_cloud
             + L_cycles(v) / F_cloud_alloc(v)
E_vehicle(v) = P_tx(v) * D_bits(v) / R_cloud
```

`R_cloud`是每个Cloud任务的固定等效上传速率，不按同槽Cloud任务数分摊，也不是现实无线速率保证或Cloud总带宽池。`psi_cloud`只放大时延中的等效传输项，不进入车辆能耗。实际Cloud任务CPU由有限40 GHz总池共同分配，满足分配总量不超过池容量。

现有正式Task Utility继续按时延和车辆能耗归一化：

```text
Q_v = omega_T * (1 - T_v / T_ref)
    + omega_E * (1 - E_vehicle(v) / E_ref)
G = sum_v(Q_v) - C_cache - C_UAV
```

Cloud的车辆等效上传能耗进入`E_vehicle`；没有为Cloud额外加入基础设施能耗项。Reward及其权重、参考值和Cache/UAV成本没有修改。

## 源码与候选路径核对

- [`system_model.py`](../src/common/system_model.py)中的`cloud_execution()`实现`psi*D/R + L/F`，并返回`P*D/R`作为车辆能耗。
- [`cache_configuration.py`](../src/decision/cache_configuration.py)把Cloud速率、时延因子和CPU参数映射到运行时合同；Cloud上传共享开关要求为`false`，有限CPU模式验证共享池容量为正。
- [`task_allocation.py`](../src/decision/task_allocation.py)的`build_feasible_targets()`在`RESOURCE_AWARE_FALLBACK`下加入合法Edge目标及Cloud；完整Matching/KKT评价重新分配Cloud CPU，并调用同一`cloud_execution()`计算时延、车辆能耗和Task Utility。Cloud不因存在Edge候选而被屏蔽。Matching仍按全任务效用选择映射，不新增Cloud门控规则。
- [`slot_environment.py`](../src/decision/slot_environment.py)仍计算`G=Q-C_cache-C_UAV`。

没有新增RSU无线接入、共享回传、Cloud接入带宽竞争、系统总能耗统计或硬SNR门槛。

## Cloud参数来源及局限

| 参数 | 当前值 | 来源判断 | 限制 |
|---|---:|---|---|
| 每任务等效Cloud上传率 | 200 Mbps | `PROJECT_SIMULATION_ASSUMPTION`；现有配置引用文献中的200 Mbps数值作为先例 | 配置明确将其设为每任务固定等效速率。文献数值先例不能证明本模型的每任务链路具有同等物理含义或保证。 |
| Cloud时延因子`psi_cloud` | 15 | `SOURCE_UNDETERMINED`；保留现有项目实验值 | 配置说明将它解释为吸收远程链路附加时延的系数，但没有给出可核实的独立标定依据。 |
| Cloud共享CPU总池 | 40 GHz | `PROJECT_SIMULATION_ASSUMPTION` | 当前有效值来自配置合同，尚非部署测量。配置邻近注释仍残留“60 GHz”，应以有效键40 GHz为准；本任务未改配置。 |

当前配置对200 Mbps的说明将文献中的“Cloud上行速率”作为**数值先例**，同时明确本项目采用每任务固定等效速率。[配置所列文献](https://doi.org/10.1109/TVT.2025.3620810)不能据此证明本项目实现了实际无线接入、共享回传或每车保证速率。`psi=15`邻近注释还残留“固定值11”的旧说明；本轮按实际配置键15核对，没有改写配置或伪造来源。

## B3-10C及旧配置隔离

通过项目环境中的Python解释器加载并校验两份配置：

| 配置 | Edge噪声模型 | SV / RSU / UAV带宽 | Cloud合同 | Cloud CPU |
|---|---|---|---|---:|
| `system_config_comm_v2_stageB3_10c.toml` | `BANDWIDTH_DEPENDENT_NOISE_V1` | 10 / 10 / 10 MHz | 200 Mbps/任务、`psi=15`、Cloud与合法Edge自由竞争 | 40 GHz有限共享池 |
| `system_config.toml` | `FIXED_TOTAL_NOISE_V0` | 20 / 20 / 10 MHz | 相同抽象Cloud公式及候选竞争合同 | 40 GHz有限共享池 |

B3-10C的Edge通信仍按分配带宽`b*B`积分噪声并使用已更新的KKT速率导数；Cloud继续走独立固定等效速率公式，不误用Edge动态噪声模型。旧正式配置的固定总噪声行为未被本轮改写。没有改动B3-10C配置、正式配置、Checkpoint、Run、Training Pool或Manifest。

## 验证与交付边界

- Python环境：`adaptive_cache_mec`注册解释器`C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`；Conda激活入口不可用，未使用base环境或安装依赖。
- 对相关源码执行了语法编译检查；通过项目配置加载器读取并校验B3-10C和旧正式配置，并核对运行时Cloud/Edge参数映射。
- 依照最终要求，以上静态检查足以确认现有实现匹配合同，因此未重跑Episode、Matching/KKT或PPO。
- 本轮未保留代码或配置改动；仅新增本报告。B3-10C既有实现与历史结果仍按其原配置/Checkpoint谱系使用。

## 后续建议

可在明确标注该抽象Cloud口径的B3-10C配置下继续后续研究。若将来要声称Cloud端到端无线/回传物理可达或优化基础设施能耗，应另行补全其通信资源合同与可核实参数，并使用独立配置及结果谱系；本次不作这类扩展。

```ini
STAGE_B3_10D = COMPLETE
CLOUD_MODEL = ABSTRACT_EQUIVALENT_LINK
CLOUD_RATE_MBPS = 200
CLOUD_DELAY_FACTOR = 15
CLOUD_CPU_GHZ = 40
CLOUD_POLICY = FREE_COMPETITION
CLOUD_VEHICLE_ENERGY = TX_ONLY
RSU_ACCESS_MODEL_ADDED = NO
SHARED_BACKHAUL_MODEL_ADDED = NO
INFRASTRUCTURE_ENERGY_ADDED = NO
REWARD_FORMULA_CHANGED = NO
B3_10C_EDGE_MODEL_PRESERVED = YES
LEGACY_CONFIG_PRESERVED = YES
CLOUD_PARAMETER_SOURCE_STATUS = R_cloud: PROJECT_SIMULATION_ASSUMPTION; psi_cloud: SOURCE_UNDETERMINED
CODE_CHANGED = NO
PPO_TRAINING_EXECUTED = NO
RECOMMENDED_NEXT_STEP = 在冻结的B3-10C抽象Cloud合同下继续研究；物理Cloud接入/回传如需研究，另立合同与配置
```