# Stage B3-10I：无线链路可靠性与RSU覆盖合理性分析

## 执行范围与复现口径

Conda激活入口在当前终端无法启动。实际使用已验证的 `adaptive_cache_mec` 环境解释器 `C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`，其 `sys.prefix` 指向同一环境，Python 3.11.15。没有改用base环境或修改依赖。

只读取Training Pool清单及Episode 20001–20300的 `tasks.csv`、`trajectory.csv`、`vehicle_roles.json`，共对齐 **1,260,159** 个到达任务；到达任务位置缺失数为0。按正式Training任务到达位置离线计算一次，调用项目 `ground_channel_gain()` 与 `uplink_rate_and_derivative()`。没有运行SUMO、Matching/KKT或PPO。FULL_BANDWIDTH是每任务独占10 MHz的理论参考；EQUAL_SHARE按同一Episode、Slot和RSU内的真实到达任务数等分10 MHz，未人为指定并发数。

用于分析的候选版本是 `configs/proposed/system_config_comm_v2_stageB3_10h_equal_share.toml`，不是默认正式主线配置。此前B3-10E等效参数和B3-10H候选合同保留，未被覆盖。

## 当前Ground通信合同

| 参数 | B3-10H当前值 | 含义及依据 |
|---|---:|---|
| RSU数 | 2 | RSU_0、RSU_1 |
| RSU带宽 | 10 MHz | 两个RSU各自独立正交频域资源池 |
| Ground载频 | 5.9 GHz | 配置中显式记录；参考增益已含参考载频损耗，不再叠加FSPL |
| 车辆发射功率 | 0.199526 W（23 dBm） | 固定任务总功率，不随带宽份额缩放 |
| 参考信道增益 | `βg=1.6368×10⁻⁵` | 1 m参考损耗47.8600 dB，与5.9 GHz自由空间1 m损耗47.8648 dB相差−0.0048 dB |
| 路径损耗指数 | `αg=2.75` | 项目配置称为VEC地面链路文献校准基线，不是硬件实测值；当前资料未提供可核验的场景标定记录 |
| 噪声PSD / 接收NF | −174 dBm/Hz / 9 dB | 9 dB是有效仿真基线，不是实测接收机参数 |
| 完整10 MHz积分噪声 | −95.000 dBm | `−174 + 10log10(10 MHz) + 9` |
| 干扰 | 不显式建模 | 配置采用独立正交资源池、`interference=NONE` |

代码中的信道和速率为：

```text
h(d) = βg · d^(-αg)
w = b · B
N(w) = N0 · w · 10^(NF/10)
SNR(w) = Ptx · h(d) / N(w)
R(w) = w · log2(1 + SNR(w))
```

这是固定总发射功率下、随KKT带宽份额积分噪声的Shannon速率代理。它不是功率谱密度固定的发送模型，也不包含同频干扰。Ground距离采用车辆与节点坐标的二维水平欧氏距离，RSU的10 m高度没有进入Ground距离。`system_model.py::ground_channel_gain()`使用`βg·d^-αg`，没有1 m最小距离截断；Training中有16个任务距最近RSU小于1 m，比例约0.0013%，这些点属于低占比但应记录的参考距离外推边界。

实际Edge候选判定要求服务已缓存、坐标和角色有效、节点CPU和带宽为正、信道增益及满带宽速率为有限正数。它没有最低SNR、最低速率、BLER/PER或outage门槛。RSU配置的250 m是候选评分/预测服务性代理尺度，Matching实际链路不把它用作硬距离门槛。超过250 m不等于本模型判定链路失败。

源码位置：`src/common/system_model.py:36`、`:57–135`；`src/decision/task_allocation.py:1740–1826`；配置位置：`configs/proposed/system_config_comm_v2_stageB3_10h_equal_share.toml:296–345`。RSU的250 m软范围见同配置`:183–186`。

## RSU部署与Training车辆空间分布

两个RSU的坐标均落在Training任务车辆轨迹坐标包络内，但相距约1.91 km。轨迹包络只用于描述实际任务车辆运动范围；本次未把二维包络误当作道路拓扑或无线覆盖图。

| RSU | 坐标 (x, y) m | 关联任务数 | 比例 | 关联距离均值 | P50 | P90 | P95 | 最大值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RSU_0 | (3364.03, 901.06) | 825,519 | 65.51% | 540.77 m | 621.62 m | 919.26 m | 934.52 m | 1,236.26 m |
| RSU_1 | (5179.65, 1491.59) | 434,640 | 34.49% | 650.40 m | 689.47 m | 950.57 m | 1,025.35 m | 1,236.53 m |

车辆任务轨迹坐标包络为 `x=3059.57–5210.82 m`、`y=231.94–1733.99 m`；RSU间直线距离1,909.24 m。两个RSU采用相同`βg`和`αg`，因此最大参考信道增益RSU与最近RSU完全相同；本次1,260,159个任务没有关联差异。当前空间样本更常接近RSU_0，RSU_1关联任务较少且距离分布略远。

| 最近RSU距离 | 任务数 | 占比 | FULL接入时延占比 | EQUAL_SHARE接入时延占比 |
|---|---:|---:|---:|---:|
| 0–100 m | 61,998 | 4.92% | 0.13% | 0.63% |
| 100–250 m | 78,225 | 6.21% | 0.52% | 1.59% |
| 250–500 m | 361,028 | 28.65% | 7.31% | 12.37% |
| 500–1000 m | 714,361 | 56.69% | 79.89% | 75.76% |
| >1000 m | 44,547 | 3.54% | 12.15% | 9.65% |

超过250 m的任务为1,119,936个，占 **88.8726%**，与B3-10E统计口径和数值一致；它们贡献了99.35%的FULL接入时间和97.78%的EQUAL_SHARE接入时间。这里的250 m仅为分布分箱边界及预测代理尺度，不是覆盖结论。约60.22%的任务超过500 m，说明Training任务长期分布在RSU周边较远区域。仅凭这些坐标统计，不能判定RSU违反了某个尚未定义的覆盖目标；若研究目标是低时延、可靠的道路覆盖，现有部署需要结合道路级覆盖标准另行评价。

## 代表距离下的理论链路能力

下表FULL列使用完整10 MHz。EQUAL_SHARE列使用真实Training的任务加权并发分布，其中`n`的P10/P50/P90为4/8/12，范围1–18；具体“中位并发”行取`n=8`。这不是对真实正式KKT最优分配的替代。噪声会随份额变化，表中的EQUAL_SHARE速率仍来自项目正式数学函数。

| 距离 | 路损 | 接收功率 | FULL噪声 | FULL SNR | FULL速率 | 1 Mbit时间 / 车辆能耗 | EQUAL `n=8`噪声 / SNR | EQUAL `n=8`速率 / 1 Mbit时间 / 能耗 | EQUAL速率P10/P50/P90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 m | 102.86 dB | −79.86 dBm | −95.00 dBm | 15.14 dB | 50.729 Mbps | 0.0197 s / 0.00393 J | −104.03 dBm / 24.17 dB | 10.044 Mbps / 0.0996 s / 0.0199 J | 7.100 / 9.640 / 16.631 Mbps |
| 250 m | 113.80 dB | −90.80 dBm | −95.00 dBm | 4.20 dB | 18.593 Mbps | 0.0538 s / 0.01073 J | −104.03 dBm / 13.23 dB | 5.576 Mbps / 0.1793 s / 0.0358 J | 4.145 / 5.386 / 8.419 Mbps |
| 500 m | 122.08 dB | −99.08 dBm | −95.00 dBm | −4.08 dB | 4.758 Mbps | 0.2102 s / 0.04194 J | −104.03 dBm / 4.95 dB | 2.556 Mbps / 0.3913 s / 0.0781 J | 2.075 / 2.495 / 3.303 Mbps |
| 1000 m | 130.36 dB | −107.36 dBm | −95.00 dBm | −12.36 dB | 0.814 Mbps | 1.2279 s / 0.24499 J | −104.03 dBm / −3.33 dB | 0.688 Mbps / 1.4532 s / 0.28995 J | 0.634 / 0.682 / 0.747 Mbps |

等份行中`n=8`对应1.25 MHz/任务。真实任务加权并发中位数为8；`n=4`和`n=12`是同一Training负载的P10/P90并发点，其积分噪声分别为−101.02和−105.79 dBm。固定总发射功率下，带宽减少会同时降低积分噪声并提高SNR，因此不能把速率简单按带宽比例缩放。

## 全部Training任务的无线统计

距离和链路统计以到达任务为单位。输入大小均值9.664 Mbit、中位数9.535 Mbit。FULL与EQUAL_SHARE的SNR是对应参考分配口径下的模型值。

| 指标 | FULL_BANDWIDTH | EQUAL_SHARE |
|---|---:|---:|
| SNR P1 / P50 / P99 | −13.315 / −6.952 / 41.506 dB | −6.827 / 2.557 / 51.298 dB |
| 速率 P1 / P10 / P50 / P90 / P99 | 0.657 / 0.999 / 2.651 / 21.346 / 137.880 Mbps | 0.572 / 0.802 / 1.742 / 6.269 / 18.161 Mbps |
| 上传时间 P50 / P90 / P99 / 最大 | 2.852 / 9.870 / 18.049 / 37.605 s | 4.511 / 12.548 / 21.489 / 41.764 s |
| 车辆发送能耗 P50 / 均值 / P99 | 0.569 / 0.826 / 3.601 J | 0.900 / 1.181 / 4.288 J |
| 全池接入时间 | 5,218,908.446 s | 7,457,245.588 s |
| 全池车辆发送能耗 | 1,041,309.135 J | 1,487,916.110 J |
| 任务加权等效速率 `ΣD/Σ(D/R)` | 2.333473 Mbps | 1.633067 Mbps |

EQUAL_SHARE比FULL的任务加权等效速率低30.02%，总接入时间和车辆发送能耗高42.89%。这是预先定义的独占带宽与同RSU同Slot等分两种参考口径之差。实际并发分配中，任务加权`n`的P10/P50/P90为4/8/12，最大18；B3-10E逐Slot等分口径的等效速率不是对速率取算术平均。

### 弱链路与远距离贡献

| 任务子集 | FULL接入时间占比 | EQUAL_SHARE接入时间占比 | 子集距离中位数（FULL / EQUAL） |
|---|---:|---:|---:|
| 最低速率1% | 4.04% | 3.20% | 1,116.9 / 1,116.4 m |
| 最低速率5% | 16.07% | 13.06% | 1,033.0 / 1,033.0 m |
| 最低速率10% | 28.10% | 23.70% | 953.7 / 952.7 m |

最低1%速率尾部并没有独自支配全池等效速率；较重要的是较宽的低速率尾部，最低10%贡献约23.7%的EQUAL_SHARE接入时间。作为纯诊断，排除最低10%速率任务后，其余任务的`ΣD/Σ(D/R)`为1.926 Mbps，而原始全池值为1.633 Mbps；这个排除样本的数值不是新参数，也不替代原拟合。

距离分组说明主要负担来自**广泛分布的中远距离和路径损耗**，而不是少数异常极端链路：500–1000 m占56.69%任务并贡献75.76%的等份接入时间；>1000 m占3.54%任务、贡献9.65%。同RSU带宽共享进一步使全池接入时间相对独占参考增加42.89%，是另一个明确因素。

### 可追溯的典型弱链路

| 样本 | Episode / Slot / 车辆 / RSU | 距离 | `n` / 份额 | 输入量 | FULL SNR / 速率 | EQUAL SNR / 速率 | EQUAL上传时间 |
|---|---|---:|---:|---:|---:|---:|---:|
| 最低等份速率 | 20292 / 171 / TV 4 / RSU_0 | 1,234.32 m | 15 / 0.0667 | 7.662 Mbit | −14.87 dB / 0.462 Mbps | −3.11 dB / 0.382 Mbps | 20.03 s |
| 等份速率约P1 | 20041 / 220 / TV 49 / RSU_1 | 1,099.34 m | 6 / 0.1667 | 4.136 Mbit | −13.49 dB / 0.632 Mbps | −5.71 dB / 0.572 Mbps | 7.23 s |
| 等份速率中位样本 | 20195 / 178 / TV 46 / RSU_0 | 633.92 m | 8 / 0.1250 | 13.695 Mbit | −6.92 dB / 2.671 Mbps | 2.11 dB / 1.742 Mbps | 7.86 s |

这些是公式下的理论吞吐与`D/R`，不是实测吞吐或成功传输概率。等份速率最低样本并发数较高，说明距离损耗与带宽共享会共同降低速率；但对1000 m附近任务，即使独占10 MHz，公式也只有约0.8 Mbps量级，距离损耗本身已很显著。

## B3-10E拟合值的含义与边界

B3-10E先以全池加权车辆发送能耗拟合：`R_eq = Σ(Ptx·D)/Σ(Ptx·D/R_access)`；由于`Ptx`相同，这等于`ΣD/Σ(D/R_access)`，不是速率算术平均。`psi_eq`再以`Σ(T_access+T_backhaul)/Σ(D/R_eq)`拟合，回传项采用“每RSU共享200 Mbps”的离线假设。因此EQUAL_SHARE的`R_eq=1.6330667 Mbps`和`psi=1.0681932`是特定Training池、关联和共享假设下的聚合拟合对。

B3-10H把这一对参数作为每个Cloud任务固定的等效链路合同；没有把车辆到某RSU的实际距离带入Cloud候选，也没有实现真实`TV→RSU→Cloud`接入和回传资源。于是实际RSU参考链路很弱的任务，在现行抽象Cloud模式下仍然可以进入Cloud候选。这是已声明的**抽象模型边界**，不是地面速率函数计算错误。聚合参数能在全池参考口径上匹配加权总量，却不能代表近远车辆各自的逐任务无线速率、时延或可靠性。

该参数适合表述为Training池加权校准的等效链路候选；若论文要解释实际车联网Cloud接入、道路覆盖或逐任务可靠性，就不能把它说成已验证的端到端无线合同。此时应分开说明等效Cloud链路的校准样本、回传假设及不包含的覆盖约束。

## 数学速率不等于物理可靠性

本次1,260,159个样本在模型内均得到有限、正的FULL和EQUAL_SHARE速率。这个事实只说明输入代入Shannon代理后可以计算，不说明实际系统能以该速率可靠交付任务。当前地面链路缺少阴影衰落、快衰落、遮挡、干扰、制式/MCS映射、BLER/PER、outage概率及按业务定义的最低可接受速率或时延门槛。负SNR在该代理公式中仍会产生正速率，代码也没有把它判为不可行。

3GPP官方记录将TR 37.885列为LTE/NR新V2X用例的评估方法研究；本环境只能访问官方条目元数据，未能访问完整报告正文，因此本报告不对其具体信道参数或门槛作条款级断言。[3GPP TR 37.885官方记录](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3209)

可访问的ETSI托管3GPP TS 22.186 V16.2.0按不同V2X应用列出各自的时延、可靠性、速率和范围目标。这些目标属于具体应用场景，不能无说明地移作一般计算卸载任务的硬SNR门槛。[3GPP TS 22.186 V16.2.0](https://www.etsi.org/deliver/etsi_ts/122100_122199/122186/16.02.00_60/ts_122186v160200p.pdf)

3GPP TR 38.901提供了按部署场景定义的三维信道模型，可作为后续补足路径损耗、阴影和衰落结构的参考；它是建模对照，不代表当前项目已经实现或验证了该模型，也不替代TR 37.885的V2X具体适用性判断。[3GPP TR 38.901 V16.1.0](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/16.01.00_60/tr_138901v160100p.pdf)

此外，当前系统使用1秒Slot及`independent_per_slot`语义，不维护跨Slot任务队列。EQUAL_SHARE参考中92.87%的任务上传时延至少1秒，FULL参考中73.45%至少1秒；这些仍是静态时延/效用数值，不能解释为任务在Slot截止时间内成功完成或跨Slot排队完成。

## 判断与建议

- **RSU部署：CONDITIONAL。** 两个RSU均位于任务车辆轨迹坐标包络内，且分别服务约65.5%和34.5%的任务；但最近RSU距离中位数636 m、P90为926 m，只有11.13%的任务在250 m以内。若研究目标要求较密集的低时延覆盖，当前部署不支持仅凭现有证据声称达到目标；需先定义道路级覆盖与服务要求。250 m本身不是硬门槛。
- **地面信道模型：CONDITIONAL。** `βg`与5.9 GHz自由空间1 m参考损耗数值自洽，噪声积分、固定总功率、速率及KKT动态带宽口径内部一致；`αg=2.75`是项目文档标注的校准基线，但没有项目场景的测量校准记录。二维、无衰落/遮挡/干扰模型只能支撑理想平均信道分析，不能支撑无线可靠性结论。
- **B3-10E等效速率代表性：CONDITIONAL。** EQUAL_SHARE 1.633 Mbps是对全池总上传能耗/时间的加权等效量，能复现该训练参考集合的聚合口径；远距离主体和共享负载都会影响它。它不代表典型单车链路，更不是逐位置Cloud无线速率。弱链路的最低1%并非唯一主因，最低10%贡献约23.7%的接入时间。
- **物理链路可靠性：未验证。** 没有可核实的MCS与BLER/PER/outage或应用级QoS约束，不能增加任意SNR门槛，也不能把正的Shannon速率当作覆盖保证。

建议保留B3-10H为独立的聚合等效链路候选，不改RSU部署、信道参数或Cloud参数。如果后续论文要作物理无线可靠性或覆盖结论，先在独立模型版本中选定适用道路/制式场景，补齐3D几何、经依据支持的衰落/遮挡与可靠性映射，并按真实训练轨迹重新评估RSU部署；最低门槛应由通信制式和具体任务QoS共同导出。若研究仍聚焦抽象MEC资源竞争，则明确写出Training池加权Cloud等效合同及其逐任务边界即可。

```ini
STAGE_B3_10I = COMPLETE
CONDA_ENV = adaptive_cache_mec (verified interpreter; conda activation unavailable)
TRAINING_EPISODES_USED = 20001–20300 (300 Training episodes)
TASKS_ANALYZED = 1,260,159
RSU_COUNT = 2
GROUND_CHANNEL_REFERENCE_GAIN = 1.6368e-5
GROUND_PATHLOSS_EXPONENT = 2.75
GROUND_BANDWIDTH_MHZ = 10
MEDIAN_RSU_DISTANCE_M = 635.822
P90_RSU_DISTANCE_M = 926.387
FRACTION_BEYOND_250M = 88.8726%
MEDIAN_SNR_DB = FULL -6.952; EQUAL_SHARE +2.557
MEDIAN_FULL_BANDWIDTH_RATE_MBPS = 2.651415
MEDIAN_EQUAL_SHARE_RATE_MBPS = 1.742056
LOW_RATE_TAIL_CONTRIBUTION = EQUAL_SHARE lowest 1%: 3.20% access time; lowest 10%: 23.70%
RSU_DEPLOYMENT_REASONABLE = CONDITIONAL
GROUND_CHANNEL_MODEL_REASONABLE = CONDITIONAL
EQUIVALENT_RATE_REPRESENTATIVE = CONDITIONAL
PHYSICAL_LINK_RELIABILITY_VALIDATED = NO
RECOMMENDED_MODEL_DIRECTION = Keep the isolated aggregate Cloud-link candidate; add a separately versioned, scenario-calibrated reliability model before making physical coverage claims.
PPO_TRAINING_EXECUTED = NO
MATCHING_KKT_EXECUTED = NO
SYSTEM_MODEL_CHANGED = NO
FORMAL_CONFIG_CHANGED = NO
```

## 保留的计算结果

机器可读汇总及少量可追溯样本见 [stageB3_10i_wireless_reliability_assessment.json](stageB3_10i_wireless_reliability_assessment.json)。该文件不保存逐任务大表。
