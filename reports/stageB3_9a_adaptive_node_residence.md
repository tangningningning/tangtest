# Stage B3-9A：节点级自适应缓存驻留门控原型

## 冻结策略与预注册范围

- Training Episode：20001, 20053, 20094, 20135, 20177, 20218, 20259, 20300。按排除指定Episode后的排序清单等间隔选择，固定清单在任何Return回放前写入JSON；8个manifest均为Training。
- 策略随机种子：930901, 930902；每个Episode/seed运行四种策略各60槽，前30槽为同轨迹截取。共64条分支，计划3840次Slot。
- 冻结策略：Architecture 33 / Checkpoint 60 / v4 multi-node；Checkpoint SHA-256 `2f79020abdbeacba8d527ab3a15b1022e7d591c7fba63e8169bbf8ccdef63710`。未训练，参数SHA-256前后相同。
- B3实验配置 `configs/proposed/system_config_multi_node_factorized.toml` SHA-256 `a86d87ca5d586e89e367aee393b7a1cd9b27813f41ac3c5a19005bfa40a9ea6c`；正式主线 `configs/proposed/system_config.toml` SHA-256 `f3131efb7832efa925c181d9abd07bec5f2bfe3e7507eca2cc2515b77067699b`。B3实验配置显式 `MULTI_NODE_FACTORIZED`，主线v3 schema默认单节点；本原型仅绑定B3独立配置，不覆盖主线。
- 环境：C:\software\anaconda3\python.exe，Python 3.12.7，NumPy 2.2.6，PyTorch 2.13.0+cpu；Conda激活失败，沿用既有Python及进程内纯PythonKKT fallback。正式gamma=1.0。
- 最终分析：64/64条有效分支、3840个Slot；Matching/KKT可行。初次GLOBAL_HOLDING_5实现错误地允许第5槽前的PPO终止采样，故丢弃并重跑该16条分支（额外960次Slot执行）；实际总执行尝试数为4800。最终有效轨迹rollout累计计时698.5s；加上作废分支后累计计时880.4s（不含环境载入和报告汇总）。PPO训练、正式配置和系统模型均未改变。

## 自适应门控定义

在每个原有Macro边界，节点年龄为 `当前Slot − 最近一次实际缓存行变化Slot`；初始Cache行把Slot 0作为驻留起点，年龄从0开始。只对真正发生变化的行重置年龄及预测锚点。

每个决策时刻使用合法可见的五步GRU预测场景：由现有预测`arrival/service_id`形成期望服务请求量，并按配置中的horizon discount汇总；预测车辆位置使用场景预测位置的均值。对每辆预测任务车和每个有效预测时刻，复用正式物理信道与uplink-rate函数计算9个Edge节点的全带宽参考速率，并以速率占比作为连续服务机会权重，不添加距离硬截断。按节点聚合并归一化服务分布，另加一个`no_forecast_mass`类别。该权重不含共享KKT资源分配或Cloud效用，因此仅是需求变化代理，不是预测的Task Utility。

比较当前节点分布与最近一次缓存行改变时保存的分布，使用 `V_m=0.5×L1` 总变差距离。当前阈值为9个节点有限V值的中位数；只在 `5≤年龄<10` 时，`V_m > 中位数` 才释放门控并让原Actor动作执行，等于阈值时保持KEEP。年龄<5一律KEEP；年龄≥10恢复原Actor正常选择。释放不强制UPDATE。每次更新只在该行实际变化后重置计时器和锚点。

零需求用独立无需求原子表示：两次均为零时V=0，零与非零之间V=1；全节点变化均为0时阈值为0且严格大于规则不释放。非有限V从Peer中位数排除并在5–9槽窗口fail-closed为KEEP；预测位置/物理速率本身非有限时停止该分支，不填造数据。

策略含义：BASELINE为原随机冻结PPO；ROW_MIN_10仅节点级强制最短10槽；GLOBAL_HOLDING_5仅安全地在推理窗口内固定Macro持续5槽；ADAPTIVE_NODE_RESIDENCE仅在原Macro边界做上述按节点门控。PPO输出先照常采样，再对被门控节点覆盖为合法候选0 KEEP；此类干预轨迹只用于冻结策略评估，不作为on-policy训练数据。

## 30槽与60槽相对BASELINE的配对结果

统计单位是8个Episode：先在同Episode、同seed配对，再平均两个seed得到Episode差值，最后跨8个Episode求均值/中位数；正收益数量也按Episode计算。成本节省为BASELINE成本减干预策略成本；ΔReturn满足 `ΔQ + Cache成本节省 + UAV成本节省`。

### 30槽

| 策略 | ΔReturn均值 | 中位数 | 正收益Episode | 最差Episode | ΔQ | Cache成本节省 | UAV成本节省 | Δ总时延(s) | Δ能耗(J) | ΔEdge任务 | ΔCloud任务 | Δ更新节点数 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ROW_MIN_10 | +3.041299 | +3.145315 | 8/8 | 20177 | -0.056176 | +3.097887 | -0.000412 | +0.270 | +0.3897 | -0.62 | +0.62 | -30.00 |
| GLOBAL_HOLDING_5 | +2.158670 | +2.097223 | 8/8 | 20300 | +0.133070 | +2.027697 | -0.002097 | -10.128 | +0.5003 | +1.38 | -1.38 | -19.19 |
| ADAPTIVE_NODE_RESIDENCE | +2.610930 | +2.589202 | 8/8 | 20177 | +0.094577 | +2.515596 | +0.000756 | -6.569 | +0.2612 | +0.75 | -0.75 | -24.56 |

### 60槽

| 策略 | ΔReturn均值 | 中位数 | 正收益Episode | 最差Episode | ΔQ | Cache成本节省 | UAV成本节省 | Δ总时延(s) | Δ能耗(J) | ΔEdge任务 | ΔCloud任务 | Δ更新节点数 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ROW_MIN_10 | +7.605949 | +7.554987 | 8/8 | 20177 | +0.137938 | +7.466368 | +0.001643 | -12.283 | +0.7862 | +2.81 | -2.81 | -71.00 |
| GLOBAL_HOLDING_5 | +6.173192 | +6.152144 | 8/8 | 20053 | +0.202887 | +5.976409 | -0.006105 | -14.755 | +0.6597 | +3.25 | -3.25 | -56.38 |
| ADAPTIVE_NODE_RESIDENCE | +6.404630 | +6.213241 | 8/8 | 20135 | +0.071947 | +6.327685 | +0.004998 | -5.802 | +0.3193 | +0.31 | -0.31 | -59.94 |

## 与两种参考控制的直接比较

| 比较 | ΔReturn均值 | 中位数 | 正收益Episode | 最差Episode | ΔQ | 对照方成本节省 |
|---|---|---|---|---|---|---|
| 30槽 ADAPTIVE_NODE_RESIDENCE − ROW_MIN_10 | -0.430369 | -0.320342 | 0/8 | 20135 | +0.150753 | -0.582291 |
| 30槽 ADAPTIVE_NODE_RESIDENCE − GLOBAL_HOLDING_5 | +0.452260 | +0.537756 | 7/8 | 20135 | -0.038492 | +0.487899 |
| 60槽 ADAPTIVE_NODE_RESIDENCE − ROW_MIN_10 | -1.201319 | -1.158251 | 0/8 | 20135 | -0.065991 | -1.138683 |
| 60槽 ADAPTIVE_NODE_RESIDENCE − GLOBAL_HOLDING_5 | +0.231438 | +0.224260 | 6/8 | 20135 | -0.130940 | +0.351275 |

## Episode级与两种固定参考的差异（60槽）

先在同Episode的两个策略种子内配对并取均值。ΔReturn与ΔQ均为自适应减参考策略；Cache节省为参考策略成本减自适应成本，正值表示自适应成本较低。

| Episode | 相对ROW_MIN_10 ΔReturn | ΔQ | Cache成本节省 | 相对GLOBAL_HOLDING_5 ΔReturn | ΔQ | Cache成本节省 |
|---|---:|---:|---:|---:|---:|---:|
| 20001 | -0.6289 | +0.0368 | -0.6785 | +1.0576 | +0.0787 | +0.9513 |
| 20053 | -1.2247 | -0.6360 | -0.6057 | +0.2296 | -0.7012 | +0.9174 |
| 20094 | -0.4606 | +0.2368 | -0.6994 | +0.6822 | +0.0501 | +0.6306 |
| 20135 | -2.0608 | -0.1706 | -1.8916 | -1.0017 | -0.4436 | -0.5734 |
| 20177 | -1.1099 | +0.2301 | -1.3442 | +0.2189 | -0.3294 | +0.5293 |
| 20218 | -1.1423 | +0.2761 | -1.4175 | +0.8558 | +0.7446 | +0.1120 |
| 20259 | -1.8091 | -0.4651 | -1.3392 | -0.4045 | -0.2930 | -0.1149 |
| 20300 | -1.1742 | -0.0360 | -1.1333 | +0.2136 | -0.1538 | +0.3578 |

- 相对ROW_MIN_10，自适应在8/8个Episode都较差：平均ΔReturn −1.201319、ΔQ −0.065991、Cache成本节省 −1.138683；实际更新节点数平均多11.0625次/轨迹。结果表明差距主要伴随更高缓存重配置成本，且任务效用也略低。
- 相对GLOBAL_HOLDING_5，自适应在20135与20259较差。两者的ΔQ分别为−0.443560、−0.293000，Cache成本节省分别为−0.573360、−0.114933，损失同时来自任务效用降低与缓存成本增加。其余6个Episode净收益较高。
- 跨8个Episode平均，自适应相对GLOBAL_HOLDING_5的ΔReturn为+0.231438，但ΔQ为−0.130940；+0.351275的Cache成本节省抵消了该效用差额。相对BASELINE则8/8个Episode为正，但这不足以证明它优于更强的固定10槽参考。
## Episode级差异

正值表示左侧节点驻留策略收益更高；下表先对每个Episode的两个seed求均值。

| Episode | ROW_MIN_10 ΔG | ΔQ | Cache节省 | GLOBAL_HOLDING_5 ΔG | ΔQ | Cache节省 | ADAPTIVE_NODE_RESIDENCE ΔG | ΔQ | Cache节省 |
|---|---|---|---|---|---|---|---|---|---|
| 20001 | +7.9464 | +0.4585 | +7.4973 | +6.2599 | +0.4167 | +5.8674 | +7.3175 | +0.4954 | +6.8188 |
| 20053 | +7.1041 | -0.3289 | +7.4435 | +5.6498 | -0.2637 | +5.9203 | +5.8794 | -0.9649 | +6.8377 |
| 20094 | +7.7194 | +0.1517 | +7.5672 | +6.5765 | +0.3383 | +6.2373 | +7.2587 | +0.3885 | +6.8678 |
| 20135 | +7.4515 | +0.3185 | +7.1244 | +6.3924 | +0.5915 | +5.8061 | +5.3907 | +0.1479 | +5.2328 |
| 20177 | +7.0202 | -0.5694 | +7.5805 | +5.6914 | -0.0099 | +5.7070 | +5.9104 | -0.3393 | +6.2364 |
| 20218 | +7.6584 | +0.2838 | +7.3671 | +5.6603 | -0.1848 | +5.8376 | +6.5161 | +0.5599 | +5.9496 |
| 20259 | +7.4490 | +0.4971 | +6.9447 | +6.0443 | +0.3250 | +5.7204 | +5.6398 | +0.0320 | +5.6055 |
| 20300 | +8.4985 | +0.2923 | +8.2063 | +7.1107 | +0.4100 | +6.7151 | +7.3243 | +0.2562 | +7.0729 |

## 节点类型与驻留行为

| 策略 | 节点类型 | 相对BASELINE的UPDATE数变化 | 成本变化（干预−BASELINE） |
|---|---|---|---|
| ROW_MIN_10 | SV | -40.688 | -3.659342 |
| ROW_MIN_10 | RSU | -16.000 | -1.723693 |
| ROW_MIN_10 | UAV | -14.312 | -2.083334 |
| GLOBAL_HOLDING_5 | SV | -31.562 | -2.777417 |
| GLOBAL_HOLDING_5 | RSU | -13.188 | -1.452708 |
| GLOBAL_HOLDING_5 | UAV | -11.625 | -1.746285 |
| ADAPTIVE_NODE_RESIDENCE | SV | -34.562 | -3.112683 |
| ADAPTIVE_NODE_RESIDENCE | RSU | -11.875 | -1.323628 |
| ADAPTIVE_NODE_RESIDENCE | UAV | -13.500 | -1.891373 |

| 策略 | 实际缓存行驻留均值(slot) | 最短 | 最长 | 观测驻留段数 |
|---|---|---|---|---|
| BASELINE | 4.912 | 1 | 25 | 1759 |
| ROW_MIN_10 | 12.290 | 1 | 25 | 703 |
| GLOBAL_HOLDING_5 | 10.082 | 5 | 45 | 857 |
| ADAPTIVE_NODE_RESIDENCE | 9.818 | 1 | 26 | 880 |

## 自适应门控是否真正随状态变化

- 宏边界×节点门控记录 3384 条；强制KEEP判断 1881 次，其中Actor原本采样UPDATE而被覆盖 956 次。
- 年龄5–9槽时决策 1330 次：需求变化严格高于当时中位数并释放 611 次；否则保持KEEP 719 次。释放后Actor实际采样UPDATE 317 次，最终实际发生缓存行变化 317 次。
- 不同有限V值数量 3203；不同中位阈值数量 360；包含至少一次早释与至少一次门控KEEP：`True`。
- 每个决策的节点ID、年龄、当前V、阈值、强制KEEP、Actor候选索引、执行候选索引、当前/锚点服务分布均逐条保存在JSON。预测任务流与Slot内真实任务相分离；Slot真实任务只用于正式Matching/KKT后的结果评价。

## Episode退化原因与结论

60槽相对BASELINE没有出现Return为负的Episode；仍需查看上表的ΔQ与成本分解，避免将成本节省单独视作效用改善。

冻结策略上出现的收益只支持本轮Training Episode内的推理控制比较，不证明重新训练后的收益。若自适应高于GLOBAL_HOLDING_5但未高于ROW_MIN_10，证据仅支持节点粒度可能有用，当前需求变化释放规则尚未优于固定节点参考。若它高于ROW_MIN_10，仍需在新的预注册Training场景复核，并评估300槽和再训练后的表现。若门控在年龄5–9的行为缺少有效早释/强制差异，应视为退化成近似固定驻留。

建议仅当自适应在两个参考对比上都呈现一致收益时，才进入节点级可学习驻留策略设计；当前被覆盖的推理轨迹不是on-policy样本，不能直接训练PPO。正式集成需在动作采样/log-prob之前建立一致Mask或明确控制变量，再单独验证Critic与联合PPO概率。

```ini
STAGE_B3_9A = COMPLETE
ADAPTIVE_GATE_IMPLEMENTED = YES
STATE_DEPENDENT_RESIDENCE_CONFIRMED = YES
ADAPTIVE_VS_BASELINE = 60-slot mean paired ΔReturn +6.404630; median +6.213241; positive Episodes 8/8
ADAPTIVE_VS_ROW_MIN_10 = 60-slot mean paired ΔReturn -1.201319; median -1.158251; positive Episodes 0/8
ADAPTIVE_VS_GLOBAL_HOLDING_5 = 60-slot mean paired ΔReturn +0.231438; median +0.224260; positive Episodes 6/8
TASK_UTILITY_CHANGE = {'adaptive_vs_baseline_mean_delta_q_60': 0.07194691178767698, 'adaptive_vs_row_min_10_mean_delta_q_60': -0.06599089563371763, 'adaptive_vs_global_holding_5_mean_delta_q_60': -0.13094036651027352}
CACHE_COST_SAVING = {'adaptive_vs_baseline_mean_60': 6.327684695967696, 'adaptive_vs_row_min_10_mean_60': -1.1386831467525442, 'adaptive_vs_global_holding_5_mean_60': 0.3512752426755347}
FORCED_KEEP_COUNT = 1881 (Actor UPDATE override=956)
EARLY_RELEASE_COUNT = 611 (Actor UPDATE=317; actual row changes=317)
ADAPTIVE_NODE_CONTROL_PROMISING = NO
LEARNABLE_RESIDENCE_POLICY_RECOMMENDED = NO
RECOMMENDED_NEXT_STEP = 当前需求变化启发式未显示优于固定ROW_MIN_10；暂不并入PPO。保留其作为冻结策略诊断原型，若继续研究，先分析可观测预测变化与候选收益标签的对齐，不扫描阈值。
PPO_TRAINING_EXECUTED = NO
SYSTEM_MODEL_CHANGED = NO
PPO_ALGORITHM_CHANGED = NO
FORMAL_CONFIG_CHANGED = NO
```

逐Slot任务指标、每个Macro/节点的门控判断、实际缓存行变更事件、30/60槽统计和逐Episode配对差值见同目录JSON。
