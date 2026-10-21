# B3-15H-S：随机学习收益与UAV贡献确认

## 执行范围与方法

- 环境解释器：C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe；未安装或升级依赖。
- Initial策略：Legacy Architecture 33 / Checkpoint 60与Sparse Architecture 34 / Checkpoint 61的各自Initial Checkpoint。Update 12数据直接复用B3-15H-R的既有随机评价，没有重跑Latest。
- 随机评价使用Training Episode 20231、20232及策略采样Seed 31501、31502；共8条Initial轨迹，每条300 Slot。相同Episode与策略Seed严格配对。
- 外生场景读取固定Training Pool文件，策略采样Seed按每条轨迹重新初始化。使用现有Evaluator的torch.inference_mode()和冻结Checkpoint合同，没有调用优化器；Return分解在每条轨迹上重新核验。
- 固定UAV组使用Legacy Update 12 Checkpoint和Training Episode 20231–20236，在进程内将UAV0、UAV1的确定性动作替换为各自唯一合法STAY候选；其对数概率同步设为被执行STAY候选的概率。Cache、Holding、环境执行、Matching/KKT和Reward仍通过现有Rollout/Evaluator路径运行。
- 固定组验证了3600/3600个STAY动作、六个Episode的位置偏移均为0、UAV Cost均为0。六条轨迹Cache更新率均为0，Holding均为1 Slot。所有Initial、Update 12和Fixed结果均满足Return = Task Utility − Cache Cost − UAV Cost。
- 样本单位限制：Stochastic有4条配对轨迹/架构，但只含2个外生Episode。两个策略采样Seed是同一环境内的重复，不能当作4个独立环境样本。结论仅描述这些样本。

## Stochastic Initial → Update 12 配对结果

每行差值均为Update 12减Initial。成本下降为负数；动作率差以百分点表示。

| 架构 | Episode | 策略Seed | ΔReturn | ΔTask Utility | ΔCache Cost | ΔUAV Cost | Δ实际缓存更新节点 | ΔALL_KEEP | ΔHolding | ΔMOVE UAV0/UAV1 | Δ位移 UAV0/UAV1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy | 20231 | 31501 | -2.988 | -9.833 | -6.954 | 0.109 | -58 | +0.80 pp | -0.337 | +12.33 pp / +3.33 pp | +12.33 pp / +3.33 pp |
| Legacy | 20231 | 31502 | 7.064 | -9.913 | -17.139 | 0.162 | -145 | +0.50 pp | -0.112 | +16.33 pp / +5.00 pp | +16.33 pp / +5.00 pp |
| Legacy | 20232 | 31501 | -6.139 | -25.959 | -20.049 | 0.229 | -175 | -0.04 pp | -0.115 | +17.67 pp / +12.33 pp | +17.67 pp / +12.33 pp |
| Legacy | 20232 | 31502 | 16.709 | 3.332 | -13.572 | 0.195 | -114 | +5.11 pp | -0.231 | +14.00 pp / +12.33 pp | +14.00 pp / +12.33 pp |
| Sparse | 20231 | 31501 | -14.876 | -20.085 | -5.131 | -0.078 | -48 | +21.30 pp | 0.269 | -7.00 pp / -3.00 pp | -7.00 pp / -3.00 pp |
| Sparse | 20231 | 31502 | 20.017 | 14.320 | -5.681 | -0.016 | -62 | +32.28 pp | 0.370 | -0.33 pp / -2.00 pp | -0.33 pp / -2.00 pp |
| Sparse | 20232 | 31501 | 0.051 | -3.645 | -3.636 | -0.061 | -41 | +19.40 pp | 0.191 | -1.67 pp / -6.33 pp | -1.67 pp / -6.33 pp |
| Sparse | 20232 | 31502 | 56.137 | 50.248 | -5.867 | -0.022 | -58 | +32.09 pp | 0.226 | +0.00 pp / -1.67 pp | +0.00 pp / -1.67 pp |

### Stochastic阶段均值和学习增益

阶段均值按4条轨迹描述性平均；配对学习增益先逐轨迹计算。环境层面只按两个Episode分组平均策略Seed。

| 架构 | 阶段 | Mean Return | Task Utility | Cache Cost | UAV Cost | 实际更新节点/Episode | ALL_KEEP动作率 | P(ALL_KEEP) | Holding | UAV0/UAV1 MOVE | UAV0/UAV1真实位移 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy | Initial | 3567.852 | 3642.377 | 73.852 | 0.673 | 695.00 | 0.61% | 0.19% | 1.843 | 48.08% / 46.92% | 48.08% / 46.92% |
| Legacy | Update 12 | 3571.514 | 3631.783 | 59.423 | 0.847 | 572.00 | 2.20% | 1.82% | 1.645 | 63.17% / 55.17% | 63.17% / 55.17% |
| Legacy | ΔUpdate12−Initial | **3.662** | -10.593 | -14.429 | 0.174 | -123.00 | +1.59 pp |  | -0.199 |  |  |
| Sparse | Initial | 3622.674 | 3637.014 | 13.649 | 0.690 | 134.00 | 15.02% | 12.46% | 1.904 | 50.00% / 46.83% | 50.00% / 46.83% |
| Sparse | Update 12 | 3638.006 | 3647.223 | 8.570 | 0.646 | 81.75 | 41.29% | 39.88% | 2.168 | 47.75% / 43.58% | 47.75% / 43.58% |
| Sparse | ΔUpdate12−Initial | **15.332** | 10.209 | -5.079 | -0.044 | -52.25 | +26.27 pp |  | 0.264 |  |  |

| 架构 | Episode 20231平均ΔReturn | Episode 20232平均ΔReturn | Return增加的逐Seed配对数 |
|---|---:|---:|---:|
| Legacy | 2.038 | 5.285 | 2/4 |
| Sparse | 2.570 | 28.094 | 3/4 |

- Legacy Stochastic：配对平均ΔReturn **3.662**，ΔTask Utility -10.593、ΔCache Cost -14.429、ΔUAV Cost 0.174。净增益主要由Cache Cost下降抵消Utility下降和UAV成本增加形成；仅2个外生Episode，逐策略Seed只有2/4条Return上升。
- Sparse Stochastic：配对平均ΔReturn **15.332**，ΔTask Utility 10.209、ΔCache Cost -5.079、ΔUAV Cost -0.044。两个Episode分组的平均Return差分别为2.570和28.094，逐策略Seed有3/4条上升；这提供了方向性样本内信号，不足以确认泛化。
- 同一批配对轨迹中，Initial Sparse−Legacy Return均差为54.822；Update12 Sparse−Legacy为66.492。差中之差为11.671。Sparse的Update12横向优势部分已存在于Initial策略，不能全归因于Sparse PPO学习。

## Legacy Deterministic Fixed-UAV对照

| Training Episode | Initial Return | Update12动态UAV Return | Update12固定STAY Return | 动态−固定 ΔReturn | ΔTask Utility | 固定STAY Cloud/Edge数 | 固定STAY节点分配：SV/RSU/UAV0/UAV1/Cloud |
|---:|---:|---:|---:|---:|---:|---:|---|
| 20231 | 3655.539 | 3759.126 | 3738.940 | 20.186 | 21.902 | 2209/2029 | 459/274/627/669/2209 |
| 20232 | 3563.366 | 3666.615 | 3646.190 | 20.424 | 22.140 | 2350/1855 | 313/469/303/770/2350 |
| 20233 | 3584.482 | 3681.097 | 3655.259 | 25.838 | 27.554 | 2358/1798 | 330/321/368/779/2358 |
| 20234 | 3721.364 | 3750.954 | 3747.059 | 3.895 | 5.611 | 2331/1877 | 609/206/675/387/2331 |
| 20235 | 3624.871 | 3664.029 | 3659.896 | 4.132 | 5.848 | 2554/1599 | 202/479/417/501/2554 |
| 20236 | 3660.735 | 3729.327 | 3720.767 | 8.559 | 10.275 | 2582/1674 | 305/293/308/768/2582 |

### 六个Episode均值分解

| 执行策略 | Return | Task Utility | Cache Cost | UAV Cost | Holding | Cloud任务/Episode | Edge任务/Episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy Initial deterministic | 3635.060 | 3635.683 | 0.623 | 0.000 | 1.417 | 2713.17 | 1489.50 |
| Legacy Update12动态UAV deterministic | 3708.524 | 3710.240 | 0.000 | 1.716 | 1.000 | 2334.67 | 1868.00 |
| Legacy Update12固定STAY | 3694.685 | 3694.685 | 0.000 | 0.000 | 1.000 | 2397.33 | 1805.33 |

- Update12动态UAV − Initial：ΔReturn 73.465 = 74.558 − (-0.623) − 1.716。
- Update12固定STAY − Initial：ΔReturn 59.626 = 59.002 − (-0.623) − 0.000。确定性策略改善仍在固定UAV时明显存在。
- Update12动态UAV − 固定STAY：ΔReturn **13.839** = 15.555 − 0.000 − 1.716。六个Training Episode中，动态UAV任务效用优势超过了移动成本，显示正的样本内净贡献。
- 固定STAY组每Episode平均节点任务数：Cloud 2397.33，SV 369.67，RSU 340.33，UAV 1095.33（UAV0 449.67，UAV1 645.67）。固定组任务分配计数按每Slot Matching结果聚合并与总任务数核对。
- 动态Update12结果的既有摘要只保留Cloud/Edge汇总，没有每个执行节点的持久计数；因此动态组仅与固定组比较Cloud/Edge，不补造UAV0/UAV1任务数。

## 确定性与随机执行差异

- Legacy Update12确定性下Cache 100% ALL_KEEP、Holding 1、UAV0/UAV1 MOVE均为100%；Stochastic下ALL_KEEP 2.20%，实际缓存更新 572.00节点/Episode，Holding 1.645，MOVE 63.17%/55.17%。
- Sparse Update12确定性下ALL_KEEP 100%、Holding 5、UAV基本STAY；Stochastic下ALL_KEEP 41.29%，更新 81.75节点/Episode，Holding 2.168，MOVE 47.75%/43.58%。
- Deterministic与Stochastic的策略分布和频率差异来自确定性概率选择与随机采样两种执行方式；不能将一方的动作频率或收益混入另一方做Checkpoint排名。

## 结论

| 字段 | 结论 |
|---|---|
| STOCHASTIC_LEGACY_LEARNING_GAIN | 3.662 Return/Episode；净增益主要来自Cache成本下降，Task Utility为负变化；证据有限。 |
| STOCHASTIC_SPARSE_LEARNING_GAIN | 15.332 Return/Episode；Utility与Cache Cost均有正向分解，方向性样本信号。 |
| SPARSE_STOCHASTIC_GAIN_SUPPORTED | PRELIMINARY_POSITIVE；两环境均值方向为正，但只有2个环境Episode，泛化仍INSUFFICIENT_EVIDENCE。 |
| LEGACY_DETERMINISTIC_UAV_CONTRIBUTION | 六个配对Training场景中，动态相对固定STAY的净Return +13.839/Episode（Task Utility +15.555，UAV Cost +1.716）；确定性训练收益大部分仍在固定UAV下保留。 |
| DETERMINISTIC_VS_STOCHASTIC_POLICY_DIFFERENCE | 动作率和成本结构明显不同；Deterministic主评价、Stochastic补充评价须分开。 |
| RECOMMENDED_EVALUATION_PROTOCOL | 沿用B3-14G：确定性30个Validation Episode Mean Return作为主Checkpoint选择；Stochastic独立报告，固定环境Seed和策略Seed映射并按环境Seed聚类；最终Test仅在模型选择后。 |
| READY_FOR_LONG_TRAINING | YES，按冻结评价协议可开展后续确认性长训练；不代表已证明架构优劣。本轮未启动训练。 |

### 证据边界

- **架构优势：** Sparse随机Update12平均优于Legacy，但Sparse Initial也有较高Return；当前差中之差仅11.671，两环境不足以定论。
- **PPO学习收益：** Legacy随机净增益小、依赖降低Cache成本；Sparse随机样本表现为正向净收益和Task Utility，但少量Episode不能判断稳定性。
- **Cache收益：** 两种策略Update12都降低随机Cache更新节点数和成本；Legacy随机Task Utility下降，Sparse整体Utility上升。确定性Legacy收益并非Cache成本下降造成，因为Initial与Update12的确定性Cache成本差仅-0.623。
- **UAV收益：** 固定STAY消除了移动成本；Legacy确定性Update12在静止UAV时仍比Initial Return高59.626，而自由UAV在这6个训练场景额外贡献净Return约13.839。Stochastic Legacy的UAV成本略升且Utility下降，不能将随机收益增量归于UAV。

本轮只新增了冻结评价结果目录及本报告。未启动训练，未运行Validation/Test，未修改PPO、Reward、源码、配置、Checkpoint或Training Pool。
