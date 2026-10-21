# 第二阶段任务需求与流行度环境检查

## 范围与方法

只读取正式配置、当前生成/预测/候选评分/Matching相关函数，以及Training Episode 20001、20150、20300。没有运行PPO、Matching/KKT优化，没有修改源码、配置或数据池。统计使用adaptive_cache_mec解释器 C:/software/Anaconda_envs/envs/adaptive_cache_mec/python.exe（sys.prefix已确认）。

三个Episode manifest均为 formal_task_transform_v1、task_data_model=500_1000_KB。它们是从旧任务实现确定性转换得到的正式任务文件，不是本轮由当前生成器重跑得到：manifest记录保留arrival与service身份，按服务新旧均值比例缩放任务输入和CPU cycles，并保留±20%输入扰动、±10%负载扰动。以下将当前生成器代码作为模型合同，并用保存的最终数据逐行核对；不声称重建了旧随机流。

## A. 已确认的实现与时间合同

- 当前配置为10种服务、20辆TV、每TV每Slot到达概率0.7；Zipf指数0.8，每20 Slot一个epoch，每个边界做一次相邻rank局部移动。任务输入按bit保存，目录KB转bit使用8000 bit/KB；任务计算量按cycles保存。
- task_scenario生成每个TV×Slot一行。arrival=0时service_id为空、输入和CPU cycles为0；arrival=1时按当前Slot的rho_star做Categorical服务抽样，再按目录服务均值和计算密度采样任务输入及计算负载。
- 三集每个tasks.csv都有6000行，即20个角色TV×300 Slot；每个slot覆盖且只覆盖20个当前TV，没有SV任务行。time_s=100+slot，episode_time_s=slot。所有arrival=0行的服务ID/输入/cycles为空或零；所有arrival=1行的service_id均为0–9，输入和cycles为正，且相对当前新服务目录的输入因子均在0.8–1.2、有效cycles/bit相对服务密度均在0.9–1.1。
- popularity.csv每集覆盖300×10个slot-service组合；rho_star逐Slot归一化，任务实际service计数与popularity.csv的request_count逐Slot逐服务完全相等。按每Slot实际到达数加权rho_star得到的预期服务请求总数与实际计数的跨服务相关系数为0.9982、0.9998、0.9991。
- prediction文件及场景包时间轴一致：每集300个origin，H=5（horizon 0–4）。CSV中observation_cutoff_slot=origin_slot-1、target_slot=origin_slot+horizon、origin_time=100+origin_slot、target_time=100+target_slot；三集全部预测行的时间、目标rho_star和abs_error对齐检查无差异。horizon=0预测当前Slot的rho_star时仍只使用前一Slot及更早观测。
- 预测场景包保存rho_hat并按rho_hat另行抽样arrival/service；metadata明确标注tasks sampled from rho_hat, not rho_star，且future_ground_truth_saved=false。CSV里的rho_star是目标真值及误差诊断字段。当前生成链没有发现实际任务与预测任务错位或用未来真实任务生成预测场景的问题。

主要代码位置：src/scenario/task_scenario.py:124, 390-539, 568-659, 661-680, 843-1001；src/scenario/popularity_prediction.py:232-260, 390-447, 520-535；src/scenario/scenario_generator.py:593-659, 1040-1087, 1170-1200；配置见configs/proposed/system_config.toml:107-126。

## B. 实际任务分布

arrival按每个TV-slot Bernoulli(0.7)生成时，每Slot到达任务数的理论均值为14、标准差约2.05。观察值接近该分布：

| Episode | 有效任务数 | 到达率 | 每Slot到达数 mean/std/min/P50/P90/max | Slot 0–99 / 100–199 / 200–299任务数 |
|---|---:|---:|---:|---:|
| 20001 | 4190 | 0.6983 | 13.97 / 2.05 / 7 / 14 / 17 / 19 | 1429 / 1380 / 1381 |
| 20150 | 4136 | 0.6893 | 13.79 / 2.04 / 9 / 14 / 17 / 19 | 1370 / 1368 / 1398 |
| 20300 | 4200 | 0.7000 | 14.00 / 2.11 / 8 / 14 / 17 / 19 | 1397 / 1383 / 1420 |

Pooled 12,526个任务的输入大小P50/P90/P95为716.4/943.4/1009.7 KB；CPU cycles为3.497/5.340/6.529 Gcycle。P95略超过目录均值上限1000 KB是预期的，因为每任务还乘以±20%输入扰动，合同允许约400–1200 KB。三集P50/P90/P95如下：

| Episode | 输入大小 (KB) | CPU cycles (Gcycle) | 有效cycles/bit |
|---|---:|---:|---:|
| 20001 | 666.9 / 898.6 / 968.9 | 3.318 / 5.184 / 6.477 | 673 / 894 / 951 |
| 20150 | 763.0 / 991.5 / 1043.2 | 3.744 / 5.189 / 6.407 | 642 / 891 / 937 |
| 20300 | 721.3 / 918.5 / 972.8 | 3.441 / 5.550 / 6.645 | 621 / 917 / 982 |

每集的服务请求总数按service_id 0–9为：
- 20001：853, 357, 305, 176, 418, 218, 785, 256, 608, 214
- 20150：264, 785, 486, 1018, 387, 225, 325, 264, 190, 192
- 20300：362, 690, 231, 277, 225, 189, 1102, 187, 565, 372

各集总到达量相近，服务请求类型随Episode的流行度实现而明显不同；输入和cycles分布落在目录及扰动合同内。没有用任何Episode Return作为环境质量判断标准。

## C. 流行度变化与缓存自适应机会

每集有15个20-Slot概率epoch、14个边界。每个epoch内rho_star不变，所有14个边界的完整服务排名都发生了变化。跨42个Episode边界的rho_star总变差距离（TVD=0.5×L1）中位数为0.0151，均值0.0302，P90及最大为0.1194；因此变化是有时间相关的局部重排，大部分边界的概率质量变化较小，少数边界变化更明显。

| Episode | Top-3顺序变化次数 | Top-3成员集合变化次数 | 按20-Slot真实请求计数重排的Top-3集合变化 |
|---|---:|---:|---:|
| 20001 | 7/14 | 3/14 | 7/14 |
| 20150 | 6/14 | 1/14 | 3/14 |
| 20300 | 3/14 | 1/14 | 5/14 |

三集rho_star Top-3成员集合共在42个边界中改变5次；完整排名每次都变，不等于热门服务集合每次都变。基于实际请求计数的Top-3集合改变15次，较多变化来自20-Slot窗口内的抽样波动，不能直接视为真实Zipf热点都改变。尽管热点集合变化不频繁，每集仍至少有一次Top-3成员替换且各epoch概率向量有变化，因此现有需求为缓存服务排序和更新提供了非零的自适应机会；它不证明UPDATE有正净收益，也不能据此认定现有变化幅度足以产生可观的策略收益。

## D. SV候选需求的100米代理尺度

当前正式SV候选需求评分把节点到预测TV位置的距离换算为权重：距离小于100米时为1−d/100，距离大于或等于100米时为0。该距离被代码明确限定为proposal signal。实际Matching的Edge可行性检查计算当前路径的信道增益与uplink rate，要求rate有限且大于0，没有相同的100米硬截断；因此该值不是实际Matching的SV服务半径。

以真实轨迹位置及实际任务作离线几何核对（非候选预测重放、非Matching）：

| Episode | TV-Slot中距所有5辆SV均≥100m | 这些位置产生的任务数/有效任务 | 对任一指定SV，距其≥100m的实际任务比例范围 |
|---|---:|---:|---:|
| 20001 | 3561/6000 (59.4%) | 2488/4190 (59.4%) | 84.7%–96.9% |
| 20150 | 3865/6000 (64.4%) | 2639/4136 (63.8%) | 83.8%–97.6% |
| 20300 | 4332/6000 (72.2%) | 3032/4200 (72.2%) | 87.0%–95.9% |

三集合计，约65.3%的TV-Slot位置和65.1%的实际任务位置距所有SV至少100米。按真实任务与真实位置套用SV proposal权重，每个指定SV会对约83.8%–97.6%的实现任务给出零距离权重。这是显著的候选需求代理低估机制：当前通道模型不因100米而硬性判定链路不可用，但相同任务在SV候选需求通道中被置零。有限距离下当前信道模型可计算正rate，故模型内存在经SV服务的可能；实际可靠链路是否可用仍取决于未在本次定义的最低速率、SINR/outage门限及缓存/资源状态，不能据距离统计断言所有远端任务都应由SV服务。

该代理权重进入node_specific_predicted_demand和node_local_predicted_service_workload的请求、输入量、CPU-work特征；它还影响prediction/coverage/intermediate锚点配置，之后进入有容量限制的Cache候选保留流程。因此它可能改变SV节点预测需求特征、锚点目标及64个候选内的覆盖。它不会删除物理上可行的Matching Edge节点，也没有被确认成候选全集的硬距离过滤。未运行候选全集召回或Matching/KKT，无法量化最终保留候选受影响多少。

实现位置：src/decision/cache_configuration.py:546-565, 724-750, 753-877, 1025-1157；SV距离尺度配置为configs/proposed/system_config.toml:141-161；实际物理可行性见src/decision/task_allocation.py:231-314, 1858-1907。

## E. 结论与限制

已确认的实现错误：未发现任务行、arrival零值、service_id、workload公式、流行度请求计数或预测时间轴错误。

合理且需声明的假设：20辆固定TV每Slot至多一个任务；每个TV-Slot按独立Bernoulli生成到达；真实服务请求由rho_star采样；流行度每20 Slot做一次小幅重排；预测GRU只使用截至origin-1的观测。

有数据支持的潜在算法设计问题：SV cache proposal的100米软权重与实际Matching可接受的正速率路径范围不一致，导致大量已实现远端服务请求在SV候选需求中为零权重。其对保留候选召回和最终收益的影响需要后续窄范围候选检查，本轮不运行。

暂时无法判断：100米外低速率链路在拟研究无线场景下是否足够可靠；候选裁剪实际漏掉多少高收益配置；需求变化带来的策略UPDATE是否有正净效用。它们需要无线门限/候选与收益分析，不应由任务到达数量或Return替代。

结论：
    TASK_GENERATION_CORRECT = YES
    TASK_TIME_ALIGNMENT = YES
    ARRIVAL_DISTRIBUTION_REASONABLE = YES
    WORKLOAD_DISTRIBUTION_REASONABLE = YES
    POPULARITY_VARIATION_SUFFICIENT = YES

    SV_PROPOSAL_RANGE_MISMATCH = YES
    SV_DEMAND_UNDERESTIMATION = YES

    TASK_DATA_REGENERATION_REQUIRED = NO

    SOURCE_CHANGED = NO
    CONFIG_CHANGED = NO
    FORMAL_DATA_CHANGED = NO
    PPO_TRAINING_EXECUTED = NO

