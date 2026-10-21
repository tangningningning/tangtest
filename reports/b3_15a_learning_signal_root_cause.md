# B3-15A：PPO学习信号与Cache/UAV动作收益根因定位

## 结论摘要

- 未发现可确认的PPO、Matching/KKT或Reward实现错误。
- 在1e-3 Update 75策略的12个Training宏决策边界、36个节点状态中，69个合法Cache UPDATE候选分支有10个在实际Holding窗口内得到正的净收益，59个为负。UPDATE的平均Task Utility增量为`+0.0157`，平均Cache Cost增量为`+0.1282`，平均净Return增量为`-0.1125`。
- Cache策略在这组状态中通常偏好KEEP；其动作概率排序与实测收益方向在69个“UPDATE对KEEP”比较中有59个一致。它也低估了少量正收益UPDATE：9个节点状态发现至少一个有益UPDATE，这9个状态的`P(KEEP)`均大于`P(UPDATE)`。
- 6个同Slot UAV对照全部显示MOVE的净收益低于STAY，差值为`-0.00187`至`-0.00312`。这只说明所采样状态不支持MOVE有净收益，不能外推到所有Slot。
- Macro实现确实把每帧一个SMDP GAE Advantage用于该帧的9个Cache头和Termination联合动作。它缺少节点级信用归因，是可能的次级贡献；本轮数据不能证明它是训练表现的唯一或首要原因。

## 证据范围与执行方式

四个Fresh Run均读取现有300条Episode与75条Update指标。1e-3组`latest_checkpoint.pt`经严格加载确认Architecture 33、Checkpoint 60、Macro Actor LR `1e-3`、完成Episode 300/Update 75；当前配置原始及语义哈希、Training Pool Manifest哈希均与Run合同一致。Checkpoint读取前后SHA-256相同。

Cache/UAV反事实只使用Training Pool的Episode 20001、20101、20201。加载冻结策略后设为eval模式，在`torch.no_grad()`下各完成一次300-Slot随机策略采集；动作采样Seed映射为`20001→27005`、`20101→27006`、`20201→27007`。没有调用优化器、写入Run或修改Checkpoint。

在每个Episode中，对Slot 40、120、200、280选择最近的真实Macro决策边界：

| Training Seed | Slot 40目标 | Slot 120目标 | Slot 200目标 | Slot 280目标 |
| --- | ---: | ---: | ---: | ---: |
| 20001 | 41（3 Slot） | 119（5 Slot） | 200（1 Slot） | 280（1 Slot） |
| 20101 | 40（5 Slot） | 120（1 Slot） | 198（4 Slot） | 279（3 Slot） |
| 20201 | 38（4 Slot） | 120（1 Slot） | 201（1 Slot） | 278（5 Slot） |

每个边界选取一个SV、RSU、UAV Cache节点，分别执行KEEP、策略概率最高的合法UPDATE，以及一个实际采样或按固定独立Seed从策略条件UPDATE分布抽取的合法UPDATE（与前者相同时去重）。候选掩码和特征与原Rollout保存的Macro Transition一致。其他Cache节点与UAV路径固定为原策略动作；未来任务和车辆轨迹只用于离线执行分支，不参与候选生成。

共执行36组节点状态、105个Cache配置分支（36 KEEP、36 Top UPDATE、33个去重后的Sampled UPDATE），覆盖298个Slot Matching/KKT执行；另做6组双机联合可行的UAV STAY/MOVE比较，共12个单Slot分支。每个UPDATE配置均确实改变了所选节点缓存内容。全部分支资源可行，逐Slot检查`Reward = Task Utility − Cache Cost − UAV Cost`，最大分解误差为0。系统执行使用`execute_configured_slot()`正式路径，没有额外运行PPO或读取Validation/Test。

执行偏差：在诊断脚本整理期间，Episode 20001因一次返回对象字段名错误、一次UAV联合位置筛选遗漏及一次基线重放核对，额外进行过3次无持久输出的冻结采集/排查。它们没有执行PPO更新、没有写Run或Checkpoint，也不计入最终样本；最终统计只使用上列每个Seed各一份的成功采集。该额外回放超出了原计划的最小执行量，特此记录。

## 四组训练日志中的策略变化

下表比较各Run前20个与末20个Training Episode的记录均值。两段Episode Seed不同，因此这里只描述训练期间变化，不作为配对性能或泛化结论。`P(KEEP)`是9个节点头的平均校正KEEP概率；“有Cache变化”是一个Macro帧内至少一个节点实际变化的比例。

| Macro Actor LR | Return 前20→末20 | Task Utility 前20→末20 | Cache Cost 前20→末20 | UAV Cost 前20→末20 | P(KEEP) | 有Cache变化率 | Holding均值 / Macro帧数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-4 | 3555.69→3598.49 | 3622.05→3626.54 | 65.61→27.12 | 0.747→0.931 | 0.501→0.539 | 99.94%→99.34% | 2.173→4.955 / 139.3→60.55 |
| 5e-4 | 3549.15→3557.50 | 3623.05→3627.53 | 73.16→69.38 | 0.744→0.642 | 0.511→0.599 | 99.71%→98.85% | 1.893→1.654 / 158.9→181.95 |
| 1e-3 | 3552.48→3596.35 | 3624.34→3631.79 | 71.15→34.82 | 0.705→0.622 | 0.512→0.720 | 99.81%→95.05% | 1.958→2.154 / 153.65→139.6 |
| 5e-3 | 3560.16→3566.50 | 3626.92→3626.81 | 66.04→59.22 | 0.727→1.084 | 0.520→0.584 | 99.68%→99.16% | 2.087→1.953 / 144.35→154.1 |

UAV动作也发生了变化。前20→末20 Episode的UAV0/UAV1 MOVE概率与实际MOVE率分别为：1e-4 `0.537→0.757 / 0.534→0.760`、`0.525→0.542 / 0.519→0.544`；5e-4 `0.519→0.431 / 0.529→0.437`、`0.517→0.466 / 0.517→0.471`；1e-3 `0.491→0.592 / 0.483→0.589`、`0.511→0.285 / 0.512→0.285`；5e-3 `0.514→0.890 / 0.515→0.900`、`0.519→0.604 / 0.513→0.596`。

前20与末20的Return均值在四组都更高，但它们不是相同场景配对，不能据此证明持续改善或反驳用户观察到的收益问题。尤其`1e-4/1e-3`的Return增幅与Cache Cost降低同时出现，Task Utility增幅较小；这只显示训练轨迹相关性。

75组更新指标中，四组Macro Actor每组均执行4步、Micro Actor均执行40步，未记录Macro Early Stop。平均Macro joint KL / joint Clip Fraction分别为：1e-4 `0.00054 / 0.0030`，5e-4 `0.00352 / 0.0358`，1e-3 `0.00716 / 0.0903`，5e-3 `0.01492 / 0.1812`。高学习率下联合裁剪更常触发，但没有直接证据表明它压制了特定有益动作。四组Macro Critic的梯度裁剪比例均为1.0，裁剪前平均范数约109.36–219.16、裁剪后约0.5；配置上`max_grad_norm=0.5`。Macro Critic平均Explained Variance为0.892–0.902。所有Episode、Update和Critic指标未发现NaN/Inf。以上均未显示确定性实现错误；梯度裁剪持续发生本身也不等于错误。

## Cache反事实结果

69个UPDATE候选相对同节点KEEP的差值：

| 指标 | 实测结果 |
| --- | ---: |
| `ΔReturn > 0` | 10/69（14.5%） |
| `ΔReturn < 0` | 59/69（85.5%） |
| 平均 `ΔTask Utility` | +0.01568 |
| 平均 `ΔCache Cost` | +0.12822 |
| 平均 `ΔUAV Cost` | 0（固定同一UAV路径） |
| 平均 / 中位 `ΔReturn` | -0.11253 / -0.11099 |
| `ΔReturn`范围 | -1.07705至+0.99860 |

25个UPDATE分支提高Task Utility，15个分支Task Utility不变，29个降低Task Utility；全部69个分支增加Cache更新成本。只有10个分支的Utility收益足以抵消成本。这里的候选并非普遍无效，但在这些真实Holding窗口内，大多数净收益较KEEP低。

策略与实测收益排序在69个UPDATE对KEEP比较中有59个同向、10个反向。36个状态中，33个`P(KEEP)>P(UPDATE组)`，平均组概率为`0.687`对`0.313`。9个节点状态至少存在一个净收益为正的UPDATE；这9个状态的`P(KEEP)`都高于`P(UPDATE组)`。正收益UPDATE共5个Top候选、5个Sampled候选；也就是说，策略总体排序多数时候与实测收益一致，但在少数有益更新上偏保守。

两个具体正收益例子：

- Seed 20101、决策Slot 198、UAV Cache节点：Top UPDATE概率`0.0198`，KEEP组概率`0.8018`，UPDATE组概率`0.1982`。在4 Slot窗口内，`ΔTask Utility=+1.24868`、`ΔCache Cost=+0.25008`、`ΔUAV Cost=0`，所以`ΔReturn=+0.99860`。
- Seed 20001、决策Slot 119、RSU_0：一个合法UPDATE候选概率`0.00952`，KEEP组概率`0.7362`、UPDATE组概率`0.2638`。在5 Slot窗口内，`ΔTask Utility=+0.83208`、`ΔCache Cost=+0.16971`，所以`ΔReturn=+0.66237`。

相反，Seed 20101、Slot 40的UAV Cache Top UPDATE在3 Slot窗口内`ΔTask Utility=-0.2856`、`ΔCache Cost=+0.2501`，`ΔReturn=-0.5357`。这说明不能仅按候选预测分数推断实际价值。

每个节点头的KEEP概率不能直接当作整个Macro帧的无变化率。九个独立因子化头全部选KEEP，才会让整帧缓存不变。以1e-3末20 Episode的平均`P(KEEP)=0.720`作均匀近似，九头全KEEP概率约为`0.720^9=5.2%`，与实测无变化率`4.95%`接近。由于这里使用的是平均概率，这只是机制解释近似；逐状态概率乘积未在Episode日志中保存。它解释了为什么每节点KEEP概率增加时，整帧仍常有至少一个节点UPDATE。

## UAV STAY/MOVE结果

每个Episode选择最接近Slot 120的实际宏边界，并对UAV0、UAV1分别固定另一架UAV的位置。MOVE候选先通过正式双机位置可行性校验，再执行正式Matching/KKT。六组MOVE相对STAY的`ΔTask Utility`平均仅`+0.0000835`，而`ΔUAV Cost`平均为`+0.00253`，因此六组`ΔReturn`均为负，均值`-0.002446`、范围`-0.003116`至`-0.001870`，全部资源可行。

Seed 20001 Slot 119两架UAV实际都选MOVE；其对照收益分别为`-0.002647`和`-0.002608`。Seed 20101与20201的四个对照Slot实际均选STAY。六组中没有观测到MOVE的净收益优势。样本仅覆盖三个Episode各一个Slot，不能据此判定所有MOVE都无价值。

## 共享Advantage与裁剪机制

当前Macro更新对每个Episode按实际Holding Duration计算SMDP GAE（TD折扣使用`gamma^duration`，GAE trace使用`(gamma*lambda)^duration`），再把更新组内各帧的Advantage拼接并归一化。每个Macro帧只有一个标量Advantage；它作用于该帧9个Cache节点动作的联合Cache log-prob，以及该帧实际采样的Termination log-prob。PPO使用联合新旧log-prob差构造一个joint ratio，并对该joint ratio进行clip。

相关实现位于：

- `src/ppo_simple/rollout.py:2077`：duration-aware Macro SMDP GAE。
- `src/ppo_simple/trainer.py:1587`、`src/ppo_simple/trainer.py:1633`：逐Episode GAE及更新组共享Advantage拼接/归一化。
- `src/ppo_simple/policy.py:510`：多节点因子化Cache动作采样。
- `src/ppo_simple/trainer.py:1990`：联合Macro log-prob ratio及裁剪目标。
- `src/decision/slot_environment.py:853`：显式Cache/UAV状态的正式单Slot执行接口。
- `src/ppo_simple/rollout.py:374`：Reward分解合同检查。

共享Advantage无法给同一Macro帧内的九个Cache头和Termination提供不同的节点级反馈；当部分节点更新有益、部分无益时，联合系统收益不会直接说明是哪一个动作造成差异。反事实样本中确实有9个状态找到正收益UPDATE，而策略仍更偏好KEEP，这与信用归因不足相符，因此可列为可能贡献。但本轮没有记录每个节点动作对实际PPO梯度的独立贡献，也没有足够样本证明共享Advantage导致了该偏好，不能把它认定为已证实的唯一根因。

## 根因分类

| 项目 | 判断 | 依据及边界 |
| --- | --- | --- |
| ACTION_REWARD_SENSITIVITY | SUPPORTED | 同状态合法Cache动作的`ΔReturn`从-1.07705到+0.99860；成本可抵消Utility增益。 |
| CACHE_CANDIDATE_QUALITY | SUPPORTED（本样本） | 69个UPDATE有59个净收益低于KEEP；但存在10个正收益候选，不能归结为候选集完全无效。 |
| CACHE_POLICY_RANKING | SUPPORTED（局部偏保守） | 59/69收益排序与动作概率同向；但9个有益UPDATE状态的UPDATE组概率都低于KEEP组。样本不支持普遍排序失效。 |
| HOLDING_CREDIT_ASSIGNMENT | INSUFFICIENT_EVIDENCE | Duration-aware GAE实现存在；没有对Termination或不同Holding长度做同状态反事实。 |
| UAV_MOVE_VALUE | NOT_SUPPORTED（本样本） | 6/6 MOVE分支净收益为负；不外推到其他Slot。 |
| MACRO_ADVANTAGE_DISCRIMINATION | SUPPORTED（可能贡献，因果未确认） | 同一帧9个节点动作与Termination共享标量Advantage；正收益Cache动作被低估与此机制相符，但未单独测量节点梯度归因。 |
| PPO_JOINT_CLIPPING_EFFECT | INSUFFICIENT_EVIDENCE | 1e-3和5e-3平均Macro joint clip fraction分别为9.03%和18.12%；没有动作级证据证明裁剪造成有益候选被压制。 |

## 最终判断与下一步

`PRIMARY_ROOT_CAUSE`：没有确认的训练代码Bug。最直接的实测压力是，当前合法Cache UPDATE候选多数在真实Holding窗口内无法用Utility抵消更新成本；与此同时，九个因子化头让“至少一个节点发生变化”的Macro帧比例保持很高。共享Macro Advantage可能使少量有益节点更新难以从联合反馈中被单独识别，但不是已证明的唯一原因。

`RECOMMENDED_MINIMAL_FIX`：本轮不改生产代码。若继续定位，只优先增加一个离线的节点级Cache动作收益/Advantage归因诊断，在更多固定Training状态中验证上述9个有益UPDATE是否持续被低估；在结果复现前不改GAE、PPO目标、Reward、Holding上限或学习率。

```text
CONFIRMED_TRAINING_BUGS = NONE
ACTION_REWARD_SENSITIVITY = SUPPORTED
CACHE_POLICY_RANKING = SUPPORTED (LOCAL UNDERWEIGHTING; NOT GENERAL FAILURE)
MACRO_CREDIT_ASSIGNMENT_ROOT_CAUSE = SUPPORTED AS A PLAUSIBLE CONTRIBUTOR; CAUSAL SHARE UNCONFIRMED
UAV_MOVE_VALUE = NOT_SUPPORTED IN THE SIX SAMPLED COMPARISONS
PRIMARY_ROOT_CAUSE = CACHE UPDATE COSTS OFTEN EXCEED SHORT-WINDOW UTILITY GAINS; NINE-HEAD FACTORIZATION KEEPS ANY-UPDATE RATE HIGH
RECOMMENDED_MINIMAL_FIX = OFFLINE NODE-LEVEL CACHE CREDIT ATTRIBUTION DIAGNOSTIC; NO PRODUCTION FIX CONFIRMED
PPO_SOURCE_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
TRAINING_STARTED = NO
VALIDATION_STARTED = NO
```
