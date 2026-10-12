# Stage5B：PPO策略动作分布与确定性决策诊断

## 范围与方法

使用Stage5记录的Checkpoint：`runs/stage5_limited_ppo_pilot_stage3b_20261011_seed20001_24ep/initial_policy/checkpoints/latest_checkpoint.pt`和`.../trainer/checkpoints/latest_checkpoint.pt`。SHA-256分别仍为`820b1700f543935ba2f5667daa49524e3b9c93c8c74dc4b14ca08f1aab1b2edf`和`5cd22587725e218c61323dda1740d61c2e4e14b40f1952856e23db096b5b50e2`。使用Stage3B配置SHA-256 `dd3f0ae104109f264758ce47a2f478d5df13655638497fe576c1874424f02ba0`。Trainer、Policy及Stage5A修复后的Evaluator哈希分别与前序报告一致；本阶段未修改代码、配置或Checkpoint。

Stage5A已保存的确定性轨迹提供了Validation seeds 30001–30004的Slot 0动作及Macro操作概率，但不保存Micro逐候选概率或Observation。为补齐同状态概率，本次从已有Episode数据恢复每个seed的Episode初始Observation和正式候选列表，只做Checkpoint加载及策略前向计算；没有调用Matching/KKT、没有执行Episode rollout，也没有运行新Validation。

同一seed的初始/最终策略共用同一份Observation、初始Cache、UAV位置、候选特征数组、动作Mask和候选顺序。Macro输入为初始Cache下的352个候选（KEEP在index 0，351个合法UPDATE）；Micro每架UAV有19个候选（STAY在index 0，18个合法MOVE）。UAV1均以同一UAV0-STAY位置作为自回归条件。为使Micro配对输入相同，Micro候选特征固定按共同的初始Cache生成，即共同KEEP分支；初始策略自然轨迹在Slot 0选择UPDATE后，其自然Micro候选特征会依更新后的Cache改变，本报告不把那种不同输入下的概率当作同状态漂移。

Holding头另在相同的Slot 1 KEEP+STAY观察、`elapsed_holding_slots=1`下比较。这是用于隔离策略概率的共同状态探针；初始策略自然轨迹在Slot 0更新Cache后进入的Holding观察不同，故这组Holding概率不代表初始策略自然路径的Slot 1概率。

逐状态Observation、Cache/Micro候选特征哈希、合法Mask数量、概率与候选排序明细保存在[stage5b_policy_action_distribution_states.json](C:/Users/tangning/Desktop/code/adaptive_cache_mec/reports/stage5b_policy_action_distribution_states.json)。分析使用Conda环境`adaptive_cache_mec`中的Python解释器`C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`；没有调用训练入口。

## 同状态策略概率

下表给出Macro操作组概率。候选概率为条件概率`P(candidate | UPDATE)`；它与整组`P(UPDATE)`不是同一个量。

| Seed | 初始 P(KEEP) / P(UPDATE) | 初始确定性动作 / Top UPDATE | 最终 P(KEEP) / P(UPDATE) | 最终确定性动作 / Top UPDATE |
|---:|---:|---|---:|---|
| 30001 | 0.497206 / 0.502794 | UPDATE #172 / 0.003075 | 0.598132 / 0.401868 | KEEP / #5：0.004143 |
| 30002 | 0.497524 / 0.502476 | UPDATE #172 / 0.003079 | 0.599994 / 0.400006 | KEEP / #5：0.004179 |
| 30003 | 0.497185 / 0.502815 | UPDATE #172 / 0.003075 | 0.597850 / 0.402149 | KEEP / #5：0.004140 |
| 30004 | 0.497452 / 0.502548 | UPDATE #172 / 0.003079 | 0.599674 / 0.400325 | KEEP / #4：0.004173 |
| **平均** | **0.497342 / 0.502658** |  | **0.598913 / 0.401087** |  |

六次PPO更新后，在这四个共同初始状态上，平均`P(KEEP)`上升10.16个百分点，`P(UPDATE)`相应下降10.16个百分点。这是可观测的Macro分布漂移；最终策略仍给UPDATE约40.1%的总概率，并非UPDATE概率塌缩到零。

Micro两架UAV的组概率如下；四个状态上的确定性组动作均为STAY。

| Seed | UAV0 初始 P(STAY) / P(MOVE) | UAV0 最终 P(STAY) / P(MOVE) | UAV1 初始 P(STAY) / P(MOVE) | UAV1 最终 P(STAY) / P(MOVE) |
|---:|---:|---:|---:|---:|
| 30001 | 0.538384 / 0.461616 | 0.560236 / 0.439764 | 0.535967 / 0.464033 | 0.534505 / 0.465494 |
| 30002 | 0.544619 / 0.455381 | 0.600175 / 0.399825 | 0.537791 / 0.462209 | 0.537847 / 0.462154 |
| 30003 | 0.541911 / 0.458090 | 0.598240 / 0.401759 | 0.537359 / 0.462641 | 0.537896 / 0.462104 |
| 30004 | 0.539788 / 0.460212 | 0.568283 / 0.431717 | 0.536541 / 0.463459 | 0.534948 / 0.465052 |
| **平均** | **0.541176 / 0.458825** | **0.581733 / 0.418266** | **0.536914 / 0.463086** | **0.536299 / 0.463701** |

UAV0平均`P(MOVE)`下降4.06个百分点；UAV1仅上升0.06个百分点，基本未变。最终策略的UAV0仍有约41.8%、UAV1约46.4%的MOVE概率质量，所以“确定性全STAY”不能解释为Micro分布没有MOVE支持。

在相同共同Slot 1 Holding观察上的平均概率为：初始`P(CONTINUE)=0.487762`、`P(TERMINATE)=0.512238`；最终为`0.546067`和`0.453933`。四个状态上初始策略均确定性选TERMINATE，最终策略均选CONTINUE。Stage5A自然轨迹里，初始策略是在已经更新Cache的不同观察上Terminate，最终策略是在KEEP状态上Continue；自然轨迹的这两个概率不作为同状态比较。

## 候选概率、实际动作与确定性规则

Seed 30001的共同初始状态中，初始策略UPDATE组内最高候选为#172，条件概率`0.003075`，配置为`RSU_1`新增服务`{1,6,8}`、删除`{2,4,5,7}`。Stage5A轨迹显示初始策略四个seed都在Slot 0实际选择了该候选，候选状态ID均为`e6fb008c0bfd90e9`；下一次Macro决策在Slot 1选择KEEP的candidate 0，candidate ID仍是该更新后配置。四个Episode各发生一次UPDATE，Cache成本均为`0.2435639`，更新配置随后被KEEP保留，没有在Slot 1回退。

同一Seed 30001状态下，最终策略组内最高合法UPDATE是#5，条件概率`0.004143`，对应SV节点`53`新增服务5、删除服务1；但最终策略的`P(KEEP)=0.598132`大于`P(UPDATE)=0.401868`，所以实际选KEEP、候选0。四个最终策略Episode均无UPDATE、Cache成本为0。UPDATE条件下某个候选只有约0.3%–0.4%的概率，是因为组内有351个合法候选；这不能和约40%–50%的UPDATE操作组概率混为一谈。

Seed 30001 Micro的代表性MOVE候选也有条件概率：初始策略UAV0的最高MOVE候选为index 2（`projected_demand`，条件概率0.05627），最终策略为index 1（`demand_centroid`，0.06029）；UAV1由index 2（`projected_demand`，0.05606）变为index 1（`demand_centroid`，0.06032）。但MOVE组总概率分别低于STAY组，因此确定性动作是STAY，且没有选中这些MOVE目标。

代码核对结果：

- 训练在[trainer.py](C:/Users/tangning/Desktop/code/adaptive_cache_mec/src/ppo_simple/trainer.py:3295)调用相同的`collect_dual_timescale_episode`，传入`deterministic=False`；Evaluator调用同一rollout，依据`policy_action_mode`传入确定性开关（[evaluator.py](C:/Users/tangning/Desktop/code/adaptive_cache_mec/src/ppo_simple/evaluator.py:596)）。
- Macro合法候选由[rollout.py](C:/Users/tangning/Desktop/code/adaptive_cache_mec/src/ppo_simple/rollout.py:345)构造；candidate 0是当前Cache，其余为正式合法UPDATE。Micro候选由同文件的`_uav_candidates`构造，candidate 0是STAY，其余来自合法目标枚举。Stage5A保存的四个状态Macro mask为352/352，Micro每架UAV mask为19/19。
- `policy.py`对Mask后的Categorical使用相同动作概率分布。Macro的确定性选择比较KEEP与UPDATE的**组概率质量**，选UPDATE时再在合法UPDATE中按raw score取最高候选；Micro比较STAY与MOVE组质量，选MOVE时再在MOVE候选中取最高raw score（[`_group_aware_deterministic_candidate`](C:/Users/tangning/Desktop/code/adaptive_cache_mec/src/ppo_simple/policy.py:122)、[`sample_macro`](C:/Users/tangning/Desktop/code/adaptive_cache_mec/src/ppo_simple/policy.py:475)、[`sample_micro_head`](C:/Users/tangning/Desktop/code/adaptive_cache_mec/src/ppo_simple/policy.py:865)）。因此它不是对全部单个候选做一次平坦argmax。
- 随机训练调用Categorical采样；确定性评估采用上述组概率排序。Slot 0直接概率重算与Stage5A保存的初始/最终Cache概率、组动作、候选index相符；Micro确定性操作组也与保存轨迹相符。本次8个同状态策略输入的Macro概率和误差最大为`2.38e-7`，UPDATE组内argmax违规数为0（由配对状态数据逐项汇总）。

因此Stage5训练中的Cache UPDATE约47.7%、UAV MOVE约44.5%/47.9%，与随机Categorical在接近均衡的组概率上采样相容。Stage5A确定性策略只选组概率较大的KEEP/STAY。两者评估的状态分布也不同：训练使用20001–20024且跨6个策略版本，Validation使用30001–30004的确定性闭环。动作比例差异不构成动作映射、Mask或候选顺序错误证据。

## 经济价值与结论边界

Stage4B报告在seed 30020、Slot 100的合法UPDATE候选#115上发现一个受控20-Slot正净收益（`ΔG=+0.168025`）。这说明Stage3B正式候选机制能表达某个合法、在特定状态下有价值的UPDATE；本次概率诊断没有在该30020/Slot 100状态上评估初始/最终Checkpoint，也没有证明策略偏好或学到了该候选的长期经济价值。这里观察到的是分布改变，不是经济收益排序改善。

在所采样状态中没有策略塌缩证据，也没有发现训练/Evaluator动作映射、合法Mask或确定性规则不一致；不需要修改代码。6次更新确实改变了Macro和UAV0的首状态概率，但四个Validation初始状态不足以判断这种漂移是否泛化或改善长期Return。是否进入更长训练验证仍为未定；若后续需要把随机采样与确定性规则对Return的影响分开，建议另做少量同seed随机策略配对评估，本阶段未启动该评估。

```makefile
STAGE5B_DONE = YES
SAME_STATE_POLICY_PROBABILITIES_COMPARED = YES; 4_VALIDATION_SEEDS_SLOT0
MACRO_PROBABILITY_DRIFT = OBSERVED; MEAN_DELTA_P_UPDATE=-0.101571
MICRO_PROBABILITY_DRIFT = UAV0_MEAN_DELTA_P_MOVE=-0.040558; UAV1=+0.000615
DETERMINISTIC_ACTION_CONSISTENT = YES; GROUP_MASS_THEN_WITHIN_GROUP_ARGMAX
TRAIN_EVAL_ACTION_CONTRACT_CONSISTENT = YES
POLICY_COLLAPSE_EVIDENCE = NO_IN_SAMPLED_STATES
IMPLEMENTATION_BUG_FOUND = NO
LONG_TERM_VALUE_LEARNING_EVIDENCE = UNDETERMINED
READY_FOR_EXTENDED_TRAINING = UNDETERMINED
```

