# 分层PPO自适应控制行为机制诊断

> 范围：仅检查现有5e-4确定性Validation（seed 30001–30030，每个300 Slot），对应逐seed `proposed_repeat_summaries.json`；只读使用既有Validation环境输入 `tasks.csv`、`popularity.csv`，并定向阅读动作执行与收益代码。没有运行训练、Validation、Test或反事实仿真，也没有修改算法或原始结果。

## 结论速览

```ini
HOLDING_STRUCTURAL_PROBLEM = CONFIRMED: 当前系统不计Macro重决策成本；Holding=1可用逐Slot KEEP复现Holding=5的Cache状态序列
CACHE_LEARNING_PROBLEM_CONFIRMED = INSUFFICIENT_EVIDENCE: 需求变化且更新候选存在，但日志不能给出KEEP与UPDATE的同状态收益差
UAV_LEARNING_PROBLEM_CONFIRMED = INSUFFICIENT_EVIDENCE: STAY集中可由Argmax解释；位置和任务匹配模型有作用，但缺少逐节点结果与MOVE反事实
RECOMMENDED_FIRST_CHANGE = 先定义Holding是否应有独立决策成本；Cache/UAV先做匹配seed反事实，不先改PPO
RETRAINING_REQUIRED_NOW = NO: 当前没有已证实的Cache/UAV学习失效或必须立刻修复的问题
```

`CONFIRMED`表示模型结构/执行逻辑已经由代码和现有记录验证；`INSUFFICIENT_EVIDENCE`不等于策略正确，只表示这些数据无法区分合理决策与价值学习失效。

## 1. Holding：决策流程与模型结构

### 实际控制流程

`rollout.py`在每个Macro frame开头生成Cache候选并选择一次Cache目标；所选配置在该frame内执行Slot环境。Slot结束后，只要达到最短Holding且未到上限，Termination head便在后状态上选择继续或终止。UAV的Micro动作仍然每Slot重新选择，与Macro Holding长度分开（`src/ppo_simple/rollout.py:601-615, 685-687, 774-815, 953-1008`）。

- **Holding=1：**每个Slot结束一个Macro frame，下一Slot重新评估Cache候选。Validation中300次Cache决策/seed，且每个非终止Slot都在elapsed=1选择TERMINATE。
- **Holding=5：**Cache目标最长连续保持5个Slot；elapsed=1–4的Termination选择CONTINUE时不重新选择Cache，达到5后强制结束frame。UAV仍然逐Slot决策。
- **重复KEEP的等价性：**从Cache配置状态序列看，Holding=1在中间Slot反复选择当前配置（candidate 0 / KEEP），即可保持与Holding=5相同的Cache配置；它额外获得了重新观察与选择更新的机会。这个等价结论仅针对配置状态序列，不表示两个策略的概率、学习轨迹或观测完全相同。

### 是否有重决策成本

当前Slot收益合同是 `system_utility = actual_task_utility - cache_cost - movement_cost`，rollout校验的reward也完全等于这三项（`src/decision/slot_environment.py:480-504`; `src/ppo_simple/rollout.py:279-291`）。没有发现另行扣除Macro决策计算、通信、信令或控制时间的成本，也没有把策略推理耗时折算进Slot时延/效用。Cache更新成本是另一项：它按实际改变的节点、载入数据量和配置参数计算；未改变配置不会产生该成本（`src/decision/cache_configuration.py:407-425`; `src/common/system_model.py:214-295`）。

因此，现有目标没有直接奖励“少做Macro决策”。Holding=1可以不付决策代价地获得更多Cache更新机会，并通过KEEP避免更新成本；当前Holding学到最短周期符合这个目标结构，不能据此判定代码错误或PPO失败。若现实系统确有控制信令/推理时延成本，应先将其明确定义并校准；否则Adaptive Holding头缺少清晰的独立优化目标，固定周期基线比继续学习该头更易解释。

### 现有Holding结果

30个seed共9,000个实际Macro frame全部为1 Slot；2–5 Slot均为0。Termination共8,970次（每个episode最后一个Slot不再采样），选择TERMINATE为100%。概率均值为0.7510，范围0.6251–0.8290；概率会随观测/时间改变，但每次都在Argmax的终止侧（>0.5）。确定性规则直接对Termination logits取Argmax（`src/ppo_simple/policy.py:595-620`）。这是“概率变化但选择侧不变”，不是策略输出完全恒定。

**Holding结论：**固定1 Slot是当前目标下可解释的优化结果；需要关注的是决策成本缺失造成的目标冗余，而不是已证实的Holding代码故障。若不打算计价决策频率，没有必要把学习Adaptive Holding作为当前优先修复项。

## 2. Cache：需求、候选与实际更新

### 已证实的行为

- 每seed 300次Cache决策；每seed实际发生5次配置变化，全部在Slot 0、1、2、3、4；Slot 5–299没有再变更配置。30/30 seed的Slot模式相同。
- 每次决策都有352个合法候选：当前配置KEEP一项及351个合法UPDATE配置。UPDATE候选要求目标二进制Cache与当前配置不同且满足容量约束（`src/decision/cache_configuration.py:907-970, 1129-1159`）。因此，后期长期KEEP不是因为候选集合为空。
- Slot 0–4的平均p(KEEP)=0.114；Slot 5–299平均为0.728，后期所有记录仍高于0.5（全局最小0.5737，最大0.8394）。p(KEEP)并非恒定，但确定性组Argmax始终落在KEEP侧。记录中的5次UPDATE与5次实际配置变化一致。
- 既有输入中的服务需求确实随时间变化。比较每个seed的Slot 5–149与150–299：实际请求服务分布的总变差距离（TVD）均非零，均值0.097、范围0.024–0.186；`popularity.csv`中rho_star服务分布的TVD均值0.084、范围0.029–0.164。所有seed的300个Slot都有至少一项到达任务。这证明后期需求输入不是静止的，但不证明当前Cache失配。

### 不能从现有记录得出的结论

Cache原始动作记录保留候选数量、候选ID哈希、所选组和p(KEEP)，但没有保存每个Slot的当前/目标Cache服务列表。汇总也没有逐任务的服务ID到执行节点匹配、命中/覆盖情况或每个候选的实际Task Utility。因此无法量化后期需求变化是否碰到未缓存服务、某个合法UPDATE会改善哪些请求、UPDATE收益是否高于真实重配置成本。不能把KEEP比例解读为缓存未命中，也不能仅凭配置不动判定策略失效。

最小补充信息是：决策Slot的当前Cache与选中UPDATE目标（各节点缓存的service_id或可反查的配置映射），以及该Slot任务的service_id/执行节点/Task Utility；若要说明收益，应有同一状态下KEEP与UPDATE的实测分支结果。候选ID哈希自身不足以回答覆盖差异。

### 最小同seed反事实方案（建议，未执行）

以每个seed Slot 5决策前的相同环境状态为分叉点，复制当前Cache、UAV位置、环境时钟/外生任务与车辆轨迹、策略与随机流。分支A选择KEEP；分支B选择一个合法且与当前配置不同的UPDATE（可预先固定规则，例如当前策略UPDATE组内最优候选）。两支使用同一5e-4确定性策略继续控制后续Slot；Cache和UAV按各自分支状态自然演化，不强行锁定未来动作。分支B按实际配置差异扣除一次真实Cache更新成本。比较完整余下Episode的Return、Task Utility、时延、能耗与Cache成本，按30个seed配对汇总。

这样的分叉会让后续观测、匹配和策略动作随配置改变而自然变化；不要每Slot重置到原配置或只把当前Slot的任务收益当作长期因果效果。应保留Slot 0–4已发生的同seed历史。当前缺少可复用的Slot 5状态快照，因此这需要未来按协议运行的配对环境分支；本次没有启动。

**Cache结论：**需求变化、更新动作可用、后期动作固定KEEP均已证实；UPDATE相对KEEP有无性能损失、是否为PPO动作价值学习失败，证据不足。暂不改Cache策略或PPO。

## 3. UAV：任务承接、移动收益与成本

### 已证实的动作与选择机制

两架UAV在9,000个Slot中各有9,000次动作，共18,000次均选择STAY；全部选择候选0、来源`stay`。两架UAV的已选坐标在每个seed内不变，而且在这30个seed之间也各只有一个坐标值。现有记录中没有一次被选中的MOVE目标，故其目标位置没有状态变化。

每个seed的平均p(MOVE)约为UAV0=0.1354（seed范围0.1290–0.1402），UAV1=0.1329（0.1280–0.1377）。也就是说确定性策略不是把MOVE概率置零，而是保持较大STAY组概率后由Argmax固定选STAY。Policy先比较STAY/MOVE组概率，再在MOVE组内选分数最高的位置（`src/ppo_simple/policy.py:145-157, 895-910`）。因此观察到的行为属于“动作组固定，实际目标位置也固定”；不代表MOVE候选位置本身缺乏变化。

系统为UAV位置提供了实际作用路径：`build_service_node_positions`把UAV坐标作为执行节点坐标传入任务Matching，UAV信道增益随车辆与UAV的水平距离变化（`src/decision/task_allocation.py:1607-1626, 2990-3005`）。执行结果包含每个任务的Matching节点、时延、车辆能耗和Utility（`src/decision/slot_environment.py:511-553`）。合法MOVE位置由需求质心、投影需求点和几何探索点生成，并经过边界、速度与多UAV安全检查；与当前位置相同的目标被排除（`src/decision/uav_configuration.py:323-435`）。所以模型机制能表达不同位置和不同链路/任务匹配结果。动作概率均值为正说明至少有状态提供MOVE候选，但现有summary没有保存每Slot合法候选数及未选MOVE坐标，无法确认所有Slot的具体候选集合。

移动成本也进入收益：按实际位移计算固定移动成本及移动能耗项，并从Task Utility中扣除（`src/decision/uav_configuration.py:570-601`; `src/common/system_model.py:299-355`; `src/decision/slot_environment.py:480-504`）。现有STAY轨迹只说明实际移动成本为0，不能回答MOVE收益是否高于移动成本。

### 现有日志能否证明UAV利用率或策略失效

不能。summary保存总任务数、总时延/能耗和联合动作率，但没有保存逐任务匹配到UAV0/UAV1、SV、RSU或Cloud的计数及其Utility/时延。运行时确有任务到执行节点的Matching结果，rollout内也临时汇总执行节点类别；正式repeat summary没有将这些逐Slot/逐UAV结果输出（`src/ppo_simple/rollout.py:75-96, 915-950`; `src/ppo_simple/evaluator.py:242-352`）。因此不能从MOVE率推断UAV任务利用率或STAY是否有效。

最小输出为每Slot按实际执行节点（尤其UAV0、UAV1）统计任务数、Task Utility及延迟，并记录UAV决策前后坐标、所选动作、合法MOVE候选数/坐标和实际移动距离/能耗/成本。这些字段可解释行为；要证明因果收益仍需配对分支。

### 最小同seed反事实方案（建议，未执行）

在后期选定的STAY状态（例如Slot 150）分叉同一seed的状态与外生任务/车辆轨迹：基线保持STAY；干预分支分别令UAV0或UAV1选择一个当时合法、坐标不同的MOVE候选。其他机制仍使用同一Checkpoint和策略，在各自更新后的真实状态继续运行，让后续任务匹配与策略动作自然响应；按系统公式只对干预移动及后续真实移动收取成本。先做单UAV分支以隔离作用，再视结果评估双UAV联合Move。使用相同30个seed配对比较Return、Task Utility、时延、能耗、每架UAV任务承接数和Movement Cost。不得把移动后分支强行重置回STAY状态。

**UAV结论：**动作组集中和位置不变是既成事实；系统具备空间信道、任务分配及移动成本机制，当前数据没有证明Move无效，也没有证明PPO学错。保留现有UAV控制，不因0% MOVE直接改策略。

## 4. 已证实问题、假设与最小改进

| 模块 | 已证实 | 尚未证实 | 若后续确认有损失，最小处理 |
|---|---|---|---|
| Holding | 当前奖励不含独立Macro重决策成本；确定性策略固定1 Slot；H=1可用KEEP复现H=5的Cache状态序列 | 决策耗时/信令在目标环境中是否有真实代价 | 若要优化低频控制，先定义并校准每次Macro决策成本/时延，再比较1–5 Slot；若没有现实成本，把Holding固定为基线周期而不继续训练Termination head |
| Cache | Slot 0–4更新，后续长期KEEP；候选持续存在；请求与流行度分布后期仍变化 | 后期Cache覆盖缺口与KEEP/UPDATE净收益；Actor/Critic是否错误估价 | 先用配对分支确认损失；若UPDATE有正净收益，只在Cache模块引入可观察的需求漂移/覆盖缺口信号并核对动作价值，不重构PPO |
| UAV | 全程双STAY且位置固定；MOVE概率非零；位置会进入链路、Matching和移动成本 | STAY承接了多少任务；任一合法Move在扣成本后是否提高Return | 先输出逐UAV Matching并做单机配对Move分支；只有净收益持续为正而策略未选择时，再局部修正该动作头的状态/价值信用链 |

目前没有发现确定性Argmax实现错误：Term head直接取logit Argmax；Cache与UAV使用组概率质量Argmax，再在UPDATE/MOVE组内选候选（`src/ppo_simple/policy.py:145-157, 595-620, 895-910`）。0.51这样的组概率就足以让确定性动作持续落在该组；本次5e-4的STAY概率约0.865，因此行为完全可由确定性选择机制解释。该解释不构成动作收益证明。

## 5. 最终判断

- **Holding：**结构层面确认缺少重决策成本，当前固定1 Slot是可解释结果；没有必要仅因固定周期就宣判Holding head训练错误。除非要在目标中加入可测决策成本，否则优先将其作为固定周期基线。
- **Cache：**有真实需求变化和合法配置选择空间，后期KEEP是否导致服务损失仍未知。现有证据不能确认Cache PPO学习失败。
- **UAV：**动作组与坐标均固定STAY，但环境模型包含位置相关链路和移动成本；缺少执行节点明细与MOVE分支结果。现有证据不能确认UAV PPO学习失败。
- **是否立即重训：**不需要。先完成上述最小同seed反事实和必要结果读出，再决定是否要改变目标或局部动作信用。若以后给目标新增决策成本或改变策略机制，再对该新目标/机制重新训练和统一评价。

### 关键数据与代码来源

- Validation原始数据：`runs/proposed_validation_lr_5e_minus_4_seed30001_30030_deterministic/episode_<seed>/proposed_repeat_summaries.json`
- 同seed环境输入：`environment_data/proposed_joint_validation/episode_<seed>/tasks.csv`、`popularity.csv`
- 机制对照：`reports/four_macro_lr_validation_comparison.md`
- 核心实现：`src/ppo_simple/rollout.py:269-291, 601-710, 774-815, 869-897, 953-1008`; `src/ppo_simple/policy.py:145-157, 595-620, 895-910`; `src/decision/slot_environment.py:480-553`; `src/decision/cache_configuration.py:407-425, 907-970, 1129-1159`; `src/decision/task_allocation.py:1607-1626, 2990-3005`; `src/decision/uav_configuration.py:323-435, 570-601`; `src/common/system_model.py:214-295, 299-355`。
