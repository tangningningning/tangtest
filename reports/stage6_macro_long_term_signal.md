# Stage6 Macro长期收益信号分析

日期：2026-10-12  
环境：`adaptive_cache_mec`（Conda环境；Python位于`C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`）  
范围：只读取Stage6已有数据与当前配置/必要PPO代码；未运行Matching/KKT、仿真、训练或测试。

## 1. 已有反事实结果与逐Slot曲线的数据边界

Stage6缺失服务报告为服务2候选保存了5和20 Slot的窗口汇总，没有保存Slot 1–20逐Slot的ΔTask Utility或ΔReward。`runs/stage6_candidate_feasibility_25_20_12/paired_slot_results.json`虽有`per_slot`，但它对应的是更早的不同Cache动作：seed 30020为SV 28删除服务7、加入服务5；seed 30023为UAV_0删除服务5、加入服务4。它不能替代当前两条RSU_0服务2 UPDATE轨迹。故不能在不重跑Matching/KKT的前提下补出精确的1–20 Slot曲线或精确回本Slot；以下只给出已保存的5、20 Slot端点及可证明的区间。

| 决策前状态 | 服务2 UPDATE | 更新成本 | ΔG累计(5 Slot) | ΔG累计(20 Slot) | PPO折扣后(5 / 20 Slot) | 成本回收时点 |
|---|---|---:|---:|---:|---:|---|
| Seed 30020 | RSU_0：删服务3、加服务2 | 0.106998 | −0.197614 | +0.132061 | −0.197614 / +0.132061 | 已知在Slot 5后、至Slot 20前达到非负；具体Slot无法确认 |
| Seed 30023 | RSU_0：删服务3、加服务2 | 0.106998 | −0.094806 | −0.238407 | −0.094806 / −0.238407 | Slot 20仍为负；是否曾在中间短暂回本无法确认 |

其中ΔG是UPDATE分支相对同seed KEEP分支的累计净Return差，已包含一次性Cache更新成本。seed 30020的任务效用差ΔQ在5/20 Slot分别为−0.090616/+0.239059；seed 30023分别为+0.012192/−0.131409。因而20 Slot仅seed 30020超过更新成本，seed 30023的20 Slot净差为负。

## 2. PPO折扣与Macro时间尺度

Stage6候选配置使用`simulation.dt_s=1 s`、`algorithm.ppo.gamma=1.0`、`gae_lambda=0.95`、Cache Macro Holding范围1–5 Slot。`cache.discount_beta=0.95`是预测需求的时域折扣，不是PPO奖励折扣。

Rollout在Macro frame内按`gamma ** elapsed_in_frame`累计slot Reward；Macro SMDP-GAE按实际持续长度D计算`discount=gamma^D`及`trace=(gamma*gae_lambda)^D`。因此本配置对奖励的折扣为`1.0^D=1`：上表中5/20 Slot实际累计ΔG也是PPO折扣后的Δ奖励差，正负号不变。GAE trace则不同：D=1时为0.95，D=5时为`0.95^5=0.773781`；多个后续Macro frame的TD残差按各个持续长度的trace因子连乘回传。它会衰减较远TD残差的贡献，但Holding结束不会重置轨迹或截断到当前frame。

## 3. 合法候选与Observation可区分性

在seed 30020和30023各自Slot 100的决策前状态上，正式`rollout._cache_candidates`集合均包含服务2 UPDATE：候选索引91，改变RSU_0，删除服务3并加入服务2；两seed候选集合各有231项，候选0为KEEP。目标配置均通过正式容量约束检查。Rollout将生成候选的mask设为全合法，因此这两个状态中的该动作可供Macro策略选择。索引是各自Slot 100候选集合内的顺序，不是固定的全局动作ID。

按现有`interface.build_step_observation`对两个已有Slot 100输入构造正式Observation（不执行环境或Matching）：Observation维度为400。两状态的Cache Observation片段相同、UAV位置片段相同；预测车辆位置片段有250维不同、预测流行度片段有50维不同，合计300维不同。因此当前Observation能够区分这两个决策前状态；这只证明状态可表达，不代表策略已经学会在其中偏好UPDATE。

## 4. SMDP-GAE可接收的信号与不能据此推出的结论

- **理论传播：** Macro rollout覆盖完整Episode，Cache选择与Holding终止记录为Macro transitions；`compute_macro_smdp_gae`沿同一Episode的Macro transition序列递推。后续KEEP frame产生的任务Reward能够通过duration-aware GAE和下一状态价值回传到先前UPDATE frame。gamma=1意味着时间间隔本身不折扣目标Reward；lambda=0.95对跨frame的TD残差施加trace衰减。
- **Critic估计范围：** Macro Value Critic用`two_step_smdp_td`目标：当前Macro frame Reward、下一Macro frame Reward，再对后续价值bootstrap。独立Long-Value Critic另以Episode realized return-to-go为目标训练。当前Actor的Macro advantage仍由Macro Critic值计算的SMDP-GAE产生；Long-Value Critic的return-to-go拟合不等于该动作的Advantage，也不能把上表的反事实ΔG直接称作Critic估计值。
- **实际证据边界：** 现有Stage6结果没有保存这两个新配置状态下的Actor概率、Macro Critic预测、GAE Advantage或逐Slot反事实Reward。故只能判定奖励目标中存在可跨Holding传播的理论路径，不能判定实际Critic已准确估值，亦不能由两个受控分支推出PPO会学到该UPDATE。

## 5. 下一步判断

Stage6下的服务2 UPDATE已在正式Macro候选中，状态Observation也能区分；gamma=1未改变两个20 Slot反事实端点的正负关系。30020在5 Slot为负、20 Slot转正，构成一个有限长期收益信号；30023在20 Slot仍为负，说明该动作价值依赖状态，不能视作普遍优选动作。

**建议可进入新配置的受限PPO Pilot，但不据此直接进入正式长训练。** Pilot应保留当前模型和策略实现，并记录Macro frame持续长度、逐Slot/逐frame Reward、Cache候选概率、Macro Critic值及SMDP-GAE Advantage，以便检验信号是否真正到达策略。精确回本Slot仍属未确定；若后续决策要求知道具体回本时间，需要另行获得逐SlotUPDATE与KEEP配对结果，本次未重跑以遵守限制。

```makefile
SLOTWISE_1_TO_20_CURVE_AVAILABLE = NO
PPO_GAMMA = 1.0
GAE_LAMBDA = 0.95
SERVICE2_UPDATE_IN_FORMAL_MACRO_CANDIDATES = YES
OBSERVATION_DISTINGUISHES_THE_TWO_STATES = YES
CROSS_HOLDING_SMDP_GAE_PATH_EXISTS = YES
ACTUAL_CRITIC_ADVANTAGE_FOR_THESE_ACTIONS = UNDETERMINED
DISCOUNTED_DELTA_G_30020_SERVICE2_20SLOT = +0.132061
DISCOUNTED_DELTA_G_30023_SERVICE2_20SLOT = -0.238407
READY_FOR_NEW_CONFIG_PPO_PILOT = YES_LIMITED_PILOT
```
