# Stage4C-2：正式仿真环境冻结

## 冻结边界与版本

本报告冻结的是论文公式下的**仿真合同和当前工作树快照**，不是目标物理设备的标定结论。核对时间：2026-10-11。Conda环境为`adaptive_cache_mec`。本任务没有运行仿真、训练或Validation，也没有改动源码、配置、历史结果或Checkpoint。

- Git branch：`main`
- Git HEAD：`ffc137e8a1e599a1ef49944db040c987a355e791`
- 工作区：dirty。相对HEAD已修改的跟踪源码：`src/common/project_config.py`、`src/common/system_model.py`、`src/decision/cache_configuration.py`、`src/decision/slot_environment.py`、`src/decision/task_allocation.py`、`src/ppo_simple/evaluator.py`。`configs/proposed/stage3_cloud_finite_60ghz.toml`为未跟踪文件。另有未跟踪`environment_data/`、诊断报告、图表和`tools/`。
- 下表的SHA-256绑定到上述HEAD加当前工作树文件内容；HEAD单独不足以重建Stage3B实现。复现时须使用表内文件版本。工作区中其他诊断、数据和工具文件保持原样。

### 正式配置及算法源码SHA-256

| 文件 | 相对HEAD状态 | SHA-256 |
|---|---|---|
| `configs/proposed/stage3_cloud_finite_60ghz.toml` | untracked | `dd3f0ae104109f264758ce47a2f478d5df13655638497fe576c1874424f02ba0` |
| `src/common/system_model.py` | modified | `e0ccd18e122e3e01d96e097128ad591ab70ef8a2fbb29d980d328597dd36f645` |
| `src/common/project_config.py` | modified | `24b8e3efe64e435731e433452974a112d6a57175af838332e41b2dbe6daa4f6b` |
| `src/decision/cache_configuration.py` | modified | `44a5580c1989d749c5a1b050f81c5b7298b6e28d58efb178dacc5dce7b738cf6` |
| `src/decision/task_allocation.py` | modified | `d785a363aef5c7f24668cfa29274a3859828cd434ce5e18cc60325c5dc3e34e1` |
| `src/decision/slot_environment.py` | modified | `3ef733984050c2674989c3eff9c05fde1ca7c48f6f6f1f83f08ff05e156bdbb1` |
| `src/decision/uav_configuration.py` | clean | `556ee3d7ab2478bef341a6393c3fa6d4ba2142e99618b5aa88cb759454252412` |
| `scripts/training/run_proposed_formal.py` | clean | `70f24d639e017b00345cb299ab83b982acc59ec33f404ce2a8493dfe6b83051b` |
| `src/ppo_simple/train_formal.py` | clean | `0868ecd8d35c94a620a5cdf3fb2ef9b9563e6837e1cc53f41dc0413d2e1cd77c` |
| `src/ppo_simple/trainer.py` | clean | `9923281c36ae3d8931f37abb9858376727e6daa4952f3e7137af71914e712f97` |
| `src/ppo_simple/policy.py` | clean | `c17fbd14085ac0cca2a3bc923de12b6865c384978c4b0f8b1dc376213b90a041` |
| `src/ppo_simple/rollout.py` | clean | `d88bbf39a85156a52ee1fbcc266d21664d62fd3a985f573831408a5f1747653f` |
| `src/ppo_simple/interface.py` | clean | `89b6a009edd953d92606d213289a10e9986082026b171863e73c0c45affb5fa3` |
| `src/ppo_simple/macro_targets.py` | clean | `a3b2ad4662a2a2f928519ff854d1bfd7330e888e4e06ed9eed52b401b6b93412` |
| `src/ppo_simple/primitive_control.py` | clean | `3d54e30c2f34902836cb6d66527901e1a5732803a7f840ddffa0311b814599e1` |
| `src/ppo_simple/reporting.py` | clean | `bb3542cdaf1668e67f10ae60be9b35755f4d2b926c5ce4ebff4b7d870dbdc35d` |
| `src/ppo_simple/evaluator.py` | modified | `2044f8e9feecfca9893e6ec662604f4508565a9fcd1de1fbeb98ee354703cb22` |
| `src/scenario/realization.py` | clean | `adcf54c3d871e251d55d88318cb018d2d3dac98dc6b605c50a9f8b8693084806` |
| `configs/proposed/system_config.toml` | clean; legacy wrapper default only | `fe5d4f7bc57a424911166bcfe921c2ec5a7627b0a6aa759c4993f7f488a3775c` |

## 冻结的系统模型

- **时间语义：**Stage3C-0确认为`INDEPENDENT_SLOT_PERFORMANCE_SNAPSHOT`。每Slot评价当Slot任务效用；任务不跨Slot延续，没有未完成工作量或跨Slot资源占用。多秒时延是效用输入，不表示任务在后续Slot继续执行。
- **Edge：**每个SV、RSU、UAV分别共享自己的CPU和带宽；各节点内CPU份额与带宽份额分别满足总和不超过1，并在候选Matching后按原KKT分配。速率`R_vm=b_vm B_m log2(1+P_v h_vm/σ²)`，时延`T_vm=D_v/R_vm+L_v/(w_vm F_m)`，车辆上传能耗`E_vm=P_v D_v/R_vm`。
- **Cloud上行：**每个Cloud任务车辆固定使用等效`R_c=200 Mbps`，不建模Cloud接入用户之间的带宽竞争；`total_bandwidth_bps`只是兼容接口字段。Cloud CPU仍是`finite_pool`总计`60e9 cycles/s`，同Slot Cloud任务按原KKT共享。
- **Cloud公式：**`T_vc=ψ_c D_v/R_c + L_v/F_c`；车辆上传能耗`E_vc=p_v D_v/R_c`；`ψ_c=11`仅影响Cloud传输时延。200 Mbps、60 GHz和`ψ_c=11`都是当前仿真合同参数，不表示物理网络保障或已完成设备标定。
- **Cache/UAV成本：**Cache UPDATE仍调用原`system_model.cache_update_cost()`：每节点成本为`weight·(z_m η_m + λ_cache ξ_m·loaded_MB_m)`，其中`loaded_MB_m`按新增服务大小计算。UAV MOVE仍调用原`system_model.uav_movement_cost()`：`E_move=ξ_move·d`、`C_move=weight·(z·η_move + λ_move E_move)`，移动距离继续受`d≤v_max Δt`限制。两者公式及权重未在本阶段更改。
- **Matching/KKT与Reward：**继续使用Transfer-Exchange Matching及原KKT资源分配。逐Slot总Reward保持`G(t)=Q(t)-C_cache(t)-C_move(t)`。

### 数值与历史哈希

Stage4C-1B已按现有配置加载器解析通过。该报告记录的Cloud值为200 Mbps、60 GHz、`resource_sharing=false`、`ψ_c=11`。当前Stage3B TOML的非注释内容哈希为`5df475736ced9a2571cd3d09e961ef195b134ee7ed540ce983048b30c860c65b`；与Stage4C-1B记录一致，表示配置键值和数值未因合同文字修正而改变。

Stage4C-1B把TOML原始SHA-256从历史值`1b14a49f430aca3d4c795e576010d9aa0c5e71405701440b3aebaec348578527`改为当前值`dd3f0ae104109f264758ce47a2f478d5df13655638497fe576c1874424f02ba0`，变化仅来自注释文字。3C及4B等历史诊断继续对应其记录的旧SHA；本报告不改写历史元数据，也不把旧实验称为使用新哈希。

## 证据、未标定项与Pilot条件

- Stage4C-0/4A/4C-1A/B支持当前Cloud公式与固定等效上行合同的仿真语义；目标蜂窝接入的物理并发速率仍未标定。60 GHz Cloud CPU池、`ψ_c=11`、若干Edge资源和无线链路假设也不是目标设备测量值。它们是固定仿真参数，不妨碍按该模型开展仿真研究，但不能表述为硬件验证。
- Stage4B在36个预测筛选的合法Cache UPDATE配对中发现1个20-slot正净收益案例：seed 30020、Slot 100、候选#115（RSU_0以服务2替换服务1、5、8、9），20-slot`ΔG=+0.168025`；5-slot`ΔG=-0.067777`。当前正式rollout候选机制包含KEEP及该状态下的合法候选集合（352项）；Stage4B的正例可作为机制诊断参照，但仅是稀疏、窗口相关的一个受控正例，不保证策略学到该动作。
- Stage3C-2冻结旧策略在两个seed合计600个Macro点中KEEP 590次、UPDATE 10次；1,200个单UAV动作均为STAY。4次受控UAV完整后缀迁移差值为小幅正值（`+0.039`至`+0.093`），而即时Reward差值为负；这不足以证明策略能学到长期移动收益。Cache/UAV Pilot都应观察配对长期Return和分项贡献，不能把UPDATE/MOVE频率作为成功标准，也不能反复拟合单个4B正例。
- Stage3B配置记载的完整realization划分互不重叠：训练seed `20001–20300`，Validation `30001–30030`，最终Test `40001–40100`。Stage5只可从训练池采样训练，并用Validation作诊断；Test seeds不得进入训练、调参或Pilot选择。
- **入口约束：**`scripts/training/run_proposed_formal.py`和`src/ppo_simple/train_formal.py`默认指向旧`configs/proposed/system_config.toml`，并设置300-episode正式训练，不应原样用于Stage5。通用`src/ppo_simple/trainer.py`解析器支持显式`--config`、`--epochs`、`--run-dir`、`--episode-pool`和`--episode-seed-start`。Pilot必须显式指定本报告冻结的Stage3B TOML、受限训练规模、全新输出目录以及训练池；不得续训旧Run/Checkpoint。Validation继续使用其独立seed区间。

进入Stage5时应预先固定Pilot样本与长期收益诊断口径，保留KEEP/STAY对照，并报告完整Pilot回报、Cache/UAV成本和任务效用分项；4B正例只作为参照，不以单例复现或动作频率作为通过条件。

```makefile
STAGE4C2_DONE = YES
SYSTEM_MODEL_FORMULAS_FROZEN = YES (simulation-contract level)
NUMERIC_PARAMETERS_FROZEN = YES (Stage3B TOML non-comment SHA recorded)
CLOUD_CONTRACT_FROZEN = YES (fixed 200 Mbps equivalent rate per task; no access contention)
MATCHING_KKT_FROZEN = YES (Transfer-Exchange + existing KKT)
PPO_IMPLEMENTATION_BASELINE_RECORDED = YES (entrypoints, direct modules, dirty evaluator hash listed)
REPRODUCIBLE_SOURCE_SNAPSHOT = YES_WITH_HASHED_DIRTY_WORKTREE; NOT_CLEAN_HEAD_ONLY
HISTORICAL_ARTIFACTS_PRESERVED = YES
READY_FOR_STAGE5_LIMITED_PILOT = YES_WITH_EXPLICIT_STAGE3B_CONFIG_OVERRIDE_AND_TRAIN/VALIDATION/TEST_SEPARATION
```
