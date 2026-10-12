# Stage5：Stage3B受限PPO Pilot

## 结论摘要

Pilot按Stage4C-2冻结的Stage3B工作树完成24个Episode和6次PPO更新。训练采样、优化与日志落盘正常；Episode和Slot两级Reward恒等式误差均为0，所有Episode、更新、Slot及Critic诊断日志中的数值均为有限值。没有发现确定的PPO数值故障或Macro长时域信用截断。

Cache UPDATE比例约47.7%，UAV MOVE比例约44.5%/47.9%；候选和移动动作都有学习轨迹，但本次没有可用的独立Validation结果，也没有同一seed的训练前后评估。因此不能据此认定策略已学会Cache经济价值或UAV长期移动价值，也不能证明策略改善或稳定收敛。结论属于“入口和更新正常，但Pilot规模太小，学习能力未定”。

## 冻结版本、哈希与运行产物

运行时间：2026-10-11；环境：`conda activate adaptive_cache_mec`。Git branch为`main`，HEAD为`ffc137e8a1e599a1ef49944db040c987a355e791`。工作区在Pilot前已dirty；Stage4C-2已将Stage3B定义为“HEAD加哈希绑定的dirty工作树”，不能仅用HEAD重建。Pilot使用的下表内容与Stage4C-2记录一致；Pilot未改动源码或正式配置。

Stage3B配置原始SHA-256为`dd3f0ae104109f264758ce47a2f478d5df13655638497fe576c1874424f02ba0`，运行Manifest中的语义配置SHA-256为`d69c6416adb3d1356ea0f233d963ba4f25d66c2eb0fd733798bd1684ef327d75`。4C-1B只改了注释，非注释配置SHA与冻结记录一致。更早诊断报告中的TOML原始SHA为`1b14a49f430aca3d4c795e576010d9aa0c5e71405701440b3aebaec348578527`；Stage3C/4历史实验仍属于其原记录哈希，本报告不改写历史元数据。

| 文件 | SHA-256 |
|---|---|
| `configs/proposed/stage3_cloud_finite_60ghz.toml` | `dd3f0ae104109f264758ce47a2f478d5df13655638497fe576c1874424f02ba0` |
| `src/common/system_model.py` | `e0ccd18e122e3e01d96e097128ad591ab70ef8a2fbb29d980d328597dd36f645` |
| `src/common/project_config.py` | `24b8e3efe64e435731e433452974a112d6a57175af838332e41b2dbe6daa4f6b` |
| `src/decision/cache_configuration.py` | `44a5580c1989d749c5a1b050f81c5b7298b6e28d58efb178dacc5dce7b738cf6` |
| `src/decision/task_allocation.py` | `d785a363aef5c7f24668cfa29274a3859828cd434ce5e18cc60325c5dc3e34e1` |
| `src/decision/slot_environment.py` | `3ef733984050c2674989c3eff9c05fde1ca7c48f6f6f1f83f08ff05e156bdbb1` |
| `src/decision/uav_configuration.py` | `556ee3d7ab2478bef341a6393c3fa6d4ba2142e99618b5aa88cb759454252412` |
| `src/ppo_simple/trainer.py` | `9923281c36ae3d8931f37abb9858376727e6daa4952f3e7137af71914e712f97` |
| `src/ppo_simple/policy.py` | `c17fbd14085ac0cca2a3bc923de12b6865c384978c4b0f8b1dc376213b90a041` |
| `src/ppo_simple/rollout.py` | `d88bbf39a85156a52ee1fbcc266d21664d62fd3a985f573831408a5f1747653f` |
| `src/ppo_simple/interface.py` | `89b6a009edd953d92606d213289a10e9986082026b171863e73c0c45affb5fa3` |
| `src/ppo_simple/macro_targets.py` | `a3b2ad4662a2a2f928519ff854d1bfd7330e888e4e06ed9eed52b401b6b93412` |
| `src/ppo_simple/primitive_control.py` | `3d54e30c2f34902836cb6d66527901e1a5732803a7f840ddffa0311b814599e1` |
| `src/ppo_simple/reporting.py` | `bb3542cdaf1668e67f10ae60be9b35755f4d2b926c5ce4ebff4b7d870dbdc35d` |
| `src/ppo_simple/evaluator.py` | `2044f8e9feecfca9893e6ec662604f4508565a9fcd1de1fbeb98ee354703cb22` |
| `src/scenario/realization.py` | `adcf54c3d871e251d55d88318cb018d2d3dac98dc6b605c50a9f8b8693084806` |

正式包装器`run_proposed_formal.py`和`train_formal.py`未调用；后者的默认配置仍是旧`system_config.toml`。Pilot直接通过通用Trainer显式传入Stage3B TOML。旧Checkpoint `runs/formal_cache_economic_v2_lr_5e_minus_4/checkpoints/latest_checkpoint.pt`未加载。

运行目录：`runs/stage5_limited_ppo_pilot_stage3b_20261011_seed20001_24ep/`。主要产物SHA-256：

| 产物 | SHA-256 |
|---|---|
| `trainer/run_config.toml` | `dd3f0ae104109f264758ce47a2f478d5df13655638497fe576c1874424f02ba0` |
| `trainer/run_manifest.json` | `abe9db25280bcaf9abe87753dae10692d7ffaaeadacb23ee868559198f709819` |
| `trainer/run_summary.json` | `b82bd2d0cabd76fd2aa1c9002f3276227ef38776bf786db0425edc72862537c7` |
| `trainer_console.log` | `7a91b6efbf69107be187dba74d00afb0d1325ce63ce0a157967ef0a36fba2c5e` |
| `trainer/checkpoints/latest_checkpoint.pt` | `5cd22587725e218c61323dda1740d61c2e4e14b40f1952856e23db096b5b50e2` |
| `trainer/checkpoints/best_train_return_checkpoint.pt` | `1366536dad6ad3b49683b4cfe31f2e6e56e1df0345bc5a489efa10826963e967` |
| `initial_policy/checkpoints/latest_checkpoint.pt`（初始策略参考） | `820b1700f543935ba2f5667daa49524e3b9c93c8c74dc4b14ca08f1aab1b2edf` |

训练运行文件、检查点及日志保留。冻结报告中既有dirty/untracked文件未覆盖或清理；仅新增本报告和独立Pilot运行目录，没有提交或推送。

## 启动命令与样本边界

实际入口和命令：

```powershell
conda activate adaptive_cache_mec
$env:PYTHONPATH = "src"
python -m ppo_simple.trainer `
  --config configs/proposed/stage3_cloud_finite_60ghz.toml `
  --epochs 24 `
  --run-dir runs/stage5_limited_ppo_pilot_stage3b_20261011_seed20001_24ep/trainer `
  --episode-pool environment_data/proposed_joint_training `
  --episode-seed-start 20001 `
  --seed 0
```

stdout/stderr写入`runs/stage5_limited_ppo_pilot_stage3b_20261011_seed20001_24ep/trainer_console.log`。没有传入学习率覆盖、resume、缺失Episode生成或旧Checkpoint参数。`--seed 0`控制新策略初始化和训练随机数；Episode realization seed按入口语义连续映射为`20001–20024`。

`--epochs 24`在该Trainer中表示24个完整300-Slot Episode，不是24个槽或24次优化。每4个Episode冻结同一旧策略采样并进行一次PPO更新，因此本次共7200个环境执行、6次策略更新。训练使用Stage3B正式默认学习率：Macro Actor `5e-4`、Micro Actor `3e-4`、Macro Critic `1e-3`、Micro Critic `3e-3`、Long-Value Critic `1e-3`。初始策略为seed 0新初始化；没有恢复旧环境Checkpoint。另保存的`initial_policy`是初始策略参考快照，不作为训练的resume输入。

训练seed与Validation seeds `30001–30004`分离；Test seeds未使用。预生成训练/Validation episode均存在，没有生成数据。训练指标文件记录24条Episode、6条更新、7200条Slot和6条Critic诊断记录。Trainer汇总的采样加更新用时为5734.05秒（约95.6分钟）；其中Matching共7200次、KKT共1,444,691次。Slot平均执行时间约0.685秒，但Logger没有拆出Matching单独耗时，不能把该总时间全部归因于Matching。

## Macro跨Holding周期信用分配

当前实现没有发现确定性的时间信用截断：

- `rollout.py:953`按Slot将Reward折入Macro frame；`rollout.py:1473–1510`的`compute_macro_smdp_gae`从Episode末尾反向递推，使用`gamma^duration`和`(gamma·lambda)^duration`。Stage3B配置为`gamma=1.0`、`gae_lambda=0.95`，Macro实际Holding范围1–5 Slot。它会随时间距离衰减，但没有在单个Holding后硬截断，后续多个Macro周期仍可反馈到Episode内较早动作。
- Macro Actor使用完整Episode的SMDP-GAE Advantage（`trainer.py:1502`附近）。Macro Critic使用两步SMDP TD目标（`trainer.py:80`、`macro_targets.py:13`），目标视野较短；单独的Long-Value Critic以完整`realized_return_to_go`训练（`trainer.py:1578`附近），但其输出不是Macro Actor的直接Advantage。
- 每Slot Reward保持`G=Task Utility−Cache Cost−UAV Movement Cost`。本次Episode级及Slot级恒等式最大绝对误差都为`0`。Cache配置未发生变化时更新成本按既有逻辑为零。

所以算法中存在跨Holding的Actor信用路径，但长距离Lambda衰减和仅6次更新会使稀疏20-Slot收益难以在小样本中学稳。Pilot没有证据证明Stage4B候选#115被识别或被反复拟合。

## 训练采样指标

下表是每次更新前，用当时策略采集的4个训练Episode均值。各组使用不同seed，不能作为同seed前后评估。

| Episode seeds | Return | Task Utility | Cache Cost | UAV Cost | Cache UPDATE率 | UAV0 MOVE率 | UAV1 MOVE率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20001–20004 | 3800.78 | 3813.80 | 12.37 | 0.650 | 46.8% | 44.4% | 47.1% |
| 20005–20008 | 3858.83 | 3872.63 | 13.11 | 0.695 | 50.0% | 47.2% | 50.7% |
| 20009–20012 | 3861.46 | 3875.96 | 13.78 | 0.715 | 49.4% | 49.0% | 50.4% |
| 20013–20016 | 3834.78 | 3847.27 | 11.84 | 0.648 | 45.5% | 45.0% | 46.8% |
| 20017–20020 | 3849.05 | 3861.43 | 11.75 | 0.629 | 45.6% | 41.8% | 48.2% |
| 20021–20024 | 3869.98 | 3882.01 | 11.45 | 0.591 | 48.8% | 39.7% | 44.3% |

全24个Episode平均Return为`3845.81`，范围`3756.75–3958.00`；平均Task Utility为`3858.85`，Cache Cost为`12.38`，UAV成本为`0.655`。共计101,032个任务；按任务加权平均时延`4.0044 s`，车辆能耗`0.03747 J/任务`。平均Holding长度`1.97 Slot`。Cache UPDATE约47.7%、KEEP约52.3%；每Episode平均选中约72.6种不同UPDATE候选，候选Top-1占比均值约1.84%。这说明策略有动作多样性，不能单凭UPDATE频率说明其学会了动作经济价值。

两个UAV的MOVE动作率分别为44.5%和47.9%，平均每Episode UAV移动成本为`0.655`。日志只给出MOVE/STAY动作及总移动成本，不提供实际移动距离和节点服务任务数。

### PPO与Critic更新诊断

| 更新 | Macro KL | Macro Clip | Macro Entropy | Macro Critic EV | Macro Critic Loss | Micro KL | Micro Clip | Micro Entropy | Micro Critic EV | Micro Critic Loss | Long-Value EV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01035 | 0.0809 | 4.914 | 0.005 | 552.14 | 0.00021 | 0.0000 | 4.103 | 0.486 | 25412.93 | 0.004 |
| 2 | 0.00240 | 0.0203 | 4.856 | 0.180 | 421.80 | 0.00423 | 0.0483 | 4.264 | 0.726 | 21690.04 | 0.012 |
| 3 | 0.00333 | 0.0372 | 4.793 | 0.300 | 489.63 | 0.00156 | 0.0004 | 4.277 | 0.758 | 14614.23 | 0.053 |
| 4 | 0.00051 | 0.0000 | 4.740 | 0.423 | 598.51 | 0.00106 | 0.0000 | 4.073 | 0.867 | 18506.29 | 0.500 |
| 5 | 0.00025 | 0.0000 | 4.707 | 0.535 | 727.03 | 0.00667 | 0.1029 | 3.781 | 0.935 | 15446.77 | 0.625 |
| 6 | 0.00134 | 0.0115 | 4.603 | 0.640 | 806.55 | 0.00543 | 0.0406 | 3.901 | 0.940 | 19463.03 | 0.868 |

六次更新均未触发Macro或Micro KL早停；更新及Critic诊断没有NaN/Inf，Actor梯度有限。Macro/Micro Critic的训练内EV逐步升高，Long-Value EV在后半段改善。这证明有限Pilot内优化与目标拟合正常，不等于独立泛化或策略收益得到验证。

Logger没有记录Cloud/Edge节点执行数量、CPU/带宽违规计数或逐节点资源利用率，本报告不从总Utility或KKT调用数推断这些指标。Matching/KKT在本Pilot的总体执行量可确认，但没有相同入口的性能基线，无法判断其耗时是否“显著增加”。

## Validation可执行性与结论边界

固定Validation episode `30001–30004`存在且与训练seed不重叠。初始策略参考快照的评估在开始执行Episode前失败，报错为`ValueError: checkpoint policy tensor/configuration contract mismatch`。具体原因是短Pilot checkpoint的`config_contract.policy_update_total=6`，而Evaluator构造Trainer时使用正式默认值`FORMAL_TRAINING_EPISODES // 4=75`；`trainer.py:1094–1109, 2904`要求契约完全相等。此处没有改写Checkpoint元数据，也没有修改Evaluator。初始/最终同seed策略对照及KEEP/STAY基线均未运行；最终策略的Validation改善为未定。

**判断：**本次属于“训练入口、采样、更新正常，但训练规模太小，尚无法判断学习能力”。训练Return组均值从首组约3801到末组约3870，但这些组使用不同seed且没有成对前后评估，不能作为策略改善证据。没有发现确定的Macro信用实现错误、Actor/Critic数值故障或策略退化证据。Cache更新与UAV移动都被采样，但长期价值学习仍未证实。

下一轮若要作出延长训练的策略判断，最小前置工作是让Evaluator能在不改变策略张量语义的情况下加载短Pilot契约，并完成30001–30004相同输入的训练前后比较。该问题在评估器契约适配，不要求据本次结果修改PPO、Reward、Matching/KKT或Stage3B系统模型。由于验证未执行且只有6次更新，当前是否进入更长训练仍为证据不足。

```makefile
STAGE5_PILOT_DONE = YES
STAGE3B_CONFIG_VERIFIED = YES
SOURCE_HASHES_MATCH_FREEZE = YES
TRAIN_VALIDATION_TEST_ISOLATED = YES; TRAIN=20001-20024; VALIDATION=30001-30004_NOT_EXECUTED; TEST=UNUSED
MACRO_LONG_HORIZON_CREDIT_CHECKED = YES; NO_HARD_CUTOFF_FOUND
PPO_UPDATES_EXECUTED = 6
ACTOR_CRITIC_NUMERICALLY_VALID = YES; NO_NONFINITE_OR_UPDATE_EARLY_STOP
CACHE_POLICY_LEARNING_SIGNAL = UNDETERMINED; ACTION_DIVERSITY_OBSERVED; ECONOMIC_RANKING_UNVERIFIED
UAV_POLICY_LEARNING_SIGNAL = UNDETERMINED; MOVE_ACTIONS_OBSERVED; LONG_TERM_VALUE_UNVERIFIED
VALIDATION_IMPROVEMENT = UNDETERMINED; EVALUATOR_CONTRACT_MISMATCH
IMPLEMENTATION_CHANGE_REQUIRED = YES_FOR_EVALUATOR_SHORT_PILOT_CONTRACT_ONLY
READY_FOR_EXTENDED_TRAINING = UNDETERMINED
```
