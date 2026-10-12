# Stage5A：Evaluator契约修复与配对Validation

## 结论

Evaluator原先按正式训练默认值构造`PPOTrainer`，将短Pilot Checkpoint的`policy_update_total=6`误配成75，导致完整配置合同校验在评估Episode开始前失败。现已由Evaluator读取Checkpoint合同中的正整数并据此构造Trainer；Checkpoint合同、架构和张量形状校验仍然启用。初始与最终策略均已在相同的Stage3B Validation Episode上完成确定性评估。

四个seed的平均配对Return差为`+0.128273`（3正、1负），约为初始平均Return的`0.0033%`。最终策略平均Task Utility低`0.115291`，但平均Cache成本少`0.243564`，因此净Return仅小幅增加。两种策略的两架UAV在全部Episode都选择STAY；本次没有观察到Cache或UAV长期动作价值改善的证据。

## 契约不匹配与最小修复

Stage5 Pilot每4个Episode执行一次PPO更新，24 Episode对应6次策略更新。初始参考Checkpoint和训练结束Checkpoint都保存了`config_contract.policy_update_total=6`，且两者的完整`config_contract`相同；策略Schema、算法标识、架构版本、系统配置合同及策略张量形状也相符。失败来自Evaluator没有把该字段传给`PPOTrainer`：Trainer构造器默认按`FORMAL_TRAINING_EPISODES // 4 = 300 // 4 = 75`设置更新总数，`load_checkpoint()`随后按完整合同严格比较，因6与75不同而拒绝加载。

本次只在`src/ppo_simple/evaluator.py`修正评估构造路径：读取并验证Checkpoint中的`policy_update_total`为正整数，再将其传给`PPOTrainer`。Checkpoint加载时现有的合同、架构与张量校验未删除或放宽；没有改写Checkpoint元数据。正式300 Episode Checkpoint若合同值为75，Evaluator仍以75构造Trainer，原有行为保持一致。`trainer.py`及Actor/Critic训练逻辑均未修改。

## 配对设置与文件版本

- 配置：`configs/proposed/stage3_cloud_finite_60ghz.toml`，原始SHA-256为`dd3f0ae104109f264758ce47a2f478d5df13655638497fe576c1874424f02ba0`，语义SHA-256为`d69c6416adb3d1356ea0f233d963ba4f25d66c2eb0fd733798bd1684ef327d75`。
- Validation数据：已有`environment_data/proposed_joint_validation/episode_30001`至`episode_30004`。初始和最终策略读取相同目录、相同四个Episode，每个Episode评估一次。
- 动作选择：两边均使用`deterministic`，CPU推理，`policy_sampling_seed=0`。两种策略使用独立输出目录。
- 没有运行训练、反向传播或Optimizer Step。Evaluator使用`torch.inference_mode()`，并在完成后检查策略参数未变且Optimizer步数未增加。
- 未使用最终Test seeds。简单KEEP/STAY策略不属于本次比较；现有Evaluator没有不改策略核心即可直接运行的该基准。

实际命令：

```powershell
conda activate adaptive_cache_mec
$env:PYTHONPATH = "src"
python -m ppo_simple.evaluator `
  --config configs/proposed/stage3_cloud_finite_60ghz.toml `
  --checkpoint runs/stage5_limited_ppo_pilot_stage3b_20261011_seed20001_24ep/initial_policy/checkpoints/latest_checkpoint.pt `
  --run-dir runs/stage5a_paired_validation_20261011/initial_policy `
  --episode-pool environment_data/proposed_joint_validation `
  --seed-start 30001 --episodes 4 --repeats 1 `
  --device cpu --policy-action-mode deterministic --policy-sampling-seed 0

python -m ppo_simple.evaluator `
  --config configs/proposed/stage3_cloud_finite_60ghz.toml `
  --checkpoint runs/stage5_limited_ppo_pilot_stage3b_20261011_seed20001_24ep/trainer/checkpoints/latest_checkpoint.pt `
  --run-dir runs/stage5a_paired_validation_20261011/final_policy `
  --episode-pool environment_data/proposed_joint_validation `
  --seed-start 30001 --episodes 4 --repeats 1 `
  --device cpu --policy-action-mode deterministic --policy-sampling-seed 0
```

运行结果保存在`runs/stage5a_paired_validation_20261011/initial_policy/`和`runs/stage5a_paired_validation_20261011/final_policy/`，每边包含四个逐Episode结果和汇总JSON。两份汇总记录的配置哈希、数据池路径、seed范围、重复次数和动作模式一致。逐seed任务数完全相同。

| 文件 | Stage5A前/冻结记录SHA-256 | Stage5A后SHA-256 |
|---|---|---|
| Stage3B TOML | `dd3f0ae104109f264758ce47a2f478d5df13655638497fe576c1874424f02ba0` | `dd3f0ae104109f264758ce47a2f478d5df13655638497fe576c1874424f02ba0` |
| `src/common/system_model.py` | `e0ccd18e122e3e01d96e097128ad591ab70ef8a2fbb29d980d328597dd36f645` | 同左 |
| `src/decision/task_allocation.py` | `d785a363aef5c7f24668cfa29274a3859828cd434ce5e18cc60325c5dc3e34e1` | 同左 |
| `src/decision/slot_environment.py` | `3ef733984050c2674989c3eff9c05fde1ca7c48f6f1f83f08ff05e156bdbb1` | 同左 |
| `src/ppo_simple/trainer.py` | `9923281c36ae3d8931f37abb9858376727e6daa4952f3e7137af71914e712f97` | 同左 |
| `src/ppo_simple/policy.py` | `c17fbd14085ac0cca2a3bc923de12b6865c384978c4b0f8b1dc376213b90a041` | 同左 |
| `src/ppo_simple/evaluator.py` | `2044f8e9feecfca9893e6ec662604f4508565a9fcd1de1fbeb98ee354703cb22` | `60ce71d14a12275b9c4b535cb4115242e636689a38cab7dd5e77fb904dc79d10` |
| 初始策略Checkpoint | `820b1700f543935ba2f5667daa49524e3b9c93c8c74dc4b14ca08f1aab1b2edf` | 同左 |
| 最终策略Checkpoint | `5cd22587725e218c61323dda1740d61c2e4e14b40f1952856e23db096b5b50e2` | 同左 |

工作区在Stage5A之前已有未提交改动；没有清理或恢复这些文件。上表Checkpoint哈希与Stage5报告记录相同，评估后再次计算也相同。Stage5A源代码修改仅限Evaluator兼容短Pilot合同；Stage3B配置、系统模型、Matching/KKT、Trainer和策略网络文件均未改变。

## 配对Validation结果

`Task Utility`列为Evaluator的`total_q`。Return恒等式为`Return = Task Utility − Cache Cost − UAV Movement Cost`；逐Episode残差不超过`1e-12`。

| Seed | 初始Return | 最终Return | ΔReturn | 初始Task Utility | 最终Task Utility | ΔTask Utility | Cache Cost 初始→最终 | UAV Cost 初始→最终 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30001 | 3982.519525 | 3982.591733 | +0.072208 | 3982.763089 | 3982.591733 | −0.171356 | 0.243564 → 0 | 0 → 0 |
| 30002 | 3782.921306 | 3782.982705 | +0.061399 | 3783.164870 | 3782.982705 | −0.182165 | 0.243564 → 0 | 0 → 0 |
| 30003 | 3893.278338 | 3893.167095 | −0.111243 | 3893.521902 | 3893.167095 | −0.354807 | 0.243564 → 0 | 0 → 0 |
| 30004 | 3962.699379 | 3963.190108 | +0.490728 | 3962.942943 | 3963.190108 | +0.247165 | 0.243564 → 0 | 0 → 0 |
| **四seed平均** | **3905.354637** | **3905.482910** | **+0.128273** | **3905.598201** | **3905.482910** | **−0.115291** | **0.243564 → 0** | **0 → 0** |

| Seed | 初始Cache KEEP / UPDATE | 最终Cache KEEP / UPDATE | 初始UAV0 / UAV1 | 最终UAV0 / UAV1 | 总任务数（两边相同） |
|---:|---:|---:|---:|---:|---:|
| 30001 | 99.609% / 0.391% | 100% / 0% | STAY 100% / STAY 100% | STAY 100% / STAY 100% | 4302 |
| 30002 | 99.667% / 0.333% | 100% / 0% | STAY 100% / STAY 100% | STAY 100% / STAY 100% | 4196 |
| 30003 | 99.667% / 0.333% | 100% / 0% | STAY 100% / STAY 100% | STAY 100% / STAY 100% | 4255 |
| 30004 | 99.643% / 0.357% | 100% / 0% | STAY 100% / STAY 100% | STAY 100% / STAY 100% | 4225 |

初始/最终策略的Return差主要来自成本项：平均Task Utility下降`0.115291`，平均Cache成本减少`0.243564`，两者合成平均Return增加`0.128273`。这表明当前确定性最终策略通过不执行Cache UPDATE节省了更新成本；不能把这点小幅Return增加解释为Cache更新经济价值被学会。两种策略均未移动UAV，故没有UAV MOVE价值改善证据。

Evaluator现有结果不直接输出Cloud/Edge节点任务数或资源约束可行性，因此本报告不推断这些指标。四个Validation Episode只能描述这组配对样本，不能证明稳定改善或收敛。

## 判断

- Evaluator与短Pilot合同不匹配已修复，且原严格合同检查保留。
- 初始与最终策略配对Validation已完成；Checkpoint未变，系统模型与PPO训练实现未变。
- 只观察到非常小的平均Return上升，且来源是成本下降、Task Utility反而下降。没有证据表明Cache更新价值或UAV移动价值得到改善。
- 四个seed和当前动作分布不足以决定是否值得启动更长训练；建议将扩训决策标记为未定，而不是把这次成本节省当作学习成功。

```makefile
STAGE5A_DONE = YES
EVALUATOR_CONTRACT_FIXED = YES
CHECKPOINT_UNMODIFIED = YES
PPO_CORE_UNCHANGED = YES
SYSTEM_MODEL_UNCHANGED = YES
PAIRED_VALIDATION_COMPLETED = YES
VALIDATION_DELTA_RETURN = +0.128273_MEAN; 3_OF_4_POSITIVE
CACHE_POLICY_VALUE_IMPROVEMENT = NO_EVIDENCE
UAV_POLICY_VALUE_IMPROVEMENT = NO_EVIDENCE
READY_FOR_EXTENDED_TRAINING = UNDETERMINED
```
