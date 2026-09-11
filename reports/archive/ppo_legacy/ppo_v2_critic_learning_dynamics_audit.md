
# Scheme B v2 Critic learning dynamics audit

审计日期：2026-09-09。范围仅为只读源码、正式配置、现有 v2 JSONL 和 checkpoint 安全元数据；本任务没有运行新的 origin、rollout、PPO update 或 Epoch。

## 1. Executive summary

已定位到真实产物目录：

runs/ppo_primitive_v2_diag_20ep_20260909/

目录名虽含 20ep，但内容实际为完整 Epoch 1--100：

- 100 条 episode 记录，每 Epoch 240 transitions；
- 1,500 条 update 记录，每 Epoch 15 个连续 16-origin chunk；
- 24,000 条 critic transition 诊断；
- latest_checkpoint.pt 与 best_train_return_checkpoint.pt 均存在。

主要结论：

1. time_to_go 的实现和输入路径正确，已进入主 observation、shared encoder 及 Cache/UAV conditional context；没有发现新的 value-target 或 GAE 代码错误。
2. 后期 critic 对时间结构响应很弱：Epoch 81--100 中，old_value 与 time_to_go 的 Pearson 相关为 -0.0534，raw_return 与 time_to_go 的相关为 0.2560；origin-mean value slope 为 -1.803，target slope 为 34.835。
3. 所有 1,500 个 update 的 recorded gradient norm 都高于 max_grad_norm=0.5。pre-clip norm 中位数为 138.452，理论 global clip scale 中位数为 0.003611。这是最直接的优化尺度证据。
4. 15 个 rollout chunk 中 14 个是 non-terminal chunk，使用当前 critic 的 bootstrap_value；只有 224:239 使用 terminal bootstrap 0。terminal 信息对更早 chunk 的传播需要跨 Epoch 的 critic 更新，存在 bootstrap fixed-point risk。
5. shared encoder 把 trigger actor、critic、Cache scorer、UAV scorer 放在同一个 global clipping 路径上；但没有 per-head gradient，因此不能把 shared interference 证明为唯一原因。
6. literal 的恒定均值 critic 没有被证明：old_value 仍有非零方差并随训练窗口整体上升；但 origin-mean value 方差远小于 target 的 origin-mean 方差，说明明显的时间/状态结构 under-response。

综合结论为 MIXED：强 global value-scale/clipping 证据加上短 rollout bootstrap 结构风险；shared encoder 是可信干扰路径，但缺少 per-head 证据。建议下一步只评估候选 A（保留 time_to_go，增加 return/value normalization），本轮没有实施。

## 2. 实际产物和正式配置

### 2.1 产物完整性

| 项目 | 结果 |
|---|---:|
| episode records | 100 |
| transitions / Epoch | 240 |
| updates / Epoch | 15 |
| transitions / update | 16 |
| non-terminal updates | 1,400 |
| terminal updates | 100 |
| critic diagnostics | 24,000 |
| seed | 4401 |
| best epoch | 99 |
| best epoch return | 1555.882689 |
| Epoch 100 return | 917.973024 |
| latest checkpoint | 存在 |
| best-train-return checkpoint | 存在 |

实际 update chunk 顺序为：
0:15, 16:31, 32:47, ..., 208:223, 224:239。

### 2.2 源码和 v2 配置

读取 system_config_ppo_primitive_v2.toml 与 src/ppo_simple/trainer.py：

~~~ini
GAMMA = 1.0
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_LOSS_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.0
LEARNING_RATE = 3e-4
MAX_GRAD_NORM = 0.5
ROLLOUT_STEPS = 16
UPDATE_EPOCHS = 1
NUM_MINIBATCHES = 4
OPTIMIZER = Adam
ACTOR_CRITIC_SHARE_OPTIMIZER = YES
ACTOR_CRITIC_SHARE_LR = YES
~~~

rollout_steps=16 也由实际日志独立确认：每个 update 的 num_transitions=16，每 Epoch 有 15 个 update；每个 update 完成 4 个 minibatch optimizer steps。

运行目录没有保存独立配置副本。其 v2-only critic_diagnostics、100 Epoch 记录和 v2 best-checkpoint 机制与当前 v2 配置/源码一致；但由于 checkpoint 安全加载被拒绝，不能用 checkpoint 内 config hash 做额外绑定证明。

## 3. Exact value-loss mathematics

PPOV2Trainer.update() 的每个 minibatch 计算为：

~~~text
raw_value_error_i = (V_theta(s_i) - return_i)^2
reported_value_loss = 0.5 * mean_i(raw_value_error_i)
total_loss = policy_loss
           + value_loss_coefficient * reported_value_loss
           - entropy_coefficient * entropy
~~~

代入当前配置：

~~~ini
RAW_VALUE_ERROR_FORMULA = (V - return)^2
REPORTED_VALUE_LOSS_FORMULA = 0.5 * mean((V - return)^2)
TOTAL_LOSS_FORMULA = policy_loss + 0.5 * reported_value_loss - 0.0 * entropy
EFFECTIVE_MSE_COEFFICIENT = 0.25
~~~

日志中的 value_loss 不是纯 MSE，而是 0.5*MSE；critic 在 total_loss 中的有效平方误差系数为 0.25。GAE returns 没有 value normalization；代码只对 policy advantage 标准化，returns_t 直接进入 value loss。

## 4. Network/shared-encoder architecture

v2 observation dimension 为 155。实际路径：

~~~text
155
 -> Linear(155,128) -> Tanh
 -> Linear(128,128) -> Tanh       shared encoder
 -> Linear(128,action_dim)        Bernoulli trigger actor
 -> Linear(128,1) -> squeeze       critic
~~~

AdaptivePrimitiveV2ActorCritic 继承 Scheme B primitive actor-critic，并增加：

~~~text
shared hidden + Cache candidate features
 -> Linear(128+feature_dim,64) -> Tanh -> Linear(64,1)  Cache scorer

shared hidden + UAV candidate features
 -> Linear(128+feature_dim,64) -> Tanh -> Linear(64,1)  UAV scorer
~~~

~~~ini
VALUE_HEAD_OUTPUT_LINEAR = YES
VALUE_OUTPUT_BOUNDED = NO
TIME_TO_GO_REACHES_SHARED_ENCODER = YES
TIME_TO_GO_VALUE_PRESERVED = YES
~~~

time_to_go=(239-origin)/239 在 rollout._observation() 和 primitive _intermediate_observation() 中构造，build_configuration_observation() 将其作为最后一个 feature；policy 只将输入转为 float32，没有覆盖或截断。critic 是 Linear(128,1)，无 sigmoid/tanh 或其他 bounded output，也无 normalization layer。

trigger actor、critic、Cache scorer、UAV scorer 共享 encoder 参数；各自最后 head 参数独立。所有 policy parameters 由同一个 Adam、同一个 learning rate、同一次 clip_grad_norm_ 管理。

## 5. Time-to-go utilization analysis

value_residual 按日志定义为 old_value - raw_return；raw_return 与 value_target 相同。相关系数使用每条 transition 的 origin 对应 time_to_go；slope 使用每个 window 的 origin mean。

### 5.1 Window-level diagnostics

| Epoch window | n | mean old | mean raw return | mean residual | mean abs residual | median abs residual | std old | std target | std origin-mean old | std origin-mean target | r(old,togo) | r(target,togo) | slope old/togo | slope target/togo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1--20 | 4800 | -4.991 | -12.864 | 7.873 | 27.533 | 19.326 | 3.112 | 39.720 | 0.274 | 15.506 | 0.082 | 0.140 | 0.885 | 19.228 |
| 21--40 | 4800 | 7.285 | 11.477 | -4.191 | 24.857 | 19.296 | 6.119 | 34.384 | 0.111 | 12.322 | -0.013 | 0.218 | -0.276 | 25.896 |
| 41--60 | 4800 | 27.690 | 42.052 | -14.362 | 29.932 | 25.040 | 7.352 | 37.243 | 0.355 | 14.870 | -0.048 | 0.216 | -1.217 | 27.810 |
| 61--80 | 4800 | 57.235 | 81.830 | -24.595 | 33.974 | 31.113 | 9.971 | 35.236 | 0.506 | 20.646 | -0.051 | 0.247 | -1.742 | 30.062 |
| 81--100 | 4800 | 91.188 | 119.650 | -28.463 | 39.631 | 37.552 | 9.795 | 39.445 | 0.524 | 25.812 | -0.053 | 0.256 | -1.803 | 34.835 |

raw_return 的 time-to-go slope 在每个 window 都为正，而后期 critic slope 略为负，late value correlations 均接近零。这支持：

~~~ini
CRITIC_TIME_SENSITIVITY_WEAK = YES
~~~

这不证明每个个体状态都完全忽略 time_to_go；critic 还接收 cache/UAV/vehicle/popularity state，且本审计没有做 per-feature attribution。

### 5.2 规定 origin points，跨全部 100 Epoch

| origin | time_to_go | n | mean old | mean raw return | mean residual | mean abs residual | median abs residual |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.000000 | 100 | 35.379 | 84.489 | -49.109 | 49.351 | 55.951 |
| 15 | 0.937238 | 100 | 35.277 | 37.615 | -2.337 | 7.516 | 7.452 |
| 16 | 0.933054 | 100 | 35.330 | 67.744 | -32.415 | 40.178 | 39.110 |
| 31 | 0.870293 | 100 | 35.277 | 37.160 | -1.883 | 5.147 | 4.810 |
| 32 | 0.866109 | 100 | 35.362 | 62.282 | -26.920 | 40.871 | 42.046 |
| 63 | 0.736402 | 100 | 35.431 | 40.250 | -4.819 | 8.317 | 8.155 |
| 95 | 0.602510 | 100 | 35.581 | 39.619 | -4.038 | 9.031 | 8.278 |
| 127 | 0.468619 | 100 | 35.709 | 38.178 | -2.469 | 7.269 | 7.377 |
| 159 | 0.334728 | 100 | 35.800 | 33.169 | 2.631 | 10.001 | 6.542 |
| 191 | 0.200837 | 100 | 35.880 | 39.500 | -3.620 | 7.455 | 7.971 |
| 207 | 0.133891 | 100 | 35.940 | 37.458 | -1.518 | 8.586 | 7.425 |
| 223 | 0.066946 | 100 | 36.001 | 36.397 | -0.396 | 6.496 | 5.780 |
| 224 | 0.062762 | 100 | 36.048 | 27.126 | 8.922 | 44.017 | 42.046 |
| 230 | 0.037657 | 100 | 36.076 | 17.303 | 18.773 | 37.678 | 25.972 |
| 235 | 0.016736 | 100 | 36.089 | 11.095 | 24.994 | 32.215 | 27.436 |
| 238 | 0.004184 | 100 | 36.085 | -2.024 | 38.109 | 41.245 | 35.368 |
| 239 | 0.000000 | 100 | 36.071 | 1.993 | 34.077 | 36.819 | 27.338 |

Fixed-origin mean critic output stays around 35--36 across the 100 Epoch aggregate while terminal-near targets fall sharply. This supports weak temporal sensitivity, not an exact constant output.

## 6. Bootstrap/rollout propagation analysis

The production sequence is correct:

~~~text
collect 16 transitions -> optional one-step bootstrap observation
-> compute GAE -> one PPO update
~~~

For non-terminal chunks, the last transition uses bootstrap_value=V(next state) in the final GAE delta. The only terminal chunk is 224:239; its final done=True masks next value and uses bootstrap 0. Existing diagnostics confirm 1,400 nonzero-bootstrap chunks and 100 zero-bootstrap terminal chunks.

With gamma=1 and lambda=0.95, a downstream delta is carried backward with factor:

~~~text
(gamma * lambda)^k = 0.95^k
~~~

| distance k | 0 | 1 | 2 | 4 | 8 | 15 |
|---:|---:|---:|---:|---:|---:|---:|
| factor | 1.000000 | 0.950000 | 0.902500 | 0.814506 | 0.663420 | 0.463291 |

Terminal information is not directly inserted into chunks 0:15 through 208:223. Those chunks bootstrap from the critic at their next state. A terminal signal can influence earlier chunks only after the critic trained on later chunks changes its next-state value, then later Epochs reuse that changed value. Cross-Epoch propagation is therefore structurally required.

~~~ini
SHORT_ROLLOUT_BOOTSTRAP_DOMINANT = YES
TERMINAL_SIGNAL_PROPAGATION_ACROSS_EPOCHS_REQUIRED = YES
BOOTSTRAP_FIXED_POINT_RISK = YES
~~~

“Dominant” means target construction: 93.33% of recorded updates are non-terminal and bootstrap-based. It is not a proof that bootstrap is the sole performance cause. Fixed-point risk is supported by nearly flat origin-mean old_value versus strongly varying origin-mean target, together with repeated reuse of current V(next state).

## 7. Gradient clipping analysis

Source path:

~~~text
total_loss.backward()
-> torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
-> optimizer.step()
~~~

PyTorch clip_grad_norm_ returns total norm before clipping; the stored gradient_norm is therefore pre-clip.

### 7.1 All 1,500 updates

| metric | value |
|---|---:|
| gradient_norm mean | 153.518 |
| median | 138.452 |
| p90 | 294.878 |
| p95 | 334.647 |
| max | 1060.180 |
| gradient_norm > 0.5 | 1500 / 1500 |
| active rate | 100.000% |
| theoretical scale mean | 0.008378 |
| theoretical scale median | 0.003611 |
| theoretical scale p10 | 0.001696 |
| theoretical scale min | 0.000472 |

The minimum recorded per-update minibatch norm was 1.45756, so clipping is not only caused by the last-minibatch field. Per requested criterion:

~~~ini
GRADIENT_NORM_LOG_MEANING = PRE_CLIP
GRADIENT_CLIPPING_ACTIVE_RATE = 100.000%
MEAN_EFFECTIVE_CLIP_SCALE = 0.008378
SEVERE_GLOBAL_CLIPPING = YES
~~~

The logged PPO clip_fraction is ratio clipping, not global gradient clipping; its mean was 0.012833.

### 7.2 Window trend

| Epoch window | gradient mean | median | p90 | max | active rate | scale mean | scale median | scale p10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1--20 | 101.982 | 77.608 | 187.622 | 1060.180 | 100% | 0.012487 | 0.006443 | 0.002665 |
| 21--40 | 104.931 | 84.842 | 199.055 | 735.942 | 100% | 0.011452 | 0.005893 | 0.002512 |
| 41--60 | 159.288 | 155.714 | 275.996 | 760.921 | 100% | 0.006681 | 0.003211 | 0.001812 |
| 61--80 | 183.911 | 188.031 | 315.324 | 554.961 | 100% | 0.006575 | 0.002659 | 0.001586 |
| 81--100 | 217.479 | 219.777 | 328.259 | 832.157 | 100% | 0.004693 | 0.002275 | 0.001524 |

Clipping becomes more severe in later windows, coincident with increasing value scale and residual burden. This is strong evidence for an optimization-scale problem, not proof of which head generated the norm.

## 8. Shared actor-critic gradient interaction

One total_loss is differentiated once, then all policy parameters are clipped together. The set includes shared encoder, trigger actor, critic head, Cache scorer and UAV scorer.

~~~ini
GLOBAL_CLIP_APPLIES_TO_ALL_HEADS = YES
VALUE_GRADIENT_CAN_SCALE_ACTOR_UPDATE = YES
ACTOR_GRADIENT_CAN_SCALE_CRITIC_UPDATE = YES
PER_HEAD_GRADIENT_DATA_AVAILABLE = NO
SHARED_ENCODER_INTERFERENCE_PLAUSIBLE = YES
~~~

The first three are source-confirmed topology statements. The last is an inference: large critic gradients can consume the shared global norm budget, and shared encoder gradients can receive actor/conditional-head directions. JSONL has no per-head gradient norms or per-head parameter deltas, so no actor/critic gradient ratio is claimed.

## 9. Value variance and possible mean collapse

| Epoch window | std(old_value) | std(raw_return) | std(origin means old) | std(origin means target) |
|---|---:|---:|---:|---:|
| 1--20 | 3.112 | 39.720 | 0.274 | 15.506 |
| 21--40 | 6.119 | 34.384 | 0.111 | 12.322 |
| 41--60 | 7.352 | 37.243 | 0.355 | 14.870 |
| 61--80 | 9.971 | 35.236 | 0.506 | 20.646 |
| 81--100 | 9.795 | 39.445 | 0.524 | 25.812 |

The output is not exact constant: it has nonzero transition variance and its level rises across windows. However, origin-mean output variance remains much smaller than target variance, while target origin means retain a positive time-to-go slope.

~~~ini
CRITIC_OUTPUT_VARIANCE = nonzero; 3.112 -> 9.795 by window
TARGET_VARIANCE = 34.384 -> 39.720 by window
CRITIC_ORIGIN_MEAN_VARIANCE = 0.111 -> 0.524 by window
TARGET_ORIGIN_MEAN_VARIANCE = 12.322 -> 25.812 by window
CRITIC_COLLAPSED_TO_MEAN_VALUE = NO
~~~

Strict constant-output collapse was not established; weak origin/time sensitivity was.

## 10. Checkpoint value-head inspection

Both checkpoints were opened in PyTorch safe mode only:

~~~python
torch.load(path, map_location="cpu", weights_only=True)
~~~

Both failed with UnpicklingError because the serialized RNG state references unsupported global numpy._core.multiarray._reconstruct. Unsafe weights_only=False loading and safe-global allowlisting were not attempted.

~~~ini
CHECKPOINT_PARAMETER_INSPECTION_BLOCKED = YES
VALUE_HEAD_WEIGHT_NORM_BEST = NOT VERIFIED
VALUE_HEAD_WEIGHT_NORM_LATEST = NOT VERIFIED
VALUE_HEAD_BIAS_BEST = NOT VERIFIED
VALUE_HEAD_BIAS_LATEST = NOT VERIFIED
~~~

Best/latest artifact facts remain verified from JSONL and file presence: best return was Epoch 99; latest completed Epoch 100 has lower return.

## 11. Best Epoch versus late Epoch

Best episode-return record is Epoch 99. Since the run ends at 100, best ±5 is Epochs 94--100.

注：trainer 在每个 minibatch 覆盖一次 UpdateResult.value_loss，因此下表的 terminal/non-terminal value loss 使用 JSONL 中该字段（每个 update 的最后 minibatch 值）；每个 update 的 min/max 字段仍覆盖其 4 个 minibatch。gradient norm 同样是最后 minibatch 的记录值，前文已同时报告其 update 内 min/max。

| group | episodes | mean return | terminal value loss | non-terminal value loss | mean abs residual | terminal abs residual | non-terminal abs residual | terminal gradient | non-terminal gradient | UAV move rate | Cache change rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best Epoch 99 | 1 | 1555.883 | 140.780 | 1138.110 | 42.140 | 101.103 | 41.893 | 34.542 | 235.036 | 0.491667 | 0.333333 |
| best ±5, 94--100 | 7 | 1347.620 | 2569.840 | 1134.390 | 42.756 | 103.787 | 42.500 | 300.344 | 225.146 | 0.488690 | 0.329563 |
| late, 81--100 | 20 | 1192.000 | 1829.040 | 1086.780 | 39.631 | 88.918 | 39.425 | 244.032 | 215.582 | 0.468021 | 0.354259 |

Return improved from the first 20 Epochs and reached its best at 99; there is no monotonic late return degradation. Terminal value-loss and gradient spikes exist, but their aggregate movement does not line up monotonically with return.

~~~ini
CRITIC_DEGRADATION_CORRELATES_WITH_RETURN_DEGRADATION = UNCERTAIN
~~~

This is not a causation claim.

## 12. UAV movement versus time-to-go

The configuration has two UAV instances. Episode metrics aggregate all 240 origins, so trigger rate below is reconstructed as uav_request_count/(240*2); uav_actual_movement_rate is directly logged.

| Epoch window | trigger rate | actual move rate | move \| trigger |
|---|---:|---:|---:|
| 1--20 | 0.542708 | 0.535000 | 0.985312 |
| 21--40 | 0.394271 | 0.385729 | 0.977333 |
| 41--60 | 0.362500 | 0.358646 | 0.989307 |
| 61--80 | 0.426354 | 0.420729 | 0.986535 |
| 81--100 | 0.474896 | 0.468021 | 0.985483 |

No retained artifact has per-transition or per-origin UAV action/movement. Therefore origin segments 0--59, 60--119, 120--179, 180--239 cannot be reconstructed without a new rollout, prohibited here. No claim that early movement exceeds late movement is justified.

~~~ini
UAV_MOVE_TIME_DEPENDENCE = UNCERTAIN
UAV_NEAR_TERMINAL_STAY_INCREASED = UNCERTAIN
~~~

## 13. Gamma analysis

The environment reward is checked against execute_slot().system_utility, and the PPO objective is the finite-horizon sum of G(t). With that objective, gamma=1 is correct undiscounted temporal weighting.

~~~ini
GAMMA_ONE_OBJECTIVE_CONSISTENT = YES
GAMMA_CHANGE_REQUIRED_NOW = NO
~~~

gamma<1 would change the optimization objective by discounting later G(t); it is not a direct repair for the observed critic scale/bootstrap evidence.

## 14. Ranked candidate fixes (audit only)

| rank | candidate | targets evidence? | changes system objective? | changes PPO objective/optimization? | checkpoint compatibility | new training? | complexity | assessment |
|---:|---|---|---|---|---|---|---|---|
| 1 | A. Keep time-to-go; return/value normalization | Yes: raw target scale and clipping | No; G unchanged | Yes: critic target/loss parameterization | Not an equivalent continuation without normalization metadata | Yes | Medium | Most direct first experiment |
| 2 | B. Independent critic optimizer/LR | Yes: separates critic pressure | No | Yes | Optimizer semantics change; new run | Yes | Medium/high with shared encoder | Direct but parameter ownership must be designed |
| 3 | C. Separate actor/critic clipping | Yes: tests global clipping bottleneck | No | Yes | Shared encoder needs explicit gradient policy | Yes | High | Relevant but not simple with shared encoder |
| 4 | F. Increase rollout horizon / delay update | Yes: addresses bootstrap targets | No | Yes: rollout/GAE semantics | Not comparable to this trajectory | Yes | Medium/high | Structural test after scale issue |
| 5 | E. Increase critic update epochs | Partial: does not remove clipping | No | Yes | New training provenance | Yes | Low/medium | May add compute without fixing clipping |
| 6 | G. Huber critic loss | Partial: may reduce outlier effect | No | Yes | New critic objective | Yes | Low/medium | Robustness experiment, not first diagnosis |
| 7 | D. Separate actor/critic encoders | Plausible interference, not proven dominant | No | Yes; architecture changes | Incompatible tensors | Yes | High | Larger intervention |
| 8 | I. Do not change | No: clipping/time weakness remains | No | No | Yes | No | None | Not recommended for blind continuation |
| 9 | H. Change gamma | No: gamma=1 matches objective | Yes | Yes | New objective semantics | Yes | Low | Not a critic bug fix |

None was implemented.

## 15. Recommended single next change

~~~ini
RECOMMENDED_NEXT_CHANGE = A
~~~

Keep v2 time_to_go and test a precisely specified return/value normalization scheme in a fresh, separately identified run. It directly addresses raw-return scale and severe global clipping while leaving environment reward G(t), Cache/UAV actions, Matching and KKT unchanged. It changes critic optimization, so it must not be presented as a transparent continuation of the current checkpoint without an explicit compatibility rule.

## 16. Required conclusion fields

~~~ini
TIME_TO_GO_IMPLEMENTATION_CORRECT = YES
CRITIC_TIME_SENSITIVITY_WEAK = YES

VALUE_HEAD_OUTPUT_LINEAR = YES
VALUE_OUTPUT_BOUNDED = NO

SHORT_ROLLOUT_BOOTSTRAP_DOMINANT = YES
TERMINAL_SIGNAL_PROPAGATION_ACROSS_EPOCHS_REQUIRED = YES
BOOTSTRAP_FIXED_POINT_RISK = YES

GRADIENT_NORM_LOG_MEANING = PRE_CLIP
GRADIENT_CLIPPING_ACTIVE_RATE = 100.000%
MEAN_EFFECTIVE_CLIP_SCALE = 0.008378
SEVERE_GLOBAL_CLIPPING = YES

GLOBAL_CLIP_APPLIES_TO_ALL_HEADS = YES
PER_HEAD_GRADIENT_DATA_AVAILABLE = NO

CRITIC_COLLAPSED_TO_MEAN_VALUE = NO
SHARED_ENCODER_INTERFERENCE_PLAUSIBLE = YES

GAMMA_ONE_OBJECTIVE_CONSISTENT = YES
GAMMA_CHANGE_REQUIRED_NOW = NO

VALUE_TARGET_IMPLEMENTATION_BUG = NO
NEW_CODE_BUG_FOUND = NO

ROOT_CAUSE = MIXED

RECOMMENDED_NEXT_CHANGE = A

CONTINUE_TRAINING_CURRENT_V2_WITHOUT_CHANGE = NO
~~~

CRITIC_COLLAPSED_TO_MEAN_VALUE=NO 表示严格恒定输出未被证明；报告证明的是 origin/time sensitivity 弱。SHORT_ROLLOUT_BOOTSTRAP_DOMINANT=YES 与 BOOTSTRAP_FIXED_POINT_RISK=YES 表示结构性风险和数据支持的风险，不表示唯一因果证明。

## 17. Scope, cleaning, and repository checks

- 本审计没有修改生产源码或配置。
- 没有运行新的 environment、rollout、PPO update、Epoch、benchmark 或测试套件。
- 没有使用不安全 checkpoint 反序列化。
- 没有保留临时脚本、CSV、JSON、图表、__pycache__ 或 .pyc。
- 已保留此前任务已有的 working-tree changes。
- git diff --check：PASS；Git 仅报告已有文件的 LF/CRLF 警告，没有 whitespace error。


