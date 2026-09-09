
# Scheme B v2 Critic residual-scale normalization 实验版本

日期：2026-09-09  
项目：C:/Users/tangning/Desktop/code/adaptive_cache_mec

## 1. 为什么做这项修改

依据 reports/ppo_v2_critic_learning_dynamics_audit.md，当前普通 v2 的 time_to_go、GAE、raw return 和 bootstrap 实现正确，但 100 Epoch 现有日志显示：

- 1,500/1,500 个 update 的 pre-clip global gradient norm 都超过 max_grad_norm=0.5；
- gradient norm 中位数为 138.452；
- 理论 global clipping scale 中位数为 0.003611；
- raw return/value residual 的尺度随训练明显增大。

因此本实验只隔离一个假设：critic raw residual 的尺度是否使现有 global clipping 长期严重触发。

本任务没有运行 environment、rollout、PPO update 或长训练。

## 2. 实施内容

新增独立配置：

system_config_ppo_primitive_v2_value_norm.toml

配置：

~~~ini
critic_value_scale_mode = "running_return_std"
critic_value_scale_epsilon = 1.0e-8
critic_value_scale_warmup_samples = 64
~~~

生产实现位于：

- src/ppo_simple/trainer.py
- src/common/project_config.py

现有 system_config_ppo_primitive_v2.toml 没有被覆盖或修改。该文件缺少新字段时使用兼容默认值 critic_value_scale_mode=none，因此普通 v2 保持原行为。policy、rollout、environment、reward、Cache/UAV optimizer、Matching/KKT 没有修改。

## 3. 精确数学公式

raw value、raw return 和 bootstrap 全部保持原始 system utility/value 单位：

~~~text
V_raw = V_theta(s)
R_raw = raw_return = A_raw + V_old(s)
bootstrap_raw = V_raw(next_state)
~~~

normalized mode 只改变 critic squared error 的输入 residual：

~~~text
residual_raw = V_raw - R_raw
normalized_value_residual = residual_raw / scale
reported_value_loss =
    0.5 * mean(normalized_value_residual^2)

total_loss =
    policy_loss
    + value_loss_coefficient * reported_value_loss
    - entropy_coefficient * entropy
~~~

当前 value_loss_coefficient=0.5、entropy_coefficient=0 时，critic 的有效 contribution 为：

~~~text
0.25 * mean(((V_raw - R_raw) / scale)^2)
~~~

V_raw 没有归一化，R_raw 没有归一化，bootstrap_raw 没有缩放。actor advantage 的现有标准化、PPO ratio 和 clip_epsilon 没有改变。

## 4. running statistics 实现

新增 RunningReturnStatistics，保存：

~~~text
count
mean
m2
~~~

使用 population Welford 更新：

~~~text
for x:
    count <- count + 1
    delta <- x - mean
    mean <- mean + delta / count
    delta_after <- x - mean
    m2 <- m2 + delta * delta_after

variance = m2 / count
scale = sqrt(variance + epsilon)
~~~

所有计算使用 float64 的统计状态，输入 raw returns 要求 non-empty 且 finite。

warm-up 和保护规则：

1. 默认 warmup_samples=64；
2. count < max(2,warmup_samples) 时 scale=1.0；
3. variance 非 finite 或 variance <= epsilon 时 scale=1.0；
4. 其他情况使用 sqrt(m2/count + epsilon)；
5. scale 必须为正且 finite，否则回退到 1.0。

因此新实验的前 4 个 16-transition update 使用 scale=1.0；第 4 个 update 完成后统计 count=64，第 5 个 update 才可能使用历史 running standard deviation。

## 5. 更新时序

PPO update 的时序为：

~~~text
rollout
 -> compute raw GAE and raw returns
 -> use update-start running statistics to get one scale
 -> all minibatches use that same scale
 -> backward / existing global clip / optimizer.step
 -> update Welford with this rollout's raw returns
 -> save checkpoint
~~~

当前 rollout 不会先更新自己的 scale 再用该 scale 训练自己。statistics 只吸收已经完成 GAE 的 train raw returns，不读取未来 Epoch、validation 或 test 数据。

mode=none 时 scale 固定为 1.0，且不更新 running statistics；这保持普通 v1/v2 的原始 loss 数学路径。

## 6. checkpoint/resume 语义

normalized checkpoint 增加：

~~~text
critic_value_scale_mode
critic_value_scale_epsilon
critic_value_scale_warmup_samples
critic_return_scale_count
critic_return_scale_mean
critic_return_scale_m2
~~~

normalized primitive checkpoint 使用 version=4；现有普通 primitive/v1/v2 checkpoint version=3 保持不变。

load_checkpoint 在读取 policy/optimizer 前显式比较 checkpoint mode 和当前配置：

- 旧 checkpoint 缺少 mode 时按 none 解释；
- normalized 配置读取普通 checkpoint 会抛出 cross-mode resume error；
- normalized checkpoint 的 epsilon、warmup、count、mean、M2 会校验；
- 不允许 plain v2 -> running_return_std 静默 resume；
- normalized 实验必须使用新的 run-dir/fresh training；
- 现有 v1/v2 checkpoint 不被修改。

## 7. diagnostics 和 update metrics

原始 critic diagnostics 字段继续保存原始单位：

~~~text
old_value
raw_advantage
raw_return
value_target
value_residual
abs_value_residual
bootstrap_value
~~~

新增：

~~~text
critic_return_scale
normalized_value_residual
~~~

update_metrics 保留 value_loss，并明确 normalized mode 下它是 optimizer 实际使用的 normalized half-MSE；新增：

~~~text
normalized_value_loss
raw_value_loss
critic_return_scale
critic_return_scale_count
~~~

raw_value_loss 和 normalized value loss 复用已经计算的 evaluation.value、returns tensor，不做第二次 forward。原始 value residual diagnostics 不替换为 normalized residual。

## 8. baseline 兼容性

~~~ini
V1_BASELINE_PRESERVED = YES
V2_BASELINE_PRESERVED = YES
VALUE_NORM_CONFIG_CREATED = YES
NORMALIZATION_STATS_CHECKPOINTED = YES
CROSS_MODE_SILENT_RESUME = NO
~~~

普通 v2 配置文件保持原样，以免改变现有 config hash；缺省 mode=none 是兼容路径。新配置的 mode、checkpoint version 和统计状态均与普通路径隔离。

gamma、gae_lambda、clip_epsilon、learning_rate、value_loss_coefficient、entropy_coefficient、max_grad_norm、update_epochs、num_minibatches、hidden_sizes 没有改动。

## 9. 纯计算验证

所有验证均使用 Python -B inline 代码，没有执行环境或 PPO update。

### 9.1 指定 residual 例子

~~~text
V = [100, 50]
R = [120, 10]
scale = 20

residual = [-20, 40]
normalized residual = [-1, 2]
raw half-MSE = 500
normalized half-MSE = 1.25
~~~

结果：PASS。

### 9.2 Welford

输入 [1,2,3,4]：

~~~text
count = 4
mean = 2.5
M2 = 5.0
population variance = 1.25
direct numpy mean/variance = 2.5 / 1.25
~~~

Welford 与直接 population mean/variance 一致；warmup=64 时 scale=1.0；取消 warmup 时 scale=sqrt(1.25+1e-8)。结果：PASS。

### 9.3 配置、forward、payload 和 resume guard

- baseline v2 mode=none：PASS；
- normalized config mode=running_return_std、epsilon=1e-8、warmup=64：PASS；
- dummy primitive v2 model forward：PASS；
- normalized payload version=4、mode 和 count/mean/M2：PASS；
- baseline payload version=3、mode=none：PASS；
- plain -> normalized cross-mode resume 显式拒绝：PASS。

dummy forward 只构造轻量 tensor/context，不运行真实环境。

## 10. 下一实验建议

下一任务再使用独立的新 run-dir 和新配置做短诊断训练，重点观察：

- warm-up 后 critic_return_scale；
- raw_value_loss 与 normalized_value_loss；
- gradient_norm > 0.5 的比例及 effective clip scale；
- 原始单位 abs_value_residual；
- time_to_go 的 origin sensitivity；
- reward/G、state、loss、gradient 的 finite 状态。

不要从普通 v2 Epoch 100 checkpoint resume，也不要同时修改 gamma、max_grad_norm、learning rate 或 value coefficient。一次短实验不能证明 PPO 收敛。

## 11. 最终状态字段

~~~ini
CRITIC_VALUE_SCALE_MODE_IMPLEMENTED = YES

VALUE_SCALE_MODE = RUNNING_RAW_RETURN_STD

RAW_RETURN_CHANGED = NO
RAW_VALUE_OUTPUT_CHANGED = NO
BOOTSTRAP_SCALE_CHANGED = NO
GAE_CHANGED = NO
ACTOR_ADVANTAGE_CHANGED = NO

SYSTEM_OBJECTIVE_CHANGED = NO
GAMMA_CHANGED = NO
MAX_GRAD_NORM_CHANGED = NO
VALUE_LOSS_COEFFICIENT_CHANGED = NO

V1_BASELINE_PRESERVED = YES
V2_BASELINE_PRESERVED = YES
VALUE_NORM_CONFIG_CREATED = YES

NORMALIZATION_STATS_CHECKPOINTED = YES
CROSS_MODE_SILENT_RESUME = NO

RAW_CRITIC_DIAGNOSTICS_PRESERVED = YES
RAW_VALUE_LOSS_LOGGED = YES
CRITIC_RETURN_SCALE_LOGGED = YES

NEW_ENVIRONMENT_RUN = NO
PPO_UPDATE_RUN = NO
LONG_TRAINING_RUN = NO
~~~

## 12. 清理与检查

- 没有创建或保留临时 Python 脚本、JSON、checkpoint 或 run-dir；
- 本任务产生的 Python cache 已清理；
- 没有 commit、push、merge 或 rebase；
- 仅保留正式实验代码、独立配置和本报告；
- git diff --check：待最后检查执行。

