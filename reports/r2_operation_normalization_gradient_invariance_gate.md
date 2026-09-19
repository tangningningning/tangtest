# R2 Operation Advantage Normalization / Gradient Invariance Gate

## Result

The gate supports the normalization-invariance diagnosis:

PRIMARY_DIAGNOSIS = OPERATION_STD_NORMALIZATION_ERASES_WEIGHT_SCALE

The raw operation-credit magnitude changes by approximately 20x between
w_move=10.0 and w_move=0.50, while production STD_ONLY normalization puts the
four advantage distributions on approximately unit standard deviation and
makes the operation-only gradients nearly equal in norm and direction. The
candidate/shared-encoder path is not the dominant cause in this fixed
minibatch.

## Diagnostic scope

The final valid diagnostic execution used:

- fresh seed 4401;
- a fresh actor and critic;
- config
  configs/proposed/system_config_ppo_recovery_r2_wmove_0p50_dev3.toml;
- frozen realization
  runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20001;
- one 300-transition rollout;
- one exact R2 counterfactual target collection;
- offline reconstruction for w_move = 10.0, 1.0, 0.50, 0.25;
- fixed operation minibatch indices 0..29;
- no PPO update, trainer.update(), backward write-back, or optimizer step.

The final collection executed 48,343 counterfactual environment branches. The
w=0.50 offline branch identity residual was 1.0825494320e-05, accepted as
floating-point replay/decomposition error. All diagnostic arrays were finite.

## Raw operation advantages

The operation source was OPERATION_ONLY; centering was disabled and the
normalization was STD_ONLY.

| weight | mean | std | mean abs | median abs | positive | negative |
|---:|---:|---:|---:|---:|---:|---:|
| 10.0 | -2.2269307e-03 | 1.9899146e-02 | 1.9961341e-02 | 2.0045395e-02 | 0.425000 | 0.575000 |
| 1.0 | -2.1950510e-04 | 1.9879733e-03 | 1.9909788e-03 | 2.0059453e-03 | 0.425000 | 0.575000 |
| 0.50 | -1.0798145e-04 | 9.9534775e-04 | 9.9616695e-04 | 1.0041545e-03 | 0.425000 | 0.575000 |
| 0.25 | -5.2219630e-05 | 5.0285608e-04 | 5.0116549e-04 | 5.0473763e-04 | 0.426667 | 0.573333 |

The key raw-scale ratios are:

- RAW_STD_RATIO_W10_TO_W050 = 19.9921545392
- RAW_MEAN_ABS_RATIO_W10_TO_W050 = 20.0381479515

The corresponding DeltaQ mean/std values were:

| weight | DeltaQ mean | DeltaQ std | positive | negative |
|---:|---:|---:|---:|---:|
| 10.0 | -4.0166485e-02 | 4.2111178e-04 | 0.000000 | 1.000000 |
| 1.0 | -4.0063849e-03 | 2.1932659e-04 | 0.000000 | 1.000000 |
| 0.50 | -1.9974905e-03 | 2.1052512e-04 | 0.003333 | 0.996667 |
| 0.25 | -9.9304331e-04 | 2.0630855e-04 | 0.008333 | 0.991667 |

## Production STD_ONLY advantages

| weight | scaled mean | scaled std | scaled mean abs | raw-to-scaled sign flips |
|---:|---:|---:|---:|---:|
| 10.0 | -0.1119108 | 0.9999995 | 1.0031250 | 0 |
| 1.0 | -0.1104160 | 0.9999950 | 1.0015068 | 0 |
| 0.50 | -0.1084851 | 0.9999900 | 1.0008130 | 0 |
| 0.25 | -0.1038440 | 0.9999801 | 0.9966182 | 0 |

For w=10.0 versus w=0.50:

- RAW_COSINE_W10_W050 = 0.9953950870
- SCALED_COSINE_W10_W050 = 0.9953950870
- RAW_PEARSON_W10_W050 = 0.9953449470
- SCALED_PEARSON_W10_W050 = 0.9953449470
- RAW_RELATIVE_L2_DIFF_W10_W050 = 0.9502413638
- SCALED_RELATIVE_L2_DIFF_W10_W050 = 0.0959502524

The signs are unchanged. Normalization removes most scale difference, while
the residual relative-L2 difference shows that it does not make the vectors
mathematically identical.

## Fixed-minibatch operation gradients

The gradients below are operation-loss-only gradients on the same fixed
minibatch and with the same old logits, sampled actions, PPO ratio, clip
epsilon, and policy parameters.

| weight | raw gradient norm | STD_ONLY gradient norm |
|---:|---:|---:|
| 10.0 | 4.2438691e-02 | 2.1326879 |
| 1.0 | 4.2592004e-03 | 2.1424729 |
| 0.50 | 2.1381188e-03 | 2.1480908 |
| 0.25 | 1.0775798e-03 | 2.1428764 |

Cross-weight comparisons:

- RAW_OP_GRAD_NORM_W10 = 0.04243869088
- RAW_OP_GRAD_NORM_W050 = 0.00213811882
- RAW_OP_GRAD_COSINE_W10_W050 = 0.9999989545
- STD_OP_GRAD_NORM_W10 = 2.1326879353
- STD_OP_GRAD_NORM_W050 = 2.1480907914
- STD_OP_GRAD_COSINE_W10_W050 = 0.9999989545

Direct same-weight raw-versus-STD comparisons:

| weight | raw/STD norm ratio | raw-vs-STD cosine |
|---:|---:|---:|
| 10.0 | 0.01989915645 | 0.999999999999989 |
| 0.50 | 0.00099535775 | 0.999999999999989 |

Therefore:

OP_STD_NORMALIZATION_ERASES_SCALE = SUPPORTED

The raw gradient scale remains highly weight-sensitive, while the normalized
gradient norm varies by less than 1% between w=10.0 and w=0.50.

## Operation/candidate shared-encoder decomposition

The operation head and candidate scorer share the four-parameter UAV encoder:

~~~
uav_encoder.0.weight
uav_encoder.0.bias
uav_encoder.2.weight
uav_encoder.2.bias
~~~

The cache and UAV encoders are otherwise decoupled. On the same fixed
minibatch:

- OP_SHARED_GRAD_NORM = 1.840977960
- CANDIDATE_SHARED_GRAD_NORM = 0.01066469738
- TOTAL_SHARED_GRAD_NORM = 0.9213225210
- CANDIDATE_TO_OPERATION_GRAD_NORM_RATIO = 0.00579295223
- OP_CANDIDATE_GRAD_COSINE = 0.1534921200

The production total is the existing 0.5 * (operation + candidate) UAV actor
loss; no loss weighting was changed. The candidate shared gradient is about
0.58% of the operation shared gradient and is not dominant. Its first-order
operation-logit direction is STAY, but its dot contribution (0.00593861) is
negligible compared with the operation-loss contribution (6.64506573). The
total first-order direction is also STAY, dominated by the operation term.

Thus:

SHARED_UAV_ENCODER = YES

CANDIDATE_OPERATION_GRADIENT_COUPLING = NOT_SUPPORTED

## Fixed-batch KL decomposition

This is a pre-update sanity diagnostic using the same policy before any
optimizer step:

- OPERATION_APPROX_KL = 4.9670538793e-09
- CANDIDATE_APPROX_KL = 2.8899222571e-08
- COMBINED_UAV_KL = 4.1723250632e-08
- KL_COMPONENT_DECOMPOSITION_AVAILABLE = YES

These values are effectively zero and do not indicate a first-update KL
problem in this diagnostic.

## Required final keys

~~~
RAW_STD_W10 = 0.01989914609
RAW_STD_W050 = 0.000995347753
RAW_STD_RATIO_W10_TO_W050 = 19.99215454

RAW_MEAN_ABS_W10 = 0.01996134063
RAW_MEAN_ABS_W050 = 0.000996166945
RAW_MEAN_ABS_RATIO_W10_TO_W050 = 20.03814795

SCALED_MEAN_ABS_W10 = 1.003124984
SCALED_MEAN_ABS_W050 = 1.000812967

RAW_COSINE_W10_W050 = 0.9953950870
SCALED_COSINE_W10_W050 = 0.9953950870
RAW_RELATIVE_L2_DIFF_W10_W050 = 0.9502413638
SCALED_RELATIVE_L2_DIFF_W10_W050 = 0.0959502524

RAW_OP_GRAD_NORM_W10 = 0.04243869088
RAW_OP_GRAD_NORM_W050 = 0.002138118822
RAW_OP_GRAD_COSINE_W10_W050 = 0.9999989545

STD_OP_GRAD_NORM_W10 = 2.132687935
STD_OP_GRAD_NORM_W050 = 2.148090791
STD_OP_GRAD_COSINE_W10_W050 = 0.9999989545

OP_STD_NORMALIZATION_ERASES_SCALE = SUPPORTED
SHARED_UAV_ENCODER = YES

OP_SHARED_GRAD_NORM = 1.840977960
CANDIDATE_SHARED_GRAD_NORM = 0.01066469738
CANDIDATE_TO_OPERATION_GRAD_NORM_RATIO = 0.00579295223
OP_CANDIDATE_GRAD_COSINE = 0.1534921200
CANDIDATE_OPERATION_GRADIENT_COUPLING = NOT_SUPPORTED

OPERATION_APPROX_KL = 4.967053879e-09
CANDIDATE_APPROX_KL = 2.889922257e-08
KL_COMPONENT_DECOMPOSITION_AVAILABLE = YES

PRIMARY_DIAGNOSIS = OPERATION_STD_NORMALIZATION_ERASES_WEIGHT_SCALE

TRAINING_EXECUTED = NO
OPTIMIZER_STEP_EXECUTED = NO

SOURCE_MODIFIED = NO
REWARD_CHANGED = NO
W_MOVE_CHANGED = NO
TARGET_KL_CHANGED = NO
ACTOR_LR_CHANGED = NO
CANDIDATE_CREDIT_CHANGED = NO
OPERATION_CREDIT_CHANGED = NO

EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO

TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
~~~

The diagnostic left max_parameter_abs_delta = 0.0 and all optimizer counters
at zero. No formal JSON artifact was generated; this markdown report is the
only formal artifact for this gate.
