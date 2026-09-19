# R3 Fixed-Reference Operation Advantage Scale — Dev3

## Outcome

The R3 correctness gate passed, so the requested three-epoch development
training was executed. The fixed reference scale preserved the expected
operation-gradient weight sensitivity, but it did not prevent the observed
fast STAY collapse during training.

Final classification:

~~~text
OPERATION_POLICY_RESULT = STAY_COLLAPSE_PERSISTS
NORMALIZATION_FIX_EFFECT = NOT_SUPPORTED
~~~

This is a training-result classification only. No MOVE target, MOVE bonus,
STAY penalty, minimum movement rate, or entropy intervention was added.

## R3 implementation

R3 uses the same Recovery R2 policy architecture and the already verified
branch-specific KL state machine. The new policy schema is:

~~~text
adaptive_joint_cache_replica_aware_uav_operation_recovery_r3
~~~

Only the operation-credit normalization changed:

~~~text
operation_advantage_normalization_mode = "fixed_reference_scale"
operation_advantage_reference_scale = 0.02
~~~

The global candidate/critic path remains centered_std; only the operation
advantage consumed by the UAV operation PPO loss uses the fixed scale. The
reference value 0.02 is a numerical conditioning constant derived from the
canonical R2 w_move=10 raw operation-advantage standard deviation; it is not
a physical parameter, reward coefficient, movement-cost coefficient, or
objective weight.

## Correctness and gradient gate

Gate inputs:

- seed 4401;
- frozen realization
  runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911/episode_20001;
- one 300-transition rollout;
- one exact R2 counterfactual collection;
- 48,343 counterfactual environment branches;
- fixed operation minibatch indices 0..29;
- no trainer.update() and no optimizer step.

The canonical R2 raw operation-advantage statistics were reproduced within
the fixed gate tolerance; the raw credit formula and counterfactual Q path
were not changed. The R3 fixed-scale gate results were:

| metric | value |
|---|---:|
| raw std, w=10 | 0.01989914609 |
| raw std, w=0.50 | 0.000995401527 |
| fixed-scale gradient norm, w=10 | 2.121934523 |
| fixed-scale gradient norm, w=0.50 | 0.1069101613 |
| fixed-scale gradient ratio, w=10 / w=0.50 | 19.84782829 |
| historical STD_ONLY gradient ratio | 0.9928295134 |
| sign flips | 0 |
| centering applied | NO |
| max parameter delta | 0 |
| optimizer steps | 0 |

Therefore:

~~~text
R3_FIXED_SCALE_GATE = PASS
RAW_ADV_IDENTICAL_TO_R2 = YES
FIXED_SCALE_PRESERVES_WEIGHT_SENSITIVITY = YES
~~~

The fixed-scale advantage standard deviations were approximately 0.99496 at
w=10 and 0.04977 at w=0.50, as expected from dividing raw advantages by
0.02.

## Dev3 training

Training used fresh actor, critic, and optimizers with seed 4401; no R2
checkpoint was loaded and no resume was used. The frozen pool supplied
episode_20001, episode_20002, and episode_20003. All episodes contained
300 slots.

| epoch | P(MOVE) mean | UAVMove | operation entropy | return | UAV KL | Cache KL | critic EV | runtime (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5194006 | 0.5750000 | 0.6901013 | -975.8790 | 0.6403847 | 0.0771462 | -0.0000355 | 694.95 |
| 2 | 0.1893280 | 0.1983333 | 0.4819045 | -3217.6790 | 0.2898669 | 0.0328263 | 0.0040417 | 1406.88 |
| 3 | 0.0435068 | 0.0550000 | 0.1779281 | -2503.0519 | 0.0787322 | 0.0318305 | 0.0185232 | 1520.44 |

All epoch finite checks passed. Each epoch executed four PPO update epochs.
The cumulative optimizer counters at the end were 100 actor steps and 120
critic steps.

Branch-specific PPO state paths:

~~~text
EPOCH1_PPO_STATE_PATH = TT -> TF -> FF -> FF
EPOCH2_PPO_STATE_PATH = TT -> TF -> TF -> TF
EPOCH3_PPO_STATE_PATH = TT -> TT -> TT -> TF
~~~

The TT -> TF -> FF -> FF path in epoch 1 confirms that the previously fixed
FF absorbing-state and both-inactive actor-suppression behavior remained
active. The critic continued to update.

The R3 P(MOVE) trajectory is still a rapid STAY collapse, qualitatively
matching the R2 trajectory 0.5194 -> 0.1886 -> 0.0349. Consequently,
fixed-reference scaling alone is not a sufficient training-level remedy.

## Required final keys

~~~text
R3_SCHEMA = adaptive_joint_cache_replica_aware_uav_operation_recovery_r3
R3_CONFIG = configs/proposed/system_config_ppo_recovery_r3_wmove_0p50_fixedopscale_dev3.toml
W_MOVE = 0.50

OP_ADV_NORMALIZATION = FIXED_REFERENCE_SCALE
OP_ADV_REFERENCE_SCALE = 0.02
OP_ADV_CENTERING = NO
R3_FIXED_SCALE_GATE = PASS
RAW_ADV_IDENTICAL_TO_R2 = YES
SIGN_FLIP_COUNT = 0

FIXED_SCALE_W10_GRAD_NORM = 2.121934523
FIXED_SCALE_W050_GRAD_NORM = 0.1069101613
FIXED_SCALE_GRAD_RATIO_W10_TO_W050 = 19.84782829
STD_ONLY_GRAD_RATIO_W10_TO_W050 = 0.9928295134
FIXED_SCALE_PRESERVES_WEIGHT_SENSITIVITY = YES

DEV3_TRAINING_EXECUTED = YES
EPOCH1_P_MOVE = 0.5194006145
EPOCH2_P_MOVE = 0.1893280214
EPOCH3_P_MOVE = 0.0435068054
EPOCH1_UAV_MOVE = 0.5750000000
EPOCH2_UAV_MOVE = 0.1983333333
EPOCH3_UAV_MOVE = 0.0550000000
EPOCH1_UAV_KL = 0.6403847344
EPOCH2_UAV_KL = 0.2898669155
EPOCH3_UAV_KL = 0.0787322370
EPOCH1_CRITIC_EV = -0.0000355070
EPOCH2_CRITIC_EV = 0.0040417294
EPOCH3_CRITIC_EV = 0.0185232150
EPOCH1_PPO_STATE_PATH = TT -> TF -> FF -> FF
EPOCH2_PPO_STATE_PATH = TT -> TF -> TF -> TF
EPOCH3_PPO_STATE_PATH = TT -> TT -> TT -> TF

OPERATION_POLICY_RESULT = STAY_COLLAPSE_PERSISTS
NORMALIZATION_FIX_EFFECT = NOT_SUPPORTED
KL_STATE_MACHINE_FIX_PRESERVED = YES

REWARD_FORMULA_CHANGED = NO
MOVEMENT_COST_FORMULA_CHANGED = NO
W_MOVE_CHANGED_FROM_DEV_BASELINE = NO
TARGET_KL_CHANGED = NO
ACTOR_LR_CHANGED = NO
CANDIDATE_CREDIT_CHANGED = NO
CACHE_CREDIT_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO

EPISODE_REGENERATION = NO
SUMO_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO
FORMAL_EVALUATION_EXECUTED = NO
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO
~~~

Formal run artifacts:

- [run manifest](<C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_recovery_r3_wmove_0p50_fixedopscale_dev3_20260919\run_manifest.json>)
- [run summary](<C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_recovery_r3_wmove_0p50_fixedopscale_dev3_20260919\run_summary.json>)
- [latest checkpoint](<C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_recovery_r3_wmove_0p50_fixedopscale_dev3_20260919\latest_checkpoint.pt>)
