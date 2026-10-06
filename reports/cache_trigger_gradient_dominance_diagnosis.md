# Cache Trigger Gradient Dominance and Encoder Conflict Diagnosis

## Scope and method

Read-only diagnosis on corrected-v2 epoch-140 checkpoints. Policies stayed in eval mode and their parameter hashes were checked before and after each stage. No optimizer was constructed or stepped; gradients were obtained only with torch.autograd.grad.

Only training-pool realizations 20001, 20100, 20200, and 20300 were used. Each checkpoint/realization pair generated a stochastic production rollout covering all 300 slots. Action RNG was reset to 20260925 + realization seed before each rollout.

For each realization, fixed minibatches 0, 4, and 8 were selected from the production 10 minibatches of 30 transitions using a seeded permutation. The same indices were used for both checkpoints. Current production compute_one_step_tdq_cache_trigger_credit was run on the union of the selected transitions and all nine Cache heads. No symmetric-Q branches were rerun.

Population standard deviations (ddof=0) are computed over valid raw TD-Q values in all selected diagnostic minibatches for that checkpoint. Global scaling pools all four realizations; per-head and per-role scaling pool their respective valid values. Each divides by std + 1e-8 with no centering; signs are verified unchanged. A_GLOBAL uses one common positive divisor for every head, so it rescales the total trigger gradient while retaining the relative head weighting. In both checkpoints, the measured max/min ratio increased under per-head and per-role rescaling, so the observed head imbalance is not attributed to the global divisor.

The trigger loss uses the production binary-TD-Q clipped PPO surrogate and sample-by-head mean. Candidate loss uses selected UPDATE candidates, production old candidate log probabilities, GLOBAL_GAE from the full rollout, checkpoint GAE normalization, and production clip epsilon. These are unweighted component gradients; production weights both by 0.5, which does not change cosine or their norm ratio.

Operation-row norms cover the two cache_trigger_head output rows per Cache head. Pairwise and total gradients are measured over shared cache_encoder parameters. A max/min ratio is undefined if any head norm is <=1e-12; such minibatches are excluded from the ratio median and counted in JSON. Conflict requires at least 75% negative trigger/candidate cosines across 12 minibatches for both checkpoints.

## Required decisions

GLOBAL_TDQ_SCALE_CAUSES_GRADIENT_DOMINANCE = NO
PER_HEAD_SCALE_REDUCES_GRADIENT_IMBALANCE = NO
PER_ROLE_SCALE_REDUCES_GRADIENT_IMBALANCE = NO
CACHE_TRIGGER_CANDIDATE_SHARED_ENCODER_CONFLICT = INCONCLUSIVE

## Checkpoint 5e-4

- GLOBAL encoder max/min = 189.34; per-head = 495.69; per-role = 1054.8.
- Operation-row max/min: global = 216.11; per-head = 533.66; per-role = 1204.2.
- GLOBAL trigger/candidate encoder cosine = 0.21219; trigger grad norm = 0.001549; candidate grad norm = 0.021873; norm ratio trigger/candidate = 0.066371. Negative minibatches = 3/12.
- GLOBAL pairwise head cosines: negative fraction = 0.4838; mean = 0.034882; p10/p50/p90 = -0.53841/0.072127/0.63454.

## Checkpoint 1e-3

- GLOBAL encoder max/min = 15389; per-head = 83317; per-role = 98090.
- Operation-row max/min: global = 17213; per-head = 78722; per-role = 1.1101e+05.
- GLOBAL trigger/candidate encoder cosine = -0.022636; trigger grad norm = 0.002713; candidate grad norm = 0.045373; norm ratio trigger/candidate = 0.059608. Negative minibatches = 6/12.
- GLOBAL pairwise head cosines: negative fraction = 0.40046; mean = 0.07375; p10/p50/p90 = -0.42598/0.082764/0.49675.

## Scale comparison

| Checkpoint | Scale | Encoder max/min | Operation-row max/min | Pairwise negative | Pairwise mean | Trigger direction vs GLOBAL | Trigger/candidate cosine | Trigger/candidate norm ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 5e-4 | GLOBAL | 189.34 | 216.11 | 0.4838 | 0.034882 | 1 | 0.21219 | 0.066371 |
| 5e-4 | PER_HEAD | 495.69 | 533.66 | 0.4838 | 0.034881 | 0.99904 | 0.21331 | 0.33961 |
| 5e-4 | PER_ROLE | 1054.8 | 1204.2 | 0.4838 | 0.034881 | 0.99366 | 0.14532 | 0.31767 |
| 1e-3 | GLOBAL | 15389 | 17213 | 0.40046 | 0.07375 | 1 | -0.022636 | 0.059608 |
| 1e-3 | PER_HEAD | 83317 | 78722 | 0.40046 | 0.073714 | 0.99931 | -0.014054 | 0.33591 |
| 1e-3 | PER_ROLE | 98090 | 1.1101e+05 | 0.40046 | 0.073783 | 0.99972 | -0.030043 | 0.34911 |

## Per-head gradient norms

Median across the 12 diagnostic minibatches. The JSON also contains the full 9-by-9 mean pairwise cosine matrix for every checkpoint and scale.

| Checkpoint | Scale | Head | Role | Encoder grad norm | Operation-row grad norm |
|---|---|---:|---|---:|---:|
| 5e-4 | GLOBAL | 0 (46) | SV | 7.2316e-06 | 2.0377e-05 |
| 5e-4 | GLOBAL | 1 (10) | SV | 7.3404e-05 | 0.00024371 |
| 5e-4 | GLOBAL | 2 (0) | SV | 2.7876e-05 | 8.9613e-05 |
| 5e-4 | GLOBAL | 3 (23) | SV | 3.0869e-05 | 8.1333e-05 |
| 5e-4 | GLOBAL | 4 (59) | SV | 1.0634e-05 | 4.1847e-05 |
| 5e-4 | GLOBAL | 5 (RSU_0) | RSU | 1.8586e-05 | 5.3945e-05 |
| 5e-4 | GLOBAL | 6 (RSU_1) | RSU | 7.814e-05 | 0.000142 |
| 5e-4 | GLOBAL | 7 (UAV_0) | UAV | 0.00013465 | 0.0003232 |
| 5e-4 | GLOBAL | 8 (UAV_1) | UAV | 0.00077465 | 0.0027887 |
| 5e-4 | PER_HEAD | 0 (46) | SV | 1.4689e-05 | 4.1392e-05 |
| 5e-4 | PER_HEAD | 1 (10) | SV | 0.00052315 | 0.0017369 |
| 5e-4 | PER_HEAD | 2 (0) | SV | 2.1791e-05 | 7.005e-05 |
| 5e-4 | PER_HEAD | 3 (23) | SV | 0.00017317 | 0.00045628 |
| 5e-4 | PER_HEAD | 4 (59) | SV | 3.0872e-05 | 0.00012148 |
| 5e-4 | PER_HEAD | 5 (RSU_0) | RSU | 3.4926e-05 | 0.00010137 |
| 5e-4 | PER_HEAD | 6 (RSU_1) | RSU | 0.00059894 | 0.0010885 |
| 5e-4 | PER_HEAD | 7 (UAV_0) | UAV | 0.00065661 | 0.0015761 |
| 5e-4 | PER_HEAD | 8 (UAV_1) | UAV | 0.00401 | 0.014436 |
| 5e-4 | PER_ROLE | 0 (46) | SV | 6.5133e-06 | 1.8346e-05 |
| 5e-4 | PER_ROLE | 1 (10) | SV | 6.6106e-05 | 0.00021948 |
| 5e-4 | PER_ROLE | 2 (0) | SV | 2.5106e-05 | 8.0708e-05 |
| 5e-4 | PER_ROLE | 3 (23) | SV | 2.7799e-05 | 7.3247e-05 |
| 5e-4 | PER_ROLE | 4 (59) | SV | 9.5771e-06 | 3.7687e-05 |
| 5e-4 | PER_ROLE | 5 (RSU_0) | RSU | 2.3051e-05 | 6.6906e-05 |
| 5e-4 | PER_ROLE | 6 (RSU_1) | RSU | 9.6901e-05 | 0.0001761 |
| 5e-4 | PER_ROLE | 7 (UAV_0) | UAV | 0.00067582 | 0.0016222 |
| 5e-4 | PER_ROLE | 8 (UAV_1) | UAV | 0.0038882 | 0.013997 |
| 1e-3 | GLOBAL | 0 (46) | SV | 4.6634e-05 | 0.00010699 |
| 1e-3 | GLOBAL | 1 (10) | SV | 1.4072e-07 | 5.3399e-07 |
| 1e-3 | GLOBAL | 2 (0) | SV | 1.9421e-05 | 4.5242e-05 |
| 1e-3 | GLOBAL | 3 (23) | SV | 2.2015e-05 | 4.9566e-05 |
| 1e-3 | GLOBAL | 4 (59) | SV | 2.2866e-06 | 5.7548e-06 |
| 1e-3 | GLOBAL | 5 (RSU_0) | RSU | 7.9109e-06 | 1.3214e-05 |
| 1e-3 | GLOBAL | 6 (RSU_1) | RSU | 6.7924e-06 | 1.5149e-05 |
| 1e-3 | GLOBAL | 7 (UAV_0) | UAV | 9.9254e-05 | 0.00033978 |
| 1e-3 | GLOBAL | 8 (UAV_1) | UAV | 0.0025802 | 0.0099832 |
| 1e-3 | PER_HEAD | 0 (46) | SV | 0.00037256 | 0.00085473 |
| 1e-3 | PER_HEAD | 1 (10) | SV | 1.731e-07 | 6.581e-07 |
| 1e-3 | PER_HEAD | 2 (0) | SV | 2.6106e-05 | 6.0815e-05 |
| 1e-3 | PER_HEAD | 3 (23) | SV | 7.0204e-05 | 0.00015807 |
| 1e-3 | PER_HEAD | 4 (59) | SV | 1.0391e-05 | 2.6152e-05 |
| 1e-3 | PER_HEAD | 5 (RSU_0) | RSU | 4.8824e-05 | 8.1553e-05 |
| 1e-3 | PER_HEAD | 6 (RSU_1) | RSU | 3.4579e-05 | 7.7121e-05 |
| 1e-3 | PER_HEAD | 7 (UAV_0) | UAV | 0.00063439 | 0.0021717 |
| 1e-3 | PER_HEAD | 8 (UAV_1) | UAV | 0.013906 | 0.053806 |
| 1e-3 | PER_ROLE | 0 (46) | SV | 4.2006e-05 | 9.6371e-05 |
| 1e-3 | PER_ROLE | 1 (10) | SV | 1.2744e-07 | 4.8365e-07 |
| 1e-3 | PER_ROLE | 2 (0) | SV | 1.7497e-05 | 4.076e-05 |
| 1e-3 | PER_ROLE | 3 (23) | SV | 1.9832e-05 | 4.4648e-05 |
| 1e-3 | PER_ROLE | 4 (59) | SV | 2.0594e-06 | 5.1837e-06 |
| 1e-3 | PER_ROLE | 5 (RSU_0) | RSU | 4.3928e-05 | 7.3376e-05 |
| 1e-3 | PER_ROLE | 6 (RSU_1) | RSU | 3.7717e-05 | 8.4115e-05 |
| 1e-3 | PER_ROLE | 7 (UAV_0) | UAV | 0.00057812 | 0.0019791 |
| 1e-3 | PER_ROLE | 8 (UAV_1) | UAV | 0.015029 | 0.058148 |

## Decision rules

- Global dominance YES requires both checkpoints to have global median encoder max/min >=5 and per-head median <=2; NO applies if global ratios are <5 in both checkpoints or per-head scaling fails to reduce the max/min ratio in both checkpoints; otherwise INCONCLUSIVE. Here per-head ratios increased in both checkpoints.
- Per-head/per-role reduction YES requires both encoder and operation-row ratios to fall in at least 75% of paired minibatches and both medians to be lower in each checkpoint. NO requires neither ratio to fall in at least 75% of pairs for both checkpoints; otherwise INCONCLUSIVE.
- Shared encoder conflict YES requires GLOBAL trigger/candidate cosine <0 in at least 75% of minibatches for both checkpoints; NO requires <=25% negative for both; otherwise INCONCLUSIVE. A single negative cosine is not treated as a root cause.

## Required status fields

GLOBAL_TDQ_SCALE_CAUSES_GRADIENT_DOMINANCE = NO
PER_HEAD_SCALE_REDUCES_GRADIENT_IMBALANCE = NO
PER_ROLE_SCALE_REDUCES_GRADIENT_IMBALANCE = NO
CACHE_TRIGGER_CANDIDATE_SHARED_ENCODER_CONFLICT = INCONCLUSIVE
GLOBAL_HEAD_GRAD_NORM_MAX_MIN_RATIO_5E4 = 189.34
PER_HEAD_GRAD_NORM_MAX_MIN_RATIO_5E4 = 495.69
PER_ROLE_GRAD_NORM_MAX_MIN_RATIO_5E4 = 1054.8
GLOBAL_HEAD_GRAD_NORM_MAX_MIN_RATIO_1E3 = 15389
PER_HEAD_GRAD_NORM_MAX_MIN_RATIO_1E3 = 83317
PER_ROLE_GRAD_NORM_MAX_MIN_RATIO_1E3 = 98090
TRIGGER_CANDIDATE_COSINE_5E4 = 0.21219
TRIGGER_CANDIDATE_COSINE_1E3 = -0.022636
TRAINING_EXECUTED = NO
OPTIMIZER_STEP_CALLS = 0
VALIDATION_EXECUTED = NO
TEST_EXECUTED = NO

## Rollout/minibatch inventory

### 5e-4
- Seed 20001: 300 transitions/300 slots, action seed 20280926, minibatch seed 410001, 810 TD-Q branches, 355 sampled updates.
- Seed 20100: 300 transitions/300 slots, action seed 20281025, minibatch seed 410100, 810 TD-Q branches, 352 sampled updates.
- Seed 20200: 300 transitions/300 slots, action seed 20281125, minibatch seed 410200, 810 TD-Q branches, 356 sampled updates.
- Seed 20300: 300 transitions/300 slots, action seed 20281225, minibatch seed 410300, 810 TD-Q branches, 360 sampled updates.

### 1e-3
- Seed 20001: 300 transitions/300 slots, action seed 20280926, minibatch seed 410001, 810 TD-Q branches, 532 sampled updates.
- Seed 20100: 300 transitions/300 slots, action seed 20281025, minibatch seed 410100, 810 TD-Q branches, 535 sampled updates.
- Seed 20200: 300 transitions/300 slots, action seed 20281125, minibatch seed 410200, 810 TD-Q branches, 534 sampled updates.
- Seed 20300: 300 transitions/300 slots, action seed 20281225, minibatch seed 410300, 810 TD-Q branches, 527 sampled updates.

## Source hashes

- src/ppo_simple/policy.py: c0a8ebede1fbcde55cf1b6666fd06f03b3c0acbebfd43223f80094ecc60952f8
- src/ppo_simple/trainer.py: 4e3d5d760ac07efc50121ac97c9257ad8ed4fe5957a7f4f9e727e9795582f7fa
- src/ppo_simple/cache_credit.py: 7bd0a2037d9e2df51b05c08dd6b985bdca7cbb3b71e1d7c4766216751507ecd3
- src/ppo_simple/rollout.py: de496c8bb0a32b0cdd8a0c070e46918389c5f7ea060fe7a39823a28578db3418
- src/ppo_simple/formal_evaluation.py: 5a1c23f9be4e0bf9218401cd451ba223279208c2a04330c8899010db15eb6a8c

No algorithm, loss, reward, model, Cache cost, LR, checkpoint, or environment data was modified. No formal training, validation, test, or final unseen seed was run.
