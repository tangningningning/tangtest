# RECOVERY_R1_UAV_DECISION_QUALITY_AUDIT

Bounded real-environment witness audit; no training or PPO/reward/system-model changes were made.

- Checkpoint: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v10d_decoupled_encoder_branch_kl_6e4_formal\latest_checkpoint.pt`
- Frozen realizations: `[20001, 20100, 20200, 20300]`; origins `[80, 200]`
- Paired comparisons: `16`
- Continuation mode: `DETERMINISTIC_ARGMAX`; common random numbers: `NO (deterministic continuation; no stochastic stream mixed)`
- Forced MOVE rule: `POLICY_TOP1_LEGAL_MOVE_CANDIDATE`

## Required fields

- `PRODUCTION_MOVE_COUNT = 16`
- `PRODUCTION_STAY_COUNT = 0`
- `FORCE_STAY_H1_BETTER_COUNT = 16`
- `FORCE_STAY_H30_BETTER_COUNT = 14`
- `FORCE_MOVE_H1_BETTER_COUNT = 0`
- `FORCE_MOVE_H30_BETTER_COUNT = 0`
- `MEAN_DELTA_G_H1 = 0.036523550`
- `MEDIAN_DELTA_G_H1 = 0.028872785`
- `MEAN_DELTA_G_H30 = 0.066566847`
- `MEDIAN_DELTA_G_H30 = 0.059723521`
- `UAV_POLICY_STATE_SENSITIVITY = SUPPORTED`
- `UAV_OVERMOVE_POLICY_ERROR = SUPPORTED`
- `UAV_TEMPORAL_CREDIT_MISMATCH = NOT_SUPPORTED`

## Paired witness differences

`ΔG = G(force alternative) - G(production)`. Positive values favor the forced alternative. Decomposition uses only actual environment outputs.

| Episode | Origin | UAV | Production | Forced | ΔG H1 | ΔG H30 | ΔQ H30 | ΔCacheCost H30 | ΔMovementCost H30 |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| 20001 | 80 | UAV_0 | MOVE | STAY | 0.029180 | 0.060757 | 0.033257 | 0.000000 | -0.027500 |
| 20001 | 80 | UAV_1 | MOVE | STAY | 0.046982 | 0.082510 | 0.035210 | 0.000000 | -0.047300 |
| 20001 | 200 | UAV_0 | MOVE | STAY | 0.028320 | 0.083606 | 0.056106 | 0.000000 | -0.027500 |
| 20001 | 200 | UAV_1 | MOVE | STAY | 0.028393 | 0.058998 | 0.031498 | 0.000000 | -0.027500 |
| 20100 | 80 | UAV_0 | MOVE | STAY | 0.028512 | 0.045563 | 0.018063 | 0.000000 | -0.027500 |
| 20100 | 80 | UAV_1 | MOVE | STAY | 0.038242 | -0.043409 | -0.080809 | 0.000000 | -0.037400 |
| 20100 | 200 | UAV_0 | MOVE | STAY | 0.028566 | 0.038647 | 0.011147 | 0.000000 | -0.027500 |
| 20100 | 200 | UAV_1 | MOVE | STAY | 0.053022 | 0.342971 | 0.295671 | 0.000000 | -0.047300 |
| 20200 | 80 | UAV_0 | MOVE | STAY | 0.031930 | 0.060449 | 0.032949 | 0.000000 | -0.027500 |
| 20200 | 80 | UAV_1 | MOVE | STAY | 0.056897 | 0.104251 | 0.056951 | 0.000000 | -0.047300 |
| 20200 | 200 | UAV_0 | MOVE | STAY | 0.028233 | 0.039608 | 0.012108 | 0.000000 | -0.027500 |
| 20200 | 200 | UAV_1 | MOVE | STAY | 0.048811 | 0.046775 | -0.000525 | 0.000000 | -0.047300 |
| 20300 | 80 | UAV_0 | MOVE | STAY | 0.028467 | 0.156794 | 0.129294 | 0.000000 | -0.027500 |
| 20300 | 80 | UAV_1 | MOVE | STAY | 0.027248 | 0.002393 | -0.025107 | 0.000000 | -0.027500 |
| 20300 | 200 | UAV_0 | MOVE | STAY | 0.024971 | -0.114450 | -0.141950 | 0.000000 | -0.027500 |
| 20300 | 200 | UAV_1 | MOVE | STAY | 0.056602 | 0.099606 | 0.042406 | 0.000000 | -0.057200 |

## Policy-state sensitivity

`P(MOVE)` min=0.909785369, max=0.989899230, mean=0.955507075, std=0.034752149, entropy_mean=0.166207567.

This is a bounded witness only; it does not establish statistical significance or causal generalization, and it does not assess Cache decisions.

## Status

`TRAINING_EXECUTED = NO`; `NEW_CHECKPOINT_CREATED = NO`; `EPISODE_REGENERATION = NO`; `FINAL_TEST_SEEDS_USED = NO`.
`SYSTEM_MODEL_CHANGED = NO`; `REWARD_CHANGED = NO`; `PPO_CHANGED = NO`; `PUSHED = NO`.
