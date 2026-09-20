# ADAPTIVE_JOINT_FRESH_ONE_EPOCH_TRAINABILITY_GATE

## 1. Scope

This is the single fresh one-epoch gate for the current `main` working tree.
It tests whether the reconfigured `proposed_joint` action representation is
trainable through the existing `EXACT_Q_LISTWISE` path. No long training,
final-test seeds, reward/model changes, or action redesign was performed.

- Branch: `main`
- HEAD at launch: `0296816`
- Working tree action-reconfiguration changes: preserved
- Experiment ID: `proposed_joint`
- Policy schema: `adaptive_joint_cache_uav_ppo`
- Config: `configs/proposed/system_config.toml`
- Control mode: `adaptive_joint_cache_uav_reconfiguration`
- Candidate objective: `EXACT_Q_LISTWISE`
- Cache high-level actions: `KEEP / UPDATE`
- UAV high-level actions: `STAY / MOVE`

## 2. Current action semantics

Cache UPDATE candidates were complete feasible target cache configurations.
UAV MOVE candidates were demand-aware target positions. Primitive `ADD`,
`DELETE`, `SWAP`, and primitive MOVE/STAY candidate semantics were not used by
this run.

## 3. Fresh initialization proof

The gate run directory was absent before launch and the trainer was invoked
with `resume=False`; the old checkpoint
`runs/exact_q_listwise_one_epoch_validation_20260919/latest_checkpoint.pt`
was not passed or loaded. The new checkpoint was created only after the first
completed epoch.

- `FRESH_INITIALIZATION = YES`
- `OLD_CHECKPOINT_LOADED = NO`
- Training realization seed: `20001`
- Policy/random seed from the formal proposed config: `4401`
- Final-test seeds `40001..40020`: not used
- Outer epochs: `1`
- PPO `update_epochs`: `4`
- PPO optimizer steps: actor `30`, critic `40`; branch-specific KL gating
  suppressed later actor branch updates but did not start another outer epoch.

The before/after candidate comparison reused the exact-Q tensors generated in
this same update: candidate features, legal masks, candidate Q values, and
candidate ordering were not regenerated after the optimizer step.

## 4. One-epoch runtime

| Component | Seconds |
| --- | ---: |
| Rollout | 66.958691 |
| Cache Exact-Q replay | 902.456074 |
| UAV Exact-Q replay | 226.721544 |
| PPO update remainder | 65.456955 |
| Total epoch wall time | 1261.773848 |

The PPO update remainder is `update_runtime_seconds` minus the Cache and UAV
Exact-Q replay times. Matching/KKT time was not separately instrumented.
The run executed `42,700` Cache counterfactual environment branches and
`11,276` UAV counterfactual environment branches.

## 5. Cache trigger diagnostics

The sampled high-level Cache rates were:

- `CACHE_ACTUAL_UPDATE_RATE = 0.497037`
- `CACHE_KEEP` sampled rate = `0.502963`
- `CACHE_UPDATE` sampled rate = `0.497037`
- requested updates = `1342 / 2700`
- actual Cache updates = `1342 / 1342` requested updates

The exact per-state high-level trigger probability tensor and `Q_KEEP` array
were not persisted by the pre-existing formal JSONL logger. They are therefore
reported as unavailable here rather than reconstructed by a second expensive
environment replay. The post-update `Q_UPDATE` quantity is the frozen
candidate-policy expected Exact-Q reported below:

- `CACHE_Q_KEEP_MEAN = NOT_PERSISTED`
- `CACHE_Q_UPDATE_MEAN_AFTER = -6.272896`
- `CACHE_Q_UPDATE_MEAN_BEFORE = -6.536208`

This logging gap does not affect the candidate before/after gate, whose Q,
probability, and mask tensors were captured directly inside `trainer.update`.

## 6. Cache target-listwise trainability

The frozen comparison batch contained `2700` Cache node-time heads with legal
UPDATE target candidates. Mean legal candidate count was `15.703704`, maximum
was `36`.

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Top-1 Exact-Q accuracy | 0.052222 | 0.193704 | +0.141481 |
| Mean Exact-Q regret | 9.845818 | 7.311057 | -2.534761 |
| Expected Exact-Q | -6.536208 | -6.272896 | +0.263312 |
| Target policy KL(old\|new) | — | — | 0.016961 |

Loss/update evidence:

- `CACHE_TRIGGER_POLICY_LOSS = 0.064242`
- `CACHE_TARGET_LISTWISE_LOSS = 0.487924`
- `CACHE_TARGET_SCORER_PARAMETER_DELTA_NORM = 1.021186`

Cache candidate ranking moved toward the Exact-Q teacher on all three primary
signals: Top-1 increased, regret decreased, and expected Exact-Q increased.

## 7. Cache proposal-source diagnostics

Exact-Q was evaluated only over the current proposal set; no claim of global
configuration-space recall is made.

- Candidate count mean/max: `15.703704 / 36`
- Best-Q source counts: `low_reconfiguration=2564`, `prediction=95`,
  `replica_aware=24`, `intermediate=17`
- Best-Q source fractions: `0.949630`, `0.035185`, `0.008889`, `0.006296`
  respectively
- Best-Q Hamming distance from current: mean `1.682593`, median `2`, max `8`

The proposal set is dominated by low-reconfiguration candidates in this
training realization; that is an observed proposal fact, not an algorithmic
change made by this gate.

## 8. UAV operation diagnostics

The sampled high-level UAV rates were:

- `UAV_ACTUAL_MOVE_RATE = 0.493333`
- `UAV_STAY` sampled rate = `0.506667`
- `UAV_MOVE` sampled rate = `0.493333`
- requested moves = `296 / 600`
- actual moves = `296 / 296` requested moves

The exact per-state `Q_STAY` array and policy-probability mean were not
persisted by the pre-existing formal JSONL logger. They are not reconstructed
with a second Exact-Q run:

- `UAV_Q_STAY_MEAN = NOT_PERSISTED`
- `UAV_Q_MOVE_MEAN_AFTER = -6.859512`
- `UAV_Q_MOVE_MEAN_BEFORE = -6.859551`

## 9. UAV target-listwise trainability

The frozen comparison batch contained `600` UAV head-time states with legal
MOVE target positions. Mean legal candidate count was `17.793333`, maximum was
`18`.

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Top-1 Exact-Q accuracy | 0.315000 | 0.318333 | +0.003333 |
| Mean Exact-Q regret | 0.000400 | 0.000377 | -0.000023 |
| Expected Exact-Q | -6.859551 | -6.859512 | +0.000039 |
| Target policy KL(old\|new) | — | — | 0.000401 |

Loss/update evidence:

- `UAV_OPERATION_POLICY_LOSS = -0.006204`
- `UAV_TARGET_LISTWISE_LOSS = 0.295695`
- `UAV_TARGET_SCORER_PARAMETER_DELTA_NORM = 0.474293`

The UAV candidate movement is small but consistently non-negative on the
specified gate criteria: Top-1 increased, regret decreased, and expected
Exact-Q increased. It should be treated as weak one-epoch support, not a
performance claim.

## 10. UAV proposal-source diagnostics

Exact-Q was evaluated only over the current target-position proposal set.

- Candidate count mean/max: `17.793333 / 18`
- Best-Q source counts: `geometry_exploration=332`, `demand_centroid=187`,
  `projected_demand=81`
- Best-Q source fractions: `0.553333`, `0.311667`, `0.135000`
  respectively
- Best-Q MOVE distance: mean `11.887500 m`, median `15.000000 m`, max
  `15.000000 m`

Exact-Q was evaluated only over the current proposal set. Geometry exploration
supplied the largest share of best-Q candidates in this realization; demand-
aware sources supplied `268/600` best-Q candidates. No proposal modification
was made.

## 11. Before/after Exact-Q policy quality

| Branch | Top-1 before | Top-1 after | Regret before | Regret after | Expected-Q before | Expected-Q after | KL(old\|new) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cache target | 0.052222 | 0.193704 | 9.845818 | 7.311057 | -6.536208 | -6.272896 | 0.016961 |
| UAV target | 0.315000 | 0.318333 | 0.000400 | 0.000377 | -6.859551 | -6.859512 | 0.000401 |

The same frozen candidate/Q batch was used for both columns. The measured
candidate scorer deltas were positive and finite for both branches.

## 12. Numerical checks and training dynamics

- Return: `-2063.866769`
- Cache change rate: `0.497037`
- UAV move rate: `0.493333`
- Value loss: `0.536825`
- Entropy: `21.330486`
- Approximate KL: `0.885203`
- Cache branch KL: `0.087568`
- UAV branch KL: `0.625867`
- All recorded reward, return, loss, gradient, advantage, and finite checks
  were true; `run_summary.nan_inf_detected = false`.
- The aggregate actor policy loss is not used as the candidate conclusion;
  trigger/operation and target-listwise losses are reported separately.

## 13. Gate conclusion

```text
CACHE_TARGET_SCORER_UPDATED = YES
CACHE_EXPECTED_Q_DIRECTION = SUPPORTED
CACHE_TARGET_POLICY_TRAINABLE = YES

UAV_TARGET_SCORER_UPDATED = YES
UAV_EXPECTED_Q_DIRECTION = SUPPORTED
UAV_TARGET_POLICY_TRAINABLE = YES

JOINT_ONE_EPOCH_TRAINABILITY_GATE = PASS
```

Both target scorers received finite non-zero parameter updates. Both frozen
candidate policies moved in the expected Exact-Q direction, with strong Cache
evidence and small-but-consistent UAV evidence. There was no numerical or
action-schema failure.

## 14. Next-step decision

```text
READY_FOR_FORMAL_TRAINING = YES
NEXT_ACTION = Do not start it automatically; obtain explicit authorization before formal long training.
```

This gate is not evidence that return will improve over long training. It only
supports that the new target candidate heads are connected to the real PPO
update and are trainable for one epoch.

## Formal artifact fields

```text
TRAINING_EXECUTED = YES
FINAL_TEST_SEEDS_USED = NO
REPORT_PATH = reports/adaptive_joint_cache_uav_one_epoch_trainability_gate.md
COMMITTED = NO
PUSHED = NO
```
