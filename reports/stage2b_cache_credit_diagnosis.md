# STAGE2B Cache credit diagnosis

## Scope and algorithm status

This is a read-only diagnosis of the current Cache credit assignment. No PPO
update, optimizer step, checkpoint write, Stage2B credit implementation, or
formal run modification was performed.

The validation manifest for `runs/stage1b_cross_branch_short_gate` records:

```text
CACHE_RATIO = PER_CACHE_HEAD
CACHE_TRIGGER_CREDIT = GLOBAL_GAE
CACHE_CANDIDATE_CREDIT = GLOBAL_GAE
UAV_CANDIDATE_KL_GATE = MEAN_OR_P90
UAV_CANDIDATE_CONFIDENCE_REFERENCE_SCALE = 0.02
```

The JSONL artifacts do not retain complete transition tensors and conditional
Cache action objects. Therefore the existing train-only realization
`environment_data/proposed_joint_training/episode_20107` was replayed
read-only with the epoch-109 validation checkpoint. The replay used the
existing environment, Matching, KKT, configuration, and reward path; it did
not call `trainer.update`.

The current PPO implementation applies one global advantage tensor to the
Cache heads when Cache credit is `GLOBAL_GAE`. The diagnostic used the same
centered/std normalization rule as the actor loss for sign selection within
the 80-slot diagnostic prefix, then compared each selected head against a
one-step counterfactual.

## Diagnostic design

Twenty-four representative transitions were selected: 12 with positive and
12 with negative normalized global GAE. Selection also covered transitions
with many Cache UPDATE triggers and many Cache KEEP triggers. Across the
selected transitions there were 115 UPDATE heads and 101 KEEP heads.

For each selected transition and each of the 9 Cache heads:

- an actual UPDATE head was compared with the same joint action with that
  head changed to KEEP;
- an actual KEEP head was compared with a legal representative UPDATE
  candidate, selected by the current policy's highest candidate probability;
- all other Cache head actions, the UAV action, realization, prediction/state,
  and downstream configuration path were held fixed.

The local diagnostic is:

`D_tm = R(actual joint action) - R(counterfactual head-m action)`.

This is a trigger-level one-step reward difference. It is not substituted into
the PPO loss and it is not a full-horizon advantage.

```text
LOCAL_CREDIT_IS_FULL_HORIZON_ADVANTAGE = NO
CACHE_CANDIDATE_CREDIT_DIAGNOSIS = DEFERRED
```

Candidate-level Exact-Q credit was intentionally not enumerated in this pass.
The UPDATE reference above is only a low-cost trigger diagnostic, not a
candidate teacher.

## Coverage and execution

```text
TRANSITIONS_ANALYZED = 24
GLOBAL_ADV_POSITIVE_TRANSITIONS = 12
GLOBAL_ADV_NEGATIVE_TRANSITIONS = 12
COUNTERFACTUAL_BRANCHES_EXECUTED = 216
COUNTERFACTUAL_BRANCH_ERRORS = 0
COUNTERFACTUAL_BRANCHES_UNAVAILABLE = 0
TOTAL_RUNTIME_SECONDS = approximately 28.7
COUNTERFACTUAL_PHASE_RUNTIME_SECONDS = 3.84
```

The total runtime includes the read-only rollout replay; the counterfactual
phase is measured separately. No dev, final, or test realization was used.

## Sign consistency results

Zero-valued local credits were absent in the 216 valid samples, so the
sign-mismatch denominator is 216.

| diagnostic | count / rate |
|---|---:|
| `GLOBAL_ADV_POS_LOCAL_CREDIT_NEG_COUNT` | 66 |
| `GLOBAL_ADV_NEG_LOCAL_CREDIT_POS_COUNT` | 50 |
| total global/local sign mismatches | 116 / 216 |
| `GLOBAL_LOCAL_SIGN_MISMATCH_RATE` | 0.537037 |
| `LOCAL_CREDIT_POSITIVE_RATE` | 0.425926 |
| `LOCAL_CREDIT_NEGATIVE_RATE` | 0.574074 |
| local credit zero rate | 0.000000 |
| `HEAD_CREDIT_MIXED_SIGN_RATE` | 24 / 24 = 1.000000 |

Every analyzed transition had at least one positive and one negative head
credit. Thus a single scalar `A_global(t)` broadcasts one sign to all nine
heads even though the same transition's measured one-step head credits have
opposite signs.

### Per-trigger split

| actual trigger | valid samples | local positive | local negative | mean D | sign mismatches | mismatch rate |
|---|---:|---:|---:|---:|---:|---:|
| KEEP | 101 | 69 | 32 | -1.9270 | 54 | 0.534653 |
| UPDATE | 115 | 23 | 92 | -3.5269 | 62 | 0.539130 |

The mismatch is not confined to one trigger class. Both KEEP and UPDATE
trigger samples show the same roughly 53--54% sign disagreement against the
global advantage.

### Per-head split

All nine heads have 24 valid samples. `mismatch` compares each head's local
credit sign with the transition's global advantage sign.

| Cache head | local positive | local negative | mean D | mismatch |
|---:|---:|---:|---:|---:|
| 0 | 14 | 10 | -0.0508 | 16 |
| 1 | 8 | 16 | -1.8012 | 10 |
| 2 | 14 | 10 | -0.3116 | 14 |
| 3 | 10 | 14 | -2.5640 | 16 |
| 4 | 12 | 12 | -3.3778 | 14 |
| 5 | 14 | 10 | -3.6109 | 10 |
| 6 | 5 | 19 | -10.4378 | 11 |
| 7 | 11 | 13 | -0.9326 | 11 |
| 8 | 4 | 20 | -1.9227 | 14 |

The local-credit distribution over all valid head samples was:

```text
min = -140.5506
p10 = -2.4587
p50 = -0.0303
p90 =  0.2784
max =  9.7309
```

The large negative tail is a reward-scale observation from the existing
system model, not a modified reward or a new objective.

## Interpretation

This sample provides direct evidence that the current global-credit broadcast
is coarse at the Cache-head level:

1. More than half of valid head-level comparisons disagree in sign with the
   broadcast global advantage.
2. Mixed-sign local credit occurs in every analyzed transition, not only in a
   single exceptional slot.
3. The pattern persists for both actual KEEP and actual UPDATE triggers.
4. The evidence comes from 216 successful calls through the existing
   environment/configuration path, with no infeasible or failed branch.

This does not prove that a future per-head credit implementation will improve
rollout return. It establishes that the current `GLOBAL_GAE` assignment loses
head-specific one-step sign information in this representative train-only
sample. A larger diagnostic across additional train-only realizations would
be useful before choosing a particular Stage2B credit design, but is not
needed to identify the present coarse-credit signal.

## Final judgment

```text
CACHE_GLOBAL_GAE_CREDIT_PROBLEM = SUPPORTED
STAGE2B_IMPLEMENTATION_NEEDED = YES
TRAINING_EXECUTED = NO
COMMIT = NO
PUSH = NO
```

`STAGE2B_IMPLEMENTATION_NEEDED=YES` is a next-step recommendation based on
the observed trigger-level credit conflict. No Stage2B implementation was
made in this task.

