# Stage2B1 Cache trigger credit design diagnosis

## Decision summary

This is a read-only diagnosis using existing train-only realizations.  No PPO
training, optimizer step, checkpoint write, Cache-loss change, Stage2B
implementation, or candidate-level Exact-Q enumeration was performed.

```
EPISODES_ANALYZED = 4
TRANSITIONS_ANALYZED = 32
COUNTERFACTUAL_BRANCHES_EXECUTED = 288
TOTAL_RUNTIME_SECONDS ~= 104.1

DR_DQ_SIGN_AGREEMENT_RATE = 0.552083 (159/288)
DR_DQ_SIGN_MISMATCH_RATE = 0.447917 (129/288)
GLOBAL_VS_DR_SIGN_MISMATCH_RATE = 0.517361 (149/288)
GLOBAL_VS_DQ_SIGN_MISMATCH_RATE = 0.430556 (124/288)
HEAD_MIXED_SIGN_RATE_DR = 1.000000 (32/32)
HEAD_MIXED_SIGN_RATE_DQ = 1.000000 (32/32)
TD_Q_CREDIT_RELIABLE = YES

CACHE_TRIGGER_CREDIT_DESIGN = INCONCLUSIVE
CACHE_CANDIDATE_CREDIT_DESIGN = DEFERRED

TRAINING_EXECUTED = NO
COMMIT = NO
PUSH = NO
```

The one-step TD-Q difference is directionally more aligned with the current
global GAE than the raw reward difference in this sample (43.1% versus 51.7%
global-sign mismatch), and it retains per-head sign differences.  However, the
critic term materially amplifies the scale and tail of the counterfactual
credit rather than stabilizing it: TD-Q absolute P90/max are 16.90/76.50,
versus 4.40/44.52 for raw reward.  Together with a 44.8% raw-vs-TD-Q sign
mismatch, this does not establish TD-Q as a stable replacement.  Raw reward is
less tail-heavy but is less aligned with the current global credit.  Therefore
the design choice remains `INCONCLUSIVE`; no credit implementation should be
started from this diagnosis alone.

## Scope and method

The diagnosis replayed the following existing train-only realization
directories:

- `environment_data/proposed_joint_training/episode_20100`
- `environment_data/proposed_joint_training/episode_20107`
- `environment_data/proposed_joint_training/episode_20108`
- `environment_data/proposed_joint_training/episode_20109`

The current validation checkpoint used for critic forward evaluation was
`runs/stage1b_cross_branch_short_gate/latest_checkpoint.pt`.  The replay used
the existing environment, Matching, KKT, and reward path.  It did not generate
new environment/task/prediction/scenario data.

For each episode, eight representative transitions were selected from an
80-slot prefix: four with positive and four with negative normalized global
GAE.  The selection covers both actual `KEEP` and `UPDATE` trigger actions.
For every selected transition and each of the nine Cache heads:

- actual `UPDATE` was compared with `KEEP`;
- actual `KEEP` was compared with one legal representative `UPDATE`, chosen as
  the current policy's highest-probability legal trigger candidate;
- the other Cache heads, UAV action, realization, state/prediction, and
  Matching/KKT configuration were held fixed;
- no full candidate enumeration was performed.

The diagnostic quantities were calculated as:

```
D_R = R_actual - R_counterfactual
D_Q = [R_actual + gamma * V(s'_actual)]
      - [R_counterfactual + gamma * V(s'_counterfactual)]
```

The critic was used only for forward passes.  Sign comparisons against global
credit use the same centered/standardized global-GAE sign used by the actor
loss in the replay prefix.  `D_R` and `D_Q` are local one-step
counterfactual diagnostics, not full-horizon advantages:

```
LOCAL_CREDIT_IS_FULL_HORIZON_ADVANTAGE = NO
```

All 288 counterfactual branches completed without an execution error or an
unavailable reference branch.

## Coverage

| episode | selected slots | positive global GAE | negative global GAE | actual UPDATE heads | actual KEEP heads |
|---:|---|---:|---:|---:|---:|
| 20100 | 6, 10, 21, 42, 54, 62, 66, 68 | 4 | 4 | 39 | 33 |
| 20107 | 14, 34, 37, 41, 43, 69, 71, 78 | 4 | 4 | 37 | 35 |
| 20108 | 4, 7, 27, 48, 57, 60, 61, 74 | 4 | 4 | 38 | 34 |
| 20109 | 14, 41, 44, 47, 53, 65, 67, 72 | 4 | 4 | 39 | 33 |
| **total** | **32 transitions** | **16** | **16** | **153** | **135** |

## Sign consistency

Overall sign counts:

| comparison | count | rate |
|---|---:|---:|
| `D_R` and `D_Q` same sign | 159/288 | 0.552083 |
| `D_R > 0`, `D_Q < 0` | 67/288 | 0.232639 |
| `D_R < 0`, `D_Q > 0` | 62/288 | 0.215278 |
| global GAE and `D_R` different sign | 149/288 | 0.517361 |
| global GAE and `D_Q` different sign | 124/288 | 0.430556 |

The trigger split is:

| actual trigger | n | DR/DQ agreement | DR/DQ mismatch | `D_R+`, `D_Q-` | `D_R-`, `D_Q+` | global/DR mismatch | global/DQ mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|
| KEEP | 135 | 74/135 (0.548148) | 61/135 (0.451852) | 54 | 7 | 63/135 (0.466667) | 64/135 (0.474074) |
| UPDATE | 153 | 85/153 (0.555556) | 68/153 (0.444444) | 13 | 55 | 86/153 (0.562092) | 60/153 (0.392157) |
| **all** | **288** | **159/288 (0.552083)** | **129/288 (0.447917)** | **67** | **62** | **149/288 (0.517361)** | **124/288 (0.430556)** |

Both local-credit definitions preserve head-level heterogeneity.  Every one of
the 32 selected transitions had at least one positive and one negative head
credit under each definition:

```
HEAD_MIXED_SIGN_RATE_DR = 32/32 = 1.000000
HEAD_MIXED_SIGN_RATE_DQ = 32/32 = 1.000000
```

Per-head sign results (`n=32` counterfactuals per head) are:

| head | DR/DQ agreement | DR/DQ mismatch | `D_R+`, `D_Q-` | `D_R-`, `D_Q+` | global/DR mismatch | global/DQ mismatch |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 18/32 (0.5625) | 14/32 (0.4375) | 8 | 6 | 18/32 (0.5625) | 20/32 (0.6250) |
| 1 | 22/32 (0.6875) | 10/32 (0.3125) | 3 | 7 | 16/32 (0.5000) | 12/32 (0.3750) |
| 2 | 18/32 (0.5625) | 14/32 (0.4375) | 8 | 6 | 12/32 (0.3750) | 12/32 (0.3750) |
| 3 | 17/32 (0.5313) | 15/32 (0.4688) | 8 | 7 | 22/32 (0.6875) | 11/32 (0.3438) |
| 4 | 18/32 (0.5625) | 14/32 (0.4375) | 5 | 9 | 21/32 (0.6563) | 13/32 (0.4063) |
| 5 | 17/32 (0.5313) | 15/32 (0.4688) | 12 | 3 | 15/32 (0.4688) | 18/32 (0.5625) |
| 6 | 15/32 (0.4688) | 17/32 (0.5313) | 5 | 12 | 17/32 (0.5313) | 10/32 (0.3125) |
| 7 | 19/32 (0.5938) | 13/32 (0.4063) | 8 | 5 | 12/32 (0.3750) | 11/32 (0.3438) |
| 8 | 15/32 (0.4688) | 17/32 (0.5313) | 10 | 7 | 16/32 (0.5000) | 17/32 (0.5313) |

## Credit scale and tails

The following statistics are over all 288 head-level counterfactuals.

| statistic | `D_R` | `D_Q` |
|---|---:|---:|
| mean | -1.137088 | -2.953734 |
| std | 6.038616 | 11.472201 |
| P10 | -3.080266 | -15.032447 |
| P50 | -0.020304 | -1.507088 |
| P90 | 0.333993 | 7.557254 |
| min | -44.522753 | -76.501305 |
| max | 24.794955 | 39.946047 |
| ABS_CREDIT_P90 | 4.404419 | 16.902631 |
| ABS_CREDIT_MAX | 44.522753 | 76.501305 |

The trigger-specific scale is:

| trigger | credit | mean | std | P10 | P50 | P90 | min | max | ABS P90 | ABS max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KEEP | `D_R` | -1.448120 | 5.903998 | -4.809962 | 0.065647 | 0.219226 | -42.285957 | 13.938184 | 6.599961 | 42.285957 |
| KEEP | `D_Q` | -5.194974 | 10.815975 | -17.047451 | -3.218267 | 4.169530 | -49.331049 | 21.739876 | 18.655185 | 49.331049 |
| UPDATE | `D_R` | -0.862648 | 6.141885 | -1.676991 | -0.075958 | 0.522288 | -44.522753 | 24.794955 | 3.291986 | 44.522753 |
| UPDATE | `D_Q` | -0.976170 | 11.669354 | -12.195428 | -0.074308 | 8.933355 | -76.501305 | 39.946047 | 15.697523 | 76.501305 |

The TD-Q critic term therefore does not mitigate the extreme reward tail in
this sample.  It approximately doubles the standard deviation (11.47 versus
6.04), raises absolute P90 (16.90 versus 4.40), and raises absolute maximum
(76.50 versus 44.52).  This is the main reason the result is not promoted to a
TD-Q design decision despite its lower mismatch with global GAE.

## Critic reliability

The current checkpoint critic was evaluated on 320 successor states: one
actual successor and one counterfactual successor per selected transition/head.

| measure | value |
|---|---:|
| finite rate | 320/320 = 1.000000 |
| value mean | 415.958071 |
| value std | 33.665643 |
| value min | 335.764984 |
| value max | 482.572387 |
| max absolute value | 482.572387 |

Therefore:

```
TD_Q_CREDIT_RELIABLE = YES
```

Here `YES` means finite and numerically stable for this sampled replay, not
that the critic is calibrated or that the one-step TD-Q estimate is a proven
full-horizon credit assignment.  Critic quality uncertainty remains a
limitation of this small diagnostic.

## Interpretation and boundary

The data support the previously established conclusion that broadcasting one
global GAE to all nine Cache heads is coarse: all 32 sampled transitions show
mixed-sign per-head local credit under both definitions.  One-step TD-Q is
closer to global GAE in sign, especially for actual `UPDATE` branches, but it
does not remove the head-level conflict and is substantially more tail-heavy.
Raw reward difference is numerically smaller but has more global-sign
mismatch.  The two definitions disagree on 129 of 288 branches, so this
sample cannot safely select one as the Stage2B trigger credit without a further
design/validation decision.

This task only addresses the trigger `KEEP`/`UPDATE` credit boundary.
Candidate-level credit remains deliberately undecided:

```
CACHE_CANDIDATE_CREDIT_DESIGN = DEFERRED
```

No conclusion here authorizes implementation of per-head credit, COMA,
candidate Exact-Q, a new loss, or any training change.

## Execution and cleanup status

```
TRAINING_EXECUTED = NO
COMMIT = NO
PUSH = NO
```

No existing run artifact, checkpoint, metrics file, environment data, or
prediction artifact was modified.
