# Stage2B2 short-horizon bootstrap-sign check and trigger-credit diagnosis

## Final status

The source audit shows that the current PPO critic is a reward-to-go critic and
uses an additive bootstrap.  The previous Stage2B2 diagnostic used a minus
sign in `Q_H`; that sign was inconsistent with the source and has been
corrected.  The H=3/H=5 replay below uses the same episodes, slots, heads, and
frozen-policy paths as the previous Stage2B2 run.

```
CRITIC_VALUE_SEMANTICS = REWARD_TO_GO
GAE_DELTA_FORMULA = delta_t = r_t + gamma * (1-done_t) * V(s_{t+1}) - V(s_t)
ONE_STEP_Q_BOOTSTRAP_SIGN = PLUS
SHORT_HORIZON_BOOTSTRAP_SIGN_REQUIRED = PLUS
STAGE2B2_CURRENT_SIGN = MINUS (previous diagnostic)
STAGE2B2_BOOTSTRAP_SIGN_CORRECT = NO

EPISODES_ANALYZED = 4
TRANSITIONS_ANALYZED = 16
COUNTERFACTUAL_HEADS_ANALYZED = 144

DR_VS_DH_SIGN_AGREEMENT_RATE = H3: 0.541667; H5: 0.527778
DQ_VS_DH_SIGN_AGREEMENT_RATE = H3: 0.777778; H5: 0.722222
DR_VS_DH_SPEARMAN = H3: 0.178909; H5: 0.129000
DQ_VS_DH_SPEARMAN = H3: 0.511679; H5: 0.492067
DH3_DH5_SIGN_AGREEMENT_RATE = 0.805556
DH3_DH5_SPEARMAN = 0.620280

FULL_PER_HEAD_COUNTERFACTUAL_TRAINING_FEASIBLE = NO
CACHE_TRIGGER_CREDIT_DESIGN = ONE_STEP_TD_Q_DIFFERENCE
CACHE_CANDIDATE_CREDIT_DESIGN = DEFERRED

TRAINING_EXECUTED = NO
COMMIT = NO
PUSH = NO
```

The current formal algorithm remains unchanged: `Cache credit = GLOBAL_GAE`,
`Cache ratio = PER_CACHE_HEAD`, and UAV candidate KL gate remains
`MEAN_OR_P90`.

## Source-level sign audit

### Reward and critic semantics

The rollout reward is `result.system_utility`, checked as:

```
system_utility = actual_task_utility - cache_cost - movement_cost
```

This is a reward scalar; the costs are already subtracted before the value
function sees the reward.  See
`src/ppo_simple/rollout.py:161-167`.

`compute_gae` in `src/ppo_simple/rollout.py:855-881` uses:

```
next_value = bootstrap_value                 # final nonterminal transition
           or values[t+1]

delta_t = reward_t + gamma * nonterminal * next_value - value_t
A_t     = delta_t + gamma * lambda * nonterminal * A_{t+1}
return_t = A_t + value_t
```

`PPOV2Trainer.update()` obtains the rollout bootstrap value from the current
policy critic and passes it to `compute_gae` (`src/ppo_simple/trainer.py:7218-7227`).
The critic target is the resulting return, optionally transformed into PopArt
normalized space; it is not a cost-to-go target
(`src/ppo_simple/trainer.py:7827-7830`).

Therefore:

```
CRITIC_VALUE_SEMANTICS = REWARD_TO_GO
GAE_DELTA_FORMULA = r_t + gamma * (1-done_t) * V(s_{t+1}) - V(s_t)
```

### Existing one-step Exact-Q / TD-Q convention

The existing Cache Exact-Q branch evaluator constructs:

```
q_values = branch_reward + gamma * next_value
```

at `src/ppo_simple/cache_credit.py:488-502`.  The UAV Exact-Q path uses the
same convention in both serial and parallel implementations:

- `src/ppo_simple/trainer.py:5636-5637`
- `src/ppo_simple/trainer.py:6394-6395`

The UAV operation branch values therefore use `PLUS`, not `MINUS`.

### Required short-horizon sign

Because `V(s_{t+H})` estimates the remaining discounted reward-to-go after the
H explicitly evaluated rewards, the compatible short-horizon reference is:

```
Q_H = sum(k=0..H-1, gamma^k * R_(t+k)) + gamma^H * V(s_(t+H))
D_H = Q_H_actual - Q_H_counterfactual
```

Thus:

```
ONE_STEP_Q_BOOTSTRAP_SIGN = PLUS
SHORT_HORIZON_BOOTSTRAP_SIGN_REQUIRED = PLUS
STAGE2B2_CURRENT_SIGN = MINUS
STAGE2B2_BOOTSTRAP_SIGN_CORRECT = NO
```

The old minus-sign results are superseded; only the plus-sign results in this
report are used for the design conclusion.

## Corrected replay scope

The corrected replay used only existing train-only realizations:

- `environment_data/proposed_joint_training/episode_20100`
- `environment_data/proposed_joint_training/episode_20107`
- `environment_data/proposed_joint_training/episode_20108`
- `environment_data/proposed_joint_training/episode_20109`

The exact fixed representative slots were:

| episode | slots | positive GAE | negative GAE |
|---:|---|---:|---:|
| 20100 | 0, 27, 77, 84 | 2 | 2 |
| 20107 | 5, 9, 78, 84 | 2 | 2 |
| 20108 | 0, 3, 81, 82 | 2 | 2 |
| 20109 | 0, 15, 21, 70 | 2 | 2 |

Each of the 16 transitions was evaluated for all 9 Cache heads.  The actual
`UPDATE` branch was compared with `KEEP`; an actual `KEEP` branch was compared
with one legal representative `UPDATE` target selected by the frozen current
policy.  Other Cache heads, the UAV action at the selected slot, realization,
state/prediction, and Matching/KKT path were unchanged.  No candidate-level
credit table was built.

H=3 values use the first three rewards and state `s_(t+3)`.  H=5 values use
the first five rewards and state `s_(t+5)`.  H=3 is the prefix of the same H=5
paired continuation.  Future actions use deterministic decoding from the
frozen checkpoint, so there is no stochastic common-RNG mismatch.

```
LOCAL_CREDIT_IS_FULL_HORIZON_ADVANTAGE = NO
```

All 144 counterfactual head branches completed successfully.

## Corrected agreement with H=3/H=5

| comparison | overall agreement | overall Spearman | KEEP agreement | KEEP Spearman | UPDATE agreement | UPDATE Spearman |
|---|---:|---:|---:|---:|---:|---:|
| `D_R` vs `D_H3` | 0.541667 | 0.178909 | 0.508772 | 0.302306 | 0.563218 | 0.064461 |
| `D_Q` vs `D_H3` | 0.777778 | 0.511679 | 0.754386 | 0.262639 | 0.793103 | 0.628454 |
| `D_R` vs `D_H5` | 0.527778 | 0.129000 | 0.491228 | 0.206962 | 0.551724 | 0.097056 |
| `D_Q` vs `D_H5` | 0.722222 | 0.492067 | 0.666667 | 0.332447 | 0.758621 | 0.622111 |
| `D_H3` vs `D_H5` | 0.805556 | 0.620280 | 0.807018 | 0.564882 | 0.804598 | 0.603230 |

Corrected required fields:

```
DR_VS_DH_SIGN_AGREEMENT_RATE = H3 0.5416666667; H5 0.5277777778
DQ_VS_DH_SIGN_AGREEMENT_RATE = H3 0.7777777778; H5 0.7222222222
DR_VS_DH_SPEARMAN = H3 0.1789087929; H5 0.1290002968
DQ_VS_DH_SPEARMAN = H3 0.5116791255; H5 0.4920665541
DH3_DH5_SIGN_AGREEMENT_RATE = 0.8055555556
DH3_DH5_SPEARMAN = 0.6202797203
```

Per-episode sign agreement is:

| episode | `D_R`/`D_H3` | `D_Q`/`D_H3` | `D_R`/`D_H5` | `D_Q`/`D_H5` | `D_H3`/`D_H5` |
|---:|---:|---:|---:|---:|---:|
| 20100 | 0.555556 | 0.555556 | 0.583333 | 0.694444 | 0.805556 |
| 20107 | 0.555556 | 0.888889 | 0.527778 | 0.861111 | 0.916667 |
| 20108 | 0.555556 | 0.833333 | 0.444444 | 0.722222 | 0.777778 |
| 20109 | 0.500000 | 0.833333 | 0.555556 | 0.611111 | 0.722222 |

TD-Q is at least as good as raw reward on the H=3 comparison in every episode
and better on H=5 in every episode.  Its rank correlation is also higher in
every episode for both horizons.

## Corrected credit scale

All values below are over 144 head-level branches and use the required plus
bootstrap.

| statistic | `D_R` | `D_Q` | `D_H3` | `D_H5` |
|---|---:|---:|---:|---:|
| mean | -0.037582 | -1.087081 | -3.157167 | -0.924777 |
| std | 5.044941 | 10.666502 | 16.866348 | 16.374726 |
| P10 | -1.932445 | -11.657058 | -25.392549 | -16.389368 |
| P50 | -0.040153 | -1.662154 | -1.170072 | -1.745676 |
| P90 | 0.687509 | 10.380041 | 14.533981 | 18.865065 |
| min | -18.469254 | -31.838117 | -59.433911 | -73.446201 |
| max | 48.712402 | 66.276855 | 57.023276 | 50.246872 |
| ABS_P90 | 4.499052 | 15.434301 | 29.496900 | 25.765255 |
| ABS_MAX | 48.712402 | 66.276855 | 59.433911 | 73.446201 |

`D_Q` is more dispersed than `D_R`, but it is less tail-heavy than the
short-horizon reference itself at ABS_P90 for both H=3/H=5 and has a lower
ABS_MAX than H=5.  The cost is larger scale than raw reward, so any future
implementation would still need an explicit scale design; this report does
not implement one.

## Global GAE context only

Global GAE is not used as the primary judge here.  The corrected sign mismatch
rates are included only for context:

| local quantity | overall mismatch vs global GAE | KEEP | UPDATE |
|---|---:|---:|---:|
| `D_R` | 0.465278 | 0.508772 | 0.436782 |
| `D_Q` | 0.493056 | 0.614035 | 0.413793 |
| `D_H3` | 0.395833 | 0.403509 | 0.390805 |
| `D_H5` | 0.395833 | 0.350877 | 0.425287 |

These values do not override the H=3/H=5 comparison because global GAE was
already diagnosed as coarse at head level.

## Runtime cost

The corrected replay executed:

```
INITIAL_REPLAY_STEPS = 340
ACTUAL_PAIRED_HORIZON_STEPS = 80
COUNTERFACTUAL_HORIZON_STEPS = 720
HORIZON_STEPS_EXECUTED = 1140

TOTAL_RUNTIME_SECONDS = 163.5873
COUNTERFACTUAL_RUNTIME_SECONDS = 72.1418
SECONDS_PER_HEAD_COUNTERFACTUAL = 0.5010
PROJECTED_300_SLOT_9_HEAD_RUNTIME = 1352.6596 seconds ~= 22.5 minutes
```

The projection is a rough single-process H=5 estimate for one 300-slot
episode with 9 heads.  It excludes formal training overhead and does not
assume parallel speedup.  A direct full per-head counterfactual inside formal
training remains too expensive:

```
FULL_PER_HEAD_COUNTERFACTUAL_TRAINING_FEASIBLE = NO
```

The short-horizon reference is supported as an offline formula/audit anchor,
not as a direct training target implementation:

```
SHORT_HORIZON_REFERENCE_SUPPORTED = YES
DIRECT_SHORT_HORIZON_TRAINING_FEASIBLE = NO
```

## Final design judgment

After correcting the bootstrap sign to match the actual reward-to-go critic
convention, one-step TD-Q is consistently more aligned with the short-horizon
counterfactual reference than raw reward difference, with higher sign
agreement and rank correlation across the overall sample, trigger strata, and
episodes.  Its scale is larger than `D_R`, but is not larger than the
short-horizon reference tail in this sample.

```
CACHE_TRIGGER_CREDIT_DESIGN = ONE_STEP_TD_Q_DIFFERENCE
CACHE_CANDIDATE_CREDIT_DESIGN = DEFERRED
```

This remains a diagnosis only.  It does not authorize changing Cache loss,
GLOBAL_GAE, reward, objective, LR, optimizer, Matching/KKT, or any formal run.

## Execution and cleanup

```
TRAINING_EXECUTED = NO
COMMIT = NO
PUSH = NO
```

The corrected replay used no new data and no formal-run writes.  The temporary
sign-check script and generated Python bytecode are removed after this report.
