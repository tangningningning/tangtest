# UAV candidate policy collapse root-cause and sanity gates

分析基准是当前工作区 `adaptive_cache_mec` 的同一 Git 工作树。用户指定的
`/home/9900299/tn/adaptive_cache_mec_server` 在本机没有挂载，当前工作区中的
源码、`runs/` artifact 与该路径记录的 Git 内容一致；没有启动、停止、续训或
修改任何正式 run。只使用了已有 train seed `4401` 的配置和 artifact，没有使用
dev/final unseen seed，也没有重新生成任何数据。

## Required conclusions

LR_SINGLE_VARIABLE_CONFIRMED = YES

ACTOR_SINGLE_PARAM_GROUP = YES

UAV_CANDIDATE_SCORER_SHARED_ACTOR_LR = YES

UAV_ENCODER_SHARED_ACTOR_LR = YES

MATCHING_MAX_UPDATE_HIT_PROBLEM = NO

KKT_CORRECTNESS_GATE = PASS

CANDIDATE_MASK_BUG = NO

CANDIDATE_LOSS_SCALING_BUG = NO

CANDIDATE_GRADIENT_ACCUMULATION_BUG = NO

UAV_CANDIDATE_COLLAPSE_PRIMARY_CAUSE = Leading evidence points to a candidate-scorer/encoder optimization excursion under the shared actor Adam update at epoch 88; the source path does not support a candidate-mask, candidate-loss-scaling, gradient-accumulation, operation-KL-gate, or proven Matching-termination defect. The exact mechanism remains unconfirmed because the existing artifacts predate the candidate-distribution diagnostics.

SAFE_NEXT_CHANGE = NONE beyond the Stage 1 candidate-only stability change; run only a future short train-only validation after code review, with no formal run modification or continuation.

## A. Actor optimizer parameter grouping

`src/ppo_simple/policy.py:1792-1808` (`AdaptiveJointCacheUAVPPOActorCritic.actor_parameters`)
returns one flat parameter tuple in this order:

1. `cache_encoder`
2. `uav_encoder`
3. `cache_trigger_head`
4. `cache_candidate_scorer`
5. `uav_operation_head`
6. `uav_candidate_scorer`

The proposed policy factory is selected by `src/ppo_simple/trainer.py:3250-3275`.
For the separate actor/critic path, `src/ppo_simple/trainer.py:3418-3432` calls
`torch.optim.Adam(actor_parameters, lr=self.effective_actor_learning_rate)` with
no per-module `param_groups`. Therefore the actor optimizer has one parameter
group and the effective actor LR applies to all six modules. A no-training policy
identity check in `adaptive_cache_mec` produced:

```text
groups 1
lr 0.0001
actor_param_count 20
module parameter counts: 4, 2, 4, 4, 2, 4
```

The object still has legacy `actor_head` parameters by inheritance, but their
parameter IDs do not intersect `actor_parameters()`; the direct check returned
`legacy_actor_head_in_actor_optimizer_params False`. The proposed constructor
also removes `cache_operation_head` at `src/ppo_simple/policy.py:1771`.

## B. Exact-Q UAV candidate learning path

The current proposed path is selected by the two run configs with
`policy_schema = "adaptive_joint_cache_uav_ppo"` and
`uav_candidate_advantage_source = "EXACT_Q_LISTWISE"`.

| Check | Result | Code evidence |
|---|---|---|
| Candidate padding/mask is carried through the path | SUPPORTED | Rollout creates a padded tensor and boolean mask; `policy.py:1520-1575` validates shape, selected-index legality, and masks raw candidate logits. |
| Infeasible/padded candidate can enter the candidate softmax | NOT_SUPPORTED | `candidate_logits = raw_scores.masked_fill(~candidate_masks_t, -inf)` in `policy.py:1568-1570`; Exact-Q replay sets `candidate_valid_mask` only for full-branch-feasible candidates. |
| Exact-Q temperature is applied to the target | SUPPORTED | `trainer.py:5825-5863` standardizes legal Q values, divides logits by the positive temperature, then masks and softmaxes. The formal UAV temperature is `0.5`. |
| Listwise target is detached | SUPPORTED | `_exact_q_listwise_teacher` returns `.detach()`; the candidate loss reuses that detached target at `trainer.py:6059-6094`. |
| Candidate loss reduction changes scale with candidate count | NOT_SUPPORTED | The KL is summed over legal candidates and then reduced with `kl[active_heads].mean()`, so candidate count is not an additional row-scale factor. |
| Duplicate backward or cross-minibatch gradient accumulation | NOT_SUPPORTED | Each separate-optimization minibatch calls actor `zero_grad(set_to_none=True)`, performs one `actor_loss.backward()`, clips, and steps once (`trainer.py:7646-7787`). |
| `zero_grad` occurs before every minibatch actor update | SUPPORTED | `trainer.py:7646-7647`; gradients are also cleared before the critic backward at `trainer.py:7847`. |
| `uav_encoder` receives both operation and candidate loss gradients | SUPPORTED | The decoupled policy maps the UAV scorer and operation head to `uav_encoder` (`policy.py:1640-1658`). The proposed branch loss includes operation-teacher KL and candidate listwise KL (`trainer.py:7323-7330`, `7468-7500`). |
| Exact-Q candidate scorer is trained by PPO ratio/clip | NOT_SUPPORTED | In the exact-Q branch `candidate_ratio` is collected for diagnostics, while the loss is `_exact_q_listwise_candidate_loss`; ratio/clip is only used in the non-Exact-Q fallback (`trainer.py:7315-7330`, `7399-7470`). |
| Candidate scorer remains controlled by the branch-specific operation KL gate | SUPPORTED | After an operation-only KL crossing, the code clears only operation-head/encoder gradients and leaves candidate-scorer gradients active (`trainer.py:7660-7678`, `7830-7844`). |
| Candidate scorer and encoder share the same Adam actor LR | SUPPORTED | Both are in the flat actor tuple and the single Adam group described in section A. |
| Old teacher/logits are reused across PPO updates or a graph is retained | NOT_SUPPORTED | Exact-Q tensors are rebuilt from the current rollout inside `update`; current logits are reevaluated per minibatch; the teacher is detached and no backward graph is retained across updates. |

The path is therefore internally consistent on the requested mask, target,
loss-reduction, and gradient-lifecycle checks. This does not prove that the
shared actor Adam update cannot be too large; it only rules out the specific
structural defects above.

## C. Diagnostic logging added for future runs

No existing `update_metrics.jsonl` was edited.  This Stage 1 pass changes only
the trainer/config source and this report; it does not touch any run artifact.
`src/ppo_simple/trainer.py` now adds the requested fields for Exact-Q proposed
updates:

- `UAV_CANDIDATE_POLICY_ENTROPY_MEAN`
- `UAV_CANDIDATE_TEACHER_ENTROPY_MEAN`
- `UAV_CANDIDATE_BEST_POLICY_PROB_MEAN`
- `UAV_CANDIDATE_BEST_POLICY_PROB_MEDIAN`
- `UAV_CANDIDATE_BEST_POLICY_RANK_MEAN`
- `UAV_CANDIDATE_BEST_POLICY_TOP1_RATE`
- `UAV_CANDIDATE_BEST_POLICY_TOP3_RATE`
- `UAV_CANDIDATE_LOGIT_STD_MEAN`
- `UAV_CANDIDATE_LOGIT_RANGE_MEAN`

These are post-update policy statistics against the detached Exact-Q teacher;
all statistics select only `candidate_valid_mask == True`. “Best” means the
highest legal Exact-Q candidate, with tied best candidates treated as a set.
Rank is the one-based best-candidate rank after sorting policy probability
descending; top-1/top-3 are set-overlap rates. Logit statistics use the raw
candidate scorer outputs, not padded logits.

For each executed PPO update epoch, the following arrays are also emitted:

- `UAV_CANDIDATE_SCORER_GRADIENT_NORM_BY_UPDATE_EPOCH`
- `UAV_CANDIDATE_SCORER_PARAMETER_DELTA_NORM_BY_UPDATE_EPOCH`
- `UAV_ENCODER_GRADIENT_NORM_BY_UPDATE_EPOCH`
- `UAV_ENCODER_PARAMETER_DELTA_NORM_BY_UPDATE_EPOCH`
- `UAV_CANDIDATE_OLD_NEW_KL_MEAN_BY_UPDATE_EPOCH`
- `UAV_CANDIDATE_OLD_NEW_KL_P90_BY_UPDATE_EPOCH`
- `UAV_CANDIDATE_OLD_NEW_KL_MAX_BY_UPDATE_EPOCH`
- `UAV_CANDIDATE_ARGMAX_FLIP_RATE_BY_UPDATE_EPOCH`
- `UAV_CANDIDATE_BEST_POLICY_PROB_CHANGE_MEAN_BY_UPDATE_EPOCH`
- `UAV_CANDIDATE_BEST_POLICY_PROB_CHANGE_P10_BY_UPDATE_EPOCH`

The final update-level aggregates are emitted as
`UAV_CANDIDATE_OLD_NEW_KL_MEAN/P50/P90/MAX`,
`UAV_CANDIDATE_ARGMAX_FLIP_RATE`, and
`UAV_CANDIDATE_BEST_POLICY_PROB_CHANGE_MEAN/P10`. The update also emits
`UAV_CANDIDATE_OPTIMIZER_STEP_COUNT`,
`UAV_ENCODER_OPTIMIZER_STEP_COUNT`, and
`UAV_CANDIDATE_SCORER_ENCODER_STEP_GAP`.

Gradient entries are the mean pre-clipping norm over minibatches in that PPO
update epoch; parameter deltas compare the epoch-start and epoch-end module
parameters. These additions do not enter the loss or optimizer decisions.

### Candidate trust-region / policy-drift status

`trainer.py` now snapshots the detached legal candidate mask, raw candidate
logits, and masked candidate probabilities at the start of each Exact-Q PPO
update. After every executed PPO update epoch it re-evaluates the same rollout
state, UAV head, candidate feature tensor, and mask under `torch.no_grad()`.
The new KL/argmax/probability-change fields remain diagnostics of candidate
policy drift from the update-start policy. Stage 1 additionally uses the
full-rollout mean of the old-vs-new candidate KL as an epoch-boundary
candidate stop condition; the diagnostic fields themselves are not a KL
penalty, PPO clip, rollback, or minibatch early-stop input. Final KL
percentiles are over legal candidate heads across executed update epochs; KL is computed as
`KL(old_policy || new_policy)` on each legal distribution. `BEST` uses the detached
Exact-Q values used to build the teacher, with tied highest-Q candidates
treated as a set.

`CANDIDATE_PPO_CLIP_ACTIVE = NO`

`CANDIDATE_SPECIFIC_KL_GATE_ACTIVE = YES`

`CANDIDATE_CAN_UPDATE_AFTER_OPERATION_STOP = YES`

`CANDIDATE_SCORER_ENCODER_STEP_DESYNC_POSSIBLE = NO`

`CANDIDATE_PPO_CLIP_ACTIVE = NO` remains a code fact about the Exact-Q
candidate path, not a diagnosis that the absence of PPO clipping is itself a
bug. Stage 1 now adds an epoch-boundary candidate-only KL gate and makes the
UAV encoder active whenever either the operation or candidate branch is active;
the existing `UAV_CANDIDATE_OPTIMIZER_STEP_COUNT`,
`UAV_ENCODER_OPTIMIZER_STEP_COUNT`, and
`UAV_CANDIDATE_SCORER_ENCODER_STEP_GAP` fields remain diagnostics rather than
loss or optimizer controls.

### Stage 1 candidate-stability change

`FILES_CHANGED = src/ppo_simple/trainer.py; configs/proposed/system_config.toml; reports/uav_candidate_collapse_root_cause.md`

The Exact-Q teacher implementation and its per-head legal-candidate
standardization are unchanged. The UAV candidate loss now uses the detached
confidence

```text
confidence = clamp(q_std(legal candidates) / reference_scale, 0, 1)
L_candidate = mean_over_active_heads(confidence * KL(teacher || policy))
```

The reference scale is `0.02`, read from
`algorithm.ppo.uav_candidate_q_reference_scale`; if absent, the code falls
back to `algorithm.ppo.operation_advantage_reference_scale`. A zero legal-Q
spread produces zero confidence and therefore zero listwise scorer gradient.
Cache candidate semantics and the teacher temperature are unchanged.

The candidate-specific drift gate uses the already-captured update-start
policy and the same rollout states, UAV heads, candidate features, and legal
mask used by the old-vs-new diagnostics. It computes
`KL(old_policy || new_policy)` over legal candidate distributions after every
completed PPO update epoch. When the mean is `>=` the effective threshold, the
candidate branch is inactive only for subsequent epochs; it adds no KL penalty,
PPO clip, rollback, or minibatch early stop. The threshold is
`algorithm.ppo.uav_candidate_target_kl` when present, otherwise the existing
`algorithm.ppo.target_kl` (`0.10` in the current proposed configuration).

The active-state contract is:

```text
uav_operation_active = operation gate state
uav_candidate_active = candidate drift-gate state
uav_encoder_active = uav_operation_active OR uav_candidate_active
```

The exact UAV branch keeps the branch scale at `0.5`: both active uses
`0.5 * (operation_loss + candidate_loss)`, operation-only uses
`0.5 * operation_loss`, candidate-only uses `0.5 * candidate_loss`, and both
inactive contributes a graph-connected zero. Candidate scorer and UAV encoder
therefore remain synchronized during candidate-only epochs.

`TEACHER_STANDARDIZATION_CHANGED = NO`

`UAV_CANDIDATE_CONFIDENCE_WEIGHTING = YES`

`CONFIDENCE_FORMULA = clamp(q_std_over_legal_candidates / reference_scale, 0, 1), detached`

`REFERENCE_SCALE = 0.02`

`REFERENCE_SCALE_SOURCE = algorithm.ppo.uav_candidate_q_reference_scale (fallback: algorithm.ppo.operation_advantage_reference_scale)`

`CANDIDATE_SPECIFIC_KL_GATE = YES`

`CANDIDATE_KL_DIRECTION = KL(old_policy || new_policy)`

`CANDIDATE_KL_THRESHOLD = algorithm.ppo.uav_candidate_target_kl if configured, otherwise algorithm.ppo.target_kl = 0.10`

`CANDIDATE_KL_CHECK_GRANULARITY = full rollout legal candidate heads after each completed PPO update epoch; subsequent epochs only`

`UAV_ENCODER_ACTIVE_RULE = uav_operation_active OR uav_candidate_active`

`SCORER_ENCODER_DESYNC_FIXED = YES`

`UAV_BRANCH_LOSS_RULE = both: 0.5*(operation_loss+candidate_loss); operation-only: 0.5*operation_loss; candidate-only: 0.5*candidate_loss; neither: graph-connected zero`

`Q_TARGET_CHANGED = NO`

`OPERATION_TEACHER_CHANGED = NO`

`CACHE_ALGORITHM_CHANGED = NO`

`MATCHING_CHANGED = NO`

`KKT_CHANGED = NO`

`OBJECTIVE_CHANGED = NO`

`LR_CHANGED = NO`

`SYNTHETIC_CASE_1 = PASS`

`SYNTHETIC_CASE_2 = PASS`

`SYNTHETIC_CASE_3 = PASS`

`SYNTHETIC_CASE_4 = PASS`

`SYNTHETIC_CASE_5 = PASS`

`SYNTHETIC_CASE_6 = PASS`

`PY_COMPILE = PASS`

`GIT_DIFF_CHECK = PASS`

`TRAINING_EXECUTED = NO`

`COMMIT = NO`

`PUSH = NO`

`STAGE1_READY_FOR_SHORT_TRAIN = YES`

## D. Matching termination gate

`src/decision/task_allocation.py:964-1011` shows the actual logic:

- the loop is bounded by `range(1, max_updates + 1)`;
- no improving transfer/exchange (`delta_J <= tolerance`) sets
  `stop_reason = "converged_by_no_improvement"`;
- exhausting the loop retains `stop_reason = "stopped_by_max_updates"`;
- only the former sets `transfer_exchange_stable = True`.

Thus `MAX_UPDATES_REACHED` and `NO_IMPROVING_TRANSFER_EXCHANGE` are distinct
termination reasons. `src/decision/slot_environment.py:72-110` now exposes these
as diagnostic aliases without changing the Matching decision logic.

Future update metrics now include the required actual-rollout aggregates:

- `MATCHING_NATURAL_STABLE_COUNT`
- `MATCHING_MAX_UPDATES_HIT_COUNT`
- `MATCHING_MAX_UPDATES_HIT_RATE`
- `MATCHING_UPDATE_COUNT_MEAN`
- `MATCHING_UPDATE_COUNT_P95`
- `MATCHING_UPDATE_COUNT_MAX`

When Exact-Q counterfactual branches are evaluated, separate
`MATCHING_CF_*` fields are emitted. The worker records the original stop reason
and update count for each unique counterfactual Matching call; it does not
change Matching or `max_updates`.

The formal config is `algorithm.matching.max_updates = 100`. A representative
train-only gate was then run over 24 calls from episode seeds 20001–20005,
covering low/middle/high task counts, four extra slots, six actual
configuration evaluations, 14 direct Matching states derived from legal
configuration outputs, and four direct UAV policy-target counterfactuals.
All 24 calls were final-feasible and naturally stable; all 24 passed the
stable check; none reached `max_updates=100`. The update-count mean/P95/max
were 7.5/12/14. The full table and per-call classifications are recorded in
`reports/matching_kkt_sanity_gate.md`.

This closes the representative termination gate only. It does not claim that
all possible Matching states have been exhaustively enumerated.

## E. KKT sanity gate

The formal KKT parameters are `min_cpu_fraction = 1e-6`,
`min_bandwidth_fraction = 1e-6`, `bisection_tolerance = 1e-12`, and
`max_bisection_iterations = 100` (`run_config.toml:510-517`).

The existing solver is reused. `solve_kkt_fraction_allocation` validates finite,
non-negative coefficients, clips allocations to `[epsilon, 1]`, and rejects a
resource sum above one. The representative train-only gate observed 3654 CPU/BW
fraction solves:

```text
closed_form_error = 0.0
per_task_check = True
case_a_feasible = True
case_b_feasible = True
case_c_feasible = True
```

| KKT check | Result | Basis |
|---|---|---|
| CPU fraction feasibility | PASS | 146 observed CPU node allocations were finite, non-negative, and had mean/P95/max sum `1.000000000000032/1.000000000000671/1.000000000000953`. |
| Bandwidth fraction feasibility | PASS | 146 observed bandwidth node allocations were finite, non-negative, and had mean/P95/max sum `0.999999999999950/1.000000000000530/1.000000000000948`. |
| Non-negativity / finite status | PASS | All 3654 diagnostic records and returned allocations passed the finite/non-negative checks. |
| Bisection convergence | PASS on representative gate | 3654/3654 solves converged by tolerance; mean/P95/max iterations were 29.7915/40/44. |
| `max_bisection_iterations` hit count | PASS on representative gate | 0/3654 solves hit the configured maximum of 100 iterations. |
| Resource residual | PASS on representative gate | Mean/max residual were `3.4634e-13/9.9942e-13`. |
| Stationarity residual | PASS on representative gate | Direct multiplier/allocation residual mean/max were `9.7263e-16/5.6843e-14`. |
| Complementary-slackness residual | PASS on representative gate | Direct residual mean/max were `2.1628e-12/1.6289e-10`. |

The resulting `KKT_CORRECTNESS_GATE = PASS` is explicitly a representative
train-only gate, not a mathematical proof for every possible state. No second
solver was written.

## F. LR single-variable check

The complete TOMLs were parsed with `tomllib` and compared recursively by field.
The exact diff is:

```text
algorithm.ppo.actor_learning_rate: 0.0001 -> 0.001
```

No other effective TOML field differs. Both runs have the same policy schema,
branch-specific KL gate, Exact-Q candidate/operation teachers, UAV temperature
`0.5`, critic LR `0.001`, seed `4401`,
`data_protocol = full_episode_randomized`, 300 episode slots, origins 0--299,
`update_epochs = 4`, `num_minibatches = 10`, `minibatch_size = 30`, and
`target_kl = 0.10`. Manifest comparison confirmed these same values and the
same `UAV_CANDIDATE_TEACHER = EXACT_Q_LISTWISE`,
`UAV_OPERATION_TEACHER = FULL_INFORMATION_EXACT_Q_TEACHER`, and
`UAV_TEACHER_UNCHANGED = YES` fields. The manifest's generic
`exact_q_listwise_temperature` field is `1.0` in both manifests and does not
encode the branch-specific UAV temperature; the TOML/code value actually used
for UAV is `0.5` in both runs.

## G. Existing artifact evidence around epoch 87 -> 88

The artifacts are partial/running snapshots: the `1e-4` run has 99 update and
episode records (status `running`, target 300), and the `1e-3` run has 111 of
300. The existing logs predate the new candidate-distribution fields.

For the `1e-4` run:

| epoch | episode return | candidate loss | UAV scorer delta | UAV encoder delta | operation-head delta | operation-only KL | candidate full/only steps | infeasible CF branches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 87 | -349.947 | 0.63155 | 0.20677 | 0.24032 | 0.01957 | 0.03673 | 40 / 0 | 0 |
| 88 | 1518.690 | 0.88816 | 0.72360 | 0.48336 | 0.01938 | 0.03963 | 40 / 0 | 0 |

All four PPO update epochs remained UAV-active in both records; the operation
KL gate did not stop the branch. The scorer delta increased about 3.5x while the
operation-head delta was essentially unchanged. This is consistent with an
optimization excursion in the candidate branch, but the episode return itself
does not show a persistent collapse from 87 to 88, and the old artifact cannot
identify whether the scorer distribution became too sharp, too flat, or
misaligned with the teacher. The new fields are specifically intended to make
that distinction in a future run.

## Verification and scope

- Conda environment used for Python checks: `adaptive_cache_mec`.
- Read-only checks passed: Python compilation of modified modules; synthetic
  old-vs-new drift helper validation including tied-best-Q set semantics;
  static field-chain validation; direct one-group actor-parameter identity
  check; `run_minimal_validation`; `run_slot20_validation`; and the 24-call
  representative Matching/KKT gate.
- The full representative gate is documented in
  `reports/matching_kkt_sanity_gate.md`; it reports `TRAINING_EXECUTED = NO`.
- No formal training was started and no existing run config, checkpoint,
  metrics, environment data, prediction artifact, or scenario artifact was
  modified.
- No temporary analysis file was created.
- No commit or push was performed.
