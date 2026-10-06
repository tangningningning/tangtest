# Cache Full-Information Teacher Gate

## Implementation

`evaluate_cache_full_information_values` is the single Cache value evaluator used by Trigger, Candidate, and the causal diagnostic. For every selected state and Cache head, it evaluates `Q_KEEP` and every legal complete UPDATE target with the existing production branch executor, then reuses identical physical branches. Values use `reward + gamma * frozen_critic(next_observation)` with `FIXED_DOWNSTREAM`; the deterministic policy-consistent replay remains diagnostic only.

The Trigger teacher is `softmax([Q_KEEP, Q_BEST_UPDATE] / 1.0)`. It does not read the target head's sampled action. The Candidate teacher is `softmax(Q_UPDATE / 1.0)` over that head's legal candidates. Both losses are detached-teacher KLs reduced only across rows with legal credit. Heads without an UPDATE candidate receive `[1, 0]` Trigger teacher probabilities and are excluded from the Trigger comparison reduction. Candidate KL runs for every head with legal candidates, including a KEEP rollout.

The PPO Cache branch no longer uses sampled-action TD-Q or a candidate loss gated on sampled UPDATE. The proposed config and manifest identify the same sources and `FULL_EXACT` / `FIXED_DOWNSTREAM` modes. Proposed-joint checkpoints now require version 18; the epoch-233 version-17 actor is diagnostic input only and is rejected by `load_checkpoint`. Entropy remains `0.0`. UAV teacher, UAV losses, reward, Matching, KKT, system model, and learning rates were not changed.

Retired Cache credit selectors, candidate-policy mode selection, ratio-granularity configuration/metadata, and the ratio-only checkpoint-resume compatibility exception were removed. Remaining per-head probability-ratio summaries are observational update diagnostics; they do not select or define the Cache teacher loss.

## Synthetic semantics

With `Q_KEEP=10` and candidate Q `[5, 12, 8]`, the evaluator semantics produce best index 1, `Q_BEST_UPDATE=12`, `DELTA_BEST=+2`, Trigger UPDATE probability `0.880797`, and Candidate teacher top1 index 1. With `Q_KEEP=15`, `DELTA_BEST=-3` and Trigger KEEP probability is `0.952574`. **PASS**.

## Real checkpoint gate

Checkpoint: `runs/stage2d_v3_fresh300_actor_lr_5e_minus_4/checkpoints/latest_checkpoint.pt`, epoch 233, SHA256 `2ff73b63e327f2293e0ad0858864725b29607fcbd00ae23d958e5af970ebdd9e`. The stochastic rollout used seed `73019`; the bounded full-exact batch selected slot 298 and evaluated all nine Cache heads. It covered 122 legal UPDATE configurations plus nine KEEP references. Identical physical branches were reused, resulting in 123 unique environment executions.

| Head | Rollout | Legal updates | Q_KEEP | Q_BEST_UPDATE | DELTA_BEST | Best index | Trigger teacher KEEP / UPDATE | Candidate teacher top1 / P(best) | Old Trigger KEEP / UPDATE | Old Candidate top1 / P(exact best) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 / 46 | KEEP | 8 | -0.061658 | -45.347404 | -45.285747 | 0 | 1.000000 / 2.15e-20 | 0 / 0.534123 | 0.996374 / 0.003626 | 6 / 0.003106 |
| 1 / 10 | KEEP | 9 | -0.061658 | 5.185748 | 5.247406 | 1 | 0.005234 / 0.994766 | 1 / 1.000000 | 0.998366 / 0.001634 | 6 / 0.002332 |
| 2 / 0 | KEEP | 8 | -0.061658 | -18.185905 | -18.124248 | 0 | 1.000000 / 1.35e-08 | 0 / 1.000000 | 0.999011 / 0.000989 | 3 / 0.337206 |
| 3 / 23 | KEEP | 9 | -0.061658 | 8.971668 | 9.033326 | 2 | 0.000119 / 0.999881 | 2 / 0.997971 | 0.999964 / 0.000036 | 6 / 0.003604 |
| 4 / 59 | KEEP | 7 | -0.061658 | 7.493207 | 7.554865 | 3 | 0.000523 / 0.999477 | 3 / 0.828108 | 0.992785 / 0.007215 | 0 / 0.004847 |
| 5 / RSU_0 | UPDATE | 21 | -0.754250 | 13.814136 | 14.568386 | 17 | 4.71e-07 / 0.9999995 | 17 / 0.992549 | 0.026546 / 0.973454 | 13 / 0.000143 |
| 6 / RSU_1 | UPDATE | 32 | 4.152884 | 13.977251 | 9.824367 | 28 | 0.000054 / 0.999946 | 28 / 0.978839 | 0.059213 / 0.940787 | 10 / 0.000189 |
| 7 / UAV_0 | KEEP | 16 | -0.061658 | 1.848108 | 1.909766 | 6 | 0.129007 / 0.870993 | 6 / 0.999990 | 0.992274 / 0.007726 | 13 / 0.003551 |
| 8 / UAV_1 | KEEP | 12 | -0.061658 | 83.165077 | 83.226735 | 4 | 7.16e-37 / 1.000000 | 4 / 1.000000 | 0.998979 / 0.001020 | 9 / 0.001041 |

For all 9 heads, `best_candidate_index` matched the exact Q argmax and Candidate teacher top1 matched the exact Q argmax. Seven heads had legal UPDATEs while the rollout sampled KEEP. On head 0 at slot 298, flipping the sampled action from KEEP to UPDATE left Q_KEEP, candidate mask/order, all candidate Qs, and Trigger teacher unchanged. The actual KEEP head's Candidate KL was differentiable and produced a nonzero Candidate scorer gradient (absolute gradient sum `391.2745`); no optimizer step was taken.

## Validation and cleanup

The one-transition × one-head smoke used epoch 233 at slot 84 on `RSU_0`: 33 legal candidate branches plus KEEP, **PASS**. The one-transition × nine-head gate took 2.32 seconds for the full-information evaluator. Old checkpoint resume was attempted against the new trainer and rejected (`checkpoint_version=17`, required `18`), **PASS**. Six touched Python files compiled, synthetic semantics passed, and `git diff --check` passed. No formal training, optimizer step, benchmark sweep, or checkpoint write occurred.

## Required summary

```text
TRAINING_EXECUTED = NO

CACHE_TRIGGER_CREDIT_SOURCE = FULL_INFORMATION_BEST_UPDATE_VALUE_TEACHER
CACHE_CANDIDATE_CREDIT_SOURCE = CANDIDATE_SPECIFIC_VALUE_LISTWISE_TEACHER
CACHE_VALUE_EVALUATOR = FULL_EXACT
CACHE_DOWNSTREAM_MODE = FIXED_DOWNSTREAM

TRIGGER_REFERENCE_ACTION_INDEPENDENT = YES
CANDIDATE_CREDIT_IS_CANDIDATE_SPECIFIC = YES
CANDIDATE_LEARNS_ON_KEEP_ROLLOUT = YES
BEST_UPDATE_MATCHES_FULL_EXACT = YES
CANDIDATE_TEACHER_TOP1_MATCHES_FULL_EXACT = YES

OLD_ONE_STEP_TDQ_PRODUCTION_PATH_REMOVED = YES
OLD_CACHE_GLOBAL_GAE_PRODUCTION_PATH_REMOVED = YES
OBSOLETE_METADATA_REMOVED = YES

CHECKPOINT_COMPATIBILITY = VERSION_18; OLD_VERSION_17_RESUME_REJECTED; FRESH_INITIALIZATION_REQUIRED
NEXT_STEP = HYBRID_VALUE_EVALUATOR_EFFICIENCY_GATE

CHANGED_FILES = configs/proposed/system_config.toml; src/common/project_config.py; src/ppo_simple/cache_credit.py; src/ppo_simple/formal_evaluation.py; src/ppo_simple/reporting.py; src/ppo_simple/trainer.py; scripts/training/fresh300_runner_common.py; reports/cache_full_information_teacher_gate.md
DELETED_OBSOLETE_SOURCE_FILES = NONE
VALIDATIONS = touched Python compile PASS; synthetic teacher semantics PASS; 1 transition x 1 head full-exact smoke PASS; real epoch-233 1 transition x 9 head full-exact gate PASS; KEEP Candidate gradient PASS; Trigger action-independence flip PASS; old checkpoint resume rejection PASS
TEMP_FILES_REMOVED = YES; no temporary test scripts/files were created; __pycache__ cleanup PASS
GIT_DIFF_CHECK = PASS
COMMIT = NO
PUSH = NO
```
