# Stage2A Cache per-head PPO ratio

本阶段只改变 proposed joint PPO 的 Cache ratio granularity；没有启动 episode、optimizer step、checkpoint resume 或任何训练。已有 run artifact 未被修改。

## Required status

```text
CACHE_CREDIT_SOURCE_CHANGED = NO
CACHE_ADVANTAGE_SOURCE = GLOBAL_GAE

CACHE_RATIO_GRANULARITY_OLD = WHOLE_CACHE_JOINT
CACHE_RATIO_GRANULARITY_NEW = PER_CACHE_HEAD

KEEP_LOGP_RULE = log P_trigger(KEEP)
UPDATE_LOGP_RULE = log P_trigger(UPDATE) + log P_candidate(selected_candidate | UPDATE)

WHOLE_CACHE_JOINT_RATIO_USED_FOR_LOSS = NO
WHOLE_CACHE_JOINT_RATIO_DIAGNOSTIC_ONLY = YES

OLD_CACHE_LOSS_REDUCTION = mean over transition t of the clipped PPO surrogate using exp(sum_m(new_logp_tm - old_logp_tm))
NEW_CACHE_LOSS_REDUCTION = mean over transition t and Cache head m of the clipped PPO surrogate using exp(new_logp_tm - old_logp_tm)

SYNTHETIC_CASE_1 = PASS
SYNTHETIC_CASE_2 = PASS
SYNTHETIC_CASE_3 = PASS
SYNTHETIC_CASE_4 = PASS
SYNTHETIC_CASE_5 = PASS
SYNTHETIC_CASE_6 = PASS

READ_ONLY_RATIO_COMPARISON = INCONCLUSIVE
WHOLE_JOINT_CLIP_FRACTION = AVAILABLE_AS_EXISTING_CACHE_CLIP_FRACTION_ONLY
PER_HEAD_CLIP_FRACTION = INCONCLUSIVE_FOR_EXISTING_ARTIFACTS

UAV_STAGE1_CHANGED = NO
REWARD_CHANGED = NO
OBJECTIVE_CHANGED = NO
LR_CHANGED = NO
MATCHING_CHANGED = NO
KKT_CHANGED = NO

PY_COMPILE = PASS
GIT_DIFF_CHECK = PASS

TRAINING_EXECUTED = NO
COMMIT = NO
PUSH = NO

STAGE2A_READY_FOR_SHORT_TRAIN = YES
```

## Implementation and invariants

- `configs/proposed/system_config.toml` now selects `PER_CACHE_HEAD` for both Cache ratio fields. `WHOLE_CACHE_JOINT` remains accepted by config validation and remains the trainer fallback for old configs/checkpoint resumes.
- The proposed policy already computes the exact selected per-head Cache log-probability. The trainer now uses that value with the rollout's per-head old operation and candidate log-probabilities. KEEP has a zero candidate contribution; UPDATE includes the selected candidate contribution. The evaluator's fixed candidate mask is used for both old/new diagnostics, so padded candidates cannot enter the distribution.
- The proposed Cache actor loss is the mean over `(transition, Cache head)` of the clipped surrogate with the unchanged global GAE advantage. No per-head Q, counterfactual credit, teacher, temperature, reward, or learning-rate change was introduced. The existing `0.5` Cache/UAV branch scale is unchanged.
- The old whole-cache joint ratio and clip fraction are still computed and emitted as diagnostics. New future `update_metrics.jsonl` records also include `CACHE_PER_HEAD_OLD_NEW_KL_MEAN/P50/P90/MAX`, `CACHE_PER_HEAD_RATIO_MEAN/P90/MAX`, `CACHE_PER_HEAD_CLIP_FRACTION`, `CACHE_RATIO_GRANULARITY`, and the two whole-joint usage markers.
- At each update start, full masked Cache operation/candidate log-probabilities are detached and cloned. At the end of each PPO update epoch, the same rollout tensors and masks are reevaluated without environment, Matching, KKT, or data regeneration. The per-head old-to-new KL is over the atomic action space `KEEP` plus legal `UPDATE` candidates.

The relevant source locations are `src/ppo_simple/policy.py:60-67, 1246-1247, 2064-2065`, `src/ppo_simple/trainer.py:1825-1945, 8250-8375, 9280-9330`, and `src/common/project_config.py:927-950`.

## Synthetic validation

The six required cases were run in-memory with the project conda interpreter:

1. all nine head ratios equal one, with the theoretical loss value;
2. one changed/clipped head and eight unchanged heads, while the legacy product ratio changed the whole transition;
3. opposite head changes, demonstrating that per-head reduction does not cancel them through a product;
4. KEEP log-probability contains only the trigger term;
5. UPDATE log-probability contains trigger plus selected candidate;
6. invalid/padded candidates remain excluded by the mask.

An additional small randomly initialized proposed policy evaluation confirmed the KEEP/UPDATE alignment and masked candidate behavior.

## Existing artifact read-only check

The existing proposed and Stage1 artifacts were read only. Their manifests still report `WHOLE_CACHE_JOINT`, and their `update_metrics.jsonl` files contain no Stage2A per-head fields; they were not rewritten. The existing `cache_clip_fraction` is available as the legacy whole-joint diagnostic. For reference, the Stage1 artifact's epochs 1–121 have whole-joint `cache_clip_fraction` mean `0.1484848526`, P90 `0.2041666728`, and max `0.2408333333`. A lossless per-head ratio/clip comparison cannot be recovered from those artifacts because the full per-head old distributions were not stored, so `READ_ONLY_RATIO_COMPARISON` remains `INCONCLUSIVE`.

No full PPO test suite or formal/diagnostic training run was started. The requested next step, if authorized separately, is a short train-only validation run using the new configuration; this report does not claim that run has been performed.
