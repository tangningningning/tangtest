# V10-B Branchwise PPO Refactor

STAGE = V10-B_BRANCHWISE_PPO

This stage adds an additive V10-B policy schema and changes only the PPO actor
surrogate used by that schema. The existing V9 schema, V10-A observability
schema, environment execution, reward computation, action sampling, and
historical run artifacts remain intact.

## CURRENT_STRUCTURE

The current implementation was checked from the V9/V10-A code paths before the
V10-B change.

`sample_and_apply_v9_joint_cache_uav_operation_configuration()` samples the
Cache branch first. The resulting Cache state and selected Cache actions are
then used to construct the conditional UAV contexts. UAV operation decisions
are sampled sequentially, and a MOVE candidate is sampled only when MOVE is
selected. The evaluation path reconstructs these same masked categorical
distributions for the stored actions.

CURRENT_CACHE_LOGPROB_FORMULA = `log_pi_C = sum_h(log_pi_C_operation[h] + indicator(operation[h] != NO_CHANGE) * log_pi_C_candidate[h])`

CURRENT_UAV_LOGPROB_FORMULA = `log_pi_U = sum_u(log_pi_U_operation[u] + indicator(operation[u] == MOVE) * log_pi_U_candidate[u])`, with each UAV context conditioned on the already sampled Cache result and preceding UAV decisions.

CURRENT_JOINT_LOGPROB_FORMULA = `log_pi_joint = log_pi_C + log_pi_U`

CURRENT_JOINT_RATIO_FORMULA = `joint_ratio = exp(new_joint_log_prob - old_joint_log_prob)`

CURRENT_JOINT_PPO_SURROGATE = `L_joint = -mean(min(joint_ratio * A, clip(joint_ratio, 1 - clip_epsilon, 1 + clip_epsilon) * A))`

CURRENT_JOINT_KL_GATE = `mean(old_joint_log_prob - new_joint_log_prob)` is accumulated over the completed minibatches of an update epoch and compared with the existing `target_kl`. The gate remains joint and is unchanged for V10-B.

The existing actor has one shared `encoder`, Cache operation/candidate modules,
and UAV operation/candidate modules. The actual V9/V10 policy has an
independent `critic_encoder` and `critic_head`; it nevertheless uses one
global reward, GAE, and PopArt target path.

## V10_B_STRUCTURE

The V10-B branch log-probabilities are recovered from the same per-head old
log-probabilities already stored by rollout and from the corresponding
evaluation outputs. No extra action is sampled.

CACHE_LOGPROB_FORMULA = `new_cache_log_prob = sum_h(new_cache_operation_log_prob[h] + indicator(operation[h] != NO_CHANGE) * new_cache_candidate_log_prob[h])`

UAV_LOGPROB_FORMULA = `new_uav_log_prob = sum_u(new_uav_operation_log_prob[u] + indicator(operation[u] == MOVE) * new_uav_candidate_log_prob[u])`, evaluated on the stored conditional UAV contexts.

CACHE_RATIO_FORMULA = `cache_ratio = exp(new_cache_log_prob - old_cache_log_prob)`

UAV_RATIO_FORMULA = `uav_ratio = exp(new_uav_log_prob - old_uav_log_prob)`

CACHE_PPO_SURROGATE = `L_cache = -mean(min(cache_ratio * A, clip(cache_ratio, 1 - clip_epsilon, 1 + clip_epsilon) * A))`

UAV_PPO_SURROGATE = `L_uav = -mean(min(uav_ratio * A, clip(uav_ratio, 1 - clip_epsilon, 1 + clip_epsilon) * A))`

COMBINED_ACTOR_LOSS = `actor_policy_loss = 0.5 * (L_cache + L_uav)`

Both branches use the same existing normalized global `A_t` for each minibatch.
The Cache and UAV branch losses do not use the other branch's ratio, so each
branch's clipping is independent. The shared encoder receives the averaged
gradient. There is still exactly one actor optimizer; no branch-specific
optimizer or duplicate Adam state was introduced.

The legacy joint ratio, joint approximate KL, and joint clip fraction are still
computed and logged for compatibility. The joint KL early-stop semantics are
unchanged. V10-B replaces only the actor policy-loss tensor used for the actor
step; critic computation and update remain on the existing path.

## PRESERVED_COMPONENTS

GLOBAL_ADVANTAGE_PRESERVED = YES

GLOBAL_REWARD_PRESERVED = YES

GLOBAL_CRITIC_TARGET_PATH_PRESERVED = YES

SHARED_CRITIC_PRESERVED = YES in the requested sense that no Cache/UAV/dual
critic was added and the single global critic reward/GAE/PopArt path is kept.
The actual representation remains the existing independent V9
`critic_encoder`/`critic_head`; it was not modified or made shared in this
stage.

SHARED_ENCODER_PRESERVED = YES

TEMPORAL_CACHE_ADAPTATION_PRESERVED = YES

TEMPORAL_UAV_ADAPTATION_PRESERVED = YES

ACTION_MASKS_PRESERVED = YES

FEASIBILITY_MASKS_PRESERVED = YES

POPART_PRESERVED = YES

CRITIC_LEARNING_RATE_PRESERVED = YES (`3e-4` in the new configuration)

CLIP_EPSILON_PRESERVED = YES (`0.2` for both branches)

ENTROPY_COEFFICIENT_PRESERVED = YES (`0.0`; branch entropies remain diagnostics)

MAX_GRAD_NORM_PRESERVED = YES (`0.5`)

Gradient metrics remain pre-clipping norms. They are measured after backward
and before the existing `clip_grad_norm_` call; Cache/UAV-specific modules and
the shared encoder are reported as separate diagnostic groups.

## NOT_IMPLEMENTED_YET

BRANCH_SPECIFIC_LR = NO

BRANCH_SPECIFIC_KL_GATE = NO

BRANCH_SPECIFIC_ADVANTAGE = NO

BRANCH_SPECIFIC_REWARD = NO

ENTROPY_BONUS = NO

LR_DECAY = NO

HARD_CACHE_TRIGGER = NO

No target_kl_cache, target_kl_uav, cache_actor_lr, uav_actor_lr, local reward,
counterfactual advantage, temporal action suppression, or additional sampling
path was introduced.

## COMPATIBILITY

ACTION_SAMPLING_CHANGED = NO

RNG_CHANGED = NO

SYSTEM_MODEL_CHANGED = NO

REWARD_CHANGED = NO

CRITIC_CHANGED = NO

PPO_OBJECTIVE_CHANGED = YES, only for the additive V10-B policy schema

V9_BACKWARD_COMPATIBLE = YES. The V9 class/schema remains unchanged, V9 uses
the existing checkpoint version 10, and a completed V9 checkpoint was loaded
successfully. V10-B uses checkpoint version 11 and retains the inherited
module/state-dict layout.

## CONFIGURATION

CONFIG = `configs/proposed/system_config_ppo_v10_branchwise_cache_uav.toml`

The configuration is based on
`configs/lr_sweep/v9_actor_lr_5e4_kl_protection.toml`. TOML semantic comparison
found only these intended differences:

- `experiment.name`: `v9_joint_cache_uav` -> `v10_branchwise_cache_uav`
- `experiment.output_tag`: `v9_joint_cache_uav` -> `v10_branchwise_cache_uav`
- `algorithm.ppo.policy_schema`: V9 -> `adaptive_joint_cache_replica_aware_uav_operation_v10_branchwise`

All PPO/system/data-protocol values otherwise remain those of the 5e-4 + KL
configuration, including `critic_learning_rate = 3e-4`, `target_kl = 0.10`,
`gamma = 1.0`, `gae_lambda = 0.95`, `clip_epsilon = 0.2`, four update epochs,
ten minibatches, minibatch size 30, `max_grad_norm = 0.5`, entropy coefficient
zero, and the existing PopArt settings.

## VALIDATION

- Python compilation of all modified Python modules: PASS.
- TOML parsing and project configuration validation: PASS.
- V10-A/V10-B/V9 direct branch sampling with identical weights and Torch RNG: PASS; selected Cache/UAV actions and RNG state matched.
- Branch ratio independence and epsilon clipping mathematical check: PASS.
- One actor optimizer parameter coverage and disjoint critic parameter check: PASS.
- V10-B temporary checkpoint save/load: PASS; temporary directory cleaned automatically.
- V9 historical completed-epoch-300 checkpoint load: PASS.
- All new scalar branch metrics finite and JSON-safe: PASS.
- No environment simulation, evaluation, episode generation, formal training, or final-test seed was run in this stage.

The previously requested V10-A diagnostic run was stopped before this stage as
required. Its partial real-data directory
`runs/lr_sweep_v9_actor_6e4_kl_protection` was preserved and neither resumed,
overwritten, nor deleted.

FORMAL_TRAINING_EXECUTED = NO

FINAL_TEST_SEEDS_USED = NO
