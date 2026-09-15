# V10-A Branch Actor Observability

STAGE = V10-A_BRANCH_OBSERVABILITY

## CURRENT_V9_POLICY_STRUCTURE

The following is the structure confirmed from the current implementation, not
inferred from the schema name:

- Cache policy distribution: `AdaptiveCacheOperationGroupV7PopArtActorCritic._operation_logits()` and `sample_cache_operation()` create a four-way per-cache-head categorical distribution in the order `NO_CHANGE/ADD/DELETE/SWAP`. `evaluate_cache_operation_primitive()` reconstructs the same masked operation distributions and the active candidate distributions.
- UAV policy distribution: `AdaptiveJointCacheReplicaAwareUAVOperationV9PopArtActorCritic._uav_operation_logits()` and `sample_uav_operation()` create a two-way per-UAV categorical distribution in the order `STAY/MOVE`. `evaluate_joint_cache_uav_operation()` reconstructs the masked operation distributions and the active MOVE candidate distributions.
- Cache logits/probabilities: shared `encoder` -> `cache_operation_head` gives raw operation logits; `sample_cache_operation()` builds raw probabilities and then applies the actual operation mask to form masked logits. Cache candidate scores come from the shared `encoder` plus `cache_candidate_scorer`, followed by the actual candidate mask.
- UAV logits/probabilities: shared `encoder` -> `uav_operation_head` gives raw operation logits; `sample_uav_operation()` builds raw probabilities and then applies the actual STAY/MOVE feasibility mask to form masked logits. UAV candidate scores come from the shared `encoder` plus `uav_candidate_scorer`, followed by the actual candidate mask.
- Cache log-probability: sum of every cache operation categorical log-probability plus candidate categorical log-probabilities for active non-`NO_CHANGE` operations.
- UAV log-probability: sum of every UAV operation categorical log-probability plus candidate categorical log-probabilities for active `MOVE` operations.
- Rollout old log-probability: `SimpleTransition.old_log_prob` stores the joint primitive log-probability. The transition also stores per-head Cache operation/candidate and UAV operation/candidate old log-probabilities; these are now stacked in `SimpleRollout` for diagnostics.
- PPO ratio: the unchanged objective uses `exp(new_joint_log_prob - old_joint_log_prob)`; policy loss and clipping still use this joint ratio.
- Actor encoder sharing: Cache and UAV branches share `self.encoder`.
- Actor optimizer parameters: one actor optimizer contains the shared encoder, `cache_operation_head`, `cache_candidate_scorer`, `uav_operation_head`, and `uav_candidate_scorer`.
- Critic sharing: `NO`. V9 uses the independent `critic_encoder` and `critic_head`; Cache/UAV and Critic still use the same global reward/GAE/PopArt target path.
- Feasibility/action masks: Cache operation masks are applied in `sample_cache_operation()` and `evaluate_cache_operation_primitive()`; Cache candidate masks are applied in the corresponding candidate distributions. UAV operation feasibility masks are applied in `sample_uav_operation()` and `evaluate_joint_cache_uav_operation()`; UAV candidate masks are applied in the MOVE candidate distribution.
- Entropy: current joint entropy is the sum of Cache operation/candidate entropy and UAV operation/candidate entropy. V10 additionally exposes those two branch sums without changing the joint entropy used by the objective.
- Approximate KL: current PPO records the sampled-action joint estimate `mean(old_joint_log_prob - new_joint_log_prob)`; it is not a full categorical-distribution KL.

CACHE_POLICY_PATH = `src/ppo_simple/policy.py::AdaptiveCacheOperationGroupV7PopArtActorCritic`

UAV_POLICY_PATH = `src/ppo_simple/policy.py::AdaptiveJointCacheReplicaAwareUAVOperationV9PopArtActorCritic`

SHARED_ENCODER = YES; both policy branches use `self.encoder`.

SHARED_CRITIC = NO; the V9/V10 policy has an independent `critic_encoder`/`critic_head`.

CURRENT_LOGPROB_STRUCTURE = joint old log-probability plus already-existing per-head Cache/UAV operation and candidate old log-probabilities in each transition.

CURRENT_PPO_RATIO_STRUCTURE = `exp(new_joint_log_prob - old_joint_log_prob)`; unchanged joint ratio remains the only ratio used by PPO loss/clipping.

CURRENT_KL_STRUCTURE = sampled-action joint approximate KL `mean(old_joint_log_prob - new_joint_log_prob)`; the old rollout does not contain complete old categorical probability vectors, so no full-distribution branch KL is claimed.

## V10-A implementation

BRANCH_DIAGNOSTICS_IMPLEMENTED = YES

CACHE_KL_AVAILABLE = YES; sampled-action approximate branch KL from the existing Cache operation/candidate old log-probabilities.

UAV_KL_AVAILABLE = YES; sampled-action approximate branch KL from the existing UAV operation/candidate old log-probabilities.

CACHE_CLIP_FRACTION_AVAILABLE = YES; sampled-action Cache branch ratio against the existing PPO `clip_epsilon`.

UAV_CLIP_FRACTION_AVAILABLE = YES; sampled-action UAV branch ratio against the existing PPO `clip_epsilon`.

CACHE_ENTROPY_AVAILABLE = YES; exact current masked Cache operation/candidate entropy sum.

UAV_ENTROPY_AVAILABLE = YES; exact current masked UAV operation/candidate entropy sum.

CACHE_HEAD_GRAD_AVAILABLE = YES; pre-clipping norm of Cache-only `cache_operation_head` plus `cache_candidate_scorer` gradients after the unchanged joint actor backward.

UAV_HEAD_GRAD_AVAILABLE = YES; pre-clipping norm of UAV-only `uav_operation_head` plus `uav_candidate_scorer` gradients after the unchanged joint actor backward.

SHARED_ENCODER_GRAD_AVAILABLE = YES; pre-clipping norm of the shared `encoder` gradients, kept separate from both branch norms.

The update log fields are:
`cache_approx_kl`, `uav_approx_kl`, `cache_clip_fraction`,
`uav_clip_fraction`, `cache_entropy`, `uav_entropy`,
`cache_head_gradient_norm`, `uav_head_gradient_norm`, and
`shared_actor_gradient_norm`. They are scalar diagnostics only; no large arrays
are written.

An additive schema,
`adaptive_joint_cache_replica_aware_uav_operation_v10_branch_observable`,
inherits the V9 module layout and sampling implementation. V9 remains the
original schema and is not renamed or overwritten.

EXTRA_POLICY_SAMPLE_ADDED = NO

RNG_SEQUENCE_CHANGED = NO

ACTION_SEMANTICS_CHANGED = NO

REWARD_CHANGED = NO

SYSTEM_MODEL_CHANGED = NO

CRITIC_CHANGED = NO

PPO_OBJECTIVE_CHANGED = NO

LR_CHANGED = NO

KL_GATE_CHANGED = NO

V9_BACKWARD_COMPATIBLE = YES; the original V9 class/schema and checkpoint path
remain intact. An existing V9 checkpoint was loaded successfully, and the
additive V10 state-dict key set is identical to V9.

## Minimal validation

- `adaptive_cache_mec` Conda environment was used.
- V9 and V10 with identical weights selected the same direct Cache/UAV test distributions under identical Torch seeds.
- V10 branch log-probability and entropy fields were finite and summed exactly to the existing joint fields.
- V9 and V10 actor parameter groups were disjoint and covered the same actor parameter set.
- Existing V9 `latest_checkpoint.pt` loaded successfully at completed epoch 300.
- V10 trainer construction and temporary checkpoint save/load succeeded; the temporary directory was cleaned up.
- V9 and V10 configuration validation succeeded.
- No training, evaluation, episode generation, or final-test seed was run.
- `git diff --check` passed.
