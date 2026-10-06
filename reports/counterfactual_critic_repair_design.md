# Counterfactual Critic Repair Design

Date: 2026-09-28

## Gate decision

**CRITIC_REPAIR_ROOT_CAUSE_CLASSIFICATION = MIXED_TARGET_AND_DATA_CONCERN**

**PRIMARY_REPAIR_DIRECTION = CRITIC_TARGET_ALIGNMENT_FIRST**

The existing actual-state Critic is poor against Monte-Carlo return, yet much closer to the GAE return used for training. GAE return itself differs substantially from MC return. The only measured Critic warmup used one full 300-slot TRAIN realization and 40 Critic optimizer steps. These observations support both target-alignment and training-coverage concerns. They do not establish either as the sole cause.

Static inspection found no deterministic implementation bug. No environment execution, checkpoint reload, model construction, optimizer step, rollout, realization generation, PPO update, or epoch was run for this design task. No production Python source changed.

## Current Critic training semantics

The proposed policy uses an independent two-layer Critic over the full configuration observation. For the active 400-dimensional input and hidden sizes [128, 128], the Critic has 67,969 trainable parameters, plus two non-trainable PopArt scalar buffers.

The input dimension follows the current specification: 9 cache nodes by 10 services (90), two UAV positions (4), 25 research-vehicle positions across five prediction slots (250), predicted popularity for 10 services across five slots (50), a five-element horizon mask, and one time-to-go feature. The resulting dimension is 400.

During rollout, the saved value is the raw denormalized Critic value V(s_t) for the pre-action observation s_t. The same transition stores reward r_t and done_t after executing the selected action. GAE is computed backward as:

delta_t = r_t + gamma * (1-done_t) * V_old(s_{t+1}) - V_old(s_t)

A_t = delta_t + gamma * lambda * (1-done_t) * A_{t+1}

R_t = A_t + V_old(s_t)

Thus the Critic target is a GAE/lambda return, not a pure MC or one-step TD target. It depends on rollout-time Critic bootstrap values at intermediate states. With gamma=1.0 and lambda=0.95, the lambda mixture gives substantial weight to bootstrapped n-step returns; the geometric mixture has an approximately 20-transition mean horizon before the episode-terminal tail. At the terminal slot, the mask removes both bootstrap and continuation. The rollout has 300 slots.

Actor advantages and Critic targets currently come from the same compute_gae call using the same gamma and lambda. PopArt statistics are updated from these returns before normalized targets are formed. The active Critic loss is:

L = 0.5 * mean((z_theta(s_t) - (R_t - mu)/sigma)^2)

where the raw value is V=mu+sigma*z. This is equivalent to 0.5/sigma^2 times the raw-space squared error. The raw-space loss computed in code is detached diagnostic output, not the optimization gradient. PopArt denormalization and output-head transformation preserve the raw value. Checkpoint load source restores the model state including PopArt buffers, restores Welford return statistics, and checks that the restored statistics agree.

The active proposed V6 path has disjoint Actor and Critic parameter groups and a separate Adam Critic optimizer. The configured and default-effective Critic learning rate is 1e-3. There is no Critic learning-rate scheduler in the inspected production path. The configured value_loss_coefficient=0.01 does not multiply the active separate Critic gradient: the Critic optimizer backpropagates value_loss directly, clips the resulting Critic gradients to max norm 0.5, then steps Adam. The coefficient appears in the unified/shared-optimizer fallback loss. No duplicate scaling is present in the active path.

Code evidence: src/ppo_simple/rollout.py lines 674-681, 804-825, and 958-986; src/ppo_simple/trainer.py lines 3593-3607, 5078-5135, 5217-5256, 7222-7304, 9410-9489, 12062-12120, and 12465-12481; src/ppo_simple/policy.py lines 749-782 and 832-900; src/ppo_simple/interface.py lines 147-179 and 193-203; src/scenario/scenario_generator.py lines 994-1002, 1190-1201, and 1257; src/decision/cache_configuration.py lines 356-368; configs/proposed/system_config.toml lines 543-559 and 598.

## Static implementation bug audit

| Check | Finding |
| --- | --- |
| Return index and t/t+1 alignment | No deterministic bug witness. The rollout saves V(s_t) and reward_t on the same pre-action observation/transition; the following TD residual consumes V(s_{t+1}). |
| Terminal mask and terminal bootstrap | No deterministic bug witness. done_t zeros both next value and propagated advantage; a completed 300-slot rollout has no bootstrap observation and uses bootstrap value 0. |
| Gamma exponent | No deterministic bug witness. The backward recurrence multiplies gamma at every step, composing gamma^k over k transitions; the configured gamma is 1.0. |
| Observation versus target | No deterministic bug witness. The value is sampled from the observation before action execution, with that transition's reward stored afterward. |
| PopArt normalization and raw restoration | No deterministic bug witness. Target normalization uses the active mean/std; raw denormalization and the output-head transform are algebraically consistent and preserve raw value. |
| Checkpoint PopArt/Welford restore | No deterministic bug witness. Static load code restores the policy state and Welford count/mean/M2, then checks active PopArt mean/std against Welford statistics. |
| Critic optimizer and gradient | No deterministic bug witness. The separate Critic optimizer step is present; the normalized residual retains gradients, only the raw diagnostic residual is detached, and actor gradients are checked absent. |

CRITIC_IMPLEMENTATION_BUG=NO and IMPLEMENTATION_BUG_FIX_REQUIRED=NO.

## Existing actual-state alignment evidence

| Metric | Before 40-step warmup | After 40-step warmup |
| --- | ---: | ---: |
| Critic MAE to MC return | 550.4785 | 537.0688 |
| Critic RMSE to MC return | 712.8836 | 695.5799 |
| Critic explained variance to MC return | 0.0000692 | 0.0060018 |
| Critic MAE to GAE training return | 103.4555 | 93.9596 |

The post-warmup Critic is about 5.7 times closer by MAE to its GAE training target than to MC return. GAE-return versus MC-return MAE is 453.7112, RMSE is 615.2593, and correlation is 0.5550. The warmup slightly improved actual-state MC and GAE metrics, but MC explained variance remained near zero. This supports CRITIC_TARGET_ALIGNMENT_CONCERN=SUPPORTED. It also means the failure cannot be attributed only to counterfactual OOD: COUNTERFACTUAL_ONLY_FAILURE=NO and COUNTERFACTUAL_GENERALIZATION_PRIMARY=NO.

One 300-transition episode is limited coverage for a 400-dimensional input and a 67,969-parameter Critic, especially given the poor actual-state MC metrics. This supports CRITIC_DATA_COVERAGE_LIMITED=SUPPORTED, while leaving open whether more data alone would resolve the error.

MORE_SAME_DATA_CRITIC_STEPS=INCONCLUSIVE. The existing 40-step run is not a 40-versus-more-steps ablation and gives no direct evidence that 80 or 120 steps would suffice.

## Counterfactual reliability evidence

The existing reliability gate remains CONCERN. The Candidate-Q label state remains CANDIDATE_Q_INTEGRITY_CONCERN, and SURROGATE_REPAIR_BLOCKED=YES.

UAV_1 is a stable direction-error witness: predicted Delta-V is +13.1373, while paired MC Delta-G values are -104.8575 and -380.0770; both signs disagree. SV deltas (+396.2438 and 0) and RSU deltas (+1082.7715 and -832.0697) vary in sign. This supports MC_VARIANCE_CONCERN=YES and a stable UAV witness, but the two replicates do not prove global Critic reliability. GLOBAL_CRITIC_RELIABILITY_PROVEN=NO.

The Critic is therefore a weak link on actual rollout states as well as counterfactual states. One-step Candidate-Q can amplify the Critic error, but N-step Candidate credit would only reduce the immediate bootstrap dependence; it would not repair the actual-state value model.

## Exact Candidate evaluation cost

The current exact branch computes Q_candidate = reward_t + gamma * V(s'_counterfactual). Source executes one slot_environment transition for each exact candidate, then evaluates the Critic on that resulting state. Hybrid selection uses K values through 16, deduplicates its proposal sources, and exact-confirms a surrogate-only winner until the winner is exact.

The saved K=16, nine-head development gate measured a mean of 20.67 exact candidates per head, approximately 186 exact candidate branches across the nine heads at one joint state. The full universe is 953 candidates per joint state: SV 95, RSU 586, UAV 272. The existing full-exact integrity artifact evaluated 2,859 candidates across three states and recorded 2,862 environment executions. The saved confirmation loop added zero candidates in its specific development reproduction; the current source may add confirmation calls when a selected winner is not already exact.

| Candidate scope | One-step calls | N=2 calls | N=5 calls | N=10 calls |
| --- | ---: | ---: | ---: | ---: |
| Typical K=16 subset across nine heads | about 186 | about 372 | about 930 | about 1,860 |
| Full 953-candidate universe | 953 | 1,906 | 4,765 | 9,530 |

Each N-step branch requires N slot_environment executions, so the multipliers are 2x, 5x, and 10x per exact candidate. These counts exclude fixed baselines and any exact-confirmation additions. At transition 149, a full terminal-MC Candidate teacher over all 953 candidates would require 142,950 branch slot executions for one replicate over the 150 remaining slots, before fixed baselines. Therefore FULL_MC_CANDIDATE_TEACHER=DIAGNOSTIC_ONLY.

## Repair-family comparison

| Repair family | Evidence for | Evidence against | Expected compute multiplier | Production intrusiveness | Addresses actual-state error | Addresses counterfactual error | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MORE_CRITIC_STEPS_SAME_DATA | 40 steps modestly improved MC and GAE MAE. | MC EV remained 0.006; no step-count ablation exists. | Same rollout costs 1x environment data; optimizer work grows with added steps. | Low | Possibly; sufficiency unproven | Indirect and uncertain; no new state coverage | INCONCLUSIVE |
| MORE_INDEPENDENT_CRITIC_DATA | One 300-transition TRAIN episode has poor actual-state MC metrics. | No multi-realization comparison proves data alone is sufficient; target gap remains. | Linear in additional TRAIN transitions. | Low to medium | Yes, by broadening TRAIN coverage | Possibly, if new data covers relevant states | PROMISING |
| CRITIC_TARGET_ALIGNMENT | Critic is much closer to GAE target than MC; GAE-to-MC MAE is 453.71. | A higher-variance long-horizon/MC target could be harder to fit; only one training realization is measured. | 1x same stored rollout and Critic-step budget when returns are re-derived from the same full episode; one separate TRAIN diagnostic episode for evaluation. | Medium | Yes | Indirectly, by improving the Critic used at counterfactual next states | PROMISING |
| COUNTERFACTUAL_NSTEP_CREDIT | Stable UAV_1 sign error; realized future rewards reduce immediate dependence on V(s'). | Actual-state MC Critic error is already severe; N-step cost scales linearly with N and does not repair the Critic. | 2x, 5x, or 10x per Exact Candidate for N=2, 5, or 10. | High | No | Partly | WEAK_EVIDENCE |

No overall score or evidence-free ranking is assigned.

## Next minimal experiment design

The next experiment direction is CRITIC_TARGET_ALIGNMENT_FIRST. Keep Actor GAE semantics unchanged and decouple only the Critic target. Compare the current GAE/lambda-return target against a full-episode MC return target computed from the same stored 300-slot TRAIN rollout. Do not select a new lambda value in this design.

For a fair comparison, collect one fixed full 300-slot TRAIN rollout once, then use its identical observations, rewards, minibatch order, and random seeds in both arms. Keep Actor and Critic initialization identical, freeze the Actor, and use the same 40-step Critic optimizer budget. Do not train Candidate or surrogate models. On one independent TRAIN diagnostic realization, compare actual-state MC MAE/RMSE/EV and the UAV_1 stable-witness Delta-V sign against paired MC Delta-G. Do not use validation, test, or final unseen data. This experiment is a design only and was not executed here.

## Current gate state

- CRITIC_IMPLEMENTATION_BUG = NO; IMPLEMENTATION_BUG_FIX_REQUIRED = NO.
- CRITIC_TARGET_ALIGNMENT_CONCERN = SUPPORTED.
- MORE_SAME_DATA_CRITIC_STEPS = INCONCLUSIVE.
- CRITIC_DATA_COVERAGE_LIMITED = SUPPORTED.
- ACTOR_GAE_CRITIC_TARGET_DECOUPLING_FEASIBLE = YES.
- CRITIC_TARGET_ALIGNMENT_REPAIR_PROMISING = YES.
- COUNTERFACTUAL_CRITIC_RELIABILITY = CONCERN.
- CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION = CANDIDATE_Q_INTEGRITY_CONCERN.
- SURROGATE_REPAIR_BLOCKED = YES.
- READY_FOR_FORMAL_TRAINING_STAGE = NO.
- NEXT_STAGE = AWAIT_AUTHORIZATION_FOR_CRITIC_TARGET_ALIGNMENT_MINIMAL_EXPERIMENT.

No new experiment or Epoch 1 should start automatically.
