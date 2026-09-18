# V10-D credit-assignment diagnostic

STAGE = V10D_CREDIT_ASSIGNMENT_DIAGNOSTIC

## Scope and static code result

This is a read-only diagnosis on the frozen V10-D latest checkpoint. No training, optimizer step, checkpoint write, PPO/reward/system change, episode generation, or final test seed was used.

The eight existing paired-witness snapshots were reused. Branch rows reuse paired H30 values. Existing target RSU/UAV rows are reused when the deterministic action matches; only missing node targets receive the minimal new H30 evaluation.

Static code confirms one global standard GAE vector is used by both branch losses. Every Cache node and every UAV node on a transition receives that transition's same normalized advantage. The critic is observation-only PopArt V(s), not action-conditioned Q(s,a).

| Code evidence | Finding |
|---|---|
| src/ppo_simple/rollout.py:858-885 | compute_gae constructs one standard temporal-difference GAE vector from rewards, V(s_t), V(s_t+1), dones, gamma, and lambda. |
| src/ppo_simple/trainer.py:2073-2081 | PPOV2Trainer.update calls compute_gae once for the rollout. |
| src/ppo_simple/trainer.py:2267-2299 | The single advantages vector becomes the normalized advantages_t actor-loss tensor. |
| src/ppo_simple/trainer.py:2509-2552 | Cache and UAV branchwise terms both multiply advantages_t[indices]; V10-D KL gating changes branch activation only. |
| src/ppo_simple/policy.py:1550-1582 | The joint evaluator exposes separate Cache/UAV log-probabilities but one shared value evaluation. |
| src/ppo_simple/trainer.py:4338-4340 | Per-UAV diagnostics repeat the same global advantage value for each UAV head. |
| src/ppo_simple/policy.py:857-870 | critic_value returns PopArt mean plus PopArt std times critic_normalized_value(observations), with no action input. |

Exact GAE alignment is NOT_AVAILABLE: the reused snapshot JSON omits the original per-transition rewards, old critic values, done flags, and terminal bootstrap value. The C-bank snapshots are not V10-D training transitions. No manual Q-from-V reconstruction was performed.

## Branch-level signed marginals

Cache signed delta = G30(selected Cache action) minus G30(best bounded evaluated Cache alternative). UAV signed delta = G30(selected UAV action) minus G30(best bounded evaluated UAV alternative). The move-minus-STAY value remains in the raw JSON as a separate node-level diagnostic.

| Snapshot | Cache signed delta | UAV signed delta | Opposite signs |
|---|---:|---:|:---:|
| BANK_C_EP_20041_A_120 | -2.120435 | -0.105745 | NO |
| BANK_C_EP_20041_A_240 | -0.407568 | 0.000000 | NO |
| BANK_C_EP_20161_A_120 | -2.668710 | -0.050662 | NO |
| BANK_C_EP_20161_A_240 | -0.313831 | -0.025584 | NO |
| BANK_D_EP_20041_A_120 | -5.836738 | -0.107152 | NO |
| BANK_D_EP_20041_A_240 | -0.151918 | -0.043946 | NO |
| BANK_D_EP_20161_A_120 | -8.043250 | -0.024646 | NO |
| BANK_D_EP_20161_A_240 | -1.678565 | -0.048676 | NO |

BRANCH_OPPOSITE_SIGN_COUNT = 0; BRANCH_OPPOSITE_SIGN_RATE = 0.000. Under the predeclared majority rule, BRANCH_LEVEL_CREDIT_ASSIGNMENT_PROBLEM = NOT_SUPPORTED.

## Cache-node diagnostic

Every snapshot includes RSU_0, RSU_1, the first configured SV node (deterministic by node index), and deterministic UAV-cache node UAV_0. Other Cache/UAV choices are held at deterministic V10-D policy selection. New alternatives are only NO_CHANGE and the deterministic scorer-highest legal update; no outcome-based candidate selection was used.

| Snapshot | Node | Role | signed delta | NODE_REGRET | Source |
|---|---|---|---:|---:|---|
| BANK_C_EP_20041_A_120 | RSU_0 | RSU | -2.120435 | 2.120435 | REUSED_V10D_PAIRED_WITNESS |
| BANK_C_EP_20041_A_120 | RSU_1 | RSU | -0.298103 | 0.298103 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_C_EP_20041_A_120 | 47 | SV | -0.000001 | 0.000001 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_C_EP_20041_A_120 | UAV_0 | UAV | 0.065111 | 0.000000 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_C_EP_20041_A_240 | RSU_0 | RSU | 0.518985 | 0.000000 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_C_EP_20041_A_240 | RSU_1 | RSU | -0.407568 | 0.407568 | REUSED_V10D_PAIRED_WITNESS |
| BANK_C_EP_20041_A_240 | 47 | SV | 0.000000 | -0.000000 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_C_EP_20041_A_240 | UAV_0 | UAV | 0.238116 | 0.000000 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_C_EP_20161_A_120 | RSU_0 | RSU | -2.668710 | 2.668710 | REUSED_V10D_PAIRED_WITNESS |
| BANK_C_EP_20161_A_120 | RSU_1 | RSU | -0.357411 | 0.357411 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_C_EP_20161_A_120 | 9 | SV | -0.046339 | 0.046339 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_C_EP_20161_A_120 | UAV_0 | UAV | 0.070477 | 0.000000 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_C_EP_20161_A_240 | RSU_0 | RSU | -0.372577 | 0.372577 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_C_EP_20161_A_240 | RSU_1 | RSU | -0.313831 | 0.313831 | REUSED_V10D_PAIRED_WITNESS |
| BANK_C_EP_20161_A_240 | 9 | SV | -0.081143 | 0.081143 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_C_EP_20161_A_240 | UAV_0 | UAV | 0.070477 | 0.000000 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_D_EP_20041_A_120 | RSU_0 | RSU | -5.836738 | 5.836738 | REUSED_V10D_PAIRED_WITNESS |
| BANK_D_EP_20041_A_120 | RSU_1 | RSU | -2.590738 | 2.590738 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_D_EP_20041_A_120 | 47 | SV | -0.073730 | 0.073730 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_D_EP_20041_A_120 | UAV_0 | UAV | -44.349721 | 44.349721 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_D_EP_20041_A_240 | RSU_0 | RSU | 2.571634 | 0.000000 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_D_EP_20041_A_240 | RSU_1 | RSU | -0.151918 | 0.151918 | REUSED_V10D_PAIRED_WITNESS |
| BANK_D_EP_20041_A_240 | 47 | SV | 0.175410 | 0.000000 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_D_EP_20041_A_240 | UAV_0 | UAV | -19.423337 | 19.423337 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_D_EP_20161_A_120 | RSU_0 | RSU | -8.043250 | 8.043250 | REUSED_V10D_PAIRED_WITNESS |
| BANK_D_EP_20161_A_120 | RSU_1 | RSU | 0.104709 | 0.000000 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_D_EP_20161_A_120 | 9 | SV | -0.073730 | 0.073730 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_D_EP_20161_A_120 | UAV_0 | UAV | -38.067308 | 38.067308 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_D_EP_20161_A_240 | RSU_0 | RSU | -2.138072 | 2.138072 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_D_EP_20161_A_240 | RSU_1 | RSU | -1.678565 | 1.678565 | REUSED_V10D_PAIRED_WITNESS |
| BANK_D_EP_20161_A_240 | 9 | SV | -0.073730 | 0.073730 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_D_EP_20161_A_240 | UAV_0 | UAV | -40.127461 | 40.127461 | NEW_MINIMAL_NODE_EVALUATION |

CACHE_NODE_OPPOSITE_SIGN_STATE_COUNT = 6; CACHE_NODE_REGRET_MEAN = 5.290450528582255; CACHE_NODE_REGRET_MEDIAN = 0.22501024645688972. Therefore CACHE_NODE_LEVEL_CREDIT_ASSIGNMENT_PROBLEM = SUPPORTED.

## UAV-node diagnostic

Every snapshot includes UAV_0 and UAV_1. Cache and the other UAV are held at deterministic V10-D policy selection; the target UAV is compared at policy-selected movement versus STAY.

| Snapshot | Node | Policy action | signed delta move-minus-stay | NODE_REGRET | Source |
|---|---|---|---:|---:|---|
| BANK_C_EP_20041_A_120 | UAV_0 | MOVE | -0.046510 | 0.046510 | REUSED_V10D_PAIRED_WITNESS |
| BANK_C_EP_20041_A_120 | UAV_1 | MOVE | -0.109504 | 0.109504 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_C_EP_20041_A_240 | UAV_0 | MOVE | -0.052211 | 0.052211 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_C_EP_20041_A_240 | UAV_1 | MOVE | 0.004204 | 0.000000 | REUSED_V10D_PAIRED_WITNESS |
| BANK_C_EP_20161_A_120 | UAV_0 | MOVE | -0.050662 | 0.050662 | REUSED_V10D_PAIRED_WITNESS |
| BANK_C_EP_20161_A_120 | UAV_1 | MOVE | -0.087206 | 0.087206 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_C_EP_20161_A_240 | UAV_0 | MOVE | -0.034697 | 0.034697 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_C_EP_20161_A_240 | UAV_1 | MOVE | -0.025584 | 0.025584 | REUSED_V10D_PAIRED_WITNESS |
| BANK_D_EP_20041_A_120 | UAV_0 | MOVE | -0.096888 | 0.096888 | REUSED_V10D_PAIRED_WITNESS |
| BANK_D_EP_20041_A_120 | UAV_1 | MOVE | -0.093327 | 0.093327 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_D_EP_20041_A_240 | UAV_0 | MOVE | -0.033337 | 0.033337 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_D_EP_20041_A_240 | UAV_1 | MOVE | -0.043946 | 0.043946 | REUSED_V10D_PAIRED_WITNESS |
| BANK_D_EP_20161_A_120 | UAV_0 | MOVE | -0.024646 | 0.024646 | REUSED_V10D_PAIRED_WITNESS |
| BANK_D_EP_20161_A_120 | UAV_1 | MOVE | -0.147366 | 0.147366 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_D_EP_20161_A_240 | UAV_0 | MOVE | -0.042659 | 0.042659 | NEW_MINIMAL_NODE_EVALUATION |
| BANK_D_EP_20161_A_240 | UAV_1 | MOVE | -0.048676 | 0.048676 | REUSED_V10D_PAIRED_WITNESS |

UAV_NODE_OPPOSITE_SIGN_STATE_COUNT = 1; UAV_NODE_REGRET_MEAN = 0.05857617499584933; UAV_NODE_REGRET_MEDIAN = 0.047592702986634094. Therefore UAV_NODE_LEVEL_CREDIT_ASSIGNMENT_PROBLEM = NOT_SUPPORTED.

## Saturation auxiliary

Cache max masked-operation probability mean/median = 0.8866/0.8706; entropy mean = 0.3623 nats; max at least 0.90 in 3/8 states. CACHE_POLICY_SATURATION_SUPPORTED = NO.
UAV max masked-operation probability mean/median = 0.9421/0.9466; entropy mean = 0.1975 nats; max at least 0.90 in 6/8 states; MOVE probability mean = 0.9421; displacement mean/median/std = 9.3750/9.3750/5.6250 m. UAV_POLICY_SATURATION_SUPPORTED = YES.

Saturation is auxiliary evidence only and does not override the signed-conflict decision.

## Decision and next direction

The code-level shared-advantage fact is confirmed, but the paired branch signs oppose only in the reported count. This diagnostic does not call that causal proof.

Because Cache-node signed conflicts occur in 6/8 snapshots while branch-level and UAV-node conflicts are not frequent, the next algorithm direction is B_CACHE_NODE_SPECIFIC_COUNTERFACTUAL_BASELINE. This is a diagnostic direction only and is not implemented here. If a later larger frozen diagnostic does not reproduce the Cache-node conflict, fall back to POLICY_DISTRIBUTION_SATURATION_OR_ACTION_PARAMETERIZATION. No reward shaping, move penalty, stay bonus, or target change is proposed.

## Required status

STAGE = V10D_CREDIT_ASSIGNMENT_DIAGNOSTIC
TRAINING_EXECUTED = NO
PPO_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
FINAL_TEST_SEEDS_USED = NO
CACHE_AND_UAV_USE_SAME_GLOBAL_ADVANTAGE = YES
CACHE_NODES_USE_SAME_ADVANTAGE = YES
UAV_NODES_USE_SAME_ADVANTAGE = YES
CRITIC_ACTION_CONDITIONED = NO
BRANCH_OPPOSITE_SIGN_COUNT = 0
BRANCH_OPPOSITE_SIGN_RATE = 0.0
CACHE_NODE_OPPOSITE_SIGN_STATE_COUNT = 6
UAV_NODE_OPPOSITE_SIGN_STATE_COUNT = 1
GLOBAL_GAE_ALIGNMENT = NOT_AVAILABLE
GLOBAL_ADV_CACHE_SIGN_AGREEMENT = NOT_AVAILABLE
GLOBAL_ADV_UAV_SIGN_AGREEMENT = NOT_AVAILABLE
CACHE_POLICY_SATURATION_SUPPORTED = NO
UAV_POLICY_SATURATION_SUPPORTED = YES
BRANCH_LEVEL_CREDIT_ASSIGNMENT_PROBLEM = NOT_SUPPORTED
CACHE_NODE_LEVEL_CREDIT_ASSIGNMENT_PROBLEM = SUPPORTED
UAV_NODE_LEVEL_CREDIT_ASSIGNMENT_PROBLEM = NOT_SUPPORTED
NEXT_ALGORITHM_DIRECTION = B_CACHE_NODE_SPECIFIC_COUNTERFACTUAL_BASELINE (diagnostic direction only; do not implement in this stage)
REPORT = C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\v10d_credit_assignment_diagnostic.md
RAW_RESULTS = C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\v10d_credit_assignment_diagnostic.json
GIT_DIFF_CHECK = PASS
TEMP_FILES_REMOVED = YES
PUSHED = NO

Raw detailed rows are in the JSON report.
