# V10-C final implementation plan

STAGE = V10-C_FINAL_IMPLEMENTATION_PLAN

## Closed decision

Implement only a Cache/UAV actor-encoder split. The mechanism gate measured Cache-to-UAV shared-encoder norm ratios above 1 in all 20 chronological blocks across the two representative 300-slot rollouts; median ratios were 3.32x at 5e-4 and 4.33x at 6e-4. Whole-rollout gradient cosines were also negative at both checkpoints. The other proposed PPO changes are not supported by direct, consistent evidence:

- Keep the current joint UAV ratio/clipping (`PER_UAV_RATIO_CLIPPING = NO`): both copy-only dry-runs had zero individual/joint clipping and zero cross-clipping mismatches.
- Keep the current joint KL gate behavior for now (`BRANCH_SPECIFIC_KL_GATE = STILL_INSUFFICIENT`): the existing runs had 58 gate events but lack per-minibatch/per-epoch branch KL values needed to attribute suppression.
- Do not add a shared UAV contextual baseline (`UAV_CONTEXTUAL_BASELINE = NO`): the held-out validation did not generalize on both representative trajectories.
- Do not add separate per-UAV baselines (`PER_UAV_CONTEXTUAL_BASELINE = NO`): the parameter-matched comparison improves only one of the two tracks.
- Do not change the reward, movement cost, system model, or action space based on the small one-step check; the sampled immediate objective favored STAY, but this is not long-term Q evidence.

## Planned code change: separate actor representations

1. Add a new, versioned V10 policy schema in `src/ppo_simple/policy.py` with distinct Cache and UAV encoders of the current encoder’s dimensions. Keep the existing Cache operation/candidate heads attached to the Cache encoder and UAV operation/candidate heads/scorer attached to the UAV encoder. Do not silently alter the existing checkpoint schema.
2. Route Cache operation contexts and Cache candidate contexts through the Cache encoder; route each pre-action UAV context and its candidate-scoring context through the UAV encoder. Preserve UAV0→UAV1 autoregressive context construction, masks, selected actions, feasible candidates, and per-head old-log-prob storage.
3. Retain the two current branchwise PPO losses, their equal branch weighting, global centered/std GAE advantage, joint UAV log-prob/ratio/clipping, current KL gate behavior, and separate critic path. The only algorithmic change in this plan is the actor representation split.
4. Keep one actor optimizer over the disjoint Cache/UAV actor parameter sets unless implementation-level validation shows a concrete optimizer-state issue. Ensure `actor_parameters()` covers each actor parameter once and branch gradient tests show Cache loss reaches only the Cache encoder/heads while UAV loss reaches only the UAV encoder/heads. Critic parameters remain disjoint.
5. Bump the policy/checkpoint schema/version. Do not resume an existing V10-C checkpoint under the new schema implicitly. Any future warm-start mapping must explicitly copy the old shared encoder into both new encoders and define how shared Adam moments are handled; record that mapping in checkpoint provenance.

## Diagnostic-only KL observability prerequisite

Before deciding whether to replace the joint KL gate in a later, separately authorized change, extend update diagnostics to retain, for each update epoch and minibatch, Cache KL, UAV KL, joint KL, the target comparison, gate trigger, and remaining epochs suppressed. Preserve the present gate logic while collecting these values. This addresses the demonstrated attribution gap (11 and 47 stop events in the two runs) without changing actor/critic update behavior.

## Required invariants

```text
GLOBAL_REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
SYSTEM_COST_CHANGED = NO
ACTION_SPACE_CHANGED = NO
TEMPORAL_ADAPTATION_PRESERVED = YES
CRITIC_CHANGED = NO
GLOBAL_GAE_AND_NORMALIZATION_CHANGED = NO
JOINT_UAV_RATIO_CLIPPING_CHANGED = NO
JOINT_KL_GATE_BEHAVIOR_CHANGED = NO
```

No code was changed and no training was started in the mechanism-closure task. Implementing this plan, migrating checkpoints, or launching any subsequent training requires a separate authorized task.
