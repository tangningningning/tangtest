# Critic Target Alignment Minimal Experiment

- Result: **INCONCLUSIVE**
- Next stage: `CRITIC_TARGET_ALIGNMENT_EXPERIMENT_FOLLOWUP`
- Current HEAD: `52c75b0ec34dd06dda9fec407472fccf1746957e`

## Completed before interruption

- Reused TRAIN realization seed `20001` from `C:\Users\tangning\Desktop\code\adaptive_cache_mec\environment_data\proposed_joint_training_preexisting_51_backup_20260921\episode_20001`; its split, generation fingerprint, named seed streams, required artifacts, and 300-slot runtime protocol passed preflight. Its `episode_manifest.json` was read only as optional provenance.
- Fresh Critic initialization equivalence passed; initial Critic digest: `ffa1c1915716f996732c7470c348fcb08854f17ffe53c2d8ef1536f23492705f`. Actor stayed frozen and the Actor optimizer step count remained 0.
- Exactly one shared 300-transition TRAIN rollout was collected. It passed the terminal structure and rollout-time raw value checks.

## Why the experiment is inconclusive

After collecting and temporarily saving the rollout arrays, the runner raised `AttributeError` while reading `PPOV2Hyperparameters.minibatch_size`, before computing either target, generating the minibatch sequence, or taking any Critic step. The scratch directory was removed during exception cleanup, so the rollout arrays are no longer available. The runner guard has been corrected to read the runtime context minibatch size. No second rollout was collected.

Consequently, neither Critic arm was trained; both step counts are 0, no diagnostic realization or diagnostic rollout was started, no actual-state MC metrics or UAV_1 branch values were computed, and the requested 40-step checkpoints were not created.

## Current gates

- Initialization equivalence: **PASS**.
- Minibatch order equivalence: **NOT RUN**.
- ARM_A / ARM_B Critic steps: `0 / 0`.
- Training / diagnostic rollout collection counts: `1 / 0`.
- Formal PPO update: **NO**; Epoch 1: **NO**; production Critic target changed: **NO**.
- Counterfactual Critic Reliability remains **CONCERN**; Candidate-Q remains **CANDIDATE_Q_INTEGRITY_CONCERN**; surrogate repair remains blocked.
- Formal training readiness remains **NO**.

### Authorized Critic Target Alignment Follow-up

- Result: **PARTIAL**
- Next stage: `CRITIC_DATA_COVERAGE_MINIMAL_EXPERIMENT`
- Current HEAD: `52c75b0ec34dd06dda9fec407472fccf1746957e`
- Training realization: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\environment_data\proposed_joint_training_preexisting_51_backup_20260921\episode_20001` (seed 20001, LOAD)
- Diagnostic realization: seed 12184115623283823756 (GENERATE)

## Fairness and execution gates

- Initialization equivalence: **PASS**; initial Critic digest `ffa1c1915716f996732c7470c348fcb08854f17ffe53c2d8ef1536f23492705f`.
- Minibatch order equivalence: **PASS**; SHA256 `61ddab16fae6fde535a82fe987c70c61fb3cb8748200b958a2c8ba67cf8c7436`.
- Critic steps: ARM_A 40, ARM_B 40; Actor optimizer steps 0; Actor delta 0.
- Train and diagnostic rollout collection counts: 1 and 1.
- Formal PPO update: NO; Epoch 1: NO; production Critic target changed: NO.

## Independent TRAIN actual-state evaluation

| Arm | Target | MC MAE | MC RMSE | MC EV | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|
| ARM_A | GAE_LAMBDA_RETURN | 531.136537 | 680.846709 | 0.037065 | 0.4774469992360514 | 0.37266325181390897 |
| ARM_B | FULL_EPISODE_MC_RETURN | 480.699960 | 527.058296 | 0.000469 | 0.08290692477484993 | 0.26819185771265025 |

- MC MAE improvement (A − B): 50.436578.
- MC RMSE improvement (A − B): 153.788413.
- MC EV improvement (B − A): -0.036596.

## Fixed UAV_1 transition-149 witness

- Config: `[0, 0, 0, 1, 0, 0, 1, 0, 0, 0]`; transition fingerprint: `2805d3e60f0b40457ab53652995bbbd58e4cb438bac06c3af5445284981d3408`; branch reconstruction: **PASS**.
- Existing paired MC signs: NEGATIVE / NEGATIVE (ΔG values -104.857464, -380.077014).
- ARM_A ΔV: 3.220560 (POSITIVE); ARM_B ΔV: -0.010498 (NEGATIVE).
- Absolute error to paired MC mean: ARM_A 245.687799; ARM_B 242.456741.
- Exactly two immediate branches were reconstructed (KEEP and the fixed Candidate); no terminal continuation was run.

## Persistent artifacts and gates

- ARM_A checkpoint: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\diagnostics\critic_target_alignment\gae_target_critic.pt`.
- ARM_B checkpoint: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\diagnostics\critic_target_alignment\mc_target_critic.pt`.
- Reliability remains `CONCERN`; Candidate-Q remains `CANDIDATE_Q_INTEGRITY_CONCERN`; surrogate repair remains blocked: `YES`.
- Formal training readiness: `NO`.
- Decision rationale: The target change improved either the actual-state metric set or the fixed UAV_1 sign, but not both required groups.
