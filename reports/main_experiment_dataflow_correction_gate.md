# MAIN_EXPERIMENT_DATAFLOW_CORRECTION_AND_ONE_EPOCH_GATE

## Gate result

`MAIN_EXPERIMENT_DATAFLOW_GATE = PASS`

The gate used only the proposed mainline (`proposed_joint`, schema
`adaptive_joint_cache_uav_ppo`, control mode
`adaptive_joint_cache_uav_reconfiguration`). The run was a fresh initialization
with one outer epoch and 300 slots; no second epoch, development seed, final-test
seed, sweep, or Cache-Q training was run.

## Before and after dataflow

Before correction, the proposed Cache branch entered the formal
`EXACT_CACHE_CREDIT_POLICY_SCHEMAS` path. It called
`compute_cache_action_q_credit()`, executed Cache counterfactual branches, and
fed Action-Q/surrogate-derived trigger and candidate credit into the Cache
policy loss.

After correction, the production dataflow is:

```text
provenance-verified episode realization
  -> actual rollout actions and rewards
  -> global critic
  -> normalized global GAE
  -> Cache joint-branch sampled PPO
  -> unchanged UAV Exact-Q teacher path
  -> Matching
  -> KKT
  -> reward
  -> PPO update
```

The proposed production schema is no longer in
`EXACT_CACHE_CREDIT_POLICY_SCHEMAS`; the trainer no longer imports or calls
`compute_cache_action_q_credit()`. `CacheActionQCritic`, Cache surrogate,
factorized-Q, and KEEP-relative diagnostic modules remain in historical code but
are not initialized or trained by the proposed trainer. The checkpoint version
is `17`, so earlier proposed checkpoints are not resume-compatible.

## Cache PPO credit

For each rollout transition, the existing joint branch log-probabilities are
used directly:

```text
cache_old_log_prob
  = sum(selected cache operation old log-probs)
  + sum(selected conditional target old log-probs)

ratio = exp(evaluation.cache_log_prob - cache_old_log_prob)

L_cache = -mean(min(ratio * A_GAE,
                    clip(ratio, 1-epsilon, 1+epsilon) * A_GAE))
```

`A_GAE` is the normalized global GAE computed from actual rollout rewards and
the global critic. No synthetic KEEP target, unexecuted-candidate advantage, or
Action-Q fill is used.

Formal metadata from the one-epoch run:

```text
CACHE_POLICY_CREDIT_SOURCE = ACTUAL_ROLLOUT_GAE
CACHE_ACTION_Q_CRITIC_INITIALIZED = NO
CACHE_ACTION_Q_USED_FOR_POLICY_CREDIT = NO
CACHE_COUNTERFACTUAL_MODE = NONE_FOR_PRODUCTION
CACHE_COUNTERFACTUAL_ENV_EXECUTIONS = 0
cache_trigger_advantage_source = GLOBAL_GAE
cache_candidate_advantage_source = GLOBAL_GAE
cache_trigger_ratio_granularity = WHOLE_CACHE_JOINT
cache_candidate_ratio_granularity = WHOLE_CACHE_JOINT
CACHE_ACTOR_GRADIENT_FINITE = YES
CACHE_ACTOR_PARAMETER_DELTA_NORM = 0.5436778843043405
```

The formal checkpoint contains no Cache Action-Q critic state, Cache Action-Q
optimizer state, Cache surrogate state, or Cache surrogate optimizer state.

## Unchanged UAV teacher and system models

The UAV operation path remains the full-information Exact-Q teacher with the
existing operation KL protection. The Exact-Q listwise candidate teacher and
candidate-only behavior after operation stop remain unchanged; no UAV
architecture, teacher temperature, loss, LR, movement cost, or candidate
proposal was changed.

```text
UAV_OPERATION_TEACHER = FULL_INFORMATION_EXACT_Q_TEACHER
UAV_CANDIDATE_TEACHER = EXACT_Q_LISTWISE
UAV_TEACHER_UNCHANGED = YES
MATCHING_CHANGED = NO
KKT_CHANGED = NO
REWARD_CHANGED = NO
SYSTEM_MODEL_CHANGED = NO
```

The non-production UAV Exact-Q teacher executed its existing
`11276` counterfactual environment evaluations. Those are UAV teacher
evaluations, not Cache production credit; Cache production counterfactual
execution remained zero.

The unchanged model assumptions are:

```text
FIXED_RESEARCH_COHORT = CURRENT_MODEL_ASSUMPTION
CLOUD_RESOURCE_SEMANTICS = FIXED_PER_TASK_FALLBACK
T_REF_E_REF_CALIBRATION = LEGACY_0_239_TRAIN_ONLY_CALIBRATION
```

## Episode provenance contract

Formal realization:

```text
EPISODE_MANIFEST_CREATED = YES
EPISODE_SEED = 20001
TRAINING_EPISODE_PROVENANCE_VERIFIED = YES
GENERATION_CONFIG_FINGERPRINT = 9ad3c6179be6cea67a0c95695f0cc2802e7570274c9be5446e51d996d5f7045d
SOURCE_GIT_SHA = c82f16b76b4698d1a7c2982c3785fcf167f3f0ef
SOURCE_GIT_DIRTY = true
SERVICE_CATALOG_SHA256 = 3adb64f9724a429a5cb6bf9c6eb609a54120d9918feaab1dd0475034af22e402
POPULARITY_GRU_SHA256 = 6f6a483c6807fca272fdb483df23280702ab1d61611af6971c7a302926a7ed8a
RESIDUAL_BANK_SHA256 = bf5790fcd62fdabdbe33ca6ce7dc3e686da377494ec84a8bd12ba2fa4ee57dcc
SUMO_NETWORK_SHA256 = 30379989e0f94505e30ed5ec27449367615847ab4baf1bfe2340d8c405388c7d
```

The manifest records schema version `episode_manifest_v1`, all 11 derived
episode streams, the generation-only configuration fingerprint, source git
identity, frozen artifact hashes, `dt=1.0`, `warmup=100.0`, `episode_slots=300`,
vehicle counts research/TV/SV=`25/20/5`, task arrival/Zipf/workload-jitter
identity, vehicle prediction horizon/noise contract, popularity model identity,
scenario count, and every generated artifact filename plus SHA-256. PPO LR,
PPO epochs, reward weights, and Matching runtime tuning are excluded from the
generation fingerprint.

The old pool
`runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911` was preserved. Its
manifest-less `episode_20001` was tested and rejected with
`REALIZATION_PROVENANCE_MISMATCH`; it was not auto-filled or overwritten.

## One-epoch structural evidence

Run artifacts:

```text
REALIZATION = runs/main_experiment_dataflow_correction_gate/episode_20001
RUN = runs/main_experiment_dataflow_correction_gate/one_epoch_seed_20001
DATA_PROTOCOL = full_episode_randomized
ORIGINS = 0..299
TERMINAL = true
TARGET_EPOCHS = 1
COMPLETED_EPOCHS = 1
```

Measured values from `episode_metrics.jsonl` and `update_metrics.jsonl`:

```text
RETURN = -2063.866768670678
CACHE_CHANGE_RATE = 0.49703703703703705
UAV_MOVEMENT_RATE = 0.49333333333333335
CACHE_APPROX_KL = 0.028373518847705094
UAV_OPERATION_KL = 0.0030716408509761095
CRITIC_EXPLAINED_VARIANCE = -0.0002765547648779787
ROLLOUT_RUNTIME_SECONDS = 66.21844480000436
UAV_EXACT_Q_RUNTIME_SECONDS = 224.121561500011
PPO_UPDATE_RUNTIME_SECONDS = 260.3547015999793
TOTAL_EPOCH_RUNTIME_SECONDS = 326.7379741000477
NAN_INF_DETECTED = false
```

The Cache actor gradient gate was finite (`gradient_finite=true`, all finite
checks true; Cache head gradient norm `0.42114208117127416`, Cache encoder
gradient norm `0.6110294334590435`). The Cache actor parameter delta norm is
`0.5436778843043405`, computed offline by reproducing the fresh seed-4401
initial policy and taking the Euclidean delta over `cache_encoder`,
`cache_trigger_head`, and `cache_candidate_scorer` against the one-epoch
checkpoint. The run recorded `1342` actual Cache updates and `296` actual UAV
movements.

## Final structural fields

```text
DEV_SEEDS_USED = NO
FINAL_TEST_SEEDS_USED = NO
TEMP_FILES_CLEANED = YES
GIT_DIFF_CHECK = PASS
REPORT_PATH = C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\main_experiment_dataflow_correction_gate.md
COMMITTED = NO
PUSHED = NO
```

The formal report, provenance realization, one-epoch checkpoint, metrics, and
plot are retained. No temporary runner or debug dump was created.
