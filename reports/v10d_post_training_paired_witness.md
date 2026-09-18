# V10-D post-training paired witness

`STAGE = V10D_POST_TRAINING_PAIRED_WITNESS`

## Frozen protocol

- No training, optimizer step, reward/system/PPO change, episode generation, unseen test, commit, or push was executed.
- Frozen pool: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v6_popart_actor1e4_lazy_episode_pool_20260911`; episodes `20041`, `20161`; anchors `120`, `240`; H=`30`.
- State banks: `BANK_C` and `BANK_D`; exactly 8 snapshots total, with deterministic alternating `RSU_0/RSU_1` and `UAV_0/UAV_1` targets.
- Each snapshot records cache, UAV positions, current vehicle positions, prediction, workload/tasks, scenario slice, future-slot fingerprints, and deterministic RNG provenance.

## Frozen checkpoints

| Policy | Latest checkpoint | Completed | Config SHA256 |
|---|---|---:|---|
| C | `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v10c_decoupled_encoder_6e4_formal\latest_checkpoint.pt` | 300/300 | `4763d139d1ffa6338707a6524889fec02070559d05baa06cbc39f03e54351f25` |
| D | `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v10d_decoupled_encoder_branch_kl_6e4_formal\latest_checkpoint.pt` | 300/300 | `1515f8cf62df1980564c1fb1a4ba95da28a6157712c679ed2895143c797c6d44` |

## Source identity

- `SOURCE_IDENTITY = PARTIALLY_VERIFIED`.
- Current repository HEAD: `f0fb60341f775c0c0210ff7f2f20e519ab16c621`; current worktree dirty: `True`.
- Config hashes match. C manifest lacks git source fields; D records launch commit b72c47d4c418eb73e506773887cf2dc12c2a50da with git_dirty=true, and neither manifest exposes a recoverable source snapshot/diff.

## Paired Cache witness

Regret is `best evaluated G30 - selected-action G30` over the bounded evaluated candidate set; it is not oracle regret.

| Scope | Continuation | N | Mean regret | Median regret | Positive rate | KEEP→beneficial update | UPDATE→KEEP better |
|---|---|---:|---:|---:|---:|---:|---:|
| BANK_C | C | 4 | 0.335843 | 0.299017 | 1.000 | 2 | 1 |
| BANK_C | D | 4 | 1.377636 | 1.264002 | 1.000 | 2 | 2 |
| BANK_D | C | 4 | 2.718256 | 3.211916 | 1.000 | 2 | 2 |
| BANK_D | D | 4 | 3.927618 | 3.757652 | 1.000 | 0 | 3 |
| ALL | C | 8 | 1.527050 | 0.533464 | 1.000 | 4 | 3 |
| ALL | D | 8 | 2.652627 | 1.899500 | 1.000 | 2 | 5 |

- Target Cache operation distributions and raw/masked operation probabilities are in the JSON under `distributions.cache` and each witness `policy_actions`.

## Paired UAV witness

`DELTA_G30_MOVE = G30(POLICY_MOVE) - G30(STAY)`; an original MOVE with negative delta is an OVER_MOVE witness.

| Scope | Continuation | N | Mean regret | Median regret | OVER_MOVE | Original MOVE N | Median ΔG30(move) | Negative Δ rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BANK_C | C | 4 | 0.037531 | 0.036649 | 4 | 4 | -0.036649 | 1.000 |
| BANK_C | D | 4 | 0.045498 | 0.038123 | 3 | 4 | -0.036047 | 0.750 |
| BANK_D | C | 4 | 0.008230 | 0.004763 | 2 | 4 | -0.002135 | 0.500 |
| BANK_D | D | 4 | 0.056105 | 0.046311 | 4 | 4 | -0.046311 | 1.000 |
| ALL | C | 8 | 0.022881 | 0.028584 | 6 | 8 | -0.028584 | 0.750 |
| ALL | D | 8 | 0.050801 | 0.046311 | 7 | 8 | -0.045228 | 0.875 |

## Decisions

- `V10D_CACHE_DECISION_QUALITY_IMPROVED = NOT_SUPPORTED`
- `V10D_UAV_OVER_MOVE_PERSISTS = YES`
- `NEXT_ROOT_CAUSE = NOT_TRIGGERED_BY_BOTH_GATES` (candidate mechanism, not causal proof).
- `V10D_5E4_REPLICATION_RECOMMENDED = WAIT`; no 5e-4 run was executed in this stage.

## Integrity and scope closure

- `TRAINING_EXECUTED = NO`; `SYSTEM_MODEL_CHANGED = NO`; `PPO_CHANGED = NO`; `FINAL_TEST_SEEDS_USED = NO`.
- `COMMON_FUTURE_CHECK = PASS`; `EPISODE_REGENERATION = NO`; `TEMP_FILES_REMOVED = YES` after runner cleanup.
- `git diff --check`: PASS (exit code 0; Git also emitted the repository's existing LF→CRLF normalization warnings for unrelated files); `PUSHED = NO` and no commit was created in this stage.

Final status: `STATE_BANK_C_COUNT = 4`; `STATE_BANK_D_COUNT = 4`; `CACHE_TARGET_STATES = 8`; `UAV_TARGET_STATES = 8`; `GIT_DIFF_CHECK = PASS`; `TEMP_FILES_REMOVED = YES`; `PUSHED = NO`.

Raw evidence: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\v10d_post_training_paired_witness.json`

