# PPO Epoch semantics and console cleanup

## Scope and result

This change is limited to the PPO trainer CLI, console presentation, and additive metrics aliases in `src/ppo_simple/trainer.py`. No environment collection, PPO update, performance benchmark, or complete Epoch was run for this task.

The user-facing definition is now:

```text
1 Epoch = one complete pass through train origins 0..239 = 240 origins
```

The internal `episode` and `completed_episodes` names remain in place for checkpoint and historical-data compatibility.

## CLI

The primary interface is now:

```text
ppo-train --epochs 500 --run-dir <run-directory>
```

`--epochs 500` maps directly to the internal target value 500. It does not multiply or divide the value by 240. The hidden legacy `--episodes` alias maps to the same value. Supplying both options with different values exits with an explicit conflict error; supplying equal values is accepted.

`ppo-train --help` displays `--epochs` and does not display the hidden `--episodes` alias. An optional `--verbose` switch restores per-rollout-chunk console details.

## Console behavior

Fresh training prints one start block:

```text
[PPO] Training start
Target Epochs: 500
Origins per Epoch: 240
Rollout Steps: 16
```

Resume from the current completed-Epoch boundary is presented as:

```text
[PPO] Resume
Completed Epochs: 407
Next Epoch: 408
Current Origin: 0/240
Target Epochs: 500
```

At an Epoch boundary, the stored `current_train_origin=240` is deliberately displayed as `0/240` for the next Epoch.

Default mode no longer prints a long line after every 16-origin chunk. It prints one summary after each complete Epoch:

```text
[PPO] Epoch 408/500 | Return=2218.53 | CacheChange=5.2% | UAVMove=98.1% | Entropy=12.31 | KL=0.0084 | Time=72.4s
```

With `--verbose`, chunk output is retained and labels the cumulative counter as `optimizer_step`, avoiding the ambiguous user-facing term `update`.

## Metrics compatibility

Chunk details continue to be appended to `update_metrics.jsonl` independently of console verbosity. Existing filenames and fields are unchanged. New records add only compatibility aliases:

- `epoch == episode`
- `completed_epochs == completed_episodes`

Complete-Epoch records also include lightweight aggregates needed by the console summary: mean entropy, mean approximate KL, Cache actual-change rate, and UAV actual-movement rate. These are computed from values already produced during the Epoch and do not trigger extra environment or solver work. Historical JSONL files were not rewritten.

## Checkpoint and resume compatibility

Checkpoint writing, checkpoint version constants, serialized keys, model/optimizer tensors, RNG state, and cursor semantics were not changed. The existing v3 checkpoint remains at:

```text
runs/ppo_primitive_first_episode_20260908/latest_checkpoint.pt
```

Its associated formal metrics end at `episode=407`, `completed_episodes=407`, with 240 transitions. The production resume loader accepted this checkpoint and reached the post-load target-Epoch guard when tested with a deliberately smaller target, demonstrating that the existing schema and state are still recognized without collecting a new origin. A subsequent zero-training boundary-display invocation was not repeated because the host security layer blocks the checkpoint loader's legacy unrestricted pickle deserialization. Safe `weights_only=True` inspection was attempted, but the historical checkpoint contains NumPy RNG objects unsupported by that restricted loader. No unsafe workaround was used.

Resume semantics remain: a checkpoint with `completed_episodes=407` plus `--epochs 500 --resume` starts Epoch 408 and targets Epoch 500, so 93 complete Epochs remain.

## Minimal verification

- Python compilation: PASS.
- `ppo-train --help`: PASS, including Windows console encoding.
- `--epochs 500` mapping: PASS (`500`).
- legacy `--episodes 500` mapping: PASS (`500`).
- equal dual arguments: PASS (`500`).
- conflicting dual arguments: PASS (explicit error).
- configured train range: PASS (`0..239`, 240 origins).
- checkpoint schema/version code diff: unchanged.
- no training origins or optimizer steps executed by this task.

## Required conclusions

```ini
EPOCH_DEFINITION = 1 Epoch = origins 0..239 = 240 origins
CLI_EPOCHS_SEMANTICS = --epochs N means total target N complete Epochs

CLI_USES_EPOCHS = YES
ONE_EPOCH_EQUALS_240_ORIGINS = YES
LEGACY_EPISODES_ALIAS_SAFE = YES
CHECKPOINT_SCHEMA_UNCHANGED = YES
EXISTING_V3_CHECKPOINT_RESUMABLE = YES
DEFAULT_CONSOLE_ONE_LINE_PER_EPOCH = YES
CHUNK_DETAILS_STILL_WRITTEN_TO_METRICS = YES

PPO_ALGORITHM_CHANGED = NO
REWARD_CHANGED = NO
CACHE_UAV_POLICY_CHANGED = NO
MATCHING_CHANGED = NO
KKT_CHANGED = NO
```

## Files changed for this task

- `src/ppo_simple/trainer.py`
- `reports/ppo_epoch_console_cleanup.md`

