# Minimal PPO v2 fixed-length rollout chunks

## Scope and external reference

This change adds standard PPO rollout-horizon behavior to `src/ppo_simple/` only. It does not change the environment, reward, Cache/UAV optimizers, Matching/KKT, Cloud, prediction artifacts, worker counts, formal parameters, train/test split, Gate, WHAT, or Direct-CF.

The design was checked against the rollout-horizon and bootstrap pattern in [CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py), including `num_steps` and reverse-time GAE bootstrap, and the `n_steps`/rollout-buffer contract in [Stable-Baselines3 PPO](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py) and its [rollout buffer](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py). No framework code was copied. The referenced repositories are MIT licensed ([CleanRL](https://github.com/vwxyzjn/cleanrl/blob/master/LICENSE), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE)).

## Rollout-chunk contract

The CLI now accepts:

```text
python -m ppo_simple.trainer --config system_config.toml --episodes 1 --rollout-steps 16 --run-dir PATH
```

`--rollout-steps` is configurable at the trainer interface and defaults to `16`; it is not a PPO loss or network constant. `episodes` still counts complete formal train episodes over origins `0..239`.

For each chunk:

1. collect the next consecutive train origins;
2. pass the previous chunk's final cache state and UAV positions into the next chunk;
3. mark `done=True` only at true train terminal origin `239`;
4. use the critic value of the next origin observation as the GAE bootstrap when the chunk boundary is non-terminal;
5. compute the existing GAE and clipped PPO update;
6. save a checkpoint before continuing.

The configured `gamma=1.0` and `gae_lambda=0.95` remain unchanged. A non-terminal chunk therefore has a false final `done` flag and a non-null bootstrap observation; the terminal chunk uses bootstrap value zero.

## Checkpoint and resume

Checkpoint version `2` preserves the existing policy/optimizer/config/RNG fields and additionally stores:

- completed episode count;
- current train-origin cursor;
- current cache state;
- current UAV positions;
- configured rollout chunk length.

After a mid-episode checkpoint, `--resume` restores the policy, optimizer, RNG, cursor, cache state, and UAV state, then continues from the saved origin. A checkpoint at origin `239` records the next cursor as `240`; if more episodes remain, the next episode explicitly resets to the formal initial state.

## Minimal validation

Only the requested bounded validation was run. No full episode, 16-slot run, 240-slot run, PPO training run, performance benchmark, or other audit was run.

| Check | Result |
|---|---|
| syntax/import | PASS |
| CLI parse with `--rollout-steps 4` | PASS |
| real train origins | `0..3` for first chunk |
| first chunk transitions | `4` |
| first chunk terminal flags | all false; non-terminal bootstrap present |
| first PPO update | PASS; 4 optimizer minibatch steps |
| temporary checkpoint save | PASS |
| saved cursor | `4` |
| saved cache/UAV state | matched the first chunk's final state |
| resume start | origin `4`, not origin `0` |
| resumed transitions | origins `4..7`, `4` transitions |
| resumed PPO update | PASS; optimizer step count advanced from `4` to `8` |
| finite update/gradient checks | PASS |
| temporary script/checkpoint cleanup | PASS |
| `__pycache__` cleanup | PASS |
| `git diff --check` | PASS; existing line-ending warning only |

The real-origin validation used a test-only neutral request fixture to keep the bounded check from entering the known long configuration path. It did not modify source policy code or project configuration.

## Files changed

- `src/ppo_simple/rollout.py`: accepts carried environment state, distinguishes chunk truncation from true terminal, and exposes the next observation for critic bootstrap.
- `src/ppo_simple/trainer.py`: adds configurable chunk collection, bootstrap value evaluation, per-update checkpointing, cursor/state/RNG resume, and `--rollout-steps`.
- `reports/minimal_ppo_v2_chunked_rollout.md`: this validation record.

`system_config.toml`, `src/decision/cache_configuration.py`, `adaptive_cache_mec.zip`, and `打包并上传.bat` were not changed by this task.

`PPO_V2_CHUNKED_READY = YES`
