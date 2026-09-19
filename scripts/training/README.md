# Training entrypoint

The PPO implementation remains in `src/ppo_simple/trainer.py` and is exposed
as `ppo-train` by `pyproject.toml`.  This directory is reserved for training
orchestration wrappers; no second training implementation is maintained here.

The current mainline entry is:

```text
ppo-train --config configs/proposed/system_config.toml
```

The proposed joint exact-Q listwise candidate implementation gate has passed;
formal training is not run by the mainline consolidation task.
