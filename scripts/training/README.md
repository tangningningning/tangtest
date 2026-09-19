# Training entrypoint

The PPO implementation remains in `src/ppo_simple/trainer.py` and is exposed
as `ppo-train` by `pyproject.toml`.  This directory is reserved for training
orchestration wrappers; no second training implementation is maintained here.

The current mainline entry is:

```text
ppo-train --config configs/proposed/system_config_ppo_recovery_r2_6e4.toml
```

Recovery R2 has passed its bounded implementation gate; formal training is
not yet completed.
