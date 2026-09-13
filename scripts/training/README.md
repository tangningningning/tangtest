# Training entrypoint

The PPO implementation remains in `src/ppo_simple/trainer.py` and is exposed
as `ppo-train` by `pyproject.toml`.  This directory is reserved for training
orchestration wrappers; no second training implementation is maintained here.
