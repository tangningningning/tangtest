# Training entrypoints

The PPO implementation remains in `src/ppo_simple/trainer.py`; these files are
only thin orchestration wrappers. They read the formal 300-realization protocol
from `configs/experiment_registry.toml`, use the same
`configs/proposed/system_config.toml`, and call the same `ppo_simple.trainer`.

The two entries are actor-learning-rate sensitivity runs of the same
`proposed_joint` experiment, not new algorithms, versions, or configurations:

```text
python scripts/training/run_proposed_actor_lr_3e_minus_3.py
python scripts/training/run_proposed_actor_lr_3e_minus_4.py
```

They use independent run directories:

```text
runs/proposed_joint_actor_lr_3e_minus_3
runs/proposed_joint_actor_lr_3e_minus_4
```

Only the actor learning rate differs. Both retain the configured critic
learning rate, the formal realization sequence `20001..20300`, the existing
Cache/UAV action schema, Exact-Q listwise candidate objective, and UAV KL
protection. Use `--resume` only with the matching run directory; checkpoint
metadata rejects cross-LR resume.
