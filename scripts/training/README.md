# Training entrypoints

The PPO implementation remains in `src/ppo_simple/trainer.py`. The current
formal actor-learning-rate sweep consists of four independent Stage2D V3
fresh-initialization runners, each using the shared
`stage2d_v3_runner_common.py` implementation and
`configs/proposed/system_config.toml`:

```text
python scripts/training/run_stage2d_v3_fresh300_actor_lr_1e_minus_4.py
python scripts/training/run_stage2d_v3_fresh300_actor_lr_5e_minus_4.py
python scripts/training/run_stage2d_v3_fresh300_actor_lr_1e_minus_3.py
python scripts/training/run_stage2d_v3_fresh300_actor_lr_5e_minus_3.py
```

Each runner has its own `runs/stage2d_v3_fresh300_actor_lr_*` directory. They
share the existing Stage2D V3 protocol: architecture version 3, 300 epochs,
checkpoint interval 10, training seeds `20001..20300`, and no validation or
test during training. The configured critic learning rate is retained; the
actor learning rate is the value encoded by each runner's filename.
