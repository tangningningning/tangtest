# Current PPO V6

Current algorithm:
PPO V6 + grouped Cache/UAV primitive policy + separate actor/critic + PopArt normalized critic

Active config:
`system_config_ppo_primitive_v6_horizon_grouped_separate_optimization_popart_actorlr1e4.toml`

Checkpoint version:
6

Formal development run:
`ppo_v6_popart_actor1e4_fresh_20260910`

Training entry:
`ppo-train`

Evaluation entry:
`ppo-eval`

The V3, V4, and V5 classes remain as V6 inheritance implementation details.
They are not active historical experiments, and their module hierarchy is retained for checkpoint compatibility.
