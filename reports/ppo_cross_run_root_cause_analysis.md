# PPO same-realization aligned cross-run root-cause analysis

This report uses only the six existing formal training runs and their persisted logs. It does not establish causality; correlations and lagged associations are diagnostic evidence only.

## Run validation

| label | run_dir | config | actor LR | critic LR | target KL | data protocol | seed | status | completed | episode rows | update rows | git dirty | git commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v9_joint_cache_uav_stage1_ep50_20001_20050 | C:\Users\tangning\Desktop\code\adaptive_cache_mec\configs\proposed\system_config_ppo_v9_joint_cache_uav.toml | 0.0001 | 0.0003 | NOT AVAILABLE | full_episode_randomized | 4401 | completed | 300 | 300 | 300 | False | 3588d695a8ea54482c26ca722d767b7853a2bb6a |
| 5e-4 | C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\lr_sweep_proposed_joint_actor_5e4 | C:\Users\tangning\Desktop\code\adaptive_cache_mec\configs\lr_sweep\v9_actor_lr_5e4.toml | 0.0005 | 0.0003 | NOT AVAILABLE | full_episode_randomized | 4401 | completed | 300 | 300 | 300 | False | 3588d695a8ea54482c26ca722d767b7853a2bb6a |
| 7e-4 | C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\lr_sweep_v9_actor_7e4 | C:\Users\tangning\Desktop\code\adaptive_cache_mec\configs\lr_sweep\v9_actor_lr_7e4.toml | 0.0007 | 0.0003 | NOT AVAILABLE | full_episode_randomized | 4401 | completed | 300 | 300 | 300 | False | e96331e619a85e147c58042d3738ca2c974b602f |
| 1e-3 | C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\lr_sweep_v9_actor_1e3 | C:\Users\tangning\Desktop\code\adaptive_cache_mec\configs\lr_sweep\v9_actor_lr_1e3.toml | 0.001 | 0.0003 | NOT AVAILABLE | full_episode_randomized | 4401 | completed | 300 | 300 | 300 | False | 3588d695a8ea54482c26ca722d767b7853a2bb6a |
| 5e-4 + KL | C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\lr_sweep_v9_actor_5e4_kl_protection | C:\Users\tangning\Desktop\code\adaptive_cache_mec\configs\lr_sweep\v9_actor_lr_5e4_kl_protection.toml | 0.0005 | 0.0003 | 0.1 | full_episode_randomized | 4401 | completed | 300 | 300 | 300 | False | e96331e619a85e147c58042d3738ca2c974b602f |
| 1e-3 + KL | C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\lr_sweep_v9_actor_1e3_kl_protection | C:\Users\tangning\Desktop\code\adaptive_cache_mec\configs\lr_sweep\v9_actor_lr_1e3_kl_protection.toml | 0.001 | 0.0003 | 0.1 | full_episode_randomized | 4401 | completed | 300 | 300 | 300 | False | e96331e619a85e147c58042d3738ca2c974b602f |

Config path resolution:
1e-4: manifest `config_path` was `C:\Users\tangning\Desktop\code\adaptive_cache_mec\system_config_ppo_v9_joint_cache_uav.toml`; the persisted `config_sha256` uniquely resolves it to `C:\Users\tangning\Desktop\code\adaptive_cache_mec\configs\proposed\system_config_ppo_v9_joint_cache_uav.toml`.

The 1e-4 run is accepted from manifest metadata (`actor_learning_rate=1e-4`, `experiment_type=main_training`, completed status), not from its directory name alone. Its episode mapping is checked below and spans pool index 0..299 / realization IDs episode_20001..episode_20300.

## Same-realization alignment

SAME_REALIZATION_ALIGNMENT = YES
MISMATCH_COUNT = 0
FIRST_MISMATCH = NONE

Compared per epoch in both `episode_metrics.jsonl` and `update_metrics.jsonl`: `realization_pool_index`, normalized `realization_directory` basename, derived episode seed/realization ID, `training_episode_index`, episode number, global seed, and global slot range.

All six episode maps are identical: epoch 1 -> pool index 0 -> episode_20001, and epoch 300 -> pool index 299 -> episode_20300. The update maps are identical as well; no resume-induced offset was found.

## Episode fields and realization-noise evidence

Actual episode fields: `actor_gradient_norm_range`, `actor_optimizer_steps`, `actual_cache_update_count`, `actual_uav_movement_count`, `approximate_kl_mean`, `approximate_kl_range`, `best_epoch`, `best_epoch_return`, `cache_actual_change_rate`, `cache_change_given_trigger`, `cache_cost`, `cache_request_count`, `cache_trigger_rate`, `checkpoint`, `completed_episodes`, `completed_epochs`, `critic_explained_variance`, `critic_gradient_norm_range`, `critic_optimizer_steps`, `critic_target_mean`, `critic_target_std`, `critic_value_mean`, `critic_value_std`, `entropy_mean`, `episode`, `episode_actor_optimizer_steps`, `episode_critic_optimizer_steps`, `episode_optimizer_steps`, `episode_return`, `epoch`, `finite_checks`, `global_training_slot_end`, `global_training_slot_start`, `gradient_norm_range`, `kl_early_stop_triggered`, `mean_G`, `mean_advantage_for_uav_move`, `mean_advantage_for_uav_stay`, `mean_reward`, `median_advantage_for_uav_move`, `median_advantage_for_uav_stay`, `movement_cost`, `num_transitions`, `optimizer_steps`, `policy_loss_range`, `ppo_update_epochs_executed`, `ppo_updates`, `q_total`, `realization_directory`, `realization_pool_index`, `record_type`, `reward_std`, `rollout_runtime_seconds`, `seed`, `training_episode_index`, `uav_action_advantage_logging_available`, `uav_actual_movement_rate`, `uav_binary_entropy_mean`, `uav_head_entropy`, `uav_move_feasible_count`, `uav_move_feasible_rate`, `uav_move_given_selected`, `uav_move_given_trigger`, `uav_move_probability_gt_0_5_rate`, `uav_move_probability_mean`, `uav_move_probability_median`, `uav_move_probability_p10`, `uav_move_probability_p90`, `uav_move_selected_count`, `uav_move_selected_rate`, `uav_request_count`, `uav_stay_move_logit_margin_mean`, `uav_stay_move_logit_margin_median`, `uav_stay_probability_mean`, `uav_trigger_rate`, `update_runtime_seconds`, `value_loss_range`, `wall_clock_runtime_seconds`

MEDIAN_RESIDUAL_PEARSON = 0.4091
MEDIAN_RESIDUAL_SPEARMAN = 0.4093
INPUT_NOISE_SUPPORTED = YES
At least 4/6 runs have negative residual: 27.4% of the 281 valid MA20 epochs. Pairwise residual correlations are moderate rather than uniformly high (Pearson range 0.216..0.606; Spearman range 0.183..0.58). This supports common realization effects as a contributor, not as ground-truth environment difficulty or the sole explanation.

### Pearson residual correlation

| label | 1e-4 | 5e-4 | 7e-4 | 1e-3 | 5e-4 + KL | 1e-3 + KL |
| --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 1 | 0.466 | 0.288 | 0.509 | 0.414 | 0.409 |
| 5e-4 | 0.466 | 1 | 0.216 | 0.5 | 0.404 | 0.478 |
| 7e-4 | 0.288 | 0.216 | 1 | 0.253 | 0.336 | 0.216 |
| 1e-3 | 0.509 | 0.5 | 0.253 | 1 | 0.439 | 0.606 |
| 5e-4 + KL | 0.414 | 0.404 | 0.336 | 0.439 | 1 | 0.395 |
| 1e-3 + KL | 0.409 | 0.478 | 0.216 | 0.606 | 0.395 | 1 |

### Spearman residual correlation

| label | 1e-4 | 5e-4 | 7e-4 | 1e-3 | 5e-4 + KL | 1e-3 + KL |
| --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 1 | 0.467 | 0.356 | 0.512 | 0.409 | 0.405 |
| 5e-4 | 0.467 | 1 | 0.293 | 0.545 | 0.41 | 0.51 |
| 7e-4 | 0.356 | 0.293 | 1 | 0.286 | 0.37 | 0.183 |
| 1e-3 | 0.512 | 0.545 | 0.286 | 1 | 0.459 | 0.58 |
| 5e-4 + KL | 0.409 | 0.41 | 0.37 | 0.459 | 1 | 0.398 |
| 1e-3 + KL | 0.405 | 0.51 | 0.183 | 0.58 | 0.398 | 1 |

### Lowest 20 common difficulty-score epochs

| epoch | realization_id | difficulty_score | negative/6 | 1e-4 residual | 5e-4 residual | 7e-4 residual | 1e-3 residual | 5e-4 + KL residual | 1e-3 + KL residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 44 | episode_20044 | -1.32 | 4 | 191 | -1.08e+03 | -180 | 12.8 | -922 | -716 |
| 189 | episode_20189 | -1.24 | 6 | -407 | -521 | -877 | -526 | -298 | -416 |
| 280 | episode_20280 | -1.12 | 6 | -362 | -669 | -509 | -465 | -361 | -463 |
| 38 | episode_20038 | -1.03 | 4 | -756 | -803 | 9.09 | 488 | -505 | -240 |
| 186 | episode_20186 | -0.998 | 6 | -322 | -292 | -1.37e+03 | -519 | -282 | -387 |
| 273 | episode_20273 | -0.969 | 6 | -301 | -404 | -304 | -411 | -468 | -407 |
| 276 | episode_20276 | -0.875 | 6 | -308 | -352 | -326 | -568 | -400 | -411 |
| 188 | episode_20188 | -0.779 | 6 | -282 | -64.1 | -473 | -427 | -123 | -315 |
| 42 | episode_20042 | -0.777 | 5 | -552 | -65.9 | 512 | -136 | -1.26e+03 | -597 |
| 153 | episode_20153 | -0.77 | 4 | -270 | 155 | 163 | -436 | -380 | -527 |
| 149 | episode_20149 | -0.716 | 4 | -618 | 120 | -1.95e+03 | -35.6 | 401 | -858 |
| 121 | episode_20121 | -0.712 | 4 | -490 | -584 | 181 | -163 | -395 | 210 |
| 224 | episode_20224 | -0.68 | 6 | -96.2 | -175 | -252 | -523 | -340 | -313 |
| 103 | episode_20103 | -0.66 | 4 | -233 | -385 | 489 | -437 | 119 | -616 |
| 178 | episode_20178 | -0.657 | 6 | -174 | -310 | -243 | -569 | -462 | -198 |
| 196 | episode_20196 | -0.651 | 6 | -199 | -49.1 | -435 | -378 | -71.1 | -335 |
| 214 | episode_20214 | -0.644 | 6 | -200 | -246 | -251 | -292 | -347 | -372 |
| 185 | episode_20185 | -0.6 | 6 | -156 | -178 | -449 | -367 | -327 | -107 |
| 52 | episode_20052 | -0.6 | 4 | -530 | -1.05e+03 | 408 | -269 | 54.2 | -149 |
| 251 | episode_20251 | -0.596 | 6 | -41.2 | -181 | -192 | -338 | -327 | -244 |

### Worst-20 residual overlap by run

| label | worst 20 epochs | mean negative other runs | fraction with any other negative |
| --- | --- | --- | --- |
| 1e-4 | 208, 67, 104, 69, 220, 172, 79, 199, 57, 211, 235, 38, 53, 55, 77, 249, 94, 149, 42, 52 | 2.15 | 0.85 |
| 5e-4 | 148, 151, 111, 44, 52, 212, 38, 112, 280, 80, 117, 121, 100, 97, 109, 77, 79, 189, 23, 55 | 2.5 | 0.95 |
| 7e-4 | 300, 140, 149, 186, 63, 119, 189, 191, 183, 287, 28, 122, 184, 69, 23, 190, 197, 83, 80, 79 | 2.35 | 0.95 |
| 1e-3 | 114, 55, 89, 122, 45, 105, 225, 178, 276, 189, 224, 186, 84, 280, 127, 103, 153, 118, 155, 188 | 3.5 | 1 |
| 5e-4 + KL | 63, 42, 28, 44, 215, 29, 90, 93, 216, 26, 35, 38, 87, 273, 178, 96, 213, 202, 276, 121 | 2.45 | 0.8 |
| 1e-3 + KL | 149, 46, 44, 220, 103, 42, 101, 287, 171, 97, 217, 153, 110, 280, 105, 189, 276, 273, 226, 72 | 2.75 | 0.9 |

## Actor PPO update intensity

Actual update fields: `actor_gradient_norm`, `actor_gradient_norm_max`, `actor_gradient_norm_min`, `actor_optimizer_steps`, `advantage_normalization_mode`, `advantages_finite`, `approximate_kl`, `approximate_kl_max`, `approximate_kl_min`, `checkpoint`, `clip_fraction`, `completed_epochs`, `critic_gradient_norm`, `critic_gradient_norm_max`, `critic_gradient_norm_min`, `critic_optimizer_steps`, `critic_return_scale`, `critic_return_scale_count`, `entropy`, `episode`, `epoch`, `global_training_slot_end`, `global_training_slot_start`, `gradient_finite`, `gradient_norm`, `gradient_norm_max`, `gradient_norm_min`, `kl_early_stop_triggered`, `normalized_value_loss`, `num_transitions`, `optimizer_steps`, `origin_end`, `origin_start`, `policy_loss`, `policy_loss_max`, `policy_loss_min`, `ppo_update_epochs_executed`, `raw_value_loss`, `realization_directory`, `realization_pool_index`, `record_type`, `returns_finite`, `rollout_runtime_seconds`, `seed`, `terminal`, `total_loss`, `total_loss_finite`, `training_episode_index`, `update_runtime_seconds`, `value_loss`, `value_loss_max`, `value_loss_min`

The focal fields are `approximate_kl`, `clip_fraction`, `actor_gradient_norm`, `kl_early_stop_triggered`, `ppo_update_epochs_executed`, `optimizer_steps`, and `entropy`. `optimizer_steps_delta` is derived only as the difference of the logged cumulative `optimizer_steps`; it is not an invented optimizer metric.

For gradient interpretation, `src/ppo_simple/trainer.py` calls `torch.nn.utils.clip_grad_norm_` and immediately stores its returned norm before `optimizer.step` (actor path around lines 1660-1666; shared path around lines 1747-1753). The logged value is therefore the pre-rescaling total norm returned by that call; values above configured `max_grad_norm=0.5` are expected evidence of pre-clipping norm, not post-clipping gradients.

Each cell below is `mean / median / std` over the stated epoch window. `NOT AVAILABLE` means that field is absent from that run's actual log.

### approximate_kl

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 0.01484 / 0.01577 / 0.03214 | 0.0114 / 0.01048 / 0.03593 | 0.01626 / 0.02203 / 0.03227 | 0.01389 / 0.01585 / 0.02973 | 0.01633 / 0.01603 / 0.03084 | 0.01631 / 0.01572 / 0.0313 | 0.02198 / 0.01451 / 0.03849 |
| 5e-4 | 0.04663 / 0.04274 / 0.05622 | 0.03064 / 0.0293 / 0.05531 | 0.0426 / 0.04093 / 0.04559 | 0.05706 / 0.05753 / 0.05378 | 0.06227 / 0.06628 / 0.06222 | 0.04056 / 0.0302 / 0.05691 | 0.04339 / 0.03914 / 0.05438 |
| 7e-4 | 0.07712 / 0.07418 / 0.08091 | 0.06119 / 0.04139 / 0.09054 | 0.08675 / 0.09052 / 0.07658 | 0.07784 / 0.0756 / 0.07487 | 0.09043 / 0.07952 / 0.09 | 0.06939 / 0.07023 / 0.06621 | 0.06863 / 0.08391 / 0.06548 |
| 1e-3 | 0.1194 / 0.1067 / 0.1238 | 0.07542 / 0.07232 / 0.08468 | 0.1119 / 0.1173 / 0.1162 | 0.1565 / 0.117 / 0.1457 | 0.1086 / 0.1036 / 0.1198 | 0.1448 / 0.1174 / 0.1274 | 0.1247 / 0.1122 / 0.08885 |
| 5e-4 + KL | 0.04338 / 0.04259 / 0.05631 | 0.0276 / 0.02342 / 0.05724 | 0.03913 / 0.04058 / 0.05463 | 0.04518 / 0.04145 / 0.05495 | 0.05405 / 0.0645 / 0.06087 | 0.05093 / 0.04563 / 0.04926 | 0.05402 / 0.04623 / 0.05716 |
| 1e-3 + KL | 0.09839 / 0.08944 / 0.09585 | 0.08072 / 0.07323 / 0.07844 | 0.118 / 0.09648 / 0.09203 | 0.1045 / 0.08954 / 0.1131 | 0.1068 / 0.1011 / 0.08962 | 0.08196 / 0.08624 / 0.09701 | 0.1074 / 0.1038 / 0.1051 |

### clip_fraction

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 0.2629 / 0.2667 / 0.09711 | 0.2933 / 0.2833 / 0.09306 | 0.2906 / 0.2667 / 0.1037 | 0.2533 / 0.2333 / 0.08907 | 0.2567 / 0.2333 / 0.09276 | 0.2206 / 0.2333 / 0.08715 | 0.2 / 0.2 / 0.08498 |
| 5e-4 | 0.4804 / 0.4833 / 0.1208 | 0.4833 / 0.4833 / 0.1215 | 0.5006 / 0.4667 / 0.1235 | 0.4778 / 0.5 / 0.1149 | 0.4878 / 0.5 / 0.1194 | 0.4528 / 0.4667 / 0.1193 | 0.4667 / 0.4833 / 0.1362 |
| 7e-4 | 0.5931 / 0.6 / 0.1144 | 0.5789 / 0.6 / 0.1066 | 0.6144 / 0.6333 / 0.1202 | 0.615 / 0.6 / 0.09377 | 0.6361 / 0.6333 / 0.1015 | 0.5211 / 0.5333 / 0.1112 | 0.4483 / 0.4833 / 0.1051 |
| 1e-3 | 0.6618 / 0.6667 / 0.1028 | 0.64 / 0.6333 / 0.07144 | 0.6872 / 0.7 / 0.1106 | 0.6856 / 0.7 / 0.1069 | 0.6433 / 0.6333 / 0.1053 | 0.6528 / 0.6667 / 0.1044 | 0.665 / 0.6667 / 0.08659 |
| 5e-4 + KL | 0.4808 / 0.4667 / 0.111 | 0.4944 / 0.4833 / 0.1013 | 0.4922 / 0.5 / 0.1257 | 0.475 / 0.4667 / 0.1064 | 0.4772 / 0.4667 / 0.1212 | 0.465 / 0.4667 / 0.09476 | 0.475 / 0.5 / 0.09711 |
| 1e-3 + KL | 0.5981 / 0.6 / 0.1218 | 0.5817 / 0.6 / 0.1051 | 0.6339 / 0.6333 / 0.1371 | 0.6128 / 0.6 / 0.1155 | 0.5917 / 0.6 / 0.1149 | 0.5706 / 0.5667 / 0.1234 | 0.5983 / 0.5667 / 0.1157 |

### actor_gradient_norm

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 2.309 / 2.3 / 0.6295 | 1.964 / 1.909 / 0.5104 | 2.396 / 2.423 / 0.6198 | 2.65 / 2.664 / 0.5351 | 2.31 / 2.272 / 0.5967 | 2.224 / 2.209 / 0.6668 | 1.962 / 1.91 / 0.5412 |
| 5e-4 | 1.891 / 1.86 / 0.5465 | 1.676 / 1.672 / 0.575 | 1.942 / 1.904 / 0.5052 | 1.804 / 1.737 / 0.4933 | 1.934 / 1.972 / 0.5392 | 2.096 / 2.019 / 0.5228 | 1.976 / 1.992 / 0.4852 |
| 7e-4 | 1.83 / 1.735 / 0.6214 | 1.38 / 1.336 / 0.4781 | 1.763 / 1.735 / 0.5412 | 1.966 / 1.993 / 0.5979 | 2.14 / 2.142 / 0.6679 | 1.898 / 1.865 / 0.5284 | 2.053 / 1.981 / 0.5834 |
| 1e-3 | 1.665 / 1.634 / 0.6379 | 1.159 / 1.049 / 0.4893 | 1.62 / 1.646 / 0.5544 | 2.026 / 1.968 / 0.6107 | 1.688 / 1.612 / 0.4897 | 1.833 / 1.759 / 0.6763 | 2.085 / 1.969 / 0.6997 |
| 5e-4 + KL | 1.758 / 1.725 / 0.5066 | 1.569 / 1.489 / 0.5266 | 1.8 / 1.746 / 0.4983 | 1.734 / 1.646 / 0.4376 | 1.684 / 1.711 / 0.484 | 2.005 / 1.885 / 0.477 | 2.051 / 1.974 / 0.5401 |
| 1e-3 + KL | 1.8 / 1.758 / 0.6307 | 1.586 / 1.478 / 0.6772 | 1.816 / 1.769 / 0.6012 | 1.814 / 1.789 / 0.5793 | 1.912 / 1.782 / 0.6384 | 1.873 / 1.818 / 0.6012 | 1.721 / 1.763 / 0.4819 |

### entropy

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 15.73 / 14.66 / 4.675 | 22.18 / 22.09 / 2.174 | 19.3 / 19.29 / 1.294 | 14.93 / 14.66 / 1.444 | 11.87 / 11.78 / 0.7404 | 10.37 / 10.36 / 1.275 | 8.918 / 8.754 / 0.4589 |
| 5e-4 | 16.9 / 17.25 / 3.878 | 21.81 / 20.84 / 2.267 | 19.54 / 19.4 / 1.026 | 17.26 / 17.25 / 1.122 | 14.02 / 14.16 / 1.261 | 11.87 / 11.97 / 1.177 | 12.27 / 12.51 / 1.106 |
| 7e-4 | 17.08 / 17.82 / 3.761 | 21.08 / 20.93 / 1.669 | 19.68 / 19.68 / 1.067 | 17.39 / 17.39 / 1.19 | 16.18 / 16.68 / 2.093 | 11.09 / 10.89 / 1.197 | 10.2 / 10.24 / 0.7185 |
| 1e-3 | 12.73 / 12.15 / 5.365 | 20.47 / 20.56 / 2.407 | 16.26 / 16.43 / 1.105 | 12.43 / 12.15 / 1.867 | 7.601 / 7.651 / 0.7772 | 6.911 / 6.933 / 0.4822 | 6.869 / 6.935 / 0.5247 |
| 5e-4 + KL | 17.08 / 16.78 / 3.719 | 22.49 / 22.08 / 1.654 | 19.27 / 19.15 / 0.9438 | 16.9 / 16.78 / 1.334 | 13.61 / 13.61 / 1.184 | 13.13 / 13.21 / 0.7996 | 12.86 / 12.97 / 0.9009 |
| 1e-3 + KL | 14.68 / 14.12 / 3.947 | 19.02 / 18.07 / 2.224 | 18.37 / 18.59 / 1.065 | 13.54 / 13.26 / 1.742 | 13.39 / 13.55 / 0.917 | 9.064 / 9.081 / 0.9745 | 9.585 / 9.487 / 0.819 |

### KL early stop, executed PPO epochs, and optimizer-step delta

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |
| 5e-4 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |
| 7e-4 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 1e-3 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |
| 5e-4 + KL | 0.01333 / 0 / 0.1147 | 0 / 0 / 0 | 0.01667 / 0 / 0.128 | 0 / 0 / 0 | 0.03333 / 0 / 0.1795 | 0.01667 / 0 / 0.128 | 0 / 0 / 0 |
| 1e-3 + KL | 0.42 / 0 / 0.4936 | 0.3833 / 0 / 0.4862 | 0.5167 / 1 / 0.4997 | 0.3333 / 0 / 0.4714 | 0.5167 / 1 / 0.4997 | 0.35 / 0 / 0.477 | 0.5 / 0.5 / 0.5 |

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |
| 5e-4 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |
| 7e-4 | 4 / 4 / 0 | 4 / 4 / 0 | 4 / 4 / 0 | 4 / 4 / 0 | 4 / 4 / 0 | 4 / 4 / 0 | 4 / 4 / 0 |
| 1e-3 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |
| 5e-4 + KL | 3.977 / 4 / 0.2069 | 4 / 4 / 0 | 3.967 / 4 / 0.256 | 4 / 4 / 0 | 3.95 / 4 / 0.2843 | 3.967 / 4 / 0.256 | 4 / 4 / 0 |
| 1e-3 + KL | 3.29 / 4 / 0.9161 | 3.2 / 4 / 1.03 | 3.067 / 3 / 1.014 | 3.517 / 4 / 0.7414 | 3.183 / 3 / 0.9035 | 3.483 / 4 / 0.7636 | 3.25 / 3.5 / 0.8292 |

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 |
| 5e-4 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 |
| 7e-4 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 |
| 1e-3 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 | 40 / 40 / 0 |
| 5e-4 + KL | 39.77 / 40 / 2.069 | 40 / 40 / 0 | 39.67 / 40 / 2.56 | 40 / 40 / 0 | 39.5 / 40 / 2.843 | 39.67 / 40 / 2.56 | 40 / 40 / 0 |
| 1e-3 + KL | 32.9 / 40 / 9.161 | 32 / 40 / 10.3 | 30.67 / 30 / 10.14 | 35.17 / 40 / 7.414 | 31.83 / 30 / 9.035 | 34.83 / 40 / 7.636 | 32.5 / 35 / 8.292 |

### Lagged update-to-future-residual association

The following uses update epoch e against residual e+1, e+2, and e+5. These are associations, not causal estimates.

| label | update field | future residual | association |
| --- | --- | --- | --- |
| 1e-4 | approximate_kl | +1 | Pearson 0.01635; Spearman 0.0298 |
| 1e-4 | approximate_kl | +2 | Pearson -0.04249; Spearman -0.0375 |
| 1e-4 | approximate_kl | +5 | Pearson 0.02877; Spearman 0.03203 |
| 1e-4 | clip_fraction | +1 | Pearson 0.1086; Spearman 0.09092 |
| 1e-4 | clip_fraction | +2 | Pearson -0.002965; Spearman -0.02186 |
| 1e-4 | clip_fraction | +5 | Pearson 0.09329; Spearman 0.1188 |
| 1e-4 | actor_gradient_norm | +1 | Pearson -0.0474; Spearman -0.02152 |
| 1e-4 | actor_gradient_norm | +2 | Pearson 0.02534; Spearman 0.07506 |
| 1e-4 | actor_gradient_norm | +5 | Pearson -0.03581; Spearman -0.001565 |
| 1e-4 | kl_early_stop_triggered | +1 | NOT AVAILABLE |
| 1e-4 | kl_early_stop_triggered | +2 | NOT AVAILABLE |
| 1e-4 | kl_early_stop_triggered | +5 | NOT AVAILABLE |
| 5e-4 | approximate_kl | +1 | Pearson -0.001205; Spearman 0.0286 |
| 5e-4 | approximate_kl | +2 | Pearson -0.05919; Spearman -0.053 |
| 5e-4 | approximate_kl | +5 | Pearson 0.01215; Spearman 0.0009454 |
| 5e-4 | clip_fraction | +1 | Pearson 0.001884; Spearman 0.0002205 |
| 5e-4 | clip_fraction | +2 | Pearson -0.02283; Spearman -0.0382 |
| 5e-4 | clip_fraction | +5 | Pearson 0.1105; Spearman 0.0581 |
| 5e-4 | actor_gradient_norm | +1 | Pearson -0.002966; Spearman -0.03025 |
| 5e-4 | actor_gradient_norm | +2 | Pearson -0.1347; Spearman -0.127 |
| 5e-4 | actor_gradient_norm | +5 | Pearson -0.09413; Spearman -0.08938 |
| 5e-4 | kl_early_stop_triggered | +1 | NOT AVAILABLE |
| 5e-4 | kl_early_stop_triggered | +2 | NOT AVAILABLE |
| 5e-4 | kl_early_stop_triggered | +5 | NOT AVAILABLE |
| 7e-4 | approximate_kl | +1 | Pearson -0.1021; Spearman -0.1416 |
| 7e-4 | approximate_kl | +2 | Pearson -0.02382; Spearman -0.0286 |
| 7e-4 | approximate_kl | +5 | Pearson -0.08362; Spearman -0.08361 |
| 7e-4 | clip_fraction | +1 | Pearson -0.05201; Spearman -0.05118 |
| 7e-4 | clip_fraction | +2 | Pearson 0.0791; Spearman 0.0218 |
| 7e-4 | clip_fraction | +5 | Pearson -0.1123; Spearman -0.1458 |
| 7e-4 | actor_gradient_norm | +1 | Pearson -0.1139; Spearman -0.1845 |
| 7e-4 | actor_gradient_norm | +2 | Pearson -0.1449; Spearman -0.1882 |
| 7e-4 | actor_gradient_norm | +5 | Pearson -0.09572; Spearman -0.07983 |
| 7e-4 | kl_early_stop_triggered | +1 | NOT AVAILABLE |
| 7e-4 | kl_early_stop_triggered | +2 | NOT AVAILABLE |
| 7e-4 | kl_early_stop_triggered | +5 | NOT AVAILABLE |
| 1e-3 | approximate_kl | +1 | Pearson -0.08433; Spearman -0.1077 |
| 1e-3 | approximate_kl | +2 | Pearson 0.01857; Spearman 0.006367 |
| 1e-3 | approximate_kl | +5 | Pearson -0.112; Spearman -0.07828 |
| 1e-3 | clip_fraction | +1 | Pearson -0.1056; Spearman -0.05899 |
| 1e-3 | clip_fraction | +2 | Pearson -0.07312; Spearman -0.03017 |
| 1e-3 | clip_fraction | +5 | Pearson -0.07926; Spearman -0.05356 |
| 1e-3 | actor_gradient_norm | +1 | Pearson -0.2111; Spearman -0.215 |
| 1e-3 | actor_gradient_norm | +2 | Pearson -0.08896; Spearman -0.08462 |
| 1e-3 | actor_gradient_norm | +5 | Pearson -0.113; Spearman -0.1346 |
| 1e-3 | kl_early_stop_triggered | +1 | NOT AVAILABLE |
| 1e-3 | kl_early_stop_triggered | +2 | NOT AVAILABLE |
| 1e-3 | kl_early_stop_triggered | +5 | NOT AVAILABLE |
| 5e-4 + KL | approximate_kl | +1 | Pearson -0.09476; Spearman -0.05288 |
| 5e-4 + KL | approximate_kl | +2 | Pearson 0.04398; Spearman 0.1223 |
| 5e-4 + KL | approximate_kl | +5 | Pearson 0.04501; Spearman -0.005702 |
| 5e-4 + KL | clip_fraction | +1 | Pearson -0.02423; Spearman -0.01106 |
| 5e-4 + KL | clip_fraction | +2 | Pearson 0.01653; Spearman 0.02045 |
| 5e-4 + KL | clip_fraction | +5 | Pearson 0.01508; Spearman -0.03823 |
| 5e-4 + KL | actor_gradient_norm | +1 | Pearson -0.1006; Spearman -0.1276 |
| 5e-4 + KL | actor_gradient_norm | +2 | Pearson -0.03314; Spearman -0.06683 |
| 5e-4 + KL | actor_gradient_norm | +5 | Pearson -0.1203; Spearman -0.09724 |
| 5e-4 + KL | kl_early_stop_triggered | +1 | Pearson -0.08329; Spearman -0.1141 |
| 5e-4 + KL | kl_early_stop_triggered | +2 | Pearson -0.01126; Spearman -0.02704 |
| 5e-4 + KL | kl_early_stop_triggered | +5 | Pearson 0.03118; Spearman 0.03444 |
| 1e-3 + KL | approximate_kl | +1 | Pearson -0.06159; Spearman -0.06115 |
| 1e-3 + KL | approximate_kl | +2 | Pearson -0.006602; Spearman -0.02101 |
| 1e-3 + KL | approximate_kl | +5 | Pearson 0.008011; Spearman -0.03999 |
| 1e-3 + KL | clip_fraction | +1 | Pearson -0.04136; Spearman -0.03382 |
| 1e-3 + KL | clip_fraction | +2 | Pearson -0.04343; Spearman -0.01643 |
| 1e-3 + KL | clip_fraction | +5 | Pearson 0.04714; Spearman 0.0519 |
| 1e-3 + KL | actor_gradient_norm | +1 | Pearson -0.02171; Spearman -0.03076 |
| 1e-3 + KL | actor_gradient_norm | +2 | Pearson -0.132; Spearman -0.1632 |
| 1e-3 + KL | actor_gradient_norm | +5 | Pearson -0.07228; Spearman -0.08101 |
| 1e-3 + KL | kl_early_stop_triggered | +1 | Pearson -0.1392; Spearman -0.1532 |
| 1e-3 + KL | kl_early_stop_triggered | +2 | Pearson -0.05655; Spearman -0.05009 |
| 1e-3 + KL | kl_early_stop_triggered | +5 | Pearson -0.07558; Spearman -0.09914 |

ACTOR_OVERUPDATE_SUPPORTED = YES
Across non-KL LR settings, whole-run approximate KL means are [0.0148, 0.0466, 0.0771, 0.1194] and clip-fraction means are [0.2629, 0.4804, 0.5931, 0.6618]; both rise with actor LR. Actor norm exceeded the configured 0.5 threshold in fractions [1.0, 1.0, 1.0, 0.997, 1.0, 1.0] of logged updates. The 1e-3 + KL run early-stopped in a substantial fraction of updates and executed fewer update epochs on average, while lagged correlations are mostly weak and inconsistent, so this supports update-intensity association without a causal claim.

### Matched KL comparisons

| comparison | window | metric | first mean | second mean | second-first |
| --- | --- | --- | --- | --- | --- |
| 5e-4 vs 5e-4 + KL | 241-300 | approximate_kl | 0.04056 | 0.05093 | 0.01036 |
| 5e-4 vs 5e-4 + KL | 241-300 | clip_fraction | 0.4528 | 0.465 | 0.01222 |
| 5e-4 vs 5e-4 + KL | 241-300 | actor_gradient_norm | 2.096 | 2.005 | -0.09128 |
| 5e-4 vs 5e-4 + KL | 241-300 | entropy | 11.87 | 13.13 | 1.257 |
| 5e-4 vs 5e-4 + KL | 281-300 | approximate_kl | 0.04339 | 0.05402 | 0.01062 |
| 5e-4 vs 5e-4 + KL | 281-300 | clip_fraction | 0.4667 | 0.475 | 0.008333 |
| 5e-4 vs 5e-4 + KL | 281-300 | actor_gradient_norm | 1.976 | 2.051 | 0.07481 |
| 5e-4 vs 5e-4 + KL | 281-300 | entropy | 12.27 | 12.86 | 0.5944 |
| 1e-3 vs 1e-3 + KL | 241-300 | approximate_kl | 0.1448 | 0.08196 | -0.06281 |
| 1e-3 vs 1e-3 + KL | 241-300 | clip_fraction | 0.6528 | 0.5706 | -0.08222 |
| 1e-3 vs 1e-3 + KL | 241-300 | actor_gradient_norm | 1.833 | 1.873 | 0.03996 |
| 1e-3 vs 1e-3 + KL | 241-300 | entropy | 6.911 | 9.064 | 2.153 |
| 1e-3 vs 1e-3 + KL | 281-300 | approximate_kl | 0.1247 | 0.1074 | -0.01729 |
| 1e-3 vs 1e-3 + KL | 281-300 | clip_fraction | 0.665 | 0.5983 | -0.06667 |
| 1e-3 vs 1e-3 + KL | 281-300 | actor_gradient_norm | 2.085 | 1.721 | -0.3639 |
| 1e-3 vs 1e-3 + KL | 281-300 | entropy | 6.869 | 9.585 | 2.715 |

## Cache behavior drift

The actual cache fields used are `cache_actual_change_rate`, `cache_trigger_rate`, and `cache_cost`. No cache-content/service identity is inferred.

### cache_actual_change_rate

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 0.4006 / 0.3602 / 0.09593 | 0.5518 / 0.5593 / 0.06982 | 0.4433 / 0.4372 / 0.03576 | 0.36 / 0.3567 / 0.0208 | 0.3339 / 0.337 / 0.01944 | 0.3142 / 0.325 / 0.02848 | 0.2784 / 0.2774 / 0.01278 |
| 5e-4 | 0.463 / 0.4574 / 0.07215 | 0.5563 / 0.5463 / 0.07521 | 0.4576 / 0.4544 / 0.03101 | 0.4477 / 0.4426 / 0.05898 | 0.4362 / 0.4337 / 0.03183 | 0.4173 / 0.4135 / 0.05602 | 0.4765 / 0.4793 / 0.02084 |
| 7e-4 | 0.5233 / 0.5385 / 0.0934 | 0.5757 / 0.5667 / 0.0505 | 0.5546 / 0.5515 / 0.03335 | 0.5116 / 0.5231 / 0.04536 | 0.5841 / 0.5998 / 0.07684 | 0.3903 / 0.4183 / 0.08221 | 0.2967 / 0.3 / 0.03227 |
| 1e-3 | 0.5409 / 0.537 / 0.05166 | 0.5947 / 0.5881 / 0.04196 | 0.5411 / 0.5474 / 0.03994 | 0.5683 / 0.5737 / 0.02884 | 0.5223 / 0.5181 / 0.02638 | 0.4779 / 0.4817 / 0.02234 | 0.4809 / 0.4806 / 0.02299 |
| 5e-4 + KL | 0.4382 / 0.4296 / 0.1195 | 0.6149 / 0.618 / 0.04638 | 0.5071 / 0.5061 / 0.03351 | 0.4276 / 0.4296 / 0.0358 | 0.3179 / 0.3181 / 0.05255 | 0.3238 / 0.325 / 0.02162 | 0.3229 / 0.3239 / 0.01785 |
| 1e-3 + KL | 0.4938 / 0.4787 / 0.05769 | 0.4751 / 0.4581 / 0.07291 | 0.5188 / 0.5204 / 0.0335 | 0.4569 / 0.4493 / 0.02678 | 0.5554 / 0.558 / 0.03979 | 0.4631 / 0.4596 / 0.02882 | 0.4822 / 0.4704 / 0.0319 |

### cache_cost

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 69.55 / 58.83 / 25.4 | 109.1 / 110.1 / 14.46 | 83.53 / 81.02 / 9.759 | 59.77 / 58.76 / 5.426 | 51.94 / 52.68 / 3.85 | 43.43 / 44.82 / 5.327 | 36.75 / 36.33 / 2.082 |
| 5e-4 | 75.49 / 72.75 / 18.46 | 100.4 / 95.29 / 18.9 | 80.02 / 79.56 / 7.925 | 74.68 / 74.3 / 8.649 | 63.24 / 62.31 / 6.252 | 59.18 / 59.44 / 10.69 | 69.47 / 70.31 / 4.009 |
| 7e-4 | 87.46 / 86.77 / 16.38 | 103.5 / 103.2 / 14.51 | 85.88 / 84.79 / 7.324 | 85.38 / 87.18 / 6.988 | 94.37 / 96.07 / 11.56 | 68.18 / 73.38 / 14.54 | 52.04 / 50.43 / 6.494 |
| 1e-3 | 83.83 / 80.62 / 13.51 | 103.3 / 99.09 / 12.51 | 87.12 / 88.4 / 8.688 | 81.41 / 82.12 / 4.967 | 75.12 / 74.13 / 5.839 | 72.13 / 72.11 / 3.366 | 72.69 / 72.2 / 3.469 |
| 5e-4 + KL | 70.54 / 64.14 / 25.28 | 110.1 / 110.6 / 13.92 | 83.36 / 82.5 / 6.307 | 64.61 / 64.14 / 6.686 | 46.73 / 46.15 / 7.138 | 47.93 / 47.33 / 5.176 | 44.27 / 44.12 / 3.264 |
| 1e-3 + KL | 78.54 / 75.06 / 12.27 | 81.34 / 73.28 / 18.43 | 84.47 / 84.36 / 8.043 | 69.07 / 67.36 / 5.575 | 86.1 / 86 / 7.208 | 71.7 / 70.72 / 5.328 | 74.97 / 73.7 / 5.969 |

CACHE_POLICY_DRIFT_SUPPORTED = YES
Early-to-final20 CacheChange deltas are [-0.2734, -0.0799, -0.2789, -0.1139, -0.292, 0.0072]; this is clear behavior drift in most runs. The direction and size of cache-cost changes are not consistent enough to make cache cost the primary explanation for all late return changes.

## UAV behavior drift

Actual UAV fields available in the logs include `uav_actual_movement_rate`; policy/mask diagnostics include `uav_move_probability_mean`, `uav_move_selected_rate`, `uav_stay_probability_mean`, `uav_stay_move_logit_margin_mean`, `uav_move_feasible_rate`, and `uav_move_given_selected` where present. Missing fields are not reconstructed.

### uav_actual_movement_rate

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 0.5002 / 0.4925 / 0.1218 | 0.582 / 0.56 / 0.05737 | 0.6538 / 0.655 / 0.03758 | 0.5116 / 0.4975 / 0.06463 | 0.3737 / 0.3692 / 0.03776 | 0.3799 / 0.3842 / 0.0548 | 0.3223 / 0.3125 / 0.03083 |
| 5e-4 | 0.5872 / 0.5917 / 0.1779 | 0.6387 / 0.6325 / 0.07247 | 0.7968 / 0.8 / 0.05501 | 0.7004 / 0.7558 / 0.1063 | 0.4246 / 0.4158 / 0.07358 | 0.3758 / 0.3642 / 0.05149 | 0.3427 / 0.3367 / 0.02348 |
| 7e-4 | 0.4559 / 0.4742 / 0.1572 | 0.4926 / 0.4367 / 0.137 | 0.601 / 0.605 / 0.06226 | 0.5606 / 0.5625 / 0.04538 | 0.4027 / 0.4058 / 0.07136 | 0.2227 / 0.2067 / 0.0564 | 0.287 / 0.2917 / 0.0397 |
| 1e-3 | 0.2505 / 0.2008 / 0.1859 | 0.4756 / 0.4875 / 0.05295 | 0.4281 / 0.4225 / 0.03859 | 0.2346 / 0.2008 / 0.09399 | 0.07961 / 0.07667 / 0.03074 | 0.03442 / 0.0275 / 0.01897 | 0.02275 / 0.02333 / 0.01432 |
| 5e-4 + KL | 0.6659 / 0.6633 / 0.06933 | 0.6123 / 0.5983 / 0.07033 | 0.7453 / 0.7433 / 0.02902 | 0.6821 / 0.6783 / 0.05173 | 0.6823 / 0.6825 / 0.03712 | 0.6076 / 0.6092 / 0.03296 | 0.5988 / 0.6025 / 0.03842 |
| 1e-3 + KL | 0.514 / 0.5142 / 0.2049 | 0.6636 / 0.6875 / 0.06774 | 0.7356 / 0.7333 / 0.04365 | 0.5333 / 0.4883 / 0.1301 | 0.4485 / 0.4542 / 0.05086 | 0.1892 / 0.1825 / 0.04802 | 0.1858 / 0.1875 / 0.03751 |

### policy and feasibility fields

| field | 1e-4 | 5e-4 | 7e-4 | 1e-3 | 5e-4 + KL | 1e-3 + KL |
| --- | --- | --- | --- | --- | --- | --- |
| uav_move_probability_mean | NOT AVAILABLE | NOT AVAILABLE | 0.2932 / 0.2989 / 0.04071 | NOT AVAILABLE | 0.5969 / 0.5956 / 0.03287 | 0.1837 / 0.1827 / 0.0412 |
| uav_move_selected_rate | NOT AVAILABLE | NOT AVAILABLE | 0.287 / 0.2917 / 0.0397 | NOT AVAILABLE | 0.5988 / 0.6025 / 0.03842 | 0.1858 / 0.1875 / 0.03751 |
| uav_stay_probability_mean | NOT AVAILABLE | NOT AVAILABLE | 0.7068 / 0.7011 / 0.04071 | NOT AVAILABLE | 0.4031 / 0.4044 / 0.03287 | 0.8163 / 0.8173 / 0.0412 |
| uav_stay_move_logit_margin_mean | NOT AVAILABLE | NOT AVAILABLE | 0.9448 / 0.9049 / 0.1805 | NOT AVAILABLE | -0.7759 / -0.8075 / 0.2337 | 2.263 / 2.313 / 0.3013 |
| uav_move_feasible_rate | NOT AVAILABLE | NOT AVAILABLE | 1 / 1 / 0 | NOT AVAILABLE | 1 / 1 / 0 | 1 / 1 / 0 |
| uav_move_given_selected | NOT AVAILABLE | NOT AVAILABLE | 1 / 1 / 0 | NOT AVAILABLE | 1 / 1 / 0 | 1 / 1 / 0 |

UAV_POLICY_COLLAPSE_SUPPORTED = PARTIAL
Clear early-to-final movement declines with a best-region-to-final return decline occur for: 1e-4, 5e-4, 7e-4, 1e-3, 1e-3 + KL. Feasibility diagnostics are not available for every run, so low movement is not universally labeled harmful collapse. Where available, feasibility remains near 1.0, arguing against masking as the explanation for those logged cases.

## Critic stability

Actual critic fields: `abs_value_residual`, `bootstrap_value`, `chunk_origin_start`, `critic_return_scale`, `done`, `episode`, `epoch`, `normalized_value_residual`, `old_value`, `optimizer_steps`, `origin`, `raw_advantage`, `raw_return`, `record_type`, `seed`, `value_residual`, `value_target`

### critic explained variance

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 0.8233 / 0.987 / 0.3069 | 0.2968 / 0.2045 / 0.292 | 0.8446 / 0.906 / 0.1494 | 0.9825 / 0.9879 / 0.02153 | 0.9942 / 0.9963 / 0.00919 | 0.9983 / 0.9985 / 0.0008295 | 0.9988 / 0.9989 / 0.0005755 |
| 5e-4 | 0.8224 / 0.9856 / 0.3227 | 0.2756 / 0.2983 / 0.3318 | 0.8631 / 0.9264 / 0.1541 | 0.9784 / 0.9851 / 0.0226 | 0.9962 / 0.9967 / 0.002005 | 0.9985 / 0.9987 / 0.0007808 | 0.9983 / 0.9984 / 0.0009863 |
| 7e-4 | 0.845 / 0.9902 / 0.3028 | 0.332 / 0.3806 / 0.3413 | 0.9128 / 0.9455 / 0.08718 | 0.9879 / 0.991 / 0.01042 | 0.9961 / 0.9974 / 0.00313 | 0.9965 / 0.9986 / 0.01363 | 0.9929 / 0.9983 / 0.02313 |
| 1e-3 | 0.8272 / 0.9943 / 0.3328 | 0.254 / 0.2296 / 0.3383 | 0.893 / 0.9485 / 0.1439 | 0.9927 / 0.9946 / 0.005375 | 0.9975 / 0.9979 / 0.001355 | 0.9987 / 0.9988 / 0.0005249 | 0.9988 / 0.9989 / 0.0005158 |
| 5e-4 + KL | 0.8323 / 0.991 / 0.3131 | 0.2939 / 0.3105 / 0.3201 | 0.8841 / 0.9325 / 0.1259 | 0.9879 / 0.9912 / 0.008603 | 0.997 / 0.9979 / 0.003337 | 0.9988 / 0.9989 / 0.0006461 | 0.9989 / 0.999 / 0.0005774 |
| 1e-3 + KL | 0.8644 / 0.9943 / 0.278 | 0.3915 / 0.3456 / 0.3144 | 0.9414 / 0.9732 / 0.07573 | 0.9931 / 0.9943 / 0.004213 | 0.9975 / 0.9977 / 0.001063 | 0.9984 / 0.9986 / 0.0009777 | 0.9982 / 0.9985 / 0.001288 |

### value loss and critic gradient

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 0.01385 / 0.001608 / 0.0504 | 0.05785 / 0.03012 / 0.101 | 0.007858 / 0.006525 / 0.005096 | 0.002124 / 0.001446 / 0.001956 | 0.001159 / 0.0005208 / 0.002868 | 0.0002589 / 0.0002051 / 0.0001698 | 0.0002195 / 0.0001909 / 0.0001208 |
| 5e-4 | 0.01231 / 0.001594 / 0.05113 | 0.04946 / 0.02054 / 0.106 | 0.008938 / 0.00691 / 0.006872 | 0.002323 / 0.001714 / 0.001861 | 0.0005386 / 0.0004157 / 0.0003731 | 0.000306 / 0.0002351 / 0.000218 | 0.0002877 / 0.0001922 / 0.0002557 |
| 7e-4 | 0.01366 / 0.001379 / 0.05162 | 0.05835 / 0.02513 / 0.1037 | 0.007129 / 0.00488 / 0.006648 | 0.00165 / 0.001053 / 0.002266 | 0.0006889 / 0.0004588 / 0.0007611 | 0.0004564 / 0.0002138 / 0.0008941 | 0.000597 / 0.0001574 / 0.001251 |
| 1e-3 | 0.01405 / 0.0009055 / 0.05712 | 0.06087 / 0.01996 / 0.1157 | 0.007725 / 0.004637 / 0.01165 | 0.001109 / 0.0008844 / 0.0009951 | 0.0003422 / 0.000248 / 0.0002305 | 0.0001959 / 0.0001508 / 0.0001301 | 0.0001738 / 0.0001306 / 0.0001204 |
| 5e-4 + KL | 0.01251 / 0.00133 / 0.05149 | 0.04969 / 0.02094 / 0.1064 | 0.01051 / 0.008198 / 0.01164 | 0.001691 / 0.001294 / 0.001263 | 0.0004384 / 0.0003556 / 0.0002924 | 0.0002451 / 0.0001521 / 0.0003885 | 0.0001412 / 0.0001347 / 6.017e-05 |
| 1e-3 + KL | 0.0217 / 0.001275 / 0.09647 | 0.09639 / 0.01925 / 0.198 | 0.009836 / 0.005354 / 0.01673 | 0.001304 / 0.0008775 / 0.00122 | 0.0006147 / 0.0004371 / 0.0004645 | 0.0003384 / 0.0002173 / 0.0003559 | 0.000329 / 0.000157 / 0.0004101 |

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 1755 / 885.3 / 2558 | 3795 / 2103 / 3909 | 2808 / 2330 / 1763 | 1107 / 803.3 / 945.2 | 853.1 / 400 / 2126 | 210.4 / 166.2 / 136.7 | 183.1 / 159.3 / 100.6 |
| 5e-4 | 1634 / 809.6 / 2211 | 3280 / 2696 / 3210 | 3061 / 2415 / 2160 | 1215 / 933.6 / 964.8 | 364.2 / 285.7 / 240.5 | 249.5 / 186.5 / 179 | 244.8 / 163 / 217 |
| 7e-4 | 1610 / 785.9 / 2181 | 3701 / 2954 / 2840 | 2480 / 1708 / 2129 | 1018 / 656.1 / 1449 | 487.1 / 323.4 / 529.5 | 364.7 / 172.9 / 721.9 | 488.3 / 128.1 / 1028 |
| 1e-3 | 2443 / 708.6 / 4708 | 6748 / 4549 / 6336 | 4142 / 2369 / 6035 | 857 / 689.9 / 723.3 | 301.8 / 217.3 / 203.7 | 167.2 / 128.6 / 111.1 | 148.7 / 111.7 / 103.1 |
| 5e-4 + KL | 1665 / 749.6 / 2705 | 3674 / 2493 / 3806 | 3159 / 2642 / 3292 | 947.6 / 747.7 / 651.3 | 335.8 / 271.7 / 223.8 | 206.9 / 131.2 / 316.5 | 123.3 / 118 / 52.64 |
| 1e-3 + KL | 2477 / 764.7 / 4811 | 6631 / 4771 / 5904 | 4255 / 2399 / 6916 | 822.8 / 571 / 756.3 | 430.1 / 308.1 / 323 | 247 / 157 / 259.7 | 243.7 / 116.5 / 304.1 |

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 0.3773 / 0.2459 / 0.4577 | 0.6734 / 0.4966 / 0.7734 | 0.4873 / 0.4963 / 0.3052 | 0.3071 / 0.1984 / 0.3022 | 0.2591 / 0.1706 / 0.2943 | 0.1594 / 0.1428 / 0.1101 | 0.1498 / 0.1338 / 0.09596 |
| 5e-4 | 0.3449 / 0.2619 / 0.3447 | 0.568 / 0.473 / 0.5167 | 0.4451 / 0.3728 / 0.3511 | 0.3353 / 0.3002 / 0.2414 | 0.2019 / 0.1475 / 0.146 | 0.1741 / 0.1252 / 0.1207 | 0.1388 / 0.1098 / 0.09569 |
| 7e-4 | 0.3506 / 0.2561 / 0.3544 | 0.6021 / 0.4391 / 0.4866 | 0.4434 / 0.3294 / 0.3768 | 0.3071 / 0.2799 / 0.2442 | 0.216 / 0.1489 / 0.1807 | 0.1844 / 0.1441 / 0.1931 | 0.1547 / 0.09688 / 0.2102 |
| 1e-3 | 0.307 / 0.2082 / 0.33 | 0.5843 / 0.4837 / 0.4196 | 0.4642 / 0.3985 / 0.3863 | 0.2308 / 0.1779 / 0.1707 | 0.1408 / 0.1142 / 0.09868 | 0.1151 / 0.075 / 0.09153 | 0.09816 / 0.06295 / 0.08814 |
| 5e-4 + KL | 0.3396 / 0.2039 / 0.3747 | 0.5423 / 0.4198 / 0.5183 | 0.5541 / 0.4569 / 0.4121 | 0.2886 / 0.1759 / 0.255 | 0.166 / 0.1154 / 0.1403 | 0.147 / 0.1082 / 0.1484 | 0.103 / 0.0969 / 0.07483 |
| 1e-3 + KL | 0.4552 / 0.2642 / 0.6932 | 1.028 / 0.6501 / 1.246 | 0.5843 / 0.4264 / 0.4783 | 0.2769 / 0.1933 / 0.2448 | 0.2285 / 0.1891 / 0.1678 | 0.1589 / 0.1141 / 0.1345 | 0.1347 / 0.06692 / 0.1403 |

### Critic residual and PopArt-related fields

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | -14.49 / -11.15 / 41.45 | -27.95 / -46.4 / 62.77 | -40.2 / -44.63 / 37.82 | -10.89 / -8.854 / 21.98 | 11.6 / 6.984 / 29.84 | -5 / -5.329 / 14.9 | -4.208 / -1.793 / 11.35 |
| 5e-4 | -16.57 / -12.6 / 42.16 | -43.05 / -59.71 / 61.31 | -34.75 / -37.42 / 33.82 | -17.58 / -27.6 / 36.58 | 3.366 / -0.5517 / 18.73 | 9.159 / 9.184 / 14.25 | -0.79 / -1.458 / 11.72 |
| 7e-4 | -15.6 / -11.84 / 45.01 | -52.28 / -65.45 / 48.8 | -37.91 / -40.32 / 31.89 | -3.033 / -9.904 / 41.58 | 7.128 / 0.2803 / 27.35 | 8.121 / 2.427 / 34.25 | 10.72 / -1.02 / 55.99 |
| 1e-3 | -18.21 / -9.874 / 44.68 | -62.04 / -73.72 / 63.56 | -36.12 / -38.48 / 35.93 | 5.269 / 8.664 / 17.56 | 4.635 / 8.233 / 23.21 | -2.815 / -5.257 / 15.83 | 2.889 / 1.815 / 12.83 |
| 5e-4 + KL | -14.85 / -9.137 / 38.35 | -33.57 / -49.27 / 61.07 | -38.61 / -34.53 / 32.48 | -6.089 / -7.002 / 19.92 | 0.9193 / -0.0336 / 19.53 | 3.087 / -0.2593 / 15.11 | -6.932 / -3.623 / 9.751 |
| 1e-3 + KL | -14.87 / -7.35 / 37.53 | -64.73 / -69.27 / 39.23 | -13.33 / -12.38 / 34.25 | -6.283 / -7.169 / 18.26 | 8.738 / 5.935 / 18.62 | 1.277 / -1.763 / 16.41 | -6.576 / -8.86 / 13.53 |

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 80.12 / 72.95 / 43.24 | 112.8 / 111.4 / 37.23 | 121.2 / 118.8 / 30.98 | 80.85 / 74.58 / 25.23 | 52.96 / 46.46 / 23.72 | 32.79 / 31.6 / 8.156 | 28.59 / 27.82 / 6.512 |
| 5e-4 | 84.21 / 75.58 / 48.14 | 123.9 / 112.9 / 39.91 | 131.9 / 127.4 / 37.58 | 82.83 / 77.32 / 25.23 | 48.55 / 46.52 / 13.9 | 33.77 / 32.46 / 8.927 | 31.77 / 31.61 / 8.738 |
| 7e-4 | 83.08 / 69.81 / 50.54 | 135.9 / 132.6 / 42.35 | 120.7 / 116.8 / 34.09 | 76.63 / 68.37 / 27.58 | 46.74 / 40.51 / 18.07 | 35.43 / 29.76 / 29.91 | 42.22 / 31.94 / 49.49 |
| 1e-3 | 87.06 / 61.87 / 62.36 | 159.6 / 162.8 / 46.66 | 142.8 / 139.4 / 45.99 | 65.12 / 60.62 / 20.15 | 39.29 / 36.6 / 12.06 | 28.5 / 28.39 / 6.15 | 27.32 / 27.04 / 6.407 |
| 5e-4 + KL | 77.17 / 69.71 / 46.07 | 115.6 / 114.4 / 32.18 | 125.6 / 119.8 / 37.39 | 74.36 / 71.57 / 22.6 | 41.03 / 39.07 / 12.37 | 29.19 / 28.63 / 8.301 | 25.77 / 25.28 / 7.168 |
| 1e-3 + KL | 82.37 / 56.09 / 59.34 | 162.6 / 164.9 / 49.28 | 121.3 / 117.7 / 40.63 | 57.73 / 53.74 / 16.94 | 38.96 / 37.51 / 9.491 | 31.18 / 30.74 / 8.589 | 29.62 / 29.57 / 8.987 |

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | -0.02349 / -0.01401 / 0.1496 | -0.04383 / -0.1181 / 0.3187 | -0.06706 / -0.07452 / 0.06339 | -0.01454 / -0.0112 / 0.02915 | 0.01352 / 0.008035 / 0.03449 | -0.005546 / -0.005899 / 0.01657 | -0.004609 / -0.001959 / 0.01244 |
| 5e-4 | -0.03178 / -0.01602 / 0.1309 | -0.08717 / -0.1553 / 0.2678 | -0.06192 / -0.06378 / 0.05954 | -0.02428 / -0.03776 / 0.05021 | 0.004282 / -0.0006901 / 0.02266 | 0.0102 / 0.009995 / 0.0157 | -0.0008506 / -0.001578 / 0.01271 |
| 7e-4 | -0.03691 / -0.01502 / 0.1391 | -0.1343 / -0.1805 / 0.2702 | -0.06457 / -0.06884 / 0.05435 | -0.003404 / -0.01228 / 0.0524 | 0.008602 / 0.0003396 / 0.03266 | 0.009094 / 0.002713 / 0.03786 | 0.01179 / -0.001132 / 0.06176 |
| 1e-3 | -0.02857 / -0.01077 / 0.1436 | -0.09935 / -0.1366 / 0.3013 | -0.05101 / -0.05168 / 0.05053 | 0.005709 / 0.009863 / 0.0197 | 0.004865 / 0.008748 / 0.02473 | -0.003051 / -0.005689 / 0.01713 | 0.003121 / 0.001964 / 0.01388 |
| 5e-4 + KL | -0.02836 / -0.01066 / 0.1293 | -0.06641 / -0.1245 / 0.269 | -0.0718 / -0.05828 / 0.06457 | -0.008284 / -0.009275 / 0.02627 | 0.001263 / -3.795e-05 / 0.02224 | 0.003429 / -0.000278 / 0.0164 | -0.007425 / -0.003875 / 0.01044 |
| 1e-3 + KL | -0.03726 / -0.009171 / 0.1705 | -0.1674 / -0.131 / 0.3455 | -0.02302 / -0.01884 / 0.05238 | -0.007816 / -0.008954 / 0.02278 | 0.01043 / 0.007138 / 0.02219 | 0.001491 / -0.002056 / 0.01919 | -0.007648 / -0.01029 / 0.01573 |

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 690 / 758.1 / 216.3 | 340 / 307.2 / 148.7 | 601 / 598.2 / 13.99 | 744.8 / 758.1 / 54.49 | 861.2 / 866 / 21.38 | 902.8 / 900.6 / 9.207 | 913.9 / 913.9 / 2.683 |
| 5e-4 | 680.7 / 732.9 / 207.3 | 337.4 / 358.6 / 106.6 | 602.9 / 597.7 / 58.42 | 728.2 / 732.9 / 21.11 | 830.6 / 841.4 / 27.12 | 904.3 / 908.1 / 18.09 | 922.8 / 922.7 / 1.61 |
| 7e-4 | 694.6 / 802.1 / 212.3 | 336.2 / 357.5 / 122.1 | 607.8 / 588.5 / 45.49 | 789.5 / 802.1 / 28.58 | 845.4 / 842.9 / 10.26 | 893.9 / 896.3 / 10.02 | 903.4 / 902.8 / 2.38 |
| 1e-3 | 797.6 / 892.1 / 195.6 | 481.3 / 572.3 / 201.1 | 751.7 / 748.1 / 55.71 | 892.1 / 892.1 / 27.36 | 938.7 / 941.2 / 5.108 | 923.9 / 923.8 / 1.309 | 924.5 / 924.2 / 0.7709 |
| 5e-4 + KL | 696.2 / 768.6 / 221.6 | 347 / 404.9 / 103 | 567.9 / 561 / 69.09 | 763.8 / 768.6 / 40.54 | 878.1 / 885.8 / 16.36 | 924.1 / 927.8 / 11.06 | 934.2 / 933.8 / 1.345 |
| 1e-3 + KL | 728.8 / 803.1 / 169 | 476 / 588.5 / 200.1 | 675.8 / 669.2 / 55.32 | 799.8 / 803.1 / 13.29 | 837.8 / 842.2 / 8.009 | 854.7 / 857 / 6.269 | 860.4 / 860.4 / 0.7162 |

| label | whole run | 1-60 | 61-120 | 121-180 | 181-240 | 241-300 | 281-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 4.515e+04 / 4.515e+04 / 2.598e+04 | 9150 / 9150 / 5195 | 2.715e+04 / 2.715e+04 / 5195 | 4.515e+04 / 4.515e+04 / 5195 | 6.315e+04 / 6.315e+04 / 5195 | 8.115e+04 / 8.115e+04 / 5195 | 8.715e+04 / 8.715e+04 / 1730 |
| 5e-4 | 4.515e+04 / 4.515e+04 / 2.598e+04 | 9150 / 9150 / 5195 | 2.715e+04 / 2.715e+04 / 5195 | 4.515e+04 / 4.515e+04 / 5195 | 6.315e+04 / 6.315e+04 / 5195 | 8.115e+04 / 8.115e+04 / 5195 | 8.715e+04 / 8.715e+04 / 1730 |
| 7e-4 | 4.515e+04 / 4.515e+04 / 2.598e+04 | 9150 / 9150 / 5195 | 2.715e+04 / 2.715e+04 / 5195 | 4.515e+04 / 4.515e+04 / 5195 | 6.315e+04 / 6.315e+04 / 5195 | 8.115e+04 / 8.115e+04 / 5195 | 8.715e+04 / 8.715e+04 / 1730 |
| 1e-3 | 4.515e+04 / 4.515e+04 / 2.598e+04 | 9150 / 9150 / 5195 | 2.715e+04 / 2.715e+04 / 5195 | 4.515e+04 / 4.515e+04 / 5195 | 6.315e+04 / 6.315e+04 / 5195 | 8.115e+04 / 8.115e+04 / 5195 | 8.715e+04 / 8.715e+04 / 1730 |
| 5e-4 + KL | 4.515e+04 / 4.515e+04 / 2.598e+04 | 9150 / 9150 / 5195 | 2.715e+04 / 2.715e+04 / 5195 | 4.515e+04 / 4.515e+04 / 5195 | 6.315e+04 / 6.315e+04 / 5195 | 8.115e+04 / 8.115e+04 / 5195 | 8.715e+04 / 8.715e+04 / 1730 |
| 1e-3 + KL | 4.515e+04 / 4.515e+04 / 2.598e+04 | 9150 / 9150 / 5195 | 2.715e+04 / 2.715e+04 / 5195 | 4.515e+04 / 4.515e+04 / 5195 | 6.315e+04 / 6.315e+04 / 5195 | 8.115e+04 / 8.115e+04 / 5195 | 8.715e+04 / 8.715e+04 / 1730 |

CRITIC_INSTABILITY_SUPPORTED = NO
Final20 EV values are [0.9988, 0.9983, 0.9929, 0.9988, 0.9989, 0.9982]; final20 normalized value losses are [0.000219, 0.000288, 0.000597, 0.000174, 0.000141, 0.000329]. EV improves to approximately 0.99 or higher and no late value-loss/critic-gradient explosion is visible, so Critic is not supported as a一级根因.

## Return / q / cost accounting

The logged identity is checked directly for every episode: `episode_return = q_total - cache_cost - movement_cost`.

Maximum absolute accounting error across all six runs = 9.71667e-13.
REWARD_ACCOUNTING_ERROR_FOUND = NO

Best-region-to-Final20 Return deltas are [-17.326, -136.392, -276.924, -179.578, -33.241, -135.185]; corresponding q deltas are [-29.604, -115.589, -301.244, -195.396, -34.812, -131.831]; combined cache-plus-movement cost deltas are [-12.279, 20.803, -24.32, -15.818, -1.57, 3.354]. The q-led pattern and mixed cost directions are calculated from the same logged windows. Therefore:

SYSTEM_COST_DOMINANCE_FOUND = NO

### Best-MA20 region -> Final20

| label | best MA20 window | Return | q_total | cache_cost | movement_cost | CacheChange | UAVMove | entropy | KL | clip | actor grad | critic EV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 253-272 | 3.34e+03 -> 3.33e+03 (delta -17.3) | 3.4e+03 -> 3.37e+03 (delta -29.6) | 46.6 -> 36.7 (delta -9.88) | 10.4 -> 7.96 (delta -2.4) | 0.337 -> 0.278 (delta -0.0583) | 0.412 -> 0.322 (delta -0.0897) | 11.2 -> 8.92 (delta -2.25) | 0.0224 -> 0.022 (delta -0.000434) | 0.225 -> 0.2 (delta -0.025) | 2.43 -> 1.96 (delta -0.468) | 0.998 -> 0.999 (delta 0.000613) |
| 5e-4 | 253-272 | 3.41e+03 -> 3.28e+03 (delta -136) | 3.47e+03 -> 3.35e+03 (delta -116) | 47.4 -> 69.5 (delta 22) | 9.65 -> 8.43 (delta -1.22) | 0.36 -> 0.476 (delta 0.117) | 0.381 -> 0.343 (delta -0.0378) | 10.8 -> 12.3 (delta 1.44) | 0.0342 -> 0.0434 (delta 0.00918) | 0.447 -> 0.467 (delta 0.02) | 2.19 -> 1.98 (delta -0.21) | 0.999 -> 0.998 (delta -0.000483) |
| 7e-4 | 253-272 | 3.34e+03 -> 3.06e+03 (delta -277) | 3.42e+03 -> 3.12e+03 (delta -301) | 79.1 -> 52 (delta -27) | 4.17 -> 6.87 (delta 2.7) | 0.459 -> 0.297 (delta -0.163) | 0.178 -> 0.287 (delta 0.109) | 11.5 -> 10.2 (delta -1.3) | 0.0854 -> 0.0686 (delta -0.0168) | 0.573 -> 0.448 (delta -0.125) | 1.79 -> 2.05 (delta 0.263) | 0.999 -> 0.993 (delta -0.00585) |
| 1e-3 | 158-177 | 3.31e+03 -> 3.13e+03 (delta -180) | 3.4e+03 -> 3.2e+03 (delta -195) | 84.4 -> 72.7 (delta -11.8) | 4.65 -> 0.592 (delta -4.06) | 0.588 -> 0.481 (delta -0.107) | 0.191 -> 0.0227 (delta -0.169) | 11.2 -> 6.87 (delta -4.32) | 0.148 -> 0.125 (delta -0.0229) | 0.705 -> 0.665 (delta -0.04) | 2.12 -> 2.09 (delta -0.0301) | 0.996 -> 0.999 (delta 0.00269) |
| 5e-4 + KL | 228-247 | 3.37e+03 -> 3.34e+03 (delta -33.2) | 3.43e+03 -> 3.4e+03 (delta -34.8) | 45.4 -> 44.3 (delta -1.11) | 15.8 -> 15.4 (delta -0.465) | 0.289 -> 0.323 (delta 0.0344) | 0.666 -> 0.599 (delta -0.0672) | 12.7 -> 12.9 (delta 0.13) | 0.0462 -> 0.054 (delta 0.0078) | 0.475 -> 0.475 (delta 0) | 1.85 -> 2.05 (delta 0.205) | 0.999 -> 0.999 (delta 0.000365) |
| 1e-3 + KL | 253-272 | 3.14e+03 -> 3e+03 (delta -135) | 3.21e+03 -> 3.08e+03 (delta -132) | 71.6 -> 75 (delta 3.4) | 4.54 -> 4.49 (delta -0.0507) | 0.462 -> 0.482 (delta 0.0207) | 0.189 -> 0.186 (delta -0.00325) | 8.89 -> 9.58 (delta 0.7) | 0.0979 -> 0.107 (delta 0.00952) | 0.56 -> 0.598 (delta 0.0383) | 1.94 -> 1.72 (delta -0.221) | 0.999 -> 0.998 (delta -0.00032) |

## Same-realization key events

A common-input event is selected from the ten lowest `difficulty_score` positions. A policy-specific event is selected by the largest gap between the worst standardized residual and the median of the other five, preferring epochs with at most two negative residuals. Each cell compares Return (R), residual (r), standardized residual (z), KL, clip fraction, CacheChange (C), actual UAV movement (U), and entropy (H) from the same epoch.

### 10 strongest common-input events

| epoch | realization_id | difficulty | negative/6 | 1e-4 | 5e-4 | 7e-4 | 1e-3 | 5e-4 + KL | 1e-3 + KL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 44 | episode_20044 | -1.32 | 4 | R=1.59e+03; r=191; z=0.413; KL=0.0131; clip=0.333; C=0.447; U=0.617; H=19.1 | R=471; r=-1.08e+03; z=-2.28; KL=0.00421; clip=0.367; C=0.498; U=0.697; H=19.6 | R=1.56e+03; r=-180; z=-0.36; KL=0.469; clip=0.867; C=0.545; U=0.562; H=18.2 | R=2.57e+03; r=12.8; z=0.0337; KL=0.0509; clip=0.5; C=0.529; U=0.403; H=17.5 | R=170; r=-922; z=-2.32; KL=0.111; clip=0.5; C=0.577; U=0.603; H=20.7 | R=1.82e+03; r=-716; z=-2.37; KL=0.0541; clip=0.633; C=0.438; U=0.705; H=17.6 |
| 189 | episode_20189 | -1.24 | 6 | R=2.63e+03; r=-407; z=-0.882; KL=0.0158; clip=0.333; C=0.34; U=0.4; H=12.3 | R=2.48e+03; r=-521; z=-1.1; KL=-0.0127; clip=0.267; C=0.464; U=0.547; H=15.8 | R=2.03e+03; r=-877; z=-1.76; KL=0.189; clip=0.633; C=0.624; U=0.395; H=16.3 | R=2.53e+03; r=-526; z=-1.39; KL=0.218; clip=0.6; C=0.507; U=0.045; H=6.25 | R=2.82e+03; r=-298; z=-0.749; KL=0.0813; clip=0.633; C=0.356; U=0.603; H=14.4 | R=2.6e+03; r=-416; z=-1.38; KL=0.153; clip=0.567; C=0.506; U=0.483; H=13.1 |
| 280 | episode_20280 | -1.12 | 6 | R=2.95e+03; r=-362; z=-0.783; KL=0.04; clip=0.167; C=0.293; U=0.392; H=9.26 | R=2.63e+03; r=-669; z=-1.42; KL=0.181; clip=0.633; C=0.468; U=0.337; H=12.2 | R=2.71e+03; r=-509; z=-1.02; KL=0.0504; clip=0.533; C=0.314; U=0.213; H=9.24 | R=2.7e+03; r=-465; z=-1.22; KL=0.051; clip=0.6; C=0.446; U=0.0217; H=6.5 | R=2.92e+03; r=-361; z=-0.908; KL=0.0388; clip=0.333; C=0.327; U=0.645; H=14.6 | R=2.59e+03; r=-463; z=-1.54; KL=-0.0115; clip=0.533; C=0.433; U=0.135; H=9.1 |
| 38 | episode_20038 | -1.03 | 4 | R=312; r=-756; z=-1.64; KL=0.0174; clip=0.333; C=0.48; U=0.527; H=19.1 | R=522; r=-803; z=-1.7; KL=0.00454; clip=0.4; C=0.519; U=0.6; H=20.1 | R=1.43e+03; r=9.09; z=0.0182; KL=0.265; clip=0.567; C=0.538; U=0.422; H=19.3 | R=2.82e+03; r=488; z=1.28; KL=0.113; clip=0.567; C=0.588; U=0.447; H=20.6 | R=517; r=-505; z=-1.27; KL=0.0448; clip=0.433; C=0.596; U=0.6; H=22.4 | R=2.3e+03; r=-240; z=-0.796; KL=0.0244; clip=0.4; C=0.418; U=0.738; H=17.9 |
| 186 | episode_20186 | -0.998 | 6 | R=2.73e+03; r=-322; z=-0.698; KL=-0.00874; clip=0.167; C=0.327; U=0.452; H=13.2 | R=2.71e+03; r=-292; z=-0.618; KL=0.0818; clip=0.567; C=0.427; U=0.608; H=14.7 | R=1.63e+03; r=-1.37e+03; z=-2.75; KL=0.0777; clip=0.7; C=0.576; U=0.492; H=16.7 | R=2.59e+03; r=-519; z=-1.37; KL=-0.0641; clip=0.567; C=0.525; U=0.0567; H=6.82 | R=2.82e+03; r=-282; z=-0.711; KL=0.0911; clip=0.667; C=0.379; U=0.613; H=13.7 | R=2.65e+03; r=-387; z=-1.28; KL=0.0793; clip=0.7; C=0.479; U=0.492; H=13.9 |
| 273 | episode_20273 | -0.969 | 6 | R=3.02e+03; r=-301; z=-0.651; KL=-0.0152; clip=0.333; C=0.346; U=0.465; H=12 | R=2.98e+03; r=-404; z=-0.855; KL=-0.0161; clip=0.3; C=0.41; U=0.368; H=11.7 | R=3.01e+03; r=-304; z=-0.61; KL=0.0769; clip=0.533; C=0.366; U=0.227; H=11 | R=2.77e+03; r=-411; z=-1.08; KL=0.209; clip=0.633; C=0.479; U=0.0333; H=6.76 | R=2.86e+03; r=-468; z=-1.18; KL=0.0521; clip=0.3; C=0.334; U=0.57; H=13.5 | R=2.69e+03; r=-407; z=-1.35; KL=0.0108; clip=0.5; C=0.417; U=0.205; H=9.22 |
| 276 | episode_20276 | -0.875 | 6 | R=3e+03; r=-308; z=-0.666; KL=-0.0304; clip=0.267; C=0.311; U=0.428; H=11.2 | R=3e+03; r=-352; z=-0.744; KL=0.00911; clip=0.467; C=0.432; U=0.343; H=11.9 | R=2.95e+03; r=-326; z=-0.654; KL=0.0261; clip=0.567; C=0.337; U=0.172; H=10.7 | R=2.58e+03; r=-568; z=-1.49; KL=0.173; clip=0.533; C=0.41; U=0.02; H=6.56 | R=2.88e+03; r=-400; z=-1.01; KL=0.0374; clip=0.5; C=0.28; U=0.613; H=12.3 | R=2.66e+03; r=-411; z=-1.36; KL=0.199; clip=0.7; C=0.455; U=0.153; H=9.37 |
| 188 | episode_20188 | -0.779 | 6 | R=2.78e+03; r=-282; z=-0.61; KL=0.0335; clip=0.233; C=0.346; U=0.428; H=12.8 | R=2.96e+03; r=-64.1; z=-0.136; KL=-0.0463; clip=0.467; C=0.461; U=0.567; H=16.1 | R=2.49e+03; r=-473; z=-0.948; KL=0.0228; clip=0.667; C=0.621; U=0.433; H=16.6 | R=2.65e+03; r=-427; z=-1.12; KL=-0.0315; clip=0.5; C=0.493; U=0.0517; H=5.49 | R=3.01e+03; r=-123; z=-0.309; KL=0.0775; clip=0.4; C=0.402; U=0.612; H=15 | R=2.73e+03; r=-315; z=-1.05; KL=-0.0078; clip=0.467; C=0.5; U=0.437; H=13.1 |
| 42 | episode_20042 | -0.777 | 5 | R=727; r=-552; z=-1.2; KL=-0.0309; clip=0.167; C=0.497; U=0.618; H=21.4 | R=1.45e+03; r=-65.9; z=-0.139; KL=0.0209; clip=0.4; C=0.487; U=0.668; H=19.2 | R=2.12e+03; r=512; z=1.03; KL=-0.0126; clip=0.6; C=0.609; U=0.453; H=21.5 | R=2.37e+03; r=-136; z=-0.358; KL=0.115; clip=0.7; C=0.54; U=0.477; H=18.1 | R=-173; r=-1.26e+03; z=-3.18; KL=-0.0323; clip=0.5; C=0.607; U=0.573; H=19.7 | R=1.98e+03; r=-597; z=-1.98; KL=0.106; clip=0.533; C=0.44; U=0.665; H=17.8 |
| 153 | episode_20153 | -0.77 | 4 | R=2.7e+03; r=-270; z=-0.584; KL=-0.00936; clip=0.4; C=0.343; U=0.488; H=13.3 | R=2.79e+03; r=155; z=0.327; KL=0.02; clip=0.667; C=0.403; U=0.787; H=16.6 | R=2.8e+03; r=163; z=0.327; KL=0.271; clip=0.633; C=0.461; U=0.59; H=16.7 | R=2.83e+03; r=-436; z=-1.15; KL=-0.0876; clip=0.6; C=0.619; U=0.12; H=11.6 | R=2.68e+03; r=-380; z=-0.957; KL=0.115; clip=0.467; C=0.44; U=0.643; H=17.3 | R=2.45e+03; r=-527; z=-1.75; KL=0.00605; clip=0.6; C=0.439; U=0.46; H=13.4 |

### 10 strongest policy-specific events

| epoch | realization_id | difficulty | negative/6 | 1e-4 | 5e-4 | 7e-4 | 1e-3 | 5e-4 + KL | 1e-3 + KL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | episode_20300 | 0.391 | 2 | R=3.51e+03; r=188; z=0.407; KL=0.0566; clip=0.333; C=0.294; U=0.303; H=9.74 | R=3.55e+03; r=271; z=0.573; KL=0.0227; clip=0.167; C=0.414; U=0.36; H=10.4 | R=-649; r=-3.71e+03; z=-7.43; KL=0.12; clip=0.567; C=0.353; U=0.347; H=11.5 | R=3.32e+03; r=191; z=0.504; KL=0.0758; clip=0.7; C=0.479; U=0.0267; H=6.88 | R=3.49e+03; r=149; z=0.376; KL=-0.00967; clip=0.4; C=0.322; U=0.53; H=11.7 | R=2.91e+03; r=-86.4; z=-0.287; KL=0.0034; clip=0.467; C=0.54; U=0.153; H=9.68 |
| 148 | episode_20148 | 0.0827 | 2 | R=3.02e+03; r=46.7; z=0.101; KL=-0.0201; clip=0.3; C=0.356; U=0.503; H=14.5 | R=-446; r=-3.18e+03; z=-6.72; KL=0.0494; clip=0.233; C=0.357; U=0.802; H=15.6 | R=3.24e+03; r=433; z=0.867; KL=0.137; clip=0.667; C=0.441; U=0.527; H=15.5 | R=3.31e+03; r=24.4; z=0.0643; KL=0.382; clip=0.933; C=0.553; U=0.228; H=12.8 | R=3.2e+03; r=208; z=0.523; KL=0.0573; clip=0.433; C=0.426; U=0.667; H=16.8 | R=3.06e+03; r=-19.2; z=-0.0637; KL=0.083; clip=0.367; C=0.434; U=0.443; H=11.3 |
| 140 | episode_20140 | 0.736 | 1 | R=3.16e+03; r=371; z=0.803; KL=-0.0333; clip=0.233; C=0.346; U=0.562; H=15.2 | R=3.23e+03; r=468; z=0.99; KL=0.0892; clip=0.433; C=0.388; U=0.79; H=15.9 | R=-162; r=-2.94e+03; z=-5.9; KL=0.0868; clip=0.433; C=0.47; U=0.48; H=15.7 | R=3.45e+03; r=343; z=0.904; KL=0.0472; clip=0.667; C=0.561; U=0.203; H=12.2 | R=3.14e+03; r=266; z=0.67; KL=0.0339; clip=0.333; C=0.473; U=0.595; H=17.9 | R=3.18e+03; r=103; z=0.341; KL=0.38; clip=0.867; C=0.479; U=0.515; H=14 |
| 208 | episode_20208 | 0.971 | 1 | R=1.17e+03; r=-1.66e+03; z=-3.6; KL=0.0136; clip=0.233; C=0.342; U=0.388; H=11.9 | R=3.42e+03; r=264; z=0.559; KL=0.149; clip=0.567; C=0.423; U=0.473; H=15.1 | R=3.43e+03; r=775; z=1.55; KL=0.157; clip=0.667; C=0.623; U=0.403; H=16.4 | R=3.39e+03; r=404; z=1.06; KL=0.101; clip=0.567; C=0.528; U=0.075; H=7.52 | R=3.58e+03; r=420; z=1.06; KL=0.0458; clip=0.367; C=0.331; U=0.69; H=13.9 | R=3.23e+03; r=267; z=0.884; KL=0.0986; clip=0.5; C=0.546; U=0.41; H=13.1 |
| 28 | episode_20028 | 0.904 | 2 | R=1.04e+03; r=620; z=1.34; KL=0.0144; clip=0.233; C=0.579; U=0.54; H=22.3 | R=1.06e+03; r=243; z=0.514; KL=0.105; clip=0.4; C=0.551; U=0.622; H=20.9 | R=383; r=-743; z=-1.49; KL=0.148; clip=0.533; C=0.541; U=0.353; H=20.6 | R=2.04e+03; r=491; z=1.29; KL=-0.00959; clip=0.667; C=0.608; U=0.498; H=20.6 | R=-135; r=-1.1e+03; z=-2.77; KL=0.0851; clip=0.433; C=0.629; U=0.645; H=21.7 | R=2.71e+03; r=448; z=1.49; KL=0.0459; clip=0.567; C=0.396; U=0.703; H=17.7 |
| 29 | episode_20029 | 0.957 | 1 | R=678; r=161; z=0.349; KL=0.0586; clip=0.433; C=0.542; U=0.492; H=20.8 | R=1.62e+03; r=720; z=1.52; KL=-0.0147; clip=0.667; C=0.572; U=0.602; H=20.7 | R=1.34e+03; r=195; z=0.39; KL=0.0166; clip=0.667; C=0.561; U=0.36; H=20.1 | R=2.47e+03; r=777; z=2.05; KL=-0.0394; clip=0.633; C=0.611; U=0.51; H=22 | R=191; r=-782; z=-1.97; KL=-0.0359; clip=0.633; C=0.617; U=0.603; H=22.5 | R=2.94e+03; r=590; z=1.96; KL=0.074; clip=0.6; C=0.408; U=0.722; H=17.7 |
| 67 | episode_20067 | 0.244 | 2 | R=418; r=-1.33e+03; z=-2.88; KL=0.0365; clip=0.167; C=0.467; U=0.623; H=20.6 | R=2.52e+03; r=165; z=0.349; KL=0.035; clip=0.5; C=0.421; U=0.685; H=19.5 | R=2.57e+03; r=69.6; z=0.139; KL=0.0783; clip=0.333; C=0.593; U=0.518; H=19.5 | R=2.94e+03; r=252; z=0.664; KL=0.0933; clip=0.6; C=0.473; U=0.418; H=15.7 | R=1.73e+03; r=-279; z=-0.702; KL=-0.0253; clip=0.567; C=0.499; U=0.788; H=19.5 | R=2.98e+03; r=283; z=0.94; KL=0.0636; clip=0.433; C=0.461; U=0.717; H=16.7 |
| 93 | episode_20093 | 1.15 | 1 | R=2.9e+03; r=828; z=1.79; KL=0.00909; clip=0.167; C=0.43; U=0.653; H=19.2 | R=3.03e+03; r=621; z=1.31; KL=-0.0692; clip=0.533; C=0.487; U=0.838; H=19.5 | R=2.74e+03; r=206; z=0.412; KL=0.158; clip=0.8; C=0.541; U=0.638; H=20 | R=3.49e+03; r=451; z=1.19; KL=0.187; clip=0.7; C=0.586; U=0.462; H=17.4 | R=1.85e+03; r=-634; z=-1.6; KL=-0.018; clip=0.533; C=0.517; U=0.727; H=19.3 | R=3.29e+03; r=337; z=1.12; KL=0.182; clip=0.767; C=0.512; U=0.803; H=19.1 |
| 111 | episode_20111 | 0.234 | 1 | R=2.53e+03; r=199; z=0.43; KL=-0.0172; clip=0.167; C=0.4; U=0.65; H=18.1 | R=1.31e+03; r=-1.18e+03; z=-2.49; KL=0.0241; clip=0.633; C=0.479; U=0.762; H=20.9 | R=3.04e+03; r=60.8; z=0.122; KL=0.107; clip=0.633; C=0.547; U=0.62; H=18.9 | R=3.28e+03; r=103; z=0.271; KL=0.132; clip=0.6; C=0.534; U=0.402; H=15 | R=2.72e+03; r=78.2; z=0.197; KL=0.0618; clip=0.5; C=0.526; U=0.787; H=19.6 | R=2.96e+03; r=257; z=0.852; KL=0.111; clip=0.467; C=0.537; U=0.658; H=17.9 |
| 57 | episode_20057 | 0.817 | 1 | R=713; r=-846; z=-1.83; KL=0.028; clip=0.233; C=0.506; U=0.675; H=22 | R=2.49e+03; r=680; z=1.44; KL=0.113; clip=0.5; C=0.46; U=0.722; H=20.7 | R=2.75e+03; r=335; z=0.672; KL=-0.0502; clip=0.333; C=0.521; U=0.73; H=20.3 | R=2.79e+03; r=279; z=0.735; KL=0.0547; clip=0.667; C=0.584; U=0.427; H=18.8 | R=2.09e+03; r=357; z=0.899; KL=0.0598; clip=0.5; C=0.566; U=0.72; H=22.2 | R=3.12e+03; r=707; z=2.35; KL=0.0399; clip=0.6; C=0.463; U=0.682; H=16.4 |

## Root-cause ranking

ROOT_CAUSE_RANKING =

1. Actor effective update intensity / policy drift
   - evidence: whole-run KL and clip fraction rise with actor LR; high-LR logs contain near-universal pre-clipping actor norms above 0.5; 1e-3 + KL invokes early stopping frequently.
   - counter-evidence: actor-gradient magnitude itself is not monotonic, and lagged update-to-return associations are weak/inconsistent.
   - confidence: medium-high

2. Shared input-realization variation
   - evidence: strict same-realization alignment; moderate positive residual correlations; 4/6-or-more simultaneous negative residuals on a substantial fraction of valid epochs.
   - counter-evidence: pairwise correlations are not uniformly high and policy-specific events remain.
   - confidence: medium

3. UAV MOVE-to-STAY behavior drift / entropy change
   - evidence: large actual movement declines in selected high-LR cases, with return/q declines; available feasibility rates do not indicate masking in those cases.
   - counter-evidence: policy/mask fields are missing for some runs, and low movement alone is not treated as collapse.
   - confidence: medium

Secondary: CacheChange behavior drifts, but cost directions are mixed; Critic instability and reward-accounting/system-cost dominance are not supported by the logged evidence.

Other candidate checks: PPO repeated-update effect is included in the KL early-stop and executed-update-epoch evidence but cannot be separated from actor update intensity in these six single-seed runs; entropy declines in some high-LR runs but is not independently identified as the root cause; no evidence supports Critic instability or a reward/system-cost accounting failure.

## Decision recommendations

A. KEEP_SYSTEM_MODEL_FIXED = YES
B. KEEP_CRITIC_LR_3E4 = YES
C. NEXT_CHANGE_PRIORITY = actor LR / effective update intensity; keep the system model, reward, critic LR, and semantics fixed
D. FIXED_LR_STILL_WORTH_SEARCHING = YES
E. RECOMMENDED_NEXT_FIXED_LR = 5e-4 with existing KL protection target_kl=0.10 (focused follow-up; no broad sweep)
F. A scheduling change is not required by this evidence yet; first isolate the lower-LR + KL setting in one focused follow-up.

## Reproducibility

All calculations use trailing MA20 only; residuals before epoch 20 are excluded. No centered/future window, test seed, new episode, checkpoint, or run file is used.
