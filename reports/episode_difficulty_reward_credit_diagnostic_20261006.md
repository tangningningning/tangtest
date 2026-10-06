# Episode Difficulty, Reward, and Macro Credit Diagnostic (5E-A)

Read-only descriptive analysis of existing 5D Joint/Freeze artifacts. Joint is the primary n=10 episode sample (20016–20025); Freeze Micro is paired auxiliary evidence only. No training, rollout, evaluation, backward call, or optimizer step was run.

## Data completeness and workload pairing

- Required field status: **FIELDS_COMPLETE**.
- Joint/Freeze workload pairing: **PASS** across scheduled task count, scheduled CPU cycles total, scheduled input total, traffic departure seed, and traffic random seed; each compared branch artifact also matches the paired report summary.
- Missing required fields: none.
- `scheduled_task_count` is the workload schedule size (6000 per realization); `task_count` is the number actually recorded by the system outcome and is reported separately.

## Table 1. Joint episode master table

| episode_seed | Return | slot_reward_mean | slot_reward_std | Macro_frame_count | scheduled_task_count | task_count | scheduled_cpu_cycles_total | scheduled_cpu_cycles_mean | scheduled_cpu_cycles_p95 | scheduled_input_size_total | scheduled_input_size_mean | scheduled_input_size_p95 | difficulty_score | mean_delay_per_task_s | mean_energy_per_task_j | task_utility | total_cache_cost | total_movement_cost | Macro_raw_advantage_mean | Macro_raw_advantage_std | Macro_raw_advantage_rms | Macro_raw_advantage_positive_fraction | Macro_raw_advantage_negative_fraction | Macro_candidate_KL | Macro_duration_KL | Macro_actor_relative_parameter_delta | Macro_critic_EV | Macro_critic_RMSE | mean_P_KEEP | KEEP_rate | UPDATE_rate | Duration_1_rate | Duration_3_rate | Duration_5_rate | UAV0_MOVE_rate | UAV1_MOVE_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20016 | -550.2 | -1.834 | 17.22 | 92 | 6000 | 4236 | 6.329e+13 | 1.055e+10 | 2.721e+10 | 9.404e+10 | 1.567e+07 | 3.693e+07 | -0.04633 | 31.33 | 3.94 | -543.8 | 5.19 | 1.149 | -160.8 | 239.7 | 288.7 | 0.3152 | 0.6848 | 0.000171 | 5.526e-07 | 0.007887 | 0.9946 | 42.8 | 0.5459 | 0.6087 | 0.3913 | 0.2826 | 0.3043 | 0.413 | 0.92 | 0.6733 |
| 20017 | 904.9 | 3.016 | 11.76 | 100 | 6000 | 4193 | 6.9e+13 | 1.15e+10 | 2.941e+10 | 9.227e+10 | 1.538e+07 | 3.637e+07 | 0.2819 | 23.88 | 2.409 | 912.8 | 6.745 | 1.17 | -78.2 | 165.4 | 183 | 0.23 | 0.77 | 0.0006085 | 4.596e-06 | 0.01096 | 0.9946 | 55.66 | 0.5373 | 0.57 | 0.43 | 0.3 | 0.39 | 0.31 | 0.8767 | 0.7567 |
| 20018 | 668.9 | 2.23 | 9.227 | 92 | 6000 | 4251 | 6.305e+13 | 1.051e+10 | 2.748e+10 | 9.05e+10 | 1.508e+07 | 3.784e+07 | -0.2715 | 25.88 | 2.558 | 676.1 | 6.021 | 1.193 | -65.2 | 153.1 | 166.4 | 0.4239 | 0.5761 | 0.0009872 | 1.218e-05 | 0.01379 | 0.9956 | 47.98 | 0.5722 | 0.5652 | 0.4348 | 0.25 | 0.3696 | 0.3804 | 0.9067 | 0.7633 |
| 20019 | 590.1 | 1.967 | 23.03 | 112 | 6000 | 4217 | 7.577e+13 | 1.263e+10 | 3.288e+10 | 1.045e+11 | 1.741e+07 | 4.291e+07 | 1.51 | 24.57 | 2.889 | 596.8 | 5.57 | 1.09 | -38.17 | 203.3 | 206.9 | 0.5714 | 0.4286 | 0.0004312 | 2.749e-05 | 0.01294 | 0.9819 | 76.79 | 0.6262 | 0.6429 | 0.3571 | 0.4375 | 0.2679 | 0.2946 | 0.8733 | 0.62 |
| 20020 | 2160 | 7.2 | 4.84 | 100 | 6000 | 4245 | 6.867e+13 | 1.145e+10 | 3.02e+10 | 8.964e+10 | 1.494e+07 | 3.533e+07 | 0.1028 | 16.91 | 1.213 | 2166 | 4.929 | 1.113 | 44.58 | 68.09 | 81.38 | 0.74 | 0.26 | 0.0005239 | 1.525e-05 | 0.01077 | 0.993 | 38.66 | 0.6422 | 0.56 | 0.44 | 0.34 | 0.31 | 0.35 | 0.85 | 0.71 |
| 20021 | 2750 | 9.168 | 2.903 | 108 | 6000 | 4264 | 5.605e+13 | 9.341e+09 | 2.677e+10 | 7.98e+10 | 1.33e+07 | 3.355e+07 | -1.429 | 12.02 | 0.901 | 2758 | 6.673 | 1.158 | 77.65 | 90.07 | 118.9 | 0.8056 | 0.1944 | 0.0001637 | 1.63e-06 | 0.009217 | 0.9928 | 37.24 | 0.656 | 0.6111 | 0.3889 | 0.3519 | 0.4074 | 0.2407 | 0.9167 | 0.7067 |
| 20022 | 2324 | 7.748 | 3.759 | 101 | 6000 | 4200 | 6.562e+13 | 1.094e+10 | 2.781e+10 | 9.257e+10 | 1.543e+07 | 3.726e+07 | 0.04444 | 16.34 | 0.9577 | 2330 | 4.508 | 1.167 | 56.85 | 68.59 | 89.08 | 0.7624 | 0.2376 | 5.159e-05 | 3.433e-07 | 0.008856 | 0.9986 | 21.33 | 0.651 | 0.6436 | 0.3564 | 0.3465 | 0.3168 | 0.3366 | 0.8633 | 0.7533 |
| 20023 | 1466 | 4.886 | 16.57 | 108 | 6000 | 4238 | 7.559e+13 | 1.26e+10 | 3.265e+10 | 1.089e+11 | 1.816e+07 | 4.245e+07 | 1.757 | 19.74 | 2.036 | 1473 | 5.515 | 1.186 | -11.9 | 177.3 | 177.7 | 0.6111 | 0.3889 | 8.215e-05 | 1.987e-06 | 0.008323 | 0.9906 | 41.43 | 0.6683 | 0.6111 | 0.3889 | 0.3796 | 0.3426 | 0.2778 | 0.9033 | 0.7467 |
| 20024 | 1386 | 4.618 | 11.25 | 119 | 6000 | 4215 | 5.989e+13 | 9.981e+09 | 2.813e+10 | 8.719e+10 | 1.453e+07 | 3.551e+07 | -0.7047 | 20.94 | 1.985 | 1393 | 6.244 | 1.155 | -8.509 | 141.6 | 141.8 | 0.4958 | 0.5042 | 0.0001535 | 1.001e-06 | 0.009035 | 0.9934 | 33.81 | 0.6067 | 0.6471 | 0.3529 | 0.4538 | 0.3193 | 0.2269 | 0.86 | 0.7467 |
| 20025 | 2030 | 6.768 | 7.718 | 107 | 6000 | 4203 | 5.676e+13 | 9.459e+09 | 2.975e+10 | 8.201e+10 | 1.367e+07 | 3.775e+07 | -1.245 | 16.49 | 1.474 | 2036 | 4.332 | 1.155 | 33.87 | 115.8 | 120.6 | 0.6542 | 0.3458 | 5.695e-05 | 1.255e-07 | 0.007762 | 0.9951 | 40.68 | 0.6583 | 0.7196 | 0.2804 | 0.3738 | 0.3458 | 0.2804 | 0.8433 | 0.77 |

## Table 2. Environment predictors of Return

Rows are ranked by absolute Spearman correlation; scheduled task count is constant across episodes, so its correlation is undefined. Correlations are descriptive only.

| Rank by |Spearman| | Environment predictor | Pearson r | Spearman ρ | n | Interpretation |
| --- | --- | --- | --- | --- | --- |
| 1 | scheduled_input_size_total | -0.4472 | -0.5273 | 10 | descriptive only |
| 2 | scheduled_input_size_mean | -0.4472 | -0.5273 | 10 | descriptive only |
| 3 | scheduled_input_size_p95 | -0.3616 | -0.4788 | 10 | descriptive only |
| 4 | scheduled_cpu_cycles_total | -0.3044 | -0.3697 | 10 | descriptive only |
| 5 | scheduled_cpu_cycles_mean | -0.3044 | -0.3697 | 10 | descriptive only |
| 6 | scheduled_cpu_cycles_p95 | -0.05945 | -0.1152 | 10 | descriptive only |
| 7 | task_count | 0.06565 | 0.1152 | 10 | descriptive only |
| 8 | scheduled_task_count | UNDEFINED_CONSTANT_SERIES | UNDEFINED_CONSTANT_SERIES | 10 | descriptive only |

## Table 3. Difficulty score vs system outcomes

Difficulty is the equally weighted mean of population z-scores for scheduled CPU-cycle total and scheduled input-size total, computed only across Joint n=10. Easier/middle/harder tertile summaries: {"easiest_3": {"Macro_raw_advantage_std": 115.81254452951116, "Return": 2055.3853466567634, "episode_seeds": [20021, 20025, 20024], "mean_delay_per_task_s": 16.486551659580588, "mean_energy_per_task_j": 1.4534431287271516, "n": 3, "task_utility": 2062.291131138172}, "hardest_3": {"Macro_raw_advantage_std": 182.00025717074357, "Return": 986.9506631140138, "episode_seeds": [20017, 20019, 20023], "mean_delay_per_task_s": 22.726431897194743, "mean_energy_per_task_j": 2.4445489407433985, "n": 3, "task_utility": 994.0425603070968}, "middle_4": {"Macro_raw_advantage_std": 132.38989167705824, "Return": 1150.7574370455961, "episode_seeds": [20018, 20016, 20022, 20020], "mean_delay_per_task_s": 22.617056563357348, "mean_energy_per_task_j": 2.167218262316603, "n": 4, "task_utility": 1157.0749745495914}}.

| Difficulty relation | Pearson r | Spearman ρ | n | Interpretation |
| --- | --- | --- | --- | --- |
| mean_energy_per_task_j | 0.381 | 0.4061 | 10 | descriptive only |
| mean_delay_per_task_s | 0.3777 | 0.3212 | 10 | descriptive only |
| task_utility | -0.3831 | -0.3212 | 10 | descriptive only |
| total_cache_cost | -0.06259 | -0.0303 | 10 | descriptive only |
| total_movement_cost | -0.1957 | 0.006061 | 10 | descriptive only |

## Table 4. System outcomes vs Return

Ranked by absolute Spearman correlation. Task utility is an already recorded system outcome, not independently reconstructed here.

| Rank by |Spearman| | System outcome vs Return | Pearson r | Spearman ρ | n | Interpretation |
| --- | --- | --- | --- | --- | --- |
| 1 | task_utility | 1 | 1 | 10 | descriptive only |
| 2 | mean_energy_per_task_j | -0.9924 | -0.9879 | 10 | descriptive only |
| 3 | mean_delay_per_task_s | -0.9918 | -0.9758 | 10 | descriptive only |
| 4 | total_cache_cost | -0.1241 | -0.2 | 10 | descriptive only |
| 5 | total_movement_cost | 0.04764 | 0.1273 | 10 | descriptive only |

## Table 5. Difficulty, Return, and system predictors vs Macro raw Advantage

Ranked by absolute Spearman correlation with raw Advantage std. The table includes correlations against raw mean, std, and RMS; all rows are descriptive only.

| Rank by |ρ vs Adv std| | Predictor | r vs Adv mean | ρ vs Adv mean | r vs Adv std | ρ vs Adv std | r vs Adv RMS | ρ vs Adv RMS | n | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mean_energy_per_task_j | -0.9658 | -0.9515 | 0.9487 | 0.903 | 0.9561 | 0.9152 | 10 | descriptive only |
| 2 | Return | 0.9775 | 0.9394 | -0.915 | -0.8667 | -0.9227 | -0.8909 | 10 | descriptive only |
| 3 | task_utility | 0.9775 | 0.9394 | -0.9149 | -0.8667 | -0.9226 | -0.8909 | 10 | descriptive only |
| 4 | mean_delay_per_task_s | -0.9744 | -0.9394 | 0.864 | 0.7939 | 0.8732 | 0.8182 | 10 | descriptive only |
| 5 | scheduled_input_size_total | -0.3765 | -0.5394 | 0.5578 | 0.6606 | 0.4551 | 0.6121 | 10 | descriptive only |
| 6 | scheduled_input_size_mean | -0.3765 | -0.5394 | 0.5578 | 0.6606 | 0.4551 | 0.6121 | 10 | descriptive only |
| 7 | scheduled_input_size_p95 | -0.2481 | -0.4061 | 0.5252 | 0.5515 | 0.376 | 0.4909 | 10 | descriptive only |
| 8 | scheduled_cpu_cycles_total | -0.2451 | -0.3939 | 0.3629 | 0.4182 | 0.2611 | 0.4061 | 10 | descriptive only |
| 9 | scheduled_cpu_cycles_mean | -0.2451 | -0.3939 | 0.3629 | 0.4182 | 0.2611 | 0.4061 | 10 | descriptive only |
| 10 | difficulty_score | -0.3168 | -0.3818 | 0.4693 | 0.4061 | 0.365 | 0.3818 | 10 | descriptive only |
| 11 | total_cache_cost | -0.1972 | -0.2606 | 0.2142 | 0.2606 | 0.2054 | 0.3212 | 10 | descriptive only |
| 12 | scheduled_cpu_cycles_p95 | 0.03012 | -0.06667 | 0.2486 | 0.1394 | 0.07383 | 0.103 | 10 | descriptive only |
| 13 | task_count | 0.07661 | 0.1758 | -0.04112 | -0.07879 | 0.02848 | -0.1515 | 10 | descriptive only |
| 14 | total_movement_cost | -0.08656 | -0.09091 | -0.03733 | -0.006061 | -0.01957 | -0.01818 | 10 | descriptive only |
| 15 | scheduled_task_count | UNDEFINED_CONSTANT_SERIES | UNDEFINED_CONSTANT_SERIES | UNDEFINED_CONSTANT_SERIES | UNDEFINED_CONSTANT_SERIES | UNDEFINED_CONSTANT_SERIES | UNDEFINED_CONSTANT_SERIES | 10 | descriptive only |

### Difficulty group means

| Group | Seeds | n | Return mean | Delay mean | Energy mean | Task utility mean | Adv std mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| easiest_3 | 20021, 20025, 20024 | 3 | 2055 | 16.49 | 1.453 | 2062 | 115.8 |
| middle_4 | 20018, 20016, 20022, 20020 | 4 | 1151 | 22.62 | 2.167 | 1157 | 132.4 |
| hardest_3 | 20017, 20019, 20023 | 3 | 987 | 22.73 | 2.445 | 994 | 182 |

## Table 6. Episode case studies 20021–20023

| Seed | Return | CPU total | Input total | Delay | Energy | Utility | Cache cost | Movement cost | Adv mean | Adv std | Adv RMS | Candidate KL | Duration KL | Actor rel Δ | Critic EV | Critic RMSE | P_KEEP | KEEP rate | UPDATE rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20021 | 2750 | 5.605e+13 | 7.98e+10 | 12.02 | 0.901 | 2758 | 6.673 | 1.158 | 77.65 | 90.07 | 118.9 | 0.0001637 | 1.63e-06 | 0.009217 | 0.9928 | 37.24 | 0.656 | 0.6111 | 0.3889 |
| 20022 | 2324 | 6.562e+13 | 9.257e+10 | 16.34 | 0.9577 | 2330 | 4.508 | 1.167 | 56.85 | 68.59 | 89.08 | 5.159e-05 | 3.433e-07 | 0.008856 | 0.9986 | 21.33 | 0.651 | 0.6436 | 0.3564 |
| 20023 | 1466 | 7.559e+13 | 1.089e+11 | 19.74 | 2.036 | 1473 | 5.515 | 1.186 | -11.9 | 177.3 | 177.7 | 8.215e-05 | 1.987e-06 | 0.008323 | 0.9906 | 41.43 | 0.6683 | 0.6111 | 0.3889 |

- **20021**: Return 2750; difficulty -1.429; CPU/input totals 5.605e+13/7.98e+10; delay/utility 12.02/2758; raw Advantage std 90.07. KL=0.0001637, actor relative delta=0.009217, Critic EV/RMSE=0.9928/37.24.
- **20022**: Return 2324; difficulty 0.04444; CPU/input totals 6.562e+13/9.257e+10; delay/utility 16.34/2330; raw Advantage std 68.59. KL=5.159e-05, actor relative delta=0.008856, Critic EV/RMSE=0.9986/21.33.
- **20023**: Return 1466; difficulty 1.757; CPU/input totals 7.559e+13/1.089e+11; delay/utility 19.74/1473; raw Advantage std 177.3. KL=8.215e-05, actor relative delta=0.008323, Critic EV/RMSE=0.9906/41.43.

## PPO update anomaly scan

- Sequential Return drops recorded across different episode/workload seeds: 5; largest is 20022→20023 (-858.5) if present, otherwise none.
- Joint candidate KL max 0.0009872, actor relative parameter delta max 0.01379, Critic EV min 0.9819, Critic RMSE max 76.79 (seed 20019).
- Drop-by-drop: 20017→20018 ΔReturn -236, candidate/duration KL 0.0009872/1.218e-05, actor rel Δ 0.01379, Critic EV/RMSE 0.9956/47.98; 20018→20019 ΔReturn -78.75, candidate/duration KL 0.0004312/2.749e-05, actor rel Δ 0.01294, Critic EV/RMSE 0.9819/76.79; 20021→20022 ΔReturn -426, candidate/duration KL 5.159e-05/3.433e-07, actor rel Δ 0.008856, Critic EV/RMSE 0.9986/21.33; 20022→20023 ΔReturn -858.5, candidate/duration KL 8.215e-05/1.987e-06, actor rel Δ 0.008323, Critic EV/RMSE 0.9906/41.43; 20023→20024 ΔReturn -80.31, candidate/duration KL 0.0001535/1.001e-06, actor rel Δ 0.009035, Critic EV/RMSE 0.9934/33.81.
- Assessment: the 20017→20018 Return drop coincides with the sample maximum KL and actor relative delta (0.000987 and 0.0138), while Critic EV is 0.9956. The 20018→20019 transition has the minimum EV and maximum RMSE (0.9819; 76.79, or 0.1348 of target SD), without an EV collapse. The largest Return drop at 20022→20023 has no KL/delta spike. No clear PPO update anomaly in existing monitors: candidate/duration KL and actor relative deltas are small; Critic EV remains above 0.98. The largest observed Return drop is paired with a harder workload realization and no corresponding KL/delta spike or EV collapse.

## Table 7. Counterexample scan

High/low means strictly above/below the median of the Joint n=10 sample; difficulty, Return, and raw Advantage std use their own medians. Unique episodes across the four categories: **6**.

| Pattern | Episode seeds | Count | Observed interpretation |
| --- | --- | --- | --- |
| High workload + high Return | 20020, 20022, 20023 | 3 | Above both Joint medians; deviation from a simple harder→lower Return pattern. |
| Low workload + low Return | 20016, 20018, 20024 | 3 | Below both Joint medians; deviation from a simple easier→higher Return pattern. |
| High Adv std + high Return | 20023 | 1 | Shows high within-episode raw advantage spread can coexist with higher Return. |
| Low Adv std + low Return | 20024 | 1 | Shows low within-episode raw advantage spread can coexist with lower Return. |

Notable episodes: 20016 (Lowest Return with the largest Macro raw-advantage std; difficulty score alone does not explain this outcome.); 20023 (Hard workload realization with Return above the sample median; workload and Return are not monotone.); 20024 (Easy-side workload realization with Return below the sample median; workload and Return are not monotone.).

## Table 8. Same-workload Joint/Freeze policy-path spread

These are paired by identical environment workload realization. Since policy parameters changed along both branches, the deltas describe policy-path spread and do not isolate a pure Micro causal effect; the 5D primary conclusion is not re-judged here.

| Seed | Workload paired | Δ Return Freeze−Joint | |Δ Return| | Δ Delay | Δ Energy | Δ Utility | Δ Cache cost | Δ Movement cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20016 | YES | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20017 | YES | 791.8 | 791.8 | -5.339 | -0.6465 | 792.5 | 0.6811 | 0.01331 |
| 20018 | YES | 1391 | 1391 | -8.968 | -1.165 | 1393 | 1.971 | -0.0979 |
| 20019 | YES | 285.1 | 285.1 | 0.5388 | -0.6006 | 286.2 | 1.125 | 0.0341 |
| 20020 | YES | -480 | 480 | 3.559 | 0.3308 | -479.3 | 0.6509 | 0.05115 |
| 20021 | YES | -241.9 | 241.9 | 1.241 | 0.2485 | -242 | -0.09326 | 0.0253 |
| 20022 | YES | -1857 | 1857 | 10.62 | 1.79 | -1855 | 1.678 | -0.05192 |
| 20023 | YES | -996.5 | 996.5 | 6.291 | 0.8572 | -996.7 | -0.07805 | -0.09823 |
| 20024 | YES | 430.7 | 430.7 | -2.381 | -0.424 | 430 | -0.6415 | -0.03663 |
| 20025 | YES | 803.3 | 803.3 | -4.414 | -0.8034 | 804.4 | 1.083 | -0.00176 |

Joint Return: mean=1373, population std=942.8, range=3300. Absolute paired Return difference Freeze−Joint: mean=727.7, median=635.9, population std=539.3, max=1857.

## Table 9. Diagnosis and convergence guidance

| Question | Evidence / result |
| --- | --- |
| Final diagnosis | MIXED_ENVIRONMENT_AND_CREDIT_VARIANCE |
| Environment difficulty vs Return | Composite Pearson r=-0.383; Spearman ρ=-0.3212; input total vs Return Spearman ρ=-0.5273; n=10 (descriptive only). |
| Difficulty and Macro credit variance | Difficulty vs delay/energy/raw Advantage std Spearman ρ=0.3212/0.4061/0.4061; Return vs raw Advantage std ρ=-0.8667. Both signals are descriptive at n=10. |
| Recorded reward/system relationship | Return vs task utility Pearson/Spearman=1/1; Return vs delay/energy Spearman=-0.9758/-0.9879. Artifacts show close association, not independent formula verification. |
| PPO update anomaly | No clear PPO update anomaly in existing monitors: candidate/duration KL and actor relative deltas are small; Critic EV remains above 0.98. The largest observed Return drop is paired with a harder workload realization and no corresponding KL/delta spike or EV collapse. |
| Macro raw Advantage std vs Return | Pearson r=-0.915; Spearman ρ=-0.8667; association only. |
| Raw training Return as standalone convergence metric | NO; retain as outcome, use fixed matched-seed validation plus rolling mean auxiliary. |
| Fixed Environment Probe | Not started; next action is to wait for explicit user direction. |
| Counterexample episode count | 6 |
| Joint Return mean / std / range | 1373 / 942.8 / 3300 |
| Mean absolute paired Return difference | 727.7 |

Keep raw Return as the training objective/observed outcome; treat its rolling mean as auxiliary. For convergence comparisons, use a fixed validation seed set and matched-seed evaluation. Label stochastic-policy sampling separately from deterministic-policy evaluation.

## Execution metadata

| Field | Value |
|---|---|
| NEW_TRAINING_STARTED | NO |
| NEW_ROLLOUT_COLLECTED | NO |
| OPTIMIZER_STEP | 0 |
| BACKWARD_CALL | 0 |
| ALGORITHM_CHANGED | NO |
| HYPERPARAMETER_CHANGED | NO |
| REWARD_CHANGED | NO |
| ENVIRONMENT_CHANGED | NO |
| CHECKPOINT_CHANGED | NO |

## Method and limitations

- Uses branch JSON episode metrics and the existing paired 5D report JSON only. No episode realization files, environment code, policy, or checkpoint were loaded.
- Pearson and tie-aware Spearman are descriptive n=10 coefficients. No significance test, causal claim, reward formula recomputation, or validation evaluation was performed.
- Reward/system fields are taken from the recorded artifacts; no absent values are inferred. If required fields are missing, they are listed above and no rollout is generated.
- Fixed-seed validation/probe was not run and must not start automatically; wait for the user.
