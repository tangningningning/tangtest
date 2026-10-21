# PER_SLOT_ANALYTIC Reward normalization recalibration

**Decision: PASS.** The new references come from Training-only analytic task samples, using a deterministic allocation that does not use `T_ref` or `E_ref` and does not invoke Matching/KKT.

## Calibration contract

- Data source: `environment_data/proposed_joint_training/` only. Sampled 12 realizations: 20001, 20028, 20055, 20082, 20110, 20137, 20164, 20191, 20218, 20245, 20272, 20300. Sampled slots 0, 33, 66, 99, 132, 165, 198, 231, 264, 299 in each realization. Those are 120 real slots spanning the beginning, middle, and end of each 300-slot episode.
- Sample size: 1,642 arrived tasks; no Validation/Test samples. Episode-specific SV vehicle IDs vary, so the same feasible initial Cache matrix was held fixed by the stable five-SV/two-RSU/two-UAV row order. Configured UAV positions were fixed at UAV_0=(3364.03, 1119.88) m and UAV_1=(5179.65, 1710.41) m.
- Deterministic reference allocation: the Cache matrix is the existing seed-4401 uniform feasible initialization (it does not read popularity, prediction, or request data). For each task, consider only cached Edge nodes with a physically feasible current link; choose the least-loaded eligible Edge node, breaking ties by node order. Use Cloud only when no Edge node is eligible. Each used Edge node divides its CPU and bandwidth equally among its assigned tasks. Cloud divides its finite 40 GHz CPU pool equally among Cloud tasks and uses the configured 20 Mbps upload rate. The Cloud factor 13 is applied to communication delay only, not transmit energy.
- Workload calculation called `build_feasible_targets` only for eligibility, then the existing analytic Edge/Cloud execution equations. It did not call `optimize_task_matching`, any KKT allocator, or `task_utility` while selecting assignments. `T_ref` and `E_ref` were applied only after delay and energy had been calculated. No data files were written or changed.

## Results

P95 uses NumPy's linear percentile interpolation over task-level samples.

| Metric | All tasks (n=1,642) | Edge (n=1,196) | Cloud (n=446) |
|---|---:|---:|---:|
| Complete delay P50 / P90 / P95 (s) | 4.5123 / 20.9075 / **30.4737** | 7.1433 / 25.0329 / 33.4614 | 3.5063 / 4.4309 / 4.7015 |
| Vehicle upload energy P50 / P90 / P95 (J) | 0.14091 / 3.12827 / **5.15746** | 0.35842 / 3.93992 / 6.11595 | 0.04768 / 0.06161 / 0.06575 |
| Mean complete delay / energy | 8.7014 s / 0.95314 J | 10.6161 s / 1.29046 J | 3.5668 s / 0.04858 J |

The Edge delay comprises upload and allocated CPU time: upload P50/P90/P95 was 1.7963/19.7464/30.6524 s and compute was 2.8817/8.9944/12.4593 s. Cloud communication delay (including factor 13) was 3.1064/4.0143/4.2839 s, and Cloud compute delay was 0.3850/0.6238/0.6894 s at P50/P90/P95. Cloud energy remains based on the unscaled 20 Mbps transmit time.

The resulting references are task-level P95 complete delay and vehicle upload energy:

```ini
T_ref = 30.473728355325758 s
E_ref = 5.1574642506299275 J
```

For these references, the mean weighted delay penalty `omega_T*T_i/T_ref` is 0.17132 per task (P50/P90/P95: 0.08884/0.41165/0.60000). The mean weighted energy penalty `omega_E*E_i/E_ref` is 0.07392 (P50/P90/P95: 0.01093/0.24262/0.40000). The fixed per-task utility term is 1; it is separate from these penalties. Thus the weights do not make the observed penalty contributions 60% and 40% of total utility. The two P95 weighted values follow from independently setting each reference to its corresponding unweighted P95, not from equalizing penalties.

The old references were 30.6289851662 s and 3.06289851662 J. The historical delay scale came from a TV Local CPU-cycle P95 converted at 1 GHz; the historical energy scale used that workload with the TV Local `kappa=1e-28` energy model. Neither measured the current complete Edge/Cloud response delay or vehicle uplink energy. The new delay P95 is 0.51% below the old delay reference, while the energy P95 is 68.39% above the old energy reference. The old energy scale would assign a weighted penalty of about 0.674 at the sampled energy P95, versus 0.400 under the new scale. Both old values therefore had a source-model mismatch; recalibrating them together makes the scales correspond to the current analytic task model.

The reference policy assigned 72.84% of tasks to Edge and 27.16% to Cloud. The P50/P95 of the largest single Edge node's share of all tasks in a sampled slot was 20.0%/42.9% (about 58.9% of Edge tasks at P95). This reflects the fixed cached-service set and current physical link eligibility. It is a visible concentration tail; the allocation rule was not changed to tune the references. It does not establish that actual dynamic Matching/KKT competition among node types is reasonable.

Mean sampled input was 722.70 KB/task and mean CPU work was 4.044 Gcycle/task. Aggregate CPU work averaged 55.34 Gcycle per sampled slot (P50 55.40, P95 72.25 Gcycle), 35.41% below the previously cited 85.68 Gcycle/slot baseline. This is a sampled static workload comparison, not evidence about real node competition.

## Formal updates and checks

- Updated both formal TOML copies to the same new `T_ref`/`E_ref` and calibration note: `configs/proposed/system_config.toml` and `configs/proposed/sparse_coordinated_cache_v1.toml`.
- Added `_training_p95_reward_norm_v1` to the legacy, Multi-Node, and Sparse policy schema IDs in `src/common/project_config.py` and `src/ppo_simple/policy.py`; the formal TOML schemas match. The tensor architecture versions are unchanged. Existing checkpoints with the old schema are rejected by the current strict schema checks rather than silently treated as this Reward contract. Existing historical results were left untouched.
- Both formal configs loaded successfully with positive finite references, ten services, and catalog mean inputs within 500–1,000 KB. `task_allocation.py` reads the references in its Matching candidate utility and CPU/bandwidth KKT objectives. Reward and Exact-Q both route through `slot_environment.execute_slot` with the same `EnvironmentContext.cache_cfg`; no Exact-Q sweep was run.
- One real Training slot (episode 20001, slot 0; 12 arrivals) ran through configured execution with Matching/KKT. It was feasible; Q=11.2899500114, cache cost=0, UAV cost=0, and G=11.2899500114. Independently recomputing Q from the resulting per-task delay/energy and the loaded references matched exactly at the reported precision. KKT diagnostics were finite and converged.
- `T_ref`/`E_ref` also enter Matching and KKT coefficients, so this is a new experiment objective contract. Historical checkpoints/results must remain separated from runs trained under the new schema.

No source data, environment pool, Matching/KKT formula, Reward weight, or PPO training run was changed or executed. `conda run -n adaptive_cache_mec python` was unavailable in this shell; checks ran with the verified environment interpreter `C:\\software\\Anaconda_envs\\envs\\adaptive_cache_mec\\python.exe` using `-B`.
