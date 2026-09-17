# V10-D 6e-4 formal training

`STAGE = V10-D_6E4_FORMAL_TRAINING`

## Completion and immutable protocol

- Status: **completed**; completed epochs: **300/300**.
- Config: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\configs\proposed\system_config_ppo_v10d_decoupled_encoder_branch_kl_6e4.toml`; config SHA256: `1515f8cf62df1980564c1fb1a4ba95da28a6157712c679ed2895143c797c6d44`.
- Policy schema: `adaptive_joint_cache_replica_aware_uav_operation_v10d_decoupled_encoder_branch_kl`; checkpoint version: V10-D branch-specific schema.
- Seed: `4401`; protocol: `full_episode_randomized`; rollout: `300` slots.
- Frozen episode pool: `runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911`, records `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v6_popart_actor1e4_lazy_episode_pool_20260911\episode_20001` through `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v6_popart_actor1e4_lazy_episode_pool_20260911\episode_20300`; pool indices are exactly 0..299 and no missing realization was generated.
- Hyperparameters retained: actor LR `0.0006`, critic LR `0.0003`, target KL `0.1`, clip `0.2`, gamma `1.0`, GAE lambda `0.95`, entropy coefficient `0.0`, update epochs `4`, minibatches `10`, minibatch size `30`, PopArt `popart_normalized`.
- No V10-D 5e-4 run, final-test seeds `40001..40020`, episode generation, or extra pilot/sweep/counterfactual evaluation was executed in this stage.

## Return and moving-window summary

| Metric | V10-D 6e-4 | V10-C 6e-4 reference | Delta (D-C) |
|---|---:|---:|---:|
| Best Return | 3555.935397 | 3591.855827 | -35.920430 |
| Final Return | 3428.664782 | 2996.299011 | 432.365771 |
| Final20 Return | 3281.020657 | 3147.194507 | 133.826150 |
| Final20 CacheChange | 0.540389 | 0.600444 | -0.060056 |
| Final20 UAVMove | 0.993167 | 0.967750 | 0.025417 |
| Final20 CriticEV | 0.999100 | 0.997723 | 0.001377 |
| MA20 max | 3291.858172 | 3335.168988 | -43.310816 |
| MA50 max | 3248.685940 | 3263.255387 | -14.569447 |

- Best epoch/Return: **245 / 3555.935397**.
- Final epoch/Return: **300 / 3428.664782**.
- Final20: Return `3281.020657`, CacheChange `54.04%`, UAVMove `99.32%`, Critic EV `0.999100`.
- MA20 maximum: `3291.858172` at epoch `247`; MA50 maximum: `3248.685940` at epoch `246`.
- Primary global-Return comparison: V10-D does **not** exceed the V10-C reference on peak/best Return (delta shown above), while its final Return and Final20 Return are higher. This is a data-driven comparison with no pre-assumed movement direction.

## Phase means

| Phase | Return | CacheChange | UAVMove | Critic EV |
|---|---:|---:|---:|---:|
| 1-60 | 1212.714324 | 63.30% | 58.35% | 0.326939 |
| 61-120 | 2603.045358 | 54.10% | 62.67% | 0.918245 |
| 121-180 | 3062.627886 | 53.04% | 91.26% | 0.992486 |
| 181-240 | 3212.541284 | 53.60% | 96.27% | 0.997798 |
| 241-300 | 3195.846096 | 58.24% | 98.57% | 0.998689 |
| 281-300 | 3281.020657 | 54.04% | 99.32% | 0.999100 |

## Branch-specific KL gate evidence

- Gate mode values: `['branch_specific']`; joint KL diagnostic values: `[True]`; joint KL control values: `[False]`.
- `CACHE_KL_STOP_COUNT`: **24** formal PPO updates; total Cache-suppressed future update epochs: **36**.
- `UAV_KL_STOP_COUNT`: **4** formal PPO updates; total UAV-suppressed future update epochs: **8**.
- `BOTH_BRANCH_STOP_COUNT`: **0**; PPO update schedules have lengths `[(4, 4)]` (Cache/UAV), with critic updates otherwise retained under the existing semantics.
- Trigger totals: Cache-only `59`, UAV-only `11`, both `0`, neither `1130`.
- `CACHE_STOP_SUPPRESSED_UAV = NO`: cross-suppression flags true in **0** rows; `24/24` Cache-stop rows kept the UAV branch active for the full remaining schedule.
- `UAV_STOP_SUPPRESSED_CACHE = NO`: cross-suppression flags true in **0** rows; `4/4` UAV-stop rows kept the Cache branch active for the full remaining schedule.
- First Cache independent stop: epoch `4`, Cache stop epoch `2`, UAV active schedule `[True, True, True, True]`, Cache suppressed `2`.
- First UAV independent stop: epoch `166`, UAV stop epoch `3`, Cache active schedule `[True, True, True, True]`, UAV suppressed `1`.

## Final20 Cache and RSU diagnostics

The added fields are diagnostic-only: per-epoch Cache operation counts/rates by role and RSU state/update statistics. They are not consumed by the reward, policy loss, gate, or critic.

| Role | KEEP count/rate | ADD count/rate | DELETE count/rate | SWAP count/rate | Total samples |
|---|---:|---:|---:|---:|---:|
| SV | 10699/35.66% | 7348/24.49% | 7342/24.47% | 4611/15.37% | 30000 |
| RSU | 4557/37.97% | 2626/21.88% | 2742/22.85% | 2075/17.29% | 12000 |
| UAV | 9563/79.69% | 1182/9.85% | 1183/9.86% | 72/0.60% | 12000 |

- Final20 RSU mean cached service count: `0.566833`; nonempty fraction: `45.68%`; capacity utilization: `8.34%`.
- Final20 RSU requested update rate/count: `62.02%` / `7443`; actual change rate/count: `62.02%` / `7443`; operation samples: `12000`.

## Health, checkpoints, and scope closure

- Recursive finite-value scan over summary, manifest, all 300 episode rows, and all 300 update rows: **NO** NaN/Inf detected.
- `run_summary.json` reports `nan_inf_detected = False`.
- Latest checkpoint: `C:/Users/tangning/Desktop/code/adaptive_cache_mec/runs/ppo_v10d_decoupled_encoder_branch_kl_6e4_formal/latest_checkpoint.pt`; best checkpoint: `C:/Users/tangning/Desktop/code/adaptive_cache_mec/runs/ppo_v10d_decoupled_encoder_branch_kl_6e4_formal/best_train_return_checkpoint.pt`.
- Stage checkpoint copies at 60/120/180/240/300 were not emitted because the existing runner has no stage-copy support; no runner refactor was made for this formal run. Latest and best checkpoints were retained.
- Formal run directory: `C:/Users/tangning/Desktop/code/adaptive_cache_mec/runs/ppo_v10d_decoupled_encoder_branch_kl_6e4_formal`; episode metrics: `C:/Users/tangning/Desktop/code/adaptive_cache_mec/runs/ppo_v10d_decoupled_encoder_branch_kl_6e4_formal/episode_metrics.jsonl`; update metrics: `C:/Users/tangning/Desktop/code/adaptive_cache_mec/runs/ppo_v10d_decoupled_encoder_branch_kl_6e4_formal/update_metrics.jsonl`.
- `FINAL_TEST_SEEDS_USED = NO`; `V10D_5E4_USED = NO`; `PUSHED = NO` at report-generation time.

## Reproducibility record

- Launch manifest Git state: `b72c47d4c418eb73e506773887cf2dc12c2a50da` with dirty working tree recorded as `True`; the formal report and requested source commit are finalized after training.
- Training was resumed once from the intact epoch-276 latest checkpoint after an intentional user interruption; the final target remains exactly 300 epochs and the pool cursor advanced from index 276 to 300.

`REPORT = reports/v10d_6e4_formal_training.md`
