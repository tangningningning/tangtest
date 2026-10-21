# PER_SLOT_ANALYTIC Reward normalization assessment

## Scope and result

Read the current formal config and code, existing calibration notes, and six real Training slots. The six slots were independently evaluated at the realization's initial Cache/UAV state with KEEP/STAY; no full Episode, Validation/Test data, or PPO training was used. Only this report is added; source, formal config, and data pools were not changed.

```ini
FORMAL_TASK_MODEL = PER_SLOT_ANALYTIC

CURRENT_T_REF = 30.6289851662 s
CURRENT_E_REF = 3.06289851662 J

T_REF_CALIBRATION_SOURCE = Stage6 Training, 3343 arrivals; P95(cpu_cycles) / 1 GHz; exact episode list not recorded
E_REF_CALIBRATION_SOURCE = Same Training P95 workload, TV Local E=kappa*cycles*frequency^2, kappa=1.0e-28; exact episode list not recorded

CURRENT_REWARD_UNIT_CONSISTENCY = YES
OLD_CALIBRATION_MODEL_MATCHES_NEW_MODEL = NO

TRAIN_TASKS_ANALYZED = 81
TRAIN_EPISODES_SAMPLED = 3

DELAY_P50_S = 3.629034
DELAY_P90_S = 4.952604
DELAY_P95_S = 5.326980

ENERGY_P50_J = 0.058785
ENERGY_P90_J = 0.143914
ENERGY_P95_J = 0.243862

NORMALIZED_DELAY_PENALTY = mean 0.068855; p50/p90/p95 0.071090/0.097018/0.104352
NORMALIZED_ENERGY_PENALTY = mean 0.010374; p50/p90/p95 0.007677/0.018794/0.031847

MATCHING_DEPENDS_ON_REFS = YES
KKT_DEPENDS_ON_REFS = YES

T_REF_CHANGE_REQUIRED = YES
E_REF_CHANGE_REQUIRED = YES

RECOMMENDED_ACTION = RECALIBRATE

SOURCE_CODE_CHANGED = NO
FORMAL_CONFIG_CHANGED = NO
FORMAL_DATA_CHANGED = NO
PPO_TRAINING_EXECUTED = NO
GITHUB_PUSHED = NO
```

## Calibration provenance

The current `configs/proposed/system_config.toml` records `T_ref=30.6289851662` and `E_ref=3.06289851662` at `objective.T_ref/E_ref`. Its comments say the values are frozen from Stage6 Training before TV Local was removed: 3,343 Training arrivals had a P95 CPU workload of `30,628,985,166.15863 cycles`; dividing by the historical TV CPU rate of `1 GHz` gives `T_ref`. The energy value uses that same percentile workload in the old local-compute model, `E_local=kappa*cycles*frequency^2`, with `kappa=1.0e-28` and `frequency=1 GHz`. The current config does not name the 3,343 arrivals' Episode IDs or slot-selection rule, so that part of the provenance remains untraceable from the local record.

`reports/PARAMETER_CALIBRATION.md` is an earlier calibration record, not the source of the current numbers: it uses 4,210 arrivals from the earlier `data/generated/tasks.csv` and reports `30.7900934699 s` / `3.079009347 J`. The current values match the later Stage6 Training annotation and `reports/stage6_reward_training_signal_check.md`, not that earlier pair.

The current execution code has no TV Local target. Tasks are matched to Edge (SV/RSU/UAV) or Cloud. Thus the old references have the right dimensions but were calibrated to a different physical quantity: local CPU completion time and local CPU energy, instead of current end-to-end offload delay and vehicle transmit energy.

## Current utility inputs and sampled scale

For each task, the current utility is `u_i = 1 - 0.6*(T_i/T_ref) - 0.4*(E_i/E_ref)`. The fixed completion contribution is therefore `1` per task; it is separate from the two deductions. In the 81-task sample, mean delay deduction was `0.068855`, mean energy deduction was `0.010374`, and mean utility was `0.920772`. The delay deduction was about 6.6 times the energy deduction in this sample. The nominal weights 0.6 and 0.4 do not imply that the realized deductions contribute 60% and 40%: the observed normalized quantities determine their magnitudes.

Sampled slots were `episode_20001` slots 0 and 150, `episode_20101` slots 0 and 150, and `episode_20300` slots 150 and 299 (absolute times 100, 250, and 399 seconds). All are Training data. Aggregate normalized ratios were:

| Metric | P50 | P90 | P95 |
|---|---:|---:|---:|
| `T_i/T_ref` | 0.118484 | 0.161697 | 0.173920 |
| `E_i/E_ref` | 0.019193 | 0.046986 | 0.079618 |
| `0.6*T_i/T_ref` | 0.071090 | 0.097018 | 0.104352 |
| `0.4*E_i/E_ref` | 0.007677 | 0.018794 | 0.031847 |

There were 32 Edge and 49 Cloud tasks. Edge mean effective upload/compute delays were `0.592466 / 1.674121 s`, with mean vehicle transmit energy `0.118213 J`; Cloud means were `3.525442 / 0.804698 s`, with `0.054109 J`. The Cloud upload-delay figure includes the configured factor 13. Aggregate means were `2.366736 s` effective communication delay and `1.148174 s` compute delay. Independently reconstructed per-task components matched the formal execution results for all 81 tasks. In this sample, mean task utility per Slot summed to about `12.43`; Cache and UAV costs were zero because the sampled action was KEEP/STAY, so this does not estimate a new policy's action-cost distribution. As historical scale context only, the old Stage6 report records mean per-Slot Cache and UAV costs of about `0.0141` and `0.0018` against mean task utility about `13.33`; those are not outcomes of a policy trained under PER_SLOT_ANALYTIC.

The Cloud energy implementation is `P_tx * input_bits / cloud_rate`; its factor 13 multiplies upload delay only, not transmit energy. Edge delay is upload time plus allocated CPU time, and Edge energy is `P_tx * upload time`. The ratios are dimensionless seconds/seconds and joules/joules, so the current Reward's unit handling is consistent even though the reference quantities are model-mismatched.

This is a small descriptive sample (81 tasks in six selected slots), not a full-Training estimate and not a final calibration sample. The observed P95 delay is about 5.8 times below current `T_ref`; sampled P95 transmit energy is about 12.6 times below current `E_ref`. Alongside the known change from local execution to offloading-only, that is sufficient evidence to recommend recalibration, but not to set new constants from these six slots.

## Dependency and recommendation

Both references affect more than the final Reward. Matching and candidate/upper-bound evaluation use normalized task utility. The CPU KKT objective contains `omega_T/T_ref`; the bandwidth KKT objective contains both `omega_T/T_ref` and `omega_E*P_tx/E_ref`. Changing either value may therefore change Matching decisions, KKT resource shares, and the resulting utility. If recalibrated, update the formal experiment contract and its recorded policy/checkpoint schema together.

Recommended next calibration: use Training only and predeclare a deterministic reference allocation/configuration independent of `T_ref/E_ref`; evaluate the current per-Slot analytic total delay and vehicle transmit energy on that fixed reference, then set each constant to its own empirical P95 to preserve the existing upper-tail convention. Record the exact Training Episode IDs, slots, reference Cache/UAV state, assignment, resource allocation rule, and task count. Do not choose values to raise Reward/Return or to equalize the two weighted deductions. The present sample is evidence of mismatch, not a replacement calibration.

## Code and record locations

- `configs/proposed/system_config.toml:398-414` — current constants and frozen 3,343-arrival Stage6 Training provenance note.
- `reports/stage6_reward_training_signal_check.md:16-24` — historical Reward formula and local-execution calibration description.
- `reports/PARAMETER_CALIBRATION.md:53-55` — earlier, distinct 4,210-arrival calibration.
- `reports/stageB1_remove_tv_local_execution.md:7-12` — TV Local removal and preservation of the old references.
- `src/common/system_model.py:173-252,456-479` — Edge/Cloud delay and vehicle energy models, and normalized task utility.
- `src/decision/task_allocation.py:2872,2912-2913,3476,3721-3745` — references in CPU/BW allocation, candidate utility, and Cloud delay/energy evaluation.
- `src/decision/slot_environment.py:614-620,1256-1269` — per-task utility aggregation and `G=Q-C_cache-C_UAV`.
