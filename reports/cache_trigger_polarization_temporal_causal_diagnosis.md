# Cache Trigger Polarization Temporal Causal Diagnosis

## Scope and execution guardrails

This diagnosis used corrected-v2 checkpoints at epochs 10, 20, ..., 140 for the 5e-4 and 1e-3 runs. It used only the training pool realizations `20001`, `20100`, `20200`, and `20300`. No training, optimizer step, validation, test, or final unseen seed was used.

Stage A evaluated every checkpoint policy on a fixed 300-slot production observation/mask panel per realization, generated once from that run's epoch-140 production rollout. This isolates checkpoint policy evolution from changing trajectory state distributions. Stage B used fresh production rollouts at the selected checkpoint and exact formal symmetric expected-Q branches at slots `20, 60, 100, 140, 180, 220, 260, 280`. Epoch 140 reused the existing symmetric report and executed zero new Q branches.

SV cache outputs are tracked by stable cache-head index. The displayed SV node is the seed-20001 node label; physical SV node IDs vary across realizations.

A direction is called significant only when at least 75% of the 32 fixed realization/slot samples have that sign. Lower rates are reported as `MIXED`. An epoch-10 onset is left-censored because no earlier requested checkpoint exists.

## Onset and temporal classification

### corrected-v2 5e-4

| Head | Node | Role | P90 onset | P99 onset | P999 onset | Class | Evidence |
|---:|---|---|---|---|---|---|---|
| 0 | 46 | SV | 20 UPDATE | 20 UPDATE | 100 KEEP | INCONCLUSIVE | sampled sign rates do not satisfy any 75% temporal causal pattern |
| 1 | 10 | SV | 50 UPDATE | 70 UPDATE | NA | INCONCLUSIVE | sampled sign rates do not satisfy any 75% temporal causal pattern |
| 2 | 0 | SV | 70 KEEP | 70 KEEP | 70 KEEP | D | policy and significant symmetric-Q direction remained aligned after onset |
| 3 | 23 | SV | 50 UPDATE | 110 UPDATE | NA | INCONCLUSIVE | sampled sign rates do not satisfy any 75% temporal causal pattern |
| 4 | 59 | SV | 30 UPDATE | 110 KEEP | 110 KEEP | A | significant symmetric Q opposed the checkpoint policy side near onset at epoch(s) [20] |
| 5 | RSU_0 | RSU | 10 UPDATE (left-censored) | 50 KEEP | 60 KEEP | INCONCLUSIVE | epoch-10 onset is left-censored and sampled sign rates do not satisfy a 75% causal pattern |
| 6 | RSU_1 | RSU | 10 UPDATE (left-censored) | 20 UPDATE | 130 KEEP | INCONCLUSIVE | epoch-10 onset is left-censored and sampled sign rates do not satisfy a 75% causal pattern |
| 7 | UAV_0 | UAV | 20 UPDATE | 60 UPDATE | NA | INCONCLUSIVE | sampled sign rates do not satisfy any 75% temporal causal pattern |
| 8 | UAV_1 | UAV | 20 UPDATE | NA | NA | INCONCLUSIVE | sampled sign rates do not satisfy any 75% temporal causal pattern |

### corrected-v2 1e-3

| Head | Node | Role | P90 onset | P99 onset | P999 onset | Class | Evidence |
|---:|---|---|---|---|---|---|---|
| 0 | 46 | SV | 20 UPDATE | 70 UPDATE | NA | INCONCLUSIVE | sampled sign rates do not satisfy any 75% temporal causal pattern |
| 1 | 10 | SV | 40 KEEP | 40 KEEP | 50 KEEP | D | policy and significant symmetric-Q direction remained aligned after onset |
| 2 | 0 | SV | 10 UPDATE (left-censored) | 70 KEEP | 70 KEEP | INCONCLUSIVE | epoch-10 onset is left-censored and sampled sign rates do not satisfy a 75% causal pattern |
| 3 | 23 | SV | 50 KEEP | 50 KEEP | 60 KEEP | INCONCLUSIVE | sampled sign rates do not satisfy any 75% temporal causal pattern |
| 4 | 59 | SV | 10 UPDATE (left-censored) | 10 UPDATE | 80 UPDATE | INCONCLUSIVE | epoch-10 onset is left-censored and sampled sign rates do not satisfy a 75% causal pattern |
| 5 | RSU_0 | RSU | 10 UPDATE (left-censored) | 50 UPDATE | 70 UPDATE | INCONCLUSIVE | epoch-10 onset is left-censored and sampled sign rates do not satisfy a 75% causal pattern |
| 6 | RSU_1 | RSU | 40 UPDATE | 50 UPDATE | 70 UPDATE | INCONCLUSIVE | sampled sign rates do not satisfy any 75% temporal causal pattern |
| 7 | UAV_0 | UAV | 50 UPDATE | 60 UPDATE | NA | INCONCLUSIVE | sampled sign rates do not satisfy any 75% temporal causal pattern |
| 8 | UAV_1 | UAV | 10 UPDATE (left-censored) | 80 UPDATE | NA | INCONCLUSIVE | epoch-10 onset is left-censored and sampled sign rates do not satisfy a 75% causal pattern |

## Checkpoint direction series

Entries use `epoch:sign(rate)` for symmetric expected-Q and production TD-Q. P(update) lists all Stage A checkpoints.

### corrected-v2 5e-4

#### Head 0 / 46 / SV

- `P_UPDATE_BY_CHECKPOINT`: 10:0.858, 20:0.997, 30:0.997, 40:0.991, 50:0.957, 60:0.649, 70:0.762, 80:0.327, 90:0.310, 100:0.000, 110:0.000, 120:0.000, 130:0.000, 140:0.000
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 10:MIXED(0.531), 20:MIXED(0.594), 30:MIXED(0.688), 80:MIXED(0.688), 140:KEEP(1.000)
- `TDQ_SIGN_BY_CHECKPOINT`: 10:MIXED(0.531), 20:MIXED(0.531), 30:MIXED(0.594), 80:MIXED(0.562), 140:KEEP(1.000)
- `TEMPORAL_CAUSAL_CLASS`: **INCONCLUSIVE**

#### Head 1 / 10 / SV

- `P_UPDATE_BY_CHECKPOINT`: 10:0.471, 20:0.842, 30:0.814, 40:0.874, 50:0.971, 60:0.981, 70:0.993, 80:0.990, 90:0.983, 100:0.983, 110:0.980, 120:0.990, 130:0.988, 140:0.994
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 40:MIXED(0.656), 50:MIXED(0.656), 60:MIXED(0.500), 80:MIXED(0.562), 140:MIXED(0.625)
- `TDQ_SIGN_BY_CHECKPOINT`: 40:MIXED(0.531), 50:MIXED(0.594), 60:MIXED(0.594), 80:MIXED(0.594), 140:MIXED(0.531)
- `TEMPORAL_CAUSAL_CLASS`: **INCONCLUSIVE**

#### Head 2 / 0 / SV

- `P_UPDATE_BY_CHECKPOINT`: 10:0.950, 20:0.747, 30:0.819, 40:0.736, 50:0.419, 60:0.190, 70:0.001, 80:0.001, 90:0.000, 100:0.000, 110:0.000, 120:0.000, 130:0.000, 140:0.000
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 60:MIXED(0.531), 70:KEEP(0.969), 80:KEEP(1.000), 140:KEEP(1.000)
- `TDQ_SIGN_BY_CHECKPOINT`: 60:MIXED(0.531), 70:KEEP(0.969), 80:KEEP(1.000), 140:KEEP(1.000)
- `TEMPORAL_CAUSAL_CLASS`: **D**

#### Head 3 / 23 / SV

- `P_UPDATE_BY_CHECKPOINT`: 10:0.422, 20:0.767, 30:0.867, 40:0.820, 50:0.970, 60:0.982, 70:0.993, 80:0.985, 90:0.969, 100:0.982, 110:0.993, 120:0.994, 130:0.985, 140:0.997
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 40:MIXED(0.531), 50:MIXED(0.562), 60:MIXED(0.656), 80:MIXED(0.562), 140:MIXED(0.625)
- `TDQ_SIGN_BY_CHECKPOINT`: 40:MIXED(0.500), 50:MIXED(0.594), 60:MIXED(0.719), 80:MIXED(0.625), 140:MIXED(0.625)
- `TEMPORAL_CAUSAL_CLASS`: **INCONCLUSIVE**

#### Head 4 / 59 / SV

- `P_UPDATE_BY_CHECKPOINT`: 10:0.127, 20:0.008, 30:0.901, 40:0.958, 50:0.983, 60:0.984, 70:0.935, 80:0.730, 90:0.617, 100:0.019, 110:0.001, 120:0.001, 130:0.000, 140:0.000
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 20:UPDATE(0.875), 30:MIXED(0.594), 40:MIXED(0.562), 80:MIXED(0.500), 140:KEEP(0.969)
- `TDQ_SIGN_BY_CHECKPOINT`: 20:MIXED(0.500), 30:MIXED(0.594), 40:MIXED(0.656), 80:MIXED(0.625), 140:KEEP(0.969)
- `TEMPORAL_CAUSAL_CLASS`: **A**

#### Head 5 / RSU_0 / RSU

- `P_UPDATE_BY_CHECKPOINT`: 10:0.981, 20:0.996, 30:0.650, 40:0.350, 50:0.002, 60:0.000, 70:0.000, 80:0.000, 90:0.000, 100:0.000, 110:0.000, 120:0.000, 130:0.000, 140:0.000
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 10:MIXED(0.625), 20:MIXED(0.500), 80:KEEP(1.000), 140:KEEP(1.000)
- `TDQ_SIGN_BY_CHECKPOINT`: 10:MIXED(0.562), 20:MIXED(0.688), 80:KEEP(1.000), 140:KEEP(1.000)
- `TEMPORAL_CAUSAL_CLASS`: **INCONCLUSIVE**

#### Head 6 / RSU_1 / RSU

- `P_UPDATE_BY_CHECKPOINT`: 10:0.947, 20:0.997, 30:0.993, 40:0.972, 50:0.986, 60:0.974, 70:0.989, 80:0.950, 90:0.965, 100:0.839, 110:0.802, 120:0.343, 130:0.000, 140:0.000
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 10:MIXED(0.656), 20:MIXED(0.656), 80:MIXED(0.531), 140:KEEP(1.000)
- `TDQ_SIGN_BY_CHECKPOINT`: 10:MIXED(0.656), 20:MIXED(0.656), 80:MIXED(0.594), 140:KEEP(1.000)
- `TEMPORAL_CAUSAL_CLASS`: **INCONCLUSIVE**

#### Head 7 / UAV_0 / UAV

- `P_UPDATE_BY_CHECKPOINT`: 10:0.613, 20:0.955, 30:0.966, 40:0.963, 50:0.981, 60:0.991, 70:0.992, 80:0.995, 90:0.996, 100:0.954, 110:0.976, 120:0.994, 130:0.946, 140:0.993
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 10:MIXED(0.562), 20:MIXED(0.562), 30:MIXED(0.625), 80:MIXED(0.531), 140:MIXED(0.594)
- `TDQ_SIGN_BY_CHECKPOINT`: 10:MIXED(0.562), 20:MIXED(0.562), 30:MIXED(0.625), 80:UPDATE(0.750), 140:MIXED(0.562)
- `TEMPORAL_CAUSAL_CLASS`: **INCONCLUSIVE**

#### Head 8 / UAV_1 / UAV

- `P_UPDATE_BY_CHECKPOINT`: 10:0.686, 20:0.959, 30:0.988, 40:0.961, 50:0.991, 60:0.964, 70:0.987, 80:0.984, 90:0.983, 100:0.848, 110:0.872, 120:0.955, 130:0.943, 140:0.976
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 10:MIXED(0.625), 20:MIXED(0.625), 30:MIXED(0.562), 80:MIXED(0.625), 140:MIXED(0.656)
- `TDQ_SIGN_BY_CHECKPOINT`: 10:MIXED(0.500), 20:MIXED(0.562), 30:MIXED(0.625), 80:MIXED(0.594), 140:MIXED(0.625)
- `TEMPORAL_CAUSAL_CLASS`: **INCONCLUSIVE**

### corrected-v2 1e-3

#### Head 0 / 46 / SV

- `P_UPDATE_BY_CHECKPOINT`: 10:0.776, 20:0.962, 30:0.984, 40:0.936, 50:0.989, 60:0.981, 70:0.999, 80:0.998, 90:0.999, 100:0.989, 110:0.991, 120:0.997, 130:0.998, 140:0.995
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 10:MIXED(0.531), 20:MIXED(0.531), 30:MIXED(0.562), 80:MIXED(0.656), 140:MIXED(0.562)
- `TDQ_SIGN_BY_CHECKPOINT`: 10:MIXED(0.562), 20:MIXED(0.594), 30:MIXED(0.531), 80:MIXED(0.531), 140:MIXED(0.562)
- `TEMPORAL_CAUSAL_CLASS`: **INCONCLUSIVE**

#### Head 1 / 10 / SV

- `P_UPDATE_BY_CHECKPOINT`: 10:0.808, 20:0.653, 30:0.166, 40:0.001, 50:0.000, 60:0.000, 70:0.000, 80:0.000, 90:0.000, 100:0.000, 110:0.000, 120:0.000, 130:0.000, 140:0.000
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 30:KEEP(0.750), 40:KEEP(0.906), 50:KEEP(0.844), 80:KEEP(0.844), 140:KEEP(1.000)
- `TDQ_SIGN_BY_CHECKPOINT`: 30:MIXED(0.688), 40:KEEP(0.875), 50:MIXED(0.719), 80:KEEP(0.844), 140:KEEP(1.000)
- `TEMPORAL_CAUSAL_CLASS`: **D**

#### Head 2 / 0 / SV

- `P_UPDATE_BY_CHECKPOINT`: 10:0.930, 20:0.986, 30:0.982, 40:0.946, 50:0.893, 60:0.044, 70:0.000, 80:0.000, 90:0.000, 100:0.000, 110:0.000, 120:0.000, 130:0.000, 140:0.000
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 10:MIXED(0.562), 20:MIXED(0.500), 80:KEEP(1.000), 140:KEEP(1.000)
- `TDQ_SIGN_BY_CHECKPOINT`: 10:MIXED(0.500), 20:MIXED(0.625), 80:KEEP(1.000), 140:KEEP(1.000)
- `TEMPORAL_CAUSAL_CLASS`: **INCONCLUSIVE**

#### Head 3 / 23 / SV

- `P_UPDATE_BY_CHECKPOINT`: 10:0.835, 20:0.898, 30:0.510, 40:0.566, 50:0.001, 60:0.000, 70:0.000, 80:0.000, 90:0.000, 100:0.000, 110:0.000, 120:0.000, 130:0.000, 140:0.000
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 40:MIXED(0.500), 50:MIXED(0.625), 60:KEEP(0.906), 80:KEEP(1.000), 140:KEEP(1.000)
- `TDQ_SIGN_BY_CHECKPOINT`: 40:MIXED(0.500), 50:MIXED(0.625), 60:KEEP(0.812), 80:KEEP(1.000), 140:KEEP(1.000)
- `TEMPORAL_CAUSAL_CLASS`: **INCONCLUSIVE**

#### Head 4 / 59 / SV

- `P_UPDATE_BY_CHECKPOINT`: 10:0.990, 20:0.997, 30:0.995, 40:0.999, 50:0.995, 60:0.985, 70:0.995, 80:1.000, 90:1.000, 100:0.999, 110:0.999, 120:0.999, 130:1.000, 140:1.000
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 10:MIXED(0.688), 20:MIXED(0.562), 80:MIXED(0.625), 140:MIXED(0.594)
- `TDQ_SIGN_BY_CHECKPOINT`: 10:MIXED(0.719), 20:MIXED(0.531), 80:MIXED(0.562), 140:MIXED(0.625)
- `TEMPORAL_CAUSAL_CLASS`: **INCONCLUSIVE**

#### Head 5 / RSU_0 / RSU

- `P_UPDATE_BY_CHECKPOINT`: 10:0.971, 20:0.986, 30:0.991, 40:0.986, 50:0.999, 60:0.998, 70:1.000, 80:1.000, 90:1.000, 100:0.997, 110:0.998, 120:0.999, 130:0.999, 140:0.999
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 10:MIXED(0.562), 20:MIXED(0.562), 80:MIXED(0.594), 140:MIXED(0.500)
- `TDQ_SIGN_BY_CHECKPOINT`: 10:MIXED(0.625), 20:MIXED(0.562), 80:MIXED(0.594), 140:MIXED(0.562)
- `TEMPORAL_CAUSAL_CLASS`: **INCONCLUSIVE**

#### Head 6 / RSU_1 / RSU

- `P_UPDATE_BY_CHECKPOINT`: 10:0.293, 20:0.828, 30:0.829, 40:0.916, 50:0.995, 60:0.995, 70:1.000, 80:1.000, 90:1.000, 100:1.000, 110:1.000, 120:1.000, 130:1.000, 140:1.000
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 30:MIXED(0.594), 40:MIXED(0.594), 50:MIXED(0.656), 80:MIXED(0.531), 140:MIXED(0.562)
- `TDQ_SIGN_BY_CHECKPOINT`: 30:MIXED(0.531), 40:MIXED(0.500), 50:MIXED(0.531), 80:MIXED(0.531), 140:MIXED(0.500)
- `TEMPORAL_CAUSAL_CLASS`: **INCONCLUSIVE**

#### Head 7 / UAV_0 / UAV

- `P_UPDATE_BY_CHECKPOINT`: 10:0.732, 20:0.737, 30:0.768, 40:0.887, 50:0.946, 60:0.995, 70:0.999, 80:0.990, 90:0.993, 100:0.995, 110:0.994, 120:0.998, 130:0.996, 140:0.992
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 40:MIXED(0.500), 50:MIXED(0.500), 60:MIXED(0.656), 80:MIXED(0.625), 140:MIXED(0.562)
- `TDQ_SIGN_BY_CHECKPOINT`: 40:MIXED(0.562), 50:MIXED(0.500), 60:MIXED(0.531), 80:MIXED(0.562), 140:MIXED(0.531)
- `TEMPORAL_CAUSAL_CLASS`: **INCONCLUSIVE**

#### Head 8 / UAV_1 / UAV

- `P_UPDATE_BY_CHECKPOINT`: 10:0.962, 20:0.967, 30:0.918, 40:0.974, 50:0.885, 60:0.985, 70:0.975, 80:0.994, 90:0.998, 100:0.995, 110:0.999, 120:0.988, 130:0.993, 140:0.949
- `EXPECTED_Q_SIGN_BY_CHECKPOINT`: 10:MIXED(0.531), 20:MIXED(0.562), 80:MIXED(0.500), 140:MIXED(0.594)
- `TDQ_SIGN_BY_CHECKPOINT`: 10:MIXED(0.594), 20:MIXED(0.500), 80:MIXED(0.625), 140:MIXED(0.625)
- `TEMPORAL_CAUSAL_CLASS`: **INCONCLUSIVE**

## Interpretation

Class counts: 5e-4 `{'A': 1, 'B': 0, 'C': 0, 'D': 1, 'INCONCLUSIVE': 7}`; 1e-3 `{'A': 0, 'B': 0, 'C': 0, 'D': 1, 'INCONCLUSIVE': 8}`.

At the P90 onset checkpoint, symmetric expected-Q was significant and aligned for `2/18` heads, significant and opposed for `0/18`, and mixed under the 75% rule for `16/18`.

Class A means significant symmetric expected-Q already opposed the policy side at the onset boundary. Class B requires significant Q support at onset followed by a significant post-onset reversal while the policy stayed polarized. Class C requires a persistent TD-Q/expected-Q sign mismatch of at least 75% before and at onset. Class D requires significant policy/Q alignment at every evaluated post-onset checkpoint. The 75% rule makes many visually directional but noisy heads inconclusive by design.

The sole class-A event is 5e-4 head 4: at epoch 20 its policy was transiently KEEP-polarized (`P(update)=0.0076`) while 87.5% of symmetric expected-Q samples favored UPDATE. This occurs before the head's confirmed sustained P90 onset at epoch 30 and its confirmed sustained P99 onset at epoch 110, which is the evidence behind `POLICY_WRONG_BEFORE_SATURATION_SUPPORTED = YES`. The head then changed side, so this event does not demonstrate an unrecoverable hysteresis.

The widespread mixed expected-Q direction at onset is compatible with early determinization, but this study did not observe the required significant post-onset Q flip or a persistent pre-onset TD-Q divergence. Premature determinization and intervention choices therefore remain inconclusive. Temporal association alone does not establish that entropy or a symmetric-reference change would fix the learned policy.

## Required conclusions

`POLICY_WRONG_BEFORE_SATURATION_SUPPORTED = YES`

`EXPECTED_Q_FLIP_AFTER_SATURATION_SUPPORTED = NO`

`PRODUCTION_TDQ_DIVERGENCE_PRECEDES_POLARIZATION = NO`

`PREMATURE_DETERMINIZATION_SUPPORTED = INCONCLUSIVE`

`REFERENCE_ASYMMETRY_CAUSALLY_PRECEDES_POLARIZATION = NO`

`ENTROPY_EXPLORATION_FIX_JUSTIFIED = INCONCLUSIVE`

`SYMMETRIC_REFERENCE_FIX_JUSTIFIED = INCONCLUSIVE`

`TRAINING_EXECUTED = NO`

`OPTIMIZER_STEP_CALLS = 0`

`VALIDATION_EXECUTED = NO`

`TEST_EXECUTED = NO`

## Reproducibility counts

- New exact environment executions: `30082`
- New exact logical candidates: `29602`
- New production TD-Q environment executions: `2080`
- Reused epoch-140 Q rows: `576`
- Stage B per-sample rows: `2656`
