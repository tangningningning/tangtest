# PPO V6 multi-realization seed pipeline plan

## 1. Scope and conclusion

This report is a static audit of the data-generation path used by the frozen PPO V6 baseline. No PPO, SUMO, Matching, KKT, formal data generation, or formal evaluation was run.

The current project is reproducible for the frozen single realization when its several independently configured seeds and software environment are held fixed, but it does **not** yet expose or enforce one complete `realization_seed` contract. The current values are manually coordinated rather than derived from one master seed, all formal outputs share `data/generated/`, and the unified regeneration path can retrain and overwrite the popularity GRU when a realization-dependent prediction artifact changes. Therefore:

- `CURRENT_SEED_CHAIN_COMPLETE = NO`
- `CROSS_REALIZATION_OVERWRITE_RISK = YES`
- `TRAIN_TEST_LEAKAGE_RISK = YES` as a pipeline capability/risk, not as evidence that the preserved V6 run is already contaminated.

The minimum safe target is to freeze service definitions and calibration/model artifacts globally, put every realization-dependent artifact in a seed-specific directory, derive named random streams from one master seed, and make per-realization popularity prediction inference-only.

## 2. Current seed dependency graph

```text
experiment.seed = 4401
├─ PPO Python / NumPy / Torch RNG
├─ runtime edge-cache initial permutation
└─ regenerate_artifacts.py cache-only fallback -> tv_local_cache.json

simulation.traffic_seed = 4401
└─ randomTrips.py -s -> sumo_demand.rou.xml
                         └─ SUMO traffic population/routes

simulation.sumo_seed = 4401
├─ SUMO --seed -> trajectory.csv
├─ research vehicle selection (random.Random(4401))
└─ SV selection (random.Random(4402), implemented as seed + 1)
   └─ vehicle_roles.json

task.master_seed = 4401
└─ NumPy SeedSequence.spawn(5)
   ├─ service_catalog       = 13703533950003060729
   ├─ popularity_evolution  = 10018347655802798118
   ├─ task_arrival          = 9948887260315234864
   ├─ service_request       = 17798879212459637611
   └─ workload_variation    = 1922379454717095758
      ├─ service_catalog.json
      ├─ popularity_history.csv
      ├─ popularity.csv
      └─ tasks.csv

prediction.vehicle.observation_noise_seed = 9101
├─ independent calibration noisy observations -> Q/R + residual_blocks.npz
└─ copied from residual bank into formal replay -> vehicle_predictions.csv

prediction.popularity.sequence_seed_base = 880001
├─ synthetic GRU train sequences: 880001..880024
├─ synthetic GRU validation sequences: 880025..880030
└─ synthetic GRU test sequences: 880031..880036

prediction.popularity.training_seed = 20260819
└─ Python / NumPy / Torch model initialization and batch order
   └─ popularity_gru.pt

scenario.master_seed = 20260820
├─ SeedSequence([seed, origin_slot]).spawn(4)
│  ├─ residual (created but vehicle residual sampling currently uses the stream below)
│  ├─ arrival
│  ├─ service_request
│  └─ workload
└─ SeedSequence([seed, 777])
   └─ vehicle residual-block sampling
      └─ prediction_scenarios.npz
```

The independent vehicle calibration artifacts record a separate traffic/SUMO/research seed `4402` and service-role seed `4403`. Those values exist in `data/calibration/sumo_demand.meta.json` and `data/calibration/vehicle_roles.json`; they are not currently represented as a dedicated calibration seed block in `system_config.toml`.

### 2.1 Seed inventory

| SEED_NAME | DEFINED_AT | CONSUMED_BY | AFFECTED_ARTIFACTS | CURRENT_VALUE | DERIVATION_RULE |
|---|---|---|---|---:|---|
| `experiment.seed` | `system_config.toml:31-35`; active V6 TOML has the same value | `src/ppo_simple/trainer.py:1370-1373`; `src/decision/cache_configuration.py:221,2151`; `scripts/regenerate_artifacts.py:903-908` | PPO stochasticity, runtime initial edge cache, cache-only regeneration of `tv_local_cache.json` | 4401 | Direct configuration value; not derived |
| `simulation.traffic_seed` | `system_config.toml:90` | `src/scenario/sumo_scenario.py:530-546` (`randomTrips.py -s`) | `sumo_demand.rou.xml`, then `trajectory.csv` and all vehicle-dependent descendants | 4401 | Direct configuration value; currently manually equal to `sumo_seed` |
| `simulation.sumo_seed` | `system_config.toml:88` | `src/scenario/sumo_scenario.py:631-642,929-952` | SUMO dynamics, `trajectory.csv`, research/SV/TV selection, `vehicle_roles.json` | 4401 | Direct value; SV selection uses hard-coded `+1` (=4402) |
| `task.master_seed` | `system_config.toml:133` | `src/scenario/task_scenario.py:367-379,1147-1185` | current `service_catalog.json`, `popularity_history.csv`, `popularity.csv`, `tasks.csv`; normal task generator also uses it for `tv_local_cache.json` | 4401 | `SeedSequence(master_seed).spawn(5)` in fixed stream-name order |
| `task.service_catalog` child | `src/scenario/task_scenario.py:57-63` | `make_service_catalog()` | service parameters and the current catalog | 13703533950003060729 | First spawned child of task master seed |
| `task.popularity_evolution` child | same | `make_popularity_trace_with_history()` | true Zipf rank realization in history/formal popularity | 10018347655802798118 | Second spawned child |
| `task.task_arrival` child | same | `generate_tasks()` | Bernoulli arrivals in prehistory and formal tasks | 9948887260315234864 | Third spawned child |
| `task.service_request` child | same | `generate_tasks()` | requested service IDs and observed request counts | 17798879212459637611 | Fourth spawned child |
| `task.workload_variation` child | same | `generate_tasks()` | task input sizes and CPU cycles | 1922379454717095758 | Fifth spawned child |
| `prediction.vehicle.observation_noise_seed` | `system_config.toml:427` | `src/scenario/vehicle_prediction.py:633-695`; copied from residual bank by `scenario_generator.py:235-240` | calibration Q/R/residual bank; formal noisy-observation replay and `vehicle_predictions.csv` | 9101 | Direct shared value; currently not realization-derived |
| `prediction.popularity.training_seed` | `system_config.toml:435` | `src/scenario/popularity_prediction.py:132-140,516-530` | GRU initialization, training batch order, `popularity_gru.pt` | 20260819 | Direct calibration/model-training seed |
| `prediction.popularity.sequence_seed_base` | `system_config.toml:436` | `src/scenario/popularity_prediction.py:322-335` | synthetic GRU train/validation/test datasets and checkpoint | 880001 | Contiguous `base + index`; train 1-24, val 25-30, test 31-36 |
| `scenario.master_seed` | `system_config.toml:449` | `src/scenario/scenario_generator.py:584-590,1069-1091` | stochastic vehicle residual draws and predicted task scenarios | 20260820 | Per origin: `SeedSequence([seed, origin_slot])`; vehicle residual stream: `SeedSequence([seed, 777])` |
| calibration traffic/SUMO/research seed | `data/calibration/sumo_demand.meta.json`; `data/calibration/vehicle_roles.json` | frozen calibration generation provenance | calibration demand, trajectory, roles, then residual bank | 4402 | Frozen artifact provenance; not in current TOML seed namespace |
| calibration SV-role seed | `data/calibration/vehicle_roles.json` | calibration role selection | calibration role partition | 4403 | Calibration seed `+1` |

### 2.2 Hard-coded/default seed findings

The source contains fallback defaults `4401`, `9101`, `20260819`, `880001`, and `20260820` in scenario dataclasses. Loaded TOML values override them in the normal path, but duplicated defaults remain a configuration-drift risk. It also contains two positional derivations that are not named protocol fields:

- `sumo_seed + 1` for SV selection (`sumo_scenario.py:944-952`);
- integer tag `777` for the vehicle residual stream (`scenario_generator.py:1080-1082`).

No unseeded production data-generation call to `random`, NumPy random, or Torch random was found. PPO sampling and minibatch permutations are seeded by `experiment.seed`, but PPO RNG is outside the realization-data master seed and should remain a separately controlled training seed.

## 3. Current artifact dependency graph

```text
GLOBAL INPUTS
  system/model parameters + SUMO network
          │
          ├─ deterministic node placement ───────────────> node_positions.json
          │
          ├─ traffic seed -> randomTrips demand
          │                    │
          │             SUMO + SUMO seed
          │                    │
          │                    └──────────────> trajectory.csv
          │                                           │
          │                       trajectory + role seed(s)
          │                                           │
          │                                           └─> vehicle_roles.json
          │                                                  │
          └─ task seed -> service catalog + popularity + arrivals/workloads
                                                             │
                   ┌─────────────────────────────────────────┼───────────────┐
                   v                                         v               v
          service_catalog.json                    popularity*.csv        tasks.csv
                   │                                      │
                   └─ roles + cache seed -> tv_local_cache.json

CALIBRATION INPUTS
  calibration demand/trajectory/roles + calibration observation seed
                   └─> frozen Q/R + vehicle residual bank

  synthetic popularity sequence seeds + GRU training seed + fixed process parameters
                   └─> frozen popularity_gru.pt

PER-REALIZATION PREDICTION
  formal trajectory/roles + frozen Q/R + formal observation noise
                   └─> vehicle_predictions.csv

  formal popularity_history/request counts + frozen GRU
                   └─> popularity_predictions.csv

  vehicle nominal predictions + frozen residual bank
  + popularity predictions + shared service catalog
  + prediction-scenario seed
                   └─> prediction_scenarios.npz
```

### 3.1 Direct upstream and target classification

The classification below is the recommended experimental contract. “Current behavior” records mismatches that must be repaired before multi-realization generation.

| Artifact | Direct upstream in current code | Class | Current behavior / required contract |
|---|---|---|---|
| `service_catalog.json` | service ranges/count + task `service_catalog` child RNG; current JSON also embeds task vehicle IDs and slot-0 popularity | **GLOBAL_SHARED** | Current generator makes it realization-dependent. Freeze one catalog and remove realization-specific vehicle/popularity metadata from the shared artifact. |
| `vehicle_roles.json` | complete vehicles in `trajectory.csv` + research-role seed + service-role seed | **REALIZATION_SPECIFIC** | Must live with its own trajectory and carry realization provenance. |
| `trajectory.csv` | SUMO network + demand route + traffic seed + SUMO seed + fixed traffic parameters | **REALIZATION_SPECIFIC** | Must be regenerated for each realization. |
| `tasks.csv` | realization TV IDs + shared catalog + true popularity process + arrival/request/workload streams | **REALIZATION_SPECIFIC** | Must be regenerated per realization. |
| `popularity.csv` | popularity evolution + realized request counts from prehistory/formal tasks | **REALIZATION_SPECIFIC** | Must be regenerated per realization. |
| `popularity_history.csv` | the continuous prehistory segment of the same popularity/task process | **REALIZATION_SPECIFIC** | Must stay in the same realization bundle as `popularity.csv` and `tasks.csv`. |
| `vehicle_predictions.csv` | realization trajectory/roles + SUMO network + frozen calibration Q/R + formal observation-noise stream | **REALIZATION_SPECIFIC** | Current formal observation noise is inherited from the calibration bank seed; add a realization-derived formal observation seed. |
| `popularity_gru.pt` | fixed synthetic train/val/test sequence seeds + fixed task-process definition + training seed/model hyperparameters | **CALIBRATION_SHARED** | Weights do not use the canonical realization for training, but the current script retrains and overwrites this file for each invalidated prediction run and embeds canonical metrics in checkpoint metadata. Train once, freeze, and perform inference-only per realization. |
| `prediction_scenarios.npz` | realization vehicle/popularity predictions + frozen residual bank + shared catalog + scenario streams | **REALIZATION_SPECIFIC** | Must be regenerated per realization with a realization-derived scenario seed. |
| `tv_local_cache.json` | realization TV IDs + shared catalog/capacity + cache initialization seed | **REALIZATION_SPECIFIC** | It is bound to `vehicle_roles.json`; use a dedicated realization-derived cache seed. |
| `node_positions.json` | SUMO network + RSU/UAV counts, altitude and placement parameters; deterministic sort/tie-breaking | **GLOBAL_SHARED** | Safe to share while network and deployment parameters are frozen. |

Additional artifacts:

- `sumo_demand.rou.xml`, its metadata, `popularity_predictions.csv`, and per-realization `generation_metadata.json` are **REALIZATION_SPECIFIC**.
- `data/sumo/map/osm.net.xml.gz`, the frozen system/model configuration, and the frozen service catalog are **GLOBAL_SHARED**.
- `data/calibration/trajectory.csv`, `vehicle_roles.json`, calibration demand metadata/routes, and `residual_blocks.npz` are **CALIBRATION_SHARED**.

## 4. Leakage and consistency assessment

### 4.1 Future truth used to train a predictor

**Not found.** `popularity_prediction.py:1085-1102` trains the GRU on synthetic split datasets produced from disjoint sequence seeds. Canonical realization data is used only after training for causal inference and diagnostics (`:1103-1118`). The checkpoint declares `canonical_episode_used_for_training = False` (`:1151-1153`).

The formal popularity input window is causally cut at `origin_slot - 1` by `build_causal_formal_inputs()` (`popularity_prediction.py:247-281`). Vehicle nominal prediction similarly requests the Kalman posterior at `origin_slot - 1` (`scenario_generator.py:275-290`).

### 4.2 Test realization can update the GRU

**Risk found.** `scripts/regenerate_artifacts.py:913-938` always runs `popularity_prediction.py` before `scenario_generator.py` when prediction regeneration is authorized. `popularity_prediction.py:1169-1185` writes `popularity_gru.pt` and realization-specific prediction output. There is no `train`/`test` split guard or inference-only mode. The `--allow-predictor-training` gate prevents silent retraining, but once enabled it does not distinguish a calibration operation from test-realization generation.

The model weights are trained on synthetic data rather than the canonical realization, so this is not evidence of target-label training leakage in the current preserved run. It is nevertheless an invalid test protocol because a test generation action is allowed to retrain/replace a shared model artifact and attach test-realization diagnostics to its checkpoint metadata.

### 4.3 Test realization can update the vehicle residual bank

**Not found in the unified regeneration path.** That path does not invoke `vehicle_prediction.py`. The vehicle calibration CLI rejects the configured formal roles or trajectory paths (`vehicle_prediction.py:1344-1356`), and `scenario_generator.py:102-154` requires residual-bank metadata identifying an independent calibration trajectory.

The path-equality guard is not a complete split-aware provenance system: a copied or differently named test trajectory would not be detected by path equality alone. The next protocol should therefore validate calibration manifest hashes/split identity, but there is no code evidence that the current residual bank was built from the formal/test realization. Its embedded source hashes identify `data/calibration/trajectory.csv` and `data/calibration/vehicle_roles.json`.

### 4.4 Terminal/future truth in prediction scenarios

**Not found.** Tail vehicle predictions extrapolate from the causal cutoff; formal prediction CSV truth/error fields are intentionally blank (`scenario_generator.py:293-326`). `prediction_scenarios.npz` stores predicted positions, sampled tasks, and `rho_hat`, not future truth (`:1203-1221`). Popularity rows may retain `rho_star` and `abs_error` for offline diagnostics, but the PPO loader reads only `rho_pred` (`src/ppo_simple/interface.py:683-717`).

`relative_distance_sanity()` receives formal truth for diagnostics, and popularity canonical truth is used for post-training metrics. Neither is used to construct policy input or sampled scenario payload according to the audited call graph.

### 4.5 Cross-realization sharing and overwrite/mismatch risk

**Risk found.** All formal paths currently point to the single `data/generated/` directory. Generation writes files individually with atomic replacement, but the whole artifact set is not staged and committed atomically. A failure between stages can leave, for example, seed-A trajectory/roles with seed-B tasks or prediction files.

Specific contract gaps are:

1. `scripts/regenerate_artifacts.py:1168` hard-codes `data/generated/generation_metadata.json`, even though most artifact paths are configurable.
2. `sumo_scenario.py` has no `--config` option and calls `load_config()` without a supplied path (`:1165-1230,1383`), making generated per-realization config files awkward to use.
3. `task_scenario.py:291-295` requires the role file seed to equal `task.master_seed`. That works only because both are currently 4401; it blocks clean named derivation of independent role and task streams.
4. `trainer.py:532` and `evaluator.py:231` hard-code `data/generated/popularity_predictions.csv`; other production inputs use configured paths.
5. The regeneration signature records `sumo_seed` but omits `traffic_seed` (`regenerate_artifacts.py:183-189`), so traffic-demand changes are not fully represented in the current consistency signature.
6. Cache-only regeneration uses `experiment.seed` (`regenerate_artifacts.py:903-908`), while normal task generation uses `task.master_seed` for the same `tv_local_cache.json` (`task_scenario.py:1912-1917`). The identical current value hides this inconsistency.
7. `service_catalog.json` is regenerated from the task seed and also carries realization-specific TV IDs and initial popularity, so it cannot currently be safely shared byte-for-byte.

## 5. Service catalog recommendation

`SERVICE_CATALOG_RECOMMENDATION = SHARED`.

The catalog defines service program size, mean task input, and cycles-per-bit (`task_scenario.py:418-470`). These are system workload-type definitions, not arrival realizations. Current cache capacities are derived as ratios of total catalog size (`scripts/regenerate_artifacts.py:531-603`), so regenerating the catalog per realization changes both service identities and absolute cache capacities. That would vary physical/problem parameters across seeds instead of varying only traffic, requests, and prediction errors.

For the paper's main experiment, generate the catalog once from a documented global catalog seed, freeze it with a content hash, and reuse it across all train/test realizations. Keep service count, generation envelopes, Zipf `zeta`, and arrival probability fixed. Per-realization generation should load the frozen catalog and sample only popularity evolution, arrivals, requested service IDs, workload jitter, and prediction errors.

## 6. Recommended directory layout

```text
data/
├─ shared/
│  ├─ system_config.toml                 # or immutable experiment protocol snapshot
│  ├─ service_catalog.json               # frozen service definitions
│  ├─ node_positions.json                # deterministic for frozen map/deployment
│  └─ manifests/
│     └─ shared_manifest.json             # hashes + code/SUMO/environment provenance
├─ calibration/
│  ├─ sumo_demand.rou.xml
│  ├─ sumo_demand.meta.json
│  ├─ trajectory.csv
│  ├─ vehicle_roles.json
│  ├─ vehicle_predictions.csv             # calibration diagnostic output
│  ├─ residual_blocks.npz                 # frozen Q/R + residual bank
│  ├─ popularity_gru.pt                   # train once, freeze
│  └─ calibration_manifest.json
└─ realizations/
   ├─ train/
   │  └─ seed_XXXX/
   │     ├─ realization_config.toml        # paths resolved to this bundle/shared/calibration
   │     ├─ realization_manifest.json      # master/derived seeds + all upstream hashes
   │     ├─ sumo_demand.rou.xml
   │     ├─ sumo_demand.meta.json
   │     ├─ trajectory.csv
   │     ├─ vehicle_roles.json
   │     ├─ popularity_history.csv
   │     ├─ popularity.csv
   │     ├─ tasks.csv
   │     ├─ tv_local_cache.json
   │     ├─ vehicle_predictions.csv
   │     ├─ popularity_predictions.csv
   │     └─ prediction_scenarios.npz
   └─ test/
      └─ seed_YYYY/
         └─ same realization-specific file set
```

Use a temporary sibling directory such as `seed_XXXX.building-<uuid>` and rename it to `seed_XXXX` only after every file, seed, upstream hash, shape, and causal-boundary check passes. Refuse to overwrite an existing completed seed directory unless an explicit, separately reviewed replacement operation is requested. This closes the seed-A/seed-B partial-overwrite mode without a broad refactor.

Per-realization TOMLs can keep the existing loaders: relative paths already resolve through `project_config.resolve_path()`. The minimal production change is to make every remaining hard-coded path configurable and generate a small config snapshot per bundle.

## 7. Recommended master-seed derivation

Expose exactly one data-generation input:

```text
realization_seed: unsigned 64-bit integer
```

Derive streams by stable **names**, not by Python `hash()`, positional `spawn()` order, or ad-hoc `+1`. A portable versioned rule is:

```python
payload = b"adaptive_cache_mec/realization/v1\0" + str(realization_seed).encode() \
          + b"\0" + stream_name.encode("ascii")
u64 = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
sumo_seed = 1 + (u64 % 2_147_483_646)  # only for SUMO-compatible integer fields
```

Use distinct names:

```text
traffic_demand
sumo_dynamics
research_vehicle_selection
service_vehicle_selection
popularity_evolution
task_arrival
service_request
workload_variation
tv_local_cache
formal_vehicle_observation
prediction_vehicle_residual
prediction_task_arrival
prediction_service_request
prediction_workload_variation
```

The derivation version and complete derived-seed map must be written to `realization_manifest.json`. The stream-name hash means adding a new stream later does not renumber or alter existing streams.

Do **not** derive these frozen shared/calibration seeds from `realization_seed`:

- shared service-catalog seed;
- calibration traffic, role, and observation-noise seeds;
- GRU synthetic train/validation/test sequence seeds;
- GRU model-training seed;
- PPO policy/training seed.

Those belong in `shared_manifest.json`, `calibration_manifest.json`, or the PPO run manifest. Separating environment realization from policy RNG allows fair common-random-number comparisons and independent policy replicates.

Reproducibility also requires recording Git SHA/dirty state, config hash, SUMO version, Python/NumPy/Torch versions, platform, network hash, shared catalog hash, calibration artifact hashes, and generator schema version. `generation_metadata.generated_at_utc` is provenance and should be excluded from deterministic content identity; otherwise two correct reruns differ byte-for-byte solely by timestamp.

## 8. What should and should not vary

### 8.1 Must vary with realization seed

- traffic demand and resulting vehicle spatiotemporal trajectories;
- research-vehicle and SV/TV role selection, because current design explicitly samples these roles by seed;
- task arrivals;
- requested service IDs under the same fixed Zipf process parameters;
- task workload jitter;
- formal observation noise and stochastic prediction residual/scenario draws;
- TV-local-cache assignment when roles/TV IDs vary.

### 8.2 Must remain fixed

- service count and the frozen service catalog/distribution envelope;
- CPU, bandwidth, cache-capacity policy/absolute baseline, UAV physical parameters;
- reward weights and PPO hyperparameters;
- Matching and KKT algorithms/parameters;
- Zipf `zeta`, arrival probability, epoch length, jitter envelopes;
- SUMO map and traffic-generation statistical parameters;
- calibration vehicle residual bank/Q/R;
- popularity GRU architecture, training data protocol, weights, and checkpoint hash.

## 9. Exact next files to change

No source file is changed by this report. A follow-up implementation should be limited to the path/seed/generation contract in these files:

1. `system_config.toml` and `system_config_ppo_primitive_v6_horizon_grouped_separate_optimization_popart_actorlr1e4.toml`
   - add/consume a versioned `realization_seed` contract and explicit shared/calibration/realization paths;
   - separate formal observation seed from calibration observation seed;
   - add `paths.popularity_predictions` and move the GRU path to calibration/shared storage;
   - do not change physical, reward, PPO, Matching, or KKT values.
2. `scripts/regenerate_artifacts.py`
   - accept split/realization seed/output root;
   - derive named seeds and write one realization manifest;
   - include `traffic_seed` and every upstream hash in signatures;
   - stage the complete bundle before atomic publication;
   - run GRU inference only, never model training, for train/test realization generation;
   - remove the hard-coded metadata path and unify the TV-cache seed source.
3. `src/scenario/sumo_scenario.py`
   - accept and honor `--config`;
   - consume separate traffic, SUMO-dynamics, research-role, and service-role seeds;
   - require seed-owned managed demand for realization generation instead of accepting unrelated existing routes implicitly;
   - write seed/upstream provenance beside the realization output.
4. `src/scenario/task_scenario.py`
   - load the frozen shared catalog rather than regenerate it per realization;
   - consume named task/cache streams;
   - replace the role-seed-equals-task-seed check with realization-manifest/hash consistency checks;
   - keep popularity prehistory and formal tasks in one continuous causal process.
5. `src/scenario/popularity_prediction.py`
   - split calibration-time `train` from realization-time `predict`;
   - in realization mode load a frozen checkpoint and write only realization predictions/diagnostics;
   - keep canonical metrics outside the frozen model checkpoint.
6. `src/scenario/scenario_generator.py`
   - consume the frozen GRU/residual bank plus realization-specific inputs;
   - use a realization-derived formal observation-noise stream rather than the calibration bank's observation seed;
   - replace literal stream tag `777` with the named derivation;
   - validate all realization/shared/calibration manifest hashes before writing output.
7. `src/common/project_config.py`
   - validate the added path keys and realization/shared/calibration provenance contract.
8. `src/ppo_simple/trainer.py` and `src/ppo_simple/evaluator.py`
   - replace the hard-coded `data/generated/popularity_predictions.csv` with `paths.popularity_predictions`;
   - no PPO algorithm change is needed.

Standalone diagnostic entrypoints in `src/decision/slot_environment.py` and `src/decision/task_allocation.py` also contain hard-coded `data/generated` paths. They are not on the audited V6 trainer/evaluator production path, but should use configured paths if they are expected to operate on arbitrary realization bundles.

## 10. Files that do not need algorithm changes

The following do not require mathematical or algorithmic modification for multi-realization support:

- `src/ppo_simple/policy.py`
- `src/ppo_simple/rollout.py`
- `src/ppo_simple/primitive_control.py`
- `src/ppo_simple/reporting.py`
- `src/ppo_simple/plotting.py`
- PPO loss, GAE, PopArt, actor/critic architecture, optimizer, learning rates, gradient clipping
- `src/decision/cache_configuration.py` cache optimization mathematics
- `src/decision/uav_configuration.py` UAV optimization mathematics
- `src/decision/task_allocation.py` Matching/KKT/task-allocation mathematics
- `src/common/system_model.py`
- reward, state, action, cache/UAV decision semantics
- `src/scenario/node_placement.py` placement algorithm (only its output path/provenance orchestration may change)
- `src/scenario/vehicle_prediction.py` calibration mathematics; the current independent-calibration guards should be retained

The required work is orchestration, namespaced RNG derivation, path resolution, provenance, and atomic artifact publication—not a PPO or system-model change.

## 11. Static audit status

- Baseline inspected: `main` at `e62f899fb7055f973585d04e4b3626cf494dc9b1` with a clean worktree before this report.
- Current TOML seed values were parsed statically.
- Calibration role/demand metadata and residual-bank metadata were read; their hashes identify independent calibration sources.
- No test-origin rows, formal test realization, or PPO run data were read.
- No source or existing artifact was modified.

