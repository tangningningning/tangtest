# UAV Movement Weight Literature and Calibration Audit

**Stage:** `RECOVERY_R2_UAV_MOVEMENT_WEIGHT_LITERATURE_AND_CALIBRATION`  
**Repository:** `adaptive_cache_mec`  
**Scope:** provenance diagnosis and bounded development calibration only.  
**Production configuration changed:** `NO`  
**Training, optimizer updates, formal evaluation, episode generation, and SUMO:** `NOT RUN`

## Executive decision

The repository does not contain a traceable literature value or an independent calibration record for `objective.uav_movement_cost_weight = 10.0`. The value is therefore classified as:

```text
W_MOVE_PROVENANCE = UNKNOWN
W_MOVE_ROLE = TRADEOFF_WEIGHT
NO_DIRECT_LITERATURE_VALUE_FOR_W_MOVE = YES
```

The physical movement coefficient is a different kind of parameter. `xi_mov = 15.0 J/m` is reproducibly derived from a published speed-dependent rotary-wing propulsion model at one representative speed, then linearized and rounded. It is `LITERATURE_DERIVED`, but not a direct literature constant and not `LITERATURE_CALIBRATED`.

For development sensitivity testing only, the bounded replay supports:

```text
PRIMARY_W_MOVE       = 0.50
LOW_SENSITIVITY_W_MOVE  = 0.25
HIGH_SENSITIVITY_W_MOVE = 1.00
```

These are calibration candidates, not a production-config change and not literature-derived numeric recommendations. `w=0.50` keeps movement materially active while retaining positive immediate candidates in 13 of the 20 audited states. `w=10.0` leaves no positive immediate movement candidate in this frozen audit.

## Method and audit boundary

The replay used the existing epoch-7 checkpoint and the frozen `episode_20001` realization already used by the preceding movement-cost audits:

| Item | Value |
|---|---:|
| checkpoint | epoch 7 |
| frozen states | 20 |
| legal move candidates | 1,600 |
| current configured `w_move` | 10.0 |
| `movement_fixed_cost` (`eta_q`) | 0.00176 |
| `lambda_movement` | 1.76e-5 |
| `xi_mov` | 15.0 J/m |
| exact decomposition residual, maximum | 8.19e-16 |
| movement linearity residual, maximum | 1.39e-17 |

One bounded H1 exact enumeration captured the full branch components for the same 20 states. The weight sweep was then performed offline over those captured components. No training or optimizer step was executed, and no additional horizon or state set was introduced.

For a candidate branch `c`, the audited immediate movement-adjusted value is:

```text
DeltaR_c(w) = PRE_COST_BENEFIT_c - w * BASE_MOVE_TERM_c
BASE_MOVE_TERM_c = z_c * eta_q + lambda_movement * xi_mov * distance_c
```

This is the repository's implemented identity:

```text
C_move = w_move * (z_actual * eta_q + lambda_movement * xi_mov * distance)
```

The scale ratio reported below is explicitly:

```text
mean(abs(w * BASE_MOVE_TERM_c)) / mean(abs(PRE_COST_BENEFIT_c))
```

over all 1,600 audited legal branches. It is a diagnostic scale ratio, not a fitted statistical estimator.

## `REPOSITORY_EVIDENCE`

### Implemented semantics

`src/common/system_model.py:309-364` computes the actual Euclidean movement distance, derives `actual_z` from the distance tolerance, checks the maximum-speed feasibility constraint, computes `movement_energy = xi_mov * effective_distance`, and returns:

```text
movement_cost = movement_cost_weight * (actual_z * movement_fixed_cost
                                        + lambda_movement * movement_energy)
```

`src/decision/uav_configuration.py:101-110` reads the four values from the configuration, and `src/decision/uav_configuration.py:524-552` passes the configured outer weight through unchanged. `src/decision/slot_environment.py:451-455` subtracts the resulting movement cost from the system utility. The current code therefore has one explicit outer tradeoff multiplier; it does not create a second physical movement-energy model.

### Parameter role and provenance classification

| Parameter | Current value | Semantic role | Repository status/evidence | Provenance decision |
|---|---:|---|---|---|
| `nodes.uav.movement.movement_energy_coefficient_j_per_m` (`xi_mov`) | 15.0 J/m | physical-energy approximation | Config status is `LITERATURE_DERIVED`; comments and `reports/physical_cost_coefficient_evidence_closure.md:119-137` document the Li/Zeng rotary-wing derivation at `V_ref=15 m/s`, linearization, and rounding | `XI_MOV_PROVENANCE = LITERATURE_DERIVED`; not a direct tabulated constant and not calibrated to a measured flight trace |
| `nodes.uav.movement.movement_fixed_cost` (`eta_q`) | 0.00176 | objective fixed action penalty applied only when a real move occurs | Config status is `EXPERIMENT_DESIGN`; the repository has no independent reconfiguration-cost measurement or reproducible physical calibration record | `ETA_Q_PROVENANCE = HEURISTIC_TUNING` (repository label: `EXPERIMENT_DESIGN`) |
| `objective.lambda_movement` | 1.76e-5 | objective normalization / mapping from movement-energy-like joules to objective cost | Config status is `EXPERIMENT_DESIGN`; `reports/lambda_objective_weight_design_closure.md` explicitly treats it as a scale mapping, not a physical constant | `LAMBDA_MOVEMENT_PROVENANCE = NORMALIZATION_CHOICE` (repository label: `EXPERIMENT_DESIGN`) |
| `objective.uav_movement_cost_weight` (`w_move`) | 10.0 | outer dimensionless tradeoff weight | Config line 406 supplies the number, but the status map has no explicit entry and repository search found no README, report, paper, or calibration record tying `10.0` to a numeric source | `W_MOVE_PROVENANCE = UNKNOWN`; role is `TRADEOFF_WEIGHT` |

The status map in `configs/proposed/system_config_ppo_recovery_r2_6e4.toml:632-649` explicitly lists `xi_mov`, `eta_q`, and `lambda_movement`, but not `uav_movement_cost_weight`. Git-history search found the value carried through configuration revisions, but no commit message or source record that identifies `10.0` as a literature value or calibrated value. Historical persistence is not provenance.

### Repository scale evidence

The preceding movement-envelope gate found 797 candidates with positive pre-cost benefit. At the current `w_move=10.0`, all 797 are suppressed, and no audited state has a positive immediate movement-adjusted candidate. That result is consistent with the branch-level sweep below; it is not evidence that `10.0` is a physical or literature-calibrated value.

## `LITERATURE_EVIDENCE`

### Physical movement-energy model

The rotary-wing model used by the repository is consistent with the model family in [Zeng, Xu, and Zhang, *Energy Minimization for Wireless Communication With Rotary-Wing UAV*](https://arxiv.org/abs/1804.02238) and its IEEE publication ([DOI: 10.1109/TWC.2019.2902559](https://doi.org/10.1109/TWC.2019.2902559)). The model contains blade-profile, induced, and parasite-power components; propulsion power depends on speed, and segment energy depends on both distance and travel time. The literature therefore supports a speed/trajectory-dependent physical model, not a universal speed-independent `15 J/m` constant.

The repository's `xi_mov=15.0 J/m` is a transparent linear equivalent: the documented reference parameters give `P(15 m/s)=220.960140363616 W`, so `P(15)/15=14.730676024241 J/m`, rounded to `15.0 J/m`. This is a defensible implementation approximation, but the representative speed, linearization, and rounding are project choices. It should not be described as a directly measured or directly tabulated literature constant.

UAV-MEC literature such as [UAV-Enabled Mobile Edge Computing: Offloading Optimization and Trajectory Design](https://doi.org/10.1109/ICC.2018.8422277) and [Stochastic Computation Offloading and Trajectory Scheduling for UAV-Assisted MEC](https://doi.org/10.1109/JIOT.2018.2890133) jointly considers trajectory, offloading, resource, and energy objectives. These formulations do not provide a universal numeric outer multiplier that can be transplanted into this repository's system-utility scale.

### Objective normalization and tradeoff weights

The weighted-objective literature supports separating three decisions that are currently easy to conflate:

1. a physical model or physical equivalent (`xi_mov`);
2. a mapping/normalization into the repository's objective units (`lambda_movement` and `eta_q`); and
3. a preference/tradeoff weight (`w_move`).

[Ishibuchi, Doi, and Nojima, *On the effect of normalization in MOEA/D for multi-objective and many-objective optimization*](https://doi.org/10.1007/s40747-017-0061-9) documents that raw objective-range differences can bias scalarization and that normalization affects the resulting search behavior. [Marler and Arora, *Survey of multi-objective optimization methods for engineering*](https://doi.org/10.1007/s00158-003-0368-6) emphasizes that weights encode preference/tradeoff decisions and that there is no universal weight independent of the application and objective scaling. These sources support calibrating `w_move` against the repository's actual branch-value envelope; they do not supply a direct value for `10.0`, `0.5`, or any other system-equivalent number.

Therefore:

```text
NO_DIRECT_LITERATURE_VALUE_FOR_W_MOVE = YES
```

## `CALIBRATION_EVIDENCE`

The following sweep uses the same 1,600 exact branches. `SURVIVING_POSITIVE_COUNT` is counted among the 797 branches whose pre-cost benefit is positive; because the movement term is non-negative, it equals the positive immediate count here. `SUPPRESSED_FRACTION` is one minus the surviving fraction over those 797 branches. State coverage counts states with at least one `DeltaR_c(w) > 0`; the UAV columns count those states by UAV branch.

| `w_move` | positive immediate branches | surviving fraction of 797 | suppressed fraction | states with a positive branch | UAV0 / UAV1 state coverage | mean abs movement contribution | movement/pre-cost scale ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 585 | 73.4003% | 26.5997% | 18 / 20 | 9 / 9 | 0.000403700 | 0.216722 |
| 0.25 | 400 | 50.1882% | 49.8118% | 16 / 20 | 9 / 7 | 0.001009250 | 0.541806 |
| 0.50 | 221 | 27.7290% | 72.2710% | 13 / 20 | 7 / 6 | 0.002018500 | 1.083612 |
| 1.00 | 64 | 8.0301% | 91.9699% | 4 / 20 | 0 / 4 | 0.004037000 | 2.167224 |
| 2.00 | 32 | 4.0151% | 95.9849% | 2 / 20 | 0 / 2 | 0.008074000 | 4.334449 |
| 10.00 | 0 | 0.0000% | 100.0000% | 0 / 20 | 0 / 0 | 0.040370000 | 21.672244 |

### Calibration interpretation

#### Primary development candidate: `w_move = 0.50`

At `w=0.50`, the movement contribution is approximately the same scale as the mean absolute pre-cost branch benefit (`ratio=1.083612`). It retains 221 positive immediate branches, covers 13 of 20 states, and preserves positive movement opportunities for both UAVs. This is the most balanced point in the bounded grid: movement is neither a negligible perturbation nor an almost absolute veto.

#### Lower-sensitivity candidate: `w_move = 0.25`

At `w=0.25`, the movement contribution is about 0.54 of the mean absolute pre-cost scale. It leaves 400 positive branches and 16 of 20 states covered. This is appropriate as a lower-sensitivity comparison, but it intentionally allows more movement than the primary candidate.

#### Higher-sensitivity candidate: `w_move = 1.00`

At `w=1.00`, the movement contribution is about 2.17 times the mean absolute pre-cost scale. Only 64 branches survive, across 4 states, all on the UAV1 side of this frozen realization. It is useful as a high-penalty sensitivity point, but it is not recommended as the primary development value because it begins to collapse state coverage.

#### Rejected extremes

`w=0.10` is relatively permissive: 73.4% of positive pre-cost branches survive and the movement term is only 0.22 of the mean absolute pre-cost scale. `w=2.0` and the current `w=10.0` are increasingly suppressive; `w=10.0` eliminates every positive immediate movement branch in this audit. The grid does not support treating `10.0` as a neutral default.

## Required conclusions

```text
ETA_Q_ROLE                    = OBJECTIVE_FIXED_ACTION_PENALTY
ETA_Q_PROVENANCE              = HEURISTIC_TUNING
LAMBDA_MOVEMENT_ROLE          = OBJECTIVE_NORMALIZATION_PARAMETER
LAMBDA_MOVEMENT_PROVENANCE    = NORMALIZATION_CHOICE
XI_MOV_ROLE                   = PHYSICAL_APPROXIMATION
XI_MOV_PROVENANCE             = LITERATURE_DERIVED
W_MOVE_ROLE                   = TRADEOFF_WEIGHT
W_MOVE_PROVENANCE             = UNKNOWN
NO_DIRECT_LITERATURE_VALUE_FOR_W_MOVE = YES
PRIMARY_W_MOVE                = 0.50
LOW_SENSITIVITY_W_MOVE        = 0.25
HIGH_SENSITIVITY_W_MOVE       = 1.00
CALIBRATION_STATUS            = DEVELOPMENT_ONLY
PRODUCTION_CONFIG_CHANGED     = NO
```

No value was written back to `configs/proposed/system_config_ppo_recovery_r2_6e4.toml`. The recommended values are evidence-backed development points for a future explicitly authorized sensitivity experiment, not a claim of physical truth or literature calibration.
