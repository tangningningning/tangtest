# Critic Repair Progress

## Current Status

- Result: **PARTIAL**. Actual-state evaluation and the fixed UAV_1 branch evaluation completed using the saved rollout; no rollout was collected.
- Actual-state diagnostic reproduction: **PASS**. Observations, rewards, and dones matched their fixed semantic hashes; actions also matched. The actual-state gate checks only those required arrays plus shape, finite-value, and terminal structure. Neither NPZ container SHA nor transition pickle SHA gates Critic inference.
- Diagnostic identity matches episode seed `12184115623283823756`, rollout RNG seed `10160482263500971619`, Actor digest `3e847420e37cd5068561b316ead29ea59214d417e4b9809489c4b4a459829ef9`, and config SHA256 `131a5ba050cdc5c454800b1e814187ec4053a83b8dacfb6b881650e2d69c6805`.
- Three existing Critic checkpoints reloaded successfully; the Coverage checkpoint digest is `fb32158378cd19271d3fb09544e641b9d67651d470252eb4b2e8e885b971f7d6`. Historical GAE and MC-1 metrics reproduced exactly.
- Transition pickle SHA differs (`b26700b8e57d105610a6cd97bd7e74ab5793cffea43ef9796a9f442a6d2752d4` vs `99cb7a5a1efb0979f21b1f17bbb8778483c39fe19c53bf889b54142c58062e93`), and whole-NPZ SHA differs (`0d9f992fbee131b23fa8d3f9227380795a82ad2a72628baf5e02370dd60ec778` vs prior `3fa12013dbd9d17df6851a98fc1462db599d46d5bc7c0c0db10128c987b39c20`). The fixed legacy business fingerprint and both historical branch next-observation hashes match; classification: `PICKLE_SERIALIZATION_MISMATCH_ONLY` for the evaluated branch semantics.
- No Critic training, TRAIN rollout collection, target generation, Epoch 1, production change, commit, or push occurred.
- Next stage: `CRITIC_TARGET_DATA_TRADEOFF_DECISION`.

## Target Alignment Result

GAE vs MC: MAE `531.1365374755859`, RMSE `680.8467091369258`, EV `0.03706454168143203`, Pearson `0.4774469992360514`, Spearman `0.37266325181390897`.

MC-1 vs MC: MAE `480.6999598757426`, RMSE `527.0582962289436`, EV `0.000468880618351597`, Pearson `0.08290692477484993`, Spearman `0.26819185771265025`. Both baseline checkpoints reproduced their historical MAE/RMSE/EV with zero recorded difference.

## Data Coverage Result

Coverage training remains the completed four-episode run: 1,200 unique MC targets and 40 Critic optimizer steps. Coverage checkpoint reload: **PASS**. On the same 300 observations and MC reward-to-go, Coverage metrics are MAE `584.5541013145447`, RMSE `641.5581968667669`, EV `0.005849390405342825`, Pearson `0.16766813401766334`, and Spearman `0.5016334629570369`.

Signed improvements use baseline error minus Coverage error for MAE/RMSE, and Coverage EV minus baseline EV:

- Versus MC-1: MAE `-103.85414143880212`, RMSE `-114.49990063782332`, EV `+0.005380509786991228`.
- Versus GAE: MAE `-53.41756383895881`, RMSE `+39.28851227015889`, EV `-0.031215151276089204`.
- `ACTUAL_STATE_COVERAGE_RESULT = PARTIAL`: Coverage improves RMSE over GAE and EV over MC-1, but MAE is worse than both and EV remains below GAE.

Diagnostic hashes recomputed from the saved NPZ using the project array-hash scheme:

| Array | Actual SHA256 | Expected SHA256 | Match |
| --- | --- | --- | --- |
| observations | `a7a7bcdd78ba06edcf83ccec4174b4f5a1fa8f0e49636916f830754a87b4458c` | `a7a7bcdd78ba06edcf83ccec4174b4f5a1fa8f0e49636916f830754a87b4458c` | YES |
| actions | `6215f89fcca090c8f0389c73bb00a59fc66206dccc450ed09ca259ef0c7a4826` | `6215f89fcca090c8f0389c73bb00a59fc66206dccc450ed09ca259ef0c7a4826` | YES |
| rewards | `cc8fa2acf019ae17fe6ac5536accaefc2841272007c20eccd8af013e3791ecae` | `cc8fa2acf019ae17fe6ac5536accaefc2841272007c20eccd8af013e3791ecae` | YES |
| dones | `2d5840178e2630906037acfe7636d495fe3986ab1b338f8edfde74a322a0380e` | `2d5840178e2630906037acfe7636d495fe3986ab1b338f8edfde74a322a0380e` | YES |
| transition_149_pickle | `b26700b8e57d105610a6cd97bd7e74ab5793cffea43ef9796a9f442a6d2752d4` | `99cb7a5a1efb0979f21b1f17bbb8778483c39fe19c53bf889b54142c58062e93` | NO |

## UAV_1 Witness

The legacy fingerprint helper hashes transition/origin/slot, action vector, previous and production-next Cache states, previous and next UAV positions, task identity/demand fields, actual vehicle positions, and config SHA using sorted-key canonical JSON. It is deterministic business data, but does not include the aligned next-step prediction inputs, so the runner now also computes `TRANSITION149_SEMANTIC_FINGERPRINT_V2` over both aligned steps, task/scenario/role/position data, prediction inputs, production action records, state arrays, branch context, and config.

`OLD_TRANSITION149_SEMANTIC_FINGERPRINT_V2 = 630e99ee74d887955a270ba22a4c4f854666f2ff43dd7038db08d859382e09e3` and `CURRENT_TRANSITION149_SEMANTIC_FINGERPRINT_V2 = 630e99ee74d887955a270ba22a4c4f854666f2ff43dd7038db08d859382e09e3`. The prior successful run removed its diagnostic NPZ; the old V2 envelope is reconstructed from its retained legacy fingerprint, matching fixed action-array hash, the same preserved TRAIN realization/aligned inputs, and saved branch-output hashes. The old pickle bytes were not used. `TRANSITION149_SEMANTIC_REPRODUCTION = PASS`; `TRANSITION149_MISMATCHED_FIELDS = []`.

Only the two current-slot branches were rebuilt (KEEP and Candidate); terminal continuations: `0`. Both next-observation hashes match the retained old witness: KEEP `59276cacbef7b42099d6d42439dcbe96864afedc0b1c7c670a599eab09712e5b`, Candidate `99beeefe83b75b9a9b3e5ed12bbc5769b77c0839070d3c1a2e67fab4880160c0`.

GAE `ΔV = +3.220560073852539`; MC-1 `ΔV = -0.010498046875`; Coverage `ΔV = +0.4283447265625`. Empirical MC values remain `-104.8574641123414` and `-380.0770139247179` (both NEGATIVE). `COVERAGE_UAV1_SIGN_AGREEMENT = NO`; `UAV1_COUNTERFACTUAL_RESULT = NOT_SUPPORTED`.

## Decision

**PARTIAL** — actual-state metrics show mixed gains; the fixed branch semantics reproduced, but Coverage's UAV_1 `ΔV` is positive against negative empirical MC. `NEXT_STAGE = CRITIC_TARGET_DATA_TRADEOFF_DECISION`.

## Formal Training Status

- `COUNTERFACTUAL_CRITIC_RELIABILITY = CONCERN`
- `CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION = CANDIDATE_Q_INTEGRITY_CONCERN`
- `SURROGATE_REPAIR_BLOCKED = YES`
- Formal PPO update: **NO**; Epoch 1: **NO**; ready for formal training: **NO**.
- Production source, reward, matching, KKT, GAE, and Critic training logic were not changed.

## History

- `TRACEABLE_CANDIDATE_Q_INTEGRITY_GATE`: **INCONCLUSIVE** after the TRAIN-only warmup and exact evaluation completed but final result assembly failed.
- `UAV0_TRANSITION149_REWARD_ROOT_CAUSE_GATE`: **REWARD_OUTLIER_VALID_TASK_EXECUTION**; actual task utility was the largest component; Candidate-Q remained inconclusive.
- `COUNTERFACTUAL_CRITIC_RELIABILITY_GATE`: reliability **CONCERN** and Candidate-Q **CANDIDATE_Q_INTEGRITY_CONCERN**; fixed UAV_1 predicted ΔV had the wrong sign against negative MC continuations.
- `COUNTERFACTUAL_CRITIC_RELIABILITY_PERSISTENCE`: same 12 continuation records persisted successfully; reliability remained **CONCERN**.
- `COUNTERFACTUAL_CRITIC_REPAIR_DESIGN`: root cause classified as **MIXED_TARGET_AND_DATA_CONCERN**; design-only, no production change.
- `CRITIC_TARGET_ALIGNMENT_MINIMAL_EXPERIMENT`: initial attempt stopped before training; authorized follow-up trained both 40-step arms and was **PARTIAL**, preserving historical GAE/MC metrics and UAV_1 values.
- `CRITIC_DATA_COVERAGE_MINIMAL_EXPERIMENT`: **INCONCLUSIVE** at the diagnostic rollout SHA gate; four TRAIN episodes, 1,200 targets, 40 Critic steps, and the coverage checkpoint were retained.
- `SECOND_STAGE_JOINT_REPAIR_GATE`: Candidate-Q integrity remained **INCONCLUSIVE** in the cited traceable gate; formal PPO update and readiness remained **NO**.
- `ONE_EPOCH_TRAINING_GATE`: Epoch 1 was **NOT EXECUTED**; formal-training readiness remained **NO**.
- `CRITIC_DATA_COVERAGE_EXPERIMENT_FOLLOWUP`: **INCONCLUSIVE** before inference due to its whole-NPZ/pickle SHA gate; no rollout was recollected.
- `DIAGNOSTIC_ROLLOUT_DETERMINISM_FIX`: **PARTIAL**; actual-state gate and baseline reproduction passed, transition branch outputs reproduced, Coverage was mixed on state metrics and had the wrong UAV_1 direction.
