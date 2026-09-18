# Recovery R1 UAV MOVE semantics and H30 candidate audit

- `STAGE`: `RECOVERY_R1_UAV_MOVE_SEMANTICS_AND_H30_CANDIDATE_AUDIT`
- `TRAINING_EXECUTED`: `NO`; `OPTIMIZER_STEP_EXECUTED`: `NO`
- `STATE_COUNT`: `4`; H30 horizon: `30` actual environment rewards, no critic bootstrap
- `OPERATION_H30_TRUTH_DEFINITION`: `POLICY_CONDITIONAL_EXPECTATION`
- `ONE_STEP_TD_OPERATION_H30_MATCH_COUNT`: `4/4`

## State-level comparison

| seed | origin | UAV | MOVE candidates | H30 STAY | production MOVE | policy-top1 MOVE | policy-expected MOVE | best-H30 MOVE | case | one-step TD | H30 truth | TD match | Spearman | candidate H30 range | production H30 rank | top1 H30 rank | best policy-prob rank |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| 20100 | 80 | UAV_0 | 80 | 266.549885 | 266.504322 | 266.504322 | 266.517394 | 266.609601 | B | STAY | STAY | YES | 0.008901 | 0.231943 | 52 | 52 | 74 |
| 20100 | 80 | UAV_1 | 44 | 266.460914 | 266.504322 | 266.504322 | 266.409974 | 266.554347 | B | STAY | STAY | YES | 0.339326 | 0.310210 | 5 | 5 | 36 |
| 20300 | 200 | UAV_0 | 44 | 220.160565 | 220.275015 | 220.275015 | 220.126505 | 220.400057 | B | STAY | STAY | YES | 0.342501 | 0.465214 | 5 | 5 | 36 |
| 20300 | 200 | UAV_1 | 44 | 220.374620 | 220.275015 | 220.275015 | 220.342026 | 220.385558 | B | STAY | STAY | YES | -0.546505 | 0.191301 | 41 | 41 | 17 |

## Aggregate conclusions

- `EXPECTED_MOVE_BEATS_STAY_COUNT`: `0`; `BEST_MOVE_BEATS_STAY_COUNT`: `4`; `GOOD_MOVE_CANDIDATES_EXIST_COUNT`: `4`.
- `MEAN_H30_CANDIDATE_POLICY_SPEARMAN`: `0.03605541115296207`; `NEAR_UNIFORM_POLICY_STATE_COUNT`: `4`.
- `H30_CANDIDATE_VALUE_VARIATION_PRESENT`: `YES`; `H30_CANDIDATE_POLICY_RANKING_LEARNED`: `NOT_PROVEN_NEAR_UNIFORM`; `LONG_HORIZON_CANDIDATE_CREDIT_PROBLEM`: `SUPPORTED`.
- Policy probability and H30 return are ranked with rank 1 as highest. Near-uniform policy probabilities are explicitly flagged; their Spearman values are descriptive only and are not treated as evidence of learned long-horizon ranking.
- The candidate-head conclusion is conditional on MOVE. The operation truth is the policy conditional expectation over all legal MOVE candidates, not the production-selected candidate.

## Reproduction contract

- checkpoint: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v10d_decoupled_encoder_branch_kl_6e4_formal\latest_checkpoint.pt`
- frozen pool: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v6_popart_actor1e4_lazy_episode_pool_20260911`
- Cache and the other UAV were fixed to the production transition; each branch used direct policy primitive configuration followed by the real Matching/KKT environment reward.
- The first slot forced STAY or one legal MOVE candidate. Slots t+1 through t+29 used the frozen checkpoint with deterministic argmax.

## Integrity

- `SOURCE_MODIFIED`: `NO`; `SYSTEM_MODEL_CHANGED`: `NO`; `REWARD_CHANGED`: `NO`.
- `FINAL_TEST_SEEDS_USED`: `NO`; `TEMP_FILES_REMOVED`: `YES`; `GIT_DIFF_CHECK`: `PASS`; `PUSHED`: `NO`.

The JSON report contains every legal MOVE candidate's policy probability, actual H30 return, H30 rank, and coordinate witness.
