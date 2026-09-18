# Recovery R1 UAV candidate TD alignment audit

- `STAGE`: `RECOVERY_R1_UAV_CANDIDATE_TD_ALIGNMENT_AUDIT`
- `TRAINING_EXECUTED`: `NO`; `OPTIMIZER_STEP_EXECUTED`: `NO`; `SOURCE_MODIFIED`: `NO`
- `STATE_COUNT`: `4`; `TOTAL_CANDIDATES`: `216` (`212` MOVE candidates + `4` STAY baselines)
- All 216 H30 source values are read directly from the preceding H30 audit JSON. They are not recomputed here.

## Candidate ranking and sign separation

| state | MOVE candidates | current policy H30 rho | one-step TD H30 rho | one-step TD top1 H30 rank | top5 hit | good/bad | good A>0 rate | bad A>0 rate | n=3 rho | n=3 rank | n=3 top5 | n=5 rho | n=5 rank | n=5 top5 |
|:---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|
| 20100/80/UAV_0 | 80 | 0.0089 | 0.2471 | 23 | NO | 21/59 | 0.714 | 0.441 | 0.2532 | 23 | NO | 0.3006 | 23 | NO |
| 20100/80/UAV_1 | 44 | 0.3393 | -0.4472 | 43 | NO | 13/31 | 0.385 | 0.613 | -0.9484 | 44 | NO | -0.8994 | 44 | NO |
| 20300/200/UAV_0 | 44 | 0.3425 | 0.8947 | 1 | YES | 16/28 | 1.000 | 0.250 | 0.9600 | 1 | YES | 0.9875 | 1 | YES |
| 20300/200/UAV_1 | 44 | -0.5465 | 0.5178 | 18 | NO | 5/39 | 1.000 | 0.487 | 0.3425 | 15 | NO | 0.3626 | 15 | NO |

## Summary

- `CURRENT_POLICY_H30_SPEARMAN_BY_STATE`: `{'20100/80/UAV_0': 0.00890051246579432, '20100/80/UAV_1': 0.3393260314873668, '20300/200/UAV_0': 0.34250059785340947, '20300/200/UAV_1': -0.5465054971947223}`
- `ONE_STEP_TD_H30_SPEARMAN_BY_STATE`: `{'20100/80/UAV_0': 0.2470698546647914, '20100/80/UAV_1': -0.4472163495419309, '20300/200/UAV_0': 0.8947145877378435, '20300/200/UAV_1': 0.517829457364341}`
- Mean H30 Spearman: current policy `0.036055`; one-step TD `0.303099`; descriptive delta `+0.267044`.
- `ONE_STEP_TD_TOP1_H30_RANK_BY_STATE`: `{'20100/80/UAV_0': 23, '20100/80/UAV_1': 43, '20300/200/UAV_0': 1, '20300/200/UAV_1': 18}`; `ONE_STEP_TD_TOP5_HIT_COUNT`: `1`; top1 exact matches: `1`
- `GOOD_CANDIDATE_COUNT`: `55`; `BAD_CANDIDATE_COUNT`: `157`
- `GOOD_CANDIDATE_POSITIVE_ADV_RATE`: `0.7454545454545455`; `BAD_CANDIDATE_POSITIVE_ADV_RATE`: `0.45222929936305734`
- n=3 mean H30 Spearman: `0.15180417659075296`; top5 hits: `1`. n=5 mean H30 Spearman: `0.18782563640526545`; top5 hits: `1`.
- `UAV_CANDIDATE_ONE_STEP_TD_CREDIT`: `NOT_SUPPORTED`; `UAV_CANDIDATE_MULTISTEP_TD_CREDIT`: `NOT_SUPPORTED`; `SHORT_HORIZON_CANDIDATE_TD_INSUFFICIENT`: `SUPPORTED`.
- The one-step candidate baseline is `B_candidate = sum_c pi(c|MOVE,s) * QTD_c`; advantages are `QTD_c - B_candidate` with no centered-mean subtraction.
- Ranking conclusions use the actual H30 return as truth. The current policy probability ranking is a descriptive comparator; its near-uniformity was established in the preceding audit.

## Integrity

- Frozen checkpoint: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v10d_decoupled_encoder_branch_kl_6e4_formal\latest_checkpoint.pt`
- Frozen pool: `C:\Users\tangning\Desktop\code\adaptive_cache_mec\runs\ppo_v6_popart_actor1e4_lazy_episode_pool_20260911`
- `FINAL_TEST_SEEDS_USED`: `NO`; `EPISODE_REGENERATION`: `NO`; `TEMP_FILES_REMOVED`: `YES`; `GIT_DIFF_CHECK`: `PASS`; `PUSHED`: `NO`.
- The n=3 and n=5 supplement uses the frozen checkpoint critic only for the n-step bootstrap; no critic update or optimizer step is executed.
