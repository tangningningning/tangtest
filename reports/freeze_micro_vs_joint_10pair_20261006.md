# Freeze Micro vs Joint: 10 Paired Episodes

Diagnosis: **INCONCLUSIVE**

Descriptive paired training diagnosis only; no deterministic evaluation or algorithm changes.

## 1. Experimental setup

- Joint/Freeze episodes: 10 / 10; base seed 27005; deterministic action sampling false.
- One Joint `episode_20016` rollout attempt was discarded before any update due to a digest-key alias; accepted training remains 10 episodes per branch (21 full collection attempts overall).
- Source checkpoint SHA256: `e056024d963fdfd1f99eb999462a2ab94a2ab7de0d064ea4f09782b4664fbaa6`; completed episodes at start: 15.

## 2. Starting checkpoint equality

- Initial policy, optimizer state, and counters identical: True / True / True.

## 3. Episode list

- Training seeds: `20016, 20017, 20018, 20019, 20020, 20021, 20022, 20023, 20024, 20025`. All are existing children of the training pool; final test seeds were not used.

## 4. Paired Return table

| Episode | Joint Return | Freeze Return | Freeze−Joint |
|---:|---:|---:|---:|
| 20016 | -550.152018 | -550.152018 | 0.000000 |
| 20017 | 904.881029 | 1696.695105 | 791.814076 |
| 20018 | 668.891902 | 2060.356923 | 1391.465021 |
| 20019 | 590.141401 | 875.218930 | 285.077530 |
| 20020 | 2159.997368 | 1679.988528 | -480.008840 |
| 20021 | 2750.288169 | 2508.351611 | -241.936558 |
| 20022 | 2324.292496 | 467.586131 | -1856.706365 |
| 20023 | 1465.829560 | 469.345035 | -996.484525 |
| 20024 | 1385.521002 | 1816.179462 | 430.658460 |
| 20025 | 2030.346869 | 2833.653501 | 803.306633 |

## 5. Return stability

- Joint mean/std/median/range: 1373.003778 / 942.783583 / 1425.675281 / 3300.440187.
- Freeze mean/std/median/range: 1385.722321 / 993.552598 / 1688.341817 / 3383.805519.
- Return std ratio: 1.0538501271139034; mean absolute return change ratio: 1.6559629266694529.

## 6. Macro Advantage

- Raw advantage std episode-series mean/std, Joint: 142.299797 / 54.283908; Freeze: 140.647249 / 48.965180.

## 7. Macro Actor update

- Candidate KL mean/std Joint: 0.00032297483390002213 / 0.00029230049441620476; Freeze: 0.0004298564440242666 / 0.0002656880128089313.
- Relative Actor parameter delta mean/std Joint: 0.009955019884032513 / 0.0019920346087154424; Freeze: 0.010324512301040473 / 0.0018037962144899328.

## 8. Macro Critic

- Explained Variance mean Joint/Freeze: 0.9930300181583567 / 0.9929082929722389; monitoring only.

## 9. Cache/Duration behavior

- Mean P_KEEP Joint/Freeze: 0.6163983636928116 / 0.60441603634468.
- Duration 1/3/5 mean rates Joint: 0.3515738118904979 / 0.3373723993119522 / 0.3110537887975499; Freeze: 0.34048827726795844 / 0.3319117897705517 / 0.3275999329614898.

## 10. Joint Micro drift

- Cumulative Actor distance from initial: 1.5696000102922885.

## 11. Freeze integrity

- Micro parameters unchanged every episode: True; Micro optimizer step delta total: 0.

## 12. System-level decomposition

- Per-episode task count, workload summary, cache/movement cost, delay, energy, and task utility are in the JSON report.

## 13. Paired differences

- Mean paired Return difference (Freeze−Joint): 12.718543142049771; Freeze greater / Joint greater: 5 / 4.

## 14. Diagnosis

- **INCONCLUSIVE**.
- Numeric decision criteria and descriptive Micro-drift correlations are recorded in the JSON report; correlations are not causal evidence.

## 15. Limitations

- One extra Joint `episode_20016` collection was discarded before optimizer update due to the report-harness digest-key alias; its three reference digests matched after correction.
- Only 10 paired training realizations from one training pool and one initial checkpoint; results are descriptive, not a generalization or significance claim.
- The Freeze treatment removes Micro Actor/Critic optimizer updates while preserving stochastic Micro actions; it does not establish that Micro learning is the sole source of Macro instability.
- Episode-to-episode correlations use 9 lag-aligned observations or 10 same-episode observations and are descriptive only.
- No deterministic evaluation was run, and no algorithm or hyperparameter was changed.

## 16. Next step

停止，等待确认是否扩展到 20 对 episode。
