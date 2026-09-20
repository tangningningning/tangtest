# Cache Q single-realization offline fit

## A. Dataset integrity

- Dataset schema valid: `True`
- Exact slot files: `14/14`
- Candidate alignment: `True`
- Random sanity samples: `{"train": [{"slot": 1, "node_index": 0, "candidate_count": 8, "alignment_verified": true}, {"slot": 5, "node_index": 0, "candidate_count": 10, "alignment_verified": true}], "validation": [{"slot": 9, "node_index": 0, "candidate_count": 9, "alignment_verified": true}, {"slot": 10, "node_index": 0, "candidate_count": 8, "alignment_verified": true}], "held_out": [{"slot": 15, "node_index": 0, "candidate_count": 12, "alignment_verified": true}, {"slot": 25, "node_index": 0, "candidate_count": 14, "alignment_verified": true}]}`

## B. Residual target distribution (train slots 1-8 only)

```json
{
  "mean": -25.585215168861424,
  "std": 49.7155473679296,
  "min": -155.53724670410156,
  "p1": -120.9492308807373,
  "p10": -120.0625213623047,
  "median": 3.0638864040374756,
  "p90": 10.19877576828003,
  "p99": 11.61630870819092,
  "max": 14.047297477722168
}
```

Target normalization: `True`.

## C. Training fit

```json
{
  "action_sample_count": 1102,
  "ranking_group_count": 72,
  "mae": 30.912694377731285,
  "rmse": 40.80128615808918,
  "pearson": 0.3388189463337174,
  "spearman": 0.2010480502261102,
  "argmax_accuracy": 0.3472222222222222,
  "recall_at_1": 0.3472222222222222,
  "recall_at_3": 0.5277777777777778,
  "recall_at_5": 0.75,
  "exact_regret": {
    "mean": 6.702471226453781,
    "median": 0.07363319396972656,
    "p90": 7.599667906761168,
    "p95": 23.70569388866427,
    "max": 128.97318649291992
  },
  "normalized_regret": {
    "mean": 0.2706955243569687,
    "p95": 0.999999999992594,
    "max": 0.999999999999965
  }
}
```
Training MSE (normalized) changed from `0.966043` to `0.062583` across `52` epochs; the reported split metrics use the best validation checkpoint at epoch `2`.

## D. Validation

```json
{
  "action_sample_count": 284,
  "ranking_group_count": 18,
  "mae": 5.60168845907898,
  "rmse": 6.880671857101699,
  "pearson": 0.09982439357050477,
  "spearman": 0.09986150788653564,
  "argmax_accuracy": 0.16666666666666666,
  "recall_at_1": 0.16666666666666666,
  "recall_at_3": 0.2777777777777778,
  "recall_at_5": 0.4444444444444444,
  "exact_regret": {
    "mean": 2.581530385547214,
    "median": 0.15347760915756226,
    "p90": 8.068413424491883,
    "p95": 10.195015108585352,
    "max": 13.836514949798584
  },
  "normalized_regret": {
    "mean": 0.513404148874177,
    "p95": 0.9999999999953448,
    "max": 0.9999999999998953
  }
}
```

## E. Held-out generalization

```json
{
  "action_sample_count": 536,
  "ranking_group_count": 36,
  "mae": 12.662081501356576,
  "rmse": 15.599789331439728,
  "pearson": 0.28491305582546544,
  "spearman": 0.21558403092391512,
  "argmax_accuracy": 0.1111111111111111,
  "recall_at_1": 0.1111111111111111,
  "recall_at_3": 0.3611111111111111,
  "recall_at_5": 0.7222222222222222,
  "exact_regret": {
    "mean": 5.492055046682556,
    "median": 0.3281361013650894,
    "p90": 23.500676453113556,
    "p95": 24.69183075428009,
    "max": 26.993042945861816
  },
  "normalized_regret": {
    "mean": 0.4469796668610958,
    "p95": 0.9999999999985596,
    "max": 0.9999999999999538
  }
}
```

## F. Ranking quality

Ranking correlations and Recall@K are averaged over individual slot×node candidate groups; MAE/RMSE use the complete KEEP+UPDATE action vectors.

## G. Root-cause classification

- `Q_CRITIC_TRAIN_FIT = FAIL`
- `Q_CRITIC_GENERALIZATION = FAIL`
- `MSE_OBJECTIVE_RANKING_MISMATCH = NOT_SUPPORTED`
- `ROOT_CAUSE = MODEL_UNDERFIT`

## H. Recommended next step

Reuse the persisted dataset for the next diagnostic. No ranking-aware loss, architecture change, extra realization, or formal PPO training was started by this task.

Capture runtime: `70.488218s`; exact capture runtime: `60.263825s`; offline train: `0.581282s`; offline evaluation: `0.059279s`.
