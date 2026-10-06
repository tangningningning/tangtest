# Cache Trigger Input Information Sufficiency Diagnosis

## Required conclusions

- RAW_LINEAR_Q_SIGNAL_GENERALIZES = NO
- RAW_NONLINEAR_Q_SIGNAL_GENERALIZES = NO
- ENCODER_DESTROYS_USABLE_TRIGGER_SIGNAL = NO
- TRIGGER_OBSERVATION_INFORMATION_INSUFFICIENT = YES

TRAINING_EXECUTED = NO (formal policy training); the requested offline diagnostic MLP was fit using manual NumPy gradients. PPO_BACKWARD_CALLS = 0; OPTIMIZER_STEP_CALLS = 0; NEW_EXACT_Q_BRANCHES = 0; VALIDATION_EXECUTED = NO; TEST_EXECUTED = NO.

## Replay and protocol

Replayed the prior production states using the same training realizations and RNG rule. Maximum saved/replayed p_update absolute difference: 1.1920929e-07 (required <= 1e-6); encoder representation max absolute difference: 0. Checkpoint parameters remained unchanged.

Analysis covers 512 rows from 16 mixed/error heads (32 states per head), four training realizations, and eight slots per realization. Raw cache_context width is 400. Ridge uses nested realization-held-out CV over alpha [0.1, 1.0, 10.0, 100.0]; preprocessing is fit on training folds only. Pooled MLP has 32 ReLU units plus physical-head one-hot identity, fixed 200 epochs, L2 1e-3, and no early stopping.

## Raw input versus encoder probes

| Probe | Median per-head OOF AUC | Median per-head OOF R2 | Median per-head sign accuracy | AUC change versus encoder |
|---|---:|---:|---:|---:|
| Raw-context nested Ridge | 0.4900 | -0.4308 | 0.5625 | 0.0632 |
| Raw-context pooled MLP | 0.5012 | -7.7578 | 0.5000 | 0.1039 |
| Prior encoder Ridge | 0.4147 | -0.1139 | 0.5312 | reference |

Pooled raw MLP OOF AUC: 0.5166; fold AUCs (held realizations 20001, 20100, 20200, 20300): 0.5425, 0.5358, 0.5190, 0.4933. Heads with raw MLP AUC >=0.60: 3/16. Raw Ridge heads with AUC >=0.65: 5/16. MLP median training-fold AUC/R2 were 0.9384/0.8494, showing fit on training realizations without held-realization generalization.

The table compares median per-head OOF AUC consistently. Pooled nonlinear AUC is reported separately in the text and JSON.

## Per-head results

| Run/checkpoint | Head/node | Role | Raw Ridge AUC | Raw Ridge R2 | Encoder AUC | Raw MLP AUC | Raw MLP R2 |
|---|---|---|---:|---:|---:|---:|---:|
| 5e-4 e20 | 0/46 | SV | 0.3806 | -0.4526 | 0.2874 | 0.6316 | -60.2174 |
| 5e-4 e50 | 1/10 | SV | 0.5411 | -0.2815 | 0.6710 | 0.6797 | -2.1989 |
| 5e-4 e50 | 3/23 | SV | 0.6071 | -0.4863 | 0.3770 | 0.4563 | -2.1845 |
| 5e-4 e30 | 4/59 | SV | 0.7490 | -0.4630 | 0.7409 | 0.4777 | -8.9779 |
| 5e-4 e10 | 5/RSU_0 | RSU | 0.4250 | -0.5961 | 0.3250 | 0.4792 | -5.7222 |
| 5e-4 e10 | 6/RSU_1 | RSU | 0.4113 | -0.1730 | 0.5065 | 0.5498 | -3.6517 |
| 5e-4 e20 | 7/UAV_0 | UAV | 0.3571 | -0.7558 | 0.3373 | 0.4563 | -35.5970 |
| 5e-4 e20 | 8/UAV_1 | UAV | 0.4042 | -0.5635 | 0.4792 | 0.3667 | -32.3732 |
| 1e-3 e20 | 0/46 | SV | 0.7216 | 0.2226 | 0.5255 | 0.5686 | -23.3154 |
| 1e-3 e10 | 2/0 | SV | 0.6587 | -0.1763 | 0.4484 | 0.5714 | -13.6061 |
| 1e-3 e50 | 3/23 | SV | 0.5250 | -0.0976 | 0.4917 | 0.5083 | -0.7176 |
| 1e-3 e10 | 4/59 | SV | 0.6727 | -0.0270 | 0.3500 | 0.5364 | -11.2545 |
| 1e-3 e10 | 5/RSU_0 | RSU | 0.3889 | -0.6252 | 0.3810 | 0.6548 | -4.8830 |
| 1e-3 e40 | 6/RSU_1 | RSU | 0.4494 | -0.9851 | 0.3158 | 0.4089 | -2.1119 |
| 1e-3 e50 | 7/UAV_0 | UAV | 0.6523 | 0.0941 | 0.3672 | 0.3125 | -18.2610 |
| 1e-3 e10 | 8/UAV_1 | UAV | 0.4549 | -0.4090 | 0.5490 | 0.4941 | -6.5376 |

## Interpretation

Both raw probes meet the predeclared weak-signal criteria, and their cross-validated scores are close to the prior encoder probe. This supports insufficient information in the formal trigger observation for these mixed/error heads.

Static source field-presence map (no modifications): src/ppo_simple/interface.py:101-180 declares and allocates the cache-state, UAV-position, predicted-vehicle, predicted-popularity, optional horizon-mask, and optional time-to-go slices; lines 392-488 build the vector. src/ppo_simple/primitive_control.py:1263-1298 passes the conditional context and node_index separately to sample_cache_trigger. No dedicated head-ID coordinate is allocated in the vector. This is a field-presence review only.

## Scope

This report addresses only whether the formal Cache Trigger encoder input contains cross-realization information predictive of symmetric expected ΔQ for the selected mixed/error heads. It does not analyze other systems or propose an algorithm change. No algorithm or checkpoint was modified.
