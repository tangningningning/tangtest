# Macro Recent Causal Context Diagnostic

**Status: FAIL.** D recent aggregate history did not reduce action-conditioned target gaps; E recent service mix also missed the 15% cheap-gate threshold. The formal Macro observation remains 400D (architecture v9, checkpoint v29).

## Source and causal alignment

Read only the existing `tasks.csv` artifacts for episode_20001 through episode_20005. Each has 300 slots and 20 task-vehicle rows per slot. Trace origins are 0, 5, …, 295 and match cumulative prior Macro duration; every feature at origin `t` reads only `[max(0,t-h), t)`. Slot `t` and later task rows are excluded. All histories are zero at the first origin.

Trace SHA-256: `1d573b2686317640aee045399e373c269646c0ffe853a775412691f4d0e4c58d`. No realization was generated and the prior augmentation report and trace were left unchanged.

## D feature statistics

| Feature | Episode | Mean | Std | Min | Max |
|---|---|---:|---:|---:|---:|
| HIST_REQ_1 | 20001 | 0.692500 | 0.141075 | 0.000000 | 0.900000 |
| HIST_REQ_1 | 20002 | 0.684167 | 0.141566 | 0.000000 | 0.900000 |
| HIST_REQ_1 | 20003 | 0.703333 | 0.151346 | 0.000000 | 1.000000 |
| HIST_REQ_1 | 20004 | 0.689167 | 0.130732 | 0.000000 | 0.900000 |
| HIST_REQ_1 | 20005 | 0.670833 | 0.132064 | 0.000000 | 0.850000 |
| HIST_REQ_1 | between-episode mean std | 0.010641 |  |  |  |
| HIST_REQ_3 | 20001 | 0.692778 | 0.105425 | 0.000000 | 0.833333 |
| HIST_REQ_3 | 20002 | 0.688333 | 0.102166 | 0.000000 | 0.800000 |
| HIST_REQ_3 | 20003 | 0.684722 | 0.111857 | 0.000000 | 0.833333 |
| HIST_REQ_3 | 20004 | 0.688056 | 0.107673 | 0.000000 | 0.816667 |
| HIST_REQ_3 | 20005 | 0.687778 | 0.108092 | 0.000000 | 0.816667 |
| HIST_REQ_3 | between-episode mean std | 0.002576 |  |  |  |
| HIST_REQ_5 | 20001 | 0.688167 | 0.099255 | 0.000000 | 0.790000 |
| HIST_REQ_5 | 20002 | 0.687500 | 0.098415 | 0.000000 | 0.790000 |
| HIST_REQ_5 | 20003 | 0.687167 | 0.102682 | 0.000000 | 0.810000 |
| HIST_REQ_5 | 20004 | 0.680167 | 0.100058 | 0.000000 | 0.800000 |
| HIST_REQ_5 | 20005 | 0.690167 | 0.102103 | 0.000000 | 0.790000 |
| HIST_REQ_5 | between-episode mean std | 0.003397 |  |  |  |
| HIST_WORK_1 | 20001 | 0.205717 | 0.051222 | 0.000000 | 0.353080 |
| HIST_WORK_1 | 20002 | 0.216822 | 0.060377 | 0.000000 | 0.375340 |
| HIST_WORK_1 | 20003 | 0.231691 | 0.062247 | 0.000000 | 0.394664 |
| HIST_WORK_1 | 20004 | 0.219694 | 0.051909 | 0.000000 | 0.350096 |
| HIST_WORK_1 | 20005 | 0.194992 | 0.049069 | 0.000000 | 0.298612 |
| HIST_WORK_1 | between-episode mean std | 0.012514 |  |  |  |
| HIST_WORK_3 | 20001 | 0.204587 | 0.036611 | 0.000000 | 0.271723 |
| HIST_WORK_3 | 20002 | 0.222199 | 0.043165 | 0.000000 | 0.301659 |
| HIST_WORK_3 | 20003 | 0.224883 | 0.042149 | 0.000000 | 0.289145 |
| HIST_WORK_3 | 20004 | 0.222329 | 0.041030 | 0.000000 | 0.284250 |
| HIST_WORK_3 | 20005 | 0.199073 | 0.034953 | 0.000000 | 0.258980 |
| HIST_WORK_3 | between-episode mean std | 0.010626 |  |  |  |
| HIST_WORK_5 | 20001 | 0.203441 | 0.032699 | 0.000000 | 0.260834 |
| HIST_WORK_5 | 20002 | 0.220667 | 0.040747 | 0.000000 | 0.296726 |
| HIST_WORK_5 | 20003 | 0.226129 | 0.039166 | 0.000000 | 0.283035 |
| HIST_WORK_5 | 20004 | 0.218547 | 0.037348 | 0.000000 | 0.279919 |
| HIST_WORK_5 | 20005 | 0.198491 | 0.031679 | 0.000000 | 0.237197 |
| HIST_WORK_5 | between-episode mean std | 0.010609 |  |  |  |

Feature/target Pearson and Spearman coefficients are in the JSON report.

## Action-conditioned nearest-neighbor gate

Distance is `L2((2*x_new-1)-(2*x_old-1))/sqrt(D)`. Each held-out realization is matched only to prior-train rows with the same KEEP/UPDATE action and requested duration. Target gaps are normalized by the held-out episode target standard deviation.

| Target | Candidate | Gap median | Gap P90 | Reduction vs A, median / P90 |
|---|---|---:|---:|---:|
| Formal GAE | A | 2.183287 | 6.681667 | 0.000000% / 0.000000% |
| Formal GAE | D | 2.214432 | 6.681667 | -1.426506% / 0.000000% |
| Formal GAE | E | 2.214432 | 6.662583 | -1.426506% / 0.285614% |
| Realized return-to-go | A | 1.965250 | 3.145150 | 0.000000% / 0.000000% |
| Realized return-to-go | D | 1.965250 | 3.145150 | 0.000000% / 0.000000% |
| Realized return-to-go | E | 2.000055 | 3.116928 | -1.771005% / 0.897328% |

D nearest-neighbor identity changed on 10.00% of held-out queries (90% retained). Its largest median/P90 gap reduction across both targets was 0.000%, below 5%; **D_CHEAP_GATE = FAIL**.

E identity changed on 14.58% of queries. Its largest median/P90 gap reduction across both targets was 0.897%, below 15%; **E_CHEAP_GATE = FAIL**.

## Critic comparison and decision

A/D/E critic TEST_EV and matched-fit TEST_EV were not run because the cheap gates failed. No critic steps were executed. `SELECTED_CONTEXT = NONE`; `RECENT_CAUSAL_CONTEXT_STATUS = FAIL`.

No source or config files were modified in this turn. The existing uncommitted config change was preserved. Matching calls: 0; KKT calls: 0; Exact-Q environment executions: 0; fresh5: no.

Next diagnostic: `CONDITIONAL_RETURN_NOISE`, under the same frozen policy; this branch was not run here.
