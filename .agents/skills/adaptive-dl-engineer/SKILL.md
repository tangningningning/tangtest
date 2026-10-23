---
description: Diagnose, implement, optimize, and verify deep-learning
  code in adaptive_cache_mec, especially PyTorch DQN/MAPPO,
  theory-to-code consistency, convergence, Replay Buffer alignment,
  candidate-action logic, and training speed. Use when asked to inspect
  or fix algorithm code, explain slow training, align equations with
  implementation, or make a concrete model/training change.
name: adaptive-dl-engineer
---

# Adaptive Deep-Learning Engineer

## Mission

Deliver the smallest correct, evidence-backed code change that resolves
the user's actual problem. Do not stop at a list of possible issues when
the available evidence supports a fix. Do not change code merely to make
it look cleaner, and do not claim a fix works beyond the verification
actually performed.

## Project-specific rules

-   Treat the current working tree and the latest user-uploaded version
    as the source of truth for implementation. In this project, a newer
    uploaded version replaces an older version unless the user
    explicitly requests a historical comparison.
-   Before editing, inspect the repository root, `git status`, current
    branch/HEAD, relevant source files, and the exact execution path.
    Preserve unrelated user changes. Never assume a remembered path, old
    audit report, or remote branch is identical to the active working
    tree.
-   Use the Conda environment `adaptive_cache_mec`. Prefer
    `conda run -n adaptive_cache_mec python ...`. If the Conda wrapper
    cannot launch a process, locate the environment's actual
    `python.exe` and use that exact interpreter; do not silently switch
    environments.
-   Do not launch formal training, long evaluation, or a real
    environment Slot that invokes Matching/KKT unless the user
    explicitly authorizes it. Static inspection and narrowly scoped
    tests are the default.
-   Do not modify formal rewards, system-model equations, configuration
    values, datasets, MAPPO, training pools, or experiment artifacts
    unless the request requires it. In particular, do not change a
    reward coefficient just because two algorithms use different values;
    first establish the intended comparison contract and obtain scope
    approval if changing the contract is outside the task.
-   Do not commit, push, or delete existing user files unless explicitly
    requested. Remove temporary files and test scripts created for the
    task before finishing.
-   Prefer a focused fix over another broad audit. Avoid repeated audits
    that do not produce new evidence or a concrete decision.

## Required workflow

### 1. Define the failure precisely

Translate the request into an observable symptom and a success
condition. Check for incorrect premises, missing information, and
logical leaps. Distinguish: - **Confirmed**: directly supported by
source, logs, or a reproducible targeted check. - **Likely**: supported
by evidence but not yet isolated causally. - **Unknown**: evidence is
missing.

Do not turn a hypothesis into a fact. If one essential detail is
missing, ask one concise question; otherwise proceed.

### 2. Trace the end-to-end path

For a learning or decision bug, trace only the relevant path end to end:

`data/forecast -> observation -> legal candidates/action mask -> action selection -> environment execution -> reward -> replay transition -> TD target -> optimizer/target sync -> checkpoint and metrics -> independent evaluation`

For every relevant equation, record the implementing function/file and
verify the semantics rather than matching variable names alone. Inspect
callers and consumers before changing a helper or interface.

### 3. Classify the root cause

Check the categories relevant to the symptom; do not mechanically audit
every category on every task.

-   **Data and state**: train/evaluation leakage, feature ordering,
    units, normalization, masks, missing values, future-information
    leakage, state sufficiency, episode boundaries.
-   **Action and constraints**: legal action generation, action
    execution identity, candidate truncation, cache capacity/REPLACE
    rules, cache/UAV mutual exclusion, deterministic tie-breaking.
-   **Learning target**: online selection versus target evaluation,
    terminal versus truncation, reward and discount semantics, detached
    targets, target-network sync, duration discounting.
-   **Optimization**: loss, gradient flow, optimizer parameter coverage,
    gradient clipping, non-finite values, accidental optimizer/network
    desynchronization.
-   **Replay and recovery**: state/action/next-state consistency,
    per-Episode mappings, schema/version checks, context lifetime,
    checkpoint identity and atomic restoration.
-   **Performance**: measured time spent in observation construction,
    candidate generation, model scoring, replay reconstruction,
    TD-target computation, environment execution, logging, plotting, and
    checkpointing.
-   **Evaluation**: training return versus held-out return, fixed
    evaluation seeds, baselines, reward-contract equality, variance
    across seeds.

Choose the smallest category set that can explain the observed failure.

### 4. Write down the actual mathematical contract

Before changing algorithm logic, state the intended equation and the
exact code path that implements it. Explicitly check: - objective and
reward units/scales; - whether the objective is discounted return,
fixed-horizon return, or average reward; - what counts as termination
versus time-limit truncation; - action-space restrictions and
candidate-search budgets; - whether the state is claimed to be Markov or
is only a prediction-based observation; - whether the implementation is
a standard algorithm or a custom variant.

Never describe a bounded candidate search as exhaustive or globally
optimal. Never infer convergence from a falling TD loss alone.

### 5. Implement a minimal causal fix

-   Change only files necessary for the identified root cause.
-   Preserve public interfaces and data contracts unless the fix
    requires a versioned change.
-   If tensor semantics, state/candidate features, network architecture,
    replay schema, or checkpoint identity changes, update the
    appropriate contract/version and reject incompatible checkpoints
    clearly. Do not silently reinterpret old Replay or checkpoints.
-   Keep terminal and truncated transitions distinct.
-   Ensure the action stored in Replay is the action actually executed,
    and ensure reward comes from the intended environment contract.
-   Keep target calculations detached from target-network gradients.
-   Do not add arbitrary Q-value penalties, reward shaping,
    normalization constants, or heuristic correction terms without
    evidence and an explicit theoretical rationale.
-   Avoid new dependencies unless necessary.

### 6. Optimize speed only from evidence

First identify the slow stage from existing timing logs or a narrowly
scoped profile. Do not guess from total runtime alone.

For candidate-based DQN, specifically inspect whether the implementation
repeatedly: - rebuilds the same legal candidate set for a transition; -
scores every candidate separately instead of batching; - recomputes
current-state category means and next-state values across transitions; -
decompresses or validates the same Replay context multiple times; - runs
expensive environment execution during candidate generation; - flushes
excessive per-Slot logs or plots too frequently.

Preserve mathematical equivalence when caching or batching. Cache keys
must include all state, candidate, episode-mapping, and contract fields
that affect the result; invalidate caches on state changes. Never remove
the gradient path needed to calculate the legal-candidate mean merely to
make training faster. Do not optimize a stage that has not been shown to
be a meaningful bottleneck.

### 7. Verify proportionally

Run only checks that directly support the change: 1. syntax/import check
for changed modules; 2. a targeted unit or contract check for the fixed
behavior; 3. `git diff --check`; 4. a narrow regression check for nearby
invariants when warranted.

Use the project environment. Do not run the full training pool, long
tests, real Matching/KKT execution, or broad test suites unless needed
and authorized. A synthetic test must be labelled synthetic and must not
be presented as proof of real-environment performance.

After verification, inspect the final diff and remove task-created
temporary scripts, scratch outputs, and `__pycache__` artifacts where
applicable. Do not delete pre-existing files.

## DQN-specific invariants for adaptive_cache_mec

When working on `src/online_joint_dqn/`, verify these against the
current checkout before relying on them:

1.  **Episode vehicle alignment**: vehicle IDs/array positions are not
    assumed to identify the same physical vehicle across Episodes. Use
    the corresponding Episode-specific canonical slot mapping and Replay
    candidate rebuilder. Do not rebuild historical experiences with the
    current Episode's mapping.
2.  **Action contract**: the current design uses `KEEP_STAY`,
    `CACHE_UPDATE`, and `UAV_MOVE`, with cache update and UAV movement
    mutually exclusive in one Slot. Candidate-generation budgets may
    omit legal combinations; report this honestly.
3.  **Custom Dueling definition**: if the code uses category-specific
    values and within-category mean-centered candidate advantages, write
    that exact formula in the theory. Do not call it identical to
    textbook Dueling DQN without qualification.
4.  **Double DQN**: the online network chooses the next candidate using
    the full defined Q value; the target network evaluates that same
    candidate. Terminal transitions do not bootstrap; truncated
    transitions should preserve the next state when the training
    contract says so.
5.  **Reward contract**: derive reward from the intended formal executor
    and verify every coefficient against the active config and
    comparison baseline. Previously inspected code showed a potential
    `DQN_BETA_Q` versus formal `objective.beta_q` mismatch; re-check the
    active code/config rather than assuming either value remains
    current.
6.  **Gamma**: `gamma=1` is justified for a fixed-length
    cumulative-return objective when all compared episodes have the same
    horizon. Reassess if the horizon or objective changes.
7.  **No global-optimum claims**: bounded cache cross-node search and
    finite UAV candidate generation solve a restricted candidate problem
    unless completeness is proven.
8.  **Evidence of learning**: distinguish training return, TD loss,
    Q-value diagnostics, and independent held-out evaluation. None alone
    proves convergence or superiority.

## Completion report

Finish with a concise, evidence-based report containing: - root cause
and status (confirmed/likely/unknown); - exact files changed and the
purpose of each change; - the mathematical/behavioral contract preserved
or changed; - checks actually run and their results; - whether formal
training, a real Slot, Matching, or KKT was executed; - remaining risks
and the single most useful next step, if any.

Do not claim tests passed if they were not run. Do not say the problem
is solved if only a hypothesis was documented.
