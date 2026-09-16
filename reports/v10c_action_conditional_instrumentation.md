# V10-C Action-Conditional Instrumentation

STAGE = V10-C_ACTION_CONDITIONAL_INSTRUMENTATION

This stage adds observation-only diagnostics to the existing V9/V10 joint
Cache/UAV training path. It does not implement a counterfactual advantage, a
new reward, or a new optimizer/objective.

## Scope and insertion points

- `src/ppo_simple/trainer.py::PPOV2Trainer.update` copies the already computed,
  globally normalized `advantages_t` after the configured normalization step.
  It does not call GAE a second time and does not construct a heuristic signal.
- The same update loop captures the already evaluated UAV branch old/new
  log-probability, ratio, and PPO clip flag for each transition. The captured
  values are the last actual PPO minibatch evaluation for that sample in the
  update; no extra policy forward or sampling call is made.
- `src/ppo_simple/trainer.py::_append_critic_diagnostics` writes compact
  transition scalars/short arrays into the existing `critic_diagnostics.jsonl`:
  epoch/origin/slot, Cache action vector, selected UAV operation, physical movement,
  branch log-probability diagnostics, normalized advantage, return target,
  reward, utility, movement cost, delay, energy, and mask summary.
- The episode record adds the required actual-MOVE/actual-STAY aggregate fields,
  selected-action mismatch counts, feasible-action summary, and a compact
  per-slot Cache trigger bitmask plus histogram. No observation tensor or full
  state dump is written.

## Definitions and provenance

`ADVANTAGE_SOURCE = actual_PPO_advantage`. `global_advantage` is the exact
centered/std (or configured normalization mode) tensor consumed by the actor
loss. `global_return_target` is the existing raw GAE return target.

`actual_uav_move` is derived from the existing physical movement flag and is
checked against the rollout's position-difference movement count. The new
`uav_move_*` and `uav_stay_*` aggregates are grouped by this physical flag,
not merely by operation index. `selected_uav_action` is retained separately as
`STAY`/`MOVE`, so action-versus-physical mismatches remain visible.

The logged `old_uav_log_prob`, `new_uav_log_prob`, `uav_ratio`, and
`uav_clipped` are the current joint UAV branch quantities used by the existing
V10 PPO branch objective, with scope explicitly marked as
`joint_uav_branch_per_transition`. They are repeated only in the compact
per-UAV conditional aggregation because one transition contains both UAV
heads; they are not presented as independent per-head PPO ratios.

The Cache action is stored as the exact binary trigger vector in transition
diagnostics. Episode metrics additionally encode it as a little-endian integer
bitmask (`bit_i` is Cache head `i`) to keep the aggregate compact.

## Added epoch-level fields

For joint V9/V10 runs, `episode_metrics.jsonl` now contains:

| group | fields |
|---|---|
| counts | `uav_move_count`, `uav_stay_count` |
| advantage | `uav_move_adv_mean`, `uav_stay_adv_mean`, `uav_move_adv_median`, `uav_stay_adv_median`, `uav_move_adv_positive_fraction`, `uav_stay_adv_positive_fraction` |
| reward/objective | `uav_move_reward_mean`, `uav_stay_reward_mean`, `uav_move_q_total_mean`, `uav_stay_q_total_mean`, `uav_move_movement_cost_mean`, `uav_stay_movement_cost_mean` |
| delay/energy | `uav_move_delay_mean`, `uav_stay_delay_mean`, `uav_move_energy_mean`, `uav_stay_energy_mean` |
| return/log-prob | `uav_move_return_target_mean`, `uav_stay_return_target_mean`, `uav_move_old_log_prob_mean`, `uav_stay_old_log_prob_mean`, `uav_move_new_log_prob_mean`, `uav_stay_new_log_prob_mean` |
| PPO ratio | `uav_move_ratio_mean`, `uav_stay_ratio_mean`, `uav_move_clip_fraction`, `uav_stay_clip_fraction` |
| mask | `uav_feasible_action_count_mean/min/max`, `uav_move_candidate_count_mean/min/max`, `uav_action_mask_summary` |

Unavailable branches retain `null`, which existing analysis treats as
`NOT_AVAILABLE`. Existing legacy fields remain present for compatibility.

## Shared-gradient diagnostic

`SHARED_GRAD_COSINE_IMPLEMENTED = NO`.

The current training loop exposes only the joint shared-encoder gradient after
the real actor backward. Obtaining `grad_shared_from_cache` and
`grad_shared_from_uav` with `autograd.grad` would require retaining/replaying
the branch graph or adding another backward path, increasing memory/time and
risking a change in the formal numerical path. No extra optimizer step,
backward pass, or gradient mutation was introduced. Existing joint shared and
branch-head norm diagnostics are unchanged.

## Observational warning

MOVE versus STAY reward, utility, cost, delay, energy, ratio, and advantage
means are observational statistics. They condition on the states reached by
the selected action and cannot be interpreted as the causal effect of MOVE.
The instrumentation preserves the selected Cache action so future analysis can
control for it, but it does not claim causal identification.

## Runtime gate

The gate used one development epoch with
`configs/lr_sweep/v10_branchwise_actor_lr_6e4_kl.toml` and the existing
`episode_20001` realization from the episode pool. It was not a performance
run or a formal experiment.

- New fields were present and JSON serialization completed.
- The gate produced 300 transition records and two UAV heads per transition:
  `uav_move_count=345`, `uav_stay_count=255`, sum `600` as expected.
- All required aggregate values and per-transition branch values were finite.
- The normalized advantage field was copied from the actual PPO tensor; an
  independent float32 reproduction of centered/std normalization differed by
  at most `1.64e-7` due only to representation/operation order.
- All 300 transition records contained selected Cache action, selected UAV
  action, physical movement flags, mask summary, reward, utility, movement
  cost, delay, energy, return target, and branch log-probability diagnostics.
- The runtime checkpoint loaded successfully (`completed_episodes=1`), and an
  existing 300-epoch V10-B checkpoint also loaded successfully after the data
  structure change.
- Static diff audit found no new action sampling, RNG call, reward calculation,
  PPO loss term, critic model change, or optimizer step. The reward/action/RNG
  path is therefore unchanged by this instrumentation.

## Backward compatibility and cleanup

Additional JSON keys are ignored by existing readers; absent new keys remain
compatible with `None`/`NOT_AVAILABLE`. No old checkpoint fields were changed.
The temporary runtime-gate run directory and generated bytecode were removed.
Only the new instrumentation code/report from this stage was added; unrelated
pre-existing worktree changes were preserved.

## Final status

STAGE = V10-C_ACTION_CONDITIONAL_INSTRUMENTATION

INSTRUMENTATION_IMPLEMENTED = YES
ADVANTAGE_SOURCE = actual_PPO_advantage

ACTUAL_UAV_MOVE_LOGGED = YES
SELECTED_UAV_ACTION_LOGGED = YES
ACTION_MASK_SUMMARY_LOGGED = YES

MOVE_ADVANTAGE_STATS_AVAILABLE = YES
STAY_ADVANTAGE_STATS_AVAILABLE = YES

MOVE_REWARD_STATS_AVAILABLE = YES
STAY_REWARD_STATS_AVAILABLE = YES

MOVE_QTOTAL_STATS_AVAILABLE = YES
STAY_QTOTAL_STATS_AVAILABLE = YES

MOVE_COST_STATS_AVAILABLE = YES
STAY_COST_STATS_AVAILABLE = YES

MOVE_RATIO_STATS_AVAILABLE = YES
STAY_RATIO_STATS_AVAILABLE = YES

SHARED_GRAD_COSINE_IMPLEMENTED = NO

ACTION_CHANGED = NO
RNG_CHANGED = NO
REWARD_CHANGED = NO
PPO_OBJECTIVE_CHANGED = NO
CRITIC_CHANGED = NO

RUNTIME_GATE_EXECUTED = YES
RUNTIME_GATE_PASS = YES

FORMAL_TRAINING_EXECUTED = NO
FINAL_TEST_SEEDS_USED = NO

REPORT = C:\Users\tangning\Desktop\code\adaptive_cache_mec\reports\v10c_action_conditional_instrumentation.md
TEMP_FILES_REMOVED = YES
GIT_DIFF_CHECK = PASS
PUSHED = NO
