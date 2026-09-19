# Single proposed mainline consolidation

本报告记录 `EXACT_Q_LISTWISE` implementation gate 通过后的 production
mainline 收敛。该整理只做配置、schema、entrypoint 和历史 Recovery runtime
路径清理；没有运行训练或正式评估。

## Result

SINGLE_PROPOSED_MAINLINE = YES
PROPOSED_CONFIG_COUNT = 1
LEGACY_RECOVERY_RUNTIME_PATHS_REMOVED = YES
BASELINE_CONFIGS_PRESERVED = YES
ABLATION_CONFIGS_PRESERVED = YES
HISTORICAL_REPORTS_PRESERVED = YES
TRAINING_EXECUTED = NO
OPTIMIZER_STEP_EXECUTED = NO
TEMP_FILES_REMOVED = YES
COMMITTED = NO
PUSHED = NO

## Mainline

- Stable experiment ID: `proposed_joint`
- Policy schema: `adaptive_joint_cache_uav_ppo`
- Control mode: `policy_primitive_joint_cache_uav_operation`
- Canonical configuration: `configs/proposed/system_config.toml`
- Candidate policy improvement: `EXACT_Q_LISTWISE`
- Exact-Q temperature: `1.0`

`configs/proposed/` contains only the canonical proposed configuration. The
six Recovery R2/R3/R4/R5/R6 proposed configuration files and their production
schema/runtime references were removed. Baseline and ablation configuration
directories were not removed or rewritten. Existing reports and run artifacts
were preserved as historical records.

## Static checks

- Exact-Q listwise implementation gate: PASS.
- Pure-tensor teacher ordering, equal-Q uniformity, virtual expected-Q
  nondecrease, and Cache/UAV gradient-scope checks: PASS.
- Proposed configuration validation with artifact checks disabled: PASS.
- Modified production modules: `py_compile` PASS.
- `scripts/experiment_registry.py --show proposed_joint`: PASS.
- Proposed policy import/entrypoint check: PASS.
- No training, optimizer step, SUMO execution, episode regeneration, or final
  seed evaluation was run.

## Invariants

Reward, system model, Matching/KKT, physical action space, episode protocol,
dataset, and final-test seed definitions were not changed. The retained exact
Cache/UAV counterfactual credit and exact-Q listwise objective are the current
mainline implementation; obsolete Recovery-specific class/schema aliases and
the old candidate PPO loss path are not production entrypoints.
