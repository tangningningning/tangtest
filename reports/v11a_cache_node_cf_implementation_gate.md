# V11-A Cache-node Counterfactual Advantage Implementation Gate

This is an implementation gate, not a formal performance experiment. Only the static Gate A, one synthetic/one-minibatch numerical Gate B, and one frozen-pool Gate C epoch were run. No 300-epoch training, LR sweep, final seed, F20 run, or unseen test was run.

## Final fields

```text
STAGE = V11A_CACHE_NODE_COUNTERFACTUAL_ADVANTAGE_IMPLEMENTATION_GATE
TRAINING_EXECUTED = YES
FORMAL_TRAINING_EXECUTED = NO
GATE_EPOCHS = 1

SYSTEM_MODEL_CHANGED = NO
GLOBAL_REWARD_CHANGED = NO
GLOBAL_GAE_FORMULA_CHANGED = NO
GLOBAL_CRITIC_CHANGED = NO
UAV_CODE_PATH_CHANGED = NO

CACHE_NODE_LOGPROB_EXACTLY_AVAILABLE = YES
CACHE_NODE_LOGPROB_SUM_MATCH = YES
MAX_ABS_LOGPROB_SUM_ERROR = 3.814697265625e-06
CACHE_CREDIT_GRANULARITY = FULL_NODE_ACTION

Q_CACHE_TARGET_SOURCE = current raw GAE return target, normalized through the active PopArt critic-target path
Q_CACHE_NORMALIZATION_SOURCE = existing PopArt normalization; no new reward or normalization
Q_CACHE_LR = 3e-4
Q_CACHE_OPTIMIZER = Adam

CF_ADV_RAW_MEAN = -0.0002901455751378779
CF_ADV_RAW_STD = 0.016912607636688972
CF_ADV_MIN = -0.036122411489486694
CF_ADV_MAX = 0.05434178188443184
CF_ADV_POSITIVE_FRACTION = 0.48185185185185186
CF_ADV_NEGATIVE_FRACTION = 0.5181481481481481

CACHE_OPERATION_ADVANTAGE_SOURCE = detached A_cf,m for the full Cache-node primitive
CACHE_SCORE_ADVANTAGE_SOURCE = detached A_cf,m for the full Cache-node primitive

STATIC_GATE_A = PASS
NUMERICAL_GATE_B = PASS
ONE_EPOCH_GATE_C = PASS

Q_CACHE_LOSS = 0.5053187608718872
Q_CACHE_EV = -0.003477334976196289
Q_CACHE_GRAD_NORM = 0.48352891206741333
ACTOR_OPTIMIZER_STEPS = 40
GLOBAL_CRITIC_OPTIMIZER_STEPS = 40
Q_CACHE_OPTIMIZER_STEPS = 1

CACHE_KL_GATE_CHANGED = NO
UAV_KL_GATE_CHANGED = NO
JOINT_KL_CONTROL_REINTRODUCED = NO
NAN_INF_DETECTED = NO
FORMAL_300_EPOCH_RECOMMENDED = WAIT

REPORT = reports/v11a_cache_node_cf_implementation_gate.md
RAW_RESULTS = runs/ppo_v11a_cache_node_cf_6e4_gate/
GIT_DIFF_CHECK = PASS
TEMP_FILES_REMOVED = YES
PUSHED = NO
```

## Implementation and contract checks

The exact Cache-node primitive log-probability is the masked operation categorical log-probability plus the masked candidate categorical log-probability for the selected operation. `NO_CHANGE` has no candidate probability contribution. The all-legal-action evaluator uses the same operation and candidate distributions and marks illegal/padded entries as `-inf`.

Gate A measured legal-action probability-sum error `0.0` and node-sum/Cache-branch error `2.384185791015625e-07`. Gate B measured probability-sum error `1.1920928955078125e-07`. The formal Gate C minibatch maximum was `3.814697265625e-06`, within the existing `1e-5` numerical tolerance.

`Q_cache(s, a_m)` is an independent two-hidden-layer `Linear + Tanh` MLP with a deterministic node one-hot and exact production primitive features. Its parameters are disjoint from all Actor/global-critic parameters, and its target is the current PopArt-normalized raw GAE return target. Its optimizer is an independent Adam optimizer at `3e-4`; gradient clipping uses the existing `0.5` rule.

For `FULL_NODE_ACTION`, both the operation head and candidate scorer use detached `A_cf,m = Q_selected - sum(pi_m(a') Q(a'))`. Counterfactual normalization is one global centered-std normalization over valid Cache node-time entries. Single-legal-action entries are excluded from both that normalization and the Cache actor surrogate. Gate B exercised this exclusion with 16 entries and observed both signs.

Global `V(s)`, reward, GAE, Cache/UAV action parameterization, environment/resource/KKT/cache-cost logic, UAV movement logic, and UAV global-advantage path were preserved. V11-A retains the V10-D branch-specific KL gate: Cache KL continues to use the whole Cache branch, UAV KL remains separate, and joint KL is diagnostic only (`joint_kl_control_enabled = false`).

## Gate results

Gate B: Q loss `0.6584768891`, Q EV `0.0015296936`, Q gradient norm `0.4999996722`; Cache actor gradient norm `0.8595752716`, UAV actor gradient norm `2.6319303513`, and global critic gradient norm `4.8737707138`. It observed 56 valid node-time entries, excluded 16 single-action entries, and observed opposite-sign CF advantages.

Gate C used `configs/proposed/system_config_ppo_v11a_cache_node_cf_6e4.toml`, seed `4401`, actor LR `6e-4`, critic/Q LR `3e-4`, target KL `0.10`, `gamma=1`, `GAE lambda=0.95`, clip `0.2`, entropy `0`, four update epochs, ten minibatches of 30, and the frozen pool `runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911`. It collected origins `0:299` (300 transitions), produced CacheChange `0.6466666667`, UAVMove `0.5283333333`, and return `-2100.313964957918`. These one-epoch values are recorded for implementation verification only and are not a performance conclusion.

The V10-D branch-specific gate stopped Cache updates after update epoch 1 while UAV updates stayed active for all four epochs. This is the retained branch-specific behavior, not reintroduced joint-KL suppression: Cache KL `1.2371718089`, UAV KL `0.0191957752`, joint KL diagnostic `1.8324466944`, and joint KL control remained disabled.

The fixed diagnostic transition was index/origin `110`, with `A_global=0.5920668840` and all nine Cache-node selected actions plus raw/normalized `A_cf,m` recorded in `episode_metrics.jsonl`, `update_metrics.jsonl`, and the JSON report.

## Checkpoint and raw results

Checkpoint version `14` is at `runs/ppo_v11a_cache_node_cf_6e4_gate/latest_checkpoint.pt`. It contains the V11-A Actor, global critic and optimizer state, `Q_cache` and its optimizer state, PopArt statistics, training origin/epoch metadata, config hash, RNG state (`numpy`, `python`, `torch`), and `FULL_NODE_ACTION` granularity. The V10-D checkpoint was not overwritten.

A lightweight config-hash checkpoint-load check passed: the restored cursor is `completed_episodes=1`, `current_train_origin=300`, and `q_cache_optimizer_steps=1`. No rollout or additional training was performed for this check.

Raw results are retained only for the required one-epoch gate artifact:

- `runs/ppo_v11a_cache_node_cf_6e4_gate/run_manifest.json`
- `runs/ppo_v11a_cache_node_cf_6e4_gate/episode_metrics.jsonl`
- `runs/ppo_v11a_cache_node_cf_6e4_gate/update_metrics.jsonl`
- `runs/ppo_v11a_cache_node_cf_6e4_gate/run_summary.json`
- `runs/ppo_v11a_cache_node_cf_6e4_gate/latest_checkpoint.pt`

The temporary Gate A/B runner, failed intermediate Gate C directories, and generated `__pycache__` directories were removed. Existing unrelated worktree changes were preserved. No commit or push was performed.
