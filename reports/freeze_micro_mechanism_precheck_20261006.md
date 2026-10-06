# Freeze Micro Mechanism Precheck (5D-0)

One episode per branch; this is a control-mechanism check, not a performance comparison or long training run.

## 1. Experiment setup

- Episode `20016` / seed `20016` from training pool `C:\Users\tangning\Desktop\code\adaptive_cache_mec\environment_data\proposed_joint_training`; base seed `27005`.
- Rollout episodes completed: Joint `1`, Freeze Micro `1`.
- Diagnostic outputs began empty: `True`. Formal long training started: `False`.

## 2. Starting checkpoint equality

- Checkpoint SHA `e056024d963fdfd1f99eb999462a2ab94a2ab7de0d064ea4f09782b4664fbaa6`; completed episodes `15`; architecture `adaptive_dual_timescale_unified_cache_candidate_cardinality_corrected_ppo_v17` v`25`; checkpoint v`49`.
- Initial policy and optimizer state digests equal: `True` / `True`.

## 3. RNG equality

- Pre-rollout RNG equal: `True`; post-rollout RNG equal: `True`.

## 4. Pre-update rollout equality

- Reward sequence: `True`; Macro actions: `True`; Micro actions: `True`; Return: `True`.
- Macro frames Joint/Freeze: `92` / `92`; Micro transitions Joint/Freeze: `300` / `300`.

## 5. Joint update result

- Joint update executed: `True`; Macro/Micro update metrics and parameter deltas are in the branch JSON.

## 6. Freeze Micro update result

- Freeze update executed: `True`; Micro Actor/Critic steps `0/0`.

## 7. Parameter ownership

- Micro/Macro overlap count `0`; Micro optimizer exactly owns Micro Actor+Critic: `True`.

## 8. Optimizer step comparison

- Joint optimizer step deltas: `{"formal_non_trigger_optimizer_steps": 184, "long_value_optimizer_steps": 50, "macro_actor_optimizer_steps": 4, "macro_cache_trigger_optimizer_steps": 0, "macro_critic_optimizer_steps": 50, "macro_optimizer_steps": 54, "micro_actor_optimizer_steps": 40, "micro_critic_optimizer_steps": 40, "micro_optimizer_steps": 40}`.
- Freeze optimizer step deltas: `{"formal_non_trigger_optimizer_steps": 104, "long_value_optimizer_steps": 50, "macro_actor_optimizer_steps": 4, "macro_cache_trigger_optimizer_steps": 0, "macro_critic_optimizer_steps": 50, "macro_optimizer_steps": 54, "micro_actor_optimizer_steps": 0, "micro_critic_optimizer_steps": 0, "micro_optimizer_steps": 0}`.

## 9. Freeze validity

- Micro Actor unchanged `True`; Micro Critic unchanged `True`; Micro optimizer state unchanged `True`.
- Micro actions still sampled `True`; Macro Actor/Critic/Long Value all updated `True`.

## 10. Gate decision

- Status: **5D-0：通过**.
- First control-variable problem: `无`.
- Performance comparison permitted: `False`.
- Next step: 停止，等待启动 Joint 10 vs Freeze Micro 10 的严格配对非平稳性实验。
