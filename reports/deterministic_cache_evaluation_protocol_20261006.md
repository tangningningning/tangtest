# 5C Fixed-Seed Deterministic Cache Evaluation Protocol Recheck

上一轮 deterministic rollout 已成功返回，但 report assembler 因 KeyError 未持久化逐帧统计；本报告为相同 episode_20001 / seed 20001 的固定 seed 修复复验。

## Scope and protocol

- Fixed episode/seed: `20001` / `20001`; base seed `27005`; rollout index `1`.
- Checkpoint SHA-256 before/after: `e056024d963fdfd1f99eb999462a2ab94a2ab7de0d064ea4f09782b4664fbaa6` / `e056024d963fdfd1f99eb999462a2ab94a2ab7de0d064ea4f09782b4664fbaa6`.
- Deterministic / Unified Cache / Cardinality Correction: `True` / `True` / `True`.
- Cache protocol: group probability mass decides KEEP vs UPDATE; ties select KEEP; UPDATE selects the legal raw-score argmax.
- No training, backward, update_from_episode, or optimizer step was called.

## Runtime summary

- Completed: `True`; Macro frames `300`; Micro slots `300`; runtime artifact persisted `True`.
- Mean P_KEEP / P_UPDATE_GROUP: `0.518483939668` / `0.481516070962`.
- New KEEP `299` (`0.996667`); UPDATE `1` (`0.003333`).
- Legacy same-frame flat KEEP `300` (`1.000000`); UPDATE `0` (`0.000000`).
- New/legacy disagreement `1` (`0.003333`).
- UPDATE group wins `1`; of these legacy flat KEEP `1`.
- Group rule violations `0`; UPDATE inner argmax violations `0`.
- Probability max difference `0`; maximum normalization error `4.76837158203e-07`; NaN/Inf `0/0`.
- Duration requested action rates: `{"1": 1.0, "3": 0.0, "5": 0.0}`; protocol changed `False`.
- UAV helper called `True` over `600` selections; UAV protocol changed `False`.
- Environment exceptions `0`; formal optimizer steps `0`.
- Policy unchanged `True`; optimizers unchanged `True`; counters unchanged `True`.

## Decision

- 5C status: **5C：通过**.
- First real protocol issue: `无`.
- Next step: 停止，等待 Freeze Micro vs Joint 非平稳性诊断实验。

Each Macro frame's required evidence is stored in the JSON artifact at `runtime.macro_frames`.
