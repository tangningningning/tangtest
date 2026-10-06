# Macro Conditional Return Noise Diagnostic

**Witnesses:** 6 × 6 continuations = 36 branches. **Policy:** frozen v9/ckpt29. **Optimizer steps:** 0.

## Result

- HIGH_CONDITIONAL_RETURN_VARIANCE = **MATERIAL**
- MACRO_STATE_INFORMATION_INSUFFICIENT = **NOT_YET**
- ROOT_CAUSE = **CONDITIONAL_RETURN_VARIANCE_MATERIAL**
- CONDITIONAL_NOISE_FRACTION = 0.025950
- MEDIAN_STD_TO_GLOBAL_STD = 0.169523
- SINGLE_TRAJECTORY_TARGET_ERROR median / P90 = 68.058815 / 142.398545
- NORMALIZED_SINGLE_TRAJECTORY_ERROR median = 0.119129

## Aggregate statistics

| Metric | Value |
|---|---:|
| Median first-frame reward std | 4.936352 |
| Median conditional return std | 96.849247 |
| Global return std | 571.303394 |
| STD/global P25 / P50 / P75 / P90 | 0.111800 / 0.169523 / 0.197899 / 0.203553 |
| Within-state variance | 8469.820857 |
| Between-state variance | 317917.746635 |
| Total decomposed variance | 326387.567492 |

## Witness outcomes

| Witness | Seeds | Return mean | Return std | CV | First-frame reward mean | std | STD/global |
|---|---|---:|---:|---:|---:|---:|---:|
| episode_20001@125 | 9306977058532202007, 2616193780790141629, 3382774268384060756, 6963032560143409197, 15693425100423169966, 5085741712948167358 | 1269.621442 | 37.124743 | 0.029241 | 45.238198 | 3.502740 | 0.064983 |
| episode_20001@25 | 18172102404730450463, 14266055862758596157, 3216345953487826884, 15655004309685879194, 6942742404333871986, 11944995410170202548 | 1975.659627 | 57.260500 | 0.028983 | 10.927143 | 8.882822 | 0.100228 |
| episode_20002@75 | 5697742635641424655, 9480029208232544233, 18018350412491288538, 15853436005865809536, 14642516548976287108, 2764273452663095657 | 1047.821953 | 118.498858 | 0.113091 | 29.809333 | 4.580632 | 0.207418 |
| episode_20003@25 | 17428376112100412104, 3512182285763493711, 16492168756720037672, 2761238390171012672, 7441695795347836799, 11455186438644935116 | 1326.786459 | 83.705161 | 0.063089 | 27.620967 | 2.894839 | 0.146516 |
| episode_20004@125 | 3404979513973898196, 4457562967893453662, 13554822099048026193, 13253462565292338796, 6076846622027750353, 1777893629440000098 | 326.253785 | 114.082633 | 0.349675 | 21.653704 | 5.292072 | 0.199688 |
| episode_20004@75 | 3960901640107006918, 5492896853870643869, 11762012909922889274, 1162407278483542726, 16128666000026263940, 3362110627148702817 | 415.464212 | 109.993333 | 0.264748 | 8.309673 | 10.279131 | 0.192531 |

Each witness retains all six returns, state hashes, continuation seeds, task sample hashes and Macro action summaries in the JSON report.

## State and execution integrity

- Pre-witness observations and Cache action/duration matched the saved same-policy trace.
- All continuations for each witness began from the same state, policy, prediction artifact/state and time-to-go; only task rows from origin through terminal were resampled.
- Matching calls: 8450; KKT calls: 1367548; Exact-Q environment executions: 0.
- No optimizer steps, checkpoint writes, formal training, fresh5, source edits or config edits occurred.
- Reported Matching/KKT counts cover the successful measured run. An earlier preflight was interrupted before branches started; per-slot calls were not recorded and are excluded.
- Temporary driver/progress files, `__pycache__` directories and `*.pyc` files were removed.
- The existing config differs only in replay-history setting 0 vs checkpoint 4; restoring it in memory reproduced checkpoint hashes.

## Next decision

先增加 witness/continuation 数量验证，不修改正式算法。
