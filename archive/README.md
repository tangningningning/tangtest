# Archive index

Archived files are retained for historical reproducibility but are not current
formal entrypoints.  Existing formal runs and reports are not moved or
rewritten.

| Old file | Replacement/current file | Reason archived |
| --- | --- | --- |
| `archive/deprecated_scripts/evaluate_episode_pool.py` | `scripts/evaluation/evaluate_v7_episode_pool.py`, `scripts/evaluation/evaluate_v8_episode_pool.py`, or `scripts/evaluation/evaluate_v9_joint_episode_pool.py` | Generic V6-era pool evaluator was superseded by version-specific frozen evaluators; implementation retained as historical reference. |

Current one-off diagnostics remain discoverable under
`scripts/diagnostics/`, while reusable numerical logic remains under `src/`.
