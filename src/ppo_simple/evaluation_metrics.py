"""Pure logging and evaluation statistics for frozen PPO episodes.

The environment already computes task delay, vehicle energy, and matching.  This
module only validates and aggregates those completed results; it does not call
the system model, Matching, KKT, or any policy code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


EXECUTION_CLASSES = ("LOCAL", "SV", "RSU", "UAV", "CLOUD")


@dataclass(frozen=True)
class SlotTaskMetrics:
    """Task performance already produced by one ``SlotExecutionResult``."""

    task_count: int
    total_task_delay_s: float
    total_vehicle_energy_j: float
    execution_target_counts: dict[str, int]


def _require_mapping(value: Any, field_name: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _finite_nonnegative(value: Any, field_name: str, key: Any) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(
            f"{field_name}[{key!r}] must be finite and non-negative, got {value!r}"
        )
    return numeric


def _execution_class(target: Any, cache_nodes: Mapping[str, Any]) -> str:
    target_id = str(target)
    if target_id == "LOCAL" or target_id.startswith("LOCAL:"):
        return "LOCAL"
    if target_id == "CLOUD":
        return "CLOUD"

    node_ids = [str(value) for value in cache_nodes["node_ids"]]
    if target_id not in node_ids:
        raise ValueError(f"unknown matching execution target: {target_id}")
    node_index = node_ids.index(target_id)
    role = str(cache_nodes["node_roles"][node_index]).upper()
    if role not in {"SV", "RSU", "UAV"}:
        raise ValueError(f"unknown cache node execution role: {role}")
    return role


def task_metrics_from_result(
    result: Any,
    cache_nodes: Mapping[str, Any],
) -> SlotTaskMetrics:
    """Extract task metrics from a completed ``SlotExecutionResult``.

    All four task-keyed mappings are required to describe exactly the same
    tasks.  The values are read as-is from the result and only checked/added;
    no task-performance formula is recomputed here.
    """

    delay = _require_mapping(result.per_task_delay_s, "per_task_delay_s")
    energy = _require_mapping(
        result.per_task_vehicle_energy_j,
        "per_task_vehicle_energy_j",
    )
    utility = _require_mapping(result.per_task_utility, "per_task_utility")
    matching = _require_mapping(result.matching, "matching")
    key_sets = {
        "per_task_delay_s": set(delay),
        "per_task_vehicle_energy_j": set(energy),
        "per_task_utility": set(utility),
        "matching": set(matching),
    }
    expected_keys = next(iter(key_sets.values()))
    if any(keys != expected_keys for keys in key_sets.values()):
        raise ValueError(
            "per_task_delay_s, per_task_vehicle_energy_j, per_task_utility, "
            "and matching must have exactly the same keys"
        )

    total_delay = 0.0
    total_energy = 0.0
    execution_counts = {name: 0 for name in EXECUTION_CLASSES}
    for task_id in delay:
        total_delay += _finite_nonnegative(
            delay[task_id],
            "per_task_delay_s",
            task_id,
        )
        total_energy += _finite_nonnegative(
            energy[task_id],
            "per_task_vehicle_energy_j",
            task_id,
        )
        execution_counts[_execution_class(matching[task_id], cache_nodes)] += 1

    task_count = len(expected_keys)
    if sum(execution_counts.values()) != task_count:
        raise AssertionError("execution target counts do not sum to task_count")
    return SlotTaskMetrics(
        task_count=task_count,
        total_task_delay_s=float(total_delay),
        total_vehicle_energy_j=float(total_energy),
        execution_target_counts=execution_counts,
    )


def slot_task_metrics_dict(transition: Any) -> dict[str, Any]:
    """Return JSON-ready task metrics for one rollout transition."""

    task_count = int(transition.task_count)
    if task_count < 0:
        raise ValueError("transition.task_count must be non-negative")
    counts = {
        name: int(transition.execution_target_counts.get(name, 0))
        for name in EXECUTION_CLASSES
    }
    if any(value < 0 for value in counts.values()):
        raise ValueError("transition execution target counts must be non-negative")
    if sum(counts.values()) != task_count:
        raise AssertionError("transition execution target counts do not sum to task_count")
    total_delay = _finite_nonnegative(
        transition.total_task_delay_s,
        "transition.total_task_delay_s",
        "slot",
    )
    total_energy = _finite_nonnegative(
        transition.total_vehicle_energy_j,
        "transition.total_vehicle_energy_j",
        "slot",
    )
    return {
        "task_count": task_count,
        "mean_task_delay_s": (
            None if task_count == 0 else float(total_delay / float(task_count))
        ),
        "mean_vehicle_energy_j": (
            None if task_count == 0 else float(total_energy / float(task_count))
        ),
        "execution_counts": counts,
    }


def aggregate_episode_task_metrics(transitions: Iterable[Any]) -> dict[str, Any]:
    """Aggregate task metrics task-weightedly across all episode slots."""

    total_tasks = 0
    total_delay = 0.0
    total_energy = 0.0
    execution_counts = {name: 0 for name in EXECUTION_CLASSES}
    for transition in transitions:
        slot_metrics = slot_task_metrics_dict(transition)
        task_count = int(slot_metrics["task_count"])
        total_tasks += task_count
        total_delay += _finite_nonnegative(
            transition.total_task_delay_s,
            "transition.total_task_delay_s",
            "episode",
        )
        total_energy += _finite_nonnegative(
            transition.total_vehicle_energy_j,
            "transition.total_vehicle_energy_j",
            "episode",
        )
        for name in EXECUTION_CLASSES:
            execution_counts[name] += int(slot_metrics["execution_counts"][name])
    if sum(execution_counts.values()) != total_tasks:
        raise AssertionError("episode execution counts do not sum to total_tasks")
    execution_ratios = {
        name: (
            0.0
            if total_tasks == 0
            else float(execution_counts[name]) / float(total_tasks)
        )
        for name in EXECUTION_CLASSES
    }
    return {
        "total_tasks": int(total_tasks),
        "mean_task_delay_s": (
            None if total_tasks == 0 else float(total_delay / float(total_tasks))
        ),
        "mean_vehicle_energy_j": (
            None if total_tasks == 0 else float(total_energy / float(total_tasks))
        ),
        "execution_counts": execution_counts,
        "execution_ratios": execution_ratios,
    }


def episode_metric_statistics(
    summaries: Iterable[Mapping[str, Any]],
    metric_names: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Compute episode-level sample statistics and Student-t 95% CIs."""

    summary_list = list(summaries)
    values_by_metric: dict[str, list[float]] = {}
    for metric_name in metric_names:
        values: list[float] = []
        for summary in summary_list:
            value = summary.get(metric_name)
            if value is None:
                raise ValueError(
                    f"episode metric {metric_name} is null; cannot pool statistics"
                )
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"episode metric {metric_name} is non-finite")
            values.append(numeric)
        values_by_metric[str(metric_name)] = values

    result: dict[str, dict[str, float]] = {}
    for metric_name, values in values_by_metric.items():
        count = len(values)
        if count < 2:
            raise ValueError("episode-level Student-t statistics require at least 2 episodes")
        array = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(array))
        std = float(np.std(array, ddof=1))
        se = float(std / np.sqrt(float(count)))
        try:
            from scipy.stats import t as student_t
        except ModuleNotFoundError as exc:  # pragma: no cover - env dependency
            raise ImportError(
                "pool statistics require scipy for the Student-t critical value"
            ) from exc
        critical = float(student_t.ppf(0.975, count - 1))
        result[metric_name] = {
            "mean": mean,
            "std": std,
            "median": float(np.median(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "se": se,
            "ci95_low": float(mean - critical * se),
            "ci95_high": float(mean + critical * se),
        }
    return result


def execution_ratio_statistics(
    summaries: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute episode-level mean/sample-std execution ratio per class."""

    summary_list = list(summaries)
    count = len(summary_list)
    if count < 2:
        raise ValueError("execution ratio statistics require at least 2 episodes")
    result: dict[str, dict[str, float]] = {}
    for name in EXECUTION_CLASSES:
        values = np.asarray(
            [float(summary["execution_ratios"][name]) for summary in summary_list],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"execution ratios for {name} contain non-finite values")
        result[name] = {
            "mean_execution_ratio": float(np.mean(values)),
            "std_execution_ratio": float(np.std(values, ddof=1)),
        }
    return result


def infer_episode_seed(realization_dir: str | Path | None) -> int | None:
    """Infer an explicit episode seed only from a conventional episode directory."""

    if realization_dir is None:
        return None
    name = Path(realization_dir).name
    if not name.startswith("episode_"):
        return None
    suffix = name.removeprefix("episode_")
    try:
        return int(suffix)
    except ValueError:
        return None
