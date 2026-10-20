"""Offline paired Cache-only/UAV-only interventions on frozen Training states.

The policy is replayed deterministically to reconstruct Cache-decision states.
Each reported intervention then goes through the production slot executor,
including a fresh Matching and KKT allocation. No optimizer is stepped.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import project_config
from decision import cache_configuration, slot_environment, task_allocation, uav_configuration
from ppo_simple import interface
from ppo_simple.primitive_control import (
    CACHE_ECONOMIC_DEMAND_HORIZONS,
    CACHE_ECONOMIC_FEATURE_DIM,
    PrimitiveTensorSpec,
)
from ppo_simple.rollout import (
    _cache_node_candidates,
    _observation,
    _positions_condition,
    _uav_candidates,
    task_metrics_from_result,
)
from ppo_simple.trainer import PPOTrainer


DEFAULT_RUN = "runs/cloud_shared_resources_logged_controlled_updates2_20261020"
DEFAULT_POOL = "environment_data/proposed_joint_training"
DEFAULT_REPORT = "reports/low_cache_hit_intervention_study.md"
DEFAULT_JSON = "reports/low_cache_hit_intervention_study.json"
HIT_TOLERANCE = 1.0e-12
CLOUD_UPLINK_CAPACITY_BPS = 20.0e6
CLOUD_CPU_CAPACITY_CYCLES_PER_S = 20.0e9
CLOUD_UPLINK_TOLERANCE_BPS = 1.0e-6
CLOUD_CPU_TOLERANCE_CYCLES_PER_S = 1.0e-3

CLOUD_SLOT_FIELDS = (
    "cloud_task_count",
    "cloud_uplink_allocated_bps",
    "cloud_uplink_capacity_bps",
    "cloud_uplink_capacity_violation_bps",
    "cloud_cpu_allocated_cycles_per_s",
    "cloud_cpu_capacity_cycles_per_s",
    "cloud_cpu_capacity_violation_cycles_per_s",
    "cloud_selected_resource_values_finite",
    "cloud_kkt_solve_count",
    "cloud_kkt_max_uplink_allocated_bps",
    "cloud_kkt_max_cpu_allocated_cycles_per_s",
    "cloud_kkt_max_uplink_capacity_violation_bps",
    "cloud_kkt_max_cpu_capacity_violation_cycles_per_s",
    "cloud_kkt_nonfinite_result_count",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _copy_positions(positions: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        str(key): np.asarray(value, dtype=np.float64).copy()
        for key, value in positions.items()
    }


def _cache_request_metrics(
    cache_state: np.ndarray,
    tasks: list[Any],
    service_count: int,
) -> dict[str, float | int | None]:
    if not tasks:
        return {
            "request_any_edge_cache_hit_rate": None,
            "catalog_cache_coverage_rate": float(
                np.count_nonzero(np.any(cache_state == 1, axis=0))
                / max(int(service_count), 1)
            ),
        }
    request_hits = sum(
        int(bool(np.any(cache_state[:, int(task.service_id)] == 1)))
        for task in tasks
    )
    return {
        "request_any_edge_cache_hit_rate": float(request_hits / len(tasks)),
        "catalog_cache_coverage_rate": float(
            np.count_nonzero(np.any(cache_state == 1, axis=0))
            / max(int(service_count), 1)
        ),
    }


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _audit_cloud_slot_metrics(path: Path, expected_slot_count: int) -> dict[str, Any]:
    """Validate the selected and candidate Cloud resource evidence before replay."""

    records: list[dict[str, Any]] = []
    failures: list[str] = []
    parse_errors: list[int] = []
    slot_keys: list[tuple[int, int]] = []
    slot_reward_by_episode: Counter[int] = Counter()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            parse_errors.append(line_number)
            continue
        if not isinstance(row, dict):
            failures.append(f"line {line_number}: row is not a JSON object")
            continue
        records.append(row)
        if type(row.get("episode")) is not int or type(row.get("slot")) is not int:
            failures.append(f"line {line_number}: missing/invalid episode-slot identity")
        elif not _is_finite_number(row.get("G")):
            failures.append(f"line {line_number}: missing/non-finite slot reward G")
        else:
            slot_keys.append((int(row["episode"]), int(row["slot"])))
            slot_reward_by_episode[int(row["episode"])] += float(row["G"])
        missing = [field for field in CLOUD_SLOT_FIELDS if field not in row]
        if missing:
            failures.append(f"line {line_number}: missing {missing}")
            continue

        for field in (
            "cloud_task_count",
            "cloud_kkt_solve_count",
            "cloud_kkt_nonfinite_result_count",
        ):
            if type(row[field]) is not int or row[field] < 0:
                failures.append(f"line {line_number}: invalid integer {field}={row[field]!r}")
        if row["cloud_selected_resource_values_finite"] is not True:
            failures.append(
                f"line {line_number}: selected resource finite flag is not true"
            )
        if not isinstance(row.get("node_task_count"), dict) or int(
            row["node_task_count"].get("Cloud", -1)
        ) != row["cloud_task_count"]:
            failures.append(f"line {line_number}: Cloud task count disagrees with node_task_count")

        selected_pairs = (
            (
                "uplink",
                "cloud_uplink_allocated_bps",
                "cloud_uplink_capacity_bps",
                "cloud_uplink_capacity_violation_bps",
                CLOUD_UPLINK_CAPACITY_BPS,
                CLOUD_UPLINK_TOLERANCE_BPS,
            ),
            (
                "CPU",
                "cloud_cpu_allocated_cycles_per_s",
                "cloud_cpu_capacity_cycles_per_s",
                "cloud_cpu_capacity_violation_cycles_per_s",
                CLOUD_CPU_CAPACITY_CYCLES_PER_S,
                CLOUD_CPU_TOLERANCE_CYCLES_PER_S,
            ),
        )
        for label, allocated_key, capacity_key, violation_key, expected_capacity, tol in selected_pairs:
            allocated = row.get(allocated_key)
            capacity = row.get(capacity_key)
            violation = row.get(violation_key)
            if not all(_is_finite_number(value) for value in (allocated, capacity, violation)):
                failures.append(f"line {line_number}: missing/non-finite selected {label} evidence")
                continue
            expected_violation = max(0.0, float(allocated) - float(capacity))
            if not math.isclose(float(capacity), expected_capacity, rel_tol=0.0, abs_tol=tol):
                failures.append(
                    f"line {line_number}: {label} capacity {capacity} != {expected_capacity}"
                )
            if float(allocated) > float(capacity) + tol:
                failures.append(
                    f"line {line_number}: selected {label} allocation exceeds capacity"
                )
            if not math.isclose(float(violation), expected_violation, rel_tol=0.0, abs_tol=tol):
                failures.append(
                    f"line {line_number}: selected {label} violation does not match max(0, allocation-capacity)"
                )

        solve_count = row.get("cloud_kkt_solve_count")
        kkt_fields = (
            "cloud_kkt_max_uplink_allocated_bps",
            "cloud_kkt_max_cpu_allocated_cycles_per_s",
            "cloud_kkt_max_uplink_capacity_violation_bps",
            "cloud_kkt_max_cpu_capacity_violation_cycles_per_s",
        )
        if type(solve_count) is int:
            if solve_count == 0:
                if any(row.get(field) is not None for field in kkt_fields):
                    failures.append(
                        f"line {line_number}: KKT aggregate values exist despite zero solves"
                    )
                if row["cloud_task_count"] > 0:
                    failures.append(
                        f"line {line_number}: selected Cloud tasks exist but no KKT result was recorded"
                    )
            else:
                values = [row.get(field) for field in kkt_fields]
                if not all(_is_finite_number(value) for value in values):
                    failures.append(
                        f"line {line_number}: KKT result exists but aggregate fields are null/non-finite"
                    )
                else:
                    uplink_allocated, cpu_allocated, uplink_violation, cpu_violation = map(
                        float, values
                    )
                    if uplink_allocated > CLOUD_UPLINK_CAPACITY_BPS + CLOUD_UPLINK_TOLERANCE_BPS:
                        failures.append(f"line {line_number}: KKT aggregate uplink exceeds capacity")
                    if cpu_allocated > CLOUD_CPU_CAPACITY_CYCLES_PER_S + CLOUD_CPU_TOLERANCE_CYCLES_PER_S:
                        failures.append(f"line {line_number}: KKT aggregate CPU exceeds capacity")
                    if uplink_violation > CLOUD_UPLINK_TOLERANCE_BPS:
                        failures.append(f"line {line_number}: KKT aggregate uplink violation {uplink_violation}")
                    if cpu_violation > CLOUD_CPU_TOLERANCE_CYCLES_PER_S:
                        failures.append(f"line {line_number}: KKT aggregate CPU violation {cpu_violation}")
        if row["cloud_kkt_nonfinite_result_count"] != 0:
            failures.append(
                f"line {line_number}: cloud_kkt_nonfinite_result_count={row['cloud_kkt_nonfinite_result_count']}"
            )

    if parse_errors:
        failures.append(f"unparseable JSONL lines: {parse_errors[:10]}")
    if len(records) != expected_slot_count:
        failures.append(
            f"slot row count {len(records)} != expected {expected_slot_count}"
        )
    duplicate_keys = [key for key, count in Counter(slot_keys).items() if count > 1]
    if duplicate_keys:
        failures.append(f"duplicate Episode-Slot records: {duplicate_keys[:10]}")
    if expected_slot_count == 2400:
        expected_keys = {(episode, slot) for episode in range(1, 9) for slot in range(300)}
        missing_keys = sorted(expected_keys - set(slot_keys))
        extra_keys = sorted(set(slot_keys) - expected_keys)
        if missing_keys or extra_keys:
            failures.append(
                f"Episode-Slot coverage mismatch: missing={missing_keys[:10]}, extra={extra_keys[:10]}"
            )
    if failures:
        raise ValueError(
            "Cloud resource log gate failed before counterfactual evaluation: "
            + "; ".join(failures[:12])
        )

    selected_cloud = sum(int(row["cloud_task_count"] > 0) for row in records)
    kkt_without_selected_cloud = sum(
        int(row["cloud_kkt_solve_count"] > 0 and row["cloud_task_count"] == 0)
        for row in records
    )
    no_kkt = sum(int(row["cloud_kkt_solve_count"] == 0) for row in records)
    return {
        "slot_records": len(records),
        "episode_slot_coverage": "episodes 1-8 each contain slots 0-299",
        "slot_reward_sum_by_episode": {
            str(key): float(value) for key, value in sorted(slot_reward_by_episode.items())
        },
        "parse_errors": 0,
        "required_field_missing_count": 0,
        "selected_resource_finite_count": len(records),
        "selected_uplink_violation_count": 0,
        "selected_cpu_violation_count": 0,
        "kkt_uplink_violation_count": 0,
        "kkt_cpu_violation_count": 0,
        "kkt_nonfinite_result_count": 0,
        "cloud_kkt_solve_count_total": int(
            sum(row["cloud_kkt_solve_count"] for row in records)
        ),
        "cloud_kkt_solve_count_min": min(
            row["cloud_kkt_solve_count"] for row in records
        ),
        "cloud_kkt_solve_count_max": max(
            row["cloud_kkt_solve_count"] for row in records
        ),
        "slots_no_kkt_result": no_kkt,
        "slots_kkt_but_no_selected_cloud": kkt_without_selected_cloud,
        "slots_kkt_and_selected_cloud": selected_cloud,
        "max_selected_uplink_allocated_bps": max(
            float(row["cloud_uplink_allocated_bps"]) for row in records
        ),
        "max_selected_cpu_allocated_cycles_per_s": max(
            float(row["cloud_cpu_allocated_cycles_per_s"]) for row in records
        ),
        "max_kkt_uplink_allocated_bps": max(
            float(row["cloud_kkt_max_uplink_allocated_bps"])
            for row in records
            if row["cloud_kkt_solve_count"] > 0
        ),
        "max_kkt_cpu_allocated_cycles_per_s": max(
            float(row["cloud_kkt_max_cpu_allocated_cycles_per_s"])
            for row in records
            if row["cloud_kkt_solve_count"] > 0
        ),
        "uplink_capacity_bps": CLOUD_UPLINK_CAPACITY_BPS,
        "cpu_capacity_cycles_per_s": CLOUD_CPU_CAPACITY_CYCLES_PER_S,
    }


def _macro_decision_states(
    policy: Any,
    context: Any,
    episode_seed: int,
    *,
    include_slot_states: bool = False,
    aligned_steps: list[Any] | None = None,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay the frozen policy's Cache/UAV state transitions without training."""

    if context.cache_action_mode != project_config.CACHE_ACTION_MODE_MULTI_NODE_FACTORIZED:
        raise ValueError(
            "This study currently requires the Run's MULTI_NODE_FACTORIZED Cache contract."
        )
    steps = (
        interface.aligned_train_steps(context)
        if aligned_steps is None
        else aligned_steps
    )
    if len(steps) != int(context.episode_slots):
        raise ValueError("Training realization does not contain one aligned step per slot.")

    tensor_spec = PrimitiveTensorSpec.create(context)
    cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
    uav = {
        str(key): np.asarray(value, dtype=np.float64).copy()
        for key, value in context.initial_uav.items()
    }
    uav_ids = tuple(str(value) for value in context.action_spec.uav_ids)
    min_holding = int(policy.adaptive_holding_min_slots)
    max_holding = int(policy.adaptive_holding_max_slots)
    if min_holding < 1 or max_holding < min_holding:
        raise ValueError("Checkpoint adaptive Cache holding bounds are invalid.")

    state_records: list[dict[str, Any]] = []
    slot_state_records: list[dict[str, Any]] = []
    frame_start = 0
    frame_elapsed = 0
    frame_cache = cache.copy()
    frame_actions = np.zeros(len(context.action_spec.cache_node_ids), dtype=np.int64)
    frame_candidates: list[list[np.ndarray]] = []

    for slot_index, step in enumerate(steps):
        previous_uav = _copy_positions(uav)
        previous_cache = cache.copy()
        if slot_index == frame_start:
            frame_observation = _observation(cache, uav, step, context)
            frame_candidates, cache_features, cache_masks = _cache_node_candidates(
                cache, uav, step, context, tensor_spec
            )
            selected_cache = policy.sample_macro_nodes(
                frame_observation,
                cache_features,
                cache_masks,
                deterministic=True,
            )
            frame_actions = np.asarray(
                selected_cache.cache_actions, dtype=np.int64
            ).reshape(-1)
            if frame_actions.shape != (len(frame_candidates),):
                raise AssertionError("Frozen Macro actions do not align with Cache nodes.")
            frame_cache = cache.copy()
            for node_index, candidate_index in enumerate(frame_actions):
                node_candidate = frame_candidates[node_index][int(candidate_index)]
                frame_cache[node_index] = node_candidate[node_index]
            cache_configuration.validate_cache_shape_and_capacity(
                frame_cache, context.service_catalog, context.cache_nodes
            )

        # The production Micro actor is autoregressive over UAV IDs.
        observation = _observation(frame_cache, uav, step, context)
        current_trial = _copy_positions(uav)
        policy_uav_candidate_indices: list[int] = []
        uav_options_by_id: dict[str, list[dict[str, Any]]] = {}
        for uav_index, uav_id in enumerate(uav_ids):
            candidates, candidate_features = _uav_candidates(
                uav_index,
                current_trial,
                previous_uav,
                frame_cache,
                step,
                context,
            )
            uav_options_by_id[uav_id] = [
                {
                    "candidate_index": int(candidate_index),
                    "target_position": np.asarray(
                        candidate.target_position, dtype=np.float64
                    ).copy(),
                    "proposal_source": str(candidate.proposal_source),
                }
                for candidate_index, candidate in enumerate(candidates)
            ]
            condition = (
                np.concatenate(
                    [
                        _positions_condition(current_trial[prior_id], context)
                        for prior_id in uav_ids[:uav_index]
                    ]
                ).astype(np.float32, copy=False)
                if uav_index
                else np.zeros(int(policy.micro_condition_dim), dtype=np.float32)
            )
            sample = policy.sample_micro_head(
                observation,
                candidate_features,
                np.ones(len(candidates), dtype=bool),
                uav_index=uav_index,
                prior_uav_position_condition=condition,
                deterministic=True,
            )
            candidate_index = int(sample.selected_index)
            policy_uav_candidate_indices.append(candidate_index)
            uav_configuration.apply_uav_candidate(
                current_trial,
                previous_uav,
                uav_id,
                candidates[candidate_index].target_position,
                {key: 1 for key in uav_ids},
                context.cfg,
                context.flight_boundary,
            )
        uav_configuration.validate_uav_configuration_transition(
            previous_uav,
            current_trial,
            {key: 1 for key in uav_ids},
            context.cfg,
            context.flight_boundary,
        )

        if slot_index == frame_start:
            cache_options: list[dict[str, Any]] = [
                {
                    "label": "KEEP",
                    "cache_state": previous_cache.copy(),
                    "updated_node_ids": [],
                }
            ]
            seen = {previous_cache.tobytes()}
            for node_index, node_candidates in enumerate(frame_candidates):
                for candidate_index, candidate_cache in enumerate(node_candidates[1:], start=1):
                    target = previous_cache.copy()
                    target[node_index] = candidate_cache[node_index]
                    signature = target.tobytes()
                    if signature in seen:
                        continue
                    seen.add(signature)
                    cache_options.append(
                        {
                            "label": "SINGLE_NODE_UPDATE",
                            "cache_state": target,
                            "updated_node_ids": [
                                str(context.action_spec.cache_node_ids[node_index])
                            ],
                            "candidate_index": int(candidate_index),
                        }
                    )
            state_records.append(
                {
                    "episode_seed": int(episode_seed),
                    "slot_index": int(slot_index),
                    "slot": int(step.slot),
                    "time_s": float(step.time_s),
                    "step": step,
                    "aligned_steps": steps,
                    "cache_state": previous_cache,
                    "uav_positions": previous_uav,
                    "actual_policy_cache_state": frame_cache.copy(),
                    "actual_policy_cache_actions": frame_actions.copy(),
                    "actual_policy_uav_positions": _copy_positions(current_trial),
                    "actual_policy_uav_candidate_indices": policy_uav_candidate_indices,
                    "cache_options": cache_options,
                    "uav_options_by_id": uav_options_by_id,
                    "request_hit_metrics": _cache_request_metrics(
                        previous_cache,
                        step.actual_tasks,
                        int(context.service_catalog["num_services"]),
                    ),
                    "task_count": len(step.actual_tasks),
                    "context": context,
                }
            )

        if include_slot_states:
            slot_state_records.append(
                {
                    "episode_seed": int(episode_seed),
                    "slot_index": int(slot_index),
                    "slot": int(step.slot),
                    "time_s": float(step.time_s),
                    "step": step,
                    "cache_state": previous_cache.copy(),
                    "uav_positions": _copy_positions(previous_uav),
                    "actual_policy_cache_state": frame_cache.copy(),
                    "actual_policy_uav_positions": _copy_positions(current_trial),
                    "context": context,
                }
            )

        # State advancement is deterministic and independent of Matching/KKT:
        # policy observations contain realized configuration and saved forecasts,
        # while current tasks only enter the subsequent execution/evaluation.
        cache = frame_cache.copy()
        uav = _copy_positions(current_trial)
        frame_elapsed += 1
        done = slot_index + 1 >= len(steps)
        end_frame = done or frame_elapsed >= max_holding
        if not end_frame and frame_elapsed >= min_holding:
            next_observation = _observation(cache, uav, steps[slot_index + 1], context)
            termination = policy.sample_termination(
                next_observation,
                frame_elapsed,
                deterministic=True,
            )
            end_frame = int(termination.selected_index) == 1
        if end_frame:
            frame_start = slot_index + 1
            frame_elapsed = 0
            frame_cache = cache.copy()

    if include_slot_states:
        return state_records, slot_state_records
    return state_records


def _scene_signature(record: dict[str, Any], context: Any) -> dict[str, Any]:
    tasks = record["step"].actual_tasks
    service_count = int(context.service_catalog["num_services"])
    histogram = np.bincount(
        [int(task.service_id) for task in tasks], minlength=service_count
    ).astype(np.float64)
    if histogram.sum() > 0.0:
        histogram /= histogram.sum()
    positions = record["step"].actual_vehicle_positions
    task_positions = [
        np.asarray(positions[str(task.vehicle_id)], dtype=np.float64)
        for task in tasks
        if str(task.vehicle_id) in positions
    ]
    centroid = (
        np.mean(np.stack(task_positions), axis=0)
        if task_positions
        else np.zeros(2, dtype=np.float64)
    )
    x_min, y_min, x_max, y_max = context.flight_boundary
    diagonal = max(math.hypot(x_max - x_min, y_max - y_min), 1.0)
    uav_points = [
        np.asarray(record["uav_positions"][str(uav_id)], dtype=np.float64)
        for uav_id in context.action_spec.uav_ids
    ]
    uav_centroid = (
        np.mean(np.stack(uav_points), axis=0)
        if uav_points
        else np.zeros(2, dtype=np.float64)
    )
    return {
        "service_histogram": histogram,
        "centroid_normalized": centroid / diagonal,
        "uav_centroid_normalized": uav_centroid / diagonal,
        "time_bucket": int(record["slot"] // 30),
    }


def _choose_low_and_controls(
    records: list[dict[str, Any]],
    sample_count: int,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    valid = [record for record in records if int(record["task_count"]) > 0]
    if not valid:
        raise ValueError("No Cache-decision states have arrived tasks.")
    hit_rates = np.asarray(
        [record["request_hit_metrics"]["request_any_edge_cache_hit_rate"] for record in valid],
        dtype=np.float64,
    )
    threshold = float(np.quantile(hit_rates, 0.25))
    low_pool = [
        record
        for record in valid
        if float(record["request_hit_metrics"]["request_any_edge_cache_hit_rate"])
        <= threshold + HIT_TOLERANCE
    ]
    high_pool = [
        record
        for record in valid
        if float(record["request_hit_metrics"]["request_any_edge_cache_hit_rate"])
        > threshold + HIT_TOLERANCE
    ]

    low_by_episode: dict[int, list[dict[str, Any]]] = {}
    for record in low_pool:
        low_by_episode.setdefault(int(record["episode_seed"]), []).append(record)
    selected_low: list[dict[str, Any]] = []
    for episode_seed in sorted(low_by_episode):
        best = min(
            low_by_episode[episode_seed],
            key=lambda item: (
                float(item["request_hit_metrics"]["request_any_edge_cache_hit_rate"]),
                int(item["task_count"]),
                int(item["slot"]),
            ),
        )
        selected_low.append(best)
    selected_low.sort(
        key=lambda item: (
            float(item["request_hit_metrics"]["request_any_edge_cache_hit_rate"]),
            int(item["episode_seed"]),
            int(item["slot"]),
        )
    )
    if len(selected_low) < sample_count:
        already = {(item["episode_seed"], item["slot"]) for item in selected_low}
        remainder = sorted(
            (
                item
                for item in low_pool
                if (item["episode_seed"], item["slot"]) not in already
            ),
            key=lambda item: (
                float(item["request_hit_metrics"]["request_any_edge_cache_hit_rate"]),
                int(item["task_count"]),
                int(item["episode_seed"]),
                int(item["slot"]),
            ),
        )
        selected_low.extend(remainder[: sample_count - len(selected_low)])
    selected_low = selected_low[:sample_count]

    chosen_controls: list[dict[str, Any]] = []
    remaining = list(high_pool)
    signatures = {id(record): _scene_signature(record, record["context"]) for record in records}
    for low in selected_low:
        if not remaining:
            break
        low_sig = signatures[id(low)]

        def score(candidate: dict[str, Any]) -> tuple[float, ...]:
            candidate_sig = signatures[id(candidate)]
            same_episode = int(candidate["episode_seed"] != low["episode_seed"])
            same_time_bucket = int(candidate_sig["time_bucket"] != low_sig["time_bucket"])
            service_distance = float(
                np.abs(
                    candidate_sig["service_histogram"] - low_sig["service_histogram"]
                ).sum()
            )
            position_distance = float(
                np.linalg.norm(
                    candidate_sig["centroid_normalized"]
                    - low_sig["centroid_normalized"]
                )
            )
            uav_distance = float(
                np.linalg.norm(
                    candidate_sig["uav_centroid_normalized"]
                    - low_sig["uav_centroid_normalized"]
                )
            )
            return (
                float(same_episode),
                float(same_time_bucket),
                float(abs(int(candidate["task_count"]) - int(low["task_count"]))),
                service_distance,
                position_distance,
                uav_distance,
                -float(candidate["request_hit_metrics"]["request_any_edge_cache_hit_rate"]),
                float(candidate["slot"]),
            )

        selected = min(remaining, key=score)
        selected["matched_to"] = {
            "episode_seed": int(low["episode_seed"]),
            "slot": int(low["slot"]),
        }
        selected["match_criteria"] = {
            "same_episode": selected["episode_seed"] == low["episode_seed"],
            "same_30_slot_bucket": (
                signatures[id(selected)]["time_bucket"] == low_sig["time_bucket"]
            ),
            "task_count_difference": abs(
                int(selected["task_count"]) - int(low["task_count"])
            ),
            "service_histogram_l1": float(
                np.abs(
                    signatures[id(selected)]["service_histogram"]
                    - low_sig["service_histogram"]
                ).sum()
            ),
            "task_vehicle_centroid_distance_normalized": float(
                np.linalg.norm(
                    signatures[id(selected)]["centroid_normalized"]
                    - low_sig["centroid_normalized"]
                )
            ),
            "uav_centroid_distance_normalized": float(
                np.linalg.norm(
                    signatures[id(selected)]["uav_centroid_normalized"]
                    - low_sig["uav_centroid_normalized"]
                )
            ),
        }
        chosen_controls.append(selected)
        remaining.remove(selected)

    return threshold, selected_low, chosen_controls


def _feasible_targets(
    cache_state: np.ndarray,
    uav_positions: dict[str, np.ndarray],
    record: dict[str, Any],
    context: Any,
) -> dict[str, list[str]]:
    service_positions = task_allocation.build_service_node_positions(
        context.cache_nodes,
        record["step"].actual_vehicle_positions,
        uav_positions,
    )
    return task_allocation.build_feasible_targets(
        record["step"].actual_tasks,
        cache_state,
        context.cache_nodes,
        vehicle_positions=record["step"].actual_vehicle_positions,
        service_node_positions=service_positions,
        cache_config=context.cache_cfg,
        cloud_fallback_mode=context.cache_cfg.cloud_fallback_mode,
    )


def _evaluate_configuration(
    record: dict[str, Any],
    cache_state: np.ndarray,
    uav_positions: dict[str, np.ndarray],
    *,
    include_task_attributions: bool = False,
) -> dict[str, Any]:
    context = record["context"]
    previous_cache = np.asarray(record["cache_state"], dtype=np.int8)
    previous_uav = _copy_positions(record["uav_positions"])
    cache_updates = np.any(cache_state != previous_cache, axis=1).astype(np.int8)
    uav_triggers = {str(key): 1 for key in context.action_spec.uav_ids}
    result = slot_environment.execute_configured_slot(
        previous_cache,
        previous_uav,
        cache_state,
        uav_positions,
        record["step"].scenario_slice,
        record["step"].actual_tasks,
        record["step"].actual_vehicle_positions,
        context.service_catalog,
        context.cfg,
        context.cache_nodes,
        context.cache_cfg,
        context.flight_boundary,
        slot_id=int(record["slot"]),
        time_s=float(record["time_s"]),
    )
    if not result.step_feasible or result.system_utility is None:
        raise RuntimeError(
            "Official slot evaluation rejected a candidate: "
            f"episode_seed={record['episode_seed']}, slot={record['slot']}, "
            f"reason={result.infeasible_reason}"
        )
    expected_utility = (
        float(result.actual_task_utility)
        - float(result.cache_cost)
        - float(result.movement_cost)
    )
    if not math.isclose(
        float(result.system_utility), expected_utility, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise AssertionError("Official G != Q - cache_cost - movement_cost.")

    cache_cost_details = cache_configuration.cache_update_cost_details(
        previous_cache,
        cache_state,
        cache_updates,
        context.service_catalog["service_sizes_mb"],
        context.cache_nodes,
        context.cache_cfg,
    )
    if not math.isclose(
        float(cache_cost_details["cache_cost"]),
        float(result.cache_cost),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        float(cache_cost_details["cache_download_delay_s"]),
        float(result.cache_download_delay_s),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise AssertionError("Slot Cache cost differs from the formal cost helper.")

    task_metrics = task_metrics_from_result(result, context.cache_nodes)
    feasible_targets = _feasible_targets(cache_state, uav_positions, record, context)
    node_ids = [str(value) for value in context.cache_nodes["node_ids"]]
    target_by_task = result.matching
    matching_summary = result.matching_summary
    cloud_task_count = int(matching_summary["cloud_task_count"])
    if cloud_task_count != int(task_metrics.execution_target_counts["CLOUD"]):
        raise AssertionError("Cloud task count differs between Matching and task metrics.")
    cloud_values = {
        "cloud_uplink_allocated_bps": matching_summary[
            "cloud_uplink_allocated_bps"
        ],
        "cloud_uplink_capacity_bps": matching_summary[
            "cloud_uplink_capacity_bps"
        ],
        "cloud_uplink_capacity_violation_bps": matching_summary[
            "cloud_uplink_capacity_violation_bps"
        ],
        "cloud_cpu_allocated_cycles_per_s": matching_summary[
            "cloud_cpu_allocated_cycles_per_s"
        ],
        "cloud_cpu_capacity_cycles_per_s": matching_summary[
            "cloud_cpu_capacity_cycles_per_s"
        ],
        "cloud_cpu_capacity_violation_cycles_per_s": matching_summary[
            "cloud_cpu_capacity_violation_cycles_per_s"
        ],
    }
    if not bool(matching_summary["cloud_selected_resource_values_finite"]):
        raise FloatingPointError("Candidate Matching has non-finite selected Cloud resources.")
    if not all(_is_finite_number(value) for value in cloud_values.values()):
        raise FloatingPointError("Candidate Matching lacks finite Cloud capacity evidence.")
    if float(cloud_values["cloud_uplink_capacity_violation_bps"]) > CLOUD_UPLINK_TOLERANCE_BPS:
        raise RuntimeError("Candidate Matching exceeded the Cloud uplink capacity.")
    if float(cloud_values["cloud_cpu_capacity_violation_cycles_per_s"]) > CLOUD_CPU_TOLERANCE_CYCLES_PER_S:
        raise RuntimeError("Candidate Matching exceeded the Cloud CPU capacity.")
    cloud_kkt_solve_count = int(matching_summary["cloud_kkt_solve_count"])
    cloud_kkt_nonfinite_count = int(matching_summary["cloud_kkt_nonfinite_result_count"])
    if cloud_kkt_nonfinite_count != 0:
        raise FloatingPointError("Candidate Matching contains non-finite Cloud KKT results.")
    cloud_kkt_fields = {
        "cloud_kkt_max_uplink_allocated_bps": matching_summary[
            "cloud_kkt_max_uplink_allocated_bps"
        ],
        "cloud_kkt_max_cpu_allocated_cycles_per_s": matching_summary[
            "cloud_kkt_max_cpu_allocated_cycles_per_s"
        ],
        "cloud_kkt_max_uplink_capacity_violation_bps": matching_summary[
            "cloud_kkt_max_uplink_capacity_violation_bps"
        ],
        "cloud_kkt_max_cpu_capacity_violation_cycles_per_s": matching_summary[
            "cloud_kkt_max_cpu_capacity_violation_cycles_per_s"
        ],
    }
    if cloud_kkt_solve_count > 0:
        if not all(_is_finite_number(value) for value in cloud_kkt_fields.values()):
            raise FloatingPointError("Candidate Matching lacks finite Cloud KKT aggregates.")
        if float(cloud_kkt_fields["cloud_kkt_max_uplink_capacity_violation_bps"]) > CLOUD_UPLINK_TOLERANCE_BPS:
            raise RuntimeError("A candidate Cloud KKT result exceeded the uplink capacity.")
        if float(cloud_kkt_fields["cloud_kkt_max_cpu_capacity_violation_cycles_per_s"]) > CLOUD_CPU_TOLERANCE_CYCLES_PER_S:
            raise RuntimeError("A candidate Cloud KKT result exceeded the CPU capacity.")
    elif cloud_task_count > 0:
        raise RuntimeError("Cloud tasks were selected without a Cloud KKT allocation result.")

    any_edge_hits = 0
    selected_edge_hits = 0
    edge_tasks = 0
    feasible_edge_tasks = 0
    cloud_with_feasible_edge = 0
    cause_counts: Counter[str] = Counter()
    task_attributions: list[dict[str, Any]] = []
    matching_history = matching_summary.get("history", [])
    for task in record["step"].actual_tasks:
        service_id = int(task.service_id)
        any_edge = bool(np.any(cache_state[:, service_id] == 1))
        cached_node_ids = [
            node_ids[index]
            for index in range(len(node_ids))
            if int(cache_state[index, service_id]) == 1
        ]
        any_edge_hits += int(any_edge)
        feasible_edge_node_ids = [
            str(target)
            for target in feasible_targets.get(str(task.task_id), [])
            if target != task_allocation.CLOUD
        ]
        has_feasible_edge = bool(feasible_edge_node_ids)
        feasible_edge_tasks += int(has_feasible_edge)
        target = str(target_by_task[str(task.task_id)])
        if target == task_allocation.CLOUD:
            category = (
                "C_CACHED_EDGE_FEASIBLE_CLOUD_SELECTED"
                if any_edge and has_feasible_edge
                else (
                    "B_CACHED_EDGE_INFEASIBLE"
                    if any_edge
                    else "A_SERVICE_NOT_CACHED"
                )
            )
            cause_counts[category] += 1
            cloud_with_feasible_edge += int(has_feasible_edge)
        else:
            if target not in node_ids:
                cause_counts["INCONSISTENT_UNKNOWN_SELECTED_NODE"] += 1
                raise AssertionError(f"Matching selected unknown execution node {target!r}.")
            edge_tasks += 1
            node_index = node_ids.index(target)
            selected_hit = int(cache_state[node_index, service_id]) == 1
            if not selected_hit:
                raise AssertionError("Matching selected an Edge node without the service cached.")
            if target not in feasible_edge_node_ids:
                raise AssertionError("Matching selected an Edge node excluded by feasible-target construction.")
            selected_edge_hits += int(selected_hit)
            category = "D_CACHED_EDGE_EXECUTED"
            cause_counts[category] += 1

        if include_task_attributions:
            cloud_resources = matching_summary.get("resource", {})
            related_history = []
            if category == "C_CACHED_EDGE_FEASIBLE_CLOUD_SELECTED":
                related_history = [
                    {
                        "update_index": item.get("update_index"),
                        "operation_type": item.get("operation_type"),
                        "old_target": next(
                            (
                                str(target_value)
                                for task_id, target_value in zip(
                                    item.get("affected_tasks", []),
                                    item.get("old_targets", []),
                                    strict=False,
                                )
                                if str(task_id) == str(task.task_id)
                            ),
                            None,
                        ),
                        "new_target": next(
                            (
                                str(target_value)
                                for task_id, target_value in zip(
                                    item.get("affected_tasks", []),
                                    item.get("new_targets", []),
                                    strict=False,
                                )
                                if str(task_id) == str(task.task_id)
                            ),
                            None,
                        ),
                        "delta_task_utility_sum": item.get("delta_J"),
                    }
                    for item in matching_history
                    if str(task.task_id)
                    in {str(value) for value in item.get("affected_tasks", [])}
                ]
            attribution = {
                "episode_seed": int(record["episode_seed"]),
                "episode": record.get("episode"),
                "slot": int(record["slot"]),
                "task_id": str(task.task_id),
                "vehicle_id": str(task.vehicle_id),
                "service_id": service_id,
                "category": category,
                "service_cached_at_any_edge": any_edge,
                "cached_node_ids": cached_node_ids,
                "feasible_edge_node_ids": feasible_edge_node_ids,
                "selected_execution_node": target,
                "selected_task_utility": float(result.per_task_utility[str(task.task_id)]),
                "selected_task_delay_s": float(result.per_task_delay_s[str(task.task_id)]),
                "selected_vehicle_upload_energy_j": float(
                    result.per_task_vehicle_energy_j[str(task.task_id)]
                ),
            }
            if target == task_allocation.CLOUD:
                attribution["selected_cloud_cpu_cycles_per_s"] = cloud_resources.get(
                    "cloud_cpu_allocations_cycles_per_s", {}
                ).get(str(task.task_id))
                attribution["selected_cloud_uplink_bps"] = cloud_resources.get(
                    "cloud_uplink_rate_allocations_bps", {}
                ).get(str(task.task_id))
            else:
                attribution["selected_edge_cpu_fraction"] = (
                    float(result.cpu_allocations[target][str(task.task_id)])
                    if str(task.task_id) in result.cpu_allocations.get(target, {})
                    else None
                )
                bandwidth_fraction = result.bandwidth_allocations.get(
                    target, {}
                ).get(str(task.task_id))
                if bandwidth_fraction is not None:
                    attribution["selected_edge_bandwidth_fraction"] = float(
                        bandwidth_fraction
                    )
            if related_history:
                attribution["matching_accepted_operations_for_task"] = related_history
            task_attributions.append(attribution)

    task_count = len(record["step"].actual_tasks)
    execution_counts = task_metrics.execution_target_counts
    if sum(execution_counts.values()) != task_count:
        raise AssertionError("Matching execution targets do not cover the actual task set.")
    movement_distance = float(
        sum(
            np.linalg.norm(
                np.asarray(uav_positions[key], dtype=np.float64)
                - np.asarray(previous_uav[key], dtype=np.float64)
            )
            for key in previous_uav
        )
    )
    return {
        "G": float(result.system_utility),
        "task_utility": float(result.actual_task_utility),
        "cache_cost": float(result.cache_cost),
        "cache_download_delay_s": float(result.cache_download_delay_s),
        "cache_loaded_mb": float(np.sum(cache_cost_details["loaded_mb"])),
        "cache_loaded_mb_by_node": {
            str(node_id): float(cache_cost_details["loaded_mb"][index])
            for index, node_id in enumerate(context.cache_nodes["node_ids"])
        },
        "movement_cost": float(result.movement_cost),
        "movement_energy_j": float(result.movement_energy),
        "movement_distance_m": movement_distance,
        "task_count": int(task_count),
        "edge_task_count": int(task_count - cloud_task_count),
        "edge_task_rate": float((task_count - cloud_task_count) / task_count)
        if task_count
        else None,
        "cloud_task_count": cloud_task_count,
        "total_task_delay_s": float(task_metrics.total_task_delay_s),
        "mean_task_delay_s": (
            float(task_metrics.total_task_delay_s / task_count) if task_count else None
        ),
        "total_vehicle_upload_energy_j": float(task_metrics.total_vehicle_energy_j),
        "mean_vehicle_upload_energy_j": (
            float(task_metrics.total_vehicle_energy_j / task_count) if task_count else None
        ),
        "execution_target_counts": {
            key: int(value) for key, value in execution_counts.items()
        },
        "cloud_offload_rate": (
            float(execution_counts["CLOUD"] / task_count) if task_count else None
        ),
        "request_any_edge_cache_hit_rate": (
            float(any_edge_hits / task_count) if task_count else None
        ),
        "catalog_cache_coverage_rate": float(
            np.count_nonzero(np.any(cache_state == 1, axis=0))
            / max(int(context.service_catalog["num_services"]), 1)
        ),
        "selected_edge_cache_hit_rate": (
            float(selected_edge_hits / task_count) if task_count else None
        ),
        "selected_edge_cache_hit_given_edge_rate": (
            float(selected_edge_hits / edge_tasks) if edge_tasks else None
        ),
        "feasible_edge_path_rate": (
            float(feasible_edge_tasks / task_count) if task_count else None
        ),
        "cloud_with_feasible_edge_count": int(cloud_with_feasible_edge),
        "cause_counts": dict(cause_counts),
        "matching_resource_feasible": bool(
            matching_summary.get("resource_feasible", result.step_feasible)
        ),
        "matching_updates": int(matching_summary.get("updates", 0)),
        "matching_candidate_evaluations": int(
            matching_summary.get("candidate_evaluations_total", 0)
        ),
        "kkt_solve_count": int(matching_summary.get("KKT_SOLVE_COUNT", 0)),
        **cloud_values,
        "cloud_selected_resource_values_finite": True,
        "cloud_kkt_solve_count": cloud_kkt_solve_count,
        **cloud_kkt_fields,
        "cloud_kkt_nonfinite_result_count": cloud_kkt_nonfinite_count,
        "cache_update_nodes": int(np.count_nonzero(cache_updates)),
        "uav_move_count": sum(
            int(
                np.linalg.norm(
                    np.asarray(uav_positions[key], dtype=np.float64)
                    - np.asarray(previous_uav[key], dtype=np.float64)
                )
                > float(uav_configuration.uav_search_config_from_project(context.cfg).position_tolerance_m)
            )
            for key in previous_uav
        ),
        **({"task_attributions": task_attributions} if include_task_attributions else {}),
        **({"selected_target_by_task": dict(target_by_task)} if include_task_attributions else {}),
    }


def _evaluate_sample(
    record: dict[str, Any], progress: dict[str, int]
) -> dict[str, Any]:
    context = record["context"]
    pre_cache = np.asarray(record["cache_state"], dtype=np.int8).copy()
    pre_uav = _copy_positions(record["uav_positions"])
    policy_cache = np.asarray(record["actual_policy_cache_state"], dtype=np.int8).copy()
    policy_uav = _copy_positions(record["actual_policy_uav_positions"])

    # The baseline is the frozen policy's actual deterministic action at this state.
    baseline = _evaluate_configuration(record, policy_cache, policy_uav)
    keep_stay_reference = _evaluate_configuration(record, pre_cache, pre_uav)

    cache_candidate_results: list[dict[str, Any]] = [
        {
            "label": "POLICY_BASELINE",
            "updated_node_ids": [
                str(node_id)
                for node_id, changed in zip(
                    context.action_spec.cache_node_ids,
                    np.any(policy_cache != pre_cache, axis=1),
                    strict=True,
                )
                if bool(changed)
            ],
            "metrics": baseline,
            "cache_state": policy_cache.copy(),
        }
    ]
    seen_cache_states = {policy_cache.tobytes()}
    for candidate in record["cache_options"]:
        candidate_cache = np.asarray(candidate["cache_state"], dtype=np.int8).copy()
        signature = candidate_cache.tobytes()
        if signature in seen_cache_states:
            continue
        seen_cache_states.add(signature)
        metrics = _evaluate_configuration(record, candidate_cache, policy_uav)
        cache_candidate_results.append(
            {
                "label": candidate["label"],
                "updated_node_ids": list(candidate["updated_node_ids"]),
                "candidate_index": candidate.get("candidate_index"),
                "metrics": metrics,
                "cache_state": candidate_cache,
            }
        )
        progress["matching_kkt_candidate_evaluations_nonbaseline"] += 1
        if progress["matching_kkt_candidate_evaluations_nonbaseline"] % 100 == 0:
            print(
                "Evaluated "
                f"{progress['matching_kkt_candidate_evaluations_nonbaseline']} "
                "non-baseline Matching+KKT candidates...",
                flush=True,
            )

    best_cache = max(cache_candidate_results, key=lambda item: item["metrics"]["G"])

    uav_candidate_results: list[dict[str, Any]] = [
        {
            "label": "POLICY_BASELINE",
            "uav_id": None,
            "candidate_index": None,
            "proposal_source": "frozen_policy",
            "metrics": baseline,
            "uav_positions": _copy_positions(policy_uav),
        }
    ]
    seen_uav_positions = {
        tuple(
            float(value)
            for uav_id in context.action_spec.uav_ids
            for value in policy_uav[str(uav_id)]
        )
    }

    def add_uav_candidate(
        label: str,
        uav_id: str | None,
        candidate_index: int | None,
        proposal_source: str,
        trial: dict[str, np.ndarray],
        target_position: np.ndarray | None = None,
    ) -> None:
        signature = tuple(
            float(value)
            for candidate_uav_id in context.action_spec.uav_ids
            for value in trial[str(candidate_uav_id)]
        )
        if signature in seen_uav_positions:
            return
        triggers = {
            str(candidate_uav_id): int(
                np.linalg.norm(
                    trial[str(candidate_uav_id)] - pre_uav[str(candidate_uav_id)]
                )
                > float(
                    uav_configuration.uav_search_config_from_project(
                        context.cfg
                    ).position_tolerance_m
                )
            )
            for candidate_uav_id in context.action_spec.uav_ids
        }
        try:
            uav_configuration.validate_uav_configuration_transition(
                pre_uav,
                trial,
                triggers,
                context.cfg,
                context.flight_boundary,
            )
        except ValueError:
            progress["uav_candidates_rejected_by_joint_legality"] += 1
            return
        seen_uav_positions.add(signature)
        metrics = _evaluate_configuration(record, policy_cache, trial)
        uav_candidate_results.append(
            {
                "label": label,
                "uav_id": uav_id,
                "candidate_index": candidate_index,
                "proposal_source": proposal_source,
                "target_position": (
                    None
                    if target_position is None
                    else [float(value) for value in target_position]
                ),
                "metrics": metrics,
                "uav_positions": _copy_positions(trial),
            }
        )
        progress["matching_kkt_candidate_evaluations_nonbaseline"] += 1
        if progress["matching_kkt_candidate_evaluations_nonbaseline"] % 100 == 0:
            print(
                "Evaluated "
                f"{progress['matching_kkt_candidate_evaluations_nonbaseline']} "
                "non-baseline Matching+KKT candidates...",
                flush=True,
            )

    all_stay = _copy_positions(pre_uav)
    add_uav_candidate("ALL_STAY", None, None, "all_uavs_stay", all_stay)
    for uav_id, candidates in record["uav_options_by_id"].items():
        for candidate in candidates:
            trial = _copy_positions(policy_uav)
            target_position = np.asarray(
                candidate["target_position"], dtype=np.float64
            ).copy()
            trial[str(uav_id)] = target_position
            add_uav_candidate(
                "SINGLE_UAV_ACTION",
                str(uav_id),
                int(candidate["candidate_index"]),
                str(candidate["proposal_source"]),
                trial,
                target_position,
            )
    best_uav = max(uav_candidate_results, key=lambda item: item["metrics"]["G"])

    cache_delta = float(best_cache["metrics"]["G"] - baseline["G"])
    uav_delta = float(best_uav["metrics"]["G"] - baseline["G"])
    if cache_delta > HIT_TOLERANCE and cache_delta > uav_delta + HIT_TOLERANCE:
        best_intervention = "CACHE_ONLY"
    elif uav_delta > HIT_TOLERANCE and uav_delta > cache_delta + HIT_TOLERANCE:
        best_intervention = "UAV_ONLY"
    elif max(cache_delta, uav_delta) <= HIT_TOLERANCE:
        best_intervention = "NEITHER"
    else:
        best_intervention = "TIE"

    candidate_metric_fields = (
        "G",
        "task_utility",
        "cache_cost",
        "cache_download_delay_s",
        "cache_loaded_mb",
        "movement_cost",
        "movement_distance_m",
        "mean_task_delay_s",
        "total_task_delay_s",
        "total_vehicle_upload_energy_j",
        "execution_target_counts",
        "cloud_offload_rate",
        "edge_task_rate",
        "request_any_edge_cache_hit_rate",
        "selected_edge_cache_hit_rate",
        "feasible_edge_path_rate",
        "matching_resource_feasible",
        "cloud_task_count",
        "cloud_uplink_allocated_bps",
        "cloud_uplink_capacity_bps",
        "cloud_uplink_capacity_violation_bps",
        "cloud_cpu_allocated_cycles_per_s",
        "cloud_cpu_capacity_cycles_per_s",
        "cloud_cpu_capacity_violation_cycles_per_s",
        "cloud_selected_resource_values_finite",
        "cloud_kkt_solve_count",
        "cloud_kkt_max_uplink_allocated_bps",
        "cloud_kkt_max_cpu_allocated_cycles_per_s",
        "cloud_kkt_max_uplink_capacity_violation_bps",
        "cloud_kkt_max_cpu_capacity_violation_cycles_per_s",
        "cloud_kkt_nonfinite_result_count",
    )

    def public_candidates(
        items: list[dict[str, Any]], branch: str
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item in items:
            metrics = item["metrics"]
            record_out = {
                "label": item["label"],
                "delta_G_from_policy_baseline": float(metrics["G"] - baseline["G"]),
                "metrics": {key: metrics.get(key) for key in candidate_metric_fields},
            }
            if branch == "cache":
                record_out["updated_node_ids"] = item["updated_node_ids"]
                record_out["candidate_index"] = item.get("candidate_index")
            else:
                record_out.update(
                    {
                        "uav_id": item.get("uav_id"),
                        "candidate_index": item.get("candidate_index"),
                        "proposal_source": item.get("proposal_source"),
                        "target_position": item.get("target_position"),
                    }
                )
            output.append(record_out)
        return output

    cache_policy_details = cache_configuration.cache_update_cost_details(
        pre_cache,
        policy_cache,
        np.any(policy_cache != pre_cache, axis=1).astype(np.int8),
        context.service_catalog["service_sizes_mb"],
        context.cache_nodes,
        context.cache_cfg,
    )
    policy_uav_distances = {
        str(key): float(np.linalg.norm(policy_uav[str(key)] - pre_uav[str(key)]))
        for key in pre_uav
    }
    keep_with_policy_uav = next(
        (
            item["metrics"]
            for item in cache_candidate_results
            if item["label"] == "KEEP"
        ),
        baseline if np.array_equal(policy_cache, pre_cache) else None,
    )
    stay_with_policy_cache = next(
        (
            item["metrics"]
            for item in uav_candidate_results
            if item["label"] == "ALL_STAY"
        ),
        baseline
        if all(
            np.array_equal(policy_uav[str(key)], pre_uav[str(key)]) for key in pre_uav
        )
        else None,
    )
    progress["sample_count"] += 1
    print(
        f"Completed sample {progress['sample_count']}: episode={record['episode_seed']} "
        f"slot={record['slot']} cache_candidates={len(cache_candidate_results)} "
        f"uav_candidates={len(uav_candidate_results)}",
        flush=True,
    )
    return {
        "episode_seed": int(record["episode_seed"]),
        "slot": int(record["slot"]),
        "slot_index": int(record["slot_index"]),
        "task_count": int(record["task_count"]),
        "pre_intervention_hit_metrics": dict(record["request_hit_metrics"]),
        "threshold_hit_rate": float(
            record["request_hit_metrics"]["request_any_edge_cache_hit_rate"]
        ),
        "initial_cache_catalog_coverage_rate": float(
            record["request_hit_metrics"]["catalog_cache_coverage_rate"]
        ),
        "matching_control": record.get("match_criteria"),
        "matched_to": record.get("matched_to"),
        "baseline": baseline,
        "keep_stay_reference": keep_stay_reference,
        "frozen_policy_action_delta_G_vs_KEEP_STAY": float(
            baseline["G"] - keep_stay_reference["G"]
        ),
        "frozen_policy_cache_action_delta_G_vs_KEEP_same_policy_UAV": (
            None
            if keep_with_policy_uav is None
            else float(baseline["G"] - keep_with_policy_uav["G"])
        ),
        "frozen_policy_uav_action_delta_G_vs_ALL_STAY_same_policy_Cache": (
            None
            if stay_with_policy_cache is None
            else float(baseline["G"] - stay_with_policy_cache["G"])
        ),
        "cache_only": {
            "candidate_count": len(cache_candidate_results),
            "non_policy_candidates_evaluated": len(cache_candidate_results) - 1,
            "candidate_scope": (
                "frozen-policy cache configuration plus KEEP and every legal single-node "
                "UPDATE proposal; no multi-node UPDATE combinations"
            ),
            "best_candidate": {
                key: value
                for key, value in best_cache.items()
                if key not in {"cache_state"}
            },
            "delta_G": cache_delta,
            "all_candidates": public_candidates(cache_candidate_results, "cache"),
        },
        "uav_only": {
            "candidate_count": len(uav_candidate_results),
            "non_policy_candidates_evaluated": len(uav_candidate_results) - 1,
            "jointly_legal_candidates_rejected": int(
                progress["uav_candidates_rejected_by_joint_legality"]
            ),
            "candidate_scope": (
                "frozen-policy UAV configuration, all-UAV STAY, and each unique legal "
                "single-head STAY/MOVE proposal with other UAV positions held at the "
                "frozen-policy positions; no UAV Cartesian product"
            ),
            "best_candidate": {
                key: value
                for key, value in best_uav.items()
                if key not in {"uav_positions"}
            },
            "delta_G": uav_delta,
            "all_candidates": public_candidates(uav_candidate_results, "uav"),
        },
        "frozen_policy_action": {
            "cache_updated_node_ids": [
                str(node_id)
                for node_id, changed in zip(
                    context.action_spec.cache_node_ids,
                    np.any(policy_cache != pre_cache, axis=1),
                    strict=True,
                )
                if bool(changed)
            ],
            "cache_loaded_mb": float(np.sum(cache_policy_details["loaded_mb"])),
            "cache_cost": float(cache_policy_details["cache_cost"]),
            "cache_download_delay_s": float(
                cache_policy_details["cache_download_delay_s"]
            ),
            "uav_candidate_indices": list(record["actual_policy_uav_candidate_indices"]),
            "uav_move_distance_by_id_m": policy_uav_distances,
            "uav_total_move_distance_m": float(sum(policy_uav_distances.values())),
            "G": float(baseline["G"]),
            "delta_G_vs_KEEP_STAY": float(baseline["G"] - keep_stay_reference["G"]),
            "decision_inputs": "deterministic policy actions use the official observation and forecast candidate features",
            "compared_to_oracle_cache_node_difference_count": int(
                np.count_nonzero(
                    np.any(
                        policy_cache
                        != np.asarray(best_cache["cache_state"], dtype=np.int8),
                        axis=1,
                    )
                )
            ),
            "compared_to_oracle_uav_total_position_difference_m": float(
                sum(
                    np.linalg.norm(policy_uav[str(key)] - best_uav["uav_positions"][str(key)])
                    for key in pre_uav
                )
            ),
        },
        "best_single_intervention": best_intervention,
        "low_hit_causes": dict(baseline["cause_counts"]),
        "_ten_slot_cache_state": np.asarray(
            best_cache["cache_state"], dtype=np.int8
        ).copy(),
        "_ten_slot_uav_positions": _copy_positions(best_uav["uav_positions"]),
        "_policy_cache_state": policy_cache.copy(),
        "_policy_uav_positions": _copy_positions(policy_uav),
        "_source_record": record,
    }


def _scheme_summary(
    samples: list[dict[str, Any]], scheme: str, bootstrap_seed: int = 27005
) -> dict[str, Any]:
    deltas = np.asarray([float(item[scheme]["delta_G"]) for item in samples], dtype=np.float64)
    positive = int(np.count_nonzero(deltas > HIT_TOLERANCE))
    negative = int(np.count_nonzero(deltas < -HIT_TOLERANCE))
    zero = int(len(deltas) - positive - negative)
    metric_names = (
        "mean_task_delay_s",
        "total_task_delay_s",
        "total_vehicle_upload_energy_j",
        "cloud_offload_rate",
        "request_any_edge_cache_hit_rate",
        "selected_edge_cache_hit_rate",
        "feasible_edge_path_rate",
        "cache_cost",
        "cache_download_delay_s",
        "cache_loaded_mb",
        "movement_cost",
        "movement_distance_m",
        "task_utility",
        "edge_task_rate",
    )
    changes: dict[str, dict[str, float | None]] = {}
    for metric in metric_names:
        difference: list[float] = []
        for item in samples:
            before = item["baseline"].get(metric)
            after = item[scheme]["best_candidate"]["metrics"].get(metric)
            if before is not None and after is not None:
                difference.append(float(after) - float(before))
        changes[metric] = {
            "mean": float(np.mean(difference)) if difference else None,
            "median": float(np.median(difference)) if difference else None,
        }
    return {
        "positive_delta_G_count": positive,
        "zero_delta_G_count": zero,
        "negative_delta_G_count": negative,
        "positive_delta_G_rate": float(positive / len(deltas)) if deltas.size else None,
        "mean_delta_G": float(np.mean(deltas)) if deltas.size else None,
        "median_delta_G": float(np.median(deltas)) if deltas.size else None,
        "mean_delta_G_bootstrap_95_ci": _bootstrap_mean_ci(
            deltas, bootstrap_seed
        ),
        "mean_change_by_metric": changes,
    }


def _candidate_level_summary(
    samples: list[dict[str, Any]],
    scheme: str,
) -> dict[str, Any]:
    """Summarize legal single-node or single-head proposals against policy action."""

    deltas = np.asarray(
        [
            float(candidate["delta_G_from_policy_baseline"])
            for sample in samples
            for candidate in sample[scheme]["all_candidates"]
            if candidate["label"]
            == ("SINGLE_NODE_UPDATE" if scheme == "cache_only" else "SINGLE_UAV_ACTION")
        ],
        dtype=np.float64,
    )
    positive = int(np.count_nonzero(deltas > HIT_TOLERANCE))
    negative = int(np.count_nonzero(deltas < -HIT_TOLERANCE))
    zero = int(deltas.size - positive - negative)
    return {
        "candidate_count": int(deltas.size),
        "positive_delta_G_count": positive,
        "zero_delta_G_count": zero,
        "negative_delta_G_count": negative,
        "mean_delta_G": float(np.mean(deltas)) if deltas.size else None,
        "median_delta_G": float(np.median(deltas)) if deltas.size else None,
        "minimum_delta_G": float(np.min(deltas)) if deltas.size else None,
        "maximum_delta_G": float(np.max(deltas)) if deltas.size else None,
        "selection_boundary": "all legal enumerated candidates compared with frozen-policy baseline",
    }


def _bootstrap_mean_ci(
    values: list[float] | np.ndarray,
    seed: int,
    iterations: int = 4000,
) -> dict[str, Any]:
    sample = np.asarray(values, dtype=np.float64)
    if sample.size == 0:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    if not np.all(np.isfinite(sample)):
        raise ValueError("Bootstrap input contains non-finite values.")
    if sample.size == 1:
        mean = float(sample[0])
        return {"n": 1, "mean": mean, "ci95_low": mean, "ci95_high": mean}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, sample.size, size=(iterations, sample.size))
    means = np.mean(sample[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "n": int(sample.size),
        "mean": float(np.mean(sample)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _sample_frozen_policy_transition(
    policy: Any,
    context: Any,
    step: Any,
    cache: np.ndarray,
    uav: dict[str, np.ndarray],
    previous_uav: dict[str, np.ndarray],
    tensor_spec: PrimitiveTensorSpec,
    choose_macro: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply one deterministic production-policy transition to a replay state."""

    next_cache = np.asarray(cache, dtype=np.int8).copy()
    if choose_macro:
        observation = _observation(next_cache, uav, step, context)
        candidates, features, masks = _cache_node_candidates(
            next_cache, uav, step, context, tensor_spec
        )
        selected = policy.sample_macro_nodes(
            observation, features, masks, deterministic=True
        )
        actions = np.asarray(selected.cache_actions, dtype=np.int64).reshape(-1)
        if actions.shape != (len(candidates),):
            raise AssertionError("Frozen Macro actions do not align with Cache nodes.")
        for node_index, candidate_index in enumerate(actions):
            next_cache[node_index] = candidates[node_index][int(candidate_index)][node_index]
        cache_configuration.validate_cache_shape_and_capacity(
            next_cache, context.service_catalog, context.cache_nodes
        )

    uav_ids = tuple(str(value) for value in context.action_spec.uav_ids)
    observation = _observation(next_cache, uav, step, context)
    next_uav = _copy_positions(uav)
    for uav_index, uav_id in enumerate(uav_ids):
        candidates, candidate_features = _uav_candidates(
            uav_index,
            next_uav,
            previous_uav,
            next_cache,
            step,
            context,
        )
        condition = (
            np.concatenate(
                [
                    _positions_condition(next_uav[prior_id], context)
                    for prior_id in uav_ids[:uav_index]
                ]
            ).astype(np.float32, copy=False)
            if uav_index
            else np.zeros(int(policy.micro_condition_dim), dtype=np.float32)
        )
        sample = policy.sample_micro_head(
            observation,
            candidate_features,
            np.ones(len(candidates), dtype=bool),
            uav_index=uav_index,
            prior_uav_position_condition=condition,
            deterministic=True,
        )
        uav_configuration.apply_uav_candidate(
            next_uav,
            previous_uav,
            uav_id,
            candidates[int(sample.selected_index)].target_position,
            {key: 1 for key in uav_ids},
            context.cfg,
            context.flight_boundary,
        )
    uav_configuration.validate_uav_configuration_transition(
        previous_uav,
        next_uav,
        {key: 1 for key in uav_ids},
        context.cfg,
        context.flight_boundary,
    )
    return next_cache, next_uav


def _replay_ten_slot_branch(
    record: dict[str, Any],
    policy: Any,
    first_cache: np.ndarray,
    first_uav: dict[str, np.ndarray],
    slot_count: int = 10,
) -> dict[str, Any]:
    """Replay one branch against the same saved exogenous Training steps."""

    context = record["context"]
    start_index = int(record["slot_index"])
    steps = record["aligned_steps"]
    if start_index + slot_count > len(steps):
        raise ValueError("The selected state does not have a complete 10 Slot window.")
    tensor_spec = PrimitiveTensorSpec.create(context)
    cache = np.asarray(record["cache_state"], dtype=np.int8).copy()
    uav = _copy_positions(record["uav_positions"])
    frame_elapsed = 0
    rows: list[dict[str, Any]] = []

    for offset in range(slot_count):
        step = steps[start_index + offset]
        previous_cache = cache.copy()
        previous_uav = _copy_positions(uav)
        runtime_record = dict(record)
        runtime_record.update(
            {
                "step": step,
                "slot": int(step.slot),
                "time_s": float(step.time_s),
                "cache_state": previous_cache,
                "uav_positions": previous_uav,
            }
        )
        if offset == 0:
            next_cache = np.asarray(first_cache, dtype=np.int8).copy()
            next_uav = _copy_positions(first_uav)
        else:
            next_cache, next_uav = _sample_frozen_policy_transition(
                policy,
                context,
                step,
                cache,
                uav,
                previous_uav,
                tensor_spec,
                choose_macro=(frame_elapsed == 0),
            )
        metrics = _evaluate_configuration(runtime_record, next_cache, next_uav)
        pre_hit = _cache_request_metrics(
            previous_cache,
            step.actual_tasks,
            int(context.service_catalog["num_services"]),
        )
        rows.append(
            {
                "slot": int(step.slot),
                "task_count": int(len(step.actual_tasks)),
                "pre_action_cache_hit_rate": pre_hit["request_any_edge_cache_hit_rate"],
                "metrics": metrics,
            }
        )
        cache = next_cache.copy()
        uav = _copy_positions(next_uav)

        frame_elapsed += 1
        is_last = offset + 1 >= slot_count or start_index + offset + 1 >= len(steps)
        end_frame = is_last or frame_elapsed >= int(policy.adaptive_holding_max_slots)
        if not end_frame and frame_elapsed >= int(policy.adaptive_holding_min_slots):
            next_step = steps[start_index + offset + 1]
            next_observation = _observation(cache, uav, next_step, context)
            termination = policy.sample_termination(
                next_observation, frame_elapsed, deterministic=True
            )
            end_frame = int(termination.selected_index) == 1
        if end_frame:
            frame_elapsed = 0

    total_tasks = sum(row["task_count"] for row in rows)
    total_delay = sum(float(row["metrics"]["total_task_delay_s"]) for row in rows)
    total_energy = sum(
        float(row["metrics"]["total_vehicle_upload_energy_j"]) for row in rows
    )
    total_cloud = sum(int(row["metrics"]["cloud_task_count"]) for row in rows)
    pre_hit_numerator = sum(
        float(row["pre_action_cache_hit_rate"] or 0.0) * row["task_count"]
        for row in rows
    )
    post_hit_numerator = sum(
        float(row["metrics"]["request_any_edge_cache_hit_rate"] or 0.0)
        * row["task_count"]
        for row in rows
    )
    return {
        "slot_count": len(rows),
        "slot_indices": [row["slot"] for row in rows],
        "cumulative_reward": float(sum(row["metrics"]["G"] for row in rows)),
        "cumulative_task_utility": float(
            sum(row["metrics"]["task_utility"] for row in rows)
        ),
        "cumulative_cache_cost": float(
            sum(row["metrics"]["cache_cost"] for row in rows)
        ),
        "cumulative_movement_cost": float(
            sum(row["metrics"]["movement_cost"] for row in rows)
        ),
        "cumulative_cache_download_delay_s": float(
            sum(row["metrics"]["cache_download_delay_s"] for row in rows)
        ),
        "cumulative_cache_loaded_mb": float(
            sum(row["metrics"]["cache_loaded_mb"] for row in rows)
        ),
        "cumulative_movement_distance_m": float(
            sum(row["metrics"]["movement_distance_m"] for row in rows)
        ),
        "mean_task_delay_s": total_delay / total_tasks if total_tasks else None,
        "total_vehicle_upload_energy_j": total_energy,
        "cloud_task_count": total_cloud,
        "cloud_offload_rate": total_cloud / total_tasks if total_tasks else None,
        "pre_action_cache_hit_rate_task_weighted": (
            pre_hit_numerator / total_tasks if total_tasks else None
        ),
        "post_action_cache_hit_rate_task_weighted": (
            post_hit_numerator / total_tasks if total_tasks else None
        ),
        "task_count": total_tasks,
        "rows": rows,
    }


def _paired_trajectory_effect(
    baseline: dict[str, Any], intervention: dict[str, Any]
) -> dict[str, Any]:
    resource_evidence_fields = (
        "cloud_task_count",
        "cloud_uplink_allocated_bps",
        "cloud_uplink_capacity_bps",
        "cloud_uplink_capacity_violation_bps",
        "cloud_cpu_allocated_cycles_per_s",
        "cloud_cpu_capacity_cycles_per_s",
        "cloud_cpu_capacity_violation_cycles_per_s",
        "cloud_selected_resource_values_finite",
        "cloud_kkt_solve_count",
        "cloud_kkt_max_uplink_allocated_bps",
        "cloud_kkt_max_cpu_allocated_cycles_per_s",
        "cloud_kkt_max_uplink_capacity_violation_bps",
        "cloud_kkt_max_cpu_capacity_violation_cycles_per_s",
        "cloud_kkt_nonfinite_result_count",
    )

    def slot_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "slot": int(row["slot"]),
                "G": float(row["metrics"]["G"]),
                "task_utility": float(row["metrics"]["task_utility"]),
                "cache_cost": float(row["metrics"]["cache_cost"]),
                "movement_cost": float(row["metrics"]["movement_cost"]),
                "mean_task_delay_s": row["metrics"]["mean_task_delay_s"],
                "total_vehicle_upload_energy_j": float(
                    row["metrics"]["total_vehicle_upload_energy_j"]
                ),
                **{
                    field: row["metrics"].get(field)
                    for field in resource_evidence_fields
                },
            }
            for row in rows
        ]

    return {
        "first_slot_delta_G": float(
            intervention["rows"][0]["metrics"]["G"]
            - baseline["rows"][0]["metrics"]["G"]
        ),
        "cumulative_10_slot_reward_delta_G": float(
            intervention["cumulative_reward"] - baseline["cumulative_reward"]
        ),
        "baseline": {key: value for key, value in baseline.items() if key != "rows"},
        "intervention": {
            key: value for key, value in intervention.items() if key != "rows"
        },
        "baseline_slot_evidence": slot_evidence(baseline["rows"]),
        "intervention_slot_evidence": slot_evidence(intervention["rows"]),
        "post_action_cache_hit_rate_change": (
            None
            if baseline["post_action_cache_hit_rate_task_weighted"] is None
            or intervention["post_action_cache_hit_rate_task_weighted"] is None
            else float(
                intervention["post_action_cache_hit_rate_task_weighted"]
                - baseline["post_action_cache_hit_rate_task_weighted"]
            )
        ),
        "cumulative_cache_cost_change": float(
            intervention["cumulative_cache_cost"] - baseline["cumulative_cache_cost"]
        ),
        "cumulative_movement_cost_change": float(
            intervention["cumulative_movement_cost"]
            - baseline["cumulative_movement_cost"]
        ),
        "mean_task_delay_change_s": (
            None
            if baseline["mean_task_delay_s"] is None
            or intervention["mean_task_delay_s"] is None
            else float(intervention["mean_task_delay_s"] - baseline["mean_task_delay_s"])
        ),
        "upload_energy_change_j": float(
            intervention["total_vehicle_upload_energy_j"]
            - baseline["total_vehicle_upload_energy_j"]
        ),
        "cloud_offload_rate_change": (
            None
            if baseline["cloud_offload_rate"] is None
            or intervention["cloud_offload_rate"] is None
            else float(intervention["cloud_offload_rate"] - baseline["cloud_offload_rate"])
        ),
    }


def _policy_action_summary(samples: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    values = np.asarray(
        [float(item["frozen_policy_action_delta_G_vs_KEEP_STAY"]) for item in samples],
        dtype=np.float64,
    )
    return {
        "n": int(values.size),
        "mean_delta_G_vs_KEEP_STAY": float(np.mean(values)) if values.size else None,
        "median_delta_G_vs_KEEP_STAY": float(np.median(values)) if values.size else None,
        "positive_count": int(np.count_nonzero(values > HIT_TOLERANCE)),
        "zero_count": int(np.count_nonzero(np.abs(values) <= HIT_TOLERANCE)),
        "negative_count": int(np.count_nonzero(values < -HIT_TOLERANCE)),
        "mean_delta_G_bootstrap_95_ci": _bootstrap_mean_ci(values, seed),
    }


def _paired_low_control_contrast(
    low_samples: list[dict[str, Any]],
    control_samples: list[dict[str, Any]],
    scheme: str,
    seed: int,
) -> dict[str, Any]:
    low_by_key = {
        (int(item["episode_seed"]), int(item["slot"])): item for item in low_samples
    }
    control_by_low_key = {
        (
            int(item["matched_to"]["episode_seed"]),
            int(item["matched_to"]["slot"]),
        ): item
        for item in control_samples
        if item.get("matched_to")
    }
    differences: list[float] = []
    pair_details: list[dict[str, Any]] = []
    for key, low in sorted(low_by_key.items()):
        control = control_by_low_key.get(key)
        if control is None:
            continue
        low_delta = float(low[scheme]["delta_G"])
        control_delta = float(control[scheme]["delta_G"])
        difference = low_delta - control_delta
        differences.append(difference)
        pair_details.append(
            {
                "low_hit_slot": {"episode_seed": key[0], "slot": key[1]},
                "control_slot": {
                    "episode_seed": int(control["episode_seed"]),
                    "slot": int(control["slot"]),
                },
                "low_hit_delta_G": low_delta,
                "matched_control_delta_G": control_delta,
                "low_minus_control_delta_G": difference,
            }
        )
    ci = _bootstrap_mean_ci(differences, seed)
    return {
        "n_matched_pairs": len(differences),
        "mean_low_minus_control_delta_G": ci["mean"],
        "median_low_minus_control_delta_G": (
            float(np.median(differences)) if differences else None
        ),
        "ci95_low": ci["ci95_low"],
        "ci95_high": ci["ci95_high"],
        "positive_pair_count": int(
            sum(value > HIT_TOLERANCE for value in differences)
        ),
        "zero_pair_count": int(
            sum(abs(value) <= HIT_TOLERANCE for value in differences)
        ),
        "negative_pair_count": int(
            sum(value < -HIT_TOLERANCE for value in differences)
        ),
        "pairs": pair_details,
    }


def _classify_low_hit_trigger(
    contrast: dict[str, Any],
    low_summary: dict[str, Any],
    control_summary: dict[str, Any],
) -> str:
    n = int(contrast["n_matched_pairs"])
    low_n = int(low_summary["positive_delta_G_count"]) + int(
        low_summary["zero_delta_G_count"]
    ) + int(low_summary["negative_delta_G_count"])
    if n < 5 or low_n < 5:
        return "INCONCLUSIVE"
    low_ci = low_summary["mean_delta_G_bootstrap_95_ci"]
    if (
        contrast["ci95_low"] is not None
        and float(contrast["ci95_low"]) > HIT_TOLERANCE
        and low_ci["ci95_low"] is not None
        and float(low_ci["ci95_low"]) > HIT_TOLERANCE
    ):
        control_ci = control_summary["mean_delta_G_bootstrap_95_ci"]
        control_n = int(control_summary["positive_delta_G_count"]) + int(
            control_summary["zero_delta_G_count"]
        ) + int(control_summary["negative_delta_G_count"])
        if (
            control_n > 0
            and control_summary["positive_delta_G_rate"] >= 0.8
            and control_ci["ci95_low"] is not None
            and float(control_ci["ci95_low"]) > HIT_TOLERANCE
        ):
            return "INCONCLUSIVE"
        return "SUPPORTED"
    if contrast["ci95_high"] is not None and float(contrast["ci95_high"]) <= HIT_TOLERANCE:
        return "NOT_SUPPORTED"
    return "INCONCLUSIVE"


def _representative_pairs(
    low_samples: list[dict[str, Any]],
    control_samples: list[dict[str, Any]],
    pair_count: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    def source(item: dict[str, Any]) -> dict[str, Any]:
        return item.get("_source_record", item)

    def hit_rate(item: dict[str, Any]) -> float:
        if "threshold_hit_rate" in item:
            return float(item["threshold_hit_rate"])
        return float(
            item["request_hit_metrics"]["request_any_edge_cache_hit_rate"]
        )

    low_by_key = {
        (int(item["episode_seed"]), int(item["slot"])): item for item in low_samples
    }
    control_by_low_key = {
        (
            int(item["matched_to"]["episode_seed"]),
            int(item["matched_to"]["slot"]),
        ): item
        for item in control_samples
        if item.get("matched_to")
    }
    eligible: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for key, low in low_by_key.items():
        control = control_by_low_key.get(key)
        if control is None:
            continue
        if int(low["slot_index"]) + 10 <= len(source(low)["aligned_steps"]) and int(
            control["slot_index"]
        ) + 10 <= len(source(control)["aligned_steps"]):
            eligible.append((low, control))
    eligible.sort(
        key=lambda pair: (
            hit_rate(pair[0]),
            int(pair[0]["task_count"]),
            int(pair[0]["episode_seed"]),
            int(pair[0]["slot"]),
        )
    )
    if len(eligible) <= pair_count:
        return eligible
    indices = sorted(set([0, (len(eligible) - 1) // 2, len(eligible) - 1]))
    chosen = [eligible[index] for index in indices[:pair_count]]
    if len(chosen) < pair_count:
        chosen_keys = {
            (int(pair[0]["episode_seed"]), int(pair[0]["slot"])) for pair in chosen
        }
        chosen.extend(
            pair
            for pair in eligible
            if (int(pair[0]["episode_seed"]), int(pair[0]["slot"])) not in chosen_keys
        )
    return chosen[:pair_count]


def _run_ten_slot_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    policy: Any,
) -> dict[str, Any]:
    paired_runs: list[dict[str, Any]] = []
    for low, control in pairs:
        pair_result: dict[str, Any] = {
            "low_hit_slot": {"episode_seed": low["episode_seed"], "slot": low["slot"]},
            "control_slot": {
                "episode_seed": control["episode_seed"],
                "slot": control["slot"],
            },
            "match_criteria": control.get("match_criteria"),
        }
        for group, sample in (("LOW_HIT", low), ("MATCHED_CONTROL", control)):
            source = sample["_source_record"]
            baseline = _replay_ten_slot_branch(
                source,
                policy,
                sample["_policy_cache_state"],
                sample["_policy_uav_positions"],
            )
            cache_branch = _replay_ten_slot_branch(
                source,
                policy,
                sample["_ten_slot_cache_state"],
                sample["_policy_uav_positions"],
            )
            uav_branch = _replay_ten_slot_branch(
                source,
                policy,
                sample["_policy_cache_state"],
                sample["_ten_slot_uav_positions"],
            )
            pair_result[group] = {
                "cache_only": _paired_trajectory_effect(baseline, cache_branch),
                "uav_only": _paired_trajectory_effect(baseline, uav_branch),
            }
        paired_runs.append(pair_result)

    output: dict[str, Any] = {
        "slot_count": 10,
        "representative_matched_pair_count": len(paired_runs),
        "paired_runs": paired_runs,
        "group_summary": {},
        "low_minus_control_cumulative_reward_contrast": {},
    }
    for group in ("LOW_HIT", "MATCHED_CONTROL"):
        output["group_summary"][group] = {}
        for scheme in ("cache_only", "uav_only"):
            effects = [
                pair[group][scheme]["cumulative_10_slot_reward_delta_G"]
                for pair in paired_runs
            ]
            first = [pair[group][scheme]["first_slot_delta_G"] for pair in paired_runs]
            output["group_summary"][group][scheme] = {
                "n": len(effects),
                "first_slot_mean_delta_G": float(np.mean(first)) if first else None,
                "first_slot_median_delta_G": float(np.median(first)) if first else None,
                "first_slot_mean_delta_G_bootstrap_95_ci": _bootstrap_mean_ci(
                    first, 34001 if group == "LOW_HIT" else 34002
                ),
                "cumulative_10_slot_mean_delta_G": float(np.mean(effects))
                if effects
                else None,
                "cumulative_10_slot_median_delta_G": float(np.median(effects))
                if effects
                else None,
                "cumulative_10_slot_mean_delta_G_bootstrap_95_ci": _bootstrap_mean_ci(
                    effects, 35001 if group == "LOW_HIT" else 35002
                ),
            }
    for scheme in ("cache_only", "uav_only"):
        differences = [
            pair["LOW_HIT"][scheme]["cumulative_10_slot_reward_delta_G"]
            - pair["MATCHED_CONTROL"][scheme]["cumulative_10_slot_reward_delta_G"]
            for pair in paired_runs
        ]
        output["low_minus_control_cumulative_reward_contrast"][scheme] = (
            _bootstrap_mean_ci(differences, 36001)
            | {
                "median": float(np.median(differences)) if differences else None,
                "positive_pair_count": int(
                    sum(value > HIT_TOLERANCE for value in differences)
                ),
                "zero_pair_count": int(
                    sum(abs(value) <= HIT_TOLERANCE for value in differences)
                ),
                "negative_pair_count": int(
                    sum(value < -HIT_TOLERANCE for value in differences)
                ),
            }
        )
    return output


def _select_and_evaluate(
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = (PROJECT_ROOT / args.run_dir).resolve()
    pool_dir = (PROJECT_ROOT / args.episode_pool).resolve()
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "run_summary.json"
    checkpoint_path = run_dir / "checkpoints" / "latest_checkpoint.pt"
    run_config_path = run_dir / "run_config.toml"
    slot_metrics_path = run_dir / "slot_metrics.jsonl"
    episode_metrics_path = run_dir / "episode_metrics.jsonl"
    update_metrics_path = run_dir / "update_metrics.jsonl"
    critic_metrics_path = run_dir / "critic_diagnostics.jsonl"
    official_config_path = PROJECT_ROOT / "configs/proposed/alternating_actor_updates_v1.toml"
    for path in (
        manifest_path,
        summary_path,
        checkpoint_path,
        run_config_path,
        slot_metrics_path,
        episode_metrics_path,
        update_metrics_path,
        critic_metrics_path,
        official_config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required existing Run artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("completed_episodes", -1)) != 8:
        raise ValueError("The controlled Run summary does not report exactly 8 Episodes.")
    if int(summary.get("policy_update_count", -1)) != 2:
        raise ValueError("The controlled Run summary does not report exactly 2 Policy Updates.")
    if summary.get("training_completion_status") != "CONTROLLED_STOP_AFTER_UPDATES":
        raise ValueError("The Run is not marked CONTROLLED_STOP_AFTER_UPDATES.")
    raw_hash = project_config.raw_config_sha256(run_config_path)
    official_raw_hash = project_config.raw_config_sha256(official_config_path)
    if raw_hash != manifest.get("config_raw_sha256") or raw_hash != official_raw_hash:
        raise ValueError("Run config snapshot raw hash does not match its Run Manifest.")
    if int(manifest.get("target_episodes", -1)) != 300 or int(
        summary.get("target_episodes", -1)
    ) != 300:
        raise ValueError("Controlled Run target is not 300 Episodes.")
    if int(manifest.get("controlled_stop_after_policy_updates", -1)) != 2 or int(
        summary.get("controlled_stop_after_policy_updates", -1)
    ) != 2:
        raise ValueError("Controlled stop limit is not exactly two Policy Updates.")
    if manifest.get("actor_update_mode") != "ALTERNATING" or manifest.get(
        "actor_update_first"
    ) != "MICRO":
        raise ValueError("Run Manifest does not identify Alternating Actor V1 from Micro.")

    pool_identity = Path(str(manifest.get("episode_pool", ""))).resolve()
    if pool_identity != pool_dir:
        raise ValueError("Run Manifest Training Pool identity differs from requested Pool.")
    pool_provenance = manifest.get("episode_pool_provenance")
    if not isinstance(pool_provenance, dict) or int(pool_provenance.get("episode_count", -1)) != 300:
        raise ValueError("Run Manifest does not identify the expected 300-Episode Pool.")
    pool_manifest_path = pool_dir / "formal_pool_manifest.json"
    if not pool_manifest_path.is_file():
        raise FileNotFoundError("Training Pool manifest is unavailable for identity check.")
    pool_manifest_sha256 = hashlib.sha256(pool_manifest_path.read_bytes()).hexdigest()
    expected_pool_hash = pool_provenance.get("manifest_sha256")
    if (
        pool_manifest_sha256 != expected_pool_hash
        or pool_manifest_sha256 != manifest.get("training_pool_manifest_sha256")
    ):
        raise ValueError("Current Training Pool manifest differs from the Run identity.")

    episode_rows = [
        json.loads(line)
        for line in episode_metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episode_seeds = [int(row["episode_seed"]) for row in episode_rows]
    if len(episode_seeds) != 8 or episode_seeds != list(range(20001, 20009)):
        raise ValueError("Run Episode seed sequence does not match 20001–20008.")
    update_rows = [
        json.loads(line)
        for line in update_metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    critic_rows = [
        json.loads(line)
        for line in critic_metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(update_rows) != 2 or len(critic_rows) != 2:
        raise ValueError("Run does not contain exactly two update and critic records.")
    expected_actor_schedule = ("MICRO", "MACRO")
    for index, (update_row, expected_actor) in enumerate(
        zip(update_rows, expected_actor_schedule, strict=True)
    ):
        if (
            int(update_row.get("policy_update_index", -1)) != index
            or update_row.get("active_actor") != expected_actor
            or bool(update_row.get("micro_actor_update_enabled"))
            != (expected_actor == "MICRO")
            or bool(update_row.get("macro_actor_update_enabled"))
            != (expected_actor == "MACRO")
        ):
            raise ValueError(f"Update group {index + 1} has an unexpected actor schedule.")
        inactive_steps = int(
            update_row.get(
                "macro_actor_optimizer_steps_this_update"
                if expected_actor == "MICRO"
                else "micro_actor_optimizer_steps_this_update",
                -1,
            )
        )
        if inactive_steps != 0:
            raise ValueError(f"Inactive {('Macro' if expected_actor == 'MICRO' else 'Micro')} Actor stepped.")
        for field in (
            "macro_critic_optimizer_steps_this_update",
            "micro_critic_optimizer_steps_this_update",
            "long_value_optimizer_steps_this_update",
        ):
            if int(update_row.get(field, 0)) <= 0:
                raise ValueError(f"Critic update evidence is missing/zero for {field}.")
    for seed in episode_seeds:
        if not (pool_dir / f"episode_{seed}").is_dir():
            raise FileNotFoundError(f"Training realization episode_{seed} is missing.")

    reference_context = interface.load_environment_context(
        str(run_config_path), realization_dir=pool_dir / f"episode_{episode_seeds[0]}"
    )
    semantic_hash = project_config.semantic_config_hash(reference_context.cfg)
    if (
        semantic_hash != manifest.get("config_semantic_sha256")
        or project_config.raw_config_sha256(run_config_path) != official_raw_hash
    ):
        raise ValueError("Run config semantic hash does not match its Run Manifest.")
    trainer = PPOTrainer(reference_context, policy_update_total=75, device=args.device)
    checkpoint_payload = trainer.load_checkpoint(
        checkpoint_path,
        config_raw_sha256=raw_hash,
        config_semantic_sha256=semantic_hash,
        episode_pool=str(pool_dir),
        restore_rng=False,
    )
    counters = checkpoint_payload.get("training_counters", {})
    if int(counters.get("completed_episodes", -1)) != 8:
        raise ValueError("Checkpoint completed Episode count is not 8.")
    if int(counters.get("policy_update_count", -1)) != 2:
        raise ValueError("Checkpoint Policy Update count is not 2.")
    if int(counters.get("macro_update_count", -1)) != 2 or int(
        counters.get("micro_update_count", -1)
    ) != 2:
        raise ValueError("Checkpoint Critic/update-group counters are inconsistent.")
    if checkpoint_payload.get("actor_update_mode") != "ALTERNATING" or checkpoint_payload.get(
        "actor_update_schedule_state", {}
    ).get("next_active_actor") != "MICRO":
        raise ValueError("Checkpoint does not resume the expected next Micro Actor group.")
    trainer.policy.eval()

    cloud_resource_audit = _audit_cloud_slot_metrics(slot_metrics_path, 2400)
    episode_return_checks: list[dict[str, Any]] = []
    for row in episode_rows:
        episode_number = int(row["episode"])
        logged_return = float(row["episode_return"])
        slot_sum = float(cloud_resource_audit["slot_reward_sum_by_episode"][str(episode_number)])
        if not math.isclose(logged_return, slot_sum, rel_tol=1.0e-10, abs_tol=1.0e-8):
            raise ValueError(
                f"Episode {episode_number} Return differs from its 300 Slot reward sum."
            )
        episode_return_checks.append(
            {
                "episode": episode_number,
                "episode_seed": int(row["episode_seed"]),
                "slot_count": 300,
                "episode_return": logged_return,
                "slot_reward_sum": slot_sum,
                "absolute_error": abs(logged_return - slot_sum),
            }
        )

    replayed_states: list[dict[str, Any]] = []
    with torch.inference_mode():
        for seed in episode_seeds:
            context = interface.load_environment_context(
                str(run_config_path), realization_dir=pool_dir / f"episode_{seed}"
            )
            interface._assert_realization_context_compatible(reference_context, context)
            episode_states = _macro_decision_states(trainer.policy, context, seed)
            replayed_states.extend(episode_states)
            print(
                f"Replayed frozen policy states for Training episode seed {seed}: "
                f"{len(episode_states)} Cache decision states",
                flush=True,
            )

    threshold, low_states, control_states = _choose_low_and_controls(
        replayed_states, int(args.sample_count)
    )
    representative_state_pairs = _representative_pairs(
        low_states, control_states, int(args.ten_slot_pairs)
    )
    representative_state_pair_keys = [
        (
            (int(low["episode_seed"]), int(low["slot"])),
            (int(control["episode_seed"]), int(control["slot"])),
        )
        for low, control in representative_state_pairs
    ]
    if len(control_states) < len(low_states):
        print(
            "Warning: fewer high-hit matched controls than low-hit samples are available.",
            flush=True,
        )

    progress = {
        "matching_kkt_candidate_evaluations_nonbaseline": 0,
        "sample_count": 0,
        "uav_candidates_rejected_by_joint_legality": 0,
    }
    analyzed: list[dict[str, Any]] = []
    for group, records in (("LOW_HIT", low_states), ("MATCHED_CONTROL", control_states)):
        for record in records:
            outcome = _evaluate_sample(record, progress)
            outcome["sample_group"] = group
            analyzed.append(outcome)

    low_results = [item for item in analyzed if item["sample_group"] == "LOW_HIT"]
    controls = [item for item in analyzed if item["sample_group"] == "MATCHED_CONTROL"]
    cause_counts: Counter[str] = Counter()
    for sample in low_results:
        cause_counts.update(sample["low_hit_causes"])

    best_by_cause: dict[str, dict[str, int]] = {}
    for sample in low_results:
        dominant_cause = (
            max(sample["low_hit_causes"].items(), key=lambda pair: pair[1])[0]
            if sample["low_hit_causes"]
            else "no_tasks_or_unresolved"
        )
        best_by_cause.setdefault(dominant_cause, {})
        selected = sample["best_single_intervention"]
        best_by_cause[dominant_cause][selected] = (
            best_by_cause[dominant_cause].get(selected, 0) + 1
        )

    low_cache_summary = _scheme_summary(low_results, "cache_only", 27011)
    low_uav_summary = _scheme_summary(low_results, "uav_only", 27012)
    control_cache_summary = _scheme_summary(controls, "cache_only", 27111)
    control_uav_summary = _scheme_summary(controls, "uav_only", 27112)
    low_cache_contrast = _paired_low_control_contrast(
        low_results, controls, "cache_only", 27051
    )
    low_uav_contrast = _paired_low_control_contrast(
        low_results, controls, "uav_only", 27052
    )
    low_outcomes_by_key = {
        (int(item["episode_seed"]), int(item["slot"])): item for item in low_results
    }
    control_outcomes_by_key = {
        (int(item["episode_seed"]), int(item["slot"])): item for item in controls
    }
    representative_pairs = [
        (low_outcomes_by_key[low_key], control_outcomes_by_key[control_key])
        for low_key, control_key in representative_state_pair_keys
    ]
    with torch.inference_mode():
        ten_slot_results = _run_ten_slot_pairs(representative_pairs, trainer.policy)

    def policy_component_summary(
        samples: list[dict[str, Any]], field: str, seed: int
    ) -> dict[str, Any]:
        values = np.asarray(
            [float(item[field]) for item in samples if item.get(field) is not None],
            dtype=np.float64,
        )
        return {
            "n": int(values.size),
            "mean_delta_G": float(np.mean(values)) if values.size else None,
            "median_delta_G": float(np.median(values)) if values.size else None,
            "positive_count": int(np.count_nonzero(values > HIT_TOLERANCE)),
            "nonpositive_count": int(np.count_nonzero(values <= HIT_TOLERANCE)),
            "mean_delta_G_bootstrap_95_ci": _bootstrap_mean_ci(values, seed),
        }

    public_samples: list[dict[str, Any]] = []
    for item in analyzed:
        public = dict(item)
        for key in (
            "_ten_slot_cache_state",
            "_ten_slot_uav_positions",
            "_policy_cache_state",
            "_policy_uav_positions",
            "_source_record",
        ):
            public.pop(key, None)
        public_samples.append(public)

    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return _json_value(
        {
            "metadata": {
                "study": "paired low-cache-hit Cache-only versus UAV-only offline counterfactual",
                "run_dir": run_dir,
                "training_pool": pool_dir,
                "training_episode_seeds_replayed": episode_seeds,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_episode": int(counters["completed_episodes"]),
                "checkpoint_policy_updates": int(counters["policy_update_count"]),
                "checkpoint_actor_update_mode": checkpoint_payload.get("actor_update_mode"),
                "checkpoint_schedule_state": checkpoint_payload.get("actor_update_schedule_state"),
                "policy_evaluation": "deterministic torch.inference_mode frozen inference; no optimizer step and no checkpoint write",
                "bootstrap_method": "percentile bootstrap of the mean with 4000 resamples and fixed per-summary seeds",
                "training_pool_episode_count": 300,
                "training_pool_manifest_sha256": pool_manifest_sha256,
                "run_config_raw_sha256": raw_hash,
                "official_config_raw_sha256": official_raw_hash,
                "run_config_semantic_sha256": semantic_hash,
                "matching_candidate_method": "production execute_configured_slot; fresh formal Matching + shared Cloud KKT/CPU allocation for each candidate and replay Slot",
                "full_training_pool_replayed": False,
                "validation_test_accessed": False,
                "training_executed": False,
                "formal_algorithm_modified": False,
            },
            "run_preflight": {
                "controlled_stop_status": summary["training_completion_status"],
                "target_episodes": int(summary["target_episodes"]),
                "completed_episodes": int(summary["completed_episodes"]),
                "controlled_stop_after_policy_updates": int(summary["controlled_stop_after_policy_updates"]),
                "controlled_stop_completed_policy_updates": int(summary["controlled_stop_completed_policy_updates"]),
                "completed_policy_updates": int(summary["policy_update_count"]),
                "config_raw_sha256_run_manifest_official_match": True,
                "config_semantic_sha256_manifest_match": True,
                "training_pool_manifest_sha256_current_run_match": True,
                "checkpoint_strict_identity_load": True,
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_completed_episodes": int(counters["completed_episodes"]),
                "checkpoint_policy_update_count": int(counters["policy_update_count"]),
                "checkpoint_next_active_actor": "MICRO",
                "slot_metrics_audit": cloud_resource_audit,
                "episode_return_slot_reward_checks": episode_return_checks,
                "actor_update_records": [
                    {
                        "policy_update_index": int(row["policy_update_index"]),
                        "active_actor": row["active_actor"],
                        "macro_actor_update_enabled": bool(row["macro_actor_update_enabled"]),
                        "micro_actor_update_enabled": bool(row["micro_actor_update_enabled"]),
                        "macro_actor_optimizer_steps_this_update": int(row["macro_actor_optimizer_steps_this_update"]),
                        "micro_actor_optimizer_steps_this_update": int(row["micro_actor_optimizer_steps_this_update"]),
                        "macro_critic_optimizer_steps_this_update": int(row["macro_critic_optimizer_steps_this_update"]),
                        "micro_critic_optimizer_steps_this_update": int(row["micro_critic_optimizer_steps_this_update"]),
                        "long_value_optimizer_steps_this_update": int(row["long_value_optimizer_steps_this_update"]),
                    }
                    for row in update_rows
                ],
                "critic_diagnostic_record_count": len(critic_rows),
            },
            "definitions": {
                "primary_cache_hit_rate": "For each actual task in a macro Cache-decision state, count a hit if its requested service exists in at least one SV/RSU/UAV edge cache; divide by actual task count. This is service availability and is not the selected execution-node hit rate.",
                "selected_edge_cache_hit_rate": "Tasks matched to an edge node whose selected node stores the requested service, divided by all actual tasks; the conditional-on-edge rate is also reported.",
                "cache_hit_metric_source": "RECOMPUTED_FROM_FROZEN_REPLAY; Run slot logs do not contain request hit fields",
                "cache_loaded_mb_source": "FORMAL_CACHE_UPDATE_COST_HELPER_FROM_POSITIVE_CACHE_DIFFERENCE",
                "cache_delay_and_cost_source": "DIRECT_SLOT_EXECUTION_RESULT; CROSS_CHECKED_AGAINST_FORMAL_CACHE_COST_HELPER",
                "feasible_edge_path_rate": "Tasks with at least one cached edge target accepted by the production path/role/rate feasibility check, divided by actual task count.",
                "low_hit_threshold": "Empirical 25th percentile of primary request-any-edge hit rates across valid macro Cache-decision states (at least one task) in the eight saved Training realizations; preliminary sample threshold, not a general optimum.",
                "threshold_state_timing": "Hit rate is computed from the pre-action Cache at a legal Macro decision boundary, before the current Cache proposal is applied.",
                "baseline": "The frozen policy's actual deterministic Cache and UAV actions at the decision state; KEEP+STAY is retained as a reference.",
                "cache_only_candidate_scope": "Frozen-policy Cache action plus every legal single-node UPDATE proposal, with frozen-policy UAV positions fixed; no multi-node combination.",
                "uav_only_candidate_scope": "Frozen-policy UAV action, all-UAV STAY and legal single-head proposals, with Cache fixed to frozen-policy Cache; no multi-UAV combination.",
                "oracle_boundary": "Actual current tasks rank candidate G; policy observations use forecasts and configuration, not actual_tasks. Candidate maxima are offline oracle upper bounds within the enumerated subsets.",
                "ten_slot_pairing": "All branches start from the same pre-action cache/UAV snapshot and replay the same saved aligned Training steps; after the first intervention, the deterministic frozen policy resumes and states evolve through the official Slot executor.",
                "task_energy": "Vehicle upload/transmission energy from the formal per-task vehicle-energy outputs.",
                "service_activation": "Formal current behavior applies candidate cache state immediately; download is charged as a reward reconfiguration penalty only. No transfer queue or across-slot download completion is simulated.",
            },
            "cloud_resource_log_audit": cloud_resource_audit,
            "state_distribution": {
                "cache_decision_states_scanned": len(replayed_states),
                "valid_cache_decision_states_with_tasks": sum(
                    int(item["task_count"] > 0) for item in replayed_states
                ),
                "threshold_p25": threshold,
                "threshold_method": "empirical P25 over valid Macro Cache-decision states with actual tasks; fixed before candidate evaluation",
                "hit_rate_min": float(
                    min(
                        float(item["request_hit_metrics"]["request_any_edge_cache_hit_rate"])
                        for item in replayed_states
                        if item["task_count"] > 0
                    )
                ),
                "hit_rate_median": float(
                    np.median(
                        [
                            float(item["request_hit_metrics"]["request_any_edge_cache_hit_rate"])
                            for item in replayed_states
                            if item["task_count"] > 0
                        ]
                    )
                ),
                "hit_rate_max": float(
                    max(
                        float(item["request_hit_metrics"]["request_any_edge_cache_hit_rate"])
                        for item in replayed_states
                        if item["task_count"] > 0
                    )
                ),
                "low_hit_sample_count": len(low_results),
                "matched_control_count": len(controls),
                "requested_sample_count_per_group": int(args.sample_count),
            },
            "low_hit_summary": {
                "cache_only": low_cache_summary,
                "uav_only": low_uav_summary,
                "cache_update_candidate_scores": _candidate_level_summary(
                    low_results, "cache_only"
                ),
                "uav_move_candidate_scores": _candidate_level_summary(
                    low_results, "uav_only"
                ),
                "frozen_policy_cache_action_vs_keep": policy_component_summary(
                    low_results,
                    "frozen_policy_cache_action_delta_G_vs_KEEP_same_policy_UAV",
                    27013,
                ),
                "frozen_policy_uav_action_vs_stay": policy_component_summary(
                    low_results,
                    "frozen_policy_uav_action_delta_G_vs_ALL_STAY_same_policy_Cache",
                    27014,
                ),
                "primary_cause_task_counts": dict(cause_counts),
                "best_intervention_by_dominant_cause": best_by_cause,
                "best_intervention_counts": dict(
                    Counter(item["best_single_intervention"] for item in low_results)
                ),
                "best_cache_candidate_label_counts": dict(
                    Counter(
                        item["cache_only"]["best_candidate"]["label"]
                        for item in low_results
                    )
                ),
                "best_uav_candidate_label_counts": dict(
                    Counter(
                        item["uav_only"]["best_candidate"]["label"]
                        for item in low_results
                    )
                ),
            },
            "matched_control_summary": {
                "cache_only": control_cache_summary,
                "uav_only": control_uav_summary,
                "cache_update_candidate_scores": _candidate_level_summary(
                    controls, "cache_only"
                ),
                "uav_move_candidate_scores": _candidate_level_summary(
                    controls, "uav_only"
                ),
                "frozen_policy_cache_action_vs_keep": policy_component_summary(
                    controls,
                    "frozen_policy_cache_action_delta_G_vs_KEEP_same_policy_UAV",
                    27113,
                ),
                "frozen_policy_uav_action_vs_stay": policy_component_summary(
                    controls,
                    "frozen_policy_uav_action_delta_G_vs_ALL_STAY_same_policy_Cache",
                    27114,
                ),
                "best_intervention_counts": dict(
                    Counter(item["best_single_intervention"] for item in controls)
                ),
                "best_cache_candidate_label_counts": dict(
                    Counter(
                        item["cache_only"]["best_candidate"]["label"]
                        for item in controls
                    )
                ),
                "best_uav_candidate_label_counts": dict(
                    Counter(
                        item["uav_only"]["best_candidate"]["label"]
                        for item in controls
                    )
                ),
            },
            "paired_low_minus_control_immediate_oracle_gain": {
                "cache_only": low_cache_contrast,
                "uav_only": low_uav_contrast,
            },
            "low_hit_trigger_assessment": {
                "cache_only": _classify_low_hit_trigger(
                    low_cache_contrast, low_cache_summary, control_cache_summary
                ),
                "uav_only": _classify_low_hit_trigger(
                    low_uav_contrast, low_uav_summary, control_uav_summary
                ),
                "decision_rule": "SUPPORTED only when n>=5, low-group oracle mean 95% CI is above zero, and matched low-minus-control oracle gain 95% CI is above zero, unless the control group also has >=80% positive gains and a positive lower 95% CI (then INCONCLUSIVE as a sufficient low-hit trigger); NOT_SUPPORTED when the contrast upper CI is at/below zero; otherwise INCONCLUSIVE.",
                "deployable_low_hit_trigger": "NOT_IMPLEMENTED",
            },
            "ten_slot_paired_replay": ten_slot_results,
            "best_intervention_counts_all_samples": dict(
                Counter(item["best_single_intervention"] for item in analyzed)
            ),
            "both_interventions_nonpositive_count": sum(
                int(
                    float(item["cache_only"]["delta_G"]) <= HIT_TOLERANCE
                    and float(item["uav_only"]["delta_G"]) <= HIT_TOLERANCE
                )
                for item in analyzed
            ),
            "samples": public_samples,
            "matching_kkt_candidate_evaluations_nonbaseline": progress[
                "matching_kkt_candidate_evaluations_nonbaseline"
            ],
            "uav_candidates_rejected_by_joint_legality": progress[
                "uav_candidates_rejected_by_joint_legality"
            ],
            "limitations": [
                "The checkpoint is an early Alternating V1 model after 8 Episodes and 2 Policy Updates; results are not evidence about a converged/final policy.",
                "The frozen policy was replayed on the same eight Training realizations used before this checkpoint, so this is an in-sample Training diagnostic.",
                "Only single-node Cache updates and one-UAV-at-a-time moves were enumerated; no joint Cache+UAV actions, multi-node updates, or simultaneous multi-UAV combinations were tested.",
                "Oracle ranking uses realized current tasks, absent from actor observation; best scores are offline counterfactual upper bounds only within the enumerated subsets.",
                "The current actor's deterministic forecast-driven actions are measured separately against KEEP/STAY references; there is no low-hit-conditioned trigger or prediction-based candidate-value selector in the formal algorithm.",
                "The study uses macro Cache-decision boundary states so Cache interventions respect Adaptive Holding; it does not include intermediate held slots as Cache action opportunities.",
                "Ten-Slot replay is limited to the representative matched pairs reported and is exploratory rather than a population estimate.",
                "Formal service activation remains immediate after an UPDATE and does not simulate an actual download queue or cross-slot completion process.",
            ],
        }
    )


def _forced_edge_task_counterfactual(
    record: dict[str, Any],
    task_attribution: dict[str, Any],
    edge_node_id: str,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Force one legal task to an Edge node and re-optimize the full Slot."""

    context = record["context"]
    tasks = record["step"].actual_tasks
    task_id = str(task_attribution["task_id"])
    service_positions = task_allocation.build_service_node_positions(
        context.cache_nodes,
        record["step"].actual_vehicle_positions,
        record["actual_policy_uav_positions"],
    )
    try:
        result = task_allocation.optimize_task_matching(
            tasks,
            record["step"].actual_vehicle_positions,
            service_positions,
            record["actual_policy_cache_state"],
            context.cfg,
            context.cache_nodes,
            context.cache_cfg,
            cloud_fallback_mode=context.cache_cfg.cloud_fallback_mode,
            fixed_matching={task_id: str(edge_node_id)},
            mutable_task_ids={
                str(task.task_id) for task in tasks if str(task.task_id) != task_id
            },
        )
        if not result.resource_feasible:
            return {
                "episode_seed": int(record["episode_seed"]),
                "episode": int(record["episode"]),
                "slot": int(record["slot"]),
                "task_id": task_id,
                "counterfactual_status": "INFEASIBLE",
                "forced_edge_node": str(edge_node_id),
                "reason": result.infeasible_reason,
            }
        if len(result.matching) != len(tasks):
            raise AssertionError("Forced-Edge Matching did not cover the full Slot task set.")
        matching_summary = slot_environment._matching_summary(
            result, context.cache_nodes, context.cache_cfg
        )
        delay_values = [float(value) for value in result.per_task_delay_s.values()]
        energy_values = [
            float(value) for value in result.per_task_vehicle_energy_j.values()
        ]
        cloud_count = sum(
            target == task_allocation.CLOUD for target in result.matching.values()
        )
        task_count = len(tasks)
        cache_cost = float(baseline["cache_cost"])
        movement_cost = float(baseline["movement_cost"])
        system_utility = float(result.task_utility_sum) - cache_cost - movement_cost
        expected_baseline_g = (
            float(baseline["task_utility"])
            - cache_cost
            - movement_cost
        )
        if not math.isclose(
            float(baseline["G"]), expected_baseline_g, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise AssertionError("Counterfactual baseline violates the formal G identity.")
        if task_id not in result.matching or result.matching[task_id] != str(edge_node_id):
            raise AssertionError("Forced-Edge task did not remain at its fixed target.")
        baseline_target = str(baseline["selected_target_by_task"][task_id])
        baseline_task_utility = float(task_attribution["selected_task_utility"])
        baseline_task_delay = float(task_attribution["selected_task_delay_s"])
        baseline_task_energy = float(
            task_attribution["selected_vehicle_upload_energy_j"]
        )
        forced_task_utility = float(result.per_task_utility[task_id])
        forced_task_delay = float(result.per_task_delay_s[task_id])
        forced_task_energy = float(result.per_task_vehicle_energy_j[task_id])
        changed_assignments = sum(
            str(target) != str(baseline["selected_target_by_task"].get(other_id))
            for other_id, target in result.matching.items()
        )
        related_history = [
            item
            for item in matching_summary.get("history", [])
            if task_id
            in {str(value) for value in item.get("affected_tasks", [])}
        ]
        return {
            "episode_seed": int(record["episode_seed"]),
            "episode": int(record["episode"]),
            "slot": int(record["slot"]),
            "task_id": task_id,
            "service_id": int(task_attribution["service_id"]),
            "counterfactual_status": "EVALUATED",
            "pairing": "same Slot tasks, vehicle positions, active Cache and UAV state",
            "matching_path": "production optimize_task_matching with fixed_matching for one task; all other Slot tasks remain mutable",
            "baseline_target": baseline_target,
            "forced_edge_node": str(edge_node_id),
            "matching_resource_feasible": bool(result.resource_feasible),
            "full_slot_task_count": task_count,
            "changed_task_assignments_after_rematching": int(changed_assignments),
            "baseline_G": float(baseline["G"]),
            "forced_edge_G": float(system_utility),
            "delta_G_forced_edge_minus_cloud_baseline": float(
                system_utility - float(baseline["G"])
            ),
            "baseline_task_utility_sum": float(baseline["task_utility"]),
            "forced_edge_task_utility_sum": float(result.task_utility_sum),
            "baseline_cache_cost": cache_cost,
            "forced_edge_cache_cost": cache_cost,
            "baseline_movement_cost": movement_cost,
            "forced_edge_movement_cost": movement_cost,
            "baseline_total_task_delay_s": float(baseline["total_task_delay_s"]),
            "forced_edge_total_task_delay_s": float(math.fsum(delay_values)),
            "delta_total_task_delay_s": float(
                math.fsum(delay_values) - float(baseline["total_task_delay_s"])
            ),
            "baseline_mean_task_delay_s": baseline.get("mean_task_delay_s"),
            "forced_edge_mean_task_delay_s": (
                float(math.fsum(delay_values) / task_count) if task_count else None
            ),
            "baseline_total_vehicle_upload_energy_j": float(
                baseline["total_vehicle_upload_energy_j"]
            ),
            "forced_edge_total_vehicle_upload_energy_j": float(math.fsum(energy_values)),
            "delta_total_vehicle_upload_energy_j": float(
                math.fsum(energy_values)
                - float(baseline["total_vehicle_upload_energy_j"])
            ),
            "baseline_cloud_task_count": int(baseline["cloud_task_count"]),
            "forced_edge_cloud_task_count": int(cloud_count),
            "baseline_cloud_offload_rate": baseline.get("cloud_offload_rate"),
            "forced_edge_cloud_offload_rate": (
                float(cloud_count / task_count) if task_count else None
            ),
            "baseline_forced_task_utility": baseline_task_utility,
            "forced_edge_task_utility": forced_task_utility,
            "delta_forced_task_utility": float(
                forced_task_utility - baseline_task_utility
            ),
            "baseline_forced_task_delay_s": baseline_task_delay,
            "forced_edge_task_delay_s": forced_task_delay,
            "baseline_forced_task_upload_energy_j": baseline_task_energy,
            "forced_edge_task_upload_energy_j": forced_task_energy,
            "cloud_uplink_allocated_bps": matching_summary[
                "cloud_uplink_allocated_bps"
            ],
            "cloud_cpu_allocated_cycles_per_s": matching_summary[
                "cloud_cpu_allocated_cycles_per_s"
            ],
            "cloud_uplink_capacity_violation_bps": matching_summary[
                "cloud_uplink_capacity_violation_bps"
            ],
            "cloud_cpu_capacity_violation_cycles_per_s": matching_summary[
                "cloud_cpu_capacity_violation_cycles_per_s"
            ],
            "matching_accepted_operations_for_forced_task": related_history,
        }
    except Exception as exc:
        return {
            "episode_seed": int(record["episode_seed"]),
            "episode": int(record["episode"]),
            "slot": int(record["slot"]),
            "task_id": task_id,
            "counterfactual_status": "UNRESOLVED",
            "forced_edge_node": str(edge_node_id),
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _task_attribution_study(args: argparse.Namespace) -> dict[str, Any]:
    """Attribute every Training task using the current Run's strict identity."""

    run_dir = (PROJECT_ROOT / args.run_dir).resolve()
    pool_dir = (PROJECT_ROOT / args.episode_pool).resolve()
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "run_summary.json"
    run_config_path = run_dir / "run_config.toml"
    checkpoint_path = run_dir / "checkpoints" / "latest_checkpoint.pt"
    slot_metrics_path = run_dir / "slot_metrics.jsonl"
    episode_metrics_path = run_dir / "episode_metrics.jsonl"
    official_config_path = PROJECT_ROOT / "configs/proposed/alternating_actor_updates_v1.toml"
    required_paths = (
        manifest_path,
        summary_path,
        run_config_path,
        checkpoint_path,
        slot_metrics_path,
        episode_metrics_path,
        official_config_path,
        pool_dir / "formal_pool_manifest.json",
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required existing study input is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("training_completion_status") != "CONTROLLED_STOP_AFTER_UPDATES"
        or int(summary.get("target_episodes", -1)) != 300
        or int(summary.get("completed_episodes", -1)) != 8
        or int(summary.get("controlled_stop_after_policy_updates", -1)) != 2
        or int(summary.get("policy_update_count", -1)) != 2
    ):
        raise ValueError("The designated Run does not satisfy its controlled-stop identity.")
    raw_hash = project_config.raw_config_sha256(run_config_path)
    official_raw_hash = project_config.raw_config_sha256(official_config_path)
    if raw_hash != manifest.get("config_raw_sha256") or raw_hash != official_raw_hash:
        raise ValueError("Run snapshot, Manifest and current formal config do not match.")
    if Path(str(manifest.get("episode_pool", ""))).resolve() != pool_dir:
        raise ValueError("Run Manifest points to a different Training Pool.")
    pool_manifest_hash = hashlib.sha256(
        (pool_dir / "formal_pool_manifest.json").read_bytes()
    ).hexdigest()
    if (
        pool_manifest_hash != manifest.get("training_pool_manifest_sha256")
        or pool_manifest_hash
        != manifest.get("episode_pool_provenance", {}).get("manifest_sha256")
    ):
        raise ValueError("Current Training Pool manifest differs from the Run identity.")
    episode_rows = [
        json.loads(line)
        for line in episode_metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episode_seeds = [int(row["episode_seed"]) for row in episode_rows]
    if episode_seeds != list(range(20001, 20009)):
        raise ValueError("Run Episode seeds do not match the designated 20001–20008 range.")
    slot_rows = [
        json.loads(line)
        for line in slot_metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cloud_audit = _audit_cloud_slot_metrics(slot_metrics_path, 2400)
    slot_log_by_key = {
        (int(row["episode"]), int(row["slot"])): row for row in slot_rows
    }
    if len(slot_log_by_key) != 2400:
        raise ValueError("Run Slot log has duplicate or missing episode-slot keys.")

    reference_context = interface.load_environment_context(
        str(run_config_path), realization_dir=pool_dir / f"episode_{episode_seeds[0]}"
    )
    semantic_hash = project_config.semantic_config_hash(reference_context.cfg)
    if semantic_hash != manifest.get("config_semantic_sha256"):
        raise ValueError("Run configuration semantic hash differs from its Manifest.")
    trainer = PPOTrainer(reference_context, policy_update_total=75, device=args.device)
    checkpoint_payload = trainer.load_checkpoint(
        checkpoint_path,
        config_raw_sha256=raw_hash,
        config_semantic_sha256=semantic_hash,
        episode_pool=str(pool_dir),
        restore_rng=False,
    )
    checkpoint_counters = checkpoint_payload.get("training_counters", {})
    if (
        int(checkpoint_counters.get("completed_episodes", -1)) != 8
        or int(checkpoint_counters.get("policy_update_count", -1)) != 2
        or checkpoint_payload.get("actor_update_mode") != "ALTERNATING"
        or checkpoint_payload.get("actor_update_schedule_state", {}).get(
            "next_active_actor"
        )
        != "MICRO"
    ):
        raise ValueError("Latest Checkpoint identity/counters do not match this Run.")
    trainer.policy.eval()

    category_names = (
        "A_SERVICE_NOT_CACHED",
        "B_CACHED_EDGE_INFEASIBLE",
        "C_CACHED_EDGE_FEASIBLE_CLOUD_SELECTED",
        "D_CACHED_EDGE_EXECUTED",
    )
    category_counts: Counter[str] = Counter()
    by_episode: dict[str, dict[str, Any]] = {}
    by_slot: dict[str, dict[str, Any]] = {}
    by_service: dict[str, dict[str, Any]] = {}
    by_episode_service_target: Counter[tuple[int, int, str, str]] = Counter()
    task_rows: list[dict[str, Any]] = []
    run_logged_node_counts: Counter[str] = Counter()
    run_logged_task_count = 0
    crosscheck_count = 0
    forecast_slot_count = 0
    forecast_shapes: dict[str, list[int]] = {}
    valid_horizon_counts: list[int] = []
    forecast_positive_service_counts: list[int] = []
    forecast_positive_event_slots = 0
    macro_state_count = 0
    representative_c_cases: dict[int, tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]] = {}
    total_slots_evaluated = 0

    for episode_index, seed in enumerate(episode_seeds, start=1):
        episode_dir = pool_dir / f"episode_{seed}"
        context = interface.load_environment_context(
            str(run_config_path), realization_dir=episode_dir
        )
        interface._assert_realization_context_compatible(reference_context, context)
        steps = interface.aligned_train_steps(context)
        if len(steps) != int(context.episode_slots):
            raise ValueError(f"episode_{seed} does not contain all aligned Training slots.")
        with torch.inference_mode():
            macro_records, slot_records = _macro_decision_states(
                trainer.policy,
                context,
                seed,
                include_slot_states=True,
                aligned_steps=steps,
            )
        macro_state_count += len(macro_records)
        if len(slot_records) != 300:
            raise ValueError(f"Frozen replay for episode_{seed} did not cover 300 slots.")

        for step in steps:
            scenario = step.scenario_slice
            arrivals = np.asarray(scenario.get("arrival"), dtype=np.float64)
            services = np.asarray(scenario.get("service_id"), dtype=np.int64)
            if arrivals.ndim != 3 or services.shape != arrivals.shape:
                raise ValueError(f"episode_{seed} slot {step.slot} has invalid forecast tensors.")
            horizon_mask = np.asarray(step.prediction_horizon_mask, dtype=bool).reshape(-1)
            if horizon_mask.shape != (arrivals.shape[1],):
                raise ValueError(f"episode_{seed} slot {step.slot} forecast horizon mask is misaligned.")
            predicted_demand = cache_configuration.predicted_vehicle_service_demand_by_horizon(
                scenario,
                int(context.service_catalog["num_services"]),
                beta_discount=context.cache_cfg.beta_discount,
                prediction_horizon_mask=horizon_mask,
            )
            if not np.isfinite(predicted_demand).all() or np.any(predicted_demand < 0.0):
                raise FloatingPointError(f"episode_{seed} slot {step.slot} forecast demand is invalid.")
            forecast_slot_count += int(bool(np.any(horizon_mask)))
            valid_horizon_counts.append(int(np.count_nonzero(horizon_mask)))
            forecast_positive_service_counts.append(
                int(np.count_nonzero(np.sum(predicted_demand, axis=(0, 1)) > 1.0e-12))
            )
            forecast_positive_event_slots += int(
                bool(np.any((arrivals[:, horizon_mask, :] > 0.0) & (services[:, horizon_mask, :] >= 0)))
            )
            if not forecast_shapes:
                forecast_shapes = {
                    "arrival": [int(value) for value in arrivals.shape],
                    "service_id": [int(value) for value in services.shape],
                    "rho_hat": [int(value) for value in np.asarray(scenario["rho_hat"]).shape],
                    "prediction_horizon_mask": [int(value) for value in horizon_mask.shape],
                }

        for record in slot_records:
            record["episode"] = int(episode_index)
            key = (int(episode_index), int(record["slot"]))
            logged = slot_log_by_key.get(key)
            if logged is None:
                raise ValueError(f"Run Slot log is missing episode/slot {key}.")
            node_counts = logged.get("node_task_count")
            if not isinstance(node_counts, dict):
                raise ValueError(f"Run Slot {key} lacks node task-count aggregates.")
            logged_count = sum(
                int(node_counts.get(role, 0)) for role in ("SV", "RSU", "UAV", "Cloud")
            )
            if logged_count != len(record["step"].actual_tasks):
                raise ValueError(
                    f"Run Slot {key} task count {logged_count} differs from Training Pool "
                    f"actual task count {len(record['step'].actual_tasks)}."
                )
            crosscheck_count += 1
            run_logged_task_count += logged_count
            for role in ("SV", "RSU", "UAV", "Cloud"):
                run_logged_node_counts[role] += int(node_counts.get(role, 0))
            for group_name in ("SV_by_node", "RSU_by_node", "UAV_by_node"):
                for node_id, count in node_counts.get(group_name, {}).items():
                    run_logged_node_counts[str(node_id)] += int(count)

            metrics = _evaluate_configuration(
                record,
                record["actual_policy_cache_state"],
                record["actual_policy_uav_positions"],
                include_task_attributions=True,
            )
            if int(metrics["task_count"]) != logged_count:
                raise AssertionError("Replay Matching did not evaluate the same Slot task set.")
            slot_bucket_key = f"{seed}:{int(record['slot'])}"
            slot_bucket = {
                "episode": int(episode_index),
                "episode_seed": int(seed),
                "slot": int(record["slot"]),
                "task_count": int(metrics["task_count"]),
                "category_counts": Counter(),
                "service_counts": Counter(),
                "selected_node_counts": Counter(),
                "request_any_edge_cached_count": 0,
                "feasible_edge_candidate_count": 0,
            }
            episode_bucket = by_episode.setdefault(
                str(seed),
                {
                    "episode": int(episode_index),
                    "episode_seed": int(seed),
                    "task_count": 0,
                    "category_counts": Counter(),
                    "selected_node_counts": Counter(),
                },
            )
            for task_attribution in metrics["task_attributions"]:
                category = str(task_attribution["category"])
                if category not in category_names:
                    raise AssertionError(f"Unrecognized task attribution category {category}.")
                selected_node = str(task_attribution["selected_execution_node"])
                service_key = str(task_attribution["service_id"])
                category_counts[category] += 1
                slot_bucket["category_counts"][category] += 1
                slot_bucket["service_counts"][service_key] += 1
                slot_bucket["selected_node_counts"][selected_node] += 1
                slot_bucket["request_any_edge_cached_count"] += int(
                    task_attribution["service_cached_at_any_edge"]
                )
                slot_bucket["feasible_edge_candidate_count"] += int(
                    bool(task_attribution["feasible_edge_node_ids"])
                )
                episode_bucket["task_count"] += 1
                episode_bucket["category_counts"][category] += 1
                episode_bucket["selected_node_counts"][selected_node] += 1
                service_bucket = by_service.setdefault(
                    service_key,
                    {
                        "service_id": int(service_key),
                        "task_count": 0,
                        "category_counts": Counter(),
                        "selected_node_counts": Counter(),
                    },
                )
                service_bucket["task_count"] += 1
                service_bucket["category_counts"][category] += 1
                service_bucket["selected_node_counts"][selected_node] += 1
                by_episode_service_target[
                    (
                        int(episode_index),
                        int(record["slot"]),
                        service_key,
                        selected_node,
                    )
                ] += 1
                public_task = dict(task_attribution)
                public_task["episode"] = int(episode_index)
                task_rows.append(public_task)

                if (
                    category == "C_CACHED_EDGE_FEASIBLE_CLOUD_SELECTED"
                    and episode_index not in representative_c_cases
                ):
                    node_order = [str(value) for value in context.cache_nodes["node_ids"]]
                    candidate_nodes = [
                        node_id
                        for node_id in node_order
                        if node_id in task_attribution["feasible_edge_node_ids"]
                    ]
                    if candidate_nodes:
                        representative_c_cases[episode_index] = (
                            record,
                            task_attribution,
                            metrics,
                            candidate_nodes[0],
                        )

            if len(metrics["task_attributions"]) != int(metrics["task_count"]):
                raise AssertionError("Task attribution did not cover every replay task.")
            by_slot[slot_bucket_key] = {
                **{
                    key_name: value
                    for key_name, value in slot_bucket.items()
                    if not isinstance(value, Counter)
                },
                "category_counts": dict(slot_bucket["category_counts"]),
                "service_counts": dict(slot_bucket["service_counts"]),
                "selected_node_counts": dict(slot_bucket["selected_node_counts"]),
            }
            total_slots_evaluated += 1
            if total_slots_evaluated % 100 == 0:
                print(
                    f"Task attribution replay: evaluated {total_slots_evaluated}/2400 Slots, "
                    f"classified {len(task_rows)} tasks",
                    flush=True,
                )

    if total_slots_evaluated != 2400 or crosscheck_count != 2400:
        raise AssertionError("Task attribution replay did not cover all Run Slots.")
    if sum(category_counts.values()) != run_logged_task_count:
        raise AssertionError("Task-level categories do not cover the Run Slot task total.")

    counterfactuals: list[dict[str, Any]] = []
    with torch.inference_mode():
        for episode_index, (record, task_attribution, baseline, edge_node) in sorted(
            representative_c_cases.items()
        ):
            print(
                f"Forced-Edge paired Matching for representative C task in Episode {episode_index}",
                flush=True,
            )
            counterfactuals.append(
                _forced_edge_task_counterfactual(
                    record,
                    task_attribution,
                    edge_node,
                    baseline,
                )
            )

    c_effects = [
        float(item["delta_G_forced_edge_minus_cloud_baseline"])
        for item in counterfactuals
        if item.get("counterfactual_status") == "EVALUATED"
    ]
    category_counts_public = {
        name: int(category_counts.get(name, 0)) for name in category_names
    }
    cloud_total = int(
        category_counts_public["A_SERVICE_NOT_CACHED"]
        + category_counts_public["B_CACHED_EDGE_INFEASIBLE"]
        + category_counts_public["C_CACHED_EDGE_FEASIBLE_CLOUD_SELECTED"]
    )
    forecast_horizon_summary = {
        "min_valid_horizons_per_slot": min(valid_horizon_counts) if valid_horizon_counts else None,
        "median_valid_horizons_per_slot": (
            float(np.median(valid_horizon_counts)) if valid_horizon_counts else None
        ),
        "max_valid_horizons_per_slot": max(valid_horizon_counts) if valid_horizon_counts else None,
        "mean_positive_predicted_services_per_slot": (
            float(np.mean(forecast_positive_service_counts))
            if forecast_positive_service_counts
            else None
        ),
    }
    update_rows = [
        json.loads(line)
        for line in (run_dir / "update_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    macro_advantage_modes = sorted(
        {
            str(value)
            for value in (
                manifest.get("macro_actor_advantage_mode"),
                *(row.get("macro_actor_advantage_mode") for row in update_rows),
            )
            if value is not None
        }
    )

    return _json_value(
        {
            "metadata": {
                "analysis_mode": "ALL_SLOT_TASK_ATTRIBUTION_FROM_STRICT_CURRENT_CHECKPOINT_DETERMINISTIC_REPLAY",
                "run_dir": run_dir,
                "training_pool": pool_dir,
                "episode_seeds": episode_seeds,
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "checkpoint_completed_episodes": int(checkpoint_counters["completed_episodes"]),
                "checkpoint_policy_updates": int(checkpoint_counters["policy_update_count"]),
                "run_config_raw_sha256": raw_hash,
                "formal_config_raw_sha256": official_raw_hash,
                "training_pool_manifest_sha256": pool_manifest_hash,
                "strict_checkpoint_identity_load": True,
                "frozen_deterministic_policy_replay": True,
                "training_executed": False,
                "checkpoint_saved_or_modified": False,
                "validation_test_accessed": False,
            },
            "run_preflight": {
                "controlled_stop_status": summary.get("training_completion_status"),
                "completed_episodes": int(summary["completed_episodes"]),
                "target_episodes": int(summary["target_episodes"]),
                "completed_policy_updates": int(summary["policy_update_count"]),
                "checkpoint_identity_match": True,
                "cloud_resource_log_audit": cloud_audit,
                "run_actual_task_count_from_node_aggregates": int(run_logged_task_count),
                "run_actual_execution_node_counts_from_slot_logs": dict(run_logged_node_counts),
                "pool_task_count_crosschecks": int(crosscheck_count),
                "pool_task_count_crosscheck_failures": 0,
            },
            "task_level_attribution": {
                "scope": "All 2400 saved Training Slots replayed with the Run's latest valid Checkpoint; not a reconstruction of stochastic per-task assignments made during the original training episodes.",
                "original_run_per_task_matching_logged": False,
                "replay_task_count": int(len(task_rows)),
                "category_definitions": {
                    "A_SERVICE_NOT_CACHED": "Request service is absent from all edge caches active at execution; selected Cloud.",
                    "B_CACHED_EDGE_INFEASIBLE": "Request service is cached somewhere, but the production feasible-target builder returns no legal/reachable Edge path; selected Cloud.",
                    "C_CACHED_EDGE_FEASIBLE_CLOUD_SELECTED": "Request service is cached and at least one production Edge target passes path/role/rate feasibility; selected Cloud after full Matching+KKT.",
                    "D_CACHED_EDGE_EXECUTED": "Matching selects an Edge node that has the request service cached and is in the production feasible Edge target list.",
                },
                "category_counts": category_counts_public,
                "category_share_of_all_tasks": {
                    key: (float(value / len(task_rows)) if task_rows else None)
                    for key, value in category_counts_public.items()
                },
                "cloud_selected_cause_counts": {
                    key: category_counts_public[key]
                    for key in category_names[:3]
                },
                "cloud_selected_cause_share": {
                    key: (
                        float(category_counts_public[key] / cloud_total)
                        if cloud_total
                        else None
                    )
                    for key in category_names[:3]
                },
                "by_episode_seed": {
                    key: {
                        **{
                            name: value
                            for name, value in bucket.items()
                            if not isinstance(value, Counter)
                        },
                        "category_counts": dict(bucket["category_counts"]),
                        "selected_node_counts": dict(bucket["selected_node_counts"]),
                    }
                    for key, bucket in sorted(by_episode.items())
                },
                "by_episode_slot": by_slot,
                "by_service_id": {
                    key: {
                        **{
                            name: value
                            for name, value in bucket.items()
                            if not isinstance(value, Counter)
                        },
                        "category_counts": dict(bucket["category_counts"]),
                        "selected_node_counts": dict(bucket["selected_node_counts"]),
                    }
                    for key, bucket in sorted(by_service.items(), key=lambda item: int(item[0]))
                },
                "by_episode_slot_service_selected_node": [
                    {
                        "episode": key[0],
                        "slot": key[1],
                        "service_id": int(key[2]),
                        "selected_execution_node": key[3],
                        "task_count": int(count),
                    }
                    for key, count in sorted(by_episode_service_target.items())
                ],
                "task_records": task_rows,
            },
            "cached_and_edge_feasible_but_cloud_selected": {
                "count": category_counts_public[
                    "C_CACHED_EDGE_FEASIBLE_CLOUD_SELECTED"
                ],
                "full_run_candidate_scores_logged": False,
                "replay_task_evidence_fields": [
                    "feasible_edge_node_ids",
                    "selected_task_utility",
                    "selected_task_delay_s",
                    "selected_vehicle_upload_energy_j",
                    "selected_cloud_cpu_cycles_per_s",
                    "selected_cloud_uplink_bps",
                    "matching_accepted_operations_for_task",
                ],
                "matching_history_limit": "Production Matching logs accepted transfer/exchange operations, not every candidate's task-level score; current original Run slot logs contain no per-task targets or scores.",
            },
            "edge_vs_cloud_counterfactual": {
                "method": "Representative C tasks selected deterministically as the first C task by (Episode, Slot, task_id), at most one per Episode; force that task to the first feasible Edge node in the formal node order, keep same Slot tasks/vehicle positions/active Cache/UAV, and rerun production Matching+KKT with all other tasks mutable.",
                "representative_case_count": len(counterfactuals),
                "evaluated_case_count": sum(
                    int(item.get("counterfactual_status") == "EVALUATED")
                    for item in counterfactuals
                ),
                "unresolved_case_count": sum(
                    int(item.get("counterfactual_status") == "UNRESOLVED")
                    for item in counterfactuals
                ),
                "mean_delta_G_forced_edge_minus_cloud": (
                    float(np.mean(c_effects)) if c_effects else None
                ),
                "median_delta_G_forced_edge_minus_cloud": (
                    float(np.median(c_effects)) if c_effects else None
                ),
                "positive_delta_G_count": sum(value > HIT_TOLERANCE for value in c_effects),
                "zero_delta_G_count": sum(abs(value) <= HIT_TOLERANCE for value in c_effects),
                "negative_delta_G_count": sum(value < -HIT_TOLERANCE for value in c_effects),
                "cases": counterfactuals,
            },
            "prediction_and_cache_value_readiness": {
                "prediction_fields_present": [
                    "arrival",
                    "service_id",
                    "rho_hat",
                    "horizons",
                    "positions_xy",
                    "task_vehicle_ids",
                    "prediction_horizon_mask",
                ],
                "prediction_tensor_shapes_first_slot": forecast_shapes,
                "slots_with_valid_forecast_horizons": int(forecast_slot_count),
                "total_training_slots_checked": 2400,
                "forecast_horizon_count_summary": forecast_horizon_summary,
                "slots_with_positive_predicted_service_events": int(
                    forecast_positive_event_slots
                ),
                "predicted_future_service_demand_available": bool(
                    forecast_slot_count == 2400
                    and forecast_positive_event_slots > 0
                ),
                "economic_candidate_feature_dim": int(CACHE_ECONOMIC_FEATURE_DIM),
                "economic_candidate_forecast_horizons": int(
                    CACHE_ECONOMIC_DEMAND_HORIZONS
                ),
                "candidate_generation_uses_forecast_service_demand": True,
                "candidate_features_include_download_cost_and_loaded_or_released_size": True,
                "candidate_features_include_predicted_demand_for_added_and_removed_services": True,
                "predicted_cache_marginal_value_status": "PROXY_ESTIMABLE; calibrated future net system utility is not directly provided by the actor candidate score and requires same-window forecast Matching evaluations.",
                "future_eviction_opportunity_cost_status": "FORECAST_DEMAND_PROXY_AVAILABLE; no direct calibrated system-utility opportunity-cost scalar is logged per candidate.",
            },
            "current_trigger_credit_reassessment": {
                "current_run_macro_actor_advantage_mode": macro_advantage_modes,
                "actor_action_semantics": "Each factorized Cache node head selects KEEP (candidate 0) or one legal UPDATE target (candidate >0); trigger and candidate are values of the same categorical decision, not separately scored modules.",
                "current_macro_training_credit": "The joint Macro Actor uses the configured formal Macro SMDP GAE / clipped PPO objective for Cache candidates and sampled termination decisions.",
                "trigger_candidate_comparison_consistency": "SHARED_RETURN_CREDIT; no separately logged predicted net-value comparator for KEEP versus each UPDATE target.",
                "historical_cache_trigger_causal_gate_scope": "The existing cache_trigger_causal_gate.md report analyzes different stage2d LR checkpoints; it is not evidence about this controlled Run's latest Checkpoint.",
                "macro_states_replayed": int(macro_state_count),
                "evidence_limit": "Only two Policy Updates are present; this establishes the code/checkpoint credit contract, not learned trigger calibration or convergence.",
            },
        }
    )


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NOT_AVAILABLE"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def _scheme_table(summary: dict[str, Any], group: str) -> str:
    output = [
        f"### {group}",
        "",
        "| 干预 | ΔG>0 | ΔG=0 | ΔG<0 | 平均 ΔG | 中位 ΔG | 平均时延变化 (s/task) | 上传能耗变化 (J) | Cloud 率变化 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("cache_only", "Cache-only"), ("uav_only", "UAV-only")):
        item = summary[key]
        changes = item["mean_change_by_metric"]
        output.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                label,
                item["positive_delta_G_count"],
                item["zero_delta_G_count"],
                item["negative_delta_G_count"],
                _fmt(item["mean_delta_G"]),
                _fmt(item["median_delta_G"]),
                _fmt(changes["mean_task_delay_s"]["mean"]),
                _fmt(changes["total_vehicle_upload_energy_j"]["mean"]),
                _fmt(changes["cloud_offload_rate"]["mean"]),
            )
        )
    return "\n".join(output)


def _render_report(data: dict[str, Any]) -> str:
    meta = data["metadata"]
    distribution = data["state_distribution"]
    preflight = data["run_preflight"]
    resources = data["cloud_resource_log_audit"]
    ten_slot = data["ten_slot_paired_replay"]
    low_summary = data["low_hit_summary"]
    control_summary = data["matched_control_summary"]
    low_samples = [item for item in data["samples"] if item["sample_group"] == "LOW_HIT"]
    control_samples = [item for item in data["samples"] if item["sample_group"] == "MATCHED_CONTROL"]
    cache_contrast = data["paired_low_minus_control_immediate_oracle_gain"]["cache_only"]
    uav_contrast = data["paired_low_minus_control_immediate_oracle_gain"]["uav_only"]
    def candidate_mean(
        samples: list[dict[str, Any]], scheme: str, metric: str
    ) -> float | None:
        values = [
            float(item[scheme]["best_candidate"]["metrics"][metric])
            for item in samples
            if item[scheme]["best_candidate"]["metrics"].get(metric) is not None
        ]
        return float(np.mean(values)) if values else None

    def ci_text(item: dict[str, Any]) -> str:
        return f"[{_fmt(item.get('ci95_low'))}, {_fmt(item.get('ci95_high'))}]"

    lines = [
        "# 低缓存命中率下 Cache 更新与 UAV 移动的配对反事实实验",
        "",
        "## 数据与 Run 门禁",
        "",
        f"Run：`{meta['run_dir']}`；Training Pool：`{meta['training_pool']}`；冻结 Checkpoint SHA-256：`{meta['checkpoint_sha256']}`。Checkpoint 处于 {meta['checkpoint_episode']} Episode / {meta['checkpoint_policy_updates']} 次 Update。策略以确定性 `torch.inference_mode()` 执行，不做梯度或优化器更新，也不保存 Checkpoint。",
        "",
        f"Run 状态为 `{preflight['controlled_stop_status']}`，目标/完成 Episode={preflight['target_episodes']}/{preflight['completed_episodes']}，停止上限/完成 Policy Update={preflight['controlled_stop_after_policy_updates']}/{preflight['completed_policy_updates']}。当前正式配置原始哈希、Run Manifest 哈希、语义哈希、Training Pool Manifest 和严格 Checkpoint 身份均匹配。",
        "",
        f"{resources['slot_records']} 条 Slot 日志完整覆盖 Episodes 1–8 的 Slot 0–299；各 Episode Return 与 Slot `G` 汇总一致，最大绝对差 {_fmt(max(row['absolute_error'] for row in preflight['episode_return_slot_reward_checks']))}。Cloud 最终分配及 KKT 汇总日志全量解析：选中上行/CPU违反分别 {resources['selected_uplink_violation_count']}/{resources['selected_cpu_violation_count']}，KKT 上行/CPU违反分别 {resources['kkt_uplink_violation_count']}/{resources['kkt_cpu_violation_count']}，非有限结果 {resources['kkt_nonfinite_result_count']}。Cloud 能力为 20 Mbps、20 GHz；日志内 KKT 标量求解数合计 {resources['cloud_kkt_solve_count_total']}，最大上行分配 {_fmt(resources['max_kkt_uplink_allocated_bps'])} bit/s，最大 CPU 分配 {_fmt(resources['max_kkt_cpu_allocated_cycles_per_s'])} cycles/s。",
        "",
        "两个 Update 日志记录的活跃 Actor 顺序为 Micro、Macro；非活跃 Actor 步数均为 0，三个 Critic 每组均有正更新步数。",
        "",
        "## 命中率、样本和决策信息",
        "",
        f"冻结策略在 8 个 Run Training Episode（Seed {', '.join(map(str, meta['training_episode_seeds_replayed']))}）上重放 {distribution['cache_decision_states_scanned']} 个 Macro Cache 决策状态，其中 {distribution['valid_cache_decision_states_with_tasks']} 个状态有任务。低命中阈值为有效状态缓存命中率的经验 P25={_fmt(distribution['threshold_p25'])}；全体有效状态的 min/median/max={_fmt(distribution['hit_rate_min'])}/{_fmt(distribution['hit_rate_median'])}/{_fmt(distribution['hit_rate_max'])}。阈值只用于本次 Training 样本。",
        "",
        f"最终抽取低命中 {len(low_samples)} 个状态，匹配较高命中对照 {len(control_samples)} 个。匹配先按同 Episode、同 30-Slot 时间段排序，再按任务数量、请求服务直方图、车辆位置中心和 UAV 位置中心接近程度匹配；该匹配在查看任何干预收益前完成。具体每对的匹配距离见 JSON。",
        "",
        "主命中率按干预前 Cache 状态和该 Slot 实际任务计算：请求服务存在任意 SV/RSU/UAV Cache 的任务数除以任务数。另行记录 Matching 选中的 Edge 节点是否缓存请求服务、是否存在可行 Edge 路径、实际 Edge/Cloud 分配比例和 Cloud 选择原因。它们不等同于缓存目录覆盖率、Edge 卸载率或相互替代的命中指标。",
        "",
        f"低命中组基线任务原因统计：{json.dumps(low_summary['primary_cause_task_counts'], ensure_ascii=False, sort_keys=True)}。分类区分服务未缓存、已缓存但无可行 Edge 路径、存在可行 Edge 但实际选择 Cloud，以及 Edge 命中。",
        "",
        "当前策略观察使用保存的预测和配置状态，不含 `actual_tasks`。基线使用冻结策略实际确定性动作；Cache-only 固定其 UAV 位置并枚举合法单节点 UPDATE，UAV-only 固定其 Cache 状态并枚举合法单 UAV 动作。Oracle 在候选求解之后使用当前实际任务最大化 `G`，因此只是候选子集内的离线 Oracle 上界。冻结策略实际 Cache/Micro 动作也单独与 KEEP/STAY 参考比较。",
        "",
        "## 即时配对收益",
        "",
    ]
    for group, title, summary in (
        ("LOW_HIT", "低命中组", low_summary),
        ("MATCHED_CONTROL", "匹配高命中对照组", control_summary),
    ):
        lines.extend([f"### {title}", "", "| 方案 | ΔG 正/零/负 | 正收益比例 | 平均 ΔG | 中位 ΔG | 平均值 95% CI | Δ任务效用 | ΔCache 成本 | Δ移动成本 | Δ时延 s/任务 | Δ上传能耗 J | ΔCloud 率 | Δ任意缓存命中率 | Δ选中 Edge 命中率 | ΔEdge 可行率 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for scheme, label in (("cache_only", "Cache-only Oracle"), ("uav_only", "UAV-only Oracle")):
            item = summary[scheme]
            changes = item["mean_change_by_metric"]
            lines.append(
                f"| {label} | {item['positive_delta_G_count']}/{item['zero_delta_G_count']}/{item['negative_delta_G_count']} | {_fmt(item['positive_delta_G_rate'])} | {_fmt(item['mean_delta_G'])} | {_fmt(item['median_delta_G'])} | {ci_text(item['mean_delta_G_bootstrap_95_ci'])} | {_fmt(changes['task_utility']['mean'])} | {_fmt(changes['cache_cost']['mean'])} | {_fmt(changes['movement_cost']['mean'])} | {_fmt(changes['mean_task_delay_s']['mean'])} | {_fmt(changes['total_vehicle_upload_energy_j']['mean'])} | {_fmt(changes['cloud_offload_rate']['mean'])} | {_fmt(changes['request_any_edge_cache_hit_rate']['mean'])} | {_fmt(changes['selected_edge_cache_hit_rate']['mean'])} | {_fmt(changes['feasible_edge_path_rate']['mean'])} |"
            )
        lines.append("")

    for samples, title in (
        (low_samples, "低命中组"),
        (control_samples, "匹配高命中组"),
    ):
        lines.append(
            f"{title}每状态 Oracle 最优 Cache 分支的平均新增加载量/下载时延/Cache 成本="
            f"{_fmt(candidate_mean(samples, 'cache_only', 'cache_loaded_mb'))} MB / "
            f"{_fmt(candidate_mean(samples, 'cache_only', 'cache_download_delay_s'))} s / "
            f"{_fmt(candidate_mean(samples, 'cache_only', 'cache_cost'))}；UAV 分支平均移动距离/移动能耗/移动成本="
            f"{_fmt(candidate_mean(samples, 'uav_only', 'movement_distance_m'))} m / "
            f"{_fmt(candidate_mean(samples, 'uav_only', 'movement_energy_j'))} J / "
            f"{_fmt(candidate_mean(samples, 'uav_only', 'movement_cost'))}。"
        )
    lines.append("")

    lines.extend([
        f"低命中减匹配对照的 Cache-only Oracle 即时收益差：n={cache_contrast['n_matched_pairs']}，均值/中位数={_fmt(cache_contrast['mean_low_minus_control_delta_G'])}/{_fmt(cache_contrast['median_low_minus_control_delta_G'])}，95% 配对 bootstrap CI={ci_text(cache_contrast)}，正/零/负配对={cache_contrast['positive_pair_count']}/{cache_contrast['zero_pair_count']}/{cache_contrast['negative_pair_count']}。",
        f"低命中减匹配对照的 UAV-only Oracle 即时收益差：n={uav_contrast['n_matched_pairs']}，均值/中位数={_fmt(uav_contrast['mean_low_minus_control_delta_G'])}/{_fmt(uav_contrast['median_low_minus_control_delta_G'])}，95% 配对 bootstrap CI={ci_text(uav_contrast)}，正/零/负配对={uav_contrast['positive_pair_count']}/{uav_contrast['zero_pair_count']}/{uav_contrast['negative_pair_count']}。",
        "",
        f"低命中 Cache-only 候选数={low_summary['cache_update_candidate_scores']['candidate_count']}，候选收益正/零/负={low_summary['cache_update_candidate_scores']['positive_delta_G_count']}/{low_summary['cache_update_candidate_scores']['zero_delta_G_count']}/{low_summary['cache_update_candidate_scores']['negative_delta_G_count']}；UAV-only 候选数={low_summary['uav_move_candidate_scores']['candidate_count']}，正/零/负={low_summary['uav_move_candidate_scores']['positive_delta_G_count']}/{low_summary['uav_move_candidate_scores']['zero_delta_G_count']}/{low_summary['uav_move_candidate_scores']['negative_delta_G_count']}。低命中 Oracle 最佳动作标签 Cache/UAV={json.dumps(low_summary['best_cache_candidate_label_counts'], ensure_ascii=False)}/{json.dumps(low_summary['best_uav_candidate_label_counts'], ensure_ascii=False)}；高命中对照={json.dumps(data['matched_control_summary']['best_cache_candidate_label_counts'], ensure_ascii=False)}/{json.dumps(data['matched_control_summary']['best_uav_candidate_label_counts'], ensure_ascii=False)}。对照组完整候选统计及每个候选实际 Cloud 上行/CPU 分配、容量和 KKT 聚合值均保存在 JSON。",
        "",
        "正式策略动作的可部署参照（相同策略 UAV 下 Cache 与 KEEP 比较；相同策略 Cache 下 UAV 与 ALL_STAY 比较）：",
        f"- 低命中组 Macro Cache 动作 ΔG 均值/95% CI={_fmt(low_summary['frozen_policy_cache_action_vs_keep']['mean_delta_G'])}/{ci_text(low_summary['frozen_policy_cache_action_vs_keep']['mean_delta_G_bootstrap_95_ci'])}；Micro UAV 动作={_fmt(low_summary['frozen_policy_uav_action_vs_stay']['mean_delta_G'])}/{ci_text(low_summary['frozen_policy_uav_action_vs_stay']['mean_delta_G_bootstrap_95_ci'])}。",
        f"- 匹配对照组 Macro Cache 动作 ΔG 均值/95% CI={_fmt(control_summary['frozen_policy_cache_action_vs_keep']['mean_delta_G'])}/{ci_text(control_summary['frozen_policy_cache_action_vs_keep']['mean_delta_G_bootstrap_95_ci'])}；Micro UAV 动作={_fmt(control_summary['frozen_policy_uav_action_vs_stay']['mean_delta_G'])}/{ci_text(control_summary['frozen_policy_uav_action_vs_stay']['mean_delta_G_bootstrap_95_ci'])}。",
        "",
        "## 10 Slot 配对回放",
        "",
    ])
    if ten_slot["representative_matched_pair_count"]:
        lines.append(
            f"对 {ten_slot['representative_matched_pair_count']} 个代表性匹配状态对，Baseline、Cache-only、UAV-only 均从相同干预前 Cache/UAV 快照开始，并使用相同的连续 10 Slot 外生任务和车辆轨迹。首 Slot 后冻结策略恢复决策，Cache/UAV 状态和 Adaptive Holding 按正式状态转移演化。"
        )
        for group in ("LOW_HIT", "MATCHED_CONTROL"):
            for scheme in ("cache_only", "uav_only"):
                item = ten_slot["group_summary"][group][scheme]
                lines.append(
                    f"- `{group}` {scheme}：首 Slot 平均 ΔG={_fmt(item['first_slot_mean_delta_G'])}；10 Slot 累计 Reward 差均值/中位数={_fmt(item['cumulative_10_slot_mean_delta_G'])}/{_fmt(item['cumulative_10_slot_median_delta_G'])}，95% CI={ci_text(item['cumulative_10_slot_mean_delta_G_bootstrap_95_ci'])}（n={item['n']}）。"
                )
        for scheme in ("cache_only", "uav_only"):
            contrast = ten_slot["low_minus_control_cumulative_reward_contrast"][scheme]
            lines.append(
                f"- 10 Slot `{scheme}` 低命中减匹配对照累计 ΔG：均值/中位数={_fmt(contrast['mean'])}/{_fmt(contrast['median'])}，95% paired bootstrap CI={ci_text(contrast)}（n={contrast['n']} 对）。"
            )
        for pair in ten_slot["paired_runs"]:
            for group in ("LOW_HIT", "MATCHED_CONTROL"):
                for scheme in ("cache_only", "uav_only"):
                    effect = pair[group][scheme]
                    lines.append(
                        f"- `{group}` {scheme} {pair[group]['cache_only']['baseline']['slot_indices'][0]} 起：首 Slot ΔG={_fmt(effect['first_slot_delta_G'])}，累计 ΔG={_fmt(effect['cumulative_10_slot_reward_delta_G'])}，Post-action 命中率变化={_fmt(effect['post_action_cache_hit_rate_change'])}，累计 Cache/movement 成本变化={_fmt(effect['cumulative_cache_cost_change'])}/{_fmt(effect['cumulative_movement_cost_change'])}，平均时延变化={_fmt(effect['mean_task_delay_change_s'])} s，上传能耗变化={_fmt(effect['upload_energy_change_j'])} J，Cloud 比例变化={_fmt(effect['cloud_offload_rate_change'])}。"
                    )
        lines.append(
            "这些 10 Slot 结果是少量代表性状态的连续配对轨迹，不外推为总体长期收益；每一 Slot 的 Reward、Matching/KKT 和 Cloud 资源证据仍来自正式执行器。"
        )
    else:
        lines.append("未找到同时具有完整连续 10 Slot 数据的匹配状态对，10 Slot 效果为 UNRESOLVED。")
    lines.extend([
        "",
        "## 解释与结论边界",
        "",
        f"低命中 Cache 触发判定：`{data['low_hit_trigger_assessment']['cache_only']}`；低命中 UAV 触发判定：`{data['low_hit_trigger_assessment']['uav_only']}`。判定基于低命中组自身 Oracle 收益和相对匹配高命中组的收益差及其 95% bootstrap 区间，定义见 JSON。",
        "当前代码没有依照低命中率触发 Cache 或 UAV 干预的规则，故 `DEPLOYABLE_PREDICTION_DRIVEN_TRIGGER=NOT_IMPLEMENTED`。现有冻结 Actor 的预测驱动动作已另行报告，但不能把 Oracle 候选收益当成 Actor 已学会的行为。",
        "",
        "所有效用都由正式路径计算并校验 `G = task_utility - cache_cost - movement_cost`。Cache-only 与 UAV-only 成本由正式 Slot 结果产生；每个候选和后续重放 Slot 都重新运行完整 Matching+KKT，资源分配违反/非有限值会使本分析失败。",
        "",
        f"40 个即时配对状态中，Cache-only 与 UAV-only 的最佳枚举动作都没有超过冻结策略基线的状态数为 {data['both_interventions_nonpositive_count']}；低命中原因与不同主因下最佳单项干预见 JSON：{json.dumps(low_summary['best_intervention_by_dominant_cause'], ensure_ascii=False, sort_keys=True)}。",
        "",
        "证据限制：Checkpoint 仅经历 8 Episode / 2 Update，状态来自同一 Training Pool 的训练内回放，候选只覆盖单节点 Cache 更新和单 UAV 动作；无联合动作搜索。Cache 更新立即生效，未模拟下载队列。Oracle 结果只代表枚举范围内的上界，不能证明 MAPPO 收敛或触发策略可部署。",
        "",
        "```ini",
        f"CURRENT_RUN_RESOURCE_LOG_AUDIT = PASS; {resources['slot_records']} records; selected/KKT violations=0; nonfinite=0; KKT scalar solves={resources['cloud_kkt_solve_count_total']}",
        f"LOW_HIT_THRESHOLD = {distribution['threshold_p25']}",
        f"LOW_HIT_SAMPLES = {len(low_samples)}",
        f"MATCHED_CONTROL_SAMPLES = {len(control_samples)}",
        f"CACHE_ONLY_IMMEDIATE_EFFECT = mean={_fmt(low_summary['cache_only']['mean_delta_G'])}; median={_fmt(low_summary['cache_only']['median_delta_G'])}; 95%CI={ci_text(low_summary['cache_only']['mean_delta_G_bootstrap_95_ci'])}",
        f"UAV_ONLY_IMMEDIATE_EFFECT = mean={_fmt(low_summary['uav_only']['mean_delta_G'])}; median={_fmt(low_summary['uav_only']['median_delta_G'])}; 95%CI={ci_text(low_summary['uav_only']['mean_delta_G_bootstrap_95_ci'])}",
        f"CACHE_ONLY_10_SLOT_EFFECT = {ten_slot['group_summary'].get('LOW_HIT', {}).get('cache_only', {}).get('cumulative_10_slot_mean_delta_G', 'UNRESOLVED')}",
        f"UAV_ONLY_10_SLOT_EFFECT = {ten_slot['group_summary'].get('LOW_HIT', {}).get('uav_only', {}).get('cumulative_10_slot_mean_delta_G', 'UNRESOLVED')}",
        f"LOW_HIT_AS_CACHE_TRIGGER = {data['low_hit_trigger_assessment']['cache_only']}",
        f"LOW_HIT_AS_UAV_TRIGGER = {data['low_hit_trigger_assessment']['uav_only']}",
        "DEPLOYABLE_PREDICTION_DRIVEN_TRIGGER = NOT_IMPLEMENTED; frozen policy action reference measured separately",
        "MAPPO_TRAINING_EXECUTED = NO",
        "FORMAL_ALGORITHM_MODIFIED = NO",
        "VALIDATION_TEST_READ = NO",
        "```",
        "",
        f"完整逐状态候选、成本、任务指标、资源分配和 10 Slot 分支见 `{DEFAULT_JSON}`。",
        "",
    ])
    task_study = data.get("task_level_attribution")
    if isinstance(task_study, dict):
        categories = task_study["category_counts"]
        cloud_causes = task_study["cloud_selected_cause_counts"]
        task_preflight = data["task_attribution_run_preflight"]
        prediction = data["prediction_and_cache_value_readiness"]
        trigger_credit = data["current_trigger_credit_reassessment"]
        edge_cloud = data["edge_vs_cloud_counterfactual"]
        episode_attributions = task_study["by_episode_seed"]
        service_attributions = task_study["by_service_id"]
        episode_table = [
            "| Episode Seed | Tasks | A 未缓存 | B 缓存但 Edge 不可行 | C 可行 Edge 但选 Cloud | D Edge 执行 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for seed, bucket in episode_attributions.items():
            counts = bucket["category_counts"]
            episode_table.append(
                f"| {seed} | {bucket['task_count']} | {counts.get('A_SERVICE_NOT_CACHED', 0)} | {counts.get('B_CACHED_EDGE_INFEASIBLE', 0)} | {counts.get('C_CACHED_EDGE_FEASIBLE_CLOUD_SELECTED', 0)} | {counts.get('D_CACHED_EDGE_EXECUTED', 0)} |"
            )
        service_table = [
            "| Service ID | Tasks | A | B | C | D |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for service_id, bucket in service_attributions.items():
            counts = bucket["category_counts"]
            service_table.append(
                f"| {service_id} | {bucket['task_count']} | {counts.get('A_SERVICE_NOT_CACHED', 0)} | {counts.get('B_CACHED_EDGE_INFEASIBLE', 0)} | {counts.get('C_CACHED_EDGE_FEASIBLE_CLOUD_SELECTED', 0)} | {counts.get('D_CACHED_EDGE_EXECUTED', 0)} |"
            )
        lines.extend(
            [
                "## 全部 Training Slot 的任务级缓存与 Cloud 归因",
                "",
                f"Run 原始 Slot 日志没有保存逐任务服务 ID、选中执行节点或候选得分；它只保存按节点聚合的任务数。以下分类因此来自当前严格身份匹配的 latest Checkpoint `{data['metadata']['task_attribution_analysis']['checkpoint_sha256']}` 对相同 8 个 Training Episode 的确定性冻结重放，不是原训练时逐任务动作的重建。重放覆盖 {task_study['replay_task_count']} 个任务 / 2,400 个 Slot；每个 Slot 的 Pool 实际任务数与 Run `node_task_count` 总数一致（{task_preflight['pool_task_count_crosschecks']} / 2,400）。原 Run 实际节点聚合计数另见 JSON `run_actual_execution_node_counts_from_slot_logs`。",
                "",
                "定义：A=执行时请求服务不在任一边缘缓存；B=服务已缓存但正式 feasible-target 构造器没有返回 Edge 路径；C=服务已缓存、有正式 Edge 候选但完整 Matching+KKT 选择 Cloud；D=服务命中所选 Edge 节点并由 Edge 执行。feasible Edge 候选按当前缓存、车辆/节点位置、角色、无线链路和正资源条件构造；共同资源分配再由正式 Matching+KKT 评价。",
                "",
                "| 类别 | 任务数 | 占全部重放任务 |",
                "|---|---:|---:|",
                f"| A：未缓存 | {categories['A_SERVICE_NOT_CACHED']} | {_fmt(task_study['category_share_of_all_tasks']['A_SERVICE_NOT_CACHED'])} |",
                f"| B：缓存但 Edge 不可行 | {categories['B_CACHED_EDGE_INFEASIBLE']} | {_fmt(task_study['category_share_of_all_tasks']['B_CACHED_EDGE_INFEASIBLE'])} |",
                f"| C：缓存且 Edge 可行，选择 Cloud | {categories['C_CACHED_EDGE_FEASIBLE_CLOUD_SELECTED']} | {_fmt(task_study['category_share_of_all_tasks']['C_CACHED_EDGE_FEASIBLE_CLOUD_SELECTED'])} |",
                f"| D：缓存命中并由 Edge 执行 | {categories['D_CACHED_EDGE_EXECUTED']} | {_fmt(task_study['category_share_of_all_tasks']['D_CACHED_EDGE_EXECUTED'])} |",
                "",
                f"重放 Cloud 任务数={sum(cloud_causes.values())}。Cloud 任务原因占比按 Cloud 任务为分母：A={_fmt(task_study['cloud_selected_cause_share']['A_SERVICE_NOT_CACHED'])}，B={_fmt(task_study['cloud_selected_cause_share']['B_CACHED_EDGE_INFEASIBLE'])}，C={_fmt(task_study['cloud_selected_cause_share']['C_CACHED_EDGE_FEASIBLE_CLOUD_SELECTED'])}。原 Run 日志实际节点聚合为 SV={task_preflight['run_actual_execution_node_counts_from_slot_logs'].get('SV', 0)}、RSU={task_preflight['run_actual_execution_node_counts_from_slot_logs'].get('RSU', 0)}、UAV={task_preflight['run_actual_execution_node_counts_from_slot_logs'].get('UAV', 0)}、Cloud={task_preflight['run_actual_execution_node_counts_from_slot_logs'].get('Cloud', 0)}；冻结 Checkpoint 重放的执行类别则为 Edge={categories['D_CACHED_EDGE_EXECUTED']}、Cloud={sum(cloud_causes.values())}。两组是不同策略状态的证据，不应混作原 Run 实际逐任务分类。完整逐任务记录和 Episode/Slot/Service/执行节点交叉表均写入 JSON `task_level_attribution`。",
                "",
                "#### 按 Episode 分类（冻结 Checkpoint 重放）",
                "",
                *episode_table,
                "",
                "#### 按请求 Service 分类（冻结 Checkpoint 重放）",
                "",
                *service_table,
                "",
                "### C 类任务：全 Slot 配对的强制 Edge 代表样本",
                "",
                f"C 类总量={categories['C_CACHED_EDGE_FEASIBLE_CLOUD_SELECTED']}；逐 Episode 最多取一个确定性首个 C 任务（按 Episode、Slot、task ID），并强制到正式节点顺序中的第一个可行 Edge 节点。每次使用同一 Slot 全部任务、车辆位置、活动 Cache/UAV 状态，调用生产 `optimize_task_matching`；其余任务继续参与完整重匹配与 KKT。代表样本 {edge_cloud['representative_case_count']} 个，完整评价成功 {edge_cloud['evaluated_case_count']} 个，未解决 {edge_cloud['unresolved_case_count']} 个。",
                "",
                "| Episode | Slot | Task | 强制 Edge | ΔG（强制 Edge−Cloud 基线） | Δ全 Slot 时延 (s) | Δ上传能耗 (J) | Cloud 率基线→反事实 | 改变分配的任务数 |",
                "|---:|---:|---|---|---:|---:|---:|---|---:|",
            ]
        )
        for case in edge_cloud["cases"]:
            if case.get("counterfactual_status") == "EVALUATED":
                lines.append(
                    f"| {case['episode']} | {case['slot']} | {case['task_id']} | {case['forced_edge_node']} | {_fmt(case['delta_G_forced_edge_minus_cloud_baseline'])} | {_fmt(case['delta_total_task_delay_s'])} | {_fmt(case['delta_total_vehicle_upload_energy_j'])} | {_fmt(case['baseline_cloud_offload_rate'])} → {_fmt(case['forced_edge_cloud_offload_rate'])} | {case['changed_task_assignments_after_rematching']} |"
                )
            else:
                lines.append(
                    f"| {case.get('episode', 'NA')} | {case.get('slot', 'NA')} | {case.get('task_id', 'NA')} | {case.get('forced_edge_node', 'NA')} | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |"
                )
        lines.extend(
            [
                "",
                f"可评价样本 ΔG 均值/中位数={_fmt(edge_cloud['mean_delta_G_forced_edge_minus_cloud'])}/{_fmt(edge_cloud['median_delta_G_forced_edge_minus_cloud'])}；正/零/负={edge_cloud['positive_delta_G_count']}/{edge_cloud['zero_delta_G_count']}/{edge_cloud['negative_delta_G_count']}。这些是少量 C 类代表任务的强制目标反事实，不是全局最优或策略可部署性的证明。原 Run 未记录每次候选评分；重放保存的是最终任务效用、时延、能耗、资源分配及该任务参与的已接受 Matching 操作，未记录被拒绝候选的逐任务分数。",
                "",
                "### 预测需求与 Trigger / Candidate 信用口径",
                "",
                f"8 个 Episode 共检查 2,400 个预测 Slot；保存场景包括 `arrival`、`service_id`、`rho_hat`、`horizons`、`positions_xy`、`task_vehicle_ids` 和 horizon mask。有效预测 horizon Slot={prediction['slots_with_valid_forecast_horizons']}，存在正预测服务事件的 Slot={prediction['slots_with_positive_predicted_service_events']}；首 Slot 张量形状={json.dumps(prediction['prediction_tensor_shapes_first_slot'], ensure_ascii=False)}。因此可构造未来窗口的逐服务需求估计。",
                f"正式候选经济特征宽度={prediction['economic_candidate_feature_dim']}，需求预测窗口={prediction['economic_candidate_forecast_horizons']} 个 horizon；候选输入包含预测新增/移除服务需求，以及由 Cache 更新成本、加载量和释放量计算的特征。因此预测需求和缓存下载代价/移除服务需求代理可用；但候选特征不是经 Matching+KKT 校准的未来净系统效用，当前日志也没有每个 KEEP/UPDATE 目标的独立预测净值。净边际效用仍需在一致的预测窗口上，对 KEEP 和候选分别运行正式预测任务 Matching+KKT 并扣除下载成本，不能直接由命中率推出。",
                f"当前 Run Macro advantage mode={json.dumps(trigger_credit['current_run_macro_actor_advantage_mode'], ensure_ascii=False)}。每节点同一 categorical head 以 candidate 0 表示 KEEP、candidate >0 表示合法 UPDATE 目标；Trigger 和 Candidate 共用当前 Macro SMDP GAE / PPO 回报信用，不是两个独立评分器。故 `CURRENT_TRIGGER_CREDIT_REASSESSMENT=SHARED_RETURN_CREDIT`，但不等于已有显式 KEEP 对每个 UPDATE 候选的预测净效用比较。历史 `cache_trigger_causal_gate.md` 的 stage2d LR Checkpoint 不是本次交替训练 Checkpoint，不能作为当前策略归因证据。",
                "",
                "```ini",
                f"TASK_CLASSIFICATION_COVERAGE = {task_study['replay_task_count']} / {task_preflight['run_actual_task_count_from_node_aggregates']} tasks; all 2400 slots; frozen latest-checkpoint replay, not original per-task Run assignments",
                f"SERVICE_NOT_CACHED_CAUSE = {categories['A_SERVICE_NOT_CACHED']}",
                f"CACHED_BUT_EDGE_INFEASIBLE_CAUSE = {categories['B_CACHED_EDGE_INFEASIBLE']}",
                f"CACHED_AND_EDGE_FEASIBLE_BUT_CLOUD_SELECTED = {categories['C_CACHED_EDGE_FEASIBLE_CLOUD_SELECTED']}",
                f"EDGE_VS_CLOUD_COUNTERFACTUAL_EFFECT = {edge_cloud['evaluated_case_count']} paired full-slot cases; mean_delta_G={_fmt(edge_cloud['mean_delta_G_forced_edge_minus_cloud'])}; unresolved={edge_cloud['unresolved_case_count']}",
                f"PREDICTED_DEMAND_DATA_AVAILABLE = {'YES' if prediction['predicted_future_service_demand_available'] else 'UNRESOLVED'}; valid_forecast_slots={prediction['slots_with_valid_forecast_horizons']}/2400",
                f"PREDICTED_CACHE_MARGINAL_VALUE_ESTIMABLE = {prediction['predicted_cache_marginal_value_status']}",
                f"CURRENT_TRIGGER_CREDIT_REASSESSMENT = {trigger_credit['trigger_candidate_comparison_consistency']}",
                "NEXT_REQUIRED_STEP = For any Trigger change, compare KEEP and legal UPDATE candidates over the same valid forecast window with production Matching+KKT, download cost, and evicted-service opportunity cost; validate on later/currently held-out Training realizations before deployment. Do not treat this 8-Episode replay as policy convergence evidence.",
                "```",
                "",
            ]
        )
    return "\n".join(lines)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN)
    parser.add_argument("--episode-pool", default=DEFAULT_POOL)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--ten-slot-pairs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--json", dest="json_path", default=DEFAULT_JSON)
    parser.add_argument(
        "--task-attribution-only",
        action="store_true",
        help="Add full Training task attribution to the existing JSON/report without rerunning the 40-state intervention study.",
    )
    args = parser.parse_args()
    if args.sample_count <= 0:
        parser.error("--sample-count must be a positive integer")
    if args.ten_slot_pairs < 0:
        parser.error("--ten-slot-pairs must be a non-negative integer")
    report_path = (PROJECT_ROOT / args.report).resolve()
    json_path = (PROJECT_ROOT / args.json_path).resolve()
    started = time.perf_counter()
    if args.task_attribution_only:
        if not json_path.is_file():
            parser.error("--task-attribution-only requires the existing study JSON")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        task_data = _task_attribution_study(args)
        data["task_attribution_run_preflight"] = task_data["run_preflight"]
        data["task_level_attribution"] = task_data["task_level_attribution"]
        data["cached_and_edge_feasible_but_cloud_selected"] = task_data[
            "cached_and_edge_feasible_but_cloud_selected"
        ]
        data["edge_vs_cloud_counterfactual"] = task_data[
            "edge_vs_cloud_counterfactual"
        ]
        data["prediction_and_cache_value_readiness"] = task_data[
            "prediction_and_cache_value_readiness"
        ]
        data["current_trigger_credit_reassessment"] = task_data[
            "current_trigger_credit_reassessment"
        ]
        data["metadata"]["task_attribution_analysis"] = task_data["metadata"]
        data["metadata"]["task_attribution_runtime_seconds"] = float(
            time.perf_counter() - started
        )
    else:
        data = _select_and_evaluate(args)
        data["metadata"]["study_runtime_seconds"] = float(time.perf_counter() - started)
    report_text = _render_report(data)
    json_text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Wrote {report_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
