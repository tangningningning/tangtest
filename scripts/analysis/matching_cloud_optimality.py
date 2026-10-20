#!/usr/bin/env python3
"""Cloud-cause classification and bounded exact assignment reference on Training slots."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common import project_config  # noqa: E402
from decision import task_allocation  # noqa: E402
from ppo_simple.interface import aligned_train_steps, load_environment_context  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "proposed" / "system_config.toml"
POOL_ROOT = ROOT / "environment_data" / "proposed_joint_training"
POOL_MANIFEST_PATH = POOL_ROOT / "formal_pool_manifest.json"
EFFICIENCY_JSON = ROOT / "reports" / "matching_kkt_training_efficiency.json"
RESULT_JSON = ROOT / "reports" / "matching_cloud_optimality.json"
RESULT_MD = ROOT / "reports" / "matching_cloud_optimality.md"
DP_TIME_LIMIT_S = 90.0
UTILITY_TOLERANCE = 1.0e-8
SAMPLES = (
    (10, 20001, 21),
    (10, 20002, 22),
    (10, 20003, 21),
    (14, 20001, 1),
    (14, 20002, 10),
    (14, 20003, 2),
    (20, 20003, 119),
    (20, 20011, 48),
    (20, 20012, 127),
)
EXACT_SLOTS = ((20001, 21), (20002, 22), (20003, 21))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(a: float, b: float, tol: float = UTILITY_TOLERANCE) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1.0e-10, abs_tol=tol)


def finite_map(values: dict[str, Any]) -> bool:
    return all(math.isfinite(float(value)) for mapping in values.values() for value in mapping.values())


def validate_resources(
    evaluation: task_allocation.KKTMatchingEvaluation,
    matching: dict[str, str],
    cache_nodes: dict[str, Any],
    cache_cfg: Any,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    if not evaluation.resource_feasible:
        raise RuntimeError(f"KKT evaluation infeasible: {evaluation.infeasible_reason}")
    if not math.isfinite(float(evaluation.task_utility_sum)):
        raise RuntimeError("non-finite task utility")
    if not finite_map(evaluation.cpu_allocations) or not finite_map(evaluation.bandwidth_allocations):
        raise RuntimeError("non-finite Edge CPU/bandwidth allocation")
    min_cpu = float(project_config.get_config_value(cfg, "algorithm.kkt.min_cpu_fraction"))
    min_bw = float(project_config.get_config_value(cfg, "algorithm.kkt.min_bandwidth_fraction"))
    tol = 1.0e-8
    for node_id, role in zip(cache_nodes["node_ids"], cache_nodes["node_roles"], strict=True):
        node_id = str(node_id)
        assigned = [task_id for task_id, target in matching.items() if target == node_id]
        if not assigned:
            continue
        cpu = evaluation.cpu_allocations.get(node_id, {})
        bw = evaluation.bandwidth_allocations.get(node_id, {})
        if set(cpu) != set(assigned) or set(bw) != set(assigned):
            raise RuntimeError(f"{node_id} allocation keys do not match assigned tasks")
        if any(float(value) < min_cpu - tol for value in cpu.values()):
            raise RuntimeError(f"{node_id} CPU share below configured minimum")
        if any(float(value) < min_bw - tol for value in bw.values()):
            raise RuntimeError(f"{node_id} bandwidth share below configured minimum")
        if float(evaluation.cpu_resource_sums.get(node_id, math.inf)) > 1.0 + tol:
            raise RuntimeError(f"{node_id} CPU resource fraction exceeds 1")
        if float(evaluation.bandwidth_resource_sums.get(node_id, math.inf)) > 1.0 + tol:
            raise RuntimeError(f"{node_id} bandwidth resource fraction exceeds 1")
    cloud_tasks = [task_id for task_id, target in matching.items() if target == task_allocation.CLOUD]
    cloud_alloc = evaluation.cloud_cpu_allocations_cycles_per_s
    cloud_capacity = evaluation.cloud_cpu_capacity_cycles_per_s
    if cloud_tasks:
        if cloud_capacity is None or not math.isfinite(float(cloud_capacity)) or float(cloud_capacity) <= 0.0:
            raise RuntimeError("finite Cloud CPU capacity missing")
        if set(cloud_alloc) != set(cloud_tasks) or not all(math.isfinite(float(v)) and float(v) > 0.0 for v in cloud_alloc.values()):
            raise RuntimeError("Cloud CPU allocations missing or invalid")
        minimum_cloud_cpu = min_cpu * float(cloud_capacity)
        if any(float(value) < minimum_cloud_cpu - max(1.0e-4, float(cloud_capacity) * 1.0e-10) for value in cloud_alloc.values()):
            raise RuntimeError("Cloud CPU allocation below configured minimum fraction")
        cloud_used = math.fsum(float(v) for v in cloud_alloc.values())
        if cloud_used > float(cloud_capacity) + max(1.0e-4, float(cloud_capacity) * 1.0e-10):
            raise RuntimeError("Cloud CPU allocation exceeds finite pool")
    else:
        cloud_used = 0.0
    if set(evaluation.per_task_utility) != set(matching):
        raise RuntimeError("per-task utility coverage differs from matching")
    if not all(math.isfinite(float(v)) for v in evaluation.per_task_utility.values()):
        raise RuntimeError("non-finite per-task utility")
    if not close(math.fsum(evaluation.per_task_utility.values()), evaluation.task_utility_sum):
        raise RuntimeError("sum of per-task utilities differs from total")
    return {
        "edge_cpu_bandwidth_feasible": True,
        "cloud_cpu_used_cycles_per_s": cloud_used,
        "cloud_cpu_capacity_cycles_per_s": cloud_capacity,
        "all_task_utilities_finite": True,
    }


def run_matching(
    context: Any,
    step: Any,
    feasible_targets: dict[str, list[str]],
    cloud_fallback_mode: str = task_allocation.RESOURCE_AWARE_FALLBACK,
) -> tuple[task_allocation.TransferExchangeMatchingResult, float, task_allocation.KKTMatchingEvaluation, dict[str, Any]]:
    service_positions = task_allocation.build_service_node_positions(
        context.cache_nodes, step.actual_vehicle_positions, context.initial_uav
    )
    started = time.perf_counter()
    result = task_allocation.optimize_task_matching(
        list(step.actual_tasks),
        step.actual_vehicle_positions,
        service_positions,
        np.asarray(context.initial_cache, dtype=np.int8).copy(),
        context.cfg,
        context.cache_nodes,
        context.cache_cfg,
        evaluation_mode="incremental",
        kkt_cache=task_allocation.KKTResourceAllocationCache(),
        cloud_fallback_mode=cloud_fallback_mode,
    )
    matching_runtime = time.perf_counter() - started
    if not result.resource_feasible:
        raise RuntimeError("formal Transfer-Exchange Matching returned infeasible")
    if set(result.matching) != {task.task_id for task in step.actual_tasks}:
        raise RuntimeError("Matching does not assign every arrived task exactly once")
    for task_id, target in result.matching.items():
        if target not in feasible_targets[task_id]:
            raise RuntimeError(f"Matching selected illegal target {target} for task {task_id}")
    evaluation = task_allocation.evaluate_matching_with_kkt(
        list(step.actual_tasks),
        result.matching,
        step.actual_vehicle_positions,
        service_positions,
        context.cfg,
        context.cache_nodes,
        context.cache_cfg,
        kkt_cache=task_allocation.KKTResourceAllocationCache(),
    )
    resource_check = validate_resources(
        evaluation, result.matching, context.cache_nodes, context.cache_cfg, context.cfg
    )
    if not close(result.task_utility_sum, evaluation.task_utility_sum):
        raise RuntimeError(
            "formal Matching result and full KKT re-evaluation differ: "
            f"matching={result.task_utility_sum}, full={evaluation.task_utility_sum}"
        )
    resource_check["matching_result_equals_full_kkt_recheck"] = True
    resource_check["service_node_positions"] = {key: np.asarray(value).tolist() for key, value in service_positions.items()}
    return result, matching_runtime, evaluation, resource_check


def classify_cloud(
    context: Any,
    step: Any,
    episode_seed: int,
    feasible_targets: dict[str, list[str]],
    matching: task_allocation.TransferExchangeMatchingResult,
    evaluation: task_allocation.KKTMatchingEvaluation,
) -> dict[str, Any]:
    node_ids = [str(value) for value in context.cache_nodes["node_ids"]]
    cache = np.asarray(context.initial_cache, dtype=np.int8)
    if cache.shape[0] != len(node_ids):
        raise RuntimeError("Cache tensor rows do not align with cache_nodes.node_ids")
    categories = {"EDGE_SELECTED": 0, "FORCED_CLOUD": 0, "VOLUNTARY_CLOUD": 0}
    no_cache_count = 0
    cached_but_path_infeasible_count = 0
    details = []
    for task in sorted(step.actual_tasks, key=lambda value: value.task_id):
        targets = feasible_targets[task.task_id]
        edge_targets = [target for target in targets if target != task_allocation.CLOUD]
        cached_nodes = [
            node_id
            for row, node_id in enumerate(node_ids)
            if int(cache[row, task.service_id]) == 1
        ]
        selected = matching.matching[task.task_id]
        if selected != task_allocation.CLOUD:
            if selected not in edge_targets:
                raise RuntimeError(f"selected Edge node {selected} is not in legal Edge targets")
            category = "EDGE_SELECTED"
            reason = None
        elif edge_targets:
            category = "VOLUNTARY_CLOUD"
            reason = None
        else:
            category = "FORCED_CLOUD"
            if cached_nodes:
                reason = "service_cached_but_edge_path_infeasible"
                cached_but_path_infeasible_count += 1
            else:
                reason = "service_not_cached_on_any_edge"
                no_cache_count += 1
        categories[category] += 1
        details.append(
            {
                "task_id": task.task_id,
                "vehicle_id": task.vehicle_id,
                "service_id": task.service_id,
                "legal_edge_nodes": edge_targets,
                "edge_candidate_count": len(edge_targets),
                "cached_edge_nodes_for_service": cached_nodes,
                "matching_target": selected,
                "category": category,
                "classification_code": (
                    "D" if category == "EDGE_SELECTED"
                    else "C" if category == "VOLUNTARY_CLOUD"
                    else "B" if reason == "service_cached_but_edge_path_infeasible"
                    else "A"
                ),
                "cloud_only": category == "FORCED_CLOUD",
                "voluntary_cloud": category == "VOLUNTARY_CLOUD",
                "forced_cloud_zero_edge_reason": reason,
                "matching_task_utility": float(evaluation.per_task_utility[task.task_id]),
                "matching_task_delay_s": float(evaluation.per_task_delay_s[task.task_id]),
                "matching_vehicle_upload_energy_j": float(evaluation.per_task_vehicle_energy_j[task.task_id]),
            }
        )
    task_count = len(step.actual_tasks)
    cloud_total = categories["FORCED_CLOUD"] + categories["VOLUNTARY_CLOUD"]
    if sum(categories.values()) != task_count or cloud_total != sum(
        selected == task_allocation.CLOUD for selected in matching.matching.values()
    ):
        raise RuntimeError("Cloud/Edge categories do not reconcile with task count")
    return {
        "episode_seed": int(episode_seed),
        "slot": int(step.slot),
        "time_s": float(step.time_s),
        "task_count": task_count,
        **{f"{key.lower()}_count": value for key, value in categories.items()},
        "cloud_total_count": cloud_total,
        "forced_cloud_ratio": categories["FORCED_CLOUD"] / task_count if task_count else 0.0,
        "voluntary_cloud_ratio": categories["VOLUNTARY_CLOUD"] / task_count if task_count else 0.0,
        "forced_share_of_cloud": categories["FORCED_CLOUD"] / cloud_total if cloud_total else None,
        "voluntary_share_of_cloud": categories["VOLUNTARY_CLOUD"] / cloud_total if cloud_total else None,
        "service_not_cached_count": no_cache_count,
        "cached_service_but_path_infeasible_count": cached_but_path_infeasible_count,
        "feasible_edge_task_node_edges": sum(len([target for target in feasible_targets[t.task_id] if target != task_allocation.CLOUD]) for t in step.actual_tasks),
        "service_request_counts": {
            str(service_id): sum(task.service_id == int(service_id) for task in step.actual_tasks)
            for service_id in sorted({str(task.service_id) for task in step.actual_tasks}, key=int)
        },
        "matching_node_task_counts": dict(evaluation.per_node_task_counts),
        "tasks": details,
    }


def evaluate_fixed_assignment(
    context: Any,
    step: Any,
    service_positions: dict[str, np.ndarray],
    matching: dict[str, str],
    tasks: list[task_allocation.Task] | None = None,
) -> tuple[task_allocation.KKTMatchingEvaluation, dict[str, Any]]:
    """Use the formal full KKT evaluator for a fixed diagnostic assignment."""
    selected_tasks = list(step.actual_tasks) if tasks is None else list(tasks)
    evaluation = task_allocation.evaluate_matching_with_kkt(
        selected_tasks,
        matching,
        step.actual_vehicle_positions,
        service_positions,
        context.cfg,
        context.cache_nodes,
        context.cache_cfg,
        kkt_cache=task_allocation.KKTResourceAllocationCache(),
    )
    if not evaluation.resource_feasible:
        return evaluation, {"resource_feasible": False, "reason": evaluation.infeasible_reason}
    return evaluation, validate_resources(
        evaluation, matching, context.cache_nodes, context.cache_cfg, context.cfg
    )


def analyze_solo_cloud_vs_edge(
    context: Any,
    step: Any,
    cloud_item: dict[str, Any],
    service_positions: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare each C-class task in isolation with the same physical inputs."""
    records: list[dict[str, Any]] = []
    for task_row in cloud_item["tasks"]:
        if task_row["classification_code"] != "C":
            continue
        task_id = str(task_row["task_id"])
        task = next(item for item in step.actual_tasks if item.task_id == task_id)
        legal_edges = list(task_row["legal_edge_nodes"])
        solo_cloud, cloud_resources = evaluate_fixed_assignment(
            context, step, service_positions, {task_id: task_allocation.CLOUD}, tasks=[task]
        )
        edge_options = []
        for edge_node in legal_edges:
            edge_eval, edge_resources = evaluate_fixed_assignment(
                context, step, service_positions, {task_id: edge_node}, tasks=[task]
            )
            edge_options.append(
                {
                    "edge_node": edge_node,
                    "resource_feasible": bool(edge_eval.resource_feasible),
                    "task_utility": float(edge_eval.task_utility_sum) if edge_eval.resource_feasible else None,
                    "delay_s": float(edge_eval.per_task_delay_s[task_id]) if edge_eval.resource_feasible else None,
                    "vehicle_upload_energy_j": float(edge_eval.per_task_vehicle_energy_j[task_id]) if edge_eval.resource_feasible else None,
                    "resource_check": edge_resources,
                }
            )
        feasible_edge_options = [item for item in edge_options if item["resource_feasible"]]
        if not solo_cloud.resource_feasible or not feasible_edge_options:
            records.append(
                {
                    "task_id": task_id,
                    "service_id": int(task.service_id),
                    "status": "INCOMPLETE_SOLO_KKT",
                    "solo_cloud_resource_feasible": bool(solo_cloud.resource_feasible),
                    "solo_cloud_reason": solo_cloud.infeasible_reason,
                    "solo_cloud_resource_check": cloud_resources,
                    "legal_edge_nodes": legal_edges,
                    "solo_edge_options": edge_options,
                }
            )
            continue
        best_edge = max(
            feasible_edge_options,
            key=lambda item: (float(item["task_utility"]), str(item["edge_node"])),
        )
        q_cloud = float(solo_cloud.task_utility_sum)
        q_edge = float(best_edge["task_utility"])
        records.append(
            {
                "task_id": task_id,
                "service_id": int(task.service_id),
                "status": "COMPLETE",
                "matching_target": task_row["matching_target"],
                "legal_edge_nodes": legal_edges,
                "solo_q_cloud": q_cloud,
                "solo_q_best_edge": q_edge,
                "solo_best_edge_node": str(best_edge["edge_node"]),
                "solo_delay_cloud_s": float(solo_cloud.per_task_delay_s[task_id]),
                "solo_delay_best_edge_s": float(best_edge["delay_s"]),
                "solo_energy_cloud_j": float(solo_cloud.per_task_vehicle_energy_j[task_id]),
                "solo_energy_best_edge_j": float(best_edge["vehicle_upload_energy_j"]),
                "solo_cloud_minus_best_edge_q": q_cloud - q_edge,
                "solo_cloud_better_than_edge": q_cloud > q_edge + UTILITY_TOLERANCE,
                "solo_edge_better_than_cloud": q_edge > q_cloud + UTILITY_TOLERANCE,
                "solo_best_edge_options": edge_options,
                "solo_cloud_resource_check": cloud_resources,
            }
        )
    complete = [item for item in records if item["status"] == "COMPLETE"]
    return {
        "c_class_task_count": len(records),
        "complete_solo_comparisons": len(complete),
        "incomplete_solo_comparisons": len(records) - len(complete),
        "solo_cloud_better_than_edge_count": sum(bool(item["solo_cloud_better_than_edge"]) for item in complete),
        "solo_edge_better_than_cloud_count": sum(bool(item["solo_edge_better_than_cloud"]) for item in complete),
        "solo_ties_within_tolerance_count": sum(
            not item["solo_cloud_better_than_edge"] and not item["solo_edge_better_than_cloud"]
            for item in complete
        ),
        "solo_edge_better_but_matching_cloud_count": sum(
            item["solo_edge_better_than_cloud"] and item["matching_target"] == task_allocation.CLOUD
            for item in complete
        ),
        "tasks": records,
    }


def analyze_fixed_single_transfers(
    context: Any,
    step: Any,
    cloud_item: dict[str, Any],
    matching: dict[str, str],
    original_evaluation: task_allocation.KKTMatchingEvaluation,
    service_positions: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Move one C-class task to each legal Edge in turn; all others stay assigned."""
    records: list[dict[str, Any]] = []
    original_q = float(original_evaluation.task_utility_sum)
    original_per_task = original_evaluation.per_task_utility
    for task_row in cloud_item["tasks"]:
        if task_row["classification_code"] != "C":
            continue
        task_id = str(task_row["task_id"])
        candidate_results: list[dict[str, Any]] = []
        for edge_node in task_row["legal_edge_nodes"]:
            moved_matching = dict(matching)
            moved_matching[task_id] = str(edge_node)
            new_eval, resources = evaluate_fixed_assignment(
                context, step, service_positions, moved_matching
            )
            if not new_eval.resource_feasible:
                candidate_results.append(
                    {
                        "edge_node": str(edge_node),
                        "resource_feasible": False,
                        "reason": new_eval.infeasible_reason,
                        "resource_check": resources,
                    }
                )
                continue
            delta_target = float(new_eval.per_task_utility[task_id] - original_per_task[task_id])
            delta_others = math.fsum(
                float(new_eval.per_task_utility[other_id] - original_per_task[other_id])
                for other_id in matching
                if other_id != task_id
            )
            delta_system = float(new_eval.task_utility_sum - original_q)
            if not close(delta_system, delta_target + delta_others, tol=5.0e-8):
                raise RuntimeError(
                    f"Single-transfer decomposition failed for task={task_id}, edge={edge_node}: "
                    f"system={delta_system}, target+others={delta_target + delta_others}"
                )
            candidate_results.append(
                {
                    "edge_node": str(edge_node),
                    "resource_feasible": True,
                    "total_task_utility": float(new_eval.task_utility_sum),
                    "delta_q_target_task": delta_target,
                    "delta_q_other_tasks": delta_others,
                    "delta_q_system": delta_system,
                    "decomposition_matches": True,
                    "resource_check": resources,
                }
            )
        feasible_moves = [item for item in candidate_results if item["resource_feasible"]]
        best = max(feasible_moves, key=lambda item: float(item["total_task_utility"])) if feasible_moves else None
        records.append(
            {
                "task_id": task_id,
                "service_id": int(next(task.service_id for task in step.actual_tasks if task.task_id == task_id)),
                "original_target": matching[task_id],
                "legal_edge_nodes": list(task_row["legal_edge_nodes"]),
                "original_total_q": original_q,
                "original_target_task_q": float(original_per_task[task_id]),
                "best_single_transfer": best,
                "candidate_transfers": candidate_results,
                "positive_system_gain_exists": any(
                    item["resource_feasible"] and float(item["delta_q_system"]) > UTILITY_TOLERANCE
                    for item in candidate_results
                ),
                "target_gain_offset_by_other_task_loss": any(
                    item["resource_feasible"]
                    and float(item["delta_q_target_task"]) > UTILITY_TOLERANCE
                    and float(item["delta_q_other_tasks"]) < -UTILITY_TOLERANCE
                    and float(item["delta_q_system"]) <= UTILITY_TOLERANCE
                    for item in candidate_results
                ),
            }
        )
    return {
        "c_class_task_count": len(records),
        "positive_single_transfer_count": sum(bool(item["positive_system_gain_exists"]) for item in records),
        "target_gain_offset_by_others_count": sum(bool(item["target_gain_offset_by_other_task_loss"]) for item in records),
        "tasks": records,
    }


def analyze_nested_load_sweeps(
    context: Any,
    step: Any,
    cloud_item: dict[str, Any],
    service_positions: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Run deterministic nested subsets from one real 20-task slot."""
    all_tasks = sorted(step.actual_tasks, key=lambda task: str(task.task_id))
    representatives = [
        task for task in all_tasks
        if task.task_id in {
            str(item["task_id"])
            for item in cloud_item["tasks"]
            if item["classification_code"] == "C"
        }
    ][:2]
    representative_records = []
    for representative in representatives:
        other_tasks = [task for task in all_tasks if task.task_id != representative.task_id]
        scale_records = []
        for size in (1, 5, 10, 14, 20):
            selected = [representative, *other_tasks[: size - 1]]
            if len(selected) != size:
                raise RuntimeError(f"Nested subset size mismatch: expected {size}, got {len(selected)}")
            feasible = task_allocation.build_feasible_targets(
                selected,
                np.asarray(context.initial_cache, dtype=np.int8),
                context.cache_nodes,
                vehicle_positions=step.actual_vehicle_positions,
                service_node_positions=service_positions,
                cache_config=context.cache_cfg,
                cloud_fallback_mode=task_allocation.RESOURCE_AWARE_FALLBACK,
            )
            matching_result = task_allocation.optimize_task_matching(
                selected,
                step.actual_vehicle_positions,
                service_positions,
                np.asarray(context.initial_cache, dtype=np.int8).copy(),
                context.cfg,
                context.cache_nodes,
                context.cache_cfg,
                evaluation_mode="incremental",
                kkt_cache=task_allocation.KKTResourceAllocationCache(),
                cloud_fallback_mode=task_allocation.RESOURCE_AWARE_FALLBACK,
            )
            record: dict[str, Any] = {
                "task_count": size,
                "task_ids": [str(task.task_id) for task in selected],
                "representative_task_id": str(representative.task_id),
                "representative_service_id": int(representative.service_id),
                "representative_legal_edge_nodes": [
                    target for target in feasible[representative.task_id]
                    if target != task_allocation.CLOUD
                ],
                "representative_cloud_feasible": task_allocation.CLOUD in feasible[representative.task_id],
                "status": "MATCHING_INFEASIBLE" if not matching_result.resource_feasible else "COMPLETE",
            }
            if matching_result.resource_feasible:
                for task_id, target in matching_result.matching.items():
                    if target not in feasible[task_id]:
                        raise RuntimeError(f"Nested Matching selected illegal target {target} for {task_id}")
                evaluation, resources = evaluate_fixed_assignment(
                    context,
                    step,
                    service_positions,
                    matching_result.matching,
                    tasks=selected,
                )
                if not evaluation.resource_feasible or not close(
                    evaluation.task_utility_sum, matching_result.task_utility_sum
                ):
                    raise RuntimeError("Nested Matching full KKT recheck differs")
                record.update(
                    {
                        "representative_matching_target": matching_result.matching[representative.task_id],
                        "matching_total_q": float(matching_result.task_utility_sum),
                        "resource_check": resources,
                    }
                )
            else:
                record["matching_failure_reason"] = matching_result.infeasible_reason
                record["representative_matching_target"] = None
            scale_records.append(record)
        representative_records.append(
            {
                "representative_task_id": str(representative.task_id),
                "representative_service_id": int(representative.service_id),
                "scales": scale_records,
            }
        )
    return {
        "slot_task_count": len(step.actual_tasks),
        "selection_rule": "at most first two C-class tasks in fixed lexical task_id order; every subset includes the representative and fills with remaining tasks in fixed lexical task_id order",
        "task_count_sequence": [1, 5, 10, 14, 20],
        "representatives": representative_records,
    }


def run_edge_gate_comparison(
    context: Any,
    step: Any,
    service_positions: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare the formal solver under the diagnostic Edge-only gate."""
    feasible_targets = task_allocation.build_feasible_targets(
        list(step.actual_tasks),
        np.asarray(context.initial_cache, dtype=np.int8),
        context.cache_nodes,
        vehicle_positions=step.actual_vehicle_positions,
        service_node_positions=service_positions,
        cache_config=context.cache_cfg,
        cloud_fallback_mode=task_allocation.LEGACY_EDGE_GATE,
    )
    for task_id, targets in feasible_targets.items():
        edge_targets = [target for target in targets if target != task_allocation.CLOUD]
        if (edge_targets and task_allocation.CLOUD in targets) or (
            not edge_targets and targets != [task_allocation.CLOUD]
        ):
            raise RuntimeError(f"LEGACY_EDGE_GATE candidate contract invalid for task {task_id}: {targets}")
    result, runtime, evaluation, resources = run_matching(
        context,
        step,
        feasible_targets,
        cloud_fallback_mode=task_allocation.LEGACY_EDGE_GATE,
    )
    return {
        "cloud_fallback_mode": task_allocation.LEGACY_EDGE_GATE,
        "task_count": len(step.actual_tasks),
        "total_task_utility": float(result.task_utility_sum),
        "cloud_count": sum(target == task_allocation.CLOUD for target in result.matching.values()),
        "mean_task_delay_s": statistics.mean(float(value) for value in evaluation.per_task_delay_s.values()),
        "mean_vehicle_upload_energy_j": statistics.mean(
            float(value) for value in evaluation.per_task_vehicle_energy_j.values()
        ),
        "matching_runtime_s": float(runtime),
        "matching": result.matching,
        "feasible_targets": feasible_targets,
        "resource_check": resources,
    }


def summarize_classes(cloud_samples: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for sample in cloud_samples:
        for task in sample["tasks"]:
            counts[str(task["classification_code"])] += 1
    return counts


def write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def enumerate_submasks(mask: int):
    subset = mask
    while True:
        yield subset
        if subset == 0:
            break
        subset = (subset - 1) & mask


def solve_exact_dp(
    tasks: list[task_allocation.Task],
    nodes: list[str],
    feasible_targets: dict[str, list[str]],
    vehicle_positions: dict[str, np.ndarray],
    service_positions: dict[str, np.ndarray],
    context: Any,
) -> dict[str, Any]:
    start = time.perf_counter()
    deadline = start + DP_TIME_LIMIT_S
    n = len(tasks)
    full_mask = (1 << n) - 1
    task_index = {task.task_id: idx for idx, task in enumerate(tasks)}
    node_allowed_mask: dict[str, int] = {}
    for node in nodes:
        bits = 0
        for task in tasks:
            if node in feasible_targets[task.task_id]:
                bits |= 1 << task_index[task.task_id]
        node_allowed_mask[node] = bits

    kkt_cache = task_allocation.KKTResourceAllocationCache()
    q_by_node: dict[str, dict[int, float]] = {}
    evaluated_subsets = 0
    infeasible_subsets: list[dict[str, str]] = []
    timed_out = False
    time_limit_phase = "subset_value_precompute"
    for node in nodes:
        node_q: dict[int, float] = {0: 0.0}
        allowed = node_allowed_mask[node]
        for subset in enumerate_submasks(allowed):
            if subset == 0:
                continue
            if time.perf_counter() >= deadline:
                timed_out = True
                break
            selected_tasks = [tasks[idx] for idx in range(n) if subset & (1 << idx)]
            matching = {task.task_id: node for task in selected_tasks}
            evaluation = task_allocation.evaluate_matching_with_kkt(
                selected_tasks,
                matching,
                vehicle_positions,
                service_positions,
                context.cfg,
                context.cache_nodes,
                context.cache_cfg,
                kkt_cache=kkt_cache,
            )
            evaluated_subsets += 1
            if evaluation.resource_feasible and math.isfinite(float(evaluation.task_utility_sum)):
                validate_resources(
                    evaluation, matching, context.cache_nodes, context.cache_cfg, context.cfg
                )
                node_q[subset] = float(evaluation.task_utility_sum)
            else:
                infeasible_subsets.append(
                    {"node_id": node, "subset_mask": str(subset), "reason": evaluation.infeasible_reason}
                )
        q_by_node[node] = node_q
        print(
            f"  node={node} legal_subsets={len(node_q)-1} evaluated={evaluated_subsets} "
            f"elapsed={time.perf_counter()-start:.1f}s",
            flush=True,
        )
        if timed_out:
            break

    transitions = 0
    parents: list[dict[int, tuple[int, int]]] = []
    dp_complete = not timed_out and len(q_by_node) == len(nodes)
    if dp_complete:
        time_limit_phase = "subset_dynamic_program"
        previous = [-math.inf] * (1 << n)
        previous[0] = 0.0
        for node_idx, node in enumerate(nodes):
            current = [-math.inf] * (1 << n)
            parent: dict[int, tuple[int, int]] = {}
            node_q = q_by_node[node]
            allowed = node_allowed_mask[node]
            for mask in range(1 << n):
                if mask % 128 == 0 and time.perf_counter() >= deadline:
                    timed_out = True
                    break
                submask = mask & allowed
                while True:
                    previous_mask = mask ^ submask
                    prior_value = previous[previous_mask]
                    q_value = node_q.get(submask)
                    if q_value is not None and math.isfinite(prior_value):
                        candidate = prior_value + q_value
                        transitions += 1
                        if candidate > current[mask] + 1.0e-12:
                            current[mask] = candidate
                            parent[mask] = (previous_mask, submask)
                    if submask == 0:
                        break
                    submask = (submask - 1) & (mask & allowed)
            if timed_out:
                dp_complete = False
                break
            previous = current
            parents.append(parent)
        if dp_complete and not math.isfinite(previous[full_mask]):
            raise RuntimeError("DP found no complete legal assignment")

    output: dict[str, Any] = {
        "status": "INCOMPLETE" if timed_out else "NOT_FINALIZED",
        "time_limit_s": DP_TIME_LIMIT_S,
        "time_limit_phase": time_limit_phase if timed_out else None,
        "runtime_s": time.perf_counter() - start,
        "execution_nodes": nodes,
        "node_legal_task_counts": {node: int(node_allowed_mask[node].bit_count()) for node in nodes},
        "node_subset_value_count": {node: len(values) for node, values in q_by_node.items()},
        "subset_kkt_evaluations_completed": evaluated_subsets,
        "subset_resource_infeasible_count": len(infeasible_subsets),
        "subset_resource_infeasible_examples": infeasible_subsets[:20],
        "dp_transitions_evaluated": transitions,
        "kkt_requests": int(kkt_cache.requests),
        "kkt_cache_hits": int(kkt_cache.hits),
        "kkt_cache_misses": int(kkt_cache.misses),
        "kkt_solver_calls": int(kkt_cache.solver_calls),
        "complete_legal_subset_search": False,
    }
    if not dp_complete:
        return output

    assignment: dict[str, str] = {}
    mask = full_mask
    for node_idx in range(len(nodes) - 1, -1, -1):
        node = nodes[node_idx]
        previous_mask, selected_mask = parents[node_idx][mask]
        for idx, task in enumerate(tasks):
            if selected_mask & (1 << idx):
                assignment[task.task_id] = node
        mask = previous_mask
    if mask != 0 or set(assignment) != {task.task_id for task in tasks}:
        raise RuntimeError("DP backtracking did not recover a complete assignment")
    for task_id, target in assignment.items():
        if target not in feasible_targets[task_id]:
            raise RuntimeError(f"DP recovered illegal target {target} for {task_id}")

    node_masks: dict[str, int] = {node: 0 for node in nodes}
    for idx, task in enumerate(tasks):
        node_masks[assignment[task.task_id]] |= 1 << idx
    subset_sum = math.fsum(q_by_node[node][node_masks[node]] for node in nodes)
    dp_value = float(previous[full_mask])
    if not close(subset_sum, dp_value):
        raise RuntimeError(f"DP value differs from recovered subset sum: {dp_value} vs {subset_sum}")

    exact_eval_cache = task_allocation.KKTResourceAllocationCache()
    exact_eval = task_allocation.evaluate_matching_with_kkt(
        tasks,
        assignment,
        vehicle_positions,
        service_positions,
        context.cfg,
        context.cache_nodes,
        context.cache_cfg,
        kkt_cache=exact_eval_cache,
    )
    exact_resources = validate_resources(
        exact_eval, assignment, context.cache_nodes, context.cache_cfg, context.cfg
    )
    if not close(dp_value, exact_eval.task_utility_sum):
        raise RuntimeError(
            "DP subset sum and complete Matching+KKT evaluation differ: "
            f"dp={dp_value}, full={exact_eval.task_utility_sum}"
        )
    if time.perf_counter() >= deadline:
        output["status"] = "INCOMPLETE"
        output["time_limit_phase"] = "final_full_kkt_verification"
        output["runtime_s"] = time.perf_counter() - start
        return output

    output.update(
        {
            "status": "EXACT_OPTIMUM_VERIFIED",
            "runtime_s": time.perf_counter() - start,
            "time_limit_phase": None,
            "complete_legal_subset_search": True,
            "exact_task_utility": dp_value,
            "exact_matching": assignment,
            "exact_node_task_counts": dict(exact_eval.per_node_task_counts),
            "exact_per_task_utility": {key: float(value) for key, value in exact_eval.per_task_utility.items()},
            "exact_per_task_delay_s": {key: float(value) for key, value in exact_eval.per_task_delay_s.items()},
            "exact_per_task_vehicle_energy_j": {key: float(value) for key, value in exact_eval.per_task_vehicle_energy_j.items()},
            "exact_cloud_cpu_allocations_cycles_per_s": exact_eval.cloud_cpu_allocations_cycles_per_s,
            "exact_cloud_cpu_used_cycles_per_s": exact_eval.cloud_cpu_used_cycles_per_s,
            "exact_cloud_cpu_capacity_cycles_per_s": exact_eval.cloud_cpu_capacity_cycles_per_s,
            "exact_resource_check": exact_resources,
            "verification_kkt_solver_calls": int(exact_eval_cache.solver_calls),
            "verification_kkt_cache_hits": int(exact_eval_cache.hits),
            "node_subset_masks_in_final_assignment": {node: str(value) for node, value in node_masks.items()},
        }
    )
    return output


def ratio(value: int, count: int) -> float | None:
    return None if count == 0 else float(value) / float(count)


def make_report(payload: dict[str, Any]) -> str:
    cloud = payload["cloud_analysis"]
    exact = payload["exact_analysis"]
    solo = payload["solo_cloud_edge_analysis"]
    transfers = payload["fixed_single_transfer_analysis"]
    load_sweeps = payload["nested_load_sweep_analysis"]
    edge_gate = payload["edge_gate_analysis"]
    forced = sum(item["forced_cloud_count"] for item in cloud)
    voluntary = sum(item["voluntary_cloud_count"] for item in cloud)
    edge = sum(item["edge_selected_count"] for item in cloud)
    no_cache = sum(item["service_not_cached_count"] for item in cloud)
    path_fail = sum(item["cached_service_but_path_infeasible_count"] for item in cloud)
    exact_done = [item for item in exact if item.get("status") == "EXACT_OPTIMUM_VERIFIED"]
    gaps = [float(item["gap_percent"]) for item in exact_done if item.get("gap_percent") is not None]
    mean_gap = statistics.mean(gaps) if gaps else None
    max_gap = max(gaps) if gaps else None
    changed_tasks = sum(int(item["changed_task_count"]) for item in exact_done)
    class_counts = payload["class_counts"]
    solo_task_records = [task for item in solo for task in item["tasks"]]
    solo_complete = [task for task in solo_task_records if task.get("status") == "COMPLETE"]
    solo_cloud_better = sum(bool(task["solo_cloud_better_than_edge"]) for task in solo_complete)
    solo_edge_better = sum(bool(task["solo_edge_better_than_cloud"]) for task in solo_complete)
    solo_edge_better_but_multi_cloud = sum(
        bool(task["solo_edge_better_than_cloud"]) and task.get("matching_target") == task_allocation.CLOUD
        for task in solo_complete
    )
    transfer_tasks = [task for item in transfers for task in item["tasks"]]
    transfer_positive = sum(bool(task["positive_system_gain_exists"]) for task in transfer_tasks)
    transfer_offsets = sum(bool(task["target_gain_offset_by_other_task_loss"]) for task in transfer_tasks)
    nested_transition_task_ids: list[str] = []
    for slot in load_sweeps:
        for representative in slot["representatives"]:
            scales = [item for item in representative["scales"] if item.get("status") == "COMPLETE"]
            if scales and scales[0].get("representative_matching_target") != task_allocation.CLOUD and any(
                item.get("representative_matching_target") == task_allocation.CLOUD for item in scales[1:]
            ):
                nested_transition_task_ids.append(
                    f"{slot['episode_seed']}/{slot['slot']}/task{representative['representative_task_id']}"
                )
    resource_aware_q = math.fsum(float(item["resource_aware_total_q"]) for item in edge_gate)
    edge_gate_q = math.fsum(float(item["total_task_utility"]) for item in edge_gate)
    resource_aware_cloud_count = sum(int(item["resource_aware_cloud_count"]) for item in edge_gate)
    edge_gate_cloud_count = sum(int(item["cloud_count"]) for item in edge_gate)
    total_tasks = sum(int(item["task_count"]) for item in edge_gate)
    resource_aware_mean_delay = math.fsum(
        float(item["resource_aware_mean_delay_s"]) * int(item["task_count"]) for item in edge_gate
    ) / total_tasks
    edge_gate_mean_delay = math.fsum(
        float(item["mean_task_delay_s"]) * int(item["task_count"]) for item in edge_gate
    ) / total_tasks
    resource_aware_mean_energy = math.fsum(
        float(item["resource_aware_mean_energy_j"]) * int(item["task_count"]) for item in edge_gate
    ) / total_tasks
    edge_gate_mean_energy = math.fsum(
        float(item["mean_vehicle_upload_energy_j"]) * int(item["task_count"]) for item in edge_gate
    ) / total_tasks
    edge_gate_delta_q = resource_aware_q - edge_gate_q
    if solo_edge_better_but_multi_cloud > 0 and transfer_offsets > 0:
        cause = "MIXED_SOLO_EDGE_PREFERENCE_WITH_OTHER_TASK_RESOURCE_LOSS"
    elif solo_cloud_better > solo_edge_better:
        cause = "SINGLE_TASK_CLOUD_ADVANTAGE_PRESENT_IN_MORE_C_CASES"
    elif solo_edge_better > solo_cloud_better:
        cause = "SINGLE_TASK_EDGE_ADVANTAGE_OFTEN_OVERTURNED_IN_MULTI_TASK_MATCHING"
    else:
        cause = "MIXED_OR_TIED_SINGLE_TASK_PREFERENCE"
    if exact_done and len(gaps) == len(exact_done) and max_gap <= 1.0e-5:
        quality = "全部已完成的 10 任务案例均在数值容差内达到精确最优；结论仅适用于指定的三个 Slot。"
        next_step = "保留 Transfer-Exchange；这三个案例不足以支持替换。只有后续研究确有需要时再扩大任务规模核查。"
    elif exact_done:
        quality = f"{sum(g <= 1.0e-5 for g in gaps)}/{len(exact_done)} 个已验证案例达到精确最优；最大相对 gap 为 {max_gap:.6g}%。"
        next_step = "不因这三个案例替换 Matching；先检查最大 gap 案例的交换邻域与候选覆盖。"
    else:
        quality = "精确搜索未完成或未通过验证，Matching 的全局最优差距尚未确定。"
        next_step = "先解决 DP 超时或一致性阻塞，再判断 Matching 质量。"
    cloud_slot_count = sum(item["forced_cloud_count"] + item["voluntary_cloud_count"] for item in cloud)
    total_tasks = sum(item["task_count"] for item in cloud)
    slots_with_voluntary_cloud = sum(item["voluntary_cloud_count"] > 0 for item in cloud)
    if voluntary == 0:
        voluntary_answer = "No task had a legal Edge candidate while being assigned to Cloud."
    else:
        voluntary_answer = f"{total_tasks} 个到达任务中有 {voluntary} 个存在合法 Edge 候选但最终分配到 Cloud；该现象在 {slots_with_voluntary_cloud}/9 个 Slot 出现。"
    if forced == voluntary:
        comparison = "Forced 与 Voluntary 数量相同"
    elif forced > voluntary:
        comparison = "Forced Cloud 较多"
    else:
        comparison = "Voluntary Cloud 较多"
    main_cause_text = (
        f"九个 Slot 中 Forced Cloud 为 {forced}、Voluntary Cloud 为 {voluntary}，{comparison}；"
        f"两类各占 Cloud 任务 {100.0 * forced / cloud_slot_count:.1f}% 和 {100.0 * voluntary / cloud_slot_count:.1f}%。"
        f"Voluntary Cloud 出现在 {slots_with_voluntary_cloud}/9 个 Slot。"
        f"Forced Cloud 中 {no_cache} 个请求在任何 Edge 节点均未缓存，{path_fail} 个请求虽有缓存但未通过当前 Edge 路径检查。"
        f"对 C 类任务，单任务 Cloud/Edge 对照为 {solo_cloud_better}/{solo_edge_better} 个；固定其他分配的转移中有 {transfer_offsets} 个出现目标收益被其他任务损失抵消。"
    )
    def percent(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.1%}"
    def number(value: float | None, digits: int = 6) -> str:
        return "NOT_AVAILABLE" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Training Slot Cloud 卸载原因与 Matching 最优性对照",
        "",
        f"Generated: {payload['created_at_utc']}",
        "",
        "## 数据、配置与复现状态",
        "",
        f"- 复用 `{payload['input_efficiency_json']}` 中预选的 9 个 Training Slot；未重新扫描整个 Pool，未读取 Validation/Test。",
        f"- 当前配置 SHA-256：`{payload['current_config_sha256']}`；效率诊断记录：`{payload['previous_config_sha256']}`；相同：`{payload['config_hash_matches_previous']}`。三个 10 任务 Slot 均在当前配置下重新运行 Matching。",
        f"- Matching/KKT 源码 SHA-256：`{payload['matching_source_sha256']}`；Pool Manifest SHA-256：`{payload['pool_manifest_sha256']}`。九个 Episode Manifest 与数据 artifact 哈希复核通过。",
        f"- Python：`{payload['python_executable']}`；Conda：`{payload['conda_environment']}`；分支/HEAD：`{payload['branch']}` / `{payload['head']}`。",
        f"- Cloud fallback：`{payload['contracts']['cloud_fallback_mode']}`；状态参考：`{payload['state_reference']}`。每个 Slot 使用相同的正式初始 Cache/UAV 参考、真实任务和实际车辆位置。",
        "",
        f"- 当前效用权重为 delay/energy `{payload['contracts']['omega_delay']}`/`{payload['contracts']['omega_energy']}`；Cache/UAV 系数为 `{payload['contracts']['lambda_cache']}`/`{payload['contracts']['lambda_movement']}`。Matching 对照保持 Cache/UAV 固定，因此仅比较任务效用 Q；成本相同，不会随卸载分配重复扣计。",
        "",
        "## A. 九个 Slot 的 Cloud 原因分类",
        "",
        "`FORCED_CLOUD`：没有合法 Edge 候选且最终选择 Cloud。`VOLUNTARY_CLOUD`：存在合法 Edge 候选但最终选择 Cloud。`EDGE_SELECTED`：选择合法 Edge。Forced/Voluntary ratio 以该 Slot 到达任务数为分母；另列 Forced Cloud 占 Cloud 的比例。",
        "",
        "| 任务数 | Episode / Slot | Edge | Forced Cloud | Voluntary Cloud | Cloud 总数 | Forced / Tasks | Voluntary / Tasks | Forced / Cloud | 服务未缓存 | 有缓存但路径失败 | 可行 Edge 边 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in cloud:
        fields = [
            item['task_count'], f"{item['episode_seed']} / {item['slot']}",
            item['edge_selected_count'], item['forced_cloud_count'], item['voluntary_cloud_count'],
            item['cloud_total_count'], percent(item['forced_cloud_ratio']),
            percent(item['voluntary_cloud_ratio']), percent(item['forced_share_of_cloud']),
            item['service_not_cached_count'], item['cached_service_but_path_infeasible_count'],
            item['feasible_edge_task_node_edges'],
        ]
        lines.append("| " + " | ".join(str(value) for value in fields) + " |")
    lines.extend([
        "",
        f"- 汇总：Edge `{edge}`，Forced Cloud `{forced}`，Voluntary Cloud `{voluntary}`，Cloud 总数 `{forced + voluntary}`。{main_cause_text}",
        f"- 有合法 Edge 候选时仍选择 Cloud：{voluntary_answer}",
        "- `build_feasible_targets` 对缓存服务还会核验节点 CPU、带宽、信道增益及上传速率是否为有限正值。因此“服务未缓存”和“存在缓存但路径检查未通过”可区分；后一类不再臆测具体是距离、信道或单独资源原因。",
        "- 同任务数 Slot 的 Cloud 比例差异对应请求服务、缓存覆盖、物理可行 Edge 边集合和 Matching 的整体任务效用选择差异。Cloud 比例本身不能说明 Matching 有误。",
        "",
        "### 每任务分类",
        "",
    ])
    for item in cloud:
        lines.extend([
            f"#### {item['episode_seed']} / Slot {item['slot']}",
            "",
            "| Task ID | Service | 合法 Edge 节点 | Edge 候选数 | Matching 节点 | 类别 | 代码 | 零 Edge 原因 | Task utility | Delay (s) | Vehicle energy (J) |",
            "|---|---:|---|---:|---|---|---|---|---:|---:|---:|",
        ])
        for task in item["tasks"]:
            legal = ", ".join(task["legal_edge_nodes"]) if task["legal_edge_nodes"] else "—"
            lines.append(
                f"| {task['task_id']} | {task['service_id']} | {legal} | {task['edge_candidate_count']} | {task['matching_target']} | {task['category']} | {task['classification_code']} | {task['forced_cloud_zero_edge_reason'] or '—'} | {task['matching_task_utility']:.8f} | {task['matching_task_delay_s']:.6f} | {task['matching_vehicle_upload_energy_j']:.8f} |"
            )
        lines.append("")

    lines.extend([
        "## B. C 类任务：无竞争时 Cloud 与 Edge 对照",
        "",
        "每个 C 类任务都固定原 Slot 的 Cache、UAV、车辆位置、路径和任务工作量，单独作为唯一任务分别调用正式 `evaluate_matching_with_kkt()`。Q 越高代表正式任务效用越高。该对照识别单任务偏好，但不能单独说明原 Matching 的多任务因果。",
        "",
        "| Episode/Slot | Task | Service | Cloud Q | Best Edge Q | Best Edge | Cloud delay | Edge delay | Cloud energy | Edge energy | 单任务偏好 |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ])
    for item in solo:
        for task in item["tasks"]:
            if task.get("status") != "COMPLETE":
                lines.append(f"| {item['episode_seed']}/{item['slot']} | {task['task_id']} | {task['service_id']} | INCOMPLETE | INCOMPLETE | — | — | — | — | — | {task.get('status')} |")
                continue
            preference = "Cloud" if task["solo_cloud_better_than_edge"] else "Edge" if task["solo_edge_better_than_cloud"] else "tie"
            lines.append(
                f"| {item['episode_seed']}/{item['slot']} | {task['task_id']} | {task['service_id']} | {task['solo_q_cloud']:.8f} | {task['solo_q_best_edge']:.8f} | {task['solo_best_edge_node']} | {task['solo_delay_cloud_s']:.6f} | {task['solo_delay_best_edge_s']:.6f} | {task['solo_energy_cloud_j']:.8f} | {task['solo_energy_best_edge_j']:.8f} | {preference} |"
            )
    lines.extend([
        "",
        f"- 完成 `{len(solo_complete)}/{len(solo_task_records)}` 个单任务比较；Cloud 单任务效用更高 `{solo_cloud_better}` 个，Edge 更高 `{solo_edge_better}` 个，容差内平局 `{len(solo_complete)-solo_cloud_better-solo_edge_better}` 个。单任务 Edge 更高、原 Slot 却选 Cloud 的任务有 `{solo_edge_better_but_multi_cloud}` 个。",
        "- 若原 Slot 选择 Cloud 且单任务 Cloud Q 已更高，多任务竞争不是 Cloud 选择的必要条件；若单任务 Edge 更高，才需进一步结合下节固定节点转移检查其他任务的资源外部效应或 Matching 局部解。",
        "",
        "## C. 固定其他卸载节点的单任务 Transfer",
        "",
        "每个 C 类目标任务分别尝试转移到其每个合法 Edge；其他任务卸载节点保持 Matching 原分配，然后用正式 KKT 重算全部任务。此为有限的一步局部对照，不是全局最优搜索。",
        "",
        "| Episode/Slot | Task | 原节点 | 最佳 Edge | ΔQ 目标任务 | ΔQ 其他任务 | ΔQ 系统 | 分解核验 | 解读 |",
        "|---|---:|---|---|---:|---:|---:|---|---|",
    ])
    for item in transfers:
        for task in item["tasks"]:
            best = task.get("best_single_transfer")
            if best is None:
                lines.append(f"| {item['episode_seed']}/{item['slot']} | {task['task_id']} | {task['original_target']} | — | — | — | — | — | 无资源可行 Edge 转移 |")
                continue
            if float(best["delta_q_target_task"]) > UTILITY_TOLERANCE and float(best["delta_q_other_tasks"]) < -UTILITY_TOLERANCE:
                interpretation = "目标获益被其他任务损失部分/全部抵消"
            elif float(best["delta_q_system"]) > UTILITY_TOLERANCE:
                interpretation = "该单次转移提高总 Q"
            elif float(best["delta_q_target_task"]) <= UTILITY_TOLERANCE:
                interpretation = "目标自身未获益"
            else:
                interpretation = "总 Q 未提高"
            lines.append(
                f"| {item['episode_seed']}/{item['slot']} | {task['task_id']} | {task['original_target']} | {best['edge_node']} | {best['delta_q_target_task']:.8f} | {best['delta_q_other_tasks']:.8f} | {best['delta_q_system']:.8f} | {best['decomposition_matches']} | {interpretation} |"
            )
    lines.extend([
        "",
        f"- C 类目标中存在正系统净 Q 单次转移的有 `{transfer_positive}/{len(transfer_tasks)}` 个；至少一个 Edge 转移表现为目标任务获益、其他任务损失且总 Q 不增的有 `{transfer_offsets}` 个。该数量是本地转移证据，不能单独证明 Cloud 是唯一或全局最优选择。",
        "",
        "## D. 20 任务 Slot 的嵌套负载对照",
        "",
        "每个 20 任务 Slot 最多取两个 C 类代表任务；子集总含该代表任务，其余任务按固定词法 Task ID 顺序补入。它是同一真实 Slot 构造的嵌套子集，不是独立自然到达 Slot，也不代表普遍负载阈值。",
        "",
        "| Episode/Slot | 代表任务 | Service | 任务数 | 合法 Edge | Cloud 可行 | Matching 节点 | 状态 |",
        "|---|---:|---:|---:|---|---|---|---|",
    ])
    for slot in load_sweeps:
        for representative in slot["representatives"]:
            for scale in representative["scales"]:
                edges = ", ".join(scale["representative_legal_edge_nodes"]) or "—"
                target = scale.get("representative_matching_target") or "—"
                lines.append(
                    f"| {slot['episode_seed']}/{slot['slot']} | {representative['representative_task_id']} | {representative['representative_service_id']} | {scale['task_count']} | {edges} | {scale['representative_cloud_feasible']} | {target} | {scale['status']} |"
                )
    lines.extend([
        "",
        f"- 按上述固定任务补入顺序，代表任务从 1 任务 Edge 解转为任一更高规模 Cloud 解的样本数：`{len(nested_transition_task_ids)}`；任务 ID：`{', '.join(nested_transition_task_ids) if nested_transition_task_ids else '—'}`。该观察只描述选定 Slot 与补入顺序。",
        "",
        "## E. RESOURCE_AWARE_FALLBACK 与 LEGACY_EDGE_GATE",
        "",
        "两种模式均调用同一正式 Transfer–Exchange Matching 与 KKT。Edge gate 仅作对照：有物理可行 Edge 时不允许 Cloud。成本状态固定，故表中比较任务效用 Q。",
        "",
        "| Episode/Slot | 任务数 | Resource-aware Q | Edge-gate Q | ΔQ (RA−gate) | Cloud 数 RA | Cloud 数 gate | Mean delay RA/gate (s) | Mean vehicle energy RA/gate (J) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in edge_gate:
        lines.append(
            f"| {item['episode_seed']}/{item['slot']} | {item['task_count']} | {item['resource_aware_total_q']:.8f} | {item['total_task_utility']:.8f} | {item['resource_aware_total_q']-item['total_task_utility']:.8f} | {item['resource_aware_cloud_count']} | {item['cloud_count']} | {item['resource_aware_mean_delay_s']:.6f}/{item['mean_task_delay_s']:.6f} | {item['resource_aware_mean_energy_j']:.8f}/{item['mean_vehicle_upload_energy_j']:.8f} |"
        )
    lines.extend([
        "",
        f"- 九个 Slot 总 ΔQ (Resource-aware − Edge-gate)：`{edge_gate_delta_q:.8f}`；任务数加权平均时延 RA/gate：`{resource_aware_mean_delay:.6f}/{edge_gate_mean_delay:.6f}` s；车辆上传能耗：`{resource_aware_mean_energy:.8f}/{edge_gate_mean_energy:.8f}` J；Cloud 数：`{resource_aware_cloud_count}/{edge_gate_cloud_count}`。",
        "- 这两组结果仍是局部搜索输出。禁止 Cloud 后的差值不能证明任一模式达到全局最优，也不能单独归因为负载竞争。",
        "",
    ])

    lines.extend([
        "## F. 三个 10 任务 Slot 的精确最优性对照",
        "",
        "### DP 分解条件",
        "",
        f"- 当前模型为 `{payload['contracts']['cross_slot_task_model']}`；频谱合同为 `{payload['contracts']['inter_node_spectrum']}`，干扰为 `{payload['contracts']['interference']}`。Edge KKT 以节点为单位独立求解，每节点 CPU 与带宽资源各自受总份额约束。",
        "- Cloud CPU 是 Cloud 节点内有限共享资源，由正式 Cloud KKT 对该子集的全部 Cloud 任务共同分配；Cloud 上行沿用正式每任务等效速率和时延因子，不与 Edge KKT 带宽耦合。",
        "- 每个任务分配到一个合法 Edge 或 Cloud。Edge 节点间没有共享资源或互扰，唯一的跨任务共享子问题在 Cloud 内，故每个节点子集的正式 KKT 最优效用可相加。DP 的目标是正式任务效用总和，Cache/UAV 固定。",
        "- 对每节点的全部合法任务子集调用 `evaluate_matching_with_kkt`；空集效用为 0，KKT 不可行的子集禁用。每个 Slot 的 DP 上限为 90 秒。只有完成全量子集枚举、DP 回溯与完整 KKT 复核才标记 `EXACT_OPTIMUM_VERIFIED`。",
        f"- Matching 计时协议：先对 Episode {payload['matching_timing_protocol']['warmup']['episode_seed']} / Slot {payload['matching_timing_protocol']['warmup']['slot']} 执行一次预热（耗时 {payload['matching_timing_protocol']['warmup']['optimizer_runtime_s_excluded_from_steady_state']:.6f} s，不计入稳态样本）；三个精确对照 Slot 各重复 {payload['matching_timing_protocol']['exact_slot_repetitions']} 次，表中采用 optimizer-only 耗时中位数。首次冷启动不是稳态 Matching 时间。",
        "",
        "| Episode / Slot | Matching Q | Exact Q | Gap | Matching 时间 | Exact DP 时间 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for case in exact:
        is_verified = case.get("status") == "EXACT_OPTIMUM_VERIFIED"
        matching_q = number(case.get("matching_task_utility"), 9)
        exact_q = number(case.get("exact_task_utility"), 9) if is_verified else "NOT_AVAILABLE"
        gap = f"{case['gap_percent']:.6f}%" if is_verified and case.get("gap_percent") is not None else "NOT_AVAILABLE"
        matching_time = f"{case['matching_runtime_s']:.6f} s" if case.get("matching_runtime_s") is not None else "NOT_AVAILABLE"
        exact_time = f"{case['runtime_s']:.3f} s" if case.get("runtime_s") is not None else "NOT_AVAILABLE"
        lines.append(f"| {case['episode_seed']} / {case['slot']} | {matching_q} | {exact_q} | {gap} | {matching_time} | {exact_time} | {case['status']} |")
    mean_gap_text = "NOT_AVAILABLE" if mean_gap is None else f"{mean_gap:.6f}%"
    max_gap_text = "NOT_AVAILABLE" if max_gap is None else f"{max_gap:.6f}%"
    lines.extend([
        "",
        f"- 尝试 `{len(exact)}` 个状态，完整验证 `{len(exact_done)}` 个；Verified 状态平均 gap `{mean_gap_text}`，最大 gap `{max_gap_text}`。",
        f"- 与 Matching 不同的任务分配数：`{changed_tasks}`；逐 Slot 明细如下。",
        "- Matching 与 DP 解分别经完整 `evaluate_matching_with_kkt` 复核；DP 子集效用和、递推结果、完整评价在 1e-8 容差内一致，且 Exact Q 不低于 Matching。资源核验包括 Edge CPU/带宽总份额及最小份额、Cloud 每任务最小 CPU 份额、Cloud 有限 CPU 总量和任务效用有限性。",
        "",
        "### 最优解中发生变化的任务",
        "",
    ])
    for case in exact_done:
        lines.extend([
            f"#### Episode {case['episode_seed']} / Slot {case['slot']}",
            "",
            f"- Matching Cloud Task IDs：`{', '.join(case['matching_cloud_task_ids']) or '—'}`；Exact Cloud Task IDs：`{', '.join(case['exact_cloud_task_ids']) or '—'}`；集合相同：`{case['cloud_task_set_identical']}`。",
            "",
            "| Task | Matching 节点 | Exact 节点 | Matching utility | Exact utility | 差值 |",
            "|---|---|---|---:|---:|---:|",
        ])
        for change in case["task_assignment_differences"]:
            lines.append(
                f"| {change['task_id']} | {change['matching_target']} | {change['exact_target']} | {change['matching_task_utility']:.8f} | {change['exact_task_utility']:.8f} | {change['task_utility_delta']:.8f} |"
            )
        if not case["task_assignment_differences"]:
            lines.append("| — | assignment identical | assignment identical | — | — | — |")
        lines.append("")

    if exact_done and len(gaps) == len(exact_done):
        replacement_judgment = (
            f"当前三个指定 Slot 不支持更换 Matching；最大相对 gap 为 {max_gap:.6f}%。"
        )
        gnn_judgment = "NO_EVIDENCE_FROM_THIS_DIAGNOSTIC"
    else:
        replacement_judgment = "精确搜索未全部完成，现有证据不足以判断是否更换 Matching。"
        gnn_judgment = "UNRESOLVED"
    lines.extend([
        "## 结论",
        "",
        f"- 四类计数：A 未缓存被迫 Cloud `{class_counts['A']}`；B 已缓存但无物理可行 Edge `{class_counts['B']}`；C 有合法 Edge 仍选 Cloud `{class_counts['C']}`；D 选择 Edge `{class_counts['D']}`。Cloud 机制结论：`{cause}`。",
        f"- 有合法 Edge 候选仍选择 Cloud：{voluntary_answer}",
        f"- 单任务对照中 Cloud 优于最佳 Edge `{solo_cloud_better}` 次、Edge 更优 `{solo_edge_better}` 次；单任务 Edge 更优而原多任务 Matching 仍选 Cloud `{solo_edge_better_but_multi_cloud}` 次。",
        f"- 固定其他任务节点的转移：{transfer_offsets} 个目标任务本身获益却伴随其他任务损失且系统 Q 未升高；存在正系统净 Q 单步 Edge 转移的 C 类目标 `{transfer_positive}/{len(transfer_tasks)}`。",
        f"- 嵌套负载对照中观察到 Edge→Cloud 的代表任务数 `{len(nested_transition_task_ids)}`。这里只对应指定真实 Slot 与 Task ID 加入顺序，不构成通用 Cloud 负载阈值。",
        f"- Resource-aware 相对 Edge-gate 的总任务效用差为 `{edge_gate_delta_q:.8f}`；两种模式都仅是局部 Matching 结果，不构成全局最优证明。",
        f"- Matching 最优性：{quality}",
        f"- 更换 Matching：{replacement_judgment}结论仅适用于完整核验的三个指定 10 任务 Slot，不能外推到更大任务数或其他状态。",
        f"- GNN 必要性：`{gnn_judgment}`。若发现明显 gap，应先定位交换邻域或候选覆盖，再考虑学习式方法。",
        f"- 下一步：{next_step}",
        "",
        "## 代码依据与边界",
        "",
        "依据 `task_allocation.py` 的 `build_feasible_targets`、`_edge_path_is_feasible`、`_allocate_phase_resources_kkt`、`_allocate_cloud_cpu_kkt`、`evaluate_matching_with_kkt` 和 `optimize_task_matching`；配置为 independent-per-slot、orthogonal-independent、no-interference 和 RESOURCE_AWARE_FALLBACK。",
        "只读取 Training 数据；未运行 PPO、Validation、Test 或完整 Episode。未修改正式 Matching/KKT、配置、Reward 或环境数据。",
        "",
        "```text",
        f"TRAINING_SLOTS_ANALYZED = {len(cloud)}",
        f"CACHE_MISS_FORCED_CLOUD_COUNT = {class_counts['A']}",
        f"CACHED_BUT_EDGE_INFEASIBLE_COUNT = {class_counts['B']}",
        f"EDGE_FEASIBLE_CLOUD_SELECTED_COUNT = {class_counts['C']}",
        f"EDGE_SELECTED_COUNT = {class_counts['D']}",
        f"VOLUNTARY_CLOUD_EXISTS = {'YES' if class_counts['C'] > 0 else 'NO'}",
        f"SOLO_CLOUD_BETTER_THAN_EDGE_COUNT = {solo_cloud_better}",
        f"SOLO_EDGE_BETTER_BUT_MULTI_CLOUD_COUNT = {solo_edge_better_but_multi_cloud}",
        f"RESOURCE_AWARE_VS_EDGE_GATE_DELTA_Q = {edge_gate_delta_q:.8f}",
        f"EXACT_SLOTS_ATTEMPTED = {len(exact)}",
        f"EXACT_SLOTS_VERIFIED = {len(exact_done)}",
        f"MATCHING_OPTIMALITY_GAP_MEAN = {mean_gap_text}",
        f"MATCHING_OPTIMALITY_GAP_MAX = {max_gap_text}",
        f"MATCHING_EXACT_SOLUTION_DIFFERENCES = {changed_tasks}",
        f"CLOUD_SELECTION_MAIN_CAUSE = {cause}",
        f"MATCHING_QUALITY_ASSESSMENT = {quality}",
        f"GNN_NECESSITY_ASSESSMENT = {gnn_judgment}",
        f"RECOMMENDED_NEXT_STEP = {next_step}",
        "PPO_TRAINING_EXECUTED = NO",
        "FORMAL_ALGORITHM_MODIFIED = NO",
        "FORMAL_CONFIG_MODIFIED = NO",
        "```",
        "",
    ])
    return "\n".join(lines)

def main() -> int:
    if Path(sys.prefix).name.lower() != "adaptive_cache_mec":
        raise RuntimeError(f"Expected adaptive_cache_mec, got {sys.prefix}")
    previous = json.loads(EFFICIENCY_JSON.read_text(encoding="utf-8"))
    pool_manifest = json.loads(POOL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if pool_manifest.get("split") != "training":
        if str(pool_manifest.get("split", "")).lower() != "train":
            raise RuntimeError("Training pool manifest split mismatch")
    cfg = project_config.load_config(str(CONFIG_PATH.relative_to(ROOT)))
    project_config.validate_for_formal_experiment(cfg)
    expected_mode = task_allocation.RESOURCE_AWARE_FALLBACK
    current_config_sha = sha256_file(CONFIG_PATH)
    if current_config_sha != previous["configuration"]["sha256"]:
        raise RuntimeError(
            "The fixed efficiency input was generated under a different config; "
            "do not combine incompatible local configuration results."
        )
    if previous.get("method", {}).get("state_reference") != "FROZEN_INITIAL_CACHE_UAV_REFERENCE":
        raise RuntimeError("Fixed efficiency samples do not identify FROZEN_INITIAL_CACHE_UAV_REFERENCE")
    sample_lookup = {
        (int(sample["episode_seed"]), int(sample["slot"])): sample
        for group in previous["samples_by_task_count"].values()
        for sample in group
    }
    samples: dict[tuple[int, int], dict[str, Any]] = {}
    matching_warmup_record: dict[str, Any] | None = None
    for task_count, seed, slot in SAMPLES:
        sample = sample_lookup.get((seed, slot))
        if sample is None or int(sample["task_count"]) != task_count:
            raise RuntimeError(f"Efficiency JSON lacks fixed sample {seed}/{slot}/{task_count}")
        episode_dir = POOL_ROOT / f"episode_{seed}"
        manifest_path = episode_dir / "episode_manifest.json"
        manifest_hash = sha256_file(manifest_path)
        if pool_manifest["episode_manifest_sha256"].get(str(seed)) != manifest_hash:
            raise RuntimeError(f"Pool/Episode manifest hash mismatch for seed {seed}")
        artifact_hashes = {}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected_hash in manifest.get("artifact_sha256", {}).items():
            artifact_path = episode_dir / name
            if not artifact_path.is_file() or sha256_file(artifact_path) != expected_hash:
                raise RuntimeError(f"Episode artifact hash mismatch: {artifact_path}")
            artifact_hashes[name] = expected_hash
        context = load_environment_context(
            str(CONFIG_PATH.relative_to(ROOT)), realization_dir=episode_dir
        )
        steps = aligned_train_steps(context, origin_start=slot, max_steps=1)
        if len(steps) != 1 or int(steps[0].slot) != slot or len(steps[0].actual_tasks) != task_count:
            raise RuntimeError(f"Training step contract mismatch at {seed}/{slot}")
        step = steps[0]
        service_positions = task_allocation.build_service_node_positions(
            context.cache_nodes, step.actual_vehicle_positions, context.initial_uav
        )
        if context.cache_cfg.cloud_fallback_mode != expected_mode:
            raise RuntimeError(f"Configured Cloud fallback differs: {context.cache_cfg.cloud_fallback_mode}")
        feasible_targets = task_allocation.build_feasible_targets(
            list(step.actual_tasks),
            np.asarray(context.initial_cache, dtype=np.int8),
            context.cache_nodes,
            vehicle_positions=step.actual_vehicle_positions,
            service_node_positions=service_positions,
            cache_config=context.cache_cfg,
            cloud_fallback_mode=expected_mode,
        )
        if matching_warmup_record is None:
            warm_result, warm_runtime, _, _ = run_matching(context, step, feasible_targets)
            matching_warmup_record = {
                "episode_seed": seed,
                "slot": slot,
                "task_count": task_count,
                "optimizer_runtime_s_excluded_from_steady_state": warm_runtime,
                "task_utility": float(warm_result.task_utility_sum),
            }
        repetitions = 3 if (seed, slot) in EXACT_SLOTS else 1
        timed_matching_runs = [
            run_matching(context, step, feasible_targets)
            for _ in range(repetitions)
        ]
        matching_result = timed_matching_runs[0][0]
        matching_eval = timed_matching_runs[0][2]
        matching_resources = timed_matching_runs[0][3]
        matching_runtime_repetitions = [float(item[1]) for item in timed_matching_runs]
        matching_runtime = statistics.median(matching_runtime_repetitions)
        for repeated_result, _, _, _ in timed_matching_runs[1:]:
            if repeated_result.matching != matching_result.matching or not close(
                repeated_result.task_utility_sum, matching_result.task_utility_sum
            ):
                raise RuntimeError(f"Matching result changed across timing repeats at {seed}/{slot}")
        if (
            matching_warmup_record["episode_seed"] == seed
            and matching_warmup_record["slot"] == slot
            and not close(warm_result.task_utility_sum, matching_result.task_utility_sum)
        ):
            raise RuntimeError(f"Warm-up Matching utility differs from timed result at {seed}/{slot}")
        cloud_item = classify_cloud(context, step, seed, feasible_targets, matching_result, matching_eval)
        solo_analysis = analyze_solo_cloud_vs_edge(
            context, step, cloud_item, service_positions
        )
        transfer_analysis = analyze_fixed_single_transfers(
            context,
            step,
            cloud_item,
            matching_result.matching,
            matching_eval,
            service_positions,
        )
        edge_gate_analysis = run_edge_gate_comparison(
            context, step, service_positions
        )
        cloud_item.update(
            {
                "task_count_requested": task_count,
                "matching_task_utility": float(matching_result.task_utility_sum),
                "matching_runtime_s": matching_runtime,
                "matching_runtime_repetitions_s": matching_runtime_repetitions,
                "matching_recheck_resource_summary": matching_resources,
                "matching_kkt_solver_calls": int(matching_result.kkt_solver_calls),
                "matching_kkt_cache_requests": int(matching_result.kkt_cache_requests),
                "matching_kkt_cache_hits": int(matching_result.kkt_cache_hits),
                "matching_kkt_cache_misses": int(matching_result.kkt_cache_misses),
                "matching": matching_result.matching,
                "matching_per_task_utility": matching_eval.per_task_utility,
                "matching_per_task_delay_s": matching_eval.per_task_delay_s,
                "matching_per_task_vehicle_upload_energy_j": matching_eval.per_task_vehicle_energy_j,
                "feasible_targets": feasible_targets,
                "episode_manifest_sha256": manifest_hash,
                "episode_artifact_sha256": artifact_hashes,
                "cache_node_ids": [str(v) for v in context.cache_nodes["node_ids"]],
                "cache_node_roles": [str(v) for v in context.cache_nodes["node_roles"]],
                "cache_state_sha256": hashlib.sha256(np.asarray(context.initial_cache, dtype=np.int8).tobytes()).hexdigest(),
                "uav_state": {key: np.asarray(value, dtype=np.float64).tolist() for key, value in context.initial_uav.items()},
            }
        )
        historical = sample["timed_repetitions"][0]
        cloud_item["historical_efficiency_json_match_equal"] = matching_result.matching == historical["matching"]
        cloud_item["historical_efficiency_json_utility_equal"] = close(
            matching_result.task_utility_sum, float(historical["task_utility_sum"])
        )
        samples[(seed, slot)] = {
            "task_count": task_count,
            "sample": sample,
            "context": context,
            "step": step,
            "service_positions": service_positions,
            "feasible_targets": feasible_targets,
            "matching_result": matching_result,
            "matching_runtime": matching_runtime,
            "matching_runtime_repetitions": matching_runtime_repetitions,
            "matching_eval": matching_eval,
            "cloud": cloud_item,
            "solo_analysis": solo_analysis,
            "transfer_analysis": transfer_analysis,
            "edge_gate_analysis": edge_gate_analysis,
        }

    contracts = {
        "cross_slot_task_model": str(project_config.get_config_value(cfg, "execution.cross_slot_task_model")),
        "inter_node_spectrum": str(project_config.get_config_value(cfg, "communication.resource_model.inter_node_spectrum")),
        "interference": str(project_config.get_config_value(cfg, "communication.resource_model.interference")),
        "cloud_fallback_mode": context.cache_cfg.cloud_fallback_mode,
        "cloud_resource_mode": context.cache_cfg.cloud_resource_mode,
        "cloud_cpu_capacity_cycles_per_s": float(context.cache_cfg.cloud_cpu_capacity_cycles_per_s),
        "cloud_uplink_rate_bps": float(context.cache_cfg.cloud_uplink_rate_bps),
        "cloud_delay_factor": float(context.cache_cfg.cloud_delay_factor),
        "omega_delay": float(cfg["objective"]["omega_delay"]),
        "omega_energy": float(cfg["objective"]["omega_energy"]),
        "lambda_cache": float(cfg["objective"]["lambda_cache"]),
        "lambda_movement": float(cfg["objective"]["lambda_movement"]),
        "kkt_min_cpu_fraction": float(project_config.get_config_value(cfg, "algorithm.kkt.min_cpu_fraction")),
        "kkt_min_bandwidth_fraction": float(project_config.get_config_value(cfg, "algorithm.kkt.min_bandwidth_fraction")),
        "kkt_bisection_tolerance": float(project_config.get_config_value(cfg, "algorithm.kkt.bisection_tolerance")),
        "kkt_max_bisection_iterations": int(project_config.get_config_value(cfg, "algorithm.kkt.max_bisection_iterations")),
    }
    separable = (
        contracts["cross_slot_task_model"] == "independent_per_slot"
        and contracts["inter_node_spectrum"] == "ORTHOGONAL_INDEPENDENT"
        and contracts["interference"] == "NONE"
        and contracts["cloud_resource_mode"] == task_allocation.FINITE_CLOUD_RESOURCE_MODE
        and contracts["cloud_fallback_mode"] == expected_mode
    )
    if not separable:
        raise RuntimeError(f"Current configuration does not meet verified DP separability contract: {contracts}")

    payload: dict[str, Any] = {
        "schema_version": "matching_cloud_optimality_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "branch": __import__("subprocess").run(["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
        "head": __import__("subprocess").run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
        "python_executable": sys.executable,
        "conda_environment": Path(sys.prefix).name,
        "platform": platform.platform(),
        "input_efficiency_json": str(EFFICIENCY_JSON.relative_to(ROOT)),
        "input_efficiency_json_sha256": sha256_file(EFFICIENCY_JSON),
        "current_config_sha256": current_config_sha,
        "previous_config_sha256": previous["configuration"]["sha256"],
        "config_hash_matches_previous": current_config_sha == previous["configuration"]["sha256"],
        "pool_manifest_sha256": sha256_file(POOL_MANIFEST_PATH),
        "state_reference": "FROZEN_INITIAL_CACHE_UAV_REFERENCE",
        "matching_source_sha256": sha256_file(ROOT / "src" / "decision" / "task_allocation.py"),
        "kkt_source_sha256": sha256_file(ROOT / "src" / "decision" / "task_allocation.py"),
        "slot_environment_source_sha256": sha256_file(ROOT / "src" / "decision" / "slot_environment.py"),
        "contracts": contracts,
        "separability_verified": separable,
        "sample_hashes_reverified": True,
        "validation_or_test_read": False,
        "matching_timing_protocol": {
            "warmup": matching_warmup_record,
            "exact_slot_repetitions": 3,
            "reported_statistic": "median of optimizer-only Matching runtime repetitions",
            "runtime_scope": "optimize_task_matching only; full KKT validation and repeated context loading are excluded",
        },
        "cloud_analysis": [samples[(seed, slot)]["cloud"] for _, seed, slot in SAMPLES],
        "solo_cloud_edge_analysis": [
            {
                "episode_seed": seed,
                "slot": slot,
                **samples[(seed, slot)]["solo_analysis"],
            }
            for _, seed, slot in SAMPLES
        ],
        "fixed_single_transfer_analysis": [
            {
                "episode_seed": seed,
                "slot": slot,
                **samples[(seed, slot)]["transfer_analysis"],
            }
            for _, seed, slot in SAMPLES
        ],
        "nested_load_sweep_analysis": [
            {
                "episode_seed": seed,
                "slot": slot,
                **analyze_nested_load_sweeps(
                    samples[(seed, slot)]["context"],
                    samples[(seed, slot)]["step"],
                    samples[(seed, slot)]["cloud"],
                    samples[(seed, slot)]["service_positions"],
                ),
            }
            for task_count, seed, slot in SAMPLES
            if task_count == 20
        ],
        "edge_gate_analysis": [
            {
                "episode_seed": seed,
                "slot": slot,
                "resource_aware_total_q": float(samples[(seed, slot)]["matching_result"].task_utility_sum),
                "resource_aware_cloud_count": sum(
                    target == task_allocation.CLOUD
                    for target in samples[(seed, slot)]["matching_result"].matching.values()
                ),
                "resource_aware_mean_delay_s": statistics.mean(
                    float(value)
                    for value in samples[(seed, slot)]["matching_eval"].per_task_delay_s.values()
                ),
                "resource_aware_mean_energy_j": statistics.mean(
                    float(value)
                    for value in samples[(seed, slot)]["matching_eval"].per_task_vehicle_energy_j.values()
                ),
                **samples[(seed, slot)]["edge_gate_analysis"],
            }
            for _, seed, slot in SAMPLES
        ],
        "class_counts": summarize_classes([samples[(seed, slot)]["cloud"] for _, seed, slot in SAMPLES]),
        "exact_time_limit_per_slot_s": DP_TIME_LIMIT_S,
        "exact_analysis": [],
        "ppo_training_executed": False,
        "formal_algorithm_modified": False,
        "formal_config_modified": False,
        "training_data_modified": False,
    }

    for seed, slot in EXACT_SLOTS:
        sample_data = samples[(seed, slot)]
        context = sample_data["context"]
        step = sample_data["step"]
        matching_result = sample_data["matching_result"]
        matching_eval = sample_data["matching_eval"]
        matching = matching_result.matching
        match_recheck_started = time.perf_counter()
        match_recheck = task_allocation.evaluate_matching_with_kkt(
            list(step.actual_tasks),
            matching,
            step.actual_vehicle_positions,
            sample_data["service_positions"],
            context.cfg,
            context.cache_nodes,
            context.cache_cfg,
            kkt_cache=task_allocation.KKTResourceAllocationCache(),
        )
        match_resources = validate_resources(
            match_recheck, matching, context.cache_nodes, context.cache_cfg, context.cfg
        )
        match_recheck_seconds = time.perf_counter() - match_recheck_started
        if not close(match_recheck.task_utility_sum, matching_result.task_utility_sum):
            raise RuntimeError(f"Matching full recheck differs at {seed}/{slot}")
        nodes = [str(v) for v in context.cache_nodes["node_ids"]] + [task_allocation.CLOUD]
        print(f"DP start episode={seed} slot={slot} nodes={len(nodes)} tasks={len(step.actual_tasks)}", flush=True)
        try:
            exact = solve_exact_dp(
                list(step.actual_tasks),
                nodes,
                sample_data["feasible_targets"],
                step.actual_vehicle_positions,
                sample_data["service_positions"],
                context,
            )
            exact["episode_seed"] = seed
            exact["slot"] = slot
            exact["task_count"] = len(step.actual_tasks)
            exact["matching_task_utility"] = float(matching_result.task_utility_sum)
            exact["matching_runtime_s"] = float(sample_data["matching_runtime"])
            exact["matching_runtime_repetitions_s"] = sample_data["matching_runtime_repetitions"]
            exact["matching_full_kkt_recheck_seconds"] = match_recheck_seconds
            exact["matching_resource_check"] = match_resources
            exact["matching_task_assignment"] = matching
            exact["matching_task_utility_recheck"] = float(match_recheck.task_utility_sum)
            exact["matching_full_kkt_resource_feasible"] = bool(match_recheck.resource_feasible)
            exact["historical_efficiency_json_match_equal"] = bool(sample_data["cloud"]["historical_efficiency_json_match_equal"])
            exact["historical_efficiency_json_utility_equal"] = bool(sample_data["cloud"]["historical_efficiency_json_utility_equal"])
            if exact.get("status") == "EXACT_OPTIMUM_VERIFIED":
                if exact["exact_task_utility"] + UTILITY_TOLERANCE < exact["matching_task_utility"]:
                    exact["status"] = "FAILED_OPTIMUM_CONSISTENCY"
                    exact["complete_legal_subset_search"] = False
                exact["delta_q_exact_minus_matching"] = exact["exact_task_utility"] - exact["matching_task_utility"]
                exact["gap_percent"] = (
                    100.0 * exact["delta_q_exact_minus_matching"] / abs(exact["exact_task_utility"])
                    if abs(exact["exact_task_utility"]) > 1.0e-15
                    else None
                )
                differences = []
                if exact["status"] == "EXACT_OPTIMUM_VERIFIED":
                    for task in step.actual_tasks:
                        old_target = matching[task.task_id]
                        new_target = exact["exact_matching"][task.task_id]
                        if old_target != new_target:
                            old_q = float(match_recheck.per_task_utility[task.task_id])
                            new_q = float(exact["exact_per_task_utility"][task.task_id])
                            differences.append(
                                {
                                    "task_id": task.task_id,
                                    "service_id": int(task.service_id),
                                    "legal_edge_nodes": [target for target in sample_data["feasible_targets"][task.task_id] if target != task_allocation.CLOUD],
                                    "matching_target": old_target,
                                    "exact_target": new_target,
                                    "matching_task_utility": old_q,
                                    "exact_task_utility": new_q,
                                    "task_utility_delta": new_q - old_q,
                                }
                            )
                exact["task_assignment_differences"] = differences
                exact["changed_task_count"] = len(differences)
                matching_cloud_tasks = sorted(
                    task_id for task_id, target in matching.items()
                    if target == task_allocation.CLOUD
                )
                exact_cloud_tasks = sorted(
                    task_id for task_id, target in exact["exact_matching"].items()
                    if target == task_allocation.CLOUD
                )
                exact["matching_cloud_task_ids"] = matching_cloud_tasks
                exact["exact_cloud_task_ids"] = exact_cloud_tasks
                exact["cloud_task_set_identical"] = matching_cloud_tasks == exact_cloud_tasks
            else:
                exact["delta_q_exact_minus_matching"] = None
                exact["gap_percent"] = None
                exact["task_assignment_differences"] = []
                exact["changed_task_count"] = None
                exact["matching_cloud_task_ids"] = None
                exact["exact_cloud_task_ids"] = None
                exact["cloud_task_set_identical"] = None
            payload["exact_analysis"].append(exact)
            print(
                f"DP done episode={seed} slot={slot} status={exact['status']} "
                f"runtime={exact['runtime_s']:.1f}s utility={exact.get('exact_task_utility', 'NA')}",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "episode_seed": seed,
                "slot": slot,
                "task_count": len(step.actual_tasks),
                "status": "INCOMPLETE",
                "error": f"{type(exc).__name__}: {exc}",
                "matching_task_utility": float(matching_result.task_utility_sum),
                "matching_runtime_s": float(sample_data["matching_runtime"]),
                "matching_full_kkt_recheck_seconds": match_recheck_seconds,
                "matching_task_assignment": matching,
                "matching_resource_check": match_resources,
                "runtime_s": None,
                "delta_q_exact_minus_matching": None,
                "gap_percent": None,
                "complete_legal_subset_search": False,
                "task_assignment_differences": [],
                "changed_task_count": None,
            }
            payload["exact_analysis"].append(failure)
            print(f"DP failed episode={seed} slot={slot}: {failure['error']}", flush=True)

    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(
        RESULT_JSON,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    write_text_atomic(RESULT_MD, make_report(payload))
    print(f"Wrote {RESULT_MD}", flush=True)
    print(f"Wrote {RESULT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
