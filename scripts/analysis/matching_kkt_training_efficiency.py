#!/usr/bin/env python3
"""Measure formal Matching+KKT cost on fixed real Training slots.

The deterministic sample rule selects the earliest slot in each eligible
Episode for each requested arrival count, then takes up to three distinct
Episode seeds in ascending order. Timed repetitions restart from each
Episode's formal initial Cache/UAV state and create a fresh per-call KKT cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from common import project_config  # noqa: E402
from decision import slot_environment, task_allocation  # noqa: E402
from ppo_simple.interface import aligned_train_steps, load_environment_context  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "proposed" / "system_config.toml"
POOL_ROOT = PROJECT_ROOT / "environment_data" / "proposed_joint_training"
POOL_MANIFEST_PATH = POOL_ROOT / "formal_pool_manifest.json"
REPORT_MD = PROJECT_ROOT / "reports" / "matching_kkt_training_efficiency.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "matching_kkt_training_efficiency.json"
TARGET_TASK_COUNTS = (10, 14, 20)
SAMPLES_PER_TASK_COUNT = 3
MAX_REPETITIONS = 3
LONG_RUN_CUTOFF_S = 2.0
TIMING_KEYS = (
    "matching_total_runtime",
    "matching_runtime",
    "matching_other_runtime",
    "matching_candidate_generation_runtime",
    "matching_upper_bound_runtime",
    "matching_exact_eval_orchestration_runtime",
    "node_cache_lookup_runtime",
    "node_eval_miss_runtime",
    "kkt_runtime",
    "kkt_solver_core_runtime",
    "kkt_edge_cpu_solver_core_runtime",
    "kkt_edge_bandwidth_solver_core_runtime",
    "kkt_cloud_cpu_solver_core_runtime",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def scan_training_pool(
    pool_manifest: dict[str, Any], expected_tv_count: int
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    """Count arrived task rows in manifest-attested complete Training Episodes."""

    episode_hashes = pool_manifest["episode_manifest_sha256"]
    expected_slots = int(pool_manifest["slots_per_episode"])
    candidates: dict[int, list[dict[str, Any]]] = {
        task_count: [] for task_count in TARGET_TASK_COUNTS
    }
    skipped: list[dict[str, Any]] = []
    scanned_episodes = 0

    for episode_dir in sorted(POOL_ROOT.glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        manifest_path = episode_dir / "episode_manifest.json"
        tasks_path = episode_dir / "tasks.csv"
        if not manifest_path.is_file() or not tasks_path.is_file():
            skipped.append({"episode": episode_dir.name, "reason": "manifest or tasks.csv missing"})
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            seed = int(manifest["episode_seed"])
            expected_manifest_hash = episode_hashes.get(str(seed))
            actual_manifest_hash = sha256_file(manifest_path)
            if expected_manifest_hash is None or actual_manifest_hash != expected_manifest_hash:
                skipped.append({"episode": episode_dir.name, "reason": "episode manifest not attested by pool manifest"})
                continue
            if (
                str(manifest.get("split", "")).lower() != "train"
                or int(manifest.get("episode_slots", -1)) != expected_slots
                or episode_dir.name != f"episode_{seed}"
            ):
                skipped.append({"episode": episode_dir.name, "reason": "episode split, seed, or slot contract mismatch"})
                continue

            row_counts: dict[int, int] = {}
            arrived_counts: dict[int, int] = {}
            with tasks_path.open("r", encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    slot = int(row["slot"])
                    if not 0 <= slot < expected_slots:
                        raise ValueError(f"task slot outside 0..{expected_slots - 1}: {slot}")
                    row_counts[slot] = row_counts.get(slot, 0) + 1
                    arrival = int(row["arrival"])
                    if arrival not in (0, 1):
                        raise ValueError(f"invalid arrival={arrival} in {tasks_path}")
                    arrived_counts[slot] = arrived_counts.get(slot, 0) + arrival
            if set(row_counts) != set(range(expected_slots)):
                skipped.append({"episode": episode_dir.name, "reason": "tasks.csv does not cover every slot"})
                continue
            scanned_episodes += 1
            if any(row_counts[slot] != expected_tv_count for slot in range(expected_slots)):
                skipped.append({"episode": episode_dir.name, "reason": "per-slot task row count differs from task vehicle count"})
                continue
            for slot in range(expected_slots):
                task_count = arrived_counts[slot]
                if task_count in candidates:
                    candidates[task_count].append(
                        {
                            "episode_seed": seed,
                            "slot": slot,
                            "time_s": None,
                            "task_count": task_count,
                            "task_rows": row_counts[slot],
                            "episode_dir": str(episode_dir),
                            "episode_manifest_sha256": actual_manifest_hash,
                        }
                    )
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            skipped.append({"episode": episode_dir.name, "reason": f"scan error: {exc}"})

    selected: dict[int, list[dict[str, Any]]] = {}
    available: dict[str, Any] = {}
    for task_count in TARGET_TASK_COUNTS:
        ordered = sorted(candidates[task_count], key=lambda item: (item["episode_seed"], item["slot"]))
        unique_by_episode: dict[int, dict[str, Any]] = {}
        for item in ordered:
            unique_by_episode.setdefault(int(item["episode_seed"]), item)
        unique_samples = list(unique_by_episode.values())
        available[str(task_count)] = {
            "matching_slots": len(ordered),
            "distinct_episodes": len(unique_samples),
        }
        selected[task_count] = unique_samples[:SAMPLES_PER_TASK_COUNT]
    scan_summary = {
        "manifest_attested_complete_episodes_scanned": scanned_episodes,
        "available_by_task_count": available,
        "selection_rule": (
            "对每个目标到达任务数，按 (episode_seed, slot) 升序排列；每个 Episode 仅取最早匹配 Slot，"
            "再取最小的三个不同 Episode seed。选样不使用计时、效用或 Matching 结果。"
        ),
        "skipped_episode_count": len(skipped),
        "skipped_episodes": skipped,
    }
    return selected, scan_summary


def validate_data_contract(cfg: dict[str, Any], pool_manifest: dict[str, Any]) -> dict[str, Any]:
    contract = pool_manifest["task_data_contract"]
    actual = {
        "service_count": int(project_config.get_config_value(cfg, "service.count")),
        "input_size_min_kb": float(project_config.get_config_value(cfg, "service.generation.input_size_min_kb")),
        "input_size_max_kb": float(project_config.get_config_value(cfg, "service.generation.input_size_max_kb")),
        "service_size_min_gb": float(project_config.get_config_value(cfg, "service.generation.service_size_min_gb")),
        "service_size_max_gb": float(project_config.get_config_value(cfg, "service.generation.service_size_max_gb")),
        "cycles_per_bit_min": float(project_config.get_config_value(cfg, "service.generation.cycles_per_bit_min")),
        "cycles_per_bit_max": float(project_config.get_config_value(cfg, "service.generation.cycles_per_bit_max")),
        "input_relative_jitter": float(project_config.get_config_value(cfg, "task.input_relative_jitter")),
        "load_relative_jitter": float(project_config.get_config_value(cfg, "task.load_relative_jitter")),
        "arrival_probability": float(project_config.get_config_value(cfg, "task.arrival_probability")),
        "popularity_zeta": float(project_config.get_config_value(cfg, "task.popularity_zeta")),
        "episode_slots": int(project_config.get_config_value(cfg, "algorithm.ppo.data_protocol.episode_slots")),
        "slot_duration_s": float(project_config.get_config_value(cfg, "simulation.dt_s")),
    }
    mismatches = {}
    for key, value in actual.items():
        expected = contract.get(key)
        if isinstance(value, float):
            matches = expected is not None and math.isclose(value, float(expected), rel_tol=0.0, abs_tol=1e-12)
        else:
            matches = value == expected
        if not matches:
            mismatches[key] = {"config": value, "pool_manifest": expected}
    catalog_path = project_config.resolve_path(
        Path(cfg["_meta"]["project_root"]),
        project_config.get_config_value(cfg, "paths.service_catalog"),
    )
    catalog_hash = sha256_file(catalog_path)
    catalog_matches = catalog_hash == str(pool_manifest["service_catalog_sha256"])
    return {
        "config_task_contract": actual,
        "pool_task_contract_sha256": pool_manifest.get("task_data_contract_sha256"),
        "task_contract_mismatches": mismatches,
        "service_catalog_path": str(catalog_path),
        "service_catalog_sha256": catalog_hash,
        "service_catalog_matches_pool": catalog_matches,
        "valid": not mismatches and catalog_matches,
    }


def validate_episode_artifacts(sample: dict[str, Any], pool_manifest: dict[str, Any]) -> dict[str, str]:
    episode_dir = Path(sample["episode_dir"])
    manifest_path = episode_dir / "episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = list(manifest.get("required_artifacts", pool_manifest["required_artifacts"]))
    artifact_hashes = manifest.get("artifact_sha256", {})
    verified: dict[str, str] = {}
    for relative_name in required:
        path = episode_dir / relative_name
        if not path.is_file():
            raise FileNotFoundError(f"Selected Episode artifact missing: {path}")
        expected = artifact_hashes.get(relative_name)
        if not expected:
            raise ValueError(f"Selected Episode manifest has no artifact hash: {relative_name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Selected Episode artifact hash mismatch: {relative_name}")
        verified[relative_name] = actual
    return verified


def run_slot(
    context: Any,
    step: Any,
    *,
    collect_timing: bool,
) -> dict[str, Any]:
    """Run the formal frozen-initial Cache/UAV slot path once."""

    cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
    uav = {key: np.asarray(value, dtype=np.float64).copy() for key, value in context.initial_uav.items()}
    performance_profile: dict[str, Any] | None = {} if collect_timing else None
    timing_profile: dict[str, float] | None = {} if collect_timing else None
    kkt_cache = task_allocation.KKTResourceAllocationCache()
    matching_results: list[task_allocation.TransferExchangeMatchingResult] = []
    formal_optimizer = task_allocation.optimize_task_matching

    def capture_matching_result(
        *args: Any, **kwargs: Any
    ) -> task_allocation.TransferExchangeMatchingResult:
        matching_result = formal_optimizer(*args, **kwargs)
        matching_results.append(matching_result)
        return matching_result

    # Capture the formal solver's full diagnostic result from the same call made
    # inside execute_configured_slot; do not issue a second Matching evaluation.
    task_allocation.optimize_task_matching = capture_matching_result
    try:
        started = time.perf_counter()
        result = slot_environment.execute_configured_slot(
            cache,
            uav,
            cache.copy(),
            {key: value.copy() for key, value in uav.items()},
            step.scenario_slice,
            list(step.actual_tasks),
            {key: value.copy() for key, value in step.actual_vehicle_positions.items()},
            context.service_catalog,
            context.cfg,
            context.cache_nodes,
            context.cache_cfg,
            context.flight_boundary,
            slot_id=int(step.slot),
            time_s=float(step.time_s),
            performance_profile=performance_profile,
            matching_kkt_cache=kkt_cache,
            exact_q_timing_profile=timing_profile,
        )
        wall_s = time.perf_counter() - started
    finally:
        task_allocation.optimize_task_matching = formal_optimizer
    if not result.step_feasible or result.system_utility is None:
        raise RuntimeError(f"formal slot execution infeasible: {result.infeasible_reason}")
    if len(matching_results) != 1:
        raise RuntimeError(f"expected one formal Matching call, captured {len(matching_results)}")
    matching_result = matching_results[0]
    summary = result.matching_summary
    if not summary.get("resource_feasible", False):
        raise RuntimeError("formal Matching returned resource-infeasible result")
    if int(matching_result.kkt_solver_calls) != int(kkt_cache.solver_calls):
        raise RuntimeError("KKT solver-call summary differs from the fresh per-call cache")
    record: dict[str, Any] = {
        "execute_configured_slot_wall_s": wall_s,
        "task_utility_sum": float(result.actual_task_utility),
        "system_utility": float(result.system_utility),
        "matching": dict(matching_result.matching),
        "resource_feasible": bool(summary["resource_feasible"]),
        "candidate_definitions_total": int(matching_result.candidate_definitions_total),
        "candidate_evaluations_total": int(matching_result.candidate_evaluations_total),
        "upper_bound_screen_calls": int(matching_result.upper_bound_screen_calls),
        "upper_bound_pruned_candidates": int(matching_result.upper_bound_pruned_candidates),
        "transfer_candidates_evaluated": int(matching_result.transfer_candidates_evaluated),
        "exchange_candidates_evaluated": int(matching_result.exchange_candidates_evaluated),
        "full_evaluator_calls": int(matching_result.full_evaluator_calls),
        "updates": int(matching_result.updates),
        "transfer_updates": int(matching_result.transfer_updates),
        "exchange_updates": int(matching_result.exchange_updates),
        "stop_reason": str(matching_result.stop_reason),
        "transfer_exchange_stable": bool(matching_result.transfer_exchange_stable),
        "stable_check_passed": bool(matching_result.stable_check_passed),
        "kkt_solver_calls": int(matching_result.kkt_solver_calls),
        "kkt_solve_count": int(matching_result.kkt_solve_count),
        "kkt_cache_requests": int(matching_result.kkt_cache_requests),
        "kkt_cache_hits": int(matching_result.kkt_cache_hits),
        "kkt_cache_misses": int(matching_result.kkt_cache_misses),
        "node_eval_cache_requests": int(matching_result.node_eval_cache_requests),
        "node_eval_cache_hits": int(matching_result.node_eval_cache_hits),
        "node_eval_cache_misses": int(matching_result.node_eval_cache_misses),
        "node_task_count": dict(summary.get("node_task_count", {})),
        "cloud_cpu_used_cycles_per_s": float(summary["resource"]["cloud_cpu_usage_cycles_per_s"]),
        "cloud_cpu_capacity_cycles_per_s": summary["resource"]["cloud_cpu_capacity_cycles_per_s"],
        "timing_s": {},
    }
    if collect_timing:
        if timing_profile is None or performance_profile is None:
            raise RuntimeError("timing collection unexpectedly disabled")
        record["timing_s"] = {
            key: (float(timing_profile[key]) if key in timing_profile else None)
            for key in TIMING_KEYS
        }
        # The formal Training-slot tasks are all NEW, so allocate_resources_kkt
        # routes them through _allocate_phase_resources_kkt. That path records
        # aggregate solver-core time (and Cloud CPU time), but does not record
        # the standalone Edge CPU/bandwidth timers used by the other execution
        # path. The slot wrapper initializes some general timing keys to zero;
        # those absent component measurements must remain unavailable, not 0.
        record["timing_s"]["kkt_edge_cpu_solver_core_runtime"] = None
        record["timing_s"]["kkt_edge_bandwidth_solver_core_runtime"] = None
        record["matching_kkt_seconds"] = float(performance_profile.get("matching_kkt_seconds", 0.0))
        record["configuration_seconds"] = float(performance_profile.get("configuration_seconds", 0.0))
    return record


def median_or_none(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def aggregate_group(task_count: int, samples: list[dict[str, Any]]) -> dict[str, Any]:
    records = [record for sample in samples for record in sample["timed_repetitions"]]
    sample_medians: dict[str, list[float]] = {key: [] for key in TIMING_KEYS}
    sample_medians.update({"matching_kkt_seconds": [], "execute_configured_slot_wall_s": []})
    for sample in samples:
        reps = sample["timed_repetitions"]
        for key in TIMING_KEYS:
            observed = [
                float(rep["timing_s"][key])
                for rep in reps
                if rep["timing_s"][key] is not None
            ]
            if observed:
                sample_medians[key].append(float(statistics.median(observed)))
        sample_medians["matching_kkt_seconds"].append(float(statistics.median(rep["matching_kkt_seconds"] for rep in reps)))
        sample_medians["execute_configured_slot_wall_s"].append(float(statistics.median(rep["execute_configured_slot_wall_s"] for rep in reps)))

    def median_stat(key: str) -> float | None:
        return median_or_none(sample_medians[key])

    count_keys = (
        "candidate_definitions_total",
        "candidate_evaluations_total",
        "upper_bound_screen_calls",
        "upper_bound_pruned_candidates",
        "transfer_candidates_evaluated",
        "exchange_candidates_evaluated",
        "full_evaluator_calls",
        "updates",
        "kkt_solver_calls",
        "kkt_cache_requests",
        "kkt_cache_hits",
        "kkt_cache_misses",
    )
    medians = {
        key: median_or_none([float(sample["timed_repetitions"][0][key]) for sample in samples])
        for key in count_keys
    }
    matching_ms = None if median_stat("matching_total_runtime") is None else median_stat("matching_total_runtime") * 1000.0
    kkt_ms = None if median_stat("kkt_runtime") is None else median_stat("kkt_runtime") * 1000.0
    ratio_values = [
        100.0 * float(rep["timing_s"]["kkt_runtime"]) / float(rep["timing_s"]["matching_total_runtime"])
        for rep in records
        if float(rep["timing_s"]["matching_total_runtime"]) > 0.0
    ]
    prune_ratio_values = [
        100.0 * float(rep["upper_bound_pruned_candidates"]) / float(rep["upper_bound_screen_calls"])
        for rep in records
        if rep["upper_bound_screen_calls"] > 0
    ]
    utility_values = [float(sample["timed_repetitions"][0]["task_utility_sum"]) for sample in samples]
    return {
        "task_count": task_count,
        "valid_slot_count": len(samples),
        "matching_time_median_ms": matching_ms,
        "matching_time_range_ms": None if not samples else [
            min(float(statistics.median(rep["timing_s"]["matching_total_runtime"] for rep in sample["timed_repetitions"])) * 1000.0 for sample in samples),
            max(float(statistics.median(rep["timing_s"]["matching_total_runtime"] for rep in sample["timed_repetitions"])) * 1000.0 for sample in samples),
        ],
        "kkt_total_time_median_ms": kkt_ms,
        "kkt_time_ratio_median_percent": median_or_none(ratio_values),
        "kkt_solver_core_time_median_ms": median_stat("kkt_solver_core_runtime") * 1000.0 if median_stat("kkt_solver_core_runtime") is not None else None,
        "cpu_kkt_core_time_median_ms": median_stat("kkt_edge_cpu_solver_core_runtime") * 1000.0 if median_stat("kkt_edge_cpu_solver_core_runtime") is not None else None,
        "bandwidth_kkt_core_time_median_ms": median_stat("kkt_edge_bandwidth_solver_core_runtime") * 1000.0 if median_stat("kkt_edge_bandwidth_solver_core_runtime") is not None else None,
        "cloud_kkt_core_time_median_ms": median_stat("kkt_cloud_cpu_solver_core_runtime") * 1000.0 if median_stat("kkt_cloud_cpu_solver_core_runtime") is not None else None,
        "matching_non_kkt_residual_median_ms": median_stat("matching_runtime") * 1000.0 if median_stat("matching_runtime") is not None else None,
        "node_evaluation_excluding_kkt_median_ms": median_stat("node_eval_miss_runtime") * 1000.0 if median_stat("node_eval_miss_runtime") is not None else None,
        "candidate_generation_median_ms": median_stat("matching_candidate_generation_runtime") * 1000.0 if median_stat("matching_candidate_generation_runtime") is not None else None,
        "upper_bound_screening_median_ms": median_stat("matching_upper_bound_runtime") * 1000.0 if median_stat("matching_upper_bound_runtime") is not None else None,
        "exact_evaluator_orchestration_median_ms": median_stat("matching_exact_eval_orchestration_runtime") * 1000.0 if median_stat("matching_exact_eval_orchestration_runtime") is not None else None,
        "matching_wrapper_median_ms": median_stat("matching_kkt_seconds") * 1000.0,
        "timed_slot_wall_median_ms": median_stat("execute_configured_slot_wall_s") * 1000.0,
        "median_pruned_over_upper_bound_screen_calls_percent": median_or_none(prune_ratio_values),
        "task_utility_sum_median": median_or_none(utility_values),
        "median_counts_per_matching": medians,
        "kkt_cache_hit_ratio_percent": (
            None if not medians["kkt_cache_requests"] else
            100.0 * float(medians["kkt_cache_hits"]) / float(medians["kkt_cache_requests"])
        ),
        "kkt_solver_core_time_per_solver_call_median_ms": (
            None
            if not medians["kkt_solver_calls"] or median_stat("kkt_solver_core_runtime") is None
            else 1000.0 * median_stat("kkt_solver_core_runtime") / float(medians["kkt_solver_calls"])
        ),
    }


def format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NOT_AVAILABLE"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def build_report(payload: dict[str, Any]) -> str:
    cfg = payload["configuration"]
    objective = cfg["objective"]
    matching = cfg["matching_contract"]
    kkt = cfg["kkt_contract"]
    groups = payload["group_summary"]
    lines = [
        "# Training Slot Matching + KKT 求解效率诊断",
        "",
        f"生成时间：{payload['created_at_utc']}",
        "",
        "## 运行与数据来源",
        "",
        f"- 仓库：`{payload['repository']['branch']}`，HEAD `{payload['repository']['head']}`。",
        f"- 工作区基线状态：本任务开始时已有大量未跟踪数据与报告；诊断只新增本脚本及本报告的 Markdown/JSON，不改写已有文件。",
        f"- Python：`{payload['environment']['python_executable']}`；`sys.prefix`=`{payload['environment']['python_prefix']}`。Conda 环境核验：`{payload['environment']['conda_environment']}`。",
        f"- 硬件：`{payload['environment']['platform']}`；处理器 `{payload['environment']['processor'] or 'OS 未提供处理器型号'}`；逻辑 CPU `{payload['environment']['logical_cpu_count']}`；内存 `{payload['environment'].get('total_memory_gib') if payload['environment'].get('total_memory_gib') is not None else 'NOT_AVAILABLE'} GiB`。",
        f"- 配置：`configs/proposed/system_config.toml`，SHA-256 `{cfg['sha256']}`。Training Pool Manifest SHA-256 `{payload['training_pool']['manifest_sha256']}`。",
        f"- Pool 配置哈希与当前配置哈希相同：`{payload['training_pool']['config_hash_matches_current']}`；任务生成合同逐字段一致且 Service Catalog 哈希匹配：`{payload['data_contract']['valid']}`。",
        "- Pool 的完整配置哈希与当前配置哈希不一致；任务生成合同字段和服务目录哈希通过核对，因此报告只确认任务数据合同一致，不声称 Pool 完整参数快照与当前配置完全相同。执行测量使用当前本地配置。",
        "- 当前工作区还存在 `configs/proposed/system_config.toml` 的未提交成本系数差异（`lambda_cache=0.005`、`lambda_movement=5.0e-6`）；本诊断未编辑该文件并保留该差异。冻结 Cache/UAV 不触发配置成本，因此不改变本次 Matching/KKT 的任务评价。",
        f"- 仅扫描 `{payload['training_pool']['path']}`。未读取 Validation/Test。Pool 声明 `{payload['training_pool']['episode_count']}` 个 Episode、每 Episode `{payload['training_pool']['slots_per_episode']}` Slot；其中经 Pool Manifest 与 Episode Manifest 绑定且任务表覆盖完整的 Episode `{payload['sample_scan']['manifest_attested_complete_episodes_scanned']}` 个。",
        "- Slot 任务数定义为 `tasks.csv` 对该 Slot 的实际行中 `arrival=1` 的数量；每个样本还与正式 `aligned_train_steps()` 的 `len(actual_tasks)` 核对。任务输入字段单位按正式 `Task.input_bits` 处理，汇总以十进制 MB 表示；计算需求以 Gcycle 表示。",
        f"- 选样规则：{payload['sample_scan']['selection_rule']}",
        f"- Cache/UAV 状态：`FROZEN_INITIAL_CACHE_UAV_REFERENCE`。每次测量使用该 Episode 由正式环境入口加载的初始 Cache/UAV，Slot 内冻结；不是 PPO 在所选 Slot 的策略动作或状态快照。",
        "- 预测文件按正式接口加载并做 Slot 对齐，但因为 Cache/UAV triggers 均为零，执行器跳过配置预测求值；任务匹配只使用该 Slot 实际任务与实际车辆位置。",
        "",
        "## 求解合同与计时口径",
        "",
        f"- Matching：`{matching['method']}`，`max_updates={matching['max_updates']}`，容差 `{matching['objective_tolerance']}`，Cloud fallback `{matching['cloud_fallback_mode']}`；评价模式为源码默认 `incremental`。正式 Runner 未传持久 `NodeEvaluationCache`，诊断同样不传；每次独立调用使用全新的 `KKTResourceAllocationCache`。",
        f"- KKT：min CPU fraction `{kkt['min_cpu_fraction']}`，min bandwidth fraction `{kkt['min_bandwidth_fraction']}`，bisection tolerance `{kkt['bisection_tolerance']}`，max iterations `{kkt['max_bisection_iterations']}`。噪声模型 `{cfg['noise_model']}`。",
        f"- 效用权重：`omega_delay={objective['omega_delay']}`，`omega_energy={objective['omega_energy']}`；T_ref `{objective['T_ref_s']}` s，E_ref `{objective['E_ref_j']}` J。Edge CPU：SV `{cfg['sv_cpu_cycles_per_s']/1e9:g}`、RSU `{cfg['rsu_cpu_cycles_per_s']/1e9:g}`、UAV `{cfg['uav_cpu_cycles_per_s']/1e9:g}` GHz；Cloud `{cfg['cloud_cpu_capacity_cycles_per_s']/1e9:g}` GHz，模式 `{cfg['cloud_resource_mode']}`。Cloud 等效上传 `{cfg['cloud_uplink_rate_bps']/1e6:g}` Mbps，时延系数 `{cfg['cloud_delay_factor']}`。",
        "- 先用所选样本中任务最多的真实 Slot 执行一次不计时预热，排除 Numba 首次编译；预热运行不纳入统计。计时最多重复 3 次；若某 Slot 首次计时的 Matching 完整调用超过 2 s，则该 Slot 不再重复。各重复调用从同一初始状态开始，KKT cache 每次新建。",
        "- `matching_total_runtime` 是源码在 `optimize_task_matching()` 调用边界内测得的完整 Matching 时间，作为主指标。`matching_runtime` 是 Matching 总时间扣除节点缓存查询、节点评价 miss 和 `kkt_runtime` 后的剩余部分。`matching_kkt_seconds` 是 Slot 包装层从构造 Service Node 位置开始到 Matching 返回的时间，包含位置构造，因此不与 `matching_total_runtime` 相加。外部 Slot wall 还包含入口校验和 Reward 组合。",
        "- `kkt_runtime` 是 KKT 资源分配上下文总时间；`kkt_solver_core_runtime` 是其子项；CPU、带宽和 Cloud solver core 也是子项，不能彼此相加后再加到总时间。候选子计时（生成、上界筛选、精确评价 orchestration）嵌套于 Matching 的非 KKT 路径，仅用于定位，不与 `matching_runtime` 重复求和。",
        "- 事后源码核对确认正式 `NEW` 任务路径未单独记录 Edge CPU 与带宽 solver core 时间；报告将这两个字段改列为 `NOT_AVAILABLE`，未重跑 Matching，既有完整调用/KKT 总耗时和计数保持原测量值。",
        "- 候选剪枝比例使用 `upper_bound_pruned_candidates / upper_bound_screen_calls`；这是被上界检查的候选中实际剪枝比例，不等同于运行时间节省比例。候选定义数、候选评价数和接受更新数分别报告。",
        "",
        "## 样本汇总",
        "",
        "| 指标 | 10 任务 | 14 任务 | 20 任务 |",
        "|---|---:|---:|---:|",
        "| 可用真实 Slot | {0} | {1} | {2} |".format(*(groups[str(n)]["valid_slot_count"] for n in TARGET_TASK_COUNTS)),
        "| Matching 完整调用中位数 (ms) | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["matching_time_median_ms"]) for n in TARGET_TASK_COUNTS)),
        "| KKT 总耗时中位数 (ms) | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["kkt_total_time_median_ms"]) for n in TARGET_TASK_COUNTS)),
        "| KKT / Matching 中位数 (%) | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["kkt_time_ratio_median_percent"]) for n in TARGET_TASK_COUNTS)),
        "| CPU KKT 核心中位数 (ms) | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["cpu_kkt_core_time_median_ms"]) for n in TARGET_TASK_COUNTS)),
        "| 带宽 KKT 核心中位数 (ms) | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["bandwidth_kkt_core_time_median_ms"]) for n in TARGET_TASK_COUNTS)),
        "| KKT Solver core 总中位数 (ms) | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["kkt_solver_core_time_median_ms"]) for n in TARGET_TASK_COUNTS)),
        "| Cloud KKT 核心中位数 (ms) | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["cloud_kkt_core_time_median_ms"]) for n in TARGET_TASK_COUNTS)),
        "| 候选定义中位数 | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["median_counts_per_matching"]["candidate_definitions_total"], 0) for n in TARGET_TASK_COUNTS)),
        "| 候选实际评价中位数 | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["median_counts_per_matching"]["candidate_evaluations_total"], 0) for n in TARGET_TASK_COUNTS)),
        "| 上界剪枝中位数 | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["median_counts_per_matching"]["upper_bound_pruned_candidates"], 0) for n in TARGET_TASK_COUNTS)),
        "| 上界剪枝率中位数 (%) | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["median_pruned_over_upper_bound_screen_calls_percent"]) for n in TARGET_TASK_COUNTS)),
        "| Transfer / Exchange 候选评价中位数 | {0} | {1} | {2} |".format(*(
            f"{format_number(groups[str(n)]['median_counts_per_matching']['transfer_candidates_evaluated'], 0)} / {format_number(groups[str(n)]['median_counts_per_matching']['exchange_candidates_evaluated'], 0)}"
            for n in TARGET_TASK_COUNTS
        )),
        "| Full evaluator 调用中位数 | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["median_counts_per_matching"]["full_evaluator_calls"], 0) for n in TARGET_TASK_COUNTS)),
        "| KKT Solver 调用中位数 | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["median_counts_per_matching"]["kkt_solver_calls"], 0) for n in TARGET_TASK_COUNTS)),
        "| KKT cache 命中率中位数 (%) | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["kkt_cache_hit_ratio_percent"]) for n in TARGET_TASK_COUNTS)),
        "| 每 KKT Solver core 中位数 (ms/call) | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["kkt_solver_core_time_per_solver_call_median_ms"]) for n in TARGET_TASK_COUNTS)),
        "| Matching 更新中位数 | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["median_counts_per_matching"]["updates"], 0) for n in TARGET_TASK_COUNTS)),
        "| 任务效用总和中位数 | {0} | {1} | {2} |".format(*(format_number(groups[str(n)]["task_utility_sum_median"], 6) for n in TARGET_TASK_COUNTS)),
        "",
        "每个任务规模的原始 Slot 与重复计时见下表。候选/KKT 计数应在同一 Slot 的重复间一致；若不一致，JSON 中保留每次结果。",
        "",
        "| 任务数 | Episode Seed | Slot | 输入总量 (MB) | CPU 总量 (Gcycle) | 可行 Edge 任务-节点边 | 节点任务分布 (SV/RSU/UAV/Cloud) | Matching ms (中位数) | KKT ms (中位数) | 候选定义/评价/剪枝 | KKT 调用/缓存命中 | 稳定停止 |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for task_count in TARGET_TASK_COUNTS:
        for sample in payload["samples_by_task_count"][str(task_count)]:
            first = sample["timed_repetitions"][0]
            time_s = statistics.median(rep["timing_s"]["matching_total_runtime"] for rep in sample["timed_repetitions"]) * 1000.0
            kkt_s = statistics.median(rep["timing_s"]["kkt_runtime"] for rep in sample["timed_repetitions"]) * 1000.0
            node_counts = first["node_task_count"]
            distribution = f"{node_counts.get('SV', 0)}/{node_counts.get('RSU', 0)}/{node_counts.get('UAV', 0)}/{node_counts.get('Cloud', 0)}"
            lines.append(
                "| {task_count} | {seed} | {slot} | {input_mb:.3f} | {cycles_g:.3f} | {edges} | {distribution} | {matching_ms:.3f} | {kkt_ms:.3f} | {defs}/{evals}/{pruned} | {calls}/{hits} | {reason} / stable={stable} |".format(
                    task_count=task_count,
                    seed=sample["episode_seed"],
                    slot=sample["slot"],
                    input_mb=sample["input_bits_sum"] / 8.0 / 1e6,
                    cycles_g=sample["cpu_cycles_sum"] / 1e9,
                    edges=sample["feasible_edge_task_node_edges"],
                    distribution=distribution,
                    matching_ms=time_s,
                    kkt_ms=kkt_s,
                    defs=first["candidate_definitions_total"],
                    evals=first["candidate_evaluations_total"],
                    pruned=first["upper_bound_pruned_candidates"],
                    calls=first["kkt_solver_calls"],
                    hits=first["kkt_cache_hits"],
                    reason=first["stop_reason"],
                    stable=first["transfer_exchange_stable"],
                )
            )
    lines.extend(
        [
            "",
            "## 诊断结论",
            "",
            f"- 本次实际样本为 `{payload['sample_count']}` 个 Slot，任务规模组分别有 `{groups['10']['valid_slot_count']}`、`{groups['14']['valid_slot_count']}`、`{groups['20']['valid_slot_count']}` 个。样本先按真实到达任务数筛选，规模组中位数只描述这些冻结初始配置下的样本，不足以证明复杂度阶数。",
            f"- 主时间分解的跨样本中位数：Matching 剩余非 KKT {format_number(payload['timing_diagnosis']['matching_non_kkt_residual_median_ms'])} ms，节点效用评价（不含 KKT）{format_number(payload['timing_diagnosis']['node_evaluation_excluding_kkt_median_ms'])} ms，KKT `{format_number(payload['timing_diagnosis']['kkt_total_time_median_ms'])}` ms，其中 solver core `{format_number(payload['timing_diagnosis']['kkt_solver_core_median_ms'])}` ms。主导项判定：**{payload['timing_diagnosis']['dominant_cost_component']}**。",
            f"- Matching 内部候选生成中位数 `{format_number(payload['timing_diagnosis']['candidate_generation_median_ms'])}` ms、上界筛选 `{format_number(payload['timing_diagnosis']['upper_bound_screening_median_ms'])}` ms、完整评价编排 `{format_number(payload['timing_diagnosis']['exact_evaluator_orchestration_median_ms'])}` ms；这些时间是非 KKT 路径的细分，不与该路径总时间相加。",
            f"- 本轮计时覆盖 `{payload['method']['timed_matching_call_count']}` 次 Matching 调用，KKT Solver 调用合计 `{payload['method']['timed_kkt_solver_call_count']}` 次；KKT cache 请求/命中/未命中为 `{payload['timing_diagnosis']['kkt_cache_requests_total']}` / `{payload['timing_diagnosis']['kkt_cache_hits_total']}` / `{payload['timing_diagnosis']['kkt_cache_misses_total']}`，总体命中率 `{format_number(payload['timing_diagnosis']['kkt_cache_hit_ratio_percent'])}%`，每次 Matching 的 KKT Solver 调用中位数 `{format_number(payload['timing_diagnosis']['median_kkt_solver_calls'], 0)}`。NodeEvaluationCache 按正式 Rollout 默认不启用，因此本轮请求/命中/未命中均为 0。",
            f"- KKT 核心细分完整性：CPU `{format_number(payload['timing_diagnosis']['cpu_kkt_core_median_ms'])}` ms、带宽 `{format_number(payload['timing_diagnosis']['bandwidth_kkt_core_median_ms'])}` ms、Cloud CPU `{format_number(payload['timing_diagnosis']['cloud_kkt_core_median_ms'])}` ms。正式任务均为 `NEW`；`src/decision/task_allocation.py:2988-3004` 将其导入阶段资源路径，该路径在 2886/2949 只累计总 solver core。CPU/带宽分项字段在 3063/3177 的另一分配路径中记录，因此当前路径的 CPU/带宽报告为 `NOT_AVAILABLE`，不解读为零。",
            f"- 按任务规模中位数估算的每次 KKT Solver core 时间：10/14/20 任务分别为 `{format_number(groups['10']['kkt_solver_core_time_per_solver_call_median_ms'])}` / `{format_number(groups['14']['kkt_solver_core_time_per_solver_call_median_ms'])}` / `{format_number(groups['20']['kkt_solver_core_time_per_solver_call_median_ms'])}` ms。随着任务数增加，候选评价数与每次求解涉及的资源维度都改变；本小样本不能把时间增量单独归因于候选数。",
            f"- 初步判断：{payload['timing_diagnosis']['interpretation']}",
            "- 是否引入 GNN：运行时间数据只提供效率瓶颈线索，不证明 Matching 解质量不足，也不证明 GNN 可以加速或提升效用。基于本轮瓶颈，建议先 `{}`。".format(payload['timing_diagnosis']['recommended_next_step']),
            "",
            "## 口径与限制",
            "",
            "- 表中任务效用仅解释样本，不是 Matching 最优性差距或跨规模因果结论。不同 Slot 的信道、计算需求、任务服务与可行节点边均不同。",
            "- 配置为 `FROZEN_INITIAL_CACHE_UAV_REFERENCE`；本结果不表示 PPO 策略的真实 Cache/UAV 动作。任务模型为每 Slot 独立解析式评价，无跨 Slot 执行队列。",
            "- KKT 与其 CPU/带宽/Cloud 核心细项是嵌套计时，不作重复求和。Matching 缓存与结果拼装不归类为 KKT solver core。",
            "- 本轮未训练 PPO、未读取 Validation/Test、未修改正式算法/配置/数据，未运行完整 Episode。",
            "",
            "## 最终字段",
            "",
            "```text",
            f"TRAINING_SLOTS_AVAILABLE_10 = {groups['10']['valid_slot_count'] if groups['10']['valid_slot_count'] else 'NOT_AVAILABLE'}",
            f"TRAINING_SLOTS_AVAILABLE_14 = {groups['14']['valid_slot_count'] if groups['14']['valid_slot_count'] else 'NOT_AVAILABLE'}",
            f"TRAINING_SLOTS_AVAILABLE_20 = {groups['20']['valid_slot_count'] if groups['20']['valid_slot_count'] else 'NOT_AVAILABLE'}",
            f"MATCHING_TIME_10_MEDIAN_MS = {format_number(groups['10']['matching_time_median_ms'])}",
            f"MATCHING_TIME_14_MEDIAN_MS = {format_number(groups['14']['matching_time_median_ms'])}",
            f"MATCHING_TIME_20_MEDIAN_MS = {format_number(groups['20']['matching_time_median_ms'])}",
            f"KKT_TIME_RATIO_10 = {format_number(groups['10']['kkt_time_ratio_median_percent'])}%",
            f"KKT_TIME_RATIO_14 = {format_number(groups['14']['kkt_time_ratio_median_percent'])}%",
            f"KKT_TIME_RATIO_20 = {format_number(groups['20']['kkt_time_ratio_median_percent'])}%",
            f"DOMINANT_COST_COMPONENT = {payload['timing_diagnosis']['dominant_cost_component']}",
            f"MATCHING_SCALING_EVIDENCE = {payload['timing_diagnosis']['matching_scaling_evidence']}",
            f"GNN_SPEEDUP_OPPORTUNITY = {payload['timing_diagnosis']['gnn_speedup_opportunity']}",
            f"RECOMMENDED_NEXT_STEP = {payload['timing_diagnosis']['recommended_next_step']}",
            "PPO_TRAINING_EXECUTED = NO",
            "FORMAL_ALGORITHM_MODIFIED = NO",
            "FORMAL_CONFIG_MODIFIED = NO",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, choices=(1, 2, 3), default=MAX_REPETITIONS)
    parser.add_argument("--long-run-cutoff-s", type=float, default=LONG_RUN_CUTOFF_S)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if Path(sys.prefix).name.lower() != "adaptive_cache_mec":
        raise RuntimeError(f"Expected adaptive_cache_mec Conda environment, got sys.prefix={sys.prefix}")
    if not POOL_MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Training Pool Manifest not found: {POOL_MANIFEST_PATH}")
    if REPORT_MD.exists() or REPORT_JSON.exists():
        raise FileExistsError("Refusing to overwrite an existing diagnostic report.")

    cfg = project_config.load_config(str(CONFIG_PATH))
    project_config.validate_for_formal_experiment(cfg)
    pool_manifest = json.loads(POOL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if str(pool_manifest.get("split", "")).lower() != "train":
        raise ValueError("Selected pool manifest is not Training.")
    contract_check = validate_data_contract(cfg, pool_manifest)
    if not contract_check["valid"]:
        raise ValueError(f"Current config and Training data contract differ: {contract_check}")

    expected_tv_count = int(project_config.get_config_value(cfg, "simulation.num_task_vehicles"))
    selected, scan_summary = scan_training_pool(pool_manifest, expected_tv_count)
    if not any(selected.values()):
        raise RuntimeError("No real Training Slot with requested arrived task counts was found.")

    selected_samples: list[dict[str, Any]] = []
    for task_count in TARGET_TASK_COUNTS:
        for sample in selected[task_count]:
            artifacts = validate_episode_artifacts(sample, pool_manifest)
            context = load_environment_context(
                str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
                realization_dir=sample["episode_dir"],
            )
            steps = aligned_train_steps(context, origin_start=int(sample["slot"]), max_steps=1)
            if len(steps) != 1:
                raise RuntimeError(f"aligned_train_steps returned {len(steps)} steps for {sample}")
            step = steps[0]
            if int(step.slot) != int(sample["slot"]) or len(step.actual_tasks) != task_count:
                raise RuntimeError(
                    "tasks.csv arrival count does not match formal aligned_train_steps: "
                    f"seed={sample['episode_seed']} slot={sample['slot']} "
                    f"csv={task_count} aligned={len(step.actual_tasks)}"
                )
            service_positions = task_allocation.build_service_node_positions(
                context.cache_nodes,
                step.actual_vehicle_positions,
                context.initial_uav,
            )
            feasible_targets = task_allocation.build_feasible_targets(
                step.actual_tasks,
                context.initial_cache,
                context.cache_nodes,
                vehicle_positions=step.actual_vehicle_positions,
                service_node_positions=service_positions,
                cache_config=context.cache_cfg,
                cloud_fallback_mode=context.cache_cfg.cloud_fallback_mode,
            )
            feasible_edge_count = sum(
                sum(target != task_allocation.CLOUD for target in targets)
                for targets in feasible_targets.values()
            )
            selected_samples.append(
                {
                    **sample,
                    "time_s": float(step.time_s),
                    "task_rows": int(sample["task_rows"]),
                    "episode_artifact_sha256": artifacts,
                    "context": context,
                    "step": step,
                    "input_bits_sum": float(math.fsum(task.input_bits for task in step.actual_tasks)),
                    "cpu_cycles_sum": float(math.fsum(task.cpu_cycles for task in step.actual_tasks)),
                    "feasible_edge_task_node_edges": int(feasible_edge_count),
                    "vehicle_positions_count": len(step.actual_vehicle_positions),
                }
            )

    # Warm Numba with the largest selected real slot outside all measured intervals.
    warmup_sample = max(selected_samples, key=lambda item: (item["task_count"], -item["episode_seed"], -item["slot"]))
    warmup_started = time.perf_counter()
    warmup_result = run_slot(warmup_sample["context"], warmup_sample["step"], collect_timing=False)
    warmup_runtime_s = time.perf_counter() - warmup_started

    for sample in selected_samples:
        sample["timed_repetitions"] = []
        repeat_determinism: list[dict[str, Any]] = []
        for repetition in range(args.repetitions):
            record = run_slot(sample["context"], sample["step"], collect_timing=True)
            record["repetition"] = repetition + 1
            sample["timed_repetitions"].append(record)
            repeat_determinism.append(
                {
                    "matching": record["matching"],
                    "candidate_definitions_total": record["candidate_definitions_total"],
                    "candidate_evaluations_total": record["candidate_evaluations_total"],
                    "upper_bound_pruned_candidates": record["upper_bound_pruned_candidates"],
                    "kkt_solver_calls": record["kkt_solver_calls"],
                    "task_utility_sum": record["task_utility_sum"],
                }
            )
            if repetition == 0 and record["timing_s"]["matching_total_runtime"] > args.long_run_cutoff_s:
                break
        first = sample["timed_repetitions"][0]
        sample["repeat_outputs_identical"] = all(
            item["matching"] == repeat_determinism[0]["matching"]
            and item["candidate_definitions_total"] == repeat_determinism[0]["candidate_definitions_total"]
            and item["candidate_evaluations_total"] == repeat_determinism[0]["candidate_evaluations_total"]
            and item["upper_bound_pruned_candidates"] == repeat_determinism[0]["upper_bound_pruned_candidates"]
            and item["kkt_solver_calls"] == repeat_determinism[0]["kkt_solver_calls"]
            and math.isclose(item["task_utility_sum"], repeat_determinism[0]["task_utility_sum"], rel_tol=1e-12, abs_tol=1e-12)
            for item in repeat_determinism[1:]
        )
        sample["actual_repetitions"] = len(sample["timed_repetitions"])
        if not first["resource_feasible"]:
            raise RuntimeError(f"Matching resource feasibility failed: {sample}")

    samples_by_count = {
        str(task_count): [sample for sample in selected_samples if sample["task_count"] == task_count]
        for task_count in TARGET_TASK_COUNTS
    }
    groups = {
        str(task_count): aggregate_group(task_count, samples_by_count[str(task_count)])
        for task_count in TARGET_TASK_COUNTS
    }
    all_timed = [rep for sample in selected_samples for rep in sample["timed_repetitions"]]
    overall_medians = {
        key: median_or_none(
            [
                float(rep["timing_s"][key]) * 1000.0
                for rep in all_timed
                if rep["timing_s"][key] is not None
            ]
        )
        for key in TIMING_KEYS
    }
    median_count = median_or_none([float(rep["kkt_solver_calls"]) for rep in all_timed])
    request_total = sum(rep["kkt_cache_requests"] for rep in all_timed)
    hit_total = sum(rep["kkt_cache_hits"] for rep in all_timed)
    non_kkt_ms = overall_medians["matching_runtime"]
    kkt_ms = overall_medians["kkt_runtime"]
    node_eval_ms = overall_medians["node_eval_miss_runtime"]
    if kkt_ms is not None and kkt_ms >= max(value or 0.0 for value in (non_kkt_ms, node_eval_ms)):
        edge_kkt_components = (
            overall_medians["kkt_edge_cpu_solver_core_runtime"],
            overall_medians["kkt_edge_bandwidth_solver_core_runtime"],
        )
        if any(value is None for value in edge_kkt_components):
            dominant = "KKT 总耗时为主要项；正式 Slot 路径未分别记录 Edge CPU/带宽核心时间，子瓶颈未分辨"
        else:
            solver_parts = {
                "CPU KKT core": float(edge_kkt_components[0]),
                "带宽 KKT core": float(edge_kkt_components[1]),
                "Cloud CPU KKT core": overall_medians["kkt_cloud_cpu_solver_core_runtime"] or 0.0,
            }
            dominant = max(solver_parts, key=solver_parts.get)
            dominant = f"KKT 总耗时为主要项；细分最大项为 {dominant}"
    else:
        candidates = {
            "Matching 非 KKT 管理/候选搜索": non_kkt_ms or 0.0,
            "节点效用评价（不含 KKT）": node_eval_ms or 0.0,
        }
        dominant = max(candidates, key=candidates.get)
    if groups["10"]["matching_time_median_ms"] is None or groups["14"]["matching_time_median_ms"] is None or groups["20"]["matching_time_median_ms"] is None:
        scaling = "某任务规模没有可用真实 Slot；不做跨规模时间比较"
    else:
        time_values = [groups[str(n)]["matching_time_median_ms"] for n in TARGET_TASK_COUNTS]
        eval_values = [groups[str(n)]["median_counts_per_matching"]["candidate_evaluations_total"] for n in TARGET_TASK_COUNTS]
        scaling = (
            f"样本组中位 Matching ms={','.join(format_number(x) for x in time_values)}，"
            f"候选评价中位数={','.join(format_number(x, 0) for x in eval_values)}；"
            "仅是固定小样本的描述性变化，不能据此拟合或断言复杂度阶数。"
        )
    kkt_hit_ratio = 100.0 * hit_total / request_total if request_total else 0.0
    matching_total_ms = overall_medians["matching_total_runtime"] or 0.0
    max_matching_ms = max(float(rep["timing_s"]["matching_total_runtime"]) * 1000.0 for rep in all_timed)
    if max_matching_ms < 1000.0:
        gnn_opportunity = f"9 个抽样 Slot 的 Matching 调用均低于 1 s，最大约 {max_matching_ms:.3f} ms；本次没有足够效率证据支持增加 GNN，不代表更大规模或更严实时限下也满足要求。"
        recommended = "暂不引入 GNN；若继续优化，先补齐正式解析式路径的 Edge CPU/带宽 KKT 分项计时，再按实测主要子项做局部实现优化。"
    elif "KKT 主导" in dominant:
        gnn_opportunity = "低；主要耗时在 KKT 资源分配，GNN 不会直接消除该瓶颈。"
        recommended = "先针对占主导的 KKT CPU/带宽/Cloud 子求解器做实现层优化，保持公式和约束不变。"
    else:
        gnn_opportunity = "有限且未证实；如候选搜索/管理主导，可在后续单独测量 GNN 生成初始匹配的收益与质量，再考虑接入。"
        recommended = "先优化当前主导的候选生成/评价路径；需要 GNN 时，以 Matching+KKT 作为质量基线做单独对照。"
    if kkt_ms is None:
        interpretation = "没有成功测得 KKT 时间，不能判定热点。"
    elif kkt_ms >= (non_kkt_ms or 0.0):
        if overall_medians["kkt_edge_cpu_solver_core_runtime"] is None or overall_medians["kkt_edge_bandwidth_solver_core_runtime"] is None:
            interpretation = "KKT 总时间不小于剩余 Matching 时间；当前正式 NEW-task 路径未分别记录 Edge CPU/带宽核心时间，先补齐分项计时再定位子瓶颈。"
        else:
            interpretation = "KKT 总时间不小于剩余 Matching 时间；结合可用 CPU/带宽/Cloud 子项和 KKT cache 命中定位。"
    elif (node_eval_ms or 0.0) > kkt_ms:
        interpretation = "节点效用评价的非 KKT 部分高于 KKT，优先核查节点评价/缓存复用。"
    else:
        interpretation = "非 KKT Matching 路径高于 KKT 总时间；结合候选细分判断管理/生成/筛选开销。"

    objective = cfg["objective"]
    cloud = cfg["nodes"]["cloud"]
    payload = {
        "schema_version": "matching_kkt_training_efficiency_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "project_root": str(PROJECT_ROOT),
            "branch": git_value("branch", "--show-current"),
            "head": git_value("rev-parse", "HEAD"),
            "preexisting_worktree_status": git_value("status", "--short", "--branch"),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "python_version": platform.python_version(),
            "conda_environment": Path(sys.prefix).name,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "total_memory_gib": None,
        },
        "configuration": {
            "path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(CONFIG_PATH),
            "objective": {
                "omega_delay": float(objective["omega_delay"]),
                "omega_energy": float(objective["omega_energy"]),
                "T_ref_s": float(objective["T_ref"]),
                "E_ref_j": float(objective["E_ref"]),
                "lambda_cache": float(objective["lambda_cache"]),
                "lambda_movement": float(objective["lambda_movement"]),
            },
            "matching_contract": {
                "method": str(cfg["algorithm"]["matching"]["method"]),
                "max_updates": int(cfg["algorithm"]["matching"]["max_updates"]),
                "objective_tolerance": float(cfg["algorithm"]["matching"]["objective_tolerance"]),
                "cloud_fallback_mode": str(cfg["algorithm"]["matching"]["cloud_fallback_mode"]),
            },
            "kkt_contract": {
                key: cfg["algorithm"]["kkt"][key]
                for key in ("min_cpu_fraction", "min_bandwidth_fraction", "bisection_tolerance", "max_bisection_iterations")
            },
            "noise_model": str(cfg["communication"]["noise"]["model"]),
            "resource_model": cfg["communication"]["resource_model"],
            "cross_slot_task_model": str(cfg["execution"]["cross_slot_task_model"]),
            "sv_cpu_cycles_per_s": float(cfg["nodes"]["sv"]["cpu_cycles_per_s"]),
            "rsu_cpu_cycles_per_s": float(cfg["nodes"]["rsu"]["cpu_cycles_per_s"]),
            "uav_cpu_cycles_per_s": float(cfg["nodes"]["uav"]["cpu_cycles_per_s"]),
            "cloud_cpu_capacity_cycles_per_s": float(cloud["cpu_capacity_cycles_per_s"]),
            "cloud_resource_mode": str(cloud["resource_mode"]),
            "cloud_uplink_rate_bps": float(cloud["uplink_rate_bps"]),
            "cloud_delay_factor": float(cloud["cloud_delay_factor"]),
        },
        "training_pool": {
            "path": str(POOL_ROOT.relative_to(PROJECT_ROOT)),
            "manifest_path": str(POOL_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "manifest_sha256": sha256_file(POOL_MANIFEST_PATH),
            "config_sha256": pool_manifest.get("config_sha256"),
            "config_hash_matches_current": pool_manifest.get("config_sha256") == sha256_file(CONFIG_PATH),
            "episode_count": int(pool_manifest["episode_count"]),
            "slots_per_episode": int(pool_manifest["slots_per_episode"]),
            "split": pool_manifest["split"],
        },
        "data_contract": contract_check,
        "sample_scan": scan_summary,
        "method": {
            "state_reference": "FROZEN_INITIAL_CACHE_UAV_REFERENCE",
            "matching_entrypoint": "src/decision/slot_environment.py:execute_configured_slot -> src/decision/task_allocation.py:optimize_task_matching",
            "sample_api": "src/ppo_simple/interface.py:load_environment_context + aligned_train_steps",
            "evaluation_mode": "incremental (formal optimize_task_matching default)",
            "numba_available": task_allocation.numba is not None,
            "numba_warmup_sample": {
                "episode_seed": warmup_sample["episode_seed"],
                "slot": warmup_sample["slot"],
                "task_count": warmup_sample["task_count"],
            },
            "numba_warmup_unmeasured_runtime_s": warmup_runtime_s,
            "warmup_matching_result": {
                "task_utility_sum": warmup_result["task_utility_sum"],
                "kkt_solver_calls": warmup_result["kkt_solver_calls"],
            },
            "max_timed_repetitions": args.repetitions,
            "long_run_cutoff_s": args.long_run_cutoff_s,
            "timed_matching_call_count": len(all_timed),
            "timed_kkt_solver_call_count": sum(int(rep["kkt_solver_calls"]) for rep in all_timed),
            "numba_warmup_excluded_from_timed_calls": True,
            "node_evaluation_cache_passed": False,
            "fresh_kkt_cache_each_call": True,
            "formal_data_modified": False,
        },
        "sample_count": len(selected_samples),
        "samples_by_task_count": {},
        "group_summary": groups,
        "timing_diagnosis": {
            "matching_total_runtime_median_ms": overall_medians["matching_total_runtime"],
            "max_matching_runtime_ms": max_matching_ms,
            "matching_non_kkt_residual_median_ms": overall_medians["matching_runtime"],
            "node_evaluation_excluding_kkt_median_ms": overall_medians["node_eval_miss_runtime"],
            "candidate_generation_median_ms": overall_medians["matching_candidate_generation_runtime"],
            "upper_bound_screening_median_ms": overall_medians["matching_upper_bound_runtime"],
            "exact_evaluator_orchestration_median_ms": overall_medians["matching_exact_eval_orchestration_runtime"],
            "kkt_total_time_median_ms": overall_medians["kkt_runtime"],
            "kkt_solver_core_median_ms": overall_medians["kkt_solver_core_runtime"],
            "cpu_kkt_core_median_ms": overall_medians["kkt_edge_cpu_solver_core_runtime"],
            "bandwidth_kkt_core_median_ms": overall_medians["kkt_edge_bandwidth_solver_core_runtime"],
            "cloud_kkt_core_median_ms": overall_medians["kkt_cloud_cpu_solver_core_runtime"],
            "median_kkt_solver_calls": median_count,
            "kkt_cache_requests_total": request_total,
            "kkt_cache_hits_total": hit_total,
            "kkt_cache_misses_total": sum(rep["kkt_cache_misses"] for rep in all_timed),
            "kkt_cache_hit_ratio_percent": kkt_hit_ratio,
            "dominant_cost_component": dominant,
            "matching_scaling_evidence": scaling,
            "gnn_speedup_opportunity": gnn_opportunity,
            "recommended_next_step": recommended,
            "interpretation": interpretation,
        },
        "ppo_training_executed": False,
        "formal_algorithm_modified": False,
        "formal_config_modified": False,
        "formal_data_modified": False,
    }
    try:
        import psutil  # type: ignore[import-not-found]

        payload["environment"]["total_memory_gib"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        pass

    for task_count in TARGET_TASK_COUNTS:
        clean_samples = []
        for sample in samples_by_count[str(task_count)]:
            clean_samples.append({key: value for key, value in sample.items() if key not in {"context", "step"}})
        payload["samples_by_task_count"][str(task_count)] = clean_samples

    report_text = build_report(payload)
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    REPORT_MD.write_text(report_text, encoding="utf-8")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")
    for task_count in TARGET_TASK_COUNTS:
        group = groups[str(task_count)]
        print(
            f"tasks={task_count} slots={group['valid_slot_count']} "
            f"matching_median_ms={format_number(group['matching_time_median_ms'])} "
            f"kkt_median_ms={format_number(group['kkt_total_time_median_ms'])} "
            f"kkt_ratio={format_number(group['kkt_time_ratio_median_percent'])}%"
        )
    print(f"dominant={dominant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
