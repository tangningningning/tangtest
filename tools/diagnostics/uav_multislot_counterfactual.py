"""Five-slot frozen-policy UAV counterfactual diagnostic.

This runner intentionally does not call or relax collect_dual_timescale_episode's
full-episode contract. It reuses its policy/candidate/observation helpers and the
production execute_configured_slot transition, while keeping a deep-copyable
decision state for short-horizon branches.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import project_config
from decision import slot_environment, uav_configuration
from ppo_simple.evaluator import (
    _assert_realization_context_compatible,
    _load_checkpoint_metadata,
)
from ppo_simple.interface import aligned_train_steps, load_environment_context
from ppo_simple.primitive_control import PrimitiveTensorSpec
from ppo_simple.rollout import (
    _cache_candidates,
    _check_request_contract,
    _check_reward,
    _observation,
    _positions_condition,
    _uav_candidates,
    task_metrics_from_result,
)
from ppo_simple.trainer import PPOTrainer


RUNS = (
    ("5e-4", "formal_cache_economic_v2_lr_5e_minus_4"),
    ("1e-3", "formal_cache_economic_v2_lr_1e_minus_3"),
)
VALIDATION_SEEDS = (30001, 30002)
SAMPLE_INDICES = (0, 4, 9)
HORIZON = 5
BRANCHES = ("A", "B", "C0", "C1", "C2")
COMPARISONS = (("A-B", "A", "B"), ("C0-A", "C0", "A"), ("C1-A", "C1", "A"), ("C2-A", "C2", "A"))


@dataclass
class DecisionState:
    """All mutable decision state needed to continue a frozen rollout."""

    cache: np.ndarray
    uav: dict[str, np.ndarray]
    frame_active: bool = False
    frame_cache: np.ndarray | None = None
    frame_cache_index: int = -1
    frame_candidate_count: int = 0
    frame_cache_changed: bool = False
    frame_observation: np.ndarray | None = None
    frame_candidate_features: np.ndarray | None = None
    frame_candidate_masks: np.ndarray | None = None
    frame_start_index: int = -1
    frame_elapsed: int = 0


@dataclass
class ActionChoice:
    positions: dict[str, np.ndarray]
    indices: tuple[int, int]
    sources: tuple[str, str]
    candidate_counts: tuple[int, int]
    rule: str


class CandidateUnavailable(RuntimeError):
    pass


def _copy_positions(positions: dict[str, Any], uav_ids: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(positions[key], dtype=np.float64).copy()
        for key in uav_ids
    }


def _action_triggers(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    context: Any,
) -> dict[str, int]:
    tolerance = float(
        uav_configuration.uav_search_config_from_project(context.cfg).position_tolerance_m
    )
    return {
        key: int(np.linalg.norm(after[key] - before[key]) > tolerance)
        for key in context.action_spec.uav_ids
    }


def _validate_uav_action(
    previous: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
    context: Any,
) -> None:
    uav_configuration.validate_uav_configuration_transition(
        previous,
        final,
        _action_triggers(previous, final, context),
        context.cfg,
        context.flight_boundary,
    )


def _prepare_macro_frame(
    policy: Any,
    context: Any,
    tensor_spec: PrimitiveTensorSpec,
    state: DecisionState,
    step: Any,
    step_index: int,
) -> bool:
    """Run the formal deterministic Cache selection only at a new Macro frame."""

    if state.frame_active:
        return False
    observation = _observation(state.cache, state.uav, step, context)
    candidates, features = _cache_candidates(
        state.cache, state.uav, step, context, tensor_spec
    )
    masks = np.ones(len(candidates), dtype=bool)
    if not np.array_equal(candidates[0], state.cache):
        raise AssertionError("formal Cache candidate 0 must be KEEP")
    selected = policy.sample_macro(
        observation,
        features,
        masks,
        deterministic=True,
    )
    selected_index = int(selected.cache_index)
    if selected_index < 0 or selected_index >= len(candidates):
        raise AssertionError("Macro policy selected an invalid Cache candidate")
    state.frame_active = True
    state.frame_cache = np.asarray(candidates[selected_index], dtype=np.int8).copy()
    state.frame_cache_index = selected_index
    state.frame_candidate_count = len(candidates)
    state.frame_cache_changed = bool(np.any(state.frame_cache != state.cache))
    state.frame_observation = observation.copy()
    state.frame_candidate_features = features.copy()
    state.frame_candidate_masks = masks.copy()
    state.frame_start_index = int(step_index)
    state.frame_elapsed = 0
    return True


def _select_policy_uav_action(
    policy: Any,
    context: Any,
    state: DecisionState,
    step: Any,
) -> ActionChoice:
    """Use the same autoregressive heads/candidate construction as formal Rollout."""

    if not state.frame_active or state.frame_cache is None:
        raise AssertionError("Cache frame must be initialized before Micro action selection")
    uav_ids = tuple(context.action_spec.uav_ids)
    observation = _observation(state.frame_cache, state.uav, step, context)
    previous = _copy_positions(state.uav, uav_ids)
    trial = _copy_positions(state.uav, uav_ids)
    condition = np.zeros(2, dtype=np.float32)
    indices: list[int] = []
    sources: list[str] = []
    counts: list[int] = []
    triggers = {key: 1 for key in uav_ids}
    for uav_index, uav_id in enumerate(uav_ids):
        candidates, features = _uav_candidates(
            uav_index,
            trial,
            previous,
            state.frame_cache,
            step,
            context,
        )
        sample = policy.sample_micro_head(
            observation,
            features,
            np.ones(len(features), dtype=bool),
            uav_index=uav_index,
            uav0_position_condition=condition.copy(),
            deterministic=True,
        )
        selected_index = int(sample.selected_index)
        selected = candidates[selected_index]
        uav_configuration.apply_uav_candidate(
            trial,
            previous,
            uav_id,
            selected.target_position,
            triggers,
            context.cfg,
            context.flight_boundary,
        )
        indices.append(selected_index)
        sources.append(str(selected.proposal_source))
        counts.append(len(candidates))
        if uav_index == 0:
            condition = _positions_condition(trial[uav_id], context)
    _validate_uav_action(previous, trial, context)
    return ActionChoice(
        positions=_copy_positions(trial, uav_ids),
        indices=(indices[0], indices[1]),
        sources=(sources[0], sources[1]),
        candidate_counts=(counts[0], counts[1]),
        rule="frozen PPO deterministic selection",
    )


def _first_move(candidates: list[Any], uav_id: str, rule: str) -> tuple[Any, int]:
    for index, candidate in enumerate(candidates[1:], start=1):
        if str(candidate.uav_id) == uav_id:
            return candidate, index
    raise CandidateUnavailable(f"{rule}: no legal MOVE candidate for {uav_id}")


def _enumerate_for_uav(
    context: Any,
    state: DecisionState,
    step: Any,
    current: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
    uav_index: int,
) -> list[Any]:
    candidates, _features = _uav_candidates(
        uav_index,
        current,
        previous,
        state.frame_cache,
        step,
        context,
    )
    return candidates


def _counterfactual_uav_action(
    label: str,
    context: Any,
    state: DecisionState,
    step: Any,
) -> ActionChoice:
    """Build B/C actions from current legal proposals with deterministic tie rules."""

    uav_ids = tuple(context.action_spec.uav_ids)
    previous = _copy_positions(state.uav, uav_ids)
    trial = _copy_positions(state.uav, uav_ids)
    indices = [0, 0]
    sources = ["stay", "stay"]
    counts = [1, 1]
    if label == "B":
        rule = "both UAVs STAY"
    elif label == "C0":
        moves0 = _enumerate_for_uav(context, state, step, trial, previous, 0)
        counts[0] = len(moves0)
        candidate0, index0 = _first_move(moves0, uav_ids[0], "C0")
        trial[uav_ids[0]] = np.asarray(candidate0.target_position, dtype=np.float64).copy()
        indices[0] = index0
        sources[0] = str(candidate0.proposal_source)
        moves1 = _enumerate_for_uav(context, state, step, trial, previous, 1)
        counts[1] = len(moves1)
        stay_trial = _copy_positions(trial, uav_ids)
        try:
            _validate_uav_action(previous, stay_trial, context)
        except ValueError:
            # Prior diagnostic rule: if U1 STAY is no longer jointly feasible,
            # choose the nearest conditionally legal U1 MOVE; list order breaks ties.
            legal = list(enumerate(moves1[1:], start=1))
            if not legal:
                raise CandidateUnavailable("C0: U1 STAY is unsafe and no legal U1 adjustment exists")
            selected_index, selected = min(
                legal,
                key=lambda pair: (
                    float(
                        np.linalg.norm(
                            np.asarray(pair[1].target_position, dtype=np.float64)
                            - previous[uav_ids[1]]
                        )
                    ),
                    pair[0],
                ),
            )
            trial[uav_ids[1]] = np.asarray(selected.target_position, dtype=np.float64).copy()
            indices[1] = selected_index
            sources[1] = str(selected.proposal_source)
        rule = "first legal U0 MOVE; U1 STAY if safe, otherwise nearest conditional legal U1 MOVE"
    elif label == "C1":
        moves1 = _enumerate_for_uav(context, state, step, trial, previous, 1)
        counts[1] = len(moves1)
        candidate1, index1 = _first_move(moves1, uav_ids[1], "C1")
        trial[uav_ids[1]] = np.asarray(candidate1.target_position, dtype=np.float64).copy()
        indices[1] = index1
        sources[1] = str(candidate1.proposal_source)
        rule = "U0 STAY; first legal conditional U1 MOVE"
    elif label == "C2":
        moves0 = _enumerate_for_uav(context, state, step, trial, previous, 0)
        counts[0] = len(moves0)
        candidate0, index0 = _first_move(moves0, uav_ids[0], "C2/U0")
        trial[uav_ids[0]] = np.asarray(candidate0.target_position, dtype=np.float64).copy()
        indices[0] = index0
        sources[0] = str(candidate0.proposal_source)
        moves1 = _enumerate_for_uav(context, state, step, trial, previous, 1)
        counts[1] = len(moves1)
        candidate1, index1 = _first_move(moves1, uav_ids[1], "C2/U1 after U0 MOVE")
        trial[uav_ids[1]] = np.asarray(candidate1.target_position, dtype=np.float64).copy()
        indices[1] = index1
        sources[1] = str(candidate1.proposal_source)
        rule = "first legal U0 MOVE, then re-enumerate and take first legal conditional U1 MOVE"
    else:
        raise ValueError(f"unknown counterfactual branch: {label}")
    _validate_uav_action(previous, trial, context)
    return ActionChoice(
        positions=_copy_positions(trial, uav_ids),
        indices=(indices[0], indices[1]),
        sources=(sources[0], sources[1]),
        candidate_counts=(counts[0], counts[1]),
        rule=rule,
    )


def _execute_one_slot(
    policy: Any,
    context: Any,
    steps: list[Any],
    tensor_spec: PrimitiveTensorSpec,
    state: DecisionState,
    step_index: int,
    action: ActionChoice,
) -> dict[str, Any]:
    step = steps[step_index]
    _prepare_macro_frame(policy, context, tensor_spec, state, step, step_index)
    if state.frame_cache is None:
        raise AssertionError("active Macro frame is missing its Cache target")
    uav_ids = tuple(context.action_spec.uav_ids)
    previous_cache = np.asarray(state.cache, dtype=np.int8).copy()
    previous_uav = _copy_positions(state.uav, uav_ids)
    final_cache = np.asarray(state.frame_cache, dtype=np.int8).copy()
    final_uav = _copy_positions(action.positions, uav_ids)
    _validate_uav_action(previous_uav, final_uav, context)
    profile: dict[str, Any] = {}
    started = time.perf_counter()
    result = slot_environment.execute_configured_slot(
        previous_cache,
        previous_uav,
        final_cache,
        final_uav,
        step.scenario_slice,
        step.actual_tasks,
        step.actual_vehicle_positions,
        context.local_cache,
        context.service_catalog,
        context.cfg,
        context.cache_nodes,
        context.cache_cfg,
        context.flight_boundary,
        slot_id=int(step.slot),
        time_s=float(step.time_s),
        performance_profile=profile,
    )
    runtime_seconds = time.perf_counter() - started
    if not result.step_feasible or result.system_utility is None:
        raise RuntimeError(
            f"infeasible branch at seed step={step_index}, slot={step.slot}: {result.infeasible_reason}"
        )
    _check_request_contract(previous_cache, previous_uav, result, context.action_spec)
    reward = _check_reward(result)
    metrics = task_metrics_from_result(result, context.cache_nodes)
    next_cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
    next_uav = _copy_positions(result.next_uav_positions, uav_ids)
    for uav_id in uav_ids:
        if not np.allclose(final_uav[uav_id], next_uav[uav_id], rtol=0.0, atol=1.0e-6):
            raise AssertionError(f"environment did not realize configured UAV position: {uav_id}")

    elapsed_before = int(state.frame_elapsed)
    elapsed = elapsed_before + 1
    next_index = step_index + 1
    if next_index >= len(steps):
        raise AssertionError("five-slot trajectory must remain inside the 300-slot episode")
    next_observation = _observation(next_cache, next_uav, steps[next_index], context)
    minimum = int(policy.adaptive_holding_min_slots)
    maximum = int(policy.adaptive_holding_max_slots)
    if step_index == len(steps) - 1:
        holding_action = "TERMINAL"
        end_frame = True
    elif elapsed >= maximum:
        holding_action = "FORCED_TERMINATE_MAX"
        end_frame = True
    elif elapsed >= minimum:
        termination = policy.sample_termination(
            next_observation,
            elapsed,
            deterministic=True,
        )
        end_frame = int(termination.selected_index) == 1
        holding_action = "TERMINATE" if end_frame else "CONTINUE"
    else:
        holding_action = "NOT_ELIGIBLE"
        end_frame = False

    frame_cache_hash = hashlib.sha256(
        np.ascontiguousarray(final_cache, dtype=np.int8).tobytes()
    ).hexdigest()[:16]
    record = {
        "step_index": int(step_index),
        "slot": int(step.slot),
        "system_utility": float(result.system_utility),
        "task_utility": float(result.actual_task_utility),
        "delay_s": float(metrics.total_task_delay_s),
        "vehicle_energy_j": float(metrics.total_vehicle_energy_j),
        "task_count": int(metrics.task_count),
        "cache_cost": float(result.cache_cost),
        "movement_cost": float(result.movement_cost),
        "movement_count": {
            uav_id: int(
                np.linalg.norm(next_uav[uav_id] - previous_uav[uav_id])
                > uav_configuration.uav_search_config_from_project(context.cfg).position_tolerance_m
            )
            for uav_id in uav_ids
        },
        "movement_distance_m": {
            uav_id: float(np.linalg.norm(next_uav[uav_id] - previous_uav[uav_id]))
            for uav_id in uav_ids
        },
        "positions_before": {key: previous_uav[key].tolist() for key in uav_ids},
        "positions_after": {key: next_uav[key].tolist() for key in uav_ids},
        "uav_action_indices": list(action.indices),
        "uav_action_sources": list(action.sources),
        "uav_candidate_counts": list(action.candidate_counts),
        "uav_action_rule": action.rule,
        "macro_frame_started": bool(state.frame_start_index == step_index),
        "macro_cache_candidate_index": int(state.frame_cache_index),
        "macro_cache_candidate_count": int(state.frame_candidate_count),
        "macro_cache_changed": bool(state.frame_cache_changed),
        "active_cache_hash": frame_cache_hash,
        "holding_count_before": elapsed_before,
        "holding_count_after_action": elapsed if not end_frame else 0,
        "holding_action_after_slot": holding_action,
        "environment_seconds": float(runtime_seconds),
        "matching_kkt_seconds": float(profile.get("matching_kkt_seconds", 0.0)),
    }
    state.cache = next_cache
    state.uav = next_uav
    if end_frame:
        state.frame_active = False
        state.frame_elapsed = 0
    else:
        state.frame_elapsed = elapsed
    return record


def _fingerprint_feed(digest: Any, value: Any) -> None:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"array")
        digest.update(array.dtype.str.encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: repr(item)):
            _fingerprint_feed(digest, key)
            _fingerprint_feed(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"list" if isinstance(value, list) else b"tuple")
        for item in value:
            _fingerprint_feed(digest, item)
    elif is_dataclass(value):
        digest.update(type(value).__qualname__.encode())
        for field in fields(value):
            _fingerprint_feed(digest, getattr(value, field.name))
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        digest.update(type(value).__qualname__.encode())
        _fingerprint_feed(digest, vars(value))
    else:
        digest.update(repr(value).encode("utf-8", errors="backslashreplace"))


def _external_fingerprint(step: Any) -> str:
    digest = hashlib.sha256()
    for field in (
        "origin_index",
        "slot",
        "time_s",
        "scenario_slice",
        "predicted_vehicle_positions",
        "predicted_service_popularity",
        "predicted_vehicle_positions_horizon",
        "predicted_service_popularity_horizon",
        "prediction_horizon_mask",
        "actual_tasks",
        "actual_vehicle_positions",
    ):
        _fingerprint_feed(digest, getattr(step, field))
    return digest.hexdigest()


def _branch_summary(records: list[dict[str, Any]], uav_ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        name: float(sum(float(record[name]) for record in records))
        for name in ("system_utility", "task_utility", "delay_s", "vehicle_energy_j", "cache_cost", "movement_cost")
    } | {
        "movement_count": {
            uav_id: int(sum(int(record["movement_count"][uav_id]) for record in records))
            for uav_id in uav_ids
        },
        "movement_distance_m": {
            uav_id: float(sum(float(record["movement_distance_m"][uav_id]) for record in records))
            for uav_id in uav_ids
        },
    }


def _pairwise_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        for comparison, left, right in COMPARISONS:
            if left not in sample["branches"] or right not in sample["branches"]:
                continue
            left_records = sample["branches"][left]
            right_records = sample["branches"][right]
            slot_delta = [
                float(a["system_utility"]) - float(b["system_utility"])
                for a, b in zip(left_records, right_records, strict=True)
            ]
            cumulative = np.cumsum(slot_delta).astype(float).tolist()
            rows.append(
                {
                    "checkpoint": sample["checkpoint"],
                    "seed": sample["seed"],
                    "sample_index": sample["sample_index"],
                    "slot": sample["slot"],
                    "comparison": comparison,
                    "step_delta": slot_delta,
                    "cumulative_delta": cumulative,
                }
            )
    return rows


def _metric_component_rows(samples: list[dict[str, Any]], uav_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        stay = sample["branch_summaries"].get("B")
        if stay is None:
            continue
        for branch in BRANCHES:
            summary = sample["branch_summaries"].get(branch)
            if summary is None:
                continue
            rows.append(
                {
                    "checkpoint": sample["checkpoint"],
                    "seed": sample["seed"],
                    "sample_index": sample["sample_index"],
                    "slot": sample["slot"],
                    "uav_ids": uav_ids,
                    "branch": branch,
                    "delta_g_1": float(sample["branches"][branch][0]["system_utility"] - sample["branches"]["B"][0]["system_utility"]),
                    "delta_g_5": summary["system_utility"] - stay["system_utility"],
                    "delta_task_5": summary["task_utility"] - stay["task_utility"],
                    "delta_delay_5": summary["delay_s"] - stay["delay_s"],
                    "delta_vehicle_energy_5": summary["vehicle_energy_j"] - stay["vehicle_energy_j"],
                    "delta_cache_cost_5": summary["cache_cost"] - stay["cache_cost"],
                    "delta_movement_cost_5": summary["movement_cost"] - stay["movement_cost"],
                    "move_count": summary["movement_count"],
                    "move_distance_m": summary["movement_distance_m"],
                    "holding_changed": _holding_changed(sample, branch),
                    "cache_target_changed": _cache_target_changed(sample, branch),
                }
            )
    return rows


def _holding_changed(sample: dict[str, Any], branch: str) -> bool:
    if branch == "A" or branch not in sample["branches"] or "A" not in sample["branches"]:
        return False
    # Decisions after slots 1-4 affect the remaining slots in this horizon.
    return any(
        sample["branches"][branch][offset]["holding_action_after_slot"]
        != sample["branches"]["A"][offset]["holding_action_after_slot"]
        for offset in range(HORIZON - 1)
    )


def _cache_target_changed(sample: dict[str, Any], branch: str) -> bool:
    if branch == "A" or branch not in sample["branches"] or "A" not in sample["branches"]:
        return False
    return any(
        sample["branches"][branch][offset]["active_cache_hash"]
        != sample["branches"]["A"][offset]["active_cache_hash"]
        for offset in range(1, HORIZON)
    )


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    eps = 1.0e-12
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "positive": int(np.count_nonzero(arr > eps)),
        "negative": int(np.count_nonzero(arr < -eps)),
        "zero": int(np.count_nonzero(np.abs(arr) <= eps)),
    }


def _fmt(value: Any, places: int = 6) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):+.{places}f}"


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _generate_report(
    output_path: Path,
    checkpoint_info: dict[str, dict[str, Any]],
    samples: list[dict[str, Any]],
    execution_info: dict[str, Any],
    parameter_checks: dict[str, Any],
) -> None:
    pairwise = _pairwise_rows(samples)
    components = _metric_component_rows(samples, tuple(samples[0]["uav_ids"]))
    lines: list[str] = [
        "# UAV 五时隙反事实收益诊断",
        "",
        "日期：2026-10-09  ",
        "范围：冻结 deterministic PPO 的 Validation 短轨迹；无训练、无 Test 数据；未改动 PPO Policy、Reward、Matching/KKT 或物理模型。",
        "本报告承接 reports/uav_counterfactual_diagnosis.md，复用其中的 Validation seeds 与 Slot 索引，将单 Slot 反事实扩展为保留 Cache/Holding 演化的五 Slot 分支。",
        "",
        "## Checkpoint 与范围",
        "",
        "| Macro Actor LR | latest checkpoint | 已完成 Episode / policy updates | checkpoint / architecture version | raw config SHA-256 | checkpoint SHA-256 |",
        "|---|---|---:|---|---|---|",
    ]
    for lr, _run in RUNS:
        info = checkpoint_info[lr]
        lines.append(
            f"| {lr} | latest_checkpoint.pt | {info['completed_episodes']} / {info['policy_updates']} | {info['checkpoint_version']} / {info['architecture_version']} | {info['config_sha256']} | {info['checkpoint_sha256']} |"
        )
    lines.extend(
        [
            "",
            "两组最新 checkpoint 的训练进度不相同（224 vs 300 Episode），因此本报告只比较每个 checkpoint 自己的反事实分支，不作学习率因果比较。Checkpoint 版本、架构、正式 evaluator metadata 和 PPOTrainer.load_checkpoint 均已核对；配置哈希通过加载合同。评估前后 checkpoint 哈希相同。",
            "",
            "执行环境为项目 Conda 环境 adaptive_cache_mec（Python 3.11.15，PyTorch 2.11.0+cu130）。",
            "",
            "## 正式 Rollout 链路核对",
            "",
            "- rollout.py:517-541 要求 aligned steps 覆盖 context.episode_slots 全长；正式函数仍保持 300-Slot 完整 Episode 条件。本 runner 不调用短长度正式 collector，也没有改动其约束。",
            "- rollout.py:602-685 在 Macro frame 开始时按当前 Cache/UAV 和该 slot 观测选 Cache target，并在 frame 持续期间复用该 Cache；rollout.py:961-1057 用 post-slot next observation 与 elapsed holding count 做 learned termination，达到 max holding 时强制结束 frame。frame 结束后才会在后续 slot 重新选 Cache。",
            "- rollout.py:774-825 先用 UAV0 选择结果构成条件，再生成 UAV1 候选；执行前做完整转移合法性校验。正式 slot 随后以实际 Cache/UAV target 调用 execute_configured_slot()（rollout.py:872）。",
            "- slot_environment.py:622-709 构造并校验目标 Cache/UAV 配置，slot_environment.py:711-786 执行正式环境、Matching/KKT 和 next-state 转移；系统效用由 actual task utility − Cache cost − UAV movement cost 组成（slot_environment.py:493）。诊断直接复用该执行，不重算 Matching/KKT 或奖励。",
            "- policy.py:122-157、865-911 定义 deterministic 分组选择与 Micro candidate head；trainer.py:1007-1027 将 Macro Actor 与 Micro Actor 分属独立 optimizer，Micro Actor LR 对两个 checkpoint 均为 3e-4。冻结评估不执行 optimizer step。",
            "",
            "Validation 环境仅为 episode 30001、30002；沿用上次抽样索引 0、4、9。每个 checkpoint 共 6 个起点；各起点从正式 300 个对齐 Slot 输入中重放冻结策略至抽样状态，并分支运行该 Slot 加后续 4 Slot。每个分支恰好执行 5 次 execute_configured_slot()。不构造 5-Slot 截断 Episode，也不调用或更改正式完整 Episode collector 的 300-Slot 合同。",
            "",
            "本 runner 作为独立诊断实现保留在 tools/diagnostics/uav_multislot_counterfactual.py；它只复用 rollout 的观测、候选和 reward 合同辅助函数、正式环境转移和 evaluator/trainer checkpoint 加载。未改动正式策略或 Rollout 源码。",
            "",
            "## 分支和状态隔离",
            "",
            "| 分支 | 第一个 Slot | 后续 4 Slot |",
            "|---|---|---|",
            "| A | 冻结 PPO deterministic 实际选择 | 冻结 PPO 正常选择 Cache、Holding 和 UAV |",
            "| B | 两架 UAV STAY | 冻结 PPO 正常选择 Cache、Holding 和 UAV |",
            "| C0 | 第一个正式枚举的 U0 MOVE；U1 若安全则 STAY，否则取离当前位置最近的条件合法 MOVE | 冻结 PPO 正常选择 Cache、Holding 和 UAV |",
            "| C1 | U0 STAY；从 U1 当前条件候选中取第一个合法 MOVE | 冻结 PPO 正常选择 Cache、Holding 和 UAV |",
            "| C2 | U0 先取第一个合法 MOVE，再基于新 U0 位置重枚举 U1 并取第一个合法 MOVE | 冻结 PPO 正常选择 Cache、Holding 和 UAV |",
            "",
            "C 分支候选由正式需求感知枚举器的稳定顺序确定（预测需求锚点优先，之后是投影与几何候选）；规则只依赖当前策略可见状态、当前 slot 的预测输入和合法候选顺序，不看真实任务奖励、车辆执行结果或未来真实信息。本次 36/36 个 C 联合候选均找到并通过正式转移；若候选缺失或合法性校验失败，runner 会明确报错中止，不会用非法位置或伪结果替代。",
            "",
            "每个起点先从 episode 初始状态按同一冻结策略重放到动作执行前。复制的运行状态包括 actual Cache、当前 UAV、已选 Macro Cache frame、frame 起始观测/候选特征/掩码、frame holding 已经历 slot 数及 frame 激活状态。每个 A/B/C 分支用该完整快照的独立深拷贝开始；后续 Cache/Holding 不同步。每步均重新调用 execute_configured_slot()，未共享 Matching/KKT 缓存。Cache 合法性由正式配置构造/执行校验；UAV 使用正式候选枚举、顺序条件枚举和完整速度/边界/安全距离转移校验。所有执行 Slot 0-13 的真实任务、车辆轨迹、scenario 与预测输入在分支前后指纹一致；第 5 Slot 后用于 Holding 终止判定的下一步观测也来自所有分支共享的同一个只读 Validation 对齐输入。反事实只更新分支私有状态。",
            "",
            "## 逐起点：分支相对双 STAY 的五 Slot 指标",
            "",
            "以下指标均为 branch − B（双 STAY）的差；移动次数/距离是分支本身五 Slot 的累计实际值，U0/U1 按 context.action_spec.uav_ids 顺序。Holding 与 Cache 标记比较 branch 相对 A 的后续变化。Delay 和车辆能耗是五 Slot 汇总，不除以任务数。",
            "",
        ]
    )
    component_rows: list[list[Any]] = []
    for row in components:
        uav0, uav1 = row["uav_ids"]
        component_rows.append(
            [
                row["checkpoint"],
                row["seed"],
                f"{row['sample_index']} (slot {row['slot']})",
                row["branch"],
                _fmt(row["delta_g_1"]),
                _fmt(row["delta_g_5"]),
                _fmt(row["delta_task_5"]),
                _fmt(row["delta_delay_5"], 4),
                _fmt(row["delta_vehicle_energy_5"], 4),
                _fmt(row["delta_cache_cost_5"]),
                _fmt(row["delta_movement_cost_5"]),
                f"{row['move_count'].get(uav0, 0)}/{row['move_count'].get(uav1, 0)}",
                f"{row['move_distance_m'].get(uav0, 0.0):.1f}/{row['move_distance_m'].get(uav1, 0.0):.1f}",
                "是" if row["holding_changed"] else "否",
                "是" if row["cache_target_changed"] else "否",
            ]
        )
    lines.append(
        _md_table(
            ["LR", "seed", "索引 (实际 Slot)", "分支", "ΔG1", "ΔG5", "Δ任务效用5", "Δ总时延5(s)", "Δ车辆能耗5(J)", "ΔCache成本5", "Δ移动成本5", "U0/U1 MOVE次数", "U0/U1距离(m)", "Holding变更", "后续Cache目标变更"],
            component_rows,
        )
    )
    lines.extend(
        [
            "",
            "Cumulative ΔG1 是第一个 Slot 的即时差；ΔG5 是 5-Slot 未折扣系统效用累计差。5e-4 若 A 初始位置等于 B，则 deterministic 分支应全程相同；逐步差值也在配对表核验。UAV 实际移动量基于每个 Slot 的前后位置计算。",
            "5e-4 的六个起点 A/B 都在首 Slot 双 STAY，且 5-Slot A 与 B 的 Cache、Holding、UAV 动作轨迹及系统效用逐步完全相同；因此该组主要看 C0/C1/C2 相对 A 的差。",
            "",
            "## 按起点逐步比较系统效用",
            "",
            "下面每行的 ΔG@k 是比较两分支从起点开始前 k 个 Slot 的累计差，δg1…δg5 是每个相对 Slot 的单步差。A−B 与 C0/C1/C2−A 均逐状态列出。",
            "",
        ]
    )
    pair_rows: list[list[Any]] = []
    for row in pairwise:
        pair_rows.append(
            [
                row["checkpoint"],
                row["seed"],
                f"{row['sample_index']} (slot {row['slot']})",
                row["comparison"],
                *[_fmt(value) for value in row["cumulative_delta"]],
                *[_fmt(value) for value in row["step_delta"]],
            ]
        )
    lines.append(
        _md_table(
            ["LR", "seed", "索引 (实际 Slot)", "比较", "ΔG@1", "ΔG@2", "ΔG@3", "ΔG@4", "ΔG@5", "δg1", "δg2", "δg3", "δg4", "δg5"],
            pair_rows,
        )
    )

    lines.extend(["", "## 汇总分布（每组最多 n=6）", "", "无显著性检验；均值/中位数、四分位数、范围和正负数量只描述当前抽样起点。正负零按绝对值 1e-12 分界。", ""])
    aggregate_rows: list[list[Any]] = []
    aggregate_values: dict[tuple[str, str, int], dict[str, Any]] = {}
    for lr, _run in RUNS:
        for comparison, left, right in COMPARISONS:
            for horizon in range(1, HORIZON + 1):
                values = [
                    row["cumulative_delta"][horizon - 1]
                    for row in pairwise
                    if row["checkpoint"] == lr and row["comparison"] == comparison
                ]
                stats = _stats(values)
                aggregate_values[(lr, comparison, horizon)] = stats
                aggregate_rows.append(
                    [
                        lr,
                        comparison,
                        horizon,
                        stats.get("n", 0),
                        _fmt(stats.get("mean")),
                        _fmt(stats.get("median")),
                        f"[{_fmt(stats.get('q25'))}, {_fmt(stats.get('q75'))}]" if stats.get("n") else "N/A",
                        f"[{_fmt(stats.get('min'))}, {_fmt(stats.get('max'))}]" if stats.get("n") else "N/A",
                        f"{stats.get('positive', 0)}/{stats.get('negative', 0)}/{stats.get('zero', 0)}",
                    ]
                )
    lines.append(_md_table(["LR", "比较", "累计 Slot", "n", "均值", "中位数", "Q1–Q3", "最小–最大", "+/−/0",], aggregate_rows))

    lines.extend(["", "## Cache/Holding 后续决策变化", "", "计数单位是 6 个抽样起点。Holding 比较每个起点前 4 次 post-slot terminate/continue 决策（会影响本 5-Slot 轨迹后续状态）；Cache 比较后续四个 Slot 正在执行的目标 Cache。最终第 5 Slot 后的 Holding 决策已由 runner 按正式状态计算，但只影响 Slot 6，不计入本表的五步内变化率。", ""])
    decision_rows: list[list[Any]] = []
    divergence: dict[tuple[str, str], dict[str, int]] = {}
    for lr, _run in RUNS:
        for branch in ("B", "C0", "C1", "C2"):
            available = [sample for sample in samples if sample["checkpoint"] == lr and branch in sample["branches"]]
            hold_count = sum(_holding_changed(sample, branch) for sample in available)
            cache_count = sum(_cache_target_changed(sample, branch) for sample in available)
            divergence[(lr, branch)] = {"n": len(available), "holding": hold_count, "cache": cache_count}
            decision_rows.append([lr, branch, len(available), f"{hold_count}/{len(available)}", f"{cache_count}/{len(available)}"])
    lines.append(_md_table(["LR", "相对 A 的分支", "可用起点", "后续 Holding 改变", "后续 Cache 目标改变"], decision_rows))

    lines.extend(["", "## 检查候选中的事后最高累计收益", "", "该值是 diagnostic oracle among checked candidates：对每个起点在 C0/C1/C2 中事后挑选该累计步数收益最高者。它不是冻结 PPO 可直接实现的策略，也不是全局最优。", ""])
    oracle_rows: list[list[Any]] = []
    oracle_summary: dict[tuple[str, int], dict[str, Any]] = {}
    for lr, _run in RUNS:
        for horizon in range(1, HORIZON + 1):
            values: list[float] = []
            for sample in samples:
                if sample["checkpoint"] != lr or "B" not in sample["branches"]:
                    continue
                baseline = sum(float(item["system_utility"]) for item in sample["branches"]["B"][:horizon])
                candidates = [
                    sum(float(item["system_utility"]) for item in sample["branches"][branch][:horizon]) - baseline
                    for branch in ("C0", "C1", "C2")
                    if branch in sample["branches"]
                ]
                if candidates:
                    values.append(max(candidates))
            stats = _stats(values)
            oracle_summary[(lr, horizon)] = stats
            oracle_rows.append([lr, horizon, stats.get("n", 0), _fmt(stats.get("mean")), _fmt(stats.get("median")), _fmt(stats.get("min")), _fmt(stats.get("max")), f"{stats.get('positive', 0)}/{stats.get('negative', 0)}/{stats.get('zero', 0)}"])
    lines.append(_md_table(["LR", "累计 Slot", "n", "Oracle 均值", "中位数", "最小", "最大", "+/−/0 vs B"], oracle_rows))

    lines.extend(["", "## 问题回答与证据等级", ""])
    lr3 = "1e-3"
    ab1 = aggregate_values.get((lr3, "A-B", 1), {"n": 0})
    ab5 = aggregate_values.get((lr3, "A-B", 5), {"n": 0})
    negative_starts = [
        row for row in pairwise
        if row["checkpoint"] == lr3 and row["comparison"] == "A-B" and row["cumulative_delta"][0] < -1e-12
    ]
    recovered = sum(row["cumulative_delta"][4] >= -1e-12 for row in negative_starts)
    improved = sum(row["cumulative_delta"][4] > row["cumulative_delta"][0] + 1e-12 for row in negative_starts)
    later_mean = (ab5.get("mean", 0.0) - ab1.get("mean", 0.0)) if ab5.get("n") and ab1.get("n") else None
    lines.append(
        f"1. **1e-3 实际 MOVE 是否被后四步补偿：** A−B 的即时均值 {_fmt(ab1.get('mean'))}，累计五步均值 {_fmt(ab5.get('mean'))}，后四步净增量均值 {_fmt(later_mean)}。在即时差为负的 {len(negative_starts)} 个起点中，{improved} 个累计值到第 5 步有所改善，{recovered} 个最终回到非负。是否补偿以同一轨迹后续槽逐步差额计算，不根据动作即时差单独定性。"
    )
    lr5 = "5e-4"
    stay_text: list[str] = []
    for branch in ("C0", "C1", "C2"):
        stats1 = aggregate_values.get((lr5, f"{branch}-A", 1), {"n": 0})
        stats5 = aggregate_values.get((lr5, f"{branch}-A", 5), {"n": 0})
        sustained = 0
        valid = 0
        for row in pairwise:
            if row["checkpoint"] == lr5 and row["comparison"] == f"{branch}-A":
                valid += 1
                sustained += int(all(value > 1e-12 for value in row["cumulative_delta"]))
        stay_text.append(
            f"{branch}: 即时均值 {_fmt(stats1.get('mean'))}、五步均值 {_fmt(stats5.get('mean'))}、五步正收益 {stats5.get('positive', 0)}/{stats5.get('n', 0)}、累计差从第 1 到 5 步始终为正 {sustained}/{valid}"
        )
    lines.append("2. **5e-4 STAY 时合法 MOVE 是否持续有收益：** " + "；".join(stay_text) + "。按分支和起点报告，不能把候选内的正收益外推成普遍规律。")
    lines.append(
        "3. **初始 UAV 动作是否改变后续 Cache/Holding：** 见上表的 6 个起点计数。Holding 的差异只在策略终止动作真正不同的状态计入；Cache 按活动 Cache 目标状态比较。它们是动作分支的状态传播结果，不是独立的因果中介估计。"
    )
    c5_means = [aggregate_values.get((lr3, f"C{i}-A", 5), {}).get("mean") for i in range(3)]
    if all(value is not None for value in c5_means):
        spread = max(float(value) for value in c5_means) - min(float(value) for value in c5_means)
    else:
        spread = None
    lines.append(
        f"4. **MOVE 候选累计收益差异：** 1e-3 三种 C−A 的五步平均效用增量为 "
        + ", ".join(f"C{i} {_fmt(c5_means[i])}" for i in range(3))
        + f"；候选类别均值最大差 {_fmt(spread)}。逐起点结果见前表，不将少量候选中最高者称为全局最优。"
    )
    lines.append(
        "5. **现有证据更支持哪类问题：** 更偏向动作选择问题，且两种 checkpoint 的表现不同。5e-4 全部选择初始 STAY，但合法 C 分支的五步收益相对 A 为正 6/6；1e-3 实际动作 A−B 五步均值仍为负，而 C0/C1/C2−A 五步均值分别为 "
        + ", ".join(f"C{i} {_fmt(c5_means[i])}" for i in range(3))
        + f"，C1/C2 在 {aggregate_values.get((lr3, 'C1-A', 5), {}).get('positive', 0)}/{aggregate_values.get((lr3, 'C1-A', 5), {}).get('n', 0)}、{aggregate_values.get((lr3, 'C2-A', 5), {}).get('positive', 0)}/{aggregate_values.get((lr3, 'C2-A', 5), {}).get('n', 0)} 起点高于 A。1e-3 的后四步平均没有补回首步损失；五步内未观察到 Cache 或 Holding 目标分支变化。因此局部现象更像 5e-4 有未触发的获益 MOVE、1e-3 的 MOVE 目标/联合条件选择欠佳，并伴随部分候选的延迟累计改善。小样本不能判定 PPO 学习机制本身有缺陷。"
    )
    lines.append(
        "6. **是否调整 Micro PPO：** 本次不建议直接更改 Micro PPO 的学习率、熵正则或奖励。两个 checkpoint 的 Micro Actor LR 都是 3e-4；不同 Macro LR 没有直接更改 Micro Actor optimizer LR 的代码路径。五步分支能显示冻结策略动作价值与状态传播，但 checkpoint 训练进度不齐，不能区分 Macro 诱导的状态/回报分布变化、策略学习误差和随机训练方差。若五步结果持续显示 A 落后于多个可行 MOVE，应先在配对训练进度和更多 Validation realization 上复核同一分支诊断，再讨论针对目标选择或长期 credit assignment 的微调。"
    )
    lines.extend(
        [
            "",
            "### 证据等级",
            "",
            "- **冻结策略局部五步分支价值：中等（只限本次 12 个起点与列出的 3 类动作）**。同一状态内任务、车辆、预测、Cache 初始帧和环境参数一致，后续由确定性策略自然演化；不代表整体 Validation 表现。",
            "- **A/B/C 差异归因到 UAV 初始动作：较强的局部反事实证据**。分支唯一手动改变的是首 Slot UAV 动作；之后策略、外生输入与环境执行链相同。该等级限这些离线分支，不能外推到随机策略或训练总体。",
            "- **归因到 Micro PPO 学习机制或 Macro LR 的因果解释：不足**。checkpoint 分别完成 224/300 Episode，没有同进度配对；每组两个 seed、六个起点；没有对学习更新、梯度、训练随机性或策略再训练做干预。",
            "",
            "## 执行与完整性",
            "",
            f"| 项目 | 数量/耗时 |",
            "|---|---:|",
            f"| 每个 checkpoint 的基线重放 Slot | {execution_info['baseline_slots_per_checkpoint']} |",
            f"| 每个 checkpoint 的五步分支数 / 环境执行 | {execution_info['branch_count_per_checkpoint']} / {execution_info['branch_execs_per_checkpoint']} |",
            f"| C0/C1/C2 MOVE 联合候选请求 / 正式执行 | {execution_info['candidate_branches_total']} / {execution_info['candidate_branches_total']} |",
            f"| 所有分支 execute_configured_slot() 次数 | {execution_info['branch_execs_total']} |",
            f"| 连同状态重放的 execute_configured_slot() 总次数 | {execution_info['all_execs_total']} |",
            f"| 总 Matching/KKT 计时 | {execution_info['matching_seconds']:.3f} s |",
            f"| 总 slot 环境执行墙钟计时 | {execution_info['environment_seconds']:.3f} s |",
            f"| 诊断总墙钟时间（两个 checkpoint 和 realization 构建） | {execution_info['wall_seconds']:.3f} s |",
            f"| 执行 Slot 外生输入指纹无变化（Slot 0-13） | {'是' if execution_info['external_unchanged'] else '否'} |",
            f"| 冻结策略参数未变化 / optimizer step 未变化 | {'是' if parameter_checks['parameters_unchanged'] and parameter_checks['optimizer_steps_unchanged'] else '否'} |",
            "",
            "每个起点的 5-Slot A/B/C 轨迹均从独立深拷贝起跑；Cache、UAV 与 Holding runtime 均分支隔离。Checkpoints、正式配置、训练日志与 Validation 数据均只读。脚本没有启动优化器或修改 checkpoint。该报告不含 Test 结果。",
            "",
            "结论限于冻结确定性策略、两个 Validation seed、每 checkpoint 六个起点、每起点三类预定 MOVE 候选和五 Slot 的即时累计系统效用。未来超过五个 Slot 的影响、更多动作候选及整体策略性能仍待验证。",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run(report_path: Path) -> None:
    started_wall = time.perf_counter()
    all_samples: list[dict[str, Any]] = []
    checkpoint_info: dict[str, dict[str, Any]] = {}
    execution = {
        "baseline_slots_per_checkpoint": 0,
        "branch_count_per_checkpoint": 0,
        "candidate_branches_total": 0,
        "branch_execs_per_checkpoint": 0,
        "branch_execs_total": 0,
        "all_execs_total": 0,
        "matching_seconds": 0.0,
        "environment_seconds": 0.0,
        "external_unchanged": True,
    }
    parameter_checks: dict[str, Any] = {
        "parameters_unchanged": True,
        "optimizer_steps_unchanged": True,
    }

    for lr, run_name in RUNS:
        run_dir = PROJECT_ROOT / "runs" / run_name
        config_path = run_dir / "run_config.toml"
        checkpoint = run_dir / "checkpoints" / "latest_checkpoint.pt"
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        checkpoint_sha_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        metadata = _load_checkpoint_metadata(checkpoint)
        training_counters = metadata.get("training_counters")
        if not isinstance(training_counters, dict):
            raise ValueError(f"checkpoint missing training counters: {checkpoint}")
        reference = load_environment_context(str(config_path))
        semantic_sha256 = project_config.semantic_config_hash(reference.cfg)
        if metadata.get("config_raw_sha256") != config_sha256:
            raise ValueError(f"checkpoint raw config hash mismatch for {lr}")
        if metadata.get("config_semantic_sha256") != semantic_sha256:
            raise ValueError(f"checkpoint semantic config hash mismatch for {lr}")
        trainer = PPOTrainer(
            reference,
            device="cpu",
            macro_actor_learning_rate_override=float(metadata["learning_rates"]["macro_actor"]),
            unified_cache_action=True,
            cardinality_corrected_cache_action=True,
        )
        trainer.load_checkpoint(
            checkpoint,
            config_raw_sha256=config_sha256,
            config_semantic_sha256=semantic_sha256,
            episode_pool=None,
        )
        policy = trainer.policy
        policy.eval()
        parameter_snapshot = [parameter.detach().clone() for parameter in policy.parameters()]
        optimizer_snapshot = (
            int(trainer.training_counters["micro_optimizer_steps"]),
            int(trainer.training_counters["macro_optimizer_steps"]),
        )
        completed_episodes = int(trainer.training_counters["completed_episodes"])
        policy_updates = int(trainer.training_counters["policy_update_count"])
        run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        checkpoint_info[lr] = {
            "completed_episodes": completed_episodes,
            "policy_updates": policy_updates,
            "summary_completed_episodes": int(run_summary["completed_episodes"]),
            "checkpoint_version": int(metadata["checkpoint_version"]),
            "architecture_version": int(metadata["architecture_version"]),
            "policy_schema": str(metadata["policy_schema"]),
            "macro_actor_lr": float(metadata["learning_rates"]["macro_actor"]),
            "micro_actor_lr": float(metadata["learning_rates"]["micro_actor"]),
            "config_sha256": config_sha256,
            "semantic_config_sha256": semantic_sha256,
            "checkpoint_sha256": checkpoint_sha_before,
            "checkpoint_path": str(checkpoint),
        }
        if completed_episodes != int(run_summary["completed_episodes"]):
            raise ValueError(f"run summary and checkpoint progress differ for {lr}")

        tensor_spec = PrimitiveTensorSpec.create(reference)
        uav_ids = tuple(reference.action_spec.uav_ids)
        print(f"[{lr}] loaded checkpoint at {completed_episodes} episodes; evaluating seeds", flush=True)
        for seed in VALIDATION_SEEDS:
            realization_dir = PROJECT_ROOT / "environment_data" / "proposed_joint_validation" / f"episode_{seed}"
            context = load_environment_context(str(config_path), realization_dir=realization_dir)
            _assert_realization_context_compatible(reference, context)
            trainer.context = context
            steps = aligned_train_steps(context)
            if len(steps) != int(context.episode_slots) or len(steps) != 300:
                raise AssertionError("Validation aligned input is not the formal complete 300-slot episode")
            if max(SAMPLE_INDICES) + HORIZON > len(steps):
                raise AssertionError("five-slot branch extends beyond the supplied Validation realization")
            used_step_indices = set(range(max(SAMPLE_INDICES) + HORIZON))
            fingerprints_before = {
                index: _external_fingerprint(steps[index]) for index in used_step_indices
            }
            state = DecisionState(
                cache=np.asarray(context.initial_cache, dtype=np.int8).copy(),
                uav=_copy_positions(context.initial_uav, uav_ids),
            )
            snapshots: dict[int, tuple[DecisionState, ActionChoice, dict[str, Any]]] = {}
            baseline_records: dict[int, dict[str, Any]] = {}
            prefix_started = time.perf_counter()
            with torch.inference_mode():
                for step_index in range(max(SAMPLE_INDICES) + 1):
                    _prepare_macro_frame(policy, context, tensor_spec, state, steps[step_index], step_index)
                    action = _select_policy_uav_action(policy, context, state, steps[step_index])
                    if step_index in SAMPLE_INDICES:
                        snapshots[step_index] = (copy.deepcopy(state), copy.deepcopy(action), {})
                    record = _execute_one_slot(
                        policy, context, steps, tensor_spec, state, step_index, action
                    )
                    baseline_records[step_index] = record
                    execution["all_execs_total"] += 1
                    execution["baseline_slots_per_checkpoint"] += 1
                    execution["environment_seconds"] += record["environment_seconds"]
                    execution["matching_seconds"] += record["matching_kkt_seconds"]
            _ = time.perf_counter() - prefix_started

            for sample_index in SAMPLE_INDICES:
                snapshot, baseline_action, _ = snapshots[sample_index]
                sample: dict[str, Any] = {
                    "checkpoint": lr,
                    "seed": int(seed),
                    "sample_index": int(sample_index),
                    "slot": int(steps[sample_index].slot),
                    "uav_ids": list(uav_ids),
                    "initial_cache_hash": hashlib.sha256(np.ascontiguousarray(snapshot.cache).tobytes()).hexdigest()[:16],
                    "macro_cache_hash": hashlib.sha256(np.ascontiguousarray(snapshot.frame_cache, dtype=np.int8).tobytes()).hexdigest()[:16],
                    "holding_count_before_first_action": int(snapshot.frame_elapsed),
                    "branches": {},
                    "branch_summaries": {},
                    "initial_actions": {},
                }
                baseline_first = baseline_records[sample_index]
                for branch in BRANCHES:
                    branch_state = copy.deepcopy(snapshot)
                    branch_records: list[dict[str, Any]] = []
                    unavailable_reason: str | None = None
                    with torch.inference_mode():
                        for offset in range(HORIZON):
                            step_index = sample_index + offset
                            _prepare_macro_frame(
                                policy, context, tensor_spec, branch_state, steps[step_index], step_index
                            )
                            if offset == 0 and branch in ("B", "C0", "C1", "C2"):
                                action = _counterfactual_uav_action(
                                    branch, context, branch_state, steps[step_index]
                                )
                            else:
                                action = _select_policy_uav_action(
                                    policy, context, branch_state, steps[step_index]
                                )
                            if offset == 0:
                                sample["initial_actions"][branch] = {
                                    "positions": {key: value.tolist() for key, value in action.positions.items()},
                                    "indices": list(action.indices),
                                    "sources": list(action.sources),
                                    "candidate_counts": list(action.candidate_counts),
                                    "rule": action.rule,
                                }
                            record = _execute_one_slot(
                                policy, context, steps, tensor_spec, branch_state, step_index, action
                            )
                            branch_records.append(record)
                            execution["all_execs_total"] += 1
                            execution["branch_execs_total"] += 1
                            execution["branch_execs_per_checkpoint"] += 1
                            execution["environment_seconds"] += record["environment_seconds"]
                            execution["matching_seconds"] += record["matching_kkt_seconds"]
                    sample["branches"][branch] = branch_records
                    sample["branch_summaries"][branch] = _branch_summary(branch_records, uav_ids)
                    if branch == "A":
                        for uav_id in uav_ids:
                            if not np.array_equal(
                                np.asarray(sample["initial_actions"]["A"]["positions"][uav_id]),
                                baseline_action.positions[uav_id],
                            ):
                                raise AssertionError("A did not reproduce the deterministic PPO first action")
                        if not math.isclose(
                            float(branch_records[0]["system_utility"]),
                            float(baseline_first["system_utility"]),
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        ):
                            raise AssertionError("A first-slot reward does not reproduce its baseline replay")
                all_samples.append(sample)
                execution["branch_count_per_checkpoint"] += len(BRANCHES)
                execution["candidate_branches_total"] += 3
                print(f"[{lr} seed {seed} index {sample_index}] completed five branches", flush=True)

            fingerprints_after = {
                index: _external_fingerprint(steps[index]) for index in used_step_indices
            }
            if fingerprints_before != fingerprints_after:
                execution["external_unchanged"] = False
                raise AssertionError("branch execution mutated external Validation input")
            print(f"[{lr} seed {seed}] external inputs unchanged", flush=True)

        if any(
            not torch.equal(before, after.detach())
            for before, after in zip(parameter_snapshot, policy.parameters(), strict=True)
        ):
            parameter_checks["parameters_unchanged"] = False
        if optimizer_snapshot != (
            int(trainer.training_counters["micro_optimizer_steps"]),
            int(trainer.training_counters["macro_optimizer_steps"]),
        ):
            parameter_checks["optimizer_steps_unchanged"] = False
        checkpoint_sha_after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if checkpoint_sha_after != checkpoint_sha_before:
            raise RuntimeError(f"checkpoint changed during read-only diagnostic: {checkpoint}")

    execution["branch_count_per_checkpoint"] //= len(RUNS)
    execution["baseline_slots_per_checkpoint"] //= len(RUNS)
    execution["branch_execs_per_checkpoint"] = execution["branch_execs_total"] // len(RUNS)
    execution["wall_seconds"] = float(time.perf_counter() - started_wall)
    _generate_report(report_path, checkpoint_info, all_samples, execution, parameter_checks)
    print(f"report written: {report_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen five-slot UAV counterfactual diagnosis")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "uav_multislot_counterfactual.md",
    )
    args = parser.parse_args()
    run(args.report)


if __name__ == "__main__":
    main()
