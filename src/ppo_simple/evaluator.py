"""Evaluate frozen Adaptive Dual-Timescale Joint MAPPO checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - 仅静态导入环境
    torch = None  # type: ignore[assignment]

from common import project_config
from decision import slot_environment
from scenario.realization import REALIZATION_ARTIFACT_FILENAMES
from .interface import (
    PER_SLOT_ANALYTIC_OBSERVATION_SCHEMA,
    PER_SLOT_ANALYTIC_TASK_MODEL,
    PROPOSED_CONTROL_MODE,
    PROPOSED_POLICY_SCHEMA,
    _assert_realization_context_compatible,
    aligned_train_steps,
    load_environment_context,
)
from .reporting import resolve_checkpoint_reference
from .rollout import collect_dual_timescale_episode
from .primitive_control import CACHE_ECONOMIC_FEATURE_DIM
from .policy import (
    AdaptiveDualTimescaleActorCritic,
    AdaptiveMultiNodeFactorizedActorCritic,
    AdaptiveSparseCoordinatedCacheActorCritic,
)
from .trainer import (
    ALGORITHM_ID,
    CHECKPOINT_METHOD,
    CHECKPOINT_VERSION,
    MULTI_NODE_CHECKPOINT_VERSION,
    SPARSE_COORDINATED_CHECKPOINT_VERSION,
    PPOTrainer,
    SUPPORTED_MACRO_ACTOR_LEARNING_RATES,
)


PROPOSED_CONFIG_PATH = "configs/proposed/system_config.toml"
DEFAULT_ENVIRONMENT_COUNT = 20
DEFAULT_POLICY_REPEATS = 1
EPISODE_METRIC_NAMES = (
    "total_return",
    "net_utility",
    "total_q",
    "task_utility",
    "total_cache_cost",
    "total_movement_cost",
    "cache_change_rate",
    "uav_movement_rate",
    "uav_stay_rate",
    "uav_move_rate",
    "uav0_move_candidate_rate",
    "uav1_move_candidate_rate",
    "cloud_task_count",
    "edge_task_count",
    "cloud_offload_rate",
    "edge_offload_rate",
    "total_tasks",
    "mean_task_delay_s",
    "mean_vehicle_energy_j",
    "uav0_group_stay_probability_mean",
    "uav0_group_move_probability_mean",
    "uav0_group_entropy_mean",
    "uav1_group_stay_probability_mean",
    "uav1_group_move_probability_mean",
    "uav1_group_entropy_mean",
    "micro_group_entropy_mean",
    "uav0_group_low_entropy_fraction",
    "uav1_group_low_entropy_fraction",
    "uav0_group_probability_over_90_fraction",
    "uav1_group_probability_over_90_fraction",
    "uav_joint_stay_stay_rate",
    "uav_joint_stay_move_rate",
    "uav_joint_move_stay_rate",
    "uav_joint_move_move_rate",
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("正式双时间尺度 evaluation 需要 PyTorch。")


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve(strict=False)


def _ensure_fresh_output(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"evaluation output is not a directory: {path}")
    if path.is_dir() and any(path.iterdir()):
        raise FileExistsError(
            "evaluation output already contains artifacts; choose a fresh --run-dir"
        )
    path.mkdir(parents=True, exist_ok=True)


def _episode_directories(pool: Path, seed_start: int, count: int) -> list[Path]:
    if not pool.is_dir():
        raise FileNotFoundError(f"episode pool missing: {pool}")
    if int(seed_start) >= 0:
        paths = [pool / f"episode_{int(seed_start) + index}" for index in range(int(count))]
        missing = [path for path in paths if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"evaluation episode missing: {missing[0]}")
        return paths
    children = [
        child for child in pool.iterdir()
        if child.is_dir()
        and any((child / name).is_file() for name in REALIZATION_ARTIFACT_FILENAMES.values())
    ]
    children.sort(
        key=lambda child: (
            int(match.group(1)) if (match := re.search(r"(\d+)$", child.name)) else math.inf,
            child.name,
        )
    )
    if not children and any((pool / name).is_file() for name in REALIZATION_ARTIFACT_FILENAMES.values()):
        children = [pool]
    if len(children) < int(count):
        raise FileNotFoundError("evaluation episode pool 不足所需完整 realization 数量。")
    return children[: int(count)]


def _validate_formal_validation_pool(
    project_root: Path,
    pool_path: Path,
    seed_start: int,
    episode_count: int,
) -> dict[str, Any]:
    """Require the frozen evaluator to use the official, disjoint Validation split."""
    training_pool = (project_root / "environment_data/proposed_joint_training").resolve()
    validation_pool = (project_root / "environment_data/proposed_joint_validation").resolve()
    if pool_path.resolve() != validation_pool:
        raise ValueError(
            "ppo-eval requires the official Validation Pool; Training or ad-hoc pools "
            "cannot be used for policy evaluation."
        )

    training_manifest_path = training_pool / "formal_pool_manifest.json"
    validation_manifest_path = validation_pool / "formal_pool_manifest.json"
    training_manifest_bytes = training_manifest_path.read_bytes()
    validation_manifest_bytes = validation_manifest_path.read_bytes()
    training_manifest = json.loads(training_manifest_bytes.decode("utf-8"))
    validation_manifest = json.loads(validation_manifest_bytes.decode("utf-8"))
    if training_manifest.get("split") != "train":
        raise ValueError("Formal Training Pool manifest has an invalid split label.")
    if validation_manifest.get("split") != "validation":
        raise ValueError("Formal Validation Pool manifest has an invalid split label.")

    def manifest_seed_ids(manifest: Mapping[str, Any], name: str) -> set[int]:
        episode_hashes = manifest.get("episode_manifest_sha256")
        if not isinstance(episode_hashes, Mapping):
            raise ValueError(f"{name} manifest lacks episode manifest IDs.")
        seeds = {int(value) for value in episode_hashes}
        seed_start = int(manifest.get("seed_start", -1))
        seed_end = int(manifest.get("seed_end", -1))
        episode_count = int(manifest.get("episode_count", -1))
        if (
            episode_count <= 0
            or len(seeds) != episode_count
            or seeds != set(range(seed_start, seed_end + 1))
        ):
            raise ValueError(f"{name} manifest has incomplete or inconsistent seed coverage.")
        return seeds

    training_seeds = manifest_seed_ids(training_manifest, "Training Pool")
    validation_seeds = manifest_seed_ids(validation_manifest, "Validation Pool")
    if training_seeds & validation_seeds:
        raise ValueError("Training and Validation Episode IDs/seeds overlap.")

    training_sources = training_manifest.get("source_episode_manifest_sha256")
    validation_sources = validation_manifest.get("source_episode_manifest_sha256")
    if not isinstance(training_sources, Mapping) or not isinstance(validation_sources, Mapping):
        raise ValueError("Formal split manifests lack source Episode provenance maps.")
    if (
        set(map(int, training_sources)) != training_seeds
        or set(map(int, validation_sources)) != validation_seeds
    ):
        raise ValueError("Formal split source provenance does not cover every declared Episode.")

    def source_manifest_hashes(values: Mapping[str, Any], name: str) -> set[str]:
        hashes = [str(value) for value in values.values()]
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
            for value in hashes
        ):
            raise ValueError(f"{name} source Episode manifest hashes are invalid.")
        return {value.lower() for value in hashes}

    shared_sources = source_manifest_hashes(
        training_sources, "Training Pool"
    ) & source_manifest_hashes(validation_sources, "Validation Pool")
    if shared_sources:
        raise ValueError("Training and Validation share source Episode manifests.")
    training_split_hash = training_manifest.get("data_split_manifest_sha256")
    validation_split_hash = validation_manifest.get("data_split_manifest_sha256")
    if not training_split_hash or training_split_hash != validation_split_hash:
        raise ValueError("Training and Validation do not reference the same frozen split manifest.")

    selected_seed_list = (
        list(range(int(seed_start), int(seed_start) + int(episode_count)))
        if int(seed_start) >= 0
        else sorted(validation_seeds)[: int(episode_count)]
    )
    selected_seeds = set(selected_seed_list)
    if (
        len(selected_seeds) != int(episode_count)
        or not selected_seeds.issubset(validation_seeds)
    ):
        raise ValueError("Selected evaluation Episodes are not unique members of frozen Validation.")
    if selected_seeds & training_seeds:
        raise ValueError("Selected evaluation Episodes overlap the Training Pool.")

    return {
        "split": "validation",
        "manifest_sha256": hashlib.sha256(validation_manifest_bytes).hexdigest(),
        "training_manifest_sha256": hashlib.sha256(training_manifest_bytes).hexdigest(),
        "data_split_manifest_sha256": str(validation_split_hash),
        "episode_seeds": selected_seed_list,
    }


def _load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    _require_torch()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    if (
        path.name == "best_train_return_checkpoint.pt"
        or payload.get("checkpoint_role")
        == "TRAINING_DIAGNOSTIC_ONLY_NOT_FINAL_SELECTION"
        or payload.get("artifact_role")
        == "TRAINING_DIAGNOSTIC_ONLY_NOT_FINAL_SELECTION"
    ):
        raise ValueError(
            "training-return diagnostic checkpoint is not a formal evaluation input"
        )
    policy_schema = payload.get("policy_schema")
    if policy_schema == AdaptiveDualTimescaleActorCritic.architecture_id:
        expected_policy = AdaptiveDualTimescaleActorCritic
        expected_version = CHECKPOINT_VERSION
        expected_unified_cache_action = True
        expected_cache_action_mode = project_config.CACHE_ACTION_MODE_SINGLE_NODE
        expected_cardinality_correction = True
    elif policy_schema == AdaptiveMultiNodeFactorizedActorCritic.architecture_id:
        expected_policy = AdaptiveMultiNodeFactorizedActorCritic
        expected_version = MULTI_NODE_CHECKPOINT_VERSION
        expected_unified_cache_action = False
        expected_cache_action_mode = project_config.CACHE_ACTION_MODE_MULTI_NODE_FACTORIZED
        expected_cardinality_correction = True
    elif policy_schema == AdaptiveSparseCoordinatedCacheActorCritic.architecture_id:
        expected_policy = AdaptiveSparseCoordinatedCacheActorCritic
        expected_version = SPARSE_COORDINATED_CHECKPOINT_VERSION
        expected_unified_cache_action = False
        expected_cache_action_mode = project_config.CACHE_ACTION_MODE_SPARSE_COORDINATED
        expected_cardinality_correction = False
    else:
        raise ValueError("checkpoint policy schema is not supported by PROPOSED")
    if int(payload.get("checkpoint_version", -1)) != expected_version:
        raise ValueError("checkpoint format is not supported by its policy schema")
    expected_architecture = expected_policy.architecture_version
    expected_schema = expected_policy.architecture_id
    if int(payload.get("architecture_version", -1)) != expected_architecture:
        raise ValueError(
            f"checkpoint architecture_version is not {expected_architecture}"
        )
    contract = payload.get("config_contract")
    if payload.get("algorithm_id") != ALGORITHM_ID:
        raise ValueError("checkpoint algorithm_id mismatch")
    if payload.get("method") != CHECKPOINT_METHOD:
        raise ValueError("checkpoint method mismatch")
    if payload.get("policy_schema") != expected_schema:
        raise ValueError("checkpoint policy_schema mismatch")
    if not isinstance(contract, dict):
        raise ValueError("checkpoint PROPOSED configuration contract is missing")
    if (
        payload.get("task_execution_model") != PER_SLOT_ANALYTIC_TASK_MODEL
        or contract.get("task_execution_model") != PER_SLOT_ANALYTIC_TASK_MODEL
        or contract.get("observation_schema") != PER_SLOT_ANALYTIC_OBSERVATION_SCHEMA
    ):
        raise ValueError("checkpoint task/observation model is not the current formal model")
    policy_update_total = contract.get("policy_update_total")
    if (
        isinstance(policy_update_total, bool)
        or not isinstance(policy_update_total, int)
        or policy_update_total <= 0
    ):
        raise ValueError(
            "checkpoint config_contract policy_update_total must be a positive integer"
        )
    if contract.get("algorithm_id") != ALGORITHM_ID:
        raise ValueError("checkpoint config_contract algorithm_id mismatch")
    if contract.get("policy_schema") != expected_schema:
        raise ValueError("checkpoint config_contract policy_schema mismatch")
    if payload.get("unified_cache_action") is not expected_unified_cache_action:
        raise ValueError("checkpoint Cache candidate action mode does not match its policy schema")
    if payload.get("cardinality_corrected_cache_action") is not expected_cardinality_correction:
        raise ValueError("checkpoint Cache probability contract does not match its schema")
    sparse_probability_contract_valid = (
        expected_cache_action_mode != project_config.CACHE_ACTION_MODE_SPARSE_COORDINATED
        or (
            contract.get("max_cache_updates_per_frame") == 1
            and contract.get("cache_group_probability_semantics")
            == "masked categorical over ALL_KEEP and each node with at least one legal UPDATE"
            and contract.get("cache_conditional_probability_semantics")
            == "masked categorical over legal UPDATE targets for the selected node only"
        )
    )
    if (
        contract.get("architecture_version") != expected_architecture
        or contract.get("unified_cache_action") is not expected_unified_cache_action
        or contract.get("cache_action_mode") != expected_cache_action_mode
        or contract.get("cardinality_corrected_cache_action") is not expected_cardinality_correction
        or not isinstance(contract.get("macro_cache_feature_dim"), int)
        or contract.get("macro_cache_feature_dim")
        != 2 * int(contract.get("cache_node_count", -1)) * int(contract.get("service_count", -1))
        + 1
        + CACHE_ECONOMIC_FEATURE_DIM
        or (
            expected_cache_action_mode == project_config.CACHE_ACTION_MODE_SPARSE_COORDINATED
            and contract.get("max_cache_updates_per_frame") != 1
        )
        or not sparse_probability_contract_valid
    ):
        raise ValueError("checkpoint PROPOSED configuration contract mismatch")
    expected_rates = {
        "micro_actor": 3.0e-4,
        "macro_critic": 1.0e-3,
        "micro_critic": 3.0e-3,
        "long_value_critic": 1.0e-3,
    }
    contract_rates = contract.get("learning_rates")
    payload_rates = payload.get("learning_rates")
    macro_actor_rate = (
        contract_rates.get("macro_actor")
        if isinstance(contract_rates, dict)
        else None
    )
    payload_macro_actor_rate = (
        payload_rates.get("macro_actor")
        if isinstance(payload_rates, dict)
        else None
    )
    supported_macro_actor_rate = next(
        (
            value
            for value in SUPPORTED_MACRO_ACTOR_LEARNING_RATES
            if isinstance(macro_actor_rate, (int, float))
            and math.isfinite(float(macro_actor_rate))
            and math.isclose(
                float(macro_actor_rate), value, rel_tol=0.0, abs_tol=1.0e-15
            )
        ),
        None,
    )
    if (
        payload.get("macro_actor_learning_rate_schedule") != "constant"
        or payload.get("optimizer_learning_rate_schedule") != "constant"
        or contract.get("macro_actor_learning_rate_schedule") != "constant"
        or contract.get("optimizer_learning_rate_schedule") != "constant"
        or not isinstance(contract_rates, dict)
        or not isinstance(payload_rates, dict)
        or supported_macro_actor_rate is None
        or payload_macro_actor_rate != supported_macro_actor_rate
        or any(contract_rates.get(key) != value for key, value in expected_rates.items())
        or any(
            payload_rates.get(key) != value
            for key, value in {
                "micro_actor": 3.0e-4,
                "macro_critic": 1.0e-3,
                "micro_critic": 3.0e-3,
                "long_value": 1.0e-3,
            }.items()
        )
        or contract.get("macro_actor_learning_rate_initial")
        != supported_macro_actor_rate
        or contract.get("macro_actor_learning_rate_final")
        != supported_macro_actor_rate
    ):
        raise ValueError("checkpoint fixed-learning-rate contract mismatch")
    return payload


def _seed_policy_sampling(seed: int) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _episode_summary(collection: Any, episode_seed: int, repeat_index: int) -> dict[str, Any]:
    micro = collection.micro.transitions
    slots = max(len(micro), 1)
    if not micro:
        raise ValueError("formal evaluation collection is empty")
    uav_count = int(len(micro[0].actions))
    if uav_count <= 0:
        raise ValueError("formal evaluation requires at least one UAV head")
    joint_action_counts: dict[str, int] = {}
    for transition in micro:
        if len(transition.actions) != uav_count:
            raise ValueError("Validation action head count changed within an episode")
        joint_key = "|".join("MOVE" if int(action) != 0 else "STAY" for action in transition.actions)
        joint_action_counts[joint_key] = joint_action_counts.get(joint_key, 0) + 1
    if uav_count == 2:
        joint_action_counts = {
            "stay_stay": int(joint_action_counts.get("STAY|STAY", 0)),
            "stay_move": int(joint_action_counts.get("STAY|MOVE", 0)),
            "move_stay": int(joint_action_counts.get("MOVE|STAY", 0)),
            "move_move": int(joint_action_counts.get("MOVE|MOVE", 0)),
        }
    joint_action_rates = {
        key: value / slots for key, value in joint_action_counts.items()
    }
    task_count = int(sum(item.task_count for item in micro))
    total_delay = float(sum(item.total_task_delay_s for item in micro))
    total_energy = float(sum(item.total_vehicle_energy_j for item in micro))
    execution_counts = {"SV": 0, "RSU": 0, "UAV": 0, "CLOUD": 0}
    for transition in micro:
        slot_counts = transition.execution_target_counts
        for execution_class in execution_counts:
            execution_counts[execution_class] += int(slot_counts.get(execution_class, 0))
        if sum(int(slot_counts.get(name, 0)) for name in execution_counts) != int(
            transition.task_count
        ):
            raise ValueError("Validation execution counts do not match the task count.")
    cloud_task_count = execution_counts["CLOUD"]
    edge_task_count = sum(execution_counts[name] for name in ("SV", "RSU", "UAV"))
    if cloud_task_count + edge_task_count != task_count:
        raise ValueError("Validation Cloud/Edge counts do not sum to the task count.")
    task_utility = float(sum(item.actual_task_utility for item in micro))
    uav_metrics = list(collection.summary["uav_metrics"])
    uav0_move_candidate_rate = float(collection.summary["uav0_move_rate"])
    uav1_move_candidate_rate = (
        None
        if collection.summary.get("uav1_move_rate") is None
        else float(collection.summary["uav1_move_rate"])
    )
    movement_count = sum(int(item.uav_movement_count) for item in micro)
    total_return = float(sum(item.reward for item in micro))
    total_cache_cost = float(sum(item.cache_cost for item in micro))
    total_movement_cost = float(sum(item.movement_cost for item in micro))
    net_utility = task_utility - total_cache_cost - total_movement_cost
    if not math.isclose(total_return, net_utility, rel_tol=1.0e-12, abs_tol=1.0e-8):
        raise ValueError("frozen evaluation reward does not equal task utility less system costs")
    joint_action_rates = {key: value / slots for key, value in joint_action_counts.items()}
    if uav_count == 2:
        legacy_joint_rates = {
            "uav_joint_stay_stay_rate": joint_action_rates.get("stay_stay", 0.0),
            "uav_joint_stay_move_rate": joint_action_rates.get("stay_move", 0.0),
            "uav_joint_move_stay_rate": joint_action_rates.get("move_stay", 0.0),
            "uav_joint_move_move_rate": joint_action_rates.get("move_move", 0.0),
        }
    else:
        legacy_joint_rates = {
            "uav_joint_stay_stay_rate": None,
            "uav_joint_stay_move_rate": None,
            "uav_joint_move_stay_rate": None,
            "uav_joint_move_move_rate": None,
        }
    record = {
        "episode_seed": int(episode_seed),
        "repeat_index": int(repeat_index),
        "total_return": total_return,
        "net_utility": net_utility,
        "total_q": task_utility,
        "task_utility": task_utility,
        "total_cache_cost": total_cache_cost,
        "total_movement_cost": total_movement_cost,
        "window_slot_count": int(collection.summary.get("window_slot_count", slots)),
        "window_truncated": bool(collection.summary.get("window_truncated", False)),
        "episode_terminal_observed": bool(
            collection.summary.get("episode_terminal_observed", True)
        ),
        "cache_change_rate": float(
            sum(item.actual_cache_update_count for item in micro) / slots
        ),
        "uav_movement_rate": float(
            movement_count / (float(uav_count) * slots)
        ),
        "uav_stay_rate": float(
            1.0 - movement_count / (float(uav_count) * slots)
        ),
        "uav_move_rate": float(
            movement_count / (float(uav_count) * slots)
        ),
        "uav_count": uav_count,
        "uav_metrics": uav_metrics,
        "uav0_move_candidate_rate": uav0_move_candidate_rate,
        "uav1_move_candidate_rate": uav1_move_candidate_rate,
        "cloud_task_count": cloud_task_count,
        "edge_task_count": edge_task_count,
        "cloud_offload_rate": None if task_count == 0 else cloud_task_count / task_count,
        "edge_offload_rate": None if task_count == 0 else edge_task_count / task_count,
        "joint_action_counts": joint_action_counts,
        "joint_action_rates": joint_action_rates,
        **legacy_joint_rates,
        "total_tasks": task_count,
        "mean_task_delay_s": None if task_count == 0 else total_delay / task_count,
        "mean_vehicle_energy_j": None if task_count == 0 else total_energy / task_count,
        "macro_frame_count": int(len(collection.macro)),
        "adaptive_holding_duration_mean": float(
            collection.summary["adaptive_holding_duration_mean"]
        ),
        "adaptive_holding_duration_std": float(
            collection.summary["adaptive_holding_duration_std"]
        ),
        "holding_duration_distribution": dict(
            collection.summary["holding_duration_distribution"]
        ),
        "termination_sample_count": int(collection.summary["termination_sample_count"]),
        "termination_eligible_count": int(collection.summary["termination_eligible_count"]),
        "termination_continue_count": int(collection.summary["termination_continue_count"]),
        "termination_stop_count": int(collection.summary["termination_stop_count"]),
        "termination_continue_rate": float(collection.summary["termination_continue_rate"]),
        "termination_stop_rate": float(collection.summary["termination_stop_rate"]),
        "termination_entropy_mean": float(collection.summary["termination_entropy_mean"]),
        "termination_probability_mean": float(
            collection.summary["termination_probability_mean"]
        ),
        "termination_probability_by_elapsed": dict(
            collection.summary["termination_probability_by_elapsed"]
        ),
        "termination_decision_diagnostics": list(
            collection.summary["termination_decision_diagnostics"]
        ),
        "forced_termination_max_holding_count": int(
            collection.summary["forced_termination_max_holding_count"]
        ),
        "terminal_frame_count": int(collection.summary["terminal_frame_count"]),
        "keep_rate": float(collection.summary["keep_rate"]),
        "actual_cache_update_rate": float(collection.summary["actual_cache_update_rate"]),
        "uav0_stay_rate": float(collection.summary["uav0_stay_rate"]),
        "uav1_stay_rate": (
            None
            if collection.summary.get("uav1_stay_rate") is None
            else float(collection.summary["uav1_stay_rate"])
        ),
        "uav0_move_rate": float(collection.summary["uav0_move_rate"]),
        "uav1_move_rate": uav1_move_candidate_rate,
        "matching_call_count": int(collection.summary["matching_call_count"]),
        "kkt_call_count": int(collection.summary["kkt_call_count"]),
        "episode_runtime_seconds": float(collection.summary["episode_runtime_seconds"]),
    }
    for metric_name in (
        "uav0_group_stay_probability_mean",
        "uav0_group_move_probability_mean",
        "uav0_group_entropy_mean",
        "uav1_group_stay_probability_mean",
        "uav1_group_move_probability_mean",
        "uav1_group_entropy_mean",
        "micro_group_entropy_mean",
        "uav0_group_low_entropy_fraction",
        "uav1_group_low_entropy_fraction",
        "uav0_group_probability_over_90_fraction",
        "uav1_group_probability_over_90_fraction",
    ):
        metric_value = collection.summary.get(metric_name)
        record[metric_name] = None if metric_value is None else float(metric_value)
    deterministic_actions = collection.summary.get("deterministic_action_selection")
    if deterministic_actions is not None:
        record["deterministic_action_selection"] = deterministic_actions
    deterministic_cache_protocol = collection.summary.get("deterministic_cache_protocol")
    if deterministic_cache_protocol is not None:
        record["deterministic_cache_protocol"] = deterministic_cache_protocol
        record["deterministic_cache_protocol_checks"] = collection.summary.get(
            "deterministic_cache_protocol_checks", {}
        )
    return record


def _evaluate_keep_stay_window(
    context: Any,
    aligned_steps: list[Any],
    max_slots: int,
) -> dict[str, Any]:
    """在同一组真实 Training Slot 上执行固定 KEEP+STAY 配对参考。"""

    if max_slots <= 0 or max_slots > len(aligned_steps):
        raise ValueError("KEEP+STAY 窗口必须落在已对齐的 Training Slot 内。")
    cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
    uav = {
        str(key): np.asarray(value, dtype=np.float64).copy()
        for key, value in context.initial_uav.items()
    }
    task_utility = 0.0
    cache_cost = 0.0
    movement_cost = 0.0
    net_utility = 0.0
    kkt_call_count = 0
    started = time.perf_counter()
    for step in aligned_steps[:max_slots]:
        result = slot_environment.execute_configured_slot(
            cache,
            uav,
            cache,
            uav,
            step.scenario_slice,
            step.actual_tasks,
            step.actual_vehicle_positions,
            context.service_catalog,
            context.cfg,
            context.cache_nodes,
            context.cache_cfg,
            context.flight_boundary,
            slot_id=int(step.slot),
            time_s=float(step.time_s),
        )
        if not result.step_feasible or result.system_utility is None:
            raise RuntimeError(
                "KEEP+STAY evaluation produced an infeasible formal Slot: "
                f"origin={step.origin_index}, reason={result.infeasible_reason}"
            )
        if not np.array_equal(np.asarray(result.next_cache_state), cache):
            raise AssertionError("KEEP+STAY changed the Cache configuration.")
        if any(
            not np.array_equal(np.asarray(result.next_uav_positions[key]), uav[key])
            for key in uav
        ):
            raise AssertionError("KEEP+STAY changed a UAV position.")
        expected = (
            float(result.actual_task_utility)
            - float(result.cache_cost)
            - float(result.movement_cost)
        )
        if not math.isclose(
            float(result.system_utility), expected, rel_tol=1.0e-12, abs_tol=1.0e-9
        ):
            raise ValueError("KEEP+STAY system utility does not match the formal reward contract")
        task_utility += float(result.actual_task_utility)
        cache_cost += float(result.cache_cost)
        movement_cost += float(result.movement_cost)
        net_utility += float(result.system_utility)
        kkt_call_count += int(
            result.matching_summary.get("KKT_SOLVER_CALL_COUNT", 0)
        )
    formula_net = task_utility - cache_cost - movement_cost
    if not math.isclose(net_utility, formula_net, rel_tol=1.0e-12, abs_tol=1.0e-8):
        raise ValueError("KEEP+STAY cumulative reward does not match task utility less costs")
    return {
        "policy": "KEEP+STAY",
        "slot_count": int(max_slots),
        "net_utility": float(net_utility),
        "task_utility": float(task_utility),
        "cache_cost": float(cache_cost),
        "movement_cost": float(movement_cost),
        "cache_update_count": 0,
        "uav_movement_count": 0,
        "kkt_call_count": int(kkt_call_count),
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _metric_statistics(
    summaries: Iterable[Mapping[str, Any]], metric_names: Iterable[str]
) -> dict[str, dict[str, float]]:
    records = list(summaries)
    if len(records) < 2:
        raise ValueError("episode statistics need at least two realizations")
    from scipy.stats import t as student_t

    result: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.asarray([float(item[name]) for item in records], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"metric {name} contains non-finite values")
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        se = std / float(np.sqrt(len(values)))
        critical = float(student_t.ppf(0.975, len(values) - 1))
        result[str(name)] = {
            "mean": mean,
            "std": std,
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "se": se,
            "ci95_low": float(mean - critical * se),
            "ci95_high": float(mean + critical * se),
        }
    return result


def _mean_environment_summary(
    episode_seed: int, repeat_summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    numeric_names = (
        "total_return",
        "net_utility",
        "total_q",
        "task_utility",
        "total_cache_cost",
        "total_movement_cost",
        "cache_change_rate",
        "uav_movement_rate",
        "uav_stay_rate",
        "uav_move_rate",
        "uav0_move_candidate_rate",
        "uav1_move_candidate_rate",
        "cloud_task_count",
        "edge_task_count",
        "cloud_offload_rate",
        "edge_offload_rate",
        "total_tasks",
        "mean_task_delay_s",
        "mean_vehicle_energy_j",
        "adaptive_holding_duration_mean",
        "adaptive_holding_duration_std",
        "termination_sample_count",
        "termination_eligible_count",
        "termination_continue_count",
        "termination_stop_count",
        "termination_continue_rate",
        "termination_stop_rate",
        "termination_entropy_mean",
        "termination_probability_mean",
        "forced_termination_max_holding_count",
        "terminal_frame_count",
        "keep_rate",
        "actual_cache_update_rate",
        "uav0_stay_rate",
        "uav1_stay_rate",
        "uav0_move_rate",
        "uav1_move_rate",
        "uav_joint_stay_stay_rate",
        "uav_joint_stay_move_rate",
        "uav_joint_move_stay_rate",
        "uav_joint_move_move_rate",
        "uav0_group_stay_probability_mean",
        "uav0_group_move_probability_mean",
        "uav0_group_entropy_mean",
        "uav1_group_stay_probability_mean",
        "uav1_group_move_probability_mean",
        "uav1_group_entropy_mean",
        "micro_group_entropy_mean",
        "uav0_group_low_entropy_fraction",
        "uav1_group_low_entropy_fraction",
        "uav0_group_probability_over_90_fraction",
        "uav1_group_probability_over_90_fraction",
        "matching_call_count",
        "kkt_call_count",
    )
    means: dict[str, Any] = {"episode_seed": int(episode_seed)}
    for key in numeric_names:
        values = [float(item[key]) for item in repeat_summaries if item.get(key) is not None]
        means[key] = None if not values else float(np.mean(values))
    means["repeat_count"] = len(repeat_summaries)
    uav_count = int(repeat_summaries[0].get("uav_count", 0)) if repeat_summaries else 0
    if any(int(item.get("uav_count", -1)) != uav_count for item in repeat_summaries):
        raise ValueError("evaluation repeats have inconsistent UAV head counts")
    means["uav_count"] = uav_count
    means["uav_metrics"] = [
        {
            "uav_id": str(repeat_summaries[0]["uav_metrics"][head]["uav_id"]),
            **{
                field: float(
                    np.mean(
                        [
                            float(item["uav_metrics"][head][field])
                            for item in repeat_summaries
                        ]
                    )
                )
                for field in (
                    "stay_rate",
                    "move_rate",
                    "stay_count",
                    "move_count",
                    "entropy",
                    "group_stay_probability_mean",
                    "group_move_probability_mean",
                    "group_entropy_mean",
                    "group_low_entropy_fraction",
                    "group_probability_over_90_fraction",
                    "mean_valid_candidates",
                )
            },
            "action_pattern": "repeat_mean",
        }
        for head in range(uav_count)
    ]
    means["holding_duration_distribution"] = {
        key: float(
            np.mean(
                [
                    float(item["holding_duration_distribution"].get(key, 0))
                    for item in repeat_summaries
                ]
            )
        )
        for key in sorted(
            {
                duration
                for item in repeat_summaries
                for duration in item["holding_duration_distribution"]
            }
        )
    }
    means["termination_probability_by_elapsed"] = {}
    for elapsed in range(1, 5):
        rows = [
            item["termination_probability_by_elapsed"].get(str(elapsed), {})
            for item in repeat_summaries
        ]
        observed = [row for row in rows if int(row.get("sample_count", 0)) > 0]
        sample_count = sum(int(row["sample_count"]) for row in observed)
        means["termination_probability_by_elapsed"][str(elapsed)] = {
            "sample_count": sample_count,
            "mean_termination_probability": (
                float(
                    sum(
                        float(row["mean_termination_probability"])
                        * int(row["sample_count"])
                        for row in observed
                    )
                    / sample_count
                )
                if sample_count
                else None
            ),
            "selected_termination_rate": (
                float(
                    sum(
                        float(row["selected_termination_rate"])
                        * int(row["sample_count"])
                        for row in observed
                    )
                    / sample_count
                )
                if sample_count
                else None
            ),
        }
    means["termination_decision_diagnostics"] = [
        diagnostic
        for item in repeat_summaries
        for diagnostic in item["termination_decision_diagnostics"]
    ]
    return means


def evaluate_pool(
    *,
    checkpoint_path: str | Path,
    run_dir: str | Path,
    config_path: str | Path,
    episode_pool: str | Path,
    seed_start: int,
    episodes: int = DEFAULT_ENVIRONMENT_COUNT,
    repeats: int = DEFAULT_POLICY_REPEATS,
    device: str = "cpu",
    policy_action_mode: str = "deterministic",
    policy_sampling_seed: int | None = None,
    max_slots: int | None = None,
) -> dict[str, Any]:
    _require_torch()
    if int(episodes) <= 0 or int(repeats) <= 0:
        raise ValueError("episodes 和 repeats 必须为正整数。")
    if policy_action_mode not in {"deterministic", "stochastic"}:
        raise ValueError("policy_action_mode must be deterministic or stochastic")
    if max_slots is not None and (int(max_slots) <= 0 or int(episodes) != 1 or int(repeats) != 1):
        raise ValueError("短窗配对评价要求正 max_slots、单个 Episode 和一次策略运行。")
    config_file = Path(config_path).expanduser().resolve(strict=True)
    checkpoint = resolve_checkpoint_reference(Path(checkpoint_path)).resolve(strict=True)
    checkpoint_metadata = _load_checkpoint_metadata(checkpoint)
    reference_context = load_environment_context(str(config_file))
    checkpoint_cache_mode = {
        AdaptiveDualTimescaleActorCritic.architecture_id:
            project_config.CACHE_ACTION_MODE_SINGLE_NODE,
        AdaptiveMultiNodeFactorizedActorCritic.architecture_id:
            project_config.CACHE_ACTION_MODE_MULTI_NODE_FACTORIZED,
        AdaptiveSparseCoordinatedCacheActorCritic.architecture_id:
            project_config.CACHE_ACTION_MODE_SPARSE_COORDINATED,
    }.get(str(checkpoint_metadata["policy_schema"]))
    if checkpoint_cache_mode is None or checkpoint_cache_mode != reference_context.cache_action_mode:
        raise ValueError(
            "checkpoint architecture Schema is incompatible with the selected formal Cache action mode"
        )
    project_root = reference_context.root
    pool_path = _resolve_path(project_root, episode_pool)
    output_path = _resolve_path(project_root, run_dir)
    validation_pool_contract = _validate_formal_validation_pool(
        project_root,
        pool_path,
        int(seed_start),
        int(episodes),
    )
    episode_paths = _episode_directories(pool_path, int(seed_start), int(episodes))
    selected_path_seeds = [
        int(re.fullmatch(r"episode_(\d+)", path.name).group(1))
        for path in episode_paths
        if re.fullmatch(r"episode_(\d+)", path.name)
    ]
    if selected_path_seeds != validation_pool_contract["episode_seeds"]:
        raise ValueError("Resolved Validation Episode paths do not match the frozen manifest seed selection.")
    _ensure_fresh_output(output_path)
    raw_config_sha256 = project_config.raw_config_sha256(config_file)
    semantic_sha256 = project_config.semantic_config_hash(reference_context.cfg)
    summaries: list[dict[str, Any]] = []
    environment_means: list[dict[str, Any]] = []
    trainer: PPOTrainer | None = None
    parameter_snapshot: list[Any] | None = None
    optimizer_step_snapshot: tuple[tuple[str, int], ...] | None = None
    paired_baselines: list[dict[str, Any]] = []

    for environment_index, realization_path in enumerate(episode_paths):
        context = load_environment_context(str(config_file), realization_dir=realization_path)
        _assert_realization_context_compatible(reference_context, context)
        if trainer is None:
            trainer = PPOTrainer(
                context,
                device=device,
                macro_actor_learning_rate_override=float(
                    checkpoint_metadata["learning_rates"]["macro_actor"]
                ),
                policy_update_total=int(
                    checkpoint_metadata["config_contract"]["policy_update_total"]
                ),
                unified_cache_action=(
                    context.cache_action_mode
                    == project_config.CACHE_ACTION_MODE_SINGLE_NODE
                ),
                cardinality_corrected_cache_action=(
                    context.cache_action_mode
                    != project_config.CACHE_ACTION_MODE_SPARSE_COORDINATED
                ),
            )
            trainer.load_checkpoint(
                checkpoint,
                config_raw_sha256=raw_config_sha256,
                config_semantic_sha256=semantic_sha256,
                episode_pool=None,
            )
            trainer.policy.eval()
            parameter_snapshot = [
                parameter.detach().clone() for parameter in trainer.policy.parameters()
            ]
            optimizer_step_snapshot = tuple(
                sorted(
                    (str(key), int(value))
                    for key, value in trainer.training_counters.items()
                )
            )
        trainer.context = context
        repeat_records: list[dict[str, Any]] = []
        episode_seed = (
            int(seed_start) + environment_index
            if int(seed_start) >= 0
            else int(re.search(r"(\d+)$", realization_path.name).group(1))
            if re.search(r"(\d+)$", realization_path.name)
            else environment_index
        )
        aligned_steps = aligned_train_steps(context)
        if max_slots is not None and int(max_slots) > len(aligned_steps):
            raise ValueError("max_slots exceeds the selected Training realization length.")
        for repeat_index in range(int(repeats)):
            action_seed = (
                None
                if policy_sampling_seed is None
                else int(policy_sampling_seed)
                + environment_index * int(repeats)
                + repeat_index
            )
            if action_seed is not None:
                _seed_policy_sampling(action_seed)
            with torch.inference_mode():
                collection = collect_dual_timescale_episode(
                    trainer.policy,
                    context,
                    aligned_steps,
                    deterministic=policy_action_mode == "deterministic",
                    episode_seed=episode_seed,
                    max_slots=max_slots,
                )
            summary = _episode_summary(collection, episode_seed, repeat_index)
            summary["policy_sampling_seed"] = action_seed
            summary["policy_action_mode"] = policy_action_mode
            repeat_records.append(summary)
            summaries.append(summary)
        keep_stay_summary = (
            _evaluate_keep_stay_window(context, aligned_steps, int(max_slots))
            if max_slots is not None
            else None
        )
        if keep_stay_summary is not None:
            keep_stay_summary["episode_seed"] = int(episode_seed)
            paired_baselines.append(keep_stay_summary)
        destination = output_path / f"episode_{episode_seed}"
        destination.mkdir(parents=True, exist_ok=False)
        with (destination / "proposed_repeat_summaries.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(repeat_records, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        if keep_stay_summary is not None:
            policy_summary = repeat_records[0]
            paired_record = {
                "episode_seed": int(episode_seed),
                "policy_action_mode": policy_action_mode,
                "window_slots": int(max_slots),
                "mappo": policy_summary,
                "keep_stay": keep_stay_summary,
                "net_utility_difference_mappo_minus_keep_stay": float(
                    policy_summary["net_utility"] - keep_stay_summary["net_utility"]
                ),
                "interpretation": "paired short-window chain verification; not a performance conclusion",
            }
            with (destination / "paired_keep_stay_window.json").open(
                "w", encoding="utf-8", newline="\n"
            ) as stream:
                json.dump(paired_record, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write("\n")
        environment_means.append(_mean_environment_summary(episode_seed, repeat_records))

    assert trainer is not None and parameter_snapshot is not None
    if any(
        not torch.equal(before, after.detach())
        for before, after in zip(parameter_snapshot, trainer.policy.parameters(), strict=True)
    ):
        raise RuntimeError("evaluation modified policy parameters")
    if optimizer_step_snapshot != tuple(
        sorted((str(key), int(value)) for key, value in trainer.training_counters.items())
    ):
        raise RuntimeError("evaluation performed a PPO optimizer step")
    if any(parameter.grad is not None for parameter in trainer.policy.parameters()):
        raise RuntimeError("frozen evaluation populated policy gradients")

    available = tuple(
        name for name in EPISODE_METRIC_NAMES
        if all(item.get(name) is not None for item in environment_means)
    )
    if len(environment_means) == 1:
        aggregate_statistics = {
            name: {
                "mean": float(environment_means[0][name]),
                "std": None,
                "median": float(environment_means[0][name]),
                "min": float(environment_means[0][name]),
                "max": float(environment_means[0][name]),
                "se": None,
                "ci95_low": None,
                "ci95_high": None,
            }
            for name in available
        }
    else:
        aggregate_statistics = _metric_statistics(environment_means, available)
    aggregate = {
        "experiment_type": (
            "paired_frozen_policy_window_evaluation"
            if max_slots is not None
            else "macro_cache_smdp_micro_uav_episode_pool_evaluation"
        ),
        "config_path": str(config_file),
        "config_sha256": raw_config_sha256,
        "config_semantic_sha256": semantic_sha256,
        "checkpoint_path": str(checkpoint),
        "checkpoint_version": int(checkpoint_metadata["checkpoint_version"]),
        "task_execution_model": str(checkpoint_metadata["task_execution_model"]),
        "checkpoint_policy_schema": checkpoint_metadata["policy_schema"],
        "policy_schema": reference_context.policy_schema,
        "configuration_control_mode": PROPOSED_CONTROL_MODE,
        "episode_pool": str(pool_path),
        "evaluation_pool_contract": validation_pool_contract,
        "episode_count": len(environment_means),
        "seed_start": int(seed_start),
        "episodes": int(episodes),
        "repeats": int(repeats),
        "policy_action_mode": policy_action_mode,
        "policy_sampling_seed": policy_sampling_seed,
        "max_slots": None if max_slots is None else int(max_slots),
        "window_is_truncated": bool(max_slots is not None),
        "statistical_unit": "environment_realization",
        "repeat_aggregation": "within_environment_mean_then_across_environment_statistics",
        "aggregate_statistics": aggregate_statistics,
        "environment_means": environment_means,
        "repeat_summaries": summaries,
        "paired_keep_stay_summaries": paired_baselines,
    }
    cache_protocols = [
        item["deterministic_cache_protocol"]
        for item in summaries
        if "deterministic_cache_protocol" in item
    ]
    if cache_protocols:
        if any(protocol != cache_protocols[0] for protocol in cache_protocols[1:]):
            raise RuntimeError("evaluation repeats used inconsistent deterministic Cache protocols")
        aggregate["deterministic_cache_protocol"] = cache_protocols[0]
    with (output_path / "proposed_episode_pool_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(aggregate, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="评估冻结的 Adaptive Dual-Timescale Joint MAPPO 策略。"
    )
    parser.add_argument("--config", default=PROPOSED_CONFIG_PATH)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episode-pool", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_ENVIRONMENT_COUNT)
    parser.add_argument("--repeats", type=int, default=DEFAULT_POLICY_REPEATS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy-action-mode", choices=("deterministic", "stochastic"), default="deterministic")
    parser.add_argument("--policy-sampling-seed", type=int, default=None)
    parser.add_argument("--max-slots", type=int, default=None, help="短窗配对模式；单 Episode 与一次策略运行，并自动比较 KEEP+STAY。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate_pool(
        checkpoint_path=args.checkpoint,
        run_dir=args.run_dir,
        config_path=args.config,
        episode_pool=args.episode_pool,
        seed_start=args.seed_start,
        episodes=args.episodes,
        repeats=args.repeats,
        device=args.device,
        policy_action_mode=args.policy_action_mode,
        policy_sampling_seed=args.policy_sampling_seed,
        max_slots=args.max_slots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
