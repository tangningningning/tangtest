#!/usr/bin/env python3
"""Run the bounded prediction-driven global Cache solver on Training data.

The solver chooses a Cache target from the predictor artifact before any
realized task rows for that slot are handed to the formal Matching+KKT executor.
UAV positions are held at their current values (STAY).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts" / "training",
):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import system_model
from decision import predictive_global_cache, slot_environment
from predictive_global_cache_data import training_context_and_steps


PROFILE_PATH = PROJECT_ROOT / "configs" / "research" / "predictive_global_cache_value_v1.json"
MACRO_INTERVAL_SLOTS = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile() -> tuple[dict[str, Any], str]:
    value = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if value.get("schema") != "predictive_global_cache_experiment_v1":
        raise ValueError("unrecognized predictive Cache solver profile")
    return value, _sha256(PROFILE_PATH)


def run_training_window(
    *,
    model_checkpoint: Path | None,
    episode_seed: int,
    slots: int,
    output_path: Path,
    policy: str = "global_value",
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if policy not in {"global_value", "keep_stay"}:
        raise ValueError(f"unsupported Training-window policy: {policy!r}")
    profile, profile_sha256 = _profile()
    context, steps, pool_info = training_context_and_steps(
        project_root.resolve(), int(episode_seed), int(slots)
    )
    checkpoint: Path | None = None
    model: predictive_global_cache.LoadedValueModel | None = None
    if policy == "global_value":
        if model_checkpoint is None:
            raise ValueError("global_value policy requires a fitted value checkpoint")
        checkpoint = model_checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"global Cache value checkpoint is missing: {checkpoint}")
        model = predictive_global_cache.load_value_model(checkpoint, device="cpu")
        if model.metadata.get("config_sha256") != pool_info["runtime_config_sha256"]:
            raise ValueError("value checkpoint formal system config does not match the Training Pool")
        if model.metadata.get("pool_manifest_sha256") != pool_info["manifest_sha256"]:
            raise ValueError("value checkpoint Training Pool manifest does not match the current pool")
        if model.metadata.get("research_profile_sha256") != profile_sha256:
            raise ValueError("value checkpoint was trained under a different research profile")
        if model.metadata.get("service_count") != int(context.service_catalog["num_services"]):
            raise ValueError("value checkpoint service count does not match this environment")
        if not model.metadata.get("calibration_metrics", {}).get("gate_passed", False):
            raise ValueError("calibration decision gate failed; refusing solver evaluation")
        training_episode_ids = {
            int(value) for value in model.metadata.get("dataset_episode_ids", [])
        }
        if int(episode_seed) in training_episode_ids:
            raise ValueError("solver Training Episode was used for fitting or calibration")
    elif model_checkpoint is not None:
        raise ValueError("KEEP+STAY reference does not accept a value checkpoint")

    search_cfg = profile["joint_search_budget"]
    search_budget = predictive_global_cache.SearchBudget(
        max_candidate_evaluations=int(search_cfg["max_candidate_evaluations"]),
        max_cache_updates=int(search_cfg["max_cache_updates"]),
        max_runtime_seconds=float(search_cfg["max_runtime_seconds"]),
        per_node_candidate_capacity=int(search_cfg["per_node_candidate_capacity"]),
    )
    cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
    uav = {
        str(key): np.asarray(value, dtype=np.float64).copy()
        for key, value in context.initial_uav.items()
    }
    initial_cache_hash = hashlib.sha256(np.ascontiguousarray(cache).tobytes()).hexdigest()
    records: list[dict[str, Any]] = []
    total_task_utility = 0.0
    total_cache_cost = 0.0
    total_uav_cost = 0.0
    total_net_utility = 0.0
    beta_ca = float(context.cfg["objective"]["beta_ca"])
    beta_q = float(context.cfg["objective"]["beta_q"])
    execution_seconds = 0.0
    search_seconds = 0.0
    cache_node_changes = 0

    for index, step in enumerate(steps):
        if index % MACRO_INTERVAL_SLOTS == 0:
            if policy == "global_value":
                assert model is not None
                started = time.perf_counter()
                target, search_metrics = predictive_global_cache.search_cache_configuration(
                    model,
                    cache,
                    step.scenario_slice,
                    step.prediction_horizon_mask,
                    uav,
                    context.service_catalog,
                    context.cache_nodes,
                    context.cache_cfg,
                    context.cfg,
                    context.flight_boundary,
                    budget=search_budget,
                )
                search_seconds += time.perf_counter() - started
            else:
                target = cache.copy()
                search_metrics = {
                    "algorithm_schema": predictive_global_cache.ALGORITHM_SCHEMA,
                    "strategy": "KEEP+STAY",
                    "candidate_evaluations": 0,
                    "accepted_cache_updates": 0,
                    "keep_is_legal_fallback": True,
                }
        else:
            target = cache.copy()
            search_metrics = None

        previous = cache.copy()
        changed_node_ids = [
            str(context.cache_nodes["node_ids"][node_index])
            for node_index in np.flatnonzero(np.any(target != previous, axis=1))
        ]
        started = time.perf_counter()
        # Realized Training tasks enter only after the predictor-only decision.
        result = slot_environment.execute_configured_slot(
            previous,
            uav,
            target,
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
        execution_seconds += time.perf_counter() - started
        if not result.step_feasible or result.system_utility is None:
            raise RuntimeError(
                f"formal Matching+KKT execution failed at slot {step.slot}: {result.infeasible_reason}"
            )
        values = (result.actual_task_utility, result.cache_cost, result.movement_cost, result.system_utility)
        if any(value is None or not math.isfinite(float(value)) for value in values):
            raise FloatingPointError(f"non-finite formal reward at slot {step.slot}")
        if result.beta_ca is None or result.beta_q is None:
            raise AssertionError("formal Slot result lacks system utility weights")
        decomposition = system_model.aggregate_system_utility(
            float(result.actual_task_utility),
            float(result.cache_cost),
            float(result.movement_cost),
            float(result.beta_ca),
            float(result.beta_q),
        )
        if not math.isclose(decomposition, float(result.system_utility), rel_tol=0.0, abs_tol=1.0e-7):
            raise ValueError(f"formal reward decomposition mismatch at slot {step.slot}")
        total_task_utility += float(result.actual_task_utility)
        total_cache_cost += float(result.cache_cost)
        total_uav_cost += float(result.movement_cost)
        total_net_utility += float(result.system_utility)
        cache_node_changes += len(changed_node_ids)
        cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
        uav = {
            str(key): np.asarray(value, dtype=np.float64).copy()
            for key, value in result.next_uav_positions.items()
        }
        records.append(
            {
                "slot": int(step.slot),
                "decision": search_metrics,
                "cache_changed_node_ids": changed_node_ids,
                "task_utility": float(result.actual_task_utility),
                "cache_cost": float(result.cache_cost),
                "uav_cost": float(result.movement_cost),
                "beta_ca": float(result.beta_ca),
                "beta_q": float(result.beta_q),
                "cache_cost_penalty": float(result.beta_ca * result.cache_cost),
                "uav_cost_penalty": float(result.beta_q * result.movement_cost),
                "net_utility": float(result.system_utility),
                "matching_feasible": bool(result.step_feasible),
            }
        )

    output = output_path.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite solver result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result_record = {
        "algorithm_schema": predictive_global_cache.ALGORITHM_SCHEMA,
        "search_schema": predictive_global_cache.SEARCH_SCHEMA,
        "strategy": policy,
        "split": "train",
        "episode_seed": int(episode_seed),
        "slot_count": int(len(steps)),
        "slot_start": int(steps[0].slot),
        "slot_end_inclusive": int(steps[-1].slot),
        "training_pool_manifest_sha256": pool_info["manifest_sha256"],
        "runtime_config_sha256": pool_info["runtime_config_sha256"],
        "research_profile_sha256": profile_sha256,
        "value_checkpoint_path": str(checkpoint) if checkpoint is not None else None,
        "value_checkpoint_sha256": _sha256(checkpoint) if checkpoint is not None else None,
        "label_schema": model.metadata["label_schema"] if model is not None else None,
        "label_horizon_slots": model.metadata["label_horizon_slots"] if model is not None else None,
        "label_continuation_policy": model.metadata["label_continuation_policy"] if model is not None else None,
        "training_label_episode_ids": model.metadata.get("dataset_episode_ids", []) if model is not None else [],
        "calibration_gate_passed": model.metadata.get("calibration_metrics", {}).get("gate_passed") if model is not None else None,
        "cache_task_utility": float(total_task_utility),
        "cache_configuration_cost": float(total_cache_cost),
        "uav_movement_cost": float(total_uav_cost),
        "beta_ca": beta_ca,
        "beta_q": beta_q,
        "cache_cost_penalty": float(beta_ca * total_cache_cost),
        "uav_movement_cost_penalty": float(beta_q * total_uav_cost),
        "net_utility": float(total_net_utility),
        "actual_cache_node_changes": int(cache_node_changes),
        "cache_change_rate_per_node_slot": float(
            cache_node_changes / max(len(steps) * cache.shape[0], 1)
        ),
        "uav_policy": "STAY at current positions",
        "cache_decision_interval_slots": MACRO_INTERVAL_SLOTS,
        "search_budget": search_cfg,
        "search_runtime_seconds": float(search_seconds),
        "matching_kkt_runtime_seconds": float(execution_seconds),
        "future_realized_tasks_used_for_decision": False,
        "validation_accessed": False,
        "test_accessed": False,
        "initial_cache_sha256": initial_cache_hash,
        "final_cache_sha256": hashlib.sha256(np.ascontiguousarray(cache).tobytes()).hexdigest(),
        "slots": records,
    }
    output.write_text(
        json.dumps(result_record, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "output_path": str(output),
        "output_sha256": _sha256(output),
        "slot_count": int(len(steps)),
        "net_utility": float(total_net_utility),
        "cache_cost": float(total_cache_cost),
        "cache_node_changes": int(cache_node_changes),
        "search_runtime_seconds": float(search_seconds),
        "matching_kkt_runtime_seconds": float(execution_seconds),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-checkpoint", type=Path)
    parser.add_argument("--policy", choices=("global_value", "keep_stay"), default="global_value")
    parser.add_argument("--episode-seed", type=int, default=20001)
    parser.add_argument("--slots", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.policy == "global_value" and args.model_checkpoint is None:
        parser.error("--model-checkpoint is required for --policy global_value")
    if args.policy == "keep_stay" and args.model_checkpoint is not None:
        parser.error("--model-checkpoint is not accepted for --policy keep_stay")
    result = run_training_window(
        model_checkpoint=args.model_checkpoint,
        episode_seed=args.episode_seed,
        slots=args.slots,
        output_path=args.output,
        policy=args.policy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

