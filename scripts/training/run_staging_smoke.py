"""Run a clearly isolated Online Joint DQN smoke run from completed staging Episodes.

This entry intentionally does not call or weaken the formal 300-Episode pool
gate. It snapshots individually verified completed Episodes and uses the same
environment adapter, Agent, Trainer, Replay Buffer, and update implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
GENERATION_ROOT = PROJECT_ROOT / "scripts" / "generation"
TRAINING_ROOT = PROJECT_ROOT / "scripts" / "training"
for import_path in (SRC_ROOT, GENERATION_ROOT, TRAINING_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import numpy as np
import torch

from common import project_config
from common.environment_interface import aligned_execution_steps, load_environment_context
from online_joint_dqn.agent import DoubleDQNAgent
from online_joint_dqn.candidate_evaluator import SlotInput, SystemModelContext
from online_joint_dqn.environment_adapter import ExclusiveDQNEnvironment
from online_joint_dqn.replay_buffer import ReplayBuffer
from online_joint_dqn.trainer import TrainerConfig, train_episode
from online_joint_dqn import run_training as formal_training
import generate_formal_vehicle_environment as formal_generator
import generate_episode_realization as episode_realization


STAGING_RELATIVE = Path("environment_data/.formal_vehicle_environment_staging")
CONFIG_RELATIVE = Path("configs/dqn/online_joint_dqn.toml")
EPISODE_COUNT = 20
ARRIVAL_HORIZON_SLOTS = 300
EXECUTION_HORIZON_SLOTS = 314
K_MAX_SLOTS = 15
REPLAY_CAPACITY = formal_training.DEFAULT_REPLAY_CAPACITY
TRAINING_SEED = formal_training.DEFAULT_SEED
FORMAL_EPSILON_DECAY_STEPS = (
    300 * EXECUTION_HORIZON_SLOTS
)

_STOP_REQUESTED = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    formal_training._write_json(path, value, atomic=True)


def _append_jsonl(stream: Any, value: Mapping[str, Any]) -> None:
    payload = formal_training._jsonable(value)
    # Matching leaves KKT residual fields as NaN when no KKT problem was
    # solved for a slot (for example, a Cloud-only assignment). This is an
    # unavailable diagnostic, not a training value; encode only that precise
    # no-solver case as JSON null. Keep strict rejection for every other
    # non-finite value so rewards, Q values, and learning metrics are never
    # silently masked.
    kkt_calls = payload.get("kkt_solver_calls") if isinstance(payload, dict) else None
    resource_summary = (
        payload.get("matching_resource_summary") if isinstance(payload, dict) else None
    )
    if isinstance(resource_summary, dict) and kkt_calls == 0:
        residual = resource_summary.get("kkt_resource_residual_max")
        if isinstance(residual, (int, float)) and not math.isfinite(float(residual)):
            resource_summary["kkt_resource_residual_max"] = None
            payload["nonfinite_diagnostic_fields_encoded_as_null"] = [
                "matching_resource_summary.kkt_resource_residual_max"
            ]
    stream.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    stream.flush()


def _status_snapshot(staging_root: Path) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    status_path = staging_root / "generation_status.json"
    plan_path = staging_root / "data_split_manifest.json"
    status_bytes = status_path.read_bytes()
    plan_bytes = plan_path.read_bytes()
    status = json.loads(status_bytes)
    plan = json.loads(plan_bytes)
    if status.get("status") not in {"generating", "complete"}:
        raise RuntimeError(f"staging generation status is not usable: {status.get('status')!r}")
    if status.get("failure"):
        raise RuntimeError(f"staging generation reports failure: {status['failure']!r}")
    if plan.get("dataset_status") not in {"generating", "complete"}:
        raise RuntimeError(f"staging plan status is not usable: {plan.get('dataset_status')!r}")
    return status, plan, status_bytes, plan_bytes


def _select_and_validate_inputs(staging_root: Path) -> dict[str, Any]:
    status, plan, status_bytes, plan_bytes = _status_snapshot(staging_root)
    completed_raw = status.get("completed_episode_seeds", {}).get("train")
    if not isinstance(completed_raw, list):
        raise RuntimeError("staging progress has no completed Train seed list")
    completed = [int(seed) for seed in completed_raw]
    if len(completed) != len(set(completed)):
        raise RuntimeError("staging progress contains duplicate completed Train seeds")
    current = status.get("current_episode") or {}
    current_seed = (
        int(current["seed"])
        if current.get("split") == "train" and current.get("seed") is not None
        else None
    )
    selected = [seed for seed in completed if seed != current_seed][:EPISODE_COUNT]
    expected = list(range(20001, 20001 + EPISODE_COUNT))
    if selected != expected:
        raise RuntimeError(
            "the earliest 20 completed Train Episodes are not the expected ordered seeds: "
            f"selected={selected!r}, expected={expected!r}"
        )
    train_plan = plan.get("train", {})
    if (
        int(train_plan.get("seed_start", -1)) != 20001
        or int(train_plan.get("seed_end", -1)) != 20300
        or int(train_plan.get("count", -1)) != 300
    ):
        raise RuntimeError("staging Train plan does not match the frozen 300-seed design")

    staged_config = staging_root / "generation_config.toml"
    if not staged_config.is_file():
        raise FileNotFoundError(f"staged generation config is missing: {staged_config}")
    formal_config_path = PROJECT_ROOT / CONFIG_RELATIVE
    if not formal_config_path.is_file():
        raise FileNotFoundError(f"formal DQN config is missing: {formal_config_path}")
    if _sha256_file(formal_config_path) != plan.get("config_sha256"):
        raise RuntimeError("formal config hash differs from the staging generation plan")
    config = episode_realization.load_and_validate_config(staged_config)
    protocol = project_config.get_episode_data_protocol(config)
    lifecycle_contract = project_config.get_cross_slot_task_parameters(config)
    if (
        int(protocol["arrival_horizon_slots"]) != ARRIVAL_HORIZON_SLOTS
        or int(protocol["execution_horizon_slots"]) != EXECUTION_HORIZON_SLOTS
        or int(lifecycle_contract["k_max_slots"]) != K_MAX_SLOTS
    ):
        raise RuntimeError(
            f"staging time contract mismatch: protocol={protocol!r}, "
            f"lifecycle={lifecycle_contract!r}"
        )
    if int(plan.get("time_contract", {}).get("execution_horizon_slots", -1)) != EXECUTION_HORIZON_SLOTS:
        raise RuntimeError("staging split manifest execution horizon is not 314 slots")

    service_catalog = project_config.resolve_path(
        PROJECT_ROOT,
        project_config.get_config_value(config, "paths.service_catalog"),
    )
    if not service_catalog.is_file():
        raise FileNotFoundError(f"staged service catalog is missing: {service_catalog}")
    service_catalog_sha256 = _sha256_file(service_catalog)
    if service_catalog_sha256 != plan.get("service_catalog_sha256"):
        raise RuntimeError("staged service catalog does not match the generation plan hash")

    episodes: list[dict[str, Any]] = []
    total_snapshot_bytes = service_catalog.stat().st_size
    for seed in selected:
        episode_dir = staging_root / "train" / f"episode_{seed}"
        episode_manifest_path = episode_dir / "episode_manifest.json"
        if not episode_manifest_path.is_file():
            raise FileNotFoundError(f"completed Episode manifest is missing: {episode_manifest_path}")
        manifest_sha256 = formal_generator._validate_episode(
            episode_dir,
            seed=seed,
            split="train",
            config_path=staged_config,
            origin_count=EXECUTION_HORIZON_SLOTS,
        )
        episode_manifest = json.loads(episode_manifest_path.read_text(encoding="utf-8"))
        if int(episode_manifest.get("episode_seed", -1)) != seed:
            raise RuntimeError(f"Episode manifest seed mismatch for {seed}")
        artifacts = episode_manifest.get("generated_artifacts")
        hashes = episode_manifest.get("generated_artifact_sha256")
        if not isinstance(artifacts, list) or not isinstance(hashes, dict):
            raise RuntimeError(f"Episode {seed} has no complete artifact hash inventory")
        artifact_hashes: dict[str, str] = {}
        artifact_bytes = episode_manifest_path.stat().st_size
        for name in artifacts:
            if Path(str(name)).name != str(name):
                raise RuntimeError(f"Episode {seed} has an unsafe artifact path: {name!r}")
            path = episode_dir / str(name)
            actual_hash = _sha256_file(path)
            if actual_hash != hashes.get(str(name)):
                raise RuntimeError(f"Episode {seed} artifact hash changed: {name}")
            artifact_hashes[str(name)] = actual_hash
            artifact_bytes += path.stat().st_size
        total_snapshot_bytes += artifact_bytes
        episodes.append(
            {
                "seed": seed,
                "source_episode_manifest_sha256": manifest_sha256,
                "artifact_count": len(artifacts),
                "artifact_bytes": artifact_bytes,
                "artifact_sha256": artifact_hashes,
            }
        )

    # Re-read only the small progress/plan manifests to ensure none of the
    # selected seeds became the actively written Episode while validation ran.
    final_status, final_plan, _, _ = _status_snapshot(staging_root)
    final_completed = {
        int(seed)
        for seed in final_status.get("completed_episode_seeds", {}).get("train", [])
    }
    final_current = final_status.get("current_episode") or {}
    final_current_seed = (
        int(final_current["seed"])
        if final_current.get("split") == "train" and final_current.get("seed") is not None
        else None
    )
    if not set(selected).issubset(final_completed) or final_current_seed in selected:
        raise RuntimeError("selected Episode completion state changed during validation")
    if final_plan.get("plan_fingerprint") != plan.get("plan_fingerprint"):
        raise RuntimeError("staging plan identity changed during input validation")

    return {
        "status": status,
        "plan": plan,
        "status_bytes": status_bytes,
        "plan_bytes": plan_bytes,
        "selected_seeds": selected,
        "episodes": episodes,
        "staged_config": staged_config,
        "service_catalog": service_catalog,
        "service_catalog_sha256": service_catalog_sha256,
        "total_snapshot_bytes": total_snapshot_bytes,
        "execution_horizon_slots": EXECUTION_HORIZON_SLOTS,
    }


def _new_run_directory(requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        requested = Path("runs") / (
            f"online_joint_dqn_staging_smoke_20ep_{stamp}_{uuid.uuid4().hex[:8]}"
        )
    target = requested if requested.is_absolute() else PROJECT_ROOT / requested
    target = target.resolve(strict=False)
    runs_root = (PROJECT_ROOT / "runs").resolve(strict=False)
    if target != runs_root and runs_root not in target.parents:
        raise ValueError("staging smoke output must be isolated under the project runs/ directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    return target


def _copy_snapshot(staging_root: Path, run_dir: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    snapshot_root = run_dir / "input_snapshot"
    snapshot_root.mkdir()
    (snapshot_root / "train").mkdir()
    (snapshot_root / "generation_status_snapshot.json").write_bytes(inputs["status_bytes"])
    (snapshot_root / "data_split_manifest_snapshot.json").write_bytes(inputs["plan_bytes"])

    catalog_target = snapshot_root / "service_catalog_500_1000.json"
    shutil.copy2(inputs["service_catalog"], catalog_target)
    catalog_hash = _sha256_file(catalog_target)
    if catalog_hash != inputs["service_catalog_sha256"]:
        raise RuntimeError("copied service catalog differs from the verified staging catalog")

    episode_snapshot: list[dict[str, Any]] = []
    for episode in inputs["episodes"]:
        seed = int(episode["seed"])
        source_dir = staging_root / "train" / f"episode_{seed}"
        target_dir = snapshot_root / "train" / f"episode_{seed}"
        target_dir.mkdir()
        manifest_path = source_dir / "episode_manifest.json"
        shutil.copy2(manifest_path, target_dir / manifest_path.name)
        copied_manifest_hash = _sha256_file(target_dir / manifest_path.name)
        if copied_manifest_hash != episode["source_episode_manifest_sha256"]:
            raise RuntimeError(f"copied Episode {seed} manifest changed during snapshot")
        artifact_hashes: dict[str, str] = {}
        for name, expected_hash in episode["artifact_sha256"].items():
            source = source_dir / name
            destination = target_dir / name
            shutil.copy2(source, destination)
            actual_hash = _sha256_file(destination)
            if actual_hash != expected_hash:
                raise RuntimeError(f"copied Episode {seed} artifact hash mismatch: {name}")
            artifact_hashes[name] = actual_hash
        episode_snapshot.append(
            {
                "seed": seed,
                "episode_manifest_sha256": copied_manifest_hash,
                "artifact_sha256": artifact_hashes,
            }
        )

    # Point only this Run's copied config at its private catalog snapshot.
    staged_config_text = inputs["staged_config"].read_text(encoding="utf-8")
    lines = staged_config_text.splitlines()
    in_paths = False
    replaced = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_paths = stripped == "[paths]"
        if in_paths and re.match(r"^\s*service_catalog\s*=", line):
            left = line.split("=", 1)[0]
            lines[index] = f"{left}= {json.dumps(catalog_target.resolve().as_posix())}"
            replaced += 1
    if replaced != 1:
        raise RuntimeError(f"expected one paths.service_catalog setting, found {replaced}")
    config_snapshot = run_dir / "config_snapshot.toml"
    config_snapshot.write_text("\n".join(lines) + "\n", encoding="utf-8")
    smoke_config = project_config.load_config(str(config_snapshot))
    configured_catalog = project_config.resolve_path(
        PROJECT_ROOT,
        project_config.get_config_value(smoke_config, "paths.service_catalog"),
    )
    if configured_catalog.resolve() != catalog_target.resolve():
        raise RuntimeError("smoke config does not resolve to the copied service catalog")
    staged_values = project_config.parse_toml_file(inputs["staged_config"])
    smoke_values = project_config.parse_toml_file(config_snapshot)
    staged_values["paths"]["service_catalog"] = "__snapshot_service_catalog__"
    smoke_values["paths"]["service_catalog"] = "__snapshot_service_catalog__"
    if staged_values != smoke_values:
        raise RuntimeError(
            "smoke config changed a value beyond redirecting paths.service_catalog"
        )

    snapshot_manifest = {
        "schema_version": "online_joint_dqn_staging_smoke_input_snapshot_v1",
        "run_type": "smoke_test",
        "source_staging_root": str(staging_root),
        "source_generation_status_sha256": hashlib.sha256(inputs["status_bytes"]).hexdigest(),
        "source_split_manifest_sha256": hashlib.sha256(inputs["plan_bytes"]).hexdigest(),
        "source_plan_fingerprint": inputs["plan"]["plan_fingerprint"],
        "formal_config_sha256": _sha256_file(PROJECT_ROOT / CONFIG_RELATIVE),
        "staged_config_sha256": _sha256_file(inputs["staged_config"]),
        "smoke_config_sha256": _sha256_file(config_snapshot),
        "smoke_config_semantic_sha256": project_config.semantic_config_hash(smoke_config),
        "service_catalog_sha256": catalog_hash,
        "execution_horizon_slots": EXECUTION_HORIZON_SLOTS,
        "arrival_horizon_slots": ARRIVAL_HORIZON_SLOTS,
        "episode_count": len(episode_snapshot),
        "episodes": episode_snapshot,
    }
    snapshot_manifest_path = snapshot_root / "input_snapshot_manifest.json"
    _write_json(snapshot_manifest_path, snapshot_manifest)
    snapshot_manifest["snapshot_manifest_sha256"] = _sha256_file(snapshot_manifest_path)
    return {
        "snapshot_root": snapshot_root,
        "config_snapshot": config_snapshot,
        "snapshot_manifest": snapshot_manifest,
        "snapshot_manifest_path": snapshot_manifest_path,
    }


def _make_environment(config_snapshot: Path, episode_dir: Path, seed: int) -> ExclusiveDQNEnvironment:
    context = load_environment_context(
        config_path=str(config_snapshot),
        realization_dir=str(episode_dir),
    )
    steps = aligned_execution_steps(context)
    if len(steps) != EXECUTION_HORIZON_SLOTS:
        raise RuntimeError(f"Episode {seed} aligned to {len(steps)} steps, expected 314")
    model = SystemModelContext(
        service_catalog=context.service_catalog,
        project_cfg=context.cfg,
        cache_nodes=context.cache_nodes,
        cache_config=context.cache_cfg,
        flight_boundary=context.flight_boundary,
    )
    slot_inputs = tuple(
        SlotInput(
            slot_id=int(step.slot),
            time_s=float(step.time_s),
            scenario_slice=step.scenario_slice,
            tasks=tuple(step.actual_tasks),
            vehicle_positions=step.actual_vehicle_positions,
            predicted_vehicle_positions=step.predicted_vehicle_positions,
            model=model,
            prediction_horizon_mask=step.prediction_horizon_mask,
            scenario_id=f"formal_train:{int(seed)}:{int(step.slot)}",
            source_kind="current_actual",
            prediction_metadata={
                "source": "formal_aligned_execution_steps",
                "episode_seed": int(seed),
                "origin_slot": int(step.slot),
                "future_ground_truth_used_for_candidate_features": False,
            },
        )
        for step in steps
    )
    environment = ExclusiveDQNEnvironment(
        context=context,
        slot_inputs=slot_inputs,
        decision_steps=steps,
        decision_count=len(slot_inputs),
        terminal_at_end=True,
    )
    if environment.episode_seed != int(seed):
        raise RuntimeError(
            f"loaded Episode seed mismatch: requested={seed}, actual={environment.episode_seed}"
        )
    return environment


def _request_boundary_stop(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(
        f"[StagingSmoke] stop requested by signal {signum}; will stop after the current Episode boundary",
        flush=True,
    )


def run(output_dir: Path | None = None) -> int:
    os.chdir(PROJECT_ROOT)
    staging_root = PROJECT_ROOT / STAGING_RELATIVE
    inputs = _select_and_validate_inputs(staging_root)
    output_parent = (
        (PROJECT_ROOT / "runs")
        if output_dir is None
        else (output_dir.parent if output_dir.is_absolute() else PROJECT_ROOT / output_dir.parent)
    )
    output_parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_parent).free
    if free_bytes < inputs["total_snapshot_bytes"] * 2 + 1024**3:
        raise RuntimeError(
            "insufficient disk space for an isolated input snapshot and training artifacts: "
            f"free={free_bytes}, snapshot={inputs['total_snapshot_bytes']}"
        )

    run_dir = _new_run_directory(output_dir)
    run_manifest: dict[str, Any] = {
        "schema_version": "online_joint_dqn_staging_smoke_run_v1",
        "run_type": "smoke_test",
        "status": "PREPARING_INPUT_SNAPSHOT",
        "target_episode_count": EPISODE_COUNT,
        "selected_episode_seeds": inputs["selected_seeds"],
        "source_generation_status": inputs["status"].get("status"),
        "source_current_episode": inputs["status"].get("current_episode"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_validation_performed": False,
        "test_evaluation_performed": False,
        "formal_resume_compatible": False,
    }
    run_manifest_path = run_dir / "run_manifest.json"
    _write_json(run_manifest_path, run_manifest)
    active_seed: int | None = None
    try:
        snapshot = _copy_snapshot(staging_root, run_dir, inputs)
        snapshot_manifest = snapshot["snapshot_manifest"]
        config = project_config.load_config(str(snapshot["config_snapshot"]))
        protocol = project_config.get_episode_data_protocol(config)
        if int(protocol["execution_horizon_slots"]) != EXECUTION_HORIZON_SLOTS:
            raise RuntimeError("smoke config no longer resolves to the 314-slot contract")

        first_seed = int(inputs["selected_seeds"][0])
        first_environment = _make_environment(
            snapshot["config_snapshot"],
            snapshot["snapshot_root"] / "train" / f"episode_{first_seed}",
            first_seed,
        )
        if (
            first_environment.arrival_horizon_slots != ARRIVAL_HORIZON_SLOTS
            or first_environment.execution_horizon_slots != EXECUTION_HORIZON_SLOTS
        ):
            raise RuntimeError("loaded smoke environment violates the 300/314 horizon contract")
        expected_state_dim = first_environment.encoder.state_dim
        expected_candidate_dim = first_environment.encoder.candidate_dim
        expected_search_budget = first_environment.search_budget

        agent = DoubleDQNAgent(
            expected_state_dim,
            expected_candidate_dim,
            learning_rate=formal_training.DEFAULT_LEARNING_RATE,
            gamma=1.0,
            seed=TRAINING_SEED,
        )
        if torch.cuda.is_available() and agent.device.type != "cuda":
            raise RuntimeError(f"CUDA is available but Agent selected {agent.device}")
        if not torch.cuda.is_available() and agent.device.type != "cpu":
            raise RuntimeError(f"CUDA is unavailable but Agent selected {agent.device}")
        replay = ReplayBuffer(REPLAY_CAPACITY, seed=TRAINING_SEED)
        trainer_config = TrainerConfig(
            epsilon_decay_steps=FORMAL_EPSILON_DECAY_STEPS
        )
        search_budget = {
            "max_total_evaluations_per_sample": int(
                first_environment.search_budget.max_total_evaluations
            ),
            "max_cache_candidates_including_keep": int(
                first_environment.search_budget.max_cache_candidates
            ),
            "max_uav_candidates_per_uav_including_stay": int(
                first_environment.search_budget.max_uav_candidates_per_uav
            ),
        }
        code_identity = formal_training._code_identity()
        smoke_identity = {
            "run_type": "smoke_test",
            "algorithm": "online_joint_dueling_double_dqn_existing_training_path",
            "time_horizon_contract": {
                "arrival_horizon_slots": ARRIVAL_HORIZON_SLOTS,
                "execution_horizon_slots": EXECUTION_HORIZON_SLOTS,
                "k_max_slots": K_MAX_SLOTS,
                "formula": "T_ext=T+K_max-1",
            },
            "input_snapshot_sha256": snapshot_manifest["snapshot_manifest_sha256"],
            "selected_episode_seeds": list(inputs["selected_seeds"]),
            "formal_config_sha256": snapshot_manifest["formal_config_sha256"],
            "smoke_config_sha256": snapshot_manifest["smoke_config_sha256"],
            "trainer_config": formal_training._jsonable(trainer_config.__dict__),
            "agent_config": {
                "state_dim": agent.state_dim,
                "candidate_dim": agent.candidate_dim,
                "learning_rate": agent.learning_rate,
                "gamma": agent.gamma,
                "target_network_frequency": agent.target_network_frequency,
                "target_tau": agent.target_tau,
                "hidden_sizes": [
                    int(module.out_features)
                    for module in agent.online_network.network
                    if isinstance(module, torch.nn.Linear)
                ][:-1],
                "device": str(agent.device),
                "seed": agent.seed,
            },
            "replay_capacity": REPLAY_CAPACITY,
            "candidate_capacity_limits": search_budget,
            "code_identity": code_identity,
            "smoke_entry_sha256": _sha256_file(Path(__file__).resolve()),
        }
        smoke_identity["identity_sha256"] = formal_training._identity_sha256(
            formal_training._identity_payload(smoke_identity)
        )

        run_manifest.update(
            {
                "status": "RUNNING",
                "identity_sha256": smoke_identity["identity_sha256"],
                "identity_payload": formal_training._identity_payload(smoke_identity),
                "code_identity": code_identity,
                "smoke_entry_sha256": smoke_identity["smoke_entry_sha256"],
                "input_snapshot_path": str(snapshot["snapshot_root"]),
                "input_snapshot_sha256": snapshot_manifest["snapshot_manifest_sha256"],
                "config_snapshot_path": str(snapshot["config_snapshot"]),
                "training_pool": {
                    "kind": "verified_completed_staging_subset",
                    "episode_count": EPISODE_COUNT,
                    "episode_seeds": list(inputs["selected_seeds"]),
                    "manifest_sha256": snapshot_manifest["source_split_manifest_sha256"],
                    "episode_manifest_sha256": {
                        str(item["seed"]): item["episode_manifest_sha256"]
                        for item in snapshot_manifest["episodes"]
                    },
                },
                "training_config": smoke_identity["trainer_config"],
                "agent_config": smoke_identity["agent_config"],
                "latest_checkpoint_path": str(run_dir / "checkpoints" / "latest_checkpoint.pt"),
                "completed_episodes": 0,
                "global_step": 0,
                "gradient_update_count": 0,
                "validation_performed": False,
                "test_performed": False,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(run_manifest_path, run_manifest)

        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir()
        episode_rows: list[dict[str, Any]] = []
        global_step = 0
        if torch.cuda.is_available():
            free_vram, total_vram = torch.cuda.mem_get_info()
            print(
                f"[StagingSmoke] device={agent.device} gpu={torch.cuda.get_device_name(agent.device)} "
                f"vram_free_gib={free_vram / 1024**3:.2f}/{total_vram / 1024**3:.2f}",
                flush=True,
            )
        else:
            print(f"[StagingSmoke] device={agent.device} (CPU fallback)", flush=True)

        old_sigint = signal.signal(signal.SIGINT, _request_boundary_stop)
        old_sigterm = signal.signal(signal.SIGTERM, _request_boundary_stop)
        step_log_path = run_dir / "step_metrics.jsonl"
        update_log_path = run_dir / "update_metrics.jsonl"
        episode_log_path = run_dir / "episode_metrics.jsonl"
        status = "COMPLETED"
        failure: dict[str, str] | None = None
        try:
            with step_log_path.open("a", encoding="utf-8", buffering=1) as step_stream, update_log_path.open(
                "a", encoding="utf-8", buffering=1
            ) as update_stream, episode_log_path.open(
                "a", encoding="utf-8", buffering=1
            ) as episode_stream:
                for episode_index, seed in enumerate(inputs["selected_seeds"]):
                    if _STOP_REQUESTED:
                        status = "STOPPED_AT_EPISODE_BOUNDARY"
                        break
                    active_seed = int(seed)
                    if episode_index == 0:
                        environment = first_environment
                        first_environment = None
                    else:
                        environment = _make_environment(
                            snapshot["config_snapshot"],
                            snapshot["snapshot_root"] / "train" / f"episode_{seed}",
                            int(seed),
                        )
                    if (
                        environment.encoder.state_dim != expected_state_dim
                        or environment.encoder.candidate_dim != expected_candidate_dim
                        or environment.search_budget != expected_search_budget
                    ):
                        raise RuntimeError(f"Episode {seed} DQN dimensions/candidate limits differ")
                    episode_started = time.perf_counter()

                    def on_step(row: dict[str, Any]) -> None:
                        _append_jsonl(
                            step_stream,
                            {
                                **row,
                                "episode_seed": int(seed),
                                "episode_index": episode_index,
                            },
                        )
                        if int(row.get("slot_id", -1)) % 10 == 0 or int(
                            row.get("slot_id", -1)
                        ) == EXECUTION_HORIZON_SLOTS - 1:
                            print(
                                f"[StagingSmoke] episode={episode_index + 1}/{EPISODE_COUNT} "
                                f"seed={seed} slot={row.get('slot_id')} "
                                f"global_step={row.get('global_step')}",
                                flush=True,
                            )

                    def on_update(row: dict[str, Any]) -> None:
                        _append_jsonl(
                            update_stream,
                            {
                                **row,
                                "episode_seed": int(seed),
                                "episode_index": episode_index,
                            },
                        )

                    metrics = train_episode(
                        environment,
                        agent,
                        replay,
                        config=trainer_config,
                        global_step_start=global_step,
                        on_step=on_step,
                        on_update=on_update,
                    )
                    elapsed = time.perf_counter() - episode_started
                    step_rows = metrics.pop("steps")
                    update_rows = metrics.pop("updates")
                    if (
                        len(step_rows) != EXECUTION_HORIZON_SLOTS
                        or int(metrics["step_count"]) != EXECUTION_HORIZON_SLOTS
                        or not metrics["episode_terminated"]
                        or metrics["window_truncated"]
                    ):
                        raise RuntimeError(f"Episode {seed} did not complete all 314 slots")
                    tail_arrivals = [
                        int(row.get("slot_id", -1))
                        for row in step_rows
                        if int(row.get("slot_id", -1)) >= ARRIVAL_HORIZON_SLOTS
                        and row.get("arrived_task_ids")
                    ]
                    if tail_arrivals:
                        raise RuntimeError(f"Episode {seed} has tail arrivals at {tail_arrivals}")
                    reward_sum = math.fsum(float(row["reward"]) for row in step_rows)
                    if not math.isclose(
                        float(metrics["episode_reward"]),
                        reward_sum,
                        rel_tol=1e-12,
                        abs_tol=1e-9,
                    ):
                        raise RuntimeError(f"Episode {seed} reward total does not match slot rewards")
                    next_global_step = int(metrics["global_step_end"])
                    if next_global_step != (episode_index + 1) * EXECUTION_HORIZON_SLOTS:
                        raise RuntimeError(f"Episode {seed} global step boundary is inconsistent")
                    if replay.write_count != next_global_step:
                        raise RuntimeError(f"Episode {seed} Replay writes do not match global steps")
                    summary = {
                        **formal_training._jsonable(metrics),
                        "episode_seed": int(seed),
                        "episode_index": episode_index,
                        "episode_wall_time_s": float(elapsed),
                        "gradient_updates_this_episode": len(update_rows),
                        "gradient_update_count": int(agent.gradient_steps),
                        "slot_reward_sum": float(reward_sum),
                        "tail_arrival_slots": tail_arrivals,
                    }
                    episode_rows.append(summary)
                    _append_jsonl(episode_stream, summary)
                    for stream in (step_stream, update_stream, episode_stream):
                        os.fsync(stream.fileno())

                    checkpoint_path = checkpoint_dir / "latest_checkpoint.pt"
                    checkpoint_payload = formal_training._checkpoint_payload(
                        checkpoint_path,
                        agent=agent,
                        replay=replay,
                        identity=smoke_identity,
                        episode_order=[int(value) for value in inputs["selected_seeds"]],
                        episode_rows=episode_rows,
                        next_episode_index=episode_index + 1,
                        global_step=next_global_step,
                        checkpoint_role="latest",
                        training_metadata={
                            "run_type": "smoke_test",
                            "validation_history": [],
                            "best_checkpoint": None,
                        },
                    )
                    checkpoint_payload["run_type"] = "smoke_test"
                    checkpoint_payload["smoke_only"] = True
                    checkpoint_bytes = formal_training._atomic_torch_save(
                        checkpoint_path, checkpoint_payload
                    )
                    global_step = next_global_step
                    active_seed = None
                    run_manifest.update(
                        {
                            "status": "RUNNING",
                            "completed_episodes": episode_index + 1,
                            "global_step": global_step,
                            "gradient_update_count": int(agent.gradient_steps),
                            "replay_write_count": int(replay.write_count),
                            "latest_checkpoint_bytes": checkpoint_bytes,
                            "latest_checkpoint_completed_episodes": episode_index + 1,
                            "latest_checkpoint_global_step": global_step,
                            "last_completed_episode_seed": int(seed),
                            "last_episode_wall_time_s": float(elapsed),
                            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    _write_json(run_manifest_path, run_manifest)
                    _write_json(
                        run_dir / "run_summary.json",
                        {
                            "run_type": "smoke_test",
                            "status": "RUNNING",
                            "completed_episodes": len(episode_rows),
                            "target_episode_count": EPISODE_COUNT,
                            "global_step": global_step,
                            "gradient_update_count": int(agent.gradient_steps),
                            "episode_rows": episode_rows,
                            "validation_performed": False,
                            "test_performed": False,
                        },
                    )
                    print(
                        f"[StagingSmoke] completed episode={episode_index + 1}/{EPISODE_COUNT} "
                        f"seed={seed} steps={metrics['step_count']} "
                        f"updates={len(update_rows)} return={metrics['episode_reward']:.9g} "
                        f"elapsed_s={elapsed:.1f} checkpoint_bytes={checkpoint_bytes}",
                        flush=True,
                    )
                    if episode_index > 0:
                        del environment
                    if _STOP_REQUESTED:
                        status = "STOPPED_AT_EPISODE_BOUNDARY"
                        break
        except BaseException as exc:
            status = "FAILED"
            failure = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
            first_environment = None

        run_manifest.update(
            {
                "status": status,
                "completed_episodes": len(episode_rows),
                "global_step": global_step,
                "gradient_update_count": int(agent.gradient_steps),
                "replay_write_count": int(replay.write_count),
                "validation_performed": False,
                "test_performed": False,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(run_manifest_path, run_manifest)
        _write_json(
            run_dir / "run_summary.json",
            {
                "run_type": "smoke_test",
                "status": status,
                "completed_episodes": len(episode_rows),
                "target_episode_count": EPISODE_COUNT,
                "global_step": global_step,
                "gradient_update_count": int(agent.gradient_steps),
                "episode_rows": episode_rows,
                "validation_performed": False,
                "test_performed": False,
            },
        )
        return 0 if status == "COMPLETED" else 2
    except BaseException as exc:
        if run_manifest.get("status") not in {"FAILED", "COMPLETED", "STOPPED_AT_EPISODE_BOUNDARY"}:
            run_manifest.update(
                {
                    "status": "FAILED",
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                    "active_episode_seed": active_seed,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            _write_json(run_manifest_path, run_manifest)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        return run(args.output_dir)
    except BaseException as exc:
        print(f"[StagingSmoke] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
