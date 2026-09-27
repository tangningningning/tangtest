"""Frozen, actor-driven evaluation on complete unseen episode realizations."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import project_config

from .evaluation_metrics import EXECUTION_CLASSES, aggregate_episode_task_metrics
from .policy import AdaptiveJointCacheUAVPPOActorCritic
from .primitive_control import PrimitiveTensorSpec
from .reporting import checkpoint_run_directory, resolve_checkpoint_reference
from .rollout import collect_rollout
from .trainer import (
    _kkt_metrics_from_diagnostics,
    _matching_termination_metrics,
    aligned_train_steps,
    hyperparameters_from_project,
    load_environment_context,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_MANIFEST = PROJECT_ROOT / "environment_data" / "data_split_manifest.json"
REQUIRED_ARCHITECTURE_VERSION = 5
SELECTION_METRIC = "mean_episode_return"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _parameter_hash(parameters: Any) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach().cpu().contiguous()
        digest.update(f"{tuple(value.shape)}:{value.dtype}\n".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_manifest(path: Path = SPLIT_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != 1 or manifest.get("slots_per_realization") != 300:
        raise ValueError("data split manifest protocol must be version 1 with 300 slots")
    for field in ("generator_entrypoint", "generator_config_path",
                  "generator_config_sha256", "generation_config_fingerprint",
                  "service_catalog_sha256", "sumo_network_sha256",
                  "nominal_scenario_identifier", "episode_seed_protocol"):
        if not manifest.get(field):
            raise ValueError(f"data split manifest missing provenance field {field}")
    config = PROJECT_ROOT / manifest["generator_config_path"]
    if _file_hash(config) != manifest["generator_config_sha256"]:
        raise ValueError("data split manifest generator config SHA256 mismatch")
    for label, path_key, hash_key in (
        ("service catalog", "service_catalog_path", "service_catalog_sha256"),
        ("popularity GRU", "popularity_gru_path", "popularity_gru_sha256"),
        ("vehicle residual bank", "vehicle_residual_bank_path", "vehicle_residual_bank_sha256"),
        ("SUMO network", "sumo_network_path", "sumo_network_sha256"),
    ):
        if not manifest.get(path_key) or not manifest.get(hash_key):
            raise ValueError(f"data split manifest missing {label} provenance")
        if _file_hash(PROJECT_ROOT / manifest[path_key]) != manifest[hash_key]:
            raise ValueError(f"data split manifest {label} SHA256 mismatch")
    expected = {"train": ("proposed_joint_training", 20001, 20300, 300),
                "validation": ("proposed_joint_validation", 30001, 30030, 30),
                "test": ("proposed_joint_test", 40001, 40100, 100)}
    seen: set[int] = set()
    for split, contract in expected.items():
        value = manifest.get(split, {})
        actual = (value.get("directory"), value.get("seed_start"),
                  value.get("seed_end"), value.get("count"))
        if actual != contract:
            raise ValueError(f"data split manifest {split} mismatch: {actual}")
        seeds = set(range(contract[1], contract[2] + 1))
        if seen & seeds:
            raise ValueError("data split manifest seeds overlap")
        seen |= seeds
    return manifest


def _checkpoint_architecture(payload: dict[str, Any]) -> tuple[int, str]:
    explicit = payload.get("architecture_version")
    state = payload.get("policy_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint lacks policy_state_dict")
    has_candidate_encoder = any(key.startswith("uav_candidate_encoder.") for key in state)
    has_binary_cache_trigger = any(key.startswith("cache_trigger_head.") for key in state)
    signature_version = (
        3 if has_binary_cache_trigger else 2 if has_candidate_encoder else 1
    )
    if explicit is None:
        return signature_version, "strict_policy_state_dict_signature"
    version = int(explicit)
    if version != signature_version:
        raise ValueError(f"checkpoint architecture metadata/state mismatch: metadata={version}, state={signature_version}")
    return version, "checkpoint_architecture_version"


def load_frozen_policy(checkpoint: Path, context: Any) -> tuple[Any, dict[str, Any]]:
    checkpoint = resolve_checkpoint_reference(checkpoint).resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    checkpoint_version, source = _checkpoint_architecture(payload)
    if checkpoint_version != REQUIRED_ARCHITECTURE_VERSION:
        raise ValueError("Checkpoint architecture mismatch: "
                         f"checkpoint={checkpoint_version}, runtime={REQUIRED_ARCHITECTURE_VERSION}")
    if payload.get("policy_schema") != context.policy_schema:
        raise ValueError("checkpoint policy schema mismatch")
    if int(payload.get("episode_slots", -1)) != 300:
        raise ValueError("checkpoint episode_slots mismatch")
    if int(payload.get("observation_dim", -1)) != context.observation_spec.observation_dim:
        raise ValueError("checkpoint observation_dim mismatch")
    if int(payload.get("action_dim", -1)) != context.action_spec.action_dim:
        raise ValueError("checkpoint action_dim mismatch")
    training_config_path = checkpoint_run_directory(checkpoint) / "run_config.toml"
    if training_config_path.is_file():
        training_config_sha256 = _file_hash(training_config_path)
        if training_config_sha256 != payload.get("config_sha256"):
            raise ValueError("checkpoint training config SHA256 mismatch")
        training_config = project_config.load_config(training_config_path)
        for reference in ("T_ref", "E_ref"):
            key = f"objective.{reference}"
            if float(project_config.get_config_value(training_config, key)) != float(
                project_config.get_config_value(context.cfg, key)
            ):
                raise ValueError(f"evaluation {reference} differs from checkpoint training config")
    else:
        training_config_sha256 = None
    spec = PrimitiveTensorSpec.create(context)
    hidden = hyperparameters_from_project(context.cfg).hidden_sizes
    policy = AdaptiveJointCacheUAVPPOActorCritic(
        int(context.observation_spec.observation_dim),
        int(context.action_spec.action_dim),
        spec.cache_feature_dim, spec.uav_feature_dim,
        spec.num_cache_heads, hidden, num_uav_heads=spec.num_uav_heads,
    )
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    policy.eval()
    if any(parameter.grad is not None for parameter in policy.parameters()):
        raise AssertionError("frozen policy unexpectedly has gradients")
    return policy, {"checkpoint_epoch": int(payload.get("completed_epochs", payload.get("episode", -1))),
                    "architecture_version_source": source,
                    "training_config_sha256": training_config_sha256,
                    "checkpoint_version": int(payload.get("checkpoint_version", -1))}


def _action_hash(rollout: Any) -> str:
    payload = {
        "action": rollout.actions.tolist(),
        "cache_operation": rollout.cache_operation_actions.tolist(),
        "cache_candidate": rollout.cache_selected_indices.tolist(),
        "uav_operation": rollout.uav_operation_actions.tolist(),
        "uav_candidate": rollout.uav_selected_indices.tolist(),
    }
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def evaluate_realization(policy: Any, context: Any, *, split: str,
                         checkpoint_id: str, seed: int) -> dict[str, Any]:
    if policy.training:
        raise AssertionError("evaluation policy must be in eval mode")
    actor_before = _parameter_hash(policy.actor_parameters())
    critic_before = _parameter_hash(policy.critic_parameters())
    steps = aligned_train_steps(context)
    if len(steps) != 300:
        raise ValueError("evaluation must execute a complete 300-slot realization")
    with torch.no_grad():
        rollout = collect_rollout(
            policy, context, steps,
            initial_cache=context.initial_cache,
            initial_uav=context.initial_uav,
            terminal_origin_index=context.terminal_origin,
            deterministic=True,
        )
    actor_changed = actor_before != _parameter_hash(policy.actor_parameters())
    critic_changed = critic_before != _parameter_hash(policy.critic_parameters())
    if actor_changed or critic_changed or any(p.grad is not None for p in policy.parameters()):
        raise AssertionError("evaluation changed policy parameters or gradients")
    transitions = rollout.transitions
    if len(transitions) != 300 or not transitions[-1].done:
        raise AssertionError("evaluation did not complete the realization")
    tasks = aggregate_episode_task_metrics(transitions)
    matching = _matching_termination_metrics(t.matching_summary for t in transitions)
    kkt = _kkt_metrics_from_diagnostics(
        record for t in transitions for record in t.matching_summary.get("KKT_SOLVE_DIAGNOSTICS", [])
    )
    rewards = np.asarray([t.reward for t in transitions], dtype=np.float64)
    q_values = np.asarray([t.actual_task_utility for t in transitions], dtype=np.float64)
    g_values = np.asarray([
        t.actual_task_utility - t.cache_cost - t.movement_cost for t in transitions
    ], dtype=np.float64)
    if not np.allclose(rewards, g_values, rtol=1e-12, atol=1e-12):
        raise AssertionError("evaluation reward differs from actual G(t)")
    slot_delay = np.asarray([
        t.total_task_delay_s / t.task_count for t in transitions if t.task_count
    ], dtype=np.float64)
    cache_cost = float(sum(t.cache_cost for t in transitions))
    movement_cost = float(sum(t.movement_cost for t in transitions))
    node_roles = [str(role).upper() for role in context.cache_nodes["node_roles"]]
    role_counts = {role: 0 for role in ("SV", "RSU", "UAV")}
    role_denominators = {role: node_roles.count(role) * len(transitions) for role in role_counts}
    for transition in transitions:
        changed = np.any(transition.previous_cache_state != transition.next_cache_state, axis=1)
        for index, role in enumerate(node_roles):
            if role in role_counts:
                role_counts[role] += int(changed[index])
    execution = {name.lower() + "_execution_count": int(tasks["execution_counts"][name])
                 for name in EXECUTION_CLASSES}
    execution.update({name.lower() + "_execution_rate": float(tasks["execution_ratios"][name])
                      for name in EXECUTION_CLASSES})
    values = [*rewards.tolist(), *g_values.tolist(), *q_values.tolist(), cache_cost, movement_cost,
              float(tasks["mean_task_delay_s"] or 0), float(tasks["mean_vehicle_energy_j"] or 0)]
    finite = bool(all(math.isfinite(x) for x in values))
    record = {
        "split": split, "checkpoint_identifier": checkpoint_id,
        "architecture_version": REQUIRED_ARCHITECTURE_VERSION,
        "realization_seed": int(seed), "slots": len(transitions),
        "episode_return": float(rewards.sum()), "mean_reward": float(rewards.mean()),
        "mean_G": float(g_values.mean()), "mean_Q": float(q_values.mean()),
        "q_total": float(q_values.sum()), "task_count": int(tasks["total_tasks"]),
        "mean_delay": tasks["mean_task_delay_s"],
        "p95_delay": float(np.percentile(slot_delay, 95)) if slot_delay.size else None,
        "p95_delay_definition": "95th percentile of per-slot mean task delay",
        "total_energy": float(sum(t.total_vehicle_energy_j for t in transitions)),
        "mean_energy": tasks["mean_vehicle_energy_j"],
        "cache_cost": cache_cost, "movement_cost": movement_cost,
        "cache_change_rate": float(sum(t.actual_cache_update_count for t in transitions) /
                                   (len(transitions) * len(node_roles))),
        "uav_movement_rate": float(sum(t.actual_uav_movement_count for t in transitions) /
                                    (len(transitions) * len(context.action_spec.uav_ids))),
        "finite": finite,
        "matching_max_updates_hit_count": int(matching["max_updates_hit_count"]),
        "kkt_converged_rate": float(kkt["converged_rate"]),
        "kkt_finite_rate": float(kkt["finite_rate"]),
        "action_sequence_sha256": _action_hash(rollout),
        "actor_parameters_changed": actor_changed,
        "critic_parameters_changed": critic_changed,
        "ppo_updates": 0, "backward_calls": 0,
        "actor_optimizer_steps": 0, "critic_optimizer_steps": 0,
        "uav_exact_q_teacher_calls_for_action_selection": 0,
        **execution,
    }
    for role in role_counts:
        record[role.lower() + "_cache_update_rate"] = (
            None if role_denominators[role] == 0 else role_counts[role] / role_denominators[role]
        )
    if not finite:
        raise FloatingPointError("evaluation produced NaN/Inf")
    return record


def _aggregate(records: list[dict[str, Any]], *, split: str, checkpoint: Path,
               checkpoint_id: str, epoch: int) -> dict[str, Any]:
    returns = np.asarray([r["episode_return"] for r in records], dtype=np.float64)
    sample_std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    se = sample_std / math.sqrt(len(returns))
    def mean(field: str) -> float | None:
        values = [float(r[field]) for r in records if r.get(field) is not None]
        return float(np.mean(values)) if values else None
    return {
        "split": split, "checkpoint": str(checkpoint), "checkpoint_identifier": checkpoint_id,
        "checkpoint_epoch": epoch, "architecture_version": REQUIRED_ARCHITECTURE_VERSION,
        "num_realizations": len(records), "selection_metric": SELECTION_METRIC if split == "validation" else None,
        "mean_episode_return": float(np.mean(returns)), "std_episode_return": sample_std,
        "median_episode_return": float(np.median(returns)),
        "p10_episode_return": float(np.percentile(returns, 10)),
        "p90_episode_return": float(np.percentile(returns, 90)),
        "standard_error_episode_return": se,
        "ci95_low": float(np.mean(returns) - 1.96 * se),
        "ci95_high": float(np.mean(returns) + 1.96 * se),
        "mean_G": mean("mean_G"), "mean_delay": mean("mean_delay"),
        "p95_delay_across_episode_aggregates": float(np.percentile(
            [r["mean_delay"] for r in records if r["mean_delay"] is not None], 95
        )) if any(r["mean_delay"] is not None for r in records) else None,
        "mean_episode_p95_delay": mean("p95_delay"),
        "mean_energy": mean("mean_energy"), "mean_cache_cost": mean("cache_cost"),
        "mean_movement_cost": mean("movement_cost"),
        "mean_cache_change_rate": mean("cache_change_rate"),
        "mean_uav_movement_rate": mean("uav_movement_rate"),
        "nan_inf_detected": any(not r["finite"] for r in records),
        "matching_max_updates_hit_total": sum(r["matching_max_updates_hit_count"] for r in records),
        "min_kkt_converged_rate": min(r["kkt_converged_rate"] for r in records),
        "min_kkt_finite_rate": min(r["kkt_finite_rate"] for r in records),
    }


def run_evaluation(*, checkpoint: Path, split: str, data_root: Path,
                   output_dir: Path, config_path: Path, smoke_one: bool = False,
                   selection_metadata: Path | None = None) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    manifest = load_split_manifest()
    split_contract = manifest[split]
    expected_root = PROJECT_ROOT / "environment_data" / split_contract["directory"]
    if data_root.resolve() != expected_root.resolve():
        raise ValueError(f"data_root does not match {split} split manifest")
    checkpoint = resolve_checkpoint_reference(checkpoint).resolve(strict=True)
    selection = None
    if split == "test":
        if selection_metadata is None:
            raise ValueError("test evaluation requires validation_selection.json")
        selection = json.loads(selection_metadata.read_text(encoding="utf-8"))
        selected_checkpoint_value = selection.get("selected_checkpoint_path")
        if (selection.get("selection_metric") != SELECTION_METRIC or
                not isinstance(selected_checkpoint_value, str) or
                not selected_checkpoint_value):
            raise ValueError("test checkpoint was not frozen by validation selection")
        selected_checkpoint = resolve_checkpoint_reference(
            Path(selected_checkpoint_value)
        ).resolve()
        if selected_checkpoint != checkpoint:
            raise ValueError("test checkpoint was not frozen by validation selection")
        if selection.get("selected_checkpoint_sha256") != _file_hash(checkpoint):
            raise ValueError("test checkpoint SHA256 differs from validation selection")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("evaluation output directory is not empty")
    seeds = range(split_contract["seed_start"], split_contract["seed_end"] + 1)
    if smoke_one:
        seeds = range(split_contract["seed_start"], split_contract["seed_start"] + 1)
    first_seed = next(iter(seeds))
    first_context = load_environment_context(str(config_path), data_root / f"episode_{first_seed}")
    policy, checkpoint_meta = load_frozen_policy(checkpoint, first_context)
    checkpoint_id = _file_hash(checkpoint)
    t_ref = float(project_config.get_config_value(first_context.cfg, "objective.T_ref"))
    e_ref = float(project_config.get_config_value(first_context.cfg, "objective.E_ref"))
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with (output_dir / "evaluation_episode_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for seed in seeds:
            context = (first_context if seed == first_seed else
                       load_environment_context(str(config_path), data_root / f"episode_{seed}"))
            if float(project_config.get_config_value(context.cfg, "objective.T_ref")) != t_ref or float(project_config.get_config_value(context.cfg, "objective.E_ref")) != e_ref:
                raise ValueError("frozen T_ref/E_ref changed across evaluation realizations")
            record = evaluate_realization(policy, context, split=split,
                                          checkpoint_id=checkpoint_id, seed=seed)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            records.append(record)
    metadata = {
        "split": split, "data_split_protocol_version": manifest["protocol_version"],
        "data_split_manifest_sha256": _file_hash(SPLIT_MANIFEST),
        "split_contract_sha256": hashlib.sha256(json.dumps(
            split_contract, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "checkpoint": str(checkpoint), "checkpoint_identifier": checkpoint_id,
        "checkpoint_epoch": checkpoint_meta["checkpoint_epoch"],
        "checkpoint_architecture_version_source": checkpoint_meta["architecture_version_source"],
        "checkpoint_training_config_sha256": checkpoint_meta["training_config_sha256"],
        "architecture_version": REQUIRED_ARCHITECTURE_VERSION,
        "deterministic_policy": True,
        "action_provider": "AdaptiveJointCacheUAVPPOActorCritic",
        "T_ref": t_ref, "E_ref": e_ref,
        "reference_source": "frozen_training_calibration",
        "selected_by": "validation" if split == "test" else None,
        "selection_metric": (
            selection.get("selection_metric") if selection else SELECTION_METRIC
        ),
        "validation_selected_epoch": selection.get("selected_checkpoint_epoch") if selection else None,
        "actor_optimizer_steps_during_eval": 0,
        "critic_optimizer_steps_during_eval": 0,
        "ppo_updates_during_eval": 0,
        "uav_exact_q_teacher_calls_for_action_selection": 0,
    }
    _write_json(output_dir / "evaluation_manifest.json", metadata)
    summary = _aggregate(records, split=split, checkpoint=checkpoint,
                         checkpoint_id=checkpoint_id, epoch=checkpoint_meta["checkpoint_epoch"])
    _write_json(output_dir / "evaluation_summary.json", summary)
    return summary
