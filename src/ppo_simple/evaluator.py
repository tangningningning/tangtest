"""Frozen PPO V6 held-out evaluator.

This module deliberately contains no optimizer or training update path.  It
loads the pinned Epoch-240 V6 checkpoint, reuses the production rollout and
environment execution path, and writes isolated held-out test artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - CLI help/import-only environments
    torch = None  # type: ignore[assignment]

from common import project_config
from . import interface
from .interface import _copy_uav_positions
from .policy import AdaptivePrimitiveHorizonGroupedSeparateCriticPopArtV6ActorCritic
from .primitive_control import PrimitiveTensorSpec
from .rollout import SimpleRollout, collect_rollout
from .trainer import (
    CRITIC_VALUE_SCALE_POPART_NORMALIZED,
    EnvironmentContext,
    hyperparameters_from_project,
    load_environment_context,
)


FORMAL_CONFIG_PATH = (
    "system_config_ppo_primitive_v6_horizon_grouped_separate_optimization_popart_"
    "actorlr1e4.toml"
)
FORMAL_TRAINING_RUN_DIR = Path("runs/ppo_v6_popart_actor1e4_fresh_20260910")
FORMAL_CHECKPOINT_PATH = FORMAL_TRAINING_RUN_DIR / "latest_checkpoint.pt"
EXPECTED_TRAINING_GIT_SHA = "0f04cb0d5250382ea8b36f16b6be720d81a20b09"
EXPECTED_CHECKPOINT_VERSION = 6
EXPECTED_POLICY_SCHEMA = (
    "adaptive_primitive_v6_horizon_grouped_separate_critic_popart"
)
TEST_ORIGIN_START = 240
TEST_ORIGIN_END = 299
TEST_SLOT_COUNT = TEST_ORIGIN_END - TEST_ORIGIN_START + 1
TEST_SEGMENTS = (
    (240, 254),
    (255, 269),
    (270, 284),
    (285, 299),
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PPO held-out evaluation requires PyTorch.")


def normalized_test_time_to_go(
    origin_index: int,
    test_origin_start: int = TEST_ORIGIN_START,
    test_origin_end: int = TEST_ORIGIN_END,
) -> float:
    """Return the finite-horizon time-to-go for the isolated test episode."""

    start = int(test_origin_start)
    end = int(test_origin_end)
    origin = int(origin_index)
    if end <= start:
        raise ValueError("test time_to_go requires test_origin_end > test_origin_start")
    if origin < start or origin > end:
        raise ValueError("test time_to_go origin is outside the held-out test range")
    value = (end - origin) / float(end - start)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise AssertionError("test time_to_go must be finite and within [0, 1]")
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_identity(project_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, None
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    dirty = dirty_result.stdout.strip() != "" if dirty_result.returncode == 0 else None
    return commit or None, dirty


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _resolve_output_dir(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def ensure_fresh_evaluation_dir(run_dir: Path) -> None:
    """Reject existing output contents instead of overwriting test results."""

    if run_dir.exists() and not run_dir.is_dir():
        raise ValueError(f"held-out evaluation output is not a directory: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            "fresh held-out test run-dir already contains evaluation artifacts; "
            "use a new --run-dir"
        )


def _load_training_provenance(
    checkpoint_path: Path,
    config_sha256: str,
) -> str:
    manifest_path = checkpoint_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"formal training run_manifest.json is required: {manifest_path}"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("config_sha256") != config_sha256:
        raise ValueError("formal training run_manifest config_sha256 mismatch")
    if manifest.get("policy_schema") != EXPECTED_POLICY_SCHEMA:
        raise ValueError("formal training run_manifest policy_schema mismatch")
    training_git_sha = manifest.get("git_commit_sha")
    if training_git_sha is None or str(training_git_sha).strip() != EXPECTED_TRAINING_GIT_SHA:
        raise ValueError(
            "formal training run_manifest git_commit_sha does not identify the frozen V6 run"
        )
    return EXPECTED_TRAINING_GIT_SHA


def _build_heldout_context(config_path: Path) -> EnvironmentContext:
    context = load_environment_context(str(config_path))
    get = project_config.get_config_value
    if context.configuration_control_mode != "policy_primitive":
        raise ValueError("held-out V6 evaluation requires policy_primitive control")
    if context.policy_schema != EXPECTED_POLICY_SCHEMA:
        raise ValueError(
            "held-out evaluator requires policy_schema="
            f"{EXPECTED_POLICY_SCHEMA}, got {context.policy_schema}"
        )
    configured_train_start = int(
        get(context.cfg, "algorithm.ppo.data_protocol.train_origin_start")
    )
    configured_train_end = int(
        get(context.cfg, "algorithm.ppo.data_protocol.train_origin_end")
    )
    configured_test_start = int(
        get(context.cfg, "algorithm.ppo.data_protocol.test_origin_start")
    )
    configured_test_end = int(
        get(context.cfg, "algorithm.ppo.data_protocol.test_origin_end")
    )
    if (configured_train_start, configured_train_end) != (0, 239):
        raise ValueError("formal V6 training origin range must remain 0..239")
    if (configured_test_start, configured_test_end) != (
        TEST_ORIGIN_START,
        TEST_ORIGIN_END,
    ):
        raise ValueError("formal V6 held-out test origin range must remain 240..299")
    if (
        str(get(context.cfg, "algorithm.ppo.critic_value_scale_mode"))
        != CRITIC_VALUE_SCALE_POPART_NORMALIZED
    ):
        raise ValueError("held-out V6 evaluation requires popart_normalized critic mode")
    if int(context.observation_spec.prediction_horizon) != 5:
        raise ValueError("formal V6 held-out evaluation requires prediction horizon H=5")
    if context.observation_spec.horizon_mask_slice is None:
        raise ValueError("formal V6 held-out evaluation requires a prediction horizon mask")
    return replace(
        context,
        train_origin_start=TEST_ORIGIN_START,
        train_origin_end=TEST_ORIGIN_END,
        observation_time_to_go=normalized_test_time_to_go,
    )


def _prepare_heldout_steps(context: EnvironmentContext) -> list[Any]:
    get = project_config.get_config_value
    root = Path(context.root)
    steps = interface.prepare_aligned_rollout_steps(
        root,
        context.scenarios,
        TEST_ORIGIN_START,
        TEST_SLOT_COUNT,
        project_config.resolve_path(
            root,
            get(context.cfg, "paths.vehicle_predictions"),
        ),
        project_config.resolve_path(
            root,
            get(
                context.cfg,
                "paths.popularity_predictions",
            ),
        ),
        project_config.resolve_path(root, get(context.cfg, "paths.tasks")),
        project_config.resolve_path(root, get(context.cfg, "paths.trajectory")),
        include_bootstrap_step=False,
        prediction_horizon=int(context.observation_spec.prediction_horizon),
        prediction_dt_s=float(get(context.cfg, "simulation.dt_s")),
        prediction_terminal_origin=TEST_ORIGIN_END,
        expected_vehicle_ids=context.observation_spec.vehicle_ids,
        expected_service_ids=context.observation_spec.service_ids,
        include_prediction_horizon_mask=(
            context.observation_spec.horizon_mask_slice is not None
        ),
    )
    expected_origins = tuple(range(TEST_ORIGIN_START, TEST_ORIGIN_END + 1))
    actual_origins = tuple(int(step.origin_index) for step in steps)
    if actual_origins != expected_origins:
        raise ValueError(
            "held-out aligned steps must contain exactly origins 240..299: "
            f"{actual_origins}"
        )
    return steps


def _load_frozen_v6_policy(
    context: EnvironmentContext,
    checkpoint_path: Path,
    config_sha256: str,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    _require_torch()
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("formal PPO checkpoint payload must be a mapping")
    checkpoint_version = int(payload.get("checkpoint_version", -1))
    if checkpoint_version != EXPECTED_CHECKPOINT_VERSION:
        raise ValueError(
            f"formal PPO checkpoint_version must be {EXPECTED_CHECKPOINT_VERSION}, "
            f"got {checkpoint_version}"
        )
    if payload.get("policy_schema") != EXPECTED_POLICY_SCHEMA:
        raise ValueError("formal PPO checkpoint policy_schema mismatch")
    completed_epochs = int(
        payload.get("completed_epochs", payload.get("completed_episodes", -1))
    )
    if completed_epochs != 240:
        raise ValueError(
            f"formal PPO checkpoint completed_epochs must be 240, got {completed_epochs}"
        )
    if payload.get("config_sha256") != config_sha256:
        raise ValueError("formal PPO checkpoint config_sha256 mismatch")
    if payload.get("critic_value_scale_mode") != CRITIC_VALUE_SCALE_POPART_NORMALIZED:
        raise ValueError("formal PPO checkpoint must use popart_normalized critic mode")
    hyper = hyperparameters_from_project(context.cfg)
    if hyper.critic_value_scale_mode != CRITIC_VALUE_SCALE_POPART_NORMALIZED:
        raise ValueError("held-out V6 configuration is not popart_normalized")
    tensor_spec = PrimitiveTensorSpec.create(context)
    policy = AdaptivePrimitiveHorizonGroupedSeparateCriticPopArtV6ActorCritic(
        int(context.observation_spec.observation_dim),
        int(context.action_spec.action_dim),
        tensor_spec.cache_feature_dim,
        tensor_spec.uav_feature_dim,
        hyper.hidden_sizes,
    ).to(device)
    state_dict = payload.get("policy_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("formal PPO checkpoint is missing policy_state_dict")
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy, payload


def _slot_records(
    rollout: SimpleRollout,
    cache_count: int,
    uav_count: int,
) -> list[dict[str, Any]]:
    if len(rollout.transitions) != TEST_SLOT_COUNT:
        raise ValueError(
            f"held-out rollout must contain {TEST_SLOT_COUNT} transitions, "
            f"got {len(rollout.transitions)}"
        )
    records: list[dict[str, Any]] = []
    for transition in rollout.transitions:
        origin = int(transition.origin_index)
        if origin < TEST_ORIGIN_START or origin > TEST_ORIGIN_END:
            raise ValueError(f"held-out transition origin outside 240..299: {origin}")
        q = float(transition.actual_task_utility)
        cache_cost = float(transition.cache_cost)
        movement_cost = float(transition.movement_cost)
        reward = float(transition.reward)
        values = (reward, q, cache_cost, movement_cost)
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError(f"held-out slot {origin} contains non-finite metrics")
        expected_reward = q - cache_cost - movement_cost
        if not np.isclose(reward, expected_reward, rtol=1.0e-12, atol=1.0e-9):
            raise AssertionError(f"held-out slot {origin} reward contract mismatch")
        action = np.asarray(transition.action, dtype=np.int64)
        if action.shape != (cache_count + uav_count,):
            raise ValueError(f"held-out slot {origin} action shape mismatch")
        records.append(
            {
                "origin": origin,
                "reward": reward,
                "q": q,
                "cache_cost": cache_cost,
                "movement_cost": movement_cost,
                "cache_trigger_count": int(action[:cache_count].sum()),
                "cache_change_count": int(transition.actual_cache_update_count),
                "uav_trigger_count": int(action[cache_count:].sum()),
                "uav_movement_count": int(transition.actual_uav_movement_count),
            }
        )
    records.sort(key=lambda record: int(record["origin"]))
    origins = tuple(int(record["origin"]) for record in records)
    if origins != tuple(range(TEST_ORIGIN_START, TEST_ORIGIN_END + 1)):
        raise ValueError(f"held-out slot origins are not continuous 240..299: {origins}")
    return records


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("held-out rate denominator must be positive")
    return float(numerator) / float(denominator)


def _summary(records: list[dict[str, Any]], cache_count: int, uav_count: int) -> dict[str, Any]:
    rewards = np.asarray([record["reward"] for record in records], dtype=np.float64)
    q_values = np.asarray([record["q"] for record in records], dtype=np.float64)
    cache_costs = np.asarray(
        [record["cache_cost"] for record in records], dtype=np.float64
    )
    movement_costs = np.asarray(
        [record["movement_cost"] for record in records], dtype=np.float64
    )
    cache_triggers = int(sum(record["cache_trigger_count"] for record in records))
    cache_changes = int(sum(record["cache_change_count"] for record in records))
    uav_triggers = int(sum(record["uav_trigger_count"] for record in records))
    uav_movements = int(sum(record["uav_movement_count"] for record in records))
    slot_count = len(records)
    summary: dict[str, Any] = {
        "test_origin_start": TEST_ORIGIN_START,
        "test_origin_end": TEST_ORIGIN_END,
        "test_slot_count": slot_count,
        "total_return": float(rewards.sum()),
        "mean_slot_reward": float(rewards.mean()),
        "total_q": float(q_values.sum()),
        "mean_q": float(q_values.mean()),
        "total_cache_cost": float(cache_costs.sum()),
        "total_movement_cost": float(movement_costs.sum()),
        "cache_trigger_rate": _rate(cache_triggers, slot_count * cache_count),
        "cache_change_rate": _rate(cache_changes, slot_count * cache_count),
        "cache_change_given_trigger_rate": (
            None if cache_triggers == 0 else _rate(cache_changes, cache_triggers)
        ),
        "uav_trigger_rate": _rate(uav_triggers, slot_count * uav_count),
        "uav_movement_rate": _rate(uav_movements, slot_count * uav_count),
        "uav_movement_given_trigger_rate": (
            None if uav_triggers == 0 else _rate(uav_movements, uav_triggers)
        ),
        "segments": [],
    }
    segments: list[dict[str, Any]] = []
    for segment_start, segment_end in TEST_SEGMENTS:
        segment_records = [
            record
            for record in records
            if segment_start <= int(record["origin"]) <= segment_end
        ]
        if len(segment_records) != segment_end - segment_start + 1:
            raise ValueError(
                f"held-out segment {segment_start}..{segment_end} is incomplete"
            )
        segments.append(
            {
                "origin_start": segment_start,
                "origin_end": segment_end,
                "mean_reward": float(
                    np.mean([record["reward"] for record in segment_records])
                ),
                "mean_q": float(np.mean([record["q"] for record in segment_records])),
                "mean_cache_cost": float(
                    np.mean([record["cache_cost"] for record in segment_records])
                ),
                "mean_movement_cost": float(
                    np.mean([record["movement_cost"] for record in segment_records])
                ),
            }
        )
    summary["segments"] = segments
    return summary


def evaluate_heldout(
    config_path: str | Path = FORMAL_CONFIG_PATH,
    checkpoint_path: str | Path = FORMAL_CHECKPOINT_PATH,
    run_dir: str | Path = "runs/ppo_v6_popart_actor1e4_fresh_20260910/test_latest_ep240",
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate the pinned V6 checkpoint on the isolated 240..299 episode."""

    _require_torch()
    project_root = project_config.find_project_root()
    resolved_config = project_config.resolve_path(project_root, config_path)
    expected_config = project_root / FORMAL_CONFIG_PATH
    if resolved_config != expected_config.resolve():
        raise ValueError(
            "held-out evaluator is pinned to the formal V6 config: "
            f"{expected_config.resolve()}"
        )
    resolved_checkpoint = project_config.resolve_path(project_root, checkpoint_path)
    expected_checkpoint = (project_root / FORMAL_CHECKPOINT_PATH).resolve()
    if resolved_checkpoint != expected_checkpoint:
        raise ValueError(
            "held-out evaluator is pinned to latest_checkpoint.pt from the formal "
            f"V6 run: {expected_checkpoint}"
        )
    if resolved_checkpoint.name != "latest_checkpoint.pt":
        raise ValueError("held-out evaluator refuses non-latest training checkpoints")
    output_dir = _resolve_output_dir(project_root, run_dir)
    ensure_fresh_evaluation_dir(output_dir)

    config_sha256 = _sha256(resolved_config)
    training_git_sha = _load_training_provenance(resolved_checkpoint, config_sha256)
    evaluation_git_sha, evaluation_git_dirty = _git_identity(project_root)
    evaluation_device = torch.device(device)
    context = _build_heldout_context(resolved_config)
    steps = _prepare_heldout_steps(context)
    policy, _ = _load_frozen_v6_policy(
        context,
        resolved_checkpoint,
        config_sha256,
        evaluation_device,
    )
    parameter_snapshot = [parameter.detach().clone() for parameter in policy.parameters()]
    popart_snapshot = policy.popart_statistics()
    with torch.inference_mode():
        rollout = collect_rollout(
            policy,
            context,
            steps,
            initial_cache=np.asarray(context.initial_cache, dtype=np.int8).copy(),
            initial_uav=_copy_uav_positions(context.initial_uav),
            terminal_origin_index=TEST_ORIGIN_END,
            observation_time_to_go=normalized_test_time_to_go,
            deterministic=True,
        )
    policy_parameters_updated = any(
        not torch.equal(before, after.detach())
        for before, after in zip(parameter_snapshot, policy.parameters(), strict=True)
    )
    if policy_parameters_updated:
        raise RuntimeError("held-out evaluation modified policy parameters")
    popart_statistics_updated = policy.popart_statistics() != popart_snapshot
    if popart_statistics_updated:
        raise RuntimeError("held-out evaluation modified PopArt statistics")
    records = _slot_records(
        rollout,
        len(context.action_spec.cache_node_ids),
        len(context.action_spec.uav_ids),
    )
    summary = _summary(
        records,
        len(context.action_spec.cache_node_ids),
        len(context.action_spec.uav_ids),
    )
    checkpoint_completed_epochs = 240
    manifest: dict[str, Any] = {
        "experiment_type": "heldout_test",
        "training_run_dir": str(resolved_checkpoint.parent),
        "checkpoint_path": str(resolved_checkpoint),
        "checkpoint_sha256": _sha256(resolved_checkpoint),
        "training_git_sha": training_git_sha,
        "evaluation_git_sha": evaluation_git_sha,
        "evaluation_git_dirty": evaluation_git_dirty,
        "config_path": str(resolved_config),
        "config_sha256": config_sha256,
        "checkpoint_completed_epochs": checkpoint_completed_epochs,
        "policy_schema": EXPECTED_POLICY_SCHEMA,
        "critic_value_scale_mode": CRITIC_VALUE_SCALE_POPART_NORMALIZED,
        "test_origin_start": TEST_ORIGIN_START,
        "test_origin_end": TEST_ORIGIN_END,
        "test_slot_count": TEST_SLOT_COUNT,
        "deterministic": True,
        "policy_parameters_updated": bool(policy_parameters_updated),
        "popart_statistics_updated": bool(popart_statistics_updated),
        "optimizer_steps": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "test_manifest.json", manifest)
    _write_json(output_dir / "test_summary.json", summary)
    with (output_dir / "test_slot_metrics.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen PPO V6 checkpoint on held-out origins 240..299."
    )
    parser.add_argument("--config", default=FORMAL_CONFIG_PATH)
    parser.add_argument("--checkpoint", default=str(FORMAL_CHECKPOINT_PATH))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate_heldout(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        run_dir=args.run_dir,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
