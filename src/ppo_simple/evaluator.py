"""Frozen PPO V6 held-out and full-episode evaluators.

This module deliberately contains no optimizer or training update path.  It
loads frozen V6 policies, reuses the production rollout and environment
execution path, and writes isolated evaluation artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from decision import cache_configuration, uav_configuration

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - CLI help/import-only environments
    torch = None  # type: ignore[assignment]

from common import project_config
from . import interface
from .interface import _copy_uav_positions
from .policy import (
    AdaptivePrimitiveHorizonGroupedSeparateCriticPopArtV6ActorCritic,
    _group_balance_candidate_logits,
)
from .primitive_control import PrimitiveTensorSpec
from .rollout import SimpleRollout, collect_rollout
from .trainer import (
    CRITIC_VALUE_SCALE_POPART_NORMALIZED,
    EnvironmentContext,
    GENERATOR_EXTRA_ARTIFACT_FILENAMES,
    _assert_realization_context_compatible,
    _realization_artifacts_from_directory,
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
FULL_EPISODE_CONFIG_PATH = "system_config_ppo_v6_full_episode_randomized.toml"
FULL_EPISODE_PROTOCOL = project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED
FULL_EPISODE_ORIGIN_START = 0
FULL_EPISODE_ORIGIN_END = 299
FULL_EPISODE_SLOT_COUNT = 300
POLICY_ACTION_MODES = (
    "candidate_deterministic",
    "grouped_deterministic",
    "stochastic",
)


def normalize_policy_action_mode(policy_action_mode: str) -> str:
    """Normalize the legacy deterministic alias without changing its semantics."""

    mode = str(policy_action_mode)
    if mode == "deterministic":
        return "candidate_deterministic"
    if mode not in POLICY_ACTION_MODES:
        allowed = ", ".join(("deterministic", *POLICY_ACTION_MODES))
        raise ValueError(f"policy_action_mode must be one of: {allowed}")
    return mode


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


def _load_full_episode_training_provenance(
    checkpoint_path: Path,
    config_sha256: str,
) -> str | None:
    """Validate provenance for a checkpoint produced by the full protocol."""

    manifest_path = checkpoint_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"full-episode training run_manifest.json is required: {manifest_path}"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("config_sha256") != config_sha256:
        raise ValueError("full-episode run_manifest config_sha256 mismatch")
    if manifest.get("policy_schema") != EXPECTED_POLICY_SCHEMA:
        raise ValueError("full-episode run_manifest policy_schema mismatch")
    if manifest.get("data_protocol") != FULL_EPISODE_PROTOCOL:
        raise ValueError("full-episode run_manifest data_protocol mismatch")
    expected_fields = {
        "episode_slots": FULL_EPISODE_SLOT_COUNT,
        "origin_start": FULL_EPISODE_ORIGIN_START,
        "origin_end": FULL_EPISODE_ORIGIN_END,
        "terminal_origin": FULL_EPISODE_ORIGIN_END,
        "rollout_steps": FULL_EPISODE_SLOT_COUNT,
        "num_minibatches": 10,
        "minibatch_size": 30,
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            raise ValueError(
                "full-episode run_manifest protocol mismatch: "
                f"field={field}, value={manifest.get(field)!r}, expected={expected}."
            )
    raw_git_sha = manifest.get("git_commit_sha")
    return None if raw_git_sha is None else str(raw_git_sha).strip() or None


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


def _assert_service_catalog_compatible(
    reference: EnvironmentContext,
    candidate: EnvironmentContext,
) -> None:
    for field in (
        "num_services",
        "service_ids",
        "service_sizes_mb",
        "mean_input_size_bit",
        "cycles_per_bit",
    ):
        reference_value = reference.service_catalog[field]
        candidate_value = candidate.service_catalog[field]
        if isinstance(reference_value, np.ndarray) or isinstance(
            candidate_value, np.ndarray
        ):
            if not np.array_equal(reference_value, candidate_value):
                raise ValueError(
                    "full-episode realization service catalog mismatch: "
                    f"field={field}"
                )
        elif reference_value != candidate_value:
            raise ValueError(
                "full-episode realization service catalog mismatch: "
                f"field={field}"
            )


def _validate_full_episode_realization_files(
    context: EnvironmentContext,
    realization_dir: Path,
) -> None:
    _realization_artifacts_from_directory(realization_dir)
    missing = [
        filename
        for filename in GENERATOR_EXTRA_ARTIFACT_FILENAMES
        if not (realization_dir / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "full-episode realization is incomplete: "
            f"directory={realization_dir}, missing={missing}"
        )
    if context.realization is None or context.realization.directory != realization_dir:
        raise AssertionError(
            "full-episode realization context does not point at the requested directory"
        )


def _build_full_episode_context(
    config_path: Path,
    realization_dir: str | Path | None = None,
) -> EnvironmentContext:
    reference_context = load_environment_context(str(config_path))
    context = (
        reference_context
        if realization_dir is None
        else load_environment_context(str(config_path), realization_dir=realization_dir)
    )
    if realization_dir is not None:
        resolved_realization_dir = Path(realization_dir)
        if not resolved_realization_dir.is_absolute():
            resolved_realization_dir = context.root / resolved_realization_dir
        resolved_realization_dir = resolved_realization_dir.resolve(strict=False)
        _validate_full_episode_realization_files(context, resolved_realization_dir)
        _assert_realization_context_compatible(reference_context, context)
        _assert_service_catalog_compatible(reference_context, context)
    if context.configuration_control_mode != "policy_primitive":
        raise ValueError("full-episode V6 evaluation requires policy_primitive control")
    if context.policy_schema != EXPECTED_POLICY_SCHEMA:
        raise ValueError(
            "full-episode evaluator requires policy_schema="
            f"{EXPECTED_POLICY_SCHEMA}, got {context.policy_schema}"
        )
    protocol = project_config.get_ppo_data_protocol(context.cfg)
    if protocol["protocol"] != FULL_EPISODE_PROTOCOL:
        raise ValueError(
            "full-episode evaluation requires data_protocol="
            f"{FULL_EPISODE_PROTOCOL}"
        )
    if (
        context.train_origin_start != FULL_EPISODE_ORIGIN_START
        or context.train_origin_end != FULL_EPISODE_ORIGIN_END
        or context.episode_slots != FULL_EPISODE_SLOT_COUNT
        or context.terminal_origin != FULL_EPISODE_ORIGIN_END
    ):
        raise ValueError("full-episode evaluation requires the 0..299 protocol")
    if int(context.observation_spec.prediction_horizon) != 5:
        raise ValueError("full-episode V6 evaluation requires prediction horizon H=5")
    if context.observation_spec.horizon_mask_slice is None:
        raise ValueError("full-episode V6 evaluation requires a prediction horizon mask")
    return context


def _prepare_full_episode_steps(context: EnvironmentContext) -> list[Any]:
    get = project_config.get_config_value
    root = Path(context.root)
    realization = context.realization
    steps = interface.prepare_aligned_rollout_steps(
        root,
        context.scenarios,
        FULL_EPISODE_ORIGIN_START,
        FULL_EPISODE_SLOT_COUNT,
        (
            realization.vehicle_predictions
            if realization is not None
            else project_config.resolve_path(
                root, get(context.cfg, "paths.vehicle_predictions")
            )
        ),
        (
            realization.popularity_predictions
            if realization is not None
            else project_config.resolve_path(
                root, get(context.cfg, "paths.popularity_predictions")
            )
        ),
        (
            realization.tasks
            if realization is not None
            else project_config.resolve_path(root, get(context.cfg, "paths.tasks"))
        ),
        (
            realization.trajectory
            if realization is not None
            else project_config.resolve_path(
                root, get(context.cfg, "paths.trajectory")
            )
        ),
        include_bootstrap_step=False,
        prediction_horizon=int(context.observation_spec.prediction_horizon),
        prediction_dt_s=float(get(context.cfg, "simulation.dt_s")),
        prediction_terminal_origin=FULL_EPISODE_ORIGIN_END,
        expected_vehicle_ids=context.observation_spec.vehicle_ids,
        expected_service_ids=context.observation_spec.service_ids,
        include_prediction_horizon_mask=(
            context.observation_spec.horizon_mask_slice is not None
        ),
    )
    expected_origins = tuple(
        range(FULL_EPISODE_ORIGIN_START, FULL_EPISODE_ORIGIN_END + 1)
    )
    actual_origins = tuple(int(step.origin_index) for step in steps)
    if actual_origins != expected_origins:
        raise ValueError(
            "full-episode aligned steps must contain exactly origins 0..299"
        )
    return steps


def _load_frozen_v6_policy(
    context: EnvironmentContext,
    checkpoint_path: Path,
    config_sha256: str,
    device: Any,
    *,
    expected_completed_epochs: int | None = 240,
    expected_protocol: str | None = None,
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
    if expected_protocol is not None:
        checkpoint_protocol = str(
            payload.get(
                "data_protocol",
                project_config.PPO_DATA_PROTOCOL_LEGACY_SPLIT,
            )
        )
        if checkpoint_protocol != expected_protocol:
            raise ValueError(
                "PPO checkpoint data_protocol mismatch; refusing cross-protocol evaluation: "
                f"checkpoint={checkpoint_protocol}, expected={expected_protocol}."
            )
        if expected_protocol == FULL_EPISODE_PROTOCOL:
            expected_protocol_fields = {
                "episode_slots": context.episode_slots,
                "origin_start": context.train_origin_start,
                "origin_end": context.train_origin_end,
                "terminal_origin": context.terminal_origin,
                "rollout_steps": context.episode_slots,
                "num_minibatches": project_config.get_ppo_data_protocol(
                    context.cfg
                )["num_minibatches"],
                "minibatch_size": context.minibatch_size,
            }
            for field, expected in expected_protocol_fields.items():
                if field not in payload or int(payload[field]) != int(expected):
                    raise ValueError(
                        "PPO full-episode checkpoint protocol mismatch: "
                        f"field={field}, checkpoint={payload.get(field)!r}, expected={expected}."
                    )
    completed_epochs = int(
        payload.get("completed_epochs", payload.get("completed_episodes", -1))
    )
    if expected_completed_epochs is not None and completed_epochs != expected_completed_epochs:
        raise ValueError(
            "formal PPO checkpoint completed_epochs must be "
            f"{expected_completed_epochs}, got {completed_epochs}"
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


def _slot_records_for_range(
    rollout: SimpleRollout,
    cache_count: int,
    uav_count: int,
    *,
    origin_start: int,
    origin_end: int,
    label: str,
) -> list[dict[str, Any]]:
    expected_slot_count = int(origin_end) - int(origin_start) + 1
    if len(rollout.transitions) != expected_slot_count:
        raise ValueError(
            f"{label} rollout must contain {expected_slot_count} transitions, "
            f"got {len(rollout.transitions)}"
        )
    records: list[dict[str, Any]] = []
    for transition in rollout.transitions:
        origin = int(transition.origin_index)
        if origin < origin_start or origin > origin_end:
            raise ValueError(
                f"{label} transition origin outside "
                f"{origin_start}..{origin_end}: {origin}"
            )
        q = float(transition.actual_task_utility)
        cache_cost = float(transition.cache_cost)
        movement_cost = float(transition.movement_cost)
        reward = float(transition.reward)
        values = (reward, q, cache_cost, movement_cost)
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError(
                f"{label} slot {origin} contains non-finite metrics"
            )
        expected_reward = q - cache_cost - movement_cost
        if not np.isclose(reward, expected_reward, rtol=1.0e-12, atol=1.0e-9):
            raise AssertionError(f"{label} slot {origin} reward contract mismatch")
        action = np.asarray(transition.action, dtype=np.int64)
        if action.shape != (cache_count + uav_count,):
            raise ValueError(f"{label} slot {origin} action shape mismatch")
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
    if origins != tuple(range(origin_start, origin_end + 1)):
        raise ValueError(
            f"{label} slot origins are not continuous "
            f"{origin_start}..{origin_end}: {origins}"
        )
    return records


def _slot_records(
    rollout: SimpleRollout,
    cache_count: int,
    uav_count: int,
) -> list[dict[str, Any]]:
    return _slot_records_for_range(
        rollout,
        cache_count,
        uav_count,
        origin_start=TEST_ORIGIN_START,
        origin_end=TEST_ORIGIN_END,
        label="held-out",
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("held-out rate denominator must be positive")
    return float(numerator) / float(denominator)


class _PolicyDiagnosticAccumulator:
    """Compact trigger-head distribution diagnostics for one full episode."""

    def __init__(self, context: EnvironmentContext) -> None:
        cache_heads = [
            (index, f"cache:{node_id}", "cache")
            for index, node_id in enumerate(context.action_spec.cache_node_ids)
        ]
        cache_count = len(cache_heads)
        uav_heads = [
            (cache_count + index, f"uav:{uav_id}", "uav")
            for index, uav_id in enumerate(context.action_spec.uav_ids)
        ]
        self.action_dim = int(context.action_spec.action_dim)
        self.heads: list[dict[str, Any]] = [
            {
                "head_index": int(head_index),
                "head_name": str(head_name),
                "head_type": str(head_type),
                "p1": [],
                "deterministic_ones": 0,
                "masked_to_zero": 0,
                "masked_to_one": 0,
            }
            for head_index, head_name, head_type in [*cache_heads, *uav_heads]
        ]
        if len(self.heads) != self.action_dim:
            raise ValueError(
                "policy diagnostic head layout does not match action_spec.action_dim"
            )
        self.slot_count = 0

    def observe(self, origin_index: int, sample: Any) -> None:
        del origin_index
        probabilities = np.asarray(sample.action_probabilities, dtype=np.float64)
        deterministic_action = np.asarray(sample.deterministic_action, dtype=np.int8)
        action_mask = np.asarray(sample.action_mask, dtype=bool)
        if probabilities.shape != (self.action_dim,):
            raise ValueError(
                "policy diagnostic probabilities shape does not match action_spec"
            )
        if deterministic_action.shape != (self.action_dim,):
            raise ValueError(
                "policy diagnostic deterministic action shape does not match action_spec"
            )
        if action_mask.shape != (self.action_dim, 2):
            raise ValueError("policy diagnostic action mask shape does not match action_spec")
        if not bool(np.all(np.isfinite(probabilities))) or bool(
            np.any(probabilities < 0.0) or np.any(probabilities > 1.0)
        ):
            raise ValueError("policy diagnostic probabilities must be finite in [0, 1]")
        if not bool(np.all(np.isin(deterministic_action, [0, 1]))):
            raise ValueError("policy diagnostic deterministic actions must be binary")
        for head_index, head in enumerate(self.heads):
            head["p1"].append(float(probabilities[head_index]))
            head["deterministic_ones"] += int(deterministic_action[head_index] == 1)
            head["masked_to_zero"] += int(not bool(action_mask[head_index, 0]))
            head["masked_to_one"] += int(not bool(action_mask[head_index, 1]))
        self.slot_count += 1

    @staticmethod
    def _summary(values: np.ndarray, deterministic_ones: int) -> dict[str, float]:
        if values.ndim != 1 or values.size == 0:
            raise ValueError("policy diagnostic summary requires non-empty values")
        count = int(values.size)
        return {
            "mean_p1": float(np.mean(values)),
            "median_p1": float(np.median(values)),
            "fraction_p1_gt_0_5": float(np.mean(values > 0.5)),
            "deterministic_one_rate": float(deterministic_ones) / float(count),
        }

    def _head_summary(self, head: dict[str, Any]) -> dict[str, Any]:
        values = np.asarray(head["p1"], dtype=np.float64)
        if values.size != self.slot_count:
            raise ValueError("policy diagnostic head slot count mismatch")
        count = int(values.size)
        return {
            "head_index": int(head["head_index"]),
            "head_name": str(head["head_name"]),
            "head_type": str(head["head_type"]),
            "mean_p1": float(np.mean(values)),
            "median_p1": float(np.median(values)),
            "min_p1": float(np.min(values)),
            "max_p1": float(np.max(values)),
            "fraction_p1_gt_0_5": float(np.mean(values > 0.5)),
            "fraction_p1_ge_0_5": float(np.mean(values >= 0.5)),
            "deterministic_one_rate": float(head["deterministic_ones"]) / float(count),
            "masked_to_zero_rate": float(head["masked_to_zero"]) / float(count),
            "masked_to_one_rate": float(head["masked_to_one"]) / float(count),
        }

    def _type_summary(
        self,
        head_type: str,
    ) -> dict[str, Any]:
        selected = [
            head
            for head in self.heads
            if str(head["head_type"]) == head_type
        ]
        values = np.asarray(
            [value for head in selected for value in head["p1"]],
            dtype=np.float64,
        )
        deterministic_ones = int(
            sum(int(head["deterministic_ones"]) for head in selected)
        )
        summary = self._summary(values, deterministic_ones)
        summary["head_count"] = len(selected)
        return summary

    def finalize(self) -> dict[str, Any]:
        if self.slot_count <= 0:
            raise ValueError("policy diagnostics did not observe any slots")
        head_summaries = [self._head_summary(head) for head in self.heads]
        cache_summary = self._type_summary("cache")
        uav_summary = self._type_summary("uav")
        return {
            "slot_count": int(self.slot_count),
            "action_dim": int(self.action_dim),
            "trigger_heads": head_summaries,
            "cache_trigger_summary": cache_summary,
            "uav_trigger_summary": uav_summary,
            "cache_trigger_mean_p1": float(cache_summary["mean_p1"]),
            "cache_trigger_median_p1": float(cache_summary["median_p1"]),
            "cache_fraction_p1_gt_0_5": float(cache_summary["fraction_p1_gt_0_5"]),
            "cache_deterministic_one_rate": float(
                cache_summary["deterministic_one_rate"]
            ),
            "uav_trigger_mean_p1": float(uav_summary["mean_p1"]),
            "uav_trigger_median_p1": float(uav_summary["median_p1"]),
            "uav_fraction_p1_gt_0_5": float(uav_summary["fraction_p1_gt_0_5"]),
            "uav_deterministic_one_rate": float(
                uav_summary["deterministic_one_rate"]
            ),
        }


def _seed_policy_sampling(seed: int) -> None:
    """Seed every RNG used by policy/environment-side evaluation sampling."""

    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("policy_sampling_seed must be non-negative")
    random.seed(seed_value)
    np.random.seed(seed_value)
    _require_torch()
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def _rollout_action_sha256(rollout: SimpleRollout) -> str:
    actions = np.ascontiguousarray(np.asarray(rollout.actions, dtype=np.int8))
    return hashlib.sha256(actions.tobytes(order="C")).hexdigest()


def _static_zero_payload(
    summary: dict[str, Any],
    rollout: SimpleRollout,
) -> dict[str, Any]:
    return {
        "diagnostic": "static_zero_top_level_triggers",
        "episode_slots": FULL_EPISODE_SLOT_COUNT,
        "origin_start": FULL_EPISODE_ORIGIN_START,
        "origin_end": FULL_EPISODE_ORIGIN_END,
        "static_zero_return": float(summary["total_return"]),
        "static_zero_q_total": float(summary["total_q"]),
        "static_zero_cache_cost": float(summary["total_cache_cost"]),
        "static_zero_movement_cost": float(summary["total_movement_cost"]),
        "cache_trigger_request_rate": float(summary["cache_trigger_request_rate"]),
        "uav_trigger_request_rate": float(summary["uav_trigger_request_rate"]),
        "cache_actual_change_rate": float(summary["cache_change_rate"]),
        "uav_actual_movement_rate": float(summary["uav_movement_rate"]),
        "selected_action_sha256": _rollout_action_sha256(rollout),
    }


def _conditional_operation_names(head_type: str) -> tuple[str, ...]:
    if head_type == "cache":
        return ("NO_CHANGE", "ADD", "DELETE", "SWAP")
    if head_type == "uav":
        return ("STAY", "MOVE")
    raise ValueError(f"unknown conditional head type: {head_type}")


def _uav_state_arrays(context: EnvironmentContext, positions: Any) -> dict[str, np.ndarray]:
    return {
        str(uav_id): np.asarray(positions[uav_id], dtype=np.float64).copy()
        for uav_id in context.action_spec.uav_ids
    }


def _replay_selected_conditional_candidates(
    context: EnvironmentContext,
    rollout: SimpleRollout,
) -> dict[tuple[int, str, int], dict[str, Any]]:
    """Replay only candidate enumeration/application for diagnostic provenance."""

    previous_cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
    previous_uav = _uav_state_arrays(context, context.initial_uav)
    replayed: dict[tuple[int, str, int], dict[str, Any]] = {}
    cache_count = len(context.action_spec.cache_node_ids)
    for transition in rollout.transitions:
        origin = int(transition.origin_index)
        current_cache = previous_cache.copy()
        current_uav = {
            uav_id: value.copy() for uav_id, value in previous_uav.items()
        }
        cache_triggers, uav_triggers = context.action_spec.action_vector_to_triggers(
            transition.action
        )
        cache_records = {
            int(item["head_index"]): item
            for item in transition.cache_conditional_actions
        }
        for head_index, node_id in enumerate(context.action_spec.cache_node_ids):
            if int(cache_triggers[head_index]) == 0:
                continue
            candidates = cache_configuration.enumerate_cache_primitive_actions(
                current_cache,
                head_index,
                context.service_catalog,
                context.cache_nodes,
            )
            expected_count = int(transition.cache_candidate_counts[head_index])
            if len(candidates) != expected_count:
                raise AssertionError(
                    "conditional cache replay candidate count mismatch: "
                    f"origin={origin}, head={head_index}, replay={len(candidates)}, "
                    f"rollout={expected_count}"
                )
            selected_index = int(transition.cache_selected_indices[head_index])
            if selected_index < 0 or selected_index >= len(candidates):
                raise AssertionError("conditional cache replay selected index out of range")
            selected = candidates[selected_index]
            applied = cache_configuration.apply_cache_candidate(
                current_cache,
                head_index,
                selected.after_node_cache,
                context.service_catalog,
                context.cache_nodes,
            )
            recorded = cache_records.get(head_index)
            if recorded is None:
                raise AssertionError(
                    f"missing recorded cache conditional action: origin={origin}, head={head_index}"
                )
            if str(recorded["operation_type"]) != str(selected.operation_type):
                raise AssertionError("conditional cache replay operation mismatch")
            if bool(recorded["actual_changed"]) != bool(applied["actual_changed"]):
                raise AssertionError("conditional cache replay actual-change mismatch")
            replayed[(origin, "cache", head_index)] = {
                "head_index": head_index,
                "head_id": str(node_id),
                "selected_index": selected_index,
                "candidate_count": len(candidates),
                "selected_operation_type": str(selected.operation_type),
                "selected_candidate_before": [
                    int(value) for value in selected.before_node_cache
                ],
                "selected_candidate_after": [
                    int(value) for value in selected.after_node_cache
                ],
                "selected_added_service_ids": [
                    int(value) for value in selected.added_service_ids
                ],
                "selected_removed_service_ids": [
                    int(value) for value in selected.removed_service_ids
                ],
                "selected_actual_changed": bool(applied["actual_changed"]),
                "apply_actual_changed": bool(applied["actual_changed"]),
            }

        uav_records = {
            int(item["head_index"]): item
            for item in transition.uav_conditional_actions
        }
        for uav_index, uav_id in enumerate(context.action_spec.uav_ids):
            if int(uav_triggers[uav_id]) == 0:
                continue
            candidates = uav_configuration.enumerate_uav_primitive_actions(
                current_uav,
                previous_uav,
                uav_id,
                uav_triggers,
                context.cfg,
                context.flight_boundary,
            )
            expected_count = int(transition.uav_candidate_counts[uav_index])
            if len(candidates) != expected_count:
                raise AssertionError(
                    "conditional UAV replay candidate count mismatch: "
                    f"origin={origin}, head={uav_index}, replay={len(candidates)}, "
                    f"rollout={expected_count}"
                )
            selected_index = int(transition.uav_selected_indices[uav_index])
            if selected_index < 0 or selected_index >= len(candidates):
                raise AssertionError("conditional UAV replay selected index out of range")
            selected = candidates[selected_index]
            applied = uav_configuration.apply_uav_candidate(
                current_uav,
                previous_uav,
                uav_id,
                selected.after_position,
                uav_triggers,
                context.cfg,
                context.flight_boundary,
            )
            recorded = uav_records.get(uav_index)
            if recorded is None:
                raise AssertionError(
                    f"missing recorded UAV conditional action: origin={origin}, head={uav_index}"
                )
            if str(recorded["operation_type"]) != str(selected.operation_type):
                raise AssertionError("conditional UAV replay operation mismatch")
            if bool(recorded["actual_moved"]) != bool(applied["actual_moved"]):
                raise AssertionError("conditional UAV replay actual-movement mismatch")
            replayed[(origin, "uav", uav_index)] = {
                "head_index": uav_index,
                "head_id": str(uav_id),
                "selected_index": selected_index,
                "candidate_count": len(candidates),
                "selected_operation_type": str(selected.operation_type),
                "selected_candidate_before": [
                    float(value) for value in selected.before_position
                ],
                "selected_candidate_after": [
                    float(value) for value in selected.after_position
                ],
                "selected_actual_changed": bool(applied["actual_moved"]),
                "selected_actual_moved": bool(applied["actual_moved"]),
                "apply_actual_moved": bool(applied["actual_moved"]),
            }

        if not np.array_equal(current_cache, transition.next_cache_state):
            raise AssertionError(
                f"conditional replay cache state mismatch at origin={origin}"
            )
        for uav_id in context.action_spec.uav_ids:
            if not np.array_equal(
                current_uav[uav_id],
                np.asarray(transition.next_uav_positions[uav_id], dtype=np.float64),
            ):
                raise AssertionError(
                    f"conditional replay UAV state mismatch at origin={origin}, id={uav_id}"
                )
        previous_cache = np.asarray(transition.next_cache_state, dtype=np.int8).copy()
        previous_uav = _uav_state_arrays(context, transition.next_uav_positions)
    return replayed


def _conditional_distribution_diagnostic(
    policy: Any,
    context: EnvironmentContext,
    transition: Any,
    head_type: str,
    head_index: int,
    replay_detail: dict[str, Any],
    conditional_action_mode: str,
) -> dict[str, Any]:
    operation_names = _conditional_operation_names(head_type)
    if head_type == "cache":
        candidate_count = int(transition.cache_candidate_counts[head_index])
        candidate_features = np.asarray(
            transition.cache_candidate_features[head_index, :candidate_count],
            dtype=np.float32,
        )
        candidate_masks = np.asarray(
            transition.cache_candidate_masks[head_index, :candidate_count],
            dtype=bool,
        )
        conditional_context = np.asarray(
            transition.cache_contexts[head_index], dtype=np.float32
        )
        feature_dim = int(policy.cache_feature_dim)
        scorer = policy.cache_candidate_scorer
        head_id = str(context.action_spec.cache_node_ids[head_index])
    elif head_type == "uav":
        candidate_count = int(transition.uav_candidate_counts[head_index])
        candidate_features = np.asarray(
            transition.uav_candidate_features[head_index, :candidate_count],
            dtype=np.float32,
        )
        candidate_masks = np.asarray(
            transition.uav_candidate_masks[head_index, :candidate_count],
            dtype=bool,
        )
        conditional_context = np.asarray(
            transition.uav_contexts[head_index], dtype=np.float32
        )
        feature_dim = int(policy.uav_feature_dim)
        scorer = policy.uav_candidate_scorer
        head_id = str(context.action_spec.uav_ids[head_index])
    else:
        raise ValueError(f"unknown conditional head type: {head_type}")
    if candidate_count <= 0:
        raise AssertionError("triggered conditional head must have candidates")
    if candidate_features.shape[0] != candidate_count or not bool(
        np.all(candidate_masks)
    ):
        raise AssertionError("conditional candidate masks do not match production candidates")
    operation_slice = policy._candidate_operation_slice(head_type)
    with torch.inference_mode():
        raw_scores, single = policy._candidate_scores(
            conditional_context,
            candidate_features,
            feature_dim,
            scorer,
        )
        if not single or tuple(raw_scores.shape) != (1, candidate_count):
            raise AssertionError("conditional diagnostic score shape mismatch")
        balanced_logits = _group_balance_candidate_logits(
            raw_scores.squeeze(0),
            candidate_features,
            candidate_masks,
            operation_slice,
        )
        probabilities = torch.softmax(balanced_logits, dim=-1).detach().cpu().numpy()
    if not bool(np.all(np.isfinite(probabilities))) or not bool(
        np.isclose(float(probabilities.sum()), 1.0, rtol=0.0, atol=1.0e-6)
    ):
        raise AssertionError("conditional diagnostic probabilities are invalid")
    start = 0 if operation_slice.start is None else int(operation_slice.start)
    stop = int(candidate_features.shape[-1]) if operation_slice.stop is None else int(
        operation_slice.stop
    )
    operation_values = candidate_features[:, start:stop]
    if tuple(operation_values.shape) != (candidate_count, len(operation_names)):
        raise AssertionError("conditional operation feature shape mismatch")
    if not bool(
        np.all((operation_values == 0.0) | (operation_values == 1.0))
    ) or not bool(np.all(operation_values.sum(axis=1) == 1.0)):
        raise AssertionError("conditional operation features are not one-hot")
    operation_indices = np.argmax(operation_values, axis=1).astype(np.int64)
    group_candidate_counts = {
        operation: int(np.sum(operation_indices == index))
        for index, operation in enumerate(operation_names)
    }
    group_probabilities = {
        operation: float(probabilities[operation_indices == index].sum())
        for index, operation in enumerate(operation_names)
    }
    candidate_argmax_index = int(np.argmax(probabilities))
    candidate_argmax_operation = operation_names[int(operation_indices[candidate_argmax_index])]
    group_map_index = int(
        np.argmax(np.asarray([group_probabilities[operation] for operation in operation_names]))
    )
    group_map_operation = operation_names[group_map_index]
    group_map_candidates = np.flatnonzero(operation_indices == group_map_index)
    within_group_index = int(
        group_map_candidates[int(np.argmax(probabilities[group_map_candidates]))]
    )
    selected_index = int(replay_detail["selected_index"])
    if (
        conditional_action_mode == "candidate_deterministic"
        and selected_index != candidate_argmax_index
    ):
        raise AssertionError(
            "deterministic conditional selected index is not candidate argmax"
        )
    if (
        conditional_action_mode == "grouped_deterministic"
        and selected_index != within_group_index
    ):
        raise AssertionError(
            "grouped deterministic conditional selected index is not group-MAP "
            "within-group candidate MAP"
        )
    selected_operation_type = str(replay_detail["selected_operation_type"])
    expected_selected_operation = operation_names[int(operation_indices[selected_index])]
    if selected_operation_type != expected_selected_operation:
        raise AssertionError("conditional selected operation does not match features")
    return {
        "head_index": int(head_index),
        "head_id": head_id,
        "head_type": head_type,
        "candidate_count": int(candidate_count),
        "operation_group_candidate_counts": group_candidate_counts,
        "candidate_probabilities": [float(value) for value in probabilities],
        "operation_group_probabilities": group_probabilities,
        "selected_index": selected_index,
        "selected_operation_type": selected_operation_type,
        "selected_candidate_probability": float(probabilities[selected_index]),
        "highest_group_probability_operation": group_map_operation,
        "group_map_operation": group_map_operation,
        "within_group_map_candidate_index": within_group_index,
        "within_group_map_candidate_probability": float(
            probabilities[within_group_index]
        ),
        "candidate_argmax_index": candidate_argmax_index,
        "candidate_argmax_operation": candidate_argmax_operation,
        "conditional_action_mode": conditional_action_mode,
        **replay_detail,
    }


def _conditional_policy_diagnostics(
    policy: Any,
    context: EnvironmentContext,
    rollout: SimpleRollout,
    conditional_action_mode: str,
) -> dict[str, Any]:
    replayed = _replay_selected_conditional_candidates(context, rollout)
    records: list[dict[str, Any]] = []
    cache_count = len(context.action_spec.cache_node_ids)
    uav_count = len(context.action_spec.uav_ids)
    cache_selected_counts = {
        operation: 0 for operation in _conditional_operation_names("cache")
    }
    uav_selected_counts = {
        operation: 0 for operation in _conditional_operation_names("uav")
    }
    cache_candidate_argmax_noop = 0
    cache_group_map_noop = 0
    cache_group_map_differs = 0
    cache_noop_to_active_group = 0
    uav_candidate_argmax_stay = 0
    uav_group_map_stay = 0
    uav_group_map_differs = 0
    uav_noop_to_active_group = 0
    active_cache_noop_count = 0
    active_uav_noop_count = 0
    for transition in rollout.transitions:
        origin = int(transition.origin_index)
        action = np.asarray(transition.action, dtype=np.int8)
        for head_index in range(cache_count):
            if int(action[head_index]) == 0:
                continue
            detail = _conditional_distribution_diagnostic(
                policy,
                context,
                transition,
                "cache",
                head_index,
                replayed[(origin, "cache", head_index)],
                conditional_action_mode,
            )
            records.append({"origin": origin, **detail})
            operation = str(detail["selected_operation_type"])
            cache_selected_counts[operation] += 1
            if str(detail["candidate_argmax_operation"]) == "NO_CHANGE":
                cache_candidate_argmax_noop += 1
            if str(detail["group_map_operation"]) == "NO_CHANGE":
                cache_group_map_noop += 1
            if str(detail["group_map_operation"]) != str(
                detail["candidate_argmax_operation"]
            ):
                cache_group_map_differs += 1
            if (
                str(detail["candidate_argmax_operation"]) == "NO_CHANGE"
                and str(detail["group_map_operation"]) != "NO_CHANGE"
            ):
                cache_noop_to_active_group += 1
            if operation != "NO_CHANGE" and not bool(detail["selected_actual_changed"]):
                active_cache_noop_count += 1
        for head_index in range(uav_count):
            if int(action[cache_count + head_index]) == 0:
                continue
            detail = _conditional_distribution_diagnostic(
                policy,
                context,
                transition,
                "uav",
                head_index,
                replayed[(origin, "uav", head_index)],
                conditional_action_mode,
            )
            records.append({"origin": origin, **detail})
            operation = str(detail["selected_operation_type"])
            uav_selected_counts[operation] += 1
            if str(detail["candidate_argmax_operation"]) == "STAY":
                uav_candidate_argmax_stay += 1
            if str(detail["group_map_operation"]) == "STAY":
                uav_group_map_stay += 1
            if str(detail["group_map_operation"]) != str(
                detail["candidate_argmax_operation"]
            ):
                uav_group_map_differs += 1
            if (
                str(detail["candidate_argmax_operation"]) == "STAY"
                and str(detail["group_map_operation"]) != "STAY"
            ):
                uav_noop_to_active_group += 1
            if operation == "MOVE" and not bool(detail["selected_actual_moved"]):
                active_uav_noop_count += 1

    def rate(numerator: int, denominator: int) -> float:
        return 0.0 if denominator == 0 else float(numerator) / float(denominator)

    cache_triggered = int(sum(cache_selected_counts.values()))
    uav_triggered = int(sum(uav_selected_counts.values()))
    cache_material_mismatch = (
        cache_triggered > 0
        and rate(cache_noop_to_active_group, cache_triggered) >= 0.5
    )
    uav_material_mismatch = (
        uav_triggered > 0
        and rate(uav_noop_to_active_group, uav_triggered) >= 0.5
    )
    group_mismatch = bool(cache_material_mismatch or uav_material_mismatch)
    total_triggered = cache_triggered + uav_triggered
    selected_noop = int(cache_selected_counts["NO_CHANGE"] + uav_selected_counts["STAY"])
    conditional_noop_mode = bool(
        total_triggered > 0 and rate(selected_noop, total_triggered) >= 0.95
    )
    execution_side_noop = bool(active_cache_noop_count + active_uav_noop_count > 0)
    if execution_side_noop:
        final_case = "CASE_C2_EXECUTION_NOOP"
    elif conditional_noop_mode and group_mismatch:
        final_case = (
            "CASE_C1_CONDITIONAL_NOOP_MODE + "
            "CASE_C3_GROUP_BALANCE_DETERMINISTIC_MISMATCH"
        )
    elif conditional_noop_mode:
        final_case = "CASE_C1_CONDITIONAL_NOOP_MODE"
    elif group_mismatch:
        final_case = "CASE_C3_GROUP_BALANCE_DETERMINISTIC_MISMATCH"
    else:
        final_case = "NO_DOMINANT_DIAGNOSTIC_CASE"
    return {
        "triggered_head_records": records,
        "conditional_action_mode": conditional_action_mode,
        "cache_triggered_head_count": cache_triggered,
        "cache_selected_no_change_count": int(cache_selected_counts["NO_CHANGE"]),
        "cache_selected_add_count": int(cache_selected_counts["ADD"]),
        "cache_selected_delete_count": int(cache_selected_counts["DELETE"]),
        "cache_selected_swap_count": int(cache_selected_counts["SWAP"]),
        "cache_selected_no_change_rate": rate(
            int(cache_selected_counts["NO_CHANGE"]), cache_triggered
        ),
        "cache_selected_add_rate": rate(int(cache_selected_counts["ADD"]), cache_triggered),
        "cache_selected_delete_rate": rate(
            int(cache_selected_counts["DELETE"]), cache_triggered
        ),
        "cache_selected_swap_rate": rate(
            int(cache_selected_counts["SWAP"]), cache_triggered
        ),
        "uav_triggered_head_count": uav_triggered,
        "uav_selected_stay_count": int(uav_selected_counts["STAY"]),
        "uav_selected_move_count": int(uav_selected_counts["MOVE"]),
        "uav_selected_stay_rate": rate(int(uav_selected_counts["STAY"]), uav_triggered),
        "uav_selected_move_rate": rate(int(uav_selected_counts["MOVE"]), uav_triggered),
        "cache_candidate_argmax_no_change_rate": rate(
            cache_candidate_argmax_noop, cache_triggered
        ),
        "cache_group_map_no_change_rate": rate(cache_group_map_noop, cache_triggered),
        "cache_group_map_differs_from_candidate_argmax_rate": rate(
            cache_group_map_differs, cache_triggered
        ),
        "cache_candidate_argmax_noop_to_active_group_rate": rate(
            cache_noop_to_active_group, cache_triggered
        ),
        "uav_candidate_argmax_stay_rate": rate(
            uav_candidate_argmax_stay, uav_triggered
        ),
        "uav_group_map_stay_rate": rate(uav_group_map_stay, uav_triggered),
        "uav_group_map_differs_from_candidate_argmax_rate": rate(
            uav_group_map_differs, uav_triggered
        ),
        "uav_candidate_argmax_stay_to_active_group_rate": rate(
            uav_noop_to_active_group, uav_triggered
        ),
        "active_cache_operation_selected_but_no_change_count": int(
            active_cache_noop_count
        ),
        "active_uav_operation_selected_but_no_move_count": int(active_uav_noop_count),
        "execution_side_noop": execution_side_noop,
        "group_balance_deterministic_decoding_mismatch": group_mismatch,
        "conditional_noop_mode": conditional_noop_mode,
        "final_diagnostic_case": final_case,
    }


def _environment_trajectory_equivalent_to_static_zero(
    rollout: SimpleRollout,
    static_zero_rollout: SimpleRollout,
    context: EnvironmentContext,
) -> bool:
    if len(rollout.transitions) != len(static_zero_rollout.transitions):
        return False
    scalar_fields = (
        "reward",
        "actual_task_utility",
        "cache_cost",
        "movement_cost",
    )
    for first, second in zip(
        rollout.transitions,
        static_zero_rollout.transitions,
        strict=True,
    ):
        if int(first.origin_index) != int(second.origin_index) or int(first.slot) != int(
            second.slot
        ):
            return False
        for field in scalar_fields:
            if float(getattr(first, field)) != float(getattr(second, field)):
                return False
        if int(first.actual_cache_update_count) != int(second.actual_cache_update_count):
            return False
        if int(first.actual_uav_movement_count) != int(second.actual_uav_movement_count):
            return False
        if not np.array_equal(first.next_cache_state, second.next_cache_state):
            return False
        for uav_id in context.action_spec.uav_ids:
            if not np.array_equal(
                np.asarray(first.next_uav_positions[uav_id], dtype=np.float64),
                np.asarray(second.next_uav_positions[uav_id], dtype=np.float64),
            ):
                return False
    return True


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


def _conditional_selection_summary(rollout: SimpleRollout) -> dict[str, Any]:
    """Summarize the primitive choices recorded by the production rollout."""

    cache_operations = ("NO_CHANGE", "ADD", "DELETE", "SWAP")
    uav_operations = ("STAY", "MOVE")
    cache_selected = {operation: 0 for operation in cache_operations}
    uav_selected = {operation: 0 for operation in uav_operations}
    for transition in rollout.transitions:
        for action in transition.cache_conditional_actions:
            operation = str(action["operation_type"])
            if operation not in cache_selected:
                raise ValueError(f"unknown cache conditional operation: {operation}")
            cache_selected[operation] += 1
        for action in transition.uav_conditional_actions:
            operation = str(action["operation_type"])
            if operation not in uav_selected:
                raise ValueError(f"unknown UAV conditional operation: {operation}")
            uav_selected[operation] += 1
    return {
        "cache_selected_operation_counts": cache_selected,
        "uav_selected_operation_counts": uav_selected,
        "cache_selected_no_change_count": int(cache_selected["NO_CHANGE"]),
        "cache_selected_add_count": int(cache_selected["ADD"]),
        "cache_selected_delete_count": int(cache_selected["DELETE"]),
        "cache_selected_swap_count": int(cache_selected["SWAP"]),
        "uav_selected_stay_count": int(uav_selected["STAY"]),
        "uav_selected_move_count": int(uav_selected["MOVE"]),
    }


def _full_episode_summary(
    records: list[dict[str, Any]],
    cache_count: int,
    uav_count: int,
) -> dict[str, Any]:
    if len(records) != FULL_EPISODE_SLOT_COUNT:
        raise ValueError("full-episode summary requires 300 slot records")
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
    summary: dict[str, Any] = {
        "data_protocol": FULL_EPISODE_PROTOCOL,
        "origin_start": FULL_EPISODE_ORIGIN_START,
        "origin_end": FULL_EPISODE_ORIGIN_END,
        "episode_slots": FULL_EPISODE_SLOT_COUNT,
        "total_return": float(rewards.sum()),
        "mean_slot_reward": float(rewards.mean()),
        "total_q": float(q_values.sum()),
        "mean_q": float(q_values.mean()),
        "total_cache_cost": float(cache_costs.sum()),
        "total_movement_cost": float(movement_costs.sum()),
        "cache_trigger_rate": _rate(
            cache_triggers, FULL_EPISODE_SLOT_COUNT * cache_count
        ),
        "cache_trigger_request_rate": _rate(
            cache_triggers, FULL_EPISODE_SLOT_COUNT * cache_count
        ),
        "cache_change_rate": _rate(
            cache_changes, FULL_EPISODE_SLOT_COUNT * cache_count
        ),
        "cache_change_given_trigger_rate": (
            None if cache_triggers == 0 else _rate(cache_changes, cache_triggers)
        ),
        "uav_trigger_rate": _rate(
            uav_triggers, FULL_EPISODE_SLOT_COUNT * uav_count
        ),
        "uav_trigger_request_rate": _rate(
            uav_triggers, FULL_EPISODE_SLOT_COUNT * uav_count
        ),
        "uav_movement_rate": _rate(
            uav_movements, FULL_EPISODE_SLOT_COUNT * uav_count
        ),
        "uav_movement_given_trigger_rate": (
            None if uav_triggers == 0 else _rate(uav_movements, uav_triggers)
        ),
        "segments": [],
    }
    base_segment_size = FULL_EPISODE_SLOT_COUNT // 4
    segment_lengths = [base_segment_size] * 3 + [
        FULL_EPISODE_SLOT_COUNT - (base_segment_size * 3)
    ]
    segment_start = FULL_EPISODE_ORIGIN_START
    segments: list[dict[str, Any]] = []
    for segment_length in segment_lengths:
        segment_end = segment_start + segment_length - 1
        segment_records = [
            record
            for record in records
            if segment_start <= int(record["origin"]) <= segment_end
        ]
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
        segment_start = segment_end + 1
    summary["segments"] = segments
    return summary


def evaluate_full_episode(
    checkpoint_path: str | Path,
    run_dir: str | Path,
    config_path: str | Path = FULL_EPISODE_CONFIG_PATH,
    device: str = "cpu",
    realization_dir: str | Path | None = None,
    record_policy_diagnostics: bool = False,
    policy_action_mode: str = "deterministic",
    policy_sampling_seed: int | None = None,
    static_zero_diagnostic: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one specified full-episode input realization without training."""

    _require_torch()
    requested_policy_action_mode = str(policy_action_mode)
    policy_action_mode = normalize_policy_action_mode(policy_action_mode)
    if policy_sampling_seed is not None:
        policy_sampling_seed = int(policy_sampling_seed)
    project_root = project_config.find_project_root()
    resolved_config = project_config.resolve_path(project_root, config_path)
    resolved_checkpoint = project_config.resolve_path(project_root, checkpoint_path)
    output_dir = _resolve_output_dir(project_root, run_dir)
    ensure_fresh_evaluation_dir(output_dir)

    config_sha256 = _sha256(resolved_config)
    context = _build_full_episode_context(
        resolved_config,
        realization_dir=realization_dir,
    )
    training_git_sha = _load_full_episode_training_provenance(
        resolved_checkpoint,
        config_sha256,
    )
    evaluation_git_sha, evaluation_git_dirty = _git_identity(project_root)
    evaluation_device = torch.device(device)
    steps = _prepare_full_episode_steps(context)
    policy, checkpoint_payload = _load_frozen_v6_policy(
        context,
        resolved_checkpoint,
        config_sha256,
        evaluation_device,
        expected_completed_epochs=None,
        expected_protocol=FULL_EPISODE_PROTOCOL,
    )
    if policy_sampling_seed is not None:
        _seed_policy_sampling(policy_sampling_seed)
    parameter_snapshot = [parameter.detach().clone() for parameter in policy.parameters()]
    popart_snapshot = policy.popart_statistics()
    diagnostic_accumulator = (
        _PolicyDiagnosticAccumulator(context)
        if record_policy_diagnostics
        else None
    )
    deterministic = policy_action_mode != "stochastic"
    with torch.inference_mode():
        rollout = collect_rollout(
            policy,
            context,
            steps,
            initial_cache=np.asarray(context.initial_cache, dtype=np.int8).copy(),
            initial_uav=_copy_uav_positions(context.initial_uav),
            terminal_origin_index=FULL_EPISODE_ORIGIN_END,
            deterministic=deterministic,
            conditional_action_mode=policy_action_mode,
            policy_diagnostic_callback=(
                None
                if diagnostic_accumulator is None
                else diagnostic_accumulator.observe
            ),
        )
    if not bool(rollout.dones[-1]) or bool(np.any(rollout.dones[:-1])):
        raise ValueError(
            "full-episode evaluation must terminate only at origin 299"
        )
    static_zero_rollout: SimpleRollout | None = None
    if static_zero_diagnostic:
        with torch.inference_mode():
            static_zero_rollout = collect_rollout(
                policy,
                context,
                steps,
                initial_cache=np.asarray(context.initial_cache, dtype=np.int8).copy(),
                initial_uav=_copy_uav_positions(context.initial_uav),
                terminal_origin_index=FULL_EPISODE_ORIGIN_END,
                deterministic=True,
                conditional_action_mode="candidate_deterministic",
                action_transform=lambda action: np.zeros_like(action),
            )
        if not bool(static_zero_rollout.dones[-1]) or bool(
            np.any(static_zero_rollout.dones[:-1])
        ):
            raise ValueError(
                "static-zero full-episode diagnostic must terminate only at origin 299"
            )
    policy_parameters_updated = any(
        not torch.equal(before, after.detach())
        for before, after in zip(parameter_snapshot, policy.parameters(), strict=True)
    )
    if policy_parameters_updated:
        raise RuntimeError("full-episode evaluation modified policy parameters")
    popart_statistics_updated = policy.popart_statistics() != popart_snapshot
    if popart_statistics_updated:
        raise RuntimeError("full-episode evaluation modified PopArt statistics")
    records = _slot_records_for_range(
        rollout,
        len(context.action_spec.cache_node_ids),
        len(context.action_spec.uav_ids),
        origin_start=FULL_EPISODE_ORIGIN_START,
        origin_end=FULL_EPISODE_ORIGIN_END,
        label="full-episode",
    )
    summary = _full_episode_summary(
        records,
        len(context.action_spec.cache_node_ids),
        len(context.action_spec.uav_ids),
    )
    summary["selected_action_sha256"] = _rollout_action_sha256(rollout)
    summary["policy_action_mode"] = policy_action_mode
    summary["policy_action_mode_requested"] = requested_policy_action_mode
    summary["policy_sampling_seed"] = policy_sampling_seed
    for key, value in _conditional_selection_summary(rollout).items():
        summary[key] = value
    conditional_diagnostics: dict[str, Any] | None = None
    if diagnostic_accumulator is not None:
        conditional_diagnostics = _conditional_policy_diagnostics(
            policy,
            context,
            rollout,
            policy_action_mode,
        )
        for key, value in conditional_diagnostics.items():
            if key != "triggered_head_records":
                summary[key] = value
    static_zero_payload: dict[str, Any] | None = None
    static_zero_summary: dict[str, Any] | None = None
    if static_zero_rollout is not None:
        static_zero_records = _slot_records_for_range(
            static_zero_rollout,
            len(context.action_spec.cache_node_ids),
            len(context.action_spec.uav_ids),
            origin_start=FULL_EPISODE_ORIGIN_START,
            origin_end=FULL_EPISODE_ORIGIN_END,
            label="static-zero full-episode",
        )
        static_zero_summary = _full_episode_summary(
            static_zero_records,
            len(context.action_spec.cache_node_ids),
            len(context.action_spec.uav_ids),
        )
        static_zero_payload = _static_zero_payload(
            static_zero_summary,
            static_zero_rollout,
        )
        static_zero_payload[
            "environment_trajectory_equivalent_to_static_zero"
        ] = _environment_trajectory_equivalent_to_static_zero(
            rollout,
            static_zero_rollout,
            context,
        )
        static_zero_payload["deterministic_p50_equivalent"] = bool(
            static_zero_payload["environment_trajectory_equivalent_to_static_zero"]
        )
        summary["static_zero_diagnostic"] = static_zero_payload
        summary["deterministic_p50_equivalent_to_static_zero"] = bool(
            static_zero_payload["deterministic_p50_equivalent"]
        )
        summary["environment_trajectory_equivalent_to_static_zero"] = bool(
            static_zero_payload["environment_trajectory_equivalent_to_static_zero"]
        )
    manifest: dict[str, Any] = {
        "experiment_type": "full_episode_evaluation",
        "data_protocol": FULL_EPISODE_PROTOCOL,
        "training_git_sha": training_git_sha,
        "evaluation_git_sha": evaluation_git_sha,
        "evaluation_git_dirty": evaluation_git_dirty,
        "config_path": str(resolved_config),
        "config_sha256": config_sha256,
        "checkpoint_path": str(resolved_checkpoint),
        "checkpoint_sha256": _sha256(resolved_checkpoint),
        "checkpoint_completed_epochs": int(
            checkpoint_payload.get(
                "completed_epochs",
                checkpoint_payload.get("completed_episodes", -1),
            )
        ),
        "policy_schema": EXPECTED_POLICY_SCHEMA,
        "episode_slots": FULL_EPISODE_SLOT_COUNT,
        "origin_start": FULL_EPISODE_ORIGIN_START,
        "origin_end": FULL_EPISODE_ORIGIN_END,
        "rollout_steps": FULL_EPISODE_SLOT_COUNT,
        "num_minibatches": int(
            project_config.get_ppo_data_protocol(context.cfg)["num_minibatches"]
        ),
        "minibatch_size": int(context.minibatch_size),
        "terminal_origin": FULL_EPISODE_ORIGIN_END,
        "realization_directory": (
            None
            if context.realization is None
            else str(context.realization.directory)
        ),
        "policy_action_mode": policy_action_mode,
        "policy_action_mode_requested": requested_policy_action_mode,
        "policy_sampling_seed": policy_sampling_seed,
        "policy_sampling_seed_derivation": (
            "caller-supplied deterministic policy_sampling_seed"
            if policy_sampling_seed is not None
            else None
        ),
        "deterministic": bool(deterministic),
        "selected_action_sha256": _rollout_action_sha256(rollout),
        "policy_diagnostics_recorded": bool(record_policy_diagnostics),
        "conditional_diagnostics_recorded": bool(conditional_diagnostics is not None),
        "static_zero_diagnostic": bool(static_zero_diagnostic),
        "policy_parameters_updated": bool(policy_parameters_updated),
        "popart_statistics_updated": bool(popart_statistics_updated),
        "optimizer_steps": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "full_episode_manifest.json", manifest)
    _write_json(output_dir / "full_episode_summary.json", summary)
    if diagnostic_accumulator is not None:
        diagnostics = diagnostic_accumulator.finalize()
        diagnostics["policy_action_mode"] = policy_action_mode
        diagnostics["policy_action_mode_requested"] = requested_policy_action_mode
        diagnostics["policy_sampling_seed"] = policy_sampling_seed
        if conditional_diagnostics is not None:
            diagnostics["conditional_diagnostics"] = conditional_diagnostics
        _write_json(output_dir / "policy_diagnostics.json", diagnostics)
    if static_zero_payload is not None:
        _write_json(output_dir / "static_zero_summary.json", static_zero_payload)
    with (output_dir / "full_episode_slot_metrics.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen PPO V6 policy without optimizer updates."
    )
    parser.add_argument(
        "--mode",
        choices=("heldout", "full_episode"),
        default="heldout",
        help="legacy held-out 240..299 or one full 0..299 episode",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=str(FORMAL_CHECKPOINT_PATH))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--realization-dir",
        default=None,
        help="full-episode realization directory; omitted uses configured data",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--record-policy-diagnostics",
        action="store_true",
        help="write compact per-trigger-head policy distribution diagnostics",
    )
    parser.add_argument(
        "--policy-action-mode",
        choices=("deterministic", *POLICY_ACTION_MODES),
        default="deterministic",
        help=(
            "full-episode action decoder: deterministic is the legacy alias for "
            "candidate_deterministic"
        ),
    )
    parser.add_argument(
        "--policy-sampling-seed",
        type=int,
        default=None,
        help="seed Python, NumPy, and torch RNGs before policy rollout sampling",
    )
    parser.add_argument(
        "--static-zero-diagnostic",
        action="store_true",
        help="also run a same-realization all-zero top-level trigger diagnostic",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "full_episode":
        evaluate_full_episode(
            checkpoint_path=args.checkpoint,
            run_dir=args.run_dir,
            config_path=args.config or FULL_EPISODE_CONFIG_PATH,
            device=args.device,
            realization_dir=args.realization_dir,
            record_policy_diagnostics=args.record_policy_diagnostics,
            policy_action_mode=args.policy_action_mode,
            policy_sampling_seed=args.policy_sampling_seed,
            static_zero_diagnostic=args.static_zero_diagnostic,
        )
    else:
        evaluate_heldout(
            config_path=args.config or FORMAL_CONFIG_PATH,
            checkpoint_path=args.checkpoint,
            run_dir=args.run_dir,
            device=args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
