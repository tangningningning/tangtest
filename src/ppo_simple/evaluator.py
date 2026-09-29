"""使用冻结策略运行 proposed_joint 正式评价。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - import-only environments
    torch = None  # type: ignore[assignment]

from common import project_config

from .evaluation_metrics import (
    aggregate_episode_task_metrics,
    episode_metric_statistics,
    execution_ratio_statistics,
)
from .interface import _copy_uav_positions
from .reporting import resolve_checkpoint_reference
from .rollout import collect_rollout
from .trainer import (
    PPOTrainer,
    _assert_realization_context_compatible,
    aligned_train_steps,
    load_environment_context,
)


PROPOSED_POLICY_SCHEMA = "adaptive_joint_cache_uav_ppo"
PROPOSED_CONTROL_MODE = "adaptive_joint_cache_uav_reconfiguration"
PROPOSED_CONFIG_PATH = "configs/proposed/system_config.toml"
FULL_EPISODE_PROTOCOL = project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED
FULL_EPISODE_SLOTS = 300
FULL_EPISODE_ORIGIN_START = 0
FULL_EPISODE_ORIGIN_END = 299
FULL_EPISODE_TERMINAL_ORIGIN = 299
FULL_EPISODE_ROLLOUT_STEPS = 300
DEFAULT_ENVIRONMENT_COUNT = 20
DEFAULT_POLICY_REPEATS = 5

EPISODE_METRIC_NAMES = (
    "total_return",
    "total_q",
    "total_cache_cost",
    "total_movement_cost",
    "cache_change_rate",
    "uav_movement_rate",
    "uav_stay_rate",
    "uav_move_rate",
    "total_tasks",
    "mean_task_delay_s",
    "mean_vehicle_energy_j",
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("proposed joint evaluation requires PyTorch.")


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_fresh_output(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"evaluation output is not a directory: {path}")
    if path.is_dir() and any(path.iterdir()):
        raise FileExistsError(
            "evaluation output already contains artifacts; use a new --run-dir: "
            f"{path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def _validate_proposed_context(context: Any) -> dict[str, Any]:
    if str(context.policy_schema) != PROPOSED_POLICY_SCHEMA:
        raise ValueError(
            "proposed joint evaluator requires policy_schema="
            f"{PROPOSED_POLICY_SCHEMA}, got {context.policy_schema}"
        )
    if str(context.configuration_control_mode) != PROPOSED_CONTROL_MODE:
        raise ValueError(
            "proposed joint evaluator requires configuration_control_mode="
            f"{PROPOSED_CONTROL_MODE}, got {context.configuration_control_mode}"
        )
    protocol = project_config.get_ppo_data_protocol(context.cfg)
    expected = {
        "protocol": FULL_EPISODE_PROTOCOL,
        "physical_episode_slots": FULL_EPISODE_SLOTS,
        "episode_slots": FULL_EPISODE_SLOTS,
        "train_origin_start": FULL_EPISODE_ORIGIN_START,
        "train_origin_end": FULL_EPISODE_ORIGIN_END,
        "terminal_origin": FULL_EPISODE_TERMINAL_ORIGIN,
        "rollout_steps": FULL_EPISODE_ROLLOUT_STEPS,
        "test_origin_start": None,
        "test_origin_end": None,
    }
    for key, expected_value in expected.items():
        actual = protocol.get(key)
        if actual != expected_value:
            raise ValueError(
                "proposed joint evaluator protocol mismatch for "
                f"{key}: {actual!r} != {expected_value!r}"
            )
    return protocol


def _load_checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    _require_torch()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"proposed checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("proposed checkpoint payload must be a mapping")
    if payload.get("policy_schema") != PROPOSED_POLICY_SCHEMA:
        raise ValueError(
            "proposed checkpoint policy_schema mismatch: "
            f"{payload.get('policy_schema')!r} != {PROPOSED_POLICY_SCHEMA!r}"
        )
    return payload


def _seed_policy_sampling(seed: int) -> None:
    _require_torch()
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _load_proposed_trainer(
    context: Any,
    checkpoint: Path,
    config_hash: str,
    checkpoint_metadata: dict[str, Any],
    device: str,
) -> PPOTrainer:
    trainer = PPOTrainer(context, device=device)
    if trainer.policy.architecture_id != context.policy_schema:
        raise TypeError(
            "Trainer policy factory did not construct the configured proposed schema: "
            f"{trainer.policy.architecture_id!r} != {context.policy_schema!r}"
        )
    if checkpoint_metadata.get("configuration_control_mode") != (
        context.configuration_control_mode
    ):
        raise ValueError(
            "proposed checkpoint configuration_control_mode mismatch: "
            f"{checkpoint_metadata.get('configuration_control_mode')!r} != "
            f"{context.configuration_control_mode!r}"
        )
    trainer.load_checkpoint(checkpoint, config_hash)
    if trainer.policy.architecture_id != PROPOSED_POLICY_SCHEMA:
        raise TypeError("proposed checkpoint restored an unexpected policy schema")
    trainer.policy.eval()
    return trainer


def _rollout_summary(
    context: Any,
    trainer: PPOTrainer,
    rollout: Any,
    *,
    episode_seed: int,
    repeat_index: int,
    policy_action_mode: str,
    policy_sampling_seed: int | None,
    checkpoint: Path,
) -> dict[str, Any]:
    if len(rollout.transitions) != FULL_EPISODE_SLOTS:
        raise ValueError(
            "proposed evaluation rollout must contain exactly "
            f"{FULL_EPISODE_SLOTS} transitions"
        )
    if not bool(rollout.dones[-1]) or bool(np.any(rollout.dones[:-1])):
        raise ValueError("proposed evaluation must terminate only at origin 299")

    task_metrics = aggregate_episode_task_metrics(rollout.transitions)
    cache_head_count = len(context.action_spec.cache_node_ids)
    uav_head_count = len(context.action_spec.uav_ids)
    total_cache_heads = FULL_EPISODE_SLOTS * cache_head_count
    total_uav_heads = FULL_EPISODE_SLOTS * uav_head_count
    if total_cache_heads <= 0 or total_uav_heads <= 0:
        raise ValueError("proposed evaluation requires Cache and UAV action heads")

    uav_operation_actions = rollout.uav_operation_actions
    if uav_operation_actions is None:
        raise ValueError("proposed evaluation rollout is missing UAV operation actions")
    operation_order = tuple(getattr(trainer.policy, "uav_operation_order", ()))
    if "STAY" not in operation_order or "MOVE" not in operation_order:
        raise ValueError("proposed policy does not expose the STAY/MOVE operation order")
    uav_operation_array = np.asarray(uav_operation_actions, dtype=np.int64)
    expected_shape = (FULL_EPISODE_SLOTS, uav_head_count)
    if uav_operation_array.shape != expected_shape:
        raise ValueError(
            "proposed UAV operation action shape mismatch: "
            f"{uav_operation_array.shape} != {expected_shape}"
        )
    move_index = operation_order.index("MOVE")
    stay_index = operation_order.index("STAY")
    uav_move_rate = float(np.mean(uav_operation_array == move_index))
    uav_stay_rate = float(np.mean(uav_operation_array == stay_index))
    actual_uav_movements = int(
        sum(int(item.actual_uav_movement_count) for item in rollout.transitions)
    )
    actual_cache_updates = int(
        sum(int(item.actual_cache_update_count) for item in rollout.transitions)
    )

    summary: dict[str, Any] = {
        "episode_seed": int(episode_seed),
        "repeat_index": int(repeat_index),
        "policy_sampling_seed": (
            None if policy_sampling_seed is None else int(policy_sampling_seed)
        ),
        "policy_action_mode": str(policy_action_mode),
        "policy_schema": str(context.policy_schema),
        "configuration_control_mode": str(context.configuration_control_mode),
        "data_protocol": str(context.data_protocol),
        "episode_slots": int(context.episode_slots),
        "origin_start": int(context.train_origin_start),
        "origin_end": int(context.train_origin_end),
        "terminal_origin": int(context.terminal_origin),
        "rollout_steps": int(context.configured_rollout_steps or 0),
        "checkpoint_path": str(checkpoint),
        "checkpoint_policy_schema": str(trainer.policy.architecture_id),
        "num_transitions": len(rollout.transitions),
        "total_return": float(np.asarray(rollout.rewards, dtype=np.float64).sum()),
        "total_q": float(
            sum(float(item.actual_task_utility) for item in rollout.transitions)
        ),
        "total_cache_cost": float(
            sum(float(item.cache_cost) for item in rollout.transitions)
        ),
        "total_movement_cost": float(
            sum(float(item.movement_cost) for item in rollout.transitions)
        ),
        "cache_change_rate": float(actual_cache_updates) / float(total_cache_heads),
        "uav_movement_rate": float(actual_uav_movements) / float(total_uav_heads),
        "uav_stay_rate": uav_stay_rate,
        "uav_move_rate": uav_move_rate,
        **task_metrics,
    }
    finite_values = (
        "total_return",
        "total_q",
        "total_cache_cost",
        "total_movement_cost",
        "cache_change_rate",
        "uav_movement_rate",
        "uav_stay_rate",
        "uav_move_rate",
    )
    if not all(math.isfinite(float(summary[name])) for name in finite_values):
        raise FloatingPointError("proposed evaluation produced a non-finite metric")
    return summary


def _mean_environment_summary(
    episode_seed: int,
    repeat_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not repeat_summaries:
        raise ValueError("an environment requires at least one policy repeat")
    result: dict[str, Any] = {
        "episode_seed": int(episode_seed),
        "repeat_count": len(repeat_summaries),
        "policy_schema": repeat_summaries[0]["policy_schema"],
        "data_protocol": repeat_summaries[0]["data_protocol"],
        "episode_slots": repeat_summaries[0]["episode_slots"],
        "origin_start": repeat_summaries[0]["origin_start"],
        "origin_end": repeat_summaries[0]["origin_end"],
        "terminal_origin": repeat_summaries[0]["terminal_origin"],
        "rollout_steps": repeat_summaries[0]["rollout_steps"],
    }
    for name in EPISODE_METRIC_NAMES:
        values = [summary.get(name) for summary in repeat_summaries]
        result[name] = (
            None
            if any(value is None for value in values)
            else float(np.mean(np.asarray(values, dtype=np.float64)))
        )
    result["execution_counts"] = {
        name: float(
            np.mean(
                [
                    float(summary["execution_counts"].get(name, 0))
                    for summary in repeat_summaries
                ]
            )
        )
        for name in ("LOCAL", "SV", "RSU", "UAV", "CLOUD")
    }
    result["execution_ratios"] = {
        name: float(
            np.mean(
                [
                    float(summary["execution_ratios"].get(name, 0.0))
                    for summary in repeat_summaries
                ]
            )
        )
        for name in ("LOCAL", "SV", "RSU", "UAV", "CLOUD")
    }
    return result


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
    policy_action_mode: str = "stochastic",
    policy_sampling_seed: int | None = None,
) -> dict[str, Any]:
    _require_torch()
    if int(episodes) < 2:
        raise ValueError("episodes must be at least 2 for environment statistics")
    if int(repeats) <= 0:
        raise ValueError("repeats must be positive")
    if policy_action_mode not in {"deterministic", "stochastic"}:
        raise ValueError("policy_action_mode must be deterministic or stochastic")

    project_root = project_config.find_project_root()
    config_file = _resolve_path(project_root, config_path)
    checkpoint = resolve_checkpoint_reference(
        _resolve_path(project_root, checkpoint_path)
    ).resolve(strict=True)
    pool_path = _resolve_path(project_root, episode_pool)
    output_path = _resolve_path(project_root, run_dir)
    if not config_file.is_file():
        raise FileNotFoundError(f"proposed config is not readable: {config_file}")
    if not pool_path.is_dir():
        raise FileNotFoundError(f"episode pool does not exist: {pool_path}")
    _ensure_fresh_output(output_path)

    checkpoint_metadata = _load_checkpoint_metadata(checkpoint)
    config_hash = _sha256(config_file)
    reference_context = load_environment_context(str(config_file))
    protocol = _validate_proposed_context(reference_context)
    environment_means: list[dict[str, Any]] = []
    repeat_summaries: list[dict[str, Any]] = []

    for environment_index in range(int(episodes)):
        episode_seed = int(seed_start) + environment_index
        realization_dir = pool_path / f"episode_{episode_seed}"
        if not realization_dir.is_dir():
            raise FileNotFoundError(
                f"episode pool realization is missing: {realization_dir}"
            )
        context = load_environment_context(
            str(config_file),
            realization_dir=realization_dir,
        )
        _validate_proposed_context(context)
        _assert_realization_context_compatible(reference_context, context)
        steps = aligned_train_steps(context)
        if len(steps) != FULL_EPISODE_SLOTS:
            raise ValueError("proposed evaluation aligned steps are not a full 300-slot episode")

        trainer = _load_proposed_trainer(
            context,
            checkpoint,
            config_hash,
            checkpoint_metadata,
            device,
        )
        parameter_snapshot = [
            parameter.detach().clone() for parameter in trainer.policy.parameters()
        ]
        popart_snapshot = trainer.policy.popart_statistics()
        optimizer_steps_snapshot = (
            int(trainer.optimizer_steps),
            int(trainer.actor_optimizer_steps),
            int(trainer.critic_optimizer_steps),
        )
        environment_repeat_summaries: list[dict[str, Any]] = []
        for repeat_index in range(int(repeats)):
            repeat_seed = (
                None
                if policy_sampling_seed is None
                else int(policy_sampling_seed)
                + environment_index * int(repeats)
                + repeat_index
            )
            if repeat_seed is not None:
                _seed_policy_sampling(repeat_seed)
            with torch.inference_mode():
                rollout = collect_rollout(
                    trainer.policy,
                    context,
                    steps,
                    initial_cache=np.asarray(context.initial_cache, dtype=np.int8).copy(),
                    initial_uav=_copy_uav_positions(context.initial_uav),
                    terminal_origin_index=FULL_EPISODE_TERMINAL_ORIGIN,
                    deterministic=policy_action_mode == "deterministic",
                )
            summary = _rollout_summary(
                context,
                trainer,
                rollout,
                episode_seed=episode_seed,
                repeat_index=repeat_index,
                policy_action_mode=policy_action_mode,
                policy_sampling_seed=repeat_seed,
                checkpoint=checkpoint,
            )
            environment_repeat_summaries.append(summary)
            repeat_summaries.append(summary)

        if any(
            not torch.equal(before, after.detach())
            for before, after in zip(
                parameter_snapshot,
                trainer.policy.parameters(),
                strict=True,
            )
        ):
            raise RuntimeError("proposed evaluation modified policy parameters")
        if trainer.policy.popart_statistics() != popart_snapshot:
            raise RuntimeError("proposed evaluation modified PopArt statistics")
        if (
            int(trainer.optimizer_steps),
            int(trainer.actor_optimizer_steps),
            int(trainer.critic_optimizer_steps),
        ) != optimizer_steps_snapshot:
            raise RuntimeError("proposed evaluation performed an optimizer step")

        environment_mean = _mean_environment_summary(
            episode_seed,
            environment_repeat_summaries,
        )
        environment_means.append(environment_mean)
        environment_output = output_path / f"episode_{episode_seed}"
        environment_output.mkdir(parents=True, exist_ok=False)
        with (environment_output / "proposed_repeat_summaries.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(
                environment_repeat_summaries,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")

    available_metric_names = tuple(
        name
        for name in EPISODE_METRIC_NAMES
        if all(summary.get(name) is not None for summary in environment_means)
    )
    aggregate = {
        "experiment_type": "proposed_joint_episode_pool_evaluation",
        "config_path": str(config_file),
        "config_sha256": config_hash,
        "checkpoint_path": str(checkpoint),
        "checkpoint_policy_schema": checkpoint_metadata.get("policy_schema"),
        "policy_schema": PROPOSED_POLICY_SCHEMA,
        "configuration_control_mode": PROPOSED_CONTROL_MODE,
        "data_protocol": str(protocol["protocol"]),
        "episode_pool": str(pool_path),
        "episode_count": len(environment_means),
        "seed_start": int(seed_start),
        "episodes": int(episodes),
        "repeats": int(repeats),
        "policy_action_mode": policy_action_mode,
        "policy_sampling_seed": (
            None if policy_sampling_seed is None else int(policy_sampling_seed)
        ),
        "statistical_unit": "environment_realization",
        "repeat_aggregation": "within_environment_mean_then_across_environment_statistics",
        "aggregate_statistics": episode_metric_statistics(
            environment_means,
            available_metric_names,
        ),
        "execution_ratio_statistics": execution_ratio_statistics(environment_means),
        "environment_means": environment_means,
        "repeat_summaries": repeat_summaries,
    }
    with (output_path / "proposed_episode_pool_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(aggregate, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen proposed joint PPO policy on an episode pool"
    )
    parser.add_argument("--config", default=PROPOSED_CONFIG_PATH)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episode-pool", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_ENVIRONMENT_COUNT)
    parser.add_argument("--repeats", type=int, default=DEFAULT_POLICY_REPEATS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--policy-action-mode",
        choices=("deterministic", "stochastic"),
        default="stochastic",
    )
    parser.add_argument(
        "--policy-sampling-seed",
        type=int,
        default=None,
        help="optional base seed; repeat seeds are derived from it",
    )
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
