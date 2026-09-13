# -*- coding: utf-8 -*-
"""Evaluate one frozen PPO checkpoint over a fixed development episode pool."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common import project_config
from ppo_simple import evaluator
from generation.generate_episode_pool import _validate_complete_realization
from generation import generate_episode_realization as realization_generator


DEVELOPMENT_SEED_START = 30001
DEVELOPMENT_SEED_END = 30020
DEVELOPMENT_COUNT = 20
DEVELOPMENT_EVALUATION_ROLE = "development_checkpoint_selection"
P50_COMPLETED_EPOCHS = 50
STUDENT_T_CRITICAL_95_DF19 = 2.093024054408263
POLICY_SEED_DERIVATION = (
    "sampling_seed_start + environment_seed * 10 + repeat_index"
)


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _assert_not_protected(path: Path, label: str) -> None:
    protected_subtrees = (
        PROJECT_ROOT / "data" / "generated",
        PROJECT_ROOT / "runs" / "ppo_v6_popart_actor1e4_fresh_20260910",
        PROJECT_ROOT / "runs" / "ppo_v6_popart_actor1e4_lazy_stage1_20260911",
        PROJECT_ROOT / "runs" / "ppo_v6_popart_actor1e4_lazy_episode_pool_20260911",
    )
    if path == PROJECT_ROOT or any(
        path == protected or _path_is_within(path, protected)
        for protected in protected_subtrees
    ):
        raise ValueError(f"{label} points at a protected project path: {path}")


def _resolve_pool_dir(value: str | Path, *, allow_missing: bool) -> Path:
    path = _resolve_path(value)
    _assert_not_protected(path, "episode pool")
    if path.exists() and not path.is_dir():
        raise ValueError(f"episode pool is not a directory: {path}")
    if not path.exists():
        if not allow_missing:
            raise FileNotFoundError(f"episode pool directory does not exist: {path}")
        path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_output_dir(value: str | Path, pool_dir: Path) -> Path:
    path = _resolve_path(value)
    _assert_not_protected(path, "evaluation output directory")
    if path == pool_dir or _path_is_within(path, pool_dir):
        raise ValueError(
            "evaluation output directory must be separate from the episode pool"
        )
    if _path_is_within(pool_dir, path):
        raise ValueError(
            "episode pool directory must be separate from the evaluation output"
        )
    return path


def _ensure_realization(
    *,
    directory: Path,
    seed: int,
    config: dict[str, Any],
    config_path: Path,
    generate_missing: bool,
    sumo_home: str | None,
    device: str,
) -> str:
    if directory.exists():
        if not directory.is_dir():
            raise RuntimeError(
                "development realization path is not a directory; refusing to "
                f"overwrite it: {directory}"
            )
        try:
            _validate_complete_realization(directory, config=config)
        except Exception as exc:
            raise RuntimeError(
                "development realization exists but is incomplete or invalid; "
                "refusing to regenerate or overwrite it: "
                f"{directory}"
            ) from exc
        return "LOAD"

    if not generate_missing:
        raise FileNotFoundError(
            "development realization is missing; use "
            "--generate-missing-episodes: "
            f"{directory}"
        )

    directory.mkdir(parents=True, exist_ok=False)
    try:
        realization_generator.run_episode(
            config_path,
            int(seed),
            directory,
            sumo_home,
            device,
        )
        _validate_complete_realization(directory, config=config)
    except Exception as exc:
        raise RuntimeError(
            "development realization generation failed; the partial directory "
            "was left in place and will not be overwritten: "
            f"{directory}"
        ) from exc
    return "GENERATE"


def _metric_record(
    seed: int,
    realization_dir: Path,
    evaluation_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    action: str,
    *,
    repeat_index: int | None,
    policy_sampling_seed: int | None,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "environment_seed": int(seed),
        "repeat_index": None if repeat_index is None else int(repeat_index),
        "policy_sampling_seed": (
            None if policy_sampling_seed is None else int(policy_sampling_seed)
        ),
        "policy_sampling_seed_derivation": (
            None if policy_sampling_seed is None else POLICY_SEED_DERIVATION
        ),
        "policy_action_mode": str(manifest["policy_action_mode"]),
        "episode_return": float(summary["total_return"]),
        "mean_reward": float(summary["mean_slot_reward"]),
        "q_total": float(summary["total_q"]),
        "cache_cost": float(summary["total_cache_cost"]),
        "movement_cost": float(summary["total_movement_cost"]),
        "cache_actual_change_rate": float(summary["cache_change_rate"]),
        "uav_actual_movement_rate": float(summary["uav_movement_rate"]),
        "cache_trigger_request_rate": float(summary["cache_trigger_request_rate"]),
        "uav_trigger_request_rate": float(summary["uav_trigger_request_rate"]),
        "cache_selected_operation_counts": dict(
            summary.get(
                "cache_selected_operation_counts",
                {"NO_CHANGE": 0, "ADD": 0, "DELETE": 0, "SWAP": 0},
            )
        ),
        "uav_selected_operation_counts": dict(
            summary.get("uav_selected_operation_counts", {"STAY": 0, "MOVE": 0})
        ),
        "realization_action": action,
        "realization_directory": str(realization_dir),
        "evaluation_directory": str(evaluation_dir),
        "deterministic": bool(manifest["deterministic"]),
        "policy_parameters_updated": bool(manifest["policy_parameters_updated"]),
        "popart_statistics_updated": bool(manifest["popart_statistics_updated"]),
        "optimizer_steps": int(manifest["optimizer_steps"]),
    }


def _sample_mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("evaluation aggregation requires finite non-empty values")
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def _build_evaluation_summary(
    records: list[dict[str, Any]],
    *,
    config_path: Path,
    checkpoint_path: Path,
    policy_action_mode: str,
    sampling_repeats: int,
    sampling_seed_start: int,
) -> dict[str, Any]:
    if policy_action_mode == "stochastic":
        if sampling_repeats <= 0:
            raise ValueError("stochastic sampling_repeats must be positive")
        expected_rollouts = DEVELOPMENT_COUNT * sampling_repeats
        if len(records) != expected_rollouts:
            raise ValueError(
                f"stochastic development evaluation requires {expected_rollouts} "
                f"records, got {len(records)}"
            )
        by_environment: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            seed = int(record["environment_seed"])
            by_environment.setdefault(seed, []).append(record)
        expected_seeds = list(
            range(DEVELOPMENT_SEED_START, DEVELOPMENT_SEED_END + 1)
        )
        if sorted(by_environment) != expected_seeds:
            raise ValueError("stochastic aggregation requires all 20 development seeds")
        environment_records: list[dict[str, Any]] = []
        within_return_variances: list[float] = []
        for seed in expected_seeds:
            environment_rows = sorted(
                by_environment[seed],
                key=lambda record: int(record["repeat_index"]),
            )
            if [int(row["repeat_index"]) for row in environment_rows] != list(
                range(sampling_repeats)
            ):
                raise ValueError(
                    f"stochastic repeats for environment {seed} are incomplete"
                )
            expected_sampling_seeds = [
                sampling_seed_start + seed * 10 + repeat_index
                for repeat_index in range(sampling_repeats)
            ]
            actual_sampling_seeds = [
                int(row["policy_sampling_seed"]) for row in environment_rows
            ]
            if actual_sampling_seeds != expected_sampling_seeds:
                raise ValueError(
                    f"stochastic sampling seed pairing is invalid for environment {seed}"
                )
            within_returns = np.asarray(
                [row["episode_return"] for row in environment_rows],
                dtype=np.float64,
            )
            within_return_variances.append(
                float(np.var(within_returns, ddof=1))
                if sampling_repeats > 1
                else 0.0
            )
            environment_records.append(
                {
                    "environment_seed": seed,
                    "episode_return": float(np.mean(within_returns)),
                    "q_total": float(
                        np.mean([row["q_total"] for row in environment_rows])
                    ),
                    "cache_cost": float(
                        np.mean([row["cache_cost"] for row in environment_rows])
                    ),
                    "movement_cost": float(
                        np.mean([row["movement_cost"] for row in environment_rows])
                    ),
                    "cache_actual_change_rate": float(
                        np.mean(
                            [
                                row["cache_actual_change_rate"]
                                for row in environment_rows
                            ]
                        )
                    ),
                    "uav_actual_movement_rate": float(
                        np.mean(
                            [
                                row["uav_actual_movement_rate"]
                                for row in environment_rows
                            ]
                        )
                    ),
                    "cache_trigger_request_rate": float(
                        np.mean(
                            [
                                row["cache_trigger_request_rate"]
                                for row in environment_rows
                            ]
                        )
                    ),
                    "uav_trigger_request_rate": float(
                        np.mean(
                            [
                                row["uav_trigger_request_rate"]
                                for row in environment_rows
                            ]
                        )
                    ),
                }
            )
        within_environment_policy_variance_mean = float(
            np.mean(within_return_variances)
        )
        aggregation_records = environment_records
    else:
        if len(records) != DEVELOPMENT_COUNT:
            raise ValueError(
                f"development evaluation requires {DEVELOPMENT_COUNT} records, "
                f"got {len(records)}"
            )
        aggregation_records = records
        within_environment_policy_variance_mean = None

    returns = np.asarray(
        [record["episode_return"] for record in aggregation_records], dtype=np.float64
    )
    return_std = float(np.std(returns, ddof=1))
    standard_error = return_std / math.sqrt(float(len(returns)))
    margin = STUDENT_T_CRITICAL_95_DF19 * standard_error
    return {
        "evaluation_role": DEVELOPMENT_EVALUATION_ROLE,
        "development_seed_start": DEVELOPMENT_SEED_START,
        "development_seed_end": DEVELOPMENT_SEED_END,
        "development_count": DEVELOPMENT_COUNT,
        "count": len(aggregation_records),
        "rollout_count": len(records),
        "environment_count": DEVELOPMENT_COUNT,
        "policy_action_mode": policy_action_mode,
        "sampling_repeats": int(sampling_repeats),
        "sampling_seed_start": (
            int(sampling_seed_start) if policy_action_mode == "stochastic" else None
        ),
        "sampling_seed_derivation": (
            POLICY_SEED_DERIVATION if policy_action_mode == "stochastic" else None
        ),
        "aggregation_level": (
            "20 environment means after per-environment policy-sample means"
            if policy_action_mode == "stochastic"
            else "20 independent environment realizations"
        ),
        "config_path": str(config_path),
        "config_sha256": evaluator._sha256(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": evaluator._sha256(checkpoint_path),
        "confidence_interval_method": (
            "Student-t 95% confidence interval for the mean; "
            "df=19; t_critical=2.093024054408263"
        ),
        "standard_deviation_method": "sample standard deviation (ddof=1)",
        "episode_return": {
            "mean": float(np.mean(returns)),
            "std": return_std,
            "median": float(np.median(returns)),
            "min": float(np.min(returns)),
            "max": float(np.max(returns)),
            "standard_error": float(standard_error),
            "confidence_interval_95": {
                "low": float(np.mean(returns) - margin),
                "high": float(np.mean(returns) + margin),
            },
        },
        "q_total": _sample_mean_std(
            [record["q_total"] for record in aggregation_records]
        ),
        "cache_cost": _sample_mean_std(
            [record["cache_cost"] for record in aggregation_records]
        ),
        "movement_cost": _sample_mean_std(
            [record["movement_cost"] for record in aggregation_records]
        ),
        "cache_actual_change_rate": _sample_mean_std(
            [record["cache_actual_change_rate"] for record in aggregation_records]
        ),
        "uav_actual_movement_rate": _sample_mean_std(
            [record["uav_actual_movement_rate"] for record in aggregation_records]
        ),
        "cache_trigger_request_rate": _sample_mean_std(
            [record["cache_trigger_request_rate"] for record in aggregation_records]
        ),
        "uav_trigger_request_rate": _sample_mean_std(
            [record["uav_trigger_request_rate"] for record in aggregation_records]
        ),
        "within_environment_policy_variance_mean": within_environment_policy_variance_mean,
        "between_environment_return_variance": (
            float(np.var(returns, ddof=1)) if len(returns) > 1 else 0.0
        ),
        "environment_return_means": [
            {
                "environment_seed": int(record["environment_seed"]),
                "mean_return": float(record["episode_return"]),
            }
            for record in aggregation_records
        ],
    }


def evaluate_episode_pool(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    episode_pool: str | Path,
    start_seed: int = DEVELOPMENT_SEED_START,
    count: int = DEVELOPMENT_COUNT,
    output_dir: str | Path,
    device: str = "cpu",
    generate_missing_episodes: bool = False,
    sumo_home: str | None = None,
    policy_action_mode: str = "deterministic",
    sampling_repeats: int = 1,
    sampling_seed_start: int = 500000,
) -> dict[str, Any]:
    if start_seed != DEVELOPMENT_SEED_START or count != DEVELOPMENT_COUNT:
        raise ValueError(
            "development evaluation is pinned to seeds 30001..30020 "
            "(start_seed=30001, count=20)"
        )
    requested_policy_action_mode = str(policy_action_mode)
    policy_action_mode = evaluator.normalize_policy_action_mode(policy_action_mode)
    sampling_repeats = int(sampling_repeats)
    sampling_seed_start = int(sampling_seed_start)
    if policy_action_mode == "stochastic":
        if sampling_repeats <= 0:
            raise ValueError("stochastic sampling_repeats must be positive")
    elif sampling_repeats != 1:
        raise ValueError(
            "sampling_repeats is only configurable for stochastic evaluation"
        )
    resolved_config = _resolve_path(config_path)
    resolved_checkpoint = _resolve_path(checkpoint_path)
    if not resolved_config.is_file():
        raise FileNotFoundError(f"config is not readable: {resolved_config}")
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(
            f"P50 checkpoint is not readable: {resolved_checkpoint}"
        )
    config = realization_generator.load_and_validate_config(resolved_config)
    pool_dir = _resolve_pool_dir(
        episode_pool,
        allow_missing=generate_missing_episodes,
    )
    output_root = _resolve_output_dir(output_dir, pool_dir)
    evaluator.ensure_fresh_evaluation_dir(output_root)

    try:
        import torch

        checkpoint_payload = torch.load(
            resolved_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise RuntimeError("could not inspect the P50 checkpoint") from exc
    completed_epochs = int(
        checkpoint_payload.get(
            "completed_epochs",
            checkpoint_payload.get("completed_episodes", -1),
        )
    )
    if completed_epochs != P50_COMPLETED_EPOCHS:
        raise ValueError(
            "development checkpoint must be the frozen P50 checkpoint: "
            f"completed_epochs={completed_epochs}"
        )
    completed_training_episodes = int(
        checkpoint_payload.get("completed_training_episodes", completed_epochs)
    )
    next_pool_index = int(
        checkpoint_payload.get("next_pool_index", completed_training_episodes)
    )
    if (
        completed_training_episodes != P50_COMPLETED_EPOCHS
        or next_pool_index != P50_COMPLETED_EPOCHS
    ):
        raise ValueError(
            "development checkpoint pool cursor is not at P50: "
            f"completed_training_episodes={completed_training_episodes}, "
            f"next_pool_index={next_pool_index}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "evaluation_metrics.jsonl"
    records: list[dict[str, Any]] = []
    with metrics_path.open("w", encoding="utf-8", newline="\n") as stream:
        for offset in range(count):
            seed = int(start_seed + offset)
            realization_dir = pool_dir / f"episode_{seed}"
            action = _ensure_realization(
                directory=realization_dir,
                seed=seed,
                config=config,
                config_path=resolved_config,
                generate_missing=generate_missing_episodes,
                sumo_home=sumo_home,
                device=device,
            )
            repeat_indices = (
                range(sampling_repeats)
                if policy_action_mode == "stochastic"
                else (None,)
            )
            for repeat_index in repeat_indices:
                policy_sampling_seed = (
                    None
                    if repeat_index is None
                    else sampling_seed_start + seed * 10 + int(repeat_index)
                )
                evaluation_dir = output_root / f"episode_{seed}"
                if repeat_index is not None:
                    evaluation_dir = evaluation_dir / f"repeat_{int(repeat_index)}"
                manifest, summary = evaluator.evaluate_full_episode(
                    checkpoint_path=resolved_checkpoint,
                    run_dir=evaluation_dir,
                    config_path=resolved_config,
                    device=device,
                    realization_dir=realization_dir,
                    policy_action_mode=policy_action_mode,
                    policy_sampling_seed=policy_sampling_seed,
                )
                manifest.update(
                    {
                        "evaluation_role": DEVELOPMENT_EVALUATION_ROLE,
                        "development_seed_start": DEVELOPMENT_SEED_START,
                        "development_seed_end": DEVELOPMENT_SEED_END,
                        "development_count": DEVELOPMENT_COUNT,
                        "realization_seed": seed,
                        "environment_seed": seed,
                        "repeat_index": (
                            None
                            if repeat_index is None
                            else int(repeat_index)
                        ),
                        "sampling_repeats": int(sampling_repeats),
                        "sampling_seed_start": (
                            int(sampling_seed_start)
                            if policy_action_mode == "stochastic"
                            else None
                        ),
                        "policy_sampling_seed_derivation": (
                            POLICY_SEED_DERIVATION
                            if policy_action_mode == "stochastic"
                            else None
                        ),
                        "policy_action_mode_requested": requested_policy_action_mode,
                    }
                )
                evaluator._write_json(
                    evaluation_dir / "full_episode_manifest.json", manifest
                )
                record = _metric_record(
                    seed,
                    realization_dir,
                    evaluation_dir,
                    manifest,
                    summary,
                    action,
                    repeat_index=repeat_index,
                    policy_sampling_seed=policy_sampling_seed,
                )
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                stream.write("\n")

    evaluation_summary = _build_evaluation_summary(
        records,
        config_path=resolved_config,
        checkpoint_path=resolved_checkpoint,
        policy_action_mode=policy_action_mode,
        sampling_repeats=sampling_repeats,
        sampling_seed_start=sampling_seed_start,
    )
    evaluation_summary["policy_action_mode_requested"] = requested_policy_action_mode
    evaluator._write_json(output_root / "evaluation_summary.json", evaluation_summary)
    return evaluation_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen P50 PPO V6 over development realizations "
            "30001..30020."
        )
    )
    parser.add_argument(
        "--config",
        default="system_config_ppo_v6_full_episode_randomized.toml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode-pool", required=True)
    parser.add_argument("--start-seed", type=int, default=DEVELOPMENT_SEED_START)
    parser.add_argument("--count", type=int, default=DEVELOPMENT_COUNT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sumo-home", default=None)
    parser.add_argument("--generate-missing-episodes", action="store_true")
    parser.add_argument(
        "--policy-action-mode",
        choices=("deterministic", *evaluator.POLICY_ACTION_MODES),
        default="deterministic",
        help=(
            "evaluation decoder; deterministic is the legacy alias for "
            "candidate_deterministic"
        ),
    )
    parser.add_argument(
        "--sampling-repeats",
        type=int,
        default=1,
        help="policy samples per environment for stochastic evaluation",
    )
    parser.add_argument(
        "--sampling-seed-start",
        type=int,
        default=500000,
        help=(
            "base for stochastic seeds: start + environment_seed * 10 + "
            "repeat_index"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evaluate_episode_pool(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            episode_pool=args.episode_pool,
            start_seed=args.start_seed,
            count=args.count,
            output_dir=args.output_dir,
            device=args.device,
            generate_missing_episodes=bool(args.generate_missing_episodes),
            sumo_home=args.sumo_home,
            policy_action_mode=args.policy_action_mode,
            sampling_repeats=args.sampling_repeats,
            sampling_seed_start=args.sampling_seed_start,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
