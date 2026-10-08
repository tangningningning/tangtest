"""在 episode pool 上评估冻结的双时间尺度 PPO 策略。"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - 仅静态导入环境
    torch = None  # type: ignore[assignment]

from common import project_config
from scenario.realization import REALIZATION_ARTIFACT_FILENAMES
from .interface import (
    PROPOSED_CONTROL_MODE,
    PROPOSED_POLICY_SCHEMA,
    _assert_realization_context_compatible,
    aligned_train_steps,
    load_environment_context,
)
from .reporting import resolve_checkpoint_reference
from .rollout import collect_dual_timescale_episode
from .policy import DualTimescaleActorCritic
from .trainer import ALGORITHM_ID, CHECKPOINT_METHOD, CHECKPOINT_VERSION, PPOTrainer


PROPOSED_CONFIG_PATH = "configs/proposed/system_config.toml"
DEFAULT_ENVIRONMENT_COUNT = 20
DEFAULT_POLICY_REPEATS = 1
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
    if int(payload.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError("checkpoint format is not supported by PROPOSED")
    expected_architecture = DualTimescaleActorCritic.architecture_version
    expected_schema = DualTimescaleActorCritic.architecture_id
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
    if contract.get("algorithm_id") != ALGORITHM_ID:
        raise ValueError("checkpoint config_contract algorithm_id mismatch")
    if contract.get("policy_schema") != expected_schema:
        raise ValueError("checkpoint config_contract policy_schema mismatch")
    if payload.get("unified_cache_action") is not True:
        raise ValueError("checkpoint does not use the unified Cache candidate action")
    if payload.get("cardinality_corrected_cache_action") is not True:
        raise ValueError("checkpoint does not use Cache cardinality correction")
    if (
        contract.get("architecture_version") != expected_architecture
        or contract.get("unified_cache_action") is not True
        or contract.get("cardinality_corrected_cache_action") is not True
        or contract.get("macro_cache_feature_dim") != 200
    ):
        raise ValueError("checkpoint PROPOSED configuration contract mismatch")
    expected_rates = {
        "macro_actor": 5.0e-4,
        "micro_actor": 3.0e-4,
        "macro_critic": 1.0e-3,
        "micro_critic": 3.0e-3,
        "long_value_critic": 1.0e-3,
    }
    contract_rates = contract.get("learning_rates")
    payload_rates = payload.get("learning_rates")
    if (
        payload.get("macro_actor_learning_rate_schedule") != "constant"
        or payload.get("optimizer_learning_rate_schedule") != "constant"
        or contract.get("macro_actor_learning_rate_schedule") != "constant"
        or contract.get("optimizer_learning_rate_schedule") != "constant"
        or not isinstance(contract_rates, dict)
        or not isinstance(payload_rates, dict)
        or any(contract_rates.get(key) != value for key, value in expected_rates.items())
        or any(
            payload_rates.get(key) != value
            for key, value in {
                "macro_actor": 5.0e-4,
                "micro_actor": 3.0e-4,
                "macro_critic": 1.0e-3,
                "micro_critic": 3.0e-3,
                "long_value": 1.0e-3,
            }.items()
        )
        or contract.get("macro_actor_learning_rate_initial") != 5.0e-4
        or contract.get("macro_actor_learning_rate_final") != 5.0e-4
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
    task_count = int(sum(item.task_count for item in micro))
    total_delay = float(sum(item.total_task_delay_s for item in micro))
    total_energy = float(sum(item.total_vehicle_energy_j for item in micro))
    record = {
        "episode_seed": int(episode_seed),
        "repeat_index": int(repeat_index),
        "total_return": float(sum(item.reward for item in micro)),
        "total_q": float(sum(item.actual_task_utility for item in micro)),
        "total_cache_cost": float(sum(item.cache_cost for item in micro)),
        "total_movement_cost": float(sum(item.movement_cost for item in micro)),
        "cache_change_rate": float(
            sum(item.actual_cache_update_count for item in micro) / slots
        ),
        "uav_movement_rate": float(
            sum(item.uav_movement_count for item in micro) / (2.0 * slots)
        ),
        "uav_stay_rate": float(
            1.0 - sum(item.uav_movement_count for item in micro) / (2.0 * slots)
        ),
        "uav_move_rate": float(
            sum(item.uav_movement_count for item in micro) / (2.0 * slots)
        ),
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
        "uav1_stay_rate": float(collection.summary["uav1_stay_rate"]),
        "uav0_move_rate": float(collection.summary["uav0_move_rate"]),
        "uav1_move_rate": float(collection.summary["uav1_move_rate"]),
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
        record[metric_name] = float(collection.summary[metric_name])
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
) -> dict[str, Any]:
    _require_torch()
    if int(episodes) <= 0 or int(repeats) <= 0:
        raise ValueError("episodes 和 repeats 必须为正整数。")
    if policy_action_mode not in {"deterministic", "stochastic"}:
        raise ValueError("policy_action_mode must be deterministic or stochastic")
    config_file = Path(config_path).expanduser().resolve(strict=True)
    checkpoint = resolve_checkpoint_reference(Path(checkpoint_path)).resolve(strict=True)
    checkpoint_metadata = _load_checkpoint_metadata(checkpoint)
    reference_context = load_environment_context(str(config_file))
    if checkpoint_metadata["policy_schema"] != reference_context.policy_schema:
        raise ValueError("checkpoint policy_schema does not match the selected formal config")
    project_root = reference_context.root
    pool_path = _resolve_path(project_root, episode_pool)
    output_path = _resolve_path(project_root, run_dir)
    _ensure_fresh_output(output_path)
    episode_paths = _episode_directories(pool_path, int(seed_start), int(episodes))
    raw_config_sha256 = project_config.raw_config_sha256(config_file)
    semantic_sha256 = project_config.semantic_config_hash(reference_context.cfg)
    summaries: list[dict[str, Any]] = []
    environment_means: list[dict[str, Any]] = []
    trainer: PPOTrainer | None = None
    parameter_snapshot: list[Any] | None = None
    optimizer_step_snapshot: tuple[int, int] | None = None

    for environment_index, realization_path in enumerate(episode_paths):
        context = load_environment_context(str(config_file), realization_dir=realization_path)
        _assert_realization_context_compatible(reference_context, context)
        if trainer is None:
            trainer = PPOTrainer(
                context,
                device=device,
                unified_cache_action=True,
                cardinality_corrected_cache_action=True,
            )
            trainer.load_checkpoint(
                checkpoint,
                config_raw_sha256=raw_config_sha256,
                config_semantic_sha256=semantic_sha256,
                episode_pool=None,
            )
            parameter_snapshot = [
                parameter.detach().clone() for parameter in trainer.policy.parameters()
            ]
            optimizer_step_snapshot = (
                int(trainer.training_counters["micro_optimizer_steps"]),
                int(trainer.training_counters["macro_optimizer_steps"]),
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
                    aligned_train_steps(context),
                    deterministic=policy_action_mode == "deterministic",
                )
            summary = _episode_summary(collection, episode_seed, repeat_index)
            summary["policy_sampling_seed"] = action_seed
            summary["policy_action_mode"] = policy_action_mode
            repeat_records.append(summary)
            summaries.append(summary)
        destination = output_path / f"episode_{episode_seed}"
        destination.mkdir(parents=True, exist_ok=False)
        with (destination / "proposed_repeat_summaries.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(repeat_records, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        environment_means.append(_mean_environment_summary(episode_seed, repeat_records))

    assert trainer is not None and parameter_snapshot is not None
    if any(
        not torch.equal(before, after.detach())
        for before, after in zip(parameter_snapshot, trainer.policy.parameters(), strict=True)
    ):
        raise RuntimeError("evaluation modified policy parameters")
    if optimizer_step_snapshot != (
        int(trainer.training_counters["micro_optimizer_steps"]),
        int(trainer.training_counters["macro_optimizer_steps"]),
    ):
        raise RuntimeError("evaluation performed a PPO optimizer step")

    available = tuple(
        name for name in EPISODE_METRIC_NAMES
        if all(item.get(name) is not None for item in environment_means)
    )
    aggregate = {
        "experiment_type": "macro_cache_smdp_micro_uav_episode_pool_evaluation",
        "config_path": str(config_file),
        "config_sha256": raw_config_sha256,
        "config_semantic_sha256": semantic_sha256,
        "checkpoint_path": str(checkpoint),
        "checkpoint_version": int(checkpoint_metadata["checkpoint_version"]),
        "checkpoint_policy_schema": checkpoint_metadata["policy_schema"],
        "policy_schema": PROPOSED_POLICY_SCHEMA,
        "configuration_control_mode": PROPOSED_CONTROL_MODE,
        "episode_pool": str(pool_path),
        "episode_count": len(environment_means),
        "seed_start": int(seed_start),
        "episodes": int(episodes),
        "repeats": int(repeats),
        "policy_action_mode": policy_action_mode,
        "policy_sampling_seed": policy_sampling_seed,
        "statistical_unit": "environment_realization",
        "repeat_aggregation": "within_environment_mean_then_across_environment_statistics",
        "aggregate_statistics": _metric_statistics(environment_means, available),
        "environment_means": environment_means,
        "repeat_summaries": summaries,
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
    parser = argparse.ArgumentParser(description="评估冻结的双时间尺度 Cache/UAV PPO 策略。")
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
