"""Evaluate Prediction-Popularity Greedy Cache on an existing episode pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ppo_simple.evaluation_metrics import (
    episode_metric_statistics,
    execution_ratio_statistics,
)
from ppo_simple.popularity_greedy_evaluator import evaluate_popularity_greedy_episode


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def evaluate_pool(
    run_dir: str | Path,
    config_path: str | Path,
    episode_pool: str | Path,
    seed_start: int,
    episodes: int,
) -> dict[str, Any]:
    if int(episodes) <= 0:
        raise ValueError("episodes must be positive")
    project_root = PROJECT_ROOT
    pool_path = _resolve(project_root, episode_pool)
    output_path = _resolve(project_root, run_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            "fresh popularity-greedy evaluation pool output already contains artifacts; "
            "use a new --run-dir"
        )

    summaries: list[dict[str, Any]] = []
    for index in range(int(episodes)):
        episode_seed = int(seed_start) + index
        realization = pool_path / f"episode_{episode_seed}"
        if not realization.is_dir():
            raise FileNotFoundError(
                f"popularity-greedy evaluation pool realization is missing: {realization}"
            )
        _, summary = evaluate_popularity_greedy_episode(
            run_dir=output_path / f"episode_{episode_seed}",
            config_path=config_path,
            realization_dir=realization,
            episode_seed=episode_seed,
        )
        if int(summary["episode_seed"]) != episode_seed:
            raise AssertionError(
                "popularity-greedy evaluator returned an unexpected episode_seed"
            )
        summaries.append(summary)

    experiment_seeds = {int(summary["experiment_seed"]) for summary in summaries}
    if len(experiment_seeds) != 1:
        raise AssertionError(
            "popularity-greedy evaluation pool contains inconsistent experiment_seed values"
        )
    metric_names = (
        "total_return",
        "total_q",
        "total_cache_cost",
        "cache_change_rate",
        "mean_task_delay_s",
        "mean_vehicle_energy_j",
        "mean_services_added_per_slot",
        "mean_services_removed_per_slot",
    )
    output_path.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "experiment_type": "prediction_popularity_greedy_episode_pool_evaluation",
        "episode_pool": str(pool_path),
        "experiment_seed": next(iter(experiment_seeds)),
        "episode_count": len(summaries),
        "seed_start": int(seed_start),
        "episodes": int(episodes),
        "aggregate_statistics": episode_metric_statistics(summaries, metric_names),
        "execution_ratio_statistics": execution_ratio_statistics(summaries),
        "summaries": summaries,
    }
    with (output_path / "popularity_greedy_episode_pool_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(aggregate, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the Prediction-Popularity Greedy Cache baseline"
    )
    parser.add_argument(
        "--config",
        default="system_config_ppo_v8_cache_operation_group_replica_aware.toml",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episode-pool", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate_pool(
        run_dir=args.run_dir,
        config_path=args.config,
        episode_pool=args.episode_pool,
        seed_start=args.seed_start,
        episodes=args.episodes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
