"""Read-only registry and entrypoint for the formal experiment set.

This command validates and displays method metadata.  It intentionally does
not invoke training, episode generation, or policy evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "experiment_registry.toml"
METHOD_IDS = (
    "proposed_joint",
    "ablation_cache_only",
    "ablation_joint_no_prediction",
    "baseline_popularity_greedy",
    "baseline_true_static",
)
LEARNING_METHOD_IDS = METHOD_IDS[:3]
REQUIRED_PRIMARY_METRICS = (
    "total_return",
    "total_q",
    "total_cache_cost",
    "total_movement_cost",
    "cache_change_rate",
    "uav_movement_rate",
    "total_tasks",
    "mean_task_delay_s",
    "mean_vehicle_energy_j",
)
EXECUTION_ROLES = ("local", "sv", "rsu", "uav", "cloud")


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    """Load and validate the registry without changing repository state."""

    registry_path = _resolve_path(path, PROJECT_ROOT)
    with registry_path.open("rb") as stream:
        registry = tomllib.load(stream)
    if not isinstance(registry, dict):
        raise ValueError("experiment registry root must be a TOML table")
    errors = validate_registry(registry, PROJECT_ROOT)
    if errors:
        raise ValueError("invalid experiment registry: " + "; ".join(errors))
    return registry


def _require_table(value: Any, name: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{name} must be a table")
        return {}
    return value


def _check_exact(value: Any, expected: Any, name: str, errors: list[str]) -> None:
    if value != expected:
        errors.append(f"{name}={value!r}, expected {expected!r}")


def validate_registry(
    registry: dict[str, Any],
    project_root: str | Path = PROJECT_ROOT,
) -> list[str]:
    """Return validation errors for stable IDs, protocol, metrics, and paths."""

    root = Path(project_root).resolve()
    errors: list[str] = []
    _check_exact(registry.get("registry_version"), 1, "registry_version", errors)
    _check_exact(registry.get("final_test_executed"), False, "final_test_executed", errors)

    training = _require_table(
        _require_table(registry.get("protocol"), "protocol", errors).get("training"),
        "protocol.training",
        errors,
    )
    evaluation = _require_table(
        _require_table(registry.get("protocol"), "protocol", errors).get("evaluation"),
        "protocol.evaluation",
        errors,
    )
    metrics = _require_table(
        _require_table(registry.get("protocol"), "protocol", errors).get("metrics"),
        "protocol.metrics",
        errors,
    )
    _check_exact(training.get("epochs"), 222, "protocol.training.epochs", errors)
    _check_exact(training.get("seed_start"), 20001, "protocol.training.seed_start", errors)
    _check_exact(training.get("seed_end"), 20222, "protocol.training.seed_end", errors)
    _check_exact(
        tuple(training.get("learning_method_ids", ())),
        LEARNING_METHOD_IDS,
        "protocol.training.learning_method_ids",
        errors,
    )
    _check_exact(
        evaluation.get("development_seed_start"),
        30001,
        "protocol.evaluation.development_seed_start",
        errors,
    )
    _check_exact(
        evaluation.get("development_seed_end"),
        30020,
        "protocol.evaluation.development_seed_end",
        errors,
    )
    _check_exact(
        evaluation.get("final_test_seed_start"),
        40001,
        "protocol.evaluation.final_test_seed_start",
        errors,
    )
    _check_exact(
        evaluation.get("final_test_seed_end"),
        40020,
        "protocol.evaluation.final_test_seed_end",
        errors,
    )
    _check_exact(
        evaluation.get("statistical_unit"),
        "environment_realization",
        "protocol.evaluation.statistical_unit",
        errors,
    )
    _check_exact(
        evaluation.get("learned_policy_repeats_per_environment"),
        5,
        "protocol.evaluation.learned_policy_repeats_per_environment",
        errors,
    )
    _check_exact(
        evaluation.get("nonlearning_baseline_repeats_per_environment"),
        1,
        "protocol.evaluation.nonlearning_baseline_repeats_per_environment",
        errors,
    )
    _check_exact(
        metrics.get("primary"),
        list(REQUIRED_PRIMARY_METRICS),
        "protocol.metrics.primary",
        errors,
    )
    _check_exact(
        metrics.get("execution_counts"),
        [f"{role}_count" for role in EXECUTION_ROLES],
        "protocol.metrics.execution_counts",
        errors,
    )
    _check_exact(
        metrics.get("execution_ratios"),
        [f"{role}_ratio" for role in EXECUTION_ROLES],
        "protocol.metrics.execution_ratios",
        errors,
    )

    methods = _require_table(registry.get("methods"), "methods", errors)
    _check_exact(
        tuple(methods.keys()),
        METHOD_IDS,
        "methods order/IDs",
        errors,
    )
    expected_flags = {
        "proposed_joint": (True, True, True, True),
        "ablation_cache_only": (True, True, False, True),
        "ablation_joint_no_prediction": (False, True, True, True),
        "baseline_popularity_greedy": (True, True, False, False),
        "baseline_true_static": (False, False, False, False),
    }
    for method_id in METHOD_IDS:
        method = _require_table(methods.get(method_id), f"methods.{method_id}", errors)
        for key in (
            "category",
            "status",
            "prediction",
            "adaptive_cache",
            "adaptive_uav",
            "learned_policy",
            "policy_mode",
            "stochastic_repeats",
            "config",
            "checkpoint",
            "training_entrypoint",
            "evaluation_entrypoint",
        ):
            if key not in method:
                errors.append(f"methods.{method_id}.{key} is missing")
        expected = expected_flags[method_id]
        actual = tuple(
            bool(method.get(key))
            for key in ("prediction", "adaptive_cache", "adaptive_uav", "learned_policy")
        )
        _check_exact(actual, expected, f"methods.{method_id}.feature_flags", errors)
        expected_category = "proposed" if method_id.startswith("proposed_") else (
            "ablation" if method_id.startswith("ablation_") else "baseline"
        )
        _check_exact(method.get("category"), expected_category, f"methods.{method_id}.category", errors)
        expected_mode = "stochastic" if method.get("learned_policy") else "deterministic"
        _check_exact(method.get("policy_mode"), expected_mode, f"methods.{method_id}.policy_mode", errors)
        expected_repeats = 5 if method.get("learned_policy") else 1
        _check_exact(
            method.get("stochastic_repeats"),
            expected_repeats,
            f"methods.{method_id}.stochastic_repeats",
            errors,
        )
        for path_key in ("config", "checkpoint", "evaluation_entrypoint"):
            value = str(method.get(path_key, ""))
            if value:
                resolved = _resolve_path(value, root)
                if not resolved.is_file():
                    errors.append(f"methods.{method_id}.{path_key} is missing: {value}")
        if method.get("learned_policy") and not str(method.get("training_entrypoint", "")):
            errors.append(f"methods.{method_id}.training_entrypoint is empty for learned policy")
        if not method.get("learned_policy") and str(method.get("training_entrypoint", "")):
            errors.append(f"methods.{method_id}.training_entrypoint must be empty for baseline")

    return errors


def _print_method_list(registry: dict[str, Any]) -> None:
    methods = registry["methods"]
    for method_id in METHOD_IDS:
        method = methods[method_id]
        print(f"{method_id}\t{method['category']}\t{method['status']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and display the read-only formal experiment registry."
    )
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY),
        help="registry TOML path",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--validate", action="store_true", help="validate registry and paths")
    actions.add_argument("--list", action="store_true", help="list stable method IDs")
    actions.add_argument("--show", metavar="METHOD_ID", help="show one method definition as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = load_registry(args.registry)
    except Exception as exc:
        print(f"REGISTRY_VALID = NO\nERROR = {exc}", file=sys.stderr)
        return 1
    if args.validate:
        print("REGISTRY_VALID = YES")
        return 0
    if args.show:
        if args.show not in METHOD_IDS:
            print(f"unknown method ID: {args.show}", file=sys.stderr)
            return 2
        print(json.dumps(registry["methods"][args.show], ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    _print_method_list(registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
