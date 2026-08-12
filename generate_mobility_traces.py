"""Stage15A 离线 SUMO 移动真值准备入口。

该脚本只做三件事：读取配置、启动 SUMO 记录车辆状态、保存每个 seed 的
``seed_<seed>.npz``。不会生成请求，也不会执行缓存、UAV、Matching、KKT、PPO
或 reward 计算。
"""

from __future__ import annotations

import argparse
import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .data import load_config
    from .mobility import (
        TRACE_SCHEMA_VERSION,
        _TRACE_FLOAT_FIELDS,
        _TRACE_REQUIRED_FIELDS,
        _TRACE_STRING_FIELDS,
        generate_mobility_trace,
        trace_path,
        trace_root_from_config,
    )
except ImportError:  # pragma: no cover - direct script execution
    from data import load_config
    from mobility import (
        TRACE_SCHEMA_VERSION,
        _TRACE_FLOAT_FIELDS,
        _TRACE_REQUIRED_FIELDS,
        _TRACE_STRING_FIELDS,
        generate_mobility_trace,
        trace_path,
        trace_root_from_config,
    )


STAGE15_TRAIN_SEEDS = tuple(range(20260811, 20261067))
STAGE15_TRAIN_RESET_RESERVE_SEEDS = (STAGE15_TRAIN_SEEDS[-1] + 1,)
STAGE15_VALIDATION_SEEDS = tuple(range(20271811, 20271816))
STAGE15_TEST_SEEDS = tuple(range(20281811, 20281821))


def _stage15_trace_partitions(config: Mapping[str, Any]) -> dict[str, Any]:
    """根据单环境 SB3 rollout 边界计算正式 trace 与 terminal reserve。"""

    simulation = config.get("simulation", {})
    ppo = config.get("ppo", {})
    if not isinstance(simulation, Mapping) or not isinstance(ppo, Mapping):
        raise ValueError("Stage15 trace capacity requires simulation and ppo mappings")
    total_timesteps = int(ppo.get("total_timesteps", 0))
    n_steps = int(ppo.get("n_steps", 0))
    episode_horizon = int(simulation.get("slot_count", 0))
    n_envs = 1  # Stage15 当前 DummyVecEnv 只有一个并行环境。
    if total_timesteps <= 0 or n_steps <= 0 or episode_horizon <= 0:
        raise ValueError("Stage15 trace capacity parameters must be positive")

    rollout_count = int(math.ceil(total_timesteps / (n_steps * n_envs)))
    effective_env_steps = rollout_count * n_steps * n_envs
    if effective_env_steps % episode_horizon == 0:
        # 最后一步正好产生 terminal；VecEnv 随后还会自动 reset 一次，
        # 该 observation 不属于新的 completed training episode。
        formal_training_count = effective_env_steps // episode_horizon
        reserve_count = 1 if effective_env_steps > 0 else 0
    else:
        formal_training_count = int(math.ceil(effective_env_steps / episode_horizon))
        reserve_count = 0

    train_seed_base = int(ppo.get("train_seed_base", STAGE15_TRAIN_SEEDS[0]))
    formal_training_seeds = tuple(
        train_seed_base + index for index in range(formal_training_count)
    )
    training_reset_reserve_seeds = tuple(
        train_seed_base + formal_training_count + index
        for index in range(reserve_count)
    )
    validation_seeds = tuple(int(seed) for seed in ppo.get("validation_seeds", ()))
    test_seeds = tuple(int(seed) for seed in ppo.get("test_seeds", ()))
    return {
        "total_timesteps": total_timesteps,
        "n_steps": n_steps,
        "n_envs": n_envs,
        "episode_horizon": episode_horizon,
        "effective_env_steps": effective_env_steps,
        "formal_training_count": formal_training_count,
        "formal_training_seeds": formal_training_seeds,
        "training_reset_reserve_count": reserve_count,
        "training_reset_reserve_seeds": training_reset_reserve_seeds,
        "validation_seeds": validation_seeds,
        "test_seeds": test_seeds,
    }


def _stage15_seeds(config: Mapping[str, Any] | None = None) -> tuple[int, ...]:
    if config is not None:
        partitions = _stage15_trace_partitions(config)
        return (
            partitions["formal_training_seeds"]
            + partitions["training_reset_reserve_seeds"]
            + partitions["validation_seeds"]
            + partitions["test_seeds"]
        )
    return (
        STAGE15_TRAIN_SEEDS
        + STAGE15_TRAIN_RESET_RESERVE_SEEDS
        + STAGE15_VALIDATION_SEEDS
        + STAGE15_TEST_SEEDS
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate offline SUMO mobility traces; this command never trains PPO."
    )
    parser.add_argument(
        "--stage15",
        action="store_true",
        help="generate the complete Stage15 train/validation/test seed set",
    )
    parser.add_argument(
        "--seed",
        dest="seeds",
        action="append",
        type=int,
        help="generate one seed; repeat --seed for a small controlled batch",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="optional config.yaml path (default: project config.yaml)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate an existing trace instead of skipping it",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="compare SUMO and Trace Reference rollouts for seeds 20260811/20260812",
    )
    parser.add_argument(
        "--verify-stage15",
        action="store_true",
        help="lightweight verification of all formal Stage15 trace files",
    )
    return parser.parse_args(argv)


def _auxiliary_action(environment: Any) -> dict[str, np.ndarray]:
    return {
        "cache_triggers": np.ones(environment._cache.shape[0], dtype=float),
        "uav_triggers": np.ones(environment._uav_positions.shape[0], dtype=float),
    }


def _mobility_snapshot(environment: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for vehicle_id, state in sorted(environment._last_actual_states.items()):
        result[vehicle_id] = {
            "position_m": np.asarray(state["position_m"], dtype=float).copy(),
            "speed_mps": float(state.get("speed_mps", 0.0)),
            "edge_id": state.get("edge_id"),
            "lane_id": state.get("lane_id"),
            "lane_position_m": float(state.get("lane_position_m", math.nan)),
        }
    return result


def _prediction_snapshot(environment: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for vehicle_id, prediction in sorted(environment._road_predictions.items()):
        result[vehicle_id] = {
            "predicted_positions_m": np.asarray(
                prediction["predicted_positions_m"], dtype=float
            ).copy(),
            "predicted_edge_ids": tuple(prediction.get("predicted_edge_ids", ())),
            "future_edge_ids": tuple(prediction.get("future_edge_ids", ())),
            "lane_ids": tuple(prediction.get("lane_ids", ())),
            "future_slots": tuple(prediction.get("future_slots", ())),
            "next_edge_distribution": dict(
                prediction.get("next_edge_distribution", {})
            ),
        }
    return result


def _rollout_snapshot(environment: Any, info: Mapping[str, Any], reward: float) -> dict[str, Any]:
    slot_index = int(info["slot_index"])
    workload = environment._data["workload"]
    return {
        "mobility": _mobility_snapshot(environment),
        "roles": {
            "service_vehicle_ids": tuple(info["service_vehicle_ids"]),
            "task_vehicle_ids": tuple(info["task_vehicle_ids"]),
            "role_change_events": tuple(info["role_change_events"]),
        },
        "requests": {
            "task_arrivals": np.asarray(workload["task_arrivals"])[slot_index].copy(),
            "task_service_types": np.asarray(
                workload["task_service_types"]
            )[slot_index].copy(),
        },
        "predictions": {
            "road": _prediction_snapshot(environment),
            "predicted_positions": np.asarray(
                environment._predicted_positions, dtype=float
            ).copy(),
            "predicted_popularity": np.asarray(
                environment._predicted_popularity, dtype=float
            ).copy(),
            "node_demand": np.asarray(
                environment.last_node_prediction.get("predicted_demand", []),
                dtype=float,
            ).copy(),
            "node_popularity": np.asarray(
                environment.last_node_prediction.get("predicted_popularity", []),
                dtype=float,
            ).copy(),
        },
        "cache": np.asarray(environment._cache, dtype=float).copy(),
        "uav": np.asarray(environment._uav_positions, dtype=float).copy(),
        "matching_kkt": {
            "assignment": np.asarray(info["assignment"], dtype=float).copy(),
            "compute_allocations": np.asarray(
                info["compute_allocations"], dtype=float
            ).copy(),
            "bandwidth_allocations": np.asarray(
                info["bandwidth_allocations"], dtype=float
            ).copy(),
            "matching_migration_count": int(info["matching_migration_count"]),
            "matching_swap_count": int(info["matching_swap_count"]),
            "kkt_failure_count": int(info["kkt_failure_count"]),
        },
        "execution": {
            "execution_counts": dict(info["execution_counts"]),
            "execution_rates": dict(info["execution_rates"]),
            "execution_nodes": tuple(info["execution_nodes"]),
            "completed_tasks": int(info["completed_tasks"]),
            "failed_tasks": int(info["failed_tasks"]),
        },
        "reward": float(reward),
        "system_utility": dict(info["system_utility"]),
    }


def _values_equal(left: Any, right: Any, *, atol: float = 1.0e-8) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_values_equal(left[key], right[key], atol=atol) for key in left)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.asarray(left).shape == np.asarray(right).shape and bool(
                np.allclose(
                    np.asarray(left, dtype=float),
                    np.asarray(right, dtype=float),
                    rtol=0.0,
                    atol=atol,
                    equal_nan=True,
                )
            )
        except (TypeError, ValueError):
            return np.array_equal(np.asarray(left), np.asarray(right))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if not isinstance(left, (tuple, list)) or not isinstance(right, (tuple, list)):
            return False
        return len(left) == len(right) and all(
            _values_equal(item_left, item_right, atol=atol)
            for item_left, item_right in zip(left, right)
        )
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        try:
            return bool(np.isclose(float(left), float(right), rtol=0.0, atol=atol, equal_nan=True))
        except (TypeError, ValueError):
            return False
    return left == right


def _run_consistency_rollout(
    config: Mapping[str, Any],
    *,
    seed: int,
    source: str,
) -> tuple[list[dict[str, Any]], int]:
    try:
        from .environment import MECEnvironment
        from .mobility import sumo_start_count
    except ImportError:  # pragma: no cover - direct script execution
        from environment import MECEnvironment
        from mobility import sumo_start_count

    rollout_config = copy.deepcopy(dict(config))
    rollout_config["mobility"] = dict(rollout_config.get("mobility", {}))
    rollout_config["mobility"]["source"] = source
    before_starts = sumo_start_count()
    environment = MECEnvironment(rollout_config)
    records: list[dict[str, Any]] = []
    try:
        environment.reset(seed=int(seed))
        action = _auxiliary_action(environment)
        for _ in range(int(rollout_config["simulation"]["slot_count"])):
            _next_state, reward, done, info = environment.step(action)
            records.append(_rollout_snapshot(environment, info, reward))
            if done:
                break
    finally:
        environment.close()
    return records, sumo_start_count() - before_starts


def _validate_seed_consistency(config: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    sumo_records, sumo_starts = _run_consistency_rollout(
        config, seed=seed, source="sumo"
    )
    trace_records, trace_starts = _run_consistency_rollout(
        config, seed=seed, source="trace"
    )
    if len(sumo_records) != len(trace_records):
        raise RuntimeError(
            f"seed {seed}: SUMO/Trace slot count mismatch "
            f"{len(sumo_records)} != {len(trace_records)}"
        )
    category_names = (
        "active_ids_and_state_xy_speed_edge_lane",
        "roles",
        "requests",
        "predictions",
        "cache",
        "uav",
        "matching_kkt",
        "execution_distribution",
        "reward",
    )
    mismatch_counts = {name: 0 for name in category_names}
    for sumo_record, trace_record in zip(sumo_records, trace_records):
        checks = {
            category_names[0]: _values_equal(
                sumo_record["mobility"], trace_record["mobility"]
            ),
            "roles": _values_equal(sumo_record["roles"], trace_record["roles"]),
            "requests": _values_equal(
                sumo_record["requests"], trace_record["requests"]
            ),
            "predictions": _values_equal(
                sumo_record["predictions"], trace_record["predictions"]
            ),
            "cache": _values_equal(sumo_record["cache"], trace_record["cache"]),
            "uav": _values_equal(sumo_record["uav"], trace_record["uav"]),
            "matching_kkt": _values_equal(
                sumo_record["matching_kkt"], trace_record["matching_kkt"]
            ),
            "execution_distribution": _values_equal(
                sumo_record["execution"], trace_record["execution"]
            ),
            "reward": _values_equal(
                {"reward": sumo_record["reward"], "utility": sumo_record["system_utility"]},
                {"reward": trace_record["reward"], "utility": trace_record["system_utility"]},
            ),
        }
        for category, matched in checks.items():
            mismatch_counts[category] += int(not matched)
    if sumo_starts != 1:
        raise RuntimeError(f"seed {seed}: expected one SUMO start, observed {sumo_starts}")
    if trace_starts != 0:
        raise RuntimeError(f"seed {seed}: Trace replay started SUMO {trace_starts} times")
    if any(mismatch_counts.values()):
        raise RuntimeError(f"seed {seed}: consistency mismatches {mismatch_counts}")
    return {
        "seed": int(seed),
        "slots": len(sumo_records),
        "status": "PASS",
        "sumo_starts": int(sumo_starts),
        "trace_starts": int(trace_starts),
        "mismatches": mismatch_counts,
    }


def _validate_short_reference(config: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    try:
        from .train_ppo import _run_reference_short_replay
        from .mobility import sumo_start_count
    except ImportError:  # pragma: no cover - direct script execution
        from train_ppo import _run_reference_short_replay
        from mobility import sumo_start_count
    before = sumo_start_count()
    summary = _run_reference_short_replay(config, seed=seed)
    starts = sumo_start_count() - before
    expected = {
        "task_total": 106,
        "execution_counts": {
            "local": 0,
            "service_vehicle": 0,
            "rsu": 15,
            "uav": 7,
            "cloud": 84,
            "failed": 0,
        },
        "failed_tasks": 0,
        "future_request_leakage": 0,
        "future_trajectory_leakage": 0,
        "kkt_failure_count": 0,
        "nan_inf_count": 0,
    }
    if starts != 0 or any(summary[key] != value for key, value in expected.items() if key != "execution_counts"):
        raise RuntimeError(f"20-slot trace replay regression failed: {summary}")
    if summary["execution_counts"] != expected["execution_counts"]:
        raise RuntimeError(f"20-slot trace execution regression failed: {summary}")
    return {
        "status": "PASS",
        "seed": int(seed),
        "slots": int(summary["steps"]),
        "task_total": int(summary["task_total"]),
        "execution_counts": summary["execution_counts"],
        "failed_tasks": int(summary["failed_tasks"]),
        "sumo_starts": int(starts),
    }


def validate_small_consistency(config: Mapping[str, Any]) -> None:
    for seed in (20260811, 20260812):
        result = _validate_seed_consistency(config, seed=seed)
        print(f"CONSISTENCY {result}")
    regression = _validate_short_reference(config, seed=20260811)
    print(f"REFERENCE_20_SLOT {regression}")


def verify_stage15_traces(config: Mapping[str, Any]) -> dict[str, Any]:
    """Lightweight, file-only acceptance check for formal traces plus reserve."""

    partitions = _stage15_trace_partitions(config)
    formal_training_seeds = tuple(partitions["formal_training_seeds"])
    training_reset_reserve_seeds = tuple(
        partitions["training_reset_reserve_seeds"]
    )
    validation_seeds = tuple(partitions["validation_seeds"])
    test_seeds = tuple(partitions["test_seeds"])
    expected_seeds = _stage15_seeds(config)
    expected_names = {f"seed_{seed}.npz": seed for seed in expected_seeds}
    root = trace_root_from_config(config)
    present_names = {
        path.name
        for path in root.glob("*.npz")
        if path.is_file()
    }
    present_expected = sorted(set(expected_names).intersection(present_names))
    missing = sorted(set(expected_names).difference(present_names))
    expected_fields = set(_TRACE_REQUIRED_FIELDS) | {"seed"}
    schema_mismatch = 0
    corrupted = 0
    seed_mismatch = 0
    slot_mismatch = 0
    nan_inf = 0
    for name in present_expected:
        path = root / name
        expected_seed = int(expected_names[name])
        try:
            with np.load(path, allow_pickle=False) as archive:
                fields = set(archive.files)
                if fields != expected_fields:
                    schema_mismatch += 1
                schema_value = int(np.asarray(archive["schema_version"]).item())
                if schema_value != TRACE_SCHEMA_VERSION:
                    schema_mismatch += 1
                internal_seed = int(np.asarray(archive["seed"]).item())
                if internal_seed != expected_seed:
                    seed_mismatch += 1
                slot_count = int(np.asarray(archive["slot_count"]).item())
                slots = np.asarray(archive["slots"], dtype=np.int64)
                times = np.asarray(archive["times"], dtype=np.float64)
                vehicle_ids = np.asarray(archive["vehicle_ids"])
                active = np.asarray(archive["active"], dtype=bool)
                if (
                    slot_count != 100
                    or slots.shape != (100,)
                    or not np.array_equal(slots, np.arange(100, dtype=np.int64))
                    or times.shape != (100,)
                    or not np.all(np.isfinite(times))
                    or np.any(np.diff(times) <= 0.0)
                    or vehicle_ids.ndim != 1
                    or vehicle_ids.size == 0
                    or active.shape != (100, vehicle_ids.size)
                ):
                    slot_mismatch += 1
                expected_shape = (100, vehicle_ids.size)
                for field in (*_TRACE_FLOAT_FIELDS, *_TRACE_STRING_FIELDS, "lane_index"):
                    if np.asarray(archive[field]).shape != expected_shape:
                        slot_mismatch += 1
                for field in (*_TRACE_FLOAT_FIELDS, *_TRACE_STRING_FIELDS, "lane_index"):
                    initial_name = f"initial_{field}"
                    if np.asarray(archive[initial_name]).shape != (vehicle_ids.size,):
                        slot_mismatch += 1
                for field in _TRACE_FLOAT_FIELDS:
                    values = np.asarray(archive[field], dtype=np.float64)
                    if np.any(~np.isfinite(values[active])):
                        nan_inf += 1
                    initial_values = np.asarray(
                        archive[f"initial_{field}"], dtype=np.float64
                    )
                    initial_active = np.asarray(
                        archive["initial_active"], dtype=bool
                    )
                    if np.any(~np.isfinite(initial_values[initial_active])):
                        nan_inf += 1
                initial_time = float(np.asarray(archive["initial_time_s"]).item())
                if not math.isfinite(initial_time):
                    nan_inf += 1
        except Exception:
            corrupted += 1

    formal_training_count = len(formal_training_seeds)
    training_reset_reserve_count = len(training_reset_reserve_seeds)
    validation_count = len(validation_seeds)
    test_count = len(test_seeds)
    overlap = {
        "train_validation": len(set(formal_training_seeds).intersection(validation_seeds)),
        "train_test": len(set(formal_training_seeds).intersection(test_seeds)),
        "validation_test": len(set(validation_seeds).intersection(test_seeds)),
        "reserve_validation": len(
            set(training_reset_reserve_seeds).intersection(validation_seeds)
        ),
        "reserve_test": len(
            set(training_reset_reserve_seeds).intersection(test_seeds)
        ),
    }
    try:
        from .train_ppo import validate_stage15_configuration
    except ImportError:  # pragma: no cover - direct script execution
        from train_ppo import validate_stage15_configuration
    settings = validate_stage15_configuration(config)
    configured_formal_training_count = int(settings["episodes_started"])
    capacity_formula = (
        "floor(effective_env_steps / episode_horizon) + 1"
        if partitions["effective_env_steps"] % partitions["episode_horizon"] == 0
        else "ceil(effective_env_steps / episode_horizon)"
    )
    training_coverage = (
        formal_training_count == configured_formal_training_count == 256
        and partitions["effective_env_steps"] == 25600
        and training_reset_reserve_count == 1
    )
    result = {
        "expected": len(expected_seeds),
        "present": len(present_expected),
        "missing": len(missing),
        "missing_files": missing,
        "corrupted": corrupted,
        "schema_mismatch": schema_mismatch,
        "seed_mismatch": seed_mismatch,
        "slot_mismatch": slot_mismatch,
        "nan_inf": nan_inf,
        "formal_training_count": formal_training_count,
        "training_reset_reserve_count": training_reset_reserve_count,
        "training_reset_reserve_seeds": list(training_reset_reserve_seeds),
        "validation_count": validation_count,
        "test_count": test_count,
        "seed_overlap": overlap,
        "effective_env_steps": int(partitions["effective_env_steps"]),
        "episode_horizon": int(partitions["episode_horizon"]),
        "capacity_formula": capacity_formula,
        "required_training_trace_count": formal_training_count
        + training_reset_reserve_count,
        "training_traces_cover_25600_timesteps": training_coverage,
        "stage15_configuration": "PASS",
    }
    failures = (
        result["missing"],
        result["corrupted"],
        result["schema_mismatch"],
        result["seed_mismatch"],
        result["slot_mismatch"],
        result["nan_inf"],
        *overlap.values(),
        int(not training_coverage),
    )
    if any(failures):
        raise RuntimeError(f"Stage15 trace dataset verification failed: {result}")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.verify_stage15:
        if args.stage15 or args.seeds or args.validate:
            raise SystemExit("--verify-stage15 cannot be combined with generation or rollout validation")
        config = load_config() if args.config is None else load_config(args.config)
        result = verify_stage15_traces(config)
        print(f"STAGE15_TRACE_DATASET {result}")
        print("stage15 trace dataset verification: PASS; model_learn_calls=0")
        return
    if args.validate:
        if args.stage15 or args.seeds:
            raise SystemExit("--validate cannot be combined with --stage15 or --seed")
        config = load_config() if args.config is None else load_config(args.config)
        validate_small_consistency(config)
        print("small SUMO/Trace consistency validation: PASS; model_learn_calls=0")
        return
    if args.stage15 and args.seeds:
        raise SystemExit("--stage15 and --seed cannot be used together")
    if not args.stage15 and not args.seeds:
        raise SystemExit("provide --stage15 or at least one --seed")

    config = load_config() if args.config is None else load_config(args.config)
    seeds = _stage15_seeds(config) if args.stage15 else tuple(dict.fromkeys(args.seeds))
    generated = 0
    skipped = 0
    for seed in seeds:
        output = trace_path(config, seed=int(seed))
        if output.is_file() and not args.overwrite:
            skipped += 1
            print(f"SKIP seed={seed} path={output}")
            continue
        generated_path = generate_mobility_trace(
            config,
            seed=int(seed),
            output_path=output,
        )
        generated += 1
        print(f"OK seed={seed} path={generated_path}")
    print(
        f"mobility trace preparation complete: requested={len(seeds)}, "
        f"generated={generated}, skipped={skipped}, model_learn_calls=0"
    )


if __name__ == "__main__":
    main()
