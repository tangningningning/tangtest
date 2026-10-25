"""冻结策略在正式 Validation/Test 场景上的共同执行路径。"""

from __future__ import annotations

import hashlib
import contextlib
import copy
import io
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from common import project_config
from .agent import DoubleDQNAgent
from .environment_adapter import ExclusiveDQNEnvironment
from .trainer import _summarize_task_lifecycle


EVALUATION_SCHEMA_VERSION = "online_joint_dqn_frozen_evaluation_v1"
FORMAL_SPLIT_CONTRACTS = {
    "validation": {
        "relative_path": Path("environment_data/proposed_joint_validation"),
        "seed_start": 30001,
        "episode_count": 30,
    },
    "test": {
        "relative_path": Path("environment_data/proposed_joint_test"),
        "seed_start": 40001,
        "episode_count": 100,
    },
}
SUPPORTED_POLICIES = ("dqn_greedy", "keep_stay", "uniform_random_legal")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def policy_state_sha256(agent: DoubleDQNAgent) -> str:
    """哈希确定性贪心策略实际读取的在线网络权重。"""

    digest = hashlib.sha256()
    for module_name in ("online_network", "online_category_network"):
        module = getattr(agent, module_name)
        for name, tensor in sorted(module.state_dict().items()):
            value = tensor.detach().to(device="cpu").contiguous()
            digest.update(module_name.encode("utf-8"))
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(list(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_formal_evaluation_pool(
    project_root: str | Path,
    split: str,
) -> tuple[Path, dict[str, Any]]:
    """验证整个独立池及每个场景快照；任何损坏都会使评价整体失败。"""

    split = str(split).strip().lower()
    if split not in FORMAL_SPLIT_CONTRACTS:
        raise ValueError(f"Unsupported evaluation split: {split!r}")
    spec = FORMAL_SPLIT_CONTRACTS[split]
    root = Path(project_root).resolve()
    scripts = root / "scripts" / "training"
    import sys

    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import training_pool_contract as pool_contract

    config = project_config.parse_toml_file(
        root / pool_contract.CONFIG_RELATIVE
    )
    pool, info = pool_contract._validate_pool(
        root,
        spec["relative_path"],
        expected_split=split,
        expected_seed_start=int(spec["seed_start"]),
        expected_episode_count=int(spec["episode_count"]),
        runtime_config=config,
    )
    seeds = [int(value) for value in info["active_episode_seeds"]]
    expected_seeds = list(
        range(
            int(spec["seed_start"]),
            int(spec["seed_start"]) + int(spec["episode_count"]),
        )
    )
    if seeds != expected_seeds:
        raise ValueError(f"Formal {split} active seed order differs from its contract")
    invalid: list[dict[str, Any]] = []
    for seed in seeds:
        try:
            pool_contract._validate_episode_snapshot(pool, seed, info)
        except Exception as exc:
            invalid.append(
                {"episode_seed": seed, "error": f"{type(exc).__name__}: {exc}"}
            )
    if invalid:
        raise RuntimeError(
            f"Formal {split} pool contains invalid scenes; no partial evaluation is allowed: "
            + json.dumps(invalid[:10], ensure_ascii=False, sort_keys=True)
        )
    return pool, info


def evaluation_pool_identity(pool_info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split": str(pool_info["split"]),
        "manifest_sha256": str(pool_info["manifest_sha256"]),
        "data_split_manifest_sha256": str(pool_info["data_split_manifest_sha256"]),
        "dataset_identity_sha256": str(pool_info["dataset_identity_sha256"]),
        "service_catalog_sha256": str(pool_info["service_catalog_sha256"]),
        "active_episode_seeds": [
            int(value) for value in pool_info["active_episode_seeds"]
        ],
        "active_episode_scene_sha256": dict(
            sorted(pool_info["active_episode_scene_sha256"].items())
        ),
    }


def should_update_best(current_best: float | None, candidate: float) -> bool:
    """仅严格改善时更新 best；并列结果保留既有模型。"""

    value = float(candidate)
    if not math.isfinite(value):
        raise ValueError("Validation 主指标必须为有限数值")
    if current_best is None:
        return True
    previous = float(current_best)
    if not math.isfinite(previous):
        raise ValueError("已保存的 best Validation 指标必须为有限数值")
    return value > previous


def _seeded_policy_rng(split: str, episode_seed: int) -> np.random.Generator:
    material = f"online-joint-dqn-eval-v1:{split}:{int(episode_seed)}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(seed)


def _choose_candidate(
    policy: str,
    state: np.ndarray,
    candidates: Sequence[Any],
    *,
    agent: DoubleDQNAgent | None,
    rng: np.random.Generator,
) -> Any:
    if policy == "dqn_greedy":
        if agent is None:
            raise ValueError("dqn_greedy evaluation requires a frozen agent")
        return agent.select_action(
            state, candidates, epsilon=0.0, explore=False
        ).candidate
    if policy == "keep_stay":
        keep_candidates = [
            item for item in candidates if item.action.operation == "KEEP_STAY"
        ]
        if len(keep_candidates) != 1:
            raise RuntimeError(
                "KEEP+STAY baseline requires exactly one legal KEEP_STAY candidate; "
                f"found={len(keep_candidates)}"
            )
        return keep_candidates[0]
    if policy == "uniform_random_legal":
        if not candidates:
            raise RuntimeError("Uniform legal baseline received no legal candidates")
        return candidates[int(rng.integers(0, len(candidates)))]
    raise ValueError(f"Unsupported evaluation policy: {policy!r}")


def evaluate_episode(
    environment: ExclusiveDQNEnvironment,
    *,
    split: str,
    policy: str,
    agent: DoubleDQNAgent | None = None,
) -> dict[str, Any]:
    """用同一正式执行器运行一个冻结策略，不访问 Replay 或优化器。"""

    policy = str(policy)
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported evaluation policy: {policy!r}")
    if policy == "dqn_greedy" and agent is None:
        raise ValueError("dqn_greedy evaluation requires a frozen agent")
    if policy != "dqn_greedy" and agent is not None:
        raise ValueError("Baseline evaluation must not receive or update a DQN agent")

    environment.reset()
    seed = int(environment.episode_seed if environment.episode_seed is not None else -1)
    rng = _seeded_policy_rng(split, seed)
    agent_rng_state = (
        None if agent is None else copy.deepcopy(agent.rng.bit_generator.state)
    )
    gradient_steps = None if agent is None else int(agent.gradient_steps)
    rows: list[dict[str, Any]] = []
    rewards: list[float] = []
    expected_slots = int(environment.execution_horizon_slots)
    arrival_horizon = int(environment.arrival_horizon_slots)
    while environment.can_act:
        state, candidates = environment.observe()
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("Evaluation state contains non-finite values")
        chosen = _choose_candidate(
            policy, state, candidates, agent=agent, rng=rng
        )
        transition = environment.step(chosen)
        diagnostics = transition.diagnostics
        slot_id = int(diagnostics["slot_id"])
        if slot_id != len(rows):
            raise RuntimeError(
                f"Formal episode slot sequence is not contiguous: expected={len(rows)}, actual={slot_id}"
            )
        reward = float(transition.reward)
        if not math.isfinite(reward):
            raise FloatingPointError(f"Evaluation reward is non-finite at slot {slot_id}")
        arrived_ids = [str(value) for value in diagnostics.get("arrived_task_ids", ())]
        if slot_id >= arrival_horizon and arrived_ids:
            raise RuntimeError(
                f"New task arrival occurred in execution tail: slot={slot_id}, tasks={arrived_ids}"
            )
        task_slot_execution = diagnostics.get("per_task_slot_execution", {})
        upload_energy = diagnostics.get("per_task_slot_upload_energy_j", {})
        if not isinstance(task_slot_execution, Mapping) or not isinstance(upload_energy, Mapping):
            raise RuntimeError("Formal executor omitted per-task execution/energy diagnostics")
        slot_upload_energy = math.fsum(float(value) for value in upload_energy.values())
        cache_cost = diagnostics.get("cache_cost")
        movement_cost = diagnostics.get("movement_cost")
        if cache_cost is None or movement_cost is None:
            raise RuntimeError("Formal executor omitted normalized system cost diagnostics")
        cache_cost = float(cache_cost)
        movement_cost = float(movement_cost)
        if not math.isfinite(cache_cost) or not math.isfinite(movement_cost):
            raise FloatingPointError("Evaluation system cost is non-finite")
        matching = diagnostics.get("matching_summary", {})
        resource = matching.get("resource", {}) if isinstance(matching, Mapping) else {}
        terminal_events = [
            _jsonable(dict(event))
            for event in diagnostics.get("task_termination_events", ())
        ]
        rows.append(
            {
                "slot_id": slot_id,
                "operation": str(transition.operation),
                "candidate_id": str(transition.candidate_id),
                "reward": reward,
                "arrived_task_ids": arrived_ids,
                "task_slot_count": len(task_slot_execution),
                "slot_upload_energy_j": slot_upload_energy,
                "task_termination_events": terminal_events,
                "active_task_count": int(diagnostics.get("active_task_count", 0)),
                "cache_cost": cache_cost,
                "uav_movement_cost": movement_cost,
                "uav_movement_energy_j": diagnostics.get("uav_movement_energy_j"),
                "uav_total_distance_m": diagnostics.get("uav_total_distance_m"),
                "resource_feasible": matching.get("resource_feasible")
                if isinstance(matching, Mapping)
                else None,
                "edge_cpu_fraction_sum_by_node": _jsonable(
                    resource.get("edge_cpu_fraction_sum_by_node")
                ) if isinstance(resource, Mapping) else None,
                "edge_bandwidth_fraction_sum_by_node": _jsonable(
                    resource.get("edge_bandwidth_fraction_sum_by_node")
                ) if isinstance(resource, Mapping) else None,
                "cloud_resource_model": matching.get("cloud_resource_model")
                if isinstance(matching, Mapping)
                else None,
            }
        )
        rewards.append(reward)

    observed_slots = [int(row["slot_id"]) for row in rows]
    if observed_slots != list(range(expected_slots)):
        raise RuntimeError(
            "Evaluation did not execute the complete contiguous Episode: "
            f"expected={expected_slots}, actual={len(observed_slots)}"
        )
    if not environment.terminated or observed_slots[-1] != expected_slots - 1:
        raise RuntimeError("Evaluation Episode did not terminate at its final execution slot")
    lifecycle = _summarize_task_lifecycle(rows)
    reward_sum = math.fsum(rewards)
    cache_total = math.fsum(float(row["cache_cost"]) for row in rows)
    movement_total = math.fsum(float(row["uav_movement_cost"]) for row in rows)
    if agent is not None:
        if agent.rng.bit_generator.state != agent_rng_state:
            raise RuntimeError("Greedy evaluation changed the agent exploration RNG")
        if int(agent.gradient_steps) != gradient_steps:
            raise RuntimeError("Greedy evaluation performed an optimizer update")
    return {
        "split": str(split),
        "episode_seed": seed,
        "policy": policy,
        "episode_return": reward_sum,
        "per_slot_reward_sum": reward_sum,
        "episode_return_matches_per_slot_reward_sum": True,
        "execution_slot_count": len(rows),
        "last_execution_slot": observed_slots[-1],
        "arrival_horizon_slots": arrival_horizon,
        "tail_arrival_task_count": sum(
            len(row["arrived_task_ids"])
            for row in rows
            if int(row["slot_id"]) >= arrival_horizon
        ),
        "cache_cost_total": cache_total,
        "cache_cost_per_slot_mean": cache_total / len(rows),
        "uav_movement_cost_total": movement_total,
        "uav_movement_cost_per_slot_mean": movement_total / len(rows),
        "slot_upload_energy_total_j": math.fsum(
            float(row["slot_upload_energy_j"]) for row in rows
        ),
        "resource_infeasible_slot_count": sum(
            row["resource_feasible"] is False for row in rows
        ),
        **lifecycle,
        "per_slot": rows,
    }


def _aggregate_policy_results(
    policy: str,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot aggregate an empty formal evaluation")
    returns = [float(row["episode_return"]) for row in results]
    total_tasks = sum(int(row["unique_task_count"]) for row in results)
    total_success = sum(int(row["successful_task_count"]) for row in results)
    total_timeout = sum(int(row["timed_out_task_count"]) for row in results)
    total_delay = math.fsum(float(row["total_task_lifetime_delay_s"]) for row in results)
    total_energy = math.fsum(
        float(row["total_cumulative_upload_energy_j"]) for row in results
    )
    return {
        "policy": policy,
        "completed_episode_count": len(results),
        "mean_episode_system_utility": math.fsum(returns) / len(returns),
        "std_episode_system_utility": float(np.std(np.asarray(returns, dtype=np.float64))),
        "total_unique_task_count": total_tasks,
        "total_task_slot_assignment_count": sum(
            int(row["task_slot_count"]) for row in results
        ),
        "task_success_rate": total_success / total_tasks if total_tasks else None,
        "task_timeout_rate": total_timeout / total_tasks if total_tasks else None,
        "mean_actual_lifetime_delay_s": total_delay / total_tasks if total_tasks else None,
        "total_actual_lifetime_delay_s": total_delay,
        "mean_cumulative_upload_energy_j_per_task": (
            total_energy / total_tasks if total_tasks else None
        ),
        "total_cumulative_upload_energy_j": total_energy,
        "mean_cache_cost_per_episode": math.fsum(
            float(row["cache_cost_total"]) for row in results
        ) / len(results),
        "total_cache_cost": math.fsum(
            float(row["cache_cost_total"]) for row in results
        ),
        "mean_uav_movement_cost_per_episode": math.fsum(
            float(row["uav_movement_cost_total"]) for row in results
        ) / len(results),
        "total_uav_movement_cost": math.fsum(
            float(row["uav_movement_cost_total"]) for row in results
        ),
    }


def evaluate_formal_split(
    project_root: str | Path,
    *,
    split: str,
    policies: Sequence[str],
    agent: DoubleDQNAgent | None = None,
    validated_pool: tuple[Path, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """对固定正式 split 串行评价所有策略；任何场景失败均中止整体结果。"""

    split = str(split).strip().lower()
    if not policies or any(policy not in SUPPORTED_POLICIES for policy in policies):
        raise ValueError(f"Invalid evaluation policy list: {policies!r}")
    if "dqn_greedy" in policies and agent is None:
        raise ValueError("dqn_greedy evaluation requires a frozen agent")
    root = Path(project_root).resolve()
    pool, pool_info = (
        validated_pool
        if validated_pool is not None
        else validate_formal_evaluation_pool(root, split)
    )
    if str(pool_info.get("split")) != split:
        raise ValueError("Validated evaluation pool split mismatch")
    pool_identity = evaluation_pool_identity(pool_info)
    agent_rng_state = (
        None if agent is None else copy.deepcopy(agent.rng.bit_generator.state)
    )
    gradient_steps = None if agent is None else int(agent.gradient_steps)
    global_python = random.getstate()
    global_numpy = np.random.get_state()
    global_torch = torch.get_rng_state()
    global_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    module_modes = [] if agent is None else [
        (module, bool(module.training))
        for module in (
            agent.online_network,
            agent.online_category_network,
            agent.target_network,
            agent.target_category_network,
        )
    ]
    episode_results: list[dict[str, Any]] = []
    try:
        for episode_seed in pool_identity["active_episode_seeds"]:
            for policy in policies:
                with contextlib.redirect_stdout(io.StringIO()):
                    environment = ExclusiveDQNEnvironment.from_formal_split_episode(
                        root,
                        split=split,
                        episode_seed=int(episode_seed),
                        origin_start=0,
                        max_steps=int(pool_info["execution_horizon_slots"]),
                        validated_pool=(pool, pool_info),
                    )
                result = evaluate_episode(
                    environment,
                    split=split,
                    policy=policy,
                    agent=agent if policy == "dqn_greedy" else None,
                )
                result["scene_sha256"] = pool_identity[
                    "active_episode_scene_sha256"
                ][str(episode_seed)]
                episode_results.append(result)
    finally:
        random.setstate(global_python)
        np.random.set_state(global_numpy)
        torch.set_rng_state(global_torch)
        if torch.cuda.is_available() and global_cuda is not None:
            torch.cuda.set_rng_state_all(global_cuda)
        if agent is not None:
            for module, training in module_modes:
                module.train(training)
            if agent.rng.bit_generator.state != agent_rng_state:
                raise RuntimeError("Frozen evaluation changed the agent exploration RNG")
            if int(agent.gradient_steps) != gradient_steps:
                raise RuntimeError("Frozen evaluation changed the optimizer update count")

    policy_results = {
        policy: [row for row in episode_results if row["policy"] == policy]
        for policy in policies
    }
    expected_count = len(pool_identity["active_episode_seeds"])
    if any(len(rows) != expected_count for rows in policy_results.values()):
        raise RuntimeError("Formal evaluation did not complete every active scene")
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "split": split,
        "status": "COMPLETE",
        "pool_identity": pool_identity,
        "policy_summaries": {
            policy: _aggregate_policy_results(policy, rows)
            for policy, rows in policy_results.items()
        },
        "episode_results": episode_results,
    }


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "FORMAL_SPLIT_CONTRACTS",
    "SUPPORTED_POLICIES",
    "evaluate_episode",
    "evaluate_formal_split",
    "evaluation_pool_identity",
    "policy_state_sha256",
    "should_update_best",
    "validate_formal_evaluation_pool",
]
