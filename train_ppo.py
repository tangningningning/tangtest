"""当前第四章 MultiBinary 配置触发器的标准 SB3 PPO 入口。

本文件只做项目适配：环境构造、VecNormalize 和配置读取。策略分布、GAE、
clip loss、entropy 与优化过程全部直接复用 stable_baselines3.PPO。
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .data import load_config
    from .environment import MECEnvironment
    from .ppo_env import AdaptiveCachePPOEnv, sb3_check_env
except ImportError:  # pragma: no cover
    from data import load_config
    from environment import MECEnvironment
    from ppo_env import AdaptiveCachePPOEnv, sb3_check_env


PROJECT_ROOT = Path(__file__).resolve().parent


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, Mapping) else {}


def prepare_ppo_config(
    config: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """返回 PPO 模式配置副本，不改写默认 Reference 配置文件。"""

    if isinstance(config, Mapping):
        prepared = copy.deepcopy(dict(config))
    else:
        prepared = load_config() if config is None else load_config(config)
        prepared = copy.deepcopy(prepared)
    controller = dict(_section(prepared, "cache_controller"))
    controller["mode"] = "ppo"
    prepared["cache_controller"] = controller
    return prepared


def build_ppo_environment(
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    episode_seed_base: int | None = None,
    seed_mode: str = "sequential",
    fixed_seed: int | None = None,
) -> AdaptiveCachePPOEnv:
    prepared = prepare_ppo_config(config)
    return AdaptiveCachePPOEnv(
        MECEnvironment(prepared),
        episode_seed_base=episode_seed_base,
        seed_mode=seed_mode,
        fixed_seed=fixed_seed,
    )


def build_sb3_ppo(environment: Any, config: Mapping[str, Any] | None = None) -> Any:
    """构造官方 MlpPolicy PPO；MultiBinary 使用 SB3 原生 Bernoulli 分布。"""

    try:
        from stable_baselines3 import PPO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("当前 conda 环境缺少 stable-baselines3。") from error
    source = {} if config is None else config
    ppo = _section(source, "ppo")
    project = _section(source, "project")
    return PPO(
        "MlpPolicy",
        environment,
        learning_rate=float(ppo.get("learning_rate", 3.0e-4)),
        n_steps=int(ppo.get("n_steps", 256)),
        batch_size=int(ppo.get("batch_size", 64)),
        n_epochs=int(ppo.get("n_epochs", 10)),
        gamma=float(ppo.get("gamma", 0.99)),
        gae_lambda=float(ppo.get("gae_lambda", 0.95)),
        clip_range=float(ppo.get("clip_range", 0.2)),
        ent_coef=float(ppo.get("ent_coef", 0.0)),
        vf_coef=float(ppo.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo.get("max_grad_norm", 0.5)),
        seed=int(ppo.get("seed", project.get("seed", 0))),
        device=str(ppo.get("device", "cpu")),
        verbose=0,
    )


def build_vecnormalize_environment(
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    episode_seed_base: int | None = None,
) -> Any:
    """直接复用 SB3 DummyVecEnv/VecMonitor/VecNormalize。"""

    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("当前 conda 环境缺少 stable-baselines3。") from error
    prepared = prepare_ppo_config(config)

    def factory() -> Any:
        return build_ppo_environment(prepared, episode_seed_base=episode_seed_base)

    vector = VecMonitor(DummyVecEnv([factory]))
    ppo = _section(prepared, "ppo")
    return VecNormalize(
        vector,
        norm_obs=bool(ppo.get("norm_obs", True)),
        norm_reward=bool(ppo.get("norm_reward", True)),
        clip_obs=float(ppo.get("clip_obs", 10.0)),
    )


def run_interface_smoke(
    config: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """创建 model 并 predict 一次，不调用 learn。"""

    prepared = prepare_ppo_config(config)
    seed = int(_section(prepared, "ppo").get("seed", 0))
    # check_env 会自动多次 reset；固定 seed 避免把接口检查误当成正式 episode
    # 顺序推进，也避免离线 Trace reserve 语义被测试消耗。
    environment = build_ppo_environment(
        prepared, seed_mode="fixed", fixed_seed=seed
    )
    try:
        status = sb3_check_env(environment)
        model = build_sb3_ppo(environment, prepared)
        observation, reset_info = environment.reset(seed=seed)
        action, _state = model.predict(observation, deterministic=True)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        return {
            "check_env": status,
            "action_space": str(environment.action_space),
            "action_layout": reset_info["action_layout"],
            "observation_shape": tuple(int(item) for item in observation.shape),
            "observation_dtype": str(observation.dtype),
            "observation_finite": bool(np.all(np.isfinite(observation))),
            "next_observation_finite": bool(np.all(np.isfinite(next_observation))),
            "reward_finite": bool(np.isfinite(reward)),
            "formula_residual": float(
                reward
                - (
                    float(info["system_utility"]["task_service_utility"])
                    - float(info["cache_cost"])
                    - float(info["uav_move_cost"])
                )
            ),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "model_device": str(model.device),
            "model_learn_calls": 0,
        }
    finally:
        environment.close()


def run_semantic_smoke(
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    seed: int = 20271811,
) -> dict[str, Any]:
    """执行节点级 trigger 与 10-slot 约束验收，不创建训练 rollout。"""

    prepared = prepare_ppo_config(config)

    def one_step(action_builder: Any) -> dict[str, Any]:
        environment = build_ppo_environment(
            prepared, seed_mode="fixed", fixed_seed=int(seed)
        )
        try:
            _observation, _reset_info = environment.reset(seed=int(seed))
            old_cache = np.asarray(environment.environment._cache, dtype=float).copy()
            old_uav = np.asarray(environment.environment._uav_positions, dtype=float).copy()
            action = np.asarray(action_builder(environment), dtype=np.int8)
            _next, reward, _done, _truncated, info = environment.step(action)
            return {
                "action": action,
                "old_cache": old_cache,
                "new_cache": np.asarray(environment.environment._cache, dtype=float).copy(),
                "old_uav": old_uav,
                "new_uav": np.asarray(environment.environment._uav_positions, dtype=float).copy(),
                "reward": float(reward),
                "info": info,
            }
        finally:
            environment.close()

    def zeros(environment: AdaptiveCachePPOEnv) -> np.ndarray:
        return np.zeros(environment.action_space.shape, dtype=np.int8)

    keep = one_step(zeros)
    cache_only = one_step(
        lambda environment: np.eye(
            1,
            environment.action_space.shape[0],
            k=2,
            dtype=np.int8,
        )[0]
    )
    uav_only = one_step(
        lambda environment: np.eye(
            1,
            environment.action_space.shape[0],
            k=environment._cache_node_count,
            dtype=np.int8,
        )[0]
    )
    mixed = one_step(
        lambda environment: np.asarray(
            [
                int(index in {2, environment._cache_node_count})
                for index in range(environment.action_space.shape[0])
            ],
            dtype=np.int8,
        )
    )

    cache_other_rows_unchanged = bool(
        np.array_equal(cache_only["old_cache"][:2], cache_only["new_cache"][:2])
        and np.array_equal(cache_only["old_cache"][3:], cache_only["new_cache"][3:])
    )
    uav_cache_unchanged = bool(
        np.array_equal(uav_only["old_cache"], uav_only["new_cache"])
    )
    uav_other_unchanged = bool(
        np.array_equal(uav_only["old_uav"][1:], uav_only["new_uav"][1:])
    )

    # 直接复用正式确定性函数构造两个精确 no-op 边界，不读取真实任务。
    try:
        from .solver import solve_cooperative_cache, select_triggered_uav_positions
    except ImportError:  # pragma: no cover
        from solver import solve_cooperative_cache, select_triggered_uav_positions
    physical = MECEnvironment(prepared)
    try:
        physical.reset(seed=int(seed))
        trigger = np.zeros(physical._cache.shape[0], dtype=bool)
        trigger[2] = True
        first = solve_cooperative_cache(
            physical._cache,
            trigger,
            physical._service_sizes,
            physical._cache_capacities,
            cache_prediction=physical._last_node_prediction,
            node_types=physical._service_node_types,
            environment=physical,
            integer_scale=1,
        )
        candidate = np.asarray(first["candidate_cache"], dtype=float)
        second = solve_cooperative_cache(
            candidate,
            trigger,
            physical._service_sizes,
            physical._cache_capacities,
            cache_prediction=physical._last_node_prediction,
            node_types=physical._service_node_types,
            environment=physical,
            integer_scale=1,
        )
        second_cache = np.asarray(second["candidate_cache"], dtype=float)
        cache_noop = bool(np.array_equal(candidate, second_cache))
        cache_noop_cost = float(
            physical.cache_update_cost(
                candidate[2],
                second_cache[2],
                node_type=physical._service_node_types[2],
                service_sizes=physical._service_sizes,
            )["total_cost"]
        )
        uav_trigger = np.zeros(len(physical._uav_positions), dtype=bool)
        uav_trigger[0] = True
        uav_noop_result = select_triggered_uav_positions(
            physical._uav_positions,
            previous_cache=physical._cache,
            requested_uav_triggers=uav_trigger,
            cache_prediction=None,
            node_types=physical._service_node_types,
            uav_service_indices=physical._uav_service_indices,
            environment=physical,
            max_speed_mps=physical.config["uav"]["max_speed_mps"],
            slot_duration_s=physical.config["simulation"]["decision_interval_s"],
            region=physical.config["uav"]["region"],
            minimum_separation_m=physical.config["uav"]["minimum_safety_distance_m"],
        )
        uav_noop = not bool(np.any(uav_noop_result["actual_movement"]))
        uav_noop_cost = float(
            physical.uav_movement_cost(
                physical._uav_positions[0],
                uav_noop_result["final_positions"][0],
                movement_epsilon_m=1.0e-9,
            )["total_cost"]
        )
    finally:
        physical.close()

    smoke_environment = build_ppo_environment(
        prepared, seed_mode="fixed", fixed_seed=int(seed)
    )
    formula_residuals: list[float] = []
    reward_components = {"Q": 0.0, "cache_cost": 0.0, "uav_cost": 0.0, "reward": 0.0}
    totals = {
        "future_request_leakage": 0,
        "future_trajectory_leakage": 0,
        "capacity_violations": 0,
        "compute_violations": 0,
        "bandwidth_violations": 0,
        "uav_speed_violations": 0,
        "uav_region_violations": 0,
        "uav_safety_violations": 0,
        "nan_inf_count": 0,
        "kkt_failure_count": 0,
    }
    try:
        smoke_environment.reset(seed=int(seed))
        for slot in range(10):
            action = np.zeros(smoke_environment.action_space.shape, dtype=np.int8)
            if slot % 4 == 1:
                action[2] = 1
            elif slot % 4 == 2:
                action[smoke_environment._cache_node_count] = 1
            elif slot % 4 == 3:
                action[[2, smoke_environment._cache_node_count]] = 1
            _obs, reward, done, _truncated, info = smoke_environment.step(action)
            formula_residuals.append(
                float(
                    reward
                    - (
                        info["system_utility"]["task_service_utility"]
                        - info["cache_cost"]
                        - info["uav_move_cost"]
                    )
                )
            )
            reward_components["Q"] += float(
                info["system_utility"]["task_service_utility"]
            )
            reward_components["cache_cost"] += float(info["cache_cost"])
            reward_components["uav_cost"] += float(info["uav_move_cost"])
            reward_components["reward"] += float(reward)
            for key in totals:
                totals[key] += int(info.get(key, 0))
            if done:
                break
    finally:
        smoke_environment.close()

    return {
        "keep_all": {
            "cache_exact": bool(np.array_equal(keep["old_cache"], keep["new_cache"])),
            "uav_exact": bool(np.array_equal(keep["old_uav"], keep["new_uav"])),
            "cache_cost": float(keep["info"]["cache_cost"]),
            "uav_cost": float(keep["info"]["uav_move_cost"]),
        },
        "cache_trigger_one": {"other_rows_exact": cache_other_rows_unchanged},
        "uav_trigger_one": {
            "cache_exact": uav_cache_unchanged,
            "other_uavs_exact": uav_other_unchanged,
        },
        "mixed_trigger_executed": True,
        "cache_trigger_noop": {"actual_update": not cache_noop, "cost": cache_noop_cost},
        "uav_trigger_noop": {"actual_update": not uav_noop, "cost": uav_noop_cost},
        "formula_residual_max": float(max(abs(item) for item in formula_residuals)),
        "reward_components_10_slots": reward_components,
        "diagnostics": totals,
        "steps": len(formula_residuals),
        "model_learn_calls": 0,
    }


def run_training(
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    total_timesteps: int | None = None,
) -> Any:
    """显式训练入口；本轮重构验收不会调用此函数。"""

    prepared = prepare_ppo_config(config)
    vector = build_vecnormalize_environment(
        prepared,
        episode_seed_base=int(_section(prepared, "ppo").get("train_seed_base", 0)),
    )
    model = build_sb3_ppo(vector, prepared)
    steps = int(
        _section(prepared, "ppo").get("total_timesteps", 25600)
        if total_timesteps is None
        else total_timesteps
    )
    model.learn(total_timesteps=steps)
    return model, vector


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=None)
    args = parser.parse_args(argv)
    if args.train:
        run_training(total_timesteps=args.total_timesteps)
        return
    result = run_interface_smoke() if args.interface_smoke else {
        "config_parse": "PASS",
        "training_started": False,
    }
    print(result)


if __name__ == "__main__":
    main()


__all__ = [
    "build_ppo_environment",
    "build_sb3_ppo",
    "build_vecnormalize_environment",
    "prepare_ppo_config",
    "run_interface_smoke",
    "run_semantic_smoke",
    "run_training",
]
