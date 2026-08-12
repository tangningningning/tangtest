"""标准 SB3 PPO 与第四章配置触发器之间的 Gymnasium 适配层。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:  # pragma: no cover
    gym = None
    spaces = None
    GYMNASIUM_AVAILABLE = False


PPO_CONFIGURATION_FEATURE_GROUPS = (
    "previous_cache",
    "previous_uav_positions",
    "predicted_vehicle_positions",
    "predicted_popularity",
    "episode_progress",
)


class _FallbackMultiBinary:
    def __init__(self, n: int):
        self.n = int(n)
        self.shape = (self.n,)
        self.dtype = np.dtype(np.int8)

    def contains(self, value: Any) -> bool:
        array = np.asarray(value)
        return bool(array.shape == self.shape and np.all((array == 0) | (array == 1)))


class _FallbackBox:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.dtype = np.dtype(np.float32)
        self.low = np.zeros(shape, dtype=np.float32)
        self.high = np.ones(shape, dtype=np.float32)

    def contains(self, value: Any) -> bool:
        array = np.asarray(value)
        return bool(
            array.shape == self.shape
            and array.dtype == self.dtype
            and np.all(np.isfinite(array))
            and np.all((array >= 0.0) & (array <= 1.0))
        )


_EnvBase = gym.Env if GYMNASIUM_AVAILABLE else object


class AdaptiveCachePPOEnv(_EnvBase):
    """MultiBinary(M+U) 只控制配置 trigger，配置内容由确定性 solver 产生。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        environment: Any,
        *,
        episode_seed_base: int | None = None,
        seed_mode: str = "sequential",
        fixed_seed: int | None = None,
    ) -> None:
        if GYMNASIUM_AVAILABLE:
            super().__init__()
        if str(environment.cache_controller_mode) != "ppo":
            raise ValueError("AdaptiveCachePPOEnv要求cache_controller.mode=ppo。")
        self.environment = environment
        mode = str(seed_mode).strip().lower()
        if mode not in {"sequential", "fixed"}:
            raise ValueError("seed_mode只能是sequential或fixed。")
        if mode == "fixed" and fixed_seed is None:
            raise ValueError("fixed seed_mode必须提供fixed_seed。")
        self._seed_mode = mode
        self._fixed_seed = None if fixed_seed is None else int(fixed_seed)
        self._episode_seed_base = None if episode_seed_base is None else int(episode_seed_base)
        self._episode_index = 0
        self._last_episode_seed: int | None = None
        self._needs_reset = True
        self._reset_action_layout: dict[str, Any] | None = None

        config = environment.config
        service_count = int(config["services"]["count"])
        sv_count = int(config["vehicle_roles"]["num_service_vehicles"])
        task_count = int(config["vehicle_roles"]["num_task_vehicles"])
        rsu_count = len(config["rsu"]["positions_m"])
        uav_count = len(config["uav"]["initial_positions_m"])
        cache_node_count = sv_count + rsu_count + uav_count
        vehicle_count = sv_count + task_count
        self._cache_node_count = cache_node_count
        self._uav_count = uav_count
        observation_dimension = (
            cache_node_count * service_count
            + 2 * uav_count
            + 2 * vehicle_count
            + service_count
            + 1
        )
        if GYMNASIUM_AVAILABLE:
            self.action_space = spaces.MultiBinary(cache_node_count + uav_count)
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(observation_dimension,),
                dtype=np.float32,
            )
        else:  # pragma: no cover
            self.action_space = _FallbackMultiBinary(cache_node_count + uav_count)
            self.observation_space = _FallbackBox((observation_dimension,))

    @property
    def feature_groups(self) -> tuple[str, ...]:
        return PPO_CONFIGURATION_FEATURE_GROUPS

    @property
    def action_layout(self) -> dict[str, Any]:
        return self.environment.action_layout

    @property
    def last_episode_seed(self) -> int | None:
        return self._last_episode_seed

    @property
    def seed_mode(self) -> str:
        return self._seed_mode

    @property
    def fixed_seed(self) -> int | None:
        return self._fixed_seed

    def _observation(self) -> np.ndarray:
        observation = np.asarray(
            self.environment.configuration_observation(), dtype=np.float32
        )
        if not self.observation_space.contains(observation):
            raise ValueError("configuration observation不满足动态Box契约。")
        return observation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if self._seed_mode == "fixed":
            resolved_seed = self._fixed_seed
        else:
            resolved_seed = seed
            if resolved_seed is None and self._episode_seed_base is not None:
                resolved_seed = self._episode_seed_base + self._episode_index
            self._episode_index += 1
        if GYMNASIUM_AVAILABLE:
            super().reset(seed=resolved_seed)
        self._last_episode_seed = None if resolved_seed is None else int(resolved_seed)
        self.environment.reset(seed=resolved_seed)
        layout = self.environment.action_layout
        if int(layout["action_length"]) != int(self.action_space.shape[0]):
            raise RuntimeError("reset后的节点布局与MultiBinary维度不一致。")
        self._reset_action_layout = layout
        observation = self._observation()
        self._needs_reset = False
        return observation, {
            "cache_controller_mode": "ppo",
            "ppo_feature_groups": self.feature_groups,
            "action_layout": layout,
            "ppo_raw_observation": observation.copy(),
            "episode_seed": self._last_episode_seed,
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise RuntimeError("PPO环境必须先调用reset()。")
        if self._reset_action_layout is not None and self.action_layout != self._reset_action_layout:
            raise RuntimeError("action_layout changed within a fixed episode")
        action_array = np.asarray(action, dtype=np.int8)
        if not self.action_space.contains(action_array):
            raise ValueError("PPO动作必须满足MultiBinary(M+U)契约。")
        cache_triggers = action_array[: self._cache_node_count].astype(bool)
        uav_triggers = action_array[self._cache_node_count :].astype(bool)
        previous_uav = np.asarray(self.environment._uav_positions, dtype=float).copy()
        _state, reward, terminated, base_info = self.environment.step(
            {
                "cache_triggers": cache_triggers,
                "uav_triggers": uav_triggers,
            }
        )
        info = dict(base_info)
        info.update(
            runtime_diagnostics(
                self.environment,
                previous_uav_positions=previous_uav,
                reward=reward,
                info=info,
            )
        )
        observation = self._observation()
        if self._reset_action_layout is not None and self.action_layout != self._reset_action_layout:
            raise RuntimeError("action_layout changed within a fixed episode")
        info.update(
            {
                "ppo_raw_reward": float(reward),
                "ppo_raw_observation": observation.copy(),
                "ppo_action": action_array.copy(),
                "action_layout": self.environment.action_layout,
                # 训练/验证记录使用当前 step 对应的 seed；DummyVecEnv 在
                # terminal 后会立即 reset，因此不能在 callback 中事后读取
                # last_episode_seed 来推断刚刚结束的 episode。
                "episode_seed": self._last_episode_seed,
                "requested_cache_trigger_count": int(np.sum(cache_triggers)),
                "requested_uav_trigger_count": int(np.sum(uav_triggers)),
            }
        )
        if terminated:
            self._needs_reset = True
        return observation, float(reward), bool(terminated), False, info

    def close(self) -> None:
        self._needs_reset = True
        close = getattr(self.environment, "close", None)
        if callable(close):
            close()


def sb3_check_env(environment: AdaptiveCachePPOEnv) -> str:
    if not GYMNASIUM_AVAILABLE:
        return "unavailable:gymnasium_missing"
    try:
        from stable_baselines3.common.env_checker import check_env
    except ImportError:  # pragma: no cover
        return "unavailable:stable_baselines3_missing"
    check_env(environment, warn=True)
    return "PASS"


def runtime_diagnostics(
    environment: Any,
    *,
    previous_uav_positions: np.ndarray | None,
    reward: float,
    info: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """只读检查容量、UAV、KKT 资源、因果契约和数值有限性。"""

    capacity_violations = compute_violations = bandwidth_violations = 0
    speed_violations = region_violations = safety_violations = 0
    finite = np.isfinite(float(reward))
    try:
        cache = np.asarray(environment._cache, dtype=float)
        sizes = np.asarray(environment._service_sizes, dtype=float)
        capacities = np.asarray(environment._cache_capacities, dtype=float)
        usage = np.sum(cache * sizes[np.newaxis, :], axis=1)
        capacity_violations = int(np.sum(usage > capacities + 1.0e-9))
        finite = bool(finite and np.all(np.isfinite(cache)) and np.all(np.isfinite(usage)))
        node_count = len(environment._service_node_types)
        for key in ("compute_allocations", "bandwidth_allocations"):
            values = np.asarray((info or {}).get(key, np.empty((0, 0))), dtype=float)
            if values.size == 0:
                continue
            finite = bool(finite and np.all(np.isfinite(values)))
            totals = np.sum(values, axis=0) if values.ndim == 2 and values.shape[1] == node_count else np.asarray([])
            count = int(np.sum(totals > 1.0 + 1.0e-9))
            if key.startswith("compute"):
                compute_violations += count
            else:
                bandwidth_violations += count

        positions = np.asarray(environment._uav_positions, dtype=float)
        region = environment.config["uav"]["region"]
        region_violations += int(np.sum(positions[:, 0] < float(region["x_min_m"]) - 1.0e-9))
        region_violations += int(np.sum(positions[:, 0] > float(region["x_max_m"]) + 1.0e-9))
        region_violations += int(np.sum(positions[:, 1] < float(region["y_min_m"]) - 1.0e-9))
        region_violations += int(np.sum(positions[:, 1] > float(region["y_max_m"]) + 1.0e-9))
        minimum = float(environment.config["uav"]["minimum_safety_distance_m"])
        for first in range(len(positions)):
            for second in range(first + 1, len(positions)):
                safety_violations += int(np.linalg.norm(positions[first] - positions[second]) < minimum - 1.0e-9)
        if previous_uav_positions is not None:
            maximum = float(environment.config["uav"]["max_speed_mps"]) * float(environment.config["simulation"]["decision_interval_s"])
            movement = np.linalg.norm(positions - np.asarray(previous_uav_positions), axis=1)
            speed_violations = int(np.sum(movement > maximum + 1.0e-9))
        finite = bool(finite and np.all(np.isfinite(positions)))
    except (KeyError, TypeError, ValueError, FloatingPointError):
        finite = False
        capacity_violations += 1
    prediction = getattr(environment, "last_prediction_diagnostics", {})
    mobility = getattr(environment, "mobility_diagnostics", {})
    base_info = info or {}
    return {
        "future_request_leakage": int(prediction.get("future_request_leakage", 0)),
        "future_trajectory_leakage": int(prediction.get("future_trajectory_leakage", 0)),
        "future_mobility_leakage": int(mobility.get("future_access_count", 0)),
        "capacity_violations": capacity_violations,
        "compute_violations": compute_violations,
        "bandwidth_violations": bandwidth_violations,
        "uav_speed_violations": speed_violations,
        "uav_region_violations": region_violations,
        "uav_safety_violations": safety_violations,
        "uav_violations": speed_violations + region_violations + safety_violations,
        "invalid_service_ids": int(base_info.get("invalid_service_ids", 0)),
        "invalid_node_ids": int(base_info.get("invalid_node_ids", 0)),
        "nan_inf_count": int(not finite),
        "kkt_failure_count": int(base_info.get("kkt_failure_count", 0)),
    }


__all__ = [
    "AdaptiveCachePPOEnv",
    "GYMNASIUM_AVAILABLE",
    "PPO_CONFIGURATION_FEATURE_GROUPS",
    "runtime_diagnostics",
    "sb3_check_env",
]
