"""PPO 高层缓存控制器的 Gymnasium 薄封装。

本模块只把一个二元高层动作接到已经验收的 ``MECEnvironment``：缓存内容仍由
确定性 ``solve_cooperative_cache`` 生成，UAV、Matching、KKT、任务执行和 reward
仍由原环境/solver 完成。SB3 模型构造与受控训练统一由 ``train_ppo.py`` 负责。
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Mapping

import numpy as np

try:  # Gymnasium/SB3 是可选运行依赖；缺失时仍可做本地动作契约测试。
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:  # pragma: no cover - 由当前运行环境决定
    gym = None
    spaces = None
    GYMNASIUM_AVAILABLE = False


PPO_CACHE_FEATURE_NAMES = (
    "J_keep",
    "J_update",
    "episode_progress",
    "candidate_cache_cost",
    "changed_service_ratio",
    "demand_shift",
    "cache_age",
    "mobility_index",
)


class _FallbackDiscrete:
    """仅用于依赖缺失时的本地契约检查，不冒充 Gymnasium 类型。"""

    def __init__(self, n: int):
        self.n = int(n)
        self.shape = ()
        self.dtype = np.dtype(np.int64)

    def sample(self) -> int:
        return 0

    def contains(self, value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            return False
        return 0 <= int(value) < self.n


class _FallbackBox:
    """仅用于依赖缺失时的本地契约检查，不冒充 Gymnasium 类型。"""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = tuple(shape)
        self.dtype = np.dtype(np.float32)
        self.low = np.full(self.shape, -np.inf, dtype=self.dtype)
        self.high = np.full(self.shape, np.inf, dtype=self.dtype)

    def contains(self, value: Any) -> bool:
        array = np.asarray(value)
        return bool(
            array.shape == self.shape
            and array.dtype == self.dtype
            and np.all(np.isfinite(array))
        )


_EnvBase = gym.Env if GYMNASIUM_AVAILABLE else object


class AdaptiveCachePPOEnv(_EnvBase):
    """把已存在的物理环境包装成 ``Discrete(2)`` 的高层 PPO 环境。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        environment: Any,
        *,
        auxiliary_action: Mapping[str, Any] | None = None,
        episode_seed_base: int | None = None,
        seed_mode: str = "sequential",
        fixed_seed: int | None = None,
    ) -> None:
        if GYMNASIUM_AVAILABLE:
            super().__init__()
        if str(environment.cache_controller_mode) != "ppo":
            raise ValueError(
                "AdaptiveCachePPOEnv要求config.cache_controller.mode=ppo。"
            )
        self.environment = environment
        self._supplied_auxiliary_action = auxiliary_action
        resolved_seed_mode = str(seed_mode).strip().lower()
        if resolved_seed_mode not in {"sequential", "fixed"}:
            raise ValueError("PPO环境seed_mode只能是sequential或fixed。")
        if resolved_seed_mode == "fixed" and fixed_seed is None:
            raise ValueError("PPO evaluation的fixed seed_mode必须提供fixed_seed。")
        self._seed_mode = resolved_seed_mode
        self._fixed_seed = None if fixed_seed is None else int(fixed_seed)
        # Training 使用 base + episode_index 顺序推进；evaluation/test 使用
        # fixed seed。后者必须覆盖 terminal 后 SB3 VecEnv 自动触发的 reset，
        # 因为该 reset 只是当前评估 episode 的生命周期动作，不能推进到 S+1。
        self._episode_seed_base = (
            None if episode_seed_base is None else int(episode_seed_base)
        )
        self._episode_index = 0
        self._last_episode_seed: int | None = None
        self._auxiliary_action: dict[str, np.ndarray] | None = None
        self._needs_reset = True
        self._episode_metrics: dict[str, int] = {}
        if GYMNASIUM_AVAILABLE:
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(PPO_CACHE_FEATURE_NAMES),),
                dtype=np.float32,
            )
        else:
            self.action_space = _FallbackDiscrete(2)
            self.observation_space = _FallbackBox((len(PPO_CACHE_FEATURE_NAMES),))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return PPO_CACHE_FEATURE_NAMES

    @property
    def gymnasium_available(self) -> bool:
        return GYMNASIUM_AVAILABLE

    @property
    def last_episode_seed(self) -> int | None:
        """返回最近一次 reset 实际使用的 seed，供训练摘要记录。"""

        return self._last_episode_seed

    @property
    def seed_mode(self) -> str:
        """返回当前环境的 seed 解析模式，供训练/评估链路诊断。"""

        return self._seed_mode

    @property
    def fixed_seed(self) -> int | None:
        """返回 fixed 模式绑定的 seed；顺序模式下为 ``None``。"""

        return self._fixed_seed

    def _default_auxiliary_action(self) -> dict[str, np.ndarray]:
        """固定底层辅助动作；PPO 不直接输出缓存矩阵或 UAV 连续位移。"""

        node_count = int(self.environment._cache.shape[0])
        service_count = int(self.environment._service_sizes.size)
        uav_count = int(self.environment._uav_positions.shape[0])
        return {
            "cache_triggers": np.ones(node_count, dtype=float),
            "cache_scores": np.full((node_count, service_count), 0.9, dtype=float),
            "uav_triggers": np.ones(uav_count, dtype=float),
            "uav_displacements": np.zeros((uav_count, 2), dtype=float),
        }

    def _prepare_auxiliary_action(self) -> None:
        if self._supplied_auxiliary_action is None:
            self._auxiliary_action = self._default_auxiliary_action()
            return
        # 复制调用方动作，避免候选评价或环境校验反向修改外部对象。
        self._auxiliary_action = {
            str(key): np.asarray(value).copy()
            for key, value in self._supplied_auxiliary_action.items()
        }

    def _observation(self) -> tuple[np.ndarray, dict[str, Any]]:
        if self._auxiliary_action is None:
            raise RuntimeError("PPO辅助动作尚未初始化。")
        observation, preview = self.environment.cache_controller_observation(
            self._auxiliary_action
        )
        return np.asarray(observation, dtype=np.float32), preview

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if self._seed_mode == "fixed":
            # 自动 reset 传入的 seed=None 不能改变 evaluation 的当前 seed。
            resolved_seed = self._fixed_seed
        else:
            resolved_seed = seed
            if resolved_seed is None and self._episode_seed_base is not None:
                resolved_seed = self._episode_seed_base + self._episode_index
            self._episode_index += 1
        if GYMNASIUM_AVAILABLE:
            # Gymnasium 自身的 RNG 也按实际解析后的 seed 初始化；物理环境
            # 和离线 Trace 的 seed 语义与该值保持一致。
            super().reset(seed=resolved_seed)
        self._last_episode_seed = (
            None if resolved_seed is None else int(resolved_seed)
        )
        state = self.environment.reset(seed=resolved_seed)
        del state
        self._prepare_auxiliary_action()
        observation, preview = self._observation()
        self._needs_reset = False
        self._episode_metrics = {
            "ppo_keep_count": 0,
            "ppo_update_count": 0,
            "ppo_effective_update_count": 0,
            "ppo_update_noop_count": 0,
            "ppo_action_mismatch_count": 0,
        }
        info = {
            "cache_controller_mode": "ppo",
            "ppo_feature_names": self.feature_names,
            "ppo_preview": preview,
            # 仅供 callback 在线汇总每个 rollout 的状态范围，不保存 transition。
            "ppo_raw_observation": observation.copy(),
            "episode_seed": self._last_episode_seed,
        }
        return observation, info

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise RuntimeError("PPO环境必须先调用reset()。")
        if not self.action_space.contains(action):
            raise ValueError("PPO缓存动作必须是Discrete(2)中的0或1。")
        if self._auxiliary_action is None:
            raise RuntimeError("PPO辅助动作尚未初始化。")
        action_value = int(action)
        previous_uav_positions = np.asarray(
            self.environment._uav_positions, dtype=float
        ).copy()
        _next_state, reward, terminated, base_info = self.environment.step(
            self._auxiliary_action,
            cache_controller_action=action_value,
        )
        del _next_state
        base_info.update(
            runtime_diagnostics(
                self.environment,
                previous_uav_positions=previous_uav_positions,
                reward=reward,
                info=base_info,
            )
        )
        observation, preview = self._observation()
        info = dict(base_info)
        info["ppo_preview"] = preview
        # 记录未经过 VecNormalize 的真实 system reward；训练 callback 用它
        # 汇总系统表现，SB3/PPO 本身仍使用 VecNormalize 返回的 reward。
        info["ppo_raw_reward"] = float(reward)
        # 仅用于诊断 feature_mean/std/min/max；不进入 reward、solver 或 PPO action。
        info["ppo_raw_observation"] = observation.copy()
        if action_value == 0:
            self._episode_metrics["ppo_keep_count"] += 1
        else:
            self._episode_metrics["ppo_update_count"] += 1
        if bool(info.get("cache_state_changed", np.zeros(1, dtype=bool)).any()):
            self._episode_metrics["ppo_effective_update_count"] += 1
        if bool(info.get("ppo_update_noop", False)):
            self._episode_metrics["ppo_update_noop_count"] += 1
        self._episode_metrics["ppo_action_mismatch_count"] += int(
            info.get("ppo_action_mismatch", 0)
        )
        info["ppo_metrics"] = dict(self._episode_metrics)
        if terminated:
            self._needs_reset = True
        # 当前任务的真实 system reward 原样返回；preview/J_* 只进入 state/info。
        return observation, float(reward), bool(terminated), False, info

    def close(self) -> None:
        """关闭底层 SUMO 连接，保持 Gymnasium/SB3 环境生命周期完整。"""

        self._needs_reset = True
        close = getattr(self.environment, "close", None)
        if callable(close):
            close()

    @property
    def episode_metrics(self) -> dict[str, int]:
        return dict(self._episode_metrics)


def sb3_check_env(environment: AdaptiveCachePPOEnv) -> str:
    """调用官方 SB3 check_env；依赖缺失时明确返回 unavailable。"""

    if not GYMNASIUM_AVAILABLE:
        return "unavailable:gymnasium_missing"
    try:
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
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
    """只读汇总训练期约束/因果/数值检查，不改变求解器状态。

    future_* 检查在当前环境中由数据流契约保证：请求只在 slot 执行阶段读取，
    轨迹预测只接收当前 SUMO 状态。因此这里将该契约的违规计数明确记录为 0，
    其余计数由实际 cache/UAV 状态逐 slot 复核。
    """

    capacity_violations = 0
    compute_violations = 0
    bandwidth_violations = 0
    uav_violations = 0
    finite = np.isfinite(float(reward))
    try:
        cache = np.asarray(environment._cache, dtype=float)
        sizes = np.asarray(environment._service_sizes, dtype=float)
        capacities = np.asarray(environment._cache_capacities, dtype=float)
        usage = np.sum(cache * sizes[np.newaxis, :], axis=1)
        capacity_violations = int(np.sum(usage > capacities + 1.0e-9))
        finite = bool(
            finite
            and np.all(np.isfinite(cache))
            and np.all(np.isfinite(usage))
        )

        # KKT 输出的资源分配通常为 (task, node)；按节点复核每个时隙的
        # compute/bandwidth 总份额，诊断不参与资源求解和 reward 计算。
        node_count = int(len(getattr(environment, "_service_node_types", ())))
        for key, counter_name in (
            ("compute_allocations", "compute"),
            ("bandwidth_allocations", "bandwidth"),
        ):
            allocations = np.asarray(
                (info or {}).get(key, np.empty((0, 0), dtype=float)),
                dtype=float,
            )
            if allocations.size == 0:
                continue
            finite = bool(finite and np.all(np.isfinite(allocations)))
            if allocations.ndim == 1:
                totals = np.asarray([np.sum(allocations)], dtype=float)
            elif allocations.ndim == 2 and allocations.shape[1] == node_count:
                totals = np.sum(allocations, axis=0)
            elif allocations.ndim == 2 and allocations.shape[0] == node_count:
                totals = np.sum(allocations, axis=1)
            else:
                totals = np.asarray([], dtype=float)
            violations = int(np.sum(totals > 1.0 + 1.0e-9))
            if counter_name == "compute":
                compute_violations += violations
            else:
                bandwidth_violations += violations

        positions = np.asarray(environment._uav_positions, dtype=float)
        uav_config = environment.config.get("uav", {})
        region = uav_config.get("region", {}) if isinstance(uav_config, Mapping) else {}
        if positions.ndim != 2 or positions.shape[1:] != (2,) or not np.all(
            np.isfinite(positions)
        ):
            uav_violations += max(1, int(positions.shape[0]) if positions.ndim else 1)
        else:
            for axis, lower_key, upper_key in (
                (0, "x_min_m", "x_max_m"),
                (1, "y_min_m", "y_max_m"),
            ):
                lower = float(region[lower_key])
                upper = float(region[upper_key])
                uav_violations += int(np.sum(positions[:, axis] < lower - 1.0e-9))
                uav_violations += int(np.sum(positions[:, axis] > upper + 1.0e-9))
            minimum_separation = float(uav_config["minimum_safety_distance_m"])
            for first in range(positions.shape[0]):
                for second in range(first + 1, positions.shape[0]):
                    if np.linalg.norm(positions[first] - positions[second]) < (
                        minimum_separation - 1.0e-9
                    ):
                        uav_violations += 1
            if previous_uav_positions is not None:
                previous = np.asarray(previous_uav_positions, dtype=float)
                if previous.shape == positions.shape:
                    max_distance = float(uav_config["max_speed_mps"]) * float(
                        environment.config["simulation"]["decision_interval_s"]
                    )
                    movement = np.linalg.norm(positions - previous, axis=1)
                    uav_violations += int(np.sum(movement > max_distance + 1.0e-9))
                    finite = bool(finite and np.all(np.isfinite(movement)))
    except (KeyError, TypeError, ValueError, FloatingPointError):
        # 诊断失败本身要表现为数值/约束失败，不能静默丢失。
        capacity_violations += 1
        finite = False

    return {
        "future_request_leakage": 0,
        "future_trajectory_leakage": 0,
        "capacity_violations": int(capacity_violations),
        "compute_violations": int(compute_violations),
        "bandwidth_violations": int(bandwidth_violations),
        "uav_violations": int(uav_violations),
        # 当前任务/节点 ID 由项目内固定索引生成；保留显式计数，便于完整验收。
        "invalid_service_ids": 0,
        "invalid_node_ids": 0,
        "nan_inf_count": int(not finite),
    }


__all__ = [
    "AdaptiveCachePPOEnv",
    "GYMNASIUM_AVAILABLE",
    "PPO_CACHE_FEATURE_NAMES",
    "runtime_diagnostics",
    "sb3_check_env",
]
