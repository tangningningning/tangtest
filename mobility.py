"""统一的车辆移动数据源与离线 SUMO 轨迹准备工具。

本模块只负责车辆移动状态：

* ``SumoMobilityProvider`` 通过官方 TraCI/libsumo 读取当前 slot；
* ``TraceMobilityProvider`` 按 slot 顺序回放离线 SUMO 真值；
* ``generate_mobility_trace`` 只启动 SUMO、记录车辆状态并保存 NPZ。

请求、车辆角色、预测、缓存、UAV、Matching、KKT 和 PPO 均不写入轨迹。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
TRACE_SCHEMA_VERSION = 1
TRACE_FILE_TEMPLATE = "seed_{seed}.npz"

_TRACE_FLOAT_FIELDS = (
    "x_m",
    "y_m",
    "speed_mps",
    "speed_x_mps",
    "speed_y_mps",
    "angle_deg",
    "acceleration_mps2",
    "allowed_speed_mps",
    "lane_position_m",
)
_TRACE_STRING_FIELDS = ("edge_id", "lane_id")
_TRACE_REQUIRED_FIELDS = {
    "schema_version",
    "slot_count",
    "slots",
    "times",
    "vehicle_ids",
    "active",
    "lane_index",
    *_TRACE_FLOAT_FIELDS,
    *_TRACE_STRING_FIELDS,
}
_TRACE_INITIAL_REQUIRED_FIELDS = {
    "initial_time_s",
    "initial_active",
    "initial_lane_index",
    *(f"initial_{field}" for field in _TRACE_FLOAT_FIELDS),
    *(f"initial_{field}" for field in _TRACE_STRING_FIELDS),
}
_TRACE_REQUIRED_FIELDS.update(_TRACE_INITIAL_REQUIRED_FIELDS)

_SUMO_START_COUNT = 0


def sumo_start_count() -> int:
    """Return the number of SUMO starts performed by this process."""

    return int(_SUMO_START_COUNT)


def _resolve_project_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def trace_root_from_config(config: Mapping[str, Any]) -> Path:
    mobility = config.get("mobility", {})
    if not isinstance(mobility, Mapping):
        mobility = {}
    root = _resolve_project_path(mobility.get("trace_root"))
    if root is None:
        raise ValueError(
            "mobility.source=trace requires mobility.trace_root; "
            "there is no SUMO fallback in trace mode"
        )
    return root


def trace_path(
    config: Mapping[str, Any] | None = None,
    *,
    seed: int,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    if root is None:
        if config is None:
            raise ValueError("trace_path requires config or root")
        root_path = trace_root_from_config(config)
    else:
        root_path = _resolve_project_path(root)
        if root_path is None:
            raise ValueError("trace root cannot be empty")
    return root_path / TRACE_FILE_TEMPLATE.format(seed=int(seed))


def _copy_states(
    states: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Copy provider output so callers cannot mutate the cached mobility state."""

    copied: dict[str, dict[str, Any]] = {}
    for vehicle_id, raw_state in states.items():
        state: dict[str, Any] = {}
        for key, value in raw_state.items():
            if isinstance(value, np.ndarray):
                state[key] = value.copy()
            else:
                state[key] = value
        copied[str(vehicle_id)] = state
    return copied


class MobilityProvider:
    """Minimal current-slot mobility interface shared by SUMO and replay."""

    source = "unknown"

    def reset(
        self,
        times_s: Sequence[float] | np.ndarray,
        *,
        begin_s: float = 0.0,
    ) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    def states_at(self, slot_index: int) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "future_access_count": 0,
            "sumo_start_count": 0,
        }


class SumoMobilityProvider(MobilityProvider):
    """Read SUMO states with the original decision-slot timing semantics."""

    source = "sumo"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        seed: int,
        configuration_file: str | os.PathLike[str] | None = None,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.configuration_file = configuration_file
        self._api: Any | None = None
        self._times = np.empty(0, dtype=np.float64)
        self._current_slot = -1
        self._states: dict[str, dict[str, Any]] = {}
        self._starts = 0

    @property
    def api(self) -> Any | None:
        return self._api

    def reset(
        self,
        times_s: Sequence[float] | np.ndarray,
        *,
        begin_s: float = 0.0,
    ) -> dict[str, dict[str, Any]]:
        global _SUMO_START_COUNT

        self.close()
        times = np.asarray(times_s, dtype=np.float64)
        if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
            raise ValueError("SUMO mobility times must be a finite non-empty vector")
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("SUMO mobility times must be strictly increasing")

        try:
            from .data import get_vehicle_states, start_sumo, step
        except ImportError:  # pragma: no cover - direct script execution
            from data import get_vehicle_states, start_sumo, step

        sumo = self.config.get("sumo", {})
        if not isinstance(sumo, Mapping):
            sumo = {}
        configuration_file = self.configuration_file or sumo.get("configuration_file")
        if configuration_file is None:
            raise ValueError("sumo mobility requires sumo.configuration_file")
        self._api = start_sumo(
            configuration_file,
            backend=str(sumo.get("backend", "traci")),
            executable=sumo.get("executable"),
            seed=self.seed,
        )
        self._starts = 1
        _SUMO_START_COUNT += 1
        try:
            first_time = float(times[0])
            previous_time = max(float(begin_s), first_time - float(times[1] - times[0]) if times.size > 1 else first_time)
            step(self._api, previous_time)
            self._times = times.copy()
            # Keep the pre-slot observation.  The first states_at(0) call
            # advances SUMO to the actual slot-0 decision time, matching the
            # original environment's reset/step order.
            self._current_slot = -1
            self._states = get_vehicle_states(self._api)
            return _copy_states(self._states)
        except Exception:
            self.close()
            raise

    def states_at(self, slot_index: int) -> dict[str, dict[str, Any]]:
        index = int(slot_index)
        if self._api is None or self._current_slot < -1:
            raise RuntimeError("SumoMobilityProvider must be reset before reading states")
        if index < self._current_slot:
            raise RuntimeError("mobility slots cannot move backwards")
        if index >= self._times.size:
            raise IndexError(f"mobility slot out of range: {index}")
        try:
            from .data import get_vehicle_states, step
        except ImportError:  # pragma: no cover - direct script execution
            from data import get_vehicle_states, step
        for next_slot in range(self._current_slot + 1, index + 1):
            step(self._api, float(self._times[next_slot]))
            self._states = get_vehicle_states(self._api)
        self._current_slot = index
        return _copy_states(self._states)

    def close(self) -> None:
        if self._api is not None:
            try:
                from .data import close
            except ImportError:  # pragma: no cover - direct script execution
                from data import close
            try:
                close(self._api)
            finally:
                self._api = None
        self._current_slot = -1
        self._states = {}

    def diagnostics(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "future_access_count": 0,
            "sumo_start_count": int(self._starts),
            "last_slot": int(self._current_slot),
        }


class TraceMobilityProvider(MobilityProvider):
    """Causal replay of one offline SUMO ground-truth trace.

    The NPZ is intentionally kept private to the provider.  ``states_at(t)``
    returns only the active state at the requested slot, so the environment
    cannot accidentally pass future trajectory rows to the predictor.
    """

    source = "trace"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        seed: int,
        trace_file: str | os.PathLike[str] | None = None,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.path = (
            _resolve_project_path(trace_file)
            if trace_file is not None
            else trace_path(config, seed=self.seed)
        )
        if self.path is None or not self.path.is_file():
            raise FileNotFoundError(
                "mobility.source=trace requires an offline trace file; "
                f"missing {self.path}. Run generate_mobility_traces.py first. "
                "Trace mode never falls back to SUMO."
            )
        self._archive: dict[str, np.ndarray] = {}
        self._load_archive()
        self._times = np.asarray(self._archive["times"], dtype=np.float64)
        self.vehicle_ids = tuple(str(item) for item in self._archive["vehicle_ids"])
        self._columns = {vehicle_id: index for index, vehicle_id in enumerate(self.vehicle_ids)}
        self._current_slot = -1
        self._requested_slots = 0

    def _load_archive(self) -> None:
        try:
            with np.load(self.path, allow_pickle=False) as archive:
                names = set(archive.files)
                missing = _TRACE_REQUIRED_FIELDS.difference(names)
                if missing:
                    raise ValueError(
                        f"trace {self.path} is missing fields: {sorted(missing)}"
                    )
                schema_version = int(np.asarray(archive["schema_version"]).item())
                if schema_version != TRACE_SCHEMA_VERSION:
                    raise ValueError(
                        f"unsupported mobility trace schema {schema_version}; "
                        f"expected {TRACE_SCHEMA_VERSION}"
                    )
                self._archive = {
                    name: np.asarray(archive[name]).copy()
                    for name in archive.files
                    if name in _TRACE_REQUIRED_FIELDS
                }
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"cannot read mobility trace {self.path}: {error}") from error

        slot_count = int(np.asarray(self._archive["slot_count"]).item())
        times = np.asarray(self._archive["times"], dtype=np.float64)
        vehicle_ids = np.asarray(self._archive["vehicle_ids"])
        if slot_count <= 0 or times.shape != (slot_count,):
            raise ValueError(f"invalid slot_count/times in mobility trace {self.path}")
        slots = np.asarray(self._archive["slots"], dtype=np.int64)
        if slots.shape != (slot_count,) or not np.array_equal(
            slots, np.arange(slot_count, dtype=np.int64)
        ):
            raise ValueError(f"trace slots must equal 0..slot_count-1: {self.path}")
        if not np.all(np.isfinite(times)) or (
            times.size > 1 and np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError(f"trace times are not finite and increasing: {self.path}")
        if vehicle_ids.ndim != 1 or vehicle_ids.size == 0:
            raise ValueError(f"trace vehicle_ids must be a non-empty vector: {self.path}")
        if len(set(str(item) for item in vehicle_ids)) != vehicle_ids.size:
            raise ValueError(f"trace vehicle_ids contain duplicates: {self.path}")
        expected_shape = (slot_count, vehicle_ids.size)
        if np.asarray(self._archive["active"]).shape != expected_shape:
            raise ValueError(f"trace active shape mismatch: {self.path}")
        for field in (*_TRACE_FLOAT_FIELDS, "lane_index", *_TRACE_STRING_FIELDS):
            if np.asarray(self._archive[field]).shape != expected_shape:
                raise ValueError(f"trace field {field} shape mismatch: {self.path}")
        for field in (*_TRACE_FLOAT_FIELDS, "lane_index", *_TRACE_STRING_FIELDS):
            if np.asarray(self._archive[f"initial_{field}"]).shape != (vehicle_ids.size,):
                raise ValueError(
                    f"trace initial field {field} shape mismatch: {self.path}"
                )
        initial_time = float(np.asarray(self._archive["initial_time_s"]).item())
        if not math.isfinite(initial_time):
            raise ValueError(f"trace initial_time_s is not finite: {self.path}")

    def reset(
        self,
        times_s: Sequence[float] | np.ndarray,
        *,
        begin_s: float = 0.0,
    ) -> dict[str, dict[str, Any]]:
        del begin_s
        requested = np.asarray(times_s, dtype=np.float64)
        if requested.ndim != 1 or requested.size == 0:
            raise ValueError("trace replay times must be a non-empty vector")
        if requested.size > self._times.size:
            raise ValueError(
                f"trace {self.path} has {self._times.size} slots, "
                f"but the environment requests {requested.size}"
            )
        if not np.allclose(
            requested,
            self._times[: requested.size],
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                f"trace times do not match the environment slot semantics: {self.path}"
            )
        self._current_slot = -1
        self._requested_slots = 0
        return self._states_from_arrays(None)

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        text = str(value)
        return text if text else None

    def _states_from_arrays(
        self,
        slot_index: int | None,
    ) -> dict[str, dict[str, Any]]:
        initial = slot_index is None
        prefix = "initial_" if initial else ""
        active = np.asarray(
            self._archive[f"{prefix}active"]
            if initial
            else self._archive["active"][int(slot_index)],
            dtype=bool,
        )
        simulation_time = float(
            np.asarray(self._archive["initial_time_s"]).item()
            if initial
            else self._times[int(slot_index)]
        )
        states: dict[str, dict[str, Any]] = {}
        for vehicle_index, vehicle_id in enumerate(self.vehicle_ids):
            if not bool(active[vehicle_index]):
                continue
            def value(field: str) -> Any:
                source = self._archive[f"initial_{field}"] if initial else self._archive[field]
                return source[vehicle_index] if initial else source[int(slot_index), vehicle_index]

            position = np.asarray((value("x_m"), value("y_m")), dtype=np.float64)
            speed = float(value("speed_mps"))
            if position.shape != (2,) or not np.all(np.isfinite(position)):
                raise ValueError(f"active trace state has invalid position: {self.path}")
            if not math.isfinite(speed) or speed < 0.0:
                raise ValueError(f"active trace state has invalid speed: {self.path}")
            lane_index = int(value("lane_index"))
            states[vehicle_id] = {
                "vehicle_id": vehicle_id,
                "position_m": position,
                "speed_mps": speed,
                "speed_x_mps": float(value("speed_x_mps")),
                "speed_y_mps": float(value("speed_y_mps")),
                "angle_deg": float(value("angle_deg")),
                "lane_id": self._optional_string(value("lane_id")),
                "lane_index": None if lane_index < 0 else lane_index,
                "lane_position_m": float(value("lane_position_m")),
                "edge_id": self._optional_string(value("edge_id")),
                "acceleration_mps2": float(value("acceleration_mps2")),
                "allowed_speed_mps": float(value("allowed_speed_mps")),
                "active": True,
                "valid": True,
                "simulation_time_s": simulation_time,
            }
        return states

    def states_at(self, slot_index: int) -> dict[str, dict[str, Any]]:
        index = int(slot_index)
        same_slot = index == self._current_slot
        if self._current_slot < 0 and index != 0:
            raise RuntimeError("causal trace replay must start at slot 0")
        if self._current_slot >= 0 and index not in {
            self._current_slot,
            self._current_slot + 1,
        }:
            if index < self._current_slot:
                raise RuntimeError("trace mobility slots cannot move backwards")
            raise RuntimeError(
                "causal trace replay cannot skip a slot; "
                f"requested {index} after {self._current_slot}"
            )
        if index < 0 or index >= self._times.size:
            raise IndexError(f"trace mobility slot out of range: {index}")

        states = self._states_from_arrays(index)
        self._current_slot = index
        if not same_slot:
            self._requested_slots += 1
        return states

    def diagnostics(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "trace_file": str(self.path),
            "trace_seed": self.seed,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "future_access_count": 0,
            "requested_slots": int(self._requested_slots),
            "last_slot": int(self._current_slot),
            "sumo_start_count": 0,
        }


def _trace_vehicle_ids(config: Mapping[str, Any]) -> tuple[str, ...]:
    resolved = config.get("_resolved")
    if isinstance(resolved, Mapping) and resolved.get("all_vehicle_ids") is not None:
        vehicle_ids = tuple(dict.fromkeys(str(item) for item in resolved["all_vehicle_ids"]))
        if vehicle_ids:
            return vehicle_ids
    try:
        from .data import _discover_vehicle_ids_from_routes
    except ImportError:  # pragma: no cover - direct script execution
        from data import _discover_vehicle_ids_from_routes
    sumo = config.get("sumo", {})
    if not isinstance(sumo, Mapping) or sumo.get("configuration_file") is None:
        raise ValueError("cannot determine SUMO vehicle universe for trace generation")
    return _discover_vehicle_ids_from_routes(sumo["configuration_file"])


def generate_mobility_trace(
    config: Mapping[str, Any],
    *,
    seed: int,
    output_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Generate one trace from SUMO using the environment's exact slot timing."""

    simulation = config.get("simulation", {})
    if not isinstance(simulation, Mapping):
        raise ValueError("simulation configuration is required")
    begin_s = float(simulation.get("begin_s", 0.0))
    optimization_begin_s = float(simulation["optimization_begin_s"])
    optimization_end_s = float(simulation["optimization_end_s"])
    interval_s = float(simulation["decision_interval_s"])
    slot_count = int(simulation["slot_count"])
    times = np.arange(
        optimization_begin_s,
        optimization_end_s,
        interval_s,
        dtype=np.float64,
    )
    if times.shape != (slot_count,):
        raise ValueError("simulation times do not match simulation.slot_count")

    provider = SumoMobilityProvider(
        config,
        seed=int(seed),
        configuration_file=(
            config.get("sumo", {}).get("configuration_file")
            if isinstance(config.get("sumo"), Mapping)
            else None
        ),
    )
    vehicle_ids = _trace_vehicle_ids(config)
    column_index = {vehicle_id: index for index, vehicle_id in enumerate(vehicle_ids)}
    vehicle_count = len(vehicle_ids)
    active = np.zeros((slot_count, vehicle_count), dtype=bool)
    numeric = {
        field: np.full((slot_count, vehicle_count), np.nan, dtype=np.float64)
        for field in _TRACE_FLOAT_FIELDS
    }
    lane_index = np.full((slot_count, vehicle_count), -1, dtype=np.int64)
    strings = {
        field: np.full((slot_count, vehicle_count), "", dtype="U256")
        for field in _TRACE_STRING_FIELDS
    }
    initial_active = np.zeros(vehicle_count, dtype=bool)
    initial_numeric = {
        field: np.full(vehicle_count, np.nan, dtype=np.float64)
        for field in _TRACE_FLOAT_FIELDS
    }
    initial_lane_index = np.full(vehicle_count, -1, dtype=np.int64)
    initial_strings = {
        field: np.full(vehicle_count, "", dtype="U256")
        for field in _TRACE_STRING_FIELDS
    }

    def record_states(
        states: Mapping[str, Mapping[str, Any]],
        *,
        slot_index: int | None,
    ) -> None:
        for vehicle_id, state in states.items():
            if vehicle_id not in column_index:
                raise ValueError(
                    f"SUMO returned vehicle {vehicle_id!r} not present in route universe"
                )
            vehicle_index = column_index[vehicle_id]
            position = np.asarray(state["position_m"], dtype=np.float64)
            if position.shape != (2,) or not np.all(np.isfinite(position)):
                raise ValueError(f"SUMO returned invalid position for {vehicle_id}")
            target_active = initial_active if slot_index is None else active[slot_index]
            target_numeric = (
                initial_numeric if slot_index is None else {
                    field: numeric[field][slot_index] for field in _TRACE_FLOAT_FIELDS
                }
            )
            target_lane_index = (
                initial_lane_index
                if slot_index is None
                else lane_index[slot_index]
            )
            target_strings = (
                initial_strings
                if slot_index is None
                else {field: strings[field][slot_index] for field in _TRACE_STRING_FIELDS}
            )
            target_active[vehicle_index] = True
            target_numeric["x_m"][vehicle_index] = position[0]
            target_numeric["y_m"][vehicle_index] = position[1]
            for field, state_key in (
                ("speed_mps", "speed_mps"),
                ("speed_x_mps", "speed_x_mps"),
                ("speed_y_mps", "speed_y_mps"),
                ("angle_deg", "angle_deg"),
                ("acceleration_mps2", "acceleration_mps2"),
                ("allowed_speed_mps", "allowed_speed_mps"),
                ("lane_position_m", "lane_position_m"),
            ):
                target_numeric[field][vehicle_index] = float(
                    state.get(state_key, math.nan)
                )
            raw_lane_index = state.get("lane_index")
            if raw_lane_index is not None:
                target_lane_index[vehicle_index] = int(raw_lane_index)
            for field in _TRACE_STRING_FIELDS:
                value = state.get(field)
                if value is not None:
                    target_strings[field][vehicle_index] = str(value)

    try:
        initial_states = provider.reset(times, begin_s=begin_s)
        record_states(initial_states, slot_index=None)
        for slot_index in range(slot_count):
            states = provider.states_at(slot_index)
            record_states(states, slot_index=slot_index)
    finally:
        provider.close()

    output = (
        trace_path(config, seed=int(seed))
        if output_path is None
        else _resolve_project_path(output_path)
    )
    if output is None:
        raise ValueError("trace output path cannot be empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(TRACE_SCHEMA_VERSION, dtype=np.int64),
        seed=np.asarray(int(seed), dtype=np.int64),
        slot_count=np.asarray(slot_count, dtype=np.int64),
        slots=np.arange(slot_count, dtype=np.int64),
        initial_time_s=np.asarray(
            max(begin_s, float(times[0] - interval_s)), dtype=np.float64
        ),
        times=times,
        vehicle_ids=np.asarray(vehicle_ids, dtype="U256"),
        active=active,
        lane_index=lane_index,
        initial_active=initial_active,
        initial_lane_index=initial_lane_index,
        **numeric,
        **strings,
        **{f"initial_{field}": value for field, value in initial_numeric.items()},
        **{f"initial_{field}": value for field, value in initial_strings.items()},
    )
    return output


# Compatibility alias with the singular name used by a few data-preparation
# callers; the on-disk convention remains one ``seed_<id>.npz`` per seed.
trace_file = trace_path


__all__ = [
    "MobilityProvider",
    "SumoMobilityProvider",
    "TraceMobilityProvider",
    "TRACE_SCHEMA_VERSION",
    "generate_mobility_trace",
    "sumo_start_count",
    "trace_file",
    "trace_path",
    "trace_root_from_config",
]
