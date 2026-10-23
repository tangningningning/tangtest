"""为重复仿真 episode 提供稳定的种子派生和 realization 路径管理。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import project_config


SEED_PROTOCOL_VERSION = "adaptive_cache_mec_episode_seed_v1"
SUMO_MAX_SEED = 2_147_483_646

REALIZATION_ARTIFACT_FILENAMES = {
    "trajectory": "trajectory.csv",
    "vehicle_roles": "vehicle_roles.json",
    "tasks": "tasks.csv",
    "popularity": "popularity.csv",
    "popularity_history": "popularity_history.csv",
    "vehicle_predictions": "vehicle_predictions.csv",
    "popularity_predictions": "popularity_predictions.csv",
    "prediction_scenarios": "prediction_scenarios.npz",
}
# 单条 episode 生成器还会输出以下来源记录文件；完整性校验会同时检查这些文件。
GENERATOR_EXTRA_ARTIFACT_FILENAMES = (
    "sumo_demand.rou.xml",
    "sumo_demand.meta.json",
)


@dataclass(frozen=True)
class RealizationArtifacts:
    """一条完整 episode realization 所需文件的路径合同。"""

    directory: Path
    trajectory: Path
    vehicle_roles: Path
    tasks: Path
    popularity: Path
    popularity_history: Path
    vehicle_predictions: Path
    popularity_predictions: Path
    prediction_scenarios: Path


def configured_realization_artifacts(
    cfg: dict[str, Any], root: Path
) -> RealizationArtifacts:
    """根据配置路径构建 episode artifact 映射。"""

    paths = {
        field: project_config.resolve_path(
            root, project_config.get_config_value(cfg, f"paths.{field}")
        )
        for field in REALIZATION_ARTIFACT_FILENAMES
    }
    return RealizationArtifacts(
        directory=paths["trajectory"].parent,
        trajectory=paths["trajectory"],
        vehicle_roles=paths["vehicle_roles"],
        tasks=paths["tasks"],
        popularity=paths["popularity"],
        popularity_history=paths["popularity_history"],
        vehicle_predictions=paths["vehicle_predictions"],
        popularity_predictions=paths["popularity_predictions"],
        prediction_scenarios=paths["prediction_scenarios"],
    )


def _realization_artifacts_from_directory(directory: Path) -> RealizationArtifacts:
    """校验 realization 目录并返回规范 artifact 路径映射。"""

    realization_dir = Path(directory).expanduser().resolve(strict=False)
    if not realization_dir.is_dir():
        raise FileNotFoundError(
            f"environment episode realization directory does not exist: {realization_dir}"
        )
    paths = {
        field: realization_dir / filename
        for field, filename in REALIZATION_ARTIFACT_FILENAMES.items()
    }
    missing = [
        filename
        for field, filename in REALIZATION_ARTIFACT_FILENAMES.items()
        if not paths[field].is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "environment episode realization is missing required artifacts: "
            f"directory={realization_dir}, missing={missing}"
        )
    return RealizationArtifacts(
        directory=realization_dir,
        trajectory=paths["trajectory"],
        vehicle_roles=paths["vehicle_roles"],
        tasks=paths["tasks"],
        popularity=paths["popularity"],
        popularity_history=paths["popularity_history"],
        vehicle_predictions=paths["vehicle_predictions"],
        popularity_predictions=paths["popularity_predictions"],
        prediction_scenarios=paths["prediction_scenarios"],
    )


EPISODE_SEED_STREAMS = (
    "traffic_seed",
    "sumo_seed",
    "research_role_seed",
    "service_role_seed",
    "popularity_seed",
    "task_arrival_seed",
    "service_request_seed",
    "workload_seed",
    "vehicle_observation_seed",
    "scenario_seed",
)


def derive_namespaced_seed(episode_seed: int, namespace: str, stream_name: str) -> int:
    """在正式 episode seed stream 命名空间之外派生确定性种子。

    用于 critic warmup 等辅助训练，不会改变 ``EPISODE_SEED_STREAMS``
    或任何现有正式 episode seed 的取值。
    """

    if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
        raise TypeError("episode_seed must be an integer")
    if episode_seed < 0 or episode_seed > (1 << 64) - 1:
        raise ValueError("episode_seed must be in the unsigned 64-bit range")
    namespace_value = str(namespace).strip()
    stream_value = str(stream_name).strip()
    if not namespace_value or not stream_value:
        raise ValueError("seed namespace and stream name must be non-empty")
    payload = (
        f"{SEED_PROTOCOL_VERSION}|{episode_seed}|{namespace_value}|{stream_value}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _derive(episode_seed: int, stream_name: str) -> int:
    if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
        raise TypeError("episode_seed must be an integer")
    if episode_seed < 0 or episode_seed > (1 << 64) - 1:
        raise ValueError("episode_seed must be in the unsigned 64-bit range")
    if stream_name not in EPISODE_SEED_STREAMS:
        raise ValueError(f"unknown episode seed stream: {stream_name}")
    payload = (
        f"{SEED_PROTOCOL_VERSION}|{episode_seed}|{stream_name}"
    ).encode("ascii")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    if stream_name in {"traffic_seed", "sumo_seed"}:
        return value % (SUMO_MAX_SEED + 1)
    return value & ((1 << 63) - 1)


def derive_episode_seeds(episode_seed: int) -> dict[str, int]:
    """为单条仿真 episode 派生稳定、具名且相互独立的种子。"""

    seeds = {
        name: _derive(episode_seed, name) for name in EPISODE_SEED_STREAMS
    }
    if len(set(seeds.values())) != len(seeds):
        raise RuntimeError("episode seed streams unexpectedly collided")
    return seeds
