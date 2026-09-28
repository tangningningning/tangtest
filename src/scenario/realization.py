"""Small deterministic seed helper for repeated simulation episodes."""

from __future__ import annotations

import hashlib


SEED_PROTOCOL_VERSION = "adaptive_cache_mec_episode_seed_v1"
SUMO_MAX_SEED = 2_147_483_646

EPISODE_SEED_STREAMS = (
    "traffic_seed",
    "sumo_seed",
    "research_role_seed",
    "service_role_seed",
    "popularity_seed",
    "task_arrival_seed",
    "service_request_seed",
    "workload_seed",
    "tv_cache_seed",
    "vehicle_observation_seed",
    "scenario_seed",
)


def derive_namespaced_seed(episode_seed: int, namespace: str, stream_name: str) -> int:
    """Derive a deterministic seed outside the formal episode stream namespace.

    This is for auxiliary training-only work such as critic warmup.  It leaves
    ``EPISODE_SEED_STREAMS`` and all existing formal episode seed values intact.
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
    """Derive stable, named and independent seeds for one simulation episode."""

    seeds = {
        name: _derive(episode_seed, name) for name in EPISODE_SEED_STREAMS
    }
    if len(set(seeds.values())) != len(seeds):
        raise RuntimeError("episode seed streams unexpectedly collided")
    return seeds
