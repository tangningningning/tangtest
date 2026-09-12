"""Shared contracts for the V8 no-prediction PPO ablation."""

from __future__ import annotations

from typing import Any

import numpy as np


NO_PREDICTION_POLICY_SCHEMA = (
    "adaptive_cache_operation_group_replica_aware_v8_no_prediction_popart"
)
ZEROED_OBSERVATION_SLICES = (
    "predicted_vehicle_slice",
    "popularity_slice",
)


def is_no_prediction_context(context: Any) -> bool:
    return str(getattr(context, "policy_schema", "")) == NO_PREDICTION_POLICY_SCHEMA


def mask_no_prediction_observation(
    observation: Any,
    observation_spec: Any,
) -> np.ndarray:
    """Zero future-prediction fields while retaining structural state fields."""

    masked = np.asarray(observation, dtype=np.float32).copy()
    expected_shape = (int(observation_spec.observation_dim),)
    if masked.shape != expected_shape:
        raise ValueError(
            "no-prediction observation shape mismatch: "
            f"got={masked.shape}, expected={expected_shape}"
        )
    for slice_name in ZEROED_OBSERVATION_SLICES:
        masked[getattr(observation_spec, slice_name)] = 0.0
    if not np.all(np.isfinite(masked)):
        raise ValueError("no-prediction observation contains NaN/Inf")
    return masked


def zero_predicted_popularity(service_count: int) -> dict[int, float]:
    count = int(service_count)
    if count < 0:
        raise ValueError("service_count must be non-negative")
    return {service_id: 0.0 for service_id in range(count)}
