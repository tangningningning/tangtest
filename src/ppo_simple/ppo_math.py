"""Shared PPO arithmetic used by the Stage3 joint policy optimizer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _normalize_advantages(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    mean = float(array.mean())
    std = float(array.std())
    if std <= 1.0e-8:
        return np.zeros_like(array), mean, std
    return (array - mean) / (std + 1.0e-8), mean, std


def _clipped_ppo_surrogate_loss(
    ratio: Any, advantage: Any, clip: float
) -> torch.Tensor:
    return -torch.minimum(
        ratio * advantage,
        torch.clamp(ratio, 1.0 - float(clip), 1.0 + float(clip)) * advantage,
    ).mean()
