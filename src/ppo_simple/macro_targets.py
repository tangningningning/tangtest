"""Canonical target builders shared by Macro training and trace validation."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_macro_n2_target(
    macro_reward: Any,
    old_value: Any,
    actual_duration: Any,
    done: Any,
    gamma: float,
) -> np.ndarray:
    """Build one episode of frozen two-step SMDP TD targets in float64.

    The caller may cast the returned targets to the model's training dtype. This
    pure function is also used by reporting so target semantics and arithmetic
    stay identical across training and trace reconstruction.
    """

    rewards = np.asarray(macro_reward, dtype=np.float64).reshape(-1)
    values = np.asarray(old_value, dtype=np.float64).reshape(-1)
    raw_durations = np.asarray(actual_duration, dtype=np.float64).reshape(-1)
    terminal = np.asarray(done, dtype=bool).reshape(-1)
    if not (rewards.shape == values.shape == raw_durations.shape == terminal.shape):
        raise ValueError("N2 Macro target arrays must have matching shapes.")
    if (
        not np.isfinite(raw_durations).all()
        or np.any(raw_durations != np.floor(raw_durations))
    ):
        raise ValueError("N2 Macro target actual durations must be finite integers.")
    durations = raw_durations.astype(np.int64)
    if rewards.size == 0 or np.any(durations <= 0):
        raise ValueError("N2 Macro target episode and actual durations must be non-empty/positive.")
    if not bool(terminal[-1]) or bool(np.any(terminal[:-1])):
        raise ValueError("N2 Macro target input must contain exactly one complete episode.")
    gamma64 = np.float64(gamma)
    if not np.isfinite(gamma64) or gamma64 < 0.0:
        raise ValueError("N2 Macro target gamma must be finite and non-negative.")
    if not np.isfinite(rewards).all() or not np.isfinite(values).all():
        raise ValueError("N2 Macro target inputs contain NaN/Inf.")

    targets = np.empty(rewards.shape, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for index in range(len(rewards)):
            if terminal[index]:
                targets[index] = rewards[index]
                continue
            first_discount = np.power(gamma64, np.float64(durations[index]))
            next_index = index + 1
            if terminal[next_index]:
                targets[index] = rewards[index] + first_discount * rewards[next_index]
                continue
            combined_duration = np.float64(durations[index] + durations[next_index])
            second_discount = np.power(gamma64, combined_duration)
            targets[index] = (
                rewards[index]
                + first_discount * rewards[next_index]
                + second_discount * values[index + 2]
            )
    if not np.isfinite(targets).all():
        raise ValueError("N2 Macro target contains NaN/Inf.")
    return targets


def build_macro_smdp_lambda_return_target(
    macro_reward: Any,
    old_next_value: Any,
    actual_duration: Any,
    done: Any,
    gamma: float,
    critic_lambda: float,
) -> np.ndarray:
    """Build one episode of duration-aware SMDP lambda-return targets in float64.

    This critic-only return is intentionally independent of the Macro Actor's
    GAE lambda. The final terminal transition is reward-only; each earlier
    transition blends its frozen next-state value with the recursively built
    target for the next transition using ``critic_lambda ** actual_duration``.
    """

    rewards = np.asarray(macro_reward, dtype=np.float64).reshape(-1)
    next_values = np.asarray(old_next_value, dtype=np.float64).reshape(-1)
    raw_durations = np.asarray(actual_duration, dtype=np.float64).reshape(-1)
    terminal = np.asarray(done, dtype=bool).reshape(-1)
    if not (rewards.shape == next_values.shape == raw_durations.shape == terminal.shape):
        raise ValueError("SMDP lambda-return Macro target arrays must have matching shapes.")
    if (
        not np.isfinite(raw_durations).all()
        or np.any(raw_durations != np.floor(raw_durations))
    ):
        raise ValueError("SMDP lambda-return actual durations must be finite integers.")
    durations = raw_durations.astype(np.int64)
    if rewards.size == 0 or np.any(durations <= 0):
        raise ValueError(
            "SMDP lambda-return episode and actual durations must be non-empty/positive."
        )
    if not bool(terminal[-1]) or bool(np.any(terminal[:-1])):
        raise ValueError(
            "SMDP lambda-return input must contain exactly one complete episode."
        )
    gamma64 = np.float64(gamma)
    lambda64 = np.float64(critic_lambda)
    if not np.isfinite(gamma64) or gamma64 < 0.0:
        raise ValueError("SMDP lambda-return gamma must be finite and non-negative.")
    if not np.isfinite(lambda64) or not 0.0 <= lambda64 <= 1.0:
        raise ValueError("SMDP lambda-return critic lambda must be finite and in [0, 1].")
    if not np.isfinite(rewards).all() or not np.isfinite(next_values).all():
        raise ValueError("SMDP lambda-return inputs contain NaN/Inf.")

    targets = np.empty(rewards.shape, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for index in range(len(rewards) - 1, -1, -1):
            if terminal[index]:
                targets[index] = rewards[index]
                continue
            duration = np.float64(durations[index])
            continuation = np.power(lambda64, duration)
            discount = np.power(gamma64, duration)
            targets[index] = rewards[index] + discount * (
                (1.0 - continuation) * next_values[index]
                + continuation * targets[index + 1]
            )
    if not np.isfinite(targets).all():
        raise ValueError("SMDP lambda-return target contains NaN/Inf.")
    return targets
