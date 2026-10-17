"""Bounded cache-action counterfactual labels and auxiliary value losses.

The replay path uses immutable aligned Training inputs and the formal slot
executor. It does not approximate Matching+KKT or alter the environment reward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import copy
import hashlib
import math
import pickle
import random
import time
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:  # pragma: no cover - import-only environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from decision import cache_configuration, slot_environment, uav_configuration

from .rollout import (
    _check_request_contract,
    _check_reward,
    _observation,
    _positions_condition,
    _uav_candidates,
)


COUNTERFACTUAL_VALUE_SCHEMA = "mappo_cache_delta_q_counterfactual_v2_common_micro_uniforms"
CACHE_NODE_VALUE_FEATURE_DIM = 7


def _snapshot_value(value: Any) -> Any:
    """Encode trace input data without pickling runtime class identities.

    Aligned Training steps are dataclasses imported through the environment
    loader.  If a process loaded that same source module under two import
    identities, pickle rejected an otherwise valid snapshot because the class
    object no longer matched ``sys.modules``.  The digest needs the exact data,
    not Python's module identity, so encode dataclass fields and containers
    structurally while preserving the original objects for replay.
    """

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            contents = _snapshot_value(value.tolist())
        else:
            contents = value.tobytes(order="C")
        return ("ndarray", value.dtype.str, tuple(value.shape), contents)
    if isinstance(value, np.generic):
        return ("numpy_scalar", value.dtype.str, _snapshot_value(value.item()))
    if is_dataclass(value) and not isinstance(value, type):
        return (
            "dataclass",
            f"{type(value).__module__}.{type(value).__qualname__}",
            tuple(
                (field.name, _snapshot_value(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (_snapshot_value(key), _snapshot_value(item))
                for key, item in value.items()
            ),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_snapshot_value(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_snapshot_value(item) for item in value))
    if isinstance(value, set):
        encoded = [_snapshot_value(item) for item in value]
        encoded.sort(key=lambda item: pickle.dumps(item, protocol=5))
        return ("set", tuple(encoded))
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    raise TypeError(
        "Counterfactual snapshot contains an unsupported value type: "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


@dataclass(frozen=True)
class Stage3CacheValueConfig:
    """Explicit Stage3 loss, optimizer and bounded-label settings."""

    trigger_ranking_coefficient: float
    candidate_ranking_coefficient: float
    delta_q_huber_coefficient: float
    value_head_learning_rate: float
    trigger_tie_tolerance: float
    candidate_tie_tolerance: float
    pairwise_temperature: float = 1.0
    huber_delta: float = 1.0
    max_macro_states_per_rollout: int = 4
    max_target_nodes_per_state: int = 2

    def validate(self) -> None:
        coefficients = (
            self.trigger_ranking_coefficient,
            self.candidate_ranking_coefficient,
            self.delta_q_huber_coefficient,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in coefficients):
            raise ValueError("Stage3 loss coefficients must be finite and non-negative.")
        if not any(float(value) > 0.0 for value in coefficients):
            raise ValueError("At least one Stage3 auxiliary loss coefficient must be positive.")
        if not math.isfinite(float(self.value_head_learning_rate)) or self.value_head_learning_rate <= 0.0:
            raise ValueError("Stage3 value-head learning rate must be finite and positive.")
        tolerances = (self.trigger_tie_tolerance, self.candidate_tie_tolerance)
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in tolerances):
            raise ValueError("Stage3 tie tolerances must be finite and non-negative.")
        if not math.isfinite(float(self.pairwise_temperature)) or self.pairwise_temperature <= 0.0:
            raise ValueError("Stage3 pairwise temperature must be finite and positive.")
        if not math.isfinite(float(self.huber_delta)) or self.huber_delta <= 0.0:
            raise ValueError("Stage3 Huber delta must be finite and positive.")
        if int(self.max_macro_states_per_rollout) <= 0:
            raise ValueError("Stage3 must select at least one Macro state per rollout.")
        if int(self.max_target_nodes_per_state) <= 0:
            raise ValueError("Stage3 must select at least one target node per state.")

    def contract(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class CounterfactualMacroTrace:
    """Self-contained bounded replay snapshot for one behavior Macro frame."""

    episode_key: str
    episode_seed: int | None
    behavior_policy_sha256: str
    rollout_index: int
    macro_index: int
    slot: int
    duration: int
    done: bool
    truncated: bool
    pre_cache: np.ndarray
    pre_uav: dict[str, np.ndarray]
    executed_cache: np.ndarray
    macro_actions: np.ndarray
    behavior_trigger_log_probs: np.ndarray
    behavior_content_log_probs: np.ndarray
    behavior_trigger_probabilities: np.ndarray
    behavior_content_probabilities: np.ndarray
    behavior_conditional_candidate_probabilities: np.ndarray
    candidate_configurations: tuple[tuple[np.ndarray, ...], ...]
    candidate_features: np.ndarray
    candidate_masks: np.ndarray
    target_node_indices: tuple[int, ...]
    aligned_steps: tuple[Any, ...]
    micro_common_uniforms: np.ndarray
    snapshot_sha256: str


@dataclass(frozen=True)
class CounterfactualTargetLabel:
    """KEEP baseline and legal UPDATE deltas for one target node."""

    node_index: int
    candidate_indices: tuple[int, ...]
    candidate_delta_q: np.ndarray
    q_keep: float
    best_delta_q: float | None
    branch_execution_count: int
    branch_cache_reuse_count: int
    branch_summaries: tuple[dict[str, Any], ...]


def conditional_policy_expected_delta_q(
    *,
    candidate_indices: tuple[int, ...],
    candidate_delta_q: Any,
    behavior_conditional_candidate_probabilities: Any,
    legal_candidate_mask: Any,
) -> float:
    """Compute E[ΔQ | UPDATE] under the frozen rollout Candidate policy."""

    indices = tuple(int(value) for value in candidate_indices)
    delta_values = np.asarray(candidate_delta_q, dtype=np.float64)
    probabilities = np.asarray(
        behavior_conditional_candidate_probabilities, dtype=np.float64
    )
    legal_mask = np.asarray(legal_candidate_mask, dtype=bool)
    if (
        probabilities.ndim != 1
        or legal_mask.shape != probabilities.shape
        or delta_values.shape != (len(indices),)
    ):
        raise ValueError("Frozen Candidate probabilities, legal mask and ΔQ labels are misaligned.")
    expected_indices = tuple(
        int(value) for value in np.flatnonzero(legal_mask) if int(value) != 0
    )
    if not expected_indices or indices != expected_indices:
        raise ValueError("Candidate ΔQ labels must cover exactly the legal UPDATE support.")
    if not np.isfinite(probabilities).all() or not np.isfinite(delta_values).all():
        raise FloatingPointError("Frozen Candidate policy probabilities or ΔQ labels are non-finite.")
    if np.any(probabilities < 0.0):
        raise ValueError("Frozen Candidate policy probabilities cannot be negative.")
    update_mask = legal_mask.copy()
    update_mask[0] = False
    if (
        probabilities[0] != 0.0
        or np.any(probabilities[~update_mask] != 0.0)
        or not np.isclose(
            float(probabilities[update_mask].sum(dtype=np.float64)),
            1.0,
            atol=1.0e-6,
            rtol=0.0,
        )
    ):
        raise ValueError(
            "Frozen Candidate probabilities must normalize only over legal UPDATE candidates."
        )
    result = float(
        np.dot(probabilities[np.asarray(indices, dtype=np.int64)], delta_values)
    )
    if not math.isfinite(result):
        raise FloatingPointError("Frozen Candidate expected ΔQ is non-finite.")
    return result


@dataclass(frozen=True)
class CounterfactualTraceResult:
    labels: tuple[CounterfactualTargetLabel, ...]
    branch_execution_count: int
    branch_cache_reuse_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class CounterfactualBranchValue:
    q_value: float
    discounted_reward: float
    bootstrap_value: float
    actual_task_utility_total: float
    cache_cost_total: float
    movement_cost_total: float
    slot_rewards: tuple[float, ...]
    slot_cache_costs: tuple[float, ...]
    slot_movement_costs: tuple[float, ...]


def _snapshot_payload(
    *,
    episode_key: str,
    episode_seed: int | None,
    behavior_policy_sha256: str,
    rollout_index: int,
    macro_index: int,
    slot: int,
    duration: int,
    done: bool,
    truncated: bool,
    pre_cache: np.ndarray,
    pre_uav: dict[str, np.ndarray],
    executed_cache: np.ndarray,
    macro_actions: np.ndarray,
    behavior_trigger_log_probs: np.ndarray,
    behavior_content_log_probs: np.ndarray,
    behavior_trigger_probabilities: np.ndarray,
    behavior_content_probabilities: np.ndarray,
    behavior_conditional_candidate_probabilities: np.ndarray,
    candidate_configurations: tuple[tuple[np.ndarray, ...], ...],
    candidate_features: np.ndarray,
    candidate_masks: np.ndarray,
    target_node_indices: tuple[int, ...],
    aligned_steps: tuple[Any, ...],
    micro_common_uniforms: np.ndarray,
) -> tuple[Any, ...]:
    return (
        str(episode_key),
        None if episode_seed is None else int(episode_seed),
        str(behavior_policy_sha256),
        int(rollout_index),
        int(macro_index),
        int(slot),
        int(duration),
        bool(done),
        bool(truncated),
        np.asarray(pre_cache, dtype=np.int8),
        {key: np.asarray(value, dtype=np.float64) for key, value in pre_uav.items()},
        np.asarray(executed_cache, dtype=np.int8),
        np.asarray(macro_actions, dtype=np.int64),
        np.asarray(behavior_trigger_log_probs, dtype=np.float32),
        np.asarray(behavior_content_log_probs, dtype=np.float32),
        np.asarray(behavior_trigger_probabilities, dtype=np.float32),
        np.asarray(behavior_content_probabilities, dtype=np.float32),
        np.asarray(behavior_conditional_candidate_probabilities, dtype=np.float32),
        candidate_configurations,
        np.asarray(candidate_features, dtype=np.float32),
        np.asarray(candidate_masks, dtype=bool),
        tuple(int(value) for value in target_node_indices),
        _snapshot_value(aligned_steps),
        np.asarray(micro_common_uniforms, dtype=np.float64),
    )


def _snapshot_digest(payload: tuple[Any, ...]) -> str:
    try:
        serialized = pickle.dumps(payload, protocol=5)
    except (pickle.PickleError, TypeError, AttributeError) as exc:
        raise ValueError(
            "Counterfactual snapshot contains an input that cannot be frozen exactly."
        ) from exc
    return hashlib.sha256(serialized).hexdigest()


def _environment_fingerprint(context: Any) -> str:
    """Fingerprint mutable environment configuration/state around isolated replays."""

    payload = (
        context.cfg,
        context.cache_cfg,
        context.service_catalog,
        context.cache_nodes,
        context.initial_cache,
        context.initial_uav,
        context.flight_boundary,
        context.action_spec,
    )
    return _snapshot_digest(payload)


def _capture_global_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


def _global_rng_state_matches(saved: dict[str, Any]) -> bool:
    current_numpy = np.random.get_state()
    saved_numpy = saved["numpy"]
    numpy_matches = (
        current_numpy[0] == saved_numpy[0]
        and np.array_equal(current_numpy[1], saved_numpy[1])
        and current_numpy[2:] == saved_numpy[2:]
    )
    current_cuda = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    cuda_matches = (
        (current_cuda is None and saved["torch_cuda"] is None)
        or (
            current_cuda is not None
            and saved["torch_cuda"] is not None
            and len(current_cuda) == len(saved["torch_cuda"])
            and all(
                torch.equal(current, previous)
                for current, previous in zip(
                    current_cuda, saved["torch_cuda"], strict=True
                )
            )
        )
    )
    return (
        random.getstate() == saved["python"]
        and numpy_matches
        and torch.equal(torch.get_rng_state(), saved["torch_cpu"])
        and cuda_matches
    )


def _restore_global_rng_state(saved: dict[str, Any]) -> None:
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch_cpu"].cpu())
    if torch.cuda.is_available() and saved["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(saved["torch_cuda"])


def make_counterfactual_trace(
    *,
    episode_key: str,
    episode_seed: int | None,
    behavior_policy_sha256: str,
    rollout_index: int,
    macro_index: int,
    slot: int,
    duration: int,
    done: bool,
    truncated: bool,
    pre_cache: np.ndarray,
    pre_uav: dict[str, np.ndarray],
    executed_cache: np.ndarray,
    macro_actions: np.ndarray,
    behavior_trigger_log_probs: np.ndarray,
    behavior_content_log_probs: np.ndarray,
    behavior_trigger_probabilities: np.ndarray,
    behavior_content_probabilities: np.ndarray,
    behavior_conditional_candidate_probabilities: np.ndarray,
    candidate_configurations: list[list[np.ndarray]],
    candidate_features: np.ndarray,
    candidate_masks: np.ndarray,
    target_node_indices: tuple[int, ...],
    aligned_steps: list[Any],
    micro_common_uniforms: list[Any],
) -> CounterfactualMacroTrace:
    """Deep-copy and fingerprint all inputs required to replay a Macro frame."""

    if int(duration) <= 0 or len(micro_common_uniforms) != int(duration):
        raise ValueError("Counterfactual Macro duration/common-uniform count mismatch.")
    if bool(done) and bool(truncated):
        raise ValueError("A Macro replay boundary cannot be both terminal and truncated.")
    if len(str(behavior_policy_sha256)) != 64:
        raise ValueError("Counterfactual trace needs the behavior policy SHA-256 fingerprint.")
    expected_step_count = int(duration) if done else int(duration) + 1
    if len(aligned_steps) != expected_step_count:
        raise ValueError(
            "Counterfactual replay needs every transition Slot and its bootstrap Slot."
        )
    frozen_steps = tuple(copy.deepcopy(step) for step in aligned_steps)
    frozen_micro_uniforms = np.asarray(
        micro_common_uniforms, dtype=np.float64
    ).copy()
    if frozen_micro_uniforms.shape != (int(duration), 2, 2):
        raise ValueError(
            "Counterfactual common uniforms must have shape (duration, 2 UAVs, 2 draws)."
        )
    if (
        not np.isfinite(frozen_micro_uniforms).all()
        or np.any(frozen_micro_uniforms < 0.0)
        or np.any(frozen_micro_uniforms >= 1.0)
    ):
        raise ValueError("Counterfactual common uniforms must be finite and in [0, 1).")
    frozen_candidates = tuple(
        tuple(np.asarray(item, dtype=np.int8).copy() for item in node_candidates)
        for node_candidates in candidate_configurations
    )
    frozen_pre_cache = np.asarray(pre_cache, dtype=np.int8).copy()
    frozen_pre_uav = {
        str(key): np.asarray(value, dtype=np.float64).copy()
        for key, value in pre_uav.items()
    }
    frozen_executed_cache = np.asarray(executed_cache, dtype=np.int8).copy()
    frozen_actions = np.asarray(macro_actions, dtype=np.int64).copy()
    frozen_trigger_log_probs = np.asarray(behavior_trigger_log_probs, dtype=np.float32).copy()
    frozen_content_log_probs = np.asarray(behavior_content_log_probs, dtype=np.float32).copy()
    frozen_trigger_probabilities = np.asarray(
        behavior_trigger_probabilities, dtype=np.float32
    ).copy()
    frozen_content_probabilities = np.asarray(
        behavior_content_probabilities, dtype=np.float32
    ).copy()
    probability_arrays = (
        frozen_trigger_log_probs,
        frozen_content_log_probs,
        frozen_trigger_probabilities,
        frozen_content_probabilities,
    )
    if any(value.shape != frozen_actions.shape for value in probability_arrays):
        raise ValueError("Saved Macro action probabilities do not align with selected actions.")
    if not all(np.isfinite(value).all() for value in probability_arrays):
        raise FloatingPointError("Saved Macro action probabilities contain NaN/Inf.")
    if (
        np.any(frozen_trigger_probabilities <= 0.0)
        or np.any(frozen_trigger_probabilities > 1.0)
        or np.any(frozen_content_probabilities <= 0.0)
        or np.any(frozen_content_probabilities > 1.0)
        or not np.allclose(
            np.exp(frozen_trigger_log_probs),
            frozen_trigger_probabilities,
            atol=1.0e-6,
            rtol=1.0e-5,
        )
        or not np.allclose(
            np.exp(frozen_content_log_probs),
            frozen_content_probabilities,
            atol=1.0e-6,
            rtol=1.0e-5,
        )
    ):
        raise ValueError("Saved Macro selected log-probability/probability pairs disagree.")
    keep_actions = frozen_actions == 0
    if np.any(keep_actions) and not np.allclose(
        frozen_content_probabilities[keep_actions], 1.0, atol=0.0, rtol=0.0
    ):
        raise ValueError("KEEP actions must not claim a sampled conditional content probability.")
    frozen_features = np.asarray(candidate_features, dtype=np.float32).copy()
    frozen_masks = np.asarray(candidate_masks, dtype=bool).copy()
    frozen_conditional_candidate_probabilities = np.asarray(
        behavior_conditional_candidate_probabilities, dtype=np.float32
    ).copy()
    frozen_nodes = tuple(int(value) for value in target_node_indices)
    node_count = int(frozen_pre_cache.shape[0]) if frozen_pre_cache.ndim == 2 else -1
    if (
        node_count <= 0
        or frozen_actions.shape != (node_count,)
        or len(frozen_candidates) != node_count
        or frozen_masks.ndim != 2
        or frozen_masks.shape[0] != node_count
        or frozen_features.shape[:2] != frozen_masks.shape
        or frozen_executed_cache.shape != frozen_pre_cache.shape
    ):
        raise ValueError("Counterfactual Macro snapshot arrays are not shape-aligned.")
    if (
        frozen_conditional_candidate_probabilities.shape != frozen_masks.shape
        or not np.isfinite(frozen_conditional_candidate_probabilities).all()
        or np.any(frozen_conditional_candidate_probabilities < 0.0)
    ):
        raise ValueError(
            "Frozen conditional Candidate probabilities must be finite and align with legal masks."
        )
    for node_index in range(node_count):
        legal_update_mask = frozen_masks[node_index].copy()
        legal_update_mask[0] = False
        probability_row = frozen_conditional_candidate_probabilities[node_index]
        if np.any(probability_row[~legal_update_mask] != 0.0):
            raise ValueError(
                "Frozen conditional Candidate probability assigns mass to KEEP or an illegal action."
            )
        expected_mass = 1.0 if bool(np.any(legal_update_mask)) else 0.0
        if not np.isclose(
            float(probability_row.sum(dtype=np.float64)),
            expected_mass,
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise ValueError(
                "Frozen conditional Candidate probability mass does not normalize over legal UPDATE actions."
            )
    expected_executed_cache = frozen_pre_cache.copy()
    for node_index, node_candidates in enumerate(frozen_candidates):
        if (
            not node_candidates
            or len(node_candidates) > frozen_masks.shape[1]
            or not np.all(frozen_masks[node_index, : len(node_candidates)])
            or np.any(frozen_masks[node_index, len(node_candidates) :])
        ):
            raise ValueError("Saved legal Cache masks do not map one-to-one to candidate matrices.")
        action_index = int(frozen_actions[node_index])
        if action_index < 0 or action_index >= len(node_candidates):
            raise ValueError("Behavior Macro action index has no saved Cache candidate matrix.")
        selected_candidate = node_candidates[action_index]
        if selected_candidate.shape != frozen_pre_cache.shape:
            raise ValueError("Saved Cache candidate matrix has the wrong state shape.")
        if not np.array_equal(selected_candidate[0:node_index], frozen_pre_cache[0:node_index]) or not np.array_equal(
            selected_candidate[node_index + 1 :], frozen_pre_cache[node_index + 1 :]
        ):
            raise ValueError("Per-node Cache action candidate changes unrelated node rows.")
        expected_executed_cache[node_index] = selected_candidate[node_index]
    if not np.array_equal(expected_executed_cache, frozen_executed_cache):
        raise ValueError("Selected Macro candidate indices do not reconstruct executed Cache state.")
    if not frozen_nodes or len(set(frozen_nodes)) != len(frozen_nodes):
        raise ValueError("Counterfactual target node indices must be non-empty and unique.")
    if any(node_index < 0 or node_index >= node_count for node_index in frozen_nodes):
        raise IndexError("Counterfactual target node index is outside the Cache matrix.")
    for offset, step in enumerate(frozen_steps):
        expected_slot = int(slot) + offset
        if int(step.slot) != expected_slot:
            raise ValueError("Counterfactual aligned Slot inputs are not contiguous at the Macro boundary.")
    payload = _snapshot_payload(
        episode_key=episode_key,
        episode_seed=episode_seed,
        behavior_policy_sha256=behavior_policy_sha256,
        rollout_index=rollout_index,
        macro_index=macro_index,
        slot=slot,
        duration=duration,
        done=done,
        truncated=truncated,
        pre_cache=frozen_pre_cache,
        pre_uav=frozen_pre_uav,
        executed_cache=frozen_executed_cache,
        macro_actions=frozen_actions,
        behavior_trigger_log_probs=frozen_trigger_log_probs,
        behavior_content_log_probs=frozen_content_log_probs,
        behavior_trigger_probabilities=frozen_trigger_probabilities,
        behavior_content_probabilities=frozen_content_probabilities,
        behavior_conditional_candidate_probabilities=(
            frozen_conditional_candidate_probabilities
        ),
        candidate_configurations=frozen_candidates,
        candidate_features=frozen_features,
        candidate_masks=frozen_masks,
        target_node_indices=frozen_nodes,
        aligned_steps=frozen_steps,
        micro_common_uniforms=frozen_micro_uniforms,
    )
    digest = _snapshot_digest(payload)
    return CounterfactualMacroTrace(
        episode_key=str(episode_key),
        episode_seed=None if episode_seed is None else int(episode_seed),
        behavior_policy_sha256=str(behavior_policy_sha256),
        rollout_index=int(rollout_index),
        macro_index=int(macro_index),
        slot=int(slot),
        duration=int(duration),
        done=bool(done),
        truncated=bool(truncated),
        pre_cache=frozen_pre_cache,
        pre_uav=frozen_pre_uav,
        executed_cache=frozen_executed_cache,
        macro_actions=frozen_actions,
        behavior_trigger_log_probs=frozen_trigger_log_probs,
        behavior_content_log_probs=frozen_content_log_probs,
        behavior_trigger_probabilities=frozen_trigger_probabilities,
        behavior_content_probabilities=frozen_content_probabilities,
        behavior_conditional_candidate_probabilities=(
            frozen_conditional_candidate_probabilities
        ),
        candidate_configurations=frozen_candidates,
        candidate_features=frozen_features,
        candidate_masks=frozen_masks,
        target_node_indices=frozen_nodes,
        aligned_steps=frozen_steps,
        micro_common_uniforms=frozen_micro_uniforms,
        snapshot_sha256=digest,
    )


def validate_counterfactual_trace(trace: CounterfactualMacroTrace) -> None:
    """Reject snapshots whose exogenous inputs or branch state were mutated."""

    uniforms = np.asarray(trace.micro_common_uniforms, dtype=np.float64)
    if uniforms.shape != (int(trace.duration), 2, 2):
        raise ValueError(
            "Counterfactual common uniforms must have shape (duration, 2 UAVs, 2 draws)."
        )
    if (
        not np.isfinite(uniforms).all()
        or np.any(uniforms < 0.0)
        or np.any(uniforms >= 1.0)
    ):
        raise ValueError("Counterfactual common uniforms must be finite and in [0, 1).")

    payload = _snapshot_payload(
        episode_key=trace.episode_key,
        episode_seed=trace.episode_seed,
        behavior_policy_sha256=trace.behavior_policy_sha256,
        rollout_index=trace.rollout_index,
        macro_index=trace.macro_index,
        slot=trace.slot,
        duration=trace.duration,
        done=trace.done,
        truncated=trace.truncated,
        pre_cache=trace.pre_cache,
        pre_uav=trace.pre_uav,
        executed_cache=trace.executed_cache,
        macro_actions=trace.macro_actions,
        behavior_trigger_log_probs=trace.behavior_trigger_log_probs,
        behavior_content_log_probs=trace.behavior_content_log_probs,
        behavior_trigger_probabilities=trace.behavior_trigger_probabilities,
        behavior_content_probabilities=trace.behavior_content_probabilities,
        behavior_conditional_candidate_probabilities=(
            trace.behavior_conditional_candidate_probabilities
        ),
        candidate_configurations=trace.candidate_configurations,
        candidate_features=trace.candidate_features,
        candidate_masks=trace.candidate_masks,
        target_node_indices=trace.target_node_indices,
        aligned_steps=trace.aligned_steps,
        micro_common_uniforms=trace.micro_common_uniforms,
    )
    if _snapshot_digest(payload) != trace.snapshot_sha256:
        raise ValueError("Counterfactual trace changed after its immutable snapshot was made.")


def cache_node_value_features(context: Any, cache: np.ndarray) -> np.ndarray:
    """Encode node role/resources and occupied storage as finite shared-head inputs."""

    roles = tuple(str(value).upper() for value in context.cache_nodes["node_roles"])
    capacities = np.asarray(context.cache_nodes["capacities_mb"], dtype=np.float64)
    eta = np.asarray(context.cache_nodes["eta_ca"], dtype=np.float64)
    xi = np.asarray(context.cache_nodes["xi_ca"], dtype=np.float64)
    cache_state = np.asarray(cache, dtype=np.float64)
    sizes_mb = np.asarray(context.service_catalog["service_sizes_mb"], dtype=np.float64)
    node_count = len(roles)
    if (
        capacities.shape != (node_count,)
        or eta.shape != (node_count,)
        or xi.shape != (node_count,)
        or cache_state.ndim != 2
        or cache_state.shape != (node_count, len(sizes_mb))
        or np.any(capacities <= 0.0)
        or np.any(sizes_mb < 0.0)
        or not all(np.isfinite(value).all() for value in (capacities, eta, xi, cache_state, sizes_mb))
    ):
        raise ValueError("Cache node value descriptors are not aligned or finite.")
    role_index = {"SV": 0, "RSU": 1, "UAV": 2}
    if any(role not in role_index for role in roles):
        raise ValueError("Cache node value descriptor has an unknown node role.")
    role_features = np.zeros((node_count, 3), dtype=np.float64)
    for index, role in enumerate(roles):
        role_features[index, role_index[role]] = 1.0
    eta_scale = max(float(np.max(np.abs(eta))), 1.0)
    xi_scale = max(float(np.max(np.abs(xi))), 1.0)
    capacity_scale = max(float(np.max(capacities)), 1.0)
    occupancy = (cache_state * sizes_mb[None, :]).sum(axis=1) / capacities
    features = np.column_stack(
        (
            role_features,
            capacities / capacity_scale,
            eta / eta_scale,
            xi / xi_scale,
            occupancy,
        )
    ).astype(np.float32)
    if features.shape != (node_count, CACHE_NODE_VALUE_FEATURE_DIM) or not np.isfinite(features).all():
        raise FloatingPointError("Cache node value descriptors are malformed or non-finite.")
    return features


if torch is not None:

    class CacheDeltaQValueHead(nn.Module):
        """Shared scorer for ΔQ(global state, target-node descriptor, candidate)."""

        def __init__(
            self,
            observation_dim: int,
            candidate_feature_dim: int,
            node_feature_dim: int = CACHE_NODE_VALUE_FEATURE_DIM,
            hidden_sizes: tuple[int, ...] = (128, 128),
        ) -> None:
            super().__init__()
            dimensions = (int(observation_dim) + int(candidate_feature_dim) + int(node_feature_dim),)
            layers: list[nn.Module] = []
            width = dimensions[0]
            for hidden_size in hidden_sizes:
                if int(hidden_size) <= 0:
                    raise ValueError("Cache ΔQ value-head hidden sizes must be positive.")
                layers.extend((nn.Linear(width, int(hidden_size)), nn.Tanh()))
                width = int(hidden_size)
            layers.append(nn.Linear(width, 1))
            self.network = nn.Sequential(*layers)
            self.observation_dim = int(observation_dim)
            self.candidate_feature_dim = int(candidate_feature_dim)
            self.node_feature_dim = int(node_feature_dim)

        def forward(self, observations: Any, node_features: Any, candidate_features: Any) -> Any:
            observation_values = torch.as_tensor(
                observations, dtype=torch.float32, device=next(self.parameters()).device
            )
            node_values = torch.as_tensor(
                node_features, dtype=torch.float32, device=observation_values.device
            )
            candidate_values = torch.as_tensor(
                candidate_features, dtype=torch.float32, device=observation_values.device
            )
            if observation_values.ndim == 1:
                observation_values = observation_values.unsqueeze(0)
            if node_values.ndim == 1:
                node_values = node_values.unsqueeze(0)
            if candidate_values.ndim == 2:
                candidate_values = candidate_values.unsqueeze(0)
            batch_size = int(observation_values.shape[0])
            if (
                observation_values.shape != (batch_size, self.observation_dim)
                or node_values.shape != (batch_size, self.node_feature_dim)
                or candidate_values.ndim != 3
                or candidate_values.shape[0] != batch_size
                or candidate_values.shape[-1] != self.candidate_feature_dim
            ):
                raise ValueError("Cache ΔQ head inputs do not match their [B,D]/[B,K,F] contract.")
            if not all(
                bool(torch.isfinite(values).all().item())
                for values in (observation_values, node_values, candidate_values)
            ):
                raise FloatingPointError("Cache ΔQ head inputs contain NaN/Inf.")
            candidate_count = int(candidate_values.shape[1])
            repeated_observations = observation_values[:, None, :].expand(-1, candidate_count, -1)
            repeated_nodes = node_values[:, None, :].expand(-1, candidate_count, -1)
            joined = torch.cat(
                (repeated_observations, repeated_nodes, candidate_values), dim=-1
            )
            output = self.network(joined).squeeze(-1)
            if not bool(torch.isfinite(output).all().item()):
                raise FloatingPointError("Cache ΔQ head produced NaN/Inf.")
            return output

else:  # pragma: no cover - import-only environments

    class CacheDeltaQValueHead:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Stage3 Cache ΔQ value learning requires PyTorch.")


def trigger_value_ranking_loss(
    update_log_odds: Any,
    trigger_delta_q: Any,
    valid_mask: Any,
    *,
    tie_tolerance: float,
) -> tuple[Any, int]:
    """Rank UPDATE-vs-KEEP log odds by the frozen Candidate-policy expected ΔQ."""

    if torch is None:
        raise ImportError("Stage3 ranking losses require PyTorch.")
    scores = torch.as_tensor(update_log_odds)
    values = torch.as_tensor(trigger_delta_q, dtype=scores.dtype, device=scores.device)
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=scores.device)
    if scores.shape != values.shape or scores.shape != mask.shape:
        raise ValueError("Trigger ranking tensors must have identical shapes.")
    active = mask & torch.isfinite(values) & (torch.abs(values) > float(tie_tolerance))
    count = int(active.sum().item())
    if count == 0:
        return scores.sum() * 0.0, 0
    directions = torch.sign(values[active])
    loss = F.softplus(-directions * scores[active]).mean()
    return loss, count


def candidate_value_pairwise_ranking_loss(
    candidate_scores: Any,
    delta_q_values: Any,
    legal_update_masks: Any,
    *,
    tie_tolerance: float,
    temperature: float,
) -> tuple[Any, int]:
    """Average legal UPDATE candidate-pair ordering loss per state/node."""

    if torch is None:
        raise ImportError("Stage3 ranking losses require PyTorch.")
    scores = torch.as_tensor(candidate_scores)
    values = torch.as_tensor(delta_q_values, dtype=scores.dtype, device=scores.device)
    masks = torch.as_tensor(legal_update_masks, dtype=torch.bool, device=scores.device)
    if scores.shape != values.shape or scores.shape != masks.shape or scores.ndim != 3:
        raise ValueError("Candidate ranking inputs must have aligned [B,N,K] shapes.")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("Candidate ranking temperature must be finite and positive.")
    node_losses: list[Any] = []
    compared_pairs = 0
    for batch_index in range(int(scores.shape[0])):
        for node_index in range(int(scores.shape[1])):
            indices = torch.nonzero(
                masks[batch_index, node_index], as_tuple=False
            ).flatten()
            if int(indices.numel()) < 2:
                continue
            pair_indices = torch.triu_indices(
                int(indices.numel()), int(indices.numel()), offset=1, device=scores.device
            )
            left = indices[pair_indices[0]]
            right = indices[pair_indices[1]]
            value_difference = (
                values[batch_index, node_index, left]
                - values[batch_index, node_index, right]
            )
            active = torch.isfinite(value_difference) & (
                torch.abs(value_difference) > float(tie_tolerance)
            )
            active_count = int(active.sum().item())
            if active_count == 0:
                continue
            score_difference = (
                scores[batch_index, node_index, left[active]]
                - scores[batch_index, node_index, right[active]]
            ) / float(temperature)
            directions = torch.sign(value_difference[active])
            node_losses.append(F.softplus(-directions * score_difference).mean())
            compared_pairs += active_count
    if not node_losses:
        return scores.sum() * 0.0, 0
    return torch.stack(node_losses).mean(), compared_pairs


def delta_q_huber_loss(
    predictions: Any,
    targets: Any,
    legal_update_masks: Any,
    *,
    huber_delta: float,
) -> tuple[Any, int]:
    """Fit only legal UPDATE ΔQ values; KEEP is the zero-valued reference."""

    if torch is None:
        raise ImportError("Stage3 ΔQ regression requires PyTorch.")
    predicted = torch.as_tensor(predictions)
    target = torch.as_tensor(targets, dtype=predicted.dtype, device=predicted.device)
    mask = torch.as_tensor(legal_update_masks, dtype=torch.bool, device=predicted.device)
    if predicted.shape != target.shape or predicted.shape != mask.shape:
        raise ValueError("ΔQ regression tensors must have identical shapes.")
    if not math.isfinite(float(huber_delta)) or float(huber_delta) <= 0.0:
        raise ValueError("ΔQ Huber delta must be finite and positive.")
    count = int(mask.sum().item())
    if count == 0:
        return predicted.sum() * 0.0, 0
    losses = F.smooth_l1_loss(
        predicted[mask], target[mask], reduction="mean", beta=float(huber_delta)
    )
    return losses, count


def _branch_cache_for_candidate(
    trace: CounterfactualMacroTrace,
    node_index: int,
    candidate_index: int,
) -> np.ndarray:
    if node_index < 0 or node_index >= len(trace.candidate_configurations):
        raise IndexError("Counterfactual target Cache node index is out of range.")
    if candidate_index < 0 or candidate_index >= len(trace.candidate_configurations[node_index]):
        raise IndexError("Counterfactual candidate index is out of range.")
    if candidate_index >= trace.candidate_masks.shape[1] or not bool(
        trace.candidate_masks[node_index, candidate_index]
    ):
        raise ValueError("Counterfactual candidate is not in the saved legal action mask.")
    candidate = trace.candidate_configurations[node_index][candidate_index]
    if candidate.shape != trace.pre_cache.shape:
        raise ValueError("Saved candidate cache matrix does not match the Macro snapshot.")
    other_rows = np.arange(trace.pre_cache.shape[0]) != int(node_index)
    if not np.array_equal(candidate[other_rows], trace.pre_cache[other_rows]):
        raise ValueError("Per-node Cache candidate unexpectedly changes another Macro node.")
    if candidate_index == 0 and not np.array_equal(candidate, trace.pre_cache):
        raise ValueError("Cache candidate zero must be the exact KEEP configuration.")
    branch_cache = trace.executed_cache.copy()
    branch_cache[node_index] = candidate[node_index]
    return branch_cache


def _evaluate_branch_value(
    trainer: Any,
    trace: CounterfactualMacroTrace,
    branch_cache: np.ndarray,
    gamma: float,
) -> CounterfactualBranchValue:
    """Replay one Cache branch with common Micro uniforms and exact Slot execution."""

    context = trainer.context
    cache = trace.pre_cache.copy()
    uav = {key: value.copy() for key, value in trace.pre_uav.items()}
    cache_configuration.validate_cache_shape_and_capacity(
        branch_cache,
        context.service_catalog,
        context.cache_nodes,
    )
    discounted_reward = 0.0
    actual_task_utility_total = 0.0
    cache_cost_total = 0.0
    movement_cost_total = 0.0
    slot_rewards: list[float] = []
    slot_cache_costs: list[float] = []
    slot_movement_costs: list[float] = []
    for offset in range(trace.duration):
        step = trace.aligned_steps[offset]
        previous_uav = {key: value.copy() for key, value in uav.items()}
        current_trial = {key: value.copy() for key, value in previous_uav.items()}
        observation = _observation(branch_cache, current_trial, step, context)
        uav0_condition = np.zeros(2, dtype=np.float32)
        triggers = {str(uav_id): 1 for uav_id in context.action_spec.uav_ids}

        # One fixed pair of uniforms is assigned to each logical UAV decision.
        # Branch-specific observations, legal sets and distributions are still
        # recomputed before applying inverse-CDF sampling.
        for uav_index, uav_id_raw in enumerate(context.action_spec.uav_ids):
            uav_id = str(uav_id_raw)
            candidates, features = _uav_candidates(
                uav_index,
                current_trial,
                previous_uav,
                branch_cache,
                step,
                context,
            )
            mask = np.ones(len(candidates), dtype=bool)
            condition = (
                np.zeros(2, dtype=np.float32)
                if uav_index == 0
                else uav0_condition.copy()
            )
            action, _, _, _ = trainer._sample_micro_head(
                observation,
                features,
                mask,
                uav_index=uav_index,
                uav0_position_condition=condition,
                deterministic=False,
                random_uniforms=trace.micro_common_uniforms[offset, uav_index],
            )
            if action < 0 or action >= len(candidates) or not bool(mask[action]):
                raise ValueError("Frozen Micro policy selected an illegal counterfactual UAV action.")
            uav_configuration.apply_uav_candidate(
                current_trial,
                previous_uav,
                uav_id,
                candidates[action].target_position,
                triggers,
                context.cfg,
                context.flight_boundary,
            )
            if uav_index == 0:
                uav0_condition = _positions_condition(current_trial[uav_id], context)

        uav_configuration.validate_uav_configuration_transition(
            previous_uav,
            current_trial,
            {
                uav_id: int(
                    np.linalg.norm(current_trial[uav_id] - previous_uav[uav_id])
                    > float(
                        uav_configuration.uav_search_config_from_project(
                            context.cfg
                        ).position_tolerance_m
                    )
                )
                for uav_id in context.action_spec.uav_ids
            },
            context.cfg,
            context.flight_boundary,
        )
        performance_profile: dict[str, Any] = {}
        result = slot_environment.execute_configured_slot(
            cache,
            previous_uav,
            branch_cache,
            current_trial,
            step.scenario_slice,
            step.actual_tasks,
            step.actual_vehicle_positions,
            context.service_catalog,
            context.cfg,
            context.cache_nodes,
            context.cache_cfg,
            context.flight_boundary,
            slot_id=int(step.slot),
            time_s=float(step.time_s),
            performance_profile=performance_profile,
        )
        if not result.step_feasible or result.system_utility is None:
            raise RuntimeError(
                "Formal Matching+KKT rejected a counterfactual Cache branch: "
                f"slot={step.slot}, reason={result.infeasible_reason}"
            )
        _check_request_contract(cache, previous_uav, result, context.action_spec)
        reward = _check_reward(result)
        task_utility = float(result.actual_task_utility)
        cache_cost = float(result.cache_cost)
        movement_cost = float(result.movement_cost)
        if not all(math.isfinite(value) for value in (task_utility, cache_cost, movement_cost)):
            raise FloatingPointError("Counterfactual environment returned a non-finite utility/cost.")
        next_cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
        if not np.array_equal(next_cache, branch_cache):
            raise ValueError("Formal Slot execution changed the counterfactual Cache target.")
        discounted_reward += (float(gamma) ** offset) * reward
        actual_task_utility_total += task_utility
        cache_cost_total += cache_cost
        movement_cost_total += movement_cost
        slot_rewards.append(float(reward))
        slot_cache_costs.append(cache_cost)
        slot_movement_costs.append(movement_cost)
        cache = next_cache
        uav = {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in result.next_uav_positions.items()
        }

    if trace.done:
        bootstrap_value = 0.0
    else:
        bootstrap_step = trace.aligned_steps[trace.duration]
        bootstrap_observation = _observation(cache, uav, bootstrap_step, context)
        with torch.no_grad():
            bootstrap_value = float(
                trainer.policy.macro_value(bootstrap_observation).detach().cpu()
            )
        if not math.isfinite(bootstrap_value):
            raise FloatingPointError("Frozen Macro Critic returned a non-finite bootstrap value.")
    value = discounted_reward + (float(gamma) ** trace.duration) * bootstrap_value
    if not math.isfinite(value):
        raise FloatingPointError("Counterfactual SMDP value is non-finite.")
    return CounterfactualBranchValue(
        q_value=float(value),
        discounted_reward=float(discounted_reward),
        bootstrap_value=float(bootstrap_value),
        actual_task_utility_total=float(actual_task_utility_total),
        cache_cost_total=float(cache_cost_total),
        movement_cost_total=float(movement_cost_total),
        slot_rewards=tuple(slot_rewards),
        slot_cache_costs=tuple(slot_cache_costs),
        slot_movement_costs=tuple(slot_movement_costs),
    )


def evaluate_counterfactual_trace(
    trainer: Any,
    trace: CounterfactualMacroTrace,
    *,
    gamma: float,
) -> CounterfactualTraceResult:
    """Evaluate KEEP and every legal UPDATE for selected target nodes."""

    validate_counterfactual_trace(trace)
    if not math.isfinite(float(gamma)) or not (0.0 <= float(gamma) <= 1.0):
        raise ValueError("Counterfactual SMDP discount must be finite and in [0, 1].")
    if trace.candidate_features.shape[:2] != trace.candidate_masks.shape:
        raise ValueError("Counterfactual candidate feature/mask shapes are misaligned.")
    if trace.candidate_masks.shape[0] != len(trace.candidate_configurations):
        raise ValueError("Counterfactual candidate node order differs from the action mask.")

    started = time.perf_counter()
    trainer_rng_state = trainer.generator.get_state().clone()
    counterfactual_rng = getattr(trainer, "counterfactual_micro_rng", None)
    counterfactual_rng_state = (
        None if counterfactual_rng is None else counterfactual_rng.getstate()
    )
    global_rng_state = _capture_global_rng_state()
    policy_fingerprint = policy_state_fingerprint(trainer.policy)
    if policy_fingerprint != trace.behavior_policy_sha256:
        raise ValueError(
            "Counterfactual labels require the exact frozen behavior actor/Critic snapshot."
        )
    environment_fingerprint = _environment_fingerprint(trainer.context)
    branch_execution_count = 0
    branch_cache_reuse_count = 0
    labels: list[CounterfactualTargetLabel] = []
    replay_error: Exception | None = None
    try:
        for node_index in trace.target_node_indices:
            if node_index < 0 or node_index >= trace.candidate_masks.shape[0]:
                raise IndexError("Counterfactual target node index is outside the action mask.")
            legal_indices = tuple(
                int(value) for value in np.flatnonzero(trace.candidate_masks[node_index])
            )
            if not legal_indices or legal_indices[0] != 0:
                raise ValueError("Counterfactual action support must begin with legal KEEP.")
            q_by_configuration: dict[bytes, CounterfactualBranchValue] = {}
            q_by_candidate: dict[int, CounterfactualBranchValue] = {}
            branch_summaries: list[dict[str, Any]] = []
            node_branch_execution_count = 0
            node_branch_reuse_count = 0
            for candidate_index in legal_indices:
                branch_cache = _branch_cache_for_candidate(
                    trace, node_index, candidate_index
                )
                key = branch_cache.tobytes()
                if key in q_by_configuration:
                    q_by_candidate[candidate_index] = q_by_configuration[key]
                    branch_cache_reuse_count += 1
                    node_branch_reuse_count += 1
                    continue
                value = _evaluate_branch_value(trainer, trace, branch_cache, gamma)
                validate_counterfactual_trace(trace)
                q_by_configuration[key] = value
                q_by_candidate[candidate_index] = value
                branch_execution_count += 1
                node_branch_execution_count += 1
            q_keep = q_by_candidate[0].q_value
            update_indices = tuple(index for index in legal_indices if index > 0)
            deltas = np.asarray(
                [q_by_candidate[index].q_value - q_keep for index in update_indices],
                dtype=np.float32,
            )
            if not np.isfinite(deltas).all():
                raise FloatingPointError("Counterfactual ΔQ labels contain NaN/Inf.")
            for candidate_index in legal_indices:
                branch_value = q_by_candidate[candidate_index]
                branch_summaries.append(
                    {
                        "candidate_index": int(candidate_index),
                        "q_value": float(branch_value.q_value),
                        "discounted_reward": float(branch_value.discounted_reward),
                        "bootstrap_value": float(branch_value.bootstrap_value),
                        "actual_task_utility_total": float(
                            branch_value.actual_task_utility_total
                        ),
                        "cache_cost_total": float(branch_value.cache_cost_total),
                        "movement_cost_total": float(branch_value.movement_cost_total),
                        "slot_rewards": list(branch_value.slot_rewards),
                        "slot_cache_costs": list(branch_value.slot_cache_costs),
                        "slot_movement_costs": list(branch_value.slot_movement_costs),
                    }
                )
            labels.append(
                CounterfactualTargetLabel(
                    node_index=int(node_index),
                    candidate_indices=update_indices,
                    candidate_delta_q=deltas,
                    q_keep=float(q_keep),
                    best_delta_q=(None if not len(deltas) else float(np.max(deltas))),
                    branch_execution_count=node_branch_execution_count,
                    branch_cache_reuse_count=node_branch_reuse_count,
                    branch_summaries=tuple(branch_summaries),
                )
            )
    except Exception as exc:
        replay_error = exc
    finally:
        trainer_rng_changed = not torch.equal(
            trainer.generator.get_state(), trainer_rng_state
        )
        counterfactual_rng_changed = (
            counterfactual_rng is not None
            and counterfactual_rng.getstate() != counterfactual_rng_state
        )
        global_rng_changed = not _global_rng_state_matches(global_rng_state)
        trainer.generator.set_state(trainer_rng_state.cpu())
        if counterfactual_rng_changed and counterfactual_rng is not None:
            counterfactual_rng.setstate(counterfactual_rng_state)
        if global_rng_changed:
            _restore_global_rng_state(global_rng_state)

    if trainer_rng_changed or counterfactual_rng_changed or global_rng_changed:
        if replay_error is not None:
            cause: Exception | None = replay_error
        else:
            cause = None
        raise RuntimeError(
            "Counterfactual replay consumed a policy/global RNG stream; its labels are rejected. "
            f"trainer_generator={trainer_rng_changed}, "
            f"counterfactual_micro={counterfactual_rng_changed}, "
            f"global_rng={global_rng_changed}."
        ) from cause
    if replay_error is not None:
        try:
            validate_counterfactual_trace(trace)
        except Exception as trace_error:
            raise RuntimeError(
                "Counterfactual replay failed and mutated its frozen exogenous/state snapshot."
            ) from trace_error
        if policy_state_fingerprint(trainer.policy) != policy_fingerprint:
            raise RuntimeError(
                "Counterfactual replay failed and changed frozen MAPPO actor/Critic parameters."
            ) from replay_error
        if _environment_fingerprint(trainer.context) != environment_fingerprint:
            raise RuntimeError(
                "Counterfactual replay failed and mutated the formal environment context."
            ) from replay_error
        raise replay_error
    validate_counterfactual_trace(trace)
    if policy_state_fingerprint(trainer.policy) != policy_fingerprint:
        raise RuntimeError("Counterfactual replay changed frozen MAPPO actor/Critic parameters.")
    if _environment_fingerprint(trainer.context) != environment_fingerprint:
        raise RuntimeError("Counterfactual replay mutated the formal environment context.")
    elapsed = time.perf_counter() - started
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise FloatingPointError("Counterfactual replay runtime is invalid.")
    return CounterfactualTraceResult(
        labels=tuple(labels),
        branch_execution_count=branch_execution_count,
        branch_cache_reuse_count=branch_cache_reuse_count,
        elapsed_seconds=float(elapsed),
    )


def policy_state_fingerprint(policy: Any) -> str:
    """Hash exact parameter/buffer tensors to verify label-generation immutability."""

    if torch is None:
        raise ImportError("Counterfactual policy fingerprinting requires PyTorch.")
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
