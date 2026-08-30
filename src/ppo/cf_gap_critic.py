#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Critic-only pilot for state/time-aware paired counterfactual gaps.

This module is deliberately separate from :mod:`ppo_training`'s Actor/PPO
update path.  It collects one frozen-Actor baseline, reuses the existing
Direct-CF continuation executor for a bounded set of new labels, trains an
independent regression critic, and evaluates it on the 24 labels from the V3
state/time diagnosis when those labels can be reconstructed exactly.

The module does not import or call any Actor-loss/update function.  The Actor
is created only to reproduce the frozen baseline and to execute the already
validated paired continuation branches.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from common import project_config
from ppo import ppo_rollout, ppo_training

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - project execution requires torch.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


TRIGGER_COUNT = 8
TIME_BINS = ("early", "middle", "late")
POLICY_VERSION = 0
DATASET_SCHEMA_VERSION = "paired_cf_gap_critic_dataset_v1"
CHECKPOINT_SCHEMA_VERSION = "paired_cf_gap_critic_checkpoint_v1"
VALIDATION_SCHEMA_VERSION = "paired_cf_gap_critic_validation_v1"
INPUT_SCHEMA_VERSION = "paired_cf_gap_input_v1"

# These are experiment seeds, not Actor/PPO sampling RNGs.  The Actor uses
# the configured formal seed; gap_data_rng is used only after the baseline is
# complete, and split_rng is used only for the fixed train/internal-val split.
DEFAULT_GAP_DATA_SEED = 2026082401
DEFAULT_SPLIT_SEED = 2026082402
DEFAULT_CRITIC_INIT_SEED = 2026082403
DEFAULT_CRITIC_LEARNING_RATE = 1.0e-3
DEFAULT_CRITIC_MAX_EPOCHS = 1000
DEFAULT_CRITIC_PATIENCE = 100
DEFAULT_CRITIC_MIN_DELTA = 1.0e-10
INTERNAL_VALIDATION_COUNT = 12
TRAINING_PER_STRATUM = 2

EXPECTED_BASELINE = {
    "steps": 276,
    "G": 780.4936401668566,
    "Q": 1089.5331440803836,
    "cache": 245.07275647419945,
    "UAV": 63.966747439327555,
}


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("Paired-CF Gap Critic pilot requires PyTorch.")


@dataclass(frozen=True)
class GapCriticInputSpec:
    """Fixed feature layout for the independent Gap Critic."""

    observation_dim: int
    prefix_dim: int = TRIGGER_COUNT
    prefix_mask_dim: int = TRIGGER_COUNT
    trigger_one_hot_dim: int = TRIGGER_COUNT
    time_dim: int = 1
    schema_version: str = INPUT_SCHEMA_VERSION

    @property
    def input_dim(self) -> int:
        return int(
            self.observation_dim
            + self.prefix_dim
            + self.prefix_mask_dim
            + self.trigger_one_hot_dim
            + self.time_dim
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layout": [
                "observation",
                "prefix_values[8]",
                "prefix_mask[8]",
                "trigger_one_hot[8]",
                "normalized_rollout_time",
            ],
            "observation_dim": int(self.observation_dim),
            "prefix_dim": int(self.prefix_dim),
            "prefix_mask_dim": int(self.prefix_mask_dim),
            "trigger_one_hot_dim": int(self.trigger_one_hot_dim),
            "time_dim": int(self.time_dim),
            "input_dim": int(self.input_dim),
            "normalization": "train-only feature mean/std; no held-out fitting",
        }


class GapCritic(nn.Module if nn is not None else object):
    """Independent MLP ``input -> 128 -> 64 -> 1`` for ``Delta_G``."""

    def __init__(self, input_dim: int, hidden_sizes: tuple[int, int] = (128, 64)) -> None:
        _require_torch()
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("GapCritic input_dim must be positive.")
        if tuple(int(size) for size in hidden_sizes) != (128, 64):
            raise ValueError("This pilot uses the fixed 128 -> 64 architecture only.")
        self.input_dim = int(input_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: Any) -> Any:
        _require_torch()
        features_t = torch.as_tensor(features, dtype=torch.float32)
        single = features_t.ndim == 1
        if single:
            features_t = features_t.unsqueeze(0)
        if features_t.ndim != 2 or int(features_t.shape[1]) != self.input_dim:
            raise ValueError(
                f"GapCritic features shape={tuple(features_t.shape)}, "
                f"expected [B,{self.input_dim}]."
            )
        if not bool(torch.isfinite(features_t).all().item()):
            raise ValueError("GapCritic features contain NaN/Inf.")
        output = self.network(features_t).squeeze(-1)
        return output.squeeze(0) if single else output


def build_gap_critic_preserving_rng(input_dim: int, seed: int) -> GapCritic:
    """Create the independent critic without consuming Actor RNG state."""

    _require_torch()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        critic = GapCritic(input_dim)
    return critic


def _finite_vector(value: Any, length: int, name: str, dtype: Any = np.float32) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape != (int(length),):
        raise ValueError(f"{name} must have shape [{length}], got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN/Inf.")
    return array


def build_gap_critic_features(
    observation: Any,
    prefix_values: Any,
    prefix_mask: Any,
    trigger_index: int,
    normalized_rollout_time: float,
    spec: GapCriticInputSpec,
) -> np.ndarray:
    """Build the critic input without exposing current/future actions.

    ``prefix_values`` is meaningful only where ``prefix_mask`` is one.  The
    current trigger is always masked out, and no suffix of the baseline action
    vector is accepted by this function.
    """

    observation_a = _finite_vector(observation, spec.observation_dim, "observation")
    prefix_a = _finite_vector(prefix_values, spec.prefix_dim, "prefix_values")
    prefix_mask_a = _finite_vector(prefix_mask, spec.prefix_mask_dim, "prefix_mask")
    if not np.all(np.isin(prefix_a, [0.0, 1.0])):
        raise ValueError("prefix_values must be binary.")
    if not np.all(np.isin(prefix_mask_a, [0.0, 1.0])):
        raise ValueError("prefix_mask must be binary.")
    index = int(trigger_index)
    if not 0 <= index < spec.trigger_one_hot_dim:
        raise ValueError("trigger_index is outside the one-hot range.")
    if np.any(prefix_mask_a[index:] != 0.0):
        raise ValueError("prefix_mask cannot expose the current trigger or future positions.")
    if np.any(prefix_mask_a[:index] != 1.0):
        raise ValueError("prefix_mask must mark every earlier autoregressive position.")
    if np.any(prefix_a[prefix_mask_a == 0.0] != 0.0):
        raise ValueError("prefix_values after the prefix must be zero-filled.")
    tau = float(normalized_rollout_time)
    if not math.isfinite(tau) or tau < 0.0 or tau > 1.0:
        raise ValueError("normalized_rollout_time must be finite and lie in [0,1].")
    trigger_one_hot = np.zeros(spec.trigger_one_hot_dim, dtype=np.float32)
    trigger_one_hot[index] = 1.0
    feature = np.concatenate(
        [
            observation_a.astype(np.float32),
            prefix_a.astype(np.float32),
            prefix_mask_a.astype(np.float32),
            trigger_one_hot,
            np.asarray([tau], dtype=np.float32),
        ]
    )
    if feature.shape != (spec.input_dim,) or not np.all(np.isfinite(feature)):
        raise ValueError("GapCritic feature construction produced an invalid vector.")
    return feature.astype(np.float32)


def _time_bin(decision_index: int, steps_per_episode: int) -> str:
    index = int(decision_index)
    steps = int(steps_per_episode)
    if steps < 3 or not 0 <= index < steps:
        raise ValueError("decision_index/steps_per_episode are invalid.")
    third = steps // 3
    if index < third:
        return "early"
    if index < 2 * third:
        return "middle"
    return "late"


def _prefix_from_snapshot(snapshot: ppo_rollout.DirectCFDecisionSnapshot) -> tuple[list[int], list[int]]:
    index = int(snapshot.trigger_index)
    prefix = _finite_vector(snapshot.action_prefix, index, "snapshot.action_prefix", dtype=np.int8)
    if not np.all(np.isin(prefix, [0, 1])):
        raise ValueError("snapshot.action_prefix must be binary.")
    values = np.zeros(TRIGGER_COUNT, dtype=np.int8)
    mask = np.zeros(TRIGGER_COUNT, dtype=np.int8)
    values[:index] = prefix
    mask[:index] = 1
    return values.astype(int).tolist(), mask.astype(int).tolist()


def _record_from_snapshot(
    snapshot: ppo_rollout.DirectCFDecisionSnapshot,
    *,
    source: str,
    target_fields: dict[str, Any],
    spec: GapCriticInputSpec,
    steps_per_episode: int,
    policy_version: int,
) -> dict[str, Any]:
    decision_index = int(snapshot.step_index)
    prefix_values, prefix_mask = _prefix_from_snapshot(snapshot)
    target = float(target_fields["Delta_G"])
    delta_q = float(target_fields["Delta_Q"])
    delta_cache = float(target_fields["Delta_cache"])
    delta_uav = float(target_fields["Delta_UAV"])
    identity_error = target - delta_q + delta_cache + delta_uav
    if not all(math.isfinite(value) for value in (target, delta_q, delta_cache, delta_uav, identity_error)):
        raise ValueError("Paired-CF label contains NaN/Inf.")
    tau = float(decision_index / (int(steps_per_episode) - 1))
    feature = build_gap_critic_features(
        snapshot.observation,
        prefix_values,
        prefix_mask,
        int(snapshot.trigger_index),
        tau,
        spec,
    )
    return {
        "source": str(source),
        "decision_index": decision_index,
        "origin_index": int(snapshot.origin_index),
        "slot": int(snapshot.slot),
        "time_s": float(target_fields.get("time_s", snapshot.slot)),
        "time_bin": _time_bin(decision_index, steps_per_episode),
        "trigger_index": int(snapshot.trigger_index),
        "trigger_name": str(snapshot.trigger_label),
        "observation": np.asarray(snapshot.observation, dtype=np.float32).copy(),
        "prefix_values": prefix_values,
        "prefix_mask": prefix_mask,
        "actual_action": int(snapshot.action_vector[snapshot.trigger_index]),
        "p_old": float(snapshot.old_p1),
        "Delta_G": target,
        "Delta_Q": delta_q,
        "Delta_cache": delta_cache,
        "Delta_UAV": delta_uav,
        "delta_identity_error": float(identity_error),
        "stored_delta_identity_error": (
            None
            if target_fields.get("delta_identity_error") is None
            else float(target_fields["delta_identity_error"])
        ),
        "branch_environment_steps": int(target_fields.get("branch_environment_steps", 0)),
        "crn_target_step_rng_aligned": bool(target_fields.get("crn_target_step_rng_aligned", True)),
        "policy_version": int(policy_version),
        "normalized_rollout_time": tau,
        "feature": feature,
    }


def _record_from_branch_sample(
    sample: ppo_rollout.DirectCFCreditSample,
    *,
    spec: GapCriticInputSpec,
    steps_per_episode: int,
    policy_version: int,
    time_s: float | None = None,
) -> dict[str, Any]:
    return _record_from_snapshot(
        ppo_rollout.DirectCFDecisionSnapshot(
            step_index=int(sample.origin_index),
            slot=int(sample.slot),
            origin_index=int(sample.origin_index),
            trigger_index=int(sample.trigger_index),
            trigger_label=str(sample.trigger_label),
            decision_id=str(sample.decision_id),
            previous_cache_state=np.empty((0,), dtype=np.int8),
            previous_uav_positions={},
            observation=np.asarray(sample.observation, dtype=np.float32),
            action_vector=np.asarray(sample.action_vector, dtype=np.int8),
            action_prefix=np.asarray(sample.action_prefix, dtype=np.int8),
            old_p1=float(sample.old_p1),
            old_chosen_log_prob=float(sample.old_chosen_log_prob),
            action_mask=np.asarray(sample.action_mask, dtype=bool),
            baseline_rng_before={},
            baseline_rng_after_slot={},
        ),
        source="train",
        target_fields={
            "time_s": float(sample.slot) if time_s is None else float(time_s),
            "Delta_G": float(sample.delta_g),
            "Delta_Q": float(sample.delta_q),
            "Delta_cache": float(sample.delta_cache),
            "Delta_UAV": float(sample.delta_uav),
            "branch_environment_steps": int(sample.branch_environment_steps),
            "crn_target_step_rng_aligned": bool(sample.crn_target_step_rng_aligned),
        },
        spec=spec,
        steps_per_episode=steps_per_episode,
        policy_version=policy_version,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(child) for child in value.tolist()]
    if torch is not None and isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Cannot serialize NaN/Inf to JSON.")
    return value


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(_jsonable(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _assert_baseline(stats: dict[str, float], batch: ppo_rollout.RolloutBatch, free_count: int) -> dict[str, Any]:
    errors = {
        "G": float(stats["raw_reward_sum"] - EXPECTED_BASELINE["G"]),
        "Q": float(stats["raw_actual_q_sum"] - EXPECTED_BASELINE["Q"]),
        "cache": float(stats["raw_cache_cost_sum"] - EXPECTED_BASELINE["cache"]),
        "UAV": float(stats["raw_movement_cost_sum"] - EXPECTED_BASELINE["UAV"]),
        "steps": float(len(batch.transitions) - EXPECTED_BASELINE["steps"]),
    }
    if any(abs(value) > 1.0e-12 for value in errors.values()):
        raise RuntimeError(f"fresh baseline mismatch: {errors}")
    if not bool(batch.metadata.get("true_terminal", False)) or bool(batch.truncated):
        raise RuntimeError("fresh baseline is not a true terminal rollout.")
    if float(batch.final_bootstrap_value) != 0.0:
        raise RuntimeError("fresh baseline bootstrap is not exactly zero.")
    if int(free_count) != EXPECTED_BASELINE["steps"] * TRIGGER_COUNT:
        raise RuntimeError(f"unexpected FREE decision count: {free_count}")
    return {
        "expected": dict(EXPECTED_BASELINE),
        "actual": {
            "steps": int(len(batch.transitions)),
            "G": float(stats["raw_reward_sum"]),
            "Q": float(stats["raw_actual_q_sum"]),
            "cache": float(stats["raw_cache_cost_sum"]),
            "UAV": float(stats["raw_movement_cost_sum"]),
            "true_terminal": True,
            "truncated": False,
            "bootstrap": 0.0,
        },
        "errors_actual_minus_expected": errors,
        "passed": True,
        "reservoir_total_free_trigger_decisions": int(free_count),
    }


def _snapshot_map(reservoir: ppo_rollout.DirectCFReservoir) -> dict[tuple[int, int], ppo_rollout.DirectCFDecisionSnapshot]:
    result: dict[tuple[int, int], ppo_rollout.DirectCFDecisionSnapshot] = {}
    for snapshot in reservoir.items:
        key = (int(snapshot.trigger_index), int(snapshot.step_index))
        if key in result:
            raise RuntimeError(f"duplicate FREE snapshot key: {key}")
        result[key] = snapshot
    return result


def _load_diagnosis(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    samples = payload.get("paired_cf", {}).get("samples")
    if not isinstance(samples, list) or len(samples) != 24:
        raise RuntimeError("diagnosis JSON must contain exactly 24 paired_cf.samples.")
    return payload


def _reconstruct_heldout(
    diagnosis: dict[str, Any],
    snapshots: dict[tuple[int, int], ppo_rollout.DirectCFDecisionSnapshot],
    spec: GapCriticInputSpec,
    steps_per_episode: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing: list[tuple[int, int]] = []
    mismatches: list[dict[str, Any]] = []
    stored_identity_errors: list[float] = []
    for label in diagnosis["paired_cf"]["samples"]:
        trigger_index = int(label["trigger_index"])
        decision_index = int(label["decision_index"])
        key = (trigger_index, decision_index)
        snapshot = snapshots.get(key)
        if snapshot is None:
            missing.append(key)
            continue
        checks = {
            "trigger_index": trigger_index == int(snapshot.trigger_index),
            "trigger_name": str(label["trigger"]) == str(snapshot.trigger_label),
            "decision_index": decision_index == int(snapshot.step_index),
            "origin_index": int(label["origin_index"]) == int(snapshot.origin_index),
            "slot": int(label["slot"]) == int(snapshot.slot),
            "actual_action": int(label["actual_action"]) == int(snapshot.action_vector[trigger_index]),
            "action": int(label["action"]) == int(snapshot.action_vector[trigger_index]),
            "p_old": math.isclose(
                float(label["p_old"]),
                float(snapshot.old_p1),
                rel_tol=0.0,
                abs_tol=1.0e-7,
            ),
            "time_bin": str(label["time_bin"]) == _time_bin(decision_index, steps_per_episode),
        }
        if not all(checks.values()):
            mismatches.append({"key": list(key), "checks": checks})
        stored_identity_error = float(label["delta_identity_error"])
        computed_identity_error = (
            float(label["Delta_G"])
            - float(label["Delta_Q"])
            + float(label["Delta_cache"])
            + float(label["Delta_UAV"])
        )
        if not math.isfinite(stored_identity_error) or abs(computed_identity_error) > 1.0e-9:
            raise RuntimeError(f"held-out diagnosis Delta identity failed for {key}.")
        stored_identity_errors.append(stored_identity_error)
        records.append(
            _record_from_snapshot(
                snapshot,
                source="heldout",
                target_fields=label,
                spec=spec,
                steps_per_episode=steps_per_episode,
                policy_version=POLICY_VERSION,
            )
        )
    if missing or mismatches or len(records) != 24:
        raise RuntimeError(
            "held-out input reconstruction failed: "
            f"missing={missing}, mismatches={mismatches}"
        )
    return records, {
        "reused_old_diagnosis": True,
        "diagnosis_path": str(diagnosis.get("_path", "")),
        "sample_count": len(records),
        "missing": missing,
        "mismatches": mismatches,
        "max_stored_identity_error": float(max(abs(value) for value in stored_identity_errors)),
        "p_old_tolerance_abs": 1.0e-7,
        "feature_source": "fresh baseline DirectCFDecisionSnapshot keyed by trigger_index + decision_index",
    }


def _select_training_snapshots(
    snapshots: dict[tuple[int, int], ppo_rollout.DirectCFDecisionSnapshot],
    heldout_records: list[dict[str, Any]],
    *,
    steps_per_episode: int,
    seed: int,
) -> tuple[list[ppo_rollout.DirectCFDecisionSnapshot], dict[str, Any]]:
    gap_data_rng = np.random.default_rng(int(seed))
    heldout_keys = {
        (int(record["trigger_index"]), int(record["decision_index"])) for record in heldout_records
    }
    by_stratum: dict[str, list[ppo_rollout.DirectCFDecisionSnapshot]] = {}
    for snapshot in snapshots.values():
        stratum = f"{snapshot.trigger_label}::{_time_bin(snapshot.step_index, steps_per_episode)}"
        if (int(snapshot.trigger_index), int(snapshot.step_index)) not in heldout_keys:
            by_stratum.setdefault(stratum, []).append(snapshot)
    selected: list[ppo_rollout.DirectCFDecisionSnapshot] = []
    selected_keys: list[list[Any]] = []
    for stratum in sorted(by_stratum):
        candidates = sorted(
            by_stratum[stratum],
            key=lambda item: (int(item.step_index), int(item.trigger_index)),
        )
        if len(candidates) < TRAINING_PER_STRATUM:
            raise RuntimeError(f"stratum {stratum} has fewer than two NEW FREE decisions.")
        choices = gap_data_rng.choice(len(candidates), size=TRAINING_PER_STRATUM, replace=False)
        for choice in sorted(int(value) for value in choices):
            snapshot = candidates[choice]
            selected.append(snapshot)
            selected_keys.append([int(snapshot.trigger_index), int(snapshot.step_index)])
    if len(selected) != 48 or len({(int(item.trigger_index), int(item.step_index)) for item in selected}) != 48:
        raise RuntimeError("training snapshot selection did not produce 48 unique decisions.")
    return selected, {
        "rng": "independent gap_data_rng",
        "seed": int(seed),
        "per_stratum": TRAINING_PER_STRATUM,
        "strata": len(by_stratum),
        "selected_count": len(selected),
        "selected_keys": selected_keys,
        "excluded_heldout_count": len(heldout_keys),
    }


def _make_selected_reservoir(
    snapshots: list[ppo_rollout.DirectCFDecisionSnapshot],
    *,
    trigger_labels: Iterable[str],
    trigger_free_counts: tuple[int, ...],
) -> ppo_rollout.DirectCFReservoir:
    labels = tuple(str(label) for label in trigger_labels)
    if not snapshots:
        raise ValueError("selected snapshot list cannot be empty.")
    reservoir = ppo_rollout.DirectCFReservoir(
        max(1, len(snapshots)),
        np.random.default_rng(0),
        trigger_count=len(labels),
        trigger_labels=labels,
    )
    # The branch executor only iterates items.  Keep the complete baseline
    # FREE counts for reporting semantics while preventing accidental new
    # sampling during this critic-only collection.
    reservoir.items = list(snapshots)
    reservoir.total_seen = int(sum(int(value) for value in trigger_free_counts))
    reservoir._trigger_free_counts = [int(value) for value in trigger_free_counts]
    return reservoir


def _split_training_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(records) != 48:
        raise ValueError("fixed train/internal-val split requires exactly 48 NEW records.")
    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = f"{record['trigger_name']}::{record['time_bin']}"
        by_stratum.setdefault(key, []).append(record)
    if len(by_stratum) != 24 or any(len(values) != 2 for values in by_stratum.values()):
        raise ValueError("NEW dataset must contain exactly two records in each Trigger x time cell.")
    cells = sorted(by_stratum)
    split_rng = np.random.default_rng(int(seed))
    cell_order = [cells[int(index)] for index in split_rng.permutation(len(cells))]
    validation_cells = set(cell_order[:INTERNAL_VALIDATION_COUNT])
    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    validation_choice_keys: list[list[Any]] = []
    for cell in cells:
        cell_records = list(by_stratum[cell])
        order = split_rng.permutation(len(cell_records))
        if cell in validation_cells:
            validation_record = cell_records[int(order[0])]
            train_record = cell_records[int(order[1])]
            validation_records.append(validation_record)
            train_records.append(train_record)
            validation_choice_keys.append(
                [int(validation_record["trigger_index"]), int(validation_record["decision_index"])]
            )
        else:
            train_records.extend(cell_records[int(index)] for index in order)
    if len(train_records) != 36 or len(validation_records) != 12:
        raise AssertionError("fixed split sizes are not 36/12.")
    return train_records, validation_records, {
        "rng": "independent fixed split_rng",
        "seed": int(seed),
        "train_count": len(train_records),
        "internal_validation_count": len(validation_records),
        "validation_cells": sorted(validation_cells),
        "validation_keys": validation_choice_keys,
        "normalization_fit_split": "train_only",
    }


def _feature_matrix(records: list[dict[str, Any]], spec: GapCriticInputSpec) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("feature matrix requires at least one record.")
    features = np.stack(
        [
            build_gap_critic_features(
                record["observation"],
                record["prefix_values"],
                record["prefix_mask"],
                int(record["trigger_index"]),
                float(record["normalized_rollout_time"]),
                spec,
            )
            for record in records
        ]
    ).astype(np.float32)
    targets = np.asarray([float(record["Delta_G"]) for record in records], dtype=np.float32)
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
        raise ValueError("dataset feature/target matrix contains NaN/Inf.")
    return features, targets


def _fit_normalization(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(train_features, dtype=np.float64).mean(axis=0)
    std = np.asarray(train_features, dtype=np.float64).std(axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("train-only normalization statistics are nonfinite.")
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_normalization(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normalized = (np.asarray(features, dtype=np.float32) - mean) / std
    if not np.all(np.isfinite(normalized)):
        raise ValueError("normalized features contain NaN/Inf.")
    return normalized.astype(np.float32)


def _gradient_norm(model: GapCritic) -> float:
    _require_torch()
    squared = torch.zeros((), dtype=torch.float32)
    found = False
    for parameter in model.parameters():
        if parameter.grad is not None:
            found = True
            squared = squared + torch.sum(parameter.grad.detach() ** 2)
    if not found:
        raise RuntimeError("Gap Critic produced no gradients.")
    value = float(torch.sqrt(squared).detach().cpu())
    if not math.isfinite(value):
        raise FloatingPointError("Gap Critic gradient norm is nonfinite.")
    return value


def _fit_gap_critic(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    init_seed: int,
) -> tuple[GapCritic, Any, dict[str, Any], np.ndarray, np.ndarray]:
    _require_torch()
    if len(train_features) != 36 or len(validation_features) != 12:
        raise ValueError("Gap Critic fixed split must be 36 train / 12 internal validation.")
    normalization_mean, normalization_std = _fit_normalization(train_features)
    train_x = _apply_normalization(train_features, normalization_mean, normalization_std)
    validation_x = _apply_normalization(
        validation_features,
        normalization_mean,
        normalization_std,
    )
    critic = build_gap_critic_preserving_rng(train_features.shape[1], init_seed)
    optimizer = torch.optim.Adam(critic.parameters(), lr=DEFAULT_CRITIC_LEARNING_RATE)
    loss_fn = nn.MSELoss()
    train_x_t = torch.as_tensor(train_x, dtype=torch.float32)
    train_y_t = torch.as_tensor(train_targets, dtype=torch.float32)
    validation_x_t = torch.as_tensor(validation_x, dtype=torch.float32)
    validation_y_t = torch.as_tensor(validation_targets, dtype=torch.float32)
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    optimizer_steps = 0
    train_history: list[float] = []
    validation_history: list[float] = []
    gradient_history: list[float] = []
    for epoch in range(1, DEFAULT_CRITIC_MAX_EPOCHS + 1):
        critic.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = critic(train_x_t)
        train_loss = loss_fn(prediction, train_y_t)
        if not bool(torch.isfinite(train_loss).item()):
            raise FloatingPointError("Gap Critic train MSE is nonfinite.")
        train_loss.backward()
        gradient_norm = _gradient_norm(critic)
        optimizer.step()
        optimizer_steps += 1
        critic.eval()
        with torch.no_grad():
            validation_loss = loss_fn(critic(validation_x_t), validation_y_t)
        validation_loss_value = float(validation_loss.detach().cpu())
        if not math.isfinite(validation_loss_value):
            raise FloatingPointError("Gap Critic internal validation MSE is nonfinite.")
        train_history.append(float(train_loss.detach().cpu()))
        validation_history.append(validation_loss_value)
        gradient_history.append(gradient_norm)
        if validation_loss_value < best_validation_loss - DEFAULT_CRITIC_MIN_DELTA:
            best_validation_loss = validation_loss_value
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in critic.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= DEFAULT_CRITIC_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("Gap Critic did not produce a best internal-validation checkpoint.")
    critic.load_state_dict(best_state)
    critic.eval()
    return critic, optimizer, {
        "architecture": "MLP 128 -> 64 -> 1",
        "activation": "ReLU",
        "output": "linear scalar Delta_G",
        "loss": "MSE",
        "learning_rate": DEFAULT_CRITIC_LEARNING_RATE,
        "max_epochs": DEFAULT_CRITIC_MAX_EPOCHS,
        "early_stopping_patience": DEFAULT_CRITIC_PATIENCE,
        "epochs_completed": len(train_history),
        "best_epoch": best_epoch,
        "best_internal_validation_loss": best_validation_loss,
        "optimizer_steps": optimizer_steps,
        "train_loss_history": train_history,
        "internal_validation_loss_history": validation_history,
        "gradient_norm_history": gradient_history,
        "gradient_finite": bool(all(math.isfinite(value) for value in gradient_history)),
        "normalization": "feature mean/std fit on train split only",
    }, normalization_mean, normalization_std


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2:
        return None
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1.0e-15:
        return None
    value = float(np.dot(left_centered, right_centered) / denominator)
    return value if math.isfinite(value) else None


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    target_a = np.asarray(targets, dtype=np.float64)
    prediction_a = np.asarray(predictions, dtype=np.float64)
    if target_a.shape != prediction_a.shape or target_a.ndim != 1:
        raise ValueError("metrics target/prediction shapes must match one-dimensional arrays.")
    if not np.all(np.isfinite(target_a)) or not np.all(np.isfinite(prediction_a)):
        raise ValueError("metrics target/prediction contains NaN/Inf.")
    errors = prediction_a - target_a
    return {
        "count": int(len(target_a)),
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "Pearson": _pearson(target_a, prediction_a),
        "Spearman": _pearson(_rankdata(target_a), _rankdata(prediction_a)),
        "sign_accuracy": float(np.mean(np.sign(target_a) == np.sign(prediction_a))),
        "mean_absolute_target": float(np.mean(np.abs(target_a))),
    }


def _group_metrics(
    records: list[dict[str, Any]],
    targets: np.ndarray,
    predictions: np.ndarray,
    field: str,
    ordered_values: Iterable[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in ordered_values:
        indices = [index for index, record in enumerate(records) if str(record[field]) == str(value)]
        result[str(value)] = _metrics(targets[indices], predictions[indices]) if indices else {
            "count": 0,
            "MAE": None,
            "sign_accuracy": None,
        }
    return result


def _state_time_dependence(
    records: list[dict[str, Any]],
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    true_flips: list[str] = []
    predicted_flips: list[str] = []
    predicted_on_true_flip: list[str] = []
    by_trigger: dict[str, dict[str, dict[str, float]]] = {}
    for trigger_index in range(TRIGGER_COUNT):
        trigger_records = [
            (index, record)
            for index, record in enumerate(records)
            if int(record["trigger_index"]) == trigger_index
        ]
        trigger_name = str(trigger_records[0][1]["trigger_name"]) if trigger_records else str(trigger_index)
        signs_true: list[int] = []
        signs_predicted: list[int] = []
        by_trigger[trigger_name] = {}
        for time_bin in TIME_BINS:
            indices = [index for index, record in trigger_records if record["time_bin"] == time_bin]
            if len(indices) != 1:
                continue
            index = indices[0]
            signs_true.append(int(np.sign(targets[index])))
            signs_predicted.append(int(np.sign(predictions[index])))
            by_trigger[trigger_name][time_bin] = {
                "target": float(targets[index]),
                "prediction": float(predictions[index]),
                "target_sign": int(np.sign(targets[index])),
                "prediction_sign": int(np.sign(predictions[index])),
            }
        if len(signs_true) == 3:
            true_flip = len(set(signs_true)) > 1
            predicted_flip = len(set(signs_predicted)) > 1
            if true_flip:
                true_flips.append(trigger_name)
            if predicted_flip:
                predicted_flips.append(trigger_name)
            if true_flip and predicted_flip:
                predicted_on_true_flip.append(trigger_name)
    return {
        "by_trigger_time": by_trigger,
        "true_sign_flip_triggers": true_flips,
        "predicted_sign_flip_triggers": predicted_flips,
        "predicted_sign_flip_on_true_flip_triggers": predicted_on_true_flip,
        "true_sign_flip_count": len(true_flips),
        "predicted_sign_flip_count": len(predicted_flips),
        "state_time_dependence_observed": bool(true_flips),
        "critic_tracks_at_least_one_true_cross_time_flip": bool(predicted_on_true_flip),
    }


def _systematic_opposite_groups(
    records: list[dict[str, Any]],
    targets: np.ndarray,
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    groups: list[tuple[str, str]] = [("trigger_name", str(value)) for value in sorted({str(r["trigger_name"]) for r in records})]
    groups.extend(("time_bin", value) for value in TIME_BINS)
    result: list[dict[str, Any]] = []
    for field, value in groups:
        indices = [index for index, record in enumerate(records) if str(record[field]) == value]
        if len(indices) < 3:
            continue
        opposite = [
            index
            for index in indices
            if float(targets[index]) != 0.0
            and float(predictions[index]) != 0.0
            and float(targets[index]) * float(predictions[index]) < 0.0
        ]
        if len(opposite) == len(indices):
            result.append({"field": field, "value": value, "count": len(indices)})
    return result


def _trigger_time_baseline(
    train_records: list[dict[str, Any]],
    heldout_records: list[dict[str, Any]],
    global_mean_fallback: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    targets = np.asarray([float(record["Delta_G"]) for record in train_records], dtype=np.float64)
    global_mean = float(np.mean(targets)) if len(targets) else 0.0
    trigger_means: dict[str, float] = {}
    cell_means: dict[str, float] = {}
    for trigger in sorted({str(record["trigger_name"]) for record in train_records}):
        values = [float(record["Delta_G"]) for record in train_records if str(record["trigger_name"]) == trigger]
        if values:
            trigger_means[trigger] = float(np.mean(values))
    for cell in sorted({f"{r['trigger_name']}::{r['time_bin']}" for r in train_records}):
        values = [
            float(record["Delta_G"])
            for record in train_records
            if f"{record['trigger_name']}::{record['time_bin']}" == cell
        ]
        if values:
            cell_means[cell] = float(np.mean(values))
    predictions: list[float] = []
    fallback_counts = {"cell": 0, "trigger": 0, "global": 0}
    for record in heldout_records:
        cell = f"{record['trigger_name']}::{record['time_bin']}"
        if cell in cell_means:
            predictions.append(cell_means[cell])
            fallback_counts["cell"] += 1
        elif str(record["trigger_name"]) in trigger_means:
            predictions.append(trigger_means[str(record["trigger_name"])])
            fallback_counts["trigger"] += 1
        elif global_mean_fallback:
            predictions.append(global_mean)
            fallback_counts["global"] += 1
        else:
            raise RuntimeError(f"no Trigger x time baseline fallback for {cell}")
    return np.asarray(predictions, dtype=np.float64), {
        "hierarchy": "Trigger x time-bin mean -> Trigger mean -> global mean",
        "cell_mean_count": len(cell_means),
        "trigger_mean_count": len(trigger_means),
        "global_mean": global_mean,
        "fallback_counts": fallback_counts,
    }


def _evaluate(
    critic: GapCritic,
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    heldout_records: list[dict[str, Any]],
    spec: GapCriticInputSpec,
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
) -> dict[str, Any]:
    _require_torch()
    train_features, train_targets = _feature_matrix(train_records, spec)
    validation_features, validation_targets = _feature_matrix(validation_records, spec)
    heldout_features, heldout_targets = _feature_matrix(heldout_records, spec)
    with torch.no_grad():
        train_predictions = critic(torch.as_tensor(_apply_normalization(train_features, normalization_mean, normalization_std))).detach().cpu().numpy().astype(np.float64)
        validation_predictions = critic(torch.as_tensor(_apply_normalization(validation_features, normalization_mean, normalization_std))).detach().cpu().numpy().astype(np.float64)
        heldout_predictions = critic(torch.as_tensor(_apply_normalization(heldout_features, normalization_mean, normalization_std))).detach().cpu().numpy().astype(np.float64)
    zero_predictions = np.zeros_like(heldout_targets, dtype=np.float64)
    trigger_time_predictions, trigger_time_metadata = _trigger_time_baseline(
        train_records,
        heldout_records,
    )
    heldout_metrics = _metrics(heldout_targets, heldout_predictions)
    zero_metrics = _metrics(heldout_targets, zero_predictions)
    trigger_time_metrics = _metrics(heldout_targets, trigger_time_predictions)
    return {
        "train": _metrics(train_targets, train_predictions),
        "internal_validation": _metrics(validation_targets, validation_predictions),
        "heldout": heldout_metrics,
        "zero_predictor": zero_metrics,
        "trigger_time_baseline": {
            **trigger_time_metrics,
            "definition": trigger_time_metadata,
        },
        "per_trigger": _group_metrics(
            heldout_records,
            heldout_targets,
            heldout_predictions,
            "trigger_name",
            sorted({str(record["trigger_name"]) for record in heldout_records}),
        ),
        "per_time_bin": _group_metrics(
            heldout_records,
            heldout_targets,
            heldout_predictions,
            "time_bin",
            TIME_BINS,
        ),
        "state_time_dependence": _state_time_dependence(
            heldout_records,
            heldout_targets,
            heldout_predictions,
        ),
        "prediction_all_finite": bool(np.all(np.isfinite(heldout_predictions))),
        "heldout_predictions": [
            {
                "trigger": str(record["trigger_name"]),
                "time_bin": str(record["time_bin"]),
                "decision_index": int(record["decision_index"]),
                "Delta_G": float(target),
                "Delta_hat": float(prediction),
            }
            for record, target, prediction in zip(
                heldout_records,
                heldout_targets,
                heldout_predictions,
            )
        ],
    }


def _pilot_gates(evaluation: dict[str, Any]) -> dict[str, Any]:
    heldout = evaluation["heldout"]
    zero = evaluation["zero_predictor"]
    trigger_time = evaluation["trigger_time_baseline"]
    systematic = _systematic_opposite_groups(
        [
            {
                "trigger_name": item["trigger"],
                "time_bin": item["time_bin"],
            }
            for item in evaluation["heldout_predictions"]
        ],
        np.asarray([item["Delta_G"] for item in evaluation["heldout_predictions"]], dtype=np.float64),
        np.asarray([item["Delta_hat"] for item in evaluation["heldout_predictions"]], dtype=np.float64),
    )
    gates = {
        "A_sign_accuracy_ge_0.70": float(heldout["sign_accuracy"]) >= 0.70,
        "B_Pearson_ge_0.50": heldout["Pearson"] is not None and float(heldout["Pearson"]) >= 0.50,
        "C_Spearman_ge_0.50": heldout["Spearman"] is not None and float(heldout["Spearman"]) >= 0.50,
        "D_MAE_at_least_20pct_below_zero": float(heldout["MAE"]) <= 0.80 * float(zero["MAE"]),
        "E_not_materially_worse_than_trigger_time": float(heldout["MAE"]) <= 1.05 * float(trigger_time["MAE"]),
        "F_all_predictions_finite": bool(evaluation["prediction_all_finite"]),
        "G_no_systematic_opposite_large_group": len(systematic) == 0,
    }
    return {
        "gates": gates,
        "pass": bool(all(gates.values())),
        "systematic_opposite_groups": systematic,
        "thresholds": {
            "sign_accuracy": 0.70,
            "Pearson": 0.50,
            "Spearman": 0.50,
            "zero_mae_reduction": 0.20,
            "trigger_time_mae_tolerance": 0.05,
        },
    }


def _record_for_dataset(record: dict[str, Any]) -> dict[str, Any]:
    """Drop transient NumPy feature arrays from the human-light record copy."""

    return {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key not in {"feature"}
    }


def run_paired_cf_gap_critic_pilot(
    *,
    config_path: str = "system_config.toml",
    diagnosis_path: str = "runs/v3_targeted_state_representation_diagnosis.json",
    run_dir: str | Path = "runs/ppo_gap_critic_paired_pilot_001",
    gap_data_seed: int = DEFAULT_GAP_DATA_SEED,
    split_seed: int = DEFAULT_SPLIT_SEED,
    critic_init_seed: int = DEFAULT_CRITIC_INIT_SEED,
) -> dict[str, Any]:
    """Execute the bounded paired-CF Gap Critic pilot.

    The function raises on any baseline/reconstruction/identity violation.
    A completed pilot returns a validation payload even when the predeclared
    engineering PASS gate is not met; a FAIL is a valid scientific result and
    never connects the critic to Actor training.
    """

    _require_torch()
    output_dir = Path(run_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty pilot directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnosis_file = Path(diagnosis_path)
    if not diagnosis_file.is_absolute():
        diagnosis_file = Path.cwd() / diagnosis_file
    diagnosis_file = diagnosis_file.resolve()
    diagnosis = _load_diagnosis(diagnosis_file)
    diagnosis["_path"] = str(diagnosis_file)

    context = ppo_training._build_context(config_path)
    protocol = ppo_training.build_formal_training_protocol(context.scenarios, context.cfg)
    ppo_training.validate_formal_training_protocol(protocol)
    if int(context.action_spec.action_dim) != TRIGGER_COUNT:
        raise RuntimeError("Gap Critic pilot currently requires exactly 8 triggers.")
    if ppo_training._is_sparse_direct_cf_mode(context.cfg) is not True:
        raise RuntimeError("Gap Critic pilot requires the existing sparse Direct-CF semantics.")
    policy_seed = int(project_config.get_config_value(context.cfg, "experiment.seed"))
    torch_generator = torch.Generator()
    ppo_training._seed_new_formal_run(policy_seed, torch_generator)
    policy, actor_optimizer, formal_critic, formal_critic_optimizer = ppo_training._build_formal_training_components(context)
    if formal_critic is not None or formal_critic_optimizer is not None:
        raise RuntimeError("V1/V2/centralized critic unexpectedly present in sparse pilot.")
    policy.eval()
    actor_state_before = {
        key: value.detach().cpu().clone() for key, value in policy.state_dict().items()
    }
    value_head_state_before = {
        key: value.detach().cpu().clone()
        for key, value in policy.state_dict().items()
        if key.startswith("value_head.")
    }
    steps = ppo_training._formal_aligned_steps(context, protocol)
    all_snapshot_reservoir = ppo_rollout.DirectCFReservoir(
        int(protocol.steps_per_episode) * TRIGGER_COUNT,
        np.random.default_rng(int(gap_data_seed) + 17),
        trigger_count=TRIGGER_COUNT,
        trigger_labels=tuple(context.action_spec.labels),
    )
    baseline_batch = ppo_training.collect_formal_episode_rollout(
        policy,
        context,
        protocol,
        all_snapshot_reservoir,
    )
    baseline_stats = ppo_training._formal_raw_rollout_statistics(baseline_batch)
    baseline_report = _assert_baseline(
        baseline_stats,
        baseline_batch,
        all_snapshot_reservoir.total_seen,
    )
    snapshots = _snapshot_map(all_snapshot_reservoir)
    spec = GapCriticInputSpec(observation_dim=int(context.observation_spec.observation_dim))
    heldout_records, heldout_report = _reconstruct_heldout(
        diagnosis,
        snapshots,
        spec,
        protocol.steps_per_episode,
    )
    selected_snapshots, selection_report = _select_training_snapshots(
        snapshots,
        heldout_records,
        steps_per_episode=protocol.steps_per_episode,
        seed=gap_data_seed,
    )
    branch_reservoir = _make_selected_reservoir(
        selected_snapshots,
        trigger_labels=context.action_spec.labels,
        trigger_free_counts=all_snapshot_reservoir.trigger_free_counts,
    )
    with torch.no_grad():
        train_branch_samples = ppo_rollout.run_direct_cf_branches(
            policy,
            branch_reservoir,
            steps,
            baseline_batch,
            context.local_cache,
            context.service_catalog,
            context.cfg,
            context.cache_nodes,
            context.cache_cfg,
            context.flight_boundary,
            context.observation_spec,
            context.action_spec,
            context.terminal_slot,
        )
    if len(train_branch_samples) != 48:
        raise RuntimeError(f"new paired branch count is {len(train_branch_samples)}, expected 48.")
    train_records_all = [
        _record_from_branch_sample(
            sample,
            spec=spec,
            steps_per_episode=protocol.steps_per_episode,
            policy_version=POLICY_VERSION,
            time_s=float(steps[int(sample.origin_index)].time_s),
        )
        for sample in train_branch_samples
    ]
    identity_errors = [abs(float(record["delta_identity_error"])) for record in train_records_all]
    if not identity_errors or max(identity_errors) > 1.0e-9:
        raise RuntimeError(f"new paired label identity error too large: {max(identity_errors)}")
    train_records, validation_records, split_report = _split_training_records(
        train_records_all,
        seed=split_seed,
    )
    train_features, train_targets = _feature_matrix(train_records, spec)
    validation_features, validation_targets = _feature_matrix(validation_records, spec)
    critic, critic_optimizer, training_report, normalization_mean, normalization_std = _fit_gap_critic(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        init_seed=critic_init_seed,
    )
    evaluation = _evaluate(
        critic,
        train_records,
        validation_records,
        heldout_records,
        spec,
        normalization_mean,
        normalization_std,
    )
    gate_report = _pilot_gates(evaluation)

    actor_state_after = {
        key: value.detach().cpu().clone() for key, value in policy.state_dict().items()
    }
    value_head_state_after = {
        key: value.detach().cpu().clone()
        for key, value in policy.state_dict().items()
        if key.startswith("value_head.")
    }
    actor_parameter_delta_max = max(
        float(torch.max(torch.abs(actor_state_after[key] - actor_state_before[key])).item())
        for key in actor_state_before
    )
    value_head_parameter_delta_max = max(
        float(torch.max(torch.abs(value_head_state_after[key] - value_head_state_before[key])).item())
        for key in value_head_state_before
    )
    if actor_parameter_delta_max != 0.0 or value_head_parameter_delta_max != 0.0:
        raise AssertionError("frozen Actor/value_head changed during critic-only pilot.")
    if int(POLICY_VERSION) != 0:
        raise AssertionError("critic-only frozen policy version must remain zero.")

    dataset_records = [
        _record_for_dataset(record)
        for record in train_records + validation_records + heldout_records
    ]
    train_count = len(train_records)
    validation_count = len(validation_records)
    heldout_count = len(heldout_records)
    dataset_path = output_dir / "paired_cf_dataset.pt"
    torch.save(
        {
            "schema_version": DATASET_SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "input_schema": spec.as_dict(),
            "records": dataset_records,
            "train_indices": list(range(0, train_count)),
            "internal_validation_indices": list(range(train_count, train_count + validation_count)),
            "heldout_indices": list(range(train_count + validation_count, train_count + validation_count + heldout_count)),
            "normalization": {
                "fit_on": "train",
                "mean": normalization_mean,
                "std": normalization_std,
            },
            "baseline": baseline_report,
            "branch_budget": {
                "fresh_baseline_count": 1,
                "new_paired_branch_count": len(train_branch_samples),
                "heldout_reused_count": len(heldout_records),
            },
            "split": split_report,
        },
        dataset_path,
    )
    checkpoint_path = output_dir / "gap_critic.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": critic.state_dict(),
            "optimizer_state_dict": critic_optimizer.state_dict(),
            "input_schema": spec.as_dict(),
            "policy_version": POLICY_VERSION,
            "train_split": split_report,
            "normalization": {
                "fit_on": "train",
                "mean": normalization_mean,
                "std": normalization_std,
            },
            "training": training_report,
        },
        checkpoint_path,
    )
    branch_steps = int(sum(int(sample.branch_environment_steps) for sample in train_branch_samples))
    validation_payload = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "pilot": "Paired-CF Gap Critic (Critic-Only)",
        "policy_version": POLICY_VERSION,
        "policy_version_consistent": True,
        "baseline": baseline_report,
        "heldout_reconstruction": heldout_report,
        "data_collection": {
            "fresh_baseline_count": 1,
            "new_paired_branch_count": len(train_branch_samples),
            "new_branch_environment_steps": branch_steps,
            "heldout_reused_count": len(heldout_records),
            "fallback_used": False,
            "selection": selection_report,
        },
        "dataset_sizes": {
            "total": len(dataset_records),
            "train": train_count,
            "internal_validation": validation_count,
            "heldout": heldout_count,
        },
        "input_schema": spec.as_dict(),
        "target": {
            "name": "Delta_G",
            "definition": "G(action1) - G(action0)",
            "source": "true paired-CF label from frozen old policy continuation",
            "mc_return": False,
            "gae": False,
            "scalar_value_target": False,
            "q_target": False,
            "return_scale": None,
        },
        "training": training_report,
        "evaluation": evaluation,
        "pilot_gate": gate_report,
        "execution_guard": {
            "formal_environment_steps": len(baseline_batch.transitions),
            "formal_baseline_count": 1,
            "counterfactual_branch_count": len(train_branch_samples),
            "counterfactual_branch_environment_steps": branch_steps,
            "formal_actor_optimizer_steps": 0,
            "formal_actor_backward_calls": 0,
            "gap_critic_optimizer_steps": int(training_report["optimizer_steps"]),
            "formal_ppo_critic_optimizer_steps": 0,
            "clone_optimizer_steps": 0,
            "clone_backward_calls": 0,
            "deterministic_evaluation": False,
            "formal_training_started": False,
            "actor_parameter_delta_max": actor_parameter_delta_max,
            "value_head_parameter_delta_max": value_head_parameter_delta_max,
            "value_head_unchanged": value_head_parameter_delta_max == 0.0,
            "v1_v2_critic_participated": False,
            "gap_critic_gradient_finite": bool(training_report["gradient_finite"]),
        },
        "artifacts": {
            "paired_cf_dataset": str(dataset_path.resolve()),
            "gap_critic_checkpoint": str(checkpoint_path.resolve()),
            "validation": str((output_dir / "gap_critic_validation.json").resolve()),
            "dataset_summary": str((output_dir / "paired_cf_dataset_summary.json").resolve()),
        },
        "decision": {
            "pass": bool(gate_report["pass"]),
            "statement_if_pass": "Paired-CF Gap Critic 在 held-out真实paired labels 上具备初步state-dependent generalization能力。",
            "next_dense_gap_actor_allowed": bool(gate_report["pass"]),
            "actor_update_performed": False,
        },
    }
    _save_json(output_dir / "gap_critic_validation.json", validation_payload)
    summary_payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_path": str(dataset_path.resolve()),
        "policy_version": POLICY_VERSION,
        "policy_version_consistent": True,
        "dataset_sizes": validation_payload["dataset_sizes"],
        "input_schema": spec.as_dict(),
        "target": validation_payload["target"],
        "baseline_passed": True,
        "fresh_baseline_count": 1,
        "new_paired_branch_count": len(train_branch_samples),
        "heldout_reused_count": len(heldout_records),
        "max_new_delta_identity_abs_error": max(identity_errors),
        "max_heldout_stored_delta_identity_abs_error": heldout_report["max_stored_identity_error"],
        "split": split_report,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "validation_path": str((output_dir / "gap_critic_validation.json").resolve()),
    }
    _save_json(output_dir / "paired_cf_dataset_summary.json", summary_payload)
    return validation_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the paired-CF Gap Critic critic-only pilot.")
    parser.add_argument("--config", default="system_config.toml")
    parser.add_argument(
        "--diagnosis",
        default="runs/v3_targeted_state_representation_diagnosis.json",
    )
    parser.add_argument(
        "--run-dir",
        default="runs/ppo_gap_critic_paired_pilot_001",
    )
    parser.add_argument("--gap-data-seed", type=int, default=DEFAULT_GAP_DATA_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--critic-init-seed", type=int, default=DEFAULT_CRITIC_INIT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_paired_cf_gap_critic_pilot(
        config_path=args.config,
        diagnosis_path=args.diagnosis,
        run_dir=args.run_dir,
        gap_data_seed=args.gap_data_seed,
        split_seed=args.split_seed,
        critic_init_seed=args.critic_init_seed,
    )
    print(
        json.dumps(
            {
                "pass": result["pilot_gate"]["pass"],
                "heldout": result["evaluation"]["heldout"],
                "zero_predictor": result["evaluation"]["zero_predictor"],
                "trigger_time_baseline": result["evaluation"]["trigger_time_baseline"],
                "artifacts": result["artifacts"],
                "execution_guard": result["execution_guard"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
