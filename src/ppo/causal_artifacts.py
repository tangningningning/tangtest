# -*- coding: utf-8 -*-
"""Atomic persistence for exact online-update causal audits.

This module is deliberately a persistence-only sidecar.  It does not sample,
run an environment, compute a loss, or advance an optimizer.  The formal
training loop supplies already-produced states and summaries; this module
serializes and validates them without consuming RNG.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ppo import ppo_rollout

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


CAUSAL_ARTIFACT_SCHEMA_ID = "exact_update_causal_artifacts_v1"
CAUSAL_ARTIFACT_SCHEMA_VERSION = 1
CAUSAL_ARTIFACT_DIRNAME = "causal_update_artifacts"
EPISODE_START_DIRNAME = "episode_start_artifacts"
LEGACY_ARTIFACT_GAP_FILENAME = "causal_artifact_legacy_gap.json"
LEGACY_ARTIFACT_GAP_SCHEMA_ID = f"{CAUSAL_ARTIFACT_SCHEMA_ID}.legacy_gap_v1"
PRE_UPDATE_FILENAME = "pre_update.pt"
POST_UPDATE_FILENAME = "post_update.pt"
CF_BATCH_FILENAME = "cf_batch.pt"
BOUNDARY_RNG_FILENAME = "boundary_rng.pt"
METADATA_FILENAME = "metadata.json"
EPISODE_START_SCHEMA_ID = f"{CAUSAL_ARTIFACT_SCHEMA_ID}.episode_start_v1"
# Hierarchical Gate artifacts are a separate extension.  Existing Direct-CF
# update directories keep their exact schema and are never rewritten by this
# path.
HIERARCHICAL_GATE_ARTIFACT_SCHEMA_ID = (
    f"{CAUSAL_ARTIFACT_SCHEMA_ID}.hierarchical_gate_update_v1"
)
HIERARCHICAL_GATE_ARTIFACT_FILENAME = "gate_update.pt"

_UPDATE_DIRECTORY_RE = re.compile(r"^update_(\d{4,})$")
_EPISODE_START_RE = re.compile(r"^episode_(\d{4,})\.pt$")


def _require_torch() -> None:
    if torch is None:
        raise ImportError("causal_artifacts.py 需要 PyTorch。")


def _clone(value: Any) -> Any:
    """Deep-copy tensors/arrays without invoking any random operation."""

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return copy.deepcopy(value)


def _to_builtin(value: Any) -> Any:
    """Convert summaries/metadata to JSON-safe builtin values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_builtin(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return _to_builtin(value.item())
    if torch is not None and isinstance(value, torch.Tensor):
        return _to_builtin(value.detach().cpu().tolist())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("causal artifact metadata包含NaN/Inf。")
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(f"causal artifact不支持的metadata类型：{type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _to_builtin(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def config_fingerprint(compatibility_snapshot: dict[str, Any]) -> str:
    """Hash the exact compatibility/config snapshot stored with an update."""

    return hashlib.sha256(_canonical_json(compatibility_snapshot).encode("utf-8")).hexdigest()


def _atomic_torch_save(payload: dict[str, Any], final_path: Path) -> None:
    _require_torch()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{final_path.name}.",
        suffix=".tmp",
        dir=str(final_path.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        # Windows rejects fsync on a read-only descriptor; use a read/write
        # descriptor solely for the durability barrier.
        with temporary_path.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, final_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_json_save(payload: dict[str, Any], final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{final_path.name}.",
        suffix=".tmp",
        dir=str(final_path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = -1
            json.dump(_to_builtin(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, final_path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path.exists():
            temporary_path.unlink()


def _torch_load(path: Path) -> Any:
    _require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location="cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name}必须是大于等于{minimum}的整数。")
    return int(value)


def _sample_to_artifact_dict(sample: ppo_rollout.DirectCFCreditSample) -> dict[str, Any]:
    value = ppo_rollout.direct_cf_sample_to_checkpoint_dict(sample)
    episode_index = int(getattr(sample, "episode_index", 0))
    value["episode"] = episode_index
    value["episode_index"] = episode_index
    return value


def _sample_from_artifact_dict(value: dict[str, Any]) -> ppo_rollout.DirectCFCreditSample:
    if not isinstance(value, dict):
        raise ValueError("causal cf sample必须是dict。")
    return ppo_rollout.direct_cf_sample_from_checkpoint_dict(value)


def _reference_to_artifact_dict(record: ppo_rollout.ReferenceStateRecord) -> dict[str, Any]:
    return ppo_rollout.reference_state_to_checkpoint_dict(record)


def _reference_from_artifact_dict(value: dict[str, Any]) -> ppo_rollout.ReferenceStateRecord:
    if not isinstance(value, dict):
        raise ValueError("causal reference record必须是dict。")
    return ppo_rollout.reference_state_from_checkpoint_dict(value)


def _validate_sample_collection(
    samples: Iterable[ppo_rollout.DirectCFCreditSample],
    expected_count: int | None = None,
) -> list[ppo_rollout.DirectCFCreditSample]:
    restored = list(samples)
    if expected_count is not None and len(restored) != int(expected_count):
        raise ValueError(
            f"causal cf sample count={len(restored)}，期望{int(expected_count)}。"
        )
    if not restored:
        raise ValueError("causal cf batch不能为空。")
    identifiers = [str(item.decision_id) for item in restored]
    if any(not item for item in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("causal cf batch decision_id必须非空且唯一。")
    versions = {int(item.collection_policy_version) for item in restored}
    if len(versions) != 1:
        raise ValueError("causal cf batch不能混合collection_policy_version。")
    return restored


def _build_cf_batch_payload(
    *,
    update_index: int,
    batch_id: int,
    samples: Iterable[ppo_rollout.DirectCFCreditSample],
    training_samples: Iterable[ppo_rollout.DirectCFCreditSample] | None,
    reference_records: Iterable[ppo_rollout.ReferenceStateRecord] | None,
    expected_sample_count: int | None,
) -> dict[str, Any]:
    sample_list = _validate_sample_collection(samples, expected_sample_count)
    training_sample_list = _validate_sample_collection(
        sample_list if training_samples is None else training_samples,
        expected_sample_count,
    )
    if [str(item.decision_id) for item in training_sample_list] != [
        str(item.decision_id) for item in sample_list
    ]:
        raise ValueError("causal raw/training cf samples decision_id顺序不一致。")
    reference_list = list(reference_records or [])
    reference_ids = [str(item.decision_id) for item in reference_list]
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("causal R_all decision_id必须唯一。")
    selected_ids = [str(item.decision_id) for item in sample_list]
    if reference_list:
        _, _, partition = _validate_reference_partition(reference_list, selected_ids)
        selected_count = int(partition["selected_count"])
        r_ref_count = int(partition["reference_count"])
        intersection_count = int(partition["intersection_count"])
    else:
        selected_count = len(selected_ids)
        r_ref_count = 0
        intersection_count = 0
    sample_versions = {int(item.collection_policy_version) for item in sample_list}
    reference_versions = {int(item.collection_policy_version) for item in reference_list}
    if reference_versions and reference_versions != sample_versions:
        raise ValueError("causal cf/reference collection policy version不一致。")
    return {
        "schema": f"{CAUSAL_ARTIFACT_SCHEMA_ID}.cf_batch_v1",
        "causal_artifact_schema": CAUSAL_ARTIFACT_SCHEMA_ID,
        "update_index": int(update_index),
        "batch_id": int(batch_id),
        "sample_count": len(sample_list),
        "samples": [_sample_to_artifact_dict(item) for item in sample_list],
        "training_samples": [
            _sample_to_artifact_dict(item) for item in training_sample_list
        ],
        "training_sample_count": len(training_sample_list),
        "r_all_records": [_reference_to_artifact_dict(item) for item in reference_list],
        "selected_ids": selected_ids,
        "r_all_count": len(reference_list),
        "selected_count": selected_count,
        "r_ref_count": r_ref_count,
        "intersection_count": intersection_count,
        "collection_policy_version": next(iter(sample_versions)),
        "training_sampling_weights": [
            float(item.sampling_weight) for item in training_sample_list
        ],
    }


def _validate_reference_partition(
    records: Iterable[ppo_rollout.ReferenceStateRecord],
    selected_ids: Iterable[str],
) -> tuple[list[ppo_rollout.ReferenceStateRecord], list[ppo_rollout.ReferenceStateRecord], dict[str, int]]:
    items = list(records)
    identifiers = [str(item.decision_id) for item in items]
    selected = [str(item) for item in selected_ids]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("causal R_all decision_id必须唯一。")
    if len(set(selected)) != len(selected):
        raise ValueError("causal selected IDs必须唯一。")
    selected_set = set(selected)
    if not selected_set.issubset(set(identifiers)):
        raise ValueError("causal selected IDs必须来自R_all。")
    selected_records = [item for item in items if str(item.decision_id) in selected_set]
    reference_records = [item for item in items if str(item.decision_id) not in selected_set]
    intersection = len(
        {str(item.decision_id) for item in selected_records}
        & {str(item.decision_id) for item in reference_records}
    )
    return selected_records, reference_records, {
        "all_count": len(items),
        "selected_count": len(selected_records),
        "reference_count": len(reference_records),
        "intersection_count": intersection,
    }


def _validate_cf_batch_payload(
    payload: Any,
    *,
    expected_update_index: int | None = None,
    expected_sample_count: int | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("causal cf_batch payload必须是dict。")
    required = {
        "schema",
        "causal_artifact_schema",
        "update_index",
        "batch_id",
        "sample_count",
        "samples",
        "training_samples",
        "training_sample_count",
        "r_all_records",
        "selected_ids",
        "r_all_count",
        "selected_count",
        "r_ref_count",
        "intersection_count",
        "collection_policy_version",
        "training_sampling_weights",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"causal cf_batch缺少字段：{missing}")
    if payload["schema"] != f"{CAUSAL_ARTIFACT_SCHEMA_ID}.cf_batch_v1":
        raise ValueError("causal cf_batch schema不兼容。")
    if payload["causal_artifact_schema"] != CAUSAL_ARTIFACT_SCHEMA_ID:
        raise ValueError("causal cf_batch causal_artifact_schema不兼容。")
    update_index = _require_int(payload["update_index"], "cf_batch.update_index", minimum=1)
    if expected_update_index is not None and update_index != int(expected_update_index):
        raise ValueError("causal cf_batch update_index不一致。")
    _require_int(payload["batch_id"], "cf_batch.batch_id", minimum=1)
    raw_samples = payload["samples"]
    raw_training_samples = payload["training_samples"]
    raw_references = payload["r_all_records"]
    if (
        not isinstance(raw_samples, list)
        or not isinstance(raw_training_samples, list)
        or not isinstance(raw_references, list)
    ):
        raise ValueError("causal cf_batch samples/training_samples/R_all必须是list。")
    samples = [_sample_from_artifact_dict(item) for item in raw_samples]
    _validate_sample_collection(samples, expected_sample_count)
    training_samples = [_sample_from_artifact_dict(item) for item in raw_training_samples]
    _validate_sample_collection(training_samples, expected_sample_count)
    if [str(item.decision_id) for item in training_samples] != [
        str(item.decision_id) for item in samples
    ]:
        raise ValueError("causal cf_batch raw/training samples decision_id顺序不一致。")
    sample_version = {int(item.collection_policy_version) for item in samples}
    training_version = {int(item.collection_policy_version) for item in training_samples}
    if training_version != sample_version:
        raise ValueError("causal raw/training cf samples collection policy version不一致。")
    if int(payload["sample_count"]) != len(samples):
        raise ValueError("causal cf_batch sample_count不一致。")
    if int(payload["training_sample_count"]) != len(training_samples):
        raise ValueError("causal cf_batch training_sample_count不一致。")
    references = [_reference_from_artifact_dict(item) for item in raw_references]
    selected_ids = [str(item) for item in payload["selected_ids"]]
    if selected_ids != [str(item.decision_id) for item in samples]:
        raise ValueError("causal cf_batch selected_ids与samples顺序不一致。")
    if references:
        _, _, partition = _validate_reference_partition(references, selected_ids)
    else:
        partition = {
            "all_count": 0,
            "selected_count": len(samples),
            "reference_count": 0,
            "intersection_count": 0,
        }
    for field_name, expected in (
        ("r_all_count", partition["all_count"]),
        ("selected_count", partition["selected_count"]),
        ("r_ref_count", partition["reference_count"]),
        ("intersection_count", partition["intersection_count"]),
    ):
        if int(payload[field_name]) != int(expected):
            raise ValueError(f"causal cf_batch {field_name}不一致。")
    if expected_sample_count is not None and len(samples) != int(expected_sample_count):
        raise ValueError("causal cf_batch expected sample count不一致。")
    if payload.get("training_sampling_weights") != [
        float(item.sampling_weight) for item in training_samples
    ]:
        raise ValueError("causal cf_batch training_sampling_weights不一致。")
    return payload


def _common_pre_metadata(
    *,
    update_index: int,
    batch_id: int,
    completed_episodes: int,
    collection_policy_version: int,
    policy_version_pre: int,
    optimizer_step_pre: int,
    pending_cf_count: int,
    pending_reference_count: int,
    reference_drift_cap: float | None,
    algorithm_schema: str,
    architecture_id: str,
    checkpoint_schema: int,
    compatibility_snapshot: dict[str, Any],
    boundary_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": f"{CAUSAL_ARTIFACT_SCHEMA_ID}.pre_update_v1",
        "causal_artifact_schema": CAUSAL_ARTIFACT_SCHEMA_ID,
        "update_index": int(update_index),
        "batch_id": int(batch_id),
        "completed_episodes": int(completed_episodes),
        "collection_policy_version": int(collection_policy_version),
        "policy_version_pre": int(policy_version_pre),
        "optimizer_step_pre": int(optimizer_step_pre),
        "pending_cf_count": int(pending_cf_count),
        "pending_cf_sample_count": int(pending_cf_count),
        "pending_reference_count": int(pending_reference_count),
        "pending_reference_state_count": int(pending_reference_count),
        "reference_drift_cap": None if reference_drift_cap is None else float(reference_drift_cap),
        "algorithm_schema": str(algorithm_schema),
        "architecture_id": str(architecture_id),
        "checkpoint_schema": int(checkpoint_schema),
        "config_fingerprint": config_fingerprint(compatibility_snapshot),
        "relevant_training_config_snapshot": _clone(compatibility_snapshot),
        "boundary_state": _clone(boundary_state),
    }


def _validate_rng_container(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name}必须是dict。")
    for required in ("python_random_state", "numpy_random_state", "torch_cpu_rng_state"):
        if required not in value:
            raise ValueError(f"{name}缺少{required}。")


def _build_pre_payload(
    *,
    metadata: dict[str, Any],
    policy: Any,
    optimizer: Any,
    rng_states: dict[str, Any],
    cf_sampling_rng_state: dict[str, Any] | None,
) -> dict[str, Any]:
    _validate_rng_container(rng_states, "pre_update.rng_states")
    return {
        **_clone(metadata),
        "theta_pre": _clone(policy.state_dict()),
        "optimizer_state_pre": _clone(optimizer.state_dict()),
        "rng_states": _clone(rng_states),
        "cf_sampling_rng_state": _clone(cf_sampling_rng_state),
        "artifact_status": "pre_update_ready",
    }


def _validate_pre_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("pre_update payload必须是dict。")
    required = {
        "schema",
        "causal_artifact_schema",
        "update_index",
        "batch_id",
        "completed_episodes",
        "collection_policy_version",
        "policy_version_pre",
        "optimizer_step_pre",
        "pending_cf_count",
        "pending_reference_count",
        "algorithm_schema",
        "architecture_id",
        "checkpoint_schema",
        "config_fingerprint",
        "relevant_training_config_snapshot",
        "boundary_state",
        "theta_pre",
        "optimizer_state_pre",
        "rng_states",
        "cf_sampling_rng_state",
        "artifact_status",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"pre_update缺少字段：{missing}")
    if payload["schema"] != f"{CAUSAL_ARTIFACT_SCHEMA_ID}.pre_update_v1":
        raise ValueError("pre_update schema不兼容。")
    if payload["causal_artifact_schema"] != CAUSAL_ARTIFACT_SCHEMA_ID:
        raise ValueError("pre_update causal_artifact_schema不兼容。")
    _require_int(payload["update_index"], "pre_update.update_index", minimum=1)
    _require_int(payload["batch_id"], "pre_update.batch_id", minimum=1)
    _require_int(payload["completed_episodes"], "pre_update.completed_episodes")
    _require_int(payload["pending_cf_count"], "pre_update.pending_cf_count")
    _require_int(payload["pending_reference_count"], "pre_update.pending_reference_count")
    if not isinstance(payload["theta_pre"], dict) or not isinstance(payload["optimizer_state_pre"], dict):
        raise ValueError("pre_update theta/optimizer state必须是dict。")
    _validate_rng_container(payload["rng_states"], "pre_update.rng_states")
    if payload["artifact_status"] != "pre_update_ready":
        raise ValueError("pre_update artifact_status不正确。")
    return payload


def _build_boundary_rng_payload(
    *,
    update_index: int,
    batch_id: int,
    boundary_state: dict[str, Any],
    rng_states: dict[str, Any],
    cf_sampling_rng_state: dict[str, Any] | None,
) -> dict[str, Any]:
    _validate_rng_container(rng_states, "boundary_rng.rng_states")
    return {
        "schema": f"{CAUSAL_ARTIFACT_SCHEMA_ID}.boundary_rng_v1",
        "causal_artifact_schema": CAUSAL_ARTIFACT_SCHEMA_ID,
        "update_index": int(update_index),
        "batch_id": int(batch_id),
        "boundary_state": _clone(boundary_state),
        "rng_states": _clone(rng_states),
        "cf_sampling_rng_state": _clone(cf_sampling_rng_state),
    }


def _validate_boundary_rng_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("boundary_rng payload必须是dict。")
    required = {
        "schema",
        "causal_artifact_schema",
        "update_index",
        "batch_id",
        "boundary_state",
        "rng_states",
        "cf_sampling_rng_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"boundary_rng缺少字段：{missing}")
    if payload["schema"] != f"{CAUSAL_ARTIFACT_SCHEMA_ID}.boundary_rng_v1":
        raise ValueError("boundary_rng schema不兼容。")
    _require_int(payload["update_index"], "boundary_rng.update_index", minimum=1)
    _validate_rng_container(payload["rng_states"], "boundary_rng.rng_states")
    return payload


def _build_post_payload(
    *,
    update_index: int,
    batch_id: int,
    policy_version_post: int,
    optimizer_step_post: int,
    policy: Any,
    optimizer: Any,
    rng_states: dict[str, Any],
    cf_sampling_rng_state: dict[str, Any] | None,
    update_summary: Any,
    critic_step_count: int,
) -> dict[str, Any]:
    _validate_rng_container(rng_states, "post_update.rng_states")
    summary = _to_builtin(update_summary)
    if not isinstance(summary, dict):
        raise ValueError("post_update update_summary必须是dict。")
    raw_drift = {
        key: summary.get(key)
        for key in (
            "reference_raw_drift_mean",
            "reference_raw_drift_median",
            "reference_raw_drift_p95",
            "reference_raw_drift_p99",
            "reference_raw_drift_max",
        )
    }
    constrained_drift = {
        key: summary.get(key)
        for key in (
            "reference_constrained_drift_mean",
            "reference_constrained_drift_median",
            "reference_constrained_drift_p95",
            "reference_constrained_drift_p99",
            "reference_constrained_drift_max",
        )
    }
    raw_vshift_sum = summary.get(
        "direct_cf_raw_vshift_sum", summary.get("cf_batch_vshift_sum")
    )
    raw_vshift_mean = summary.get(
        "direct_cf_raw_vshift_mean", summary.get("cf_batch_vshift_mean")
    )
    raw_vshift_positive_fraction = summary.get(
        "direct_cf_raw_vshift_positive_fraction",
        summary.get("cf_batch_vshift_positive_fraction"),
    )
    constrained_vshift_sum = summary.get(
        "direct_cf_constrained_vshift_sum", summary.get("cf_batch_vshift_sum")
    )
    constrained_vshift_mean = summary.get(
        "direct_cf_constrained_vshift_mean", summary.get("cf_batch_vshift_mean")
    )
    constrained_vshift_positive_fraction = summary.get(
        "direct_cf_constrained_vshift_positive_fraction",
        summary.get("cf_batch_vshift_positive_fraction"),
    )
    return {
        "schema": f"{CAUSAL_ARTIFACT_SCHEMA_ID}.post_update_v1",
        "causal_artifact_schema": CAUSAL_ARTIFACT_SCHEMA_ID,
        "update_index": int(update_index),
        "batch_id": int(batch_id),
        "policy_version_post": int(policy_version_post),
        "optimizer_step_post": int(optimizer_step_post),
        "theta_post": _clone(policy.state_dict()),
        "optimizer_state_post": _clone(optimizer.state_dict()),
        "rng_states": _clone(rng_states),
        "cf_sampling_rng_state": _clone(cf_sampling_rng_state),
        "alpha": summary.get("reference_drift_alpha"),
        "raw_drift_stats": raw_drift,
        "constrained_drift_stats": constrained_drift,
        "gate_h": {
            "actual_error": summary.get("old_p_reconstruction_max_error"),
            "dtype": summary.get("probability_dtype"),
            "tolerance": summary.get("old_p_reconstruction_tolerance"),
        },
        "vshift": {
            "raw_sum": raw_vshift_sum,
            "raw_mean": raw_vshift_mean,
            "raw_positive_fraction": raw_vshift_positive_fraction,
            "constrained_sum": constrained_vshift_sum,
            "constrained_mean": constrained_vshift_mean,
            "constrained_positive_fraction": constrained_vshift_positive_fraction,
            "retention_ratio": summary.get("direct_cf_vshift_retention_ratio"),
        },
        "gradient_norm": summary.get("gradient_norm_mean"),
        "gradient_finite": summary.get("gradient_finite"),
        "actor_gradient_finite": summary.get("actor_gradient_finite"),
        "value_head_delta": summary.get("value_head_parameter_delta_norm"),
        "critic_step_count": int(critic_step_count),
        "update_summary": summary,
        "update_success": True,
        "artifact_status": "post_update_ready",
    }


def _validate_post_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("post_update payload必须是dict。")
    required = {
        "schema",
        "causal_artifact_schema",
        "update_index",
        "batch_id",
        "policy_version_post",
        "optimizer_step_post",
        "theta_post",
        "optimizer_state_post",
        "rng_states",
        "cf_sampling_rng_state",
        "alpha",
        "raw_drift_stats",
        "constrained_drift_stats",
        "gate_h",
        "vshift",
        "gradient_norm",
        "gradient_finite",
        "actor_gradient_finite",
        "value_head_delta",
        "critic_step_count",
        "update_summary",
        "update_success",
        "artifact_status",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"post_update缺少字段：{missing}")
    if payload["schema"] != f"{CAUSAL_ARTIFACT_SCHEMA_ID}.post_update_v1":
        raise ValueError("post_update schema不兼容。")
    if payload["causal_artifact_schema"] != CAUSAL_ARTIFACT_SCHEMA_ID:
        raise ValueError("post_update causal_artifact_schema不兼容。")
    _require_int(payload["update_index"], "post_update.update_index", minimum=1)
    _require_int(payload["batch_id"], "post_update.batch_id", minimum=1)
    _require_int(payload["critic_step_count"], "post_update.critic_step_count")
    if not isinstance(payload["theta_post"], dict) or not isinstance(payload["optimizer_state_post"], dict):
        raise ValueError("post_update theta/optimizer state必须是dict。")
    _validate_rng_container(payload["rng_states"], "post_update.rng_states")
    if payload["artifact_status"] != "post_update_ready" or payload["update_success"] is not True:
        raise ValueError("post_update success/status不正确。")
    return payload


class UpdateArtifactHandle:
    """Opaque handle returned after the pre-update files are durable."""

    def __init__(self, path: Path, update_index: int, batch_id: int) -> None:
        self.path = Path(path)
        self.update_index = int(update_index)
        self.batch_id = int(batch_id)


def update_artifact_path(run_dir: str | Path, update_index: int) -> Path:
    index = _require_int(update_index, "update_index", minimum=1)
    return Path(run_dir) / CAUSAL_ARTIFACT_DIRNAME / f"update_{index:04d}"


def assert_next_update_index_available(run_dir: str | Path, update_index: int) -> None:
    """Reject gaps, duplicates, and silent overwrite before a formal update."""

    index = _require_int(update_index, "update_index", minimum=1)
    base = Path(run_dir) / CAUSAL_ARTIFACT_DIRNAME
    existing = list_update_artifact_indices(run_dir)
    expected_previous = list(range(1, index))
    legacy_updates = set(load_legacy_artifact_gap(run_dir)["update_indices"])
    if not legacy_updates.issubset(set(expected_previous)):
        raise RuntimeError(
            "FORMAL_ARTIFACT_PERSISTENCE_FAIL: legacy update gap包含当前update之后的索引："
            f"legacy={sorted(legacy_updates)}, expected_previous={expected_previous}"
        )
    expected_available = [item for item in expected_previous if item not in legacy_updates]
    if existing != expected_available:
        raise RuntimeError(
            "FORMAL_ARTIFACT_PERSISTENCE_FAIL: update artifact index不连续："
            f"existing={existing}, expected_available={expected_available}, "
            f"legacy_missing={sorted(legacy_updates)}"
        )
    target = base / f"update_{index:04d}"
    if target.exists():
        raise FileExistsError(
            f"FORMAL_ARTIFACT_PERSISTENCE_FAIL: 禁止覆盖已有update artifact：{target}"
        )


def persist_pre_update_artifacts(
    *,
    run_dir: str | Path,
    update_index: int,
    batch_id: int,
    completed_episodes: int,
    collection_policy_version: int,
    policy_version_pre: int,
    optimizer_step_pre: int,
    policy: Any,
    optimizer: Any,
    rng_states: dict[str, Any],
    cf_sampling_rng_state: dict[str, Any] | None,
    samples: Iterable[ppo_rollout.DirectCFCreditSample],
    training_samples: Iterable[ppo_rollout.DirectCFCreditSample] | None = None,
    reference_records: Iterable[ppo_rollout.ReferenceStateRecord] | None,
    pending_reference_count: int,
    reference_drift_cap: float | None,
    algorithm_schema: str,
    architecture_id: str,
    checkpoint_schema: int,
    compatibility_snapshot: dict[str, Any],
    boundary_state: dict[str, Any],
    expected_sample_count: int | None = None,
) -> UpdateArtifactHandle:
    index = _require_int(update_index, "update_index", minimum=1)
    _require_int(batch_id, "batch_id", minimum=1)
    _require_int(completed_episodes, "completed_episodes")
    _require_int(pending_reference_count, "pending_reference_count")
    assert_next_update_index_available(run_dir, index)
    target = update_artifact_path(run_dir, index)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    try:
        # Materialize once so a caller may provide either a list or an iterable
        # without losing the samples while computing the persisted count.
        sample_list = list(samples)
        pre_metadata = _common_pre_metadata(
            update_index=index,
            batch_id=int(batch_id),
            completed_episodes=int(completed_episodes),
            collection_policy_version=int(collection_policy_version),
            policy_version_pre=int(policy_version_pre),
            optimizer_step_pre=int(optimizer_step_pre),
            pending_cf_count=len(sample_list),
            pending_reference_count=int(pending_reference_count),
            reference_drift_cap=reference_drift_cap,
            algorithm_schema=algorithm_schema,
            architecture_id=architecture_id,
            checkpoint_schema=int(checkpoint_schema),
            compatibility_snapshot=compatibility_snapshot,
            boundary_state=boundary_state,
        )
        pre_metadata["pending_cf_count"] = len(sample_list)
        pre_metadata["pending_cf_sample_count"] = len(sample_list)
        pre_payload = _build_pre_payload(
            metadata=pre_metadata,
            policy=policy,
            optimizer=optimizer,
            rng_states=rng_states,
            cf_sampling_rng_state=cf_sampling_rng_state,
        )
        cf_payload = _build_cf_batch_payload(
            update_index=index,
            batch_id=int(batch_id),
            samples=sample_list,
            training_samples=training_samples,
            reference_records=reference_records,
            expected_sample_count=expected_sample_count,
        )
        boundary_payload = _build_boundary_rng_payload(
            update_index=index,
            batch_id=int(batch_id),
            boundary_state=boundary_state,
            rng_states=rng_states,
            cf_sampling_rng_state=cf_sampling_rng_state,
        )
        _atomic_torch_save(pre_payload, target / PRE_UPDATE_FILENAME)
        _atomic_torch_save(cf_payload, target / CF_BATCH_FILENAME)
        _atomic_torch_save(boundary_payload, target / BOUNDARY_RNG_FILENAME)
        _validate_pre_payload(_torch_load(target / PRE_UPDATE_FILENAME))
        _validate_cf_batch_payload(
            _torch_load(target / CF_BATCH_FILENAME),
            expected_update_index=index,
            expected_sample_count=expected_sample_count,
        )
        _validate_boundary_rng_payload(_torch_load(target / BOUNDARY_RNG_FILENAME))
        metadata = {
            "causal_artifact_schema": CAUSAL_ARTIFACT_SCHEMA_ID,
            "schema_version": CAUSAL_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "online_update",
            "artifact_status": "pre_update_ready",
            "update_index": index,
            "batch_id": int(batch_id),
            "completed_episodes": int(completed_episodes),
            "collection_policy_version": int(collection_policy_version),
            "policy_version_pre": int(policy_version_pre),
            "optimizer_step_pre": int(optimizer_step_pre),
            "algorithm_schema": str(algorithm_schema),
            "architecture_id": str(architecture_id),
            "checkpoint_schema": int(checkpoint_schema),
            "config_fingerprint": pre_metadata["config_fingerprint"],
            "sample_count": len(sample_list),
            "training_sample_count": int(cf_payload["training_sample_count"]),
            "r_all_count": int(cf_payload["r_all_count"]),
            "selected_count": int(cf_payload["selected_count"]),
            "r_ref_count": int(cf_payload["r_ref_count"]),
            "intersection_count": int(cf_payload["intersection_count"]),
            "files": {
                PRE_UPDATE_FILENAME: PRE_UPDATE_FILENAME,
                CF_BATCH_FILENAME: CF_BATCH_FILENAME,
                BOUNDARY_RNG_FILENAME: BOUNDARY_RNG_FILENAME,
                POST_UPDATE_FILENAME: POST_UPDATE_FILENAME,
            },
            "sha256": {
                PRE_UPDATE_FILENAME: _sha256(target / PRE_UPDATE_FILENAME),
                CF_BATCH_FILENAME: _sha256(target / CF_BATCH_FILENAME),
                BOUNDARY_RNG_FILENAME: _sha256(target / BOUNDARY_RNG_FILENAME),
            },
        }
        _atomic_json_save(metadata, target / METADATA_FILENAME)
    except Exception as exc:
        raise RuntimeError(f"FORMAL_ARTIFACT_PERSISTENCE_FAIL: pre artifact写入失败：{target}") from exc
    return UpdateArtifactHandle(target, index, int(batch_id))


def finalize_post_update_artifact(
    handle: UpdateArtifactHandle,
    *,
    policy_version_post: int,
    optimizer_step_post: int,
    policy: Any,
    optimizer: Any,
    rng_states: dict[str, Any],
    cf_sampling_rng_state: dict[str, Any] | None,
    update_summary: Any,
    critic_step_count: int,
) -> None:
    target = Path(handle.path)
    metadata_path = target / METADATA_FILENAME
    if not metadata_path.exists():
        raise RuntimeError("FORMAL_ARTIFACT_PERSISTENCE_FAIL: pre metadata不存在。")
    if (target / POST_UPDATE_FILENAME).exists():
        raise FileExistsError(
            f"FORMAL_ARTIFACT_PERSISTENCE_FAIL: 禁止覆盖已有post artifact：{target / POST_UPDATE_FILENAME}"
        )
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("artifact_status") != "pre_update_ready":
        raise RuntimeError("FORMAL_ARTIFACT_PERSISTENCE_FAIL: pre metadata状态不正确。")
    post_payload = _build_post_payload(
        update_index=handle.update_index,
        batch_id=handle.batch_id,
        policy_version_post=int(policy_version_post),
        optimizer_step_post=int(optimizer_step_post),
        policy=policy,
        optimizer=optimizer,
        rng_states=rng_states,
        cf_sampling_rng_state=cf_sampling_rng_state,
        update_summary=update_summary,
        critic_step_count=int(critic_step_count),
    )
    try:
        _atomic_torch_save(post_payload, target / POST_UPDATE_FILENAME)
        _validate_post_payload(_torch_load(target / POST_UPDATE_FILENAME))
        metadata = dict(metadata)
        metadata["artifact_status"] = "complete"
        metadata["policy_version_post"] = int(policy_version_post)
        metadata["optimizer_step_post"] = int(optimizer_step_post)
        metadata["sha256"] = dict(metadata.get("sha256", {}))
        metadata["sha256"][POST_UPDATE_FILENAME] = _sha256(target / POST_UPDATE_FILENAME)
        _atomic_json_save(metadata, metadata_path)
        validate_update_artifact(target, require_complete=True)
    except Exception as exc:
        raise RuntimeError(f"FORMAL_ARTIFACT_PERSISTENCE_FAIL: post artifact写入失败：{target}") from exc


def load_pre_update_artifact(path: str | Path) -> dict[str, Any]:
    return _validate_pre_payload(_torch_load(Path(path) / PRE_UPDATE_FILENAME))


def load_post_update_artifact(path: str | Path) -> dict[str, Any]:
    return _validate_post_payload(_torch_load(Path(path) / POST_UPDATE_FILENAME))


def load_cf_batch_artifact(path: str | Path) -> dict[str, Any]:
    return _validate_cf_batch_payload(_torch_load(Path(path) / CF_BATCH_FILENAME))


def validate_update_artifact(
    path: str | Path,
    *,
    require_complete: bool = True,
    expected_update_index: int | None = None,
) -> dict[str, Any]:
    target = Path(path)
    metadata_path = target / METADATA_FILENAME
    if not target.is_dir() or not metadata_path.exists():
        raise ValueError(f"causal update artifact目录/metadata不存在：{target}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("causal_artifact_schema") != CAUSAL_ARTIFACT_SCHEMA_ID:
        raise ValueError("causal update metadata schema不兼容。")
    if int(metadata.get("schema_version", -1)) != CAUSAL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("causal update metadata schema_version不兼容。")
    index = _require_int(metadata.get("update_index", -1), "metadata.update_index", minimum=1)
    if expected_update_index is not None and index != int(expected_update_index):
        raise ValueError("causal update metadata update_index不一致。")
    if require_complete and metadata.get("artifact_status") != "complete":
        raise ValueError("causal update artifact尚未complete。")
    for filename in (PRE_UPDATE_FILENAME, CF_BATCH_FILENAME, BOUNDARY_RNG_FILENAME):
        file_path = target / filename
        if not file_path.exists():
            raise ValueError(f"causal update缺少{filename}。")
    if require_complete and not (target / POST_UPDATE_FILENAME).exists():
        raise ValueError("causal update缺少post_update.pt。")
    hashes = metadata.get("sha256", {})
    for filename, digest in hashes.items():
        file_path = target / filename
        if file_path.exists() and _sha256(file_path) != digest:
            raise ValueError(f"causal update {filename} sha256不一致。")
    pre = load_pre_update_artifact(target)
    cf = load_cf_batch_artifact(target)
    boundary = _validate_boundary_rng_payload(_torch_load(target / BOUNDARY_RNG_FILENAME))
    result: dict[str, Any] = {
        "metadata": metadata,
        "pre_update": pre,
        "cf_batch": cf,
        "boundary_rng": boundary,
    }
    if require_complete:
        result["post_update"] = load_post_update_artifact(target)
    return result


def list_update_artifact_indices(run_dir: str | Path) -> list[int]:
    base = Path(run_dir) / CAUSAL_ARTIFACT_DIRNAME
    if not base.exists():
        return []
    if not base.is_dir():
        raise ValueError(f"causal artifact path不是目录：{base}")
    indices: list[int] = []
    for child in base.iterdir():
        if not child.is_dir():
            raise ValueError(f"causal artifact目录包含非目录项：{child}")
        match = _UPDATE_DIRECTORY_RE.fullmatch(child.name)
        if match is None:
            raise ValueError(f"causal artifact目录包含未知目录：{child.name}")
        indices.append(int(match.group(1)))
    return sorted(indices)


def legacy_artifact_gap_path(run_dir: str | Path) -> Path:
    """Return the explicit migration manifest path for unavailable legacy artifacts."""

    return Path(run_dir) / LEGACY_ARTIFACT_GAP_FILENAME


def load_legacy_artifact_gap(run_dir: str | Path) -> dict[str, list[int]]:
    """Load the explicit legacy-artifact gap manifest, if a migration created one.

    The manifest is metadata only.  It never stands in for an update or
    episode-start artifact and is accepted only for the exact indices it
    declares as unavailable.  This lets a schema migration preserve an old
    run without manufacturing historical causal files, while keeping future
    index-gap and overwrite checks strict.
    """

    path = legacy_artifact_gap_path(run_dir)
    if not path.exists():
        return {"update_indices": [], "episode_indices": []}
    if not path.is_file():
        raise ValueError(f"legacy artifact gap path不是文件：{path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"legacy artifact gap manifest无法读取：{path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("legacy artifact gap manifest必须是dict。")
    if payload.get("schema") != LEGACY_ARTIFACT_GAP_SCHEMA_ID:
        raise ValueError("legacy artifact gap manifest schema不兼容。")
    result: dict[str, list[int]] = {}
    for field_name in ("update_indices", "episode_indices"):
        values = payload.get(field_name)
        if not isinstance(values, list):
            raise ValueError(f"legacy artifact gap {field_name}必须是list。")
        normalized = [_require_int(item, f"legacy artifact gap {field_name}", minimum=1) for item in values]
        if normalized != sorted(set(normalized)):
            raise ValueError(f"legacy artifact gap {field_name}必须严格升序且无重复。")
        result[field_name] = normalized
    return result


def _build_episode_start_payload(
    *,
    episode: int,
    policy_version: int,
    scenario_reset_identifier: dict[str, Any],
    rng_states: dict[str, Any],
    cf_sampling_rng_state: dict[str, Any] | None,
    environment_reset_rng: Any,
    reset_semantics: str,
    initial_dynamic_cache: Any,
    initial_uav_state: dict[str, Any],
    tv_local_cache_state: dict[str, Any],
) -> dict[str, Any]:
    _validate_rng_container(rng_states, "episode_start.rng_states")
    return {
        "schema": EPISODE_START_SCHEMA_ID,
        "causal_artifact_schema": CAUSAL_ARTIFACT_SCHEMA_ID,
        "episode": _require_int(episode, "episode", minimum=1),
        "policy_version": _require_int(policy_version, "policy_version"),
        "scenario_reset_identifier": _clone(scenario_reset_identifier),
        "rng_states": _clone(rng_states),
        "cf_sampling_rng_state": _clone(cf_sampling_rng_state),
        "environment_reset_rng": _clone(environment_reset_rng),
        "reset_semantics": str(reset_semantics),
        "initial_dynamic_cache": _clone(initial_dynamic_cache),
        "initial_uav_state": _clone(initial_uav_state),
        "tv_local_cache_state": _clone(tv_local_cache_state),
    }


def _validate_episode_start_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("episode_start payload必须是dict。")
    required = {
        "schema",
        "causal_artifact_schema",
        "episode",
        "policy_version",
        "scenario_reset_identifier",
        "rng_states",
        "cf_sampling_rng_state",
        "environment_reset_rng",
        "reset_semantics",
        "initial_dynamic_cache",
        "initial_uav_state",
        "tv_local_cache_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"episode_start缺少字段：{missing}")
    if payload["schema"] != EPISODE_START_SCHEMA_ID:
        raise ValueError("episode_start schema不兼容。")
    if payload["causal_artifact_schema"] != CAUSAL_ARTIFACT_SCHEMA_ID:
        raise ValueError("episode_start causal_artifact_schema不兼容。")
    _require_int(payload["episode"], "episode_start.episode", minimum=1)
    _require_int(payload["policy_version"], "episode_start.policy_version")
    _validate_rng_container(payload["rng_states"], "episode_start.rng_states")
    if not isinstance(payload["scenario_reset_identifier"], dict):
        raise ValueError("episode_start scenario_reset_identifier必须是dict。")
    return payload


def persist_episode_start_artifact(
    *,
    run_dir: str | Path,
    episode: int,
    policy_version: int,
    scenario_reset_identifier: dict[str, Any],
    rng_states: dict[str, Any],
    cf_sampling_rng_state: dict[str, Any] | None,
    environment_reset_rng: Any,
    reset_semantics: str,
    initial_dynamic_cache: Any,
    initial_uav_state: dict[str, Any],
    tv_local_cache_state: dict[str, Any],
) -> Path:
    episode_number = _require_int(episode, "episode", minimum=1)
    base = Path(run_dir) / EPISODE_START_DIRNAME
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"episode_{episode_number:04d}.pt"
    if target.exists():
        raise FileExistsError(
            f"FORMAL_ARTIFACT_PERSISTENCE_FAIL: 禁止覆盖已有episode start artifact：{target}"
        )
    payload = _build_episode_start_payload(
        episode=episode_number,
        policy_version=policy_version,
        scenario_reset_identifier=scenario_reset_identifier,
        rng_states=rng_states,
        cf_sampling_rng_state=cf_sampling_rng_state,
        environment_reset_rng=environment_reset_rng,
        reset_semantics=reset_semantics,
        initial_dynamic_cache=initial_dynamic_cache,
        initial_uav_state=initial_uav_state,
        tv_local_cache_state=tv_local_cache_state,
    )
    try:
        _atomic_torch_save(payload, target)
        _validate_episode_start_payload(_torch_load(target))
    except Exception as exc:
        raise RuntimeError(f"FORMAL_ARTIFACT_PERSISTENCE_FAIL: episode start写入失败：{target}") from exc
    return target


def load_episode_start_artifact(path: str | Path) -> dict[str, Any]:
    return _validate_episode_start_payload(_torch_load(Path(path)))


def _episode_start_indices(run_dir: str | Path) -> list[int]:
    base = Path(run_dir) / EPISODE_START_DIRNAME
    if not base.exists():
        return []
    if not base.is_dir():
        raise ValueError(f"episode start artifact path不是目录：{base}")
    result: list[int] = []
    for child in base.iterdir():
        if not child.is_file():
            raise ValueError(f"episode start目录包含非文件项：{child}")
        match = _EPISODE_START_RE.fullmatch(child.name)
        if match is None:
            raise ValueError(f"episode start目录包含未知文件：{child.name}")
        result.append(int(match.group(1)))
    return sorted(result)


def validate_resume_artifact_index(
    run_dir: str | Path,
    *,
    completed_batch_count: int,
    completed_episodes: int,
) -> dict[str, Any]:
    """Validate all prior update and episode-start artifacts before resume."""

    batches = _require_int(completed_batch_count, "completed_batch_count")
    episodes = _require_int(completed_episodes, "completed_episodes")
    legacy_gap = load_legacy_artifact_gap(run_dir)
    legacy_updates = set(legacy_gap["update_indices"])
    legacy_episodes = set(legacy_gap["episode_indices"])
    update_indices = list_update_artifact_indices(run_dir)
    expected_updates = list(range(1, batches + 1))
    if not legacy_updates.issubset(set(expected_updates)):
        raise RuntimeError(
            "FORMAL_ARTIFACT_PERSISTENCE_FAIL: legacy update gap超出completed batch范围："
            f"legacy={sorted(legacy_updates)}, expected={expected_updates}"
        )
    expected_available_updates = [item for item in expected_updates if item not in legacy_updates]
    if update_indices != expected_available_updates:
        raise RuntimeError(
            "FORMAL_ARTIFACT_PERSISTENCE_FAIL: resume update artifact index不一致："
            f"actual={update_indices}, expected_available={expected_available_updates}, "
            f"legacy_missing={sorted(legacy_updates)}"
        )
    for index in update_indices:
        validate_update_artifact(
            update_artifact_path(run_dir, index),
            require_complete=True,
            expected_update_index=index,
        )
    episode_indices = _episode_start_indices(run_dir)
    expected_episodes = list(range(1, episodes + 1))
    if not legacy_episodes.issubset(set(expected_episodes)):
        raise RuntimeError(
            "FORMAL_ARTIFACT_PERSISTENCE_FAIL: legacy episode-start gap超出completed episode范围："
            f"legacy={sorted(legacy_episodes)}, expected={expected_episodes}"
        )
    expected_available_episodes = [item for item in expected_episodes if item not in legacy_episodes]
    if episode_indices != expected_available_episodes:
        raise RuntimeError(
            "FORMAL_ARTIFACT_PERSISTENCE_FAIL: resume episode-start index不一致："
            f"actual={episode_indices}, expected_available={expected_available_episodes}, "
            f"legacy_missing={sorted(legacy_episodes)}"
        )
    for episode in episode_indices:
        load_episode_start_artifact(
            Path(run_dir) / EPISODE_START_DIRNAME / f"episode_{episode:04d}.pt"
        )
    return {
        "update_indices": update_indices,
        "episode_indices": episode_indices,
        "legacy_missing_update_indices": sorted(legacy_updates),
        "legacy_missing_episode_indices": sorted(legacy_episodes),
        "update_index_pass": True,
        "episode_start_index_pass": True,
    }


def values_equal(left: Any, right: Any) -> bool:
    """Exact recursive equality for fixture roundtrip gates."""

    if torch is not None and isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and bool(torch.equal(left, right))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape and bool(np.array_equal(left, right))
    if isinstance(left, dict):
        return isinstance(right, dict) and list(left.keys()) == list(right.keys()) and all(
            values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            values_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, float):
        return isinstance(right, float) and left == right
    return left == right


def persist_hierarchical_gate_update_artifact(
    path: str | Path,
    *,
    update_index: int,
    collection_policy_version: int,
    gate_sample: Any | None = None,
    gate_samples: Iterable[Any] | None = None,
    gate_parameters_pre: dict[str, Any],
    gate_parameters_post: dict[str, Any],
    gate_optimizer_state_pre: dict[str, Any],
    gate_optimizer_state_post: dict[str, Any],
    compatibility_snapshot: dict[str, Any] | None = None,
    algorithm_schema: str = (
        "proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift_v1"
    ),
    crn_metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist one standalone Gate update using the causal-artifact writer.

    This extension is intentionally independent from ``update_XXXX`` legacy
    directories.  It stores both Gate parameter/Adam states and the complete
    Joint-Null sample, so a later caller can reconstruct the Gate update
    without an environment replay or a new random draw.
    """

    _require_torch()
    from ppo import hierarchical_reconfiguration_gate

    index = _require_int(update_index, "hierarchical_gate.update_index", minimum=1)
    version = _require_int(
        collection_policy_version,
        "hierarchical_gate.collection_policy_version",
        minimum=0,
    )
    if not isinstance(gate_parameters_pre, dict) or not isinstance(gate_parameters_post, dict):
        raise ValueError("hierarchical Gate parameter states必须是dict。")
    if not isinstance(gate_optimizer_state_pre, dict) or not isinstance(gate_optimizer_state_post, dict):
        raise ValueError("hierarchical Gate optimizer states必须是dict。")
    if gate_samples is None:
        if gate_sample is None:
            raise ValueError("hierarchical Gate artifact至少需要一个gate sample。")
        sample_list = [gate_sample]
    else:
        sample_list = list(gate_samples)
        if gate_sample is not None:
            if not sample_list:
                sample_list = [gate_sample]
            elif sample_list[0] is not gate_sample:
                # Keep the old keyword as a compatibility alias, but make
                # the first persisted sample canonical when both are passed.
                sample_list = [gate_sample, *sample_list]
    if not sample_list:
        raise ValueError("hierarchical Gate artifact至少需要一个gate sample。")
    sample_dicts = [
        hierarchical_reconfiguration_gate.gate_counterfactual_sample_to_dict(item)
        for item in sample_list
    ]
    sample_dict = sample_dicts[0]
    sample_versions = {int(item["collection_policy_version"]) for item in sample_dicts}
    if sample_versions != {version}:
        raise ValueError(
            "hierarchical Gate artifact samples的collection_policy_version必须一致且匹配artifact。"
        )
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"FORMAL_ARTIFACT_PERSISTENCE_FAIL: 禁止覆盖已有Gate artifact：{target}"
        )
    payload = {
        "schema": HIERARCHICAL_GATE_ARTIFACT_SCHEMA_ID,
        "causal_artifact_schema": CAUSAL_ARTIFACT_SCHEMA_ID,
        "artifact_status": "complete",
        "update_index": index,
        "collection_policy_version": version,
        "algorithm_schema": str(algorithm_schema),
        "gate_sample": sample_dict,
        "gate_samples": sample_dicts,
        "gate_sample_count": len(sample_dicts),
        "gate_observation": sample_dict["gate_observation"],
        "gate_action": sample_dict["gate_action"],
        "p_old_gate": sample_dict["p_old_gate"],
        "gate_old_logprob": sample_dict["gate_old_logprob"],
        "config_joint_action": sample_dict["config_joint_action"],
        "null_joint_action": sample_dict["null_joint_action"],
        "Return_CONFIG": sample_dict["Return_CONFIG"],
        "Return_NULL": sample_dict["Return_NULL"],
        "Delta_gate": sample_dict["Delta_gate"],
        "A_gate": sample_dict["A_gate"],
        "DeltaQ": sample_dict["DeltaQ"],
        "DeltaCache": sample_dict["DeltaCache"],
        "DeltaUAV": sample_dict["DeltaUAV"],
        "crn_metadata": copy.deepcopy(
            sample_dict["crn_metadata"] if crn_metadata is None else crn_metadata
        ),
        "gate_parameters_pre": _clone(gate_parameters_pre),
        "gate_parameters_post": _clone(gate_parameters_post),
        "gate_optimizer_state_pre": _clone(gate_optimizer_state_pre),
        "gate_optimizer_state_post": _clone(gate_optimizer_state_post),
        "compatibility_snapshot": _clone(compatibility_snapshot or {}),
    }
    # Validate/normalize before the atomic write.  _to_builtin is also the
    # persistence-side finite-number guard and does not touch any RNG.
    _to_builtin(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        _atomic_torch_save(payload, target / HIERARCHICAL_GATE_ARTIFACT_FILENAME)
        validate_hierarchical_gate_update_artifact(target)
        metadata = {
            "schema": HIERARCHICAL_GATE_ARTIFACT_SCHEMA_ID,
            "causal_artifact_schema": CAUSAL_ARTIFACT_SCHEMA_ID,
            "artifact_status": "complete",
            "update_index": index,
            "collection_policy_version": version,
            "algorithm_schema": str(algorithm_schema),
            "gate_sample_decision_id": str(sample_dict["decision_id"]),
            "gate_sample_count": len(sample_dicts),
            "files": {HIERARCHICAL_GATE_ARTIFACT_FILENAME: HIERARCHICAL_GATE_ARTIFACT_FILENAME},
            "sha256": {
                HIERARCHICAL_GATE_ARTIFACT_FILENAME: _sha256(
                    target / HIERARCHICAL_GATE_ARTIFACT_FILENAME
                )
            },
        }
        _atomic_json_save(metadata, target / "metadata.json")
    except Exception as exc:
        raise RuntimeError(
            f"FORMAL_ARTIFACT_PERSISTENCE_FAIL: hierarchical Gate artifact写入失败：{target}"
        ) from exc
    return target


def validate_hierarchical_gate_update_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate the standalone Gate artifact without random calls."""

    _require_torch()
    from ppo import hierarchical_reconfiguration_gate

    target = Path(path)
    payload_path = target / HIERARCHICAL_GATE_ARTIFACT_FILENAME
    if not target.is_dir() or not payload_path.exists():
        raise ValueError(f"hierarchical Gate artifact不存在：{target}")
    payload = _torch_load(payload_path)
    if not isinstance(payload, dict):
        raise ValueError("hierarchical Gate artifact顶层必须是dict。")
    required = {
        "schema",
        "causal_artifact_schema",
        "artifact_status",
        "update_index",
        "collection_policy_version",
        "algorithm_schema",
        "gate_sample",
        "gate_parameters_pre",
        "gate_parameters_post",
        "gate_optimizer_state_pre",
        "gate_optimizer_state_post",
        "compatibility_snapshot",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"hierarchical Gate artifact缺少字段：{missing}")
    if payload["schema"] != HIERARCHICAL_GATE_ARTIFACT_SCHEMA_ID:
        raise ValueError("hierarchical Gate artifact schema不兼容。")
    if payload["causal_artifact_schema"] != CAUSAL_ARTIFACT_SCHEMA_ID:
        raise ValueError("hierarchical Gate artifact causal schema不兼容。")
    if payload["artifact_status"] != "complete":
        raise ValueError("hierarchical Gate artifact状态不正确。")
    _require_int(payload["update_index"], "hierarchical_gate.update_index", minimum=1)
    _require_int(
        payload["collection_policy_version"],
        "hierarchical_gate.collection_policy_version",
        minimum=0,
    )
    if not isinstance(payload["gate_parameters_pre"], dict) or not isinstance(
        payload["gate_parameters_post"], dict
    ):
        raise ValueError("hierarchical Gate parameter states必须是dict。")
    if not isinstance(payload["gate_optimizer_state_pre"], dict) or not isinstance(
        payload["gate_optimizer_state_post"], dict
    ):
        raise ValueError("hierarchical Gate optimizer states必须是dict。")
    raw_samples = payload.get("gate_samples")
    # Early module-level fixture artifacts carried only the aliased first
    # sample.  Promote that legacy shape in memory; the formal writer always
    # persists the complete list and count for a Batch4 update.
    if raw_samples is None:
        raw_samples = [payload["gate_sample"]]
    if "gate_sample_count" not in payload:
        payload["gate_sample_count"] = len(raw_samples)
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("hierarchical Gate artifact gate_samples必须是非空list。")
    if int(payload.get("gate_sample_count", -1)) != len(raw_samples):
        raise ValueError("hierarchical Gate artifact gate_sample_count不一致。")
    samples = [
        hierarchical_reconfiguration_gate.gate_counterfactual_sample_from_dict(item)
        for item in raw_samples
    ]
    sample = samples[0]
    if payload.get("gate_sample") != raw_samples[0]:
        raise ValueError("hierarchical Gate artifact gate_sample与gate_samples[0]不一致。")
    if any(
        int(item.collection_policy_version) != int(payload["collection_policy_version"])
        for item in samples
    ):
        raise ValueError("hierarchical Gate samples与artifact collection_policy_version不一致。")
    required_redundant = (
        "gate_observation",
        "gate_action",
        "p_old_gate",
        "gate_old_logprob",
        "config_joint_action",
        "null_joint_action",
        "Return_CONFIG",
        "Return_NULL",
        "Delta_gate",
        "A_gate",
        "crn_metadata",
    )
    for name in required_redundant:
        if name not in payload:
            raise ValueError(f"hierarchical Gate artifact缺少冗余字段：{name}")
    canonical = hierarchical_reconfiguration_gate.gate_counterfactual_sample_to_dict(sample)
    for name in required_redundant[:-1]:
        if payload[name] != canonical[name]:
            raise ValueError(f"hierarchical Gate artifact字段不一致：{name}")
    for name in ("DeltaQ", "DeltaCache", "DeltaUAV"):
        if name in payload:
            if payload[name] != canonical[name]:
                raise ValueError(f"hierarchical Gate artifact字段不一致：{name}")
        elif canonical[name] is not None:
            raise ValueError(f"hierarchical Gate artifact缺少冗余字段：{name}")
    _to_builtin(payload)
    return payload


def load_hierarchical_gate_update_artifact(path: str | Path) -> dict[str, Any]:
    return validate_hierarchical_gate_update_artifact(path)
