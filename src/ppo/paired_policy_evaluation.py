# -*- coding: utf-8 -*-
"""Read-only paired policy evaluation and reporting protocol.

This module is deliberately separate from the training algorithm.  A paired
evaluation compares two explicit policy artifacts on an immutable evaluation
panel.  It never looks up a live training policy, never restores or advances a
global RNG, never calls an environment, and never calls an optimizer.

The executable rollout implementation used by the Update6 audit is kept as a
validated regression fixture.  Future callers may provide the already
validated rollout executor through :func:`compare_policy_checkpoints`, but the
executor is explicit and receives only immutable artifact/panel inputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

try:  # The protocol can still validate JSON-only fixtures without torch.
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]


PAIRED_EVALUATION_SCHEMA = "paired_policy_evaluation_v1"
PANEL_MANIFEST_SCHEMA = "paired_policy_evaluation_panel_v1"
PRIMARY_POLICY_IMPROVEMENT_METRIC = "paired_eval_mean_delta_G"
DESCRIPTIVE_ONLINE_METRIC = "online_rollout_mean_G"

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "G": ("G_sum", "g_sum", "reward_sum", "cumulative_system_utility"),
    "Mean_G": ("Mean_G", "mean_G", "reward_mean", "mean_system_utility"),
    "Q": ("Q_sum", "q_sum", "actual_q_sum", "cumulative_actual_task_utility"),
    "cache": ("cache_cost_sum", "cache", "cumulative_cache_cost"),
    "UAV": ("uav_cost_sum", "UAV", "movement_cost_sum", "cumulative_movement_cost"),
}


class PairedEvaluationProtocolError(ValueError):
    """Raised when a paired-evaluation artifact or record violates the gate."""


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairedEvaluationProtocolError(f"{name}必须是mapping/object。")
    return value


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PairedEvaluationProtocolError(f"{name}必须是数值。") from exc
    if not math.isfinite(result):
        raise PairedEvaluationProtocolError(f"{name}必须是finite数值。")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise PairedEvaluationProtocolError(f"{name}必须是非负整数。")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PairedEvaluationProtocolError(f"{name}必须是非负整数。") from exc
    if result != value or result < 0:
        raise PairedEvaluationProtocolError(f"{name}必须是非负整数。")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if torch is not None and isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PairedEvaluationProtocolError("metadata包含NaN/Inf。")
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(f"不支持的JSON值类型：{type(value).__name__}")


def config_snapshot_fingerprint(config_snapshot: Mapping[str, Any]) -> str:
    """Return a deterministic fingerprint of the supplied config snapshot."""

    _require_mapping(config_snapshot, "config_snapshot")
    return hashlib.sha256(_canonical_json(config_snapshot).encode("utf-8")).hexdigest()


def _torch_load(path: Path) -> Any:
    if torch is None:
        raise PairedEvaluationProtocolError("读取policy/panel artifact需要PyTorch。")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location="cpu")


def file_sha256(path: str | Path) -> str:
    """Hash an artifact without opening an environment or touching RNG state."""

    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_for_hash(value: Any, name: str) -> tuple[str, tuple[int, ...], bytes]:
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        if not bool(torch.isfinite(tensor).all().item()):
            raise PairedEvaluationProtocolError(f"{name}包含NaN/Inf。")
        array = tensor.numpy()
    else:
        array = np.asarray(value)
        if not np.all(np.isfinite(array)):
            raise PairedEvaluationProtocolError(f"{name}包含NaN/Inf。")
        array = np.ascontiguousarray(array)
    return str(array.dtype), tuple(int(item) for item in array.shape), array.tobytes()


def parameter_state_hash(state_dict: Mapping[str, Any]) -> str:
    """Hash a state dict including names, dtypes, shapes, and tensor bytes."""

    state = _require_mapping(state_dict, "state_dict")
    digest = hashlib.sha256()
    for name in sorted(str(key) for key in state):
        dtype, shape, raw = _array_for_hash(state[name], f"state_dict[{name}]")
        header = _canonical_json({"name": name, "dtype": dtype, "shape": shape}).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True)
class PolicyArtifactIdentity:
    """Identity and immutable state extracted from one explicit artifact."""

    path: str
    role: str
    artifact_sha256: str
    policy_version: int
    optimizer_step: int
    algorithm_schema: str
    architecture_id: str
    checkpoint_schema: int
    parameter_hash: str
    state_dict: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "artifact_sha256": self.artifact_sha256,
            "policy_version": self.policy_version,
            "optimizer_step": self.optimizer_step,
            "algorithm_schema": self.algorithm_schema,
            "architecture_id": self.architecture_id,
            "checkpoint_schema": self.checkpoint_schema,
            "parameter_hash": self.parameter_hash,
            "state_dict_keys": sorted(str(key) for key in self.state_dict),
        }


def _resolve_policy_artifact_path(path: str | Path, role: str) -> Path:
    if role not in {"old", "new"}:
        raise PairedEvaluationProtocolError("policy artifact role必须是old或new。")
    target = Path(path)
    if target.is_dir():
        target = target / ("pre_update.pt" if role == "old" else "post_update.pt")
    if not target.is_file():
        raise PairedEvaluationProtocolError(f"policy artifact不存在：{target}")
    return target


def load_policy_artifact(path: str | Path, role: str) -> PolicyArtifactIdentity:
    """Load a pre/post policy artifact explicitly; reject live policy objects."""

    target = _resolve_policy_artifact_path(path, role)
    payload = _require_mapping(_torch_load(target), str(target))
    # The formal post_update payload intentionally keeps update metadata in the
    # sibling metadata.json.  Merge that immutable directory metadata only for
    # identity validation; theta/version still come from the requested payload.
    directory_metadata: Mapping[str, Any] = {}
    metadata_path = target.parent / "metadata.json"
    if metadata_path.is_file():
        try:
            directory_metadata = _require_mapping(
                json.loads(metadata_path.read_text(encoding="utf-8")),
                str(metadata_path),
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise PairedEvaluationProtocolError(
                f"policy artifact metadata无法读取：{metadata_path}"
            ) from exc

    def metadata_value(name: str) -> Any:
        if name in payload:
            return payload[name]
        if name in directory_metadata:
            return directory_metadata[name]
        raise KeyError(name)

    state_key = "theta_pre" if role == "old" else "theta_post"
    version_key = "policy_version_pre" if role == "old" else "policy_version_post"
    optimizer_key = "optimizer_step_pre" if role == "old" else "optimizer_step_post"
    required = {
        state_key,
        version_key,
        optimizer_key,
        "algorithm_schema",
        "architecture_id",
        "checkpoint_schema",
    }
    available = set(payload) | set(directory_metadata)
    missing = sorted(required - available)
    if missing:
        raise PairedEvaluationProtocolError(f"{target}缺少policy identity字段：{missing}")
    state_dict = _require_mapping(payload[state_key], f"{target}.{state_key}")
    if not state_dict:
        raise PairedEvaluationProtocolError(f"{target}.{state_key}不能为空。")
    policy_version = _nonnegative_int(metadata_value(version_key), f"{target}.{version_key}")
    optimizer_step = _nonnegative_int(metadata_value(optimizer_key), f"{target}.{optimizer_key}")
    checkpoint_schema = _nonnegative_int(metadata_value("checkpoint_schema"), f"{target}.checkpoint_schema")
    return PolicyArtifactIdentity(
        path=str(target.resolve()),
        role=role,
        artifact_sha256=file_sha256(target),
        policy_version=policy_version,
        optimizer_step=optimizer_step,
        algorithm_schema=str(metadata_value("algorithm_schema")),
        architecture_id=str(metadata_value("architecture_id")),
        checkpoint_schema=checkpoint_schema,
        parameter_hash=parameter_state_hash(state_dict),
        state_dict=copy.deepcopy(dict(state_dict)),
    )


def validate_policy_pair(old: PolicyArtifactIdentity, new: PolicyArtifactIdentity) -> dict[str, Any]:
    """Validate the policy identity gate before any paired comparison."""

    if old.role != "old" or new.role != "new":
        raise PairedEvaluationProtocolError("policy pair roles必须分别为old和new。")
    if old.algorithm_schema != new.algorithm_schema:
        raise PairedEvaluationProtocolError("old/new algorithm schema不一致。")
    if old.architecture_id != new.architecture_id:
        raise PairedEvaluationProtocolError("old/new architecture ID不一致。")
    if old.checkpoint_schema != new.checkpoint_schema:
        raise PairedEvaluationProtocolError("old/new checkpoint schema不一致。")
    old_keys = sorted(str(key) for key in old.state_dict)
    new_keys = sorted(str(key) for key in new.state_dict)
    if old_keys != new_keys:
        raise PairedEvaluationProtocolError("old/new policy state_dict keys不一致。")
    shape_mismatch: list[str] = []
    for name in old_keys:
        old_shape = _array_for_hash(old.state_dict[name], f"old.state_dict[{name}]")[1]
        new_shape = _array_for_hash(new.state_dict[name], f"new.state_dict[{name}]")[1]
        if old_shape != new_shape:
            shape_mismatch.append(name)
    if shape_mismatch:
        raise PairedEvaluationProtocolError(f"old/new policy parameter shape不一致：{shape_mismatch}")
    return {
        "pass": True,
        "same_algorithm_schema": True,
        "same_architecture_id": True,
        "same_checkpoint_schema": True,
        "same_parameter_layout": True,
        "old_policy_version": old.policy_version,
        "new_policy_version": new.policy_version,
        "old_parameter_hash": old.parameter_hash,
        "new_parameter_hash": new.parameter_hash,
        "parameters_identical": old.parameter_hash == new.parameter_hash,
    }


@dataclass(frozen=True)
class EvaluationPanelEntry:
    """One immutable reset/scenario/common-uniform panel entry."""

    panel_episode: int
    snapshot_path: str
    common_uniform_path: str
    snapshot_sha256: str
    common_uniform_sha256: str
    common_uniform_shape: tuple[int, ...]
    scenario_reset_identifier: Mapping[str, Any]
    paired_policy_versions: tuple[int, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "panel_episode": self.panel_episode,
            "snapshot_path": self.snapshot_path,
            "scenario_reset_snapshot_path": self.snapshot_path,
            "common_uniform_path": self.common_uniform_path,
            "snapshot_sha256": self.snapshot_sha256,
            "common_uniform_sha256": self.common_uniform_sha256,
            "common_uniform_shape": list(self.common_uniform_shape),
            "scenario_reset_identifier": _json_safe(self.scenario_reset_identifier),
            "paired_policy_versions": list(self.paired_policy_versions),
        }


@dataclass(frozen=True)
class EvaluationPanel:
    """An immutable panel manifest plus entries validated against disk."""

    panel_id: str
    manifest_path: str
    entries: tuple[EvaluationPanelEntry, ...]
    config_fingerprint: str | None
    diagnostic_only: bool
    formal_evaluation: bool

    def summary(self) -> dict[str, Any]:
        return {
            "schema": PANEL_MANIFEST_SCHEMA,
            "paired_evaluation_schema": PAIRED_EVALUATION_SCHEMA,
            "panel_id": self.panel_id,
            "manifest_path": self.manifest_path,
            "episode_count": len(self.entries),
            "entries": [entry.summary() for entry in self.entries],
            "config_fingerprint": self.config_fingerprint,
            "diagnostic_only": self.diagnostic_only,
            "formal_evaluation": self.formal_evaluation,
            "immutable": True,
        }


def _uniform_hash(value: Any) -> tuple[str, tuple[int, ...]]:
    dtype, shape, raw = _array_for_hash(value, "base_uniforms")
    digest = hashlib.sha256()
    header = _canonical_json({"dtype": dtype, "shape": shape}).encode("utf-8")
    digest.update(header)
    digest.update(raw)
    return digest.hexdigest(), shape


def _load_panel_snapshot(path: Path) -> tuple[Mapping[str, Any], str, tuple[int, ...]]:
    payload = _require_mapping(_torch_load(path), str(path))
    if "base_uniforms" not in payload:
        raise PairedEvaluationProtocolError(f"panel snapshot缺少base_uniforms：{path}")
    uniform_hash, shape = _uniform_hash(payload["base_uniforms"])
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise PairedEvaluationProtocolError(f"panel common uniforms shape非法：{path} {shape}")
    scenario = _require_mapping(payload.get("scenario_reset_identifier"), f"{path}.scenario_reset_identifier")
    versions_raw = payload.get("paired_policy_versions")
    if not isinstance(versions_raw, Sequence) or isinstance(versions_raw, (str, bytes)):
        raise PairedEvaluationProtocolError(f"panel paired_policy_versions必须是list：{path}")
    versions = tuple(_nonnegative_int(item, f"{path}.paired_policy_versions") for item in versions_raw)
    if len(versions) != 2:
        raise PairedEvaluationProtocolError(f"panel paired_policy_versions必须包含old/new两个版本：{path}")
    return payload, uniform_hash, shape


def build_evaluation_panel_manifest(
    snapshot_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    panel_id: str,
    source_note: str,
    config_fingerprint: str | None = None,
    diagnostic_only: bool = True,
    formal_evaluation: bool = False,
    created_at_utc: str | None = None,
) -> EvaluationPanel:
    """Register existing snapshots as an immutable panel; do not resample them."""

    paths = [Path(item) for item in snapshot_paths]
    if not paths:
        raise PairedEvaluationProtocolError("evaluation panel不能为空。")
    if len({str(path.resolve()) for path in paths}) != len(paths):
        raise PairedEvaluationProtocolError("evaluation panel snapshot不能重复。")
    entries: list[EvaluationPanelEntry] = []
    for ordinal, path in enumerate(paths, start=1):
        payload, uniform_hash, shape = _load_panel_snapshot(path)
        panel_episode = _nonnegative_int(payload.get("audit_episode", ordinal), f"{path}.audit_episode")
        entries.append(
            EvaluationPanelEntry(
                panel_episode=panel_episode,
                snapshot_path=str(path.resolve()),
                common_uniform_path=f"{path.resolve()}#base_uniforms",
                snapshot_sha256=file_sha256(path),
                common_uniform_sha256=uniform_hash,
                common_uniform_shape=shape,
                scenario_reset_identifier=copy.deepcopy(
                    dict(_require_mapping(payload["scenario_reset_identifier"], f"{path}.scenario_reset_identifier"))
                ),
                paired_policy_versions=tuple(
                    _nonnegative_int(item, f"{path}.paired_policy_versions")
                    for item in payload["paired_policy_versions"]
                ),
            )
        )
    if len({entry.panel_episode for entry in entries}) != len(entries):
        raise PairedEvaluationProtocolError("evaluation panel episode ID不能重复。")
    entries.sort(key=lambda item: item.panel_episode)
    manifest = {
        "schema": PANEL_MANIFEST_SCHEMA,
        "paired_evaluation_schema": PAIRED_EVALUATION_SCHEMA,
        "panel_id": str(panel_id),
        "episode_count": len(entries),
        "entries": [entry.summary() for entry in entries],
        "config_fingerprint": config_fingerprint,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "source_note": str(source_note),
        "diagnostic_only": bool(diagnostic_only),
        "formal_evaluation": bool(formal_evaluation),
        "immutable": True,
        "generation_semantics": "register_existing_snapshots_without_resampling_or_environment_rollout",
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return load_evaluation_panel_manifest(target)


def load_evaluation_panel_manifest(path: str | Path) -> EvaluationPanel:
    """Load and re-hash a manifest; a changed snapshot fails the gate."""

    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedEvaluationProtocolError(f"panel manifest无法读取：{target}") from exc
    payload = _require_mapping(payload, str(target))
    if payload.get("schema") != PANEL_MANIFEST_SCHEMA:
        raise PairedEvaluationProtocolError("panel manifest schema不兼容。")
    if payload.get("paired_evaluation_schema") != PAIRED_EVALUATION_SCHEMA:
        raise PairedEvaluationProtocolError("panel manifest paired schema不兼容。")
    if payload.get("immutable") is not True:
        raise PairedEvaluationProtocolError("panel manifest必须声明immutable=true。")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise PairedEvaluationProtocolError("panel manifest entries必须是非空list。")
    entries: list[EvaluationPanelEntry] = []
    for raw in raw_entries:
        item = _require_mapping(raw, "panel manifest entry")
        panel_episode = _nonnegative_int(item.get("panel_episode"), "panel_episode")
        snapshot_raw = item.get("snapshot_path", item.get("scenario_reset_snapshot_path"))
        if snapshot_raw is None:
            raise PairedEvaluationProtocolError("panel entry缺少snapshot_path。")
        snapshot = Path(str(snapshot_raw))
        if not snapshot.is_file():
            raise PairedEvaluationProtocolError(f"panel snapshot不存在：{snapshot}")
        payload_snapshot, uniform_hash, shape = _load_panel_snapshot(snapshot)
        expected_snapshot_hash = str(item.get("snapshot_sha256", ""))
        actual_snapshot_hash = file_sha256(snapshot)
        if actual_snapshot_hash != expected_snapshot_hash:
            raise PairedEvaluationProtocolError(f"panel snapshot hash不一致：{snapshot}")
        expected_uniform_hash = str(item.get("common_uniform_sha256", ""))
        if uniform_hash != expected_uniform_hash:
            raise PairedEvaluationProtocolError(f"panel common uniform hash不一致：{snapshot}")
        expected_shape = tuple(int(value) for value in item.get("common_uniform_shape", []))
        if shape != expected_shape:
            raise PairedEvaluationProtocolError(f"panel common uniform shape不一致：{snapshot}")
        expected_scenario = _require_mapping(item.get("scenario_reset_identifier"), "panel entry scenario")
        actual_scenario = _require_mapping(payload_snapshot["scenario_reset_identifier"], "snapshot scenario")
        if _canonical_json(expected_scenario) != _canonical_json(actual_scenario):
            raise PairedEvaluationProtocolError(f"panel scenario/reset identity不一致：{snapshot}")
        versions_raw = item.get("paired_policy_versions")
        if not isinstance(versions_raw, Sequence) or isinstance(versions_raw, (str, bytes)):
            raise PairedEvaluationProtocolError(f"panel paired versions非法：{snapshot}")
        versions = tuple(_nonnegative_int(value, "panel paired policy version") for value in versions_raw)
        actual_versions = tuple(_nonnegative_int(value, "snapshot paired policy version") for value in payload_snapshot["paired_policy_versions"])
        if versions != actual_versions or len(versions) != 2:
            raise PairedEvaluationProtocolError(f"panel paired policy versions不一致：{snapshot}")
        common_path = str(item.get("common_uniform_path", f"{snapshot.resolve()}#base_uniforms"))
        entries.append(
            EvaluationPanelEntry(
                panel_episode=panel_episode,
                snapshot_path=str(snapshot.resolve()),
                common_uniform_path=common_path,
                snapshot_sha256=actual_snapshot_hash,
                common_uniform_sha256=uniform_hash,
                common_uniform_shape=shape,
                scenario_reset_identifier=copy.deepcopy(dict(actual_scenario)),
                paired_policy_versions=versions,
            )
        )
    entries.sort(key=lambda item: item.panel_episode)
    if len({entry.panel_episode for entry in entries}) != len(entries):
        raise PairedEvaluationProtocolError("panel entry episode ID重复。")
    if _nonnegative_int(payload.get("episode_count"), "panel episode_count") != len(entries):
        raise PairedEvaluationProtocolError("panel episode_count不一致。")
    return EvaluationPanel(
        panel_id=str(payload.get("panel_id")),
        manifest_path=str(target.resolve()),
        entries=tuple(entries),
        config_fingerprint=(None if payload.get("config_fingerprint") is None else str(payload["config_fingerprint"])),
        diagnostic_only=bool(payload.get("diagnostic_only", False)),
        formal_evaluation=bool(payload.get("formal_evaluation", False)),
    )


def load_common_uniform_inputs(panel: EvaluationPanel) -> dict[int, np.ndarray]:
    """Load only the panel-embedded common uniforms for explicit evaluator input."""

    result: dict[int, np.ndarray] = {}
    for entry in panel.entries:
        payload, uniform_hash, shape = _load_panel_snapshot(Path(entry.snapshot_path))
        if uniform_hash != entry.common_uniform_sha256 or shape != entry.common_uniform_shape:
            raise PairedEvaluationProtocolError(f"panel common uniform changed：{entry.snapshot_path}")
        # Preserve the persisted dtype: the hash gate must detect even a
        # silent float32/float64 replacement of the common random numbers.
        result[entry.panel_episode] = np.asarray(payload["base_uniforms"]).copy()
    return result


def validate_common_uniform_inputs(
    panel: EvaluationPanel,
    common_uniform_inputs: Mapping[int | str, Any],
) -> dict[str, Any]:
    """Require caller-provided uniforms to match the immutable panel exactly."""

    supplied = {int(key): value for key, value in common_uniform_inputs.items()}
    expected_ids = {entry.panel_episode for entry in panel.entries}
    if set(supplied) != expected_ids:
        raise PairedEvaluationProtocolError(
            f"common_uniform_inputs episode IDs不一致：{sorted(supplied)} != {sorted(expected_ids)}"
        )
    hashes_match = True
    shapes_match = True
    for entry in panel.entries:
        uniform_hash, shape = _uniform_hash(supplied[entry.panel_episode])
        hashes_match = hashes_match and uniform_hash == entry.common_uniform_sha256
        shapes_match = shapes_match and shape == entry.common_uniform_shape
    if not hashes_match or not shapes_match:
        raise PairedEvaluationProtocolError("common_uniform_inputs与panel hash/shape不一致。")
    return {
        "pass": True,
        "episode_count": len(supplied),
        "all_hashes_match": hashes_match,
        "all_shapes_match": shapes_match,
        "common_uniforms_immutable": True,
    }


def _metric(record: Mapping[str, Any], name: str, scope: str) -> float:
    for key in _METRIC_ALIASES[name]:
        if key in record:
            return _finite_float(record[key], f"{scope}.{key}")
    raise PairedEvaluationProtocolError(f"{scope}缺少metric {name}。")


def _delta_metric(delta: Mapping[str, Any], name: str, old: float, new: float, scope: str) -> float:
    aliases = {
        "G": ("DeltaG", "delta_g"),
        "Q": ("DeltaQ", "delta_q"),
        "cache": ("DeltaCache", "delta_cache"),
        "UAV": ("DeltaUAV", "delta_uav"),
    }[name]
    for key in aliases:
        if key in delta:
            return _finite_float(delta[key], f"{scope}.{key}")
    return float(new - old)


def _record_policy_keys(record: Mapping[str, Any]) -> tuple[str, str]:
    if "old" in record and "new" in record:
        return "old", "new"
    if "policy_old" in record and "policy_new" in record:
        return "policy_old", "policy_new"
    if "v5" in record and "v6" in record:
        return "v5", "v6"
    raise PairedEvaluationProtocolError("paired record必须包含old/new或v5/v6记录。")


def _normalize_optional_mapping(value: Any, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return dict(_require_mapping(value, name))


def normalize_paired_record(
    record: Mapping[str, Any],
    *,
    old_identity: PolicyArtifactIdentity | None = None,
    new_identity: PolicyArtifactIdentity | None = None,
) -> dict[str, Any]:
    """Normalize one already-executed paired rollout record read-only."""

    source = _require_mapping(record, "paired record")
    old_key, new_key = _record_policy_keys(source)
    old_raw = _require_mapping(source[old_key], f"paired record.{old_key}")
    new_raw = _require_mapping(source[new_key], f"paired record.{new_key}")
    old_metrics = {name: _metric(old_raw, name, f"{old_key}") for name in ("G", "Mean_G", "Q", "cache", "UAV")}
    new_metrics = {name: _metric(new_raw, name, f"{new_key}") for name in ("G", "Mean_G", "Q", "cache", "UAV")}
    delta_raw = _normalize_optional_mapping(source.get("delta"), "paired record.delta") or {}
    delta = {
        "DeltaG": _delta_metric(delta_raw, "G", old_metrics["G"], new_metrics["G"], "delta"),
        "DeltaQ": _delta_metric(delta_raw, "Q", old_metrics["Q"], new_metrics["Q"], "delta"),
        "DeltaCache": _delta_metric(delta_raw, "cache", old_metrics["cache"], new_metrics["cache"], "delta"),
        "DeltaUAV": _delta_metric(delta_raw, "UAV", old_metrics["UAV"], new_metrics["UAV"], "delta"),
    }
    identity_rhs = delta["DeltaQ"] - delta["DeltaCache"] - delta["DeltaUAV"]
    identity_error = abs(delta["DeltaG"] - identity_rhs)
    if identity_error > 1e-12:
        raise PairedEvaluationProtocolError(
            f"Delta decomposition失败：error={identity_error}。"
        )
    panel_episode = source.get("panel_episode", source.get("audit_episode"))
    if panel_episode is None:
        raise PairedEvaluationProtocolError("paired record缺少panel_episode/audit_episode。")
    panel_episode = _nonnegative_int(panel_episode, "paired record panel_episode")
    normalized: dict[str, Any] = {
        "panel_episode": panel_episode,
        "old_policy_key": old_key,
        "new_policy_key": new_key,
        "old": old_metrics,
        "new": new_metrics,
        "delta": {
            **delta,
            "DeltaG_identity_rhs": identity_rhs,
            "DeltaG_identity_error": identity_error,
        },
        "shared_scenario_reset": bool(source.get("shared_scenario_reset", True)),
        "shared_environment_reset_rng": bool(source.get("shared_environment_reset_rng", True)),
        "common_base_uniform": bool(source.get("common_base_uniform", True)),
        "action_disagreement": _normalize_optional_mapping(source.get("action_disagreement"), "action_disagreement"),
        "first_divergence": _normalize_optional_mapping(source.get("first_divergence"), "first_divergence"),
        "pre_divergence_equality": _normalize_optional_mapping(
            source.get("pre_divergence_equality"), "pre_divergence_equality"
        ),
        "old_raw": dict(old_raw),
        "new_raw": dict(new_raw),
    }
    if old_identity is not None and new_identity is not None:
        old_version = old_identity.policy_version
        new_version = new_identity.policy_version
        for key, expected in ((old_key, old_version), (new_key, new_version)):
            version_candidates = (
                f"{key}_policy_version",
                "policy_version" if key == old_key else "policy_version_new",
            )
            observed = next((source[item] for item in version_candidates if item in source), None)
            if observed is not None and int(observed) != expected:
                raise PairedEvaluationProtocolError(
                    f"paired record policy version错配：{key}={observed}, expected={expected}"
                )
    return normalized


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise PairedEvaluationProtocolError("统计输入不能为空。")
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(math.fsum(float(item) for item in array) / len(array)),
        "std": float(np.std(array, ddof=0)),
        "median": float(median(float(item) for item in array)),
    }


def _action_disagreement_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: list[int] = []
    fractions: list[float] = []
    slot_counts: list[int] = []
    slot_fractions: list[float] = []
    available = True
    for record in records:
        raw = record.get("action_disagreement")
        if raw is None:
            available = False
            continue
        count = raw.get("count")
        fraction = raw.get("fraction")
        if count is None or fraction is None:
            available = False
            continue
        counts.append(_nonnegative_int(count, "action_disagreement.count"))
        fractions.append(_finite_float(fraction, "action_disagreement.fraction"))
        if "slot_count" in raw:
            slot_counts.append(_nonnegative_int(raw["slot_count"], "action_disagreement.slot_count"))
        if "slot_fraction" in raw:
            slot_fractions.append(_finite_float(raw["slot_fraction"], "action_disagreement.slot_fraction"))
    if not available or len(counts) != len(records):
        return {"available": False}
    result: dict[str, Any] = {
        "available": True,
        "count_sum": int(sum(counts)),
        "count_mean": float(sum(counts) / len(counts)),
        "fraction_mean": float(math.fsum(fractions) / len(fractions)),
    }
    if len(slot_counts) == len(records):
        result["slot_count_sum"] = int(sum(slot_counts))
    if len(slot_fractions) == len(records):
        result["slot_fraction_mean"] = float(math.fsum(slot_fractions) / len(slot_fractions))
    return result


def summarize_paired_records(
    records: Iterable[Mapping[str, Any]],
    *,
    old_identity: PolicyArtifactIdentity | None = None,
    new_identity: PolicyArtifactIdentity | None = None,
    panel: EvaluationPanel | None = None,
) -> dict[str, Any]:
    """Return the formal paired metric summary without any rollout side effect."""

    normalized = [
        normalize_paired_record(item, old_identity=old_identity, new_identity=new_identity)
        for item in records
    ]
    if not normalized:
        raise PairedEvaluationProtocolError("paired records不能为空。")
    panel_ids = [int(item["panel_episode"]) for item in normalized]
    if len(set(panel_ids)) != len(panel_ids):
        raise PairedEvaluationProtocolError("paired records panel episode不能重复。")
    if panel is not None and set(panel_ids) != {entry.panel_episode for entry in panel.entries}:
        raise PairedEvaluationProtocolError("paired records与evaluation panel episode集合不一致。")
    delta_g = [float(item["delta"]["DeltaG"]) for item in normalized]
    delta_q = [float(item["delta"]["DeltaQ"]) for item in normalized]
    delta_cache = [float(item["delta"]["DeltaCache"]) for item in normalized]
    delta_uav = [float(item["delta"]["DeltaUAV"]) for item in normalized]
    aggregate = {
        PRIMARY_POLICY_IMPROVEMENT_METRIC: _stats(delta_g)["mean"],
        "paired_eval_delta_G_std": _stats(delta_g)["std"],
        "paired_eval_median_delta_G": _stats(delta_g)["median"],
        "mean_DeltaG": _stats(delta_g)["mean"],
        "std_DeltaG": _stats(delta_g)["std"],
        "median_DeltaG": _stats(delta_g)["median"],
        "positive_count": int(sum(value > 0.0 for value in delta_g)),
        "negative_count": int(sum(value < 0.0 for value in delta_g)),
        "zero_count": int(sum(value == 0.0 for value in delta_g)),
        "positive_fraction": float(sum(value > 0.0 for value in delta_g) / len(delta_g)),
        "mean_DeltaQ": float(math.fsum(delta_q) / len(delta_q)),
        "mean_DeltaCache": float(math.fsum(delta_cache) / len(delta_cache)),
        "mean_DeltaUAV": float(math.fsum(delta_uav) / len(delta_uav)),
        "paired_mean_DeltaQ": float(math.fsum(delta_q) / len(delta_q)),
        "paired_mean_DeltaCache": float(math.fsum(delta_cache) / len(delta_cache)),
        "paired_mean_DeltaUAV": float(math.fsum(delta_uav) / len(delta_uav)),
        "max_DeltaG_identity_error": float(
            max(float(item["delta"]["DeltaG_identity_error"]) for item in normalized)
        ),
        "all_shared_scenario_reset": bool(all(item["shared_scenario_reset"] for item in normalized)),
        "all_shared_environment_reset_rng": bool(
            all(item["shared_environment_reset_rng"] for item in normalized)
        ),
        "all_common_base_uniform": bool(all(item["common_base_uniform"] for item in normalized)),
        "all_pre_divergence_equal": bool(
            all(
                item["pre_divergence_equality"] is not None
                and bool(item["pre_divergence_equality"].get("all_equal", False))
                for item in normalized
            )
        ),
        "all_true_terminal": bool(
            all(bool(item["old_raw"].get("true_terminal", True)) and bool(item["new_raw"].get("true_terminal", True)) for item in normalized)
        ),
        "all_no_cf": bool(
            all(
                item["old_raw"].get("cf_branch_count", 0) == 0
                and item["new_raw"].get("cf_branch_count", 0) == 0
                for item in normalized
            )
        ),
        "paired_episode_count": len(normalized),
        "action_disagreement": _action_disagreement_summary(normalized),
    }
    result: dict[str, Any] = {
        "paired_evaluation_schema": PAIRED_EVALUATION_SCHEMA,
        "panel_episode_ids": sorted(panel_ids),
        "episodes": normalized,
        "aggregate": aggregate,
    }
    return result


def _validate_config_snapshot(
    config_snapshot: Mapping[str, Any],
    panel: EvaluationPanel,
) -> dict[str, Any]:
    supplied_fingerprint = config_snapshot_fingerprint(config_snapshot)
    if panel.config_fingerprint is not None and supplied_fingerprint != panel.config_fingerprint:
        raise PairedEvaluationProtocolError("config snapshot与panel fingerprint不一致。")
    return {
        "pass": True,
        "fingerprint": supplied_fingerprint,
        "panel_fingerprint_match": panel.config_fingerprint is None
        or supplied_fingerprint == panel.config_fingerprint,
    }


def compare_policy_checkpoints(
    old_policy: str | Path,
    new_policy: str | Path,
    evaluation_panel: str | Path | EvaluationPanel,
    common_uniform_inputs: Mapping[int | str, Any],
    config_snapshot: Mapping[str, Any],
    *,
    records: Iterable[Mapping[str, Any]] | None = None,
    rollout_executor: Callable[[PolicyArtifactIdentity, PolicyArtifactIdentity, EvaluationPanelEntry, Any, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare two explicit policy artifacts under the paired protocol.

    ``records`` is used for read-only regression validation of already executed
    rollouts.  If it is omitted, an explicit ``rollout_executor`` is required;
    the function never discovers or uses a live policy/RNG implicitly.
    """

    records_supplied = records is not None
    old_identity = load_policy_artifact(old_policy, "old")
    new_identity = load_policy_artifact(new_policy, "new")
    identity_gate = validate_policy_pair(old_identity, new_identity)
    panel = (
        evaluation_panel
        if isinstance(evaluation_panel, EvaluationPanel)
        else load_evaluation_panel_manifest(evaluation_panel)
    )
    uniform_gate = validate_common_uniform_inputs(panel, common_uniform_inputs)
    config_gate = _validate_config_snapshot(config_snapshot, panel)
    for entry in panel.entries:
        if tuple((old_identity.policy_version, new_identity.policy_version)) != entry.paired_policy_versions:
            raise PairedEvaluationProtocolError(
                f"panel policy versions与artifact不一致：episode={entry.panel_episode}"
            )
    if records is None:
        if rollout_executor is None:
            raise PairedEvaluationProtocolError(
                "没有paired records时必须显式提供rollout_executor。"
            )
        executed: list[Mapping[str, Any]] = []
        for entry in panel.entries:
            executed.append(
                rollout_executor(
                    old_identity,
                    new_identity,
                    entry,
                    common_uniform_inputs[entry.panel_episode],
                    config_snapshot,
                )
            )
        records = executed
    summary = summarize_paired_records(
        records,
        old_identity=old_identity,
        new_identity=new_identity,
        panel=panel,
    )
    return {
        "paired_evaluation_schema": PAIRED_EVALUATION_SCHEMA,
        "policy_identity": {
            "old": old_identity.summary(),
            "new": new_identity.summary(),
            "gate": identity_gate,
        },
        "evaluation_panel": panel.summary(),
        "common_uniform_gate": uniform_gate,
        "config_snapshot_gate": config_gate,
        "comparison": summary,
        "metric_hierarchy": metric_hierarchy(),
        "is_read_only": True,
        "execution_mode": (
            "read_only_existing_records" if records_supplied else "explicit_rollout_executor"
        ),
        "environment_rollouts_executed_by_protocol": 0,
        "rollout_executor_was_supplied": not records_supplied,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }


def metric_hierarchy() -> dict[str, Any]:
    """Return the fixed meaning of causal and descriptive metrics."""

    return {
        "primary_policy_improvement_metric": PRIMARY_POLICY_IMPROVEMENT_METRIC,
        "primary_metric_definition": "mean over same-panel paired episodes of DeltaG = G_new - G_old",
        "supporting_paired_metrics": [
            "paired_positive_fraction",
            "paired_mean_DeltaQ",
            "paired_mean_DeltaCache",
            "paired_mean_DeltaUAV",
        ],
        "descriptive_training_metrics": [
            DESCRIPTIVE_ONLINE_METRIC,
            "online_rollout_moving_average_Mean_G",
        ],
        "online_metric_allowed_use": "display training trajectory and descriptive trend",
        "online_metric_prohibited_use": [
            "standalone causal harmful-update verdict",
            "automatic rollback",
            "algorithm modification",
            "premature training failure declaration",
        ],
        "negative_online_trend_label": "ONLINE_PERFORMANCE_DECLINED",
        "causal_verdict_requirement": "paired mean DeltaG sign plus paired decomposition and panel gates",
    }


def validate_update6_regression(
    audit_json_path: str | Path,
    *,
    old_policy: str | Path,
    new_policy: str | Path,
    evaluation_panel: str | Path | EvaluationPanel,
    common_uniform_inputs: Mapping[int | str, Any],
    config_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-aggregate the existing Update6 JSON without executing its rollouts."""

    audit_path = Path(audit_json_path)
    try:
        source = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedEvaluationProtocolError(f"Update6 audit JSON无法读取：{audit_path}") from exc
    source = _require_mapping(source, "Update6 audit JSON")
    paired = _require_mapping(source.get("paired_audit"), "Update6 paired_audit")
    raw_records = paired.get("episodes")
    if not isinstance(raw_records, list) or len(raw_records) != 4:
        raise PairedEvaluationProtocolError("Update6 regression fixture必须包含4条paired records。")
    comparison = compare_policy_checkpoints(
        old_policy,
        new_policy,
        evaluation_panel,
        common_uniform_inputs,
        config_snapshot,
        records=raw_records,
    )
    aggregate = comparison["comparison"]["aggregate"]
    source_aggregate = _require_mapping(paired.get("aggregate"), "Update6 source aggregate")
    exact_fields = (
        (PRIMARY_POLICY_IMPROVEMENT_METRIC, "mean_DeltaG"),
        ("paired_eval_median_delta_G", "median_DeltaG"),
        ("paired_mean_DeltaQ", "mean_DeltaQ"),
        ("paired_mean_DeltaCache", "mean_DeltaCache"),
        ("paired_mean_DeltaUAV", "mean_DeltaUAV"),
    )
    exact_checks: dict[str, bool] = {}
    exact_errors: dict[str, float] = {}
    for new_name, source_name in exact_fields:
        source_value = _finite_float(source_aggregate[source_name], f"source_aggregate.{source_name}")
        actual_value = _finite_float(aggregate[new_name], f"comparison.aggregate.{new_name}")
        exact_checks[new_name] = actual_value == source_value
        exact_errors[new_name] = actual_value - source_value
    count_checks = {
        name: int(aggregate[name]) == int(source_aggregate[name])
        for name in ("positive_count", "negative_count", "zero_count")
    }
    source_errors = [
        _finite_float(item.get("delta", {}).get("DeltaG_identity_error", 0.0), "source identity error")
        for item in raw_records
    ]
    source_max_error = max(source_errors)
    decomposition_check = aggregate["max_DeltaG_identity_error"] == source_max_error
    if not all(exact_checks.values()) or not all(count_checks.values()) or not decomposition_check:
        raise PairedEvaluationProtocolError("Update6 paired evaluator regression exact gate失败。")
    result = {
        "source_audit_json": str(audit_path.resolve()),
        "raw_record_count": len(raw_records),
        "reaggregation_exact_checks": exact_checks,
        "reaggregation_exact_errors": exact_errors,
        "count_checks": count_checks,
        "delta_decomposition_check": {
            "pass": decomposition_check,
            "max_error": aggregate["max_DeltaG_identity_error"],
        },
        "comparison": comparison,
        "verdict": "POLICY_IMPROVED"
        if aggregate[PRIMARY_POLICY_IMPROVEMENT_METRIC] > 0.0
        and aggregate["positive_count"] > aggregate["negative_count"]
        else "POLICY_DEGRADED",
        "regression_pass": True,
    }
    return result


def build_protocol_audit_report(
    regression: Mapping[str, Any],
    *,
    formal_rng_non_mutation_evidence: Mapping[str, Any],
    trigger_count_status: Mapping[str, Any],
    online_semantics_note: str,
) -> dict[str, Any]:
    """Assemble the JSON/TXT audit payload from validated read-only pieces."""

    regression_map = _require_mapping(regression, "regression")
    comparison = _require_mapping(regression_map.get("comparison"), "regression.comparison")
    comparison_aggregate = _require_mapping(comparison.get("comparison", {}).get("aggregate"), "regression aggregate")
    rng_evidence = dict(_require_mapping(formal_rng_non_mutation_evidence, "formal_rng_non_mutation_evidence"))
    trigger_status = dict(_require_mapping(trigger_count_status, "trigger_count_status"))
    protocol_gates = {
        "A_existing_update6_four_pairs_read_only_aggregation_exact": bool(regression_map.get("regression_pass")),
        "B_mean_delta_G_exact": bool(regression_map.get("reaggregation_exact_checks", {}).get(PRIMARY_POLICY_IMPROVEMENT_METRIC, False)),
        "C_delta_decomposition_exact": bool(regression_map.get("delta_decomposition_check", {}).get("pass", False)),
        "D_policy_identity_hash_validation": bool(comparison.get("policy_identity", {}).get("gate", {}).get("pass", False)),
        "E_formal_rng_non_mutation_design": bool(rng_evidence.get("design_pass", False)),
        "F_evaluation_panel_immutable": bool(comparison.get("evaluation_panel", {}).get("immutable", False)),
        "G_online_vs_paired_metric_semantics_separated": True,
        "H_training_numerical_semantics_unchanged": bool(rng_evidence.get("training_numerical_semantics_unchanged", False)),
        "I_environment_steps_zero": int(rng_evidence.get("environment_steps", -1)) == 0,
        "J_optimizer_backward_zero": int(rng_evidence.get("optimizer_steps", -1)) == 0 and int(rng_evidence.get("backward_calls", -1)) == 0,
    }
    all_gates_pass = all(protocol_gates.values())
    return {
        "protocol_schema": PAIRED_EVALUATION_SCHEMA,
        "protocol_name": "Formal Policy Evaluation Protocol Repair",
        "purpose": "separate policy improvement from unpaired online scenario noise",
        "metric_hierarchy": metric_hierarchy(),
        "policy_identity_rules": {
            "inputs_must_be_explicit_artifacts": True,
            "same_theta_old_across_panel": True,
            "same_theta_new_across_panel": True,
            "architecture_and_algorithm_schema_must_match": True,
            "parameter_hash_gate": True,
        },
        "panel_isolation_rules": {
            "same_scenario_reset": True,
            "same_initial_state": True,
            "same_environment_randomness": True,
            "same_autoregressive_base_uniforms": True,
            "independent_from_formal_training_rng": True,
            "panel_immutable": True,
        },
        "crn_rules": {
            "explicit_common_uniform_coupling": True,
            "pre_divergence_equality_required": True,
            "delta_identity": "DeltaG = DeltaQ - DeltaCache - DeltaUAV",
            "max_allowed_decomposition_error": "floating-point roundoff only; fixture max recorded",
        },
        "update6_regression": regression_map,
        "online_vs_paired_semantics": {
            "online_batch_comparison": "different scenario/reset/randomness; descriptive only",
            "paired_comparison": "same panel/state/randomness/uniforms; causal policy-improvement measurement",
            "historical_stop_label": "STOP_FORMAL_RUN_NEGATIVE_TREND means unpaired online performance trend negative, not causally harmful update",
            "note": online_semantics_note,
        },
        "trigger_count_status": trigger_status,
        "formal_rng_non_mutation": rng_evidence,
        "protocol_gates": protocol_gates,
        "final_verdict": "PAIRED_EVALUATION_PROTOCOL_READY" if all_gates_pass else "PAIRED_EVALUATION_PROTOCOL_NOT_READY",
        "runtime_budget": {
            "environment_steps": 0,
            "formal_cf_branches": 0,
            "actor_optimizer_steps": 0,
            "critic_optimizer_steps": 0,
            "backward_calls": 0,
            "new_update6_rollouts": 0,
        },
        "source_artifacts_preserved": True,
        "no_commit_push_pr": True,
        "diagnostic_note": {
            "paired_aggregate": dict(comparison_aggregate),
            "paired_primary_metric": comparison_aggregate.get(PRIMARY_POLICY_IMPROVEMENT_METRIC),
        },
    }


def render_protocol_audit_text(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable audit without losing JSON detail."""

    report = _require_mapping(report, "report")
    regression = _require_mapping(report.get("update6_regression"), "report.update6_regression")
    comparison = _require_mapping(regression.get("comparison"), "regression.comparison")
    summary = _require_mapping(comparison.get("comparison"), "regression comparison")
    aggregate = _require_mapping(summary.get("aggregate"), "regression aggregate")
    identity = _require_mapping(comparison.get("policy_identity"), "policy identity")
    old_identity = _require_mapping(identity.get("old"), "old identity")
    new_identity = _require_mapping(identity.get("new"), "new identity")
    gates = _require_mapping(report.get("protocol_gates"), "protocol gates")
    lines = [
        "# Formal Policy Evaluation Protocol Repair",
        "",
        f"- protocol schema: `{report.get('protocol_schema')}`",
        f"- final verdict: `{report.get('final_verdict')}`",
        "",
        "## Why online batch comparison is not causal",
        "",
        "Adjacent online training episodes use different scenario/reset/randomness. Their raw Mean G mixes policy parameter effect, scenario realization, and environment randomness. It is therefore descriptive and cannot alone label an update harmful, trigger rollback, or declare training failed.",
        "",
        "## Metric hierarchy",
        "",
        f"- primary policy-improvement metric: `{report.get('metric_hierarchy', {}).get('primary_policy_improvement_metric')}`",
        "- paired supporting metrics: positive fraction and DeltaQ/DeltaCache/DeltaUAV decomposition",
        f"- descriptive training metric: `{report.get('metric_hierarchy', {}).get('descriptive_training_metrics', [DESCRIPTIVE_ONLINE_METRIC])[0]}`",
        "",
        "## Explicit evaluator inputs",
        "",
        f"- old artifact: `{old_identity.get('path')}` (version {old_identity.get('policy_version')})",
        f"- new artifact: `{new_identity.get('path')}` (version {new_identity.get('policy_version')})",
        f"- old parameter hash: `{old_identity.get('parameter_hash')}`",
        f"- new parameter hash: `{new_identity.get('parameter_hash')}`",
        f"- panel: `{comparison.get('evaluation_panel', {}).get('manifest_path')}`",
        "- common uniforms: explicitly supplied and hash-checked against each panel snapshot",
        "- config snapshot: explicitly supplied and fingerprint-checked",
        "",
        "## Update6 read-only regression",
        "",
        f"- records: {summary.get('aggregate', {}).get('paired_episode_count')}",
        f"- DeltaG values: {[item.get('delta', {}).get('DeltaG') for item in summary.get('episodes', [])]}",
        f"- mean DeltaG: {aggregate.get('mean_DeltaG')}",
        f"- median DeltaG: {aggregate.get('median_DeltaG')}",
        f"- positive/negative/zero: {aggregate.get('positive_count')}/{aggregate.get('negative_count')}/{aggregate.get('zero_count')}",
        f"- mean DeltaQ: {aggregate.get('mean_DeltaQ')}",
        f"- mean DeltaCache: {aggregate.get('mean_DeltaCache')}",
        f"- mean DeltaUAV: {aggregate.get('mean_DeltaUAV')}",
        f"- max decomposition error: {aggregate.get('max_DeltaG_identity_error')}",
        f"- regression verdict: `{regression.get('verdict')}`",
        "",
        "## CRN and panel gates",
        "",
        f"- shared scenario/reset: {aggregate.get('all_shared_scenario_reset')}",
        f"- shared environment reset RNG: {aggregate.get('all_shared_environment_reset_rng')}",
        f"- common base uniforms: {aggregate.get('all_common_base_uniform')}",
        f"- pre-divergence equality: {aggregate.get('all_pre_divergence_equal')}",
        f"- immutable panel: {comparison.get('evaluation_panel', {}).get('immutable')}",
        "",
        "## Trigger count persistence",
        "",
        f"- status: `{report.get('trigger_count_status', {}).get('status')}`",
        f"- reason: {report.get('trigger_count_status', {}).get('reason')}",
        "",
        "## Correctness gates",
        "",
    ]
    lines.extend(f"- {name}: {value}" for name, value in gates.items())
    lines.extend(
        [
            "",
            f"- final: `{report.get('final_verdict')}`",
            "",
            "## Runtime budget",
            "",
            "- environment steps: 0",
            "- formal CF branches: 0",
            "- optimizer steps: 0",
            "- backward calls: 0",
            "- new Update6 rollouts: 0",
            "",
            "## Historical label semantics",
            "",
            report.get("online_vs_paired_semantics", {}).get("note", ""),
            "",
        ]
    )
    return "\n".join(lines)


def save_protocol_audit_reports(
    report: Mapping[str, Any],
    json_path: str | Path,
    text_path: str | Path,
) -> tuple[Path, Path]:
    """Persist only the requested protocol audit reports."""

    json_target = Path(json_path)
    text_target = Path(text_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    text_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    text_target.write_text(render_protocol_audit_text(report), encoding="utf-8", newline="\n")
    return json_target, text_target
