from __future__ import annotations

import copy
import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common import project_config
from decision import cache_configuration
from ppo_simple import cache_credit
from ppo_simple.rollout import compute_gae
from ppo_simple.trainer import (
    JOINT_PROPOSED_CHECKPOINT_VERSION,
    PPOV2Trainer,
    RunningReturnStatistics,
    _assert_realization_context_compatible,
    _config_hash,
    _ensure_episode_realization,
    _load_single_episode_generator,
    _run_with_isolated_rng,
    _validate_episode_realization_complete,
    aligned_train_steps,
    load_environment_context,
)
from scenario.realization import derive_episode_seeds, derive_namespaced_seed


TASK_NAME = "CRITIC_TARGET_ALIGNMENT_MINIMAL_EXPERIMENT"
CONFIG_PATH = ROOT / "configs" / "proposed" / "system_config.toml"
SPLIT_MANIFEST_PATH = ROOT / "environment_data" / "data_split_manifest.json"
TRAIN_REALIZATION_SEED = 20001
TRAIN_REALIZATION = (
    ROOT
    / "environment_data"
    / "proposed_joint_training_preexisting_51_backup_20260921"
    / f"episode_{TRAIN_REALIZATION_SEED}"
)
DIAGNOSTIC_REALIZATION_SEED = 12184115623283823756
DIAGNOSTIC_NAMESPACE = "ppo_candidate_q_integrity"
TRAIN_NAMESPACE = "critic_target_alignment_training"
TARGET_TRANSITION = 149
TARGET_HEAD = "UAV_1"
TARGET_CONFIG = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
MC_DELTA_R0 = -104.8574641123414
MC_DELTA_R1 = -380.0770139247179
EXPECTED_TRANSITION_FINGERPRINT = (
    "2805d3e60f0b40457ab53652995bbbd58e4cb438bac06c3af5445284981d3408"
)
EXPECTED_INITIAL_CRITIC_DIGEST = (
    "ffa1c1915716f996732c7470c348fcb08854f17ffe53c2d8ef1536f23492705f"
)
CRITIC_CHECKPOINT_SCHEMA_VERSION = 1
OUTPUT_DIR = ROOT / "runs" / "diagnostics" / "critic_target_alignment"
RECOVERY_DIR = OUTPUT_DIR / "recovery"
TRAIN_ROLLOUT_PATH = RECOVERY_DIR / "shared_train_rollout.npz"
TRAIN_ROLLOUT_METADATA_PATH = RECOVERY_DIR / "shared_train_rollout.metadata.json"
DIAGNOSTIC_ROLLOUT_PATH = RECOVERY_DIR / "diagnostic_rollout.npz"
DIAGNOSTIC_ROLLOUT_METADATA_PATH = RECOVERY_DIR / "diagnostic_rollout.metadata.json"
TARGET_ARRAYS_PATH = RECOVERY_DIR / "critic_targets.npz"
TARGET_ARRAYS_METADATA_PATH = RECOVERY_DIR / "critic_targets.metadata.json"
MINIBATCH_SEQUENCE_PATH = RECOVERY_DIR / "minibatch_indices.json"
INITIAL_CRITIC_STATE_PATH = RECOVERY_DIR / "initial_critic_state.pt"
INITIAL_CRITIC_STATE_METADATA_PATH = RECOVERY_DIR / "initial_critic_state.metadata.json"
DIAGNOSTIC_REALIZATION_PATH = RECOVERY_DIR / "diagnostic_train_realization"
REPORT_JSON = ROOT / "reports" / "critic_target_alignment_minimal_experiment.json"
REPORT_MD = ROOT / "reports" / "critic_target_alignment_minimal_experiment.md"
CURRENT_ATTEMPT = "authorized_compensatory_target_alignment_followup"

RUN_STATE: dict[str, Any] = {
    "stage": "NOT_STARTED",
    "runner_configuration_preflight": "NOT_RUN",
    "train_rollout_collection_count_this_run": 0,
    "diagnostic_rollout_collection_count_this_run": 0,
    "train_realization_action": "LOAD",
    "diagnostic_realization_action": "NOT_STARTED",
    "initialization_reproduction_gate": "NOT_RUN",
    "initialization_equivalence_gate": "NOT_RUN",
    "critic_steps_a": 0,
    "critic_steps_b": 0,
}

REQUIRED_ARTIFACTS = (
    "trajectory.csv",
    "vehicle_roles.json",
    "tasks.csv",
    "popularity.csv",
    "popularity_history.csv",
    "vehicle_predictions.csv",
    "popularity_predictions.csv",
    "prediction_scenarios.npz",
    "tv_local_cache.json",
    "sumo_demand.rou.xml",
    "sumo_demand.meta.json",
)


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _atomic_write_bytes(path, payload)


def _atomic_write_npz(path: Path, arrays: dict[str, Any]) -> None:
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    _atomic_write_bytes(path, buffer.getvalue())


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]).copy() for key in loaded.files}


def _recovery_archive_metadata(
    path: Path,
    arrays: dict[str, Any],
    *,
    kind: str,
    episode_seed: int,
    rollout_seed: int,
    actor_digest: str,
    initial_critic_digest: str,
    config_sha256: str,
    transition_count: int,
) -> dict[str, Any]:
    done_values = np.asarray(arrays["dones"], dtype=bool).reshape(-1)
    terminal_structure = {
        "terminal_count": int(np.count_nonzero(done_values)),
        "last_transition_terminal": bool(done_values[-1]) if len(done_values) else False,
        "earlier_transitions_terminal": bool(np.any(done_values[:-1])) if len(done_values) else False,
    }
    return {
        "artifact_type": kind,
        "episode_seed": int(episode_seed),
        "rollout_rng_seed": int(rollout_seed),
        "rollout_rng_provenance": {
            "episode_seed": int(episode_seed),
            "namespace": TRAIN_NAMESPACE if kind == "SHARED_TRAIN_ROLLOUT" else DIAGNOSTIC_NAMESPACE,
            "purpose": "rollout_collection",
            "derivation": "derive_namespaced_seed_v1",
        },
        "actor_parameter_digest": actor_digest,
        "initial_critic_parameter_digest": initial_critic_digest,
        "transition_count": int(transition_count),
        "collection_count": 1,
        "terminal_structure": terminal_structure,
        "arrays": {name: _array_record(value) for name, value in arrays.items()},
        "npz_sha256": _sha256_file(path),
        "config_sha256": config_sha256,
    }


def _persist_and_reload_recovery_archive(
    path: Path,
    metadata_path: Path,
    arrays: dict[str, Any],
    *,
    kind: str,
    episode_seed: int,
    rollout_seed: int,
    actor_digest: str,
    initial_critic_digest: str,
    config_sha256: str,
    transition_count: int = 300,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if path.exists():
        reloaded_arrays = _load_npz_arrays(path)
        if set(reloaded_arrays) != set(arrays):
            raise AssertionError(f"recovery archive schema changed: {path}")
        for name, expected in arrays.items():
            if _array_sha256(reloaded_arrays[name]) != _array_sha256(expected):
                raise AssertionError(f"recovery archive data differs from the current stage: {name}")
    else:
        _atomic_write_npz(path, arrays)
        reloaded_arrays = _load_npz_arrays(path)
    computed = _recovery_archive_metadata(
        path,
        reloaded_arrays,
        kind=kind,
        episode_seed=episode_seed,
        rollout_seed=rollout_seed,
        actor_digest=actor_digest,
        initial_critic_digest=initial_critic_digest,
        config_sha256=config_sha256,
        transition_count=transition_count,
    )
    if metadata_path.is_file():
        prior = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in (
            "artifact_type",
            "episode_seed",
            "rollout_rng_seed",
            "rollout_rng_provenance",
            "actor_parameter_digest",
            "initial_critic_parameter_digest",
            "transition_count",
            "collection_count",
            "arrays",
            "npz_sha256",
            "config_sha256",
        ):
            if prior.get(key) != computed.get(key):
                raise AssertionError(f"recovery metadata mismatch for {key}: {path}")
        metadata = prior
    else:
        metadata = computed
        _atomic_write_json(metadata_path, metadata)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("npz_sha256") != _sha256_file(path):
        raise AssertionError(f"recovery NPZ reload SHA256 mismatch: {path}")
    if metadata.get("arrays") != {name: _array_record(value) for name, value in reloaded_arrays.items()}:
        raise AssertionError(f"recovery NPZ array reload metadata mismatch: {path}")
    if any(
        not np.all(np.isfinite(value))
        for name, value in reloaded_arrays.items()
        if name != "dones" and name != "transition_149_pickle"
    ):
        raise FloatingPointError(f"recovery NPZ contains non-finite arrays: {path}")
    return reloaded_arrays, metadata


def _load_recovery_archive(
    path: Path,
    metadata_path: Path,
    *,
    kind: str,
    episode_seed: int,
    rollout_seed: int,
    actor_digest: str,
    initial_critic_digest: str,
    config_sha256: str,
    required_arrays: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    if not path.exists():
        if metadata_path.exists():
            raise FileNotFoundError(f"recovery metadata exists without its NPZ: {metadata_path}")
        return None
    arrays = _load_npz_arrays(path)
    if not required_arrays.issubset(arrays):
        raise AssertionError(f"recovery archive is missing required arrays: {path}")
    count = int(np.asarray(arrays["rewards"]).reshape(-1).shape[0])
    if count != 300 or np.asarray(arrays["observations"]).shape[0] != 300:
        raise AssertionError(f"recovery archive does not contain 300 transitions: {path}")
    dones = np.asarray(arrays["dones"], dtype=bool).reshape(-1)
    if not bool(dones[-1]) or bool(np.any(dones[:-1])):
        raise AssertionError(f"recovery archive terminal structure is invalid: {path}")
    current = _recovery_archive_metadata(
        path,
        arrays,
        kind=kind,
        episode_seed=episode_seed,
        rollout_seed=rollout_seed,
        actor_digest=actor_digest,
        initial_critic_digest=initial_critic_digest,
        config_sha256=config_sha256,
        transition_count=count,
    )
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in (
            "artifact_type",
            "episode_seed",
            "rollout_rng_seed",
            "rollout_rng_provenance",
            "actor_parameter_digest",
            "initial_critic_parameter_digest",
            "transition_count",
            "collection_count",
            "arrays",
            "npz_sha256",
            "config_sha256",
        ):
            if metadata.get(key) != current.get(key):
                raise AssertionError(f"recovery metadata mismatch for {key}: {path}")
    else:
        metadata = current
        _atomic_write_json(metadata_path, metadata)
    if metadata.get("episode_seed") != int(episode_seed):
        raise AssertionError("recovery archive episode seed differs from this experiment")
    if metadata.get("rollout_rng_seed") != int(rollout_seed):
        raise AssertionError("recovery archive rollout RNG seed differs from this experiment")
    if metadata.get("actor_parameter_digest") != actor_digest:
        raise AssertionError("recovery archive Actor digest differs from this initialization")
    if metadata.get("initial_critic_parameter_digest") != initial_critic_digest:
        raise AssertionError("recovery archive initial Critic digest differs from this initialization")
    if metadata.get("config_sha256") != config_sha256:
        raise AssertionError("recovery archive config SHA differs from this run")
    if any(
        not np.all(np.isfinite(value))
        for name, value in arrays.items()
        if name not in {"dones", "transition_149_pickle"}
    ):
        raise FloatingPointError(f"recovery archive contains non-finite values: {path}")
    return arrays, metadata


def _persist_targets(
    targets_a: np.ndarray,
    targets_b: np.ndarray,
    *,
    config_sha256: str,
    training_rollout_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, str], dict[str, Any]]:
    expected = {
        "arm_a_gae_lambda_return": np.asarray(targets_a, dtype=np.float32).reshape(-1),
        "arm_b_full_episode_mc_return": np.asarray(targets_b, dtype=np.float32).reshape(-1),
    }
    if any(len(value) != 300 or not np.all(np.isfinite(value)) for value in expected.values()):
        raise FloatingPointError("target arrays must each be 300 finite values")
    if TARGET_ARRAYS_PATH.exists():
        arrays = _load_npz_arrays(TARGET_ARRAYS_PATH)
        if set(arrays) != set(expected) or any(
            _array_sha256(arrays[name]) != _array_sha256(value)
            for name, value in expected.items()
        ):
            raise AssertionError("persisted recovery targets differ from the shared TRAIN rollout")
    else:
        _atomic_write_npz(TARGET_ARRAYS_PATH, expected)
        arrays = _load_npz_arrays(TARGET_ARRAYS_PATH)
    hashes = {name: _array_sha256(value) for name, value in arrays.items()}
    metadata = {
        "config_sha256": config_sha256,
        "training_rollout_npz_sha256": training_rollout_sha256,
        "arrays": {name: _array_record(value) for name, value in arrays.items()},
        "npz_sha256": _sha256_file(TARGET_ARRAYS_PATH),
    }
    if TARGET_ARRAYS_METADATA_PATH.is_file():
        saved = json.loads(TARGET_ARRAYS_METADATA_PATH.read_text(encoding="utf-8"))
        if saved != metadata:
            raise AssertionError("persisted target metadata differs from recomputed arrays")
    else:
        _atomic_write_json(TARGET_ARRAYS_METADATA_PATH, metadata)
    if _sha256_file(TARGET_ARRAYS_PATH) != metadata["npz_sha256"]:
        raise AssertionError("target NPZ reload SHA256 mismatch")
    if metadata["config_sha256"] != config_sha256:
        raise AssertionError("target recovery config SHA mismatch")
    return (
        arrays["arm_a_gae_lambda_return"],
        arrays["arm_b_full_episode_mc_return"],
        hashes,
        metadata,
    )


def _persist_minibatch_sequence(
    seed: int,
    sample_count: int,
    epochs: int,
    minibatches: int,
) -> tuple[list[np.ndarray], str, str]:
    if MINIBATCH_SEQUENCE_PATH.is_file():
        saved = json.loads(MINIBATCH_SEQUENCE_PATH.read_text(encoding="utf-8"))
        if int(saved.get("optimizer_rng_seed", -1)) != int(seed):
            raise AssertionError("persisted minibatch RNG seed differs from the experiment seed")
        sequence = [
            np.asarray(indices, dtype=np.int64)
            for indices in saved["minibatch_indices"]
        ]
        rng_sha = str(saved["initial_rng_state_sha256"])
    else:
        seed_state, sequence = _generate_minibatches(seed, sample_count, epochs, minibatches)
        rng_sha = hashlib.sha256(seed_state.numpy().tobytes()).hexdigest()
        saved = {
            "optimizer_rng_seed": int(seed),
            "initial_rng_state_sha256": rng_sha,
            "epochs": int(epochs),
            "minibatches_per_epoch": int(minibatches),
            "samples_per_minibatch": int(sample_count // minibatches),
            "minibatch_indices": [indices.astype(int).tolist() for indices in sequence],
            "minibatch_index_sequence_sha256": _minibatch_sequence_digest(sequence),
        }
        _atomic_write_json(MINIBATCH_SEQUENCE_PATH, saved)
        saved = json.loads(MINIBATCH_SEQUENCE_PATH.read_text(encoding="utf-8"))
        sequence = [np.asarray(indices, dtype=np.int64) for indices in saved["minibatch_indices"]]
    sequence_sha = _minibatch_sequence_digest(sequence)
    if len(sequence) != epochs * minibatches or any(
        len(indices) != sample_count // minibatches for indices in sequence
    ):
        raise AssertionError("persisted minibatch index sequence has the wrong dimensions")
    if saved.get("minibatch_index_sequence_sha256") != sequence_sha:
        raise AssertionError("minibatch index sequence SHA256 reload assertion failed")
    if not rng_sha:
        raise AssertionError("minibatch RNG state digest is missing")
    for epoch_index in range(epochs):
        epoch_indices = np.concatenate(sequence[epoch_index * minibatches : (epoch_index + 1) * minibatches])
        if not np.array_equal(np.sort(epoch_indices), np.arange(sample_count, dtype=np.int64)):
            raise AssertionError("persisted minibatch epoch is not a full sample permutation")
    return sequence, sequence_sha, rng_sha


def _historical_attempts() -> list[dict[str, Any]]:
    if REPORT_JSON.is_file():
        prior = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
        existing = prior.get("HISTORICAL_ATTEMPTS")
        if isinstance(existing, list) and existing:
            return existing
        return [
            {
                "attempt": "original_target_alignment_attempt",
                "result": "INCONCLUSIVE",
                "train_rollout_collection_count": 1,
                "critic_steps_a": 0,
                "critic_steps_b": 0,
                "diagnostic_rollout_collection_count": 0,
                "failure": "AttributeError: PPOV2Hyperparameters has no attribute 'minibatch_size'",
                "rollout_arrays_persisted": "NO",
            }
        ]
    raise FileNotFoundError("the original target-alignment attempt report is required")


def _write_followup_reports(report: dict[str, Any], *, completed: bool) -> None:
    historical = _historical_attempts()
    report["HISTORICAL_ATTEMPTS"] = historical
    report["CURRENT_ATTEMPT"] = {
        "attempt": CURRENT_ATTEMPT,
        "result": report.get("CRITIC_TARGET_ALIGNMENT_EXPERIMENT_RESULT", "INCONCLUSIVE"),
        "train_rollout_collection_count": int(RUN_STATE["train_rollout_collection_count_this_run"]),
        "critic_steps_a": int(RUN_STATE["critic_steps_a"]),
        "critic_steps_b": int(RUN_STATE["critic_steps_b"]),
        "diagnostic_rollout_collection_count": int(RUN_STATE["diagnostic_rollout_collection_count_this_run"]),
        "stage_at_completion_or_abort": RUN_STATE["stage"],
    }
    _atomic_write_json(REPORT_JSON, report)
    reloaded = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    if (
        reloaded.get("HISTORICAL_ATTEMPTS") != historical
        or reloaded.get("CURRENT_ATTEMPT", {}).get("attempt") != CURRENT_ATTEMPT
    ):
        raise AssertionError("formal result JSON history/reload assertion failed")
    if completed:
        rendered = _render_markdown(report).replace(
            "# Critic Target Alignment Minimal Experiment",
            "### Authorized Critic Target Alignment Follow-up",
            1,
        )
    else:
        rendered = "\n".join(
            [
                "### Authorized Critic Target Alignment Follow-up",
                "",
                f"- Result: **{report.get('CRITIC_TARGET_ALIGNMENT_EXPERIMENT_RESULT', 'INCONCLUSIVE')}**.",
                f"- Aborted at stage: `{RUN_STATE['stage']}`.",
                f"- Failure: `{report.get('EXECUTION_FAILURE', {}).get('message', 'unknown')}`.",
                f"- Training rollout collected during this run: `{RUN_STATE['train_rollout_collection_count_this_run']}`.",
                f"- Critic steps ARM_A / ARM_B: `{RUN_STATE['critic_steps_a']} / {RUN_STATE['critic_steps_b']}`.",
                f"- Recovery artifacts were retained: `{report.get('RECOVERY_ARTIFACT_STATUS', 'UNKNOWN')}`.",
                "- Formal PPO update / Epoch 1: NO; readiness: NO; production source and Critic target unchanged.",
                "",
            ]
        )
    report_text = REPORT_MD.read_text(encoding="utf-8") if REPORT_MD.is_file() else ""
    start_marker = "### Authorized Critic Target Alignment Follow-up"
    if start_marker in report_text:
        report_text = report_text.split(start_marker, 1)[0].rstrip()
    report_text = report_text.rstrip() + "\n\n" + rendered.rstrip() + "\n"
    _atomic_write_bytes(REPORT_MD, report_text.encode("utf-8"))

    section = rendered
    second_stage_path = ROOT / "reports" / "second_stage_joint_repair_gate.md"
    second_stage = second_stage_path.read_text(encoding="utf-8")
    heading = "### Authorized Critic Target Alignment Follow-up"
    if heading in second_stage:
        second_stage = second_stage.split(heading, 1)[0].rstrip()
    _atomic_write_bytes(
        second_stage_path,
        (second_stage.rstrip() + "\n\n" + section.rstrip() + "\n").encode("utf-8"),
    )

    one_epoch_path = ROOT / "reports" / "one_epoch_training_gate.md"
    one_epoch = one_epoch_path.read_text(encoding="utf-8")
    reason = (
        "ONE_EPOCH_NOT_EXECUTED_REASON = formal PPO epoch remains deferred to "
        f"{report.get('NEXT_STAGE', 'CRITIC_TARGET_ALIGNMENT_EXPERIMENT_FOLLOWUP')} "
        "after the authorized Critic target-alignment follow-up"
    )
    lines = one_epoch.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("ONE_EPOCH_NOT_EXECUTED_REASON ="):
            lines[index] = reason
            break
    one_epoch = "\n".join(lines).rstrip() + "\n"
    latest_title = "## Latest Critic Target Alignment Decision"
    latest = (
        latest_title
        + "\n\n"
        + f"Authorized compensatory Critic Target Alignment follow-up: `{report.get('CRITIC_TARGET_ALIGNMENT_EXPERIMENT_RESULT', 'INCONCLUSIVE')}`. "
        + f"Current blocker / next stage: `{report.get('NEXT_STAGE', 'CRITIC_TARGET_ALIGNMENT_EXPERIMENT_FOLLOWUP')}`. "
        + "Formal PPO update and Epoch 1 remain NO; readiness remains NO. Counterfactual Critic Reliability remains CONCERN, "
        + "Candidate-Q remains CANDIDATE_Q_INTEGRITY_CONCERN, and surrogate repair remains blocked.\n"
    )
    if latest_title in one_epoch:
        one_epoch = one_epoch.split(latest_title, 1)[0].rstrip() + "\n\n" + latest
    else:
        one_epoch = one_epoch.rstrip() + "\n\n" + latest
    _atomic_write_bytes(one_epoch_path, one_epoch.encode("utf-8"))


def _load_authoritative_reports() -> dict[str, Any]:
    report_paths = {
        "previous_target_alignment": REPORT_JSON,
        "critic_repair_design": ROOT / "reports" / "counterfactual_critic_repair_design.json",
        "critic_reliability": ROOT / "reports" / "counterfactual_critic_reliability.json",
        "candidate_q_integrity": ROOT / "reports" / "traceable_candidate_q_integrity_result.json",
    }
    loaded: dict[str, Any] = {}
    for name, path in report_paths.items():
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    continuation_path = ROOT / "reports" / "counterfactual_critic_reliability_continuations.jsonl"
    continuation_count = sum(
        1 for line in continuation_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    for path in (
        ROOT / "reports" / "counterfactual_critic_repair_design.md",
        ROOT / "reports" / "counterfactual_critic_reliability.json",
        ROOT / "reports" / "counterfactual_critic_reliability_continuations.jsonl",
        ROOT / "reports" / "second_stage_joint_repair_gate.md",
        ROOT / "reports" / "one_epoch_training_gate.md",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required experiment history is missing: {path}")
    loaded["continuation_record_count"] = continuation_count
    return loaded


def configuration_preflight() -> dict[str, Any]:
    """Read and resolve all runtime config fields without collecting a rollout."""

    RUN_STATE["stage"] = "CONFIGURATION_PREFLIGHT"
    RUN_STATE["runner_configuration_preflight"] = "RUNNING"
    if not CONFIG_PATH.is_file() or not SPLIT_MANIFEST_PATH.is_file():
        raise FileNotFoundError("formal config or data split manifest is missing")
    history = _load_authoritative_reports()
    generator = _load_single_episode_generator(ROOT)
    cfg = generator.load_and_validate_config(CONFIG_PATH)
    split_manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    generation_fingerprint = generator.generation_config_fingerprint(cfg)
    source_hashes = generator._source_artifact_hashes(cfg)
    if split_manifest.get("generation_config_fingerprint") != generation_fingerprint:
        raise ValueError("TRAIN split manifest conflicts with the current generation config")
    train_range = split_manifest["train"]
    if not int(train_range["seed_start"]) <= TRAIN_REALIZATION_SEED <= int(train_range["seed_end"]):
        raise ValueError("fixed TRAIN realization seed is outside the TRAIN split")
    if int(split_manifest.get("slots_per_realization", -1)) != 300:
        raise ValueError("TRAIN split does not define 300 slots per realization")
    for manifest_key, source_key in (
        ("service_catalog_sha256", "service_catalog_sha256"),
        ("popularity_gru_sha256", "popularity_gru_sha256"),
        ("vehicle_residual_bank_sha256", "vehicle_residual_bank_sha256"),
        ("sumo_network_sha256", "sumo_network_sha256"),
    ):
        if split_manifest.get(manifest_key) != source_hashes.get(source_key):
            raise ValueError(f"TRAIN split source artifact conflicts: {manifest_key}")
    _check_artifact_manifest(
        TRAIN_REALIZATION,
        TRAIN_REALIZATION_SEED,
        generation_fingerprint,
        source_hashes,
    )
    generator.validate_written_outputs(
        TRAIN_REALIZATION,
        task_vehicle_count=int(project_config.get_config_value(cfg, "simulation.num_task_vehicles")),
        research_vehicle_count=int(project_config.get_config_value(cfg, "simulation.num_research_vehicles")),
        horizon_steps=int(project_config.get_config_value(cfg, "prediction.vehicle.horizon")),
    )
    train_context = load_environment_context(str(CONFIG_PATH), realization_dir=TRAIN_REALIZATION)
    _validate_episode_realization_complete(
        TRAIN_REALIZATION, train_context, generator, TRAIN_REALIZATION_SEED
    )
    if train_context.data_protocol != "full_episode_randomized" or train_context.episode_slots != 300:
        raise ValueError("fixed TRAIN runtime protocol is not a complete 300-slot episode")
    if train_context.configured_rollout_steps != 300 or int(train_context.minibatch_size) != 30:
        raise ValueError("resolved runtime rollout/minibatch sizes differ from the authorized protocol")
    config_values = {
        "gamma": float(project_config.get_config_value(cfg, "algorithm.ppo.gamma")),
        "gae_lambda": float(project_config.get_config_value(cfg, "algorithm.ppo.gae_lambda")),
        "critic_learning_rate": float(project_config.get_config_value(cfg, "algorithm.ppo.critic_learning_rate")),
    }
    base_seed = int(project_config.get_config_value(cfg, "experiment.seed"))
    initialization_seed = derive_namespaced_seed(base_seed, DIAGNOSTIC_NAMESPACE, "trainer_initialization")
    trainer = _run_with_isolated_rng(
        initialization_seed,
        lambda: PPOV2Trainer(train_context, device="cpu", exact_q_parallel_workers=1),
    )
    hyper = trainer.hyperparameters
    runtime_values = {
        **config_values,
        "minibatch_size": int(train_context.minibatch_size),
        "update_epochs": int(hyper.update_epochs),
        "num_minibatches": int(hyper.num_minibatches),
        "critic_learning_rate_resolved": float(hyper.critic_learning_rate),
        "max_grad_norm": float(hyper.max_grad_norm),
        "critic_value_scale_epsilon": float(hyper.critic_value_scale_epsilon),
        "critic_value_scale_warmup_samples": int(hyper.critic_value_scale_warmup_samples),
        "initial_critic_digest": trainer._critic_parameter_digest(),
        "actor_optimizer_steps": int(trainer.actor_optimizer_steps),
        "critic_optimizer_steps": int(trainer.critic_optimizer_steps),
    }
    if runtime_values["gamma"] != 1.0 or runtime_values["gae_lambda"] != 0.95:
        raise AssertionError("gamma or Actor GAE lambda differs from the authorized experiment")
    if (
        runtime_values["minibatch_size"] != 30
        or runtime_values["update_epochs"] != 4
        or runtime_values["num_minibatches"] != 10
        or runtime_values["critic_learning_rate"] != 0.001
        or runtime_values["critic_learning_rate_resolved"] != 0.001
        or runtime_values["max_grad_norm"] != 0.5
    ):
        raise AssertionError("resolved Critic training configuration differs from the authorization")
    if runtime_values["initial_critic_digest"] != EXPECTED_INITIAL_CRITIC_DIGEST:
        RUN_STATE["initialization_reproduction_gate"] = "FAIL"
        raise AssertionError("fresh Critic digest does not reproduce the required initialization")
    if runtime_values["actor_optimizer_steps"] != 0 or runtime_values["critic_optimizer_steps"] != 0:
        raise AssertionError("preflight trainer started with non-zero optimizer steps")
    RUN_STATE["initialization_reproduction_gate"] = "PASS"
    RUN_STATE["runner_configuration_preflight"] = "PASS"
    RUN_STATE["stage"] = "CONFIGURATION_PREFLIGHT_PASS"
    return {
        "generator": generator,
        "cfg": cfg,
        "split_manifest": split_manifest,
        "generation_fingerprint": generation_fingerprint,
        "source_hashes": source_hashes,
        "train_context": train_context,
        "trainer": trainer,
        "history": history,
        "config_values": runtime_values,
        "initialization_seed": initialization_seed,
    }


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_hash(items: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items, key=lambda item: item[0]):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _actor_digest(policy: Any) -> str:
    named = dict(policy.named_parameters())
    actor_ids = {id(parameter) for parameter in policy.actor_parameters()}
    values = [(name, parameter) for name, parameter in named.items() if id(parameter) in actor_ids]
    if len(values) != len(actor_ids):
        raise RuntimeError("Actor digest does not include every actor parameter")
    # Match the digest encoding used by the persisted traceable Candidate-Q
    # diagnostic so the fixed UAV_1 transition can be reproduced exactly.
    digest = hashlib.sha256()
    for name, tensor in sorted(values, key=lambda item: item[0]):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _critic_digest(policy: Any) -> str:
    critic_parameters = tuple(policy.critic_parameters())
    critic_ids = {id(parameter) for parameter in critic_parameters}
    values = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if id(parameter) in critic_ids
    ]
    if len(values) != len(critic_parameters):
        raise RuntimeError("Critic digest does not include every critic parameter")
    for name, buffer in policy.named_buffers():
        if name in {"popart_mean", "popart_std"}:
            values.append((name, buffer))
    return _tensor_hash(values)


def _critic_state_dict(policy: Any) -> dict[str, torch.Tensor]:
    critic_ids = {id(parameter) for parameter in policy.critic_parameters()}
    named_parameters = dict(policy.named_parameters())
    selected_names = {
        name for name, parameter in named_parameters.items() if id(parameter) in critic_ids
    }
    selected_names.update({"popart_mean", "popart_std"})
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in policy.state_dict().items()
        if name in selected_names
    }


def _welford_record(statistics: RunningReturnStatistics) -> dict[str, Any]:
    return {
        "count": int(statistics.count),
        "mean": float(statistics.mean),
        "m2": float(statistics.m2),
    }


def _optimizer_record(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = optimizer.state_dict()
    groups = []
    for group in state["param_groups"]:
        groups.append(
            {
                **{key: value for key, value in group.items() if key != "params"},
                "parameter_count": len(group["params"]),
            }
        )
    return {"state_entry_count": len(state["state"]), "param_groups": groups}


def _optimizer_state_hash(optimizer: torch.optim.Optimizer) -> str:
    return _json_hash(_optimizer_record(optimizer))


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_record(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("array contains NaN or Inf")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _array_sha256(array),
    }


def _mc_reward_to_go(rewards: Any, dones: Any, gamma: float) -> np.ndarray:
    reward_array = np.asarray(rewards, dtype=np.float32).reshape(-1)
    done_array = np.asarray(dones, dtype=bool).reshape(-1)
    if not len(reward_array) or len(reward_array) != len(done_array):
        raise ValueError("MC return inputs must be equal-length, non-empty arrays")
    if not np.all(np.isfinite(reward_array)):
        raise FloatingPointError("MC return rewards contain NaN or Inf")
    result = np.zeros_like(reward_array, dtype=np.float32)
    running = 0.0
    for index in range(len(reward_array) - 1, -1, -1):
        if bool(done_array[index]):
            running = 0.0
        running = float(reward_array[index]) + float(gamma) * running
        result[index] = running
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("MC returns contain NaN or Inf")
    expected = np.empty_like(result)
    for index in range(len(result)):
        next_return = 0.0 if bool(done_array[index]) or index + 1 == len(result) else float(result[index + 1])
        expected[index] = float(reward_array[index]) + float(gamma) * next_return
    if not np.allclose(result, expected, rtol=1.0e-6, atol=1.0e-5):
        raise AssertionError("MC terminal recursion assertion failed")
    return result


def _rankdata_average(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def _correlation(left: Any, right: Any, *, spearman: bool = False) -> float | str:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(x) != len(y) or not len(x) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise FloatingPointError("correlation arrays are empty, misaligned, or non-finite")
    if spearman:
        x = _rankdata_average(x)
        y = _rankdata_average(y)
    if float(np.std(x)) <= np.finfo(np.float64).eps or float(np.std(y)) <= np.finfo(np.float64).eps:
        return "NOT_MEANINGFUL"
    value = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(value):
        raise FloatingPointError("correlation is non-finite")
    return value


def _metrics(prediction: Any, target: Any) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1)
    expected = np.asarray(target, dtype=np.float64).reshape(-1)
    if len(predicted) != len(expected) or not len(expected):
        raise ValueError("prediction and target must be equal-length and non-empty")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(expected)):
        raise FloatingPointError("prediction or target contains NaN or Inf")
    residual = predicted - expected
    target_variance = float(np.var(expected))
    if target_variance <= 1.0e-12 * max(1.0, float(np.mean(expected) ** 2)):
        explained_variance: float | str = "NOT_MEANINGFUL"
    else:
        explained_variance = float(1.0 - np.var(residual) / target_variance)
        if not math.isfinite(explained_variance):
            raise FloatingPointError("explained variance is non-finite")
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "explained_variance": explained_variance,
        "pearson": _correlation(predicted, expected),
        "spearman": _correlation(predicted, expected, spearman=True),
        "sample_count": int(len(expected)),
    }


def _sign(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise FloatingPointError("Delta-V is non-finite")
    if number < 0.0:
        return "NEGATIVE"
    if number > 0.0:
        return "POSITIVE"
    return "ZERO"


def _task_record(task: Any) -> dict[str, Any]:
    return {
        "task_id": str(task.task_id),
        "vehicle_id": str(task.vehicle_id),
        "service_id": int(task.service_id),
        "input_bits": float(task.input_bits),
        "cpu_cycles": float(task.cpu_cycles),
    }


def _position_record(positions: dict[str, Any]) -> dict[str, list[float]]:
    return {
        str(key): np.asarray(value, dtype=np.float64).reshape(-1).tolist()
        for key, value in sorted(positions.items(), key=lambda item: str(item[0]))
    }


def _transition_fingerprint(transition: Any, step: Any, config_sha: str) -> str:
    payload = {
        "transition_index": int(TARGET_TRANSITION),
        "origin_index": int(transition.origin_index),
        "slot": int(step.slot),
        "action": np.asarray(transition.action, dtype=np.int8).astype(int).tolist(),
        "previous_cache_state": np.asarray(
            transition.previous_cache_state, dtype=np.int8
        ).astype(int).tolist(),
        "production_next_cache_state": np.asarray(
            transition.next_cache_state, dtype=np.int8
        ).astype(int).tolist(),
        "previous_uav_positions": _position_record(transition.previous_uav_positions),
        "production_next_uav_positions": _position_record(transition.next_uav_positions),
        "actual_tasks": [_task_record(task) for task in step.actual_tasks],
        "actual_vehicle_positions": _position_record(step.actual_vehicle_positions),
        "config_sha256": config_sha,
    }
    return _json_hash(payload)


def _check_artifact_manifest(
    directory: Path,
    episode_seed: int,
    current_fingerprint: str,
    expected_source_hashes: dict[str, str],
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"realization is missing required artifacts: {missing}")
    for name in REQUIRED_ARTIFACTS:
        with (directory / name).open("rb") as stream:
            stream.read(1)
    manifest_path = directory / "episode_manifest.json"
    manifest_status = "MISSING_OPTIONAL"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest_status = "READ_OPTIONAL"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("episode_seed", -1)) != int(episode_seed):
            raise ValueError("episode_manifest episode seed conflicts with selected realization")
        if manifest.get("generation_config_fingerprint") != current_fingerprint:
            raise ValueError("episode_manifest generation fingerprint conflicts with current config")
        if int(manifest.get("simulation", {}).get("episode_slots", -1)) != 300:
            raise ValueError("episode_manifest slot count conflicts with the 300-slot protocol")
        expected_artifacts = manifest.get("generated_artifact_sha256", {})
        for name, expected_hash in expected_artifacts.items():
            artifact_path = directory / name
            if not artifact_path.is_file() or _sha256_file(artifact_path) != expected_hash:
                raise ValueError(f"episode_manifest artifact hash mismatch: {name}")
        source_pairs = (
            ("service_catalog_sha256", "service_catalog_sha256"),
            ("popularity_gru_sha256", "popularity_gru_sha256"),
            ("vehicle_residual_bank_sha256", "vehicle_residual_bank_sha256"),
            ("sumo_network_sha256", "sumo_network_sha256"),
        )
        for manifest_key, current_key in source_pairs:
            if manifest.get(manifest_key) != expected_source_hashes.get(current_key):
                raise ValueError(f"episode_manifest source artifact conflict: {manifest_key}")
        if manifest.get("derived_episode_seed_streams") != derive_episode_seeds(episode_seed):
            raise ValueError("episode_manifest derived seed streams conflict with episode_seed_v1")
    return {
        "episode_manifest_status": manifest_status,
        "episode_manifest_sha256": _sha256_file(manifest_path) if manifest_path.is_file() else None,
        "derived_episode_seed_streams": derive_episode_seeds(episode_seed),
    }


def _realization_provenance(
    *,
    directory: Path,
    episode_seed: int,
    split: str,
    action: str,
    current_config_sha: str,
    current_fingerprint: str,
    split_manifest: dict[str, Any] | None,
    generator_module: Any,
    cfg: dict[str, Any],
    isolated_generation_rng_seed: int | None,
) -> dict[str, Any]:
    source_hashes = generator_module._source_artifact_hashes(cfg)
    optional_manifest = _check_artifact_manifest(
        directory, episode_seed, current_fingerprint, source_hashes
    )
    for filename in REQUIRED_ARTIFACTS:
        _sha256_file(directory / filename)
    split_evidence = "DETERMINISTIC_DIAGNOSTIC_TRAIN_PATH"
    if split_manifest is not None:
        train = split_manifest["train"]
        if not int(train["seed_start"]) <= int(episode_seed) <= int(train["seed_end"]):
            raise ValueError("selected realization seed is outside the TRAIN seed range")
        if int(split_manifest.get("slots_per_realization", -1)) != 300:
            raise ValueError("data split manifest does not define 300 slots")
        if split_manifest.get("generation_config_fingerprint") != current_fingerprint:
            raise ValueError("data split manifest generation fingerprint conflicts with current config")
        split_source_map = {
            "service_catalog_sha256": "service_catalog_sha256",
            "popularity_gru_sha256": "popularity_gru_sha256",
            "vehicle_residual_bank_sha256": "vehicle_residual_bank_sha256",
            "sumo_network_sha256": "sumo_network_sha256",
        }
        for manifest_key, source_key in split_source_map.items():
            if split_manifest.get(manifest_key) != source_hashes.get(source_key):
                raise ValueError(f"data_split_manifest source hash conflicts: {manifest_key}")
        split_evidence = "DATA_SPLIT_MANIFEST_TRAIN_SEED_RANGE"
    return {
        "split": split,
        "split_evidence": split_evidence,
        "action": action,
        "path": str(directory.resolve()),
        "episode_seed": int(episode_seed),
        "generation_rng_seed": (
            {
                "isolated_episode_generation_rng_seed": int(isolated_generation_rng_seed),
                "episode_seed_v1_streams": optional_manifest["derived_episode_seed_streams"],
            }
            if isolated_generation_rng_seed is not None
            else {
                "protocol": "adaptive_cache_mec_episode_seed_v1",
                "episode_seed_v1_streams": optional_manifest["derived_episode_seed_streams"],
            }
        ),
        "generation_config_fingerprint": current_fingerprint,
        "config_sha256_at_diagnostic_run": current_config_sha,
        "data_protocol": "full_episode_randomized",
        "episode_slots": 300,
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "required_artifacts_status": "PASS",
        **optional_manifest,
        "explicit_compatibility_conflicts": [],
        "manifest_is_required_for_reuse": "NO",
    }


def _candidate_realizations(roots: tuple[Path, ...], episode_seed: int) -> list[Path]:
    name = f"episode_{int(episode_seed)}"
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob(name):
            if path.is_dir() and path not in found:
                found.append(path)
    return found


def _find_diagnostic_realization(
    generator_module: Any,
    cfg: dict[str, Any],
    current_fingerprint: str,
    context: Any,
) -> Path | None:
    roots = (ROOT / "environment_data", ROOT / "runs" / "diagnostics")
    source_hashes = generator_module._source_artifact_hashes(cfg)
    for candidate in _candidate_realizations(roots, DIAGNOSTIC_REALIZATION_SEED):
        normalized = str(candidate).lower()
        if "proposed_joint_validation" in normalized or "proposed_joint_test" in normalized:
            continue
        if "proposed_joint_training" not in normalized and "diagnostic_train_realization" not in normalized:
            continue
        try:
            _check_artifact_manifest(
                candidate,
                DIAGNOSTIC_REALIZATION_SEED,
                current_fingerprint,
                source_hashes,
            )
            _validate_episode_realization_complete(
                candidate, context, generator_module, DIAGNOSTIC_REALIZATION_SEED
            )
        except Exception:
            continue
        return candidate
    return None


def _generate_minibatches(seed: int, sample_count: int, epochs: int, minibatches: int) -> tuple[torch.Tensor, list[np.ndarray]]:
    if sample_count % minibatches != 0:
        raise ValueError("sample count must be divisible by minibatch count")
    initial_generator = torch.Generator(device="cpu")
    initial_generator.manual_seed(int(seed))
    initial_state = initial_generator.get_state().clone()

    def build_sequence() -> list[np.ndarray]:
        generator = torch.Generator(device="cpu")
        generator.set_state(initial_state.clone())
        sequence: list[np.ndarray] = []
        size = sample_count // minibatches
        for _epoch in range(epochs):
            permutation = torch.randperm(sample_count, generator=generator, device="cpu")
            for start in range(0, sample_count, size):
                sequence.append(permutation[start : start + size].numpy().astype(np.int64, copy=True))
        return sequence

    return initial_state, build_sequence()


def _minibatch_sequence_digest(sequence: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for index, minibatch in enumerate(sequence):
        array = np.ascontiguousarray(minibatch.astype("<i8", copy=False))
        digest.update(index.to_bytes(4, "little", signed=False))
        digest.update(len(array).to_bytes(4, "little", signed=False))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _make_critic_arm(
    base_trainer: PPOV2Trainer,
    initial_optimizer_state: dict[str, Any],
    device: str,
) -> SimpleNamespace:
    policy = copy.deepcopy(base_trainer.policy).to(device)
    for parameter in policy.actor_parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        policy.critic_parameters(), lr=float(base_trainer.effective_critic_learning_rate)
    )
    optimizer.load_state_dict(copy.deepcopy(initial_optimizer_state))
    return SimpleNamespace(
        policy=policy,
        critic_optimizer=optimizer,
        critic_return_statistics=RunningReturnStatistics(),
        hyperparameters=base_trainer.hyperparameters,
        device=torch.device(device),
        critic_optimizer_steps=0,
    )


def _assert_initialization_equivalence(
    arm_a: Any,
    arm_b: Any,
    initial_optimizer_record: dict[str, Any],
    expected_critic_digest: str,
    expected_actor_digest: str,
) -> dict[str, Any]:
    actor_a = _actor_digest(arm_a.policy)
    actor_b = _actor_digest(arm_b.policy)
    critic_a = _critic_digest(arm_a.policy)
    critic_b = _critic_digest(arm_b.policy)
    optimizer_a = _optimizer_record(arm_a.critic_optimizer)
    optimizer_b = _optimizer_record(arm_b.critic_optimizer)
    stats_a = _welford_record(arm_a.critic_return_statistics)
    stats_b = _welford_record(arm_b.critic_return_statistics)
    popart_a = list(arm_a.policy.popart_statistics())
    popart_b = list(arm_b.policy.popart_statistics())
    checks = {
        "critic_digest_equal": critic_a == critic_b == expected_critic_digest,
        "actor_digest_equal": actor_a == actor_b == expected_actor_digest,
        "optimizer_state_equal": optimizer_a == optimizer_b == initial_optimizer_record,
        "optimizer_step_count_equal": arm_a.critic_optimizer_steps == arm_b.critic_optimizer_steps == 0,
        "popart_equal": popart_a == popart_b == [0.0, 1.0],
        "welford_equal": stats_a == stats_b == {"count": 0, "mean": 0.0, "m2": 0.0},
    }
    if not all(checks.values()):
        raise AssertionError(f"Critic initialization equivalence failed: {checks}")
    return {
        "status": "PASS",
        "checks": checks,
        "initial_critic_parameter_digest": critic_a,
        "initial_actor_parameter_digest": actor_a,
        "initial_optimizer_state": optimizer_a,
        "initial_optimizer_state_sha256": _json_hash(optimizer_a),
        "initial_optimizer_steps": 0,
        "initial_popart": {"mean": popart_a[0], "std": popart_a[1]},
        "initial_welford": stats_a,
    }


def _prepare_arm_targets(
    arm: Any,
    targets: np.ndarray,
    training_observations: np.ndarray,
    epsilon: float,
    warmup_samples: int,
) -> dict[str, Any]:
    with torch.no_grad():
        observations = torch.as_tensor(
            training_observations, dtype=torch.float32, device=arm.device
        )
        raw_before = arm.policy.critic_value(observations).detach().cpu().numpy().copy()
    arm.critic_return_statistics.update(targets)
    mean, std = arm.critic_return_statistics.active_statistics(epsilon, warmup_samples)
    arm.policy.update_popart_statistics(mean, std)
    with torch.no_grad():
        raw_after = arm.policy.critic_value(observations).detach().cpu().numpy().copy()
    if not np.all(np.isfinite(raw_before)) or not np.all(np.isfinite(raw_after)):
        raise FloatingPointError("PopArt raw value preservation check found NaN or Inf")
    max_abs_delta = float(np.max(np.abs(raw_after.astype(np.float64) - raw_before.astype(np.float64))))
    if not np.allclose(raw_before, raw_after, rtol=1.0e-5, atol=1.0e-4):
        raise AssertionError(f"PopArt did not preserve raw values: max_abs_delta={max_abs_delta}")
    normalized_targets = (np.asarray(targets, dtype=np.float32) - float(mean)) / float(std)
    if not np.all(np.isfinite(normalized_targets)):
        raise FloatingPointError("normalized Critic targets contain NaN or Inf")
    return {
        "popart_mean": float(mean),
        "popart_std": float(std),
        "welford": _welford_record(arm.critic_return_statistics),
        "raw_value_preservation_finite": "YES",
        "raw_value_preservation_max_abs_delta": max_abs_delta,
        "normalized_targets": normalized_targets.astype(np.float32, copy=False),
    }


def _train_arm(
    arm: Any,
    observations: np.ndarray,
    returns: np.ndarray,
    normalized_targets: np.ndarray,
    sequence: list[np.ndarray],
    critic_steps_expected: int,
) -> dict[str, Any]:
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=arm.device)
    return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=arm.device)
    target_tensor = torch.as_tensor(normalized_targets, dtype=torch.float32, device=arm.device)
    actor_parameters = list(arm.policy.actor_parameters())
    losses: list[float] = []
    gradient_norms: list[float] = []
    actor_digest_before = _actor_digest(arm.policy)
    for indices_array in sequence:
        indices = torch.as_tensor(indices_array, dtype=torch.long, device=arm.device)
        raw_loss, normalized_loss, gradient_norm = PPOV2Trainer._critic_minibatch_update(
            arm,
            observation_tensor[indices],
            return_tensor[indices],
            target_tensor[indices],
            float(arm.policy.popart_statistics()[1]),
            popart_critic=True,
            actor_parameters=actor_parameters,
        )
        if not bool(torch.isfinite(raw_loss).item()) or not bool(torch.isfinite(normalized_loss).item()):
            raise FloatingPointError("Critic minibatch update produced a non-finite loss")
        losses.append(float(normalized_loss.detach().cpu()))
        gradient_norms.append(float(gradient_norm))
    actor_digest_after = _actor_digest(arm.policy)
    if actor_digest_before != actor_digest_after:
        raise AssertionError("Actor parameters changed during Critic-only training")
    if int(arm.critic_optimizer_steps) != int(critic_steps_expected):
        raise AssertionError(
            f"Critic step count is {arm.critic_optimizer_steps}, expected {critic_steps_expected}"
        )
    return {
        "critic_optimizer_steps": int(arm.critic_optimizer_steps),
        "actor_optimizer_steps": 0,
        "actor_parameter_digest_before": actor_digest_before,
        "actor_parameter_digest_after": actor_digest_after,
        "actor_parameter_delta": 0.0,
        "normalized_value_loss_mean": float(np.mean(losses)),
        "critic_gradient_norm_max": float(np.max(gradient_norms)),
        "minibatch_index_sequence_sha256": _minibatch_sequence_digest(sequence),
    }


def _evaluate_values(policy: Any, observations: np.ndarray, device: str) -> np.ndarray:
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    with torch.no_grad():
        values = policy.critic_value(observation_tensor)
    result = values.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    if len(result) != len(observations) or not np.all(np.isfinite(result)):
        raise FloatingPointError("Critic returned invalid actual-state predictions")
    return result


def _save_critic_checkpoint(
    *,
    arm: Any,
    arm_name: str,
    target_type: str,
    target_bootstrap: str,
    target_sha256: str,
    sequence_sha256: str,
    train_provenance: dict[str, Any],
    seeds: dict[str, int],
    config_sha256: str,
    initial_critic_digest: str,
    actor_digest: str,
) -> dict[str, Any]:
    stem = "gae_target_critic" if arm_name == "ARM_A" else "mc_target_critic"
    checkpoint_path = OUTPUT_DIR / f"{stem}.pt"
    metadata_path = OUTPUT_DIR / f"{stem}.metadata.json"
    if checkpoint_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic checkpoint: {checkpoint_path}")
    critic_state = _critic_state_dict(arm.policy)
    digest = _critic_digest(arm.policy)
    popart_mean, popart_std = arm.policy.popart_statistics()
    metadata = {
        "checkpoint_schema": "critic_target_alignment_diagnostic_v1",
        "checkpoint_version": CRITIC_CHECKPOINT_SCHEMA_VERSION,
        "production_checkpoint_version": int(JOINT_PROPOSED_CHECKPOINT_VERSION),
        "architecture_version": int(arm.policy.architecture_version),
        "architecture_id": str(arm.policy.architecture_id),
        "target_type": target_type,
        "target_bootstrap": target_bootstrap,
        "critic_digest": digest,
        "initial_critic_digest": initial_critic_digest,
        "actor_parameter_digest": actor_digest,
        "critic_optimizer_steps": int(arm.critic_optimizer_steps),
        "critic_optimizer_state_dict": arm.critic_optimizer.state_dict(),
        "popart": {"mean": float(popart_mean), "std": float(popart_std)},
        "welford": _welford_record(arm.critic_return_statistics),
        "training_realization_seed": int(train_provenance["episode_seed"]),
        "training_realization_path": str(train_provenance["path"]),
        "training_rollout_seed": int(seeds["training_rollout_seed"]),
        "optimizer_seed": int(seeds["optimizer_seed"]),
        "minibatch_index_sequence_sha256": sequence_sha256,
        "target_array_sha256": target_sha256,
        "config_sha256": config_sha256,
        "DIAGNOSTIC_ONLY": "YES",
    }
    checkpoint_payload = {
        **metadata,
        "critic_state_dict": critic_state,
    }
    checkpoint_buffer = BytesIO()
    torch.save(checkpoint_payload, checkpoint_buffer)
    _atomic_write_bytes(checkpoint_path, checkpoint_buffer.getvalue())
    metadata["checkpoint_sha256"] = _sha256_file(checkpoint_path)
    metadata_json = {
        key: value
        for key, value in metadata.items()
        if key != "critic_optimizer_state_dict"
    }
    _atomic_write_json(metadata_path, metadata_json)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        reloaded.get("critic_digest") != digest
        or int(reloaded.get("critic_optimizer_steps", -1)) != 40
        or _tensor_hash(list(reloaded["critic_state_dict"].items()))
        != _tensor_hash(list(critic_state.items()))
    ):
        raise AssertionError(f"{arm_name} checkpoint reload gate failed")
    reloaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if reloaded_metadata.get("checkpoint_sha256") != _sha256_file(checkpoint_path):
        raise AssertionError(f"{arm_name} checkpoint metadata reload gate failed")
    return {
        **metadata_json,
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(metadata_path),
        "critic_state_keys": sorted(critic_state),
        "checkpoint_reload_gate": "PASS",
    }


def _restore_checkpoint_into_arm(
    *,
    checkpoint_path: Path,
    metadata_path: Path,
    arm: Any,
    target_type: str,
    target_sha256: str,
    sequence_sha256: str,
    expected_initial_critic_digest: str,
    expected_actor_digest: str,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "target_type": target_type,
        "target_array_sha256": target_sha256,
        "minibatch_index_sequence_sha256": sequence_sha256,
        "initial_critic_digest": expected_initial_critic_digest,
        "actor_parameter_digest": expected_actor_digest,
        "DIAGNOSTIC_ONLY": "YES",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise AssertionError(f"checkpoint recovery metadata mismatch for {key}")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("checkpoint_sha256") != _sha256_file(checkpoint_path):
            raise AssertionError("checkpoint SHA256 differs from its JSON metadata")
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise AssertionError(f"checkpoint JSON metadata mismatch for {key}")
    else:
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"critic_optimizer_state_dict", "critic_state_dict"}
        }
        metadata["checkpoint_sha256"] = _sha256_file(checkpoint_path)
        _atomic_write_json(metadata_path, metadata)
    expected_popart = list(arm.policy.popart_statistics())
    expected_welford = _welford_record(arm.critic_return_statistics)
    missing, unexpected = arm.policy.load_state_dict(payload["critic_state_dict"], strict=False)
    if unexpected:
        raise AssertionError(f"unexpected checkpoint state keys: {unexpected}")
    if not set(payload["critic_state_dict"]).issubset(set(arm.policy.state_dict())):
        raise AssertionError("Critic checkpoint contains a key absent from the current architecture")
    arm.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
    arm.critic_optimizer_steps = int(payload["critic_optimizer_steps"])
    saved_popart = payload["popart"]
    saved_welford = payload["welford"]
    current_popart = list(arm.policy.popart_statistics())
    current_welford = _welford_record(arm.critic_return_statistics)
    saved_popart_values = [float(saved_popart["mean"]), float(saved_popart["std"])]
    if expected_popart != saved_popart_values or current_popart != saved_popart_values:
        raise AssertionError("checkpoint reload PopArt state differs from arm preparation")
    if expected_welford != saved_welford or current_welford != saved_welford:
        raise AssertionError("checkpoint reload Welford state differs from arm preparation")
    digest = _critic_digest(arm.policy)
    if digest != payload.get("critic_digest") or arm.critic_optimizer_steps != 40:
        raise AssertionError("checkpoint reload Critic digest or step count failed")
    if _actor_digest(arm.policy) != expected_actor_digest:
        raise AssertionError("checkpoint reload changed the frozen Actor")
    if metadata.get("checkpoint_schema") != "critic_target_alignment_diagnostic_v1":
        raise AssertionError("checkpoint reload schema is not diagnostic-only v1")
    metadata["checkpoint_reload_gate"] = "PASS"
    return metadata


def _classify_result(arm_a: dict[str, Any], arm_b: dict[str, Any], branch: dict[str, Any], gates_pass: bool) -> tuple[str, str]:
    if not gates_pass or branch.get("branch_reproduction_gate") != "PASS":
        return "INCONCLUSIVE", "A required fairness, finiteness, realization, or branch reproduction gate failed."
    a_mae = float(arm_a["metrics"]["mae"])
    b_mae = float(arm_b["metrics"]["mae"])
    a_rmse = float(arm_a["metrics"]["rmse"])
    b_rmse = float(arm_b["metrics"]["rmse"])
    a_ev = float(arm_a["metrics"]["explained_variance"])
    b_ev = float(arm_b["metrics"]["explained_variance"])
    b_negative = branch["arm_b_delta_v_sign"] == "NEGATIVE"
    all_metrics_improve = b_mae < a_mae and b_rmse < a_rmse and b_ev > a_ev
    if all_metrics_improve and b_negative:
        return "SUPPORTED", "MC target improved all three actual-state metrics and matched the fixed negative UAV_1 witness direction."
    no_metric_improves = b_mae >= a_mae and b_rmse >= a_rmse and b_ev <= a_ev
    if no_metric_improves and not b_negative:
        return "NOT_SUPPORTED", "MC target improved none of MAE, RMSE, or EV, and the UAV_1 witness direction remained wrong."
    if all_metrics_improve or b_negative:
        return "PARTIAL", "The target change improved either the actual-state metric set or the fixed UAV_1 sign, but not both required groups."
    return "INCONCLUSIVE", "The mixed metric and witness result does not meet the defined supported, partial, or not-supported cases."


def _render_markdown(result: dict[str, Any]) -> str:
    a = result["ARM_A"]
    b = result["ARM_B"]
    witness = result["UAV1_WITNESS"]
    lines = [
        "# Critic Target Alignment Minimal Experiment",
        "",
        f"- Result: **{result['CRITIC_TARGET_ALIGNMENT_EXPERIMENT_RESULT']}**",
        f"- Next stage: `{result['NEXT_STAGE']}`",
        f"- Current HEAD: `{result['CURRENT_HEAD']}`",
        f"- Training realization: `{result['TRAIN_REALIZATION']['path']}` (seed {result['TRAIN_REALIZATION']['episode_seed']}, {result['TRAIN_REALIZATION_ACTION']})",
        f"- Diagnostic realization: seed {result['DIAGNOSTIC_REALIZATION']['episode_seed']} ({result['DIAGNOSTIC_REALIZATION_ACTION']})",
        "",
        "## Fairness and execution gates",
        "",
        f"- Initialization equivalence: **{result['INITIALIZATION_EQUIVALENCE_GATE']}**; initial Critic digest `{result['INITIAL_CRITIC_PARAMETER_DIGEST']}`.",
        f"- Minibatch order equivalence: **{result['MINIBATCH_ORDER_EQUIVALENCE_GATE']}**; SHA256 `{result['MINIBATCH_INDEX_SEQUENCE_SHA256_ARM_A']}`.",
        f"- Critic steps: ARM_A {result['ARM_A_CRITIC_STEPS']}, ARM_B {result['ARM_B_CRITIC_STEPS']}; Actor optimizer steps 0; Actor delta 0.",
        f"- Train and diagnostic rollout collection counts: {result['TRAIN_ROLLOUT_COLLECTION_COUNT']} and {result['DIAGNOSTIC_ROLLOUT_COLLECTION_COUNT']}.",
        "- Formal PPO update: NO; Epoch 1: NO; production Critic target changed: NO.",
        "",
        "## Independent TRAIN actual-state evaluation",
        "",
        "| Arm | Target | MC MAE | MC RMSE | MC EV | Pearson | Spearman |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| ARM_A | GAE_LAMBDA_RETURN | {a['metrics']['mae']:.6f} | {a['metrics']['rmse']:.6f} | {a['metrics']['explained_variance']:.6f} | {a['metrics']['pearson']} | {a['metrics']['spearman']} |",
        f"| ARM_B | FULL_EPISODE_MC_RETURN | {b['metrics']['mae']:.6f} | {b['metrics']['rmse']:.6f} | {b['metrics']['explained_variance']:.6f} | {b['metrics']['pearson']} | {b['metrics']['spearman']} |",
        "",
        f"- MC MAE improvement (A − B): {result['MC_MAE_IMPROVEMENT']:.6f}.",
        f"- MC RMSE improvement (A − B): {result['MC_RMSE_IMPROVEMENT']:.6f}.",
        f"- MC EV improvement (B − A): {result['MC_EV_IMPROVEMENT']:.6f}.",
        "",
        "## Fixed UAV_1 transition-149 witness",
        "",
        f"- Config: `{TARGET_CONFIG}`; transition fingerprint: `{witness['transition_fingerprint_sha256']}`; branch reconstruction: **{witness['branch_reproduction_gate']}**.",
        f"- Existing paired MC signs: NEGATIVE / NEGATIVE (ΔG values {MC_DELTA_R0:.6f}, {MC_DELTA_R1:.6f}).",
        f"- ARM_A ΔV: {witness['arm_a_delta_v']:.6f} ({witness['arm_a_delta_v_sign']}); ARM_B ΔV: {witness['arm_b_delta_v']:.6f} ({witness['arm_b_delta_v_sign']}).",
        f"- Absolute error to paired MC mean: ARM_A {witness['arm_a_abs_delta_v_error_to_mc_mean']:.6f}; ARM_B {witness['arm_b_abs_delta_v_error_to_mc_mean']:.6f}.",
        "- Exactly two immediate branches were reconstructed (KEEP and the fixed Candidate); no terminal continuation was run.",
        "",
        "## Persistent artifacts and gates",
        "",
        f"- ARM_A checkpoint: `{result['ARM_A']['checkpoint']['checkpoint_path']}`.",
        f"- ARM_B checkpoint: `{result['ARM_B']['checkpoint']['checkpoint_path']}`.",
        f"- Reliability remains `{result['COUNTERFACTUAL_CRITIC_RELIABILITY']}`; Candidate-Q remains `{result['CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION']}`; surrogate repair remains blocked: `{result['SURROGATE_REPAIR_BLOCKED']}`.",
        f"- Formal training readiness: `{result['READY_FOR_FORMAL_TRAINING_STAGE']}`.",
        f"- Decision rationale: {result['DECISION_RATIONALE']}",
        "",
    ]
    return "\n".join(lines)


def _append_current_reports(result: dict[str, Any]) -> None:
    section = _render_markdown(result)
    gate_path = ROOT / "reports" / "second_stage_joint_repair_gate.md"
    current_title = "## Critic Target Alignment Minimal Experiment"
    existing_gate = gate_path.read_text(encoding="utf-8")
    if current_title in existing_gate:
        raise FileExistsError("second-stage report already contains the target-alignment section")
    gate_section = section.replace(
        "# Critic Target Alignment Minimal Experiment",
        "## Critic Target Alignment Minimal Experiment",
        1,
    )
    gate_path.write_text(existing_gate.rstrip() + "\n\n" + gate_section + "\n", encoding="utf-8")

    one_epoch_path = ROOT / "reports" / "one_epoch_training_gate.md"
    existing_one_epoch = one_epoch_path.read_text(encoding="utf-8")
    old_reason = "ONE_EPOCH_NOT_EXECUTED_REASON = formal PPO epoch remains deferred to COUNTERFACTUAL_CRITIC_REPAIR_DESIGN after the reliability gate was classified CONCERN"
    new_reason = (
        "ONE_EPOCH_NOT_EXECUTED_REASON = formal PPO epoch remains deferred to "
        f"{result['NEXT_STAGE']} after Critic target alignment was classified "
        f"{result['CRITIC_TARGET_ALIGNMENT_EXPERIMENT_RESULT']}"
    )
    if old_reason in existing_one_epoch:
        existing_one_epoch = existing_one_epoch.replace(old_reason, new_reason, 1)
    latest_section = (
        "## Latest Critic Target Alignment Decision\n\n"
        f"Critic Target Alignment Minimal Experiment: `{result['CRITIC_TARGET_ALIGNMENT_EXPERIMENT_RESULT']}`. "
        f"Current blocker / next stage: `{result['NEXT_STAGE']}`. "
        "No formal PPO update or Epoch 1 was run; readiness remains NO. "
        "Counterfactual Critic Reliability remains CONCERN, Candidate-Q remains "
        "CANDIDATE_Q_INTEGRITY_CONCERN, and surrogate repair remains blocked.\n"
    )
    if "## Latest Critic Target Alignment Decision" in existing_one_epoch:
        raise FileExistsError("one-epoch report already contains the current target-alignment decision")
    one_epoch_path.write_text(existing_one_epoch.rstrip() + "\n\n" + latest_section + "\n", encoding="utf-8")


def _persist_initial_critic_state(
    trainer: PPOV2Trainer,
    critic_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    *,
    critic_digest: str,
    actor_digest: str,
    config_sha256: str,
    initialization_seed: int,
) -> dict[str, Any]:
    popart_mean, popart_std = trainer.policy.popart_statistics()
    welford = _welford_record(trainer.critic_return_statistics)
    payload = {
        "schema": "critic_target_alignment_initial_state_v1",
        "critic_state_dict": critic_state,
        "critic_optimizer_state_dict": optimizer_state,
        "critic_digest": critic_digest,
        "critic_state_dict_sha256": _tensor_hash(list(critic_state.items())),
        "critic_optimizer_record": _optimizer_record(trainer.critic_optimizer),
        "critic_optimizer_record_sha256": _optimizer_state_hash(trainer.critic_optimizer),
        "actor_parameter_digest": actor_digest,
        "popart": {"mean": float(popart_mean), "std": float(popart_std)},
        "welford": welford,
        "critic_optimizer_steps": int(trainer.critic_optimizer_steps),
        "actor_optimizer_steps": int(trainer.actor_optimizer_steps),
        "architecture_id": str(trainer.policy.architecture_id),
        "architecture_version": int(trainer.policy.architecture_version),
        "initialization_seed": int(initialization_seed),
        "config_sha256": config_sha256,
        "DIAGNOSTIC_ONLY": "YES",
    }
    if INITIAL_CRITIC_STATE_PATH.exists():
        reloaded = torch.load(INITIAL_CRITIC_STATE_PATH, map_location="cpu", weights_only=False)
        for key in (
            "critic_digest",
            "critic_state_dict_sha256",
            "critic_optimizer_record_sha256",
            "actor_parameter_digest",
            "initialization_seed",
            "config_sha256",
        ):
            if reloaded.get(key) != payload.get(key):
                raise AssertionError(f"persisted initial state differs for {key}")
    else:
        buffer = BytesIO()
        torch.save(payload, buffer)
        _atomic_write_bytes(INITIAL_CRITIC_STATE_PATH, buffer.getvalue())
        reloaded = torch.load(INITIAL_CRITIC_STATE_PATH, map_location="cpu", weights_only=False)
    if _tensor_hash(list(reloaded["critic_state_dict"].items())) != critic_digest:
        raise AssertionError("initial Critic state_dict reload digest failed")
    if reloaded.get("critic_optimizer_record") != _optimizer_record(trainer.critic_optimizer):
        raise AssertionError("initial Critic optimizer reload assertion failed")
    metadata = {
        key: value for key, value in reloaded.items()
        if key not in {"critic_state_dict", "critic_optimizer_state_dict"}
    }
    _atomic_write_json(INITIAL_CRITIC_STATE_METADATA_PATH, metadata)
    return metadata


def _write_failure_report(exception: BaseException) -> None:
    config_sha = _config_hash(str(CONFIG_PATH)) if CONFIG_PATH.is_file() else None
    recovery_files = [
        TRAIN_ROLLOUT_PATH,
        TRAIN_ROLLOUT_METADATA_PATH,
        TARGET_ARRAYS_PATH,
        TARGET_ARRAYS_METADATA_PATH,
        MINIBATCH_SEQUENCE_PATH,
        OUTPUT_DIR / "gae_target_critic.pt",
        OUTPUT_DIR / "gae_target_critic.metadata.json",
        OUTPUT_DIR / "mc_target_critic.pt",
        OUTPUT_DIR / "mc_target_critic.metadata.json",
        DIAGNOSTIC_ROLLOUT_PATH,
        DIAGNOSTIC_ROLLOUT_METADATA_PATH,
    ]
    present = [
        {
            "path": str(path),
            "sha256": _sha256_file(path) if path.is_file() else None,
            "exists": path.exists(),
        }
        for path in recovery_files
    ]
    try:
        current_head = _git_head()
    except Exception:
        current_head = "UNKNOWN"
    try:
        history = _historical_attempts()
    except Exception:
        history = []
    total_train_count = int(
        TRAIN_ROLLOUT_METADATA_PATH.is_file()
        or TRAIN_ROLLOUT_PATH.is_file()
        or RUN_STATE["train_rollout_collection_count_this_run"]
    )
    total_diagnostic_count = int(
        DIAGNOSTIC_ROLLOUT_METADATA_PATH.is_file()
        or DIAGNOSTIC_ROLLOUT_PATH.is_file()
        or RUN_STATE["diagnostic_rollout_collection_count_this_run"]
    )
    report = {
        "TASK": TASK_NAME,
        "CURRENT_HEAD": current_head,
        "CONFIG_PATH": str(CONFIG_PATH),
        "CONFIG_SHA256": config_sha,
        "AUTHORIZED_COMPENSATORY_EXPERIMENT_RERUN": "YES",
        "CURRENT_ATTEMPT_NAME": CURRENT_ATTEMPT,
        "RUNNER_CONFIGURATION_PREFLIGHT": RUN_STATE["runner_configuration_preflight"],
        "EXECUTION_STATUS": "ABORTED",
        "EXECUTION_FAILURE": {
            "exception": type(exception).__name__,
            "message": str(exception),
            "stage": RUN_STATE["stage"],
        },
        "TRAIN_REALIZATION_ACTION": "LOAD",
        "TRAIN_REALIZATION_SEED": TRAIN_REALIZATION_SEED,
        "TRAIN_ROLLOUT_COLLECTION_COUNT_THIS_RUN": int(RUN_STATE["train_rollout_collection_count_this_run"]),
        "TRAIN_ROLLOUT_COLLECTION_COUNT": total_train_count,
        "SHARED_TRAIN_ROLLOUT_PERSISTENCE_GATE": "PASS" if TRAIN_ROLLOUT_METADATA_PATH.is_file() else "NOT_PASS",
        "DIAGNOSTIC_REALIZATION_ACTION": RUN_STATE["diagnostic_realization_action"],
        "DIAGNOSTIC_REALIZATION_SEED": DIAGNOSTIC_REALIZATION_SEED,
        "DIAGNOSTIC_ROLLOUT_COLLECTION_COUNT": total_diagnostic_count,
        "INITIALIZATION_REPRODUCTION_GATE": RUN_STATE["initialization_reproduction_gate"],
        "INITIALIZATION_EQUIVALENCE_GATE": RUN_STATE["initialization_equivalence_gate"],
        "INITIAL_CRITIC_PARAMETER_DIGEST": EXPECTED_INITIAL_CRITIC_DIGEST,
        "ARM_A_TARGET": "GAE_LAMBDA_RETURN",
        "ARM_B_TARGET": "FULL_EPISODE_MC_RETURN",
        "ARM_A_CRITIC_STEPS": int(RUN_STATE["critic_steps_a"]),
        "ARM_B_CRITIC_STEPS": int(RUN_STATE["critic_steps_b"]),
        "ARM_A_CHECKPOINT_RELOAD_GATE": "PASS" if (OUTPUT_DIR / "gae_target_critic.pt").is_file() else "NOT_RUN",
        "ARM_B_CHECKPOINT_RELOAD_GATE": "PASS" if (OUTPUT_DIR / "mc_target_critic.pt").is_file() else "NOT_RUN",
        "DIAGNOSTIC_ROLLOUT_COLLECTION_COUNT_THIS_RUN": int(RUN_STATE["diagnostic_rollout_collection_count_this_run"]),
        "ARM_A_MC_MAE": None,
        "ARM_A_MC_RMSE": None,
        "ARM_A_MC_EV": None,
        "ARM_B_MC_MAE": None,
        "ARM_B_MC_RMSE": None,
        "ARM_B_MC_EV": None,
        "MC_MAE_IMPROVEMENT": None,
        "MC_RMSE_IMPROVEMENT": None,
        "MC_EV_IMPROVEMENT": None,
        "UAV1_EMPIRICAL_MC_SIGN": "NEGATIVE",
        "ARM_A_UAV1_DELTA_V": None,
        "ARM_A_UAV1_SIGN_AGREEMENT": None,
        "ARM_B_UAV1_DELTA_V": None,
        "ARM_B_UAV1_SIGN_AGREEMENT": None,
        "ACTOR_PARAMETERS_FROZEN": "YES",
        "CRITIC_TARGET_ALIGNMENT_EXPERIMENT_RESULT": "INCONCLUSIVE",
        "NEXT_STAGE": "CRITIC_TARGET_ALIGNMENT_EXPERIMENT_FOLLOWUP",
        "COUNTERFACTUAL_CRITIC_RELIABILITY": "CONCERN",
        "CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION": "CANDIDATE_Q_INTEGRITY_CONCERN",
        "SURROGATE_REPAIR_BLOCKED": "YES",
        "FORMAL_PPO_UPDATE_EXECUTED": "NO",
        "ONE_EPOCH_TRAINING_EXECUTED": "NO",
        "ONE_EPOCH_COMPLETED_EPOCHS": 0,
        "LAST_COMPLETED_EPOCH": "NONE",
        "EPOCH_2_STARTED": "NO",
        "READY_FOR_FORMAL_TRAINING_STAGE": "NO",
        "FORMAL_300_EPOCH_TRAINING_EXECUTED": "NO",
        "PRODUCTION_CRITIC_TARGET_CHANGED": "NO",
        "PRODUCTION_SOURCE_CHANGED": "NO",
        "RECOVERY_ARTIFACT_STATUS": present,
        "GIT_COMMIT": "NO",
        "GIT_PUSH": "NO",
    }
    report["HISTORICAL_ATTEMPTS"] = history
    try:
        _write_followup_reports(report, completed=False)
    except Exception as report_exc:
        print(f"Failure report persistence also failed: {type(report_exc).__name__}: {report_exc}", flush=True)


def _legacy_run_experiment() -> dict[str, Any]:
    if not CONFIG_PATH.is_file() or not SPLIT_MANIFEST_PATH.is_file():
        raise FileNotFoundError("formal config or data split manifest is missing")
    if REPORT_JSON.exists() or REPORT_MD.exists():
        raise FileExistsError("refusing to overwrite target-alignment report artifacts")
    expected_output_files = (
        OUTPUT_DIR / "gae_target_critic.pt",
        OUTPUT_DIR / "gae_target_critic.metadata.json",
        OUTPUT_DIR / "mc_target_critic.pt",
        OUTPUT_DIR / "mc_target_critic.metadata.json",
    )
    collisions = [str(path) for path in expected_output_files if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite diagnostic checkpoints: {collisions}")

    current_head = _git_head()
    config_sha = _config_hash(str(CONFIG_PATH))
    split_manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    generator = _load_single_episode_generator(ROOT)
    cfg = generator.load_and_validate_config(CONFIG_PATH)
    generation_fingerprint = generator.generation_config_fingerprint(cfg)
    source_hashes = generator._source_artifact_hashes(cfg)
    if split_manifest.get("generation_config_fingerprint") != generation_fingerprint:
        raise ValueError("current generation config conflicts with data_split_manifest")
    train_split = split_manifest["train"]
    if str(train_split.get("directory")) != "proposed_joint_training":
        raise ValueError("data_split_manifest TRAIN directory changed")
    if not int(train_split["seed_start"]) <= TRAIN_REALIZATION_SEED <= int(train_split["seed_end"]):
        raise ValueError("selected training realization seed is outside TRAIN range")
    if int(split_manifest.get("slots_per_realization", -1)) != 300:
        raise ValueError("data_split_manifest does not define 300 slots per realization")
    for split_hash, source_key in (
        ("service_catalog_sha256", "service_catalog_sha256"),
        ("popularity_gru_sha256", "popularity_gru_sha256"),
        ("vehicle_residual_bank_sha256", "vehicle_residual_bank_sha256"),
        ("sumo_network_sha256", "sumo_network_sha256"),
    ):
        if split_manifest.get(split_hash) != source_hashes.get(source_key):
            raise ValueError(f"data_split_manifest source artifact hash conflicts: {split_hash}")
    episode_directory_seed = TRAIN_REALIZATION.name.removeprefix("episode_")
    if episode_directory_seed != str(TRAIN_REALIZATION_SEED):
        raise AssertionError("selected TRAIN realization path does not match its episode seed")
    if not TRAIN_REALIZATION.is_dir():
        raise FileNotFoundError(f"selected existing TRAIN realization is missing: {TRAIN_REALIZATION}")
    _check_artifact_manifest(TRAIN_REALIZATION, TRAIN_REALIZATION_SEED, generation_fingerprint, source_hashes)
    generator.validate_written_outputs(
        TRAIN_REALIZATION,
        task_vehicle_count=int(project_config.get_config_value(cfg, "simulation.num_task_vehicles")),
        research_vehicle_count=int(project_config.get_config_value(cfg, "simulation.num_research_vehicles")),
        horizon_steps=int(project_config.get_config_value(cfg, "prediction.vehicle.horizon")),
    )
    train_context = load_environment_context(str(CONFIG_PATH), realization_dir=TRAIN_REALIZATION)
    _validate_episode_realization_complete(
        TRAIN_REALIZATION, train_context, generator, TRAIN_REALIZATION_SEED
    )
    if train_context.data_protocol != "full_episode_randomized" or train_context.episode_slots != 300:
        raise ValueError("selected TRAIN realization runtime protocol is not the required 300-slot protocol")
    if train_context.configured_rollout_steps != 300:
        raise ValueError("selected TRAIN realization runtime rollout length is not 300")
    train_provenance = _realization_provenance(
        directory=TRAIN_REALIZATION,
        episode_seed=TRAIN_REALIZATION_SEED,
        split="TRAIN",
        action="LOAD",
        current_config_sha=config_sha,
        current_fingerprint=generation_fingerprint,
        split_manifest=split_manifest,
        generator_module=generator,
        cfg=cfg,
        isolated_generation_rng_seed=None,
    )
    train_provenance["episode_manifest_provenance_note"] = (
        "Read only as optional provenance; reuse is established by TRAIN seed range, "
        "matching generation fingerprint, derived episode streams, complete artifacts, "
        "and runtime validation."
    )
    train_provenance["data_split_manifest_sha256"] = _sha256_file(SPLIT_MANIFEST_PATH)
    train_provenance["current_config_sha256"] = config_sha
    train_provenance["generation_config_fingerprint"] = generation_fingerprint
    train_provenance["data_split_manifest_config_sha256"] = split_manifest.get(
        "generator_config_sha256"
    )
    train_provenance["data_split_manifest_config_sha256_at_generation_start"] = split_manifest.get(
        "generator_config_sha256_at_generation_start"
    )
    train_provenance["config_sha256_compatibility_note"] = split_manifest.get(
        "generator_config_text_change_note"
    )

    base_seed = int(project_config.get_config_value(cfg, "experiment.seed"))
    initialization_seed = derive_namespaced_seed(
        base_seed, DIAGNOSTIC_NAMESPACE, "trainer_initialization"
    )
    diagnostic_generation_seed = derive_namespaced_seed(
        DIAGNOSTIC_REALIZATION_SEED, DIAGNOSTIC_NAMESPACE, "episode_generation"
    )
    diagnostic_rollout_seed = derive_namespaced_seed(
        DIAGNOSTIC_REALIZATION_SEED, DIAGNOSTIC_NAMESPACE, "rollout_collection"
    )
    stored_transition_provenance = json.loads(
        (ROOT / "reports" / "uav0_transition149_reward_root_cause.json").read_text(encoding="utf-8")
    )
    stored_seeds = stored_transition_provenance["SEEDS"]
    if initialization_seed != int(stored_seeds["TRAINER_INITIALIZATION_RNG_SEED"]):
        raise AssertionError("fresh trainer initialization seed does not match the fixed witness provenance")
    if diagnostic_generation_seed != int(stored_seeds["DIAGNOSTIC_GENERATION_RNG_SEED"]):
        raise AssertionError("diagnostic realization generation seed does not match fixed witness provenance")
    if diagnostic_rollout_seed != int(stored_seeds["DIAGNOSTIC_ROLLOUT_RNG_SEED"]):
        raise AssertionError("diagnostic rollout seed does not match fixed witness provenance")

    training_rollout_seed = derive_namespaced_seed(
        TRAIN_REALIZATION_SEED, TRAIN_NAMESPACE, "rollout_collection"
    )
    optimizer_seed = derive_namespaced_seed(
        TRAIN_REALIZATION_SEED, TRAIN_NAMESPACE, "critic_optimizer"
    )
    # Adam has no internal stochastic sampling. This one saved optimizer RNG
    # seed is also the generator seed used for the matched minibatch sequence.
    minibatch_seed = optimizer_seed
    output_directory_created = not OUTPUT_DIR.exists()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="_scratch_", dir=OUTPUT_DIR) as scratch_name:
        scratch = Path(scratch_name)
        base_trainer = _run_with_isolated_rng(
            initialization_seed,
            lambda: PPOV2Trainer(train_context, device="cpu", exact_q_parallel_workers=1),
        )
        if base_trainer.actor_optimizer_steps != 0 or base_trainer.critic_optimizer_steps != 0:
            raise AssertionError("fresh trainer starts with non-zero optimizer steps")
        if base_trainer.critic_optimizer is None:
            raise RuntimeError("fresh proposed Critic optimizer is missing")
        if base_trainer.policy.architecture_id != "adaptive_joint_cache_uav_ppo":
            raise AssertionError("fresh trainer did not initialize the proposed-joint policy")
        if base_trainer._critic_parameter_digest() != _critic_digest(base_trainer.policy):
            raise AssertionError("diagnostic Critic digest differs from the production trainer digest")
        initial_critic_state = _critic_state_dict(base_trainer.policy)
        initial_critic_state_hash = _tensor_hash(list(initial_critic_state.items()))
        initial_critic_digest = base_trainer._critic_parameter_digest()
        initial_actor_digest = _actor_digest(base_trainer.policy)
        initial_optimizer_state = copy.deepcopy(base_trainer.critic_optimizer.state_dict())
        initial_optimizer_record = _optimizer_record(base_trainer.critic_optimizer)
        initial_optimizer_state_hash = _optimizer_state_hash(base_trainer.critic_optimizer)
        initial_welford = _welford_record(base_trainer.critic_return_statistics)
        initial_popart = list(base_trainer.policy.popart_statistics())
        for parameter in base_trainer.policy.actor_parameters():
            parameter.requires_grad_(False)

        arm_a = _make_critic_arm(base_trainer, initial_optimizer_state, "cpu")
        arm_b = _make_critic_arm(base_trainer, initial_optimizer_state, "cpu")
        for arm in (arm_a, arm_b):
            missing, unexpected = arm.policy.load_state_dict(
                copy.deepcopy(initial_critic_state), strict=False
            )
            if unexpected or set(initial_critic_state).intersection(missing):
                raise AssertionError(
                    "saved initial Critic state did not load cleanly into an independent arm"
                )
        if base_trainer.actor_optimizer_steps != 0 or base_trainer.critic_optimizer_steps != 0:
            raise AssertionError("trainer changed optimizer steps during arm initialization")
        initialization = _assert_initialization_equivalence(
            arm_a,
            arm_b,
            initial_optimizer_record,
            initial_critic_digest,
            initial_actor_digest,
        )
        if initial_welford != {"count": 0, "mean": 0.0, "m2": 0.0} or initial_popart != [0.0, 1.0]:
            raise AssertionError("fresh Critic PopArt/Welford state is not the expected neutral state")
        initial_critic_state.clear()

        def forbidden_formal_update(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("trainer.update() is forbidden in this diagnostic experiment")

        base_trainer.update = forbidden_formal_update  # type: ignore[method-assign]
        actor_parameters_before = tuple(
            parameter.detach().clone() for parameter in base_trainer.policy.actor_parameters()
        )
        print("Collecting the one shared 300-transition TRAIN rollout.", flush=True)
        training_rollout = _run_with_isolated_rng(
            training_rollout_seed,
            lambda: base_trainer.collect_chunk(
                train_context.train_origin_start,
                train_context.initial_cache,
                train_context.initial_uav,
                train_context.configured_rollout_steps,
            ),
        )
        if len(training_rollout.transitions) != 300 or len(training_rollout.observations) != 300:
            raise AssertionError("shared TRAIN rollout did not contain exactly 300 transitions")
        if not bool(training_rollout.dones[-1]) or bool(np.any(training_rollout.dones[:-1])):
            raise AssertionError("TRAIN rollout must terminate only at transition 299")
        training_rollout_arrays = {
            "observations": np.asarray(training_rollout.observations),
            "actions": np.asarray(training_rollout.actions),
            "rewards": np.asarray(training_rollout.rewards),
            "dones": np.asarray(training_rollout.dones),
            "rollout_time_raw_values": np.asarray(training_rollout.old_values),
        }
        if any(not np.all(np.isfinite(value)) for key, value in training_rollout_arrays.items() if key != "dones"):
            raise FloatingPointError("shared TRAIN rollout contains a non-finite numeric array")
        np.savez_compressed(scratch / "training_rollout.npz", **training_rollout_arrays)
        actor_parameters_after_rollout = tuple(
            parameter.detach().clone() for parameter in base_trainer.policy.actor_parameters()
        )
        if any(not torch.equal(before, after) for before, after in zip(actor_parameters_before, actor_parameters_after_rollout)):
            raise AssertionError("Actor parameters changed during TRAIN rollout collection")
        if base_trainer.actor_optimizer_steps != 0 or base_trainer.critic_optimizer_steps != 0:
            raise AssertionError("fresh trainer optimizer stepped during rollout collection")
        rollout_critic_values = _evaluate_values(
            base_trainer.policy, training_rollout.observations, "cpu"
        )
        if not np.allclose(rollout_critic_values, training_rollout.old_values, rtol=1.0e-6, atol=1.0e-5):
            raise AssertionError("saved rollout-time raw values differ from the untouched initial Critic")

        gamma = float(project_config.get_config_value(cfg, "algorithm.ppo.gamma"))
        gae_lambda = float(project_config.get_config_value(cfg, "algorithm.ppo.gae_lambda"))
        hyper = base_trainer.hyperparameters
        if (
            gamma != 1.0
            or gae_lambda != 0.95
            or float(hyper.critic_learning_rate) != 0.001
            or int(hyper.update_epochs) != 4
            or int(hyper.num_minibatches) != 10
            or int(train_context.minibatch_size) != 30
            or float(hyper.max_grad_norm) != 0.5
        ):
            raise AssertionError("formal Critic target-alignment experiment parameters differ from the authorized configuration")
        if training_rollout.bootstrap_observation is None:
            bootstrap_value = 0.0
        else:
            with torch.no_grad():
                bootstrap_value = float(
                    base_trainer.policy.critic_value(
                        torch.as_tensor(training_rollout.bootstrap_observation, dtype=torch.float32)
                    ).detach().cpu()
                )
        advantages_a, targets_a = compute_gae(
            training_rollout.rewards,
            training_rollout.old_values,
            training_rollout.dones,
            bootstrap_value,
            gamma,
            gae_lambda,
        )
        targets_b = _mc_reward_to_go(training_rollout.rewards, training_rollout.dones, gamma)
        if len(targets_a) != 300 or len(targets_b) != 300:
            raise AssertionError("Critic target length assertion failed")
        if not np.all(np.isfinite(advantages_a)) or not np.all(np.isfinite(targets_a)) or not np.all(np.isfinite(targets_b)):
            raise FloatingPointError("Critic targets or Actor-GAE diagnostic values are non-finite")
        target_hashes = {
            "ARM_A_GAE_TARGET_SHA256": _array_sha256(targets_a),
            "ARM_B_MC_TARGET_SHA256": _array_sha256(targets_b),
        }
        if base_trainer.device.type != "cpu":
            raise AssertionError("matched minibatch sequence is expected to run on CPU")
        initial_rng_state, sequence_a = _generate_minibatches(
            minibatch_seed, 300, int(hyper.update_epochs), int(hyper.num_minibatches)
        )
        initial_rng_state_b, sequence_b = _generate_minibatches(
            minibatch_seed, 300, int(hyper.update_epochs), int(hyper.num_minibatches)
        )
        if not torch.equal(initial_rng_state, initial_rng_state_b):
            raise AssertionError("minibatch RNG initial states differ between arms")
        sequence_hash_a = _minibatch_sequence_digest(sequence_a)
        sequence_hash_b = _minibatch_sequence_digest(sequence_b)
        if len(sequence_a) != len(sequence_b) or not all(
            np.array_equal(left, right) for left, right in zip(sequence_a, sequence_b)
        ):
            raise AssertionError("minibatch index sequences differ between arms")
        if len(sequence_a) != 40 or any(len(indices) != 30 for indices in sequence_a):
            raise AssertionError("minibatch order does not define 4 epochs x 10 minibatches of 30")

        epsilon = float(hyper.critic_value_scale_epsilon)
        warmup_samples = int(hyper.critic_value_scale_warmup_samples)
        prep_a = _prepare_arm_targets(
            arm_a, targets_a, training_rollout.observations, epsilon, warmup_samples
        )
        prep_b = _prepare_arm_targets(
            arm_b, targets_b, training_rollout.observations, epsilon, warmup_samples
        )
        normalized_a = prep_a.pop("normalized_targets")
        normalized_b = prep_b.pop("normalized_targets")
        print("Updating ARM_A Critic for 40 steps using the shared rollout and GAE targets.", flush=True)
        arm_a_training = _train_arm(
            arm_a, training_rollout.observations, targets_a, normalized_a,
            sequence_a, int(hyper.update_epochs) * int(hyper.num_minibatches),
        )
        print("Updating ARM_B Critic for 40 steps using the shared rollout and full MC targets.", flush=True)
        arm_b_training = _train_arm(
            arm_b, training_rollout.observations, targets_b, normalized_b,
            sequence_b, int(hyper.update_epochs) * int(hyper.num_minibatches),
        )
        if sequence_hash_a != sequence_hash_b:
            raise AssertionError("minibatch sequence hash equality failed")
        if arm_a_training["critic_optimizer_steps"] != 40 or arm_b_training["critic_optimizer_steps"] != 40:
            raise AssertionError("Critic optimizer step count assertion failed")
        if _actor_digest(arm_a.policy) != _actor_digest(arm_b.policy) or _actor_digest(arm_a.policy) != initial_actor_digest:
            raise AssertionError("Actor freeze assertion failed after Critic training")

        seeds = {
            "trainer_initialization_seed": int(initialization_seed),
            "training_rollout_seed": int(training_rollout_seed),
            "optimizer_seed": int(optimizer_seed),
            "minibatch_seed": int(minibatch_seed),
            "diagnostic_generation_seed": int(diagnostic_generation_seed),
            "diagnostic_rollout_seed": int(diagnostic_rollout_seed),
        }
        checkpoint_a = _save_critic_checkpoint(
            arm=arm_a,
            arm_name="ARM_A",
            target_type="GAE_LAMBDA_RETURN",
            target_bootstrap="YES",
            target_sha256=target_hashes["ARM_A_GAE_TARGET_SHA256"],
            sequence_sha256=sequence_hash_a,
            train_provenance=train_provenance,
            seeds={"training_rollout_seed": training_rollout_seed, "optimizer_seed": optimizer_seed},
            config_sha256=config_sha,
        )
        checkpoint_b = _save_critic_checkpoint(
            arm=arm_b,
            arm_name="ARM_B",
            target_type="FULL_EPISODE_MC_RETURN",
            target_bootstrap="NO",
            target_sha256=target_hashes["ARM_B_MC_TARGET_SHA256"],
            sequence_sha256=sequence_hash_b,
            train_provenance=train_provenance,
            seeds={"training_rollout_seed": training_rollout_seed, "optimizer_seed": optimizer_seed},
            config_sha256=config_sha,
        )

        existing_diagnostic = _find_diagnostic_realization(
            generator, cfg, generation_fingerprint, train_context
        )
        if existing_diagnostic is None:
            diagnostic_action = "GENERATE"
            diagnostic_dir = scratch / "diagnostic_train_realization"
            print(
                "No compatible existing diagnostic realization was found; generating the fixed TRAIN seed once.",
                flush=True,
            )
            _run_with_isolated_rng(
                diagnostic_generation_seed,
                lambda: _ensure_episode_realization(
                    train_context,
                    str(CONFIG_PATH),
                    diagnostic_dir,
                    DIAGNOSTIC_REALIZATION_SEED,
                    generate_missing=True,
                    generation_device="cpu",
                    generator_module=generator,
                ),
            )
        else:
            diagnostic_action = "LOAD"
            diagnostic_dir = existing_diagnostic
        diagnostic_context = load_environment_context(
            str(CONFIG_PATH), realization_dir=diagnostic_dir
        )
        _assert_realization_context_compatible(train_context, diagnostic_context)
        if diagnostic_context.data_protocol != "full_episode_randomized" or diagnostic_context.episode_slots != 300:
            raise ValueError("diagnostic realization is not a 300-slot TRAIN protocol")
        _validate_episode_realization_complete(
            diagnostic_dir, diagnostic_context, generator, DIAGNOSTIC_REALIZATION_SEED
        )
        diagnostic_provenance = _realization_provenance(
            directory=diagnostic_dir,
            episode_seed=DIAGNOSTIC_REALIZATION_SEED,
            split="TRAIN",
            action=diagnostic_action,
            current_config_sha=config_sha,
            current_fingerprint=generation_fingerprint,
            split_manifest=(split_manifest if existing_diagnostic is not None else None),
            generator_module=generator,
            cfg=cfg,
            isolated_generation_rng_seed=(
                diagnostic_generation_seed if diagnostic_action != "LOAD" else None
            ),
        )
        if existing_diagnostic is None:
            diagnostic_provenance["split_evidence"] = "AUTHORIZED_INDEPENDENT_DIAGNOSTIC_TRAIN_REALIZATION"
        base_trainer.context = diagnostic_context
        actor_before_diagnostic = _actor_digest(base_trainer.policy)
        print("Collecting the one independent 300-transition TRAIN diagnostic rollout.", flush=True)
        diagnostic_rollout = _run_with_isolated_rng(
            diagnostic_rollout_seed,
            lambda: base_trainer.collect_chunk(
                diagnostic_context.train_origin_start,
                diagnostic_context.initial_cache,
                diagnostic_context.initial_uav,
                diagnostic_context.configured_rollout_steps,
            ),
        )
        if len(diagnostic_rollout.transitions) != 300 or len(diagnostic_rollout.observations) != 300:
            raise AssertionError("diagnostic rollout did not contain exactly 300 transitions")
        if not bool(diagnostic_rollout.dones[-1]) or bool(np.any(diagnostic_rollout.dones[:-1])):
            raise AssertionError("diagnostic rollout must terminate only at transition 299")
        diagnostic_arrays = {
            "observations": np.asarray(diagnostic_rollout.observations),
            "actions": np.asarray(diagnostic_rollout.actions),
            "rewards": np.asarray(diagnostic_rollout.rewards),
            "dones": np.asarray(diagnostic_rollout.dones),
        }
        np.savez_compressed(scratch / "diagnostic_rollout.npz", **diagnostic_arrays)
        if _actor_digest(base_trainer.policy) != actor_before_diagnostic or actor_before_diagnostic != initial_actor_digest:
            raise AssertionError("Actor parameters changed during diagnostic rollout collection")
        if base_trainer.actor_optimizer_steps != 0 or base_trainer.critic_optimizer_steps != 0:
            raise AssertionError("base trainer optimizer counters changed during diagnostics")
        mc_diagnostic_returns = _mc_reward_to_go(
            diagnostic_rollout.rewards, diagnostic_rollout.dones, gamma
        )
        predictions_a = _evaluate_values(arm_a.policy, diagnostic_rollout.observations, "cpu")
        predictions_b = _evaluate_values(arm_b.policy, diagnostic_rollout.observations, "cpu")
        metrics_a = _metrics(predictions_a, mc_diagnostic_returns)
        metrics_b = _metrics(predictions_b, mc_diagnostic_returns)
        for metrics in (metrics_a, metrics_b):
            for key in ("mae", "rmse"):
                if not math.isfinite(float(metrics[key])):
                    raise FloatingPointError(f"diagnostic metric is non-finite: {key}")
            if not isinstance(metrics["explained_variance"], float) or not math.isfinite(metrics["explained_variance"]):
                raise FloatingPointError("diagnostic explained variance is not finite")
        _array_sha256(mc_diagnostic_returns)

        transition = diagnostic_rollout.transitions[TARGET_TRANSITION]
        target_origin = int(transition.origin_index)
        if target_origin != TARGET_TRANSITION:
            raise AssertionError(f"fixed witness origin changed: {target_origin}")
        witness_steps = aligned_train_steps(
            diagnostic_context,
            origin_start=target_origin,
            max_steps=2,
            include_bootstrap_step=False,
        )
        if len(witness_steps) != 2 or int(witness_steps[0].origin_index) != TARGET_TRANSITION:
            raise AssertionError("fixed UAV_1 witness current/next aligned steps are unavailable")
        witness_step, next_step = witness_steps
        fingerprint = _transition_fingerprint(transition, witness_step, config_sha)
        if fingerprint != EXPECTED_TRANSITION_FINGERPRINT:
            raise AssertionError(
                "reconstructed transition-149 fingerprint differs from retained paired-MC witness: "
                f"{fingerprint}"
            )
        node_ids = [str(value) for value in diagnostic_context.action_spec.cache_node_ids]
        if TARGET_HEAD not in node_ids:
            raise AssertionError("fixed UAV_1 head is absent from the production ActionSpec")
        node_index = node_ids.index(TARGET_HEAD)
        working_cache, previous_cache = cache_credit._replay_cache_prefix(transition, node_index)
        candidate = cache_configuration.CacheTargetCandidate(
            node_index=int(node_index),
            node_id=TARGET_HEAD,
            before_node_cache=np.asarray(working_cache[node_index], dtype=np.int8).copy(),
            target_node_cache=np.asarray(TARGET_CONFIG, dtype=np.int8),
            proposal_source="fixed_existing_UAV1_reliability_witness",
        )
        branch_state = {"working_cache": working_cache, "previous_cache": previous_cache}
        print("Reconstructing only the fixed UAV_1 KEEP and Candidate immediate branches.", flush=True)
        keep_branch = cache_credit._execute_cache_branch(
            base_trainer, transition, witness_step, next_step, node_index,
            branch_state, None,
        )
        candidate_branch = cache_credit._execute_cache_branch(
            base_trainer, transition, witness_step, next_step, node_index,
            branch_state, candidate,
        )
        next_observation_keep = keep_branch.get("next_observation")
        next_observation_candidate = candidate_branch.get("next_observation")
        if next_observation_keep is None or next_observation_candidate is None:
            raise AssertionError("production branch builder did not return both next observations")
        if (
            np.asarray(next_observation_keep).shape != (diagnostic_context.observation_spec.observation_dim,)
            or np.asarray(next_observation_candidate).shape != (diagnostic_context.observation_spec.observation_dim,)
        ):
            raise AssertionError("production branch next-observation shape is incorrect")
        keep_batch = np.stack([next_observation_keep]).astype(np.float32, copy=False)
        candidate_batch = np.stack([next_observation_candidate]).astype(np.float32, copy=False)
        branch_values = {
            "arm_a_keep_value": float(_evaluate_values(arm_a.policy, keep_batch, "cpu")[0]),
            "arm_a_candidate_value": float(_evaluate_values(arm_a.policy, candidate_batch, "cpu")[0]),
            "arm_b_keep_value": float(_evaluate_values(arm_b.policy, keep_batch, "cpu")[0]),
            "arm_b_candidate_value": float(_evaluate_values(arm_b.policy, candidate_batch, "cpu")[0]),
        }
        if not all(math.isfinite(value) for value in branch_values.values()):
            raise FloatingPointError("UAV_1 branch Critic values are non-finite")
        delta_a = branch_values["arm_a_candidate_value"] - branch_values["arm_a_keep_value"]
        delta_b = branch_values["arm_b_candidate_value"] - branch_values["arm_b_keep_value"]
        empirical_mc_mean = 0.5 * (MC_DELTA_R0 + MC_DELTA_R1)
        witness = {
            "target_transition": TARGET_TRANSITION,
            "target_origin_index": target_origin,
            "target_head": TARGET_HEAD,
            "target_config": TARGET_CONFIG,
            "transition_fingerprint_sha256": fingerprint,
            "branch_reproduction_gate": "PASS",
            "immediate_branch_execution_count": 2,
            "terminal_continuation_count": 0,
            "keep_next_observation_sha256": _array_sha256(next_observation_keep),
            "candidate_next_observation_sha256": _array_sha256(next_observation_candidate),
            **branch_values,
            "arm_a_delta_v": float(delta_a),
            "arm_b_delta_v": float(delta_b),
            "arm_a_delta_v_sign": _sign(delta_a),
            "arm_b_delta_v_sign": _sign(delta_b),
            "empirical_mc_delta_g_r0": MC_DELTA_R0,
            "empirical_mc_delta_g_r1": MC_DELTA_R1,
            "empirical_mc_sign_r0": "NEGATIVE",
            "empirical_mc_sign_r1": "NEGATIVE",
            "empirical_mc_mean": float(empirical_mc_mean),
            "arm_a_sign_agreement": _sign(delta_a) == "NEGATIVE",
            "arm_b_sign_agreement": _sign(delta_b) == "NEGATIVE",
            "arm_a_abs_delta_v_error_to_mc_mean": float(abs(delta_a - empirical_mc_mean)),
            "arm_b_abs_delta_v_error_to_mc_mean": float(abs(delta_b - empirical_mc_mean)),
        }
        if not np.all(np.isfinite(np.asarray(next_observation_keep))) or not np.all(np.isfinite(np.asarray(next_observation_candidate))):
            raise FloatingPointError("UAV_1 branch next observations are non-finite")

        arm_a_result = {
            "target": "GAE_LAMBDA_RETURN",
            "target_uses_bootstrap": "YES",
            "target_array": _array_record(targets_a),
            "training": {**arm_a_training, **prep_a},
            "critic_digest": _critic_digest(arm_a.policy),
            "popart": {"mean": float(arm_a.policy.popart_statistics()[0]), "std": float(arm_a.policy.popart_statistics()[1])},
            "welford": _welford_record(arm_a.critic_return_statistics),
            "metrics": metrics_a,
            "checkpoint": checkpoint_a,
        }
        arm_b_result = {
            "target": "FULL_EPISODE_MC_RETURN",
            "target_uses_bootstrap": "NO",
            "target_array": _array_record(targets_b),
            "training": {**arm_b_training, **prep_b},
            "critic_digest": _critic_digest(arm_b.policy),
            "popart": {"mean": float(arm_b.policy.popart_statistics()[0]), "std": float(arm_b.policy.popart_statistics()[1])},
            "welford": _welford_record(arm_b.critic_return_statistics),
            "metrics": metrics_b,
            "checkpoint": checkpoint_b,
        }
        gates_pass = bool(
            initialization["status"] == "PASS"
            and sequence_hash_a == sequence_hash_b
            and arm_a_training["critic_optimizer_steps"] == arm_b_training["critic_optimizer_steps"] == 40
            and arm_a_training["actor_parameter_delta"] == arm_b_training["actor_parameter_delta"] == 0.0
            and all(value == "YES" for value in (prep_a["raw_value_preservation_finite"], prep_b["raw_value_preservation_finite"]))
            and all(math.isfinite(float(item)) for metrics in (metrics_a, metrics_b) for item in (metrics["mae"], metrics["rmse"], metrics["explained_variance"]))
        )
        experiment_result, rationale = _classify_result(arm_a_result, arm_b_result, witness, gates_pass)
        if experiment_result == "SUPPORTED":
            next_stage = "CRITIC_TARGET_ALIGNMENT_PRODUCTION_REPAIR_DESIGN"
        elif experiment_result in {"PARTIAL", "NOT_SUPPORTED"}:
            next_stage = "CRITIC_DATA_COVERAGE_MINIMAL_EXPERIMENT"
        else:
            next_stage = "CRITIC_TARGET_ALIGNMENT_EXPERIMENT_FOLLOWUP"

        report = {
            "TASK": TASK_NAME,
            "CURRENT_HEAD": current_head,
            "CONFIG_PATH": str(CONFIG_PATH),
            "CONFIG_SHA256": config_sha,
            "GENERATION_CONFIG_FINGERPRINT": generation_fingerprint,
            "PRODUCTION_CRITIC_TARGET_CHANGED": "NO",
            "PRODUCTION_CRITIC_TARGET": "GAE_LAMBDA_RETURN",
            "FORMAL_PPO_UPDATE_EXECUTED": "NO",
            "ONE_EPOCH_TRAINING_EXECUTED": "NO",
            "ONE_EPOCH_COMPLETED_EPOCHS": 0,
            "LAST_COMPLETED_EPOCH": "NONE",
            "EPOCH_2_STARTED": "NO",
            "FORMAL_300_EPOCH_TRAINING_EXECUTED": "NO",
            "READY_FOR_FORMAL_TRAINING_STAGE": "NO",
            "TRAIN_REALIZATION_ACTION": "LOAD",
            "TRAIN_REALIZATION_REUSE_EXISTING": "YES",
            "TRAIN_REALIZATION_GENERATED": "NO",
            "TRAIN_REALIZATION": train_provenance,
            "TRAIN_ROLLOUT_COLLECTION_COUNT": 1,
            "TRAIN_ROLLOUT_ARRAYS": {key: _array_record(value) for key, value in training_rollout_arrays.items()},
            "TRAIN_ROLLOUT_TARGET_ARRAY_SHA256": target_hashes,
            "INITIALIZATION_EQUIVALENCE_GATE": initialization["status"],
            "INITIAL_CRITIC_PARAMETER_DIGEST": initial_critic_digest,
            "INITIAL_CRITIC_STATE_DICT_CAPTURED": "YES",
            "INITIAL_CRITIC_STATE_DICT_SHA256": initial_critic_state_hash,
            "INITIAL_CRITIC_OPTIMIZER_STATE_SHA256": initial_optimizer_state_hash,
            "INITIAL_CRITIC_OPTIMIZER_STATE": initial_optimizer_record,
            "INITIAL_POPART_STATE": {"mean": float(initial_popart[0]), "std": float(initial_popart[1])},
            "INITIAL_WELFORD_STATE": initial_welford,
            "INITIALIZATION_EQUIVALENCE_DETAILS": initialization,
            "ARM_A_TARGET": "GAE_LAMBDA_RETURN",
            "ARM_A_TARGET_USES_BOOTSTRAP": "YES",
            "ARM_B_TARGET": "FULL_EPISODE_MC_RETURN",
            "ARM_B_TARGET_USES_BOOTSTRAP": "NO",
            "ACTOR_ADVANTAGE_TARGET": "CURRENT_GAE_UNCHANGED",
            "ACTOR_GAE_CHANGED": "NO",
            "ACTOR_PARAMETERS_FROZEN": "YES",
            "ACTOR_PARAMETER_DIGEST_INITIAL": initial_actor_digest,
            "ACTOR_PARAMETER_DIGEST_FINAL": _actor_digest(base_trainer.policy),
            "ACTOR_PARAMETER_DELTA": 0.0,
            "ACTOR_OPTIMIZER_STEPS": 0,
            "ARM_A_CRITIC_STEPS": int(arm_a.critic_optimizer_steps),
            "ARM_B_CRITIC_STEPS": int(arm_b.critic_optimizer_steps),
            "ARM_A_POPART_MEAN": float(arm_a.policy.popart_statistics()[0]),
            "ARM_A_POPART_STD": float(arm_a.policy.popart_statistics()[1]),
            "ARM_B_POPART_MEAN": float(arm_b.policy.popart_statistics()[0]),
            "ARM_B_POPART_STD": float(arm_b.policy.popart_statistics()[1]),
            "ARM_A_WELFORD_STATE": _welford_record(arm_a.critic_return_statistics),
            "ARM_B_WELFORD_STATE": _welford_record(arm_b.critic_return_statistics),
            "CRITIC_LEARNING_RATE": float(hyper.critic_learning_rate),
            "CRITIC_OPTIMIZER": "Adam",
            "UPDATE_EPOCHS": int(hyper.update_epochs),
            "NUM_MINIBATCHES": int(hyper.num_minibatches),
            "GRAD_CLIP_MAX_NORM": float(hyper.max_grad_norm),
            "MINIBATCH_ORDER_EQUIVALENCE_GATE": "PASS",
            "MINIBATCH_INDEX_SEQUENCE_SHA256_ARM_A": sequence_hash_a,
            "MINIBATCH_INDEX_SEQUENCE_SHA256_ARM_B": sequence_hash_b,
            "MINIBATCH_INITIAL_RNG_STATE_SHA256": hashlib.sha256(initial_rng_state.numpy().tobytes()).hexdigest(),
            "TRAINING_SEEDS": seeds,
            "ARM_A": arm_a_result,
            "ARM_B": arm_b_result,
            "DIAGNOSTIC_REALIZATION_ACTION": diagnostic_action,
            "DIAGNOSTIC_REALIZATION_REUSE_EXISTING": "YES" if diagnostic_action == "LOAD" else "NO",
            "DIAGNOSTIC_REALIZATION_GENERATED": "NO" if diagnostic_action == "LOAD" else "YES",
            "DIAGNOSTIC_REALIZATION": diagnostic_provenance,
            "DIAGNOSTIC_SPLIT": "TRAIN",
            "DIAGNOSTIC_ROLLOUT_COLLECTION_COUNT": 1,
            "DIAGNOSTIC_ROLLOUT_ARRAYS": {key: _array_record(value) for key, value in diagnostic_arrays.items()},
            "DIAGNOSTIC_MC_RETURN_TARGET": _array_record(mc_diagnostic_returns),
            "ACTUAL_STATE_DIAGNOSTIC": "INDEPENDENT_TRAIN_REALIZATION_MC_REWARD_TO_GO",
            "MC_MAE_IMPROVEMENT": float(metrics_a["mae"] - metrics_b["mae"]),
            "MC_RMSE_IMPROVEMENT": float(metrics_a["rmse"] - metrics_b["rmse"]),
            "MC_EV_IMPROVEMENT": float(metrics_b["explained_variance"] - metrics_a["explained_variance"]),
            "ARM_A_MC_MAE": float(metrics_a["mae"]),
            "ARM_A_MC_RMSE": float(metrics_a["rmse"]),
            "ARM_A_MC_EV": float(metrics_a["explained_variance"]),
            "ARM_A_MC_PEARSON": metrics_a["pearson"],
            "ARM_A_MC_SPEARMAN": metrics_a["spearman"],
            "ARM_B_MC_MAE": float(metrics_b["mae"]),
            "ARM_B_MC_RMSE": float(metrics_b["rmse"]),
            "ARM_B_MC_EV": float(metrics_b["explained_variance"]),
            "ARM_B_MC_PEARSON": metrics_b["pearson"],
            "ARM_B_MC_SPEARMAN": metrics_b["spearman"],
            "UAV1_WITNESS": witness,
            "ARM_A_UAV1_DELTA_V": float(delta_a),
            "ARM_A_UAV1_SIGN_AGREEMENT": witness["arm_a_sign_agreement"],
            "ARM_A_UAV1_ABS_DELTA_V_ERROR_TO_MC_MEAN": witness["arm_a_abs_delta_v_error_to_mc_mean"],
            "ARM_B_UAV1_DELTA_V": float(delta_b),
            "ARM_B_UAV1_SIGN_AGREEMENT": witness["arm_b_sign_agreement"],
            "ARM_B_UAV1_ABS_DELTA_V_ERROR_TO_MC_MEAN": witness["arm_b_abs_delta_v_error_to_mc_mean"],
            "CRITIC_TARGET_ALIGNMENT_EXPERIMENT_RESULT": experiment_result,
            "DECISION_RATIONALE": rationale,
            "NEXT_STAGE": next_stage,
            "COUNTERFACTUAL_CRITIC_RELIABILITY": "CONCERN",
            "CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION": "CANDIDATE_Q_INTEGRITY_CONCERN",
            "SURROGATE_REPAIR_BLOCKED": "YES",
            "TEMPORARY_REALIZATION_AND_SCRATCH_CLEANED": "YES",
            "TEMPORARY_REALIZATION_PATH": (
                str(diagnostic_dir.resolve()) if diagnostic_action != "LOAD" else None
            ),
            "TRACEABLE_CRITIC_CHECKPOINT_OVERWRITTEN": "NO",
            "OUTPUT_DIRECTORY_CREATED_FOR_DIAGNOSTICS": "YES" if output_directory_created else "ALREADY_PRESENT_NO_COLLISION",
        }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    reloaded = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    if reloaded["TASK"] != TASK_NAME or reloaded["CRITIC_TARGET_ALIGNMENT_EXPERIMENT_RESULT"] != report["CRITIC_TARGET_ALIGNMENT_EXPERIMENT_RESULT"]:
        raise AssertionError("output JSON reload validation failed")
    REPORT_MD.write_text(_render_markdown(report), encoding="utf-8")
    _append_current_reports(report)
    return report


def _legacy_main() -> int:
    report = run_experiment()
    print(
        json.dumps(
            {
                "result": report["CRITIC_TARGET_ALIGNMENT_EXPERIMENT_RESULT"],
                "next_stage": report["NEXT_STAGE"],
                "arm_a_metrics": report["ARM_A"]["metrics"],
                "arm_b_metrics": report["ARM_B"]["metrics"],
                "uav1_delta_v": {
                    "arm_a": report["UAV1_WITNESS"]["arm_a_delta_v"],
                    "arm_b": report["UAV1_WITNESS"]["arm_b_delta_v"],
                },
                "report_json": str(REPORT_JSON),
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    pass

def run_experiment(preflight: dict[str, Any] | None = None) -> dict[str, Any]:
    if preflight is None:
        preflight = configuration_preflight()
    if RUN_STATE["runner_configuration_preflight"] != "PASS":
        raise AssertionError("configuration preflight did not pass")
    prior_result = preflight["history"]["previous_target_alignment"]
    existing_attempt = prior_result.get("CURRENT_ATTEMPT")
    if isinstance(existing_attempt, dict) and existing_attempt.get("attempt") == CURRENT_ATTEMPT:
        if existing_attempt.get("result") in {"SUPPORTED", "PARTIAL", "NOT_SUPPORTED"}:
            raise FileExistsError("authorized compensatory follow-up is already complete")

    RUN_STATE["stage"] = "PREPARING_RECOVERY_DIRECTORY"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
    current_head = _git_head()
    config_sha = _config_hash(str(CONFIG_PATH))
    generator = preflight["generator"]
    cfg = preflight["cfg"]
    split_manifest = preflight["split_manifest"]
    generation_fingerprint = preflight["generation_fingerprint"]
    source_hashes = preflight["source_hashes"]
    train_context = preflight["train_context"]
    base_trainer: PPOV2Trainer = preflight["trainer"]
    hyper = base_trainer.hyperparameters
    history = preflight["history"]
    if history["continuation_record_count"] != 12:
        raise AssertionError("the existing twelve paired continuation records are incomplete")

    config_values = preflight["config_values"]
    gamma = float(config_values["gamma"])
    gae_lambda = float(config_values["gae_lambda"])
    if base_trainer.critic_optimizer is None:
        raise RuntimeError("fresh proposed Critic optimizer is missing")
    if base_trainer.policy.architecture_id != "adaptive_joint_cache_uav_ppo":
        raise AssertionError("fresh trainer did not initialize the proposed-joint policy")
    if base_trainer._critic_parameter_digest() != _critic_digest(base_trainer.policy):
        raise AssertionError("diagnostic Critic digest differs from the trainer digest")
    if base_trainer._critic_parameter_digest() != EXPECTED_INITIAL_CRITIC_DIGEST:
        RUN_STATE["initialization_reproduction_gate"] = "FAIL"
        raise AssertionError("fresh Critic initialization digest did not reproduce")
    RUN_STATE["initialization_reproduction_gate"] = "PASS"

    base_seed = int(project_config.get_config_value(cfg, "experiment.seed"))
    initialization_seed = int(preflight["initialization_seed"])
    diagnostic_generation_seed = derive_namespaced_seed(
        DIAGNOSTIC_REALIZATION_SEED, DIAGNOSTIC_NAMESPACE, "episode_generation"
    )
    diagnostic_rollout_seed = derive_namespaced_seed(
        DIAGNOSTIC_REALIZATION_SEED, DIAGNOSTIC_NAMESPACE, "rollout_collection"
    )
    training_rollout_seed = derive_namespaced_seed(
        TRAIN_REALIZATION_SEED, TRAIN_NAMESPACE, "rollout_collection"
    )
    optimizer_seed = derive_namespaced_seed(
        TRAIN_REALIZATION_SEED, TRAIN_NAMESPACE, "critic_optimizer"
    )
    minibatch_seed = optimizer_seed
    transition_report = json.loads(
        (ROOT / "reports" / "uav0_transition149_reward_root_cause.json").read_text(encoding="utf-8")
    )
    stored_seeds = transition_report["SEEDS"]
    if (
        initialization_seed != int(stored_seeds["TRAINER_INITIALIZATION_RNG_SEED"])
        or diagnostic_generation_seed != int(stored_seeds["DIAGNOSTIC_GENERATION_RNG_SEED"])
        or diagnostic_rollout_seed != int(stored_seeds["DIAGNOSTIC_ROLLOUT_RNG_SEED"])
    ):
        raise AssertionError("fixed deterministic witness seed derivation changed")

    train_provenance = _realization_provenance(
        directory=TRAIN_REALIZATION,
        episode_seed=TRAIN_REALIZATION_SEED,
        split="TRAIN",
        action="LOAD",
        current_config_sha=config_sha,
        current_fingerprint=generation_fingerprint,
        split_manifest=split_manifest,
        generator_module=generator,
        cfg=cfg,
        isolated_generation_rng_seed=None,
    )
    train_provenance.update(
        {
            "data_split_manifest_sha256": _sha256_file(SPLIT_MANIFEST_PATH),
            "current_config_sha256": config_sha,
            "generation_config_fingerprint": generation_fingerprint,
            "explicit_config_conflicts": [],
            "episode_manifest_provenance_note": (
                "Optional provenance only; the realization is reusable by TRAIN identity, "
                "seed, compatible generation fingerprint, complete artifacts, and runtime validation."
            ),
        }
    )
    if train_context.data_protocol != "full_episode_randomized" or train_context.episode_slots != 300:
        raise AssertionError("TRAIN realization protocol changed after preflight")
    if train_context.configured_rollout_steps != 300:
        raise AssertionError("TRAIN runtime rollout length changed after preflight")

    initial_critic_state = _critic_state_dict(base_trainer.policy)
    initial_critic_state_hash = _tensor_hash(list(initial_critic_state.items()))
    initial_critic_digest = base_trainer._critic_parameter_digest()
    initial_actor_digest = _actor_digest(base_trainer.policy)
    initial_optimizer_state = copy.deepcopy(base_trainer.critic_optimizer.state_dict())
    initial_optimizer_record = _optimizer_record(base_trainer.critic_optimizer)
    initial_optimizer_state_hash = _optimizer_state_hash(base_trainer.critic_optimizer)
    initial_welford = _welford_record(base_trainer.critic_return_statistics)
    initial_popart = list(base_trainer.policy.popart_statistics())
    if initial_welford != {"count": 0, "mean": 0.0, "m2": 0.0} or initial_popart != [0.0, 1.0]:
        raise AssertionError("fresh Critic PopArt/Welford state is not neutral")
    for parameter in base_trainer.policy.actor_parameters():
        parameter.requires_grad_(False)
    initial_state_metadata = _persist_initial_critic_state(
        base_trainer,
        initial_critic_state,
        initial_optimizer_state,
        critic_digest=initial_critic_digest,
        actor_digest=initial_actor_digest,
        config_sha256=config_sha,
        initialization_seed=initialization_seed,
    )

    arm_a = _make_critic_arm(base_trainer, initial_optimizer_state, "cpu")
    arm_b = _make_critic_arm(base_trainer, initial_optimizer_state, "cpu")
    for arm in (arm_a, arm_b):
        missing, unexpected = arm.policy.load_state_dict(copy.deepcopy(initial_critic_state), strict=False)
        if unexpected or set(initial_critic_state).intersection(missing):
            raise AssertionError("saved initial Critic state did not load into an independent arm")
    initialization = _assert_initialization_equivalence(
        arm_a,
        arm_b,
        initial_optimizer_record,
        initial_critic_digest,
        initial_actor_digest,
    )
    RUN_STATE["initialization_equivalence_gate"] = "PASS"
    initial_critic_state.clear()

    def forbidden_formal_update(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("trainer.update() is forbidden in this diagnostic experiment")

    base_trainer.update = forbidden_formal_update  # type: ignore[method-assign]
    actor_parameters_before = tuple(
        parameter.detach().clone() for parameter in base_trainer.policy.actor_parameters()
    )

    RUN_STATE["stage"] = "SHARED_TRAIN_ROLLOUT"
    train_archive = _load_recovery_archive(
        TRAIN_ROLLOUT_PATH,
        TRAIN_ROLLOUT_METADATA_PATH,
        kind="SHARED_TRAIN_ROLLOUT",
        episode_seed=TRAIN_REALIZATION_SEED,
        rollout_seed=training_rollout_seed,
        actor_digest=initial_actor_digest,
        initial_critic_digest=initial_critic_digest,
        config_sha256=config_sha,
        required_arrays={"observations", "actions", "rewards", "dones", "rollout_time_raw_values"},
    )
    if train_archive is None:
        print("Collecting the single compensatory shared 300-transition TRAIN rollout.", flush=True)
        training_rollout = _run_with_isolated_rng(
            training_rollout_seed,
            lambda: base_trainer.collect_chunk(
                train_context.train_origin_start,
                train_context.initial_cache,
                train_context.initial_uav,
                train_context.configured_rollout_steps,
            ),
        )
        training_arrays = {
            "observations": np.asarray(training_rollout.observations),
            "actions": np.asarray(training_rollout.actions),
            "rewards": np.asarray(training_rollout.rewards),
            "dones": np.asarray(training_rollout.dones, dtype=bool),
            "rollout_time_raw_values": np.asarray(training_rollout.old_values),
        }
        RUN_STATE["train_rollout_collection_count_this_run"] = 1
        training_arrays, train_archive_metadata = _persist_and_reload_recovery_archive(
            TRAIN_ROLLOUT_PATH,
            TRAIN_ROLLOUT_METADATA_PATH,
            training_arrays,
            kind="SHARED_TRAIN_ROLLOUT",
            episode_seed=TRAIN_REALIZATION_SEED,
            rollout_seed=training_rollout_seed,
            actor_digest=initial_actor_digest,
            initial_critic_digest=initial_critic_digest,
            config_sha256=config_sha,
        )
    else:
        training_arrays, train_archive_metadata = train_archive
    RUN_STATE["stage"] = "SHARED_TRAIN_ROLLOUT_PERSISTED"
    if (
        len(training_arrays["observations"]) != 300
        or len(training_arrays["rewards"].reshape(-1)) != 300
        or not bool(training_arrays["dones"].reshape(-1)[-1])
        or bool(np.any(training_arrays["dones"].reshape(-1)[:-1]))
    ):
        raise AssertionError("persisted TRAIN rollout terminal structure/length failed")
    if _actor_digest(base_trainer.policy) != initial_actor_digest:
        raise AssertionError("Actor changed during TRAIN rollout collection")
    if base_trainer.actor_optimizer_steps != 0 or base_trainer.critic_optimizer_steps != 0:
        raise AssertionError("base trainer optimizer stepped during TRAIN rollout collection")
    rollout_critic_values = _evaluate_values(
        base_trainer.policy, training_arrays["observations"], "cpu"
    )
    if not np.allclose(
        rollout_critic_values,
        training_arrays["rollout_time_raw_values"].reshape(-1),
        rtol=1.0e-6,
        atol=1.0e-5,
    ):
        raise AssertionError("persisted rollout-time raw values differ from the untouched Critic")
    train_persistence_gate = "PASS"

    RUN_STATE["stage"] = "TARGET_CONSTRUCTION_AND_PERSISTENCE"
    bootstrap_value = 0.0
    advantages_a, targets_a = compute_gae(
        training_arrays["rewards"],
        training_arrays["rollout_time_raw_values"],
        training_arrays["dones"],
        bootstrap_value,
        gamma,
        gae_lambda,
    )
    targets_b = _mc_reward_to_go(
        training_arrays["rewards"], training_arrays["dones"], gamma
    )
    if (
        len(targets_a) != 300
        or len(targets_b) != 300
        or not np.all(np.isfinite(advantages_a))
        or not np.all(np.isfinite(targets_a))
        or not np.all(np.isfinite(targets_b))
    ):
        raise FloatingPointError("Critic target length or finite-value assertion failed")
    targets_a, targets_b, target_hashes, target_metadata = _persist_targets(
        targets_a,
        targets_b,
        config_sha256=config_sha,
        training_rollout_sha256=train_archive_metadata["npz_sha256"],
    )
    if len(targets_a) != 300 or len(targets_b) != 300:
        raise AssertionError("reloaded target arrays do not each contain 300 entries")
    sequence, sequence_sha, minibatch_rng_state_sha = _persist_minibatch_sequence(
        minibatch_seed,
        300,
        int(hyper.update_epochs),
        int(hyper.num_minibatches),
    )
    sequence_a = sequence
    sequence_b = sequence
    sequence_hash_a = _minibatch_sequence_digest(sequence_a)
    sequence_hash_b = _minibatch_sequence_digest(sequence_b)
    if sequence_hash_a != sequence_hash_b or sequence_sha != sequence_hash_a:
        raise AssertionError("shared minibatch sequence equivalence failed")
    if base_trainer.device.type != "cpu":
        raise AssertionError("matched minibatch sequence requires CPU execution")
    minibatch_equivalence_gate = "PASS"

    epsilon = float(hyper.critic_value_scale_epsilon)
    warmup_samples = int(hyper.critic_value_scale_warmup_samples)
    prep_a = _prepare_arm_targets(arm_a, targets_a, training_arrays["observations"], epsilon, warmup_samples)
    prep_b = _prepare_arm_targets(arm_b, targets_b, training_arrays["observations"], epsilon, warmup_samples)
    normalized_a = prep_a["normalized_targets"]
    normalized_b = prep_b["normalized_targets"]
    prep_a_record = {key: value for key, value in prep_a.items() if key != "normalized_targets"}
    prep_b_record = {key: value for key, value in prep_b.items() if key != "normalized_targets"}

    seeds = {
        "trainer_initialization_seed": int(initialization_seed),
        "training_rollout_seed": int(training_rollout_seed),
        "optimizer_seed": int(optimizer_seed),
        "minibatch_seed": int(minibatch_seed),
        "diagnostic_generation_seed": int(diagnostic_generation_seed),
        "diagnostic_rollout_seed": int(diagnostic_rollout_seed),
    }

    def train_or_restore_arm(
        *,
        arm: Any,
        arm_name: str,
        target_type: str,
        target_bootstrap: str,
        target: np.ndarray,
        normalized_target: np.ndarray,
        prep_record: dict[str, Any],
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        stem = "gae_target_critic" if arm_name == "ARM_A" else "mc_target_critic"
        checkpoint_path = OUTPUT_DIR / f"{stem}.pt"
        metadata_path = OUTPUT_DIR / f"{stem}.metadata.json"
        target_sha = _array_sha256(target)
        if checkpoint_path.is_file():
            RUN_STATE["stage"] = f"{arm_name}_CHECKPOINT_RECOVERY"
            metadata = _restore_checkpoint_into_arm(
                checkpoint_path=checkpoint_path,
                metadata_path=metadata_path,
                arm=arm,
                target_type=target_type,
                target_sha256=target_sha,
                sequence_sha256=sequence_sha,
                expected_initial_critic_digest=initial_critic_digest,
                expected_actor_digest=initial_actor_digest,
            )
            training = {
                "critic_optimizer_steps": int(arm.critic_optimizer_steps),
                "actor_optimizer_steps": 0,
                "actor_parameter_digest_before": initial_actor_digest,
                "actor_parameter_digest_after": _actor_digest(arm.policy),
                "actor_parameter_delta": 0.0,
                "minibatch_index_sequence_sha256": sequence_sha,
                "recovered_from_checkpoint": "YES",
                "normalized_value_loss_mean": None,
                "critic_gradient_norm_max": None,
            }
            RUN_STATE["critic_steps_a" if arm_name == "ARM_A" else "critic_steps_b"] = 40
            return arm, training, {**metadata, "checkpoint_path": str(checkpoint_path), "metadata_path": str(metadata_path), "checkpoint_reload_gate": "PASS"}

        RUN_STATE["stage"] = f"{arm_name}_CRITIC_ONLY_TRAINING"
        try:
            training = _train_arm(
                arm,
                training_arrays["observations"],
                target,
                normalized_target,
                sequence,
                int(hyper.update_epochs) * int(hyper.num_minibatches),
            )
        except BaseException:
            RUN_STATE["critic_steps_a" if arm_name == "ARM_A" else "critic_steps_b"] = int(arm.critic_optimizer_steps)
            raise
        RUN_STATE["critic_steps_a" if arm_name == "ARM_A" else "critic_steps_b"] = int(arm.critic_optimizer_steps)
        if int(arm.critic_optimizer_steps) != 40:
            raise AssertionError(f"{arm_name} Critic step count is not exactly 40")
        checkpoint = _save_critic_checkpoint(
            arm=arm,
            arm_name=arm_name,
            target_type=target_type,
            target_bootstrap=target_bootstrap,
            target_sha256=target_sha,
            sequence_sha256=sequence_sha,
            train_provenance=train_provenance,
            seeds={"training_rollout_seed": training_rollout_seed, "optimizer_seed": optimizer_seed},
            config_sha256=config_sha,
            initial_critic_digest=initial_critic_digest,
            actor_digest=initial_actor_digest,
        )
        verifier = _make_critic_arm(base_trainer, initial_optimizer_state, "cpu")
        verifier_prep = _prepare_arm_targets(
            verifier, target, training_arrays["observations"], epsilon, warmup_samples
        )
        verifier_prep.pop("normalized_targets")
        reloaded_metadata = _restore_checkpoint_into_arm(
            checkpoint_path=checkpoint_path,
            metadata_path=metadata_path,
            arm=verifier,
            target_type=target_type,
            target_sha256=target_sha,
            sequence_sha256=sequence_sha,
            expected_initial_critic_digest=initial_critic_digest,
            expected_actor_digest=initial_actor_digest,
        )
        if _critic_digest(verifier.policy) != _critic_digest(arm.policy):
            raise AssertionError(f"{arm_name} checkpoint reload did not reproduce the trained Critic")
        checkpoint.update(reloaded_metadata)
        checkpoint["checkpoint_reload_gate"] = "PASS"
        return verifier, {**training, **prep_record}, checkpoint

    arm_a, arm_a_training, checkpoint_a = train_or_restore_arm(
        arm=arm_a,
        arm_name="ARM_A",
        target_type="GAE_LAMBDA_RETURN",
        target_bootstrap="YES",
        target=targets_a,
        normalized_target=normalized_a,
        prep_record=prep_a_record,
    )
    if arm_a_training["critic_optimizer_steps"] != 40 or checkpoint_a["checkpoint_reload_gate"] != "PASS":
        raise AssertionError("ARM_A must complete and reload its 40-step checkpoint before ARM_B")
    arm_b, arm_b_training, checkpoint_b = train_or_restore_arm(
        arm=arm_b,
        arm_name="ARM_B",
        target_type="FULL_EPISODE_MC_RETURN",
        target_bootstrap="NO",
        target=targets_b,
        normalized_target=normalized_b,
        prep_record=prep_b_record,
    )
    if arm_b_training["critic_optimizer_steps"] != 40 or checkpoint_b["checkpoint_reload_gate"] != "PASS":
        raise AssertionError("ARM_B must complete and reload its 40-step checkpoint")
    if sequence_hash_a != sequence_hash_b:
        raise AssertionError("ARM_A and ARM_B consumed different minibatch sequences")
    if _actor_digest(arm_a.policy) != _actor_digest(arm_b.policy) or _actor_digest(arm_a.policy) != initial_actor_digest:
        raise AssertionError("Actor freeze assertion failed after Critic-only training")
    if base_trainer.actor_optimizer_steps != 0 or base_trainer.critic_optimizer_steps != 0:
        raise AssertionError("base trainer performed an optimizer step")

    RUN_STATE["stage"] = "DIAGNOSTIC_REALIZATION_PREFLIGHT"
    existing_diagnostic = _find_diagnostic_realization(
        generator, cfg, generation_fingerprint, train_context
    )
    if existing_diagnostic is None:
        diagnostic_action = "GENERATE"
        diagnostic_dir = DIAGNOSTIC_REALIZATION_PATH / f"episode_{DIAGNOSTIC_REALIZATION_SEED}"
        if diagnostic_dir.exists():
            raise RuntimeError("fixed diagnostic realization path exists but is incompatible; refusing overwrite")
        print("Generating the fixed independent TRAIN diagnostic realization once.", flush=True)
        RUN_STATE["diagnostic_realization_action"] = diagnostic_action
        _run_with_isolated_rng(
            diagnostic_generation_seed,
            lambda: _ensure_episode_realization(
                train_context,
                str(CONFIG_PATH),
                diagnostic_dir,
                DIAGNOSTIC_REALIZATION_SEED,
                generate_missing=True,
                generation_device="cpu",
                generator_module=generator,
            ),
        )
    else:
        diagnostic_action = "LOAD"
        diagnostic_dir = existing_diagnostic
        RUN_STATE["diagnostic_realization_action"] = diagnostic_action
    normalized_diagnostic = str(diagnostic_dir).lower()
    if "proposed_joint_validation" in normalized_diagnostic or "proposed_joint_test" in normalized_diagnostic:
        raise AssertionError("diagnostic realization resolved to a held-out split")
    diagnostic_context = load_environment_context(str(CONFIG_PATH), realization_dir=diagnostic_dir)
    _assert_realization_context_compatible(train_context, diagnostic_context)
    if diagnostic_context.data_protocol != "full_episode_randomized" or diagnostic_context.episode_slots != 300:
        raise AssertionError("diagnostic realization is not a complete TRAIN 300-slot protocol")
    _validate_episode_realization_complete(
        diagnostic_dir, diagnostic_context, generator, DIAGNOSTIC_REALIZATION_SEED
    )
    diagnostic_provenance = _realization_provenance(
        directory=diagnostic_dir,
        episode_seed=DIAGNOSTIC_REALIZATION_SEED,
        split="TRAIN",
        action=diagnostic_action,
        current_config_sha=config_sha,
        current_fingerprint=generation_fingerprint,
        split_manifest=None,
        generator_module=generator,
        cfg=cfg,
        isolated_generation_rng_seed=(diagnostic_generation_seed if diagnostic_action == "GENERATE" else None),
    )
    diagnostic_provenance["split_evidence"] = (
        "FIXED_COUNTERFACTUAL_RELIABILITY_WITNESS_TRAIN_IDENTITY"
        if diagnostic_action == "LOAD"
        else "AUTHORIZED_GENERATION_OF_FIXED_COUNTERFACTUAL_RELIABILITY_WITNESS_TRAIN_IDENTITY"
    )
    diagnostic_provenance["independent_from_training_realization"] = "YES"
    diagnostic_provenance["explicit_config_conflicts"] = []
    base_trainer.context = diagnostic_context

    RUN_STATE["stage"] = "DIAGNOSTIC_ROLLOUT"
    diagnostic_archive = _load_recovery_archive(
        DIAGNOSTIC_ROLLOUT_PATH,
        DIAGNOSTIC_ROLLOUT_METADATA_PATH,
        kind="INDEPENDENT_TRAIN_DIAGNOSTIC_ROLLOUT",
        episode_seed=DIAGNOSTIC_REALIZATION_SEED,
        rollout_seed=diagnostic_rollout_seed,
        actor_digest=initial_actor_digest,
        initial_critic_digest=initial_critic_digest,
        config_sha256=config_sha,
        required_arrays={"observations", "actions", "rewards", "dones", "transition_149_pickle"},
    )
    if diagnostic_archive is None:
        actor_before_diagnostic = _actor_digest(base_trainer.policy)
        print("Collecting the single independent 300-transition TRAIN diagnostic rollout.", flush=True)
        diagnostic_rollout = _run_with_isolated_rng(
            diagnostic_rollout_seed,
            lambda: base_trainer.collect_chunk(
                diagnostic_context.train_origin_start,
                diagnostic_context.initial_cache,
                diagnostic_context.initial_uav,
                diagnostic_context.configured_rollout_steps,
            ),
        )
        transition_blob = pickle.dumps(
            diagnostic_rollout.transitions[TARGET_TRANSITION], protocol=5
        )
        diagnostic_arrays = {
            "observations": np.asarray(diagnostic_rollout.observations),
            "actions": np.asarray(diagnostic_rollout.actions),
            "rewards": np.asarray(diagnostic_rollout.rewards),
            "dones": np.asarray(diagnostic_rollout.dones, dtype=bool),
            "transition_149_pickle": np.frombuffer(transition_blob, dtype=np.uint8).copy(),
        }
        RUN_STATE["diagnostic_rollout_collection_count_this_run"] = 1
        diagnostic_arrays, diagnostic_archive_metadata = _persist_and_reload_recovery_archive(
            DIAGNOSTIC_ROLLOUT_PATH,
            DIAGNOSTIC_ROLLOUT_METADATA_PATH,
            diagnostic_arrays,
            kind="INDEPENDENT_TRAIN_DIAGNOSTIC_ROLLOUT",
            episode_seed=DIAGNOSTIC_REALIZATION_SEED,
            rollout_seed=diagnostic_rollout_seed,
            actor_digest=initial_actor_digest,
            initial_critic_digest=initial_critic_digest,
            config_sha256=config_sha,
        )
        if _actor_digest(base_trainer.policy) != actor_before_diagnostic:
            raise AssertionError("Actor parameters changed during diagnostic rollout collection")
    else:
        diagnostic_arrays, diagnostic_archive_metadata = diagnostic_archive
    RUN_STATE["stage"] = "DIAGNOSTIC_ROLLOUT_PERSISTED"
    if (
        len(diagnostic_arrays["observations"]) != 300
        or len(diagnostic_arrays["rewards"].reshape(-1)) != 300
        or not bool(diagnostic_arrays["dones"].reshape(-1)[-1])
        or bool(np.any(diagnostic_arrays["dones"].reshape(-1)[:-1]))
    ):
        raise AssertionError("persisted diagnostic rollout terminal structure/length failed")
    if base_trainer.actor_optimizer_steps != 0 or base_trainer.critic_optimizer_steps != 0:
        raise AssertionError("base trainer optimizer counters changed during diagnostic collection")

    RUN_STATE["stage"] = "INDEPENDENT_TRAIN_METRICS"
    mc_diagnostic_returns = _mc_reward_to_go(
        diagnostic_arrays["rewards"], diagnostic_arrays["dones"], gamma
    )
    predictions_a = _evaluate_values(arm_a.policy, diagnostic_arrays["observations"], "cpu")
    predictions_b = _evaluate_values(arm_b.policy, diagnostic_arrays["observations"], "cpu")
    metrics_a = _metrics(predictions_a, mc_diagnostic_returns)
    metrics_b = _metrics(predictions_b, mc_diagnostic_returns)
    for metrics in (metrics_a, metrics_b):
        for key in ("mae", "rmse", "explained_variance", "pearson", "spearman"):
            if not isinstance(metrics[key], (int, float)) or not math.isfinite(float(metrics[key])):
                raise FloatingPointError(f"diagnostic metric is non-finite or undefined: {key}")
    _array_sha256(mc_diagnostic_returns)

    RUN_STATE["stage"] = "UAV1_BRANCH_REPRODUCTION"
    transition_bytes = diagnostic_arrays["transition_149_pickle"].astype(np.uint8, copy=False).tobytes()
    transition = pickle.loads(transition_bytes)
    target_origin = int(transition.origin_index)
    if target_origin != TARGET_TRANSITION:
        raise AssertionError(f"fixed witness origin changed: {target_origin}")
    witness_steps = aligned_train_steps(
        diagnostic_context,
        origin_start=target_origin,
        max_steps=2,
        include_bootstrap_step=False,
    )
    if len(witness_steps) != 2 or int(witness_steps[0].origin_index) != TARGET_TRANSITION:
        raise AssertionError("fixed UAV_1 witness current/next aligned steps are unavailable")
    witness_step, next_step = witness_steps
    fingerprint = _transition_fingerprint(transition, witness_step, config_sha)
    if fingerprint != EXPECTED_TRANSITION_FINGERPRINT:
        raise AssertionError(
            "reconstructed transition-149 fingerprint differs from paired-MC witness: "
            f"{fingerprint}"
        )
    node_ids = [str(value) for value in diagnostic_context.action_spec.cache_node_ids]
    if TARGET_HEAD not in node_ids:
        raise AssertionError("fixed UAV_1 head is absent from the production ActionSpec")
    node_index = node_ids.index(TARGET_HEAD)
    working_cache, previous_cache = cache_credit._replay_cache_prefix(transition, node_index)
    candidate = cache_configuration.CacheTargetCandidate(
        node_index=int(node_index),
        node_id=TARGET_HEAD,
        before_node_cache=np.asarray(working_cache[node_index], dtype=np.int8).copy(),
        target_node_cache=np.asarray(TARGET_CONFIG, dtype=np.int8),
        proposal_source="fixed_existing_UAV1_reliability_witness",
    )
    branch_state = {"working_cache": working_cache, "previous_cache": previous_cache}
    print("Reconstructing only the fixed UAV_1 KEEP and Candidate immediate branches.", flush=True)
    keep_branch = cache_credit._execute_cache_branch(
        base_trainer, transition, witness_step, next_step, node_index, branch_state, None
    )
    candidate_branch = cache_credit._execute_cache_branch(
        base_trainer, transition, witness_step, next_step, node_index, branch_state, candidate
    )
    next_observation_keep = keep_branch.get("next_observation")
    next_observation_candidate = candidate_branch.get("next_observation")
    if next_observation_keep is None or next_observation_candidate is None:
        raise AssertionError("production branch builder did not return both next observations")
    expected_observation_shape = (diagnostic_context.observation_spec.observation_dim,)
    if (
        np.asarray(next_observation_keep).shape != expected_observation_shape
        or np.asarray(next_observation_candidate).shape != expected_observation_shape
    ):
        raise AssertionError("fixed UAV_1 branch next-observation shape is invalid")
    keep_batch = np.stack([next_observation_keep]).astype(np.float32, copy=False)
    candidate_batch = np.stack([next_observation_candidate]).astype(np.float32, copy=False)
    branch_values = {
        "arm_a_keep_value": float(_evaluate_values(arm_a.policy, keep_batch, "cpu")[0]),
        "arm_a_candidate_value": float(_evaluate_values(arm_a.policy, candidate_batch, "cpu")[0]),
        "arm_b_keep_value": float(_evaluate_values(arm_b.policy, keep_batch, "cpu")[0]),
        "arm_b_candidate_value": float(_evaluate_values(arm_b.policy, candidate_batch, "cpu")[0]),
    }
    if not all(math.isfinite(value) for value in branch_values.values()):
        raise FloatingPointError("fixed UAV_1 branch values contain NaN or Inf")
    delta_a = branch_values["arm_a_candidate_value"] - branch_values["arm_a_keep_value"]
    delta_b = branch_values["arm_b_candidate_value"] - branch_values["arm_b_keep_value"]
    empirical_mc_mean = 0.5 * (MC_DELTA_R0 + MC_DELTA_R1)
    witness = {
        "target_transition": TARGET_TRANSITION,
        "target_origin_index": target_origin,
        "target_head": TARGET_HEAD,
        "target_config": TARGET_CONFIG,
        "transition_fingerprint_sha256": fingerprint,
        "branch_reproduction_gate": "PASS",
        "immediate_branch_execution_count": 2,
        "terminal_continuation_count": 0,
        "keep_next_observation_sha256": _array_sha256(next_observation_keep),
        "candidate_next_observation_sha256": _array_sha256(next_observation_candidate),
        **branch_values,
        "arm_a_delta_v": float(delta_a),
        "arm_b_delta_v": float(delta_b),
        "arm_a_delta_v_sign": _sign(delta_a),
        "arm_b_delta_v_sign": _sign(delta_b),
        "empirical_mc_delta_g_r0": MC_DELTA_R0,
        "empirical_mc_delta_g_r1": MC_DELTA_R1,
        "empirical_mc_sign_r0": "NEGATIVE",
        "empirical_mc_sign_r1": "NEGATIVE",
        "empirical_mc_mean": float(empirical_mc_mean),
        "arm_a_sign_agreement": _sign(delta_a) == "NEGATIVE",
        "arm_b_sign_agreement": _sign(delta_b) == "NEGATIVE",
        "arm_a_abs_delta_v_error_to_mc_mean": float(abs(delta_a - empirical_mc_mean)),
        "arm_b_abs_delta_v_error_to_mc_mean": float(abs(delta_b - empirical_mc_mean)),
    }

    arm_a_result = {
        "target": "GAE_LAMBDA_RETURN",
        "target_uses_bootstrap": "YES",
        "target_array": _array_record(targets_a),
        "training": arm_a_training,
        "critic_digest": _critic_digest(arm_a.policy),
        "popart": {"mean": float(arm_a.policy.popart_statistics()[0]), "std": float(arm_a.policy.popart_statistics()[1])},
        "welford": _welford_record(arm_a.critic_return_statistics),
        "metrics": metrics_a,
        "checkpoint": checkpoint_a,
    }
    arm_b_result = {
        "target": "FULL_EPISODE_MC_RETURN",
        "target_uses_bootstrap": "NO",
        "target_array": _array_record(targets_b),
        "training": arm_b_training,
        "critic_digest": _critic_digest(arm_b.policy),
        "popart": {"mean": float(arm_b.policy.popart_statistics()[0]), "std": float(arm_b.policy.popart_statistics()[1])},
        "welford": _welford_record(arm_b.critic_return_statistics),
        "metrics": metrics_b,
        "checkpoint": checkpoint_b,
    }
    gates_pass = bool(
        initialization["status"] == "PASS"
        and initial_critic_digest == EXPECTED_INITIAL_CRITIC_DIGEST
        and sequence_hash_a == sequence_hash_b
        and arm_a_training["critic_optimizer_steps"] == arm_b_training["critic_optimizer_steps"] == 40
        and arm_a_training["actor_parameter_delta"] == arm_b_training["actor_parameter_delta"] == 0.0
        and checkpoint_a["checkpoint_reload_gate"] == checkpoint_b["checkpoint_reload_gate"] == "PASS"
        and prep_a_record["raw_value_preservation_finite"] == prep_b_record["raw_value_preservation_finite"] == "YES"
        and all(
            math.isfinite(float(metrics[key]))
            for metrics in (metrics_a, metrics_b)
            for key in ("mae", "rmse", "explained_variance", "pearson", "spearman")
        )
    )
    experiment_result, rationale = _classify_result(arm_a_result, arm_b_result, witness, gates_pass)
    if experiment_result == "SUPPORTED":
        next_stage = "CRITIC_TARGET_ALIGNMENT_PRODUCTION_REPAIR_DESIGN"
    elif experiment_result in {"PARTIAL", "NOT_SUPPORTED"}:
        next_stage = "CRITIC_DATA_COVERAGE_MINIMAL_EXPERIMENT"
    else:
        next_stage = "CRITIC_TARGET_ALIGNMENT_EXPERIMENT_FOLLOWUP"

    report = {
        "TASK": TASK_NAME,
        "ATTEMPT": CURRENT_ATTEMPT,
        "CURRENT_HEAD": current_head,
        "CONFIG_PATH": str(CONFIG_PATH),
        "CONFIG_SHA256": config_sha,
        "GENERATION_CONFIG_FINGERPRINT": generation_fingerprint,
        "AUTHORIZED_COMPENSATORY_EXPERIMENT_RERUN": "YES",
        "RUNNER_CONFIGURATION_PREFLIGHT": "PASS",
        "PRODUCTION_SOURCE_CHANGED": "NO",
        "PRODUCTION_CRITIC_TARGET_CHANGED": "NO",
        "PRODUCTION_CRITIC_TARGET": "GAE_LAMBDA_RETURN",
        "FORMAL_PPO_UPDATE_EXECUTED": "NO",
        "ONE_EPOCH_TRAINING_EXECUTED": "NO",
        "ONE_EPOCH_COMPLETED_EPOCHS": 0,
        "LAST_COMPLETED_EPOCH": "NONE",
        "EPOCH_2_STARTED": "NO",
        "FORMAL_300_EPOCH_TRAINING_EXECUTED": "NO",
        "READY_FOR_FORMAL_TRAINING_STAGE": "NO",
        "TRAIN_REALIZATION_ACTION": "LOAD",
        "TRAIN_REALIZATION_SEED": TRAIN_REALIZATION_SEED,
        "TRAIN_REALIZATION_REUSE_EXISTING": "YES",
        "TRAIN_REALIZATION_GENERATED": "NO",
        "TRAIN_REALIZATION": train_provenance,
        "TRAIN_ROLLOUT_COLLECTION_COUNT_THIS_RUN": int(RUN_STATE["train_rollout_collection_count_this_run"]),
        "TRAIN_ROLLOUT_COLLECTION_COUNT": int(train_archive_metadata["collection_count"]),
        "SHARED_TRAIN_ROLLOUT_PERSISTENCE_GATE": train_persistence_gate,
        "SHARED_TRAIN_ROLLOUT_SHA256": train_archive_metadata["npz_sha256"],
        "SHARED_TRAIN_ROLLOUT_METADATA_SHA256": _sha256_file(TRAIN_ROLLOUT_METADATA_PATH),
        "TRAIN_ROLLOUT_ARRAYS": train_archive_metadata["arrays"],
        "TRAIN_ROLLOUT_TERMINAL_STRUCTURE": train_archive_metadata["terminal_structure"],
        "TRAIN_ROLLOUT_TARGET_ARRAY_SHA256": {
            "ARM_A_TARGET_SHA256": target_hashes["arm_a_gae_lambda_return"],
            "ARM_B_TARGET_SHA256": target_hashes["arm_b_full_episode_mc_return"],
        },
        "ARM_A_TARGET": "GAE_LAMBDA_RETURN",
        "ARM_A_TARGET_USES_BOOTSTRAP": "YES",
        "ARM_A_TARGET_SHA256": target_hashes["arm_a_gae_lambda_return"],
        "ARM_B_TARGET": "FULL_EPISODE_MC_RETURN",
        "ARM_B_TARGET_USES_BOOTSTRAP": "NO",
        "ARM_B_TARGET_SHA256": target_hashes["arm_b_full_episode_mc_return"],
        "TARGET_ARRAYS_METADATA": target_metadata,
        "INITIALIZATION_REPRODUCTION_GATE": "PASS",
        "INITIALIZATION_EQUIVALENCE_GATE": initialization["status"],
        "INITIAL_CRITIC_PARAMETER_DIGEST_EXPECTED": EXPECTED_INITIAL_CRITIC_DIGEST,
        "INITIAL_CRITIC_PARAMETER_DIGEST": initial_critic_digest,
        "INITIAL_CRITIC_STATE_CAPTURE": initial_state_metadata,
        "INITIAL_CRITIC_STATE_PATH": str(INITIAL_CRITIC_STATE_PATH),
        "INITIAL_CRITIC_STATE_METADATA_PATH": str(INITIAL_CRITIC_STATE_METADATA_PATH),
        "INITIAL_CRITIC_STATE_DICT_SHA256": initial_critic_state_hash,
        "INITIAL_CRITIC_OPTIMIZER_STATE_SHA256": initial_optimizer_state_hash,
        "INITIAL_CRITIC_OPTIMIZER_STATE": initial_optimizer_record,
        "INITIAL_POPART_STATE": {"mean": float(initial_popart[0]), "std": float(initial_popart[1])},
        "INITIAL_WELFORD_STATE": initial_welford,
        "INITIALIZATION_EQUIVALENCE_DETAILS": initialization,
        "ARM_A": arm_a_result,
        "ARM_B": arm_b_result,
        "ARM_A_CRITIC_STEPS": int(arm_a.critic_optimizer_steps),
        "ARM_B_CRITIC_STEPS": int(arm_b.critic_optimizer_steps),
        "ARM_A_CHECKPOINT_RELOAD_GATE": checkpoint_a["checkpoint_reload_gate"],
        "ARM_B_CHECKPOINT_RELOAD_GATE": checkpoint_b["checkpoint_reload_gate"],
        "ARM_A_POPART_MEAN": float(arm_a.policy.popart_statistics()[0]),
        "ARM_A_POPART_STD": float(arm_a.policy.popart_statistics()[1]),
        "ARM_A_WELFORD_COUNT": int(arm_a.critic_return_statistics.count),
        "ARM_A_WELFORD_STATE": _welford_record(arm_a.critic_return_statistics),
        "ARM_B_POPART_MEAN": float(arm_b.policy.popart_statistics()[0]),
        "ARM_B_POPART_STD": float(arm_b.policy.popart_statistics()[1]),
        "ARM_B_WELFORD_COUNT": int(arm_b.critic_return_statistics.count),
        "ARM_B_WELFORD_STATE": _welford_record(arm_b.critic_return_statistics),
        "CRITIC_LEARNING_RATE": float(hyper.critic_learning_rate),
        "CRITIC_OPTIMIZER": "Adam",
        "UPDATE_EPOCHS": int(hyper.update_epochs),
        "NUM_MINIBATCHES": int(hyper.num_minibatches),
        "MINIBATCH_SIZE": int(train_context.minibatch_size),
        "GRAD_CLIP_MAX_NORM": float(hyper.max_grad_norm),
        "MINIBATCH_ORDER_EQUIVALENCE_GATE": "PASS",
        "MINIBATCH_INDEX_SEQUENCE_SHA256": sequence_sha,
        "MINIBATCH_INDEX_SEQUENCE_SHA256_ARM_A": sequence_hash_a,
        "MINIBATCH_INDEX_SEQUENCE_SHA256_ARM_B": sequence_hash_b,
        "MINIBATCH_INITIAL_RNG_STATE_SHA256": minibatch_rng_state_sha,
        "MINIBATCH_SEQUENCE_PERSISTED": "YES",
        "TRAINING_SEEDS": seeds,
        "ACTOR_ADVANTAGE_TARGET": "CURRENT_GAE_UNCHANGED",
        "ACTOR_GAE_CHANGED": "NO",
        "ACTOR_PARAMETERS_FROZEN": "YES",
        "ACTOR_PARAMETER_DIGEST_INITIAL": initial_actor_digest,
        "ACTOR_PARAMETER_DIGEST_FINAL": _actor_digest(base_trainer.policy),
        "ACTOR_PARAMETER_DELTA": 0.0,
        "ACTOR_OPTIMIZER_STEPS": 0,
        "DIAGNOSTIC_REALIZATION_ACTION": diagnostic_action,
        "DIAGNOSTIC_REALIZATION_SEED": DIAGNOSTIC_REALIZATION_SEED,
        "DIAGNOSTIC_REALIZATION_REUSE_EXISTING": "YES" if diagnostic_action == "LOAD" else "NO",
        "DIAGNOSTIC_REALIZATION_GENERATED": "NO" if diagnostic_action == "LOAD" else "YES",
        "DIAGNOSTIC_REALIZATION": diagnostic_provenance,
        "DIAGNOSTIC_SPLIT": "TRAIN",
        "DIAGNOSTIC_ROLLOUT_COLLECTION_COUNT_THIS_RUN": int(RUN_STATE["diagnostic_rollout_collection_count_this_run"]),
        "DIAGNOSTIC_ROLLOUT_COLLECTION_COUNT": int(diagnostic_archive_metadata["collection_count"]),
        "DIAGNOSTIC_ROLLOUT_PERSISTENCE_GATE": "PASS",
        "DIAGNOSTIC_ROLLOUT_SHA256": diagnostic_archive_metadata["npz_sha256"],
        "DIAGNOSTIC_ROLLOUT_METADATA_SHA256": _sha256_file(DIAGNOSTIC_ROLLOUT_METADATA_PATH),
        "DIAGNOSTIC_ROLLOUT_ARRAYS": diagnostic_archive_metadata["arrays"],
        "DIAGNOSTIC_MC_RETURN_TARGET": _array_record(mc_diagnostic_returns),
        "ACTUAL_STATE_DIAGNOSTIC": "INDEPENDENT_TRAIN_REALIZATION_MC_REWARD_TO_GO",
        "MC_MAE_IMPROVEMENT": float(metrics_a["mae"] - metrics_b["mae"]),
        "MC_RMSE_IMPROVEMENT": float(metrics_a["rmse"] - metrics_b["rmse"]),
        "MC_EV_IMPROVEMENT": float(metrics_b["explained_variance"] - metrics_a["explained_variance"]),
        "ARM_A_MC_MAE": float(metrics_a["mae"]),
        "ARM_A_MC_RMSE": float(metrics_a["rmse"]),
        "ARM_A_MC_EV": float(metrics_a["explained_variance"]),
        "ARM_A_MC_PEARSON": metrics_a["pearson"],
        "ARM_A_MC_SPEARMAN": metrics_a["spearman"],
        "ARM_B_MC_MAE": float(metrics_b["mae"]),
        "ARM_B_MC_RMSE": float(metrics_b["rmse"]),
        "ARM_B_MC_EV": float(metrics_b["explained_variance"]),
        "ARM_B_MC_PEARSON": metrics_b["pearson"],
        "ARM_B_MC_SPEARMAN": metrics_b["spearman"],
        "UAV1_WITNESS": witness,
        "UAV1_EMPIRICAL_MC_SIGN": "NEGATIVE",
        "ARM_A_UAV1_DELTA_V": float(delta_a),
        "ARM_A_UAV1_SIGN_AGREEMENT": witness["arm_a_sign_agreement"],
        "ARM_A_UAV1_ABS_DELTA_V_ERROR_TO_MC_MEAN": witness["arm_a_abs_delta_v_error_to_mc_mean"],
        "ARM_B_UAV1_DELTA_V": float(delta_b),
        "ARM_B_UAV1_SIGN_AGREEMENT": witness["arm_b_sign_agreement"],
        "ARM_B_UAV1_ABS_DELTA_V_ERROR_TO_MC_MEAN": witness["arm_b_abs_delta_v_error_to_mc_mean"],
        "CRITIC_TARGET_ALIGNMENT_EXPERIMENT_RESULT": experiment_result,
        "DECISION_RATIONALE": rationale,
        "NEXT_STAGE": next_stage,
        "COUNTERFACTUAL_CRITIC_RELIABILITY": "CONCERN",
        "CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION": "CANDIDATE_Q_INTEGRITY_CONCERN",
        "SURROGATE_REPAIR_BLOCKED": "YES",
        "FORMAL_DIAGNOSTIC_CHECKPOINTS_RELOADED": "YES",
        "RECOVERY_ARTIFACT_STATUS": "CLEANUP_PENDING_AFTER_SUCCESS_VALIDATION",
        "RECOVERY_ARTIFACTS": {
            "shared_train_rollout_npz": str(TRAIN_ROLLOUT_PATH),
            "shared_train_rollout_sha256": train_archive_metadata["npz_sha256"],
            "shared_train_rollout_metadata": str(TRAIN_ROLLOUT_METADATA_PATH),
            "diagnostic_rollout_npz": str(DIAGNOSTIC_ROLLOUT_PATH),
            "diagnostic_rollout_sha256": diagnostic_archive_metadata["npz_sha256"],
            "diagnostic_rollout_metadata": str(DIAGNOSTIC_ROLLOUT_METADATA_PATH),
            "targets_npz": str(TARGET_ARRAYS_PATH),
            "targets_npz_sha256": target_metadata["npz_sha256"],
            "targets_metadata": str(TARGET_ARRAYS_METADATA_PATH),
            "minibatch_indices": str(MINIBATCH_SEQUENCE_PATH),
        },
        "TEMPORARY_REALIZATION_PATH": str(diagnostic_dir.resolve()) if diagnostic_action == "GENERATE" else None,
        "TRACEABLE_CRITIC_CHECKPOINT_OVERWRITTEN": "NO",
        "ENVIRONMENT_DATA_MODIFIED": "NO",
        "GIT_COMMIT": "NO",
        "GIT_PUSH": "NO",
    }
    RUN_STATE["stage"] = "FINAL_REPORT_PERSISTENCE"
    _write_followup_reports(report, completed=True)
    if (
        not REPORT_JSON.is_file()
        or json.loads(REPORT_JSON.read_text(encoding="utf-8"))["CURRENT_ATTEMPT"]["attempt"] != CURRENT_ATTEMPT
        or not REPORT_MD.is_file()
    ):
        raise AssertionError("final JSON/Markdown report reload gate failed")

    RUN_STATE["stage"] = "SUCCESS_CLEANUP"
    for path in (TRAIN_ROLLOUT_PATH, DIAGNOSTIC_ROLLOUT_PATH, TARGET_ARRAYS_PATH, MINIBATCH_SEQUENCE_PATH):
        if path.is_file():
            path.unlink()
    report["RECOVERY_ARTIFACT_STATUS"] = "ROLLOUT_TARGET_MINIBATCH_NPZ_REMOVED_AFTER_SUCCESS_METADATA_RETAINED"
    for key in ("shared_train_rollout_npz", "diagnostic_rollout_npz", "targets_npz", "minibatch_indices"):
        report["RECOVERY_ARTIFACTS"][key + "_removed_after_success"] = "YES"
    RUN_STATE["stage"] = "SUCCESS"
    _write_followup_reports(report, completed=True)
    reloaded = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    if (
        reloaded.get("CRITIC_TARGET_ALIGNMENT_EXPERIMENT_RESULT") != experiment_result
        or reloaded.get("ARM_A_CHECKPOINT_RELOAD_GATE") != "PASS"
        or reloaded.get("ARM_B_CHECKPOINT_RELOAD_GATE") != "PASS"
        or reloaded.get("RECOVERY_ARTIFACT_STATUS") != report["RECOVERY_ARTIFACT_STATUS"]
    ):
        raise AssertionError("final experiment report reload validation failed")
    return reloaded


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the authorized Critic target-alignment follow-up.")
    parser.add_argument("--preflight-only", action="store_true", help="resolve runner config and initialization without collecting a rollout")
    args = parser.parse_args()
    try:
        preflight = configuration_preflight()
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "RUNNER_CONFIGURATION_PREFLIGHT": "PASS",
                        "NO_ENVIRONMENT_ROLLOUTS": True,
                        "TRAIN_REALIZATION_ACTION": "LOAD",
                        "TRAIN_REALIZATION_SEED": TRAIN_REALIZATION_SEED,
                        "resolved_config": preflight["config_values"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                ),
                flush=True,
            )
            return 0
        report = run_experiment(preflight)
    except BaseException as exc:
        RUN_STATE["stage"] = RUN_STATE.get("stage", "UNKNOWN")
        _write_failure_report(exc)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    print(
        json.dumps(
            {
                "result": report["CRITIC_TARGET_ALIGNMENT_EXPERIMENT_RESULT"],
                "next_stage": report["NEXT_STAGE"],
                "arm_a_metrics": report["ARM_A"]["metrics"],
                "arm_b_metrics": report["ARM_B"]["metrics"],
                "uav1_delta_v": {
                    "arm_a": report["UAV1_WITNESS"]["arm_a_delta_v"],
                    "arm_b": report["UAV1_WITNESS"]["arm_b_delta_v"],
                },
                "report_json": str(REPORT_JSON),
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
