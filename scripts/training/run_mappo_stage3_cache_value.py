"""Stage3's sole MAPPO training entry using the reviewed Training Pool only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import uuid
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
TRAINING_ROOT = PROJECT_ROOT / "scripts" / "training"
for import_root in (SOURCE_ROOT, TRAINING_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from ppo_simple.cache_counterfactual_value import (  # noqa: E402
    COUNTERFACTUAL_VALUE_SCHEMA,
    TRIGGER_TARGET_SCHEMA,
    Stage3CacheValueConfig,
)
from ppo_simple.mappo_baseline import (  # noqa: E402
    MAPPOBaselineTrainer,
    MAPPO_BASELINE_SCHEMA,
    MAPPO_STAGE3_CHECKPOINT_VERSION,
    CounterfactualLabelBatchError,
)
import mappo_stage3_pool_contract as pool_contract  # noqa: E402


COEFFICIENT_PROFILE = (
    PROJECT_ROOT / "configs" / "research" / "mappo_stage3_loss_coefficients_v1.json"
)
EXPECTED_COEFFICIENT_PROFILE_SHA256 = (
    "c91bccc6bf91f76c407a18c9953b4fa2c2d346fcfef3337ad63ee49b23d8bdb4"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_event(stream: Any, event: dict[str, Any]) -> None:
    stream.write(
        json.dumps(
            _jsonable(event),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    stream.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        help=(
            "Restore an exact current Stage3 checkpoint, or initialize/migrate "
            "a supported MAPPO v2 checkpoint. Older Stage3 checkpoints require "
            "--initialize-from-stage3-checkpoint."
        ),
    )
    parser.add_argument(
        "--initialize-from-stage3-checkpoint",
        type=Path,
        help=(
            "Warm-start model weights from Stage3 v4/v5 while resetting all "
            "optimizers, progress counters and RNG state. This is not resume."
        ),
    )
    parser.add_argument(
        "--initialize-random",
        action="store_true",
        help="Start a fresh policy/value state instead of loading a checkpoint.",
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=PROJECT_ROOT / "runs" / "research_mappo_stage3_cache_value",
    )
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--slots-per-update", type=int, default=40)
    parser.add_argument("--training-seed-start", type=int, default=20001)
    parser.add_argument("--trainer-seed", type=int, default=71301)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--coefficient-profile",
        type=Path,
        default=COEFFICIENT_PROFILE,
        help=(
            "Load the frozen Stage3 coefficient profile; its SHA-256 and values "
            "are checked and recorded in run metadata and Checkpoints."
        ),
    )
    parser.add_argument("--max-macro-states-per-rollout", type=int, default=4)
    parser.add_argument("--max-target-nodes-per-state", type=int, default=2)
    return parser


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"Duplicate JSON key in Stage3 coefficient profile: {key!r}."
            )
        result[key] = value
    return result


def _load_stage3_coefficient_profile(
    path: Path,
    *,
    max_macro_states_per_rollout: int,
    max_target_nodes_per_state: int,
) -> tuple[Stage3CacheValueConfig, dict[str, Any]]:
    """Load a strict, hashed, independently versioned Stage3 coefficient profile."""

    source = path.expanduser().resolve()
    raw = source.read_bytes()
    canonical_raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid Stage3 coefficient profile {source}: {exc}") from exc
    expected_top_level = {
        "profile_schema",
        "profile_id",
        "algorithm_schema",
        "loss_coefficients",
        "value_head_learning_rate",
        "tie_tolerances",
        "pairwise_temperature",
        "huber_delta",
    }
    if not isinstance(payload, dict) or set(payload) != expected_top_level:
        raise ValueError(
            "Stage3 coefficient profile has missing or unknown top-level fields."
        )
    if payload["profile_schema"] != "mappo_stage3_loss_coefficient_profile_v1":
        raise ValueError("Unsupported Stage3 coefficient profile schema.")
    if payload["algorithm_schema"] != COUNTERFACTUAL_VALUE_SCHEMA:
        raise ValueError("Stage3 coefficient profile algorithm schema mismatch.")
    if not isinstance(payload["profile_id"], str) or not payload["profile_id"].strip():
        raise ValueError("Stage3 coefficient profile_id must be a non-empty string.")
    coefficients = payload["loss_coefficients"]
    tolerance_values = payload["tie_tolerances"]
    if not isinstance(coefficients, dict) or set(coefficients) != {
        "trigger_ranking",
        "candidate_ranking",
        "delta_q_huber",
    }:
        raise ValueError(
            "Stage3 coefficient profile loss_coefficients schema mismatch."
        )
    if not isinstance(tolerance_values, dict) or set(tolerance_values) != {
        "trigger",
        "candidate",
    }:
        raise ValueError("Stage3 coefficient profile tie_tolerances schema mismatch.")

    numeric_values = (
        *coefficients.values(),
        payload["value_head_learning_rate"],
        *tolerance_values.values(),
        payload["pairwise_temperature"],
        payload["huber_delta"],
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric_values
    ):
        raise ValueError(
            "Stage3 coefficient profile numeric fields must be finite numbers."
        )

    config = Stage3CacheValueConfig(
        trigger_ranking_coefficient=float(coefficients["trigger_ranking"]),
        candidate_ranking_coefficient=float(coefficients["candidate_ranking"]),
        delta_q_huber_coefficient=float(coefficients["delta_q_huber"]),
        value_head_learning_rate=float(payload["value_head_learning_rate"]),
        trigger_tie_tolerance=float(tolerance_values["trigger"]),
        candidate_tie_tolerance=float(tolerance_values["candidate"]),
        pairwise_temperature=float(payload["pairwise_temperature"]),
        huber_delta=float(payload["huber_delta"]),
        max_macro_states_per_rollout=int(max_macro_states_per_rollout),
        max_target_nodes_per_state=int(max_target_nodes_per_state),
    )
    config.validate()
    metadata = {
        "profile_id": payload["profile_id"],
        "profile_schema": payload["profile_schema"],
        "algorithm_schema": payload["algorithm_schema"],
        "profile_path": str(source),
        "profile_sha256": hashlib.sha256(canonical_raw).hexdigest(),
        "profile_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "profile_contents": payload,
    }
    if metadata["profile_sha256"] != EXPECTED_COEFFICIENT_PROFILE_SHA256:
        raise ValueError(
            "Stage3 coefficient profile SHA-256 mismatch: "
            f"{metadata['profile_sha256']} != "
            f"{EXPECTED_COEFFICIENT_PROFILE_SHA256}."
        )
    return config, metadata


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    initialization_modes = (
        args.source_checkpoint is not None,
        args.initialize_from_stage3_checkpoint is not None,
        bool(args.initialize_random),
    )
    if sum(initialization_modes) != 1:
        parser.error(
            "Choose exactly one of --source-checkpoint, "
            "--initialize-from-stage3-checkpoint, or --initialize-random."
        )
    if args.updates <= 0 or args.slots_per_update <= 0 or args.epochs <= 0:
        parser.error("updates, slots-per-update and epochs must be positive.")
    stage3_config, stage3_profile_metadata = _load_stage3_coefficient_profile(
        args.coefficient_profile,
        max_macro_states_per_rollout=args.max_macro_states_per_rollout,
        max_target_nodes_per_state=args.max_target_nodes_per_state,
    )
    stage3_config.validate()
    formal_root = PROJECT_ROOT
    train_pool, pool_info = pool_contract._validate_pool(
        formal_root,
        pool_contract.TRAIN_POOL_RELATIVE,
        expected_split="train",
        expected_seed_start=pool_contract.DEFAULT_TRAINING_SEED_START,
        expected_episode_count=300,
    )
    training_seeds = list(
        range(args.training_seed_start, args.training_seed_start + args.updates)
    )
    pool_first = int(pool_contract.DEFAULT_TRAINING_SEED_START)
    pool_last = pool_first + 299
    if any(seed < pool_first or seed > pool_last for seed in training_seeds):
        parser.error(
            f"Training seeds must remain in the reviewed Training Pool range {pool_first}..{pool_last}."
        )
    for episode_seed in training_seeds:
        pool_contract._validate_episode_snapshot(train_pool, episode_seed, pool_info)

    source_checkpoint_arg = (
        args.initialize_from_stage3_checkpoint
        if args.initialize_from_stage3_checkpoint is not None
        else args.source_checkpoint
    )
    source_checkpoint = (
        None if source_checkpoint_arg is None else source_checkpoint_arg.resolve()
    )
    if source_checkpoint is not None and not source_checkpoint.is_file():
        raise FileNotFoundError(f"Source MAPPO checkpoint is missing: {source_checkpoint}")
    first_context = pool_contract._episode_context(
        formal_root, train_pool, training_seeds[0], pool_info
    )
    trainer = MAPPOBaselineTrainer(
        first_context,
        device=args.device,
        seed=args.trainer_seed,
    )
    trainer.enable_stage3_counterfactual_cache_value(stage3_config)
    source_sha256 = None if source_checkpoint is None else _sha256(source_checkpoint)
    if source_checkpoint is not None:
        resume = trainer.load_checkpoint(
            source_checkpoint,
            allow_v2_initialization=True,
            allow_stage3_policy_initialization=(
                args.initialize_from_stage3_checkpoint is not None
            ),
        )
        if resume.get("seed") != int(args.trainer_seed):
            raise ValueError(
                "Source Checkpoint trainer seed differs from --trainer-seed; "
                "resume must preserve the Checkpoint RNG contract exactly."
            )
    else:
        resume = {
            "checkpoint_version": MAPPO_STAGE3_CHECKPOINT_VERSION,
            "update_count": 0,
            "initialization": "fresh_random_initialization",
        }

    args.output_parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"stage3_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir = args.output_parent.resolve() / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    current_config_sha256 = str(trainer._checkpoint_contract()["config_sha256"])
    run_metadata = {
        "stage": "MAPPO_STAGE3_CACHE_VALUE",
        "run_id": run_id,
        "project_root": str(PROJECT_ROOT),
        "formal_root": str(formal_root),
        "training_pool_path": str(train_pool),
        "training_pool_manifest_sha256": pool_info["manifest_sha256"],
        "training_pool_source_config_sha256": pool_info.get("pool_config_sha256"),
        "training_pool_source_semantic_config_sha256": pool_info.get(
            "pool_semantic_config_sha256"
        ),
        "runtime_config_sha256": current_config_sha256,
        "training_pool_config_compatibility": pool_info.get("config_compatibility"),
        "baseline_schema": MAPPO_BASELINE_SCHEMA,
        "checkpoint_version": MAPPO_STAGE3_CHECKPOINT_VERSION,
        "algorithm_mode": COUNTERFACTUAL_VALUE_SCHEMA,
        "trigger_target_schema": TRIGGER_TARGET_SCHEMA,
        "source_checkpoint": None if source_checkpoint is None else str(source_checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_load": resume,
        "stage3_enabled": True,
        "stage3_config": stage3_config.contract(),
        "stage3_coefficient_profile": stage3_profile_metadata,
        "updates_requested": int(args.updates),
        "slots_per_update": int(args.slots_per_update),
        "epochs_per_update": int(args.epochs),
        "training_episode_seeds": training_seeds,
        "trainer_seed": int(args.trainer_seed),
        "device": args.device,
        "split": "train",
        "validation_accessed": False,
        "test_accessed": False,
        "smdp_contract": "fixed two-slot Macro decisions; final bootstrap; per-Slot Micro GAE",
    }
    with (run_dir / "run_metadata.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(
            _jsonable(run_metadata),
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")

    metrics_path = run_dir / "training_metrics.jsonl"
    last_checkpoint = source_checkpoint
    failure: dict[str, Any] | None = None
    with metrics_path.open("x", encoding="utf-8", newline="\n") as metrics_stream:
        if last_checkpoint is None:
            last_checkpoint = trainer.save_checkpoint(
                checkpoint_dir / "checkpoint_initial.pt",
                metadata={**run_metadata, "phase": "initial"},
            )
        for update_offset, episode_seed in enumerate(training_seeds):
            try:
                if update_offset > 0:
                    context = pool_contract._episode_context(
                        formal_root, train_pool, episode_seed, pool_info
                    )
                    trainer = MAPPOBaselineTrainer(
                        context,
                        device=args.device,
                        seed=args.trainer_seed,
                    )
                    trainer.enable_stage3_counterfactual_cache_value(stage3_config)
                    if last_checkpoint is None:
                        raise RuntimeError("No checkpoint is available to continue the run.")
                    trainer.load_checkpoint(
                        last_checkpoint,
                        allow_v2_initialization=False,
                    )
                elif trainer.context is not first_context:
                    raise RuntimeError("Initial MAPPO training context changed unexpectedly.")

                aligned_steps = pool_contract._aligned_window(
                    trainer.context,
                    int(trainer.context.train_origin_start),
                    args.slots_per_update,
                )
                if len(aligned_steps) != args.slots_per_update + 1:
                    raise ValueError(
                        "Training alignment must contain the requested Slots and one bootstrap step."
                    )
                slot_ids = [int(step.slot) for step in aligned_steps]
                if slot_ids != list(range(slot_ids[0], slot_ids[0] + len(slot_ids))):
                    raise ValueError("Training window Slot indices are not consecutive.")

                collection_started = time.perf_counter()
                rollout = trainer.collect_episode(
                    aligned_steps,
                    max_slots=args.slots_per_update,
                    deterministic=False,
                    episode_seed=episode_seed,
                    profile_slot_end_to_end=True,
                )
                collection_seconds = time.perf_counter() - collection_started
                if int(rollout.summary["transition_slots"]) != args.slots_per_update:
                    raise ValueError("Collected Training Slot count differs from the requested window.")
                event = {
                    "event": "training_update",
                    "update_offset": update_offset + 1,
                    "episode_seed": int(episode_seed),
                    "origin_start": int(trainer.context.train_origin_start),
                    "transition_slots": int(args.slots_per_update),
                    "collection_seconds": float(collection_seconds),
                    "rollout_environment": rollout.summary,
                    "stage3_enabled": True,
                }
                optimization_started = time.perf_counter()
                metrics = trainer.update(rollout, epochs=args.epochs)
                optimization_seconds = time.perf_counter() - optimization_started
                metrics["training_window"] = event
                metrics["optimization_seconds"] = float(optimization_seconds)
                metrics["complete_episode_return_reported"] = False
                checkpoint_path = trainer.save_checkpoint(
                    checkpoint_dir / f"checkpoint_update_{trainer.update_count:04d}.pt",
                    metadata={
                        **run_metadata,
                        "phase": "after_update",
                        "update_index": int(trainer.update_count),
                        "last_training_episode_seed": int(episode_seed),
                        "last_training_window_slots": int(args.slots_per_update),
                    },
                )
                metrics["checkpoint_after_update"] = str(checkpoint_path)
                last_checkpoint = checkpoint_path
                _write_event(metrics_stream, {"event": "update_complete", **metrics})
            except Exception as exc:
                failure = {
                    "event": "update_failed",
                    "update_offset": update_offset + 1,
                    "episode_seed": int(episode_seed),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "last_checkpoint": (
                        None if last_checkpoint is None else str(last_checkpoint)
                    ),
                    "stage3_loss_coefficients": {
                        "trigger_ranking": stage3_config.trigger_ranking_coefficient,
                        "candidate_ranking": stage3_config.candidate_ranking_coefficient,
                        "delta_q_huber": stage3_config.delta_q_huber_coefficient,
                    },
                }
                if isinstance(exc, CounterfactualLabelBatchError):
                    failure["stage3_counterfactual"] = exc.summary
                _write_event(metrics_stream, failure)
                break

    completion = {
        "run_dir": str(run_dir),
        "last_checkpoint": None if last_checkpoint is None else str(last_checkpoint),
        "completed_updates": (
            int(trainer.update_count) - int(resume.get("update_count", 0))
        ),
        "stage3_enabled": True,
        "failure": failure,
    }
    with (run_dir / "run_completion.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(
            _jsonable(completion),
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    print(json.dumps(_jsonable(completion), ensure_ascii=False, sort_keys=True))
    return 1 if failure is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
