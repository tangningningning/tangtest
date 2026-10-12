"""Shared formal PROPOSED runner for the Stage6 learning-rate comparison."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for candidate in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

CONFIG_RELATIVE = Path("configs/proposed/system_config.toml")
POOL_RELATIVE = Path("environment_data/stage6_candidate_resource_reduced_training")
POOL_MANIFEST_NAME = "stage6_training_pool_manifest.json"
EPISODE_COUNT = 300
EPISODE_SEED_START = 20001
EPISODE_SEED_END = EPISODE_SEED_START + EPISODE_COUNT - 1
PPO_INITIALIZATION_SEED = 27005

RUNS_BY_RATE = {
    1.0e-4: "stage6_lr_1e_minus_4",
    5.0e-4: "stage6_lr_5e_minus_4",
    1.0e-3: "stage6_lr_1e_minus_3",
    5.0e-3: "stage6_lr_5e_minus_3",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_close(field: str, actual: Any, expected: float) -> None:
    if not math.isclose(float(actual), expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(f"Stage6 config mismatch for {field}: {actual!r} != {expected!r}")


def _load_stage6_contract() -> tuple[dict[str, Any], str, str]:
    from common import project_config
    from generation import generate_episode_realization

    config_path = PROJECT_ROOT / CONFIG_RELATIVE
    cfg = project_config.load_config(str(config_path))
    project_config.validate_for_formal_experiment(cfg)

    expected_values = {
        "nodes.cloud.cpu_capacity_cycles_per_s": (cfg["nodes"]["cloud"]["cpu_capacity_cycles_per_s"], 40.0e9),
        "nodes.cloud.cloud_delay_factor": (cfg["nodes"]["cloud"]["cloud_delay_factor"], 15.0),
        "nodes.rsu.cpu_cycles_per_s": (cfg["nodes"]["rsu"]["cpu_cycles_per_s"], 10.0e9),
        "nodes.rsu.bandwidth_hz": (cfg["nodes"]["rsu"]["bandwidth_hz"], 20.0e6),
        "nodes.rsu.cache_capacity_mb": (cfg["nodes"]["rsu"]["cache_capacity_mb"], 25000.0),
        "nodes.uav.cpu_cycles_per_s": (cfg["nodes"]["uav"]["cpu_cycles_per_s"], 5.0e9),
        "nodes.uav.bandwidth_hz": (cfg["nodes"]["uav"]["bandwidth_hz"], 10.0e6),
        "nodes.uav.cache_capacity_mb": (cfg["nodes"]["uav"]["cache_capacity_mb"], 20000.0),
        "nodes.sv.cpu_cycles_per_s": (cfg["nodes"]["sv"]["cpu_cycles_per_s"], 1.0e9),
        "nodes.sv.bandwidth_hz": (cfg["nodes"]["sv"]["bandwidth_hz"], 20.0e6),
        "nodes.sv.cache_capacity_mb": (cfg["nodes"]["sv"]["cache_capacity_mb"], 12000.0),
        "communication.ground.bandwidth_hz": (cfg["communication"]["ground"]["bandwidth_hz"], 20.0e6),
        "service.generation.input_size_min_kb": (cfg["service"]["generation"]["input_size_min_kb"], 500.0),
        "service.generation.input_size_max_kb": (cfg["service"]["generation"]["input_size_max_kb"], 2000.0),
    }
    for field, (actual, expected) in expected_values.items():
        _assert_close(field, actual, expected)
    if cfg["nodes"]["cloud"].get("resource_mode") != "finite_pool":
        raise ValueError("Stage6 Cloud contract must use finite_pool.")
    if cfg["algorithm"]["matching"].get("cloud_fallback_mode") != "RESOURCE_AWARE_FALLBACK":
        raise ValueError("Stage6 Matching contract must use RESOURCE_AWARE_FALLBACK.")

    ppo = cfg["algorithm"]["ppo"]
    if int(ppo.get("realizations_per_policy_update", 0)) != 4:
        raise ValueError("Stage6 PPO contract must update once per four Episodes.")
    if ppo.get("macro_actor_learning_rate_schedule") != "constant":
        raise ValueError("Macro Actor learning-rate schedule must be constant.")
    if ppo.get("optimizer_learning_rate_schedule") != "constant":
        raise ValueError("All other optimizer learning-rate schedules must be constant.")
    if EPISODE_COUNT % int(ppo["realizations_per_policy_update"]):
        raise ValueError("Training Episode count must align with the PPO update group size.")

    raw_hash = project_config.raw_config_sha256(config_path)
    generation_fingerprint = generate_episode_realization.generation_config_fingerprint(cfg)
    semantic_hash = project_config.semantic_config_hash(cfg)
    return cfg, raw_hash, generation_fingerprint + ":" + semantic_hash


def _validate_training_pool(raw_config_hash: str, combined_fingerprint: str) -> Path:
    pool_path = PROJECT_ROOT / POOL_RELATIVE
    pool_manifest_path = pool_path / POOL_MANIFEST_NAME
    if not pool_manifest_path.is_file():
        raise FileNotFoundError(
            f"Stage6 training pool manifest is missing: {pool_manifest_path}. "
            "Prepare it from the existing proposed_joint_training pool using the Stage6 workload overlay."
        )
    manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "stage6_training_pool_manifest_v1",
        "split": "train",
        "seed_start": EPISODE_SEED_START,
        "seed_end": EPISODE_SEED_END,
        "episode_count": EPISODE_COUNT,
        "config_sha256": raw_config_hash,
        "generation_and_semantic_fingerprint": combined_fingerprint,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"Stage6 training pool {field} mismatch: {manifest.get(field)!r} != {value!r}"
            )

    split_path = PROJECT_ROOT / "environment_data/data_split_manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_split = split.get("train", {})
    validation_split = split.get("validation", {})
    test_split = split.get("test", {})
    if not (
        int(train_split.get("seed_start", -1)) <= EPISODE_SEED_START
        and int(train_split.get("seed_end", -1)) >= EPISODE_SEED_END
        and EPISODE_SEED_END < int(validation_split.get("seed_start", -1))
        and EPISODE_SEED_END < int(test_split.get("seed_start", -1))
    ):
        raise ValueError("Stage6 episode seeds are not isolated inside the declared Training split.")

    manifest_hashes = manifest.get("episode_manifest_sha256")
    expected_seeds = [str(seed) for seed in range(EPISODE_SEED_START, EPISODE_SEED_END + 1)]
    if not isinstance(manifest_hashes, dict) or sorted(manifest_hashes) != expected_seeds:
        raise ValueError("Stage6 pool manifest does not enumerate all 300 Training Episodes.")
    for seed in range(EPISODE_SEED_START, EPISODE_SEED_END + 1):
        seed_text = str(seed)
        episode_dir = pool_path / f"episode_{seed}"
        episode_manifest_path = episode_dir / "episode_manifest.json"
        if not episode_manifest_path.is_file():
            raise FileNotFoundError(f"Stage6 Training Episode manifest missing: {episode_manifest_path}")
        if _sha256(episode_manifest_path) != manifest_hashes[seed_text]:
            raise ValueError(f"Stage6 Training Episode manifest hash mismatch for seed {seed}.")
        episode_manifest = json.loads(episode_manifest_path.read_text(encoding="utf-8"))
        if int(episode_manifest.get("episode_seed", -1)) != seed:
            raise ValueError(f"Episode manifest seed mismatch for {seed}.")
        fingerprint = episode_manifest.get("generation_config_fingerprint")
        if fingerprint != combined_fingerprint.split(":", maxsplit=1)[0]:
            raise ValueError(f"Stage6 generation fingerprint mismatch for seed {seed}.")
        required_artifacts = (
            "trajectory.csv",
            "vehicle_roles.json",
            "tasks.csv",
            "popularity.csv",
            "popularity_history.csv",
            "vehicle_predictions.csv",
            "popularity_predictions.csv",
            "prediction_scenarios.npz",
            "tv_local_cache.json",
        )
        missing = [name for name in required_artifacts if not (episode_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Stage6 Training Episode {seed} is incomplete; missing {missing}."
            )
    return pool_path


def main_for_macro_actor_learning_rate(rate: float) -> int:
    requested_rate = float(rate)
    matched_rate = next(
        (value for value in RUNS_BY_RATE if math.isclose(requested_rate, value, rel_tol=0.0, abs_tol=1.0e-15)),
        None,
    )
    if matched_rate is None:
        raise ValueError(f"Unsupported Stage6 Macro Actor learning rate: {rate!r}")

    cfg, raw_config_hash, combined_fingerprint = _load_stage6_contract()
    pool_path = _validate_training_pool(raw_config_hash, combined_fingerprint)
    run_name = RUNS_BY_RATE[matched_rate]
    run_path = PROJECT_ROOT / "runs" / run_name
    if run_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Stage6 run: {run_path}")

    import os

    os.chdir(PROJECT_ROOT)
    from ppo_simple.trainer import main as trainer_main

    print(f"Stage6 config: {CONFIG_RELATIVE.as_posix()}", flush=True)
    print(f"Stage6 training pool: {POOL_RELATIVE.as_posix()} (20001–20300)", flush=True)
    print(f"Macro Actor learning rate: {matched_rate:.1e} (constant override)", flush=True)
    print(f"PPO initialization seed: {PPO_INITIALIZATION_SEED}", flush=True)
    print(f"Output run: runs/{run_name}", flush=True)
    return trainer_main(
        [
            "--config",
            CONFIG_RELATIVE.as_posix(),
            "--epochs",
            str(EPISODE_COUNT),
            "--run-dir",
            f"runs/{run_name}",
            "--episode-pool",
            POOL_RELATIVE.as_posix(),
            "--episode-seed-start",
            str(EPISODE_SEED_START),
            "--macro-actor-learning-rate",
            f"{matched_rate:.1e}",
            "--seed",
            str(PPO_INITIALIZATION_SEED),
        ]
    )
