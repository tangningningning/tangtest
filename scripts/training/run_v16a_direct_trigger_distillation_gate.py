from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "configs" / "proposed" / "system_config.toml"
EPISODE_POOL = PROJECT_ROOT / "environment_data" / "proposed_joint_training"
RUN_DIR = PROJECT_ROOT / "runs" / "formal_v16a_direct_trigger_distillation_gate"
REPORT_PATH = PROJECT_ROOT / "reports" / "v16a_direct_trigger_distillation_gate.json"
V14B_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "formal_v14b_actor_lr_1e_minus_3"
    / "checkpoints"
    / "latest_checkpoint.pt"
)
V14B_SHA256 = "82a835d7e5be6325566f14ffa65afeb388d7e3ba47e4ddce487a14e6b7d1e55c"
C2_RUNNER_PATH = PROJECT_ROOT / "scripts" / "training" / "run_v15c2_trigger_normalization_gate.py"
BASE_SEED = 27005
EPISODE_SEED_START = 20001
TARGET_EPISODES = 20
REALIZATIONS_PER_UPDATE = 4
EXPECTED_NORMALIZATION_SOURCE_SHA256 = (
    "1be40fb05ffce97bf3fe58085c1229661feb8b5c92f4556010b4ab5396324af2"
)
FORMAL_COMMAND = (
    "conda activate adaptive_cache_mec; "
    "python scripts/training/run_v16a_direct_trigger_distillation_gate.py"
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _validate_preflight(c2: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    required = [CONFIG_PATH, V14B_CHECKPOINT, c2.V15C_NORMALIZATION_CHECKPOINT, c2.NORMALIZATION_SOURCE]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V16-A input artifacts missing: {missing}")
    if hashlib.sha256(V14B_CHECKPOINT.read_bytes()).hexdigest() != V14B_SHA256:
        raise ValueError("V16-A V14b episode-300 warm-start checkpoint SHA256 mismatch.")
    if RUN_DIR.exists() and (not RUN_DIR.is_dir() or any(RUN_DIR.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty V16-A run directory: {RUN_DIR}")
    seeds = list(range(EPISODE_SEED_START, EPISODE_SEED_START + TARGET_EPISODES))
    missing_realizations = [
        str(EPISODE_POOL / f"episode_{seed}")
        for seed in seeds
        if not (EPISODE_POOL / f"episode_{seed}").is_dir()
    ]
    if missing_realizations:
        raise FileNotFoundError(f"V16-A training realization(s) missing: {missing_realizations}")
    normalization, source = c2._load_training_only_normalizer()
    if source.get("source_sha256") != EXPECTED_NORMALIZATION_SOURCE_SHA256:
        raise ValueError("V16-A must inherit the verified V15-C2/C4 normalization source.")
    raw_mean = np.asarray(normalization["raw_mean"], dtype=np.float64)
    raw_std = np.asarray(normalization["raw_std"], dtype=np.float64)
    scale = np.asarray(normalization["scale"], dtype=np.float64)
    if any(array.shape != (400,) or not np.isfinite(array).all() for array in (raw_mean, raw_std, scale)):
        raise ValueError("V16-A normalization arrays must be finite 400D vectors.")
    if not np.array_equal(scale, np.where(raw_std <= 1.0e-6, 1.0, raw_std)):
        raise AssertionError("V16-A changed the V15-C2/C4 raw-std normalization rule.")
    if int(source.get("low_variance_feature_count", -1)) != 73:
        raise AssertionError("V16-A normalization low-variance feature count must remain 73.")
    return normalization, source


def main() -> int:
    c2 = _load_module("v15c2_normalization_gate_v16a", C2_RUNNER_PATH)
    normalization, normalization_source = _validate_preflight(c2)
    preflight = {
        "pass": True,
        "c4_replay_observation_fields_present": False,
        "c4_artifact_decision": "fresh_training_collection_required",
        "training_episode_seeds": list(range(EPISODE_SEED_START, EPISODE_SEED_START + TARGET_EPISODES)),
        "teacher_recomputed_from_new_training_rollouts": True,
        "normalization_source_sha256": normalization_source["source_sha256"],
        "normalization_low_variance_feature_count": normalization_source["low_variance_feature_count"],
        "architecture_version": 23,
        "checkpoint_version": 47,
    }
    from ppo_simple.direct_trigger_distillation import run_v16a_direct_trigger_distillation

    result = run_v16a_direct_trigger_distillation(
        config_path=CONFIG_PATH,
        episode_pool=EPISODE_POOL,
        run_dir=RUN_DIR,
        report_path=REPORT_PATH,
        episode_seed_start=EPISODE_SEED_START,
        base_seed=BASE_SEED,
        initial_checkpoint_path=V14B_CHECKPOINT,
        initial_checkpoint_sha256=V14B_SHA256,
        normalization=normalization,
        normalization_source=normalization_source,
        formal_runner_path=Path(__file__),
        formal_command=FORMAL_COMMAND,
    )
    result["PREFLIGHT"] = preflight
    _write_json(REPORT_PATH, result)
    print(json.dumps({key: result[key] for key in (
        "IMPLEMENTATION_COMPLETE",
        "TRIGGER_PPO_ACTIVE",
        "DIRECT_TRIGGER_DISTILLATION_ACTIVE",
        "EPISODES_COMPLETED",
        "SYNC_UPDATES_COMPLETED",
        "FRESH_PREUPDATE_SPEARMAN_UPDATES_3_5_MEAN",
        "DIRECT_DISTILLATION_GENERALIZATION_SUPPORTED",
        "READY_FOR_STAGE2",
        "PRIMARY_FAILURE_MODE",
    )}, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
