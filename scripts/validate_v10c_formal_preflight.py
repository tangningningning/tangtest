from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys
import tomllib

SCHEMA = "adaptive_joint_cache_replica_aware_uav_operation_v10c_decoupled_encoder"
EXPECTED_EPISODES = tuple(range(20001, 20301))
REQUIRED_ARTIFACTS = {
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
}
REQUIRED_KL_FIELDS = (
    '"cache_approx_kl"',
    '"uav_approx_kl"',
    '"joint_approx_kl"',
    '"cache_only_kl_trigger_count"',
    '"uav_only_kl_trigger_count"',
    '"both_kl_trigger_count"',
    '"neither_kl_trigger_count"',
    '"cache_trigger_suppressed_uav_update"',
    '"uav_trigger_suppressed_cache_update"',
)


def fail(message: str) -> None:
    raise SystemExit(message)


def load_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def leaves(value: dict, prefix: tuple[str, ...] = ()):
    for key, item in value.items():
        current = prefix + (key,)
        if isinstance(item, dict):
            yield from leaves(item, current)
        else:
            yield current, item


def require_close(values: dict, field: str, expected: float) -> None:
    actual = float(values[field])
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
        fail(f"{field}={actual} != expected {expected}")


def validate(args: argparse.Namespace) -> None:
    if sys.version_info < (3, 11):
        fail("Python 3.11+ is required for standard-library TOML validation")
    if importlib.util.find_spec("torch") is None:
        fail("PyTorch is not importable in the selected Conda environment")
    if importlib.util.find_spec("ppo_simple") is None:
        fail("ppo_simple is not importable from this repository")

    import torch
    from ppo_simple.policy import (
        AdaptiveJointCacheReplicaAwareUAVOperationV10CDecoupledEncoderActorCritic,
    )

    config_path = Path(args.config)
    reference_path = Path(args.reference_config)
    pool_path = Path(args.episode_pool)
    run_path = Path(args.run_dir)
    config = load_toml(config_path)
    reference = load_toml(reference_path)
    ppo = config["algorithm"]["ppo"]
    reference_ppo = reference["algorithm"]["ppo"]

    if reference_ppo["policy_schema"] != "adaptive_joint_cache_replica_aware_uav_operation_v10_branchwise":
        fail("reference config is not the frozen V10-B branchwise config")
    if not math.isclose(
        float(reference_ppo["actor_learning_rate"]),
        0.0005,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        fail("reference config actor LR is not the expected V10-B 5e-4 baseline")

    if ppo["policy_schema"] != SCHEMA:
        fail(f"unexpected policy schema: {ppo['policy_schema']}")
    if AdaptiveJointCacheReplicaAwareUAVOperationV10CDecoupledEncoderActorCritic.architecture_id != SCHEMA:
        fail("V10-C policy implementation/schema identity mismatch")
    if config["experiment"]["name"] != args.expected_tag:
        fail("experiment.name does not identify this formal LR run")
    if config["experiment"]["output_tag"] != args.expected_tag:
        fail("experiment.output_tag does not identify this formal LR run")

    allowed_differences = {
        ("experiment", "name"),
        ("experiment", "output_tag"),
        ("algorithm", "ppo", "policy_schema"),
        ("algorithm", "ppo", "actor_learning_rate"),
    }
    actual = dict(leaves(config))
    baseline = dict(leaves(reference))
    if actual.keys() != baseline.keys():
        fail("V10-C and V10-B config fields differ; refusing unreviewed config drift")
    unexpected = sorted(
        ".".join(key)
        for key in actual
        if actual[key] != baseline[key] and key not in allowed_differences
    )
    if unexpected:
        fail(f"unexpected config differences from V10-B: {unexpected}")

    for field, expected in (
        ("actor_learning_rate", args.expected_actor_lr),
        ("critic_learning_rate", 0.0003),
        ("target_kl", 0.10),
        ("clip_epsilon", 0.2),
        ("gamma", 1.0),
        ("gae_lambda", 0.95),
        ("entropy_coefficient", 0.0),
    ):
        require_close(ppo, field, expected)
    for field, expected in (
        ("update_epochs", 4),
        ("num_minibatches", 10),
        ("minibatch_size", 30),
    ):
        if int(ppo[field]) != expected:
            fail(f"{field}={ppo[field]} != expected {expected}")
    if int(config["experiment"]["seed"]) != 4401:
        fail("experiment.seed must remain 4401")
    if ppo["critic_value_scale_mode"] != "popart_normalized":
        fail("PopArt critic scaling mode changed")
    if ppo["configuration_control_mode"] != "policy_primitive_v9_joint_cache_uav_operation":
        fail("configuration_control_mode changed")
    protocol = ppo["data_protocol"]
    if protocol["protocol"] != "full_episode_randomized":
        fail("unexpected PPO data protocol")
    for field, expected in (
        ("episode_slots", 300),
        ("train_origin_start", 0),
        ("train_origin_end", 299),
        ("rollout_steps", 300),
        ("terminal_origin", 299),
    ):
        if int(protocol[field]) != expected:
            fail(f"data_protocol.{field}={protocol[field]} != expected {expected}")

    if not pool_path.is_dir():
        fail(f"episode pool does not exist: {pool_path}")
    episode_dirs = []
    for path in pool_path.iterdir():
        if not path.name.startswith("episode_"):
            continue
        if not path.is_dir():
            fail(f"episode entry is not a directory: {path}")
        try:
            episode_id = int(path.name.removeprefix("episode_"))
        except ValueError:
            fail(f"invalid episode directory name: {path.name}")
        episode_dirs.append((episode_id, path))
    episode_dirs.sort(key=lambda item: item[0])
    actual_ids = tuple(episode_id for episode_id, _ in episode_dirs)
    if actual_ids != EXPECTED_EPISODES:
        fail(
            "episode pool must contain exactly episode_20001..episode_20300; "
            f"found {len(actual_ids)} episode directories"
        )
    for episode_id, path in episode_dirs:
        missing = sorted(name for name in REQUIRED_ARTIFACTS if not (path / name).is_file())
        if missing:
            fail(f"episode_{episode_id} is incomplete: {missing}")

    trainer_source = (
        Path.cwd() / "src" / "ppo_simple" / "trainer.py"
    ).read_text(encoding="utf-8")
    missing_fields = [field for field in REQUIRED_KL_FIELDS if field not in trainer_source]
    if missing_fields:
        fail(f"required V10-C KL instrumentation is missing: {missing_fields}")
    if "and float(np.mean(update_epoch_approximate_kls)) > hyper.target_kl" not in trainer_source:
        fail("the existing joint target-KL gate was not found")
    if "if joint_gate_triggered:\n                break" not in trainer_source:
        fail("the existing shared joint KL early-stop behavior was not found")

    if run_path.exists():
        if not run_path.is_dir():
            fail(f"run path exists but is not a directory: {run_path}")
        if any(run_path.iterdir()):
            fail(f"run directory is not empty; refusing overwrite or resume: {run_path}")

    print(f"preflight: Python={sys.executable}")
    print(f"preflight: PyTorch={torch.__version__}")
    print(f"preflight: config={config_path}")
    print(f"preflight: policy_schema={SCHEMA}")
    print(
        f"preflight: episode_pool={pool_path} count={len(episode_dirs)} "
        f"first={actual_ids[0]} last={actual_ids[-1]}"
    )
    print(f"preflight: run_dir={'absent' if not run_path.exists() else 'empty'}")
    print("preflight: V10-C KL instrumentation and unchanged joint gate=present")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only V10-C formal-run preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-config", required=True)
    parser.add_argument("--episode-pool", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-actor-lr", required=True, type=float)
    parser.add_argument("--expected-tag", required=True)
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
