"""在正式 Test 池运行冻结 best 策略和共享执行器基线。"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from common import project_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from online_joint_dqn.agent import DoubleDQNAgent  # noqa: E402
from online_joint_dqn.environment_adapter import ExclusiveDQNEnvironment  # noqa: E402
from online_joint_dqn.evaluation import (  # noqa: E402
    EVALUATION_SCHEMA_VERSION,
    evaluate_formal_split,
    evaluation_pool_identity,
    policy_state_sha256,
    validate_formal_evaluation_pool,
)
from online_joint_dqn.run_training import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
    _code_identity,
    _identity_sha256,
    _jsonable,
    _sha256,
)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    _jsonable(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_best_policy(
    project_root: Path,
    run_dir: Path,
    run_manifest: Mapping[str, Any],
    test_pool: tuple[Path, Mapping[str, Any]],
) -> tuple[DoubleDQNAgent, dict[str, Any], str, ExclusiveDQNEnvironment]:
    checkpoint_path = run_dir / "checkpoints" / "best_checkpoint.pt"
    if run_manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Run Manifest schema is incompatible with frozen Test evaluation")
    if (
        run_manifest.get("training_status") != "FORMAL_TRAINING_COMPLETE"
        or int(run_manifest.get("completed_episodes", -1)) != 300
        or int(run_manifest.get("completed_slots", -1)) != 300 * 314
    ):
        raise ValueError("Frozen Test evaluation requires the completed 300-Episode training run")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Frozen best Checkpoint is missing: {checkpoint_path}")
    checkpoint_sha = _sha256(checkpoint_path)
    manifest_best = run_manifest.get("best_checkpoint")
    if not isinstance(manifest_best, Mapping):
        raise ValueError("Run Manifest has no Validation-selected best Checkpoint")
    if manifest_best.get("file_sha256") != checkpoint_sha:
        raise ValueError("best_checkpoint.pt SHA-256 differs from the Run Manifest")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("checkpoint_role") != "best"
    ):
        raise ValueError("The selected checkpoint is not a compatible best snapshot")
    identity_payload = run_manifest.get("identity_payload")
    checkpoint_identity = checkpoint.get("identity_payload")
    if not isinstance(identity_payload, Mapping) or not isinstance(checkpoint_identity, Mapping):
        raise ValueError("Run Manifest or best Checkpoint lacks its identity payload")
    if dict(identity_payload) != dict(checkpoint_identity):
        raise ValueError("Best Checkpoint and Run Manifest identities differ")
    identity_sha = _identity_sha256(identity_payload)
    if (
        run_manifest.get("identity_sha256") != identity_sha
        or checkpoint.get("identity_sha256") != identity_sha
    ):
        raise ValueError("Run identity SHA-256 does not match the stored payload")
    current_code_identity = _code_identity()
    stored_code_identity = identity_payload.get("code_identity")
    if not isinstance(stored_code_identity, Mapping) or any(
        stored_code_identity.get(field) != current_code_identity.get(field)
        for field in ("head", "effective_source_sha256")
    ):
        raise ValueError("Current Online Joint DQN code differs from the trained identity")
    config_path = project_root / "configs" / "dqn" / "online_joint_dqn.toml"
    runtime_config = project_config.parse_toml_file(config_path)
    if (
        project_config.raw_config_sha256(config_path)
        != identity_payload.get("config_raw_sha256")
        or project_config.semantic_config_hash(runtime_config)
        != identity_payload.get("config_semantic_sha256")
    ):
        raise ValueError("Current formal configuration differs from the trained identity")

    training_metadata = checkpoint.get("training_metadata")
    if not isinstance(training_metadata, Mapping):
        raise ValueError("Best Checkpoint lacks selection metadata")
    best_metadata = training_metadata.get("best_checkpoint")
    if not isinstance(best_metadata, Mapping):
        raise ValueError("Best Checkpoint lacks its Validation metric")
    for key in (
        "episode_count",
        "selection_metric",
        "selection_metric_value",
        "policy_state_sha256",
        "checkpoint_identity",
    ):
        if manifest_best.get(key) != best_metadata.get(key):
            raise ValueError(f"Best Checkpoint selection identity mismatch: {key}")
    checkpoint_episode_count = int(checkpoint.get("next_episode_index", -1))
    if int(best_metadata.get("episode_count", -2)) != checkpoint_episode_count:
        raise ValueError("Best Checkpoint progress and Validation episode count differ")
    expected_progress = checkpoint_episode_count * 314
    if int(checkpoint.get("global_step", -1)) != expected_progress:
        raise ValueError("Best Checkpoint does not end on a complete formal Episode")

    test_pool_path, test_pool_info = test_pool
    if (
        identity_payload.get("service_catalog_sha256")
        != test_pool_info.get("service_catalog_sha256")
        or identity_payload.get("training_task_contract_sha256")
        != test_pool_info.get("task_data_contract_sha256")
    ):
        raise ValueError("Test pool service/task identity differs from the trained run")
    test_identity = evaluation_pool_identity(test_pool_info)
    test_seeds = list(test_identity["active_episode_seeds"])
    with contextlib.redirect_stdout(io.StringIO()):
        first_environment = ExclusiveDQNEnvironment.from_formal_split_episode(
            project_root,
            split="test",
            episode_seed=int(test_seeds[0]),
            origin_start=0,
            max_steps=int(test_pool_info["execution_horizon_slots"]),
            validated_pool=(test_pool_path, test_pool_info),
        )
    state, candidates = first_environment.observe()
    agent_config = identity_payload.get("agent_config")
    if not isinstance(agent_config, Mapping):
        raise ValueError("Run identity lacks the DQN architecture and learning contract")
    if (
        int(agent_config["state_dim"]) != len(state)
        or int(agent_config["candidate_dim"]) != len(candidates[0].features)
    ):
        raise ValueError("Best Checkpoint state/candidate dimensions differ from Test input")
    agent = DoubleDQNAgent(
        len(state),
        len(candidates[0].features),
        learning_rate=float(agent_config["learning_rate"]),
        gamma=float(agent_config["gamma"]),
        target_network_frequency=int(
            agent_config["target_network_frequency_gradient_updates"]
        ),
        target_tau=float(agent_config["target_tau"]),
        max_grad_norm=float(agent_config["max_grad_norm"]),
        hidden_sizes=tuple(int(value) for value in agent_config["hidden_sizes"]),
        seed=int(agent_config["seed"]),
        device=str(agent_config["device"]),
    )
    agent.load_state_dict(checkpoint["agent_state"])
    actual_policy_sha = policy_state_sha256(agent)
    if actual_policy_sha != best_metadata.get("policy_state_sha256"):
        raise ValueError("Loaded best policy weights differ from Validation selection identity")
    agent.online_network.eval()
    agent.online_category_network.eval()
    agent.target_network.eval()
    agent.target_category_network.eval()
    return agent, dict(best_metadata), checkpoint_sha, first_environment


def evaluate_test(
    *,
    run_dir: Path,
    output_dir: Path,
    project_root: Path = PROJECT_ROOT,
) -> int:
    root = project_root.resolve()
    run_dir = (run_dir if run_dir.is_absolute() else root / run_dir).resolve(strict=True)
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    output_dir = output_dir.resolve()
    run_manifest_path = run_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"Run Manifest is missing: {run_manifest_path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    pool_path, pool_info = validate_formal_evaluation_pool(root, "test")
    agent, best_metadata, checkpoint_sha, _ = _load_best_policy(
        root, run_dir, run_manifest, (pool_path, pool_info)
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "RUNNING",
        "split": "test",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(run_dir / "checkpoints" / "best_checkpoint.pt"),
            "file_sha256": checkpoint_sha,
            "run_identity_sha256": run_manifest["identity_sha256"],
            "episode_count": best_metadata["episode_count"],
            "selection_metric": best_metadata["selection_metric"],
            "selection_metric_value": best_metadata["selection_metric_value"],
            "policy_state_sha256": best_metadata["policy_state_sha256"],
        },
        "config_identity": {
            "config_raw_sha256": run_manifest["identity_payload"]["config_raw_sha256"],
            "config_semantic_sha256": run_manifest["identity_payload"]["config_semantic_sha256"],
        },
        "test_pool_identity": evaluation_pool_identity(pool_info),
        "policies": ["dqn_greedy", "keep_stay", "uniform_random_legal"],
        "scene_count": len(pool_info["active_episode_seeds"]),
        "completed_scene_policy_runs": 0,
        "failure": None,
    }
    _write_json_atomic(output_dir / "evaluation_manifest.json", manifest)
    try:
        result = evaluate_formal_split(
            root,
            split="test",
            policies=tuple(manifest["policies"]),
            agent=agent,
            validated_pool=(pool_path, pool_info),
        )
        if result.get("status") != "COMPLETE":
            raise RuntimeError("Test evaluation did not report COMPLETE")
        expected_count = int(manifest["scene_count"])
        for policy in manifest["policies"]:
            if result["policy_summaries"][policy]["completed_episode_count"] != expected_count:
                raise RuntimeError(f"Policy {policy} did not complete every Test scene")
        episode_rows: list[dict[str, Any]] = []
        slot_rows: list[dict[str, Any]] = []
        for episode in result["episode_results"]:
            summary = {key: value for key, value in episode.items() if key != "per_slot"}
            episode_rows.append(summary)
            for slot in episode["per_slot"]:
                slot_rows.append(
                    {
                        "split": "test",
                        "episode_seed": episode["episode_seed"],
                        "scene_sha256": episode["scene_sha256"],
                        "policy": episode["policy"],
                        **slot,
                    }
                )
        _write_jsonl_atomic(output_dir / "episode_summaries.jsonl", episode_rows)
        _write_jsonl_atomic(output_dir / "slot_metrics.jsonl", slot_rows)
        manifest.update(
            {
                "status": "COMPLETE",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_scene_policy_runs": len(episode_rows),
                "policy_summaries": result["policy_summaries"],
                "episode_summaries_sha256": _sha256(output_dir / "episode_summaries.jsonl"),
                "slot_metrics_sha256": _sha256(output_dir / "slot_metrics.jsonl"),
                "output_contract": {
                    "episode_summaries": "one row per policy and Test scene",
                    "slot_metrics": "one row per executed slot, retaining unique task and task-slot metrics",
                    "test_used_for_model_selection": False,
                    "optimizer_updates": 0,
                    "training_replay_writes": 0,
                },
            }
        )
        _write_json_atomic(output_dir / "evaluation_manifest.json", manifest)
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "scene_count": expected_count,
                    "policy_summaries": manifest["policy_summaries"],
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    except BaseException as exc:
        manifest.update(
            {
                "status": "FAILED",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _write_json_atomic(output_dir / "evaluation_manifest.json", manifest)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return evaluate_test(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
