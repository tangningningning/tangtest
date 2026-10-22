"""运行带身份记录、逐 Slot 日志和可恢复 Checkpoint 的独立 DQN 训练。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from common import project_config  # noqa: E402
from online_joint_dqn.agent import DoubleDQNAgent  # noqa: E402
from online_joint_dqn.candidate_features import (  # noqa: E402
    CANDIDATE_FEATURE_CONTRACT_VERSION,
    NETWORK_ARCHITECTURE_VERSION,
    STATE_FEATURE_CONTRACT_VERSION,
)
from online_joint_dqn.environment_adapter import (  # noqa: E402
    ExclusiveDQNEnvironment,
)
from online_joint_dqn.replay_buffer import ReplayBuffer  # noqa: E402
from online_joint_dqn.trainer import TrainerConfig, train_episode  # noqa: E402
import training_pool_contract as pool_contract  # noqa: E402


RUN_MANIFEST_SCHEMA_VERSION = "online_joint_dqn_run_v3"
CHECKPOINT_SCHEMA_VERSION = 4
DEFAULT_SEED = 20261022
DEFAULT_REPLAY_CAPACITY = 10000

_CODE_IDENTITY_REQUIRED_PATHS = (
    "src/common/project_config.py",
    "src/common/system_model.py",
    "src/decision/cache_configuration.py",
    "src/decision/uav_configuration.py",
    "src/decision/slot_environment.py",
    "src/decision/task_allocation.py",
    "src/ppo_simple/interface.py",
    "src/scenario/realization.py",
    "src/scenario/vehicle_order.py",
    "scripts/training/training_pool_contract.py",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _write_json(path: Path, payload: Any, *, atomic: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if not atomic:
        path.write_text(text, encoding="utf-8", newline="\n")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _append_jsonl(stream: Any, payload: Any) -> None:
    stream.write(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    stream.flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _code_identity() -> dict[str, Any]:
    # 哈希在线决策、执行、数据对齐和效用合同的源码，不扫描无关仓库文件。
    paths = sorted(
        {
            *PROJECT_ROOT.glob("src/online_joint_dqn/**/*.py"),
            *PROJECT_ROOT.glob("src/online_joint_optimizer/**/*.py"),
            *(PROJECT_ROOT / relative_path for relative_path in _CODE_IDENTITY_REQUIRED_PATHS),
        }
    )
    missing = [
        relative_path
        for relative_path in _CODE_IDENTITY_REQUIRED_PATHS
        if not (PROJECT_ROOT / relative_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "关键训练源码缺失，不能建立源码身份：" + ", ".join(missing)
        )
    hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path)
        for path in paths
        if path.is_file()
    }
    if not any(path.startswith("src/online_joint_dqn/") for path in hashes):
        raise FileNotFoundError("online_joint_dqn 源码目录中没有可哈希的 Python 文件")
    if not any(path.startswith("src/online_joint_optimizer/") for path in hashes):
        raise FileNotFoundError("online_joint_optimizer 源码目录中没有可哈希的 Python 文件")
    status = _git_output("status", "--porcelain")
    return {
        "head": _git_output("rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "worktree_status_porcelain": status.splitlines() if status else [],
        "effective_source_sha256": hashes,
    }


def _identity_payload(identity: Mapping[str, Any]) -> dict[str, Any]:
    """返回参与训练身份摘要的字段，不包含摘要本身。"""

    return {
        str(key): _jsonable(value)
        for key, value in identity.items()
        if str(key) != "identity_sha256"
    }


def _identity_sha256(identity_payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(identity_payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity_differences(
    expected: Any, actual: Any, *, prefix: str = "identity"
) -> list[str]:
    """按字段路径列出身份不匹配，便于恢复失败时定位具体项目。"""

    differences: list[str] = []
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) | set(actual), key=str):
            field = f"{prefix}.{key}"
            if key not in expected:
                differences.append(f"{field}: current identity missing")
            elif key not in actual:
                differences.append(f"{field}: stored identity missing")
            else:
                differences.extend(
                    _identity_differences(expected[key], actual[key], prefix=field)
                )
        return differences
    if isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
        if len(expected) != len(actual):
            differences.append(
                f"{prefix}: length mismatch (current={len(expected)}, stored={len(actual)})"
            )
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=False)
        ):
            differences.extend(
                _identity_differences(
                    expected_item, actual_item, prefix=f"{prefix}[{index}]"
                )
            )
        return differences
    if expected != actual:
        differences.append(f"{prefix}: current={expected!r}, stored={actual!r}")
    return differences


def _validate_resume_identity(
    current_identity: Mapping[str, Any],
    manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    """严格验证当前源码、Manifest 与 Checkpoint 的身份字段及摘要。"""

    failures: list[str] = []
    current_payload = _identity_payload(current_identity)
    current_digest = str(current_identity.get("identity_sha256", ""))

    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        failures.append(
            "Manifest schema 不包含完整源码身份合同 "
            f"(stored={manifest.get('schema_version')!r}, "
            f"required={RUN_MANIFEST_SCHEMA_VERSION!r})"
        )
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        failures.append(
            "Checkpoint schema 不包含完整源码身份合同 "
            f"(stored={checkpoint.get('schema_version')!r}, "
            f"required={CHECKPOINT_SCHEMA_VERSION!r})"
        )

    manifest_payload = manifest.get("identity_payload")
    if not isinstance(manifest_payload, Mapping):
        failures.append("Manifest 缺少 identity_payload，拒绝恢复")
        manifest_payload = {}
    checkpoint_payload = checkpoint.get("identity_payload")
    if not isinstance(checkpoint_payload, Mapping):
        failures.append("Checkpoint 缺少 identity_payload，拒绝恢复")
        checkpoint_payload = {}

    for label, stored_payload in (
        ("Manifest", manifest_payload),
        ("Checkpoint", checkpoint_payload),
    ):
        missing_fields = sorted(set(current_payload) - set(stored_payload))
        extra_fields = sorted(set(stored_payload) - set(current_payload))
        if missing_fields:
            failures.append(
                f"{label} 缺少身份字段：" + ", ".join(missing_fields)
            )
        if extra_fields:
            failures.append(
                f"{label} 包含当前版本不识别的身份字段：" + ", ".join(extra_fields)
            )

    manifest_digest = _identity_sha256(manifest_payload) if manifest_payload else ""
    checkpoint_digest = (
        _identity_sha256(checkpoint_payload) if checkpoint_payload else ""
    )
    if manifest_digest and manifest.get("identity_sha256") != manifest_digest:
        failures.append("Manifest identity_sha256 与其 identity_payload 不一致")
    if checkpoint_digest and checkpoint.get("identity_sha256") != checkpoint_digest:
        failures.append("Checkpoint identity_sha256 与其 identity_payload 不一致")

    if manifest.get("identity_sha256") != current_digest:
        failures.append("当前训练身份摘要与 Manifest identity_sha256 不一致")
    if checkpoint.get("identity_sha256") != current_digest:
        failures.append("当前训练身份摘要与 Checkpoint identity_sha256 不一致")

    failures.extend(
        _identity_differences(current_payload, manifest_payload, prefix="manifest")
    )
    failures.extend(
        _identity_differences(current_payload, checkpoint_payload, prefix="checkpoint")
    )
    failures.extend(
        _identity_differences(
            manifest_payload, checkpoint_payload, prefix="manifest_vs_checkpoint"
        )
    )

    for field, expected_value in current_payload.items():
        if field not in manifest:
            failures.append(f"Manifest 顶层缺少身份字段：{field}")
            continue
        failures.extend(
            _identity_differences(
                expected_value,
                manifest[field],
                prefix=f"manifest.top_level.{field}",
            )
        )

    manifest_code_identity = manifest.get("code_identity")
    if manifest_code_identity is None:
        failures.append("Manifest 缺少 code_identity 字段")
    elif manifest_code_identity != manifest_payload.get("code_identity"):
        failures.extend(
            _identity_differences(
                manifest_payload.get("code_identity"),
                manifest_code_identity,
                prefix="manifest.code_identity_copy",
            )
        )

    # 去重后按顺序报告，避免同一不匹配原因重复出现。
    failures = list(dict.fromkeys(failures))
    if failures:
        detail = "\n - ".join(failures)
        raise ValueError("Checkpoint 恢复身份校验失败：\n - " + detail)


def _validate_training_pool() -> dict[str, Any]:
    config_path = PROJECT_ROOT / pool_contract.CONFIG_RELATIVE
    config = project_config.parse_toml_file(config_path)
    info = pool_contract.validate_formal_training_pool_for_run(
        formal_root=PROJECT_ROOT,
        config_path=config_path,
        runtime_config=config,
        episode_pool=pool_contract.TRAIN_POOL_RELATIVE,
        episode_seed_start=pool_contract.DEFAULT_TRAINING_SEED_START,
        episode_count=pool_contract.FORMAL_TRAINING_EPISODE_COUNT,
        generate_missing=False,
    )
    pool = Path(info["pool_path"])
    valid: list[int] = []
    invalid: list[dict[str, Any]] = []
    for seed in range(
        pool_contract.DEFAULT_TRAINING_SEED_START,
        pool_contract.DEFAULT_TRAINING_SEED_START
        + pool_contract.FORMAL_TRAINING_EPISODE_COUNT,
    ):
        try:
            pool_contract._validate_episode_snapshot(pool, seed, info)
            valid.append(seed)
        except Exception as exc:
            invalid.append(
                {"episode_seed": seed, "error": f"{type(exc).__name__}: {exc}"}
            )
    if invalid:
        raise RuntimeError(
            "正式 Training Pool 存在不满足当前合同的 Episode；拒绝混入训练："
            + json.dumps(invalid[:10], ensure_ascii=False)
        )
    return {
        **info,
        "validated_episode_count": len(valid),
        "invalid_episode_count": 0,
        "episode_seeds": valid,
        "integrity_baseline_sha256": pool_contract._sha256(
            pool / "formal_pool_integrity_baseline.json"
        ),
        "service_catalog_file_sha256": _sha256(
            PROJECT_ROOT
            / "environment_data"
            / "proposed_joint_training"
            / "service_catalog_500_1000.json"
        ),
    }


def _new_run_directory(requested: Path | None) -> Path:
    if requested is not None:
        target = requested if requested.is_absolute() else PROJECT_ROOT / requested
        target = target.resolve()
        target.mkdir(parents=True, exist_ok=False)
        return target
    stem = PROJECT_ROOT / "runs" / (
        "online_joint_dqn_exclusive_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in range(1000):
        target = stem if suffix == 0 else stem.with_name(f"{stem.name}_{suffix:03d}")
        try:
            target.mkdir(exist_ok=False)
            return target
        except FileExistsError:
            continue
    raise FileExistsError("无法分配唯一 DQN 输出目录")


def _run_identity(
    pool_info: Mapping[str, Any],
    trainer_config: TrainerConfig,
    *,
    seed: int,
    replay_capacity: int,
    agent: DoubleDQNAgent,
    search_budget: Mapping[str, int],
) -> dict[str, Any]:
    config_path = PROJECT_ROOT / pool_contract.CONFIG_RELATIVE
    config = project_config.parse_toml_file(config_path)
    identity = {
        "algorithm": "candidate_conditioned_double_dqn_exclusive_v2",
        "state_feature_contract_version": STATE_FEATURE_CONTRACT_VERSION,
        "candidate_feature_contract_version": CANDIDATE_FEATURE_CONTRACT_VERSION,
        "network_architecture_version": NETWORK_ARCHITECTURE_VERSION,
        "seed": int(seed),
        "config_path": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "config_raw_sha256": project_config.raw_config_sha256(config_path),
        "config_semantic_sha256": project_config.semantic_config_hash(config),
        "training_pool_manifest_sha256": pool_info["manifest_sha256"],
        "training_pool_integrity_baseline_sha256": pool_info[
            "integrity_baseline_sha256"
        ],
        "training_task_contract_sha256": pool_info["task_data_contract_sha256"],
        "service_catalog_sha256": pool_info["service_catalog_sha256"],
        "trainer_config": _jsonable(trainer_config.__dict__),
        "agent_config": {
            "state_dim": agent.state_dim,
            "candidate_dim": agent.candidate_dim,
            "gamma": agent.gamma,
            "target_network_frequency_gradient_updates": agent.target_network_frequency,
            "target_tau": agent.target_tau,
            "target_network_sync_mode": (
                "polyak_interpolation" if agent.target_tau < 1.0 else "hard_copy"
            ),
            "max_grad_norm": agent.max_grad_norm,
            "seed": agent.seed,
            "device": str(agent.device),
            "hidden_sizes": [
                int(module.out_features)
                for module in agent.online_network.network
                if isinstance(module, torch.nn.Linear)
            ][:-1],
        },
        "replay_capacity": int(replay_capacity),
        "candidate_capacity_limits": dict(search_budget),
        "code_identity": _code_identity(),
    }
    identity["identity_sha256"] = _identity_sha256(identity)
    return identity


def _save_checkpoint(
    path: Path,
    *,
    agent: DoubleDQNAgent,
    replay: ReplayBuffer,
    identity: Mapping[str, Any],
    episode_order: list[int],
    episode_rows: list[dict[str, Any]],
    next_episode_index: int,
    global_step: int,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "identity_sha256": str(identity["identity_sha256"]),
        "identity_payload": _identity_payload(identity),
        "agent_state": agent.state_dict(),
        "replay_state": replay.state_dict(),
        "episode_order": list(episode_order),
        "completed_episode_rows": list(episode_rows),
        "next_episode_index": int(next_episode_index),
        "completed_episode_count": int(next_episode_index),
        "global_step": int(global_step),
        "epsilon_progress_global_step": int(global_step),
        "gradient_update_count": int(agent.gradient_steps),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return int(path.stat().st_size)


def _read_episode_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_episode_metrics_atomic(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    """按 Checkpoint 提交点原子重写完整 Episode 汇总。"""
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


def _write_run_summary(
    path: Path,
    *,
    status: str,
    episode_rows: list[dict[str, Any]],
    global_step: int,
    gradient_steps: int,
    last_checkpoint: str | None,
    last_checkpoint_bytes: int | None,
    elapsed_seconds: float,
    total_slots: int,
) -> None:
    operation_names = ("KEEP_STAY", "CACHE_UPDATE", "UAV_MOVE")
    totals = {
        key: sum(
            int(row.get("action_counts", {}).get(key, 0)) for row in episode_rows
        )
        for key in operation_names
    }
    episode_utilities = [
        float(row["episode_summary"]["cumulative_system_utility"])
        for row in episode_rows
    ]
    episode_mean_utilities = [
        float(row["episode_summary"]["mean_system_utility_per_slot"])
        for row in episode_rows
        if row["episode_summary"].get("mean_system_utility_per_slot") is not None
    ]
    summary = {
        "status": status,
        "completed_episodes": len(episode_rows),
        "total_planned_episodes": 300,
        "completed_slots": int(global_step),
        "planned_slots": int(total_slots),
        "gradient_updates": int(gradient_steps),
        "action_execution_counts": totals,
        "mean_episode_cumulative_utility": (
            sum(episode_utilities) / len(episode_utilities) if episode_utilities else None
        ),
        "mean_episode_utility_per_slot": (
            sum(episode_mean_utilities) / len(episode_mean_utilities)
            if episode_mean_utilities
            else None
        ),
        "total_executor_calls": sum(
            int(row["episode_summary"]["executor_calls"]) for row in episode_rows
        ),
        "total_matching_calls": sum(
            int(row["episode_summary"]["matching_calls"]) for row in episode_rows
        ),
        "total_kkt_solver_calls": sum(
            int(row["episode_summary"]["kkt_solver_calls"]) for row in episode_rows
        ),
        "total_decision_wall_time_s": sum(
            float(row["episode_summary"]["decision_wall_time_s"])
            for row in episode_rows
        ),
        "training_elapsed_seconds": float(elapsed_seconds),
        "measured_slot_rate_per_second": (
            float(global_step) / float(elapsed_seconds) if elapsed_seconds > 0 else None
        ),
        "estimated_remaining_seconds_from_measured_rate": (
            max(0, int(total_slots) - int(global_step))
            / (float(global_step) / float(elapsed_seconds))
            if elapsed_seconds > 0 and global_step > 0
            else None
        ),
        "last_checkpoint": last_checkpoint,
        "last_checkpoint_bytes": last_checkpoint_bytes,
        "paired_evaluation_status": "NOT_RUN",
        "paired_evaluation_reason": (
            "现有独立评价目录为空；当前 Test Pool 未通过官方 Episode source 合同校验。"
        ),
    }
    _write_json(path, summary, atomic=True)


def run_training(
    *,
    output_dir: Path | None = None,
    resume_checkpoint: Path | None = None,
    seed: int = DEFAULT_SEED,
    replay_capacity: int = DEFAULT_REPLAY_CAPACITY,
) -> int:
    pool_info = _validate_training_pool()
    config_path = PROJECT_ROOT / pool_contract.CONFIG_RELATIVE

    first_seed = pool_contract.DEFAULT_TRAINING_SEED_START
    first_environment = ExclusiveDQNEnvironment.from_formal_training_episode(
        PROJECT_ROOT,
        episode_seed=first_seed,
        origin_start=0,
        max_steps=pool_contract.FORMAL_TRAINING_EPISODE_SLOTS,
    )
    state, candidates = first_environment.observe()
    agent = DoubleDQNAgent(
        len(state),
        len(candidates[0].features),
        seed=int(seed),
    )
    replay = ReplayBuffer(int(replay_capacity), seed=int(seed))
    trainer_config = TrainerConfig()
    search_budget = {
        "max_total_evaluations_per_sample": int(
            first_environment.search_budget.max_total_evaluations
        ),
        "max_cache_candidates_including_keep": int(
            first_environment.search_budget.max_cache_candidates
        ),
        "max_uav_candidates_per_uav_including_stay": int(
            first_environment.search_budget.max_uav_candidates_per_uav
        ),
    }
    identity = _run_identity(
        pool_info,
        trainer_config,
        seed=seed,
        replay_capacity=replay_capacity,
        agent=agent,
        search_budget=search_budget,
    )

    if resume_checkpoint is None:
        run_dir = _new_run_directory(output_dir)
        run_dir.joinpath("checkpoints").mkdir()
        run_dir.joinpath("config_snapshot.toml").write_bytes(config_path.read_bytes())
        _write_json(run_dir / "training_pool_identity.json", pool_info)
        manifest = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            **identity,
            "identity_payload": _identity_payload(identity),
            "code_identity": identity["code_identity"],
            "training_pool": {
                "split": "train",
                "path": str(PROJECT_ROOT / pool_contract.TRAIN_POOL_RELATIVE),
                "episode_count": pool_info["validated_episode_count"],
                "episode_slots": pool_contract.FORMAL_TRAINING_EPISODE_SLOTS,
                "manifest_sha256": pool_info["manifest_sha256"],
                "integrity_baseline_sha256": pool_info[
                    "integrity_baseline_sha256"
                ],
                "task_contract_sha256": pool_info["task_data_contract_sha256"],
                "service_catalog_sha256": pool_info["service_catalog_sha256"],
            },
            "training_episode_order": pool_info["episode_seeds"],
            "evaluation_status": "BLOCKED_BEFORE_EVALUATION",
            "evaluation_blockers": [
                "evaluation_data/ 中没有现有独立评价数据目录。",
                "environment_data/proposed_joint_test 的 100 个 Episode 未通过当前官方 source 合同校验。",
                "已检查的完整 MAPPO Checkpoint 配置哈希与当前正式配置不匹配。",
            ],
            "training_status": "PREPARED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(run_dir / "run_manifest.json", manifest, atomic=True)
        episode_order = list(pool_info["episode_seeds"])
        next_episode_index = 0
        global_step = 0
        checkpoint_episode_rows: list[dict[str, Any]] = []
    else:
        checkpoint_path = resume_checkpoint.resolve(strict=True)
        run_dir = checkpoint_path.parent.parent
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("恢复训练缺少原始 run_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _validate_resume_identity(identity, manifest, payload)
        agent.load_state_dict(payload["agent_state"])
        replay.load_state_dict(payload["replay_state"])
        episode_order = [int(value) for value in payload["episode_order"]]
        if episode_order != list(pool_info["episode_seeds"]):
            raise ValueError("Checkpoint Episode 顺序与正式 Training Pool 不匹配")
        next_episode_index = int(payload["next_episode_index"])
        global_step = int(payload["global_step"])
        checkpoint_episode_rows = [
            dict(row) for row in payload.get("completed_episode_rows", [])
        ]
        if len(checkpoint_episode_rows) != next_episode_index:
            raise ValueError("Checkpoint Episode 汇总数与完成进度不一致")
        expected_indices = list(range(next_episode_index))
        actual_indices = [int(row.get("episode_index", -1)) for row in checkpoint_episode_rows]
        actual_seeds = [int(row.get("episode_seed", -1)) for row in checkpoint_episode_rows]
        if actual_indices != expected_indices or actual_seeds != episode_order[:next_episode_index]:
            raise ValueError("Checkpoint Episode 汇总与正式训练顺序不一致")
        expected_global_step = (
            next_episode_index * pool_contract.FORMAL_TRAINING_EPISODE_SLOTS
        )
        if global_step != expected_global_step:
            raise ValueError("Checkpoint Slot 数与完整 Episode 进度不一致")

    if not np.all(np.isfinite(state)):
        raise FloatingPointError("正式训练初始状态包含非有限值")
    _write_json(
        run_dir / "evaluation_status.json",
        {
            "status": "BLOCKED_BEFORE_EVALUATION",
            "independent_evaluation_data_directories": [],
            "existing_test_pool": "environment_data/proposed_joint_test",
            "official_episode_contract_valid": False,
            "valid_episode_count": 0,
            "invalid_episode_count": 100,
            "reason": (
                "官方 _validate_episode_snapshot 拒绝 Test Pool 的 v1 source 字段"
                "（与当前预期缺少 residual_sampling_contract 且 schema 不同）；"
                "本次不重写验证期望、不跳过门禁。"
            ),
            "mappo_checkpoint_status": "UNAVAILABLE_CONFIG_IDENTITY_MISMATCH",
        },
        atomic=True,
    )

    episode_metrics_path = run_dir / "episode_metrics.jsonl"
    episode_rows = checkpoint_episode_rows
    if _read_episode_metrics(episode_metrics_path) != episode_rows:
        # Checkpoint 是完成进度的提交点；恢复时修复可能落后的派生汇总文件。
        _write_episode_metrics_atomic(episode_metrics_path, episode_rows)
    total_slots = len(episode_order) * pool_contract.FORMAL_TRAINING_EPISODE_SLOTS
    run_start = time.perf_counter()
    checkpoint_path = run_dir / "checkpoints" / "latest_checkpoint.pt"
    last_checkpoint_bytes: int | None = None
    if checkpoint_path.is_file():
        last_checkpoint_bytes = int(checkpoint_path.stat().st_size)
    checkpointed_gradient_steps = int(agent.gradient_steps)

    manifest["training_status"] = "RUNNING"
    manifest["resumed_from_checkpoint"] = (
        str(resume_checkpoint) if resume_checkpoint is not None else None
    )
    manifest["completed_episodes"] = next_episode_index
    manifest["completed_slots"] = global_step
    manifest["gradient_update_count"] = checkpointed_gradient_steps
    _write_json(run_dir / "run_manifest.json", manifest, atomic=True)
    _write_run_summary(
        run_dir / "run_summary.json",
        status="RUNNING",
        episode_rows=episode_rows,
        global_step=global_step,
        gradient_steps=checkpointed_gradient_steps,
        last_checkpoint=str(checkpoint_path) if checkpoint_path.is_file() else None,
        last_checkpoint_bytes=last_checkpoint_bytes,
        elapsed_seconds=0.0,
        total_slots=total_slots,
    )

    stop_requested = False
    previous_signal_handlers: dict[int, Any] = {}

    def request_episode_boundary_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(
            json.dumps(
                {
                    "status": "STOP_REQUESTED",
                    "signal": int(signum),
                    "message": "将在当前完整 Episode 提交后停止",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    for signal_number in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if signal_number is not None:
            previous_signal_handlers[int(signal_number)] = signal.getsignal(signal_number)
            signal.signal(signal_number, request_episode_boundary_stop)

    try:
        attempts_by_seed: dict[int, int] = {}
        for existing_row in _read_jsonl(run_dir / "step_metrics.jsonl"):
            seed_value = int(existing_row.get("episode_seed", -1))
            attempts_by_seed[seed_value] = max(
                attempts_by_seed.get(seed_value, 0),
                int(existing_row.get("attempt", 0)),
            )
        with (run_dir / "step_metrics.jsonl").open(
            "a", encoding="utf-8", buffering=1
        ) as step_stream, (run_dir / "update_metrics.jsonl").open(
            "a", encoding="utf-8", buffering=1
        ) as update_stream:
            for episode_index in range(next_episode_index, len(episode_order)):
                episode_seed = int(episode_order[episode_index])
                attempt = attempts_by_seed.get(episode_seed, 0) + 1
                attempts_by_seed[episode_seed] = attempt
                if episode_index == next_episode_index and episode_index == 0:
                    environment = first_environment
                else:
                    environment = ExclusiveDQNEnvironment.from_formal_training_episode(
                        PROJECT_ROOT,
                        episode_seed=episode_seed,
                        origin_start=0,
                        max_steps=pool_contract.FORMAL_TRAINING_EPISODE_SLOTS,
                    )
                if (
                    environment.encoder.state_dim != agent.state_dim
                    or environment.encoder.candidate_dim != agent.candidate_dim
                    or environment.search_budget.max_cache_candidates
                    != first_environment.search_budget.max_cache_candidates
                    or environment.search_budget.max_uav_candidates_per_uav
                    != first_environment.search_budget.max_uav_candidates_per_uav
                ):
                    raise ValueError(
                        f"Episode {episode_seed} 的状态/候选维度或候选上限不一致"
                    )
                episode_started = time.perf_counter()

                def write_step(row: dict[str, Any]) -> None:
                    _append_jsonl(
                        step_stream,
                        {
                            **row,
                            "episode_seed": episode_seed,
                            "episode_index": episode_index,
                            "attempt": attempt,
                        },
                    )

                def write_update(row: dict[str, float]) -> None:
                    _append_jsonl(
                        update_stream,
                        {
                            **row,
                            "episode_seed": episode_seed,
                            "episode_index": episode_index,
                            "attempt": attempt,
                        },
                    )

                metrics = train_episode(
                    environment,
                    agent,
                    replay,
                    config=trainer_config,
                    global_step_start=global_step,
                    on_step=write_step,
                    on_update=write_update,
                )
                episode_elapsed = time.perf_counter() - episode_started
                if metrics["step_count"] != pool_contract.FORMAL_TRAINING_EPISODE_SLOTS:
                    raise RuntimeError(
                        f"Episode {episode_seed} 实际执行 {metrics['step_count']} Slot，"
                        "与正式 Episode 长度不一致"
                    )
                if not metrics["episode_terminated"] or metrics["window_truncated"]:
                    raise RuntimeError(f"Episode {episode_seed} 未按正式终止状态结束")
                row = {
                    "episode_seed": episode_seed,
                    "episode_index": episode_index,
                    "attempt": attempt,
                    "episode_wall_time_s": episode_elapsed,
                    **metrics,
                }
                # 逐步与逐更新明细已写入 JSONL；Episode 汇总不重复嵌入大数组。
                row.pop("steps", None)
                row.pop("updates", None)
                committed_episode_rows = [*episode_rows, row]
                committed_global_step = int(metrics["global_step_end"])
                committed_episode_index = episode_index + 1
                if hasattr(step_stream, "fileno"):
                    os.fsync(step_stream.fileno())
                    os.fsync(update_stream.fileno())
                last_checkpoint_bytes = _save_checkpoint(
                    checkpoint_path,
                    agent=agent,
                    replay=replay,
                    identity=identity,
                    episode_order=episode_order,
                    episode_rows=committed_episode_rows,
                    next_episode_index=committed_episode_index,
                    global_step=committed_global_step,
                )
                # Checkpoint 成功替换后才提交对应的内存进度与派生日志。
                episode_rows = committed_episode_rows
                global_step = committed_global_step
                next_episode_index = committed_episode_index
                checkpointed_gradient_steps = int(agent.gradient_steps)
                _write_episode_metrics_atomic(episode_metrics_path, episode_rows)
                manifest["training_status"] = "RUNNING"
                manifest["completed_episodes"] = next_episode_index
                manifest["completed_slots"] = global_step
                manifest["gradient_update_count"] = checkpointed_gradient_steps
                manifest["last_checkpoint"] = str(checkpoint_path)
                manifest["last_checkpoint_bytes"] = last_checkpoint_bytes
                manifest["last_completed_episode_seed"] = episode_seed
                _write_json(run_dir / "run_manifest.json", manifest, atomic=True)
                _write_run_summary(
                    run_dir / "run_summary.json",
                    status="RUNNING",
                    episode_rows=episode_rows,
                    global_step=global_step,
                    gradient_steps=checkpointed_gradient_steps,
                    last_checkpoint=str(checkpoint_path),
                    last_checkpoint_bytes=last_checkpoint_bytes,
                    elapsed_seconds=time.perf_counter() - run_start,
                    total_slots=total_slots,
                )
                elapsed = time.perf_counter() - run_start
                rate = global_step / elapsed if elapsed > 0 else 0.0
                remaining = max(0, total_slots - global_step)
                eta = remaining / rate if rate > 0 else None
                print(
                    json.dumps(
                        {
                            "episode_seed": episode_seed,
                            "completed_episodes": next_episode_index,
                            "total_episodes": len(episode_order),
                            "completed_slots": global_step,
                            "gradient_updates": agent.gradient_steps,
                            "episode_seconds": episode_elapsed,
                            "measured_slots_per_second": rate,
                            "estimated_remaining_seconds": eta,
                            "checkpoint_bytes": last_checkpoint_bytes,
                            "run_dir": str(run_dir),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if stop_requested:
                    break
        status = (
            "STOPPED_AT_EPISODE_BOUNDARY"
            if stop_requested and next_episode_index < len(episode_order)
            else "FORMAL_TRAINING_COMPLETE"
        )
    except BaseException as exc:
        status = "STOPPED_WITH_ERROR"
        failure = {
            "status": status,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_episodes": next_episode_index,
            "completed_slots_checkpointed": global_step,
            "gradient_updates_checkpointed": checkpointed_gradient_steps,
            "last_checkpoint": str(checkpoint_path) if checkpoint_path.is_file() else None,
            "diagnostic_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(run_dir / "failure.json", failure, atomic=True)
        manifest["training_status"] = status
        manifest["failure"] = failure
        _write_json(run_dir / "run_manifest.json", manifest, atomic=True)
        _write_run_summary(
            run_dir / "run_summary.json",
            status=status,
            episode_rows=episode_rows,
            global_step=global_step,
            gradient_steps=checkpointed_gradient_steps,
            last_checkpoint=str(checkpoint_path) if checkpoint_path.is_file() else None,
            last_checkpoint_bytes=last_checkpoint_bytes,
            elapsed_seconds=time.perf_counter() - run_start,
            total_slots=total_slots,
        )
        raise
    finally:
        for signal_number, previous_handler in previous_signal_handlers.items():
            signal.signal(signal_number, previous_handler)

    manifest["training_status"] = status
    manifest["completed_episodes"] = next_episode_index
    manifest["completed_slots"] = global_step
    manifest["gradient_update_count"] = checkpointed_gradient_steps
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(run_dir / "run_manifest.json", manifest, atomic=True)
    _write_run_summary(
        run_dir / "run_summary.json",
        status=status,
        episode_rows=episode_rows,
        global_step=global_step,
        gradient_steps=checkpointed_gradient_steps,
        last_checkpoint=str(checkpoint_path),
        last_checkpoint_bytes=last_checkpoint_bytes,
        elapsed_seconds=time.perf_counter() - run_start,
        total_slots=total_slots,
    )
    print(json.dumps({"status": status, "run_dir": str(run_dir)}, ensure_ascii=False))
    return 130 if status == "STOPPED_AT_EPISODE_BOUNDARY" else 0


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.is_file():
        return []
    rows = []
    with file_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--replay-capacity", type=int, default=DEFAULT_REPLAY_CAPACITY)
    args = parser.parse_args()
    if int(args.replay_capacity) <= 0:
        parser.error("--replay-capacity 必须为正整数")
    return run_training(
        output_dir=args.output_dir,
        resume_checkpoint=args.resume_checkpoint,
        seed=args.seed,
        replay_capacity=args.replay_capacity,
    )


if __name__ == "__main__":
    raise SystemExit(main())
