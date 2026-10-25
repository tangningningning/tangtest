"""运行带身份记录、逐 Slot 日志和可恢复 Checkpoint 的独立 DQN 训练。"""

from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import Future, ThreadPoolExecutor
import copy
import hashlib
import io
import json
import math
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
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
from decision import uav_configuration  # noqa: E402
from online_joint_dqn.agent import DoubleDQNAgent  # noqa: E402
from online_joint_dqn.candidate_features import (  # noqa: E402
    CANDIDATE_FEATURE_CONTRACT_VERSION,
    NETWORK_ARCHITECTURE_VERSION,
    STATE_FEATURE_CONTRACT_VERSION,
)
from online_joint_dqn.environment_adapter import (  # noqa: E402
    CANDIDATE_GENERATION_CONTRACT_VERSION,
    DQN_CROSS_NODE_PAIR_PROPOSAL_LIMIT,
    DQN_CROSS_NODE_PAIR_SCAN_LIMIT,
    ExclusiveDQNEnvironment,
    candidate_generation_contract,
)
from online_joint_dqn.evaluation import (  # noqa: E402
    evaluate_formal_split,
    evaluation_pool_identity,
    policy_state_sha256,
    should_update_best,
    validate_formal_evaluation_pool,
)
from online_joint_dqn.replay_buffer import ReplayBuffer  # noqa: E402
from online_joint_dqn.trainer import TrainerConfig, train_episode  # noqa: E402
import training_pool_contract as pool_contract  # noqa: E402


RUN_MANIFEST_SCHEMA_VERSION = "online_joint_dqn_run_v13_validation_selection"
CHECKPOINT_SCHEMA_VERSION = 15
DEFAULT_SEED = 20261022
DEFAULT_REPLAY_CAPACITY = 10000
DEFAULT_LEARNING_RATE = 2.5e-4
DEFAULT_RUN_NAME_PREFIX = "online_joint_dqn_fixed_input"
PLOT_REFRESH_INTERVAL_EPISODES = 10
VALIDATION_INTERVAL_EPISODES = 50
MILESTONE_EPISODES = (50, 100, 150, 200, 250, 300)


def _should_refresh_training_plots(completed_episode_count: int) -> bool:
    """只在完整 Episode 数达到 10 的倍数时请求一次后台绘图。"""

    count = int(completed_episode_count)
    return count > 0 and count % PLOT_REFRESH_INTERVAL_EPISODES == 0

_CODE_IDENTITY_REQUIRED_PATHS = (
    "src/common/project_config.py",
    "src/common/system_model.py",
    "src/decision/cache_configuration.py",
    "src/decision/uav_configuration.py",
    "src/decision/slot_environment.py",
    "src/decision/task_allocation.py",
    "src/decision/task_lifecycle.py",
    "src/common/environment_interface.py",
    "src/scenario/realization.py",
    "src/scenario/vehicle_order.py",
    "scripts/training/training_pool_contract.py",
    "scripts/evaluation/evaluate_online_joint_dqn.py",
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


def _capture_global_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_global_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


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
    info = pool_contract.validate_dqn_training_pool_for_run(
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
    for seed in info["active_episode_seeds"]:
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


def _validate_independent_validation_pool() -> dict[str, Any]:
    """训练前冻结 Validation 场景身份，并拒绝不完整或损坏的场景。"""

    pool, info = validate_formal_evaluation_pool(PROJECT_ROOT, "validation")
    return {
        **info,
        "pool_path": str(pool),
        "validated_episode_count": len(info["active_episode_seeds"]),
        "validation_pool_identity": evaluation_pool_identity(info),
    }


def _validate_independent_test_pool(
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    """只验证冻结 Test Pool 合同，不运行任何环境 Slot。"""

    pool_path = PROJECT_ROOT / pool_contract.TEST_POOL_RELATIVE
    try:
        pool, info = pool_contract._validate_pool(
            PROJECT_ROOT,
            pool_contract.TEST_POOL_RELATIVE,
            expected_split="test",
            expected_seed_start=pool_contract.DEFAULT_TEST_SEED_START,
            expected_episode_count=pool_contract.FORMAL_TEST_EPISODE_COUNT,
            runtime_config=dict(runtime_config),
        )
        valid_seeds: list[int] = []
        invalid_episodes: list[dict[str, Any]] = []
        for seed in info["active_episode_seeds"]:
            try:
                pool_contract._validate_episode_snapshot(pool, seed, info)
                valid_seeds.append(seed)
            except Exception as exc:
                invalid_episodes.append(
                    {"episode_seed": seed, "error": f"{type(exc).__name__}: {exc}"}
                )
        return {
            "status": "PASS" if not invalid_episodes else "BLOCKED",
            "path": str(pool_path),
            "split": "test",
            "seed_start": min(info["active_episode_seeds"]),
            "seed_end": max(info["active_episode_seeds"]),
            "active_episode_seeds": info["active_episode_seeds"],
            "manifest_sha256": info["manifest_sha256"],
            "config_compatibility": info["config_compatibility"],
            "valid_episode_count": len(valid_seeds),
            "invalid_episode_count": len(invalid_episodes),
            "invalid_episodes": invalid_episodes[:10],
        }
    except Exception as exc:
        return {
            "status": "BLOCKED",
            "path": str(pool_path),
            "split": "test",
            "valid_episode_count": 0,
            "invalid_episode_count": pool_contract.FORMAL_TEST_EPISODE_COUNT,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _independent_evaluation_gate(
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    test_pool = _validate_independent_test_pool(runtime_config)
    blockers: list[str] = []
    if test_pool.get("status") != "PASS":
        blockers.append("正式独立 Test Pool 未通过当前逐 Episode 数据合同。")
    return {
        "status": "BLOCKED_BEFORE_EVALUATION" if blockers else "READY_NOT_RUN",
        "blockers": blockers,
        "test_pool": test_pool,
        "evaluation_executed": False,
    }


def _evaluation_status_payload(gate: Mapping[str, Any]) -> dict[str, Any]:
    """把数据和基线门禁的真实结果写入独立评价状态文件。"""

    test_pool = gate.get("test_pool", {})
    pool_valid = test_pool.get("status") == "PASS"
    blockers = list(gate.get("blockers", []))
    return {
        "status": str(gate.get("status", "UNAVAILABLE")),
        "evaluation_executed": False,
        "independent_evaluation_data_directories": (
            [test_pool["path"]] if pool_valid and test_pool.get("path") else []
        ),
        "existing_test_pool": test_pool.get("path"),
        "official_episode_contract_valid": pool_valid,
        "valid_episode_count": int(test_pool.get("valid_episode_count", 0)),
        "invalid_episode_count": int(test_pool.get("invalid_episode_count", 0)),
        "test_pool": _jsonable(test_pool),
        "blockers": blockers,
        "reason": (
            "；".join(str(item) for item in blockers)
            if blockers
            else "独立 Test Pool 通过数据合同；本次未执行 DQN 冻结评价。"
        ),
    }


def _create_unique_run_directory(stem: Path) -> Path:
    """按原子目录创建分配唯一训练目录；碰撞时递增短序号。"""

    stem.parent.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        target = (
            stem
            if suffix == 0
            else stem.with_name(f"{stem.name}_{suffix:02d}")
        )
        try:
            target.mkdir(exist_ok=False)
            return target
        except FileExistsError:
            suffix += 1


def _new_run_directory(
    requested: Path | None,
    *,
    run_name_prefix: str = DEFAULT_RUN_NAME_PREFIX,
) -> Path:
    if requested is not None:
        target = requested if requested.is_absolute() else PROJECT_ROOT / requested
        target = target.resolve()
        target.mkdir(parents=True, exist_ok=False)
        return target
    # 使用本机当地时间，便于直接辨认训练启动时间。
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    stem = PROJECT_ROOT / "runs" / f"{run_name_prefix}_{timestamp}"
    return _create_unique_run_directory(stem)


def _run_identity(
    pool_info: Mapping[str, Any],
    validation_pool_info: Mapping[str, Any],
    trainer_config: TrainerConfig,
    *,
    seed: int,
    replay_capacity: int,
    agent: DoubleDQNAgent,
    search_budget: Mapping[str, int],
) -> dict[str, Any]:
    config_path = PROJECT_ROOT / pool_contract.CONFIG_RELATIVE
    config = project_config.parse_toml_file(config_path)
    time_horizon = project_config.get_cross_slot_task_parameters(config)
    identity = {
        "algorithm": "legal_candidate_mean_centered_dueling_double_dqn_exclusive_v6_active_task_state",
        "objective_contract": "finite_episode_undiscounted_terminal_utility_extended_horizon_v3",
        "time_horizon_contract": {
            "arrival_horizon_slots": int(time_horizon["arrival_horizon_slots"]),
            "execution_horizon_slots": int(time_horizon["execution_horizon_slots"]),
            "k_max_slots": int(time_horizon["k_max_slots"]),
            "dt_s": float(time_horizon["dt_s"]),
            "terminal_origin": int(time_horizon["terminal_origin"]),
            "terminal_execution_slot": int(time_horizon["terminal_execution_slot"]),
            "formula": "execution_horizon_slots=arrival_horizon_slots+k_max_slots-1",
            "indexing": "zero_based",
        },
        "dqn_system_utility": {
            "formula": "G(t)=Q(t)-C_ca(t)-C_q(t)",
            "task_utility": "PDF terminal utility for tasks terminating after this slot",
            "cache_cost_coefficient": 1.0,
            "uav_movement_cost_coefficient": 1.0,
            "legacy_objective_beta_fields_applied": False,
        },
        "state_feature_contract_version": STATE_FEATURE_CONTRACT_VERSION,
        "candidate_feature_contract_version": CANDIDATE_FEATURE_CONTRACT_VERSION,
        "candidate_generation_contract_version": CANDIDATE_GENERATION_CONTRACT_VERSION,
        "candidate_generation_contract": candidate_generation_contract(
            max_cache_candidates_including_keep=int(
                search_budget["max_cache_candidates_including_keep"]
            ),
            max_uav_candidates_per_uav_including_stay=int(
                search_budget["max_uav_candidates_per_uav_including_stay"]
            ),
        ),
        "network_architecture_version": NETWORK_ARCHITECTURE_VERSION,
        "seed": int(seed),
        "learning_rate": float(agent.optimizer.param_groups[0]["lr"]),
        "config_path": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "config_raw_sha256": project_config.raw_config_sha256(config_path),
        "config_semantic_sha256": project_config.semantic_config_hash(config),
        "training_pool_manifest_sha256": pool_info["manifest_sha256"],
        "training_pool_integrity_baseline_sha256": pool_info[
            "integrity_baseline_sha256"
        ],
        "training_task_contract_sha256": pool_info["task_data_contract_sha256"],
        "service_catalog_sha256": pool_info["service_catalog_sha256"],
        "validation_pool_identity": dict(
            validation_pool_info["validation_pool_identity"]
        ),
        "trainer_config": _jsonable(trainer_config.__dict__),
        "agent_config": {
            "state_dim": agent.state_dim,
            "candidate_dim": agent.candidate_dim,
            "gamma": agent.gamma,
            "learning_rate": float(agent.optimizer.param_groups[0]["lr"]),
            "optimizer_type": type(agent.optimizer).__name__,
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
        "replay_buffer_schema_version": ReplayBuffer.STATE_SCHEMA_VERSION,
        "candidate_capacity_limits": dict(search_budget),
        "candidate_generation_limits": {
            "cross_node_pair_proposal_limit": DQN_CROSS_NODE_PAIR_PROPOSAL_LIMIT,
            "cross_node_pair_scan_limit": DQN_CROSS_NODE_PAIR_SCAN_LIMIT,
        },
        "code_identity": _code_identity(),
    }
    identity["identity_sha256"] = _identity_sha256(identity)
    return identity


def _checkpoint_payload(
    path: Path,
    *,
    agent: DoubleDQNAgent,
    replay: ReplayBuffer,
    identity: Mapping[str, Any],
    episode_order: list[int],
    episode_rows: list[dict[str, Any]],
    next_episode_index: int,
    global_step: int,
    checkpoint_role: str = "latest",
    training_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if checkpoint_role not in {"latest", "milestone", "best"}:
        raise ValueError(f"Unsupported checkpoint role: {checkpoint_role!r}")
    episode_index = int(next_episode_index)
    step = int(global_step)
    if not 0 <= episode_index <= len(episode_order):
        raise ValueError("Checkpoint Episode progress is outside the formal order")
    if len(episode_rows) != episode_index:
        raise ValueError("Checkpoint Episode summaries do not match its progress")
    identity_payload = _identity_payload(identity)
    if _identity_sha256(identity_payload) != str(identity.get("identity_sha256", "")):
        raise ValueError("Cannot save a Checkpoint with an invalid run identity digest")
    time_contract = identity_payload.get("time_horizon_contract", {})
    execution_horizon = int(time_contract.get("execution_horizon_slots", 0))
    if execution_horizon and step != episode_index * execution_horizon:
        raise ValueError("Checkpoint global step is inconsistent with complete Episode progress")
    if checkpoint_role == "milestone" and episode_index not in MILESTONE_EPISODES:
        raise ValueError("A milestone Checkpoint must match a formal milestone Episode")
    if checkpoint_role == "best" and not isinstance(
        (training_metadata or {}).get("best_checkpoint"), Mapping
    ):
        raise ValueError("A best Checkpoint requires Validation selection metadata")
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": checkpoint_role,
        "identity_sha256": str(identity["identity_sha256"]),
        "identity_payload": identity_payload,
        "agent_state": agent.state_dict(),
        "replay_state": replay.state_dict(),
        "episode_order": list(episode_order),
        "completed_episode_rows": list(episode_rows),
        "next_episode_index": episode_index,
        "completed_episode_count": episode_index,
        "global_step": step,
        "epsilon_progress_global_step": step,
        "gradient_update_count": int(agent.gradient_steps),
        "training_metadata": _jsonable(training_metadata or {}),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return int(path.stat().st_size)


def _checkpoint_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
        ):
            return False
        try:
            return bool(np.array_equal(left, right, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(left, right))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_checkpoint_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            isinstance(left, (tuple, list))
            and isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(
                _checkpoint_values_equal(a, b)
                for a, b in zip(left, right, strict=True)
            )
        )
    try:
        result = left == right
        return bool(result) if not isinstance(result, np.ndarray) else bool(result.all())
    except Exception:
        return False


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
    checkpoint_role: str = "latest",
    training_metadata: Mapping[str, Any] | None = None,
) -> int:
    payload = _checkpoint_payload(
        path,
        agent=agent,
        replay=replay,
        identity=identity,
        episode_order=episode_order,
        episode_rows=episode_rows,
        next_episode_index=next_episode_index,
        global_step=global_step,
        checkpoint_role=checkpoint_role,
        training_metadata=training_metadata,
    )
    return _atomic_torch_save(path, payload)


def _save_immutable_checkpoint(
    path: Path,
    **checkpoint_arguments: Any,
) -> int:
    """写一次性里程碑；重复续跑仅允许精确复用相同快照。"""

    payload = _checkpoint_payload(path, **checkpoint_arguments)
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        comparable_existing = {
            key: value for key, value in existing.items() if key != "saved_at_utc"
        }
        comparable_payload = {
            key: value for key, value in payload.items() if key != "saved_at_utc"
        }
        if not _checkpoint_values_equal(comparable_existing, comparable_payload):
            raise FileExistsError(
                "Existing milestone checkpoint has different identity or state; "
                f"refusing to overwrite: {path}"
            )
        return int(path.stat().st_size)
    return _atomic_torch_save(path, payload)


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


def _validation_artifact(
    run_dir: Path,
    *,
    completed_episode_count: int,
    identity: Mapping[str, Any],
    agent: DoubleDQNAgent,
    validation_pool_info: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, str]:
    """读取或原子创建一个完整且身份固定的 Validation 结果文件。"""

    from online_joint_dqn.evaluation import EVALUATION_SCHEMA_VERSION

    pool_identity = dict(validation_pool_info["validation_pool_identity"])
    policy_hash = policy_state_sha256(agent)
    checkpoint_identity = {
        "run_identity_sha256": str(identity["identity_sha256"]),
        "completed_episode_count": int(completed_episode_count),
        "policy_state_sha256": policy_hash,
        "checkpoint_role": "latest_policy_at_validation_boundary",
    }
    path = run_dir / "validation" / f"validation_episode_{completed_episode_count:03d}.json"
    if path.is_file():
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if (
            artifact.get("schema_version") != EVALUATION_SCHEMA_VERSION
            or artifact.get("status") != "COMPLETE"
            or artifact.get("completed_episode_count") != completed_episode_count
            or artifact.get("run_identity_sha256") != identity["identity_sha256"]
            or artifact.get("checkpoint_identity") != checkpoint_identity
            or artifact.get("validation_pool_identity") != pool_identity
        ):
            raise ValueError(
                "Existing Validation artifact has a different checkpoint or scene identity; "
                f"refusing to reuse or overwrite: {path}"
            )
        expected_seeds = list(pool_identity["active_episode_seeds"])
        rows = artifact.get("episode_summaries")
        if (
            not isinstance(rows, list)
            or [int(row.get("episode_seed", -1)) for row in rows] != expected_seeds
            or [row.get("scene_sha256") for row in rows]
            != [pool_identity["active_episode_scene_sha256"][str(seed)] for seed in expected_seeds]
        ):
            raise ValueError("Existing Validation artifact is missing scenes or scene identities")
        metric = float(artifact["selection_metric_value"])
        if not math.isfinite(metric):
            raise ValueError("Existing Validation artifact has a non-finite selection metric")
        return artifact, path, _sha256(path)

    result = evaluate_formal_split(
        PROJECT_ROOT,
        split="validation",
        policies=("dqn_greedy",),
        agent=agent,
        validated_pool=(Path(validation_pool_info["pool_path"]), validation_pool_info),
    )
    if result.get("status") != "COMPLETE" or result.get("pool_identity") != pool_identity:
        raise RuntimeError("Validation did not complete against the pinned formal pool")
    policy_summary = result["policy_summaries"]["dqn_greedy"]
    metric = float(policy_summary["mean_episode_system_utility"])
    if not math.isfinite(metric):
        raise FloatingPointError("Validation mean Episode utility is non-finite")
    episode_summaries = [
        {
            key: value
            for key, value in row.items()
            if key != "per_slot"
        }
        for row in result["episode_results"]
    ]
    expected_seeds = list(pool_identity["active_episode_seeds"])
    if [int(row["episode_seed"]) for row in episode_summaries] != expected_seeds:
        raise RuntimeError("Validation did not evaluate the full active seed order")
    artifact = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "COMPLETE",
        "split": "validation",
        "completed_episode_count": int(completed_episode_count),
        "run_identity_sha256": str(identity["identity_sha256"]),
        "checkpoint_identity": checkpoint_identity,
        "validation_pool_identity": pool_identity,
        "selection_metric": "mean_episode_system_utility",
        "selection_metric_value": metric,
        "policy_summary": policy_summary,
        "episode_summaries": episode_summaries,
    }
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite a Validation artifact: {path}")
    _write_json(path, artifact, atomic=True)
    return artifact, path, _sha256(path)


def _finish_validation_checkpoint_boundary(
    run_dir: Path,
    *,
    completed_episode_count: int,
    agent: DoubleDQNAgent,
    replay: ReplayBuffer,
    identity: Mapping[str, Any],
    episode_order: list[int],
    episode_rows: list[dict[str, Any]],
    global_step: int,
    training_metadata: dict[str, Any],
    validation_pool_info: Mapping[str, Any],
) -> tuple[dict[str, Any], int, int | None]:
    if completed_episode_count not in MILESTONE_EPISODES:
        raise ValueError("Validation/checkpoint boundary is not a formal milestone")
    artifact, artifact_path, artifact_sha = _validation_artifact(
        run_dir,
        completed_episode_count=completed_episode_count,
        identity=identity,
        agent=agent,
        validation_pool_info=validation_pool_info,
    )
    metric = float(artifact["selection_metric_value"])
    checkpoint_identity = dict(artifact["checkpoint_identity"])
    pool_identity = dict(artifact["validation_pool_identity"])
    validation_history = list(training_metadata.get("validation_history", []))
    record = {
        "episode_count": int(completed_episode_count),
        "checkpoint_identity": checkpoint_identity,
        "validation_pool_identity": pool_identity,
        "selection_metric": "mean_episode_system_utility",
        "selection_metric_value": metric,
        "policy_summary": dict(artifact["policy_summary"]),
        "scene_identities": [
            {
                "episode_seed": int(row["episode_seed"]),
                "scene_sha256": str(row["scene_sha256"]),
            }
            for row in artifact["episode_summaries"]
        ],
        "artifact_path": artifact_path.relative_to(run_dir).as_posix(),
        "artifact_sha256": artifact_sha,
    }
    matching_records = [
        item for item in validation_history
        if int(item.get("episode_count", -1)) == completed_episode_count
    ]
    if matching_records:
        if len(matching_records) != 1 or matching_records[0] != record:
            raise ValueError(
                "Checkpoint Validation history conflicts with its immutable result artifact"
            )
    else:
        validation_history.append(record)
        validation_history.sort(key=lambda item: int(item["episode_count"]))

    best_metadata = training_metadata.get("best_checkpoint")
    if best_metadata is not None and not isinstance(best_metadata, Mapping):
        raise ValueError("Checkpoint best-selection metadata is malformed")
    best_score = (
        None
        if best_metadata is None
        else float(best_metadata["selection_metric_value"])
    )
    best_path = run_dir / "checkpoints" / "best_checkpoint.pt"
    best_file_sha = training_metadata.get("best_checkpoint_file_sha256")
    if best_path.is_file():
        actual_best_sha = _sha256(best_path)
        if best_file_sha is not None and best_file_sha != actual_best_sha:
            raise ValueError("best_checkpoint.pt SHA-256 differs from latest checkpoint metadata")
        best_file_sha = actual_best_sha
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        if (
            best_payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or best_payload.get("identity_sha256") != identity["identity_sha256"]
            or best_payload.get("checkpoint_role") != "best"
        ):
            raise ValueError("Existing best checkpoint has an incompatible run identity")
        stored_best_metadata = best_payload.get("training_metadata", {}).get(
            "best_checkpoint"
        )
        if best_metadata is None or not _checkpoint_values_equal(
            dict(best_metadata), stored_best_metadata
        ):
            raise ValueError("Existing best checkpoint differs from latest selection metadata")
    elif best_metadata is not None:
        raise FileNotFoundError("Latest Checkpoint names a best model whose file is missing")

    improved = should_update_best(best_score, metric)
    if improved:
        best_metadata = {
            "episode_count": int(completed_episode_count),
            "selection_metric": "mean_episode_system_utility",
            "selection_metric_value": metric,
            "policy_state_sha256": str(checkpoint_identity["policy_state_sha256"]),
            "checkpoint_identity": checkpoint_identity,
        }
        best_training_metadata = {
            "validation_history": validation_history,
            "best_checkpoint": best_metadata,
        }
        _save_checkpoint(
            best_path,
            agent=agent,
            replay=replay,
            identity=identity,
            episode_order=episode_order,
            episode_rows=episode_rows,
            next_episode_index=completed_episode_count,
            global_step=global_step,
            checkpoint_role="best",
            training_metadata=best_training_metadata,
        )
        best_file_sha = _sha256(best_path)
    training_metadata["validation_history"] = validation_history
    training_metadata["best_checkpoint"] = (
        None if best_metadata is None else dict(best_metadata)
    )
    training_metadata["best_checkpoint_file_sha256"] = best_file_sha
    latest_path = run_dir / "checkpoints" / "latest_checkpoint.pt"
    latest_bytes = _save_checkpoint(
        latest_path,
        agent=agent,
        replay=replay,
        identity=identity,
        episode_order=episode_order,
        episode_rows=episode_rows,
        next_episode_index=completed_episode_count,
        global_step=global_step,
        checkpoint_role="latest",
        training_metadata=training_metadata,
    )
    milestone_path = (
        run_dir
        / "checkpoints"
        / f"milestone_episode_{completed_episode_count:03d}.pt"
    )
    milestone_bytes = _save_immutable_checkpoint(
        milestone_path,
        agent=agent,
        replay=replay,
        identity=identity,
        episode_order=episode_order,
        episode_rows=episode_rows,
        next_episode_index=completed_episode_count,
        global_step=global_step,
        checkpoint_role="milestone",
        training_metadata=training_metadata,
    )
    return {
        "record": record,
        "best_checkpoint": (
            None
            if best_metadata is None
            else {**dict(best_metadata), "file_sha256": best_file_sha}
        ),
        "best_updated": improved,
    }, latest_bytes, milestone_bytes


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
    learning_rate: float,
    beta_ca: float,
    beta_q: float,
) -> None:
    evaluation_status: dict[str, Any] = {}
    evaluation_status_path = path.parent / "evaluation_status.json"
    if evaluation_status_path.is_file():
        try:
            loaded_status = json.loads(
                evaluation_status_path.read_text(encoding="utf-8")
            )
            if isinstance(loaded_status, dict):
                evaluation_status = loaded_status
        except (OSError, json.JSONDecodeError):
            evaluation_status = {"status": "UNAVAILABLE"}
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
    execution_node_counts: Counter[str] = Counter()
    execution_node_data_available_episodes = 0
    for row in episode_rows:
        episode_summary = row.get("episode_summary", {})
        counts = episode_summary.get("actual_execution_node_counts")
        if (
            episode_summary.get("actual_execution_node_data_available") is True
            and isinstance(counts, Mapping)
        ):
            execution_node_data_available_episodes += 1
            for node_id, count in counts.items():
                execution_node_counts[str(node_id)] += int(count)
    execution_node_total = sum(execution_node_counts.values())
    execution_node_shares = {
        node_id: count / execution_node_total
        for node_id, count in sorted(execution_node_counts.items())
    } if execution_node_total else {}
    all_episode_node_data_available = (
        bool(episode_rows)
        and execution_node_data_available_episodes == len(episode_rows)
    )
    most_common_execution_node = (
        min(
            execution_node_counts,
            key=lambda node_id: (-execution_node_counts[node_id], node_id),
        )
        if all_episode_node_data_available and execution_node_total
        else None
    )
    summary = {
        "status": status,
        "completed_episodes": len(episode_rows),
        "total_planned_episodes": 300,
        "completed_slots": int(global_step),
        "planned_slots": int(total_slots),
        "gradient_updates": int(gradient_steps),
        "learning_rate": float(learning_rate),
        "optimizer_type": "Adam",
        "system_utility_formula": "G(t)=Q(t)-C_ca(t)-C_q(t)",
        "legacy_objective_beta_fields_applied": False,
        "legacy_objective_beta_ca_config_not_applied": float(beta_ca),
        "legacy_objective_beta_q_config_not_applied": float(beta_q),
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
        "actual_execution_node_data_available": all_episode_node_data_available,
        "actual_execution_node_data_available_episode_count": (
            execution_node_data_available_episodes
        ),
        "actual_execution_node_data_unavailable_episode_count": (
            len(episode_rows) - execution_node_data_available_episodes
        ),
        "actual_execution_node_task_count": execution_node_total,
        "actual_execution_node_counts": dict(sorted(execution_node_counts.items())),
        "actual_execution_node_shares": execution_node_shares,
        "most_common_actual_execution_node": most_common_execution_node,
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
        "independent_evaluation_status": "NOT_RUN",
        "independent_evaluation_reason": (
            str(evaluation_status.get("reason"))
            if evaluation_status.get("reason")
            else "本运行未执行独立 DQN 评价。"
        ),
        "independent_evaluation_gate_status": evaluation_status.get(
            "status", "UNAVAILABLE"
        ),
        "independent_evaluation_test_pool_valid_episode_count": evaluation_status.get(
            "valid_episode_count"
        ),
        "independent_evaluation_test_pool_invalid_episode_count": evaluation_status.get(
            "invalid_episode_count"
        ),
    }
    _write_json(path, summary, atomic=True)


def _format_eta(seconds: float | None) -> str:
    """将剩余训练时间统一格式化为 HH:MM:SS。"""

    if seconds is None or not math.isfinite(float(seconds)):
        return "--:--:--"
    total_seconds = max(0, int(math.ceil(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_remainder = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_remainder:02d}"


def _normalization_config_line(config: Mapping[str, Any]) -> str:
    """从正式配置生成唯一的归一化参数摘要行。"""

    get = project_config.get_config_value
    cache_reference = float(
        get(config, "objective.cost_normalization.cache_reference_s_per_node")
    )
    beta_ca = float(get(config, "objective.beta_ca"))
    beta_q = float(get(config, "objective.beta_q"))
    energy_references = uav_configuration.movement_energy_upper_bounds_by_uav_j(
        dict(config)
    )
    if not energy_references:
        raise ValueError("正式配置未提供 UAV 能耗归一化参考值")
    reference_values = tuple(float(value) for value in energy_references.values())
    first_reference = reference_values[0]
    if all(
        math.isclose(value, first_reference, rel_tol=1.0e-12, abs_tol=1.0e-12)
        for value in reference_values[1:]
    ):
        uav_reference = f"{first_reference:g}J/UAV"
    else:
        uav_reference = (
            "["
            + ",".join(f"{value:g}" for value in reference_values)
            + "]J/UAV"
        )
    return (
        f"[CONFIG] cache_ref={cache_reference:g}s/node | "
        f"uav_energy_ref={uav_reference} | "
        f"legacy_beta_ca_unused={beta_ca:g} | legacy_beta_q_unused={beta_q:g}"
    )


def _format_episode_progress(
    *,
    episode_index: int,
    total_episodes: int,
    episode_seed: int,
    episode_return: float,
    mean_utility: float,
    completed_slots: int,
    total_slots: int,
    gradient_updates: int,
    speed_slots_per_second: float,
    eta_seconds: float | None,
    cache_updates: int | None,
    cache_nodes: int | None,
    uav_moves: int | None,
    uav_move_slots: int | None,
    uav_total_distance_m: float | None,
    top_execution_node: str,
) -> str:
    """生成单个已提交 Episode 的一行控制台进度。"""

    format_count = lambda value: "N/A" if value is None else str(int(value))
    distance_text = (
        "N/A" if uav_total_distance_m is None else f"{uav_total_distance_m:.1f}m"
    )
    return (
        f"[DQN] EP {episode_index:03d}/{total_episodes} "
        f"| Seed {episode_seed} "
        f"| Return {episode_return:.2f} "
        f"| Mean {mean_utility:.3f} "
        f"| Slots {completed_slots}/{total_slots} "
        f"| Updates {gradient_updates} "
        f"| Speed {speed_slots_per_second:.2f} slot/s "
        f"| ETA {_format_eta(eta_seconds)}"
        f" | CacheUpdates {format_count(cache_updates)}"
        f" | CacheNodes {format_count(cache_nodes)}"
        f" | UAVMoves {format_count(uav_moves)}"
        f" | UAVSlots {format_count(uav_move_slots)}"
        f" | UAVDistance {distance_text}"
        f" | TopExec {top_execution_node}"
    )


def run_training(
    *,
    output_dir: Path | None = None,
    resume_checkpoint: Path | None = None,
    seed: int = DEFAULT_SEED,
    replay_capacity: int = DEFAULT_REPLAY_CAPACITY,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    run_name_prefix: str = DEFAULT_RUN_NAME_PREFIX,
    max_episodes: int | None = None,
) -> int:
    if max_episodes is not None:
        if isinstance(max_episodes, bool) or int(max_episodes) != max_episodes:
            raise ValueError("max_episodes 必须为正整数")
        max_episodes = int(max_episodes)
        if max_episodes <= 0:
            raise ValueError("max_episodes 必须为正整数")

    pool_info = _validate_training_pool()
    config_path = PROJECT_ROOT / pool_contract.CONFIG_RELATIVE
    runtime_config = project_config.parse_toml_file(config_path)
    validation_pool_info = _validate_independent_validation_pool()
    beta_ca = float(
        project_config.get_config_value(runtime_config, "objective.beta_ca")
    )
    beta_q = float(
        project_config.get_config_value(runtime_config, "objective.beta_q")
    )
    evaluation_gate = _independent_evaluation_gate(runtime_config)
    print(_normalization_config_line(runtime_config), flush=True)

    first_seed = int(pool_info["episode_seeds"][0])
    # 环境初始化会打印冗长的逐节点归一化详情；控制台只保留上方摘要行。
    with contextlib.redirect_stdout(io.StringIO()):
        first_environment = ExclusiveDQNEnvironment.from_formal_training_episode(
            PROJECT_ROOT,
            episode_seed=first_seed,
            origin_start=0,
            max_steps=pool_contract.FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS,
        )
    state, candidates = first_environment.observe()
    agent = DoubleDQNAgent(
        len(state),
        len(candidates[0].features),
        learning_rate=float(learning_rate),
        gamma=1.0,
        seed=int(seed),
    )
    replay = ReplayBuffer(int(replay_capacity), seed=int(seed))
    # The formal exploration schedule spans the full planned finite-horizon
    # training run.  Keep the schedule global across Episodes and resume from
    # the checkpointed global_step rather than reaching epsilon_min after only
    # a small fraction of the 300-Episode pool.
    formal_training_decision_steps = (
        len(pool_info["episode_seeds"])
        * pool_contract.FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS
    )
    trainer_config = TrainerConfig(
        epsilon_decay_steps=formal_training_decision_steps
    )
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
        validation_pool_info,
        trainer_config,
        seed=seed,
        replay_capacity=replay_capacity,
        agent=agent,
        search_budget=search_budget,
    )
    # Agent dimensions and the formal candidate limits are now captured as
    # scalars. Keep the first environment only when it will serve Episode 0.
    del state, candidates

    if resume_checkpoint is None:
        run_dir = _new_run_directory(
            output_dir, run_name_prefix=run_name_prefix
        )
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
                "arrival_horizon_slots": pool_contract.FORMAL_TRAINING_ARRIVAL_HORIZON_SLOTS,
                "execution_horizon_slots": pool_contract.FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS,
                "terminal_origin": pool_contract.FORMAL_TRAINING_ARRIVAL_HORIZON_SLOTS - 1,
                "terminal_execution_slot": pool_contract.FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS - 1,
                "manifest_sha256": pool_info["manifest_sha256"],
                "integrity_baseline_sha256": pool_info[
                    "integrity_baseline_sha256"
                ],
                "task_contract_sha256": pool_info["task_data_contract_sha256"],
                "service_catalog_sha256": pool_info["service_catalog_sha256"],
            },
            "training_episode_order": pool_info["episode_seeds"],
            "validation_pool": validation_pool_info["validation_pool_identity"],
            "validation_interval_episodes": VALIDATION_INTERVAL_EPISODES,
            "validation_selection_metric": "mean_episode_system_utility",
            "validation_history": [],
            "best_checkpoint": None,
            "checkpoint_contract": {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "latest_each_completed_episode": True,
                "milestone_episode_counts": list(MILESTONE_EPISODES),
                "best_updated_only_on_strict_validation_improvement": True,
            },
            "evaluation_status": evaluation_gate["status"],
            "evaluation_blockers": evaluation_gate["blockers"],
            "evaluation_gate": evaluation_gate,
            "training_status": "PREPARED",
            "episode_limit_this_invocation": max_episodes,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(run_dir / "run_manifest.json", manifest, atomic=True)
        episode_order = list(pool_info["episode_seeds"])
        next_episode_index = 0
        global_step = 0
        checkpoint_episode_rows: list[dict[str, Any]] = []
        training_metadata: dict[str, Any] = {
            "validation_history": [],
            "best_checkpoint": None,
        }
    else:
        checkpoint_path = resume_checkpoint.resolve(strict=True)
        run_dir = checkpoint_path.parent.parent
        expected_latest_path = (run_dir / "checkpoints" / "latest_checkpoint.pt").resolve()
        if checkpoint_path != expected_latest_path:
            raise ValueError(
                "恢复训练必须从本 Run 的 latest_checkpoint.pt 继续；"
                "使用里程碑或 best 会回退训练进度并覆盖更晚的 latest。"
            )
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("恢复训练缺少原始 run_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _validate_resume_identity(identity, manifest, payload)
        if payload.get("checkpoint_role") != "latest":
            raise ValueError("恢复路径中的 Checkpoint 角色不是 latest")
        rng_before_restore = _capture_global_rng_state()
        try:
            staged_agent = copy.deepcopy(agent)
            staged_replay = ReplayBuffer(int(replay_capacity), seed=int(seed))
            staged_agent.load_state_dict(payload["agent_state"])
            staged_replay.load_state_dict(payload["replay_state"])
            staged_replay.validate_rebuild_contexts()

            restored_episode_order = [
                int(value) for value in payload["episode_order"]
            ]
            if restored_episode_order != list(pool_info["episode_seeds"]):
                raise ValueError("Checkpoint Episode 顺序与正式 Training Pool 不匹配")
            restored_next_episode_index = int(payload["next_episode_index"])
            restored_global_step = int(payload["global_step"])
            if int(payload.get("completed_episode_count", -1)) != restored_next_episode_index:
                raise ValueError("Checkpoint completed Episode count differs from its resume index")
            if not 0 <= restored_next_episode_index <= len(pool_info["episode_seeds"]):
                raise ValueError("Checkpoint Episode 进度超出正式训练计划")
            restored_episode_rows = [
                dict(row) for row in payload.get("completed_episode_rows", [])
            ]
            if len(restored_episode_rows) != restored_next_episode_index:
                raise ValueError("Checkpoint Episode 汇总数与完成进度不一致")
            expected_indices = list(range(restored_next_episode_index))
            actual_indices = [
                int(row.get("episode_index", -1)) for row in restored_episode_rows
            ]
            actual_seeds = [
                int(row.get("episode_seed", -1)) for row in restored_episode_rows
            ]
            if (
                actual_indices != expected_indices
                or actual_seeds
                != restored_episode_order[:restored_next_episode_index]
            ):
                raise ValueError("Checkpoint Episode 汇总与正式训练顺序不一致")
            expected_global_step = (
                restored_next_episode_index
                * pool_contract.FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS
            )
            if restored_global_step != expected_global_step:
                raise ValueError("Checkpoint Slot 数与完整 Episode 进度不一致")
            if int(payload.get("epsilon_progress_global_step", -1)) != restored_global_step:
                raise ValueError("Checkpoint 探索率进度与全局 Slot 数不一致")
            if int(payload.get("gradient_update_count", -1)) != int(staged_agent.gradient_steps):
                raise ValueError("Checkpoint 梯度更新数与 Agent 状态不一致")
            if int(staged_replay.write_count) != restored_global_step:
                raise ValueError("Checkpoint Replay 写入进度与全局 Slot 数不一致")
            raw_training_metadata = payload.get("training_metadata")
            if not isinstance(raw_training_metadata, Mapping):
                raise ValueError("Checkpoint 缺少训练/Validation 进度元数据")
            training_metadata = dict(raw_training_metadata)
            if not isinstance(training_metadata.get("validation_history"), list):
                raise ValueError("Checkpoint Validation 历史格式非法")
        except BaseException:
            _restore_global_rng_state(rng_before_restore)
            raise

        # 仅在网络、回放、候选合同和进度全部校验通过后替换本地对象。
        agent = staged_agent
        replay = staged_replay
        episode_order = restored_episode_order
        next_episode_index = restored_next_episode_index
        # A resumed run constructs its next Episode from the checkpoint order;
        # the bootstrap environment above is no longer needed.
        if next_episode_index > 0:
            first_environment = None
        global_step = restored_global_step
        checkpoint_episode_rows = restored_episode_rows

        # A crash after the atomic best write but before the subsequent latest
        # write must not lose the validated best selection. Reconcile that
        # strictly identity-matched best snapshot into the latest metadata.
        best_path = run_dir / "checkpoints" / "best_checkpoint.pt"
        if best_path.is_file():
            best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
            _validate_resume_identity(identity, manifest, best_payload)
            if best_payload.get("checkpoint_role") != "best":
                raise ValueError("best_checkpoint.pt has an invalid checkpoint role")
            best_metadata = best_payload.get("training_metadata", {}).get(
                "best_checkpoint"
            )
            if not isinstance(best_metadata, Mapping):
                raise ValueError("best Checkpoint lacks its Validation selection identity")
            current_best = training_metadata.get("best_checkpoint")
            current_count = (
                -1 if not isinstance(current_best, Mapping)
                else int(current_best.get("episode_count", -1))
            )
            best_count = int(best_metadata.get("episode_count", -1))
            if best_count > current_count:
                training_metadata["best_checkpoint"] = dict(best_metadata)
            elif best_count == current_count and not _checkpoint_values_equal(
                dict(current_best), dict(best_metadata)
            ):
                raise ValueError("latest and best Checkpoints disagree on the selected model")
            merged_history: dict[int, dict[str, Any]] = {}
            for history_source in (
                training_metadata.get("validation_history", []),
                best_payload.get("training_metadata", {}).get("validation_history", []),
            ):
                if not isinstance(history_source, list):
                    raise ValueError("best/latest Checkpoint Validation history is malformed")
                for validation_record in history_source:
                    count = int(validation_record.get("episode_count", -1))
                    previous = merged_history.get(count)
                    if previous is not None and previous != validation_record:
                        raise ValueError(
                            "best/latest Checkpoints contain conflicting Validation records"
                        )
                    merged_history[count] = dict(validation_record)
            training_metadata["validation_history"] = [
                merged_history[count] for count in sorted(merged_history)
            ]
            training_metadata["best_checkpoint_file_sha256"] = _sha256(best_path)

        manifest["evaluation_status"] = evaluation_gate["status"]
        manifest["evaluation_blockers"] = evaluation_gate["blockers"]
        manifest["evaluation_gate"] = evaluation_gate
        manifest["validation_history"] = list(
            training_metadata.get("validation_history", [])
        )
        restored_best = training_metadata.get("best_checkpoint")
        manifest["best_checkpoint"] = (
            None
            if restored_best is None
            else {
                **dict(restored_best),
                "file_sha256": training_metadata.get(
                    "best_checkpoint_file_sha256"
                ),
            }
        )

    manifest["episode_limit_this_invocation"] = max_episodes

    if not np.all(np.isfinite(state)):
        raise FloatingPointError("正式训练初始状态包含非有限值")
    _write_json(
        run_dir / "evaluation_status.json",
        _evaluation_status_payload(evaluation_gate),
        atomic=True,
    )
    _write_json(run_dir / "run_manifest.json", manifest, atomic=True)

    episode_metrics_path = run_dir / "episode_metrics.jsonl"
    episode_rows = checkpoint_episode_rows
    if _read_episode_metrics(episode_metrics_path) != episode_rows:
        # Checkpoint 是完成进度的提交点；恢复时修复可能落后的派生汇总文件。
        _write_episode_metrics_atomic(episode_metrics_path, episode_rows)
    total_slots = len(episode_order) * pool_contract.FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS
    if trainer_config.epsilon_decay_steps != total_slots:
        raise RuntimeError(
            "Formal epsilon schedule must span the complete training decision horizon: "
            f"decay_steps={trainer_config.epsilon_decay_steps}, total_slots={total_slots}"
        )
    previous_summary_path = run_dir / "run_summary.json"
    previous_training_elapsed_seconds = sum(
        max(0.0, float(row.get("episode_wall_time_s", 0.0)))
        for row in episode_rows
        if isinstance(row.get("episode_wall_time_s", 0.0), (int, float))
        and math.isfinite(float(row.get("episode_wall_time_s", 0.0)))
    )
    if resume_checkpoint is not None and previous_summary_path.is_file():
        previous_summary = json.loads(
            previous_summary_path.read_text(encoding="utf-8")
        )
        recorded_elapsed = previous_summary.get("training_elapsed_seconds", 0.0)
        if isinstance(recorded_elapsed, (int, float)) and math.isfinite(
            float(recorded_elapsed)
        ):
            previous_training_elapsed_seconds = max(
                previous_training_elapsed_seconds, float(recorded_elapsed)
            )
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
    manifest["validation_history"] = list(
        training_metadata.get("validation_history", [])
    )
    manifest["best_checkpoint"] = training_metadata.get("best_checkpoint")
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
        learning_rate=learning_rate,
        beta_ca=beta_ca,
        beta_q=beta_q,
    )

    if next_episode_index in MILESTONE_EPISODES:
        boundary, last_checkpoint_bytes, milestone_bytes = (
            _finish_validation_checkpoint_boundary(
                run_dir,
                completed_episode_count=next_episode_index,
                agent=agent,
                replay=replay,
                identity=identity,
                episode_order=episode_order,
                episode_rows=episode_rows,
                global_step=global_step,
                training_metadata=training_metadata,
                validation_pool_info=validation_pool_info,
            )
        )
        manifest["validation_history"] = list(
            training_metadata.get("validation_history", [])
        )
        manifest["best_checkpoint"] = boundary["best_checkpoint"]
        manifest["last_checkpoint"] = str(checkpoint_path)
        manifest["last_checkpoint_bytes"] = last_checkpoint_bytes
        manifest["latest_milestone_checkpoint"] = str(
            run_dir
            / "checkpoints"
            / f"milestone_episode_{next_episode_index:03d}.pt"
        )
        manifest["latest_milestone_checkpoint_bytes"] = milestone_bytes
        _write_json(run_dir / "run_manifest.json", manifest, atomic=True)
        _write_run_summary(
            run_dir / "run_summary.json",
            status="RUNNING",
            episode_rows=episode_rows,
            global_step=global_step,
            gradient_steps=checkpointed_gradient_steps,
            last_checkpoint=str(checkpoint_path),
            last_checkpoint_bytes=last_checkpoint_bytes,
            elapsed_seconds=0.0,
            total_slots=total_slots,
            learning_rate=learning_rate,
            beta_ca=beta_ca,
            beta_q=beta_q,
        )

    plot_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="online-joint-dqn-plot"
    )
    plot_jobs: list[tuple[int, Future[dict[str, Any]]]] = []
    plot_result_history: list[dict[str, Any]] = []
    plot_result_latest: dict[str, Any] | None = None

    def submit_plot_snapshot(episode_count: int) -> None:
        nonlocal plot_result_latest
        def generate_snapshot() -> dict[str, Any]:
            from online_joint_dqn.plot_training import generate_training_plots

            return generate_training_plots(
                run_dir,
                snapshot_episode=int(episode_count),
            )

        try:
            future = plot_executor.submit(generate_snapshot)
        except Exception as exc:
            result = {
                "status": "FAILED",
                "snapshot_episode": int(episode_count),
                "generated": [],
                "skipped": {},
                "failed": {
                    "plot_job_submit": f"{type(exc).__name__}: {exc}"
                },
                "run_dir": str(run_dir),
                "plot_dir": str(run_dir / "plots"),
            }
            plot_result_latest = result
            plot_result_history.append(
                {
                    "snapshot_episode": int(episode_count),
                    "status": "FAILED",
                    "generated_count": 0,
                    "skipped": {},
                    "failed": dict(result["failed"]),
                    "data_rows": {},
                }
            )
            print(
                f"[PLOT][ERROR] EP {episode_count:04d} submit failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return
        plot_jobs.append((int(episode_count), future))
        print(
            f"[PLOT] EP {episode_count:04d} queued; training continues.",
            flush=True,
        )

    def collect_plot_jobs(*, wait_for_all: bool) -> None:
        nonlocal plot_result_latest
        pending: list[tuple[int, Future[dict[str, Any]]]] = []
        for episode_count, future in plot_jobs:
            if not wait_for_all and not future.done():
                pending.append((episode_count, future))
                continue
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "FAILED",
                    "snapshot_episode": episode_count,
                    "generated": [],
                    "skipped": {},
                    "failed": {
                        "plot_job": f"{type(exc).__name__}: {exc}"
                    },
                    "run_dir": str(run_dir),
                    "plot_dir": str(run_dir / "plots"),
                }
            plot_result_latest = result
            plot_result_history.append(
                {
                    "snapshot_episode": episode_count,
                    "status": result.get("status", "UNKNOWN"),
                    "generated_count": len(result.get("generated", [])),
                    "skipped": dict(result.get("skipped", {})),
                    "failed": dict(result.get("failed", {})),
                    "data_rows": dict(result.get("data_rows", {})),
                }
            )
            print(
                f"[PLOT] EP {episode_count:04d} | "
                f"{result.get('status', 'UNKNOWN')} | "
                f"Generated {len(result.get('generated', []))} | "
                f"Skipped {len(result.get('skipped', {}))} | "
                f"Failed {len(result.get('failed', {}))} | "
                f"Path {run_dir / 'plots'}",
                flush=True,
            )
            for plot_name, error in result.get("failed", {}).items():
                print(
                    f"[PLOT][ERROR] EP {episode_count:04d} {plot_name}: {error}",
                    flush=True,
                )
        plot_jobs[:] = pending

    def persist_plot_results() -> None:
        if plot_result_latest is None:
            return
        try:
            summary_path = run_dir / "run_summary.json"
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_data["plot_generation"] = plot_result_latest
            summary_data["plot_generation_history"] = list(plot_result_history)
            _write_json(summary_path, summary_data, atomic=True)
        except Exception as exc:
            print(
                f"[PLOT][ERROR] run_summary update failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    stop_requested = False
    previous_signal_handlers: dict[int, Any] = {}

    def request_episode_boundary_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print("[DQN] Stop requested; will stop after the current Episode.", flush=True)

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
            episode_stop_index = len(episode_order)
            if max_episodes is not None:
                episode_stop_index = min(
                    episode_stop_index, next_episode_index + max_episodes
                )
            for episode_index in range(next_episode_index, episode_stop_index):
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
                        max_steps=pool_contract.FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS,
                    )
                if (
                    environment.encoder.state_dim != agent.state_dim
                    or environment.encoder.candidate_dim != agent.candidate_dim
                    or environment.search_budget.max_cache_candidates
                    != search_budget["max_cache_candidates_including_keep"]
                    or environment.search_budget.max_uav_candidates_per_uav
                    != search_budget["max_uav_candidates_per_uav_including_stay"]
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

                def write_update(row: dict[str, Any]) -> None:
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
                if metrics["step_count"] != pool_contract.FORMAL_TRAINING_EXECUTION_HORIZON_SLOTS:
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
                    checkpoint_role="latest",
                    training_metadata=training_metadata,
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
                boundary: dict[str, Any] | None = None
                milestone_bytes: int | None = None
                if next_episode_index in MILESTONE_EPISODES:
                    boundary, last_checkpoint_bytes, milestone_bytes = (
                        _finish_validation_checkpoint_boundary(
                            run_dir,
                            completed_episode_count=next_episode_index,
                            agent=agent,
                            replay=replay,
                            identity=identity,
                            episode_order=episode_order,
                            episode_rows=episode_rows,
                            global_step=global_step,
                            training_metadata=training_metadata,
                            validation_pool_info=validation_pool_info,
                        )
                    )
                manifest["training_status"] = "RUNNING"
                manifest["completed_episodes"] = next_episode_index
                manifest["completed_slots"] = global_step
                manifest["gradient_update_count"] = checkpointed_gradient_steps
                manifest["last_checkpoint"] = str(checkpoint_path)
                manifest["last_checkpoint_bytes"] = last_checkpoint_bytes
                manifest["last_completed_episode_seed"] = episode_seed
                manifest["validation_history"] = list(
                    training_metadata.get("validation_history", [])
                )
                best_manifest = training_metadata.get("best_checkpoint")
                manifest["best_checkpoint"] = (
                    None
                    if best_manifest is None
                    else {
                        **dict(best_manifest),
                        "file_sha256": training_metadata.get(
                            "best_checkpoint_file_sha256"
                        ),
                    }
                )
                if next_episode_index in MILESTONE_EPISODES:
                    manifest["latest_milestone_checkpoint"] = str(
                        run_dir
                        / "checkpoints"
                        / f"milestone_episode_{next_episode_index:03d}.pt"
                    )
                    manifest["latest_milestone_checkpoint_bytes"] = milestone_bytes
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
                    learning_rate=learning_rate,
                    beta_ca=beta_ca,
                    beta_q=beta_q,
                )
                elapsed = previous_training_elapsed_seconds + (
                    time.perf_counter() - run_start
                )
                rate = global_step / elapsed if elapsed > 0 else 0.0
                remaining = max(0, total_slots - global_step)
                eta = remaining / rate if rate > 0 else None
                episode_return = float(metrics["episode_reward"])
                mean_utility = float(
                    metrics["episode_summary"]["mean_system_utility_per_slot"]
                )
                print(
                    _format_episode_progress(
                        episode_index=next_episode_index,
                        total_episodes=len(episode_order),
                        episode_seed=episode_seed,
                        episode_return=episode_return,
                        mean_utility=mean_utility,
                        completed_slots=global_step,
                        total_slots=total_slots,
                        gradient_updates=agent.gradient_steps,
                        speed_slots_per_second=rate,
                        eta_seconds=eta,
                        cache_updates=metrics["episode_summary"].get(
                            "actual_cache_update_count"
                        ),
                        cache_nodes=metrics["episode_summary"].get(
                            "actual_cache_changed_node_count"
                        ),
                        uav_moves=metrics["episode_summary"].get(
                            "uav_movement_count"
                        ),
                        uav_move_slots=metrics["episode_summary"].get(
                            "uav_movement_slot_count"
                        ),
                        uav_total_distance_m=metrics["episode_summary"].get(
                            "uav_total_distance"
                        ),
                        top_execution_node=(
                            str(
                                metrics["episode_summary"].get(
                                    "most_common_actual_execution_node"
                                )
                            )
                            if metrics["episode_summary"].get(
                                "actual_execution_node_data_available"
                            )
                            and metrics["episode_summary"].get(
                                "most_common_actual_execution_node"
                            )
                            else "N/A"
                        ),
                    ),
                    flush=True,
                )
                if _should_refresh_training_plots(next_episode_index):
                    submit_plot_snapshot(next_episode_index)
                collect_plot_jobs(wait_for_all=False)
                # train_episode returns per-Slot diagnostic rows in metrics;
                # the compact Episode row is already committed above.
                metrics = None
                # Release the just-finished full Episode before constructing
                # the next one; only its compact Replay rebuild adapter remains.
                if episode_index == 0:
                    first_environment = None
                environment = None
                if stop_requested:
                    break
        status = (
            "STOPPED_AT_EPISODE_BOUNDARY"
            if next_episode_index < len(episode_order)
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
            learning_rate=learning_rate,
            beta_ca=beta_ca,
            beta_q=beta_q,
        )
        collect_plot_jobs(wait_for_all=False)
        plot_executor.shutdown(wait=False, cancel_futures=False)
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
        learning_rate=learning_rate,
        beta_ca=beta_ca,
        beta_q=beta_q,
    )
    if status == "FORMAL_TRAINING_COMPLETE":
        submit_plot_snapshot(next_episode_index)
    collect_plot_jobs(wait_for_all=True)
    persist_plot_results()
    plot_executor.shutdown(wait=False, cancel_futures=False)
    print(
        f"[DQN] {status} | Episodes {next_episode_index}/{len(episode_order)} "
        f"| Slots {global_step}/{total_slots}",
        flush=True,
    )
    return 130 if stop_requested else 0


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


def main(
    *,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    run_name_prefix: str = DEFAULT_RUN_NAME_PREFIX,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--replay-capacity", type=int, default=DEFAULT_REPLAY_CAPACITY)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="本次运行最多完成的 Episode 数；在 Episode 边界安全停止，Checkpoint 仍按完整训练池记录",
    )
    args = parser.parse_args()
    if int(args.replay_capacity) <= 0:
        parser.error("--replay-capacity 必须为正整数")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes 必须为正整数")
    return run_training(
        output_dir=args.output_dir,
        resume_checkpoint=args.resume_checkpoint,
        seed=args.seed,
        replay_capacity=args.replay_capacity,
        learning_rate=learning_rate,
        run_name_prefix=run_name_prefix,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
