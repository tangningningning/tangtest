from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "configs" / "proposed" / "system_config.toml"
EPISODE_POOL = PROJECT_ROOT / "environment_data" / "proposed_joint_training"
RUN_DIR = PROJECT_ROOT / "runs" / "formal_unified_cache_cardinality_corrected_gate"
REPORT_PATH = PROJECT_ROOT / "reports" / "unified_cache_cardinality_corrected_gate.json"
V14B_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "formal_v14b_actor_lr_1e_minus_3"
    / "checkpoints"
    / "latest_checkpoint.pt"
)
BASE_SEED = 27005
EPISODE_SEED_START = 20001
TARGET_EPISODES = 15
ARCHITECTURE_VERSION = 25
CHECKPOINT_VERSION = 49
EXPECTED_V14B_SHA256 = "82a835d7e5be6325566f14ffa65afeb388d7e3ba47e4ddce487a14e6b7d1e55c"
FORMAL_COMMAND = (
    "conda activate adaptive_cache_mec; "
    "python scripts/training/run_unified_cache_cardinality_corrected_gate.py"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def _contains_nonfinite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, bool, np.bool_)):
        return False
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "fc" and not bool(np.isfinite(value).all())
    if isinstance(value, (int, float, np.number)):
        return not math.isfinite(float(value))
    return False


def _version_high_watermarks() -> tuple[int, int]:
    architecture_versions: list[int] = []
    checkpoint_versions: list[int] = []
    for manifest_path in (PROJECT_ROOT / "runs").rglob("run_manifest.json"):
        if manifest_path.parent.resolve() == RUN_DIR.resolve():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key, target in (
            ("architecture_version", architecture_versions),
            ("checkpoint_version", checkpoint_versions),
        ):
            try:
                target.append(int(manifest[key]))
            except (KeyError, TypeError, ValueError):
                pass
    return max(architecture_versions, default=-1), max(checkpoint_versions, default=-1)


def _resume_prefix() -> tuple[bool, int]:
    if not RUN_DIR.exists() or not any(RUN_DIR.iterdir()):
        return False, 0
    manifest_path = RUN_DIR / "run_manifest.json"
    summary_path = RUN_DIR / "run_summary.json"
    checkpoint_path = RUN_DIR / "checkpoints" / "latest_checkpoint.pt"
    metrics_path = RUN_DIR / "episode_metrics.jsonl"
    updates_path = RUN_DIR / "update_metrics.jsonl"
    required = (manifest_path, summary_path, checkpoint_path, metrics_path, updates_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileExistsError(f"Existing unified run lacks a resumable checkpoint prefix: {missing}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "algorithm_id": "UNIFIED_CACHE_ACTION_CARDINALITY_CORRECTED_V1",
        "architecture_version": ARCHITECTURE_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "target_episodes": TARGET_EPISODES,
        "episode_seed_start": EPISODE_SEED_START,
        "seed": BASE_SEED,
        "unified_cache_action": True,
        "cardinality_corrected_cache_action": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise FileExistsError("Existing run manifest does not match the unified action protocol.")
    episodes = [
        item for item in _read_jsonl(metrics_path) if item.get("record_type") == "episode"
    ]
    updates = [
        item for item in _read_jsonl(updates_path) if item.get("record_type") == "update"
    ]
    completed = len(episodes)
    if (
        completed > TARGET_EPISODES
        or [int(item.get("episode", -1)) for item in episodes] != list(range(1, completed + 1))
        or [int(item.get("episode", -1)) for item in updates] != list(range(1, completed + 1))
        or [int(item.get("episode_seed", -1)) for item in episodes]
        != list(range(EPISODE_SEED_START, EPISODE_SEED_START + completed))
        or int(summary.get("completed_episodes", -1)) != completed
        or int(summary.get("target_episodes", -1)) != TARGET_EPISODES
    ):
        raise FileExistsError("Existing unified run artifacts do not form a consistent episode prefix.")
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        int(checkpoint.get("architecture_version", -1)) != ARCHITECTURE_VERSION
        or int(checkpoint.get("checkpoint_version", -1)) != CHECKPOINT_VERSION
        or checkpoint.get("unified_cache_action") is not True
        or checkpoint.get("cardinality_corrected_cache_action") is not True
        or int(checkpoint.get("training_counters", {}).get("completed_episodes", -1)) != completed
    ):
        raise FileExistsError("Latest unified checkpoint does not match the saved episode prefix.")
    if completed == TARGET_EPISODES or summary.get("early_stop_reason"):
        return True, completed
    return True, completed


def _assert_protocol_inputs() -> dict[str, Any]:
    from ppo_simple import trainer as training

    if not CONFIG_PATH.is_file() or not EPISODE_POOL.is_dir():
        raise FileNotFoundError("Canonical config or proposed training realization pool is missing.")
    if not V14B_CHECKPOINT.is_file():
        raise FileNotFoundError(f"V14b episode-300 checkpoint is missing: {V14B_CHECKPOINT}")
    checkpoint_sha256 = hashlib.sha256(V14B_CHECKPOINT.read_bytes()).hexdigest()
    if checkpoint_sha256 != EXPECTED_V14B_SHA256:
        raise ValueError("V14b episode-300 checkpoint SHA256 does not match the formal artifact.")
    episode_paths = [
        EPISODE_POOL / f"episode_{seed}"
        for seed in range(EPISODE_SEED_START, EPISODE_SEED_START + TARGET_EPISODES)
    ]
    missing = [str(path) for path in episode_paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Requested training realizations are missing: {missing}")
    highest_architecture, highest_checkpoint = _version_high_watermarks()
    if ARCHITECTURE_VERSION != highest_architecture + 1:
        raise RuntimeError(
            f"Architecture version {ARCHITECTURE_VERSION} is not next after {highest_architecture}."
        )
    if CHECKPOINT_VERSION != highest_checkpoint + 1:
        raise RuntimeError(
            f"Checkpoint version {CHECKPOINT_VERSION} is not next after {highest_checkpoint}."
        )
    resume, resume_episode_count = _resume_prefix()
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "training_seed_start": EPISODE_SEED_START,
        "training_seed_end": EPISODE_SEED_START + TARGET_EPISODES - 1,
        "training_realization_count": len(episode_paths),
        "highest_architecture_version_before_run": highest_architecture,
        "highest_checkpoint_version_before_run": highest_checkpoint,
        "architecture_version_allocated": ARCHITECTURE_VERSION,
        "checkpoint_version_allocated": CHECKPOINT_VERSION,
        "validation_or_test_realizations_used": False,
        "teacher_branches_planned": 0,
        "recurrent_or_rl_stages_other_than_15_episode_macro_micro_training": False,
        "run_dir_is_fresh": not resume,
        "resume_existing_prefix": resume,
        "resume_from_completed_episodes": resume_episode_count,
        "v14b_checkpoint_episode": 300,
        "v14b_optimizer_state_transplanted": False,
        "v14b_candidate_selection_head_warmstart_used": False,
    }


def _implementation_preflight() -> dict[str, Any]:
    import torch

    from decision import cache_configuration
    from common import system_model
    from ppo_simple.interface import load_environment_context, aligned_train_steps
    from ppo_simple.policy import DualTimescaleActorCritic, _masked_categorical
    from ppo_simple.primitive_control import PrimitiveTensorSpec
    from ppo_simple.rollout import _cache_candidates, cache_update_mask
    from ppo_simple.trainer import PPOTrainer, _seed_everything

    realization = EPISODE_POOL / f"episode_{EPISODE_SEED_START}"
    context = load_environment_context(str(CONFIG_PATH), realization_dir=realization)
    first_step = aligned_train_steps(context)[0]
    current_cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
    candidates, features = _cache_candidates(
        current_cache,
        context.initial_uav,
        first_step,
        context,
        PrimitiveTensorSpec.create(context),
    )
    candidate_mask = np.ones(len(candidates), dtype=bool)
    actual_legal_update_count = int(np.count_nonzero(candidate_mask[1:]))
    candidate0_exact_keep = bool(
        candidate_mask[0] and np.array_equal(candidates[0], current_cache)
    )
    if not candidate0_exact_keep:
        raise AssertionError("Candidate 0 is not the exact current Cache configuration.")

    candidate0_change_mask = cache_update_mask(current_cache, candidates[0])
    if int(np.count_nonzero(candidate0_change_mask)) != 0:
        raise AssertionError("Candidate 0 produced a Cache change mask.")
    zero_trigger = np.zeros(current_cache.shape[0], dtype=np.int8)
    cache_cost, loading_energy = cache_configuration.cache_update_cost(
        current_cache,
        candidates[0],
        zero_trigger,
        context.service_catalog["service_sizes_mb"],
        context.cache_nodes,
        context.cache_cfg,
    )
    detailed_cost = system_model.cache_update_cost(
        current_cache,
        candidates[0],
        zero_trigger,
        context.service_catalog["service_sizes_mb"],
        context.cache_nodes["eta_ca"],
        context.cache_nodes["xi_ca"],
        context.cache_cfg.lambda_cache,
        cache_cost_weight=context.cache_cfg.cache_cost_weight,
    )
    if (
        cache_cost != 0.0
        or float(np.sum(loading_energy)) != 0.0
        or float(np.sum(detailed_cost["loaded_mb"])) != 0.0
    ):
        raise AssertionError("KEEP candidate 0 incurred cache reconfiguration or loading cost.")

    _seed_everything(BASE_SEED)
    trainer = PPOTrainer(
        context,
        unified_cache_action=True,
        cardinality_corrected_cache_action=True,
    )
    trainer.load_v14b_unified_checkpoint(
        V14B_CHECKPOINT, expected_sha256=EXPECTED_V14B_SHA256
    )
    policy = trainer.policy
    trigger_forward_calls = 0

    def _count_trigger_forward(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal trigger_forward_calls
        trigger_forward_calls += 1

    hook = policy.macro_cache_trigger_actor.register_forward_hook(_count_trigger_forward)
    try:
        observation = np.zeros(context.observation_spec.observation_dim, dtype=np.float32)
        action = policy.sample_macro(
            observation, features, candidate_mask, deterministic=False
        )
        components = policy.evaluate_macro_components(
            observation,
            features,
            candidate_mask,
            action.cache_index,
            np.ones(len(policy.duration_candidates), dtype=bool),
            action.duration_index,
        )
        probability_sum = float(np.sum(action.cache_probabilities))
        logprob_error = float(
            torch.max(torch.abs(components.joint_log_prob - action.log_prob)).detach().cpu()
        )
        duration_probabilities = policy.duration_probabilities_for_candidates(
            observation, features
        )
        duration_probability_error = abs(
            float(action.duration_log_prob.exp().detach().cpu())
            - float(duration_probabilities[action.cache_index, action.duration_index].cpu())
        )
        equal_logit_checks: dict[str, dict[str, float]] = {}
        cardinalities = sorted({1, 5, 20, actual_legal_update_count})
        max_cardinality = max(cardinalities)
        for update_count in cardinalities:
            logits = torch.zeros(max_cardinality + 1, dtype=torch.float32)
            legal_mask = torch.zeros(max_cardinality + 1, dtype=torch.bool)
            legal_mask[: update_count + 1] = True
            corrected_logits = policy._cardinality_corrected_cache_logits(
                logits, legal_mask
            )
            distribution = _masked_categorical(corrected_logits, legal_mask)
            probabilities = distribution.probs
            keep_probability = float(probabilities[0].cpu())
            update_probability = float(probabilities[1:].sum().cpu())
            per_update_probability = float(probabilities[1].cpu())
            sum_error = abs(float(probabilities.sum().cpu()) - 1.0)
            illegal_probability = float(probabilities[~legal_mask].max().cpu()) if bool(
                torch.any(~legal_mask).item()
            ) else 0.0
            if (
                abs(keep_probability - 0.5) > 1.0e-6
                or abs(update_probability - 0.5) > 1.0e-6
                or abs(per_update_probability - 0.5 / update_count) > 1.0e-6
                or sum_error > 1.0e-6
                or illegal_probability != 0.0
                or not bool(torch.isfinite(probabilities).all().item())
            ):
                raise AssertionError(
                    f"Equal-logit cardinality correction failed at K={update_count}."
                )
            equal_logit_checks[str(update_count)] = {
                "keep_probability": keep_probability,
                "update_group_probability": update_probability,
                "per_update_probability": per_update_probability,
                "probability_sum_error": sum_error,
                "illegal_probability_max": illegal_probability,
            }
        zero_update_logits = torch.zeros(2, dtype=torch.float32)
        zero_update_mask = torch.tensor([True, False], dtype=torch.bool)
        zero_update_probabilities = _masked_categorical(
            policy._cardinality_corrected_cache_logits(
                zero_update_logits, zero_update_mask
            ),
            zero_update_mask,
        ).probs
        if not torch.equal(zero_update_probabilities, torch.tensor([1.0, 0.0])):
            raise AssertionError("K_s=0 must yield KEEP probability 1 and illegal UPDATE 0.")
    finally:
        hook.remove()
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise AssertionError("Unified candidate probabilities do not sum to one.")
    if logprob_error > 1.0e-6:
        raise AssertionError("Stored rollout log-probability does not match PPO evaluation.")
    if duration_probability_error > 1.0e-6:
        raise AssertionError("Duration is not conditioned on the selected Cache candidate.")
    if trigger_forward_calls != 0:
        raise AssertionError("Trigger Actor forward was called in unified action preflight.")
    if int(trainer.training_counters["macro_cache_trigger_optimizer_steps"]) != 0:
        raise AssertionError("Trigger optimizer has a nonzero step count before training.")
    if not bool(torch.isfinite(action.log_prob).item()) or not bool(torch.isfinite(action.entropy).item()):
        raise FloatingPointError("Unified Macro sample contains NaN/Inf.")

    return {
        "candidate0_is_exact_keep": candidate0_exact_keep,
        "legal_update_candidate_count": actual_legal_update_count,
        "candidate0_legal": bool(candidate_mask[0]),
        "candidate0_cache_change_count": int(np.count_nonzero(candidate0_change_mask)),
        "candidate0_cache_cost": float(cache_cost),
        "candidate0_loaded_cache_mb": float(np.sum(detailed_cost["loaded_mb"])),
        "candidate0_loading_energy": float(np.sum(loading_energy)),
        "unified_candidate_probability_sum": probability_sum,
        "sampled_candidate_id": int(action.cache_index),
        "sampled_requested_duration": int(action.requested_duration),
        "sampled_joint_logprob_ppo_stored_error": logprob_error,
        "raw_flat_candidate0_probability": float(action.raw_cache_probabilities[0]),
        "corrected_candidate0_probability": float(action.cache_probabilities[0]),
        "corrected_update_group_probability": float(action.cache_probabilities[1:].sum()),
        "equal_logit_cardinality_checks": equal_logit_checks,
        "equal_logit_k0_keep_probability": float(zero_update_probabilities[0].cpu()),
        "equal_logit_k0_illegal_update_probability": float(zero_update_probabilities[1].cpu()),
        "duration_conditioning_probability_error": duration_probability_error,
        "trigger_forward_calls": trigger_forward_calls,
        "trigger_optimizer_steps": int(
            trainer.training_counters["macro_cache_trigger_optimizer_steps"]
        ),
        "candidate_selection_head_warmstart_used": False,
        "candidate_shared_representation_loaded_for_duration": True,
        "duration_candidates": list(policy.duration_candidates),
        "finite_sample": True,
    }


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else None


def _flatten_state_diagnostics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for record in records
        for item in record.get("macro_candidate_state_diagnostics", [])
    ]


def _duration_rates(records: list[dict[str, Any]]) -> dict[str, float | None]:
    counts = {"1": 0, "3": 0, "5": 0}
    total = 0
    for record in records:
        values = record.get("requested_duration_counts", {})
        for key in counts:
            count = int(values.get(key, 0))
            counts[key] += count
            total += count
    return {
        key: (float(value / total) if total else None) for key, value in counts.items()
    }


def _build_report(
    *,
    preflight: dict[str, Any],
    preflight_check: dict[str, Any],
    run_result: dict[str, Any],
) -> dict[str, Any]:
    episode_records = _read_jsonl(RUN_DIR / "episode_metrics.jsonl")
    update_records = _read_jsonl(RUN_DIR / "update_metrics.jsonl")
    episodes = sorted(
        (item for item in episode_records if item.get("record_type") == "episode"),
        key=lambda item: int(item.get("episode", -1)),
    )
    updates = sorted(
        (item for item in update_records if item.get("record_type") == "update"),
        key=lambda item: int(item.get("episode", -1)),
    )
    first5, middle5, last5 = episodes[:5], episodes[5:10], episodes[-5:]
    first3 = episodes[:3]
    first_states = _flatten_state_diagnostics(first5)
    last_states = _flatten_state_diagnostics(last5)
    macro_kls = [
        float(item["macro_approx_kl"])
        for item in updates
        if item.get("macro_approx_kl") is not None
    ]
    macro_kl_max = max((abs(value) for value in macro_kls), default=None)
    duration_rates = _duration_rates(last5)
    critic_evs = [
        float(item["macro_explained_variance"])
        for item in updates[-5:]
        if item.get("macro_explained_variance") is not None
    ]
    episode_summary = json.loads((RUN_DIR / "run_summary.json").read_text(encoding="utf-8"))
    complete = (
        int(run_result.get("completed_episodes", len(episodes))) == TARGET_EPISODES
        and len(episodes) == TARGET_EPISODES
        and [int(item.get("episode_seed", -1)) for item in episodes]
        == list(range(EPISODE_SEED_START, EPISODE_SEED_START + TARGET_EPISODES))
    )
    nan_inf_free = not _contains_nonfinite(
        {"episodes": episodes, "updates": updates, "summary": episode_summary}
    )
    count_weighted_total = sum(int(item.get("macro_frame_count", 0)) for item in episodes)
    legal_update_count_mean = (
        sum(
            float(item["legal_update_candidate_count_mean"])
            * int(item.get("macro_frame_count", 0))
            for item in episodes
            if item.get("legal_update_candidate_count_mean") is not None
        )
        / count_weighted_total
        if count_weighted_total
        else None
    )
    all_legal_counts = [
        value
        for item in episodes
        for value in (
            item.get("legal_update_candidate_count_min"),
            item.get("legal_update_candidate_count_max"),
        )
        if value is not None
    ]
    returns = {
        "first5": _mean(first5, "episode_return"),
        "middle5": _mean(middle5, "episode_return"),
        "last5": _mean(last5, "episode_return"),
    }
    keep_first5 = _mean(first5, "keep_rate")
    keep_last5 = _mean(last5, "keep_rate")
    update_first5 = _mean(first5, "update_rate")
    update_last5 = _mean(last5, "update_rate")
    raw_keep_first5 = _mean(first5, "raw_candidate0_probability_mean")
    corrected_keep_first3 = _mean(first3, "corrected_candidate0_probability_mean")
    corrected_keep_first5 = _mean(first5, "corrected_candidate0_probability_mean")
    corrected_keep_last5 = _mean(last5, "corrected_candidate0_probability_mean")
    corrected_update_last5 = _mean(last5, "corrected_update_group_probability_mean")
    keep_state_std_last5 = _mean(last5, "candidate0_probability_state_std")
    update_margins = [float(item["update_margin"]) for item in last_states]
    candidate_entropy_first5 = [
        float(item["candidate_entropy"])
        for item in first_states
        if item.get("candidate_entropy") is not None
    ]
    candidate_entropy_last5 = [
        float(item["candidate_entropy"])
        for item in last_states
        if item.get("candidate_entropy") is not None
    ]
    actual_change_last5 = _mean(
        last5, "actual_cache_configuration_change_rate"
    )
    update_actions_last5 = sum(int(item.get("update_count", 0)) for item in last5)
    no_effect_count_last5 = sum(
        float(item.get("nonzero_candidate_no_effect_rate", 0.0))
        * int(item.get("update_count", 0))
        for item in last5
    )
    no_effect_last5 = (
        no_effect_count_last5 / update_actions_last5 if update_actions_last5 else 0.0
    )
    entropy_first5 = (
        float(np.mean(candidate_entropy_first5)) if candidate_entropy_first5 else None
    )
    entropy_last5 = (
        float(np.mean(candidate_entropy_last5)) if candidate_entropy_last5 else None
    )
    static_checks = preflight_check["equal_logit_cardinality_checks"]
    cardinality_tests_pass = bool(
        preflight_check["equal_logit_k0_keep_probability"] == 1.0
        and preflight_check["equal_logit_k0_illegal_update_probability"] == 0.0
        and all(
            abs(float(value["keep_probability"]) - 0.5) <= 1.0e-6
            and abs(float(value["update_group_probability"]) - 0.5) <= 1.0e-6
            and float(value["probability_sum_error"]) <= 1.0e-6
            and float(value["illegal_probability_max"]) == 0.0
            for value in static_checks.values()
        )
    )
    cardinality_fixed = bool(
        cardinality_tests_pass
        and float(preflight_check["corrected_candidate0_probability"]) > 0.05
        and corrected_keep_first3 is not None
        and corrected_keep_first3 > 0.05
    )
    learned_update_collapse = bool(
        cardinality_fixed
        and keep_last5 is not None
        and update_last5 is not None
        and (keep_last5 <= 0.05 or update_last5 <= 0.05)
    )
    action_collapse = bool(
        keep_last5 is None
        or update_last5 is None
        or keep_last5 <= 0.05
        or keep_last5 >= 0.95
        or update_last5 <= 0.05
        or update_last5 >= 0.95
    )
    return_ok = bool(
        returns["first5"] is not None
        and returns["last5"] is not None
        and returns["last5"]
        >= returns["first5"] - max(abs(returns["first5"]) * 0.05, 1.0e-6)
    )
    entropy_ok = bool(
        entropy_first5 is not None
        and entropy_last5 is not None
        and entropy_last5 > max(0.05, entropy_first5 * 0.01)
    )
    numerical_ok = bool(
        nan_inf_free and macro_kl_max is not None and macro_kl_max <= 0.05
    )
    real_updates_ok = bool(
        actual_change_last5 is not None
        and actual_change_last5 > 0.0
        and no_effect_last5 <= 0.05
    )
    candidate0_variation = keep_state_std_last5
    gate_supported = bool(
        complete
        and cardinality_fixed
        and not action_collapse
        and candidate0_variation is not None
        and candidate0_variation > 0.02
        and return_ok
        and entropy_ok
        and numerical_ok
        and real_updates_ok
    )
    ready_for_longer = bool(
        complete
        and cardinality_fixed
        and not learned_update_collapse
        and return_ok
        and entropy_ok
        and numerical_ok
        and real_updates_ok
    )
    if not cardinality_fixed:
        failure_mode = "CARDINALITY_CORRECTION_IMPLEMENTATION_ERROR"
    elif learned_update_collapse:
        failure_mode = "LEARNED_UPDATE_COLLAPSE"
    elif not numerical_ok:
        failure_mode = "UNIFIED_POLICY_OPTIMIZATION_INSTABILITY"
    elif not return_ok or not real_updates_ok:
        failure_mode = "UNIFIED_CANDIDATE_OR_DURATION_LEARNING_FAILURE"
    elif candidate0_variation is None or candidate0_variation <= 0.02:
        failure_mode = "STATE_DEPENDENCE_NOT_YET_ESTABLISHED"
    elif gate_supported:
        failure_mode = "NONE"
    else:
        failure_mode = "UNIFIED_CANDIDATE_OR_DURATION_LEARNING_FAILURE"

    macro_frames = sum(int(item.get("macro_frame_count", 0)) for item in episodes)
    return {
        "experiment": "UNIFIED CACHE ACTION CARDINALITY CORRECTION, 15-episode gate",
        "status": "COMPLETED" if complete else "INCOMPLETE",
        "formal_runner_path": str(Path(__file__).resolve()),
        "formal_command": FORMAL_COMMAND,
        "run_dir": str(RUN_DIR.resolve()),
        "report_path": str(REPORT_PATH.resolve()),
        "config_path": str(CONFIG_PATH.resolve()),
        "episode_pool": str(EPISODE_POOL.resolve()),
        "base_seed": BASE_SEED,
        "episode_seed_start": EPISODE_SEED_START,
        "episode_seed_end": EPISODE_SEED_START + TARGET_EPISODES - 1,
        "target_episodes": TARGET_EPISODES,
        "episodes_completed": int(run_result.get("completed_episodes", len(episodes))),
        "architecture_version": ARCHITECTURE_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "unified_cache_action_active": True,
        "cardinality_correction_active": True,
        "trigger_active": False,
        "trigger_optimizer_steps": int(
            episode_summary.get("training_counters", {}).get(
                "macro_cache_trigger_optimizer_steps", 0
            )
        ),
        "trigger_teacher_used": any(
            bool(item.get("trigger_teacher_used")) for item in episodes
        ),
        "trigger_counterfactual_branches": int(
            sum(int(item.get("counterfactual_branch_count", 0)) for item in episodes)
        ),
        "candidate0_is_exact_keep": bool(preflight_check.get("candidate0_is_exact_keep")),
        "candidate_warmstart_used": bool(
            preflight_check.get("candidate_selection_head_warmstart_used", False)
        ),
        "warmstart_metadata": episode_summary.get("initialization_metadata", {}),
        "preflight": preflight,
        "implementation_preflight": preflight_check,
        "metrics": {
            "legal_update_candidate_count_mean": legal_update_count_mean,
            "legal_update_candidate_count_min": min(all_legal_counts) if all_legal_counts else None,
            "legal_update_candidate_count_max": max(all_legal_counts) if all_legal_counts else None,
            "return_first5_mean": returns["first5"],
            "return_middle5_mean": returns["middle5"],
            "return_last5_mean": returns["last5"],
            "keep_rate_first5_mean": keep_first5,
            "keep_rate_last5_mean": keep_last5,
            "cache_update_rate_first5_mean": update_first5,
            "cache_update_rate_last5_mean": update_last5,
            "raw_candidate0_probability_first5_mean": raw_keep_first5,
            "corrected_candidate0_probability_first3_mean": corrected_keep_first3,
            "corrected_candidate0_probability_first5_mean": corrected_keep_first5,
            "corrected_candidate0_probability_last5_mean": corrected_keep_last5,
            "corrected_update_group_probability_last5_mean": corrected_update_last5,
            "candidate0_probability_state_std_last5_mean": candidate0_variation,
            "update_margin_last5_mean": float(np.mean(update_margins)) if update_margins else None,
            "update_margin_positive_fraction_last5": (
                float(np.mean(np.asarray(update_margins) > 0.0)) if update_margins else None
            ),
            "candidate_entropy_first5_mean": entropy_first5,
            "candidate_entropy_last5_mean": entropy_last5,
            "actual_cache_configuration_change_rate_last5": actual_change_last5,
            "nonzero_candidate_no_effect_rate_last5": float(no_effect_last5),
            "duration_1_rate_last5": duration_rates["1"],
            "duration_3_rate_last5": duration_rates["3"],
            "duration_5_rate_last5": duration_rates["5"],
            "macro_kl_max": macro_kl_max,
            "macro_critic_ev_last5_mean": float(np.mean(critic_evs)) if critic_evs else None,
        },
        "gate_checks": {
            "all_15_episodes_completed": complete,
            "equal_logit_and_illegal_mask_tests_pass": cardinality_tests_pass,
            "cardinality_bias_fixed": cardinality_fixed,
            "keep_and_update_both_present_last5": not action_collapse,
            "candidate_state_variation_gt_0_02": (
                candidate0_variation is not None and candidate0_variation > 0.02
            ),
            "return_not_materially_below_first5": return_ok,
            "candidate_entropy_not_collapsed": entropy_ok,
            "macro_ppo_numerically_stable": numerical_ok,
            "real_cache_updates_without_no_effects": real_updates_ok,
            "nan_inf_free": nan_inf_free,
        },
        "static_checks": {
            "equal_logit_cardinality_checks": static_checks,
            "k0_keep_probability": preflight_check["equal_logit_k0_keep_probability"],
            "k0_illegal_update_probability": preflight_check[
                "equal_logit_k0_illegal_update_probability"
            ],
            "old_new_logprob_error": preflight_check[
                "sampled_joint_logprob_ppo_stored_error"
            ],
            "illegal_probability_max": max(
                (float(item["illegal_probability_max"]) for item in static_checks.values()),
                default=0.0,
            ),
            "trigger_forward_calls": preflight_check["trigger_forward_calls"],
            "trigger_optimizer_steps": preflight_check["trigger_optimizer_steps"],
            "trigger_teacher_calls": 0,
        },
        "nan_inf_free": nan_inf_free,
        "cardinality_bias_fixed": cardinality_fixed,
        "learned_update_collapse": learned_update_collapse,
        "cache_action_collapse": action_collapse,
        "unified_cache_action_gate_supported": gate_supported,
        "ready_for_longer_gate": ready_for_longer,
        "primary_failure_mode": failure_mode,
        "next_step": (
            "Proceed to a longer stability gate while preserving the corrected distribution."
            if ready_for_longer
            else "Stop; inspect this gate report before any further training."
        ),
        "episode_metrics": episodes,
        "update_metrics": updates,
    }


def main() -> int:
    from ppo_simple.trainer import run_training

    preflight = _assert_protocol_inputs()
    preflight_check = _implementation_preflight()
    run_result = run_training(
        CONFIG_PATH,
        TARGET_EPISODES,
        RUN_DIR,
        episode_pool=EPISODE_POOL,
        episode_seed_start=EPISODE_SEED_START,
        generate_missing_episodes=False,
        seed=BASE_SEED,
        resume=bool(preflight["resume_existing_prefix"]),
        verbose=False,
        initial_checkpoint_path=(
            None if preflight["resume_existing_prefix"] else V14B_CHECKPOINT
        ),
        expected_initial_checkpoint_sha256=(
            None if preflight["resume_existing_prefix"] else EXPECTED_V14B_SHA256
        ),
        unified_cache_action=True,
        cardinality_corrected_cache_action=True,
    )
    report = _build_report(
        preflight=preflight,
        preflight_check=preflight_check,
        run_result=run_result,
    )
    _write_json(REPORT_PATH, report)

    manifest_path = RUN_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "formal_runner_path": str(Path(__file__).resolve()),
            "formal_command": FORMAL_COMMAND,
            "report_path": str(REPORT_PATH.resolve()),
            "trigger_optimizer_steps": report["trigger_optimizer_steps"],
            "trigger_teacher_used": report["trigger_teacher_used"],
            "trigger_counterfactual_branches": report["trigger_counterfactual_branches"],
            "candidate0_is_exact_keep": report["candidate0_is_exact_keep"],
            "candidate_warmstart_used": report["candidate_warmstart_used"],
            "cardinality_corrected_cache_action": True,
            "cardinality_bias_fixed": report["cardinality_bias_fixed"],
            "learned_update_collapse": report["learned_update_collapse"],
            "unified_cache_action_gate_supported": report[
                "unified_cache_action_gate_supported"
            ],
        }
    )
    _write_json(manifest_path, manifest)
    summary_path = RUN_DIR / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["unified_cache_action_gate"] = {
        "report_path": str(REPORT_PATH.resolve()),
        "cardinality_corrected_cache_action": True,
        "cardinality_bias_fixed": report["cardinality_bias_fixed"],
        "learned_update_collapse": report["learned_update_collapse"],
        "supported": report["unified_cache_action_gate_supported"],
        "primary_failure_mode": report["primary_failure_mode"],
        "trigger_optimizer_steps": report["trigger_optimizer_steps"],
        "trigger_teacher_used": report["trigger_teacher_used"],
        "trigger_counterfactual_branches": report["trigger_counterfactual_branches"],
    }
    _write_json(summary_path, summary)
    print(json.dumps({
        "episodes_completed": report["episodes_completed"],
        "cardinality_bias_fixed": report["cardinality_bias_fixed"],
        "learned_update_collapse": report["learned_update_collapse"],
        "unified_cache_action_gate_supported": report["unified_cache_action_gate_supported"],
        "primary_failure_mode": report["primary_failure_mode"],
        "report_path": str(REPORT_PATH),
    }, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report["episodes_completed"] == TARGET_EPISODES else 2


if __name__ == "__main__":
    raise SystemExit(main())
