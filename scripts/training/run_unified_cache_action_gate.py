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
RUN_DIR = PROJECT_ROOT / "runs" / "formal_unified_cache_action_gate"
REPORT_PATH = PROJECT_ROOT / "reports" / "unified_cache_action_gate.json"
V14B_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "formal_v14b_actor_lr_1e_minus_3"
    / "checkpoints"
    / "latest_checkpoint.pt"
)
BASE_SEED = 27005
EPISODE_SEED_START = 20001
TARGET_EPISODES = 30
ARCHITECTURE_VERSION = 24
CHECKPOINT_VERSION = 48
EXPECTED_V14B_SHA256 = "82a835d7e5be6325566f14ffa65afeb388d7e3ba47e4ddce487a14e6b7d1e55c"
FORMAL_COMMAND = (
    "conda activate adaptive_cache_mec; "
    "python scripts/training/run_unified_cache_action_gate.py"
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
        "algorithm_id": "UNIFIED_CACHE_ACTION_V1",
        "architecture_version": ARCHITECTURE_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "target_episodes": TARGET_EPISODES,
        "episode_seed_start": EPISODE_SEED_START,
        "seed": BASE_SEED,
        "unified_cache_action": True,
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
        "recurrent_or_rl_stages_other_than_30_episode_macro_micro_training": False,
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
    from ppo_simple.policy import DualTimescaleActorCritic
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
    trainer = PPOTrainer(context, unified_cache_action=True)
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
        "candidate0_legal": bool(candidate_mask[0]),
        "candidate0_cache_change_count": int(np.count_nonzero(candidate0_change_mask)),
        "candidate0_cache_cost": float(cache_cost),
        "candidate0_loaded_cache_mb": float(np.sum(detailed_cost["loaded_mb"])),
        "candidate0_loading_energy": float(np.sum(loading_energy)),
        "unified_candidate_probability_sum": probability_sum,
        "sampled_candidate_id": int(action.cache_index),
        "sampled_requested_duration": int(action.requested_duration),
        "sampled_joint_logprob_ppo_stored_error": logprob_error,
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
    episodes = [item for item in episode_records if item.get("record_type") == "episode"]
    updates = [item for item in update_records if item.get("record_type") == "update"]
    episodes.sort(key=lambda item: int(item.get("episode", -1)))
    updates.sort(key=lambda item: int(item.get("episode", -1)))
    first10 = episodes[:10]
    middle10 = episodes[10:20]
    last10 = episodes[-10:]
    last_updates = updates[-10:]
    first_states = _flatten_state_diagnostics(first10)
    last_states = _flatten_state_diagnostics(last10)
    last_margins = [float(item["update_margin"]) for item in last_states]
    candidate_entropies_first = [
        float(item["candidate_entropy"])
        for item in first_states
        if item.get("candidate_entropy") is not None
    ]
    candidate_entropies_last = [
        float(item["candidate_entropy"])
        for item in last_states
        if item.get("candidate_entropy") is not None
    ]
    duration_rates = _duration_rates(last10)
    macro_kls = [
        float(item["macro_approx_kl"])
        for item in updates
        if item.get("macro_approx_kl") is not None
    ]
    macro_kl_max = max(macro_kls) if macro_kls else None
    macro_kl_max_abs = max((abs(value) for value in macro_kls), default=None)
    return_first10 = _mean(first10, "episode_return")
    return_last10 = _mean(last10, "episode_return")
    keep_first10 = _mean(first10, "keep_rate")
    keep_last10 = _mean(last10, "keep_rate")
    update_first10 = _mean(first10, "update_rate")
    update_last10 = _mean(last10, "update_rate")
    candidate0_first10 = _mean(first10, "candidate0_probability_mean")
    candidate0_last10 = _mean(last10, "candidate0_probability_mean")
    candidate0_state_std_last10 = _mean(last10, "candidate0_probability_state_std")
    candidate_entropy_first10 = (
        float(np.mean(candidate_entropies_first)) if candidate_entropies_first else None
    )
    candidate_entropy_last10 = (
        float(np.mean(candidate_entropies_last)) if candidate_entropies_last else None
    )
    update_margin_last10 = float(np.mean(last_margins)) if last_margins else None
    margin_positive_fraction = (
        float(np.mean(np.asarray(last_margins) > 0.0)) if last_margins else None
    )
    actual_change_last10 = _mean(last10, "actual_cache_configuration_change_rate")
    update_action_count = sum(int(item.get("update_count", 0)) for item in last10)
    no_effect_numerator = sum(
        float(item.get("nonzero_candidate_no_effect_rate", 0.0))
        * int(item.get("update_count", 0))
        for item in last10
    )
    no_effect_last10 = (
        float(no_effect_numerator / update_action_count) if update_action_count else 0.0
    )
    critic_evs = [
        float(item["macro_explained_variance"])
        for item in last_updates
        if item.get("macro_explained_variance") is not None
    ]
    critic_ev_last10 = float(np.mean(critic_evs)) if critic_evs else None
    episode_summary = json.loads((RUN_DIR / "run_summary.json").read_text(encoding="utf-8"))
    completed = int(run_result.get("completed_episodes", len(episodes)))
    nan_inf_free = not _contains_nonfinite(
        {"episodes": episodes, "updates": updates, "summary": episode_summary}
    )
    finite_and_complete = completed == TARGET_EPISODES and len(episodes) == TARGET_EPISODES
    no_action_collapse = bool(
        keep_last10 is not None and 0.05 < keep_last10 < 0.95
    )
    entropy_ok = bool(
        candidate_entropy_last10 is not None
        and candidate_entropy_first10 is not None
        and candidate_entropy_last10 > max(0.05, candidate_entropy_first10 * 0.01)
    )
    state_variation_ok = bool(
        candidate0_state_std_last10 is not None and candidate0_state_std_last10 > 0.02
    )
    return_ok = bool(
        return_first10 is not None
        and return_last10 is not None
        and return_last10
        >= return_first10 - max(abs(return_first10) * 0.05, 1.0e-6)
    )
    update_effect_ok = bool(
        update_last10 is not None
        and update_last10 > 0.0
        and actual_change_last10 is not None
        and actual_change_last10 > 0.0
        and no_effect_last10 <= 0.05
    )
    numerical_stability_ok = bool(
        nan_inf_free and macro_kl_max_abs is not None and macro_kl_max_abs <= 0.05
    )
    gate_supported = bool(
        finite_and_complete
        and no_action_collapse
        and entropy_ok
        and state_variation_ok
        and return_ok
        and update_effect_ok
        and numerical_stability_ok
        and preflight_check.get("candidate0_is_exact_keep") is True
        and preflight_check.get("trigger_forward_calls") == 0
    )
    if not no_action_collapse and keep_last10 is not None and keep_last10 >= 0.95:
        failure_mode = "UNIFIED_POLICY_KEEP_COLLAPSE"
    elif not no_action_collapse and keep_last10 is not None and keep_last10 <= 0.05:
        failure_mode = "UNIFIED_POLICY_UPDATE_COLLAPSE"
    elif not numerical_stability_ok:
        failure_mode = "UNIFIED_POLICY_OPTIMIZATION_INSTABILITY"
    elif not return_ok:
        failure_mode = "UNIFIED_CANDIDATE_QUALITY_OR_DURATION_COUPLING"
    elif not preflight_check.get("candidate0_is_exact_keep"):
        failure_mode = "UNIFIED_ACTION_IMPLEMENTATION_ERROR"
    elif gate_supported:
        failure_mode = "NONE"
    else:
        failure_mode = "UNIFIED_POLICY_OPTIMIZATION_INSTABILITY"

    return {
        "experiment": "UNIFIED CACHE ACTION REPAIR, 30-episode training gate",
        "status": "COMPLETED" if finite_and_complete else "INCOMPLETE",
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
        "episodes_completed": completed,
        "architecture_version": ARCHITECTURE_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "unified_cache_action_active": True,
        "trigger_active": False,
        "trigger_optimizer_steps": int(
            episode_summary.get("training_counters", {}).get(
                "macro_cache_trigger_optimizer_steps", 0
            )
        ),
        "trigger_teacher_used": any(bool(item.get("trigger_teacher_used")) for item in episodes),
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
            "return_first10_mean": return_first10,
            "return_middle10_mean": _mean(middle10, "episode_return"),
            "return_last10_mean": return_last10,
            "keep_rate_first10_mean": keep_first10,
            "keep_rate_last10_mean": keep_last10,
            "cache_update_rate_first10_mean": update_first10,
            "cache_update_rate_last10_mean": update_last10,
            "candidate0_probability_first10_mean": candidate0_first10,
            "candidate0_probability_last10_mean": candidate0_last10,
            "candidate0_probability_state_std_last10_mean": candidate0_state_std_last10,
            "update_margin_last10_mean": update_margin_last10,
            "update_margin_positive_fraction_last10": margin_positive_fraction,
            "candidate_entropy_first10_mean": candidate_entropy_first10,
            "candidate_entropy_last10_mean": candidate_entropy_last10,
            "actual_cache_configuration_change_rate_last10": actual_change_last10,
            "nonzero_candidate_no_effect_rate_last10": no_effect_last10,
            "duration_1_rate_last10": duration_rates["1"],
            "duration_3_rate_last10": duration_rates["3"],
            "duration_5_rate_last10": duration_rates["5"],
            "macro_kl_max": macro_kl_max,
            "macro_kl_max_abs_for_gate": macro_kl_max_abs,
            "macro_critic_ev_last10_mean": critic_ev_last10,
        },
        "gate_checks": {
            "all_30_episodes_completed": finite_and_complete,
            "no_action_collapse": no_action_collapse,
            "candidate_entropy_not_collapsed": entropy_ok,
            "candidate0_state_variation_gt_0_02": state_variation_ok,
            "return_last10_not_materially_below_first10": return_ok,
            "real_cache_updates_without_nonzero_no_effects": update_effect_ok,
            "macro_ppo_numerically_stable": numerical_stability_ok,
            "nan_inf_free": nan_inf_free,
        },
        "nan_inf_free": nan_inf_free,
        "cache_action_collapse": not no_action_collapse,
        "unified_cache_action_gate_supported": gate_supported,
        "trigger_removal_supported": gate_supported,
        "ready_for_100_episode_run": gate_supported,
        "primary_failure_mode": failure_mode,
        "next_step": (
            "Run the authorized unified architecture 100-episode stability training."
            if gate_supported
            else "Stop; do not start another algorithm or training stage."
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
        "supported": report["unified_cache_action_gate_supported"],
        "primary_failure_mode": report["primary_failure_mode"],
        "trigger_optimizer_steps": report["trigger_optimizer_steps"],
        "trigger_teacher_used": report["trigger_teacher_used"],
        "trigger_counterfactual_branches": report["trigger_counterfactual_branches"],
    }
    _write_json(summary_path, summary)
    print(json.dumps({
        "episodes_completed": report["episodes_completed"],
        "unified_cache_action_gate_supported": report["unified_cache_action_gate_supported"],
        "primary_failure_mode": report["primary_failure_mode"],
        "report_path": str(REPORT_PATH),
    }, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report["episodes_completed"] == TARGET_EPISODES else 2


if __name__ == "__main__":
    raise SystemExit(main())
