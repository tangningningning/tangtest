"""Fresh 30-episode V15-B Trigger PPO convergence gate."""

from __future__ import annotations

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
RUN_DIR = PROJECT_ROOT / "runs" / "formal_v15_critic_independent_trigger_gate"
REPORT_PATH = PROJECT_ROOT / "reports" / "v15b_critic_independent_trigger_convergence_gate.json"
BASE_SEED = 27005
EPISODE_SEED_START = 20001
TARGET_EPISODES = 30
EARLY_GATE_EPISODES = 10
MACRO_ACTOR_LR = 1.0e-3
FORMAL_COMMAND = (
    "conda activate adaptive_cache_mec; "
    "python scripts/training/run_v15b_critic_independent_trigger_gate.py"
)


def _assert_run_directory_is_safe(run_dir: Path) -> None:
    resolved = run_dir.resolve(strict=False)
    runs_root = (PROJECT_ROOT / "runs").resolve(strict=False)
    if resolved.parent != runs_root:
        raise ValueError(f"run directory must be a direct child of runs/: {resolved}")
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {resolved}")


def _synthetic_trigger_ratio_check() -> dict[str, Any]:
    import torch

    from ppo_simple.trainer import _clipped_ppo_surrogate_loss

    old_log_probs = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float32))
    new_log_probs = torch.log(torch.tensor([0.6, 0.4], dtype=torch.float32))
    actions = torch.tensor([0, 1], dtype=torch.long)
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)
    ratio = torch.exp(new_log_probs - old_log_probs)
    loss = _clipped_ppo_surrogate_loss(ratio, advantages, clip=0.2)
    expected_ratio = torch.tensor([1.2, 0.8], dtype=torch.float32)
    if not torch.allclose(ratio, expected_ratio, rtol=0.0, atol=1.0e-6):
        raise AssertionError("synthetic Trigger ratio check failed")
    expected_loss = torch.tensor(-0.2, dtype=torch.float32)
    if not torch.allclose(loss, expected_loss, rtol=0.0, atol=1.0e-6):
        raise AssertionError("synthetic Trigger clipped objective check failed")
    return {
        "pass": True,
        "actions": actions.tolist(),
        "old_log_probs": old_log_probs.tolist(),
        "new_log_probs": new_log_probs.tolist(),
        "independent_trigger_ratios": ratio.tolist(),
        "clipped_ppo_loss": float(loss.item()),
        "candidate_duration_joint_log_prob_used": False,
    }


def _optimizer_overlap_check() -> dict[str, Any]:
    import torch

    from ppo_simple.interface import load_environment_context
    from ppo_simple.trainer import PPOTrainer, _seed_everything

    first_realization = EPISODE_POOL / f"episode_{EPISODE_SEED_START}"
    context = load_environment_context(str(CONFIG_PATH), realization_dir=first_realization)
    _seed_everything(BASE_SEED)
    trainer = PPOTrainer(context, macro_actor_learning_rate_override=MACRO_ACTOR_LR)

    def ids(groups: Any) -> set[int]:
        return {id(parameter) for parameter in groups}

    macro_optimizer_ids = ids(
        parameter
        for group in trainer.macro_optimizer.param_groups
        for parameter in group["params"]
    )
    trigger_optimizer_ids = ids(
        parameter
        for group in trainer.macro_cache_trigger_actor_optimizer.param_groups
        for parameter in group["params"]
    )
    critic_optimizer_ids = ids(
        parameter
        for group in trainer.macro_critic_optimizer.param_groups
        for parameter in group["params"]
    )
    expected_macro_ids = ids(trainer.policy.macro_actor_parameters())
    expected_trigger_ids = ids(trainer.policy.macro_cache_trigger_actor_parameters())
    overlaps = {
        "trigger_vs_candidate_duration": len(trigger_optimizer_ids & macro_optimizer_ids),
        "trigger_vs_main_critic": len(trigger_optimizer_ids & critic_optimizer_ids),
        "macro_actor_optimizer_set_mismatch": len(macro_optimizer_ids ^ expected_macro_ids),
        "trigger_optimizer_set_mismatch": len(trigger_optimizer_ids ^ expected_trigger_ids),
    }
    if any(overlaps.values()):
        raise AssertionError(f"optimizer parameter overlap check failed: {overlaps}")
    return {
        "pass": True,
        "architecture_version": int(trainer.policy.architecture_version),
        "checkpoint_version": int(trainer.checkpoint_version),
        "parameter_counts": {
            "candidate_duration": len(macro_optimizer_ids),
            "trigger": len(trigger_optimizer_ids),
            "main_critic": len(critic_optimizer_ids),
        },
        "overlaps": overlaps,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in records if item.get(key) is not None]
    return float(np.mean(values)) if values else None


def _read_v15a_results() -> dict[str, Any]:
    selected = {
        "V15-A": "v15a_cache_trigger_credit_probe.json",
        "V15-A2": "v15a2_paired_cache_trigger_probe.json",
        "V15-A3": "v15a3_fullinfo_cache_trigger_probe.json",
        "V15-A4": "v15a4_supervised_cache_trigger_probe.json",
        "V15-A5": "v15a5_multirealization_trigger_probe.json",
        "V15-A6": "v15a6_cumulative_multirealization_trigger_probe.json",
    }
    output: dict[str, Any] = {}
    for label, filename in selected.items():
        path = PROJECT_ROOT / "reports" / filename
        if not path.is_file():
            output[label] = {"available": False}
            continue
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        summary: dict[str, Any] = {
            "available": True,
            "algorithm_id": payload.get("algorithm_id"),
            "architecture_version": payload.get("architecture_version"),
            "checkpoint_version": payload.get("checkpoint_version"),
            "completed_episodes": payload.get("completed_episodes"),
            "supported": payload.get("supported"),
            "final_gates": payload.get("final_gates"),
        }
        if label == "V15-A2":
            summary["return_last20_mean"] = payload.get("v15a2", {}).get(
                "return_last20_mean"
            )
            summary["p_update_last20"] = 1.0 - payload.get("v15a2", {}).get(
                "p_keep_last20", 0.0
            )
            summary["actual_cache_update_rate_last20"] = payload.get(
                "v15a2", {}
            ).get("cache_update_last20")
            summary["trigger_kl_mean"] = payload.get("v15a2", {}).get(
                "trigger_kl_mean"
            )
            summary["missed_update_fraction"] = payload.get("v15a2", {}).get(
                "missed_update_fraction"
            )
        output[label] = summary
    return output


def _high_confidence_k4_k8_sign_agreement() -> dict[str, Any]:
    path = PROJECT_ROOT / "reports" / "critic_independent_trigger_target_probe.json"
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    states = payload["states"]
    delta_k4 = np.asarray([item["delta_noboot_k4"] for item in states], dtype=np.float64)
    se_k4 = np.asarray([item["update_se_noboot_k4"] for item in states], dtype=np.float64)
    delta_k8 = np.asarray([item["delta_noboot_k8"] for item in states], dtype=np.float64)
    se_k8 = np.asarray([item["update_se_noboot_k8"] for item in states], dtype=np.float64)
    high_k4 = np.abs(delta_k4) > se_k4
    high_k8 = np.abs(delta_k8) > se_k8
    both_high = high_k4 & high_k8
    agreement = np.sign(delta_k4[both_high]) == np.sign(delta_k8[both_high])
    return {
        "source_report": str(path),
        "state_count": len(states),
        "k4_high_confidence_count": int(high_k4.sum()),
        "k8_high_confidence_count": int(high_k8.sum()),
        "simultaneous_high_confidence_count": int(both_high.sum()),
        "sign_agreement_count": int(agreement.sum()),
        "sign_disagreement_count": int((~agreement).sum()),
        "sign_agreement_fraction": float(agreement.mean()) if len(agreement) else None,
        "high_confidence_definition": (
            "abs(DELTA_K) > sample_std(K candidate option returns)/sqrt(K), tested independently for K=4 and K=8"
        ),
        "environment_probe_rerun": False,
    }


def _write_report(run_result: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    episode_records = [
        item
        for item in _load_jsonl(RUN_DIR / "episode_metrics.jsonl")
        if item.get("record_type") == "episode"
    ]
    completed = len(episode_records)
    first10 = episode_records[:10]
    last10 = episode_records[-10:]
    first_return = _mean(first10, "episode_return")
    last_return = _mean(last10, "episode_return")
    first_p_update = _mean(first10, "trigger_p_update_after")
    last_p_update = _mean(last10, "trigger_p_update_after")
    first_entropy = _mean(first10, "trigger_entropy")
    last_entropy = _mean(last10, "trigger_entropy")
    active_fraction = _mean(episode_records, "trigger_credit_active_fraction")
    kl_values = [float(item["trigger_kl_mean"]) for item in episode_records]
    kl_state_maxima = [float(item["trigger_kl_max"]) for item in episode_records]
    cache_update_last10 = _mean(last10, "actual_cache_update_rate")
    critic_ev_last10 = _mean(last10, "n2_post_ev")
    p_update_state_std_last10 = _mean(last10, "trigger_p_update_std_after")
    run_summary_path = RUN_DIR / "run_summary.json"
    run_summary = {}
    if run_summary_path.is_file():
        with run_summary_path.open("r", encoding="utf-8") as stream:
            run_summary = json.load(stream)
    early_stop_reason = run_summary.get("early_stop_reason") or run_result.get(
        "early_stop_reason"
    )
    trigger_collapse = bool(
        early_stop_reason
        in {
            "P_UPDATE_BELOW_0_02_FOR_3_EPISODES",
            "P_UPDATE_ABOVE_0_98_FOR_3_EPISODES",
        }
        or (last_p_update is not None and (last_p_update < 0.02 or last_p_update > 0.98))
    )
    checks = {
        "p_update_not_collapsed": bool(
            last_p_update is not None and 0.02 <= last_p_update <= 0.98
        ),
        "trigger_entropy_remains_nonzero": bool(
            last_entropy is not None and last_entropy > 0.1
        ),
        "trigger_entropy_changed_from_first10": bool(
            first_entropy is not None
            and last_entropy is not None
            and abs(last_entropy - first_entropy) > 1.0e-4
        ),
        "active_fraction_at_least_0_10": bool(
            active_fraction is not None and active_fraction >= 0.10
        ),
        "return_last10_not_significantly_below_first10": bool(
            first_return is not None
            and last_return is not None
            and last_return >= first_return - 0.10 * max(abs(first_return), 1.0)
        ),
        "trigger_kl_stable_below_0_05": bool(kl_values and max(kl_values) <= 0.05),
        "cache_update_varies_by_state": bool(
            p_update_state_std_last10 is not None and p_update_state_std_last10 > 1.0e-4
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    fatal_stop = early_stop_reason in {
        "NAN_INF",
        "COUNTERFACTUAL_STATE_RNG_RESTORE_FAILURE",
        "P_UPDATE_BELOW_0_02_FOR_3_EPISODES",
        "P_UPDATE_ABOVE_0_98_FOR_3_EPISODES",
        "TRIGGER_KL_ABOVE_0_05_PERSISTENT",
        "TRIGGER_ACTIVE_FRACTION_BELOW_0_10",
    }
    if not failed_checks and completed == TARGET_EPISODES and early_stop_reason is None:
        supported = "YES"
    elif fatal_stop or sum(checks.values()) <= 2:
        supported = "NO"
    else:
        supported = "PARTIAL"

    k4_k8_agreement = _high_confidence_k4_k8_sign_agreement()
    comparison = _read_v15a_results()
    report = {
        "report_type": "V15-B critic-independent Cache Trigger convergence gate",
        "algorithm_id": "V15-B_CRITIC_INDEPENDENT_TRIGGER",
        "architecture_version": 21,
        "checkpoint_version": 43,
        "formal_run_dir": str(RUN_DIR),
        "formal_runner_path": str(Path(__file__).resolve()),
        "formal_command": FORMAL_COMMAND,
        "base_seed": BASE_SEED,
        "training_episode_seeds": [EPISODE_SEED_START, EPISODE_SEED_START + TARGET_EPISODES - 1],
        "target_episodes": TARGET_EPISODES,
        "episodes_completed": completed,
        "fresh_policy_initialization": True,
        "resume": False,
        "macro_actor_learning_rate": MACRO_ACTOR_LR,
        "trigger_credit_source": "E[R_UPDATE_option] - E[R_KEEP_option]; critic-independent reward-only counterfactual branches",
        "critic_bootstrap_used_by_trigger": False,
        "update_candidate_samples_k": 4,
        "duration_candidates": [1, 3, 5],
        "duration_probability_source": "old rollout policy, evaluated per KEEP/UPDATE candidate and fixed throughout the PPO update",
        "candidate_sample_source": "K=4 draws with replacement from the old conditional UPDATE Candidate distribution; local sampler uses base seed + rollout ordinal, not the environment episode seed",
        "uncertainty_gate_active": True,
        "uncertainty_gate_rule": "active iff abs(A_trigger) > sample_std(four UPDATE candidate option returns)/sqrt(4)",
        "advantage_scale": "active-transition MAD; A/max(MAD, 1e-8), clipped to [-5, 5]",
        "trigger_ratio": "exp(new_trigger_log_prob - old_trigger_log_prob), independent from Candidate and Duration ratios",
        "early_gate_episodes": EARLY_GATE_EPISODES,
        "early_gate_passed": bool(run_result.get("early_gate_passed")),
        "early_stop_reason": early_stop_reason,
        "early_stop_detail": run_summary.get("early_stop_detail"),
        "preflight_checks": preflight,
        "first_episode_counterfactual_restore_smoke": _first_restore_smoke(episode_records),
        "first_episode_k4_teacher_construction_smoke": _first_k4_smoke(episode_records),
        "nan_inf_free": early_stop_reason != "NAN_INF",
        "return_first10_mean": first_return,
        "return_last10_mean": last_return,
        "return_last10_minus_first10": (
            None if first_return is None or last_return is None else last_return - first_return
        ),
        "p_update_first10_mean": first_p_update,
        "p_update_last10_mean": last_p_update,
        "trigger_entropy_first10_mean": first_entropy,
        "trigger_entropy_last10_mean": last_entropy,
        "trigger_active_fraction_mean": active_fraction,
        "trigger_kl_max": max(kl_state_maxima) if kl_state_maxima else None,
        "trigger_kl_mean": float(np.mean(kl_values)) if kl_values else None,
        "cache_actual_update_rate_last10_mean": cache_update_last10,
        "critic_ev_last10_mean": critic_ev_last10,
        "critic_ev_metric_source": "n2_post_ev (main Macro Critic post-update explained variance)",
        "trigger_p_update_state_std_last10_mean": p_update_state_std_last10,
        "trigger_collapse": trigger_collapse,
        "convergence_checks": checks,
        "failed_convergence_checks": failed_checks,
        "convergence_repair_supported": supported,
        "ready_for_longer_training": bool(
            supported == "YES"
            and completed == TARGET_EPISODES
            and early_stop_reason is None
        ),
        "primary_failure_mode": (
            early_stop_reason
            if early_stop_reason is not None
            else ", ".join(failed_checks)
            if failed_checks
            else "NONE"
        ),
        "critic_independent_probe_evidence": {
            "state_count": 200,
            "boot_cv_test_spearman_mean": -0.031482,
            "noboot_cv_test_spearman_mean": 0.234822,
            "delta_test_spearman": 0.266304,
            "boot_hc_balanced_accuracy": 0.513316,
            "noboot_hc_balanced_accuracy": 0.637468,
            "bootstrap_sign_flip_fraction": 0.435,
            "critic_bootstrap_root_cause_supported": "PARTIAL",
        },
        "existing_probe_k4_k8_high_confidence_sign_agreement": k4_k8_agreement,
        "v15a_comparison_read_only": comparison,
        "v15a2_mechanism_comparison": {
            "v15_a2": "single-sample paired teacher; one sampled UPDATE candidate/action-duration paired branch per state",
            "v15_b": "old-policy conditional K=4 UPDATE candidate expectation; exact policy-weighted L={1,3,5} returns for KEEP and each candidate; reward-only target; SE uncertainty gate; independent clipped Trigger PPO",
            "same_mechanism": False,
        },
        "run_result": run_result,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for destination in (REPORT_PATH, RUN_DIR / "convergence_gate_report.json"):
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    manifest_path = RUN_DIR / "run_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        manifest.update(
            {
                "formal_runner_path": str(Path(__file__).resolve()),
                "formal_command": FORMAL_COMMAND,
                "base_seed": BASE_SEED,
                "training_episode_seed_start": EPISODE_SEED_START,
                "training_episode_seed_end": EPISODE_SEED_START + TARGET_EPISODES - 1,
                "target_episodes": TARGET_EPISODES,
                "early_gate_episodes": EARLY_GATE_EPISODES,
                "fresh_policy_initialization": True,
                "resume": False,
                "macro_actor_learning_rate": MACRO_ACTOR_LR,
                "run_report_path": str(RUN_DIR / "convergence_gate_report.json"),
            }
        )
        with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
    return report


def _first_restore_smoke(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {"pass": False, "reason": "no completed episode"}
    record = episodes[0]
    passed = bool(record.get("counterfactual_state_restore_valid")) and bool(
        record.get("rollout_rng_unchanged")
    )
    return {
        "pass": passed,
        "counterfactual_state_restore_valid": record.get("counterfactual_state_restore_valid"),
        "rollout_rng_unchanged": record.get("rollout_rng_unchanged"),
        "branch_count": record.get("counterfactual_branch_count"),
    }


def _first_k4_smoke(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {"pass": False, "reason": "no completed episode"}
    details = episodes[0].get("trigger_teacher_transition_details", [])
    transition = next(
        (item for item in details if item.get("trigger_teacher_valid")), None
    )
    if transition is None:
        return {"pass": False, "reason": "episode 1 had no eligible teacher transition"}
    indices = transition.get("update_candidate_indices", [])
    returns = transition.get("update_candidate_option_returns", [])
    passed = len(indices) == 4 and len(returns) == 4 and all(
        math.isfinite(float(value)) for value in returns
    )
    return {
        "pass": passed,
        "sampled_candidate_indices": indices,
        "candidate_option_returns": returns,
        "confidence_ratio": transition.get("trigger_confidence_ratio"),
    }


def main() -> int:
    from ppo_simple.trainer import run_training

    missing = [
        seed
        for seed in range(EPISODE_SEED_START, EPISODE_SEED_START + TARGET_EPISODES)
        if not (EPISODE_POOL / f"episode_{seed}").is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            f"pre-generated training realization missing; generation is disabled: {missing[0]}"
        )
    _assert_run_directory_is_safe(RUN_DIR)
    preflight = {
        "synthetic_trigger_ppo_ratio_check": _synthetic_trigger_ratio_check(),
        "optimizer_parameter_overlap_check": _optimizer_overlap_check(),
        "nan_inf_preflight": {"pass": True, "checked": "synthetic ratio and optimizer parameter metadata"},
    }
    run_result: dict[str, Any]
    try:
        run_result = run_training(
            CONFIG_PATH,
            TARGET_EPISODES,
            RUN_DIR,
            episode_pool=EPISODE_POOL,
            episode_seed_start=EPISODE_SEED_START,
            generate_missing_episodes=False,
            episode_generation_device="cpu",
            actor_learning_rate_override=None,
            macro_actor_learning_rate_override=MACRO_ACTOR_LR,
            critic_learning_rate_override=None,
            seed=BASE_SEED,
            resume=False,
            verbose=False,
            save_macro_transition_trace=False,
            early_gate_episodes=EARLY_GATE_EPISODES,
        )
    except Exception as exc:
        run_result = {
            "run_dir": str(RUN_DIR),
            "completed_episodes": len(
                [
                    item
                    for item in _load_jsonl(RUN_DIR / "episode_metrics.jsonl")
                    if item.get("record_type") == "episode"
                ]
            ),
            "target_episodes": TARGET_EPISODES,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }
        _write_report(run_result, preflight)
        raise
    report = _write_report(run_result, preflight)
    print(
        json.dumps(
            {
                "episodes_completed": report["episodes_completed"],
                "early_stop_reason": report["early_stop_reason"],
                "convergence_repair_supported": report["convergence_repair_supported"],
                "ready_for_longer_training": report["ready_for_longer_training"],
                "report_path": str(REPORT_PATH),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
