"""Frozen paired Validation adapter using the existing Stage1C evaluator.

The adapter enables strict Stage3-v4 checkpoint loading, then delegates action
collection and formal metric checks to the existing MAPPO baseline collector
and Stage1C frozen evaluator. It never calls Trainer.update().
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_simple.mappo_baseline import (  # noqa: E402
    CounterfactualLabelBatchError,
    MAPPOBaselineTrainer,
)
from ppo_simple.cache_counterfactual_value import (  # noqa: E402
    Stage3CacheValueConfig,
)
from scripts.training import run_mappo_stage3_cache_value as stage3_runner  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_existing_evaluator(path: Path) -> Any:
    source = path.resolve(strict=True)
    spec = importlib.util.spec_from_file_location(
        "stage3_existing_stage1c_frozen_evaluator", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load existing frozen evaluator: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    base = module.FrozenEvaluationTrainer.__mro__[1]
    if base is not MAPPOBaselineTrainer:
        raise RuntimeError(
            "Existing Stage1C evaluator did not bind to the active Stage3-worktree "
            "MAPPOBaselineTrainer; refusing mixed-code evaluation."
        )
    return module


def _stage3_compatible_trainer_class(evaluator: Any) -> type:
    """Adapt the Stage1C evaluator override to the Stage3 Micro CRN contract.

    Stage1C's override predates ``random_uniforms``. Counterfactual replay must
    reach the active MAPPO implementation so each branch consumes the same two
    recorded uniforms while reapplying that branch's own legal action masks.
    The forced KEEP+STAY path remains delegated to the original evaluator.
    """

    class Stage3CompatibleFrozenEvaluationTrainer(
        evaluator.FrozenEvaluationTrainer
    ):
        def _sample_micro_head(
            self,
            observation: np.ndarray,
            features: np.ndarray,
            mask: np.ndarray,
            *,
            uav_index: int,
            uav0_position_condition: np.ndarray,
            deterministic: bool = False,
            random_uniforms: np.ndarray | tuple[float, float] | None = None,
        ) -> tuple[int, float, np.ndarray, float]:
            if bool(getattr(self, "keep_stay", False)):
                if random_uniforms is not None:
                    raise ValueError(
                        "KEEP+STAY evaluation cannot consume counterfactual Micro uniforms."
                    )
                return super()._sample_micro_head(
                    observation,
                    features,
                    mask,
                    uav_index=uav_index,
                    uav0_position_condition=uav0_position_condition,
                    deterministic=deterministic,
                )
            return MAPPOBaselineTrainer._sample_micro_head(
                self,
                observation,
                features,
                mask,
                uav_index=uav_index,
                uav0_position_condition=uav0_position_condition,
                deterministic=deterministic,
                random_uniforms=random_uniforms,
            )

    return Stage3CompatibleFrozenEvaluationTrainer


def _stage3_label_metrics(
    trainer: Any,
    rollout: Any,
    *,
    profile: Stage3CacheValueConfig,
) -> dict[str, Any]:
    try:
        batch, diagnostics = trainer._build_counterfactual_training_batch(rollout)
    except CounterfactualLabelBatchError as exc:
        return {
            "status": "unresolved_no_valid_legal_update_labels",
            "counterfactual": exc.summary,
            "delta_q_rmse": None,
            "delta_q_mae": None,
            "delta_q_sign_accuracy_excluding_ties": None,
            "trigger_accuracy": None,
            "candidate_pairwise_rank_accuracy": None,
            "candidate_pair_count": 0,
        }
    head = trainer.cache_delta_q_value_head
    if head is None:
        raise RuntimeError("Stage3 checkpoint did not restore its ΔQ value head.")
    with torch.inference_mode():
        predictions = head(
            batch["observations"],
            batch["node_features"],
            batch["candidate_features"],
        ).detach().cpu().numpy()
    predicted = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(batch["delta_q_targets"], dtype=np.float64)
    masks = np.asarray(batch["update_masks"], dtype=bool)
    if (
        predicted.shape != targets.shape
        or not np.isfinite(predicted).all()
        or not np.isfinite(targets).all()
    ):
        raise FloatingPointError("Frozen Validation ΔQ labels/predictions are malformed.")
    errors = predicted[masks] - targets[masks]
    valid_targets = targets[masks]
    sign_valid = np.abs(valid_targets) > float(profile.candidate_tie_tolerance)
    sign_accuracy = (
        float(np.mean(np.sign(predicted[masks][sign_valid]) == np.sign(valid_targets[sign_valid])))
        if bool(np.any(sign_valid))
        else None
    )
    trigger_actual: list[bool] = []
    trigger_predicted: list[bool] = []
    correct_pairs = 0
    compared_pairs = 0
    for row in range(len(targets)):
        legal = np.flatnonzero(masks[row])
        true_best = float(np.max(targets[row, legal]))
        predicted_best = float(np.max(predicted[row, legal]))
        trigger_actual.append(true_best > float(profile.trigger_tie_tolerance))
        trigger_predicted.append(predicted_best > float(profile.trigger_tie_tolerance))
        for left_pos, left in enumerate(legal):
            for right in legal[left_pos + 1 :]:
                true_delta = float(targets[row, left] - targets[row, right])
                if abs(true_delta) <= float(profile.candidate_tie_tolerance):
                    continue
                estimate_delta = float(predicted[row, left] - predicted[row, right])
                correct_pairs += int(true_delta * estimate_delta > 0.0)
                compared_pairs += 1
    return {
        "counterfactual": diagnostics,
        "candidate_count": int(np.count_nonzero(masks)),
        "delta_q_rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        "delta_q_mae": float(np.mean(np.abs(errors))) if len(errors) else None,
        "delta_q_sign_accuracy_excluding_ties": sign_accuracy,
        "trigger_accuracy": float(
            np.mean(np.asarray(trigger_actual) == np.asarray(trigger_predicted))
        ),
        "trigger_positive_count": int(sum(trigger_actual)),
        "trigger_predicted_positive_count": int(sum(trigger_predicted)),
        "candidate_pairwise_rank_accuracy": (
            correct_pairs / compared_pairs if compared_pairs else None
        ),
        "candidate_pair_count": compared_pairs,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--stage1b-helper", type=Path, required=True)
    parser.add_argument("--existing-evaluator", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--stage3-checkpoint", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-seeds", type=int, nargs=3, default=(30001, 30002, 30003))
    args = parser.parse_args()

    formal_root = args.formal_root.resolve(strict=True)
    helper_path = args.stage1b_helper.resolve(strict=True)
    helper_spec = importlib.util.spec_from_file_location(
        "stage3_main_experiment_pool_contract", helper_path
    )
    if helper_spec is None or helper_spec.loader is None:
        raise ImportError(f"Cannot load Stage3 pool contract: {helper_path}")
    helper = importlib.util.module_from_spec(helper_spec)
    sys.modules[helper_spec.name] = helper
    helper_spec.loader.exec_module(helper)
    evaluator = _load_existing_evaluator(args.existing_evaluator)
    base_checkpoint = args.base_checkpoint.resolve(strict=True)
    stage3_checkpoint = args.stage3_checkpoint.resolve(strict=True)
    profile_path = args.profile.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    seeds = tuple(int(value) for value in args.validation_seeds)
    if seeds != (30001, 30002, 30003):
        raise ValueError("The frozen main experiment uses Validation seeds 30001–30003 only.")
    validation_pool, pool_info = helper._validate_pool(
        formal_root,
        helper.VALIDATION_POOL_RELATIVE,
        expected_split="validation",
        expected_seed_start=30001,
        expected_episode_count=30,
    )
    for seed in seeds:
        helper._validate_episode_snapshot(validation_pool, seed, pool_info)
    profile, profile_meta = stage3_runner._load_stage3_coefficient_profile(
        profile_path,
        max_macro_states_per_rollout=4,
        max_target_nodes_per_state=2,
    )
    metadata = {
        "experiment": "MAPPO_STAGE3_MAIN_FROZEN_VALIDATION",
        "validation_pool_manifest_sha256": pool_info["manifest_sha256"],
        "validation_seeds": list(seeds),
        "evaluation_action_decode": (
            "operation-first conditional greedy for learned policies; forced legal action 0 "
            "for KEEP+STAY"
        ),
        "base_checkpoint_sha256": _sha256(base_checkpoint),
        "stage3_checkpoint_sha256": _sha256(stage3_checkpoint),
        "profile_sha256": _sha256(profile_path),
        "evaluation_adapter_sha256": _sha256(Path(__file__).resolve()),
        "stage3_profile": profile_meta,
        "existing_evaluator_path": str(args.existing_evaluator.resolve()),
        "existing_evaluator_sha256": _sha256(args.existing_evaluator.resolve(strict=True)),
        "validation_accessed": True,
        "test_accessed": False,
        "trainer_update_called": False,
    }
    (output / "evaluation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result_path = output / "validation_results.json"
    episodes: list[dict[str, Any]] = []

    for seed in seeds:
        print(json.dumps({"phase": "prepare_validation_episode", "seed": seed}), flush=True)
        context = helper._episode_context(formal_root, validation_pool, seed, pool_info)
        initial_cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
        initial_uav = {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in context.initial_uav.items()
        }
        aligned = helper._aligned_window(
            context, int(context.train_origin_start), int(context.episode_slots)
        )
        if len(aligned) < int(context.episode_slots):
            raise ValueError(f"Validation seed {seed} lacks a complete aligned Episode.")
        policies: dict[str, Any] = {}
        for name, checkpoint, stage3_config, keep_stay in (
            ("base_mappo_v2", base_checkpoint, None, False),
            ("stage3_mappo", stage3_checkpoint, profile, False),
            ("keep_stay", base_checkpoint, None, True),
        ):
            trainer = _stage3_compatible_trainer_class(evaluator)(
                context,
                device="cpu",
                seed=73101 + seed,
                keep_stay=keep_stay,
            )
            if stage3_config is not None:
                trainer.enable_stage3_counterfactual_cache_value(stage3_config)
            loaded = trainer.load_checkpoint(checkpoint)
            if not np.array_equal(initial_cache, context.initial_cache) or any(
                not np.array_equal(initial_uav[key], context.initial_uav[key])
                for key in initial_uav
            ):
                raise ValueError("The policy evaluation initial states differ.")
            trainer.generator.manual_seed(73101 + seed)
            started = time.perf_counter()
            with torch.inference_mode():
                rollout = trainer.collect_episode(
                    aligned,
                    max_slots=int(context.episode_slots),
                    terminated_at_end=True,
                    deterministic=True,
                    profile_slot_end_to_end=True,
                    episode_seed=seed,
                )
            rollout_seconds = time.perf_counter() - started
            metrics = evaluator._episode_metrics(rollout)
            metrics["rollout_runtime_seconds"] = float(rollout_seconds)
            metrics["checkpoint_update_count"] = int(loaded["update_count"])
            metrics["action_decode"] = (
                "forced KEEP at all Cache nodes and STAY at both UAVs"
                if keep_stay
                else "argmax legal operation group, then argmax legal conditional candidate"
            )
            if name == "stage3_mappo":
                label_started = time.perf_counter()
                metrics["stage3_value_diagnostics"] = _stage3_label_metrics(
                    trainer, rollout, profile=profile
                )
                metrics["stage3_value_diagnostics"]["wall_seconds"] = float(
                    time.perf_counter() - label_started
                )
            policies[name] = metrics
            partial_result = {
                **metadata,
                "result_status": "partial_policy_results",
                "episodes": episodes
                + [{"episode_seed": seed, "policies": dict(policies)}],
            }
            result_path.write_text(
                json.dumps(
                    partial_result,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            del trainer, rollout
            print(
                json.dumps(
                    {
                        "phase": "evaluate_policy",
                        "seed": seed,
                        "policy": name,
                        "net": metrics["actual_net_utility_total"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        episodes.append({"episode_seed": seed, "policies": policies})
        result_path.write_text(
            json.dumps(
                {**metadata, "episodes": episodes},
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    paired = [
        {
            "episode_seed": episode["episode_seed"],
            "stage3_minus_base": (
                episode["policies"]["stage3_mappo"]["actual_net_utility_total"]
                - episode["policies"]["base_mappo_v2"]["actual_net_utility_total"]
            ),
            "base_minus_keep_stay": (
                episode["policies"]["base_mappo_v2"]["actual_net_utility_total"]
                - episode["policies"]["keep_stay"]["actual_net_utility_total"]
            ),
            "stage3_minus_keep_stay": (
                episode["policies"]["stage3_mappo"]["actual_net_utility_total"]
                - episode["policies"]["keep_stay"]["actual_net_utility_total"]
            ),
        }
        for episode in episodes
    ]
    result = {
        **metadata,
        "episodes": episodes,
        "paired_differences": paired,
        "mean_stage3_minus_base": float(
            np.mean([item["stage3_minus_base"] for item in paired])
        ),
        "positive_stage3_minus_base_episode_count": int(
            sum(item["stage3_minus_base"] > 0.0 for item in paired)
        ),
        "constraints_and_reward_contract_valid": True,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "mean_stage3_minus_base": result["mean_stage3_minus_base"],
                "positive_episode_count": result["positive_stage3_minus_base_episode_count"],
                "output": str(result_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
