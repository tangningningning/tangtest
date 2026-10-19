#!/usr/bin/env python3
"""Prepare bounded paired Training labels and fit the global Cache value model.

This entry reads Training Pool only. `prepare` runs a fixed number of paired
formal Matching+KKT branches; `fit` trains only from its saved Training labels.
Neither mode reads Validation/Test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts" / "training",
):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from decision import cache_configuration, predictive_global_cache, slot_environment
from predictive_global_cache_data import training_context_and_steps


PROFILE_PATH = PROJECT_ROOT / "configs" / "research" / "predictive_global_cache_value_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_profile() -> tuple[dict[str, Any], str]:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if profile.get("schema") != "predictive_global_cache_experiment_v1":
        raise ValueError("unrecognized predictive Cache experiment profile")
    return profile, _sha256(PROFILE_PATH)


def _state_horizon(step: Any, remaining_steps: int, requested: int) -> int:
    raw = getattr(step, "prediction_horizon_mask", None)
    scenario_horizons = int(np.asarray(step.scenario_slice["arrival"]).shape[1])
    mask = cache_configuration.normalize_prediction_horizon_mask(raw, scenario_horizons)
    valid_count = 0
    for value in mask[: int(requested)]:
        if not bool(value):
            break
        valid_count += 1
    horizon = min(valid_count, int(remaining_steps), int(requested))
    if horizon <= 0:
        raise ValueError(f"no valid Training label horizon at origin slot {step.slot}")
    return horizon


def prepare_training_labels(
    *,
    output_path: Path,
    budget: predictive_global_cache.TrainingBudget,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Create a non-overwriting, manifest-linked paired Training label set."""

    budget.validate()
    profile, profile_sha256 = _load_profile()
    import mappo_stage3_pool_contract as pool_contract

    pool, pool_info = pool_contract._validate_pool(
        project_root.resolve(),
        pool_contract.TRAIN_POOL_RELATIVE,
        expected_split="train",
        expected_seed_start=pool_contract.DEFAULT_TRAINING_SEED_START,
        expected_episode_count=300,
    )
    pool_manifest = json.loads((pool / "formal_pool_manifest.json").read_text(encoding="utf-8"))
    first_seed = int(pool_manifest["seed_start"])
    episode_seeds = [first_seed + offset for offset in range(int(budget.episode_count))]
    state_budget_total = int(budget.episode_count) * int(budget.states_per_episode)
    # Non-overlapping label windows keep the grouped holdout from sharing the
    # same realized Training slots between adjacent states.
    requested_slots = int(budget.states_per_episode) * int(budget.horizon_slots)

    feature_rows: list[np.ndarray] = []
    label_rows: list[float] = []
    state_groups: list[str] = []
    episode_groups: list[str] = []
    state_slots: list[int] = []
    label_horizons: list[int] = []
    candidate_nodes: list[int] = []
    candidate_sources: list[str] = []
    target_caches: list[np.ndarray] = []
    pair_count = 0

    for episode_index, seed in enumerate(episode_seeds):
        context, steps, _ = training_context_and_steps(
            project_root.resolve(),
            int(seed),
            requested_slots,
        )
        current_cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
        uav_positions = {
            str(key): np.asarray(value, dtype=np.float64).copy()
            for key, value in context.initial_uav.items()
        }
        states_for_episode = min(int(budget.states_per_episode), len(steps) // int(budget.horizon_slots))
        for state_index in range(states_for_episode):
            remaining_pair_budget = int(budget.candidate_pair_budget) - pair_count
            remaining_states = state_budget_total - (episode_index * int(budget.states_per_episode) + state_index)
            if remaining_pair_budget <= 0:
                break
            state_pair_budget = min(
                int(budget.candidate_pair_budget),
                int(np.ceil(remaining_pair_budget / max(remaining_states, 1))),
            )
            origin_offset = state_index * int(budget.horizon_slots)
            step = steps[origin_offset]
            available_horizon = _state_horizon(
                step,
                len(steps) - origin_offset,
                int(budget.horizon_slots),
            )
            horizon_steps = steps[origin_offset : origin_offset + available_horizon]
            candidates = predictive_global_cache.bounded_single_node_candidates(
                current_cache,
                step.scenario_slice,
                step.prediction_horizon_mask,
                context.service_catalog,
                context.cache_nodes,
                context.cache_cfg,
                candidate_budget=state_pair_budget,
                candidates_per_node=int(budget.candidates_per_node),
                node_offset=episode_index * int(budget.states_per_episode) + state_index,
            )
            if not candidates:
                continue
            forecast_context = predictive_global_cache.prepare_global_forecast_features(
                step.scenario_slice,
                step.prediction_horizon_mask,
                uav_positions,
                context.service_catalog,
                context.cache_nodes,
                context.cache_cfg,
                context.cfg,
                context.flight_boundary,
            )
            keep_metrics = predictive_global_cache.evaluate_fixed_cache_training_horizon(
                context,
                horizon_steps,
                current_cache,
                current_cache,
                uav_positions,
                horizon_slots=available_horizon,
            )
            group_id = f"train_{int(seed)}_slot_{int(step.slot)}"
            for node_index, target_row, proposal_source in candidates:
                target_cache = current_cache.copy()
                target_cache[int(node_index)] = np.asarray(target_row, dtype=np.int8)
                features = predictive_global_cache.build_global_candidate_features(
                    current_cache,
                    target_cache,
                    int(node_index),
                    step.scenario_slice,
                    step.prediction_horizon_mask,
                    uav_positions,
                    context.service_catalog,
                    context.cache_nodes,
                    context.cache_cfg,
                    context.cfg,
                    context.flight_boundary,
                    forecast_context=forecast_context,
                )
                paired = predictive_global_cache.paired_training_delta_label(
                    context,
                    horizon_steps,
                    current_cache,
                    target_cache,
                    uav_positions,
                    horizon_slots=available_horizon,
                    keep_metrics=keep_metrics,
                )
                feature_rows.append(features)
                label_rows.append(float(paired["delta"]["net_utility_delta"]))
                state_groups.append(group_id)
                episode_groups.append(str(int(seed)))
                state_slots.append(int(step.slot))
                label_horizons.append(int(available_horizon))
                candidate_nodes.append(int(node_index))
                candidate_sources.append(str(proposal_source))
                target_caches.append(target_cache)
                pair_count += 1
                if pair_count > int(budget.candidate_pair_budget):
                    raise RuntimeError("paired label budget exceeded")
        if pair_count >= int(budget.candidate_pair_budget):
            break

    if len(feature_rows) < 5 or len(set(state_groups)) < 5:
        raise ValueError("fixed Training budget produced too few labeled states for fitting")
    if len(set(episode_groups)) < 7:
        raise ValueError(
            "episode-isolated fitting requires at least seven Training Episodes"
        )
    output = output_path.resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("label dataset output path must end in .npz")
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite label dataset or metadata: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    features_array = np.stack(feature_rows).astype(np.float32)
    labels_array = np.asarray(label_rows, dtype=np.float32)
    group_array = np.asarray(state_groups, dtype="U64")
    episode_group_array = np.asarray(episode_groups, dtype="U32")
    target_array = np.stack(target_caches).astype(np.int8)
    np.savez_compressed(
        output,
        features=features_array,
        targets=labels_array,
        state_groups=group_array,
        episode_groups=episode_group_array,
        state_slots=np.asarray(state_slots, dtype=np.int32),
        label_horizons=np.asarray(label_horizons, dtype=np.int8),
        candidate_node_indices=np.asarray(candidate_nodes, dtype=np.int16),
        candidate_sources=np.asarray(candidate_sources, dtype="U32"),
        target_caches=target_array,
    )
    dataset_sha256 = _sha256(output)
    near_zero_tolerance = 1.0e-6
    label_counts_by_node = {
        str(node_index): int(candidate_nodes.count(node_index))
        for node_index in sorted(set(candidate_nodes))
    }
    label_counts_by_episode = {
        str(seed): int(episode_groups.count(str(seed))) for seed in episode_seeds
    }
    metadata = {
        "algorithm_schema": predictive_global_cache.ALGORITHM_SCHEMA,
        "feature_schema": predictive_global_cache.FEATURE_SCHEMA,
        "label_schema": predictive_global_cache.LABEL_SCHEMA,
        "search_schema": predictive_global_cache.SEARCH_SCHEMA,
        "label_source_split": "train",
        "split_schema": "episode_grouped_fit_early_stop_calibration_v1",
        "training_pool_manifest_sha256": pool_info["manifest_sha256"],
        "training_pool_manifest_path": str(Path(pool_info["manifest_path"]).resolve()),
        "config_sha256": pool_info["runtime_config_sha256"],
        "loss_profile_sha256": profile_sha256,
        "episode_seeds": episode_seeds,
        "state_slots": state_slots,
        "state_window_selection": "fixed chronological starts spaced by label_horizon_slots; label windows do not overlap within an Episode",
        "state_group_count": int(len(set(state_groups))),
        "episode_group_count": int(len(set(episode_groups))),
        "candidate_pair_count": int(len(label_rows)),
        "candidate_pair_budget": int(budget.candidate_pair_budget),
        "dataset_sha256": dataset_sha256,
        "horizon_slots_per_label": label_horizons,
        "candidate_source_policy": "existing legal prediction/coverage/intermediate anchors and one-add/delete/swap neighbors; bounded node-balanced subset",
        "candidate_pair_labels": "candidate-minus-KEEP sum of formal system_utility over the label horizon",
        "label_continuation_policy": "hold each branch Cache fixed; keep UAV at decision-time position; no terminal bootstrap; cache reconfiguration cost is paid only on the first branch slot",
        "future_training_tasks_used_only_for_labels": True,
        "future_tasks_used_in_features": False,
        "validation_accessed": False,
        "test_accessed": False,
        "feature_width": int(features_array.shape[1]),
        "service_count": int(target_array.shape[-1]),
        "candidate_node_indices": candidate_nodes,
        "candidate_sources": candidate_sources,
        "candidate_labels_by_node": label_counts_by_node,
        "candidate_labels_by_episode": label_counts_by_episode,
        "label_sign_counts": {
            "positive": int(np.count_nonzero(labels_array > near_zero_tolerance)),
            "negative": int(np.count_nonzero(labels_array < -near_zero_tolerance)),
            "near_zero": int(np.count_nonzero(np.abs(labels_array) <= near_zero_tolerance)),
            "near_zero_tolerance": near_zero_tolerance,
        },
        "fixed_budget": asdict_training_budget(budget),
        "profile": profile,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "dataset_path": str(output),
        "dataset_sha256": dataset_sha256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "sample_count": int(len(label_rows)),
        "state_group_count": int(len(set(state_groups))),
        "episode_group_count": int(len(set(episode_groups))),
        "candidate_labels_by_node": label_counts_by_node,
        "candidate_labels_by_episode": label_counts_by_episode,
        "label_sign_counts": metadata["label_sign_counts"],
        "episode_seeds": episode_seeds,
        "pool_manifest_sha256": pool_info["manifest_sha256"],
    }


def prepare_state_coverage_training_labels(
    *,
    output_path: Path,
    episode_seeds: Sequence[int],
    excluded_episode_ids: Sequence[int],
    max_candidate_pairs: int = 128,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Collect bounded labels at initial and formally reached noninitial Cache states.

    For each Training Episode, a legal prediction-derived one-node update is
    executed at the first Macro boundary and then held while the formal slot
    executor advances to the second sampled state. The added state therefore
    comes from an executed transition, not a constructed Cache matrix.
    """

    seeds = [int(value) for value in episode_seeds]
    excluded = {int(value) for value in excluded_episode_ids}
    if not seeds or len(set(seeds)) != len(seeds) or any(value <= 0 for value in seeds):
        raise ValueError("state-coverage Training Episode seeds must be unique positive IDs")
    if set(seeds) & excluded:
        raise ValueError("state-coverage Episodes overlap the excluded fit/holdout/diagnostic IDs")
    if not 1 <= int(max_candidate_pairs) <= 128:
        raise ValueError("state-coverage candidate-pair budget must be between 1 and 128")
    output = output_path.resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("state-coverage label output must end in .npz")
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite state-coverage labels: {output}")

    sample_offsets = (0, 6)
    candidates_per_state = 8
    profile, profile_sha256 = _load_profile()
    profile_horizon = int(profile["label_horizon_slots"])
    requested_slots = sample_offsets[-1] + profile_horizon
    feature_rows: list[np.ndarray] = []
    label_rows: list[float] = []
    state_groups: list[str] = []
    episode_groups: list[str] = []
    state_slots: list[int] = []
    label_horizons: list[int] = []
    candidate_nodes: list[int] = []
    candidate_sources: list[str] = []
    target_caches: list[np.ndarray] = []
    current_caches: list[np.ndarray] = []
    state_is_initial_rows: list[bool] = []
    state_cache_hash_rows: list[str] = []
    state_records: list[dict[str, Any]] = []
    candidate_counts_by_node: dict[str, int] = {}
    candidate_counts_by_role: dict[str, int] = {}
    pair_count = 0
    pool_info: dict[str, Any] | None = None

    def empty_lock_mask(seed: int, cache: np.ndarray, context: Any) -> np.ndarray:
        # The formal predictive-Cache executor evaluates each Slot independently
        # and carries no active task ledger. Constructing its empty ledger makes
        # the service-lock contract explicit for these decision states.
        ledger = slot_environment.CrossSlotTaskLedger(episode_id=f"train_{seed}")
        return slot_environment.active_cache_service_locks(
            ledger,
            cache,
            context.service_catalog,
            context.cache_nodes,
        )

    for episode_index, seed in enumerate(seeds):
        context, steps, loaded_pool_info = training_context_and_steps(
            project_root.resolve(), seed, requested_slots
        )
        if pool_info is None:
            pool_info = loaded_pool_info
        elif pool_info["manifest_sha256"] != loaded_pool_info["manifest_sha256"]:
            raise ValueError("Training Episodes resolved to different Pool Manifests")
        initial_cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
        current_cache = initial_cache.copy()
        initial_uav = {
            str(key): np.asarray(value, dtype=np.float64).copy()
            for key, value in context.initial_uav.items()
        }
        uav_positions = {key: value.copy() for key, value in initial_uav.items()}
        transition_node: int | None = None
        transition_cache_hash: str | None = None

        for state_index, offset in enumerate(sample_offsets):
            if pair_count >= int(max_candidate_pairs):
                break
            step = steps[offset]
            locks = empty_lock_mask(seed, current_cache, context)
            remaining_budget = int(max_candidate_pairs) - pair_count
            state_pair_budget = min(candidates_per_state, remaining_budget)
            proposals = predictive_global_cache.bounded_single_node_candidates(
                current_cache,
                step.scenario_slice,
                step.prediction_horizon_mask,
                context.service_catalog,
                context.cache_nodes,
                context.cache_cfg,
                candidate_budget=state_pair_budget,
                candidates_per_node=1,
                node_offset=episode_index * len(sample_offsets) + state_index,
            )
            legal_proposals = [
                (int(node), np.asarray(row, dtype=np.int8).copy(), str(source))
                for node, row, source in proposals
                if not np.any(locks[int(node)] & (current_cache[int(node)] != np.asarray(row, dtype=np.int8)))
            ]
            if not legal_proposals:
                raise RuntimeError(
                    f"no legal state-coverage candidates for Training seed {seed}, Slot {step.slot}"
                )
            horizon = _state_horizon(step, len(steps) - offset, profile_horizon)
            horizon_steps = steps[offset : offset + horizon]
            forecast_context = predictive_global_cache.prepare_global_forecast_features(
                step.scenario_slice,
                step.prediction_horizon_mask,
                uav_positions,
                context.service_catalog,
                context.cache_nodes,
                context.cache_cfg,
                context.cfg,
                context.flight_boundary,
            )
            keep_metrics = predictive_global_cache.evaluate_fixed_cache_training_horizon(
                context,
                horizon_steps,
                current_cache,
                current_cache,
                uav_positions,
                horizon_slots=horizon,
            )
            is_initial = bool(np.array_equal(current_cache, initial_cache))
            state_cache_hash = hashlib.sha256(
                np.ascontiguousarray(current_cache).tobytes()
            ).hexdigest()
            state_group = f"train_{seed}_slot_{int(step.slot)}_{'initial' if is_initial else 'noninitial'}"
            state_records.append(
                {
                    "episode_seed": seed,
                    "slot": int(step.slot),
                    "state_group": state_group,
                    "cache_is_initial": is_initial,
                    "current_cache_sha256": state_cache_hash,
                    "candidate_count": len(legal_proposals),
                    "transition_node_from_initial": transition_node,
                    "transition_cache_sha256_from_initial": transition_cache_hash,
                    "horizon_slots": horizon,
                }
            )
            for node_index, target_row, proposal_source in legal_proposals:
                target_cache = current_cache.copy()
                target_cache[node_index] = target_row
                cache_configuration.validate_cache_shape_and_capacity(
                    target_cache, context.service_catalog, context.cache_nodes
                )
                if np.any(locks[node_index] & (current_cache[node_index] != target_row)):
                    raise ValueError("candidate update would evict a service locked by the formal task ledger")
                features = predictive_global_cache.build_global_candidate_features(
                    current_cache,
                    target_cache,
                    node_index,
                    step.scenario_slice,
                    step.prediction_horizon_mask,
                    uav_positions,
                    context.service_catalog,
                    context.cache_nodes,
                    context.cache_cfg,
                    context.cfg,
                    context.flight_boundary,
                    forecast_context=forecast_context,
                )
                paired = predictive_global_cache.paired_training_delta_label(
                    context,
                    horizon_steps,
                    current_cache,
                    target_cache,
                    uav_positions,
                    horizon_slots=horizon,
                    keep_metrics=keep_metrics,
                )
                feature_rows.append(features)
                label_rows.append(float(paired["delta"]["net_utility_delta"]))
                state_groups.append(state_group)
                episode_groups.append(str(seed))
                state_slots.append(int(step.slot))
                label_horizons.append(horizon)
                candidate_nodes.append(node_index)
                candidate_sources.append(proposal_source)
                target_caches.append(target_cache)
                current_caches.append(current_cache.copy())
                state_is_initial_rows.append(is_initial)
                state_cache_hash_rows.append(state_cache_hash)
                node_id = str(context.cache_nodes["node_ids"][node_index])
                role = str(context.cache_nodes["node_roles"][node_index]).upper()
                candidate_counts_by_node[node_id] = candidate_counts_by_node.get(node_id, 0) + 1
                candidate_counts_by_role[role] = candidate_counts_by_role.get(role, 0) + 1
                pair_count += 1
                if pair_count > int(max_candidate_pairs):
                    raise RuntimeError("state-coverage Training label budget exceeded")

            if offset == sample_offsets[0] and state_index + 1 < len(sample_offsets):
                # Choose the state-transition update using only the current
                # forecast-derived legal candidates, then execute it formally.
                all_nodes = len(context.cache_nodes["node_ids"])
                transition_proposals = predictive_global_cache.bounded_single_node_candidates(
                    current_cache,
                    step.scenario_slice,
                    step.prediction_horizon_mask,
                    context.service_catalog,
                    context.cache_nodes,
                    context.cache_cfg,
                    candidate_budget=all_nodes,
                    candidates_per_node=1,
                    node_offset=episode_index,
                )
                proposals_by_node = {int(node): (np.asarray(row, dtype=np.int8), str(source)) for node, row, source in transition_proposals}
                desired_node = episode_index % max(all_nodes, 1)
                ordered_nodes = [(desired_node + shift) % all_nodes for shift in range(all_nodes)]
                selected = next(
                    ((node, proposals_by_node[node]) for node in ordered_nodes if node in proposals_by_node and not np.any(locks[node] & (current_cache[node] != proposals_by_node[node][0]))),
                    None,
                )
                if selected is None:
                    raise RuntimeError(f"no unlocked legal Cache transition for Training seed {seed}")
                transition_node, (transition_row, _transition_source) = selected
                transition_target = current_cache.copy()
                transition_target[transition_node] = transition_row
                cache_configuration.validate_cache_shape_and_capacity(
                    transition_target, context.service_catalog, context.cache_nodes
                )
                if np.array_equal(transition_target, current_cache):
                    raise RuntimeError("selected legal Cache transition did not change state")

                # Advance the real Training trajectory through Slots 0..5 with
                # this one update and then KEEP. The later state is the formal
                # executor's output, never a hand-built Cache matrix.
                for rollout_offset in range(offset, sample_offsets[state_index + 1]):
                    rollout_step = steps[rollout_offset]
                    slot_target = transition_target if rollout_offset == offset else current_cache.copy()
                    result = slot_environment.execute_configured_slot(
                        current_cache,
                        uav_positions,
                        slot_target,
                        uav_positions,
                        rollout_step.scenario_slice,
                        rollout_step.actual_tasks,
                        rollout_step.actual_vehicle_positions,
                        context.service_catalog,
                        context.cfg,
                        context.cache_nodes,
                        context.cache_cfg,
                        context.flight_boundary,
                        slot_id=int(rollout_step.slot),
                        time_s=float(rollout_step.time_s),
                    )
                    if not result.step_feasible or result.system_utility is None:
                        raise RuntimeError(
                            f"formal Cache state transition failed at seed {seed}, Slot {rollout_step.slot}: {result.infeasible_reason}"
                        )
                    if not math.isfinite(float(result.system_utility)):
                        raise FloatingPointError("formal state-transition reward is non-finite")
                    expected_cache = np.asarray(slot_target, dtype=np.int8)
                    next_cache = np.asarray(result.next_cache_state, dtype=np.int8)
                    cache_configuration.validate_cache_shape_and_capacity(
                        next_cache, context.service_catalog, context.cache_nodes
                    )
                    if not np.array_equal(next_cache, expected_cache):
                        raise RuntimeError("formal executor returned a different Cache state than the feasible target")
                    next_uav = {
                        str(key): np.asarray(value, dtype=np.float64).copy()
                        for key, value in result.next_uav_positions.items()
                    }
                    if set(next_uav) != set(uav_positions) or any(
                        not np.array_equal(next_uav[key], uav_positions[key]) for key in uav_positions
                    ):
                        raise RuntimeError("state-coverage rollout unexpectedly moved a UAV")
                    current_cache = next_cache
                    uav_positions = next_uav
                if np.array_equal(current_cache, initial_cache):
                    raise RuntimeError("formal Cache transition did not create a noninitial state")
                transition_cache_hash = hashlib.sha256(
                    np.ascontiguousarray(current_cache).tobytes()
                ).hexdigest()

    if pair_count == 0 or not state_records:
        raise ValueError("state-coverage budget produced no usable Training labels")
    labels = np.asarray(label_rows, dtype=np.float32)
    near_zero_tolerance = 1.0e-6
    state_count_by_initial = {
        "initial": sum(bool(row["cache_is_initial"]) for row in state_records),
        "noninitial": sum(not bool(row["cache_is_initial"]) for row in state_records),
    }
    unique_noninitial_states = len(
        {
            str(row["current_cache_sha256"])
            for row in state_records
            if not bool(row["cache_is_initial"])
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.stack(feature_rows).astype(np.float32),
        targets=labels,
        state_groups=np.asarray(state_groups, dtype="U96"),
        episode_groups=np.asarray(episode_groups, dtype="U32"),
        state_slots=np.asarray(state_slots, dtype=np.int32),
        label_horizons=np.asarray(label_horizons, dtype=np.int8),
        candidate_node_indices=np.asarray(candidate_nodes, dtype=np.int16),
        candidate_sources=np.asarray(candidate_sources, dtype="U32"),
        target_caches=np.stack(target_caches).astype(np.int8),
        current_caches=np.stack(current_caches).astype(np.int8),
        cache_state_is_initial=np.asarray(state_is_initial_rows, dtype=np.bool_),
        cache_state_sha256=np.asarray(state_cache_hash_rows, dtype="U64"),
    )
    dataset_sha256 = _sha256(output)
    metadata = {
        "algorithm_schema": predictive_global_cache.ALGORITHM_SCHEMA,
        "feature_schema": predictive_global_cache.FEATURE_SCHEMA,
        "label_schema": predictive_global_cache.LABEL_SCHEMA,
        "search_schema": predictive_global_cache.SEARCH_SCHEMA,
        "label_source_split": "train",
        "training_pool_manifest_sha256": pool_info["manifest_sha256"],
        "config_sha256": pool_info["runtime_config_sha256"],
        "loss_profile_sha256": profile_sha256,
        "episode_seeds": seeds,
        "excluded_episode_ids": sorted(excluded),
        "state_slots": state_slots,
        "state_group_count": len(state_records),
        "episode_group_count": len(set(episode_groups)),
        "candidate_pair_count": pair_count,
        "candidate_pair_budget": int(max_candidate_pairs),
        "dataset_sha256": dataset_sha256,
        "horizon_slots_per_label": label_horizons,
        "candidate_pair_labels": "candidate-minus-KEEP sum of formal system_utility over H<=5 Training slots",
        "label_continuation_policy": "hold candidate Cache fixed; UAV STAY; no terminal bootstrap; Cache reconfiguration cost paid only on first branch Slot",
        "future_training_tasks_used_only_for_labels": True,
        "future_tasks_used_in_features": False,
        "validation_accessed": False,
        "test_accessed": False,
        "feature_width": int(feature_rows[0].shape[0]),
        "service_count": int(target_caches[0].shape[-1]),
        "candidate_labels_by_node_id": candidate_counts_by_node,
        "candidate_labels_by_role": candidate_counts_by_role,
        "label_sign_counts": {
            "positive": int(np.count_nonzero(labels > near_zero_tolerance)),
            "negative": int(np.count_nonzero(labels < -near_zero_tolerance)),
            "near_zero": int(np.count_nonzero(np.abs(labels) <= near_zero_tolerance)),
            "near_zero_tolerance": near_zero_tolerance,
        },
        "cache_state_counts": state_count_by_initial,
        "label_rows_by_cache_state": {
            "initial": int(sum(state_is_initial_rows)),
            "noninitial": int(len(state_is_initial_rows) - sum(state_is_initial_rows)),
        },
        "unique_noninitial_cache_configurations": unique_noninitial_states,
        "state_generation_contract": "Slot 0 initial state; apply one forecast-derived legal single-node update through formal execute_configured_slot, then KEEP through Slot 5; sample resulting noninitial state at Slot 6",
        "state_records": state_records,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "dataset_path": str(output),
        "dataset_sha256": dataset_sha256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "sample_count": pair_count,
        "state_group_count": len(state_records),
        "cache_state_counts": state_count_by_initial,
        "label_rows_by_cache_state": metadata["label_rows_by_cache_state"],
        "unique_noninitial_cache_configurations": unique_noninitial_states,
        "candidate_labels_by_node_id": candidate_counts_by_node,
        "candidate_labels_by_role": candidate_counts_by_role,
        "label_sign_counts": metadata["label_sign_counts"],
        "episode_seeds": seeds,
        "pool_manifest_sha256": pool_info["manifest_sha256"],
    }


def asdict_training_budget(budget: predictive_global_cache.TrainingBudget) -> dict[str, int]:
    return {
        "episode_count": int(budget.episode_count),
        "states_per_episode": int(budget.states_per_episode),
        "candidate_pair_budget": int(budget.candidate_pair_budget),
        "horizon_slots": int(budget.horizon_slots),
        "candidates_per_node": int(budget.candidates_per_node),
    }


def fit_from_training_dataset(
    dataset_path: Path,
    metadata_path: Path,
    checkpoint_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    profile, profile_sha256 = _load_profile()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("label_source_split") != "train" or metadata.get("test_accessed"):
        raise ValueError("value fitting accepts Training-only labels")
    if metadata.get("split_schema") != "episode_grouped_fit_early_stop_calibration_v1":
        raise ValueError("label dataset does not use the episode-isolated split contract")
    if metadata.get("loss_profile_sha256") != profile_sha256:
        raise ValueError("label dataset was built under a different research profile")
    dataset_path = dataset_path.resolve()
    if metadata.get("dataset_sha256") != _sha256(dataset_path):
        raise ValueError("Training label dataset hash does not match its metadata")
    import mappo_stage3_pool_contract as pool_contract

    _pool, current_pool_info = pool_contract._validate_pool(
        project_root.resolve(),
        pool_contract.TRAIN_POOL_RELATIVE,
        expected_split="train",
        expected_seed_start=pool_contract.DEFAULT_TRAINING_SEED_START,
        expected_episode_count=300,
    )
    if metadata.get("training_pool_manifest_sha256") != current_pool_info["manifest_sha256"]:
        raise ValueError("Training label dataset was built from a different pool manifest")
    if metadata.get("config_sha256") != cache_configuration.project_config.raw_config_sha256(
        project_root / "configs" / "proposed" / "system_config.toml"
    ):
        raise ValueError("label dataset config hash does not match the current formal system config")
    with np.load(dataset_path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        targets = np.asarray(data["targets"], dtype=np.float32)
        groups = np.asarray(data["episode_groups"]).astype(str)
    if sorted(set(groups.tolist())) != sorted(str(seed) for seed in metadata["episode_seeds"]):
        raise ValueError("episode group IDs do not match label dataset provenance")
    settings = profile["value_model"]
    result = predictive_global_cache.fit_value_model(
        features,
        targets,
        groups,
        output_path=checkpoint_path,
        pool_manifest_sha256=str(metadata["training_pool_manifest_sha256"]),
        config_sha256=str(metadata["config_sha256"]),
        profile_sha256=profile_sha256,
        training_dataset_sha256=_sha256(dataset_path),
        service_count=int(metadata["service_count"]),
        label_horizon_slots=int(profile["label_horizon_slots"]),
        seed=int(settings["training_seed"]),
        epochs=int(settings["max_epochs"]),
        batch_size=int(settings["batch_size"]),
        hidden_width=int(settings["hidden_width"]),
        learning_rate=float(settings["learning_rate"]),
        early_stopping_patience=int(settings["early_stopping_patience"]),
        calibration_quantile=float(settings["calibration_quantile"]),
        minimum_calibration_episodes=int(settings["minimum_calibration_episodes"]),
        calibration_minimum_rows=int(settings["calibration_minimum_rows"]),
        episode_partition_map=metadata.get("episode_partition_map"),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    profile, _profile_sha256 = _load_profile()
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="generate bounded paired Training labels")
    prepare.add_argument("--output", type=Path, required=True, help="new .npz label dataset path")
    state_coverage = subparsers.add_parser(
        "prepare-state-coverage",
        help="generate bounded labels at initial and formally reached noninitial Training Cache states",
    )
    state_coverage.add_argument("--output", type=Path, required=True)
    state_coverage.add_argument("--episode-seeds", type=int, nargs="+", required=True)
    state_coverage.add_argument("--excluded-episode-ids", type=int, nargs="*", default=[])
    state_coverage.add_argument("--max-candidate-pairs", type=int, default=128)
    fit = subparsers.add_parser("fit", help="fit from an existing Training-only label dataset")
    fit.add_argument("--dataset", type=Path, required=True)
    fit.add_argument("--metadata", type=Path, required=True)
    fit.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        label_budget = profile["training_label_budget"]
        result = prepare_training_labels(
            output_path=args.output,
            budget=predictive_global_cache.TrainingBudget(
                episode_count=int(label_budget["episode_count"]),
                states_per_episode=int(label_budget["states_per_episode"]),
                candidate_pair_budget=int(label_budget["candidate_pair_budget"]),
                horizon_slots=int(profile["label_horizon_slots"]),
                candidates_per_node=int(label_budget["candidates_per_node"]),
            ),
        )
    elif args.command == "prepare-state-coverage":
        result = prepare_state_coverage_training_labels(
            output_path=args.output,
            episode_seeds=args.episode_seeds,
            excluded_episode_ids=args.excluded_episode_ids,
            max_candidate_pairs=args.max_candidate_pairs,
        )
    else:
        result = fit_from_training_dataset(args.dataset.resolve(), args.metadata.resolve(), args.checkpoint.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

