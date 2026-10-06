"""V16-R replay-only trigger-state identifiability diagnostic.

The sole input is the saved V16-A trigger replay dataset. This script does not
construct an environment or load a teacher/formal training checkpoint.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "runs/formal_v16a_direct_trigger_distillation_gate/trigger_replay_dataset.npz"
REPORT_PATH = ROOT / "reports/v16r_trigger_state_identifiability_root_cause.json"
EXPECTED_FIELDS = {
    "normalized_trigger_observation",
    "raw_delta",
    "se_update",
    "active",
    "episode_seed",
    "macro_index",
    "realization_id",
}
BASE_SEED = 27005
EPOCHS = 200
LEARNING_RATE = 1e-3
HUBER_DELTA = 1.0
GRAD_CLIP_NORM = 0.5
NEAR_DUPLICATE_L2_MAX = 1e-3
NEAR_DUPLICATE_COSINE_SIM_MIN = 0.999999


class TriggerRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(400, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


def json_number(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return json_number(spearmanr(x, y).statistic)


def sign_accuracy(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) == 0:
        return None
    return float(np.mean(np.sign(a) == np.sign(b)))


def robust_scale(delta: np.ndarray) -> float:
    center = float(np.median(delta))
    scale = float(1.4826 * np.median(np.abs(delta - center)))
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(delta))
    return scale if math.isfinite(scale) and scale > 1e-12 else 1.0


def grouped_fold_assignment(groups: np.ndarray, n_splits: int = 5):
    """Non-shuffled GroupKFold-equivalent greedy sample-balanced assignment."""
    group_ids, counts = np.unique(groups, return_counts=True)
    if len(group_ids) < n_splits:
        raise ValueError(f"need >= {n_splits} groups, got {len(group_ids)}")
    # GroupKFold's non-shuffled strategy balances rows by placing larger groups
    # first into the currently smallest fold. Stable ties make this reproducible.
    order = np.argsort(-counts, kind="stable")
    fold_sizes = np.zeros(n_splits, dtype=np.int64)
    group_to_fold: dict[int, int] = {}
    for position in order:
        fold = int(np.argmin(fold_sizes))
        group = int(group_ids[position])
        group_to_fold[group] = fold
        fold_sizes[fold] += int(counts[position])
    splits = []
    for fold in range(n_splits):
        test_mask = np.fromiter((group_to_fold[int(g)] == fold for g in groups), bool, len(groups))
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)
        train_groups = set(groups[train_idx].tolist())
        test_groups = set(groups[test_idx].tolist())
        if train_groups & test_groups:
            raise AssertionError("GroupKFold leakage detected")
        splits.append((train_idx, test_idx, sorted(test_groups)))
    return splits, group_to_fold, fold_sizes.tolist()


def fit_and_score(
    x: np.ndarray,
    delta: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> dict:
    train_group_ids = np.unique(groups[train_idx])
    scale = robust_scale(delta[train_idx])
    y_all = delta / scale  # robust-scale only; retain DELTA's zero/sign origin.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    model = TriggerRegressor().to(device="cpu", dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device="cpu")
    y_tensor = torch.as_tensor(y_all, dtype=torch.float32, device="cpu")
    train_tensor = torch.as_tensor(train_idx, dtype=torch.long)
    group_masks = [
        torch.as_tensor(groups[train_idx] == group_id, dtype=torch.bool)
        for group_id in train_group_ids
    ]
    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_tensor[train_tensor])
        train_target = y_tensor[train_tensor]
        group_losses = [
            F.huber_loss(prediction[mask], train_target[mask], delta=HUBER_DELTA)
            for mask in group_masks
        ]
        loss = torch.stack(group_losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        prediction = model(x_tensor).cpu().numpy().astype(np.float64)
    return {
        "train_spearman": spearman(delta[train_idx], prediction[train_idx]),
        "test_spearman": spearman(delta[test_idx], prediction[test_idx]),
        "test_sign_accuracy": sign_accuracy(delta[test_idx], prediction[test_idx]),
        "train_prediction": prediction[train_idx],
        "test_prediction": prediction[test_idx],
        "scale": scale,
    }


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    finite = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=np.float64)
    if not len(finite):
        return None, None
    return float(np.mean(finite)), float(np.std(finite, ddof=0))


def pair_statistics(delta: np.ndarray, query: np.ndarray, neighbor: np.ndarray) -> dict:
    query_delta = delta[query]
    neighbor_delta = delta[neighbor]
    return {
        "pair_count": int(len(query)),
        "delta_sign_agreement": sign_accuracy(query_delta, neighbor_delta),
        "mean_absolute_delta_difference": json_number(np.mean(np.abs(query_delta - neighbor_delta)))
        if len(query) else None,
        "delta_spearman": spearman(query_delta, neighbor_delta),
    }


def nearest_cross_realization(
    x: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = cdist(x.astype(np.float64), x.astype(np.float64), metric="euclidean")
    distances[groups[:, None] == groups[None, :]] = np.inf
    neighbor = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(groups)), neighbor]
    if not np.isfinite(nearest_distance).all():
        raise AssertionError("some samples have no cross-realization neighbor")
    return neighbor, nearest_distance, distances


def duplicate_witnesses(
    x: np.ndarray,
    delta: np.ndarray,
    groups: np.ndarray,
    macro_index: np.ndarray,
    distances: np.ndarray,
) -> tuple[int, int, list[dict]]:
    upper = np.triu(np.ones(distances.shape, dtype=bool), k=1)
    cross = groups[:, None] != groups[None, :]
    exact = upper & cross & (distances <= 1e-12)
    candidates = np.argwhere(upper & cross & (distances <= NEAR_DUPLICATE_L2_MAX))
    if len(candidates):
        x64 = x.astype(np.float64)
        conflict_chunks = []
        for start in range(0, len(candidates), 8192):
            chunk = candidates[start : start + 8192]
            i_chunk, j_chunk = chunk[:, 0], chunk[:, 1]
            dot = np.einsum("ij,ij->i", x64[i_chunk], x64[j_chunk])
            norm_a = np.linalg.norm(x64[i_chunk], axis=1)
            norm_b = np.linalg.norm(x64[j_chunk], axis=1)
            denominator = norm_a * norm_b
            cosine = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
            cosine[(norm_a == 0) & (norm_b == 0)] = 1.0
            cosine = np.clip(cosine, -1.0, 1.0)
            near_candidate = cosine >= NEAR_DUPLICATE_COSINE_SIM_MIN
            sign_conflict = (np.sign(delta[i_chunk]) != np.sign(delta[j_chunk]))
            sign_conflict &= (delta[i_chunk] != 0) & (delta[j_chunk] != 0)
            conflict_chunks.append(chunk[near_candidate & sign_conflict])
        nonempty_chunks = [chunk for chunk in conflict_chunks if len(chunk)]
        conflicting_pairs = np.concatenate(nonempty_chunks, axis=0) if nonempty_chunks else np.empty((0, 2), dtype=np.int64)
    else:
        conflicting_pairs = np.empty((0, 2), dtype=np.int64)
    ordered = sorted(conflicting_pairs.tolist(), key=lambda pair: float(distances[pair[0], pair[1]]))[:20]
    witnesses = []
    for i, j in ordered:
        witnesses.append({
            "realization_a": int(groups[i]),
            "realization_b": int(groups[j]),
            "macro_index_a": int(macro_index[i]),
            "macro_index_b": int(macro_index[j]),
            "observation_distance": float(distances[i, j]),
            "delta_a": float(delta[i]),
            "delta_b": float(delta[j]),
            "sign_conflict": True,
        })
    return int(exact.sum()), int(len(conflicting_pairs)), witnesses


def main() -> None:
    print("V16-R: loading saved replay dataset", flush=True)
    with np.load(DATASET_PATH, allow_pickle=False) as dataset:
        fields = list(dataset.files)
        missing = sorted(EXPECTED_FIELDS - set(fields))
        if missing:
            raise ValueError(f"dataset missing required fields: {missing}")
        x_all = dataset["normalized_trigger_observation"]
        delta_all = dataset["raw_delta"]
        se_all = dataset["se_update"]
        active_all = dataset["active"].astype(bool)
        episode_seed_all = dataset["episode_seed"]
        macro_all = dataset["macro_index"]
        realization_all = dataset["realization_id"]

    lengths = {len(x_all), len(delta_all), len(se_all), len(active_all), len(episode_seed_all), len(macro_all), len(realization_all)}
    if len(lengths) != 1:
        raise ValueError("dataset fields have inconsistent row counts")
    if x_all.ndim != 2 or x_all.shape[1] != 400:
        raise ValueError(f"expected normalized 400D input, got {x_all.shape}")
    finite_checks = {
        "normalized_trigger_observation": bool(np.isfinite(x_all).all()),
        "raw_delta": bool(np.isfinite(delta_all).all()),
        "se_update": bool(np.isfinite(se_all).all()),
    }
    if not all(finite_checks.values()):
        raise ValueError(f"non-finite dataset values: {finite_checks}")

    active_idx = np.flatnonzero(active_all)
    x = np.asarray(x_all[active_idx], dtype=np.float32)
    delta = np.asarray(delta_all[active_idx], dtype=np.float64)
    se = np.asarray(se_all[active_idx], dtype=np.float64)
    groups = np.asarray(realization_all[active_idx], dtype=np.int64)
    macro_index = np.asarray(macro_all[active_idx], dtype=np.int64)
    episode_seed = np.asarray(episode_seed_all[active_idx], dtype=np.int64)
    group_ids = np.unique(groups)
    if len(group_ids) != 20 or len(active_idx) != 1511:
        raise ValueError(f"unexpected dataset size: groups={len(group_ids)}, active={len(active_idx)}")

    folds, group_to_fold, fold_sizes = grouped_fold_assignment(groups, n_splits=5)
    print(f"V16-R: {len(group_ids)} groups, {len(active_idx)} active samples; fitting 5 grouped folds", flush=True)
    fold_results = []
    fold_train_ranks = []
    fold_test_ranks = []
    fold_test_signs = []
    fold_scales = []
    for fold_number, (train_idx, test_idx, test_group_ids) in enumerate(folds):
        print(f"V16-R: fitting grouped fold {fold_number + 1}/5", flush=True)
        train_group_ids = np.unique(groups[train_idx])
        if set(train_group_ids.tolist()) & set(test_group_ids):
            raise AssertionError("group leakage in fold")
        result = fit_and_score(x, delta, groups, train_idx, test_idx, BASE_SEED + fold_number)
        fold_train_ranks.append(result["train_spearman"])
        fold_test_ranks.append(result["test_spearman"])
        fold_test_signs.append(result["test_sign_accuracy"])
        fold_scales.append(result["scale"])
        fold_results.append({
            "fold": fold_number,
            "train_realizations": [int(v) for v in train_group_ids],
            "test_realizations": [int(v) for v in test_group_ids],
            "train_sample_count": int(len(train_idx)),
            "test_sample_count": int(len(test_idx)),
            "train_spearman": result["train_spearman"],
            "test_spearman": result["test_spearman"],
            "test_sign_accuracy": result["test_sign_accuracy"],
            "train_delta_robust_scale": result["scale"],
        })
        print(f"V16-R: grouped fold {fold_number + 1}/5 complete", flush=True)
    train_mean, train_std = mean_std(fold_train_ranks)
    test_mean, test_std = mean_std(fold_test_ranks)
    test_sign_mean, test_sign_std = mean_std(fold_test_signs)

    split_order = np.random.default_rng(BASE_SEED).permutation(group_ids)
    curve_results = []
    curve_train = []
    curve_heldout = []
    for training_group_count in (4, 8, 12, 16):
        print(f"V16-R: fitting learning curve with {training_group_count} train groups", flush=True)
        training_groups = split_order[:training_group_count]
        heldout_groups = split_order[training_group_count:]
        train_idx = np.flatnonzero(np.isin(groups, training_groups))
        heldout_idx = np.flatnonzero(np.isin(groups, heldout_groups))
        if set(groups[train_idx].tolist()) & set(groups[heldout_idx].tolist()):
            raise AssertionError("learning-curve group leakage detected")
        result = fit_and_score(x, delta, groups, train_idx, heldout_idx, BASE_SEED + 100 + training_group_count)
        curve_train.append(result["train_spearman"])
        curve_heldout.append(result["test_spearman"])
        curve_results.append({
            "train_realization_count": training_group_count,
            "train_realizations": [int(v) for v in training_groups],
            "heldout_realizations": [int(v) for v in heldout_groups],
            "train_sample_count": int(len(train_idx)),
            "heldout_sample_count": int(len(heldout_idx)),
            "train_spearman": result["train_spearman"],
            "heldout_spearman": result["test_spearman"],
            "train_delta_robust_scale": result["scale"],
        })
        print(f"V16-R: learning curve {training_group_count} groups complete", flush=True)

    print("V16-R: computing cross-realization nearest neighbors", flush=True)
    nearest_idx, nearest_distance, distances = nearest_cross_realization(x, groups)
    queries = np.arange(len(x))
    all_pairs = pair_statistics(delta, queries, nearest_idx)
    local_results = {}
    for percentage in (10, 20, 30):
        count = int(math.ceil(len(x) * percentage / 100.0))
        selected = np.argsort(nearest_distance, kind="stable")[:count]
        local_results[str(percentage)] = pair_statistics(delta, selected, nearest_idx[selected])

    confidence_ratio = np.divide(
        np.abs(delta), se,
        out=np.full_like(delta, np.inf),
        where=se > 0,
    )
    confidence_ratio[(se == 0) & (delta == 0)] = 0.0
    high_confidence = confidence_ratio >= 2.0
    high_conf_idx = np.flatnonzero(high_confidence)
    confidence_neighbor_metrics = None
    if len(high_conf_idx) >= 2 and len(np.unique(groups[high_conf_idx])) >= 2:
        hc_neighbor_local, hc_distance, _ = nearest_cross_realization(x[high_conf_idx], groups[high_conf_idx])
        hc_count = int(math.ceil(len(high_conf_idx) * 0.10))
        hc_selected = np.argsort(hc_distance, kind="stable")[:hc_count]
        hc_query = high_conf_idx[hc_selected]
        hc_neighbor = high_conf_idx[hc_neighbor_local[hc_selected]]
        confidence_neighbor_metrics = pair_statistics(delta, hc_query, hc_neighbor)

    exact_duplicate_count, near_conflict_count, witnesses = duplicate_witnesses(
        x, delta, groups, macro_index, distances
    )
    print("V16-R: writing report", flush=True)

    heldout_finite = [value for value in curve_heldout if value is not None]
    curve_steps = [heldout_finite[i + 1] - heldout_finite[i] for i in range(len(heldout_finite) - 1)]
    data_limit_supported = bool(
        len(heldout_finite) == 4
        and all(step > 0 for step in curve_steps)
        and (heldout_finite[-1] - heldout_finite[0]) >= 0.05
    )
    train_curve_finite = [value for value in curve_train if value is not None]
    train_vs_test_gap = (train_mean - test_mean) if train_mean is not None and test_mean is not None else None
    nearest10_sign = local_results["10"]["delta_sign_agreement"]
    nearest1nn_sign = all_pairs["delta_sign_agreement"]
    nearest1nn_rank = all_pairs["delta_spearman"]
    observation_aliasing_supported = bool(nearest10_sign is not None and nearest10_sign <= 0.60)
    representation_limit_supported = bool(
        test_mean is not None
        and test_mean < 0.20
        and train_vs_test_gap is not None
        and train_vs_test_gap >= 0.10
        and not data_limit_supported
    )
    optimization_limit = bool(
        test_mean is not None
        and test_mean < 0.20
        and nearest10_sign is not None
        and nearest10_sign >= 0.80
        and nearest1nn_rank is not None
        and nearest1nn_rank >= 0.50
    )
    root_cause_confirmed = bool(
        test_mean is not None
        and test_mean < 0.20
        and train_vs_test_gap is not None
        and train_vs_test_gap >= 0.10
        and not data_limit_supported
        and observation_aliasing_supported
    )
    if root_cause_confirmed:
        primary_root_cause = "TRIGGER_OBSERVATION_ALIASING"
        next_step = "重新设计 Trigger observation 后再评估；本诊断未启动 Stage-2。"
    elif data_limit_supported:
        primary_root_cause = "INSUFFICIENT_TRAINING_REALIZATION_DATA"
        next_step = "优先扩大 training realization pool；本诊断未启动 Stage-2。"
    elif optimization_limit:
        primary_root_cause = "OPTIMIZATION_OR_MODEL_GENERALIZATION_LIMIT"
        next_step = "检查优化或模型泛化能力；不据此修改 observation。"
    elif representation_limit_supported or observation_aliasing_supported:
        primary_root_cause = "REPRESENTATION_LIMIT_SUSPECTED_NOT_CONFIRMED"
        next_step = "当前 replay 支持表示问题迹象，但未满足全部确认门槛；先审查证据再决定后续实验。"
    else:
        primary_root_cause = "INCONCLUSIVE_FROM_REPLAY"
        next_step = "当前 replay 无法区分数据量与模型泛化；先根据本报告决定新的最小诊断。"

    raw_observation_fields = [field for field in fields if field in {"raw_observation", "observation", "trigger_observation_raw"}]
    report = {
        "task": "V16-R TRIGGER STATE IDENTIFIABILITY ROOT-CAUSE GATE",
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)).replace("\\", "/"),
        "dataset_fields": fields,
        "required_fields_present": True,
        "raw_observation_fields": raw_observation_fields,
        "input_field": "normalized_trigger_observation",
        "input_shape_all_rows": [int(v) for v in x_all.shape],
        "finite_checks": finite_checks,
        "replay_realizations": int(len(group_ids)),
        "realization_ids": [int(v) for v in group_ids],
        "active_samples": int(len(active_idx)),
        "episode_seed_unique": [int(v) for v in np.unique(episode_seed)],
        "active_samples_by_realization": {
            str(int(group_id)): int(np.sum(groups == group_id)) for group_id in group_ids
        },
        "target": {
            "source": "raw_delta",
            "transform": "divide by training-fold 1.4826*MAD; no centering, preserving zero/sign origin",
            "loss": "Huber",
            "huber_delta": HUBER_DELTA,
            "equal_realization_weighting": True,
        },
        "model": {
            "architecture": "400-128-128-1 ReLU MLP",
            "epochs": EPOCHS,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "device": "CPU",
            "deterministic_seed_base": BASE_SEED,
        },
        "grouped_evaluation": {
            "method": "5-fold non-shuffled GroupKFold-equivalent greedy sample-balanced assignment",
            "sklearn_available": False,
            "group_to_fold": {str(k): int(v) for k, v in sorted(group_to_fold.items())},
            "fold_sample_counts": fold_sizes,
            "group_leakage_check_passed": True,
            "folds": fold_results,
            "train_spearman_mean": train_mean,
            "train_spearman_std": train_std,
            "test_spearman_mean": test_mean,
            "test_spearman_std": test_std,
            "test_sign_accuracy_mean": test_sign_mean,
            "test_sign_accuracy_std": test_sign_std,
            "train_test_spearman_gap": train_vs_test_gap,
        },
        "learning_curve": {
            "split_method": "one fixed permutation of sorted realization IDs from NumPy default_rng(27005); prefix trains, remainder held out",
            "split_order": [int(v) for v in split_order],
            "heldout_spearman_rise_threshold": "strictly increasing at each step and train16-train4 >= 0.05",
            "train_spearman_curve": train_curve_finite,
            "heldout_spearman_curve": heldout_finite,
            "curves": curve_results,
            "data_limit_supported": data_limit_supported,
        },
        "cross_realization_nearest_neighbor": {
            "metric": "Euclidean distance on normalized 400D observations",
            "same_realization_neighbors_excluded": True,
            "pairing": "one directed nearest-neighbor pair per active query state",
            **all_pairs,
            "mean_nearest_observation_distance": json_number(np.mean(nearest_distance)),
            "median_nearest_observation_distance": json_number(np.median(nearest_distance)),
        },
        "local_aliasing": local_results,
        "teacher_noise_check": {
            "se_update_field_present": True,
            "confidence_ratio": "abs(raw_delta)/se_update; zero SE and nonzero DELTA treated as infinite",
            "threshold": 2.0,
            "high_confidence_active_sample_count": int(np.sum(high_confidence)),
            "high_confidence_fraction": float(np.mean(high_confidence)),
            "nearest_neighbor_pool": "high-confidence active states only; cross-realization constraint",
            "nearest_10_percent_metrics": confidence_neighbor_metrics,
        },
        "duplicate_check": {
            "pair_scope": "unordered cross-realization active pairs",
            "exact_l2_threshold": 1e-12,
            "near_l2_max": NEAR_DUPLICATE_L2_MAX,
            "near_cosine_similarity_min": NEAR_DUPLICATE_COSINE_SIM_MIN,
            "exact_cross_realization_pair_count": exact_duplicate_count,
            "near_duplicate_sign_conflict_count": near_conflict_count,
            "witness_count_saved": len(witnesses),
            "witnesses": witnesses,
        },
        "decision": {
            "data_limit_supported": data_limit_supported,
            "representation_limit_supported": representation_limit_supported,
            "observation_aliasing_supported": observation_aliasing_supported,
            "optimization_or_model_generalization_limit": optimization_limit,
            "state_representation_root_cause_confirmed": root_cause_confirmed,
            "root_cause_confirmation_rule": "test Spearman <0.20; train-test gap >=0.10; learning curve does not meet rise rule; nearest10 sign agreement <=0.60",
            "primary_root_cause": primary_root_cause,
            "next_step": next_step,
        },
        "scope": {
            "environment_run": False,
            "teacher_recomputed": False,
            "formal_rl_trained": False,
            "stage_2_started": False,
            "validation_or_test_seed_used": False,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "report_path": str(REPORT_PATH),
        "replay_realizations": report["replay_realizations"],
        "active_samples": report["active_samples"],
        "grouped_train_spearman_mean": train_mean,
        "grouped_test_spearman_mean": test_mean,
        "grouped_test_spearman_std": test_std,
        "grouped_test_sign_accuracy_mean": test_sign_mean,
        "learning_curve_heldout_spearman": heldout_finite,
        "cross_realization_1nn_sign_agreement": all_pairs["delta_sign_agreement"],
        "cross_realization_1nn_delta_spearman": all_pairs["delta_spearman"],
        "nearest10_sign_agreement": nearest10_sign,
        "nearest20_sign_agreement": local_results["20"]["delta_sign_agreement"],
        "nearest30_sign_agreement": local_results["30"]["delta_sign_agreement"],
        "high_confidence_nearest10": confidence_neighbor_metrics,
        "near_duplicate_sign_conflict_count": near_conflict_count,
        "data_limit_supported": data_limit_supported,
        "observation_aliasing_supported": observation_aliasing_supported,
        "optimization_or_model_generalization_limit": optimization_limit,
        "state_representation_root_cause_confirmed": root_cause_confirmed,
        "primary_root_cause": primary_root_cause,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
