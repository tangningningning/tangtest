#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""服务流行度 GRU 离线训练与 canonical episode 因果预测。

本模块只实现：

    request_count -> observed rho_obs -> GRU supervised training -> H_p rho_star prediction

训练序列使用 ``task_scenario.py`` 中已经实现的动态 Zipf、Bernoulli 到达和
Categorical 服务请求逻辑生成；当前 ``data/generated/popularity.csv`` 只作为
held-out canonical episode 评估，不进入训练。

PyTorch 组件直接使用官方成熟 API，没有复制 PyTorch 源码：

* https://docs.pytorch.org/docs/main/generated/torch.nn.GRU.html
* https://docs.pytorch.org/docs/main/generated/torch.nn.Linear.html
* https://docs.pytorch.org/docs/main/generated/torch.optim.Adam.html
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scenario.task_scenario import (
    TaskConfig,
    build_observed_popularity,
    generate_tasks,
    load_json,
    make_popularity_trace,
    make_rngs,
    make_service_catalog,
    task_config_from_project_config,
)


# ============================================================================
# 集中配置区：均为实现阶段 provisional defaults
# ============================================================================


@dataclass(frozen=True)
class PopularityPredictionConfig:
    horizon_steps: int = 5
    history_length: int = 20
    hidden_size: int = 32
    num_layers: int = 1

    training_seed: int = 20260819
    sequence_seed_base: int = 880001
    num_train_sequences: int = 24
    num_val_sequences: int = 6
    num_test_sequences: int = 6
    sequence_length: int = 300

    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 8

    def validate(self) -> None:
        if self.horizon_steps <= 0:
            raise ValueError("H_p 必须大于 0。")
        if self.history_length < 1:
            raise ValueError("L_rho 必须 >= 1。")
        if self.hidden_size <= 0 or self.num_layers != 1:
            raise ValueError("本阶段只使用单层轻量 GRU，hidden_size 必须大于 0。")
        if min(
            self.num_train_sequences,
            self.num_val_sequences,
            self.num_test_sequences,
            self.sequence_length,
            self.batch_size,
            self.max_epochs,
            self.patience,
        ) <= 0:
            raise ValueError("训练规模、batch、epoch 和 patience 都必须大于 0。")
        if self.sequence_length < self.history_length + self.horizon_steps:
            raise ValueError("sequence_length 不足以构造完整输入历史和预测标签。")


DEFAULT_CONFIG = PopularityPredictionConfig()

OBSERVATION_DEFINITION = "per_slot_request_share_carry_forward"

PREDICTION_FIELDS = (
    "origin_slot",
    "origin_time_s",
    "observation_cutoff_slot",
    "horizon",
    "target_slot",
    "target_time_s",
    "service_id",
    "rho_pred",
    "rho_star",
    "abs_error",
)


# ============================================================================
# 基础 IO 和可复现设置
# ============================================================================


def find_project_root() -> Path:
    candidates = (Path(__file__).resolve().parents[2], Path.cwd().resolve())
    for candidate in candidates:
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("无法找到 adaptive_cache_mec 项目根目录。")


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


# ============================================================================
# 正式 episode 读取与 observed rho_obs 构造
# ============================================================================


def load_popularity_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """读取 canonical rho_star/request_count，rho_obs 由 request_count 重建。"""
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"time_s", "slot", "service_id", "rho_star", "request_count"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("popularity.csv 缺少必要字段。")
        rows.extend(reader)

    slots = sorted({int(row["slot"]) for row in rows})
    services = sorted({int(row["service_id"]) for row in rows})
    if slots != list(range(len(slots))) or services != list(range(len(services))):
        raise ValueError("popularity.csv 的 slot/service_id 必须从 0 连续编号。")

    rho_star = np.zeros((len(slots), len(services)), dtype=np.float64)
    request_counts = np.zeros_like(rho_star)
    times = [0.0 for _ in slots]
    seen: set[tuple[int, int]] = set()
    for row in rows:
        slot = int(row["slot"])
        service_id = int(row["service_id"])
        key = (slot, service_id)
        if key in seen:
            raise ValueError("popularity.csv 存在重复 slot/service_id 行。")
        seen.add(key)
        times[slot] = float(row["time_s"])
        rho_star[slot, service_id] = float(row["rho_star"])
        request_counts[slot, service_id] = float(row["request_count"])

    if len(seen) != len(slots) * len(services):
        raise ValueError("popularity.csv 没有覆盖完整 slot x service 网格。")
    if not np.allclose(rho_star.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("canonical rho_star 未逐时隙归一化。")
    return rho_star, request_counts, times


def load_popularity_history_csv(
    path: Path,
) -> tuple[np.ndarray, list[float], list[int]]:
    """读取独立 prehistory；history_index 和 formal_slot 均严格校验。"""
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "history_index",
            "formal_slot",
            "time_s",
            "service_id",
            "request_count",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("popularity_history.csv 缺少必要字段。")
        rows.extend(reader)

    history_indices = sorted({int(row["history_index"]) for row in rows})
    formal_slots = sorted({int(row["formal_slot"]) for row in rows})
    services = sorted({int(row["service_id"]) for row in rows})
    if history_indices != list(range(len(history_indices))):
        raise ValueError("popularity_history.csv 的 history_index 必须从 0 连续编号。")
    expected_formal_slots = [index - len(history_indices) for index in history_indices]
    if formal_slots != expected_formal_slots:
        raise ValueError(
            "popularity_history.csv 的 formal_slot 必须对应 -L_rho ... -1。"
        )
    if services != list(range(len(services))):
        raise ValueError("popularity_history.csv 的 service_id 必须从 0 连续编号。")

    counts = np.zeros((len(history_indices), len(services)), dtype=np.float64)
    times = [0.0 for _ in history_indices]
    seen: set[tuple[int, int]] = set()
    for row in rows:
        history_index = int(row["history_index"])
        service_id = int(row["service_id"])
        key = (history_index, service_id)
        if key in seen:
            raise ValueError("popularity_history.csv 存在重复 history_index/service_id 行。")
        seen.add(key)
        counts[history_index, service_id] = float(row["request_count"])
        times[history_index] = float(row["time_s"])
    if len(seen) != len(history_indices) * len(services):
        raise ValueError("popularity_history.csv 没有覆盖完整 history x service 网格。")
    return counts, times, formal_slots


def build_causal_formal_inputs(
    history_request_counts: np.ndarray,
    formal_request_counts: np.ndarray,
    history_length: int,
    origin_slots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """为每个 formal origin 构造只含 ``<= origin-1`` 的 observed 窗口。

    返回 ``(inputs, combined_rho_obs)``。history 的最后一个观测对应
    formal slot -1；因此 origin=0 的窗口恰好是全部 prehistory。
    """
    history = np.asarray(history_request_counts, dtype=np.float64)
    formal = np.asarray(formal_request_counts, dtype=np.float64)
    if history.ndim != 2 or formal.ndim != 2 or history.shape[1] != formal.shape[1]:
        raise ValueError("history/formal request_counts 必须是相同服务维度的二维数组。")
    if len(history) < history_length:
        raise ValueError("prehistory 长度不足以构造 L_rho 输入窗口。")
    if np.any(history < 0.0) or np.any(formal < 0.0):
        raise ValueError("request_count 不能为负数。")
    combined = np.vstack((history, formal))
    combined_rho_obs = build_observed_popularity(combined)
    history_end = len(history)
    inputs: list[np.ndarray] = []
    for origin in np.asarray(origin_slots, dtype=np.int64).tolist():
        if origin < 0 or origin >= len(formal):
            raise ValueError(f"formal origin_slot={origin} 越界。")
        end = history_end + int(origin)
        start = end - history_length
        if start < 0:
            raise ValueError(f"origin_slot={origin} 缺少 L_rho causal history。")
        inputs.append(combined_rho_obs[start:end])
    stacked = np.stack(inputs).astype(np.float32) if inputs else np.empty(
        (0, history_length, formal.shape[1]), dtype=np.float32
    )
    return stacked, combined_rho_obs


def rho_obs_quality(rho_obs: np.ndarray, rho_star: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(rho_obs - rho_star))),
        "cross_entropy": distribution_cross_entropy_np(rho_obs, rho_star),
    }


# ============================================================================
# 训练序列生成：复用 task_scenario.py 的动态 Zipf 与请求逻辑
# ============================================================================


def task_config_from_catalog(
    catalog: dict[str, Any],
    project_config: dict[str, Any] | None = None,
) -> TaskConfig:
    if project_config is not None:
        config = task_config_from_project_config(project_config)
        if int(catalog["service_count"]) != config.num_services:
            raise ValueError("service_catalog.json 的 service_count 与 configs/system_config.toml 不一致。")
        return config
    generation = catalog.get("generation_config", {})
    return TaskConfig(
        num_services=int(catalog["service_count"]),
        arrival_probability=float(generation.get("arrival_probability", 0.5)),
        zipf_exponent=float(generation.get("zipf_exponent", 0.8)),
        popularity_epoch_length=int(generation.get("popularity_epoch_length", 20)),
        popularity_changes_per_epoch=int(
            generation.get("popularity_changes_per_epoch", 1)
        ),
        popularity_history_length=int(
            generation.get("popularity_history_length", 20)
        ),
        input_relative_jitter=float(generation.get("input_relative_jitter", 0.2)),
        load_relative_jitter=float(generation.get("load_relative_jitter", 0.1)),
    )


def make_sequence_seeds(config: PopularityPredictionConfig) -> dict[str, list[int]]:
    total = (
        config.num_train_sequences
        + config.num_val_sequences
        + config.num_test_sequences
    )
    seeds = [config.sequence_seed_base + index for index in range(total)]
    train_end = config.num_train_sequences
    val_end = train_end + config.num_val_sequences
    return {
        "train": seeds[:train_end],
        "val": seeds[train_end:val_end],
        "test": seeds[val_end:],
    }


def generate_popularity_request_sequence(
    sequence_seed: int,
    task_config: TaskConfig,
    num_tv: int,
    num_slots: int,
) -> tuple[np.ndarray, np.ndarray]:
    """生成一条独立 popularity/request sequence，不运行 SUMO。"""
    stream_names = (
        "service_catalog",
        "popularity_evolution",
        "task_arrival",
        "service_request",
        "workload_variation",
    )
    rngs, _ = make_rngs(sequence_seed, stream_names)
    catalog = make_service_catalog(task_config, rngs["service_catalog"])
    _, rho_star, _ = make_popularity_trace(
        task_config.num_services,
        num_slots,
        task_config.fallback_dt_s,
        0.0,
        task_config,
        rngs["popularity_evolution"],
    )
    task_vehicles = [f"tv_{index}" for index in range(num_tv)]
    _, request_counts = generate_tasks(
        task_vehicles,
        catalog,
        rho_star,
        num_slots,
        task_config.fallback_dt_s,
        0.0,
        task_config,
        rngs["task_arrival"],
        rngs["service_request"],
        rngs["workload_variation"],
    )
    return rho_star.astype(np.float64), request_counts.astype(np.float64)


def build_windows(
    rho_obs: np.ndarray,
    rho_star: np.ndarray,
    history_length: int,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    origins: list[int] = []
    max_origin = rho_star.shape[0] - horizon_steps
    for origin in range(history_length, max_origin + 1):
        inputs.append(rho_obs[origin - history_length : origin])
        labels.append(rho_star[origin : origin + horizon_steps])
        origins.append(origin)
    if not inputs:
        raise ValueError("没有可用滑动窗口样本。")
    return (
        np.stack(inputs).astype(np.float32),
        np.stack(labels).astype(np.float32),
        np.asarray(origins, dtype=np.int64),
    )


def make_split_dataset(
    seeds: list[int],
    task_config: TaskConfig,
    num_tv: int,
    pred_config: PopularityPredictionConfig,
) -> dict[str, Any]:
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_sequence_ids: list[int] = []
    for sequence_index, seed in enumerate(seeds):
        rho_star, request_counts = generate_popularity_request_sequence(
            seed,
            task_config,
            num_tv,
            pred_config.sequence_length,
        )
        rho_obs = build_observed_popularity(request_counts)
        x, y, _ = build_windows(
            rho_obs,
            rho_star,
            pred_config.history_length,
            pred_config.horizon_steps,
        )
        all_x.append(x)
        all_y.append(y)
        all_sequence_ids.extend([sequence_index] * len(x))
    return {
        "x": np.concatenate(all_x, axis=0),
        "y": np.concatenate(all_y, axis=0),
        "sequence_ids": np.asarray(all_sequence_ids, dtype=np.int64),
        "seeds": list(seeds),
    }


# ============================================================================
# GRU 模型与训练
# ============================================================================


class PopularityGRU(nn.Module):
    """单向单层 GRU：历史 rho_obs -> 一次输出 H_p 个 rho_star 分布。"""

    def __init__(
        self,
        num_services: int,
        hidden_size: int,
        horizon_steps: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_services = num_services
        self.horizon_steps = horizon_steps
        self.gru = nn.GRU(
            input_size=num_services,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, horizon_steps * num_services)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        last_hidden = hidden[-1]
        logits = self.head(last_hidden)
        return logits.reshape(-1, self.horizon_steps, self.num_services)


def distribution_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return -(target * log_prob).sum(dim=-1).mean()


def distribution_cross_entropy_np(pred: np.ndarray, target: np.ndarray) -> float:
    clipped = np.clip(pred, 1e-12, 1.0)
    return float(-(target * np.log(clipped)).sum(axis=-1).mean())


def iter_batches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(x))
    if rng is not None:
        rng.shuffle(indices)
    return [
        (x[indices[start : start + batch_size]], y[indices[start : start + batch_size]])
        for start in range(0, len(indices), batch_size)
    ]


def evaluate_loss(
    model: PopularityGRU,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for xb, yb in iter_batches(x, y, batch_size, None):
            logits = model(torch.from_numpy(xb).to(device))
            target = torch.from_numpy(yb).to(device)
            batch_loss = float(distribution_cross_entropy(logits, target).cpu())
            batch_samples = len(xb)
            total_loss += batch_loss * batch_samples
            total_samples += batch_samples
    if total_samples == 0:
        raise ValueError("评估数据集不能为空。")
    return float(total_loss / total_samples)


def train_model(
    datasets: dict[str, dict[str, Any]],
    num_services: int,
    config: PopularityPredictionConfig,
    device: torch.device,
) -> tuple[PopularityGRU, dict[str, Any]]:
    set_reproducible(config.training_seed)
    model = PopularityGRU(
        num_services=num_services,
        hidden_size=config.hidden_size,
        horizon_steps=config.horizon_steps,
        num_layers=config.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_rng = np.random.default_rng(config.training_seed)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for xb, yb in iter_batches(
            datasets["train"]["x"],
            datasets["train"]["y"],
            config.batch_size,
            train_rng,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.from_numpy(xb).to(device))
            target = torch.from_numpy(yb).to(device)
            loss = distribution_cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("训练 loss 出现 NaN/Inf。")
            loss.backward()
            optimizer.step()
            batch_samples = len(xb)
            train_loss_sum += float(loss.detach().cpu()) * batch_samples
            train_sample_count += batch_samples

        if train_sample_count == 0:
            raise ValueError("训练数据集不能为空。")
        train_loss = float(train_loss_sum / train_sample_count)
        val_loss = evaluate_loss(
            model,
            datasets["val"]["x"],
            datasets["val"]["y"],
            config.batch_size,
            device,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss + 1e-12 < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("训练没有产生有效 checkpoint。")
    model.load_state_dict(best_state)
    test_loss = evaluate_loss(
        model,
        datasets["test"]["x"],
        datasets["test"]["y"],
        config.batch_size,
        device,
    )
    return model, {
        "best_validation_loss": best_val,
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "test_loss": test_loss,
        "history": history,
    }


# ============================================================================
# 推理、评估和 checkpoint
# ============================================================================


def predict_probabilities(
    model: PopularityGRU,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in iter_batches(x, np.zeros((len(x), 1), dtype=np.float32), batch_size, None):
            logits = model(torch.from_numpy(xb).to(device))
            prob = F.softmax(logits, dim=-1)
            if not torch.isfinite(prob).all():
                raise FloatingPointError("预测概率出现 NaN/Inf。")
            chunks.append(prob.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


def evaluate_predictions(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    horizon_steps = pred.shape[1]
    metrics: dict[str, Any] = {"per_horizon": {}, "probability_sum_mean": {}}
    for horizon in range(horizon_steps):
        pred_h = pred[:, horizon, :]
        truth_h = truth[:, horizon, :]
        pred_top1 = np.argmax(pred_h, axis=1)
        truth_top1 = np.argmax(truth_h, axis=1)
        top1 = float(np.mean(pred_top1 == truth_top1))
        top3_overlap = []
        for row in range(len(pred_h)):
            pred_top3 = set(np.argsort(pred_h[row])[-3:])
            truth_top3 = set(np.argsort(truth_h[row])[-3:])
            top3_overlap.append(len(pred_top3 & truth_top3) / 3.0)
        metrics["per_horizon"][str(horizon)] = {
            "cross_entropy": distribution_cross_entropy_np(pred_h, truth_h),
            "mae": float(np.mean(np.abs(pred_h - truth_h))),
            "top1_accuracy": top1,
            "top3_overlap": float(np.mean(top3_overlap)),
        }
        metrics["probability_sum_mean"][str(horizon)] = float(
            np.mean(pred_h.sum(axis=1))
        )
    return metrics


def canonical_prediction_rows(
    pred: np.ndarray,
    truth: np.ndarray,
    origins: np.ndarray,
    times: list[float],
    dt_s: float | None = None,
) -> list[dict[str, Any]]:
    """保存全部 formal origins；episode 外 target 只保留预测值。"""
    rows: list[dict[str, Any]] = []
    num_services = pred.shape[2]
    truth_array = np.asarray(truth)
    if truth_array.ndim not in {2, 3}:
        raise ValueError("canonical truth 必须是 [slot, service] 或 [origin, horizon, service]。")
    if dt_s is None:
        if len(times) < 2:
            dt_s = 1.0
        else:
            dt_s = float(times[1] - times[0])
    for sample_index, origin in enumerate(origins.tolist()):
        cutoff = origin - 1
        for horizon in range(pred.shape[1]):
            target_slot = origin + horizon
            has_truth = target_slot < len(times)
            for service_id in range(num_services):
                rho_pred = float(pred[sample_index, horizon, service_id])
                if truth_array.ndim == 2:
                    has_truth = has_truth and target_slot < truth_array.shape[0]
                    rho_star: float | str = (
                        float(truth_array[target_slot, service_id])
                        if has_truth
                        else ""
                    )
                else:
                    has_truth = has_truth and sample_index < truth_array.shape[0]
                    rho_star = (
                        float(truth_array[sample_index, horizon, service_id])
                        if has_truth
                        else ""
                    )
                target_time = (
                    float(times[target_slot])
                    if target_slot < len(times)
                    else float(times[-1] + (target_slot - len(times) + 1) * dt_s)
                )
                rows.append(
                    {
                        "origin_slot": origin,
                        "origin_time_s": times[origin],
                        "observation_cutoff_slot": cutoff,
                        "horizon": horizon,
                        "target_slot": target_slot,
                        "target_time_s": target_time,
                        "service_id": service_id,
                        "rho_pred": rho_pred,
                        "rho_star": rho_star,
                        "abs_error": abs(rho_pred - float(rho_star))
                        if rho_star != ""
                        else "",
                    }
                )
    return rows


def save_checkpoint(
    path: Path,
    model: PopularityGRU,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = {"model_state_dict": model.state_dict(), "metadata": metadata}
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_checkpoint_model(
    path: Path,
    device: torch.device,
) -> tuple[PopularityGRU, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("GRU checkpoint 缺少 metadata。")
    if metadata.get("observation_definition") != OBSERVATION_DEFINITION:
        raise ValueError(
            "old GRU checkpoint uses incompatible popularity observation "
            "definition; retrain popularity GRU"
        )
    model = PopularityGRU(
        num_services=int(metadata["F"]),
        hidden_size=int(metadata["hidden_size"]),
        horizon_steps=int(metadata["H_p"]),
        num_layers=int(metadata["num_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, metadata


def run_frozen_inference(
    checkpoint_path: Path,
    inputs: np.ndarray,
    batch_size: int,
    device: torch.device,
    expected_metadata: dict[str, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a frozen GRU and predict; this path has no training/save operation."""

    model, metadata = load_checkpoint_model(checkpoint_path, device)
    for key, value in expected_metadata.items():
        if int(metadata[key]) != int(value):
            raise ValueError(
                f"frozen GRU checkpoint {key}={metadata[key]} 与配置 {value} 不一致。"
            )
    return predict_probabilities(model, inputs, batch_size, device), metadata


def verify_outputs(
    rows: list[dict[str, Any]],
    pred: np.ndarray,
    truth: np.ndarray,
    origins: np.ndarray,
    config: PopularityPredictionConfig,
    reload_pred: np.ndarray,
) -> None:
    if not rows:
        raise AssertionError("popularity_predictions.csv 将为空。")
    if pred.ndim != 3 or truth.ndim != 2 or pred.shape[2] != truth.shape[1]:
        raise AssertionError("预测或 formal 标签 shape 不符合约定。")
    if pred.shape[1] != config.horizon_steps:
        raise AssertionError("H_p 输出维度错误。")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise AssertionError("预测或标签存在 NaN/Inf。")
    expected_rows = len(origins) * config.horizon_steps * pred.shape[2]
    if len(rows) != expected_rows:
        raise AssertionError(
            f"预测输出行数={len(rows)}，期望 {expected_rows}（全部 formal origins）。"
        )
    if not np.array_equal(origins, np.arange(len(origins), dtype=np.int64)):
        raise AssertionError("formal prediction origins 必须完整覆盖 0...N-1。")
    if not np.allclose(pred.sum(axis=-1), 1.0, atol=1e-5, rtol=0.0):
        raise AssertionError("每个 horizon 的预测概率和不为 1。")
    if any(int(row["observation_cutoff_slot"]) != int(row["origin_slot"]) - 1 for row in rows):
        raise AssertionError("因果 cutoff 不等于 origin_slot - 1。")
    if any(
        int(row["target_slot"]) != int(row["origin_slot"]) + int(row["horizon"])
        for row in rows
    ):
        raise AssertionError("target_slot 与 origin/horizon 不一致。")
    for row in rows:
        target = int(row["target_slot"])
        if target < truth.shape[0]:
            if row["rho_star"] in ("", None) or row["abs_error"] in ("", None):
                raise AssertionError("formal truth 范围内的目标不应缺少 rho_star/abs_error。")
        elif row["rho_star"] not in ("", None) or row["abs_error"] not in ("", None):
            raise AssertionError("episode 外 target 只能保留预测值，不能伪造 truth。")
    if not np.allclose(pred, reload_pred, atol=0.0, rtol=0.0):
        raise AssertionError("checkpoint reload 后预测不一致。")


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练服务流行度 GRU 并预测 canonical episode。")
    parser.add_argument("--config", default="configs/system_config.toml")
    parser.add_argument(
        "--mode",
        choices=("train", "predict"),
        default="train",
        help="Legacy defaults to training; realization generation must use predict.",
    )
    parser.add_argument("--popularity-file", default=None)
    parser.add_argument("--popularity-history-file", default=None)
    parser.add_argument("--service-catalog-file", default=None)
    parser.add_argument("--roles-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prediction-file", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--sequence-seed-base", type=int, default=None)
    parser.add_argument("--num-train-sequences", type=int, default=None)
    parser.add_argument("--num-val-sequences", type=int, default=None)
    parser.add_argument("--num-test-sequences", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default=None, choices=("cpu", "cuda"))
    return parser


def config_from_args(
    args: argparse.Namespace,
    project_config: dict[str, Any] | None = None,
) -> PopularityPredictionConfig:
    if project_config is None:
        base = DEFAULT_CONFIG
    else:
        from common.project_config import get_config_value

        simulation_steps = int(
            round(
                float(get_config_value(project_config, "simulation.episode_time_s"))
                / float(get_config_value(project_config, "simulation.dt_s"))
            )
        )
        base = PopularityPredictionConfig(
            horizon_steps=int(
                get_config_value(
                    project_config,
                    "prediction.vehicle.horizon",
                    DEFAULT_CONFIG.horizon_steps,
                )
            ),
            history_length=int(
                get_config_value(
                    project_config,
                    "prediction.popularity.history_length",
                    DEFAULT_CONFIG.history_length,
                )
            ),
            hidden_size=int(
                get_config_value(
                    project_config,
                    "prediction.popularity.hidden_size",
                    DEFAULT_CONFIG.hidden_size,
                )
            ),
            training_seed=int(
                get_config_value(
                    project_config,
                    "prediction.popularity.training_seed",
                    DEFAULT_CONFIG.training_seed,
                )
            ),
            sequence_seed_base=int(
                get_config_value(
                    project_config,
                    "prediction.popularity.sequence_seed_base",
                    DEFAULT_CONFIG.sequence_seed_base,
                )
            ),
            num_train_sequences=int(
                get_config_value(
                    project_config,
                    "prediction.popularity.num_train_sequences",
                    DEFAULT_CONFIG.num_train_sequences,
                )
            ),
            num_val_sequences=int(
                get_config_value(
                    project_config,
                    "prediction.popularity.num_val_sequences",
                    DEFAULT_CONFIG.num_val_sequences,
                )
            ),
            num_test_sequences=int(
                get_config_value(
                    project_config,
                    "prediction.popularity.num_test_sequences",
                    DEFAULT_CONFIG.num_test_sequences,
                )
            ),
            sequence_length=simulation_steps,
            learning_rate=float(
                get_config_value(
                    project_config,
                    "prediction.popularity.learning_rate",
                    DEFAULT_CONFIG.learning_rate,
                )
            ),
            batch_size=int(
                get_config_value(
                    project_config,
                    "prediction.popularity.batch_size",
                    DEFAULT_CONFIG.batch_size,
                )
            ),
            max_epochs=int(
                get_config_value(
                    project_config,
                    "prediction.popularity.max_epochs",
                    DEFAULT_CONFIG.max_epochs,
                )
            ),
            patience=int(
                get_config_value(
                    project_config,
                    "prediction.popularity.patience",
                    DEFAULT_CONFIG.patience,
                )
            ),
        )

    def choose(name: str, value: Any) -> Any:
        return getattr(base, name) if value is None else value

    return PopularityPredictionConfig(
        horizon_steps=choose("horizon_steps", args.horizon),
        history_length=choose("history_length", args.history_length),
        hidden_size=choose("hidden_size", args.hidden_size),
        training_seed=choose("training_seed", args.training_seed),
        sequence_seed_base=choose("sequence_seed_base", args.sequence_seed_base),
        num_train_sequences=choose("num_train_sequences", args.num_train_sequences),
        num_val_sequences=choose("num_val_sequences", args.num_val_sequences),
        num_test_sequences=choose("num_test_sequences", args.num_test_sequences),
        sequence_length=choose("sequence_length", args.sequence_length),
        learning_rate=choose("learning_rate", args.learning_rate),
        batch_size=choose("batch_size", args.batch_size),
        max_epochs=choose("max_epochs", args.max_epochs),
        patience=choose("patience", args.patience),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        project_root = find_project_root()
        from common import project_config as project_config_module

        formal_config = project_config_module.load_config(
            args.config,
            require_artifacts=args.mode == "train",
        )
        config = config_from_args(args, formal_config)
        config.validate()
        device_name = args.device or "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求使用 CUDA，但当前 PyTorch 不可用 CUDA。")
        device = torch.device(device_name)

        config_path = project_config_module.get_config_value
        popularity_file = resolve_path(
            project_root,
            args.popularity_file
            or config_path(
                formal_config,
                "paths.popularity",
                "data/generated/popularity.csv",
            ),
        )
        popularity_history_file = resolve_path(
            project_root,
            args.popularity_history_file
            or config_path(
                formal_config,
                "paths.popularity_history",
                "data/generated/popularity_history.csv",
            ),
        )
        catalog_file = resolve_path(
            project_root,
            args.service_catalog_file
            or config_path(
                formal_config,
                "paths.service_catalog",
                "data/generated/service_catalog.json",
            ),
        )
        roles_file = resolve_path(
            project_root,
            args.roles_file
            or config_path(
                formal_config,
                "paths.vehicle_roles",
                "data/generated/vehicle_roles.json",
            ),
        )
        output_dir = resolve_path(
            project_root,
            args.output_dir
            or catalog_file.parent,
        )
        checkpoint_file = resolve_path(
            project_root,
            args.checkpoint
            or (
                config_path(formal_config, "paths.popularity_gru")
                if args.mode == "predict"
                else output_dir / "popularity_gru.pt"
            ),
        )
        prediction_file = resolve_path(
            project_root,
            args.prediction_file
            or config_path(
                formal_config,
                "paths.popularity_predictions",
                output_dir / "popularity_predictions.csv",
            ),
        )

        service_catalog = load_json(catalog_file)
        roles = load_json(roles_file)
        task_config = task_config_from_catalog(service_catalog, formal_config)
        task_config.validate(len(roles.get("task_vehicles", [])))
        num_tv = len([str(vehicle_id) for vehicle_id in roles.get("task_vehicles", [])])
        if num_tv <= 0:
            raise ValueError("vehicle_roles.json 中没有 task_vehicles。")

        canonical_rho_star, canonical_request_counts, canonical_times = load_popularity_csv(
            popularity_file
        )
        history_request_counts, history_times, history_formal_slots = (
            load_popularity_history_csv(popularity_history_file)
        )
        if len(history_request_counts) != config.history_length:
            raise ValueError(
                "popularity_history.csv 长度与 checkpoint/正式配置 L_rho 不一致。"
            )
        if history_formal_slots != list(range(-config.history_length, 0)):
            raise ValueError("popularity_history.csv 未覆盖 formal slot -L_rho ... -1。")
        if canonical_times and history_times:
            dt_from_formal = (
                float(canonical_times[1] - canonical_times[0])
                if len(canonical_times) > 1
                else 1.0
            )
            expected_history_times = [
                float(canonical_times[0] + slot * dt_from_formal)
                for slot in history_formal_slots
            ]
            if not np.allclose(
                np.asarray(history_times, dtype=np.float64),
                np.asarray(expected_history_times, dtype=np.float64),
                atol=1e-6,
                rtol=0.0,
            ):
                raise ValueError(
                    "prehistory absolute time 未与 formal start 前的连续时间轴对齐。"
                )
        if canonical_rho_star.shape[1] != task_config.num_services:
            raise ValueError("canonical F 与 service_catalog.json 不一致。")
        if canonical_request_counts.shape[1] != history_request_counts.shape[1]:
            raise ValueError("formal/prehistory request_counts 的服务维度不一致。")
        configured_num_slots = int(
            round(
                float(config_path(formal_config, "simulation.episode_time_s"))
                / float(config_path(formal_config, "simulation.dt_s"))
            )
        )
        if canonical_rho_star.shape[0] != configured_num_slots:
            raise ValueError(
                f"formal popularity slots={canonical_rho_star.shape[0]} 与 configs/system_config.toml="
                f"{configured_num_slots} 不一致。"
            )
        origin_slots = np.arange(canonical_rho_star.shape[0], dtype=np.int64)
        x_can, combined_rho_obs = build_causal_formal_inputs(
            history_request_counts,
            canonical_request_counts,
            config.history_length,
            origin_slots,
        )
        canonical_rho_obs = combined_rho_obs[len(history_request_counts) :]
        valid_count = canonical_rho_star.shape[0] - config.horizon_steps + 1
        if valid_count <= 0:
            raise ValueError("formal episode 长度不足以评估一个完整 H_p。")
        eval_origins = np.arange(valid_count, dtype=np.int64)
        y_can = np.stack(
            [
                canonical_rho_star[origin : origin + config.horizon_steps]
                for origin in eval_origins.tolist()
            ],
            axis=0,
        ).astype(np.float32)

        if args.mode == "predict":
            expected = {
                "F": task_config.num_services,
                "H_p": config.horizon_steps,
                "L_rho": config.history_length,
            }
            pred_can, _ = run_frozen_inference(
                checkpoint_file,
                x_can,
                config.batch_size,
                device,
                expected,
            )
            rows = canonical_prediction_rows(
                pred_can,
                canonical_rho_star,
                origin_slots,
                canonical_times,
                dt_s=(canonical_times[1] - canonical_times[0])
                if len(canonical_times) > 1
                else 1.0,
            )
            repeat_pred, _ = run_frozen_inference(
                checkpoint_file,
                x_can,
                config.batch_size,
                device,
                expected,
            )
            verify_outputs(
                rows,
                pred_can,
                canonical_rho_star,
                origin_slots,
                config,
                repeat_pred,
            )
            write_csv_atomic(prediction_file, rows)
            print("服务流行度 frozen-GRU 推理完成：")
            print(f"  checkpoint_unchanged=True")
            print(f"  formal origins={len(origin_slots)}，canonical rows={len(rows)}")
            print(f"  output: {prediction_file}")
            return 0

        sequence_seeds = make_sequence_seeds(config)
        datasets = {
            split: make_split_dataset(seeds, task_config, num_tv, config)
            for split, seeds in sequence_seeds.items()
        }
        if set(sequence_seeds["train"]) & set(sequence_seeds["val"]):
            raise AssertionError("train/val sequence seed 存在交集。")
        if set(sequence_seeds["train"]) & set(sequence_seeds["test"]):
            raise AssertionError("train/test sequence seed 存在交集。")
        if set(sequence_seeds["val"]) & set(sequence_seeds["test"]):
            raise AssertionError("val/test sequence seed 存在交集。")

        model, training_summary = train_model(
            datasets,
            task_config.num_services,
            config,
            device,
        )
        pred_can = predict_probabilities(
            model,
            x_can,
            config.batch_size,
            device,
        )
        canonical_metrics = evaluate_predictions(pred_can[:valid_count], y_can)
        rows = canonical_prediction_rows(
            pred_can,
            canonical_rho_star,
            origin_slots,
            canonical_times,
            dt_s=(canonical_times[1] - canonical_times[0])
            if len(canonical_times) > 1
            else 1.0,
        )

        metadata = {
            "schema_version": 3,
            "F": task_config.num_services,
            "H_p": config.horizon_steps,
            "L_rho": config.history_length,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "observation_definition": OBSERVATION_DEFINITION,
            "training_seed": config.training_seed,
            "sequence_seed_base": config.sequence_seed_base,
            "train_sequence_seeds": sequence_seeds["train"],
            "val_sequence_seeds": sequence_seeds["val"],
            "test_sequence_seeds": sequence_seeds["test"],
            "num_train_sequences": len(sequence_seeds["train"]),
            "num_val_sequences": len(sequence_seeds["val"]),
            "num_test_sequences": len(sequence_seeds["test"]),
            "sequence_length": config.sequence_length,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "task_generation_parameters": {
                "num_services": task_config.num_services,
                "arrival_probability": task_config.arrival_probability,
                "zipf_exponent": task_config.zipf_exponent,
                "popularity_epoch_length": task_config.popularity_epoch_length,
                "popularity_changes_per_epoch": task_config.popularity_changes_per_epoch,
                "num_task_vehicles": num_tv,
            },
            "training_summary": training_summary,
            "canonical_metrics": canonical_metrics,
            "rho_obs_quality": rho_obs_quality(canonical_rho_obs, canonical_rho_star),
            "canonical_episode_used_for_training": False,
            "formal_time_slot_count": int(canonical_rho_star.shape[0]),
            "formal_origin_count": int(len(origin_slots)),
            "truth_evaluation_origin_count": int(valid_count),
            "prediction_tail_origins_without_formal_truth": [
                int(origin) for origin in origin_slots if origin >= valid_count
            ],
            "prehistory": {
                "file": str(popularity_history_file),
                "history_slots": int(len(history_request_counts)),
                "formal_slots": [int(value) for value in history_formal_slots],
                "formal_origin_0_uses": "prehistory request_count only",
            },
            "official_pytorch_api": ["torch.nn.GRU", "torch.nn.Linear", "torch.optim.Adam"],
            "config": asdict(config),
        }

        save_checkpoint(checkpoint_file, model, metadata)
        reloaded_model, _ = load_checkpoint_model(checkpoint_file, device)
        reload_pred = predict_probabilities(
            reloaded_model,
            x_can,
            config.batch_size,
            device,
        )
        verify_outputs(
            rows,
            pred_can,
            canonical_rho_star,
            origin_slots,
            config,
            reload_pred,
        )
        write_csv_atomic(prediction_file, rows)

        print("服务流行度预测完成：")
        print(
            f"  sequences train/val/test="
            f"{len(sequence_seeds['train'])}/{len(sequence_seeds['val'])}/"
            f"{len(sequence_seeds['test'])}"
        )
        print(
            f"  F={task_config.num_services}, L_rho={config.history_length}, "
            f"H_p={config.horizon_steps}, hidden={config.hidden_size}"
        )
        print(
            f"  best_val={training_summary['best_validation_loss']:.6f}, "
            f"best_epoch={training_summary['best_epoch']}, "
            f"stopped_epoch={training_summary['stopped_epoch']}"
        )
        print(
            f"  formal origins={len(origin_slots)}，"
            f"truth-evaluated origins={valid_count}，tail origins={len(origin_slots) - valid_count}"
        )
        print(f"  canonical rows={len(rows)}")
        print(f"  checkpoint_reload_consistent=True")
        print(f"  output: {checkpoint_file}")
        print(f"  output: {prediction_file}")
        return 0
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
