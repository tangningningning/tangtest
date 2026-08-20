#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""服务流行度 GRU 离线训练与 canonical episode 因果预测。

本模块只实现：

    request_count -> EWMA rho_obs -> GRU supervised training -> H_p rho_star prediction

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

from task_scenario import (
    DEFAULT_CONFIG as TASK_DEFAULT_CONFIG,
    TaskConfig,
    generate_tasks,
    load_json,
    make_popularity_trace,
    make_rngs,
    make_service_catalog,
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
    beta_rho: float = 0.90
    alpha_rho: float = 0.10

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
    canonical_tolerance: float = 1e-5

    def validate(self) -> None:
        if self.horizon_steps <= 0:
            raise ValueError("H_p 必须大于 0。")
        if self.history_length < 1:
            raise ValueError("L_rho 必须 >= 1。")
        if self.hidden_size <= 0 or self.num_layers != 1:
            raise ValueError("本阶段只使用单层轻量 GRU，hidden_size 必须大于 0。")
        if not 0.0 <= self.beta_rho <= 1.0:
            raise ValueError("beta_rho 必须位于 [0, 1]。")
        if self.alpha_rho <= 0.0:
            raise ValueError("alpha_rho 必须大于 0。")
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
    candidates = (Path(__file__).resolve().parents[1], Path.cwd().resolve())
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
# 正式 episode 读取与 EWMA rho_obs 构造
# ============================================================================


def load_popularity_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """读取 canonical rho_star/request_count，忽略旧 rho_obs 定义。"""
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


def build_ewma_rho_obs(
    request_counts: np.ndarray,
    beta_rho: float,
    alpha_rho: float,
) -> np.ndarray:
    """因果 EWMA：rho_obs(t) 只使用 <= t 的真实 request_count。"""
    num_slots, num_services = request_counts.shape
    counts = np.full(num_services, alpha_rho, dtype=np.float64)
    rho_obs = np.zeros((num_slots, num_services), dtype=np.float64)
    for slot in range(num_slots):
        counts = beta_rho * counts + request_counts[slot]
        rho_obs[slot] = (counts + alpha_rho) / (
            float(counts.sum()) + num_services * alpha_rho
        )
    if not np.allclose(rho_obs.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise AssertionError("EWMA rho_obs 未逐时隙归一化。")
    return rho_obs


def rho_obs_quality(rho_obs: np.ndarray, rho_star: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(rho_obs - rho_star))),
        "cross_entropy": distribution_cross_entropy_np(rho_obs, rho_star),
    }


# ============================================================================
# 训练序列生成：复用 task_scenario.py 的动态 Zipf 与请求逻辑
# ============================================================================


def task_config_from_catalog(catalog: dict[str, Any]) -> TaskConfig:
    generation = catalog.get("generation_config", {})
    return TaskConfig(
        num_services=int(catalog["service_count"]),
        arrival_probability=float(generation.get("arrival_probability", 0.5)),
        zipf_exponent=float(generation.get("zipf_exponent", 0.8)),
        popularity_epoch_length=int(generation.get("popularity_epoch_length", 20)),
        popularity_changes_per_epoch=int(
            generation.get("popularity_changes_per_epoch", 1)
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
        TASK_DEFAULT_CONFIG.fallback_dt_s,
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
        TASK_DEFAULT_CONFIG.fallback_dt_s,
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
        rho_obs = build_ewma_rho_obs(
            request_counts,
            pred_config.beta_rho,
            pred_config.alpha_rho,
        )
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
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in iter_batches(x, y, batch_size, None):
            logits = model(torch.from_numpy(xb).to(device))
            target = torch.from_numpy(yb).to(device)
            losses.append(float(distribution_cross_entropy(logits, target).cpu()))
    return float(np.mean(losses))


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
        batch_losses: list[float] = []
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
            batch_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(batch_losses))
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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    num_services = pred.shape[2]
    for sample_index, origin in enumerate(origins.tolist()):
        cutoff = origin - 1
        for horizon in range(pred.shape[1]):
            target_slot = origin + horizon
            for service_id in range(num_services):
                rho_pred = float(pred[sample_index, horizon, service_id])
                rho_star = float(truth[sample_index, horizon, service_id])
                rows.append(
                    {
                        "origin_slot": origin,
                        "origin_time_s": times[origin],
                        "observation_cutoff_slot": cutoff,
                        "horizon": horizon,
                        "target_slot": target_slot,
                        "target_time_s": times[target_slot],
                        "service_id": service_id,
                        "rho_pred": rho_pred,
                        "rho_star": rho_star,
                        "abs_error": abs(rho_pred - rho_star),
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
    metadata = checkpoint["metadata"]
    model = PopularityGRU(
        num_services=int(metadata["F"]),
        hidden_size=int(metadata["hidden_size"]),
        horizon_steps=int(metadata["H_p"]),
        num_layers=int(metadata["num_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, metadata


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
    if pred.shape != truth.shape:
        raise AssertionError("预测与标签 shape 不一致。")
    if pred.shape[1] != config.horizon_steps:
        raise AssertionError("H_p 输出维度错误。")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise AssertionError("预测或标签存在 NaN/Inf。")
    if not np.allclose(pred.sum(axis=-1), 1.0, atol=1e-5, rtol=0.0):
        raise AssertionError("每个 horizon 的预测概率和不为 1。")
    if any(int(row["observation_cutoff_slot"]) != int(row["origin_slot"]) - 1 for row in rows):
        raise AssertionError("因果 cutoff 不等于 origin_slot - 1。")
    if any(
        int(row["target_slot"]) != int(row["origin_slot"]) + int(row["horizon"])
        for row in rows
    ):
        raise AssertionError("target_slot 与 origin/horizon 不一致。")
    if int(origins.min()) < config.history_length:
        raise AssertionError("早期历史不足的 origin 被错误输出。")
    if not np.allclose(pred, reload_pred, atol=0.0, rtol=0.0):
        raise AssertionError("checkpoint reload 后预测不一致。")


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练服务流行度 GRU 并预测 canonical episode。")
    parser.add_argument("--popularity-file", default="data/generated/popularity.csv")
    parser.add_argument("--service-catalog-file", default="data/generated/service_catalog.json")
    parser.add_argument("--roles-file", default="data/generated/vehicle_roles.json")
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--horizon", type=int, default=DEFAULT_CONFIG.horizon_steps)
    parser.add_argument("--history-length", type=int, default=DEFAULT_CONFIG.history_length)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_CONFIG.hidden_size)
    parser.add_argument("--beta-rho", type=float, default=DEFAULT_CONFIG.beta_rho)
    parser.add_argument("--alpha-rho", type=float, default=DEFAULT_CONFIG.alpha_rho)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_CONFIG.training_seed)
    parser.add_argument("--sequence-seed-base", type=int, default=DEFAULT_CONFIG.sequence_seed_base)
    parser.add_argument("--num-train-sequences", type=int, default=DEFAULT_CONFIG.num_train_sequences)
    parser.add_argument("--num-val-sequences", type=int, default=DEFAULT_CONFIG.num_val_sequences)
    parser.add_argument("--num-test-sequences", type=int, default=DEFAULT_CONFIG.num_test_sequences)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_CONFIG.sequence_length)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG.learning_rate)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_CONFIG.max_epochs)
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG.patience)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    return parser


def config_from_args(args: argparse.Namespace) -> PopularityPredictionConfig:
    return PopularityPredictionConfig(
        horizon_steps=args.horizon,
        history_length=args.history_length,
        hidden_size=args.hidden_size,
        beta_rho=args.beta_rho,
        alpha_rho=args.alpha_rho,
        training_seed=args.training_seed,
        sequence_seed_base=args.sequence_seed_base,
        num_train_sequences=args.num_train_sequences,
        num_val_sequences=args.num_val_sequences,
        num_test_sequences=args.num_test_sequences,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        project_root = find_project_root()
        config = config_from_args(args)
        config.validate()
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求使用 CUDA，但当前 PyTorch 不可用 CUDA。")
        device = torch.device(args.device)

        popularity_file = resolve_path(project_root, args.popularity_file)
        catalog_file = resolve_path(project_root, args.service_catalog_file)
        roles_file = resolve_path(project_root, args.roles_file)
        output_dir = resolve_path(project_root, args.output_dir)
        checkpoint_file = output_dir / "popularity_gru.pt"
        prediction_file = output_dir / "popularity_predictions.csv"

        service_catalog = load_json(catalog_file)
        roles = load_json(roles_file)
        task_config = task_config_from_catalog(service_catalog)
        task_config.validate(len(roles.get("task_vehicles", [])))
        num_tv = len([str(vehicle_id) for vehicle_id in roles.get("task_vehicles", [])])
        if num_tv <= 0:
            raise ValueError("vehicle_roles.json 中没有 task_vehicles。")

        canonical_rho_star, canonical_request_counts, canonical_times = load_popularity_csv(
            popularity_file
        )
        if canonical_rho_star.shape[1] != task_config.num_services:
            raise ValueError("canonical F 与 service_catalog.json 不一致。")
        canonical_rho_obs = build_ewma_rho_obs(
            canonical_request_counts,
            config.beta_rho,
            config.alpha_rho,
        )
        x_can, y_can, origins_can = build_windows(
            canonical_rho_obs,
            canonical_rho_star,
            config.history_length,
            config.horizon_steps,
        )

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
        canonical_metrics = evaluate_predictions(pred_can, y_can)
        rows = canonical_prediction_rows(pred_can, y_can, origins_can, canonical_times)

        metadata = {
            "schema_version": 1,
            "F": task_config.num_services,
            "H_p": config.horizon_steps,
            "L_rho": config.history_length,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "beta_rho": config.beta_rho,
            "alpha_rho": config.alpha_rho,
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
        verify_outputs(rows, pred_can, y_can, origins_can, config, reload_pred)
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
