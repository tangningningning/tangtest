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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scenario.task_scenario import (
    TaskConfig,
    build_observed_popularity,
    load_json,
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
            raise ValueError("service_catalog.json 的 service_count 与 configs/proposed/system_config.toml 不一致。")
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
            "GRU checkpoint uses an incompatible popularity observation "
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
    parser = argparse.ArgumentParser(description="使用冻结的服务流行度 GRU 生成 canonical episode 预测。")
    parser.add_argument("--config", default="configs/proposed/system_config.toml")
    parser.add_argument("--popularity-file", default=None)
    parser.add_argument("--popularity-history-file", default=None)
    parser.add_argument("--service-catalog-file", default=None)
    parser.add_argument("--roles-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prediction-file", default=None)
    parser.add_argument("--device", default=None, choices=("cpu", "cuda"))
    return parser


def config_from_project_config(
    project_config: dict[str, Any],
) -> PopularityPredictionConfig:
    from common.project_config import get_config_value

    simulation_steps = int(
        round(
            float(get_config_value(project_config, "simulation.episode_time_s"))
            / float(get_config_value(project_config, "simulation.dt_s"))
        )
    )
    return PopularityPredictionConfig(
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        project_root = find_project_root()
        from common import project_config as project_config_module

        formal_config = project_config_module.load_config(
            args.config,
            require_artifacts=False,
        )
        config = config_from_project_config(formal_config)
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
            or config_path(formal_config, "paths.popularity_gru"),
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
                f"formal popularity slots={canonical_rho_star.shape[0]} 与 configs/proposed/system_config.toml="
                f"{configured_num_slots} 不一致。"
            )
        origin_slots = np.arange(canonical_rho_star.shape[0], dtype=np.int64)
        x_can, _ = build_causal_formal_inputs(
            history_request_counts,
            canonical_request_counts,
            config.history_length,
            origin_slots,
        )

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

    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
