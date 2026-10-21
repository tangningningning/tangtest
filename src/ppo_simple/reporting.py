"""记录正式 joint MAPPO run 的来源信息、清单和汇总指标。"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - 无 PyTorch 环境也可检查 reporting
    torch = None  # type: ignore[assignment]

from common import project_config
from .macro_targets import (
    build_macro_n2_target,
)

SLOT_METRICS_FILENAME = "slot_metrics.jsonl"
MACRO_TRANSITION_TRACE_FILENAME = "macro_transition_trace.npz"

EPISODE_METRICS_FILENAME = "episode_metrics.jsonl"

TRAINING_EPOCH_RETURN_PLOT_FILENAME = "training_epoch_return.png"


def build_episode_pool_provenance(
    episode_pool: str | Path | None,
    *,
    validated_integrity_baseline_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Separate historical data-source identity from the current run config."""

    if episode_pool is None:
        return None
    pool_path = Path(episode_pool).expanduser().resolve(strict=False)
    manifest_path = pool_path / "formal_pool_manifest.json"
    if not manifest_path.is_file():
        return None
    raw_manifest = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid training pool manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Training pool manifest root must be an object: {manifest_path}")
    if manifest.get("schema_version") != "formal_task_data_pool_manifest_v1":
        raise ValueError(f"Unsupported formal task-data pool manifest: {manifest_path}")
    fields = (
        "schema_version",
        "task_data_model",
        "split",
        "seed_start",
        "seed_end",
        "episode_count",
        "config_sha256",
        "semantic_config_sha256",
        "target_generation_config_fingerprint",
        "task_data_contract",
        "task_data_contract_sha256",
        "service_catalog_path",
        "service_catalog_sha256",
        "data_split_manifest_sha256",
        "source_provenance",
        "transformed_fields",
    )
    provenance = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        **{field: manifest.get(field) for field in fields},
    }
    baseline_path = pool_path / "formal_pool_integrity_baseline.json"
    if validated_integrity_baseline_sha256 is not None and not baseline_path.is_file():
        raise FileNotFoundError(
            f"Validated Training Pool integrity baseline is missing: {baseline_path}"
        )
    if baseline_path.is_file():
        raw_baseline = baseline_path.read_bytes()
        baseline_hash = hashlib.sha256(raw_baseline).hexdigest()
        if (
            validated_integrity_baseline_sha256 is not None
            and baseline_hash != validated_integrity_baseline_sha256
        ):
            raise ValueError("Training Pool integrity baseline changed after Runner validation.")
        try:
            baseline = json.loads(raw_baseline.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid Training Pool integrity baseline: {baseline_path}") from exc
        if not isinstance(baseline, dict):
            raise ValueError("Training Pool integrity baseline root must be an object.")
        if baseline.get("pool_manifest_sha256") != provenance["manifest_sha256"]:
            raise ValueError("Training Pool integrity baseline does not identify this Pool Manifest.")
        if baseline.get("task_data_contract_sha256") != provenance.get(
            "task_data_contract_sha256"
        ):
            raise ValueError("Training Pool integrity baseline task-data contract mismatch.")
        provenance.update(
            {
                "integrity_baseline_path": str(baseline_path),
                "integrity_baseline_sha256": baseline_hash,
                "generation_config_fingerprint": manifest.get(
                    "target_generation_config_fingerprint"
                ),
                "data_validation_level": (
                    "CURRENT_SNAPSHOT_VERIFIED"
                    if validated_integrity_baseline_sha256 is not None
                    else "CURRENT_SNAPSHOT_NOT_RUNNER_VERIFIED"
                ),
                "historical_source_status": baseline.get(
                    "historical_source_status", "SOURCE_SNAPSHOT_RECORDED"
                ),
            }
        )
    else:
        provenance.update(
            {
                "generation_config_fingerprint": manifest.get(
                    "target_generation_config_fingerprint"
                ),
                "data_validation_level": "POOL_MANIFEST_ONLY",
                "historical_source_status": manifest.get("historical_source_status", "UNKNOWN"),
            }
        )
    return provenance

def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")

def _finite_episode_return(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("episode_metrics episode_return must be numeric") from exc
    if not math.isfinite(converted):
        raise ValueError("episode_metrics episode_return must be finite")
    return converted

def _read_episode_returns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取每个连续完成 epoch 的有限原始回报。"""

    records: dict[int, float] = {}
    if not path.exists():
        raise ValueError("episode_metrics.jsonl is empty or does not exist")
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(
                    f"episode_metrics contains a blank line at {line_number}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"episode_metrics contains invalid JSON at line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"episode_metrics line {line_number} is not an object"
                )
            if record.get("record_type", "episode") != "episode":
                continue
            if "epoch" not in record and "episode" not in record:
                raise ValueError(
                    f"episode_metrics line {line_number} is missing epoch/episode"
                )
            raw_epoch = record.get("epoch", record.get("episode"))
            try:
                epoch = int(raw_epoch)
                epoch_as_float = float(raw_epoch)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"episode_metrics line {line_number} has invalid epoch"
                ) from exc
            if (
                isinstance(raw_epoch, bool)
                or not math.isfinite(epoch_as_float)
                or epoch_as_float != epoch
                or epoch < 1
            ):
                raise ValueError(
                    f"episode_metrics line {line_number} has invalid epoch: {raw_epoch!r}"
                )
            if epoch in records:
                raise ValueError(f"episode_metrics duplicate epoch: {epoch}")
            records[epoch] = _finite_episode_return(record.get("episode_return"))

    if not records:
        raise ValueError("episode_metrics.jsonl has no episode records")
    epochs = sorted(records)
    expected_epochs = list(range(1, len(epochs) + 1))
    if epochs != expected_epochs:
        raise ValueError(
            "episode_metrics epochs are not contiguous from 1: "
            f"expected {expected_epochs}, got {epochs}"
        )
    return (
        np.asarray(epochs, dtype=np.int64),
        np.asarray([records[epoch] for epoch in epochs], dtype=np.float64),
    )

def _full_window_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if int(window) <= 0:
        raise ValueError("moving-average window must be positive")
    moving = np.full(values.shape, np.nan, dtype=np.float64)
    if values.size >= int(window):
        for index in range(int(window) - 1, values.size):
            moving[index] = float(np.mean(values[index - int(window) + 1 : index + 1]))
    return moving

def update_training_epoch_return_plot(run_dir: Path) -> Path:
    """根据 epoch 回报生成当前正式 PPO 训练图。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    run_path = Path(run_dir)
    epochs, returns = _read_episode_returns(
        run_path / EPISODE_METRICS_FILENAME
    )
    ma20 = _full_window_moving_average(returns, 20)

    figure, axis = plt.subplots(figsize=(12, 6))
    try:
        axis.plot(
            epochs,
            returns,
            label="_nolegend_",
            color="#2563eb",
            linewidth=0.8,
            alpha=0.15,
        )
        valid = np.isfinite(ma20)
        if np.any(valid):
            axis.plot(
                epochs[valid],
                ma20[valid],
                label="MA20",
                color="#2563eb",
                linewidth=2.2,
                marker="o",
                markersize=4.5,
                markevery=10,
            )
        axis.set_xlabel("Training Epoch")
        axis.set_ylabel("Episode Return")
        axis.set_title("PPO Training Return (MA20)")
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        if axis.get_legend_handles_labels()[1]:
            axis.legend(frameon=False)
        figure.tight_layout()
        output_path = run_path / "plots" / TRAINING_EPOCH_RETURN_PLOT_FILENAME
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=320, facecolor="white")
    finally:
        plt.close(figure)
    return output_path


def _format_metric(value: Any, precision: int) -> str:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(converted):
        return "NA"
    return f"{converted:.{int(precision)}f}"


def _format_rate_percent(value: Any) -> str:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return "NA"
    return _format_metric(100.0 * converted, 1)


def format_training_epoch_log(
    record: dict[str, Any],
    *,
    target_epochs: int,
) -> str:
    """Format one compact formal PPO episode line from episode/update metrics."""

    epoch = record.get("epoch", record.get("episode", "?"))
    uav_move_rates = [record.get("uav0_move_rate"), record.get("uav1_move_rate")]
    if all(value is not None for value in uav_move_rates):
        try:
            uav_move_rate = sum(float(value) for value in uav_move_rates) / 2.0
        except (TypeError, ValueError):
            uav_move_rate = math.nan
    else:
        uav_move_rate = math.nan
    return (
        f"[PPO] Episode {epoch}/{int(target_epochs)} "
        f"| Return={_format_metric(record.get('episode_return'), 2)} "
        f"| N2EV={_format_metric(record.get('n2_post_ev', record.get('post_update_target_ev')), 3)} "
        f"| LongCorr={_format_metric(record.get('long_value_post_correlation'), 3)} "
        f"| CacheChange={_format_rate_percent(record.get('actual_cache_update_rate'))}% "
        f"| UAVMove={_format_rate_percent(uav_move_rate)}% "
        f"| Time={_format_metric(record.get('training_epoch_runtime_seconds', record.get('episode_runtime_seconds')), 1)}s"
    )


FORMAL_ACTOR_LR_COMPARISON_RUNS = (
    ("LR=1e-4", "formal_actor_lr_1e_minus_4_300", 1.0e-4),
    ("LR=3e-4", "formal_actor_lr_3e_minus_4_300", 3.0e-4),
    ("LR=1e-3", "formal_actor_lr_1e_minus_3_300", 1.0e-3),
    ("LR=3e-3", "formal_actor_lr_3e_minus_3_300", 3.0e-3),
)


def plot_formal_actor_lr_comparison_300(project_root: Path | None = None) -> Path:
    """Plot four complete 300-epoch runs; refuse to draw if any run is incomplete."""

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    runs_root = root / "runs"
    validated: list[tuple[str, np.ndarray, np.ndarray]] = []
    run_identities: list[tuple[Any, ...]] = []
    missing: list[str] = []

    for label, run_name, expected_lr in FORMAL_ACTOR_LR_COMPARISON_RUNS:
        run_path = runs_root / run_name
        metrics_path = run_path / EPISODE_METRICS_FILENAME
        manifest_path = run_path / RUN_MANIFEST_FILENAME
        summary_path = run_path / RUN_SUMMARY_FILENAME
        checkpoint_path = run_path / CHECKPOINTS_DIRECTORY / "latest_checkpoint.pt"
        try:
            if not metrics_path.is_file():
                raise ValueError("episode_metrics.jsonl is missing")
            if not manifest_path.is_file():
                raise ValueError("run_manifest.json is missing")
            if not summary_path.is_file():
                raise ValueError("run_summary.json is missing")
            if not checkpoint_path.is_file():
                raise ValueError("latest_checkpoint.pt is missing")
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            with summary_path.open("r", encoding="utf-8") as stream:
                summary = json.load(stream)
            if not isinstance(manifest, dict) or not isinstance(summary, dict):
                raise ValueError("manifest/summary must contain JSON objects")
            if int(manifest.get("target_episodes", -1)) != 300:
                raise ValueError("manifest target_episodes is not 300")
            if int(summary.get("target_episodes", -1)) != 300:
                raise ValueError("summary target_episodes is not 300")
            if int(summary.get("completed_episodes", -1)) != 300:
                raise ValueError("run_summary does not report 300 completed epochs")
            counters = manifest.get("training_counters")
            if not isinstance(counters, dict) or int(
                counters.get("completed_episodes", -1)
            ) != 300:
                raise ValueError("run manifest does not report 300 completed epochs")
            recorded_lr = float(
                manifest.get("macro_actor_learning_rate", manifest.get("actor_learning_rate"))
            )
            if not math.isclose(recorded_lr, expected_lr, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f"Macro Actor LR is {recorded_lr:g}, expected {expected_lr:g}")
            if int(manifest.get("architecture_version", -1)) != 10:
                raise ValueError("architecture_version is not 10")
            if int(manifest.get("checkpoint_version", -1)) != 33:
                raise ValueError("checkpoint_version is not 33")
            if manifest.get("macro_actor_advantage_mode") != "duration_aware_smdp_gae":
                raise ValueError("Macro Actor advantage mode does not match the formal run")
            if manifest.get("macro_critic_target_mode") != "two_step_smdp_td":
                raise ValueError("Macro Critic target is not N2")
            if manifest.get("long_value_target") != "realized_return_to_go":
                raise ValueError("Long Critic target does not match the formal run")
            if manifest.get("long_value_enabled") is not True:
                raise ValueError("independent Long Critic is not enabled")
            if int(manifest.get("seed", -1)) != 27005:
                raise ValueError("initialization seed does not match the shared seed 27005")
            if int(manifest.get("episode_seed_start", -1)) != 20001:
                raise ValueError("episode seed start does not match the shared order 20001")
            if Path(str(manifest.get("config_path", ""))).resolve() != (
                root / "configs" / "proposed" / "alternating_actor_updates_v1.toml"
            ).resolve():
                raise ValueError("run does not use the formal alternating_actor_updates_v1.toml")
            epochs, returns = _read_episode_returns(metrics_path)
            if len(epochs) != 300 or not np.array_equal(epochs, np.arange(1, 301)):
                raise ValueError("episode_metrics does not contain exactly epochs 1..300")
            run_identities.append(
                (
                    int(manifest["seed"]),
                    int(manifest["episode_seed_start"]),
                    str(Path(str(manifest["episode_pool"])).resolve()),
                    str(Path(str(manifest["config_path"])).resolve()),
                    str(manifest["config_raw_sha256"]),
                    str(manifest["config_semantic_sha256"]),
                    float(manifest.get("micro_actor_learning_rate", math.nan)),
                    float(manifest["macro_critic_learning_rate"]),
                    float(manifest["micro_critic_learning_rate"]),
                    float(manifest["long_value_learning_rate"]),
                )
            )
            validated.append((label, epochs, returns))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            missing.append(f"{run_name}: {exc}")

    if missing:
        raise RuntimeError(
            "Formal actor-LR comparison refused; incomplete or invalid run data: "
            + "; ".join(missing)
        )
    if len(set(run_identities)) != 1:
        raise RuntimeError(
            "Formal actor-LR comparison refused; shared seed, episode order/pool, "
            "config, Micro Actor LR, or Critic LR differs between runs."
        )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 6))
    try:
        for label, epochs, returns in validated:
            ma20 = _full_window_moving_average(returns, 20)
            valid = np.isfinite(ma20)
            axis.plot(epochs[valid], ma20[valid], label=label, linewidth=2.0)
        axis.set_xlabel("Training Epoch")
        axis.set_ylabel("Episode Return (MA20)")
        axis.set_title("Formal Macro Actor Learning-Rate Comparison (300 Epochs)")
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        if axis.get_legend_handles_labels()[1]:
            axis.legend(frameon=False)
        figure.tight_layout()
        output_path = root / "reports" / "plots" / "formal_actor_lr_comparison_300.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=320, facecolor="white")
    finally:
        plt.close(figure)
    return output_path

def _gradient_group_norm(parameters: Any) -> float:
    """返回一个诊断参数组在梯度裁剪前的 L2 范数。"""

    squared_norm = None
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        component = torch.sum(gradient.detach() ** 2)
        squared_norm = component if squared_norm is None else squared_norm + component
    if squared_norm is None:
        return 0.0
    return float(torch.sqrt(squared_norm).detach().cpu())

def _module_parameter_delta_norm(before_parameters: dict[str, Any], module: Any) -> float:
    """返回模块具名参数变化量的范数，仅用于诊断。"""

    delta_squared = 0.0
    current_names = set()
    for name, parameter in module.named_parameters():
        key = str(name)
        current_names.add(key)
        if key not in before_parameters:
            raise RuntimeError(f"diagnostic module parameter disappeared: {key}")
        delta_squared += float(
            torch.sum((parameter.detach() - before_parameters[key]) ** 2).cpu()
        )
    if current_names != set(before_parameters):
        raise RuntimeError("diagnostic module parameter names changed during PPO update")
    return float(np.sqrt(delta_squared))

def _matching_termination_metrics(
    summaries: Any,
) -> dict[str, Any]:
    """汇总 Matching 停止原因，不改变 Matching 决策。"""

    natural_stable_count = 0
    max_updates_hit_count = 0
    update_counts: list[int] = []
    for summary in summaries:
        if not summary or not isinstance(summary, dict):
            continue
        stop_reason = str(summary.get("stop_reason", ""))
        if stop_reason == "converged_by_no_improvement":
            natural_stable_count += 1
        elif stop_reason == "stopped_by_max_updates":
            max_updates_hit_count += 1
        if "updates" not in summary:
            continue
        update_count = int(summary["updates"])
        if update_count < 0:
            raise ValueError("Matching update count must be non-negative")
        update_counts.append(update_count)
    return _matching_metrics_from_counts(
        natural_stable_count,
        max_updates_hit_count,
        update_counts,
    )

def _matching_metrics_from_counts(
    natural_stable_count: int,
    max_updates_hit_count: int,
    update_counts: Any,
) -> dict[str, Any]:
    """从 Exact-Q worker 汇总数据中构建 Matching 指标。"""

    counts_list = [int(value) for value in update_counts]
    if any(value < 0 for value in counts_list):
        raise ValueError("Matching update count must be non-negative")
    if not counts_list:
        return {
            "natural_stable_count": int(natural_stable_count),
            "max_updates_hit_count": int(max_updates_hit_count),
            "max_updates_hit_rate": 0.0,
            "update_count_mean": 0.0,
            "update_count_p95": 0.0,
            "update_count_max": 0,
        }
    counts = np.asarray(counts_list, dtype=np.float64)
    return {
        "natural_stable_count": int(natural_stable_count),
        "max_updates_hit_count": int(max_updates_hit_count),
        "max_updates_hit_rate": float(max_updates_hit_count / len(counts_list)),
        "update_count_mean": float(np.mean(counts)),
        "update_count_p95": float(np.percentile(counts, 95.0)),
        "update_count_max": int(np.max(counts)),
    }

def _kkt_metrics_from_diagnostics(records: Any) -> dict[str, Any]:
    """从 Matching 汇总数据中聚合只读 KKT 求解诊断。"""

    normalized: list[dict[str, Any]] = []
    for record in records or ():
        if not isinstance(record, dict):
            continue
        normalized.append(record)

    solve_count = len(normalized)
    converged_count = sum(
        bool(record.get("converged_by_tolerance", False))
        for record in normalized
    )
    max_iter_hit_count = sum(
        bool(record.get("hit_max_bisection_iterations", False))
        for record in normalized
    )
    finite_count = sum(bool(record.get("finite", False)) for record in normalized)

    def finite_values(field: str) -> np.ndarray:
        values = []
        for record in normalized:
            try:
                value = float(record.get(field, math.nan))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return np.asarray(values, dtype=np.float64)

    iterations = finite_values("iterations")
    resource_residuals = finite_values("resource_sum_residual")
    stationarity_residuals = finite_values("stationarity_residual")
    complementary_residuals = finite_values("complementary_slackness_residual")

    def mean(values: np.ndarray) -> float:
        return 0.0 if values.size == 0 else float(np.mean(values))

    def maximum(values: np.ndarray) -> float:
        return math.nan if values.size == 0 else float(np.max(values))

    return {
        "solve_count": int(solve_count),
        "converged_count": int(converged_count),
        "converged_rate": (
            float(converged_count / solve_count) if solve_count else 0.0
        ),
        "max_iter_hit_count": int(max_iter_hit_count),
        "max_iter_hit_rate": (
            float(max_iter_hit_count / solve_count) if solve_count else 0.0
        ),
        "finite_count": int(finite_count),
        "finite_rate": float(finite_count / solve_count) if solve_count else 0.0,
        "iteration_mean": mean(iterations),
        "iteration_p95": (
            0.0 if iterations.size == 0 else float(np.percentile(iterations, 95.0))
        ),
        "iteration_max": int(np.max(iterations)) if iterations.size else 0,
        "resource_residual_mean": mean(resource_residuals),
        "resource_residual_max": maximum(resource_residuals),
        "stationarity_residual_mean": mean(stationarity_residuals),
        "stationarity_residual_max": maximum(stationarity_residuals),
        "complementary_slackness_residual_mean": mean(complementary_residuals),
        "complementary_slackness_residual_max": maximum(complementary_residuals),
    }

RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_SUMMARY_FILENAME = "run_summary.json"
CHECKPOINTS_DIRECTORY = "checkpoints"
PLOTS_DIRECTORY = "plots"
FRESH_RUN_ARTIFACTS = (
    RUN_MANIFEST_FILENAME,
    RUN_SUMMARY_FILENAME,
    "episode_metrics.jsonl",
    "update_metrics.jsonl",
    "critic_diagnostics.jsonl",
    "slot_metrics.jsonl",
    f"{CHECKPOINTS_DIRECTORY}/latest_checkpoint.pt",
    f"{CHECKPOINTS_DIRECTORY}/best_train_return_checkpoint.pt",
    f"plots/{TRAINING_EPOCH_RETURN_PLOT_FILENAME}",
    MACRO_TRANSITION_TRACE_FILENAME,
)


def checkpoint_path_for_write(run_dir: Path, filename: str) -> Path:
    """返回正式 checkpoint 的保存路径。"""

    return Path(run_dir) / CHECKPOINTS_DIRECTORY / filename


def checkpoint_path_for_resume(run_dir: Path, filename: str) -> Path:
    """返回正式 checkpoint 的恢复路径。"""

    return Path(run_dir) / CHECKPOINTS_DIRECTORY / filename


def resolve_checkpoint_reference(checkpoint: Path) -> Path:
    """解析 checkpoint 文件、正式 checkpoint 目录或 run 目录。"""

    path = Path(checkpoint).expanduser()
    if path.is_dir():
        if path.name == CHECKPOINTS_DIRECTORY:
            return checkpoint_path_for_resume(path.parent, "latest_checkpoint.pt")
        return checkpoint_path_for_resume(path, "latest_checkpoint.pt")
    if path.parent.name == CHECKPOINTS_DIRECTORY:
        return path
    return path.parent / CHECKPOINTS_DIRECTORY / path.name


def ensure_run_artifact_directories(run_dir: Path) -> None:
    """创建正式 checkpoint 和 plot 目录。"""

    run_path = Path(run_dir)
    (run_path / CHECKPOINTS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    (run_path / PLOTS_DIRECTORY).mkdir(parents=True, exist_ok=True)


def ensure_fresh_run_dir_is_clean(run_dir: Path) -> None:
    """如果新 run 会与现有实验混合，则拒绝启动。"""

    run_path = Path(run_dir)
    if not run_path.exists():
        return
    if not run_path.is_dir():
        raise RuntimeError(
            "fresh PPO run-dir already exists but is not a directory; "
            "use a new --run-dir or --resume"
        )
    existing_artifacts = [
        relative_path
        for relative_path in FRESH_RUN_ARTIFACTS
        if (run_path / relative_path).exists()
    ]
    for checkpoint_directory in (run_path / CHECKPOINTS_DIRECTORY, run_path):
        if checkpoint_directory.is_dir():
            existing_artifacts.extend(
                str(path.relative_to(run_path))
                for path in checkpoint_directory.glob("*.pt")
                if str(path.relative_to(run_path)) not in existing_artifacts
            )
    if existing_artifacts:
        raise RuntimeError(
            "fresh PPO run-dir already contains experiment artifacts; "
            "use a new --run-dir or --resume"
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream, parse_constant=_parse_nonfinite)
    if not isinstance(payload, dict):
        raise ValueError(f"reporting JSON root must be an object: {path}")
    return payload


def _parse_nonfinite(value: str) -> float:
    return float(value)


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    nan_inf_detected = False
    if not path.exists():
        return records, nan_inf_detected
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"reporting JSONL contains a blank line: {path}:{line_number}")
            try:
                record = json.loads(line, parse_constant=_parse_nonfinite)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"reporting JSONL contains invalid JSON: {path}:{line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"reporting JSONL line must be an object: {path}:{line_number}"
                )
            nan_inf_detected = nan_inf_detected or _contains_nonfinite(record)
            records.append(record)
    return records, nan_inf_detected


def _finite_float(value: Any, field: str) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reporting field {field} must be numeric or null") from exc
    return converted if math.isfinite(converted) else None


def _episode_records(path: Path) -> tuple[list[dict[str, Any]], bool]:
    raw_records, nan_inf_detected = _read_jsonl(path)
    by_epoch: dict[int, dict[str, Any]] = {}
    for record in raw_records:
        if record.get("record_type", "episode") != "episode":
            continue
        try:
            epoch = int(record["epoch"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"episode_metrics record has invalid epoch: {record}") from exc
        if epoch < 1:
            raise ValueError(f"episode_metrics epoch must be >=1: {epoch}")
        if epoch in by_epoch:
            raise ValueError(f"episode_metrics duplicate epoch: {epoch}")
        by_epoch[epoch] = record
    ordered = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    return ordered, nan_inf_detected


def _metric_values(
    records: list[dict[str, Any]], field: str, *, allow_null: bool = True
) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _finite_float(record.get(field), field)
        if value is None:
            if not allow_null:
                raise ValueError(f"episode_metrics field {field} is missing or null")
            continue
        values.append(value)
    return values


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _rolling_mean(
    returns: list[float], epochs: list[int], window: int
) -> tuple[float | None, float | None]:
    if not returns:
        return None, None
    actual_window = min(int(window), len(returns))
    return float(sum(returns[-actual_window:]) / actual_window), None


def _rolling_max(
    returns: list[float], epochs: list[int], window: int
) -> tuple[float | None, int | None]:
    if len(returns) < int(window):
        return None, None
    candidates = [
        (
            float(sum(returns[index - window + 1 : index + 1]) / window),
            int(epochs[index]),
        )
        for index in range(window - 1, len(returns))
    ]
    best = max(candidates, key=lambda item: item[0])
    return best

def _cache_operation_epoch_metrics(
    transitions: Any,
    initial_cache: np.ndarray,
    cache_nodes: dict[str, Any],
    service_sizes_mb: np.ndarray,
) -> dict[str, Any]:
    """汇总 KEEP/UPDATE 行为，不影响 rollout 或 PPO 更新。"""

    operation_names = ("KEEP", "UPDATE")
    roles = ("SV", "RSU", "UAV")
    node_roles = [str(value).upper() for value in cache_nodes["node_roles"]]
    node_count = len(node_roles)
    capacities = np.asarray(cache_nodes["capacities_mb"], dtype=np.float64).reshape(-1)
    sizes = np.asarray(service_sizes_mb, dtype=np.float64).reshape(-1)
    operation_counts = {
        role: {operation: 0 for operation in operation_names} for role in roles
    }
    role_samples = {role: 0 for role in roles}
    previous_cache = np.asarray(initial_cache, dtype=np.int8).copy()
    rsu_service_count: list[float] = []
    rsu_capacity_utilization: list[float] = []
    rsu_nonempty: list[bool] = []
    rsu_requested_updates = 0
    rsu_actual_changes = 0
    rsu_samples = 0

    for transition in transitions:
        next_cache = np.asarray(transition.next_cache_state, dtype=np.int8)
        if next_cache.shape != previous_cache.shape:
            raise ValueError("cache behavior diagnostics encountered a state shape change")
        actions = {
            int(item["head_index"]): item
            for item in transition.cache_conditional_actions
            if "head_index" in item
        }
        for node_index, role in enumerate(node_roles):
            role_samples[role] = role_samples.get(role, 0) + 1
            action = actions.get(node_index)
            if action is not None and role in operation_counts:
                conditional_action = str(action.get("conditional_action", "")).upper()
                raw_operation = str(action.get("operation_type", "")).upper()
                if conditional_action == "KEEP" or raw_operation == "NO_CHANGE":
                    operation = "KEEP"
                elif conditional_action == "TARGET_CONFIGURATION" or raw_operation in {
                    "ADD",
                    "DELETE",
                    "SWAP",
                }:
                    operation = "UPDATE"
                else:
                    operation = ""
                if operation in operation_counts[role]:
                    operation_counts[role][operation] += 1
            if role != "RSU":
                continue
            row = next_cache[node_index]
            used_mb = float(np.dot(row.astype(np.float64), sizes))
            capacity = float(capacities[node_index])
            if not math.isfinite(used_mb) or not math.isfinite(capacity) or capacity <= 0.0:
                raise FloatingPointError("RSU cache behavior diagnostic is non-finite")
            rsu_service_count.append(float(row.sum()))
            rsu_capacity_utilization.append(used_mb / capacity)
            rsu_nonempty.append(bool(row.sum() > 0))
            rsu_requested_updates += int(
                np.asarray(transition.action, dtype=np.int8)[node_index] == 1
            )
            rsu_actual_changes += int(
                np.any(row != previous_cache[node_index])
            )
            rsu_samples += 1
        previous_cache = next_cache.copy()

    def _rate(count: int, denominator: int) -> float:
        return 0.0 if denominator <= 0 else float(count) / float(denominator)

    operation_rates = {
        role: {
            operation: _rate(count, role_samples[role])
            for operation, count in counts.items()
        }
        for role, counts in operation_counts.items()
    }
    return {
        "cache_operation_counts": operation_counts,
        "cache_operation_rates": operation_rates,
        "RSU_mean_cached_service_count": (
            0.0 if not rsu_service_count else float(np.mean(rsu_service_count))
        ),
        "RSU_nonempty_fraction": (
            0.0 if not rsu_nonempty else float(np.mean(rsu_nonempty))
        ),
        "RSU_cache_capacity_utilization": (
            0.0
            if not rsu_capacity_utilization
            else float(np.mean(rsu_capacity_utilization))
        ),
        "RSU_requested_update_rate": _rate(rsu_requested_updates, rsu_samples),
        "RSU_actual_change_rate": _rate(rsu_actual_changes, rsu_samples),
        "RSU_operation_sample_count": int(rsu_samples),
    }

def _uav_episode_diagnostics(
    *,
    available: bool,
    move_probabilities: list[float],
    stay_probabilities: list[float],
    logit_margins: list[float],
    move_feasible: list[bool],
    move_selected: list[bool],
    actual_movement: list[bool],
    move_advantages: list[float],
    stay_advantages: list[float],
    global_advantages: list[float],
    global_return_targets: list[float],
    rewards: list[float],
    q_values: list[float],
    movement_costs: list[float],
    delays: list[float],
    energies: list[float],
    old_log_probs: list[float],
    new_log_probs: list[float],
    ratios: list[float],
    clipped: list[bool],
    feasible_action_counts: list[int],
    move_candidate_counts: list[int],
) -> dict[str, Any]:
    """不再调用策略，汇总已采样的 UAV 诊断。按经核验的 actual_movement 标记对 MOVE/STAY 条件字段分组，并单独保留所选操作标签，供后续审计动作与物理结果是否一致；不会把动作索引当作移动结果。"""

    empty = {
        "uav_move_probability_mean": None,
        "uav_move_probability_median": None,
        "uav_move_probability_p10": None,
        "uav_move_probability_p90": None,
        "uav_move_probability_gt_0_5_rate": None,
        "uav_stay_probability_mean": None,
        "uav_head_entropy": None,
        "uav_binary_entropy_mean": None,
        "uav_stay_move_logit_margin_mean": None,
        "uav_stay_move_logit_margin_median": None,
        "uav_move_feasible_count": None,
        "uav_move_feasible_rate": None,
        "uav_move_selected_count": None,
        "uav_move_selected_rate": None,
        "uav_move_given_selected": None,
        "uav_action_advantage_logging_available": False,
        "mean_advantage_for_uav_move": None,
        "median_advantage_for_uav_move": None,
        "mean_advantage_for_uav_stay": None,
        "median_advantage_for_uav_stay": None,
        "uav_action_conditional_logging_available": False,
        "uav_move_count": None,
        "uav_stay_count": None,
        "uav_move_adv_mean": None,
        "uav_stay_adv_mean": None,
        "uav_move_adv_median": None,
        "uav_stay_adv_median": None,
        "uav_move_adv_positive_fraction": None,
        "uav_stay_adv_positive_fraction": None,
        "uav_move_reward_mean": None,
        "uav_stay_reward_mean": None,
        "uav_move_q_total_mean": None,
        "uav_stay_q_total_mean": None,
        "uav_move_movement_cost_mean": None,
        "uav_stay_movement_cost_mean": None,
        "uav_move_delay_mean": None,
        "uav_stay_delay_mean": None,
        "uav_move_energy_mean": None,
        "uav_stay_energy_mean": None,
        "uav_move_return_target_mean": None,
        "uav_stay_return_target_mean": None,
        "uav_move_old_log_prob_mean": None,
        "uav_stay_old_log_prob_mean": None,
        "uav_move_new_log_prob_mean": None,
        "uav_stay_new_log_prob_mean": None,
        "uav_move_ratio_mean": None,
        "uav_stay_ratio_mean": None,
        "uav_move_clip_fraction": None,
        "uav_stay_clip_fraction": None,
        "uav_selected_move_count": None,
        "uav_selected_stay_count": None,
        "uav_selected_move_actual_move_count": None,
        "uav_selected_move_actual_stay_count": None,
        "uav_selected_stay_actual_move_count": None,
        "uav_selected_stay_actual_stay_count": None,
        "uav_feasible_action_count_mean": None,
        "uav_feasible_action_count_min": None,
        "uav_feasible_action_count_max": None,
        "uav_move_candidate_count_mean": None,
        "uav_move_candidate_count_min": None,
        "uav_move_candidate_count_max": None,
        "uav_action_mask_summary": None,
    }
    if not available:
        return empty

    move = np.asarray(move_probabilities, dtype=np.float64)
    stay = np.asarray(stay_probabilities, dtype=np.float64)
    margins = np.asarray(logit_margins, dtype=np.float64)
    feasible = np.asarray(move_feasible, dtype=bool)
    selected = np.asarray(move_selected, dtype=bool)
    actual = np.asarray(actual_movement, dtype=bool)
    expected_length = len(move)
    if expected_length <= 0 or not all(
        array.ndim == 1 and len(array) == expected_length
        for array in (stay, margins, feasible, selected, actual)
    ):
        raise AssertionError("UAV diagnostic arrays must be aligned and non-empty")
    if not (
        np.all(np.isfinite(move))
        and np.all(np.isfinite(stay))
        and np.all(np.isfinite(margins))
    ):
        raise FloatingPointError("UAV diagnostic arrays contain NaN/Inf")
    if not (
        np.all((move >= 0.0) & (move <= 1.0))
        and np.all((stay >= 0.0) & (stay <= 1.0))
        and np.allclose(move + stay, 1.0, rtol=0.0, atol=1.0e-6)
    ):
        raise AssertionError("UAV diagnostic probabilities are invalid")

    def _aligned_float_array(values: list[float], field: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or len(array) != expected_length:
            raise AssertionError(f"{field} must align with UAV diagnostic samples")
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(f"{field} contains NaN/Inf")
        return array

    advantages = _aligned_float_array(global_advantages, "global_advantages")
    return_targets = _aligned_float_array(
        global_return_targets, "global_return_targets"
    )
    rewards_a = _aligned_float_array(rewards, "rewards")
    q_values_a = _aligned_float_array(q_values, "q_values")
    movement_costs_a = _aligned_float_array(movement_costs, "movement_costs")
    delays_a = _aligned_float_array(delays, "delays")
    energies_a = _aligned_float_array(energies, "energies")
    old_log_probs_a = _aligned_float_array(old_log_probs, "old_log_probs")
    new_log_probs_a = _aligned_float_array(new_log_probs, "new_log_probs")
    ratios_a = _aligned_float_array(ratios, "ratios")
    if np.any(ratios_a < 0.0):
        raise AssertionError("UAV branch ratios must be non-negative")
    clipped_a = np.asarray(clipped, dtype=bool)
    if clipped_a.ndim != 1 or len(clipped_a) != expected_length:
        raise AssertionError("clipped must align with UAV diagnostic samples")
    feasible_counts_a = np.asarray(feasible_action_counts, dtype=np.float64)
    candidate_counts_a = np.asarray(move_candidate_counts, dtype=np.float64)
    if not (
        feasible_counts_a.ndim == candidate_counts_a.ndim == 1
        and len(feasible_counts_a) == len(candidate_counts_a) == expected_length
        and np.all(np.isfinite(feasible_counts_a))
        and np.all(np.isfinite(candidate_counts_a))
        and np.all(feasible_counts_a >= 0.0)
        and np.all(candidate_counts_a >= 0.0)
        and np.all(feasible_counts_a == np.floor(feasible_counts_a))
        and np.all(candidate_counts_a == np.floor(candidate_counts_a))
    ):
        raise AssertionError("UAV action mask counts are invalid")

    def _percentile(values: np.ndarray, percentile: float) -> float:
        return float(np.percentile(values, percentile))

    def _advantage_summary(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        values_a = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values_a)):
            raise FloatingPointError("UAV action advantages contain NaN/Inf")
        return float(np.mean(values_a)), float(np.median(values_a))

    def _group_stats(mask: np.ndarray) -> dict[str, Any]:
        if mask.ndim != 1 or len(mask) != expected_length:
            raise AssertionError("UAV action-conditional mask is not aligned")
        if not bool(np.any(mask)):
            return {
                "count": 0,
                "adv_mean": None,
                "adv_median": None,
                "adv_positive_fraction": None,
                "reward_mean": None,
                "q_total_mean": None,
                "movement_cost_mean": None,
                "delay_mean": None,
                "energy_mean": None,
                "return_target_mean": None,
                "old_log_prob_mean": None,
                "new_log_prob_mean": None,
                "ratio_mean": None,
                "clip_fraction": None,
            }

        def _mean(values: np.ndarray) -> float:
            result = float(np.mean(values[mask]))
            if not np.isfinite(result):
                raise FloatingPointError("UAV action-conditional mean is non-finite")
            return result

        def _median(values: np.ndarray) -> float:
            result = float(np.median(values[mask]))
            if not np.isfinite(result):
                raise FloatingPointError("UAV action-conditional median is non-finite")
            return result

        return {
            "count": int(np.sum(mask)),
            "adv_mean": _mean(advantages),
            "adv_median": _median(advantages),
            "adv_positive_fraction": float(np.mean(advantages[mask] > 0.0)),
            "reward_mean": _mean(rewards_a),
            "q_total_mean": _mean(q_values_a),
            "movement_cost_mean": _mean(movement_costs_a),
            "delay_mean": _mean(delays_a),
            "energy_mean": _mean(energies_a),
            "return_target_mean": _mean(return_targets),
            "old_log_prob_mean": _mean(old_log_probs_a),
            "new_log_prob_mean": _mean(new_log_probs_a),
            "ratio_mean": _mean(ratios_a),
            "clip_fraction": float(np.mean(clipped_a[mask])),
        }

    move_advantage_mean, move_advantage_median = _advantage_summary(move_advantages)
    stay_advantage_mean, stay_advantage_median = _advantage_summary(stay_advantages)
    actual_move_stats = _group_stats(actual)
    actual_stay_stats = _group_stats(~actual)
    probability_entropy = -(
        np.where(move > 0.0, move * np.log(np.maximum(move, 1.0e-12)), 0.0)
        + np.where(stay > 0.0, stay * np.log(np.maximum(stay, 1.0e-12)), 0.0)
    )
    selected_count = int(np.sum(selected))
    return {
        "uav_move_probability_mean": float(np.mean(move)),
        "uav_move_probability_median": float(np.median(move)),
        "uav_move_probability_p10": _percentile(move, 10.0),
        "uav_move_probability_p90": _percentile(move, 90.0),
        "uav_move_probability_gt_0_5_rate": float(np.mean(move > 0.5)),
        "uav_stay_probability_mean": float(np.mean(stay)),
        "uav_head_entropy": float(np.mean(probability_entropy)),
        "uav_binary_entropy_mean": float(np.mean(probability_entropy)),
        "uav_stay_move_logit_margin_mean": float(np.mean(margins)),
        "uav_stay_move_logit_margin_median": float(np.median(margins)),
        "uav_move_feasible_count": int(np.sum(feasible)),
        "uav_move_feasible_rate": float(np.mean(feasible)),
        "uav_move_selected_count": selected_count,
        "uav_move_selected_rate": float(np.mean(selected)),
        "uav_move_given_selected": (
            None
            if selected_count == 0
            else float(np.sum(actual & selected)) / float(selected_count)
        ),
        "uav_action_advantage_logging_available": True,
        "mean_advantage_for_uav_move": move_advantage_mean,
        "median_advantage_for_uav_move": move_advantage_median,
        "mean_advantage_for_uav_stay": stay_advantage_mean,
        "median_advantage_for_uav_stay": stay_advantage_median,
        "uav_action_conditional_logging_available": True,
        "uav_move_count": actual_move_stats["count"],
        "uav_stay_count": actual_stay_stats["count"],
        "uav_move_adv_mean": actual_move_stats["adv_mean"],
        "uav_stay_adv_mean": actual_stay_stats["adv_mean"],
        "uav_move_adv_median": actual_move_stats["adv_median"],
        "uav_stay_adv_median": actual_stay_stats["adv_median"],
        "uav_move_adv_positive_fraction": actual_move_stats[
            "adv_positive_fraction"
        ],
        "uav_stay_adv_positive_fraction": actual_stay_stats[
            "adv_positive_fraction"
        ],
        "uav_move_reward_mean": actual_move_stats["reward_mean"],
        "uav_stay_reward_mean": actual_stay_stats["reward_mean"],
        "uav_move_q_total_mean": actual_move_stats["q_total_mean"],
        "uav_stay_q_total_mean": actual_stay_stats["q_total_mean"],
        "uav_move_movement_cost_mean": actual_move_stats["movement_cost_mean"],
        "uav_stay_movement_cost_mean": actual_stay_stats["movement_cost_mean"],
        "uav_move_delay_mean": actual_move_stats["delay_mean"],
        "uav_stay_delay_mean": actual_stay_stats["delay_mean"],
        "uav_move_energy_mean": actual_move_stats["energy_mean"],
        "uav_stay_energy_mean": actual_stay_stats["energy_mean"],
        "uav_move_return_target_mean": actual_move_stats["return_target_mean"],
        "uav_stay_return_target_mean": actual_stay_stats["return_target_mean"],
        "uav_move_old_log_prob_mean": actual_move_stats["old_log_prob_mean"],
        "uav_stay_old_log_prob_mean": actual_stay_stats["old_log_prob_mean"],
        "uav_move_new_log_prob_mean": actual_move_stats["new_log_prob_mean"],
        "uav_stay_new_log_prob_mean": actual_stay_stats["new_log_prob_mean"],
        "uav_move_ratio_mean": actual_move_stats["ratio_mean"],
        "uav_stay_ratio_mean": actual_stay_stats["ratio_mean"],
        "uav_move_clip_fraction": actual_move_stats["clip_fraction"],
        "uav_stay_clip_fraction": actual_stay_stats["clip_fraction"],
        "uav_selected_move_count": selected_count,
        "uav_selected_stay_count": int(expected_length - selected_count),
        "uav_selected_move_actual_move_count": int(np.sum(selected & actual)),
        "uav_selected_move_actual_stay_count": int(np.sum(selected & ~actual)),
        "uav_selected_stay_actual_move_count": int(np.sum(~selected & actual)),
        "uav_selected_stay_actual_stay_count": int(np.sum(~selected & ~actual)),
        "uav_feasible_action_count_mean": float(np.mean(feasible_counts_a)),
        "uav_feasible_action_count_min": int(np.min(feasible_counts_a)),
        "uav_feasible_action_count_max": int(np.max(feasible_counts_a)),
        "uav_move_candidate_count_mean": float(np.mean(candidate_counts_a)),
        "uav_move_candidate_count_min": int(np.min(candidate_counts_a)),
        "uav_move_candidate_count_max": int(np.max(candidate_counts_a)),
        "uav_action_mask_summary": {
            "operation_width": 2,
            "feasible_operation_count_mean": float(np.mean(feasible_counts_a)),
            "feasible_operation_count_min": int(np.min(feasible_counts_a)),
            "feasible_operation_count_max": int(np.max(feasible_counts_a)),
            "move_candidate_count_mean": float(np.mean(candidate_counts_a)),
            "move_candidate_count_min": int(np.min(candidate_counts_a)),
            "move_candidate_count_max": int(np.max(candidate_counts_a)),
        },
    }

def _critic_episode_metrics(
    old_values: list[float],
    raw_targets: list[float],
) -> tuple[float | None, float, float]:
    """将采集阶段的 Critic 价值与正式 GAE target 进行汇总比较。"""

    values = np.asarray(old_values, dtype=np.float64)
    targets = np.asarray(raw_targets, dtype=np.float64)
    if values.shape != targets.shape or values.ndim != 1 or not len(values):
        raise ValueError("critic episode metric arrays must be aligned and non-empty")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(targets)):
        raise FloatingPointError("critic episode metrics contain NaN/Inf")
    target_variance = float(np.var(targets))
    value_std = float(np.std(values))
    target_std = float(np.std(targets))
    if not all(np.isfinite(value) for value in (target_variance, value_std, target_std)):
        raise FloatingPointError("critic episode metric summary is non-finite")
    explained_variance: float | None
    if target_variance <= 1.0e-12:
        explained_variance = None
    else:
        explained_variance = float(
            1.0 - float(np.var(targets - values)) / target_variance
        )
        if not np.isfinite(explained_variance):
            raise FloatingPointError("critic explained variance is non-finite")
    return explained_variance, value_std, target_std

def _append_critic_diagnostics(
    path: Path,
    *,
    seed: int,
    epoch: int,
    chunk_origin_start: int,
    optimizer_steps: int,
    rollout: Any,
    update: Any,
    cache_count: int | None = None,
) -> None:
    """追加 PPO 更新中已计算的 Critic 原始目标值。"""

    transitions = rollout.transitions
    if not (
        len(transitions)
        == len(update.raw_advantages)
        == len(update.ppo_advantages)
        == len(update.raw_returns)
    ):
        raise AssertionError("critic diagnostics length mismatch")
    for name, values in (
        ("Cache", update.cache_q_component_diagnostics),
        ("UAV", update.uav_q_component_diagnostics),
    ):
        if values is not None and len(values) != len(transitions):
            raise AssertionError(f"{name} Q-component diagnostics length mismatch")
    for transition_index, (transition, raw_advantage, raw_return) in enumerate(zip(
        transitions,
        update.raw_advantages,
        update.raw_returns,
        strict=True,
    )):
        old_value = float(transition.old_value)
        target = float(raw_return)
        residual = old_value - target
        normalized_residual = residual / float(update.critic_return_scale)
        global_advantage = float(update.ppo_advantages[transition_index])
        values = (
            old_value,
            float(raw_advantage),
            global_advantage,
            target,
            update.bootstrap_value,
            residual,
        )
        if not all(np.isfinite(value) for value in values):
            raise FloatingPointError("critic diagnostics contain non-finite values")
        record: dict[str, Any] = {
            "record_type": "critic_transition",
            "seed": int(seed),
            "episode": int(epoch),
            "epoch": int(epoch),
            "chunk_origin_start": int(chunk_origin_start),
            "optimizer_steps": int(optimizer_steps),
            "origin": int(transition.origin_index),
            "slot": int(transition.slot),
            "done": bool(transition.done),
            "old_value": old_value,
            "raw_advantage": float(raw_advantage),
            "global_advantage": global_advantage,
            "advantage_source": "actual_PPO_advantage",
            "raw_return": target,
            "global_return_target": target,
            "value_target": target,
            "bootstrap_value": float(update.bootstrap_value),
            "value_residual": residual,
            "abs_value_residual": abs(residual),
            "critic_return_scale": float(update.critic_return_scale),
            "normalized_value_residual": normalized_residual,
        }
        if update.teacher_generation_executed:
            record.update(
                {
                    "critic_bootstrap_teacher_warmup_complete": bool(
                        update.critic_bootstrap_teacher_warmup_complete
                    ),
                    "critic_optimizer_steps_before_teacher": int(
                        update.critic_optimizer_steps_before_teacher or 0
                    ),
                    "critic_optimizer_steps_after_update": int(
                        update.critic_optimizer_steps_after_update
                    ),
                    "popart_mean_before_teacher": update.popart_mean_before_teacher,
                    "popart_std_before_teacher": update.popart_std_before_teacher,
                    "critic_parameter_digest_before_teacher": (
                        update.critic_parameter_digest_before_teacher
                    ),
                    "teacher_source": update.teacher_source,
                    "max_q_component_reconstruction_error": float(
                        update.max_q_component_reconstruction_error
                    ),
                }
            )
            if update.cache_q_component_diagnostics is not None:
                record["cache_q_components"] = update.cache_q_component_diagnostics[
                    transition_index
                ]
            if update.uav_q_component_diagnostics is not None:
                record["uav_q_components"] = update.uav_q_component_diagnostics[
                    transition_index
                ]
        if transition.uav_operation_actions is not None:
            if cache_count is None:
                raise AssertionError(
                    "cache_count is required for joint UAV action diagnostics"
                )
            if (
                transition.uav_operation_masks is None
                or transition.uav_candidate_counts is None
                or update.uav_branch_old_log_probs is None
                or update.uav_branch_new_log_probs is None
                or update.uav_branch_ratios is None
                or update.uav_branch_clipped is None
            ):
                raise AssertionError(
                    "joint UAV action diagnostics are incomplete"
                )
            operation_actions = np.asarray(
                transition.uav_operation_actions, dtype=np.int64
            ).reshape(-1)
            operation_masks = np.asarray(
                transition.uav_operation_masks, dtype=bool
            )
            candidate_counts = np.asarray(
                transition.uav_candidate_counts, dtype=np.int64
            ).reshape(-1)
            actual_flags = np.asarray(
                [
                    bool(action.get("actual_moved", False))
                    for action in transition.uav_conditional_actions
                ],
                dtype=bool,
            ).reshape(-1)
            uav_count = len(operation_actions)
            if not (
                operation_masks.shape == (uav_count, 2)
                and len(candidate_counts) == uav_count
                and len(actual_flags) == uav_count
                and np.all(operation_actions >= 0)
                and np.all(operation_actions < 2)
                and np.all(candidate_counts >= 0)
                and np.all(np.isfinite(operation_masks))
            ):
                raise AssertionError("joint UAV action diagnostic shapes are invalid")
            if int(np.sum(actual_flags)) != int(transition.actual_uav_movement_count):
                raise AssertionError(
                    "UAV actual movement flags disagree with rollout movement count"
                )
            branch_old = float(update.uav_branch_old_log_probs[transition_index])
            branch_new = float(update.uav_branch_new_log_probs[transition_index])
            branch_ratio = float(update.uav_branch_ratios[transition_index])
            branch_clipped = bool(update.uav_branch_clipped[transition_index])
            branch_values = (branch_old, branch_new, branch_ratio)
            if not all(np.isfinite(value) for value in branch_values):
                raise FloatingPointError(
                    "joint UAV branch action diagnostics are non-finite"
                )
            cache_action = np.asarray(
                transition.action[: int(cache_count)], dtype=np.int8
            ).reshape(-1)
            if len(cache_action) != int(cache_count) or not np.all(
                (cache_action == 0) | (cache_action == 1)
            ):
                raise AssertionError("selected Cache action is not binary")
            record.update(
                {
                    "selected_cache_action": [int(value) for value in cache_action],
                    "selected_uav_action": [
                        "MOVE" if int(value) == 1 else "STAY"
                        for value in operation_actions
                    ],
                    "actual_uav_move": [int(value) for value in actual_flags],
                    "old_uav_log_prob": branch_old,
                    "new_uav_log_prob": branch_new,
                    "uav_ratio": branch_ratio,
                    "uav_clipped": branch_clipped,
                    "uav_log_prob_scope": "joint_uav_branch_per_transition",
                    "feasible_uav_action_count": [
                        int(value) for value in operation_masks.sum(axis=1)
                    ],
                    "uav_action_mask": [
                        [int(value) for value in row]
                        for row in operation_masks
                    ],
                    "uav_move_candidate_count": [
                        int(value) for value in candidate_counts
                    ],
                    "reward": float(transition.reward),
                    "q_total": float(transition.actual_task_utility),
                    "movement_cost": float(transition.movement_cost),
                    "delay": float(transition.total_task_delay_s),
                    "energy": float(transition.total_vehicle_energy_j),
                }
            )
        _append_jsonl(
            path,
            record,
        )

def append_dual_timescale_record(path: Path, record: dict[str, Any]) -> None:
    """追加新主线的 episode/update JSONL 记录。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _integer_histogram(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def build_macro_transition_trace_arrays(
    episode_batches: Iterable[dict[str, Any]], *, gamma: float,
    expected_macro_frame_counts: dict[int, int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """组装并校验正式的两步 SMDP Macro trace。"""

    batches = list(episode_batches)
    if not batches:
        raise ValueError("Macro transition trace requires at least one episode.")
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or gamma_value < 0.0:
        raise ValueError("Macro transition trace gamma must be finite and non-negative.")
    fields = (
        "observation",
        "next_observation",
        "macro_reward",
        "actual_duration",
        "done",
        "old_value",
        "old_next_value",
        "critic_training_target",
        "realized_return_to_go",
        "macro_frame_index",
        "n2_target",
        "n2_bootstrap_old_value",
        "n2_bootstrap_frame_index",
    )
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in fields}
    termination_fields = (
        "termination_observations",
        "termination_elapsed_slots",
        "termination_actions",
        "termination_masks",
        "old_termination_log_probs",
        "old_termination_entropies",
    )
    termination_max_count = max(
        int(np.asarray(batch["termination_masks"]).shape[-1]) for batch in batches
    )
    if termination_max_count > 4:
        raise ValueError("Adaptive holding trace exceeds four sampled decisions per frame.")
    chunks.update({name: [] for name in termination_fields})
    chunks["episode_id"] = []
    chunks["episode_seed"] = []
    episode_transition_counts: dict[str, int] = {}
    seen_episodes: set[int] = set()
    episode_order: list[int] = []
    target_max_abs_error = 0.0
    return_max_abs_error = 0.0

    for batch in batches:
        episode_value = batch.get("episode_id", batch.get("episode"))
        if episode_value is None:
            raise ValueError("Macro transition trace episode_id is missing.")
        episode = int(episode_value)
        episode_seed = int(batch["episode_seed"])
        if episode in seen_episodes:
            raise ValueError(f"Macro transition trace repeats episode {episode}.")
        seen_episodes.add(episode)
        episode_order.append(episode)
        raw = {name: np.asarray(batch[name]) for name in fields}
        observation = np.asarray(raw["observation"], dtype=np.float32)
        next_observation = np.asarray(raw["next_observation"], dtype=np.float32)
        reward = np.asarray(raw["macro_reward"], dtype=np.float32).reshape(-1)
        duration_raw = np.asarray(raw["actual_duration"]).reshape(-1)
        if (
            not np.isfinite(duration_raw).all()
            or np.any(duration_raw != np.floor(duration_raw))
        ):
            raise ValueError(f"Macro transition trace episode {episode} has invalid durations.")
        duration = duration_raw.astype(np.int64)
        done = np.asarray(raw["done"], dtype=bool).reshape(-1)
        old_value = np.asarray(raw["old_value"], dtype=np.float32).reshape(-1)
        old_next_value = np.asarray(raw["old_next_value"], dtype=np.float32).reshape(-1)
        critic_target = np.asarray(raw["critic_training_target"], dtype=np.float32).reshape(-1)
        realized_return = np.asarray(raw["realized_return_to_go"], dtype=np.float32).reshape(-1)
        frame_index = np.asarray(raw["macro_frame_index"], dtype=np.int64).reshape(-1)
        n2_target = np.asarray(raw["n2_target"], dtype=np.float32).reshape(-1)
        n2_bootstrap_old_value = np.asarray(
            raw["n2_bootstrap_old_value"], dtype=np.float32
        ).reshape(-1)
        n2_bootstrap_frame_index = np.asarray(
            raw["n2_bootstrap_frame_index"], dtype=np.int64
        ).reshape(-1)
        count = int(reward.size)
        termination_observations = np.asarray(
            batch["termination_observations"], dtype=np.float32
        )
        termination_elapsed_slots = np.asarray(
            batch["termination_elapsed_slots"], dtype=np.int64
        )
        termination_actions = np.asarray(batch["termination_actions"], dtype=np.int64)
        termination_masks = np.asarray(batch["termination_masks"], dtype=bool)
        old_termination_log_probs = np.asarray(
            batch["old_termination_log_probs"], dtype=np.float32
        )
        old_termination_entropies = np.asarray(
            batch["old_termination_entropies"], dtype=np.float32
        )
        if (
            termination_observations.ndim != 3
            or termination_observations.shape[0] != count
            or termination_observations.shape[-1] != observation.shape[-1]
        ):
            raise ValueError(
                f"Macro transition trace episode {episode} has invalid termination observations."
            )
        decision_shape = (count, int(termination_observations.shape[1]))
        if any(
            value.shape != decision_shape
            for value in (
                termination_elapsed_slots,
                termination_actions,
                termination_masks,
                old_termination_log_probs,
                old_termination_entropies,
            )
        ):
            raise ValueError(
                f"Macro transition trace episode {episode} has misaligned termination fields."
            )
        if count <= 0:
            raise ValueError(f"Macro transition trace episode {episode} is empty.")
        vectors = (
            duration,
            done,
            old_value,
            old_next_value,
            critic_target,
            realized_return,
            frame_index,
            n2_target,
            n2_bootstrap_old_value,
            n2_bootstrap_frame_index,
        )
        if any(value.shape != (count,) for value in vectors):
            raise ValueError(f"Macro transition trace episode {episode} has misaligned fields.")
        if (
            observation.ndim != 2
            or observation.shape[0] != count
            or observation.shape[1] <= 0
            or next_observation.shape != observation.shape
        ):
            raise ValueError(
                f"Macro transition trace episode {episode} has misaligned observations."
            )
        floats = (
            observation,
            next_observation,
            reward,
            old_value,
            old_next_value,
            critic_target,
            realized_return,
            n2_target,
            n2_bootstrap_old_value,
        )
        if any(not np.isfinite(value).all() for value in floats):
            raise ValueError(f"Macro transition trace episode {episode} contains NaN/Inf.")
        if np.any(duration < 1) or np.any(duration > 5):
            raise ValueError(f"Macro transition trace episode {episode} has invalid durations.")
        if not np.array_equal(frame_index, np.arange(count, dtype=np.int64)):
            raise ValueError(f"Macro frame indexes are not continuous in episode {episode}.")
        if not bool(done[-1]) or bool(np.any(done[:-1])):
            raise ValueError(f"Terminal flag must occur only on the final transition of episode {episode}.")
        if float(old_next_value[-1]) != 0.0 or np.any(next_observation[-1] != 0.0):
            raise ValueError(f"Terminal bootstrap state must be zero in episode {episode}.")
        if count > 1 and not np.allclose(
            next_observation[:-1], observation[1:], rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(f"Macro next-observation linkage breaks within episode {episode}.")

        for index in range(count):
            sampled_count = int(np.count_nonzero(termination_masks[index]))
            expected_mask = np.arange(termination_masks.shape[1]) < sampled_count
            elapsed = termination_elapsed_slots[index, :sampled_count]
            actions = termination_actions[index, :sampled_count]
            if (
                not np.array_equal(termination_masks[index], expected_mask)
                or sampled_count > 4
                or not np.array_equal(
                    elapsed, np.arange(1, sampled_count + 1, dtype=np.int64)
                )
                or np.any((actions < 0) | (actions > 1))
                or (sampled_count > 1 and np.any(actions[:-1] == 1))
                or not np.isfinite(
                    old_termination_log_probs[index, :sampled_count]
                ).all()
                or not np.isfinite(
                    old_termination_entropies[index, :sampled_count]
                ).all()
                or not np.isfinite(
                    termination_observations[index, :sampled_count]
                ).all()
            ):
                raise ValueError(
                    f"Termination trace contains invalid sampled decisions in episode {episode}."
                )
            if done[index] and sampled_count and np.any(actions == 1):
                raise ValueError(
                    f"Terminal frame sampled a termination decision in episode {episode}."
                )
            if sampled_count == 0:
                expected_duration = 1
                if not done[index]:
                    raise ValueError(
                        f"Nonterminal one-slot frame lacks an eligible decision in episode {episode}."
                    )
            elif int(actions[-1]) == 1:
                expected_duration = int(elapsed[-1])
            else:
                expected_duration = int(elapsed[-1]) + 1
            if expected_duration != int(duration[index]):
                raise ValueError(
                    f"Termination trace does not explain duration in episode {episode}."
                )

        expected_bootstrap_frame_index = np.full(count, -1, dtype=np.int64)
        expected_bootstrap_old_value = np.zeros(count, dtype=np.float32)
        for index in range(count):
            if done[index] or done[index + 1]:
                continue
            expected_bootstrap_frame_index[index] = frame_index[index + 2]
            expected_bootstrap_old_value[index] = old_value[index + 2]
        if not np.array_equal(
            n2_bootstrap_frame_index, expected_bootstrap_frame_index
        ) or not np.array_equal(n2_bootstrap_old_value, expected_bootstrap_old_value):
            raise ValueError(f"N2 frozen old-value k+2 linkage breaks in episode {episode}.")

        target_expected = build_macro_n2_target(
            reward, old_value, duration, done, gamma_value
        )
        target_expected_for_training = np.asarray(target_expected, dtype=np.float32)
        target_error = max(
            float(
                np.max(
                    np.abs(
                        target_expected_for_training.astype(np.float64)
                        - target.astype(np.float64)
                    )
                )
            )
            for target in (critic_target, n2_target)
        )
        if target_error > 1.0e-6:
            raise ValueError(
                f"two_step_smdp_td trace reconstruction error {target_error} exceeds 1e-6 in episode {episode}."
            )
        terminal_target_error = float(np.max(np.abs(n2_target[done] - reward[done])))
        if terminal_target_error > 1.0e-6:
            raise ValueError(f"Terminal N2 target differs from reward in episode {episode}.")

        return_expected = np.zeros(count, dtype=np.float64)
        running_return = 0.0
        reward64 = reward.astype(np.float64)
        for index in range(count - 1, -1, -1):
            if done[index]:
                running_return = 0.0
            running_return = float(
                reward64[index]
                + gamma_value ** int(duration[index]) * running_return
            )
            return_expected[index] = running_return
        return_error = float(
            np.max(
                np.abs(
                    return_expected.astype(np.float32).astype(np.float64)
                    - realized_return.astype(np.float64)
                )
            )
        )
        if return_error > 1.0e-6:
            raise ValueError(
                f"Realized return-to-go reconstruction error {return_error} exceeds 1e-6 in episode {episode}."
            )
        target_max_abs_error = max(target_max_abs_error, target_error)
        return_max_abs_error = max(return_max_abs_error, return_error)
        episode_transition_counts[f"E{episode}"] = count
        chunks["episode_id"].append(np.full(count, episode, dtype=np.int64))
        chunks["episode_seed"].append(np.full(count, episode_seed, dtype=np.int64))
        for name, value in (
            ("observation", observation),
            ("next_observation", next_observation),
            ("macro_reward", reward),
            ("actual_duration", duration),
            ("done", done),
            ("old_value", old_value),
            ("old_next_value", old_next_value),
            ("critic_training_target", critic_target),
            ("realized_return_to_go", realized_return),
            ("macro_frame_index", frame_index),
            ("n2_target", n2_target),
            ("n2_bootstrap_old_value", n2_bootstrap_old_value),
            ("n2_bootstrap_frame_index", n2_bootstrap_frame_index),
        ):
            chunks[name].append(value)
        term_pad = termination_max_count - termination_masks.shape[1]
        if term_pad:
            termination_observations = np.pad(
                termination_observations, ((0, 0), (0, term_pad), (0, 0))
            )
            termination_elapsed_slots = np.pad(
                termination_elapsed_slots, ((0, 0), (0, term_pad))
            )
            termination_actions = np.pad(
                termination_actions, ((0, 0), (0, term_pad))
            )
            termination_masks = np.pad(termination_masks, ((0, 0), (0, term_pad)))
            old_termination_log_probs = np.pad(
                old_termination_log_probs, ((0, 0), (0, term_pad))
            )
            old_termination_entropies = np.pad(
                old_termination_entropies, ((0, 0), (0, term_pad))
            )
        for name, value in (
            ("termination_observations", termination_observations),
            ("termination_elapsed_slots", termination_elapsed_slots),
            ("termination_actions", termination_actions),
            ("termination_masks", termination_masks),
            ("old_termination_log_probs", old_termination_log_probs),
            ("old_termination_entropies", old_termination_entropies),
        ):
            chunks[name].append(value)

    if episode_order != sorted(episode_order):
        raise ValueError("Macro transition trace episodes must be ordered and unique.")
    if expected_macro_frame_counts is not None:
        expected = {int(key): int(value) for key, value in expected_macro_frame_counts.items()}
        actual = {int(key[1:]): int(value) for key, value in episode_transition_counts.items()}
        if actual != expected:
            raise ValueError(
                f"Macro transition trace frame counts disagree with episode metrics: {actual} != {expected}."
            )

    arrays = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
    arrays["gamma"] = np.asarray(gamma_value, dtype=np.float64)
    duration = arrays["actual_duration"]
    summary = {
        "macro_transition_trace_path": None,
        "transition_count": int(len(arrays["episode_id"])),
        "episode_transition_counts": episode_transition_counts,
        "actual_duration_histogram": _integer_histogram(duration),
        "holding_duration_distribution": {
            str(value): int(np.count_nonzero(duration == value))
            for value in range(1, 6)
        },
        "critic_training_target_mode": "two_step_smdp_td",
        "critic_target_recompute_max_abs_error": target_max_abs_error,
        "realized_return_recompute_max_abs_error": return_max_abs_error,
        "macro_transition_trace_fields": list(arrays),
        "reporting_only_change": True,
    }
    return arrays, summary


def write_macro_transition_trace_artifact(
    run_dir: Path,
    episode_batches: Iterable[dict[str, Any]],
    *,
    gamma: float,
) -> dict[str, Any]:
    """Write a compressed validated Macro trace and add its index to run reports."""

    destination = Path(run_dir)
    episode_records, _ = _read_jsonl(destination / EPISODE_METRICS_FILENAME)
    expected_counts = {
        int(record["episode"]): int(record["macro_frame_count"])
        for record in episode_records
        if record.get("record_type") == "episode"
    }
    arrays, summary = build_macro_transition_trace_arrays(
        episode_batches,
        gamma=gamma,
        expected_macro_frame_counts=expected_counts,
    )
    trace_path = destination / MACRO_TRANSITION_TRACE_FILENAME
    try:
        with trace_path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
    except Exception:
        if trace_path.exists():
            trace_path.unlink()
        raise
    summary["macro_transition_trace_path"] = str(trace_path.resolve())
    summary["macro_transition_trace_format"] = "npz"
    summary["trace_format_version"] = 4
    for filename in (RUN_MANIFEST_FILENAME, RUN_SUMMARY_FILENAME):
        report_path = destination / filename
        payload = _read_json(report_path)
        payload.update(summary)
        _write_json(report_path, payload)
    return summary


def write_dual_timescale_run_manifest(
    path: Path,
    *,
    config_path: str | Path,
    config_raw_sha256: str,
    config_semantic_sha256: str,
    seed: int,
    trainer: Any,
    context: Any,
    target_episodes: int,
    episode_pool: str | Path | None = None,
    episode_pool_provenance: dict[str, Any] | None = None,
    data_input_compatibility: dict[str, Any] | None = None,
    episode_seed_start: int | None = None,
    resume: bool = False,
    training_completion_status: str | None = None,
    controlled_stop_after_policy_updates: int | None = None,
    controlled_stop_completed_policy_updates: int | None = None,
) -> dict[str, Any]:
    """记录当前 PROPOSED 训练、动作、Critic 和数据协议。"""

    if episode_pool_provenance is None:
        episode_pool_provenance = build_episode_pool_provenance(episode_pool)

    manifest = {
        "experiment_label": "PROPOSED",
        "algorithm_id": str(trainer.algorithm_id),
        "task_execution_model": trainer.config_contract["task_execution_model"],
        "observation_schema": trainer.config_contract["observation_schema"],
        "macro_actor_update_control_version": str(
            trainer.macro_actor_update_control_version
        ),
        "method": str(trainer.checkpoint_method),
        "experiment_type": "adaptive_dual_timescale_alternating_actor_updates_v1_mappo",
        "config_path": str(Path(config_path).resolve()),
        "config_raw_sha256": str(config_raw_sha256),
        "config_semantic_sha256": str(config_semantic_sha256),
        "runtime_config_identity": {
            "path": str(Path(config_path).resolve()),
            "raw_sha256": str(config_raw_sha256),
            "semantic_sha256": str(config_semantic_sha256),
        },
        "seed": int(seed),
        "target_episodes": int(target_episodes),
        "realizations_per_policy_update": int(
            trainer.realizations_per_policy_update
        ),
        "policy_update_total": int(trainer.policy_update_total),
        "expected_episode_metrics_count": int(target_episodes),
        "expected_update_metrics_count": int(trainer.policy_update_total),
        "rollout_group_contract": "four independent realizations collected with one frozen old policy; targets are episode-local; advantages are normalized over the combined update group",
        "episode_pool": None if episode_pool is None else str(Path(episode_pool).resolve()),
        "episode_pool_provenance": episode_pool_provenance,
        # Descriptive loader compatibility is recorded separately from the
        # unchanged Checkpoint identity and episode-pool provenance contract.
        "data_input_compatibility": data_input_compatibility,
        "training_pool_manifest_sha256": (
            None
            if episode_pool_provenance is None
            else episode_pool_provenance.get("manifest_sha256")
        ),
        "integrity_baseline_sha256": (
            None
            if episode_pool_provenance is None
            else episode_pool_provenance.get("integrity_baseline_sha256")
        ),
        "generation_config_fingerprint": (
            None
            if episode_pool_provenance is None
            else episode_pool_provenance.get("generation_config_fingerprint")
        ),
        "data_validation_level": (
            None
            if episode_pool_provenance is None
            else episode_pool_provenance.get("data_validation_level")
        ),
        "historical_source_status": (
            None
            if episode_pool_provenance is None
            else episode_pool_provenance.get("historical_source_status")
        ),
        "episode_seed_start": None if episode_seed_start is None else int(episode_seed_start),
        "episode_seed_end": (
            None
            if episode_seed_start is None
            else int(episode_seed_start) + int(target_episodes) - 1
        ),
        "data_protocol": str(context.data_protocol),
        "episode_slots": int(context.episode_slots),
        "origin_start": int(context.train_origin_start),
        "origin_end": int(context.train_origin_end),
        "terminal_origin": int(context.terminal_origin),
        "policy_schema": str(trainer.config_contract["policy_schema"]),
        "architecture_version": int(trainer.config_contract["architecture_version"]),
        "checkpoint_version": int(trainer.checkpoint_version),
        "checkpoint_method": str(trainer.checkpoint_method),
        "macro_action": (
            "shared masked factorized Cache candidates by node"
            if not trainer.unified_cache_action
            else "single masked Cache candidate distribution"
        ),
        "adaptive_holding_enabled": bool(trainer.config["adaptive_holding_enabled"]),
        "adaptive_holding_min_slots": int(trainer.config["adaptive_holding_min_slots"]),
        "adaptive_holding_max_slots": int(trainer.config["adaptive_holding_max_slots"]),
        "macro_actor_objective": "one joint PPO clipped surrogate over Cache candidate and sampled termination sequence",
        "macro_actor_advantage_mode": str(trainer.config_contract["macro_actor_advantage_mode"]),
        "macro_actor_learning_rate_schedule": str(
            trainer.macro_actor_learning_rate_schedule
        ),
        "optimizer_learning_rate_schedule": str(
            trainer.optimizer_learning_rate_schedule
        ),
        "macro_actor_learning_rate_initial": float(
            trainer.learning_rates["macro_actor_initial"]
        ),
        "macro_actor_learning_rate_final": float(
            trainer.learning_rates["macro_actor_final"]
        ),
        "cache_candidate_zero": "current Cache configuration (KEEP)",
        "cache_candidate_action_space": (
            "one shared-parameter masked categorical per cache node over KEEP and that node's legal UPDATE candidates; each UPDATE group uses legal-cardinality correction"
            if not trainer.unified_cache_action
            else "one categorical over KEEP and legal UPDATE candidates with legal-cardinality correction"
        ),
        "micro_action": "configured UAV ids in stable order; each head conditions on all previously selected UAV positions",
        "micro_actor_objective": "single joint autoregressive UAV PPO clipped surrogate + per-head STAY/MOVE group entropy-floor regularization",
        "uav_ids": [str(value) for value in context.action_spec.uav_ids],
        "uav_count": len(context.action_spec.uav_ids),
        "cache_node_count": len(context.action_spec.cache_node_ids),
        "cache_node_roles": [str(value).upper() for value in context.cache_nodes["node_roles"]],
        "micro_group_entropy_floor_enabled": True,
        "micro_group_probability_soft_cap": float(
            trainer.config["micro_group_probability_soft_cap"]
        ),
        "micro_group_entropy_floor": float(
            -float(trainer.config["micro_group_probability_soft_cap"])
            * math.log(float(trainer.config["micro_group_probability_soft_cap"]))
            -(1.0 - float(trainer.config["micro_group_probability_soft_cap"]))
            * math.log(1.0 - float(trainer.config["micro_group_probability_soft_cap"]))
        ),
        "micro_group_entropy_floor_coefficient": float(
            trainer.config["micro_group_entropy_floor_coefficient"]
        ),
        "micro_target_kl": float(trainer.config["micro_target_kl"]),
        "macro_target_kl": float(trainer.config["macro_target_kl"]),
        "macro_critic_target_mode": str(trainer.macro_critic_target_mode),
        "macro_critic_architecture": (
            f"{int(context.observation_spec.observation_dim)}-128-128-1 (SiLU)"
        ),
        "macro_critic_optimizer": "single scalar Critic Adam optimizer",
        "macro_critic_last1_length": 1,
        "macro_critic_last1_semantics": "the final realization from the preceding committed policy-update group",
        "macro_critic_gradient_conflict_method": "last1 norm matching, current-priority last1-only projection, final norm matching to current",
        "long_value_enabled": bool(trainer.long_value_enabled),
        "long_value_target": str(trainer.long_value_target),
        "reward_definition": "task utility minus Cache cost and UAV movement cost",
        "matching": "Transfer-Exchange",
        "kkt": "continuous resource allocation",
        "configuration_control_mode": str(context.configuration_control_mode),
        "observation_dim": int(context.observation_spec.observation_dim),
        "macro_cache_candidate_feature_dim": int(trainer.policy.macro_cache_feature_dim),
        "micro_uav_candidate_feature_dim": int(trainer.policy.micro_uav_feature_dim),
        "prediction_horizon": int(context.observation_spec.prediction_horizon),
        "actor_learning_rate": float(trainer.learning_rates["actor"]),
        "macro_actor_learning_rate": float(trainer.learning_rates["macro_actor"]),
        "micro_actor_learning_rate": float(trainer.learning_rates["micro_actor"]),
        "macro_critic_learning_rate": float(trainer.learning_rates["macro_critic"]),
        "micro_critic_learning_rate": float(trainer.learning_rates["micro_critic"]),
        "long_value_learning_rate": float(trainer.learning_rates["long_value"]),
        "macro_critic_value_loss_coefficient": float(trainer.config["value_loss_coefficient"]),
        "macro_critic_steps_per_update": int(trainer.config.get("macro_critic_steps_per_update", 50)),
        "unified_cache_action": bool(trainer.unified_cache_action),
        "cache_action_mode": str(context.cache_action_mode),
        "cache_reconfiguration_model": str(context.cache_reconfiguration_model),
        "cache_node_ids": [str(value) for value in context.action_spec.cache_node_ids],
        "cache_joint_logprob": trainer.config_contract["cache_joint_logprob"],
        "cache_entropy_aggregation": (
            "sum of per-node categorical entropies for the factorized joint cache policy"
            if not trainer.unified_cache_action
            else "single unified categorical entropy"
        ),
        "cardinality_corrected_cache_action": bool(trainer.cardinality_corrected_cache_action),
        "macro_full_batch": True,
        "macro_actor_update_epochs": int(trainer.config["update_epochs"]),
        "micro_macro_buffers_separate": True,
        "prediction_model_frozen": True,
        "training_counters": dict(trainer.training_counters),
        "resume": bool(resume),
    }
    manifest.update(
        {
            "actor_update_mode": trainer.actor_update_mode,
            "actor_update_schedule_version": trainer.actor_update_schedule_version,
            "actor_update_first": trainer.actor_update_first,
            "actor_update_period": trainer.actor_update_period,
            "actor_update_schedule_state": dict(
                trainer.actor_update_schedule_state
            ),
        }
    )
    if training_completion_status is not None:
        manifest["training_completion_status"] = str(training_completion_status)
    if controlled_stop_after_policy_updates is not None:
        manifest["controlled_stop_after_policy_updates"] = int(
            controlled_stop_after_policy_updates
        )
    if controlled_stop_completed_policy_updates is not None:
        manifest["controlled_stop_completed_policy_updates"] = int(
            controlled_stop_completed_policy_updates
        )
    _write_json(Path(path), manifest)
    return manifest


def append_dual_timescale_slot_metrics(
    path: Path,
    *,
    episode: int,
    transitions: Iterable[Any],
) -> int:
    """记录当前主线的实际系统 slot 结果。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        for transition in transitions:
            matching_summary = transition.matching_summary
            expected_reward = (
                float(transition.actual_task_utility)
                - float(transition.cache_cost)
                - float(transition.movement_cost)
            )
            if not math.isclose(
                float(transition.reward), expected_reward, rel_tol=1e-12, abs_tol=1e-9
            ):
                raise ValueError("slot utility 与任务效用、Cache cost、移动 cost 不一致。")
            record = {
                "record_type": "slot",
                "episode": int(episode),
                "origin": int(transition.origin_index),
                "slot": int(transition.slot),
                "node_task_count": dict(
                    matching_summary.get("node_task_count", {})
                ),
                "cloud_task_count": int(matching_summary["cloud_task_count"]),
                "cloud_uplink_allocated_bps": matching_summary[
                    "cloud_uplink_allocated_bps"
                ],
                "cloud_uplink_capacity_bps": matching_summary[
                    "cloud_uplink_capacity_bps"
                ],
                "cloud_uplink_capacity_violation_bps": matching_summary[
                    "cloud_uplink_capacity_violation_bps"
                ],
                "cloud_cpu_allocated_cycles_per_s": matching_summary[
                    "cloud_cpu_allocated_cycles_per_s"
                ],
                "cloud_cpu_capacity_cycles_per_s": matching_summary[
                    "cloud_cpu_capacity_cycles_per_s"
                ],
                "cloud_cpu_capacity_violation_cycles_per_s": matching_summary[
                    "cloud_cpu_capacity_violation_cycles_per_s"
                ],
                "cloud_selected_resource_values_finite": bool(
                    matching_summary["cloud_selected_resource_values_finite"]
                ),
                "cloud_kkt_solve_count": int(
                    matching_summary["cloud_kkt_solve_count"]
                ),
                "cloud_kkt_max_uplink_allocated_bps": matching_summary[
                    "cloud_kkt_max_uplink_allocated_bps"
                ],
                "cloud_kkt_max_cpu_allocated_cycles_per_s": matching_summary[
                    "cloud_kkt_max_cpu_allocated_cycles_per_s"
                ],
                "cloud_kkt_max_uplink_capacity_violation_bps": matching_summary[
                    "cloud_kkt_max_uplink_capacity_violation_bps"
                ],
                "cloud_kkt_max_cpu_capacity_violation_cycles_per_s": (
                    matching_summary[
                        "cloud_kkt_max_cpu_capacity_violation_cycles_per_s"
                    ]
                ),
                "cloud_kkt_nonfinite_result_count": int(
                    matching_summary["cloud_kkt_nonfinite_result_count"]
                ),
                "G": float(transition.reward),
                "task_utility": float(transition.actual_task_utility),
                "delay": float(transition.total_task_delay_s),
                "energy": float(transition.total_vehicle_energy_j),
                "cache_download_delay_s": float(
                    transition.cache_download_delay_s
                ),
                "cache_cost": float(transition.cache_cost),
                "movement_cost": float(transition.movement_cost),
                "actual_cache_update_count": int(transition.actual_cache_update_count),
                "uav_action_indices": [int(value) for value in transition.actions],
                "uav_ids": [str(value) for value in transition.uav_ids],
                "uav_stay_by_id": {
                    str(uav_id): bool(int(transition.actions[index]) == 0)
                    for index, uav_id in enumerate(transition.uav_ids)
                },
                "uav0_stay": bool(int(transition.actions[0]) == 0),
                "uav1_stay": (
                    bool(int(transition.actions[1]) == 0)
                    if len(transition.actions) > 1
                    else None
                ),
                "uav0_move": bool(int(transition.actions[0]) != 0),
                "uav1_move": bool(int(transition.actions[1]) != 0),
                "matching_calls": 1,
                "kkt_calls": int(
                    matching_summary.get("KKT_SOLVER_CALL_COUNT", 0)
                ),
                "slot_runtime_seconds": float(transition.slot_runtime_seconds),
            }
            json.dump(record, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")
            count += 1
    return count
