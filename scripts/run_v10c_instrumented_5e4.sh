#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
export PYTHONPATH="${REPO_ROOT}/src"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${REPO_ROOT}/configs/proposed/system_config_ppo_v10_branchwise_cache_uav.toml"
EPISODE_POOL="${REPO_ROOT}/runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911"
RUN_DIR="${REPO_ROOT}/runs/ppo_v10c_instrumented_5e4_ep240"
EXPECTED_ACTOR_LR="0.0005"
EPISODE_SEED_START=20001
TARGET_EPOCHS=240

fail() {
    echo "[V10-C 5e-4] ERROR: $*" >&2
    exit 1
}

[[ -d "${REPO_ROOT}" ]] || fail "repository root does not exist: ${REPO_ROOT}"
[[ -f "${CONFIG_PATH}" ]] || fail "config does not exist: ${CONFIG_PATH}"
[[ -d "${EPISODE_POOL}" ]] || fail "episode pool does not exist: ${EPISODE_POOL}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python executable not found: ${PYTHON_BIN}"

if [[ -e "${RUN_DIR}" ]]; then
    [[ -d "${RUN_DIR}" ]] || fail "run path exists but is not a directory: ${RUN_DIR}"
    [[ -z "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || \
        fail "run directory is not empty; refusing to overwrite: ${RUN_DIR}"
fi

cd "${REPO_ROOT}"

"${PYTHON_BIN}" - "${CONFIG_PATH}" "${EPISODE_POOL}" "${RUN_DIR}" "${EXPECTED_ACTOR_LR}" <<'PY'
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import tomllib


config_path = Path(sys.argv[1])
pool_path = Path(sys.argv[2])
run_path = Path(sys.argv[3])
expected_actor_lr = float(sys.argv[4])

if importlib.util.find_spec("torch") is None:
    raise SystemExit("PyTorch is not importable in the selected environment")
if importlib.util.find_spec("ppo_simple") is None:
    raise SystemExit("ppo_simple is not importable from the repository")
import torch
import ppo_simple  # noqa: F401

with config_path.open("rb") as stream:
    config = tomllib.load(stream)
ppo = config["algorithm"]["ppo"]
data_protocol = ppo["data_protocol"]


def require_close(field: str, expected: float) -> None:
    actual = float(ppo[field])
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
        raise SystemExit(f"{field}={actual} != expected {expected}")


require_close("actor_learning_rate", expected_actor_lr)
require_close("critic_learning_rate", 0.0003)
require_close("target_kl", 0.10)
require_close("clip_epsilon", 0.2)
require_close("max_grad_norm", 0.5)
require_close("entropy_coefficient", 0.0)
require_close("gamma", 1.0)
require_close("gae_lambda", 0.95)
if int(ppo["update_epochs"]) != 4:
    raise SystemExit("update_epochs must remain 4")
if int(ppo["num_minibatches"]) != 10:
    raise SystemExit("num_minibatches must remain 10")
if int(ppo["minibatch_size"]) != 30:
    raise SystemExit("minibatch_size must remain 30")
if int(config["experiment"]["seed"]) != 4401:
    raise SystemExit("experiment.seed must remain 4401")
if ppo["critic_value_scale_mode"] != "popart_normalized":
    raise SystemExit("critic_value_scale_mode must remain popart_normalized")
if ppo["configuration_control_mode"] != "policy_primitive_v9_joint_cache_uav_operation":
    raise SystemExit("unexpected configuration_control_mode")
if ppo["policy_schema"] != "adaptive_joint_cache_replica_aware_uav_operation_v10_branchwise":
    raise SystemExit("unexpected policy_schema")
if data_protocol["protocol"] != "full_episode_randomized":
    raise SystemExit("unexpected PPO data protocol")
if any(
    int(data_protocol[field]) != expected
    for field, expected in (
        ("episode_slots", 300),
        ("train_origin_start", 0),
        ("train_origin_end", 299),
        ("rollout_steps", 300),
        ("terminal_origin", 299),
    )
):
    raise SystemExit("PPO data protocol is not the fixed 300-slot contract")

episode_dirs = []
for path in pool_path.iterdir():
    if not path.is_dir() or not path.name.startswith("episode_"):
        continue
    try:
        episode_id = int(path.name.split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise SystemExit(f"invalid episode directory name: {path.name}") from exc
    episode_dirs.append((episode_id, path))
episode_dirs.sort(key=lambda item: item[0])
expected_ids = list(range(20001, 20301))
actual_ids = [episode_id for episode_id, _ in episode_dirs]
if actual_ids != expected_ids:
    raise SystemExit(
        f"episode pool must contain exactly episode_20001..episode_20300; got {actual_ids[:3]}..{actual_ids[-3:]}"
    )
for endpoint_name in ("episode_20001", "episode_20240"):
    if not (pool_path / endpoint_name).is_dir():
        raise SystemExit(f"required used episode is missing: {endpoint_name}")
required_artifacts = {
    "trajectory.csv",
    "vehicle_roles.json",
    "tasks.csv",
    "popularity.csv",
    "popularity_history.csv",
    "vehicle_predictions.csv",
    "popularity_predictions.csv",
    "prediction_scenarios.npz",
    "tv_local_cache.json",
    "sumo_demand.rou.xml",
    "sumo_demand.meta.json",
}
for episode_id, episode_path in episode_dirs:
    missing = sorted(
        name for name in required_artifacts if not (episode_path / name).is_file()
    )
    if missing:
        raise SystemExit(f"episode_{episode_id} is missing artifacts: {missing}")

trainer_source = (
    Path.cwd() / "src" / "ppo_simple" / "trainer.py"
).read_text(encoding="utf-8")
required_instrumentation = (
    "uav_move_adv_positive_fraction",
    "uav_stay_adv_positive_fraction",
    "uav_move_reward_mean",
    "uav_stay_reward_mean",
    "uav_move_q_total_mean",
    "uav_stay_q_total_mean",
    "uav_move_movement_cost_mean",
    "uav_stay_movement_cost_mean",
    "uav_move_ratio_mean",
    "uav_stay_ratio_mean",
    "actual_PPO_advantage",
    "selected_uav_action",
    "selected_cache_action",
)
missing_instrumentation = [
    field for field in required_instrumentation if field not in trainer_source
]
if missing_instrumentation:
    raise SystemExit(
        f"instrumentation fields missing from trainer.py: {missing_instrumentation}"
    )

print(f"preflight: Python={sys.executable}")
print(f"preflight: PyTorch={torch.__version__}")
print(f"preflight: ppo_simple={Path(ppo_simple.__file__).resolve()}")
print(f"preflight: config={config_path}")
print(f"preflight: episode_pool={pool_path} count={len(episode_dirs)}")
print(f"preflight: run_dir={'absent' if not run_path.exists() else 'empty'}")
print("preflight: instrumentation fields=present")
PY

echo "[V10-C 5e-4] Preflight PASS"
echo "[V10-C 5e-4] Launching ${TARGET_EPOCHS} epochs with frozen pool; without resume and without episode generation"

exec "${PYTHON_BIN}" -m ppo_simple.trainer \
    --config "${CONFIG_PATH}" \
    --epochs "${TARGET_EPOCHS}" \
    --run-dir "${RUN_DIR}" \
    --episode-pool "${EPISODE_POOL}" \
    --episode-seed-start "${EPISODE_SEED_START}"
