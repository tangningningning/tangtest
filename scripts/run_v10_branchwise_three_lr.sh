#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src"

if [[ ! -f "${REPO_ROOT}/configs/system_config.toml" || ! -f "${REPO_ROOT}/src/ppo_simple/trainer.py" ]]; then
    echo "Not an adaptive_cache_mec repository root: ${REPO_ROOT}" >&2
    exit 1
fi

if ! command -v python >/dev/null 2>&1; then
    echo "python was not found in the server container" >&2
    exit 1
fi

echo "Python runtime check:"
python -c 'import sys; import torch; import ppo_simple; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(ppo_simple.__file__)'

EPISODE_POOL="runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911"
if [[ ! -d "${EPISODE_POOL}" ]]; then
    echo "Episode pool is missing: ${EPISODE_POOL}" >&2
    exit 1
fi

EPISODE_COUNT="$(find "${EPISODE_POOL}" -mindepth 1 -maxdepth 1 -type d -name 'episode_*' | wc -l | tr -d '[:space:]')"
if [[ "${EPISODE_COUNT}" != "300" ]]; then
    echo "Expected 300 episode directories, found ${EPISODE_COUNT}" >&2
    exit 1
fi
for seed in $(seq 20001 20300); do
    if [[ ! -d "${EPISODE_POOL}/episode_${seed}" ]]; then
        echo "Missing episode realization: episode_${seed}" >&2
        exit 1
    fi
done

assert_run_dir_available() {
    local run_dir="$1"
    if [[ -e "${run_dir}" && -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing to overwrite non-empty run directory: ${run_dir}" >&2
        exit 1
    fi
}

CONFIG_5E4="configs/proposed/system_config_ppo_v10_branchwise_cache_uav.toml"
CONFIG_6E4="configs/lr_sweep/v10_branchwise_actor_lr_6e4_kl.toml"
CONFIG_1E3="configs/lr_sweep/v10_branchwise_actor_lr_1e3_kl.toml"
RUN_5E4="runs/ppo_v10_branchwise_cache_uav_5e4_kl_formal"
RUN_6E4="runs/ppo_v10_branchwise_cache_uav_6e4_kl_formal"
RUN_1E3="runs/ppo_v10_branchwise_cache_uav_1e3_kl_formal"

for config in "${CONFIG_5E4}" "${CONFIG_6E4}" "${CONFIG_1E3}"; do
    if [[ ! -f "${config}" ]]; then
        echo "Configuration is missing: ${config}" >&2
        exit 1
    fi
done
for run_dir in "${RUN_5E4}" "${RUN_6E4}" "${RUN_1E3}"; do
    assert_run_dir_available "${run_dir}"
done

run_formal_training() {
    local label="$1"
    local config="$2"
    local run_dir="$3"
    assert_run_dir_available "${run_dir}"
    echo "Starting V10-B actor LR ${label}"
    if python -B -m ppo_simple.trainer \
        --config "${config}" \
        --epochs 300 \
        --run-dir "${run_dir}" \
        --episode-pool "${EPISODE_POOL}" \
        --episode-seed-start 20001; then
        :
    else
        status=$?
        echo "V10-B training failed for actor LR ${label}; remaining runs were not started (exit ${status})." >&2
        exit "${status}"
    fi
    echo "Completed V10-B actor LR ${label}"
}

echo "V10-B serial training preflight passed."
echo "Episode pool: ${EPISODE_POOL} (episode_20001..episode_20300)"
run_formal_training "5e-4" "${CONFIG_5E4}" "${RUN_5E4}"
run_formal_training "6e-4" "${CONFIG_6E4}" "${RUN_6E4}"
run_formal_training "1e-3" "${CONFIG_1E3}" "${RUN_1E3}"
echo "All three V10-B serial training runs completed successfully."
