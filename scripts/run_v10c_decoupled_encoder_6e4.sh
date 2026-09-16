#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
CONFIG_PATH="${REPO_ROOT}/configs/proposed/system_config_ppo_v10c_decoupled_encoder_6e4_formal.toml"
REFERENCE_CONFIG_PATH="${REPO_ROOT}/configs/proposed/system_config_ppo_v10_branchwise_cache_uav.toml"
EPISODE_POOL="${REPO_ROOT}/runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911"
RUN_DIR="${REPO_ROOT}/runs/ppo_v10c_decoupled_encoder_6e4_formal"
CONDA_ENV_NAME="adaptive_cache_mec"
EXPECTED_ACTOR_LR="0.0006"
EXPECTED_TAG="v10c_decoupled_encoder_6e4_formal"
EXPECTED_POLICY_SCHEMA="adaptive_joint_cache_replica_aware_uav_operation_v10c_decoupled_encoder"
TARGET_EPOCHS=300

fail() {
    echo "[V10-C 6e-4] ERROR: $*" >&2
    exit 1
}

if [[ -e "${RUN_DIR}" ]]; then
    [[ -d "${RUN_DIR}" ]] || fail "run path exists but is not a directory: ${RUN_DIR}"
    [[ -z "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || fail "run directory is not empty; refusing overwrite or resume: ${RUN_DIR}"
fi

[[ -d "${REPO_ROOT}" ]] || fail "repository root does not exist: ${REPO_ROOT}"
[[ -f "${CONFIG_PATH}" ]] || fail "V10-C config does not exist: ${CONFIG_PATH}"
[[ -f "${REFERENCE_CONFIG_PATH}" ]] || fail "V10-B reference config does not exist: ${REFERENCE_CONFIG_PATH}"
[[ -d "${EPISODE_POOL}" ]] || fail "episode pool does not exist: ${EPISODE_POOL}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
    command -v conda >/dev/null 2>&1 || fail "activate Conda environment ${CONDA_ENV_NAME} before launching"
    CONDA_BASE="$(conda info --base)" || fail "could not locate the Conda installation"
    [[ -r "${CONDA_BASE}/etc/profile.d/conda.sh" ]] || fail "Conda shell hook is unavailable under ${CONDA_BASE}"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}" || fail "could not activate Conda environment ${CONDA_ENV_NAME}"
fi

command -v python >/dev/null 2>&1 || fail "Python executable not found in ${CONDA_ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

python "${SCRIPT_DIR}/validate_v10c_formal_preflight.py" --config "${CONFIG_PATH}" --reference-config "${REFERENCE_CONFIG_PATH}" --episode-pool "${EPISODE_POOL}" --run-dir "${RUN_DIR}" --expected-actor-lr "${EXPECTED_ACTOR_LR}" --expected-tag "${EXPECTED_TAG}"

echo "[V10-C 6e-4] Preflight PASS"
echo "[V10-C 6e-4] Launching ${TARGET_EPOCHS} epochs; existing pool only, no resume or episode generation"

exec python -m ppo_simple.trainer --config "${CONFIG_PATH}" --epochs "${TARGET_EPOCHS}" --run-dir "${RUN_DIR}" --episode-pool "${EPISODE_POOL}"
