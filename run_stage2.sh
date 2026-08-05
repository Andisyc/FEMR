#!/usr/bin/env bash
set -euo pipefail

# FrontRES Stage 2 HSL warmup launcher.
#
# Status: active.
# Upstream: Stage 1 motion dataset and user shell command.
# Downstream: warmup checkpoint consumed by Stage 3.
# Evidence: script-level route, not a live validation.
#
# B1: Runtime owner. Select GPU and log sink.

FEMR_ROOT="${FEMR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
FEMR_DATA_ROOT="${FEMR_DATA_ROOT:-$(dirname "${FEMR_ROOT}")}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
LOG_PATH="${LOG_PATH:-${FEMR_ROOT}/train_stage2_hsl_warmup.txt}"

# B2: Training contract. Positional args are shortcuts for these values.

MOTION_PATH="${1:-${FEMR_DATA_ROOT}/AMASS_G1NPZ_Final}"
NUM_ENVS="${2:-12000}"
MAX_ITERS="${3:-200}"

# Stage 2 default is pure supervised warmup.
SUPERVISED_WARMUP_ITERS="${SUPERVISED_WARMUP_ITERS:-${MAX_ITERS}}"

# B3: Logging and distributed launch.

LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-FEMR}"
RUN_NAME="${RUN_NAME:-FEMR_STAGE2_HSL_WARMUP}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

cd "${FEMR_ROOT}"

export CUDA_VISIBLE_DEVICES
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export FEMR_ROOT
export FEMR_DATA_ROOT
export FEMR_LOG_ROOT="${FEMR_LOG_ROOT:-${FEMR_ROOT}}"
export WANDB_DIR="${WANDB_DIR:-${FEMR_LOG_ROOT}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${FEMR_LOG_ROOT}/.wandb_cache}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" scripts/rsl_rl/train.py --distributed)
else
  LAUNCH=(python scripts/rsl_rl/train.py)
fi

mkdir -p "$(dirname "${LOG_PATH}")"

CMD=(
  "${LAUNCH[@]}"
  --task=FrontRES-Unified-Tracking-Flat-G1-v0
  --num_envs="${NUM_ENVS}"
  --motion "${MOTION_PATH}"
  --headless
  --logger tensorboard
  --log_project_name "${LOG_PROJECT_NAME}"
  --experiment_name g1_flat_frontres_stage2_hsl
  --run_name "${RUN_NAME}"
  --max_iterations "${MAX_ITERS}"
  --supervised_warmup_iterations "${SUPERVISED_WARMUP_ITERS}"
  --frontres_stage stage2_hsl_warmup
)

nohup "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
PID="$!"

echo "[FrontRES Stage2] submitted pid=${PID}"
echo "[FrontRES Stage2] motion=${MOTION_PATH}"
echo "[FrontRES Stage2] num_envs=${NUM_ENVS}"
echo "[FrontRES Stage2] max_iters=${MAX_ITERS}"
echo "[FrontRES Stage2] supervised_warmup_iters=${SUPERVISED_WARMUP_ITERS}"
echo "[FrontRES Stage2] log=${LOG_PATH}"
echo "[FrontRES Stage2] follow: tail -f ${LOG_PATH}"
