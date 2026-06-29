#!/usr/bin/env bash
set -euo pipefail

MOTION_PATH="${1:-/hdd1/cyx/AMASS_G1NPZ_Final}"
NUM_ENVS="${2:-12000}"
MAX_ITERS="${3:-200}"
SUPERVISED_WARMUP_ITERS="${SUPERVISED_WARMUP_ITERS:-${MAX_ITERS}}"
LOG_PATH="${LOG_PATH:-/hdd1/cyx/FEMR/train_stage2_hsl_warmup.txt}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-FEMR}"
RUN_NAME="${RUN_NAME:-FEMR_STAGE2_HSL_WARMUP}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

cd "$(dirname "$0")"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export FEMR_LOG_ROOT="${FEMR_LOG_ROOT:-/hdd1/cyx/FEMR}"
export WANDB_DIR="${WANDB_DIR:-/hdd1/cyx/FEMR}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/hdd1/cyx/FEMR/.wandb_cache}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" scripts/rsl_rl/train.py --distributed)
else
  LAUNCH=(python scripts/rsl_rl/train.py)
fi

"${LAUNCH[@]}" \
  --task=FrontRES-Unified-Tracking-Flat-G1-v0 \
  --num_envs="${NUM_ENVS}" \
  --motion "${MOTION_PATH}" \
  --headless \
  --logger tensorboard \
  --log_project_name "${LOG_PROJECT_NAME}" \
  --experiment_name g1_flat_frontres_stage2_hsl \
  --run_name "${RUN_NAME}" \
  --max_iterations "${MAX_ITERS}" \
  --supervised_warmup_iterations "${SUPERVISED_WARMUP_ITERS}" \
  --frontres_stage stage2_hsl_warmup \
  >"${LOG_PATH}" 2>&1
