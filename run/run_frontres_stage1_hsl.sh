#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run/run_frontres_stage1_hsl.sh MOTION_PATH [NUM_ENVS] [MAX_ITERS]"
  echo
  echo "Stage 1 trains the Clean-oriented FrontRES Delta SE proposal."
  echo "Example:"
  echo "  bash run/run_frontres_stage1_hsl.sh /path/to/motions 12000 800"
  exit 1
fi

MOTION_PATH="$1"
NUM_ENVS="${2:-12000}"
MAX_ITERS="${3:-800}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-FEMR}"
RUN_NAME="${RUN_NAME:-FrontRES_STAGE1_HSL}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" scripts/rsl_rl/train.py --distributed)
else
  LAUNCH=(python scripts/rsl_rl/train.py)
fi

HYDRA_FULL_ERROR=1 "${LAUNCH[@]}" \
  --task=FrontRES-Unified-Tracking-Flat-G1-v0 \
  --num_envs="${NUM_ENVS}" \
  --motion "${MOTION_PATH}" \
  --headless \
  --logger wandb \
  --log_project_name "${LOG_PROJECT_NAME}" \
  --experiment_name g1_flat_frontres_stage1_hsl \
  --run_name "${RUN_NAME}" \
  --max_iterations "${MAX_ITERS}" \
  --frontres_stage stage1_hsl
