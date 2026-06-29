#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_stage3.sh STAGE2_CHECKPOINT [MOTION_PATH] [NUM_ENVS] [MAX_ITERS] [UPDATE_STEPS] [MODE]"
  echo
  echo "MODE can be: train, sentinel, probe, storage, single_update, update_loop."
  exit 1
fi

STAGE2_CHECKPOINT="$1"
MOTION_PATH="${2:-/hdd1/cyx/AMASS_G1NPZ_Final}"
NUM_ENVS="${3:-12000}"
MAX_ITERS="${4:-2000}"
UPDATE_STEPS="${5:-4}"
MODE="${6:-train}"
CACHE_DIR="${CACHE_DIR:-/hdd1/cyx/AMASS_G1Segment}"
LOG_PATH="${LOG_PATH:-/hdd1/cyx/FEMR/train_stage3_segment_hrl.txt}"

cd "$(dirname "$0")"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export FEMR_LOG_ROOT="${FEMR_LOG_ROOT:-/hdd1/cyx/FEMR}"
export WANDB_DIR="${WANDB_DIR:-/hdd1/cyx/FEMR}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/hdd1/cyx/FEMR/.wandb_cache}"

bash run/run_frontres_stage3_segment_hrl.sh \
  "${STAGE2_CHECKPOINT}" \
  "${MOTION_PATH}" \
  "${NUM_ENVS}" \
  "${MAX_ITERS}" \
  "${UPDATE_STEPS}" \
  "${MODE}" \
  >"${LOG_PATH}" 2>&1
