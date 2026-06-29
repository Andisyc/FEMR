#!/usr/bin/env bash
set -euo pipefail

MOTION_PATH="${1:-/hdd1/cyx/AMASS_G1NPZ_Final}"
NUM_ENVS="${2:-12000}"
SEGMENT_K="${3:-4}"
CACHE_DIR="${4:-/hdd1/cyx/AMASS_G1Segment}"
LOG_PATH="${LOG_PATH:-/hdd1/cyx/FEMR/train_stage1_segment_cache.txt}"

cd "$(dirname "$0")"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export FEMR_LOG_ROOT="${FEMR_LOG_ROOT:-/hdd1/cyx/FEMR}"
export WANDB_DIR="${WANDB_DIR:-/hdd1/cyx/FEMR}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/hdd1/cyx/FEMR/.wandb_cache}"
export MAX_MOTIONS="${MAX_MOTIONS:-all}"
export MAX_SEGMENTS="${MAX_SEGMENTS:-all}"

bash run/run_frontres_stage1_segment_cache.sh \
  "${MOTION_PATH}" \
  "${NUM_ENVS}" \
  "${SEGMENT_K}" \
  "${CACHE_DIR}" \
  >"${LOG_PATH}" 2>&1
