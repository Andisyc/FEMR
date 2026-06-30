#!/usr/bin/env bash
set -euo pipefail

MOTION_PATH="${1:-/hdd1/cyx/AMASS_G1NPZ_Final}"
NUM_ENVS="${2:-1}"
SEGMENT_K="${3:-4}"
CACHE_DIR="${4:-/hdd1/cyx/AMASS_G1Segment}"
STAGE1_FULL="${STAGE1_FULL:-0}"
STAGE1_MODE="tiny"
if [[ "${STAGE1_FULL}" == "1" ]]; then
  STAGE1_MODE="full"
fi
LOG_PATH="${LOG_PATH:-/hdd1/cyx/FEMR/train_stage1_segment_cache_${STAGE1_MODE}.txt}"
RUN_FOREGROUND="${RUN_FOREGROUND:-0}"

cd "$(dirname "$0")"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export FEMR_LOG_ROOT="${FEMR_LOG_ROOT:-/hdd1/cyx/FEMR}"
export WANDB_DIR="${WANDB_DIR:-/hdd1/cyx/FEMR}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/hdd1/cyx/FEMR/.wandb_cache}"
export DEVICE="${DEVICE:-cuda:0}"
if [[ "${STAGE1_FULL}" == "1" ]]; then
  export MAX_MOTIONS="${MAX_MOTIONS:-all}"
  export MAX_SEGMENTS="${MAX_SEGMENTS:-all}"
  export FRAME_STRIDE="${FRAME_STRIDE:-1}"
  export CACHE_CHUNK_SIZE="${CACHE_CHUNK_SIZE:-128}"
  export PERTURBATION_MODE="${PERTURBATION_MODE:-hrl_curriculum_bank}"
  export CURRICULUM_BANK_SIZE="${CURRICULUM_BANK_SIZE:-16}"
else
  export MAX_MOTIONS="${MAX_MOTIONS:-1}"
  export MAX_SEGMENTS="${MAX_SEGMENTS:-all}"
  export FRAME_STRIDE="${FRAME_STRIDE:-100000}"
  export CACHE_CHUNK_SIZE="${CACHE_CHUNK_SIZE:-1}"
  export PERTURBATION_MODE="${PERTURBATION_MODE:-discrete_bank}"
  export PERTURBATION_STRENGTHS="${PERTURBATION_STRENGTHS:-0.0}"
  export CURRICULUM_BANK_SIZE="${CURRICULUM_BANK_SIZE:-1}"
  export VALIDATION_REQUIRE_BOUNDARY_DIAGNOSTIC="${VALIDATION_REQUIRE_BOUNDARY_DIAGNOSTIC:-0}"
fi
export VARIANTS_PER_STRENGTH="${VARIANTS_PER_STRENGTH:-1}"
export VALIDATION_MIN_SEGMENTS="${VALIDATION_MIN_SEGMENTS:-1}"
export VALIDATION_MIN_NOISY="${VALIDATION_MIN_NOISY:-1}"

mkdir -p "$(dirname "${LOG_PATH}")"

CMD=(
  bash run/run_frontres_stage1_segment_cache.sh
  "${MOTION_PATH}"
  "${NUM_ENVS}"
  "${SEGMENT_K}"
  "${CACHE_DIR}"
)

if [[ "${RUN_FOREGROUND}" == "1" ]]; then
  echo "[FrontRES Stage1] running in foreground; log=${LOG_PATH}"
  echo "[FrontRES Stage1] mode=${STAGE1_MODE}"
  echo "[FrontRES Stage1] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset} device=${DEVICE}"
  echo "[FrontRES Stage1] motion=${MOTION_PATH} num_envs=${NUM_ENVS} segment_k=${SEGMENT_K} cache_dir=${CACHE_DIR}"
  echo "[FrontRES Stage1] max_motions=${MAX_MOTIONS} max_segments=${MAX_SEGMENTS} frame_stride=${FRAME_STRIDE} cache_chunk_size=${CACHE_CHUNK_SIZE} perturbation_mode=${PERTURBATION_MODE} variants_per_strength=${VARIANTS_PER_STRENGTH}"
  "${CMD[@]}" >"${LOG_PATH}" 2>&1
else
  nohup "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
  PID="$!"
  echo "[FrontRES Stage1] submitted pid=${PID}"
  echo "[FrontRES Stage1] log=${LOG_PATH}"
  echo "[FrontRES Stage1] mode=${STAGE1_MODE}"
  echo "[FrontRES Stage1] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset} device=${DEVICE}"
  echo "[FrontRES Stage1] motion=${MOTION_PATH} num_envs=${NUM_ENVS} segment_k=${SEGMENT_K} cache_dir=${CACHE_DIR}"
  echo "[FrontRES Stage1] max_motions=${MAX_MOTIONS} max_segments=${MAX_SEGMENTS} frame_stride=${FRAME_STRIDE} cache_chunk_size=${CACHE_CHUNK_SIZE} perturbation_mode=${PERTURBATION_MODE} variants_per_strength=${VARIANTS_PER_STRENGTH}"
  echo "[FrontRES Stage1] follow: tail -f ${LOG_PATH}"
fi
