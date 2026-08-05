#!/usr/bin/env bash
set -euo pipefail

# FrontRES Stage 1 segment index/cache launcher.
#
# Status: active.
# Upstream: user shell command.
# Downstream: Stage 1 segment index/cache consumed by Stage 3 Segment Replay.
# Evidence: script-level route, not a live validation.
#
# B1: Runtime owner. Select GPU, python env, and log sink.

FEMR_ROOT="${FEMR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
FEMR_DATA_ROOT="${FEMR_DATA_ROOT:-$(dirname "${FEMR_ROOT}")}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
LOG_PATH="${LOG_PATH:-${FEMR_ROOT}/train_stage1_segment_index_full.txt}"

# B2: Dataset and artifact contract. Positional args are shortcuts for these.

MOTION_PATH="${1:-${FEMR_DATA_ROOT}/AMASS_G1NPZ_Final}"
NUM_ENVS="${2:-1}"
SEGMENT_K="${3:-4}"
CACHE_DIR="${4:-${FEMR_DATA_ROOT}/AMASS_G1Segment}"

# B3: Stage 1 mode. Index mode is the normal fast precompute path.

STAGE1_FULL="${STAGE1_FULL:-1}"
STAGE1_BUILD_ROLLOUT_CACHE="${STAGE1_BUILD_ROLLOUT_CACHE:-0}"

# B4: Device string after CUDA_VISIBLE_DEVICES masking.

DEVICE="${DEVICE:-cuda:0}"

if [[ "${STAGE1_FULL}" == "1" ]]; then
  STAGE1_MODE="full"

  # Full mode: use dense segment starts over the selected motion set.
  MAX_MOTIONS="${MAX_MOTIONS:-all}"
  MAX_SEGMENTS="${MAX_SEGMENTS:-all}"
  FRAME_STRIDE="${FRAME_STRIDE:-1}"

  # Rollout-cache-only knobs. They are harmless in index-only mode.
  CACHE_CHUNK_SIZE="${CACHE_CHUNK_SIZE:-128}"
  PERTURBATION_MODE="${PERTURBATION_MODE:-hrl_curriculum_bank}"
  CURRICULUM_BANK_SIZE="${CURRICULUM_BANK_SIZE:-16}"
else
  STAGE1_MODE="tiny"

  # Tiny mode: route smoke test only, not real training data.
  MAX_MOTIONS="${MAX_MOTIONS:-1}"
  MAX_SEGMENTS="${MAX_SEGMENTS:-all}"
  FRAME_STRIDE="${FRAME_STRIDE:-100000}"
  CACHE_CHUNK_SIZE="${CACHE_CHUNK_SIZE:-1}"
  PERTURBATION_MODE="${PERTURBATION_MODE:-discrete_bank}"
  PERTURBATION_STRENGTHS="${PERTURBATION_STRENGTHS:-0.0}"
  CURRICULUM_BANK_SIZE="${CURRICULUM_BANK_SIZE:-1}"
  VALIDATION_REQUIRE_BOUNDARY_DIAGNOSTIC="${VALIDATION_REQUIRE_BOUNDARY_DIAGNOSTIC:-0}"
fi

if [[ "${STAGE1_BUILD_ROLLOUT_CACHE}" != "1" ]]; then
  STAGE1_MODE="index"
fi

# B5: Validation and rollout-cache thresholds.

VARIANTS_PER_STRENGTH="${VARIANTS_PER_STRENGTH:-1}"
VALIDATION_MIN_SEGMENTS="${VALIDATION_MIN_SEGMENTS:-1}"
VALIDATION_MIN_NOISY="${VALIDATION_MIN_NOISY:-1}"

cd "${FEMR_ROOT}"

export CUDA_VISIBLE_DEVICES
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export FEMR_ROOT
export FEMR_DATA_ROOT
export FEMR_LOG_ROOT="${FEMR_LOG_ROOT:-${FEMR_ROOT}}"
export WANDB_DIR="${WANDB_DIR:-${FEMR_LOG_ROOT}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${FEMR_LOG_ROOT}/.wandb_cache}"
export DEVICE
export PYTHON_BIN
export MAX_MOTIONS
export MAX_SEGMENTS
export FRAME_STRIDE
export CACHE_CHUNK_SIZE
export PERTURBATION_MODE
export CURRICULUM_BANK_SIZE
export VARIANTS_PER_STRENGTH
export VALIDATION_MIN_SEGMENTS
export VALIDATION_MIN_NOISY
if [[ "${STAGE1_FULL}" != "1" ]]; then
  export PERTURBATION_STRENGTHS
  export VALIDATION_REQUIRE_BOUNDARY_DIAGNOSTIC
fi

mkdir -p "$(dirname "${LOG_PATH}")"

if [[ "${STAGE1_BUILD_ROLLOUT_CACHE}" == "1" ]]; then
  CMD=(
    bash run/run_frontres_stage1_segment_cache.sh
    "${MOTION_PATH}"
    "${NUM_ENVS}"
    "${SEGMENT_K}"
    "${CACHE_DIR}"
  )
else
  CMD=(
    "${PYTHON_BIN}" scripts/rsl_rl/build_frontres_stage1_segment_index.py
    "${MOTION_PATH}"
    "${CACHE_DIR}"
    --segment-k "${SEGMENT_K}"
    --frame-stride "${FRAME_STRIDE}"
    --max-motions "${MAX_MOTIONS}"
    --max-segments "${MAX_SEGMENTS}"
  )
fi

nohup "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
PID="$!"

echo "[FrontRES Stage1] submitted pid=${PID}"
echo "[FrontRES Stage1] log=${LOG_PATH}"
echo "[FrontRES Stage1] mode=${STAGE1_MODE}"
echo "[FrontRES Stage1] build_rollout_cache=${STAGE1_BUILD_ROLLOUT_CACHE}"
echo "[FrontRES Stage1] cuda_visible_devices=${CUDA_VISIBLE_DEVICES} device=${DEVICE}"
echo "[FrontRES Stage1] python=${PYTHON_BIN}"
echo "[FrontRES Stage1] motion=${MOTION_PATH}"
echo "[FrontRES Stage1] num_envs=${NUM_ENVS}"
echo "[FrontRES Stage1] segment_k=${SEGMENT_K}"
echo "[FrontRES Stage1] cache_dir=${CACHE_DIR}"
echo "[FrontRES Stage1] max_motions=${MAX_MOTIONS}"
echo "[FrontRES Stage1] max_segments=${MAX_SEGMENTS}"
echo "[FrontRES Stage1] frame_stride=${FRAME_STRIDE}"
echo "[FrontRES Stage1] cache_chunk_size=${CACHE_CHUNK_SIZE}"
echo "[FrontRES Stage1] perturbation_mode=${PERTURBATION_MODE}"
echo "[FrontRES Stage1] variants_per_strength=${VARIANTS_PER_STRENGTH}"
echo "[FrontRES Stage1] follow: tail -f ${LOG_PATH}"
