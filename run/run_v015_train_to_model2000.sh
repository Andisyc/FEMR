#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU_ID="${CUDA_VISIBLE_DEVICES:-1}"
RESUME_CHECKPOINT="${FRONTRES_V015_RESUME_CHECKPOINT:-/hdd1/cyx/FEMR/g1_flat_frontres_stage3_segment_hrl/2026-07-24_18-02-23_P4_ACTOR_RAMP_BLOCK50/model_251.pt}"
HSL_CHECKPOINT="${FRONTRES_V015_HSL_CHECKPOINT:-/hdd1/cyx/FEMR/g1_flat_frontres_stage1_hsl/2026-07-21_17-06-12_G2_S4_BOUND_HSL/model_warmup.pt}"
MOTION_ROOT="${MOTION_ROOT:-/hdd1/cyx/AMASS_G1NPZ_Final}"
CACHE_DIR="${CACHE_DIR:-/hdd1/cyx/AMASS_G1Segment}"
LOG_PATH="${LOG_PATH:-/hdd1/cyx/FEMR/v015_train_to_model2000_gpu${GPU_ID}.log}"
CURRENT_ITERATION=251
TARGET_ITERATION=2000
ADDITIONAL_ITERATIONS=$((TARGET_ITERATION - CURRENT_ITERATION))

for required_file in "${RESUME_CHECKPOINT}" "${HSL_CHECKPOINT}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[V015-LONG-TRAIN] missing required checkpoint: ${required_file}" >&2
    exit 2
  fi
done
for required_dir in "${MOTION_ROOT}" "${CACHE_DIR}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "[V015-LONG-TRAIN] missing required directory: ${required_dir}" >&2
    exit 2
  fi
done
if [[ -e "${LOG_PATH}" ]]; then
  echo "[V015-LONG-TRAIN] refusing to overwrite existing log: ${LOG_PATH}" >&2
  exit 2
fi
if pgrep -af "V015_TRAIN_TO_MODEL2000" >/dev/null 2>&1; then
  echo "[V015-LONG-TRAIN] an existing model-2000 training process is active" >&2
  pgrep -af "V015_TRAIN_TO_MODEL2000" >&2 || true
  exit 2
fi

echo "[V015-LONG-TRAIN] resume=${RESUME_CHECKPOINT}"
echo "[V015-LONG-TRAIN] absolute=${CURRENT_ITERATION}->${TARGET_ITERATION} additional=${ADDITIONAL_ITERATIONS}"
echo "[V015-LONG-TRAIN] actor_ramp_remaining=449 joint_updates=1300 checkpoint_interval=50"
echo "[V015-LONG-TRAIN] gpu=${GPU_ID} log=${LOG_PATH}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
CACHE_DIR="${CACHE_DIR}" \
LOG_PATH="${LOG_PATH}" \
RUN_NAME="V015_TRAIN_TO_MODEL2000" \
FRONTRES_CHECKPOINT_INTERVAL=50 \
FRONTRES_V015_K_CURRICULUM=8:200:500:0 \
FRONTRES_V015_RESUME_CHECKPOINT="${RESUME_CHECKPOINT}" \
bash run_stage3.sh \
  "${HSL_CHECKPOINT}" \
  "${MOTION_ROOT}" \
  8 "${ADDITIONAL_ITERATIONS}" 1 train

echo "[V015-LONG-TRAIN] submitted"
echo "[V015-LONG-TRAIN] follow: tail -f ${LOG_PATH}"
