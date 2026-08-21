#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
HSL="${HSL_CHECKPOINT:-/hdd1/cyx/FEMR/g1_flat_frontres_stage1_hsl/2026-08-04_18-14-12_V017_HSL_V2_FULL/model_warmup.pt}"
MOTION="${MOTION_ROOT:-/hdd1/cyx/AMASS_G1NPZ_Final}"
CACHE="${CACHE_DIR:-/hdd1/cyx/AMASS_G1Segment}"
MANIFEST="${CLEAN_CALIBRATION_MANIFEST:-${ROOT}/note/testing/manifests/frontres_clean_calibration_v010_server.json}"
TAG="${RUN_TAG:-20260821}"
RESULT="${CLEAN_CALIBRATION_RESULT:-${ROOT}/FRS_EVAL_V010_CLEAN_CALIBRATION_R1_${TAG}.json}"
LOG="${CLEAN_CALIBRATION_LOG:-${ROOT}/FRS_EVAL_V010_CLEAN_CALIBRATION_R1_${TAG}.log}"

[[ -f "${HSL}" ]] || { echo "missing HSL checkpoint: ${HSL}" >&2; exit 2; }
[[ -d "${MOTION}" ]] || { echo "missing motion root: ${MOTION}" >&2; exit 2; }
[[ -d "${CACHE}" ]] || { echo "missing segment cache: ${CACHE}" >&2; exit 2; }
[[ -f "${MANIFEST}" ]] || { echo "missing v010 manifest: ${MANIFEST}" >&2; exit 2; }
[[ ! -e "${RESULT}" ]] || { echo "result already exists: ${RESULT}" >&2; exit 2; }
[[ ! -e "${LOG}" ]] || { echo "log already exists: ${LOG}" >&2; exit 2; }

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"

nohup "${PYTHON_BIN}" scripts/rsl_rl/train.py \
  --task=FrontRES-Unified-Tracking-Flat-G1-v0 \
  --num_envs=16 \
  --motion="${MOTION}" \
  --headless \
  --max_iterations=0 \
  --frontres_stage=stage3_segment_hrl \
  --frontres_specialist_mode=rp \
  --frontres_segment_cache_dir="${CACHE}" \
  --frontres_v015_hsl_initializer_checkpoint="${HSL}" \
  --frontres_clean_calibration_collect_only \
  --frontres_clean_calibration_manifest="${MANIFEST}" \
  --frontres_clean_calibration_result="${RESULT}" \
  --experiment_name=g1_flat_frontres_stage3_segment_hrl \
  --run_name=FRS_EVAL_V010_CLEAN_CALIBRATION_R1_${TAG} \
  >"${LOG}" 2>&1 &

PID=$!
echo "started PID=${PID}"
echo "log=${LOG}"
echo "result=${RESULT}"
