#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run/run_frontres_stage1_segment_cache.sh MOTION_PATH [NUM_ENVS] [SEGMENT_K] [CACHE_DIR]"
  echo
  echo "Stage 1 builds the Segment Replay cache: motion segments, Clean states, and stepped Noisy baselines."
  echo "After a successful build, the script validates the written cache by default."
  echo "MAX_MOTIONS/MAX_SEGMENTS accept positive integers or all/auto."
  echo "CACHE_CHUNK_SIZE controls how many cache records are written per payload shard."
  echo "Set FRONTRES_STAGE1_PREFLIGHT_ONLY=1 to print and validate the startup command without launching IsaacLab."
  echo "Example:"
  echo "  MAX_MOTIONS=all MAX_SEGMENTS=all CACHE_CHUNK_SIZE=128 bash run/run_frontres_stage1_segment_cache.sh /path/to/motions 12000 4 /hdd1/cyx/AMASS_G1Segment"
  exit 1
fi

MOTION_PATH="$1"
NUM_ENVS="${2:-12000}"
SEGMENT_K="${3:-4}"
CACHE_DIR="${4:-/hdd1/cyx/AMASS_G1Segment}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEVICE="${DEVICE:-cuda:0}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-FEMR}"
RUN_NAME="${RUN_NAME:-FrontRES_STAGE1_SEGMENT_CACHE}"
PERTURBATION_MODE="${PERTURBATION_MODE:-hrl_curriculum_bank}"
PERTURBATION_STRENGTHS="${PERTURBATION_STRENGTHS:-0.0,0.25,0.5,0.75,1.0}"
CURRICULUM_BANK_SIZE="${CURRICULUM_BANK_SIZE:-16}"
CURRICULUM_FRONTIER_SCALE="${CURRICULUM_FRONTIER_SCALE:-2.0}"
CURRICULUM_DR_MIN="${CURRICULUM_DR_MIN:-1.25}"
CURRICULUM_DR_MAX="${CURRICULUM_DR_MAX:-4.5}"
CURRICULUM_PROGRESS="${CURRICULUM_PROGRESS:-0.8}"
CURRICULUM_SEQ_IDX="${CURRICULUM_SEQ_IDX:-17}"
CURRICULUM_ACTIVE_DIMS="${CURRICULUM_ACTIVE_DIMS:-0,1,2,3,4,5}"
CURRICULUM_TEMPORAL_MODE="${CURRICULUM_TEMPORAL_MODE:-single}"
CURRICULUM_BURST_MIN_STEPS="${CURRICULUM_BURST_MIN_STEPS:-4}"
CURRICULUM_BURST_MAX_STEPS="${CURRICULUM_BURST_MAX_STEPS:-8}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_MOTIONS="${MAX_MOTIONS:-all}"
MAX_SEGMENTS="${MAX_SEGMENTS:-all}"
VARIANTS_PER_STRENGTH="${VARIANTS_PER_STRENGTH:-1}"
CACHE_CHUNK_SIZE="${CACHE_CHUNK_SIZE:-128}"
VALIDATE_AFTER_BUILD="${VALIDATE_AFTER_BUILD:-1}"
VALIDATION_PYTHON_BIN="${VALIDATION_PYTHON_BIN:-python}"
VALIDATION_EXPECT_MODE="${VALIDATION_EXPECT_MODE:-${PERTURBATION_MODE}}"
VALIDATION_MIN_SEGMENTS="${VALIDATION_MIN_SEGMENTS:-1}"
VALIDATION_MIN_NOISY="${VALIDATION_MIN_NOISY:-1}"
VALIDATION_REQUIRE_BOUNDARY_DIAGNOSTIC="${VALIDATION_REQUIRE_BOUNDARY_DIAGNOSTIC:-auto}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" scripts/rsl_rl/train.py --distributed)
else
  LAUNCH=(python scripts/rsl_rl/train.py)
fi

CMD=(
  env HYDRA_FULL_ERROR=1
  "${LAUNCH[@]}"
  --task=FrontRES-Unified-Tracking-Flat-G1-v0
  --num_envs="${NUM_ENVS}"
  --motion "${MOTION_PATH}"
  --headless
  --device "${DEVICE}"
  --logger wandb
  --log_project_name "${LOG_PROJECT_NAME}"
  --experiment_name g1_flat_frontres_stage1_segment_cache
  --run_name "${RUN_NAME}"
  --max_iterations 0
  --frontres_stage stage1_segment_cache
  --frontres_segment_cache_k "${SEGMENT_K}"
  --frontres_segment_cache_frame_stride "${FRAME_STRIDE}"
  --frontres_segment_cache_max_motions "${MAX_MOTIONS}"
  --frontres_segment_cache_max_segments "${MAX_SEGMENTS}"
  --frontres_segment_cache_variants_per_strength "${VARIANTS_PER_STRENGTH}"
  --frontres_segment_cache_chunk_size "${CACHE_CHUNK_SIZE}"
  --frontres_segment_cache_perturbation_mode "${PERTURBATION_MODE}"
  --frontres_segment_cache_perturbation_strengths "${PERTURBATION_STRENGTHS}"
  --frontres_segment_cache_curriculum_bank_size "${CURRICULUM_BANK_SIZE}"
  --frontres_segment_cache_curriculum_frontier_scale "${CURRICULUM_FRONTIER_SCALE}"
  --frontres_segment_cache_curriculum_dr_min "${CURRICULUM_DR_MIN}"
  --frontres_segment_cache_curriculum_dr_max "${CURRICULUM_DR_MAX}"
  --frontres_segment_cache_curriculum_progress "${CURRICULUM_PROGRESS}"
  --frontres_segment_cache_curriculum_seq_idx "${CURRICULUM_SEQ_IDX}"
  --frontres_segment_cache_curriculum_active_dims "${CURRICULUM_ACTIVE_DIMS}"
  --frontres_segment_cache_curriculum_temporal_mode "${CURRICULUM_TEMPORAL_MODE}"
  --frontres_segment_cache_curriculum_burst_min_steps "${CURRICULUM_BURST_MIN_STEPS}"
  --frontres_segment_cache_curriculum_burst_max_steps "${CURRICULUM_BURST_MAX_STEPS}"
)

if [[ "${CURRICULUM_INCLUDE_HARD_AS_TRAIN:-0}" == "1" ]]; then
  CMD+=(--frontres_segment_cache_curriculum_include_hard_as_train)
fi

if [[ -n "${CACHE_DIR}" ]]; then
  CMD+=(--frontres_segment_cache_dir "${CACHE_DIR}")
fi

if [[ "${FRONTRES_STAGE1_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  joined=" ${CMD[*]} "
  for required in \
    " scripts/rsl_rl/train.py " \
    " --frontres_stage stage1_segment_cache " \
    " --max_iterations 0 " \
    " --frontres_segment_cache_dir ${CACHE_DIR} " \
    " --frontres_segment_cache_k ${SEGMENT_K} " \
    " --frontres_segment_cache_frame_stride ${FRAME_STRIDE} " \
    " --frontres_segment_cache_max_motions ${MAX_MOTIONS} " \
    " --frontres_segment_cache_max_segments ${MAX_SEGMENTS} " \
    " --frontres_segment_cache_chunk_size ${CACHE_CHUNK_SIZE} " \
    " --frontres_segment_cache_perturbation_mode ${PERTURBATION_MODE} " \
    " --frontres_segment_cache_curriculum_bank_size ${CURRICULUM_BANK_SIZE} " \
    " --experiment_name g1_flat_frontres_stage1_segment_cache "
  do
    if [[ "${joined}" != *"${required}"* ]]; then
      echo "Stage 1 startup preflight failed; missing cmd fragment:${required}" >&2
      echo -n "Command: " >&2
      printf '%q ' "${CMD[@]}" >&2
      echo >&2
      exit 4
    fi
  done
  echo "[FrontRES Stage1 startup preflight] PASS"
  echo -n "Command: "
  printf '%q ' "${CMD[@]}"
  echo
  if [[ "${VALIDATE_AFTER_BUILD}" == "1" ]]; then
    echo "[FrontRES Stage1 validator preflight] enabled cache_dir=${CACHE_DIR} expect_mode=${VALIDATION_EXPECT_MODE} min_segments=${VALIDATION_MIN_SEGMENTS} min_noisy=${VALIDATION_MIN_NOISY}"
  else
    echo "[FrontRES Stage1 validator preflight] disabled"
  fi
  exit 0
fi

"${CMD[@]}"

if [[ "${VALIDATE_AFTER_BUILD}" == "1" ]]; then
  VALIDATE_CMD=(
    "${VALIDATION_PYTHON_BIN}"
    source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_validator.py
    "${CACHE_DIR}"
    --expect-mode "${VALIDATION_EXPECT_MODE}"
    --min-segments "${VALIDATION_MIN_SEGMENTS}"
    --min-noisy "${VALIDATION_MIN_NOISY}"
  )
  if [[ "${VALIDATION_REQUIRE_BOUNDARY_DIAGNOSTIC}" == "1" ]] || {
    [[ "${VALIDATION_REQUIRE_BOUNDARY_DIAGNOSTIC}" == "auto" ]] &&
    [[ "${PERTURBATION_MODE}" == "hrl_curriculum_bank" ]] &&
    [[ "${CURRICULUM_INCLUDE_HARD_AS_TRAIN:-0}" != "1" ]]
  }; then
    VALIDATE_CMD+=(--require-boundary-diagnostic)
  fi
  "${VALIDATE_CMD[@]}"
fi
