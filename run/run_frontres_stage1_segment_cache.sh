#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run/run_frontres_stage1_segment_cache.sh MOTION_PATH [NUM_ENVS] [SEGMENT_K] [CACHE_DIR]"
  echo
  echo "Stage 1 builds the Segment Replay cache: motion segments, Clean states, and stepped Noisy baselines."
  echo "Example:"
  echo "  bash run/run_frontres_stage1_segment_cache.sh /path/to/motions 12000 4 /hdd1/cyx/AMASS_G1Segment"
  exit 1
fi

MOTION_PATH="$1"
NUM_ENVS="${2:-12000}"
SEGMENT_K="${3:-4}"
CACHE_DIR="${4:-/hdd1/cyx/AMASS_G1Segment}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-FEMR}"
RUN_NAME="${RUN_NAME:-FrontRES_STAGE1_SEGMENT_CACHE}"
PERTURBATION_STRENGTHS="${PERTURBATION_STRENGTHS:-0.0,0.25,0.5,0.75,1.0}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_MOTIONS="${MAX_MOTIONS:-1}"
MAX_SEGMENTS="${MAX_SEGMENTS:-1}"
VARIANTS_PER_STRENGTH="${VARIANTS_PER_STRENGTH:-1}"

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
  --frontres_segment_cache_perturbation_strengths "${PERTURBATION_STRENGTHS}"
)

if [[ -n "${CACHE_DIR}" ]]; then
  CMD+=(--frontres_segment_cache_dir "${CACHE_DIR}")
fi

"${CMD[@]}"
