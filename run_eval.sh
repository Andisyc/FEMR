#!/usr/bin/env bash
set -euo pipefail

# FrontRES Stage 3 sequence offline eval launcher.
#
# Status: active.
# Upstream: Stage 3 checkpoint and Stage 1 segment cache.
# Downstream: sequence eval log with per-motion and aggregate metrics.
# Evidence: script-level route, not a live validation.
#
# This script calls scripts/rsl_rl/train.py directly. It does not wrap
# run_stage3.sh, so eval parameters stay explicit.
#
# B1: Runtime owner. Select GPU, cache, and log sink.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
CACHE_DIR="${CACHE_DIR:-/hdd1/cyx/AMASS_G1Segment}"
LOG_PATH="${LOG_PATH:-/hdd1/cyx/FEMR/stage3_sequence_eval.txt}"

# B2: Model and dataset contract. Positional args only override these two.

MODEL_PATH="${1:-/hdd1/cyx/FEMR/model/model_600.pt}"
MOTION_PATH="${2:-/hdd1/cyx/AMASS_G1NPZ_Final}"

# B3: Sequence eval sampling contract.

NUM_ENVS="${NUM_ENVS:-4}"
OFFLINE_EVAL_SEQUENCES="${OFFLINE_EVAL_SEQUENCES:-2}"
OFFLINE_EVAL_STEPS="${OFFLINE_EVAL_STEPS:-120}"
OFFLINE_EVAL_MAX_PREROLL_STEPS="${OFFLINE_EVAL_MAX_PREROLL_STEPS:-120}"

# B4: Eval mode, cache, and log labels.

FRONTRES_SPECIALIST_MODE="${FRONTRES_SPECIALIST_MODE:-rp}"
SHARD_CACHE_SIZE="${SHARD_CACHE_SIZE:-8}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-FEMR}"
RUN_NAME="${RUN_NAME:-FEMR_STAGE3_SEQUENCE_EVAL}"

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export FEMR_LOG_ROOT="${FEMR_LOG_ROOT:-/hdd1/cyx/FEMR}"
export WANDB_DIR="${WANDB_DIR:-/hdd1/cyx/FEMR}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/hdd1/cyx/FEMR/.wandb_cache}"

mkdir -p "$(dirname "${LOG_PATH}")"

CMD=(
  python scripts/rsl_rl/train.py
  --task=FrontRES-Unified-Tracking-Flat-G1-v0
  --num_envs="${NUM_ENVS}"
  --motion "${MOTION_PATH}"
  --headless
  --logger tensorboard
  --log_project_name "${LOG_PROJECT_NAME}"
  --experiment_name g1_flat_frontres_stage3_sequence_eval
  --run_name "${RUN_NAME}"
  --max_iterations 1
  --resume_student_checkpoint "${MODEL_PATH}"
  --is_full_resume False
  --frontres_stage stage3_segment_hrl
  --frontres_specialist_mode "${FRONTRES_SPECIALIST_MODE}"
  --frontres_segment_cache_dir "${CACHE_DIR}"
  --frontres_segment_shard_cache_size "${SHARD_CACHE_SIZE}"
  --frontres_segment_live_update_steps 1
  --frontres_segment_sequence_offline_eval_only
  --frontres_segment_sequence_eval_sequences "${OFFLINE_EVAL_SEQUENCES}"
  --frontres_segment_sequence_eval_max_preroll_steps "${OFFLINE_EVAL_MAX_PREROLL_STEPS}"
  --frontres_segment_offline_eval_steps "${OFFLINE_EVAL_STEPS}"
)

nohup "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
PID="$!"

echo "[FrontRES Sequence Eval] submitted pid=${PID}"
echo "[FrontRES Sequence Eval] model=${MODEL_PATH}"
echo "[FrontRES Sequence Eval] motion=${MOTION_PATH}"
echo "[FrontRES Sequence Eval] num_envs=${NUM_ENVS}"
echo "[FrontRES Sequence Eval] sequences=${OFFLINE_EVAL_SEQUENCES}"
echo "[FrontRES Sequence Eval] steps=${OFFLINE_EVAL_STEPS}"
echo "[FrontRES Sequence Eval] max_preroll=${OFFLINE_EVAL_MAX_PREROLL_STEPS}"
echo "[FrontRES Sequence Eval] specialist=${FRONTRES_SPECIALIST_MODE}"
echo "[FrontRES Sequence Eval] log=${LOG_PATH}"
echo "[FrontRES Sequence Eval] follow: tail -f ${LOG_PATH}"
