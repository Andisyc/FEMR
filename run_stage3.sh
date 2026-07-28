#!/usr/bin/env bash
set -euo pipefail

# FrontRES Stage 3 Segment Replay training launcher.
#
# Status: active.
# Upstream: Stage 2 warmup checkpoint and Stage 1 segment cache.
# Downstream: Segment Replay HRL checkpoints and online eval logs.
# Evidence: script-level route, not a live validation.
#
# B1: Runtime owner. Select GPU, cache, and log sink.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
CACHE_DIR="${CACHE_DIR:-/hdd1/cyx/AMASS_G1Segment}"
LOG_PATH="${LOG_PATH:-/hdd1/cyx/FEMR/train_stage3_segment_hrl.txt}"

# B2: Model and dataset contract. Positional args only override these two.

MODEL_PATH="${1:-/hdd1/cyx/FEMR/model/model_warmup.pt}"
MOTION_PATH="${2:-/hdd1/cyx/AMASS_G1NPZ_Final}"

# B3: Stage 3 training schedule.

NUM_ENVS="${3:-${NUM_ENVS:-12000}}"
MAX_ITERS="${4:-${MAX_ITERS:-2000}}"
UPDATE_STEPS="${5:-${UPDATE_STEPS:-4}}"
MODE="${6:-${MODE:-train}}"
EXTRA_TRAIN_ARGS=("${@:7}")

# B4: Online eval. Use run_eval.sh for sequence/offline eval.

PERIODIC_EVAL_ENABLED="${PERIODIC_EVAL_ENABLED:-0}"
PERIODIC_EVAL_INTERVAL="${PERIODIC_EVAL_INTERVAL:-100}"

# B5: PPO safety knobs for direct Delta SE Stage 3.

FRONTRES_SEGMENT_PPO_SCHEDULE="${FRONTRES_SEGMENT_PPO_SCHEDULE:-adaptive}"
FRONTRES_SEGMENT_PPO_LR="${FRONTRES_SEGMENT_PPO_LR:-1e-6}"

# B6: Cache, logging, and distributed launch.

SHARD_CACHE_SIZE="${SHARD_CACHE_SIZE:-8}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-FEMR}"
RUN_NAME="${RUN_NAME:-FEMR_STAGE3_SEGMENT_HRL}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
FRONTRES_V015_FUTURE_OFFSETS="${FRONTRES_V015_FUTURE_OFFSETS:-1,2}"
FRONTRES_V015_K_CURRICULUM="${FRONTRES_V015_K_CURRICULUM:-8:2:200:500:1300,16:3:300:300:900,32:4:400:300:625}"
FRONTRES_G5_S4_BOUNDED="${FRONTRES_G5_S4_BOUNDED:-0}"

if [[ "${MODE}" == "train" && "${FRONTRES_V015_K_CURRICULUM}" != "8:2:200:500:1300,16:3:300:300:900,32:4:400:300:625" ]]; then
  echo "FRS-TRAIN-v011 requires the frozen K8/M2,K16/M3,K32/M4 schedule" >&2
  exit 4
fi

if [[ "${FRONTRES_G5_S4_BOUNDED}" == "1" ]]; then
  if [[ "${MODE}" != "train" || "${NUM_ENVS}" != "8" || "${MAX_ITERS}" != "1" || "${UPDATE_STEPS}" != "1" ]]; then
    echo "G5-S4 bounded Stage 3 requires train mode, 8 envs, 1 iteration, and 1 update" >&2
    exit 4
  fi
  RUN_NAME="G5_S4_BOUND_V015"
fi

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export FEMR_LOG_ROOT="${FEMR_LOG_ROOT:-/hdd1/cyx/FEMR}"
export WANDB_DIR="${WANDB_DIR:-/hdd1/cyx/FEMR}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/hdd1/cyx/FEMR/.wandb_cache}"
export CACHE_DIR
export SHARD_CACHE_SIZE
export PERIODIC_EVAL_ENABLED
export PERIODIC_EVAL_INTERVAL
export LOG_PROJECT_NAME
export RUN_NAME
export NPROC_PER_NODE
export FRONTRES_V015_FUTURE_OFFSETS
export FRONTRES_V015_K_CURRICULUM
export FRONTRES_G5_S4_BOUNDED
export FRONTRES_CHECKPOINT_INTERVAL="${FRONTRES_CHECKPOINT_INTERVAL:-1}"

mkdir -p "$(dirname "${LOG_PATH}")"

CMD=(
  bash run/run_frontres_stage3_segment_hrl.sh
  "${MODEL_PATH}"
  "${MOTION_PATH}"
  "${NUM_ENVS}"
  "${MAX_ITERS}"
  "${UPDATE_STEPS}"
  "${MODE}"
  --frontres_segment_ppo_schedule "${FRONTRES_SEGMENT_PPO_SCHEDULE}"
  --frontres_segment_ppo_lr "${FRONTRES_SEGMENT_PPO_LR}"
)

if [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_TRAIN_ARGS[@]}")
fi

if [[ "${FRONTRES_STAGE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[FrontRES Stage3] preflight only"
  printf '%q ' "${CMD[@]}"
  echo
  "${CMD[@]}"
  exit 0
fi

nohup "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
PID="$!"

echo "[FrontRES Stage3] submitted pid=${PID}"
echo "[FrontRES Stage3] mode=${MODE}"
echo "[FrontRES Stage3] model=${MODEL_PATH}"
echo "[FrontRES Stage3] motion=${MOTION_PATH}"
echo "[FrontRES Stage3] num_envs=${NUM_ENVS}"
echo "[FrontRES Stage3] max_iters=${MAX_ITERS}"
echo "[FrontRES Stage3] update_steps=${UPDATE_STEPS}"
echo "[FrontRES Stage3] periodic_eval=${PERIODIC_EVAL_ENABLED}"
echo "[FrontRES Stage3] eval_interval=${PERIODIC_EVAL_INTERVAL}"
echo "[FrontRES Stage3] ppo_schedule=${FRONTRES_SEGMENT_PPO_SCHEDULE}"
echo "[FrontRES Stage3] ppo_lr=${FRONTRES_SEGMENT_PPO_LR}"
echo "[FrontRES Stage3] log=${LOG_PATH}"
echo "[FrontRES Stage3] follow: tail -f ${LOG_PATH}"
