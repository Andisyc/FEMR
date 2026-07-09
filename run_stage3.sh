#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_stage3.sh STAGE2_CHECKPOINT [MOTION_PATH] [NUM_ENVS] [MAX_ITERS] [UPDATE_STEPS] [MODE] [TRAIN_ARGS...]"
  echo
 echo "MODE can be: train, sentinel, probe, storage, single_update, update_loop, offline_eval, sequence_eval."
 echo "CACHE_DIR selects the Stage 1 Segment Replay cache used by Stage 3."
 echo "PERIODIC_EVAL_ENABLED=1 enables periodic long-rollout eval; PERIODIC_EVAL_INTERVAL controls its interval."
echo "OFFLINE_EVAL_STEPS controls checkpoint eval rollout length when MODE=offline_eval or sequence_eval."
echo "OFFLINE_EVAL_SEQUENCES and OFFLINE_EVAL_MAX_PREROLL_STEPS control MODE=sequence_eval."
echo "FRONTRES_SPECIALIST_MODE selects Stage 3 perturbation family preset, default rp."
echo "SHARD_CACHE_SIZE controls the lazy Stage 1 cache LRU size."
echo "Append --frontres_segment_ppo_schedule adaptive --frontres_segment_ppo_lr 1e-6 to test adaptive Segment PPO trust-region control."
  echo "Set FRONTRES_STAGE_PREFLIGHT_ONLY=1 to print and validate the startup command without launching IsaacLab."
  exit 1
fi

STAGE2_CHECKPOINT="$1"
MOTION_PATH="${2:-/hdd1/cyx/AMASS_G1NPZ_Final}"
NUM_ENVS="${3:-12000}"
MAX_ITERS="${4:-2000}"
UPDATE_STEPS="${5:-4}"
MODE="${6:-train}"
EXTRA_TRAIN_ARGS=("${@:7}")
CACHE_DIR="${CACHE_DIR:-/hdd1/cyx/AMASS_G1Segment}"
SHARD_CACHE_SIZE="${SHARD_CACHE_SIZE:-8}"
LOG_PATH="${LOG_PATH:-/hdd1/cyx/FEMR/train_stage3_segment_hrl.txt}"
RUN_FOREGROUND="${RUN_FOREGROUND:-0}"

cd "$(dirname "$0")"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export FEMR_LOG_ROOT="${FEMR_LOG_ROOT:-/hdd1/cyx/FEMR}"
export WANDB_DIR="${WANDB_DIR:-/hdd1/cyx/FEMR}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/hdd1/cyx/FEMR/.wandb_cache}"
export CACHE_DIR
export SHARD_CACHE_SIZE
export PERIODIC_EVAL_ENABLED="${PERIODIC_EVAL_ENABLED:-0}"
export PERIODIC_EVAL_INTERVAL="${PERIODIC_EVAL_INTERVAL:-100}"
export OFFLINE_EVAL_SEGMENTS="${OFFLINE_EVAL_SEGMENTS:-${NUM_ENVS}}"
export OFFLINE_EVAL_STEPS="${OFFLINE_EVAL_STEPS:-500}"
export FRONTRES_SPECIALIST_MODE="${FRONTRES_SPECIALIST_MODE:-rp}"

CMD=(
  bash run/run_frontres_stage3_segment_hrl.sh
  "${STAGE2_CHECKPOINT}"
  "${MOTION_PATH}"
  "${NUM_ENVS}"
  "${MAX_ITERS}"
  "${UPDATE_STEPS}"
  "${MODE}"
)
if [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_TRAIN_ARGS[@]}")
fi

if [[ "${FRONTRES_STAGE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[FrontRES Stage3] preflight only"
  "${CMD[@]}"
  exit 0
fi

mkdir -p "$(dirname "${LOG_PATH}")"

if [[ "${RUN_FOREGROUND}" == "1" ]]; then
  echo "[FrontRES Stage3] running in foreground; log=${LOG_PATH}"
  "${CMD[@]}" >"${LOG_PATH}" 2>&1
else
  nohup "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
  PID="$!"
  echo "[FrontRES Stage3] submitted pid=${PID}"
  echo "[FrontRES Stage3] log=${LOG_PATH}"
  echo "[FrontRES Stage3] follow: tail -f ${LOG_PATH}"
fi
