#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run/run_frontres_stage3_segment_hrl.sh STAGE1_CHECKPOINT MOTION_PATH [NUM_ENVS] [MAX_ITERS] [UPDATE_STEPS] [MODE]"
  echo
  echo "Stage 3 loads a Stage 1 Delta SE proposal checkpoint and trains Segment Replay HRL."
  echo "MODE can be: train, sentinel, probe, storage, single_update, update_loop."
  echo "Example:"
  echo "  bash run/run_frontres_stage3_segment_hrl.sh /path/to/stage1/model.pt /path/to/motions 12000 2000 4 train"
  echo "  bash run/run_frontres_stage3_segment_hrl.sh /path/to/stage1/model.pt /path/to/motions 1 1 1 update_loop"
  exit 1
fi

STAGE1_CHECKPOINT="$1"
MOTION_PATH="$2"
NUM_ENVS="${3:-12000}"
MAX_ITERS="${4:-2000}"
UPDATE_STEPS="${5:-4}"
MODE="${6:-train}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-FEMR}"
RUN_NAME="${RUN_NAME:-FEMR_STAGE3_SEGMENT_HRL}"
CACHE_DIR="${CACHE_DIR:-/hdd1/cyx/AMASS_G1Segment}"
CONTRACT_SUITE="${FRONTRES_STAGE3_CONTRACT_SUITE:-source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py}"
CONTRACT_PYTHON="${FRONTRES_STAGE3_CONTRACT_PYTHON:-python}"

if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
  echo "Stage 1 checkpoint not found: ${STAGE1_CHECKPOINT}" >&2
  exit 2
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" scripts/rsl_rl/train.py --distributed)
else
  LAUNCH=(python scripts/rsl_rl/train.py)
fi

MODE_ARGS=()
case "${MODE}" in
  train)
    ;;
  sentinel)
    MODE_ARGS=(--frontres_segment_live_sentinel_only)
    ;;
  probe)
    MODE_ARGS=(--frontres_segment_live_probe_only)
    ;;
  storage)
    MODE_ARGS=(--frontres_segment_live_storage_write_only)
    ;;
  single_update)
    MODE_ARGS=(--frontres_segment_live_single_update_only)
    ;;
  update_loop)
    MODE_ARGS=(--frontres_segment_live_update_loop_only)
    ;;
  *)
    echo "Unknown Stage 3 MODE: ${MODE}" >&2
    exit 3
    ;;
esac

TRAIN_CMD=(
  "${LAUNCH[@]}"
  --task=FrontRES-Unified-Tracking-Flat-G1-v0
  --num_envs="${NUM_ENVS}"
  --motion "${MOTION_PATH}"
  --headless
  --logger wandb
  --log_project_name "${LOG_PROJECT_NAME}"
  --experiment_name g1_flat_frontres_stage3_segment_hrl
  --run_name "${RUN_NAME}"
  --max_iterations "${MAX_ITERS}"
  --resume_student_checkpoint "${STAGE1_CHECKPOINT}"
  --is_full_resume False
  --frontres_stage stage3_segment_hrl
  --frontres_segment_cache_dir "${CACHE_DIR}"
  --frontres_segment_live_update_steps "${UPDATE_STEPS}"
)

if [[ ${#MODE_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${MODE_ARGS[@]}")
fi

if [[ "${FRONTRES_STAGE3_RUN_CONTRACTS:-0}" == "1" ]]; then
  echo "[FrontRES Stage3 contract preflight] START suite=${CONTRACT_SUITE} python=${CONTRACT_PYTHON}"
  "${CONTRACT_PYTHON}" "${CONTRACT_SUITE}"
  echo "[FrontRES Stage3 contract preflight] PASS suite=${CONTRACT_SUITE}"
fi

if [[ "${FRONTRES_STAGE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  joined=" ${TRAIN_CMD[*]} "
  for required in \
    " scripts/rsl_rl/train.py " \
    " --frontres_stage stage3_segment_hrl " \
    " --resume_student_checkpoint ${STAGE1_CHECKPOINT} " \
    " --is_full_resume False " \
    " --frontres_segment_cache_dir ${CACHE_DIR} " \
    " --frontres_segment_live_update_steps ${UPDATE_STEPS} " \
    " --experiment_name g1_flat_frontres_stage3_segment_hrl "
  do
    if [[ "${joined}" != *"${required}"* ]]; then
      echo "Stage 3 startup preflight failed; missing cmd fragment:${required}" >&2
      echo -n "Command: " >&2
      printf '%q ' "${TRAIN_CMD[@]}" >&2
      echo >&2
      exit 4
    fi
  done
  echo "[FrontRES Stage3 startup preflight] PASS mode=${MODE}"
  echo -n "Command: "
  printf '%q ' "${TRAIN_CMD[@]}"
  echo
  exit 0
fi

HYDRA_FULL_ERROR=1 "${TRAIN_CMD[@]}"
