#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run/run_frontres_stage2_acceptance.sh STAGE1_CHECKPOINT MOTION_PATH [NUM_ENVS] [MAX_ITERS]"
  echo
  echo "Stage 2 loads a Stage 1 Delta SE proposal checkpoint and trains the masked acceptance head."
  echo "Only the Stage 1 proposal is transferred; this is not a full optimizer/runner resume."
  echo "Example:"
  echo "  bash run/run_frontres_stage2_acceptance.sh /path/to/stage1/model_warmup.pt /path/to/motions 12000 2000"
  exit 1
fi

STAGE1_CHECKPOINT="$1"
MOTION_PATH="$2"
NUM_ENVS="${3:-12000}"
MAX_ITERS="${4:-2000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-FEMR}"
RUN_NAME="${RUN_NAME:-FEMR_STAGE2_ACCEPTANCE}"

if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
  echo "Stage 1 checkpoint not found: ${STAGE1_CHECKPOINT}" >&2
  exit 2
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" scripts/rsl_rl/train.py --distributed)
else
  LAUNCH=(python scripts/rsl_rl/train.py)
fi

TRAIN_CMD=(
  "${LAUNCH[@]}"
  --task=FrontRES-Unified-Tracking-Flat-G1-v0 \
  --num_envs="${NUM_ENVS}" \
  --motion "${MOTION_PATH}" \
  --headless \
  --logger wandb \
  --log_project_name "${LOG_PROJECT_NAME}" \
  --experiment_name g1_flat_frontres_stage2_acceptance \
  --run_name "${RUN_NAME}" \
  --max_iterations "${MAX_ITERS}" \
  --supervised_warmup_iterations 0 \
  --resume_student_checkpoint "${STAGE1_CHECKPOINT}" \
  --is_full_resume False \
  --frontres_stage stage2_acceptance
)

if [[ "${FRONTRES_STAGE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  joined=" ${TRAIN_CMD[*]} "
  for required in \
    " scripts/rsl_rl/train.py " \
    " --frontres_stage stage2_acceptance " \
    " --resume_student_checkpoint ${STAGE1_CHECKPOINT} " \
    " --is_full_resume False " \
    " --experiment_name g1_flat_frontres_stage2_acceptance "
  do
    if [[ "${joined}" != *"${required}"* ]]; then
      echo "Stage 2 startup preflight failed; missing command fragment:${required}" >&2
      echo -n "Command: " >&2
      printf '%q ' "${TRAIN_CMD[@]}" >&2
      echo >&2
      exit 3
    fi
  done
  echo "[FrontRES Stage2 startup preflight] PASS"
  echo -n "Command: "
  printf '%q ' "${TRAIN_CMD[@]}"
  echo
  exit 0
fi

HYDRA_FULL_ERROR=1 "${TRAIN_CMD[@]}"
