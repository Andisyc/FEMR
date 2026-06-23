#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run/run_frontres_stage2_authority.sh STAGE1_CHECKPOINT MOTION_PATH [NUM_ENVS] [MAX_ITERS]"
  echo
  echo "Stage 2 loads a Stage 1 proposal checkpoint and trains the authority rho actor-critic."
  echo "Use --is_full_resume False for Stage1 -> Stage2 transfer."
  echo "Example:"
  echo "  bash run/run_frontres_stage2_authority.sh /path/to/stage1/model_800.pt /path/to/motions 12000 2000"
  exit 1
fi

STAGE1_CHECKPOINT="$1"
MOTION_PATH="$2"
NUM_ENVS="${3:-12000}"
MAX_ITERS="${4:-2000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-MOSAIC_FrontRES}"
RUN_NAME="${RUN_NAME:-FrontRES_STAGE2_AUTHORITY}"

if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
  echo "Stage 1 checkpoint not found: ${STAGE1_CHECKPOINT}" >&2
  exit 2
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" scripts/rsl_rl/train.py --distributed)
else
  LAUNCH=(python scripts/rsl_rl/train.py)
fi

HYDRA_FULL_ERROR=1 "${LAUNCH[@]}" \
  --task=FrontRES-Unified-Tracking-Flat-G1-v0 \
  --num_envs="${NUM_ENVS}" \
  --motion "${MOTION_PATH}" \
  --headless \
  --logger wandb \
  --log_project_name "${LOG_PROJECT_NAME}" \
  --run_name "${RUN_NAME}" \
  --max_iterations "${MAX_ITERS}" \
  --supervised_warmup_iterations 0 \
  --resume_student_checkpoint "${STAGE1_CHECKPOINT}" \
  --is_full_resume False \
  experiment_name=g1_flat_frontres_stage2_authority \
  algorithm.frontres_training_objective=hsl_hybrid \
  algorithm.lambda_supervised=0.0 \
  algorithm.lambda_supervised_min=0.0 \
  algorithm.lambda_supervised_decay=1.0 \
  algorithm.frontres_authority_actor_critic_enabled=True \
  algorithm.frontres_authority_actor_loss_weight=1.0 \
  algorithm.frontres_authority_critic_loss_weight=1.0 \
  algorithm.frontres_authority_actor_warmup_iterations=200 \
  algorithm.frontres_authority_actor_ramp_iterations=200 \
  algorithm.frontres_authority_return_horizon=8 \
  algorithm.frontres_structured_joint_rl_enabled=False \
  algorithm.frontres_structured_joint_rl_weight=0.0 \
  algorithm.frontres_acceptance_preference_weight=0.0 \
  algorithm.frontres_state_alpha_weight=0.0 \
  policy.frontres_authority_actor_critic=True \
  frontres_perturbation_temporal_mode=burst \
  frontres_perturbation_burst_min_steps=4 \
  frontres_perturbation_burst_max_steps=8 \
  frontres_authority_return_horizon=8 \
  critic_warmup_iterations=200
