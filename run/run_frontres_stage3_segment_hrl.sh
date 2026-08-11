#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run/run_frontres_stage3_segment_hrl.sh HSL_CHECKPOINT MOTION_PATH [NUM_ENVS] [MAX_ITERS] [UPDATE_STEPS] [MODE] [TRAIN_ARGS...]"
  echo
  echo "Stage 3 loads an HSL Delta SE proposal checkpoint and trains Segment Replay HRL."
  echo "MODE can be: train, sentinel, probe, storage, policy_quality_eval."
  echo "SHARD_CACHE_SIZE controls the lazy Stage 1 cache LRU size."
  echo "Evaluation is launched independently through Held-out Policy Quality, Deployment Composition, or DR Sweep."
  echo "FRONTRES_SPECIALIST_MODE selects the perturbation preset for train/eval; default rp."
  echo "FRS-TRAIN-v022 uses actual Actor LR=3e-7->1e-6 and Critic LR=1e-5 with B8/M4; shared/adaptive overrides are rejected."
  echo "Example:"
  echo "  SHARD_CACHE_SIZE=8 bash run/run_frontres_stage3_segment_hrl.sh /path/to/hsl/model.pt /path/to/motions 12000 2000 4 train"
  exit 1
fi

HSL_CHECKPOINT="$1"
MOTION_PATH="$2"
FEMR_ROOT="${FEMR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FEMR_DATA_ROOT="${FEMR_DATA_ROOT:-$(dirname "${FEMR_ROOT}")}"
NUM_ENVS="${3:-12000}"
MAX_ITERS="${4:-2000}"
UPDATE_STEPS="${5:-4}"
MODE="${6:-train}"
EXTRA_TRAIN_ARGS=("${@:7}")
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-FEMR}"
RUN_NAME="${RUN_NAME:-FEMR_STAGE3_SEGMENT_HRL}"
CACHE_DIR="${CACHE_DIR:-${FEMR_DATA_ROOT}/AMASS_G1Segment}"
CHECKPOINT_INTERVAL="${FRONTRES_CHECKPOINT_INTERVAL:-1}"
SHARD_CACHE_SIZE="${SHARD_CACHE_SIZE:-8}"
FRONTRES_SPECIALIST_MODE="${FRONTRES_SPECIALIST_MODE:-rp}"
FRONTRES_V015_FUTURE_OFFSETS="${FRONTRES_V015_FUTURE_OFFSETS:-1,2}"
FRONTRES_V015_K_CURRICULUM="${FRONTRES_V015_K_CURRICULUM:-}"
FRONTRES_V015_RESUME_CHECKPOINT="${FRONTRES_V015_RESUME_CHECKPOINT:-}"
POLICY_QUALITY_REPEAT_COUNT="${POLICY_QUALITY_REPEAT_COUNT:-1}"
FRONTRES_SEGMENT_ACTOR_LR_INIT="${FRONTRES_SEGMENT_ACTOR_LR_INIT:-3e-7}"
FRONTRES_SEGMENT_ACTOR_LR="${FRONTRES_SEGMENT_ACTOR_LR:-1e-6}"
FRONTRES_SEGMENT_CRITIC_LR="${FRONTRES_SEGMENT_CRITIC_LR:-1e-5}"

if ! [[ "${CHECKPOINT_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FRONTRES_CHECKPOINT_INTERVAL must be a positive integer" >&2
  exit 2
fi

if [[ ("${MODE}" == "train" || "${MODE}" == "policy_quality_eval") && -z "${FRONTRES_V015_K_CURRICULUM}" ]]; then
  echo "FRS-TRAIN-v022 requires an explicit ten-field K/M/DR schedule; no hidden DR defaults are allowed" >&2
  exit 4
fi
FRONTRES_G5_S4_BOUNDED="${FRONTRES_G5_S4_BOUNDED:-0}"
CONTRACT_SUITE="${FRONTRES_STAGE3_CONTRACT_SUITE:-source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py}"
CONTRACT_PYTHON="${FRONTRES_STAGE3_CONTRACT_PYTHON:-python}"

if [[ -n "${FRONTRES_V015_RESUME_CHECKPOINT}" && ! -f "${FRONTRES_V015_RESUME_CHECKPOINT}" ]]; then
  echo "checkpoint-v17 resume checkpoint not found: ${FRONTRES_V015_RESUME_CHECKPOINT}" >&2
  exit 2
fi
if [[ -z "${FRONTRES_V015_RESUME_CHECKPOINT}" && ! -f "${HSL_CHECKPOINT}" ]]; then
  echo "HSL checkpoint not found: ${HSL_CHECKPOINT}" >&2
  exit 2
fi
if [[ "${MODE}" == "train" && "${NUM_ENVS}" != "64" ]]; then
  echo "FRS-TRAIN-v022 K8/B8/M4 campaign requires NUM_ENVS=64" >&2
  exit 4
fi

if [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]]; then
  for arg in "${EXTRA_TRAIN_ARGS[@]}"; do
    case "${arg}" in
      --resume|--resume=*|--resume_student_checkpoint|--resume_student_checkpoint=*|--is_full_resume|--is_full_resume=*)
        echo "v020 Stage 3 forbids legacy resume arguments: ${arg}" >&2
        exit 4
        ;;
    esac
  done
fi

if [[ "${FRONTRES_G5_S4_BOUNDED}" == "1" ]]; then
  if [[ "${MODE}" != "train" || "${NUM_ENVS}" != "64" || "${MAX_ITERS}" != "1" || "${UPDATE_STEPS}" != "1" ]]; then
    echo "G5-S4 bounded Stage 3 requires train mode, 64 envs, 1 iteration, and 1 update" >&2
    exit 4
  fi
elif [[ "${FRONTRES_G5_S4_BOUNDED}" != "0" ]]; then
  echo "FRONTRES_G5_S4_BOUNDED must be 0 or 1" >&2
  exit 4
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
  policy_quality_eval)
    if [[ "${NUM_ENVS}" != "16" ]]; then
      echo "EVAL-v004 K8/K16 M4 policy quality requires NUM_ENVS=16" >&2
      exit 4
    fi
    if ! [[ "${POLICY_QUALITY_REPEAT_COUNT}" =~ ^[1-9][0-9]*$ ]] || (( POLICY_QUALITY_REPEAT_COUNT > 16 )); then
      echo "POLICY_QUALITY_REPEAT_COUNT must be an integer from 1 to 16" >&2
      exit 4
    fi
    required_quality_vars=(
      POLICY_QUALITY_MANIFEST
      POLICY_QUALITY_POLICY_CHECKPOINT
      POLICY_QUALITY_RESULT
    )
    for name in "${required_quality_vars[@]}"; do
      if [[ -z "${!name:-}" ]]; then
        echo "EVAL-v004 policy quality requires ${name}" >&2
        exit 4
      fi
    done
    MODE_ARGS=(
      --frontres_policy_quality_eval_only
      --frontres_policy_quality_manifest "${POLICY_QUALITY_MANIFEST}"
      --frontres_policy_quality_hsl_checkpoint "${HSL_CHECKPOINT}"
      --frontres_policy_quality_policy_checkpoint "${POLICY_QUALITY_POLICY_CHECKPOINT}"
      --frontres_policy_quality_result "${POLICY_QUALITY_RESULT}"
      --frontres_policy_quality_repeat_count "${POLICY_QUALITY_REPEAT_COUNT}"
    )
    ;;
  single_update|update_loop)
  echo "FRS-PPO-v010 rejects retired optimizer-writing Stage 3 mode: ${MODE}" >&2
    exit 4
    ;;
  offline_eval|sequence_eval|policy_quality_q2d_eval)
    echo "FRS-EVAL-v004 rejects legacy v002/v006/quartet local evaluation mode: ${MODE}" >&2
    exit 4
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
  --frontres_stage stage3_segment_hrl
  --frontres_specialist_mode "${FRONTRES_SPECIALIST_MODE}"
  --frontres_segment_cache_dir "${CACHE_DIR}"
  --frontres_segment_shard_cache_size "${SHARD_CACHE_SIZE}"
  --frontres_segment_live_update_steps "${UPDATE_STEPS}"
  --frontres_segment_ppo_schedule fixed
  --frontres_segment_actor_lr_init "${FRONTRES_SEGMENT_ACTOR_LR_INIT}"
  --frontres_segment_actor_lr "${FRONTRES_SEGMENT_ACTOR_LR}"
  --frontres_segment_critic_lr "${FRONTRES_SEGMENT_CRITIC_LR}"
  --frontres_v015_future_offsets "${FRONTRES_V015_FUTURE_OFFSETS}"
  --frontres_segment_k_curriculum "${FRONTRES_V015_K_CURRICULUM}"
)

if [[ -n "${FRONTRES_V015_RESUME_CHECKPOINT}" ]]; then
  TRAIN_CMD+=(--frontres_v015_resume_checkpoint "${FRONTRES_V015_RESUME_CHECKPOINT}")
else
  TRAIN_CMD+=(--frontres_v015_hsl_initializer_checkpoint "${HSL_CHECKPOINT}")
fi

if [[ "${MODE}" == "train" ]]; then
  TRAIN_CMD+=(--frontres_checkpoint_interval "${CHECKPOINT_INTERVAL}")
fi

if [[ "${FRONTRES_G5_S4_BOUNDED}" == "1" ]]; then
  TRAIN_CMD+=(
    --frontres_formal_runtime_audit
    --frontres_segment_critic_warmup_iterations 200
    --frontres_segment_actor_warmup_iterations 500
  )
fi

if [[ ${#MODE_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${MODE_ARGS[@]}")
fi

if [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${EXTRA_TRAIN_ARGS[@]}")
fi

if [[ "${FRONTRES_STAGE3_RUN_CONTRACTS:-0}" == "1" ]]; then
  echo "[FrontRES Stage3 contract preflight] START suite=${CONTRACT_SUITE} python=${CONTRACT_PYTHON}"
  "${CONTRACT_PYTHON}" "${CONTRACT_SUITE}"
  echo "[FrontRES Stage3 contract preflight] PASS suite=${CONTRACT_SUITE}"
fi

if [[ ! -d "${MOTION_PATH}" ]]; then
  echo "motion path not found: ${MOTION_PATH}" >&2
  exit 2
fi
if [[ ! -d "${CACHE_DIR}" ]]; then
  echo "Segment cache directory not found: ${CACHE_DIR}" >&2
  exit 2
fi

if [[ "${FRONTRES_STAGE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  joined=" ${TRAIN_CMD[*]} "
  identity_fragment=" --frontres_v015_hsl_initializer_checkpoint ${HSL_CHECKPOINT} "
  if [[ -n "${FRONTRES_V015_RESUME_CHECKPOINT}" ]]; then
    identity_fragment=" --frontres_v015_resume_checkpoint ${FRONTRES_V015_RESUME_CHECKPOINT} "
  fi
  for required in \
    " scripts/rsl_rl/train.py " \
    " --frontres_stage stage3_segment_hrl " \
    " --frontres_specialist_mode ${FRONTRES_SPECIALIST_MODE} " \
    "${identity_fragment}" \
    " --frontres_v015_future_offsets ${FRONTRES_V015_FUTURE_OFFSETS} " \
    " --frontres_segment_k_curriculum ${FRONTRES_V015_K_CURRICULUM} " \
    " --frontres_segment_actor_lr_init ${FRONTRES_SEGMENT_ACTOR_LR_INIT} " \
    " --frontres_segment_actor_lr ${FRONTRES_SEGMENT_ACTOR_LR} " \
    " --frontres_segment_critic_lr ${FRONTRES_SEGMENT_CRITIC_LR} " \
    " --frontres_segment_cache_dir ${CACHE_DIR} " \
    " --frontres_segment_shard_cache_size ${SHARD_CACHE_SIZE} " \
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
