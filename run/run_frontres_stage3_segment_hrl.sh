#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run/run_frontres_stage3_segment_hrl.sh HSL_CHECKPOINT MOTION_PATH [NUM_ENVS] [MAX_ITERS] [UPDATE_STEPS] [MODE] [TRAIN_ARGS...]"
  echo
  echo "Stage 3 loads an HSL Delta SE proposal checkpoint and trains Segment Replay HRL."
echo "MODE can be: train, sentinel, probe, storage, single_update, update_loop, offline_eval, sequence_eval, policy_quality_eval, policy_quality_q2d_eval."
  echo "SHARD_CACHE_SIZE controls the lazy Stage 1 cache LRU size."
 echo "offline_eval loads the checkpoint, samples NUM_ENVS indexed segments, runs OFFLINE_EVAL_STEPS rollout steps, and exits."
 echo "sequence_eval loads the checkpoint, evaluates OFFLINE_EVAL_SEQUENCES unique motions from frame 0 to sampled segment starts, and exits."
 echo "OFFLINE_EVAL_MAX_PREROLL_STEPS caps sampled segment starts for smoke tests; set 0 for unbounded full evaluation."
 echo "FRONTRES_SPECIALIST_MODE selects the perturbation preset for train/eval; default rp."
  echo "Append --frontres_segment_ppo_schedule adaptive --frontres_segment_ppo_lr 1e-6 to test adaptive Segment PPO trust-region control."
  echo "Example:"
  echo "  SHARD_CACHE_SIZE=8 bash run/run_frontres_stage3_segment_hrl.sh /path/to/hsl/model.pt /path/to/motions 12000 2000 4 train"
  echo "  bash run/run_frontres_stage3_segment_hrl.sh /path/to/hsl/model.pt /path/to/motions 1 1 1 update_loop"
  exit 1
fi

HSL_CHECKPOINT="$1"
MOTION_PATH="$2"
NUM_ENVS="${3:-12000}"
MAX_ITERS="${4:-2000}"
UPDATE_STEPS="${5:-4}"
MODE="${6:-train}"
EXTRA_TRAIN_ARGS=("${@:7}")
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-FEMR}"
RUN_NAME="${RUN_NAME:-FEMR_STAGE3_SEGMENT_HRL}"
CACHE_DIR="${CACHE_DIR:-/hdd1/cyx/AMASS_G1Segment}"
SHARD_CACHE_SIZE="${SHARD_CACHE_SIZE:-8}"
PERIODIC_EVAL_ENABLED="${PERIODIC_EVAL_ENABLED:-0}"
PERIODIC_EVAL_INTERVAL="${PERIODIC_EVAL_INTERVAL:-100}"
FRONTRES_SPECIALIST_MODE="${FRONTRES_SPECIALIST_MODE:-rp}"
FRONTRES_V015_FUTURE_OFFSETS="${FRONTRES_V015_FUTURE_OFFSETS:-1,2}"
FRONTRES_G5_S4_BOUNDED="${FRONTRES_G5_S4_BOUNDED:-0}"
CONTRACT_SUITE="${FRONTRES_STAGE3_CONTRACT_SUITE:-source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py}"
CONTRACT_PYTHON="${FRONTRES_STAGE3_CONTRACT_PYTHON:-python}"

if [[ ! -f "${HSL_CHECKPOINT}" ]]; then
  echo "HSL checkpoint not found: ${HSL_CHECKPOINT}" >&2
  exit 2
fi

if [[ "${PERIODIC_EVAL_ENABLED}" != "0" ]]; then
  echo "v015 Stage 3 forbids legacy periodic evaluation" >&2
  exit 4
fi

if [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]]; then
  for arg in "${EXTRA_TRAIN_ARGS[@]}"; do
    case "${arg}" in
      --resume|--resume=*|--resume_student_checkpoint|--resume_student_checkpoint=*|--is_full_resume|--is_full_resume=*|--frontres_segment_periodic_eval_enabled|--frontres_segment_periodic_eval_enabled=*)
        echo "v015 Stage 3 forbids resume and legacy periodic-evaluation arguments: ${arg}" >&2
        exit 4
        ;;
    esac
  done
fi

if [[ "${FRONTRES_G5_S4_BOUNDED}" == "1" ]]; then
  if [[ "${MODE}" != "train" || "${NUM_ENVS}" != "8" || "${MAX_ITERS}" != "1" || "${UPDATE_STEPS}" != "1" ]]; then
    echo "G5-S4 bounded Stage 3 requires train mode, 8 envs, 1 iteration, and 1 update" >&2
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
  single_update)
    MODE_ARGS=(--frontres_segment_live_single_update_only)
    ;;
  update_loop)
    MODE_ARGS=(--frontres_segment_live_update_loop_only)
    ;;
  offline_eval)
    MODE_ARGS=(
      --frontres_segment_offline_eval_only
      --frontres_segment_offline_eval_segments "${OFFLINE_EVAL_SEGMENTS:-${NUM_ENVS}}"
      --frontres_segment_offline_eval_steps "${OFFLINE_EVAL_STEPS:-500}"
    )
    ;;
  sequence_eval)
    MODE_ARGS=(
      --frontres_segment_sequence_offline_eval_only
      --frontres_segment_sequence_eval_sequences "${OFFLINE_EVAL_SEQUENCES:-10}"
      --frontres_segment_sequence_eval_max_preroll_steps "${OFFLINE_EVAL_MAX_PREROLL_STEPS:-2000}"
      --frontres_segment_offline_eval_steps "${OFFLINE_EVAL_STEPS:-500}"
    )
    ;;
  policy_quality_eval)
    : "${POLICY_QUALITY_MANIFEST:?Set POLICY_QUALITY_MANIFEST for policy_quality_eval}"
    : "${POLICY_QUALITY_HSL_CHECKPOINT:?Set POLICY_QUALITY_HSL_CHECKPOINT for policy_quality_eval}"
    : "${POLICY_QUALITY_POLICY_CHECKPOINT:?Set POLICY_QUALITY_POLICY_CHECKPOINT for policy_quality_eval}"
    : "${POLICY_QUALITY_RESULT:?Set POLICY_QUALITY_RESULT for policy_quality_eval}"
    # Quality selection is manifest-owned; never restore checkpoint sampler,
    # optimizer, or warmup state into the runner used for evaluation.
    MODE_ARGS=(
      --frontres_policy_quality_eval_only
      --frontres_policy_quality_manifest "${POLICY_QUALITY_MANIFEST}"
      --frontres_policy_quality_hsl_checkpoint "${POLICY_QUALITY_HSL_CHECKPOINT}"
      --frontres_policy_quality_policy_checkpoint "${POLICY_QUALITY_POLICY_CHECKPOINT}"
      --frontres_policy_quality_result "${POLICY_QUALITY_RESULT}"
    )
    ;;
  policy_quality_q2d_eval)
    : "${POLICY_QUALITY_MANIFEST:?Set POLICY_QUALITY_MANIFEST for policy_quality_q2d_eval}"
    : "${POLICY_QUALITY_HSL_CHECKPOINT:?Set POLICY_QUALITY_HSL_CHECKPOINT for policy_quality_q2d_eval}"
    : "${POLICY_QUALITY_POLICY_CHECKPOINT:?Set POLICY_QUALITY_POLICY_CHECKPOINT for policy_quality_q2d_eval}"
    : "${POLICY_QUALITY_Q2D_RESULT:?Set POLICY_QUALITY_Q2D_RESULT for policy_quality_q2d_eval}"
    MODE_ARGS=(
      --frontres_policy_quality_q2d_eval_only
      --frontres_policy_quality_manifest "${POLICY_QUALITY_MANIFEST}"
      --frontres_policy_quality_hsl_checkpoint "${POLICY_QUALITY_HSL_CHECKPOINT}"
      --frontres_policy_quality_policy_checkpoint "${POLICY_QUALITY_POLICY_CHECKPOINT}"
      --frontres_policy_quality_q2d_result "${POLICY_QUALITY_Q2D_RESULT}"
    )
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
  --frontres_v015_future_offsets "${FRONTRES_V015_FUTURE_OFFSETS}"
  --frontres_v015_hsl_initializer_checkpoint "${HSL_CHECKPOINT}"
)

if [[ "${FRONTRES_G5_S4_BOUNDED}" == "1" ]]; then
  TRAIN_CMD+=(
    --frontres_checkpoint_interval 1
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

if [[ "${FRONTRES_STAGE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  joined=" ${TRAIN_CMD[*]} "
  for required in \
    " scripts/rsl_rl/train.py " \
    " --frontres_stage stage3_segment_hrl " \
    " --frontres_specialist_mode ${FRONTRES_SPECIALIST_MODE} " \
    " --frontres_v015_hsl_initializer_checkpoint ${HSL_CHECKPOINT} " \
    " --frontres_v015_future_offsets ${FRONTRES_V015_FUTURE_OFFSETS} " \
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
