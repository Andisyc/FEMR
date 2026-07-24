#!/usr/bin/env bash
set -euo pipefail

ROOT="${FEMR_ROOT:-/hdd1/cyx/FEMR}"
SCRIPT_PATH="${ROOT}/run/run_v015_p4_policy_quality_closure.sh"
DRIVER_LOG="${ROOT}/v015_p4_policy_quality_closure_driver_gpu${CUDA_VISIBLE_DEVICES:-3}.log"

if [[ "${1:-}" != "--worker" ]]; then
  cd "${ROOT}"
  nohup bash "${SCRIPT_PATH}" --worker >"${DRIVER_LOG}" 2>&1 </dev/null &
  pid=$!
  echo "[P4-CLOSURE] submitted pid=${pid}"
  echo "[P4-CLOSURE] driver_log=${DRIVER_LOG}"
  echo "[P4-CLOSURE] follow: tail -f ${DRIVER_LOG}"
  exit 0
fi

cd "${ROOT}"

LOCK_FILE="${ROOT}/.v015_p4_policy_quality_closure.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[P4-CLOSURE] another closure worker is already running" >&2
  exit 6
fi

export PATH="${MOSAIC_BIN:-/hdd1/cyx/miniconda3/envs/mosaic/bin}:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CACHE_DIR="${CACHE_DIR:-/hdd1/cyx/AMASS_G1Segment}"
export FEMR_LOG_ROOT="${FEMR_LOG_ROOT:-${ROOT}}"
export PERIODIC_EVAL_ENABLED=0

MODEL201="${P4_MODEL201:-${ROOT}/g1_flat_frontres_stage3_segment_hrl/2026-07-24_15-56-00_G5_S4_BOUND_V015/model_201.pt}"
HSL="${P4_HSL_CHECKPOINT:-${ROOT}/g1_flat_frontres_stage1_hsl/2026-07-21_17-06-12_G2_S4_BOUND_HSL/model_warmup.pt}"
MOTIONS="${P4_MOTION_ROOT:-/hdd1/cyx/AMASS_G1NPZ_Final}"
MANIFEST="${P4_MANIFEST:-${ROOT}/note/testing/manifests/frontres_v015_policy_quality_heldout_v1.json}"

BASE_JSON="${ROOT}/v015_p4_quality_model201.json"
TRAIN_LOG="${ROOT}/v015_p4_actor_ramp_block50_gpu${CUDA_VISIBLE_DEVICES}.log"
AFTER_JSON="${ROOT}/v015_p4_quality_model251.json"
AFTER_LOG="${ROOT}/v015_p4_quality_model251_gpu${CUDA_VISIBLE_DEVICES}.log"
PATH_FILE="${ROOT}/v015_p4_model251_path.txt"
ARCHIVE="${ROOT}/v015_p4_policy_quality_closure_gpu${CUDA_VISIBLE_DEVICES}.tar.gz"

for required in "${MODEL201}" "${HSL}" "${MANIFEST}" "${BASE_JSON}"; do
  if [[ ! -s "${required}" ]]; then
    echo "[P4-CLOSURE] missing required artifact: ${required}" >&2
    exit 2
  fi
done
if [[ ! -d "${MOTIONS}" || ! -d "${CACHE_DIR}" ]]; then
  echo "[P4-CLOSURE] motion/cache directory missing" >&2
  echo "motion_root=${MOTIONS}" >&2
  echo "cache_dir=${CACHE_DIR}" >&2
  exit 2
fi

existing_train="$(pgrep -af '[s]cripts/rsl_rl/train.py.*model_201.pt' || true)"
if [[ -n "${existing_train}" ]]; then
  echo "[P4-CLOSURE] an earlier model_201 training process is still active:" >&2
  echo "${existing_train}" >&2
  exit 7
fi

echo "[P4-CLOSURE] TRAIN_START model=${MODEL201} updates=50 gpu=${CUDA_VISIBLE_DEVICES}"
RUN_NAME=P4_ACTOR_RAMP_BLOCK50 \
FRONTRES_G5_S4_BOUNDED=0 \
FRONTRES_V015_FUTURE_OFFSETS=1,2 \
FRONTRES_V015_K_CURRICULUM=8:200:500:0 \
FRONTRES_V015_RESUME_CHECKPOINT="${MODEL201}" \
bash run/run_frontres_stage3_segment_hrl.sh \
  "${MODEL201}" "${MOTIONS}" 8 50 1 train \
  --frontres_segment_ppo_schedule adaptive \
  --frontres_segment_ppo_lr 1e-6 \
  --frontres_checkpoint_interval 50 \
  --frontres_formal_runtime_audit \
  --frontres_segment_critic_warmup_iterations 200 \
  --frontres_segment_actor_warmup_iterations 500 \
  >"${TRAIN_LOG}" 2>&1

POLICY_AFTER="$(sed -n 's/^  save.path: //p' "${TRAIN_LOG}" | tail -n1)"
if [[ -z "${POLICY_AFTER}" || ! -s "${POLICY_AFTER}" ]]; then
  echo "[P4-CLOSURE] model_251 checkpoint was not produced" >&2
  tail -n 120 "${TRAIN_LOG}" >&2
  exit 3
fi
grep -q 'save.iteration: 251' "${TRAIN_LOG}"
grep -q '"optimizer_step_delta":1' "${TRAIN_LOG}"
grep -q '"update_count":1' "${TRAIN_LOG}"
printf '%s\n' "${POLICY_AFTER}" >"${PATH_FILE}"
echo "[P4-CLOSURE] TRAIN_COMPLETE checkpoint=${POLICY_AFTER}"

echo "[P4-CLOSURE] QUALITY_START manifest=${MANIFEST}"
RUN_NAME=P4_QUALITY_MODEL251 \
FRONTRES_V015_FUTURE_OFFSETS=1,2 \
FRONTRES_V015_K_CURRICULUM=8:200:500:0 \
POLICY_QUALITY_MANIFEST="${MANIFEST}" \
POLICY_QUALITY_HSL_CHECKPOINT="${HSL}" \
POLICY_QUALITY_POLICY_CHECKPOINT="${POLICY_AFTER}" \
POLICY_QUALITY_RESULT="${AFTER_JSON}" \
bash run/run_frontres_stage3_segment_hrl.sh \
  "${HSL}" "${MOTIONS}" 8 0 1 policy_quality_eval \
  --frontres_segment_ppo_schedule adaptive \
  --frontres_segment_ppo_lr 1e-6 \
  >"${AFTER_LOG}" 2>&1

if [[ ! -s "${AFTER_JSON}" ]]; then
  echo "[P4-CLOSURE] quality report was not produced" >&2
  tail -n 120 "${AFTER_LOG}" >&2
  exit 4
fi
if grep -q 'Traceback' "${TRAIN_LOG}" "${AFTER_LOG}"; then
  echo "[P4-CLOSURE] traceback detected" >&2
  exit 5
fi
python -c 'import json,sys; json.load(open(sys.argv[1])); json.load(open(sys.argv[2]))' \
  "${BASE_JSON}" "${AFTER_JSON}"
echo "[P4-CLOSURE] QUALITY_COMPLETE report=${AFTER_JSON}"

tar -czf "${ARCHIVE}" \
  -C "${ROOT}" \
  "$(basename "${BASE_JSON}")" \
  "$(basename "${TRAIN_LOG}")" \
  "$(basename "${AFTER_JSON}")" \
  "$(basename "${AFTER_LOG}")" \
  "$(basename "${PATH_FILE}")" \
  "note/testing/manifests/$(basename "${MANIFEST}")"

echo "[P4-CLOSURE] COMPLETE"
echo "[P4-CLOSURE] checkpoint=${POLICY_AFTER}"
echo "[P4-CLOSURE] artifact=${ARCHIVE}"
