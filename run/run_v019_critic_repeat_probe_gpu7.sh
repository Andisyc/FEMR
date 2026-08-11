#!/usr/bin/env bash
set -euo pipefail

ROOT="${FEMR_ROOT:-/hdd0/yuxuancheng/FEMR}"
DATA_ROOT="${FEMR_DATA_ROOT:-/hdd0/yuxuancheng}"
GPU="${CUDA_VISIBLE_DEVICES:-7}"
REPEAT_COUNT="${POLICY_QUALITY_REPEAT_COUNT:-8}"
HSL="${HSL_CHECKPOINT:-${ROOT}/g1_flat_frontres_stage1_hsl/2026-08-04_18-14-12_V017_HSL_V2_FULL/model_warmup.pt}"
POLICY="${POLICY_CHECKPOINT:-${ROOT}/g1_flat_frontres_stage3_segment_hrl/2026-08-09_19-52-27_FRS_TRAIN_V019_K8_M4_FULL_COLDSTART_20260810/model_2000.pt}"
MOTIONS="${MOTION_ROOT:-${DATA_ROOT}/AMASS_G1NPZ_Final}"
CACHE="${CACHE_DIR:-${DATA_ROOT}/AMASS_G1Segment}"
MANIFEST="${POLICY_QUALITY_MANIFEST:-${ROOT}/note/testing/manifests/frontres_v019_critic_repeat_k8_m4_v1.json}"
RESULT="${POLICY_QUALITY_RESULT:-${ROOT}/log/FRS_EVAL_V004_V019_CRITIC_REPEAT_K8_M4_R${REPEAT_COUNT}_GPU${GPU}.json}"
LOG="${LOG_PATH:-${ROOT}/log/FRS_EVAL_V004_V019_CRITIC_REPEAT_K8_M4_R${REPEAT_COUNT}_GPU${GPU}.log}"
SCHEDULE="8:4:200:500:1300:lower-k8:0.5:linear-coupled-v1:700:2.381,16:4:300:300:900:lower-k16:0.6:linear-coupled-v1:600:2.381,32:4:400:300:625:lower-k32:0.7:linear-coupled-v1:700:2.381"

for artifact in "${HSL}" "${POLICY}" "${MANIFEST}"; do
  [[ -s "${artifact}" ]] || { echo "missing artifact: ${artifact}" >&2; exit 2; }
done
[[ -d "${MOTIONS}" ]] || { echo "missing motion root: ${MOTIONS}" >&2; exit 2; }
mkdir -p "${ROOT}/log"

CUDA_VISIBLE_DEVICES="${GPU}" \
FEMR_ROOT="${ROOT}" \
FEMR_DATA_ROOT="${DATA_ROOT}" \
CACHE_DIR="${CACHE}" \
LOG_PATH="${LOG}" \
RUN_NAME="FRS_EVAL_V004_V019_CRITIC_REPEAT_K8_M4" \
FRONTRES_V015_K_CURRICULUM="${SCHEDULE}" \
POLICY_QUALITY_MANIFEST="${MANIFEST}" \
POLICY_QUALITY_POLICY_CHECKPOINT="${POLICY}" \
POLICY_QUALITY_RESULT="${RESULT}" \
POLICY_QUALITY_REPEAT_COUNT="${REPEAT_COUNT}" \
bash "${ROOT}/run_stage3.sh" "${HSL}" "${MOTIONS}" 16 0 1 policy_quality_eval

echo "result=${RESULT}"
echo "log=${LOG}"
