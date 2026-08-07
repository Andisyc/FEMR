#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$(dirname "${ROOT}")"
GPU="${CUDA_VISIBLE_DEVICES:-7}"

HSL="${ROOT}/g1_flat_frontres_stage1_hsl/2026-08-04_18-14-12_V017_HSL_V2_FULL/model_warmup.pt"
MANIFEST="${ROOT}/note/testing/manifests/frontres_v017_policy_quality_k16_v1.json"
MOTIONS="${DATA_ROOT}/AMASS_G1NPZ_Final"
CACHE="${DATA_ROOT}/AMASS_G1Segment"
MODEL3500_SHA="e550324786e35210c9ec944148407c0b93f4287ce2aa9e82eec12fcb9c54b0af"

SCHEDULE='8:2:200:500:1300:lower-k8:0.5:linear-joint-v1:1300:2.381,16:3:300:300:900:lower-k16:0.6:linear-joint-v1:900:2.381,32:4:400:300:625:lower-k32:0.7:linear-joint-v1:625:2.381'

MODEL3500="$({
  find "${ROOT}/g1_flat_frontres_stage3_segment_hrl" \
    -type f -name model_3500.pt -exec sha256sum {} \;
} | awk -v expected="${MODEL3500_SHA}" '$1 == expected {print $2; exit}')"

if [[ -z "${MODEL3500}" ]]; then
  echo "[K16-TRAJECTORY] cannot locate the evaluated model_3500.pt" >&2
  exit 2
fi

MODEL_DIR="$(dirname "${MODEL3500}")"

for required in "${HSL}" "${MANIFEST}" "${MOTIONS}" "${CACHE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[K16-TRAJECTORY] missing required artifact: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${ROOT}/.runtime_tmp"
echo "[K16-TRAJECTORY] model_dir=${MODEL_DIR}"

for iteration in 2300 2600 3000; do
  checkpoint="${MODEL_DIR}/model_${iteration}.pt"
  result="${ROOT}/v017_model${iteration}_policy_quality_k16_gpu${GPU}.json"
  log="${ROOT}/v017_model${iteration}_policy_quality_k16_gpu${GPU}.log"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "[K16-TRAJECTORY] missing checkpoint: ${checkpoint}" >&2
    exit 2
  fi

  echo "[K16-TRAJECTORY] START iteration=${iteration}"

  CUDA_VISIBLE_DEVICES="${GPU}" \
  FEMR_ROOT="${ROOT}" \
  FEMR_DATA_ROOT="${DATA_ROOT}" \
  FEMR_LOG_ROOT="${ROOT}" \
  FRONTRES_TMPDIR="${ROOT}/.runtime_tmp" \
  CACHE_DIR="${CACHE}" \
  RUN_NAME="V017_MODEL${iteration}_POLICY_QUALITY_K16" \
  POLICY_QUALITY_MANIFEST="${MANIFEST}" \
  POLICY_QUALITY_POLICY_CHECKPOINT="${checkpoint}" \
  POLICY_QUALITY_RESULT="${result}" \
  FRONTRES_V015_K_CURRICULUM="${SCHEDULE}" \
  bash "${ROOT}/run/run_frontres_stage3_segment_hrl.sh" \
    "${HSL}" "${MOTIONS}" 12 0 1 policy_quality_eval \
    >"${log}" 2>&1

  python - "${result}" "${iteration}" <<'PY'
import json
import sys

path, iteration = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as file:
    report = json.load(file)

assert report["schema_version"] == "frontres-v017-policy-quality-report-v1"
assert report["gain_contract_id"] == "FRS-GAIN-v007"
assert report["evaluation_contract_id"] == "FRS-EVAL-v004"
assert report["horizon_k"] == 16
assert report["attempts_per_segment"] == 3
assert len(report["transactions"]) == 4

print(
    f"[K16-TRAJECTORY] PASS iteration={iteration} "
    f"transactions={len(report['transactions'])} result={path}"
)
PY
done

echo "[K16-TRAJECTORY] COMPLETE"
