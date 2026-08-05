#!/usr/bin/env bash
set -euo pipefail

FEMR_ROOT="${FEMR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FEMR_DATA_ROOT="${FEMR_DATA_ROOT:-$(dirname "${FEMR_ROOT}")}"
CACHE_DIR="${1:-${FEMR_DATA_ROOT}/AMASS_G1Segment}"
EXPECT_MODE="${EXPECT_MODE:-hrl_curriculum_bank}"
MIN_SEGMENTS="${MIN_SEGMENTS:-1}"
MIN_NOISY="${MIN_NOISY:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CMD=(
  env "PYTHONPATH=${PWD}/source/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"
  "${PYTHON_BIN}"
  -m rsl_rl.frontres.frontres_segment_cache_validator
  "${CACHE_DIR}"
  --expect-mode "${EXPECT_MODE}"
  --min-segments "${MIN_SEGMENTS}"
  --min-noisy "${MIN_NOISY}"
)

if [[ "${REQUIRE_BOUNDARY_DIAGNOSTIC:-1}" == "1" ]]; then
  CMD+=(--require-boundary-diagnostic)
fi

"${CMD[@]}"
