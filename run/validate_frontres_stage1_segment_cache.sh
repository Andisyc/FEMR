#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${1:-/hdd1/cyx/AMASS_G1Segment}"
EXPECT_MODE="${EXPECT_MODE:-hrl_curriculum_bank}"
MIN_SEGMENTS="${MIN_SEGMENTS:-1}"
MIN_NOISY="${MIN_NOISY:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CMD=(
  "${PYTHON_BIN}"
  source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_validator.py
  "${CACHE_DIR}"
  --expect-mode "${EXPECT_MODE}"
  --min-segments "${MIN_SEGMENTS}"
  --min-noisy "${MIN_NOISY}"
)

if [[ "${REQUIRE_BOUNDARY_DIAGNOSTIC:-1}" == "1" ]]; then
  CMD+=(--require-boundary-diagnostic)
fi

"${CMD[@]}"
