#!/usr/bin/env bash
set -euo pipefail

AMASS_ROOT="${1:-${AMASS_ROOT:-/hdd1/cyx/AMASS_G1NPZ_Final}}"
CACHE_DIR="${2:-${CACHE_DIR:-/hdd1/cyx/AMASS_G1Segment}}"
HORIZON_K="${HORIZON_K:-4}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_MOTIONS="${MAX_MOTIONS:-all}"
MAX_SEGMENTS="${MAX_SEGMENTS:-all}"
EXPECT_MODE="${EXPECT_MODE:-hrl_curriculum_bank}"
DEEP_SHARD_READ="${DEEP_SHARD_READ:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[FrontRES Stage1 Completion] amass_root=${AMASS_ROOT}"
echo "[FrontRES Stage1 Completion] cache_dir=${CACHE_DIR}"
echo "[FrontRES Stage1 Completion] horizon_k=${HORIZON_K} frame_stride=${FRAME_STRIDE} max_motions=${MAX_MOTIONS} max_segments=${MAX_SEGMENTS}"
echo "[FrontRES Stage1 Completion] expect_mode=${EXPECT_MODE} deep_shard_read=${DEEP_SHARD_READ}"

"${PYTHON_BIN}" scripts/rsl_rl/check_frontres_stage1_segment_cache_completion.py \
  --amass-root "${AMASS_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --horizon-k "${HORIZON_K}" \
  --frame-stride "${FRAME_STRIDE}" \
  --max-motions "${MAX_MOTIONS}" \
  --max-segments "${MAX_SEGMENTS}" \
  --expect-mode "${EXPECT_MODE}" \
  --deep-shard-read "${DEEP_SHARD_READ}"
