#!/usr/bin/env bash
set -euo pipefail

FEMR_ROOT="${FEMR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${FRONTRES_PYTHON:-${FEMR_ROOT}/frontres/bin/python}"
GMT_CHECKPOINT="${FRONTRES_GMT_CHECKPOINT:-}"

if [[ -z "${GMT_CHECKPOINT}" ]]; then
  for candidate in \
    "${FEMR_ROOT}/model/model_27000.pt" \
    "/home/yuxuancheng/MOSAIC/model/model_27000.pt" \
    "/hdd1/cyx/MOSAIC/model/model_27000.pt"; do
    if [[ -f "${candidate}" ]]; then
      GMT_CHECKPOINT="${candidate}"
      break
    fi
  done
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "FrontRES Python runtime not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ -z "${GMT_CHECKPOINT}" || ! -f "${GMT_CHECKPOINT}" ]]; then
  echo "Frozen GMT checkpoint not found; set FRONTRES_GMT_CHECKPOINT" >&2
  exit 2
fi

export FRONTRES_GMT_CHECKPOINT="${GMT_CHECKPOINT}"
export PYTHONPATH="${FEMR_ROOT}/source/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" \
  "${FEMR_ROOT}/source/rsl_rl/rsl_rl/tests/frontres_v022_global_simplified_formal_test.py"
