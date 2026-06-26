#!/usr/bin/env bash
set -euo pipefail

echo "Legacy ablation entrypoint: Stage 2 authority actor-critic is retired from active FEMR." >&2
echo "Use: bash run/run_frontres_stage2_acceptance.sh STAGE1_CHECKPOINT MOTION_PATH [NUM_ENVS] [MAX_ITERS]" >&2
exit 2
