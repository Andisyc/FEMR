#!/usr/bin/env bash
set -euo pipefail

cd /hdd1/cyx/FEMR

GPU="${GPU:-3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SOURCE_REFERENCE_NPZ="${SOURCE_REFERENCE_NPZ:-/hdd1/cyx/AMASS_G1NPZ_Final/KIT/674/amass_g1_wash_head01_poses.npz}"
FRONTRES_CHECKPOINT="${FRONTRES_CHECKPOINT:-/hdd1/cyx/FEMR/g1_flat_frontres_stage3_segment_hrl/2026-07-28_14-06-10_V015_GAIN_V006_POSTFIX_TO_MODEL2000/model_2000.pt}"
GMT_CHECKPOINT="${GMT_CHECKPOINT:-/hdd1/cyx/MOSAIC/model/model_27000.pt}"
CARRIER_NPZ="${CARRIER_NPZ:-/hdd1/cyx/FEMR/v015_model2000_demo_${RUN_TAG}_carrier.npz}"
REPORT_PATH="${REPORT_PATH:-/hdd1/cyx/FEMR/v015_model2000_demo_${RUN_TAG}.json}"
LOG_PATH="${LOG_PATH:-/hdd1/cyx/FEMR/v015_model2000_demo_${RUN_TAG}_gpu${GPU}.log}"
CORRUPTION_ID="${CORRUPTION_ID:-model2000-demo-local-rp-${RUN_TAG}}"
CORRUPTION_SEED="${CORRUPTION_SEED:-42}"
CORRUPTION_PARAMETERS='{"source":"pre_materialized_deployment_npz","scale":1.25,"roll_std":0.06,"pitch_std":0.08,"root_body_index":0}'

for path in "$SOURCE_REFERENCE_NPZ" "$FRONTRES_CHECKPOINT" "$GMT_CHECKPOINT"; do
  [[ -f "$path" ]] || { echo "[V015-DEMO] missing required artifact: $path" >&2; exit 2; }
done
for path in "$CARRIER_NPZ" "$REPORT_PATH" "$LOG_PATH"; do
  [[ ! -e "$path" ]] || { echo "[V015-DEMO] refusing existing output: $path" >&2; exit 2; }
done

SOURCE_REFERENCE_NPZ="$SOURCE_REFERENCE_NPZ" \
CARRIER_NPZ="$CARRIER_NPZ" \
CORRUPTION_ID="$CORRUPTION_ID" \
CORRUPTION_SEED="$CORRUPTION_SEED" \
CORRUPTION_PARAMETERS="$CORRUPTION_PARAMETERS" \
./frontres/bin/python - <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import sys

owner_path = Path("source/rsl_rl/rsl_rl/runners/frontres_segment_sequence_eval.py").resolve()
spec = importlib.util.spec_from_file_location("frontres_v015_demo_materializer", owner_path)
owner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = owner
spec.loader.exec_module(owner)
parameters = json.loads(os.environ["CORRUPTION_PARAMETERS"])
protocol = owner.build_frontres_v015_persistent_corruption_protocol(
    corruption_id=os.environ["CORRUPTION_ID"],
    family="local_rp",
    seed=int(os.environ["CORRUPTION_SEED"]),
    parameters=parameters,
)
carrier = owner.materialize_frontres_v015_deployment_carrier(
    source_path=os.environ["SOURCE_REFERENCE_NPZ"],
    output_path=os.environ["CARRIER_NPZ"],
    corruption_protocol=protocol,
)
print(
    "[V015-DEMO-CARRIER] "
    f"source_hash={carrier.source_reference_file_hash} "
    f"carrier_hash={carrier.carrier_file_hash} "
    f"protocol_hash={carrier.corruption_protocol.protocol_hash} "
    f"delta_se3={carrier.materialized_delta_se3}",
    flush=True,
)
PY

echo "[V015-DEMO] gpu=$GPU source=$SOURCE_REFERENCE_NPZ carrier=$CARRIER_NPZ report=$REPORT_PATH"
CUDA_VISIBLE_DEVICES="$GPU" HYDRA_FULL_ERROR=1 \
bash ~/IsaacLab_mosaic/isaaclab.sh -p scripts/rsl_rl/frontres_v015_deployment_composition.py \
  --frontres_checkpoint "$FRONTRES_CHECKPOINT" \
  --gmt_checkpoint "$GMT_CHECKPOINT" \
  --source_reference_npz "$SOURCE_REFERENCE_NPZ" \
  --reference_npz "$CARRIER_NPZ" \
  --report_path "$REPORT_PATH" \
  --future_offsets 1,2 \
  --corruption_id "$CORRUPTION_ID" \
  --corruption_family local_rp \
  --corruption_seed "$CORRUPTION_SEED" \
  --corruption_parameters_json "$CORRUPTION_PARAMETERS" \
  --num_envs 8 \
  --device cuda:0 \
  --headless 2>&1 | tee "$LOG_PATH"

echo "[V015-DEMO-COMPLETE] report=$REPORT_PATH log=$LOG_PATH"
