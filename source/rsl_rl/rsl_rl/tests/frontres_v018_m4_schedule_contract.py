#!/usr/bin/env python3
"""TEST-02 contract for the all-stage exact-M4 campaign."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "source/rsl_rl"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
OWNER = SOURCE_ROOT / "rsl_rl/frontres/frontres_segment_warmup.py"
spec = importlib.util.spec_from_file_location("frontres_v018_m4_schedule_owner", OWNER)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


SCHEDULE = (
    "8:4:200:500:1300:lower-k8:0.5:linear-joint-v1:1300:2.381,"
    "16:4:300:300:900:lower-k16:0.6:linear-joint-v1:900:2.381,"
    "32:4:400:300:625:lower-k32:0.7:linear-joint-v1:625:2.381"
)


def main() -> None:
    schedule = module.require_frontres_v013_campaign_schedule(
        module.parse_frontres_k_stage_schedule(SCHEDULE, max_horizon_k=32)
    )
    assert tuple((row.horizon_k, row.attempts_m) for row in schedule) == ((8, 4), (16, 4), (32, 4))
    for iteration, expected_k in ((0, 8), (1999, 8), (2000, 16), (3500, 32)):
        identity = module.resolve_frontres_k_stage_identity(
            schedule=schedule,
            committed_update_iteration=iteration,
            max_horizon_k=32,
        )
        assert identity.active_k == expected_k and identity.active_m == 4
    old = SCHEDULE.replace("8:4:", "8:2:").replace("16:4:", "16:3:")
    try:
        module.require_frontres_v013_campaign_schedule(module.parse_frontres_k_stage_schedule(old))
    except ValueError as exc:
        assert "M4" in str(exc) or "schedule" in str(exc)
    else:
        raise AssertionError("TRAIN-v018 accepted the retired M2/M3 campaign")
    print("frontres_v018_m4_schedule_contract: K8/K16/K32 all M4", flush=True)


if __name__ == "__main__":
    main()
