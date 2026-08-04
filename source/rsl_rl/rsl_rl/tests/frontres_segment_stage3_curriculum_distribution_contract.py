#!/usr/bin/env python3
"""Independent TRAIN-v013 four-class DR curriculum contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "source" / "rsl_rl"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
OWNER = ROOT / "source" / "rsl_rl" / "rsl_rl" / "frontres" / "frontres_segment_warmup.py"
spec = importlib.util.spec_from_file_location("frontres_v013_dr_contract_owner", OWNER)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


SCHEDULE_TEXT = (
    "8:2:200:500:1300:lower-k8:0.5:linear-joint-v1:1300:2.381,"
    "16:3:300:300:900:lower-k16:0.6:linear-joint-v1:900:2.381,"
    "32:4:400:300:625:lower-k32:0.7:linear-joint-v1:625:2.381"
)


def test_exact_four_class_support_and_weight() -> None:
    schedule = module.require_frontres_v013_campaign_schedule(module.parse_frontres_k_stage_schedule(SCHEDULE_TEXT))
    identity = module.resolve_frontres_k_stage_identity(schedule=schedule, committed_update_iteration=1350)
    counts = [0, 0, 0, 0]
    for key in range(20_000):
        sample = module.sample_frontres_v013_dr_strength(identity, sample_key=key)
        counts[sample.class_index] += 1
        low_high = (
            (0.0, 0.25 * identity.d_cap),
            (0.25 * identity.d_cap, 0.70 * identity.d_cap),
            (0.70 * identity.d_cap, identity.d_cap),
            (identity.d_cap, min(1.10 * identity.d_cap, 2.381)),
        )[sample.class_index]
        assert low_high[0] <= sample.strength <= low_high[1]
        if sample.class_name in {"easy", "medium"}:
            assert sample.strength < low_high[1]
        if sample.class_name == "broken":
            assert sample.strength > identity.d_cap
    observed = tuple(count / 20_000 for count in counts)
    for actual, expected in zip(observed, (0.20, 0.30, 0.40, 0.10), strict=True):
        assert abs(actual - expected) < 0.015


def test_commit_only_restart_and_no_feedback_surface() -> None:
    schedule = module.require_frontres_v013_campaign_schedule(module.parse_frontres_k_stage_schedule(SCHEDULE_TEXT))
    # Failed/partial work cannot alter this pure mapping because the sole input is
    # the persisted committed-update count.
    before = module.resolve_frontres_k_stage_identity(schedule=schedule, committed_update_iteration=1999)
    switched = module.resolve_frontres_k_stage_identity(schedule=schedule, committed_update_iteration=2000)
    assert before.active_k == 8 and switched.active_k == 16
    assert switched.phase.name == "critic_only" and switched.phase.actor_loss_weight == 0.0
    assert switched.dr_progress == 0.0 and switched.d_cap == 0.6
    first = module.sample_frontres_v013_dr_strength(switched, sample_key=17)
    second = module.sample_frontres_v013_dr_strength(switched, sample_key=17)
    assert first == second
    assert set(module.sample_frontres_v013_dr_strength.__annotations__) == {"identity", "sample_key", "return"}


def main() -> None:
    test_exact_four_class_support_and_weight()
    test_commit_only_restart_and_no_feedback_surface()
    print("frontres_segment_stage3_curriculum_distribution_contract: v013 ok")


if __name__ == "__main__":
    main()
