"""Offline contract for v025 relational training diagnostics projection."""

from __future__ import annotations

import math
import pathlib
import sys
import types
from types import SimpleNamespace

_RUNNERS_DIR = pathlib.Path(__file__).resolve().parents[1] / "runners"
_runners_package = types.ModuleType("rsl_rl.runners")
_runners_package.__path__ = [str(_RUNNERS_DIR)]
sys.modules.setdefault("rsl_rl.runners", _runners_package)

from rsl_rl.frontres.frontres_relational_evaluation import FrontRESRelationalEvaluationReport
from rsl_rl.frontres.frontres_hierarchical_gain_candidate import Outcome
from rsl_rl.runners.frontres_segment_training_telemetry import (
    build_frontres_transaction_telemetry,
)


def _report(transaction_id: str):
    scenario_ids = tuple(f"scenario-{index // 4}" for index in range(32))
    hashes = tuple(f"hash-{index // 4}" for index in range(32))
    return FrontRESRelationalEvaluationReport(
        transaction_id=transaction_id,
        policy_row_count=32,
        scenario_ids=scenario_ids,
        noisy_segment_hashes=hashes,
        source_statuses=("READY",) * 8,
        comparable_pair_count_by_row=(1,) * 32,
        preference_edges=((0, 1),),
        outcomes=(Outcome(),) * 32,
    )


def main() -> None:
    transaction_id = "diagnostics-contract"
    diagnostics = {
        "scalar_target_id": "none",
        "v009_relational_reports": (_report(transaction_id),),
        "preference_edges": ((0, 1),),
        "gradient_clip_max_norm": 0.5,
        "actor_gradient_pre_clip_norm": 1.25,
        "actor_gradient_post_clip_norm": 0.5,
        "actor_gradient_clip_coefficient": 0.4,
        "actor_gradient_nonzero_parameter_count": 7,
        "actor_parameter_delta_l2": 0.001,
        "action_l2_mean": 0.12,
        "action_l2_max": 0.25,
        "action_nonzero_fraction": 1.0,
        "active_k": 8,
        "active_m": 4,
        "selected_segment_count": 8,
        "role_row_count": 64,
        "actor_learning_rate": 3e-7,
        "training_iteration": 1,
        "curriculum_fingerprint": "curriculum",
        "k_stage_index": 0,
        "k_stage_iteration": 1,
        "warmup_phase": "low_dr_joint_init",
        "warmup_phase_iteration": 1,
        "actor_loss_weight": 1.0,
        "dr_stage_fingerprint": "dr",
        "dr_progress": 0.1,
        "d_cap": 0.5,
        "dr_class_by_segment": ("easy",),
        "dr_strength_by_segment": (0.1,),
        "edge_count": 1,
        "outer_replay": {},
    }
    result = SimpleNamespace(
        transaction_id=transaction_id,
        policy_snapshot_id="snapshot",
        policy_attempt_count=32,
        optimizer_step_delta=1,
        update_invocation_count=1,
        diagnostics=diagnostics,
    )
    ppo = SimpleNamespace(
        actor_credit=__import__("torch").tensor((1.0, -1.0) + (0.0,) * 30),
        edge_count=1,
        valid_count=2,
        status="READY",
    )
    telemetry = build_frontres_transaction_telemetry(result, ppo=ppo)
    for name, expected in {
        "actor_gradient_pre_clip_norm": 1.25,
        "actor_gradient_post_clip_norm": 0.5,
        "actor_gradient_clip_coefficient": 0.4,
        "actor_parameter_delta_l2": 0.001,
        "action_l2_mean": 0.12,
        "action_l2_max": 0.25,
        "action_nonzero_fraction": 1.0,
    }.items():
        assert math.isclose(float(telemetry[name]), expected)
    assert telemetry["actor_gradient_nonzero_parameter_count"] == 7
    assert telemetry["critic_gradient_post_clip_norm"] == 0.0
    assert telemetry["outcome_schema_id"] == "frs-gain-v009-outcome-v1"
    assert len(telemetry["relational_outcomes"]) == 32
    first_outcome = telemetry["relational_outcomes"][0]
    assert first_outcome["survival_ok"] is True
    assert first_outcome["expected_support_no_load"] == 0.0
    assert first_outcome["zmp_applicable"] is True
    assert first_outcome["intent_error"] == (0.10, 0.10)
    assert first_outcome["repair_cost"] == 0.10
    print("frontres_relational_training_diagnostics_alignment: OBJECTIVE-ALIGNED")


if __name__ == "__main__":
    main()
