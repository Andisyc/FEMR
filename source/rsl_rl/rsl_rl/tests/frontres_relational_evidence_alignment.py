"""Offline contract for the formal evidence-to-edge adapter."""

import torch

from rsl_rl.frontres.frontres_hierarchical_gain_candidate import Outcome
from rsl_rl.frontres.frontres_segment_evidence import (
    FrontRESExecutedKTrajectory,
    FrontRESRepairAttemptEvidence,
    FrontRESSealedRecoveryAwareGainBatch,
    FrontRESSegmentBaselineEvidence,
)
from rsl_rl.frontres.frontres_segment_grouped_adapter import build_frontres_v025_relational_candidate_storage


def _trajectory(outcome: Outcome) -> FrontRESExecutedKTrajectory:
    k = 4
    return FrontRESExecutedKTrajectory(
        joint_pos=torch.zeros(k, 1, 29),
        root_pos=torch.zeros(k, 1, 3),
        root_quat=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * k),
        key_body_pos=torch.zeros(k, 1, 1, 3),
        root_lin_vel=torch.zeros(k, 1, 3),
        root_ang_vel=torch.zeros(k, 1, 3),
        foot_pos=torch.zeros(k, 1, 2, 3),
        contact=torch.ones(k, 1, 2),
        zmp_margin=torch.ones(k, 1),
        survival=torch.ones(k, 1),
        valid_mask=torch.ones(k, 1, dtype=torch.bool),
        relational_outcome=outcome,
    )


def main() -> None:
    stable = Outcome()
    outcomes = (stable, Outcome(capture_margin=0.0, stable_hold_steps=0), Outcome(expected_support_no_load=0.2), Outcome(survival_ok=False, survival_failure_duration=0.2))
    baseline = FrontRESSegmentBaselineEvidence(
        transaction_id="tx",
        policy_snapshot_id="policy",
        scenario_id="scenario",
        noisy_segment_hash="hash",
        x_t_identity="xt",
        source_index=0,
        segment_id=0,
        horizon_k=4,
        expected_support=torch.ones(4, 1, 2),
        clean=_trajectory(stable),
        noisy=_trajectory(stable),
    )
    attempts = tuple(
        FrontRESRepairAttemptEvidence(
            transaction_id="tx",
            policy_snapshot_id="policy",
            scenario_id="scenario",
            noisy_segment_hash="hash",
            x_t_identity="xt",
            source_index=0,
            segment_id=0,
            trial_index=index,
            horizon_k=4,
            policy_observation=torch.zeros(2),
            policy_privileged_observation=torch.zeros(2),
            policy_action=torch.zeros(6),
            policy_log_prob=torch.zeros(()),
            policy_value=torch.zeros(()),
            policy_mean=torch.zeros(6),
            policy_sigma=torch.ones(6),
            repair=_trajectory(outcome),
        )
        for index, outcome in enumerate(outcomes)
    )
    evidence = FrontRESSealedRecoveryAwareGainBatch((baseline,), attempts, active_m=4)
    assert evidence.relational_preference_edges() == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    storage = build_frontres_v025_relational_candidate_storage(
        evidence,
        motion_ids=("motion",) * 4,
        start_frames=torch.zeros(4, dtype=torch.long),
        intent_q29_provenance="deployment_noisy_q29",
        intent_q29_source="deploy_tail",
    )
    assert storage.preference_edges == evidence.relational_preference_edges()
    assert not hasattr(storage, "returns")
    assert not hasattr(storage, "advantages")
    assert not hasattr(storage, "old_values")
    missing = attempts[:-1] + (FrontRESRepairAttemptEvidence(
        transaction_id="tx",
        policy_snapshot_id="policy",
        scenario_id="scenario",
        noisy_segment_hash="hash",
        x_t_identity="xt",
        source_index=0,
        segment_id=0,
        trial_index=3,
        horizon_k=4,
        policy_observation=torch.zeros(2),
        policy_privileged_observation=torch.zeros(2),
        policy_action=torch.zeros(6),
        policy_log_prob=torch.zeros(()),
        policy_value=torch.zeros(()),
        policy_mean=torch.zeros(6),
        policy_sigma=torch.ones(6),
        repair=_trajectory(Outcome()).__class__(
            **{**_trajectory(Outcome()).__dict__, "relational_outcome": None}
        ),
    ),)
    missing_evidence = FrontRESSealedRecoveryAwareGainBatch((baseline,), missing, active_m=4)
    try:
        missing_evidence.relational_preference_edges()
    except ValueError:
        pass
    else:
        raise AssertionError("missing relational Outcome must fail closed")
    print("frontres_relational_evidence_alignment: OBJECTIVE-ALIGNED")


if __name__ == "__main__":
    main()
