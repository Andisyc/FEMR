"""Deterministic S1 contracts for FRS-GAIN-v008 and sealed baseline reuse."""

from __future__ import annotations

from pathlib import Path

import torch

from rsl_rl.frontres.frontres_gain import (
    FrontRESRecoveryAwareGainConfig,
    compute_recovery_aware_gain,
)
from rsl_rl.frontres.frontres_segment_evidence import (
    FrontRESExecutedKTrajectory,
    FrontRESRepairAttemptEvidence,
    FrontRESSealedRecoveryAwareGainBatch,
    FrontRESSegmentBaselineEvidence,
)


def _trajectory(
    *,
    joint: float,
    root_x: float = 0.0,
    root_z: float = 1.0,
    foot_x: float = 0.0,
    contact: tuple[float, float] = (1.0, 1.0),
    zmp: float = 0.01,
    survive: tuple[float, float] = (1.0, 1.0),
) -> FrontRESExecutedKTrajectory:
    k_steps = 2
    root_pos = torch.tensor([[[root_x, 0.0, root_z]], [[root_x, 0.0, root_z]]])
    root_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]])
    foot_pos = torch.zeros(k_steps, 1, 2, 3)
    foot_pos[..., 0] = foot_x
    contact_steps = torch.tensor(contact).view(1, 1, 2).expand(k_steps, 1, 2).clone()
    valid = torch.ones(k_steps, 1, dtype=torch.bool)
    zmp_steps = torch.full((k_steps, 1), zmp)
    loaded = contact_steps.bool().any(dim=-1)
    zmp_steps = torch.where(loaded, zmp_steps, torch.full_like(zmp_steps, float("nan")))
    result = FrontRESExecutedKTrajectory(
        joint_pos=torch.full((k_steps, 1, 29), joint),
        root_pos=root_pos,
        root_quat=root_quat,
        key_body_pos=torch.zeros(k_steps, 1, 3, 3),
        root_lin_vel=torch.zeros(k_steps, 1, 3),
        root_ang_vel=torch.zeros(k_steps, 1, 3),
        foot_pos=foot_pos,
        contact=contact_steps,
        zmp_margin=zmp_steps,
        survival=torch.tensor(survive).view(k_steps, 1),
        valid_mask=valid,
    )
    result.validate()
    return result


def _baseline(source: int) -> FrontRESSegmentBaselineEvidence:
    result = FrontRESSegmentBaselineEvidence(
        transaction_id="tx-v007",
        policy_snapshot_id="pi-old-v007",
        scenario_id=f"scenario-{source}",
        noisy_segment_hash=f"hash-{source}",
        x_t_identity=f"x-{source}",
        source_index=source,
        segment_id=10 + source,
        horizon_k=2,
        expected_support=torch.ones(2, 1, 2),
        clean=_trajectory(joint=0.0),
        noisy=_trajectory(joint=0.087, foot_x=0.03, zmp=-0.02),
    )
    result.validate()
    return result


def _attempt(source: int, trial: int, *, joint: float, action: float = 0.0) -> FrontRESRepairAttemptEvidence:
    result = FrontRESRepairAttemptEvidence(
        transaction_id="tx-v007",
        policy_snapshot_id="pi-old-v007",
        scenario_id=f"scenario-{source}",
        noisy_segment_hash=f"hash-{source}",
        x_t_identity=f"x-{source}",
        source_index=source,
        segment_id=10 + source,
        trial_index=trial,
        horizon_k=2,
        policy_observation=torch.zeros(158),
        policy_privileged_observation=torch.zeros(289),
        policy_action=torch.full((6,), action),
        policy_log_prob=torch.tensor(0.0),
        policy_value=torch.tensor(0.0),
        policy_mean=torch.zeros(6),
        policy_sigma=torch.ones(6),
        repair=_trajectory(joint=joint, foot_x=0.01, zmp=0.0),
    )
    result.validate()
    return result


def _batch() -> FrontRESSealedRecoveryAwareGainBatch:
    result = FrontRESSealedRecoveryAwareGainBatch(
        baselines=tuple(_baseline(source) for source in range(8)),
        attempts=tuple(
            _attempt(source, trial, joint=0.0435 if trial == 0 else 0.087)
            for source in range(8)
            for trial in range(2)
        ),
        active_m=2,
    )
    result.validate()
    return result


def test_clean_anchor_pressure_and_row_permutation() -> None:
    batch = _batch()
    config = FrontRESRecoveryAwareGainConfig(beta=0.02)
    result = compute_recovery_aware_gain(batch.to_gain_input(), config=config)
    assert tuple(result.gain_total.shape) == (16,)
    assert bool(torch.isfinite(result.gain_total).all())
    assert bool((result.intent_gain[0::2] > result.intent_gain[1::2]).all())
    expected_weighted = 0.5 * (
        result.physics_remaining_noisy.square() - result.physics_remaining_repaired.square()
    )
    assert torch.allclose(result.weighted_physics_gain, expected_weighted, atol=1.0e-6)
    assert torch.allclose(result.gain_total, result.cost_free_score - 0.02 * result.repair_cost)

    permuted = FrontRESSealedRecoveryAwareGainBatch(
        baselines=tuple(reversed(batch.baselines)),
        attempts=tuple(reversed(batch.attempts)),
        active_m=2,
    )
    permuted_result = compute_recovery_aware_gain(permuted.to_gain_input(), config=config)
    assert torch.allclose(result.gain_total, permuted_result.gain_total)


def test_gain_evidence_accepts_b2_m4_while_source_identity_fails_closed() -> None:
    """The shared Gain value object supports evaluation B2; training owns B8."""

    batch = FrontRESSealedRecoveryAwareGainBatch(
        baselines=tuple(_baseline(source) for source in range(2)),
        attempts=tuple(
            _attempt(source, trial, joint=0.0435 + 0.005 * trial, action=0.001 * trial)
            for source in range(2)
            for trial in range(4)
        ),
        active_m=4,
    )
    batch.validate()
    gain_input = batch.to_gain_input()
    assert tuple(gain_input.repair_actions.shape) == (8, 6)
    assert tuple(gain_input.clean_joint_pos.shape[:2]) == (2, 8)

    for baselines, required_text in (
        ((_baseline(0), _baseline(0)), "distinct contiguous source"),
        ((_baseline(0), _baseline(2)), "distinct contiguous source"),
    ):
        bad = FrontRESSealedRecoveryAwareGainBatch(
            baselines=baselines,
            attempts=batch.attempts,
            active_m=4,
        )
        try:
            bad.validate()
        except ValueError as exc:
            assert required_text in str(exc)
        else:
            raise AssertionError("duplicate/missing Gain source identity did not fail closed")

    formal_source = (
        Path(__file__).resolve().parents[1] / "runners" / "frontres_segment_formal_transaction.py"
    ).read_text(encoding="utf-8")
    assert 'request.plan.selected_segment_count != 8' in formal_source
    assert 'formal update rejects mixed-M or non-B8 transactions' in formal_source


def test_baseline_alias_and_execution_count_fail_closed() -> None:
    batch = _batch()
    gain_input = batch.to_gain_input()
    assert torch.equal(gain_input.clean_joint_pos[:, 0], gain_input.clean_joint_pos[:, 1])
    assert torch.equal(gain_input.noisy_joint_pos[:, 0], gain_input.noisy_joint_pos[:, 1])
    bad = FrontRESSegmentBaselineEvidence(
        **{**_baseline(0).__dict__, "clean_execution_count": 2}
    )
    try:
        bad.validate()
    except ValueError as exc:
        assert "exactly once" in str(exc)
    else:
        raise AssertionError("v017 baseline execution count must fail closed")


def test_loaded_support_zmp_na_and_malformed_payload() -> None:
    batch = _batch()
    gain_input = batch.to_gain_input()
    no_load_contact = torch.zeros_like(gain_input.repaired_contact)
    no_load_zmp = torch.full_like(gain_input.repaired_zmp_margin, float("nan"))
    no_load = type(gain_input)(
        **{
            **gain_input.__dict__,
            "repaired_contact": no_load_contact,
            "repaired_zmp_margin": no_load_zmp,
        }
    )
    result = compute_recovery_aware_gain(no_load, config=FrontRESRecoveryAwareGainConfig())
    assert bool(torch.isfinite(result.gain_total).all())
    assert bool(torch.isnan(result.physics_channel_repaired[:, 2]).all())
    assert bool((result.physics_channel_repaired[:, 0] > 0).all())

    malformed = type(gain_input)(
        **{
            **gain_input.__dict__,
            "repaired_contact": no_load_contact,
            "repaired_zmp_margin": torch.zeros_like(gain_input.repaired_zmp_margin),
        }
    )
    try:
        compute_recovery_aware_gain(malformed, config=FrontRESRecoveryAwareGainConfig())
    except ValueError as exc:
        assert "ZMP" in str(exc)
    else:
        raise AssertionError("v017 finite no-load ZMP must fail closed")


def test_flight_support_drift_is_na_without_erasing_physics_score() -> None:
    gain_input = _batch().to_gain_input()
    expected = torch.zeros_like(gain_input.expected_support)
    expected[:, 0::2] = 1.0
    contact = expected.clone()
    zmp = torch.full_like(gain_input.clean_zmp_margin, float("nan"))
    zmp[:, 0::2] = 0.01
    mixed = type(gain_input)(
        **{
            **gain_input.__dict__,
            "expected_support": expected,
            "clean_contact": contact,
            "noisy_contact": contact,
            "repaired_contact": contact,
            "clean_zmp_margin": zmp,
            "noisy_zmp_margin": zmp,
            "repaired_zmp_margin": zmp,
        }
    )
    result = compute_recovery_aware_gain(mixed, config=FrontRESRecoveryAwareGainConfig())
    assert bool(torch.isfinite(result.gain_total).all())
    assert bool(torch.isfinite(result.physics_channel_repaired[0::2, 1]).all())
    assert bool(torch.isnan(result.physics_channel_repaired[1::2, 1]).all())
    assert bool(torch.isnan(result.physics_channel_repaired[1::2, 2]).all())
    assert bool(torch.isfinite(result.physics_remaining_repaired).all())

    row_count = int(result.gain_total.numel())
    permutation = torch.arange(row_count - 1, -1, -1)
    permuted = type(mixed)(
        **{
            name: value.index_select(1, permutation)
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == row_count
            else value.index_select(0, permutation)
            if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[0] == row_count
            else value
            for name, value in mixed.__dict__.items()
        }
    )
    permuted_result = compute_recovery_aware_gain(permuted, config=FrontRESRecoveryAwareGainConfig())
    torch.testing.assert_close(permuted_result.gain_total, result.gain_total.index_select(0, permutation))


def test_cost_breaks_equal_recovery_without_changing_channels() -> None:
    batch = _batch()
    attempts = list(batch.attempts)
    attempts[1] = _attempt(0, 1, joint=0.0435, action=0.01)
    costly = FrontRESSealedRecoveryAwareGainBatch(
        baselines=batch.baselines,
        attempts=tuple(attempts),
        active_m=2,
    )
    result = compute_recovery_aware_gain(costly.to_gain_input(), config=FrontRESRecoveryAwareGainConfig())
    assert torch.allclose(result.cost_free_score[0], result.cost_free_score[1])
    assert result.gain_total[0] > result.gain_total[1]


def main() -> None:
    test_clean_anchor_pressure_and_row_permutation()
    test_gain_evidence_accepts_b2_m4_while_source_identity_fails_closed()
    test_baseline_alias_and_execution_count_fail_closed()
    test_loaded_support_zmp_na_and_malformed_payload()
    test_flight_support_drift_is_na_without_erasing_physics_score()
    test_cost_breaks_equal_recovery_without_changing_channels()
    print("[T-v007-gain] Clean anchor, pressure, cost, N/A and immutable baseline pass", flush=True)


if __name__ == "__main__":
    main()
