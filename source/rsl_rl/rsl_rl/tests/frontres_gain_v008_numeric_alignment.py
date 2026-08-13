#!/usr/bin/env python3
"""S1 numeric alignment test for the production FRS-GAIN-v008 boundary.

The fixture uses four named Repair rows with hand-checkable normalized
remaining problems.  It does not invoke a simulator or reconstruct Gain in a
test helper.
"""

from __future__ import annotations

from dataclasses import fields, replace
import math

import torch

from rsl_rl.frontres.frontres_gain import (
    FrontRESRecoveryAwareGainConfig,
    FrontRESRecoveryAwareGainInput,
    compute_recovery_aware_gain,
)


K_STEPS = 10
ROW_COUNT = 4
ATOL = 2.0e-5
RTOL = 1.0e-5


def _semantic_fixture() -> FrontRESRecoveryAwareGainInput:
    """Build four rows: recover, unchanged, deteriorate, and cost-only."""

    clean_joint = torch.zeros(K_STEPS, ROW_COUNT, 29, dtype=torch.float64)
    clean_root_pos = torch.zeros(K_STEPS, ROW_COUNT, 3, dtype=torch.float64)
    clean_root_pos[..., 2] = 1.0
    clean_root_quat = torch.zeros(K_STEPS, ROW_COUNT, 4, dtype=torch.float64)
    clean_root_quat[..., 0] = 1.0

    clean_key_body = torch.tensor(
        ((0.20, 0.00, 1.20), (-0.20, 0.00, 1.00), (0.00, 0.00, 1.50)),
        dtype=torch.float64,
    ).view(1, 1, 3, 3).expand(K_STEPS, ROW_COUNT, 3, 3).clone()
    clean_lin_vel = torch.zeros(K_STEPS, ROW_COUNT, 3, dtype=torch.float64)
    clean_ang_vel = torch.zeros(K_STEPS, ROW_COUNT, 3, dtype=torch.float64)
    clean_foot_pos = torch.zeros(K_STEPS, ROW_COUNT, 2, 3, dtype=torch.float64)
    clean_foot_pos[..., 0, 0] = -0.10
    clean_foot_pos[..., 1, 0] = 0.10
    expected_support = torch.ones(K_STEPS, ROW_COUNT, 2, dtype=torch.float64)
    clean_contact = expected_support.clone()
    clean_zmp = torch.full((K_STEPS, ROW_COUNT), 0.01, dtype=torch.float64)
    clean_survival = torch.ones(K_STEPS, ROW_COUNT, dtype=torch.float64)
    valid = torch.ones(K_STEPS, ROW_COUNT, dtype=torch.bool)

    def role_with_one_unit_problem(bad_rows: tuple[int, ...]) -> dict[str, torch.Tensor]:
        joint = clean_joint.clone()
        root_pos = clean_root_pos.clone()
        root_quat = clean_root_quat.clone()
        key_body = clean_key_body.clone()
        lin_vel = clean_lin_vel.clone()
        ang_vel = clean_ang_vel.clone()
        foot_pos = clean_foot_pos.clone()
        contact = clean_contact.clone()
        zmp = clean_zmp.clone()
        survival = clean_survival.clone()
        if bad_rows:
            rows = torch.tensor(bad_rows, dtype=torch.long)
            joint[:, rows] = 0.087
            root_pos[:, rows, 2] = 1.05
            root_quat[:, rows, 0] = math.cos(0.087 / 2.0)
            root_quat[:, rows, 3] = math.sin(0.087 / 2.0)
            key_body[:, rows, :, 0] += 0.10
            lin_vel[:, rows, 0] = 0.75
            ang_vel[:, rows, 2] = 2.0
            foot_pos[:, rows, :, 0] += 0.03
            contact[0, rows] = 0.0
            zmp[:, rows] = -0.02
            zmp[0, rows] = float("nan")
            survival[-1, rows] = 0.0
        return {
            "joint_pos": joint,
            "root_pos": root_pos,
            "root_quat": root_quat,
            "key_body_pos": key_body,
            "root_lin_vel": lin_vel,
            "root_ang_vel": ang_vel,
            "foot_pos": foot_pos,
            "contact": contact,
            "zmp_margin": zmp,
            "survival": survival,
        }

    noisy = role_with_one_unit_problem((0, 1))
    repaired = role_with_one_unit_problem((1, 2))
    repair_actions = torch.zeros(ROW_COUNT, 6, dtype=torch.float64)
    repair_actions[0, 0] = 0.10
    repair_actions[0, 5] = math.radians(5.0)
    repair_actions[2, 0] = 0.10
    repair_actions[3, 5] = math.radians(5.0)

    return FrontRESRecoveryAwareGainInput(
        clean_joint_pos=clean_joint,
        noisy_joint_pos=noisy["joint_pos"],
        repaired_joint_pos=repaired["joint_pos"],
        clean_root_pos=clean_root_pos,
        noisy_root_pos=noisy["root_pos"],
        repaired_root_pos=repaired["root_pos"],
        clean_root_quat=clean_root_quat,
        noisy_root_quat=noisy["root_quat"],
        repaired_root_quat=repaired["root_quat"],
        clean_key_body_pos=clean_key_body,
        noisy_key_body_pos=noisy["key_body_pos"],
        repaired_key_body_pos=repaired["key_body_pos"],
        clean_root_lin_vel=clean_lin_vel,
        noisy_root_lin_vel=noisy["root_lin_vel"],
        repaired_root_lin_vel=repaired["root_lin_vel"],
        clean_root_ang_vel=clean_ang_vel,
        noisy_root_ang_vel=noisy["root_ang_vel"],
        repaired_root_ang_vel=repaired["root_ang_vel"],
        clean_foot_pos=clean_foot_pos,
        noisy_foot_pos=noisy["foot_pos"],
        repaired_foot_pos=repaired["foot_pos"],
        expected_support=expected_support,
        clean_contact=clean_contact,
        noisy_contact=noisy["contact"],
        repaired_contact=repaired["contact"],
        clean_zmp_margin=clean_zmp,
        noisy_zmp_margin=noisy["zmp_margin"],
        repaired_zmp_margin=repaired["zmp_margin"],
        clean_survival=clean_survival,
        noisy_survival=noisy["survival"],
        repaired_survival=repaired["survival"],
        clean_valid_mask=valid,
        noisy_valid_mask=valid.clone(),
        repaired_valid_mask=valid.clone(),
        repair_actions=repair_actions,
    )


def _compute(evidence: FrontRESRecoveryAwareGainInput):
    return compute_recovery_aware_gain(
        evidence,
        config=FrontRESRecoveryAwareGainConfig(beta=0.02, contact_timing_tolerance=0),
    )


def test_hand_calculated_gain_values() -> None:
    """S1/C1+C2/T-value: compare every scalar stage with a fixed hand oracle."""

    result = _compute(_semantic_fixture())
    one_zero = torch.tensor((1.0, 1.0, 0.0, 0.0))
    zero_one = torch.tensor((0.0, 1.0, 1.0, 0.0))
    signed = torch.tensor((1.0, 0.0, -1.0, 0.0))
    noisy_channels = torch.tensor(
        ((1.0, 1.0, 1.0, 1.0, 1.0, 1.0),) * 2
        + ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0),) * 2
    )
    repaired_channels = torch.tensor(
        (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
    )
    noisy_physics = noisy_channels[:, :4]
    repaired_physics = repaired_channels[:, :4]

    torch.testing.assert_close(result.intent_channel_noisy, noisy_channels, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(result.intent_channel_repaired, repaired_channels, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(result.physics_channel_noisy, noisy_physics, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(result.physics_channel_repaired, repaired_physics, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(result.intent_remaining_noisy, one_zero, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(result.intent_remaining_repaired, zero_one, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(result.physics_remaining_noisy, one_zero, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(result.physics_remaining_repaired, zero_one, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(result.intent_gain, signed, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(result.physics_gain, signed, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        result.recovery_pressure,
        torch.tensor((0.5, 1.0, 0.5, 0.0)),
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        result.weighted_physics_gain,
        torch.tensor((0.5, 0.0, -0.5, 0.0)),
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        result.repair_cost,
        torch.tensor((1.4142135624, 0.0, 1.0, 1.0)),
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        result.gain_total,
        torch.tensor((1.4717157288, 0.0, -1.52, -0.02)),
        atol=ATOL,
        rtol=RTOL,
    )


def test_row_permutation_preserves_semantic_identity() -> None:
    """S1/C4/T-role+T-order: permuting Repair rows must only permute results."""

    evidence = _semantic_fixture()
    permutation = torch.tensor((2, 0, 3, 1), dtype=torch.long)
    permuted_values: dict[str, torch.Tensor] = {}
    for field in fields(evidence):
        value = getattr(evidence, field.name)
        dimension = 0 if field.name == "repair_actions" else 1
        permuted_values[field.name] = value.index_select(dimension, permutation)
    result = _compute(evidence)
    permuted_result = _compute(type(evidence)(**permuted_values))
    torch.testing.assert_close(
        permuted_result.gain_total,
        result.gain_total.index_select(0, permutation),
        atol=ATOL,
        rtol=RTOL,
    )


def test_finite_zmp_without_loaded_support_fails_closed() -> None:
    """S1/C3/T-mask: invalid N/A evidence must not become numeric zero."""

    evidence = _semantic_fixture()
    malformed_zmp = evidence.repaired_zmp_margin.clone()
    malformed_zmp[0, 1] = 0.0
    try:
        _compute(replace(evidence, repaired_zmp_margin=malformed_zmp))
    except ValueError as exc:
        assert "ZMP" in str(exc)
    else:
        raise AssertionError("finite ZMP on a no-load step must fail closed")


def test_oracle_kills_physics_sign_mutation() -> None:
    """Sensitivity: the fixed oracle rejects a +Physics to -Physics mutation."""

    result = _compute(_semantic_fixture())
    sign_mutated = torch.tensor((0.4717157288, 0.0, -0.52, -0.02))
    try:
        torch.testing.assert_close(result.gain_total, sign_mutated, atol=ATOL, rtol=RTOL)
    except AssertionError:
        return
    raise AssertionError("the numeric oracle did not detect the Physics-sign mutation")


def main() -> None:
    test_hand_calculated_gain_values()
    test_row_permutation_preserves_semantic_identity()
    test_finite_zmp_without_loaded_support_fails_closed()
    test_oracle_kills_physics_sign_mutation()
    print(
        "[T-GAIN-v008-numeric-alignment] S1 C1/C2/C3/C4 value, role, mask and sensitivity pass",
        flush=True,
    )


if __name__ == "__main__":
    main()
