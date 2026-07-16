#!/usr/bin/env python3
"""S1 contracts for the paired Style/Physics/Repair gain owner."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "frontres_gain_contract_target",
    ROOT / "rsl_rl" / "frontres" / "frontres_gain.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_style_gain_is_clean_paired_and_sign_correct() -> None:
    clean = torch.zeros(2, 3, 1, 3)
    noisy = clean.clone()
    repaired = clean.clone()
    noisy[..., 0] = 0.20
    repaired[..., 0] = 0.05
    clean_q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    noisy_q = torch.stack(
        (torch.cos(torch.tensor(0.10)), torch.tensor(0.0), torch.sin(torch.tensor(0.10)), torch.tensor(0.0))
    ).repeat(2, 1)
    repaired_q = torch.stack(
        (torch.cos(torch.tensor(0.025)), torch.tensor(0.0), torch.sin(torch.tensor(0.025)), torch.tensor(0.0))
    ).repeat(2, 1)
    result = MODULE.compute_paired_style_gain(
        clean,
        repaired,
        noisy,
        config=MODULE.FrontRESSegmentGainConfig(mpjpe_scale=1.0, velocity_scale=1.0, acceleration_scale=1.0),
        clean_root_quaternions=clean_q,
        repaired_root_quaternions=repaired_q,
        noisy_root_quaternions=noisy_q,
    )
    torch.testing.assert_close(result["mpjpe"], torch.full((2,), 0.15))
    torch.testing.assert_close(result["root_orientation"], torch.full((2,), 0.15), atol=1e-5, rtol=1e-5)
    assert bool((result["style"] > 0.0).all())


def test_physics_gain_uses_paired_success_and_survival_only() -> None:
    result = MODULE.compute_paired_physics_gain(
        torch.tensor([True, False]),
        torch.tensor([False, False]),
        torch.tensor([4.0, 1.0]),
        torch.tensor([2.0, 1.0]),
        config=MODULE.FrontRESSegmentGainConfig(),
        effective_horizon_k=4,
    )
    torch.testing.assert_close(result["success"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(result["survival"], torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(result["physics"], torch.tensor([0.75, 0.0]))


def test_physics_gain_includes_optional_zmp_and_contact_components() -> None:
    result = MODULE.compute_paired_physics_gain(
        torch.tensor([True, False]),
        torch.tensor([False, False]),
        torch.tensor([4.0, 1.0]),
        torch.tensor([2.0, 1.0]),
        repaired_zmp_margin=torch.tensor([0.3, -0.1]),
        noisy_zmp_margin=torch.tensor([0.1, -0.1]),
        repaired_contact=torch.tensor([1.0, 0.5]),
        noisy_contact=torch.tensor([0.5, 0.5]),
        config=MODULE.FrontRESSegmentGainConfig(),
        effective_horizon_k=4,
    )
    torch.testing.assert_close(result["zmp"], torch.tensor([0.2, 0.0]))
    torch.testing.assert_close(result["contact"], torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(result["physics"], torch.tensor([2.2 / 4.0, 0.0]))


def test_repair_cost_is_full_six_d_and_has_temporal_term() -> None:
    actions = torch.zeros(3, 2, 6)
    actions[0, :, 0] = 1.0
    actions[1, :, 0] = 2.0
    actions[2, :, 0] = 2.0
    result = MODULE.compute_repair_cost(actions, config=MODULE.FrontRESSegmentGainConfig())
    torch.testing.assert_close(result["norm"], torch.tensor([5.0 / 3.0, 5.0 / 3.0]))
    torch.testing.assert_close(result["temporal"], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(result["cost"], torch.tensor([13.0 / 12.0, 13.0 / 12.0]))


def test_repair_cost_respects_mixed_k_and_done_mask() -> None:
    actions = torch.zeros(4, 2, 6)
    actions[:, 0, 0] = torch.tensor([1.0, 3.0, 100.0, 100.0])
    actions[:, 1, 0] = torch.tensor([0.0, 1.0, 3.0, 100.0])
    valid_steps = torch.tensor(
        [
            [True, True],
            [True, True],
            [False, True],
            [False, False],
        ]
    )
    result = MODULE.compute_repair_cost(
        actions,
        valid_steps=valid_steps,
        config=MODULE.FrontRESSegmentGainConfig(),
    )
    torch.testing.assert_close(result["norm"], torch.tensor([2.0, 4.0 / 3.0]))
    torch.testing.assert_close(result["temporal"], torch.tensor([2.0, 1.5]))
    torch.testing.assert_close(result["cost"], torch.tensor([2.0, 17.0 / 12.0]))


def test_repair_cost_reports_clean_noop_separately() -> None:
    clean_actions = torch.zeros(2, 1, 6)
    result = MODULE.compute_repair_cost(
        torch.ones(2, 1, 6),
        clean_action_steps=clean_actions,
        config=MODULE.FrontRESSegmentGainConfig(),
    )
    torch.testing.assert_close(result["clean_norm"], torch.tensor([0.0]))
    torch.testing.assert_close(result["clean_cost"], torch.tensor([0.0]))


def test_segment_gain_preserves_mixed_k_pairing_and_row_permutation() -> None:
    clean = torch.zeros(2, 3, 1, 3)
    repaired = clean.clone()
    noisy = clean.clone()
    repaired[0, :, 0, 0] = 0.1
    noisy[0, :, 0, 0] = 0.2
    repaired[1, :, 0, 0] = 0.1
    noisy[1, :, 0, 0] = 0.4
    actions = torch.zeros(3, 2, 6)
    actions[:, 0, 0] = torch.tensor([1.0, 2.0, 100.0])
    actions[:, 1, 0] = torch.tensor([3.0, 2.0, 1.0])
    action_mask = torch.tensor(
        [
            [True, True],
            [True, True],
            [False, True],
        ]
    )
    temporal_mask = action_mask.clone()
    config = MODULE.FrontRESSegmentGainConfig(
        mpjpe_scale=1.0,
        velocity_scale=1.0,
        acceleration_scale=1.0,
        repair_weight=0.15,
    )

    def compute(order: torch.Tensor):
        return MODULE.compute_segment_gain(
            clean_positions=clean[order],
            repaired_positions=repaired[order],
            noisy_positions=noisy[order],
            repaired_success=torch.tensor([True, True])[order],
            noisy_success=torch.tensor([False, True])[order],
            repaired_survival=torch.tensor([1.0, 1.0])[order],
            noisy_survival=torch.tensor([0.0, 1.0])[order],
            action_steps=actions[:, order],
            action_step_mask=action_mask[:, order],
            temporal_mask=temporal_mask[order],
            config=config,
            effective_horizon_k=1,
        )

    result = compute(torch.tensor([0, 1]))
    permuted = compute(torch.tensor([1, 0]))
    torch.testing.assert_close(result.style_gain, torch.tensor([0.1 / 3.0, 0.3 / 3.0]))
    torch.testing.assert_close(result.repair_norm, torch.tensor([1.5, 2.0]))
    torch.testing.assert_close(result.repair_temporal_change, torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(result.gain_total, torch.tensor([0.8458333, -0.1250]))
    torch.testing.assert_close(permuted.gain_total, result.gain_total[torch.tensor([1, 0])])


def test_missing_style_or_temporal_evidence_is_not_zero() -> None:
    clean = torch.zeros(1, 1, 1, 3)
    result = MODULE.compute_segment_gain(
        clean_positions=None,
        repaired_positions=None,
        noisy_positions=None,
        repaired_success=torch.tensor([True]),
        noisy_success=torch.tensor([False]),
        repaired_survival=torch.tensor([1.0]),
        noisy_survival=torch.tensor([0.5]),
        action_steps=torch.ones(1, 1, 6),
        config=MODULE.FrontRESSegmentGainConfig(),
        effective_horizon_k=1,
    )
    assert torch.isnan(result.style_gain).all()
    assert torch.isnan(result.repair_temporal_change).all()
    assert torch.isnan(result.gain_total).all()


def test_step_gain_uses_current_full6d_action_and_shared_signs() -> None:
    clean = torch.zeros(2, 1, 3)
    noisy = clean.clone()
    repaired = clean.clone()
    noisy[..., 0] = 0.2
    repaired[..., 0] = 0.1
    result = MODULE.compute_segment_gain_step(
        clean_position=clean,
        repaired_position=repaired,
        noisy_position=noisy,
        previous_clean_position=None,
        previous_repaired_position=None,
        previous_noisy_position=None,
        previous_previous_clean_position=None,
        previous_previous_repaired_position=None,
        previous_previous_noisy_position=None,
        clean_root_quaternion=None,
        repaired_root_quaternion=None,
        noisy_root_quaternion=None,
        repaired_success=torch.tensor([True, True]),
        noisy_success=torch.tensor([False, True]),
        repaired_survival=torch.tensor([1.0, 1.0]),
        noisy_survival=torch.tensor([0.0, 1.0]),
        action=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        previous_action=None,
        config=MODULE.FrontRESSegmentGainConfig(
            mpjpe_scale=1.0,
            repair_norm_scale=1.0,
            repair_temporal_scale=1.0,
        ),
        effective_horizon_k=2,
    )
    assert bool((result.style_gain > 0.0).all())
    assert bool((result.physics_gain >= 0.0).all())
    assert bool((result.repair_norm > 0.0).all())
    assert bool(torch.isfinite(result.gain_total).all())


def test_survival_unit_and_k_aggregation_probe() -> None:
    """Prove raw reporting, K quality, and per-step-to-final aggregation."""

    config = MODULE.FrontRESSegmentGainConfig()
    k4 = MODULE.compute_paired_physics_gain(
        repaired_success=torch.tensor([True]),
        noisy_success=torch.tensor([True]),
        repaired_survival=torch.tensor([4.0]),
        noisy_survival=torch.tensor([2.0]),
        config=config,
        effective_horizon_k=4,
    )
    k1 = MODULE.compute_paired_physics_gain(
        repaired_success=torch.tensor([True]),
        noisy_success=torch.tensor([True]),
        repaired_survival=torch.tensor([1.0]),
        noisy_survival=torch.tensor([0.0]),
        config=config,
        effective_horizon_k=1,
    )
    k8 = MODULE.compute_paired_physics_gain(
        repaired_success=torch.tensor([True]),
        noisy_success=torch.tensor([True]),
        repaired_survival=torch.tensor([4.0]),
        noisy_survival=torch.tensor([2.0]),
        config=config,
        effective_horizon_k=8,
    )

    repaired_steps = torch.tensor([1.0, 1.0, 1.0, 1.0])
    noisy_steps = torch.tensor([1.0, 1.0, 0.0, 0.0])
    per_step = torch.stack(
        [
            MODULE.compute_paired_physics_gain(
                repaired_success=torch.tensor([True]),
                noisy_success=torch.tensor([True]),
                repaired_survival=repaired_steps[index].reshape(1),
                noisy_survival=noisy_steps[index].reshape(1),
                config=config,
                effective_horizon_k=4,
            )["survival"].reshape(())
            for index in range(4)
        ]
    )

    torch.testing.assert_close(k1["survival"], torch.tensor([1.0]))
    torch.testing.assert_close(k4["survival"], torch.tensor([0.5]))
    torch.testing.assert_close(k8["survival"], torch.tensor([0.25]))
    torch.testing.assert_close(per_step, torch.tensor([0.0, 0.0, 0.25, 0.25]))
    torch.testing.assert_close(per_step.sum().reshape(1), k4["survival"])

    print(
        "[probe survival-unit] "
        "raw_steps_repaired=4.0 raw_steps_noisy=2.0 "
        f"k1_quality_gain={float(k1['survival'][0]):.6f} "
        f"k4_quality_gain={float(k4['survival'][0]):.6f} "
        f"k8_quality_gain={float(k8['survival'][0]):.6f} "
        f"per_step_delta={per_step.tolist()} per_step_sum={float(per_step.sum()):.6f} k=4",
        flush=True,
    )


def test_missing_effective_k_does_not_fallback_to_raw_steps() -> None:
    result = MODULE.compute_paired_physics_gain(
        repaired_success=torch.tensor([True]),
        noisy_success=torch.tensor([True]),
        repaired_survival=torch.tensor([4.0]),
        noisy_survival=torch.tensor([2.0]),
        config=MODULE.FrontRESSegmentGainConfig(),
        effective_horizon_k=None,
    )
    assert torch.isnan(result["survival"]).all()
    assert not torch.equal(result["survival"], torch.tensor([2.0]))


def main() -> None:
    test_style_gain_is_clean_paired_and_sign_correct()
    test_physics_gain_uses_paired_success_and_survival_only()
    test_physics_gain_includes_optional_zmp_and_contact_components()
    test_repair_cost_is_full_six_d_and_has_temporal_term()
    test_repair_cost_respects_mixed_k_and_done_mask()
    test_repair_cost_reports_clean_noop_separately()
    test_segment_gain_preserves_mixed_k_pairing_and_row_permutation()
    test_missing_style_or_temporal_evidence_is_not_zero()
    test_step_gain_uses_current_full6d_action_and_shared_signs()
    test_survival_unit_and_k_aggregation_probe()
    test_missing_effective_k_does_not_fallback_to_raw_steps()
    print("frontres_gain_components_contract: ok")


if __name__ == "__main__":
    main()
