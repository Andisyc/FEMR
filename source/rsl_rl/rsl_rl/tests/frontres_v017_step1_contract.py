"""Deterministic S1/S2 contract for the recovery-aware formal data path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from frontres_contract_imports import install_frontres_contract_packages

install_frontres_contract_packages()

import torch

from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESSegmentPPOBatch,
    FrontRESSegmentPPOConfig,
    compute_frontres_segment_ppo_loss,
)
from rsl_rl.frontres.frontres_gain import FrontRESRecoveryAwareGainConfig, compute_recovery_aware_gain
from rsl_rl.frontres.frontres_local_evaluation import (
    _v017_lateral_roll,
    _v017_unplanned_contact_transitions,
    _v017_zmp_recovery_projection,
    build_frontres_v017_local_evaluation_report,
)
from rsl_rl.frontres.frontres_segment_evidence import (
    FrontRESExecutedKTrajectory,
    FrontRESRepairAttemptEvidence,
    FrontRESSealedRecoveryAwareGainBatch,
    FrontRESSegmentBaselineEvidence,
)
from rsl_rl.frontres.frontres_segment_grouped_adapter import build_frontres_v017_grouped_candidate_storage
from rsl_rl.runners import frontres_segment_one_action_k as one_action
from rsl_rl.runners.frontres_segment_physics import FrontRESV017ExecutionFrame


K = 8


def test_baseline_capture_projects_only_authoritative_segment_rows() -> None:
    authoritative = torch.tensor([1, 6])
    step_count = 0
    capture_count = 0

    class _Command:
        def begin_frontres_local_scenario_k_execution(self) -> None:
            pass

        def advance_frontres_local_scenario_k_execution(self) -> dict[str, torch.Tensor]:
            return {"valid_mask": torch.ones(8, dtype=torch.bool)}

        def end_frontres_local_scenario_k_execution(self) -> None:
            pass

    class _Env:
        device = torch.device("cpu")

        def step(self, _actions):
            nonlocal step_count
            step_count += 1
            return torch.zeros(8, 1), torch.zeros(8), torch.zeros(8, dtype=torch.bool), {}

    def capture(_runner, *, selected_rows: torch.Tensor) -> FrontRESV017ExecutionFrame:
        nonlocal capture_count
        capture_count += 1
        assert selected_rows.tolist() == authoritative.tolist()
        row_values = selected_rows.float()
        return FrontRESV017ExecutionFrame(
            joint_pos=row_values[:, None].expand(2, 29).clone(),
            root_pos=torch.zeros(2, 3),
            root_quat=torch.nn.functional.one_hot(torch.zeros(2, dtype=torch.long), num_classes=4).float(),
            key_body_pos=torch.zeros(2, 3, 3),
            root_lin_vel=torch.zeros(2, 3),
            root_ang_vel=torch.zeros(2, 3),
            foot_pos=torch.zeros(2, 2, 3),
            expected_support=torch.ones(2, 2),
            contact=torch.ones(2, 2),
            zmp_margin=row_values.clone(),
        )

    command = _Command()
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        env=_Env(),
        alg=SimpleNamespace(policy=SimpleNamespace(run_frozen_gmt_from_suffix=lambda obs: torch.zeros(8, 1))),
    )
    original_command = one_action.frontres_motion_command
    original_suffix = one_action._read_v017_normalized_gmt_suffix
    original_capture = one_action.capture_frontres_v017_execution_frame
    try:
        one_action.frontres_motion_command = lambda _runner: command
        one_action._read_v017_normalized_gmt_suffix = lambda _runner: torch.zeros(8, 770)
        one_action.capture_frontres_v017_execution_frame = capture
        trajectory, support = one_action.collect_frontres_v017_no_actor_baseline(
            runner,
            horizon_k=2,
            authoritative_rows=authoritative,
        )
    finally:
        one_action.frontres_motion_command = original_command
        one_action._read_v017_normalized_gmt_suffix = original_suffix
        one_action.capture_frontres_v017_execution_frame = original_capture
    assert step_count == 3
    assert capture_count == 2
    assert tuple(trajectory.joint_pos.shape) == (2, 2, 29)
    assert tuple(support.shape) == (2, 2, 2)


def _trajectory(joint: float, *, foot: float, zmp: float) -> FrontRESExecutedKTrajectory:
    contact = torch.ones(K, 1, 2)
    result = FrontRESExecutedKTrajectory(
        joint_pos=torch.full((K, 1, 29), joint),
        root_pos=torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 3).expand(K, 1, 3).clone(),
        root_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(K, 1, 4).clone(),
        key_body_pos=torch.zeros(K, 1, 3, 3),
        root_lin_vel=torch.zeros(K, 1, 3),
        root_ang_vel=torch.zeros(K, 1, 3),
        foot_pos=torch.full((K, 1, 2, 3), foot),
        contact=contact,
        zmp_margin=torch.full((K, 1), zmp),
        survival=torch.ones(K, 1),
        valid_mask=torch.ones(K, 1, dtype=torch.bool),
    )
    result.validate()
    return result


def _baseline(source: int) -> FrontRESSegmentBaselineEvidence:
    result = FrontRESSegmentBaselineEvidence(
        transaction_id="tx-v017",
        policy_snapshot_id="pi-old-v017",
        scenario_id=f"scenario-{source}",
        noisy_segment_hash=f"hash-{source}",
        x_t_identity=f"x-{source}",
        source_index=source,
        segment_id=10 + source,
        horizon_k=K,
        expected_support=torch.ones(K, 1, 2),
        clean=_trajectory(0.0, foot=0.0, zmp=0.02),
        noisy=_trajectory(0.087, foot=0.03, zmp=-0.02),
    )
    result.validate()
    return result


def _attempt(source: int, trial: int, joint: float) -> FrontRESRepairAttemptEvidence:
    result = FrontRESRepairAttemptEvidence(
        transaction_id="tx-v017",
        policy_snapshot_id="pi-old-v017",
        scenario_id=f"scenario-{source}",
        noisy_segment_hash=f"hash-{source}",
        x_t_identity=f"x-{source}",
        source_index=source,
        segment_id=10 + source,
        trial_index=trial,
        horizon_k=K,
        policy_observation=torch.tensor([float(source), float(trial), 1.0]),
        policy_privileged_observation=torch.tensor([float(source), float(trial), 2.0]),
        policy_action=torch.full((6,), 0.005 * (trial + 1)),
        policy_log_prob=torch.tensor(0.0),
        policy_value=torch.tensor(0.01 * trial),
        policy_mean=torch.zeros(6),
        policy_sigma=torch.ones(6),
        repair=_trajectory(joint, foot=0.01, zmp=0.0),
    )
    result.validate()
    return result


def _sealed() -> FrontRESSealedRecoveryAwareGainBatch:
    result = FrontRESSealedRecoveryAwareGainBatch(
        baselines=(_baseline(0), _baseline(1)),
        attempts=(
            _attempt(0, 0, 0.0435),
            _attempt(0, 1, 0.06525),
            _attempt(1, 0, 0.0435),
            _attempt(1, 1, 0.06525),
        ),
        active_m=2,
    )
    result.validate()
    return result


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.value_scale = torch.nn.Parameter(torch.tensor(0.0))

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        return {
            "log_prob": self.log_scale * observations[:, 2],
            "value": self.value_scale * observations[:, 2],
            "entropy": torch.zeros(observations.shape[0]),
        }


def test_v017_sealed_gain_to_grouped_scalar_ppo() -> None:
    sealed = _sealed()
    gain = compute_recovery_aware_gain(sealed.to_gain_input(), config=FrontRESRecoveryAwareGainConfig(beta=0.02))
    storage = build_frontres_v017_grouped_candidate_storage(
        sealed,
        gain,
        motion_ids=("motion-a", "motion-a", "motion-b", "motion-b"),
        start_frames=torch.tensor([4, 4, 8, 8]),
        intent_q29_provenance="deployment_noisy_q29",
        intent_q29_source="motion_internal_q29",
    )
    batch = storage.to_grouped_ppo_candidate_batch(FrontRESSegmentPPOBatch)
    assert torch.equal(batch.returns, gain.gain_total)
    assert torch.equal(batch.advantages, gain.gain_total - batch.old_values)
    result = compute_frontres_segment_ppo_loss(
        _Policy(),
        batch,
        cfg=FrontRESSegmentPPOConfig(
            normalize_advantages=False,
            advantage_normalization="grouped_scale_only",
        ),
    )
    assert result.valid_count == 4
    assert result.grouped_segment_count == 2
    assert result.grouped_attempt_count == 4
    assert all(abs(value - 0.25) < 1.0e-7 for value in result.grouped_attempt_mass_shares)

    report = build_frontres_v017_local_evaluation_report(sealed, gain)
    report.validate()
    assert report.gain_total == tuple(float(value) for value in gain.gain_total.tolist())
    assert report.intent_remaining_noisy == tuple(float(value) for value in gain.intent_remaining_noisy.tolist())
    assert report.physics_remaining_repaired == tuple(float(value) for value in gain.physics_remaining_repaired.tolist())
    assert report.intent_channel_noisy == tuple(tuple(float(item) for item in row) for row in gain.intent_channel_noisy.tolist())
    assert report.physics_channel_repaired == tuple(tuple(float(item) for item in row) for row in gain.physics_channel_repaired.tolist())
    assert report.cost_free_score == tuple(float(value) for value in gain.cost_free_score.tolist())
    assert report.beta == 0.02
    assert report.clean_execution_count == (1, 1)
    assert report.noisy_execution_count == (1, 1)
    assert all(all(not value for value in row) for row in report.contact_violation_repair_steps)
    assert all(all(value for value in row) for row in report.zmp_applicable_repair_steps)
    assert report.sustained_lean_repair == (False, False, False, False)


def test_v017_evaluation_contact_zmp_lean_and_unplanned_cases_are_hand_checkable() -> None:
    valid = torch.ones(4, dtype=torch.bool)
    expected_left = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=torch.bool)
    actual_left = expected_left.clone()
    actual_unloaded = torch.zeros_like(expected_left)
    assert not bool(torch.any(expected_left != actual_left, dim=-1).any())
    assert bool(torch.any(expected_left != actual_unloaded, dim=-1).all())
    applicable_left = valid & expected_left.any(dim=-1) & actual_left.any(dim=-1)
    applicable_unloaded = valid & expected_left.any(dim=-1) & actual_unloaded.any(dim=-1)
    assert applicable_left.tolist() == [True, True, True, True]
    assert applicable_unloaded.tolist() == [False, False, False, False]

    expected_flight = torch.zeros_like(expected_left)
    flight_violation = valid & torch.any(expected_flight != actual_unloaded, dim=-1)
    flight_applicable = valid & expected_flight.any(dim=-1) & actual_unloaded.any(dim=-1)
    assert flight_violation.tolist() == [False, False, False, False]
    assert flight_applicable.tolist() == [False, False, False, False]

    actual_extra_step = torch.tensor([[1, 1], [1, 0], [1, 1], [1, 1]], dtype=torch.bool)
    unplanned = _v017_unplanned_contact_transitions(
        torch.ones(4, 2, dtype=torch.bool),
        actual_extra_step,
        valid,
        tolerance=1,
    )
    assert unplanned.tolist() == [False, True, True, False]

    violation = torch.tensor([0.0, 0.3, 0.1, 0.0])
    recovery = _v017_zmp_recovery_projection(violation, torch.ones(4, dtype=torch.bool))
    assert recovery[0] is None
    torch.testing.assert_close(
        torch.tensor(recovery[1:]),
        torch.tensor([0.3, 0.1, 0.0]),
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    assert _v017_zmp_recovery_projection(violation, torch.zeros(4, dtype=torch.bool)) == (None,) * 4

    angle = torch.tensor(0.2)
    roll_quat = torch.tensor([torch.cos(angle / 2), torch.sin(angle / 2), 0.0, 0.0]).repeat(4, 1)
    roll = _v017_lateral_roll(roll_quat)
    torch.testing.assert_close(roll, torch.full((4,), 0.2), atol=1e-6, rtol=1e-6)


def test_v017_evaluation_is_read_only_and_atomic() -> None:
    sealed = _sealed()
    gain = compute_recovery_aware_gain(
        sealed.to_gain_input(),
        config=FrontRESRecoveryAwareGainConfig(),
    )
    before = tuple(value.policy_action.clone() for value in sealed.attempts)
    report = build_frontres_v017_local_evaluation_report(sealed, gain)
    report.validate()
    for expected, attempt in zip(before, sealed.attempts, strict=True):
        torch.testing.assert_close(expected, attempt.policy_action)

    zmp_na_rows = tuple((row[0], row[1], None, row[3]) for row in report.physics_channel_repaired)
    replace(report, physics_channel_repaired=zmp_na_rows).validate()
    try:
        replace(report, physics_channel_repaired=((None, 0.0, None, 0.0),) * report.policy_row_count).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("missing Contact diagnostics must fail closed rather than become zero")
    try:
        replace(report, beta=float("nan")).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("missing/non-finite Gain identity must fail closed")


def test_v017_partial_or_mixed_batch_rejects() -> None:
    sealed = _sealed()
    try:
        FrontRESSealedRecoveryAwareGainBatch(
            baselines=sealed.baselines,
            attempts=sealed.attempts[:-1],
            active_m=2,
        ).validate()
    except ValueError as exc:
        assert "exact" in str(exc)
    else:
        raise AssertionError("v017 partial transaction must fail closed")


def test_active_stage3_rejects_legacy_local_evaluation() -> None:
    train_source = (Path(__file__).resolve().parents[4] / "scripts" / "rsl_rl" / "train.py").read_text(
        encoding="utf-8"
    )
    assert "FRS-EVAL-v004 rejects legacy v002/v006/quartet local evaluation" in train_source
    assert '"frontres_segment_offline_eval_only"' not in train_source
    assert '"frontres_policy_quality_eval_only"' in train_source
    assert '"frontres_policy_quality_q2d_eval_only"' in train_source
    assert "full-sequence composition remains a separate mode" in train_source


def main() -> None:
    test_baseline_capture_projects_only_authoritative_segment_rows()
    test_v017_sealed_gain_to_grouped_scalar_ppo()
    test_v017_partial_or_mixed_batch_rejects()
    test_v017_evaluation_contact_zmp_lean_and_unplanned_cases_are_hand_checkable()
    test_v017_evaluation_is_read_only_and_atomic()
    test_active_stage3_rejects_legacy_local_evaluation()
    print("[T-v017-step1] baseline -> Gain -> grouped scalar PPO -> report pass", flush=True)


if __name__ == "__main__":
    main()
