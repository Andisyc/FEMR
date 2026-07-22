from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]

DIAG_PATH = ROOT / "rsl_rl" / "frontres" / "frontres_segment_diagnostics.py"
diag_spec = importlib.util.spec_from_file_location("frontres_segment_diagnostics", DIAG_PATH)
diag_module = importlib.util.module_from_spec(diag_spec)
assert diag_spec.loader is not None
sys.modules[diag_spec.name] = diag_module
diag_spec.loader.exec_module(diag_module)
format_segment_replay_log = diag_module.format_segment_replay_log
format_segment_train_effect_log = diag_module.format_segment_train_effect_log
format_segment_motion_quality_log = diag_module.format_segment_motion_quality_log
format_segment_periodic_eval_log = diag_module.format_segment_periodic_eval_log
action_distribution_health_summary = diag_module.action_distribution_health_summary
motion_quality_summary_to_scalars = diag_module.motion_quality_summary_to_scalars
periodic_eval_summary_to_scalars = diag_module.periodic_eval_summary_to_scalars
repair_effect_summary_to_scalars = diag_module.repair_effect_summary_to_scalars
segment_summary_to_scalars = diag_module.segment_summary_to_scalars
summarize_segment_batch = diag_module.summarize_segment_batch


def _v015_candidate_evidence() -> SimpleNamespace:
    actions = torch.tensor(
        [
            [0.10, -0.20, 0.30, -0.40, 0.50, -0.60],
            [-0.11, 0.21, -0.31, 0.41, -0.51, 0.61],
            [0.12, 0.22, 0.32, 0.42, 0.52, 0.62],
        ],
        dtype=torch.float32,
    )
    return_evidence = SimpleNamespace(
        validate=lambda: None,
        gain_source="FRS-GAIN-v003-intent-physics-local-repair",
        policy_actions=actions,
        policy_row_valid=torch.tensor([True, True, False]),
        intent_gain=torch.tensor([0.30, -0.10, float("nan")]),
        physics_gain=torch.tensor([0.20, -0.20, float("nan")]),
        repair_cost=torch.tensor([0.05, 0.05, float("nan")]),
        gain_total=torch.tensor([0.45, -0.35, float("nan")]),
        policy_values=torch.tensor([0.10, 0.20, float("nan")]),
        return_k=torch.tensor([0.45, -0.35, float("nan")]),
        advantage_k=torch.tensor([0.35, -0.55, float("nan")]),
        repaired_success=torch.tensor([1.0, 0.0, float("nan")]),
        noisy_success=torch.tensor([0.0, 0.0, float("nan")]),
        repaired_survival=torch.tensor([4.0, 2.0, float("nan")]),
        noisy_survival=torch.tensor([2.0, 2.0, float("nan")]),
        physics_survival_quality_repaired=torch.tensor([1.0, 0.5, float("nan")]),
        physics_survival_quality_noisy=torch.tensor([0.5, 0.5, float("nan")]),
        repaired_zmp_margin=torch.tensor([0.3, 0.1, float("nan")]),
        noisy_zmp_margin=torch.tensor([0.1, 0.1, float("nan")]),
        repaired_contact=torch.tensor([1.0, 0.5, float("nan")]),
        noisy_contact=torch.tensor([0.5, 0.5, float("nan")]),
        physics_success_gain=torch.tensor([1.0, 0.0, float("nan")]),
        physics_survival_gain=torch.tensor([0.5, 0.0, float("nan")]),
        physics_zmp_gain=torch.tensor([0.2, 0.0, float("nan")]),
        physics_contact_gain=torch.tensor([0.5, 0.0, float("nan")]),
        physics_valid_step_count=torch.tensor([4, 2, 0]),
        horizon_k=torch.tensor([4, 4, 4]),
        scenario_ids=("scenario-a", "scenario-b", "scenario-c"),
        noisy_segment_hashes=("hash-a", "hash-b", "hash-c"),
        x_t_identities=("x-a", "x-b", "x-c"),
        intent_q29_provenance="deployment_noisy_q29",
        intent_q29_source="fixture-deployment-motion-q29",
    )
    return SimpleNamespace(validate=lambda: None, return_evidence=return_evidence)


def test_v015_transaction_telemetry_projects_sealed_rows_without_recompute() -> None:
    candidate = _v015_candidate_evidence()
    report = diag_module.build_frontres_v015_local_evaluation_report(
        candidate,
        transaction_id="tx-v015-g5-s1",
    )

    assert report.transaction_id == "tx-v015-g5-s1"
    assert report.policy_actions == tuple(tuple(float(value) for value in row) for row in candidate.return_evidence.policy_actions)
    assert report.valid_policy_row_mask == (True, True, False)
    assert report.intent_gain[:2] == (0.30000001192092896, -0.10000000149011612)
    assert report.physics_gain[:2] == (0.20000000298023224, -0.20000000298023224)
    assert report.repair_cost[:2] == (0.05000000074505806, 0.05000000074505806)
    assert report.gain_total[:2] == (0.44999998807907104, -0.3499999940395355)
    assert report.policy_values[:2] == (0.10000000149011612, 0.20000000298023224)
    assert report.raw_advantages[:2] == (0.3499999940395355, -0.550000011920929)
    assert math.isclose(report.physics_zmp_gain[0], 0.2, rel_tol=0.0, abs_tol=1.0e-6)
    assert report.physics_zmp_gain[1] == 0.0
    assert report.physics_contact_gain[:2] == (0.5, 0.0)
    assert all(math.isnan(values[2]) for values in (report.intent_gain, report.physics_gain, report.repair_cost, report.gain_total))
    assert report.gain_total_pos_frac == 0.5
    assert report.gain_total_neg_frac == 0.5
    assert report.scenario_ids == ("scenario-a", "scenario-b", "scenario-c")
    assert report.noisy_segment_hashes == ("hash-a", "hash-b", "hash-c")
    assert report.return_feedback is False
    assert report.priority_feedback is False
    assert report.ppo_feedback is False

    delattr(candidate.return_evidence, "physics_gain")
    try:
        diag_module.build_frontres_v015_local_evaluation_report(candidate, transaction_id="tx-missing")
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("missing v003 telemetry must fail closed instead of being filled with zero")


@dataclass(frozen=True)
class FakeSample:
    source: tuple[str, ...]
    priority: torch.Tensor
    horizon_k: torch.Tensor


@dataclass(frozen=True)
class FakeReward:
    score_noisy: torch.Tensor
    score_repaired: torch.Tensor
    score_clean: torch.Tensor
    gain_over_noisy: torch.Tensor
    fall_flag: torch.Tensor
    contact_consistency: torch.Tensor
    valid_mask: torch.Tensor
    solved_mask: torch.Tensor
    hopeless_mask: torch.Tensor


@dataclass(frozen=True)
class FakeReset:
    success_mask: torch.Tensor
    preroll_mask: torch.Tensor


@dataclass(frozen=True)
class FakeActionStats:
    action_norm_mean: float
    per_dim_norm: torch.Tensor


def test_segment_diagnostics_required_keys_and_no_acceptance_keys() -> None:
    summary = summarize_segment_batch(
        FakeSample(
            source=("global", "replay", "review", "replay"),
            priority=torch.tensor([0.1, 0.5, 0.0, 0.2]),
            horizon_k=torch.tensor([4, 4, 4, 4]),
        ),
        FakeReward(
            score_noisy=torch.tensor([0.2, 0.4, 0.9, 0.1]),
            score_repaired=torch.tensor([0.5, 0.6, 0.92, 0.1]),
            score_clean=torch.ones(4),
            gain_over_noisy=torch.tensor([0.3, 0.2, 0.02, 0.0]),
            fall_flag=torch.tensor([False, False, False, True]),
            contact_consistency=torch.tensor([1.0, 0.8, 1.0, 0.2]),
            valid_mask=torch.tensor([True, True, True, False]),
            solved_mask=torch.tensor([False, False, True, False]),
            hopeless_mask=torch.tensor([False, False, False, True]),
        ),
        FakeReset(success_mask=torch.tensor([True, True, True, False]), preroll_mask=torch.tensor([False, True, False, True])),
        FakeActionStats(action_norm_mean=0.7, per_dim_norm=torch.arange(6, dtype=torch.float32)),
    )
    scalars = segment_summary_to_scalars(summary)
    required = {
        "segment/global_frac",
        "segment/replay_frac",
        "segment/review_frac",
        "segment/replay_pool_size",
        "segment/priority_mean",
        "segment/priority_p90",
        "segment/solved_frac",
        "segment/active_frac",
        "segment/hopeless_frac",
        "segment/reset_success_frac",
        "segment/preroll_frac",
        "segment/k",
        "segment/score_noisy",
        "segment/score_repaired",
        "segment/score_clean",
        "segment/gain_over_noisy",
        "segment/fall_frac",
        "segment/contact_consistency",
        "segment/action_norm",
        "segment/action_norm_dx",
        "segment/action_norm_dy",
        "segment/action_norm_dz",
        "segment/action_norm_droll",
        "segment/action_norm_dpitch",
        "segment/action_norm_dyaw",
    }
    assert required.issubset(scalars.keys())
    forbidden = {"acceptance_gt", "acceptance_mask", "acceptance_margin", "acceptance_prob"}
    assert forbidden.isdisjoint(scalars.keys())
    assert scalars["segment/global_frac"] == 0.25
    assert scalars["segment/replay_frac"] == 0.5
    assert scalars["segment/review_frac"] == 0.25
    assert scalars["segment/reset_success_frac"] == 0.75
    assert scalars["segment/preroll_frac"] == 0.5
    assert scalars["segment/action_norm_dyaw"] == 5.0


def test_segment_log_contains_live_path_sentinel() -> None:
    summary = summarize_segment_batch(
        FakeSample(source=("global",), priority=torch.tensor([0.0]), horizon_k=torch.tensor([8])),
        FakeReward(
            score_noisy=torch.tensor([0.2]),
            score_repaired=torch.tensor([0.5]),
            score_clean=torch.tensor([1.0]),
            gain_over_noisy=torch.tensor([0.3]),
            fall_flag=torch.tensor([False]),
            contact_consistency=torch.tensor([1.0]),
            valid_mask=torch.tensor([True]),
            solved_mask=torch.tensor([False]),
            hopeless_mask=torch.tensor([False]),
        ),
        FakeReset(success_mask=torch.tensor([True]), preroll_mask=torch.tensor([False])),
        FakeActionStats(action_norm_mean=0.1, per_dim_norm=torch.zeros(6)),
    )
    log = format_segment_replay_log(summary)
    assert "FrontRES Segment HRL active" in log
    assert "stage=stage3_segment_hrl" in log
    assert "objective=segment_replay_hrl" in log
    assert "k=8" in log
    assert "gain=0.3000" in log


def test_repair_effect_summary_formats_training_fit_metrics() -> None:
    summary = {
        "gain_style_mean": 0.02,
        "gain_physics_mean": 0.04,
        "gain_repair_cost_mean": 0.01,
        "gain_total_mean": 0.05,
        "gain_total_pos_frac": 0.75,
        "done_frac": 0.1,
        "storage_valid_frac": 0.25,
        "sampler_replay_pool_size": 32,
        "sampler_replay_candidates": 12,
        "audit_identity_mode": "aggregate",
        "audit_transaction_count": 2,
        "audit_transaction_ids": ("iter1:capture1", "iter1:capture2"),
        "audit_batch_signature_count": 2,
        "audit_batch_signatures": ("batch-a", "batch-b"),
        "audit_same_transaction": False,
    }
    scalars = repair_effect_summary_to_scalars(summary)
    assert scalars["segment/train_effect_gain_style"] == 0.02
    assert scalars["segment/train_effect_gain_physics"] == 0.04
    assert scalars["segment/train_effect_repair_cost"] == 0.01
    assert scalars["segment/train_effect_gain_total"] == 0.05
    assert scalars["segment/train_effect_gain_pos_frac"] == 0.75
    assert scalars["segment/train_effect_fall_rate"] == 0.1
    assert scalars["segment/train_effect_valid_frac"] == 0.25
    assert scalars["segment/train_effect_replay_candidates"] == 12.0
    assert scalars["segment/train_effect_replay_pool_size"] == 32.0
    log = format_segment_train_effect_log(summary)
    assert "[FrontRES Segment Train Effect]" in log
    assert "style=0.020000" in log
    assert "physics=0.040000" in log
    assert "repair_cost=0.010000" in log
    assert "total=0.050000" in log
    assert "gain_pos=75.0%" in log
    assert "fall=10.0%" in log
    assert "pool=32" in log
    assert "mode=aggregate" in log
    assert "transactions=2" in log
    assert "same_transaction=False" in log


def test_repair_effect_summary_uses_live_sampler_candidate_field() -> None:
    summary = {
        "gain_style_mean": 0.02,
        "gain_physics_mean": 0.04,
        "gain_repair_cost_mean": 0.01,
        "gain_total_mean": 0.05,
        "gain_total_pos_frac": 0.75,
        "done_frac": 0.1,
        "storage_valid_frac": 0.25,
        "sampler_replay_pool_size": 32,
        "sampler_replay_candidates": 0,
        "sampler_update_replay_candidate_count": 8347,
    }
    scalars = repair_effect_summary_to_scalars(summary)
    assert scalars["segment/train_effect_replay_candidates"] == 8347.0
    log = format_segment_train_effect_log(summary)
    assert "candidates=8347" in log


def test_train_effect_ignores_legacy_score_fields_and_marks_missing_gain() -> None:
    canonical = {
        "gain_style_mean": 0.02,
        "gain_physics_mean": 0.04,
        "gain_repair_cost_mean": 0.01,
        "gain_total_mean": 0.05,
        "gain_total_pos_frac": 0.75,
        "done_frac": 0.1,
        "storage_valid_frac": 0.25,
        "sampler_replay_pool_size": 32,
        "sampler_update_replay_candidate_count": 12,
        "score_noisy_mean": -999.0,
        "score_repaired_mean": 999.0,
        "score_gain_mean": 999.0,
    }
    log = format_segment_train_effect_log(canonical)
    assert "total=0.050000" in log
    assert "999.000000" not in log
    missing = format_segment_train_effect_log(
        {
            "done_frac": 0.0,
            "storage_valid_frac": 1.0,
            "sampler_replay_pool_size": 0,
            "sampler_update_replay_candidate_count": 0,
        }
    )
    assert "style=UNCONFIRMED" in missing
    assert "physics=UNCONFIRMED" in missing
    assert "repair_cost=UNCONFIRMED" in missing
    assert "total=UNCONFIRMED" in missing


def test_motion_quality_summary_measures_pose_velocity_and_delta_se() -> None:
    clean = torch.tensor(
        [[
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
        ]]
    )
    repaired = clean.clone()
    repaired[..., 0] += 0.1
    noisy = clean.clone()
    noisy[..., 0] += 0.3
    delta_se = torch.tensor([[0.1, 0.2, -0.3, 0.0, 0.0, 0.4]])
    scalars = motion_quality_summary_to_scalars(
        clean_positions=clean,
        repaired_positions=repaired,
        noisy_positions=noisy,
        delta_se=delta_se,
        valid_mask=torch.tensor([True]),
    )
    assert abs(scalars["segment/motion_mpjpe_repaired_clean"] - 0.1) < 1e-6
    assert abs(scalars["segment/motion_mpjpe_noisy_clean"] - 0.3) < 1e-6
    assert scalars["segment/motion_vel_error_repaired_clean"] == 0.0
    assert scalars["segment/motion_acc_error_repaired_clean"] == 0.0
    assert abs(scalars["segment/motion_delta_se_norm"] - float(torch.linalg.norm(delta_se, dim=-1).item())) < 1e-6
    assert scalars["segment/motion_delta_z_up_frac"] == 0.0
    log = format_segment_motion_quality_log(scalars)
    assert "[FrontRES Segment Motion Quality]" in log
    assert "mpjpe_repaired=0.100000" in log
    assert "vel_err=0.000000" in log
    assert "dz_up=0.0%" in log


def test_motion_quality_summary_respects_per_row_horizon_mask() -> None:
    clean = torch.zeros((2, 4, 1, 3))
    repaired = clean.clone()
    repaired[0, 1:, 0, 0] = 100.0
    repaired[1, :2, 0, 0] = 2.0
    repaired[1, 2:, 0, 0] = 100.0

    scalars = motion_quality_summary_to_scalars(
        clean_positions=clean,
        repaired_positions=repaired,
        noisy_positions=clean,
        valid_mask=torch.tensor([True, True]),
        temporal_mask=torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
            ]
        ),
    )

    assert abs(scalars["segment/motion_mpjpe_repaired_clean"] - (4.0 / 3.0)) < 1e-6


def test_motion_quality_missing_positions_are_unconfirmed_not_zero() -> None:
    delta_se = torch.tensor([[0.1, 0.2, -0.3, 0.0, 0.0, 0.4]])
    scalars = motion_quality_summary_to_scalars(delta_se=delta_se)
    assert math.isnan(scalars["segment/motion_mpjpe_repaired_clean"])
    assert math.isnan(scalars["segment/motion_mpjpe_noisy_clean"])
    assert math.isnan(scalars["segment/motion_vel_error_repaired_clean"])
    assert math.isnan(scalars["segment/motion_acc_error_repaired_clean"])
    assert scalars["segment/motion_delta_se_norm"] > 0.0
    log = format_segment_motion_quality_log(scalars)
    assert "mpjpe_repaired=UNCONFIRMED" in log
    assert "mpjpe_noisy=UNCONFIRMED" in log
    assert "vel_err=UNCONFIRMED" in log
    assert "acc_err=UNCONFIRMED" in log
    assert "mpjpe_repaired=0.000000" not in log


def test_motion_quality_keeps_action_diagnostics_when_all_samples_fall() -> None:
    clean = torch.zeros((2, 3, 1, 3))
    repaired = clean.clone()
    repaired[..., 0] = 1.0
    delta_se = torch.tensor(
        [
            [0.1, 0.0, 0.2, 0.0, 0.0, 0.3],
            [0.4, 0.0, -0.5, 0.0, 0.0, 0.6],
        ]
    )
    scalars = motion_quality_summary_to_scalars(
        clean_positions=clean,
        repaired_positions=repaired,
        noisy_positions=repaired,
        delta_se=delta_se,
        valid_mask=torch.tensor([False, False]),
    )
    expected_delta = float(torch.linalg.norm(delta_se, dim=-1).mean().item())
    assert scalars["segment/motion_mpjpe_repaired_clean"] == 0.0
    assert abs(scalars["segment/motion_delta_se_norm"] - expected_delta) < 1e-6
    assert scalars["segment/motion_delta_z_up_frac"] == 0.5
    print(
        "[probe step6] all_fall_motion_quality_action_visible "
        f"mpjpe={scalars['segment/motion_mpjpe_repaired_clean']:.6f} "
        f"delta={scalars['segment/motion_delta_se_norm']:.6f} "
        f"dz_up={scalars['segment/motion_delta_z_up_frac']:.3f}",
        flush=True,
    )


def test_periodic_eval_summary_formats_long_rollout_metrics() -> None:
    summary = {
        "episode_length": 500,
        "success_rate": 0.7,
        "fall_rate": 0.2,
        "mean_survival_steps": 430,
        "gain_source": "FRS-GAIN-v002",
        "gain_style_mean": 0.08,
        "gain_physics_mean": 0.06,
        "gain_repair_cost_mean": 0.02,
        "gain_total_mean": 0.12,
        "gain_total_pos_frac": 0.75,
        "segment/motion_mpjpe_repaired_clean": 0.11,
        "segment/motion_mpjpe_noisy_clean": 0.44,
        "segment/motion_vel_error_repaired_clean": 0.02,
        "segment/motion_acc_error_repaired_clean": 0.03,
        "segment/motion_delta_se_norm": 0.42,
        "segment/motion_delta_z_up_frac": 0.25,
    }
    scalars = periodic_eval_summary_to_scalars(summary)
    assert scalars["segment/eval_episode_length"] == 500.0
    assert scalars["segment/eval_success_rate"] == 0.7
    assert scalars["segment/eval_fall_rate"] == 0.2
    assert scalars["segment/eval_mean_survival_steps"] == 430.0
    assert scalars["segment/eval_gain_style"] == 0.08
    assert scalars["segment/eval_gain_physics"] == 0.06
    assert scalars["segment/eval_gain_repair_cost"] == 0.02
    assert scalars["segment/eval_gain_total"] == 0.12
    assert scalars["segment/eval_gain_total_pos_frac"] == 0.75
    log = format_segment_periodic_eval_log(summary)
    assert "[FrontRES Segment Periodic Eval]" in log
    assert "episode_length=500.0" in log
    assert "survival=430.0" in log
    assert "success=70.0%" in log
    assert "fall=20.0%" in log
    assert "source=FRS-GAIN-v002" in log
    assert "style=0.080000 physics=0.060000 repair_cost=0.020000 total=0.120000 positive=75.0%" in log
    assert "score:" not in log
    assert "mpjpe_repaired=0.110000" in log
    assert "mpjpe_noisy=0.440000" in log
    assert "vel_err=0.020000" in log
    assert "acc_err=0.030000" in log
    assert "delta_se_norm=0.420000" in log
    assert "dz_up=25.0%" in log


def test_action_distribution_health_flags_raw_mean_saturation() -> None:
    means = torch.tensor(
        [
            [0.0, 0.0, 0.0, 6_844_382.0, 6_480_798.5, 0.0],
            [0.0, 0.0, 0.0, 7_088_041.0, 6_711_514.0, 0.0],
        ]
    )
    sigmas = torch.full((2, 6), 0.01)
    actions = torch.zeros(2, 6)
    actions[:, 3:5] = 0.4
    target = torch.zeros(2, 6)
    target[:, 3:5] = torch.tensor([[0.0017, 0.0185], [0.0051, 0.0217]])

    summary = action_distribution_health_summary(
        means=means,
        sigmas=sigmas,
        actions=actions,
        supervised_target=target,
    )

    assert summary["available"] is True
    assert summary["status"] == "BAD_RAW_MEAN_SATURATED"
    assert float(summary["raw_mean_abs_max"]) > 1_000_000.0
    assert float(summary["raw_saturated_frac_abs_gt_20"]) > 0.0
    assert abs(float(summary["sigma_mean"]) - 0.01) < 1e-6
    assert float(summary["action_norm_over_target_norm"]) > 10.0


def main() -> None:
    test_v015_transaction_telemetry_projects_sealed_rows_without_recompute()
    test_segment_diagnostics_required_keys_and_no_acceptance_keys()
    test_segment_log_contains_live_path_sentinel()
    test_repair_effect_summary_formats_training_fit_metrics()
    test_repair_effect_summary_uses_live_sampler_candidate_field()
    test_motion_quality_summary_measures_pose_velocity_and_delta_se()
    test_motion_quality_summary_respects_per_row_horizon_mask()
    test_motion_quality_missing_positions_are_unconfirmed_not_zero()
    test_motion_quality_keeps_action_diagnostics_when_all_samples_fall()
    test_periodic_eval_summary_formats_long_rollout_metrics()
    test_action_distribution_health_flags_raw_mean_saturation()
    print("result: PASS")


if __name__ == "__main__":
    main()
