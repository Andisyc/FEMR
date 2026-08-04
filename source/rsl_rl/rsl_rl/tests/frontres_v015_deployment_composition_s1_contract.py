#!/usr/bin/env python3
"""Deterministic S1 contract for the v015 deployment-composition kernel."""

from __future__ import annotations

from dataclasses import fields, replace
import importlib.util
from pathlib import Path
import sys
import tempfile

import numpy as np
from frontres_contract_imports import install_frontres_contract_packages


ROOT = Path(__file__).resolve().parents[2]
OWNER_PATH = ROOT / "rsl_rl" / "runners" / "frontres_segment_sequence_eval.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
install_frontres_contract_packages(ROOT / "rsl_rl")


def _load_owner():
    spec = importlib.util.spec_from_file_location("frontres_v015_deployment_composition_owner", OWNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_npz(path: Path, *, frame_count: int = 5, offset: float = 0.0) -> None:
    body_count = 3
    q = np.arange(frame_count * 29, dtype=np.float32).reshape(frame_count, 29) / 100.0 + offset
    dq = q / 10.0
    body_pos = np.arange(frame_count * body_count * 3, dtype=np.float32).reshape(frame_count, body_count, 3)
    body_quat = np.zeros((frame_count, body_count, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_lin = np.full((frame_count, body_count, 3), 0.25, dtype=np.float32)
    body_ang = np.full((frame_count, body_count, 3), -0.5, dtype=np.float32)
    np.savez(
        path,
        fps=np.asarray(50.0, dtype=np.float32),
        joint_pos=q,
        joint_vel=dq,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin,
        body_ang_vel_w=body_ang,
    )


def _expect_value_error(fn) -> str:
    try:
        fn()
    except ValueError as exc:
        return str(exc)
    raise AssertionError("expected ValueError")


def test_t_npz_schema_identity_and_corruption_protocol(owner) -> None:
    protocol_a = owner.build_frontres_v015_persistent_corruption_protocol(
        corruption_id="rp-sequence-0001",
        family="local_rp",
        seed=20260721,
        parameters={"pitch_std": 0.08, "roll_std": 0.06},
    )
    protocol_b = owner.build_frontres_v015_persistent_corruption_protocol(
        corruption_id="rp-sequence-0001",
        family="local_rp",
        seed=20260721,
        parameters={"roll_std": 0.06, "pitch_std": 0.08},
    )
    assert protocol_a == protocol_b
    assert protocol_a.temporal_mode == "persistent_full_sequence"
    assert len(protocol_a.protocol_hash) == 64
    composite_a = owner.build_frontres_v015_persistent_corruption_protocol(
        corruption_id="composite",
        family="yaw+local_rp",
        seed=4,
        parameters={},
    )
    composite_b = owner.build_frontres_v015_persistent_corruption_protocol(
        corruption_id="composite",
        family="local_rp+yaw",
        seed=4,
        parameters={},
    )
    assert composite_a == composite_b
    assert "protocol_hash" in _expect_value_error(
        lambda: replace(protocol_a, protocol_hash="0" * 64).validate()
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        reference = root / "deployment_motion.npz"
        _write_npz(reference)
        config = owner.FrontRESV015DeploymentCompositionConfig(
            enabled=True,
            source_reference_path=str(reference),
            reference_path=str(reference),
            future_offsets=(1, 2),
            corruption_protocol=protocol_a,
        )
        request_a = owner.load_frontres_v015_deployment_composition_request(config)
        request_b = owner.load_frontres_v015_deployment_composition_request(config)
        request_a.validate()
        assert request_a == request_b
        assert request_a.reference_path == str(reference.resolve())
        assert request_a.source_reference_path == str(reference.resolve())
        assert request_a.source_reference_file_hash == request_a.reference_file_hash
        assert request_a.reference_stream_id.startswith("deployment-npz:")
        assert len(request_a.reference_file_hash) == 64
        assert request_a.reference_provenance == "deployment_reference_stream"
        assert request_a.frame_count == 5
        assert request_a.joint_dof == 29
        assert request_a.body_count == 3
        assert request_a.future_offsets == (1, 2)

        _write_npz(reference, offset=1.0)
        request_c = owner.load_frontres_v015_deployment_composition_request(config)
        assert request_c.reference_file_hash != request_a.reference_file_hash
        assert request_c.reference_stream_id != request_a.reference_stream_id

        malformed = root / "malformed.npz"
        np.savez(malformed, fps=np.asarray(50.0), joint_pos=np.zeros((5, 28), dtype=np.float32))
        bad_config = replace(config, reference_path=str(malformed))
        assert "required arrays" in _expect_value_error(
            lambda: owner.load_frontres_v015_deployment_composition_request(bad_config)
        )
    print(
        "[T-npz-schema/T-identity/T-corruption-protocol] "
        "structured deployment NPZ and canonical persistent protocol are immutable",
        flush=True,
    )


def test_t_report_and_no_feedback(owner) -> None:
    protocol = owner.build_frontres_v015_persistent_corruption_protocol(
        corruption_id="persistent-rp",
        family="local_rp",
        seed=7,
        parameters={"scale": 1.0},
    )
    with tempfile.TemporaryDirectory() as tmp:
        reference = Path(tmp) / "deployment_motion.npz"
        _write_npz(reference, frame_count=4)
        request = owner.load_frontres_v015_deployment_composition_request(
            owner.FrontRESV015DeploymentCompositionConfig(
                enabled=True,
                source_reference_path=str(reference),
                reference_path=str(reference),
                future_offsets=(1, 2),
                corruption_protocol=protocol,
            )
        )
        route_start_hash = "a" * 64
        baseline = owner.FrontRESV015DeploymentBranchReport(
            role="baseline",
            route_start_state_hash=route_start_hash,
            per_frame_femr_action_used=(False, False),
            per_frame_intent_q29_error=(0.3, 0.2),
            per_frame_physics_success=(False, False),
            per_frame_fall=(False, True),
            per_frame_zmp_margin=(0.05, None),
            per_frame_contact_consistency=(0.8, 0.2),
            per_frame_policy_actions=(((0.0,) * 6,), ((0.0,) * 6,)),
            actual_contact_steps=(((True, False),), ((False, False),)),
            contact_mismatch_steps=(((False, True),), ((True, False),)),
            phase_zmp_applicable_steps=((True,), (False,)),
            phase_zmp_violation_steps=((0.1,), (None,)),
            phase_zmp_recovery_steps=((False,), (True,)),
            survival_steps=((True,), (False,)),
            lateral_roll_rad_steps=((0.02,), (0.04,)),
            lateral_roll_cumulative_mean_rad_steps=((0.02,), (0.03,)),
            unplanned_contact_steps=((False,), (True,)),
        )
        repair = owner.FrontRESV015DeploymentBranchReport(
            role="repair",
            route_start_state_hash=route_start_hash,
            per_frame_femr_action_used=(True, True),
            per_frame_intent_q29_error=(0.2, 0.1),
            per_frame_physics_success=(True, False),
            per_frame_fall=(False, True),
            per_frame_zmp_margin=(0.1, -0.03),
            per_frame_contact_consistency=(1.0, 0.2),
            per_frame_policy_actions=(((0.0,) * 6,), ((0.1,) * 6,)),
            actual_contact_steps=(((True, True),), ((False, False),)),
            contact_mismatch_steps=(((False, False),), ((True, False),)),
            phase_zmp_applicable_steps=((True,), (False,)),
            phase_zmp_violation_steps=((0.0,), (None,)),
            phase_zmp_recovery_steps=((False,), (True,)),
            survival_steps=((True,), (False,)),
            lateral_roll_rad_steps=((0.01,), (0.03,)),
            lateral_roll_cumulative_mean_rad_steps=((0.01,), (0.02,)),
            unplanned_contact_steps=((False,), (True,)),
        )
        report = owner.FrontRESV015DeploymentCompositionReport(
            request=request,
            baseline=baseline,
            repair=repair,
            route_start_state_hash=route_start_hash,
            expected_contact_steps=(((True, True),), ((True, False),)),
        )
        report.validate()
        assert report.reference_frame_count == 4
        assert report.frame_count == 2
        assert report.femr_action_count == 2
        assert report.accumulated_failure_count == 1
        assert report.unplanned_contact_event_count == 1
        assert report.phase_zmp_violation_count == 0
        assert report.survival_fraction == 0.5
        assert report.max_abs_cumulative_lateral_roll_rad == 0.02
        assert report.return_feedback is False
        assert report.priority_feedback is False
        assert report.ppo_feedback is False
        assert report.sampler_feedback is False
        assert report.optimizer_feedback is False

        forbidden = {
            "return_evidence",
            "priority_evidence",
            "ppo_batch",
            "sampler_state",
            "optimizer_state",
            "clean_continuation",
            "local_scenario",
        }
        assert forbidden.isdisjoint({field.name for field in fields(type(report))})
        try:
            type(report)(
                request=request,
                per_frame_femr_action_used=(True,) * 2,
                per_frame_intent_q29_error=(0.0,) * 2,
                per_frame_physics_success=(True,) * 2,
                per_frame_fall=(False,) * 2,
                per_frame_zmp_margin=(0.0,) * 2,
                per_frame_contact_consistency=(1.0,) * 2,
                return_evidence=object(),
            )
        except TypeError:
            pass
        else:
            raise AssertionError("v015 composition report accepted a local return carrier")
        assert "length" in _expect_value_error(
            lambda: replace(report, repair=replace(repair, per_frame_fall=(False,))).validate()
        )
        assert "success and fall" in _expect_value_error(
            lambda: replace(
                report,
                repair=replace(repair, per_frame_physics_success=(True, True)),
            ).validate()
        )
        assert "applicability" in _expect_value_error(
            lambda: replace(
                report,
                repair=replace(repair, phase_zmp_violation_steps=((None,), (None,))),
            ).validate()
        )
        assert "bool" in _expect_value_error(
            lambda: replace(
                report,
                repair=replace(repair, unplanned_contact_steps=((0,), (True,))),
            ).validate()
        )
        assert "same-state Baseline and Repair" in _expect_value_error(
            lambda: replace(report, baseline=replace(baseline, route_start_state_hash="b" * 64)).validate()
        )
        assert "never invoke FEMR" in _expect_value_error(
            lambda: replace(
                report,
                baseline=replace(baseline, per_frame_femr_action_used=(True, False)),
            ).validate()
        )
        assert "exact zero" in _expect_value_error(
            lambda: replace(
                report,
                baseline=replace(
                    baseline,
                    per_frame_policy_actions=(((0.01,) + (0.0,) * 5,), ((0.0,) * 6,)),
                ),
            ).validate()
        )
    print(
        "[T-report/T-no-feedback] per-frame deployment metrics expose no local return or training-state carrier",
        flush=True,
    )


def test_t_config_fail_closed_and_legacy_reject(owner) -> None:
    protocol = owner.build_frontres_v015_persistent_corruption_protocol(
        corruption_id="persistent-rp",
        family="local_rp",
        seed=3,
        parameters={"scale": 0.5},
    )
    base = owner.FrontRESV015DeploymentCompositionConfig(
        enabled=True,
        source_reference_path="ordinary_motion.npz",
        reference_path="deployment_motion.npz",
        future_offsets=(1, 2),
        corruption_protocol=protocol,
    )
    base.validate()
    assert "enabled" in _expect_value_error(lambda: replace(base, enabled=False).validate())
    assert ".npz" in _expect_value_error(lambda: replace(base, reference_path="motion.pt").validate())
    assert "future_offsets" in _expect_value_error(lambda: replace(base, future_offsets=()).validate())
    try:
        replace(base, legacy_modes=("offline_eval",))
    except TypeError:
        pass
    else:
        raise AssertionError("retired legacy evaluator field remains constructible")
    print(
        "[T-config-fail-closed/T-legacy-reject] v015 composition cannot mix with the v002 sequence mode",
        flush=True,
    )


def test_t_controlled_carrier_materialization(owner) -> None:
    protocol = owner.build_frontres_v015_persistent_corruption_protocol(
        corruption_id="g4-controlled-rp-planar",
        family="planar+local_rp",
        seed=20260721,
        parameters={
            "xy_std": 0.04,
            "roll_std": 0.06,
            "pitch_std": 0.08,
            "scale": 1.0,
            "root_body_index": 0,
        },
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "ordinary_reference.npz"
        first_path = root / "carrier_a.npz"
        second_path = root / "carrier_b.npz"
        changed_path = root / "carrier_changed.npz"
        changed_source = root / "ordinary_reference_changed.npz"
        changed_source_path = root / "carrier_source_changed.npz"
        missing_root_path = root / "carrier_missing_root.npz"
        missing_scale_path = root / "carrier_missing_scale.npz"
        unknown_parameter_path = root / "carrier_unknown_parameter.npz"
        all_family_path = root / "carrier_all_families.npz"
        _write_npz(source, frame_count=6)

        lifecycle = owner.FrontRESV015DeploymentCarrierLifecycle(
            source_path=str(source),
            output_path=str(first_path),
            corruption_protocol=protocol,
        )
        first = lifecycle.materialize()
        first.validate()
        assert lifecycle.snapshot() is first
        assert first.source_reference_file_hash != first.carrier_file_hash
        assert first.source_reference_stream_id.startswith("ordinary-npz:")
        assert first.carrier_stream_id.startswith("deployment-carrier-npz:")
        assert first.corruption_protocol.protocol_hash == protocol.protocol_hash
        assert first.root_body_index == 0
        assert len(first.materialization_hash) == 64
        assert first.intent_q29_hash

        with np.load(source, allow_pickle=False) as source_data, np.load(first_path, allow_pickle=False) as carrier_data:
            assert set(carrier_data.files) == set(owner.FRONTRES_V015_REQUIRED_NPZ_ARRAYS)
            np.testing.assert_array_equal(carrier_data["joint_pos"], source_data["joint_pos"])
            np.testing.assert_array_equal(carrier_data["joint_vel"], source_data["joint_vel"])
            assert not np.array_equal(carrier_data["body_pos_w"], source_data["body_pos_w"])
            assert not np.array_equal(carrier_data["body_quat_w"], source_data["body_quat_w"])
            forbidden = ("corruption", "protocol", "seed", "label", "truth", "clean")
            assert all(not any(token in name.lower() for token in forbidden) for name in carrier_data.files)

        try:
            lifecycle.materialize()
        except RuntimeError as exc:
            assert "already sealed" in str(exc)
        else:
            raise AssertionError("G4 lifecycle resampled one sealed carrier")

        second = owner.FrontRESV015DeploymentCarrierLifecycle(
            source_path=str(source),
            output_path=str(second_path),
            corruption_protocol=protocol,
        ).materialize()
        assert second.carrier_file_hash == first.carrier_file_hash
        assert second.materialization_hash == first.materialization_hash
        assert second.intent_q29_hash == first.intent_q29_hash
        assert second.materialized_delta_se3 == first.materialized_delta_se3

        changed_protocol = owner.build_frontres_v015_persistent_corruption_protocol(
            corruption_id="g4-controlled-rp-planar",
            family="planar+local_rp",
            seed=20260722,
            parameters={
                "xy_std": 0.04,
                "roll_std": 0.06,
                "pitch_std": 0.08,
                "scale": 1.0,
                "root_body_index": 0,
            },
        )
        changed = owner.materialize_frontres_v015_deployment_carrier(
            source_path=str(source),
            output_path=str(changed_path),
            corruption_protocol=changed_protocol,
        )
        assert changed.carrier_file_hash != first.carrier_file_hash
        assert changed.materialization_hash != first.materialization_hash
        _write_npz(changed_source, frame_count=6, offset=1.0)
        source_changed = owner.materialize_frontres_v015_deployment_carrier(
            source_path=str(changed_source),
            output_path=str(changed_source_path),
            corruption_protocol=protocol,
        )
        assert source_changed.source_reference_file_hash != first.source_reference_file_hash
        assert source_changed.carrier_file_hash != first.carrier_file_hash
        assert source_changed.materialization_hash != first.materialization_hash
        missing_root = owner.build_frontres_v015_persistent_corruption_protocol(
            corruption_id="g4-missing-root",
            family="local_rp",
            seed=1,
            parameters={"roll_std": 0.1},
        )
        assert "root_body_index" in _expect_value_error(
            lambda: owner.materialize_frontres_v015_deployment_carrier(
                source_path=str(source),
                output_path=str(missing_root_path),
                corruption_protocol=missing_root,
            )
        )
        assert not missing_root_path.exists()
        missing_scale = owner.build_frontres_v015_persistent_corruption_protocol(
            corruption_id="g4-missing-scale",
            family="local_rp",
            seed=1,
            parameters={"root_body_index": 0},
        )
        assert "roll_std" in _expect_value_error(
            lambda: owner.materialize_frontres_v015_deployment_carrier(
                source_path=str(source),
                output_path=str(missing_scale_path),
                corruption_protocol=missing_scale,
            )
        )
        unknown_parameter = owner.build_frontres_v015_persistent_corruption_protocol(
            corruption_id="g4-unknown-parameter",
            family="local_rp",
            seed=1,
            parameters={"root_body_index": 0, "roll_std": 0.1, "noise_label": "rp"},
        )
        assert "unknown corruption parameters" in _expect_value_error(
            lambda: owner.materialize_frontres_v015_deployment_carrier(
                source_path=str(source),
                output_path=str(unknown_parameter_path),
                corruption_protocol=unknown_parameter,
            )
        )
        assert not missing_scale_path.exists()
        assert not unknown_parameter_path.exists()
        all_family_protocol = owner.build_frontres_v015_persistent_corruption_protocol(
            corruption_id="g4-all-families",
            family="yaw+global_z+planar+local_rp",
            seed=31,
            parameters={
                "xy_std": 0.02,
                "z_std": 0.03,
                "roll_std": 0.04,
                "pitch_std": 0.05,
                "yaw_std": 0.06,
                "root_body_index": 0,
            },
        )
        all_family = owner.materialize_frontres_v015_deployment_carrier(
            source_path=str(source),
            output_path=str(all_family_path),
            corruption_protocol=all_family_protocol,
        )
        assert all(abs(value) > 0.0 for value in all_family.materialized_delta_se3)
        assert all_family.intent_q29_hash == first.intent_q29_hash
        assert "exists" in _expect_value_error(
            lambda: owner.materialize_frontres_v015_deployment_carrier(
                source_path=str(source),
                output_path=str(first_path),
                corruption_protocol=protocol,
            )
        )
    print(
        "[T-materialize/T-hash/T-determinism/T-q29-invariant/T-no-label/T-no-resample] "
        "ordinary NPZ + fixed protocol yields one deterministic immutable carrier",
        flush=True,
    )


def main() -> None:
    owner = _load_owner()
    test_t_npz_schema_identity_and_corruption_protocol(owner)
    test_t_report_and_no_feedback(owner)
    test_t_config_fail_closed_and_legacy_reject(owner)
    test_t_controlled_carrier_materialization(owner)
    print("frontres_v015_deployment_composition_s1_contract: ok", flush=True)


if __name__ == "__main__":
    main()
