#!/usr/bin/env python3
"""Deterministic S1 contract for the v015 deployment-composition kernel."""

from __future__ import annotations

from dataclasses import fields, replace
import importlib.util
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OWNER_PATH = ROOT / "rsl_rl" / "runners" / "frontres_segment_sequence_eval.py"


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
            reference_path=str(reference),
            future_offsets=(1, 2),
            corruption_protocol=protocol_a,
        )
        request_a = owner.load_frontres_v015_deployment_composition_request(config)
        request_b = owner.load_frontres_v015_deployment_composition_request(config)
        request_a.validate()
        assert request_a == request_b
        assert request_a.reference_path == str(reference.resolve())
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
                reference_path=str(reference),
                future_offsets=(1, 2),
                corruption_protocol=protocol,
            )
        )
        report = owner.FrontRESV015DeploymentCompositionReport(
            request=request,
            per_frame_femr_action_used=(True, True),
            per_frame_intent_q29_error=(0.2, 0.1),
            per_frame_physics_success=(True, False),
            per_frame_fall=(False, True),
            per_frame_zmp_margin=(0.1, -0.03),
            per_frame_contact_consistency=(1.0, 0.2),
        )
        report.validate()
        assert report.reference_frame_count == 4
        assert report.frame_count == 2
        assert report.femr_action_count == 2
        assert report.accumulated_failure_count == 1
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
            lambda: replace(report, per_frame_fall=(False,)).validate()
        )
        assert "success and fall" in _expect_value_error(
            lambda: replace(
                report,
                per_frame_physics_success=(True, True),
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
        reference_path="deployment_motion.npz",
        future_offsets=(1, 2),
        corruption_protocol=protocol,
    )
    base.validate()
    assert "enabled" in _expect_value_error(lambda: replace(base, enabled=False).validate())
    assert ".npz" in _expect_value_error(lambda: replace(base, reference_path="motion.pt").validate())
    assert "future_offsets" in _expect_value_error(lambda: replace(base, future_offsets=()).validate())
    assert "legacy" in _expect_value_error(
        lambda: replace(
            base,
            legacy_modes=("frontres_segment_sequence_offline_eval_only",),
        ).validate()
    )
    print(
        "[T-config-fail-closed/T-legacy-reject] v015 composition cannot mix with the v002 sequence mode",
        flush=True,
    )


def main() -> None:
    owner = _load_owner()
    test_t_npz_schema_identity_and_corruption_protocol(owner)
    test_t_report_and_no_feedback(owner)
    test_t_config_fail_closed_and_legacy_reject(owner)
    print("frontres_v015_deployment_composition_s1_contract: ok", flush=True)


if __name__ == "__main__":
    main()
