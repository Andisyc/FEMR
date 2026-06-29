#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import torch


ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "frontres" / "frontres_segment_cache_schema.py"
spec = importlib.util.spec_from_file_location("frontres_segment_cache_schema", MODULE_PATH)
schema = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = schema
spec.loader.exec_module(schema)

FrontRESSegmentIndex = schema.FrontRESSegmentIndex
FrontRESRobotRolloutState = schema.FrontRESRobotRolloutState
FrontRESPerturbationDescriptor = schema.FrontRESPerturbationDescriptor
FrontRESNoisyVariant = schema.FrontRESNoisyVariant


def _state(offset: float = 0.0) -> FrontRESRobotRolloutState:
    batch = 1
    num_dofs = 29
    num_bodies = 30
    return FrontRESRobotRolloutState(
        root_pos=torch.full((batch, 3), 1.0 + offset),
        root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        root_lin_vel=torch.full((batch, 3), 0.1 + offset),
        root_ang_vel=torch.full((batch, 3), 0.2 + offset),
        joint_pos=torch.arange(num_dofs, dtype=torch.float32).view(batch, num_dofs) + offset,
        joint_vel=torch.arange(num_dofs, dtype=torch.float32).view(batch, num_dofs) * 0.01 + offset,
        body_pos_w=torch.ones(batch, num_bodies, 3) + offset,
        body_quat_w=torch.zeros(batch, num_bodies, 4).index_fill(2, torch.tensor([0]), 1.0),
        body_lin_vel_w=torch.full((batch, num_bodies, 3), 0.3 + offset),
        body_ang_vel_w=torch.full((batch, num_bodies, 3), 0.4 + offset),
        contact_state=torch.zeros(batch, 4),
        action_history=torch.zeros(batch, 2, num_dofs),
    )


def test_schema_traces_segment_and_perturbation_ids() -> None:
    segment = FrontRESSegmentIndex(
        segment_id=7,
        motion_rel_path="KIT/359/amass_g1_go_over_beam09_poses.npz",
        motion_num_frames=437,
        fps=30.0,
        start_frame=10,
        horizon_k=4,
    )
    descriptor = FrontRESPerturbationDescriptor(
        perturbation_id=3,
        segment_id=segment.segment_id,
        strength=0.5,
        seed=1234,
        family="external_push",
        start_step=0,
        duration=2,
        target="torso_link",
        frame="world",
        params={"axis": [1.0, 0.0, 0.0], "magnitude": 0.5},
    )
    variant = FrontRESNoisyVariant(
        segment=segment,
        descriptor=descriptor,
        noisy_state=_state(offset=0.5),
        noisy_baseline_score=torch.tensor([0.25], dtype=torch.float32),
        noisy_fall=torch.tensor([0.0], dtype=torch.float32),
        noisy_rollout_len=torch.tensor([4.0], dtype=torch.float32),
    )

    segment.validate()
    descriptor.validate()
    probe = variant.probe()
    print(
        "[cache_schema trace] "
        f"segment.segment_id={segment.segment_id} "
        f"descriptor.segment_id={descriptor.segment_id} "
        f"variant.segment_id={variant.segment_id} "
        f"descriptor.perturbation_id={descriptor.perturbation_id} "
        f"variant.perturbation_id={variant.perturbation_id} "
        f"state_shape={probe['noisy_state.body_pos_shape']} "
        f"finite={probe['noisy_state.finite']} "
        f"requires_grad={probe['noisy_state.requires_grad']}"
    )
    assert probe["segment_id"] == 7
    assert probe["descriptor.segment_id"] == 7
    assert probe["perturbation_id"] == 3
    assert probe["noisy_state.batch_size"] == 1
    assert probe["noisy_state.num_dofs"] == 29
    assert probe["noisy_state.num_bodies"] == 30
    assert probe["noisy_state.finite"] is True
    assert probe["noisy_state.requires_grad"] is False
    assert probe["baseline_score_shape"] == (1,)


def test_schema_rejects_id_drift() -> None:
    segment = FrontRESSegmentIndex(
        segment_id=7,
        motion_rel_path="KIT/359/amass_g1_go_over_beam09_poses.npz",
        motion_num_frames=437,
        fps=30.0,
        start_frame=10,
        horizon_k=4,
    )
    descriptor = FrontRESPerturbationDescriptor(
        perturbation_id=3,
        segment_id=8,
        strength=0.5,
        seed=1234,
        family="external_push",
        start_step=0,
        duration=2,
        target="torso_link",
        frame="world",
        params={},
    )
    variant = FrontRESNoisyVariant(
        segment=segment,
        descriptor=descriptor,
        noisy_state=_state(),
        noisy_baseline_score=torch.tensor([0.25], dtype=torch.float32),
        noisy_fall=torch.tensor([0.0], dtype=torch.float32),
        noisy_rollout_len=torch.tensor([4.0], dtype=torch.float32),
    )
    try:
        variant.validate()
    except ValueError as exc:
        print(f"[cache_schema trace] rejected_id_drift={exc}")
        assert "does not match" in str(exc)
        return
    raise AssertionError("id drift should be rejected")


def test_schema_rejects_grad_cache_state() -> None:
    state = _state()
    bad_state = FrontRESRobotRolloutState(
        root_pos=state.root_pos.clone().requires_grad_(True),
        root_quat=state.root_quat,
        root_lin_vel=state.root_lin_vel,
        root_ang_vel=state.root_ang_vel,
        joint_pos=state.joint_pos,
        joint_vel=state.joint_vel,
        body_pos_w=state.body_pos_w,
        body_quat_w=state.body_quat_w,
        body_lin_vel_w=state.body_lin_vel_w,
        body_ang_vel_w=state.body_ang_vel_w,
    )
    try:
        bad_state.validate(name="bad_state")
    except ValueError as exc:
        print(f"[cache_schema trace] rejected_grad_state={exc}")
        assert "detached cache data" in str(exc)
        return
    raise AssertionError("cache state with requires_grad should be rejected")


if __name__ == "__main__":
    test_schema_traces_segment_and_perturbation_ids()
    test_schema_rejects_id_drift()
    test_schema_rejects_grad_cache_state()
    print("PASS: FrontRES Segment cache schema validates ids and rollout state tensors.")
