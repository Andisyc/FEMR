#!/usr/bin/env python3
"""Deterministic S1 alignment for trajectory -> relational Outcome."""

from __future__ import annotations

from dataclasses import replace

import torch

from rsl_rl.frontres.frontres_hierarchical_gain_candidate import compare
from rsl_rl.frontres.frontres_relational_outcome import build_frontres_relational_outcome
from rsl_rl.frontres.frontres_segment_evidence import FrontRESExecutedKTrajectory


def _trajectory(*, root_x: float = 0.0, joint_error: float = 0.0) -> FrontRESExecutedKTrajectory:
    k = 4
    root = torch.zeros(k, 1, 3)
    root[..., 0] = root_x
    root[..., 2] = 0.8
    feet = torch.zeros(k, 1, 2, 3)
    feet[:, 0, 0, 0] = -0.10
    feet[:, 0, 1, 0] = 0.10
    quat = torch.zeros(k, 1, 4)
    quat[..., 0] = 1.0
    return FrontRESExecutedKTrajectory(
        joint_pos=torch.full((k, 1, 29), joint_error),
        root_pos=root,
        root_quat=quat,
        key_body_pos=root.unsqueeze(2).repeat(1, 1, 2, 1),
        root_lin_vel=torch.zeros(k, 1, 3),
        root_ang_vel=torch.zeros(k, 1, 3),
        foot_pos=feet,
        contact=torch.ones(k, 1, 2),
        zmp_margin=torch.full((k, 1), 0.03),
        survival=torch.ones(k, 1),
        valid_mask=torch.ones(k, 1, dtype=torch.bool),
        env_origin=torch.zeros(k, 1, 3),
    )


def _outcome(clean, repair, expected, action=None):
    return build_frontres_relational_outcome(
        clean=clean,
        repair=repair,
        expected_support=expected,
        repair_action=torch.zeros(6) if action is None else action,
    )


def main() -> None:
    clean = _trajectory()
    expected = torch.ones(4, 1, 2)
    stable = _outcome(clean, _trajectory(), expected)
    assert compare(stable, stable).left_level == "L3_ADMISSIBLE_STABLE"

    no_load_traj = _trajectory()
    no_load_traj = replace(
        no_load_traj,
        contact=torch.tensor([[[1.0, 1.0]], [[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 1.0]]]),
        zmp_margin=torch.full((4, 1), 0.03),
    )
    no_load = _outcome(clean, no_load_traj, expected)
    assert compare(no_load, no_load).left_level == "L1_CONTACT_INVALID"
    assert compare(stable, no_load).relation == "BETTER"

    failed_traj = replace(_trajectory(), survival=torch.tensor([[1.0], [1.0], [0.0], [0.0]]))
    failed = _outcome(clean, failed_traj, expected)
    assert compare(failed, failed).left_level == "L0_PHYSICS_FAILED"
    assert compare(no_load, failed).relation == "BETTER"

    better_intent = _outcome(clean, _trajectory(joint_error=0.01), expected, torch.ones(6) * 0.01)
    worse_intent = _outcome(clean, _trajectory(joint_error=0.05), expected, torch.ones(6) * 0.01)
    assert compare(better_intent, worse_intent).relation == "BETTER"

    missing_origin = replace(_trajectory(), env_origin=None)
    try:
        _outcome(clean, missing_origin, expected)
    except ValueError as error:
        assert "environment origins" in str(error)
    else:
        raise AssertionError("missing environment origin did not fail closed")
    print("frontres_relational_outcome_alignment: OBJECTIVE-ALIGNED", flush=True)


if __name__ == "__main__":
    main()
