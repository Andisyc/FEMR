#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import torch


ROOT = Path(__file__).resolve().parents[4]


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


capture = _load_module(
    "frontres_segment_cache_noisy_capture",
    "source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_noisy_capture.py",
)

FrontRESPerturbationDescriptor = capture.FrontRESPerturbationDescriptor
FrontRESRobotRolloutState = capture.FrontRESRobotRolloutState
FrontRESSegmentIndex = capture.FrontRESSegmentIndex


class FakeRobotData:
    def __init__(self) -> None:
        batch = 2
        dofs = 29
        bodies = 30
        self.root_pos_w = torch.zeros(batch, 3)
        self.root_quat_w = torch.zeros(batch, 4)
        self.root_quat_w[:, 0] = 1.0
        self.root_lin_vel_w = torch.zeros(batch, 3)
        self.root_ang_vel_w = torch.zeros(batch, 3)
        self.joint_pos = torch.zeros(batch, dofs)
        self.joint_vel = torch.zeros(batch, dofs)
        self.body_pos_w = torch.zeros(batch, bodies, 3)
        self.body_quat_w = torch.zeros(batch, bodies, 4)
        self.body_quat_w[:, :, 0] = 1.0
        self.body_lin_vel_w = torch.zeros(batch, bodies, 3)
        self.body_ang_vel_w = torch.zeros(batch, bodies, 3)


class FakeRobot:
    def __init__(self) -> None:
        self.data = FakeRobotData()


class FakeScene:
    def __init__(self) -> None:
        self.robot = FakeRobot()

    def __getitem__(self, name: str):
        if name != "robot":
            raise KeyError(name)
        return self.robot


class FakeEnv:
    def __init__(self) -> None:
        self.scene = FakeScene()
        self.unwrapped = self
        self.reset_calls: list[list[int]] = []
        self.perturbation_calls: list[tuple[int, list[int]]] = []
        self.baseline_calls: list[tuple[int, int, list[int]]] = []

    def set_frontres_rollout_state(self, *, clean_state: FrontRESRobotRolloutState, env_ids: torch.Tensor):
        ids = env_ids.cpu().long()
        data = self.scene.robot.data
        self.reset_calls.append(ids.tolist())
        data.root_pos_w[ids] = clean_state.root_pos.cpu()
        data.root_quat_w[ids] = clean_state.root_quat.cpu()
        data.root_lin_vel_w[ids] = clean_state.root_lin_vel.cpu()
        data.root_ang_vel_w[ids] = clean_state.root_ang_vel.cpu()
        data.joint_pos[ids] = clean_state.joint_pos.cpu()
        data.joint_vel[ids] = clean_state.joint_vel.cpu()
        data.body_pos_w[ids] = clean_state.body_pos_w.cpu()
        data.body_quat_w[ids] = clean_state.body_quat_w.cpu()
        data.body_lin_vel_w[ids] = clean_state.body_lin_vel_w.cpu()
        data.body_ang_vel_w[ids] = clean_state.body_ang_vel_w.cpu()
        return {"success": torch.ones(ids.numel(), dtype=torch.bool)}

    def apply_frontres_segment_perturbation(
        self,
        *,
        descriptor: FrontRESPerturbationDescriptor,
        env_ids: torch.Tensor,
    ):
        ids = env_ids.cpu().long()
        data = self.scene.robot.data
        axis = torch.tensor(descriptor.params["axis"], dtype=torch.float32).view(1, 3)
        magnitude = float(descriptor.params["signed_magnitude"])
        delta = axis * magnitude
        self.perturbation_calls.append((descriptor.perturbation_id, ids.tolist()))
        data.root_pos_w[ids] = data.root_pos_w[ids] + delta
        data.root_lin_vel_w[ids] = data.root_lin_vel_w[ids] + delta
        data.body_pos_w[ids, 0, :] = data.body_pos_w[ids, 0, :] + delta
        return {"perturbation_success": torch.ones(ids.numel(), dtype=torch.bool)}

    def rollout_frontres_noisy_baseline(
        self,
        *,
        segment: FrontRESSegmentIndex,
        descriptor: FrontRESPerturbationDescriptor,
        env_ids: torch.Tensor,
    ):
        ids = env_ids.cpu().long()
        self.baseline_calls.append((segment.segment_id, descriptor.perturbation_id, ids.tolist()))
        score = 1.0 - torch.linalg.vector_norm(self.scene.robot.data.root_lin_vel_w[ids], dim=-1)
        return {
            "score": score.detach(),
            "fall": torch.zeros(ids.numel(), dtype=torch.float32),
            "rollout_len": torch.full((ids.numel(),), float(segment.horizon_k), dtype=torch.float32),
        }


def _clean_state() -> FrontRESRobotRolloutState:
    batch = 1
    dofs = 29
    bodies = 30
    return FrontRESRobotRolloutState(
        root_pos=torch.tensor([[1.0, 2.0, 3.0]]),
        root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        root_lin_vel=torch.tensor([[0.1, 0.0, 0.0]]),
        root_ang_vel=torch.tensor([[0.0, 0.1, 0.0]]),
        joint_pos=torch.arange(dofs, dtype=torch.float32).view(batch, dofs),
        joint_vel=torch.full((batch, dofs), 0.01),
        body_pos_w=torch.full((batch, bodies, 3), 2.0),
        body_quat_w=torch.zeros(batch, bodies, 4).index_fill(2, torch.tensor([0]), 1.0),
        body_lin_vel_w=torch.full((batch, bodies, 3), 0.2),
        body_ang_vel_w=torch.full((batch, bodies, 3), 0.3),
    )


def _segment() -> FrontRESSegmentIndex:
    return FrontRESSegmentIndex(
        segment_id=7,
        motion_rel_path="KIT/359/motion_a.npz",
        motion_num_frames=20,
        fps=30.0,
        start_frame=3,
        horizon_k=4,
    )


def _descriptor(segment_id: int = 7) -> FrontRESPerturbationDescriptor:
    return FrontRESPerturbationDescriptor(
        perturbation_id=3,
        segment_id=segment_id,
        strength=0.5,
        seed=123,
        family="external_push",
        start_step=0,
        duration=2,
        target="torso_link",
        frame="world",
        params={"axis": [1.0, 0.0, 0.0], "signed_magnitude": 0.5, "frame": "world"},
    )


def test_noisy_capture_resets_perturbs_extracts_and_builds_variant() -> None:
    env = FakeEnv()
    result = capture.capture_noisy_variant(
        env,
        segment=_segment(),
        clean_state=_clean_state(),
        descriptor=_descriptor(),
        env_ids=torch.tensor([1]),
    )
    probe = result.probe()
    print(
        "[cache_noisy_capture trace] capture "
        f"segment_id={probe['segment_id']} perturbation_id={probe['perturbation_id']} "
        f"reset_success={probe['reset_success_count']} perturb_success={probe['perturbation_success_count']} "
        f"baseline_mean={probe['baseline_score_mean']:.4f} rollout_len={probe['rollout_len_mean']} "
        f"root_shape={probe['noisy_root_pos_shape']} body_shape={probe['noisy_body_pos_shape']} "
        f"requires_grad={probe['noisy_requires_grad']} "
        f"reset_calls={env.reset_calls} perturb_calls={env.perturbation_calls} baseline_calls={env.baseline_calls}"
    )
    assert probe["segment_id"] == 7
    assert probe["perturbation_id"] == 3
    assert probe["reset_success_count"] == 1
    assert probe["perturbation_success_count"] == 1
    assert probe["noisy_requires_grad"] is False
    assert env.reset_calls == [[1]]
    assert env.perturbation_calls == [(3, [1])]
    assert env.baseline_calls == [(7, 3, [1])]
    torch.testing.assert_close(result.variant.noisy_state.root_pos, torch.tensor([[1.5, 2.0, 3.0]]))
    torch.testing.assert_close(result.variant.noisy_state.root_lin_vel, torch.tensor([[0.6, 0.0, 0.0]]))
    assert result.variant.noisy_baseline_score.shape == (1,)


def test_noisy_capture_rejects_descriptor_segment_drift() -> None:
    env = FakeEnv()
    try:
        capture.capture_noisy_variant(
            env,
            segment=_segment(),
            clean_state=_clean_state(),
            descriptor=_descriptor(segment_id=8),
            env_ids=[1],
        )
    except ValueError as exc:
        print(f"[cache_noisy_capture trace] rejected_descriptor_drift={exc}")
        assert "does not match" in str(exc)
        return
    raise AssertionError("descriptor segment drift should be rejected")


if __name__ == "__main__":
    test_noisy_capture_resets_perturbs_extracts_and_builds_variant()
    test_noisy_capture_rejects_descriptor_segment_drift()
    print("PASS: FrontRES noisy capture interface builds noisy variants through reset and perturbation hooks.")
