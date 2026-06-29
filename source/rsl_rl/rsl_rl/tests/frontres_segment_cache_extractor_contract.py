#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import tempfile

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


extractor = _load_module(
    "frontres_segment_cache_extractor",
    "source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_extractor.py",
)
cache_io = _load_module(
    "frontres_segment_cache_io",
    "source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_io.py",
)

FrontRESCleanStateEntry = cache_io.FrontRESCleanStateEntry
FrontRESSegmentIndex = cache_io.FrontRESSegmentIndex


class FakeRobotData:
    def __init__(self) -> None:
        batch = 2
        dofs = 29
        bodies = 30
        self.root_pos_w = _grad_tensor(torch.arange(batch * 3, dtype=torch.float32).view(batch, 3))
        self.root_quat_w = _grad_tensor(
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        )
        self.root_lin_vel_w = _grad_tensor(torch.full((batch, 3), 0.1))
        self.root_ang_vel_w = _grad_tensor(torch.full((batch, 3), 0.2))
        self.joint_pos = _grad_tensor(torch.arange(batch * dofs, dtype=torch.float32).view(batch, dofs))
        self.joint_vel = _grad_tensor(torch.arange(batch * dofs, dtype=torch.float32).view(batch, dofs) * 0.01)
        self.body_pos_w = _grad_tensor(torch.arange(batch * bodies * 3, dtype=torch.float32).view(batch, bodies, 3))
        self.body_quat_w = _grad_tensor(torch.zeros(batch, bodies, 4).index_fill(2, torch.tensor([0]), 1.0))
        self.body_lin_vel_w = _grad_tensor(torch.full((batch, bodies, 3), 0.3))
        self.body_ang_vel_w = _grad_tensor(torch.full((batch, bodies, 3), 0.4))


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


def _grad_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone().requires_grad_(True)


def _segment() -> FrontRESSegmentIndex:
    return FrontRESSegmentIndex(
        segment_id=7,
        motion_rel_path="KIT/359/amass_g1_go_over_beam09_poses.npz",
        motion_num_frames=437,
        fps=30.0,
        start_frame=10,
        horizon_k=4,
    )


def test_extractor_selects_env_id_and_detaches_state() -> None:
    env = FakeEnv()
    contact_state = _grad_tensor(torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]))
    action_history = _grad_tensor(torch.arange(2 * 2 * 29, dtype=torch.float32).view(2, 2, 29))

    state = extractor.extract_robot_rollout_state(
        env,
        env_ids=torch.tensor([1]),
        contact_state=contact_state,
        action_history=action_history,
    )
    probe = extractor.robot_state_probe(state)
    print(
        "[cache_extractor trace] extract "
        f"env_ids={[1]} "
        f"root_pos_shape={probe['extracted_state.root_pos_shape']} "
        f"joint_pos_shape={probe['extracted_state.joint_pos_shape']} "
        f"body_pos_shape={probe['extracted_state.body_pos_shape']} "
        f"finite={probe['extracted_state.finite']} "
        f"requires_grad={probe['extracted_state.requires_grad']} "
        f"root_pos={state.root_pos.flatten().tolist()}"
    )
    assert probe["extracted_state.batch_size"] == 1
    assert probe["extracted_state.num_dofs"] == 29
    assert probe["extracted_state.num_bodies"] == 30
    assert probe["extracted_state.finite"] is True
    assert probe["extracted_state.requires_grad"] is False
    torch.testing.assert_close(state.root_pos, env.scene.robot.data.root_pos_w[1:2].detach())
    torch.testing.assert_close(state.joint_pos, env.scene.robot.data.joint_pos[1:2].detach())
    torch.testing.assert_close(state.body_pos_w, env.scene.robot.data.body_pos_w[1:2].detach())
    torch.testing.assert_close(state.contact_state, contact_state[1:2].detach())
    torch.testing.assert_close(state.action_history, action_history[1:2].detach())


def test_extracted_clean_state_round_trips_through_cache_io() -> None:
    env = FakeEnv()
    state = extractor.extract_robot_rollout_state(env, env_ids=[1])
    entry = FrontRESCleanStateEntry(segment=_segment(), clean_state=state)
    with tempfile.TemporaryDirectory() as tmp:
        path = cache_io.write_clean_state_shard(tmp, [entry], shard_id=0)
        loaded = cache_io.read_clean_state_shard(path)
    print(
        "[cache_extractor trace] clean_roundtrip "
        f"segment_id={loaded[0].segment_id} "
        f"root_pos_shape={loaded[0].clean_state.root_pos.shape} "
        f"body_pos_shape={loaded[0].clean_state.body_pos_w.shape} "
        f"requires_grad={loaded[0].clean_state.root_pos.requires_grad}"
    )
    assert len(loaded) == 1
    assert loaded[0].segment_id == 7
    torch.testing.assert_close(loaded[0].clean_state.root_pos, state.root_pos)
    torch.testing.assert_close(loaded[0].clean_state.body_pos_w, state.body_pos_w)
    assert loaded[0].clean_state.root_pos.requires_grad is False


if __name__ == "__main__":
    test_extractor_selects_env_id_and_detaches_state()
    test_extracted_clean_state_round_trips_through_cache_io()
    print("PASS: FrontRES clean state extractor captures detached robot rollout state.")
