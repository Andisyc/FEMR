#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reset_module = _load(
    "frontres_segment_reset_for_live_reset_hook_contract",
    ROOT / "rsl_rl" / "frontres" / "frontres_segment_reset.py",
)

FrontRESSegmentResetAdapter = reset_module.FrontRESSegmentResetAdapter
FrontRESSegmentResetRequest = reset_module.FrontRESSegmentResetRequest
ensure_frontres_segment_live_reset_hook = reset_module.ensure_frontres_segment_live_reset_hook


class FakeRobotData:
    def __init__(self, num_envs: int = 2, dofs: int = 4) -> None:
        self.root_pos_w = torch.zeros(num_envs, 3)
        self.root_quat_w = torch.zeros(num_envs, 4)
        self.root_quat_w[:, 0] = 1.0
        self.root_lin_vel_w = torch.zeros(num_envs, 3)
        self.root_ang_vel_w = torch.zeros(num_envs, 3)
        self.joint_pos = torch.zeros(num_envs, dofs)
        self.joint_vel = torch.zeros(num_envs, dofs)


class FakeRobot:
    def __init__(self) -> None:
        self.data = FakeRobotData()
        self.root_writes: list[torch.Tensor] = []
        self.joint_writes: list[tuple[torch.Tensor, torch.Tensor]] = []

    def write_root_state_to_sim(self, root_state: torch.Tensor, *, env_ids: torch.Tensor) -> None:
        self.root_writes.append(root_state.detach().clone())
        self.data.root_pos_w[env_ids] = root_state[:, 0:3]
        self.data.root_quat_w[env_ids] = root_state[:, 3:7]
        self.data.root_lin_vel_w[env_ids] = root_state[:, 7:10]
        self.data.root_ang_vel_w[env_ids] = root_state[:, 10:13]

    def write_joint_state_to_sim(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, *, env_ids: torch.Tensor) -> None:
        self.joint_writes.append((joint_pos.detach().clone(), joint_vel.detach().clone()))
        self.data.joint_pos[env_ids] = joint_pos
        self.data.joint_vel[env_ids] = joint_vel


class FakeScene:
    def __init__(self, robot: FakeRobot) -> None:
        self.robot = robot

    def __getitem__(self, name: str):
        assert name == "robot"
        return self.robot


class FakePerturber:
    def __init__(self) -> None:
        self.reset_calls: list[list[int]] = []

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        self.reset_calls.append(env_ids.detach().cpu().tolist())


class FakeCommand:
    def __init__(self) -> None:
        self._frontres_pos_correction = torch.ones(2, 3)
        self._frontres_quat_correction = torch.zeros(2, 4)
        self._frontres_quat_correction[:, 2] = 0.5
        self.perturber = FakePerturber()
        self.reference_window_writes: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.reference_window = torch.zeros(2, 3, 6)

    def set_frontres_reference_window(self, reference_window: torch.Tensor, *, env_ids: torch.Tensor) -> torch.Tensor:
        self.reference_window_writes.append((reference_window.detach().clone(), env_ids.detach().clone()))
        self.reference_window[env_ids] = reference_window
        return torch.ones(int(env_ids.numel()), dtype=torch.bool, device=env_ids.device)


class FakeCommandManager:
    def __init__(self, command: FakeCommand) -> None:
        self.command = command

    def get_term(self, name: str):
        assert name == "motion"
        return self.command


class FakeBaseEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.robot = FakeRobot()
        self.scene = FakeScene(self.robot)
        self.command = FakeCommand()
        self.command_manager = FakeCommandManager(self.command)
        self.episode_length_buf = torch.tensor([5, 6], dtype=torch.long)


class FakeVecWrapper:
    def __init__(self) -> None:
        self.unwrapped = FakeBaseEnv()
        self.episode_length_buf = torch.tensor([7, 8], dtype=torch.long)


def _request() -> object:
    return FrontRESSegmentResetRequest(
        segment_ids=torch.tensor([11, 13], dtype=torch.long),
        root_pos=torch.tensor([[1.0, 2.0, 0.5], [3.0, 4.0, 0.7]], dtype=torch.float32),
        root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        root_lin_vel=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32),
        root_ang_vel=torch.tensor([[0.0, 0.1, 0.0], [0.0, 0.2, 0.0]], dtype=torch.float32),
        dof_pos=torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=torch.float32),
        dof_vel=torch.tensor([[0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 0.07, 0.08]], dtype=torch.float32),
        reference_window=torch.tensor(
            [
                [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [1.0, 1.1, 1.2, 1.3, 1.4, 1.5], [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]],
                [[3.0, 3.1, 3.2, 3.3, 3.4, 3.5], [4.0, 4.1, 4.2, 4.3, 4.4, 4.5], [5.0, 5.1, 5.2, 5.3, 5.4, 5.5]],
            ],
            dtype=torch.float32,
        ),
        mode=("direct", "direct"),
        preroll_steps=torch.zeros(2, dtype=torch.long),
        valid_mask=torch.ones(2, dtype=torch.bool),
    )


def _probe_tensor(name: str, tensor: torch.Tensor, semantic: str) -> None:
    data = tensor.detach().cpu()
    numeric = data.float() if data.dtype == torch.bool else data
    print(
        f"[probe step13] {name}: shape={tuple(data.shape)} dtype={data.dtype} "
        f"min={numeric.min().item():.6f} max={numeric.max().item():.6f} "
        f"mean={numeric.float().mean().item():.6f} semantic={semantic}",
        flush=True,
    )


def test_live_reset_hook_writes_dynamic_state_through_wrapper() -> None:
    env = FakeVecWrapper()
    request = _request()
    hook = ensure_frontres_segment_live_reset_hook(env, trace=True)
    assert ensure_frontres_segment_live_reset_hook(env, trace=True) is hook
    result = FrontRESSegmentResetAdapter().apply(env, request)

    robot = env.unwrapped.robot
    command = env.unwrapped.command
    _probe_tensor("request.root_pos", request.root_pos, "dynamic reset root position from sampled segment batch")
    _probe_tensor("robot.root_pos_w", robot.data.root_pos_w, "root position after live reset hook write")
    _probe_tensor("request.dof_vel", request.dof_vel, "dynamic reset joint velocity from sampled segment batch")
    _probe_tensor("robot.joint_vel", robot.data.joint_vel, "joint velocity after live reset hook write")
    _probe_tensor("request.reference_window", request.reference_window, "sampled GMT reference window for this segment batch")
    _probe_tensor("command.reference_window", command.reference_window, "reference window after command hook write")
    _probe_tensor("result.success_mask", result.success_mask, "adapter result consumed by live storage and sampler masks")
    print(
        "[probe step13] live_reset_hook_summary: "
        f"has_hook={hasattr(env, 'apply_frontres_segment_reset')} "
        f"root_write_count={len(robot.root_writes)} "
        f"joint_write_count={len(robot.joint_writes)} "
        f"wrapper_episode={env.episode_length_buf.tolist()} "
        f"base_episode={env.unwrapped.episode_length_buf.tolist()} "
        f"pos_correction={command._frontres_pos_correction.tolist()} "
        f"quat_correction={command._frontres_quat_correction.tolist()} "
        f"perturber_reset={command.perturber.reset_calls} "
        f"reference_write_count={len(command.reference_window_writes)} "
        f"reference_applied_frac={result.diagnostics.get('reference_window_applied_frac')} "
        f"velocity_mismatch={result.velocity_mismatch.tolist()}",
        flush=True,
    )
    print(
        "[probe step15] reference_window_path: "
        f"segment_ids={request.segment_ids.tolist()} "
        f"request_shape={tuple(request.reference_window.shape)} "
        f"write_count={len(command.reference_window_writes)} "
        f"written_env_ids={command.reference_window_writes[0][1].tolist()} "
        f"applied_frac={result.diagnostics.get('reference_window_applied_frac')} "
        f"first_row_mean={float(command.reference_window[0].float().mean().item()):.6f} "
        f"second_row_mean={float(command.reference_window[1].float().mean().item()):.6f}",
        flush=True,
    )

    assert hasattr(env, "apply_frontres_segment_reset")
    assert len(robot.root_writes) == 1
    assert len(robot.joint_writes) == 1
    torch.testing.assert_close(robot.data.root_pos_w, request.root_pos)
    torch.testing.assert_close(robot.data.root_quat_w, request.root_quat)
    torch.testing.assert_close(robot.data.root_lin_vel_w, request.root_lin_vel)
    torch.testing.assert_close(robot.data.root_ang_vel_w, request.root_ang_vel)
    torch.testing.assert_close(robot.data.joint_pos, request.dof_pos)
    torch.testing.assert_close(robot.data.joint_vel, request.dof_vel)
    assert result.success_mask.tolist() == [True, True]
    torch.testing.assert_close(result.velocity_mismatch, torch.zeros(2))
    assert env.episode_length_buf.tolist() == [0, 0]
    assert env.unwrapped.episode_length_buf.tolist() == [0, 0]
    torch.testing.assert_close(command._frontres_pos_correction, torch.zeros(2, 3))
    torch.testing.assert_close(
        command._frontres_quat_correction,
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )
    assert command.perturber.reset_calls == [[0, 1]]
    assert len(command.reference_window_writes) == 1
    assert command.reference_window_writes[0][1].tolist() == [0, 1]
    torch.testing.assert_close(command.reference_window, request.reference_window)
    assert result.diagnostics["reference_window_applied_frac"] == 1.0


if __name__ == "__main__":
    test_live_reset_hook_writes_dynamic_state_through_wrapper()
    print("frontres_segment_live_reset_hook_contract: ok")
