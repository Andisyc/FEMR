#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


REPO = Path(__file__).resolve().parents[4]
RSL_ROOT = REPO / "source" / "rsl_rl"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _install_isaac_stubs() -> None:
    isaaclab = _package("isaaclab")
    assets = _package("isaaclab.assets")
    managers = _package("isaaclab.managers")
    markers = _package("isaaclab.markers")
    markers_config = _package("isaaclab.markers.config")
    utils = _package("isaaclab.utils")
    math_mod = _package("isaaclab.utils.math")

    class _Dummy:
        def __init__(self, *args, **kwargs) -> None:
            self.markers = {"frame": SimpleNamespace(scale=None)}

        def replace(self, **kwargs):
            return self

        def copy(self):
            return self

    assets.Articulation = _Dummy
    managers.CommandTerm = _Dummy
    managers.CommandTermCfg = _Dummy
    markers.VisualizationMarkers = _Dummy
    markers.VisualizationMarkersCfg = _Dummy
    markers_config.FRAME_MARKER_CFG = _Dummy()
    utils.configclass = lambda cls: cls

    def _identity_first(x, *args, **kwargs):
        return x

    math_mod.euler_xyz_from_quat = lambda q: (q[..., 0], q[..., 0], q[..., 0])
    math_mod.quat_apply = lambda q, v: v
    math_mod.quat_error_magnitude = lambda a, b: torch.zeros(a.shape[0], device=a.device)
    math_mod.quat_from_euler_xyz = lambda x, y, z: torch.stack(
        [torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)], dim=-1
    )
    math_mod.quat_inv = _identity_first
    math_mod.quat_mul = lambda a, b: b
    math_mod.sample_uniform = lambda low, high, shape, device=None: torch.zeros(shape, device=device)
    math_mod.yaw_quat = _identity_first

    isaaclab.assets = assets
    isaaclab.managers = managers
    isaaclab.markers = markers
    isaaclab.utils = utils

    _package("whole_body_tracking")
    _package("whole_body_tracking.whole_body_tracking")
    _package("whole_body_tracking.whole_body_tracking.tasks")
    _package("whole_body_tracking.whole_body_tracking.tasks.tracking")
    mdp_pkg = _package("whole_body_tracking.whole_body_tracking.tasks.tracking.mdp")
    perturbations = types.ModuleType("whole_body_tracking.whole_body_tracking.tasks.tracking.mdp.motion_perturbations")
    perturbations.MotionPerturber = _Dummy
    sys.modules[perturbations.__name__] = perturbations
    mdp_pkg.motion_perturbations = perturbations


def _load_commands_module():
    _install_isaac_stubs()
    return _load(
        "whole_body_tracking.whole_body_tracking.tasks.tracking.mdp.commands",
        REPO / "source" / "whole_body_tracking" / "whole_body_tracking" / "tasks" / "tracking" / "mdp" / "commands.py",
    )


def _load_dataset_module():
    return _load(
        "frontres_segment_dataset_for_motion_command_reference_contract",
        RSL_ROOT / "rsl_rl" / "frontres" / "frontres_segment_dataset.py",
    )


class FakeMotionDirLoader:
    def __init__(self, dof: int = 2) -> None:
        self.joint_pos = torch.zeros(12, dof)
        self.joint_vel = torch.zeros(12, dof)

    def gather(self, attr: str, motion_indices: torch.Tensor, frame_indices: torch.Tensor, out_device):
        frame = frame_indices.to(torch.float32).unsqueeze(-1)
        base = 10.0 * frame + torch.arange(self.joint_pos.shape[-1], dtype=torch.float32).view(1, -1)
        if attr == "joint_vel":
            base = base + 1000.0
        return base.to(out_device)


def _fake_command(command_cls):
    command = object.__new__(command_cls)
    command.num_envs = 3
    command.device = torch.device("cpu")
    command.cfg = SimpleNamespace(motion_horizon=3, command_velocity=True)
    command.motion_dir_loader = FakeMotionDirLoader(dof=2)
    command.env_motion_indices = torch.zeros(3, dtype=torch.long)
    command.time_steps = torch.zeros(3, dtype=torch.long)
    command.motion_lengths_minus_one = torch.tensor([11], dtype=torch.long)
    command._init_frontres_reference_window_buffers()
    return command


def _probe(name: str, tensor: torch.Tensor, semantic: str) -> None:
    data = tensor.detach().cpu()
    numeric = data.float()
    print(
        f"[probe step16] {name}: shape={tuple(data.shape)} dtype={data.dtype} "
        f"min={numeric.min().item():.6f} max={numeric.max().item():.6f} "
        f"mean={numeric.mean().item():.6f} semantic={semantic}",
        flush=True,
    )


def test_dataset_reference_window_is_joint_command_payload() -> None:
    dataset_module = _load_dataset_module()
    FrontRESSegmentDataset = dataset_module.FrontRESSegmentDataset
    frames = 5
    dof = 2
    motion = {
        "motion_id": "toy",
        "root_pos": torch.zeros(frames, 3),
        "root_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(frames, 1),
        "root_lin_vel": torch.zeros(frames, 3),
        "root_ang_vel": torch.zeros(frames, 3),
        "dof_pos": torch.arange(frames * dof, dtype=torch.float32).reshape(frames, dof),
        "dof_vel": torch.arange(frames * dof, dtype=torch.float32).reshape(frames, dof) + 100.0,
    }
    dataset = FrontRESSegmentDataset([motion], dt=1.0 / 30.0, default_horizon_k=3, device="cpu")
    batch = dataset.get_segments(torch.tensor([0]))
    expected = torch.cat([motion["dof_pos"][:4], motion["dof_vel"][:4]], dim=-1).unsqueeze(0)
    _probe("dataset.reference_window", batch.reference_window, "joint_pos and joint_vel command payload from segment dataset")
    torch.testing.assert_close(batch.reference_window, expected)
    assert tuple(batch.reference_window.shape) == (1, 4, 2 * dof)


def test_multi_motion_command_reference_window_override_lifecycle() -> None:
    commands_module = _load_commands_module()
    command = _fake_command(commands_module.MultiMotionCommand)
    env_ids = torch.tensor([0, 2], dtype=torch.long)
    reference_window = torch.tensor(
        [
            [[10.0, 11.0, 110.0, 111.0], [20.0, 21.0, 120.0, 121.0], [30.0, 31.0, 130.0, 131.0], [40.0, 41.0, 140.0, 141.0]],
            [[50.0, 51.0, 150.0, 151.0], [60.0, 61.0, 160.0, 161.0], [70.0, 71.0, 170.0, 171.0], [80.0, 81.0, 180.0, 181.0]],
        ],
        dtype=torch.float32,
    )
    applied = command.set_frontres_reference_window(reference_window, env_ids=env_ids)
    command_matrix = command.command.reshape(3, 3, 4)
    _probe("request.reference_window", reference_window, "batched segment joint reference window")
    _probe("command.first_read", command_matrix, "GMT command after reference-window override at cursor 0")
    assert applied.tolist() == [True, True]
    torch.testing.assert_close(command_matrix[0], reference_window[0, :3])
    torch.testing.assert_close(command_matrix[2], reference_window[1, :3])
    assert command._frontres_reference_window_active.tolist() == [True, False, True]

    command._advance_frontres_reference_window()
    shifted = command.command.reshape(3, 3, 4)
    _probe("command.after_advance", shifted, "GMT command after reference-window cursor advances by one")
    torch.testing.assert_close(shifted[0], reference_window[0, 1:4])
    torch.testing.assert_close(shifted[2], reference_window[1, 1:4])

    command.clear_frontres_reference_window(torch.tensor([0]))
    partially_cleared = command.command.reshape(3, 3, 4)
    _probe("command.after_partial_clear", partially_cleared, "env 0 returns to motion loader while env 2 remains overridden")
    assert command._frontres_reference_window_active.tolist() == [False, False, True]
    torch.testing.assert_close(partially_cleared[2], reference_window[1, 1:4])
    assert not torch.allclose(partially_cleared[0], reference_window[0, 1:4])

    for _ in range(4):
        command._advance_frontres_reference_window()
    assert command._frontres_reference_window_active.tolist() == [False, False, False]
    print(
        "[probe step16] reference_window_lifecycle: "
        f"applied={applied.tolist()} "
        f"active_after_expire={command._frontres_reference_window_active.tolist()} "
        f"cursor={command._frontres_reference_window_cursor.tolist()}",
        flush=True,
    )


def main() -> None:
    test_dataset_reference_window_is_joint_command_payload()
    test_multi_motion_command_reference_window_override_lifecycle()
    print("frontres_segment_motion_command_reference_contract: ok")


if __name__ == "__main__":
    main()
