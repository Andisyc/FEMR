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
        if attr == "body_pos_w":
            return torch.stack([frame[:, 0], frame[:, 0] + 1.0, frame[:, 0] + 2.0], dim=-1).unsqueeze(1).to(out_device)
        if attr == "body_quat_w":
            quat = torch.zeros(frame.shape[0], 1, 4, dtype=torch.float32)
            quat[..., 0] = 1.0
            quat[..., 1] = frame
            return quat.to(out_device)
        if attr == "joint_vel":
            base = base + 1000.0
        return base.to(out_device)


class MarkerJointPerturber:
    """Expose whether a command path consumes the joint-perturbation owner."""

    def __init__(self, offset: float = 5000.0) -> None:
        self.offset = float(offset)
        self.calls = 0

    def apply_joint_perturbation(self, joint_pos: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return joint_pos + self.offset


class RejectPerturber:
    """Fail if a fixed-tape route attempts a fresh perturbation draw."""

    def __init__(self) -> None:
        self.calls = 0

    def _reject(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("fixed Noisy tape must not resample the perturber")

    apply_perturbations = _reject
    apply_quat_perturbation = _reject
    apply_joint_perturbation = _reject


class DeterministicTapePerturber:
    """Small stateful stand-in for selection-time fixed-tape materialization."""

    instances: list["DeterministicTapePerturber"] = []

    def __init__(self, cfg, num_envs: int, device) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = device
        self.scale = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.family_masks = None
        self.step = 0
        type(self).instances.append(self)

    def set_dr_scale_env(self, scale: torch.Tensor) -> None:
        self.scale = scale.to(self.device, dtype=torch.float32).clone()

    def set_family_env_masks(self, masks) -> None:
        self.family_masks = {key: value.clone() for key, value in masks.items()}

    def apply_perturbations(self, root_pos: torch.Tensor, *_feet: torch.Tensor) -> torch.Tensor:
        self.step += 1
        return root_pos + self.scale[:, None] * torch.tensor([1.0, 2.0, 3.0], device=root_pos.device)

    def apply_quat_perturbation(self, root_quat: torch.Tensor) -> torch.Tensor:
        out = root_quat.clone()
        out[:, 1] += self.scale
        return out

    def apply_joint_perturbation(self, joint_pos: torch.Tensor) -> torch.Tensor:
        return joint_pos + self.scale[:, None]


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
    command.motion_anchor_body_index = 0
    command.left_foot_idx = 0
    command.right_foot_idx = 0
    command._env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(3, 3)))
    command._cached_perturbed_pos = torch.zeros(3, 3)
    command._cached_perturbed_quat = torch.zeros(3, 4)
    command._cached_perturbed_quat[:, 0] = 1.0
    command._dr_supervised_target = torch.zeros(3, 6)
    command.jump_degree = torch.zeros(3)
    command._compute_jump_degree = lambda: None
    command._frontres_pair_train_ids = None
    command._frontres_pair_candidate_ids = None
    command._frontres_pair_base_ids = None
    command._frontres_pair_clean_ids = None
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


def test_current_command_future_path_bypasses_joint_perturbation_owner() -> None:
    commands_module = _load_commands_module()
    command = _fake_command(commands_module.MultiMotionCommand)
    marker = MarkerJointPerturber()
    command.perturber = marker

    command_sequence = command.command.reshape(3, 3, 4)
    calls_after_command = marker.calls
    direct_joint_pos = command.joint_pos

    _probe(
        "command.future_path",
        command_sequence,
        "current command/current-future reference gathered without joint perturbation owner",
    )
    _probe(
        "joint_pos.owner_path",
        direct_joint_pos,
        "current-frame joint reference after explicit joint perturbation owner",
    )
    print(
        "[probe step1A] command_vs_joint_perturbation: "
        f"perturber_calls_after_command={calls_after_command} "
        f"perturber_calls_after_joint_pos={marker.calls} "
        f"marker_offset={marker.offset:.1f}",
        flush=True,
    )

    assert calls_after_command == 0
    assert marker.calls == 1
    assert not torch.allclose(command_sequence[:, 0, :2], direct_joint_pos)


def test_fixed_noisy_tape_controls_command_context_and_cursor() -> None:
    commands_module = _load_commands_module()
    command = _fake_command(commands_module.MultiMotionCommand)
    env_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    dof = 2
    feature_dim = 2 * dof + 7
    tape = torch.zeros(3, 5, feature_dim, dtype=torch.float32)
    for row in range(3):
        for frame in range(5):
            base = 1000.0 * (row + 1) + 100.0 * frame
            tape[row, frame, :dof] = torch.tensor([base, base + 1.0])
            tape[row, frame, dof : 2 * dof] = torch.tensor([base + 10.0, base + 11.0])
            tape[row, frame, 2 * dof : 2 * dof + 3] = torch.tensor([base + 20.0, base + 21.0, base + 22.0])
            tape[row, frame, 2 * dof + 3 :] = torch.tensor([1.0, base + 30.0, base + 31.0, base + 32.0])

    applied = command.set_frontres_fixed_noisy_tape(
        tape,
        tape_lengths=torch.tensor([5, 4, 5], dtype=torch.long),
        scenario_ids=("scenario-a", "scenario-clean", "scenario-b"),
        noisy_segment_hashes=("hash-a", "hash-clean", "hash-b"),
        execution_mask=torch.tensor([True, False, True]),
        env_ids=env_ids,
    )

    command_matrix = command.command.reshape(3, 3, 2 * dof)
    context_before = command.frontres_fixed_noisy_future_context((1, 3))
    cursor_before = command._frontres_fixed_noisy_tape_cursor.clone()
    _probe("fixed_tape.command_before", command_matrix, "execution rows read q/dq from the sealed Noisy tape")
    _probe("fixed_tape.context_before", context_before, "all actor rows read ordered H offsets from the same tape")
    assert applied.tolist() == [True, True, True]
    torch.testing.assert_close(command_matrix[0], tape[0, :3, : 2 * dof])
    torch.testing.assert_close(command_matrix[2], tape[2, :3, : 2 * dof])
    torch.testing.assert_close(context_before[0], torch.cat([tape[0, 1], tape[0, 3]], dim=-1))
    torch.testing.assert_close(context_before[2], torch.cat([tape[2, 1], tape[2, 3]], dim=-1))
    torch.testing.assert_close(cursor_before, torch.zeros_like(cursor_before))

    command._advance_frontres_fixed_noisy_tape()
    command_matrix_after = command.command.reshape(3, 3, 2 * dof)
    context_after = command.frontres_fixed_noisy_future_context((1, 3))
    _probe("fixed_tape.command_after", command_matrix_after, "K execution advances the tape cursor exactly once")
    _probe("fixed_tape.context_after", context_after, "H lookup follows the advanced cursor without advancing it")
    torch.testing.assert_close(command_matrix_after[0], tape[0, 1:4, : 2 * dof])
    torch.testing.assert_close(context_after[0], torch.cat([tape[0, 2], tape[0, 4]], dim=-1))
    assert command._frontres_fixed_noisy_tape_cursor.tolist() == [1, 1, 1]
    assert command.frontres_fixed_noisy_tape_hashes(env_ids) == ("hash-a", "hash-clean", "hash-b")


def test_fixed_noisy_tape_owns_current_anchor_and_joint_reference_without_resampling() -> None:
    commands_module = _load_commands_module()
    command = _fake_command(commands_module.MultiMotionCommand)
    env_ids = torch.arange(3, dtype=torch.long)
    dof = 2
    tape = torch.zeros(3, 4, 2 * dof + 7, dtype=torch.float32)
    for row in range(3):
        base = 100.0 * (row + 1)
        tape[row, :, :dof] = torch.tensor([base, base + 1.0])
        tape[row, :, dof : 2 * dof] = torch.tensor([base + 10.0, base + 11.0])
        tape[row, :, 2 * dof : 2 * dof + 3] = torch.tensor([base + 20.0, base + 21.0, base + 22.0])
        tape[row, :, 2 * dof + 3 :] = torch.tensor([1.0, base + 30.0, base + 31.0, base + 32.0])

    command.set_frontres_fixed_noisy_tape(
        tape,
        tape_lengths=torch.full((3,), 4, dtype=torch.long),
        scenario_ids=("scenario-a", "scenario-clean", "scenario-b"),
        noisy_segment_hashes=("hash-a", "hash-clean", "hash-b"),
        execution_mask=torch.tensor([True, False, True]),
        env_ids=env_ids,
    )
    perturber = RejectPerturber()
    command.perturber = perturber

    command.refresh_frontres_reference_cache_current_frame()
    joint_pos = command.joint_pos
    joint_vel = command.joint_vel
    clean_root_pos = command.motion_dir_loader.gather(
        "body_pos_w", command.env_motion_indices, command.time_steps, command.device
    )[:, 0]

    _probe("fixed_tape.anchor_cache", command._cached_perturbed_pos, "execution anchors read tape while Clean execution remains clean")
    _probe("fixed_tape.joint_pos", joint_pos, "execution q reads tape without a perturber draw")
    _probe("fixed_tape.joint_vel", joint_vel, "execution dq reads tape without a perturber draw")
    assert perturber.calls == 0
    torch.testing.assert_close(command._cached_perturbed_pos[0], tape[0, 0, 2 * dof : 2 * dof + 3])
    torch.testing.assert_close(command._cached_perturbed_pos[2], tape[2, 0, 2 * dof : 2 * dof + 3])
    torch.testing.assert_close(command._cached_perturbed_pos[1], clean_root_pos[1])
    torch.testing.assert_close(joint_pos[0], tape[0, 0, :dof])
    torch.testing.assert_close(joint_pos[2], tape[2, 0, :dof])
    torch.testing.assert_close(joint_vel[0], tape[0, 0, dof : 2 * dof])
    torch.testing.assert_close(joint_vel[2], tape[2, 0, dof : 2 * dof])


def test_command_materializes_complete_fixed_noisy_tape_once() -> None:
    commands_module = _load_commands_module()
    command = _fake_command(commands_module.MultiMotionCommand)
    DeterministicTapePerturber.instances.clear()
    command.perturber = DeterministicTapePerturber(SimpleNamespace(), 3, command.device)

    tape = command.materialize_frontres_fixed_noisy_tape(
        motion_index=0,
        start_frame=2,
        frame_count=5,
        perturbation_family="local_rp",
        perturbation_strength=0.25,
    )

    _probe("fixed_tape.materialized", tape, "selection-time carrier is [L, q+dq+anchor_pos+anchor_quat]")
    assert tuple(tape.shape) == (5, 11)
    torch.testing.assert_close(tape[:, :2], torch.tensor([[20.25, 21.25], [30.25, 31.25], [40.25, 41.25], [50.25, 51.25], [60.25, 61.25]]))
    torch.testing.assert_close(tape[:, 2:4], torch.tensor([[1020.0, 1021.0], [1030.0, 1031.0], [1040.0, 1041.0], [1050.0, 1051.0], [1060.0, 1061.0]]))
    torch.testing.assert_close(tape[:, 4:7], torch.tensor([[2.25, 3.5, 4.75], [3.25, 4.5, 5.75], [4.25, 5.5, 6.75], [5.25, 6.5, 7.75], [6.25, 7.5, 8.75]]))
    assert len(DeterministicTapePerturber.instances) == 2
    materializer = DeterministicTapePerturber.instances[-1]
    assert materializer.step == 5
    assert materializer.family_masks["local_rp"].tolist() == [True]


def test_noisy_variant_adapter_repeats_one_state_not_a_future_trajectory() -> None:
    dataset_module = _load_dataset_module()
    variant = SimpleNamespace(
        segment=SimpleNamespace(segment_id=7, horizon_k=3, fps=50.0),
        descriptor=SimpleNamespace(perturbation_id=11, family="local_rp", strength=0.2, params={}),
        noisy_state=SimpleNamespace(
            root_pos=torch.tensor([[1.0, 2.0, 3.0]]),
            root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_lin_vel=torch.tensor([[4.0, 5.0, 6.0]]),
            root_ang_vel=torch.tensor([[7.0, 8.0, 9.0]]),
            joint_pos=torch.tensor([[10.0, 11.0]]),
            joint_vel=torch.tensor([[110.0, 111.0]]),
            body_pos_w=torch.zeros(1, 1, 3),
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        ),
        noisy_baseline_score=torch.tensor([0.0]),
        noisy_fall=torch.tensor([False]),
    )

    motion = dataset_module._motion_from_noisy_variant(variant, role="train")
    reference = motion["reference"]
    _probe(
        "noisy_variant.reference_adapter",
        reference,
        "legacy cache adapter repeats one noisy state across the requested K horizon",
    )
    print(
        "[probe step1A] noisy_variant_future_semantics: "
        f"frames={reference.shape[0]} "
        f"all_frames_equal={bool(torch.equal(reference, reference[:1].expand_as(reference)))}",
        flush=True,
    )

    assert tuple(reference.shape) == (4, 4)
    assert torch.equal(reference, reference[:1].expand_as(reference))


def main() -> None:
    test_dataset_reference_window_is_joint_command_payload()
    test_multi_motion_command_reference_window_override_lifecycle()
    test_current_command_future_path_bypasses_joint_perturbation_owner()
    test_fixed_noisy_tape_controls_command_context_and_cursor()
    test_fixed_noisy_tape_owns_current_anchor_and_joint_reference_without_resampling()
    test_command_materializes_complete_fixed_noisy_tape_once()
    test_noisy_variant_adapter_repeats_one_state_not_a_future_trajectory()
    print("frontres_segment_motion_command_reference_contract: ok")


if __name__ == "__main__":
    main()
