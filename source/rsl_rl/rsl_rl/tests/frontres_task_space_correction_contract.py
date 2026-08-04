#!/usr/bin/env python3
"""TEST-11 golden contract for world-frame full-6D task correction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import torch

ROOT = Path(__file__).resolve().parents[4]
executability_stub = ModuleType("rsl_rl.frontres.frontres_executability")
def _rotvec_to_quat(value: torch.Tensor) -> torch.Tensor:
    angle = value.norm(dim=-1, keepdim=True)
    scale = torch.where(angle > 1e-8, torch.sin(angle / 2.0) / angle, torch.full_like(angle, 0.5))
    return torch.cat((torch.cos(angle / 2.0), value * scale), dim=-1)


executability_stub.rotvec_to_quat_wxyz = _rotvec_to_quat
probe_stub = ModuleType("rsl_rl.frontres.frontres_formal_runtime_probe")
probe_stub.emit_formal_runtime_probe = lambda *_args, **_kwargs: None
modules_stub = ModuleType("rsl_rl.modules")
modules_stub.FrontRESActorCritic = type("FrontRESActorCritic", (), {})
sys.modules["rsl_rl.frontres.frontres_executability"] = executability_stub
sys.modules["rsl_rl.frontres.frontres_formal_runtime_probe"] = probe_stub
sys.modules["rsl_rl.modules"] = modules_stub

MODULE_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "frontres" / "task_space_correction.py"
spec = importlib.util.spec_from_file_location("frontres_task_space_correction_contract_module", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
apply_frontres_task_corrections = module.apply_frontres_task_corrections
_quat_mul_wxyz = module._quat_mul_wxyz


class _Policy(modules_stub.FrontRESActorCritic):
    num_task_corrections = 6


class _Command:
    def __init__(self, raw_quat: torch.Tensor):
        self.anchor_quat_w_raw = raw_quat.clone()
        self._frontres_pos_correction = torch.full((raw_quat.shape[0], 3), 77.0)
        self._frontres_quat_correction = torch.full((raw_quat.shape[0], 4), 88.0)


def _runner(command: _Command) -> SimpleNamespace:
    env = SimpleNamespace(num_envs=int(command.anchor_quat_w_raw.shape[0]))
    env.command_manager = SimpleNamespace(_terms={"motion": command})
    env.unwrapped = env
    return SimpleNamespace(env=env, alg=SimpleNamespace(policy=_Policy()))


def test_world_translation_is_written_without_scale_or_clamp() -> None:
    command = _Command(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    correction = torch.tensor([[0.10, -0.20, 0.50, 0.0, 0.0, 0.0]])
    before = correction.clone()

    apply_frontres_task_corrections(_runner(command), correction)

    torch.testing.assert_close(command._frontres_pos_correction, correction[:, :3])
    torch.testing.assert_close(correction, before)


def test_world_rotation_is_applied_on_the_left_by_right_multiply_host() -> None:
    half = torch.tensor(torch.pi / 4.0)
    raw = torch.tensor([[torch.cos(half), 0.0, 0.0, torch.sin(half)]])
    command = _Command(raw)
    correction = torch.tensor([[0.0, 0.0, 0.0, torch.pi / 2.0, 0.0, 0.0]])
    world_delta = _rotvec_to_quat(correction[:, 3:])

    apply_frontres_task_corrections(_runner(command), correction)

    host_result = _quat_mul_wxyz(raw, command._frontres_quat_correction)
    expected = _quat_mul_wxyz(world_delta, raw)
    torch.testing.assert_close(host_result, expected, atol=1e-6, rtol=1e-6)


def test_zero_and_row_permutation_are_exact() -> None:
    raw = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.9238795, 0.0, 0.3826834, 0.0]])
    actions = torch.tensor([[0.0] * 6, [0.2, 0.1, -0.3, 0.0, 0.2, 0.0]])
    first = _Command(raw)
    second = _Command(raw.flip(0))
    apply_frontres_task_corrections(_runner(first), actions)
    apply_frontres_task_corrections(_runner(second), actions.flip(0))
    torch.testing.assert_close(first._frontres_pos_correction.flip(0), second._frontres_pos_correction)
    torch.testing.assert_close(first._frontres_quat_correction.flip(0), second._frontres_quat_correction)
    torch.testing.assert_close(first._frontres_pos_correction[0], torch.zeros(3))
    torch.testing.assert_close(first._frontres_quat_correction[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_nonfinite_and_bad_shape_reject_before_buffer_mutation() -> None:
    command = _Command(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    runner = _runner(command)
    original_pos = command._frontres_pos_correction.clone()
    original_quat = command._frontres_quat_correction.clone()
    for invalid in (torch.zeros(1, 5), torch.tensor([[0.0, 0.0, float("nan"), 0.0, 0.0, 0.0]])):
        try:
            apply_frontres_task_corrections(runner, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid full-6D correction must fail closed")
        torch.testing.assert_close(command._frontres_pos_correction, original_pos)
        torch.testing.assert_close(command._frontres_quat_correction, original_quat)


if __name__ == "__main__":
    test_world_translation_is_written_without_scale_or_clamp()
    test_world_rotation_is_applied_on_the_left_by_right_multiply_host()
    test_zero_and_row_permutation_are_exact()
    test_nonfinite_and_bad_shape_reject_before_buffer_mutation()
    print("frontres_task_space_correction_contract: ok")
