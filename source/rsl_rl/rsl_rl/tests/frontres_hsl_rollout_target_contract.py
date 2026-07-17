#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "runners" / "frontres_hsl_rollout_target.py"

math_stub = types.ModuleType("isaaclab.utils.math")
math_stub.quat_inv = lambda quat: quat
math_stub.quat_mul = lambda lhs, _rhs: lhs
sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
sys.modules.setdefault("isaaclab.utils", types.ModuleType("isaaclab.utils"))
sys.modules["isaaclab.utils.math"] = math_stub

spec = importlib.util.spec_from_file_location("frontres_hsl_rollout_target_contract_owner", MODULE_PATH)
owner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = owner
spec.loader.exec_module(owner)


class _Cone:
    def project_task_target(self, _command, target: torch.Tensor) -> torch.Tensor:
        return target


def _runner() -> SimpleNamespace:
    root_pos = torch.tensor(
        [
            [0.2, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1)
    data = SimpleNamespace(root_pos_w=root_pos, root_quat_w=root_quat)
    command = SimpleNamespace(robot=SimpleNamespace(data=data))
    env = SimpleNamespace(
        num_envs=4,
        scene=SimpleNamespace(env_origins=torch.zeros(4, 3)),
    )
    runner = SimpleNamespace(
        cfg={"frontres_hsl_rollout_label_enabled": True},
        device=torch.device("cpu"),
        env=env,
        alg=SimpleNamespace(transition=SimpleNamespace(sentinel="unchanged")),
        _frontres_action_cone=_Cone(),
    )
    runner.command = command
    return runner


def test_target_math_and_non_mutating_audit_mode() -> None:
    runner = _runner()
    actions = torch.zeros(4, 6)
    current_pos = torch.tensor([[0.1, 0.0, 0.0]])
    current_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    result = owner.build_frontres_hsl_rollout_target(
        runner,
        command=runner.command,
        actions=actions,
        dones=torch.zeros(4, dtype=torch.bool),
        current_pos_correction=current_pos,
        current_quat_correction=current_quat,
        n_train=1,
        n_candidate=1,
        n_base=1,
        n_clean=1,
        quat_to_rotvec_wxyz=owner.quat_to_rotvec_wxyz,
        write_transition=False,
    )
    assert result is not None
    noisy_error = torch.tensor(0.3)
    front_error = torch.tensor(0.2)
    safe_weight = torch.sigmoid((torch.tensor(0.03) - noisy_error) / 0.01)
    broken_weight = torch.sigmoid((noisy_error - 0.35) / 0.05)
    repair_weight = torch.sigmoid((noisy_error - 0.03) / 0.01) * torch.sigmoid((0.35 - noisy_error) / 0.05)
    harm_weight = torch.sigmoid((front_error - noisy_error) / 0.02)
    noop_weight = safe_weight + broken_weight + 2.0 * harm_weight
    expected_target_x = -0.1 * repair_weight / (repair_weight + noop_weight)
    torch.testing.assert_close(result.target[0, 0], expected_target_x)
    torch.testing.assert_close(result.weight[0, 0], repair_weight + noop_weight)
    torch.testing.assert_close(result.harm_weight[0, 0], harm_weight)
    torch.testing.assert_close(result.target[1:], torch.zeros(3, 6))
    assert vars(runner.alg.transition) == {"sentinel": "unchanged"}

    written = owner.build_frontres_hsl_rollout_target(
        runner,
        command=runner.command,
        actions=actions,
        dones=torch.zeros(4, dtype=torch.bool),
        current_pos_correction=current_pos,
        current_quat_correction=current_quat,
        n_train=1,
        n_candidate=1,
        n_base=1,
        n_clean=1,
        quat_to_rotvec_wxyz=owner.quat_to_rotvec_wxyz,
    )
    assert written is not None
    torch.testing.assert_close(runner.alg.transition.supervised_target, written.target)
    torch.testing.assert_close(runner.alg.transition.supervised_weight, written.weight)
    torch.testing.assert_close(runner.alg.transition.supervised_harm_weight, written.harm_weight)


if __name__ == "__main__":
    test_target_math_and_non_mutating_audit_mode()
    print("PASS: canonical HSL rollout target supports non-mutating quality audit capture.")
