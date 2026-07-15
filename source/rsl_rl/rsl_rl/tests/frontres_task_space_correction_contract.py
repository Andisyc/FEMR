#!/usr/bin/env python3
"""Semantic contract for contact-consistent full-6D task correction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import torch

ROOT = Path(__file__).resolve().parents[4]
executability_stub = ModuleType("rsl_rl.frontres.frontres_executability")
executability_stub.rotvec_to_quat_wxyz = lambda value: value
modules_stub = ModuleType("rsl_rl.modules")
modules_stub.FrontRESActorCritic = type("FrontRESActorCritic", (), {})
sys.modules["rsl_rl.frontres.frontres_executability"] = executability_stub
sys.modules["rsl_rl.modules"] = modules_stub

MODULE_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "frontres" / "task_space_correction.py"
spec = importlib.util.spec_from_file_location("frontres_task_space_correction_contract_module", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
_frontres_contact_consistent_position_correction = module._frontres_contact_consistent_position_correction


def test_per_row_contact_bounds_preserve_full6_application_semantics() -> None:
    policy = SimpleNamespace(max_delta_pos=0.3)
    runner = SimpleNamespace(alg=SimpleNamespace(policy=policy))
    command = SimpleNamespace(
        jump_degree=torch.tensor([0.0, 0.5, 1.0, 0.2], dtype=torch.float32),
        anchor_penetration_depth=torch.tensor([0.2, 0.2, 0.1, 0.2], dtype=torch.float32),
    )
    correction = torch.tensor(
        [
            [0.10, -0.20, -0.50],
            [0.10, -0.20, -0.10],
            [0.10, -0.20, 0.20],
            [0.10, -0.20, 0.50],
        ],
        dtype=torch.float64,
    )

    actual = _frontres_contact_consistent_position_correction(
        runner,
        command,
        correction.clone(),
        n_rows=4,
    )
    expected = torch.tensor(
        [
            [0.10, -0.20, -0.30],
            [0.05, -0.10, -0.10],
            [0.00, 0.00, 0.10],
            [0.08, -0.16, 0.04],
        ],
        dtype=torch.float64,
    )

    assert actual.dtype == correction.dtype
    torch.testing.assert_close(actual, expected)


def test_missing_contact_state_blocks_upward_dz_without_narrowing_xy() -> None:
    runner = SimpleNamespace(alg=SimpleNamespace(policy=SimpleNamespace(max_delta_pos=0.3)))
    command = SimpleNamespace()
    correction = torch.tensor([[0.12, -0.08, 0.20], [-0.05, 0.03, -0.40]])

    actual = _frontres_contact_consistent_position_correction(
        runner,
        command,
        correction.clone(),
        n_rows=2,
    )

    expected = torch.tensor([[0.12, -0.08, 0.00], [-0.05, 0.03, -0.30]])
    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    test_per_row_contact_bounds_preserve_full6_application_semantics()
    test_missing_contact_state_blocks_upward_dz_without_narrowing_xy()
    print("frontres_task_space_correction_contract: ok")
