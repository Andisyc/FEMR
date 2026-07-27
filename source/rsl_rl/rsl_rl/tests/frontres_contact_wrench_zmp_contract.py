"""Deterministic S1/S2 contract for FRS-GAIN-v005 contact-wrench ZMP authority."""

from __future__ import annotations

import importlib.util
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[4]
BALANCE = ROOT / "source/rsl_rl/rsl_rl/frontres/frontres_balance.py"
LIVE_PROBE = ROOT / "source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py"
G1_CFG = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py"


def _owner():
    spec = importlib.util.spec_from_file_location("frontres_contact_wrench_zmp_under_test", BALANCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_contact_wrench_golden_and_permutation() -> None:
    owner = _owner()
    points = torch.zeros(2, 2, 2, 3)
    points[0, 0, 0, :2] = torch.tensor([0.0, 0.0])
    points[0, 0, 1, :2] = torch.tensor([1.0, 0.0])
    points[1] = points[0] + torch.tensor([2.0, -1.0, 0.0])
    forces = torch.tensor([[[1.0, 3.0], [0.0, 0.0]], [[1.0, 3.0], [0.0, 0.0]]])
    normals = torch.zeros_like(points)
    normals[..., 2] = 1.0
    valid = forces > 0.0
    zmp, zmp_valid = owner.contact_wrench_zmp_xy(points, forces, normals, valid)
    torch.testing.assert_close(zmp, torch.tensor([[0.75, 0.0], [2.75, -1.0]]))
    assert bool(zmp_valid.all())

    permutation = torch.tensor([1, 0])
    permuted, permuted_valid = owner.contact_wrench_zmp_xy(
        points.index_select(0, permutation),
        forces.index_select(0, permutation),
        normals.index_select(0, permutation),
        valid.index_select(0, permutation),
    )
    torch.testing.assert_close(permuted, zmp.index_select(0, permutation))
    assert torch.equal(permuted_valid, zmp_valid.index_select(0, permutation))

    moved = points.clone()
    moved[0, 0, 1, 0] = 2.0
    moved_zmp, _ = owner.contact_wrench_zmp_xy(moved, forces, normals, valid)
    assert float(moved_zmp[0, 0]) == 1.5
    # Same resultant force, different application point must not collapse.
    assert not torch.equal(moved_zmp[0], zmp[0])


def test_expected_envelope_and_missing_fail_closed() -> None:
    owner = _owner()
    zmp = torch.tensor([[0.05, 0.01], [0.30, 0.0], [float("nan"), float("nan")]])
    envelope = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 0.10, 0.05],
            [0.0, 0.0, 1.0, 0.0, 0.10, 0.05],
            [0.0, 0.0, 1.0, 0.0, 0.10, 0.05],
        ]
    )
    support = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    margin = owner.expected_support_envelope_margin(zmp, envelope, support)
    torch.testing.assert_close(margin[:2], torch.tensor([0.04, -0.20]))
    assert bool(torch.isnan(margin[2]))

    points = torch.zeros(1, 2, 1, 3)
    forces = torch.zeros(1, 2, 1)
    normals = torch.zeros_like(points)
    valid = torch.zeros_like(forces, dtype=torch.bool)
    missing_zmp, missing = owner.contact_wrench_zmp_xy(points, forces, normals, valid)
    assert not bool(missing.item()) and bool(torch.isnan(missing_zmp).all())
    try:
        owner.expected_support_envelope_margin(
            missing_zmp,
            envelope[:1],
            torch.ones(1, 2),
        )
    except ValueError as exc:
        assert "finite contact-wrench ZMP" in str(exc)
    else:
        raise AssertionError("supported phase silently accepted missing ZMP")


def test_formal_owner_isolation() -> None:
    probe = LIVE_PROBE.read_text(encoding="utf-8")
    capture = probe[probe.index("def _capture_physics_frame"):probe.index("def _capture_v015_quality_lateral_lean_frame")]
    assert "_contact_wrench_zmp_pair" in capture
    assert "_frontres_branch_balance_margin" not in capture
    cfg = G1_CFG.read_text(encoding="utf-8")
    assert 'frontres_left_foot_contacts = ContactSensorCfg(' in cfg
    assert 'frontres_right_foot_contacts = ContactSensorCfg(' in cfg
    assert "track_contact_points=True" in cfg
    assert 'filter_prim_paths_expr=ground_filter' in cfg


def test_raw_contact_owner_unpacking() -> None:
    tree = ast.parse(LIVE_PROBE.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_raw_filtered_contact_rows"
    )
    namespace: dict[str, Any] = {"torch": torch, "Any": Any}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(LIVE_PROBE), "exec"), namespace)
    raw = (
        torch.tensor([[10.0], [30.0], [20.0]]),
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]] * 3),
        torch.zeros(3, 1),
        torch.tensor([[2], [1]], dtype=torch.long),
        torch.tensor([[0], [2]], dtype=torch.long),
    )
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(update_period=0.01),
        _sim_physics_dt=0.005,
        contact_physx_view=SimpleNamespace(get_contact_data=lambda dt: raw),
    )
    points, forces, normals, valid = namespace["_raw_filtered_contact_rows"](
        sensor,
        num_envs=2,
        device=torch.device("cpu"),
    )
    assert tuple(points.shape) == (2, 1, 2, 3)
    torch.testing.assert_close(forces[:, 0], torch.tensor([[10.0, 30.0], [20.0, 0.0]]))
    torch.testing.assert_close(points[1, 0, 0], torch.tensor([2.0, 0.0, 0.0]))
    assert valid[:, 0].tolist() == [[True, True], [True, False]]


if __name__ == "__main__":
    test_contact_wrench_golden_and_permutation()
    test_expected_envelope_and_missing_fail_closed()
    test_formal_owner_isolation()
    test_raw_contact_owner_unpacking()
    print("frontres_contact_wrench_zmp_contract: ok", flush=True)
