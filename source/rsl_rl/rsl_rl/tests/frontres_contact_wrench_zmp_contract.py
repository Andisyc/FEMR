"""Deterministic S1/S2 contract for FRS-GAIN-v006 contact-wrench ZMP authority."""

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
TRACKING_CFG = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"


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


def test_expected_envelope_and_physical_no_load_is_na() -> None:
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
    missing_margin = owner.expected_support_envelope_margin(
        missing_zmp,
        envelope[:1],
        torch.ones(1, 2),
    )
    assert bool(torch.isnan(missing_margin).all())

    malformed_points = points.clone()
    malformed_points[0, 0, 0, 0] = float("nan")
    malformed_valid = valid.clone()
    malformed_valid[0, 0, 0] = True
    try:
        owner.contact_wrench_zmp_xy(malformed_points, forces, normals, malformed_valid)
    except ValueError as exc:
        assert "must be finite" in str(exc)
    else:
        raise AssertionError("non-finite valid raw contact payload did not fail closed")


def test_formal_owner_isolation() -> None:
    probe = LIVE_PROBE.read_text(encoding="utf-8")
    capture = probe[probe.index("def _capture_physics_frame"):probe.index("def _capture_v015_quality_lateral_lean_frame")]
    assert "_contact_wrench_zmp_pair" in capture
    assert "_frontres_branch_balance_margin" not in capture
    cfg = G1_CFG.read_text(encoding="utf-8")
    scene_cfg = TRACKING_CFG.read_text(encoding="utf-8")
    assert 'frontres_left_foot_contacts = ContactSensorCfg(' in cfg
    assert 'frontres_right_foot_contacts = ContactSensorCfg(' in cfg
    assert 'filter_prim_paths_expr=ground_filter' in cfg
    assert 'ground_filter = ["/World/ground/terrain/mesh"]' in cfg
    assert 'ground_filter = ["/World/ground/terrain"]' not in cfg
    assert 'ground_filter = ["/World/ground/terrain/.*"]' not in cfg
    assert 'prim_path="/World/ground"' in scene_cfg
    assert 'terrain_type="generator"' in scene_cfg
    assert "track_contact_points=" not in cfg
    assert "max_contact_data_count_per_prim=" not in cfg


def test_raw_contact_owner_unpacking() -> None:
    tree = ast.parse(LIVE_PROBE.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"_ensure_frontres_raw_contact_view", "_raw_filtered_contact_rows"}
    ]
    namespace: dict[str, Any] = {"torch": torch, "Any": Any, "re": __import__("re")}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(LIVE_PROBE), "exec"), namespace)
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
        _frontres_raw_contact_capacity=64,
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


def test_legacy_contact_view_capacity_upgrade_and_fail_closed() -> None:
    tree = ast.parse(LIVE_PROBE.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_ensure_frontres_raw_contact_view"
    )
    namespace: dict[str, Any] = {"Any": Any, "re": __import__("re")}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(LIVE_PROBE), "exec"), namespace)

    calls: list[tuple[str, dict[str, Any]]] = []
    upgraded = SimpleNamespace(count=8, filter_count=1)

    def create_view(pattern: str, **kwargs: Any) -> Any:
        calls.append((pattern, kwargs))
        return upgraded

    legacy = SimpleNamespace(
        count=8,
        filter_count=1,
        get_contact_data=lambda dt: (_ for _ in ()).throw(Exception("max_contact_data_count = 0")),
    )
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(
            prim_path="/World/envs/env_.*/Robot/left_ankle_roll_link",
            filter_prim_paths_expr=["/World/ground/terrain/mesh"],
        ),
        body_names=["left_ankle_roll_link"],
        contact_physx_view=legacy,
        _contact_physx_view=legacy,
        _physics_sim_view=SimpleNamespace(create_rigid_contact_view=create_view),
    )
    result = namespace["_ensure_frontres_raw_contact_view"](sensor, num_envs=8)
    assert result is upgraded and sensor._contact_physx_view is upgraded
    assert sensor._frontres_raw_contact_capacity == 128
    assert calls == [
        (
            "/World/envs/env_*/Robot/(left_ankle_roll_link)",
            {
                "filter_patterns": ["/World/ground/terrain/mesh"],
                "max_contact_data_count": 128,
            },
        )
    ]

    bad = SimpleNamespace(count=7, filter_count=1)
    sensor._frontres_raw_contact_capacity = 0
    sensor._physics_sim_view = SimpleNamespace(create_rigid_contact_view=lambda *args, **kwargs: bad)
    try:
        namespace["_ensure_frontres_raw_contact_view"](sensor, num_envs=8)
    except RuntimeError as exc:
        assert "body/env identity" in str(exc)
    else:
        raise AssertionError("raw contact view silently changed role/env identity")


def test_formal_zmp_capture_preserves_first_invalid_error() -> None:
    tree = ast.parse(LIVE_PROBE.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_contact_wrench_zmp_pair"
    )
    namespace: dict[str, Any] = {"Any": Any, "Mapping": __import__("collections.abc").abc.Mapping, "torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(LIVE_PROBE), "exec"), namespace)
    runner = SimpleNamespace(env=SimpleNamespace(scene={}), device=torch.device("cpu"))
    command = SimpleNamespace(
        num_envs=8,
        frontres_local_scenario_k_execution_snapshot=lambda: {
            "expected_support_envelope": torch.zeros(7, 6),
        },
    )
    try:
        namespace["_contact_wrench_zmp_pair"](
            runner,
            command,
            SimpleNamespace(n_train=4, n_candidate=0),
            torch.ones(8, 2),
            torch.ones(4, 2),
            torch.ones(4, 2),
            4,
        )
    except RuntimeError as exc:
        assert "expected_support_envelope [8,6], got (7, 6)" in str(exc)
    else:
        raise AssertionError("formal ZMP capture collapsed a first-invalid shape into None")


def test_eight_role_ground_contact_reaches_finite_zmp() -> None:
    tree = ast.parse(LIVE_PROBE.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"_ensure_frontres_raw_contact_view", "_raw_filtered_contact_rows"}
    ]
    namespace: dict[str, Any] = {"torch": torch, "Any": Any, "re": __import__("re")}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(LIVE_PROBE), "exec"), namespace)

    num_envs = 8
    points_w = torch.zeros(num_envs, 3)
    points_w[:, 0] = torch.arange(num_envs, dtype=torch.float32) * 0.01
    raw = (
        torch.full((num_envs, 1), 100.0),
        points_w,
        torch.tensor([[0.0, 0.0, 1.0]]).repeat(num_envs, 1),
        torch.zeros(num_envs, 1),
        torch.ones(num_envs, 1, dtype=torch.long),
        torch.arange(num_envs, dtype=torch.long).reshape(-1, 1),
    )
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(update_period=0.01),
        _sim_physics_dt=0.005,
        _frontres_raw_contact_capacity=128,
        contact_physx_view=SimpleNamespace(get_contact_data=lambda dt: raw),
    )
    unpacked = namespace["_raw_filtered_contact_rows"](
        sensor,
        num_envs=num_envs,
        device=torch.device("cpu"),
    )
    zmp, valid = _owner().contact_wrench_zmp_xy(*unpacked)
    assert tuple(zmp.shape) == (num_envs, 2)
    assert bool(valid.all()) and bool(torch.isfinite(zmp).all())
    torch.testing.assert_close(zmp, points_w[:, :2])


def test_raw_views_are_installed_before_reset_and_never_lazily_on_read() -> None:
    source = LIVE_PROBE.read_text(encoding="utf-8")
    builder = source[
        source.index("def _build_frontres_v015_local_transaction_request") :
        source.index("def _build_frontres_v015_local_identity_sentinel_request")
    ]
    assert builder.index("_prepare_frontres_raw_contact_views(runner)") < builder.index(
        "prepare_frontres_v015_local_sentinel_batch(runner)"
    )
    assert builder.index("_prepare_frontres_raw_contact_views(runner)") < builder.index(
        "_apply_current_segment_reset(runner"
    )
    raw_reader = source[
        source.index("def _raw_filtered_contact_rows") : source.index("def _contact_wrench_zmp_pair")
    ]
    assert "_ensure_frontres_raw_contact_view(" not in raw_reader
    assert "installed before the scored physics step" in raw_reader

    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"_ensure_frontres_raw_contact_view", "_prepare_frontres_raw_contact_views"}
    ]
    namespace: dict[str, Any] = {"Any": Any, "re": __import__("re")}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(LIVE_PROBE), "exec"), namespace)

    events: list[str] = []

    def sensor(side: str) -> Any:
        legacy = SimpleNamespace(count=8, filter_count=1)

        def create_view(pattern: str, **kwargs: Any) -> Any:
            events.append(f"install-{side}")
            return SimpleNamespace(count=8, filter_count=1)

        return SimpleNamespace(
            cfg=SimpleNamespace(
                prim_path=f"/World/envs/env_.*/Robot/{side}_ankle_roll_link",
                filter_prim_paths_expr=["/World/ground/terrain/mesh"],
            ),
            body_names=[f"{side}_ankle_roll_link"],
            contact_physx_view=legacy,
            _contact_physx_view=legacy,
            _physics_sim_view=SimpleNamespace(create_rigid_contact_view=create_view),
        )

    left, right = sensor("left"), sensor("right")
    runner = SimpleNamespace(
        env=SimpleNamespace(
            num_envs=8,
            scene={"frontres_left_foot_contacts": left, "frontres_right_foot_contacts": right},
        )
    )
    namespace["_prepare_frontres_raw_contact_views"](runner)
    assert events == ["install-left", "install-right"]
    assert left._frontres_raw_contact_capacity == 128
    assert right._frontres_raw_contact_capacity == 128


def test_asymmetric_foot_contact_slots_pad_without_changing_evidence() -> None:
    tree = ast.parse(LIVE_PROBE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_pad_raw_contact_slots"
    )
    namespace: dict[str, Any] = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(LIVE_PROBE), "exec"), namespace)

    def raw(slots: int, x: float) -> tuple[torch.Tensor, ...]:
        points = torch.zeros(8, 1, slots, 3)
        points[:, :, 0, 0] = x
        forces = torch.zeros(8, 1, slots)
        forces[:, :, 0] = 100.0
        normals = torch.zeros_like(points)
        normals[:, :, 0, 2] = 1.0
        valid = forces > 0.0
        return points, forces, normals, valid

    left = namespace["_pad_raw_contact_slots"](raw(10, -0.1), contact_slots=10)
    right = namespace["_pad_raw_contact_slots"](raw(3, 0.1), contact_slots=10)
    for left_value, right_value in zip(left, right, strict=True):
        assert int(left_value.shape[2]) == int(right_value.shape[2]) == 10
    assert right[3][:, :, :3].sum().item() == 8
    assert not bool(right[3][:, :, 3:].any())
    points = torch.cat((left[0], right[0]), dim=1)
    forces = torch.cat((left[1], right[1]), dim=1)
    normals = torch.cat((left[2], right[2]), dim=1)
    valid = torch.cat((left[3], right[3]), dim=1)
    zmp, zmp_valid = _owner().contact_wrench_zmp_xy(points, forces, normals, valid)
    assert bool(zmp_valid.all())
    torch.testing.assert_close(zmp, torch.zeros(8, 2))

    swapped_left = namespace["_pad_raw_contact_slots"](raw(3, 0.1), contact_slots=10)
    swapped_right = namespace["_pad_raw_contact_slots"](raw(10, -0.1), contact_slots=10)
    swapped_zmp, swapped_valid = _owner().contact_wrench_zmp_xy(
        torch.cat((swapped_left[0], swapped_right[0]), dim=1),
        torch.cat((swapped_left[1], swapped_right[1]), dim=1),
        torch.cat((swapped_left[2], swapped_right[2]), dim=1),
        torch.cat((swapped_left[3], swapped_right[3]), dim=1),
    )
    assert torch.equal(swapped_valid, zmp_valid)
    torch.testing.assert_close(swapped_zmp, zmp)


if __name__ == "__main__":
    test_contact_wrench_golden_and_permutation()
    test_expected_envelope_and_physical_no_load_is_na()
    test_formal_owner_isolation()
    test_raw_contact_owner_unpacking()
    test_legacy_contact_view_capacity_upgrade_and_fail_closed()
    test_formal_zmp_capture_preserves_first_invalid_error()
    test_eight_role_ground_contact_reaches_finite_zmp()
    test_raw_views_are_installed_before_reset_and_never_lazily_on_read()
    test_asymmetric_foot_contact_slots_pad_without_changing_evidence()
    print("frontres_contact_wrench_zmp_contract: ok", flush=True)
