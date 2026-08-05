"""Deterministic S1/S2 contract for FRS-GAIN-v006 contact-wrench ZMP authority."""

from __future__ import annotations

import importlib.util
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from frontres_contract_imports import install_frontres_contract_packages


install_frontres_contract_packages()

from rsl_rl.runners import frontres_segment_physics as execution_owner


ROOT = Path(__file__).resolve().parents[4]
BALANCE = ROOT / "source/rsl_rl/rsl_rl/frontres/frontres_balance.py"
PHYSICS_OWNER = ROOT / "source/rsl_rl/rsl_rl/runners/frontres_segment_physics.py"
FORMAL_TRANSACTION = ROOT / "source/rsl_rl/rsl_rl/runners/frontres_segment_formal_transaction.py"
G1_CFG = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py"
TRACKING_CFG = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
_USE_PHYSICAL_ORIGINS = object()
_MISSING_ORIGINS = object()


def _owner():
    spec = importlib.util.spec_from_file_location("frontres_contact_wrench_zmp_under_test", BALANCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _capture_execution_fixture(
    *,
    physical_origins: torch.Tensor,
    local_foot_pos: torch.Tensor,
    selected_rows: torch.Tensor,
    scene_origins: object = _USE_PHYSICAL_ORIGINS,
):
    num_envs = int(physical_origins.shape[0])
    world_foot_pos = local_foot_pos + physical_origins[:, None, :]
    body_pos_w = torch.zeros(num_envs, 3, 3)
    body_pos_w[:, 0] = physical_origins + torch.tensor([0.0, 0.0, 1.0])
    body_pos_w[:, 1:] = world_foot_pos

    class _Scene(dict):
        pass

    def _sensor(side: int) -> Any:
        force_matrix = torch.zeros(num_envs, 1, 1, 3)
        force_matrix[..., 2] = 100.0
        return SimpleNamespace(
            side=side,
            data=SimpleNamespace(force_matrix_w=force_matrix),
            cfg=SimpleNamespace(force_threshold=10.0),
        )

    scene = _Scene(
        frontres_left_foot_contacts=_sensor(0),
        frontres_right_foot_contacts=_sensor(1),
    )
    if scene_origins is _USE_PHYSICAL_ORIGINS:
        scene.env_origins = physical_origins.clone()
    elif scene_origins is not _MISSING_ORIGINS:
        scene.env_origins = scene_origins
    command = SimpleNamespace(
        num_envs=num_envs,
        left_foot_idx=1,
        right_foot_idx=2,
        robot_joint_pos=torch.zeros(num_envs, 29),
        robot_anchor_pos_w=body_pos_w[:, 0].clone(),
        robot_anchor_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(num_envs, 1),
        robot_body_pos_w=body_pos_w,
        robot_anchor_lin_vel_w=torch.zeros(num_envs, 3),
        robot_anchor_ang_vel_w=torch.zeros(num_envs, 3),
        frontres_local_scenario_k_execution_snapshot=lambda: {
            "expected_support": torch.ones(num_envs, 2),
            "expected_support_envelope": torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]).repeat(
                num_envs, 1
            ),
        },
    )
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        env=SimpleNamespace(scene=scene),
    )

    def _raw(sensor: Any, *, num_envs: int, device: torch.device):
        points = world_foot_pos[:, sensor.side : sensor.side + 1, None, :].to(device=device)
        forces = torch.full((num_envs, 1, 1), 100.0, device=device)
        normals = torch.zeros_like(points)
        normals[..., 2] = 1.0
        return points, forces, normals, torch.ones_like(forces, dtype=torch.bool)

    original_command = execution_owner._motion_command_for_runner
    original_raw = execution_owner.read_frontres_raw_filtered_contact_rows
    try:
        execution_owner._motion_command_for_runner = lambda _runner: command
        execution_owner.read_frontres_raw_filtered_contact_rows = _raw
        return execution_owner.capture_frontres_v017_execution_frame(
            runner,
            selected_rows=selected_rows,
        )
    finally:
        execution_owner._motion_command_for_runner = original_command
        execution_owner.read_frontres_raw_filtered_contact_rows = original_raw


def test_execution_foot_pos_is_environment_local_and_row_aligned() -> None:
    origins = torch.tensor([[0.0, 0.0, 0.0], [48.0, -32.0, 0.0], [-16.0, 64.0, 0.0]])
    shared = torch.tensor([[-0.125, 0.0625, 0.03125], [0.125, -0.0625, 0.03125]])
    local = shared.repeat(3, 1, 1)
    local[1, :, 0] += 0.03125
    frame = _capture_execution_fixture(
        physical_origins=origins,
        local_foot_pos=local,
        selected_rows=torch.tensor([0, 1, 2]),
    )
    torch.testing.assert_close(frame.foot_pos, local)
    assert float(torch.linalg.vector_norm(frame.foot_pos[0] - frame.foot_pos[2])) == 0.0
    torch.testing.assert_close(
        torch.linalg.vector_norm(frame.foot_pos[1] - frame.foot_pos[0], dim=-1),
        torch.full((2,), 0.03125),
    )

    permutation = torch.tensor([2, 0, 1])
    permuted = _capture_execution_fixture(
        physical_origins=origins,
        local_foot_pos=local,
        selected_rows=permutation,
    )
    torch.testing.assert_close(permuted.foot_pos, local.index_select(0, permutation))


def test_execution_foot_pos_origin_identity_fails_closed() -> None:
    origins = torch.tensor([[0.0, 0.0, 0.0], [8.0, -4.0, 0.0]])
    local = torch.tensor([[[-0.1, 0.1, 0.02], [0.1, -0.1, 0.02]]]).repeat(2, 1, 1)
    for malformed in (
        _MISSING_ORIGINS,
        torch.zeros(2, 2),
        torch.tensor([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]),
    ):
        try:
            _capture_execution_fixture(
                physical_origins=origins,
                local_foot_pos=local,
                selected_rows=torch.tensor([0, 1]),
                scene_origins=malformed,
            )
        except (RuntimeError, ValueError) as exc:
            assert "env_origins" in str(exc) or "finite" in str(exc) or "ZMP" in str(exc)
        else:
            raise AssertionError("missing or malformed scene.env_origins did not fail closed")


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


def test_expected_support_envelope_is_derived_from_clean_foot_pose() -> None:
    owner = _owner()
    foot_pos = torch.tensor(
        [
            [[-0.10, 0.10, 0.02], [0.10, -0.10, 0.03]],
            [[-0.10, 0.10, 0.20], [0.20, -0.10, 0.02]],
            [[-0.10, 0.10, 0.20], [0.20, -0.10, 0.22]],
        ]
    )
    foot_quat = torch.zeros(3, 2, 4)
    foot_quat[..., 0] = 1.0
    support, envelope = owner.expected_support_and_envelope_from_foot_pose(
        foot_pos,
        foot_quat,
        contact_height=0.08,
        foot_half_length=0.10,
        foot_half_width=0.05,
    )
    assert support.tolist() == [[True, True], [False, True], [False, False]]
    assert tuple(envelope.shape) == (3, 6)
    torch.testing.assert_close(envelope[0, :4], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(envelope[1, :4], torch.tensor([0.20, -0.10, 1.0, 0.0]))
    assert bool(torch.isfinite(envelope).all())
    assert bool((envelope[:, 4:] > 0.0).all())
    try:
        owner.expected_support_and_envelope_from_foot_pose(
            foot_pos,
            foot_quat,
            contact_height=float("inf"),
            foot_half_length=0.10,
            foot_half_width=0.05,
        )
    except ValueError as exc:
        assert "thresholds/extents" in str(exc)
    else:
        raise AssertionError("non-finite expected-support threshold did not fail closed")


def test_formal_owner_isolation() -> None:
    physics = PHYSICS_OWNER.read_text(encoding="utf-8")
    balance = BALANCE.read_text(encoding="utf-8")
    capture = physics[
        physics.index("def _capture_physics_frame"):
        physics.index("def _capture_v015_quality_lateral_lean_frame")
    ]
    contact_owner = physics[
        physics.index("def _contact_sensor_pair"):
        physics.index("def _root_relative_body_pos")
    ]
    assert "_contact_wrench_zmp_pair" in capture
    assert "_frontres_branch_balance_margin" not in capture
    assert "force_matrix_w" in contact_owner
    assert '"net_forces_w"' not in contact_owner
    assert "def ensure_frontres_raw_contact_view" in balance
    assert "def read_frontres_raw_filtered_contact_rows" in balance
    assert "from rsl_rl.runners.frontres_segment_live_probe import _" not in (
        ROOT / "source/rsl_rl/rsl_rl/runners/frontres_segment_sequence_eval.py"
    ).read_text(encoding="utf-8")
    cfg = G1_CFG.read_text(encoding="utf-8")
    scene_cfg = TRACKING_CFG.read_text(encoding="utf-8")
    assert 'frontres_left_foot_contacts = ContactSensorCfg(' in cfg
    assert 'frontres_right_foot_contacts = ContactSensorCfg(' in cfg
    assert 'filter_prim_paths_expr=ground_filter' in cfg
    assert cfg.count("update_period=0.0") >= 2
    assert cfg.count("force_threshold=10.0") >= 2
    assert 'ground_filter = ["/World/ground/terrain/mesh"]' in cfg
    assert 'ground_filter = ["/World/ground/terrain"]' not in cfg
    assert 'ground_filter = ["/World/ground/terrain/.*"]' not in cfg
    assert 'prim_path="/World/ground"' in scene_cfg
    assert 'terrain_type="generator"' in scene_cfg
    assert "track_contact_points=" not in cfg
    assert "max_contact_data_count_per_prim=" not in cfg


def test_raw_contact_owner_unpacking() -> None:
    owner = _owner()
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
    points, forces, normals, valid = owner.read_frontres_raw_filtered_contact_rows(
        sensor,
        num_envs=2,
        device=torch.device("cpu"),
    )
    assert tuple(points.shape) == (2, 1, 2, 3)
    torch.testing.assert_close(forces[:, 0], torch.tensor([[10.0, 30.0], [20.0, 0.0]]))
    torch.testing.assert_close(points[1, 0, 0], torch.tensor([2.0, 0.0, 0.0]))
    assert valid[:, 0].tolist() == [[True, True], [True, False]]


def test_raw_contact_capacity_saturation_remains_fail_closed() -> None:
    owner = _owner()
    capacity = 4
    raw = (
        torch.ones(capacity, 1),
        torch.zeros(capacity, 3),
        torch.tensor([[0.0, 0.0, 1.0]]).repeat(capacity, 1),
        torch.zeros(capacity, 1),
        torch.tensor([[2], [2]], dtype=torch.long),
        torch.tensor([[0], [2]], dtype=torch.long),
    )
    sensor = SimpleNamespace(
        _sim_physics_dt=0.005,
        _frontres_raw_contact_capacity=capacity,
        contact_physx_view=SimpleNamespace(get_contact_data=lambda dt: raw),
    )
    try:
        owner.read_frontres_raw_filtered_contact_rows(
            sensor,
            num_envs=2,
            device=torch.device("cpu"),
        )
    except RuntimeError as exc:
        assert "reached capacity" in str(exc)
    else:
        raise AssertionError("a saturated raw-contact buffer was silently accepted")


def test_legacy_contact_view_capacity_upgrade_and_fail_closed() -> None:
    owner = _owner()

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
    result = owner.ensure_frontres_raw_contact_view(sensor, num_envs=8)
    assert result is upgraded and sensor._contact_physx_view is upgraded
    assert sensor._frontres_raw_contact_capacity == 2048
    assert calls == [
        (
            "/World/envs/env_*/Robot/(left_ankle_roll_link)",
            {
                "filter_patterns": ["/World/ground/terrain/mesh"],
                "max_contact_data_count": 2048,
            },
        )
    ]

    bad = SimpleNamespace(count=7, filter_count=1)
    sensor._frontres_raw_contact_capacity = 0
    sensor._physics_sim_view = SimpleNamespace(create_rigid_contact_view=lambda *args, **kwargs: bad)
    try:
        owner.ensure_frontres_raw_contact_view(sensor, num_envs=8)
    except RuntimeError as exc:
        assert "body/env identity" in str(exc)
    else:
        raise AssertionError("raw contact view silently changed role/env identity")


def test_formal_zmp_capture_preserves_first_invalid_error() -> None:
    tree = ast.parse(PHYSICS_OWNER.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_contact_wrench_zmp_pair"
    )
    namespace: dict[str, Any] = {"Any": Any, "Mapping": __import__("collections.abc").abc.Mapping, "torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PHYSICS_OWNER), "exec"), namespace)
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
    owner = _owner()

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
    unpacked = owner.read_frontres_raw_filtered_contact_rows(
        sensor,
        num_envs=num_envs,
        device=torch.device("cpu"),
    )
    zmp, valid = owner.contact_wrench_zmp_xy(*unpacked)
    assert tuple(zmp.shape) == (num_envs, 2)
    assert bool(valid.all()) and bool(torch.isfinite(zmp).all())
    torch.testing.assert_close(zmp, points_w[:, :2])


def test_raw_views_are_installed_before_reset_and_never_lazily_on_read() -> None:
    source = FORMAL_TRANSACTION.read_text(encoding="utf-8")
    balance = BALANCE.read_text(encoding="utf-8")
    builder = source[
        source.index("def _build_frontres_v015_local_transaction_request") :
        source.index("def _build_frontres_v015_local_identity_sentinel_request")
    ]
    assert builder.index("prepare_frontres_raw_contact_views(runner)") < builder.index(
        "prepare_frontres_v015_local_sentinel_batch(runner)"
    )
    assert builder.index("prepare_frontres_raw_contact_views(runner)") < builder.index(
        'reset_phase("clean_baseline")'
    )
    raw_reader = balance[
        balance.index("def read_frontres_raw_filtered_contact_rows") : balance.index(
            "def pad_frontres_raw_contact_slots"
        )
    ]
    assert "ensure_frontres_raw_contact_view(" not in raw_reader
    assert "installed before the scored physics step" in raw_reader
    owner = _owner()

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
    owner.prepare_frontres_raw_contact_views(runner)
    assert events == ["install-left", "install-right"]
    assert left._frontres_raw_contact_capacity == 2048
    assert right._frontres_raw_contact_capacity == 2048


def test_asymmetric_foot_contact_slots_pad_without_changing_evidence() -> None:
    owner = _owner()

    def raw(slots: int, x: float) -> tuple[torch.Tensor, ...]:
        points = torch.zeros(8, 1, slots, 3)
        points[:, :, 0, 0] = x
        forces = torch.zeros(8, 1, slots)
        forces[:, :, 0] = 100.0
        normals = torch.zeros_like(points)
        normals[:, :, 0, 2] = 1.0
        valid = forces > 0.0
        return points, forces, normals, valid

    left = owner.pad_frontres_raw_contact_slots(raw(10, -0.1), contact_slots=10)
    right = owner.pad_frontres_raw_contact_slots(raw(3, 0.1), contact_slots=10)
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

    swapped_left = owner.pad_frontres_raw_contact_slots(raw(3, 0.1), contact_slots=10)
    swapped_right = owner.pad_frontres_raw_contact_slots(raw(10, -0.1), contact_slots=10)
    swapped_zmp, swapped_valid = _owner().contact_wrench_zmp_xy(
        torch.cat((swapped_left[0], swapped_right[0]), dim=1),
        torch.cat((swapped_left[1], swapped_right[1]), dim=1),
        torch.cat((swapped_left[2], swapped_right[2]), dim=1),
        torch.cat((swapped_left[3], swapped_right[3]), dim=1),
    )
    assert torch.equal(swapped_valid, zmp_valid)
    torch.testing.assert_close(swapped_zmp, zmp)


if __name__ == "__main__":
    test_execution_foot_pos_is_environment_local_and_row_aligned()
    test_execution_foot_pos_origin_identity_fails_closed()
    test_contact_wrench_golden_and_permutation()
    test_expected_envelope_and_physical_no_load_is_na()
    test_expected_support_envelope_is_derived_from_clean_foot_pose()
    test_formal_owner_isolation()
    test_raw_contact_owner_unpacking()
    test_raw_contact_capacity_saturation_remains_fail_closed()
    test_legacy_contact_view_capacity_upgrade_and_fail_closed()
    test_formal_zmp_capture_preserves_first_invalid_error()
    test_eight_role_ground_contact_reaches_finite_zmp()
    test_raw_views_are_installed_before_reset_and_never_lazily_on_read()
    test_asymmetric_foot_contact_slots_pad_without_changing_evidence()
    print("frontres_contact_wrench_zmp_contract: ok", flush=True)
