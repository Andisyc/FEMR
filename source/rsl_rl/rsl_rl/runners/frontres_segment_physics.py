"""IsaacLab motion-quality, Contact, and ZMP evidence gateway."""





from __future__ import annotations





from collections.abc import Mapping

from dataclasses import dataclass


from typing import Any


import torch


from rsl_rl.frontres.frontres_balance import ensure_frontres_raw_contact_view, pad_frontres_raw_contact_slots, prepare_frontres_raw_contact_views, read_frontres_raw_filtered_contact_rows


from rsl_rl.runners.frontres_formal_runtime_audit import emit_formal_runtime_probe





from rsl_rl.runners.frontres_segment_probe_logging import (
    audit_identity_kwargs as _audit_identity_kwargs,
    try_frontres_motion_command as _motion_command_for_runner,
)


@dataclass(frozen=True)
class FrontRESV017ExecutionFrame:
    """One framework-owned execution frame before Gain aggregation."""

    joint_pos: torch.Tensor
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    key_body_pos: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    foot_pos: torch.Tensor
    expected_support: torch.Tensor
    contact: torch.Tensor
    zmp_margin: torch.Tensor

    def validate(self) -> None:
        rows = int(self.joint_pos.shape[0]) if self.joint_pos.ndim == 2 else -1
        shapes = {
            "joint_pos": (rows, 29),
            "root_pos": (rows, 3),
            "root_quat": (rows, 4),
            "root_lin_vel": (rows, 3),
            "root_ang_vel": (rows, 3),
            "foot_pos": (rows, 2, 3),
            "expected_support": (rows, 2),
            "contact": (rows, 2),
            "zmp_margin": (rows,),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"v017 execution frame {name} must be {shape}")
        if (
            self.key_body_pos.ndim != 3
            or tuple(self.key_body_pos.shape[:1]) != (rows,)
            or int(self.key_body_pos.shape[1]) <= 0
            or int(self.key_body_pos.shape[2]) != 3
        ):
            raise ValueError("v017 execution frame key_body_pos must be [N,J,3]")
        if bool(((self.expected_support != 0) & (self.expected_support != 1)).any()) or bool(
            ((self.contact != 0) & (self.contact != 1)).any()
        ):
            raise ValueError("v017 execution frame support and Contact must be binary")
        required = (
            self.joint_pos,
            self.root_pos,
            self.root_quat,
            self.key_body_pos,
            self.root_lin_vel,
            self.root_ang_vel,
            self.foot_pos,
        )
        if any(value.requires_grad or not bool(torch.isfinite(value.float()).all()) for value in required):
            raise ValueError("v017 execution frame state must be detached and finite")
        applicable = self.expected_support.bool().any(dim=-1) & self.contact.bool().any(dim=-1)
        finite = torch.isfinite(self.zmp_margin.float())
        if not bool(finite[applicable].all()) or bool(finite[~applicable].any()):
            raise ValueError("v017 execution frame ZMP must be finite exactly on loaded support")





def _capture_motion_quality_frame(
    runner: Any,
    pair_layout: Any,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    # QUALITY-EXEC-01: 检查 applied repair -> frozen-GMT physical execution evidence.
    # Result: PENDING_Q_EVIDENCE; Q-E3 only proves execution callback connectivity.
    # B1: env.step 后 role states 尚在时捕获 success/fall/survival 与 action identity.
    # B2: 同帧记录 ZMP/contact/MPJPE/velocity/acceleration evidence.
    # B3: Gain/sequence aggregator 前保留 short-K 与 long-sequence metric boundary.
    """截获同一 quartet frame 的 Clean/Repaired/Noisy Style evidence.

    函数名说明:
        `_capture_motion_quality_frame` 是 paired Style capture adapter, 只对齐并
        返回 root-relative body positions; 它不是 MPJPE 聚合器或 Gain 公式.

    主链路:
        上游: env.step 后的 motion command 和 split-env pair layout.
        下游: `compute_segment_gain` 的 Style component 比较 matching motion/frame.

    语义:
        三个分支必须来自同一 motion/frame. 任一字段缺失时返回 None, diagnostics
        应标记 UNCONFIRMED, 不得静默写成 0.
    """
    # B1: 从一个 quartet frame 读取 matching Clean, Repaired 和 Noisy rows.
    command = _motion_command_for_runner(runner)
    if command is None:
        return None, None, None
    clean_ref = getattr(command, "body_pos_w", None)
    if not isinstance(clean_ref, torch.Tensor):
        clean_ref = getattr(command, "body_pos_relative_w", None)
    robot_pos = getattr(command, "robot_body_pos_w", None)
    if not isinstance(clean_ref, torch.Tensor) or not isinstance(robot_pos, torch.Tensor):
        return None, None, None
    n_train = max(0, int(pair_layout.n_train))
    n_candidate = max(0, int(pair_layout.n_candidate))
    n_base = max(0, int(pair_layout.n_base))
    n_clean = max(0, int(pair_layout.n_clean))
    n = min(n_train, n_base, n_clean)
    base_start = n_train + n_candidate
    clean_start = base_start + n_base
    if n <= 0 or int(robot_pos.shape[0]) < base_start + n or int(clean_ref.shape[0]) < clean_start + n:
        return None, None, None
    # B2: 按 role 对齐 root-relative body positions, 不跨 motion 聚合.
    frame = (
        _root_relative_body_pos(clean_ref[clean_start : clean_start + n]),
        _root_relative_body_pos(robot_pos[:n]),
        _root_relative_body_pos(robot_pos[base_start : base_start + n]),
    )
    # AUDIT-PAIR-EVIDENCE-01: Record style evidence before canonical Gain consumes it.
    # Result: E67 LIVE PASS for one capture; style evidence shares the
    # canonical transaction/batch identity.
    emit_formal_runtime_probe(
        "AUDIT-PAIR-EVIDENCE-01",
        clean_positions=frame[0],
        repaired_positions=frame[1],
        noisy_positions=frame[2],
        **_audit_identity_kwargs(getattr(runner, "_frontres_segment_live_audit_identity", None)),
    )
    return frame


def _capture_root_orientation_frame(
    runner: Any,
    pair_layout: Any,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Capture Clean-target and executed root quaternions for one quartet.

    Status: active Style capture boundary.
    Upstream: motion command quartet and robot anchor state after env.step.
    Downstream: frontres_gain geodesic Style component.
    Evidence: source-confirmed fields; runtime availability still requires S4.
    Gap: absent anchor quaternions remain UNCONFIRMED rather than zero.
    """
    command = _motion_command_for_runner(runner)
    if command is None:
        return None, None, None
    clean_ref = getattr(command, "anchor_quat_w_original", None)
    robot_quat = getattr(command, "robot_anchor_quat_w", None)
    if not isinstance(clean_ref, torch.Tensor) or not isinstance(robot_quat, torch.Tensor):
        return None, None, None
    n_train = max(0, int(pair_layout.n_train))
    n_candidate = max(0, int(pair_layout.n_candidate))
    n_base = max(0, int(pair_layout.n_base))
    n_clean = max(0, int(pair_layout.n_clean))
    n = min(n_train, n_base, n_clean)
    base_start = n_train + n_candidate
    clean_start = base_start + n_base
    if n <= 0 or clean_ref.shape[-1] != 4 or robot_quat.shape[-1] != 4:
        return None, None, None
    if int(clean_ref.shape[0]) < clean_start + n or int(robot_quat.shape[0]) < base_start + n:
        return None, None, None
    return (
        clean_ref[clean_start : clean_start + n].detach().clone(),
        robot_quat[:n].detach().clone(),
        robot_quat[base_start : base_start + n].detach().clone(),
    )


def _capture_physics_frame(
    runner: Any,
    pair_layout: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Capture paired ZMP plus sealed expected and sensor-authoritative Contact.

    函数名说明:
        `_capture_physics_frame` 是 paired Physics capture adapter, 读取 frozen-GMT
        执行结果; 它不是 environment reward, 也不构造 Style Gain.

    主链路:
        上游: env.step 后的 robot state, motion command 和 paired role layout.
        下游: `compute_paired_physics_gain` 比较 Repaired/Noisy executability.

    语义:
        ZMP/support 必须按同一 Repair/Noisy frame 配对. Actual Contact 只来自
        已配置的 contact_forces ContactSensor；缺失时 fail closed.
    """
    # B1: 读取同一 quartet frame 的 paired frozen-GMT execution state.
    command = _motion_command_for_runner(runner)
    if command is None:
        return None
    n_train = max(0, int(pair_layout.n_train))
    n_candidate = max(0, int(pair_layout.n_candidate))
    n_base = max(0, int(pair_layout.n_base))
    n = min(n_train, n_base)
    if n <= 0:
        return None
    contact = _contact_sensor_pair(runner, command, pair_layout, n)
    if contact is None:
        return None
    expected_support, contact_repaired, contact_noisy = contact
    zmp_pair = _contact_wrench_zmp_pair(
        runner,
        command,
        pair_layout,
        expected_support,
        contact_repaired,
        contact_noisy,
        n,
    )
    if zmp_pair is None:
        return None
    zmp_repaired, zmp_noisy = zmp_pair
    # B2: 对齐 Repaired/Noisy ZMP 和 contact evidence, 产出 canonical Physics 输入.
    frame = (zmp_repaired, zmp_noisy, expected_support, contact_repaired, contact_noisy)
    # AUDIT-PAIR-EVIDENCE-01: Record physics evidence beside style evidence.
    # Result: E67 LIVE PASS for one capture; physics evidence shares the
    # canonical transaction/batch identity.
    emit_formal_runtime_probe(
        "AUDIT-PAIR-EVIDENCE-01",
        zmp_repaired=frame[0],
        zmp_noisy=frame[1],
        expected_support=frame[2],
        contact_repaired=frame[3],
        contact_noisy=frame[4],
        **_audit_identity_kwargs(getattr(runner, "_frontres_segment_live_audit_identity", None)),
    )
    return frame


def _capture_v015_quality_lateral_lean_frame(
    runner: Any,
    pair_layout: Any,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Read paired robot root roll for evaluation only; never expose it to training."""

    command = _motion_command_for_runner(runner)
    robot_quat = getattr(command, "robot_anchor_quat_w", None) if command is not None else None
    n = min(max(0, int(pair_layout.n_train)), max(0, int(pair_layout.n_base)))
    base_start = int(pair_layout.n_train) + int(pair_layout.n_candidate)
    if (
        not isinstance(robot_quat, torch.Tensor)
        or n <= 0
        or robot_quat.ndim != 2
        or int(robot_quat.shape[1]) != 4
        or int(robot_quat.shape[0]) < base_start + n
    ):
        return None

    def roll_wxyz(quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat.unbind(dim=-1)
        return torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))

    return (
        roll_wxyz(robot_quat[:n]).detach().clone(),
        roll_wxyz(robot_quat[base_start : base_start + n]).detach().clone(),
    )


def _contact_sensor_pair(
    runner: Any,
    command: Any,
    pair_layout: Any,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    snapshot = getattr(command, "frontres_local_scenario_k_execution_snapshot", None)
    if not callable(snapshot):
        return None
    sealed = snapshot()
    support_rows = sealed.get("expected_support") if isinstance(sealed, Mapping) else None
    if not isinstance(support_rows, torch.Tensor) or tuple(support_rows.shape) != (int(command.num_envs), 2):
        return None
    n_train = int(pair_layout.n_train)
    n_candidate = int(pair_layout.n_candidate)
    n_base = int(pair_layout.n_base)
    base_start = n_train + n_candidate
    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    scene = getattr(env, "scene", None)
    if scene is None:
        return None
    # B1: Actual support 与 raw ZMP 必须服从同一 foot-to-ground filtered view.
    # 未过滤 net_forces_w 会包含足部与机器人/其他物体的接触, 不能定义地面支撑.
    actual_feet: list[torch.Tensor] = []
    sensors = getattr(scene, "sensors", None)
    for name in ("frontres_left_foot_contacts", "frontres_right_foot_contacts"):
        try:
            sensor = scene[name]
        except (KeyError, TypeError, AssertionError):
            sensor = sensors.get(name) if isinstance(sensors, Mapping) else None
        if sensor is None:
            return None
        force_matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
        if not isinstance(force_matrix, torch.Tensor):
            return None
        force_matrix = force_matrix.to(device=runner.device, dtype=torch.float32)
        expected_shape = (int(command.num_envs), 1)
        if (
            force_matrix.ndim != 4
            or tuple(force_matrix.shape[:2]) != expected_shape
            or int(force_matrix.shape[2]) <= 0
            or int(force_matrix.shape[3]) != 3
        ):
            raise RuntimeError(
                f"{name} filtered force matrix must be [N,1,F,3], got {tuple(force_matrix.shape)}"
            )
        if not bool(torch.isfinite(force_matrix).all()):
            raise RuntimeError(f"{name} filtered force matrix must be finite")
        threshold_value = getattr(getattr(sensor, "cfg", None), "force_threshold", None)
        if not isinstance(threshold_value, (int, float)) or isinstance(threshold_value, bool):
            raise RuntimeError(f"{name} requires an explicit numeric force threshold")
        threshold = float(threshold_value)
        if threshold <= 0.0:
            raise RuntimeError(f"{name} requires a positive force threshold")
        vertical_ground_load = force_matrix[..., 2].sum(dim=(1, 2)).abs()
        actual_feet.append(vertical_ground_load >= threshold)
    actual = torch.stack(actual_feet, dim=-1)
    expected_repair = support_rows[:n].bool()
    expected_noisy = support_rows[base_start : base_start + n].bool()
    if not torch.equal(expected_repair, expected_noisy):
        raise RuntimeError("FRS-GAIN-v006 paired roles do not share sealed expected support identity")
    return expected_repair, actual[:n], actual[base_start : base_start + n]


_ensure_frontres_raw_contact_view = ensure_frontres_raw_contact_view


_prepare_frontres_raw_contact_views = prepare_frontres_raw_contact_views


_raw_filtered_contact_rows = read_frontres_raw_filtered_contact_rows


_pad_raw_contact_slots = pad_frontres_raw_contact_slots


def _contact_wrench_zmp_pair(
    runner: Any,
    command: Any,
    pair_layout: Any,
    expected_support: torch.Tensor,
    contact_repaired: torch.Tensor,
    contact_noisy: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Produce paired true contact-wrench ZMP margins; no proxy fallback exists."""

    snapshot = getattr(command, "frontres_local_scenario_k_execution_snapshot", None)
    if not callable(snapshot):
        raise RuntimeError("contact-wrench ZMP requires the sealed local-scenario K snapshot")
    sealed = snapshot()
    envelope = sealed.get("expected_support_envelope") if isinstance(sealed, Mapping) else None
    if not isinstance(envelope, torch.Tensor) or tuple(envelope.shape) != (int(command.num_envs), 6):
        shape = tuple(envelope.shape) if isinstance(envelope, torch.Tensor) else None
        raise RuntimeError(
            "contact-wrench ZMP requires sealed expected_support_envelope "
            f"[{int(command.num_envs)},6], got {shape}"
        )
    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    scene = getattr(env, "scene", None)
    if scene is None:
        raise RuntimeError("contact-wrench ZMP requires the formal IsaacLab scene")
    try:
        left_sensor = scene["frontres_left_foot_contacts"]
        right_sensor = scene["frontres_right_foot_contacts"]
        from rsl_rl.frontres.frontres_balance import contact_wrench_zmp_xy, expected_support_envelope_margin

        raw_left = read_frontres_raw_filtered_contact_rows(
            left_sensor, num_envs=int(command.num_envs), device=runner.device
        )
        raw_right = read_frontres_raw_filtered_contact_rows(
            right_sensor, num_envs=int(command.num_envs), device=runner.device
        )
        contact_slots = max(int(raw_left[0].shape[2]), int(raw_right[0].shape[2]))
        raw_left = pad_frontres_raw_contact_slots(raw_left, contact_slots=contact_slots)
        raw_right = pad_frontres_raw_contact_slots(raw_right, contact_slots=contact_slots)
        points = torch.cat((raw_left[0], raw_right[0]), dim=1)
        forces = torch.cat((raw_left[1], raw_right[1]), dim=1)
        normals = torch.cat((raw_left[2], raw_right[2]), dim=1)
        valid = torch.cat((raw_left[3], raw_right[3]), dim=1)
        zmp_xy, zmp_valid = contact_wrench_zmp_xy(points, forces, normals, valid)
        origins_xy = getattr(scene, "env_origins", None)
        if not isinstance(origins_xy, torch.Tensor):
            raise RuntimeError("contact-wrench ZMP requires scene.env_origins")
        support_all = sealed.get("expected_support")
        if not isinstance(support_all, torch.Tensor) or tuple(support_all.shape) != (int(command.num_envs), 2):
            shape = tuple(support_all.shape) if isinstance(support_all, torch.Tensor) else None
            raise RuntimeError(
                f"contact-wrench ZMP requires sealed expected_support [{int(command.num_envs)},2], got {shape}"
            )
        margin = expected_support_envelope_margin(
            zmp_xy,
            envelope.to(device=runner.device),
            support_all.to(device=runner.device),
            env_origins_xy=origins_xy[:, :2].to(device=runner.device),
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"contact-wrench ZMP capture failed at {type(exc).__name__}: {exc}"
        ) from exc
    base_start = int(pair_layout.n_train) + int(pair_layout.n_candidate)
    expected_loaded = expected_support.to(device=runner.device, dtype=torch.bool).any(dim=-1)
    branch_rows = (
        ("Repair", slice(0, n), contact_repaired),
        ("Noisy", slice(base_start, base_start + n), contact_noisy),
    )
    outputs: list[torch.Tensor] = []
    for role, rows, actual_contact in branch_rows:
        actual_loaded = actual_contact.to(device=runner.device, dtype=torch.bool).any(dim=-1)
        required = expected_loaded & actual_loaded
        branch_valid = zmp_valid[rows]
        if bool((required & ~branch_valid).any()):
            missing_rows = torch.nonzero(required & ~branch_valid, as_tuple=False).reshape(-1).tolist()
            raise RuntimeError(
                f"{role} loaded support is missing a finite raw contact-wrench resultant; "
                f"branch_rows={missing_rows}"
            )
        branch_margin = margin[rows]
        outputs.append(
            torch.where(required, branch_margin, torch.full_like(branch_margin, float("nan"))).detach()
        )
    return outputs[0], outputs[1]


def _capture_v017_execution_frame(
    runner: Any,
    *,
    selected_rows: torch.Tensor,
) -> FrontRESV017ExecutionFrame:
    """Capture only the selected scored rows from vectorized sensors."""

    # B1: 校验 sealed scenario 与 scored role rows, 产出本次 capture identity.
    command = _motion_command_for_runner(runner)
    if command is None:
        raise RuntimeError("v017 execution capture requires the formal motion command")
    snapshot_owner = getattr(command, "frontres_local_scenario_k_execution_snapshot", None)
    if not callable(snapshot_owner):
        raise RuntimeError("v017 execution capture requires the sealed K snapshot")
    sealed = snapshot_owner()
    expected = sealed.get("expected_support") if isinstance(sealed, Mapping) else None
    envelope = sealed.get("expected_support_envelope") if isinstance(sealed, Mapping) else None
    allocated_rows = int(getattr(command, "num_envs", 0))
    ids = selected_rows.to(device=runner.device, dtype=torch.long).reshape(-1)
    if (
        int(ids.numel()) <= 0
        or int(torch.unique(ids).numel()) != int(ids.numel())
        or bool((ids < 0).any())
        or bool((ids >= allocated_rows).any())
    ):
        raise ValueError("v017 execution capture requires unique selected role rows")
    rows = int(ids.numel())
    if not isinstance(expected, torch.Tensor) or tuple(expected.shape) != (allocated_rows, 2):
        raise RuntimeError("v017 execution capture requires expected support [N,2]")
    if not isinstance(envelope, torch.Tensor) or tuple(envelope.shape) != (allocated_rows, 6):
        raise RuntimeError("v017 execution capture requires expected support envelope [N,6]")
    expected = expected.index_select(0, ids)
    envelope = envelope.index_select(0, ids)

    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    scene = getattr(env, "scene", None)
    if scene is None:
        raise RuntimeError("v017 execution capture requires the formal IsaacLab scene")
    # B2: 读取 ContactSensor 与 raw wrench, 产出 role-aligned Contact/ZMP evidence.
    actual_feet: list[torch.Tensor] = []
    raw_rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for name in ("frontres_left_foot_contacts", "frontres_right_foot_contacts"):
        try:
            sensor = scene[name]
        except (KeyError, TypeError, AssertionError) as exc:
            raise RuntimeError(f"v017 execution capture is missing {name}") from exc
        force_matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
        threshold = getattr(getattr(sensor, "cfg", None), "force_threshold", None)
        if (
            not isinstance(force_matrix, torch.Tensor)
            or force_matrix.ndim != 4
            or tuple(force_matrix.shape[:2]) != (allocated_rows, 1)
            or int(force_matrix.shape[-1]) != 3
            or not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or float(threshold) <= 0.0
        ):
            raise RuntimeError(f"v017 execution capture received malformed {name} payload")
        force_matrix = force_matrix.to(device=runner.device, dtype=torch.float32).index_select(0, ids)
        actual_feet.append(force_matrix[..., 2].sum(dim=(1, 2)).abs() >= float(threshold))
        raw = read_frontres_raw_filtered_contact_rows(sensor, num_envs=allocated_rows, device=runner.device)
        raw_rows.append(tuple(value.index_select(0, ids) for value in raw))
    contact = torch.stack(actual_feet, dim=-1)
    contact_slots = max(int(raw_rows[0][0].shape[2]), int(raw_rows[1][0].shape[2]))
    left = pad_frontres_raw_contact_slots(raw_rows[0], contact_slots=contact_slots)
    right = pad_frontres_raw_contact_slots(raw_rows[1], contact_slots=contact_slots)
    points = torch.cat((left[0], right[0]), dim=1)
    forces = torch.cat((left[1], right[1]), dim=1)
    normals = torch.cat((left[2], right[2]), dim=1)
    valid = torch.cat((left[3], right[3]), dim=1)
    from rsl_rl.frontres.frontres_balance import contact_wrench_zmp_xy, expected_support_envelope_margin

    zmp_xy, zmp_valid = contact_wrench_zmp_xy(points, forces, normals, valid)
    origins = getattr(scene, "env_origins", None)
    if not isinstance(origins, torch.Tensor) or tuple(origins.shape) != (allocated_rows, 3):
        raise RuntimeError("v017 execution capture requires scene.env_origins [N,3]")
    origins = origins.index_select(0, ids)
    if not bool(torch.isfinite(origins.float()).all()):
        raise RuntimeError("v017 execution capture requires finite scene.env_origins")
    margin = expected_support_envelope_margin(
        zmp_xy,
        envelope.to(device=runner.device),
        expected.to(device=runner.device),
        env_origins_xy=origins[:, :2].to(device=runner.device),
    )
    applicable = expected.bool().any(dim=-1) & contact.any(dim=-1)
    if bool((applicable & ~zmp_valid).any()):
        raise RuntimeError("v017 loaded support is missing a finite contact-wrench ZMP")
    zmp_margin = torch.where(applicable, margin, torch.full_like(margin, float("nan")))

    # B3: 捕获 dynamic state, 并将脚位置从 world frame 转成 environment-local frame.
    state_names = (
        "robot_joint_pos",
        "robot_anchor_pos_w",
        "robot_anchor_quat_w",
        "robot_body_pos_w",
        "robot_anchor_lin_vel_w",
        "robot_anchor_ang_vel_w",
    )
    state = {name: getattr(command, name, None) for name in state_names}
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError("v017 execution capture is missing robot dynamic state")
    state = {name: value.index_select(0, ids) for name, value in state.items()}
    body = state["robot_body_pos_w"]
    left_index = int(getattr(command, "left_foot_idx", -1))
    right_index = int(getattr(command, "right_foot_idx", -1))
    if min(left_index, right_index) < 0 or max(left_index, right_index) >= int(body.shape[1]):
        raise RuntimeError("v017 execution capture cannot resolve the two support feet")
    foot_pos_local = body[:, (left_index, right_index)] - origins.to(
        device=body.device,
        dtype=body.dtype,
    ).unsqueeze(1)
    # B4: 封存 immutable execution frame, 供 one-action-K evidence 与 Gain consumer 使用.
    frame = FrontRESV017ExecutionFrame(
        joint_pos=state["robot_joint_pos"].detach().clone(),
        root_pos=state["robot_anchor_pos_w"].detach().clone(),
        root_quat=state["robot_anchor_quat_w"].detach().clone(),
        key_body_pos=body.detach().clone(),
        root_lin_vel=state["robot_anchor_lin_vel_w"].detach().clone(),
        root_ang_vel=state["robot_anchor_ang_vel_w"].detach().clone(),
        foot_pos=foot_pos_local.detach().clone(),
        expected_support=expected.detach().clone(),
        contact=contact.detach().clone(),
        zmp_margin=zmp_margin.detach().clone(),
    )
    frame.validate()
    return frame


def _root_relative_body_pos(body_pos: torch.Tensor) -> torch.Tensor:
    if body_pos.ndim < 3 or int(body_pos.shape[-2]) <= 0:
        return body_pos.detach().clone()
    return (body_pos - body_pos[..., :1, :]).detach().clone()


def _stack_motion_quality_frames(frames: list[torch.Tensor]) -> torch.Tensor | None:
    if not frames:
        return None
    return torch.stack(frames, dim=1)


# Public IsaacLab evidence gateway surface.
capture_frontres_motion_quality_frame = _capture_motion_quality_frame
capture_frontres_root_orientation_frame = _capture_root_orientation_frame
capture_frontres_physics_frame = _capture_physics_frame
capture_frontres_quality_lateral_lean_frame = _capture_v015_quality_lateral_lean_frame
capture_frontres_v017_execution_frame = _capture_v017_execution_frame
stack_frontres_motion_quality_frames = _stack_motion_quality_frames
