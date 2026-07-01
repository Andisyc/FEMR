from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FrontRESSegmentResetRequest:
    segment_ids: torch.Tensor
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    reference_window: Any
    mode: tuple[str, ...]
    preroll_steps: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class FrontRESSegmentResetResult:
    success_mask: torch.Tensor
    direct_reset_mask: torch.Tensor
    preroll_mask: torch.Tensor
    invalid_static_reset_mask: torch.Tensor
    fall_at_reset_mask: torch.Tensor
    contact_mismatch_mask: torch.Tensor
    velocity_mismatch: torch.Tensor
    diagnostics: dict[str, float]


class FrontRESSegmentResetAdapter:
    """Build and validate dynamic segment reset requests.

    The adapter is intentionally simulator-thin.  It checks that a request
    carries dynamic state, then passes the request to an env-owned reset hook.
    """

    def __init__(
        self,
        default_preroll_steps: int = 0,
        velocity_mismatch_tolerance: float = 1e-3,
    ) -> None:
        if default_preroll_steps < 0:
            raise ValueError("default_preroll_steps must be non-negative")
        if velocity_mismatch_tolerance < 0.0:
            raise ValueError("velocity_mismatch_tolerance must be non-negative")
        self.default_preroll_steps = int(default_preroll_steps)
        self.velocity_mismatch_tolerance = float(velocity_mismatch_tolerance)

    def build_request(self, batch: Any, mode: str = "auto") -> FrontRESSegmentResetRequest:
        if mode not in {"auto", "direct", "preroll"}:
            raise ValueError(f"unsupported reset mode: {mode}")
        state = getattr(batch, "clean_state", None)
        if state is None:
            raise ValueError("batch must expose clean_state")
        self._validate_dynamic_state(state)
        segment_ids = getattr(batch, "segment_ids")
        count = int(segment_ids.numel())
        device = segment_ids.device
        needs_preroll = self.needs_preroll(batch)
        if mode == "direct":
            mode_names = tuple("direct" for _ in range(count))
            preroll = torch.zeros(count, dtype=torch.long, device=device)
        elif mode == "preroll":
            mode_names = tuple("preroll" for _ in range(count))
            preroll = torch.full((count,), self.default_preroll_steps, dtype=torch.long, device=device)
        else:
            mode_names = tuple("preroll" if bool(flag) else "direct" for flag in needs_preroll.tolist())
            preroll = torch.where(
                needs_preroll,
                torch.full((count,), self.default_preroll_steps, dtype=torch.long, device=device),
                torch.zeros(count, dtype=torch.long, device=device),
            )
        if self.default_preroll_steps == 0 and any(name == "preroll" for name in mode_names):
            preroll = torch.where(preroll > 0, preroll, torch.ones_like(preroll))
        return FrontRESSegmentResetRequest(
            segment_ids=segment_ids,
            root_pos=state.root_pos,
            root_quat=state.root_quat,
            root_lin_vel=state.root_lin_vel,
            root_ang_vel=state.root_ang_vel,
            dof_pos=state.dof_pos,
            dof_vel=state.dof_vel,
            reference_window=getattr(batch, "reference_window", None),
            mode=mode_names,
            preroll_steps=preroll,
            valid_mask=torch.ones(count, dtype=torch.bool, device=device),
        )

    def apply(self, env: Any, request: FrontRESSegmentResetRequest) -> FrontRESSegmentResetResult:
        hook = None
        for name in ("apply_frontres_segment_reset", "reset_to_segment", "set_segment_state"):
            if hasattr(env, name):
                hook = getattr(env, name)
                break
        if hook is None:
            raise AttributeError("env must define apply_frontres_segment_reset, reset_to_segment, or set_segment_state")
        result = hook(request)
        if isinstance(result, FrontRESSegmentResetResult):
            return result
        if result is None:
            result = {}
        return self._result_from_mapping(result, request)

    def validate_after_reset(
        self,
        obs: Any,
        infos: dict[str, Any],
        request: FrontRESSegmentResetRequest,
    ) -> FrontRESSegmentResetResult:
        mapping = dict(infos)
        if "root_lin_vel" in mapping:
            mapping["velocity_mismatch"] = self._velocity_mismatch(mapping["root_lin_vel"], request.root_lin_vel)
        if "dof_vel" in mapping:
            dof_mismatch = self._velocity_mismatch(mapping["dof_vel"], request.dof_vel)
            existing = mapping.get("velocity_mismatch", torch.zeros_like(dof_mismatch))
            mapping["velocity_mismatch"] = torch.maximum(existing.to(dof_mismatch.device), dof_mismatch)
        return self._result_from_mapping(mapping, request)

    def needs_preroll(self, batch: Any) -> torch.Tensor:
        segment_ids = getattr(batch, "segment_ids")
        flags = torch.zeros(int(segment_ids.numel()), dtype=torch.bool, device=segment_ids.device)
        specs = getattr(batch, "specs", None)
        if specs is not None:
            for i, spec in enumerate(specs):
                if getattr(spec, "reset_mode_hint", "auto") == "preroll":
                    flags[i] = True
        phase = getattr(batch, "phase", None)
        if phase is not None:
            phase = phase.to(segment_ids.device).float()
            flags = flags | ((phase > 0.45) & (phase < 0.55))
        return flags

    def _result_from_mapping(self, mapping: dict[str, Any], request: FrontRESSegmentResetRequest) -> FrontRESSegmentResetResult:
        count = int(request.segment_ids.numel())
        device = request.segment_ids.device
        success = self._bool_field(mapping, ("success_mask", "reset_success", "valid_mask"), count, device, True)
        fall = self._bool_field(mapping, ("fall_at_reset_mask", "fall_at_reset", "fall"), count, device, False)
        contact = self._bool_field(mapping, ("contact_mismatch_mask", "contact_mismatch"), count, device, False)
        velocity = self._float_field(mapping, ("velocity_mismatch",), count, device, 0.0)
        reference_applied = self._bool_field(mapping, ("reference_window_applied",), count, device, False)
        invalid_static = self._static_reset_mask(request)
        success = success & (~fall) & (~contact) & (velocity <= self.velocity_mismatch_tolerance) & (~invalid_static)
        direct = torch.tensor([name == "direct" for name in request.mode], dtype=torch.bool, device=device)
        preroll = torch.tensor([name == "preroll" for name in request.mode], dtype=torch.bool, device=device)
        diagnostics = {
            "reset_success_frac": float(success.float().mean().item()) if count else 0.0,
            "direct_frac": float(direct.float().mean().item()) if count else 0.0,
            "preroll_frac": float(preroll.float().mean().item()) if count else 0.0,
            "invalid_static_frac": float(invalid_static.float().mean().item()) if count else 0.0,
            "fall_at_reset_frac": float(fall.float().mean().item()) if count else 0.0,
            "contact_mismatch_frac": float(contact.float().mean().item()) if count else 0.0,
            "velocity_mismatch_mean": float(velocity.float().mean().item()) if count else 0.0,
            "reference_window_applied_frac": float(reference_applied.float().mean().item()) if count else 0.0,
        }
        return FrontRESSegmentResetResult(
            success_mask=success,
            direct_reset_mask=direct,
            preroll_mask=preroll,
            invalid_static_reset_mask=invalid_static,
            fall_at_reset_mask=fall,
            contact_mismatch_mask=contact,
            velocity_mismatch=velocity,
            diagnostics=diagnostics,
        )

    def _validate_dynamic_state(self, state: Any) -> None:
        for name in ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "dof_pos", "dof_vel"):
            value = getattr(state, name, None)
            if value is None:
                raise ValueError(f"dynamic reset state is missing {name}")
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        batch = int(state.root_pos.shape[0])
        required_shapes = {
            "root_pos": (batch, 3),
            "root_quat": (batch, 4),
            "root_lin_vel": (batch, 3),
            "root_ang_vel": (batch, 3),
        }
        for name, shape in required_shapes.items():
            if tuple(getattr(state, name).shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(getattr(state, name).shape)}")
        if state.dof_pos.ndim != 2 or tuple(state.dof_pos.shape) != tuple(state.dof_vel.shape):
            raise ValueError("dof_pos and dof_vel must be rank-2 tensors with the same shape")

    def _static_reset_mask(self, request: FrontRESSegmentResetRequest) -> torch.Tensor:
        dyn_mag = (
            request.root_lin_vel.abs().sum(dim=-1)
            + request.root_ang_vel.abs().sum(dim=-1)
            + request.dof_vel.abs().sum(dim=-1)
        )
        static_like = dyn_mag <= 0.0
        dynamic_required = torch.tensor([name == "direct" for name in request.mode], dtype=torch.bool, device=request.segment_ids.device)
        return dynamic_required & static_like

    def _velocity_mismatch(self, actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = actual.to(target.device).float() - target.float()
        return torch.linalg.vector_norm(diff.flatten(start_dim=1), dim=-1)

    def _bool_field(
        self,
        mapping: dict[str, Any],
        names: tuple[str, ...],
        count: int,
        device: torch.device,
        default: bool,
    ) -> torch.Tensor:
        for name in names:
            if name in mapping:
                return mapping[name].to(device=device).bool().flatten()
        return torch.full((count,), default, dtype=torch.bool, device=device)

    def _float_field(
        self,
        mapping: dict[str, Any],
        names: tuple[str, ...],
        count: int,
        device: torch.device,
        default: float,
    ) -> torch.Tensor:
        for name in names:
            if name in mapping:
                return mapping[name].to(device=device).float().flatten()
        return torch.full((count,), default, dtype=torch.float32, device=device)


class FrontRESSegmentLiveResetHook:
    def __init__(self, env: Any, *, robot_name: str = "robot", trace: bool = True) -> None:
        self.env = env
        self.base_env = _unwrap_env(env)
        self.robot_name = str(robot_name)
        self.trace = bool(trace)
        self.robot = _resolve_robot(self.base_env, self.robot_name)

    def __call__(self, request: FrontRESSegmentResetRequest) -> dict[str, torch.Tensor]:
        count = int(request.segment_ids.numel())
        device = _robot_device(self.robot, request.root_pos.device)
        env_ids = torch.arange(count, dtype=torch.long, device=device)
        num_envs = int(getattr(self.base_env, "num_envs", count) or count)
        if count > num_envs:
            raise ValueError(f"reset request has {count} rows but env exposes only {num_envs} envs")

        root_before = _optional_index(getattr(getattr(self.robot, "data", None), "root_pos_w", None), env_ids)
        root_state = torch.cat(
            [
                request.root_pos.to(device),
                request.root_quat.to(device),
                request.root_lin_vel.to(device),
                request.root_ang_vel.to(device),
            ],
            dim=-1,
        )
        with torch.inference_mode():
            self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
            self.robot.write_joint_state_to_sim(request.dof_pos.to(device), request.dof_vel.to(device), env_ids=env_ids)
            _reset_motion_command_state(self.base_env, env_ids)
            reference_applied = _apply_motion_reference_window(self.base_env, env_ids, request.reference_window)
            _reset_episode_length(self.env, env_ids)
            _reset_episode_length(self.base_env, env_ids)

        root_after = _optional_index(getattr(getattr(self.robot, "data", None), "root_pos_w", None), env_ids)
        root_lin_after = _optional_index(getattr(getattr(self.robot, "data", None), "root_lin_vel_w", None), env_ids)
        dof_vel_after = _optional_index(getattr(getattr(self.robot, "data", None), "joint_vel", None), env_ids)
        velocity_mismatch = torch.zeros(count, dtype=torch.float32, device=device)
        if root_lin_after is not None:
            velocity_mismatch = torch.maximum(
                velocity_mismatch,
                torch.linalg.vector_norm(root_lin_after.float() - request.root_lin_vel.to(device).float(), dim=-1),
            )
        if dof_vel_after is not None:
            velocity_mismatch = torch.maximum(
                velocity_mismatch,
                torch.linalg.vector_norm(dof_vel_after.float() - request.dof_vel.to(device).float(), dim=-1),
            )
        success = torch.ones(count, dtype=torch.bool, device=device)
        if self.trace:
            print(
                "[FrontRES Segment Live Reset Hook] "
                f"count={count} "
                f"env_ids={env_ids.detach().cpu().tolist()} "
                f"segment_ids={request.segment_ids.detach().cpu().tolist()} "
                f"mode={tuple(request.mode)} "
                f"root_before={_trace_tensor(root_before)} "
                f"root_after={_trace_tensor(root_after)} "
                f"dof_pos={_trace_tensor(request.dof_pos)} "
                f"dof_vel={_trace_tensor(request.dof_vel)} "
                f"reference_window={_trace_tensor(request.reference_window if isinstance(request.reference_window, torch.Tensor) else None)} "
                f"reference_applied={reference_applied.detach().cpu().tolist()} "
                f"velocity_mismatch_mean={float(velocity_mismatch.float().mean().detach().cpu().item()):.6f}",
                flush=True,
            )
        return {
            "reset_success": success.to(request.segment_ids.device),
            "velocity_mismatch": velocity_mismatch.to(request.segment_ids.device),
            "reference_window_applied": reference_applied.to(request.segment_ids.device),
        }


def ensure_frontres_segment_live_reset_hook(
    env: Any,
    *,
    robot_name: str = "robot",
    trace: bool = True,
) -> FrontRESSegmentLiveResetHook:
    existing = getattr(env, "_frontres_segment_live_reset_hook", None)
    if isinstance(existing, FrontRESSegmentLiveResetHook):
        return existing
    hook = FrontRESSegmentLiveResetHook(env, robot_name=robot_name, trace=trace)
    setattr(env, "_frontres_segment_live_reset_hook", hook)
    setattr(env, "apply_frontres_segment_reset", hook)
    return hook


def _unwrap_env(env: Any) -> Any:
    current = env
    for _ in range(8):
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is None or unwrapped is current:
            return current
        current = unwrapped
    return current


def _resolve_robot(base_env: Any, robot_name: str) -> Any:
    scene = getattr(base_env, "scene", None)
    if scene is None:
        raise AttributeError("Segment live reset requires env.unwrapped.scene")
    try:
        return scene[robot_name]
    except (KeyError, TypeError):
        pass
    if hasattr(scene, robot_name):
        return getattr(scene, robot_name)
    raise AttributeError(f"could not resolve robot {robot_name!r} from env scene")


def _robot_device(robot: Any, fallback: torch.device) -> torch.device:
    data = getattr(robot, "data", None)
    root = getattr(data, "root_pos_w", None)
    if isinstance(root, torch.Tensor):
        return root.device
    return torch.device(fallback)


def _optional_index(tensor: Any, env_ids: torch.Tensor) -> torch.Tensor | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    return tensor.index_select(0, env_ids.to(tensor.device)).detach().clone()


def _reset_motion_command_state(base_env: Any, env_ids: torch.Tensor) -> None:
    command = _motion_command(base_env)
    if command is None:
        return
    with torch.inference_mode():
        _zero_indexed(command, "_frontres_pos_correction", env_ids)
        quat = getattr(command, "_frontres_quat_correction", None)
        if isinstance(quat, torch.Tensor):
            ids = env_ids.to(quat.device)
            quat[ids] = 0.0
            quat[ids, 0] = 1.0
        perturber = getattr(command, "perturber", None)
        reset_envs = getattr(perturber, "reset_envs", None)
        if callable(reset_envs):
            reset_envs(env_ids)


def _motion_command(base_env: Any) -> Any | None:
    manager = getattr(base_env, "command_manager", None)
    if manager is None or not hasattr(manager, "get_term"):
        return None
    try:
        return manager.get_term("motion")
    except Exception:
        return None


def _apply_motion_reference_window(
    base_env: Any,
    env_ids: torch.Tensor,
    reference_window: Any,
) -> torch.Tensor:
    if not isinstance(reference_window, torch.Tensor):
        return torch.zeros(int(env_ids.numel()), dtype=torch.bool, device=env_ids.device)
    if reference_window.ndim < 2 or int(reference_window.shape[0]) != int(env_ids.numel()):
        raise ValueError(
            "reference_window must be a tensor with first dimension matching reset env count, "
            f"got {tuple(reference_window.shape)} for {int(env_ids.numel())} envs"
        )
    command = _motion_command(base_env)
    if command is None:
        return torch.zeros(int(env_ids.numel()), dtype=torch.bool, device=env_ids.device)
    for name in ("set_frontres_reference_window", "apply_frontres_reference_window", "set_segment_reference_window"):
        method = getattr(command, name, None)
        if callable(method):
            return _call_reference_window_hook(method, reference_window, env_ids)
    stored = getattr(command, "_frontres_reference_window", None)
    if isinstance(stored, torch.Tensor):
        ids = env_ids.to(stored.device)
        stored[ids].copy_(reference_window.to(stored.device))
        return torch.ones(int(env_ids.numel()), dtype=torch.bool, device=env_ids.device)
    return torch.zeros(int(env_ids.numel()), dtype=torch.bool, device=env_ids.device)


def _call_reference_window_hook(method: Any, reference_window: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    try:
        result = method(reference_window=reference_window, env_ids=env_ids)
    except TypeError:
        try:
            result = method(reference_window, env_ids=env_ids)
        except TypeError:
            result = method(reference_window, env_ids)
    return _coerce_reference_applied(result, env_ids)


def _coerce_reference_applied(result: Any, env_ids: torch.Tensor) -> torch.Tensor:
    count = int(env_ids.numel())
    if result is None:
        return torch.ones(count, dtype=torch.bool, device=env_ids.device)
    if isinstance(result, torch.Tensor):
        applied = result.to(device=env_ids.device).bool().reshape(-1)
        if int(applied.numel()) != count:
            raise ValueError(f"reference hook result must have {count} rows, got {int(applied.numel())}")
        return applied.detach()
    if isinstance(result, (list, tuple)):
        if len(result) != count:
            raise ValueError(f"reference hook result must have {count} rows, got {len(result)}")
        return torch.tensor(result, dtype=torch.bool, device=env_ids.device)
    return torch.full((count,), bool(result), dtype=torch.bool, device=env_ids.device)


def _zero_indexed(owner: Any, name: str, env_ids: torch.Tensor) -> None:
    value = getattr(owner, name, None)
    if isinstance(value, torch.Tensor):
        with torch.inference_mode():
            value[env_ids.to(value.device)] = 0.0


def _reset_episode_length(env: Any, env_ids: torch.Tensor) -> None:
    value = getattr(env, "episode_length_buf", None)
    if isinstance(value, torch.Tensor):
        with torch.inference_mode():
            value[env_ids.to(value.device)] = 0


def _trace_tensor(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    tensor = value.detach()
    if tensor.numel() == 0:
        return {"shape": tuple(tensor.shape), "numel": 0}
    numeric = tensor.float()
    return {
        "shape": tuple(tensor.shape),
        "device": str(tensor.device),
        "finite": bool(torch.isfinite(numeric).all().item()),
        "min": float(numeric.min().item()),
        "max": float(numeric.max().item()),
        "mean": float(numeric.mean().item()),
        "requires_grad": bool(tensor.requires_grad),
    }
