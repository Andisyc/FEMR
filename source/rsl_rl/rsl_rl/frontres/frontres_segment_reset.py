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
