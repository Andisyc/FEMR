from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch


@dataclass(frozen=True)
class FrontRESRepairAction:
    delta_se: torch.Tensor
    active_mask: torch.Tensor
    projected_delta_se: torch.Tensor
    action_norm: torch.Tensor
    per_dim_norm: torch.Tensor


@dataclass(frozen=True)
class FrontRESHRLActionStats:
    action_norm_mean: float
    action_norm_max: float
    per_dim_norm: torch.Tensor
    active_frac: float


class FrontRESHRLActionProjector:
    """Project HRL policy output into a bounded 6D Delta SE(3) repair action."""

    def __init__(
        self,
        action_cone: Any | None = None,
        active_task_dims: Iterable[int] | torch.Tensor | None = None,
        action_scale: float | Sequence[float] | torch.Tensor = 1.0,
        upward_dz_rule: str | float = "nonpositive",
        hsl_init_mode: str | None = None,
    ) -> None:
        self.action_cone = action_cone
        self.active_task_dims = active_task_dims
        self.action_scale = action_scale
        self.upward_dz_rule = upward_dz_rule
        self.hsl_init_mode = hsl_init_mode

    def project(
        self,
        raw_action: torch.Tensor,
        mode_groups: list[tuple[str, ...]] | tuple[tuple[str, ...], ...] | None = None,
    ) -> FrontRESRepairAction:
        if raw_action.ndim != 2 or raw_action.shape[-1] != 6:
            raise ValueError(f"raw_action must have shape [B, 6], got {tuple(raw_action.shape)}")
        scale = self._scale(raw_action.device, raw_action.dtype)
        delta_se = raw_action * scale.view(1, 6)
        active_mask = self._active_mask(raw_action.shape[0], raw_action.device, raw_action.dtype)
        if mode_groups is not None:
            active_mask = active_mask * self._mode_mask(mode_groups, raw_action.shape[0], raw_action.device, raw_action.dtype)
        projected = delta_se * active_mask
        projected = self._apply_dz_rule(projected)
        action_norm = torch.linalg.vector_norm(projected, dim=-1)
        per_dim_norm = projected.abs().mean(dim=0)
        return FrontRESRepairAction(
            delta_se=delta_se,
            active_mask=active_mask,
            projected_delta_se=projected,
            action_norm=action_norm,
            per_dim_norm=per_dim_norm,
        )

    def apply_to_reference(self, command: Any, repair_action: FrontRESRepairAction) -> Any:
        delta = repair_action.projected_delta_se
        if isinstance(command, dict):
            updated = dict(command)
            updated["frontres_delta_se"] = delta
            updated["frontres_pos_correction"] = delta[:, :3]
            updated["frontres_rpy_correction"] = delta[:, 3:6]
            return updated
        setattr(command, "_frontres_pos_correction", delta[:, :3])
        setattr(command, "_frontres_rpy_correction", delta[:, 3:6])
        setattr(command, "_frontres_delta_se", delta)
        return command

    def mask_for_segment(self, batch: Any) -> torch.Tensor:
        segment_ids = getattr(batch, "segment_ids", None)
        if segment_ids is None:
            raise ValueError("batch must expose segment_ids")
        count = int(segment_ids.numel())
        device = segment_ids.device
        mask = self._active_mask(count, device, torch.float32)
        families = getattr(batch, "perturbation_family", None)
        if families is not None:
            mode_groups = tuple((family,) if isinstance(family, str) else tuple(family) for family in families)
            mask = mask * self._mode_mask(mode_groups, count, device, torch.float32)
        return mask

    def stats(self, repair_action: FrontRESRepairAction) -> FrontRESHRLActionStats:
        active = repair_action.active_mask > 0
        return FrontRESHRLActionStats(
            action_norm_mean=float(repair_action.action_norm.mean().item()) if repair_action.action_norm.numel() else 0.0,
            action_norm_max=float(repair_action.action_norm.max().item()) if repair_action.action_norm.numel() else 0.0,
            per_dim_norm=repair_action.per_dim_norm.detach().clone(),
            active_frac=float(active.float().mean().item()) if active.numel() else 0.0,
        )

    @staticmethod
    def initialize_repair_actor_from_hsl(repair_actor: Any, hsl_actor: Any) -> tuple[str, ...]:
        repair_state = repair_actor.state_dict()
        hsl_state = hsl_actor.state_dict()
        copied: list[str] = []
        for key, value in repair_state.items():
            source = hsl_state.get(key)
            if source is not None and tuple(source.shape) == tuple(value.shape):
                repair_state[key] = source.detach().clone()
                copied.append(key)
        repair_actor.load_state_dict(repair_state, strict=True)
        return tuple(copied)

    def _scale(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(self.action_scale, torch.Tensor):
            scale = self.action_scale.to(device=device, dtype=dtype).flatten()
        elif isinstance(self.action_scale, (int, float)):
            scale = torch.full((6,), float(self.action_scale), device=device, dtype=dtype)
        else:
            scale = torch.tensor(list(self.action_scale), device=device, dtype=dtype).flatten()
        if scale.numel() != 6:
            raise ValueError(f"action_scale must contain 6 values, got {scale.numel()}")
        return scale

    def _active_mask(self, count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.active_task_dims is None:
            base = torch.ones(6, device=device, dtype=dtype)
        elif isinstance(self.active_task_dims, torch.Tensor):
            dims = self.active_task_dims.to(device=device)
            if dims.numel() == 6 and dims.dtype == torch.bool:
                base = dims.to(dtype=dtype)
            else:
                base = torch.zeros(6, device=device, dtype=dtype)
                for dim in dims.flatten().tolist():
                    if 0 <= int(dim) < 6:
                        base[int(dim)] = 1.0
        else:
            base = torch.zeros(6, device=device, dtype=dtype)
            for dim in self.active_task_dims:
                if 0 <= int(dim) < 6:
                    base[int(dim)] = 1.0
        return base.view(1, 6).repeat(count, 1)

    def _mode_mask(
        self,
        mode_groups: list[tuple[str, ...]] | tuple[tuple[str, ...], ...],
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.action_cone is not None and hasattr(self.action_cone, "mode_dim_mask"):
            return self.action_cone.mode_dim_mask(mode_groups, count, device, dtype)
        mask = torch.zeros(count, 6, device=device, dtype=dtype)
        for env_i, modes in enumerate(list(mode_groups)[:count]):
            mode_set = set(modes)
            if "planar" in mode_set:
                mask[env_i, 0] = 1.0
                mask[env_i, 1] = 1.0
            if "global_z" in mode_set:
                mask[env_i, 2] = 1.0
            if "local_rp" in mode_set:
                mask[env_i, 3] = 1.0
                mask[env_i, 4] = 1.0
            if "yaw" in mode_set:
                mask[env_i, 5] = 1.0
        return mask

    def _apply_dz_rule(self, projected: torch.Tensor) -> torch.Tensor:
        out = projected.clone()
        if self.upward_dz_rule == "allow":
            return out
        if self.upward_dz_rule == "nonpositive":
            out[:, 2] = torch.minimum(out[:, 2], torch.zeros_like(out[:, 2]))
            return out
        if isinstance(self.upward_dz_rule, (int, float)):
            out[:, 2] = torch.minimum(out[:, 2], torch.full_like(out[:, 2], float(self.upward_dz_rule)))
            return out
        raise ValueError(f"unsupported upward_dz_rule: {self.upward_dz_rule}")
