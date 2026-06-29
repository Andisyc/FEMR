from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class FrontRESSegmentIndex:
    segment_id: int
    motion_rel_path: str
    motion_num_frames: int
    fps: float
    start_frame: int
    horizon_k: int

    def validate(self) -> None:
        if int(self.segment_id) < 0:
            raise ValueError(f"segment_id must be non-negative, got {self.segment_id}")
        if not self.motion_rel_path:
            raise ValueError("motion_rel_path must be non-empty")
        if int(self.motion_num_frames) <= 0:
            raise ValueError(f"motion_num_frames must be positive, got {self.motion_num_frames}")
        if float(self.fps) <= 0.0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if int(self.start_frame) < 0:
            raise ValueError(f"start_frame must be non-negative, got {self.start_frame}")
        if int(self.horizon_k) <= 0:
            raise ValueError(f"horizon_k must be positive, got {self.horizon_k}")
        if int(self.start_frame) + int(self.horizon_k) >= int(self.motion_num_frames):
            raise ValueError(
                "segment window exceeds motion length: "
                f"start_frame={self.start_frame}, horizon_k={self.horizon_k}, "
                f"motion_num_frames={self.motion_num_frames}"
            )


@dataclass(frozen=True)
class FrontRESRobotRolloutState:
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    contact_state: torch.Tensor | None = None
    action_history: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.root_pos.shape[0])

    @property
    def num_dofs(self) -> int:
        return int(self.joint_pos.shape[1])

    @property
    def num_bodies(self) -> int:
        return int(self.body_pos_w.shape[1])

    def validate(self, *, name: str = "rollout_state") -> None:
        batch = self.batch_size
        _require_shape(f"{name}.root_pos", self.root_pos, (batch, 3))
        _require_shape(f"{name}.root_quat", self.root_quat, (batch, 4))
        _require_shape(f"{name}.root_lin_vel", self.root_lin_vel, (batch, 3))
        _require_shape(f"{name}.root_ang_vel", self.root_ang_vel, (batch, 3))
        _require_rank(f"{name}.joint_pos", self.joint_pos, 2)
        _require_shape(f"{name}.joint_vel", self.joint_vel, tuple(self.joint_pos.shape))
        _require_shape(f"{name}.body_pos_w", self.body_pos_w, (batch, self.num_bodies, 3))
        _require_shape(f"{name}.body_quat_w", self.body_quat_w, (batch, self.num_bodies, 4))
        _require_shape(f"{name}.body_lin_vel_w", self.body_lin_vel_w, (batch, self.num_bodies, 3))
        _require_shape(f"{name}.body_ang_vel_w", self.body_ang_vel_w, (batch, self.num_bodies, 3))
        if self.contact_state is not None and int(self.contact_state.shape[0]) != batch:
            raise ValueError(f"{name}.contact_state first dim must be batch={batch}")
        if self.action_history is not None and int(self.action_history.shape[0]) != batch:
            raise ValueError(f"{name}.action_history first dim must be batch={batch}")
        for field_name, value in self.tensor_items():
            _require_floating_tensor(f"{name}.{field_name}", value)
            _require_finite(f"{name}.{field_name}", value)
            if value.requires_grad:
                raise ValueError(f"{name}.{field_name} must be detached cache data")

    def tensor_items(self) -> tuple[tuple[str, torch.Tensor], ...]:
        items: list[tuple[str, torch.Tensor]] = [
            ("root_pos", self.root_pos),
            ("root_quat", self.root_quat),
            ("root_lin_vel", self.root_lin_vel),
            ("root_ang_vel", self.root_ang_vel),
            ("joint_pos", self.joint_pos),
            ("joint_vel", self.joint_vel),
            ("body_pos_w", self.body_pos_w),
            ("body_quat_w", self.body_quat_w),
            ("body_lin_vel_w", self.body_lin_vel_w),
            ("body_ang_vel_w", self.body_ang_vel_w),
        ]
        if self.contact_state is not None:
            items.append(("contact_state", self.contact_state))
        if self.action_history is not None:
            items.append(("action_history", self.action_history))
        return tuple(items)

    def probe(self, *, prefix: str = "state") -> dict[str, Any]:
        self.validate(name=prefix)
        return {
            f"{prefix}.batch_size": self.batch_size,
            f"{prefix}.num_dofs": self.num_dofs,
            f"{prefix}.num_bodies": self.num_bodies,
            f"{prefix}.root_pos_shape": tuple(self.root_pos.shape),
            f"{prefix}.joint_pos_shape": tuple(self.joint_pos.shape),
            f"{prefix}.body_pos_shape": tuple(self.body_pos_w.shape),
            f"{prefix}.finite": all(bool(torch.isfinite(value).all().item()) for _, value in self.tensor_items()),
            f"{prefix}.requires_grad": any(bool(value.requires_grad) for _, value in self.tensor_items()),
        }


@dataclass(frozen=True)
class FrontRESPerturbationDescriptor:
    perturbation_id: int
    segment_id: int
    strength: float
    seed: int
    family: str
    start_step: int
    duration: int
    target: str
    frame: str
    params: Mapping[str, Any]

    def validate(self) -> None:
        if int(self.perturbation_id) < 0:
            raise ValueError(f"perturbation_id must be non-negative, got {self.perturbation_id}")
        if int(self.segment_id) < 0:
            raise ValueError(f"segment_id must be non-negative, got {self.segment_id}")
        if float(self.strength) < 0.0:
            raise ValueError(f"strength must be non-negative, got {self.strength}")
        if int(self.start_step) < 0:
            raise ValueError(f"start_step must be non-negative, got {self.start_step}")
        if int(self.duration) <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if not self.family:
            raise ValueError("family must be non-empty")
        if not self.target:
            raise ValueError("target must be non-empty")
        if self.frame not in {"world", "local", "joint"}:
            raise ValueError(f"frame must be world, local, or joint, got {self.frame}")
        if "level_index" in self.params and int(self.params["level_index"]) < 0:
            raise ValueError(f"level_index must be non-negative, got {self.params['level_index']}")
        if "level_name" in self.params and not str(self.params["level_name"]):
            raise ValueError("level_name must be non-empty when provided")
        if "curriculum_mode" in self.params and str(self.params["curriculum_mode"]) not in {
            "discrete_bank",
            "hrl_curriculum_bank",
        }:
            raise ValueError(f"unsupported curriculum_mode: {self.params['curriculum_mode']}")
        if "family_group" in self.params and not tuple(self.params["family_group"]):
            raise ValueError("family_group must be non-empty when provided")
        if "mix_class" in self.params and str(self.params["mix_class"]) not in {"easy", "frontier", "hard", "fixed"}:
            raise ValueError(f"unsupported mix_class: {self.params['mix_class']}")
        if "mix_class_index" in self.params and int(self.params["mix_class_index"]) not in {-1, 0, 1, 2}:
            raise ValueError(f"unsupported mix_class_index: {self.params['mix_class_index']}")
        if "frontier_scale" in self.params and float(self.params["frontier_scale"]) < 0.0:
            raise ValueError(f"frontier_scale must be non-negative, got {self.params['frontier_scale']}")
        if "dr_factor" in self.params and float(self.params["dr_factor"]) < 0.0:
            raise ValueError(f"dr_factor must be non-negative, got {self.params['dr_factor']}")
        if "actual_dr_scale" in self.params and float(self.params["actual_dr_scale"]) < 0.0:
            raise ValueError(f"actual_dr_scale must be non-negative, got {self.params['actual_dr_scale']}")
        if "perturbation_role" in self.params and str(self.params["perturbation_role"]) not in {
            "train",
            "boundary_diagnostic",
        }:
            raise ValueError(f"unsupported perturbation_role: {self.params['perturbation_role']}")
        if "burst_min_steps" in self.params and int(self.params["burst_min_steps"]) <= 0:
            raise ValueError(f"burst_min_steps must be positive, got {self.params['burst_min_steps']}")
        if "burst_max_steps" in self.params and int(self.params["burst_max_steps"]) <= 0:
            raise ValueError(f"burst_max_steps must be positive, got {self.params['burst_max_steps']}")
        if (
            "burst_min_steps" in self.params
            and "burst_max_steps" in self.params
            and int(self.params["burst_max_steps"]) < int(self.params["burst_min_steps"])
        ):
            raise ValueError(
                "burst_max_steps must be >= burst_min_steps, "
                f"got {self.params['burst_max_steps']} < {self.params['burst_min_steps']}"
            )


@dataclass(frozen=True)
class FrontRESNoisyVariant:
    segment: FrontRESSegmentIndex
    descriptor: FrontRESPerturbationDescriptor
    noisy_state: FrontRESRobotRolloutState
    noisy_baseline_score: torch.Tensor
    noisy_fall: torch.Tensor
    noisy_rollout_len: torch.Tensor

    @property
    def segment_id(self) -> int:
        return int(self.segment.segment_id)

    @property
    def perturbation_id(self) -> int:
        return int(self.descriptor.perturbation_id)

    def validate(self) -> None:
        self.segment.validate()
        self.descriptor.validate()
        if int(self.descriptor.segment_id) != int(self.segment.segment_id):
            raise ValueError(
                f"descriptor segment_id={self.descriptor.segment_id} does not match "
                f"segment segment_id={self.segment.segment_id}"
            )
        self.noisy_state.validate(name="noisy_state")
        batch = self.noisy_state.batch_size
        _require_shape("noisy_baseline_score", self.noisy_baseline_score, (batch,))
        _require_shape("noisy_fall", self.noisy_fall, (batch,))
        _require_shape("noisy_rollout_len", self.noisy_rollout_len, (batch,))
        _require_floating_tensor("noisy_baseline_score", self.noisy_baseline_score)
        _require_finite("noisy_baseline_score", self.noisy_baseline_score)
        if self.noisy_baseline_score.requires_grad:
            raise ValueError("noisy_baseline_score must be detached cache data")
        if self.noisy_rollout_len.requires_grad:
            raise ValueError("noisy_rollout_len must be detached cache data")

    def probe(self) -> dict[str, Any]:
        self.validate()
        result = {
            "segment_id": self.segment_id,
            "perturbation_id": self.perturbation_id,
            "descriptor.segment_id": int(self.descriptor.segment_id),
            "descriptor.strength": float(self.descriptor.strength),
            "baseline_score_shape": tuple(self.noisy_baseline_score.shape),
            "fall_shape": tuple(self.noisy_fall.shape),
            "rollout_len_shape": tuple(self.noisy_rollout_len.shape),
        }
        result.update(self.noisy_state.probe(prefix="noisy_state"))
        return result


def _require_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must be rank-{rank}, got shape {tuple(tensor.shape)}")


def _require_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def _require_floating_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be a floating tensor, got dtype={tensor.dtype}")


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains non-finite values")
