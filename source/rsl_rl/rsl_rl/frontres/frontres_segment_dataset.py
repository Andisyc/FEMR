from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch


@dataclass(frozen=True)
class FrontRESSegmentState:
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    key_body_pos: torch.Tensor | None = None
    key_body_quat: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.root_pos.device

    @property
    def batch_size(self) -> int:
        return int(self.root_pos.shape[0])

    def validate(self) -> None:
        batch = self.batch_size
        _require_shape("root_pos", self.root_pos, (batch, 3))
        _require_shape("root_quat", self.root_quat, (batch, 4))
        _require_shape("root_lin_vel", self.root_lin_vel, (batch, 3))
        _require_shape("root_ang_vel", self.root_ang_vel, (batch, 3))
        if self.dof_pos.ndim != 2:
            raise ValueError(f"dof_pos must be rank-2, got shape {tuple(self.dof_pos.shape)}")
        if self.dof_vel.shape != self.dof_pos.shape:
            raise ValueError(
                f"dof_vel shape {tuple(self.dof_vel.shape)} must match dof_pos {tuple(self.dof_pos.shape)}"
            )
        if self.key_body_pos is not None and (self.key_body_pos.ndim != 3 or self.key_body_pos.shape[0] != batch):
            raise ValueError("key_body_pos must have shape [B, K, 3]")
        if self.key_body_quat is not None and (self.key_body_quat.ndim != 3 or self.key_body_quat.shape[0] != batch):
            raise ValueError("key_body_quat must have shape [B, K, 4]")


@dataclass(frozen=True)
class FrontRESSegmentSpec:
    segment_id: int
    motion_id: int | str
    start_frame: int | None = None
    start_time: float | None = None
    phase: float = 0.0
    horizon_k: int = 1
    perturbation_family: str = "none"
    perturbation_strength: float = 0.0
    reset_mode_hint: str = "auto"
    valid_for_training: bool = True

    def __post_init__(self) -> None:
        if self.start_frame is None and self.start_time is None:
            raise ValueError("FrontRESSegmentSpec requires start_frame or start_time")
        if not 0.0 <= float(self.phase) <= 1.0:
            raise ValueError(f"phase must be in [0, 1], got {self.phase}")
        if int(self.horizon_k) <= 0:
            raise ValueError(f"horizon_k must be positive, got {self.horizon_k}")
        if self.reset_mode_hint not in {"direct", "preroll", "auto"}:
            raise ValueError(f"unsupported reset_mode_hint: {self.reset_mode_hint}")


@dataclass(frozen=True)
class FrontRESSegmentBatch:
    segment_ids: torch.Tensor
    specs: tuple[FrontRESSegmentSpec, ...]
    clean_state: FrontRESSegmentState
    reference_window: torch.Tensor | Any
    phase: torch.Tensor
    horizon_k: torch.Tensor
    perturbation_family: tuple[str, ...]
    perturbation_strength: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.segment_ids.numel())

    def validate(self) -> None:
        batch = self.batch_size
        if len(self.specs) != batch:
            raise ValueError("spec count must match segment_ids")
        self.clean_state.validate()
        if self.clean_state.batch_size != batch:
            raise ValueError("clean_state batch size must match segment_ids")
        _require_shape("phase", self.phase, (batch,))
        _require_shape("horizon_k", self.horizon_k, (batch,))
        _require_shape("perturbation_strength", self.perturbation_strength, (batch,))
        if len(self.perturbation_family) != batch:
            raise ValueError("perturbation_family count must match segment_ids")
        if isinstance(self.reference_window, torch.Tensor) and self.reference_window.shape[0] != batch:
            raise ValueError("reference_window first dimension must match batch")


@dataclass(frozen=True)
class FrontRESSegmentValidation:
    valid_mask: torch.Tensor
    reasons: tuple[str, ...]

    @property
    def all_valid(self) -> bool:
        return bool(torch.all(self.valid_mask).item())


class FrontRESSegmentDataset:
    """Pure segment dataset for Segment Replay HRL.

    The dataset stores dynamic reset payloads.  It intentionally does not know
    about IsaacLab, GMT, PPO, or runner internals.
    """

    def __init__(
        self,
        motion_source: Sequence[dict[str, Any]],
        dt: float,
        default_horizon_k: int,
        device: str | torch.device = "cpu",
        cache_policy: str = "eager",
        motion_normalizer: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
        reference_builder: Callable[[dict[str, torch.Tensor], int, int], torch.Tensor] | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if default_horizon_k <= 0:
            raise ValueError(f"default_horizon_k must be positive, got {default_horizon_k}")
        if cache_policy not in {"eager", "manual"}:
            raise ValueError(f"unsupported cache_policy: {cache_policy}")

        self.dt = float(dt)
        self.default_horizon_k = int(default_horizon_k)
        self.device = torch.device(device)
        self.cache_policy = cache_policy
        self.motion_normalizer = motion_normalizer
        self.reference_builder = reference_builder
        self._motions = [self._prepare_motion(motion) for motion in motion_source]
        self._specs = self._build_specs()
        self._spec_by_id = {spec.segment_id: spec for spec in self._specs}
        self._invalid_reasons: dict[int, str] = {}
        self._noisy_baseline: dict[int, Any] = {}
        if cache_policy == "eager":
            self.build_clean_cache()

    def num_segments(self) -> int:
        return len(self._specs)

    def sample_global(self, batch_size: int, generator: torch.Generator | None = None) -> FrontRESSegmentBatch:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        valid_ids = [spec.segment_id for spec in self._specs if spec.valid_for_training]
        if not valid_ids:
            raise RuntimeError("no valid FrontRES segments are available")
        pool = torch.tensor(valid_ids, dtype=torch.long, device=self.device)
        idx = torch.randint(0, pool.numel(), (batch_size,), generator=generator, device=self.device)
        return self.get_segments(pool[idx])

    def get_segments(self, segment_ids: Iterable[int] | torch.Tensor) -> FrontRESSegmentBatch:
        ids = self._ids_tensor(segment_ids)
        specs = tuple(self._spec_by_id[int(segment_id)] for segment_id in ids.tolist())
        clean_state = self._state_for_specs(specs)
        phase = torch.tensor([float(spec.phase) for spec in specs], dtype=torch.float32, device=self.device)
        horizon_k = torch.tensor([int(spec.horizon_k) for spec in specs], dtype=torch.long, device=self.device)
        perturbation_strength = torch.tensor(
            [float(spec.perturbation_strength) for spec in specs], dtype=torch.float32, device=self.device
        )
        reference_window = self._reference_for_specs(specs)
        batch = FrontRESSegmentBatch(
            segment_ids=ids,
            specs=specs,
            clean_state=clean_state,
            reference_window=reference_window,
            phase=phase,
            horizon_k=horizon_k,
            perturbation_family=tuple(spec.perturbation_family for spec in specs),
            perturbation_strength=perturbation_strength,
        )
        batch.validate()
        return batch

    def build_clean_cache(self) -> None:
        for motion in self._motions:
            for name in ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "dof_pos", "dof_vel"):
                if name not in motion:
                    raise ValueError(f"motion is missing required dynamic state field: {name}")

    def validate_batch(self, batch: FrontRESSegmentBatch) -> FrontRESSegmentValidation:
        reasons: list[str] = []
        valid = torch.ones(batch.batch_size, dtype=torch.bool, device=batch.segment_ids.device)
        for i, spec in enumerate(batch.specs):
            reason = self._invalid_reasons.get(spec.segment_id, "")
            if reason or not spec.valid_for_training:
                valid[i] = False
                reasons.append(reason or "marked invalid")
            else:
                reasons.append("")
        try:
            batch.validate()
        except ValueError as exc:
            valid[:] = False
            reasons = [str(exc)] * batch.batch_size
        return FrontRESSegmentValidation(valid_mask=valid, reasons=tuple(reasons))

    def write_noisy_baseline(self, segment_ids: Iterable[int] | torch.Tensor, evidence: Any) -> None:
        for segment_id in self._ids_tensor(segment_ids).tolist():
            self._noisy_baseline[int(segment_id)] = evidence

    def read_noisy_baseline(self, segment_ids: Iterable[int] | torch.Tensor) -> dict[int, Any]:
        result: dict[int, Any] = {}
        for segment_id in self._ids_tensor(segment_ids).tolist():
            if int(segment_id) in self._noisy_baseline:
                result[int(segment_id)] = self._noisy_baseline[int(segment_id)]
        return result

    def update_validity(self, segment_ids: Iterable[int] | torch.Tensor, flags: Iterable[bool], reason: str = "invalid") -> None:
        for segment_id, flag in zip(self._ids_tensor(segment_ids).tolist(), flags):
            spec = self._spec_by_id[int(segment_id)]
            updated = FrontRESSegmentSpec(
                segment_id=spec.segment_id,
                motion_id=spec.motion_id,
                start_frame=spec.start_frame,
                start_time=spec.start_time,
                phase=spec.phase,
                horizon_k=spec.horizon_k,
                perturbation_family=spec.perturbation_family,
                perturbation_strength=spec.perturbation_strength,
                reset_mode_hint=spec.reset_mode_hint,
                valid_for_training=bool(flag),
            )
            self._spec_by_id[int(segment_id)] = updated
            self._specs[int(segment_id)] = updated
            if flag:
                self._invalid_reasons.pop(int(segment_id), None)
            else:
                self._invalid_reasons[int(segment_id)] = reason

    def state_dict(self) -> dict[str, Any]:
        return {
            "dt": self.dt,
            "default_horizon_k": self.default_horizon_k,
            "invalid_reasons": dict(self._invalid_reasons),
            "noisy_baseline": dict(self._noisy_baseline),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._invalid_reasons = {int(k): str(v) for k, v in state.get("invalid_reasons", {}).items()}
        self._noisy_baseline = {int(k): v for k, v in state.get("noisy_baseline", {}).items()}
        restored_specs: list[FrontRESSegmentSpec] = []
        for spec in self._specs:
            restored_specs.append(
                FrontRESSegmentSpec(
                    segment_id=spec.segment_id,
                    motion_id=spec.motion_id,
                    start_frame=spec.start_frame,
                    start_time=spec.start_time,
                    phase=spec.phase,
                    horizon_k=spec.horizon_k,
                    perturbation_family=spec.perturbation_family,
                    perturbation_strength=spec.perturbation_strength,
                    reset_mode_hint=spec.reset_mode_hint,
                    valid_for_training=spec.segment_id not in self._invalid_reasons,
                )
            )
        self._specs = restored_specs
        self._spec_by_id = {spec.segment_id: spec for spec in self._specs}

    def _prepare_motion(self, motion: dict[str, Any]) -> dict[str, torch.Tensor | Any]:
        tensors: dict[str, torch.Tensor | Any] = {}
        for key, value in motion.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value.to(self.device)
            else:
                tensors[key] = value
        if self.motion_normalizer is not None:
            tensor_only = {k: v for k, v in tensors.items() if isinstance(v, torch.Tensor)}
            tensors.update(self.motion_normalizer(tensor_only))
        return tensors

    def _build_specs(self) -> list[FrontRESSegmentSpec]:
        specs: list[FrontRESSegmentSpec] = []
        next_id = 0
        for motion_index, motion in enumerate(self._motions):
            num_frames = int(motion["root_pos"].shape[0])
            if num_frames < self.default_horizon_k + 1:
                continue
            max_start = num_frames - self.default_horizon_k
            for start_frame in range(max_start):
                denom = max(1, num_frames - 1)
                specs.append(
                    FrontRESSegmentSpec(
                        segment_id=next_id,
                        motion_id=motion.get("motion_id", motion_index),
                        start_frame=start_frame,
                        start_time=float(start_frame) * self.dt,
                        phase=float(start_frame) / float(denom),
                        horizon_k=self.default_horizon_k,
                        perturbation_family=str(motion.get("perturbation_family", "none")),
                        perturbation_strength=float(motion.get("perturbation_strength", 0.0)),
                        reset_mode_hint=str(motion.get("reset_mode_hint", "auto")),
                        valid_for_training=True,
                    )
                )
                next_id += 1
        if not specs:
            raise ValueError("motion_source does not contain any segment long enough for default_horizon_k")
        return specs

    def _state_for_specs(self, specs: Sequence[FrontRESSegmentSpec]) -> FrontRESSegmentState:
        motion_lookup = {motion.get("motion_id", i): motion for i, motion in enumerate(self._motions)}
        rows: dict[str, list[torch.Tensor]] = {
            "root_pos": [],
            "root_quat": [],
            "root_lin_vel": [],
            "root_ang_vel": [],
            "dof_pos": [],
            "dof_vel": [],
        }
        key_body_pos: list[torch.Tensor] = []
        key_body_quat: list[torch.Tensor] = []
        for spec in specs:
            motion = motion_lookup[spec.motion_id]
            frame = int(spec.start_frame if spec.start_frame is not None else round(float(spec.start_time) / self.dt))
            for name in rows:
                rows[name].append(motion[name][frame])
            if "key_body_pos" in motion:
                key_body_pos.append(motion["key_body_pos"][frame])
            if "key_body_quat" in motion:
                key_body_quat.append(motion["key_body_quat"][frame])
        return FrontRESSegmentState(
            root_pos=torch.stack(rows["root_pos"], dim=0),
            root_quat=torch.stack(rows["root_quat"], dim=0),
            root_lin_vel=torch.stack(rows["root_lin_vel"], dim=0),
            root_ang_vel=torch.stack(rows["root_ang_vel"], dim=0),
            dof_pos=torch.stack(rows["dof_pos"], dim=0),
            dof_vel=torch.stack(rows["dof_vel"], dim=0),
            key_body_pos=torch.stack(key_body_pos, dim=0) if key_body_pos else None,
            key_body_quat=torch.stack(key_body_quat, dim=0) if key_body_quat else None,
        )

    def _reference_for_specs(self, specs: Sequence[FrontRESSegmentSpec]) -> torch.Tensor:
        motion_lookup = {motion.get("motion_id", i): motion for i, motion in enumerate(self._motions)}
        windows: list[torch.Tensor] = []
        for spec in specs:
            motion = motion_lookup[spec.motion_id]
            frame = int(spec.start_frame if spec.start_frame is not None else round(float(spec.start_time) / self.dt))
            if self.reference_builder is not None:
                windows.append(self.reference_builder(motion, frame, int(spec.horizon_k)))
            elif "reference" in motion:
                windows.append(motion["reference"][frame : frame + int(spec.horizon_k) + 1])
            else:
                windows.append(motion["root_pos"][frame : frame + int(spec.horizon_k) + 1])
        return torch.stack(windows, dim=0)

    def _ids_tensor(self, segment_ids: Iterable[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(segment_ids, torch.Tensor):
            ids = segment_ids.to(device=self.device, dtype=torch.long).flatten()
        else:
            ids = torch.tensor(list(segment_ids), dtype=torch.long, device=self.device)
        for segment_id in ids.tolist():
            if int(segment_id) not in self._spec_by_id:
                raise KeyError(f"unknown segment_id: {segment_id}")
        return ids


def _require_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
