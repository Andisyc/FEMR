from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Sequence

import torch


_LOG_SEPARATOR = "-" * 80


def _log_block(*lines: str) -> str:
    return "\n".join(("", _LOG_SEPARATOR, "", *lines))


def _load_same_dir(module_name: str):
    path = Path(__file__).with_name(f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_stage1_metadata_raw(cache_dir: str | Path) -> dict[str, Any]:
    with (Path(cache_dir) / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


try:
    from rsl_rl.frontres.frontres_segment_cache_io import (
        FrontRESSegmentShardLRU,
        read_cache_metadata,
        read_noisy_variant_manifest_records,
        read_noisy_variant_record,
        read_noisy_variant_shard,
    )
except ModuleNotFoundError:
    _cache_io = _load_same_dir("frontres_segment_cache_io")
    FrontRESSegmentShardLRU = _cache_io.FrontRESSegmentShardLRU
    read_cache_metadata = _cache_io.read_cache_metadata
    read_noisy_variant_manifest_records = _cache_io.read_noisy_variant_manifest_records
    read_noisy_variant_record = _cache_io.read_noisy_variant_record
    read_noisy_variant_shard = _cache_io.read_noisy_variant_shard

try:
    from rsl_rl.frontres.frontres_segment_cache_indexer import read_amass_segment_index
except ModuleNotFoundError:
    _cache_indexer = _load_same_dir("frontres_segment_cache_indexer")
    read_amass_segment_index = _cache_indexer.read_amass_segment_index


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
    perturbation_role: str = "train"
    reset_mode_hint: str = "auto"
    valid_for_training: bool = True

    def __post_init__(self) -> None:
        if self.start_frame is None and self.start_time is None:
            raise ValueError("FrontRESSegmentSpec requires start_frame or start_time")
        if not 0.0 <= float(self.phase) <= 1.0:
            raise ValueError(f"phase must be in [0, 1], got {self.phase}")
        if int(self.horizon_k) <= 0:
            raise ValueError(f"horizon_k must be positive, got {self.horizon_k}")
        if self.perturbation_role not in {"train", "boundary_diagnostic"}:
            raise ValueError(f"unsupported perturbation_role: {self.perturbation_role}")
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
    perturbation_role: tuple[str, ...] = ()

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
        if self.perturbation_role and len(self.perturbation_role) != batch:
            raise ValueError("perturbation_role count must match segment_ids")
        if isinstance(self.reference_window, torch.Tensor) and self.reference_window.shape[0] != batch:
            raise ValueError("reference_window first dimension must match batch")


@dataclass(frozen=True)
class FrontRESSegmentValidation:
    valid_mask: torch.Tensor
    reasons: tuple[str, ...]

    @property
    def all_valid(self) -> bool:
        return bool(torch.all(self.valid_mask).item())


@dataclass(frozen=True)
class FrontRESStage1CacheDatasetLoadSummary:
    cache_dir: str
    perturbation_curriculum_mode: str
    metadata_noisy_count: int
    loaded_motion_count: int
    skipped_boundary_diagnostic_count: int
    included_boundary_diagnostic_count: int
    role_counts: dict[str, int]

    def probe(self) -> dict[str, Any]:
        return {
            "cache_dir": self.cache_dir,
            "perturbation_curriculum_mode": self.perturbation_curriculum_mode,
            "metadata_noisy_count": self.metadata_noisy_count,
            "loaded_motion_count": self.loaded_motion_count,
            "skipped_boundary_diagnostic_count": self.skipped_boundary_diagnostic_count,
            "included_boundary_diagnostic_count": self.included_boundary_diagnostic_count,
            "role_counts": dict(self.role_counts),
        }


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
        self._cache_metadata: dict[str, Any] | None = None
        if cache_policy == "eager":
            self.build_clean_cache()

    def num_segments(self) -> int:
        return len(self._specs)

    def sample_global(self, batch_size: int, generator: torch.Generator | None = None) -> FrontRESSegmentBatch:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        valid_ids = [
            spec.segment_id
            for spec in self._specs
            if spec.valid_for_training and spec.segment_id not in self._invalid_reasons
        ]
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
            perturbation_role=tuple(spec.perturbation_role for spec in specs),
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
                perturbation_role=spec.perturbation_role,
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
            "cache_metadata": self.cache_metadata(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._invalid_reasons = {int(k): str(v) for k, v in state.get("invalid_reasons", {}).items()}
        self._noisy_baseline = {int(k): v for k, v in state.get("noisy_baseline", {}).items()}
        if "cache_metadata" in state:
            self.load_cache_metadata(state["cache_metadata"])
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
                    perturbation_role=spec.perturbation_role,
                    reset_mode_hint=spec.reset_mode_hint,
                    valid_for_training=spec.segment_id not in self._invalid_reasons,
                )
            )
        self._specs = restored_specs
        self._spec_by_id = {spec.segment_id: spec for spec in self._specs}

    def cache_metadata(self) -> dict[str, Any] | None:
        return None if self._cache_metadata is None else dict(self._cache_metadata)

    def load_cache_metadata(self, metadata: dict[str, Any] | None) -> None:
        self._cache_metadata = None if metadata is None else dict(metadata)

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
                        perturbation_role=str(motion.get("perturbation_role", "train")),
                        reset_mode_hint=str(motion.get("reset_mode_hint", "auto")),
                        valid_for_training=bool(
                            motion.get(
                                "valid_for_training",
                                str(motion.get("perturbation_role", "train")) == "train",
                            )
                        ),
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
            elif "dof_pos" in motion and "dof_vel" in motion:
                joint_pos = motion["dof_pos"][frame : frame + int(spec.horizon_k) + 1]
                joint_vel = motion["dof_vel"][frame : frame + int(spec.horizon_k) + 1]
                windows.append(torch.cat([joint_pos, joint_vel], dim=-1))
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


@dataclass(frozen=True)
class FrontRESStage1CacheRecord:
    segment_id: int
    manifest_record: dict[str, Any]
    role: str
    valid_for_training: bool
    horizon_k: int
    fps: float
    perturbation_family: str
    perturbation_strength: float
    source_segment_id: int
    source_perturbation_id: int


class FrontRESStage1LazyCacheDataset:
    def __init__(
        self,
        *,
        cache_dir: str | Path,
        records: Sequence[FrontRESStage1CacheRecord],
        summary: FrontRESStage1CacheDatasetLoadSummary,
        device: str | torch.device = "cpu",
        shard_cache_size: int = 8,
    ) -> None:
        if not records:
            raise ValueError(f"Stage 1 cache produced no trainable segment motions: {cache_dir}")
        self.cache_dir = Path(cache_dir)
        self.device = torch.device(device)
        self._records = tuple(records)
        self._specs = tuple(self._spec_from_record(record) for record in self._records)
        self._spec_by_id = {spec.segment_id: spec for spec in self._specs}
        self._record_by_id = {record.segment_id: record for record in self._records}
        self._invalid_reasons: dict[int, str] = {}
        self._noisy_baseline: dict[int, Any] = {}
        self._cache_metadata = summary.probe()
        self._shard_cache = FrontRESSegmentShardLRU(max_shards=shard_cache_size)

    def num_segments(self) -> int:
        return len(self._records)

    def sample_global(self, batch_size: int, generator: torch.Generator | None = None) -> FrontRESSegmentBatch:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        valid_ids = [
            spec.segment_id
            for spec in self._specs
            if spec.valid_for_training and spec.segment_id not in self._invalid_reasons
        ]
        if not valid_ids:
            raise RuntimeError("no valid FrontRES segments are available")
        pool = torch.tensor(valid_ids, dtype=torch.long, device=self.device)
        idx = torch.randint(0, pool.numel(), (batch_size,), generator=generator, device=self.device)
        return self.get_segments(pool[idx])

    def get_segments(self, segment_ids: Iterable[int] | torch.Tensor) -> FrontRESSegmentBatch:
        ids = self._ids_tensor(segment_ids)
        specs = tuple(self._spec_by_id[int(segment_id)] for segment_id in ids.tolist())
        variants = [
            read_noisy_variant_record(
                self.cache_dir,
                self._record_by_id[int(segment_id)].manifest_record,
                shard_cache=self._shard_cache,
            )
            for segment_id in ids.tolist()
        ]
        motions = [_motion_from_noisy_variant(variant, role=spec.perturbation_role) for variant, spec in zip(variants, specs)]
        clean_state = self._state_from_motions(motions)
        reference_window = self._reference_from_motions(motions)
        phase = torch.tensor([float(spec.phase) for spec in specs], dtype=torch.float32, device=self.device)
        horizon_k = torch.tensor([int(spec.horizon_k) for spec in specs], dtype=torch.long, device=self.device)
        perturbation_strength = torch.tensor(
            [float(spec.perturbation_strength) for spec in specs], dtype=torch.float32, device=self.device
        )
        batch = FrontRESSegmentBatch(
            segment_ids=ids,
            specs=specs,
            clean_state=clean_state,
            reference_window=reference_window,
            phase=phase,
            horizon_k=horizon_k,
            perturbation_family=tuple(spec.perturbation_family for spec in specs),
            perturbation_strength=perturbation_strength,
            perturbation_role=tuple(spec.perturbation_role for spec in specs),
        )
        batch.validate()
        return batch

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
            if flag:
                self._invalid_reasons.pop(int(segment_id), None)
            else:
                self._invalid_reasons[int(segment_id)] = reason

    def state_dict(self) -> dict[str, Any]:
        return {
            "invalid_reasons": dict(self._invalid_reasons),
            "noisy_baseline": dict(self._noisy_baseline),
            "cache_metadata": self.cache_metadata(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._invalid_reasons = {int(k): str(v) for k, v in state.get("invalid_reasons", {}).items()}
        self._noisy_baseline = {int(k): v for k, v in state.get("noisy_baseline", {}).items()}
        if "cache_metadata" in state:
            self.load_cache_metadata(state["cache_metadata"])

    def cache_metadata(self) -> dict[str, Any] | None:
        metadata = None if self._cache_metadata is None else dict(self._cache_metadata)
        if metadata is not None:
            metadata["shard_cache"] = self._shard_cache.probe()
        return metadata

    def load_cache_metadata(self, metadata: dict[str, Any] | None) -> None:
        self._cache_metadata = None if metadata is None else dict(metadata)

    def _ids_tensor(self, segment_ids: Iterable[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(segment_ids, torch.Tensor):
            ids = segment_ids.to(device=self.device, dtype=torch.long).flatten()
        else:
            ids = torch.tensor(list(segment_ids), dtype=torch.long, device=self.device)
        for segment_id in ids.tolist():
            if int(segment_id) not in self._record_by_id:
                raise KeyError(f"unknown segment_id: {segment_id}")
        return ids

    def _spec_from_record(self, record: FrontRESStage1CacheRecord) -> FrontRESSegmentSpec:
        return FrontRESSegmentSpec(
            segment_id=int(record.segment_id),
            motion_id=f"{int(record.source_segment_id)}:{int(record.source_perturbation_id)}",
            start_frame=0,
            start_time=0.0,
            phase=0.0,
            horizon_k=int(record.horizon_k),
            perturbation_family=str(record.perturbation_family),
            perturbation_strength=float(record.perturbation_strength),
            perturbation_role=str(record.role),
            reset_mode_hint="direct",
            valid_for_training=bool(record.valid_for_training),
        )

    def _state_from_motions(self, motions: Sequence[dict[str, Any]]) -> FrontRESSegmentState:
        rows = {name: [] for name in ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "dof_pos", "dof_vel")}
        key_body_pos: list[torch.Tensor] = []
        key_body_quat: list[torch.Tensor] = []
        for motion in motions:
            for name in rows:
                rows[name].append(motion[name][0].to(self.device))
            if "key_body_pos" in motion:
                key_body_pos.append(motion["key_body_pos"][0].to(self.device))
            if "key_body_quat" in motion:
                key_body_quat.append(motion["key_body_quat"][0].to(self.device))
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

    def _reference_from_motions(self, motions: Sequence[dict[str, Any]]) -> torch.Tensor:
        return torch.stack([motion["reference"].to(self.device) for motion in motions], dim=0)


class FrontRESStage1IndexDataset(FrontRESSegmentDataset):
    def __init__(
        self,
        *,
        cache_dir: str | Path,
        metadata: dict[str, Any],
        device: str | torch.device = "cpu",
        dof_dim: int = 29,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.device = torch.device(device)
        self.dof_dim = int(dof_dim)
        self._segments = tuple(read_amass_segment_index(self.cache_dir / "segment_index.jsonl"))
        self._specs = [self._spec_from_segment(segment) for segment in self._segments]
        self._spec_by_id = {spec.segment_id: spec for spec in self._specs}
        self._invalid_reasons: dict[int, str] = {}
        self._noisy_baseline: dict[int, Any] = {}
        self._cache_metadata = {
            **dict(metadata),
            "index_only": True,
            "loaded_motion_count": len(self._segments),
        }

    def _spec_from_segment(self, segment: Any) -> FrontRESSegmentSpec:
        denom = max(1, int(segment.motion_num_frames) - 1)
        return FrontRESSegmentSpec(
            segment_id=int(segment.segment_id),
            motion_id=str(segment.motion_rel_path),
            start_frame=int(segment.start_frame),
            start_time=float(segment.start_frame) / float(segment.fps),
            phase=float(segment.start_frame) / float(denom),
            horizon_k=int(segment.horizon_k),
            perturbation_family="index_only",
            perturbation_strength=0.0,
            perturbation_role="train",
            reset_mode_hint="auto",
            valid_for_training=True,
        )

    def _state_for_specs(self, specs: Sequence[FrontRESSegmentSpec]) -> FrontRESSegmentState:
        batch_size = len(specs)
        root_quat = torch.zeros((batch_size, 4), dtype=torch.float32, device=self.device)
        root_quat[:, 0] = 1.0
        return FrontRESSegmentState(
            root_pos=torch.zeros((batch_size, 3), dtype=torch.float32, device=self.device),
            root_quat=root_quat,
            root_lin_vel=torch.zeros((batch_size, 3), dtype=torch.float32, device=self.device),
            root_ang_vel=torch.zeros((batch_size, 3), dtype=torch.float32, device=self.device),
            dof_pos=torch.zeros((batch_size, self.dof_dim), dtype=torch.float32, device=self.device),
            dof_vel=torch.zeros((batch_size, self.dof_dim), dtype=torch.float32, device=self.device),
        )

    def _reference_for_specs(self, specs: Sequence[FrontRESSegmentSpec]) -> torch.Tensor:
        frames = max(int(spec.horizon_k) for spec in specs) + 1
        return torch.zeros((len(specs), frames, self.dof_dim * 2), dtype=torch.float32, device=self.device)


def _require_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def build_stage1_cache_motion_source(
    cache_dir: str | Path,
    *,
    include_boundary_diagnostic: bool = False,
) -> tuple[list[dict[str, Any]], FrontRESStage1CacheDatasetLoadSummary]:
    root = Path(cache_dir)
    metadata = read_cache_metadata(root)
    noisy_shard_id = int(metadata.get("noisy_shard_id", 0))
    strengths = [float(item) for item in metadata.get("strengths", [])]
    motion_source: list[dict[str, Any]] = []
    role_counts: dict[str, int] = {}
    skipped_boundary = 0
    included_boundary = 0
    for strength in strengths:
        noisy_path = (
            root
            / "manifests"
            / "noisy_variants"
            / _strength_dir(float(strength))
            / f"shard_{noisy_shard_id:06d}.pt"
        )
        for variant in read_noisy_variant_shard(noisy_path):
            params = dict(variant.descriptor.params)
            role = str(params.get("perturbation_role", "train"))
            role_counts[role] = role_counts.get(role, 0) + 1
            if role == "boundary_diagnostic" and not include_boundary_diagnostic:
                skipped_boundary += 1
                continue
            if role == "boundary_diagnostic":
                included_boundary += 1
            motion_source.append(_motion_from_noisy_variant(variant, role=role))
    summary = FrontRESStage1CacheDatasetLoadSummary(
        cache_dir=str(root),
        perturbation_curriculum_mode=str(metadata.get("perturbation_curriculum_mode", "")),
        metadata_noisy_count=int(metadata.get("noisy_count", 0)),
        loaded_motion_count=len(motion_source),
        skipped_boundary_diagnostic_count=skipped_boundary,
        included_boundary_diagnostic_count=included_boundary,
        role_counts=role_counts,
    )
    return motion_source, summary


def build_stage1_cache_lazy_records(
    cache_dir: str | Path,
    *,
    include_boundary_diagnostic: bool = False,
) -> tuple[list[FrontRESStage1CacheRecord], FrontRESStage1CacheDatasetLoadSummary]:
    root = Path(cache_dir)
    metadata = read_cache_metadata(root)
    noisy_shard_id = int(metadata.get("noisy_shard_id", 0))
    strengths = [float(item) for item in metadata.get("strengths", [])]
    records: list[FrontRESStage1CacheRecord] = []
    role_counts: dict[str, int] = {}
    skipped_boundary = 0
    included_boundary = 0
    next_id = 0
    for strength in strengths:
        noisy_path = (
            root
            / "manifests"
            / "noisy_variants"
            / _strength_dir(float(strength))
            / f"shard_{noisy_shard_id:06d}.pt"
        )
        _, manifest_records = read_noisy_variant_manifest_records(noisy_path)
        for manifest_record in manifest_records:
            descriptor = manifest_record["descriptor"]
            segment = manifest_record["segment"]
            params = dict(descriptor.get("params", {}))
            role = str(params.get("perturbation_role", "train"))
            role_counts[role] = role_counts.get(role, 0) + 1
            if role == "boundary_diagnostic" and not include_boundary_diagnostic:
                skipped_boundary += 1
                continue
            if role == "boundary_diagnostic":
                included_boundary += 1
            records.append(
                FrontRESStage1CacheRecord(
                    segment_id=next_id,
                    manifest_record=dict(manifest_record),
                    role=role,
                    valid_for_training=role == "train",
                    horizon_k=int(segment["horizon_k"]),
                    fps=float(segment["fps"]),
                    perturbation_family=str(descriptor["family"]),
                    perturbation_strength=float(descriptor["strength"]),
                    source_segment_id=int(segment["segment_id"]),
                    source_perturbation_id=int(descriptor["perturbation_id"]),
                )
            )
            next_id += 1
    summary = FrontRESStage1CacheDatasetLoadSummary(
        cache_dir=str(root),
        perturbation_curriculum_mode=str(metadata.get("perturbation_curriculum_mode", "")),
        metadata_noisy_count=int(metadata.get("noisy_count", 0)),
        loaded_motion_count=len(records),
        skipped_boundary_diagnostic_count=skipped_boundary,
        included_boundary_diagnostic_count=included_boundary,
        role_counts=role_counts,
    )
    return records, summary


def load_stage1_cache_dataset(
    cache_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    include_boundary_diagnostic: bool = False,
    lazy: bool = True,
    shard_cache_size: int = 8,
) -> FrontRESSegmentDataset | FrontRESStage1LazyCacheDataset | FrontRESStage1IndexDataset:
    metadata = _read_stage1_metadata_raw(cache_dir)
    if metadata.get("format") == "frontres_segment_cache_index_v1":
        dataset = FrontRESStage1IndexDataset(cache_dir=cache_dir, metadata=metadata, device=device)
        print(
            _log_block(
                "[FrontRES Segment Dataset]",
                "  cache_load: "
                "mode=index_only "
                f"loaded_motion_count={dataset.num_segments()} "
                "metadata_noisy_count=0 "
                "lazy=False index_only=True",
            ),
            flush=True,
        )
        return dataset
    if lazy:
        records, summary = build_stage1_cache_lazy_records(
            cache_dir,
            include_boundary_diagnostic=include_boundary_diagnostic,
        )
        dataset = FrontRESStage1LazyCacheDataset(
            cache_dir=cache_dir,
            records=records,
            summary=summary,
            device=device,
            shard_cache_size=shard_cache_size,
        )
        print(
            _log_block(
                "[FrontRES Segment Dataset]",
                "  cache_load: "
                f"mode={summary.perturbation_curriculum_mode} "
                f"metadata_noisy_count={summary.metadata_noisy_count} "
                f"loaded_motion_count={summary.loaded_motion_count} "
                f"skipped_boundary_diagnostic_count={summary.skipped_boundary_diagnostic_count} "
                f"included_boundary_diagnostic_count={summary.included_boundary_diagnostic_count} "
                f"role_counts={summary.role_counts} "
                f"lazy=True shard_cache_size={shard_cache_size}",
            ),
            flush=True,
        )
        return dataset
    motion_source, summary = build_stage1_cache_motion_source(
        cache_dir,
        include_boundary_diagnostic=include_boundary_diagnostic,
    )
    if not motion_source:
        raise ValueError(f"Stage 1 cache produced no trainable segment motions: {cache_dir}")
    horizon_k = int(motion_source[0]["horizon_k"])
    fps = float(motion_source[0]["fps"])
    dataset = FrontRESSegmentDataset(
        motion_source=motion_source,
        dt=1.0 / fps,
        default_horizon_k=horizon_k,
        device=device,
    )
    dataset.load_cache_metadata(summary.probe())
    print(
        _log_block(
            "[FrontRES Segment Dataset]",
            "  cache_load: "
            f"mode={summary.perturbation_curriculum_mode} "
            f"metadata_noisy_count={summary.metadata_noisy_count} "
            f"loaded_motion_count={summary.loaded_motion_count} "
            f"skipped_boundary_diagnostic_count={summary.skipped_boundary_diagnostic_count} "
            f"included_boundary_diagnostic_count={summary.included_boundary_diagnostic_count} "
            f"role_counts={summary.role_counts}",
        ),
        flush=True,
    )
    return dataset


def _motion_from_noisy_variant(variant: Any, *, role: str) -> dict[str, Any]:
    segment = variant.segment
    descriptor = variant.descriptor
    state = variant.noisy_state
    horizon_k = int(segment.horizon_k)
    frames = horizon_k + 1
    valid_for_training = role == "train"
    motion_id = f"{int(segment.segment_id)}:{int(descriptor.perturbation_id)}"
    return {
        "motion_id": motion_id,
        "source_segment_id": int(segment.segment_id),
        "source_perturbation_id": int(descriptor.perturbation_id),
        "fps": float(segment.fps),
        "horizon_k": horizon_k,
        "root_pos": _repeat_first_env(state.root_pos, frames),
        "root_quat": _repeat_first_env(state.root_quat, frames),
        "root_lin_vel": _repeat_first_env(state.root_lin_vel, frames),
        "root_ang_vel": _repeat_first_env(state.root_ang_vel, frames),
        "dof_pos": _repeat_first_env(state.joint_pos, frames),
        "dof_vel": _repeat_first_env(state.joint_vel, frames),
        "key_body_pos": _repeat_first_env(state.body_pos_w, frames),
        "key_body_quat": _repeat_first_env(state.body_quat_w, frames),
        "reference": torch.cat(
            [_repeat_first_env(state.joint_pos, frames), _repeat_first_env(state.joint_vel, frames)],
            dim=-1,
        ),
        "perturbation_family": str(descriptor.family),
        "perturbation_strength": float(descriptor.strength),
        "perturbation_role": role,
        "valid_for_training": valid_for_training,
        "reset_mode_hint": "direct",
        "noisy_baseline_score": variant.noisy_baseline_score.detach().cpu(),
        "noisy_fall": variant.noisy_fall.detach().cpu(),
        "descriptor_params": dict(descriptor.params),
    }


def _repeat_first_env(tensor: torch.Tensor, frames: int) -> torch.Tensor:
    value = tensor.detach().cpu()
    if value.shape[0] != 1:
        value = value[:1]
    return value[0].unsqueeze(0).repeat((int(frames),) + (1,) * (value.ndim - 1)).contiguous()


def _strength_dir(strength: float) -> str:
    text = f"{float(strength):.6f}".rstrip("0").rstrip(".")
    return "strength_" + text.replace("-", "neg_").replace(".", "p")
