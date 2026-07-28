"""Sequence-evaluation schemas, v015 composition, and legacy planning helpers.

Status: E-FI-46 connects ordinary NPZ plus fixed protocol to one deterministic
carrier. E-FI-28--E-FI-30 connect its NPZ/protocol identity through the
command-owned deployment carrier, per-frame FEMR, frozen GMT, and the immutable
no-feedback report. The older plan/reset helpers below remain legacy v002 and
are rejected by the v015 runner boundary.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch


_V015_DEPLOYMENT_COMPOSITION_KIND = "deployment_composition_v015"
_V015_DEPLOYMENT_REFERENCE_PROVENANCE = "deployment_reference_stream"
_V015_DEPLOYMENT_CARRIER_PROVENANCE = "materialized_deployment_carrier"
_V015_PERSISTENT_TEMPORAL_MODE = "persistent_full_sequence"
_V015_SUPPORTED_CORRUPTION_FAMILIES = frozenset(("planar", "yaw", "global_z", "local_rp"))
_V015_REQUIRED_NPZ_ARRAYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


class _FrontRESV015NoTrainingFeedback:
    """Expose immutable false feedback facts without accepting training payloads."""

    @property
    def return_feedback(self) -> bool:
        return False

    @property
    def priority_feedback(self) -> bool:
        return False

    @property
    def ppo_feedback(self) -> bool:
        return False

    @property
    def sampler_feedback(self) -> bool:
        return False

    @property
    def optimizer_feedback(self) -> bool:
        return False


@dataclass(frozen=True)
class FrontRESV015PersistentCorruptionProtocol:
    """Immutable identity of one full-sequence deployment corruption protocol."""

    corruption_id: str
    family: str
    seed: int
    parameters: tuple[tuple[str, str | int | float | bool], ...]
    temporal_mode: str
    protocol_hash: str

    def validate(self) -> None:
        if not self.corruption_id:
            raise ValueError("v015 persistent corruption requires a nonempty corruption_id")
        family_parts = tuple(part.strip() for part in self.family.split("+") if part.strip())
        if not family_parts or not set(family_parts).issubset(_V015_SUPPORTED_CORRUPTION_FAMILIES):
            raise ValueError(
                "v015 persistent corruption family must use only "
                f"{tuple(sorted(_V015_SUPPORTED_CORRUPTION_FAMILIES))}, got {self.family!r}"
            )
        if "+".join(sorted(set(family_parts))) != self.family:
            raise ValueError("v015 persistent corruption family must use canonical sorted unique names")
        if self.temporal_mode != _V015_PERSISTENT_TEMPORAL_MODE:
            raise ValueError("v015 composition requires persistent_full_sequence corruption")
        _validate_v015_corruption_parameters(self.parameters)
        expected_hash = _frontres_v015_corruption_protocol_hash(
            corruption_id=self.corruption_id,
            family=self.family,
            seed=self.seed,
            parameters=self.parameters,
            temporal_mode=self.temporal_mode,
        )
        if self.protocol_hash != expected_hash:
            raise ValueError("v015 persistent corruption protocol_hash does not match its immutable fields")


def build_frontres_v015_persistent_corruption_protocol(
    *,
    corruption_id: str,
    family: str,
    seed: int,
    parameters: Mapping[str, str | int | float | bool],
) -> FrontRESV015PersistentCorruptionProtocol:
    """Canonicalize scalar protocol metadata and seal its order-independent hash."""

    if not isinstance(parameters, Mapping):
        raise TypeError("v015 corruption parameters must be a scalar mapping")
    family_parts = tuple(part.strip() for part in str(family).split("+") if part.strip())
    canonical_family = "+".join(sorted(set(family_parts)))
    canonical_parameters = tuple(sorted((str(name), value) for name, value in parameters.items()))
    protocol = FrontRESV015PersistentCorruptionProtocol(
        corruption_id=str(corruption_id),
        family=canonical_family,
        seed=int(seed),
        parameters=canonical_parameters,
        temporal_mode=_V015_PERSISTENT_TEMPORAL_MODE,
        protocol_hash=_frontres_v015_corruption_protocol_hash(
            corruption_id=str(corruption_id),
            family=canonical_family,
            seed=int(seed),
            parameters=canonical_parameters,
            temporal_mode=_V015_PERSISTENT_TEMPORAL_MODE,
        ),
    )
    protocol.validate()
    return protocol


@dataclass(frozen=True)
class FrontRESV015DeploymentCompositionConfig:
    """Fail-closed identity config used to build the formal v015 request."""

    enabled: bool
    source_reference_path: str
    reference_path: str
    future_offsets: tuple[int, ...]
    corruption_protocol: FrontRESV015PersistentCorruptionProtocol
    legacy_modes: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.enabled is not True:
            raise ValueError("v015 deployment composition config must be explicitly enabled")
        if not self.source_reference_path or Path(self.source_reference_path).suffix.lower() != ".npz":
            raise ValueError("v015 deployment composition source_reference_path must name one explicit .npz file")
        if not self.reference_path or Path(self.reference_path).suffix.lower() != ".npz":
            raise ValueError("v015 deployment composition reference_path must name one explicit .npz file")
        if (
            not self.future_offsets
            or any(int(offset) <= 0 for offset in self.future_offsets)
            or tuple(sorted(set(int(offset) for offset in self.future_offsets))) != tuple(self.future_offsets)
        ):
            raise ValueError("v015 deployment composition future_offsets must be ordered unique positive integers")
        if not isinstance(self.corruption_protocol, FrontRESV015PersistentCorruptionProtocol):
            raise ValueError("v015 deployment composition requires a persistent corruption protocol")
        self.corruption_protocol.validate()
        if self.legacy_modes:
            raise ValueError(
                "v015 deployment composition rejects legacy evaluator mode mixing: "
                f"{tuple(self.legacy_modes)}"
            )


@dataclass(frozen=True)
class FrontRESV015DeploymentCompositionRequest(_FrontRESV015NoTrainingFeedback):
    """Validated identity of one structured deployment reference and protocol."""

    source_reference_path: str
    source_reference_file_hash: str
    reference_path: str
    reference_stream_id: str
    reference_file_hash: str
    frame_count: int
    joint_dof: int
    body_count: int
    fps: float
    future_offsets: tuple[int, ...]
    corruption_protocol: FrontRESV015PersistentCorruptionProtocol
    reference_provenance: str = _V015_DEPLOYMENT_REFERENCE_PROVENANCE
    evaluation_kind: str = _V015_DEPLOYMENT_COMPOSITION_KIND

    def validate(self) -> None:
        if Path(self.source_reference_path).suffix.lower() != ".npz":
            raise ValueError("v015 deployment request must retain an explicit source .npz path")
        if len(self.source_reference_file_hash) != 64 or any(
            ch not in "0123456789abcdef" for ch in self.source_reference_file_hash
        ):
            raise ValueError("v015 deployment source_reference_file_hash must be lowercase sha256")
        if Path(self.reference_path).suffix.lower() != ".npz":
            raise ValueError("v015 deployment request must retain an explicit .npz path")
        if self.reference_stream_id != f"deployment-npz:{self.reference_file_hash}":
            raise ValueError("v015 deployment reference_stream_id must be derived from the file hash")
        if len(self.reference_file_hash) != 64 or any(ch not in "0123456789abcdef" for ch in self.reference_file_hash):
            raise ValueError("v015 deployment reference_file_hash must be lowercase sha256")
        if self.reference_provenance != _V015_DEPLOYMENT_REFERENCE_PROVENANCE:
            raise ValueError("v015 composition requires deployment_reference_stream provenance")
        if self.evaluation_kind != _V015_DEPLOYMENT_COMPOSITION_KIND:
            raise ValueError("v015 deployment request has an invalid evaluation kind")
        if self.frame_count <= max(self.future_offsets, default=0):
            raise ValueError("v015 deployment reference is too short for its future_offsets")
        if self.joint_dof != 29 or self.body_count <= 0 or not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("v015 deployment reference requires q29, nonempty bodies, and positive finite fps")
        if tuple(sorted(set(self.future_offsets))) != self.future_offsets or any(value <= 0 for value in self.future_offsets):
            raise ValueError("v015 deployment request has invalid future_offsets")
        self.corruption_protocol.validate()


@dataclass(frozen=True)
class FrontRESV015DeploymentCarrier(_FrontRESV015NoTrainingFeedback):
    """封存 ordinary reference 到 controlled deployment carrier 的不可变回执."""

    source_reference_path: str
    source_reference_file_hash: str
    source_reference_stream_id: str
    carrier_path: str
    carrier_file_hash: str
    carrier_stream_id: str
    frame_count: int
    joint_dof: int
    body_count: int
    root_body_index: int
    fps: float
    corruption_protocol: FrontRESV015PersistentCorruptionProtocol
    materialized_delta_se3: tuple[float, float, float, float, float, float]
    intent_q29_hash: str
    materialization_hash: str
    provenance: str = _V015_DEPLOYMENT_CARRIER_PROVENANCE

    def validate(self) -> None:
        self.corruption_protocol.validate()
        source_path = Path(self.source_reference_path)
        carrier_path = Path(self.carrier_path)
        if not source_path.is_absolute() or not carrier_path.is_absolute():
            raise ValueError("v015 deployment carrier paths must be absolute")
        if source_path.suffix.lower() != ".npz" or carrier_path.suffix.lower() != ".npz":
            raise ValueError("v015 deployment carrier source/output must both be .npz")
        for name, value in (
            ("source_reference_file_hash", self.source_reference_file_hash),
            ("carrier_file_hash", self.carrier_file_hash),
            ("intent_q29_hash", self.intent_q29_hash),
            ("materialization_hash", self.materialization_hash),
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"v015 deployment carrier {name} must be lowercase sha256")
        if self.source_reference_stream_id != f"ordinary-npz:{self.source_reference_file_hash}":
            raise ValueError("v015 deployment carrier source stream id disagrees with its file hash")
        if self.carrier_stream_id != f"deployment-carrier-npz:{self.carrier_file_hash}":
            raise ValueError("v015 deployment carrier stream id disagrees with its file hash")
        if self.provenance != _V015_DEPLOYMENT_CARRIER_PROVENANCE:
            raise ValueError("v015 deployment carrier has invalid provenance")
        if (
            self.frame_count <= 0
            or self.joint_dof != 29
            or self.body_count <= 0
            or self.root_body_index < 0
            or self.root_body_index >= self.body_count
            or not math.isfinite(self.fps)
            or self.fps <= 0.0
        ):
            raise ValueError("v015 deployment carrier has invalid shape/fps identity")
        if len(self.materialized_delta_se3) != 6 or any(
            not math.isfinite(float(value)) for value in self.materialized_delta_se3
        ):
            raise ValueError("v015 deployment carrier Delta SE(3) must be finite [6]")
        if _sha256_file(source_path) != self.source_reference_file_hash:
            raise ValueError("v015 deployment carrier source file hash changed after materialization")
        if _sha256_file(carrier_path) != self.carrier_file_hash:
            raise ValueError("v015 deployment carrier file hash changed after materialization")
        if _frontres_v015_npz_intent_hash(source_path) != self.intent_q29_hash:
            raise ValueError("v015 deployment carrier source q29 identity changed")
        if _frontres_v015_npz_intent_hash(carrier_path) != self.intent_q29_hash:
            raise ValueError("v015 deployment carrier must preserve q29/dq29 exactly")
        expected = _frontres_v015_deployment_materialization_hash(
            source_file_hash=self.source_reference_file_hash,
            protocol_hash=self.corruption_protocol.protocol_hash,
            carrier_file_hash=self.carrier_file_hash,
            materialized_delta_se3=self.materialized_delta_se3,
            frame_count=self.frame_count,
            body_count=self.body_count,
            root_body_index=self.root_body_index,
            fps=self.fps,
        )
        if self.materialization_hash != expected:
            raise ValueError("v015 deployment carrier materialization hash disagrees with its sealed fields")


class FrontRESV015DeploymentCarrierLifecycle:
    """一个 lifecycle 只 materialize 一次, 后续读取不能重采样 protocol."""

    def __init__(
        self,
        *,
        source_path: str,
        output_path: str,
        corruption_protocol: FrontRESV015PersistentCorruptionProtocol,
    ) -> None:
        self._source_path = str(source_path)
        self._output_path = str(output_path)
        self._corruption_protocol = corruption_protocol
        self._carrier: FrontRESV015DeploymentCarrier | None = None
        self._state = "ready"

    @property
    def state(self) -> str:
        return self._state

    def materialize(self) -> FrontRESV015DeploymentCarrier:
        if self._state != "ready":
            raise RuntimeError("v015 deployment carrier lifecycle is already sealed; resampling is forbidden")
        carrier = materialize_frontres_v015_deployment_carrier(
            source_path=self._source_path,
            output_path=self._output_path,
            corruption_protocol=self._corruption_protocol,
        )
        self._carrier = carrier
        self._state = "sealed"
        return carrier

    def snapshot(self) -> FrontRESV015DeploymentCarrier:
        if self._state != "sealed" or self._carrier is None:
            raise RuntimeError("v015 deployment carrier is not sealed")
        return self._carrier


@dataclass(frozen=True)
class FrontRESV015DeploymentCompositionReport(_FrontRESV015NoTrainingFeedback):
    """Per-frame deployment-only metrics, separate from local K Gain and training."""

    request: FrontRESV015DeploymentCompositionRequest
    per_frame_femr_action_used: tuple[bool, ...]
    per_frame_intent_q29_error: tuple[float, ...]
    per_frame_physics_success: tuple[bool, ...]
    per_frame_fall: tuple[bool, ...]
    per_frame_zmp_margin: tuple[float | None, ...]
    per_frame_contact_consistency: tuple[float, ...]
    per_frame_policy_actions: tuple[tuple[tuple[float, ...], ...], ...]
    expected_contact_steps: tuple[tuple[tuple[bool, bool], ...], ...]
    actual_contact_steps: tuple[tuple[tuple[bool, bool], ...], ...]
    contact_mismatch_steps: tuple[tuple[tuple[bool, bool], ...], ...]
    phase_zmp_applicable_steps: tuple[tuple[bool, ...], ...]
    phase_zmp_violation_steps: tuple[tuple[float | None, ...], ...]
    phase_zmp_recovery_steps: tuple[tuple[bool, ...], ...]
    survival_steps: tuple[tuple[bool, ...], ...]
    lateral_roll_rad_steps: tuple[tuple[float, ...], ...]
    lateral_roll_cumulative_mean_rad_steps: tuple[tuple[float, ...], ...]
    unplanned_contact_steps: tuple[tuple[bool, ...], ...]
    evaluation_kind: str = _V015_DEPLOYMENT_COMPOSITION_KIND

    @property
    def frame_count(self) -> int:
        return self.request.frame_count - max(self.request.future_offsets)

    @property
    def reference_frame_count(self) -> int:
        return self.request.frame_count

    @property
    def femr_action_count(self) -> int:
        return sum(bool(value) for value in self.per_frame_femr_action_used)

    @property
    def accumulated_failure_count(self) -> int:
        return sum(not bool(value) for value in self.per_frame_physics_success)

    @property
    def mean_intent_q29_error(self) -> float:
        return sum(self.per_frame_intent_q29_error) / max(self.frame_count, 1)

    @property
    def contact_preservation_fraction(self) -> float:
        total = self.frame_count * len(self.contact_mismatch_steps[0])
        return 1.0 - sum(any(foot) for frame in self.contact_mismatch_steps for foot in frame) / max(total, 1)

    @property
    def phase_zmp_applicable_count(self) -> int:
        return sum(bool(value) for frame in self.phase_zmp_applicable_steps for value in frame)

    @property
    def phase_zmp_violation_count(self) -> int:
        return sum(
            value is not None and float(value) > 0.0
            for frame in self.phase_zmp_violation_steps
            for value in frame
        )

    @property
    def survival_fraction(self) -> float:
        total = self.frame_count * len(self.survival_steps[0])
        return sum(bool(value) for frame in self.survival_steps for value in frame) / max(total, 1)

    @property
    def max_abs_cumulative_lateral_roll_rad(self) -> float:
        return max(abs(float(value)) for frame in self.lateral_roll_cumulative_mean_rad_steps for value in frame)

    @property
    def unplanned_contact_event_count(self) -> int:
        return sum(bool(value) for frame in self.unplanned_contact_steps for value in frame)

    def validate(self) -> None:
        self.request.validate()
        if self.evaluation_kind != _V015_DEPLOYMENT_COMPOSITION_KIND:
            raise ValueError("v015 deployment report has an invalid evaluation kind")
        rows = (
            self.per_frame_femr_action_used,
            self.per_frame_intent_q29_error,
            self.per_frame_physics_success,
            self.per_frame_fall,
            self.per_frame_zmp_margin,
            self.per_frame_contact_consistency,
        )
        if any(not isinstance(values, tuple) or len(values) != self.frame_count for values in rows):
            raise ValueError("v015 deployment report per-frame length must equal its unclamped evaluated frame count")
        rich_rows = (
            self.per_frame_policy_actions,
            self.expected_contact_steps,
            self.actual_contact_steps,
            self.contact_mismatch_steps,
            self.phase_zmp_applicable_steps,
            self.phase_zmp_violation_steps,
            self.phase_zmp_recovery_steps,
            self.survival_steps,
            self.lateral_roll_rad_steps,
            self.lateral_roll_cumulative_mean_rad_steps,
            self.unplanned_contact_steps,
        )
        if any(not isinstance(values, tuple) or len(values) != self.frame_count for values in rich_rows):
            raise ValueError("v015 deployment quality trajectories must align with every evaluated frame")
        batch_sizes = {len(values[0]) for values in rich_rows if values}
        if len(batch_sizes) != 1 or next(iter(batch_sizes), 0) <= 0:
            raise ValueError("v015 deployment quality trajectories must share one positive row count")
        batch_size = next(iter(batch_sizes))
        if any(len(frame) != batch_size for values in rich_rows for frame in values):
            raise ValueError("v015 deployment quality trajectories lost row alignment")
        if any(len(action) != 6 for frame in self.per_frame_policy_actions for action in frame):
            raise ValueError("v015 deployment policy actions must be [T,B,6]")
        contact_rows = (self.expected_contact_steps, self.actual_contact_steps, self.contact_mismatch_steps)
        if any(len(contact) != 2 for values in contact_rows for frame in values for contact in frame):
            raise ValueError("v015 deployment Contact trajectories must be [T,B,2]")
        boolean_rows = (
            self.expected_contact_steps,
            self.actual_contact_steps,
            self.contact_mismatch_steps,
            self.phase_zmp_applicable_steps,
            self.phase_zmp_recovery_steps,
            self.survival_steps,
            self.unplanned_contact_steps,
        )
        if any(
            type(value) is not bool
            for rows_ in boolean_rows
            for frame in rows_
            for row in frame
            for value in (row if isinstance(row, tuple) else (row,))
        ):
            raise ValueError("v015 deployment Contact/ZMP/survival trajectories must contain bool values")
        if any(type(value) is not bool for value in self.per_frame_femr_action_used):
            raise ValueError("v015 per-frame FEMR action flags must be bool")
        if any(type(value) is not bool for value in self.per_frame_physics_success + self.per_frame_fall):
            raise ValueError("v015 per-frame physics success/fall flags must be bool")
        if any(success and fall for success, fall in zip(self.per_frame_physics_success, self.per_frame_fall, strict=True)):
            raise ValueError("v015 composition frame cannot report physics success and fall together")
        numeric_rows = (self.per_frame_intent_q29_error, self.per_frame_contact_consistency)
        if any(not math.isfinite(float(value)) for values in numeric_rows for value in values):
            raise ValueError("v015 deployment report metrics must be finite")
        if any(value is not None and not math.isfinite(float(value)) for value in self.per_frame_zmp_margin):
            raise ValueError("v015 deployment ZMP margins must be finite or explicit N/A")
        if any(float(value) < 0.0 for value in self.per_frame_intent_q29_error):
            raise ValueError("v015 per-frame q29 intent error must be nonnegative")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.per_frame_contact_consistency):
            raise ValueError("v015 per-frame contact consistency must be in [0,1]")
        finite_values = (
            value
            for rows_ in (
                self.per_frame_policy_actions,
                self.lateral_roll_rad_steps,
                self.lateral_roll_cumulative_mean_rad_steps,
            )
            for frame in rows_
            for row in frame
            for value in (row if isinstance(row, tuple) else (row,))
        )
        if any(not math.isfinite(float(value)) for value in finite_values):
            raise ValueError("v015 deployment action/lean trajectories must be finite")
        if any(
            value is not None and (not math.isfinite(float(value)) or float(value) < 0.0)
            for frame in self.phase_zmp_violation_steps
            for value in frame
        ):
            raise ValueError("v015 phase-ZMP violation must be nonnegative or explicit N/A")
        if any(
            bool(applicable) != (value is not None)
            for app_frame, value_frame in zip(
                self.phase_zmp_applicable_steps,
                self.phase_zmp_violation_steps,
                strict=True,
            )
            for applicable, value in zip(app_frame, value_frame, strict=True)
        ):
            raise ValueError("v015 phase-ZMP violation N/A must exactly match applicability")


@dataclass(frozen=True)
class FrontRESV015DeploymentCompositionRunConfig:
    """Formal S2B entry config for one pre-materialized deployment stream."""

    request_config: FrontRESV015DeploymentCompositionConfig
    report_path: str

    def validate(self) -> None:
        if not isinstance(self.request_config, FrontRESV015DeploymentCompositionConfig):
            raise TypeError("v015 composition run requires its dedicated request config")
        self.request_config.validate()
        parameters = dict(self.request_config.corruption_protocol.parameters)
        if parameters.get("source") != "pre_materialized_deployment_npz":
            raise ValueError(
                "v015 formal composition requires source=pre_materialized_deployment_npz; "
                "S2B does not invent or resample a corruption"
            )
        report_path = Path(self.report_path).expanduser()
        if not report_path.is_absolute() or report_path.suffix.lower() != ".json":
            raise ValueError("v015 composition report_path must be one absolute .json path")
        if not report_path.parent.is_dir():
            raise ValueError("v015 composition report parent directory must already exist")
        if report_path.exists():
            raise ValueError("v015 composition refuses to overwrite an existing report identity")


def load_frontres_v015_deployment_composition_request(
    config: FrontRESV015DeploymentCompositionConfig,
) -> FrontRESV015DeploymentCompositionRequest:
    """Validate one deployment `.npz` and return its immutable S1 identity."""

    if not isinstance(config, FrontRESV015DeploymentCompositionConfig):
        raise TypeError("v015 deployment composition request requires its dedicated config")
    config.validate()
    source_path = Path(config.source_reference_path).expanduser().resolve(strict=True)
    reference_path = Path(config.reference_path).expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(f"v015 deployment source reference is not a file: {source_path}")
    if not reference_path.is_file():
        raise ValueError(f"v015 deployment reference is not a file: {reference_path}")

    try:
        with np.load(source_path, allow_pickle=False) as data:
            missing = tuple(name for name in _V015_REQUIRED_NPZ_ARRAYS if name not in data.files)
            if missing:
                raise ValueError(f"v015 deployment source .npz is missing required arrays: {missing}")
            source_arrays = {name: np.asarray(data[name]) for name in _V015_REQUIRED_NPZ_ARRAYS}
        with np.load(reference_path, allow_pickle=False) as data:
            missing = tuple(name for name in _V015_REQUIRED_NPZ_ARRAYS if name not in data.files)
            if missing:
                raise ValueError(f"v015 deployment .npz is missing required arrays: {missing}")
            arrays = {name: np.asarray(data[name]) for name in _V015_REQUIRED_NPZ_ARRAYS}
    except (OSError, TypeError) as exc:
        raise ValueError(f"v015 deployment reference cannot be read as a safe .npz: {reference_path}") from exc

    source_shape = _validate_v015_deployment_npz_arrays(source_arrays)
    frame_count, joint_dof, body_count, fps = _validate_v015_deployment_npz_arrays(arrays)
    if source_shape != (frame_count, joint_dof, body_count, fps):
        raise ValueError("v015 deployment source/carrier shape and fps identity must match")
    if not np.array_equal(source_arrays["joint_pos"], arrays["joint_pos"]) or not np.array_equal(
        source_arrays["joint_vel"], arrays["joint_vel"]
    ):
        raise ValueError("v015 deployment source/carrier q29 intent identity must match exactly")
    source_file_hash = _sha256_file(source_path)
    file_hash = _sha256_file(reference_path)
    request = FrontRESV015DeploymentCompositionRequest(
        source_reference_path=str(source_path),
        source_reference_file_hash=source_file_hash,
        reference_path=str(reference_path),
        reference_stream_id=f"deployment-npz:{file_hash}",
        reference_file_hash=file_hash,
        frame_count=frame_count,
        joint_dof=joint_dof,
        body_count=body_count,
        fps=fps,
        future_offsets=tuple(config.future_offsets),
        corruption_protocol=config.corruption_protocol,
    )
    request.validate()
    return request


def materialize_frontres_v015_deployment_carrier(
    *,
    source_path: str,
    output_path: str,
    corruption_protocol: FrontRESV015PersistentCorruptionProtocol,
) -> FrontRESV015DeploymentCarrier:
    """只生成一次确定性的 root/global artifact carrier.

    这个 owner 只修改 body-frame reference arrays. q29/dq29 按 bit 复制,
    protocol metadata 只保留在回执中, 不进入 FEMR/GMT 消费的 archive.
    """

    if not isinstance(corruption_protocol, FrontRESV015PersistentCorruptionProtocol):
        raise TypeError("v015 deployment materializer requires its persistent corruption protocol")
    corruption_protocol.validate()
    source = Path(source_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser()
    if not output.is_absolute():
        output = output.resolve(strict=False)
    if source.suffix.lower() != ".npz" or output.suffix.lower() != ".npz":
        raise ValueError("v015 deployment materializer source/output must both be .npz")
    if not source.is_file():
        raise ValueError(f"v015 deployment materializer source is not a file: {source}")
    if not output.parent.is_dir():
        raise ValueError("v015 deployment materializer output parent must already exist")
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary.exists():
        raise ValueError("v015 deployment materializer output or partial path already exists")

    source_hash_before = _sha256_file(source)
    arrays = _frontres_v015_load_npz_arrays(source)
    frame_count, joint_dof, body_count, fps = _validate_v015_deployment_npz_arrays(arrays)
    source_hash_after = _sha256_file(source)
    if source_hash_after != source_hash_before:
        raise RuntimeError("v015 deployment materializer source changed while it was being read")
    intent_hash = _frontres_v015_intent_array_hash(arrays["joint_pos"], arrays["joint_vel"])
    delta = _frontres_v015_sample_persistent_delta(corruption_protocol)
    root_body_index = _frontres_v015_root_body_index(corruption_protocol, body_count=body_count)
    carrier_arrays = _frontres_v015_apply_persistent_delta(
        arrays,
        delta,
        root_body_index=root_body_index,
    )
    if _frontres_v015_intent_array_hash(
        carrier_arrays["joint_pos"], carrier_arrays["joint_vel"]
    ) != intent_hash:
        raise RuntimeError("v015 deployment materializer changed q29/dq29")
    _frontres_v015_write_deterministic_npz(output, carrier_arrays)
    carrier_hash = _sha256_file(output)
    materialization_hash = _frontres_v015_deployment_materialization_hash(
        source_file_hash=source_hash_before,
        protocol_hash=corruption_protocol.protocol_hash,
        carrier_file_hash=carrier_hash,
        materialized_delta_se3=delta,
        frame_count=frame_count,
        body_count=body_count,
        root_body_index=root_body_index,
        fps=fps,
    )
    carrier = FrontRESV015DeploymentCarrier(
        source_reference_path=str(source),
        source_reference_file_hash=source_hash_before,
        source_reference_stream_id=f"ordinary-npz:{source_hash_before}",
        carrier_path=str(output),
        carrier_file_hash=carrier_hash,
        carrier_stream_id=f"deployment-carrier-npz:{carrier_hash}",
        frame_count=frame_count,
        joint_dof=joint_dof,
        body_count=body_count,
        root_body_index=root_body_index,
        fps=fps,
        corruption_protocol=corruption_protocol,
        materialized_delta_se3=delta,
        intent_q29_hash=intent_hash,
        materialization_hash=materialization_hash,
    )
    try:
        carrier.validate()
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return carrier


def _frontres_v015_load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = tuple(name for name in _V015_REQUIRED_NPZ_ARRAYS if name not in data.files)
            if missing:
                raise ValueError(f"v015 deployment .npz is missing required arrays: {missing}")
            return {name: np.asarray(data[name]).copy() for name in _V015_REQUIRED_NPZ_ARRAYS}
    except (OSError, TypeError) as exc:
        raise ValueError(f"v015 deployment reference cannot be read as a safe .npz: {path}") from exc


def _frontres_v015_sample_persistent_delta(
    protocol: FrontRESV015PersistentCorruptionProtocol,
) -> tuple[float, float, float, float, float, float]:
    if int(protocol.seed) < 0:
        raise ValueError("v015 deployment materializer seed must be nonnegative")
    parameters = dict(protocol.parameters)
    allowed = {
        "source",
        "scale",
        "xy_std",
        "x_std",
        "y_std",
        "z_std",
        "roll_std",
        "pitch_std",
        "yaw_std",
        "root_body_index",
    }
    unknown = tuple(sorted(set(parameters) - allowed))
    if unknown:
        raise ValueError(f"v015 deployment materializer has unknown corruption parameters: {unknown}")
    scale = _frontres_v015_nonnegative_parameter(parameters, "scale", default=1.0)
    if scale <= 0.0:
        raise ValueError("v015 deployment materializer scale must be positive")
    family = set(protocol.family.split("+"))
    std = np.zeros(6, dtype=np.float64)
    if "planar" in family:
        xy_std = _frontres_v015_nonnegative_parameter(parameters, "xy_std", default=0.0)
        std[0] = _frontres_v015_nonnegative_parameter(parameters, "x_std", default=xy_std)
        std[1] = _frontres_v015_nonnegative_parameter(parameters, "y_std", default=xy_std)
        if std[0] == 0.0 and std[1] == 0.0:
            raise ValueError("v015 planar materialization requires xy_std or x_std/y_std")
    if "global_z" in family:
        std[2] = _frontres_v015_nonnegative_parameter(parameters, "z_std", default=0.0)
        if std[2] == 0.0:
            raise ValueError("v015 global_z materialization requires z_std")
    if "local_rp" in family:
        std[3] = _frontres_v015_nonnegative_parameter(parameters, "roll_std", default=0.0)
        std[4] = _frontres_v015_nonnegative_parameter(parameters, "pitch_std", default=0.0)
        if std[3] == 0.0 and std[4] == 0.0:
            raise ValueError("v015 local_rp materialization requires roll_std and/or pitch_std")
    if "yaw" in family:
        std[5] = _frontres_v015_nonnegative_parameter(parameters, "yaw_std", default=0.0)
        if std[5] == 0.0:
            raise ValueError("v015 yaw materialization requires yaw_std")
    rng = np.random.default_rng(int(protocol.seed))
    values = rng.normal(loc=0.0, scale=std * scale)
    return tuple(float(value) for value in values)


def _frontres_v015_root_body_index(
    protocol: FrontRESV015PersistentCorruptionProtocol,
    *,
    body_count: int,
) -> int:
    value = dict(protocol.parameters).get("root_body_index")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("v015 deployment materializer requires integer root_body_index in the protocol")
    index = int(value)
    if index < 0 or index >= int(body_count):
        raise ValueError(
            f"v015 deployment materializer root_body_index={index} is outside body_count={body_count}"
        )
    return index


def _frontres_v015_nonnegative_parameter(
    parameters: Mapping[str, str | int | float | bool],
    name: str,
    *,
    default: float,
) -> float:
    value = parameters.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"v015 deployment materializer parameter {name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"v015 deployment materializer parameter {name} must be finite and nonnegative")
    return value


def _frontres_v015_apply_persistent_delta(
    arrays: Mapping[str, np.ndarray],
    delta: tuple[float, float, float, float, float, float],
    *,
    root_body_index: int,
) -> dict[str, np.ndarray]:
    output = {name: np.asarray(value).copy() for name, value in arrays.items()}
    translation = np.asarray(delta[:3], dtype=np.float64)
    delta_quat = _frontres_v015_quat_from_euler(*delta[3:])
    rotation = _frontres_v015_quat_rotation_matrix(delta_quat)
    body_pos = np.asarray(arrays["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(arrays["body_quat_w"], dtype=np.float64)
    quat_norm = np.linalg.norm(body_quat, axis=-1)
    if not bool(np.isfinite(quat_norm).all()) or not bool(np.allclose(quat_norm, 1.0, rtol=1e-4, atol=1e-4)):
        raise ValueError("v015 deployment materializer requires normalized body quaternions")
    root_pos = body_pos[:, root_body_index : root_body_index + 1, :]
    relative = body_pos - root_pos
    transformed_pos = root_pos + np.einsum("ij,tkj->tki", rotation, relative) + translation
    transformed_quat = _frontres_v015_quat_multiply(delta_quat, body_quat)
    transformed_quat /= np.linalg.norm(transformed_quat, axis=-1, keepdims=True)
    transformed_lin = np.einsum(
        "ij,tkj->tki", rotation, np.asarray(arrays["body_lin_vel_w"], dtype=np.float64)
    )
    transformed_ang = np.einsum(
        "ij,tkj->tki", rotation, np.asarray(arrays["body_ang_vel_w"], dtype=np.float64)
    )
    output["body_pos_w"] = transformed_pos.astype(arrays["body_pos_w"].dtype, copy=False)
    output["body_quat_w"] = transformed_quat.astype(arrays["body_quat_w"].dtype, copy=False)
    output["body_lin_vel_w"] = transformed_lin.astype(arrays["body_lin_vel_w"].dtype, copy=False)
    output["body_ang_vel_w"] = transformed_ang.astype(arrays["body_ang_vel_w"].dtype, copy=False)
    return output


def _frontres_v015_quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _frontres_v015_quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(np.asarray(lhs), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(rhs), -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _frontres_v015_quat_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _frontres_v015_write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, mode="x", compression=zipfile.ZIP_STORED) as archive:
            for name in _V015_REQUIRED_NPZ_ARRAYS:
                buffer = io.BytesIO()
                np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(filename=f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _frontres_v015_intent_array_hash(joint_pos: np.ndarray, joint_vel: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, value in (("joint_pos", joint_pos), ("joint_vel", joint_vel)):
        array = np.asarray(value)
        digest.update(f"{name}:{array.dtype}:{tuple(array.shape)}:".encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _frontres_v015_npz_intent_hash(path: Path) -> str:
    arrays = _frontres_v015_load_npz_arrays(path)
    return _frontres_v015_intent_array_hash(arrays["joint_pos"], arrays["joint_vel"])


def _frontres_v015_deployment_materialization_hash(
    *,
    source_file_hash: str,
    protocol_hash: str,
    carrier_file_hash: str,
    materialized_delta_se3: tuple[float, float, float, float, float, float],
    frame_count: int,
    body_count: int,
    root_body_index: int,
    fps: float,
) -> str:
    payload = {
        "body_count": int(body_count),
        "carrier_file_hash": str(carrier_file_hash),
        "fps": float(fps),
        "frame_count": int(frame_count),
        "materialized_delta_se3": tuple(float(value) for value in materialized_delta_se3),
        "protocol_hash": str(protocol_hash),
        "root_body_index": int(root_body_index),
        "source_file_hash": str(source_file_hash),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def run_frontres_v015_deployment_composition_eval(
    runner: Any,
    *,
    config: FrontRESV015DeploymentCompositionRunConfig,
) -> FrontRESV015DeploymentCompositionReport:
    """Execute one isolated deployment sequence and atomically write its report.

    Status: S2B formal offline connector. This path performs deterministic
    FEMR inference and frozen-GMT execution but has no training-feedback call.
    """

    if not isinstance(config, FrontRESV015DeploymentCompositionRunConfig):
        raise TypeError("v015 composition executor requires its dedicated run config")
    config.validate()
    request = load_frontres_v015_deployment_composition_request(config.request_config)
    command = _frontres_v015_deployment_motion_command(runner)
    set_sequence = getattr(command, "set_frontres_v015_deployment_sequence", None)
    clear_sequence = getattr(command, "clear_frontres_v015_deployment_sequence", None)
    advance_sequence = getattr(command, "advance_frontres_v015_deployment_sequence", None)
    if not all(callable(value) for value in (set_sequence, clear_sequence, advance_sequence)):
        raise RuntimeError("v015 composition requires the verified S2A command carrier lifecycle")
    if int(getattr(getattr(command, "cfg", None), "motion_horizon", 0)) != 1 or not bool(
        getattr(getattr(command, "cfg", None), "command_velocity", False)
    ):
        raise RuntimeError("v015 composition requires GMT current command [q29,dq29] with motion_horizon=1")
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    if policy is None or int(getattr(policy, "num_task_corrections", 0) or 0) != 6:
        raise RuntimeError("v015 composition requires the full-6D FEMR policy")
    gmt_policy = getattr(policy, "gmt_policy", None)
    if gmt_policy is None or bool(getattr(gmt_policy, "training", True)) or any(
        parameter.requires_grad for parameter in gmt_policy.parameters()
    ):
        raise RuntimeError("v015 composition requires frozen GMT eval parameters")
    get_correction = getattr(policy, "get_task_correction_inference", None)
    get_env_action = getattr(policy, "get_env_action", None)
    apply_correction = getattr(runner, "_apply_frontres_task_corrections", None)
    read_context = getattr(runner, "_read_frontres_v015_deployment_context", None)
    build_observation = getattr(runner, "_build_frontres_v015_deployment_observation", None)
    if not all(callable(value) for value in (get_correction, get_env_action, apply_correction, read_context, build_observation)):
        raise RuntimeError("v015 composition formal actor/GMT connectors are incomplete")

    training_before = _frontres_v015_training_state_fingerprint(runner)
    femr_used: list[bool] = []
    intent_error: list[float] = []
    physics_success: list[bool] = []
    fall: list[bool] = []
    policy_actions: list[torch.Tensor] = []
    actual_contacts: list[torch.Tensor] = []
    zmp_margins: list[torch.Tensor] = []
    survival_rows: list[torch.Tensor] = []
    lateral_roll_rows: list[torch.Tensor] = []
    evaluated_frames = request.frame_count - max(request.future_offsets)
    expected_sequence, envelope_sequence = _frontres_v015_deployment_expected_physics(request, command, runner)
    metric_provider = getattr(runner, "_frontres_v015_deployment_metric_provider", None)
    if not callable(metric_provider):
        from rsl_rl.runners.frontres_segment_live_probe import _prepare_frontres_raw_contact_views

        _prepare_frontres_raw_contact_views(runner)
    alive = torch.ones(int(getattr(command, "num_envs", 0)), device=runner.device, dtype=torch.bool)
    set_sequence(request)
    try:
        with _frontres_v015_deployment_inference_mode(runner), torch.inference_mode():
            for frame_index in range(evaluated_frames):
                snapshot = read_context()
                cursors = snapshot["frame_indices"]
                if not torch.equal(cursors, torch.full_like(cursors, frame_index)):
                    raise RuntimeError(
                        "v015 composition frame/cursor identity diverged: "
                        f"expected={frame_index} got={cursors.detach().cpu().tolist()}"
                    )
                raw_obs, _extras = _frontres_v015_read_policy_observation(runner)
                actor_obs = build_observation(raw_obs, snapshot=snapshot)
                actor_obs = _frontres_v015_normalize_observation(runner, actor_obs)
                correction = get_correction(actor_obs)
                if (
                    not isinstance(correction, torch.Tensor)
                    or tuple(correction.shape) != (int(raw_obs.shape[0]), 6)
                    or correction.requires_grad
                    or not bool(torch.isfinite(correction).all().item())
                ):
                    raise RuntimeError("v015 composition FEMR correction must be detached finite [B,6]")
                apply_correction(correction, int(correction.shape[0]), allow_oracle=False)

                corrected_raw, _corrected_extras = _frontres_v015_read_policy_observation(runner)
                corrected_obs = build_observation(corrected_raw, snapshot=snapshot)
                corrected_obs = _frontres_v015_normalize_observation(runner, corrected_obs)
                motor_action = get_env_action(corrected_obs, correction)
                if (
                    not isinstance(motor_action, torch.Tensor)
                    or motor_action.ndim != 2
                    or int(motor_action.shape[0]) != int(raw_obs.shape[0])
                    or motor_action.requires_grad
                    or not bool(torch.isfinite(motor_action).all().item())
                ):
                    raise RuntimeError("v015 composition frozen GMT action must be detached finite [B,A]")
                _next_obs, _reward, dones, infos = runner.env.step(motor_action.to(runner.env.device))
                dones = torch.as_tensor(dones, device=runner.device, dtype=torch.bool).flatten()
                if int(dones.numel()) != int(raw_obs.shape[0]):
                    raise RuntimeError("v015 composition dones must align with command rows")
                frame_metrics = _frontres_v015_deployment_frame_metrics(
                    runner,
                    command,
                    frame_index=frame_index,
                    dones=dones,
                    infos=infos,
                    expected_support=expected_sequence[frame_index].unsqueeze(0).expand(int(dones.numel()), -1),
                    expected_support_envelope=envelope_sequence[frame_index].unsqueeze(0).expand(
                        int(dones.numel()), -1
                    ),
                )
                executed_q29 = getattr(getattr(command, "robot", None), "data", None)
                executed_q29 = getattr(executed_q29, "joint_pos", None)
                if not isinstance(executed_q29, torch.Tensor) or tuple(executed_q29.shape) != tuple(
                    snapshot["intent_q29"][:, 0].shape
                ):
                    raise RuntimeError("v015 composition requires executed robot q29 aligned to deployment intent")
                q29_error = (executed_q29 - snapshot["intent_q29"][:, 0]).abs().mean()

                femr_used.append(True)
                intent_error.append(float(q29_error.item()))
                fall.append(bool(frame_metrics["fall"].any().item()))
                policy_actions.append(correction.detach().clone())
                actual_contacts.append(frame_metrics["actual_contact"].detach().clone())
                zmp_margins.append(frame_metrics["zmp_margin"].detach().clone())
                alive = alive & ~frame_metrics["fall"]
                survival_rows.append(alive.detach().clone())
                lateral_roll_rows.append(frame_metrics["lateral_roll_rad"].detach().clone())
                if frame_index + 1 < evaluated_frames:
                    advance_sequence()
    finally:
        clear_sequence()

    training_after = _frontres_v015_training_state_fingerprint(runner)
    if training_after != training_before:
        changed = tuple(name for name in training_before if training_before[name] != training_after.get(name))
        raise RuntimeError(f"v015 composition mutated forbidden training state: {changed}")
    expected_steps = expected_sequence[:evaluated_frames].unsqueeze(1).expand(-1, int(command.num_envs), -1)
    actual_steps = torch.stack(actual_contacts, dim=0)
    margin_steps = torch.stack(zmp_margins, dim=0)
    valid_steps = torch.ones_like(margin_steps, dtype=torch.bool)
    phase_provider = getattr(runner, "_frontres_v015_deployment_phase_provider", None)
    if not callable(phase_provider):
        from rsl_rl.frontres.frontres_gain import evaluate_phase_conditioned_physics as phase_provider
    phase = phase_provider(
        expected_steps,
        actual_steps,
        margin_steps,
        valid_steps,
        timing_tolerance=int(getattr(runner, "cfg", {}).get("frontres_physics_contact_timing_tolerance", 1)),
        recovery_window=int(getattr(runner, "cfg", {}).get("frontres_physics_zmp_recovery_window", 1)),
        zmp_violation_scale=float(getattr(runner, "cfg", {}).get("frontres_physics_zmp_violation_scale", 0.05)),
        dt=1.0 / float(request.fps),
    )
    mismatch = phase["contact_mismatch_steps"].bool()
    applicable = phase["zmp_applicable_steps"].bool()
    violation = phase["zmp_step_violation"].float()
    survival = torch.stack(survival_rows, dim=0)
    roll = torch.stack(lateral_roll_rows, dim=0)
    roll_cumulative = roll.cumsum(dim=0) / torch.arange(
        1, evaluated_frames + 1, device=roll.device, dtype=roll.dtype
    ).unsqueeze(1)
    recovery = expected_steps.any(dim=-1) & actual_steps.any(dim=-1) & ~applicable
    unplanned = _frontres_v015_unplanned_contact_steps(
        expected_steps,
        actual_steps,
        timing_tolerance=int(getattr(runner, "cfg", {}).get("frontres_physics_contact_timing_tolerance", 1)),
    )
    row_success = survival & ~mismatch.any(dim=-1) & ~(applicable & (violation > 0.0))
    physics_success.extend(bool(value) for value in row_success.all(dim=1).detach().cpu().tolist())
    contact_consistency = (1.0 - mismatch.float().mean(dim=(1, 2))).detach().cpu().tolist()
    zmp_margin: list[float | None] = []
    for frame_margin, frame_applicable in zip(margin_steps, applicable, strict=True):
        values = frame_margin[frame_applicable]
        zmp_margin.append(None if int(values.numel()) == 0 else float(values.mean().item()))

    def nested(value: torch.Tensor) -> tuple:
        return tuple(_frontres_v015_tuple_tree(item) for item in value.detach().cpu().tolist())

    zmp_violation_steps = tuple(
        tuple(float(violation[t, b].item()) if bool(applicable[t, b]) else None for b in range(int(applicable.shape[1])))
        for t in range(evaluated_frames)
    )
    report = FrontRESV015DeploymentCompositionReport(
        request=request,
        per_frame_femr_action_used=tuple(femr_used),
        per_frame_intent_q29_error=tuple(intent_error),
        per_frame_physics_success=tuple(physics_success),
        per_frame_fall=tuple(fall),
        per_frame_zmp_margin=tuple(zmp_margin),
        per_frame_contact_consistency=tuple(contact_consistency),
        per_frame_policy_actions=nested(torch.stack(policy_actions, dim=0)),
        expected_contact_steps=nested(expected_steps.bool()),
        actual_contact_steps=nested(actual_steps.bool()),
        contact_mismatch_steps=nested(mismatch),
        phase_zmp_applicable_steps=nested(applicable),
        phase_zmp_violation_steps=zmp_violation_steps,
        phase_zmp_recovery_steps=nested(recovery),
        survival_steps=nested(survival),
        lateral_roll_rad_steps=nested(roll),
        lateral_roll_cumulative_mean_rad_steps=nested(roll_cumulative),
        unplanned_contact_steps=nested(unplanned),
    )
    report.validate()
    _write_frontres_v015_deployment_composition_report(report, Path(config.report_path))
    return report


def _frontres_v015_deployment_motion_command(runner: Any) -> Any:
    env = getattr(runner, "env", None)
    env = getattr(env, "unwrapped", env)
    manager = getattr(env, "command_manager", None)
    get_term = getattr(manager, "get_term", None)
    command = get_term("motion") if callable(get_term) else getattr(manager, "_terms", {}).get("motion")
    if command is None:
        raise RuntimeError("v015 composition requires the formal motion command owner")
    return command


def _frontres_v015_read_policy_observation(runner: Any) -> tuple[torch.Tensor, Mapping[str, Any]]:
    obs, extras = runner.env.get_observations()
    obs_dict = extras.get("observations", {}) if isinstance(extras, Mapping) else {}
    obs_type = getattr(runner, "policy_obs_type", None)
    if obs_type is not None and obs_type in obs_dict:
        obs = obs_dict[obs_type]
    obs = torch.as_tensor(obs, device=runner.device)
    if obs.ndim != 2 or not torch.is_floating_point(obs) or not bool(torch.isfinite(obs).all().item()):
        raise RuntimeError("v015 composition raw policy observation must be finite [B,D]")
    return obs, obs_dict


def _frontres_v015_normalize_observation(runner: Any, obs: torch.Tensor) -> torch.Tensor:
    if not bool(getattr(runner, "cfg", {}).get("empirical_normalization", False)):
        return obs
    normalize = getattr(runner, "_apply_obs_normalizer", None)
    if not callable(normalize):
        raise RuntimeError("v015 composition requires the formal observation normalizer")
    normalized = normalize(obs)
    if tuple(normalized.shape) != tuple(obs.shape) or not bool(torch.isfinite(normalized).all().item()):
        raise RuntimeError("v015 composition normalizer changed shape or produced non-finite values")
    return normalized


def _frontres_v015_deployment_expected_physics(
    request: FrontRESV015DeploymentCompositionRequest,
    command: Any,
    runner: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Clean, evaluator-only Contact phase and support envelopes."""

    provider = getattr(runner, "_frontres_v015_deployment_expected_physics_provider", None)
    if callable(provider):
        support, envelope = provider(request=request, command=command)
        support = torch.as_tensor(support, device=runner.device, dtype=torch.bool)
        envelope = torch.as_tensor(envelope, device=runner.device, dtype=torch.float32)
        if tuple(support.shape) != (request.frame_count, 2) or tuple(envelope.shape) != (
            request.frame_count,
            6,
        ):
            raise RuntimeError("v015 deployment expected-Physics provider returned an invalid trajectory shape")
        return support.detach().clone(), envelope.detach().clone()

    arrays = _frontres_v015_load_npz_arrays(Path(request.source_reference_path))
    body_pos = torch.as_tensor(arrays["body_pos_w"], device=runner.device, dtype=torch.float32)
    body_quat = torch.as_tensor(arrays["body_quat_w"], device=runner.device, dtype=torch.float32)
    left = int(getattr(command, "left_foot_idx", -1))
    right = int(getattr(command, "right_foot_idx", -1))
    if left < 0 or right < 0 or left == right or max(left, right) >= int(body_pos.shape[1]):
        raise RuntimeError("v015 deployment Physics requires two valid command-owned Clean foot indices")
    from rsl_rl.frontres.frontres_balance import expected_support_and_envelope_from_foot_pose

    return expected_support_and_envelope_from_foot_pose(
        body_pos[:, (left, right)],
        body_quat[:, (left, right)],
        contact_height=float(getattr(getattr(command, "cfg", None), "frontres_expected_contact_height", 0.08)),
        foot_half_length=float(
            getattr(getattr(command, "cfg", None), "frontres_expected_foot_half_length", 0.10)
        ),
        foot_half_width=float(
            getattr(getattr(command, "cfg", None), "frontres_expected_foot_half_width", 0.05)
        ),
    )


def _frontres_v015_deployment_contact_wrench_frame(
    runner: Any,
    *,
    expected_support: torch.Tensor,
    expected_support_envelope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture sensor-authoritative Contact and contact-wrench ZMP for one frame."""

    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    scene = getattr(env, "scene", None)
    if scene is None:
        raise RuntimeError("v015 deployment Physics requires the formal IsaacLab scene")
    from rsl_rl.runners.frontres_segment_live_probe import _pad_raw_contact_slots, _raw_filtered_contact_rows
    from rsl_rl.frontres.frontres_balance import contact_wrench_zmp_xy, expected_support_envelope_margin

    sensors = []
    actual = []
    for name in ("frontres_left_foot_contacts", "frontres_right_foot_contacts"):
        try:
            sensor = scene[name]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"v015 deployment Physics is missing {name}") from exc
        force_matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
        threshold = getattr(getattr(sensor, "cfg", None), "force_threshold", None)
        if not isinstance(force_matrix, torch.Tensor) or not isinstance(threshold, (int, float)):
            raise RuntimeError(f"v015 deployment Physics requires filtered force_matrix_w for {name}")
        force_matrix = force_matrix.to(device=runner.device, dtype=torch.float32)
        if force_matrix.ndim != 4 or tuple(force_matrix.shape[:2]) != (int(expected_support.shape[0]), 1):
            raise RuntimeError(f"{name} filtered force matrix must be [B,1,F,3]")
        if not bool(torch.isfinite(force_matrix).all()) or not math.isfinite(float(threshold)) or float(threshold) <= 0.0:
            raise RuntimeError(f"{name} filtered contact forces/threshold must be finite with positive threshold")
        actual.append(force_matrix[..., 2].sum(dim=(1, 2)).abs() >= float(threshold))
        sensors.append(sensor)
    actual_contact = torch.stack(actual, dim=-1)
    raw = [
        _raw_filtered_contact_rows(sensor, num_envs=int(expected_support.shape[0]), device=runner.device)
        for sensor in sensors
    ]
    contact_slots = max(int(value[0].shape[2]) for value in raw)
    raw = [_pad_raw_contact_slots(value, contact_slots=contact_slots) for value in raw]
    points = torch.cat(tuple(value[0] for value in raw), dim=1)
    forces = torch.cat(tuple(value[1] for value in raw), dim=1)
    normals = torch.cat(tuple(value[2] for value in raw), dim=1)
    valid = torch.cat(tuple(value[3] for value in raw), dim=1)
    zmp_xy, zmp_valid = contact_wrench_zmp_xy(points, forces, normals, valid)
    origins = getattr(scene, "env_origins", None)
    if not isinstance(origins, torch.Tensor) or tuple(origins.shape[:1]) != (int(expected_support.shape[0]),):
        raise RuntimeError("v015 deployment Physics requires row-aligned scene.env_origins")
    margin = expected_support_envelope_margin(
        zmp_xy,
        expected_support_envelope,
        expected_support,
        env_origins_xy=origins[:, :2].to(device=runner.device, dtype=torch.float32),
    )
    required = expected_support.bool().any(dim=-1) & actual_contact.any(dim=-1)
    if bool((required & ~zmp_valid).any()):
        raise RuntimeError("v015 deployment loaded support is missing a finite raw contact-wrench resultant")
    margin = torch.where(required, margin, torch.full_like(margin, float("nan")))
    return actual_contact.detach().clone(), margin.detach().clone()


def _frontres_v015_robot_lateral_roll(command: Any) -> torch.Tensor:
    quat = getattr(command, "robot_anchor_quat_w", None)
    if not isinstance(quat, torch.Tensor) or quat.ndim != 2 or int(quat.shape[1]) != 4:
        raise RuntimeError("v015 deployment sustained-lean evidence requires robot root quaternion [B,4]")
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square())).detach().clone()


def _frontres_v015_unplanned_contact_steps(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    timing_tolerance: int,
) -> torch.Tensor:
    expected_transition = torch.zeros(expected.shape[:2], device=expected.device, dtype=torch.bool)
    actual_transition = torch.zeros_like(expected_transition)
    if int(expected.shape[0]) > 1:
        expected_transition[1:] = (expected[1:] != expected[:-1]).any(dim=-1)
        actual_transition[1:] = (actual[1:] != actual[:-1]).any(dim=-1)
    planned = torch.zeros_like(expected_transition)
    for delta in range(-int(timing_tolerance), int(timing_tolerance) + 1):
        source = torch.arange(int(expected.shape[0]), device=expected.device) + delta
        inside = (source >= 0) & (source < int(expected.shape[0]))
        planned |= expected_transition.index_select(0, source.clamp(0, int(expected.shape[0]) - 1)) & inside.unsqueeze(1)
    return actual_transition & ~planned


def _frontres_v015_tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_frontres_v015_tuple_tree(item) for item in value)
    return value


def _frontres_v015_deployment_frame_metrics(
    runner: Any,
    command: Any,
    *,
    frame_index: int,
    dones: torch.Tensor,
    infos: Any,
    expected_support: torch.Tensor,
    expected_support_envelope: torch.Tensor,
) -> dict[str, torch.Tensor]:
    provider = getattr(runner, "_frontres_v015_deployment_metric_provider", None)
    if callable(provider):
        values = provider(
            frame_index=frame_index,
            dones=dones,
            infos=infos,
            command=command,
            expected_support=expected_support,
            expected_support_envelope=expected_support_envelope,
        )
    else:
        time_outs = infos.get("time_outs") if isinstance(infos, Mapping) else None
        time_outs = (
            torch.as_tensor(time_outs, device=dones.device, dtype=torch.bool)
            if time_outs is not None
            else torch.zeros_like(dones)
        )
        fall = dones & ~time_outs
        actual_contact, zmp_margin = _frontres_v015_deployment_contact_wrench_frame(
            runner,
            expected_support=expected_support,
            expected_support_envelope=expected_support_envelope,
        )
        values = {
            "fall": fall,
            "zmp_margin": zmp_margin,
            "actual_contact": actual_contact,
            "lateral_roll_rad": _frontres_v015_robot_lateral_roll(command),
        }
    required = {"fall", "zmp_margin", "actual_contact", "lateral_roll_rad"}
    if not isinstance(values, Mapping) or set(values) != required:
        raise RuntimeError("v015 composition metric provider returned an invalid schema")
    output: dict[str, torch.Tensor] = {}
    for name in required:
        value = torch.as_tensor(values[name], device=runner.device)
        if name == "actual_contact":
            if tuple(value.shape) != (int(dones.numel()), 2):
                raise RuntimeError("v015 composition actual Contact must be row-aligned [B,2]")
            output[name] = value.bool().detach().clone()
            continue
        value = value.flatten()
        if int(value.numel()) != int(dones.numel()):
            raise RuntimeError(f"v015 composition metric {name} must be row-aligned [B]")
        if name == "fall":
            value = value.bool()
        else:
            value = value.float()
            if name != "zmp_margin" and not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(f"v015 composition metric {name} must be finite")
        output[name] = value.detach().clone()
    return output


def _frontres_v015_contact_consistency(runner: Any, command: Any) -> torch.Tensor:
    body_names = list(getattr(getattr(command, "cfg", None), "body_names", ()))
    foot_names = getattr(runner, "cfg", {}).get(
        "frontres_balance_foot_body_names",
        getattr(runner, "cfg", {}).get(
            "frontres_exec_foot_body_names", ["left_ankle_roll_link", "right_ankle_roll_link"]
        ),
    )
    foot_ids = [index for index, name in enumerate(body_names) if name in set(foot_names)]
    if len(foot_ids) != 2:
        raise RuntimeError("v015 composition contact metric requires exactly two configured foot bodies")
    reference = getattr(command, "body_pos_w", None)
    robot = getattr(command, "robot_body_pos_w", None)
    if not isinstance(reference, torch.Tensor) or not isinstance(robot, torch.Tensor):
        raise RuntimeError("v015 composition contact metric requires reference and robot body positions")
    threshold = float(getattr(runner, "cfg", {}).get("frontres_balance_contact_height", 0.08))
    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    origins = getattr(getattr(env, "scene", None), "env_origins", None)
    reference_z = reference[:, foot_ids, 2]
    robot_z = robot[:, foot_ids, 2]
    if isinstance(origins, torch.Tensor):
        reference_z = reference_z - origins[:, 2].unsqueeze(1)
        robot_z = robot_z - origins[:, 2].unsqueeze(1)
    return ((reference_z <= threshold) == (robot_z <= threshold)).float().mean(dim=-1)


def _frontres_v015_training_state_fingerprint(runner: Any) -> dict[str, str]:
    alg = getattr(runner, "alg", None)
    objects = {
        "optimizer": getattr(alg, "optimizer", None),
        "sampler": getattr(runner, "_frontres_segment_sampler", None),
        "storage": getattr(runner, "storage", None),
        "transition": getattr(alg, "transition", None),
        "prefix_normalizer": getattr(runner, "_frontres_extra_normalizer", None),
        "gmt_normalizer": getattr(runner, "obs_normalizer", None),
        "privileged_normalizer": getattr(runner, "privileged_obs_normalizer", None),
        "teacher_normalizer": getattr(runner, "teacher_obs_normalizer", None),
    }
    return {name: _frontres_v015_object_state_hash(value) for name, value in objects.items()}


@contextmanager
def _frontres_v015_deployment_inference_mode(runner: Any):
    """Freeze inference owners without losing their original mixed modes."""

    roots = (
        getattr(getattr(runner, "alg", None), "policy", None),
        getattr(runner, "_frontres_extra_normalizer", None),
        getattr(runner, "obs_normalizer", None),
        getattr(runner, "privileged_obs_normalizer", None),
        getattr(runner, "teacher_obs_normalizer", None),
    )
    module_modes: dict[torch.nn.Module, bool] = {}
    for root in roots:
        if not isinstance(root, torch.nn.Module):
            continue
        for module in root.modules():
            module_modes.setdefault(module, bool(module.training))
    for module in module_modes:
        module.training = False
    try:
        yield
    finally:
        for module, was_training in module_modes.items():
            module.training = was_training


def _frontres_v015_object_state_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item, key=lambda entry: str(entry)):
                update(str(key))
                update(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(f"sequence:{len(item)}[".encode("ascii"))
            for entry in item:
                update(entry)
            digest.update(b"]")
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(repr(item).encode("utf-8"))
        elif hasattr(item, "state_dict") and callable(item.state_dict):
            update(item.state_dict())
        elif hasattr(item, "__dict__"):
            update({key: entry for key, entry in vars(item).items() if not callable(entry)})
        else:
            digest.update(type(item).__qualname__.encode("utf-8"))

    update(value)
    return digest.hexdigest()


def _write_frontres_v015_deployment_composition_report(
    report: FrontRESV015DeploymentCompositionReport,
    path: Path,
) -> None:
    report.validate()
    payload = {
        "evaluation_kind": report.evaluation_kind,
        "source_reference_path": report.request.source_reference_path,
        "source_reference_file_hash": report.request.source_reference_file_hash,
        "reference_path": report.request.reference_path,
        "reference_stream_id": report.request.reference_stream_id,
        "reference_file_hash": report.request.reference_file_hash,
        "reference_provenance": report.request.reference_provenance,
        "reference_frame_count": report.reference_frame_count,
        "evaluated_frame_count": report.frame_count,
        "future_offsets": list(report.request.future_offsets),
        "corruption_id": report.request.corruption_protocol.corruption_id,
        "corruption_family": report.request.corruption_protocol.family,
        "corruption_seed": report.request.corruption_protocol.seed,
        "corruption_parameters": dict(report.request.corruption_protocol.parameters),
        "corruption_temporal_mode": report.request.corruption_protocol.temporal_mode,
        "corruption_protocol_hash": report.request.corruption_protocol.protocol_hash,
        "femr_action_count": report.femr_action_count,
        "accumulated_failure_count": report.accumulated_failure_count,
        "per_frame_femr_action_used": list(report.per_frame_femr_action_used),
        "per_frame_intent_q29_error": list(report.per_frame_intent_q29_error),
        "per_frame_physics_success": list(report.per_frame_physics_success),
        "per_frame_fall": list(report.per_frame_fall),
        "per_frame_zmp_margin": list(report.per_frame_zmp_margin),
        "per_frame_contact_consistency": list(report.per_frame_contact_consistency),
        "per_frame_policy_actions": report.per_frame_policy_actions,
        "expected_contact_steps": report.expected_contact_steps,
        "actual_contact_steps": report.actual_contact_steps,
        "contact_mismatch_steps": report.contact_mismatch_steps,
        "phase_zmp_applicable_steps": report.phase_zmp_applicable_steps,
        "phase_zmp_violation_steps": report.phase_zmp_violation_steps,
        "phase_zmp_recovery_steps": report.phase_zmp_recovery_steps,
        "survival_steps": report.survival_steps,
        "evaluation_only_sustained_lean": {
            "lateral_roll_rad": report.lateral_roll_rad_steps,
            "cumulative_mean_rad": report.lateral_roll_cumulative_mean_rad_steps,
        },
        "unplanned_contact_steps": report.unplanned_contact_steps,
        "summary": {
            "mean_intent_q29_error": report.mean_intent_q29_error,
            "contact_preservation_fraction": report.contact_preservation_fraction,
            "phase_zmp_applicable_count": report.phase_zmp_applicable_count,
            "phase_zmp_violation_count": report.phase_zmp_violation_count,
            "survival_fraction": report.survival_fraction,
            "max_abs_cumulative_lateral_roll_rad": report.max_abs_cumulative_lateral_roll_rad,
            "unplanned_contact_event_count": report.unplanned_contact_event_count,
        },
        "return_feedback": False,
        "priority_feedback": False,
        "ppo_feedback": False,
        "sampler_feedback": False,
        "optimizer_feedback": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError("v015 composition refuses an existing report or partial report path")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_v015_deployment_npz_arrays(arrays: Mapping[str, np.ndarray]) -> tuple[int, int, int, float]:
    joint_pos = arrays["joint_pos"]
    joint_vel = arrays["joint_vel"]
    if joint_pos.ndim != 2 or tuple(joint_vel.shape) != tuple(joint_pos.shape) or int(joint_pos.shape[1]) != 29:
        raise ValueError(
            "v015 deployment q29 arrays must both have shape [T,29], got "
            f"joint_pos={tuple(joint_pos.shape)} joint_vel={tuple(joint_vel.shape)}"
        )
    frame_count = int(joint_pos.shape[0])
    body_pos = arrays["body_pos_w"]
    if body_pos.ndim != 3 or int(body_pos.shape[0]) != frame_count or int(body_pos.shape[2]) != 3:
        raise ValueError("v015 deployment body_pos_w must have shape [T,J,3]")
    body_count = int(body_pos.shape[1])
    expected_shapes = {
        "body_quat_w": (frame_count, body_count, 4),
        "body_lin_vel_w": (frame_count, body_count, 3),
        "body_ang_vel_w": (frame_count, body_count, 3),
    }
    for name, expected in expected_shapes.items():
        if tuple(arrays[name].shape) != expected:
            raise ValueError(f"v015 deployment {name} must have shape {expected}, got {tuple(arrays[name].shape)}")
    for name, value in arrays.items():
        if not np.issubdtype(value.dtype, np.number) or not bool(np.isfinite(value).all()):
            raise ValueError(f"v015 deployment array {name} must be finite numeric data")
    fps_values = arrays["fps"].reshape(-1)
    if int(fps_values.size) != 1:
        raise ValueError("v015 deployment fps must be scalar")
    fps = float(fps_values[0])
    if frame_count <= 0 or body_count <= 0 or not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("v015 deployment reference requires positive frames, bodies, and fps")
    return frame_count, 29, body_count, fps


def _validate_v015_corruption_parameters(
    parameters: tuple[tuple[str, str | int | float | bool], ...],
) -> None:
    if not isinstance(parameters, tuple) or tuple(sorted(parameters, key=lambda item: item[0])) != parameters:
        raise ValueError("v015 corruption parameters must be a canonical sorted tuple")
    names = tuple(name for name, _ in parameters)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("v015 corruption parameter names must be unique and nonempty")
    for name, value in parameters:
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"v015 corruption parameter {name} must be scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"v015 corruption parameter {name} must be finite")


def _frontres_v015_corruption_protocol_hash(
    *,
    corruption_id: str,
    family: str,
    seed: int,
    parameters: tuple[tuple[str, str | int | float | bool], ...],
    temporal_mode: str,
) -> str:
    payload = {
        "corruption_id": corruption_id,
        "family": family,
        "parameters": parameters,
        "seed": int(seed),
        "temporal_mode": temporal_mode,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrontRESSegmentSequenceEvalItem:
    segment_id: int
    motion_id: str
    reset_frame: int
    preroll_steps: int
    eval_start_frame: int
    eval_rollout_steps: int
    segment_horizon_k: int

    @property
    def eval_end_frame(self) -> int:
        return self.eval_start_frame + self.eval_rollout_steps


@dataclass(frozen=True)
class FrontRESSegmentSequenceEvalPlan:
    items: tuple[FrontRESSegmentSequenceEvalItem, ...]
    requested_sequences: int
    available_envs: int
    paired_envs_per_sequence: int
    chunk_capacity: int
    max_preroll_steps: int | None = None

    @property
    def sequence_count(self) -> int:
        return len(self.items)

    @property
    def chunk_count(self) -> int:
        return (self.sequence_count + self.chunk_capacity - 1) // self.chunk_capacity

    @property
    def motion_ids(self) -> tuple[str, ...]:
        return tuple(item.motion_id for item in self.items)


def build_frontres_sequence_eval_plan(
    specs: Sequence[Any],
    *,
    requested_sequences: int = 10,
    available_envs: int = 0,
    paired_envs_per_sequence: int = 4,
    eval_rollout_steps: int | None = None,
    max_preroll_steps: int | None = None,
) -> FrontRESSegmentSequenceEvalPlan:
    if requested_sequences <= 0:
        raise ValueError("requested_sequences must be positive")
    if paired_envs_per_sequence <= 0:
        raise ValueError("paired_envs_per_sequence must be positive")

    preroll_cap = None if max_preroll_steps is None or int(max_preroll_steps) <= 0 else int(max_preroll_steps)

    # FRS3-EVAL-004: choose unique motion sequences and derive frame0->segment eval windows.
    items: list[FrontRESSegmentSequenceEvalItem] = []
    seen: set[str] = set()
    for index, spec in enumerate(specs):
        motion_id = str(getattr(spec, "motion_id", ""))
        if not motion_id:
            raise ValueError("sequence eval specs must expose motion_id")
        start_frame = _required_nonnegative_int(spec, "start_frame")
        if preroll_cap is not None and start_frame > preroll_cap:
            continue
        if motion_id in seen:
            continue
        seen.add(motion_id)
        horizon_k = _positive_int(getattr(spec, "horizon_k", 1), "horizon_k")
        rollout_steps = _positive_int(eval_rollout_steps if eval_rollout_steps is not None else horizon_k, "eval_rollout_steps")
        items.append(
            FrontRESSegmentSequenceEvalItem(
                segment_id=int(getattr(spec, "segment_id", index)),
                motion_id=motion_id,
                reset_frame=0,
                preroll_steps=start_frame,
                eval_start_frame=start_frame,
                eval_rollout_steps=rollout_steps,
                segment_horizon_k=horizon_k,
            )
        )
        if len(items) >= requested_sequences:
            break

    if len(items) < requested_sequences:
        cap_note = "" if preroll_cap is None else f" with max_preroll_steps<={preroll_cap}"
        raise ValueError(
            f"sequence eval requires {requested_sequences} unique motion ids{cap_note}, got {len(items)}"
        )

    envs = max(0, int(available_envs))
    chunk_capacity = len(items) if envs <= 0 else max(1, envs // int(paired_envs_per_sequence))
    return FrontRESSegmentSequenceEvalPlan(
        items=tuple(items),
        requested_sequences=int(requested_sequences),
        available_envs=envs,
        paired_envs_per_sequence=int(paired_envs_per_sequence),
        chunk_capacity=chunk_capacity,
        max_preroll_steps=preroll_cap,
    )


def segment_ids_for_sequence_eval_item(
    item: FrontRESSegmentSequenceEvalItem,
    *,
    env_count: int,
) -> tuple[int, ...]:
    # FRS3-EVAL-005: repeat one segment across the full B1 role layout.
    count = _positive_int(env_count, "env_count")
    return tuple(int(item.segment_id) for _ in range(count))


def build_frontres_sequence_eval_reset_batch(
    batch: Any,
    item: FrontRESSegmentSequenceEvalItem,
) -> Any:
    # FRS3-EVAL-006: rewrite reset specs to motion frame 0 before preroll.
    specs = tuple(getattr(batch, "specs", ()) or ())
    if not specs:
        raise ValueError("sequence eval reset batch requires specs")
    reset_specs = tuple(_replace_spec_start_frame(spec, item.reset_frame) for spec in specs)
    if is_dataclass(batch):
        reset_batch = replace(batch, specs=reset_specs)
        _copy_sequence_eval_dynamic_attrs(batch, reset_batch)
        return reset_batch
    values = dict(vars(batch))
    values["specs"] = reset_specs
    return SimpleNamespace(**values)


def _copy_sequence_eval_dynamic_attrs(src: Any, dst: Any) -> None:
    for name in (
        "stage3_index_perturbation_family",
        "stage3_index_perturbation_strength",
        "stage3_index_perturbation_plan",
    ):
        if hasattr(src, name):
            object.__setattr__(dst, name, getattr(src, name))


def _replace_spec_start_frame(spec: Any, start_frame: int) -> Any:
    if is_dataclass(spec):
        changes: dict[str, Any] = {"start_frame": int(start_frame)}
        if hasattr(spec, "start_time"):
            changes["start_time"] = 0.0
        if hasattr(spec, "phase"):
            changes["phase"] = 0.0
        return replace(spec, **changes)
    values = dict(vars(spec))
    values["start_frame"] = int(start_frame)
    if "start_time" in values:
        values["start_time"] = 0.0
    if "phase" in values:
        values["phase"] = 0.0
    return SimpleNamespace(**values)


def _required_nonnegative_int(spec: Any, name: str) -> int:
    value = getattr(spec, name, None)
    if value is None:
        raise ValueError(f"sequence eval spec requires {name}")
    value_int = int(value)
    if value_int < 0:
        raise ValueError(f"{name} must be non-negative")
    return value_int


def _positive_int(value: Any, name: str) -> int:
    value_int = int(value)
    if value_int <= 0:
        raise ValueError(f"{name} must be positive")
    return value_int
