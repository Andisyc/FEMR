"""Immutable deployment carrier and deterministic NPZ materialization."""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

FRONTRES_V015_DEPLOYMENT_COMPOSITION_KIND = "deployment_composition_v015"


_V015_DEPLOYMENT_REFERENCE_PROVENANCE = "deployment_reference_stream"


_V015_DEPLOYMENT_CARRIER_PROVENANCE = "materialized_deployment_carrier"


_V015_PERSISTENT_TEMPORAL_MODE = "persistent_full_sequence"


_V015_SUPPORTED_CORRUPTION_FAMILIES = frozenset(("planar", "yaw", "global_z", "local_rp"))


FRONTRES_V015_REQUIRED_NPZ_ARRAYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


class FrontRESV015NoTrainingFeedback:
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
        # B1: 校验 corruption family, temporal mode, parameters 与 protocol hash.
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
    # B1: 规范化 corruption fields 并计算 identity, 产出 immutable protocol.
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

    def validate(self) -> None:
        # B1: 校验 source/reference, offsets, corruption 与 report boundary.
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


@dataclass(frozen=True)
class FrontRESV015DeploymentCompositionRequest(FrontRESV015NoTrainingFeedback):
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
    evaluation_kind: str = FRONTRES_V015_DEPLOYMENT_COMPOSITION_KIND

    def validate(self) -> None:
        # B1: 校验 materialized source/reference identities 与 dense q29 contract.
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
        if self.evaluation_kind != FRONTRES_V015_DEPLOYMENT_COMPOSITION_KIND:
            raise ValueError("v015 deployment request has an invalid evaluation kind")
        if self.frame_count <= max(self.future_offsets, default=0):
            raise ValueError("v015 deployment reference is too short for its future_offsets")
        if self.joint_dof != 29 or self.body_count <= 0 or not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("v015 deployment reference requires q29, nonempty bodies, and positive finite fps")
        if tuple(sorted(set(self.future_offsets))) != self.future_offsets or any(value <= 0 for value in self.future_offsets):
            raise ValueError("v015 deployment request has invalid future_offsets")
        self.corruption_protocol.validate()


@dataclass(frozen=True)
class FrontRESV015DeploymentCarrier(FrontRESV015NoTrainingFeedback):
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
        # B1: 校验 carrier files, hashes, shapes 与 q29 provenance.
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
        # B1: 绑定 validated config, 初始化 single-materialization lifecycle state.
        self._source_path = str(source_path)
        self._output_path = str(output_path)
        self._corruption_protocol = corruption_protocol
        self._carrier: FrontRESV015DeploymentCarrier | None = None
        self._state = "ready"

    @property
    def state(self) -> str:
        return self._state

    def materialize(self) -> FrontRESV015DeploymentCarrier:
        # B1: 拒绝重复 materialization, 调用唯一 carrier owner 并封存结果.
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
class FrontRESV015DeploymentCompositionRunConfig:
    """Formal S2B entry config for one pre-materialized deployment stream."""

    request_config: FrontRESV015DeploymentCompositionConfig
    report_path: str

    def validate(self) -> None:
        # B1: 校验 request config, checkpoints, row count 与 frozen inference flags.
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
    # B1: 读取 source/reference arrays 与 hashes, 产出 strict composition request.
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
            missing = tuple(name for name in FRONTRES_V015_REQUIRED_NPZ_ARRAYS if name not in data.files)
            if missing:
                raise ValueError(f"v015 deployment source .npz is missing required arrays: {missing}")
            source_arrays = {name: np.asarray(data[name]) for name in FRONTRES_V015_REQUIRED_NPZ_ARRAYS}
        with np.load(reference_path, allow_pickle=False) as data:
            missing = tuple(name for name in FRONTRES_V015_REQUIRED_NPZ_ARRAYS if name not in data.files)
            if missing:
                raise ValueError(f"v015 deployment .npz is missing required arrays: {missing}")
            arrays = {name: np.asarray(data[name]) for name in FRONTRES_V015_REQUIRED_NPZ_ARRAYS}
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

    # B1: 验证 protocol 与 source/output 路径, 产出唯一可写的 materialization 边界.
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

    # B2: 读取 source 并只修改 root/global arrays, 产出 q29/dq29 不变的 deterministic carrier.
    source_hash_before = _sha256_file(source)
    arrays = load_frontres_v015_reference_arrays(source)
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
    # B3: 封存 carrier receipt 并在校验失败时删除 partial output.
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


def load_frontres_v015_reference_arrays(path: Path) -> dict[str, np.ndarray]:
    # B1: 读取 required arrays 并复制为 owned buffers, 产出 validated NPZ mapping.
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = tuple(name for name in FRONTRES_V015_REQUIRED_NPZ_ARRAYS if name not in data.files)
            if missing:
                raise ValueError(f"v015 deployment .npz is missing required arrays: {missing}")
            return {name: np.asarray(data[name]).copy() for name in FRONTRES_V015_REQUIRED_NPZ_ARRAYS}
    except (OSError, TypeError) as exc:
        raise ValueError(f"v015 deployment reference cannot be read as a safe .npz: {path}") from exc


def _frontres_v015_sample_persistent_delta(
    protocol: FrontRESV015PersistentCorruptionProtocol,
) -> tuple[float, float, float, float, float, float]:
    # B1: 从 fixed seed 与 protocol ranges 采样一次 full-sequence Delta SE(3).
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
    # B1: 解析并校验 protocol root body index, 产出唯一 corruption target row.
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
    # B1: 读取 optional scalar 并拒绝负数/非有限值, 产出 corruption range.
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
    # B1: 复制 source arrays 并对 root body 全帧施加同一 task-space delta.
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
    # B1: 将 XYZ Euler half-angles 转为 normalized xyzw quaternion.
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
    # B1: 计算 xyzw Hamilton product, 产出 normalized composed rotation.
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
    # B1: 将 normalized xyzw quaternion 转为 3x3 rotation matrix.
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
    # B1: 按稳定 array order/metadata 编码 NPZ, 通过 temporary replace 原子提交.
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, mode="x", compression=zipfile.ZIP_STORED) as archive:
            for name in FRONTRES_V015_REQUIRED_NPZ_ARRAYS:
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
    # B1: 哈希 q29/dq29 shape, dtype 与 bytes, 产出 deployment Intent identity.
    digest = hashlib.sha256()
    for name, value in (("joint_pos", joint_pos), ("joint_vel", joint_vel)):
        array = np.asarray(value)
        digest.update(f"{name}:{array.dtype}:{tuple(array.shape)}:".encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _frontres_v015_npz_intent_hash(path: Path) -> str:
    arrays = load_frontres_v015_reference_arrays(path)
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
    # B1: 哈希 source/protocol/carrier/delta/layout fields, 产出 materialization identity.
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


def _validate_v015_deployment_npz_arrays(arrays: Mapping[str, np.ndarray]) -> tuple[int, int, int, float]:
    # B1: 校验 required NPZ arrays 的 frame/body/q29 shape 与 finite semantics.
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
    # B1: 校验 canonical parameter names/types/ranges, 拒绝未知 corruption freedom.
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
    # B1: 稳定编码 corruption identity fields, 产出 immutable protocol hash.
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
    # B1: 分块读取 artifact bytes, 产出文件内容 SHA-256 identity.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
