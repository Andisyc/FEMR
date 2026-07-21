"""Sequence-evaluation schemas, v015 composition, and legacy planning helpers.

Status: E-FI-28--E-FI-30 connect the explicit NPZ/protocol identity through
the command-owned deployment carrier, per-frame FEMR, frozen GMT, and the
immutable no-feedback report. The older plan/reset helpers below remain legacy
v002 and are rejected by the v015 runner boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch


_V015_DEPLOYMENT_COMPOSITION_KIND = "deployment_composition_v015"
_V015_DEPLOYMENT_REFERENCE_PROVENANCE = "deployment_reference_stream"
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
    reference_path: str
    future_offsets: tuple[int, ...]
    corruption_protocol: FrontRESV015PersistentCorruptionProtocol
    legacy_modes: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.enabled is not True:
            raise ValueError("v015 deployment composition config must be explicitly enabled")
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
class FrontRESV015DeploymentCompositionReport(_FrontRESV015NoTrainingFeedback):
    """Per-frame deployment-only metrics, separate from local K Gain and training."""

    request: FrontRESV015DeploymentCompositionRequest
    per_frame_femr_action_used: tuple[bool, ...]
    per_frame_intent_q29_error: tuple[float, ...]
    per_frame_physics_success: tuple[bool, ...]
    per_frame_fall: tuple[bool, ...]
    per_frame_zmp_margin: tuple[float, ...]
    per_frame_contact_consistency: tuple[float, ...]
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
        if any(type(value) is not bool for value in self.per_frame_femr_action_used):
            raise ValueError("v015 per-frame FEMR action flags must be bool")
        if any(type(value) is not bool for value in self.per_frame_physics_success + self.per_frame_fall):
            raise ValueError("v015 per-frame physics success/fall flags must be bool")
        if any(success and fall for success, fall in zip(self.per_frame_physics_success, self.per_frame_fall, strict=True)):
            raise ValueError("v015 composition frame cannot report physics success and fall together")
        numeric_rows = (
            self.per_frame_intent_q29_error,
            self.per_frame_zmp_margin,
            self.per_frame_contact_consistency,
        )
        if any(not math.isfinite(float(value)) for values in numeric_rows for value in values):
            raise ValueError("v015 deployment report metrics must be finite")
        if any(float(value) < 0.0 for value in self.per_frame_intent_q29_error):
            raise ValueError("v015 per-frame q29 intent error must be nonnegative")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.per_frame_contact_consistency):
            raise ValueError("v015 per-frame contact consistency must be in [0,1]")


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
    reference_path = Path(config.reference_path).expanduser().resolve(strict=True)
    if not reference_path.is_file():
        raise ValueError(f"v015 deployment reference is not a file: {reference_path}")

    try:
        with np.load(reference_path, allow_pickle=False) as data:
            missing = tuple(name for name in _V015_REQUIRED_NPZ_ARRAYS if name not in data.files)
            if missing:
                raise ValueError(f"v015 deployment .npz is missing required arrays: {missing}")
            arrays = {name: np.asarray(data[name]) for name in _V015_REQUIRED_NPZ_ARRAYS}
    except (OSError, TypeError) as exc:
        raise ValueError(f"v015 deployment reference cannot be read as a safe .npz: {reference_path}") from exc

    frame_count, joint_dof, body_count, fps = _validate_v015_deployment_npz_arrays(arrays)
    file_hash = _sha256_file(reference_path)
    request = FrontRESV015DeploymentCompositionRequest(
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
    zmp_margin: list[float] = []
    contact_consistency: list[float] = []
    evaluated_frames = request.frame_count - max(request.future_offsets)
    set_sequence(request)
    try:
        with torch.inference_mode():
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
                physics_success.append(bool(frame_metrics["physics_success"].all().item()))
                fall.append(bool(frame_metrics["fall"].any().item()))
                zmp_margin.append(float(frame_metrics["zmp_margin"].mean().item()))
                contact_consistency.append(float(frame_metrics["contact_consistency"].mean().item()))
                if frame_index + 1 < evaluated_frames:
                    advance_sequence()
    finally:
        clear_sequence()

    training_after = _frontres_v015_training_state_fingerprint(runner)
    if training_after != training_before:
        changed = tuple(name for name in training_before if training_before[name] != training_after.get(name))
        raise RuntimeError(f"v015 composition mutated forbidden training state: {changed}")
    report = FrontRESV015DeploymentCompositionReport(
        request=request,
        per_frame_femr_action_used=tuple(femr_used),
        per_frame_intent_q29_error=tuple(intent_error),
        per_frame_physics_success=tuple(physics_success),
        per_frame_fall=tuple(fall),
        per_frame_zmp_margin=tuple(zmp_margin),
        per_frame_contact_consistency=tuple(contact_consistency),
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


def _frontres_v015_deployment_frame_metrics(
    runner: Any,
    command: Any,
    *,
    frame_index: int,
    dones: torch.Tensor,
    infos: Any,
) -> dict[str, torch.Tensor]:
    provider = getattr(runner, "_frontres_v015_deployment_metric_provider", None)
    if callable(provider):
        values = provider(frame_index=frame_index, dones=dones, infos=infos, command=command)
    else:
        from rsl_rl.frontres.frontres_balance import _frontres_branch_balance_margin

        time_outs = infos.get("time_outs") if isinstance(infos, Mapping) else None
        time_outs = (
            torch.as_tensor(time_outs, device=dones.device, dtype=torch.bool)
            if time_outs is not None
            else torch.zeros_like(dones)
        )
        fall = dones & ~time_outs
        values = {
            "physics_success": ~fall,
            "fall": fall,
            "zmp_margin": _frontres_branch_balance_margin(
                runner, command, start=0, count=int(dones.numel()), device=runner.device
            ),
            "contact_consistency": _frontres_v015_contact_consistency(runner, command),
        }
    required = {"physics_success", "fall", "zmp_margin", "contact_consistency"}
    if not isinstance(values, Mapping) or set(values) != required:
        raise RuntimeError("v015 composition metric provider returned an invalid schema")
    output: dict[str, torch.Tensor] = {}
    for name in required:
        value = torch.as_tensor(values[name], device=runner.device).flatten()
        if int(value.numel()) != int(dones.numel()):
            raise RuntimeError(f"v015 composition metric {name} must be row-aligned [B]")
        if name in {"physics_success", "fall"}:
            value = value.bool()
        else:
            value = value.float()
            if not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(f"v015 composition metric {name} must be finite")
        output[name] = value.detach().clone()
    if bool((output["physics_success"] & output["fall"]).any()):
        raise RuntimeError("v015 composition cannot report success and fall for the same row")
    if bool(((output["contact_consistency"] < 0.0) | (output["contact_consistency"] > 1.0)).any()):
        raise RuntimeError("v015 composition contact consistency must remain in [0,1]")
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
    }
    return {name: _frontres_v015_object_state_hash(value) for name, value in objects.items()}


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
