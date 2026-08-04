"""Read-only identity and inspection for strict v015 quality checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Literal, Mapping

import torch

from rsl_rl.modules.frontres_observation_layout import (
    FRONTRES_FUTURE_INTENT_LAYOUT_VERSION,
)
from rsl_rl.frontres.frontres_segment_warmup import (
    FRONTRES_V011_MAX_ABSOLUTE_ITERATION,
    FRONTRES_V011_REVIEW_BOUNDARIES,
    FRONTRES_V011_SELECTED_SEGMENT_COUNT,
    frontres_k_stage_schedule_tuple,
    require_frontres_v013_campaign_schedule as _require_frontres_v013_campaign_schedule,
    resolve_frontres_k_stage_identity,
)

_V015_CHECKPOINT_IDENTITY_KEY = "frontres_v015_checkpoint_identity"
_V015_CHECKPOINT_FORMAT = "frontres-v017-checkpoint-v9"
_V015_GROUPED_CANDIDATE_LAYOUT = "frontres-v015-local-scenario-v1"
_V015_HSL_CHECKPOINT_IDENTITY_KEY = "frontres_v015_hsl_checkpoint_identity"
_V015_HSL_CHECKPOINT_FORMAT = "frontres-v017-hsl-proposal-v2"
_V015_HSL_PREFIX_NORM_KEY = "frontres_prefix_norm_state_dict"
_V015_HSL_TOP_LEVEL_KEYS = {
    _V015_HSL_CHECKPOINT_IDENTITY_KEY,
    "model_state_dict",
    _V015_HSL_PREFIX_NORM_KEY,
}
_EMPIRICAL_NORMALIZER_STATE_KEYS = {"_mean", "_var", "_std", "count"}


def _validate_safe_checkpoint_value(value: object, *, path: str = "payload") -> None:
    """Reject object-bearing checkpoint graphs after restricted deserialization."""

    if value is None or isinstance(value, (str, bytes, bool, int, float, torch.Tensor)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise RuntimeError(f"FrontRES checkpoint {path} has unsupported mapping key {type(key).__name__}")
            _validate_safe_checkpoint_value(item, path=f"{path}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_safe_checkpoint_value(item, path=f"{path}[{index}]")
        return
    raise RuntimeError(f"FrontRES checkpoint {path} has unsupported value type {type(value).__name__}")


def load_frontres_checkpoint_mapping(
    checkpoint_path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    """Safely deserialize one FrontRES checkpoint before any runner mutation."""

    path = Path(checkpoint_path).expanduser().resolve()
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"FrontRES checkpoint restricted load failed: {path}") from exc
    if not isinstance(checkpoint, Mapping) or any(not isinstance(key, str) for key in checkpoint):
        raise RuntimeError("FrontRES checkpoint payload must be a string-keyed mapping")
    _validate_safe_checkpoint_value(checkpoint)
    return checkpoint


def require_frontres_v013_campaign_schedule(schedule: object) -> None:
    """Normalize schedule-identity failures to the checkpoint API error type."""

    try:
        _require_frontres_v013_campaign_schedule(schedule)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _v015_committed_transaction_receipt(state: Mapping[str, Any]) -> dict[str, Any]:
    """校验可跨越 v015 checkpoint boundary 的 metadata-only receipt."""

    if str(state.get("state", "")) != "committed":
        raise RuntimeError("v015 checkpoint transaction must be idle or committed")
    receipt = state.get("receipt")
    if not isinstance(receipt, Mapping):
        raise RuntimeError("v015 committed checkpoint transaction requires a receipt")
    required = (
        "method_contract_id",
        "gain_contract_id",
        "optimization_contract_id",
        "training_contract_id",
        "scalar_target_id",
        "physics_schema_id",
        "grouped_schema_id",
        "transaction_id",
        "policy_snapshot_id",
        "plan_identity_hash",
        "scenario_identity_hash",
        "expected_policy_row_count",
        "collected_policy_attempt_count",
        "valid_policy_row_count",
        "optimizer_step_before",
        "optimizer_step_after",
        "optimizer_step_delta",
        "curriculum_fingerprint",
        "k_stage_index",
        "active_k",
        "active_m",
        "selected_segment_count",
        "policy_row_count",
        "role_row_count",
        "k_stage_iteration",
        "training_iteration",
        "dr_stage_fingerprint",
        "dr_progress",
        "d_cap",
    )
    if set(receipt) != set(required):
        raise RuntimeError("v015 committed checkpoint receipt has an unexpected field set")
    result = {name: receipt[name] for name in required}
    for name in (
        "method_contract_id",
        "gain_contract_id",
        "optimization_contract_id",
        "training_contract_id",
        "scalar_target_id",
        "physics_schema_id",
        "grouped_schema_id",
        "transaction_id",
        "policy_snapshot_id",
        "plan_identity_hash",
        "scenario_identity_hash",
        "curriculum_fingerprint",
        "dr_stage_fingerprint",
    ):
        if not isinstance(result[name], str) or not result[name]:
            raise RuntimeError(f"v015 committed checkpoint receipt has invalid {name}")
    expected_identity = {
        "method_contract_id": "FRS-METHOD-v017",
        "gain_contract_id": "FRS-GAIN-v007",
        "optimization_contract_id": "FRS-PPO-v005",
        "training_contract_id": "FRS-TRAIN-v014",
        "scalar_target_id": "clean-anchored-recovery-aware-gain-v1",
        "physics_schema_id": "clean-anchored-contact-zmp-survival-v1",
        "grouped_schema_id": "grouped-all-attempt-scalar-v1",
    }
    if any(result[name] != value for name, value in expected_identity.items()):
        raise RuntimeError("v015 committed checkpoint receipt has legacy contract identity")
    for name in (
        "expected_policy_row_count",
        "collected_policy_attempt_count",
        "valid_policy_row_count",
        "optimizer_step_before",
        "optimizer_step_after",
        "optimizer_step_delta",
        "k_stage_index",
        "active_k",
        "active_m",
        "selected_segment_count",
        "policy_row_count",
        "role_row_count",
        "k_stage_iteration",
        "training_iteration",
    ):
        result[name] = int(result[name])
    result["dr_progress"] = float(result["dr_progress"])
    result["d_cap"] = float(result["d_cap"])
    if (
        result["expected_policy_row_count"] <= 0
        or result["collected_policy_attempt_count"] != result["expected_policy_row_count"]
        or result["valid_policy_row_count"] <= 0
        or result["valid_policy_row_count"] > result["collected_policy_attempt_count"]
        or result["optimizer_step_delta"] != 1
        or result["optimizer_step_after"] != result["optimizer_step_before"] + 1
        or len(result["curriculum_fingerprint"]) != 64
        or result["k_stage_index"] < 0
        or result["active_k"] <= 0
        or result["active_m"] < 2
        or result["selected_segment_count"] != FRONTRES_V011_SELECTED_SEGMENT_COUNT
        or result["policy_row_count"] != result["selected_segment_count"] * result["active_m"]
        or result["role_row_count"] != 2 * result["policy_row_count"]
        or result["k_stage_iteration"] < 0
        or result["training_iteration"] < 0
        or len(result["dr_stage_fingerprint"]) != 64
        or not 0.0 <= result["dr_progress"] <= 1.0
        or not 0.0 < result["d_cap"] <= 2.381
    ):
        raise RuntimeError("v015 committed checkpoint receipt is not an exact-one completed transaction")
    return result


def _v015_tensor_fingerprint(*values: torch.Tensor) -> str:
    """返回 detached checkpoint tensor 的值敏感 identity."""

    digest = hashlib.sha256()
    for value in values:
        if not isinstance(value, torch.Tensor):
            raise TypeError("v015 checkpoint normalizer identity requires tensors")
        cpu = value.detach().to(device="cpu").contiguous()
        digest.update(str(tuple(cpu.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(cpu.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(cpu.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _v015_state_dict_fingerprint(state: Mapping[str, torch.Tensor], *, label: str) -> str:
    """Return one key/order/value-sensitive fingerprint for a tensor state dict."""

    if not isinstance(state, Mapping) or not state:
        raise RuntimeError(f"{label} must be a nonempty tensor state dict")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise RuntimeError(f"{label} must contain only named tensors")
        cpu = value.detach().to(device="cpu").contiguous()
        if torch.is_floating_point(cpu) and not bool(torch.isfinite(cpu).all().item()):
            raise RuntimeError(f"{label} contains non-finite tensor {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(cpu.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(cpu.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(cpu.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _v015_clone_tensor_state(state: Mapping[str, torch.Tensor], *, label: str) -> dict[str, torch.Tensor]:
    _v015_state_dict_fingerprint(state, label=label)
    return {name: value.detach().clone() for name, value in state.items()}


def _validate_v015_normalizer_state(
    state: Mapping[str, torch.Tensor],
    *,
    dim: int,
    label: str,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping) or set(state) != _EMPIRICAL_NORMALIZER_STATE_KEYS:
        raise RuntimeError(f"{label} has an unexpected state schema")
    for name in ("_mean", "_var", "_std"):
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (1, int(dim))
            or not torch.is_floating_point(value)
            or not bool(torch.isfinite(value).all().item())
        ):
            raise RuntimeError(f"{label} {name} must be finite [1,{int(dim)}]")
    if bool((state["_var"] < 0).any().item()) or bool((state["_std"] <= 0).any().item()):
        raise RuntimeError(f"{label} variance/std state is invalid")
    count = state["count"]
    if not isinstance(count, torch.Tensor) or count.numel() != 1 or int(count.item()) < 0:
        raise RuntimeError(f"{label} count must be a nonnegative scalar")
    return state


def _v015_file_sha256(path: str | os.PathLike[str]) -> str:
    """Hash the exact frozen GMT artifact bound to a FrontRES checkpoint."""

    artifact = Path(path).expanduser().resolve()
    if not artifact.is_file():
        raise RuntimeError(f"FrontRES requires an existing frozen GMT checkpoint artifact: {artifact}")
    digest = hashlib.sha256()
    with artifact.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrontRESActiveQualityCheckpointIdentity:
    """Read-only identity receipt for one strict policy-quality checkpoint."""

    route: Literal["hsl", "policy"]
    format: str
    file_sha256: str
    method_contract_id: str
    training_contract_id: str
    gain_contract_id: str | None
    ppo_contract_id: str | None
    future_intent_layout: tuple[tuple[str, object], ...]
    action_kind: str
    action_dim: int
    action_semantics: str
    normalizer_key: str
    actor_fingerprint: str
    distribution_key: str
    distribution_fingerprint: str
    normalizer_fingerprint: str


def _v015_quality_expected_layout() -> dict[str, object]:
    return {
        "layout_version": FRONTRES_FUTURE_INTENT_LAYOUT_VERSION,
        "future_offsets": (1, 2),
        "intent_dim": 29,
        "actor_tail_dim": 58,
        "environment_obs_dim": 870,
        "current_frontres_prefix_dim": 100,
        "actor_dim": 928,
        "prefix_dim": 158,
        "gmt_dim": 770,
    }


def _v015_quality_require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError(f"{label} must be a lowercase SHA-256 fingerprint")
    return value


def _v015_quality_model_identity(
    checkpoint: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, str, str]:
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise RuntimeError(f"{label} requires model_state_dict")
    actor_state = model_state.get("residual_actor")
    if not isinstance(actor_state, Mapping) or not actor_state:
        raise RuntimeError(f"{label} requires residual_actor state")
    actor_fingerprint = _v015_state_dict_fingerprint(actor_state, label=f"{label} residual actor")
    distribution_keys = tuple(name for name in ("std", "log_std") if name in model_state)
    if len(distribution_keys) != 1:
        raise RuntimeError(f"{label} requires exactly one std or log_std tensor")
    distribution_key = distribution_keys[0]
    distribution = model_state[distribution_key]
    if (
        not isinstance(distribution, torch.Tensor)
        or tuple(distribution.shape) != (6,)
        or not bool(torch.isfinite(distribution).all().item())
    ):
        raise RuntimeError(f"{label} requires finite full-6D distribution identity")
    if not any(
        isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == 6
        for value in actor_state.values()
    ):
        raise RuntimeError(f"{label} residual actor has no full-6D output identity")
    return actor_fingerprint, distribution_key, _v015_tensor_fingerprint(distribution)


def _inspect_frontres_v015_hsl_quality_payload(
    checkpoint: Mapping[str, Any],
    *,
    file_sha256: str,
) -> FrontRESActiveQualityCheckpointIdentity:
    if set(checkpoint) != _V015_HSL_TOP_LEVEL_KEYS:
        raise RuntimeError("quality HSL requires the exact proposal-only HSL payload")
    identity = checkpoint.get(_V015_HSL_CHECKPOINT_IDENTITY_KEY)
    required_identity = {
        "format",
        "method_contract_id",
        "training_contract_id",
        "objective",
        "future_intent_layout",
        "action",
        "gmt",
        "payload",
    }
    if not isinstance(identity, Mapping) or set(identity) != required_identity:
        raise RuntimeError("quality HSL identity is missing, legacy, or malformed")
    if (
        identity["format"] != _V015_HSL_CHECKPOINT_FORMAT
        or identity["method_contract_id"] != "FRS-METHOD-v017"
        or identity["training_contract_id"] != "FRS-TRAIN-v014"
        or identity["objective"] != "proposal_only_current_antidr_delta_se3"
        or identity["future_intent_layout"] != _v015_quality_expected_layout()
        or identity["action"]
        != {"kind": "delta_se3", "dim": 6, "semantics": "direct-world-full6-v1"}
    ):
        raise RuntimeError("quality HSL has an incompatible v015 layout or action identity")
    gmt = identity["gmt"]
    if (
        not isinstance(gmt, Mapping)
        or set(gmt) != {"checkpoint_sha256", "normalizer_dim", "normalizer_fingerprint"}
        or int(gmt.get("normalizer_dim", -1)) != 770
    ):
        raise RuntimeError("quality HSL GMT identity is malformed")
    _v015_quality_require_sha256(gmt["checkpoint_sha256"], label="quality HSL GMT checkpoint")
    _v015_quality_require_sha256(gmt["normalizer_fingerprint"], label="quality HSL GMT normalizer")
    actor_fingerprint, distribution_key, distribution_fingerprint = _v015_quality_model_identity(
        checkpoint, label="quality HSL"
    )
    prefix_state = checkpoint.get(_V015_HSL_PREFIX_NORM_KEY)
    _validate_v015_normalizer_state(prefix_state, dim=158, label="quality HSL prefix normalizer")
    prefix_fingerprint = _v015_state_dict_fingerprint(prefix_state, label="quality HSL prefix normalizer")
    payload_identity = identity["payload"]
    required_payload = {
        "top_level_keys",
        "model_keys",
        "residual_actor_fingerprint",
        "distribution_key",
        "distribution_fingerprint",
        "prefix_normalizer_keys",
        "prefix_normalizer_fingerprint",
    }
    if (
        not isinstance(payload_identity, Mapping)
        or set(payload_identity) != required_payload
        or tuple(payload_identity["top_level_keys"]) != tuple(sorted(_V015_HSL_TOP_LEVEL_KEYS))
        or tuple(payload_identity["model_keys"]) != tuple(sorted(("residual_actor", distribution_key)))
        or tuple(payload_identity["prefix_normalizer_keys"])
        != tuple(sorted(_EMPIRICAL_NORMALIZER_STATE_KEYS))
        or payload_identity["residual_actor_fingerprint"] != actor_fingerprint
        or payload_identity["distribution_key"] != distribution_key
        or payload_identity["distribution_fingerprint"] != distribution_fingerprint
        or payload_identity["prefix_normalizer_fingerprint"] != prefix_fingerprint
    ):
        raise RuntimeError("quality HSL payload fingerprint mismatch")
    return FrontRESActiveQualityCheckpointIdentity(
        route="hsl",
        format=_V015_HSL_CHECKPOINT_FORMAT,
        file_sha256=file_sha256,
        method_contract_id="FRS-METHOD-v017",
        training_contract_id="FRS-TRAIN-v014",
        gain_contract_id=None,
        ppo_contract_id=None,
        future_intent_layout=tuple(_v015_quality_expected_layout().items()),
        action_kind="delta_se3",
        action_dim=6,
        action_semantics="direct-world-full6-v1",
        normalizer_key=_V015_HSL_PREFIX_NORM_KEY,
        actor_fingerprint=actor_fingerprint,
        distribution_key=distribution_key,
        distribution_fingerprint=distribution_fingerprint,
        normalizer_fingerprint=prefix_fingerprint,
    )


def _inspect_frontres_v015_policy_quality_payload(
    checkpoint: Mapping[str, Any],
    *,
    file_sha256: str,
) -> FrontRESActiveQualityCheckpointIdentity:
    identity = checkpoint.get(_V015_CHECKPOINT_IDENTITY_KEY)
    if not isinstance(identity, Mapping):
        raise RuntimeError("quality policy requires the strict Stage-3 v015 checkpoint identity")
    if (
        identity.get("format") != _V015_CHECKPOINT_FORMAT
        or identity.get("method_contract_id") != "FRS-METHOD-v017"
        or identity.get("training_contract_id") != "FRS-TRAIN-v014"
        or identity.get("gain_contract_id") != "FRS-GAIN-v007"
        or identity.get("optimization_contract_id") != "FRS-PPO-v005"
        or identity.get("future_intent_layout") != _v015_quality_expected_layout()
        or identity.get("action")
        != {"kind": "delta_se3", "dim": 6, "semantics": "direct-world-full6-v1"}
    ):
        raise RuntimeError("quality policy has an incompatible v015 contract or layout identity")
    if identity.get("grouped_loss") != {
        "advantage_normalization": "grouped_scale_only",
        "candidate_layout_version": _V015_GROUPED_CANDIDATE_LAYOUT,
        "policy_rows_per_attempt": 1,
    }:
        raise RuntimeError("quality policy has an incompatible grouped-loss identity")
    if identity.get("physics_schema_id") != "clean-anchored-contact-zmp-survival-v1" or identity.get(
        "grouped_schema_id"
    ) != "grouped-all-attempt-scalar-v1":
        raise RuntimeError("quality policy has an incompatible v007 scalar/Physics identity")
    curriculum = identity.get("curriculum")
    if not isinstance(curriculum, Mapping):
        raise RuntimeError("quality policy has no FRS-TRAIN-v014 curriculum identity")
    schedule = curriculum.get("schedule")
    require_frontres_v013_campaign_schedule(schedule if isinstance(schedule, (tuple, list)) else ())
    iteration = int(curriculum.get("absolute_iteration", -1))
    expected = resolve_frontres_k_stage_identity(
        schedule=schedule if isinstance(schedule, (tuple, list)) else (),
        committed_update_iteration=iteration,
        max_horizon_k=max((int(row[0]) for row in schedule), default=0) if isinstance(schedule, (tuple, list)) else 0,
    )
    expected_payload = {
        "schedule": frontres_k_stage_schedule_tuple(schedule),
        "schedule_fingerprint": expected.schedule_fingerprint,
        "k_stage_index": expected.stage_index,
        "active_k": expected.active_k,
        "active_m": expected.active_m,
        "selected_segment_count": FRONTRES_V011_SELECTED_SEGMENT_COUNT,
        "policy_row_count": FRONTRES_V011_SELECTED_SEGMENT_COUNT * expected.active_m,
        "role_row_count": 2 * FRONTRES_V011_SELECTED_SEGMENT_COUNT * expected.active_m,
        "maximum_absolute_iteration": FRONTRES_V011_MAX_ABSOLUTE_ITERATION,
        "checkpoint_review_boundaries": FRONTRES_V011_REVIEW_BOUNDARIES,
        "stage_iteration": expected.stage_iteration,
        "absolute_iteration": expected.absolute_iteration,
        "phase": expected.phase.name,
        "phase_iteration": expected.phase.phase_iteration,
        "actor_loss_weight": expected.phase.actor_loss_weight,
        "dr_stage_fingerprint": expected.dr_stage_fingerprint,
        "dr_progress": expected.dr_progress,
        "d_cap": expected.d_cap,
    }
    if dict(curriculum) != expected_payload:
        raise RuntimeError("quality policy has an inconsistent FRS-TRAIN-v014 curriculum identity")
    transaction = identity.get("transaction")
    if not isinstance(transaction, Mapping) or str(transaction.get("state", "")) not in {"idle", "committed"}:
        raise RuntimeError("quality policy rejects partial or malformed transaction identity")
    if transaction["state"] == "idle" and set(transaction) != {"state"}:
        raise RuntimeError("quality policy idle transaction identity is malformed")
    if transaction["state"] == "committed":
        if set(transaction) != {"state", "receipt"}:
            raise RuntimeError("quality policy committed transaction identity is malformed")
        _v015_committed_transaction_receipt(transaction)
    normalizer_identity = identity.get("normalizer")
    if (
        not isinstance(normalizer_identity, Mapping)
        or set(normalizer_identity)
        != {"mode", "prefix_layout_version", "prefix_dim", "combined_dim", "prefix_stats_fingerprint"}
        or normalizer_identity.get("mode") != "empirical_prefix_plus_frozen_gmt"
        or normalizer_identity.get("prefix_layout_version") != FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
        or int(normalizer_identity.get("prefix_dim", -1)) != 158
        or int(normalizer_identity.get("combined_dim", -1)) != 928
    ):
        raise RuntimeError("quality policy normalizer identity is incompatible or padded")
    obs_norm = checkpoint.get("obs_norm_state_dict")
    _validate_v015_normalizer_state(obs_norm, dim=928, label="quality policy observation normalizer")
    prefix_fingerprint = _v015_tensor_fingerprint(obs_norm["_mean"][..., :158], obs_norm["_std"][..., :158])
    if normalizer_identity["prefix_stats_fingerprint"] != prefix_fingerprint:
        raise RuntimeError("quality policy prefix normalizer fingerprint mismatch")
    actor_fingerprint, distribution_key, distribution_fingerprint = _v015_quality_model_identity(
        checkpoint, label="quality policy"
    )
    return FrontRESActiveQualityCheckpointIdentity(
        route="policy",
        format=_V015_CHECKPOINT_FORMAT,
        file_sha256=file_sha256,
        method_contract_id="FRS-METHOD-v017",
        training_contract_id="FRS-TRAIN-v014",
        gain_contract_id="FRS-GAIN-v007",
        ppo_contract_id="FRS-PPO-v005",
        future_intent_layout=tuple(_v015_quality_expected_layout().items()),
        action_kind="delta_se3",
        action_dim=6,
        action_semantics="direct-world-full6-v1",
        normalizer_key="obs_norm_state_dict",
        actor_fingerprint=actor_fingerprint,
        distribution_key=distribution_key,
        distribution_fingerprint=distribution_fingerprint,
        normalizer_fingerprint=prefix_fingerprint,
    )


def inspect_frontres_quality_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    route: Literal["hsl", "policy"],
) -> FrontRESActiveQualityCheckpointIdentity:
    """Validate one quality artifact without restoring any mutable runner state."""

    if route not in {"hsl", "policy"}:
        raise ValueError("v015 quality checkpoint route must be hsl or policy")
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"v015 quality checkpoint does not exist: {path}")
    file_sha256 = _v015_file_sha256(path)
    checkpoint = load_frontres_checkpoint_mapping(path, map_location="cpu")
    if route == "hsl":
        return _inspect_frontres_v015_hsl_quality_payload(checkpoint, file_sha256=file_sha256)
    return _inspect_frontres_v015_policy_quality_payload(checkpoint, file_sha256=file_sha256)


# Public identity surface consumed by the mutable checkpoint gateway. Private
# aliases remain local implementation details for compatibility inside this owner.
EMPIRICAL_NORMALIZER_STATE_KEYS = _EMPIRICAL_NORMALIZER_STATE_KEYS
FRONTRES_ACTIVE_CHECKPOINT_FORMAT = _V015_CHECKPOINT_FORMAT
FRONTRES_ACTIVE_CHECKPOINT_IDENTITY_KEY = _V015_CHECKPOINT_IDENTITY_KEY
FRONTRES_ACTIVE_GROUPED_CANDIDATE_LAYOUT = _V015_GROUPED_CANDIDATE_LAYOUT
FRONTRES_HSL_CHECKPOINT_FORMAT = _V015_HSL_CHECKPOINT_FORMAT
FRONTRES_HSL_CHECKPOINT_IDENTITY_KEY = _V015_HSL_CHECKPOINT_IDENTITY_KEY
FRONTRES_HSL_PREFIX_NORM_KEY = _V015_HSL_PREFIX_NORM_KEY
FRONTRES_HSL_TOP_LEVEL_KEYS = _V015_HSL_TOP_LEVEL_KEYS
frontres_v015_clone_tensor_state = _v015_clone_tensor_state
frontres_v015_committed_transaction_receipt = _v015_committed_transaction_receipt
frontres_v015_file_sha256 = _v015_file_sha256
frontres_v015_state_dict_fingerprint = _v015_state_dict_fingerprint
frontres_v015_tensor_fingerprint = _v015_tensor_fingerprint
validate_frontres_v015_normalizer_state = _validate_v015_normalizer_state

__all__ = (
    "EMPIRICAL_NORMALIZER_STATE_KEYS",
    "FRONTRES_ACTIVE_CHECKPOINT_FORMAT",
    "FRONTRES_ACTIVE_CHECKPOINT_IDENTITY_KEY",
    "FRONTRES_ACTIVE_GROUPED_CANDIDATE_LAYOUT",
    "FRONTRES_HSL_CHECKPOINT_FORMAT",
    "FRONTRES_HSL_CHECKPOINT_IDENTITY_KEY",
    "FRONTRES_HSL_PREFIX_NORM_KEY",
    "FRONTRES_HSL_TOP_LEVEL_KEYS",
    "FrontRESActiveQualityCheckpointIdentity",
    "frontres_v015_clone_tensor_state",
    "frontres_v015_committed_transaction_receipt",
    "frontres_v015_file_sha256",
    "frontres_v015_state_dict_fingerprint",
    "frontres_v015_tensor_fingerprint",
    "inspect_frontres_quality_checkpoint",
    "require_frontres_v013_campaign_schedule",
    "validate_frontres_v015_normalizer_state",
)
