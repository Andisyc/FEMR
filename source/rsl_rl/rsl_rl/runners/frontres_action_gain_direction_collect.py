"""Bounded checkpoint-v19 action-to-Gain direction evidence collector.

This is an independent EVAL-v006 diagnostic entrypoint.  It reuses the public
fixed-Scenario materializer and recovery-aware collection owner, but it is not
the formal training entrypoint and therefore never claims Formal PASS.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, ContextManager, Mapping

import numpy as np
import torch

from rsl_rl.frontres.frontres_return_utility import frontres_symmetric_log_utility
from rsl_rl.runners.frontres_checkpoint_quality import (
    FrontRESActiveQualityCheckpointIdentity,
    inspect_frontres_quality_checkpoint,
)
from rsl_rl.runners.frontres_checkpointing import frontres_quality_route_actor
from rsl_rl.runners.frontres_evaluation_reporting import write_frontres_atomic_json
from rsl_rl.runners.frontres_segment_formal_transaction import (
    collect_frontres_recovery_aware_evaluation,
    frontres_readonly_collection_scope,
)
from rsl_rl.runners.frontres_segment_live_sampler import (
    close_frontres_local_scenarios,
    ensure_frontres_policy_quality_reset_support,
    prepare_frontres_action_gain_direction_fixed_k_m4_batch,
)
from rsl_rl.runners.frontres_stage3_engine import frontres_stage3_transaction_aggregate


SCHEMA = "frontres-action-gain-direction-v2"
DIAGNOSTIC_CLASS = "BOUNDED-DIAGNOSTIC/ALTERNATE-PATH"
_MANIFEST_SCHEMA = "frontres-v024-action-gain-direction-manifest-v1"
_EXPECTED_CONTRACTS = {
    "method_contract_id": "FRS-METHOD-v025",
    "training_contract_id": "FRS-TRAIN-v024",
    "gain_contract_id": "FRS-GAIN-v008",
    "ppo_contract_id": "FRS-PPO-v012",
    "evaluation_contract_id": "FRS-EVAL-v006",
    "checkpoint_format": "frontres-v024-checkpoint-v19",
}
_ITEM_FIELDS = {
    "item_id",
    "motion_id",
    "start_frame",
    "perturbation_family",
    "perturbation_parameters",
    "effective_horizon_k",
    "seed",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _comparison_signature(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class FrontRESActionGainDirectionItem:
    item_id: str
    motion_id: str
    start_frame: int
    perturbation_family: str
    perturbation_parameters: tuple[tuple[str, float], ...]
    effective_horizon_k: int
    seed: int
    comparison_signature: str


@dataclass(frozen=True)
class FrontRESActionGainDirectionManifest:
    manifest_file_sha256: str
    contracts: tuple[tuple[str, str], ...]
    action_dim: int
    attempts_per_segment: int
    repeat_count: int
    segments_per_transaction: int
    items: tuple[FrontRESActionGainDirectionItem, ...]


@dataclass(frozen=True)
class FrontRESActionGainDirectionRequest:
    manifest_path: str
    policy_checkpoint_path: str
    result_path: str
    manifest: FrontRESActionGainDirectionManifest
    checkpoint: FrontRESActiveQualityCheckpointIdentity


@dataclass(frozen=True)
class FrontRESActionGainDirectionOwners:
    ensure_reset_support: Callable[[Any], None]
    transaction_aggregate: Callable[[Any], Any]
    policy_route: Callable[[Any, str, str], ContextManager[Any]]
    readonly_scope: Callable[[Any], ContextManager[None]]
    prepare_batch: Callable[[Any, tuple[Any, Any], int], Any]
    close_prepared: Callable[[Any], None]
    collect: Callable[[Any, Any, Any | None, float, int], Any]
    training_state_hashes: Callable[[Any], dict[str, str]]
    replay_owner_present: Callable[[Any], bool]
    write_json: Callable[[str, Mapping[str, Any]], None]


def _parse_manifest(path: Path) -> FrontRESActionGainDirectionManifest:
    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("action-Gain direction manifest must be UTF-8 JSON") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != _MANIFEST_SCHEMA:
        raise ValueError(f"action-Gain direction manifest requires schema {_MANIFEST_SCHEMA!r}")
    for name, expected in _EXPECTED_CONTRACTS.items():
        if payload.get(name) != expected:
            raise ValueError(f"action-Gain direction manifest requires {name}={expected!r}")
    fixed = {
        "action_dim": 6,
        "horizon_k": 8,
        "attempts_per_segment": 4,
        "repeat_count": 8,
        "segments_per_transaction": 2,
        "fixed_segment_count": 4,
    }
    for name, expected in fixed.items():
        if payload.get(name) != expected:
            raise ValueError(f"action-Gain direction manifest requires {name}={expected}")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or len(raw_items) != fixed["fixed_segment_count"]:
        raise ValueError("action-Gain direction manifest requires exactly four fixed items")
    items: list[FrontRESActionGainDirectionItem] = []
    item_ids: set[str] = set()
    locations: set[tuple[str, int]] = set()
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, Mapping) or set(raw_item) != _ITEM_FIELDS:
            raise ValueError(f"action-Gain direction item {index} has an invalid field schema")
        item_id = raw_item.get("item_id")
        motion_id = raw_item.get("motion_id")
        family = raw_item.get("perturbation_family")
        start_frame = raw_item.get("start_frame")
        seed = raw_item.get("seed")
        horizon = raw_item.get("effective_horizon_k")
        if not all(isinstance(value, str) and value for value in (item_id, motion_id, family)):
            raise ValueError(f"action-Gain direction item {index} requires non-empty string identity")
        if family != "local_rp":
            raise ValueError("action-Gain direction items require the active local_rp family")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start_frame, seed, horizon)):
            raise ValueError(f"action-Gain direction item {index} has invalid integer fields")
        if start_frame < 0 or horizon != 8:
            raise ValueError("action-Gain direction items require a nonnegative frame and K8")
        raw_params = raw_item.get("perturbation_parameters")
        if not isinstance(raw_params, list) or len(raw_params) != 1:
            raise ValueError("action-Gain direction item requires exactly one perturbation parameter")
        parameter = raw_params[0]
        if (
            not isinstance(parameter, list)
            or len(parameter) != 2
            or parameter[0] != "dr_scale"
            or isinstance(parameter[1], bool)
            or not isinstance(parameter[1], (int, float))
            or not math.isfinite(float(parameter[1]))
            or float(parameter[1]) < 0.0
        ):
            raise ValueError("action-Gain direction item requires one finite nonnegative dr_scale")
        if item_id in item_ids or (motion_id, start_frame) in locations:
            raise ValueError("action-Gain direction manifest repeats an item or Segment location")
        item_ids.add(item_id)
        locations.add((motion_id, start_frame))
        items.append(
            FrontRESActionGainDirectionItem(
                item_id=item_id,
                motion_id=motion_id,
                start_frame=start_frame,
                perturbation_family=family,
                perturbation_parameters=(("dr_scale", float(parameter[1])),),
                effective_horizon_k=horizon,
                seed=seed,
                comparison_signature=_comparison_signature(raw_item),
            )
        )
    return FrontRESActionGainDirectionManifest(
        manifest_file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        contracts=tuple((name, str(payload[name])) for name in _EXPECTED_CONTRACTS),
        action_dim=6,
        attempts_per_segment=4,
        repeat_count=8,
        segments_per_transaction=2,
        items=tuple(items),
    )


def _validate_checkpoint(identity: FrontRESActiveQualityCheckpointIdentity) -> None:
    expected = dict(_EXPECTED_CONTRACTS)
    observed = {
        "checkpoint_format": identity.format,
        "method_contract_id": identity.method_contract_id,
        "training_contract_id": identity.training_contract_id,
        "gain_contract_id": identity.gain_contract_id,
        "ppo_contract_id": identity.ppo_contract_id,
    }
    for name, value in observed.items():
        if value != expected[name]:
            raise ValueError(f"action-Gain direction checkpoint requires {name}={expected[name]!r}")
    if (
        identity.route != "policy"
        or identity.action_kind != "delta_se3"
        or identity.action_dim != 6
        or identity.action_semantics != "direct-world-full6-v1"
        or identity.critic_input_dim != 449
        or identity.critic_value_kind != "state_value"
        or identity.critic_action_conditioned is not False
        or identity.critic_target_id != "scenario-current-exact-m4-mean-symlog-v1"
    ):
        raise ValueError("action-Gain direction checkpoint has incompatible Actor/Critic identity")


def build_frontres_action_gain_direction_request(
    *,
    manifest_path: str,
    policy_checkpoint_path: str,
    result_path: str,
    checkpoint_inspector: Callable[..., FrontRESActiveQualityCheckpointIdentity] = inspect_frontres_quality_checkpoint,
) -> FrontRESActionGainDirectionRequest:
    manifest_file = Path(manifest_path).expanduser().resolve()
    checkpoint_file = Path(policy_checkpoint_path).expanduser().resolve()
    result_file = Path(result_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"action-Gain direction manifest does not exist: {manifest_file}")
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"action-Gain direction checkpoint does not exist: {checkpoint_file}")
    if not result_file.parent.is_dir():
        raise FileNotFoundError(f"action-Gain direction result directory does not exist: {result_file.parent}")
    if result_file.exists() or result_file.with_suffix(result_file.suffix + ".tmp").exists():
        raise RuntimeError(f"action-Gain direction refuses existing result identity: {result_file}")
    manifest = _parse_manifest(manifest_file)
    checkpoint = checkpoint_inspector(checkpoint_file, route="policy")
    _validate_checkpoint(checkpoint)
    if checkpoint.file_sha256 != _sha256_file(checkpoint_file):
        raise RuntimeError("action-Gain direction checkpoint identity changed during inspection")
    return FrontRESActionGainDirectionRequest(
        manifest_path=str(manifest_file),
        policy_checkpoint_path=str(checkpoint_file),
        result_path=str(result_file),
        manifest=manifest,
        checkpoint=checkpoint,
    )


def _hash_state(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, Mapping):
        for key in sorted(value, key=repr):
            digest.update(repr(key).encode("utf-8"))
            _hash_state(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(type(value).__name__.encode("ascii"))
        for item in value:
            _hash_state(digest, item)
    else:
        digest.update(repr(value).encode("utf-8"))


def _field_hash(name: str, value: Any) -> str:
    digest = hashlib.sha256(name.encode("ascii"))
    _hash_state(digest, value)
    return digest.hexdigest()


def _training_state_hashes(runner: Any) -> dict[str, str]:
    alg = getattr(runner, "alg", None)
    policy = getattr(alg, "policy", None)
    aggregate = frontres_stage3_transaction_aggregate(runner)
    mode_roots = (
        ("policy", policy),
        ("prefix_normalizer", getattr(runner, "_frontres_extra_normalizer", None)),
        ("gmt_normalizer", getattr(runner, "obs_normalizer", None)),
        ("privileged_normalizer", getattr(runner, "privileged_obs_normalizer", None)),
        ("teacher_normalizer", getattr(runner, "teacher_obs_normalizer", None)),
    )
    module_modes = tuple(
        (f"{root_name}.{name}", bool(module.training))
        for root_name, root in mode_roots
        if isinstance(root, torch.nn.Module)
        for name, module in root.named_modules()
    )
    curriculum = {
        name: value
        for name, value in vars(runner).items()
        if name.startswith("_frontres_curriculum_") or name.startswith("_frontres_dr_")
    }
    replay = getattr(runner, "_frontres_outer_scenario_replay", None)
    values = (
        ("actor", getattr(getattr(policy, "residual_actor", None), "state_dict", lambda: {})()),
        ("critic", getattr(getattr(policy, "critic", None), "state_dict", lambda: {})()),
        ("policy_distribution", getattr(policy, "std", getattr(policy, "log_std", None))),
        ("optimizer", getattr(getattr(alg, "optimizer", None), "state_dict", lambda: {})()),
        ("critic_value_normalizer", getattr(getattr(alg, "frontres_critic_value_normalizer_state", None), "state_dict", lambda: {})()),
        ("prefix_normalizer", getattr(getattr(runner, "_frontres_extra_normalizer", None), "state_dict", lambda: {})()),
        ("gmt_normalizer", getattr(getattr(runner, "obs_normalizer", None), "state_dict", lambda: {})()),
        ("privileged_normalizer", getattr(getattr(runner, "privileged_obs_normalizer", None), "state_dict", lambda: {})()),
        ("sampler", getattr(getattr(runner, "_frontres_segment_sampler", None), "state_dict", lambda: {})()),
        ("outer_replay", None if replay is None else getattr(replay, "state_dict", lambda: {})()),
        ("transaction", {"execution": aggregate.execution_phase, "persisted": aggregate.as_dict()}),
        ("receipt", getattr(runner, "_frontres_last_committed_transaction_receipt", None)),
        ("checkpoint_transaction", getattr(runner, "_frontres_checkpoint_transaction_state", None)),
        ("curriculum", curriculum),
        ("module_modes", module_modes),
        ("warmup", getattr(runner, "_frontres_warmup_complete", None)),
        ("iteration", getattr(runner, "current_learning_iteration", None)),
    )
    return {name: _field_hash(name, value) for name, value in values}


def _real_policy_route(runner: Any, checkpoint_path: str, file_sha256: str) -> ContextManager[Any]:
    return frontres_quality_route_actor(
        runner,
        checkpoint_path,
        route="policy",
        expected_file_sha256=file_sha256,
    )


def _real_prepare_batch(runner: Any, items: tuple[Any, Any], attempts: int) -> Any:
    return prepare_frontres_action_gain_direction_fixed_k_m4_batch(
        runner,
        items,
        attempts_per_segment=attempts,
    )


def _rng_state_hashes() -> dict[str, str]:
    values = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else (),
    }
    return {name: _field_hash(name, value) for name, value in values.items()}


@contextmanager
def _action_rng_scope(seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_state:
            torch.cuda.set_rng_state_all(cuda_state)


def _action_seed(request: FrontRESActionGainDirectionRequest, pair_key: tuple[str, str], repeat: int) -> int:
    material = "|".join(
        (
            request.manifest.manifest_file_sha256,
            request.checkpoint.file_sha256,
            *pair_key,
            str(repeat),
            "repair-action-rng-v1",
        )
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63 - 1)


def _runtime_seed(request: FrontRESActionGainDirectionRequest, pair_key: tuple[str, str]) -> int:
    material = "|".join(
        (
            request.manifest.manifest_file_sha256,
            request.checkpoint.file_sha256,
            *pair_key,
            "fixed-runtime-rng-v1",
        )
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63 - 1)


def _used_policy_input_drifts(current: Any, frozen: Any | None) -> tuple[float, float]:
    """Compare the policy inputs actually consumed, not unused live-history observations."""

    values: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    for name in ("obs", "privileged_obs"):
        value = getattr(current, name, None)
        if not isinstance(value, torch.Tensor) or value.requires_grad or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"action-Gain direction collection lost finite detached used {name}")
        values.append(value)
        if frozen is not None:
            reference = getattr(frozen, name, None)
            if not isinstance(reference, torch.Tensor) or tuple(reference.shape) != tuple(value.shape):
                raise RuntimeError(f"action-Gain direction collection changed used {name} shape")
            references.append(reference)
    if frozen is None:
        return 0.0, 0.0
    return tuple(
        float((value - reference).abs().max().detach().cpu().item())
        for value, reference in zip(values, references, strict=True)
    )


def _real_collect(
    runner: Any,
    prepared: Any,
    frozen: Any | None,
    beta: float,
    action_seed: int,
) -> Any:
    return collect_frontres_recovery_aware_evaluation(
        runner,
        prepared,
        route="policy_quality",
        label="EVAL-v006 bounded action-Gain direction diagnostic",
        beta=beta,
        policy_observations=frozen,
        policy_action_seed=action_seed,
    )


def _real_owners() -> FrontRESActionGainDirectionOwners:
    return FrontRESActionGainDirectionOwners(
        ensure_reset_support=ensure_frontres_policy_quality_reset_support,
        transaction_aggregate=frontres_stage3_transaction_aggregate,
        policy_route=_real_policy_route,
        readonly_scope=frontres_readonly_collection_scope,
        prepare_batch=_real_prepare_batch,
        close_prepared=lambda prepared: close_frontres_local_scenarios(
            getattr(prepared, "batch", None)
        ),
        collect=_real_collect,
        training_state_hashes=_training_state_hashes,
        replay_owner_present=lambda runner: getattr(runner, "_frontres_outer_scenario_replay", None) is not None,
        write_json=lambda path, payload: write_frontres_atomic_json(path, payload, compact=True),
    )


def _finite_vector(value: Any, *, label: str, positive: bool = False) -> list[float]:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (6,) or not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{label} must be one finite 6D tensor")
    result = [float(item) for item in value.detach().to(device="cpu", dtype=torch.float64).tolist()]
    if positive and any(item <= 0.0 for item in result):
        raise RuntimeError(f"{label} must be strictly positive")
    return result


def _json_tensor(value: torch.Tensor, *, label: str) -> Any:
    """Serialize raw executed evidence without inventing a derived score."""

    if not isinstance(value, torch.Tensor) or value.requires_grad or bool(torch.isinf(value).any()):
        raise RuntimeError(f"{label} must be a detached finite tensor with optional NaN semantics")
    array = value.detach().to(device="cpu").tolist()

    def clean(item: Any) -> Any:
        if isinstance(item, list):
            return [clean(child) for child in item]
        if isinstance(item, float) and math.isnan(item):
            return None
        if isinstance(item, (int, float, bool)):
            return item
        raise RuntimeError(f"{label} contains a non-JSON scalar")

    return clean(array)


def _trajectory_payload(trajectory: Any, *, label: str) -> dict[str, Any]:
    fields = (
        "joint_pos",
        "root_pos",
        "root_quat",
        "key_body_pos",
        "root_lin_vel",
        "root_ang_vel",
        "foot_pos",
        "contact",
        "zmp_margin",
        "survival",
        "valid_mask",
    )
    payload = {name: _json_tensor(getattr(trajectory, name), label=f"{label}.{name}") for name in fields}
    payload["schema"] = "frontres-raw-k-trajectory-v1"
    return payload


def _collection_rows(collection: Any, *, expected_sources: int, active_m: int) -> dict[int, dict[str, Any]]:
    evidence = getattr(collection, "evidence", None)
    gain = getattr(collection, "gain", None)
    attempts = tuple(getattr(evidence, "ordered_attempts", ()) or ())
    count = expected_sources * active_m
    if len(attempts) != count:
        raise RuntimeError("action-Gain direction collection lost exact B2 x M4 attempts")
    component_tensors = {
        "gain_total": getattr(gain, "gain_total", None),
        "intent_gain": getattr(gain, "intent_gain", None),
        "physics_remaining_noisy": getattr(gain, "physics_remaining_noisy", None),
        "physics_remaining_repaired": getattr(gain, "physics_remaining_repaired", None),
        "physics_gain": getattr(gain, "physics_gain", None),
        "recovery_pressure": getattr(gain, "recovery_pressure", None),
        "weighted_physics_gain": getattr(gain, "weighted_physics_gain", None),
        "repair_penalty": getattr(gain, "repair_penalty", None),
    }
    if any(
        not isinstance(value, torch.Tensor)
        or tuple(value.reshape(-1).shape) != (count,)
        or not bool(torch.isfinite(value).all())
        for value in component_tensors.values()
    ):
        raise RuntimeError("action-Gain direction collection has incomplete finite Gain components")
    physics_channel_tensors = {
        "physics_channel_noisy": getattr(gain, "physics_channel_noisy", None),
        "physics_channel_repaired": getattr(gain, "physics_channel_repaired", None),
    }
    if any(
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (count, 4)
        or not bool(torch.isfinite(value[:, (0, 3)]).all())
        or not bool((torch.isfinite(value[:, (1, 2)]) | torch.isnan(value[:, (1, 2)])).all())
        for value in physics_channel_tensors.values()
    ):
        raise RuntimeError("action-Gain direction collection has malformed Physics channels")

    def optional_channel_row(value: torch.Tensor) -> list[float | None]:
        return [None if math.isnan(float(item)) else float(item) for item in value.detach().cpu().tolist()]
    totals = component_tensors["gain_total"].detach().reshape(-1)
    utilities = frontres_symmetric_log_utility(totals).detach()
    rows_by_source: dict[int, dict[str, Any]] = {}
    baseline_by_source = {
        int(value.source_index): value for value in getattr(evidence, "baselines", ())
    }
    if set(baseline_by_source) != set(range(expected_sources)):
        raise RuntimeError("action-Gain direction collection lost raw Clean/Noisy baseline identity")
    for source in range(expected_sources):
        indices = [index for index, attempt in enumerate(attempts) if int(attempt.source_index) == source]
        if len(indices) != active_m or sorted(int(attempts[index].trial_index) for index in indices) != list(range(active_m)):
            raise RuntimeError("action-Gain direction collection lost exact source/trial identity")
        ordered = sorted(indices, key=lambda index: int(attempts[index].trial_index))
        source_attempts = [attempts[index] for index in ordered]
        means = [_finite_vector(attempt.policy_mean, label="policy_mean") for attempt in source_attempts]
        sigmas = [_finite_vector(attempt.policy_sigma, label="policy_sigma", positive=True) for attempt in source_attempts]
        if any(value != means[0] for value in means[1:]) or any(value != sigmas[0] for value in sigmas[1:]):
            raise RuntimeError("action-Gain direction same-Scenario M4 lost shared Actor distribution")
        identities = {
            "scenario_id": {str(attempt.scenario_id) for attempt in source_attempts},
            "noisy_segment_hash": {str(attempt.noisy_segment_hash) for attempt in source_attempts},
            "x_t_identity": {str(attempt.x_t_identity) for attempt in source_attempts},
            "segment_id": {int(attempt.segment_id) for attempt in source_attempts},
        }
        if any(len(value) != 1 for value in identities.values()):
            raise RuntimeError("action-Gain direction M4 mixed Scenario/state identity")
        rows = []
        action_fingerprints: set[str] = set()
        for index in ordered:
            attempt = attempts[index]
            action = _finite_vector(attempt.policy_action, label="policy_action")
            fingerprint = hashlib.sha256(json.dumps(action, separators=(",", ":")).encode("ascii")).hexdigest()
            action_fingerprints.add(fingerprint)
            gain_total = float(totals[index].cpu().item())
            repair_penalty = float(component_tensors["repair_penalty"][index].detach().cpu().item())
            rows.append(
                {
                    "action": action,
                    "components": {
                        "utility": float(utilities[index].cpu().item()),
                        "raw_return": gain_total,
                        "gain_total": gain_total,
                        "intent_gain": float(component_tensors["intent_gain"][index].detach().cpu().item()),
                        "physics_remaining_noisy": float(component_tensors["physics_remaining_noisy"][index].detach().cpu().item()),
                        "physics_remaining_repaired": float(component_tensors["physics_remaining_repaired"][index].detach().cpu().item()),
                        "physics_gain": float(component_tensors["physics_gain"][index].detach().cpu().item()),
                        "recovery_pressure": float(component_tensors["recovery_pressure"][index].detach().cpu().item()),
                        "weighted_physics_gain": float(component_tensors["weighted_physics_gain"][index].detach().cpu().item()),
                        "physics_channel_noisy": optional_channel_row(
                            physics_channel_tensors["physics_channel_noisy"][index]
                        ),
                        "physics_channel_repaired": optional_channel_row(
                            physics_channel_tensors["physics_channel_repaired"][index]
                        ),
                        "repair_penalty": repair_penalty,
                        "negative_repair_cost": -repair_penalty,
                    },
                    "raw_physics": {
                        "repair": _trajectory_payload(
                            attempt.repair,
                            label=f"source={source},trial={attempt.trial_index}.repair",
                        ),
                    },
                }
            )
        if len(action_fingerprints) != active_m:
            raise RuntimeError("action-Gain direction requires four distinct sampled actions per M4")
        rows_by_source[source] = {
            "actor_mean": means[0],
            "actor_sigma": sigmas[0],
            "scenario_id": next(iter(identities["scenario_id"])),
            "noisy_segment_hash": next(iter(identities["noisy_segment_hash"])),
            "x_t_identity": next(iter(identities["x_t_identity"])),
            "segment_id": next(iter(identities["segment_id"])),
            "raw_physics_baseline": {
                "expected_support": _json_tensor(
                    baseline_by_source[source].expected_support,
                    label=f"source={source}.expected_support",
                ),
                "clean": _trajectory_payload(
                    baseline_by_source[source].clean,
                    label=f"source={source}.clean",
                ),
                "noisy": _trajectory_payload(
                    baseline_by_source[source].noisy,
                    label=f"source={source}.noisy",
                ),
            },
            "rows": rows,
        }
    return rows_by_source


def run_frontres_action_gain_direction_collect(
    runner: Any,
    *,
    request: FrontRESActionGainDirectionRequest,
    owners: FrontRESActionGainDirectionOwners | None = None,
) -> dict[str, Any]:
    """Collect four fixed K8 Scenarios x 32 Repair rows without training writes."""

    if not isinstance(request, FrontRESActionGainDirectionRequest):
        raise TypeError("action-Gain direction collector requires a strict request")
    active_owners = _real_owners() if owners is None else owners
    alg = getattr(runner, "alg", None)
    if not bool(getattr(alg, "frontres_formal_transaction_enabled", False)):
        raise RuntimeError("action-Gain direction collector requires active Stage-3 observation/collection owners")
    if not bool(getattr(alg, "frontres_policy_quality_eval_only", False)):
        raise RuntimeError("action-Gain direction collector requires the shared evaluation-only config")
    if bool(getattr(alg, "frontres_segment_replay_enabled", False)) or bool(
        getattr(alg, "frontres_segment_live_runner_enabled", False)
    ):
        raise RuntimeError("action-Gain direction collector forbids Replay and live-training routes")

    active_owners.ensure_reset_support(runner)
    aggregate = active_owners.transaction_aggregate(runner)
    if aggregate.execution_phase != "idle" or aggregate.persistence_phase in {"collecting", "sealed"}:
        raise RuntimeError("action-Gain direction collector requires an idle transaction owner")
    if active_owners.replay_owner_present(runner):
        raise RuntimeError("action-Gain direction collector requires Replay owner absence")
    state_before = active_owners.training_state_hashes(runner)
    rng_before = _rng_state_hashes()
    manifest = request.manifest
    scenario_rows: dict[str, dict[str, Any]] = {
        item.item_id: {
            "item_id": item.item_id,
            "active_m": manifest.attempts_per_segment,
            "checkpoint_file_sha256": request.checkpoint.file_sha256,
            "manifest_file_sha256": manifest.manifest_file_sha256,
            "visits": [],
            "rows": [],
        }
        for item in manifest.items
    }
    frozen_observations: dict[tuple[str, str], Any] = {}
    policy_state_before: dict[str, str] | None = None
    policy_state_after: dict[str, str] | None = None
    beta = getattr(alg, "frontres_gain_beta", None)
    if beta is None or not math.isfinite(float(beta)) or float(beta) < 0.0:
        raise RuntimeError("action-Gain direction collector requires finite FRS-GAIN-v008 beta")

    with active_owners.policy_route(
        runner,
        request.policy_checkpoint_path,
        request.checkpoint.file_sha256,
    ):
        policy_state_before = active_owners.training_state_hashes(runner)
        for repeat_index in range(manifest.repeat_count):
            for offset in range(0, len(manifest.items), manifest.segments_per_transaction):
                item_pair = tuple(manifest.items[offset : offset + manifest.segments_per_transaction])
                pair_key = tuple(item.item_id for item in item_pair)
                frozen = frozen_observations.get(pair_key)
                action_seed = _action_seed(request, pair_key, repeat_index)
                runtime_seed = _runtime_seed(request, pair_key)
                rng_pair_before = _rng_state_hashes()
                prepared = None
                with _action_rng_scope(runtime_seed):
                    try:
                        with active_owners.readonly_scope(runner):
                            prepared = active_owners.prepare_batch(
                                runner, item_pair, manifest.attempts_per_segment
                            )
                            collection = active_owners.collect(
                                runner, prepared, frozen, float(beta), action_seed
                            )
                    finally:
                        if prepared is not None:
                            active_owners.close_prepared(prepared)
                if _rng_state_hashes() != rng_pair_before:
                    raise RuntimeError("action-Gain direction action RNG scope failed to restore global RNG state")
                trace = dict(getattr(collection, "observation_trace", {}) or {})
                live_actor_drift = float(trace.get("repeat_live_actor_input_max_abs_diff", float("nan")))
                live_critic_drift = float(trace.get("repeat_live_critic_input_max_abs_diff", float("nan")))
                expected_source = "live-first-repeat" if frozen is None else "first-repeat-frozen"
                if (
                    not math.isfinite(live_actor_drift)
                    or not math.isfinite(live_critic_drift)
                    or live_actor_drift < 0.0
                    or live_critic_drift < 0.0
                    or trace.get("repeat_policy_input_source") != expected_source
                ):
                    raise RuntimeError(
                        "action-Gain direction collection lost policy-input provenance; "
                        f"source={trace.get('repeat_policy_input_source')!r} "
                        f"live_actor_drift={live_actor_drift} live_critic_drift={live_critic_drift}"
                    )
                used = getattr(collection, "policy_observations", None)
                actor_drift, critic_drift = _used_policy_input_drifts(used, frozen)
                if actor_drift != 0.0 or critic_drift != 0.0:
                    raise RuntimeError(
                        "action-Gain direction repeat changed used Actor/Critic inputs; "
                        f"actor_drift={actor_drift} critic_drift={critic_drift}"
                    )
                if frozen is None:
                    frozen_observations[pair_key] = used
                collected = _collection_rows(
                    collection,
                    expected_sources=manifest.segments_per_transaction,
                    active_m=manifest.attempts_per_segment,
                )
                for source, item in enumerate(item_pair):
                    data = collected[source]
                    target = scenario_rows[item.item_id]
                    identity_fields = (
                        "actor_mean",
                        "actor_sigma",
                        "scenario_id",
                        "noisy_segment_hash",
                        "x_t_identity",
                        "segment_id",
                        "raw_physics_baseline",
                    )
                    if repeat_index == 0:
                        for name in identity_fields:
                            target[name] = data[name]
                    elif any(target[name] != data[name] for name in identity_fields):
                        raise RuntimeError("action-Gain direction repeat changed frozen Scenario/policy identity")
                    target["visits"].append(
                        {
                            "visit_index": repeat_index,
                            "action_seed": action_seed,
                            "runtime_seed": runtime_seed,
                            "actor_input_max_abs_diff": actor_drift,
                            "critic_input_max_abs_diff": critic_drift,
                            "live_actor_input_max_abs_diff": live_actor_drift,
                            "live_critic_input_max_abs_diff": live_critic_drift,
                        }
                    )
                    for trial_offset, row in enumerate(data["rows"]):
                        target["rows"].append(
                            {
                                "repair_index": repeat_index * manifest.attempts_per_segment + trial_offset,
                                "visit_index": repeat_index,
                                "attempt_index": trial_offset,
                                "action_seed": action_seed,
                                "runtime_seed": runtime_seed,
                                "checkpoint_file_sha256": request.checkpoint.file_sha256,
                                "manifest_file_sha256": manifest.manifest_file_sha256,
                                **row,
                            }
                        )
                current_state = active_owners.training_state_hashes(runner)
                if current_state != policy_state_before:
                    differing = tuple(name for name in policy_state_before if current_state.get(name) != policy_state_before[name])
                    raise RuntimeError(
                        "action-Gain direction collection mutated training state; "
                        f"differing_fields={differing}"
                    )
                if active_owners.replay_owner_present(runner):
                    raise RuntimeError("action-Gain direction collection constructed a Replay owner")
        policy_state_after = active_owners.training_state_hashes(runner)

    state_after = active_owners.training_state_hashes(runner)
    rng_after = _rng_state_hashes()
    if state_after != state_before:
        differing = tuple(name for name in state_before if state_after.get(name) != state_before[name])
        raise RuntimeError(
            "action-Gain direction checkpoint route failed to restore state; "
            f"differing_fields={differing}"
        )
    if active_owners.replay_owner_present(runner):
        raise RuntimeError("action-Gain direction collector left a Replay owner")
    if rng_after != rng_before:
        raise RuntimeError("action-Gain direction collector failed to restore global RNG state")
    for item in manifest.items:
        scenario = scenario_rows[item.item_id]
        if len(scenario["rows"]) != 32 or len(scenario["visits"]) != 8:
            raise RuntimeError("action-Gain direction collector did not produce exactly 32 rows per Scenario")
        visit_fingerprints = {
            hashlib.sha256(
                json.dumps(
                    [row["action"] for row in scenario["rows"] if row["visit_index"] == visit],
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest()
            for visit in range(8)
        }
        if len(visit_fingerprints) != 8:
            raise RuntimeError("action-Gain direction requires eight unique M4 action-group fingerprints")

    payload = {
        "schema": SCHEMA,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "formal_pass": False,
        "manifest_file_sha256": manifest.manifest_file_sha256,
        "checkpoint_file_sha256": request.checkpoint.file_sha256,
        "contracts": dict(manifest.contracts),
        "collection_identity": {
            "fixed_segment_count": len(manifest.items),
            "active_k": 8,
            "active_m": manifest.attempts_per_segment,
            "repeat_count": manifest.repeat_count,
            "rows_per_scenario": manifest.attempts_per_segment * manifest.repeat_count,
            "actor_checkpoint_frozen": True,
            "first_repeat_policy_observations_frozen": True,
            "runtime_rng_fixed_across_visits": True,
            "policy_action_rng_isolated_per_visit": True,
            "actor_updated": False,
            "critic_updated": False,
            "optimizer_updated": False,
            "normalizer_updated": False,
            "replay_constructed": False,
            "raw_physics_exported": True,
            "raw_physics_schema": "frontres-raw-k-trajectory-v1",
        },
        "state_hashes": {
            "before_checkpoint_route": state_before,
            "policy_before_collection": policy_state_before,
            "policy_after_collection": policy_state_after,
            "after_checkpoint_restore": state_after,
        },
        "rng_hashes": {
            "before_collection": rng_before,
            "after_collection": rng_after,
        },
        "scenarios": [scenario_rows[item.item_id] for item in manifest.items],
    }
    active_owners.write_json(request.result_path, payload)
    return payload


def collect_frontres_action_gain_direction(
    runner: Any,
    *,
    manifest_path: str,
    policy_checkpoint_path: str,
    result_path: str,
) -> dict[str, Any]:
    request = build_frontres_action_gain_direction_request(
        manifest_path=manifest_path,
        policy_checkpoint_path=policy_checkpoint_path,
        result_path=result_path,
    )
    return run_frontres_action_gain_direction_collect(runner, request=request)


__all__ = (
    "DIAGNOSTIC_CLASS",
    "SCHEMA",
    "FrontRESActionGainDirectionItem",
    "FrontRESActionGainDirectionManifest",
    "FrontRESActionGainDirectionOwners",
    "FrontRESActionGainDirectionRequest",
    "build_frontres_action_gain_direction_request",
    "collect_frontres_action_gain_direction",
    "run_frontres_action_gain_direction_collect",
)
