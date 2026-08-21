"""Official read-only gateway for Clean-repeat calibration evidence.

This gateway owns manifest-to-request composition and the existing reset/
materialization lifecycle.  It never invokes Gain, fills missing evidence, or
performs an optimizer update.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - fail closed when the runtime lacks NumPy
    np = None

import torch

from rsl_rl.frontres.frontres_clean_relative_calibration import (
    CleanCalibrationCollectionRequest,
    CleanCalibrationCollectionIdentity,
    CleanCalibrationRepeatSpec,
    ReadOnlyCleanWindow,
    ReadOnlyCleanCollection,
    ReadOnlyCleanCollectionReceipt,
    adapt_read_only_clean_collection,
)
from rsl_rl.runners.frontres_segment_live_sampler import (
    close_frontres_local_scenarios,
    ensure_frontres_readonly_reset_support,
    prepare_frontres_fixed_k_m4_evaluation_batch,
)
from rsl_rl.runners.frontres_segment_formal_transaction import frontres_readonly_collection_scope
from rsl_rl.runners.frontres_segment_formal_transaction import resolve_frontres_mode_state
from rsl_rl.runners.frontres_stage3_engine import frontres_stage3_transaction_aggregate
from rsl_rl.runners.frontres_segment_live_reset import apply_frontres_current_segment_reset
from rsl_rl.runners.frontres_segment_runtime_types import bind_frontres_collection_context
from rsl_rl.runners.frontres_segment_transaction import capture_frontres_frozen_policy_snapshot
from rsl_rl.frontres.frontres_balance import prepare_frontres_raw_contact_views
from rsl_rl.runners.frontres_training_setup import configure_frontres_pair_layout
from rsl_rl.runners.frontres_segment_one_action_k import collect_frontres_v017_no_actor_baseline
from rsl_rl.modules import FrontRESActorCritic
from rsl_rl.runners.frontres_clean_calibration_telemetry import (
    FrontRESCleanRawWindow,
    build_clean_calibration_measurement,
)
from rsl_rl.runners.frontres_evaluation_reporting import write_frontres_atomic_json


FRONTRES_CLEAN_CALIBRATION_ROUTE = "clean_calibration"
FRONTRES_CLEAN_CALIBRATION_ROUTE_ID = "FRS-EVAL-v010-clean-calibration-v001"
FRONTRES_CLEAN_CALIBRATION_COLLECTOR_ID = "frontres-v010-clean-telemetry"
FRONTRES_CLEAN_CALIBRATION_COLLECTOR_VERSION = "v1"
_CLEAN_MANIFEST_REQUIRED = frozenset(
    {
        "route_id",
        "calibration_id",
        "domain_id",
        "field_schema_id",
        "horizon_k",
        "timestep_seconds",
        "seed_protocol_id",
        "coverage",
        "scenario_source_index",
        "repeats",
        "segments",
    }
)
_CLEAN_MANIFEST_OPTIONAL = frozenset({"expected_identity"})


def _hash_payload(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_digest(value: Any) -> str:
    """Hash an immutable tensor carrier without changing its owning device."""

    if not isinstance(value, torch.Tensor) or value.requires_grad:
        raise RuntimeError("clean calibration identity requires detached tensor carriers")
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _strict_clean_manifest(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("clean calibration manifest must be an object")
    keys = frozenset(payload)
    missing = sorted(_CLEAN_MANIFEST_REQUIRED - keys)
    unexpected = sorted(keys - _CLEAN_MANIFEST_REQUIRED - _CLEAN_MANIFEST_OPTIONAL)
    if missing:
        raise ValueError(f"clean calibration manifest missing fields: {missing}")
    if unexpected:
        raise ValueError(f"clean calibration manifest has unexpected fields: {unexpected}")
    if payload["route_id"] != FRONTRES_CLEAN_CALIBRATION_ROUTE_ID:
        raise ValueError("clean calibration manifest route identity mismatch")
    if payload["calibration_id"] != FRONTRES_CLEAN_CALIBRATION_ROUTE_ID:
        raise ValueError("clean calibration calibration_id must equal its route identity")
    if not isinstance(payload["domain_id"], str) or not payload["domain_id"]:
        raise ValueError("clean calibration domain_id must be non-empty")
    if not isinstance(payload["field_schema_id"], str) or not payload["field_schema_id"]:
        raise ValueError("clean calibration field_schema_id must be non-empty")
    if isinstance(payload["horizon_k"], bool) or not isinstance(payload["horizon_k"], int) or payload["horizon_k"] <= 0:
        raise ValueError("clean calibration horizon_k must be positive")
    if not isinstance(payload["timestep_seconds"], (int, float)) or float(payload["timestep_seconds"]) <= 0.0:
        raise ValueError("clean calibration timestep_seconds must be positive")
    if not isinstance(payload["seed_protocol_id"], str) or not payload["seed_protocol_id"]:
        raise ValueError("clean calibration seed_protocol_id must be non-empty")
    if not isinstance(payload["coverage"], (int, float)) or not 0.0 < float(payload["coverage"]) < 1.0:
        raise ValueError("clean calibration coverage must be in (0,1)")
    if not isinstance(payload["repeats"], list) or len(payload["repeats"]) < 2:
        raise ValueError("clean calibration manifest requires at least two repeats")
    if not isinstance(payload["segments"], list) or len(payload["segments"]) != 2:
        raise ValueError("clean calibration manifest requires exactly two materialized Segments")
    if isinstance(payload["scenario_source_index"], bool) or not isinstance(payload["scenario_source_index"], int):
        raise ValueError("clean calibration scenario_source_index must be an integer")
    if payload["scenario_source_index"] not in (0, 1):
        raise ValueError("clean calibration scenario_source_index must select one of the two prepared sources")
    required_item = frozenset(
        {
            "item_id",
            "motion_id",
            "start_frame",
            "perturbation_family",
            "perturbation_parameters",
            "effective_horizon_k",
            "seed",
        }
    )
    seen_ids: set[str] = set()
    for item in payload["segments"]:
        if not isinstance(item, dict) or frozenset(item) != required_item:
            raise ValueError("clean calibration Segment item schema mismatch")
        item_id = item["item_id"]
        if not isinstance(item_id, str) or not item_id or item_id in seen_ids:
            raise ValueError("clean calibration Segment item identities must be unique")
        seen_ids.add(item_id)
        if item["effective_horizon_k"] != payload["horizon_k"]:
            raise ValueError("clean calibration Segment K differs from manifest K")
        if item["perturbation_family"] != "local_rp":
            raise ValueError("clean calibration requires the local_rp clean materializer family")
        if item["perturbation_parameters"] != [["strength", 0.0]]:
            raise ValueError("clean calibration requires zero perturbation strength")
    required_repeat = frozenset({"repeat_id", "seed"})
    repeat_ids: list[str] = []
    for repeat in payload["repeats"]:
        if not isinstance(repeat, dict) or frozenset(repeat) != required_repeat:
            raise ValueError("clean calibration repeat schema mismatch")
        repeat_id = repeat["repeat_id"]
        if not isinstance(repeat_id, str) or not repeat_id or repeat_id in repeat_ids:
            raise ValueError("clean calibration repeat identities must be unique")
        repeat_ids.append(repeat_id)
    if repeat_ids != sorted(repeat_ids):
        raise ValueError("clean calibration repeat identities must be sorted")
    repeat_seeds = [repeat["seed"] for repeat in payload["repeats"]]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in repeat_seeds):
        raise ValueError("clean calibration repeat seeds must be integers")
    if len(set(repeat_seeds)) != len(repeat_seeds):
        raise ValueError("clean calibration repeat seeds must be distinct")
    return payload


def _manifest_items(payload: dict[str, Any]) -> tuple[Any, Any]:
    return tuple(
        SimpleNamespace(
            item_id=str(item["item_id"]),
            motion_id=str(item["motion_id"]),
            start_frame=int(item["start_frame"]),
            perturbation_family=str(item["perturbation_family"]),
            perturbation_parameters=tuple(
                (str(key), value) for key, value in item["perturbation_parameters"]
            ),
            effective_horizon_k=int(item["effective_horizon_k"]),
            seed=int(item["seed"]),
            comparison_signature=_hash_payload(item),
        )
        for item in payload["segments"]
    )  # type: ignore[return-value]


def _build_request_from_prepared(
    runner: Any,
    payload: dict[str, Any],
    prepared: FrontRESCleanCalibrationPreparedOwner,
) -> CleanCalibrationCollectionRequest:
    plan = prepared.plan
    source_index = int(payload["scenario_source_index"])
    rows = [index for index, value in enumerate(plan.source_index.tolist()) if int(value) == source_index]
    if not rows:
        raise RuntimeError("clean calibration manifest source_index is absent from the prepared plan")
    row = rows[0]
    scenario_id = str(plan.scenario_ids[row])
    x_t_identity = str(plan.x_t_identities[row])
    cache_artifact_hash = str(plan.noisy_segment_hashes[row])
    if len(cache_artifact_hash) != 64:
        raise RuntimeError("prepared clean calibration cache identity is not a SHA-256 hash")
    scenarios = getattr(prepared.batch, "frontres_local_scenario_rows", None)
    if scenarios is None or not callable(getattr(scenarios, "scenario_for_row", None)):
        raise RuntimeError("clean calibration requires the sealed Stage-1 local scenario owner")
    scenario = scenarios.scenario_for_row(row)
    clean_artifact_hash = _hash_payload(
        {
            "x_t_identity": x_t_identity,
            "clean_reference_t": _tensor_digest(scenario.clean_reference_t),
            "clean_continuation": _tensor_digest(scenario.clean_continuation),
            "expected_support": _tensor_digest(scenario.expected_support),
            "expected_support_envelope": _tensor_digest(scenario.expected_support_envelope),
        }
    )
    expected_support_hash = _tensor_digest(scenario.expected_support)
    snapshot = capture_frontres_frozen_policy_snapshot(
        runner,
        transaction_id=str(getattr(plan, "transaction_id", payload["calibration_id"])),
    )
    # The current runner exposes a loaded GMT policy-state hash, not a file
    # digest.  Bind that exact state identity and do not relabel a snapshot
    # serialization as a checkpoint artifact.
    gmt_checkpoint_hash = str(getattr(snapshot, "policy_state_hash", ""))
    if len(gmt_checkpoint_hash) != 64 or any(char not in "0123456789abcdef" for char in gmt_checkpoint_hash):
        raise RuntimeError("clean calibration requires the loaded GMT policy-state identity owner")
    normalizer = getattr(runner, "obs_normalizer", None)
    if normalizer is None or not callable(getattr(normalizer, "state_dict", None)):
        raise RuntimeError("clean calibration requires the official GMT normalizer state owner")
    normalizer_state = normalizer.state_dict()
    gmt_normalizer_hash = _hash_payload(normalizer_state)
    identity_values = {
        "domain_id": str(payload["domain_id"]),
        "scenario_id": scenario_id,
        "segment_identity": x_t_identity,
        "clean_artifact_hash": clean_artifact_hash,
        "cache_artifact_hash": cache_artifact_hash,
        "expected_support_hash": expected_support_hash,
        "gmt_checkpoint_hash": gmt_checkpoint_hash,
        "gmt_normalizer_hash": gmt_normalizer_hash,
        "field_schema_id": str(payload["field_schema_id"]),
        "horizon_k": int(payload["horizon_k"]),
        "timestep_seconds": float(payload["timestep_seconds"]),
        "seed_protocol_id": str(payload["seed_protocol_id"]),
    }
    expected = payload.get("expected_identity")
    if expected is not None:
        if not isinstance(expected, dict) or frozenset(expected) != frozenset(identity_values):
            raise ValueError("clean calibration expected_identity schema mismatch")
        if any(expected[name] != value for name, value in identity_values.items()):
            raise RuntimeError("prepared clean calibration identity differs from the manifest binding")
    identity = CleanCalibrationCollectionIdentity(**identity_values)
    request = CleanCalibrationCollectionRequest(
        calibration_id=str(payload["calibration_id"]),
        identity=identity,
        repeats=tuple(
            CleanCalibrationRepeatSpec(repeat_id=str(item["repeat_id"]), seed=int(item["seed"]))
            for item in payload["repeats"]
        ),
        coverage=float(payload["coverage"]),
    )
    request.validate()
    return request


def _receipt_payload(receipt: ReadOnlyCleanCollectionReceipt) -> dict[str, Any]:
    return {
        "status": "OK",
        "route_id": FRONTRES_CLEAN_CALIBRATION_ROUTE_ID,
        "path_class": receipt.path_class,
        "request_hash": receipt.request_hash,
        "collection_hash": receipt.collection_hash,
        "training_state_before_hash": receipt.training_state_before_hash,
        "training_state_after_hash": receipt.training_state_after_hash,
        "rng_state_before_hash": receipt.rng_state_before_hash,
        "rng_state_after_hash": receipt.rng_state_after_hash,
        "repeat_ids": receipt.repeat_ids,
        "collected_count": receipt.collected_count,
        "collector_id": receipt.collector_id,
        "collector_version": receipt.collector_version,
        "calibration": receipt.calibration.hash_payload(),
    }


def _rng_hash() -> str:
    state: dict[str, Any] = {
        "python": repr(random.getstate()),
        "torch": bytes(torch.get_rng_state().tolist()),
    }
    if np is not None:
        state["numpy"] = repr(np.random.get_state())
    if torch.cuda.is_available():
        state["cuda"] = [bytes(value.tolist()) for value in torch.cuda.get_rng_state_all()]
    return _hash_payload(state)


@contextmanager
def _clean_repeat_seed(seed: int):
    """Seed one repeat while restoring every process RNG at the boundary."""

    python_state = random.getstate()
    numpy_state = np.random.get_state() if np is not None else None
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(int(seed))
    if np is not None:
        np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        yield
    finally:
        random.setstate(python_state)
        if np is not None and numpy_state is not None:
            np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _clean_calibration_representative_row(
    request: CleanCalibrationCollectionRequest,
    plan: Any,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    scenario_id = request.identity.scenario_id
    scenario_ids = tuple(str(value) for value in getattr(plan, "scenario_ids", ()))
    source_index = getattr(plan, "source_index", None)
    horizon_k = getattr(plan, "horizon_k", None)
    if not isinstance(source_index, torch.Tensor) or not isinstance(horizon_k, torch.Tensor):
        raise RuntimeError("clean calibration producer requires typed source_index/horizon_k plan rows")
    if source_index.device != horizon_k.device:
        raise RuntimeError("clean calibration sealed plan tensors must share one device")
    plan_device = source_index.device
    rows = torch.tensor(
        [index for index, value in enumerate(scenario_ids) if value == scenario_id],
        device=plan_device,
        dtype=torch.long,
    )
    if int(rows.numel()) <= 0:
        raise RuntimeError(f"clean calibration Scenario {scenario_id!r} is absent from the sealed plan")
    k_values = torch.unique(horizon_k.index_select(0, rows).to(dtype=torch.long))
    if int(k_values.numel()) != 1 or int(k_values[0].item()) != int(request.identity.horizon_k):
        raise RuntimeError("clean calibration request K differs from the sealed plan")
    return source_index.index_select(0, rows[:1]).to(device=device, dtype=torch.long), int(k_values[0].item())


def _collect_raw_clean_collection(
    runner: Any,
    request: CleanCalibrationCollectionRequest,
    prepared: FrontRESCleanCalibrationPreparedOwner,
) -> ReadOnlyCleanCollection:
    """Collect repeated Clean windows through the existing K-step owner."""

    request.validate()
    prepared.validate()
    plan = prepared.plan
    device = torch.device(getattr(runner, "device", "cpu"))
    authoritative_rows, active_k = _clean_calibration_representative_row(request, plan, device=device)
    pair_layout = prepared.pair_layout
    if pair_layout is None:
        mode = resolve_frontres_mode_state(runner, FrontRESActorCritic)
        pair_layout = configure_frontres_pair_layout(runner, is_frontres=mode.is_frontres)

    campaign_state_before = _state_hash(runner, request.calibration_id)
    campaign_rng_before = _rng_hash()
    reference: FrontRESCleanRawWindow | None = None
    windows: list[ReadOnlyCleanWindow] = []
    closed_repeat_ids: list[str] = []
    try:
        for repeat in request.repeats:
            repeat_state_before = _state_hash(runner, request.calibration_id)
            repeat_rng_before = _rng_hash()
            with _clean_repeat_seed(repeat.seed):
                apply_frontres_current_segment_reset(
                    runner,
                    pair_layout=pair_layout,
                    local_scenario_execution_mode="clean_baseline",
                )
                trajectory, expected_support = collect_frontres_v017_no_actor_baseline(
                    runner,
                    horizon_k=active_k,
                    authoritative_rows=authoritative_rows,
                )
            raw_window = FrontRESCleanRawWindow(
                repeat_id=repeat.repeat_id,
                trajectory=trajectory,
                expected_support=expected_support.detach().clone(),
                timestep_seconds=request.identity.timestep_seconds,
            )
            if reference is None:
                reference = raw_window
            measurement = build_clean_calibration_measurement(
                reference=reference,
                candidate=raw_window,
                domain_id=request.identity.domain_id,
                scenario_id=request.identity.scenario_id,
            )
            repeat_state_after = _state_hash(runner, request.calibration_id)
            repeat_rng_after = _rng_hash()
            if repeat_state_before != repeat_state_after or repeat_rng_before != repeat_rng_after:
                raise RuntimeError("clean repeat changed protected state or failed RNG restoration")
            repeat_seed_hash = _hash_payload(
                {
                    "seed_protocol_id": request.identity.seed_protocol_id,
                    "repeat_id": repeat.repeat_id,
                    "seed": repeat.seed,
                }
            )
            windows.append(
                ReadOnlyCleanWindow(
                    observation=measurement.observation,
                    identity=request.identity,
                    repeat_seed=repeat.seed,
                    repeat_seed_hash=repeat_seed_hash,
                    training_state_hash=campaign_state_before,
                    rng_restore_hash=campaign_rng_before,
                    collector_id=FRONTRES_CLEAN_CALIBRATION_COLLECTOR_ID,
                    collector_version=FRONTRES_CLEAN_CALIBRATION_COLLECTOR_VERSION,
                    hard_events=measurement.hard_events,
                )
            )
            closed_repeat_ids.append(repeat.repeat_id)
    finally:
        # The outer read-only scope remains the lifecycle owner.  This block
        # only prevents a partially collected result from escaping.
        if len(closed_repeat_ids) != len(windows):
            windows.clear()

    campaign_state_after = _state_hash(runner, request.calibration_id)
    campaign_rng_after = _rng_hash()
    collection = ReadOnlyCleanCollection(
        windows=tuple(windows),
        training_state_before_hash=campaign_state_before,
        training_state_after_hash=campaign_state_after,
        rng_state_before_hash=campaign_rng_before,
        rng_state_after_hash=campaign_rng_after,
        closed_repeat_ids=tuple(closed_repeat_ids),
        collector_id=FRONTRES_CLEAN_CALIBRATION_COLLECTOR_ID,
        collector_version=FRONTRES_CLEAN_CALIBRATION_COLLECTOR_VERSION,
    )
    collection.validate(request)
    return collection


def _state_hash(runner: Any, transaction_id: str) -> str:
    """Use the runtime's combined immutable-state owner; never silently fall back."""

    owner = getattr(runner, "frontres_clean_calibration_state_fingerprint", None)
    if callable(owner):
        value = owner(str(transaction_id))
        if not isinstance(value, str) or not value:
            raise RuntimeError("clean calibration state owner returned an invalid fingerprint")
        return value

    alg = getattr(runner, "alg", None)
    runtime_owner = (
        getattr(runner, "frontres_runtime", None)
        or getattr(runner, "_frontres_runtime", None)
        or getattr(runner, "_frontres_deployment_runtime", None)
    )
    if alg is not None:
        if runtime_owner is None or not callable(getattr(runtime_owner, "training_state_fingerprint", None)):
            raise RuntimeError(
                "clean calibration requires the official runtime state owner with training_state_fingerprint()"
            )
        policy_snapshot = capture_frontres_frozen_policy_snapshot(
            runner,
            transaction_id=str(transaction_id),
        )
        runtime_state = runtime_owner.training_state_fingerprint()
        policy = getattr(alg, "policy", None)
        if policy is None or not callable(getattr(policy, "modules", None)):
            raise RuntimeError("clean calibration requires the Actor/GMT policy mode owner")
        policy_modes = tuple(
            (type(module).__qualname__, bool(module.training)) for module in policy.modules()
        )
        return _hash_payload(
            {
                "policy_snapshot": vars(policy_snapshot),
                "runtime_state": runtime_state,
                "policy_modes": policy_modes,
            }
        )

    raise RuntimeError(
        "clean calibration requires a concrete state owner combining policy, optimizer, "
        "storage, sampler, normalizers and mode fingerprints"
    )


def compute_frontres_clean_calibration_state_fingerprint(runner: Any, transaction_id: str) -> str:
    """Build the official protected-state identity for this read-only route."""

    alg = getattr(runner, "alg", None)
    policy = getattr(alg, "policy", None)
    if alg is None or policy is None:
        raise RuntimeError("clean calibration state fingerprint requires runner.alg.policy")
    aggregate = frontres_stage3_transaction_aggregate(runner)
    snapshot = capture_frontres_frozen_policy_snapshot(runner, transaction_id=str(transaction_id))

    def state_dict(value: Any) -> Any:
        fn = getattr(value, "state_dict", None)
        return fn() if callable(fn) else value

    module_roots = (
        ("policy", policy),
        ("prefix_normalizer", getattr(runner, "_frontres_extra_normalizer", None)),
        ("gmt_normalizer", getattr(runner, "obs_normalizer", None)),
        ("privileged_normalizer", getattr(runner, "privileged_obs_normalizer", None)),
        ("teacher_normalizer", getattr(runner, "teacher_obs_normalizer", None)),
    )
    module_modes = tuple(
        (root_name, name, bool(module.training))
        for root_name, root in module_roots
        if root is not None and callable(getattr(root, "named_modules", None))
        for name, module in root.named_modules()
    )
    values = {
        "route_id": FRONTRES_CLEAN_CALIBRATION_ROUTE_ID,
        "transaction_id": str(transaction_id),
        "policy_snapshot": vars(snapshot),
        "policy": state_dict(policy),
        "optimizer": state_dict(getattr(alg, "optimizer", None)),
        "prefix_normalizer": state_dict(getattr(runner, "_frontres_extra_normalizer", None)),
        "gmt_normalizer": state_dict(getattr(runner, "obs_normalizer", None)),
        "privileged_normalizer": state_dict(getattr(runner, "privileged_obs_normalizer", None)),
        "teacher_normalizer": state_dict(getattr(runner, "teacher_obs_normalizer", None)),
        "sampler": state_dict(getattr(runner, "_frontres_segment_sampler", None)),
        "outer_replay": state_dict(getattr(runner, "_frontres_outer_scenario_replay", None)),
        "transaction": state_dict(aggregate),
        "curriculum": {
            name: value
            for name, value in vars(runner).items()
            if name.startswith("_frontres_curriculum_") or name.startswith("_frontres_dr_")
        },
        "module_modes": module_modes,
        "iteration": getattr(runner, "current_learning_iteration", None),
    }
    return _hash_payload(values)


@dataclass(frozen=True)
class FrontRESCleanCalibrationPreparedOwner:
    sample: Any
    batch: Any
    plan: Any
    pair_layout: Any | None = None

    @classmethod
    def from_prepared(cls, prepared: Any) -> "FrontRESCleanCalibrationPreparedOwner":
        return cls(
            sample=getattr(prepared, "sample", None),
            batch=getattr(prepared, "batch", None),
            plan=getattr(prepared, "plan", None),
            pair_layout=getattr(prepared, "pair_layout", None),
        )

    def validate(self) -> None:
        if self.sample is None or self.batch is None or not callable(getattr(self.plan, "validate", None)):
            raise TypeError("clean calibration prepared owner requires sample, batch and plan")


@dataclass(frozen=True)
class FrontRESCleanCalibrationGatewayInput:
    request: CleanCalibrationCollectionRequest
    prepared: FrontRESCleanCalibrationPreparedOwner
    collection: ReadOnlyCleanCollection
    route_id: str = FRONTRES_CLEAN_CALIBRATION_ROUTE_ID

    def validate(self) -> None:
        if not isinstance(self.request, CleanCalibrationCollectionRequest):
            raise TypeError("clean calibration gateway requires a typed collection request")
        if not isinstance(self.collection, ReadOnlyCleanCollection):
            raise TypeError("clean calibration gateway requires a typed collection result")
        if self.route_id != FRONTRES_CLEAN_CALIBRATION_ROUTE_ID:
            raise ValueError("clean calibration gateway route identity mismatch")
        if not isinstance(self.prepared, FrontRESCleanCalibrationPreparedOwner):
            raise TypeError("clean calibration gateway requires a typed prepared owner")
        self.prepared.validate()


def collect_frontres_clean_calibration_gateway(
    runner: Any,
    gateway_input: FrontRESCleanCalibrationGatewayInput,
) -> ReadOnlyCleanCollectionReceipt:
    """Run the existing read-only/reset lifecycle and consume typed telemetry once."""

    gateway_input.validate()
    request = gateway_input.request
    prepared = gateway_input.prepared
    before_state = _state_hash(runner, request.calibration_id)
    before_rng = _rng_hash()
    collection_error: BaseException | None = None
    result: ReadOnlyCleanCollectionReceipt | None = None
    try:
        with frontres_readonly_collection_scope(runner, route=FRONTRES_CLEAN_CALIBRATION_ROUTE):
            bind_frontres_collection_context(
                runner,
                route=FRONTRES_CLEAN_CALIBRATION_ROUTE,
                sample=prepared.sample,
                batch=prepared.batch,
            )
            prepared.plan.validate()
            prepare_frontres_raw_contact_views(runner)
            apply_frontres_current_segment_reset(
                runner,
                pair_layout=getattr(prepared, "pair_layout", None),
                local_scenario_execution_mode="clean_baseline",
            )
            result = adapt_read_only_clean_collection(request, gateway_input.collection)
    except BaseException as exc:
        collection_error = exc
    finally:
        after_state = _state_hash(runner, request.calibration_id)
        after_rng = _rng_hash()
        if after_state != before_state or after_rng != before_rng:
            mutation_error = RuntimeError(
                "clean calibration mutated protected state or failed to restore RNG: "
                f"state_before={before_state} state_after={after_state} "
                f"rng_before={before_rng} rng_after={after_rng}"
            )
            collection_error = mutation_error if collection_error is None else RuntimeError(
                f"{collection_error}; {mutation_error}"
            )
    if collection_error is not None:
        raise collection_error
    if result is None:  # pragma: no cover - defensive fail closed
        raise RuntimeError("clean calibration gateway completed without a receipt")
    result.validate_for_request(request)
    return result


def collect_frontres_clean_calibration_raw_gateway(
    runner: Any,
    *,
    request: CleanCalibrationCollectionRequest,
    prepared: FrontRESCleanCalibrationPreparedOwner,
) -> ReadOnlyCleanCollectionReceipt:
    """Official producer path: reset, repeated Clean K-rollouts, then adapter."""

    request.validate()
    prepared.validate()
    before_state = _state_hash(runner, request.calibration_id)
    before_rng = _rng_hash()
    collection_error: BaseException | None = None
    result: ReadOnlyCleanCollectionReceipt | None = None
    try:
        with frontres_readonly_collection_scope(runner, route=FRONTRES_CLEAN_CALIBRATION_ROUTE):
            bind_frontres_collection_context(
                runner,
                route=FRONTRES_CLEAN_CALIBRATION_ROUTE,
                sample=prepared.sample,
                batch=prepared.batch,
            )
            prepared.plan.validate()
            prepare_frontres_raw_contact_views(runner)
            collection = _collect_raw_clean_collection(runner, request, prepared)
            result = adapt_read_only_clean_collection(request, collection)
    except BaseException as exc:
        collection_error = exc
    finally:
        after_state = _state_hash(runner, request.calibration_id)
        after_rng = _rng_hash()
        if after_state != before_state or after_rng != before_rng:
            mutation_error = RuntimeError(
                "clean calibration raw gateway mutated protected state or failed RNG restoration: "
                f"state_before={before_state} state_after={after_state} "
                f"rng_before={before_rng} rng_after={after_rng}"
            )
            collection_error = mutation_error if collection_error is None else RuntimeError(
                f"{collection_error}; {mutation_error}"
            )
    if collection_error is not None:
        raise collection_error
    if result is None:
        raise RuntimeError("clean calibration raw gateway completed without a receipt")
    result.validate_for_request(request)
    return result


def collect_frontres_clean_calibration_from_manifest(
    runner: Any,
    *,
    manifest_path: str,
    result_path: str,
) -> dict[str, object]:
    """Construct the official typed request and run one read-only raw campaign."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    result_file = Path(result_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"clean calibration manifest does not exist: {manifest_file}")
    if result_file.exists():
        raise RuntimeError(f"clean calibration refuses existing result identity: {result_file}")
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest = _strict_clean_manifest(payload)
    prepared: FrontRESCleanCalibrationPreparedOwner | None = None
    try:
        ensure_frontres_readonly_reset_support(runner)
        prepared_raw = prepare_frontres_fixed_k_m4_evaluation_batch(
            runner,
            _manifest_items(manifest),
            attempts_per_segment=4,
            allowed_horizons=(int(manifest["horizon_k"]),),
            transaction_namespace=FRONTRES_CLEAN_CALIBRATION_ROUTE_ID,
            route_label="FRS-EVAL-v010 clean calibration",
        )
        prepared = FrontRESCleanCalibrationPreparedOwner.from_prepared(prepared_raw)
        request = _build_request_from_prepared(runner, manifest, prepared)
        typed_connector = getattr(runner, "run_frontres_clean_calibration_collect_typed", None)
        if not callable(typed_connector):
            raise RuntimeError("clean calibration requires the official typed composition-root connector")
        receipt = typed_connector(request=request, prepared=prepared)
        result = _receipt_payload(receipt)
        result_file.parent.mkdir(parents=True, exist_ok=True)
        write_frontres_atomic_json(result_file, result)
        return result
    except BaseException:
        if prepared is not None:
            close_frontres_local_scenarios(prepared.batch)
        raise


__all__ = (
    "FRONTRES_CLEAN_CALIBRATION_ROUTE",
    "FRONTRES_CLEAN_CALIBRATION_ROUTE_ID",
    "FrontRESCleanCalibrationGatewayInput",
    "FrontRESCleanCalibrationPreparedOwner",
    "collect_frontres_clean_calibration_gateway",
    "collect_frontres_clean_calibration_raw_gateway",
    "collect_frontres_clean_calibration_from_manifest",
    "compute_frontres_clean_calibration_state_fingerprint",
)
