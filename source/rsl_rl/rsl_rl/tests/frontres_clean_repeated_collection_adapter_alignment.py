#!/usr/bin/env python3
"""Offline contract tests for the typed repeated-Clean collection adapter.

The fixture is a completed gateway result. It does not reimplement Clean
metrics, Gain, training, or the active FRS-GAIN-v009 route.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

from rsl_rl.frontres.frontres_clean_relative_calibration import (
    CleanCalibrationCollectionIdentity,
    CleanCalibrationCollectionRequest,
    CleanCalibrationObservation,
    CleanCalibrationRepeatSpec,
    CleanHardEventEvidence,
    ReadOnlyCleanCollection,
    ReadOnlyCleanWindow,
    adapt_read_only_clean_collection,
)


def _identity() -> CleanCalibrationCollectionIdentity:
    return CleanCalibrationCollectionIdentity(
        domain_id="gmt-pseudo-domain",
        scenario_id="scenario-7",
        segment_identity="motion=walk|frame=42|segment=7|H=8",
        clean_artifact_hash="a" * 64,
        cache_artifact_hash="b" * 64,
        expected_support_hash="c" * 64,
        gmt_checkpoint_hash="d" * 64,
        gmt_normalizer_hash="e" * 64,
        field_schema_id="frontres-clean-relative-fields-v1",
        horizon_k=8,
        timestep_seconds=0.02,
        seed_protocol_id="same-state-clean-repeat-rng-v1",
    )


def _request() -> CleanCalibrationCollectionRequest:
    return CleanCalibrationCollectionRequest(
        calibration_id="clean-collection-adapter-pseudo-v2",
        identity=_identity(),
        repeats=tuple(CleanCalibrationRepeatSpec(f"repeat-{index}", 100 + index) for index in range(3)),
        coverage=0.95,
    )


def _observation(request: CleanCalibrationCollectionRequest, repeat_id: str) -> CleanCalibrationObservation:
    index = int(repeat_id.rsplit("-", 1)[1])
    offset = 0.001 * index
    return CleanCalibrationObservation(
        domain_id=request.identity.domain_id,
        scenario_id=request.identity.scenario_id,
        repeat_id=repeat_id,
        capture_margin=0.04 + offset,
        capture_margin_trend=0.01 + offset,
        zmp_applicable=True,
        zmp_margin=0.03 + offset,
        linear_momentum_error=0.04 + offset,
        angular_momentum_error=0.08 + offset,
        support_drift=0.02 + offset,
    )


def _window(request: CleanCalibrationCollectionRequest, repeat: CleanCalibrationRepeatSpec) -> ReadOnlyCleanWindow:
    repeat_seed_hash = hashlib.sha256(
        json.dumps(
            {
                "seed_protocol_id": request.identity.seed_protocol_id,
                "repeat_id": repeat.repeat_id,
                "seed": repeat.seed,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return ReadOnlyCleanWindow(
        observation=_observation(request, repeat.repeat_id),
        identity=request.identity,
        repeat_seed=repeat.seed,
        repeat_seed_hash=repeat_seed_hash,
        training_state_hash="1" * 64,
        rng_restore_hash="2" * 64,
        collector_id="formal-clean-gateway-pseudo",
        collector_version="v1",
        hard_events=CleanHardEventEvidence(
            survival_ok=True,
            survival_failure_duration=0.0,
            expected_support_no_load=0.0,
            unplanned_support_switch=0.0,
            illegal_contact_duration=0.0,
            valid_step_count=request.identity.horizon_k,
            zmp_applicable_step_count=request.identity.horizon_k,
        ),
    )


def _collection(request: CleanCalibrationCollectionRequest) -> ReadOnlyCleanCollection:
    repeat_ids = tuple(repeat.repeat_id for repeat in request.repeats)
    return ReadOnlyCleanCollection(
        windows=tuple(_window(request, repeat) for repeat in request.repeats),
        training_state_before_hash="1" * 64,
        training_state_after_hash="1" * 64,
        rng_state_before_hash="2" * 64,
        rng_state_after_hash="2" * 64,
        closed_repeat_ids=repeat_ids,
        collector_id="formal-clean-gateway-pseudo",
        collector_version="v1",
    )


def _assert_rejected(callable_, message: str) -> None:
    try:
        callable_()
    except (RuntimeError, TypeError, ValueError):
        return
    raise AssertionError(message)


def main() -> None:
    request = _request()
    collection = _collection(request)
    receipt = adapt_read_only_clean_collection(request, collection)
    receipt.validate_for_request(request)
    assert receipt.collected_count == 3
    assert receipt.repeat_ids == tuple(repeat.repeat_id for repeat in request.repeats)
    assert receipt.calibration.repeated_sample_count == 3
    assert receipt.calibration.repeated_pair_count == 3
    assert receipt.calibration.request_hash == receipt.request_hash
    assert receipt.training_state_before_hash == receipt.training_state_after_hash
    assert receipt.rng_state_before_hash == receipt.rng_state_after_hash

    bad_request = replace(
        request,
        repeats=(
            CleanCalibrationRepeatSpec("repeat-0", 100),
            CleanCalibrationRepeatSpec("repeat-2", 102),
            CleanCalibrationRepeatSpec("repeat-2", 102),
        ),
    )
    _assert_rejected(lambda: adapt_read_only_clean_collection(bad_request, collection), "duplicate repeats must fail closed")
    _assert_rejected(
        lambda: adapt_read_only_clean_collection(
            request, replace(collection, closed_repeat_ids=("repeat-0", "repeat-1"))
        ),
        "partial cleanup must fail closed",
    )

    drifted_identity = replace(request.identity, cache_artifact_hash="f" * 64)
    drifted_window = replace(collection.windows[1], identity=drifted_identity)
    _assert_rejected(
        lambda: adapt_read_only_clean_collection(
            request, replace(collection, windows=(collection.windows[0], drifted_window, collection.windows[2]))
        ),
        "cache identity drift must fail closed",
    )

    _assert_rejected(
        lambda: adapt_read_only_clean_collection(
            request, replace(collection, training_state_after_hash="3" * 64)
        ),
        "training-state drift must fail closed",
    )
    _assert_rejected(
        lambda: adapt_read_only_clean_collection(
            request, replace(collection, rng_state_after_hash="4" * 64)
        ),
        "RNG drift must fail closed",
    )

    tampered_request = replace(request, calibration_id="other-calibration")
    _assert_rejected(
        lambda: receipt.validate_for_request(tampered_request),
        "receipt must remain bound to its request identity",
    )

    print("frontres_clean_repeated_collection_adapter_alignment: MODULE-CORRECT")


if __name__ == "__main__":
    main()
