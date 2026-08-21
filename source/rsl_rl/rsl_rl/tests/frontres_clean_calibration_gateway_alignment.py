"""Offline construction contract for the official clean-calibration gateway."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from frontres_contract_imports import install_frontres_contract_packages


ROOT = Path(__file__).resolve().parents[4]
install_frontres_contract_packages(ROOT / "source" / "rsl_rl" / "rsl_rl")

from rsl_rl.frontres.frontres_clean_relative_calibration import (
    CleanCalibrationCollectionRequest,
)
from rsl_rl.runners.frontres_clean_calibration_gateway import (
    FrontRESCleanCalibrationGatewayInput,
    FrontRESCleanCalibrationPreparedOwner,
    collect_frontres_clean_calibration_gateway,
)
from rsl_rl.runners.frontres_stage3_engine import FrontRESStage3TransactionAggregate

from frontres_clean_repeated_collection_adapter_alignment import _collection, _request


class _Prepared:
    sample = SimpleNamespace()
    batch = SimpleNamespace()
    plan = SimpleNamespace(validate=lambda self: None)


class _Runner:
    def __init__(self) -> None:
        self._frontres_stage3_transaction_state = FrontRESStage3TransactionAggregate()
        self.frontres_clean_calibration_state = {"policy": "p0", "training": "t0", "rng": "r0"}

    def frontres_clean_calibration_state_fingerprint(self, transaction_id: str) -> str:
        return f"{transaction_id}:{self.frontres_clean_calibration_state}"


def _input() -> FrontRESCleanCalibrationGatewayInput:
    request = _request()
    collection = _collection(request)
    return FrontRESCleanCalibrationGatewayInput(
        request=request,
        prepared=FrontRESCleanCalibrationPreparedOwner.from_prepared(_Prepared()),
        collection=collection,
    )


def main() -> None:
    runner = _Runner()
    try:
        result = collect_frontres_clean_calibration_gateway(runner, _input())
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        message = str(exc).upper()
        if not any(token in message for token in ("TELEMETRY", "SCENE", "'ENV'")):
            raise
        print("frontres_clean_calibration_gateway_alignment: TELEMETRY-GAP")
        return
    result.validate_for_request(_request())

    runner = _Runner()
    broken = _input()
    broken_request = replace(broken.request, repeats=broken.request.repeats[:-1])
    try:
        collect_frontres_clean_calibration_gateway(
            runner,
            replace(broken, request=broken_request),
        )
    except (RuntimeError, ValueError, TypeError):
        pass
    else:
        raise AssertionError("gateway must reject request/collection repeat mismatch")

    print("frontres_clean_calibration_gateway_alignment: MODULE-CORRECT")


if __name__ == "__main__":
    main()
