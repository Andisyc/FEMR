"""Focused contract for the v010 official manifest-to-gateway connector."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from rsl_rl.runners.frontres_clean_calibration_gateway import (
    FRONTRES_CLEAN_CALIBRATION_ROUTE_ID,
    collect_frontres_clean_calibration_from_manifest,
)


def main() -> None:
    payload = {
        "route_id": FRONTRES_CLEAN_CALIBRATION_ROUTE_ID,
        "calibration_id": FRONTRES_CLEAN_CALIBRATION_ROUTE_ID,
        "domain_id": "frontres-stage3",
        "field_schema_id": "frontres-clean-calibration-fields-v1",
        "horizon_k": 8,
        "timestep_seconds": 0.02,
        "seed_protocol_id": "clean-repeat-v1",
        "coverage": 0.95,
        "scenario_source_index": 0,
        "repeats": [
            {"repeat_id": "repeat-00", "seed": 100},
            {"repeat_id": "repeat-01", "seed": 101},
        ],
        "segments": [
            {
                "item_id": "segment-00",
                "motion_id": "motion-a.npz",
                "start_frame": 10,
                "perturbation_family": "local_rp",
                "perturbation_parameters": [["strength", 0.0]],
                "effective_horizon_k": 8,
                "seed": 100,
            },
            {
                "item_id": "segment-01",
                "motion_id": "motion-b.npz",
                "start_frame": 20,
                "perturbation_family": "local_rp",
                "perturbation_parameters": [["strength", 0.0]],
                "effective_horizon_k": 8,
                "seed": 101,
            },
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.json"
        result = Path(tmp) / "result.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        try:
            collect_frontres_clean_calibration_from_manifest(
                object(), manifest_path=str(manifest), result_path=str(result)
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            message = str(exc).lower()
            if not any(token in message for token in ("typed", "runner", "stage-1", "dataset")):
                raise
            if result.exists():
                raise AssertionError("official route must not emit a partial result on preflight failure")
            print("frontres_clean_calibration_official_route_alignment: ROUTE_REACHED_TYPED_PREPARE")
        else:
            raise AssertionError("official clean-calibration route must reach typed owner, not silently preflight")

    # The preparation owner has already materialized scenarios here.  A later
    # identity/connector failure must close that batch before returning.
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.json"
        result = Path(tmp) / "result.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        prepared_batch = object()
        prepared = SimpleNamespace(
            sample=object(),
            batch=prepared_batch,
            plan=SimpleNamespace(validate=lambda: None),
        )
        closed: list[object] = []
        with (
            patch(
                "rsl_rl.runners.frontres_clean_calibration_gateway.prepare_frontres_fixed_k_m4_evaluation_batch",
                return_value=prepared,
            ),
            patch(
                "rsl_rl.runners.frontres_clean_calibration_gateway._build_request_from_prepared",
                side_effect=RuntimeError("identity mismatch"),
            ),
            patch(
                "rsl_rl.runners.frontres_clean_calibration_gateway.close_frontres_local_scenarios",
                side_effect=lambda batch: closed.append(batch),
            ),
        ):
            try:
                collect_frontres_clean_calibration_from_manifest(
                    object(), manifest_path=str(manifest), result_path=str(result)
                )
            except RuntimeError as exc:
                assert "identity mismatch" in str(exc)
            else:
                raise AssertionError("identity failure must abort the official route")
        assert closed == [prepared_batch]
        assert not result.exists()
        print("frontres_clean_calibration_official_route_alignment: PREPARE_FAILURE_CLEANUP_PASS")


if __name__ == "__main__":
    main()
