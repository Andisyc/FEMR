from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "rsl_rl" / "frontres" / "frontres_segment_sampler.py"
spec = importlib.util.spec_from_file_location("frontres_segment_sampler_fixed_noisy", MODULE_PATH)
sampler_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = sampler_module
spec.loader.exec_module(sampler_module)

FrontRESFixedNoisyScenarioLifecycle = sampler_module.FrontRESFixedNoisyScenarioLifecycle
FrontRESNoisyReferenceMaterialization = sampler_module.FrontRESNoisyReferenceMaterialization
FrontRESSegmentSample = sampler_module.FrontRESSegmentSample


def _sample() -> FrontRESSegmentSample:
    return FrontRESSegmentSample(
        segment_ids=torch.tensor([17, 17, 17, 17, 17], dtype=torch.long),
        source=("global",) * 5,
        priority=torch.zeros(5),
        staleness=torch.zeros(5),
        valid_mask=torch.ones(5, dtype=torch.bool),
        horizon_k=torch.tensor([3, 3, 3, 2, 2], dtype=torch.long),
        source_index=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
        trial_index=torch.tensor([0, 1, 2, 0, 1], dtype=torch.long),
    )


class _RecordingMaterializer:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, str]] = []

    def __call__(self, request):
        self.calls.append((request.segment_id, request.source_index, request.horizon_k, request.scenario_id))
        frames = request.required_frame_count
        reference = torch.arange(frames * 3, dtype=torch.float32).reshape(frames, 3)
        reference += float(100 * request.source_index)
        return FrontRESNoisyReferenceMaterialization(
            reference_sequence=reference,
            provenance={"factory": "semantic_toy", "source_index": request.source_index},
        )


def test_one_materialization_per_source_and_immutable_trial_reuse() -> None:
    materializer = _RecordingMaterializer()
    lifecycle = FrontRESFixedNoisyScenarioLifecycle(
        transaction_id="tx-fixed-noisy-7",
        future_offsets=(1, 2),
        materialize_reference=materializer,
    )

    rows = lifecycle.bind_rows(_sample())
    first = rows.scenario_for_row(0)
    second_source = rows.scenario_for_row(3)

    assert len(materializer.calls) == 2
    assert rows.scenario_for_row(0) is rows.scenario_for_row(1) is rows.scenario_for_row(2)
    assert rows.scenario_for_row(3) is rows.scenario_for_row(4)
    assert first.scenario_id != second_source.scenario_id
    assert first.noisy_segment_hash != second_source.noisy_segment_hash
    assert first.required_frame_count == 5
    assert second_source.required_frame_count == 4
    assert first.reference_sequence.shape == (5, 3)
    assert second_source.reference_sequence.shape == (4, 3)

    visible_copy = first.reference_sequence
    visible_copy.fill_(-999.0)
    assert float(first.reference_sequence[0, 0].item()) == 0.0

    rebound = lifecycle.bind_rows(_sample())
    assert len(materializer.calls) == 2
    assert rebound.scenario_for_row(0) is first
    assert rebound.scenario_for_row(3) is second_source

    print(
        "[probe fixed_noisy_lifecycle] "
        f"base_scenarios={len(materializer.calls)} "
        f"rows={rows.batch_size} "
        f"scenario_ids={rows.scenario_ids} "
        f"hashes={[value[:12] for value in rows.noisy_segment_hashes]} "
        f"coverage={rows.required_frame_counts.tolist()} "
        f"source_index={rows.source_index.tolist()} "
        f"trial_index={rows.trial_index.tolist()}",
        flush=True,
    )


def test_closed_scenario_keeps_evidence_but_cannot_be_rematerialized() -> None:
    materializer = _RecordingMaterializer()
    lifecycle = FrontRESFixedNoisyScenarioLifecycle(
        transaction_id="tx-close-9",
        future_offsets=(1, 2),
        materialize_reference=materializer,
    )
    rows = lifecycle.bind_rows(_sample())
    scenario = rows.scenario_for_row(0)

    closed = lifecycle.close_scenario(scenario.scenario_id)
    assert closed is scenario
    assert lifecycle.closed_scenario(scenario.scenario_id) is scenario

    try:
        lifecycle.bind_rows(_sample())
    except RuntimeError as exc:
        assert scenario.scenario_id in str(exc)
    else:
        raise AssertionError("closed scenario identity must reject a new materialization")
    assert len(materializer.calls) == 2


def test_rejects_insufficient_coverage_and_clean_payload() -> None:
    def short_materializer(request):
        return FrontRESNoisyReferenceMaterialization(
            reference_sequence=torch.zeros(request.required_frame_count - 1, 3),
            provenance={"factory": "too_short"},
        )

    short_lifecycle = FrontRESFixedNoisyScenarioLifecycle(
        transaction_id="tx-short-1",
        future_offsets=(1, 2),
        materialize_reference=short_materializer,
    )
    try:
        short_lifecycle.bind_rows(_sample())
    except ValueError as exc:
        assert "coverage" in str(exc)
    else:
        raise AssertionError("coverage shorter than K + H_max must fail")

    def contaminated_materializer(request):
        return FrontRESNoisyReferenceMaterialization(
            reference_sequence=torch.zeros(request.required_frame_count, 3),
            provenance={"clean_reference": "forbidden"},
        )

    contaminated_lifecycle = FrontRESFixedNoisyScenarioLifecycle(
        transaction_id="tx-clean-2",
        future_offsets=(1, 2),
        materialize_reference=contaminated_materializer,
    )
    try:
        contaminated_lifecycle.bind_rows(_sample())
    except ValueError as exc:
        assert "Clean" in str(exc)
    else:
        raise AssertionError("scenario provenance must not carry a Clean reference")


def test_rejects_conflicting_source_shape() -> None:
    sample = _sample()
    sample = FrontRESSegmentSample(
        segment_ids=sample.segment_ids,
        source=sample.source,
        priority=sample.priority,
        staleness=sample.staleness,
        valid_mask=sample.valid_mask,
        horizon_k=torch.tensor([3, 4, 3, 2, 2], dtype=torch.long),
        source_index=sample.source_index,
        trial_index=sample.trial_index,
    )
    lifecycle = FrontRESFixedNoisyScenarioLifecycle(
        transaction_id="tx-conflict-3",
        future_offsets=(1, 2),
        materialize_reference=_RecordingMaterializer(),
    )
    try:
        lifecycle.bind_rows(sample)
    except ValueError as exc:
        assert "source_index=0" in str(exc)
    else:
        raise AssertionError("one source index must have one Segment and K identity")


def main() -> None:
    test_one_materialization_per_source_and_immutable_trial_reuse()
    test_closed_scenario_keeps_evidence_but_cannot_be_rematerialized()
    test_rejects_insufficient_coverage_and_clean_payload()
    test_rejects_conflicting_source_shape()
    print("frontres_fixed_noisy_segment_lifecycle_contract: ok")


if __name__ == "__main__":
    main()
