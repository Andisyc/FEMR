#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ppo_module = _load("frontres_segment_ppo_grouped_contract", ROOT / "rsl_rl" / "algorithms" / "frontres_segment_ppo.py")
storage_module = _load(
    "frontres_segment_storage_grouped_contract",
    ROOT / "rsl_rl" / "frontres" / "frontres_segment_storage.py",
)

FrontRESSegmentPPOBatch = ppo_module.FrontRESSegmentPPOBatch
FrontRESSegmentPPOConfig = ppo_module.FrontRESSegmentPPOConfig
compute_frontres_segment_ppo_loss = ppo_module.compute_frontres_segment_ppo_loss
FrontRESSegmentStorageBatch = storage_module.FrontRESSegmentStorageBatch
FrontRESV015GroupedCandidateMetadata = storage_module.FrontRESV015GroupedCandidateMetadata


@dataclass(frozen=True)
class _TransactionMetadata:
    transaction_id: str
    policy_snapshot_id: str
    motion_ids: tuple[str, ...]
    segment_ids: torch.Tensor
    source_index: torch.Tensor
    trial_index: torch.Tensor
    horizon_k: torch.Tensor
    trial_role: tuple[str, ...]
    noisy_segment_hashes: tuple[str, ...]
    scenario_ids: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return int(self.segment_ids.numel())

    def validate(self) -> None:
        assert self.transaction_id
        assert self.policy_snapshot_id
        assert len(self.motion_ids) == self.batch_size
        assert len(self.trial_role) == self.batch_size
        assert len(self.noisy_segment_hashes) == self.batch_size
        assert len(self.scenario_ids) == self.batch_size
        for value in (self.source_index, self.trial_index, self.horizon_k):
            assert value.ndim == 1 and int(value.numel()) == self.batch_size

    def permute(self, order: torch.Tensor) -> "_TransactionMetadata":
        indices = [int(index) for index in order.tolist()]
        return _TransactionMetadata(
            transaction_id=self.transaction_id,
            policy_snapshot_id=self.policy_snapshot_id,
            motion_ids=tuple(self.motion_ids[index] for index in indices),
            segment_ids=self.segment_ids[order],
            source_index=self.source_index[order],
            trial_index=self.trial_index[order],
            horizon_k=self.horizon_k[order],
            trial_role=tuple(self.trial_role[index] for index in indices),
            noisy_segment_hashes=tuple(self.noisy_segment_hashes[index] for index in indices),
            scenario_ids=tuple(self.scenario_ids[index] for index in indices),
        )


class _ZeroRatioPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_prob_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.value_scale = torch.nn.Parameter(torch.tensor(0.0))

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        return {
            "log_prob": self.log_prob_scale * observations[:, 0],
            "value": self.value_scale * observations[:, 1],
            "entropy": torch.zeros_like(observations[:, 0]),
        }


def _metadata() -> _TransactionMetadata:
    return _TransactionMetadata(
        transaction_id="tx-grouped",
        policy_snapshot_id="tx-grouped:pi-old",
        motion_ids=(
            "motion-a",
            "motion-a",
            "motion-a",
            "motion-a",
            "motion-a",
            "motion-a",
            "motion-b",
            "motion-b",
        ),
        segment_ids=torch.tensor([10, 10, 10, 11, 11, 11, 20, 20], dtype=torch.long),
        source_index=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.long),
        trial_index=torch.tensor([0, 0, 1, 0, 0, 0, 0, 1], dtype=torch.long),
        horizon_k=torch.tensor([2, 2, 2, 3, 3, 3, 1, 3], dtype=torch.long),
        trial_role=("policy",) * 8,
        noisy_segment_hashes=("hash-a", "hash-a", "hash-a", "hash-b", "hash-b", "hash-b", "hash-c", "hash-c"),
        scenario_ids=("scenario-a", "scenario-a", "scenario-a", "scenario-b", "scenario-b", "scenario-b", "scenario-c", "scenario-c"),
    )


def _v015_metadata() -> FrontRESV015GroupedCandidateMetadata:
    return FrontRESV015GroupedCandidateMetadata(
        transaction_id="tx-v015-grouped",
        policy_snapshot_id="tx-v015-grouped:pi-old",
        motion_ids=(
            "motion-a",
            "motion-a",
            "motion-a",
            "motion-a",
            "motion-a",
            "motion-a",
            "motion-b",
            "motion-b",
        ),
        start_frames=torch.tensor([12, 12, 12, 24, 24, 24, 8, 8], dtype=torch.long),
        segment_ids=torch.tensor([10, 10, 10, 11, 11, 11, 20, 20], dtype=torch.long),
        source_index=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.long),
        trial_index=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long),
        horizon_k=torch.tensor([2, 2, 2, 3, 3, 3, 1, 1], dtype=torch.long),
        evidence_valid_step_count=torch.tensor([2, 2, 1, 3, 2, 3, 1, 1], dtype=torch.long),
        trial_role=("policy",) * 8,
        noisy_segment_hashes=("hash-a", "hash-a", "hash-a", "hash-b", "hash-b", "hash-b", "hash-c", "hash-c"),
        scenario_ids=("scenario-a", "scenario-a", "scenario-a", "scenario-b", "scenario-b", "scenario-b", "scenario-c", "scenario-c"),
        x_t_identities=("x-a", "x-a", "x-a", "x-b", "x-b", "x-b", "x-c", "x-c"),
        intent_q29_provenance="deployment_noisy_q29",
        intent_q29_source="motion_internal_q29",
    )


def _batch(
    metadata: _TransactionMetadata | None = None,
    *,
    advantages: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> FrontRESSegmentPPOBatch:
    metadata = _metadata() if metadata is None else metadata
    if advantages is None:
        advantages = torch.tensor([1.0, 3.0, 2.0, -4.0, -4.0, -4.0, 8.0, 100.0])
    if valid_mask is None:
        valid_mask = torch.tensor([True, True, True, True, True, True, True, False])
    batch_size = metadata.batch_size
    assert int(advantages.numel()) == batch_size
    assert int(valid_mask.numel()) == batch_size
    return FrontRESSegmentPPOBatch(
        observations=torch.stack((torch.arange(1, batch_size + 1, dtype=torch.float32), torch.ones(batch_size)), dim=1),
        actions=torch.zeros(batch_size, 6),
        old_log_probs=torch.zeros(batch_size),
        old_values=torch.zeros(batch_size),
        returns=torch.zeros(batch_size),
        advantages=advantages,
        valid_mask=valid_mask,
        segment_ids=metadata.segment_ids.clone(),
        transaction_metadata=metadata,
    )


def _metadata_with_duplicate_step_and_equivalent_attempt() -> _TransactionMetadata:
    base = _metadata()
    return _TransactionMetadata(
        transaction_id=base.transaction_id,
        policy_snapshot_id=base.policy_snapshot_id,
        motion_ids=(*base.motion_ids, "motion-a", "motion-a"),
        segment_ids=torch.cat((base.segment_ids, torch.tensor([10, 10], dtype=torch.long))),
        source_index=torch.cat((base.source_index, torch.tensor([0, 0], dtype=torch.long))),
        trial_index=torch.cat((base.trial_index, torch.tensor([1, 2], dtype=torch.long))),
        horizon_k=torch.cat((base.horizon_k, torch.tensor([2, 2], dtype=torch.long))),
        trial_role=(*base.trial_role, "policy", "policy"),
        noisy_segment_hashes=(*base.noisy_segment_hashes, "hash-a", "hash-a"),
        scenario_ids=(*base.scenario_ids, "scenario-a", "scenario-a"),
    )


def _manual_grouped_actor_loss(metadata: _TransactionMetadata, advantages: torch.Tensor, valid_mask: torch.Tensor) -> float:
    valid_rows = [index for index, flag in enumerate(valid_mask.tolist()) if flag]
    raw = advantages[valid_mask]
    transaction_rms = math.sqrt(float(raw.square().mean()))
    segment_rows: dict[tuple[str, int, int], list[int]] = {}
    attempt_rows: dict[tuple[str, int, int, int], list[int]] = {}
    for row in valid_rows:
        segment_key = (
            metadata.motion_ids[row],
            int(metadata.segment_ids[row]),
            int(metadata.source_index[row]),
        )
        attempt_key = (*segment_key, int(metadata.trial_index[row]))
        segment_rows.setdefault(segment_key, []).append(row)
        attempt_rows.setdefault(attempt_key, []).append(row)
    scaled: dict[int, float] = {}
    for segment_key, rows in segment_rows.items():
        segment_rms = math.sqrt(sum(float(advantages[row].square()) for row in rows) / len(rows))
        denominator = max(segment_rms, transaction_rms)
        for row in rows:
            scaled[row] = float(advantages[row]) / denominator if denominator > 0.0 else 0.0
    motion_segments: dict[str, list[tuple[str, int, int]]] = {}
    for segment_key in segment_rows:
        motion_segments.setdefault(segment_key[0], []).append(segment_key)
    motion_losses: list[float] = []
    for motion, segments in motion_segments.items():
        segment_losses: list[float] = []
        for segment_key in segments:
            attempts = [key for key in attempt_rows if key[:3] == segment_key]
            attempt_losses = [sum(-scaled[row] for row in attempt_rows[key]) / len(attempt_rows[key]) for key in attempts]
            segment_losses.append(sum(attempt_losses) / len(attempt_losses))
        motion_losses.append(sum(segment_losses) / len(segment_losses))
    return sum(motion_losses) / len(motion_losses)


def _grouped_cfg() -> FrontRESSegmentPPOConfig:
    return FrontRESSegmentPPOConfig(
        advantage_normalization="grouped_scale_only",
        value_loss_coef=0.0,
        entropy_coef=0.0,
    )


def _mass_by_key(keys: tuple[str, ...], shares: tuple[float, ...]) -> dict[str, float]:
    return dict(zip(keys, shares, strict=True))


def test_grouped_nested_reduction_is_hand_computed_and_row_permutation_invariant() -> None:
    metadata = _metadata()
    batch = _batch(metadata)
    result = compute_frontres_segment_ppo_loss(_ZeroRatioPolicy(), batch, _grouped_cfg())
    expected = _manual_grouped_actor_loss(metadata, batch.advantages, batch.valid_mask)

    torch.testing.assert_close(result.actor_loss.detach(), torch.tensor(expected), atol=1.0e-7, rtol=0.0)
    assert result.grouped_reduction_active
    assert result.grouped_motion_count == 2
    assert result.grouped_segment_count == 3
    assert result.grouped_attempt_count == 4
    assert result.grouped_valid_step_count == 7
    assert result.advantage_sign_flip_count == 0
    torch.testing.assert_close(torch.tensor(result.grouped_motion_mass_shares), torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(torch.tensor(result.grouped_attempt_mass_shares), torch.tensor([0.125, 0.125, 0.25, 0.5]))
    torch.testing.assert_close(
        torch.tensor(result.grouped_valid_step_mass_shares),
        torch.tensor([0.0625, 0.0625, 0.125, 1.0 / 12.0, 1.0 / 12.0, 1.0 / 12.0, 0.5]),
    )
    assert abs(sum(result.grouped_valid_step_mass_shares) - 1.0) < 1.0e-8

    order = torch.tensor([6, 2, 4, 0, 5, 1, 7, 3], dtype=torch.long)
    permuted_metadata = metadata.permute(order)
    permuted = FrontRESSegmentPPOBatch(
        observations=batch.observations[order],
        actions=batch.actions[order],
        old_log_probs=batch.old_log_probs[order],
        old_values=batch.old_values[order],
        returns=batch.returns[order],
        advantages=batch.advantages[order],
        valid_mask=batch.valid_mask[order],
        segment_ids=permuted_metadata.segment_ids.clone(),
        transaction_metadata=permuted_metadata,
    )
    permuted_result = compute_frontres_segment_ppo_loss(_ZeroRatioPolicy(), permuted, _grouped_cfg())

    torch.testing.assert_close(result.actor_loss.detach(), permuted_result.actor_loss.detach(), atol=1.0e-7, rtol=0.0)
    assert _mass_by_key(result.grouped_attempt_keys, result.grouped_attempt_mass_shares) == _mass_by_key(
        permuted_result.grouped_attempt_keys,
        permuted_result.grouped_attempt_mass_shares,
    )
    print(
        "[probe grouped_ppo] "
        f"actor_loss={result.actor_loss.detach().item():.9f} expected={expected:.9f} "
        f"motions={result.grouped_motion_count} segments={result.grouped_segment_count} "
        f"attempts={result.grouped_attempt_count} valid_steps={result.grouped_valid_step_count} "
        f"step_mass_sum={sum(result.grouped_valid_step_mass_shares):.9f} "
        f"txn_rms={result.grouped_transaction_advantage_rms:.9f} sign_flips={result.advantage_sign_flip_count}",
        flush=True,
    )


def test_grouped_scale_preserves_sign_and_does_not_amplify_low_scale_segment() -> None:
    metadata = _TransactionMetadata(
        transaction_id="tx-scale",
        policy_snapshot_id="tx-scale:pi-old",
        motion_ids=("low", "low", "high", "high"),
        segment_ids=torch.tensor([1, 1, 2, 2]),
        source_index=torch.tensor([0, 0, 1, 1]),
        trial_index=torch.tensor([0, 0, 0, 0]),
        horizon_k=torch.tensor([2, 2, 2, 2]),
        trial_role=("policy",) * 4,
        noisy_segment_hashes=("h-low", "h-low", "h-high", "h-high"),
        scenario_ids=("s-low", "s-low", "s-high", "s-high"),
    )
    batch = FrontRESSegmentPPOBatch(
        observations=torch.ones(4, 2),
        actions=torch.zeros(4, 6),
        old_log_probs=torch.zeros(4),
        old_values=torch.zeros(4),
        returns=torch.zeros(4),
        advantages=torch.tensor([1.0, -1.0, 10.0, -10.0]),
        valid_mask=torch.ones(4, dtype=torch.bool),
        segment_ids=metadata.segment_ids.clone(),
        transaction_metadata=metadata,
    )
    result = compute_frontres_segment_ppo_loss(_ZeroRatioPolicy(), batch, _grouped_cfg())

    expected_transaction_rms = math.sqrt(50.5)
    assert result.advantage_sign_flip_count == 0
    assert result.advantage_min < 0.0 < result.advantage_max
    assert abs(result.grouped_transaction_advantage_rms - expected_transaction_rms) < 1.0e-7
    assert result.grouped_segment_advantage_scales[0] == result.grouped_transaction_advantage_rms
    assert result.grouped_segment_advantage_scales[0] > result.grouped_segment_advantage_rms[0]
    assert result.grouped_segment_advantage_scales[1] == result.grouped_segment_advantage_rms[1]
    print(
        "[probe grouped_scale] "
        f"txn_rms={result.grouped_transaction_advantage_rms:.9f} "
        f"segment_rms={result.grouped_segment_advantage_rms} "
        f"scales={result.grouped_segment_advantage_scales} "
        f"sign_flips={result.advantage_sign_flip_count}",
        flush=True,
    )


def test_group_mass_is_invariant_to_duplicate_step_and_equivalent_attempt() -> None:
    base_result = compute_frontres_segment_ppo_loss(_ZeroRatioPolicy(), _batch(), _grouped_cfg())
    extended_metadata = _metadata_with_duplicate_step_and_equivalent_attempt()
    extended_result = compute_frontres_segment_ppo_loss(
        _ZeroRatioPolicy(),
        _batch(
            extended_metadata,
            advantages=torch.tensor([1.0, 3.0, 2.0, -4.0, -4.0, -4.0, 8.0, 100.0, 2.0, 2.0]),
            valid_mask=torch.tensor([True, True, True, True, True, True, True, False, True, True]),
        ),
        _grouped_cfg(),
    )
    segment_key = "motion-a|segment=10|source=0"
    base_segments = _mass_by_key(base_result.grouped_segment_keys, base_result.grouped_segment_mass_shares)
    extended_segments = _mass_by_key(
        extended_result.grouped_segment_keys,
        extended_result.grouped_segment_mass_shares,
    )
    extended_attempts = _mass_by_key(
        extended_result.grouped_attempt_keys,
        extended_result.grouped_attempt_mass_shares,
    )
    extended_steps = dict(
        zip(
            extended_result.grouped_valid_step_row_indices,
            extended_result.grouped_valid_step_mass_shares,
            strict=True,
        )
    )

    assert base_segments[segment_key] == extended_segments[segment_key] == 0.25
    assert sum(value for key, value in extended_attempts.items() if key.startswith(segment_key)) == 0.25
    assert extended_steps[2] == extended_steps[8] == 1.0 / 24.0
    assert extended_steps[9] == 1.0 / 12.0
    print(
        "[probe grouped_mk_invariance] "
        f"segment_mass={extended_segments[segment_key]:.9f} "
        f"source_attempt_mass={sum(value for key, value in extended_attempts.items() if key.startswith(segment_key)):.9f} "
        f"duplicate_step_mass={extended_steps[2]:.9f} equivalent_attempt_mass={extended_steps[9]:.9f}",
        flush=True,
    )


def test_grouped_loss_fails_closed_for_missing_or_misaligned_transaction_metadata() -> None:
    batch = _batch()
    missing = FrontRESSegmentPPOBatch(
        observations=batch.observations,
        actions=batch.actions,
        old_log_probs=batch.old_log_probs,
        old_values=batch.old_values,
        returns=batch.returns,
        advantages=batch.advantages,
        valid_mask=batch.valid_mask,
        segment_ids=batch.segment_ids,
    )
    try:
        compute_frontres_segment_ppo_loss(_ZeroRatioPolicy(), missing, _grouped_cfg())
    except ValueError as exc:
        assert "transaction metadata" in str(exc)
    else:
        raise AssertionError("grouped_scale_only must reject a missing transaction metadata carrier")

    bad_metadata = _metadata()
    bad_metadata = _TransactionMetadata(
        **{
            **bad_metadata.__dict__,
            "segment_ids": torch.tensor([99, *bad_metadata.segment_ids[1:].tolist()], dtype=torch.long),
        }
    )
    original = _batch()
    mismatched = FrontRESSegmentPPOBatch(
        observations=original.observations,
        actions=original.actions,
        old_log_probs=original.old_log_probs,
        old_values=original.old_values,
        returns=original.returns,
        advantages=original.advantages,
        valid_mask=original.valid_mask,
        segment_ids=original.segment_ids,
        transaction_metadata=bad_metadata,
    )
    try:
        compute_frontres_segment_ppo_loss(_ZeroRatioPolicy(), mismatched, _grouped_cfg())
    except ValueError as exc:
        assert "segment_ids" in str(exc)
    else:
        raise AssertionError("grouped_scale_only must reject a metadata row that disagrees with storage")

    partial = FrontRESSegmentPPOBatch(
        observations=batch.observations[:4],
        actions=batch.actions[:4],
        old_log_probs=batch.old_log_probs[:4],
        old_values=batch.old_values[:4],
        returns=batch.returns[:4],
        advantages=batch.advantages[:4],
        valid_mask=batch.valid_mask[:4],
        segment_ids=batch.segment_ids[:4],
        transaction_metadata=_metadata(),
        transaction_row_indices=torch.arange(4),
    )
    try:
        compute_frontres_segment_ppo_loss(_ZeroRatioPolicy(), partial, _grouped_cfg())
    except ValueError as exc:
        assert "transaction-complete" in str(exc)
    else:
        raise AssertionError("grouped_scale_only must not silently reweight a partial transaction mini-batch")


def test_storage_adapter_preserves_sealed_transaction_metadata_for_grouped_loss() -> None:
    metadata = _v015_metadata()
    storage_batch = FrontRESSegmentStorageBatch(
        observations=torch.zeros(8, 2),
        actions=torch.zeros(8, 6),
        old_log_probs=torch.zeros(8),
        old_values=torch.zeros(8),
        rewards=torch.zeros(8),
        returns=torch.zeros(8),
        advantages=torch.tensor([1.0, 3.0, 2.0, -4.0, -4.0, -4.0, 8.0, 100.0]),
        valid_mask=torch.tensor([True, True, True, True, True, True, True, False]),
        segment_ids=metadata.segment_ids.clone(),
        transaction_metadata=metadata,
    )
    ppo_batch = storage_batch.to_grouped_ppo_candidate_batch(FrontRESSegmentPPOBatch)
    try:
        storage_batch.to_ppo_batch(FrontRESSegmentPPOBatch)
    except ValueError as exc:
        assert "must not enter legacy" in str(exc)
    else:
        raise AssertionError("v015 grouped candidate metadata must reject the legacy PPO adapter")
    result = compute_frontres_segment_ppo_loss(_ZeroRatioPolicy(), ppo_batch, _grouped_cfg())

    assert ppo_batch.transaction_metadata is metadata
    torch.testing.assert_close(ppo_batch.transaction_row_indices, torch.arange(8))
    assert result.grouped_reduction_active
    assert result.grouped_motion_count == 2
    assert result.grouped_segment_count == 3
    assert result.grouped_attempt_count == 7
    attempt_mass = dict(zip(result.grouped_attempt_keys, result.grouped_attempt_mass_shares, strict=True))
    assert abs(sum(value for key, value in attempt_mass.items() if "motion-a|segment=10" in key) - 0.25) < 1.0e-8
    assert abs(sum(value for key, value in attempt_mass.items() if "motion-a|segment=11" in key) - 0.25) < 1.0e-8
    assert abs(sum(value for key, value in attempt_mass.items() if "motion-b|segment=20" in key) - 0.5) < 1.0e-8


def test_grouped_reducer_has_no_sampling_or_replay_loss_multiplier() -> None:
    source = inspect.getsource(ppo_module._reduce_frontres_grouped_rows)
    for forbidden in ("priority", "gain", "focal", "horizon_k", "trial_count"):
        assert forbidden not in source
    legacy_adapter = inspect.getsource(FrontRESSegmentStorageBatch.to_ppo_batch)
    candidate_adapter = inspect.getsource(FrontRESSegmentStorageBatch.to_grouped_ppo_candidate_batch)
    runner_source = (ROOT / "rsl_rl" / "runners" / "frontres_segment_live_probe.py").read_text()
    assert "include_transaction_metadata=False" in legacy_adapter
    assert "include_transaction_metadata=True" in candidate_adapter
    assert "def build_frontres_v015_grouped_candidate_batch" in runner_source
    assert runner_source.count("to_grouped_ppo_candidate_batch") == 1


def main() -> None:
    test_grouped_nested_reduction_is_hand_computed_and_row_permutation_invariant()
    test_grouped_scale_preserves_sign_and_does_not_amplify_low_scale_segment()
    test_group_mass_is_invariant_to_duplicate_step_and_equivalent_attempt()
    test_grouped_loss_fails_closed_for_missing_or_misaligned_transaction_metadata()
    test_storage_adapter_preserves_sealed_transaction_metadata_for_grouped_loss()
    test_grouped_reducer_has_no_sampling_or_replay_loss_multiplier()
    print("frontres_segment_grouped_ppo_contract: ok")


if __name__ == "__main__":
    main()
