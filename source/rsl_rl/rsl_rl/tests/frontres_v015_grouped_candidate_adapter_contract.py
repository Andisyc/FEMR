#!/usr/bin/env python3
"""Deterministic Step 4A contract for v015 metadata -> grouped candidate PPO."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"
GAIN_CONSUMER_TEST = RSL_ROOT / "tests" / "frontres_v015_gain_consumer_contract.py"
PPO_PATH = RSL_ROOT / "algorithms" / "frontres_segment_ppo.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expect_value_error(fn) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


class _ZeroRatioPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_prob_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.std = torch.nn.Parameter(torch.zeros(6))
        self.critic = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.critic.weight)

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        del actions
        return {
            "log_prob": self.log_prob_scale * observations[:, 0],
            "value": self.critic(observations[:, 1:2]).squeeze(-1),
            "entropy": torch.zeros_like(observations[:, 0]),
        }


def _load_owners():
    gain_contract = _load("frontres_v015_grouped_candidate_gain_helper", GAIN_CONSUMER_TEST)
    one_action, helper, commands, hooks, setup, live_probe, gain, sampler = gain_contract._load_owners()
    storage = sys.modules["rsl_rl.frontres.frontres_segment_storage"]
    ppo = _load("frontres_v015_grouped_candidate_ppo", PPO_PATH)
    live_probe.FrontRESSegmentPPOBatch = ppo.FrontRESSegmentPPOBatch
    return gain_contract, one_action, helper, commands, hooks, setup, live_probe, storage, ppo


def _candidate_kwargs(row_count: int) -> dict[str, object]:
    assert row_count == 2, "the shared deterministic v015 fixture exposes two Repair policy rows"
    return {
        "transaction_id": "tx-v015-candidate",
        "policy_snapshot_id": "tx-v015-candidate:pi-old",
        "motion_ids": ("motion-a", "motion-b"),
        "start_frames": torch.tensor([12, 24], dtype=torch.long),
        "segment_ids": torch.tensor([101, 202], dtype=torch.long),
        "source_index": torch.tensor([0, 1], dtype=torch.long),
        "trial_index": torch.tensor([0, 0], dtype=torch.long),
    }


def _capture_and_build(gain_contract, one_action, helper, commands, hooks, setup, live_probe, ppo):
    captured = gain_contract._capture_consumer(one_action, helper, commands, hooks, setup, live_probe)
    candidate = captured.result
    kwargs = _candidate_kwargs(int(candidate.return_evidence.policy_actions.shape[0]))
    batch = live_probe.build_frontres_v015_grouped_candidate_batch(candidate, **kwargs)
    return captured, kwargs, batch


def _permute_metadata(metadata, order: torch.Tensor):
    indices = [int(index) for index in order.tolist()]
    return replace(
        metadata,
        motion_ids=tuple(metadata.motion_ids[index] for index in indices),
        start_frames=metadata.start_frames[order],
        segment_ids=metadata.segment_ids[order],
        source_index=metadata.source_index[order],
        trial_index=metadata.trial_index[order],
        horizon_k=metadata.horizon_k[order],
        evidence_valid_step_count=metadata.evidence_valid_step_count[order],
        trial_role=tuple(metadata.trial_role[index] for index in indices),
        noisy_segment_hashes=tuple(metadata.noisy_segment_hashes[index] for index in indices),
        scenario_ids=tuple(metadata.scenario_ids[index] for index in indices),
        x_t_identities=tuple(metadata.x_t_identities[index] for index in indices),
    )


def _permute_batch(ppo, batch, metadata, order: torch.Tensor):
    return ppo.FrontRESSegmentPPOBatch(
        observations=batch.observations[order],
        actions=batch.actions[order],
        old_log_probs=batch.old_log_probs[order],
        old_values=batch.old_values[order],
        returns=batch.returns[order],
        advantages=batch.advantages[order],
        valid_mask=batch.valid_mask[order],
        segment_ids=batch.segment_ids[order],
        old_means=batch.old_means[order] if batch.old_means is not None else None,
        old_sigmas=batch.old_sigmas[order] if batch.old_sigmas is not None else None,
        transaction_metadata=metadata,
        transaction_row_indices=torch.arange(int(order.numel()), dtype=torch.long),
    )


def _grouped_cfg(ppo):
    return ppo.FrontRESSegmentPPOConfig(
        advantage_normalization="grouped_scale_only",
        value_loss_coef=0.0,
        entropy_coef=0.0,
    )


def test_t_schema_row_metadata_and_legacy_reject(
    gain_contract,
    one_action,
    helper,
    commands,
    hooks,
    setup,
    live_probe,
    storage,
    ppo,
) -> None:
    captured, kwargs, batch = _capture_and_build(
        gain_contract, one_action, helper, commands, hooks, setup, live_probe, ppo
    )
    candidate = captured.result
    returned = candidate.return_evidence
    metadata = batch.transaction_metadata

    metadata.validate()
    assert metadata.layout_version == "frontres-v015-local-scenario-v1"
    assert tuple(batch.actions.shape) == tuple(returned.policy_actions.shape) == (2, 6)
    torch.testing.assert_close(batch.actions, returned.policy_actions)
    torch.testing.assert_close(
        batch.privileged_observations,
        candidate.one_action.policy_privileged_observations,
    )
    torch.testing.assert_close(batch.returns, returned.return_k)
    torch.testing.assert_close(batch.advantages, returned.advantage_k)
    torch.testing.assert_close(batch.valid_mask, returned.policy_row_valid)
    torch.testing.assert_close(metadata.horizon_k, returned.horizon_k)
    repair_rows = candidate.one_action.policy_row_indices
    expected_evidence_steps = candidate.one_action.survival_steps.index_select(0, repair_rows).to(dtype=torch.long)
    torch.testing.assert_close(metadata.evidence_valid_step_count, expected_evidence_steps)
    assert metadata.scenario_ids == returned.scenario_ids
    assert metadata.noisy_segment_hashes == returned.noisy_segment_hashes
    assert metadata.x_t_identities == returned.x_t_identities
    assert metadata.intent_q29_provenance == "deployment_noisy_q29"
    assert metadata.intent_q29_source == "motion_internal_q29"
    torch.testing.assert_close(batch.transaction_row_indices, torch.arange(2, dtype=torch.long))

    storage_batch = storage.build_frontres_v015_grouped_candidate_storage(candidate, **kwargs)
    _expect_value_error(lambda: storage_batch.to_ppo_batch(ppo.FrontRESSegmentPPOBatch))

    _expect_value_error(
        lambda: replace(
            metadata,
            source_index=torch.tensor([0, 0], dtype=torch.long),
            trial_index=torch.tensor([0, 1], dtype=torch.long),
            scenario_ids=(metadata.scenario_ids[0], metadata.scenario_ids[0]),
            noisy_segment_hashes=(metadata.noisy_segment_hashes[0], metadata.noisy_segment_hashes[0]),
            x_t_identities=(metadata.x_t_identities[0], "mixed-x-t"),
        )
    )
    print(
        "[T-schema/T-row/T-metadata/T-legacy-reject] one Repair row keeps sealed local identity; legacy adapter rejects v015",
        flush=True,
    )


def test_t_permute_scale_and_k_evidence_mass_isolation(
    gain_contract,
    one_action,
    helper,
    commands,
    hooks,
    setup,
    live_probe,
    storage,
    ppo,
) -> None:
    del storage
    _captured, _kwargs, batch = _capture_and_build(
        gain_contract, one_action, helper, commands, hooks, setup, live_probe, ppo
    )
    cfg = _grouped_cfg(ppo)
    policy = _ZeroRatioPolicy()
    result = ppo.compute_frontres_segment_ppo_loss(policy, batch, cfg)
    assert result.grouped_reduction_active
    assert result.grouped_motion_count == 2
    assert result.grouped_segment_count == 2
    assert result.grouped_attempt_count == 2
    assert result.advantage_sign_flip_count == 0
    torch.testing.assert_close(torch.tensor(result.grouped_motion_mass_shares), torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(torch.tensor(result.grouped_attempt_mass_shares), torch.tensor([0.5, 0.5]))

    order = torch.tensor([1, 0], dtype=torch.long)
    permuted_metadata = _permute_metadata(batch.transaction_metadata, order)
    permuted = _permute_batch(ppo, batch, permuted_metadata, order)
    permuted_result = ppo.compute_frontres_segment_ppo_loss(_ZeroRatioPolicy(), permuted, cfg)
    torch.testing.assert_close(result.actor_loss.detach(), permuted_result.actor_loss.detach(), atol=1.0e-7, rtol=0.0)

    changed_evidence_count = replace(
        batch.transaction_metadata,
        evidence_valid_step_count=torch.zeros_like(batch.transaction_metadata.evidence_valid_step_count),
    )
    changed_evidence_count.validate()
    k_isolated = _permute_batch(
        ppo,
        batch,
        changed_evidence_count,
        torch.arange(int(batch.actions.shape[0]), dtype=torch.long),
    )
    k_isolated_result = ppo.compute_frontres_segment_ppo_loss(_ZeroRatioPolicy(), k_isolated, cfg)
    torch.testing.assert_close(result.actor_loss.detach(), k_isolated_result.actor_loss.detach(), atol=1.0e-7, rtol=0.0)
    assert result.grouped_attempt_mass_shares == k_isolated_result.grouped_attempt_mass_shares

    partial = ppo.FrontRESSegmentPPOBatch(
        observations=batch.observations[:1],
        actions=batch.actions[:1],
        old_log_probs=batch.old_log_probs[:1],
        old_values=batch.old_values[:1],
        returns=batch.returns[:1],
        advantages=batch.advantages[:1],
        valid_mask=batch.valid_mask[:1],
        segment_ids=batch.segment_ids[:1],
        old_means=batch.old_means[:1] if batch.old_means is not None else None,
        old_sigmas=batch.old_sigmas[:1] if batch.old_sigmas is not None else None,
        transaction_metadata=batch.transaction_metadata,
        transaction_row_indices=torch.tensor([0], dtype=torch.long),
    )
    _expect_value_error(lambda: ppo.compute_frontres_segment_ppo_loss(_ZeroRatioPolicy(), partial, cfg))
    print(
        "[T-permute/T-scale/T-k-isolation/T-fail-closed] grouped mass is row-permutation and K-evidence invariant",
        flush=True,
    )


def main() -> None:
    owners = _load_owners()
    test_t_schema_row_metadata_and_legacy_reject(*owners)
    test_t_permute_scale_and_k_evidence_mass_isolation(*owners)
    print("frontres_v015_grouped_candidate_adapter_contract: ok", flush=True)


if __name__ == "__main__":
    main()
