#!/usr/bin/env python3
"""Deterministic grouped scalar PPO contracts for FRS-PPO-v005."""

from __future__ import annotations

from dataclasses import replace

import torch

from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESSegmentPPOBatch,
    FrontRESSegmentPPOConfig,
    compute_frontres_segment_ppo_loss,
    install_frontres_v005_scalar_gradients,
    step_frontres_v005_scalar_optimizer,
)
from rsl_rl.frontres.frontres_segment_storage_records import FrontRESV015GroupedCandidateMetadata


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(2, 1, bias=False)
        self.critic = torch.nn.Linear(2, 1, bias=False)

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        return {
            "log_prob": self.actor(observations).reshape(-1),
            "value": self.critic(observations).reshape(-1),
            "entropy": torch.zeros(observations.shape[0]),
        }


def _metadata(order: torch.Tensor | None = None) -> FrontRESV015GroupedCandidateMetadata:
    order = torch.arange(4) if order is None else order
    values = {
        "motion_ids": ("motion-a", "motion-a", "motion-b", "motion-b"),
        "start_frames": torch.tensor([4, 4, 8, 8]),
        "segment_ids": torch.tensor([10, 10, 11, 11]),
        "source_index": torch.tensor([0, 0, 1, 1]),
        "trial_index": torch.tensor([0, 1, 0, 1]),
        "horizon_k": torch.tensor([8, 8, 8, 8]),
        "evidence_valid_step_count": torch.tensor([8, 8, 8, 8]),
        "noisy_segment_hashes": ("ha", "ha", "hb", "hb"),
        "scenario_ids": ("sa", "sa", "sb", "sb"),
        "x_t_identities": ("xa", "xa", "xb", "xb"),
    }
    indices = [int(value) for value in order.tolist()]
    return FrontRESV015GroupedCandidateMetadata(
        transaction_id="tx-v005",
        policy_snapshot_id="pi-old-v005",
        motion_ids=tuple(values["motion_ids"][index] for index in indices),
        start_frames=values["start_frames"][order],
        segment_ids=values["segment_ids"][order],
        source_index=values["source_index"][order],
        trial_index=values["trial_index"][order],
        horizon_k=values["horizon_k"][order],
        evidence_valid_step_count=values["evidence_valid_step_count"][order],
        trial_role=("policy",) * 4,
        noisy_segment_hashes=tuple(values["noisy_segment_hashes"][index] for index in indices),
        scenario_ids=tuple(values["scenario_ids"][index] for index in indices),
        x_t_identities=tuple(values["x_t_identities"][index] for index in indices),
        intent_q29_provenance="deployment_noisy_q29",
        intent_q29_source="motion_internal_q29",
    )


def _batch(order: torch.Tensor | None = None) -> FrontRESSegmentPPOBatch:
    order = torch.arange(4) if order is None else order
    observations = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    returns = torch.tensor([1.0, 2.0, -1.0, -2.0])
    values = torch.zeros(4)
    return FrontRESSegmentPPOBatch(
        observations=observations[order],
        privileged_observations=observations[order],
        actions=torch.zeros(4, 6),
        old_log_probs=torch.zeros(4),
        old_values=values[order],
        returns=returns[order],
        advantages=(returns - values)[order],
        valid_mask=torch.ones(4, dtype=torch.bool),
        segment_ids=torch.tensor([10, 10, 11, 11])[order],
        old_means=torch.zeros(4, 6),
        old_sigmas=torch.ones(4, 6),
        transaction_metadata=_metadata(order),
        transaction_row_indices=torch.arange(4),
    )


def _cfg(weight: float = 1.0) -> FrontRESSegmentPPOConfig:
    return FrontRESSegmentPPOConfig(
        normalize_advantages=False,
        advantage_normalization="grouped_scale_only",
        actor_loss_weight=weight,
    )


def test_equal_mass_sign_and_permutation() -> None:
    policy = _Policy()
    result = compute_frontres_segment_ppo_loss(policy, _batch(), _cfg())
    assert result.valid_count == 4
    assert result.grouped_segment_count == 2 and result.grouped_attempt_count == 4
    assert all(abs(value - 0.5) < 1.0e-7 for value in result.grouped_segment_mass_shares)
    assert all(abs(value - 0.25) < 1.0e-7 for value in result.grouped_attempt_mass_shares)
    prepared = torch.tensor(result.prepared_advantages)
    assert torch.equal(torch.sign(prepared), torch.tensor([1.0, 1.0, -1.0, -1.0]))

    order = torch.tensor([2, 0, 3, 1])
    permuted = compute_frontres_segment_ppo_loss(policy, _batch(order), _cfg())
    torch.testing.assert_close(result.total_loss, permuted.total_loss)


def test_missing_or_partial_metadata_rejects() -> None:
    batch = _batch()
    try:
        compute_frontres_segment_ppo_loss(_Policy(), replace(batch, transaction_metadata=None), _cfg())
    except ValueError as exc:
        assert "metadata" in str(exc).lower()
    else:
        raise AssertionError("grouped scalar PPO must reject missing transaction metadata")


def test_scalar_gradient_and_critic_only_authority() -> None:
    policy = _Policy()
    result = compute_frontres_segment_ppo_loss(policy, _batch(), _cfg(weight=0.0))
    parameters = tuple(policy.parameters())
    snapshots = {id(parameter): parameter.detach().clone() for parameter in parameters}
    actor_parameters, critic_parameters = install_frontres_v005_scalar_gradients(
        policy, result, _cfg(weight=0.0), parameters
    )
    assert actor_parameters and critic_parameters
    assert all(parameter.grad is None for parameter in actor_parameters)
    assert all(parameter.grad is not None for parameter in critic_parameters)
    optimizer = torch.optim.Adam(parameters, lr=1.0e-3)
    commit = step_frontres_v005_scalar_optimizer(
        optimizer,
        actor_parameters,
        snapshots,
        actor_loss_weight=0.0,
    )
    assert commit.committed_actor_delta_l2 == 0.0
    assert commit.actor_optimizer_state_preserved
    assert all(torch.equal(parameter.detach(), snapshots[id(parameter)]) for parameter in actor_parameters)


def main() -> None:
    test_equal_mass_sign_and_permutation()
    test_missing_or_partial_metadata_rejects()
    test_scalar_gradient_and_critic_only_authority()
    print("frontres_segment_grouped_ppo_contract: scalar v005 ok", flush=True)


if __name__ == "__main__":
    main()
