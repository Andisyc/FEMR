"""TEST ONLY: FrontRES authority storage contract.

This test checks storage plumbing only.  It does not test runner rollout
construction or authority actor-critic loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.storage.rollout_storage import RolloutStorage


AUTHORITY_START = 28


def _make_storage(*, num_steps: int = 2, num_envs: int = 3) -> RolloutStorage:
    return RolloutStorage(
        training_type="frontres",
        num_envs=num_envs,
        num_transitions_per_env=num_steps,
        obs_shape=(4,),
        privileged_obs_shape=None,
        actions_shape=(8,),
        device="cpu",
    )


def _authority_values(step: int, num_envs: int) -> dict[str, torch.Tensor]:
    base = torch.arange(num_envs, dtype=torch.float32).unsqueeze(-1) + step * 10.0
    dims = torch.arange(6, dtype=torch.float32).unsqueeze(0)
    return {
        "proposal_delta_se": base + dims * 0.01,
        "authority_action": base + 0.10 + dims * 0.01,
        "authority_log_prob": base + 0.20,
        "authority_rho": torch.sigmoid(base + dims * 0.01),
        "authority_return_k": base + 0.30,
        "authority_return_zero_k": torch.zeros(num_envs, 1),
        "authority_return_one_k": base + 0.40,
        "authority_mask": (base.remainder(2.0) == 0.0).float(),
    }


def _add_transition(storage: RolloutStorage, step: int) -> None:
    num_envs = storage.num_envs
    transition = RolloutStorage.Transition()
    transition.observations = torch.full((num_envs, 4), float(step))
    transition.actions = torch.full((num_envs, 8), float(step + 1))
    transition.rewards = torch.full((num_envs,), float(step + 2))
    transition.dones = torch.zeros(num_envs)
    transition.values = torch.full((num_envs, 1), float(step + 3))
    transition.actions_log_prob = torch.full((num_envs,), float(step + 4))
    transition.action_mean = torch.full((num_envs, 8), float(step + 5))
    transition.action_sigma = torch.full((num_envs, 8), float(step + 6))
    transition.frontres_mask = torch.ones(num_envs, 1)
    transition.supervised_target = torch.zeros(num_envs, 6)

    for name, value in _authority_values(step, num_envs).items():
        setattr(transition, name, value)
    storage.add_transitions(transition)


def _expected_flat(storage: RolloutStorage, name: str, batch_indices: torch.Tensor) -> torch.Tensor:
    return getattr(storage, name).flatten(0, 1)[batch_indices]


def test_feedforward_authority_storage_round_trip() -> None:
    storage = _make_storage()
    for step in range(storage.num_transitions_per_env):
        _add_transition(storage, step)
    storage.yield_batch_indices = True

    batch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
    if len(batch) != 37:
        raise AssertionError(f"FrontRES feedforward batch should have 37 entries, got {len(batch)}.")
    batch_indices = batch[36]
    names = (
        "proposal_delta_se",
        "authority_action",
        "authority_log_prob",
        "authority_rho",
        "authority_return_k",
        "authority_return_zero_k",
        "authority_return_one_k",
        "authority_mask",
    )
    for offset, name in enumerate(names):
        actual = batch[AUTHORITY_START + offset]
        expected = _expected_flat(storage, name, batch_indices)
        torch.testing.assert_close(actual, expected)


def test_recurrent_authority_storage_round_trip() -> None:
    storage = _make_storage(num_steps=3, num_envs=2)
    for step in range(storage.num_transitions_per_env):
        transition = RolloutStorage.Transition()
        transition.observations = torch.full((storage.num_envs, 4), float(step))
        transition.actions = torch.full((storage.num_envs, 8), float(step + 1))
        transition.rewards = torch.full((storage.num_envs,), float(step + 2))
        transition.dones = torch.zeros(storage.num_envs)
        transition.values = torch.full((storage.num_envs, 1), float(step + 3))
        transition.actions_log_prob = torch.full((storage.num_envs,), float(step + 4))
        transition.action_mean = torch.full((storage.num_envs, 8), float(step + 5))
        transition.action_sigma = torch.full((storage.num_envs, 8), float(step + 6))
        transition.frontres_mask = torch.ones(storage.num_envs, 1)
        transition.supervised_target = torch.zeros(storage.num_envs, 6)
        transition.hidden_states = (
            torch.zeros(1, storage.num_envs, 4),
            torch.zeros(1, storage.num_envs, 4),
        )
        for name, value in _authority_values(step, storage.num_envs).items():
            setattr(transition, name, value)
        storage.add_transitions(transition)

    batch = next(storage.recurrent_mini_batch_generator(num_mini_batches=1, num_epochs=1))
    if len(batch) != 36:
        raise AssertionError(f"FrontRES recurrent batch should have 36 entries, got {len(batch)}.")
    torch.testing.assert_close(batch[28], storage.proposal_delta_se)
    torch.testing.assert_close(batch[29], storage.authority_action)
    torch.testing.assert_close(batch[30], storage.authority_log_prob)
    torch.testing.assert_close(batch[31], storage.authority_rho)
    torch.testing.assert_close(batch[32], storage.authority_return_k)
    torch.testing.assert_close(batch[33], storage.authority_return_zero_k)
    torch.testing.assert_close(batch[34], storage.authority_return_one_k)
    torch.testing.assert_close(batch[35], storage.authority_mask)


def test_default_authority_storage_is_zero() -> None:
    storage = _make_storage(num_steps=1, num_envs=2)
    transition = RolloutStorage.Transition()
    transition.observations = torch.zeros(2, 4)
    transition.actions = torch.zeros(2, 8)
    transition.rewards = torch.zeros(2)
    transition.dones = torch.zeros(2)
    transition.values = torch.zeros(2, 1)
    transition.actions_log_prob = torch.zeros(2)
    transition.action_mean = torch.zeros(2, 8)
    transition.action_sigma = torch.ones(2, 8)
    storage.add_transitions(transition)
    if float(storage.authority_mask.abs().sum().item()) != 0.0:
        raise AssertionError("Default authority_mask should be zero when no authority event is stored.")
    if float(storage.authority_rho.abs().sum().item()) != 0.0:
        raise AssertionError("Default authority_rho should be zero when no authority event is stored.")
    if float(storage.authority_return_one_k.abs().sum().item()) != 0.0:
        raise AssertionError("Default authority_return_one_k should be zero when no endpoint target is stored.")


def main() -> None:
    test_feedforward_authority_storage_round_trip()
    test_recurrent_authority_storage_round_trip()
    test_default_authority_storage_is_zero()
    print("=== FrontRES Authority Storage TEST ONLY ===")
    print("checks=transition copy, endpoint returns, feedforward tuple, recurrent tuple, default inactive zeros")
    print("result: PASS")


if __name__ == "__main__":
    main()
