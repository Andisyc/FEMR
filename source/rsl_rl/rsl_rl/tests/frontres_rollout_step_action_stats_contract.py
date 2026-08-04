#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rollout_step = _load(
    "frontres_rollout_step_action_stats",
    ROOT / "rsl_rl" / "runners" / "frontres_rollout_step.py",
)


class FakePolicy:
    def __init__(self) -> None:
        self.action_mean = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, -1.946]])
        self.action_std = torch.full((1, 6), 0.01)
        self.num_task_corrections = 6

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Normal(self.action_mean, self.action_std).log_prob(actions).sum(dim=-1)


def test_full6_task_action_preserves_sampled_distribution_tuple() -> None:
    """Stored full-6D actions and old distribution stats must remain same-source."""

    policy = FakePolicy()
    runner = SimpleNamespace(
        cfg={},
        alg=SimpleNamespace(
            policy=policy,
            transition=SimpleNamespace(
                action_mean=policy.action_mean.detach().clone(),
                action_sigma=policy.action_std.detach().clone(),
            ),
        )
    )
    sampled_actions = policy.action_mean.detach().clone()
    rollout_step._record_direct_task_space_log_prob(runner, sampled_actions)

    stored_actions = runner.alg.transition.actions
    stored_mean = runner.alg.transition.action_mean
    stored_sigma = runner.alg.transition.action_sigma
    action_old_mean = stored_actions - stored_mean

    print(
        "[probe rollout_full6_action_stats] "
        f"action_dim6={float(stored_actions[0, 5]):.6f} "
        f"old_mean_dim6={float(stored_mean[0, 5]):.6f} "
        f"sigma_dim6={float(stored_sigma[0, 5]):.6f} "
        f"action_minus_old_mean_dim6={float(action_old_mean[0, 5]):.6f} "
        f"log_prob={float(runner.alg.transition.actions_log_prob[0]):.6f}",
        flush=True,
    )

    assert abs(float(stored_actions[0, 5])) > 0.3
    assert abs(float(stored_mean[0, 5]) + 1.946) < 1e-6
    assert abs(float(stored_sigma[0, 5]) - 0.01) < 1e-8
    assert abs(float(action_old_mean[0, 5])) < 1e-6


def main() -> None:
    test_full6_task_action_preserves_sampled_distribution_tuple()
    print("frontres_rollout_step_action_stats_contract: ok")


if __name__ == "__main__":
    main()
