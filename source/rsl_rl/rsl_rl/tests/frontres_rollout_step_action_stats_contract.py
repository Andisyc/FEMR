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
        self.max_delta_pos = 0.2
        self.max_delta_rpy = 0.4

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        max_d = torch.tensor([[0.2, 0.2, 0.2, 0.4, 0.4, 0.4]], dtype=actions.dtype)
        normalized = (actions[:, :6] / max_d).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(normalized)
        log_prob = torch.distributions.Normal(self.action_mean, self.action_std).log_prob(raw).sum(dim=-1)
        log_j = (torch.log(max_d) + torch.log(1.0 - normalized.pow(2) + 1e-6)).sum(dim=-1)
        return log_prob - log_j


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
    scales = torch.tensor([[0.2, 0.2, 0.2, 0.4, 0.4, 0.4]])
    sampled_actions = torch.tanh(policy.action_mean) * scales
    rollout_step._rewrite_task_space_log_prob(runner, sampled_actions)

    stored_actions = runner.alg.transition.actions
    stored_mean = runner.alg.transition.action_mean
    stored_sigma = runner.alg.transition.action_sigma
    raw_actions = torch.atanh((stored_actions / torch.tensor([[0.2, 0.2, 0.2, 0.4, 0.4, 0.4]])).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    raw_action_old_mean = raw_actions - stored_mean

    print(
        "[probe rollout_full6_action_stats] "
        f"action_dim6={float(stored_actions[0, 5]):.6f} "
        f"old_mean_dim6={float(stored_mean[0, 5]):.6f} "
        f"sigma_dim6={float(stored_sigma[0, 5]):.6f} "
        f"raw_minus_old_mean_dim6={float(raw_action_old_mean[0, 5]):.6f} "
        f"log_prob={float(runner.alg.transition.actions_log_prob[0]):.6f}",
        flush=True,
    )

    assert abs(float(stored_actions[0, 5])) > 0.3
    assert abs(float(stored_mean[0, 5]) + 1.946) < 1e-6
    assert abs(float(stored_sigma[0, 5]) - 0.01) < 1e-8
    assert abs(float(raw_action_old_mean[0, 5])) < 1e-6


def main() -> None:
    test_full6_task_action_preserves_sampled_distribution_tuple()
    print("frontres_rollout_step_action_stats_contract: ok")


if __name__ == "__main__":
    main()
