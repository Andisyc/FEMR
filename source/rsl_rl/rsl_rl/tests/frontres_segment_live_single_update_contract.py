#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_import_stubs():
    rsl_rl_pkg = _package("rsl_rl")
    algorithms_pkg = _package("rsl_rl.algorithms")
    frontres_pkg = _package("rsl_rl.frontres")
    runners_pkg = _package("rsl_rl.runners")

    rsl_rl_pkg.algorithms = algorithms_pkg
    rsl_rl_pkg.frontres = frontres_pkg
    rsl_rl_pkg.runners = runners_pkg
    algorithms_pkg.FrontRESUnified = object

    ppo_module = _load(
        "rsl_rl.algorithms.frontres_segment_ppo",
        ROOT / "rsl_rl" / "algorithms" / "frontres_segment_ppo.py",
    )
    algorithms_pkg.frontres_segment_ppo = ppo_module

    storage_module = _load(
        "rsl_rl.frontres.frontres_segment_storage",
        ROOT / "rsl_rl" / "frontres" / "frontres_segment_storage.py",
    )
    frontres_pkg.frontres_segment_storage = storage_module

    training_schedule = types.ModuleType("rsl_rl.frontres.training_schedule")
    training_schedule.configure_frontres_pair_layout = lambda *_args, **_kwargs: None
    training_schedule.resolve_frontres_mode_state = lambda *_args, **_kwargs: None
    sys.modules[training_schedule.__name__] = training_schedule
    frontres_pkg.training_schedule = training_schedule

    modules_pkg = types.ModuleType("rsl_rl.modules")
    modules_pkg.FrontRESActorCritic = object
    sys.modules[modules_pkg.__name__] = modules_pkg
    rsl_rl_pkg.modules = modules_pkg

    rollout_step = types.ModuleType("rsl_rl.runners.frontres_rollout_step")
    rollout_step.prepare_frontres_rollout_step = lambda *_args, **_kwargs: None
    sys.modules[rollout_step.__name__] = rollout_step
    runners_pkg.frontres_rollout_step = rollout_step

    live_probe_module = _load(
        "rsl_rl.runners.frontres_segment_live_probe",
        ROOT / "rsl_rl" / "runners" / "frontres_segment_live_probe.py",
    )
    runners_pkg.frontres_segment_live_probe = live_probe_module
    return live_probe_module, storage_module


live_probe, storage_module = _install_import_stubs()
FrontRESSegmentRolloutStorage = storage_module.FrontRESSegmentRolloutStorage
FrontRESSegmentTransition = storage_module.FrontRESSegmentTransition
run_frontres_segment_single_update = live_probe.run_frontres_segment_single_update


def _probe_tensor(name: str, tensor: torch.Tensor, semantic: str) -> None:
    data = tensor.detach().cpu()
    numeric = data.float() if data.dtype == torch.bool else data
    print(
        f"[probe step3] {name}: shape={tuple(data.shape)} dtype={data.dtype} "
        f"device={tensor.device} min={numeric.min().item():.6f} "
        f"max={numeric.max().item():.6f} mean={numeric.float().mean().item():.6f} "
        f"requires_grad={tensor.requires_grad} grad_fn={type(tensor.grad_fn).__name__ if tensor.grad_fn else None} "
        f"semantic={semantic}",
        flush=True,
    )


def _probe_update(name: str, result, runner: "FakeRunner", before_actor: torch.Tensor, before_critic: torch.Tensor) -> None:
    actor_delta = (runner.alg.policy.actor.weight.detach() - before_actor).norm().item()
    critic_delta = (runner.alg.policy.critic.weight.detach() - before_critic).norm().item()
    actor_grad_norm = (
        runner.alg.policy.actor.weight.grad.detach().norm().item()
        if runner.alg.policy.actor.weight.grad is not None
        else 0.0
    )
    critic_grad_norm = (
        runner.alg.policy.critic.weight.grad.detach().norm().item()
        if runner.alg.policy.critic.weight.grad is not None
        else 0.0
    )
    print(
        f"[probe step3] {name}: should_step={result.should_step} valid_count={result.valid_count} "
        f"loss={result.total_loss.detach().item():.6f} "
        f"loss_requires_grad={result.total_loss.requires_grad} "
        f"actor_grad_norm={actor_grad_norm:.6f} critic_grad_norm={critic_grad_norm:.6f} "
        f"actor_delta_norm={actor_delta:.6f} critic_delta_norm={critic_delta:.6f} "
        f"mode_trace={runner.mode_trace} actor_obs_trace={runner.alg.policy.actor_obs_trace} "
        f"critic_obs_trace={runner.alg.policy.critic_obs_trace}",
        flush=True,
    )


class FakeLivePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(4, 6, bias=False)
        self.critic = torch.nn.Linear(3, 1, bias=False)
        torch.nn.init.zeros_(self.actor.weight)
        torch.nn.init.zeros_(self.critic.weight)
        self.last_mean: torch.Tensor | None = None
        self.actor_obs_trace: list[tuple[int, int]] = []
        self.critic_obs_trace: list[tuple[int, int]] = []

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        self.actor_obs_trace.append(tuple(observations.shape))
        self.last_mean = self.actor(observations)
        return self.last_mean

    def evaluate(self, observations: torch.Tensor) -> torch.Tensor:
        self.critic_obs_trace.append(tuple(observations.shape))
        return self.critic(observations)

    def entropy(self) -> torch.Tensor:
        assert self.last_mean is not None
        return torch.ones(self.last_mean.shape[0])


class FakeAlg:
    def __init__(self) -> None:
        self.policy = FakeLivePolicy()
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=0.1)
        self.use_estimate_ref_vel = False
        self.clip_param = 0.2
        self.value_loss_coef = 1.0
        self.entropy_coef = 0.0
        self.use_clipped_value_loss = True
        self.normalize_advantage_per_mini_batch = False
        self.max_grad_norm = 1.0

    def _get_actor_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.policy.last_mean is not None
        return -0.5 * (actions - self.policy.last_mean).square().sum(dim=-1)


class FakeRunner:
    def __init__(self) -> None:
        self.alg = FakeAlg()
        self.mode_trace: list[str] = []

    def train_mode(self) -> None:
        self.mode_trace.append("train")

    def eval_mode(self) -> None:
        self.mode_trace.append("eval")


def _storage_batch(valid_mask: torch.Tensor) -> object:
    storage = FrontRESSegmentRolloutStorage(
        capacity=2,
        obs_shape=(4,),
        privileged_obs_shape=(3,),
        action_dim=6,
        device="cpu",
    )
    storage.add_transition(
        FrontRESSegmentTransition(
            observations=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            privileged_observations=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            actions=torch.tensor([[0.50, 0.0, 0.0, 0.0, 0.0, 0.0], [50.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            old_log_probs=torch.zeros(2),
            values=torch.zeros(2),
            rewards=torch.tensor([1.0, 1000.0]),
            valid_mask=valid_mask,
            reset_mask=torch.ones(2, dtype=torch.bool),
            segment_ids=torch.tensor([0, 1]),
            old_means=torch.zeros(2, 6),
            old_sigmas=torch.ones(2, 6),
            action_mask=torch.ones(2, 6),
        )
    )
    return storage.full_batch()


def test_single_update_steps_optimizer_with_valid_segment() -> None:
    runner = FakeRunner()
    before_actor = runner.alg.policy.actor.weight.detach().clone()
    before_critic = runner.alg.policy.critic.weight.detach().clone()
    storage_batch = _storage_batch(torch.tensor([True, False]))
    _probe_tensor("storage_batch.observations", storage_batch.observations, "policy observation passed to actor")
    _probe_tensor(
        "storage_batch.privileged_observations",
        storage_batch.privileged_observations,
        "privileged observation passed to critic",
    )
    _probe_tensor("storage_batch.actions", storage_batch.actions, "6D actions passed through live adapter")
    _probe_tensor("storage_batch.valid_mask", storage_batch.valid_mask, "valid row should trigger optimizer step")

    result = run_frontres_segment_single_update(runner, storage_batch)
    _probe_update("valid_single_update", result, runner, before_actor, before_critic)

    assert result.should_step
    assert result.valid_count == 1
    assert runner.mode_trace == ["train", "eval"]
    assert runner.alg.policy.actor_obs_trace == [(2, 4)]
    assert runner.alg.policy.critic_obs_trace == [(2, 3)]
    assert not torch.allclose(runner.alg.policy.actor.weight.detach(), before_actor)
    assert not torch.allclose(runner.alg.policy.critic.weight.detach(), before_critic)


def test_single_update_does_not_step_optimizer_without_valid_segments() -> None:
    runner = FakeRunner()
    before_actor = runner.alg.policy.actor.weight.detach().clone()
    before_critic = runner.alg.policy.critic.weight.detach().clone()
    storage_batch = _storage_batch(torch.tensor([False, False]))
    _probe_tensor("storage_batch.observations", storage_batch.observations, "policy observation still evaluated")
    _probe_tensor("storage_batch.valid_mask", storage_batch.valid_mask, "no valid row should suppress optimizer step")

    result = run_frontres_segment_single_update(runner, storage_batch)
    _probe_update("all_invalid_single_update", result, runner, before_actor, before_critic)

    assert not result.should_step
    assert result.valid_count == 0
    assert runner.mode_trace == ["train", "eval"]
    torch.testing.assert_close(runner.alg.policy.actor.weight.detach(), before_actor)
    torch.testing.assert_close(runner.alg.policy.critic.weight.detach(), before_critic)


if __name__ == "__main__":
    test_single_update_steps_optimizer_with_valid_segment()
    test_single_update_does_not_step_optimizer_without_valid_segments()
    print("frontres_segment_live_single_update_contract: ok")
