#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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
    return live_probe_module, ppo_module


live_probe, segment_ppo = _install_import_stubs()
FrontRESSegmentLiveRolloutCapture = live_probe.FrontRESSegmentLiveRolloutCapture
build_live_segment_storage = live_probe.build_live_segment_storage
FrontRESSegmentPPOBatch = segment_ppo.FrontRESSegmentPPOBatch
FrontRESSegmentPPOConfig = segment_ppo.FrontRESSegmentPPOConfig
FrontRESSegmentPolicyEval = segment_ppo.FrontRESSegmentPolicyEval
compute_frontres_segment_ppo_loss = segment_ppo.compute_frontres_segment_ppo_loss


def _probe_tensor(name: str, tensor: torch.Tensor, semantic: str) -> None:
    data = tensor.detach().cpu()
    numeric = data.float() if data.dtype == torch.bool else data
    print(
        f"[probe step2] {name}: shape={tuple(data.shape)} dtype={data.dtype} "
        f"device={tensor.device} min={numeric.min().item():.6f} "
        f"max={numeric.max().item():.6f} mean={numeric.float().mean().item():.6f} "
        f"requires_grad={tensor.requires_grad} grad_fn={type(tensor.grad_fn).__name__ if tensor.grad_fn else None} "
        f"semantic={semantic}",
        flush=True,
    )


def _probe_result(name: str, result) -> None:
    print(
        f"[probe step2] {name}: valid_count={result.valid_count} "
        f"valid_frac={result.valid_frac:.6f} total_loss={result.total_loss.detach().item():.6f} "
        f"actor_loss={result.actor_loss.detach().item():.6f} "
        f"value_loss={result.value_loss.detach().item():.6f} "
        f"loss_requires_grad={result.total_loss.requires_grad} "
        f"loss_grad_fn={type(result.total_loss.grad_fn).__name__ if result.total_loss.grad_fn else None}",
        flush=True,
    )


class FakeSegmentPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(4, 6, bias=False)
        self.critic = torch.nn.Linear(4, 1, bias=False)
        torch.nn.init.zeros_(self.actor.weight)
        torch.nn.init.zeros_(self.critic.weight)

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> FrontRESSegmentPolicyEval:
        mean = self.actor(observations)
        value = self.critic(observations).squeeze(-1)
        log_prob = -0.5 * (actions - mean).square().sum(dim=-1)
        entropy = torch.ones_like(log_prob) * 0.5
        return FrontRESSegmentPolicyEval(log_prob=log_prob, value=value, entropy=entropy, mean=mean)


def _capture(invalid_action: float, invalid_reward_accum: float) -> FrontRESSegmentLiveRolloutCapture:
    transition_actions = torch.tensor(
        [
            [0.50, 0.00, 0.00, 0.00, 0.00, 0.00],
            [invalid_action, 0.00, 0.00, 0.00, 0.00, 0.00],
        ]
    )
    return FrontRESSegmentLiveRolloutCapture(
        rollout_k=2,
        reward_mean=0.0,
        done_frac=0.5,
        last_obs_shape=(2, 4),
        action_shape=tuple(transition_actions.shape),
        env_action_shape=(2, 12),
        transition_obs=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        transition_privileged_obs=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        transition_actions=transition_actions,
        transition_log_probs=torch.zeros(2),
        transition_values=torch.zeros(2),
        transition_means=torch.zeros(2, 6),
        transition_sigmas=torch.ones(2, 6),
        reward_accum=torch.tensor([2.0, invalid_reward_accum]),
        done_any=torch.tensor([False, True]),
    )


def _ppo_batch(invalid_action: float, invalid_reward_accum: float) -> FrontRESSegmentPPOBatch:
    runner = SimpleNamespace(device=torch.device("cpu"))
    storage = build_live_segment_storage(runner, _capture(invalid_action, invalid_reward_accum))
    return storage.full_batch().to_ppo_batch(FrontRESSegmentPPOBatch)


def test_live_probe_storage_batch_masks_invalid_segment_before_ppo_loss() -> None:
    policy = FakeSegmentPolicy()
    clean_batch = _ppo_batch(invalid_action=1.0, invalid_reward_accum=2.0)
    extreme_batch = _ppo_batch(invalid_action=1e6, invalid_reward_accum=1e6)
    _probe_tensor("clean_batch.actions", clean_batch.actions, "valid row normal, invalid row harmless baseline")
    _probe_tensor("extreme_batch.actions", extreme_batch.actions, "invalid row deliberately extreme")
    _probe_tensor("clean_batch.valid_mask", clean_batch.valid_mask, "mask before PPO loss")
    _probe_tensor("extreme_batch.valid_mask", extreme_batch.valid_mask, "same mask must exclude extreme row")
    _probe_tensor("clean_batch.returns", clean_batch.returns, "returns before valid indexing")
    _probe_tensor("extreme_batch.returns", extreme_batch.returns, "extreme invalid return before valid indexing")
    clean_invalid = compute_frontres_segment_ppo_loss(
        policy,
        clean_batch,
        FrontRESSegmentPPOConfig(entropy_coef=0.0),
    )
    extreme_invalid = compute_frontres_segment_ppo_loss(
        policy,
        extreme_batch,
        FrontRESSegmentPPOConfig(entropy_coef=0.0),
    )
    _probe_result("clean_invalid_result", clean_invalid)
    _probe_result("extreme_invalid_result", extreme_invalid)

    assert clean_invalid.valid_count == 1
    assert extreme_invalid.valid_count == 1
    assert clean_invalid.valid_frac == 0.5
    torch.testing.assert_close(clean_invalid.actor_loss, extreme_invalid.actor_loss)
    torch.testing.assert_close(clean_invalid.value_loss, extreme_invalid.value_loss)
    torch.testing.assert_close(clean_invalid.total_loss, extreme_invalid.total_loss)


def test_live_probe_storage_batch_backpropagates_only_valid_segment() -> None:
    policy = FakeSegmentPolicy()
    batch = _ppo_batch(invalid_action=1e6, invalid_reward_accum=1e6)
    result = compute_frontres_segment_ppo_loss(policy, batch, FrontRESSegmentPPOConfig(entropy_coef=0.0))

    result.total_loss.backward()

    _probe_tensor("ppo_batch.actions", batch.actions, "PPO action tensor before masked gradient")
    _probe_tensor("ppo_batch.valid_mask", batch.valid_mask, "only first row should contribute gradient")
    _probe_result("backward_result", result)
    _probe_tensor("actor.weight.grad", policy.actor.weight.grad, "actor gradient after masked PPO backward")
    _probe_tensor("critic.weight.grad", policy.critic.weight.grad, "critic gradient after masked value backward")

    assert result.should_step
    assert result.valid_count == 1
    assert policy.actor.weight.grad is not None
    assert policy.critic.weight.grad is not None
    assert torch.count_nonzero(policy.actor.weight.grad[:, 0]) > 0
    assert torch.count_nonzero(policy.actor.weight.grad[:, 1:]) == 0
    assert torch.count_nonzero(policy.critic.weight.grad[:, 0]) > 0
    assert torch.count_nonzero(policy.critic.weight.grad[:, 1:]) == 0


if __name__ == "__main__":
    test_live_probe_storage_batch_masks_invalid_segment_before_ppo_loss()
    test_live_probe_storage_batch_backpropagates_only_valid_segment()
    print("frontres_segment_live_probe_ppo_contract: ok")
