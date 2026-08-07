#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str((ROOT / "rsl_rl").joinpath(*name.split(".")[1:]))]
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
    reset_module = _load(
        "rsl_rl.frontres.frontres_segment_reset",
        ROOT / "rsl_rl" / "frontres" / "frontres_segment_reset.py",
    )
    frontres_pkg.frontres_segment_reset = reset_module
    frontres_pkg.frontres_balance = _load(
        "rsl_rl.frontres.frontres_balance",
        ROOT / "rsl_rl" / "frontres" / "frontres_balance.py",
    )
    frontres_pkg.frontres_interfaces = _load(
        "rsl_rl.frontres.frontres_interfaces",
        ROOT / "rsl_rl" / "frontres" / "frontres_interfaces.py",
    )
    warmup_module = _load(
        "rsl_rl.frontres.frontres_segment_warmup",
        ROOT / "rsl_rl" / "frontres" / "frontres_segment_warmup.py",
    )
    frontres_pkg.frontres_segment_warmup = warmup_module
    q2d_module = _load(
        "rsl_rl.frontres.frontres_policy_quality_q2d",
        ROOT / "rsl_rl" / "frontres" / "frontres_policy_quality_q2d.py",
    )
    frontres_pkg.frontres_policy_quality_q2d = q2d_module

    training_schedule = types.ModuleType("rsl_rl.frontres.training_schedule")
    training_schedule.resolve_frontres_mode_state = lambda *_args, **_kwargs: None
    sys.modules[training_schedule.__name__] = training_schedule
    frontres_pkg.training_schedule = training_schedule

    training_setup = types.ModuleType("rsl_rl.runners.frontres_training_setup")
    training_setup.configure_frontres_pair_layout = lambda *_args, **_kwargs: None
    sys.modules[training_setup.__name__] = training_setup
    runners_pkg.frontres_training_setup = training_setup

    modules_pkg = _package("rsl_rl.modules")
    modules_pkg.FrontRESActorCritic = object
    sys.modules[modules_pkg.__name__] = modules_pkg
    rsl_rl_pkg.modules = modules_pkg

    rollout_step = types.ModuleType("rsl_rl.runners.frontres_rollout_step")
    rollout_step.append_frontres_future_intent_actor_context = lambda _runner, obs: obs
    rollout_step.frontres_motion_command = lambda runner: runner.env.command_manager.get_term("motion")
    rollout_step.prepare_frontres_rollout_step = lambda *_args, **_kwargs: None
    rollout_step.prepare_frontres_v015_frozen_gmt_step = lambda *_args, **_kwargs: None
    rollout_step.prepare_frontres_v015_one_action_at_t = lambda *_args, **_kwargs: None
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
        f"result_param_delta_max_abs={getattr(result, 'param_delta_max_abs', 0.0):.6f} "
        f"result_param_delta_l2={getattr(result, 'param_delta_l2', 0.0):.6f} "
        f"result_param_delta_changed={getattr(result, 'param_delta_changed', 0)}/"
        f"{getattr(result, 'param_delta_total', 0)} "
        f"result_param_grad_norm={getattr(result, 'param_grad_norm', 0.0):.6f} "
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
        self.action_mean: torch.Tensor | None = None
        self.action_std: torch.Tensor | None = None
        self.actor_obs_trace: list[tuple[int, int]] = []
        self.critic_obs_trace: list[tuple[int, int]] = []

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        self.actor_obs_trace.append(tuple(observations.shape))
        self.last_mean = self.actor(observations)
        self.action_mean = self.last_mean
        self.action_std = torch.ones_like(self.last_mean)
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
        self.schedule = "fixed"
        self.desired_kl = 0.01
        self.learning_rate = 0.1
        self.frontres_segment_trust_region_max_retries = 2

    def _get_actor_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.policy.last_mean is not None
        return -0.5 * (actions - self.policy.last_mean).square().sum(dim=-1)


class FakeRunner:
    def __init__(self) -> None:
        self.alg = FakeAlg()
        self.current_learning_iteration = 0
        self.mode_trace: list[str] = []

    def train_mode(self) -> None:
        self.mode_trace.append("train")

    def eval_mode(self) -> None:
        self.mode_trace.append("eval")


def _storage_batch(valid_mask: torch.Tensor, old_means: torch.Tensor | None = None) -> object:
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
            old_means=torch.zeros(2, 6) if old_means is None else old_means,
            old_sigmas=torch.ones(2, 6),
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
    assert runner.alg.policy.actor_obs_trace == [(2, 4), (2, 4)]
    assert runner.alg.policy.critic_obs_trace == [(2, 3), (2, 3)]
    assert result.param_delta_total == 2
    assert result.param_delta_changed == 2
    assert result.param_delta_max_abs > 0.0
    assert result.param_delta_l2 > 0.0
    assert result.param_grad_norm > 0.0


def test_single_update_captures_real_credit_tuple_before_optimizer() -> None:
    runner = FakeRunner()
    result_path = ROOT / "rsl_rl" / "tests" / ".frontres_q2d_credit_tuple.json"
    runner._frontres_policy_quality_q2d_credit_result = str(result_path)
    storage_batch = replace(
        _storage_batch(torch.tensor([True, False])),
        audit_transaction_id="txn-live-1",
        audit_batch_signature="batch-live-1",
        audit_identity_state="complete",
    )
    before_rewards = storage_batch.rewards.detach().clone()
    before_advantages = storage_batch.advantages.detach().clone()

    run_frontres_segment_single_update(runner, storage_batch)
    payload = json.loads(result_path.read_text())
    result_path.unlink()

    assert payload["schema_version"] == "frontres_policy_quality_q2d_credit_v1"
    assert payload["audit_transaction_id"] == "txn-live-1"
    assert payload["audit_batch_signature"] == "batch-live-1"
    assert payload["row_count"] == 2
    assert len(payload["raw_actions"]) == len(payload["old_means"]) == 2
    torch.testing.assert_close(storage_batch.rewards, before_rewards)
    torch.testing.assert_close(storage_batch.advantages, before_advantages)


def test_dp09_critic_only_phase_holds_actor_and_updates_critic() -> None:
    runner = FakeRunner()
    runner.alg.frontres_segment_critic_warmup_iterations = 2
    runner.alg.frontres_segment_actor_warmup_iterations = 4
    before_actor = runner.alg.policy.actor.weight.detach().clone()
    before_critic = runner.alg.policy.critic.weight.detach().clone()

    result = run_frontres_segment_single_update(runner, _storage_batch(torch.tensor([True, False])))

    assert result.warmup_phase == "critic_only"
    assert result.actor_loss_weight == 0.0
    assert torch.equal(runner.alg.policy.actor.weight, before_actor)
    assert not torch.equal(runner.alg.policy.critic.weight, before_critic)


def test_dp09_actor_ramp_phase_uses_linear_actor_weight() -> None:
    runner = FakeRunner()
    runner.alg.frontres_segment_critic_warmup_iterations = 2
    runner.alg.frontres_segment_actor_warmup_iterations = 4
    runner.current_learning_iteration = 2
    before_actor = runner.alg.policy.actor.weight.detach().clone()

    result = run_frontres_segment_single_update(runner, _storage_batch(torch.tensor([True, False])))

    assert result.warmup_phase == "actor_ramp"
    assert result.actor_loss_weight == 0.25
    assert not torch.equal(runner.alg.policy.actor.weight, before_actor)


def test_segment_live_update_uses_scale_only_advantages_independent_of_base_ppo_flag() -> None:
    runner = FakeRunner()
    runner.alg.normalize_advantage_per_mini_batch = False
    runner.alg.frontres_segment_trust_region_rollback = False
    storage_batch = _storage_batch(torch.tensor([True, True]))

    result = run_frontres_segment_single_update(runner, storage_batch)
    print(
        "[probe step3] segment_advantage_scale_only: "
        f"base_flag={runner.alg.normalize_advantage_per_mini_batch} "
        f"adv_scale={result.advantage_scale:.6f} "
        f"sign_flips={result.advantage_sign_flip_count} "
        f"adv_mean={result.advantage_mean:.6f} "
        f"adv_min={result.advantage_min:.6f} "
        f"adv_max={result.advantage_max:.6f} "
        f"adv_abs_max={result.advantage_abs_max:.6f}",
        flush=True,
    )

    assert result.valid_count == 2
    assert abs(result.advantage_scale - torch.sqrt(torch.tensor((1.0**2 + 1000.0**2) / 2.0)).item()) < 1e-3
    assert result.advantage_sign_flip_count == 0
    assert result.advantage_min > 0.0
    assert result.advantage_max > result.advantage_min


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
    assert result.param_delta_total == 2
    assert result.param_delta_changed == 0
    assert result.param_delta_max_abs == 0.0
    assert result.param_delta_l2 == 0.0
    assert result.param_grad_norm == 0.0
    torch.testing.assert_close(runner.alg.policy.actor.weight.detach(), before_actor)
    torch.testing.assert_close(runner.alg.policy.critic.weight.detach(), before_critic)


def test_single_update_applies_mosaic_style_adaptive_lr_from_old_stats_kl() -> None:
    runner = FakeRunner()
    runner.alg.schedule = "adaptive"
    runner.alg.desired_kl = 1e-5
    runner.alg.learning_rate = 0.1
    runner.alg.frontres_segment_trust_region_rollback = False
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = runner.alg.learning_rate
    storage_batch = _storage_batch(torch.tensor([True, False]), old_means=torch.full((2, 6), 0.5))

    result = run_frontres_segment_single_update(runner, storage_batch)
    print(
        "[probe step3] adaptive_lr_old_stats: "
        f"distribution_kl_mean={result.distribution_kl_mean:.6f} "
        f"post_distribution_kl_mean={result.post_update_distribution_kl_mean:.6f} "
        f"adaptive_lr_kl_mean={result.adaptive_lr_kl_mean:.6f} "
        f"mosaic_pre_step_kl={result.mosaic_pre_step_adaptive_lr_kl_mean:.6f} "
        f"desired_kl={runner.alg.desired_kl:.6f} "
        f"learning_rate_after={runner.alg.learning_rate:.8f} "
        f"param_group_lr={runner.alg.optimizer.param_groups[0]['lr']:.8f}",
        flush=True,
    )

    assert result.distribution_kl_mean > runner.alg.desired_kl * 2.0
    assert abs(result.adaptive_lr_kl_mean - result.distribution_kl_mean) < 1e-8
    assert abs(result.mosaic_pre_step_adaptive_lr_kl_mean - result.distribution_kl_mean) < 1e-8
    assert result.mosaic_pre_step_adaptive_lr_before == 0.1
    assert result.mosaic_pre_step_adaptive_lr_after < result.mosaic_pre_step_adaptive_lr_before
    assert runner.alg.learning_rate < 0.1
    assert runner.alg.optimizer.param_groups[0]["lr"] == runner.alg.learning_rate
    assert result.adaptive_lr_desired_kl == runner.alg.desired_kl


def test_single_update_uses_mosaic_pre_step_lr_for_optimizer_step() -> None:
    old_means = torch.full((2, 6), 0.5)
    fixed_runner = FakeRunner()
    fixed_runner.alg.schedule = "fixed"
    fixed_runner.alg.learning_rate = 0.1
    for group in fixed_runner.alg.optimizer.param_groups:
        group["lr"] = fixed_runner.alg.learning_rate
    fixed_result = run_frontres_segment_single_update(
        fixed_runner,
        _storage_batch(torch.tensor([True, False]), old_means=old_means),
    )

    adaptive_runner = FakeRunner()
    adaptive_runner.alg.schedule = "adaptive"
    adaptive_runner.alg.desired_kl = 0.01
    adaptive_runner.alg.learning_rate = 0.1
    adaptive_runner.alg.frontres_segment_trust_region_rollback = False
    for group in adaptive_runner.alg.optimizer.param_groups:
        group["lr"] = adaptive_runner.alg.learning_rate
    adaptive_result = run_frontres_segment_single_update(
        adaptive_runner,
        _storage_batch(torch.tensor([True, False]), old_means=old_means),
    )

    print(
        "[probe step3] mosaic_pre_step_lr_controls_step: "
        f"fixed_delta_l2={fixed_result.param_delta_l2:.6f} "
        f"adaptive_delta_l2={adaptive_result.param_delta_l2:.6f} "
        f"pre_kl={adaptive_result.mosaic_pre_step_adaptive_lr_kl_mean:.6f} "
        f"pre_lr_before={adaptive_result.mosaic_pre_step_adaptive_lr_before:.8f} "
        f"pre_lr_after={adaptive_result.mosaic_pre_step_adaptive_lr_after:.8f}",
        flush=True,
    )

    assert fixed_result.param_delta_l2 > 0.0
    assert adaptive_result.param_delta_l2 > 0.0
    assert adaptive_result.mosaic_pre_step_adaptive_lr_after < adaptive_result.mosaic_pre_step_adaptive_lr_before
    assert adaptive_result.param_delta_l2 < fixed_result.param_delta_l2 * 0.2


def test_segment_adaptive_lr_does_not_pre_increase_from_near_zero_pre_kl() -> None:
    runner = FakeRunner()
    runner.alg.schedule = "adaptive"
    runner.alg.desired_kl = 0.01
    runner.alg.learning_rate = 0.1
    runner.alg.frontres_segment_trust_region_rollback = False
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = runner.alg.learning_rate
    storage_batch = _storage_batch(torch.tensor([True, False]))

    result = run_frontres_segment_single_update(runner, storage_batch)
    print(
        "[probe step3] adaptive_lr_no_pre_increase_on_near_zero_pre_kl: "
        f"pre_kl={result.mosaic_pre_step_adaptive_lr_kl_mean:.9f} "
        f"pre_lr_before={result.mosaic_pre_step_adaptive_lr_before:.8f} "
        f"pre_lr_after={result.mosaic_pre_step_adaptive_lr_after:.8f} "
        f"allow_increase={result.mosaic_pre_step_adaptive_lr_allow_increase} "
        f"post_kl={result.post_update_distribution_kl_mean:.9f}",
        flush=True,
    )

    assert 0.0 < result.mosaic_pre_step_adaptive_lr_kl_mean < runner.alg.desired_kl / 2.0
    assert result.mosaic_pre_step_adaptive_lr_allow_increase == 0
    assert result.mosaic_pre_step_adaptive_lr_after == result.mosaic_pre_step_adaptive_lr_before
    assert runner.alg.learning_rate == result.mosaic_pre_step_adaptive_lr_before


def test_single_update_reports_post_update_trust_region_kl() -> None:
    runner = FakeRunner()
    runner.alg.learning_rate = 0.1
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = runner.alg.learning_rate
    storage_batch = _storage_batch(torch.tensor([True, False]))

    result = run_frontres_segment_single_update(runner, storage_batch)
    print(
        "[probe step3] post_update_trust_region_kl: "
        f"pre_distribution_kl={result.distribution_kl_mean:.6f} "
        f"post_distribution_kl={getattr(result, 'post_update_distribution_kl_mean', -1.0):.6f} "
        f"reported_kl={result.approx_kl:.6f} "
        f"param_delta_l2={result.param_delta_l2:.6f}",
        flush=True,
    )

    assert result.param_delta_l2 > 0.0
    assert result.distribution_kl_mean < 1e-4
    assert result.post_update_distribution_kl_mean > result.distribution_kl_mean
    assert abs(result.approx_kl - result.post_update_distribution_kl_mean) < 1e-8


def test_direct_delta_se_logprob_uses_same_action_source_as_policy_stats() -> None:
    policy = types.SimpleNamespace(num_task_corrections=6)
    actions = torch.tensor([[0.30, -0.20, 0.10, 0.50, -0.40, 0.20]])
    mean = torch.tensor([[0.10, -0.05, 0.00, 0.20, -0.10, 0.00]])
    std = torch.full((1, 6), 0.50)
    observed = live_probe._evaluate_segment_delta_se_log_prob_from_stats(
        policy,
        actions,
        mean,
        std,
    )
    expected = torch.distributions.Normal(mean, std).log_prob(actions).sum(dim=-1)
    print(
        "[probe step3] bounded_logprob_source: "
        f"action={actions[0].tolist()} "
        f"observed={observed.item():.9f} "
        f"expected={expected.item():.9f}",
        flush=True,
    )

    torch.testing.assert_close(observed, expected)


def test_single_update_reports_post_update_mean_delta_from_old_distribution() -> None:
    runner = FakeRunner()
    runner.alg.learning_rate = 0.1
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = runner.alg.learning_rate
    storage_batch = _storage_batch(torch.tensor([True, False]))

    result = run_frontres_segment_single_update(runner, storage_batch)
    print(
        "[probe step3] post_update_mean_delta: "
        f"pre_mean_delta_l2={result.distribution_mean_delta_l2_mean:.6f} "
        f"post_mean_delta_l2={result.post_update_mean_delta_l2_mean:.6f} "
        f"post_mean_delta_max={result.post_update_mean_delta_max_abs:.6f} "
        f"old_sigma_min={result.post_update_old_sigma_min:.6f} "
        f"sigma_min={result.post_update_sigma_min:.6f}",
        flush=True,
    )

    assert result.distribution_mean_delta_l2_mean < 1e-8
    assert result.post_update_mean_delta_l2_mean > 0.0
    assert result.post_update_mean_delta_max_abs > 0.0
    assert result.post_update_old_sigma_min == 1.0
    assert result.post_update_sigma_min == 1.0


def test_single_update_rejects_explosive_adaptive_post_kl_and_reports_post_ratio_max() -> None:
    runner = FakeRunner()
    runner.alg.schedule = "adaptive"
    runner.alg.desired_kl = 1e-5
    runner.alg.learning_rate = 0.1
    runner.alg.frontres_segment_trust_region_max_retries = 0
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = runner.alg.learning_rate
    before_actor = runner.alg.policy.actor.weight.detach().clone()
    before_critic = runner.alg.policy.critic.weight.detach().clone()
    storage_batch = _storage_batch(torch.tensor([True, False]))

    result = run_frontres_segment_single_update(runner, storage_batch)
    print(
        "[probe step3] trust_region_rejects_post_kl: "
        f"post_kl={result.post_update_distribution_kl_mean:.6f} "
        f"post_ratio_mean={result.post_update_ratio_mean:.6e} "
        f"post_ratio_max={result.post_update_ratio_max:.6e} "
        f"rejected={result.trust_region_rejected_count} "
        f"accepted={result.trust_region_accepted} "
        f"lr_after={runner.alg.learning_rate:.8f} "
        f"param_delta_l2={result.param_delta_l2:.6f}",
        flush=True,
    )

    assert result.post_update_distribution_kl_mean > runner.alg.desired_kl * 2.0
    assert result.trust_region_rejected_count == 1
    assert result.trust_region_accepted == 0
    assert result.param_delta_changed == 0
    assert result.param_delta_l2 == 0.0
    assert result.post_update_ratio_max >= result.post_update_ratio_mean
    assert result.ratio_max == result.post_update_ratio_max
    assert runner.alg.learning_rate < 0.1
    torch.testing.assert_close(runner.alg.policy.actor.weight.detach(), before_actor)
    torch.testing.assert_close(runner.alg.policy.critic.weight.detach(), before_critic)


def test_single_update_requires_separate_pre_and_post_ratio_diagnostics() -> None:
    runner = FakeRunner()
    runner.alg.learning_rate = 0.1
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = runner.alg.learning_rate
    storage_batch = _storage_batch(torch.tensor([True, False]))

    result = run_frontres_segment_single_update(runner, storage_batch)
    required_fields = (
        "pre_update_raw_log_ratio_mean",
        "pre_update_raw_log_ratio_min",
        "pre_update_raw_log_ratio_max",
        "pre_update_clamped_ratio_mean",
        "pre_update_clamped_ratio_max",
        "post_update_raw_log_ratio_mean",
        "post_update_raw_log_ratio_min",
        "post_update_raw_log_ratio_max",
        "post_update_clamped_ratio_mean",
        "post_update_clamped_ratio_max",
    )
    missing = [name for name in required_fields if not hasattr(result, name)]
    print(
        "[probe stepA] ratio_diagnostic_contract: "
        f"missing={missing} "
        f"legacy_raw_log_ratio_mean={getattr(result, 'raw_log_ratio_mean', 'missing')} "
        f"legacy_ratio_mean={getattr(result, 'ratio_mean', 'missing')} "
        f"post_ratio_mean={getattr(result, 'post_update_ratio_mean', 'missing')} "
        f"post_ratio_max={getattr(result, 'post_update_ratio_max', 'missing')}",
        flush=True,
    )

    assert not missing, f"missing separated pre/post ratio diagnostics: {missing}"
    assert result.pre_update_clamped_ratio_max >= result.pre_update_clamped_ratio_mean
    assert result.post_update_clamped_ratio_max >= result.post_update_clamped_ratio_mean
    assert result.post_update_raw_log_ratio_mean != result.pre_update_raw_log_ratio_mean


if __name__ == "__main__":
    test_single_update_steps_optimizer_with_valid_segment()
    test_single_update_captures_real_credit_tuple_before_optimizer()
    test_dp09_critic_only_phase_holds_actor_and_updates_critic()
    test_dp09_actor_ramp_phase_uses_linear_actor_weight()
    test_segment_live_update_uses_scale_only_advantages_independent_of_base_ppo_flag()
    test_single_update_does_not_step_optimizer_without_valid_segments()
    test_single_update_applies_mosaic_style_adaptive_lr_from_old_stats_kl()
    test_single_update_uses_mosaic_pre_step_lr_for_optimizer_step()
    test_segment_adaptive_lr_does_not_pre_increase_from_near_zero_pre_kl()
    test_single_update_reports_post_update_trust_region_kl()
    test_direct_delta_se_logprob_uses_same_action_source_as_policy_stats()
    test_single_update_reports_post_update_mean_delta_from_old_distribution()
    test_single_update_rejects_explosive_adaptive_post_kl_and_reports_post_ratio_max()
    test_single_update_requires_separate_pre_and_post_ratio_diagnostics()
    print("frontres_segment_live_single_update_contract: ok")
