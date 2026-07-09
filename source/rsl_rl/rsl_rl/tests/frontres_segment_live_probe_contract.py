#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
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


def _install_live_probe_import_stubs():
    rsl_rl_pkg = _package("rsl_rl")
    algorithms_pkg = _package("rsl_rl.algorithms")
    frontres_pkg = _package("rsl_rl.frontres")
    runners_pkg = _package("rsl_rl.runners")

    rsl_rl_pkg.algorithms = algorithms_pkg
    rsl_rl_pkg.frontres = frontres_pkg
    rsl_rl_pkg.runners = runners_pkg

    algorithms_pkg.FrontRESUnified = object

    ppo_module = types.ModuleType("rsl_rl.algorithms.frontres_segment_ppo")
    ppo_module.FrontRESSegmentPPOBatch = object
    ppo_module.FrontRESSegmentPPOConfig = object

    def _unused_ppo_loss(*_args, **_kwargs):
        raise AssertionError("Step 1 storage test must not enter PPO loss")

    ppo_module.compute_frontres_segment_ppo_loss = _unused_ppo_loss
    sys.modules[ppo_module.__name__] = ppo_module
    algorithms_pkg.frontres_segment_ppo = ppo_module

    training_schedule = types.ModuleType("rsl_rl.frontres.training_schedule")
    training_schedule.resolve_frontres_mode_state = lambda *_args, **_kwargs: SimpleNamespace(
        is_frontres=True,
        is_task_space_mode=True,
    )
    sys.modules[training_schedule.__name__] = training_schedule
    frontres_pkg.training_schedule = training_schedule

    training_setup = types.ModuleType("rsl_rl.runners.frontres_training_setup")
    training_setup.configure_frontres_pair_layout = lambda *_args, **_kwargs: SimpleNamespace(
        n_train=1,
        n_candidate=0,
        n_base=0,
        n_clean=0,
    )
    sys.modules[training_setup.__name__] = training_setup
    runners_pkg.frontres_training_setup = training_setup

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

    modules_pkg = types.ModuleType("rsl_rl.modules")
    modules_pkg.FrontRESActorCritic = object
    sys.modules[modules_pkg.__name__] = modules_pkg
    rsl_rl_pkg.modules = modules_pkg

    rollout_step = types.ModuleType("rsl_rl.runners.frontres_rollout_step")

    def _prepare_frontres_rollout_step(runner, **kwargs):
        batch = int(kwargs["obs"].shape[0])
        actions = getattr(runner, "_frontres_test_policy_action", torch.zeros(batch, 6)).detach().clone()
        runner.alg.transition.observations = kwargs["obs"].detach().clone()
        runner.alg.transition.privileged_observations = kwargs["privileged_obs"].detach().clone()
        runner.alg.transition.actions_log_prob = torch.zeros(batch)
        runner.alg.transition.values = torch.zeros(batch)
        runner.alg.transition.action_mean = actions.detach().clone()
        runner.alg.transition.action_sigma = torch.ones_like(actions)
        return SimpleNamespace(actions=actions, env_actions=actions.detach().clone())

    rollout_step.prepare_frontres_rollout_step = _prepare_frontres_rollout_step
    sys.modules[rollout_step.__name__] = rollout_step
    runners_pkg.frontres_rollout_step = rollout_step

    live_probe_module = _load(
        "rsl_rl.runners.frontres_segment_live_probe",
        ROOT / "rsl_rl" / "runners" / "frontres_segment_live_probe.py",
    )
    runners_pkg.frontres_segment_live_probe = live_probe_module
    return live_probe_module


live_probe = _install_live_probe_import_stubs()
FrontRESSegmentLiveRolloutCapture = live_probe.FrontRESSegmentLiveRolloutCapture
FrontRESSegmentLiveObservations = live_probe.FrontRESSegmentLiveObservations
build_live_segment_storage = live_probe.build_live_segment_storage
run_frontres_segment_live_probe = live_probe.run_frontres_segment_live_probe
run_live_rollout_capture = live_probe._run_live_rollout_capture


def _probe_tensor(name: str, tensor: torch.Tensor, semantic: str) -> None:
    data = tensor.detach().cpu()
    numeric = data.float() if data.dtype == torch.bool else data
    print(
        f"[probe step1] {name}: shape={tuple(data.shape)} dtype={data.dtype} "
        f"device={tensor.device} min={numeric.min().item():.6f} "
        f"max={numeric.max().item():.6f} mean={numeric.float().mean().item():.6f} "
        f"requires_grad={tensor.requires_grad} grad_fn={type(tensor.grad_fn).__name__ if tensor.grad_fn else None} "
        f"semantic={semantic}",
        flush=True,
    )


def _capture(actions: torch.Tensor | None = None) -> FrontRESSegmentLiveRolloutCapture:
    transition_actions = (
        actions
        if actions is not None
        else torch.tensor(
            [
                [0.10, -0.20, 0.00, 0.30, 0.00, -0.10],
                [-0.40, 0.00, 0.20, 0.00, -0.30, 0.50],
            ]
        )
    )
    return FrontRESSegmentLiveRolloutCapture(
        rollout_k=2,
        reward_mean=1.5,
        done_frac=0.5,
        last_obs_shape=(2, 4),
        action_shape=tuple(transition_actions.shape),
        env_action_shape=(2, 12),
        transition_obs=torch.arange(8, dtype=torch.float32).reshape(2, 4),
        transition_privileged_obs=torch.arange(6, dtype=torch.float32).reshape(2, 3),
        transition_actions=transition_actions,
        transition_log_probs=torch.tensor([-0.1, -0.2]),
        transition_values=torch.tensor([0.5, -0.5]),
        transition_means=transition_actions + 0.1,
        transition_sigmas=torch.full_like(transition_actions, 0.2),
        reward_accum=torch.tensor([2.0, 4.0]),
        reward_steps=torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
        done_steps=torch.tensor([[False, False], [False, True]]),
        done_any=torch.tensor([False, True]),
        n_train=2,
        n_base=0,
        n_clean=0,
    )


def test_build_live_segment_storage_preserves_first_step_tuple_trace() -> None:
    runner = SimpleNamespace(device=torch.device("cpu"))
    capture = _capture()

    storage = build_live_segment_storage(runner, capture)
    batch = storage.full_batch()

    expected_rewards = capture.reward_accum.reshape(-1) / float(capture.rollout_k)
    expected_valid = ~capture.done_any.reshape(-1).bool()
    _probe_tensor("capture.transition_actions", capture.transition_actions, "live first-step 6D policy action")
    _probe_tensor("storage.actions", storage.actions[: storage.step], "stored 6D action after add_transition")
    _probe_tensor("batch.actions", batch.actions, "full_batch 6D action consumed downstream")
    _probe_tensor("capture.reward_accum", capture.reward_accum, "K-step accumulated env reward before averaging")
    _probe_tensor("expected_rewards", expected_rewards, "reward_accum divided by rollout_k")
    _probe_tensor("storage.rewards", storage.rewards[: storage.step], "stored averaged segment reward")
    _probe_tensor("batch.returns", batch.returns, "PPO return uses captured K-step reward trace")
    _probe_tensor("capture.done_any", capture.done_any, "whether any env done occurred during K-step rollout")
    _probe_tensor("expected_valid", expected_valid, "valid segment mask derived as not done_any")
    _probe_tensor("storage.valid_mask", storage.valid_mask[: storage.step], "stored valid mask")
    _probe_tensor("batch.valid_mask", batch.valid_mask, "full_batch valid mask consumed by PPO")
    _probe_tensor("batch.segment_ids", batch.segment_ids, "live storage assigns one segment id per env row")

    assert storage.step == 2
    assert batch.actions.shape == (2, 6)
    torch.testing.assert_close(batch.actions, capture.transition_actions)
    torch.testing.assert_close(storage.rewards[: storage.step], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(batch.returns, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(batch.advantages, torch.tensor([1.5, 4.5]))
    assert batch.valid_mask.tolist() == [True, False]
    assert storage.valid_mask[: storage.step].tolist() == [True, False]
    assert batch.segment_ids.tolist() == [0, 1]
    assert storage.segment_source == ["live_storage_probe", "live_storage_probe"]
    torch.testing.assert_close(batch.old_log_probs, torch.tensor([-0.1, -0.2]))
    torch.testing.assert_close(batch.old_values, torch.tensor([0.5, -0.5]))
    torch.testing.assert_close(batch.old_means, capture.transition_means)
    torch.testing.assert_close(batch.old_sigmas, capture.transition_sigmas)
    assert batch.action_mask.shape == (2, 6)
    assert batch.action_mask.bool().all().item()


def test_build_live_segment_storage_uses_b1_paired_gain_when_available() -> None:
    runner = SimpleNamespace(device=torch.device("cpu"))
    capture = _capture()
    actions = torch.cat([capture.transition_actions, torch.zeros(4, 6)], dim=0)
    capture = FrontRESSegmentLiveRolloutCapture(
        rollout_k=capture.rollout_k,
        reward_mean=capture.reward_mean,
        done_frac=0.0,
        last_obs_shape=capture.last_obs_shape,
        action_shape=tuple(actions.shape),
        env_action_shape=capture.env_action_shape,
        transition_obs=torch.zeros(6, 4),
        transition_privileged_obs=torch.zeros(6, 3),
        transition_actions=actions,
        transition_log_probs=torch.zeros(6),
        transition_values=torch.zeros(6),
        transition_means=actions,
        transition_sigmas=torch.ones_like(actions),
        reward_accum=torch.tensor([0.2, 0.8, 0.1, 0.6, 1.0, 1.0]),
        reward_steps=torch.tensor(
            [
                [0.1, 0.4, 0.05, 0.3, 0.5, 0.5],
                [0.1, 0.4, 0.05, 0.3, 0.5, 0.5],
            ]
        ),
        done_steps=torch.zeros(2, 6, dtype=torch.bool),
        done_any=torch.tensor([False, False, False, False, False, False]),
        actor_update_mask=torch.tensor([True, True, False, False, False, False]),
        n_train=2,
        n_candidate=0,
        n_base=2,
        n_clean=2,
    )

    storage = build_live_segment_storage(runner, capture)
    batch = storage.full_batch()

    _probe_tensor("capture.reward_accum", capture.reward_accum, "B1 quartet raw scores: repaired, noisy, clean")
    _probe_tensor("batch.returns", batch.returns, "PPO should learn K-step repaired-minus-noisy gain when paired scores exist")
    torch.testing.assert_close(batch.returns[:2], torch.tensor([0.10, 0.20]))
    torch.testing.assert_close(batch.advantages[:2], torch.tensor([0.10, 0.20]))
    assert batch.valid_mask.tolist() == [True, True, False, False, False, False]


def test_build_live_segment_storage_uses_discounted_reward_trace_for_ppo_returns() -> None:
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        alg=SimpleNamespace(gamma=0.9, frontres_segment_k=4),
    )
    capture = _capture()
    capture = FrontRESSegmentLiveRolloutCapture(
        rollout_k=4,
        reward_mean=capture.reward_mean,
        done_frac=0.0,
        last_obs_shape=capture.last_obs_shape,
        action_shape=capture.action_shape,
        env_action_shape=capture.env_action_shape,
        transition_obs=capture.transition_obs,
        transition_privileged_obs=capture.transition_privileged_obs,
        transition_actions=capture.transition_actions,
        transition_log_probs=capture.transition_log_probs,
        transition_values=torch.tensor([0.1, 0.2]),
        transition_means=capture.transition_means,
        transition_sigmas=capture.transition_sigmas,
        reward_accum=torch.tensor([-5.0, 1.25]),
        reward_steps=torch.tensor(
            [
                [1.0, 0.5],
                [-2.0, 0.25],
                [-2.0, 0.25],
                [-2.0, 0.25],
            ]
        ),
        done_steps=torch.tensor(
            [
                [False, False],
                [False, True],
                [False, False],
                [False, False],
            ]
        ),
        done_any=torch.tensor([False, False]),
        actor_update_mask=torch.tensor([True, True]),
        n_train=2,
        n_base=0,
        n_clean=0,
    )

    storage = build_live_segment_storage(runner, capture)
    batch = storage.full_batch()

    expected_first = 1.0 + 0.9 * -2.0 + 0.9 * 0.9 * -2.0 + 0.9 * 0.9 * 0.9 * -2.0
    expected_second = 0.5 + 0.9 * 0.25
    _probe_tensor("capture.reward_steps", capture.reward_steps, "per-step executable reward trace")
    _probe_tensor("capture.done_steps", capture.done_steps, "per-step done mask for K-step return")
    _probe_tensor("batch.returns", batch.returns, "discounted K-step return consumed by Segment PPO")
    _probe_tensor("batch.advantages", batch.advantages, "discounted K-step return minus first-step value")
    torch.testing.assert_close(batch.returns, torch.tensor([expected_first, expected_second]))
    torch.testing.assert_close(batch.advantages, batch.returns - torch.tensor([0.1, 0.2]))
    assert batch.returns[0] < 0.0


def test_build_live_segment_storage_masks_non_actor_rows() -> None:
    runner = SimpleNamespace(device=torch.device("cpu"))
    capture = _capture()
    capture = FrontRESSegmentLiveRolloutCapture(
        rollout_k=capture.rollout_k,
        reward_mean=capture.reward_mean,
        done_frac=0.0,
        last_obs_shape=capture.last_obs_shape,
        action_shape=capture.action_shape,
        env_action_shape=capture.env_action_shape,
        transition_obs=capture.transition_obs,
        transition_privileged_obs=capture.transition_privileged_obs,
        transition_actions=capture.transition_actions,
        transition_log_probs=capture.transition_log_probs,
        transition_values=capture.transition_values,
        transition_means=capture.transition_means,
        transition_sigmas=capture.transition_sigmas,
        reward_accum=capture.reward_accum,
        reward_steps=capture.reward_steps,
        done_steps=capture.done_steps,
        done_any=torch.tensor([False, False]),
        actor_update_mask=torch.tensor([True, False]),
    )

    storage = build_live_segment_storage(runner, capture)
    batch = storage.full_batch()

    _probe_tensor("capture.actor_update_mask", capture.actor_update_mask, "only actor-owned quartet rows may update PPO")
    _probe_tensor("capture.done_any", capture.done_any, "rollout survived mask before actor ownership")
    _probe_tensor("storage.valid_mask", storage.valid_mask[: storage.step], "PPO-valid rows after actor ownership masking")
    _probe_tensor("batch.valid_mask", batch.valid_mask, "full_batch valid mask consumed by PPO")

    assert storage.valid_mask[: storage.step].tolist() == [True, False]
    assert batch.valid_mask.tolist() == [True, False]


def test_build_live_segment_storage_rejects_non_6d_actions() -> None:
    runner = SimpleNamespace(device=torch.device("cpu"))
    capture = _capture(actions=torch.zeros(2, 5))

    try:
        build_live_segment_storage(runner, capture)
    except ValueError as exc:
        assert "requires 6D actions" in str(exc)
    else:
        raise AssertionError("non-6D live probe actions must be rejected before storage write")


def test_live_probe_selects_6d_delta_se_from_12d_rollout_action() -> None:
    raw_actions = torch.tensor(
        [
            [0.10, -0.05, 0.00, 0.03, -0.02, 0.01, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
            [-0.08, 0.02, 0.04, -0.01, 0.02, -0.03, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
        ],
        dtype=torch.float32,
    )
    action_mean = torch.zeros_like(raw_actions)
    action_sigma = torch.ones_like(raw_actions) * 0.5
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            transition=SimpleNamespace(
                actions_log_prob=torch.tensor([-9.0, -8.0]),
                action_mean=action_mean,
                action_sigma=action_sigma,
            ),
            policy=SimpleNamespace(
                num_task_corrections=6,
                max_delta_pos=0.3,
                max_delta_rpy=0.3,
            ),
        )
    )

    segment_actions, log_probs = live_probe._select_segment_transition_actions(runner, actions=raw_actions)
    expected_log_probs = live_probe._evaluate_segment_delta_se_log_prob_from_stats(
        runner.alg.policy,
        raw_actions[:, :6],
        action_mean,
        action_sigma,
    )

    _probe_tensor("raw_actions", raw_actions, "12D rollout action from legacy HSL+acceptance policy")
    _probe_tensor("segment_actions", segment_actions, "selected 6D Delta SE action for Segment Replay storage")
    _probe_tensor("selected_log_probs", log_probs, "old 6D log_prob rebuilt from rollout mean/sigma with Delta-SE transform")
    _probe_tensor("expected_log_probs", expected_log_probs, "same formula used by PPO eval for new 6D log_prob")
    _probe_tensor("transition_mean_6d", runner.alg.transition.action_mean[:, :6], "old mean sliced to same 6D action space")
    _probe_tensor("transition_sigma_6d", runner.alg.transition.action_sigma[:, :6], "old sigma sliced to same 6D action space")

    assert segment_actions.shape == (2, 6)
    torch.testing.assert_close(segment_actions, raw_actions[:, :6])
    torch.testing.assert_close(log_probs, expected_log_probs)


def test_live_probe_trace_prints_once_without_verbose() -> None:
    raw_actions = torch.arange(24, dtype=torch.float32).reshape(2, 12) * 0.1
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            frontres_segment_verbose_probe=False,
            transition=SimpleNamespace(actions_log_prob=torch.zeros(2)),
            policy=SimpleNamespace(
                get_actions_log_prob_selected=lambda actions, selected_dims: actions[:, selected_dims].sum(dim=-1)
            ),
        )
    )

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        live_probe._select_segment_transition_actions(runner, actions=raw_actions)
        live_probe._select_segment_transition_actions(runner, actions=raw_actions)
    output = buffer.getvalue()
    trace_count = output.count("[FrontRES Segment Live Probe Trace]")
    print(
        "[probe step4] live_probe_trace_rate: "
        f"trace_count={trace_count} "
        f"verbose={runner.alg.frontres_segment_verbose_probe}",
        flush=True,
    )

    assert trace_count == 1

    verbose_runner = SimpleNamespace(
        alg=SimpleNamespace(
            frontres_segment_verbose_probe=True,
            transition=SimpleNamespace(actions_log_prob=torch.zeros(2)),
            policy=runner.alg.policy,
        )
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        live_probe._select_segment_transition_actions(verbose_runner, actions=raw_actions)
        live_probe._select_segment_transition_actions(verbose_runner, actions=raw_actions)
    verbose_trace_count = buffer.getvalue().count("[FrontRES Segment Live Probe Trace]")
    print(
        "[probe step4] live_probe_trace_verbose_rate: "
        f"trace_count={verbose_trace_count} "
        f"verbose={verbose_runner.alg.frontres_segment_verbose_probe}",
        flush=True,
    )
    assert verbose_trace_count == 2


def test_live_probe_trace_reports_native_6d_policy_surface() -> None:
    actions = torch.arange(12, dtype=torch.float32).reshape(2, 6) * 0.1
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            frontres_segment_verbose_probe=False,
            transition=SimpleNamespace(actions_log_prob=torch.tensor([-1.0, -2.0])),
        )
    )

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        segment_actions, log_probs = live_probe._select_segment_transition_actions(runner, actions=actions)
        live_probe._select_segment_transition_actions(runner, actions=actions)
    output = buffer.getvalue()
    trace_count = output.count("[FrontRES Segment Live Probe Trace]")
    print(
        "[probe step5b] native_6d_live_probe_trace: "
        f"trace_count={trace_count} "
        f"native_6d={'semantic=storage_uses_native_6d_delta_se_policy' in output} "
        f"legacy_slice={'semantic=storage_uses_first_6_delta_se_dims' in output}",
        flush=True,
    )

    assert trace_count == 1
    assert "raw_action_shape=(2, 6)" in output
    assert "segment_action_shape=(2, 6)" in output
    assert "semantic=storage_uses_native_6d_delta_se_policy" in output
    assert "semantic=storage_uses_first_6_delta_se_dims" not in output
    torch.testing.assert_close(segment_actions, actions)
    torch.testing.assert_close(log_probs, torch.tensor([-1.0, -2.0]))


class _FakePPOEvalPolicy:
    def __init__(self) -> None:
        self.distribution = torch.distributions.Normal(torch.zeros(2, 6), torch.ones(2, 6))
        self.action_mean = None
        self.action_std = None

    def act(self, observations: torch.Tensor) -> None:
        self.action_mean = torch.zeros(observations.shape[0], 6)
        self.action_std = torch.ones(observations.shape[0], 6)

    def evaluate(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.zeros(observations.shape[0], 1)


def test_ppo_eval_trace_prints_once_without_verbose() -> None:
    alg = SimpleNamespace(
        frontres_segment_verbose_probe=False,
        policy=_FakePPOEvalPolicy(),
    )
    adapter = live_probe.FrontRESSegmentLivePolicyAdapter(alg, privileged_observations=None)
    observations = torch.zeros(2, 4)
    actions = torch.zeros(2, 6)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        adapter.evaluate_segment_actions(observations, actions)
        adapter.evaluate_segment_actions(observations, actions)
    output = buffer.getvalue()
    trace_count = output.count("[FrontRES Segment PPO Eval Trace]")
    print(
        "[probe step4] ppo_eval_trace_rate: "
        f"trace_count={trace_count} "
        f"verbose={alg.frontres_segment_verbose_probe}",
        flush=True,
    )

    assert trace_count == 1

    verbose_alg = SimpleNamespace(
        frontres_segment_verbose_probe=True,
        policy=_FakePPOEvalPolicy(),
    )
    verbose_adapter = live_probe.FrontRESSegmentLivePolicyAdapter(verbose_alg, privileged_observations=None)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        verbose_adapter.evaluate_segment_actions(observations, actions)
        verbose_adapter.evaluate_segment_actions(observations, actions)
    verbose_trace_count = buffer.getvalue().count("[FrontRES Segment PPO Eval Trace]")
    print(
        "[probe step4] ppo_eval_trace_verbose_rate: "
        f"trace_count={verbose_trace_count} "
        f"verbose={verbose_alg.frontres_segment_verbose_probe}",
        flush=True,
    )
    assert verbose_trace_count == 2


def test_build_live_segment_storage_masks_failed_reset_samples() -> None:
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        _frontres_segment_live_current_reset_result=SimpleNamespace(
            success_mask=torch.tensor([True, False]),
        ),
    )
    capture = _capture()
    capture = FrontRESSegmentLiveRolloutCapture(
        rollout_k=capture.rollout_k,
        reward_mean=capture.reward_mean,
        done_frac=0.0,
        last_obs_shape=capture.last_obs_shape,
        action_shape=capture.action_shape,
        env_action_shape=capture.env_action_shape,
        transition_obs=capture.transition_obs,
        transition_privileged_obs=capture.transition_privileged_obs,
        transition_actions=capture.transition_actions,
        transition_log_probs=capture.transition_log_probs,
        transition_values=capture.transition_values,
        transition_means=capture.transition_means,
        transition_sigmas=capture.transition_sigmas,
        reward_accum=capture.reward_accum,
        reward_steps=capture.reward_steps,
        done_steps=capture.done_steps,
        done_any=torch.tensor([False, False]),
    )

    storage = build_live_segment_storage(runner, capture)
    batch = storage.full_batch()
    stats = storage.stats()

    _probe_tensor(
        "reset_result.success_mask",
        runner._frontres_segment_live_current_reset_result.success_mask,
        "reset hook success per sampled segment",
    )
    _probe_tensor("capture.done_any", capture.done_any, "rollout done mask before storage validity")
    _probe_tensor("storage.reset_mask", storage.reset_mask[: storage.step], "reset success stored beside PPO tuple")
    _probe_tensor("storage.valid_mask", storage.valid_mask[: storage.step], "valid means reset succeeded and rollout survived")
    _probe_tensor("batch.valid_mask", batch.valid_mask, "PPO-valid rows after failed reset masking")
    print(
        "[probe step12] storage_reset_mask: "
        f"reset_success={runner._frontres_segment_live_current_reset_result.success_mask.tolist()} "
        f"done_any={capture.done_any.tolist()} "
        f"storage_reset={storage.reset_mask[: storage.step].tolist()} "
        f"storage_valid={storage.valid_mask[: storage.step].tolist()} "
        f"reset_success_frac={stats.reset_success_frac:.6f} "
        f"valid_frac={stats.valid_frac:.6f}",
        flush=True,
    )

    assert storage.reset_mask[: storage.step].tolist() == [True, False]
    assert storage.valid_mask[: storage.step].tolist() == [True, False]
    assert batch.valid_mask.tolist() == [True, False]
    assert stats.reset_success_frac == 0.5
    assert stats.valid_frac == 0.5


class _FakeLiveEnv:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.max_episode_length = 16
        self.events: list[str] = []

    def apply_frontres_segment_reset(self, request):
        self.events.append("reset")
        self.last_reset_request = request
        return {
            "success_mask": torch.ones(2, dtype=torch.bool),
            "velocity_mismatch": torch.zeros(2),
        }

    def get_observations(self):
        self.events.append("get_obs")
        obs = torch.ones(2, 4)
        return obs, {"observations": {}}

    def step(self, actions):
        self.events.append("step")
        self.last_step_actions = actions.detach().clone()
        obs = torch.ones(2, 4) * 2.0
        rewards = torch.tensor([1.0, 0.5])
        dones = torch.tensor([False, False])
        return obs, rewards, dones, {"observations": {}}


class _FakeIndexResetLiveEnv(_FakeLiveEnv):
    def apply_frontres_segment_index_reset(self, request):
        self.events.append("index_reset")
        self.last_index_reset_request = request
        return {
            "success_mask": torch.ones(int(request.segment_ids.numel()), dtype=torch.bool),
            "velocity_mismatch": torch.zeros(int(request.segment_ids.numel())),
        }


def test_live_rollout_capture_zero_segment_action_reaches_env_step() -> None:
    env = _FakeLiveEnv()
    runner = SimpleNamespace(
        env=env,
        device=torch.device("cpu"),
        training_type="frontres",
        policy_obs_type=None,
        privileged_obs_type=None,
        teacher_obs_type=None,
        ref_vel_estimator_obs_type=None,
        current_learning_iteration=0,
        _frontres_segment_replay_boundary=SimpleNamespace(segment_k=1),
        _frontres_test_policy_action=torch.tensor(
            [
                [0.2, -0.1, 0.3, 0.4, -0.5, 0.6],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        alg=SimpleNamespace(
            frontres_segment_k=1,
            transition=SimpleNamespace(),
            policy=SimpleNamespace(get_env_action=lambda _obs, actions: actions),
        ),
        _apply_obs_normalizer=lambda obs: obs,
        _apply_frontres_task_corrections=lambda *_args, **_kwargs: None,
    )
    observations = FrontRESSegmentLiveObservations(
        obs=torch.ones(2, 4),
        privileged_obs=torch.ones(2, 3),
        teacher_obs=torch.ones(2, 3),
        ref_vel_estimator_obs=None,
    )

    real_capture = run_live_rollout_capture(runner, observations, rollout_steps=1)
    real_step_actions = env.last_step_actions.clone()
    zero_capture = run_live_rollout_capture(runner, observations, rollout_steps=1, zero_segment_action=True)
    zero_step_actions = env.last_step_actions.clone()

    assert torch.linalg.norm(real_capture.transition_actions).item() > 0.0
    assert torch.linalg.norm(real_step_actions).item() > 0.0
    assert torch.linalg.norm(zero_capture.transition_actions).item() == 0.0
    assert torch.linalg.norm(zero_step_actions).item() == 0.0
    print(
        "[probe step11] zero_segment_action_reaches_env_step "
        f"real_norm={torch.linalg.norm(real_step_actions).item():.6f} "
        f"zero_norm={torch.linalg.norm(zero_step_actions).item():.6f}",
        flush=True,
    )


def test_live_rollout_capture_snapshots_signed_rp_perturbation() -> None:
    env = _FakeLiveEnv()
    env.command_manager = SimpleNamespace(
        _terms={
            "motion": SimpleNamespace(
                perturber=SimpleNamespace(
                    _roll_state=torch.tensor([0.10, -0.40]),
                    _pitch_state=torch.tensor([-0.20, 0.50]),
                    _iid_event_rp=torch.tensor([[0.03, -0.04], [0.00, 0.10]]),
                    _family_masks={"local_rp": torch.tensor([True, False])},
                    _baseline_mask=torch.tensor([False, False]),
                )
            )
        }
    )
    runner = SimpleNamespace(
        env=env,
        device=torch.device("cpu"),
        training_type="frontres",
        policy_obs_type=None,
        privileged_obs_type=None,
        teacher_obs_type=None,
        ref_vel_estimator_obs_type=None,
        current_learning_iteration=0,
        _frontres_segment_replay_boundary=SimpleNamespace(segment_k=1),
        _frontres_test_policy_action=torch.zeros(2, 6),
        alg=SimpleNamespace(
            frontres_segment_k=1,
            transition=SimpleNamespace(),
            policy=SimpleNamespace(get_env_action=lambda _obs, actions: actions),
        ),
        _apply_obs_normalizer=lambda obs: obs,
        _apply_frontres_task_corrections=lambda *_args, **_kwargs: None,
    )
    observations = FrontRESSegmentLiveObservations(
        obs=torch.ones(2, 4),
        privileged_obs=torch.ones(2, 3),
        teacher_obs=torch.ones(2, 3),
        ref_vel_estimator_obs=None,
    )

    capture = run_live_rollout_capture(runner, observations, rollout_steps=1)

    assert capture.transition_perturbation_rp is not None
    torch.testing.assert_close(
        capture.transition_perturbation_rp,
        torch.tensor([[0.13, -0.24], [0.0, 0.0]]),
    )
    print(
        "[probe step12] signed_rp_perturbation_snapshot "
        f"rp={capture.transition_perturbation_rp.tolist()}",
        flush=True,
    )


def _reset_batch() -> SimpleNamespace:
    return SimpleNamespace(
        segment_ids=torch.tensor([7, 9], dtype=torch.long),
        clean_state=SimpleNamespace(
            root_pos=torch.zeros(2, 3),
            root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            root_lin_vel=torch.ones(2, 3) * 0.1,
            root_ang_vel=torch.ones(2, 3) * 0.2,
            dof_pos=torch.zeros(2, 29),
            dof_vel=torch.ones(2, 29) * 0.01,
        ),
        reference_window=torch.zeros(2, 4, 6),
        phase=torch.tensor([0.1, 0.2]),
        specs=(),
    )


def _index_only_reset_batch() -> SimpleNamespace:
    batch = _reset_batch()
    batch.specs = (
        SimpleNamespace(
            motion_id="KIT/359/motion_a.npz",
            start_frame=12,
            perturbation_family="index_only",
        ),
        SimpleNamespace(
            motion_id="KIT/359/motion_b.npz",
            start_frame=24,
            perturbation_family="index_only",
        ),
    )
    batch.perturbation_family = ("index_only", "index_only")
    return batch


def _large_index_only_reset_batch(count: int = 12000) -> SimpleNamespace:
    specs = tuple(
        SimpleNamespace(
            motion_id=f"Corpus/Subj{idx % 4}/motion_{idx % 4}.npz",
            start_frame=idx,
            horizon_k=4 + (idx % 2),
            perturbation_family="index_only",
        )
        for idx in range(count)
    )
    return SimpleNamespace(
        segment_ids=torch.arange(count, dtype=torch.long),
        specs=specs,
        perturbation_family=("index_only",) * count,
    )


def test_large_index_reset_probe_uses_summary_not_full_lists() -> None:
    env = _FakeIndexResetLiveEnv()
    runner = SimpleNamespace(
        env=env,
        alg=SimpleNamespace(frontres_segment_verbose_probe=False),
    )
    batch = _large_index_only_reset_batch()

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = live_probe._apply_index_only_segment_reset(runner, batch)
    output = buffer.getvalue()

    has_count = "count=12000" in output
    has_motion_summary = "unique_motion_count=4" in output
    has_start_range = "start_min=0" in output and "start_max=11999" in output
    has_horizon_range = "horizon_min=4" in output and "horizon_max=5" in output
    has_segment_list = "segment_ids=[0, 1, 2" in output
    has_motion_list = "motion_ids=['Corpus" in output
    has_start_list = "start_frames=[0, 1, 2" in output
    has_horizon_list = "horizon_k=[4, 5" in output
    print(
        "[probe step3] reset_log_summary: "
        f"contains_count={has_count} "
        f"contains_motion_summary={has_motion_summary} "
        f"contains_start_range={has_start_range} "
        f"contains_horizon_range={has_horizon_range} "
        f"contains_segment_ids_list={has_segment_list} "
        f"contains_motion_ids_list={has_motion_list} "
        f"contains_start_frames_list={has_start_list} "
        f"contains_horizon_k_list={has_horizon_list}",
        flush=True,
    )

    assert result is not None
    assert env.events == ["index_reset"]
    assert runner._frontres_segment_live_current_reset_request.segment_ids.numel() == 12000
    assert result.success_mask.numel() == 12000
    assert has_count
    assert has_motion_summary
    assert has_start_range
    assert has_horizon_range
    assert not has_segment_list
    assert not has_motion_list
    assert not has_start_list
    assert not has_horizon_list


def test_live_probe_applies_current_segment_batch_reset_before_rollout() -> None:
    env = _FakeLiveEnv()
    runner = SimpleNamespace(
        env=env,
        device=torch.device("cpu"),
        policy_obs_type=None,
        privileged_obs_type=None,
        teacher_obs_type=None,
        ref_vel_estimator_obs_type=None,
        current_learning_iteration=0,
        _frontres_segment_replay_boundary=SimpleNamespace(
            live_probe_only=True,
            live_storage_write_only=False,
            live_single_update_only=False,
            live_update_loop_only=False,
            live_train_enabled=False,
            segment_k=1,
            reset_mode="direct",
        ),
        _frontres_segment_live_current_batch=_reset_batch(),
        alg=SimpleNamespace(
            frontres_training_objective="segment_replay_hrl",
            frontres_segment_k=1,
            frontres_segment_reset_mode="direct",
            frontres_segment_preroll_steps=0,
            transition=SimpleNamespace(),
        ),
        eval_mode=lambda: None,
        _apply_obs_normalizer=lambda obs: obs,
        privileged_obs_normalizer=lambda obs: obs,
        teacher_obs_normalizer=lambda obs: obs,
    )

    summary = run_frontres_segment_live_probe(runner, init_at_random_ep_len=False)
    request = runner._frontres_segment_live_current_reset_request
    result = runner._frontres_segment_live_current_reset_result

    _probe_tensor("batch.segment_ids", runner._frontres_segment_live_current_batch.segment_ids, "sampled ids before reset request")
    _probe_tensor("request.segment_ids", request.segment_ids, "same ids inside reset request")
    _probe_tensor("request.valid_mask", request.valid_mask, "reset request validity before env hook")
    _probe_tensor("result.success_mask", result.success_mask, "env reset result after adapter validation")
    print(
        "[probe step11] live_reset_summary: "
        f"events={env.events} "
        f"segment_reset={summary['segment_reset']} "
        f"success_frac={summary['segment_reset_success_frac']} "
        f"direct_frac={summary['segment_reset_direct_frac']} "
        f"reward_mean={summary['reward_mean']}",
        flush=True,
    )

    assert env.events == ["reset", "get_obs", "step"]
    assert request.segment_ids.tolist() == [7, 9]
    assert tuple(request.mode) == ("direct", "direct")
    assert request.valid_mask.tolist() == [True, True]
    assert result.success_mask.tolist() == [True, True]
    assert summary["segment_reset"] is True
    assert summary["segment_reset_success_frac"] == 1.0
    assert summary["segment_reset_direct_frac"] == 1.0
    assert summary["done_frac"] == 0.0


def test_live_probe_skips_dynamic_reset_for_index_only_segments() -> None:
    env = _FakeLiveEnv()
    runner = SimpleNamespace(
        env=env,
        device=torch.device("cpu"),
        policy_obs_type=None,
        privileged_obs_type=None,
        teacher_obs_type=None,
        ref_vel_estimator_obs_type=None,
        current_learning_iteration=0,
        _frontres_segment_replay_boundary=SimpleNamespace(
            live_probe_only=True,
            live_storage_write_only=False,
            live_single_update_only=False,
            live_update_loop_only=False,
            live_train_enabled=False,
            segment_k=1,
            reset_mode="direct",
        ),
        _frontres_segment_live_current_batch=_index_only_reset_batch(),
        alg=SimpleNamespace(
            frontres_training_objective="segment_replay_hrl",
            frontres_segment_k=1,
            frontres_segment_reset_mode="direct",
            frontres_segment_preroll_steps=0,
            transition=SimpleNamespace(),
        ),
        eval_mode=lambda: None,
        _apply_obs_normalizer=lambda obs: obs,
        privileged_obs_normalizer=lambda obs: obs,
        teacher_obs_normalizer=lambda obs: obs,
    )

    summary = run_frontres_segment_live_probe(runner, init_at_random_ep_len=False)
    batch = runner._frontres_segment_live_current_batch

    _probe_tensor("index_only.segment_ids", batch.segment_ids, "sampled ids from Stage 1 index-only candidate pool")
    print(
        "[probe step4] index_only_reset_skip: "
        f"events={env.events} "
        f"families={batch.perturbation_family} "
        f"motion_ids={[spec.motion_id for spec in batch.specs]} "
        f"start_frames={[spec.start_frame for spec in batch.specs]} "
        f"skip_reason={runner._frontres_segment_live_current_reset_skip_reason} "
        f"segment_reset={summary['segment_reset']} "
        f"reward_mean={summary['reward_mean']}",
        flush=True,
    )

    assert env.events == ["get_obs", "step"]
    assert runner._frontres_segment_live_current_reset_request is None
    assert runner._frontres_segment_live_current_reset_result is None
    assert runner._frontres_segment_live_current_reset_skip_reason == "index_only_segment_index"
    assert summary["segment_reset"] is False
    assert summary["reward_mean"] == 0.75


def test_live_probe_applies_index_reset_for_index_only_segments_when_env_supports_it() -> None:
    env = _FakeIndexResetLiveEnv()
    runner = SimpleNamespace(
        env=env,
        device=torch.device("cpu"),
        policy_obs_type=None,
        privileged_obs_type=None,
        teacher_obs_type=None,
        ref_vel_estimator_obs_type=None,
        current_learning_iteration=0,
        _frontres_segment_replay_boundary=SimpleNamespace(
            live_probe_only=True,
            live_storage_write_only=False,
            live_single_update_only=False,
            live_update_loop_only=False,
            live_train_enabled=False,
            segment_k=1,
            reset_mode="direct",
        ),
        _frontres_segment_live_current_batch=_index_only_reset_batch(),
        alg=SimpleNamespace(
            frontres_training_objective="segment_replay_hrl",
            frontres_segment_k=1,
            frontres_segment_reset_mode="direct",
            frontres_segment_preroll_steps=0,
            transition=SimpleNamespace(),
        ),
        eval_mode=lambda: None,
        _apply_obs_normalizer=lambda obs: obs,
        privileged_obs_normalizer=lambda obs: obs,
        teacher_obs_normalizer=lambda obs: obs,
    )

    summary = run_frontres_segment_live_probe(runner, init_at_random_ep_len=False)
    request = runner._frontres_segment_live_current_reset_request
    result = runner._frontres_segment_live_current_reset_result

    _probe_tensor("index_request.segment_ids", request.segment_ids, "ids passed from index-only batch into env index reset")
    _probe_tensor("index_request.start_frames", request.start_frames, "motion frame chosen by Stage 1 segment index")
    _probe_tensor("index_request.horizon_k", request.horizon_k, "segment rollout horizon for live probing")
    _probe_tensor("index_result.success_mask", result.success_mask, "env index reset success used by storage validity")
    print(
        "[probe step5] index_only_reset_apply: "
        f"events={env.events} "
        f"motion_ids={list(request.motion_ids)} "
        f"start_frames={request.start_frames.tolist()} "
        f"horizon_k={request.horizon_k.tolist()} "
        f"segment_reset={summary['segment_reset']} "
        f"success_frac={summary['segment_reset_success_frac']} "
        f"reward_mean={summary['reward_mean']}",
        flush=True,
    )

    assert env.events == ["index_reset", "get_obs", "step"]
    assert request.segment_ids.tolist() == [7, 9]
    assert list(request.motion_ids) == ["KIT/359/motion_a.npz", "KIT/359/motion_b.npz"]
    assert request.start_frames.tolist() == [12, 24]
    assert request.horizon_k.tolist() == [1, 1]
    assert result.success_mask.tolist() == [True, True]
    assert runner._frontres_segment_live_current_reset_skip_reason == ""
    assert summary["segment_reset"] is True
    assert summary["segment_reset_success_frac"] == 1.0
    assert summary["reward_mean"] == 0.75


def test_index_reset_request_carries_stage3_dynamic_perturbation() -> None:
    env = _FakeIndexResetLiveEnv()
    batch = _index_only_reset_batch()
    batch.stage3_index_perturbation_family = ("planar+yaw", "yaw")
    batch.stage3_index_perturbation_strength = torch.tensor([2.0, 1.5])
    runner = SimpleNamespace(
        env=env,
        alg=SimpleNamespace(frontres_segment_verbose_probe=True),
        _frontres_segment_replay_boundary=SimpleNamespace(reset_mode="direct"),
        _frontres_segment_live_detail_log_enabled=True,
    )

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        result = live_probe._apply_index_only_segment_reset(runner, batch)
    output = stream.getvalue()
    request = runner._frontres_segment_live_current_reset_request

    _probe_tensor("index_request.perturbation_strength", request.perturbation_strength, "dynamic Stage 3 strength attached before env hook")
    print(
        "[probe step3] index_request_dynamic_perturbation: "
        f"family={request.perturbation_family} "
        f"strength={request.perturbation_strength.tolist()} "
        f"nonzero_logged={'reset.request_strength_nonzero_frac: 100.0%' in output} "
        f"success={result.success_mask.tolist()}",
        flush=True,
    )

    assert env.events == ["index_reset"]
    assert request.perturbation_family == ("planar+yaw", "yaw")
    torch.testing.assert_close(request.perturbation_strength, torch.tensor([2.0, 1.5]))
    assert "reset.request_strength_nonzero_frac: 100.0%" in output
    assert result.success_mask.tolist() == [True, True]


def test_live_probe_detail_gate_suppresses_reset_and_summary_logs() -> None:
    env = _FakeIndexResetLiveEnv()
    runner = SimpleNamespace(
        env=env,
        device=torch.device("cpu"),
        alg=SimpleNamespace(
            frontres_training_objective="segment_replay_hrl",
            frontres_segment_verbose_probe=False,
        ),
        _frontres_segment_replay_boundary=SimpleNamespace(reset_mode="direct"),
        _frontres_segment_live_detail_log_enabled=False,
    )
    batch = _index_only_reset_batch()

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        result = live_probe._apply_index_only_segment_reset(runner, batch)
        live_probe._print_live_probe_summary(
            runner,
            _capture(),
            {
                "segment_reset": True,
                "segment_reset_success_frac": 1.0,
                "segment_reset_direct_frac": 0.0,
                "segment_reset_preroll_frac": 0.0,
                "segment_reset_velocity_mismatch_mean": 0.0,
                "segment_reference_window_applied_frac": 0.0,
                "valid_mask_frac": 1.0,
                "reward_mean": 0.5,
                "done_frac": 0.0,
                "storage_write": False,
                "storage_size": 0,
                "storage_valid_frac": 0.0,
                "storage_reward_mean": 0.0,
                "single_update": False,
                "ppo_update": False,
                "ppo_valid_count": 0,
                "ppo_total_loss": 0.0,
                "ppo_actor_loss": 0.0,
                "ppo_value_loss": 0.0,
                "ppo_approx_kl": 0.0,
                "ppo_clip_frac": 0.0,
            },
        )
    output = stream.getvalue()
    reset_count = output.count("[FrontRES Segment Reset]")
    live_probe_count = output.count("[FrontRES Segment Live Probe]")
    print(
        "[probe step6] live_probe_detail_gate: "
        f"reset_count={reset_count} "
        f"live_probe_count={live_probe_count} "
        f"success_count={int(result.success_mask.sum().item())}",
        flush=True,
    )

    assert result is not None
    assert int(result.success_mask.sum().item()) == 2
    assert reset_count == 0
    assert live_probe_count == 0


def test_live_probe_summary_uses_readable_metric_blocks() -> None:
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            frontres_training_objective="segment_replay_hrl",
            frontres_segment_verbose_probe=True,
        ),
        _frontres_segment_replay_boundary=SimpleNamespace(reset_mode="direct"),
        _frontres_segment_live_detail_log_enabled=True,
    )
    summary = {
        "segment_reset": True,
        "segment_reset_skip_reason": "",
        "segment_reset_success_frac": 1.0,
        "segment_reset_direct_frac": 0.5,
        "segment_reset_preroll_frac": 0.5,
        "segment_reset_velocity_mismatch_mean": 0.0,
        "segment_reference_window_applied_frac": 0.0,
        "valid_mask_frac": 1.0,
        "reward_mean": 0.5,
        "env_reward_mean": 0.5,
        "train_reward_mean": 0.4,
        "score_gain_mean": 0.4,
        "done_frac": 0.0,
        "storage_write": True,
        "storage_size": 2,
        "storage_valid_frac": 1.0,
        "storage_reward_mean": 0.5,
        "single_update": True,
        "ppo_update": True,
        "ppo_valid_count": 2,
        "ppo_total_loss": 1.0,
        "ppo_actor_loss": 0.5,
        "ppo_value_loss": 0.25,
        "ppo_approx_kl": -0.01,
        "ppo_clip_frac": 0.2,
        "ppo_old_log_prob_mean": -2.0,
        "ppo_new_log_prob_mean": -1.0,
        "ppo_raw_log_ratio_mean": 1.0,
        "ppo_raw_log_ratio_min": -0.5,
        "ppo_raw_log_ratio_max": 2.0,
        "ppo_ratio_mean": 3.0,
        "ppo_ratio_max": 7.0,
        "ppo_pre_update_raw_log_ratio_mean": 1.0,
        "ppo_pre_update_raw_log_ratio_min": -0.5,
        "ppo_pre_update_raw_log_ratio_max": 2.0,
        "ppo_pre_update_clamped_ratio_mean": 3.0,
        "ppo_pre_update_clamped_ratio_max": 7.0,
        "ppo_post_update_raw_log_ratio_mean": 1.5,
        "ppo_post_update_raw_log_ratio_min": -0.25,
        "ppo_post_update_raw_log_ratio_max": 2.5,
        "ppo_post_update_clamped_ratio_mean": 4.0,
        "ppo_post_update_clamped_ratio_max": 8.0,
        "ppo_advantage_mean": 0.1,
        "ppo_advantage_min": -0.2,
        "ppo_advantage_max": 0.4,
        "evidence_row_count": 2,
        "score_source": "b1_paired_env_rewards",
        "score_noisy_per_sample": [0.2, 0.3],
        "score_repaired_per_sample": [0.7, 0.6],
        "evidence_valid_mask_per_sample": [True, False],
    }
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        live_probe._print_live_probe_summary(runner, _capture(), summary)
    output = stream.getvalue()
    print(
        "[probe step6] readable_metric_blocks: "
        f"live_probe={'[FrontRES Segment Live Probe]' in output} "
        f"ppo_probe={'[FrontRES Segment PPO Probe]' in output} "
        f"route={'  route.objective:' in output} "
        f"reset={'  reset.enabled:' in output} "
        f"pre_log_ratio={'  pre_log_ratio.mean:' in output}",
        flush=True,
    )
    assert "[FrontRES Segment Live Probe]" in output
    assert "[FrontRES Segment PPO Probe]" in output
    for label in (
        "  route.objective:",
        "  reset.enabled:",
        "  rollout.obs:",
        "  rollout.env_reward:",
        "  score.source:",
        "  score.gain:",
        "  storage.write:",
        "  storage.train_reward:",
        "  ppo.valid:",
        "  pre_log_ratio.mean:",
        "  post_log_ratio.mean:",
    ):
        assert label in output
    assert "reset.reason: applied" in output
    assert "rollout.policy_dim: 6" in output
    assert "rollout.env_reward: 0.500000" in output
    assert "rollout.segment_delta_se_6d: True" in output
    assert "score.source: b1_paired_env_rewards" in output
    assert "score.gain: 0.400000" in output
    assert "score.rows: 2" in output
    assert "storage.train_reward: 0.400000" in output
    assert "storage.all_reward: 0.500000" in output
    assert output.startswith("\n" + live_probe._LOG_SEPARATOR + "\n")
    assert f"\n{live_probe._LOG_SEPARATOR}\n\n[FrontRES Segment PPO Probe]" in output
    assert not output.rstrip().endswith(live_probe._LOG_SEPARATOR)


def test_live_probe_summary_requires_separate_pre_and_post_ratio_blocks() -> None:
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            frontres_training_objective="segment_replay_hrl",
            frontres_segment_verbose_probe=True,
        ),
        _frontres_segment_replay_boundary=SimpleNamespace(reset_mode="direct"),
        _frontres_segment_live_detail_log_enabled=True,
    )
    summary = {
        "segment_reset": True,
        "segment_reset_skip_reason": "",
        "segment_reset_success_frac": 1.0,
        "segment_reset_direct_frac": 1.0,
        "segment_reset_preroll_frac": 0.0,
        "segment_reset_velocity_mismatch_mean": 0.0,
        "segment_reference_window_applied_frac": 0.0,
        "valid_mask_frac": 1.0,
        "reward_mean": 0.5,
        "env_reward_mean": 0.5,
        "train_reward_mean": 0.4,
        "score_gain_mean": 0.4,
        "done_frac": 0.0,
        "storage_write": True,
        "storage_size": 2,
        "storage_valid_frac": 1.0,
        "storage_reward_mean": 0.5,
        "single_update": True,
        "ppo_update": True,
        "ppo_valid_count": 2,
        "ppo_total_loss": 1.0,
        "ppo_actor_loss": 0.5,
        "ppo_value_loss": 0.25,
        "ppo_approx_kl": 0.01,
        "ppo_clip_frac": 0.5,
        "ppo_old_log_prob_mean": -2.0,
        "ppo_new_log_prob_mean": -1.0,
        "ppo_pre_update_raw_log_ratio_mean": 1.0,
        "ppo_pre_update_raw_log_ratio_min": -0.5,
        "ppo_pre_update_raw_log_ratio_max": 2.0,
        "ppo_pre_update_clamped_ratio_mean": 3.0,
        "ppo_pre_update_clamped_ratio_max": 7.0,
        "ppo_post_update_raw_log_ratio_mean": -5.0,
        "ppo_post_update_raw_log_ratio_min": -6.0,
        "ppo_post_update_raw_log_ratio_max": -4.0,
        "ppo_post_update_clamped_ratio_mean": 0.01,
        "ppo_post_update_clamped_ratio_max": 0.02,
        "ppo_pre_distribution_kl_mean": 0.001,
        "ppo_pre_logprob_approx_kl": 0.0,
        "ppo_post_update_distribution_kl_mean": 0.009,
        "ppo_post_update_logprob_approx_kl": 5.0,
        "ppo_distribution_kl_available": True,
        "ppo_trust_region_accepted": 1,
        "ppo_trust_region_rejected_count": 0,
        "ppo_adaptive_lr_before": 1e-4,
        "ppo_adaptive_lr_after": 1e-4,
        "ppo_adaptive_lr_desired_kl": 0.01,
        "ppo_trust_region_schedule": "adaptive",
        "ppo_trust_region_rollback_enabled": 1,
        "ppo_trust_region_max_retries": 2,
        "ppo_advantage_mean": 0.1,
        "ppo_advantage_min": -0.2,
        "ppo_advantage_max": 0.4,
        "evidence_row_count": 2,
        "score_source": "b1_paired_env_rewards",
        "score_noisy_per_sample": [0.2, 0.3],
        "score_repaired_per_sample": [0.7, 0.6],
        "evidence_valid_mask_per_sample": [True, True],
    }
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        live_probe._print_live_probe_summary(runner, _capture(), summary)
    output = stream.getvalue()
    print(
        "[probe stepA] readable_ratio_blocks: "
        f"pre_log_ratio={'  pre_log_ratio.mean:' in output} "
        f"pre_ratio={'  pre_ratio.clamped_mean:' in output} "
        f"post_log_ratio={'  post_log_ratio.mean:' in output} "
        f"post_ratio={'  post_ratio.clamped_mean:' in output} "
        f"legacy_reported={'  ratio.reported_mean:' in output}",
        flush=True,
    )

    assert "  pre_log_ratio.mean:" in output
    assert "  pre_ratio.clamped_mean:" in output
    assert "  post_log_ratio.mean:" in output
    assert "  post_ratio.clamped_mean:" in output
    assert "  ratio.reported_mean:" not in output


def test_live_probe_summary_reports_raw_policy_and_segment_delta_dims() -> None:
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            frontres_training_objective="segment_replay_hrl",
            frontres_segment_verbose_probe=True,
        ),
        _frontres_segment_replay_boundary=SimpleNamespace(reset_mode="auto"),
        _frontres_segment_live_detail_log_enabled=True,
    )
    capture = _capture()
    capture.action_shape = (2, 12)
    summary = {
        "segment_reset": False,
        "segment_reset_skip_reason": "no_current_segment_batch",
        "segment_reset_success_frac": 0.0,
        "segment_reset_direct_frac": 0.0,
        "segment_reset_preroll_frac": 0.0,
        "segment_reset_velocity_mismatch_mean": 0.0,
        "segment_reference_window_applied_frac": 0.0,
        "valid_mask_frac": 1.0,
        "reward_mean": 0.5,
        "done_frac": 0.0,
        "storage_write": True,
        "storage_size": 2,
        "storage_valid_frac": 1.0,
        "storage_reward_mean": 0.5,
        "single_update": True,
        "ppo_update": False,
        "ppo_valid_count": 0,
        "ppo_total_loss": 0.0,
        "ppo_actor_loss": 0.0,
        "ppo_value_loss": 0.0,
        "ppo_approx_kl": 0.0,
        "ppo_clip_frac": 0.0,
    }
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        live_probe._print_live_probe_summary(runner, capture, summary)
    output = stream.getvalue()
    print(
        "[probe bug3] action_dim_summary: "
        f"policy_dim_12={'rollout.policy_dim: 12' in output} "
        f"segment_6d={'rollout.segment_delta_se_6d: True' in output} "
        f"reset_reason={'reset.reason: no_current_segment_batch' in output}",
        flush=True,
    )
    assert "rollout.policy_dim: 12" in output
    assert "rollout.segment_action: (2, 6)" in output
    assert "rollout.segment_delta_se_6d: True" in output
    assert "reset.reason: no_current_segment_batch" in output


def test_live_probe_summary_extracts_b1_noisy_repaired_scores() -> None:
    capture = FrontRESSegmentLiveRolloutCapture(
        rollout_k=2,
        reward_mean=0.0,
        done_frac=0.0,
        last_obs_shape=(8, 4),
        action_shape=(8, 6),
        env_action_shape=(8, 12),
        transition_obs=torch.zeros(8, 4),
        transition_privileged_obs=torch.zeros(8, 3),
        transition_actions=torch.zeros(8, 6),
        transition_log_probs=torch.zeros(8),
        transition_values=torch.zeros(8),
        transition_means=torch.zeros(8, 6),
        transition_sigmas=torch.ones(8, 6),
        reward_accum=torch.tensor([1.4, 1.8, 0.0, 0.0, 0.4, 1.0, 2.0, 2.0]),
        reward_steps=torch.tensor([[1.4, 1.8, 0.0, 0.0, 0.4, 1.0, 2.0, 2.0]]),
        done_steps=torch.zeros(1, 8, dtype=torch.bool),
        done_any=torch.tensor([False, True, False, False, False, False, False, False]),
        n_train=2,
        n_candidate=2,
        n_base=2,
        n_clean=2,
    )
    summary = live_probe._initial_live_probe_summary(capture, storage_write=True, single_update=False)
    print(
        "[probe step2] b1_score_summary: "
        f"rows={summary['evidence_row_count']} "
        f"repaired={summary['score_repaired_per_sample']} "
        f"noisy={summary['score_noisy_per_sample']} "
        f"clean={summary['score_clean_per_sample']} "
        f"valid={summary['evidence_valid_mask_per_sample']}",
        flush=True,
    )
    assert summary["evidence_row_count"] == 2
    torch.testing.assert_close(torch.tensor(summary["score_repaired_per_sample"]), torch.tensor([0.7, 0.9]))
    torch.testing.assert_close(torch.tensor(summary["score_noisy_per_sample"]), torch.tensor([0.2, 0.5]))
    torch.testing.assert_close(torch.tensor(summary["gain_over_noisy_per_sample"]), torch.tensor([0.5, 0.4]))
    torch.testing.assert_close(torch.tensor(summary["score_clean_per_sample"]), torch.tensor([1.0, 1.0]))
    assert summary["evidence_valid_mask_per_sample"] == [True, False]


def test_live_probe_summary_preserves_b1_gain_when_env_rewards_are_negative() -> None:
    capture = FrontRESSegmentLiveRolloutCapture(
        rollout_k=1,
        reward_mean=0.0,
        done_frac=0.0,
        last_obs_shape=(8, 4),
        action_shape=(8, 6),
        env_action_shape=(8, 12),
        transition_obs=torch.zeros(8, 4),
        transition_privileged_obs=torch.zeros(8, 3),
        transition_actions=torch.zeros(8, 6),
        transition_log_probs=torch.zeros(8),
        transition_values=torch.zeros(8),
        transition_means=torch.zeros(8, 6),
        transition_sigmas=torch.ones(8, 6),
        reward_accum=torch.tensor([-0.08, -0.02, 0.0, 0.0, -0.10, -0.07, 0.0, 0.0]),
        reward_steps=torch.tensor([[-0.08, -0.02, 0.0, 0.0, -0.10, -0.07, 0.0, 0.0]]),
        done_steps=torch.zeros(1, 8, dtype=torch.bool),
        done_any=torch.tensor([False, False, False, False, False, False, False, False]),
        n_train=2,
        n_candidate=2,
        n_base=2,
        n_clean=2,
    )
    summary = live_probe._initial_live_probe_summary(capture, storage_write=True, single_update=False)
    print(
        "[probe step1] b1_negative_reward_gain_summary: "
        f"repaired={summary['score_repaired_per_sample']} "
        f"noisy={summary['score_noisy_per_sample']} "
        f"gain={summary['gain_over_noisy_per_sample']} "
        f"gain_mean={summary['score_gain_mean']}",
        flush=True,
    )
    torch.testing.assert_close(torch.tensor(summary["score_repaired_per_sample"]), torch.tensor([-0.08, -0.02]))
    torch.testing.assert_close(torch.tensor(summary["score_noisy_per_sample"]), torch.tensor([-0.10, -0.07]))
    torch.testing.assert_close(torch.tensor(summary["gain_over_noisy_per_sample"]), torch.tensor([0.02, 0.05]))
    assert abs(float(summary["score_gain_mean"]) - 0.035) < 1e-6


if __name__ == "__main__":
    test_build_live_segment_storage_preserves_first_step_tuple_trace()
    test_build_live_segment_storage_uses_b1_paired_gain_when_available()
    test_build_live_segment_storage_uses_discounted_reward_trace_for_ppo_returns()
    test_build_live_segment_storage_masks_non_actor_rows()
    test_build_live_segment_storage_rejects_non_6d_actions()
    test_live_probe_selects_6d_delta_se_from_12d_rollout_action()
    test_live_probe_trace_prints_once_without_verbose()
    test_live_probe_trace_reports_native_6d_policy_surface()
    test_ppo_eval_trace_prints_once_without_verbose()
    test_build_live_segment_storage_masks_failed_reset_samples()
    test_large_index_reset_probe_uses_summary_not_full_lists()
    test_live_rollout_capture_zero_segment_action_reaches_env_step()
    test_live_rollout_capture_snapshots_signed_rp_perturbation()
    test_live_probe_applies_current_segment_batch_reset_before_rollout()
    test_live_probe_skips_dynamic_reset_for_index_only_segments()
    test_live_probe_applies_index_reset_for_index_only_segments_when_env_supports_it()
    test_index_reset_request_carries_stage3_dynamic_perturbation()
    test_live_probe_detail_gate_suppresses_reset_and_summary_logs()
    test_live_probe_summary_uses_readable_metric_blocks()
    test_live_probe_summary_requires_separate_pre_and_post_ratio_blocks()
    test_live_probe_summary_reports_raw_policy_and_segment_delta_dims()
    test_live_probe_summary_extracts_b1_noisy_repaired_scores()
    test_live_probe_summary_preserves_b1_gain_when_env_rewards_are_negative()
    print("frontres_segment_live_probe_contract: ok")
