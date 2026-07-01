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
        actions = torch.zeros(batch, 6)
        runner.alg.transition.observations = kwargs["obs"].detach().clone()
        runner.alg.transition.privileged_observations = kwargs["privileged_obs"].detach().clone()
        runner.alg.transition.actions_log_prob = torch.zeros(batch)
        runner.alg.transition.values = torch.zeros(batch)
        runner.alg.transition.action_mean = actions.detach().clone()
        runner.alg.transition.action_sigma = torch.ones_like(actions)
        return SimpleNamespace(actions=actions, env_actions=torch.zeros(batch, 12))

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
build_live_segment_storage = live_probe.build_live_segment_storage
run_frontres_segment_live_probe = live_probe.run_frontres_segment_live_probe


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
        done_any=torch.tensor([False, True]),
        n_train=2,
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
    _probe_tensor("batch.returns", batch.returns, "PPO return defaults to stored reward")
    _probe_tensor("capture.done_any", capture.done_any, "whether any env done occurred during K-step rollout")
    _probe_tensor("expected_valid", expected_valid, "valid segment mask derived as not done_any")
    _probe_tensor("storage.valid_mask", storage.valid_mask[: storage.step], "stored valid mask")
    _probe_tensor("batch.valid_mask", batch.valid_mask, "full_batch valid mask consumed by PPO")
    _probe_tensor("batch.segment_ids", batch.segment_ids, "live storage assigns one segment id per env row")

    assert storage.step == 2
    assert batch.actions.shape == (2, 6)
    torch.testing.assert_close(batch.actions, capture.transition_actions)
    torch.testing.assert_close(storage.rewards[: storage.step], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(batch.returns, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(batch.advantages, torch.tensor([0.5, 2.5]))
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
        f"log_ratio={'  log_ratio.mean:' in output}",
        flush=True,
    )
    assert "[FrontRES Segment Live Probe]" in output
    assert "[FrontRES Segment PPO Probe]" in output
    for label in (
        "  route.objective:",
        "  reset.enabled:",
        "  rollout.obs:",
        "  score.source:",
        "  storage.write:",
        "  ppo.valid:",
        "  log_ratio.mean:",
    ):
        assert label in output
    assert "reset.reason: applied" in output
    assert "rollout.policy_dim: 6" in output
    assert "rollout.segment_delta_se_6d: True" in output
    assert "score.source: b1_paired_env_rewards" in output
    assert "score.rows: 2" in output
    assert output.startswith("\n" + live_probe._LOG_SEPARATOR + "\n")
    assert f"\n{live_probe._LOG_SEPARATOR}\n\n[FrontRES Segment PPO Probe]" in output
    assert not output.rstrip().endswith(live_probe._LOG_SEPARATOR)


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
    torch.testing.assert_close(torch.tensor(summary["score_clean_per_sample"]), torch.tensor([1.0, 1.0]))
    assert summary["evidence_valid_mask_per_sample"] == [True, False]


if __name__ == "__main__":
    test_build_live_segment_storage_preserves_first_step_tuple_trace()
    test_build_live_segment_storage_masks_non_actor_rows()
    test_build_live_segment_storage_rejects_non_6d_actions()
    test_live_probe_selects_6d_delta_se_from_12d_rollout_action()
    test_live_probe_trace_prints_once_without_verbose()
    test_ppo_eval_trace_prints_once_without_verbose()
    test_build_live_segment_storage_masks_failed_reset_samples()
    test_large_index_reset_probe_uses_summary_not_full_lists()
    test_live_probe_applies_current_segment_batch_reset_before_rollout()
    test_live_probe_skips_dynamic_reset_for_index_only_segments()
    test_live_probe_applies_index_reset_for_index_only_segments_when_env_supports_it()
    test_live_probe_detail_gate_suppresses_reset_and_summary_logs()
    test_live_probe_summary_uses_readable_metric_blocks()
    test_live_probe_summary_reports_raw_policy_and_segment_delta_dims()
    test_live_probe_summary_extracts_b1_noisy_repaired_scores()
    print("frontres_segment_live_probe_contract: ok")
