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
    training_schedule.resolve_frontres_mode_state = lambda *_args, **_kwargs: None
    sys.modules[training_schedule.__name__] = training_schedule
    frontres_pkg.training_schedule = training_schedule

    training_setup = types.ModuleType("rsl_rl.runners.frontres_training_setup")
    training_setup.configure_frontres_pair_layout = lambda *_args, **_kwargs: None
    sys.modules[training_setup.__name__] = training_setup
    runners_pkg.frontres_training_setup = training_setup

    storage_module = _load(
        "rsl_rl.frontres.frontres_segment_storage",
        ROOT / "rsl_rl" / "frontres" / "frontres_segment_storage.py",
    )
    frontres_pkg.frontres_segment_storage = storage_module

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
    return live_probe_module


live_probe = _install_live_probe_import_stubs()
FrontRESSegmentLiveRolloutCapture = live_probe.FrontRESSegmentLiveRolloutCapture
build_live_segment_storage = live_probe.build_live_segment_storage


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


def test_build_live_segment_storage_rejects_non_6d_actions() -> None:
    runner = SimpleNamespace(device=torch.device("cpu"))
    capture = _capture(actions=torch.zeros(2, 5))

    try:
        build_live_segment_storage(runner, capture)
    except ValueError as exc:
        assert "requires 6D actions" in str(exc)
    else:
        raise AssertionError("non-6D live probe actions must be rejected before storage write")


if __name__ == "__main__":
    test_build_live_segment_storage_preserves_first_step_tuple_trace()
    test_build_live_segment_storage_rejects_non_6d_actions()
    print("frontres_segment_live_probe_contract: ok")
