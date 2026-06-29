#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
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

    ppo_module = types.ModuleType("rsl_rl.algorithms.frontres_segment_ppo")
    ppo_module.FrontRESSegmentPPOBatch = object
    ppo_module.FrontRESSegmentPPOConfig = object
    ppo_module.compute_frontres_segment_ppo_loss = lambda *_args, **_kwargs: None
    sys.modules[ppo_module.__name__] = ppo_module
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

    modules_pkg = types.ModuleType("rsl_rl.modules")
    modules_pkg.FrontRESActorCritic = object
    sys.modules[modules_pkg.__name__] = modules_pkg
    rsl_rl_pkg.modules = modules_pkg

    rollout_step = types.ModuleType("rsl_rl.runners.frontres_rollout_step")

    def _prepare_frontres_rollout_step(runner, **kwargs):
        batch = int(kwargs["obs"].shape[0])
        actions = torch.tensor(
            [[0.10, -0.20, 0.00, 0.30, 0.00, -0.10]],
            dtype=torch.float32,
        ).expand(batch, -1).clone()
        runner.alg.transition.observations = kwargs["obs"].detach().clone()
        runner.alg.transition.privileged_observations = kwargs["privileged_obs"].detach().clone()
        runner.alg.transition.actions_log_prob = torch.full((batch,), -0.25)
        runner.alg.transition.values = torch.full((batch,), 0.5)
        runner.alg.transition.action_mean = actions.detach().clone()
        runner.alg.transition.action_sigma = torch.ones_like(actions) * 0.2
        return SimpleNamespace(actions=actions, env_actions=torch.zeros(batch, 12))

    rollout_step.prepare_frontres_rollout_step = _prepare_frontres_rollout_step
    sys.modules[rollout_step.__name__] = rollout_step
    runners_pkg.frontres_rollout_step = rollout_step


_install_import_stubs()


dataset_module = _load(
    "frontres_segment_dataset_for_step17_closed_loop",
    ROOT / "rsl_rl" / "frontres" / "frontres_segment_dataset.py",
)
sampler_module = _load(
    "frontres_segment_sampler_for_step17_closed_loop",
    ROOT / "rsl_rl" / "frontres" / "frontres_segment_sampler.py",
)
live_probe_module = _load(
    "rsl_rl.runners.frontres_segment_live_probe",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_live_probe.py",
)
live_sampler_module = _load(
    "frontres_segment_live_sampler_for_step17_closed_loop",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_live_sampler.py",
)

FrontRESSegmentDataset = dataset_module.FrontRESSegmentDataset
FrontRESSegmentSampler = sampler_module.FrontRESSegmentSampler
run_frontres_segment_sampler_step = live_sampler_module.run_frontres_segment_sampler_step
run_frontres_segment_live_probe = live_probe_module.run_frontres_segment_live_probe


def _probe_tensor(name: str, tensor: torch.Tensor, semantic: str) -> None:
    data = tensor.detach().cpu()
    numeric = data.float() if data.dtype == torch.bool else data.float()
    print(
        f"[probe step17] {name}: shape={tuple(data.shape)} dtype={data.dtype} "
        f"device={tensor.device} min={numeric.min().item():.6f} "
        f"max={numeric.max().item():.6f} mean={numeric.mean().item():.6f} "
        f"requires_grad={tensor.requires_grad} "
        f"grad_fn={type(tensor.grad_fn).__name__ if tensor.grad_fn else None} "
        f"semantic={semantic}",
        flush=True,
    )


class FakeCommand:
    def __init__(self) -> None:
        self.reference_window = None
        self.env_ids = None
        self.call_count = 0

    def set_frontres_reference_window(self, reference_window: torch.Tensor, *, env_ids: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        self.reference_window = reference_window.detach().clone()
        self.env_ids = env_ids.detach().clone()
        return torch.ones(int(env_ids.numel()), dtype=torch.bool, device=env_ids.device)


class FakeEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.max_episode_length = 16
        self.events: list[str] = []
        self.command = FakeCommand()

    def apply_frontres_segment_reset(self, request):
        self.events.append("reset")
        self.last_reset_request = request
        applied = self.command.set_frontres_reference_window(
            reference_window=request.reference_window,
            env_ids=torch.arange(int(request.segment_ids.numel()), dtype=torch.long),
        )
        return {
            "success_mask": torch.ones(int(request.segment_ids.numel()), dtype=torch.bool),
            "velocity_mismatch": torch.zeros(int(request.segment_ids.numel())),
            "reference_window_applied": applied,
        }

    def get_observations(self):
        self.events.append("get_obs")
        obs = torch.tensor([[1.0, 0.0, 0.5, -0.5]], dtype=torch.float32)
        return obs, {"observations": {}}

    def step(self, actions):
        self.events.append("step")
        obs = torch.tensor([[2.0, 0.0, 0.5, -0.5]], dtype=torch.float32)
        rewards = torch.tensor([2.0], dtype=torch.float32)
        dones = torch.tensor([False], dtype=torch.bool)
        return obs, rewards, dones, {"observations": {}}


class FakeRunner:
    def __init__(self) -> None:
        self.env = FakeEnv()
        self.device = torch.device("cpu")
        self.seed = 17
        self.current_learning_iteration = 0
        self.policy_obs_type = None
        self.privileged_obs_type = None
        self.teacher_obs_type = None
        self.ref_vel_estimator_obs_type = None
        self._frontres_segment_replay_boundary = SimpleNamespace(
            requested=True,
            live_runner_enabled=True,
            live_probe_only=False,
            live_storage_write_only=True,
            live_single_update_only=False,
            live_update_loop_only=False,
            live_train_enabled=False,
            segment_k=2,
            reset_mode="direct",
        )
        self.alg = SimpleNamespace(
            frontres_training_objective="segment_replay_hrl",
            frontres_segment_k=2,
            frontres_segment_reset_mode="direct",
            frontres_segment_preroll_steps=0,
            frontres_segment_reset_velocity_tolerance=1e-3,
            transition=SimpleNamespace(),
        )
        self._frontres_segment_dataset = _dataset()
        self._frontres_segment_sampler = FrontRESSegmentSampler(
            num_segments=self._frontres_segment_dataset.num_segments(),
            seed=17,
            global_frac=1.0,
            replay_frac=0.0,
            review_frac=0.0,
            device=self.device,
        )

    def eval_mode(self) -> None:
        self.mode = "eval"

    def _apply_obs_normalizer(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def privileged_obs_normalizer(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def teacher_obs_normalizer(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def run_frontres_segment_live_probe(self, *, init_at_random_ep_len: bool):
        return run_frontres_segment_live_probe(self, init_at_random_ep_len=init_at_random_ep_len)


def _dataset() -> FrontRESSegmentDataset:
    frames = 3
    dof = 2
    motion = {
        "motion_id": "step17_toy_motion",
        "root_pos": torch.tensor([[0.0, 0.0, 0.9], [0.1, 0.0, 0.9], [0.2, 0.0, 0.9]], dtype=torch.float32),
        "root_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(frames, 1),
        "root_lin_vel": torch.ones(frames, 3) * 0.1,
        "root_ang_vel": torch.ones(frames, 3) * 0.2,
        "dof_pos": torch.arange(frames * dof, dtype=torch.float32).reshape(frames, dof),
        "dof_vel": torch.arange(frames * dof, dtype=torch.float32).reshape(frames, dof) + 100.0,
        "horizon_k": 2,
        "perturbation_family": "hrl_curriculum_bank",
        "perturbation_strength": 1.0,
        "perturbation_role": "train",
        "reset_mode_hint": "direct",
    }
    return FrontRESSegmentDataset([motion], dt=1.0 / 30.0, default_horizon_k=2, device="cpu")


def test_segment_sampler_step_crosses_reset_rollout_storage_evidence_boundary() -> None:
    runner = FakeRunner()
    summary = run_frontres_segment_sampler_step(
        runner,
        init_at_random_ep_len=False,
        update_step=17,
    )
    sampler = runner._frontres_segment_sampler
    request = runner.env.last_reset_request
    expected_reference = runner.env.command.reference_window

    _probe_tensor("sampled.segment_ids", request.segment_ids, "sampled ids carried into reset request")
    _probe_tensor("batch.reference_window", request.reference_window, "dataset reference window before env reset")
    _probe_tensor("command.reference_window", expected_reference, "reference window written into command hook")
    _probe_tensor("sampler.seen", sampler.seen, "sampler seen flags after evidence update")
    _probe_tensor("sampler.invalid", sampler.invalid, "sampler invalid flags after evidence update")
    _probe_tensor("sampler.priority", sampler.priority, "sampler priority after reward evidence update")
    print(
        "[probe step17] closed_loop_summary: "
        f"events={runner.env.events} "
        f"reference_applied_frac={summary['segment_reference_window_applied_frac']} "
        f"storage_size={summary['storage_size']} "
        f"storage_segment_ids={summary['storage_segment_ids']} "
        f"storage_reward={summary['storage_reward_per_sample']} "
        f"storage_valid={summary['storage_valid_mask_per_sample']} "
        f"sampler_update={summary['sampler_update']} "
        f"sampler_seen={int(sampler.seen.sum().item())} "
        f"sampler_priority={sampler.priority.detach().cpu().tolist()}",
        flush=True,
    )

    assert runner.env.events == ["reset", "get_obs", "step", "step"]
    assert request.segment_ids.tolist() == [0]
    assert tuple(request.reference_window.shape) == (1, 3, 4)
    torch.testing.assert_close(expected_reference, request.reference_window)
    assert runner.env.command.call_count == 1
    assert summary["segment_reset"] is True
    assert summary["segment_reference_window_applied_frac"] == 1.0
    assert summary["storage_write"] is True
    assert summary["storage_size"] == 1
    assert summary["storage_segment_ids"] == [0]
    assert summary["storage_valid_mask_per_sample"] == [True]
    assert summary["storage_reward_per_sample"] == [2.0]
    assert summary["sampler_update"] is True
    assert int(sampler.seen.sum().item()) == 1
    assert float(sampler.priority[0].item()) > 0.0
    assert getattr(runner, "_frontres_segment_live_current_batch", None) is None
    assert getattr(runner, "_frontres_segment_live_current_reset_request", None) is None
    assert getattr(runner, "_frontres_segment_live_current_reset_result", None) is None


def main() -> None:
    test_segment_sampler_step_crosses_reset_rollout_storage_evidence_boundary()
    print("frontres_segment_live_closed_loop_contract: ok")


if __name__ == "__main__":
    main()
