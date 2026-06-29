#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
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


def _install_import_stubs() -> None:
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

    training_schedule = types.ModuleType("rsl_rl.frontres.training_schedule")
    training_schedule.configure_frontres_pair_layout = lambda *_args, **_kwargs: None
    training_schedule.resolve_frontres_mode_state = lambda *_args, **_kwargs: None
    sys.modules[training_schedule.__name__] = training_schedule
    frontres_pkg.training_schedule = training_schedule

    storage_module = _load(
        "rsl_rl.frontres.frontres_segment_storage",
        ROOT / "rsl_rl" / "frontres" / "frontres_segment_storage.py",
    )
    frontres_pkg.frontres_segment_storage = storage_module

    class _FakeFrontRESActorCritic:
        pass

    class _FakeResidualActorCritic:
        pass

    modules_pkg = types.ModuleType("rsl_rl.modules")
    modules_pkg.FrontRESActorCritic = _FakeFrontRESActorCritic
    modules_pkg.ResidualActorCritic = _FakeResidualActorCritic
    sys.modules[modules_pkg.__name__] = modules_pkg
    rsl_rl_pkg.modules = modules_pkg

    rollout_step = types.ModuleType("rsl_rl.runners.frontres_rollout_step")
    rollout_step.prepare_frontres_rollout_step = lambda *_args, **_kwargs: None
    sys.modules[rollout_step.__name__] = rollout_step
    runners_pkg.frontres_rollout_step = rollout_step


_install_import_stubs()


sampler_module = _load(
    "frontres_segment_sampler",
    ROOT / "rsl_rl" / "frontres" / "frontres_segment_sampler.py",
)
live_sampler_module = _load(
    "frontres_segment_live_sampler",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_live_sampler.py",
)
live_update_loop_module = _load(
    "frontres_segment_live_update_loop",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_live_update_loop.py",
)
live_probe_module = _load(
    "frontres_segment_live_probe",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_live_probe.py",
)
checkpointing_module = _load(
    "frontres_checkpointing",
    ROOT / "rsl_rl" / "runners" / "frontres_checkpointing.py",
)

FrontRESSegmentSampler = sampler_module.FrontRESSegmentSampler
initialize_frontres_segment_live_sampler = live_sampler_module.initialize_frontres_segment_live_sampler
build_live_sampler_evidence = live_sampler_module.build_live_sampler_evidence
run_frontres_segment_live_update_loop = live_update_loop_module.run_frontres_segment_live_update_loop
FrontRESSegmentLiveRolloutCapture = live_probe_module.FrontRESSegmentLiveRolloutCapture
build_live_segment_storage = live_probe_module.build_live_segment_storage
save_runner = checkpointing_module.save_runner
load_runner = checkpointing_module.load_runner


class FakeBoundary:
    requested = True
    live_runner_enabled = True
    live_update_loop_only = True
    live_train_enabled = False
    live_update_steps = 3


class FakeEnv:
    num_envs = 2


class FakeRunner:
    def __init__(self, summaries: list[dict] | None = None) -> None:
        self._frontres_segment_replay_boundary = FakeBoundary()
        self.env = FakeEnv()
        self.device = "cpu"
        self.seed = 7
        self.alg = SimpleNamespace(
            frontres_training_objective="segment_replay_hrl",
            frontres_segment_sampler_global_frac=0.4,
            frontres_segment_sampler_replay_frac=0.5,
            frontres_segment_sampler_review_frac=0.1,
            frontres_segment_live_update_steps=3,
            frontres_segment_k=4,
        )
        self.summaries = summaries or []
        self.probe_init_flags: list[bool] = []

    def run_frontres_segment_live_probe(self, *, init_at_random_ep_len: bool) -> dict:
        self.probe_init_flags.append(init_at_random_ep_len)
        index = min(len(self.probe_init_flags) - 1, len(self.summaries) - 1)
        summary = dict(self.summaries[index])
        print(
            "[probe step22] fake_live_probe: "
            f"call={len(self.probe_init_flags)} "
            f"init_at_random_ep_len={init_at_random_ep_len} "
            f"reward_mean={summary['reward_mean']} "
            f"storage_reward_mean={summary['storage_reward_mean']} "
            f"ppo_valid_count={summary['ppo_valid_count']}",
            flush=True,
        )
        return summary


def _summary(reward: float, valid_count: int = 2) -> dict:
    return {
        "ppo_update": valid_count > 0,
        "ppo_valid_count": valid_count,
        "reward_mean": reward,
        "done_frac": 0.0,
        "storage_size": 2,
        "storage_valid_frac": 1.0 if valid_count > 0 else 0.0,
        "storage_reward_mean": reward,
        "ppo_total_loss": 0.5,
        "ppo_actor_loss": 0.1,
        "ppo_value_loss": 0.2,
        "ppo_approx_kl": 0.01,
        "ppo_clip_frac": 0.0,
    }


def test_live_summary_becomes_sampler_evidence() -> None:
    sampler = FrontRESSegmentSampler(4, seed=1)
    sample = sampler.sample(2)
    evidence = build_live_sampler_evidence(sample, _summary(0.4), horizon_k=4)
    print(
        "[probe step22] evidence: "
        f"ids={evidence.segment_ids.tolist()} "
        f"gain_mean={float(evidence.gain_over_noisy.mean()):.6f} "
        f"valid_count={int(evidence.valid_reward.sum())} "
        f"horizon={evidence.horizon_k.tolist()}",
        flush=True,
    )
    assert tuple(evidence.segment_ids.shape) == (2,)
    assert evidence.valid_reward.tolist() == [True, True]
    assert torch.all(evidence.gain_over_noisy > 0.0)
    assert evidence.horizon_k.tolist() == [4, 4]


def test_live_update_loop_samples_and_updates_priority() -> None:
    runner = FakeRunner([_summary(0.4), _summary(0.2), _summary(0.1)])
    initialize_frontres_segment_live_sampler(runner)
    assert runner._frontres_segment_sampler.stats().seen_count == 0
    result = run_frontres_segment_live_update_loop(runner, init_at_random_ep_len=True, runner_learn=True)
    stats = runner._frontres_segment_sampler.stats()
    print(
        "[probe step22] after_update_loop: "
        f"probe_init_flags={runner.probe_init_flags} "
        f"sampler_update_count={result['sampler_update_count']} "
        f"source_total={result['sampler_global_count'] + result['sampler_replay_count'] + result['sampler_review_count']} "
        f"seen_count={stats.seen_count} "
        f"priority_mean={stats.priority_mean:.6f} "
        f"replay_pool_size={stats.replay_pool_size}",
        flush=True,
    )
    assert runner.probe_init_flags == [True, False, False]
    assert result["sampler_update_count"] == 3
    assert result["sampler_global_count"] + result["sampler_replay_count"] + result["sampler_review_count"] == 6
    assert stats.seen_count > 0
    assert stats.priority_mean > 0.0


def test_live_storage_uses_sampled_segment_ids_and_sources() -> None:
    runner = FakeRunner()
    initialize_frontres_segment_live_sampler(runner)
    sample = runner._frontres_segment_sampler.sample(2)
    runner._frontres_segment_live_current_sample = sample
    capture = FrontRESSegmentLiveRolloutCapture(
        rollout_k=4,
        reward_mean=0.4,
        done_frac=0.0,
        last_obs_shape=(2, 3),
        action_shape=(2, 6),
        env_action_shape=(2, 12),
        transition_obs=torch.zeros(2, 3),
        transition_privileged_obs=torch.ones(2, 4),
        transition_actions=torch.ones(2, 6),
        transition_log_probs=torch.zeros(2),
        transition_values=torch.zeros(2),
        transition_means=torch.zeros(2, 6),
        transition_sigmas=torch.ones(2, 6),
        reward_accum=torch.ones(2),
        done_any=torch.zeros(2, dtype=torch.bool),
    )
    storage = build_live_segment_storage(runner, capture)
    state = storage.state_dict()
    print(
        "[probe step22] storage_ids: "
        f"sample_ids={sample.segment_ids.tolist()} "
        f"storage_ids={state['segment_ids'].tolist()} "
        f"sample_sources={list(sample.source)} "
        f"storage_sources={list(state['segment_source'])}",
        flush=True,
    )
    assert state["segment_ids"].tolist() == sample.segment_ids.tolist()
    assert tuple(state["segment_source"]) == tuple(sample.source)


def test_runner_checkpoint_saves_and_restores_sampler_state() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "model_3.pt")
        policy = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        runner = SimpleNamespace(
            alg=SimpleNamespace(policy=policy, optimizer=optimizer),
            current_learning_iteration=3,
            cfg={"is_full_resume": True},
            alg_cfg={"learning_rate": 1e-3},
            policy_cfg={"init_noise_std": 1.0, "noise_std_type": "scalar"},
            empirical_normalization=False,
            training_type="frontres",
            logger_type="",
            disable_logs=True,
            writer=None,
            device="cpu",
            _frontres_segment_sampler=FrontRESSegmentSampler(4, seed=3),
        )
        sample = runner._frontres_segment_sampler.sample(2)
        evidence = build_live_sampler_evidence(sample, _summary(0.5), horizon_k=4)
        runner._frontres_segment_sampler.update(evidence)
        saved_priority = runner._frontres_segment_sampler.priority.detach().clone()
        save_runner(runner, path)

        resumed_policy = torch.nn.Linear(1, 1)
        resumed = SimpleNamespace(
            alg=SimpleNamespace(policy=resumed_policy, optimizer=torch.optim.Adam(resumed_policy.parameters(), lr=1e-3)),
            current_learning_iteration=0,
            cfg={"is_full_resume": True},
            alg_cfg={"learning_rate": 1e-3},
            policy_cfg={"init_noise_std": 1.0, "noise_std_type": "scalar"},
            empirical_normalization=False,
            training_type="frontres",
            logger_type="",
            disable_logs=True,
            writer=None,
            device="cpu",
            _frontres_segment_sampler=FrontRESSegmentSampler(4, seed=4),
        )
        load_runner(resumed, path, load_optimizer=False)
        print(
            "[probe step22] checkpoint_sampler: "
            f"loaded_path={resumed._frontres_last_loaded_checkpoint_path} "
            f"iteration={resumed.current_learning_iteration} "
            f"saved_priority={saved_priority.tolist()} "
            f"loaded_priority={resumed._frontres_segment_sampler.priority.tolist()}",
            flush=True,
        )
        torch.testing.assert_close(resumed._frontres_segment_sampler.priority, saved_priority)
        assert resumed.current_learning_iteration == 3


def main() -> None:
    test_live_summary_becomes_sampler_evidence()
    test_live_update_loop_samples_and_updates_priority()
    test_live_storage_uses_sampled_segment_ids_and_sources()
    test_runner_checkpoint_saves_and_restores_sampler_state()
    print("frontres_segment_live_sampler_contract: ok")


if __name__ == "__main__":
    main()
