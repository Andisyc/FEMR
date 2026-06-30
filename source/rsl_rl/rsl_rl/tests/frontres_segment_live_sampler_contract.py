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
    reset_module = _load(
        "rsl_rl.frontres.frontres_segment_reset",
        ROOT / "rsl_rl" / "frontres" / "frontres_segment_reset.py",
    )
    frontres_pkg.frontres_segment_reset = reset_module

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
schema_module = _load(
    "frontres_segment_cache_schema_for_live_sampler_contract",
    ROOT / "rsl_rl" / "frontres" / "frontres_segment_cache_schema.py",
)
indexer_module = _load(
    "frontres_segment_cache_indexer_for_live_sampler_contract",
    ROOT / "rsl_rl" / "frontres" / "frontres_segment_cache_indexer.py",
)
cache_io_module = _load(
    "frontres_segment_cache_io_for_live_sampler_contract",
    ROOT / "rsl_rl" / "frontres" / "frontres_segment_cache_io.py",
)

FrontRESSegmentSampler = sampler_module.FrontRESSegmentSampler
initialize_frontres_segment_live_sampler = live_sampler_module.initialize_frontres_segment_live_sampler
build_live_sampler_evidence = live_sampler_module.build_live_sampler_evidence
run_frontres_segment_live_update_loop = live_update_loop_module.run_frontres_segment_live_update_loop
FrontRESSegmentLiveRolloutCapture = live_probe_module.FrontRESSegmentLiveRolloutCapture
build_live_segment_storage = live_probe_module.build_live_segment_storage
save_runner = checkpointing_module.save_runner
load_runner = checkpointing_module.load_runner
FrontRESSegmentIndex = schema_module.FrontRESSegmentIndex
FrontRESRobotRolloutState = schema_module.FrontRESRobotRolloutState
FrontRESPerturbationDescriptor = schema_module.FrontRESPerturbationDescriptor
FrontRESNoisyVariant = schema_module.FrontRESNoisyVariant
FrontRESAMASSIndexSummary = indexer_module.FrontRESAMASSIndexSummary
FrontRESCleanStateEntry = cache_io_module.FrontRESCleanStateEntry


class FakeBoundary:
    requested = True
    live_runner_enabled = True
    live_update_loop_only = True
    live_train_enabled = False
    live_update_steps = 3


class FakeEnv:
    num_envs = 2


class FakeRunner:
    def __init__(
        self,
        summaries: list[dict] | None = None,
        cache_dir: str = "",
        shard_cache_size: int = 8,
    ) -> None:
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
            frontres_segment_cache_dir=cache_dir,
            frontres_segment_shard_cache_size=shard_cache_size,
            frontres_segment_include_boundary_diagnostic=False,
        )
        self.summaries = summaries or []
        self.probe_init_flags: list[bool] = []
        self.probe_batch_roles: list[tuple[str, ...] | None] = []
        self.probe_batch_ids: list[list[int] | None] = []

    def run_frontres_segment_live_probe(self, *, init_at_random_ep_len: bool) -> dict:
        self.probe_init_flags.append(init_at_random_ep_len)
        batch = getattr(self, "_frontres_segment_live_current_batch", None)
        self.probe_batch_roles.append(None if batch is None else tuple(batch.perturbation_role))
        self.probe_batch_ids.append(None if batch is None else batch.segment_ids.detach().cpu().tolist())
        index = min(len(self.probe_init_flags) - 1, len(self.summaries) - 1)
        summary = dict(self.summaries[index])
        print(
            "[probe step22] fake_live_probe: "
            f"call={len(self.probe_init_flags)} "
            f"init_at_random_ep_len={init_at_random_ep_len} "
            f"batch_ids={self.probe_batch_ids[-1]} "
            f"batch_roles={self.probe_batch_roles[-1]} "
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


def _summary_per_sample(
    rewards: list[float],
    storage_valid: list[bool],
    done_any: list[bool],
) -> dict:
    assert len(rewards) == len(storage_valid) == len(done_any)
    valid_count = sum(1 for item in storage_valid if item)
    return {
        "ppo_update": valid_count > 0,
        "ppo_valid_count": valid_count,
        "reward_mean": float(sum(rewards) / max(1, len(rewards))),
        "reward_per_sample": list(rewards),
        "done_frac": float(sum(1 for item in done_any if item) / max(1, len(done_any))),
        "done_any_per_sample": list(done_any),
        "storage_size": len(rewards),
        "storage_valid_frac": float(valid_count / max(1, len(storage_valid))),
        "storage_reward_mean": float(sum(rewards) / max(1, len(rewards))),
        "storage_reward_per_sample": list(rewards),
        "storage_valid_mask_per_sample": list(storage_valid),
        "ppo_total_loss": 0.5,
        "ppo_actor_loss": 0.1,
        "ppo_value_loss": 0.2,
        "ppo_approx_kl": 0.01,
        "ppo_clip_frac": 0.0,
    }


def _cache_segment() -> FrontRESSegmentIndex:
    return FrontRESSegmentIndex(
        segment_id=0,
        motion_rel_path="KIT/359/motion_a.npz",
        motion_num_frames=8,
        fps=30.0,
        start_frame=2,
        horizon_k=4,
    )


def _cache_state(offset: float) -> FrontRESRobotRolloutState:
    batch = 1
    dofs = 29
    bodies = 30
    return FrontRESRobotRolloutState(
        root_pos=torch.full((batch, 3), offset),
        root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        root_lin_vel=torch.full((batch, 3), offset + 0.1),
        root_ang_vel=torch.full((batch, 3), offset + 0.2),
        joint_pos=torch.arange(dofs, dtype=torch.float32).view(batch, dofs) + offset,
        joint_vel=torch.arange(dofs, dtype=torch.float32).view(batch, dofs) * 0.01 + offset,
        body_pos_w=torch.full((batch, bodies, 3), offset + 1.0),
        body_quat_w=torch.zeros(batch, bodies, 4).index_fill(2, torch.tensor([0]), 1.0),
        body_lin_vel_w=torch.full((batch, bodies, 3), offset + 0.3),
        body_ang_vel_w=torch.full((batch, bodies, 3), offset + 0.4),
        contact_state=torch.ones(batch, 4),
        action_history=torch.zeros(batch, 2, dofs),
    )


def _cache_descriptor(perturbation_id: int, strength: float, role: str) -> FrontRESPerturbationDescriptor:
    return FrontRESPerturbationDescriptor(
        perturbation_id=perturbation_id,
        segment_id=0,
        strength=strength,
        seed=900 + perturbation_id,
        family="hrl_curriculum_bank",
        start_step=0,
        duration=4,
        target="torso_link",
        frame="world",
        params={
            "curriculum_mode": "hrl_curriculum_bank",
            "family_group": ("planar", "yaw"),
            "mix_class": "hard" if role == "boundary_diagnostic" else "frontier",
            "mix_class_index": 2 if role == "boundary_diagnostic" else 1,
            "frontier_scale": 2.0,
            "dr_factor": 1.08 if role == "boundary_diagnostic" else 1.0,
            "actual_dr_scale": strength,
            "perturbation_role": role,
            "temporal_mode": "single",
            "burst_min_steps": 4,
            "burst_max_steps": 8,
        },
    )


def _write_stage1_cache(cache_dir: Path) -> None:
    segment = _cache_segment()
    summary = FrontRESAMASSIndexSummary(
        amass_root="/tmp/fake_amass",
        motion_count=1,
        segment_count=1,
        horizon_k=4,
        frame_stride=1,
        skipped_short_motions=0,
    )
    indexer_module.write_amass_segment_index(cache_dir, [segment], summary)
    cache_io_module.write_clean_state_shard(
        cache_dir,
        [FrontRESCleanStateEntry(segment=segment, clean_state=_cache_state(0.0))],
        shard_id=0,
    )
    for perturbation_id, strength, role in (
        (0, 1.5, "train"),
        (1, 2.16, "boundary_diagnostic"),
    ):
        cache_io_module.write_noisy_variant_shard(
            cache_dir,
            [
                FrontRESNoisyVariant(
                    segment=segment,
                    descriptor=_cache_descriptor(perturbation_id, strength, role),
                    noisy_state=_cache_state(strength),
                    noisy_baseline_score=torch.tensor([0.4 + strength], dtype=torch.float32),
                    noisy_fall=torch.tensor([1.0 if role == "boundary_diagnostic" else 0.0], dtype=torch.float32),
                    noisy_rollout_len=torch.tensor([4.0], dtype=torch.float32),
                )
            ],
            strength=strength,
            shard_id=0,
        )
    cache_io_module.write_cache_metadata(
        cache_dir,
        {
            "stage": "stage1_segment_cache",
            "segment_count": 1,
            "clean_count": 1,
            "noisy_count": 2,
            "horizon_k": 4,
            "frame_stride": 1,
            "strengths": [1.5, 2.16],
            "perturbation_curriculum_mode": "hrl_curriculum_bank",
            "clean_shard_id": 0,
            "noisy_shard_id": 0,
        },
    )


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


def test_live_sampler_evidence_carries_partial_reset_failure() -> None:
    sampler = FrontRESSegmentSampler(4, seed=2)
    sample = sampler.sample(2)
    reset_result = SimpleNamespace(success_mask=torch.tensor([True, False]))
    evidence = build_live_sampler_evidence(sample, _summary(0.4), horizon_k=4, reset_result=reset_result)
    print(
        "[probe step12] sampler_reset_evidence: "
        f"ids={evidence.segment_ids.tolist()} "
        f"reset_success={evidence.reset_success.tolist()} "
        f"valid_reward={evidence.valid_reward.tolist()} "
        f"gain={evidence.gain_over_noisy.tolist()}",
        flush=True,
    )
    assert evidence.reset_success.tolist() == [True, False]
    assert evidence.valid_reward.tolist() == [True, False]
    torch.testing.assert_close(evidence.gain_over_noisy, torch.full((2,), 0.4))


def test_live_sampler_evidence_preserves_per_sample_rollout_facts() -> None:
    sampler = FrontRESSegmentSampler(4, seed=12)
    sample = sampler.sample(2)
    reset_result = SimpleNamespace(success_mask=torch.tensor([True, False]))
    summary = _summary_per_sample(
        rewards=[0.8, -0.2],
        storage_valid=[True, False],
        done_any=[False, True],
    )
    evidence = build_live_sampler_evidence(sample, summary, horizon_k=4, reset_result=reset_result)
    print(
        "[probe step14] per_sample_evidence: "
        f"ids={evidence.segment_ids.tolist()} "
        f"reward={summary['storage_reward_per_sample']} "
        f"reset_success={evidence.reset_success.tolist()} "
        f"valid_reward={evidence.valid_reward.tolist()} "
        f"fall={evidence.fall_repaired.tolist()} "
        f"gain={evidence.gain_over_noisy.tolist()}",
        flush=True,
    )
    assert evidence.reset_success.tolist() == [True, False]
    assert evidence.valid_reward.tolist() == [True, False]
    assert evidence.fall_repaired.tolist() == [False, True]
    torch.testing.assert_close(evidence.gain_over_noisy, torch.tensor([0.8, -0.2]))


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


def test_live_sampler_initializes_dataset_from_stage1_cache_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "AMASS_G1Segment"
        _write_stage1_cache(cache_dir)
        runner = FakeRunner(cache_dir=str(cache_dir))
        initialize_frontres_segment_live_sampler(runner)
        metadata = runner._frontres_segment_dataset.cache_metadata()
        print(
            "[probe step23] cache_dataset_sampler: "
            f"cache_dir={cache_dir} "
            f"dataset_segments={runner._frontres_segment_dataset.num_segments()} "
            f"sampler_segments={runner._frontres_segment_sampler.num_segments} "
            f"metadata={metadata}",
            flush=True,
        )
        assert runner._frontres_segment_dataset.num_segments() == 1
        assert runner._frontres_segment_sampler.num_segments == 1
        assert metadata["loaded_motion_count"] == 1
        assert metadata["skipped_boundary_diagnostic_count"] == 1
        assert metadata["role_counts"] == {"train": 1, "boundary_diagnostic": 1}


def test_live_sampler_passes_nondefault_shard_cache_size_to_lazy_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "AMASS_G1Segment"
        _write_stage1_cache(cache_dir)
        runner = FakeRunner(cache_dir=str(cache_dir), shard_cache_size=1)
        initialize_frontres_segment_live_sampler(runner)
        metadata = runner._frontres_segment_dataset.cache_metadata()
        print(
            "[probe step23] shard_cache_size: "
            f"alg_value={runner.alg.frontres_segment_shard_cache_size} "
            f"metadata={metadata}",
            flush=True,
        )
        assert metadata["shard_cache"]["max_shards"] == 1


def test_live_sampler_builds_current_batch_before_probe() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "AMASS_G1Segment"
        _write_stage1_cache(cache_dir)
        runner = FakeRunner([_summary(0.3)], cache_dir=str(cache_dir))
        initialize_frontres_segment_live_sampler(runner)
        result = run_frontres_segment_live_update_loop(runner, init_at_random_ep_len=True, runner_learn=True)
        print(
            "[probe step24] current_batch: "
            f"probe_batch_ids={runner.probe_batch_ids} "
            f"probe_batch_roles={runner.probe_batch_roles} "
            f"sampler_update_count={result['sampler_update_count']}",
            flush=True,
        )
        assert runner.probe_batch_ids == [[0, 0], [0, 0], [0, 0]]
        assert runner.probe_batch_roles == [("train", "train"), ("train", "train"), ("train", "train")]
        assert result["sampler_update_count"] == 3
        assert getattr(runner, "_frontres_segment_live_current_batch", None) is None
        assert getattr(runner, "_frontres_segment_live_current_reset_request", None) is None
        assert getattr(runner, "_frontres_segment_live_current_reset_result", None) is None


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
    test_live_sampler_evidence_carries_partial_reset_failure()
    test_live_sampler_evidence_preserves_per_sample_rollout_facts()
    test_live_update_loop_samples_and_updates_priority()
    test_live_sampler_initializes_dataset_from_stage1_cache_dir()
    test_live_sampler_passes_nondefault_shard_cache_size_to_lazy_dataset()
    test_live_sampler_builds_current_batch_before_probe()
    test_live_storage_uses_sampled_segment_ids_and_sources()
    test_runner_checkpoint_saves_and_restores_sampler_state()
    print("frontres_segment_live_sampler_contract: ok")


if __name__ == "__main__":
    main()
