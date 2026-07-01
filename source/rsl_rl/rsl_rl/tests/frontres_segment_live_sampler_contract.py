#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import contextlib
import io
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
stage1_env_hooks_module = _load(
    "rsl_rl.frontres.frontres_segment_stage1_env_hooks",
    ROOT / "rsl_rl" / "frontres" / "frontres_segment_stage1_env_hooks.py",
)
stage1_hooks_contract = _load(
    "frontres_segment_stage1_env_hooks_contract_for_live_sampler",
    ROOT / "rsl_rl" / "tests" / "frontres_segment_stage1_env_hooks_contract.py",
)

FrontRESSegmentSampler = sampler_module.FrontRESSegmentSampler
FrontRESSegmentSample = sampler_module.FrontRESSegmentSample
initialize_frontres_segment_live_sampler = live_sampler_module.initialize_frontres_segment_live_sampler
build_live_sampler_evidence = live_sampler_module.build_live_sampler_evidence
run_frontres_segment_sampler_step = live_sampler_module.run_frontres_segment_sampler_step
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
        env: object | None = None,
    ) -> None:
        self._frontres_segment_replay_boundary = FakeBoundary()
        self.env = env if env is not None else FakeEnv()
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


def _write_stage1_index_cache(cache_dir: Path, amass_root: Path) -> None:
    segment = FrontRESSegmentIndex(
        segment_id=0,
        motion_rel_path="KIT/359/motion_a.npz",
        motion_num_frames=8,
        fps=30.0,
        start_frame=3,
        horizon_k=4,
    )
    summary = FrontRESAMASSIndexSummary(
        amass_root=str(amass_root),
        motion_count=1,
        segment_count=1,
        horizon_k=4,
        frame_stride=1,
        skipped_short_motions=0,
    )
    indexer_module.write_amass_segment_index(cache_dir, [segment], summary)


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


def test_large_sampler_probe_uses_summary_not_full_lists() -> None:
    count = 12000
    ids = torch.arange(count, dtype=torch.long)
    sample = FrontRESSegmentSample(
        segment_ids=ids,
        source=tuple("global" if i % 3 == 0 else "replay" if i % 3 == 1 else "review" for i in range(count)),
        priority=torch.linspace(0.0, 1.0, count),
        staleness=torch.ones(count),
        valid_mask=torch.ones(count, dtype=torch.bool),
    )

    class _Dataset:
        def get_segments(self, segment_ids):
            return SimpleNamespace(
                segment_ids=segment_ids,
                perturbation_role=tuple("train" if int(i) % 2 == 0 else "review" for i in segment_ids.tolist()),
                perturbation_strength=torch.linspace(0.0, 2.0, int(segment_ids.numel())),
            )

        def validate_batch(self, batch):
            return SimpleNamespace(valid_mask=torch.ones_like(batch.segment_ids, dtype=torch.bool))

    runner = SimpleNamespace(
        alg=SimpleNamespace(frontres_segment_verbose_probe=False),
        _frontres_segment_dataset=_Dataset(),
    )
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        live_sampler_module._print_sample_probe(0, sample, verbose=False)
        live_sampler_module._build_current_segment_batch(runner, sample, update_step=0)
    output = stream.getvalue()
    contains_sources_list = "sources=['global'" in output
    contains_segment_ids_list = "segment_ids=[0, 1, 2" in output
    contains_source_counts = "source_counts" in output
    contains_role_counts = "role_counts" in output
    print(
        "[probe step22] large_log_summary: "
        f"contains_count={'count=12000' in output} "
        f"contains_sources_list={contains_sources_list} "
        f"contains_segment_ids_list={contains_segment_ids_list} "
        f"contains_source_counts={contains_source_counts} "
        f"contains_role_counts={contains_role_counts}",
        flush=True,
    )
    assert "count=12000" in output
    assert "sample.source_counts:" in output
    assert "batch.role_counts:" in output
    assert "strength_count=12000" in output
    assert "sources=['global'" not in output
    assert "segment_ids=[0, 1, 2" not in output
    assert "strength=[0.0" not in output


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


def test_live_sampler_summary_exposes_update_probe_boundary() -> None:
    runner = FakeRunner([_summary_per_sample([0.8, -0.2], [True, True], [False, True])])
    initialize_frontres_segment_live_sampler(runner)
    result = run_frontres_segment_sampler_step(runner, init_at_random_ep_len=False, update_step=0)
    print(
        "[probe step22] sampler_update_boundary: "
        f"useful_mean={result['sampler_update_useful_mean']:.6f} "
        f"useful_max={result['sampler_update_useful_max']:.6f} "
        f"priority_before={result['sampler_update_priority_before_mean']:.6f} "
        f"priority_after={result['sampler_update_priority_after_mean']:.6f} "
        f"gain_pos_frac={result['sampler_update_gain_pos_frac']:.6f} "
        f"replay_candidates={result['sampler_update_replay_candidate_count']}",
        flush=True,
    )
    assert result["sampler_update_valid_count"] == 2
    assert result["sampler_update_fall_count"] == 1
    assert result["sampler_update_useful_max"] > 0.0
    assert result["sampler_update_priority_after_mean"] > result["sampler_update_priority_before_mean"]
    assert result["sampler_update_hopeless_count"] == 1


def test_live_detail_logs_are_rate_limited_by_default_and_verbose() -> None:
    runner = FakeRunner([_summary(0.1) for _ in range(12)])
    initialize_frontres_segment_live_sampler(runner)

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        for update_step in range(12):
            run_frontres_segment_sampler_step(runner, init_at_random_ep_len=False, update_step=update_step)
    output = stream.getvalue()
    sample_count = output.count("[FrontRES Segment Sample]")
    batch_count = output.count("[FrontRES Segment Batch]")
    evidence_count = output.count("[probe step14] evidence_path:")
    sampler_count = output.count("[FrontRES Segment Sampler]")
    print(
        "[probe step6] live_detail_log_rate: "
        f"sample_count={sample_count} "
        f"batch_count={batch_count} "
        f"evidence_count={evidence_count} "
        f"sampler_count={sampler_count} "
        f"call_count={runner._frontres_segment_live_detail_log_count}",
        flush=True,
    )

    assert sample_count == 4
    assert batch_count == 4
    assert evidence_count == 4
    assert sampler_count == 4
    assert runner._frontres_segment_live_detail_log_count == 12

    verbose_runner = FakeRunner([_summary(0.1) for _ in range(4)])
    verbose_runner.alg.frontres_segment_verbose_probe = True
    initialize_frontres_segment_live_sampler(verbose_runner)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        for update_step in range(4):
            run_frontres_segment_sampler_step(verbose_runner, init_at_random_ep_len=False, update_step=update_step)
    verbose_output = stream.getvalue()
    verbose_sample_count = verbose_output.count("[FrontRES Segment Sample]")
    verbose_sampler_count = verbose_output.count("[FrontRES Segment Sampler]")
    print(
        "[probe step6] live_detail_log_verbose_rate: "
        f"sample_count={verbose_sample_count} "
        f"sampler_count={verbose_sampler_count} "
        f"verbose={verbose_runner.alg.frontres_segment_verbose_probe}",
        flush=True,
    )

    assert verbose_sample_count == 4
    assert verbose_sampler_count == 4


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


def test_live_sampler_installs_index_reset_hook_for_index_only_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        amass_root = Path(tmp) / "AMASS_G1NPZ_Final"
        cache_dir = Path(tmp) / "AMASS_G1Segment"
        stage1_hooks_contract._write_fake_amass(amass_root / "KIT" / "359" / "motion_a.npz")
        _write_stage1_index_cache(cache_dir, amass_root)
        env = stage1_hooks_contract.FakeGymEnv(amass_root)
        runner = FakeRunner(cache_dir=str(cache_dir), env=env)
        initialize_frontres_segment_live_sampler(runner)
        hook = getattr(runner.env, "apply_frontres_segment_index_reset", None)
        assert callable(hook)

        batch = runner._frontres_segment_dataset.get_segments([0])
        request = SimpleNamespace(
            segment_ids=batch.segment_ids,
            motion_ids=tuple(spec.motion_id for spec in batch.specs),
            start_frames=torch.tensor([int(spec.start_frame) for spec in batch.specs], dtype=torch.long),
            horizon_k=batch.horizon_k,
            valid_mask=torch.ones_like(batch.segment_ids, dtype=torch.bool),
        )
        result = hook(request)
        print(
            "[probe step7] index_reset_hook_install: "
            f"metadata={runner._frontres_segment_dataset.cache_metadata()} "
            f"motion_ids={list(request.motion_ids)} "
            f"start_frames={request.start_frames.tolist()} "
            f"success={result['reset_success'].tolist()} "
            f"command_motion={env.unwrapped.command.env_motion_indices.tolist()} "
            f"command_time={env.unwrapped.command.time_steps.tolist()} "
            f"root_pos={env.unwrapped.robot.data.root_pos_w.tolist()}",
            flush=True,
        )
        assert result["reset_success"].tolist() == [True]
        assert env.unwrapped.command.env_motion_indices.tolist() == [0]
        assert env.unwrapped.command.time_steps.tolist() == [3]
        torch.testing.assert_close(env.unwrapped.robot.data.root_pos_w, torch.tensor([[3.0, 0.0, 1.0]]))


def test_live_sampler_filters_index_dataset_to_loaded_motions_before_sampling() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        amass_root = Path(tmp) / "AMASS_G1NPZ_Final"
        cache_dir = Path(tmp) / "AMASS_G1Segment"
        stage1_hooks_contract._write_fake_amass(amass_root / "KIT" / "359" / "motion_a.npz")
        stage1_hooks_contract._write_fake_amass(amass_root / "KIT" / "1346" / "motion_b.npz")
        segments = [
            FrontRESSegmentIndex(
                segment_id=0,
                motion_rel_path="KIT/359/motion_a.npz",
                motion_num_frames=8,
                fps=30.0,
                start_frame=2,
                horizon_k=4,
            ),
            FrontRESSegmentIndex(
                segment_id=1,
                motion_rel_path="KIT/1346/motion_b.npz",
                motion_num_frames=8,
                fps=30.0,
                start_frame=3,
                horizon_k=4,
            ),
        ]
        indexer_module.write_amass_segment_index(
            cache_dir,
            segments,
            FrontRESAMASSIndexSummary(
                amass_root=str(amass_root),
                motion_count=2,
                segment_count=2,
                horizon_k=4,
                frame_stride=1,
                skipped_short_motions=0,
            ),
        )
        env = stage1_hooks_contract.FakeGymEnv(amass_root)
        env.unwrapped.command.motion_dir_loader.motion_paths = [
            str(amass_root / "KIT" / "359" / "motion_a.npz"),
        ]
        env.unwrapped.command.motion_dir_loader.motion_paths_all = [
            str(amass_root / "KIT" / "359" / "motion_a.npz"),
            str(amass_root / "KIT" / "1346" / "motion_b.npz"),
        ]
        env.unwrapped.command.motion_dir_loader.shard_info = {"selected_motions": 1, "total_motions": 2}
        runner = FakeRunner(cache_dir=str(cache_dir), env=env)

        initialize_frontres_segment_live_sampler(runner)
        metadata = runner._frontres_segment_dataset.cache_metadata()
        batch = runner._frontres_segment_dataset.get_segments([0])
        print(
            "[probe bug-index-reset] filter_loaded_motions: "
            f"dataset_segments={runner._frontres_segment_dataset.num_segments()} "
            f"sampler_segments={runner._frontres_segment_sampler.num_segments} "
            f"index_filter={metadata.get('index_filter')} "
            f"motion_ids={[spec.motion_id for spec in batch.specs]}",
            flush=True,
        )

        assert runner._frontres_segment_dataset.num_segments() == 1
        assert runner._frontres_segment_sampler.num_segments == 1
        assert metadata["index_filter"]["filtered"] is True
        assert metadata["index_filter"]["source_segments"] == 2
        assert metadata["index_filter"]["kept_segments"] == 1
        assert [spec.motion_id for spec in batch.specs] == ["KIT/359/motion_a.npz"]


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


def test_missing_dataset_probe_reports_cache_and_sampler_state() -> None:
    runner = FakeRunner(cache_dir="/tmp/missing_stage1_index")
    runner._frontres_segment_sampler = FrontRESSegmentSampler(3, seed=8)
    sample = runner._frontres_segment_sampler.sample(2)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        batch = live_sampler_module._build_current_segment_batch(runner, sample, update_step=0)
    output = stream.getvalue()
    print(
        "[probe bug4] missing_dataset_probe: "
        f"batch_is_none={batch is None} "
        f"has_cache_dir={'skipped.cache_dir: /tmp/missing_stage1_index' in output} "
        f"has_sampler_segments={'skipped.sampler_segments: 3' in output}",
        flush=True,
    )
    assert batch is None
    assert "skipped.reason: no_dataset" in output
    assert "skipped.cache_dir: /tmp/missing_stage1_index" in output
    assert "skipped.has_dataset: False" in output
    assert "skipped.dataset_has_get_segments: False" in output
    assert "skipped.sampler_segments: 3" in output


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


def test_runner_checkpoint_save_does_not_require_logger_type() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "model_no_logger_type.pt")
        policy = torch.nn.Linear(1, 1)
        runner = SimpleNamespace(
            alg=SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters(), lr=1e-3)),
            current_learning_iteration=5,
            cfg={"is_full_resume": True},
            alg_cfg={"learning_rate": 1e-3},
            policy_cfg={"init_noise_std": 1.0, "noise_std_type": "scalar"},
            empirical_normalization=False,
            training_type="frontres",
            device="cpu",
            _frontres_segment_sampler=FrontRESSegmentSampler(2, seed=5),
        )

        save_runner(runner, path)
        saved = torch.load(path, weights_only=False)
        print(
            "[probe checkpoint_logger_type] "
            f"has_logger_type={hasattr(runner, 'logger_type')} "
            f"saved_iter={saved['iter']} "
            f"has_sampler_state={'frontres_segment_sampler_state_dict' in saved}",
            flush=True,
        )
        assert not hasattr(runner, "logger_type")
        assert saved["iter"] == 5
        assert "frontres_segment_sampler_state_dict" in saved


def test_runner_checkpoint_save_skips_missing_external_writer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "model_no_writer.pt")
        policy = torch.nn.Linear(1, 1)
        runner = SimpleNamespace(
            alg=SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters(), lr=1e-3)),
            current_learning_iteration=6,
            cfg={"logger": "wandb", "is_full_resume": True},
            alg_cfg={"learning_rate": 1e-3},
            policy_cfg={"init_noise_std": 1.0, "noise_std_type": "scalar"},
            empirical_normalization=False,
            training_type="frontres",
            disable_logs=False,
            writer=None,
            device="cpu",
            _frontres_segment_sampler=FrontRESSegmentSampler(2, seed=6),
        )

        save_runner(runner, path)
        saved = torch.load(path, weights_only=False)
        print(
            "[probe checkpoint_writer] "
            f"cfg_logger={runner.cfg['logger']} "
            f"writer_is_none={runner.writer is None} "
            f"saved_iter={saved['iter']}",
            flush=True,
        )
        assert runner.writer is None
        assert saved["iter"] == 6


def main() -> None:
    test_live_summary_becomes_sampler_evidence()
    test_live_sampler_evidence_carries_partial_reset_failure()
    test_live_sampler_evidence_preserves_per_sample_rollout_facts()
    test_large_sampler_probe_uses_summary_not_full_lists()
    test_live_update_loop_samples_and_updates_priority()
    test_live_sampler_summary_exposes_update_probe_boundary()
    test_live_detail_logs_are_rate_limited_by_default_and_verbose()
    test_live_sampler_initializes_dataset_from_stage1_cache_dir()
    test_live_sampler_installs_index_reset_hook_for_index_only_dataset()
    test_live_sampler_filters_index_dataset_to_loaded_motions_before_sampling()
    test_live_sampler_passes_nondefault_shard_cache_size_to_lazy_dataset()
    test_live_sampler_builds_current_batch_before_probe()
    test_live_storage_uses_sampled_segment_ids_and_sources()
    test_missing_dataset_probe_reports_cache_and_sampler_state()
    test_runner_checkpoint_saves_and_restores_sampler_state()
    test_runner_checkpoint_save_does_not_require_logger_type()
    test_runner_checkpoint_save_skips_missing_external_writer()
    print("frontres_segment_live_sampler_contract: ok")


if __name__ == "__main__":
    main()
