#!/usr/bin/env python3
"""Pseudo-parameter contract for Stage 3 Segment Replay live training.

This test uses fake runner arguments instead of IsaacLab so interface mistakes
are caught before the first real server run.
"""
from __future__ import annotations

import contextlib
import io
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


live_training_module = _load(
    "frontres_segment_live_training",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_live_training.py",
)
diagnostics_module = _load(
    "frontres_segment_diagnostics",
    ROOT / "rsl_rl" / "frontres" / "frontres_segment_diagnostics.py",
)

run_frontres_segment_live_training_loop = live_training_module.run_frontres_segment_live_training_loop
run_frontres_segment_periodic_eval = live_training_module.run_frontres_segment_periodic_eval
live_training_module._apply_current_segment_reset = lambda runner: None
live_training_module._read_live_observations = lambda runner: "fake_obs"
live_training_module._capture_paired_gain = lambda capture: None
live_training_module._run_live_rollout_capture = lambda runner, observations, *, rollout_steps: runner.fake_eval_capture(
    rollout_steps
)
offline_eval_summary = live_training_module._offline_eval_summary
format_offline_eval_log = live_training_module._format_offline_eval_log
motion_quality_summary_to_scalars = diagnostics_module.motion_quality_summary_to_scalars


def _probe_summary(name: str, summary: dict) -> None:
    print(
        f"[probe step5] {name}: "
        f"update_steps={summary.get('update_steps')} "
        f"update_count={summary.get('update_count')} "
        f"ppo_valid_count={summary.get('ppo_valid_count')} "
        f"reward_mean={summary.get('reward_mean')} "
        f"storage_valid_frac={summary.get('storage_valid_frac')} "
        f"ppo_total_loss_mean={summary.get('ppo_total_loss_mean')} "
        f"ppo_actor_loss_mean={summary.get('ppo_actor_loss_mean')} "
        f"ppo_value_loss_mean={summary.get('ppo_value_loss_mean')} "
        f"ppo_approx_kl_mean={summary.get('ppo_approx_kl_mean')} "
        f"ppo_clip_frac_mean={summary.get('ppo_clip_frac_mean')}",
        flush=True,
    )


def _probe_runner_state(name: str, runner: "FakeRunner") -> None:
    print(
        f"[probe step5] {name}: "
        f"current_learning_iteration={runner.current_learning_iteration} "
        f"update_calls={runner.update_calls} "
        f"saved_paths={runner.saved_paths} "
        f"probe_record_count={len(runner.probe_records)} "
        f"probe_checkpoint_paths={[path for _, path in runner.probe_records]}",
        flush=True,
    )


def _probe_exception(name: str, exc: Exception) -> None:
    print(f"[probe step5] {name}: exception={type(exc).__name__} message={exc}", flush=True)


class FakeBoundary:
    def __init__(self, live_train_enabled: bool = True, periodic_eval_enabled: bool = False, periodic_eval_interval: int = 100):
        self.live_train_enabled = live_train_enabled
        self.periodic_eval_enabled = periodic_eval_enabled
        self.periodic_eval_interval = periodic_eval_interval


class FakeAlg:
    def __init__(
        self,
        *,
        fail_on_invalid_update: bool = True,
        min_valid_count: int = 1,
        fail_on_nonfinite: bool = True,
    ):
        self.frontres_segment_live_fail_on_invalid_update = fail_on_invalid_update
        self.frontres_segment_live_min_valid_count = min_valid_count
        self.frontres_segment_live_fail_on_nonfinite = fail_on_nonfinite


class FakeEnv:
    max_episode_length = 32


def _full_summary(**overrides) -> dict:
    summary = {
        "update_steps": 4,
        "update_count": 4,
        "ppo_valid_count": 8,
        "reward_mean": 0.25,
        "train_reward_mean": 0.25,
        "env_reward_mean": -0.50,
        "gain_style_mean": 0.20,
        "gain_physics_mean": 0.40,
        "gain_repair_cost_mean": 0.15,
        "gain_total_mean": 0.75,
        "gain_total_pos_frac": 0.80,
        "done_frac": 0.10,
        "motion_delta_se_norm": 0.42,
        "motion_delta_z_up_frac": 0.25,
        "sampler_update_gain_mean": 0.30,
        "sampler_update_gain_pos_frac": 0.60,
        "sampler_update_useful_mean": 0.40,
        "sampler_update_replay_candidate_count": 5,
        "sampler_priority_mean": 0.07,
        "sampler_replay_pool_size": 11,
        "sampler_hopeless_frac": 0.20,
        "storage_valid_frac": 1.0,
        "ppo_total_loss_mean": 0.5,
        "ppo_actor_loss_mean": 0.1,
        "ppo_value_loss_mean": 0.2,
        "ppo_approx_kl_mean": 0.01,
        "ppo_clip_frac_mean": 0.0,
        "ppo_trust_region_schedule": "adaptive",
        "ppo_trust_region_rollback_enabled_min": 1,
        "ppo_trust_region_max_retries_max": 2,
    }
    summary.update(overrides)
    policy_count = int(summary.get("trial_policy_count", summary["ppo_valid_count"]))
    search_count = int(summary.get("trial_search_count", 0))
    evidence_rows = int(summary.get("ppo_boundary_evidence_rows", policy_count + search_count))
    summary.setdefault("trial_policy_count", policy_count)
    summary.setdefault("trial_search_count", search_count)
    summary.setdefault("ppo_boundary_evidence_rows", evidence_rows)
    summary.setdefault("ppo_boundary_policy_rows", policy_count)
    summary.setdefault("ppo_boundary_search_rows", search_count)
    summary.setdefault("ppo_boundary_eligible_rows", int(summary["ppo_valid_count"]))
    summary.setdefault("ppo_boundary_search_evidence_only_rows", search_count)
    summary.setdefault("ppo_boundary_policy_invalid_rows", max(0, policy_count - int(summary["ppo_valid_count"])))
    summary.setdefault("ppo_boundary_valid_policy_frac", float(int(summary["ppo_valid_count"]) / max(1, policy_count)))
    summary.setdefault("ppo_boundary_valid_evidence_frac", float(int(summary["ppo_valid_count"]) / max(1, evidence_rows)))
    return summary


class FakeRunner:
    def __init__(
        self,
        *,
        log_dir: str | None = "/tmp/frontres-pseudo",
        live_train_enabled: bool = True,
        fail_on_invalid_update: bool = True,
        min_valid_count: int = 1,
        fail_on_nonfinite: bool = True,
        fail_save_paths: set[str] | None = None,
        periodic_eval_enabled: bool = False,
        periodic_eval_interval: int = 100,
    ):
        self._frontres_segment_replay_boundary = FakeBoundary(
            live_train_enabled=live_train_enabled,
            periodic_eval_enabled=periodic_eval_enabled,
            periodic_eval_interval=periodic_eval_interval,
        )
        self.alg = FakeAlg(
            fail_on_invalid_update=fail_on_invalid_update,
            min_valid_count=min_valid_count,
            fail_on_nonfinite=fail_on_nonfinite,
        )
        self.current_learning_iteration = 0
        self.log_dir = log_dir
        self.disable_logs = False
        self.save_interval = 1
        self.update_calls: list[tuple[bool, bool]] = []
        self.saved_paths: list[str] = []
        self.probe_records: list[tuple[dict, str]] = []
        self.fail_save_paths = fail_save_paths or set()
        self.eval_calls: list[tuple[int, int]] = []
        self.periodic_eval_hook_enabled = True
        self.env = FakeEnv()
        self.eval_mode_called = False
        self.train_mode_calls = 0

    def run_frontres_segment_live_update_loop(self, *, init_at_random_ep_len: bool, runner_learn: bool) -> dict:
        self.update_calls.append((init_at_random_ep_len, runner_learn))
        call_id = len(self.update_calls)
        summary = _full_summary(
            ppo_valid_count=8 * call_id,
            reward_mean=0.25 * call_id,
            ppo_total_loss_mean=0.5 * call_id,
            ppo_actor_loss_mean=0.1 * call_id,
            ppo_value_loss_mean=0.2 * call_id,
            ppo_approx_kl_mean=0.01 * call_id,
        )
        _probe_summary(f"update_loop_summary_{call_id}", summary)
        return summary

    def save(self, path: str) -> None:
        if path in self.fail_save_paths:
            raise RuntimeError(f"synthetic checkpoint write failure: {path}")
        self.saved_paths.append(path)

    def _record_frontres_checkpoint_probe(self, locs: dict, checkpoint_path: str) -> None:
        self.probe_records.append((locs, checkpoint_path))

    def run_frontres_segment_periodic_eval(self, *, iteration: int, train_summary: dict) -> dict:
        self.eval_calls.append((iteration, int(train_summary["update_count"])))
        return {
            "episode_length": 32,
            "success_rate": 0.75,
            "fall_rate": 0.25,
            "mean_survival_steps": 28,
            "gain_source": "FRS-GAIN-v001",
            "gain_style_mean": 0.20,
            "gain_physics_mean": 0.40,
            "gain_repair_cost_mean": 0.15,
            "gain_total_mean": float(train_summary["gain_total_mean"]),
            "gain_total_pos_frac": 0.80,
        }

    def fake_eval_capture(self, rollout_steps: int):
        class Capture:
            pass

        capture = Capture()
        capture.done_any = __import__("torch").tensor([False, True, False])
        capture.survival_steps = __import__("torch").tensor([float(rollout_steps), 3.0, float(rollout_steps)])
        capture.rollout_k = int(rollout_steps)
        capture.reward_accum = __import__("torch").tensor([10.0, 11.0, 2.0, 3.0])
        clean = __import__("torch").zeros(2, int(rollout_steps), 1, 3)
        repaired = clean.clone()
        noisy = clean.clone()
        repaired[:, :, :, 0] = 0.1
        noisy[:, :, :, 0] = 0.4
        repaired[:, 1:, :, 1] = 0.05
        noisy[:, 1:, :, 1] = 0.2
        capture.motion_clean_body_pos = clean
        capture.motion_repaired_body_pos = repaired
        capture.motion_noisy_body_pos = noisy
        capture.n_train = 2
        capture.n_candidate = 0
        capture.n_base = 2
        capture.n_clean = 2
        capture.transition_actions = __import__("torch").tensor(
            [
                [0.0, 0.0, 0.0, 0.10, 0.20, 0.0],
                [0.0, 0.0, 0.0, 0.20, 0.10, 0.0],
            ]
        )
        capture.actor_update_mask = __import__("torch").tensor([True, True])
        return capture

    def eval_mode(self) -> None:
        self.eval_mode_called = True

    def train_mode(self) -> None:
        self.train_mode_calls += 1


def test_pseudo_live_training_runs_two_iterations_and_saves_checkpoints() -> None:
    runner = FakeRunner()
    run_frontres_segment_live_training_loop(
        runner,
        num_learning_iterations=2,
        init_at_random_ep_len=True,
    )
    _probe_runner_state("after_two_iteration_training", runner)
    assert runner.current_learning_iteration == 2
    assert runner.update_calls == [(True, True), (False, True)]
    assert runner.saved_paths[0].endswith("model_1.pt")
    assert runner.saved_paths[1].endswith("model_2.pt")
    assert runner.saved_paths[-1].endswith("model_2.pt")
    assert len(runner.probe_records) == 2
    assert runner.probe_records[0][0]["update_steps"] == 4
    assert runner.probe_records[1][0]["ppo_valid_count"] == 16


def test_pseudo_live_training_zero_iterations_does_not_touch_update_loop() -> None:
    runner = FakeRunner()
    run_frontres_segment_live_training_loop(
        runner,
        num_learning_iterations=0,
        init_at_random_ep_len=True,
    )
    _probe_runner_state("after_zero_iteration_training", runner)
    assert runner.current_learning_iteration == 0
    assert runner.update_calls == []
    assert runner.saved_paths == []
    assert runner.probe_records == []


def test_pseudo_live_training_requires_train_flag() -> None:
    runner = FakeRunner(live_train_enabled=False)
    try:
        run_frontres_segment_live_training_loop(
            runner,
            num_learning_iterations=1,
            init_at_random_ep_len=True,
        )
    except ValueError as exc:
        _probe_exception("requires_train_flag", exc)
        assert "frontres_segment_live_train_enabled=True" in str(exc)
    else:
        raise AssertionError("live training must reject fake runners without the train flag")


def test_pseudo_live_training_rejects_incomplete_summary() -> None:
    runner = FakeRunner()

    def bad_update_loop(*, init_at_random_ep_len: bool, runner_learn: bool) -> dict:
        runner.update_calls.append((init_at_random_ep_len, runner_learn))
        summary = {"update_steps": 4}
        _probe_summary("incomplete_summary", summary)
        return summary

    runner.run_frontres_segment_live_update_loop = bad_update_loop
    try:
        run_frontres_segment_live_training_loop(
            runner,
            num_learning_iterations=1,
            init_at_random_ep_len=True,
        )
    except KeyError as exc:
        _probe_exception("rejects_incomplete_summary", exc)
        assert "missing keys" in str(exc)
        assert "ppo_valid_count" in str(exc)
    else:
        raise AssertionError("live training must reject incomplete update summaries")


def test_pseudo_live_training_rejects_nonfinite_summary() -> None:
    runner = FakeRunner()

    def bad_update_loop(*, init_at_random_ep_len: bool, runner_learn: bool) -> dict:
        runner.update_calls.append((init_at_random_ep_len, runner_learn))
        summary = _full_summary(ppo_total_loss_mean=float("nan"))
        _probe_summary("nonfinite_summary", summary)
        return summary

    runner.run_frontres_segment_live_update_loop = bad_update_loop
    try:
        run_frontres_segment_live_training_loop(
            runner,
            num_learning_iterations=1,
            init_at_random_ep_len=True,
        )
    except FloatingPointError as exc:
        _probe_exception("rejects_nonfinite_summary", exc)
        assert "non-finite" in str(exc)
        assert "ppo_total_loss_mean" in str(exc)
    else:
        raise AssertionError("live training must reject non-finite update summaries")


def test_pseudo_live_training_rejects_zero_update_count() -> None:
    runner = FakeRunner()

    def bad_update_loop(*, init_at_random_ep_len: bool, runner_learn: bool) -> dict:
        runner.update_calls.append((init_at_random_ep_len, runner_learn))
        summary = _full_summary(update_count=0, ppo_valid_count=8)
        _probe_summary("zero_update_count_summary", summary)
        return summary

    runner.run_frontres_segment_live_update_loop = bad_update_loop
    try:
        run_frontres_segment_live_training_loop(
            runner,
            num_learning_iterations=1,
            init_at_random_ep_len=True,
        )
    except RuntimeError as exc:
        _probe_exception("rejects_zero_update_count", exc)
        assert "update_count=0" in str(exc)
    else:
        raise AssertionError("live training must reject empty update iterations")


def test_pseudo_live_training_rejects_too_few_valid_samples() -> None:
    runner = FakeRunner(min_valid_count=2)

    def bad_update_loop(*, init_at_random_ep_len: bool, runner_learn: bool) -> dict:
        runner.update_calls.append((init_at_random_ep_len, runner_learn))
        summary = _full_summary(update_count=1, ppo_valid_count=1)
        _probe_summary("too_few_valid_summary", summary)
        return summary

    runner.run_frontres_segment_live_update_loop = bad_update_loop
    try:
        run_frontres_segment_live_training_loop(
            runner,
            num_learning_iterations=1,
            init_at_random_ep_len=True,
        )
    except RuntimeError as exc:
        _probe_exception("rejects_too_few_valid_samples", exc)
        assert "too few valid PPO samples" in str(exc)
        assert "ppo_valid_count=1" in str(exc)
    else:
        raise AssertionError("live training must reject iterations with too few valid PPO samples")


def test_pseudo_live_training_can_disable_fail_fast_guards() -> None:
    runner = FakeRunner(
        fail_on_invalid_update=False,
        fail_on_nonfinite=False,
    )

    def unchecked_update_loop(*, init_at_random_ep_len: bool, runner_learn: bool) -> dict:
        runner.update_calls.append((init_at_random_ep_len, runner_learn))
        summary = _full_summary(update_count=0, ppo_valid_count=0, ppo_total_loss_mean=float("nan"))
        _probe_summary("unchecked_invalid_summary", summary)
        return summary

    runner.run_frontres_segment_live_update_loop = unchecked_update_loop
    run_frontres_segment_live_training_loop(
        runner,
        num_learning_iterations=1,
        init_at_random_ep_len=True,
    )
    _probe_runner_state("after_disabled_fail_fast_training", runner)
    assert runner.current_learning_iteration == 1
    assert runner.update_calls == [(True, True)]


def test_pseudo_live_training_log_formats_large_loss_readably() -> None:
    runner = FakeRunner(fail_on_nonfinite=True)

    def large_loss_update_loop(*, init_at_random_ep_len: bool, runner_learn: bool) -> dict:
        runner.update_calls.append((init_at_random_ep_len, runner_learn))
        return _full_summary(
            ppo_total_loss_mean=1.5157918219343223e23,
            ppo_actor_loss_mean=1.5157918219343223e23,
            ppo_value_loss_mean=0.00114,
            ppo_approx_kl_mean=-0.004483,
            ppo_clip_frac_mean=0.376726,
            trial_policy_count=12,
            trial_search_count=4,
            ppo_boundary_evidence_rows=16,
            ppo_boundary_eligible_rows=8,
            ppo_boundary_search_evidence_only_rows=4,
            ppo_boundary_policy_invalid_rows=4,
            ppo_boundary_valid_policy_frac=8.0 / 12.0,
            ppo_boundary_valid_evidence_frac=0.5,
        )

    runner.run_frontres_segment_live_update_loop = large_loss_update_loop
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        run_frontres_segment_live_training_loop(
            runner,
            num_learning_iterations=1,
            init_at_random_ep_len=False,
        )
    output = buffer.getvalue()
    lines = output.splitlines()
    header_idx = lines.index("[FrontRES Segment Live Train]")
    print(f"[probe readable_log] live_train_block={lines[max(0, header_idx - 3):header_idx + 7]}", flush=True)

    assert "[FrontRES Segment Live Train]" in output
    assert lines[header_idx - 2] == "-" * 80
    assert lines[header_idx - 1] == ""
    assert lines[header_idx + 1].startswith("  progress:")
    assert lines[header_idx + 6].startswith("  trust:")
    assert lines[header_idx + 7].startswith("  scale:")
    assert lines[header_idx + 8] == ""
    assert lines[header_idx + 9] != "-" * 80
    assert "  progress: iter=1/1 updates=4/4 runner_learn=True" in output
    assert "  data: valid=8 valid_frac=100.0% train_reward=0.250000 env_reward=-0.500000 gain_total=0.750000" in output
    assert "  trial: policy=12 search=4 evidence=16 ppo_valid=8 search_evidence_only=4 policy_invalid=4 valid_policy=66.7% valid_evidence=50.0%" in output
    assert "  sampler: gain=0.300000 gain_pos=60.0% useful=0.400000 replay_candidates=5 priority=0.070000 pool=11 hopeless=20.0%" in output
    assert "actor_weight=1.000000 loss_total=1.516e+23" in output
    assert "  trust: accepted=1 rejected=0" in output
    assert "schedule=adaptive rollback=True max_retries=2" in output
    assert "  scale: adv_top1=" in output
    assert "[FrontRES Segment Train Effect]" in output
    assert "  gain: style=0.200000 physics=0.400000 repair_cost=0.150000 total=0.750000 gain_pos=80.0%" in output
    assert "[FrontRES Segment Motion Quality]" in output
    assert "  action: delta_se_norm=0.420000 dz_up=25.0%" in output
    assert "loss_total=1.516e+23" in output
    assert "actor=1.516e+23" in output
    assert "clip=37.7%" in output
    assert "status=BAD_LOSS_EXPLOSION" in output
    assert "151579182193432229576704.000000" not in output


def test_pseudo_live_training_continues_after_periodic_checkpoint_failure() -> None:
    runner = FakeRunner(fail_save_paths={"/tmp/frontres-pseudo/model_1.pt"})
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        run_frontres_segment_live_training_loop(
            runner,
            num_learning_iterations=2,
            init_at_random_ep_len=False,
        )
    output = buffer.getvalue()
    print(
        "[probe checkpoint_failure] "
        f"saved_paths={runner.saved_paths} "
        f"probe_record_count={len(runner.probe_records)} "
        f"has_failed={'save.status: FAILED' in output}",
        flush=True,
    )
    assert runner.current_learning_iteration == 2
    assert runner.saved_paths == ["/tmp/frontres-pseudo/model_2.pt"]
    assert len(runner.probe_records) == 1
    assert "save.status: FAILED" in output
    assert "synthetic checkpoint write failure" in output
    assert "save.path: /tmp/frontres-pseudo/model_1.pt" in output
    assert "save.status: OK" in output


def test_pseudo_live_training_runs_periodic_eval_only_on_interval() -> None:
    runner = FakeRunner(periodic_eval_enabled=True, periodic_eval_interval=2)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        run_frontres_segment_live_training_loop(
            runner,
            num_learning_iterations=3,
            init_at_random_ep_len=False,
        )
    output = buffer.getvalue()
    print(f"[probe periodic_eval] eval_calls={runner.eval_calls}", flush=True)
    assert runner.eval_calls == [(2, 4)]
    assert output.count("[FrontRES Segment Periodic Eval]") == 1
    assert "episode_length=32.0" in output
    assert "success=75.0%" in output
    assert "total=0.750000" in output


def test_periodic_eval_marks_missing_gain_and_preserves_motion_metrics_when_all_samples_fall() -> None:
    torch = __import__("torch")
    runner = FakeRunner()
    runner._frontres_segment_sampler = SimpleNamespace(
        seen=torch.tensor([False, False]),
        staleness=torch.tensor([0.0, 0.0]),
        generator=torch.Generator().manual_seed(3),
    )
    capture = runner.fake_eval_capture(32)
    capture.done_any = __import__("torch").tensor([True, True, True])
    capture.survival_steps = __import__("torch").tensor([12.0, 8.0, 10.0])
    runner.fake_eval_capture = lambda rollout_steps: capture
    batch = SimpleNamespace(specs=(SimpleNamespace(motion_id="a", start_frame=1), SimpleNamespace(motion_id="b", start_frame=2)))
    old_sample_rows = getattr(live_training_module, "_sample_live_segment_rows", None)
    old_batch_builder = getattr(live_training_module, "_build_current_segment_batch", None)
    live_training_module._sample_live_segment_rows = lambda *_args: SimpleNamespace(segment_ids=torch.tensor([0, 1]))
    live_training_module._build_current_segment_batch = lambda *_args, **_kwargs: batch
    try:
        summary = run_frontres_segment_periodic_eval(
            runner,
            iteration=100,
            train_summary={"gain_total_mean": 0.0},
        )
    finally:
        live_training_module._sample_live_segment_rows = old_sample_rows
        live_training_module._build_current_segment_batch = old_batch_builder
    log = diagnostics_module.format_segment_periodic_eval_log(summary)
    print(f"[probe periodic_eval_metrics] {log.replace(chr(10), ' | ')}", flush=True)
    assert summary["success_rate"] == 0.0
    assert summary["fall_rate"] == 1.0
    assert "source=UNCONFIRMED" in log
    assert "total=UNCONFIRMED" in log
    assert "score:" not in log
    assert "mpjpe_repaired=0.111435" in log
    assert "mpjpe_noisy=0.445738" in log
    assert "delta_se_norm=0.223607" in log


def test_periodic_eval_uses_independent_batch_and_restores_training_state() -> None:
    torch = __import__("torch")
    runner = FakeRunner()
    runner.env.num_envs = 8
    runner.cfg = {"frontres_candidate_rollout_enabled": True}
    runner._frontres_segment_live_current_sample = "training_sample"
    runner._frontres_segment_live_current_batch = "training_batch"
    runner._frontres_segment_live_current_reset_request = "training_request"
    runner._frontres_segment_live_current_reset_result = "training_result"

    class FakeSampler:
        def __init__(self):
            self.seen = torch.tensor([False, True])
            self.staleness = torch.tensor([3.0, 7.0])
            self.generator = torch.Generator().manual_seed(17)

    sampler = FakeSampler()
    runner._frontres_segment_sampler = sampler
    seen_before = sampler.seen.clone()
    staleness_before = sampler.staleness.clone()
    rng_before = sampler.generator.get_state().clone()
    sample = SimpleNamespace(segment_ids=torch.tensor([0, 1]))
    batch = SimpleNamespace(
        specs=(
            SimpleNamespace(motion_id="motion_a.npz", start_frame=12),
            SimpleNamespace(motion_id="motion_b.npz", start_frame=24),
        ),
        stage3_index_perturbation_family=("local_rp", "local_rp"),
        stage3_index_perturbation_strength=torch.tensor([0.15, 0.25]),
    )
    calls: list[str] = []

    def fake_sample_rows(_runner, _sampler):
        calls.append("sample")
        _sampler.seen[:] = True
        _sampler.staleness += 10.0
        torch.rand((), generator=_sampler.generator)
        return sample

    def fake_build(_runner, actual_sample, *, update_step, print_probe):
        calls.append("build")
        assert actual_sample is sample
        assert update_step == 0
        assert print_probe is False
        return batch

    def fake_reset(actual_runner):
        calls.append("reset")
        assert actual_runner._frontres_segment_live_current_sample is sample
        assert actual_runner._frontres_segment_live_current_batch is batch
        actual_runner._frontres_segment_live_current_reset_result = SimpleNamespace(ok=True)

    capture = runner.fake_eval_capture(32)
    capture.reward_accum = torch.tensor([10.0, 11.0, 100.0, 101.0, 2.0, 3.0])
    capture.n_candidate = 2
    old_sample_rows = getattr(live_training_module, "_sample_live_segment_rows", None)
    old_batch_builder = getattr(live_training_module, "_build_current_segment_batch", None)
    old_reset = live_training_module._apply_current_segment_reset
    old_capture = live_training_module._run_live_rollout_capture
    old_gain = live_training_module._capture_paired_gain
    live_training_module._sample_live_segment_rows = fake_sample_rows
    live_training_module._build_current_segment_batch = fake_build
    live_training_module._apply_current_segment_reset = fake_reset
    live_training_module._run_live_rollout_capture = lambda *_args, **_kwargs: capture
    live_training_module._capture_paired_gain = lambda _capture: SimpleNamespace(
        style_gain=torch.tensor([1.0, 3.0]),
        physics_gain=torch.tensor([2.0, 4.0]),
        repair_cost=torch.tensor([0.5, 1.0]),
        gain_total=torch.tensor([2.5, 6.0]),
    )
    try:
        summary = run_frontres_segment_periodic_eval(
            runner,
            iteration=100,
            train_summary={"gain_total_mean": 999.0},
        )
    finally:
        live_training_module._sample_live_segment_rows = old_sample_rows
        live_training_module._build_current_segment_batch = old_batch_builder
        live_training_module._apply_current_segment_reset = old_reset
        live_training_module._run_live_rollout_capture = old_capture
        live_training_module._capture_paired_gain = old_gain

    log = diagnostics_module.format_segment_periodic_eval_log(summary)
    print(f"[probe periodic_eval_independent] {log.replace(chr(10), ' | ')}", flush=True)
    assert calls == ["sample", "build", "reset"]
    assert runner.eval_mode_called is True
    assert runner.train_mode_calls == 1
    assert summary["eval_batch_source"] == "independent_sampler"
    assert summary["eval_reset_applied"] is True
    assert summary["gain_source"] == "FRS-GAIN-v001"
    assert summary["gain_style_mean"] == 2.0
    assert summary["gain_physics_mean"] == 3.0
    assert summary["gain_repair_cost_mean"] == 0.75
    assert summary["gain_total_mean"] == 4.25
    assert summary["gain_total_pos_frac"] == 1.0
    assert summary["motion_ids"] == ("motion_a.npz", "motion_b.npz")
    assert summary["start_frames"] == (12, 24)
    assert summary["perturbation_family_counts"] == {"local_rp": 2}
    assert summary["perturbation_strength_min"] == float(torch.tensor(0.15))
    assert summary["perturbation_strength_max"] == 0.25
    assert runner._frontres_segment_live_current_sample == "training_sample"
    assert runner._frontres_segment_live_current_batch == "training_batch"
    assert runner._frontres_segment_live_current_reset_request == "training_request"
    assert runner._frontres_segment_live_current_reset_result == "training_result"
    assert torch.equal(sampler.seen, seen_before)
    assert torch.equal(sampler.staleness, staleness_before)
    assert torch.equal(sampler.generator.get_state(), rng_before)
    assert "batch: source=independent_sampler" in log
    assert "reset=True" in log
    assert "motion_ids=('motion_a.npz', 'motion_b.npz')" in log
    assert "start_frames=(12, 24)" in log
    assert "families={'local_rp': 2}" in log
    assert "total=4.250000" in log
    assert "score:" not in log


def test_periodic_eval_restores_train_mode_when_rollout_raises() -> None:
    torch = __import__("torch")
    runner = FakeRunner()
    runner._frontres_segment_sampler = SimpleNamespace(
        seen=torch.tensor([False]),
        staleness=torch.tensor([0.0]),
        generator=torch.Generator().manual_seed(5),
    )
    sample = SimpleNamespace(segment_ids=torch.tensor([0]))
    batch = SimpleNamespace(specs=(SimpleNamespace(motion_id="motion_a.npz", start_frame=12),))
    old_sample_rows = getattr(live_training_module, "_sample_live_segment_rows", None)
    old_batch_builder = getattr(live_training_module, "_build_current_segment_batch", None)
    old_capture = live_training_module._run_live_rollout_capture
    live_training_module._sample_live_segment_rows = lambda *_args: sample
    live_training_module._build_current_segment_batch = lambda *_args, **_kwargs: batch
    live_training_module._run_live_rollout_capture = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("synthetic periodic rollout failure")
    )
    try:
        try:
            run_frontres_segment_periodic_eval(runner, iteration=100, train_summary={})
        except RuntimeError as exc:
            assert "synthetic periodic rollout failure" in str(exc)
        else:
            raise AssertionError("periodic eval must propagate rollout failures")
    finally:
        live_training_module._sample_live_segment_rows = old_sample_rows
        live_training_module._build_current_segment_batch = old_batch_builder
        live_training_module._run_live_rollout_capture = old_capture
    assert runner.train_mode_calls == 1
    assert not hasattr(runner, "_frontres_segment_live_current_sample")
    assert not hasattr(runner, "_frontres_segment_live_current_batch")


def test_pseudo_live_training_periodic_eval_requires_hook() -> None:
    runner = FakeRunner(periodic_eval_enabled=True, periodic_eval_interval=1)
    runner.run_frontres_segment_periodic_eval = None
    try:
        run_frontres_segment_live_training_loop(
            runner,
            num_learning_iterations=1,
            init_at_random_ep_len=False,
        )
    except NotImplementedError as exc:
        _probe_exception("periodic_eval_requires_hook", exc)
        assert "run_frontres_segment_periodic_eval" in str(exc)
    else:
        raise AssertionError("periodic eval must fail loudly when the live eval hook is missing")


def test_pseudo_offline_eval_summary_uses_canonical_gain() -> None:
    runner = FakeRunner()
    capture = runner.fake_eval_capture(rollout_steps=5)
    torch = __import__("torch")
    old_gain = live_training_module._capture_paired_gain
    live_training_module._capture_paired_gain = lambda _capture: SimpleNamespace(
        style_gain=torch.tensor([1.0, 3.0]),
        physics_gain=torch.tensor([2.0, 4.0]),
        repair_cost=torch.tensor([0.5, 1.0]),
        gain_total=torch.tensor([2.5, 6.0]),
    )
    try:
        summary = offline_eval_summary(capture, sample_count=2, motion_ids=("motion_a.npz", "motion_b.npz"))
    finally:
        live_training_module._capture_paired_gain = old_gain
    print(
        "[probe offline_eval_summary] "
        f"sample_count={summary['sample_count']} "
        f"episode_length={summary['episode_length']} "
        f"success_rate={summary['success_rate']} "
        f"fall_rate={summary['fall_rate']} "
        f"survival={summary['mean_survival_steps']} "
        f"gain_source={summary['gain_source']} "
        f"style={summary['gain_style_mean']} "
        f"physics={summary['gain_physics_mean']} "
        f"repair_cost={summary['gain_repair_cost_mean']} "
        f"gain_total={summary['gain_total_mean']}",
        flush=True,
    )
    assert summary["sample_count"] == 2.0
    assert summary["episode_length"] == 5.0
    assert round(summary["success_rate"], 6) == round(2.0 / 3.0, 6)
    assert round(summary["fall_rate"], 6) == round(1.0 / 3.0, 6)
    assert summary["gain_source"] == "FRS-GAIN-v001"
    assert round(summary["gain_style_mean"], 6) == 2.0
    assert round(summary["gain_physics_mean"], 6) == 3.0
    assert round(summary["gain_repair_cost_mean"], 6) == 0.75
    assert round(summary["gain_total_mean"], 6) == 4.25
    log = format_offline_eval_log(summary)
    assert "[FrontRES Segment Offline Eval / Per Motion]" in log
    assert "gain: source=FRS-GAIN-v001" in log
    assert "score:" not in log
    assert "[FrontRES Segment Offline Eval / Mean]" in log
    assert "id=motion_a.npz samples=1" in log
    assert "id=motion_b.npz samples=1" in log
    assert "sample_count=2" in log
    assert "success=66.7%" in log
    assert "fall=33.3%" in log
    assert "style=2.000000" in log
    assert "physics=3.000000" in log
    assert "repair_cost=0.750000" in log
    assert "total=4.250000" in log
    assert "mpjpe_repaired=" in log
    assert "mpjpe_noisy=" in log
    assert "vel_err=" in log
    assert "acc_err=" in log
    assert "delta_se_norm=" in log


def test_pseudo_offline_eval_capture_exposes_motion_quality_tensors() -> None:
    runner = FakeRunner()
    capture = runner.fake_eval_capture(rollout_steps=5)
    scalars = motion_quality_summary_to_scalars(
        clean_positions=capture.motion_clean_body_pos,
        repaired_positions=capture.motion_repaired_body_pos,
        noisy_positions=capture.motion_noisy_body_pos,
    )
    print(
        "[probe offline_eval_motion_quality] "
        f"clean_shape={tuple(capture.motion_clean_body_pos.shape)} "
        f"repaired_mpjpe={scalars['segment/motion_mpjpe_repaired_clean']} "
        f"noisy_mpjpe={scalars['segment/motion_mpjpe_noisy_clean']} "
        f"vel={scalars['segment/motion_vel_error_repaired_clean']} "
        f"acc={scalars['segment/motion_acc_error_repaired_clean']}",
        flush=True,
    )
    assert tuple(capture.motion_clean_body_pos.shape) == (2, 5, 1, 3)
    assert scalars["segment/motion_mpjpe_repaired_clean"] > 0.0
    assert scalars["segment/motion_mpjpe_noisy_clean"] > scalars["segment/motion_mpjpe_repaired_clean"]
    assert scalars["segment/motion_vel_error_repaired_clean"] > 0.0
    assert scalars["segment/motion_acc_error_repaired_clean"] > 0.0


def main() -> None:
    test_pseudo_live_training_runs_two_iterations_and_saves_checkpoints()
    test_pseudo_live_training_zero_iterations_does_not_touch_update_loop()
    test_pseudo_live_training_requires_train_flag()
    test_pseudo_live_training_rejects_incomplete_summary()
    test_pseudo_live_training_rejects_nonfinite_summary()
    test_pseudo_live_training_rejects_zero_update_count()
    test_pseudo_live_training_rejects_too_few_valid_samples()
    test_pseudo_live_training_can_disable_fail_fast_guards()
    test_pseudo_live_training_log_formats_large_loss_readably()
    test_pseudo_live_training_continues_after_periodic_checkpoint_failure()
    test_pseudo_live_training_runs_periodic_eval_only_on_interval()
    test_periodic_eval_marks_missing_gain_and_preserves_motion_metrics_when_all_samples_fall()
    test_periodic_eval_uses_independent_batch_and_restores_training_state()
    test_periodic_eval_restores_train_mode_when_rollout_raises()
    test_pseudo_live_training_periodic_eval_requires_hook()
    test_pseudo_offline_eval_summary_uses_canonical_gain()
    test_pseudo_offline_eval_capture_exposes_motion_quality_tensors()
    print("result: PASS")


if __name__ == "__main__":
    main()
