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
live_training_module._apply_current_segment_reset = lambda runner: None
live_training_module._read_live_observations = lambda runner: "fake_obs"
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


def _full_summary(**overrides) -> dict:
    summary = {
        "update_steps": 4,
        "update_count": 4,
        "ppo_valid_count": 8,
        "reward_mean": 0.25,
        "train_reward_mean": 0.25,
        "env_reward_mean": -0.50,
        "score_noisy_mean": -0.10,
        "score_repaired_mean": 0.65,
        "score_gain_mean": 0.75,
        "score_gain_pos_frac": 0.80,
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
    }
    summary.update(overrides)
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
            "continuous_rollout_gain": float(train_summary["score_gain_mean"]),
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
        return capture


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
    print(f"[probe readable_log] live_train_block={lines[max(0, header_idx - 3):header_idx + 6]}", flush=True)

    assert "[FrontRES Segment Live Train]" in output
    assert lines[header_idx - 2] == "-" * 80
    assert lines[header_idx - 1] == ""
    assert lines[header_idx + 1].startswith("  progress:")
    assert lines[header_idx + 5].startswith("  trust:")
    assert lines[header_idx + 6] == ""
    assert lines[header_idx + 7] != "-" * 80
    assert "  progress: iter=1/1 updates=4/4 runner_learn=True" in output
    assert "  data: valid=8 valid_frac=100.0% train_reward=0.250000 env_reward=-0.500000 gain=0.750000" in output
    assert "  sampler: gain=0.300000 gain_pos=60.0% useful=0.400000 replay_candidates=5 priority=0.070000 pool=11 hopeless=20.0%" in output
    assert "  ppo: loss_total=1.516e+23" in output
    assert "  trust: accepted=1 rejected=0" in output
    assert "[FrontRES Segment Train Effect]" in output
    assert "  score: noisy=-0.100000 repaired=0.650000 gain=0.750000 gain_pos=80.0%" in output
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
    assert "gain=0.750000" in output


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


def test_pseudo_offline_eval_summary_scores_repaired_against_noisy_baseline() -> None:
    runner = FakeRunner()
    capture = runner.fake_eval_capture(rollout_steps=5)
    summary = offline_eval_summary(capture, sample_count=2, motion_ids=("motion_a.npz", "motion_b.npz"))
    print(
        "[probe offline_eval_summary] "
        f"sample_count={summary['sample_count']} "
        f"episode_length={summary['episode_length']} "
        f"success_rate={summary['success_rate']} "
        f"fall_rate={summary['fall_rate']} "
        f"survival={summary['mean_survival_steps']} "
        f"noisy={summary['score_noisy']} "
        f"repaired={summary['score_repaired']} "
        f"gain={summary['continuous_rollout_gain']}",
        flush=True,
    )
    assert summary["sample_count"] == 2.0
    assert summary["episode_length"] == 5.0
    assert round(summary["success_rate"], 6) == round(2.0 / 3.0, 6)
    assert round(summary["fall_rate"], 6) == round(1.0 / 3.0, 6)
    assert round(summary["score_repaired"], 6) == 2.1
    assert round(summary["score_noisy"], 6) == 0.5
    assert round(summary["continuous_rollout_gain"], 6) == 1.6
    log = format_offline_eval_log(summary)
    assert "[FrontRES Segment Offline Eval / Per Motion]" in log
    assert "[FrontRES Segment Offline Eval / Mean]" in log
    assert "id=motion_a.npz samples=1" in log
    assert "id=motion_b.npz samples=1" in log
    assert "sample_count=2" in log
    assert "success=66.7%" in log
    assert "fall=33.3%" in log
    assert "noisy=0.500000" in log
    assert "repaired=2.100000" in log
    assert "gain=1.600000" in log
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
    test_pseudo_live_training_periodic_eval_requires_hook()
    test_pseudo_offline_eval_summary_scores_repaired_against_noisy_baseline()
    test_pseudo_offline_eval_capture_exposes_motion_quality_tensors()
    print("result: PASS")


if __name__ == "__main__":
    main()
