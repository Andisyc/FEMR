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
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

rsl_rl_pkg = types.ModuleType("rsl_rl")
rsl_rl_pkg.__path__ = [str(ROOT / "rsl_rl")]
frontres_pkg = types.ModuleType("rsl_rl.frontres")
frontres_pkg.__path__ = [str(ROOT / "rsl_rl" / "frontres")]
runners_pkg = types.ModuleType("rsl_rl.runners")
runners_pkg.__path__ = [str(ROOT / "rsl_rl" / "runners")]
rsl_rl_pkg.frontres = frontres_pkg
rsl_rl_pkg.runners = runners_pkg
sys.modules["rsl_rl"] = rsl_rl_pkg
sys.modules["rsl_rl.frontres"] = frontres_pkg
sys.modules["rsl_rl.runners"] = runners_pkg

probe_stub = types.ModuleType("rsl_rl.runners.frontres_segment_live_probe")
probe_stub.apply_frontres_current_segment_reset = lambda runner: None
probe_stub.read_frontres_live_observations = lambda runner: "fake_obs"
probe_stub.capture_frontres_paired_gain = lambda capture: None
probe_stub.run_frontres_live_rollout_capture = (
    lambda runner, observations, *, rollout_steps, **_kwargs: runner.fake_eval_capture(rollout_steps)
)
sys.modules[probe_stub.__name__] = probe_stub


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
run_frontres_segment_live_training_loop = live_training_module.run_frontres_segment_live_training_loop
live_training_module.read_frontres_live_observations = lambda runner: "fake_obs"


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
    def __init__(self, live_train_enabled: bool = True):
        self.live_train_enabled = live_train_enabled


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
    ):
        self._frontres_segment_replay_boundary = FakeBoundary(
            live_train_enabled=live_train_enabled,
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
    assert "  progress: absolute_iter=1 local=1/1 updates=4/4 runner_learn=True" in output
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


def test_training_process_has_no_embedded_evaluator() -> None:
    source = (ROOT / "rsl_rl" / "runners" / "frontres_segment_live_training.py").read_text(encoding="utf-8")
    for retired in ("periodic_eval", "offline_eval", "sequence_offline_eval"):
        assert retired not in source
    print("[probe evaluation_isolation] training loop has no embedded evaluator", flush=True)


def test_resume_progress_separates_absolute_and_local_iterations() -> None:
    """Resume 日志必须区分 checkpoint 绝对迭代与本次命令局部进度."""

    runner = FakeRunner()
    runner.current_learning_iteration = 221
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        live_training_module._print_live_train_summary(
            runner,
            local_iteration=1,
            num_learning_iterations=1,
            summary=_full_summary(),
        )
    output = buffer.getvalue()
    print(f"[probe resume_progress] {output.splitlines()[3]}", flush=True)
    assert "progress: absolute_iter=221 local=1/1" in output
    assert "iter=221/1" not in output


def _formal_stage_runner(*, iteration: int, num_envs: int, log_dir: str | None = "/tmp/frontres-stage"):
    calls: list[int] = []
    schedule = (
        (8, 2, 200, 500, 1300, "lower-k8", 0.5, "linear-joint-v1", 1300, 2.381),
        (16, 3, 300, 300, 900, "lower-k16", 0.6, "linear-joint-v1", 900, 2.381),
        (32, 4, 400, 300, 625, "lower-k32", 0.7, "linear-joint-v1", 625, 2.381),
    )
    runner = types.SimpleNamespace(
        _frontres_segment_replay_boundary=FakeBoundary(live_train_enabled=True),
        alg=types.SimpleNamespace(
            frontres_formal_transaction_enabled=True,
            frontres_segment_k_curriculum=schedule,
            frontres_segment_max_horizon_k=32,
        ),
        current_learning_iteration=iteration,
        log_dir=log_dir,
        disable_logs=log_dir is None,
        save_interval=100,
        env=types.SimpleNamespace(num_envs=num_envs),
    )

    def run_transaction(*, init_at_random_ep_len: bool):
        calls.append(runner.current_learning_iteration)
        return {"transaction_id": f"tx-{runner.current_learning_iteration}"}

    runner.run_frontres_formal_training_transaction = run_transaction
    return runner, calls


def test_formal_k_stage_boundary_saves_then_requires_new_env_width() -> None:
    saved: list[str] = []
    originals = (
        live_training_module.print_formal_route_audit,
        live_training_module._require_v015_committed_result,
        live_training_module._print_v015_formal_train_summary,
        live_training_module._save_live_checkpoint,
    )
    live_training_module.print_formal_route_audit = lambda *_args, **_kwargs: None
    live_training_module._require_v015_committed_result = lambda _runner, result: result
    live_training_module._print_v015_formal_train_summary = lambda *_args, **_kwargs: None

    def save_checkpoint(_runner, *, checkpoint_path: str, **_kwargs):
        saved.append(checkpoint_path)
        return True

    live_training_module._save_live_checkpoint = save_checkpoint
    try:
        runner, calls = _formal_stage_runner(iteration=1999, num_envs=8)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            run_frontres_segment_live_training_loop(runner, num_learning_iterations=2)
        assert calls == [1999]
        assert runner.current_learning_iteration == 2000
        assert saved == ["/tmp/frontres-stage/model_2000.pt"]
        assert "status=RESTART_REQUIRED" in buffer.getvalue()
        assert "next_k=16 next_m=3 next_num_envs=12" in buffer.getvalue()

        resumed, resumed_calls = _formal_stage_runner(iteration=2000, num_envs=12)
        run_frontres_segment_live_training_loop(resumed, num_learning_iterations=1)
        assert resumed_calls == [2000]
        assert resumed.current_learning_iteration == 2001

        k16_runner, k16_calls = _formal_stage_runner(iteration=3499, num_envs=12)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            run_frontres_segment_live_training_loop(k16_runner, num_learning_iterations=2)
        assert k16_calls == [3499]
        assert k16_runner.current_learning_iteration == 3500
        assert saved[-1] == "/tmp/frontres-stage/model_3500.pt"
        assert "next_k=32 next_m=4 next_num_envs=16" in buffer.getvalue()

        k32_runner, k32_calls = _formal_stage_runner(iteration=3500, num_envs=16)
        run_frontres_segment_live_training_loop(k32_runner, num_learning_iterations=1)
        assert k32_calls == [3500]
        assert k32_runner.current_learning_iteration == 3501
    finally:
        (
            live_training_module.print_formal_route_audit,
            live_training_module._require_v015_committed_result,
            live_training_module._print_v015_formal_train_summary,
            live_training_module._save_live_checkpoint,
        ) = originals


def test_formal_k_stage_handoff_rejects_missing_checkpoint_owner() -> None:
    originals = (
        live_training_module.print_formal_route_audit,
        live_training_module._require_v015_committed_result,
        live_training_module._print_v015_formal_train_summary,
    )
    live_training_module.print_formal_route_audit = lambda *_args, **_kwargs: None
    live_training_module._require_v015_committed_result = lambda _runner, result: result
    live_training_module._print_v015_formal_train_summary = lambda *_args, **_kwargs: None
    try:
        runner, calls = _formal_stage_runner(iteration=3499, num_envs=12, log_dir=None)
        try:
            run_frontres_segment_live_training_loop(runner, num_learning_iterations=2)
        except RuntimeError as exc:
            assert "requires an enabled committed checkpoint owner" in str(exc)
        else:
            raise AssertionError("K-stage handoff without checkpoint owner must fail closed")
        assert calls == [3499]
        assert runner.current_learning_iteration == 3500
    finally:
        (
            live_training_module.print_formal_route_audit,
            live_training_module._require_v015_committed_result,
            live_training_module._print_v015_formal_train_summary,
        ) = originals


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
    test_training_process_has_no_embedded_evaluator()
    test_resume_progress_separates_absolute_and_local_iterations()
    test_formal_k_stage_boundary_saves_then_requires_new_env_width()
    test_formal_k_stage_handoff_rejects_missing_checkpoint_owner()
    print("result: PASS")


if __name__ == "__main__":
    main()
