#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


update_loop_module = _load(
    "frontres_segment_live_update_loop",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_live_update_loop.py",
)
run_frontres_segment_live_update_loop = update_loop_module.run_frontres_segment_live_update_loop


class FakeBoundary:
    def __init__(
        self,
        *,
        live_update_loop_only: bool = True,
        live_train_enabled: bool = False,
        live_update_steps: int = 3,
    ) -> None:
        self.live_update_loop_only = live_update_loop_only
        self.live_train_enabled = live_train_enabled
        self.live_update_steps = live_update_steps


class FakeRunner:
    def __init__(
        self,
        summaries: list[dict],
        *,
        boundary: FakeBoundary | None = None,
        alg_update_steps: int | None = None,
    ) -> None:
        self._frontres_segment_replay_boundary = boundary or FakeBoundary(live_update_steps=len(summaries))
        self.alg = SimpleNamespace(frontres_training_objective="stage3_segment_hrl")
        if alg_update_steps is not None:
            self.alg.frontres_segment_live_update_steps = alg_update_steps
        self.summaries = summaries
        self.probe_init_flags: list[bool] = []

    def run_frontres_segment_live_probe(self, *, init_at_random_ep_len: bool) -> dict:
        self.probe_init_flags.append(init_at_random_ep_len)
        return self.summaries[len(self.probe_init_flags) - 1]


def _summary(
    *,
    ppo_update: bool,
    ppo_valid_count: int,
    reward_mean: float,
    storage_valid_frac: float,
    ppo_total_loss: float,
    ppo_actor_loss: float,
    ppo_value_loss: float,
    ppo_approx_kl: float,
    ppo_clip_frac: float,
    env_reward_mean: float | None = None,
    train_reward_mean: float | None = None,
    gain_style_mean: float | None = None,
    gain_physics_mean: float | None = None,
    gain_repair_cost_mean: float | None = None,
    gain_total_mean: float | None = None,
    gain_total_pos_frac: float | None = None,
    sampler_update_gain_mean: float = 0.0,
    sampler_update_gain_pos_frac: float = 0.0,
    sampler_update_useful_mean: float = 0.0,
    sampler_update_replay_candidate_count: int = 0,
    sampler_update_priority_before_mean: float = 0.0,
    sampler_update_priority_after_mean: float = 0.0,
    motion_mpjpe_repaired: float | None = None,
    motion_mpjpe_noisy: float | None = None,
    trust_region_rejected_count: int = 0,
    trust_region_accepted: int = 1,
    adaptive_lr_before: float = 0.0,
    adaptive_lr_after: float = 0.0,
    adaptive_lr_desired_kl: float = 0.0,
    mosaic_pre_step_lr_before: float = 0.0,
    mosaic_pre_step_lr_after: float = 0.0,
    mosaic_pre_step_lr_kl: float = 0.0,
    ppo_distribution_kl_available: bool = False,
    ppo_old_sigma_min: float | None = None,
    ppo_sigma_min: float | None = None,
    trial_policy_count: int = 0,
    trial_search_count: int = 0,
    ppo_boundary_evidence_rows: int | None = None,
    ppo_boundary_search_evidence_only_rows: int | None = None,
    ppo_boundary_policy_invalid_rows: int = 0,
) -> dict:
    evidence_rows = (
        int(ppo_boundary_evidence_rows)
        if ppo_boundary_evidence_rows is not None
        else int(trial_policy_count) + int(trial_search_count)
    )
    result = {
        "ppo_update": ppo_update,
        "ppo_valid_count": ppo_valid_count,
        "reward_mean": reward_mean,
        "env_reward_mean": reward_mean if env_reward_mean is None else env_reward_mean,
        "train_reward_mean": reward_mean if train_reward_mean is None else train_reward_mean,
        "gain_style_mean": gain_style_mean,
        "gain_physics_mean": gain_physics_mean,
        "gain_repair_cost_mean": gain_repair_cost_mean,
        "gain_total_mean": gain_total_mean,
        "gain_total_pos_frac": gain_total_pos_frac,
        "sampler_update_gain_mean": sampler_update_gain_mean,
        "sampler_update_gain_pos_frac": sampler_update_gain_pos_frac,
        "sampler_update_useful_mean": sampler_update_useful_mean,
        "sampler_update_replay_candidate_count": sampler_update_replay_candidate_count,
        "sampler_update_priority_before_mean": sampler_update_priority_before_mean,
        "sampler_update_priority_after_mean": sampler_update_priority_after_mean,
        "storage_valid_frac": storage_valid_frac,
        "ppo_total_loss": ppo_total_loss,
        "ppo_actor_loss": ppo_actor_loss,
        "ppo_value_loss": ppo_value_loss,
        "ppo_approx_kl": ppo_approx_kl,
        "ppo_clip_frac": ppo_clip_frac,
        "ppo_trust_region_rejected_count": trust_region_rejected_count,
        "ppo_trust_region_accepted": trust_region_accepted,
        "ppo_adaptive_lr_before": adaptive_lr_before,
        "ppo_adaptive_lr_after": adaptive_lr_after,
        "ppo_adaptive_lr_desired_kl": adaptive_lr_desired_kl,
        "ppo_mosaic_pre_step_adaptive_lr_before": mosaic_pre_step_lr_before,
        "ppo_mosaic_pre_step_adaptive_lr_after": mosaic_pre_step_lr_after,
        "ppo_mosaic_pre_step_adaptive_lr_kl_mean": mosaic_pre_step_lr_kl,
        "ppo_distribution_kl_available": ppo_distribution_kl_available,
        "trial_policy_count": trial_policy_count,
        "trial_search_count": trial_search_count,
        "ppo_boundary_evidence_rows": evidence_rows,
        "ppo_boundary_policy_rows": trial_policy_count,
        "ppo_boundary_search_rows": trial_search_count,
        "ppo_boundary_eligible_rows": ppo_valid_count,
        "ppo_boundary_search_evidence_only_rows": (
            trial_search_count
            if ppo_boundary_search_evidence_only_rows is None
            else ppo_boundary_search_evidence_only_rows
        ),
        "ppo_boundary_policy_invalid_rows": ppo_boundary_policy_invalid_rows,
        "ppo_boundary_valid_policy_frac": float(ppo_valid_count / max(1, trial_policy_count)),
        "ppo_boundary_valid_evidence_frac": float(ppo_valid_count / max(1, evidence_rows)),
    }
    if ppo_old_sigma_min is not None:
        result["ppo_old_sigma_min"] = ppo_old_sigma_min
    if ppo_sigma_min is not None:
        result["ppo_sigma_min"] = ppo_sigma_min
    if motion_mpjpe_repaired is not None:
        result["segment/motion_mpjpe_repaired_clean"] = motion_mpjpe_repaired
    if motion_mpjpe_noisy is not None:
        result["segment/motion_mpjpe_noisy_clean"] = motion_mpjpe_noisy
    return result


def test_live_update_loop_aggregates_probe_metrics_and_init_flag() -> None:
    runner = FakeRunner(
        [
            _summary(
                ppo_update=True,
                ppo_valid_count=2,
                reward_mean=1.0,
                storage_valid_frac=0.50,
                ppo_total_loss=10.0,
                ppo_actor_loss=1.0,
                ppo_value_loss=2.0,
                ppo_approx_kl=0.01,
                ppo_clip_frac=0.10,
                trial_policy_count=2,
                trial_search_count=1,
            ),
            _summary(
                ppo_update=False,
                ppo_valid_count=0,
                reward_mean=2.0,
                storage_valid_frac=0.25,
                ppo_total_loss=20.0,
                ppo_actor_loss=3.0,
                ppo_value_loss=4.0,
                ppo_approx_kl=0.02,
                ppo_clip_frac=0.20,
                trial_policy_count=1,
                trial_search_count=1,
                ppo_boundary_policy_invalid_rows=1,
            ),
            _summary(
                ppo_update=True,
                ppo_valid_count=4,
                reward_mean=3.0,
                storage_valid_frac=1.00,
                ppo_total_loss=30.0,
                ppo_actor_loss=5.0,
                ppo_value_loss=6.0,
                ppo_approx_kl=0.03,
                ppo_clip_frac=0.30,
                trial_policy_count=4,
                trial_search_count=2,
            ),
        ]
    )

    result = run_frontres_segment_live_update_loop(runner, init_at_random_ep_len=True, runner_learn=True)

    assert runner.probe_init_flags == [True, False, False]
    assert result["update_steps"] == 3
    assert result["update_count"] == 2
    assert result["ppo_valid_count"] == 6
    assert result["trial_policy_count"] == 7
    assert result["trial_search_count"] == 4
    assert result["ppo_boundary_evidence_rows"] == 11
    assert result["ppo_boundary_search_evidence_only_rows"] == 4
    assert result["ppo_boundary_policy_invalid_rows"] == 1
    assert abs(result["ppo_boundary_valid_policy_frac"] - (6.0 / 7.0)) < 1e-8
    assert abs(result["ppo_boundary_valid_evidence_frac"] - (6.0 / 11.0)) < 1e-8
    assert result["reward_mean"] == 2.0
    assert result["storage_valid_frac"] == (0.50 + 0.25 + 1.00) / 3.0
    assert result["ppo_total_loss_mean"] == 20.0
    assert result["ppo_actor_loss_mean"] == 3.0
    assert result["ppo_value_loss_mean"] == 4.0
    assert abs(result["ppo_approx_kl_mean"] - 0.02) < 1e-8
    assert abs(result["ppo_clip_frac_mean"] - 0.20) < 1e-8


def test_live_update_loop_sigma_summary_ignores_invalid_distribution_steps() -> None:
    runner = FakeRunner(
        [
            _summary(
                ppo_update=True,
                ppo_valid_count=2,
                reward_mean=1.0,
                storage_valid_frac=1.0,
                ppo_total_loss=1.0,
                ppo_actor_loss=1.0,
                ppo_value_loss=1.0,
                ppo_approx_kl=0.01,
                ppo_clip_frac=0.1,
                ppo_distribution_kl_available=True,
                ppo_old_sigma_min=0.01,
                ppo_sigma_min=0.01,
            ),
            _summary(
                ppo_update=False,
                ppo_valid_count=0,
                reward_mean=1.0,
                storage_valid_frac=0.0,
                ppo_total_loss=0.0,
                ppo_actor_loss=0.0,
                ppo_value_loss=0.0,
                ppo_approx_kl=0.0,
                ppo_clip_frac=0.0,
                ppo_distribution_kl_available=False,
                ppo_old_sigma_min=0.0,
                ppo_sigma_min=0.0,
            ),
            _summary(
                ppo_update=True,
                ppo_valid_count=2,
                reward_mean=1.0,
                storage_valid_frac=1.0,
                ppo_total_loss=1.0,
                ppo_actor_loss=1.0,
                ppo_value_loss=1.0,
                ppo_approx_kl=0.01,
                ppo_clip_frac=0.1,
                ppo_distribution_kl_available=True,
                ppo_old_sigma_min=0.02,
                ppo_sigma_min=0.02,
            ),
        ]
    )

    result = run_frontres_segment_live_update_loop(runner, init_at_random_ep_len=False)

    print(
        "[probe update_loop] sigma_summary_filters_invalid_steps: "
        f"old_sigma_min={result['ppo_old_sigma_min']} "
        f"sigma_min={result['ppo_sigma_min']}",
        flush=True,
    )
    assert abs(result["ppo_old_sigma_min"] - 0.01) < 1e-8
    assert abs(result["ppo_sigma_min"] - 0.01) < 1e-8


def test_live_update_loop_aggregates_trust_region_and_motion_quality_metrics() -> None:
    runner = FakeRunner(
        [
            _summary(
                ppo_update=True,
                ppo_valid_count=2,
                reward_mean=1.0,
                storage_valid_frac=0.50,
                ppo_total_loss=1.0,
                ppo_actor_loss=0.1,
                ppo_value_loss=0.2,
                ppo_approx_kl=0.01,
                ppo_clip_frac=0.10,
                motion_mpjpe_repaired=0.04,
                motion_mpjpe_noisy=0.08,
                trust_region_rejected_count=1,
                trust_region_accepted=0,
                adaptive_lr_before=1e-4,
                adaptive_lr_after=1e-6,
                adaptive_lr_desired_kl=0.01,
                mosaic_pre_step_lr_before=1e-4,
                mosaic_pre_step_lr_after=5e-5,
                mosaic_pre_step_lr_kl=0.03,
            ),
            _summary(
                ppo_update=True,
                ppo_valid_count=2,
                reward_mean=1.0,
                storage_valid_frac=0.50,
                ppo_total_loss=1.0,
                ppo_actor_loss=0.1,
                ppo_value_loss=0.2,
                ppo_approx_kl=0.01,
                ppo_clip_frac=0.10,
                motion_mpjpe_repaired=0.06,
                motion_mpjpe_noisy=0.10,
                trust_region_rejected_count=0,
                trust_region_accepted=1,
                adaptive_lr_before=1e-6,
                adaptive_lr_after=1e-6,
                adaptive_lr_desired_kl=0.01,
                mosaic_pre_step_lr_before=5e-5,
                mosaic_pre_step_lr_after=5e-5,
                mosaic_pre_step_lr_kl=0.01,
            ),
        ]
    )

    result = run_frontres_segment_live_update_loop(runner, init_at_random_ep_len=False)
    print(
        "[probe update_loop] trust_motion_aggregate: "
        f"rejected={result['ppo_trust_region_rejected_count_sum']} "
        f"accepted_min={result['ppo_trust_region_accepted_min']} "
        f"mpjpe={result['segment/motion_mpjpe_repaired_clean']:.6f} "
        f"lr_after={result['ppo_adaptive_lr_after_last']:.8f} "
        f"pre_lr_after={result['ppo_mosaic_pre_step_adaptive_lr_after_last']:.8f} "
        f"pre_kl={result['ppo_mosaic_pre_step_adaptive_lr_kl_mean']:.6f}",
        flush=True,
    )

    assert result["ppo_trust_region_rejected_count_sum"] == 1
    assert result["ppo_trust_region_accepted_min"] == 0
    assert abs(result["segment/motion_mpjpe_repaired_clean"] - 0.05) < 1e-8
    assert abs(result["segment/motion_mpjpe_noisy_clean"] - 0.09) < 1e-8
    assert result["ppo_adaptive_lr_before_first"] == 1e-4
    assert result["ppo_adaptive_lr_after_last"] == 1e-6
    assert result["ppo_adaptive_lr_desired_kl_mean"] == 0.01
    assert result["ppo_mosaic_pre_step_adaptive_lr_before_first"] == 1e-4
    assert result["ppo_mosaic_pre_step_adaptive_lr_after_last"] == 5e-5
    assert abs(result["ppo_mosaic_pre_step_adaptive_lr_kl_mean"] - 0.02) < 1e-8


def test_live_update_loop_uses_algorithm_update_steps_override() -> None:
    runner = FakeRunner(
        [
            _summary(
                ppo_update=True,
                ppo_valid_count=1,
                reward_mean=1.0,
                storage_valid_frac=1.0,
                ppo_total_loss=1.0,
                ppo_actor_loss=1.0,
                ppo_value_loss=1.0,
                ppo_approx_kl=0.0,
                ppo_clip_frac=0.0,
            ),
            _summary(
                ppo_update=True,
                ppo_valid_count=1,
                reward_mean=3.0,
                storage_valid_frac=1.0,
                ppo_total_loss=3.0,
                ppo_actor_loss=3.0,
                ppo_value_loss=3.0,
                ppo_approx_kl=0.0,
                ppo_clip_frac=0.0,
            ),
        ],
        boundary=FakeBoundary(live_update_steps=5),
        alg_update_steps=2,
    )

    result = run_frontres_segment_live_update_loop(runner, init_at_random_ep_len=False)

    assert runner.probe_init_flags == [False, False]
    assert result["update_steps"] == 2
    assert result["reward_mean"] == 2.0


def test_live_update_loop_requires_enabled_boundary() -> None:
    runner = FakeRunner(
        [],
        boundary=FakeBoundary(live_update_loop_only=False, live_train_enabled=False, live_update_steps=1),
    )

    try:
        run_frontres_segment_live_update_loop(runner)
    except ValueError as exc:
        assert "frontres_segment_live_update_loop_only=True" in str(exc)
    else:
        raise AssertionError("update loop must reject disabled live runner boundary")


def test_live_update_loop_summary_print_rate_default_and_verbose() -> None:
    summaries = [
        _summary(
            ppo_update=True,
            ppo_valid_count=1,
            reward_mean=1.0,
            storage_valid_frac=1.0,
            ppo_total_loss=1.0,
            ppo_actor_loss=1.0,
            ppo_value_loss=1.0,
            ppo_approx_kl=0.0,
            ppo_clip_frac=0.0,
        )
        for _ in range(12)
    ]
    runner = FakeRunner(summaries, boundary=FakeBoundary(live_update_steps=1))

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        for _ in range(12):
            run_frontres_segment_live_update_loop(runner, init_at_random_ep_len=False)
    output = buffer.getvalue()
    default_count = output.count("[FrontRES Segment Live Update Loop]")
    print(
        "[probe step5] update_loop_log_rate: "
        f"default_count={default_count} "
        f"call_count={runner._frontres_segment_live_update_loop_summary_count}",
        flush=True,
    )

    assert default_count == 4
    assert runner._frontres_segment_live_update_loop_summary_count == 12

    verbose_runner = FakeRunner(summaries[:4], boundary=FakeBoundary(live_update_steps=1))
    verbose_runner.alg.frontres_segment_verbose_probe = True
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        for _ in range(4):
            run_frontres_segment_live_update_loop(verbose_runner, init_at_random_ep_len=False)
    verbose_count = buffer.getvalue().count("[FrontRES Segment Live Update Loop]")
    print(
        "[probe step5] update_loop_log_verbose_rate: "
        f"verbose_count={verbose_count} "
        f"verbose={verbose_runner.alg.frontres_segment_verbose_probe}",
        flush=True,
    )

    assert verbose_count == 4


def test_live_update_loop_log_formats_large_loss_readably() -> None:
    runner = FakeRunner(
        [
            _summary(
                ppo_update=True,
                ppo_valid_count=12000,
                reward_mean=0.015,
                storage_valid_frac=0.967,
                ppo_total_loss=1.5157918219343223e23,
                ppo_actor_loss=1.5157918219343223e23,
                ppo_value_loss=0.00114,
                ppo_approx_kl=-0.004483,
                ppo_clip_frac=0.376726,
                trial_policy_count=16000,
                trial_search_count=8000,
                ppo_boundary_evidence_rows=24000,
                ppo_boundary_search_evidence_only_rows=8000,
                ppo_boundary_policy_invalid_rows=4000,
            )
        ],
        boundary=FakeBoundary(live_update_steps=1),
    )

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        run_frontres_segment_live_update_loop(runner, init_at_random_ep_len=False, runner_learn=True)
    output = buffer.getvalue()
    print(f"[probe readable_log] update_loop_line={output.strip()}", flush=True)

    assert "loss_total=1.516e+23" in output
    assert "trial: policy=16000 search=8000 evidence=24000 ppo_valid=12000" in output
    assert "search_evidence_only=8000 policy_invalid=4000" in output
    assert "valid_policy=75.0% valid_evidence=50.0%" in output
    assert "actor=1.516e+23" in output
    assert "clip=37.7%" in output
    assert "status=BAD_LOSS_EXPLOSION" in output
    assert "151579182193432229576704.000000" not in output


def test_live_update_loop_reports_train_env_and_gain_rewards_separately() -> None:
    runner = FakeRunner(
        [
            _summary(
                ppo_update=True,
                ppo_valid_count=2,
                reward_mean=-0.5,
                env_reward_mean=-0.5,
                train_reward_mean=0.25,
                gain_total_mean=0.10,
                storage_valid_frac=1.0,
                ppo_total_loss=1.0,
                ppo_actor_loss=1.0,
                ppo_value_loss=1.0,
                ppo_approx_kl=0.0,
                ppo_clip_frac=0.0,
            ),
            _summary(
                ppo_update=True,
                ppo_valid_count=2,
                reward_mean=-0.25,
                env_reward_mean=-0.25,
                train_reward_mean=0.75,
                gain_total_mean=0.30,
                storage_valid_frac=1.0,
                ppo_total_loss=1.0,
                ppo_actor_loss=1.0,
                ppo_value_loss=1.0,
                ppo_approx_kl=0.0,
                ppo_clip_frac=0.0,
            ),
        ],
        boundary=FakeBoundary(live_update_steps=2),
    )

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = run_frontres_segment_live_update_loop(runner, init_at_random_ep_len=False)
    output = buffer.getvalue()
    print(
        "[probe step3] update_loop_reward_semantics: "
        f"reward_mean={result['reward_mean']} "
        f"env_reward_mean={result['env_reward_mean']} "
        f"gain_total_mean={result['gain_total_mean']}",
        flush=True,
    )

    assert result["reward_mean"] == 0.5
    assert result["train_reward_mean"] == 0.5
    assert result["env_reward_mean"] == -0.375
    assert abs(result["gain_total_mean"] - 0.20) < 1e-8
    assert "train_reward=0.500000" in output
    assert "env_reward=-0.375000" in output
    assert "gain_total=0.200000" in output


def test_live_update_loop_reports_sampler_evidence_update_metrics() -> None:
    runner = FakeRunner(
        [
            _summary(
                ppo_update=True,
                ppo_valid_count=2,
                reward_mean=0.1,
                storage_valid_frac=1.0,
                ppo_total_loss=1.0,
                ppo_actor_loss=1.0,
                ppo_value_loss=1.0,
                ppo_approx_kl=0.0,
                ppo_clip_frac=0.0,
                sampler_update_gain_mean=0.20,
                sampler_update_gain_pos_frac=0.50,
                sampler_update_useful_mean=0.30,
                sampler_update_replay_candidate_count=3,
                sampler_update_priority_before_mean=0.01,
                sampler_update_priority_after_mean=0.04,
            ),
            _summary(
                ppo_update=True,
                ppo_valid_count=2,
                reward_mean=0.1,
                storage_valid_frac=1.0,
                ppo_total_loss=1.0,
                ppo_actor_loss=1.0,
                ppo_value_loss=1.0,
                ppo_approx_kl=0.0,
                ppo_clip_frac=0.0,
                sampler_update_gain_mean=0.40,
                sampler_update_gain_pos_frac=1.00,
                sampler_update_useful_mean=0.50,
                sampler_update_replay_candidate_count=5,
                sampler_update_priority_before_mean=0.02,
                sampler_update_priority_after_mean=0.08,
            ),
        ],
        boundary=FakeBoundary(live_update_steps=2),
    )

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = run_frontres_segment_live_update_loop(runner, init_at_random_ep_len=False)
    output = buffer.getvalue()
    print(
        "[probe step4] update_loop_sampler_update_metrics: "
        f"gain={result['sampler_update_gain_mean']} "
        f"gain_pos={result['sampler_update_gain_pos_frac']} "
        f"replay_candidates={result['sampler_update_replay_candidate_count']} "
        f"priority_after={result['sampler_update_priority_after_mean']}",
        flush=True,
    )

    assert abs(result["sampler_update_gain_mean"] - 0.30) < 1e-8
    assert abs(result["sampler_update_gain_pos_frac"] - 0.75) < 1e-8
    assert abs(result["sampler_update_useful_mean"] - 0.40) < 1e-8
    assert result["sampler_update_replay_candidate_count"] == 8
    assert abs(result["sampler_update_priority_before_mean"] - 0.015) < 1e-8
    assert abs(result["sampler_update_priority_after_mean"] - 0.06) < 1e-8
    assert "sampler_update:" in output
    assert "gain=0.300000" in output
    assert "gain_pos=75.0%" in output
    assert "replay_candidates=8" in output


if __name__ == "__main__":
    test_live_update_loop_aggregates_probe_metrics_and_init_flag()
    test_live_update_loop_sigma_summary_ignores_invalid_distribution_steps()
    test_live_update_loop_aggregates_trust_region_and_motion_quality_metrics()
    test_live_update_loop_uses_algorithm_update_steps_override()
    test_live_update_loop_requires_enabled_boundary()
    test_live_update_loop_summary_print_rate_default_and_verbose()
    test_live_update_loop_log_formats_large_loss_readably()
    test_live_update_loop_reports_train_env_and_gain_rewards_separately()
    test_live_update_loop_reports_sampler_evidence_update_metrics()
    print("frontres_segment_live_update_loop_contract: ok")
