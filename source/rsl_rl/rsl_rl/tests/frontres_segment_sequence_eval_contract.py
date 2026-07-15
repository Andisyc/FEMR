#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sequence_eval_module = _load(
    "frontres_segment_sequence_eval",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_sequence_eval.py",
)

build_frontres_sequence_eval_plan = sequence_eval_module.build_frontres_sequence_eval_plan
build_frontres_sequence_eval_reset_batch = sequence_eval_module.build_frontres_sequence_eval_reset_batch
segment_ids_for_sequence_eval_item = sequence_eval_module.segment_ids_for_sequence_eval_item

live_training_module = _load(
    "frontres_segment_live_training",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_live_training.py",
)
run_frontres_segment_sequence_offline_eval = live_training_module.run_frontres_segment_sequence_offline_eval
format_sequence_offline_eval_log = live_training_module._format_sequence_offline_eval_log
format_sequence_eval_item_log = live_training_module._format_sequence_eval_item_log
format_sequence_eval_debug_log = live_training_module._format_sequence_eval_debug_log
format_sequence_eval_differential_log = live_training_module._format_sequence_eval_differential_log
offline_eval_summary = live_training_module._offline_eval_summary
sequence_offline_eval_summary = live_training_module._sequence_offline_eval_summary


def _fake_paired_gain(capture):
    sequence_id = float(getattr(capture, "sequence_id", 1))
    total = torch.tensor([sequence_id, sequence_id + 1.0])
    return SimpleNamespace(
        style_gain=total + 0.1,
        physics_gain=total + 0.2,
        repair_cost=torch.full((2,), 0.1),
        gain_total=total,
    )


live_training_module._capture_paired_gain = _fake_paired_gain


@dataclass(frozen=True)
class FakeSpec:
    segment_id: int
    motion_id: str
    start_frame: int
    horizon_k: int


@dataclass(frozen=True)
class FakeBatch:
    specs: tuple[FakeSpec, ...]


def test_sequence_eval_prerolls_from_motion_start() -> None:
    plan = build_frontres_sequence_eval_plan(
        (
            FakeSpec(1, "walk_a", 12, 4),
            FakeSpec(2, "walk_a", 99, 4),
            FakeSpec(3, "walk_b", 0, 5),
            FakeSpec(4, "walk_c", 25, 6),
        ),
        requested_sequences=3,
        available_envs=8,
        eval_rollout_steps=50,
    )
    assert plan.motion_ids == ("walk_a", "walk_b", "walk_c")
    assert plan.sequence_count == 3
    assert plan.chunk_capacity == 2
    assert plan.chunk_count == 2
    assert tuple(item.segment_id for item in plan.items) == (1, 3, 4)
    assert tuple(item.reset_frame for item in plan.items) == (0, 0, 0)
    assert tuple(item.preroll_steps for item in plan.items) == (12, 0, 25)
    assert tuple(item.eval_start_frame for item in plan.items) == (12, 0, 25)
    assert tuple(item.eval_end_frame for item in plan.items) == (62, 50, 75)
    assert tuple(item.segment_horizon_k for item in plan.items) == (4, 5, 6)
    print(
        "[probe step2] sequence_plan_identity "
        f"motion_ids={plan.motion_ids} "
        f"segment_ids={[item.segment_id for item in plan.items]} "
        f"reset_frames={[item.reset_frame for item in plan.items]} "
        f"eval_start_frames={[item.eval_start_frame for item in plan.items]}",
        flush=True,
    )


def test_sequence_count_is_not_env_count() -> None:
    specs = tuple(FakeSpec(i, f"motion_{i}", i + 1, 4) for i in range(10))
    plan = build_frontres_sequence_eval_plan(specs, requested_sequences=10, available_envs=8)
    assert plan.sequence_count == 10
    assert plan.chunk_capacity == 2
    assert plan.chunk_count == 5


def test_sequence_eval_requires_unique_motion_ids() -> None:
    try:
        build_frontres_sequence_eval_plan(
            (
                FakeSpec(1, "same_motion", 10, 4),
                FakeSpec(2, "same_motion", 20, 4),
            ),
            requested_sequences=2,
        )
    except ValueError as exc:
        assert "unique motion ids" in str(exc)
    else:
        raise AssertionError("duplicate-only specs must not satisfy sequence eval")


def test_sequence_eval_can_cap_smoke_preroll_depth() -> None:
    plan = build_frontres_sequence_eval_plan(
        (
            FakeSpec(1, "late_a", 5000, 4),
            FakeSpec(2, "early_b", 100, 4),
            FakeSpec(3, "early_c", 200, 4),
        ),
        requested_sequences=2,
        max_preroll_steps=200,
    )
    assert plan.motion_ids == ("early_b", "early_c")
    assert tuple(item.preroll_steps for item in plan.items) == (100, 200)
    assert plan.max_preroll_steps == 200

    try:
        build_frontres_sequence_eval_plan(
            (FakeSpec(1, "late_a", 5000, 4),),
            requested_sequences=1,
            max_preroll_steps=200,
        )
    except ValueError as exc:
        assert "max_preroll_steps<=200" in str(exc)
    else:
        raise AssertionError("cap should reject all-too-deep sequence eval specs")


def test_sequence_eval_reset_batch_rewrites_start_only() -> None:
    item = build_frontres_sequence_eval_plan(
        (FakeSpec(7, "walk_reset", 31, 4),),
        requested_sequences=1,
    ).items[0]
    batch = SimpleNamespace(specs=(FakeSpec(7, "walk_reset", 31, 4),))
    reset_batch = build_frontres_sequence_eval_reset_batch(batch, item)
    assert reset_batch.specs[0].start_frame == 0
    assert reset_batch.specs[0].segment_id == 7
    assert reset_batch.specs[0].motion_id == "walk_reset"
    assert reset_batch.specs[0].horizon_k == 4
    assert batch.specs[0].start_frame == 31
    assert segment_ids_for_sequence_eval_item(item, env_count=8) == (7,) * 8

    dataclass_batch = FakeBatch(specs=(FakeSpec(7, "walk_reset", 31, 4),))
    object.__setattr__(dataclass_batch, "stage3_index_perturbation_family", ("local_rp",))
    object.__setattr__(dataclass_batch, "stage3_index_perturbation_strength", torch.tensor([1.25]))
    reset_dataclass_batch = build_frontres_sequence_eval_reset_batch(dataclass_batch, item)
    assert reset_dataclass_batch.specs[0].start_frame == 0
    assert reset_dataclass_batch.specs[0].segment_id == 7
    assert reset_dataclass_batch.specs[0].motion_id == "walk_reset"
    assert reset_dataclass_batch.specs[0].horizon_k == 4
    assert reset_dataclass_batch.stage3_index_perturbation_family == ("local_rp",)
    assert float(reset_dataclass_batch.stage3_index_perturbation_strength[0]) == 1.25
    print(
        "[probe step2] reset_spec_identity "
        f"segment_id={reset_batch.specs[0].segment_id} "
        f"motion_id={reset_batch.specs[0].motion_id} "
        f"reset_frame={reset_batch.specs[0].start_frame} "
        f"original_start_frame={batch.specs[0].start_frame} "
        f"horizon_k={reset_batch.specs[0].horizon_k}",
        flush=True,
    )


def test_sequence_eval_log_prints_motion_quality_metrics() -> None:
    log = format_sequence_offline_eval_log(
        {
            "sequence_count": 2.0,
            "requested_sequences": 2.0,
            "env_count": 8.0,
            "episode_length": 50.0,
            "success_rate": 0.5,
            "fall_rate": 0.5,
            "mean_survival_steps": 25.0,
            "gain_source": "FRS-GAIN-v001",
            "gain_style_mean": 0.25,
            "gain_physics_mean": 0.35,
            "gain_repair_cost_mean": 0.05,
            "gain_total_mean": 0.1,
            "gain_total_pos_frac": 1.0,
            "segment/motion_mpjpe_repaired_clean": 0.03,
            "segment/motion_mpjpe_noisy_clean": 0.05,
            "segment/motion_vel_error_repaired_clean": 0.07,
            "segment/motion_acc_error_repaired_clean": 0.09,
            "segment/motion_delta_se_norm": 0.11,
            "motion_ids": ("motion_a", "motion_b"),
            "per_motion": (
                {
                    "motion_id": "motion_a",
                    "sample_count": 1.0,
                    "success_rate": 1.0,
                    "fall_rate": 0.0,
                    "mean_survival_steps": 50.0,
                    "gain_source": "FRS-GAIN-v001",
                    "gain_style_mean": 0.25,
                    "gain_physics_mean": 0.35,
                    "gain_repair_cost_mean": 0.05,
                    "gain_total_mean": 0.1,
                    "gain_total_pos_frac": 1.0,
                    "segment/motion_mpjpe_repaired_clean": 0.03,
                    "segment/motion_mpjpe_noisy_clean": 0.05,
                    "segment/motion_vel_error_repaired_clean": 0.07,
                    "segment/motion_acc_error_repaired_clean": 0.09,
                    "segment/motion_delta_se_norm": 0.11,
                },
            ),
        }
    )
    assert "[FrontRES Segment Sequence Eval / Per Motion]" in log
    assert "mpjpe_repaired=0.030000" in log
    assert "mpjpe_noisy=0.050000" in log
    assert "vel_err=0.070000" in log
    assert "acc_err=0.090000" in log
    assert "delta_se_norm=0.110000" in log

    item_log = format_sequence_eval_item_log(
        1,
        10,
        {
            "motion_id": "motion_a",
            "reset_frame": 0.0,
            "preroll_steps": 12.0,
            "eval_start_frame": 12.0,
            "success_rate": 0.5,
            "fall_rate": 0.5,
            "mean_survival_steps": 25.0,
            "gain_source": "FRS-GAIN-v001",
            "gain_style_mean": 0.25,
            "gain_physics_mean": 0.35,
            "gain_repair_cost_mean": 0.05,
            "gain_total_mean": 0.1,
            "gain_total_pos_frac": 1.0,
            "segment/motion_mpjpe_repaired_clean": 0.03,
            "segment/motion_mpjpe_noisy_clean": 0.05,
            "segment/motion_vel_error_repaired_clean": 0.07,
            "segment/motion_acc_error_repaired_clean": 0.09,
            "segment/motion_delta_se_norm": 0.11,
            "perturbation_family_counts": {"local_rp": 8},
            "perturbation_strength_min": 1.25,
            "perturbation_strength_mean": 1.25,
            "perturbation_strength_max": 1.25,
            "perturbation_local_rp_frac": 1.0,
            "perturbation_non_rp_frac": 0.0,
        },
    )
    assert "[FrontRES Segment Sequence Eval Item]" in item_log
    assert "family_counts={'local_rp': 8}" in item_log
    assert "local_rp_frac=100.0%" in item_log
    assert "non_rp_frac=0.0%" in item_log


class FakeSampler:
    def __init__(self):
        self.calls: list[int] = []

    def sample(self, batch_size: int):
        self.calls.append(int(batch_size))
        return SimpleNamespace(segment_ids=torch.arange(batch_size, dtype=torch.long))


class FakeRunner:
    def __init__(self):
        self.env = SimpleNamespace(
            num_envs=8,
            _frontres_segment_index_reset_adapter=SimpleNamespace(trace=True),
        )
        self.env.unwrapped = self.env
        self.device = torch.device("cpu")
        self._frontres_segment_sampler = FakeSampler()
        self._frontres_segment_live_current_sample = None
        self._frontres_segment_live_current_batch = None
        self._frontres_segment_live_current_reset_request = None
        self._frontres_segment_live_current_reset_result = None
        self._frontres_segment_live_detail_log_enabled = True
        self.events: list[tuple[str, object]] = []
        self.read_obs_count = 0
        self.scoring_capture_count = 0

    def eval_mode(self):
        self.events.append(("eval_mode", None))


class FakeCapture:
    def __init__(self, *, rollout_k: int = 50, sequence_id: int = 0) -> None:
        self.rollout_k = int(rollout_k)
        self.sequence_id = int(sequence_id)
        self.reward_mean = 0.0
        self.done_frac = 0.0
        self.last_obs_shape = (8, 29)
        self.action_shape = (8, 6)
        self.env_action_shape = (8, 12)
        self.transition_obs = None
        self.transition_privileged_obs = None
        action_scale = 0.1 * float(sequence_id)
        self.transition_actions = torch.full((8, 6), action_scale)
        self.transition_log_probs = None
        self.transition_values = None
        self.transition_means = None
        self.transition_sigmas = None
        self.transition_perturbation_rp = None
        repaired = 10.0 + float(sequence_id)
        noisy = 2.0 + 0.5 * float(sequence_id)
        self.reward_accum = torch.tensor([repaired, repaired, 0.0, 0.0, noisy, noisy, 0.0, 0.0])
        self.done_any = torch.zeros(8, dtype=torch.bool)
        if sequence_id == 2:
            self.done_any[0] = True
        self.actor_update_mask = None
        self.n_train = 2
        self.n_candidate = 2
        self.n_base = 2
        self.n_clean = 2
        self.survival_steps = torch.full((8,), 40.0 + 3.0 * float(sequence_id))
        if sequence_id == 2:
            self.survival_steps[0] = 17.0
        clean = torch.zeros(2, self.rollout_k, 1, 3)
        repaired_body = clean.clone()
        noisy_body = clean.clone()
        repaired_body[:, :, :, 0] = 0.05 * float(sequence_id)
        noisy_body[:, :, :, 0] = 0.20 * float(sequence_id)
        self.motion_clean_body_pos = clean
        self.motion_repaired_body_pos = repaired_body
        self.motion_noisy_body_pos = noisy_body


def test_sequence_eval_debug_log_prints_key_runtime_parameters() -> None:
    item = build_frontres_sequence_eval_plan((FakeSpec(5, "debug_motion", 7, 4),), requested_sequences=1).items[0]
    batch = SimpleNamespace(
        segment_ids=torch.tensor([5, 5, 5, 5]),
        specs=(FakeSpec(5, "debug_motion", 7, 4),) * 4,
        perturbation_family=("index_only",) * 4,
        perturbation_strength=torch.zeros(4),
        stage3_index_perturbation_family=("local_rp",) * 4,
        stage3_index_perturbation_strength=torch.full((4,), 1.25),
    )
    batch.stage3_index_perturbation_plan = SimpleNamespace(
        perturbation_family=("local_rp",) * 4,
        perturbation_strength=torch.full((4,), 1.25),
        active_modes=("local_rp",),
        complexity="single",
        mix_mode="fixed",
        mix_diag={"frontier_scale": 1.25},
        progress=0.5,
        seq_idx=17,
    )
    reset_batch = build_frontres_sequence_eval_reset_batch(batch, item)
    capture = FakeCapture(rollout_k=50, sequence_id=2)
    capture.env_actions = torch.full((4, 12), 0.05)
    capture.transition_log_probs = torch.linspace(-0.3, -0.1, 4)
    capture.transition_values = torch.linspace(0.1, 0.4, 4)
    capture.transition_actions = torch.zeros(4, 6)
    capture.transition_actions[:, 3:5] = torch.tensor([[-0.1, 0.2], [-0.3, 0.4], [0.0, 0.0], [0.0, 0.0]])
    capture.transition_means = capture.transition_actions.clone()
    capture.transition_sigmas = torch.full((4, 6), 0.5)
    capture.transition_perturbation_rp = torch.tensor([[0.1, -0.2], [0.3, -0.4], [0.0, 0.0], [0.0, 0.0]])
    capture.transition_supervised_target = torch.zeros(4, 6)
    capture.transition_supervised_target[:, 3:5] = torch.tensor([[-0.1, 0.2], [-0.3, 0.4], [0.0, 0.0], [0.0, 0.0]])
    capture.max_delta_rpy = 0.4
    summary = offline_eval_summary(
        capture,
        sample_count=4,
        motion_ids=("debug_motion", "debug_motion", "debug_motion", "debug_motion"),
    )
    summary.update(
        {
            "motion_id": "debug_motion",
            "reset_frame": 0.0,
            "preroll_steps": 7.0,
            "eval_start_frame": 7.0,
        }
    )
    summary.update(live_training_module._offline_eval_perturbation_summary(reset_batch))
    reset_request = SimpleNamespace(
        segment_ids=batch.segment_ids,
        motion_ids=("debug_motion",) * 4,
        start_frames=torch.zeros(4, dtype=torch.long),
        horizon_k=torch.full((4,), 4),
        perturbation_family=("local_rp",) * 4,
        perturbation_strength=torch.full((4,), 1.25),
        valid_mask=torch.ones(4, dtype=torch.bool),
    )
    reset_result = SimpleNamespace(
        success_mask=torch.ones(4, dtype=torch.bool),
        direct_reset_mask=torch.ones(4, dtype=torch.bool),
        preroll_mask=torch.zeros(4, dtype=torch.bool),
        velocity_mismatch=torch.zeros(4),
    )
    log = format_sequence_eval_debug_log(
        item_index=1,
        sequence_count=1,
        item=item,
        eval_batch=batch,
        reset_batch=reset_batch,
        capture=capture,
        summary=summary,
        scoring_observations=torch.zeros(4, 29),
        reset_request=reset_request,
        reset_result=reset_result,
    )
    for marker in (
        "[FrontRES Segment Sequence Eval Debug]",
        "eval_batch:",
        "reset_batch:",
        "stage3_plan=",
        "reset_request:",
        "reset_result:",
        "capture_shapes:",
        "capture_legacy_reward_accum_raw:",
        "raw_policy_action:",
        "segment_transition_actions:",
        "policy_anti_rp_alignment:",
        "target_vs_anti_sign_agree_frac': 1.0",
        "raw_to_delta_available': True",
        "mean_delta_norm_over_target_norm",
        "transition_supervised_target:",
        "action_anti_sign_agree_frac': 1.0",
        "action_same_as_perturb_frac': 0.0",
        "oracles:",
        "reset_frame0': True",
        "eval_batch_frame': True",
        "rp_only': True",
        "metric_shapes_aligned': True",
        "differential_proxy:",
        "action_distribution_health:",
        "status': 'OK",
        "segment_action_nonzero': True",
        "mpjpe_repaired_minus_noisy",
        "transition_log_probs:",
        "motion_clean_body_pos:",
        "motion_role_errors:",
        "local_rp",
        "summary:",
    ):
        assert marker in log
    print(
        "[probe step10] sequence_eval_debug_log_covers_oracles_and_differential_proxy=True",
        flush=True,
    )


def test_sequence_eval_policy_target_scaling_diagnostic_contract() -> None:
    raw_rp = torch.tensor([[3.0, -3.0], [0.5, -0.5]])
    max_delta_rpy = 0.4
    target_rp = torch.tanh(raw_rp) * max_delta_rpy
    target = torch.zeros(2, 6)
    means = torch.zeros(2, 6)
    actions = torch.zeros(2, 6)
    target[:, 3:5] = target_rp
    means[:, 3:5] = raw_rp
    actions[:, 3:5] = target_rp
    capture = SimpleNamespace(
        n_train=2,
        max_delta_rpy=max_delta_rpy,
        transition_perturbation_rp=-target_rp,
        transition_supervised_target=target,
        transition_means=means,
        transition_actions=actions,
    )

    result = live_training_module._sequence_eval_anti_rp_alignment(capture)

    assert result["available"] is True
    assert result["raw_to_delta_available"] is True
    assert result["target_vs_anti_sign_agree_frac"] == 1.0
    assert result["target_norm_over_anti_norm"] == 1.0
    assert result["mean_delta_norm_over_target_norm"] == 1.0
    assert result["action_norm_over_target_norm"] == 1.0
    assert result["mean_delta_vs_target_sign_agree_frac"] == 1.0
    assert result["action_vs_target_sign_agree_frac"] == 1.0
    assert result["mean_raw_abs_max"] == 3.0
    assert result["mean_raw_saturated_frac_abs_gt_2"] == 0.5
    assert result["mean_delta_rp_head"] == live_training_module._round_list(target_rp.tolist())
    print(
        "[probe step13c] policy_target_scaling_diagnostic "
        "target_vs_anti=True raw_to_delta=True saturation_frac=0.5",
        flush=True,
    )


def test_sequence_eval_differential_log_compares_real_and_zero_policy() -> None:
    real_capture = FakeCapture(rollout_k=50, sequence_id=2)
    zero_capture = FakeCapture(rollout_k=50, sequence_id=1)
    zero_capture.transition_actions = torch.zeros_like(real_capture.transition_actions)
    real_summary = offline_eval_summary(real_capture, sample_count=4, motion_ids=("diff_motion",) * 4)
    zero_summary = offline_eval_summary(zero_capture, sample_count=4, motion_ids=("diff_motion",) * 4)
    real_summary["motion_id"] = "diff_motion"
    zero_summary["motion_id"] = "diff_motion"

    log = format_sequence_eval_differential_log(
        item_index=1,
        sequence_count=1,
        summary=real_summary,
        zero_summary=zero_summary,
        capture=real_capture,
        zero_capture=zero_capture,
    )
    for marker in (
        "[FrontRES Segment Sequence Eval Differential]",
        "real_policy:",
        "zero_policy:",
        "real_minus_zero:",
        "zero_action_is_zero=True",
    ):
        assert marker in log
    print("[probe step11] sequence_eval_differential_log_compares_real_zero=True", flush=True)


def test_sequence_offline_eval_owner_orders_reset_preroll_eval() -> None:
    runner = FakeRunner()

    def fake_build_current_segment_batch(runner_arg, sample, *, update_step: int, print_probe: bool):
        specs = tuple(
            FakeSpec(
                int(segment_id),
                f"motion_{int(segment_id)}",
                int(segment_id) + 3,
                4,
            )
            for segment_id in sample.segment_ids.detach().cpu().tolist()
        )
        n = int(sample.segment_ids.numel())
        return SimpleNamespace(
            segment_ids=sample.segment_ids,
            specs=specs,
            perturbation_family=("index_only",) * n,
            perturbation_strength=torch.zeros(n),
            stage3_index_perturbation_family=("local_rp",) * n,
            stage3_index_perturbation_strength=torch.full((n,), 1.25),
        )

    def fake_apply_current_segment_reset(runner_arg):
        starts = tuple(spec.start_frame for spec in runner_arg._frontres_segment_live_current_batch.specs)
        trace = bool(runner_arg.env._frontres_segment_index_reset_adapter.trace)
        families = tuple(getattr(runner_arg._frontres_segment_live_current_batch, "stage3_index_perturbation_family", ()))
        runner_arg.events.append(("reset", starts, trace, families))

    def fake_read_live_observations(runner_arg):
        runner_arg.read_obs_count += 1
        obs = f"obs_{runner_arg.read_obs_count}"
        runner_arg.events.append(("read_obs", obs))
        return obs

    def fake_run_live_rollout_capture(
        runner_arg,
        observations,
        *,
        rollout_steps: int,
        capture_motion_quality: bool = True,
        zero_segment_action: bool = False,
    ):
        starts = tuple(spec.start_frame for spec in runner_arg._frontres_segment_live_current_batch.specs)
        runner_arg.events.append(
            ("rollout", int(rollout_steps), starts, bool(capture_motion_quality), observations, bool(zero_segment_action))
        )
        if capture_motion_quality:
            runner_arg.scoring_capture_count += 1
            capture = FakeCapture(rollout_k=int(rollout_steps), sequence_id=runner_arg.scoring_capture_count)
            if zero_segment_action:
                capture.transition_actions = torch.zeros_like(capture.transition_actions)
            return capture
        return FakeCapture(rollout_k=int(rollout_steps), sequence_id=0)

    live_training_module._build_current_segment_batch = fake_build_current_segment_batch
    live_training_module.build_frontres_sequence_eval_plan = build_frontres_sequence_eval_plan
    live_training_module.build_frontres_sequence_eval_reset_batch = build_frontres_sequence_eval_reset_batch
    live_training_module.segment_ids_for_sequence_eval_item = segment_ids_for_sequence_eval_item
    live_training_module._apply_current_segment_reset = fake_apply_current_segment_reset
    live_training_module._read_live_observations = fake_read_live_observations
    live_training_module._run_live_rollout_capture = fake_run_live_rollout_capture
    original_plan_builder = build_frontres_sequence_eval_plan
    captured_plan_kwargs = {}

    def fake_plan_builder(*args, **kwargs):
        captured_plan_kwargs.update(kwargs)
        return original_plan_builder(*args, **kwargs)

    live_training_module.build_frontres_sequence_eval_plan = fake_plan_builder

    summary = run_frontres_segment_sequence_offline_eval(
        runner,
        num_eval_sequences=2,
        rollout_steps=50,
        max_preroll_steps=10,
    )
    reset_events = [event for event in runner.events if event[0] == "reset"]
    rollout_events = [event for event in runner.events if event[0] == "rollout"]
    assert len(reset_events) == 4
    assert all(set(event[1]) == {0} for event in reset_events)
    assert all(event[2] is False for event in reset_events)
    assert all(set(event[3]) == {"local_rp"} for event in reset_events)
    assert runner.env._frontres_segment_index_reset_adapter.trace is True
    assert rollout_events[0][1:] == (3, (0, 0, 0, 0, 0, 0, 0, 0), False, "obs_1", False)
    assert rollout_events[1][1:] == (50, (3, 3, 3, 3, 3, 3, 3, 3), True, "obs_2", False)
    assert rollout_events[2][1:] == (3, (0, 0, 0, 0, 0, 0, 0, 0), False, "obs_3", False)
    assert rollout_events[3][1:] == (50, (3, 3, 3, 3, 3, 3, 3, 3), True, "obs_4", True)
    assert rollout_events[4][1:] == (4, (0, 0, 0, 0, 0, 0, 0, 0), False, "obs_5", False)
    assert rollout_events[5][1:] == (50, (4, 4, 4, 4, 4, 4, 4, 4), True, "obs_6", False)
    assert rollout_events[6][1:] == (4, (0, 0, 0, 0, 0, 0, 0, 0), False, "obs_7", False)
    assert rollout_events[7][1:] == (50, (4, 4, 4, 4, 4, 4, 4, 4), True, "obs_8", True)
    event_names = [event[0] for event in runner.events]
    assert event_names == [
        "reset",
        "read_obs",
        "eval_mode",
        "rollout",
        "read_obs",
        "rollout",
        "reset",
        "read_obs",
        "eval_mode",
        "rollout",
        "read_obs",
        "rollout",
        "reset",
        "read_obs",
        "eval_mode",
        "rollout",
        "read_obs",
        "rollout",
        "reset",
        "read_obs",
        "eval_mode",
        "rollout",
        "read_obs",
        "rollout",
    ]
    assert int(summary["sequence_count"]) == 2
    assert summary["motion_ids"] == ("motion_0", "motion_1")
    assert captured_plan_kwargs["max_preroll_steps"] == 10
    per_motion = {row["motion_id"]: row for row in summary["per_motion"]}
    assert per_motion["motion_0"]["gain_total_mean"] != per_motion["motion_1"]["gain_total_mean"]
    assert per_motion["motion_0"]["mean_survival_steps"] != per_motion["motion_1"]["mean_survival_steps"]
    assert per_motion["motion_0"]["segment/motion_delta_se_norm"] != per_motion["motion_1"][
        "segment/motion_delta_se_norm"
    ]
    print(
        "[probe step2] sequence_owner_identity "
        f"motion_ids={summary['motion_ids']} "
        f"reset_starts={[event[1] for event in reset_events]} "
        f"eval_starts={[event[2] for event in rollout_events if event[3] is True]}",
        flush=True,
    )
    print(
        "[probe step3] reset_preroll_scoring_order "
        f"events={event_names} "
        f"preroll_capture={[event[3] for event in rollout_events[0::2]]} "
        f"scoring_capture={[event[3] for event in rollout_events[1::2]]} "
        f"scoring_obs={[event[4] for event in rollout_events[1::2]]}",
        flush=True,
    )
    print(
        "[probe step5] rollout_state_isolation "
        f"motion_0_gain={per_motion['motion_0']['gain_total_mean']:.6f} "
        f"motion_1_gain={per_motion['motion_1']['gain_total_mean']:.6f} "
        f"motion_0_survival={per_motion['motion_0']['mean_survival_steps']:.1f} "
        f"motion_1_survival={per_motion['motion_1']['mean_survival_steps']:.1f} "
        f"motion_0_delta={per_motion['motion_0']['segment/motion_delta_se_norm']:.6f} "
        f"motion_1_delta={per_motion['motion_1']['segment/motion_delta_se_norm']:.6f}",
        flush=True,
    )


def test_sequence_eval_all_fall_keeps_action_diagnostics_in_summaries() -> None:
    capture = FakeCapture(rollout_k=50, sequence_id=3)
    capture.done_any = torch.ones(8, dtype=torch.bool)
    capture.survival_steps = torch.zeros(8)
    summary = offline_eval_summary(capture, sample_count=2, motion_ids=("fall_a", "fall_b"))
    expected_delta = float(torch.linalg.norm(capture.transition_actions.float(), dim=-1).mean().item())
    assert summary["fall_rate"] == 1.0
    assert abs(summary["segment/motion_delta_se_norm"] - expected_delta) < 1e-6
    per_motion_delta = [float(row["segment/motion_delta_se_norm"]) for row in summary["per_motion"]]
    assert per_motion_delta and all(delta > 0.0 for delta in per_motion_delta)

    merged = sequence_offline_eval_summary(
        [summary],
        plan=SimpleNamespace(requested_sequences=1, motion_ids=("fall_a", "fall_b")),
        env_count=8,
    )
    assert abs(merged["segment/motion_delta_se_norm"] - expected_delta) < 1e-6
    log = format_sequence_offline_eval_log(merged)
    assert f"delta_se_norm={expected_delta:.6f}" in log
    print(
        "[probe step6] sequence_aggregation_all_fall_action_visible "
        f"item_delta={summary['segment/motion_delta_se_norm']:.6f} "
        f"per_motion_min={min(per_motion_delta):.6f} "
        f"final_delta={merged['segment/motion_delta_se_norm']:.6f}",
        flush=True,
    )


def test_sequence_eval_per_motion_uses_item_scope_for_repeated_motion_roles() -> None:
    capture = FakeCapture(rollout_k=50, sequence_id=4)
    capture.n_train = 1
    capture.n_candidate = 1
    capture.n_base = 1
    capture.n_clean = 1
    capture.reward_accum = torch.tensor([8.0, 0.0, 2.0, 0.0])
    capture.done_any = torch.tensor([False, True, True, True])
    capture.survival_steps = torch.tensor([50.0, 10.0, 20.0, 30.0])
    capture.transition_actions = torch.full((4, 6), 0.2)
    clean = torch.zeros(4, 50, 1, 3)
    repaired = clean.clone()
    noisy = clean.clone()
    repaired[:, :, :, 0] = 0.1
    noisy[:, :, :, 0] = 0.3
    capture.motion_clean_body_pos = clean
    capture.motion_repaired_body_pos = repaired
    capture.motion_noisy_body_pos = noisy

    summary = offline_eval_summary(
        capture,
        sample_count=4,
        motion_ids=("same_motion", "same_motion", "same_motion", "same_motion"),
    )
    row = summary["per_motion"][0]
    assert row["motion_id"] == "same_motion"
    assert row["sample_count"] == 4.0
    assert row["gain_source"] == "FRS-GAIN-v001"
    assert row["gain_total_mean"] == 4.5
    for key in (
        "success_rate",
        "fall_rate",
        "mean_survival_steps",
        "gain_total_mean",
        "segment/motion_delta_se_norm",
    ):
        assert abs(float(row[key]) - float(summary[key])) < 1e-6
    merged = sequence_offline_eval_summary(
        [summary],
        plan=SimpleNamespace(requested_sequences=1, motion_ids=("same_motion",)),
        env_count=4,
    )
    assert abs(float(merged["success_rate"]) - float(row["success_rate"])) < 1e-6
    assert abs(float(merged["segment/motion_delta_se_norm"]) - float(row["segment/motion_delta_se_norm"])) < 1e-6
    print(
        "[probe step8] per_motion_scope_matches_item "
        f"success={summary['success_rate']:.3f} "
        f"per_motion_success={row['success_rate']:.3f} "
        f"delta={summary['segment/motion_delta_se_norm']:.6f} "
        f"per_motion_delta={row['segment/motion_delta_se_norm']:.6f}",
        flush=True,
    )


def main() -> None:
    test_sequence_eval_prerolls_from_motion_start()
    test_sequence_count_is_not_env_count()
    test_sequence_eval_requires_unique_motion_ids()
    test_sequence_eval_can_cap_smoke_preroll_depth()
    test_sequence_eval_reset_batch_rewrites_start_only()
    test_sequence_eval_log_prints_motion_quality_metrics()
    test_sequence_eval_debug_log_prints_key_runtime_parameters()
    test_sequence_eval_policy_target_scaling_diagnostic_contract()
    test_sequence_eval_differential_log_compares_real_and_zero_policy()
    test_sequence_offline_eval_owner_orders_reset_preroll_eval()
    test_sequence_eval_all_fall_keeps_action_diagnostics_in_summaries()
    test_sequence_eval_per_motion_uses_item_scope_for_repeated_motion_roles()
    print(
        "[probe step23] sequence_eval_contract "
        "unique_motion_ids=True reset_frame=0 eval_start_frame=segment_start "
        "segment_id_preserved=True requested_sequences_not_env_count=True",
        flush=True,
    )
    print(
        "[probe step24] sequence_eval_live_owner "
        "reset_before_preroll=True preroll_no_capture=True obs_refresh_before_eval=True preroll_before_eval=True "
        "eval_capture=True reset_trace_silenced=True role_envs_repeated=True "
        "fresh_sequence_capture=True max_preroll_routed=True motion_metrics_printed=True perturbation_rp_preserved=True",
        flush=True,
    )
    print("frontres_segment_sequence_eval_contract: ok")


if __name__ == "__main__":
    main()
