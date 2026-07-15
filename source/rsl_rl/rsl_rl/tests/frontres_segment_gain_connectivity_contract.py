#!/usr/bin/env python3
"""S2 contract: accepted Gain reaches Segment storage and K-step returns."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _stub(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _load_live_probe():
    class Dummy:
        pass

    sys.modules["rsl_rl"] = types.ModuleType("rsl_rl")
    frontres_pkg = _stub("rsl_rl.frontres")
    _stub("rsl_rl.algorithms", FrontRESUnified=Dummy)
    _stub(
        "rsl_rl.algorithms.frontres_segment_ppo",
        FrontRESSegmentPPOBatch=Dummy,
        FrontRESSegmentPPOConfig=Dummy,
        compute_frontres_segment_ppo_loss=lambda *args, **kwargs: None,
    )
    storage_spec = importlib.util.spec_from_file_location(
        "rsl_rl.frontres.frontres_segment_storage",
        ROOT / "rsl_rl" / "frontres" / "frontres_segment_storage.py",
    )
    assert storage_spec is not None and storage_spec.loader is not None
    storage_module = importlib.util.module_from_spec(storage_spec)
    sys.modules[storage_spec.name] = storage_module
    setattr(frontres_pkg, "frontres_segment_storage", storage_module)
    storage_spec.loader.exec_module(storage_module)
    _stub("rsl_rl.frontres.frontres_segment_reset", FrontRESSegmentResetAdapter=Dummy, FrontRESSegmentResetResult=Dummy, ensure_frontres_segment_live_reset_hook=lambda *args, **kwargs: None)
    _stub("rsl_rl.frontres.training_schedule", resolve_frontres_mode_state=lambda *args, **kwargs: None)
    _stub("rsl_rl.modules", FrontRESActorCritic=Dummy)
    _stub("rsl_rl.runners.frontres_training_setup", configure_frontres_pair_layout=lambda *args, **kwargs: None)
    _stub("rsl_rl.runners.frontres_rollout_step", prepare_frontres_rollout_step=lambda *args, **kwargs: None)

    gain_spec = importlib.util.spec_from_file_location(
        "rsl_rl.frontres.frontres_gain",
        ROOT / "rsl_rl" / "frontres" / "frontres_gain.py",
    )
    assert gain_spec is not None and gain_spec.loader is not None
    gain_module = importlib.util.module_from_spec(gain_spec)
    sys.modules[gain_spec.name] = gain_module
    setattr(frontres_pkg, "frontres_gain", gain_module)
    gain_spec.loader.exec_module(gain_module)

    probe_spec = importlib.util.spec_from_file_location(
        "frontres_segment_gain_connectivity_target",
        ROOT / "rsl_rl" / "runners" / "frontres_segment_live_probe.py",
    )
    assert probe_spec is not None and probe_spec.loader is not None
    probe = importlib.util.module_from_spec(probe_spec)
    sys.modules[probe_spec.name] = probe
    probe_spec.loader.exec_module(probe)
    return probe, gain_module


def test_paired_gain_replaces_old_training_score() -> None:
    probe, gain = _load_live_probe()
    clean = torch.zeros(1, 2, 1, 3)
    repaired = clean.clone()
    noisy = clean.clone()
    repaired[..., 0] = 0.1
    noisy[..., 0] = 0.2
    config = gain.FrontRESSegmentGainConfig(
        mpjpe_scale=1.0,
        velocity_scale=1.0,
        acceleration_scale=1.0,
        repair_weight=0.15,
    )
    capture = probe.FrontRESSegmentLiveRolloutCapture(
        rollout_k=2,
        reward_mean=99.0,
        done_frac=0.0,
        last_obs_shape=(4, 3),
        action_shape=(4, 6),
        env_action_shape=(4, 6),
        transition_obs=torch.zeros(4, 3),
        transition_privileged_obs=torch.zeros(4, 3),
        transition_actions=torch.zeros(4, 6),
        transition_log_probs=torch.zeros(4),
        transition_values=torch.zeros(4),
        transition_means=torch.zeros(4, 6),
        transition_sigmas=torch.ones(4, 6),
        reward_accum=torch.full((4,), 99.0),
        done_any=torch.zeros(4, dtype=torch.bool),
        reward_steps=torch.full((2, 4), 99.0),
        done_steps=torch.zeros(2, 4, dtype=torch.bool),
        horizon_k=torch.full((4,), 2, dtype=torch.long),
        n_train=1,
        n_candidate=0,
        n_base=1,
        n_clean=1,
        survival_steps=torch.full((4,), 2.0),
        motion_clean_body_pos=clean,
        motion_repaired_body_pos=repaired,
        motion_noisy_body_pos=noisy,
        transition_action_steps=torch.zeros(2, 4, 6),
        gain_steps=torch.tensor([[0.025, float("nan"), float("nan"), float("nan")], [0.025, float("nan"), float("nan"), float("nan")]]),
        gain_config=config,
        repair_score_accum=torch.full((4,), -500.0),
        repair_score_steps=torch.full((2, 4), -500.0),
    )
    rewards = probe._segment_storage_rewards(capture, batch_size=4, device=torch.device("cpu"))
    reward_steps = probe._segment_storage_reward_steps(capture, batch_size=4, device=torch.device("cpu"))
    summary = probe._initial_live_probe_summary(capture, storage_write=True, single_update=False)
    assert summary["gain_source"] == "FRS-GAIN-v001"
    assert "score_gain_mean" not in summary
    assert torch.isfinite(torch.tensor(float(summary["gain_total_mean"])))
    canonical_gain = float(summary["gain_total_mean"])
    capture.reward_accum = torch.full((4,), -999.0)
    poisoned_summary = probe._initial_live_probe_summary(capture, storage_write=True, single_update=False)
    assert float(poisoned_summary["gain_total_mean"]) == canonical_gain
    torch.testing.assert_close(rewards[0], torch.tensor(0.05))
    assert reward_steps is not None
    torch.testing.assert_close(reward_steps[:, 0], torch.tensor([0.025, 0.025]))
    assert rewards[0].item() != -500.0
    torch.testing.assert_close(rewards[1:], torch.full((3,), 49.5))


def test_formal_gain_reaches_storage_and_ppo_batch() -> None:
    probe, gain = _load_live_probe()
    clean = torch.zeros(1, 2, 1, 3)
    repaired = clean.clone()
    noisy = clean.clone()
    repaired[..., 0] = 0.1
    noisy[..., 0] = 0.2
    capture = probe.FrontRESSegmentLiveRolloutCapture(
        rollout_k=2,
        reward_mean=99.0,
        done_frac=0.0,
        last_obs_shape=(4, 3),
        action_shape=(4, 6),
        env_action_shape=(4, 6),
        transition_obs=torch.zeros(4, 3),
        transition_privileged_obs=torch.zeros(4, 3),
        transition_actions=torch.zeros(4, 6),
        transition_log_probs=torch.zeros(4),
        transition_values=torch.zeros(4),
        transition_means=torch.zeros(4, 6),
        transition_sigmas=torch.ones(4, 6),
        reward_accum=torch.full((4,), 99.0),
        done_any=torch.zeros(4, dtype=torch.bool),
        reward_steps=torch.full((2, 4), 99.0),
        done_steps=torch.zeros(2, 4, dtype=torch.bool),
        horizon_k=torch.full((4,), 2, dtype=torch.long),
        n_train=1,
        n_candidate=0,
        n_base=1,
        n_clean=1,
        survival_steps=torch.full((4,), 2.0),
        motion_clean_body_pos=clean,
        motion_repaired_body_pos=repaired,
        motion_noisy_body_pos=noisy,
        transition_action_steps=torch.zeros(2, 4, 6),
        gain_steps=torch.tensor(
            [[0.025, float("nan"), float("nan"), float("nan")],
             [0.025, float("nan"), float("nan"), float("nan")]]
        ),
        gain_config=gain.FrontRESSegmentGainConfig(
            mpjpe_scale=1.0,
            velocity_scale=1.0,
            acceleration_scale=1.0,
            repair_weight=0.15,
        ),
        repair_score_accum=torch.full((4,), -500.0),
        repair_score_steps=torch.full((2, 4), -500.0),
    )
    runner = types.SimpleNamespace(
        device=torch.device("cpu"),
        _frontres_segment_live_current_batch=types.SimpleNamespace(
            frontres_segment_trial_role=("policy", "baseline", "baseline", "baseline")
        ),
    )
    storage = probe.build_live_segment_storage(runner, capture)
    batch = storage.full_batch()
    torch.testing.assert_close(batch.returns[0], torch.tensor(0.05))
    torch.testing.assert_close(batch.advantages[0], torch.tensor(0.05))
    assert batch.returns[0].item() != -500.0
    assert batch.valid_mask.tolist() == [True, False, False, False]

    class _PPOBatch:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    ppo_batch = batch.to_ppo_batch(_PPOBatch)
    torch.testing.assert_close(ppo_batch.returns, batch.returns)
    torch.testing.assert_close(ppo_batch.advantages, batch.advantages)
    torch.testing.assert_close(ppo_batch.valid_mask, batch.valid_mask)


def test_fall_keeps_prefall_style_evidence_but_blocks_ppo_row() -> None:
    """A terminal fall must truncate Style in time, not erase its valid prefix."""
    probe, gain = _load_live_probe()
    clean = torch.zeros(1, 3, 1, 3)
    repaired = clean.clone()
    noisy = clean.clone()
    repaired[:, :2, :, 0] = 0.1
    noisy[:, :2, :, 0] = 0.2
    repaired[:, 2, :, 0] = 99.0
    noisy[:, 2, :, 0] = -99.0
    capture = probe.FrontRESSegmentLiveRolloutCapture(
        rollout_k=3,
        reward_mean=0.0,
        done_frac=1.0 / 3.0,
        last_obs_shape=(3, 3),
        action_shape=(3, 6),
        env_action_shape=(3, 6),
        transition_obs=torch.zeros(3, 3),
        transition_privileged_obs=torch.zeros(3, 3),
        transition_actions=torch.zeros(3, 6),
        transition_log_probs=torch.zeros(3),
        transition_values=torch.zeros(3),
        transition_means=torch.zeros(3, 6),
        transition_sigmas=torch.ones(3, 6),
        reward_accum=torch.zeros(3),
        done_any=torch.tensor([True, False, False]),
        reward_steps=torch.zeros(3, 3),
        done_steps=torch.tensor(
            [
                [False, False, False],
                [True, False, False],
                [False, False, False],
            ]
        ),
        horizon_k=torch.full((3,), 3, dtype=torch.long),
        n_train=1,
        n_candidate=0,
        n_base=1,
        n_clean=1,
        survival_steps=torch.tensor([2.0, 3.0, 3.0]),
        motion_clean_body_pos=clean,
        motion_repaired_body_pos=repaired,
        motion_noisy_body_pos=noisy,
        transition_action_steps=torch.zeros(3, 3, 6),
        gain_steps=torch.tensor(
            [
                [0.1, float("nan"), float("nan")],
                [0.1, float("nan"), float("nan")],
                [0.0, float("nan"), float("nan")],
            ]
        ),
        gain_config=gain.FrontRESSegmentGainConfig(
            mpjpe_scale=1.0,
            velocity_scale=1.0,
            acceleration_scale=1.0,
            repair_weight=0.0,
        ),
    )

    result = probe._capture_paired_gain(capture)
    assert result is not None
    assert bool(torch.isfinite(result.style_gain).all())
    torch.testing.assert_close(result.style_mpjpe_gain, torch.tensor([0.1]))

    runner = types.SimpleNamespace(
        device=torch.device("cpu"),
        _frontres_segment_live_current_batch=types.SimpleNamespace(
            frontres_segment_trial_role=("policy", "baseline", "baseline")
        ),
    )
    storage = probe.build_live_segment_storage(runner, capture)
    assert storage.full_batch().valid_mask.tolist() == [False, False, False]


def test_formal_gain_route_rejects_legacy_score_fallback() -> None:
    probe, gain = _load_live_probe()
    capture = probe.FrontRESSegmentLiveRolloutCapture(
        rollout_k=1,
        reward_mean=100.0,
        done_frac=0.0,
        last_obs_shape=(2, 3),
        action_shape=(2, 6),
        env_action_shape=(2, 6),
        transition_obs=torch.zeros(2, 3),
        transition_privileged_obs=torch.zeros(2, 3),
        transition_actions=torch.zeros(2, 6),
        transition_log_probs=torch.zeros(2),
        transition_values=torch.zeros(2),
        transition_means=torch.zeros(2, 6),
        transition_sigmas=torch.ones(2, 6),
        reward_accum=torch.full((2,), 100.0),
        done_any=torch.zeros(2, dtype=torch.bool),
        horizon_k=torch.ones(2, dtype=torch.long),
        n_train=1,
        n_base=1,
        repair_score_accum=torch.tensor([-9.0, -8.0]),
        transition_action_steps=torch.zeros(1, 2, 6),
        gain_config=gain.FrontRESSegmentGainConfig(),
    )
    original_gain_module = probe._gain_module
    probe._gain_module = lambda: None
    try:
        try:
            probe._segment_storage_rewards(capture, batch_size=2, device=torch.device("cpu"))
        except RuntimeError as exc:
            assert "FRS-GAIN" in str(exc)
        else:
            raise AssertionError("formal Gain route silently fell back to legacy repair_score")
    finally:
        probe._gain_module = original_gain_module


def main() -> None:
    test_paired_gain_replaces_old_training_score()
    test_formal_gain_reaches_storage_and_ppo_batch()
    test_fall_keeps_prefall_style_evidence_but_blocks_ppo_row()
    test_formal_gain_route_rejects_legacy_score_fallback()
    print("frontres_segment_gain_connectivity_contract: ok")


if __name__ == "__main__":
    main()
