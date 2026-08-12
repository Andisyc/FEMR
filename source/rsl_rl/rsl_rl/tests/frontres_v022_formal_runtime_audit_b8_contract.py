#!/usr/bin/env python3
"""Current B8/M4 shape contract for the v022 formal runtime audit."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from frontres_contract_imports import install_frontres_contract_packages


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"
install_frontres_contract_packages(RSL_ROOT)
AUDIT_PATH = RSL_ROOT / "runners" / "frontres_formal_runtime_audit.py"
spec = importlib.util.spec_from_file_location("frontres_v022_formal_runtime_audit_module", AUDIT_PATH)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit)


def _runner() -> SimpleNamespace:
    gmt_policy = nn.Linear(3, 3)
    gmt_policy.eval()
    for parameter in gmt_policy.parameters():
        parameter.requires_grad = False
    policy = SimpleNamespace(gmt_policy=gmt_policy)
    alg = SimpleNamespace(
        frontres_formal_runtime_audit=True,
        frontres_critic_value_kind="state_value",
        frontres_critic_input_dim=449,
        frontres_critic_support_context_id="action-pre-support-plan-kmax32-v1",
        frontres_critic_action_conditioned=False,
        frontres_critic_target_id="scenario-compatible-robust-mean-symlog-v1",
        policy=policy,
    )
    return SimpleNamespace(
        alg=alg,
        _frontres_v015_one_action_k_phase="frozen",
    )


def _install_observation_trace(runner: SimpleNamespace) -> None:
    from rsl_rl.runners.frontres_segment_runtime_types import (
        bind_frontres_collection_context,
        open_frontres_checkpoint_transaction_barrier,
        update_frontres_observation_trace,
    )

    open_frontres_checkpoint_transaction_barrier(runner)
    bind_frontres_collection_context(runner, route="training", sample=object(), batch=object())
    update_frontres_observation_trace(
        runner,
        role_row_count=64,
        current_command_dim=58,
        raw_observation_dim=870,
        q29_tail_dim=58,
        combined_observation_dim=928,
        normalized_observation_dim=928,
        femr_visible_dim=158,
        gmt_suffix_dim=770,
        gmt_input_dim=770,
        critic_current_observation_dim=289,
        critic_future_intent_dim=58,
        critic_support_context_dim=102,
        critic_observation_dim=449,
        actor_segment_state_max_abs_diff=0.0,
        critic_segment_state_max_abs_diff=0.0,
        actor_raw_observation_max_abs_diff=0.0,
        critic_raw_observation_max_abs_diff=0.0,
        post_advance_gmt_read_count=8,
    )


def _telemetry() -> dict[str, object]:
    source_index = tuple(source for source in range(8) for _ in range(4))
    trial_index = tuple(trial for _source in range(8) for trial in range(4))
    gains = tuple(0.1 * (source + 1) + 0.01 * trial for source, trial in zip(source_index, trial_index))
    utilities = tuple(math.log1p(value) for value in gains)
    segment_targets = tuple(
        float(torch.tensor(utilities[4 * source : 4 * (source + 1)], dtype=torch.float32).mean().item())
        for source in range(8)
    )
    critic_targets = tuple(segment_targets[source] for source in source_index)
    gains_tensor = torch.tensor(gains, dtype=torch.float32)
    return {
        "transaction_id": "tx-v022-b8-audit",
        "source_index": source_index,
        "trial_index": trial_index,
        "scenario_ids": tuple(f"scenario-{source}" for source in source_index),
        "noisy_segment_hashes": tuple(f"hash-{source}" for source in source_index),
        "policy_row_count": 32,
        "active_k": 8,
        "active_m": 4,
        "selected_segment_count": 8,
        "role_row_count": 64,
        "valid_policy_row_mask": (True,) * 32,
        "clean_execution_count": (1,) * 8,
        "noisy_execution_count": (1,) * 8,
        "intent_remaining_noisy": (0.4,) * 32,
        "intent_remaining_repaired": (0.2,) * 32,
        "physics_remaining_noisy": (0.5,) * 32,
        "physics_remaining_repaired": (0.4,) * 32,
        "intent_gain": gains,
        "physics_gain": gains,
        "recovery_pressure": (1.0,) * 32,
        "weighted_physics_gain": gains,
        "repair_cost": (0.01,) * 32,
        "repair_penalty": (0.01,) * 32,
        "cost_free_score": gains,
        "gain_total": gains,
        "policy_values": (0.0,) * 32,
        "raw_advantages": gains,
        "raw_returns": gains,
        "utility_returns": utilities,
        "return_utility_id": "symmetric-log-gain-g0-1-v1",
        "return_utility_scale": 1.0,
        "critic_value_targets": critic_targets,
        "critic_segment_target_means": segment_targets,
        "actor_advantages": utilities,
        "return_mean": float(gains_tensor.mean().item()),
        "return_min": float(gains_tensor.min().item()),
        "return_max": float(gains_tensor.max().item()),
        "grouped_reduction_active": True,
        "grouped_segment_mass_shares": (0.125,) * 8,
        "grouped_attempt_mass_shares": (0.03125,) * 32,
        "optimizer_step_delta": 1,
        "update_count": 1,
        "outer_replay_state_delta": 1,
        "outer_replay_sources": ("global",) * 8,
        "outer_replay_scenario_key_digests": tuple(f"{index:064x}" for index in range(8)),
        "outer_replay_score_kind": "critic_calibration",
        "outer_replay_critic_calibration_values": tuple(0.1 * index for index in range(8)),
        "outer_replay_repair_spread_values": tuple(0.2 * index for index in range(8)),
        "outer_replay_priority_scores": tuple(0.1 * index for index in range(8)),
        "outer_replay_critic_target_means": segment_targets,
        "outer_replay_outcome_variances": (0.1,) * 8,
        "outer_replay_standard_errors": (0.05,) * 8,
        "outer_replay_confidence_half_widths": (0.098,) * 8,
        "outer_replay_compatible_sample_counts": (4,) * 8,
        "outer_replay_policy_symmetric_kls": (0.0,) * 8,
        "outer_replay_pool_sizes": (8, 0),
        "actor_observation_dim": 158,
        "critic_observation_dim": 449,
        "gmt_observation_dim": 770,
        "critic_value_kind": "state_value",
        "critic_support_context_id": "action-pre-support-plan-kmax32-v1",
        "critic_action_conditioned": False,
        "critic_target_id": "scenario-compatible-robust-mean-symlog-v1",
        "gradient_clip_identity": "separate-actor-critic-v1",
        "gradient_clip_max_norm": 0.5,
        "critic_raw_value_loss": 0.4,
        "critic_scaled_value_loss": 0.1,
        "critic_value_normalization_id": "ema-target-std-nonamplifying-v1",
        "critic_value_scale": 2.0,
        "critic_value_normalizer_decay": 0.9,
        "critic_value_normalizer_scale_floor": 1.0,
        "critic_value_normalizer_update_count_before": 0,
        "critic_value_normalizer_update_count_after": 1,
        "actor_gradient_pre_clip_norm": 0.4,
        "actor_gradient_post_clip_norm": 0.4,
        "actor_gradient_clip_coefficient": 1.0,
        "critic_gradient_pre_clip_norm": 1.0,
        "critic_gradient_post_clip_norm": 0.5,
        "critic_gradient_clip_coefficient": 0.5 / (1.0 + 1.0e-6),
        "actor_gradient_nonzero_parameter_count": 4,
        "critic_gradient_nonzero_parameter_count": 10,
        "actor_learning_rate": 3.0e-7,
        "critic_learning_rate": 1.0e-5,
        "warmup_phase": "low_dr_joint_init",
        "k_stage_iteration": 0,
        "actor_std_parameter_delta": {"param_delta_max_abs": 0.01},
        "critic_parameter_delta": {"param_delta_max_abs": 0.1},
    }


def test_b03_b04_accept_current_b8_roles_and_reject_old_b2_shapes() -> None:
    runner = _runner()
    _install_observation_trace(runner)
    kwargs = {
        "roles": ("repair",) * 32 + ("noisy",) * 32,
        "provenance": ("deployment_noisy_q29",) * 64,
        "sources": ("sealed_noisy_q29",) * 64,
        "policy_actions": torch.zeros(32, 6),
        "horizon_k": torch.full((64,), 8, dtype=torch.long),
        "gmt_action_shapes": ((64, 29),) * 8,
        "gmt_actions_finite": True,
    }
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        audit.print_v017_repair_attempts_audit(runner, **kwargs)
    assert stream.getvalue().count("[AUDIT-B03]") == 1
    assert stream.getvalue().count("[AUDIT-B04]") == 1

    try:
        audit.print_v017_repair_attempts_audit(
            runner,
            **{**kwargs, "gmt_action_shapes": ((16, 29),) * 8},
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("v022 audit accepted the retired B2 role shape")


def test_b02_b07_accept_current_b8_telemetry_and_reject_partial_shapes() -> None:
    runner = _runner()
    telemetry = _telemetry()
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        audit.print_phase_b_telemetry_audit(runner, telemetry=telemetry)
    for label in ("AUDIT-B02", "AUDIT-B05", "AUDIT-B06", "AUDIT-B07"):
        assert stream.getvalue().count(f"[{label}]") == 1

    for changed in (
        {"valid_policy_row_mask": (True,) * 8},
        {"outer_replay_sources": ("global",) * 2},
        {"source_index": (0,) * 16 + (1,) * 16},
    ):
        try:
            audit.print_phase_b_telemetry_audit(runner, telemetry={**telemetry, **changed})
        except AssertionError:
            pass
        else:
            raise AssertionError(f"v022 audit accepted incomplete B8 telemetry: {tuple(changed)}")


def main() -> None:
    test_b03_b04_accept_current_b8_roles_and_reject_old_b2_shapes()
    test_b02_b07_accept_current_b8_telemetry_and_reject_partial_shapes()
    print("frontres_v022_formal_runtime_audit_b8_contract: PASS")


if __name__ == "__main__":
    main()
