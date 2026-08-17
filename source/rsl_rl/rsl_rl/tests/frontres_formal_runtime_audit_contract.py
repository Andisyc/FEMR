#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from frontres_contract_imports import install_frontres_contract_packages
from frontres_v022_formal_runtime_audit_b8_contract import build_current_b8_telemetry_fixture


ROOT = Path(__file__).resolve().parents[4]
install_frontres_contract_packages(ROOT / "source" / "rsl_rl" / "rsl_rl")
AUDIT_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "runners" / "frontres_formal_runtime_audit.py"
TERMINATIONS_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "terminations.py"
)
spec = importlib.util.spec_from_file_location("frontres_formal_runtime_audit_contract_module", AUDIT_PATH)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit)

SCHEDULE = (
    (8, 4, 200, 500, 1300, "lower-k8", 0.5, "linear-coupled-v1", 700, 2.381),
    (16, 4, 300, 300, 900, "lower-k16", 0.6, "linear-coupled-v1", 600, 2.381),
    (32, 4, 400, 300, 625, "lower-k32", 0.7, "linear-coupled-v1", 700, 2.381),
)


def _runner(enabled: bool = True) -> SimpleNamespace:
    policy = SimpleNamespace(
        gmt_policy=nn.Linear(3, 3),
        residual_actor=nn.Linear(158, 6),
        critic=nn.Linear(449, 1),
        num_frontres_obs=158,
        num_actor_obs=928,
    )
    policy.gmt_policy.eval()

    for param in policy.gmt_policy.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(
        [
            {
                "params": list(policy.residual_actor.parameters()),
                "lr": 3.0e-7,
                "frontres_role": "actor",
                "frontres_step_count": 0,
            },
            {
                "params": list(policy.critic.parameters()),
                "lr": 1.0e-5,
                "frontres_role": "critic",
                "frontres_step_count": 0,
            },
        ]
    )
    alg = SimpleNamespace(
        frontres_formal_runtime_audit=enabled,
        frontres_training_objective="segment_replay_hrl",
        frontres_segment_max_horizon_k=64,
        frontres_future_offsets=(1, 2),
        frontres_method_contract_id="FRS-METHOD-v025",
        frontres_gain_contract_id="FRS-GAIN-v008",
        frontres_optimization_contract_id="FRS-PPO-v012",
        frontres_training_contract_id="FRS-TRAIN-v024",
        frontres_critic_value_kind="state_value",
        frontres_critic_input_dim=449,
        frontres_critic_support_context_id="action-pre-support-plan-kmax32-v1",
        frontres_critic_action_conditioned=False,
        frontres_critic_target_id="scenario-current-exact-m4-mean-symlog-v1",
        frontres_return_utility_id="symmetric-log-gain-g0-1-v1",
        frontres_return_utility_scale=1.0,
        frontres_gradient_clip_identity="separate-actor-critic-v1",
        frontres_critic_value_normalization="ema-target-std-nonamplifying-v1",
        frontres_critic_value_normalizer_decay=0.9,
        frontres_critic_value_normalizer_scale_floor=1.0,
        frontres_critic_value_normalizer_state=audit.FrontRESValueNormalizerState(),
        max_grad_norm=0.5,
        policy=policy,
        optimizer=optimizer,
    )
    boundary = SimpleNamespace(
        live_train_enabled=True,
        live_sentinel_only=False,
        live_probe_only=False,
        live_storage_write_only=False,
        live_single_update_only=False,
        live_update_loop_only=False,
    )
    return SimpleNamespace(
        alg=alg,
        cfg={
            "frontres_specialist_mode": "rp",
            "frontres_perturbation_channels": "rp",
            "dr_scale_init": 1.25,
        },
        _dr_scale=1.25,
        _frontres_segment_replay_boundary=boundary,
        current_learning_iteration=0,
        _frontres_critic_observation_dim=449,
        _frontres_last_loaded_checkpoint_path="/tmp/frontres-v017-hsl-proposal-v2.pt",
    )


def _committed_receipt(*, transaction_id: str = "tx-v016") -> dict[str, object]:
    return {
        "transaction_id": transaction_id,
        "method_contract_id": "FRS-METHOD-v025",
        "gain_contract_id": "FRS-GAIN-v008",
        "optimization_contract_id": "FRS-PPO-v012",
        "training_contract_id": "FRS-TRAIN-v024",
        "optimizer_step_delta": 1,
        "selected_segment_count": 8,
        "policy_row_count": 32,
        "role_row_count": 64,
        "active_k": 8,
        "active_m": 4,
    }


def test_return_audit_uses_policy_gain_rows_only() -> None:
    policy_gain_steps = torch.ones(8, 8)
    non_policy_gain_steps = torch.full((8, 24), float("nan"))
    capture = SimpleNamespace(gain_steps=torch.cat((policy_gain_steps, non_policy_gain_steps), dim=1))

    selected = audit._policy_gain_steps_for_audit(capture, 8)

    assert selected is not None
    assert tuple(selected.shape) == (8, 8)
    assert bool(torch.isfinite(selected).all().item())


def test_structured_phase_b_snapshots_cover_all_formal_boundaries() -> None:
    runner = _runner()
    result = SimpleNamespace(
        transaction_id="tx-v016",
        policy_snapshot_id="pi-old-v016",
        segment_count=8,
        source_count=8,
        policy_attempt_count=32,
        valid_row_count=32,
        optimizer_step_delta=1,
        update_invocation_count=1,
        diagnostics={
            "method_contract_id": "FRS-METHOD-v025",
            "gain_contract_id": "FRS-GAIN-v008",
            "optimization_contract_id": "FRS-PPO-v012",
            "training_contract_id": "FRS-TRAIN-v024",
            "critic_support_context_id": "action-pre-support-plan-kmax32-v1",
            "selected_segment_count": 8,
            "active_m": 4,
            "policy_row_count": 32,
            "role_row_count": 64,
            "grouped_motion_mass_shares": (0.125,) * 8,
            "grouped_segment_mass_shares": (0.125,) * 8,
            "grouped_attempt_mass_shares": (0.03125,) * 32,
        },
    )
    checkpoint_payload = {
                "iter": 4,
                "model_state_dict": {},
                "optimizer_state_dict": {
                    "state": {},
                    "param_groups": [
                        {"frontres_role": "actor", "lr": 3.0e-7},
                        {"frontres_role": "critic", "lr": 1.0e-5},
                    ],
                },
                "obs_norm_state_dict": {},
                "privileged_obs_norm_state_dict": {},
                "frontres_segment_sampler_state_dict": {},
                "frontres_segment_k_curriculum": SCHEDULE,
                "frontres_critic_value_normalizer_state_dict": {
                    "normalization_id": "ema-target-std-nonamplifying-v1",
                    "mean": 0.0,
                    "second_moment": 1.0,
                    "update_count": 4,
                },
                "frontres_v015_checkpoint_identity": {
                    "format": "frontres-v024-checkpoint-v19",
                    "method_contract_id": "FRS-METHOD-v025",
                    "gain_contract_id": "FRS-GAIN-v008",
                    "optimization_contract_id": "FRS-PPO-v012",
                    "training_contract_id": "FRS-TRAIN-v024",
                    "dr_curriculum_schema_id": "nested-k-dr-four-class-v1",
                    "scalar_target_id": "symmetric-log-recovery-aware-utility-v1",
                    "return_utility": {
                        "identity": "symmetric-log-gain-g0-1-v1",
                        "scale": 1.0,
                        "placement": "per-attempt-before-current-exact-m4-mean",
                    },
                    "physics_schema_id": "clean-anchored-contact-zmp-survival-v1",
                    "grouped_schema_id": "grouped-all-attempt-scalar-v1",
                    "critic": {
                        "value_kind": "state_value",
                        "input_dim": 449,
                        "support_context_id": "action-pre-support-plan-kmax32-v1",
                        "action_conditioned": False,
                        "target_id": "scenario-current-exact-m4-mean-symlog-v1",
                        "return_utility_id": "symmetric-log-gain-g0-1-v1",
                        "return_utility_scale": 1.0,
                    },
                    "gradient_clip": {"identity": "separate-actor-critic-v1", "max_norm": 0.5},
                    "critic_value_normalizer": {
                        "identity": "ema-target-std-nonamplifying-v1",
                        "decay": 0.9,
                        "scale_floor": 1.0,
                    },
                    "gain": {"beta": 0.02},
                    "gmt": {
                        "checkpoint_sha256": "a" * 64,
                        "normalizer_dim": 770,
                        "normalizer_fingerprint": "b" * 64,
                    },
                    "future_intent_layout": {
                        "actor_dim": 928,
                        "prefix_dim": 158,
                        "gmt_dim": 770,
                        "future_offsets": (1, 2),
                    },
                    "curriculum": {
                        "schedule": SCHEDULE,
                        "schedule_fingerprint": "f" * 64,
                        "k_stage_index": 0,
                        "active_k": 8,
                        "stage_iteration": 0,
                        "absolute_iteration": 4,
                        "phase": "low_dr_joint_init",
                        "phase_iteration": 0,
                        "actor_loss_weight": 1.0 / 700.0,
                    },
                    "normalizer": {"mode": "disabled"},
                    "transaction": {"state": "committed", "receipt": _committed_receipt()},
                },
            }
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        audit.print_formal_route_audit(runner, num_learning_iterations=1)
        audit.print_segment_replay_transaction_audit(runner, result=result)
        audit.print_checkpoint_payload_audit(runner, path="/tmp/model_4.pt", payload=checkpoint_payload)
        audit.print_checkpoint_reload_audit(
            runner,
            path="/tmp/model_4.pt",
            payload=checkpoint_payload,
            validated_identity=checkpoint_payload["frontres_v015_checkpoint_identity"],
            file_sha256="c" * 64,
        )
    output = stream.getvalue()
    for label in (
        "AUDIT-ROUTE-01",
        "AUDIT-PERTURB-01",
        "AUDIT-HSL-LOAD-01",
        "AUDIT-B01",
        "AUDIT-SEGMENT-REPLAY-01",
        "AUDIT-PERSIST-01",
        "AUDIT-B08",
    ):
        assert output.count(f"[{label}]") == 1
    assert "alternate_modes=0" in output
    assert "specialist_mode=rp" in output
    assert "perturbation_channels=rp" in output
    assert "dr_scale=1.25" in output
    assert "max_horizon_k=64" in output
    transaction_line = next(line for line in output.splitlines() if line.startswith("[AUDIT-SEGMENT-REPLAY-01]"))
    assert "segments=8" in transaction_line
    assert "attempts_per_segment=4" in transaction_line
    assert "policy_rows=32" in transaction_line and "valid_rows=32" in transaction_line
    assert "segment_voting_weights=count=8,head=(0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125)" in transaction_line
    assert "attempt_voting_weights=count=32,head=(0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125)" in transaction_line
    assert "optimizer_step_delta=1" in transaction_line and "update_invocations=1" in transaction_line
    assert "FRS-METHOD-v025/FRS-GAIN-v008/FRS-PPO-v012/FRS-TRAIN-v024" in transaction_line
    assert "FRS-GAIN-v002" not in output and "shape=(2, 870)" not in output
    assert "lower-k8" in output and "active_k=8" in output

    for changed in (
        {"optimization_contract_id": "FRS-PPO-v004"},
        {"grouped_attempt_mass_shares": (0.5, 0.5)},
        {"grouped_attempt_mass_shares": (0.7, 0.1, 0.1, 0.1)},
    ):
        invalid = SimpleNamespace(**vars(result))
        invalid.diagnostics = {**result.diagnostics, **changed}
        try:
            audit.print_segment_replay_transaction_audit(runner, result=invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"active Segment Replay audit accepted invalid facts: {changed}")


def test_checkpoint_audit_rejects_missing_or_mixed_v013_curriculum() -> None:
    runner = _runner()
    schedule = SCHEDULE
    base = {
        "iter": 4,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "obs_norm_state_dict": {},
        "privileged_obs_norm_state_dict": {},
        "frontres_segment_sampler_state_dict": {},
        "frontres_segment_k_curriculum": schedule,
        "frontres_critic_value_normalizer_state_dict": {
            "normalization_id": "ema-target-std-nonamplifying-v1",
            "mean": 0.0,
            "second_moment": 1.0,
            "update_count": 4,
        },
        "frontres_v015_checkpoint_identity": {
            "format": "frontres-v024-checkpoint-v19",
            "method_contract_id": "FRS-METHOD-v025",
            "gain_contract_id": "FRS-GAIN-v008",
            "optimization_contract_id": "FRS-PPO-v012",
            "training_contract_id": "FRS-TRAIN-v024",
            "dr_curriculum_schema_id": "nested-k-dr-four-class-v1",
            "scalar_target_id": "symmetric-log-recovery-aware-utility-v1",
            "return_utility": {
                "identity": "symmetric-log-gain-g0-1-v1",
                "scale": 1.0,
                "placement": "per-attempt-before-current-exact-m4-mean",
            },
            "physics_schema_id": "clean-anchored-contact-zmp-survival-v1",
            "grouped_schema_id": "grouped-all-attempt-scalar-v1",
            "critic": {
                "value_kind": "state_value",
                "input_dim": 449,
                "support_context_id": "action-pre-support-plan-kmax32-v1",
                "action_conditioned": False,
                "target_id": "scenario-current-exact-m4-mean-symlog-v1",
                "return_utility_id": "symmetric-log-gain-g0-1-v1",
                "return_utility_scale": 1.0,
            },
            "gradient_clip": {"identity": "separate-actor-critic-v1", "max_norm": 0.5},
            "critic_value_normalizer": {
                "identity": "ema-target-std-nonamplifying-v1",
                "decay": 0.9,
                "scale_floor": 1.0,
            },
            "gain": {"beta": 0.02},
            "gmt": {
                "checkpoint_sha256": "a" * 64,
                "normalizer_dim": 770,
                "normalizer_fingerprint": "b" * 64,
            },
            "future_intent_layout": {
                "actor_dim": 928,
                "prefix_dim": 158,
                "gmt_dim": 770,
                "future_offsets": (1, 2),
            },
            "curriculum": {"schedule": schedule, "active_k": 8},
            "transaction": {"state": "committed", "receipt": _committed_receipt()},
        },
    }
    for mutate in (
        lambda payload: payload.pop("frontres_segment_k_curriculum"),
        lambda payload: payload["frontres_v015_checkpoint_identity"]["curriculum"].update(
            schedule=((8, 2, 200, 500, 1300), (16, 3, 300, 300, 900), (32, 3, 400, 300, 625))
        ),
    ):
        payload = {
            **base,
            "frontres_v015_checkpoint_identity": {
                **base["frontres_v015_checkpoint_identity"],
                "curriculum": dict(base["frontres_v015_checkpoint_identity"]["curriculum"]),
            },
        }
        mutate(payload)
        try:
            audit.print_checkpoint_payload_audit(runner, path="/tmp/model_bad.pt", payload=payload)
        except AssertionError:
            pass
        else:
            raise AssertionError("formal audit accepted missing or mixed v013 curriculum")


def test_phase_b_one_action_and_final_telemetry_are_fail_closed() -> None:
    from rsl_rl.runners.frontres_segment_runtime_types import (
        bind_frontres_collection_context,
        open_frontres_checkpoint_transaction_barrier,
        update_frontres_observation_trace,
    )

    runner = _runner()
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
    evidence = SimpleNamespace(
        roles=("repair",) * 32 + ("noisy",) * 32,
        intent_q29_provenance=("deployment_noisy_q29",) * 64,
        intent_q29_source=("sealed_noisy_q29",) * 64,
        policy_actions=torch.zeros(32, 6),
        horizon_k=torch.full((64,), 8, dtype=torch.long),
        continuation=torch.zeros(64, 8, 65),
        frozen_gmt_env_actions=torch.zeros(64, 8, 29),
        actor_forward_count=1,
        later_femr_action_count=0,
    )
    telemetry = build_current_b8_telemetry_fixture()
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        audit.print_one_action_k_audit(runner, evidence=evidence)
        audit.print_phase_b_telemetry_audit(runner, telemetry=telemetry)
    output = stream.getvalue()
    for label in ("AUDIT-B02", "AUDIT-B03", "AUDIT-B04", "AUDIT-B05", "AUDIT-B06", "AUDIT-B07"):
        assert output.count(f"[{label}]") == 1

    update_frontres_observation_trace(runner, post_advance_gmt_read_count=16)
    try:
        audit.print_one_action_k_audit(runner, evidence=evidence)
    except AssertionError:
        pass
    else:
        raise AssertionError("AUDIT-B04 scaled frozen-GMT read count with M/role rows")
    update_frontres_observation_trace(runner, post_advance_gmt_read_count=8)

    invalid_evidence = SimpleNamespace(**{**vars(evidence), "later_femr_action_count": 1})
    try:
        audit.print_one_action_k_audit(runner, evidence=invalid_evidence)
    except AssertionError:
        pass
    else:
        raise AssertionError("AUDIT-B04 accepted a later FEMR action")

    runner._frontres_v015_one_action_k_phase = "frozen"
    active_stream = io.StringIO()
    active_kwargs = {
        "roles": evidence.roles,
        "provenance": evidence.intent_q29_provenance,
        "sources": evidence.intent_q29_source,
        "policy_actions": evidence.policy_actions,
        "horizon_k": evidence.horizon_k,
        "gmt_action_shapes": ((64, 29),) * 8,
        "gmt_actions_finite": True,
    }
    with contextlib.redirect_stdout(active_stream):
        audit.print_v017_repair_attempts_audit(runner, **active_kwargs)
    assert active_stream.getvalue().count("[AUDIT-B03]") == 1
    assert active_stream.getvalue().count("[AUDIT-B04]") == 1
    try:
        audit.print_v017_repair_attempts_audit(
            runner,
            **{**active_kwargs, "gmt_action_shapes": ((16, 29),) * 7},
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("AUDIT-B04 accepted seven frozen-GMT steps for K8")

    invalid_telemetry = {
        **telemetry,
        "scenario_ids": ("mixed",) + tuple(telemetry["scenario_ids"])[1:],
    }
    try:
        audit.print_phase_b_telemetry_audit(runner, telemetry=invalid_telemetry)
    except AssertionError:
        pass
    else:
        raise AssertionError("AUDIT-B02 accepted mixed scenario identity")

    for changed, label in (
        (
            {
                "critic_value_targets": (
                    float(telemetry["critic_value_targets"][0]) + 1.0,
                ) + tuple(telemetry["critic_value_targets"])[1:]
            },
            "AUDIT-B06 accepted per-attempt Critic targets",
        ),
        ({"critic_gradient_post_clip_norm": 0.6}, "AUDIT-B07 accepted an over-limit Critic gradient"),
        ({"actor_gradient_clip_coefficient": 0.5}, "AUDIT-B07 accepted a coupled critic-only Actor clip"),
    ):
        try:
            audit.print_phase_b_telemetry_audit(runner, telemetry={**telemetry, **changed})
        except AssertionError:
            pass
        else:
            raise AssertionError(label)


def test_audit_flag_off_is_silent_and_hooks_are_on_formal_owners() -> None:
    runner = _runner(enabled=False)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        audit.print_formal_route_audit(runner, num_learning_iterations=1)
        audit.print_segment_replay_transaction_audit(runner, result=SimpleNamespace())
    assert stream.getvalue() == ""

    expected_hooks = (
        ("source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py", "print_formal_route_audit("),
        ("source/rsl_rl/rsl_rl/runners/frontres_segment_formal_transaction.py", "print_segment_replay_transaction_audit("),
        ("source/rsl_rl/rsl_rl/runners/frontres_checkpointing.py", "print_checkpoint_payload_audit("),
        ("source/rsl_rl/rsl_rl/runners/frontres_checkpointing.py", "print_checkpoint_reload_audit("),
        ("scripts/rsl_rl/train.py", "--frontres_formal_runtime_audit"),
    )
    for path_key, marker in expected_hooks:
        path = path_key.split("#", 1)[0]
        assert marker in (ROOT / path).read_text(), f"missing {marker} in {path}"

    train_source = (ROOT / "scripts/rsl_rl/train.py").read_text()
    dataset_source = (ROOT / "source/rsl_rl/rsl_rl/frontres/frontres_segment_dataset.py").read_text()
    assert "getattr(agent_cfg, 'frontres_specialist_mode', 'missing')" in train_source
    assert "getattr(agent_cfg, 'frontres_perturbation_channels', 'missing')" in train_source
    max_horizon_set = train_source.index('_set_if_present(alg_cfg, "frontres_segment_max_horizon_k", 64)')
    perturb_probe = train_source.index('"[AUDIT-PERTURB-01] "')
    assert max_horizon_set < perturb_probe, "AUDIT-PERTURB-01 must print the finalized horizon preset"
    assert "cache_horizon_k=batch.horizon_k" in dataset_source


def test_b03_b04_are_connected_to_the_active_v017_repair_collector() -> None:
    collector_source = (
        ROOT / "source/rsl_rl/rsl_rl/runners/frontres_segment_one_action_k.py"
    ).read_text()
    active_start = collector_source.index("def collect_frontres_v017_repair_attempts(")
    legacy_start = collector_source.index("def collect_frontres_v015_one_action_k_evidence(")
    active_collector = collector_source[active_start:legacy_start]
    assert "print_v017_repair_attempts_audit(" in active_collector, (
        "AUDIT-B03/B04 must consume the active formal v017 Repair collector facts"
    )


def test_phase_b_return_audit_reproduces_float32_reduction() -> None:
    runner = _runner()
    telemetry = build_current_b8_telemetry_fixture()
    gains = torch.tensor(
        (
            0.7266710996627808,
            -1314020.0,
            2.5,
            -3.25,
            -0.04292364418506622,
            -2306889.25,
            4.0,
            -5.5,
            *((0.0,) * 24),
        ),
        dtype=torch.float32,
    )
    utility_returns = torch.sign(gains) * torch.log1p(torch.abs(gains))
    source_index = torch.tensor(telemetry["source_index"], dtype=torch.long)
    segment_targets = tuple(
        float(utility_returns[source_index == source].mean().item()) for source in range(8)
    )
    telemetry.update(
        {
            "intent_gain": tuple(float(value) for value in gains.tolist()),
            "physics_gain": tuple(float(value) for value in gains.tolist()),
            "weighted_physics_gain": tuple(float(value) for value in gains.tolist()),
            "cost_free_score": tuple(float(value) for value in gains.tolist()),
            "gain_total": tuple(float(value) for value in gains.tolist()),
            "policy_values": (0.0,) * 32,
            "raw_advantages": tuple(float(value) for value in gains.tolist()),
            "raw_returns": tuple(float(value) for value in gains.tolist()),
            "utility_returns": tuple(float(value) for value in utility_returns.tolist()),
            "critic_value_targets": tuple(segment_targets[int(source)] for source in source_index.tolist()),
            "critic_segment_target_means": segment_targets,
            "actor_advantages": tuple(float(value) for value in utility_returns.tolist()),
            "outer_replay_critic_target_means": segment_targets,
            "return_mean": float(gains.mean().item()),
            "return_min": float(gains.min().item()),
            "return_max": float(gains.max().item()),
        }
    )
    audit.print_phase_b_telemetry_audit(runner, telemetry=telemetry)

    corrupted = {**telemetry, "return_mean": telemetry["return_mean"] + 1.0}
    try:
        audit.print_phase_b_telemetry_audit(runner, telemetry=corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("AUDIT-B06 accepted a corrupted float32 return mean")


def test_active_k_audit_ids_exclude_legacy_state_driven_sampler() -> None:
    audit_source = AUDIT_PATH.read_text()
    sampler_source = (
        ROOT / "source" / "rsl_rl" / "rsl_rl" / "frontres" / "frontres_segment_sampler.py"
    ).read_text()
    warmup_source = (
        ROOT / "source" / "rsl_rl" / "rsl_rl" / "frontres" / "frontres_segment_warmup.py"
    ).read_text()

    assert '"AUDIT-KPLAN-01"' not in audit_source
    assert '"AUDIT-KPLAN-01"' in warmup_source
    assert '"AUDIT-LEGACY-KPLAN-01"' in sampler_source
    assert '"AUDIT-LEGACY-KROLLOUT-01"' in sampler_source
    assert "formal TRAIN-v013 直接消费 sealed K/M" in sampler_source


def test_ppo_audit_reports_zero_valid_batch_without_changing_training_control_flow() -> None:
    runner = _runner()
    result = SimpleNamespace(
        warmup_phase="low_dr_joint_init",
        warmup_phase_iteration=0,
        actor_loss_weight=1.0 / 700.0,
        valid_count=0,
        total_loss=torch.tensor(0.0),
        param_grad_norm=0.0,
        param_delta_l2=0.0,
        distribution_kl_mean=0.0,
        post_update_distribution_kl_mean=0.0,
        trust_region_accepted=0,
    )
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        audit.print_ppo_audit(runner, result=result)
    output = stream.getvalue()
    assert "[AUDIT-PPO-01]" in output
    assert "valid=0" in output
    assert "update_observed=0" in output


def test_reset_lifecycle_audit_is_role_aware_and_separates_timeout_from_termination() -> None:
    runner = _runner()
    root = torch.zeros(8, 3)
    root[4:6] = 1.0
    joint_pos = torch.zeros(8, 2)
    joint_pos[6:8] = 2.0
    runner.env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_pos_w=root,
                    root_quat_w=torch.zeros(8, 4),
                    root_lin_vel_w=torch.zeros(8, 3),
                    root_ang_vel_w=torch.zeros(8, 3),
                    joint_pos=joint_pos,
                    joint_vel=torch.zeros(8, 2),
                )
            )
        }
    )
    layout = SimpleNamespace(n_train=2, n_candidate=2, n_base=2, n_clean=2)
    pair_state = audit.snapshot_reset_pair_state(runner, layout)
    dones = torch.tensor([True, False, False, True, True, True, False, False])
    time_outs = torch.tensor([True, False, False, False, False, False, False, False])
    terminated = dones & ~time_outs
    stream = io.StringIO()
    audit.configure_formal_runtime_probe(False)
    with contextlib.redirect_stdout(stream):
        audit.print_reset_lifecycle_audit(
            runner,
            pair_layout=layout,
            phase="reset",
            episode_before=torch.arange(8),
            episode_randomized=torch.arange(8) + 10,
            episode_after_reset=torch.tensor([0, 0, 12, 13, 14, 15, 16, 17]),
            pair_state=pair_state,
        )
        audit.print_reset_lifecycle_audit(
            runner,
            pair_layout=layout,
            phase="step",
            rollout_step=0,
            dones=dones,
            time_outs=time_outs,
            terminated=terminated,
            alive=~dones,
            survival_steps=torch.ones(8),
        )
        audit.print_reset_lifecycle_audit(
            runner,
            pair_layout=layout,
            phase="final",
            first_done_step=torch.tensor([0, -1, -1, 0, 0, 0, -1, -1]),
        )
    output = stream.getvalue()
    assert output.count("[AUDIT-RESET-LIFECYCLE-01]") == 3
    assert "phase=reset" in output and "episode_after_reset=" in output
    assert "noisy:count=2 max=1" in output
    assert "clean:count=2 max=2" in output
    assert "done={policy:1,candidate:1,noisy:2,clean:0}" in output
    assert "time_out={policy:1,candidate:0,noisy:0,clean:0}" in output
    assert "terminated={policy:0,candidate:1,noisy:2,clean:0}" in output
    assert "phase=final" in output and "first_done_step=" in output


def test_reset_pair_root_error_removes_per_environment_world_origins() -> None:
    runner = _runner()
    origins = torch.tensor([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [40.0, 0.0, 0.0], [60.0, 0.0, 0.0]])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=origins.clone(),
            root_quat_w=torch.zeros(4, 4),
            root_lin_vel_w=torch.zeros(4, 3),
            root_ang_vel_w=torch.zeros(4, 3),
            joint_pos=torch.zeros(4, 2),
            joint_vel=torch.zeros(4, 2),
        )
    )
    runner.env = SimpleNamespace(scene=SimpleNamespace(robot=robot, env_origins=origins))
    layout = SimpleNamespace(n_train=1, n_candidate=1, n_base=1, n_clean=1)
    snapshot = audit.snapshot_reset_pair_state(runner, layout)
    assert all("max=0" in value for value in snapshot["root_pair_error"].values())


def test_termination_term_snapshot_preserves_term_and_role_identity() -> None:
    layout = SimpleNamespace(n_train=2, n_candidate=2, n_base=2, n_clean=2)
    terms = {
        "motion_end": torch.zeros(8, dtype=torch.bool),
        "anchor_pos": torch.tensor([True, False, False, False, True, True, False, False]),
        "anchor_ori": torch.tensor([False, False, True, True, False, False, True, False]),
    }
    manager = SimpleNamespace(
        active_terms=tuple(terms),
        get_term=lambda name: terms[name],
    )
    runner = SimpleNamespace(env=SimpleNamespace(termination_manager=manager))
    snapshot = audit.snapshot_termination_terms(runner, layout, batch_size=8)
    assert snapshot["motion_end"] == {"policy": 0, "candidate": 0, "noisy": 0, "clean": 0}
    assert snapshot["anchor_pos"] == {"policy": 1, "candidate": 0, "noisy": 2, "clean": 0}
    assert snapshot["anchor_ori"] == {"policy": 0, "candidate": 2, "noisy": 0, "clean": 1}


def test_relational_formal_route_audit_accepts_actor_only_identity() -> None:
    runner = _runner()
    for parameter in runner.alg.policy.critic.parameters():
        parameter.requires_grad_(False)
    runner.alg.optimizer = torch.optim.Adam(
        [{
            "params": list(runner.alg.policy.residual_actor.parameters()),
            "lr": 3.0e-7,
            "frontres_role": "actor",
            "frontres_step_count": 0,
        }]
    )
    runner.alg.frontres_relational_actor_only = True
    runner.alg.frontres_training_objective = "segment_replay_relational"
    runner.alg.frontres_method_contract_id = "FRS-METHOD-v026"
    runner.alg.frontres_gain_contract_id = "FRS-GAIN-v009"
    runner.alg.frontres_optimization_contract_id = "FRS-PPO-v013"
    runner.alg.frontres_training_contract_id = "FRS-TRAIN-v025"
    runner.alg.frontres_critic_support_context_id = "none"
    runner.alg.frontres_critic_value_normalization = "none"
    with contextlib.redirect_stdout(io.StringIO()) as buffer:
        audit.print_formal_route_audit(runner, num_learning_iterations=1)
    assert "FRS-GAIN-v009" in buffer.getvalue()


if __name__ == "__main__":
    test_return_audit_uses_policy_gain_rows_only()
    test_structured_phase_b_snapshots_cover_all_formal_boundaries()
    test_phase_b_one_action_and_final_telemetry_are_fail_closed()
    test_checkpoint_audit_rejects_missing_or_mixed_v013_curriculum()
    test_audit_flag_off_is_silent_and_hooks_are_on_formal_owners()
    test_b03_b04_are_connected_to_the_active_v017_repair_collector()
    test_phase_b_return_audit_reproduces_float32_reduction()
    test_active_k_audit_ids_exclude_legacy_state_driven_sampler()
    test_ppo_audit_reports_zero_valid_batch_without_changing_training_control_flow()
    test_reset_lifecycle_audit_is_role_aware_and_separates_timeout_from_termination()
    test_reset_pair_root_error_removes_per_environment_world_origins()
    test_termination_term_snapshot_preserves_term_and_role_identity()
    test_relational_formal_route_audit_accepts_actor_only_identity()
    print("frontres_formal_runtime_audit_contract: ok")
