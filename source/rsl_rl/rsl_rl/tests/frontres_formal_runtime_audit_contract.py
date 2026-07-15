#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[4]
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


def _runner(enabled: bool = True) -> SimpleNamespace:
    policy = SimpleNamespace(
        gmt_policy=nn.Linear(3, 3),
    )
    for param in policy.gmt_policy.parameters():
        param.requires_grad = False
    actor_param = nn.Parameter(torch.ones(1))
    alg = SimpleNamespace(
        frontres_formal_runtime_audit=enabled,
        frontres_training_objective="segment_replay_hrl",
        frontres_segment_max_horizon_k=64,
        policy=policy,
        optimizer=torch.optim.SGD([actor_param], lr=0.1),
    )
    boundary = SimpleNamespace(
        live_train_enabled=True,
        live_sentinel_only=False,
        live_probe_only=False,
        live_storage_write_only=False,
        live_single_update_only=False,
        live_update_loop_only=False,
        offline_eval_only=False,
        sequence_offline_eval_only=False,
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
        current_learning_iteration=3,
    )


def test_structured_phase_b_snapshots_cover_all_formal_boundaries() -> None:
    runner = _runner()
    sample = SimpleNamespace(
        segment_ids=torch.tensor([3, 4]),
        source_index=torch.tensor([0, 1]),
        horizon_k=torch.tensor([8, 16]),
    )
    batch = SimpleNamespace(frontres_segment_trial_role=("policy", "search"))
    capture = SimpleNamespace(
        transition_obs=torch.ones(2, 870),
        transition_actions=torch.ones(2, 6),
        transition_means=torch.zeros(2, 6),
        transition_sigmas=torch.ones(2, 6) * 0.01,
        transition_perturbation_rp=torch.tensor([[0.1, -0.1], [0.2, -0.2]]),
        n_train=1,
        n_candidate=1,
        n_base=0,
        n_clean=0,
    )
    storage = SimpleNamespace(
        actions=torch.ones(2, 6),
        old_means=torch.zeros(2, 6),
        old_sigmas=torch.ones(2, 6) * 0.01,
        returns=torch.tensor([0.2, -0.1]),
        valid_mask=torch.tensor([True, False]),
    )
    result = SimpleNamespace(
        warmup_phase="critic_only",
        warmup_phase_iteration=3,
        actor_loss_weight=0.0,
        valid_count=1,
        total_loss=torch.tensor(0.5),
        param_grad_norm=0.3,
        param_delta_l2=0.2,
        distribution_kl_mean=0.001,
        post_update_distribution_kl_mean=0.002,
        trust_region_accepted=1,
    )
    summary = {
        "reset_success_count": 2,
        "ppo_valid_count": 1,
        "gain_style_mean": 0.3,
        "gain_physics_mean": 0.2,
        "gain_repair_cost_mean": 0.1,
        "gain_total_mean": 0.4,
        "sampler_update_priority_before_mean": 0.5,
        "sampler_update_priority_after_mean": 0.6,
    }
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        audit.print_formal_route_audit(runner, num_learning_iterations=1)
        audit.print_sampler_audit(runner, update_step=0, sample=sample, batch=batch, summary=summary)
        audit.print_rollout_storage_audit(runner, capture=capture, summary=summary, storage_batch=storage)
        audit.print_ppo_audit(runner, result=result)
        audit.print_checkpoint_payload_audit(
            runner,
            path="/tmp/model_4.pt",
            payload={
                "iter": 4,
                "model_state_dict": {},
                "optimizer_state_dict": {},
                "obs_norm_state_dict": {},
                "frontres_segment_sampler_state_dict": {},
                "frontres_gain_config": {},
                "frontres_segment_warmup_config": {"critic_warmup_iterations": 200, "actor_warmup_iterations": 500},
            },
        )
    output = stream.getvalue()
    for label in (
        "AUDIT-ROUTE-01", "AUDIT-PERTURB-01", "AUDIT-PERTURB-02", "AUDIT-SEGDATA-01",
        "AUDIT-SAMPLER-01", "AUDIT-KPLAN-01", "AUDIT-KROLLOUT-01", "AUDIT-OBS-01",
        "AUDIT-ACTION-01", "AUDIT-APPLY-01", "AUDIT-GMT-01", "AUDIT-PAIR-01",
        "AUDIT-PAIR-EVIDENCE-01", "AUDIT-GAIN-01", "AUDIT-RETURN-01", "AUDIT-HSL-LOAD-01",
        "AUDIT-WARMUP-01", "AUDIT-PPO-01", "AUDIT-PERSIST-01", "AUDIT-DIAG-01",
    ):
        assert output.count(f"[{label}]") == 1
    assert "alternate_modes=0" in output
    assert "specialist_mode=rp" in output
    assert "perturbation_channels=rp" in output
    assert "dr_scale=1.25" in output
    assert "max_horizon_k=64" in output
    assert "shape=(2, 870)" in output
    assert "prefix100=shape=(2, 100)" in output
    assert "suffix770=shape=(2, 770)" in output
    assert "shape=(2, 6)" in output
    assert "sigma=shape=(2, 6)" in output
    assert "perturb_rp=shape=(2, 2)" in output
    assert "gmt_trainable=0 gmt_in_optimizer=0" in output
    assert "warmup={'critic_warmup_iterations': 200, 'actor_warmup_iterations': 500}" in output


def test_audit_flag_off_is_silent_and_hooks_are_on_formal_owners() -> None:
    runner = _runner(enabled=False)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        audit.print_formal_route_audit(runner, num_learning_iterations=1)
    assert stream.getvalue() == ""

    expected_hooks = {
        "source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py": "print_formal_route_audit(",
        "source/rsl_rl/rsl_rl/runners/frontres_segment_live_sampler.py": "print_sampler_audit(",
        "source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py": "print_rollout_storage_audit(",
        "source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py#reset": "print_reset_lifecycle_audit(",
        "source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py#ppo": "print_ppo_audit(",
        "source/rsl_rl/rsl_rl/runners/frontres_checkpointing.py": "print_checkpoint_payload_audit(",
        "scripts/rsl_rl/train.py": "--frontres_formal_runtime_audit",
    }
    for path_key, marker in expected_hooks.items():
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


def test_ppo_audit_reports_zero_valid_batch_without_changing_training_control_flow() -> None:
    runner = _runner()
    result = SimpleNamespace(
        warmup_phase="critic_only",
        warmup_phase_iteration=0,
        actor_loss_weight=0.0,
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


def test_runtime_audit_atlas_source_comments_and_checklist_share_ids() -> None:
    atlas_path = ROOT / "note" / "architecture" / "runtime" / "04_stage3_formal_runtime_audit.data.json"
    atlas = json.loads(atlas_path.read_text())
    atlas_text = json.dumps(atlas, ensure_ascii=False)
    atlas_entry = (ROOT / "note" / "architecture" / "04_stage3_formal_runtime_audit.html").read_text()
    assert 'window.location.protocol === "file:"' in atlas_entry
    assert "http://127.0.0.1:8765/04_stage3_formal_runtime_audit.html" in atlas_entry
    checklist = (ROOT / "note" / "frontres_core" / "checklists" / "modification_checklist.md").read_text()
    audit_ids = [
        "AUDIT-ROUTE-01", "AUDIT-PERTURB-01", "AUDIT-PERTURB-02", "AUDIT-SEGDATA-01",
        "AUDIT-SAMPLER-01", "AUDIT-KPLAN-01", "AUDIT-KROLLOUT-01", "AUDIT-RESET-LIFECYCLE-01",
        "AUDIT-ANCHOR-Z-01",
        "AUDIT-OBS-01",
        "AUDIT-ACTION-01", "AUDIT-APPLY-01", "AUDIT-GMT-01", "AUDIT-PAIR-01",
        "AUDIT-PAIR-EVIDENCE-01", "AUDIT-GAIN-01", "AUDIT-RETURN-01", "AUDIT-HSL-LOAD-01",
        "AUDIT-WARMUP-01", "AUDIT-PPO-01", "AUDIT-PERSIST-01", "AUDIT-DIAG-01",
    ]
    audit_source = AUDIT_PATH.read_text()
    termination_source = TERMINATIONS_PATH.read_text()
    for audit_id in audit_ids:
        assert audit_id in atlas_text
        assert audit_id in checklist
        owner_source = termination_source if audit_id == "AUDIT-ANCHOR-Z-01" else audit_source
        assert audit_id in owner_source
    assert atlas["layout"] == "repository_reading_atlas"
    assert atlas["runtimeOrder"] == audit_ids
    modules = {
        module["id"]: module
        for system in atlas["systems"]
        for module in system["modules"]
    }
    assert len(modules) == 22
    why_here_texts: list[str] = []
    for audit_id in audit_ids:
        module = modules[audit_id]
        assert module["cardKind"] == "runtime_probe"
        assert module["title"].startswith("Probe ")
        assert module["probe"]["owner"]
        assert module["probe"]["insertion"]
        assert module["probe"]["capture"]
        assert module["probe"]["failIf"]
        assert len(module["probeSteps"]) == len(module["mainRoute"])
        assert all(
            step["location"] and step["capture"] and step["whyHere"] and step["failureOwner"]
            for step in module["probeSteps"]
        )
        why_here_texts.extend(step["whyHere"] for step in module["probeSteps"])
        assert any(str(value).startswith("Design:") for value in module["objects"])
        assert module["gap"]
        assert len(module["mainRoute"]) == len(module["mainRouteTitles"])
        owner_path = ROOT / module["files"][0]["path"]
        assert owner_path.exists(), f"Atlas owner path does not exist: {owner_path}"
        owner_text = owner_path.read_text()
        owner_lines = owner_text.splitlines()
        assert audit_id in owner_text, f"{audit_id} is not inserted at Atlas owner {owner_path}"
        if audit_id == "AUDIT-RESET-LIFECYCLE-01":
            assert "quartet reset is live-aligned" in owner_text
        elif audit_id == "AUDIT-ANCHOR-Z-01":
            assert "cache fix is inserted" in owner_text
        else:
            assert "Result: PENDING_LIVE." in owner_text, f"{audit_id} owner lacks a PENDING_LIVE comment"
        for block_id in ("B1", "B2", "B3"):
            assert f"# {block_id}:" in owner_text, f"{audit_id} owner lacks human-readable {block_id} comments"
        for step_index, step in enumerate(module["probeSteps"], start=1):
            assert step["sourceHref"].startswith("/open-source?path=")
            assert f"line={step['sourceLine']}" in step["sourceHref"]
            query = parse_qs(urlparse(step["sourceHref"]).query)
            assert query.get("path") == [module["files"][0]["path"]], (
                f"Atlas source link path drift: {query.get('path')} != {module['files'][0]['path']}"
            )
            linked_path = ROOT / query["path"][0]
            assert linked_path.is_file(), f"Atlas source link target does not exist: {linked_path}"
            source_line = int(step["sourceLine"])
            assert 1 <= source_line <= len(owner_lines)
            assert f"# B{step_index}:" in owner_lines[source_line - 1]
    assert modules["AUDIT-PPO-01"]["gap"].startswith("blocked:")
    assert "8/8 policy rows" in modules["AUDIT-PPO-01"]["gap"]
    assert modules["AUDIT-PERSIST-01"]["gap"].startswith("unconfirmed:")
    assert len(why_here_texts) == 66
    assert len(set(why_here_texts)) == 66, "whyHere must not be a shared template across probe boundaries"

    assert "result = (error > threshold) | torch.isnan(error)" in termination_source
    assert "return result" in termination_source
    for field in (
        "reference_z=role_reference_z",
        "robot_z=",
        "signed_error=",
        "abs_error=",
        "threshold=float(threshold)",
        "clean_reference_z=",
        "raw_reference_z=",
        "correction_z=",
        "time_steps=",
        "motion_indices=",
    ):
        assert field in termination_source


if __name__ == "__main__":
    test_structured_phase_b_snapshots_cover_all_formal_boundaries()
    test_audit_flag_off_is_silent_and_hooks_are_on_formal_owners()
    test_ppo_audit_reports_zero_valid_batch_without_changing_training_control_flow()
    test_reset_lifecycle_audit_is_role_aware_and_separates_timeout_from_termination()
    test_reset_pair_root_error_removes_per_environment_world_origins()
    test_termination_term_snapshot_preserves_term_and_role_identity()
    test_runtime_audit_atlas_source_comments_and_checklist_share_ids()
    print("frontres_formal_runtime_audit_contract: ok")
